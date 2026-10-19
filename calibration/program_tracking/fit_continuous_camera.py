"""Fit complete camera groups with movable C(t) and zoom-dependent optics.

The full-programme v12 remains the visual baseline. This solver uses the v13
shared scene and audited source observations; each output names that scene.
No per-frame fixed-centre ray factor or old-pose attraction enters this graph.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from scipy.optimize import least_squares

from backend import IncrementalCameraGraph
from continuous_camera_inputs import prepare_inputs, load_exposure_aliases
from continuous_camera_model import (ShotSpline, FrameBinding, ContinuousCameraModel,
                                     initialize_from_baseline, adapt_factor)
from continuous_camera_solver import solve, write_json
from continuous_exposure_clock import (CLOCK_POLICIES,
                                       build_exposure_clock, require_resume_clock_policy)
from projection import from_state, project_fast, attach_visible_surfaces
from robust_observations import pointwise_huber
from correlated_motion import wrap_correlated_pair
from reference_board_pair import BoardIntersectionError


TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(data, aliases, clock, interval):
    splines, frames = {}, []
    cursor = 1_000_000_000
    baseline = {p: from_state(data["states"][p], data["captured"]["templates"][p])
                for p in sorted(data["pts"])}
    group = data["camera_id"]
    for shot in data["shots"]:
        stamps = np.asarray(shot["pts"], np.int64)
        lo, hi = min(clock[int(p)] for p in stamps), max(clock[int(p)] for p in stamps)
        interior = np.arange(lo+interval*15360., hi, interval*15360.)
        knots = tuple([lo]*4 + interior.tolist() + [hi]*4)
        controls = tuple(range(cursor, cursor+len(knots)-4))
        cursor += len(controls)
        splines[int(shot["index"])] = ShotSpline(int(shot["index"]), controls, knots)
        for stamp in map(int, stamps):
            frames.append(FrameBinding(stamp, aliases[stamp], int(shot["index"]),
                                       group, baseline[stamp], float(clock[stamp])))
    base_key, slope_key = cursor, cursor+1
    zref = float(np.median([np.mean(c[6:8]) for c in baseline.values()]))
    model = ContinuousCameraModel(splines, frames, {group: base_key},
                                  {group: slope_key}, {group: zref})
    initial, report = initialize_from_baseline(model, baseline)
    return model, initial, baseline, report


def adapt_initializer(model, initial, baseline, scene, pairs):
    """Preserve previous views only as a starting point for the new model.

    Synthetic old-view projections below are NEVER used as fit observations.
    All subsequent optimization factors use actual source pixels.
    """
    first_pair = {}
    for a, b, pair in pairs:
        first_pair.setdefault(a, pair)
    chosen = {}
    for stamp, binding in model.frames.items():
        chosen.setdefault(binding.frame_key, stamp)
    samples = np.asarray([(x, y) for y in np.linspace(300, 1040, 6)
                          for x in np.linspace(60, 1860, 10)], float)
    errors = []
    for key, stamp in chosen.items():
        old = baseline[stamp]
        xyz, _, valid = attach_visible_surfaces(scene, old, samples)
        xyz = xyz[valid]
        if stamp in first_pair:
            hits = first_pair[stamp].source_hits(data_state(old))
            xyz = np.vstack((xyz, hits.xyz[hits.valid]))
        if len(xyz) < 4:
            raise ValueError(f"No multi-point old view for initializer at {stamp}")
        target = project_fast(old, xyz)[0]
        q = initial[key]

        def evaluate(v, jacobian=False):
            q[3:7] = v
            state, blocks = model.physical_frame(stamp, initial)
            result = project_fast(from_state(state, old), xyz, jacobian)
            if jacobian:
                return result[2] @ blocks[key][:, 3:7]
            return (result[0]-target).reshape(-1)

        fitted = least_squares(evaluate, q[3:7].copy(), jac=lambda v: evaluate(v, True),
                               max_nfev=30, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q[3:7] = fitted.x
        errors.append(float(np.sqrt(np.mean(evaluate(fitted.x)**2))))
    return dict(exposures=len(chosen), old_view_rms_px_p50=float(np.median(errors)),
                old_view_rms_px_max=float(max(errors)),
                role="initial_values_only_not_observations_or_priors")


def data_state(camera):
    return np.r_[camera[:3]/100., camera[3:8], camera[8:11]/1000.]


def raw_spec(spec):
    if spec["kind"] == "prior":
        target = np.asarray(spec["args"][1], float)
        return (lambda q: q[0]-target, lambda q: [np.eye(11)],
                dict(residual_dim=11, sigma=spec["args"][2]))
    if spec["kind"] == "between":
        delta = np.asarray(spec["args"][2], float)
        return (lambda q: q[1]-q[0]-delta, lambda q: [-np.eye(11), np.eye(11)],
                dict(residual_dim=11, sigma=spec["args"][3]))
    options = spec["kwargs"].copy()
    jac = options.pop("jac_fn", None)
    return spec["args"][1], jac, options


def is_old_zoom_prior(spec, pts):
    if (spec["kind"] != "observation" or len(spec["keys"]) != 1
            or spec["keys"][0] not in pts or spec["file"] != "joint_tracking_refine.py"
            or spec["kwargs"].get("residual_dim") != 1):
        return False
    fn = spec["kwargs"].get("jac_fn")
    if fn is None:
        return False
    j = np.asarray(fn([np.zeros(11)])[0])
    expected = np.zeros((1, 11)); expected[0, 6:8] = .5
    return j.shape == expected.shape and np.array_equal(j, expected)


def dummy_factor(graph, key, live_columns):
    cols = [i for i in range(11) if i not in live_columns]
    jac = np.eye(11)[cols]
    graph.add_observation([key], lambda q, j=jac: j @ q[0],
                          jac_fn=lambda q, j=jac: [j], residual_dim=len(cols), sigma=1.)


def acceleration_factor(graph, keys, times, columns, sigma):
    dt0, dt1 = np.diff(times)
    mid = .5*(dt0+dt1)
    coefficients = np.asarray([1/dt0, -1/dt0-1/dt1, 1/dt1])/mid
    selection = np.eye(11)[columns]
    matrices = [x*selection for x in coefficients]
    graph.add_observation(keys,
        lambda q, js=matrices: sum(j @ v for j, v in zip(js, q)),
        jac_fn=lambda q, js=matrices: js, residual_dim=len(columns),
        sigma=np.asarray(sigma)/np.sqrt(mid))


def regularize(graph, model, initial, data, clock, valid_time_pairs):
    """Explicit broad calibration and continuous-motion priors, not source truth."""
    for key in sorted(model.frame_keys):
        dummy_factor(graph, key, (3, 4, 5, 6))
    for shot_id, spline in model.shot_splines.items():
        for key in spline.control_keys:
            dummy_factor(graph, key, (0, 1, 2))
            selection = np.eye(11)[:3]
            target = initial[key][:3].copy()
            graph.add_observation([key], lambda q, t=target: q[0][:3]-t,
                                  jac_fn=lambda q, j=selection: [j], residual_dim=3,
                                  sigma=np.asarray([10., 10., 5.])/100.)
        times = np.asarray([np.mean(spline.knot_pts[i+1:i+4])/15360.
                            for i in range(len(spline.control_keys))])
        for i in range(1, len(times)-1):
            acceleration_factor(graph, spline.control_keys[i-1:i+2], times[i-1:i+2],
                                [0, 1, 2], np.asarray([3., 3., 3.])/100.)
        stamps = sorted({row.frame_key for row in model.frames.values()
                         if row.shot_id == shot_id})
        for i in range(1, len(stamps)-1):
            if ((stamps[i-1],stamps[i]) not in valid_time_pairs
                    or (stamps[i],stamps[i+1]) not in valid_time_pairs):
                continue
            acceleration_factor(graph, stamps[i-1:i+2],
                                np.asarray([clock[p] for p in stamps[i-1:i+2]])/15360.,
                                [3, 4, 5, 6], [.5, .5, .5, .5])
        # First frame is adjustable. A broad initialization prior supplies a
        # gauge in weakly observed directions; no source accuracy is asserted.
        first = stamps[0]
        target = initial[first][3:7].copy()
        selection = np.eye(11)[3:7]
        graph.add_observation([first], lambda q, t=target: q[0][3:7]-t,
                              jac_fn=lambda q, j=selection: [j], residual_dim=4,
                              sigma=[.3, .3, .3, .5])
    for key in model.lens_base_keys.values():
        dummy_factor(graph, key, (6, 8, 9, 10))
        cols = [6, 8, 9, 10]
        selection = np.eye(11)[cols]
        target = initial[key][cols].copy()
        graph.add_observation([key], lambda q, t=target: q[0][cols]-t,
                              jac_fn=lambda q, j=selection: [j], residual_dim=4,
                              sigma=[.2, .3, .3, .1])
    for key in model.lens_slope_keys.values():
        dummy_factor(graph, key, (6, 8, 9, 10))
        cols = [6, 8, 9, 10]
        selection = np.eye(11)[cols]
        graph.add_observation([key], lambda q: q[0][[6,8,9,10]],
                              jac_fn=lambda q, j=selection: [j], residual_dim=4,
                              sigma=[.1, .2, .2, .1])


def build_graph(data, pairs, model, initial, aliases, clock, valid_time_pairs):
    graph = IncrementalCameraGraph(11, window_size=len(initial)+2,
        relinearize_threshold=1e-5, refinement_updates=1, optimization="dogleg",
        dogleg_initial_delta=.001, diagnose_cost=True)
    active = set(initial)
    counts = Counter()

    def add(keys, raw, jac, options, family):
        adapted = adapt_factor(keys, raw, jac, options, active, data["states"], model)
        if adapted is None:
            counts["constant_"+family] += 1
            return
        keys, raw, jac, options = adapted
        def residual(q, fn=raw):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = family
                raise
        if jac is None:
            derivative = None
        else:
            def derivative(q, fn=jac):
                try:
                    return fn(q)
                except BoardIntersectionError as exc:
                    exc.evaluation_stage = "jacobian"
                    exc.factor_family = family
                    raise
        graph.add_observation(keys, residual, jac_fn=derivative, **options)
        counts[family] += 1

    for spec in data["retained"]:
        keys = set(spec["keys"])
        if not keys & data["pts"]:
            counts["external_constants"] += 1
            continue
        if spec["function"] in ("_mount_equality", "add_rig_equality", "add_frame_lens",
                                "_lens_factor", "_aspect_factor"):
            counts["removed_old_"+spec["function"]] += 1
            continue
        if spec["kind"] in ("prior", "between") and keys <= data["pts"]:
            counts["removed_old_pose_"+spec["kind"]] += 1
            continue
        if is_old_zoom_prior(spec, data["pts"]):
            counts["removed_old_zoom_prior"] += 1
            continue
        raw, jac, options = raw_spec(spec)
        add(spec["keys"], raw, jac, options, spec["file"]+":"+spec["function"])
    quality = data["captured"]["source_quality"]
    for a, b, pair in pairs:
        sigma = float(getattr(pair,"rinkspace_sigma_px",
                              data["config"].get("native_pair_sigma_px", 2.)))
        residual, jac = quality.wrap(pair._raw_residual, pair._raw_jacobians,
            quality.pair_scales(a, b, pair.xy0, pair.xy1,
                                getattr(pair,"rinkspace_quality_kind","native_motion")))
        residual, jac = pointwise_huber(residual, jac, group_size=2, delta=2*sigma)
        similarity = getattr(pair,"rinkspace_similarity_noise_px",
                             data["config"].get("native_pair_similarity_noise_px"))
        if similarity is not None:
            residual, jac, _ = wrap_correlated_pair(residual, jac, pair.xy1,
                sigma_point_px=sigma,
                sigma_modes_px=similarity)
            sigma = 1.
        add((a,b), residual, jac, dict(residual_dim=2*len(pair.xy0), sigma=sigma),
            "actual_uv_dynamic_board_motion")
    regularize(graph, model, initial, data, clock, valid_time_pairs)
    counts["total_with_model_priors"] = len(graph._pending)
    return graph, dict(counts)


def export(model, states, data, aliases, path):
    pts = np.asarray(sorted(model.frames), np.int64)
    cameras = np.asarray([model.camera25(int(p), states) for p in pts])
    if not np.isfinite(cameras).all():
        raise ValueError("Nonfinite camera output")
    by_pts = {int(p): c for p,c in zip(pts,cameras)}
    if any(not np.array_equal(by_pts[p], by_pts[r]) for p,r in aliases.items()):
        raise ValueError("Confirmed repeated exposure cameras differ")
    np.savez_compressed(path, pts=pts, camera25=cameras,
        shot_index=np.asarray([model.frames[int(p)].shot_id for p in pts], np.int16),
        exposure_representative_pts=np.asarray([aliases[int(p)] for p in pts],np.int64),
        source_sha256=data["diag"]["source_sha256"], model_sha256=data["diag"]["model_sha256"])
    return {"frames":len(pts), "center_span_scene":np.ptp(cameras[:,:3],axis=0).tolist(),
            "principal_skew_span_px":np.ptp(cameras[:,8:11],axis=0).tolist(),
            "sha256":sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="low_end")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--knot-seconds", type=float, default=1.)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exposure-clock-policy", choices=CLOCK_POLICIES,
                        default=None, required=True,
                        help="Explicit source clock policy; a changed policy requires a fresh graph")
    parser.add_argument("--warm-physical-cameras",type=Path,
                        help="Saved prior camera estimates only initialize this changed model")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume:
        require_resume_clock_policy(output, args.exposure_clock_policy)
    if not output.is_relative_to(TMP.resolve()) or args.knot_seconds <= 0:
        raise ValueError("Output must be in the existing programme temp root")
    if (output/"summary.json").exists() and not args.resume:
        raise ValueError("Completed result exists; inspect rather than rerun")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data,pairs,metadata = prepare_inputs(output,args.camera)
    aliases,alias_metadata = load_exposure_aliases(data)
    clock,clock_metadata,valid_time_pairs = build_exposure_clock(
        data, aliases, policy=args.exposure_clock_policy)
    model,initial,baseline,initial_report = make_model(data,aliases,clock,args.knot_seconds)
    adapted = adapt_initializer(model,initial,baseline,data["scene"],pairs)
    graph,counts = build_graph(data,pairs,model,initial,aliases,clock,valid_time_pairs)
    warm = None
    if args.warm_physical_cameras is not None:
        with np.load(args.warm_physical_cameras,allow_pickle=False) as z:
            if (str(z["source_sha256"])!=data["diag"]["source_sha256"]
                    or str(z["model_sha256"])!=data["diag"]["model_sha256"]
                    or not np.array_equal(z["pts"],np.asarray(sorted(data["pts"])))):
                raise ValueError("Warm camera source/model/complete-group PTS differ")
            previous={int(p):c.copy() for p,c in zip(z["pts"],z["camera25"])}
        initial,warm_report=initialize_from_baseline(model,previous)
        warm_adapted=adapt_initializer(model,initial,previous,data["scene"],pairs)
        warm=dict(path=str(args.warm_physical_cameras.resolve()),
                  sha256=sha(args.warm_physical_cameras),initializer=warm_report,
                  adaptation=warm_adapted,policy="initial_only; broad prior targets remain v13")
    phase = dict(schema="rinkspace.continuous-movable-centre-camera.v1",
        camera_id=args.camera, source_sha256=data["diag"]["source_sha256"],
        model_sha256=data["diag"]["model_sha256"], source_inputs=metadata,
        aliases=alias_metadata, exposure_clock=clock_metadata,
        exposure_clock_policy=args.exposure_clock_policy,
        frame_clock_virtual_pts={str(p):clock[p] for p in sorted(clock)},
        valid_temporal_acceleration_pairs=sorted([list(p) for p in valid_time_pairs]),
        knot_seconds=args.knot_seconds,
        shot_knots={str(k):dict(keys=list(s.control_keys),pts=list(s.knot_pts))
                    for k,s in model.shot_splines.items()},
        lens_base_keys=model.lens_base_keys,lens_slope_keys=model.lens_slope_keys,
        zoom_reference=model.lens_zoom_reference, initializer=initial_report,
        initializer_adaptation=adapted, warm_start=warm, factor_counts=counts,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in
                     [Path(__file__).name,"continuous_camera_model.py",
                      "continuous_camera_inputs.py","continuous_camera_solver.py",
                      "continuous_exposure_clock.py","continuous_camera_auxiliary.py"]},
        old_pose_priors_removed=True, fixed_c_motion_removed=True,
        first_camera_fixed=False, brown_fixed_unvalidated=True,
        source_pixels_changed=False, fitted_scene_geometry=False,
        final_holdouts_fitted=False, full_programme_accepted=False,
        regularization="C spline: 10/10/5 scene-unit broad controls, 3 scene-unit/s2 acceleration; R/zoom .5/s2 acceleration; first R .3rad / logzoom .5 broad; lens priors explicit in code. These are modeling priors, not measurements.")
    phase_path = output/"phase_00.json"
    if args.resume:
        old = json.loads(phase_path.read_text(encoding="utf8"))
        if old != phase:
            raise ValueError("Resume model/code/input differs from saved phase")
    else:
        write_json(phase_path,phase)
        snapshot=output/"code_at_start"
        snapshot.mkdir(exist_ok=True)
        for name in phase["code_sha256"]:
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print(json.dumps(dict(stage="model_ready",camera=args.camera,frames=len(model.frames),
                           exposures=len(model.frame_keys),states=len(initial),
                           factors=len(graph._pending),preparation_seconds=time.perf_counter()-started,
                           initializer=adapted)),flush=True)
    export(model,initial,data,aliases,output/"initial_cameras.npz")
    solved,trace=solve(graph,initial,output,max_iterations=args.iterations,resume=args.resume)
    exported=export(model,solved,data,aliases,output/"shot_cameras.npz")
    summary=dict(phase_sha256=sha(phase_path),camera_id=args.camera,
        shot_indices=list(data["shot_indices"]),**exported,
        source_sha256=data["diag"]["source_sha256"],model_sha256=data["diag"]["model_sha256"],
        seconds=time.perf_counter()-started,solver=trace,accepted=False,
        scope="Complete camera-group candidate; must compare real source video to v12",
        limitations="Brown fixed; scene conditioned on v13; model priors not physical measurements; unseen verification still required")
    write_json(output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="solver"}),flush=True)


if __name__=="__main__":
    main()
