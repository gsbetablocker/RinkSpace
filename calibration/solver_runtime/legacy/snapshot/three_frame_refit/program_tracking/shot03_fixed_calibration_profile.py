"""Condition original 119-frame shot03 graph on fitted 19D C/lens shape.

The original 633 factors, source UV, quality, Huber rules, and prior targets
are preserved. Only each frame's Rodrigues(3) and mean(logfx,logfy) vary.
Default invocation writes phase_00 and checks equality; ``--run`` requires a
matching saved preflight and performs one constrained solve. No final holdout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import json
import os
from pathlib import Path
import time

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
import conditioned_shot03_lens19_seed as seed_driver
from backend import IncrementalCameraGraph
from rig_parameterization import camera25 as derived_camera25


TMP = seed_driver.TMP
OUTPUT = TMP / "shot03_lens19_calibration_profile_v1"
SEED_PHASE = seed_driver.OUTPUT / "phase_00.json"
SEED_STATES = seed_driver.OUTPUT / "initial_states.npz"
SEED_PHASE_SHA256 = "4fc359521dd72a82cd4f9267005451489c1f9863d76c68c916f6d215269137b4"
SEED_STATES_SHA256 = "2e331d27c8b96b609b03c76cd8eec75cafbc8475d027c6bae51892ffbf7b8187"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class RecordingGraph(IncrementalCameraGraph):
    """Original 11D graph plus exact API-call ledger, without changing factors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = []

    def add_prior(self, frame_id, state, sigma):
        super().add_prior(frame_id, state, sigma)
        self.ledger.append(dict(kind="prior", keys=(int(frame_id),),
                                target=np.asarray(state, float).copy(),
                                sigma=np.asarray(sigma, float).copy()))

    def add_between(self, previous_frame_id, frame_id, delta, sigma):
        super().add_between(previous_frame_id, frame_id, delta, sigma)
        self.ledger.append(dict(kind="between", keys=(int(previous_frame_id), int(frame_id)),
                                delta=np.asarray(delta, float).copy(),
                                sigma=np.asarray(sigma, float).copy()))

    def add_observation(self, keys, residual_fn, *, residual_dim, sigma,
                        jac_fn=None, huber_k=None, fd_step=1e-5):
        caller = inspect.currentframe().f_back
        super().add_observation(keys, residual_fn, residual_dim=residual_dim,
                                sigma=sigma, jac_fn=jac_fn, huber_k=huber_k,
                                fd_step=fd_step)
        self.ledger.append(dict(kind="observation", keys=tuple(map(int, keys)),
                                residual_fn=residual_fn, jac_fn=jac_fn,
                                residual_dim=int(residual_dim),
                                sigma=np.asarray(sigma, float).copy(),
                                huber_k=huber_k, fd_step=fd_step,
                                source_file=Path(caller.f_code.co_filename).name,
                                source_function=caller.f_code.co_name))


def original_graph(data: dict, pairs: list):
    original = driver.IncrementalCameraGraph
    driver.IncrementalCameraGraph = RecordingGraph
    try:
        graph, _ = driver.build_graph(data, pairs, "fixed", "derived")
    finally:
        driver.IncrementalCameraGraph = original
    require(len(graph._pending) == len(graph.ledger) == 633,
            "Original 633-factor graph or recorder order changed")
    return graph


class FrameProfile:
    """Affine map q4=(R3,zoom) to exact-derived original frame Vector11."""

    def __init__(self, data: dict, initial11: dict, anchors: np.ndarray):
        self.frames = tuple(sorted(map(int, data["pts"])))
        self.frame_set = set(self.frames)
        require(len(self.frames) == 119 and set(initial11) == set(data["active"]),
                "Profile initial state keys differ from original shot graph")
        require(anchors.shape == (3, 25) and np.isfinite(anchors).all(),
                "Invalid fitted 19D anchors")
        ratio = anchors[:, 6]-anchors[:, 7]
        common = anchors[:, 8:11]
        centre = anchors[:, :3]
        require(np.max(np.abs(ratio-ratio[0])) < 1e-10 and
                np.max(np.abs(common-common[0])) < 1e-8 and
                np.max(np.abs(centre-centre[0])) < 1e-8,
                "19D anchors do not share one centre and lens shape")
        self.ratio = float(ratio[0])
        self.common = common[0].copy()
        self.C = centre[0].copy()
        self.mount = int(data["mount"])
        self.lens = int(data["lens"])
        self.constants = {self.mount: initial11[self.mount].copy(),
                          self.lens: initial11[self.lens].copy()}
        require(np.max(np.abs(self.constants[self.mount][:3]*100.-self.C)) < 1e-8 and
                abs(self.constants[self.lens][6]-self.ratio) < 1e-10 and
                np.max(np.abs(self.constants[self.lens][8:11]*1000.-self.common)) < 1e-8,
                "Saved lens19 initializer does not have the fitted constants")
        self.T = np.zeros((11, 4))
        self.T[3:6, :3] = np.eye(3)
        self.T[6:8, 3] = 1.
        self.offset = np.zeros(11)
        self.offset[6:8] = (self.ratio/2., -self.ratio/2.)
        self.offset[8:11] = self.common/1000.
        self.initial4 = {p: np.r_[initial11[p][3:6],
                                  np.mean(initial11[p][6:8])]
                         for p in self.frames}
        for p in self.frames:
            require(np.max(np.abs(self.frame(self.initial4[p])-initial11[p])) < 1e-10,
                    f"Original lens19 seed is not in the profile subspace: {p}")

    def frame(self, q: np.ndarray) -> np.ndarray:
        value = np.asarray(q, float)
        require(value.shape == (4,) and np.isfinite(value).all(),
                "Profile frame must be four finite values")
        return self.offset+self.T@value

    def expand(self, keys: tuple[int, ...], reduced_states: list[np.ndarray]) -> list[np.ndarray]:
        selected = iter(reduced_states)
        return [self.frame(next(selected)) if key in self.frame_set
                else self.constants[key] for key in keys]

    def full_states(self, frames4: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        return {**self.constants,
                **{p: self.frame(frames4[p]) for p in self.frames}}


def observation(spec: dict):
    if spec["kind"] == "prior":
        target = spec["target"]
        return (lambda q: q[0]-target, lambda q: [np.eye(11)],
                dict(residual_dim=11, sigma=spec["sigma"]))
    if spec["kind"] == "between":
        delta = spec["delta"]
        return (lambda q: q[1]-q[0]-delta,
                lambda q: [-np.eye(11), np.eye(11)],
                dict(residual_dim=11, sigma=spec["sigma"]))
    return (spec["residual_fn"], spec["jac_fn"],
            dict(residual_dim=spec["residual_dim"], sigma=spec["sigma"],
                 huber_k=spec["huber_k"], fd_step=spec["fd_step"]))


def pure_constant(index: int, spec: dict, data: dict, profile: FrameProfile) -> bool:
    if not set(spec["keys"]) & profile.frame_set:
        return True
    retained = len(data["retained"])
    if retained <= index < retained+len(profile.frames):
        return True  # Original exact-derived dummy-centre factor.
    if index < retained:
        source = data["retained"][index]
        if (source["file"], source["function"]) == ("shared_lens.py", "add_frame_lens"):
            return True  # Frame ratio/pp/skew and lens are all fixed.
    return False


def reduced_graph(data: dict, pairs: list, original: RecordingGraph,
                  profile: FrameProfile):
    config = data["config"]
    graph = IncrementalCameraGraph(4, window_size=len(profile.frames)+2,
        relinearize_threshold=1e-5, refinement_updates=config["refinement_updates"],
        optimization=config["optimization"], diagnose_cost=True)
    variable_map = {}
    constants = []
    checks = []
    for index, spec in enumerate(original.ledger):
        if pure_constant(index, spec, data, profile):
            constants.append(index)
            continue
        active = tuple(k for k in spec["keys"] if k in profile.frame_set)
        require(active, f"Nonconstant original factor has no variable frame: {index}")
        raw, raw_jac, options = observation(spec)
        def residual(q, keys=spec["keys"], fn=raw):
            return fn(profile.expand(keys, q))
        def jacobian(q, keys=spec["keys"], fn=raw_jac):
            matrices = fn(profile.expand(keys, q))
            return [np.asarray(matrices[i], float) @ profile.T
                    for i, key in enumerate(keys) if key in profile.frame_set]
        graph.add_observation(active, residual,
                              jac_fn=jacobian if raw_jac is not None else None,
                              **options)
        variable_map[index] = len(graph._pending)-1
        checks.append(dict(index=index, keys=list(active),
                           source_file=spec.get("source_file"),
                           source_function=spec.get("source_function")))
    require(len(constants)+len(variable_map) == 633 and
            len(graph._pending) == len(variable_map),
            "Profile factor partition lost an original factor")
    return graph, variable_map, constants, checks


def values(graph, states: dict[int, np.ndarray]) -> gtsam.Values:
    result = gtsam.Values()
    for key, q in states.items():
        result.insert(graph._key(key), q)
    return result


def verify_factors(original, reduced, variable_map, constants, profile, data):
    full = values(original, profile.full_states(profile.initial4))
    small = values(reduced, profile.initial4)
    shifted4 = {p:q+np.asarray([1e-4,-2e-4,3e-4,4e-5])
                for p,q in profile.initial4.items()}
    shifted_full = values(original, profile.full_states(shifted4))
    differences = []
    constant_differences = []
    constant_costs = defaultdict(lambda: [0, 0.])
    for index, factor in enumerate(original._pending):
        old = float(factor.error(full))
        if index in variable_map:
            new = float(reduced._pending[variable_map[index]].error(small))
        else:
            new = old
            constant_differences.append(abs(old-float(factor.error(shifted_full))))
            label = ("dummy_center" if len(data["retained"]) <= index <
                     len(data["retained"])+len(profile.frames) else
                     "frame_lens_fixed_shape" if index < len(data["retained"]) and
                     (data["retained"][index]["file"],
                      data["retained"][index]["function"]) ==
                     ("shared_lens.py", "add_frame_lens") else
                     "mount_lens_or_other_fixed")
            constant_costs[label][0] += 1
            constant_costs[label][1] += old
        differences.append(abs(old-new))
    max_difference = max(differences)
    require(max_difference < 1e-8,
            f"Conditioned factor error differs from original: {max_difference}")
    max_constant_difference = max(constant_differences)
    require(max_constant_difference < 1e-8,
            f"Supposed fixed factor varies under profile perturbation: {max_constant_difference}")
    constant_total = float(sum(item[1] for item in constant_costs.values()))
    variable_total = float(sum(f.error(small) for f in reduced._pending))
    full_total = float(sum(f.error(full) for f in original._pending))
    require(abs(full_total-variable_total-constant_total) < 1e-7,
            "Constant factors were not preserved in total objective")
    return dict(original_factors=633, reduced_variable_factors=len(variable_map),
        omitted_from_solver_but_accounted_constant_factors=len(constants),
        max_per_factor_cost_difference=max_difference,
        max_constant_factor_cost_perturbation=max_constant_difference,
        original_total=full_total, reduced_variable_total=variable_total,
        pure_constant_total=constant_total,
        pure_constant_by_type={k:dict(factors=int(v[0]),cost=float(v[1]))
                               for k,v in constant_costs.items()},
        comparable_profile_total=variable_total+constant_total)


def verify_jacobians(original, reduced, variable_map, profile, data, pairs):
    retained = len(data["retained"])
    candidates = {
        "edge": [i for i,s in enumerate(data["retained"])
                 if (s["file"],s["function"]) ==
                 ("track_sequence.py", "_add_source_edge_factors")
                 and int(s["keys"][0]) in seed_driver.ANCHORS],
        "ad": [i for i,s in enumerate(data["retained"])
               if (s["file"],s["function"]) == ("ad_landmarks.py", "factor")
               and int(s["keys"][0]) in seed_driver.ANCHORS],
        "native": list(range(retained+len(profile.frames),633)),
    }
    result = {}
    full_values = values(original, profile.full_states(profile.initial4))
    small_values = values(reduced, profile.initial4)
    shot_config = next(s for s in data["config"]["shots"] if s["shot_index"] == 3)
    with np.load(shot_config["source_edges"], allow_pickle=False) as source_edges:
        edge_pts = source_edges["pts"].copy()
        edge_offsets = source_edges["offsets"].copy()
        edge_uv = source_edges["observed_uv"].copy()
        edge_kind = source_edges["edge_id"].copy()
    require(np.array_equal(edge_pts, np.asarray(profile.frames)),
            "Frozen source-edge packet PTS differ from shot03 profile")
    for category, indices in candidates.items():
        checked = []
        for index in indices:
            if index not in variable_map:
                continue
            spec = original.ledger[index]
            if spec["kind"] != "observation" or spec["jac_fn"] is None:
                continue
            keys = tuple(k for k in spec["keys"] if k in profile.frame_set)
            q = [profile.initial4[k].copy() for k in keys]
            raw, raw_jac, _ = observation(spec)
            full = profile.expand(spec["keys"], q)
            analytic = [np.asarray(matrix)@profile.T for key,matrix in
                        zip(spec["keys"],raw_jac(full)) if key in profile.frame_set]
            original_linear = original._pending[index].linearize(full_values)
            reduced_linear = reduced._pending[variable_map[index]].linearize(small_values)
            original_A = original_linear.getA()
            mapped_A = np.hstack([
                original_A[:, list(original_linear.keys()).index(original._key(k))*11:
                           (list(original_linear.keys()).index(original._key(k))+1)*11]
                @ profile.T for k in keys])
            linear_A_error = float(np.max(np.abs(mapped_A-reduced_linear.getA())))
            linear_b_error = float(np.max(np.abs(original_linear.getb()-reduced_linear.getb())))
            require(linear_A_error < 1e-8 and linear_b_error < 1e-8,
                    f"Conditioned linearized factor differs: {category}/{index}"
                    f" A={linear_A_error} b={linear_b_error}")
            fd_checks = []
            for h in (1e-6, 1e-7, 1e-8):
                max_err = 0.
                max_relative = 0.
                worst = None
                for j, state in enumerate(q):
                    for column in range(4):
                        hi = [x.copy() for x in q]; lo = [x.copy() for x in q]
                        hi[j][column] += h; lo[j][column] -= h
                        top = np.asarray(raw(profile.expand(spec["keys"], hi)),float)
                        bottom = np.asarray(raw(profile.expand(spec["keys"], lo)),float)
                        difference = np.abs((top-bottom)/(2*h)-analytic[j][:,column])
                        row = int(np.argmax(difference))
                        if float(difference[row]) > max_err:
                            max_err = float(difference[row])
                            max_relative = float(difference[row]/
                                                 (1.+abs(analytic[j][row,column])))
                            worst = dict(state_key=int(keys[j]), residual_row=row,
                                         parameter_column=column,
                                         analytic_value=float(analytic[j][row,column]))
                fd_checks.append(dict(step=h,max_absolute_error=max_err,
                                      worst_relative_error=max_relative,
                                      worst=worst))
            source_pixel = None
            if category == "edge":
                stamp = int(keys[0])
                frame_index = int(np.searchsorted(edge_pts, stamp))
                edge_indices = [i for i in candidates["edge"]
                                if int(data["retained"][i]["keys"][0]) == stamp]
                edge_id = edge_indices.index(index)
                start, stop = map(int, edge_offsets[frame_index:frame_index+2])
                observed = edge_uv[start:stop][edge_kind[start:stop] == edge_id]
                point_index = fd_checks[0]["worst"]["residual_row"]//2
                require(edge_id in (0,1) and spec["residual_dim"] == 2*len(observed)
                        and point_index < len(observed),
                        "Source edge finite-vector row cannot map to original UV")
                source_pixel = dict(edge_name=("board_top","ice_board_contact")[edge_id],
                                    point_index=point_index,
                                    observed_uv=observed[point_index].tolist())
            checked.append(dict(original_factor_index=index,
                pts=list(keys), linearized_A_chain_error=linear_A_error,
                linearized_b_error=linear_b_error,
                finite_difference_convergence=fd_checks,
                source_edge_worst_pixel=source_pixel))
            if len(checked) >= 2:
                break
        require(len(checked) == 2 and all(row["finite_difference_convergence"][1]
                                           ["max_absolute_error"] < 1e-3
                                           for row in checked),
                f"Chain-rule Jacobian check failed: {category}/{checked}")
        result[category] = checked
    return result


def construct(output: Path):
    require(output.resolve().is_relative_to(TMP.resolve()),
            "Profile output must stay in tracking TMP")
    packet = seed_driver.packets()
    require(driver.digest(SEED_PHASE) == SEED_PHASE_SHA256 and
            driver.digest(SEED_STATES) == SEED_STATES_SHA256,
            "Reviewed 19D-seeded 119-frame initializer changed")
    with np.load(SEED_STATES, allow_pickle=False) as file:
        initial11 = {int(k):v.copy() for k,v in zip(file["keys"],file["states"])}
        require(str(file["source_sha256"]) == packet["source_sha256"] and
                str(file["model_sha256"]) == packet["scene_sha256"],
                "Saved initializer source/scene differs")
    data = driver.prepare(seed_driver.CONFIG, seed_driver.CANDIDATE,
                          seed_driver.MANIFEST, (3,), output,
                          complete_camera_group=False)
    pairs, counts, _, _, validity = driver.native_pairs(data)
    require(counts["adjacent_pairs"] == 118 and counts["pixels"] == 8132 and
            validity["invalid_pair_count"] == 0 and
            data["diag"]["model_sha256"] == packet["scene_sha256"] and
            data["captured"]["source_hash"] == packet["source_sha256"],
            "Original board/source/scene graph changed")
    profile = FrameProfile(data, initial11, packet["new_camera"])
    original = original_graph(data, pairs)
    reduced, variable_map, constants, ledger = reduced_graph(data, pairs, original, profile)
    equality = verify_factors(original, reduced, variable_map, constants, profile, data)
    jacobians = verify_jacobians(original, reduced, variable_map, profile, data, pairs)
    return packet, data, profile, original, reduced, equality, jacobians


def final_factor_audit(original: RecordingGraph, reduced_factors: list,
                       profile: FrameProfile, data: dict,
                       final4: dict[int, np.ndarray], constant_total: float) -> dict:
    """Score the same original 633 factors at the conditioned final state."""
    full = values(original, profile.full_states(final4))
    small = gtsam.Values()
    for p, q in final4.items():
        small.insert(original._key(p), q)
    labels = [("retained", spec["file"], spec["function"], spec["loop"] or "")
              for spec in data["retained"]]
    labels += [("dummy_center_prior", "", "", "")]*len(profile.frames)
    labels += [("native_board_source", "", "", "")]*118
    require(len(labels) == len(original._pending) == 633,
            "Final original graph family ledger changed")
    families = defaultdict(lambda: [0, 0.])
    for label, factor in zip(labels, original._pending):
        families[label][0] += 1
        families[label][1] += float(factor.error(full))
    original_total = float(sum(v[1] for v in families.values()))
    reduced_total = float(sum(f.error(small) for f in reduced_factors))
    require(abs(original_total-reduced_total-constant_total) < 1e-6,
            "Final original 633 cost differs from conditioned variable + constant cost")
    return dict(original_633_total=original_total,
                variable_factor_total=reduced_total,
                exact_constant_factor_total=constant_total,
                max_total_difference=abs(original_total-reduced_total-constant_total),
                by_captured_family=[dict(group=list(k), factors=int(v[0]),
                                         final_cost=float(v[1]))
                                    for k,v in sorted(families.items())])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true",
                        help="Requires reviewed phase_00, then one constrained 4D solve")
    args = parser.parse_args()
    output = args.output
    if args.run:
        require((output/"phase_00.json").is_file(),
                "Run default preflight first; review its phase_00")
    else:
        if output.exists():
            require(not (output/"phase_00.json").exists() and
                    {p.name for p in output.iterdir()} <= {"graph_capture"},
                    "Profile preflight output already exists")
        else:
            output.mkdir(parents=True)
    packet, data, profile, original, reduced, equality, jacobians = construct(output)
    initial_path = output/"initial_4d_states.npz"
    stamps = np.asarray(profile.frames, np.int64)
    q4 = np.asarray([profile.initial4[int(p)] for p in stamps])
    if args.run:
        with np.load(initial_path, allow_pickle=False) as file:
            require(np.array_equal(file["pts"], stamps) and
                    np.array_equal(file["q4"], q4),
                    "Saved 4D preflight initializer changed")
    else:
        np.savez_compressed(initial_path, pts=stamps, q4=q4,
            source_sha256=np.asarray(packet["source_sha256"]),
            model_sha256=np.asarray(packet["scene_sha256"]))
    phase = dict(schema="rinkspace.shot03-fixed-calibration-profile.v1",
        scope="One 119-frame fixed-C/lens-shape conditional profile, only R3+zoom per frame free; different restricted feasible set from original 11D graph",
        source_sha256=packet["source_sha256"],
        scene_sha256=packet["scene_sha256"],
        original_full_graph_factors=633, original_native_pairs=118,
        original_native_source_pixels=8132,
        original_source_uv_quality_huber_and_priors_unchanged=True,
        original_external_scene_reference_nodes_fixed_at_v13=True,
        fixed_C_scene=profile.C.tolist(), fixed_log_fx_over_fy=profile.ratio,
        fixed_cx_cy_skew_px=profile.common.tolist(),
        seed_phase_sha256=SEED_PHASE_SHA256, seed_states_sha256=SEED_STATES_SHA256,
        pinned_packet_sha256={str(p):h for p,h in seed_driver.PINNED_SHA.items()},
        script_sha256=driver.digest(__file__),
        driver_code_sha256=driver.digest(driver.__file__),
        initial_4d_states_sha256=driver.digest(initial_path),
        factor_error_equivalence=equality,
        chain_rule_jacobian_checks=jacobians,
        solver_started=False)
    phase_path = output/"phase_00.json"
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Reviewed profile preflight changed; do not run")
    else:
        phase_path.write_text(json.dumps(phase,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(dict(phase="preflight", path=str(phase_path),
        sha256=driver.digest(phase_path),
        original_cost=equality["original_total"],
        reduced_cost=equality["reduced_variable_total"],
        fixed_constant_cost=equality["pure_constant_total"],
        max_factor_difference=equality["max_per_factor_cost_difference"],
        jacobian_checks=jacobians)), flush=True)
    if not args.run:
        return
    target = output/"profile"
    require(not (target/"summary.json").exists() and
            not (target/"post_isam_4d_states.npz").exists(),
            "Existing profile solve stage needs review")
    target.mkdir()
    start = time.perf_counter()
    reduced_factors = list(reduced._pending)
    solved = reduced.update(profile.initial4)
    post_path = target/"post_isam_4d_states.npz"
    np.savez_compressed(post_path, pts=stamps,
        q4=np.asarray([solved[int(p)] for p in stamps]))
    (target/"post_isam_summary.json").write_text(json.dumps(dict(
        stage="post_isam_pre_batch_lm", candidate_result=False,
        initial_phase_sha256=driver.digest(phase_path),
        states_sha256=driver.digest(post_path),
        graph_update=reduced.last_update_summary,
        seconds=time.perf_counter()-start),indent=2),encoding="utf8")
    solved = reduced.batch_polish(**data["config"]["batch_polish"])
    cameras = np.asarray([derived_camera25(profile.frame(solved[int(p)]),
        profile.constants[profile.mount],
        data["captured"]["templates"][int(p)], "fixed") for p in stamps])
    camera_path = target/"shot_cameras.npz"
    np.savez_compressed(camera_path, pts=stamps, shot_index=np.full(len(stamps),3,np.int8),
        camera25=cameras,
        source_sha256=np.asarray(packet["source_sha256"]),
        model_sha256=np.asarray(packet["scene_sha256"]))
    state_path = target/"states4.npz"
    np.savez_compressed(state_path, pts=stamps,
        q4=np.asarray([solved[int(p)] for p in stamps]))
    expanded = profile.full_states(solved)
    expanded_keys = np.asarray(sorted(expanded), np.int64)
    expanded_path = target/"expanded_original_states11.npz"
    np.savez_compressed(expanded_path, keys=expanded_keys,
        states=np.asarray([expanded[int(k)] for k in expanded_keys]),
        source_sha256=np.asarray(packet["source_sha256"]),
        model_sha256=np.asarray(packet["scene_sha256"]),
        semantics=np.asarray("4D_restricted_profile_expanded_into_original_121_nodes"))
    audit = final_factor_audit(original, reduced_factors, profile, data, solved,
                               equality["pure_constant_total"])
    batch_cost = reduced.last_batch_diagnostics.get("final_nonlinear_cost")
    if batch_cost is not None:
        require(abs(float(batch_cost)-audit["variable_factor_total"]) < 1e-5,
                "Reduced graph batch cost differs from saved factor sum")
    summary = dict(schema="rinkspace.shot03-fixed-calibration-profile-result.v1",
        scope=phase["scope"], frames=len(stamps), source_sha256=packet["source_sha256"],
        scene_sha256=packet["scene_sha256"], phase_00_sha256=driver.digest(phase_path),
        fixed_C_scene=profile.C.tolist(), fixed_log_fx_over_fy=profile.ratio,
        fixed_cx_cy_skew_px=profile.common.tolist(),
        original_factor_count=633, variable_factor_count=len(reduced_factors),
        pure_constant_cost=equality["pure_constant_total"],
        graph_update=reduced.last_update_summary,
        batch=reduced.last_batch_diagnostics,
        callback_metadata=reduced.callback_metadata,
        process_id=os.getpid(),
        final_original_factor_audit=audit,
        comparable_final_cost=audit["original_633_total"],
        post_isam_sha256=driver.digest(post_path),
        camera_sha256=driver.digest(camera_path), states4_sha256=driver.digest(state_path),
        expanded_original_states11_sha256=driver.digest(expanded_path),
        seconds=time.perf_counter()-start,
        not_programme_acceptance=True)
    summary_path = target/"summary.json"
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(dict(phase="solve_complete", summary=str(summary_path),
                          sha256=driver.digest(summary_path))), flush=True)


if __name__ == "__main__":
    main()
