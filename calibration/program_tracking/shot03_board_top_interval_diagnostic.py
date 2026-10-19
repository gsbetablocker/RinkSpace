"""Opt-in shot03 blue-top source interval diagnostic on the original 119-frame graph.

Default invocation is a read-only graph preflight.  It never solves.  A frozen
source-only support report and its caller-supplied SHA are required for the
interval objective; without them, the replacement callback must reproduce all
119 original top-edge factors.  ``--run`` is additionally gated by a matching
saved preflight.  No final holdout is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
import conditioned_shot03_lens19_seed as seed_driver
from backend import IncrementalCameraGraph
from projection import from_state
from rig_parameterization import physical_frame
from robust_observations import pointwise_huber
from source_edge_intervals import SourceNormalIntervals
from track_sequence import (SourceEdges, _closed_edge_ring,
                            _edge_curve_error, _edge_segment_candidates)


TMP = seed_driver.TMP
PROFILE_SUMMARY = TMP / "shot03_lens19_calibration_profile_v1/profile/summary.json"
PROFILE_STATES = TMP / "shot03_lens19_calibration_profile_v1/profile/expanded_original_states11.npz"
EXPECTED_PROFILE_SUMMARY_SHA256 = "d48faaa19197d29af2b21a27ec38e366ec3cec4aa62c1de5ed413421b7a32c73"
DEFAULT_OUTPUT = TMP / "shot03_board_top_interval_identity_preflight_v1"
TOP_KIND = ("track_sequence.py", "_add_source_edge_factors")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frozen_profile_states(data: dict) -> tuple[dict[int, np.ndarray], str]:
    summary_sha = driver.digest(PROFILE_SUMMARY)
    if EXPECTED_PROFILE_SUMMARY_SHA256:
        require(summary_sha == EXPECTED_PROFILE_SUMMARY_SHA256,
                "Reviewed full119 profile summary changed")
    summary = json.loads(PROFILE_SUMMARY.read_text(encoding="utf8"))
    require(driver.digest(PROFILE_STATES) == summary["expanded_original_states11_sha256"]
            and summary["source_sha256"] == data["captured"]["source_hash"]
            and summary["scene_sha256"] == data["diag"]["model_sha256"],
            "Profile 121-node initializer differs from current source/scene")
    with np.load(PROFILE_STATES, allow_pickle=False) as packet:
        require(str(packet["source_sha256"]) == summary["source_sha256"]
                and str(packet["model_sha256"]) == summary["scene_sha256"],
                "Profile initializer source/scene tags changed")
        keys = np.asarray(packet["keys"], np.int64)
        values = np.asarray(packet["states"], np.float64)
    require(keys.shape == (121,) and values.shape == (121, 11)
            and len(set(map(int, keys))) == 121 and set(map(int, keys)) == set(data["active"])
            and np.isfinite(values).all(), "Invalid 119+mount+lens profile states")
    return {int(k): v.copy() for k, v in zip(keys, values)}, summary_sha


def source_top(data: dict):
    shot = next(row for row in data["config"]["shots"] if int(row["shot_index"]) == 3)
    edge_path = Path(shot["source_edges"])
    pts = np.asarray(data["shots"][0]["pts"], np.int64)
    edges = SourceEdges.load(edge_path, pts,
                             source_sha256=data["captured"]["source_hash"],
                             model_sha256=data["diag"]["base_model_sha256"])
    require(len(pts) == 119 and len(data["pts"]) == 119,
            "Interval diagnostic must contain exactly shot03")
    offsets = np.r_[0, np.cumsum([len(edges.frame(i, 0)) for i in range(len(pts))])]
    uv = np.concatenate([edges.frame(i, 0) for i in range(len(pts))])
    foot_counts = np.asarray([len(edges.frame(i, 1)) for i in range(len(pts))], np.int64)
    require(len(uv) == 3995, "Original shot03 top edge no longer has 3995 UVs")
    return edge_path, pts, offsets, uv, foot_counts


def load_support(report_path: Path | None, expected_sha: str | None,
                 edge_path: Path, pts: np.ndarray, offsets: np.ndarray,
                 uv: np.ndarray, data: dict):
    if report_path is None:
        require(expected_sha is None, "Support SHA requires a support report")
        return (np.zeros(len(uv), bool), np.full((len(uv), 2), np.nan),
                np.full((len(uv), 2), np.nan), None)
    require(expected_sha is not None and len(expected_sha) == 64
            and driver.digest(report_path) == expected_sha,
            "Caller-supplied frozen source-support report SHA is missing or differs")
    report = json.loads(report_path.read_text(encoding="utf8"))
    require(report["schema"] == "rinkspace.shot03-blue-top-source-support.v1"
            and report["counts"]["frames"] == 119
            and report["counts"]["original_uv"] == 3995
            and report["source_video"]["sha256"] == data["captured"]["source_hash"]
            and report["source_edge_cache"]["sha256"] == driver.digest(edge_path)
            and Path(report["source_edge_cache"]["path"]).resolve() == edge_path.resolve()
            and report["v13_config_sha256"] == driver.digest(seed_driver.CONFIG)
            and report["original_sigma_px"] == 2.
            and report["original_quality_and_huber_unchanged"] is True
            and report["ineligible_points_remain_original_model"] is True
            and report["source_pixels_changed"] is False
            and report["fit_started"] is False
            and report["final_holdout_read"] is False,
            "Source-only support report provenance/semantics changed")
    npz_path = Path(report["output_npz"]["path"])
    require(npz_path.parent.resolve() == report_path.parent.resolve()
            and driver.digest(npz_path) == report["output_npz"]["sha256"],
            "Source-support NPZ path or hash differs")
    with np.load(npz_path, allow_pickle=False) as packet:
        require(np.array_equal(packet["pts"], pts)
                and np.array_equal(packet["offsets"], offsets)
                and np.array_equal(packet["uv"], uv)
                and str(packet["source_sha256"]) == data["captured"]["source_hash"]
                and str(packet["source_edge_sha256"]) == driver.digest(edge_path),
                "Support PTS/offsets/UV or source edge identity changed")
        eligible = np.asarray(packet["eligible"], bool).copy()
        normals = np.asarray(packet["normal_uv"], np.float64).copy()
        bounds = np.asarray(packet["bounds_normal_px"], np.float64).copy()
    require(eligible.shape == (len(uv),)
            and normals.shape == bounds.shape == (len(uv), 2)
            and int(np.sum(eligible)) == report["counts"]["eligible"],
            "Source-support point qualification count/shape changed")
    SourceNormalIntervals(normals, bounds, eligible)
    return eligible, normals, bounds, dict(report_path=str(report_path.resolve()),
        report_sha256=expected_sha, npz_path=str(npz_path.resolve()),
        npz_sha256=report["output_npz"]["sha256"], eligible_points=int(eligible.sum()),
        source_qa_approved_by_this_script=False)


def top_factor_indices(data: dict, pts: np.ndarray, top_offsets: np.ndarray,
                       foot_counts: np.ndarray) -> dict[int, int]:
    answer = {}
    for frame_index, stamp in enumerate(map(int, pts)):
        found = [(index, spec) for index, spec in enumerate(data["retained"])
                 if (spec["file"], spec["function"]) == TOP_KIND
                 and spec["keys"] == (stamp,)]
        require(len(found) == 2 and all(spec["kind"] == "observation"
                for _, spec in found), f"Original source edge factors changed at {stamp}")
        top_count = int(top_offsets[frame_index+1]-top_offsets[frame_index])
        require(found[0][1]["kwargs"]["residual_dim"] == 2*top_count
                and found[1][1]["kwargs"]["residual_dim"] == 2*foot_counts[frame_index]
                and float(found[0][1]["kwargs"]["sigma"]) == 2.
                and float(found[1][1]["kwargs"]["sigma"]) == 3.,
                f"Source top/contact factor order, UV count, or sigma changed at {stamp}")
        answer[stamp] = found[0][0]  # _add_source_edge_factors loops top before contact.
    require(len(set(answer.values())) == 119, "Top-edge factor indices are not unique")
    return answer


def replacement(spec: dict, stamp: int, observed: np.ndarray,
                world: np.ndarray, template: np.ndarray,
                intervals: SourceNormalIntervals, quality) -> dict:
    require(spec["kwargs"]["residual_dim"] == 2*len(observed)
            and float(spec["kwargs"]["sigma"]) == 2.
            and spec["kwargs"]["jac_fn"] is not None,
            f"Original top factor shape/sigma/J changed at {stamp}")

    def native(q, with_jacobian=False):
        camera = from_state(np.asarray(q[0], np.float64), template)
        segments = _edge_segment_candidates(camera, world, observed)
        return _edge_curve_error(camera, world, observed, segments,
                                 with_jacobian=with_jacobian, finite_vector=True)

    def raw_residual(q):
        return intervals.apply(native(q))

    def raw_jacobian(q):
        residual, matrix = native(q, with_jacobian=True)
        return intervals.apply(residual, [matrix])[1]

    multipliers = quality.scales(stamp, observed, "native_edge:board_top")
    residual, jacobian = quality.wrap(raw_residual, raw_jacobian,
                                      multipliers, group_size=2)
    residual, jacobian = pointwise_huber(residual, jacobian,
                                        group_size=2, delta=4.)
    changed = spec.copy()
    changed["args"] = (spec["args"][0], residual)
    changed["kwargs"] = {**spec["kwargs"], "jac_fn": jacobian}
    return changed


def callback_checks(data: dict, changed_data: dict, top_indices: dict[int, int],
                    initial: dict[int, np.ndarray], flags: np.ndarray,
                    offsets: np.ndarray, support_enabled: bool):
    max_residual = max_jacobian = max_cost = 0.
    max_unqualified_residual = max_unqualified_jacobian = 0.
    ordered = sorted(top_indices)
    for stamp, index in top_indices.items():
        old = data["retained"][index]
        new = changed_data["retained"][index]
        physical = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]
        q = [physical]
        old_r = np.asarray(old["args"][1](q), float)
        new_r = np.asarray(new["args"][1](q), float)
        old_j = np.asarray(old["kwargs"]["jac_fn"](q)[0], float)
        new_j = np.asarray(new["kwargs"]["jac_fn"](q)[0], float)
        require(old_r.shape == new_r.shape and old_j.shape == new_j.shape,
                f"Top factor callback dimension changed: {stamp}")
        max_residual = max(max_residual, float(np.max(np.abs(old_r-new_r))))
        max_jacobian = max(max_jacobian, float(np.max(np.abs(old_j-new_j))))
        old_cost = .5*float(np.sum((old_r/2.)**2))
        new_cost = .5*float(np.sum((new_r/2.)**2))
        max_cost = max(max_cost, abs(old_cost-new_cost))
        frame_index = ordered.index(stamp)
        start, stop = map(int, offsets[frame_index:frame_index+2])
        unqualified = np.repeat(~flags[start:stop], 2)
        if np.any(unqualified):
            max_unqualified_residual = max(max_unqualified_residual,
                float(np.max(np.abs(old_r[unqualified]-new_r[unqualified]))))
            max_unqualified_jacobian = max(max_unqualified_jacobian,
                float(np.max(np.abs(old_j[unqualified]-new_j[unqualified]))))
        if not support_enabled:
            require(max_residual < 1e-9 and max_jacobian < 1e-7
                    and max_cost < 1e-8,
                    f"Disabled interval callback changed original top factor: {stamp}")
    require(max_unqualified_residual < 1e-9 and max_unqualified_jacobian < 1e-7,
            "Unqualified original UV rows changed under source interval model")
    return dict(top_factors=len(top_indices), max_residual_delta=max_residual,
                max_jacobian_delta=max_jacobian, max_cost_delta=max_cost,
        max_unqualified_residual_delta=max_unqualified_residual,
        max_unqualified_jacobian_delta=max_unqualified_jacobian,
                disabled_must_equal_original=not support_enabled)


def qualified_fd(data: dict, initial: dict[int, np.ndarray], pts: np.ndarray,
                 offsets: np.ndarray, uv: np.ndarray, flags: np.ndarray,
                 normals: np.ndarray, bounds: np.ndarray,
                 world: np.ndarray) -> list[dict]:
    """Finite-difference actual eligible source rows on both interval branches."""
    if not np.any(flags):
        return []
    candidates: dict[str, list[tuple[float, int, int]]] = {"interior": [], "exterior": []}
    prepared = {}
    for frame_index, stamp in enumerate(map(int, pts)):
        start, stop = map(int, offsets[frame_index:frame_index+2])
        if not np.any(flags[start:stop]):
            continue
        observed = uv[start:stop]
        interval = SourceNormalIntervals(normals[start:stop], bounds[start:stop], flags[start:stop])
        template = data["captured"]["templates"][stamp]
        state = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]

        def raw(q, with_jacobian=False, template=template, observed=observed):
            camera = from_state(q, template)
            segments = _edge_segment_candidates(camera, world, observed)
            return _edge_curve_error(camera, world, observed, segments,
                                     with_jacobian=with_jacobian, finite_vector=True)

        residual, jacobian = raw(state, True)
        prepared[stamp] = (state, observed, interval, raw, residual, jacobian)
        vector = residual.reshape(-1, 2)
        for point in np.flatnonzero(flags[start:stop]):
            point = int(point)
            normal = normals[start+point]
            low, high = bounds[start+point]
            signed = float(np.dot(vector[point], normal))
            if low+.2 < signed < high-.2:
                candidates["interior"].append((min(signed-low, high-signed), stamp, point))
            elif signed < low-.2 or signed > high+.2:
                candidates["exterior"].append((max(low-signed, signed-high), stamp, point))
    checked = []
    for branch in ("interior", "exterior"):
        require(candidates[branch], f"No actual qualified {branch} source point")
        margin, stamp, point = sorted(candidates[branch], reverse=True)[0]
        state, observed, interval, raw, residual, jacobian = prepared[stamp]
        transformed, mapped = interval.apply(residual, [jacobian])
        h = 1e-7
        errors = []
        relative = []
        for column in range(3, 11):
            hi, lo = state.copy(), state.copy()
            hi[column] += h
            lo[column] -= h
            numeric = (interval.apply(raw(hi))-interval.apply(raw(lo)))/(2*h)
            rows = slice(2*point, 2*point+2)
            diff = numeric[rows]-mapped[0][rows, column]
            errors.append(float(np.max(np.abs(diff))))
            relative.append(float(np.max(np.abs(diff)/
                (1.+np.abs(mapped[0][rows, column])))))
        max_abs, max_rel = max(errors), max(relative)
        start = int(offsets[np.searchsorted(pts, stamp)])
        low, high = map(float, bounds[start+point])
        signed = float(np.dot(residual.reshape(-1, 2)[point],
                              normals[start+point]))
        result = dict(branch=branch, selection="largest_margin_actual_eligible_source_point",
            pts=stamp, original_uv=observed[point].tolist(),
            support_bounds_px=[low, high], signed_raw_normal_px=signed,
            distance_to_support_boundary_px=float(margin),
            transformed_raw_xy=transformed.reshape(-1, 2)[point].tolist(),
            h=h, columns=list(range(3, 11)),
            max_absolute_jacobian_error=max_abs,
            max_relative_jacobian_error=max_rel)
        require(max_abs < .01 and max_rel < 1e-5,
                f"Selected actual qualified {branch} camera-chain FD failed: {result}")
        checked.append(result)
    return checked


def factor_checks(data: dict, changed_data: dict, pairs: list,
                  top_indices: dict[int, int], initial: dict[int, np.ndarray],
                  support_enabled: bool):
    old, _ = driver.build_graph(data, pairs, "fixed", "derived")
    new, _ = driver.build_graph(changed_data, pairs, "fixed", "derived")
    require(len(old._pending) == len(new._pending) == 633,
            "Original/new conditioned graphs are not both 633 factors")
    values = gtsam.Values()
    for key, state in initial.items():
        values.insert(old._key(key), state)
    top = set(top_indices.values())
    unchanged = [i for i in range(633) if i not in top]
    max_other = max(abs(old._pending[i].error(values)-new._pending[i].error(values))
                    for i in unchanged)
    top_cost = {name:float(sum(graph._pending[i].error(values) for i in top))
                for name, graph in (("original", old), ("interval", new))}
    require(max_other < 1e-9, "A non-top factor changed under interval model")
    if not support_enabled:
        require(abs(top_cost["original"]-top_cost["interval"]) < 1e-8,
                "Disabled interval graph changed top factor cost")
    return dict(original_factors=633, replaced_top_factors=119,
        other_factors=514, max_unchanged_factor_cost_difference=max_other,
        top_factor_cost_at_profile_initializer=top_cost,
        objective_unchanged=not support_enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--support-report", type=Path)
    parser.add_argument("--support-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require((args.support_report is None) == (args.support_sha256 is None),
            "Support report and caller-supplied SHA must be provided together")
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Diagnostic outputs must remain in project tracking TMP")
    if args.run:
        require(args.support_report is not None and (args.output/"phase_00.json").is_file(),
                "Run requires frozen source support and prior matching preflight")
    else:
        require(not (args.output/"phase_00.json").exists(),
                "Preflight already exists; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    data = driver.prepare(seed_driver.CONFIG, seed_driver.CANDIDATE,
                          seed_driver.MANIFEST, (3,), args.output,
                          complete_camera_group=False)
    pairs, counts, _, _, validity = driver.native_pairs(data)
    require(counts["adjacent_pairs"] == 118 and counts["pixels"] == 8132
            and validity["invalid_pair_count"] == 0,
            "Original shot03 native board motion changed")
    initial, profile_sha = frozen_profile_states(data)
    edge_path, pts, offsets, source_uv, foot_counts = source_top(data)
    flags, normals, bounds, support = load_support(args.support_report,
        args.support_sha256, edge_path, pts, offsets, source_uv, data)
    indices = top_factor_indices(data, pts, offsets, foot_counts)
    changed = {**data, "retained":data["retained"].copy()}
    identity = {**data, "retained":data["retained"].copy()}
    world = _closed_edge_ring(np.asarray(data["scene"]["board_surface"]["top_edge_xyz"], float))
    for frame_index, stamp in enumerate(map(int, pts)):
        start, stop = map(int, offsets[frame_index:frame_index+2])
        interval = SourceNormalIntervals(normals[start:stop], bounds[start:stop],
                                         flags[start:stop])
        unknown = SourceNormalIntervals(np.full((stop-start, 2), np.nan),
                                        np.full((stop-start, 2), np.nan),
                                        np.zeros(stop-start, bool))
        index = indices[stamp]
        identity["retained"][index] = replacement(data["retained"][index], stamp,
            source_uv[start:stop], world, data["captured"]["templates"][stamp],
            unknown, data["captured"]["source_quality"])
        changed["retained"][index] = replacement(data["retained"][index], stamp,
            source_uv[start:stop], world, data["captured"]["templates"][stamp],
            interval, data["captured"]["source_quality"])
    top_set = set(indices.values())
    require(len(changed["retained"]) == len(data["retained"]) and
            all(changed["retained"][i] is data["retained"][i]
                for i in range(len(data["retained"])) if i not in top_set),
            "A non-top captured source factor was replaced")
    identity_check = callback_checks(data, identity, indices, initial,
                                     np.zeros(len(source_uv), bool), offsets, False)
    callback = callback_checks(data, changed, indices, initial, flags, offsets,
                               support is not None)
    factor = factor_checks(data, changed, pairs, indices, initial, support is not None)
    fd = qualified_fd(data, initial, pts, offsets, source_uv, flags, normals, bounds,
                      world) if support is not None else []
    phase = dict(schema="rinkspace.shot03-blue-top-interval-preflight.v1",
        scope="119-frame exact-mount fixed-mode diagnostic; original scene, UV, 118 native pairs and 514 non-top factors retained",
        source_sha256=data["captured"]["source_hash"],
        scene_sha256=data["diag"]["model_sha256"],
        support=support, support_qa_approved_by_this_script=False,
        original_objective_unchanged=support is None,
        only_board_top_factors_changed=support is not None,
        observed_uv_changed=False, source_quality_sigma_huber_unchanged=True,
        complete_programme_acceptance=False, final_holdout_read=False,
        input_sha256=dict(config=driver.digest(seed_driver.CONFIG),
            candidate_scene=driver.digest(seed_driver.CANDIDATE/"candidate_scene.json"),
            manifest=driver.digest(seed_driver.MANIFEST),
            edge=driver.digest(edge_path), profile_summary=profile_sha,
            profile_initial_states=driver.digest(PROFILE_STATES),
            driver=driver.digest(driver.__file__),
            wrapper=driver.digest(__file__),
            interval_core=driver.digest(Path(__file__).with_name("source_edge_intervals.py")),
            track_sequence=driver.digest(Path(__file__).with_name("track_sequence.py"))),
        native_pair_count=118, native_source_points=8132,
        source_top_points=len(source_uv), qualified_top_points=int(flags.sum()),
        original_top_identity_from_source_uv_world=identity_check,
        callback_equivalence=callback, factor_equivalence=factor,
        actual_qualified_camera_chain_fd=fd,
        solver_started=False)
    phase_path = args.output/"phase_00.json"
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Frozen preflight differs from current support/code/objective")
    else:
        phase_path.write_text(json.dumps(phase, indent=2), encoding="utf8")
    print(json.dumps(dict(phase="preflight", path=str(phase_path),
        sha256=driver.digest(phase_path), qualified_top_points=int(flags.sum()),
        callback=callback, factor=factor)), flush=True)
    if not args.run:
        return
    target = args.output/"fixed"
    require(not target.exists(), "Existing interval solve stage needs review")
    provenance = dict(source="profile_119_exact_derived_121_nodes",
        expanded_profile_states_sha256=driver.digest(PROFILE_STATES),
        profile_summary_sha256=profile_sha,
        support_report_sha256=args.support_sha256,
        support_npz_sha256=support["npz_sha256"],
        original_objective_unchanged=False,
        only_board_top_factors_changed=True,
        original_source_uv_unchanged=True,
        preflight_sha256=driver.digest(phase_path))
    result = driver.solve_one(changed, pairs, "fixed", target,
        "derived", initial_override=initial, initial_provenance=provenance)
    print(json.dumps(dict(phase="solve_complete", summary=str(target/"summary.json"),
                          summary_sha256=driver.digest(target/"summary.json"),
                          final_cost=result["batch"]["final_nonlinear_cost"])), flush=True)


if __name__ == "__main__":
    main()
