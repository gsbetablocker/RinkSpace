"""Conditional shot03 source-interval diagnostic for both original board edges.

Without a contact support report this only checks that rebuilt callbacks and
the 633-factor graph reproduce the original objective. A frozen source-only
contact report, supplied with its SHA, enables both edge intervals. The top
support is the previously frozen blue-top report. ``--qualified-only`` keeps
every original UV in the archive but gives source-unknown rows zero residual
and Jacobian before the unchanged quality/Huber pipeline. No other factor is
changed. ``--run`` requires a matching saved active preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
import conditioned_shot03_lens19_seed as seed_driver
import shot03_board_top_interval_diagnostic as top
from projection import from_state
from rig_parameterization import physical_frame
from robust_observations import pointwise_huber
from source_edge_intervals import SourceNormalIntervals
from track_sequence import (SourceEdges, _closed_edge_ring,
                            _edge_curve_error, _edge_segment_candidates)


TMP = seed_driver.TMP
DEFAULT_OUTPUT = TMP / "shot03_board_edges_interval_identity_preflight_v1"
TOP_SUPPORT = TMP / "shot03_board_top_support_v1/source_support.json"
TOP_SUPPORT_SHA256 = "c31433e6a1ea6aa968f06f4e2eb5c080af18493127e531aafbcc73f8f241ba3a"
EDGE_KIND = ("track_sequence.py", "_add_source_edge_factors")
EDGE_NAMES = ("board_top", "ice_board_contact")
EDGE_SIGMAS = (2., 3.)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_edges(data: dict):
    edge_path, pts, top_offsets, top_uv, _ = top.source_top(data)
    edges = SourceEdges.load(edge_path, pts,
        source_sha256=data["captured"]["source_hash"],
        model_sha256=data["diag"]["base_model_sha256"])
    contact_frames = [edges.frame(i, 1) for i in range(len(pts))]
    contact_offsets = np.r_[0, np.cumsum([len(frame) for frame in contact_frames])]
    contact_uv = np.concatenate(contact_frames)
    require(len(contact_uv) == 6579, "Original contact source UV count changed")
    return edge_path, pts, ((top_offsets, top_uv), (contact_offsets, contact_uv))


def load_contact_support(report_path: Path | None, expected_sha: str | None,
                         edge_path: Path, pts: np.ndarray, offsets: np.ndarray,
                         uv: np.ndarray, data: dict):
    if report_path is None:
        require(expected_sha is None, "Contact SHA requires a report")
        return (np.zeros(len(uv), bool), np.full((len(uv), 2), np.nan),
                np.full((len(uv), 2), np.nan), None)
    require(expected_sha is not None and len(expected_sha) == 64
            and driver.digest(report_path) == expected_sha,
            "Frozen contact report SHA is absent or differs")
    report = json.loads(report_path.read_text(encoding="utf8"))
    require(report["schema"] == "rinkspace.shot03-contact-source-support.v2"
            and report["mode"] == "all"
            and report["counts"]["frames"] == 119
            and report["counts"]["original_uv"] == 6579
            and report["source_video"]["sha256"] == data["captured"]["source_hash"]
            and report["source_edge_cache"]["sha256"] == driver.digest(edge_path)
            and Path(report["source_edge_cache"]["path"]).resolve() == edge_path.resolve()
            and report["source_edge_cache"]["original_contact_uv"] == 6579
            and report["source_edge_cache"]["original_top_uv"] == 3995
            and report["original_sigma_px"] == 3.
            and report["original_quality_and_huber_unchanged"] is True
            and report["all_ineligible_uv_retained"] is True
            and report["fit_started"] is False
            and report["candidate_camera_or_residual_read"] is False
            and report["final_holdout_read"] is False,
            "Contact report source identity or source-only semantics changed")
    require([int(row["pts"]) for row in report["per_frame"]] == list(map(int, pts))
            and [int(row["original_contact_uv"]) for row in report["per_frame"]]
            == list(map(int, np.diff(offsets))),
            "Contact report did not retain all original frame PTS and UV counts")
    npz_path = Path(report["output_npz"]["path"])
    require(npz_path.parent.resolve() == report_path.parent.resolve()
            and driver.digest(npz_path) == report["output_npz"]["sha256"],
            "Contact NPZ path or SHA differs from frozen report")
    with np.load(npz_path, allow_pickle=False) as packet:
        require(np.array_equal(packet["pts"], pts)
                and np.array_equal(packet["offsets"], offsets)
                and np.array_equal(packet["uv"], uv)
                and str(packet["source_sha256"]) == data["captured"]["source_hash"]
                and str(packet["source_edge_sha256"]) == driver.digest(edge_path),
                "Contact support changed the original PTS, offsets or UV")
        flags = np.asarray(packet["eligible"], bool).copy()
        normals = np.asarray(packet["normal_uv"], np.float64).copy()
        bounds = np.asarray(packet["bounds_normal_px"], np.float64).copy()
    require(flags.shape == (len(uv),)
            and normals.shape == bounds.shape == (len(uv), 2)
            and int(flags.sum()) == report["counts"]["eligible"],
            "Contact eligibility shape/count differs")
    SourceNormalIntervals(normals, bounds, flags)
    return flags, normals, bounds, dict(report_path=str(report_path.resolve()),
        report_sha256=expected_sha, npz_path=str(npz_path.resolve()),
        npz_sha256=report["output_npz"]["sha256"],
        eligible_points=int(flags.sum()), source_qa_approved_by_this_script=False)


def factor_indices(data: dict, pts: np.ndarray,
                   edge_data: tuple[tuple[np.ndarray, np.ndarray], ...]):
    indices: dict[tuple[int, int], int] = {}
    for frame_index, stamp in enumerate(map(int, pts)):
        found = [(index, spec) for index, spec in enumerate(data["retained"])
                 if (spec["file"], spec["function"]) == EDGE_KIND
                 and spec["keys"] == (stamp,)]
        require(len(found) == 2, f"Original edge-factor count changed at {stamp}")
        for edge_id, (index, spec) in enumerate(found):
            offsets, _ = edge_data[edge_id]
            count = int(offsets[frame_index+1]-offsets[frame_index])
            require(spec["kind"] == "observation"
                    and spec["kwargs"]["residual_dim"] == 2*count
                    and float(spec["kwargs"]["sigma"]) == EDGE_SIGMAS[edge_id]
                    and spec["kwargs"]["jac_fn"] is not None,
                    f"Original {EDGE_NAMES[edge_id]} factor order/shape changed at {stamp}")
            indices[(stamp, edge_id)] = index
    require(len(indices) == 238 and len(set(indices.values())) == 238,
            "Original source board factors are not 119 pairs")
    return indices


def replacement(spec: dict, stamp: int, edge_id: int, observed: np.ndarray,
                world: np.ndarray, template: np.ndarray,
                interval: SourceNormalIntervals, quality, *, qualified_only=False) -> dict:
    # This replaces the complete objective: nearest-curve residual first,
    # then source-normal interval, optional unknown-row zeroing, source
    # quality, and Huber. The original source-edge guard/native factor does
    # not encode the interval or qualification mask and cannot be rebound.
    sigma = EDGE_SIGMAS[edge_id]
    require(spec["kwargs"]["residual_dim"] == 2*len(observed)
            and float(spec["kwargs"]["sigma"]) == sigma,
            f"Original {EDGE_NAMES[edge_id]} factor changed at {stamp}")

    def native(q, with_jacobian=False):
        camera = from_state(np.asarray(q[0], np.float64), template)
        segments = _edge_segment_candidates(camera, world, observed)
        return _edge_curve_error(camera, world, observed, segments,
                                 with_jacobian=with_jacobian, finite_vector=True)

    unknown_rows = np.repeat(~interval.eligible, 2) if qualified_only else None

    def raw_residual(q):
        result = interval.apply(native(q))
        if unknown_rows is not None:
            result[unknown_rows] = 0.
        return result

    def raw_jacobian(q):
        residual, matrix = native(q, with_jacobian=True)
        result = interval.apply(residual, [matrix])[1]
        if unknown_rows is not None:
            result[0][unknown_rows] = 0.
        return result

    scales = quality.scales(stamp, observed, "native_edge:"+EDGE_NAMES[edge_id])
    residual, jacobian = quality.wrap(raw_residual, raw_jacobian, scales, group_size=2)
    residual, jacobian = pointwise_huber(residual, jacobian,
                                        group_size=2, delta=2*sigma)
    changed = spec.copy()
    changed["args"] = (spec["args"][0], residual)
    changed["kwargs"] = {**spec["kwargs"], "jac_fn": jacobian}
    return changed


def callback_checks(data: dict, changed: dict, indices: dict, initial: dict,
                    edge_data: tuple, flags: tuple, enabled: bool,
                    qualified_only: bool = False):
    maxima = dict(residual=0., jacobian=0., cost=0.,
                  ineligible_residual=0., ineligible_jacobian=0.)
    # The source cache, not the retained-factor capture order, defines offsets.
    frame_index = {stamp:i for i, stamp in enumerate(sorted(s for s, edge_id in indices
                                                             if edge_id == 0))}
    for (stamp, edge_id), index in indices.items():
        old, new = data["retained"][index], changed["retained"][index]
        physical = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]
        q = [physical]
        old_r = np.asarray(old["args"][1](q), float)
        new_r = np.asarray(new["args"][1](q), float)
        old_j = np.asarray(old["kwargs"]["jac_fn"](q)[0], float)
        new_j = np.asarray(new["kwargs"]["jac_fn"](q)[0], float)
        require(old_r.shape == new_r.shape and old_j.shape == new_j.shape,
                f"Edge callback shape changed at {stamp}/{EDGE_NAMES[edge_id]}")
        maxima["residual"] = max(maxima["residual"], float(np.max(np.abs(old_r-new_r))))
        maxima["jacobian"] = max(maxima["jacobian"], float(np.max(np.abs(old_j-new_j))))
        sigma = EDGE_SIGMAS[edge_id]
        maxima["cost"] = max(maxima["cost"], abs(.5*float(np.sum((old_r/sigma)**2))
                                          -.5*float(np.sum((new_r/sigma)**2))))
        offsets, _ = edge_data[edge_id]
        start, stop = map(int, offsets[frame_index[stamp]:frame_index[stamp]+2])
        ineligible = np.repeat(~flags[edge_id][start:stop], 2)
        if np.any(ineligible):
            reference_r = np.zeros_like(new_r) if qualified_only else old_r
            reference_j = np.zeros_like(new_j) if qualified_only else old_j
            maxima["ineligible_residual"] = max(maxima["ineligible_residual"],
                float(np.max(np.abs(reference_r[ineligible]-new_r[ineligible]))))
            maxima["ineligible_jacobian"] = max(maxima["ineligible_jacobian"],
                float(np.max(np.abs(reference_j[ineligible]-new_j[ineligible]))))
    require(maxima["ineligible_residual"] < 1e-9
            and maxima["ineligible_jacobian"] < 1e-7,
            "An ineligible source UV row violates its selected policy")
    if not enabled:
        require(maxima["residual"] < 1e-9 and maxima["jacobian"] < 1e-7
                and maxima["cost"] < 1e-8,
                "Disabled edge intervals changed the original callback")
    return dict(factors=len(indices), **maxima,
                ineligible_policy=("zero_data_rows" if qualified_only else "original_rows"),
                disabled_must_equal_original=not enabled)


def factor_checks(data: dict, changed: dict, pairs: list, indices: dict,
                  initial: dict, enabled: bool):
    old, _ = driver.build_graph(data, pairs, "fixed", "derived")
    new, _ = driver.build_graph(changed, pairs, "fixed", "derived")
    require(len(old._pending) == len(new._pending) == 633,
            "Original/new shot03 graph must each have 633 factors")
    values = gtsam.Values()
    for key, state in initial.items():
        values.insert(old._key(key), state)
    replaced = set(indices.values())
    other = [i for i in range(633) if i not in replaced]
    require(len(other) == 395, "Unexpected number of untouched factors")
    max_other = max(abs(old._pending[i].error(values)-new._pending[i].error(values))
                    for i in other)
    require(max_other < 1e-9, "A non-board-edge factor changed")
    edge_costs = {}
    for edge_id, name in enumerate(EDGE_NAMES):
        selected = [i for (stamp, kind), i in indices.items() if kind == edge_id]
        edge_costs[name] = {label:float(sum(graph._pending[i].error(values)
                           for i in selected))
                            for label, graph in (("original", old), ("interval", new))}
        if not enabled:
            require(abs(edge_costs[name]["original"]-edge_costs[name]["interval"]) < 1e-8,
                    f"Disabled {name} graph factor costs changed")
    return dict(original_factors=633, replaceable_edge_factors=238,
                other_factors=395, max_other_factor_cost_delta=max_other,
                edge_costs_at_profile_initializer=edge_costs,
                objective_unchanged=not enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contact-report", type=Path)
    parser.add_argument("--contact-sha256")
    parser.add_argument("--qualified-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require((args.contact_report is None) == (args.contact_sha256 is None),
            "Contact report and caller-supplied SHA must be provided together")
    enabled = args.contact_report is not None
    require(not args.qualified_only or enabled,
            "Qualified-only needs both frozen source support reports")
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Diagnostic outputs must remain in project tracking TMP")
    if args.run:
        require(enabled and (args.output/"phase_00.json").is_file(),
                "Run requires frozen dual-edge preflight and contact support")
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
            "Original 118/8132 shot03 native motion changed")
    initial, profile_sha = top.frozen_profile_states(data)
    edge_path, pts, edge_data = source_edges(data)
    indices = factor_indices(data, pts, edge_data)
    support = [None, None]
    interval_inputs = []
    if enabled:
        top_offsets, top_uv = edge_data[0]
        interval_inputs.append(top.load_support(TOP_SUPPORT, TOP_SUPPORT_SHA256,
            edge_path, pts, top_offsets, top_uv, data))
        foot_offsets, foot_uv = edge_data[1]
        interval_inputs.append(load_contact_support(args.contact_report,
            args.contact_sha256, edge_path, pts, foot_offsets, foot_uv, data))
    else:
        for offsets, uv in edge_data:
            interval_inputs.append((np.zeros(len(uv), bool),
                np.full((len(uv), 2), np.nan), np.full((len(uv), 2), np.nan), None))
    flags = tuple(item[0] for item in interval_inputs)
    support = [item[3] for item in interval_inputs]
    worlds = (_closed_edge_ring(np.asarray(data["scene"]["board_surface"]["top_edge_xyz"], float)),
              _closed_edge_ring(np.asarray(data["scene"]["ice_boundary"]["points_xyz"], float)))
    identity = {**data, "retained":data["retained"].copy()}
    changed = {**data, "retained":data["retained"].copy()}
    for frame_index, stamp in enumerate(map(int, pts)):
        for edge_id in range(2):
            offsets, uv = edge_data[edge_id]
            start, stop = map(int, offsets[frame_index:frame_index+2])
            qualified, normals, bounds, _ = interval_inputs[edge_id]
            actual = SourceNormalIntervals(normals[start:stop], bounds[start:stop],
                                           qualified[start:stop])
            unknown = SourceNormalIntervals(np.full((stop-start, 2), np.nan),
                np.full((stop-start, 2), np.nan), np.zeros(stop-start, bool))
            index = indices[(stamp, edge_id)]
            args_for_replacement = (data["retained"][index], stamp, edge_id,
                uv[start:stop], worlds[edge_id], data["captured"]["templates"][stamp])
            quality = data["captured"]["source_quality"]
            identity["retained"][index] = replacement(*args_for_replacement, unknown, quality)
            changed["retained"][index] = replacement(*args_for_replacement, actual, quality,
                qualified_only=args.qualified_only)
    replaced = set(indices.values())
    require(len(data["retained"]) == len(changed["retained"])
            and all(changed["retained"][i] is data["retained"][i]
                    for i in range(len(data["retained"])) if i not in replaced),
            "A factor outside the original 238 board edges was replaced")
    zero_flags = tuple(np.zeros(len(uv), bool) for _, uv in edge_data)
    identity_check = callback_checks(data, identity, indices, initial,
                                     edge_data, zero_flags, False)
    active_check = callback_checks(data, changed, indices, initial,
                                   edge_data, flags, enabled, args.qualified_only)
    factor = factor_checks(data, changed, pairs, indices, initial, enabled)
    actual_fd = {}
    if enabled:
        for edge_id, name in enumerate(EDGE_NAMES):
            offsets, uv = edge_data[edge_id]
            qualified, normals, bounds, _ = interval_inputs[edge_id]
            actual_fd[name] = top.qualified_fd(data, initial, pts, offsets, uv,
                                                qualified, normals, bounds, worlds[edge_id])
    phase = dict(schema="rinkspace.shot03-dual-board-edge-interval-preflight.v1",
        scope="119-frame exact-mount fixed shot03 diagnostic; original scene, UV, 118 native pairs and 395 non-edge factors retained",
        source_sha256=data["captured"]["source_hash"],
        scene_sha256=data["diag"]["model_sha256"],
        support=dict(zip(EDGE_NAMES, support)),
        support_qa_approved_by_this_script=False,
        original_objective_unchanged=not enabled,
        only_original_238_board_edge_factors_replaceable=True,
        ineligible_source_rows_policy=("zero_data_rows" if args.qualified_only
                                       else "original_rows"),
        ineligible_rows_zeroed_before_quality_huber=args.qualified_only,
        observed_uv_changed=False, source_quality_sigma_huber_unchanged=True,
        complete_programme_acceptance=False, final_holdout_read=False,
        input_sha256=dict(config=driver.digest(seed_driver.CONFIG),
            candidate_scene=driver.digest(seed_driver.CANDIDATE/"candidate_scene.json"),
            manifest=driver.digest(seed_driver.MANIFEST),
            edge=driver.digest(edge_path), profile_summary=profile_sha,
            profile_initial_states=driver.digest(top.PROFILE_STATES),
            driver=driver.digest(driver.__file__), wrapper=driver.digest(__file__),
            top_wrapper=driver.digest(top.__file__),
            interval_core=driver.digest(Path(__file__).with_name("source_edge_intervals.py")),
            track_sequence=driver.digest(Path(__file__).with_name("track_sequence.py"))),
        original_factors=633, native_pair_count=118, native_source_points=8132,
        original_edge_points={name:len(edge_data[i][1]) for i,name in enumerate(EDGE_NAMES)},
        qualified_points={name:int(flags[i].sum()) for i,name in enumerate(EDGE_NAMES)},
        qualified_counts_by_original_pts={name:[dict(pts=int(stamp), original_points=int(stop-start),
            qualified_points=int(flags[i][start:stop].sum()))
            for stamp, start, stop in zip(pts, edge_data[i][0][:-1], edge_data[i][0][1:])]
            for i,name in enumerate(EDGE_NAMES)},
        original_edge_identity_from_source_uv_world=identity_check,
        callback_equivalence=active_check, factor_equivalence=factor,
        actual_qualified_camera_chain_fd=actual_fd,
        solver_started=False)
    phase_path = args.output/"phase_00.json"
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Frozen preflight differs from current source support, code or objective")
    else:
        phase_path.write_text(json.dumps(phase, indent=2), encoding="utf8")
    print(json.dumps(dict(phase="preflight", path=str(phase_path),
        sha256=driver.digest(phase_path), qualified_points=phase["qualified_points"],
        callbacks=active_check, factors=factor)), flush=True)
    if not args.run:
        return
    target = args.output/"fixed"
    require(not target.exists(), "Existing dual-edge solve stage needs review")
    provenance = dict(source="profile_119_exact_derived_121_nodes",
        expanded_profile_states_sha256=driver.digest(top.PROFILE_STATES),
        profile_summary_sha256=profile_sha,
        source_support_sha256={name:support[i]["report_sha256"]
                               for i,name in enumerate(EDGE_NAMES)},
        support_npz_sha256={name:support[i]["npz_sha256"]
                            for i,name in enumerate(EDGE_NAMES)},
        original_objective_unchanged=False,
        only_board_top_and_contact_factors_changed=True,
        ineligible_source_rows_policy=("zero_data_rows" if args.qualified_only
                                       else "original_rows"),
        original_source_uv_unchanged=True,
        preflight_sha256=driver.digest(phase_path))
    result = driver.solve_one(changed, pairs, "fixed", target,
        "derived", initial_override=initial, initial_provenance=provenance)
    print(json.dumps(dict(phase="solve_complete", summary=str(target/"summary.json"),
        summary_sha256=driver.digest(target/"summary.json"),
        final_cost=result["batch"]["final_nonlinear_cost"])), flush=True)


if __name__ == "__main__":
    main()
