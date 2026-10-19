"""Source-normal board-edge interval diagnostic on all low_end shots.

The original fixed/derived 11D graph, shared mount/lens, priors, source UV,
quality, Huber rules, and native motion remain unchanged. Only qualified
board-top/contact rows use frozen source-colour support; unknown rows retain
their original residual. ``disabled`` builds the exact original objective.
This is a conditional diagnostic, not programme acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
import low_end_calibration_transfer_profile as transfer
import shot03_board_edges_interval_diagnostic as shot03
import shot03_board_top_interval_diagnostic as shot03_top
from projection import from_state
from rig_parameterization import physical_frame
from source_edge_intervals import SourceNormalIntervals
from track_sequence import (SourceEdges, _closed_edge_ring,
                            _edge_curve_error, _edge_segment_candidates)


TMP = transfer.TMP
OUTPUT = TMP / "low_end_board_edges_interval_diagnostic_v1"
ADAPTIVE_FULL = TMP / "low_end_source_edge_support_05_08_adaptive_v2/full"
ADAPTIVE_REPORT = ADAPTIVE_FULL / "source_support.json"
ADAPTIVE_RULE_SHA256 = "a6add98e66e807bcc665b3c2839d28ffeaeb8012a09f1e83aa148f18f6a4c841"
SHOT03_TOP_REPORT = shot03.TOP_SUPPORT
SHOT03_TOP_SHA256 = shot03.TOP_SUPPORT_SHA256
SHOT03_CONTACT_REPORT = TMP / "shot03_contact_support_v2/source_support.json"
SHOT03_CONTACT_SHA256 = "5e7216d54713a2c857de4755e6a47db0fb9fc2af675577381f9fbda560bc284e"
EDGE_SHA256 = {
    3: "28c3cea01dbfe6d9654cc49edcb968867b3cb35d921a19373855d422e1dbee04",
    5: "20c576eb4d26d3cd4e11d909f8bce85b940e1ca3bc303cbe577b0b572292fb8e",
    8: "1fd5ab70618ee284ba05473bfe78522a492246b4075256b816d290843887ad86",
}
EXPECTED_EDGE_POINTS = {3: (3995, 6579), 5: (81823, 60190), 8: (27223, 18976)}
EXPECTED_FRAMES = {3: 119, 5: 1653, 8: 447}
EDGE_NAMES = shot03.EDGE_NAMES
EDGE_SIGMAS = shot03.EDGE_SIGMAS
EDGE_KIND = shot03.EDGE_KIND


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def source_edges(data: dict) -> tuple[dict, dict[int, int]]:
    """Preserve the exact source-edge order and original float32 UVs."""
    edge_data = {}
    shot_by_pts = {}
    config_shots = {int(row["shot_index"]): row for row in data["config"]["shots"]}
    for shot in data["shots"]:
        number = int(shot["index"])
        require(number in EXPECTED_FRAMES, f"Unexpected low_end shot {number}")
        pts = np.asarray(shot["pts"], np.int64)
        require(len(pts) == EXPECTED_FRAMES[number],
                f"Original shot{number:02d} frame count changed")
        path = Path(config_shots[number]["source_edges"])
        require(driver.digest(path) == EDGE_SHA256[number],
                f"Original shot{number:02d} source-edge cache changed")
        edges = SourceEdges.load(path, pts,
            source_sha256=data["captured"]["source_hash"],
            model_sha256=data["diag"]["base_model_sha256"])
        # SourceEdges intentionally promotes native float32 pixels to float64
        # for projection. Keep a separate byte-exact copy for source identity.
        with np.load(path, allow_pickle=False) as native:
            raw_pts = np.asarray(native["pts"], np.int64)
            raw_offsets = np.asarray(native["offsets"], np.int64)
            raw_uv_all = np.asarray(native["observed_uv"]).copy()
            raw_edge_id = np.asarray(native["edge_id"], np.int8)
        require(np.array_equal(raw_pts, pts)
                and np.array_equal(raw_offsets, edges.offsets)
                and raw_uv_all.dtype == np.float32
                and np.array_equal(raw_edge_id, edges.edge_id),
                f"Original shot{number:02d} raw edge packet changed")
        for edge_id in range(2):
            parts = [edges.frame(i, edge_id) for i in range(len(pts))]
            raw_parts = [raw_uv_all[raw_offsets[i]:raw_offsets[i+1]][
                raw_edge_id[raw_offsets[i]:raw_offsets[i+1]] == edge_id]
                for i in range(len(pts))]
            offsets = np.r_[0, np.cumsum([len(part) for part in parts])].astype(np.int64)
            uv = np.concatenate(parts)
            raw_uv = np.concatenate(raw_parts)
            require(len(uv) == EXPECTED_EDGE_POINTS[number][edge_id]
                    and uv.dtype == np.float64 and raw_uv.dtype == np.float32
                    and offsets[-1] == len(uv)
                    and np.array_equal(uv, raw_uv.astype(np.float64)),
                    f"Original shot{number:02d} {EDGE_NAMES[edge_id]} UV changed")
            edge_data[(number, edge_id)] = dict(path=path, pts=pts,
                offsets=offsets, uv=uv, raw_uv=raw_uv)
        for stamp in map(int, pts):
            require(stamp not in shot_by_pts, "Overlapping original low_end PTS")
            shot_by_pts[stamp] = number
    require(len(shot_by_pts) == 2219, "Original low_end PTS coverage changed")
    return edge_data, shot_by_pts


def load_adaptive_packet(number: int, edge_id: int, expected_sha: str,
                         original: dict, data: dict) -> dict:
    """Check the independent source packet against every original UV/PTS."""
    require(len(expected_sha) == 64, "A reviewed 05/08 packet SHA-256 is required")
    name = "board_top" if edge_id == 0 else "contact"
    path = ADAPTIVE_FULL / f"shot{number:02d}_{name}_support.npz"
    require(driver.digest(path) == expected_sha,
            f"Frozen shot{number:02d} {name} support packet SHA changed")
    with np.load(path, allow_pickle=False) as packet:
        pts = np.asarray(packet["pts"], np.int64)
        offsets = np.asarray(packet["offsets"], np.int64)
        uv = np.asarray(packet["uv"])
        require(np.array_equal(pts, original["pts"])
                and np.array_equal(offsets, original["offsets"])
                and uv.dtype == original["raw_uv"].dtype
                and uv.tobytes() == original["raw_uv"].tobytes()
                and np.array_equal(uv.astype(np.float64), original["uv"])
                and str(packet["source_sha256"]) == data["captured"]["source_hash"]
                and str(packet["source_edge_sha256"]) == EDGE_SHA256[number]
                and str(packet["adaptive_rule_sha256"]) == ADAPTIVE_RULE_SHA256,
                f"Shot{number:02d} {name} support changed original PTS/UV/source")
        flags = np.asarray(packet["eligible"], bool).copy()
        normals = np.asarray(packet["normal_uv"], np.float64).copy()
        bounds = np.asarray(packet["bounds_normal_px"], np.float64).copy()
        reasons = np.asarray(packet["reason"])
        mask_status = np.asarray(packet["mask_status"])
        decoded_hashes = np.asarray(packet["decoded_bgr_sha256"])
    count = len(original["uv"])
    require(flags.shape == (count,) and normals.shape == bounds.shape == (count, 2)
            and reasons.shape == (count,)
            and mask_status.shape == decoded_hashes.shape == (len(pts),)
            and np.all(mask_status == "measured_mask")
            and all(len(str(value)) == 64 for value in decoded_hashes),
            f"Shot{number:02d} {name} support shapes or mask status changed")
    SourceNormalIntervals(normals, bounds, flags)
    return dict(flags=flags, normals=normals, bounds=bounds,
                path=str(path.resolve()), sha256=expected_sha,
                eligible_points=int(flags.sum()), original_points=count)


def verify_adaptive_manifest(data: dict, edge_data: dict, packet_sha: dict,
                             expected_sha: str) -> dict:
    """Bind four reviewed packets to one frozen source-only full-scan report."""
    require(len(expected_sha) == 64 and driver.digest(ADAPTIVE_REPORT) == expected_sha,
            "A reviewed full 05/08 source-support report SHA-256 is required")
    report = json.loads(ADAPTIVE_REPORT.read_text(encoding="utf8"))
    require(report["schema"] == "rinkspace.low-end-source-edge-support-05-08.v1"
            and report["source_video"]["sha256"] == data["captured"]["source_hash"]
            and report["adaptive_rule_sha256"] == ADAPTIVE_RULE_SHA256
            and report["original_frames"] == 2100
            and report["original_uv"] == 188212
            and report["original_uv_changed"] is False
            and report["fit_started"] is False
            and report["final_holdout_read"] is False,
            "Full 05/08 source-support report provenance changed")
    shots = report["shots"]
    require(len(shots) == 2 and {int(row["shot"]) for row in shots} == {5, 8},
            "Full source-support report shot set changed")
    for row in shots:
        number = int(row["shot"])
        require(row["frames"] == EXPECTED_FRAMES[number]
                and row["original_edge_packet_sha256"] == EDGE_SHA256[number]
                and row["source_only"] is True
                and row["source_pixels_changed"] is False
                and set(row["packets"]) == {"board_top", "contact"},
                f"Shot{number:02d} source-only provenance changed")
        for edge_id, name in enumerate(("board_top", "contact")):
            entry = row["packets"][name]
            path = ADAPTIVE_FULL / f"shot{number:02d}_{name}_support.npz"
            require(Path(entry["path"]).resolve() == path.resolve()
                    and entry["sha256"] == packet_sha[(number, edge_id)]
                    and entry["frames"] == len(edge_data[(number, edge_id)]["pts"])
                    and entry["original_uv"] == len(edge_data[(number, edge_id)]["uv"]),
                    f"Shot{number:02d} {name} report/packet identity changed")
    return dict(path=str(ADAPTIVE_REPORT.resolve()), sha256=expected_sha,
                adaptive_rule_sha256=ADAPTIVE_RULE_SHA256)


def load_supports(data: dict, edge_data: dict, packet_sha: dict,
                  report_sha: str) -> tuple[dict, dict]:
    manifest = verify_adaptive_manifest(data, edge_data, packet_sha, report_sha)
    supports = {}
    for edge_id in range(2):
        original = edge_data[(3, edge_id)]
        if edge_id == 0:
            flags, normals, bounds, meta = shot03_top.load_support(
                SHOT03_TOP_REPORT, SHOT03_TOP_SHA256, original["path"],
                original["pts"], original["offsets"], original["uv"], data)
        else:
            flags, normals, bounds, meta = shot03.load_contact_support(
                SHOT03_CONTACT_REPORT, SHOT03_CONTACT_SHA256, original["path"],
                original["pts"], original["offsets"], original["uv"], data)
        supports[(3, edge_id)] = dict(flags=flags, normals=normals,
                                      bounds=bounds, **meta)
    for number in (5, 8):
        for edge_id in range(2):
            supports[(number, edge_id)] = load_adaptive_packet(
                number, edge_id, packet_sha[(number, edge_id)],
                edge_data[(number, edge_id)], data)
    return supports, manifest


def factor_indices(data: dict, edge_data: dict, shot_by_pts: dict) -> dict:
    by_stamp: dict[int, list] = {stamp: [] for stamp in shot_by_pts}
    for index, spec in enumerate(data["retained"]):
        if (spec["file"], spec["function"]) != EDGE_KIND:
            continue
        keys = spec["keys"]
        if len(keys) == 1 and keys[0] in by_stamp:
            by_stamp[keys[0]].append((index, spec))
    indices = {}
    for stamp, found in by_stamp.items():
        number = shot_by_pts[stamp]
        pos = int(np.searchsorted(edge_data[(number, 0)]["pts"], stamp))
        expected = [edge_id for edge_id in range(2) if
            edge_data[(number, edge_id)]["offsets"][pos+1] >
            edge_data[(number, edge_id)]["offsets"][pos]]
        require(len(found) == len(expected),
                f"Original nonempty board-edge factor count changed at {stamp}")
        for edge_id, (index, spec) in zip(expected, found):
            original = edge_data[(number, edge_id)]
            count = int(original["offsets"][pos+1]-original["offsets"][pos])
            require(spec["kind"] == "observation" and spec["keys"] == (stamp,)
                    and spec["kwargs"]["residual_dim"] == 2*count
                    and float(spec["kwargs"]["sigma"]) == EDGE_SIGMAS[edge_id]
                    and spec["kwargs"]["jac_fn"] is not None,
                    f"Original {EDGE_NAMES[edge_id]} factor order/shape changed at {stamp}")
            indices[(stamp, edge_id)] = index
    require(len(indices) == 4262 and len(set(indices.values())) == 4262,
            "Expected 4262 nonempty original source-edge factors")
    return indices


def changed_data(data: dict, edge_data: dict, shot_by_pts: dict,
                 indices: dict, supports: dict | None) -> dict:
    if supports is None:
        return data  # Disabled is exactly the original graph and callbacks.
    changed = {**data, "retained": data["retained"].copy()}
    worlds = (_closed_edge_ring(np.asarray(data["scene"]["board_surface"]["top_edge_xyz"], float)),
              _closed_edge_ring(np.asarray(data["scene"]["ice_boundary"]["points_xyz"], float)))
    quality = data["captured"]["source_quality"]
    for (stamp, edge_id), index in indices.items():
        number = shot_by_pts[stamp]
        original = edge_data[(number, edge_id)]
        pos = int(np.searchsorted(original["pts"], stamp))
        start, stop = map(int, original["offsets"][pos:pos+2])
        support = supports[(number, edge_id)]
        interval = SourceNormalIntervals(
            support["normals"][start:stop], support["bounds"][start:stop],
            support["flags"][start:stop])
        changed["retained"][index] = shot03.replacement(
            data["retained"][index], stamp, edge_id, original["uv"][start:stop],
            worlds[edge_id], data["captured"]["templates"][stamp],
            interval, quality)
    replaced = set(indices.values())
    require(len(data["retained"]) == len(changed["retained"])
            and all(changed["retained"][i] is data["retained"][i]
                    for i in range(len(data["retained"])) if i not in replaced),
            "A non-board factor or original prior target changed")
    return changed


def unknown_callback_checks(data: dict, changed: dict, initial: dict,
                            edge_data: dict, supports: dict, indices: dict) -> list[dict]:
    """Check one original unknown row block per shot/edge at the base state."""
    checks = []
    for number in (3, 5, 8):
        for edge_id in range(2):
            original, support = edge_data[(number, edge_id)], supports[(number, edge_id)]
            offsets = original["offsets"]
            unknown_counts = np.asarray([int(np.sum(~support["flags"][a:b]))
                for a, b in zip(offsets[:-1], offsets[1:])])
            if not np.any(unknown_counts):
                checks.append(dict(shot=number, edge=EDGE_NAMES[edge_id],
                                   unknown_points=0, status="all_source_qualified"))
                continue
            pos = int(np.argmax(unknown_counts))
            stamp = int(original["pts"][pos])
            start, stop = map(int, offsets[pos:pos+2])
            rows = np.repeat(~support["flags"][start:stop], 2)
            physical = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]
            index = indices[(stamp, edge_id)]
            old, new = data["retained"][index], changed["retained"][index]
            old_residual = np.asarray(old["args"][1]([physical]), np.float64)
            new_residual = np.asarray(new["args"][1]([physical]), np.float64)
            old_jac = np.asarray(old["kwargs"]["jac_fn"]([physical])[0], np.float64)
            new_jac = np.asarray(new["kwargs"]["jac_fn"]([physical])[0], np.float64)
            require(old_residual.shape == new_residual.shape == (2*(stop-start),)
                    and old_jac.shape == new_jac.shape == (2*(stop-start), 11),
                    f"Original/interval edge callback shape changed at {stamp}")
            error_r = float(np.max(np.abs(old_residual[rows]-new_residual[rows])))
            error_j = float(np.max(np.abs(old_jac[rows]-new_jac[rows])))
            require(error_r < 1e-9 and error_j < 1e-7,
                    f"Unknown source rows changed at {stamp}/{EDGE_NAMES[edge_id]}")
            checks.append(dict(shot=number, edge=EDGE_NAMES[edge_id], pts=stamp,
                original_unknown_points=int(unknown_counts[pos]),
                max_unknown_residual_difference=error_r,
                max_unknown_jacobian_difference=error_j))
    return checks


def representative_fd(data: dict, initial: dict, edge_data: dict,
                      supports: dict) -> list[dict]:
    """Check actual source-qualified pixels away from a piecewise boundary."""
    worlds = (_closed_edge_ring(np.asarray(data["scene"]["board_surface"]["top_edge_xyz"], float)),
              _closed_edge_ring(np.asarray(data["scene"]["ice_boundary"]["points_xyz"], float)))
    results = []
    for number in (3, 5, 8):
        for edge_id in range(2):
            original, support = edge_data[(number, edge_id)], supports[(number, edge_id)]
            pts, offsets = original["pts"], original["offsets"]
            qualified_count = np.asarray([int(np.sum(support["flags"][a:b]))
                for a, b in zip(offsets[:-1], offsets[1:])])
            require(np.any(qualified_count),
                    f"No actual qualified {EDGE_NAMES[edge_id]} in shot{number:02d}")
            chosen = None
            # Selection is for derivative QA only; it never changes a factor.
            for pos in np.argsort(-qualified_count, kind="stable")[:8]:
                if not qualified_count[pos]:
                    continue
                stamp = int(pts[pos])
                start, stop = map(int, offsets[pos:pos+2])
                observed = original["uv"][start:stop]
                interval = SourceNormalIntervals(
                    support["normals"][start:stop], support["bounds"][start:stop],
                    support["flags"][start:stop])
                physical = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]
                template = data["captured"]["templates"][stamp]

                def native(state, jac=False):
                    camera = from_state(state, template)
                    selected = _edge_segment_candidates(camera, worlds[edge_id], observed)
                    return _edge_curve_error(camera, worlds[edge_id], observed, selected,
                                             with_jacobian=jac, finite_vector=True)

                raw, jacobian = native(physical, True)
                normal = support["normals"][start:stop]
                bounds = support["bounds"][start:stop]
                candidates = []
                for point in np.flatnonzero(interval.eligible):
                    value = float(np.dot(raw.reshape(-1, 2)[point], normal[point]))
                    lo, hi = bounds[point]
                    margin = (min(value-lo, hi-value) if lo < value < hi else
                              max(lo-value, value-hi))
                    if margin > .2:
                        candidates.append((margin, int(point), value))
                if candidates:
                    margin, point, signed = max(candidates)
                    chosen = (stamp, observed, physical, interval, native,
                              raw, jacobian, point, signed, margin)
                    break
            require(chosen is not None,
                    f"No stable actual qualified FD point in shot{number:02d}/{edge_id}")
            stamp, observed, physical, interval, native, raw, jacobian, point, signed, margin = chosen
            transformed, mapped = interval.apply(raw, [jacobian])
            h = 1e-7
            largest_abs = largest_rel = 0.
            for column in range(11):
                hi, lo = physical.copy(), physical.copy()
                hi[column] += h
                lo[column] -= h
                numeric = (interval.apply(native(hi))-interval.apply(native(lo)))/(2*h)
                rows = slice(2*point, 2*point+2)
                difference = np.abs(numeric[rows]-mapped[0][rows, column])
                largest_abs = max(largest_abs, float(difference.max()))
                largest_rel = max(largest_rel, float(np.max(difference/
                    (1.+np.abs(mapped[0][rows, column])))))
            result = dict(shot=number, edge=EDGE_NAMES[edge_id], pts=stamp,
                original_uv=observed[point].tolist(), signed_raw_normal_px=signed,
                bounds_normal_px=interval.bounds[point].tolist(),
                margin_to_interval_boundary_px=margin, h=h,
                physical_camera_columns=list(range(11)),
                shared_C_columns_in_original_derived_graph=[0, 1, 2],
                max_absolute_error=largest_abs,
                max_relative_error=largest_rel,
                selected_for_derivative_QA_only=True)
            require(largest_abs < .01 and largest_rel < 1e-5,
                    f"Actual source-normal derivative differs: {result}")
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("disabled", "interval"), default="disabled")
    parser.add_argument("--source-report-sha256")
    for number in (5, 8):
        for label in ("top", "contact"):
            parser.add_argument(f"--shot{number}-{label}-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Diagnostic output must remain in project tracking TMP")
    require(not args.run or args.mode == "interval",
            "Only the source-supported interval objective can be run")
    if args.run:
        require((args.output / "phase_00.json").is_file(),
                "Solve requires a reviewed matching interval preflight")
    else:
        require(not (args.output / "phase_00.json").exists(),
                "Preflight already exists; select a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    data, pairs, profile, _, base_summary = transfer.load_inputs(args.output)
    initial = profile.baseline  # All 2221 solved original 11D states, not the 4D embedding.
    require(set(initial) == set(data["active"]) and len(initial) == 2221,
            "Frozen original 11D initializer does not cover the full graph")
    edge_data, shot_by_pts = source_edges(data)
    indices = factor_indices(data, edge_data, shot_by_pts)
    no_contact_frames = [int(stamp) for number in (3, 5, 8)
        for pos, stamp in enumerate(edge_data[(number, 1)]["pts"])
        if edge_data[(number, 1)]["offsets"][pos] ==
           edge_data[(number, 1)]["offsets"][pos+1]]
    require(len(no_contact_frames) == 176 and all(shot_by_pts[p] == 5
            for p in no_contact_frames),
            "Original 176 shot05 frames without a contact factor changed")
    packet_sha = {}
    for number in (5, 8):
        for edge_id, label in enumerate(("top", "contact")):
            value = getattr(args, f"shot{number}_{label}_sha256")
            require(args.mode == "disabled" or value is not None,
                    f"Reviewed shot{number} {label} source packet SHA is required")
            if value is not None:
                packet_sha[(number, edge_id)] = value.lower()
    if args.mode == "interval":
        require(args.source_report_sha256 is not None,
                "Reviewed full 05/08 source report SHA is required")
        supports, support_manifest = load_supports(
            data, edge_data, packet_sha, args.source_report_sha256.lower())
    else:
        supports, support_manifest = None, None
    changed = changed_data(data, edge_data, shot_by_pts, indices, supports)
    unknown_identity = (unknown_callback_checks(data, changed, initial, edge_data,
                        supports, indices) if supports else [])
    graph, graph_initial = driver.build_graph(changed, pairs, "fixed", "derived")
    require(len(graph._pending) == 11199 and set(graph_initial) == set(initial),
            "Original 11199-factor full low_end graph changed")
    values = gtsam.Values()
    for key, state in initial.items():
        values.insert(graph._key(key), state)
    initial_cost = float(sum(factor.error(values) for factor in graph._pending))
    require(np.isfinite(initial_cost), "Original-state objective is nonfinite")
    if args.mode == "disabled":
        require(changed is data and abs(initial_cost-
                float(base_summary["batch"]["final_nonlinear_cost"])) < 1e-5,
                "Disabled objective differs from the frozen original full solve")
    fd = representative_fd(data, initial, edge_data, supports) if supports else []
    phase = dict(schema="rinkspace.low-end-board-edge-interval-diagnostic.v1",
        mode=args.mode, scope="Full low_end shots03/05/08; original 11D fixed/derived shared mount and lens",
        source_sha256=data["captured"]["source_hash"],
        scene_sha256=data["diag"]["model_sha256"],
        original_factors=11199, replaceable_edge_factors=len(indices),
        untouched_factors=11199-len(indices), native_pairs=len(pairs),
        frames_without_original_contact_factor=no_contact_frames,
        native_pixels=104831, active_original_states=len(initial),
        disabled_is_original_objective=args.mode == "disabled",
        prior_targets_and_original_quality_sigma_huber_unchanged=True,
        unknown_rows_pass_through_original_residual=True,
        no_qualified_only_exclusion=True, original_uv_changed=False,
        initial_state_source="conditioned_low_end_source_support_v2/fixed/states.npz",
        initial_state_sha256=transfer.BASE_STATES_SHA256,
        initial_cost_at_base_states=initial_cost,
        support={f"shot{number}_{EDGE_NAMES[edge_id]}": (
            {"original_points": len(edge_data[(number, edge_id)]["uv"]),
             "eligible_points": int(supports[(number, edge_id)]["flags"].sum()),
             "sha256": supports[(number, edge_id)].get("sha256",
                        supports[(number, edge_id)].get("report_sha256"))}
            if supports else None)
            for number in (3, 5, 8) for edge_id in range(2)},
        support_manifest=support_manifest,
        representative_unknown_callback_equivalence=unknown_identity,
        actual_source_normal_fd=fd,
        input_sha256=dict(config=driver.digest(transfer.seed_driver.CONFIG),
            candidate_scene=driver.digest(transfer.seed_driver.CANDIDATE/"candidate_scene.json"),
            manifest=driver.digest(transfer.seed_driver.MANIFEST),
            base_summary=transfer.BASE_SUMMARY_SHA256,
            base_states=transfer.BASE_STATES_SHA256,
            driver=driver.digest(driver.__file__),
            transfer_loader=driver.digest(transfer.__file__),
            replacement=driver.digest(shot03.__file__),
            interval_core=driver.digest(Path(__file__).with_name("source_edge_intervals.py")),
            script=driver.digest(__file__)),
        solver_started=False, final_holdout_read=False,
        not_programme_acceptance=True)
    phase_path = args.output / "phase_00.json"
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Frozen interval preflight differs from code, source or objective")
    else:
        phase_path.write_text(json.dumps(phase, indent=2), encoding="utf8")
    print(json.dumps(dict(phase="preflight", path=str(phase_path),
        sha256=driver.digest(phase_path), mode=args.mode,
        edge_factors=len(indices), untouched_factors=11199-len(indices),
        initial_cost=initial_cost)), flush=True)
    if args.run:
        target = args.output / "fixed"
        require(not target.exists(), "Existing full diagnostic solve must be reviewed")
        provenance = dict(initial_states_sha256=transfer.BASE_STATES_SHA256,
            interval_phase_sha256=driver.digest(phase_path),
            source_support=phase["support"],
            all_original_2219_frame_11d_initial_values=True,
            no_qualified_only_exclusion=True)
        driver.solve_one(changed, pairs, "fixed", target, "derived",
            initial_override=initial, initial_provenance=provenance)


if __name__ == "__main__":
    main()
