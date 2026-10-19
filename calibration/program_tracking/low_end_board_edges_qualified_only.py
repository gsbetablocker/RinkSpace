"""One qualified-only source-edge ablation over the frozen full fixed solution.

The 2219-frame fixed/derived graph, scene, source UV, original priors, native
motion, and all non-edge observations remain unchanged. Qualified board-top
and contact pixels retain their frozen source-normal intervals; unknown pixels
contribute zero residual and Jacobian rows before quality and Huber. This is a
conditional sensitivity test, not a new source measurement or acceptance test.
The default invocation only writes a checked preflight. ``--run`` requires that
exact preflight and starts the one full-graph solve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
from continuous_central_edge_optin import note_interval_replacement
import low_end_board_edges_interval_diagnostic as frozen
import low_end_board_edges_interval_pivot as fixed_result_loader
import low_end_calibration_transfer_profile as transfer
import shot03_board_edges_interval_diagnostic as shot03
from rig_parameterization import physical_frame
from source_edge_intervals import SourceNormalIntervals
from track_sequence import _closed_edge_ring


TMP = transfer.TMP
OUTPUT = TMP / "low_end_board_edges_qualified_only_v1"
FIXED = TMP / "low_end_board_edges_interval_diagnostic_v1/fixed"
EDGE_NAMES = frozen.EDGE_NAMES


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def qualified_data(data: dict, interval_data: dict, edge_data: dict,
                   shot_by_pts: dict, indices: dict, supports: dict) -> dict:
    """Replace exactly the original edge specs, using their original UVs."""
    changed = {**interval_data, "retained": interval_data["retained"].copy()}
    worlds = (
        _closed_edge_ring(np.asarray(data["scene"]["board_surface"]["top_edge_xyz"], float)),
        _closed_edge_ring(np.asarray(data["scene"]["ice_boundary"]["points_xyz"], float)),
    )
    quality = data["captured"]["source_quality"]
    for (stamp, edge_id), index in indices.items():
        number = shot_by_pts[stamp]
        original = edge_data[(number, edge_id)]
        pos = int(np.searchsorted(original["pts"], stamp))
        start, stop = map(int, original["offsets"][pos:pos+2])
        support = supports[(number, edge_id)]
        intervals = SourceNormalIntervals(
            support["normals"][start:stop], support["bounds"][start:stop],
            support["flags"][start:stop])
        replacement = shot03.replacement(
            data["retained"][index], stamp, edge_id,
            original["uv"][start:stop], worlds[edge_id],
            data["captured"]["templates"][stamp], intervals, quality,
            qualified_only=True)
        # The interval and unknown-row policy changes the complete residual.
        # Record its native exclusion explicitly; the original edge guard
        # cannot safely be rebound to this replacement callback.
        note_interval_replacement(
            data["retained"][index], replacement,
            observed=original["uv"][start:stop], eligible=intervals.eligible,
            edge_name=EDGE_NAMES[edge_id], qualified_only=True)
        changed["retained"][index] = replacement
    replaced = set(indices.values())
    require(len(replaced) == 4262 and len(changed["retained"]) == len(data["retained"])
            == len(interval_data["retained"]), "Original graph edge count changed")
    require(all(changed["retained"][i] is interval_data["retained"][i]
                is data["retained"][i] for i in range(len(data["retained"]))
                if i not in replaced),
            "A non-edge factor spec, prior target, or native motion changed")
    return changed


def edge_callback_audit(interval_data: dict, qualified: dict, initial: dict,
                        data: dict, edge_data: dict, shot_by_pts: dict,
                        indices: dict, supports: dict) -> dict:
    """Evaluate every changed factor at the saved fixed solution."""
    maxima = dict(eligible_residual=0., eligible_jacobian=0.,
                  unknown_residual=0., unknown_jacobian=0.)
    checked = eligible_rows = unknown_rows = 0
    for (stamp, edge_id), index in indices.items():
        number = shot_by_pts[stamp]
        original = edge_data[(number, edge_id)]
        pos = int(np.searchsorted(original["pts"], stamp))
        start, stop = map(int, original["offsets"][pos:pos+2])
        flag = supports[(number, edge_id)]["flags"][start:stop]
        physical = physical_frame(initial[stamp], initial[data["mount"]], "fixed")[0]
        q = [physical]
        reference, altered = interval_data["retained"][index], qualified["retained"][index]
        r0 = np.asarray(reference["args"][1](q), np.float64)
        r1 = np.asarray(altered["args"][1](q), np.float64)
        j0 = np.asarray(reference["kwargs"]["jac_fn"](q)[0], np.float64)
        j1 = np.asarray(altered["kwargs"]["jac_fn"](q)[0], np.float64)
        expected = (2*(stop-start),)
        require(r0.shape == r1.shape == expected
                and j0.shape == j1.shape == (expected[0], 11)
                and np.isfinite(r0).all() and np.isfinite(r1).all()
                and np.isfinite(j0).all() and np.isfinite(j1).all(),
                f"Bad edge callback at {stamp}/{EDGE_NAMES[edge_id]}")
        yes, no = np.repeat(flag, 2), np.repeat(~flag, 2)
        if yes.any():
            maxima["eligible_residual"] = max(maxima["eligible_residual"],
                float(np.max(np.abs(r1[yes]-r0[yes]))))
            maxima["eligible_jacobian"] = max(maxima["eligible_jacobian"],
                float(np.max(np.abs(j1[yes]-j0[yes]))))
        if no.any():
            maxima["unknown_residual"] = max(maxima["unknown_residual"],
                float(np.max(np.abs(r1[no]))))
            maxima["unknown_jacobian"] = max(maxima["unknown_jacobian"],
                float(np.max(np.abs(j1[no]))))
        eligible_rows += int(yes.sum())
        unknown_rows += int(no.sum())
        checked += 1
    require(all(value == 0. for value in maxima.values()),
            f"Qualified-only residual/J row policy failed: {maxima}")
    return dict(factors_checked=checked, eligible_pixel_rows=eligible_rows,
                unknown_pixel_rows=unknown_rows, max_absolute_error=maxima,
                checked_at="all 2221 solved fixed/interval states")


def source_coverage(edge_data: dict, supports: dict) -> dict:
    """Record lost spatial evidence without treating unknown UVs as wrong."""
    report = {}
    for number in (3, 5, 8):
        for edge_id in range(2):
            original = edge_data[(number, edge_id)]
            flags = supports[(number, edge_id)]["flags"]
            offsets = original["offsets"]
            with_uv = zero_with_uv = four_quartiles = 0
            quartile_good = np.zeros(4, np.int64)
            quartile_total = np.zeros(4, np.int64)
            for pos in range(len(original["pts"])):
                a, b = map(int, offsets[pos:pos+2])
                if a == b:
                    continue
                with_uv += 1
                if not flags[a:b].any():
                    zero_with_uv += 1
                order = np.argsort(original["uv"][a:b, 0], kind="stable")
                slices = np.array_split(flags[a:b][order], 4)
                four_quartiles += all(part.any() for part in slices)
                for q, part in enumerate(slices):
                    quartile_good[q] += int(part.sum())
                    quartile_total[q] += len(part)
            report[f"shot{number}_{EDGE_NAMES[edge_id]}"] = dict(
                frames=len(original["pts"]), frames_with_original_uv=with_uv,
                frames_with_uv_but_no_qualified=zero_with_uv,
                frames_qualified_in_all_four_x_rank_quartiles=int(four_quartiles),
                original_uv=len(flags), qualified_uv=int(flags.sum()),
                unknown_uv=int((~flags).sum()),
                qualified_by_x_rank_quartile=quartile_good.tolist(),
                original_by_x_rank_quartile=quartile_total.tolist())
    require(report["shot5_board_top"]["frames_with_uv_but_no_qualified"] == 0
            and report["shot8_board_top"]["frames_with_uv_but_no_qualified"] == 0
            and report["shot5_ice_board_contact"]["frames_with_uv_but_no_qualified"] == 30
            and report["shot8_ice_board_contact"]["frames_with_uv_but_no_qualified"] == 38,
            "Frozen qualified-source coverage changed")
    return report


def factor_cost_audit(interval_data: dict, qualified: dict, pairs: list,
                      initial: dict, indices: dict, fixed_summary: dict) -> dict:
    original_graph, old_initial = driver.build_graph(interval_data, pairs, "fixed", "derived")
    changed_graph, new_initial = driver.build_graph(qualified, pairs, "fixed", "derived")
    require(len(original_graph._pending) == len(changed_graph._pending) == 11199
            and set(old_initial) == set(new_initial) == set(initial),
            "Original 11199-factor and 2221-state graph changed")
    values = gtsam.Values()
    for key, state in initial.items():
        values.insert(original_graph._key(key), state)
    replaced = set(indices.values())
    old_edge = new_edge = old_other = new_other = max_other = 0.
    for i, (a, b) in enumerate(zip(original_graph._pending, changed_graph._pending)):
        old_cost, new_cost = float(a.error(values)), float(b.error(values))
        require(np.isfinite(old_cost) and np.isfinite(new_cost),
                f"Nonfinite factor cost at index {i}")
        if i in replaced:
            old_edge += old_cost
            new_edge += new_cost
        else:
            old_other += old_cost
            new_other += new_cost
            max_other = max(max_other, abs(old_cost-new_cost))
    original_total = old_edge+old_other
    require(max_other < 1e-9 and abs(old_other-new_other) < 1e-7,
            "A non-edge factor cost changed")
    require(abs(original_total-float(fixed_summary["batch"]["final_nonlinear_cost"])) < 1e-4,
            "Rebuilt interval graph does not match frozen fixed final cost")
    return dict(factors=11199, edge_factors=len(replaced), untouched_factors=11199-len(replaced),
                interval_edge_cost=old_edge, qualified_only_edge_cost=new_edge,
                interval_untouched_cost=old_other, qualified_only_untouched_cost=new_other,
                max_untouched_factor_cost_difference=max_other,
                interval_total_cost=original_total,
                qualified_only_total_cost=new_edge+new_other,
                cost_evaluated_at="same frozen full fixed/interval 2221-state solution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Output must stay within the programme tracking TMP")
    phase_path = args.output / "phase_00.json"
    if args.run:
        require(phase_path.is_file(), "Solve requires its frozen qualified-only preflight")
        require(not (args.output / "fixed").exists(), "This ablation already has a result")
    else:
        require(not phase_path.exists(), "Preflight already exists; do not overwrite it")
    args.output.mkdir(parents=True, exist_ok=True)
    data, pairs, profile, _, _ = transfer.load_inputs(args.output)
    fixed_phase, fixed_summary, initial = fixed_result_loader.fixed_result(data)
    require(set(initial) == set(data["active"]) and len(initial) == 2221
            and len(pairs) == 2216, "Frozen full fixed state or native pairs changed")
    edge_data, shot_by_pts = frozen.source_edges(data)
    indices = frozen.factor_indices(data, edge_data, shot_by_pts)
    packet_sha = {(number, edge_id): fixed_phase["support"][
        f"shot{number}_{EDGE_NAMES[edge_id]}"]["sha256"]
        for number in (5, 8) for edge_id in range(2)}
    supports, support_manifest = frozen.load_supports(data, edge_data,
        packet_sha, fixed_phase["support_manifest"]["sha256"])
    require(support_manifest == fixed_phase["support_manifest"],
            "A frozen 05/08 source-support manifest changed")
    interval_data = frozen.changed_data(data, edge_data, shot_by_pts, indices, supports)
    qualified = qualified_data(data, interval_data, edge_data,
                               shot_by_pts, indices, supports)
    callbacks = edge_callback_audit(interval_data, qualified, initial,
                                    data, edge_data, shot_by_pts, indices, supports)
    coverage = source_coverage(edge_data, supports)
    costs = factor_cost_audit(interval_data, qualified, pairs, initial, indices, fixed_summary)
    phase = dict(schema="rinkspace.low-end-board-edge-qualified-only-ablation.v1",
        mode="qualified_only_ablation", shot_indices=[3, 5, 8], frames=2219,
        states=2221, source_sha256=fixed_phase["source_sha256"],
        scene_sha256=fixed_phase["scene_sha256"],
        source_support=fixed_phase["support"],
        source_support_manifest=support_manifest,
        original_uv_changed=False, qualified_intervals_unchanged=True,
        unknown_data_rows_zeroed_before_original_quality_sigma_huber=True,
        prior_targets_and_original_quality_sigma_huber_unchanged=True,
        native_pairs=2216, native_pixels=104831,
        callback_audit=callbacks, factor_cost_audit=costs,
        source_coverage=coverage,
        qualification_risk=(
            "This ablation removes exact point likelihood from source-unknown UV; it "
            "does not show that those original UV are wrong. Contact qualification "
            "is sparse and biased toward image-left, with conservative false negatives; "
            "full motion and other retained evidence do not guarantee absolute pose accuracy."),
        initial_state_source=str((FIXED / "states.npz").resolve()),
        input_sha256=dict(fixed_interval_phase=fixed_result_loader.FIXED_PHASE_SHA256,
            fixed_interval_summary=fixed_result_loader.FIXED_SUMMARY_SHA256,
            fixed_interval_states=fixed_result_loader.FIXED_STATES_SHA256,
            fixed_interval_cameras=fixed_result_loader.FIXED_CAMERAS_SHA256,
            original_interval_script=driver.digest(frozen.__file__),
            fixed_result_loader=driver.digest(fixed_result_loader.__file__),
            transfer_loader=driver.digest(transfer.__file__),
            driver=driver.digest(driver.__file__),
            replacement=driver.digest(shot03.__file__),
            interval_core=driver.digest(Path(__file__).with_name("source_edge_intervals.py")),
            script=driver.digest(__file__)),
        solver_started=False, final_holdout_read=False,
        not_programme_acceptance=True)
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Qualified-only preflight no longer matches code, source, or initializer")
    else:
        phase_path.write_text(json.dumps(phase, indent=2), encoding="utf8")
    print(json.dumps(dict(phase="preflight", path=str(phase_path),
        sha256=driver.digest(phase_path), interval_cost=costs["interval_total_cost"],
        qualified_only_cost=costs["qualified_only_total_cost"],
        edge_factors=len(indices), unchanged_factors=11199-len(indices))), flush=True)
    if args.run:
        provenance = dict(initial_state_source=phase["initial_state_source"],
            initial_state_sha256=fixed_result_loader.FIXED_STATES_SHA256,
            qualified_only_phase_sha256=driver.digest(phase_path),
            original_objective_unchanged=False,
            changed_likelihood="only source-unknown board-top/contact data rows zeroed",
            all_original_2219_frame_11d_initial_values=True)
        driver.solve_one(qualified, pairs, "fixed", args.output / "fixed", "derived",
            initial_override=initial, initial_provenance=provenance)


if __name__ == "__main__":
    main()
