"""Source-only original low motion/v2-board overlap for a complete cut.

Generalizes the frozen cut5 audit without changing its packets or implementation.
Only existing numeric source rows, clock aliases and source masks are read.
The original-LK mode inventories measurements for both-endpoint RGB review:
a second detector need not rediscover an already measured corner. Neither
this inventory nor a colour-region navigation mask admits a material role.
"""
from pathlib import Path
import argparse
import collections
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
sys.dont_write_bytecode = True
from audit_original_low_motion_board_overlap_full_cut5 import (
    ADMISSION, DATA, ICE, PEOPLE, PREP, RETIREMENT, TRACKS,
    numeric, read, refs, sha, source_people_clear, source_UI_clear,
)
import numpy as np
from scipy.spatial import cKDTree


ORIGINAL_LK_MODE = "original_lk_both_RGB_review"
INVENTORY_REASONS = {
    64: "conservative_people_navigation_mask; actual_occlusion_review_needed",
    512: "outside_finite_original_native_RGB",
    1024: "fixed_broadcast_UI_navigation_mask; actual_material_review_needed",
}
REGION_NOT_PREVIOUSLY_MEASURED = 2048


def original_candidate_records(packet):
    """Iterate unchanged review records in the legacy or compact inventory."""
    reference = packet.get("candidate_numeric_archive")
    if reference is None:
        yield from packet["records"]
        return
    if packet.get("candidate_mode") != ORIGINAL_LK_MODE:
        raise ValueError("A compact archive needs the explicit original-LK inventory contract")
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Original LK numeric review inventory changed")
    with np.load(reference["path"], allow_pickle=False) as source:
        if list(source.files) != packet["candidate_numeric_archive_keys"]:
            raise ValueError("Original LK inventory numeric keys changed")
        count = int(packet["summary"]["source_candidate_count"])
        if any(len(source[key]) != count for key in source.files):
            raise ValueError("Original LK inventory arrays are not aligned")
        # NPZ members are loaded once, not decompressed per point.
        values = {key: source[key] for key in source.files}
    for index in range(count):
        yield dict(shot_index=packet["shot_index"],
            **{key: np.asarray(value[index]).tolist() for key, value in values.items()},
            role_status="original_unknown_depth_FIT; both_native_RGB_material_inspection_pending",
            both_end_material_reviewed=False, automatically_material_qualified=False,
            material_role_not_inferred_from_navigation_or_model_residual=True)


def original_lk_review_inventory_reasons(source_uv, target_uv, *,
        source_people_clear, target_people_clear, source_UI_clear,
        target_UI_clear, native_hw=(1080, 1920)):
    """Keep an existing LK measurement separate from material qualification.

    Returned [N,2] reasons only organize original-image review. A person box,
    missing cap/foot navigator, or another detector's missing corner cannot
    revoke an actual both-endpoint material review. Conversely, zero reasons
    are not a material certificate: an unmasked background/equipment point
    remains unknown. No UV, sigma, role, weight or correspondence is changed.
    """
    first, last = np.asarray(source_uv), np.asarray(target_uv)
    if first.ndim != 2 or first.shape[1:] != (2,) or last.shape != first.shape:
        raise ValueError("Matching original source/target [N,2] UV required")
    height, width = map(int, native_hw)
    if min(height, width) < 2:
        raise ValueError("Positive original native image dimensions required")
    result = np.zeros((len(first), 2), np.uint16)
    for endpoint, uv, person, ui in ((0, first, source_people_clear, source_UI_clear),
                                    (1, last, target_people_clear, target_UI_clear)):
        person, ui = np.asarray(person), np.asarray(ui)
        if (person.shape != (len(first),) or ui.shape != (len(first),)
                or person.dtype != np.bool_ or ui.dtype != np.bool_):
            raise ValueError("Explicit Boolean navigation masks aligned to original UV required")
        native = (np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        result[~native, endpoint] |= 512
        result[~person, endpoint] |= 64
        result[~ui, endpoint] |= 1024
    return result


def admitted_low_indices(path):
    packet = read(path)
    if "source_packets" in packet:
        result = set()
        for ref in packet["source_packets"]:
            assert sha(ref["path"]) == ref["sha256"]
            result |= admitted_low_indices(ref["path"])
        return result
    if "original-low-" not in packet["schema"]:
        return set()
    return {int(r.get("current_original_prefix_bundle_index", r.get("original_global_bundle_index")))
            for r in packet.get("records", packet.get("admitted_source_only_rows", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, choices=(3, 5, 8), required=True)
    ap.add_argument("--clock-state", type=Path, required=True)
    ap.add_argument("--existing-role-packet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--representative-bin-seconds", type=float, default=2.0)
    ap.add_argument("--candidate-mode", choices=("legacy_independent_detector_overlap", ORIGINAL_LK_MODE),
                    default="legacy_independent_detector_overlap",
                    help="Original LK mode retains both-endpoint review inventory without detector rediscovery")
    ap.add_argument("--compact-source-records", action="store_true",
                    help="Store unchanged numeric review inventory once in NPZ instead of repeated JSON UV rows")
    ap.add_argument("--existing-source-region-report", type=Path,
                    help="Keep previously authenticated navigation reason bits; unavailable rows remain unknown")
    a = ap.parse_args()
    if a.compact_source_records and a.candidate_mode != ORIGINAL_LK_MODE:
        raise ValueError("Compact inventory is explicit original-LK mode only; frozen legacy format remains unchanged")
    assert np.isfinite(a.representative_bin_seconds) and a.representative_bin_seconds > 0
    adapter_path = PREP / "low_source/low_source_factory/original_source_adapter_inputs/input_manifest.json"
    adapter = read(adapter_path)
    motionrefs = adapter["source_motion"]
    chosen = next(r for r in motionrefs if r["cut"] == a.shot)
    offset = sum(int(r["matches"]) for r in motionrefs if r["cut"] < a.shot)
    selected_path = DATA / "history/work_sessions/20261002_01a0fa4f-6592-7ab2-bc33-42ff983aff59/z190_low_source_authority_rf_v1" / f"low{a.shot}_retained_original_source_rows.npz"
    raw_path = DATA / "programmes/okada_mei_sp/history/tracking_runs/motion" / f"motion_shot_{a.shot:02d}.npz"
    assert sha(selected_path) == chosen["selected_sha256"] and sha(raw_path) == chosen["rawsha256"]
    selected = numeric(selected_path, ("original_global_row_indices", "native_pair_pts"))
    raw = numeric(raw_path, ("pair_pts", "offsets", "xy0", "xy1"))
    admission = numeric(ADMISSION, ("active_original_global_index", "source_uv_a", "source_uv_b", "source_endpoint_index_a", "source_endpoint_index_b", "original_sigma_xy"))
    ids = selected["original_global_row_indices"]
    assert len(ids) == chosen["matches"]
    full = offset + np.arange(len(ids))
    active = admission["active_original_global_index"]
    remap = np.full(len(admission["source_uv_a"]), -1, np.int64)
    remap[active] = np.arange(len(active))
    assert len(active) == 434991
    assert np.array_equal(admission["source_uv_a"][full], raw["xy0"][ids])
    assert np.array_equal(admission["source_uv_b"][full], raw["xy1"][ids])
    assert np.array_equal(selected["native_pair_pts"], raw["pair_pts"])
    rawpair = np.repeat(np.arange(len(raw["pair_pts"])), np.diff(raw["offsets"]))[ids]
    retirement = read(RETIREMENT)
    assert sha(ADMISSION) == retirement["original_foreground_admission_npz"]["sha256"]
    aliasref = retirement["source_actual_exposure_alias_authority"]
    assert sha(aliasref["path"]) == aliasref["sha256"]
    az = numeric(aliasref["path"], ("pts", "exposure_representative_pts"))
    aliases = dict(zip(map(int, az["pts"]), map(int, az["exposure_representative_pts"])))
    saved = numeric(a.clock_state, ("aux__block_names", "aux__block_lengths", "aux__canonical_pts", "aux__qualified_original_motion_ice_contract_json"))
    lengths = saved["aux__block_lengths"] // 4
    bi = list(saved["aux__block_names"]).index("low")
    lo = int(lengths[:bi].sum())
    stamps = saved["aux__canonical_pts"][lo:lo + int(lengths[bi])]
    assert len(stamps) == 1851
    aa = stamps[admission["source_endpoint_index_a"][full]]
    bb = stamps[admission["source_endpoint_index_b"][full]]
    assert np.array_equal(aa, [aliases[int(raw["pair_pts"][j, 0])] for j in rawpair])
    assert np.array_equal(bb, [aliases[int(raw["pair_pts"][j, 1])] for j in rawpair])
    typed_path = PREP / "low_source/low_source_factory/finite_material_source_binding.json"
    typed_records = read(typed_path)
    typed = {int(r["global_bundle_index"]) for r in typed_records}
    original_ice3 = set(map(int, json.loads(str(saved["aux__qualified_original_motion_ice_contract_json"].item()))["global_bundle_indices"]))
    existing = admitted_low_indices(a.existing_role_packet)
    retired = {int(r["current_bundle_motion_row"]) for r in retirement["records"]}
    unavailable = typed | original_ice3 | existing | retired
    # ICE is explicitly the cut5 raw-index ledger. Do not apply its raw integers
    # to a different cut, where the same integer identifies a different point.
    known_ice_raw = {int(r["original_global_index"]) for r in read(ICE)["qualified_rows"] if r["split"] == "fit"} if a.shot == 5 else set()
    original_lk = a.candidate_mode == ORIGINAL_LK_MODE
    tracks = None
    track_receipt_path = TRACKS.parent / "source_extraction_receipt.json"
    lookup = {}
    if not original_lk:
        tracks = numeric(TRACKS, ("source_pts", "target_pts", "shot_index", "offsets", "uv0", "uv1", "forward_backward_error_px"))
        track_receipt = read(track_receipt_path)
        assert track_receipt["source_UV_selected_by_model"] is False and track_receipt["no_cross_cut_pairs"] is True
        assert track_receipt["source_board_navigation"] == "existing_native_RGB_cap_and_yellow_foot_strip"
        for ti in np.flatnonzero(tracks["shot_index"] == a.shot):
            key = (int(tracks["source_pts"][ti]), int(tracks["target_pts"][ti]))
            assert key not in lookup
            lookup[key] = int(ti)
    people = {int(r["pts"]): r["boxes"] for r in read(PEOPLE)["frames"]}
    prior_region, prior_region_arrays, prior_region_reason_by_index = None, None, {}
    if a.existing_source_region_report:
        if not original_lk:
            raise ValueError("Prior region evidence is explicit original-LK mode only")
        prior_region = read(a.existing_source_region_report)
        if prior_region["source_video_sha256_from_existing_verified_ledger"] != retirement["source_video_sha256"]:
            raise ValueError("Prior region evidence is from a different original video")
        old_candidates = prior_region["original_candidates"]
        if sha(old_candidates["path"]) != old_candidates["sha256"]:
            raise ValueError("Preserved region candidate source changed")
        old_candidate = read(old_candidates["path"])
        if old_candidate["original_source_ledger"] != refs(ADMISSION):
            raise ValueError("Prior region indices belong to a different original source ledger")
        prior_region_arrays = prior_region["mask_archive"]
        if sha(prior_region_arrays["path"]) != prior_region_arrays["sha256"]:
            raise ValueError("Preserved source-region navigation arrays changed")
        prior_mask = numeric(prior_region_arrays["path"], ("original_source_index", "point_endpoint_reason_bits"))
        if len(np.unique(prior_mask["original_source_index"])) != len(prior_mask["original_source_index"]):
            raise ValueError("Prior navigation point indices are ambiguous")
        for row, bits in zip(prior_mask["original_source_index"], prior_mask["point_endpoint_reason_bits"]):
            prior_region_reason_by_index[int(row)] = bits
    records, groups, total = [], [], collections.Counter()
    compact = collections.defaultdict(list)
    start = int(min(aa.min(), bb.min()))
    for j, pair in enumerate(raw["pair_pts"]):
        pa, pb = (aliases[int(x)] for x in pair)
        rr = np.flatnonzero(rawpair == j)
        active_rows = rr[remap[full[rr]] >= 0]
        g = dict(shot_index=a.shot, original_raw_pair_index=j, source_native_pair_pts=list(map(int, pair)), source_pts=pa, target_pts=pb, selected_original_count=len(rr), active_original_count=len(active_rows), candidate_count=0)
        total["selected_original_count"] += len(rr)
        total["active_original_count"] += len(active_rows)
        if pa == pb:
            g["status"] = "same_actual_exposure_not_motion"
            groups.append(g)
            continue
        usable = np.asarray([int(ri) for ri in active_rows if int(remap[full[ri]]) not in unavailable and int(ids[ri]) not in known_ice_raw], np.int64)
        g["untyped_nonretired_candidate_scope_count"] = len(usable)
        total["untyped_nonretired_true_exposure_motion_count"] += len(usable)
        uv0, uv1 = raw["xy0"][ids[usable]], raw["xy1"][ids[usable]]
        clear0, clear1 = source_people_clear(uv0, pa, people), source_people_clear(uv1, pb, people)
        ui0, ui1 = source_UI_clear(uv0), source_UI_clear(uv1)
        review_reasons = original_lk_review_inventory_reasons(uv0, uv1,
            source_people_clear=clear0, target_people_clear=clear1,
            source_UI_clear=ui0, target_UI_clear=ui1)
        if original_lk:
            # Only impossible/native-invalid coordinates are absent from the
            # image-review inventory. People/UI navigation remains explicitly
            # unknown; it is not quietly admitted or erased from the ledger.
            keep = np.all((review_reasons & 512) == 0, axis=1)
            g.update(candidate_count=int(keep.sum()),
                navigation_clear_review_count=int(np.all(review_reasons[keep] == 0, axis=1).sum()),
                conservative_navigation_unknown_count=int(np.any(review_reasons[keep] != 0, axis=1).sum()),
                status="existing_LK_both_original_RGB_material_review_pending",
                material_qualified_count=0)
            total["native_invalid_original_points"] += int((~keep).sum())
            total["navigation_clear_review_count"] += g["navigation_clear_review_count"]
            total["conservative_navigation_unknown_count"] += g["conservative_navigation_unknown_count"]
            chosen = usable[keep]
            numeric_rows = dict(original_raw_point_index=ids[chosen],
                original_raw_pair_index=np.full(len(chosen), j, np.int64),
                original_raw_pair_local_index=ids[chosen] - int(raw["offsets"][j]),
                original_selected_row=chosen, original_full_low_bundle_row=full[chosen],
                current_original_prefix_bundle_index=remap[full[chosen]],
                source_pts=np.full(len(chosen), pa, np.int64), target_pts=np.full(len(chosen), pb, np.int64),
                source_native_pair_pts=np.tile(np.asarray(pair, np.int64), (len(chosen), 1)),
                source_uv=uv0[keep], target_uv=uv1[keep], source_sigma_px=admission["original_sigma_xy"][full[chosen]],
                source_navigation_reason_bits=review_reasons[keep])
            if prior_region_arrays is not None:
                prefix = numeric_rows["current_original_prefix_bundle_index"]
                numeric_rows["existing_source_region_reason_bits"] = np.asarray([
                    prior_region_reason_by_index.get(int(i), np.full(2, REGION_NOT_PREVIOUSLY_MEASURED, np.uint16))
                    for i in prefix], np.uint16).reshape((-1, 2))
                numeric_rows["existing_source_region_point_reasons_available"] = np.asarray([
                    int(i) in prior_region_reason_by_index for i in prefix], np.bool_)
            if a.compact_source_records:
                for key, value in numeric_rows.items(): compact[key].append(value)
            else:
                for k in range(len(chosen)):
                    records.append(dict(shot_index=a.shot,
                        **{key: np.asarray(value[k]).tolist() for key, value in numeric_rows.items()},
                        role_status="original_unknown_depth_FIT; both_native_RGB_material_inspection_pending",
                        both_end_material_reviewed=False, automatically_material_qualified=False,
                        material_role_not_inferred_from_navigation_or_model_residual=True))
            groups.append(g)
            continue
        ti = lookup.get((pa, pb))
        reverse = False
        if ti is None:
            ti, reverse = lookup.get((pb, pa)), True
        if ti is None:
            g["status"] = "independent_actual_pair_missing"
            groups.append(g)
            continue
        tl, th = map(int, tracks["offsets"][ti:ti + 2])
        ta, tb = tracks["uv0"][tl:th], tracks["uv1"][tl:th]
        if reverse:
            ta, tb = tb, ta
        g.update(independent_actual_pair_index=ti, independent_source_track_count=len(ta), independent_pair_reversed=reverse)
        if not len(ta) or not len(usable):
            g["status"] = "source_navigation_sparse_or_no_original_unknowns"
            groups.append(g)
            continue
        _, near = cKDTree(np.c_[ta, tb]).query(np.c_[uv0, uv1])
        ea, eb = np.linalg.norm(uv0 - ta[near], axis=1), np.linalg.norm(uv1 - tb[near], axis=1)
        overlap = (ea <= 2.) & (eb <= 2.)
        clear = clear0 & clear1
        ui = ui0 & ui1
        scope = (uv0[:, 1] < 800) & (uv1[:, 1] < 800)
        keep = overlap & clear & ui & scope
        g.update(double_endpoint_overlap_count=int(overlap.sum()), conservative_people_excluded=int((overlap & ~clear).sum()), fixed_UI_excluded=int((overlap & ~ui).sum()), source_pixel_scope_excluded=int((overlap & ~scope).sum()), candidate_count=int(keep.sum()), status="source_only_numeric_candidates_pending_actual_RGB")
        total["double_endpoint_overlap"] += int(overlap.sum())
        for ri, k in zip(usable[keep], np.flatnonzero(keep)):
            fi, rawi = int(full[ri]), int(ids[ri])
            records.append(dict(shot_index=a.shot, original_raw_point_index=rawi, original_raw_pair_index=j, original_raw_pair_local_index=rawi - int(raw["offsets"][j]), original_selected_row=int(ri), original_full_low_bundle_row=fi, current_original_prefix_bundle_index=int(remap[fi]), source_native_pair_pts=list(map(int, pair)), source_pts=pa, target_pts=pb, source_uv=uv0[k].tolist(), target_uv=uv1[k].tolist(), source_sigma_px=admission["original_sigma_xy"][fi].tolist(), deep_LK_endpoint_difference_px=[float(ea[k]), float(eb[k])], independent_track_index=int(tl + near[k]), independent_actual_pair_index=ti, independent_pair_reversed=reverse, independent_deep_LK_FB_native_px=float(tracks["forward_backward_error_px"][tl + near[k]]), role_status="original_unknown_depth_FIT; both_native_RGB_material_inspection_pending", both_end_material_reviewed=False, source_only_people_clear=True, source_fixed_UI_clear=True, material_role_not_inferred_from_model_residual=True))
        groups.append(g)
    archive_ref = None
    candidate_count = len(records)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.compact_source_records:
        packed = {key: np.concatenate(value, axis=0) for key, value in compact.items()}
        if not packed:
            raise ValueError("No original native LK review inventory to serialize")
        indices = packed["current_original_prefix_bundle_index"]
        unique, candidate_count = set(map(int, indices)), len(indices)
        path = a.output / f"original_low_cut{a.shot}_LK_both_RGB_review_inventory_v2.npz"
        np.savez_compressed(path, **packed)
        archive_ref = refs(path)
    else:
        unique = {r["current_original_prefix_bundle_index"] for r in records}
    assert len(unique) == candidate_count and not unique & unavailable
    width = int(round(a.representative_bin_seconds * 15360))
    bins, representatives = [], []
    for b in range((max(max(g["source_pts"], g["target_pts"]) for g in groups) - start) // width + 1):
        gs = [g for g in groups if (g["source_pts"] - start) // width == b]
        if not gs:
            continue
        bins.append(dict(bin_index=b, source_lower_pts=start + b * width, source_upper_pts=start + (b + 1) * width, candidate_points=sum(g["candidate_count"] for g in gs), candidate_pairs=sum(g["candidate_count"] > 0 for g in gs), original_pairs=len(gs)))
        available = [g for g in gs if g["candidate_count"]]
        if available:
            representatives.append(min(available, key=lambda g: (abs(g["source_pts"] - (start + (b + .5) * width)), g["original_raw_pair_index"])))
    selected_active = set(map(int, remap[full][remap[full] >= 0]))
    kindcounts = collections.Counter(r["surface_kind"] for r in typed_records if int(r["global_bundle_index"]) in selected_active)
    report = dict(schema="rinkspace.original-low-FIT-full-cut-source-only-board-overlap-candidate.v1", status="numeric_source_navigation_candidates_not_material_qualified_not_admitted", shot_index=a.shot, block="low", source_video_sha256=retirement["source_video_sha256"], time_base="1/15360", original_source_ledger=refs(ADMISSION), original_selected_rows=refs(selected_path), original_raw_motion=refs(raw_path), original_adapter_manifest=refs(adapter_path), full_original_cut_offset=offset, clock_alias_authority=aliasref, clock_state=dict(path=str(a.clock_state), read_arrays=list(saved), q_or_camera_values_read=False), independent_v2_tracks=None if original_lk else refs(TRACKS), independent_v2_source_receipt=None if original_lk else refs(track_receipt_path), people_boxes=refs(PEOPLE), static_UI_retirement=refs(RETIREMENT), original_typed_prefix_contract=refs(typed_path), existing_role_packet=refs(a.existing_role_packet), summary=dict(total, source_candidate_count=candidate_count, source_candidate_pairs=sum(g["candidate_count"] > 0 for g in groups), original_raw_pair_count=len(groups), same_actual_exposure_pairs=sum(g["status"] == "same_actual_exposure_not_motion" for g in groups), existing_typed_by_surface_kind=dict(kindcounts), existing_admitted_original_low_points_in_cut=len(existing & selected_active), original_retired_points_in_cut=len(retired & selected_active), source_first_actual_pts=start, source_last_actual_pts=max(max(g["source_pts"], g["target_pts"]) for g in groups)), temporal_bins=bins, all_original_pair_coverage=groups, records=records, representative_pair_navigation=representatives, original_UV_sigma_native_actual_PTS_and_prefix_raw_indices_bitexact=True, source_measurements_roles_weights_modified=False, new_source_rows=0, HELD_promoted=False, material_identity_not_established_by_pixel_overlap_or_mask=True, geometry_camera_residual_J_graph_solver_loaded=False, gates=dict(double_endpoint_limit_px=2., conservative_people_and_fixed_UI_gates_inherited_from_frozen_cut5_tool=True, native_y_less_than800_is_navigation_scope_not_physical_material_identity=True, missing_people_frame_excluded=True), verifier_script=refs(__file__))
    if original_lk:
        report.update(schema="rinkspace.original-LK-FIT-whole-cut-both-RGB-material-review-inventory.v2",
            status="original_measurements_for_material_review_not_qualified_not_admitted",
            candidate_mode=ORIGINAL_LK_MODE, candidate_numeric_archive=archive_ref,
            candidate_numeric_archive_keys=list(compact) if archive_ref else None,
            source_navigation_reason_bits=INVENTORY_REASONS,
            existing_source_region_report=refs(a.existing_source_region_report) if prior_region else None,
            existing_source_region_archive=prior_region_arrays,
            existing_source_region_reason_definitions=prior_region.get("reason_bits") if prior_region else None,
            existing_source_region_not_previously_measured_bit=REGION_NOT_PREVIOUSLY_MEASURED,
            existing_source_region_reason_bits_preserved_exactly=True,
            automatically_material_qualified_count=0, material_review_may_reject_or_keep_unknown=True,
            both_endpoint_actual_original_RGB_material_certificate_required=True,
            conservative_people_UI_or_missing_colour_navigation_not_material_certificates=True,
            independent_corner_redetection_not_required=True,
            gates=dict(native_RGB_only=True, repeated_detector_overlap_required=False,
                long_blue_cap_and_yellow_foot_required=False,
                people_and_UI_navigation_reasons_retained_without_automatic_role_admission=True,
                source_original_unknown_roles_preserved=True))
    suffix = "LK_both_RGB_review_inventory_v2" if original_lk else "full_source_overlap_candidates_v1"
    out = a.output / f"original_low_cut{a.shot}_{suffix}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(out), sha256=sha(out), summary=report["summary"], temporal_bins=bins)))


if __name__ == "__main__":
    main()
