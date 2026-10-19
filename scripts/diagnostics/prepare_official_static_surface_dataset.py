"""Prepare immutable official 50fps fit inputs; never launch an optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from calibration.program_tracking.official_static_surface_dataset import (
    build_dataset, iter_fit_rows, read_reference, reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--primary-source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scope-overrides", type=Path)
    parser.add_argument("--exclude-candidate-batch", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.manifest.exists() or args.selection.exists():
        raise ValueError("An immutable dataset output already exists")
    dataset_ref, dataset = build_dataset(args.model, args.primary_source, args.selection,
        excluded_candidate_batches=[reference(p) for p in args.exclude_candidate_batch],
        scope_overrides=reference(args.scope_overrides) if args.scope_overrides else None)
    points, rows, exposures = 0, 0, set()
    for row in iter_fit_rows(dataset_ref):
        if row["diagnostic_unit_weights"] or row["time_base"] != "1/12800":
            raise ValueError("Diagnostic or legacy-clock row entered FIT")
        points += len(row["uv"])
        rows += 1
        exposures.add(row["native_pts"])
    if points != dataset["counts"].get("fit", 0):
        raise ValueError("Dataset consumer did not preserve every admitted FIT point")
    model = read_reference(dataset["model"])
    old_input = read_reference(model["parent"]["input_manifest"])
    old_identity = read_reference(model["parent"]["state_identity"])
    navigation = []
    for batch in model["accepted_batch_datasets"]:
        manifest = read_reference({k: batch[k] for k in ("path", "sha256")})
        mapping = read_reference(manifest["navigation_mapping"])
        seed = mapping["model_seed"]["camera_seeds"]
        if reference(seed["path"]) != seed or mapping["model_seed"]["q_sha256"] != dataset["parent_q_sha256"]:
            raise ValueError("Official clock camera initializer differs from parent")
        navigation.append(dict(mapping=manifest["navigation_mapping"], camera_seeds=seed,
            match_summary=mapping["summary"], purpose="initialization_only_not_source_observation"))
    manifest = dict(schema=dataset["schema"], selection=dataset_ref,
        event_id="ankara_2026", programme_id="okada_mei_sp", prepared_date_local="2026-10-14",
        parent=model["parent"], source=read_reference(dataset["primary_source"])["primary_source"],
        model=dataset["model"], source_clock=dataset["source_clock"], navigation_initializers=navigation,
        admitted_frame_range=[min(r["index"] for r in dataset["frames"]), max(r["index"] for r in dataset["frames"])],
        frame_count=len(dataset["frames"]), first_native_pts=min(r["native_pts"] for r in dataset["frames"]),
        last_native_pts=max(r["native_pts"] for r in dataset["frames"]), counts=dataset["counts"],
        scope_counts=dataset["scope_counts"], target_counts=dataset["target_counts"],
        excluded_candidate_batches=dataset["excluded_candidate_batches"],
        old_source_inventory_excluded=old_identity["current_source_points"]["points"],
        old_motion_pairs_excluded=sum(old_input["original_motion_edge_counts"]),
        old_source_exclusion_reason="legacy_video_clock_and_unverified_official_source_material",
        old_pixels_classified_as_all_outside_scope=False, old_parameters_changed=False,
        original_files_modified=False, legacy_objectives_enabled=False, unknown_depth_points_admitted=0,
        ordinary_fit_consumer="calibration.program_tracking.official_static_surface_dataset.iter_fit_rows",
        other_evidence_consumer="calibration.program_tracking.official_static_surface_dataset.iter_observations",
        ordinary_fit_consumer_check=dict(rows=rows, points=points, actual_exposures=len(exposures),
            all_rows_built_by_existing_framewise_residual_interface=True, diagnostic_unit_weights_used=False),
        source_scales_are_original_localization_estimates_not_calibrated_noise=True,
        shared_line_and_derived_corner_weighting_installed=False,
        missing_source_scales_filled=False, full_programme_coverage=False,
        independent_HELD_created=False, accuracy_accepted=False, solver_started=False,
        full_solver_official_clock_adapter_installed=False,
        preparation_status="confirmed_scope_selection_and_ordinary_fit_rows_prepared")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(manifest=reference(args.manifest), frame_count=manifest["frame_count"],
        counts=manifest["counts"], scope_counts=manifest["scope_counts"],
        consumer=manifest["ordinary_fit_consumer_check"], solver_started=False)))


if __name__ == "__main__":
    main()
