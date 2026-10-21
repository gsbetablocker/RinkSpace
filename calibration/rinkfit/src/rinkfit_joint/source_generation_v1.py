"""Lossless source-generation overlay over the established packet normalizer.

Uses exact current batch refs and native clocks, preserving stable logical row
IDs, complete raw evidence, fine print/unbound assets, and manual provenance.
No raw packet or mask is copied, overwritten, or made solver-eligible here.
"""
from __future__ import annotations
import csv, json, math
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any
from .data_ingest import (_read_json, _sha, _jsonl_writer, _emit, _at, _kind,
                         _evidence_class, _mask_accessor, HISTORICAL_RIG_IDS)
SCHEMA='rinkfit.S01_S04.normalized_source_generation.v1'

class GenerationResolver:
    def __init__(self,root,generation):
        self.root=Path(root).resolve()
        self.manifest=_read_json(self.root/'manifest.json')
        self.aliases=self.manifest.get('aliases',{})
        self.entries={x['archive_path']:x for x in self.manifest['files']}
        self.generation=_read_json(Path(generation))
        self.overrides=self.generation['references'];self._hashes={}
    def canonical(self,logical):
        seen=set()
        while logical in self.aliases:
            if logical in seen:raise ValueError('Source alias cycle')
            seen.add(logical);logical=self.aliases[logical]
        if logical in self.entries:logical=self.entries[logical].get('canonical_path',logical)
        parts=PurePosixPath(logical)
        if parts.is_absolute() or '..' in parts.parts or '\\' in logical:raise ValueError('Unsafe logical source path')
        return logical
    def entry(self,logical):
        canonical=self.canonical(logical)
        return self.overrides.get(logical) or self.overrides.get(canonical) or self.entries.get(logical) or self.entries.get(canonical)
    def resolve(self,logical):
        entry=self.entry(logical)
        if entry:return Path(entry.get('path') or entry['original_path']).resolve()
        # Only the optional old prior-label inventory may be absent. Required
        # paths fail verify/load before use; never guess another source tree.
        return self.root/'_unbound_reference_'/self.canonical(logical)
    def load_json(self,logical):return _read_json(self.resolve(logical))
    def digest(self,logical):
        p=self.resolve(logical)
        if p not in self._hashes:self._hashes[p]=_sha(p)
        return self._hashes[p]
    def verify(self,logical):
        entry=self.entry(logical)
        if not entry:raise KeyError('No pinned source-generation entry: '+logical)
        p=self.resolve(logical);actual=self.digest(logical)
        if actual!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('Source-generation dependency changed: '+logical)
        return {'logical_path':logical,'canonical_path':self.canonical(logical),
                'path':str(p),'sha256':actual,'bytes':p.stat().st_size,
                'role':entry.get('role'),'new_generation_overlay':logical in self.overrides}

def normalize_source_generation(input_root: str | Path, output_dir: str | Path, *,
                      source_generation: str | Path, shot_ids=("s02","s03","s04"), verify_hashes: bool = True,
                      rig_by_shot: dict[str, str] | None = None) -> dict[str, Any]:
    """Normalize the authoritative frame inventory; preserve all 110350 rows.

    Returns summary and writes observations/groups/frames JSONL.gz, source
    receipts, findings, per-shot counts, and target-level semantic crosswalk.
    No row is made solver eligible.  A later source-quality/identity decision
    ledger must explicitly promote it without modifying its original UV.
    """
    root, output = Path(input_root).resolve(), Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Write normalized output outside the immutable input tree")
    if output.exists():
        raise ValueError("Source generation requires a new immutable normalized output")
    output.mkdir(parents=True, exist_ok=False)
    resolver = GenerationResolver(root, source_generation)
    rig_by_shot = dict(rig_by_shot or HISTORICAL_RIG_IDS)
    with (root / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
        inventory = [r for r in csv.DictReader(handle) if f"s{int(r['shot']):02d}" in set(shot_ids)]
        inventory = [{**r, **resolver.generation.get("inventory_overrides",{}).get(f"s{int(r['shot']):02d}:{int(r['programme_index'])}",{})} for r in inventory]
    issues: list[dict[str, Any]] = []
    counts, evidence_counts, flags_count = Counter(), Counter(), Counter()
    by_shot, operators = defaultdict(Counter), defaultdict(Counter)
    materials: dict[str, dict[str, Any]] = {}
    source_receipts, verified = [], set()
    seen_source_ids, seen_row_ids = {}, set()
    registry_cache = {}
    prior_labels = set()
    prior_path = "reference_s01/canonical_checkpoint/model.json"
    if resolver.resolve(prior_path).exists():
        prior_labels = {x["label"] for x in resolver.load_json(prior_path)["materials"]}

    def issue(code, frame=None, severity="review", **details):
        item = {"code": code, "severity": severity, **details}
        if frame:
            item.update(shot_id=f"s{int(frame['shot']):02d}",
                        programme_frame_index=int(frame["programme_index"]))
        issues.append(item)

    def receipt(logical):
        if logical and logical not in verified:
            if verify_hashes:
                source_receipts.append(resolver.verify(logical))
            verified.add(logical)

    with ExitStack() as stack:
        out_obs = _jsonl_writer(stack, output / "observations.jsonl.gz")
        out_groups = _jsonl_writer(stack, output / "groups.jsonl.gz")
        out_frames = _jsonl_writer(stack, output / "frames.jsonl.gz")
        for frame in inventory:
            shot = f"s{int(frame['shot']):02d}"
            rig_id = rig_by_shot[shot]
            frame_index = int(frame["programme_index"])
            native_pts = int(frame["native_pts"])
            tb = Fraction(frame["time_base"])
            if abs(float(native_pts * tb) - float(frame["source_seconds"])) > 1e-8:
                raise ValueError(f"Frame PTS/time mismatch: {shot}/{frame_index}")
            # The supplied event timeline uses 50 Hz source PTS; validate, do
            # not silently shift the full YouTube video to programme zero.
            expected_pts = 256256 + (frame_index - 1) * 256
            if native_pts != expected_pts:
                raise ValueError(f"Programme/native clock mismatch: {shot}/{frame_index}")
            paths = [x.strip() for x in frame["packets"].split("|") if x.strip()]
            if frame["stage"] == "assembled" and len(paths) != 1:
                raise ValueError(f"Assembled frame has multiple packet authorities: {shot}/{frame_index}")
            registry_path = f"{shot}/model/feature_registry.json"
            if not resolver.resolve(registry_path).exists():
                registry_path = "s08/model/feature_registry.json"
            if registry_path not in registry_cache:
                receipt(registry_path)
                registry_cache[registry_path] = resolver.load_json(registry_path)["targets"]
            registry = registry_cache[registry_path]
            binding_map = {}
            if frame["binding"]:
                receipt(frame["binding"])
                binding = resolver.load_json(frame["binding"])
                if (binding["source_pts"] != native_pts or
                        Fraction(binding["source_time_base"]) != tb):
                    raise ValueError(f"Binding clock mismatch: {shot}/{frame_index}")
                if binding["source_packet"]["sha256"] != resolver.digest(paths[0]):
                    raise ValueError(f"Binding uses stale packet: {shot}/{frame_index}")
                for binding_observation in binding["observations"]:
                    vertices = binding_observation["source_vertex_indices"]
                    source_ids = binding_observation["source_point_ids"]
                    if len(vertices) != len(source_ids):
                        raise ValueError("Binding source ID cardinality mismatch")
                    for vi, source_id in zip(vertices, source_ids):
                        key = (binding_observation["source_feature_id"],
                               binding_observation["source_segment_index"], vi)
                        if key in binding_map:
                            raise ValueError(f"Duplicate binding of source representative: {key}")
                        binding_map[key] = (binding_observation, source_id)
            frame_counts = Counter()
            mask, mask_status = None, "not_loaded"
            packet_stats = []
            frame_coordinates = {}
            binding_used = set()
            for logical in paths:
                receipt(logical)
                packet = resolver.load_json(logical)
                source = packet["source"]
                if (int(source["native_pts"]) != native_pts or
                        Fraction(source["time_base"]) != tb or
                        int(source["programme_frame_index_1based"]) != frame_index):
                    raise ValueError(f"Packet/frame clock mismatch: {logical}")
                native_hw = source.get("native_hw", [1080, 1920])
                if mask_status == "not_loaded":
                    receipt(frame.get("mask", ""))
                    mask, mask_status = _mask_accessor(resolver, frame.get("mask", ""), native_hw)
                    if mask_status not in ("checked_nearest_pixel_nonzero_exclusion",):
                        issue("mask_audit_unavailable", frame, status=mask_status)
                if source.get("mask_sha256") and frame.get("mask"):
                    if source["mask_sha256"] != resolver.digest(frame["mask"]):
                        raise ValueError(f"Stale packet mask: {logical}")
                manual = packet.get("manual_editor", {})
                if manual.get("authoritative_over_extraction_candidates") and "observations/frames/" not in logical:
                    raise ValueError("Manual authority must use merged packet")
                declarations = packet.get("source_correlated_groups", [])
                packet_named = packet_unbound = 0
                for collection in ("features", "unbound_static_materials"):
                    for fi, feature in enumerate(packet.get(collection, [])):
                        feature_id = feature.get("id", f"unnamed_feature_{fi}")
                        feature_ptr = f"/{collection}/{fi}"
                        feature_group_id = f"{shot}:{native_pts}:{collection}:{fi}:{Path(logical).parent.name}:feature"
                        original_target = feature.get("target_id")
                        measured_operator = feature.get("measurement_operator")
                        explicit_groups = [g for g in declarations if g.get("feature_id") == feature_id]
                        if any(s.get("uv") for s in feature.get("segments", [])):
                            # Family fitting evidence (including original two-side
                            # reads and old geometry ratio assumptions) is stored
                            # once per feature, never once per representative.
                            _emit(out_groups, {
                                "group_id": feature_group_id, "group_type": "feature_source_evidence",
                                "shot_id": shot, "rig_id": rig_id, "native_pts": native_pts,
                                "source_packet": logical, "feature_pointer": feature_ptr,
                                "source_feature_id": feature_id,
                                "source_circle_fit": feature.get("source_circle_fit"),
                                "source_line_fit": feature.get("source_line_fit"),
                                "source_correlated_groups": explicit_groups,
                                "joint_group_navigation": feature.get("joint_group_navigation"),
                                "source_evidence": feature.get("source_evidence"),
                                "physical_identity_evidence": feature.get("physical_identity_evidence"),
                                "material_transition": feature.get("material_transition"),
                                "measurement_operator": measured_operator,
                                "manual_editor": manual,
                                "manual_deletion_applied": feature.get("manual_deletion_applied"),
                                "excluded_features_by_user": packet.get("excluded_features_by_user", []),
                                "source_review_trimming": feature.get("source_review_trimming"),
                                "manual_history_logical_root": "s02/evidence/point_editor_history" if manual else None,
                                "source_raw_reuse_requires_manual_visibility_and_deletion_intersection": bool(manual),
                                "covariance_status": "not_reconstructed_from_provenance_flags",
                            })
                            counts["source_feature_groups"] += 1
                        for si, segment in enumerate(feature.get("segments", [])):
                            uv_list = segment.get("uv", [])
                            if not uv_list:
                                continue
                            segment_ptr = feature_ptr + f"/segments/{si}"
                            group_id = f"{shot}:{native_pts}:{collection}:{fi}:{si}:{Path(logical).parent.name}"
                            evidence_type = _evidence_class(segment)
                            declared_ids = [g["id"] for g in explicit_groups
                                            if g.get("segment_index", si) == si]
                            for gid in (feature.get("source_evidence_group_id"),
                                        segment.get("source_evidence_group_id")):
                                if gid and gid not in declared_ids:
                                    declared_ids.append(gid)
                            # Circle-family fitting can correlate different ring
                            # roles even when each role has a separate legacy ID.
                            family = feature.get("joint_group_navigation", {}).get("house_id")
                            family_group = f"{shot}:{native_pts}:circle_family:{family}" if family else None
                            source_flags = {k: v for k, v in segment.items()
                                            if isinstance(v, bool)}
                            _emit(out_groups, {
                                "group_id": group_id, "group_type": "segment_source_evidence",
                                "shot_id": shot, "rig_id": rig_id, "native_pts": native_pts,
                                "feature_source_group_id": feature_group_id,
                                "source_packet": logical, "packet_sha256": resolver.digest(logical),
                                "segment_pointer": segment_ptr, "feature_pointer": feature_ptr,
                                "source_feature_id": feature_id, "representatives": len(uv_list),
                                "raw_readout_count": len(segment.get("source_native_readout_uv", [])),
                                "source_native_readout_uv": segment.get("source_native_readout_uv", []),
                                "source_native_transition_iqr_px": segment.get("source_native_transition_iqr_px"),
                                "source_native_station_evidence": segment.get("source_native_station_evidence"),
                                "source_native_station_sigma_px": segment.get("source_native_station_sigma_px"),
                                "point_source_evidence": segment.get("source_evidence"),
                                "source_review_trimming": segment.get("source_review_trimming"),
                                "raw_readout_pointer": segment_ptr + "/source_native_readout_uv"
                                    if "source_native_readout_uv" in segment else None,
                                "source_evidence_pointer": segment_ptr + "/source_evidence"
                                    if "source_evidence" in segment else None,
                                "feature_source_circle_fit_pointer": feature_ptr + "/source_circle_fit"
                                    if "source_circle_fit" in feature else None,
                                "declared_correlation_ids": [f"{shot}:{native_pts}:{x}" for x in declared_ids],
                                "circle_family_coupling_hint": family_group,
                                "legacy_group_declarations": explicit_groups,
                                "evidence_class": evidence_type,
                                "measurement_origin": segment.get("measurement_origin"),
                                "measurement_operator": measured_operator,
                                "source_flags": source_flags,
                                "covariance_status": "not_reconstructed_from_provenance_flags",
                                "raw_and_representative_may_not_both_be_counted": True,
                                "manual_region_authority_must_be_applied_before_raw_reuse": bool(manual),
                            })
                            counts["source_segment_groups"] += 1
                            for vi, uv in enumerate(uv_list):
                                if not (isinstance(uv, list) and len(uv) == 2 and
                                        all(isinstance(x, (int, float)) and math.isfinite(x) for x in uv)):
                                    raise ValueError(f"Invalid source UV at {logical}{segment_ptr}/{vi}")
                                point_binding = _at(segment.get("point_bindings", []), vi) or {}
                                binding_key = (feature_id, si, vi)
                                legacy_binding, binding_source_id = binding_map.get(binding_key, ({}, None))
                                if collection == "unbound_static_materials":
                                    legacy_binding, binding_source_id = {}, None
                                elif binding_map:
                                    if not legacy_binding:
                                        raise ValueError(f"Merged named point lacks typed binding: {logical}/{binding_key}")
                                    binding_used.add(binding_key)
                                proposed_target = point_binding.get("physical_target") or original_target
                                target = legacy_binding.get("target_id") or proposed_target
                                source_id = _at(segment.get("point_ids", []), vi) or binding_source_id
                                if binding_source_id and source_id != binding_source_id:
                                    raise ValueError(f"Packet/binding source point ID mismatch: {logical}/{binding_key}")
                                row_id = f"{logical}#{segment_ptr}/uv/{vi}"
                                if row_id in seen_row_ids:
                                    raise ValueError(f"Repeated authoritative row: {row_id}")
                                seen_row_ids.add(row_id)
                                row_flags = []
                                if source_id:
                                    if source_id in seen_source_ids:
                                        row_flags.append("reused_legacy_source_point_id")
                                        issue("reused_legacy_source_point_id", frame, row_id=row_id,
                                              earlier_row_id=seen_source_ids[source_id], source_id=source_id)
                                    else:
                                        seen_source_ids[source_id] = row_id
                                else:
                                    counts["rows_without_legacy_point_id"] += 1
                                if not (0 <= uv[0] < native_hw[1] and 0 <= uv[1] < native_hw[0]):
                                    row_flags.append("outside_native_image")
                                in_mask = None
                                if mask is not None and "outside_native_image" not in row_flags:
                                    x = min(native_hw[1]-1, max(0, int(math.floor(uv[0]+0.5))))
                                    y = min(native_hw[0]-1, max(0, int(math.floor(uv[1]+0.5))))
                                    in_mask = bool(mask.getpixel((x, y)))
                                    if in_mask:
                                        row_flags.append("inside_supplied_person_mask")
                                if segment.get("hidden_span_filled"):
                                    row_flags.append("inferred_hidden_span")
                                if (packet.get("observation_coordinates_from_projection") or
                                        feature.get("projected_uv_used_as_observations") or
                                        segment.get("observation_coordinates_from_projection")):
                                    row_flags.append("projection_derived_observation")
                                if not segment.get("source_evidence_measured_in_this_frame", feature.get("source_evidence_measured_in_this_frame", False)):
                                    row_flags.append("current_source_evidence_unspecified")
                                category = _kind(target, registry)
                                raw_surface = feature.get("physical_surface")
                                surface = raw_surface
                                if surface is None and named:
                                    if category == "ice_curve":
                                        surface = "ice"
                                    elif category in ("board_edge", "ad_edge", "ad_corner", "ad_print", "ad_boundary_unspecified"):
                                        surface = "board"
                                if category == "ice_curve" and measured_operator is None:
                                    row_flags.append("ice_measurement_operator_unspecified")
                                sigma_source = "sigma_px" if "sigma_px" in segment else "localization_sigma_px"
                                sigma = _at(segment.get(sigma_source), vi)
                                if sigma is not None and not (isinstance(sigma, (int, float)) and math.isfinite(sigma) and sigma > 0):
                                    row_flags.append("invalid_declared_sigma")
                                    sigma = None
                                if sigma is None:
                                    row_flags.append("declared_sigma_missing")
                                coord_key = (target, uv[0], uv[1])
                                if coord_key in frame_coordinates:
                                    row_flags.append("exact_duplicate_same_target_uv")
                                    issue("exact_duplicate_same_target_uv", frame, row_id=row_id,
                                          earlier_row_id=frame_coordinates[coord_key], target_id=target)
                                else:
                                    frame_coordinates[coord_key] = row_id
                                named = collection == "features"
                                state = ("merged_named_pending_auto_qa" if legacy_binding else
                                         "candidate_named_pending_assembly") if named else "unbound_pending_identity"
                                correspondence = legacy_binding.get("point_correspondence", point_binding.get("kind") == "corner")
                                residual = ("point_to_corresponding_corner_2d" if correspondence else
                                            point_binding.get("required_residual") or registry.get(target, {}).get("residual"))
                                result = {
                                    "row_id": row_id, "original_source_point_id": source_id,
                                    "source_row_id": row_id, "shot_id": shot, "rig_id": rig_id,
                                    "programme_frame_index_1based": frame_index,
                                    "native_pts": native_pts, "time_base": str(tb),
                                    "source_seconds": float(native_pts * tb), "uv": uv, "source_uv": uv,
                                    "source_packet": logical, "source_packet_sha256": resolver.digest(logical),
                                    "source_pointer": segment_ptr + f"/uv/{vi}",
                                    "source_feature_id": feature_id, "source_segment_index": si,
                                    "source_vertex_index": vi, "original_feature_target_id": original_target,
                                    "raw_target": original_target,
                                    "target_id": target if named else None,
                                    "material_id": target if named else None,
                                    "unbound_candidate_id": feature_id if not named else None,
                                    "joint_material_id": None, "physical_surface": surface,
                                    "raw_physical_surface": raw_surface,
                                    "physical_surface_source": "explicit_source_field" if raw_surface else "target_category_hint",
                                    "category": category if named else "unbound", "measurement_operator": measured_operator,
                                    "residual_kind": residual if named else None,
                                    "boundary_role": (segment.get("boundary_role") or
                                                      registry.get(target, {}).get("role") or
                                                      feature.get("material_transition") or
                                                      segment.get("material_interface_definition")),
                                    "point_correspondence": bool(correspondence) if named else False,
                                    "declared_sigma_px": sigma, "sigma_source_field": sigma_source,
                                    "source_group_id": group_id,
                                    "provenance_group_id": group_id,
                                    "legacy_dependence_group": legacy_binding.get("dependence_group"),
                                    "evidence_class": evidence_type, "measurement_origin": segment.get("measurement_origin"),
                                    "manual_revision": manual.get("revision_id"),
                                    "manual_authoritative": bool(manual.get("authoritative_over_extraction_candidates")),
                                    "source_user_accepted": packet.get("user_accepted", False),
                                    "source_allowed_for_fit": feature.get("allowed_for_fit"),
                                    "original_allowed_for_fit": feature.get("allowed_for_fit"),
                                    "fit_role": "reference_manual" if evidence_type == "manual_native_click" else "candidate",
                                    "binding_status": state,
                                    "ingest_state": state, "solver_eligible": False,
                                    "binding_path": frame.get("binding") or None,
                                    "binding_registry_path": registry_path,
                                    "inside_supplied_person_mask": in_mask, "qa_flags": sorted(set(row_flags)),
                                }
                                _emit(out_obs, result)
                                counts["rows_total"] += 1
                                frame_counts["named" if named else "unbound"] += 1
                                by_shot[shot]["named" if named else "unbound"] += 1
                                evidence_counts[evidence_type] += 1
                                flags_count.update(row_flags)
                                if legacy_binding:
                                    counts["merged_bound_rows"] += 1
                                    if target != original_target:
                                        counts["per_vertex_target_overrides"] += 1
                                if named:
                                    packet_named += 1
                                    operators[target][str(measured_operator)] += 1
                                    material = materials.setdefault(target, {
                                        "target_id": target, "category": category, "shots": set(),
                                        "rows": 0, "matches_s01_label_only": target in prior_labels,
                                        "joint_binding_status": "requires_physical_and_operator_crosswalk",
                                    })
                                    material["shots"].add(shot)
                                    material["rows"] += 1
                                else:
                                    packet_unbound += 1
                if int(packet.get("point_count", packet_named)) != packet_named:
                    issue("packet_named_count_metadata_mismatch", frame, packet=logical,
                          declared=packet.get("point_count"), actual=packet_named)
                if "source_point_count" in packet and int(packet["source_point_count"]) != packet_named + packet_unbound:
                    issue("packet_total_count_metadata_mismatch", frame, packet=logical,
                          declared=packet["source_point_count"], actual=packet_named+packet_unbound)
                packet_stats.append({"packet": logical, "named": packet_named, "unbound": packet_unbound,
                                     "manual_editor": manual,
                                     "inferred_segments": len(packet.get("inferred_segments", [])),
                                     "unresolved_source_regions": len(packet.get("unresolved_source_regions", []))})
            if set(binding_map) != binding_used:
                raise ValueError(f"Unused bindings: {shot}/{frame_index}")
            for field, key in (("bound_representatives", "named"), ("unbound_representatives", "unbound")):
                if frame_counts[key] != int(frame[field]):
                    raise ValueError(f"Inventory row count mismatch: {shot}/{frame_index}/{key}")
            by_shot[shot]["selected_frames"] += 1
            _emit(out_frames, {"shot_id": shot, "rig_id": rig_id, "programme_frame_index_1based": frame_index,
                               "native_pts": native_pts, "time_base": str(tb), "source_seconds": float(native_pts*tb),
                               "source_video_sha256": source.get("source_video_sha256"),
                               "native_hw": native_hw, "stage": frame["stage"],
                               "named_representatives": frame_counts["named"],
                               "unbound_representatives": frame_counts["unbound"],
                               "mask_path": frame.get("mask"), "mask_audit_status": mask_status,
                               "review_image": frame.get("review_image"), "packets": packet_stats})
    for target, counter in operators.items():
        substantive = {x for x in counter if x != "None"}
        if len(substantive) > 1:
            issue("same_target_has_different_measurement_operators", target_id=target,
                  operators=dict(counter), severity="semantic_gate")
    target_rows = []
    for target, material in sorted(materials.items()):
        material["shots"] = sorted(material["shots"])
        material["measurement_operators"] = dict(operators[target])
        target_rows.append(material)
    counts["frames"] = len(inventory)
    summary = {"schema": SCHEMA, "counts": dict(counts),
               "by_shot": {k: dict(v) for k, v in sorted(by_shot.items())},
               "evidence_class_counts": dict(evidence_counts),
               "qa_flag_counts": dict(flags_count),
               "finding_counts": dict(Counter(x["code"] for x in issues)),
               "named_target_count": len(materials),
               "targets_matching_s01_label_only": sum(t in prior_labels for t in materials),
               "hashes_verified": verify_hashes, "verified_unique_paths": len(verified),
               "solver_eligible_rows": 0,
               "source_quality_status": "structure_and_mask_audit_only_not_visual_certification",
               "rig_by_shot": rig_by_shot,
               "rig_mapping_scope": "historical_physical_installation_shot_identity_bridge_only_no_time_transfer",
               "promotion_required": "automated_source_evidence_and_physical_identity_decision_ledger",
               "source_inventory_sha256": _sha(root / "frames.csv"),
               "source_manifest_sha256": _sha(root / "manifest.json"),
               "source_generation": {"path": str(Path(source_generation).resolve()), "sha256": _sha(Path(source_generation))},
               "normalization_engine_v1_sha256": "8866760079382b9920630ac3c3bb87fa0d3def0aef41a54dea82b003e982f724"}
    for filename, value in (("summary.json", summary), ("source_receipts.json", source_receipts),
                            ("material_crosswalk_candidates.json", target_rows)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (output / "qa_findings.jsonl").open("w", encoding="utf-8") as handle:
        for item in issues:
            _emit(handle, item)
    return summary
