"""Authenticated RGB-based revisions without rewriting approved source history.

An approved pixel and its material identity are distinct facts. Revisions keep
the former immutable and record the evidence for correcting the latter. A
quarantined correspondence remains in the review inventory, outside FIT.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .official_static_surface_dataset import read_reference

SOURCE_SCHEMA = "rinkspace.native-source-correspondence-revision.v1"
MATERIAL_SCHEMA = "rinkspace.native-material-template-revision.v1"
ESTIMATOR_SCHEMA = "rinkspace.native-correlated-source-estimator-replacement.v1"
_KINDS = {"ad_print", "ad_edge", "ad_corner", "board_edge", "ice_curve"}


def _evidence(item):
    if item.get("authority") != "agent_rgb_review" or not item.get("reason"):
        raise ValueError("A revision requires its actual RGB reviewer and reason")
    if item.get("identity_confirmed_by_user") is True:
        raise ValueError("A new agent review cannot invent a user confirmation")
    refs = item.get("evidence_references", [])
    if not refs:
        raise ValueError("A revision requires authenticated source evidence")
    for ref in refs:
        if hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("Source revision evidence changed")


def _target(value):
    if not isinstance(value, dict) or value.get("kind") not in _KINDS:
        raise ValueError("Only verified static ice or board material is admissible")
    return deepcopy(value)


def _key(target):
    value = deepcopy(target)
    if value["kind"] == "board_edge" and "association_policy" in value:
        value.pop("part", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _source_address(item):
    return int(item["native_pts"]), item["source_point_id"]


def _check_original(item, row, index):
    if (not np.array_equal(np.asarray(item["original_UV"], float), row["uv"][index])
            or item["original_target"] != row["target"]):
        raise ValueError("Revision source UV or original material identity differs")


def _final_native_line_support(frame, item, binding):
    """Authenticate the final support through the unchanged original packet."""
    if item["source_packet"] != frame["source_packet"]:
        raise ValueError("Estimator replacement changed its original source packet")
    packet = read_reference(frame["source_packet"])
    native_ref = packet["ice_source_readout_packet"]
    if item["native_readout_packet"] != native_ref:
        raise ValueError("Estimator replacement uses another native support packet")
    native = read_reference(native_ref)
    for source in (packet["source"], native["source"]):
        if (int(source["native_pts"]) != int(frame["native_pts"])
                or int(source["actual_pts"]) != int(frame["native_pts"])
                or source["time_base"] != binding["source_clock"]["time_base"]
                or source["source_video_sha256"] != binding["source"]["sha256"]):
            raise ValueError("Estimator replacement has foreign native support or source clock")
    if packet["source"]["decoded_BGR_sha256"] != native["source"]["decoded_BGR_sha256"]:
        raise ValueError("Estimator support comes from different native RGB pixels")
    features = [f for f in native["features"] if f["id"] == item["source_feature_id"]]
    if len(features) != 1 or not features[0].get("source_line_fit"):
        raise ValueError("Estimator replacement requires one actual final native line feature")
    return features[0]["source_line_fit"]


def build_correlated_estimator_replacement(frame, group, binding, *, reason, evidence_references):
    """Prepare one whole-group derived replacement, preserving its original facts."""
    from .first_shot_source_evidence import rebuild_final_native_line_estimator
    indices = np.asarray(group["point_indices"], int)
    source = [(r, j) for r in frame["rows"] for j in range(len(r["uv"]))]
    members = [source[i] for i in indices]
    features = {r["source_feature_id"] for r, j in members}
    if len(features) != 1:
        raise ValueError("A derived source-line replacement cannot combine source features")
    fid = features.pop()
    packet = read_reference(frame["source_packet"])
    item = dict(schema=ESTIMATOR_SCHEMA, native_pts=int(frame["native_pts"]),
        group_id=group["operator"].group_id, source_feature_id=fid,
        source_packet=frame["source_packet"], native_readout_packet=packet["ice_source_readout_packet"],
        original_point_indices=indices.tolist(), original_operator=group["operator"].to_record(),
        original_members=[dict(source_point_id=r["source_point_ids"][j],
            original_UV=r["uv"][j].tolist(), original_sigma_px=float(r["sigma"][j]),
            original_target=deepcopy(r["target"]), original_source_feature_id=r["source_feature_id"],
            original_source_segment_index=int(r["source_segment_index"]),
            original_source_vertex_index=int(r["original_vertex_indices"][j])) for r, j in members],
        measurement_kind="derived_correlated_source_line_estimator",
        derived_from_same_final_native_support=True, independent_native_readouts_added=0,
        representative_count_increases_source_information=False,
        source_identity_confirmed_by_this_operation=False,
        original_source_UV_IDs_and_approvals_modified=False,
        raw_whitening_row_or_column_slice_used=False, authority="agent_rgb_review",
        reason=reason, evidence_references=evidence_references)
    line = _final_native_line_support(frame, item, binding)
    rebuilt = rebuild_final_native_line_estimator(line, fid,
        group["operator"].source_point_ids, np.asarray([r["uv"][j] for r, j in members]),
        group_id=item["group_id"])
    item.update(estimator_contract=rebuilt["metadata"], new_operator=rebuilt["operator"].to_record(),
        derived_representatives=[dict(source_point_id=pid, UV=uv.tolist(), fixed_source_station=float(t))
            for pid, uv, t in zip(group["operator"].source_point_ids, rebuilt["uv"], rebuilt["stations"], strict=True)])
    _validate_estimator_replacement(item, frame, group, source, binding)
    return item


def _validate_estimator_replacement(item, frame, group, source, binding):
    from .first_shot_source_evidence import rebuild_final_native_line_estimator
    from .source_evidence_whitening import SourceWhiteningOperator
    if (item.get("schema") != ESTIMATOR_SCHEMA
            or item.get("measurement_kind") != "derived_correlated_source_line_estimator"
            or item.get("derived_from_same_final_native_support") is not True
            or item.get("independent_native_readouts_added") != 0
            or item.get("representative_count_increases_source_information") is not False
            or item.get("source_identity_confirmed_by_this_operation") is not False
            or item.get("original_source_UV_IDs_and_approvals_modified") is not False
            or item.get("raw_whitening_row_or_column_slice_used") is not False):
        raise ValueError("An explicit derived whole-group source estimator, not independent readouts, is required")
    indices = np.asarray(group["point_indices"], int)
    original = item["original_members"]
    old = SourceWhiteningOperator.from_record(item["original_operator"])
    new = SourceWhiteningOperator.from_record(item["new_operator"])
    ids = tuple(source[i][0]["source_point_ids"][source[i][1]] for i in indices)
    if (item["original_point_indices"] != indices.tolist() or len(original) != len(indices)
            or old.group_id != group["operator"].group_id or old.group_id != item["group_id"]
            or new.group_id != old.group_id or old.source_point_ids != ids or new.source_point_ids != ids
            or not np.array_equal(old.matrix, group["operator"].matrix) or old.rank != new.rank
            or item["group_id"] != f"{frame['native_pts']}/{item['source_feature_id']}/source_line"):
        raise ValueError("Estimator replacement differs from its exact original full group/IDs/W or adds residual modes")
    for member, index in zip(original, indices, strict=True):
        row, j = source[index]
        _check_original(member, row, j)
        if (member["source_point_id"] != row["source_point_ids"][j]
                or member["original_sigma_px"] != float(row["sigma"][j])
                or member["original_source_feature_id"] != row["source_feature_id"]
                or member["original_source_segment_index"] != row["source_segment_index"]
                or member["original_source_vertex_index"] != row["original_vertex_indices"][j]
                or row["source_feature_id"] != item["source_feature_id"]
                or row["target"]["kind"] != "ice_curve"
                or "ice."+row["target"]["curve_id"] != item["source_feature_id"]):
            raise ValueError("Estimator replacement changed an original member's ID/sigma/material lineage")
    line = _final_native_line_support(frame, item, binding)
    rebuilt = rebuild_final_native_line_estimator(line, item["source_feature_id"], ids,
        np.asarray([m["original_UV"] for m in original]), group_id=item["group_id"])
    if item["estimator_contract"] != rebuilt["metadata"]:
        raise ValueError("Estimator mean and differential do not certify the same final native support/D")
    derived = item["derived_representatives"]
    if (len(derived) != len(ids) or tuple(m["source_point_id"] for m in derived) != ids
            or not np.array_equal(np.asarray([m["UV"] for m in derived]), rebuilt["uv"])
            or not np.array_equal(np.asarray([m["fixed_source_station"] for m in derived]), rebuilt["stations"])):
        raise ValueError("Partial or inconsistent derived mean/station replacement")
    if not np.array_equal(new.matrix, rebuilt["operator"].matrix):
        raise ValueError("Derived mean and full W do not come from the same final native support differential")
    return rebuilt


def material_revision(reference, binding, original_template_reference):
    """Load one source-derived common template, shared by fit and rendering."""
    revision = read_reference(reference)
    if (revision.get("schema") != MATERIAL_SCHEMA
            or revision.get("base_material_templates") != original_template_reference
            or revision.get("source_video_sha256") != binding["source"]["sha256"]
            or revision.get("source_time_base") != binding["source_clock"]["time_base"]
            or revision.get("original_templates_modified") is not False):
        raise ValueError("Material revision is not bound to this source and original template")
    _evidence(revision)
    result = read_reference(revision["material_templates"])
    return result, dict(reference=reference, base=original_template_reference,
                       material_templates=revision["material_templates"],
                       authority=revision["authority"],
                       reset_layout_advertisement_ids=revision["reset_layout_advertisement_ids"],
                       original_templates_modified=False)


def revise_frames(frames, reference, binding, *, policy=None):
    """Apply exact source-addressed edits and preserve intact correlated groups.

    Partial removal or identity changes inside a correlated group require a
    fresh source-evidence operator. They are rejected here rather than silently
    cutting columns from W, which would change the statistical observation.
    Added ordinary rows must be independently read native material transitions.
    """
    revision = read_reference(reference)
    if (revision.get("schema") != SOURCE_SCHEMA
            or revision.get("source_video_sha256") != binding["source"]["sha256"]
            or revision.get("source_time_base") != binding["source_clock"]["time_base"]
            or revision.get("original_source_UV_IDs_and_approvals_modified") is not False):
        raise ValueError("Revision must preserve and identify the original native source")
    base = read_reference(revision["base_binding"])
    if (base["complete_fit_input"] != binding["complete_fit_input"]
            or base["model"] != binding["model"]
            or base["source"] != binding["source"]):
        raise ValueError("Revision belongs to another base dataset or material model")
    changes, quarantine, additions = {}, {}, {}
    for field, table in (("identity_corrections", changes), ("quarantined_correspondences", quarantine)):
        for item in revision.get(field, []):
            _evidence(item)
            address = _source_address(item)
            if address in changes or address in quarantine:
                raise ValueError("Duplicate or conflicting source revision address")
            table[address] = item
    for item in revision.get("added_native_readouts", []):
        _evidence(item)
        if (item.get("measurement_kind") != "direct_material_transition"
                or item.get("derived_from_shared_fit") is not False):
            raise ValueError("Added ordinary points must be independent native readouts")
        additions.setdefault(int(item["native_pts"]), []).append(item)
    replacements, estimator_replacements = {}, {}
    for item in revision.get("correlated_operator_replacements", []):
        _evidence(item)
        address = int(item["native_pts"]), item["group_id"]
        if address in replacements or item.get("raw_whitening_row_or_column_slice_used") is not False:
            raise ValueError("A source group requires a unique independently rebuilt operator")
        replacements[address] = item
    for item in revision.get("correlated_estimator_replacements", []):
        _evidence(item)
        address = int(item["native_pts"]), item["group_id"]
        if address in replacements or address in estimator_replacements:
            raise ValueError("Conflicting operator-only or derived estimator replacement for one source group")
        estimator_replacements[address] = item
    used, used_additions, used_replacements, used_estimators = set(), set(), set(), set()
    derived_count = 0
    result, descriptors, audit = [], OrderedDict(), []
    for original in frames:
        frame = dict(original)
        pts = int(frame["native_pts"])
        source = [(row, j) for row in original["rows"] for j in range(len(row["uv"]))]
        ids = [row["source_point_ids"][j] for row, j in source]
        if len(set(ids)) != len(ids):
            raise ValueError("The base source has duplicate point IDs")
        grouped = {int(i) for group in original["whitening_groups"] for i in group["point_indices"]}
        keep = np.ones(len(source), bool)
        rows, inactive = [], []
        original_hash = hashlib.sha256()
        for i, (row, j) in enumerate(source):
            pid = ids[i]
            original_hash.update(pid.encode("utf8"))
            original_hash.update(row["uv"][j].tobytes())
            address = pts, pid
            item = changes.get(address, quarantine.get(address))
            if item is not None:
                _check_original(item, row, j)
                used.add(address)
            if address in quarantine:
                keep[i] = False
                inactive.append(dict(source_point_id=pid, UV=row["uv"][j].tolist(),
                                     original_target=deepcopy(row["target"]), revision=item))
                continue
            current = dict(row)
            current.update(uv=row["uv"][j:j+1].copy(), sigma=row["sigma"][j:j+1].copy(),
                           source_point_ids=[pid], original_vertex_indices=[row["original_vertex_indices"][j]])
            if address in changes:
                if i in grouped:
                    raise ValueError("A grouped identity change needs rebuilt physical source-normal evidence")
                target = _target(item["target"])
                if target["kind"] == "board_edge" and policy is not None:
                    from .continuous_board_association import with_continuous_board_association
                    target = with_continuous_board_association(target, policy)
                current.update(target=target, _target_key=_key(target), source_identity_revision=item,
                               original_model_target=deepcopy(row["target"]))
                current["point_correspondence"] = target["kind"] == "ad_corner"
            rows.append(current)
        index_map = np.full(len(source), -1, int)
        index_map[keep] = np.arange(int(keep.sum()))
        groups = []
        for group in original["whitening_groups"]:
            indices = np.asarray(group["point_indices"], int)
            retained = keep[indices]
            address = pts, getattr(group["operator"], "group_id", "")
            estimator = estimator_replacements.get(address)
            if estimator is not None:
                if (not retained.all() or any((pts, ids[i]) in changes for i in indices)):
                    raise ValueError("Partial identity/quarantine cannot be mixed with a whole-group estimator replacement")
                rebuilt = _validate_estimator_replacement(estimator, original, group, source, binding)
                for k, index in enumerate(indices):
                    current = rows[int(index_map[index])]
                    current.update(uv=rebuilt["uv"][k:k+1].copy(),
                        original_source_uv=source[index][0]["uv"][source[index][1]:source[index][1]+1].copy(),
                        source_estimator_revision=estimator, derived_correlated_source_representative=True,
                        new_independent_native_readout=False)
                groups.append(dict(group, point_indices=index_map[indices].tolist(),
                    operator=rebuilt["operator"], source_estimator_revision=estimator,
                    original_source_evidence=deepcopy(group.get("source_evidence")),
                    source_evidence=dict(group.get("source_evidence", {}),
                        estimator_contract=rebuilt["metadata"], derived_correlated_representatives=True,
                        original_native_readout_support_preserved=True)))
                used_estimators.add(address); derived_count += len(indices)
                continue
            replacement = replacements.get(address)
            if replacement is not None:
                from .source_evidence_whitening import SourceWhiteningOperator
                old = SourceWhiteningOperator.from_record(replacement["original_operator"])
                new = SourceWhiteningOperator.from_record(replacement["new_operator"])
                if (replacement["original_point_indices"] != indices.tolist()
                        or tuple(ids[i] for i in indices) != old.source_point_ids
                        or not np.array_equal(old.matrix, group["operator"].matrix)
                        or tuple(ids[i] for i in indices[retained]) != new.source_point_ids):
                    raise ValueError("Rebuilt source operator does not match its exact original and retained members")
                groups.append(dict(group, point_indices=index_map[indices[retained]].tolist(),
                                   operator=new, source_evidence_revision=replacement))
                used_replacements.add(address)
                continue
            if retained.any() and not retained.all():
                raise ValueError("Partial correlated-group quarantine needs rebuilt source evidence")
            if retained.all():
                new = dict(group)
                new["point_indices"] = index_map[indices].tolist()
                groups.append(new)
        added = additions.get(pts, [])
        for item in added:
            pid = item["source_point_id"]
            address = pts, pid
            if pid in ids or address in used_additions:
                raise ValueError("New native readout duplicates an original or added point")
            uv = np.asarray(item["UV"], float)
            sigma = float(item["sigma_px"])
            if (uv.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(sigma)
                    or sigma <= 0 or np.any(uv < 0)
                    or uv[0] >= binding["source"]["width"] or uv[1] >= binding["source"]["height"]):
                raise ValueError("New native readout is outside the source or has invalid support")
            target = _target(item["target"])
            if target["kind"] == "board_edge" and policy is not None:
                from .continuous_board_association import with_continuous_board_association
                target = with_continuous_board_association(target, policy)
            rows.append(dict(target=target, _target_key=_key(target), id=pid,
                uv=uv[None].copy(), sigma=np.array([sigma]), source_point_ids=[pid],
                original_vertex_indices=[0], point_correspondence=target["kind"] == "ad_corner",
                native_pts=pts, actual_pts=pts, time_base=binding["source_clock"]["time_base"],
                source_video_sha256=binding["source"]["sha256"], source_packet=frame["source_packet"],
                source_feature_id=item["source_feature_id"], source_segment_index=0,
                added_native_readout=item))
            used_additions.add(address)
        for row in rows:
            descriptors.setdefault(row["_target_key"], row["target"])
        frame.update(rows=rows, source_point_count=len(rows), whitening_groups=groups,
                     inactive_source_correspondences=inactive,
                     source_revision_reference=reference,
                     original_approved_source_point_count=len(source))
        result.append(frame)
        audit.append(dict(frame=frame["index"], native_pts=pts,
            original_points=len(source), active_original_points=int(keep.sum()),
            quarantined_points=len(inactive), added_native_readouts=len(added),
            original_ID_UV_sha256=original_hash.hexdigest()))
    if used != set(changes) | set(quarantine) or len(used_additions) != sum(map(len, additions.values())):
        raise ValueError("A revision refers to a missing native exposure or source point")
    if used_replacements != set(replacements):
        raise ValueError("A replacement refers to a missing correlated source group")
    if used_estimators != set(estimator_replacements):
        raise ValueError("A derived estimator replacement refers to a missing correlated source group")
    proof = dict(reference=reference, base_binding=revision["base_binding"], frames=audit,
                 identity_corrections=len(changes), quarantined_correspondences=len(quarantine),
                 wrong_entity_corrections=revision.get("wrong_entity_corrections", len(changes)),
                 word_to_verified_letter_refinements=revision.get("word_to_verified_letter_refinements", 0),
                 added_native_readouts=len(used_additions), original_source_files_modified=False,
                 rebuilt_correlated_source_groups=len(used_replacements)+len(used_estimators),
                 correlated_estimator_replacements=len(used_estimators),
                 derived_correlated_representatives=derived_count,
                 active_derived_source_UV_replaced=bool(used_estimators),
                 new_independent_residual_rows_from_estimator_replacements=0,
                 original_user_approval_rewritten=False, revised_identity_authority="agent_rgb_review")
    return result, descriptors, proof
