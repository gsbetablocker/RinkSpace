"""Explicit, source-reviewed Ad7 path identities on the existing print bank.

The old raster component labels are retained as source IDs. A frozen list of
reviewed source pixels may select one true SVG path for an existing source row.
No source values, W, parameters, camera execution or material bank are changed.
Import has no side effects; install only after historical source authentication
and before any pool, world Jacobian or source Jacobian has been constructed.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
from types import MethodType

import numpy as np

from .first_shot_source_transition import reference, read

SCHEMA = "rinkspace.source-reviewed-Ad7-true-paths.v1"
SOURCE_PATHS = ("logo_part00", "logo_part01", "logo_part04", "logo_part05")
DESTINATION_PATH = "logo_part03"


def _load(mapping_reference):
    value = read(mapping_reference)
    if (value.get("schema") != SCHEMA
            or value.get("event_id") != "ankara_2026"
            or value.get("programme_id") != "okada_mei_sp"
            or value.get("time_base") != "1/12800"
            or value.get("authority") != "agent_rgb_review"
            or value.get("review_complete") is not True
            or value.get("model_argmin_is_identity_authority") is not False
            or value.get("source_UV_ID_sigma_W_clock_unchanged") is not True):
        raise ValueError("Explicit source-reviewed Ad7 path mapping required")
    if value.get("user_confirmed") is True or value.get("human_confirmed") is True:
        raise ValueError("Agent pixel-identity review cannot claim human confirmation")
    if value.get("producer") != reference(__file__):
        raise ValueError("Reviewed source path producer differs")
    for item in value.get("evidence", []):
        if reference(item["path"]) != item:
            raise ValueError("Reviewed source path evidence changed")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("At least one explicitly reviewed source pixel required")
    lookup = {}
    for item in records:
        frame = item.get("frame")
        point = item.get("source_point_id")
        uv = np.asarray(item.get("source_uv"), float)
        if (type(frame) is not int or not 1 <= frame <= 1248
                or item.get("native_pts") != 256256 + (frame-1)*256
                or not isinstance(point, str)
                or not point.startswith("Ad7.print.component000/segment/")
                or uv.shape != (2,) or not np.isfinite(uv).all()
                or item.get("true_path_id") != DESTINATION_PATH
                or item.get("source_physical_identity") != "ISU_logo_S_lower_detached_ink_boundary"
                or item.get("source_image_reviewed") is not True):
            raise ValueError("Complete finite source-reviewed lower-S identity required")
        key = frame, point
        if key in lookup:
            raise ValueError("Duplicate source-reviewed frame/point identity")
        lookup[key] = item
    return value, lookup


def reviewed_target(target, true_path_id):
    """Replace the mistaken alias with ONE already-existing physical path."""
    from .first_shot_disjoint_print import KIND, PathUnion
    from .first_shot_print_layout import print_layout_template
    from . import first_shot_disjoint_print as disjoint
    if (disjoint._CONFIG is None or target.get("kind") != KIND
            or target.get("advertisement_id") != 7
            or target.get("contour_key") != "component000"
            or tuple(PathUnion.from_target(target).ids) != SOURCE_PATHS
            or true_path_id != DESTINATION_PATH):
        raise ValueError("Only the reviewed original Ad7 component000 alias is supported")
    material = print_layout_template(disjoint._CONFIG["templates"], 7)
    path = material["contours"][true_path_id]
    uv = np.asarray(path["material_uv"], float)
    count = len(uv) + int(path["closed"] and not np.array_equal(uv[0], uv[-1]))
    result = deepcopy(target)
    result["curve_topology"] = dict(result["curve_topology"],
        path_ids=[true_path_id], offsets=[0, count], closed=[path["closed"]])
    PathUnion.from_target(result)
    return result


def transition_frames(frames, *, lookup, mapping_reference, target_builder=reviewed_target):
    """Pure source metadata transition; every listed current pixel must match."""
    from .first_shot_fit import _target_key
    from .first_shot_source_snapshot import _frame_contract
    before = _frame_contract(frames)
    changed, used, output = [], set(), []
    for old_frame in frames:
        frame = dict(old_frame)
        rows = []
        for old_row in old_frame["rows"]:
            keys = [(int(frame["index"]), str(pid)) for pid in old_row.get("source_point_ids", [])]
            matched = [key for key in keys if key in lookup]
            if not matched:
                rows.append(old_row)
                continue
            if len(keys) != 1 or len(matched) != 1 or len(old_row["uv"]) != 1:
                raise ValueError("Reviewed path transition requires the original one-pixel source rows")
            key = matched[0]
            item = lookup[key]
            if (frame["native_pts"] != item["native_pts"]
                    or not np.array_equal(np.asarray(old_row["uv"])[0], np.asarray(item["source_uv"]))):
                raise ValueError("Reviewed source UV or native clock changed")
            target = target_builder(old_row["target"], item["true_path_id"])
            row = dict(old_row, target=target, _target_key=_target_key(target))
            row["source_true_path_history"] = deepcopy(old_row.get("source_true_path_history", [])) + [
                dict(previous_target=deepcopy(old_row["target"]),
                     previous_target_key=old_row["_target_key"], mapping=deepcopy(mapping_reference),
                     source_point_id=key[1], physical_identity=item["source_physical_identity"],
                     true_path_id=item["true_path_id"])]
            rows.append(row)
            used.add(key)
            changed.append(dict(frame=key[0], source_point_id=key[1], true_path_id=item["true_path_id"]))
        frame["rows"] = rows
        output.append(frame)
    if used != set(lookup):
        raise ValueError("Some reviewed source identities are absent from the current source")
    after = _frame_contract(output)
    if before != after:
        raise ValueError("Reviewed path mapping changed source UV/ID/sigma/W/clock")
    return output, dict(changed_source_points=len(changed), records=changed,
        source_contract_before=before, source_contract_after=after,
        source_values_and_complete_W_unchanged=True)


def install_successor(owner, *, mapping_reference):
    """Install after original factory authentication, before its first spawn."""
    value, lookup = _load(mapping_reference)
    programme = owner.programme
    bundle = programme.bundle
    if any(pool is not None for pool in (programme.pool, bundle._geometry_pool, bundle._source_pool)):
        raise ValueError("Install reviewed path identities before all execution pools")
    if getattr(owner, "ad7_source_path_receipt", None) is not None:
        raise ValueError("Reviewed source path identities must be installed once")
    before_q = np.asarray(owner.start).copy()
    before_aux = {key:np.asarray(item).copy() for key,item in owner.checkpoint_arrays().items()}
    frames, transition = transition_frames(bundle.frames, lookup=lookup, mapping_reference=mapping_reference)
    bundle.frames = frames
    bundle.descriptors = OrderedDict()
    for frame in frames:
        for row in frame["rows"]:
            bundle.descriptors.setdefault(row["_target_key"], deepcopy(row["target"]))
    bundle.keys = tuple(bundle.descriptors)
    bundle.target_descriptors = tuple(bundle.descriptors.values())
    from .first_shot_disjoint_print import _clear_caches
    _clear_caches(programme)
    receipt = dict(schema=SCHEMA, producer=reference(__file__), mapping=deepcopy(mapping_reference),
        transition=transition, fixed_parameters_or_new_parameters_added=False,
        material_bank_or_geometry_changed=False, source_labels_retained=True,
        original_complete_source_W_retained=True, row_time_camera_execution_unchanged=True,
        original_source_or_world_J_reusable=False, fresh_source_J_required=True,
        wrong_upper_S_path_is_not_a_candidate_for_reviewed_lower_S_pixels=True,
        original_source_target_history_preserved=True)
    previous_identity, previous_arrays = owner.checkpoint_identity, owner.checkpoint_arrays
    archive = "ad7_source_paths_" + mapping_reference["sha256"][:16] + "_json"
    metadata = np.asarray(json.dumps(receipt, sort_keys=True))
    if archive in before_aux:
        old = before_aux[archive]
        if old.dtype != metadata.dtype or old.shape != metadata.shape or old.tobytes() != metadata.tobytes():
            raise ValueError("Inherited reviewed source path auxiliary conflicts")
    def identity(this):
        return dict(previous_identity(), ad7_source_paths=deepcopy(receipt))
    def arrays(this):
        return dict(previous_arrays(), **{archive:metadata.copy()})
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.checkpoint_arrays = MethodType(arrays, owner)
    for ref in (reference(__file__), deepcopy(mapping_reference)):
        if reference(ref["path"]) != ref:
            raise ValueError("Reviewed path producer or mapping changed")
        for obj, name in ((owner,"source_producers"), (owner,"_references"), (programme,"source_producers")):
            values = getattr(obj, name)
            if ref not in values:
                values.append(deepcopy(ref))
    if not np.array_equal(np.asarray(owner.start).view(np.uint8), before_q.view(np.uint8)):
        raise ValueError("Reviewed path installation changed canonical parameters")
    after_aux = owner.checkpoint_arrays()
    for key, expected in before_aux.items():
        actual = after_aux[key]
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("Reviewed path installation changed inherited auxiliary: " + key)
    owner.ad7_source_path_receipt = deepcopy(receipt)
    return deepcopy(receipt)
