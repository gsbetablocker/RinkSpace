"""Replace source localization covariance without changing its measurement.

Material width and edge-spread width describe the material/profile, not the
sampling error of its measured midpoint. This revision keeps the source mean,
estimator differential, material identity, and observable residual modes fixed.
Only an authenticated source-image noise factor may replace the old support.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .official_static_surface_dataset import read_reference
from .source_evidence_whitening import SourceWhiteningOperator, latent_whitening_operator

SCHEMA = "rinkspace.native-localization-noise-revision.v1"
GROUP_SCHEMA = "rinkspace.fixed-estimator-localization-noise.v1"


def _check_evidence(ref):
    if hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("Native localization noise evidence changed")


def source_members_digest(frame, indices):
    """Bind noise to the exact active measurement after identity/mean revisions."""
    source = [(row, j) for row in frame["rows"] for j in range(len(row["uv"]))]
    digest = hashlib.sha256()
    for index in indices:
        row, j = source[int(index)]
        target = deepcopy(row["target"])
        if target.get("kind") == "board_edge" and "association_policy" in target:
            # The outer revision authenticates the exact old/new bindings.
            # Rebinding identical physical board association to that new
            # dataset does not change the observed material or its kernel.
            target["association_policy"].pop("source_binding", None)
        address = dict(native_pts=int(frame["native_pts"]),
            source_point_id=row["source_point_ids"][j], target=target,
            source_feature_id=row["source_feature_id"],
            original_vertex_index=int(row["original_vertex_indices"][j]))
        digest.update(json.dumps(address, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.asarray(row["uv"][j], dtype=np.float64).tobytes())
        digest.update(np.asarray(row["sigma"][j], dtype=np.float64).tobytes())
    return digest.hexdigest()


def fixed_mode_transform(old, new):
    """Return the exact change of units within unchanged physical source modes."""
    if (old.source_point_ids != new.source_point_ids or old.group_id != new.group_id
            or old.rank != new.rank):
        raise ValueError("Noise replacement cannot change source membership or residual rank")
    transform = np.linalg.lstsq(old.matrix.T, new.matrix.T, rcond=None)[0].T
    mismatch = transform @ old.matrix-new.matrix
    relative = np.linalg.norm(mismatch)/max(np.linalg.norm(new.matrix), np.finfo(float).tiny)
    if relative > 2e-10 or np.linalg.matrix_rank(transform) != old.rank:
        raise ValueError("Noise replacement changed the physical source residual row space")
    return transform, float(relative)


def _noise_operator(item, old):
    if (item.get("schema") != GROUP_SCHEMA or item.get("source_mean_and_D_unchanged") is not True
            or item.get("material_width_used_as_position_noise") is not False
            or item.get("world_or_camera_residual_used_for_noise") is not False):
        raise ValueError("Explicit source-only localization noise and unchanged estimator required")
    basis = np.asarray(item["physical_basis"], dtype=float)
    sensitivity = np.asarray(item["estimator_output_sensitivity"], dtype=float)
    differential = np.asarray(item["coefficient_influence"], dtype=float)
    support = np.asarray(item["original_support_factor"], dtype=float)
    noise = np.asarray(item["localization_noise_factor"], dtype=float)
    if (any(a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all()
            for a in (basis, sensitivity, differential, support, noise))
            or support.shape[0] != noise.shape[0]
            or differential.shape[1] != support.shape[0]):
        raise ValueError("Native localization covariance has an invalid measured inventory")
    previous_factor = differential @ support
    old_rebuilt = latent_whitening_operator(old.source_point_ids, sensitivity,
        explicit_noise_factor=previous_factor, physical_basis=basis,
        group_id=old.group_id, covariance_origin="authenticated_previous_profile_support")
    fixed_mode_transform(old, old_rebuilt)
    standardized = old.matrix @ basis.T @ sensitivity @ previous_factor
    if not np.allclose(standardized @ standardized.T, np.eye(old.rank), rtol=2e-8, atol=2e-9):
        raise ValueError("Declared fixed estimator D does not reproduce the original source covariance")
    updated = latent_whitening_operator(old.source_point_ids, sensitivity,
        explicit_noise_factor=differential @ noise, physical_basis=basis,
        group_id=old.group_id, covariance_origin="native_image_localization_noise_through_unchanged_estimator")
    fixed_mode_transform(old, updated)
    return updated


def make_noise_replacement(frame, group, *, physical_basis, estimator_output_sensitivity,
        coefficient_influence, original_support_factor, localization_noise_factor,
        evidence_references, noise_method):
    """Build a complete reviewable noise revision from measured native factors."""
    if not evidence_references or not noise_method:
        raise ValueError("Measured native noise needs its method and authenticated source evidence")
    for ref in evidence_references:
        _check_evidence(ref)
    old = group["operator"]
    item = dict(schema=GROUP_SCHEMA, native_pts=int(frame["native_pts"]),
        group_id=old.group_id, point_indices=list(map(int, group["point_indices"])),
        source_members_sha256=source_members_digest(frame, group["point_indices"]),
        original_operator=old.to_record(),
        source_mean_and_D_unchanged=True, material_width_used_as_position_noise=False,
        world_or_camera_residual_used_for_noise=False, source_points_added_or_removed=False,
        evidence_references=evidence_references, noise_method=noise_method)
    for name, value in dict(physical_basis=physical_basis,
            estimator_output_sensitivity=estimator_output_sensitivity,
            coefficient_influence=coefficient_influence,
            original_support_factor=original_support_factor,
            localization_noise_factor=localization_noise_factor).items():
        item[name] = np.asarray(value, dtype=float).tolist()
    item["new_operator"] = _noise_operator(item, old).to_record()
    return item


def revise_localization_noise(frames, revision_reference, binding):
    """Apply complete covariance replacements after existing source revisions."""
    revision = read_reference(revision_reference)
    base = read_reference(revision["base_binding"])
    if (revision.get("schema") != SCHEMA
            or revision.get("source_video_sha256") != binding["source"]["sha256"]
            or revision.get("source_time_base") != binding["source_clock"]["time_base"]
            or any(base.get(k) != binding.get(k) for k in (
                "source", "source_clock", "complete_fit_input", "source_revision", "model"))
            or revision.get("source_UV_IDs_targets_and_estimator_D_modified") is not False):
        raise ValueError("Localization noise revision belongs to another active source measurement")
    replacements = {}
    checked_evidence = set()
    for item in revision["groups"]:
        address = int(item["native_pts"]), item["group_id"]
        if address in replacements or item.get("source_points_added_or_removed") is not False:
            raise ValueError("Duplicate or membership-changing localization noise revision")
        refs = item.get("evidence_references", [])
        if not refs or not item.get("noise_method"):
            raise ValueError("Localization noise must retain its native-image evidence")
        for ref in refs:
            key = (ref["path"], ref["sha256"])
            if key not in checked_evidence:
                _check_evidence(ref)
                checked_evidence.add(key)
        replacements[address] = item
    used = set()
    result = []
    max_row_space_error = 0.
    for original in frames:
        frame = dict(original)
        groups = []
        for original_group in original["whitening_groups"]:
            old = original_group["operator"]
            address = int(frame["native_pts"]), old.group_id
            item = replacements.get(address)
            if item is None:
                groups.append(original_group)
                continue
            saved = SourceWhiteningOperator.from_record(item["original_operator"])
            if (item["point_indices"] != list(original_group["point_indices"])
                    or item["source_members_sha256"] != source_members_digest(frame, item["point_indices"])
                    or saved.source_point_ids != old.source_point_ids or saved.group_id != old.group_id
                    or not np.array_equal(saved.matrix, old.matrix)):
                raise ValueError("Localization noise revision changed its original means/IDs/targets/operator")
            updated = _noise_operator(item, old)
            supplied = SourceWhiteningOperator.from_record(item["new_operator"])
            if not np.array_equal(updated.matrix, supplied.matrix):
                raise ValueError("New whitening operator is not the measured noise propagated through fixed D")
            _, error = fixed_mode_transform(old, updated)
            max_row_space_error = max(max_row_space_error, error)
            group = dict(original_group, operator=updated,
                localization_noise_revision=deepcopy(item),
                original_support_operator=old.to_record())
            groups.append(group)
            used.add(address)
        frame["whitening_groups"] = groups
        result.append(frame)
    if used != set(replacements):
        raise ValueError("Localization noise refers to a missing active source group")
    return result, dict(reference=revision_reference, base_binding=revision["base_binding"],
        replaced_groups=len(used), source_UV_IDs_targets_and_estimator_D_modified=False,
        residual_rank_and_row_space_unchanged=True, max_row_space_relative_error=max_row_space_error,
        material_width_used_as_position_noise_for_replaced_groups=False,
        world_or_camera_residual_used_for_noise=False,
        scope="explicit authenticated groups; unlisted source noise is preserved")
