"""Explicit saved RF4 feasible initializer for one original qualified exposure.

No observation, loss, world material, shared mount or physical domain changes.
A later already-valid state skips the old candidate.  Only the authenticated
invalid parent may receive its four nonreference Rodrigues/logf coordinates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from installation_mount_candidate.mount_graph import validate_mounts
from projection import project_central_valid
from reference_board_pair import BoardIntersectionError
from z190_secondary_original_ring_source_qualification import STRICT_KIND

from .paired_programme_chart import CORE_COUNT, WORLD_INDICES


def _q_sha(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _source_signature(rows):
    result = []
    for row in rows:
        entry = dict(id=row['id'], pts=int(row['pts']), split=row.get('split'),
                     branch=row['scanline_branch'],
                     original_source_point_indices=list(row['original_source_point_indices']))
        for field in ('uv', 'sigma', 'normal', 'normal_interval'):
            value = np.asarray(row[field])
            entry[field] = hashlib.sha256(
                str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()
        result.append(entry)
    return result


def apply_qualified_ring_pose_initializer(state, q, packet_path):
    """Return a verified full q copy and report; never assign a live state q."""
    path = Path(packet_path).resolve()
    payload = path.read_bytes()
    packet = json.loads(payload.decode('utf8'))
    if (packet.get('schema') != 'rinkspace.single-actual-RF4-original-source-domain-feasible-initializer-candidate.v1'
            or packet.get('all_constraints_feasible') is not True
            or packet.get('original_source_UV_sigma_intervals_roles_unchanged') is not True
            or packet.get('positive_P_shared_lever_bounds_and_world_unchanged') is not True
            or packet.get('candidate_applied_to_actual_runtime') is not False
            or packet.get('accuracy_accepted') is not False):
        raise ValueError('Explicit saved original qualified-ring RF4 initializer candidate required')
    parent = np.asarray(q, float)
    if parent.ndim != 1 or len(parent) < CORE_COUNT or not np.isfinite(parent).all():
        raise ValueError('Complete finite current physical state required')
    original = parent.copy()
    core = parent[:CORE_COUNT]
    pts = int(packet['actual_pts'])
    if pts not in state.bypts:
        raise ValueError('Initializer actual exposure is absent from the prepared source')
    block = state.bypts[pts]
    if (block.bundle.calibration.installation_id != packet['installation_id']
            or pts == int(block.bundle.reference_pts)
            or int(block.bundle.reference_pts) != int(packet['installation_reference_pts'])):
        raise ValueError('Initializer must belong to the same explicit nonreference installation exposure')
    rows = [row for actual, group in block.groups if int(actual) == pts for row in group
            if row.get('original_outer_source_qualification_operator') == STRICT_KIND
            and row.get('physical_curve_id') in ('R2_outer', 'R3_outer')]
    identities = [(row['id'], row.get('scanline_branch')) for row in rows]
    expected = {('R2_outer', 'min_y'), ('R2_outer', 'max_y'),
                ('R3_outer', 'min_y'), ('R3_outer', 'max_y')}
    if len(rows) != 4 or set(identities) != expected or sum(len(row['uv']) for row in rows) != 29:
        raise ValueError('Exact four original qualified R2/R3 branches and 29 immutable UV required')
    authority = packet['source_qualification']
    if any(row.get('original_source_qualification', {}).get('sha256') != authority['sha256']
           for row in rows):
        raise ValueError('Initializer original qualification source SHA differs from the prepared rows')
    if hashlib.sha256(Path(authority['path']).read_bytes()).hexdigest() != authority['sha256']:
        raise ValueError('Original qualified source packet changed')
    evaluator = block.bundle.absolute_evaluator
    if (evaluator is not block.absolute
            or getattr(evaluator, '__name__', '') != 'qualified_original_ring_absolute_value_jac15'):
        raise ValueError('Actual original qualified absolute evaluator must remain installed')
    signature = _source_signature(rows)
    world = block.geometry_values(core)
    if not np.array_equal(world, parent[WORLD_INDICES]):
        raise ValueError('Actual block and declared current world chart disagree')
    moved = [block.geometry.rebind(row, world) for row in rows]
    view = block.start + 4 * block.bundle.index[pts]
    indices = np.arange(view, view + 4)
    report = dict(
        schema='rinkspace.explicit-qualified-ring-RF4-feasible-initialization.v1',
        initializer_packet=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()),
        source_qualification=authority, actual_pts=pts,
        installation_id=packet['installation_id'], reference_pts=int(block.bundle.reference_pts),
        parent_q_sha256=_q_sha(parent), view_parameter_indices=indices.tolist(),
        original_source_signature=signature, source_UV_count=29,
        source_UV_sigma_intervals_roles_and_operators_changed=False,
        geometry_shared_P_mount_PP_Brown_or_other_exposures_changed=False,
        nonlinear_objective_optimized=False, numerical_accepted_step=False,
        parent_q_used_readonly=True, state_q_not_assigned=True, accuracy_accepted=False)
    initialized = parent.copy()
    try:
        camera = block.bundle.camera(block.local(core), pts)
        try:
            value = np.asarray(evaluator(camera, moved, False), float)
        except BoardIntersectionError as exc:
            if exc.reason != 'qualified_outer_source_X_outside_complete_curve_domain':
                raise
            report['original_x_domain_failure'] = dict(
                reason=exc.reason, pair_indices=exc.indices.tolist())
        else:
            if not np.isfinite(value).all():
                raise ValueError('Actual original qualified-ring values are nonfinite')
            validate_mounts(state, parent)
            report.update(applied=False, reason='Current original typed source already valid; saved candidate skipped',
                          changed_parameter_indices=[], initialized_q_sha256=_q_sha(initialized),
                          original_typed_value_calculable=True)
            return initialized, report

        if (_q_sha(parent) != packet['parent_q_sha256']
                or _q_sha(parent[WORLD_INDICES]) != packet['geometry_world66_sha256']
                or not np.array_equal(indices, packet['changed_nonreference_view_indices'])
                or not np.array_equal(parent[indices], packet['original_RF4'])):
            raise ValueError('Invalid state is not the exact saved initializer parent/world/view; candidate refused')
        candidate = np.asarray(packet['candidate_RF4'], float)
        if candidate.shape != (4,) or not np.isfinite(candidate).all():
            raise ValueError('Four finite actual Rodrigues3/logf initializer values required')
        initialized[indices] = candidate
        candidate_camera = block.bundle.camera(block.local(initialized[:CORE_COUNT]), pts)
        if not np.array_equal(np.asarray(candidate_camera)[8:], np.asarray(camera)[8:]):
            raise ValueError('RF4 initializer would change the fixed PP/Brown camera chart')
        value = np.asarray(evaluator(candidate_camera, moved, False), float)
        if not np.isfinite(value).all():
            raise ValueError('Candidate original qualified-ring values are nonfinite')
        validate_mounts(state, initialized)
        if np.any(initialized[np.asarray(block.xyz_slots)][2:] <= 0.):
            raise ValueError('RF4 initializer must preserve the positive fixed-pivot domain')
        margin = float(packet['engineering_FD_interior_margin_px'])
        if not np.isfinite(margin) or margin < 0.:
            raise ValueError('Finite nonnegative stricter numerical domain margin required')
        domains = []
        for row in moved:
            projected, depth, valid = project_central_valid(candidate_camera, np.asarray(row['world'], float))
            if not valid.all() or np.any(depth <= 0.) or not np.isfinite(projected).all():
                raise ValueError('Candidate original curve leaves the actual central-forward domain')
            x = np.asarray(row['uv'])[:, 0]
            low, high = float(projected[:, 0].min()), float(projected[:, 0].max())
            interior = min(float(x.min() - low), float(high - x.max()))
            if interior < margin - 1e-8:
                raise ValueError('Candidate lacks the recorded stricter FD domain interior margin')
            domains.append(dict(id=row['id'], branch=row['scanline_branch'], source_UV_count=len(x),
                                actual_projected_X_domain=[low, high],
                                source_X_range=[float(x.min()), float(x.max())],
                                minimum_X_domain_interior_px=interior))
        changed = np.flatnonzero(initialized != original)
        if (not np.array_equal(parent, original) or not set(changed).issubset(set(indices))
                or _source_signature(rows) != signature):
            raise ValueError('RF4 initializer changed protected state or original source observations')
        report.update(
            applied=True, reason='Exact invalid parent received explicit saved nonreference RF4 candidate',
            changed_parameter_indices=changed.tolist(), initialized_q_sha256=_q_sha(initialized),
            before_RF4=parent[indices].tolist(), after_RF4=initialized[indices].tolist(),
            original_typed_value_calculable=True, original_typed_scalar_rows=len(value),
            normalized_typed_value_RMS=float(np.sqrt(np.mean(value * value))),
            engineering_FD_interior_margin_px=margin,
            margin_is_not_source_domain_expansion_or_user_accuracy_tolerance=True,
            actual_source_domains=domains)
        return initialized, report
    finally:
        # Restore this prepared member's original nominal calibration/local
        # caches.  The caller explicitly installs the returned q afterwards.
        nominal = block.local(core)
        block.bundle.camera(nominal, pts)
