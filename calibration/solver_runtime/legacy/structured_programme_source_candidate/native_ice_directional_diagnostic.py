"""Record the existing native-ice qualification without changing its decision.

The caller supplies its already-produced values/J/direction.  Only this small
factor's current camera/world branch is replayed.  Original source values,
UV, sigma, roles, support and the driver tolerance remain untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .native_ice_curve_source import (
    _curve_error_on_association, curve_vector_error_full_brown,
    _native_error_with_optical_support,
)
from calibration.solver_runtime.source_scanline_curve import (
    projected_horizontal_scanline_value, horizontal_scanline_curve_chart_response,
)


def _hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _branch(camera, world, observed):
    """Mirror the actual operator's finite support and nearest association."""
    project = curve_vector_error_full_brown.__globals__['project_central_valid']
    nearest = curve_vector_error_full_brown.__globals__['nearest_segments']
    projected, depth, central = project(camera, world)
    start, finish = projected[:-1], projected[1:]
    direction = finish - start
    length2 = np.sum(direction * direction, axis=1)
    midpoint = project(camera, .5 * (world[:-1] + world[1:]))[-1]
    valid = (central[:-1] & central[1:] & midpoint
             & np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[:-1] > 0.) & (depth[1:] > 0.) & (length2 > 1e-4))
    if not valid.any():
        raise ValueError('Diagnostic current curve has no original valid segment')
    index, raw_t, clipped_t, hit = nearest(
        observed, start, direction, length2, valid)
    return dict(projected_uv=projected, depth=depth, central=central,
                midpoint_central=midpoint, segment_valid=valid, length2=length2,
                index=index, raw_t=raw_t, clipped_t=clipped_t, nearest_uv=hit,
                category=np.where(raw_t <= 0., -1, np.where(raw_t >= 1., 1, 0)))


def _extent_branch(camera, world, row):
    """Replay the actual whole-curve extent, with no nearest association."""
    from z190_source_camera_diagnostic import horizontal_extent_value_jac15
    raw = horizontal_extent_value_jac15(camera, dict(row, world=world), False)
    project = horizontal_extent_value_jac15.__globals__['project_central_valid']
    projected, depth, central = project(camera, world)
    choose = np.argmax if row['extent_side'] == 'right' else np.argmin
    index = int(choose(projected[:, 0]))
    category = np.zeros(len(row['uv']), dtype=int)
    support = row.get('_source_optical_support')
    if support is not None:
        signed = np.einsum('ni,ni->n', raw.reshape(-1, 2), support.normals)
        category = np.where(signed <= support.bounds[:, 0], -1,
                            np.where(signed >= support.bounds[:, 1], 1, 0))
    return dict(projected_uv=projected, depth=depth, central=central,
                extent_knots_valid=(central & (depth > 0.)
                                    & np.isfinite(projected).all(axis=1)),
                extent_knot_index=np.full(len(row['uv']), index, dtype=int),
                extent_raw_vector=raw,
                source_normal_support_category=category)


def _scanline_branch(camera, world, row):
    """Replay the actual measured x(y) branch, never a nearest-foot proxy."""
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    projected, depth, central = project_central_valid_full_brown(camera, world)
    raw, association = projected_horizontal_scanline_value(projected, central,
        row['uv'], side='right', with_association=True)
    return dict(projected_uv=projected, depth=depth, central=central,
                scanline_segment_index=association['segment_indices'],
                scanline_station=association['nominal_station'],
                scanline_regular=association['regular'],
                scanline_dy_sign=association['nominal_dy_sign'],
                scanline_duplicate_witness_count=association['duplicate_witness_count'],
                scanline_raw_vector=raw,
                source_normal_support_category=np.zeros(len(row['uv']), dtype=int))


def actual_native_ice_row_branch_signature(camera, world, row):
    """Original row operator's branch signature for the startup FD stencil."""
    kind = row.get('curve_observation_kind')
    if kind == 'axis_scanline_branch':
        branch = _scanline_branch(camera, world, row)
        return (branch['scanline_segment_index'], branch['scanline_regular'],
                branch['scanline_dy_sign'])
    if kind == 'horizontal_extent':
        branch = _extent_branch(camera, world, row)
        return branch['extent_knot_index'], branch['source_normal_support_category']
    branch = _branch(camera, world, row['uv'])
    return branch['index'], branch['category']


def record_native_ice_directional_diagnostic(
        extra, extraq, fullq, eb, direction, step, vp, vm, predicted, out,
        identity):
    """Persist original admission FD and current branches; return JSON metadata.

    This never replaces the caller's observed derivative or its err/allowed.
    ``out`` is the actual run directory under the project's external Data root.
    A compressed small-factor CSR is saved, never the complete source matrix.
    Branch replay errors are recorded so the original qualification can still
    raise its own failure after this function returns.
    """
    # Caller values/J already omit source-review disqualified observations.
    # Replay that same typed active view, retaining immutable full source rows
    # on the caller for truth/review and the original UV identities below.
    original_scalar_rows=sum(2*len(row['uv']) for row in extra.rows)
    active_view=getattr(extra,'_qualified_active_fit_view',lambda:None)()
    if active_view is not None:extra=active_view
    original_point_maps={entry['id']:entry['retained_original_point_indices']
                         for entry in getattr(extra,'_qualified_original_index_map',())}
    out = Path(out).resolve()
    data_root = Path('D:/CodexData/RinkSpace_Data').resolve()
    if out != data_root and data_root not in out.parents:
        raise ValueError('Native-ice numeric diagnostics must remain in Data')
    extraq = np.asarray(extraq, float)
    fullq = np.asarray(fullq, float)
    direction = np.asarray(direction, float)
    vp, vm, predicted = (np.asarray(a, float) for a in (vp, vm, predicted))
    base = np.asarray(eb.r, float)
    step = float(step)
    layout = getattr(extra, 'canonical_parameter_layout', None)
    evaluation_fullq = fullq if layout is None else layout.evaluation_q(fullq)
    if (extraq.shape != (extra.parameter_count,)
            or fullq.ndim != 1 or len(fullq) < len(extraq)
            or not np.array_equal(evaluation_fullq[:len(extraq)], extraq)
            or direction.shape != extraq.shape or not step > 0.
            or any(a.shape != (extra.row_count,)
                   for a in (base, vp, vm, predicted))
            or eb.J.shape != (extra.row_count, extra.parameter_count)):
        raise ValueError('Exact caller native-ice state/value/J dimensions required')
    observed = (vp - vm) / (2. * step)
    delta = np.abs(observed - predicted)
    err = float(delta.max())
    allowed = 3e-3 + 1e-3 * float(np.abs(observed).max())
    rows, cursor = [], 0
    for ordinal, row in enumerate(extra.rows):
        count = 2 * len(row['uv'])
        rows.append(dict(ordinal=ordinal, id=row['id'], split=row['split'],
                         feature_id=row['feature_id'], curve=row['curve'],
                         native_pts=int(row['pts']), actual_pts=int(row['actual_pts']),
                         scalar_start=cursor, scalar_stop=cursor + count,
                         source_uv_count=len(row['uv']),
                         original_source_UV_indices=original_point_maps.get(row['id'],list(range(len(row['uv'])))),
                         source_raw_frame=row['source_raw_frame'],
                         source_file_sha256=row['source_file_sha256']))
        if row.get('curve_observation_kind') in ('horizontal_extent', 'axis_scanline_branch'):
            rows[-1]['curve_observation_kind'] = row['curve_observation_kind']
        cursor += count
    if cursor != extra.row_count:
        raise ValueError('Native-ice row ledger does not cover the caller residual')

    def scalar_identity(index):
        entry = next(row for row in rows
                     if row['scalar_start'] <= index < row['scalar_stop'])
        offset = int(index - entry['scalar_start'])
        return dict(scalar_row=int(index), id=entry['id'],
                    feature_id=entry['feature_id'], native_pts=entry['native_pts'],
                    actual_pts=entry['actual_pts'], source_UV_index=entry['original_source_UV_indices'][offset // 2],
                    active_source_UV_index=offset // 2,
                    coordinate='x' if offset % 2 == 0 else 'y')

    numeric_path = out / 'actual_native_ice_directional_diagnostic.npz'
    report_path = out / 'actual_native_ice_directional_diagnostic.json'
    source_path = Path(__file__).resolve()
    report = dict(
        schema='rinkspace.actual-native-ice-directional-diagnostic.v1',
        identity=dict(identity), complete_q_sha256=_hash_array(fullq),
        component_q_sha256=_hash_array(extraq),
        component_parameter_count=len(extraq), complete_parameter_count=len(fullq),
        complete_parameter_layout=None if layout is None else layout.manifest(),
        component_state_is_explicit_evaluation_prefix=True,
        packet_sha256=extra.contract['packet_sha256'],
        diagnostic_module=dict(path=str(source_path),
                               sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()),
        step=step, original_error=err, original_allowed=allowed,
        original_directional_passed=err <= allowed,
        original_argmax=dict(scalar_identity(int(np.argmax(delta))),
                             observed=float(observed[np.argmax(delta)]),
                             predicted=float(predicted[np.argmax(delta)])),
        row_ledger=rows, scalar_rows=extra.row_count,original_scalar_rows=original_scalar_rows,
        source_review_excluded_scalar_rows=original_scalar_rows-extra.row_count,
        branch_replay_uses_same_qualified_active_layout_as_caller=True,
        source_UV_sigma_roles_support_and_masks_changed=False,
        observed_replaced=False, driver_tolerance_changed=False,
        solver_or_full_source_called=False, full_source_J_saved=False,
        accuracy_accepted=False, numeric_path=str(numeric_path))
    matrix = eb.J.tocsr()
    arrays = dict(
        fullq=fullq, evaluation_fullq=evaluation_fullq, extraq=extraq,
        evaluation_to_canonical_indices=(np.arange(len(fullq)) if layout is None
                                         else layout.evaluation_to_canonical),
        direction=direction, step=np.asarray(step),
        base=base, plus=vp, minus=vm, observed=observed, predicted=predicted,
        original_error=np.asarray(err), original_allowed=np.asarray(allowed),
        scales=np.asarray(extra.scales), quadratic=np.asarray(extra.quadratic),
        J_data=matrix.data, J_indices=matrix.indices, J_indptr=matrix.indptr,
        J_shape=np.asarray(matrix.shape), row_ledger_json=np.asarray(json.dumps(rows)),
        identity_json=np.asarray(json.dumps(dict(identity))))

    def save():
        np.savez_compressed(numeric_path, **arrays)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf8')

    # Preserve the actual qualification before any additional replay can fail.
    save()
    states = dict(base=extraq, plus=extraq + step * direction,
                  minus=extraq - step * direction)
    snapshots = {}
    nominal = {side: np.full(extra.row_count, np.nan)
               for side in ('plus', 'minus')}
    replay_errors = []
    try:
        for label, value in states.items():
            cameras = {pts: np.asarray(camera, float).copy()
                       for pts, camera in extra._cameras(value).items()}
            current = []
            replayed = []
            for ordinal, row in enumerate(extra.rows):
                camera = cameras[row['actual_pts']]
                world = np.asarray(extra._world(value, row['curve']), float).copy()
                if row.get('curve_observation_kind') == 'axis_scanline_branch':
                    branch = _scanline_branch(camera, world, row)
                    raw = branch['scanline_raw_vector']
                elif row.get('curve_observation_kind') == 'horizontal_extent':
                    branch = _extent_branch(camera, world, row)
                    raw = branch['extent_raw_vector']
                else:
                    branch = _branch(camera, world, row['uv'])
                    raw = (branch['nearest_uv'] - row['uv']).reshape(-1)
                support = row.get('_source_optical_support')
                category = np.zeros(len(row['uv']), dtype=int)
                if support is not None:
                    signed = np.einsum('ni,ni->n', raw.reshape(-1, 2), support.normals)
                    category = np.where(signed <= support.bounds[:, 0], -1,
                                        np.where(signed >= support.bounds[:, 1], 1, 0))
                branch['source_normal_support_category'] = category
                current.append((camera, world, branch))
                prefix = label + '_row_' + str(ordinal) + '_'
                arrays[prefix + 'camera25'] = camera
                arrays[prefix + 'world'] = world
                arrays.update({prefix + key: val for key, val in branch.items()})
                if label == 'base':
                    arrays['row_' + str(ordinal) + '_uv'] = row['uv']
                    arrays['row_' + str(ordinal) + '_sigma'] = row['sigma']
                replayed.append(_native_error_with_optical_support(row, raw))
            snapshots[label] = current
            arrays[label + '_branch_replayed_residual'] = (
                np.concatenate(replayed) / extra.scales)
        report['branch_value_replay_max_difference'] = {
            label: float(np.abs(arrays[label + '_branch_replayed_residual'] - actual).max())
            for label, actual in (('base', base), ('plus', vp), ('minus', vm))}
        per_row = []
        for ordinal, (row, ledger) in enumerate(zip(extra.rows, rows)):
            span = slice(ledger['scalar_start'], ledger['scalar_stop'])
            baseline = snapshots['base'][ordinal][2]
            branches = {label: snapshots[label][ordinal][2] for label in states}
            entry = dict(id=row['id'], scalar_start=ledger['scalar_start'],
                         scalar_stop=ledger['scalar_stop'],
                         original_max_absolute_error=float(delta[span].max()),
                         original_argmax=scalar_identity(
                             ledger['scalar_start'] + int(np.argmax(delta[span]))))
            entry['changed_branches'] = {}
            if row.get('curve_observation_kind') == 'axis_scanline_branch':
                association = dict(side='right', observed_uv=row['uv'],
                    segment_indices=baseline['scanline_segment_index'],
                    regular=baseline['scanline_regular'],
                    nominal_dy_sign=baseline['scanline_dy_sign'])
                for side in ('plus', 'minus'):
                    current = branches[side]
                    entry['changed_branches'][side] = dict(
                        scanline_segment_UV=np.flatnonzero(
                            current['scanline_segment_index'] != baseline['scanline_segment_index']).tolist(),
                        scanline_regularity_UV=np.flatnonzero(
                            current['scanline_regular'] != baseline['scanline_regular']).tolist(),
                        central_knots=np.flatnonzero(current['central'] != baseline['central']).tolist())
                    camera, world, _ = snapshots[side][ordinal]
                    try:
                        raw = horizontal_scanline_curve_chart_response(camera, world, row['uv'], association)
                        nominal[side][span] = raw / extra.scales[span]
                    except Exception as exc:
                        replay_errors.append(dict(id=row['id'], side=side,
                            error_type=type(exc).__name__, error=str(exc)))
                fixed = (nominal['plus'][span] - nominal['minus'][span]) / (2. * step)
                entry.update(measured_axis_scanline_actual_operator_replayed=True,
                    scanline_nominal_chart_used_only_for_diagnostic=True,
                    nominal_association_directional_available=bool(np.isfinite(fixed).all()),
                    nominal_association_max_absolute_error=(float(np.abs(fixed-predicted[span]).max())
                        if np.isfinite(fixed).all() else None))
                per_row.append(entry)
                continue
            if row.get('curve_observation_kind') == 'horizontal_extent':
                for side in ('plus', 'minus'):
                    current = branches[side]
                    entry['changed_branches'][side] = dict(
                        extent_knot_UV=np.flatnonzero(
                            current['extent_knot_index'] != baseline['extent_knot_index']).tolist(),
                        central_knots=np.flatnonzero(
                            current['central'] != baseline['central']).tolist(),
                        valid_knots=np.flatnonzero(
                            current['extent_knots_valid'] != baseline['extent_knots_valid']).tolist(),
                        source_normal_support_UV=np.flatnonzero(
                            current['source_normal_support_category'] !=
                            baseline['source_normal_support_category']).tolist())
                    nominal[side][span] = arrays[side + '_branch_replayed_residual'][span]
                extent_observed = (nominal['plus'][span] - nominal['minus'][span]) / (2. * step)
                entry.update(whole_curve_extent_actual_operator_replayed=True,
                             nominal_association_replay_used=False,
                             actual_extent_directional_max_absolute_error=float(
                                 np.abs(extent_observed - predicted[span]).max()))
                per_row.append(entry)
                continue
            associations = (baseline['index'], baseline['raw_t'])
            for side in ('plus', 'minus'):
                current = branches[side]
                entry['changed_branches'][side] = dict(
                    nearest_segment_UV=np.flatnonzero(
                        current['index'] != baseline['index']).tolist(),
                    clamp_category_UV=np.flatnonzero(
                        current['category'] != baseline['category']).tolist(),
                    base_selected_segment_invalid_UV=np.flatnonzero(
                        ~current['segment_valid'][baseline['index']]).tolist(),
                    central_knots=np.flatnonzero(
                        current['central'] != baseline['central']).tolist(),
                    midpoint_central_segments=np.flatnonzero(
                        current['midpoint_central'] != baseline['midpoint_central']).tolist(),
                    valid_segments=np.flatnonzero(
                        current['segment_valid'] != baseline['segment_valid']).tolist(),
                    source_normal_support_UV=np.flatnonzero(
                        current['source_normal_support_category'] !=
                        baseline['source_normal_support_category']).tolist())
                camera, world, _ = snapshots[side][ordinal]
                try:
                    fixed, _ = _curve_error_on_association(
                        camera, world, row['uv'], associations)
                    nominal[side][span] = _native_error_with_optical_support(row, fixed) / extra.scales[span]
                except Exception as exc:
                    replay_errors.append(dict(id=row['id'], side=side,
                                              error_type=type(exc).__name__, error=str(exc)))
            fixed_observed = (nominal['plus'][span] - nominal['minus'][span]) / (2. * step)
            finite = np.isfinite(fixed_observed)
            entry['nominal_association_directional_available'] = bool(finite.all())
            entry['nominal_association_max_absolute_error'] = (
                float(np.abs(fixed_observed - predicted[span]).max())
                if finite.all() else None)
            per_row.append(entry)
        fixed = (nominal['plus'] - nominal['minus']) / (2. * step)
        arrays.update(nominal_association_plus=nominal['plus'],
                      nominal_association_minus=nominal['minus'],
                      nominal_association_observed=fixed)
        finite = np.isfinite(fixed)
        report['nominal_association_directional_available'] = bool(finite.all())
        report['nominal_association_max_absolute_error'] = (
            float(np.abs(fixed - predicted).max()) if finite.all() else None)
        report['nominal_association_within_original_allowed'] = (
            bool(np.abs(fixed - predicted).max() <= allowed) if finite.all() else None)
        report['nominal_association_replay_errors'] = replay_errors
        report['per_row'] = per_row
        report['branch_replay_completed'] = True
    except Exception as exc:
        report['branch_replay_completed'] = False
        report['diagnostic_error'] = dict(error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            extra._cameras(extraq)
            report['base_camera_owner_restored'] = True
        except Exception as exc:
            report['base_camera_owner_restored'] = False
            report['restore_error'] = dict(error_type=type(exc).__name__, error=str(exc))
        save()
    report['numeric_bytes'] = numeric_path.stat().st_size
    report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    return report


def check_actual_native_ice_directional_branches(
        extraq, eb, direction, step, vp, vm, predicted, diagnostic):
    """Qualify only a proved real nominal nearest/clamp/normal-support derivative.

    The raw two-sided FD and its original global tolerance remain recorded.
    A failing UV can use one real side only when its selected finite segment
    and clamp/normal-support categories match the nominal branch, the other
    side changes that branch, and all original validity masks remain identical.  No fixed-
    association replay is used to admit a source or alter its mathematical J.
    Unproved or still-failing UVs retain the original two-sided derivative.
    """
    extraq = np.asarray(extraq, float)
    base = np.asarray(eb.r, float)
    direction = np.asarray(direction, float)
    vp, vm, predicted = (np.asarray(a, float) for a in (vp, vm, predicted))
    step = float(step)
    observed = (vp - vm) / (2. * step)
    checked = observed.copy()
    allowed = 3e-3 + 1e-3 * float(np.abs(observed).max())
    original_error = float(np.abs(observed - predicted).max())
    summary = dict(
        schema='rinkspace.actual-native-ice-piecewise-directional-qualification.v1',
        original_two_sided_max_absolute_error=original_error,
        original_allowed=allowed, checked_allowed=allowed,
        qualified_UV=[], unqualified_UV=[],
        original_validity_masks_changed=False,
        original_source_rows_removed=0, source_UV_sigma_roles_and_support_changed=False,
        fixed_association_values_used_for_qualification=False,
        mathematical_J_or_trial_value_operator_changed=False,
        actual_real_one_sided_values_used=False, accuracy_accepted=False)

    def finish():
        error = float(np.abs(checked - predicted).max())
        summary.update(checked_max_absolute_error=error, passed=error <= allowed)
        return checked, summary

    if original_error <= allowed:
        summary['qualification_needed'] = False
        return finish()
    summary['qualification_needed'] = True
    try:
        if (diagnostic.get('schema') != 'rinkspace.actual-native-ice-directional-diagnostic.v1'
                or diagnostic.get('branch_replay_completed') is not True
                or diagnostic.get('base_camera_owner_restored') is not True):
            raise ValueError('Complete restored current native-ice branch replay required')
        with np.load(diagnostic['numeric_path'], allow_pickle=False) as saved:
            matrix = eb.J.tocsr()
            exact = dict(extraq=extraq, base=base, plus=vp, minus=vm,
                         direction=direction, predicted=predicted,
                         scales=np.asarray(eb.scales), quadratic=np.asarray(eb.quadratic),
                         J_data=matrix.data, J_indices=matrix.indices,
                         J_indptr=matrix.indptr, J_shape=np.asarray(matrix.shape))
            if any(not np.array_equal(saved[key], value) for key, value in exact.items()):
                raise ValueError('Saved native-ice branch proof differs from caller state/value/J')
            if (float(saved['step']) != step or float(saved['original_allowed']) != allowed
                    or float(saved['original_error']) != original_error
                    or json.loads(str(saved['identity_json'].item())) != diagnostic['identity']
                    or json.loads(str(saved['row_ledger_json'].item())) != diagnostic['row_ledger']):
                raise ValueError('Saved native-ice branch proof identity or tolerance differs')
            for label, actual in (('base', base), ('plus', vp), ('minus', vm)):
                if not np.array_equal(saved[label + '_branch_replayed_residual'], actual):
                    raise ValueError('Actual real residual branch replay is not byte-exact')
            summary['real_branch_residual_replay_exact'] = True
            bad_uv = np.abs(observed - predicted).reshape(-1, 2).max(axis=1) > allowed
            for row in diagnostic['row_ledger']:
                start, stop = row['scalar_start'], row['scalar_stop']
                ordinal = row['ordinal']
                for uv in np.flatnonzero(bad_uv[start // 2:stop // 2]):
                    uv = int(uv)
                    scalar = slice(start + 2 * uv, start + 2 * uv + 2)
                    entry = dict(id=row['id'], feature_id=row['feature_id'],
                                 native_pts=row['native_pts'], actual_pts=row['actual_pts'],
                                 source_UV_index=uv,
                                 scalar_rows=[start + 2 * uv, start + 2 * uv + 1],
                                 original_two_sided_max_absolute_error=float(
                                     np.abs(observed[scalar] - predicted[scalar]).max()))
                    if row.get('curve_observation_kind') in ('horizontal_extent', 'axis_scanline_branch'):
                        entry['refusal_reason'] = ('Whole-curve extent retains its actual '
                                                   'two-sided derivative; nearest/clamp '
                                                   'qualification does not apply' if
                                                   row['curve_observation_kind'] == 'horizontal_extent' else
                                                   'Measured-axis scanline retains its actual two-sided derivative; '
                                                   'nearest/clamp qualification does not apply')
                        summary['unqualified_UV'].append(entry)
                        continue
                    branches = {}
                    for label in ('base', 'plus', 'minus'):
                        prefix = label + '_row_' + str(ordinal) + '_'
                        index = int(saved[prefix + 'index'][uv])
                        raw_t = float(saved[prefix + 'raw_t'][uv])
                        category = -1 if raw_t <= 0. else 1 if raw_t >= 1. else 0
                        if category != int(saved[prefix + 'category'][uv]):
                            raise ValueError('Saved real native-ice clamp category is inconsistent')
                        branches[label] = dict(index=index, raw_t=raw_t, category=category,
                                               valid=saved[prefix + 'segment_valid'],
                                               central=saved[prefix + 'central'],
                                               midpoint=saved[prefix + 'midpoint_central'],
                                               source_normal_support_category=int(
                                                   saved[prefix + 'source_normal_support_category'][uv]))
                    nominal = branches['base']
                    segment = nominal['index']
                    reason = None
                    # A real side must retain every nominal material/support
                    # branch. Optical/finite-domain validity changes stay refused.
                    if any(not np.array_equal(branches[side][field], nominal[field])
                           for side in ('plus', 'minus')
                           for field in ('valid', 'central', 'midpoint')):
                        reason = 'Original optical/finite-support validity branch changed'
                    elif any(not branches[side]['valid'][segment]
                             or not branches[side]['valid'][branches[side]['index']]
                             for side in ('base', 'plus', 'minus')):
                        reason = 'Nominal or actual selected finite segment is invalid'
                    stable = {side: branches[side]['index'] == segment
                              and branches[side]['category'] == nominal['category']
                              and branches[side]['source_normal_support_category'] == nominal['source_normal_support_category']
                              for side in ('plus', 'minus')}
                    if reason is None and sum(stable.values()) != 1:
                        reason = ('Both real sides changed the nominal nearest/clamp/normal-support branch'
                                  if not any(stable.values())
                                  else 'Neither real side changed the nominal nearest/clamp/normal-support branch')
                    entry['real_branches'] = {
                        label: dict(segment=value['index'], raw_t=value['raw_t'],
                                    clamp_category=value['category'],
                                    source_normal_support_category=value['source_normal_support_category'])
                        for label, value in branches.items()}
                    if reason is None:
                        side = 'plus' if stable['plus'] else 'minus'
                        slope = ((vp[scalar] - base[scalar]) / step if side == 'plus'
                                 else (base[scalar] - vm[scalar]) / step)
                        error = float(np.abs(slope - predicted[scalar]).max())
                        entry.update(stable_real_side=side,
                                     real_one_sided_observed=slope.tolist(),
                                     predicted=predicted[scalar].tolist(),
                                     one_sided_max_absolute_error=error,
                                     allowed=allowed)
                        if np.isfinite(slope).all() and error <= allowed:
                            checked[scalar] = slope
                            summary['qualified_UV'].append(entry)
                            summary['actual_real_one_sided_values_used'] = True
                            continue
                        reason = 'Stable real one-sided derivative exceeds original allowed'
                    entry['refusal_reason'] = reason
                    summary['unqualified_UV'].append(entry)
    except Exception as exc:
        # No partially authenticated proof may admit an otherwise failing FD.
        checked = observed.copy()
        summary['qualified_UV'] = []
        summary['actual_real_one_sided_values_used'] = False
        summary['proof_refusal'] = dict(error_type=type(exc).__name__, error=str(exc))
    return finish()
