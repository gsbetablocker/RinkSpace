"""One axial(logf) or PPy(logf) absolute response; stdout, no fit or images.

The axial coefficient is working metres per log-focal ratio; PPy uses native
pixels per log-focal ratio. Existing source
operators, retained points, loss budgets, qualification and splits are reused.
HELD responses have a separate diagnostic normalization and never enter FIT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]

import joint_absolute_source_context_20261011 as ctx
import cv2
from verify_original_measured_locations_actual_20261011 import (
    install_on_source_context, current_revised)
from calibration.program_tracking.z190_physical_camera import ViewState
from calibration.solver_runtime.source_loss import (
    SourceLossWeights, loss_weight_arrays, source_irls_weights, weighted_source_cost)

np, a = ctx.np, ctx.a
RUN = a.R / 'okada_review86_q091f_photo_mount_box_release_20261011'
Q11 = '11c84ac8596e4e14f42268051b68d3128d8bc9c0a909bf4366bc52171988b36b'


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False,
                     allow_nan=False, default=a.plain), flush=True)


def physical_native(row):
    curve = row['curve']
    if curve['kind'] == 'house_ring':
        return f"{curve['side']}{curve['row']}:{curve['ring']}"
    if 'feature_id' in curve:
        return str(curve['feature_id'])
    number = curve.get('advertisement_number', curve.get('advertisement_id'))
    return 'Ad' + str(number) if number is not None else str(curve['kind'])


def load_dataset(state_dir=RUN / 'accepted_states' / Q11, config=None):
    """Authenticate one saved state and build only its small absolute factors."""
    state_dir = Path(state_dir).resolve()
    if state_dir.is_file():
        state_dir = state_dir.parent
    if state_dir.parent.name != 'accepted_states':
        raise ValueError('An existing accepted-state directory is required')
    a.RUN, a.CHILD = state_dir.parent.parent, state_dir.name
    a.CONFIG = Path(config or (str(a.RUN) + '.bootstrap/input_config.json')).resolve()
    c = ctx.load_context()
    revised = {}
    if c.options.get('--original-measured-transition-location-source'):
        _, revised = install_on_source_context(
            c, c.options['--original-measured-transition-location-source'])
    groups = []
    for index, item in enumerate(c.originals):
        row, keep = current_revised(c, index, revised)
        if not len(keep):
            continue
        at, split = int(item['group_actual_pts']), item['collection_role']
        c.provider.register_row(item['source_row'], actual_pts=at,
            allow_previously_qualified_held_ad2_representation=split == 'HELD')
        bound = c.provider.rebind(row, c.world)
        binding = c.provider._binding(item['source_row'])[1]
        physical = (binding.get('name') or '+'.join('Ad' + str(n) for n in
                    sorted({n for n, _ in binding.get('ids', [])})) or binding['kind'])
        retired = (item['block'] == 'secondary' and at == 230445824
                   and item['group_order'] == 131 and item['row_order'] == 2)
        if split == 'FIT':
            loss, quadratic = ctx.original_loss(c, index), ctx.original_quadratic(c, index)
        elif split == 'HELD':
            # These original held rows have no production objective budget.
            loss, quadratic = None, np.zeros(2 * len(keep), bool)
        else:
            raise ValueError('Original FIT/HELD collection role changed')
        groups.append(dict(kind='original', family='original_absolute:' + item['block'],
            physical=physical, split=split, pts=at, id=row['id'], row=bound,
            call=c.calls[item['block']], loss=loss, quadratic=quadratic,
            retired=retired, packet_row_index=index, points=len(keep)))
    if sum(g['retired'] for g in groups) != 1:
        raise ValueError('The original sole retired Ad7 material row changed')
    families = [('additional_real_native_ice_curves', c.ice),
                ('additional_real_native_board_edges', c.board),
                ('additional_real_native_board_material_points', c.material),
                ('additional_real_native_board_circles', c.circle)]
    for family, factor in families:
        view = factor._qualified_active_fit_view() if hasattr(factor, '_qualified_active_fit_view') else None
        fit = factor.rows if view is None else view.rows
        payload = c.integration._payloads.get(family)
        count = sum(2 * len(row['uv']) for row in fit)
        weights, normalization = loss_weight_arrays(payload, count)
        offset = 0
        for split, rows in [('FIT', fit), ('HELD', factor.held)]:
            for row in rows:
                size = 2 * len(row['uv'])
                if not size:
                    continue
                loss = None
                if split == 'FIT':
                    stop = offset + size
                    if weights is not None:
                        loss = SourceLossWeights(weights[offset:stop], None if normalization is None
                                                 else normalization[offset:stop])
                    offset = stop
                groups.append(dict(kind='native', family=family, physical=physical_native(row),
                    split=split, pts=int(row['actual_pts']), id=row['id'], row=row,
                    factor=factor, loss=loss, quadratic=np.ones(size, bool),
                    retired=False, points=len(row['uv'])))
        if offset != count:
            raise ValueError('Original native full-family budget was not exhausted')
    for ordinal, factor in enumerate(c.centres):
        offset = 0
        for row in factor.rows:
            stop = offset + 2 * len(row['uv'])
            groups.append(dict(kind='centre', family='direct_manual_sparse_correlated_centre_secondary',
                physical=physical_native(row), split='FIT', pts=int(row['actual_pts']),
                id=row['id'], row=row, factor=factor, mean_scale=factor._manual_centre_mean_scale[offset:stop],
                loss=None, quadratic=np.ones(stop - offset, bool), retired=False,
                points=len(row['uv']), ordinal=ordinal))
            offset = stop
        if offset != len(factor._manual_centre_mean_scale):
            raise ValueError('Manual centre mean normalization changed')
    return c, groups


class AxialCameras:
    """Public camera25 formula; the existing default is the axial coefficient."""
    def __init__(self, c, kind='axial'):
        if kind not in ('axial', 'PPy_logf'):
            raise ValueError('Unknown single camera-response coefficient')
        self.c = c
        self.kind = kind
        self.unit = ('working metres per logf ratio' if kind == 'axial'
                     else 'native pixels per logf ratio')
        self.cache = {}
        self.installations = {}
        self.x = {}
        for at, block in c.state.bypts.items():
            calibration = block.bundle.calibration
            begin = block.start + 4 * block.bundle.index[at]
            view = c.q[begin:begin + 4]
            self.installations[at] = calibration.installation_id
            self.x[at] = float(view[3] - np.log(1920.) - calibration.q_reference)
        self.names = sorted(set(self.installations.values()))

    def camera(self, at, installation, amount):
        at = int(at)
        amount = float(amount) if self.installations[at] == installation else 0.
        key = at, amount
        if key not in self.cache:
            block = self.c.state.bypts[at]
            calibration = block.bundle.calibration
            begin = block.start + 4 * block.bundle.index[at]
            v = self.c.q[begin:begin + 4]
            field = 'axial' if self.kind == 'axial' else 'principal_y_fraction'
            coefficients = list(getattr(calibration, field))
            coefficients.extend([0.] * max(0, 2 - len(coefficients)))
            coefficients[1] += amount / (self.c.provider.scene_unit_m
                                         if self.kind == 'axial' else 1080.)
            changed = replace(calibration, **{field: tuple(coefficients)})
            camera = changed.camera25(ViewState(tuple(v[:3]), float(v[3] - np.log(1920.))))
            nominal = ctx.camera(self.c, self.c.q, at)
            if amount == 0.:
                if np.max(abs(camera - nominal)) > 1e-11:
                    raise ValueError('Public camera does not reproduce the saved actual state')
                if self.kind == 'PPy_logf':
                    if not np.array_equal(camera, nominal):
                        raise ValueError('Zero PPy coefficient is not bit-exact to the saved camera')
            elif self.kind == 'PPy_logf':
                other = np.arange(len(camera)) != 9
                if not np.array_equal(camera[other], nominal[other]):
                    raise ValueError('PPy response changed another camera quantity')
                if abs(camera[9] - nominal[9] - amount * self.x[at]) > 5e-13:
                    raise ValueError('PPy native-pixel displacement differs from a*x')
            elif not np.array_equal(camera[3:], nominal[3:]):
                raise ValueError('Axial response changed a non-centre camera quantity')
            if amount != 0. and self.kind == 'axial':
                expected = (cv2.Rodrigues(v[:3])[0].T[:, 2] * amount * self.x[at]
                            / self.c.provider.scene_unit_m)
                if np.max(abs((camera[:3] - nominal[:3]) - expected)) > 5e-14:
                    raise ValueError('Public axial camera has a different physical centre displacement')
            self.cache[key] = camera
        return self.cache[key]

    def inventory(self, installation, actuals):
        selected = sorted(p for p in actuals if self.installations[p] == installation)
        values = np.asarray([self.x[p] for p in selected])
        result = dict(exposures=len(selected), q_reference=next(b.bundle.calibration.q_reference
                    for b in self.c.state.blocks if b.bundle.calibration.installation_id == installation),
                    x_range=[float(values.min()), float(values.max())]) if selected else dict(exposures=0)
        if selected:
            if self.kind == 'axial':
                result['centre_displacement_norm_working_m_per_unit_a_range'] = [
                    float(abs(values).min()), float(abs(values).max())]
            else:
                result['PPy_displacement_native_pixels_per_unit_a_range'] = [
                    float(values.min()), float(values.max())]
        return result


def source_value(c, group, cameras, installation, amount):
    camera = cameras.camera(group['pts'], installation, amount)
    if group['kind'] == 'original':
        result = group['call'](camera, [group['row']], False)
    else:
        factor = group['factor']; cams = {group['pts']: camera}
        if factor is c.material:
            with c.placement.pattern_placement_context(c.q):
                result = factor._residual(c.q[:factor.parameter_count], [group['row']], cameras=cams)
        elif factor is c.circle:
            result = factor._residual(c.q[:factor.parameter_count], [group['row']], cameras=cams)
        else:
            result = factor._residual(c.q, [group['row']], cameras=cams)
        if group['kind'] == 'centre':
            result = result / group['mean_scale']
    result = np.asarray(result, float)
    if result.shape != group['quadratic'].shape or not np.isfinite(result).all():
        raise ValueError('Original source value shape or finite domain changed')
    return result


def response_column(c, groups, installation, *, h=1e-5, cameras=None, include_column=False):
    """Return one installation's true-value h/h2 FD, never a complete J."""
    if not np.isfinite(h) or h <= 0.:
        raise ValueError('Positive finite coefficient FD step required')
    cameras = cameras or AxialCameras(c)
    if installation not in cameras.names:
        raise ValueError('Unknown actual physical installation')
    records = []
    for ordinal, group in enumerate(groups):
        if cameras.installations[group['pts']] != installation:
            continue
        r = source_value(c, group, cameras, installation, 0.)
        values = [source_value(c, group, cameras, installation, sign * step)
                  for step in (h, h / 2.) for sign in (1., -1.)]
        j = (values[0] - values[1]) / (2 * h)
        j2 = (values[2] - values[3]) / h
        quad, loss = group['quadratic'], group['loss']
        if group['retired']:
            cost = gradient = gradient2 = cost_fd = norm = 0.
        else:
            cost = weighted_source_cost(r, quad, loss)
            weight = source_irls_weights(r, quad, loss)
            gradient, gradient2 = float((weight * r) @ j), float((weight * r) @ j2)
            cost_fd = (weighted_source_cost(values[2], quad, loss)
                       - weighted_source_cost(values[3], quad, loss)) / h
            norm = float(np.sqrt(weight @ j2**2))
        records.append(dict(family=group['family'], physical=group['physical'], split=group['split'],
            pts=group['pts'], id=group['id'], points=group['points'], retired=group['retired'],
            baseline_cost=cost, gradient=gradient2, gradient_h=gradient,
            cost_FD_gradient_h2=cost_fd, weighted_column_l2=norm,
            column_h_h2_max_absolute_error=float(abs(j - j2).max(initial=0.)),
            column_h_h2_relative_l2_error=float(np.linalg.norm(j - j2) / max(np.linalg.norm(j2), 1e-12)),
            scalar_rows=len(r)))
        if include_column:
            records[-1].update(source_normalized_value=r, source_normalized_column_h=j,
                               source_normalized_column_h2=j2)
        if (group['kind'] == 'native' and
                group['row'].get('curve_observation_kind') == 'axis_scanline_branch'):
            sigma = np.asarray(group['row']['sigma'])[:, None]
            records[-1].update(axis_scanline_pixel_column_h2=(j2.reshape(-1, 2) * sigma).tolist(),
                               axis_scanline_signed_pixel_residual=(r.reshape(-1, 2) * sigma).tolist())
        if (ordinal + 1) % 2000 == 0:
            emit('installation_value_progress', installation=installation, source_groups_processed=ordinal + 1)
    return records


def finite_response(c, groups, installation, *, coefficient=.01, cameras=None):
    """Actual +/- fixed coefficient values, with no derivative or fitting."""
    if not np.isfinite(coefficient) or coefficient <= 0.:
        raise ValueError('A fixed finite positive response coefficient is required')
    cameras = cameras or AxialCameras(c)
    result, scanlines = [], []
    for group in groups:
        scanline = (group['kind'] == 'native' and
                    group['row'].get('curve_observation_kind') == 'axis_scanline_branch')
        if cameras.installations[group['pts']] != installation or (group['split'] != 'FIT' and not scanline):
            continue
        values = [source_value(c, group, cameras, installation, amount)
                  for amount in (0., coefficient, -coefficient)]
        if scanline:
            sigma = np.asarray(group['row']['sigma'])[:, None]
            pixels = [r.reshape(-1, 2) * sigma for r in values]
            scanlines.append(dict(id=group['id'], pts=group['pts'], split=group['split'],
                original_uv=group['row']['uv'], original_sigma=group['row']['sigma'],
                baseline_signed_pixel_residual=pixels[0],
                plus_pixel_response=pixels[1] - pixels[0], minus_pixel_response=pixels[2] - pixels[0]))
        if group['split'] != 'FIT':
            continue
        costs = [0.] * 3 if group['retired'] else [
            weighted_source_cost(r, group['quadratic'], group['loss']) for r in values]
        result.append(dict(family=group['family'], physical=group['physical'], id=group['id'],
            pts=group['pts'], retired=group['retired'], points=group['points'],
            before=costs[0], plus=costs[1], minus=costs[2],
            plus_delta=costs[1] - costs[0], minus_delta=costs[2] - costs[0]))
    output = []
    grouped = defaultdict(list)
    for row in result:
        grouped[row['family'], row['physical']].append(row)
    for (family, physical), rows in sorted(grouped.items()):
        output.append(dict(family=family, physical=physical, operators=len(rows),
            points=sum(r['points'] for r in rows),
            **{name: sum(r[name] for r in rows)
               for name in ('before', 'plus', 'minus', 'plus_delta', 'minus_delta')}))
    return dict(groups=output, listed_absolute_FIT_subtotal={name: sum(r[name] for r in result)
                for name in ('before', 'plus', 'minus', 'plus_delta', 'minus_delta')},
                operators=len(result), retired_operators=sum(r['retired'] for r in result),
                axis_scanline_actual_pixel_responses=scanlines)


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['split'], r['family'], r['physical']].append(r)
    result = []
    for (split, family, physical), rows in sorted(groups.items()):
        gradient = sum(r['gradient'] for r in rows)
        result.append(dict(split=split, family=family, physical=physical, operators=len(rows),
            points=sum(r['points'] for r in rows), scalar_rows=sum(r['scalar_rows'] for r in rows),
            baseline_cost=sum(r['baseline_cost'] for r in rows), gradient=gradient,
            gradient_h=sum(r['gradient_h'] for r in rows),
            cost_FD_gradient_h2=sum(r['cost_FD_gradient_h2'] for r in rows),
            locally_helping_a_sign=('-' if gradient > 0 else '+' if gradient < 0 else 'zero'),
            weighted_column_l2=float(np.sqrt(sum(r['weighted_column_l2']**2 for r in rows))),
            column_h_h2_max_absolute_error=max(r['column_h_h2_max_absolute_error'] for r in rows),
            column_h_h2_max_relative_l2_error=max(r['column_h_h2_relative_l2_error'] for r in rows),
            retired_operators=sum(r['retired'] for r in rows)))
    return result


def verify_saved_baseline(c, groups, cameras):
    """Check reconstructed budgets against the actual saved production costs."""
    values = defaultdict(float)
    for group in groups:
        if group['split'] != 'FIT' or group['retired']:
            continue
        installation = cameras.installations[group['pts']]
        r = source_value(c, group, cameras, installation, 0.)
        key = (group['family'], group.get('ordinal', 0))
        values[key] += weighted_source_cost(r, group['quadratic'], group['loss'])
    checks = []
    for family in ('additional_real_native_ice_curves', 'additional_real_native_board_edges',
                   'additional_real_native_board_material_points', 'additional_real_native_board_circles',
                   'direct_manual_sparse_correlated_centre_secondary'):
        for ordinal, saved in enumerate(c.ledgers[family]):
            current, gold = values[family, ordinal], saved['normalized_cost']
            if abs(current - gold) > 2e-8:
                raise ValueError('Original actual native/centre source budget mismatch: ' + family)
            checks.append(dict(family=family, ordinal=ordinal, actual_cost=current,
                               saved_cost=gold, error=current - gold))
    for block in ('main1', 'main7'):
        receipt = a.read(a.RUN / f'actual_trial_{block}_source_row_cost_pass_0001.json')
        gold = next(r['actual_after_cost'] for r in receipt['actual_source_partition']
                    if r['family'] == 'absolute_original_FIT')
        current = values['original_absolute:' + block, 0]
        if abs(current - gold) > 2e-8:
            raise ValueError('Original saved absolute partition mismatch: ' + block)
        checks.append(dict(family='original_absolute:' + block, actual_cost=current,
                           saved_cost=gold, error=current - gold))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=RUN / 'accepted_states' / Q11)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--installation', help='Comma-separated installation IDs or zero-based ordinals; default all three')
    parser.add_argument('--kind', choices=('axial', 'PPy_logf'), default='axial')
    parser.add_argument('--h', type=float)
    parser.add_argument('--compact', action='store_true', help='Small per-physical-group stdout summary')
    parser.add_argument('--baseline-only', action='store_true', help='Authenticate values/budgets and camera ranges without FD')
    parser.add_argument('--finite-coefficient', type=float,
                        help='Evaluate only actual +/- this fixed coefficient; no FD')
    parser.add_argument('--also-finite-coefficient', type=float,
                        help='Evaluate actual +/- this fixed coefficient after the single FD column')
    args = parser.parse_args(); started = time.perf_counter()
    if args.finite_coefficient is not None and args.also_finite_coefficient is not None:
        parser.error('Choose either finite-only or FD plus finite responses')
    if args.h is None:
        args.h = 1e-5 if args.kind == 'axial' else 1e-3
    c, groups = load_dataset(args.state, args.config); cameras = AxialCameras(c, kind=args.kind)
    emit('absolute_context_loaded', q_sha256=c.identity['q_sha256'], camera_error=c.camera_error,
         load_seconds=c.load_seconds, operators_by_split=dict(Counter(g['split'] for g in groups)),
         original_FIT_operators=sum(g['kind'] == 'original' and g['split'] == 'FIT' for g in groups),
         original_FIT_points=sum(g['points'] for g in groups if g['kind'] == 'original' and g['split'] == 'FIT'),
         source_images_decoded=0, source_images_written=0)
    emit('actual_saved_source_budget_replay', checks=verify_saved_baseline(c, groups, cameras))
    selected = cameras.names if args.installation is None else [
        cameras.names[int(name)] if name.isdigit() else name for name in args.installation.split(',')]
    for installation in selected:
        emit(args.kind + '_coefficient_inventory', installation=installation,
             all_programme_exposures=cameras.inventory(installation, c.state.bypts),
             actual_absolute_source_exposures=cameras.inventory(installation, {g['pts'] for g in groups}),
             coefficient_unit=cameras.unit, fd_h=args.h, fd_h2=args.h / 2.,
             coefficient_formula=('C_original + R.T @ ez * a * (logf-log(1920)-q_reference) / scene_unit_m'
                if args.kind == 'axial' else 'principal_y_fraction[1] += a/1080; native PPy delta = a*x'),
             **({'centre_formula': 'C_original + R.T @ ez * a * (logf-log(1920)-q_reference) / scene_unit_m'}
                if args.kind == 'axial' else {}))
        if args.baseline_only:
            continue
        if args.finite_coefficient is not None:
            emit('finite_' + args.kind + '_absolute_response', installation=installation,
                 coefficient=args.finite_coefficient, coefficient_unit=cameras.unit,
                 **({'coefficient_working_m_per_logf': args.finite_coefficient}
                    if args.kind == 'axial' else {}),
                 **finite_response(c, groups, installation, coefficient=args.finite_coefficient, cameras=cameras),
                 no_derivative_or_optimization=True,
                 omitted_families=['motion', 'patterns', 'tracks', 'wide', 'external', 'global_priors', 'caps'])
            continue
        rows = response_column(c, groups, installation, h=args.h, cameras=cameras)
        summary = summarize(rows)
        if args.compact:
            summary = [{k: r[k] for k in ('split', 'family', 'physical', 'operators', 'points',
                       'baseline_cost', 'gradient', 'locally_helping_a_sign')} for r in summary]
        fit = [r for r in rows if r['split'] == 'FIT']
        emit(args.kind + '_absolute_source_response', installation=installation, groups=summary,
             listed_absolute_FIT_subtotal_gradient=sum(r['gradient'] for r in fit),
             listed_absolute_FIT_subtotal_gradient_h=sum(r['gradient_h'] for r in fit),
             listed_absolute_FIT_subtotal_cost_FD_gradient_h2=sum(r['cost_FD_gradient_h2'] for r in fit),
             maximum_absolute_column_h_h2_difference=max(r['column_h_h2_max_absolute_error'] for r in rows),
             worst_h_h2_rows=sorted(rows, key=lambda r: r['column_h_h2_max_absolute_error'], reverse=True)[:3],
             HELD_normalization='Original supported sigma-normalized value/loss diagnostic; no FIT budget and never in FIT totals',
             HELD_identity_scope='Original declared producer/packet identity, not newly qualified independent truth',
             gradient_scope='Subtotal of listed absolute sources only',
             omitted_families=['motion', 'patterns', 'tracks', 'wide', 'external', 'global_priors', 'caps'],
             no_other_parameter_profile_or_weight_changed=True, no_optimization_or_full_J=True)
        target = [r for r in rows if r['family'] == 'additional_real_native_ice_curves'
                  and r['physical'] in ('B1', 'F3:outer')]
        emit('individual_B1_F3_source_rows', installation=installation, rows=target)
        if args.also_finite_coefficient is not None:
            emit('finite_' + args.kind + '_absolute_response', installation=installation,
                 coefficient=args.also_finite_coefficient, coefficient_unit=cameras.unit,
                 **finite_response(c, groups, installation, coefficient=args.also_finite_coefficient, cameras=cameras),
                 no_optimization=True,
                 omitted_families=['motion', 'patterns', 'tracks', 'wide', 'external', 'global_priors', 'caps'])
    emit(args.kind + '_absolute_diagnostic_complete', seconds=time.perf_counter() - started,
         saved_report=False, new_source_images_or_states=False,
         conditional_one_column_not_global_model_identifiability_or_accuracy=True)


if __name__ == '__main__':
    main()
