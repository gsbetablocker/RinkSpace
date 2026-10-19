"""Small actual-state native-source tangent-space check; stdout only.

No fitted/candidate state, motion graph, whole-source J or pixels are created.
The native source operators, sigma and curve budgets come from --config.
HELD is an exposure inventory only: no HELD response/fit is inferred from a
FIT-only private-camera tangent space.  This is not a full-source rank test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from calibration.solver_runtime import install, configure_native_runtime
install()
configure_native_runtime(stage='readonly_shared_zoom_source_subspace', native_threads=1)
from rinkspace.vision_camera_probe import FrozenCameraSourceProbe
from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
from calibration.solver_runtime.source_loss import SourceLossWeights, source_irls_weights
from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from z190_physical_camera import ViewState


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(stage, **value):
    print(json.dumps(dict(stage=stage, **value), allow_nan=False,
                     default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x)), flush=True)


def selected(row):
    curve = row['curve']
    return ((curve.get('kind') == 'advertisement_boundary' and curve.get('advertisement_number') in (4, 7))
            or (curve.get('kind') == 'house_ring' and curve.get('side') == 'F'
                and curve.get('row') == 3 and curve.get('ring') == 'outer')
            or (curve.get('kind') == 'current_ice_segment' and curve.get('feature_id') == 'B1'))


def shot_at(pts, shots):
    matches = [i + 1 for i, shot in enumerate(shots)
               if int(shot['start_pts']) <= pts < int(shot['end_pts'])]
    if len(matches) != 1:
        raise ValueError('One original source shot required: ' + str(pts))
    return matches[0]


def change_calibration(calibration, kind, axis, amount):
    """Perturb public physical values, never legacy conditioned camera columns."""
    if kind in ('pivot', 'lever'):
        field = kind + '_xyz'
        value = list(getattr(calibration, field)); value[axis] += amount
        return replace(calibration, **{field: tuple(value)})
    field = {'PPx': 'principal_x_fraction', 'PPy': 'principal_y_fraction',
             'k1': 'k1', 'PPy_logf': 'principal_y_fraction',
             'k1_logf': 'k1', 'axial_logf': 'axial'}[kind]
    degree = 1 if kind.endswith('_logf') else 0
    value = list(getattr(calibration, field))
    value.extend([0.] * max(0, degree + 1 - len(value)))
    if kind in ('PPx', 'PPy', 'PPy_logf'):
        amount /= calibration.output_mode.domain_wh[0 if kind == 'PPx' else 1]
    value[degree] += amount
    return replace(calibration, **{field: tuple(value)})


def contexts(probe, options):
    """Inherit full-factor budget slices before selecting any native rows."""
    base = SimpleNamespace(contract={}, native_board_factor=probe.native['native_board'],
                           native_ice_factor=probe.native['native_ice'])
    integration = SourceWithNativeBaseCurveIntegration(base, options['--native-base-curve-integration-policy'])
    output = []
    for family, factor in probe.native.items():
        payload = integration._payloads[{'native_board': 'additional_real_native_board_edges',
                                         'native_ice': 'additional_real_native_ice_curves'}[family]]
        qualification = getattr(factor, 'source_point_qualification', None)
        active = factor._qualified_active_fit_view() if family == 'native_ice' else None
        active_by_id = {r['id']: r for r in (active.rows if active is not None else factor.rows)}
        offset = 0
        for original in factor.rows:
            indices = (np.arange(len(original['uv'])) if qualification is None
                       else np.flatnonzero(qualification.point_mask(original)))
            stop = offset + 2 * len(indices)
            if selected(original) and len(indices):
                row = active_by_id[original['id']]
                output.append(dict(family=family, factor=factor, row=row,
                    loss=SourceLossWeights(payload.integration_weights[offset:stop],
                        None if payload.working_normalization_multiplier is None else
                        payload.working_normalization_multiplier[offset:stop]),
                    quadratic=factor.quadratic[offset:stop]))
            offset = stop
        if offset != factor.row_count:
            raise ValueError('Complete native budget scalar layout changed')
    if not output:
        raise ValueError('No current selected FIT source')
    return output


def orthogonal_basis(matrix, relative_tolerance):
    norms = np.linalg.norm(matrix, axis=0)
    used = norms > 0.
    if not np.any(used):
        return np.empty((len(matrix), 0)), np.empty(0), used
    u, singular, _ = np.linalg.svd(matrix[:, used] / norms[used], full_matrices=False)
    rank = int(np.sum(singular > relative_tolerance * singular[0]))
    return u[:, :rank], singular, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True, help='Accepted-state directory or NPZ')
    parser.add_argument('--config', type=Path, required=True, help='Exact source-options config to inspect')
    parser.add_argument('--rank-tolerance', type=float, default=1e-8)
    parser.add_argument('--fixed-source-normal', type=Path,
                        help='Saved initial normal containing original canonical fixed columns; default state run')
    args = parser.parse_args()
    if not 0. < args.rank_tolerance < 1.:
        raise ValueError('Finite rank tolerance inside (0,1) required')
    directory = args.state.parent if args.state.is_file() else args.state
    state_path = directory / 'complete_state_UNACCEPTED.npz'
    identity = json.loads((directory / 'state_identity.json').read_text(encoding='utf8'))
    if sha(state_path) != identity['state_npz_sha256'] or sha(identity['camera_path']) != identity['camera_sha256']:
        raise ValueError('Accepted actual state/camera identity changed')
    options = json.loads(args.config.read_text(encoding='utf-8-sig'))['source_options']
    probe = FrozenCameraSourceProbe(state_path, identity['camera_path'], options['--world-init'], options)
    install_source_ice_material_view(probe.binder, max_cache_bytes=32 * 1024**2)
    fixed_path = args.fixed_source_normal or directory.parent.parent / 'first_complete_source_normal.json'
    fixed_packet = json.loads(fixed_path.read_text(encoding='utf8'))
    fixed_canonical = np.asarray(fixed_packet['fixed_parameter_indices'], int)
    with np.load(state_path, allow_pickle=False) as archive:
        declaration = json.loads(str(archive['aux__current_complete_parameter_layout_json'].item()))
    layout = CompleteProgrammeParameterLayout(declaration['original_complete_parameter_count'],
                                             declaration['additional_world_parameter_count'])
    fixed = set(layout.canonical_to_evaluation[fixed_canonical].tolist())
    source = contexts(probe, options)
    shots = json.loads((ROOT / 'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    actuals = sorted({int(item['row']['actual_pts']) for item in source})
    cameras = {at: probe.camera['camera25'][probe.actual_index[at]].copy() for at in actuals}
    views = {at: probe.camera['view_state4'][probe.actual_index[at]].copy() for at in actuals}
    installation = {at: str(probe.camera['installation_id'][probe.actual_index[at]]) for at in actuals}
    offsets = {at: probe.camera['effective_pp_offset_px'][probe.actual_index[at]].copy() for at in actuals}
    for at in actuals:
        rebuilt = probe.calibrations[installation[at]].camera25(ViewState.from_state4(
            views[at], effective_pp_offset_px=offsets[at]))
        if not np.array_equal(rebuilt, cameras[at]):
            raise ValueError('Public camera does not reproduce actual nominal q')
    owner = probe.native['native_board'].world_indices
    if not np.array_equal(owner, probe.native['native_ice'].world_indices):
        raise ValueError('Native geometry owners differ')

    def residual(q=probe.q, current=cameras):
        return np.concatenate([item['factor']._residual(q, [item['row']], cameras=current) for item in source])

    nominal = residual()
    cursor = 0
    for item in source:
        item['start'], item['stop'] = cursor, cursor + 2 * len(item['row']['uv'])
        cursor = item['stop']
    weights = np.concatenate([source_irls_weights(
        item['factor']._residual(probe.q, [item['row']], cameras=cameras),
        item['quadratic'], item['loss']) for item in source])
    root_weights = np.sqrt(weights)
    row_shots = np.concatenate([np.full(2 * len(item['row']['uv']), shot_at(int(item['row']['actual_pts']), shots))
                                for item in source])
    declared = dict()
    for iid in sorted(set(installation.values())):
        cal = probe.calibrations[iid]
        declared[iid] = dict(PPx=cal.principal_x_fraction, PPy=cal.principal_y_fraction,
            k1=cal.k1, k2=cal.k2, axial=cal.axial, tangential=cal.tangential_p1_p2,
            allow_effective_pp_offset=cal.allow_effective_pp_offset,
            q_reference=cal.q_reference, focal_domain=cal.output_mode.domain,
            selected_FIT_exposures=sum(value == iid for value in installation.values()))
    emit('nominal', state=str(state_path.resolve()), state_sha256=sha(state_path),
         config=str(args.config.resolve()), config_sha256=sha(args.config),
         actual_q_public_camera_bitexact=True, current_optics=declared,
         fixed_column_reference=dict(path=str(fixed_path.resolve()), sha256=sha(fixed_path)),
         fixed_evaluation_columns=sorted(fixed),
         selected_FIT_rows=len(source), scalar_rows=len(nominal), FIT_exposures=len(actuals),
         original_sigma_curve_budgets_frozen_IRLS_inherited=True,
         selected_source_scope='native B1 / F3 outer / Ad4 / Ad7 only; no original/motion/prior/guard rank certificate')

    def camera_column(kind, iid, axis, amount, at_only=None):
        current = dict(cameras)
        for at in actuals:
            if installation[at] != iid or (at_only is not None and at != at_only):
                continue
            cal = probe.calibrations[iid]; view = views[at].copy()
            if kind == 'RF4':
                view[axis] += amount
            else:
                cal = change_calibration(cal, kind, axis, amount)
            current[at] = cal.camera25(ViewState.from_state4(view, effective_pp_offset_px=offsets[at]))
        return current

    descriptors = [('world', int(col), 0, 1e-6, None) for col in owner if int(col) not in fixed]
    for at in actuals:
        block = probe.native['native_ice'].blocks.get(at) or probe.native['native_board'].blocks[at]
        start = block.start + 4 * block.bundle.index[at]
        descriptors.extend([('RF4', installation[at], axis, 1e-6 if axis < 3 else 1e-5, at)
                            for axis in range(4) if start + axis not in fixed])
    for iid in sorted(declared):
        at = next(at for at in actuals if installation[at] == iid)
        block = probe.native['native_ice'].blocks.get(at) or probe.native['native_board'].blocks[at]
        descriptors.extend([(kind, iid, axis, 1e-5, None)
                            for kind in ('pivot', 'lever') for axis in range(3)
                            if int((block.xyz_slots if kind == 'pivot' else block.optics_chart.mount_indices)[axis]) not in fixed])
        descriptors.extend([(kind, iid, 0, 1e-3 if kind != 'k1' else 1e-4, None)
                            for index, kind in enumerate(('PPx', 'PPy', 'k1'))
                            if int(block.optics_chart.indices[index]) not in fixed])
    candidates = [(kind, iid, 0, 1e-3 if kind != 'axial_logf' else 1e-5, None)
                  for iid in sorted(declared) for kind in ('PPy_logf', 'k1_logf', 'axial_logf')]

    def derivatives(items, multiplier):
        columns = []
        for kind, iid, axis, step, at in items:
            h = step * multiplier
            if kind == 'world':
                high = probe.q.copy(); low = probe.q.copy()
                high[iid] += h; low[iid] -= h
                delta = residual(high) - residual(low)
            else:
                high = camera_column(kind, iid, axis, h, at)
                low = camera_column(kind, iid, axis, -h, at)
                delta = np.zeros(len(nominal))
                for item in source:
                    actual = int(item['row']['actual_pts'])
                    if installation[actual] != iid or (at is not None and actual != at):
                        continue
                    delta[item['start']:item['stop']] = (
                        item['factor']._residual(probe.q, [item['row']], cameras=high)
                        - item['factor']._residual(probe.q, [item['row']], cameras=low))
            columns.append(root_weights * delta / (2. * h))
        return np.column_stack(columns)

    first = None
    for multiplier in (1., .5):
        base = derivatives(descriptors, multiplier); new = derivatives(candidates, multiplier)
        u, singular, used = orthogonal_basis(base, args.rank_tolerance)
        remaining = new - u @ (u.T @ new)
        residual_left = root_weights * nominal - u @ (u.T @ (root_weights * nominal))
        # Keep each candidate's ORIGINAL norm. Renormalizing a tiny projected
        # column would turn FD roundoff into an apparently independent mode.
        new_norms = np.linalg.norm(new, axis=0)
        nonzero_new = new_norms > 0.
        candidate_u, candidate_s, _ = np.linalg.svd(
            remaining[:, nonzero_new] / new_norms[nonzero_new], full_matrices=False)
        added_rank = int(np.sum(candidate_s > args.rank_tolerance * max(1., candidate_s[0])))
        candidate_u = candidate_u[:, :added_rank]
        # This is an orthogonal energy diagnostic, never a fitted parameter/q.
        energy = float(residual_left @ residual_left)
        aligned = float(np.sum((candidate_u.T @ residual_left) ** 2))
        candidate_rows = []
        for col, (kind, iid, _, _, _) in enumerate(candidates):
            total = float(np.linalg.norm(new[:, col])); independent = float(np.linalg.norm(remaining[:, col]))
            candidate_rows.append(dict(kind=kind, installation=iid,
                independent_to_total_norm=None if total == 0. else independent / total,
                independent_norm=independent, total_norm=total,
                dot_with_base_unexplained_by_shot={str(shot): float(remaining[row_shots == shot, col]
                    @ residual_left[row_shots == shot]) for shot in sorted(set(row_shots.tolist()))}))
        emit('tangent_identifiability', FD_multiplier=multiplier, base_columns=len(descriptors),
             nonzero_base_columns=int(used.sum()), base_rank=u.shape[1],
             base_smallest_retained_singular=None if not u.shape[1] else float(singular[u.shape[1] - 1]),
             added_rank=candidate_u.shape[1], added_singular=candidate_s.tolist(),
             original_frozen_model_residual_energy=float(np.sum((root_weights * nominal) ** 2)),
             base_unexplained_energy=energy, candidate_aligned_energy=aligned,
             candidates=candidate_rows, no_parameters_estimated_or_state_updated=True)
        if first is None:
            first = (base, new)
        else:
            emit('two_scale_derivatives', base_max_abs_delta=float(abs(base - first[0]).max(initial=0.)),
                 candidate_max_abs_delta=float(abs(new - first[1]).max(initial=0.)),
                 rank_tolerance=args.rank_tolerance,
                 branch_nonsmoothness_or_rank_sensitivity_is_not_model_insufficiency=True)
    held = [dict(family=family, row=row) for family, factor in probe.native.items()
            for row in factor.held if selected(row)]
    emit('held_inventory_only', HELD_rows=len(held), by_shot=dict(Counter(str(shot_at(int(x['row']['actual_pts']), shots)) for x in held)),
         same_FIT_exposure_rows=sum(int(x['row']['actual_pts']) in actuals for x in held),
         unseen_exposure_rows=sum(int(x['row']['actual_pts']) not in actuals for x in held),
         HELD_not_used_for_basis_or_parameter_estimation=True,
         no_HELD_prediction_or_zoom_generalization_claim=True,
         unseen_HELD_RF4_requires_original_motion_propagation_not_zero_delta=True)
    emit('complete', source_images_decoded=0, source_images_written=0,
         fitted_or_candidate_states_written=0, matrix_or_report_files_written=0,
         no_physical_domain_or_full_source_feasible_improvement_certified=True,
         effective_log_focal_is_not_measured_optical_zoom=True)


if __name__ == '__main__':
    main()
