"""Read-only reconstruction and source-interval audit for the quadratic rig.

This never constructs an optimizer or edits a running result directory.  An
accepted_states packet is an intermediate diagnostic, not a final camera.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np

from continuous_camera_model import FrameBinding, ShotSpline
from continuous_rig_camera import ContinuousRigCameraModel
from continuous_quadratic_rig_optin import QuadraticRigCameraModel
from continuous_reference_calibration import StaticReferenceBasis
from reference_calibration_source_loader import load_inputs as load_reference_inputs
import reference_calibration_source_loader as reference_loader


TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
DEFAULT_BASELINE = TMP / 'continuous_camera_model_v9_round10/secondary_long_side'
SUPPORTED_GROUPS = frozenset(('main_long_side', 'secondary_long_side'))


def baseline_for_group(group: str) -> Path:
    if group not in SUPPORTED_GROUPS:
        raise ValueError(f'Unsupported quadratic camera group: {group}')
    return TMP/'continuous_camera_model_v9_round10'/group


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values) -> dict:
    a = np.asarray(values, np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Summary requires finite nonempty values')
    return dict(min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def _packet(path: Path) -> tuple[dict[int, np.ndarray], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        keys = np.asarray(z['keys'], np.int64)
        values = np.asarray(z['states'], np.float64)
    if (values.shape != (len(keys), 11) or len(set(map(int, keys))) != len(keys)
            or not np.isfinite(values).all()):
        raise ValueError('Invalid Vector11 state packet')
    return ({int(key): value.copy() for key, value in zip(keys, values)}, digest)


def _camera_packet(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key].copy() for key in z.files}
    n = len(result['pts'])
    if (result['camera25'].shape != (n, 25)
            or result['shot_index'].shape != (n,)
            or result['exposure_representative_pts'].shape != (n,)
            or len(set(map(int, result['pts']))) != n):
        raise ValueError('Camera export has invalid native frame identities')
    return result


def reconstruct(directory: Path, *, state_name: str,
                require_quadratic: bool) -> tuple[dict, object, object, dict, dict]:
    """Use the exact phase roles, including both curvature keys when required."""
    directory = Path(directory).resolve()
    phase_path = directory/'phase_00.json'
    phase = json.loads(phase_path.read_text(encoding='utf8'))
    qspec = phase.get('quadratic_logzoom')
    if require_quadratic:
        if (not isinstance(qspec, dict) or
                qspec.get('schema') != 'rinkspace.optional-quadratic-logzoom-rig.v1'):
            raise ValueError('Quadratic phase/key manifest absent; refuse affine loader')
    elif qspec:
        raise ValueError('Baseline is unexpectedly quadratic')
    state_path = directory/state_name
    states, state_digest = _packet(state_path)
    template_path = directory/('initial_cameras.npz' if require_quadratic
                                else 'shot_cameras.npz')
    template = _camera_packet(template_path)
    if (str(template['source_sha256']) != phase['source_sha256'] or
            str(template['model_sha256']) != phase['model_sha256'] or
            phase['camera_id'] not in SUPPORTED_GROUPS):
        raise ValueError('Programme camera/source/scene provenance differs')
    group = phase['camera_id']
    knots = {int(i): ShotSpline(int(i), tuple(map(int, row['keys'])),
                                tuple(map(float, row['pts'])))
             for i, row in phase['shot_knots'].items()}
    frames = [FrameBinding(int(pts), int(rep), int(shot), group, camera.copy(),
                           float(phase['frame_clock_virtual_pts'][str(int(pts))]))
              for pts, rep, shot, camera in zip(
                  template['pts'], template['exposure_representative_pts'],
                  template['shot_index'], template['camera25'])]
    maps = {name: phase[name] for name in (
        'lens_base_keys', 'lens_slope_keys', 'mount_keys', 'offset_keys',
        'offset_slope_keys', 'brown_keys', 'brown_slope_keys')}
    affine = ContinuousRigCameraModel(
        knots, frames, maps['lens_base_keys'], maps['lens_slope_keys'],
        phase['zoom_reference'], mount_keys=maps['mount_keys'],
        offset_keys=maps['offset_keys'],
        offset_slope_keys=maps['offset_slope_keys'],
        brown_keys=maps['brown_keys'],
        brown_slope_keys=maps['brown_slope_keys'])
    if require_quadratic:
        lens = {str(k): int(v) for k, v in qspec['lens_quadratic_keys'].items()}
        brown = {str(k): int(v) for k, v in qspec['brown_quadratic_keys'].items()}
        model = QuadraticRigCameraModel.from_rig(
            affine, lens_quadratic_keys=lens, brown_quadratic_keys=brown)
        if (set(lens) != {group} or set(brown) != {group}
                or lens[group] not in states or brown[group] not in states
                or set(states) != set(_packet(directory/'prior_initializer_states.npz')[0])):
            raise ValueError('Quadratic node identity or full state keyset changed')
    else:
        model = affine
    # The reference basis has its own frame/C-residual nodes and shares *all*
    # optics/mechanics nodes with the programme, including the new two keys.
    # The historical source loader resolves one relative source-image path
    # against cwd. Run it at its documented repository root and restore cwd.
    previous_cwd = Path.cwd()
    try:
        os.chdir(reference_loader.ROOT)
        inventory = load_reference_inputs()
    finally:
        os.chdir(previous_cwd)
    reference_meta = phase['reference_calibration']
    if (inventory['source_video_sha256'] != phase['source_sha256'] or
            inventory['scene_sha256'] != reference_meta['source_reference_scene_sha256']):
        raise ValueError('Reference bank provenance differs')
    ref_rows = [r for r in inventory['references'] if r['camera_id'] == group]
    by_pts = {int(r['source_pts']): r for r in ref_rows}
    expected = set(map(int, reference_meta['reference_pts']))
    if set(by_pts) != expected or set(map(int, reference_meta['new_graph_nodes'])) != expected:
        raise ValueError('Reference exposure identity differs')
    ref_splines, ref_frames = {}, []
    for ordinal, pts in enumerate(reference_meta['reference_pts']):
        pts = int(pts)
        row = by_pts[pts]
        nodes = reference_meta['new_graph_nodes'][str(pts)]
        if row['image_sha256'] != reference_meta['reference_image_sha256'][str(pts)]:
            raise ValueError('Reference image hash differs')
        control = int(nodes['c_residual_key'])
        shot = 1000 + ordinal
        ref_splines[shot] = StaticReferenceBasis(shot, pts, (control,))
        ref_frames.append(FrameBinding(pts, int(nodes['frame_key']), shot,
                                       group, np.asarray(row['initial_camera25'],
                                                         np.float64).copy()))
    ref_affine = ContinuousRigCameraModel(
        ref_splines, ref_frames, maps['lens_base_keys'], maps['lens_slope_keys'],
        phase['zoom_reference'], mount_keys=maps['mount_keys'],
        offset_keys=maps['offset_keys'],
        offset_slope_keys=maps['offset_slope_keys'],
        brown_keys=maps['brown_keys'],
        brown_slope_keys=maps['brown_slope_keys'])
    ref_model = (QuadraticRigCameraModel.from_rig(
        ref_affine, lens_quadratic_keys=lens, brown_quadratic_keys=brown)
        if require_quadratic else ref_affine)
    missing = {key for m in (model, ref_model) for pts in m.frames
               for key in m.dependencies(pts)} - set(states)
    if missing:
        raise ValueError(f'Camera model depends on absent state keys: {sorted(missing)[:8]}')
    provenance = dict(directory=str(directory), phase_sha256=sha(phase_path),
                      state_path=str(state_path), state_sha256=state_digest,
                      template_path=str(template_path),
                      template_sha256=sha(template_path),
                      source_sha256=phase['source_sha256'],
                      scene_sha256=phase['model_sha256'],
                      reference_image_sha256=reference_meta['reference_image_sha256'])
    return phase, model, ref_model, states, dict(template=template,
                                                provenance=provenance)


def _representatives(template: dict) -> np.ndarray:
    reps = np.unique(np.asarray(template['exposure_representative_pts'], np.int64))
    if not set(map(int, reps)).issubset(set(map(int, template['pts']))):
        raise ValueError('Exposure representative is missing from native frame table')
    return reps


def _decompose(model, states, pts: int) -> dict:
    binding = model.frames[int(pts)]
    group = binding.camera_id
    frame = states[binding.frame_key]
    z = float(frame[6]); dz = z - model.lens_zoom_reference[group]
    R = cv2.Rodrigues(frame[3:6])[0]
    mount = states[model.mount_keys[group]][:3]
    d0 = states[model.offset_keys[group]][:3]
    d1 = states[model.offset_slope_keys[group]][:3]
    mechanical = R.T @ (d0 + dz*d1)
    spline = model.shot_splines[binding.shot_id]
    canonical = binding.pts if binding.spline_pts is None else binding.spline_pts
    residual = sum((weight*states[key][:3] for key, weight in
                    spline.basis_pairs(canonical)), np.zeros(3))
    q = model.physical_state(pts, states)
    if not np.allclose(q[:3], mount+mechanical+residual, atol=1e-11, rtol=0):
        raise ValueError('C decomposition does not reproduce physical camera')
    optics = np.r_[q[6]-q[7], q[8:11]*1000., q[11:15]]
    quadratic = np.zeros(8)
    if isinstance(model, QuadraticRigCameraModel):
        a = states[model.lens_quadratic_keys[group]]
        b = states[model.brown_quadratic_keys[group]]
        quadratic = .5*dz*dz*np.r_[a[6], a[8:11]*1000., b[:4]]
    return dict(z=z, centre=q[:3]*100., mount=mount*100.,
                mechanical=mechanical*100., residual=residual*100.,
                optics=optics, quadratic=quadratic)


def _span(array: np.ndarray) -> list[float]:
    return np.ptp(np.asarray(array, np.float64), axis=0).tolist()


def _quadratic_prior_cost(phase: dict, states: dict[int,np.ndarray]) -> dict:
    """Exactly the four new selected/dummy zero-target factor costs."""
    group=phase['camera_id']
    q=phase['quadratic_logzoom']
    policy=phase['quadratic_priors']
    lens=states[int(q['lens_quadratic_keys'][group])]
    brown=states[int(q['brown_quadratic_keys'][group])]
    lens_columns=np.asarray([6,8,9,10]);brown_columns=np.asarray([0,1,2,3])
    lens_sigma=np.asarray(policy['proposed_lens_curvature_sigma_state'],np.float64)
    brown_sigma=np.asarray(policy['proposed_brown_curvature_sigma'],np.float64)
    if lens_sigma.shape!=(4,) or brown_sigma.shape!=(4,):
        raise ValueError('Curvature prior metadata differs')
    lens_cost=.5*float(np.sum((lens[lens_columns]/lens_sigma)**2))
    brown_cost=.5*float(np.sum((brown[brown_columns]/brown_sigma)**2))
    lens_dummy=.5*float(np.sum(lens[np.setdiff1d(np.arange(11),lens_columns)]**2))
    brown_dummy=.5*float(np.sum(brown[np.setdiff1d(np.arange(11),brown_columns)]**2))
    return dict(lens_selected=lens_cost,brown_selected=brown_cost,
                lens_dummy=lens_dummy,brown_dummy=brown_dummy,
                total=lens_cost+brown_cost+lens_dummy+brown_dummy)


def _export_check(directory: Path, model, ref_model, states, *, completed: bool) -> dict:
    if not completed:
        return dict(available=False,
                    reason='No accepted-state camera25 export during intermediate solve')
    programme_file = directory/'shot_cameras.npz'
    reference_file = directory/'reference_cameras.npz'
    if not programme_file.is_file() or not reference_file.is_file():
        raise FileNotFoundError('Completed camera25 exports are required')
    with np.load(programme_file, allow_pickle=False) as z:
        pts, exported = z['pts'], z['camera25']
    with np.load(reference_file, allow_pickle=False) as z:
        ref_pts, ref_exported = z['pts'], z['camera25']
    if (set(map(int, pts)) != set(model.frames) or
            set(map(int, ref_pts)) != set(ref_model.frames)):
        raise ValueError('Exported frame identities differ from reconstructed model')
    computed = np.asarray([model.camera25(int(p), states) for p in pts])
    ref_computed = np.asarray([ref_model.camera25(int(p), states) for p in ref_pts])
    error = float(np.max(np.abs(exported-computed)))
    ref_error = float(np.max(np.abs(ref_exported-ref_computed)))
    if error > 1e-10 or ref_error > 1e-10:
        raise ValueError(f'Reconstructed camera25 export mismatch: {error}, {ref_error}')
    return dict(available=True, programme_frames=len(pts),
                reference_frames=len(ref_pts), max_programme_abs=error,
                max_reference_abs=ref_error,
                programme_export_sha256=sha(programme_file),
                reference_export_sha256=sha(reference_file))


def summarize(candidate: Path, baseline: Path | None = None, *,
              state_name: str = 'accepted_states.npz',
              scope: str = 'intermediate',
              affine_control: Path | None = None,
              _candidate_loaded=None) -> dict:
    if scope not in ('intermediate', 'completed'):
        raise ValueError('Scope must be intermediate or completed')
    if scope == 'completed' and state_name != 'states.npz':
        raise ValueError('Completed report requires final states.npz')
    candidate = Path(candidate).resolve()
    phase, model, ref_model, states, extra = (_candidate_loaded if _candidate_loaded is not None
        else reconstruct(candidate, state_name=state_name, require_quadratic=True))
    baseline = Path(baseline if baseline is not None else
                    baseline_for_group(phase['camera_id'])).resolve()
    prior_phase, old_model, old_ref_model, old_states, old_extra = reconstruct(
        baseline, state_name='states.npz', require_quadratic=False)
    if (phase['camera_id'] != prior_phase['camera_id'] or
            phase['source_sha256'] != prior_phase['source_sha256'] or
            phase['model_sha256'] != prior_phase['model_sha256']):
        raise ValueError('v9/v10 comparison has different source or scene')
    reps = _representatives(extra['template'])
    old_reps = _representatives(old_extra['template'])
    if not np.array_equal(reps, old_reps):
        raise ValueError('v9/v10 exposure representative set differs')
    current = [_decompose(model, states, int(p)) for p in reps]
    former = [_decompose(old_model, old_states, int(p)) for p in reps]
    current_ref = [_decompose(ref_model, states, int(p)) for p in sorted(ref_model.frames)]
    former_ref = [_decompose(old_ref_model, old_states, int(p))
                  for p in sorted(old_ref_model.frames)]
    if set(ref_model.frames) != set(old_ref_model.frames):
        raise ValueError('v9/v10 reference exposure set differs')
    def rows(items, field): return np.stack([x[field] for x in items])
    dz = rows(current, 'z') - model.lens_zoom_reference[phase['camera_id']]
    qspec = phase['quadratic_logzoom']
    lens = states[int(qspec['lens_quadratic_keys'][phase['camera_id']])]
    brown = states[int(qspec['brown_quadratic_keys'][phase['camera_id']])]
    zoom = rows(current, 'z'); ref_zoom = rows(current_ref, 'z')
    report = dict(schema='rinkspace.quadratic-result-review.v1', scope=scope,
        accuracy_certification=False, source_observations_modified=False,
        candidate=extra['provenance'], baseline=old_extra['provenance'],
        quadratic_node_keys=dict(lens=qspec['lens_quadratic_keys'],
                                 brown=qspec['brown_quadratic_keys']),
        curvature_coefficients=dict(lens_ratio_cx_cy_skew=[float(lens[i]) for i in (6,8,9,10)],
                                    brown4=brown[:4].tolist()),
        exposure_coverage=dict(programme_confirmed_exposures=len(reps),
            programme_z=stats(zoom), reference_exposures=len(ref_zoom),
            reference_z=stats(ref_zoom),
            programme_outside_reference_z=int(np.sum((zoom < ref_zoom.min()) | (zoom > ref_zoom.max())))),
        programme=dict(
            centre_span_scene=_span(rows(current, 'centre')),
            mount_scene=rows(current, 'mount')[0].tolist(),
            mechanical_span_scene=_span(rows(current, 'mechanical')),
            spline_residual_span_scene=_span(rows(current, 'residual')),
            centre_delta_v9_to_v10_norm_scene=stats(np.linalg.norm(
                rows(current,'centre')-rows(former,'centre'),axis=1)),
            optics_v9_to_v10_max_abs=np.max(np.abs(
                rows(current,'optics')-rows(former,'optics')),axis=0).tolist(),
            optics_quadratic_term_span=_span(rows(current,'quadratic')),
            optics_quadratic_term_max_abs=np.max(np.abs(rows(current,'quadratic')),axis=0).tolist()),
        reference=dict(
            centre_delta_v9_to_v10_norm_scene=stats(np.linalg.norm(
                rows(current_ref,'centre')-rows(former_ref,'centre'),axis=1)),
            optics_v9_to_v10_max_abs=np.max(np.abs(
                rows(current_ref,'optics')-rows(former_ref,'optics')),axis=0).tolist(),
            optics_quadratic_term_max_abs=np.max(np.abs(
                rows(current_ref,'quadratic')),axis=0).tolist()),
        centre_model='P + R.T*(d0+d1*dz) + shot spline; all components adjustable',
        optics_order=['fx_over_fy_logratio','cx_px','cy_px','skew_px',
                      'k1','k2','p1','p2'],
        export_camera25_check=_export_check(candidate,model,ref_model,states,
                                             completed=scope=='completed'))
    if scope=='completed':
        old_trace=json.loads((baseline/'solver_trace.json').read_text(encoding='utf8'))
        new_trace=json.loads((candidate/'solver_trace.json').read_text(encoding='utf8'))
        old_total=float(old_trace['final_cost'])
        new_total=float(new_trace['final_cost'])
        new_prior=_quadratic_prior_cost(phase,states)
        original_count=int(phase['original_complete_factor_count'])
        if (original_count != int(prior_phase['graph_factors_with_source_calibration']) or
                int(phase['graph_factors_with_source_calibration']) != original_count+4):
            raise ValueError('Original complete factor inventory is not preserved')
        report['objective_decomposition']=dict(
            historical_v9_total=old_total,quadratic_total=new_total,
            new_quadratic_prior=new_prior,
            original_factor_count=original_count,
            original_factor_set_cost_in_quadratic_solution=new_total-new_prior['total'],
            original_factor_set_reduction_vs_v9=
                old_total-(new_total-new_prior['total']),
            attribution_limit=(
                'Only new curvature-prior cost separates algebraically. '
                'The saved solver trace has no cost by original factor family; '
                'v9 versus quadratic also differs in optimization iterations.'))
        if original_count == 6200:
            report['objective_decomposition'][
                'original_6200_factor_set_cost_in_quadratic_solution'] = (
                    new_total-new_prior['total'])
    if affine_control is not None:
        if scope != 'completed':
            raise ValueError('Affine continuation control requires two completed outputs')
        control = Path(affine_control).resolve()
        ctrl_phase, ctrl_model, ctrl_ref_model, ctrl_states, ctrl_extra = reconstruct(
            control, state_name='states.npz', require_quadratic=False)
        if (ctrl_phase['camera_id'] != phase['camera_id'] or
                ctrl_phase['source_sha256'] != phase['source_sha256'] or
                ctrl_phase['model_sha256'] != phase['model_sha256'] or
                ctrl_phase['factor_counts'] != prior_phase['factor_counts'] or
                ctrl_phase['graph_factors_with_source_calibration'] !=
                    phase['original_complete_factor_count'] or
                ctrl_phase['source_inputs'] != phase['source_inputs'] or
                ctrl_phase.get('user_confirmed_nearboard_top', {}).get('report_sha256') !=
                    phase.get('user_confirmed_nearboard_top', {}).get('report_sha256') or
                not np.array_equal(_representatives(ctrl_extra['template']), reps) or
                set(ctrl_ref_model.frames) != set(ref_model.frames)):
            raise ValueError('Affine continuation is not the same complete source graph')
        candidate_prior,_=_packet(candidate/'prior_initializer_states.npz')
        control_prior,_=_packet(control/'prior_initializer_states.npz')
        new_keys={int(qspec['lens_quadratic_keys'][phase['camera_id']]),
                  int(qspec['brown_quadratic_keys'][phase['camera_id']])}
        if (set(candidate_prior)!=set(control_prior)|new_keys or
                any(not np.array_equal(candidate_prior[k],control_prior[k])
                    for k in control_prior) or
                any(not np.array_equal(candidate_prior[k],np.zeros(11))
                    for k in new_keys)):
            raise ValueError('Original prior targets changed in affine control')
        ctrl = [_decompose(ctrl_model, ctrl_states, int(p)) for p in reps]
        ctrl_ref = [_decompose(ctrl_ref_model, ctrl_states, int(p))
                    for p in sorted(ctrl_ref_model.frames)]
        def cost(directory):
            trace=json.loads((directory/'solver_trace.json').read_text(encoding='utf8'))
            return dict(final_cost=float(trace['final_cost']),
                        solver_trace_sha256=sha(directory/'solver_trace.json'),
                        accepted_steps=sum(step.get('status')=='accepted'
                                           for step in trace['steps']))
        report['affine_continuation_control'] = dict(
            provenance=ctrl_extra['provenance'],
            export_camera25_check=_export_check(
                control,ctrl_model,ctrl_ref_model,ctrl_states,completed=True),
            programme=dict(
                centre_delta_quad_vs_control_norm_scene=stats(np.linalg.norm(
                    rows(current,'centre')-rows(ctrl,'centre'),axis=1)),
                optics_quad_vs_control_max_abs=np.max(np.abs(
                    rows(current,'optics')-rows(ctrl,'optics')),axis=0).tolist(),
                centre_delta_control_vs_v9_norm_scene=stats(np.linalg.norm(
                    rows(ctrl,'centre')-rows(former,'centre'),axis=1)),
                optics_control_vs_v9_max_abs=np.max(np.abs(
                    rows(ctrl,'optics')-rows(former,'optics')),axis=0).tolist()),
            reference=dict(
                centre_delta_quad_vs_control_norm_scene=stats(np.linalg.norm(
                    rows(current_ref,'centre')-rows(ctrl_ref,'centre'),axis=1)),
                optics_quad_vs_control_max_abs=np.max(np.abs(
                    rows(current_ref,'optics')-rows(ctrl_ref,'optics')),axis=0).tolist()),
            objectives=dict(historical_v9=cost(baseline),
                            continued_affine=cost(control),
                            quadratic=cost(candidate)),
            original_prior_targets_bitwise_equal=True,
            interpretation=(
                'Compare quadratic with same-additional-iteration affine control; '
                'v9-to-quadratic change alone confounds model freedom and optimization time. '
                'Neither optimizer cap nor cost establishes source-image accuracy.'))
    return report


def export_intermediate_snapshot(candidate: Path, output: Path,
                                 baseline: Path | None = None) -> dict:
    """Export one accepted-state read and its derived cameras, outside run dir."""
    candidate=Path(candidate).resolve(); output=Path(output).resolve()
    if output.exists() or output.is_relative_to(candidate):
        raise ValueError('Use a fresh independent snapshot directory')
    loaded=reconstruct(candidate,state_name='accepted_states.npz',
                       require_quadratic=True)
    phase,model,ref_model,states,extra=loaded
    report=summarize(candidate,baseline,state_name='accepted_states.npz',
                     scope='intermediate',_candidate_loaded=loaded)
    packet=extra['template']
    pts=np.asarray(packet['pts'],np.int64)
    cameras=np.asarray([model.camera25(int(p),states) for p in pts])
    ref_pts=np.asarray(sorted(ref_model.frames),np.int64)
    refs=np.asarray([ref_model.camera25(int(p),states) for p in ref_pts])
    if (cameras.shape!=(len(model.frames),25) or
            refs.shape!=(len(ref_model.frames),25)
            or not np.isfinite(cameras).all() or not np.isfinite(refs).all()):
        raise ValueError('Intermediate camera reconstruction incomplete')
    output.mkdir(parents=True)
    keys=np.asarray(sorted(states),np.int64)
    np.savez_compressed(output/'accepted_states_snapshot.npz',keys=keys,
                        states=np.asarray([states[int(k)] for k in keys]))
    np.savez_compressed(output/'shot_cameras_intermediate.npz',pts=pts,
        camera25=cameras,shot_index=packet['shot_index'],
        exposure_representative_pts=packet['exposure_representative_pts'],
        source_sha256=phase['source_sha256'],model_sha256=phase['model_sha256'],
        source_accepted_states_sha256=extra['provenance']['state_sha256'],
        scope='intermediate_not_final')
    np.savez_compressed(output/'reference_cameras_intermediate.npz',
        pts=ref_pts,camera25=refs,source_sha256=phase['source_sha256'],
        model_sha256=phase['model_sha256'],
        source_accepted_states_sha256=extra['provenance']['state_sha256'],
        scope='intermediate_not_final')
    report['intermediate_camera25_export']=dict(
        claim='one accepted-state numerical snapshot; not a final camera',
        programme_frames=len(pts),reference_frames=len(ref_pts),
        accepted_states_snapshot_sha256=sha(output/'accepted_states_snapshot.npz'),
        programme_camera_file_sha256=sha(output/'shot_cameras_intermediate.npz'),
        reference_camera_file_sha256=sha(output/'reference_cameras_intermediate.npz'),
        original_accepted_states_sha256=extra['provenance']['state_sha256'])
    with (output/'summary.json').open('x',encoding='utf8') as file:
        json.dump(report,file,ensure_ascii=False,indent=2,allow_nan=False)
    return report


def score_nearboard(candidate: Path, baseline: Path, scene_path: Path,
                    affine_control: Path | None = None) -> dict:
    """Score all 65 frozen original intervals with the actual adaptive factor."""
    from secondary_nearboard_top_source import (
        EDGE_SIGMA_PX, prepare_secondary_nearboard_top)
    from secondary_nearboard_adaptive_curve import (
        AdaptiveNearboardCurve, AdaptiveCurveResolutionError)
    from reference_board_pair import BoardIntersectionError

    candidate = Path(candidate).resolve(); baseline = Path(baseline).resolve()
    phase, model, ref_model, states, extra = reconstruct(
        candidate, state_name='states.npz', require_quadratic=True)
    if phase['camera_id'] != 'secondary_long_side':
        raise ValueError('Near-board source score belongs only to secondary_long_side')
    old_phase, old_model, _, old_states, old_extra = reconstruct(
        baseline, state_name='states.npz', require_quadratic=False)
    if not _export_check(candidate,model,ref_model,
                         states,completed=True)['available']:
        raise ValueError('Candidate final camera export not verified')
    scene_path = Path(scene_path).resolve()
    if sha(scene_path) != phase['model_sha256'] or phase['model_sha256'] != old_phase['model_sha256']:
        raise ValueError('Same v13 world scene is required')
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    template = extra['template']
    aliases = {int(p): int(rep) for p, rep in zip(
        template['pts'],template['exposure_representative_pts'])}
    data = dict(camera_id=phase['camera_id'],
        captured=dict(source_hash=phase['source_sha256']),
        shot_indices=phase['source_inputs']['shot_indices'],
        pts=set(map(int,template['pts'])), scene=scene)
    manifest = phase['user_confirmed_nearboard_top']
    rows, row_meta = prepare_secondary_nearboard_top(
        data, aliases, report_path=Path(manifest['report_path']),
        expected_report_sha256=manifest['report_sha256'])
    if len(rows) != manifest['graph_exposure_factors'] or len(rows) != 65:
        raise ValueError('Frozen near-board exposure-factor count changed')
    def one(m, s):
        distances=[]; factor_rows=[]; failures=[]; graph_cost=0.
        for row in rows:
            pts=int(row['representative_pts'])
            native=tuple(map(int,row['reviewed_native_pts']))
            curve=AdaptiveNearboardCurve(row['world_xyz'],row['measurements'])
            try:
                residual, diagnostic=curve.residual(m.camera25(pts,s),source_keys=native)
            except (BoardIntersectionError, AdaptiveCurveResolutionError) as exc:
                failures.append(dict(representative_pts=pts,
                    source_native_pts=list(native),error_type=type(exc).__name__,
                    reason=getattr(exc,'reason',str(exc))))
                continue
            row_distance=np.linalg.norm(np.asarray(residual).reshape(-1,2),axis=1)
            distances.extend(row_distance.tolist())
            delta=2.*EDGE_SIGMA_PX
            rho=np.where(row_distance<=delta,row_distance**2,
                         2.*delta*row_distance-delta**2)
            factor_cost=.5*float(np.sum(rho))/(EDGE_SIGMA_PX**2*len(row_distance))
            graph_cost+=factor_cost
            factor_rows.append(dict(representative_pts=pts,
                source_native_pts=list(native),source_rows=len(row_distance),
                max_interval_distance_px=float(row_distance.max()),
                inside_interval_rows=int(np.sum(row_distance<=1e-8)),
                exact_graph_cost_component=factor_cost,
                max_winning_sampled_chord_error_px=(
                    float(diagnostic['max_winning_sampled_chord_error_px'])
                    if 'max_winning_sampled_chord_error_px' in diagnostic else None)))
        return dict(valid_factors=len(factor_rows),invalid_factors=failures,
            measured_source_rows=len(distances),interval_hits=int(np.sum(
                np.asarray(distances)<=1e-8)),
            exact_graph_cost_component=(graph_cost if not failures else None),
            interval_distance_px=stats(distances) if distances else None,
            factors=factor_rows)
    report=dict(schema='rinkspace.quadratic-nearboard-source-review.v1',
        scope='completed_same_65_original_source_intervals',
        source_uv_modified=False, physical_world_modified=False,
        candidate_state_sha256=sha(candidate/'states.npz'),
        baseline_state_sha256=sha(baseline/'states.npz'),
        source_sha256=phase['source_sha256'], scene_sha256=phase['model_sha256'],
        scene_path=str(scene_path), nearboard_report_sha256=manifest['report_sha256'],
        nearboard_loader=row_meta, candidate=one(model,states),
        baseline=one(old_model,old_states),
        interpretation='Source interval fit only; not an independent final accuracy certification')
    if affine_control is not None:
        control=Path(affine_control).resolve()
        ctrl_phase,ctrl_model,_,ctrl_states,ctrl_extra=reconstruct(
            control,state_name='states.npz',require_quadratic=False)
        if (ctrl_phase['source_sha256']!=phase['source_sha256'] or
                ctrl_phase['model_sha256']!=phase['model_sha256'] or
                not np.array_equal(_representatives(ctrl_extra['template']),
                                   _representatives(template))):
            raise ValueError('Near-board affine control source identity differs')
        report['affine_continuation_control']=one(ctrl_model,ctrl_states)
        report['affine_control_state_sha256']=sha(control/'states.npz')
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,
                        help='Completed v9 affine group; defaults to matching candidate camera_id')
    parser.add_argument('--affine-control',type=Path,
                        help='Completed same-data affine continuation for fair comparison')
    parser.add_argument('--state',choices=('accepted_states.npz','states.npz'),
                        default='accepted_states.npz')
    parser.add_argument('--scope',choices=('intermediate','completed'),
                        default='intermediate')
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--score-nearboard',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--snapshot-directory',type=Path,
                        help='One intermediate accepted-state camera export; no final claim')
    args=parser.parse_args()
    if args.snapshot_directory is not None:
        if (args.scope!='intermediate' or args.state!='accepted_states.npz'
                or args.score_nearboard or args.affine_control or args.output):
            parser.error('Snapshot requires intermediate accepted state only')
        result=export_intermediate_snapshot(
            args.candidate,args.snapshot_directory,args.baseline)
        print(json.dumps(dict(snapshot=str(args.snapshot_directory.resolve()),
            state_sha256=result['candidate']['state_sha256'],
            programme_frames=result['intermediate_camera25_export']['programme_frames'],
            scope='intermediate_not_final')))
        return
    if args.output is None:
        parser.error('Report --output or --snapshot-directory is required')
    result=summarize(args.candidate,args.baseline,state_name=args.state,
                     scope=args.scope,affine_control=args.affine_control)
    if args.score_nearboard:
        if args.scope!='completed' or args.scene is None:
            parser.error('Nearboard score requires completed states/export and --scene')
        result['nearboard_source_review']=score_nearboard(
            args.candidate,args.baseline,args.scene,args.affine_control)
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(args.candidate.resolve()):
        raise ValueError('Write a fresh independent report outside the running candidate directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as file:
        json.dump(result,file,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(output),scope=args.scope,
                          export_verified=result['export_camera25_check']['available'],
                          nearboard_scored=args.score_nearboard)))


if __name__=='__main__':
    main()
