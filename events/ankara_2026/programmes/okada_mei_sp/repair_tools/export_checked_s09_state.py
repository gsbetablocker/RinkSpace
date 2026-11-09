
"""Export the saved local repair through the native complete-programme path."""
from copy import copy
from pathlib import Path
import json
import sys
import time

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.body_clock import time_base
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.programme_com import acceleration_operator, export_com_diagnostics
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_delivery import export_candidate
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

project, prepared, source, previous, parameter, stage, previous_param = map(Path, sys.argv[1:])
started = time.perf_counter()
parent = load(previous_param)
saved = load(parameter)
ref, camera, camera_info, document, _ = load_reference(project, prepared, source)
validate_checkpoint_identity(saved, ref.fixed_checkpoint_identity)
for key, path in [('prepared_input_sha256',prepared/'input.npz'),('prepared_observations_sha256',prepared/'observations.npz'),
                  ('source_evidence_sha256',source)]:
    if str(saved[key]) != digest(path):
        raise ValueError('Saved identity changed: ' + key)
np.testing.assert_array_equal(saved['x'][:17], parent['x'][:17])
for k in ('absolute_knots','relative_skate_knots','relative_skate_rotation_controls','camera_binding_json',
          'sequence_contact_modes','sequence_contact_status'):
    np.testing.assert_array_equal(saved[k],parent[k],err_msg=k)
knots = saved['absolute_knots']
P0,P1 = parent['absolute_position_controls'], saved['absolute_position_controls']
Q0,Q1 = parent['absolute_rotation_controls'], saved['absolute_rotation_controls']
E0,E1 = parent['absolute_rotation_edge_reference_quaternion'], saved['absolute_rotation_edge_reference_quaternion']
delta_edges = np.any(E0!=E1,axis=(1,2))
delta = np.any(P0!=P1,axis=1)|np.any(Q0!=Q1,axis=(1,2,3))
changed = np.union1d(np.flatnonzero(delta),np.r_[np.flatnonzero(delta_edges),np.flatnonzero(delta_edges)+1])
def affected(t):
    span = np.clip(np.searchsorted(knots,t,side='right')-4,0,len(P0)-4)
    return np.isin(span[:,None]+np.arange(4),changed).any(axis=1)

stage.mkdir(parents=True,exist_ok=False)
fitted = stage/'fitted'
fitted.mkdir()
current = absolute_reference(ref,curve_from_packet(saved),root=prepared)
current.relative_skate_curve = shoe_curve_from_packet(saved)
current.binding = copy(ref.binding)
current.binding.reference = current
current.binding.x = saved['x'][:17].copy()
current.binding._parameter_key = current.binding._packet = None
obs = load(prepared/'observations.npz')
pts = ref.p['pts']
times = ref.times
tb = time_base(ref.p)
np.testing.assert_array_equal(pts,np.arange(283136,2415104+256,256))
np.testing.assert_array_equal(tb,[1,12800])
np.testing.assert_array_equal(pts,obs['pts'])
exposure_affected = affected(times)
tau = camera.readout_times(np.arange(len(times)))
node_times = times[:,None]+tau[:,None]*np.linspace(-.5,.5,5)
projection_affected = affected(node_times.ravel()).reshape(len(times),5).any(axis=1)
print(json.dumps(dict(event='programme_export',native=len(pts),native_changed_support=int(exposure_affected.sum()),
                      rolling_shutter_changed_support=int(projection_affected.sum()))),flush=True)
packet = export_candidate(prepared,stage,current,camera,camera_info,document.get('supports',[]),write=False)
native = current.at(times)
zero = np.zeros((len(times),69))
body = fk_state(current,native,zero)
mass = ContinuousMassGeometry(current)
centers, axes = mass.evaluate(native,zero)
packet.update(export_com_diagnostics(mass,times,centers,axes))
quadrature = load(previous/'com_physics_quadrature.npz')
qclock = quadrature['evaluation_time_s']
q_mask = affected(qclock)
qcenters,_ = mass.evaluate(current.at(qclock[q_mask]),np.zeros((int(q_mask.sum()),69)))
quadcom = quadrature['com_world_m'].copy()
quadcom[q_mask] = mass.com(qcenters)
quadrature['com_world_m'] = quadcom
quadrature['acceleration_world_mps2'] = acceleration_operator(qclock) @ quadcom
quadrature['parameter_sha256'] = np.array(digest(parameter))
quadrature['source_evidence_sha256'] = np.array(digest(source))
qpath = stage/'com_physics_quadrature.npz'
with qpath.open('xb') as handle:
    np.savez_compressed(handle,**quadrature)
packet.update(
    com_physics_quadrature_path=np.array(str(qpath)),com_physics_quadrature_sha256=np.array(digest(qpath)),
    native_body_local_R_mid=body['local_R'],native_body_FK_world_R_mid=body['Q'][:,None]@body['frames'],
    source_action_ledger_sha256=np.array(digest(source)),
    P_velocity_world_mps=current.root_derivative(times,1)*current.binding.ratio,
    source_shot_ids=ref.p['shot_ids'],
    trajectory_evidence_status=np.where(obs['use'].sum(1)>=4,'source_constrained_estimate','temporal_inference'),
    position_uncertainty_status=np.full(len(times),'conditional_scale_and_depth; no independent accuracy acceptance'),
    ice_plane_known=np.array(ref.ice_plane_known),
    sequence_contact_modes=saved['sequence_contact_modes'],sequence_contact_status=saved['sequence_contact_status'],
    shared_betas=ref.p['shared_betas'],shared_body_scale=ref.p['body_scale'],
    source_body_model_joint_ids=body_joint_assignment(obs),
    body_joint_assignment_provenance_json=obs['body_joint_assignment_provenance_json'],
    lsp_revision_parameter_sha256=np.array(digest(parameter)),
    lsp_revision_support_mask=exposure_affected)
np.testing.assert_array_equal(packet['pts'],pts)
np.testing.assert_array_equal(packet['source_pts'],pts)
np.testing.assert_allclose(packet['native_body_local_R_mid'],packet['local_R_mid'],atol=2e-12,rtol=0)
assert json.loads(str(packet['camera_binding_json']))['state_sha256']==camera_info['state_sha256']

# Prove that the displayed state stays the same outside exact spline support.
fields = ('P_world_mid','Q_world_mid','native_body_local_R_mid','native_body_FK_world_R_mid',
          'hand_local_R_mid','COCO_world_mid','skate_R_world_mid','skate_lower_world_mid',
          'skate_upper_world_mid','skate_reference_world_mid','com_world_mid',
          'P_velocity_world_mps','sequence_contact_modes','sequence_contact_status')
off_support = {}
projection_fields=('COCO_uv','equipment_uv','world_at_row','source_row_time')
with np.load(previous/'fitted/result.npz',allow_pickle=False) as old:
    for names,selected,tolerance in ((fields,exposure_affected,2e-9),(projection_fields,projection_affected,3e-7)):
        for key in names:
            new=packet[key][~selected]
            previous=old[key][~selected]
            exact=np.array_equal(new,previous)
            off_support[key]=dict(bitwise=bool(exact))
            if not exact:
                np.testing.assert_allclose(new,previous,rtol=0,atol=tolerance,err_msg=key)
                off_support[key]['max_abs_difference']=float(np.max(abs(new-previous)))
    before_cavity = old['foot_cavity_max_violation_m_mid'][exposure_affected]
    after_cavity = packet['foot_cavity_max_violation_m_mid'][exposure_affected]

pending=fitted/'result.pending.npz'
with pending.open('xb') as handle:
    np.savez_compressed(handle,**packet)
result=fitted/'result.npz'
pending.rename(result)
intervals=np.split(np.flatnonzero(exposure_affected),np.flatnonzero(np.diff(np.flatnonzero(exposure_affected))>1)+1)
report=dict(
    result=str(result),result_sha256=digest(result),parameter=str(parameter),parameter_sha256=digest(parameter),
    parent_parameter_sha256=digest(previous_param),source_evidence_sha256=digest(source),
    camera_sha256=camera_info['state_sha256'],whole_programme_exposures=len(pts),source_time_base=[1,12800],
    source_first_last_pts=[int(pts[0]),int(pts[-1])],
    conservative_spline_support_intervals_seconds=[[float(pts[q[0]]/12800),float(pts[q[-1]]/12800)] for q in intervals],
    native_support_rows=int(exposure_affected.sum()),unchanged_native_rows=int((~exposure_affected).sum()),
    rolling_shutter_support_rows=int(projection_affected.sum()),unchanged_outside_support=off_support,
    model_boot_cavity_p95_by_foot_before_m=np.percentile(before_cavity,95,axis=0).tolist(),
    model_boot_cavity_p95_by_foot_after_m=np.percentile(after_cavity,95,axis=0).tolist(),
    full_programme_reoptimized=False,no_neural_inference=True,accuracy_accepted=False,seconds=time.perf_counter()-started)
(stage/'export_receipt.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(dict(event='native_export_verified',**report)),flush=True)
