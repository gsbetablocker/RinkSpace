"""Explicit full-programme input revisions and conditional numeric initializers.

An input revision is not a blind checkpoint migration: the parent is validated
against its own prepared identity, new binding/clock identities are generated
from the new input, and unchanged states are replayed numerically. Original
finite pose/root priors remain separate from the current warm-start pose.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,os,shutil
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsmr
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .trajectory_backend import load_reference,primary_policy
from .absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,greville_times
from .absolute_motion_state import curve_from_packet,absolute_reference,curve_metadata
from .sequence_skate_state import SequenceState,shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity,checkpoint_identity
from .skate_template import SkateParameters
from .body_clock import source_seconds


def inherited_episode_hypotheses(parent,evidence_sha256):
    """Keep track identity hypotheses, rebuild geometry-dependent associations."""
    if str(parent.get('source_evidence_sha256',''))!=evidence_sha256:return {}
    fields={}
    for key,value in parent.items():
        if (key.startswith('source_association_extra_')and key.endswith('_choice')
                and not key.endswith('_fixed_choice')):
            if np.asarray(value).shape!=():raise ValueError('Episode hypothesis must be scalar: '+key)
            fields[key]=np.asarray(value).copy()
    if'source_association_hypotheses_json'in parent:
        fields['source_association_hypotheses_json']=parent['source_association_hypotheses_json'].copy()
    return fields


def apply_body_leg_assignment(prepared,parent_prepared,source_pts,provenance):
    """Repair explicit network label correspondence in a new input revision.

    Raw UV, confidence/use and sigma arrays are kept byte-for-byte unchanged.
    Only their correspondence to the fixed physical model joints is added.
    """
    prepared,parent_prepared=Path(prepared).resolve(),Path(parent_prepared).resolve()
    if prepared==parent_prepared:raise ValueError('Source assignment needs a distinct prepared revision')
    original=parent_prepared/'observations.npz';original_sha=digest(original)
    if digest(prepared/'observations.npz')!=original_sha:raise ValueError('Start from the unchanged parent observations')
    obs=load(original)
    if'model_joint_ids'in obs:raise ValueError('Explicitly review an existing correspondence before revising it')
    stamps=np.asarray(source_pts,dtype=np.int64)
    if stamps.ndim!=1 or not len(stamps)or np.any(np.diff(stamps)<=0):raise ValueError('Exact ordered source PTS required')
    rows=np.searchsorted(obs['pts'],stamps)
    if np.any(rows>=len(obs['pts']))or not np.array_equal(obs['pts'][rows],stamps):raise ValueError('Assignment outside native source PTS')
    ids=np.broadcast_to(np.arange(17),(len(obs['pts']),17)).copy()
    ids[np.ix_(rows,np.arange(11,17))]=np.array([12,11,14,13,16,15])
    record=dict(provenance,kind='source_observation_correspondence',source_pts=stamps.tolist(),
        source_measurements_unchanged=True,parent_observations=dict(path=str(original),sha256=original_sha),
        point_kind='same anatomical joint kinds; network bilateral labels only; not boot/blade/contact points',
        physical_model_identity_changed=False)
    obs.update(model_joint_ids=ids,body_joint_assignment_provenance_json=np.array(json.dumps(record,sort_keys=True)))
    from .fullbody_fit import body_joint_assignment
    body_joint_assignment(obs)
    target=prepared/'observations.npz';temporary=prepared/'observations.assignment.pending.npz'
    if temporary.exists():raise FileExistsError(temporary)
    np.savez_compressed(temporary,**obs);temporary.replace(target)
    document=json.loads((prepared/'source_evidence.json').read_text(encoding='utf8'))
    if'body_observation_correspondence'in document:raise ValueError('Existing source correspondence ledger')
    document['body_observation_correspondence']=record
    write_json(prepared/'source_evidence.json',document)
    if digest(original)!=original_sha:raise ValueError('Parent observations changed')
    verify=load(target);old=load(original)
    for key,value in old.items():np.testing.assert_array_equal(verify[key],value,err_msg='Original source changed: '+key)
    report=dict(**record,new_observations_sha256=digest(target),all_parent_arrays_equal=True)
    write_json(prepared/'body_joint_assignment_revision.json',report)
    return report


def copy_prepared(source,destination,*,scale_factor=1.):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    # Immutable models/cameras/observations may share bytes. Editable evidence
    # and body parameters are copied, so a new version cannot alter its parent.
    mutable={'input.npz','initial.npz','fixed_binding.npz','source_evidence.json',
        'export_source.npz','preparation.json','absolute_seed_state.npz'}
    for path in source.iterdir():
        if not path.is_file():continue
        if path.name=='absolute_seed_state.npz':continue
        if path.name=='fixed_binding.npz'and scale_factor!=1.:continue
        target=destination/path.name
        if path.name in mutable:shutil.copyfile(path,target)
        else:os.link(path,target)
    p=load(destination/'input.npz')
    p['body_scale']=np.asarray(float(p['body_scale'])*scale_factor)
    np.savez_compressed(destination/'input.npz',**p)
    return destination


def _depth_controls(curve,camera,times,amplitude_m,ratio):
    current=curve.basis_sparse(times)@curve.position_controls
    centres=camera.camera_centres(np.arange(len(times)))
    ray=current-centres;ray/=np.linalg.norm(ray,axis=1)[:,None]
    # One low-frequency direction on the whole source clock. A camera cut
    # cannot introduce a discontinuous physical displacement into this profile.
    lo,hi=curve.knots[[3,-4]]
    knots=clamped_cubic_knots(np.r_[lo,np.arange(lo+6.,hi,6.),hi])
    B=BSpline.design_matrix(times,knots,3,extrapolate=True).tocsr()
    C=np.stack([lsmr(B,ray[:,j],atol=1e-10,btol=1e-10)[0]for j in range(3)],1)
    field=B@C;peak=np.max(np.linalg.norm(field,axis=1))
    target=field*(amplitude_m/ratio/peak)
    full=curve.basis_sparse(times)
    controls=np.stack([lsmr(full,target[:,j],atol=1e-10,btol=1e-10,maxiter=1000)[0]for j in range(3)],1)
    actual=full@controls
    return controls,dict(direction='whole-clock six-second cubic camera-ray field; mixed ray directions near cuts',
        requested_peak_m=amplitude_m,actual_max_displacement_m=float(np.linalg.norm(actual,axis=1).max()*ratio),
        control_fit_max_m=float(np.linalg.norm(actual-target,axis=1).max()*ratio),
        source_pts_first_last=camera.p['pts'][[0,-1]].tolist(),no_independent_shot_translation=True)


def initialize(project,parent_prepared,parent_parameter,prepared,output,*,scale_factor=1.,depth_m=0.):
    project,parent_prepared,parent_parameter,prepared,output=map(Path,(project,parent_prepared,parent_parameter,prepared,output))
    if output.exists():raise FileExistsError(output)
    if scale_factor<=0 or not np.isfinite([scale_factor,depth_m]).all():raise ValueError('Finite positive shared scale and depth required')
    parent=load(parent_parameter);old_ref,old_camera,old_info,old_document,_=load_reference(project,parent_prepared)
    validate_checkpoint_identity(parent,old_ref.fixed_checkpoint_identity)
    for key,name in(('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz')):
        if str(parent[key])!=digest(parent_prepared/name):raise ValueError('Parent is not bound to its supplied prepared input: '+key)
    p=load(prepared/'input.npz');old_p=load(parent_prepared/'input.npz')
    np.testing.assert_array_equal(p['pts'],old_p['pts']);np.testing.assert_array_equal(p['time_base'],old_p['time_base'])
    np.testing.assert_array_equal(p['shared_betas'],old_p['shared_betas'])
    np.testing.assert_allclose(float(p['body_scale']),float(old_p['body_scale'])*scale_factor,rtol=0,atol=1e-14)
    if digest(prepared/'camera_context.json')!=digest(parent_prepared/'camera_context.json'):raise ValueError('Camera must stay frozen')
    if digest(prepared/'model_identity.json')!=digest(parent_prepared/'model_identity.json'):raise ValueError('Generic SMPL-X asset changed')
    seed=curve_from_packet(parent,seed=True);current=curve_from_packet(parent);shoe_seed=shoe_curve_from_packet(parent,seed=True)
    shoe_current=shoe_curve_from_packet(parent);x=parent['x'].copy();m=len(seed.position_controls)
    xP=x[17:17+m*69].reshape(m,69)[:,:3]
    if scale_factor!=1.:
        gt=greville_times(seed.knots);nearest=np.clip(np.searchsorted(old_ref.times,gt),0,len(old_ref.times)-1)
        centres=old_camera.camera_centres(nearest)
        seed=AbsoluteMotionCurve(seed.knots,centres+scale_factor*(seed.position_controls-centres),seed.rotation_controls,seed.edge_reference_quaternion)
        current=AbsoluteMotionCurve(current.knots,centres+scale_factor*(current.position_controls-centres),current.rotation_controls,current.edge_reference_quaternion)
        xP*=scale_factor
    depth_record=None
    if depth_m:
        dc,depth_record=_depth_controls(current,old_camera,old_ref.times,depth_m,old_ref.binding.ratio)
        xP+=dc
        current=AbsoluteMotionCurve(current.knots,current.position_controls+dc,current.rotation_controls,current.edge_reference_quaternion)
    # Save the original (possibly scale-transformed) prior, never the current
    # fitted pose as a new finite-network target.
    state=dict(parent);state.update(**curve_metadata(seed),absolute_knots=seed.knots,
        absolute_seed_position_controls=seed.position_controls,absolute_seed_rotation_controls=seed.rotation_controls,
        absolute_position_controls=current.position_controls,absolute_rotation_controls=current.rotation_controls)
    initial=load(prepared/'initial.npz');values=absolute_reference(old_ref,seed).at(old_ref.times)
    changed_initial=any(not np.array_equal(initial[k],values[k])for k in('P','Q','local_R'))
    initial.update(P=values['P'],Q=values['Q'],local_R=values['local_R'])
    if scale_factor!=1. and changed_initial:np.savez_compressed(prepared/'initial.npz',**initial)
    state.update(prepared_input_sha256=np.array(digest(prepared/'input.npz')),
        prepared_initial_sha256=np.array(digest(prepared/'initial.npz')))
    explicit=prepared/'absolute_seed_state.npz'
    seed_fields={k:v for k,v in state.items()if k.startswith('absolute_')or k.startswith('relative_skate_')or k=='state_schema_version'}
    seed_fields.update(prepared_input_sha256=state['prepared_input_sha256'],prepared_initial_sha256=state['prepared_initial_sha256'])
    if explicit.exists():
        declared=load(explicit)
        keys=['state_schema_version','absolute_knots','absolute_seed_position_controls','absolute_seed_rotation_controls',
            'relative_skate_schema','relative_skate_knots','relative_skate_seed_rotation_controls','prepared_input_sha256','prepared_initial_sha256']
        if 'absolute_rotation_edge_reference_quaternion'in seed_fields:keys.append('absolute_rotation_edge_reference_quaternion')
        for key in keys:np.testing.assert_array_equal(declared[key],seed_fields[key],err_msg='Do not replace a different declared prior: '+key)
    else:np.savez_compressed(explicit,**seed_fields)
    ref,camera,info,document,seed_report=load_reference(project,prepared)
    if scale_factor!=1.:
        # Preserve the current fitted shoe dimensions and registration angles.
        # Re-register the same heel target against the newly scaled skin/ankle;
        # sample membership, cavity and COM are rebuilt from this new body.
        binding=ref.binding;old=old_ref.binding
        binding.base_parameters=copy.deepcopy(old.base_parameters);binding.R0=old.R0.copy()
        changes=[]
        for f,a in enumerate((7,8)):
            old_offset=old.foot_correspondences[f]['heel']-old.neutral_native[0,a]
            new_offset=binding.foot_correspondences[f]['heel']-binding.neutral_native[0,a]
            changes.append(new_offset-old_offset)
        binding.t0=old.t0+np.asarray(changes);binding.x=parent['x'][:17].copy()
        binding._parameter_key=binding._packet=None
        if old.collar_constraints_enabled:binding.enable_collar_constraints()
        ref.fixed_checkpoint_identity=checkpoint_identity(binding,project,info)
        np.savez_compressed(prepared/'fixed_binding.npz',**ref.fixed_checkpoint_identity,x=binding.x,
            shared_geometry=ref.geometry,collar_constraints_enabled=np.array(binding.collar_constraints_enabled))
    state_owner=SequenceState(ref,tuple(source_seconds(p)[[0,-1]]));packet=state_owner.packet(x[17:])
    replay=state_owner.current_curve(x[17:])
    np.testing.assert_allclose(replay.position_controls,current.position_controls,rtol=0,atol=2e-12)
    np.testing.assert_allclose(replay.rotation_controls,current.rotation_controls,rtol=0,atol=2e-12)
    if scale_factor==1. and not depth_m:
        np.testing.assert_array_equal(packet['absolute_seed_rotation_controls'],parent['absolute_seed_rotation_controls'])
        np.testing.assert_allclose(packet['absolute_position_controls'],parent['absolute_position_controls'],rtol=0,atol=2e-12)
    output.mkdir(parents=True);(output/'solve').mkdir()
    obs=load(prepared/'observations.npz');joint_ids=np.nonzero(obs['use'])[1]
    dummy=SimpleNamespace(body=SimpleNamespace(oj=joint_ids));policy=primary_policy(dummy)
    write_json(output/'objective_policy.json',policy);np.savez_compressed(output/'objective_block_weights.npz',**dummy.objective_block_weights)
    parent_stage=parent_parameter.parent.parent;baseline=json.loads((parent_stage/'primary_quality_baseline.json').read_text(encoding='utf-8'))
    baseline.setdefault('body_observation_reference',dict(path=str(parent_prepared/'observations.npz'),sha256=digest(parent_prepared/'observations.npz')))
    write_json(output/'primary_quality_baseline.json',baseline)
    fields=dict(ref.fixed_checkpoint_identity,**packet,x=x,knots=state_owner.knots,active=state_owner.active,
        seconds=np.asarray(state_owner.seconds),collar_constraints_enabled=np.array(ref.binding.collar_constraints_enabled),
        ice_soft_policy_json=parent['ice_soft_policy_json'],prepared_input_sha256=np.array(digest(prepared/'input.npz')),
        prepared_observations_sha256=np.array(digest(prepared/'observations.npz')),
        source_evidence_sha256=np.array(digest(prepared/'source_evidence.json')),
        parent_accepted_steps=np.array(0),input_revision_initializer=np.array(True),
        input_revision_parent_parameter_sha256=np.array(digest(parent_parameter)))
    fields.update(inherited_episode_hypotheses(parent,digest(prepared/'source_evidence.json')))
    np.savez_compressed(output/'solve/candidate.npz',**fields)
    write_json(output/'solve/candidate_selection.json',dict(retained_step=0,attempt_step=0,available=True,quality_failed=False))
    report=dict(parent_prepared=str(parent_prepared),parent_parameter=str(parent_parameter),parent_parameter_sha256=digest(parent_parameter),
        prepared=str(prepared),parameter=str(output/'solve/candidate.npz'),source_frames=len(p['pts']),
        scale_factor=scale_factor,body_scale=float(p['body_scale']),shoe_policy='same fitted metric shoe dimensions; new scaled-skin heel registration, cavity and mass/cache rebuild',
        depth_profile=depth_record,original_pose_prior_preserved=True,old_quality_baseline_preserved=True,
        new_identity_from_actual_input_and_binding=True,no_old_identity_forged=True,accepted_steps=0)
    write_json(output/'initialization.json',report)
    return report


def main():
    ap=argparse.ArgumentParser()
    for name in('project','parent-prepared','parent-parameter','prepared','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--copy-input-from',type=Path);ap.add_argument('--scale-factor',type=float,default=1.);ap.add_argument('--depth-m',type=float,default=0.)
    a=ap.parse_args()
    if a.copy_input_from:copy_prepared(a.copy_input_from,a.prepared,scale_factor=a.scale_factor)
    print(json.dumps(initialize(a.project,a.parent_prepared,a.parent_parameter,a.prepared,a.output,scale_factor=a.scale_factor,depth_m=a.depth_m)),flush=True)
if __name__=='__main__':main()
