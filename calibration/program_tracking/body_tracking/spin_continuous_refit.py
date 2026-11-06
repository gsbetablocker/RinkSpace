"""S05 absolute continuous refit: source/kinematics, then finite flight physics.

No network inference, independent-window translation, display smoothing or
new camera/equipment fitting. Both stages consume the same frozen evidence.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,os,shutil,time,hashlib
import numpy as np
from scipy.sparse import csr_matrix,vstack
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .support_root_placement import restore_parent
from .absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,greville_times,_log
from .absolute_motion_state import AbsoluteMotionState,absolute_reference,evaluate_curve,SCHEMA,curve_from_packet,curve_metadata
from .continuous_motion_factors import ActualMotionFactors
from .continuous_physics import ContinuousMassGeometry,ConfirmedFlightFactors
from .body_execution import BodyExecution,BodyExecutionConfig
from .skate_integration import JointContactProblem,source_evidence
from .spin_grasp_factors import SpinGraspFactors
from .support_image import SupportImageFactors
from .source_track_evidence import build_track_region_document
from .skate_run_policy import RunPolicy
from .stage_quality import StageQualityTrace
from .support_refinement import measure
from .exmr_integration import PROGRAMME,CASE_FIELD,case_identity,save_packet
from .skate_checkpoint import checkpoint_identity,validate_checkpoint_identity
from .centroidal_physics_diagnostics import discrete_segment_rates,centroidal_momentum,ground_force_torque_diagnostics,phase_safe_flight_diagnostics


def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def preserve_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if digest(src)!=digest(dst):raise FileExistsError('Changed frozen input '+str(dst))
    else:shutil.copy2(src,dst)


def seed_curve(parent,shutter):
    lo=-abs(shutter)/2-.012;hi=parent.times[-1]+abs(shutter)/2+.012
    absolute=np.array([lo+76,80.8,82.4,84.,86.,88.,92.2,92.4,93.2,hi+76]);steps=[.12,.04,.08,.04,.08,.06,.08,.04,.12]
    breaks=[]
    for a,b,step in zip(absolute[:-1],absolute[1:],steps):breaks.extend(np.linspace(a-76,b-76,max(1,int(np.ceil((b-a)/step)))+1)[:-1])
    breaks=np.r_[breaks,hi];refinements=[]
    for _ in range(8):
        try:curve=AbsoluteMotionCurve.seed_from_reference(parent.at,breaks,source_seconds=parent.times);break
        except ValueError as exc:
            if 'path' not in str(exc) and 'branch' not in str(exc):raise
            k=clamped_cubic_knots(breaks);g=greville_times(k);dense=np.unique(np.r_[g,parent.times]);s=parent.at(dense);R=np.concatenate((s['Q'][:,None],s['local_R']),1)
            angles=np.linalg.norm(_log(R[:-1].swapaxes(-1,-2)@R[1:]),axis=-1);add=[]
            for a,b in zip(g[:-1],g[1:]):
                use=(dense[:-1]>=a)&(dense[1:]<=b)
                if np.any(angles[use].sum(axis=0)>=np.pi-1e-5):add.append((a+b)/2)
            if not add:raise
            breaks=np.unique(np.r_[breaks,add]);refinements.extend(add)
    else:raise ValueError('Source rotation path still under-resolved after local refinement')
    # Position controls least-squares initialize the saved actual path; rotations
    # remain a smooth path-resolved approximation, not exact dense interpolation.
    curve.position_controls=np.linalg.lstsq(curve.basis(parent.times),parent.at(parent.times)['P'],rcond=None)[0]
    old=parent.at(parent.times);new=evaluate_curve(curve,parent.times)
    return curve,dict(schema=SCHEMA,controls=len(curve.position_controls),variables=len(curve.position_controls)*69,
        break_seconds=breaks.tolist(),extra_path_refinement_times=refinements,full_source_path_checked=True,
        initialization_rotation_difference_degrees_p95=float(np.percentile(np.rad2deg(np.linalg.norm(_log(old['Q'].swapaxes(-1,-2)@new['Q']),axis=-1)),95)),
        initialization_position_error_solver_max=float(np.linalg.norm(old['P']-new['P'],axis=1).max()),
        initialization_not_accuracy_improvement=True,no_frozen_high_frequency_pose_added_back=True,
        hand_state='explicit original HandMR rotations retained; both wrist local rotations optimized',readout_halo_elapsed_seconds=[float(lo),float(hi)])


def source_document(project,base,root):
    old=base/'spin_entry_motion_1104';doc=json.loads((old/'support_and_free_shoe_regions.json').read_text(encoding='utf-8'))
    track=load(old/'source_support_track.npz');anchors=json.loads((project/PROGRAMME/'spin_support_midpoint_anchors_1104.json').read_text(encoding='utf-8'))
    review=json.loads((root/'source_review/seed_visual_review.json').read_text(encoding='utf-8'))
    dense=root/'source_review/dense_visible_blade_rows.json'
    if not dense.exists():raise FileNotFoundError('Complete the selected source-exposure review before freezing evidence: '+str(dense))
    extra=json.loads(dense.read_text(encoding='utf-8'));by={(r['pts'],r.get('source_semantic'),r.get('blade_endpoint')):r for r in review['rows']}
    for r in extra['rows']:
        if r.get('optimization_enabled') and r.get('material_identity_confirmed'):by[(r['pts'],r.get('source_semantic'),r.get('blade_endpoint'))]=r
    review['rows']=list(by.values())
    # Four previously direct-reviewed geometric-middle regions remain valid
    # finite source evidence. They do not upgrade the other coarse seed rows.
    previous=base/'spin_grasp_1104/source_review/support_direct_check_v2.json'
    for row in json.loads(previous.read_text(encoding='utf-8'))['frames']:
        if row.get('blade_midpoint_px') is None:continue
        box=np.asarray(row['blade_midpoint_tolerance_xyxy_px'],float);image=Path(row['image'])
        review['rows'].append(dict(pts=int(row['native_pts']),foot=1,role='fit',visibility='visible_blurred',
            source_semantic='blade_midpoint_region',centre_uv=row['blade_midpoint_px'],region_half_extent_uv=((box[2:]-box[:2])/2).tolist(),
            source_frame=str(image),source_frame_sha256=digest(image),provenance='Preserved previous direct native source review '+str(previous),
            material_identity_confirmed=True,optimization_enabled=True,reliability=.65,contact_state='unknown',
            source_uncertainty='finite approximate visible geometric blade middle, not a calibrated covariance or certified contact/material measurement'))
    protected={int(r['pts'])+d for r in doc['observations'] if r['role']=='review' for d in [-256,0,256]}
    new,prepared=build_track_region_document(doc,track,anchors['anchors'],review_overrides=review,excluded_review_pts=protected,native_step_pts=256)
    new['additional_direct_review_unknowns']=[r for r in extra['rows'] if not r.get('optimization_enabled')]
    write_json(root/'source_review/frozen_review_overrides.json',review);write_json(root/'source_evidence.json',new)
    write_json(root/'source_exposure_audit.json',prepared['source_exposure_audit'])
    np.savez_compressed(root/'source_support_track.npz',**{k:v for k,v in prepared.items() if isinstance(v,np.ndarray)})
    return new

class ContinuousFactors(SpinGraspFactors):
    def __init__(self,problem,relations,parent,flight=False):
        super().__init__(problem,relations);self.motion=ActualMotionFactors(problem,parent);self.mass=ContinuousMassGeometry(problem.reference)
        self.flight=ConfirmedFlightFactors(problem,self.mass) if flight else None
    def evaluate(self,x,jacobian=False):
        out=super().evaluate(x,jacobian)
        for f in [self.motion,self.flight]:
            if f is None:continue
            v=f.evaluate(x,jacobian);out['residual']=np.r_[out['residual'],v['residual']]
            if jacobian:out['jac']=vstack((out['jac'],v['jac']),format='csr')
            out['parts'].update(v['parts']);out['metrics']['actual_motion' if f is self.motion else 'finite_flight']=v['metrics']
        return out

class ContinuousJoint(JointContactProblem):
    linear_solver_backend='sparse_direct'
    def checkpoint_fields(self,x):
        d=super().checkpoint_fields(x);d.update(self.body.state.packet(x[17:]));return d


def prepare(project,base,root):
    root.mkdir(exist_ok=True);param=project/PROGRAMME/'spin_entry_gravity_1104.npz';previous=base/'spin_entry_gravity_1104'
    parent,camera,info,saved=restore_parent(project,base,previous,root/'parent_geometry',param)
    if (root/'initialization.json').exists():
        initial=load(root/'common/exmr_hand_nofoot/initial.npz');curve=curve_from_packet(initial,seed=True)
        return parent,camera,info,curve,json.loads((root/'source_evidence.json').read_text(encoding='utf-8')),project/PROGRAMME/'spin_continuous_initial_1105.npz'
    actual=load(previous/'common/exmr_hand_nofoot/fitted/result.npz');at=parent.at(parent.times)
    np.testing.assert_array_equal(actual['source_pts'],parent.p['pts'])
    for key,skey in [('P','P_world_mid'),('Q','Q_world_mid'),('local_R','local_R_mid')]:np.testing.assert_allclose(at[key]*(parent.binding.ratio if key=='P' else 1),actual[skey],rtol=0,atol=2e-10)
    curve,initialization=seed_curve(parent,float(camera.p['rig'][37]));document=source_document(project,base,root)
    common=root/'common/exmr_hand_nofoot';common.mkdir(parents=True,exist_ok=True)
    for name in ['input.npz','observations.npz','compact_smplx.npz','export_source.npz']:
        preserve_copy(previous/'common/exmr_hand_nofoot'/name,common/name)
    # Create a new ledger revision. UV/detector labels/sigma stay intact; the
    # isolated 81.40 label reversal is evidence-unknown, not swapped to fit.
    obs=load(common/'observations.npz');raw_obs_sha=digest(common/'observations.npz');review=json.loads((root/'source_review/flight_body_visibility_review.json').read_text(encoding='utf-8'))
    f=np.flatnonzero(obs['pts']==1041920)[0];old_use=obs['use'][f].copy();obs['use'][f,[9,10,13,14,15,16]]=False
    np.savez_compressed(common/'observations.npz',**obs)
    write_json(root/'body_observation_revision.json',dict(parent_ledger_sha256=raw_obs_sha,current_ledger_sha256=digest(common/'observations.npz'),
        pts=1041920,unknown_side_coco_indices=[9,10,13,14,15,16],old_use=old_use.tolist(),raw_uv_sigma_and_labels_preserved=True,reason='Direct three-source-frame review identifies isolated left/right association reversal',source_review_sha256=digest(root/'source_review/flight_body_visibility_review.json')))
    initial=load(previous/'common/exmr_hand_nofoot/initial.npz');initial.update(P=at['P'],Q=at['Q'],local_R=at['local_R'],absolute_knots=curve.knots,absolute_seed_position_controls=curve.position_controls,absolute_seed_rotation_controls=curve.rotation_controls,state_schema_version=np.array(SCHEMA))
    np.savez_compressed(common/'initial.npz',**initial)
    archive=project/PROGRAMME/'spin_continuous_initial_1105.npz'
    from rinkspace.path_policy import guard_project_write
    guard_project_write(archive,project_root=project)
    if archive.exists():raise FileExistsError('Existing absolute initialization is preserved')
    shutil.copy2(common/'initial.npz',archive)
    for name in ['exmr_body_hand_nofoot.npz','palm_grip_geometry.json','grasp_relations.json']:
        preserve_copy(previous/name,root/name)
    relation_path=root/'grasp_relations.json';rel=json.loads(relation_path.read_text(encoding='utf-8'));old_configs={k:rel['factor_parameters'].pop(k) for k in ['entry_motion','flight_ballistic'] if k in rel['factor_parameters']}
    rel['historical_unused_correction_and_flight_configs']=old_configs;rel['absolute_motion_integration']=dict(schema=SCHEMA,actual_so3_acceleration=True,finite_flight_stage_B_only=True,flight_interior_seconds=[81.36,81.46],physics_model_is_proxy=True)
    write_json(relation_path,rel)
    write_json(root/'initialization.json',dict(initialization,parent_parameter=str(param),parent_parameter_sha256=digest(param),initial_archive=str(archive),camera=info,
        shared_s01_sha256=digest(project/PROGRAMME/'skate_s01_1103.npz'),parent_replayed_before_conversion=True))
    return parent,camera,info,curve,document,archive


def make_problem(project,base,root,stage,parent,camera,info,curve,document,execution,flight,log,*,existing=False):
    dest=root/stage;case=dest/'common/exmr_hand_nofoot'
    common=root/'common/exmr_hand_nofoot'
    if existing:
        if not case.is_dir():raise FileNotFoundError('Prepared case is absent: '+str(case))
        for name in ['input.npz','observations.npz','compact_smplx.npz','export_source.npz','initial.npz']:
            if digest(common/name)!=digest(case/name):raise ValueError('Prepared common input changed: '+name)
        for name in ['exmr_body_hand_nofoot.npz','palm_grip_geometry.json','grasp_relations.json']:
            if digest(root/name)!=digest(dest/name):raise ValueError('Prepared resource changed: '+name)
        if json.loads((dest/'support_and_free_shoe_regions.json').read_text(encoding='utf-8'))!=document:
            raise ValueError('Prepared support document changed')
    else:
        case.mkdir(parents=True)
        for name in ['input.npz','observations.npz','compact_smplx.npz','export_source.npz','initial.npz']:
            os.link(common/name,case/name)
        for name in ['exmr_body_hand_nofoot.npz','palm_grip_geometry.json','grasp_relations.json']:
            os.link(root/name,dest/name)
        write_json(dest/'support_and_free_shoe_regions.json',document)
    ref=absolute_reference(parent,curve,root=case);ref.export_source_path=case/'export_source.npz'
    ref.grasp_relations=json.loads((dest/'grasp_relations.json').read_text(encoding='utf-8'))
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance=dict(source_document_sha256=digest(dest/'support_and_free_shoe_regions.json'),same_frozen_source_evidence=True))
    identity=case_identity(project,base,dest,'exmr_hand_nofoot',info)
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info);ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True));execution.attach(ref)
    if existing:
        if json.loads((case/'comparison_case_identity.json').read_text(encoding='utf-8'))!=identity:
            raise ValueError('Prepared case identity changed')
    else:write_json(case/'comparison_case_identity.json',identity)
    sec=tuple(curve.knots[[3,-4]]+76.)
    supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=document['supports'])
    problem=ContinuousJoint(ref,camera,case/'observations.npz',sec,supports,dest/'solve',log,'absolute_motion_v1',optimize_shared=False,record_acceptance_before_jacobian=True)
    # No inherited opening-only factors or correction-derivative priors.
    class NoOpening:
        def evaluate(self,x,jacobian=False):return dict(residual=np.empty(0),jac=csr_matrix((0,problem.body.size)),parts={},metrics={})
    problem.source_motion=NoOpening()
    runtime=copy.deepcopy(document)
    mult=ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
    for r in runtime['observations']:r['objective_weight']*=mult
    problem.support_image=SupportImageFactors(problem,runtime,arc_ids={})
    problem.action_factors=ContinuousFactors(problem,ref.grasp_relations,parent,flight=flight)
    report=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
    problem.checkpoint_extra=dict(parent_parameter_sha256=np.array(digest(project/PROGRAMME/'spin_entry_gravity_1104.npz')),
        initial_reference=np.array((PROGRAMME/'spin_continuous_initial_1105.npz').as_posix()),initial_reference_sha256=np.array(digest(project/PROGRAMME/'spin_continuous_initial_1105.npz')),
        absolute_parent_case_root=np.array(str(base/'spin_entry_gravity_1104')),absolute_case_root=np.array(str(dest)),
        evidence_document_sha256=np.array(digest(dest/'support_and_free_shoe_regions.json')),actual_observations_sha256=np.array(digest(case/'observations.npz')),
        finite_flight_enabled=np.array(flight))
    return problem,report,supports


def derivative_check(problem,x,output):
    tick=time.perf_counter();v=problem.evaluate(x,True);initial_body_cache=problem.body.cache;J=v['jac'];elapsed=time.perf_counter()-tick;checks=[];state=problem.body.state
    for name,cols,when in [('root',np.arange(6),(80.8,82.4)),('legs',np.arange(6,30),(85.52,85.66)),('non_grip_right_wrist',np.arange(66,69),(89.26,89.44))]:
        t=greville_times(state.knots)+76.;active=(t>=when[0])&(t<=when[1]);rng=np.random.default_rng(1105+len(cols));d=np.zeros_like(x);local=d[17:].reshape(state.m,69);local[np.ix_(active,cols)]=rng.normal(size=(active.sum(),len(cols)));d/=np.linalg.norm(d);h=1e-6
        numeric=(problem.evaluate(x+h*d)['residual']-problem.evaluate(x-h*d)['residual'])/(2*h);exact=J@d
        rel=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12));checks.append(dict(block=name,relative_norm=rel,max_absolute=float(np.max(abs(exact-numeric)))))
        if rel>3e-3:write_json(output,checks);raise ValueError('Mixed absolute/native Jacobian mismatch: '+str(checks[-1]))
    # Reuse the exact first-state J; verification trial evaluations must not force a duplicate.
    problem.cache_x=x.copy();problem.cache=v;problem.body.cache_x=x[17:].copy();problem.body.cache=initial_body_cache
    problem.jacobian_seconds_hint=elapsed;write_json(output,dict(first_jacobian_seconds=elapsed,checks=checks));return v,elapsed


def export_physics(problem,final_ref,dest,packet):
    geometry=ContinuousMassGeometry(final_ref);times=final_ref.times;C,R=geometry.evaluate(final_ref.at(times),np.zeros((len(times),69)));com=geometry.com(C)
    source=final_ref.p['pts']/12800.;phase=np.full(len(times),'unknown',dtype='<U32');confirmed=np.zeros(len(times),bool)
    flight=(source>=81.36-1e-9)&(source<=81.46+1e-9);phase[flight]='confirmed_flight_interior';confirmed[flight]=True
    for idx,interval in enumerate(final_ref.grasp_relations['support_intervals']):
        mask=(final_ref.p['pts']>=interval['start_pts'])&(final_ref.p['pts']<=interval['end_pts']);phase[mask]='source_support_episode_'+str(idx);confirmed[mask]=True
    rates=discrete_segment_rates(times,C,R,phase_ids=phase,confirmed_phase=confirmed);i=rates['source_indices']
    momentum=centroidal_momentum(centres_world_m=C[i],velocity_world_mps=rates['segment_velocity_world_mps'],rotation_world_from_segment=R[i],omega_world_radps=rates['segment_omega_world_radps'],segment_mass_fractions=geometry.mass,unit_mass_inertia_segment_m2=geometry.inertia)
    H=np.full((len(times),3),np.nan);H[i]=momentum['specific_angular_momentum_world_m2ps'];valid=np.zeros(len(times),bool);valid[i]=rates['derivative_stencil_valid']
    support=confirmed&~flight
    ground=ground_force_torque_diagnostics(times,com,np.nan_to_num(H),support,[0,0,-9.81],H_rate_stencil_valid=valid,phase_ids=phase)
    sensitivity=[]
    for lower,shoe in [(.8,.01),(1.2,.03)]:
        mass=geometry.mass.copy();legs=np.isin(geometry.ids,[1,2,4,5,7,8,10,11]);body=mass[:geometry.count].copy()*np.where(legs,lower,1);body/=body.sum();mass=np.r_[body,shoe,shoe];mass/=mass.sum()
        alt=centroidal_momentum(centres_world_m=C[i],velocity_world_mps=rates['segment_velocity_world_mps'],rotation_world_from_segment=R[i],omega_world_radps=rates['segment_omega_world_radps'],segment_mass_fractions=mass,unit_mass_inertia_segment_m2=geometry.inertia)
        sensitivity.append(dict(lower_mass_multiplier=lower,relative_skate_mass=shoe,max_com_change_m=float(np.linalg.norm(alt['com_world_m']-com[i],axis=1).max()),max_specific_H_change_m2ps=float(np.linalg.norm(alt['specific_angular_momentum_world_m2ps']-H[i],axis=1).max())))
    arrays={k:v for k,v in geometry.proxy.items() if isinstance(v,np.ndarray)}
    arrays.update(physics_pts=final_ref.p['pts'],physics_time_base=np.array([1,12800]),segment_com_world_m=C,segment_R_world=R,
        rates_source_indices=i,segment_velocity_world_mps=rates['segment_velocity_world_mps'],segment_omega_world_radps=rates['segment_omega_world_radps'],
        com_world_m=com,specific_angular_momentum_world_m2ps=H,specific_H_orbital_world_m2ps=momentum['specific_H_orbital_world_m2ps'],specific_H_intrinsic_world_m2ps=momentum['specific_H_intrinsic_world_m2ps'],
        phase_evidence=phase,derivative_stencil_valid=valid,model_identity_json=np.array(json.dumps(serial(geometry.proxy),sort_keys=True)),state_sha256=np.array(digest(Path(str(packet['absolute_parameter_path'])))))
    np.savez_compressed(dest/'centroidal_physics.npz',**arrays)
    write_json(dest/'physics_diagnostics.json',serial(dict(ground=ground,mass_sensitivity=sensitivity,
        flight=phase_safe_flight_diagnostics(times,com,flight,[0,0,-9.81],specific_H_world_m2ps=np.nan_to_num(H),H_rate_stencil_valid=valid),
        model_is_unmeasured_proxy=True,no_supported_spin_H_conservation_constraint=True,all_derivatives_same_final_state=True)))
    p=final_ref.binding.packet()['parameters'];from .skate_template import profile_height
    x=np.array([-p.runner_rear_extension_m,.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),p.sole_length_m-p.runner_front_inset_m]);q=np.c_[x,np.zeros(3),profile_height(x,p)]
    blade=(packet['skate_origin_world_mid'][:,:,None]+np.einsum('nfij,kj->nfki',packet['skate_R_world_mid'],q*final_ref.binding.ratio))
    metadata=curve_metadata(final_ref.absolute_curve)
    packet.update(metadata)
    packet.update(com_world_mid=com,blade_material_world_mid=blade,physics_model_sha256=np.array(digest(dest/'centroidal_physics.npz')),
        body_refinement_window_mask=np.ones(len(times),bool),window_mask=np.ones(len(times),bool),absolute_motion_schema=metadata['state_schema_version'].copy(),
        physics_identity_json=np.array(json.dumps(dict(mass_variant='uniform-volume-body+rigid-shoe-proxy',mass_measured=False,com_is_pelvis=False,H_has_orbital_and_intrinsic=True,shoe_mass_assumed=True))))
    return packet


def run(project,base,root,source_budget=900.,flight_budget=360.,check_only=False):
    project,base,root=map(lambda p:Path(p).resolve(),(project,base,root));os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll')
    parent,camera,info,curve,doc,archive=prepare(project,base,root);execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
    initial_x=None;original_baseline=None
    try:
        with execution.limits():
            for name,flight,budget in [('source_stage',False,source_budget),('flight_stage',True,flight_budget)]:
                dest=root/name
                if (dest/'run_summary.json').exists():
                    prior_summary=json.loads((dest/'run_summary.json').read_text(encoding='utf-8'));initial_x=load(Path(prior_summary['parameter']))['x'].copy()
                    original_baseline=json.loads((dest/'cross_stage_quality.json').read_text(encoding='utf-8'))['original_A']
                    print(json.dumps(dict(stage=name,preserved_completed_stage=True,accepted=prior_summary['accepted'])),flush=True);continue
                if flight:
                    execution.close();execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
                started=time.perf_counter()
                with (root/('progress_'+name+'.jsonl')).open('w',encoding='utf-8',buffering=1) as log:
                    problem,review,supports=make_problem(project,base,root,name,parent,camera,info,curve,doc,execution,flight,log)
                    x=np.r_[problem.binding.x,np.zeros(problem.body.size)] if initial_x is None else initial_x.copy()
                    # Same immutable original source baseline in both stages.
                    base_metrics=measure(problem,review,x,problem.evaluate(x,False))
                    if original_baseline is None:original_baseline=base_metrics
                    problem.stage_quality=StageQualityTrace(dest/'cross_stage_quality.json',dict(parent_parameter_sha256=digest(project/PROGRAMME/'spin_entry_gravity_1104.npz'),observations_sha256=digest(problem.body.ledger_path),source_document_sha256=digest(root/'source_evidence.json'),same_evidence_A_B=True),original_baseline,lambda xx,v:measure(problem,review,xx,v))
                    value,jsec=derivative_check(problem,x,dest/'mixed_jacobian_checks.json')
                    write_json(dest/'execution.json',dict(initial_J_seconds=jsec,variables=problem.body.size,controls=problem.body.state.m,actual_source_frames=problem.body.n,execution=execution.record(),inner_linear_backend=problem.linear_solver_backend,budget_seconds=budget))
                    if check_only:return
                    remaining=budget-(time.perf_counter()-started)
                    if remaining<2*jsec:raise RuntimeError('Remaining budget insufficient for actual progress')
                    result,report=problem.solve(x,RunPolicy(wall_seconds=remaining,max_accepted=200,max_outer=1,stagnation_window=6,allow_penalty_growth=False))
                    saved=load(problem.output/'candidate.npz');initial_x=saved['x'].copy()
                    parameter=project/PROGRAMME/('spin_continuous_flight_1105.npz' if flight else 'spin_continuous_source_1105.npz')
                    from rinkspace.path_policy import guard_project_write
                    guard_project_write(parameter,project_root=project)
                    if parameter.exists():raise FileExistsError('Existing continuous parameter is preserved')
                    np.savez_compressed(parameter,**saved)
                    final=absolute_reference(problem.reference,problem.body.state.current_curve(initial_x[17:]),root=problem.reference.root)
                    shell=copy.copy(problem);shell.reference=final
                    packet=save_packet(shell,info,supports,doc,dest/'common/exmr_hand_nofoot',status='joint_fit',report=report)
                    packet['absolute_parameter_path']=np.array(str(parameter));packet=export_physics(problem,final,dest,packet)
                    np.savez_compressed(dest/'common/exmr_hand_nofoot/fitted/result.npz',**packet)
                    summary=dict(parameter=str(parameter),result=str(dest/'common/exmr_hand_nofoot/fitted/result.npz'),accepted=problem.accepted,retained_step=report['selection']['retained_step'],
                        wall_seconds=time.perf_counter()-started,budget_seconds=budget,source_frames=len(packet['pts']),source_context_seconds=[76,95],schema=str(packet['state_schema_version']),
                        same_frozen_observations_sha256=digest(problem.body.ledger_path),source_evidence_sha256=digest(root/'source_evidence.json'),finite_flight=flight,inner_linear_backend=problem.linear_solver_backend,linear_solver_statistics=getattr(problem,'linear_solver_statistics',[]),
                        accuracy_accepted=False,numerical_convergence=False,stop=report['stop'],whole_shot_adjustable=True)
                    write_json(dest/'run_summary.json',summary);print(json.dumps({k:summary[k] for k in ('parameter','accepted','retained_step','wall_seconds') }),flush=True)
    finally:execution.close()


def restore_absolute(project,base,root,parameter):
    saved=load(parameter)
    if 'relative_skate_schema' in saved:
        from .blade_current_reference import restore_current_reference
        return restore_current_reference(project,Path(str(saved['absolute_case_root'])),parameter,seed=False)
    curve=curve_from_packet(saved)
    native=base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll';os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(native)
    parent,camera,info,_=restore_parent(project,base,Path(str(saved['absolute_parent_case_root'])),root/'absolute_replay_geometry',project/PROGRAMME/'spin_entry_gravity_1104.npz')
    ref=absolute_reference(parent,curve,root=Path(str(saved['absolute_case_root']))/'common/exmr_hand_nofoot')
    from .exmr_integration import resolve_initial_reference,validate_case_identity
    dest=Path(str(saved['absolute_case_root']));expected=case_identity(project,base,dest,'exmr_hand_nofoot',info,initial=resolve_initial_reference(project,saved))
    validate_case_identity(saved,expected);ref.fixed_checkpoint_identity={k:v for k,v in saved.items() if k in parent.fixed_checkpoint_identity};return ref,camera,info,saved


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-budget',type=float,default=900.);p.add_argument('--flight-budget',type=float,default=360.);p.add_argument('--check-only',action='store_true');a=p.parse_args(argv)
    return run(a.project,a.base_root,a.output,a.source_budget,a.flight_budget,a.check_only)
if __name__=='__main__':main()
