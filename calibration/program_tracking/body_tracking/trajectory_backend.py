"""Trajectory-first preparation/solve on the existing continuous body backend.

All states, attached shoes and factors use the same FK, native projection and
JointContactProblem. A missing FS ice plane removes invented ice rows only for
that scene; Ankara retains its finite ice objective.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import argparse,copy,json,time,os
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix,vstack,hstack,diags
from scipy.sparse.linalg import lsmr
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .body_clock import source_seconds,time_base
from .absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,greville_times,_log
from .absolute_motion_state import absolute_reference,curve_metadata
from .action_integration import ClipReference
from .sequence_refit import SequenceJoint,SequenceActions
from .sequence_dynamics import TemporalContactFactors
from .skate_binding import SkateBinding
from .skate_template import SkateParameters
from .skate_checkpoint import checkpoint_identity
from .skate_run_policy import RunPolicy
from .continuous_motion_factors import ActualMotionFactors
from .continuous_physics import ContinuousMassGeometry,ConfirmedFlightFactors
from .spin_grasp_factors import SpinGraspFactors,tolerant_support_response
from .skate_physics import fk_state,batch_features,body_constraint_jacobian,NORMALIZATION
from .skate_contact import trapezoid_time_weights
from .body_execution import BodyExecution,BodyExecutionConfig
from .skate_delivery import export_candidate
from .programme_camera import ProgrammeCamera,StaticCropCamera
from .trajectory_probe import GROUPS,body_groups
from .weighted_blocks import block,verify

class GripOnly(SpinGraspFactors):
    def continuous_support(self,x,jacobian):return np.empty(0),csr_matrix((0,self.problem.size))if jacobian else None,[]

class EpisodeSupport:
    """Finite source-gated support; either foot, multiple episodes, no locking."""
    def __init__(self,problem,document):
        self.p=problem;self.ref=problem.reference;pts=self.ref.p['pts'];rows={}
        if self.ref.ice_plane_known:
            for e in document.get('source_episode_review',{}).get('episodes',[]):
                if e['kind']!='support_candidate'or e.get('foot')not in (0,1):continue
                ids=np.flatnonzero((pts>=e['start_pts'])&(pts<=e['end_pts_inclusive']))[::2]
                for i,w in zip(ids,trapezoid_time_weights(self.ref.times[ids],singleton_duration_s=.04)if len(ids)else[]):rows[i,e['foot']]=max(rows.get((i,e['foot']),0.),w)
        keys=sorted(rows);self.indices=np.array([k[0]for k in keys],int);self.feet=np.array([k[1]for k in keys],int);self.dt=np.array([rows[k]for k in keys]);self.times=self.ref.times[self.indices];self.base=self.ref.at(self.times);self.B=problem.body.state.basis(self.times)
    def evaluate(self,x,jacobian=False):
        if not len(self.times):return dict(residual=np.empty(0),jac=csr_matrix((0,self.p.size))if jacobian else None,parts={'episode_support':0.},metrics={})
        d=self.p.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d);f=batch_features(self.p.binding,s['R'],s['O'],s);eligible=self.p.binding.packet()['template']['support_feature_ids']['glide'];ids=np.array([i for i,k in enumerate(f['ids'])if k in eligible]);ii=np.arange(len(self.times));h=f['height'][ii[:,None],self.feet[:,None],ids[None]];best=ids[np.argmin(h,axis=1)];height=f['height'][ii,self.feet,best];q=f['local_point'][ii,self.feet,best]
        cfg=getattr(self.ref,'grasp_relations',None);cfg={}if not cfg else cfg['factor_parameters']['continuous_support'];scale=float(cfg.get('scale_m',.08));weight=float(cfg.get('weight',20.));tolerance=float(cfg.get('deadzone_m',.025));response,slope=tolerant_support_response(height,tolerance);w=np.sqrt(self.dt*weight)/scale;r=w*response;J=None
        if jacobian:
            e=dict(rows=[(i,int(self.feet[i]),q[i],-1.)for i in ii],row_upper_vertex=[-1]*len(ii),times=self.times);D=-NORMALIZATION*body_constraint_jacobian(self.ref,s,d,self.p.body.state.active,self.B,e);J=hstack((csr_matrix((len(r),17)),diags(w*slope)@D),format='csr')
        return dict(residual=r,jac=J,parts={'episode_support':float(r@r/2)},weighted_blocks=[block('episode_support',r,J)],metrics=dict(episode_support=dict(samples=len(r),feet=self.feet.tolist(),source_pts=self.ref.p['pts'][self.indices].tolist(),raw_height_m=height.tolist(),finite_weight=weight,scale_m=scale,tolerance_is_not_accuracy_gate=True)))

class PrimaryActions:
    def __init__(self,problem,parent,mass,document):
        self.motion=ActualMotionFactors(problem,parent);self.support=EpisodeSupport(problem,document);self.grasp=SimpleNamespace(last={'image_cost':0.});self.factors=[self.motion,self.support]
        relations=getattr(problem.reference,'grasp_relations',None)
        if relations and relations.get('grip_interval'):
            relations=copy.deepcopy(relations);relations['support_intervals']=[];held=relations['grip_interval'];tb=time_base(problem.reference.p);relations['context_source_seconds']=[held['start_pts']*tb[0]/tb[1]-.04,held['end_pts']*tb[0]/tb[1]+.04];self.grasp=GripOnly(problem,relations);self.factors.append(self.grasp)
        for e in document.get('source_episode_review',{}).get('episodes',[]):
            if e['kind']=='flight_visible_candidate':self.factors.append(ConfirmedFlightFactors(problem,mass,episode=e))
    def evaluate(self,x,jacobian=False):
        out=dict(residual=[],jac=[],parts={},metrics={},weighted_blocks=[],factor_row_counts=[])
        for factor in self.factors:
            v=factor.evaluate(x,jacobian);out['residual'].append(v['residual'])
            if jacobian:out['jac'].append(v['jac'])
            out['parts'].update(v['parts']);out['metrics'].update(v['metrics']);out['weighted_blocks'].extend(v.get('weighted_blocks',[]));out['factor_row_counts'].append((type(factor).__name__,len(v['residual'])))
        out['residual']=np.concatenate(out['residual']);out['jac']=vstack(out['jac'],format='csr')if jacobian else None;return out

def seed_reference(shared,initial,document,camera):
    initial=dict(initial)
    if'hand_local_R'in initial:
        from .hand_state import SCHEMA,MEAN
        initial.update(hand_pose_schema_version=np.array(SCHEMA),hand_mean_convention=np.array(MEAN),hand_pose_origin=np.array('explicit frozen saved hands; unobserved context retains generic model hand prior'),hand_prior_sha256=np.array(digest(shared.root/'initial.npz')))
    t=shared.times;half=max(abs(camera.readout_times(np.arange(len(t)))))/2;lo,hi=t[0]-half-.005,t[-1]+half+.005
    explicit=shared.root/'absolute_seed_state.npz'
    if explicit.is_file():
        from .absolute_motion_state import curve_from_packet
        from .sequence_skate_state import shoe_curve_from_packet
        packet=load(explicit)
        for key,name in(('prepared_input_sha256','input.npz'),('prepared_initial_sha256','initial.npz')):
            if str(packet[key])!=digest(shared.root/name):raise ValueError('Explicit prior seed identity changed: '+key)
        curve=curve_from_packet(packet,seed=True);shoe=shoe_curve_from_packet(packet,seed=True)
        if shoe is None or curve.knots[3]>lo or curve.knots[-4]<hi:raise ValueError('Explicit prior seed does not cover the complete exposure clock')
        # ClipReference's temporary P spline is superseded by this exact saved
        # absolute curve. Reuse that curve rather than fitting another dense
        # initialization spline solely to construct the read-only reference.
        skipped_temporary_root_fit='root_curve_schema' not in initial
        if skipped_temporary_root_fit:
            initial=dict(initial,P=curve.basis_sparse(t)@curve.position_controls,
                root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),
                root_curve_knots=curve.knots,root_curve_controls=curve.position_controls,
                root_curve_degree=np.array(3))
        seed=ClipReference(shared,shared.root,shared.p,initial);ref=absolute_reference(seed,curve,root=shared.root)
        ref.relative_skate_curve=shoe;ref.binding.reference=ref
        return ref,dict(body_controls=len(curve.position_controls),shoe_controls=len(shoe.position_controls),
            explicit_prior_state=str(explicit),explicit_prior_state_sha256=digest(explicit),
            no_prior_reset_to_warm_pose=True,whole_seconds=source_seconds(shared.p)[[0,-1]].tolist(),
            skipped_unused_dense_temporary_seed_fit=skipped_temporary_root_fit)
    absolute=source_seconds(shared.p);origin=absolute[0];breaks=np.arange(lo,hi,.24)
    dense=[]
    for e in document.get('source_episode_review',{}).get('episodes',[]):
        a=max(lo,e['start_pts']/12800.-origin-.1);b=min(hi,e['end_pts_inclusive']/12800.-origin+.1)
        if b>a:dense.extend(np.arange(a,b,.06))
    changes=np.flatnonzero(shared.p.get('shot_ids',np.full(len(t),'single'))[1:]!=shared.p.get('shot_ids',np.full(len(t),'single'))[:-1])
    for i in changes:dense.extend(t[i]+np.array([-.04,-.02,0.,.02,.04,.06]))
    breaks=np.unique(np.r_[lo,breaks,dense,hi]);breaks=breaks[(breaks>=lo)&(breaks<=hi)]
    # Pure initialization against saved P; this is not source accuracy evidence.
    for attempt in range(8):
        knots=clamped_cubic_knots(breaks);B=BSpline.design_matrix(t,knots,3,extrapolate=True).tocsr();C=np.stack([lsmr(B,initial['P'][:,j],damp=.001,atol=1e-8,btol=1e-8,maxiter=300)[0]for j in range(3)],axis=1)
        init=dict(initial,P=B@C,root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),root_curve_knots=knots,root_curve_controls=C,root_curve_degree=np.array(3))
        seed=ClipReference(shared,shared.root,shared.p,init)
        try:curve=AbsoluteMotionCurve.seed_from_reference(seed.at,breaks,source_seconds=t);break
        except ValueError as error:
            if'path'not in str(error)and'branch'not in str(error):raise
            g=greville_times(knots);dense_t=np.unique(np.r_[g,t]);v=seed.at(dense_t);R=np.concatenate((v['Q'][:,None],v['local_R']),1);angles=np.linalg.norm(_log(R[:-1].swapaxes(-1,-2)@R[1:]),axis=-1);extra=[]
            for a,b in zip(g[:-1],g[1:]):
                if np.any(angles[(dense_t[:-1]>=a)&(dense_t[1:]<=b)].sum(0)>=np.pi-1e-5):extra.append((a+b)/2)
            if not extra:raise
            breaks=np.unique(np.r_[breaks,extra])
    else:raise ValueError('Source rotation path remains unresolved in initialization')
    curve=AbsoluteMotionCurve(knots,C,curve.rotation_controls)
    relative=initial.get('skate_relative_R',np.broadcast_to(np.eye(3),(len(t),2,3,3)))
    shoe_knots=clamped_cubic_knots(np.r_[lo,np.arange(lo+.45,hi,.45),hi]);gt=greville_times(shoe_knots)
    from scipy.spatial.transform import RotationSpline
    Rshoe=np.stack([RotationSpline(t,Rotation.from_matrix(relative[:,f]))(gt).as_matrix()for f in range(2)],1)
    shoe=AbsoluteMotionCurve(shoe_knots,np.zeros((len(gt),3)),Rshoe)
    ref=absolute_reference(seed,curve,root=shared.root);ref.relative_skate_curve=shoe;ref.binding.reference=ref
    return ref,dict(body_controls=len(C),shoe_controls=len(gt),P_seed_max_solver_m=float(np.max(np.linalg.norm(curve.basis(t)@C-initial['P'],axis=-1))),no_fixed_dense_pose_added_back=True,whole_seconds=absolute[[0,-1]].tolist())

def load_reference(project,prepared):
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.prepare import FullSurface
    from .footwear_layout import OBSERVATION_LAYOUT
    from .footwear_geometry import INITIAL
    prepared=Path(prepared);p=load(prepared/'input.npz');initial=load(prepared/'initial.npz');document=json.loads((prepared/'source_evidence.json').read_text(encoding='utf-8'))
    model=json.loads((prepared/'model_identity.json').read_text(encoding='utf-8'));model_path=Path(model['path']);assert digest(model_path)==model['sha256']
    surface=FullSurface(SimpleNamespace(root=prepared,packet=p,model_path=model_path));fixed=load(prepared/'fixed_binding.npz')if(prepared/'fixed_binding.npz').exists()else None
    shared=SimpleNamespace(root=prepared,p=p,times=p['fit_time_seconds'],body=surface.body,geometry=INITIAL.copy()if fixed is None else fixed['shared_geometry'],layout=OBSERVATION_LAYOUT,binding=SimpleNamespace(x=np.zeros(17)),new_refinements=[],refinements=[],hashes=dict(input=digest(prepared/'input.npz'),observations=digest(prepared/'observations.npz'),model=model['sha256']))
    if'shot_ids'in p:camera=ProgrammeCamera(project,p,p['shot_ids']);info=camera.info
    else:
        spec=load(prepared/'static_camera.npz');camera=StaticCropCamera(p,spec['K_input'],spec['R_w2c'],spec['C_world_m']);info=json.loads((prepared/'camera_context.json').read_text(encoding='utf-8'))
    ref,seed_report=seed_reference(shared,initial,document,camera);ref.bounded_memory=True;ref.source_document=document;ref.observations_path=prepared/'observations.npz';ref.export_source_path=prepared/'export_source.npz';ref.ice_plane_known=bool(document['scene_contract']['ice_plane_known']);ref.ice_policy=None
    ref.identity_dependencies=dict(hand_sha256=digest(prepared/'initial.npz'),timing_sha256=digest(prepared/'input.npz'))
    ref.binding=SkateBinding(ref,surface,info)
    if fixed is not None:
        ref.binding.base_parameters=SkateParameters(**json.loads(str(fixed['base_skate_parameters_json'])));ref.binding.R0=fixed['R_FB_initial'];ref.binding.t0=fixed['t_FB_initial_solver'];ref.binding.x=fixed['x'];ref.binding._parameter_key=None;ref.binding._packet=None
        if bool(fixed['collar_constraints_enabled']):ref.binding.enable_collar_constraints()
    ref.binding.reference=ref;ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance={'source':'explicit programme structure document; no invented FS steel pixels'})
    ref.grasp_relations=json.loads((prepared/'grasp_relations.json').read_text(encoding='utf-8'))if(prepared/'grasp_relations.json').exists()else None
    ref.grasp_geometry=json.loads((prepared/'palm_grip_geometry.json').read_text(encoding='utf-8'))if(prepared/'palm_grip_geometry.json').exists()else None
    return ref,camera,info,document,seed_report

def primary_policy(problem):
    joints=problem.body.oj;weight=np.full(len(joints),.07);weight[np.isin(joints,GROUPS['face'])]=.025;weight[np.isin(joints,GROUPS['torso'])]=.35;weight[np.isin(joints,GROUPS['lower_limbs'])]=.45;weight[np.isin(joints,[15,16])]=.045
    problem.objective_block_weights=dict(body_pixels=np.repeat(weight,2),finite_network_pose_prior=.2,finite_parent_root_prior=.04,dynamic_foot_compatibility=.35,finite_relative_skate_wear_prior=.6,full_source_skate_structure=5.,complete_runner_extent=5.,complete_visible_structure_extent=5.,native_source_support_contact=2.,source_pixel_edge_profile=3.)
    problem.trajectory_primary_mode=True;problem.image_safe_trials=True;problem.source_group_baseline={}
    return dict(schema='rinkspace.trajectory-primary-policy.v1',priority=['P/G with reliable torso and source support','both attached skates and finite physics','decoration'],body_confidences=dict(face=.025,torso=.35,lower=.45,upper=.07,boot_occluded_ankle=.045),source_budget_multipliers={k:v for k,v in problem.objective_block_weights.items()if'source'in k or'extent'in k},raw_UV_sigma_use_unchanged=True,finite_ice_and_footwear_objective_retained=True,internal_mm_structure_not_delivery_gate=True,FS_reference_or_holdout_used_for_policy=False)

def run(project,prepared,output,*,wall_seconds,max_accepted,resume=None,gpu_linear=False,device_batch=1024,quality_baseline=None,freeze_root=False,diagnostic_profile=False,protected_source_evidence=None,source_episode_choices=None):
    project,prepared,output=map(Path,(project,prepared,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);log_path=output/'progress.jsonl';print(json.dumps(dict(main_log=str(log_path.resolve()),whole_prepared_state=True)),flush=True);started=time.perf_counter()
    ref,camera,info,document,seed=load_reference(project,prepared);write_json(output/'seed_report.json',seed)
    saved=None;parent_steps=0
    if resume is not None:
        from .skate_checkpoint import validate_checkpoint_identity
        from .absolute_motion_state import curve_from_packet
        from .sequence_skate_state import shoe_curve_from_packet
        saved=load(resume);validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
        for field,name in(('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz')):
            if str(saved[field])!=digest(prepared/name):raise ValueError('Resume changed prepared identity: '+field)
        ref.absolute_curve=curve_from_packet(saved,seed=True);ref.relative_skate_curve=shoe_curve_from_packet(saved,seed=True)
        old_stage=Path(resume).parent.parent
        selection=json.loads((old_stage/'solve/candidate_selection.json').read_text(encoding='utf-8'))
        parent_steps=int(selection['retained_step'])+int(saved.get('parent_accepted_steps',0))
        write_json(output/'resume.json',dict(parameter=str(resume),parameter_sha256=digest(resume),parent_accepted_steps=parent_steps,original_seed_and_objective_retained=True,baseline=str(old_stage/'primary_quality_baseline.json')))
    from .ice_soft import IceSoftPolicy
    if saved is not None:
        policy=IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))
    else:
        policy_path=prepared/'ice_soft_policy.json'
        if not policy_path.is_file():raise FileNotFoundError('Prepared input must carry its explicit finite ice policy: '+str(policy_path))
        policy=IceSoftPolicy(**json.loads(policy_path.read_text(encoding='utf-8'))['policy'])
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,gpu_batch_size=device_batch,local_derivatives='analytic',compatibility_backend='cuda_resident'));execution.attach(ref)
    with log_path.open('a',encoding='utf-8',buffering=1)as log,execution.limits():
        sec=source_seconds(ref.p);problem=SequenceJoint(ref,camera,prepared/'observations.npz',tuple(sec[[0,-1]]),document.get('supports',[]),output/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=policy,record_acceptance_before_jacobian=True)
        problem.body.compatibility_difference_step=1e-8;x=np.r_[ref.binding.x,np.zeros(problem.body.size)];mass=ContinuousMassGeometry(ref);base=PrimaryActions(problem,ref,mass,document);problem.action_factors=SequenceActions(problem,ref,mass,document,x[17:],base_factors=base)
        if saved is not None:
            if saved['x'].shape!=x.shape:raise ValueError('Resume control layout changed')
            np.testing.assert_array_equal(problem.body.state.knots,saved['absolute_knots']);x=saved['x'].copy()
        root_columns=17+(np.arange(problem.body.state.m)[:,None]*69+np.arange(3)[None]).ravel()
        profile_root=x[root_columns].copy()
        if freeze_root:
            problem.active_parameter_indices=np.setdiff1d(np.arange(17,problem.size),root_columns)
            problem.local_control_seconds='whole-programme fixed-P conditional profile; Q/local/skates remain free'
        protected_columns=np.empty(0,int)
        if protected_source_evidence is not None:
            if freeze_root:raise ValueError('Separate source-protected initialization and fixed-P diagnostic modes')
            old_source=json.loads(Path(protected_source_evidence).read_text(encoding='utf-8'))
            stamps=sorted({int(v['pts'])for v in old_source.get('full_source_skate_structures',{}).get('rows',[])}|
                {int(v['pts'])for v in old_source.get('unassigned_source_skate_sets',[])})
            frames=np.searchsorted(ref.p['pts'],stamps);np.testing.assert_array_equal(ref.p['pts'][frames],stamps)
            tau=np.abs(camera.readout_times(frames));nodes=(ref.times[frames,None]+tau[:,None]*np.linspace(-.5,.5,5)).ravel()
            body_controls=np.unique(problem.body.state.basis(nodes).indices)
            chain=np.r_[np.arange(6),*[np.arange(6+(j-1)*3,9+(j-1)*3)for j in(1,2,4,5,7,8)]]
            body_columns=17+(body_controls[:,None]*69+chain[None]).ravel()
            shoe_controls=np.unique(problem.body.state.shoe_curve.basis_sparse(nodes).indices)
            shoe_columns=17+problem.body.state.body_size+(shoe_controls[:,None]*6+np.arange(6)[None]).ravel()
            protected_columns=np.unique(np.r_[body_columns,shoe_columns])
            problem.active_parameter_indices=np.setdiff1d(np.arange(17,problem.size),protected_columns)
            problem.local_control_seconds='global source-protected initialization; existing source-chain controls held, all other global controls free; later full joint release required'
            write_json(output/'source_control_protection.json',dict(source=str(protected_source_evidence),source_sha256=digest(protected_source_evidence),
                source_pts=stamps,columns=protected_columns.tolist(),new_source_not_frozen=True,no_separate_window_or_XYZ_stitching=True,not_final_joint_freeze=True))
        protected_values=x[protected_columns].copy()
        problem.linear_solver_backend='cuda_resident' if gpu_linear else 'sparse_direct'
        problem.checkpoint_extra=dict(programme_context_json=np.array(json.dumps(document.get('programme_scope',document['scene_contract']),sort_keys=True)),prepared_input_sha256=np.array(digest(prepared/'input.npz')),prepared_observations_sha256=np.array(digest(prepared/'observations.npz')))
        weights=primary_policy(problem);write_json(output/'objective_policy.json',weights);np.savez_compressed(output/'objective_block_weights.npz',**problem.objective_block_weights)
        problem.checkpoint_extra.update(objective_phase=np.array('trajectory_primary'),objective_policy_sha256=np.array(digest(output/'objective_policy.json')),objective_block_weights_sha256=np.array(digest(output/'objective_block_weights.npz')))
        problem.checkpoint_extra['parent_accepted_steps']=np.array(parent_steps)
        if saved is not None:
            previous=load(old_stage/'objective_block_weights.npz')
            if set(previous)!=set(problem.objective_block_weights):raise ValueError('Resume objective blocks changed')
            for k,v in previous.items():np.testing.assert_array_equal(v,problem.objective_block_weights[k])
        # A continuation must not silently choose different source segments or
        # physical-identity branches at its new starting pose. Build frozen
        # associations from the same archived global seed, then record them.
        extra_associations={};source_episode_choices=dict(source_episode_choices or {})
        known_episodes={name for name,_ in problem.action_factors.extra_unassigned_episodes}
        if set(source_episode_choices)-known_episodes:raise ValueError('Unknown explicit source association hypothesis')
        hypothesis_records=[]
        for episode_id,factor in problem.action_factors.extra_unassigned_episodes:
            prefix='source_association_extra_'+episode_id+'_'
            if episode_id in source_episode_choices:
                choice=source_episode_choices[episode_id]
                if type(choice)is not int or not 0<=choice<len(factor.episode_factors):raise ValueError('Invalid explicit episode choice: '+episode_id)
                factor.episode_choice=choice
                factor.episode_costs=[v.evaluate(x[17:],False)['image_cost']for v in factor.episode_factors]
                factor.evaluate(x[17:],False)
                hypothesis_records.append(dict(episode_id=episode_id,choice=choice,
                    parent_choice=None if saved is None or prefix+'choice'not in saved else int(saved[prefix+'choice']),
                    common_warm_state_image_costs=factor.episode_costs,
                    meaning='Explicit conditional association hypothesis; source anatomical identity remains unconfirmed; no UV/type/support change'))
            elif saved is not None and prefix+'choice'in saved:
                factor.episode_choice=int(saved[prefix+'choice'])
                if prefix+'candidate_costs'in saved:factor.episode_costs=saved[prefix+'candidate_costs'].tolist()
                else:factor.episode_costs=[v.evaluate(x[17:],False)['image_cost']for v in factor.episode_factors]
                selected=factor.episode_factors[factor.episode_choice]
                for field in('fixed_choice','visible_mask'):
                    if prefix+field in saved:setattr(selected,field,saved[prefix+field].copy())
                if not any(prefix+field in saved for field in('fixed_choice','visible_mask')):
                    # A scale/depth input revision retains the association
                    # hypothesis, but establishes geometry-dependent masks on
                    # its actual warm geometry rather than the archived prior.
                    factor.evaluate(x[17:],False)
            else:
                # Newly read source tracks have no anatomical identity. Choose
                # one episode association on the actual common warm state,
                # record both scores, then keep it fixed during this solve.
                factor.evaluate(x[17:],False)
            extra_associations[prefix+'choice']=np.array(factor.episode_choice)
            extra_associations[prefix+'candidate_costs']=np.asarray(factor.episode_costs)
            selected=factor.episode_factors[factor.episode_choice]
            for field in('fixed_choice','visible_mask'):
                value=getattr(selected,field,None)
                if value is not None:extra_associations[prefix+field]=np.asarray(value).copy()
        if hypothesis_records:
            write_json(output/'source_association_hypotheses.json',hypothesis_records)
            problem.checkpoint_extra['source_association_hypotheses_json']=np.array(json.dumps(hypothesis_records,sort_keys=True))
        elif saved is not None and'source_association_hypotheses_json'in saved:
            problem.checkpoint_extra['source_association_hypotheses_json']=saved['source_association_hypotheses_json'].copy()
        initial_action=problem.action_factors.evaluate(np.zeros(problem.body.size),False)
        frozen={}
        for label,factor in(('contour',problem.action_factors.contour),('runner_extent',problem.action_factors.extent)):
            if factor is None:continue
            for field in('fixed_choice','visible_mask'):
                value=getattr(factor,field,None)
                if value is not None:frozen['source_association_'+label+'_'+field]=np.asarray(value).copy()
        unassigned=problem.action_factors.unassigned
        if unassigned is not None and unassigned.episode_choice is not None:frozen['source_association_episode_choice']=np.array(unassigned.episode_choice)
        frozen.update(extra_associations)
        problem.checkpoint_extra.update(frozen,source_association_origin=np.array('archived global seed for existing tracks; recorded first common warm state for new episode tracks'),source_evidence_sha256=np.array(digest(prepared/'source_evidence.json')))
        np.savez_compressed(output/'source_associations.npz',**frozen)
        # Compare source quality before an expensive complete J. A rejected
        # trial's infinity marker must never become an initial linear-system RHS.
        frozen_baseline=None
        if saved is not None:frozen_baseline=old_stage/'primary_quality_baseline.json'
        elif quality_baseline is not None:frozen_baseline=Path(quality_baseline)
        if frozen_baseline is not None and not diagnostic_profile:
            source_limits=json.loads(frozen_baseline.read_text(encoding='utf-8'))['source'];repairs=[];original_x=x.copy()
            for fraction in(1.,.75,.5,.25,0.):
                candidate=original_x.copy();candidate[17:]*=fraction
                v=problem.action_factors.evaluate(candidate[17:],False);bad=[]
                for row in v['metrics'].get('visible_runner_contours',[])+v['metrics'].get('visible_unassigned_skate_contours',[]):
                    if row['id']not in source_limits:continue
                    before,radius=source_limits[row['id']];after=float(np.percentile(row['outside_visible_band_px'],95))
                    if after>before+.25*radius:bad.append(dict(id=row['id'],before=before,after=after,limit=before+.25*radius))
                repairs.append(dict(fraction=fraction,rejected_rows=bad))
                if not bad:x=candidate;break
            else:raise ValueError('Archived seed fails its own frozen source baseline')
            write_json(output/'source_initialization_guard.json',dict(original_baseline=str(frozen_baseline),trials=repairs,selected_fraction=fraction,one_global_control_state=True,no_per_frame_or_per_shot_repositioning=True))
        tick=time.perf_counter();first=problem.evaluate(x,True);first_J=time.perf_counter()-tick
        if gpu_linear:
            cache=output/'linear_cache';cache.mkdir()
            for name,a in(('data',first['jac'].data),('indices',first['jac'].indices),('indptr',first['jac'].indptr),('residual',first['residual']),('x',x)):np.save(cache/(name+'.npy'),a)
            write_json(cache/'identity.json',dict(shape=list(first['jac'].shape),prepared_input_sha256=digest(prepared/'input.npz'),prepared_observations_sha256=digest(prepared/'observations.npz'),objective_policy_sha256=digest(output/'objective_policy.json'),actual_weighted_CSR=True))
        write_json(output/'weighted_registry.json',verify(first['weighted_blocks'],first['residual'],first['jac']))
        write_json(output/'initial_metrics.json',dict(groups=body_groups(problem,first),weighted_parts={b.name:float(b.residual@b.residual/2)for b in first['weighted_blocks']},structure=first['gate']))
        # A fixed per-source baseline survives every primary stage. Decoration
        # never gets to rank a candidate using a lower aggregate image cost.
        baseline={r['id']:(float(np.percentile(r['outside_visible_band_px'],95)),float(r['pixel_uncertainty_radius_px']))for r in first['action_metrics'].get('visible_runner_contours',[])+first['action_metrics'].get('visible_unassigned_skate_contours',[])}
        torso_before=body_groups(problem,first)['torso']
        if saved is not None:
            original=json.loads((old_stage/'primary_quality_baseline.json').read_text(encoding='utf-8'));baseline.update(original['source']);torso_before=original['torso']
        elif quality_baseline is not None:
            original=json.loads(Path(quality_baseline).read_text(encoding='utf-8'));baseline.update(original['source']);torso_before=original['torso']
        else:original={}
        guard_observations=None;population=original.get('body_observation_reference')
        if population is not None:
            if digest(population['path'])!=population['sha256']:raise ValueError('Fixed body quality population changed')
            old_observations=load(population['path']);np.testing.assert_array_equal(old_observations['pts'],ref.p['pts'])
            guard_observations=old_observations['use']
            if np.any(guard_observations & ~problem.body.obs['use']):raise ValueError('An old body evidence row disappeared')
            for key in('uv','sigma_px'):np.testing.assert_array_equal(old_observations[key][guard_observations],problem.body.obs[key][guard_observations])
        def guard(value,relative_limit):
            checks={}
            for row in value['action_metrics'].get('visible_runner_contours',[])+value['action_metrics'].get('visible_unassigned_skate_contours',[]):
                old,radius=baseline[row['id']];now=float(np.percentile(row['outside_visible_band_px'],95));checks[row['id']]=dict(rejected=now>old+.25*radius,before=old,after=now,original_radius_px=radius)
            torso=body_groups(problem,value,observation_mask=guard_observations)['torso'];checks['torso']=dict(rejected=torso['pixel_p95']>torso_before['pixel_p95']+max(2.,.1*torso_before['pixel_p95']),before=torso_before['pixel_p95'],after=torso['pixel_p95'],meaning='same source population across stages; finite engineering source guard, not 3D truth')
            return checks
        problem.additional_trial_quality=guard
        if diagnostic_profile:problem.image_safe_trials=False
        quality=dict(source=baseline,torso=torso_before,stage_reset_allowed=False)
        if population is not None:quality['body_observation_reference']=population
        write_json(output/'primary_quality_baseline.json',quality);problem.jacobian_seconds_hint=first_J
        del first
        print(json.dumps(dict(event='primary_fit_start',frames=len(ref.times),variables=problem.size,first_full_J_seconds=first_J,wall_seconds=wall_seconds)),flush=True)
        remaining=max(30.,wall_seconds-(time.perf_counter()-started)-30.)
        result,report=problem.solve(x,RunPolicy(wall_seconds=remaining,max_accepted=max_accepted,max_outer=1,allow_penalty_growth=False,stagnation_window=12))
        if freeze_root:np.testing.assert_array_equal(result[root_columns],profile_root)
        if len(protected_columns):np.testing.assert_array_equal(result[protected_columns],protected_values)
        final=problem.evaluate(result,False);write_json(output/'final_metrics.json',dict(groups=body_groups(problem,final),weighted_parts={b.name:float(b.residual@b.residual/2)for b in final['weighted_blocks']},structure=final['gate'],primary_guards=guard(final,.05)))
        candidate=absolute_reference(ref,problem.body.state.current_curve(result[17:]),root=prepared);candidate.relative_skate_curve=problem.body.state.shoe_current_curve(result[17:]);candidate.binding=copy.copy(ref.binding);candidate.binding.reference=candidate
        packet=export_candidate(prepared,output,candidate,camera,info,document.get('supports',[]));C,Rot=mass.evaluate(candidate.at(candidate.times),np.zeros((len(candidate.times),69)));packet['com_world_mid']=mass.com(C)
        packet.update(P_velocity_world_mps=candidate.root_derivative(candidate.times,1)*candidate.binding.ratio,source_shot_ids=ref.p.get('shot_ids',np.full(len(ref.times),'FS_cam7')),trajectory_evidence_status=np.where(np.asarray(problem.body.obs['use']).sum(1)>=4,'source_constrained_estimate','temporal_inference'),position_uncertainty_status=np.full(len(ref.times),'conditional_scale_and_depth; no independent accuracy acceptance'),ice_plane_known=np.array(ref.ice_plane_known),sequence_contact_modes=problem.action_factors.modes.modes,sequence_contact_status=problem.action_factors.modes.status,
            shared_betas=np.asarray(ref.p['shared_betas']).copy(),shared_body_scale=np.asarray(ref.p['body_scale']).copy())
        from .fullbody_fit import body_joint_assignment
        packet['source_body_model_joint_ids']=body_joint_assignment(problem.body.obs).copy()
        if'body_joint_assignment_provenance_json'in problem.body.obs:
            packet['body_joint_assignment_provenance_json']=problem.body.obs['body_joint_assignment_provenance_json'].copy()
        np.savez_compressed(output/'fitted/result.npz',**packet)
        summary=dict(parameter=str(output/'solve/candidate.npz'),result=str(output/'fitted/result.npz'),source_frames=len(ref.times),whole_seconds=sec[[0,-1]].tolist(),accepted_steps=parent_steps+problem.accepted,additional_accepted_steps=problem.accepted,retained_step=parent_steps+report['selection']['retained_step'],stop=report['stop'],first_full_J_seconds=first_J,wall_seconds=time.perf_counter()-started,engineering_ready=True,trajectory_candidate=True,external_validation_status='not_evaluated_here',source_visual_review='pending',accuracy_accepted=False,ice_plane_known=ref.ice_plane_known,primary_guard_baseline=str(output/'primary_quality_baseline.json'),no_per_shot_alignment=True)
        summary.update(root_profile_fixed=freeze_root,diagnostic_profile=diagnostic_profile,
            source_protected_initialization=protected_source_evidence is not None,
            original_primary_guards_pass=not any(v['rejected']for v in guard(final,.05).values()),
            trajectory_candidate=not diagnostic_profile)
        write_json(output/'run_summary.json',summary);write_json(output/'result_status.json',{k:summary[k]for k in ('engineering_ready','trajectory_candidate','external_validation_status','source_visual_review','accuracy_accepted')});print(json.dumps(dict(event='trajectory_primary_complete',accepted=problem.accepted,result=summary['result'])),flush=True)
    execution.close();return summary

def main():
    ap=argparse.ArgumentParser()
    for name in ('project','prepared','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--wall-seconds',type=float,required=True);ap.add_argument('--max-accepted',type=int,required=True)
    ap.add_argument('--memory-gib',type=float,default=12.)
    ap.add_argument('--resume',type=Path);ap.add_argument('--gpu-linear',action='store_true');ap.add_argument('--device-batch',type=int,default=1024);ap.add_argument('--quality-baseline',type=Path)
    ap.add_argument('--freeze-root',action='store_true');ap.add_argument('--diagnostic-profile',action='store_true')
    ap.add_argument('--protected-source-evidence',type=Path)
    ap.add_argument('--source-episode-choice',action='append',default=[],metavar='EPISODE=INDEX',help='Explicit conditional source association; preserves source identity unknown')
    a=ap.parse_args()
    choices={}
    for item in a.source_episode_choice:
        name,choice=item.rsplit('=',1)
        if name in choices:raise ValueError('Duplicate source episode hypothesis')
        choices[name]=int(choice)
    if a.output.exists():raise FileExistsError(a.output)
    from .memory_guard import ProcessMemoryGuard
    memory=ProcessMemoryGuard(a.output.parent/(a.output.name+'_resources'),a.memory_gib)
    try:run(a.project,a.prepared,a.output,wall_seconds=a.wall_seconds,max_accepted=a.max_accepted,resume=a.resume,gpu_linear=a.gpu_linear,device_batch=a.device_batch,quality_baseline=a.quality_baseline,freeze_root=a.freeze_root,diagnostic_profile=a.diagnostic_profile,protected_source_evidence=a.protected_source_evidence,source_episode_choices=choices)
    except MemoryError:
        write_json(a.output/'MEMORY_LIMIT.json',dict(status='stopped_at_process_memory_ceiling',no_auto_restart=True,**memory.snapshot()));raise
    finally:memory.close()
if __name__=='__main__':main()
