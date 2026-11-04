"""Whole-event A/B using the existing body/skate solver and frozen cameras."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation,RotationSpline
from .fullbody_state import matrices,camera_binding
from .rebuild_prepare import load,digest
from .skate_integration import (SkateReference,build,JointContactProblem,
    restore_checkpoint_layers,source_evidence)
from .skate_checkpoint import checkpoint_identity,validate_checkpoint_identity
from .skate_run_policy import RunPolicy
from .skate_delivery import export_candidate
from .skating_action_factors import validate_source_timeline,root_so3_diagnostics
from .fullbody_round import write_json
from .source_support_export import apply_source_support


def _native(data,*names):
    for name in names:
        if name in data:return data[name]
    raise KeyError('Missing explicit native input: '+','.join(names))


class ClipReference(SkateReference):
    """One actual world trajectory, without the historical opening pose layers."""
    def __init__(self,shared,root,p,initial):
        self.__dict__.update(shared.__dict__)
        self.parent=shared;self.root=Path(root);self.p=p;self.times=p['fit_time_seconds']
        self.new_refinements=[];self.refinements=[]
        self.binding=copy.copy(shared.binding);self.binding.x=shared.binding.x.copy()
        self.q_curve=RotationSpline(self.times,Rotation.from_matrix(initial['Q']))
        self.local_curves=[RotationSpline(self.times,Rotation.from_matrix(initial['local_R'][:,j]))for j in range(21)]
        if 'root_curve_schema' in initial:
            if str(initial['root_curve_schema'])!='rinkspace.source-clock-root-bspline.v1':raise ValueError('Unsupported archived root curve schema')
            knots=np.asarray(initial['root_curve_knots'],float);controls=np.asarray(initial['root_curve_controls'],float);degree=int(initial['root_curve_degree'])
            if degree!=3 or knots.ndim!=1 or controls.shape!=(len(knots)-degree-1,3) or np.any(np.diff(knots)<0) or not np.isfinite(knots).all() or not np.isfinite(controls).all():raise ValueError('Invalid archived source-clock root curve')
            self.p_curve=BSpline(knots,controls,degree,extrapolate=True)
            np.testing.assert_allclose(self.p_curve(self.times),initial['P'],rtol=0,atol=1e-10,err_msg='Archived root curve does not reproduce its source exposures')
        else:
            a,b=self.times[[0,-1]];knots=np.r_[[a]*4,np.arange(a+.2,b-1e-8,.2),[b]*4]
            B=BSpline.design_matrix(self.times,knots,3).toarray()
            controls=np.linalg.lstsq(B,initial['P'],rcond=None)[0]
            self.p_curve=BSpline(knots,controls,3,extrapolate=True)
        self.export_source_path=self.root/'export_source.npz'
        from .hand_state import hand_curves,META
        self.hand_metadata={k:initial[k] for k in META if k in initial}
        self.hand_curves=None
        if 'hand_local_R' in initial:
            ht=(initial['hand_prior_pts']-p['pts'][0])/12800. if 'hand_prior_pts' in initial else self.times
            self.hand_curves=hand_curves(ht,initial.get('hand_prior_local_R',initial['hand_local_R']))
        elif self.hand_metadata:raise ValueError('Hand schema present without archived hand state')

    def root_derivative(self,t,order):
        t=np.atleast_1d(t).astype(float);value=self.p_curve(t,order)
        for ws,x in self.new_refinements:
            delta=np.zeros((len(t),69))
            delta[:,ws.active]=ws.basis(t,order)@np.asarray(x).reshape(ws.m,ws.d)
            value+=delta[:,:3]
        return value

    def at(self,t):
        t=np.atleast_1d(t).astype(float)
        result=dict(P=self.p_curve(t),Q=self.q_curve(t).as_matrix(),
            local_R=np.stack([c(t).as_matrix()for c in self.local_curves],axis=1))
        if self.hand_curves is not None:
            from .hand_state import evaluate_hands
            result['hand_local_R']=evaluate_hands(self.hand_curves,t)
        for ws,x in self.new_refinements:
            d=ws.increments(x,t);result['P']+=d[:,:3]
            result['Q']=result['Q']@matrices(d[:,3:6])
            result['local_R']=result['local_R']@matrices(d[:,6:].reshape(-1,21,3))
        return result


def initialize_clip(project,base_root,output,native_path,prior_path,shot_id,*,heldout_pts=(),prior_label='GVHMR'):
    """Use real pixels and an explicit camera-frame prior; shared S01 supplies shape only."""
    project,base_root,output=map(Path,(project,base_root,output));output.mkdir(parents=True,exist_ok=True)
    native=load(native_path);prior=load(prior_path)
    pts=_native(native,'pts','source_pts').astype(np.int64)
    source_uv=_native(native,'body_uv','source_uv','native_coco17_xy')
    score=_native(native,'body_score','source_score','keypoint_scores')
    qualified=_native(native,'body_use','source_qualified','use')
    if qualified.ndim==1:qualified=np.broadcast_to(qualified[:,None],score.shape).copy()
    qualified=qualified.astype(bool)&np.isfinite(source_uv).all(axis=2)&(score>.25)
    missing=native.get('source_mapping_missing',~qualified.any(axis=1)).astype(bool)
    seed=load(project/'events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.npz')
    shared,camera0,info0,catalog=build(project,base_root,output/'shared')
    # Re-use its measured fixed attachment/shape identity, never its body pose.
    validate_checkpoint_identity(seed,shared.fixed_checkpoint_identity)
    shared.binding.x=seed['x'][:17].copy();shared.binding.enable_collar_constraints()
    old=load(base_root/'input.npz');n=len(pts)
    p=dict(pts=pts,fit_time_seconds=(pts-pts[0])/12800.,shared_betas=old['shared_betas'],
        body_scale=old['body_scale'],metres_per_scene_unit=old['metres_per_scene_unit'],
        source_uv=source_uv,source_score=score,source_qualified=qualified,
        source_bbox=_native(native,'bbox','source_bbox','bbox_native_xyxy'),
        lower_body_inferred=missing.copy(),source_identity_suspect=missing.copy())
    shell=type('Input',(),{'p':p})()
    camera,info,state,pose=camera_binding(project,shell,shot_id=shot_id)
    p=camera.p;p['camera_rotation']=pose['rotation'];p['camera_centre']=pose['centre']
    p['camera_values']=pose['values']
    prior_times=(prior['pts']-pts[0])/12800.
    raw_local=matrices(prior['body_pose'].reshape(-1,21,3))
    local=np.stack([RotationSpline(prior_times,Rotation.from_matrix(raw_local[:,j]))(p['fit_time_seconds']).as_matrix()for j in range(21)],axis=1)
    raw_world=prior['R_w2c'].transpose(0,2,1)@Rotation.from_rotvec(prior['global_orient']).as_matrix()
    Q=RotationSpline(prior_times,Rotation.from_matrix(raw_world))(p['fit_time_seconds']).as_matrix()
    from .hand_state import prior_hand_fields
    hand_fields=prior_hand_fields(prior,p['fit_time_seconds'],pts[0],prior_label,digest(prior_path))
    coco=shared.body.forward_rotations(local,**({} if not hand_fields else {'hand_local_R':hand_fields['hand_local_R']}))[0]
    rel=np.einsum('nij,nkj->nki',pose['rotation']@Q,coco)
    # Brown ray unprojection uses the same explicit camera25 implementation
    # as preparation; the helper records its native-camera closure check.
    from .camera_bridge import native_to_rectified,rectified_intrinsics
    camera25=_native(native,'camera25','camera25_native')
    Pcam=np.full((n,3),np.nan)
    for i in range(n):
        mask=qualified[i]
        if mask.sum()<4:continue
        uv,ray_valid=native_to_rectified(source_uv[i,mask],camera25[i])
        if ray_valid.sum()<4:continue
        uv=uv[ray_valid]
        K=rectified_intrinsics(camera25[i]);rays=(uv-K[:2,2])/np.diag(K)[:2]
        xyz=rel[i,mask][ray_valid];w=np.sqrt(np.clip(score[i,mask][ray_valid],0,1))
        A=np.zeros((2*len(xyz),3));rhs=np.empty(2*len(xyz))
        A[0::2,0]=1;A[1::2,1]=1;A[0::2,2]=-rays[:,0];A[1::2,2]=-rays[:,1]
        rhs[0::2]=rays[:,0]*xyz[:,2]-xyz[:,0];rhs[1::2]=rays[:,1]*xyz[:,2]-xyz[:,1]
        Pcam[i]=np.linalg.lstsq(A*np.repeat(w,2)[:,None],rhs*np.repeat(w,2),rcond=None)[0]
    valid=np.isfinite(Pcam).all(axis=1)&(Pcam[:,2]>0)
    if valid.sum()<4:raise ValueError('Insufficient real body observations for world-root initialization')
    P=pose['centre']*float(p['metres_per_scene_unit'])+np.einsum('nji,nj->ni',pose['rotation'],np.nan_to_num(Pcam))
    for j in range(3):P[:,j]=np.interp(p['fit_time_seconds'],p['fit_time_seconds'][valid],P[valid,j])
    p['pose_axis_angle']=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(n,21,3)
    p['raw_root_prior_Q_world']=Q
    initial_path=output/'initial.npz'
    if initial_path.exists():
        previous=load(initial_path)
        np.testing.assert_array_equal(previous['pts'],pts)
        for name,value in dict(P=P,Q=Q,local_R=local,**{k:v for k,v in hand_fields.items() if k not in ('hand_pose_schema_version','hand_mean_convention','hand_pose_origin','hand_prior_sha256')}).items():
            np.testing.assert_allclose(previous[name],value,rtol=0,atol=1e-12)
        # Reuse the saved arrays exactly across A/B; tiny BLAS roundoff is
        # verified above, not written as a distinct initialization.
        P,Q,local=(previous[k] for k in ('P','Q','local_R'))
    else:np.savez_compressed(initial_path,pts=pts,P=P,Q=Q,local_R=local,**hand_fields)
    ref=ClipReference(shared,output,p,dict(P=P,Q=Q,local_R=local,**hand_fields))
    ref.hashes=dict(native_input_sha256=digest(native_path),camera_prior_sha256=digest(prior_path),
        shared_s01_parameters_sha256=digest(project/'events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.npz'))
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],
        provenance={'status':'new clip has no admitted shoe pixels; held-out shoe checks remain outside fitting'})
    if shot_id=='s01':
        from .fullbody_fit import footwear_rows
        gear,provenance=footwear_rows(base_root,np.arange(n),pts)
        held=set(map(int,heldout_pts))
        for name in ('visible_rows','blade_rows','landmark_rows'):
            ref.footwear_observations[name]=[row for row in getattr(gear,name) if int(row['pts']) not in held]
        ref.footwear_observations['provenance']=dict(provenance,heldout_pts=sorted(held),
            status='existing source-admitted shoe/blade observations inherited except explicit held-out exposures')
    np.savez_compressed(output/'input.npz',**p)
    compact=base_root/'compact_smplx.npz'
    if not (output/'compact_smplx.npz').exists():(output/'compact_smplx.npz').write_bytes(compact.read_bytes())
    use=qualified.copy();heldout=np.isin(pts,np.asarray(heldout_pts,np.int64))
    # Protect one neighbouring exposure on each side; these frames remain
    # correlated development checks, never independent 3D ground truth.
    hold=np.convolve(heldout.astype(int),[1,1,1],mode='same')>0;use[hold]=False
    sigma=np.maximum(2.,8.*(1-np.clip(np.nan_to_num(score),0,1)))
    np.savez_compressed(output/'observations.npz',pts=pts,uv=source_uv,sigma_px=sigma,use=use)
    np.savez_compressed(ref.export_source_path,pts=pts,source_pts=pts,time_base=np.array([1,12800]),
        state_schema_version=np.array('rinkspace.fullbody.action-event.v1'),
        source_lower_body_inferred=missing,source_identity_suspect=missing,
        window_mask=np.zeros(n,bool),source_observation_use=use,
        coordinate_convention=np.array('same frozen ice world; Z up; conditional metres'),
        shared_geometry=ref.geometry,user_accepted=np.array(False))
    write_json(output/'initialization.json',dict(source_pts=[int(pts[0]),int(pts[-1])],source_exposures=n,
        camera=info,hashes=ref.hashes,shared_shape_and_attachment_fixed=True,
        body_pose_origin=f'real uniform-25Hz {prior_label} incam; world rotation through this frozen camera',
        root_origin='weighted native source body rays and fixed shared body size; network translation unused',
        interpolation='SO(3) RotationSpline of actual root/local rotations; cubic root position least-squares, 0.2s knots',
        inferred_root_pts=pts[~valid].tolist(),mapping_missing_pts=pts[missing].tolist(),
        heldout_pts=pts[heldout].tolist(),heldout_protection_pts=pts[hold].tolist(),
        prior_pose_anchor=f'current clip {prior_label} only; no historical opening or S01 body-pose layers'))
    return ref,camera,info


def summarize(packet,evidence):
    clock=validate_source_timeline(packet['pts'],(1,12800),source_id=evidence['source_id'])
    root=root_so3_diagnostics(clock,packet['Q_world_mid'])
    clearance=packet['skate_clearance_m_mid'];support=packet['skate_support_gap_m_mid']
    return dict(source_exposures=len(clock),foot_frames=int(clearance.size),
        maximum_nonpenetration_violation_m=float(np.maximum(-clearance,0).max()),
        p95_nonpenetration_violation_m=float(np.quantile(np.maximum(-clearance,0),.95)),
        support_observed_exposures=int(np.isfinite(support).sum()),
        maximum_observed_support_raw_height_m=float(np.nanmax(support))if np.isfinite(support).any()else None,
        maximum_root_angle_step_deg=float(np.rad2deg(root.principal_increment_rad.max())),
        maximum_root_angular_speed_radps=float(np.linalg.norm(root.omega_world_radps,axis=1).max()),
        maximum_cavity_breach_m=float(packet['foot_cavity_max_violation_m_mid'].max()),
        unknown_contact_foot_frames=int(np.count_nonzero(packet['skate_contact_mode_mid']=='unknown')),
        independent_3d_truth=False)


def register_player(project,base_root,root,clip_id):
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    programme=Path(project)/'events/ankara_2026/programmes/okada_mei_sp'
    base=json.loads((programme/'viewer3d_fullbody_1102.json').read_text(encoding='utf-8'))
    config=dict(base,shot='action_'+clip_id,source_run=str(root),source_identity_field='source_identity_suspect',
        default_variant='A',initial_source_pts=int(load(root/'input.npz')['pts'][0]),comparison_variants=['A','B'],
        variants=[dict(id=arm,label=clip_id+' '+arm+(' 基础路线'if arm=='A'else' 有证据门控动作因子'),
            kind='fullbody',result=arm+'/fitted/result.npz',shell_topology='saved',
            body_reference_label='same_uniform25_gvhmr',body_refinement_intervals_seconds=[])for arm in ('A','B')])
    path=programme/('viewer3d_action_'+clip_id+'_1103.json')
    if len(str(path).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    write_json(path,config)
    catalog=ResultCatalog(project=project,data_root=Path(base_root).parents[1],config=path)
    for arm in ('A','B'):prepare_variant(catalog,catalog.variant(arm))
    return str(path)


def run_arm(project,base_root,root,native,prior,clip_id,evidence,arm,budget=300):
    root=Path(root);out=root/arm;out.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'),arm=arm,clip=clip_id)),flush=True)
    ref,camera,info=initialize_clip(project,base_root,root,native,prior,evidence['shot_id'],
        heldout_pts=evidence.get('heldout_pts',[]))
    supports=source_evidence(root,shot_id=evidence['shot_id'],source_pts=ref.p['pts'],records=evidence.get('supports',[]))
    write_json(out/'source_evidence.json',supports)
    first,last=ref.p['pts'][[0,-1]]/12800.;padding=max(abs(float(camera.p['rig'][37]))/2,.01)
    seconds=(float(first-padding),float(last+padding))
    with (out/'fit_progress.jsonl').open('a',encoding='utf-8',buffering=1)as log:
        problem=JointContactProblem(ref,camera,root/'observations.npz',seconds,supports,out/'fit',log,
            'continuous_shot',optimize_shared=False)
        from .action_live import GlideActionFactors
        problem.action_factors=GlideActionFactors(problem,evidence,sigma_mps=evidence['glide_sigma_mps'],enabled=arm=='B')
        initial=np.r_[ref.binding.x,np.zeros(problem.body.size)]
        diagnostics=problem.action_factors.evaluate(initial[17:],False)
        write_json(out/'initial_action_diagnostics.json',diagnostics['metrics'])
        x,report=problem.solve(initial,RunPolicy(wall_seconds=budget,max_accepted=3,max_outer=1,allow_penalty_growth=False))
    ref.binding.x=x[:17].copy();ref.new_refinements.append((problem.body.state,x[17:].copy()))
    np.testing.assert_array_equal(x[:17],initial[:17])
    packet=export_candidate(root,out,ref,camera,info,supports)
    phases=np.full((len(packet['pts']),2),'unknown',dtype='<U32')
    for interval in evidence['intervals']:
        foot=interval.get('physical_foot')
        if foot in (0,1):
            mask=(packet['pts']>=interval['start_pts'])&(packet['pts']<=interval['end_pts'])
            phases[mask,foot]=interval['phase']
    apply_source_support(packet,supports,shot_id=evidence['shot_id'])
    packet['source_action_phase_mid']=phases
    packet['ice_soft_policy_json']=np.array(json.dumps(vars(problem.ice_policy),sort_keys=True))
    packet['ice_is_delivery_gate']=np.array(False)
    packet['action_evidence_json']=np.array(json.dumps(evidence,sort_keys=True))
    packet['initialization_sha256']=np.array(digest(root/'initial.npz'))
    packet['costs_json']=np.array(json.dumps(report['parts']))
    np.savez_compressed(out/'fitted/result.npz',**packet)
    params=dict(load(out/'fit/candidate.npz'),prior_identity_json=np.array(json.dumps(ref.hashes,sort_keys=True)))
    target=Path(project)/'events/ankara_2026/programmes/okada_mei_sp'/('action_'+clip_id+'_'+arm+'_1103.npz')
    if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    np.savez_compressed(target,**params)
    metrics=summarize(packet,evidence)
    with (out/'fit_progress.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(dict(event='export_complete',metrics=metrics))+'\n')
    write_json(out/'metrics.json',dict(arm=arm,clip=clip_id,metrics=metrics,run=report,
        action_diagnostics=problem.action_factors.evaluate(x[17:],False)['metrics'],parameters=str(target)))
    return metrics


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--base-root',required=True)
    p.add_argument('--root',required=True);p.add_argument('--native',required=True);p.add_argument('--prior',required=True)
    p.add_argument('--clip',required=True);p.add_argument('--evidence',required=True);p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--budget',type=float,default=300)
    a=p.parse_args(argv);e=json.loads(Path(a.evidence).read_text(encoding='utf-8'))['clips'][a.clip]
    run_arm(a.project,a.base_root,a.root,a.native,a.prior,a.clip,e,a.arm,a.budget)
    if all((Path(a.root)/arm/'fitted/result.npz').exists()for arm in ('A','B')):
        print(json.dumps(dict(player_config=register_player(a.project,a.base_root,Path(a.root),a.clip))),flush=True)


if __name__=='__main__':main()
