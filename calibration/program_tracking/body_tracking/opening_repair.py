"""Restore cropped-opening inference on the current continuous body/boot state."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.sparse import coo_matrix,csr_matrix,vstack
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import FullSurface,prepare_variant
from .rebuild_prepare import load,digest
from .fullbody_state import WindowState,matrices,camera_binding
from .fullbody_fit import FullBodyProblem,verify_problem
from .footwear_binding import IntegratedBodyState,FootwearBinding
from .boot_integration import edges_for,export
from .fullbody_round import write_json

DONOR_LEGACY_PTS=229175552
START,STATIONARY_END,END=22.12,24.44,25.0


def smoothstep(u):
    u=np.clip(u,0.,1.)
    return u**3*(10-15*u+6*u*u)


class OpeningReference:
    def __init__(self,parent,camera):
        self.parent=parent;self.refinements=[]
        p=parent.p;raw=load(parent.root/'source_cache/raw_shot_01.npz')
        official=np.flatnonzero(p['source_legacy_pts']==DONOR_LEGACY_PTS)
        original=np.flatnonzero(raw['source_pts']==DONOR_LEGACY_PTS)
        if len(official)!=1 or len(original)!=1:raise ValueError('Reviewed upright opening donor must have an exact source-clock match')
        self.donor_frame=int(official[0]);ri=int(original[0]);self.donor_pts=int(p['pts'][self.donor_frame])
        self.donor_R=matrices(raw['body_pose'][ri].reshape(21,3))
        lookup={int(v):i for i,v in enumerate(p['source_legacy_pts'])}
        pairs=[(lookup[int(v)],i) for i,v in enumerate(raw['source_pts']) if int(v) in lookup and raw['chosen_window'][i]>=0]
        pairs=sorted(pairs);fi=np.array([a for a,b in pairs]);raw_i=np.array([b for a,b in pairs])
        pose=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],float(camera.p['first_control_time']),
            float(camera.p['control_dt']),camera.p['motion_times'][fi])
        q=pose['rotation'].transpose(0,2,1)@matrices(raw['global_orient'][raw_i])
        # Retain the cached heading, but do not transfer portrait roll/pitch.
        forward=q[:,:,2];heading=np.unwrap(np.arctan2(forward[:,1],forward[:,0]))
        self.heading_times=p['fit_time_seconds'][fi];self.heading=heading
        anchor_i=int(np.searchsorted(p['pts'],round(STATIONARY_END*12800)))
        self.anchor=parent.at([p['fit_time_seconds'][anchor_i]])['P'][0]
        self.origin=float(p['pts'][0]/12800)
        self.preparation_path=parent.root/'opening_repair_1102/preparation_pose.npz'
        self.preparation=None
        if self.preparation_path.is_file():
            self.preparation=load(self.preparation_path)
            self.anchor=self.preparation['P'].copy()
            q=self.preparation['Q'];yaw=np.arctan2(q[1,2],q[0,2])
            self.heading+=np.arctan2(np.sin(yaw-self.heading[0]),np.cos(yaw-self.heading[0]))

    def __getattr__(self,name):return getattr(self.parent,name)

    def at(self,t):
        t=np.atleast_1d(t).astype(float);old=self.parent.at(t);source=t+self.origin
        weight=1-smoothstep((source-STATIONARY_END)/(END-STATIONARY_END))
        heading=np.interp(t,self.heading_times,self.heading);c,s=np.cos(heading),np.sin(heading)
        upright=np.zeros((len(t),3,3));upright[:,0,0]=-s;upright[:,1,0]=c;upright[:,2,1]=1;upright[:,0,2]=c;upright[:,1,2]=s
        qlog=Rotation.from_matrix(old['Q'].transpose(0,2,1)@upright).as_rotvec()
        dlog=Rotation.from_matrix((old['local_R'].swapaxes(-1,-2)@self.donor_R).reshape(-1,3,3)).as_rotvec().reshape(-1,21,3)
        r=dict(P=old['P']+weight[:,None]*(self.anchor-old['P']),Q=old['Q']@matrices(weight[:,None]*qlog),
            local_R=old['local_R']@matrices(weight[:,None,None]*dlog),times=t)
        if self.preparation is not None:
            # This is a cross-shot pose prior, not measured hidden motion.
            # Progressively release it as S01 reveals the legs and starts turning.
            strength=(1-smoothstep((source-START)/(STATIONARY_END-START)))*weight
            for j in (1,2,4,5,7,8,10,11):
                relative=Rotation.from_matrix(r['local_R'][:,j-1].transpose(0,2,1)@self.preparation['local_R'][j-1]).as_rotvec()
                r['local_R'][:,j-1]=r['local_R'][:,j-1]@matrices(strength[:,None]*relative)
        # Exact parent identity outside support, including the final boundary.
        outside=source>=END
        for key in ('P','Q','local_R'):r[key][outside]=old[key][outside]
        for ws,x in self.refinements:
            d=ws.increments(x,t);r.update(P=r['P']+d[:,:3],Q=r['Q']@matrices(d[:,3:6]),local_R=r['local_R']@matrices(d[:,6:].reshape(-1,21,3)))
        return r

    def world(self,reference,delta=None):return self.parent.world(reference,delta)


class OpeningWindow(WindowState):
    def __init__(self,reference,seconds,mode):
        # Left support extends before the first source frame: its value is free,
        # unlike an interior correction window that vanishes at both ends.
        super().__init__(reference,(START-.52,END),mode,spacing=.25,taper=.35)
        self.seconds=(START,END)
        from .footwear_geometry import LEG_DOFS
        self.active=np.array(list(range(6))+[6+(j-1)*3+a for j in (3,6,9,12,15,16,17,18,19) for a in range(3)]+
            [6+(j-1)*3+a for j,a in LEG_DOFS])
        self.d=len(self.active);self.size=self.m*self.d


class OpeningProblem(FullBodyProblem):
    def __init__(self,reference,camera,observations,log):
        super().__init__(reference,camera,observations,(START,END),'full',log,state_factory=OpeningWindow)
        self.mid_times=reference.times[self.frames];self.mid_ref=reference.at(self.mid_times)
        self.standing=reference.p['pts'][self.frames]/12800<=STATIONARY_END+1e-9
        self.velocity_scale=np.array([.1,.1,.15])
        h=1e-4
        self.reference_velocity=(reference.at(self.mid_times+h)['P']-reference.at(self.mid_times-h)['P'])/(2*h)

    def visible_boot_edges(self,world):return edges_for(world,self.frames,self.projector,self.reference.binding)

    def physical_rows(self,delta):
        s=self.reference.world(self.mid_ref,delta);torso=s['native'][:,12]-s['native'][:,0]
        torso/=np.linalg.norm(torso,axis=1)[:,None]
        # A coarse standing condition from the user's observed opening; no
        # individual blade station/physical left-right identity is declared.
        support=np.min(s['gear'][:,:,14:,2],axis=(1,2))
        root_up=s['Q'][:,:,1]
        location=(s['P'][:,:2]-self.reference.anchor[:2])/.35
        return np.c_[torso[:,:2]/.20,np.maximum(.85-torso[:,2],0)/.10,support/.05,
            root_up[:,:2]/.25,location]*self.standing[:,None]/np.sqrt(2)

    def evaluate(self,x,jacobian=False):
        stream=self.log;self.log=None
        try:base=super().evaluate(x,jacobian)
        finally:self.log=stream
        if base.get('opening_added'):return base
        d=self.state.increments(x,self.mid_times);physical=self.physical_rows(d)
        velocity=(self.reference_velocity+self.state.basis(self.mid_times,1)@x.reshape(self.state.m,self.state.d)[:,:3])
        velocity=velocity/self.velocity_scale*self.standing[:,None]/np.sqrt(2)
        if jacobian:
            h=1e-5;columns=[]
            for active in self.state.active:
                a=d.copy();b=d.copy();a[:,active]+=h;b[:,active]-=h
                columns.append((self.physical_rows(a)-self.physical_rows(b))/(2*h))
            local=np.stack(columns,-1);B=self.state.basis(self.mid_times);fr,kn=np.nonzero(B)
            val=local[fr]*B[fr,kn,None,None]
            rr=np.broadcast_to(fr[:,None,None]*physical.shape[1]+np.arange(physical.shape[1])[None,:,None],val.shape).ravel()
            cc=np.broadcast_to(kn[:,None,None]*self.state.d+np.arange(self.state.d),val.shape).ravel()
            J=coo_matrix((val.ravel(),(rr,cc)),shape=(physical.size,self.size)).tocsr()
            V=self.state.basis(self.mid_times,1)*self.standing[:,None]/np.sqrt(2)
            fr,kn=np.nonzero(V);val=V[fr,kn,None]/self.velocity_scale
            r=fr[:,None]*3+np.arange(3);c=kn[:,None]*self.state.d+np.arange(3)
            JV=coo_matrix((val.ravel(),(r.ravel(),c.ravel())),shape=(velocity.size,self.size)).tocsr()
            base['jac']=vstack((base['jac'],J,JV),format='csr')
        base['residual']=np.r_[base['residual'],physical.ravel(),velocity.ravel()]
        base['parts'].update(opening_upright_and_support=float(np.sum(physical**2)/2),opening_near_static_velocity=float(np.sum(velocity**2)/2))
        base['opening_added']=True
        if stream:stream.write(json.dumps(dict(event='opening_evaluation',with_jacobian=jacobian,evaluation=self.evaluations,
            cost=float(base['residual']@base['residual']/2),parts=base['parts'],seconds=time.perf_counter()-self.start))+'\n')
        return base


def setup(project,root,out):
    parent=IntegratedBodyState(root,root/'fullbody_1102');camera,info,_,_=camera_binding(project,parent)
    catalog=ResultCatalog(project=project,data_root=root.parents[1],config=project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_boot_1102.json')
    binding=FootwearBinding(parent,FullSurface(catalog),info);binding.x=load(root/'boot_integration_1102/joint_result.npz')['x'][:14].copy();parent.binding=binding
    for i in (1,2):
        z=load(root/'boot_integration_1102'/f'w{i}/result.npz');ws=WindowState(parent,z['seconds'],'full')
        np.testing.assert_array_equal(ws.knots,z['knots']);parent.refinements.append((ws,z['x']))
    ref=OpeningReference(parent,camera)
    return ref,camera,info,catalog


def deliver(project,root,out,ref,problem,x,info):
    programme=project/'events/ankara_2026/programmes/okada_mei_sp'
    parameters=programme/'opening_repair_1102.npz';record_path=programme/'opening_repair_1102.json'
    for path in (parameters,record_path):
        if len(str(path.resolve()).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path length')
    preparation_camera=load(out/'preparation_camera.npz')
    np.savez_compressed(parameters,x=x,knots=problem.state.knots,active=problem.state.active,
        seconds=problem.state.seconds,taper=problem.state.taper,donor_local_R=ref.donor_R,
        heading_times=ref.heading_times,heading=ref.heading,initial_anchor=ref.anchor,
        preparation_local_R=ref.preparation['local_R'] if ref.preparation is not None else ref.donor_R,
        preparation_P=ref.preparation['P'],preparation_Q=ref.preparation['Q'],
        preparation_candidate_local_R=ref.preparation['candidate_local_R'],
        preparation_candidate_P=ref.preparation['candidate_P'],preparation_candidate_Q=ref.preparation['candidate_Q'],
        preparation_camera_rig=preparation_camera['rig'],preparation_camera_ptz=preparation_camera['ptz'],
        preparation_intersection_scene=preparation_camera['intersection_scene'],
        source_donor_pts=np.array(ref.donor_pts),source_time_base=np.array([1,12800]))
    for key,value in load(parameters).items():
        if key=='x':np.testing.assert_array_equal(value,x)
    evidence={}
    for name in ('inputs.json','ledger.json','fit/config.json','fit/result.json','derivative_checks.json',
                 'opening_report.json','results.txt','source_opening.png','source_opening.json',
                 'player_check/checks.json','top_check/checks.json','verification.json',
                 'preparation_camera.json','preparation_geometry_overlay.png','preparation_geometry_check.json',
                 'preparation_pose.json','preparation_pose_overlay.png','metrics_plot.png','metrics_plot.json',
                 'opening_visual_review.json'):
        path=out/name
        if path.is_file():evidence[name]=dict(path=str(path),sha256=digest(path))
    record=dict(schema='rinkspace.opening-body-repair.v1',event='ankara_2026',programme='okada_mei_sp',
        source_seconds=[START,END],source_time_base=[1,12800],body_refined_frames=145,
        near_static_condition_seconds=[START,STATIONARY_END],lower_body_inferred_frames=125,
        source_user_observation='Opening nearly stationary turn; previous secondary view shows crossed legs near B2/F2-R2 intersection.',
        donor=dict(legacy_pts=DONOR_LEGACY_PTS,official_pts=ref.donor_pts,seconds=ref.donor_pts/12800,
            origin='previously reviewed upright upper body; raw cached pose, no network rerun'),
        parent=dict(path=str(root/'boot_integration_1102/fitted/result.npz'),sha256=digest(root/'boot_integration_1102/fitted/result.npz')),
        result=dict(path=str(out/'fitted/result.npz'),sha256=digest(out/'fitted/result.npz')),
        parameters=dict(path=str(parameters),sha256=digest(parameters)),camera=info,
        priors=dict(velocity_scale_m_s=problem.velocity_scale.tolist(),torso_horizontal_axis_scale=.20,
            root_horizontal_axis_scale=.25,root_neighbourhood_scale_m=.35,
            support_envelope_height_scale_m=.05,contact_point_identity='unknown; coarse standing condition, not a source contact measurement'),
        boundary='First frame free; parent attachment at25s is C2; later body and footwear unchanged',
        active_local_joints=[1,2,3,4,5,6,7,8,9,12,15,16,17,18,19],
        preparation=dict(pose_path=str(ref.preparation_path),pose_sha256=digest(ref.preparation_path),
            source_pts=255488,seconds=19.96,side_identity='conditional; both side candidates retained',
            lower_pose_release_seconds=[START,STATIONARY_END]),
        state_ownership='One continuous P/Q/local_R drives the complete body, registered lower boots, blades and local uppers at exposure nodes.',
        input_contract=json.loads((out/'inputs.json').read_text(encoding='utf8')),
        observation_policy='Original source UV/use unchanged; no fabricated off-screen lower body points',
        evidence=evidence,implementation={str(f):digest(f) for f in (
            Path(__file__),Path(__file__).with_name('opening_preparation.py'),Path(__file__).with_name('opening_review.py'),
            Path(__file__).with_name('fullbody_fit.py'),project/'scripts/run_opening_repair.py')},
        viewer_entry=str(project/'START_3D_COMPARISON.cmd'),
        replay_entry=str(project/'scripts/run_opening_repair.py'))
    write_json(record_path,record)
    return dict(binding=str(record_path),parameters=str(parameters))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['prepare','fit','export','run'],default='run');a=p.parse_args(argv)
    project,root,out=Path(a.project),Path(a.root),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))),flush=True)
    ref,camera,info,catalog=setup(project,root,out)
    obs_path=root/'boot_integration_1102/observations.npz';obs=load(obs_path);np.savez_compressed(out/'observations.npz',**obs)
    ledger=json.loads((root/'boot_integration_1102/ledger.json').read_text(encoding='utf8'))
    ledger.update(parent_ledger=str(root/'boot_integration_1102/ledger.json'),parent_sha256=digest(root/'boot_integration_1102/ledger.json'),
        opening_user_evidence='User reports cropped opening is almost standing in place while rotating',
        opening_inference=dict(donor_legacy_pts=DONOR_LEGACY_PTS,donor_official_pts=ref.donor_pts,
            near_static_seconds=[START,STATIONARY_END],transition_end=END,
            preparation_source_pts=255488,preparation_pose_sha256=digest(ref.preparation_path),
            unseen_lower_body='crossed-leg preparation prior progressively released; original offscreen observations remain excluded'),
        observation_array_sha256=digest(out/'observations.npz'))
    write_json(out/'ledger.json',ledger)
    contract=dict(parent_result_sha256=digest(root/'boot_integration_1102/fitted/result.npz'),reference_hashes=ref.hashes,
        raw_cache_sha256=digest(root/'source_cache/raw_shot_01.npz'),camera=info,observation_sha256=digest(obs_path),
        donor_pts=ref.donor_pts,source_seconds=[START,END],standing_until=STATIONARY_END,
        velocity_scale_m_s=[.1,.1,.15],torso_horizontal_scale=.20,standing_support_height_scale_m=.05,
        root_horizontal_scale=.25,root_neighbourhood_scale_m=.35,
        preparation_pose_sha256=digest(ref.preparation_path),preparation_camera_sha256=digest(out/'preparation_camera.npz'))
    if (out/'inputs.json').is_file() and json.loads((out/'inputs.json').read_text(encoding='utf8'))!=contract:raise ValueError('Opening inputs changed; retain previous output and choose a separate run')
    write_json(out/'inputs.json',contract)
    if a.stage=='prepare':return
    with (out/'fit_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem=OpeningProblem(ref,camera,out/'observations.npz',log)
        if (out/'fit/result.npz').is_file():
            z=load(out/'fit/result.npz');x=z['x'];report=json.loads((out/'fit/result.json').read_text(encoding='utf8'))
            np.testing.assert_array_equal(z['knots'],problem.state.knots);np.testing.assert_array_equal(z['active'],problem.state.active)
        elif a.stage in ('fit','run'):
            print(json.dumps(dict(derivatives=verify_problem(problem,out/'derivative_checks.json'))),flush=True)
            warm=out/'history/preparation_not_yet_used/fit/result.npz'
            initial=None
            if warm.is_file():
                old=load(warm);initial=np.zeros((problem.state.m,problem.state.d))
                previous=old['x'].reshape(problem.state.m,-1)
                for k,active in enumerate(old['active']):
                    if active>=3:initial[:,np.flatnonzero(problem.state.active==active)[0]]=previous[:,k]
                initial=initial.ravel()
                write_json(out/'initialization.json',dict(path=str(warm),sha256=digest(warm),reason='upper-pose numerical seed; translation reinitialized from the preparation view, new leg variables zero'))
            x,report=problem.solve(out/'fit',initial_x=initial);print(json.dumps(report),flush=True)
        else:raise FileNotFoundError('No fitted opening state')
    write_json(out/'joint_result.json',report);ref.refinements.append((problem.state,x))
    if a.stage in ('export','run'):
        config=json.loads((out/'fit/config.json').read_text(encoding='utf8'))
        config.update(mode='cropped_opening',boundary='Left correction support begins before the source; C2 attachment to the boot parent at25s',
            frozen=['camera','scene','beta','body scale','bone lengths','shared foot-boot registration'],
            active_head_neck=True,physical_contact_factor='coarse standing envelope prior from user motion observation; no physical blade station identified',
            opening_condition=ledger['opening_inference'],opening_velocity_scale_m_s=problem.velocity_scale.tolist())
        write_json(out/'fit/config.json',config)
        packet=export(ref,camera,info,out,source_path=root/'boot_integration_1102/fitted/result.npz')
        packet.update(opening_inferred_mask=ref.p['lower_body_inferred'],opening_donor_official_pts=np.array(ref.donor_pts),
            opening_static_condition_mask=(ref.p['pts']/12800<=STATIONARY_END),opening_condition_json=np.array(json.dumps(ledger['opening_inference'])),
            solver_config_json=np.array(json.dumps(dict(opening='fit/config.json',result='fit/result.json',ledger='ledger.json',
                preparation_camera='preparation_camera.json',preparation_pose='preparation_pose.json'))),
            active_blocks_json=np.array(json.dumps(dict(shared_binding=0,frozen_shared_binding_parameters=14,body_columns=problem.state.active.tolist()))),
            stopping_reason_json=np.array(json.dumps(report['stopping_reason'])),costs_json=np.array(json.dumps(report['final_parts'])))
        np.savez_compressed(out/'fitted/result.npz',**packet);write_json(out/'binding.json',ref.binding.record())
        print(json.dumps(dict(export=str(out/'fitted/result.npz'),frames=len(packet['pts']))),flush=True)
        from .opening_review import run as review
        review(root,out)
        viewer=ResultCatalog(project=project,data_root=root.parents[1],config=project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json')
        prepare_variant(viewer,viewer.variant('opening1102'),ref.binding.surface)
        print(json.dumps(deliver(project,root,out,ref,problem,x,info)),flush=True)


if __name__=='__main__':main()
