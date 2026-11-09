"""Conditional native endpoint-camera block in the existing programme objective.

Static curves are profiled on their own scene geometry. Body nodes come from the
same complete saved spline and use the existing five-node implicit row model.
This is one block of alternating joint estimation, not a new world measurement.
"""
from pathlib import Path
import argparse,copy,json,os,re,time,sys
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import NativeSceneGPU
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference,GROUPS
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
def summary(r):
    e=np.linalg.norm(r,axis=-1)
    return dict(rows=len(e),cost=float((e*e).sum()/2),rms=float(np.sqrt(np.mean(e*e))),p95=float(np.percentile(e,95)),max=float(e.max()))
def groups_for(scene,path_ids):
    lookup={p['id']:p for p in scene['paths']}
    out=[]
    for pid in path_ids:
        if 'print_instance/' in pid:
            m=re.search(r'Ad(\d+)',pid);out.append('Ad'+str((int(m[1])-1)%13+1) if m else pid.split('/')[1])
        else:
            p=lookup.get(pid,{})
            entity=p.get('entity_id',p.get('label',pid))
            out.append(re.sub(r'[:/](outer|middle|inner|center|\d+)$','',entity))
    return np.array(out)

class StaticWindow:
    def __init__(self,native,manifest,state,points,path_ids,shot,directions,rows):
        self.native=native;self.m=manifest;self.state=state;self.points=points;self.rows=rows;self.dirs=directions
        self.shot=shot;self.timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
        self.camera=read(manifest['cameras'][shot['camera_index']]['camera_definition'])
        self.camera.update(first_control_time=self.timeline.first,control_dt=self.timeline.dt)
        self.paths=np.asarray(path_ids)[points['path_indices'][rows]]
        self.pts=points['pts'][rows];self.stations=state['stations'][rows].copy()
    def evaluate(self,controls,jacobian=False):
        n=len(self.rows);ids=np.arange(n,dtype=np.int32);q=self.dirs.shape[-1]
        with self.native.resident(camera=self.camera,controls=controls,rigs=self.state['rig'][self.shot['camera_index']],
                path_ids=self.paths,stations=self.stations,pts=self.pts,observed=self.points['uv'][self.rows],
                motion_times=self.timeline.exposure_time(self.shot,self.pts),
                scene_directions=np.zeros((len(self.state['scene_values']),q))) as rr:
            rr.configure_projection_time_support(half_seconds=self.m['max_readout']/2)
            curve=self.points['kind'][self.rows]==1;local=np.full(n,-1,np.int32);local[curve]=np.arange(curve.sum(),dtype=np.int32)
            rr.configure_pixel_linearization(directions=self.dirs,local_ids=local,local_scales=np.ones(curve.sum()))
            bounds=rr._path_supports[rr._codes].copy()
            rr.configure_solver(lower=bounds[curve,0],upper=bounds[curve,1],device_budget_bytes=512*1024**2,host_budget_bytes=512*1024**2,profile_trial_stations=True)
            profile=rr.refine_curve_stations(mode='local')
            v=rr.evaluate('cuda')
            if np.any(v['status']):raise ValueError('Native static camera projection invalid')
            residual=v['values'][:,9:11].copy()
            if not jacobian:return residual,None,profile
            rr.linearize('cuda');D=rr.probe_rows(ids)
            # Eliminate an ordinary curve tangent; zero tangent at a polyline
            # corner keeps BOTH pixel rows. Never turn it into a missing row.
            tangent=D[:,:,-1].copy();A=D[:,:,:q]
            station=rr.export_state()['stations'];gradient=np.einsum('ni,ni->n',tangent,residual)
            active=((station<=bounds[:,0]+1e-10)&(gradient>=0))|((station>=bounds[:,1]-1e-10)&(gradient<=0))|~curve
            # A native polyline/warp knot is a nonsmooth station candidate.
            # Its two one-sided tangents cannot be replaced by one free tangent.
            # Retain both UV rows when the profiled optimum sits at that knot.
            corner=np.zeros(n,bool)
            for pid in np.unique(self.paths):
                path=self.native.paths[pid];knots=np.asarray(path.get('knots',[]),float)
                if not len(knots) and path['kind'] in (2,5):knots=np.linspace(0,1,len(path.get('vertices',[])))
                if len(knots)>2:
                    rows=np.flatnonzero(self.paths==pid)
                    corner[rows]=np.min(abs(station[rows,None]-knots[None,1:-1]),axis=1)<=1e-9
            active|=corner
            raw_gradient=np.einsum('ni,niq->q',residual,A)
            tangent[active]=0.;norm=np.sum(tangent*tangent,axis=1)
            valid=norm>1e-20;den=np.where(valid,norm,1.)
            A=A-tangent[:,:,None]*(np.einsum('ni,niq->nq',tangent,A)/den[:,None])[:,None,:]
            projected=residual-tangent*(np.einsum('ni,ni->n',tangent,residual)/den)[:,None]
            return residual,(A,projected),dict(profile,zero_tangent_rows=int((~valid).sum()),active_finite_bound_rows=int(active.sum()),profiled_corner_rows=int(corner.sum()),raw_static_gradient=raw_gradient.tolist())

class BodyWindow:
    def __init__(self,reference,projector,obs,shot,seconds):
        self.camera=projector;self.shot=shot
        np.testing.assert_array_equal(obs['pts'],reference.p['pts'])
        sec=reference.p['pts']/12800.
        frames=np.flatnonzero((projector.shot_ids==shot['id'])&(sec>=seconds[0]-1e-9)&(sec<=seconds[1]+1e-9))
        f,j=np.nonzero(obs['use'][frames]);self.frames=frames[f]
        assignment=body_joint_assignment(obs)[self.frames,j]
        ts=(reference.times[frames,None]+projector.readout_times(frames)[:,None]*np.linspace(-.5,.5,5)).ravel()
        s=reference.world(reference.at(ts))
        world=s['world'].reshape(len(frames),5,reference.layout.point_count,3)
        self.nodes=world[f,:,reference.layout.body_landmarks[assignment]][:,:,None]
        self.target=obs['uv'][self.frames,j];self.sigma=obs['sigma_px'][self.frames,j]
        weight=np.full(len(j),.07);weight[np.isin(j,GROUPS['face'])]=.025;weight[np.isin(j,GROUPS['torso'])]=.35;weight[np.isin(j,GROUPS['lower_limbs'])]=.45;weight[np.isin(j,[15,16])]=.045
        self.weight=np.sqrt(weight*.02)
    def evaluate(self,controls):
        self.camera.p['controls']=controls
        for c in self.camera.cameras.values():c.p['controls']=controls
        uv=moving_projection(self.camera,self.nodes,self.frames)['uv'][:,0]
        error=(uv-self.target)/self.sigma[:,None]
        k=np.sqrt(2/(np.sqrt(1+np.sum(error**2,axis=1)/4)+1))
        return (error*k[:,None]*self.weight[:,None]).ravel()

def patch_direction(controls,timeline,shot,side,width=.4):
    ids=np.arange(shot['control_begin'],shot['control_stop'])
    times=timeline.first+ids*timeline.dt
    edge=timeline.exposure_time(shot,np.array([shot['first_pts'] if side=='start' else shot['last_pts']]))[0]
    inward=times-edge if side=='start' else edge-times
    u=np.clip(1-inward/width,0,1);weight=u*u*u*(10-15*u+6*u*u)
    dirs=np.zeros((*controls.shape,9))
    u=inward/width
    # Endpoint value, slope and curvature are distinct, each C2-joined to the
    # unchanged shot interior through the cardinal-control basis.
    for order in range(3):
        for j in range(3):dirs[ids,j,order*3+j]=weight*u**order
    return dirs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('project','prepared','parameter','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--camera-record',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record_path=a.camera_record or a.project/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.json'
    record=read(record_path);prepared=Path(record['prepared']);m=read(prepared/'joint_manifest.json');scene_path=Path(m['scene_path'])
    state=load(record['state']['path']);compact=load(record['state']['compact_path']);points=load(prepared/'point_observations.npz');path_ids=read(prepared/'mapping.json')['path_ids']
    np.testing.assert_array_equal(state['source_rows'],points['source_rows'])
    model=SceneParameterModel(read(scene_path),scene_path,optimize_scene=True);parameter_state=ParameterState(model.initial_state.layout,state['scene_values'])
    core=NativeCore(record['native_library']['path']);native=NativeSceneGPU(scene_path,core,parameter_state=parameter_state)
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    root,camera,info,doc,_=load_reference(a.project,a.prepared);packet=load(a.parameter)
    from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
    camera=ProgrammeCamera(a.project,root.p,root.p['shot_ids'],record_path=record_path)
    sys.path.insert(0,str(a.project/'events/ankara_2026/programmes/okada_mei_sp'))
    import prepare_native_multicamera as native_prepared
    from fit_native_s01_joint import mount_configuration
    metres=read(a.project/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    envelopes=[mount_configuration(metres,r,native_prepared.camera_working_envelope_path(m))[2] for r in state['rig']]
    def certificate(candidate_controls,ci):
        e=envelopes[ci];windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==ci]]
        limits=[*e['forward_bounds'],*e['pupil_bounds'],e['max_pan_tilt'],e['max_tilt_deviation'],*e['elevation_bounds'],*e['roll_bounds'],
            *e.get('max_speed',[np.inf]*3),*e.get('max_acceleration',[np.inf]*3),*e.get('max_jerk',[np.inf]*3),*((windows-timeline.first)/timeline.dt).ravel()]
        return core.evaluate_camera_envelope(state['rig'][ci],candidate_controls,timeline.first,timeline.dt,limits,joined_zoom_domain=e.get('joined_zoom_domain',False))
    reference=absolute_reference(root,curve_from_packet(packet),root=a.prepared);reference.relative_skate_curve=shoe_curve_from_packet(packet)
    reference.binding=copy.copy(root.binding);reference.binding.reference=reference;reference.binding.x=packet['x'][:17].copy();reference.binding._packet=None;reference.binding._parameter_key=None
    obs=load(a.prepared/'observations.npz');controls=compact['controls'].copy();original=controls.copy();groups=groups_for(read(scene_path),path_ids)
    reports=[];history=[];all_dirs=[]
    for shot_id,side,protected,validation in [('s05','end',{'entity/ice.F2'},{'entity/ice.B1'}),('s08','start',{'entity/ice.B1'},{'entity/ice.lane3','entity/ice.lane4'}),('s08','end',{'Ad5'},{'entity/ice.A3'})]:
        shot=next(s for s in m['shots'] if s['id']==shot_id)
        edge=(shot['first_pts'] if side=='start' else shot['last_pts'])/12800.
        nominal_seconds=[edge,edge+.4] if side=='start' else [edge-.4,edge]
        dirs=patch_direction(controls,timeline,shot,side)
        support=np.flatnonzero(np.any(dirs!=0,axis=(1,2)))
        shift=timeline.exposure_time(shot,np.array([shot['first_pts']]))[0]-shot['first_pts']/12800.
        half=abs(state['rig'][shot['camera_index'],37])/2
        seconds=[max(shot['first_pts']/12800.,timeline.first+(support[0]-2)*timeline.dt-shift-half),
                 min(shot['last_pts']/12800.,timeline.first+(support[-1]+2)*timeline.dt-shift+half)]
        np.testing.assert_array_equal(state['rig'][shot['camera_index']],camera.cameras[shot_id].p['rig'])
        if camera.cameras[shot_id].p['first_control_time']!=timeline.first or camera.cameras[shot_id].p['control_dt']!=timeline.dt:raise ValueError('Body and background camera timelines differ')
        rows=np.flatnonzero((points['pts']/12800.>=seconds[0]-1e-9)&(points['pts']/12800.<=seconds[1]+1e-9))
        labels=groups[points['path_indices'][rows]];heldout=protected;hold=np.isin(labels,list(protected));val=np.isin(labels,list(validation))
        fit=~hold&~val
        if not val.any():raise ValueError('Predeclared final heldout entities absent')
        names,counts=np.unique(labels,return_counts=True)
        if not hold.any():raise ValueError('Predeclared protected entity absent')
        dirs=patch_direction(controls,timeline,shot,side)
        static=StaticWindow(native,m,state,points,path_ids,shot,dirs,rows)
        body=BodyWindow(reference,camera,obs,shot,seconds)
        scale=np.tile([.002,.002,.01],3);q=len(scale);z=np.zeros(q)
        base_controls=controls.copy()
        def evaluate(z,jacobian=False):
            trial=base_controls+np.einsum('ncq,q->nc',dirs,z*scale)
            cert=certificate(trial,shot['camera_index'])
            if cert['margin'] < -1e-10:raise ValueError('Candidate leaves original native physical camera envelope: '+str(cert))
            r,lin,prof=static.evaluate(trial,jacobian);prof['camera_certificate']=cert;br=body.evaluate(trial)
            residual=np.r_[r[fit].ravel(),br,z*.1]
            if not jacobian:return residual,r,br,prof
            A,rp=lin;h=1e-5;B=[]
            for k in range(q):
                step=dirs[:,:,k]*(scale[k]*h)
                B.append((body.evaluate(trial+step)-body.evaluate(trial-step))/(2*h))
            J=np.vstack([(A[fit]*scale).reshape(-1,q),np.stack(B,axis=1),np.eye(q)*.1])
            solved_r=np.r_[rp[fit].ravel(),br,z*.1]
            return residual,r,br,prof,J,solved_r,A*scale
        v=evaluate(z,True);initial=v;singular=np.linalg.svd(v[4][:fit.sum()*2],compute_uv=False)
        # Verify the real profiled native gradient on this frozen observation set.
        check_direction=np.arange(1,q+1,dtype=float);check_direction[1::2]*=-1;check_direction/=np.linalg.norm(check_direction)
        analytic=float(v[5]@(v[4]@check_direction));direction_checks=[]
        for h in (1e-3,5e-4):
            plus=evaluate(h*check_direction)[0];minus=evaluate(-h*check_direction)[0]
            numeric=float((plus@plus-minus@minus)/(4*h))
            direction_checks.append(dict(step=h,analytic=analytic,central_difference=numeric,
                relative_error=abs(analytic-numeric)/max(1.,abs(analytic),abs(numeric))))
        gradient_valid=all(row['relative_error']<1e-3 for row in direction_checks)
        print(json.dumps(dict(event='camera_native_direction_check',shot=shot_id,side=side,checks=direction_checks,valid=gradient_valid)),flush=True)
        rank=int(np.sum(singular>singular[0]*1e-8));stop='static_rank_deficient_frozen' if gradient_valid else 'native_profile_gradient_unresolved_frozen'
        print(json.dumps(dict(event='camera_window',shot=shot_id,side=side,seconds=seconds,rows=len(rows),groups=dict(zip(names.tolist(),counts.tolist())),heldout=sorted(heldout),singular_values=singular.tolist(),rank=rank)),flush=True)
        if rank==q and gradient_valid:
            stop='line_search_no_joint_and_background_descent'
            while True:
                H=v[4].T@v[4]+np.eye(q)*1e-9;g=v[4].T@v[5]
                C=[];limits=[]
                for label in heldout:
                    take=labels==label
                    if not take.any():continue
                    C.append(np.einsum('ni,niq->q',v[1][take],v[6][take]))
                    limits.append(.5*(np.sum(initial[1][take]**2)-np.sum(v[1][take]**2)))
                C=np.array(C);limits=np.array(limits)
                answer=minimize(lambda d:.5*d@H@d+g@d,np.zeros(q),jac=lambda d:H@d+g,
                    bounds=list(zip(-1-z,1-z)),constraints=[dict(type='ineq',fun=lambda d:limits-C@d,jac=lambda d:-C)],
                    method='SLSQP',options=dict(ftol=1e-10,maxiter=100))
                dz=answer.x
                if np.linalg.norm(dz,np.inf)<1e-6:stop='small_conditional_step';break
                alpha=1.;accepted=False
                while alpha>1e-6:
                    trial_z=z+alpha*dz
                    if np.max(abs(trial_z))>1:alpha*=.5;continue
                    try:t=evaluate(trial_z)
                    except ValueError:alpha*=.5;continue
                    oldcost=float(v[0]@v[0]);newcost=float(t[0]@t[0])
                    # All held-out entities are a strict incremental gate.
                    group_ok=all(np.sum(t[1][labels==g]**2)<=np.sum(initial[1][labels==g]**2)+1e-8 for g in heldout)
                    if newcost<oldcost and group_ok:
                        history.append(dict(shot=shot_id,side=side,z=trial_z.tolist(),fit_cost=newcost/2,heldout=summary(t[1][hold]),body_cost=float(t[2]@t[2]/2)))
                        z=trial_z;v=evaluate(z,True);accepted=True
                        print(json.dumps(dict(event='camera_accepted',**history[-1])),flush=True)
                        if (oldcost-newcost)/max(1.,oldcost)<1e-7:stop='conditional_objective_stagnation';accepted=False
                        break
                    alpha*=.5
                if not accepted:break
        validation_ok=all(np.sum(v[1][labels==g]**2)<=np.sum(initial[1][labels==g]**2)+1e-8 for g in validation)
        attempted_z=z.copy();attempted_report={g:summary(v[1][labels==g]) for g in validation}
        if not validation_ok:z[:]=0;v=initial;stop='final_entity_holdout_rejected_original_camera_retained'
        controls=base_controls+np.einsum('ncq,q->nc',dirs,z*scale);all_dirs.append(dirs)
        report=dict(shot=shot_id,side=side,seconds=seconds,nominal_seconds=nominal_seconds,validation_entities=sorted(validation),validation_pass=validation_ok,attempted_z=attempted_z.tolist(),attempted_validation=attempted_report,delta_pan_tilt_logf=(z*scale).tolist(),static_rank=rank,static_singular_values=singular.tolist(),native_direction_checks=direction_checks,gradient_valid=gradient_valid,conditional_information='With fixed world gauge and scene, static A.T A is a PSD lower bound on the full body-eliminated camera Schur information; body/depth cannot absorb these rows.',protected_entities=sorted(protected),before=dict(fit=summary(initial[1][fit]),heldout=summary(initial[1][hold]),body_cost=float(initial[2]@initial[2]/2)),after=dict(fit=summary(v[1][fit]),heldout=summary(v[1][hold]),body_cost=float(v[2]@v[2]/2)),stop=stop,rows=len(rows),groups={str(g):dict(rows=int((labels==g).sum()),before=summary(initial[1][labels==g]),after=summary(v[1][labels==g])) for g in names},station_profile=v[3])
        reports.append(report)
        np.savez_compressed(a.output/(shot_id+'_'+side+'_static.npz'),source_rows=points['source_rows'][rows],pts=points['pts'][rows],groups=labels,protected=hold,heldout=val,before=initial[1],after=v[1],delta=z*scale)
    compact['controls']=controls;compact['endpoint_joint_candidate_json']=np.array(json.dumps(dict(body_parameter=str(a.parameter),body_sha256=digest(a.parameter),windows=reports)))
    np.savez_compressed(a.output/'camera.npz',**compact)
    # Separate selection receipt; no old default metadata is rewritten.
    candidate=dict(record);candidate['state']=dict(record['state'],compact_path=str(a.output/'camera.npz'),compact_sha256=digest(a.output/'camera.npz'))
    candidate.update(status='conditional_camera_body_endpoint_candidate',parent_camera_record=str(record_path),accuracy_accepted=False)
    write(a.output/'camera.json',candidate)
    write(a.output/'report.json',dict(schema='rinkspace.endpoint-camera-block.v1',script_sha256=digest(Path(__file__)),native_library_sha256=digest(core.library_path),static_scene_sha256=digest(scene_path),static_observations_sha256=digest(prepared/'point_observations.npz'),body_parameter=str(a.parameter),body_sha256=digest(a.parameter),body_exposures=len(reference.times),body_networks_run=False,windows=reports,history=history,control_change_max=float(np.max(abs(controls-original))),camera_model='unchanged native Brown; rig,pupil chain,distortion,readout fixed; same implicit five-body-node projection',patch='three compact C2 cardinal-control directions per pan,tilt,logf (value, slope, curvature) in each observed 0.4s end; no cross-cut factors',scope='conditional block of alternating joint estimation; full body must be solved again using this exact camera',heldout_scope='protected groups constrain steps; distinct final heldout entities scored once after coefficients fixed; all rows were used in original calibration',bounds='engineering local trust box: pan/tilt 0.002 rad and logf 0.01; not measurement accuracy',accuracy_accepted=False))
    np.savez_compressed(a.output/'patch_directions.npz',directions=np.concatenate(all_dirs,axis=-1),original=original,final=controls)
    print(json.dumps(dict(event='camera_complete',output=str(a.output))),flush=True)

if __name__=='__main__':
    os.environ.setdefault('CUDA_PATH','D:/Program Files/CUDA')
    with threadpool_limits(limits=1):main()
