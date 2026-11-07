"""A 2A refinement on the saved full-programme controls. No per-frame states."""
from pathlib import Path
from types import SimpleNamespace
import json, time
import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.trajectory_backend import (
    load_reference, load, digest, SourceTurnPath,
    ContinuousMassGeometry, ConfirmedFlightFactors, EpisodeSupport, fk_state,
    BodyExecution, BodyExecutionConfig, source_seconds, write_json,
)
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, with_warm_rotation_chart
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState, shoe_curve_from_packet

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
PROJECT=Path(r"D:/OneDrive/codex_projs/RinkSpace")
PRE=BASE/'prepared_physics'
PARENT=BASE/'physics_guarded'
OLD=BASE.parent/'trajectory_continue_1107/temporal_joint_cached'
ACTION_ID=('axel_rotation','jump_2a','axel_left_approach','axel_exit_right','support_78_4')
START, END=77.30,78.76
MAX_WALL=125.0
MAX_STEPS=11
TORSO=np.array([5,6,11,12])
LEGS=np.array([13,14,15,16])

def setup():
    saved=load(PARENT/'solve/candidate.npz')
    ref,camera,info,doc,seed=load_reference(PROJECT,PRE)
    validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    for field,name in [('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz'),('source_evidence_sha256','source_evidence.json')]:
        assert str(saved[field])==digest(PRE/name), name
    ref.absolute_curve=with_warm_rotation_chart(curve_from_packet(saved,seed=True),saved,lift=True)
    ref.relative_skate_curve=shoe_curve_from_packet(saved,seed=True)
    return saved,ref,camera,info,doc,seed

def axis(s):
    trunk=s['native_world'][:,[16,17]].mean(1)-s['native_world'][:,[1,2]].mean(1)
    return trunk/np.linalg.norm(trunk,axis=1,keepdims=True)

def scores(uv, obs):
    norm=np.linalg.norm(uv-obs,axis=-1)
    diff=uv[:,[5,6]].mean(1)-uv[:,[11,12]].mean(1)
    src=obs[:,[5,6]].mean(1)-obs[:,[11,12]].mean(1)
    def stat(v): return dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
    return dict(torso_px=stat(norm[:,TORSO]),legs_px=stat(norm[:,LEGS]),ankles_px=stat(norm[:,[15,16]]),
                trunk_vector_px=stat(np.linalg.norm(diff-src,axis=1)),
                ankles_midpoint_px=stat(np.linalg.norm(uv[:,[15,16]].mean(1)-obs[:,[15,16]].mean(1),axis=1)))

def run():
    started=time.monotonic()
    with (OUT/'progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        def emit(event,**fields):
            record=dict(event=event,seconds=time.monotonic()-started,**fields)
            log.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        emit('begin',wall_seconds=MAX_WALL,max_accepted=MAX_STEPS,rotation_chart='checkpoint-continuous SU2', protected_landing_runner_pts=1003520)
        saved,ref,camera,info,doc,seed=setup()
        execution=BodyExecution(BodyExecutionConfig(workers=4,blas_threads=1))
        execution.attach(ref)
        with execution.limits():
            t=source_seconds(ref.p)
            b=FullBodyProblem(ref,camera,PRE/'observations.npz',(START-.04,END+.04),'full',state_factory=SequenceState)
            state=b.state
            anchor=saved['x'][17:].copy()
            knots=state.knots+t[0]
            # Preserve the 78.40 source-confirmed runner at all five exposure
            # nodes. These are still the same programme spline controls, and
            # the landing at ~78.18 and immediate recovery stay adjustable.
            source_runner_frame=np.flatnonzero(ref.p['pts']==1003520)
            first_runner_node=78.40-abs(camera.readout_times(source_runner_frame)[0])*.5
            controls=np.flatnonzero((knots[:-4]>=START)&(knots[4:]<first_runner_node))
            joints=[1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21]
            tracks=np.r_[np.arange(6),*[np.arange(3*j+3,3*j+6) for j in joints]]
            cols=(controls[:,None]*69+tracks).ravel()
            isroot=(cols%69)<6
            trans=(cols%69)<3
            knee=np.isin((cols%69-3)//3,[4,5])
            tangent_prior=np.where(trans,1.4,np.where(isroot,.5,np.where(knee,.7,.5)))
            limit=np.where(trans,.13,np.where(isroot,.85,np.where(knee,.65,.65)))
            reg=diags(tangent_prior,format='csr')
            mask=np.isin(np.flatnonzero(b.obs['use'].ravel())//17,b.frames)
            old_weights=load(PARENT/'objective_block_weights.npz')
            pixel_weight=np.sqrt(.02*old_weights['body_pixels'][np.repeat(mask,2)])
            assert len(pixel_weight)==2*len(b.of)
            # Image observations remain exactly those in the common programme
            # objective. Crossed/occluded ankles keep their low existing weight.
            episodes={e['episode_id']:e for e in doc['source_episode_review']['episodes']}
            prob=SimpleNamespace(body=b,reference=ref,camera=camera,binding=ref.binding,size=state.size+17)
            mass=ContinuousMassGeometry(ref)
            turn=SourceTurnPath(prob,episodes[ACTION_ID[0]],anchor)
            flight=ConfirmedFlightFactors(prob,mass,episode=episodes[ACTION_ID[1]],event_average=True)
            actions=[turn,flight]+[EpisodeSupport(prob,doc,episode=episodes[k]) for k in ACTION_ID[2:]]
            # Weak depth conditioning from the more stable older lean. Pixel
            # data controls the image-plane direction; never align Q or freeze
            # the athlete upright. This conditional prior is reported explicitly.
            axis_ids=np.flatnonzero((t>=77.80)&(t<=78.14))
            axis_time=ref.times[axis_ids]
            axis_base=ref.at(axis_time)
            old=load(OLD/'final/fitted/result.npz')
            np.testing.assert_array_equal(old['pts'],ref.p['pts'])
            old_world=old['native22_world_mid'][axis_ids]
            old_axis=axis(dict(native_world=old_world))
            ray=camera.camera_centres(axis_ids)-axis_base['P'];ray[:,2]=0
            ray/=np.linalg.norm(ray,axis=1,keepdims=True)
            # Degrees of out-of-view lean remain uncertain from one camera.
            old_lean=np.sum(old_axis*ray,axis=1)
            axis_w=1./(.11*np.sqrt(len(axis_time)))
            axis_columns=np.r_[np.arange(3,6),*[np.arange(3*j+3,3*j+6) for j in (3,6,9,13,14)]]
            measure=(t[b.frames]>=77.78)&(t[b.frames]<=78.16)
            obs_uv=b.obs['uv'][b.frames][measure]
            emit('shared_controls',programme_frames=len(ref.times),variables=len(cols),control_count=len(controls),
                 support_seconds=[float(knots[controls[0]]),float(knots[controls[-1]+4])],measurement_rows=int(measure.sum()),initial_ice_policy=str(saved['ice_soft_policy_json']))
            def evaluate(v,jac=False):
                x=anchor.copy();x[cols]+=v
                body=b.evaluate(x,jac)
                uv=body['uv'][:,b.layout.body_landmarks]
                raw=body['weighted_blocks'][0]
                residuals=[raw.residual*pixel_weight]
                mats=[diags(pixel_weight)@raw.jacobian[:,cols]] if jac else []
                costs={'body_pixels':float(residuals[0]@residuals[0]/2)}
                rows={}
                for eid,factor in zip(ACTION_ID,actions):
                    val=factor.evaluate(x,jac)
                    rows[eid]=val['metrics']
                    for block in val['weighted_blocks']:
                        weight=np.sqrt(old_weights.get('episode/'+eid+'/'+block.name,1.))
                        r=block.residual*weight
                        residuals.append(r);costs[eid+'/'+block.name]=float(r@r/2)
                        if jac:mats.append(block.jacobian[:,cols+17]*weight)
                d=state.increments(x,axis_time)
                u=axis(fk_state(ref,axis_base,d))
                a=(np.sum(u*ray,axis=1)-old_lean)*axis_w
                residuals.extend([a,v*tangent_prior])
                if jac:
                    loc=np.zeros((len(d),1,69))
                    for col in axis_columns:
                        pos=d.copy();neg=d.copy()
                        pos[:,col]+=1.e-6;neg[:,col]-=1.e-6
                        loc[:,0,col]=np.sum((axis(fk_state(ref,axis_base,pos))-axis(fk_state(ref,axis_base,neg)))*ray,axis=1)*(axis_w/2.e-6)
                    mats.extend([d.chain(loc)[:,cols],reg])
                summary=dict(groups=scores(uv[measure],obs_uv),parts=costs,reverse_intervals=rows[ACTION_ID[0]]['reverse_intervals'],
                             path_yaw_turns=rows[ACTION_ID[0]]['integrated_signed_yaw_proxy_turns'],
                             axis_peak_deg=float(np.degrees(np.arccos(np.clip(u[:,2],-1,1))).max()),
                             axis_peak_source_seconds=float(t[axis_ids[np.argmin(u[:,2])]]))
                return np.concatenate(residuals),vstack(mats,format='csr') if jac else None,summary
            v=np.zeros(len(cols));r,_,initial=evaluate(v)
            cost=float(r@r/2);emit('initial',cost=cost,summary=initial)
            accepted=0
            for k in range(MAX_STEPS):
                if time.monotonic()-started>MAX_WALL:break
                r,J,summary=evaluate(v,True)
                # Damped local updates within exact SO(3) spline support.
                step=lsmr(J,-r,damp=.70,atol=1.e-6,btol=1.e-6,maxiter=150)[0]
                step*=min(1.,.35/max(np.max(np.abs(step)),1.e-9))
                found=False
                for alpha in (1.,.5,.25,.125):
                    if time.monotonic()-started>MAX_WALL:break
                    trial=np.clip(v+alpha*step,-limit,limit)
                    try:rt,_,s=evaluate(trial)
                    except (ArithmeticError,ValueError) as error:
                        emit('trial_domain',iteration=k,alpha=alpha,error=str(error));continue
                    now=float(rt@rt/2)
                    emit('trial',iteration=k,alpha=alpha,cost=now,summary=s)
                    if now<cost-.015 and s['reverse_intervals']<=initial['reverse_intervals']:
                        found=True;accepted+=1;v=trial;cost=now
                        np.savez_compressed(OUT/'proposed_controls.npz',columns=cols,increment=v,
                            full_scope_seconds=t[[0,-1]],local_accepted_steps=accepted)
                        write_json(OUT/'local_comparison.json',dict(before=initial,after=s,accepted=accepted,trial_cost=cost))
                        emit('accepted',accepted=accepted,cost=cost)
                        break
                if not found:break
            emit('refinement_complete',accepted=accepted)
        execution.close()
if __name__=='__main__':
    run()
