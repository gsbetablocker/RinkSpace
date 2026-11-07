"""Replay both states with the complete frozen programme objective before delivery."""
import copy, json, shutil, time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from refit_axel_axis import (
    OUT, BASE, PROJECT, PRE, PARENT, OLD, setup, axis, scores,
    ContinuousMassGeometry, BodyExecution, BodyExecutionConfig, source_seconds,
    write_json, fk_state, load, digest,
)
from calibration.program_tracking.body_tracking.trajectory_backend import (
    SequenceJoint, SequenceActions, PrimaryActions, primary_policy, record_action_factors,
    body_groups, export_candidate,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy
from calibration.program_tracking.body_tracking.audit_probe_contract import restore_saved_objective
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment

STAGE=BASE/'axel_axis_joint_1107'
VARIANT='programme_action_axel_1107'
FRACS=(1.,)
MAX_WALL=310.

def restore_source(problem,saved):
    factors=problem.action_factors
    for name,factor in factors.extra_unassigned_episodes:
        prefix='source_association_extra_'+name+'_'
        if prefix+'choice' in saved:
            factor.episode_choice=int(saved[prefix+'choice'])
            factor.episode_costs=saved[prefix+'candidate_costs'].tolist()
            for key in ('fixed_choice','visible_mask'):
                if prefix+key in saved:
                    setattr(factor.episode_factors[factor.episode_choice],key,saved[prefix+key].copy())
    factors.evaluate(np.zeros(problem.body.size),False)
    if 'source_association_episode_choice' in saved:
        assert factors.unassigned.episode_choice==int(saved['source_association_episode_choice'])
    for label,factor in (('contour',factors.contour),('runner_extent',factors.extent)):
        for key in ('fixed_choice','visible_mask'):
            field='source_association_'+label+'_'+key
            if field in saved:
                np.testing.assert_array_equal(getattr(factor,key),saved[field])

def read_metrics(problem,v,t,phase):
    source=v['action_metrics']
    action={k:source['episode/'+k+'/metrics'] for k in
            ('axel_rotation','jump_2a','axel_left_approach','axel_exit_right','support_78_4')}
    err=problem.body.cache['uv'][:,problem.body.layout.body_landmarks]
    obs=problem.body.obs
    order=body_joint_assignment(obs)
    err=err[np.arange(len(err))[:,None],order]
    src=scores(err[phase],obs['uv'][phase])
    steel={r['id']:float(np.percentile(r['outside_visible_band_px'],95)) for r in
           source.get('visible_runner_contours',[])+source.get('visible_unassigned_skate_contours',[])}
    return dict(cost=float(v['residual']@v['residual']/2),weighted_parts=v['weighted_parts'],
                gate=v['gate'],image=src,action=action,steel=steel,whole_image=body_groups(problem,v))

def evaluate_axes(result,baseline,phase,t):
    records={}
    for key,z in [('physics',baseline),('axis',result)]:
        u=axis(dict(native_world=z['native22_world_mid']))
        tilt=np.degrees(np.arccos(np.clip(u[:,2],-1.,1.)))
        c=z['com_world_mid'];jump=(t>=77.88)&(t<=78.14)
        s=t[jump]-t[jump][0];b=np.c_[np.ones(len(s)),s]
        proxy=c[jump]+.5*9.81*s[:,None]**2*np.array([0.,0.,1.])
        residual=proxy-b@np.linalg.lstsq(b,proxy,rcond=None)[0]
        records[key]=dict(tilt_78_02_deg=float(tilt[np.argmin(abs(t-78.02))]),
                          tilt_compact_peak_deg=float(tilt[(t>=77.80)&(t<=78.14)].max()),
                          COM_to_gravity_arc_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(residual**2,axis=-1)))))
    outside=(t<77.29)|(t>78.77)
    unchanged={}
    for key in ('P_world_mid','Q_world_mid','native22_world_mid','skate_R_world_mid','COCO_uv'):
        unchanged[key]=float(np.max(np.abs(result[key][outside]-baseline[key][outside])))
        np.testing.assert_allclose(result[key][outside],baseline[key][outside],atol=2e-8,rtol=0)
    return dict(axes=records,outside_2a_max_abs_change=unchanged,
                max_P_shift_m=float(np.linalg.norm(result['P_world_mid']-baseline['P_world_mid'],axis=-1).max()),
                pts_verified=len(t))

def run():
    started=time.monotonic()
    STAGE.mkdir(exist_ok=True)
    (STAGE/'solve').mkdir(exist_ok=True)
    if (STAGE/'solve/candidate.npz').exists():
        raise FileExistsError(STAGE)
    with (STAGE/'progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        def emit(event,**data):
            row=dict(event=event,seconds=time.monotonic()-started,**data)
            log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        emit('begin_full_programme_replay',wall_seconds=MAX_WALL,max_trials=len(FRACS))
        saved,ref,camera,info,doc,seed=setup()
        old_seed=curve_from_packet(saved,seed=True)
        np.testing.assert_allclose(old_seed._check_log_chart(),ref.absolute_curve._check_log_chart(),atol=1e-12,rtol=0)
        x=saved['x'].copy()
        proposal=load(OUT/'proposed_controls.npz')
        new=x.copy()
        new[17+proposal['columns']]+=proposal['increment']
        execution=BodyExecution(BodyExecutionConfig(workers=8,blas_threads=1,
            gpu_batch_size=512,compatibility_backend='cuda_resident'))
        execution.attach(ref)
        with execution.limits():
            t=source_seconds(ref.p)
            phase=(t>=77.78)&(t<=78.16)
            policy=IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))
            p=SequenceJoint(ref,camera,PRE/'observations.npz',(t[0],t[-1]),doc.get('supports',[]),
                STAGE/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=policy)
            p.al.multiplier[:]=saved['multiplier']
            p.al.penalty=float(saved['penalty'])
            mass=ContinuousMassGeometry(ref)
            base=PrimaryActions(p,ref,mass,doc,warm=x[17:])
            p.action_factors=SequenceActions(p,ref,mass,doc,x[17:],base_factors=base)
            primary_policy(p)
            restored=restore_saved_objective(p,PARENT,saved)
            restore_source(p,saved)
            prior_replay=json.loads((OUT/'lifted_unprotected_full_replay.json').read_text('utf8'))
            assert prior_replay['objective']==restored
            before=prior_replay['baseline']
            archived=json.loads((PARENT/'final_metrics.json').read_text('utf8'))['weighted_parts']
            for key,w in before['weighted_parts'].items():
                np.testing.assert_allclose(w,archived[key],atol=3e-7,rtol=3e-7,err_msg=key)
            emit('baseline_replayed',cost=before['cost'],collar_unknown=before['gate']['collar_unknown_sections'])
            frozen=json.loads((PARENT/'primary_quality_baseline.json').read_text('utf8'))['source']
            trials=[]
            retained=None
            for fraction in FRACS:
                if time.monotonic()-started>MAX_WALL:break
                trial=x+fraction*(new-x)
                val=p.evaluate(trial,False)
                current=read_metrics(p,val,t,phase)
                steel_fail=[]
                for key,old in before['steel'].items():
                    other,radius=frozen[key]
                    if current['steel'][key]>max(old,other+.25*radius)+1e-7:
                        steel_fail.append(key)
                checks=dict(
                    facing=current['action']['axel_rotation']['reverse_intervals']==0 and
                           current['weighted_parts']['episode/axel_rotation/source_facing_cone']<=
                           before['weighted_parts']['episode/axel_rotation/source_facing_cone'],
                    centerline=current['image']['trunk_vector_px']['p95']<before['image']['trunk_vector_px']['p95'],
                    both_leg_chains=current['image']['legs_px']['p95']<before['image']['legs_px']['p95'],
                    flight=all(current['weighted_parts']['episode/jump_2a/'+key]<=before['weighted_parts']['episode/jump_2a/'+key]
                               for key in ('flight_integrated_com','flight_specific_momentum')),
                    no_new_unknown_collar=current['gate']['collar_unknown_sections']<=before['gate']['collar_unknown_sections'],
                    max_structure_violation_no_regression=current['gate']['max_violation_m']<=before['gate']['max_violation_m']+1e-6,
                    visible_steel_guard=not steel_fail,
                    full_cost=current['cost']<before['cost'])
                trials.append(dict(fraction=fraction,checks=checks,failed_steel=steel_fail,after=current))
                write_json(STAGE/'replay_comparison.json',dict(baseline=before,trials=trials,objective=restored))
                emit('global_trial',fraction=fraction,checks=checks,cost=current['cost'],collar_unknown=current['gate']['collar_unknown_sections'])
                if all(checks.values()):
                    retained=trial
                    break
            if retained is None:
                emit('held_for_review')
                return
            emit('exporting_same_programme',native_exposures=len(t),local_accepted_steps=int(proposal['local_accepted_steps']))
            for name in ('objective_policy.json','objective_block_weights.npz','primary_quality_baseline.json','source_associations.npz'):
                shutil.copy2(PARENT/name,STAGE/name)
            state=p.body.state
            candidate=absolute_reference(ref,state.current_curve(retained[17:]),root=PRE)
            candidate.relative_skate_curve=state.shoe_current_curve(retained[17:])
            candidate.binding=copy.copy(ref.binding)
            candidate.binding.reference=candidate
            archived_state=dict(saved)
            archived_state.update(state.packet(retained[17:]))
            parent_steps=json.loads((PARENT/'run_summary.json').read_text('utf8'))['retained_step']
            archived_state.update(x=retained,parent_accepted_steps=np.array(parent_steps),
                                  axis_refinement_parent_sha256=np.array(digest(PARENT/'solve/candidate.npz')))
            np.savez_compressed(STAGE/'solve/candidate.npz',**archived_state)
            write_json(STAGE/'solve/candidate_selection.json',dict(retained_step=int(proposal['local_accepted_steps']),
                available=True,quality_failed=False,whole_objective_replayed=True))
            z=export_candidate(PRE,STAGE,candidate,camera,info,doc.get('supports',[]))
            native=candidate.at(candidate.times)
            C,_=mass.evaluate(native,np.zeros((len(t),69)))
            body=fk_state(candidate,native,np.zeros((len(t),69)))
            z.update(com_world_mid=mass.com(C),native_body_local_R_mid=body['local_R'],
                native_body_FK_world_R_mid=body['Q'][:,None]@body['frames'],
                source_action_ledger_sha256=np.array(digest(PRE/'source_evidence.json')),
                P_velocity_world_mps=candidate.root_derivative(candidate.times,1)*candidate.binding.ratio,
                source_shot_ids=ref.p['shot_ids'],trajectory_evidence_status=np.where(p.body.obs['use'].sum(1)>=4,
                    'source_constrained_estimate','temporal_inference'),
                position_uncertainty_status=np.full(len(t),'single camera; axial lean/depth conditional'),
                ice_plane_known=np.array(ref.ice_plane_known),sequence_contact_modes=p.action_factors.modes.modes,
                sequence_contact_status=p.action_factors.modes.status,shared_betas=ref.p['shared_betas'],
                shared_body_scale=ref.p['body_scale'],
                source_body_model_joint_ids=body_joint_assignment(p.body.obs))
            if 'body_joint_assignment_provenance_json' in p.body.obs:
                z['body_joint_assignment_provenance_json']=p.body.obs['body_joint_assignment_provenance_json']
            with np.load(PARENT/'final/fitted/result.npz',allow_pickle=False) as prev:
                for k in ('pts','time_base','metres_per_scene_unit','shared_body_scale'):
                    np.testing.assert_array_equal(z[k],prev[k])
                equivalence=evaluate_axes(z,prev,phase,t)
            np.savez_compressed(STAGE/'fitted/result.npz',**z)
            record_action_factors(p,val,STAGE/'final_action_factors.json',jacobian=False)
            write_json(STAGE/'final_metrics.json',dict(groups=current['whole_image'],weighted_parts=current['weighted_parts'],
                structure=current['gate'],local=equivalence,global_checks=checks))
            write_json(STAGE/'run_summary.json',dict(parameter=str(STAGE/'solve/candidate.npz'),result=str(STAGE/'fitted/result.npz'),
                whole_seconds=t[[0,-1]].tolist(),source_frames=len(t),local_accepted_steps=int(proposal['local_accepted_steps']),
                whole_objective_selected_fraction=fraction,accuracy_accepted=False,source_visual_review='pending',
                engineering_ready=True,trajectory_candidate=True,remaining_unknown='Single view axis depth and exact steel contact; earlier non-2A failures retained'))
            emit('candidate_saved',result=str(STAGE/'fitted/result.npz'),**equivalence)
        execution.close()

if __name__=='__main__':
    run()
