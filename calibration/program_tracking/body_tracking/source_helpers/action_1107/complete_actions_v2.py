"""Action block proposals, accepted only by the actual whole programme joint objective."""
from pathlib import Path
import copy, json, os, shutil, sys, time
from collections import defaultdict
import numpy as np
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr

from calibration.program_tracking.body_tracking.trajectory_backend import (
    load_reference, SequenceJoint, SequenceActions, PrimaryActions, primary_policy,
    body_groups, record_action_factors, export_candidate, fk_state,
    source_seconds, ContinuousMassGeometry, BodyExecution, BodyExecutionConfig,
    load, digest, write_json,
)
from calibration.program_tracking.body_tracking.episode_factor_adapter import EpisodeScopedFactor
from calibration.program_tracking.body_tracking.programme_anatomy import ProgrammeAnatomy
from calibration.program_tracking.body_tracking.programme_action_factors import (
    turn_episode_image_metrics, turn_episode_image_checks,
)
from calibration.program_tracking.body_tracking.source_skate_structure_factors import FullSkateStructureFactors
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, absolute_reference
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState, shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_integration import ContactBodyProblem, NORMALIZATION
from calibration.program_tracking.body_tracking.contact_feasibility import ContactAL
from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy
from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem, body_joint_assignment
from calibration.program_tracking.body_tracking.memory_guard import ProcessMemoryGuard, check_memory_budget
from calibration.program_tracking.body_tracking.weighted_blocks import block, signature, verify

PROJECT=Path(r"D:/OneDrive/codex_projs/RinkSpace")
ROOT=Path(r"D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/trajectory_action_1107/action_completion_1107")
BASE=ROOT.parent
PRE=ROOT/'prepared_travel'
PARENT=ROOT/'solve_v1'
OUT=ROOT/'solve_v2'
RANGES=((47.56,50.64),(97.40,98.36),(104.08,105.12),(116.90,118.22),
        (143.76,151.80),(155.38,166.38),(171.04,185.30))
MAX_SECONDS=315.
MAX_ACCEPTED=3
VARIANT='programme_action_finish_1107'

def in_ranges(times,ranges=RANGES):
    return np.logical_or.reduce([(times>=lo)&(times<=hi) for lo,hi in ranges])

def load_associations(p,saved,x):
    factors=p.action_factors
    archived=load(PARENT/'source_associations.npz')
    for name,f in factors.extra_unassigned_episodes:
        prefix=f'source_association_extra_{name}_'
        f.episode_choice=int(saved[prefix+'choice'])
        f.episode_costs=saved[prefix+'candidate_costs'].tolist()
        selected=f.episode_factors[f.episode_choice]
        for attr in ('fixed_choice','visible_mask'):
            if prefix+attr in saved:setattr(selected,attr,saved[prefix+attr].copy())
    old=archived['source_association_contour_fixed_choice']
    contour=factors.contour
    # Match additional contours on the common 2A warm state. Keep the actual
    # archived assignments for each inherited source polyline, unchanged.
    contour.evaluate(x[17:],False)
    assert len(contour.fixed_choice)==len(old)
    contour.fixed_choice[:len(old)]=old
    factors.evaluate(np.zeros(p.body.size),False)
    if factors.unassigned is not None:
        assert factors.unassigned.episode_choice==int(saved['source_association_episode_choice'])
    assert np.array_equal(factors.extent.visible_mask,saved['source_association_runner_extent_visible_mask'])
    archived['source_association_contour_fixed_choice']=contour.fixed_choice.copy()
    if 'sequence_contact_modes' in saved:
        ref=p.reference
        added=np.zeros(len(ref.times),bool)
        for e in ref.source_document['source_episode_review']['episodes']:
            if e['episode_id'] in ('steps_lunge_recovery_left','steps_transfer_right','steps_transfer_left'):
                added|=(ref.p['pts']>=e['start_pts'])&(ref.p['pts']<=e['end_pts_inclusive'])
        m=factors.modes
        m.modes[~added]=saved['sequence_contact_modes'][~added]
        m.status[~added]=saved['sequence_contact_status'][~added]
        m.association_signature=signature(m.modes,m.status)
        m.record['supported_samples']=int(np.count_nonzero(m.modes<5))
    return archived

def blocks_from(value,name=None):
    if 'weighted_blocks' in value:return value['weighted_blocks']
    return [block(name,value['residual'],value['jac'])]

def active_controls(state,origin):
    k=state.knots+origin
    pick=np.flatnonzero(np.logical_or.reduce([(k[:-4]>=lo)&(k[4:]<=hi) for lo,hi in RANGES]))
    # All primary world and lower chain controls; small arm/spine corrections
    # help true cross-limb collisions and source silhouette. Hand state stays frozen.
    joints=(1,2,3,4,5,6,7,8,9,12,13,14,16,17,18,19,20,21)
    chain=np.r_[np.arange(6),*[np.arange(3*j+3,3*j+6) for j in joints]]
    body=(pick[:,None]*69+chain).ravel()
    shoe=state.shoe_curve.knots+origin
    s=np.flatnonzero(np.logical_or.reduce([(shoe[:-4]>=lo)&(shoe[4:]<=hi) for lo,hi in RANGES]))
    return 17+np.r_[body,state.body_size+(s[:,None]*6+np.arange(6)).ravel()]

def local_problem(problem,base,t,x,archived,first):
    ref=problem.reference;weights=problem.objective_block_weights
    frames=np.flatnonzero(in_ranges(t))
    compact_execution=BodyExecution(BodyExecutionConfig(workers=8,blas_threads=1,
        gpu_batch_size=512,compatibility_backend='cuda_resident'))
    body=ContactBodyProblem(ref,problem.camera,PRE/'observations.npz',(t[0],t[-1]),
        state_kind='sequence_skates_v1',source_frames=frames,execution=compact_execution)
    body.split_jacobian_only=True
    body.compatibility_difference_step=1e-8
    per_frame=problem.cavity_rows_per_frame
    al_idx=(per_frame*(frames[:,None]+1)+np.arange(per_frame)[None]).ravel()
    al=ContactAL(problem.al.multiplier[al_idx],problem.al.penalty)
    al_time=np.repeat(np.where((frames==0)|(frames==len(t)-1),.5,1.)*problem.image_time_weight,per_frame)
    max_margin=first['gate']['max_violation_m']
    pick=np.isin(np.flatnonzero(body.obs['use'].ravel())//17,frames)
    pixel=np.sqrt(problem.image_time_weight*weights['body_pixels'][np.repeat(pick,2)])
    contour=problem.action_factors.contour
    active=np.isin([row['pts'] for row in contour.rows],ref.p['pts'][frames])
    structures=FullSkateStructureFactors(problem,dict(ref.source_document['full_source_skate_structures'],
        rows=[row for row,keep in zip(contour.rows,active) if keep]))
    structures.fixed_choice=contour.fixed_choice[active[contour.point_rows]].copy()
    mode=copy.copy(problem.action_factors.modes)
    for attr in ('times','pts','weights','modes','status'):
        setattr(mode,attr,getattr(mode,attr)[frames])
    mode.base=ref.at(mode.times)
    anatomy=copy.copy(next(f for f in base.factors if isinstance(f,ProgrammeAnatomy)))
    sub=in_ranges(anatomy.times+t[0])
    for attr in ('times','weights','nearest_source'):
        setattr(anatomy,attr,getattr(anatomy,attr)[sub])
    anatomy.base=ref.at(anatomy.times)
    source_ids={e['episode_id'] for e in ref.source_document['source_episode_review']['episodes']
                if np.any(in_ranges(np.array([e['start_pts'],e['end_pts_inclusive']])/12800))}
    episode_factors=[f for f in base.factors if isinstance(f,EpisodeScopedFactor)
                     and f.episode_id.removesuffix('_source_band') in source_ids]
    # Small quadratic scaling controls step conditioning only. Acceptance
    # always replays the full current objective without this scaling.
    cols=active_controls(problem.body.state,t[0])
    slot=(cols-17)%69
    body_cols=cols<17+problem.body.state.body_size
    translation=body_cols&(slot<3)
    root=body_cols&(slot>=3)&(slot<6)
    bounds=np.where(translation,.10,np.where(root,.24,.29))
    bounds[~body_cols]=.16
    branch_factors=[(structures,'full_source_skate_structure'),(mode,'temporal_contact_mode')]
    def evaluate(y,jac):
        forward=body.evaluate(y[17:],jac)
        by=forward['weighted_blocks'][0]
        foot=next(b for b in forward['weighted_blocks'] if b.name=='dynamic_foot_compatibility')
        wc=np.sqrt(problem.image_time_weight*weights['dynamic_foot_compatibility'])
        rr=[by.residual*pixel,foot.residual*wc]
        jj=[diags(pixel)@by.jacobian[:,cols-17],foot.jacobian[:,cols-17]*wc] if jac else []
        margin,which=problem.cavity_margins(foot.residual*np.sqrt(2))
        D=None
        if jac:
            idx=np.c_[12+which[:,0],20+which[:,1]]
            if per_frame==4:
                idx=np.column_stack((idx,np.broadcast_to([32,33],idx.shape)))
            idx=(np.arange(body.n)[:,None]*problem.compatibility_width+idx).ravel()
            D=foot.jacobian[idx][:,cols-17]*(-.003*np.sqrt(2))
        finite,Jfinite=al.residual_and_jacobian(margin/NORMALIZATION,
            None if D is None else D/NORMALIZATION,row_weights=al_time)
        rr.append(finite)
        if jac:jj.append(Jfinite)
        unknown=forward.get('resident_collar_unknown')
        local_gate=dict(margin=margin,max_m=float(margin.max()),unknown=0 if unknown is None else int(np.sum(unknown)),J=D)
        vals=[b for factor in episode_factors for b in factor.evaluate(y[17:],jac)['weighted_blocks']]
        for factor,name in branch_factors:
            vals+=blocks_from(factor.evaluate(y[17:],jac),name)
        vals.append(anatomy.collision(y[17:],jac)[0])
        costs={'body_pixels':float(rr[0]@rr[0]/2),
               'dynamic_foot_compatibility':float(rr[1]@rr[1]/2),
               'cavity_AL':float(finite@finite/2)}
        for b in vals:
            w=np.sqrt(weights.get(b.name,1.))
            q=b.residual*w
            rr.append(q)
            costs[b.name]=float(q@q/2)
            if jac:jj.append(b.jacobian[:,cols]*w)
        return np.concatenate(rr), vstack(jj,format='csr') if jac else None, costs, local_gate
    return cols,bounds,frames,evaluate,compact_execution

def protect_structure_direction(direction,limits,margin,derivative):
    # Project the proposed increment onto linearized active upper bounds.
    # This changes only the step; the actual nonlinear margin is checked next.
    room=np.maximum(limits-margin,0.)
    target=np.maximum(room-2e-5,0.)
    J=derivative.tocsr()
    changed=0
    for sweep in range(14):
        violation=J@direction-target
        bad=np.flatnonzero(violation>1e-7)
        if not len(bad):break
        for row in bad[np.argsort(violation[bad])[::-1]]:
            a,b=J.indptr[row:row+2]
            cols,val=J.indices[a:b],J.data[a:b]
            energy=val@val
            if energy<1e-18:continue
            overshoot=val@direction[cols]-target[row]
            if overshoot>0:
                direction[cols]-=(overshoot/energy)*val
                changed+=1
    return direction,changed

def metrics(problem,base,val):
    b=val['action_metrics']
    source={r['id']:[float(np.percentile(r['outside_visible_band_px'],95)),
             float(r['pixel_uncertainty_radius_px'])] for r in
        b.get('visible_runner_contours',[])+b.get('visible_unassigned_skate_contours',[])}
    bands={}
    families=defaultdict(list)
    for key,value in b.items():
        if not key.startswith('episode/') or not key.endswith('_source_band/metrics'):continue
        name=key.split('/')[1].removesuffix('_source_band')
        for row in value['native_source_support_contacts']:
            stamp=str(row['pts'])
            bands[stamp]=dict(seconds=row['pts']/12800,foot=row['foot'],episode=name,
                outside_px=row['outside_candidate_band_px'],witness_m=row['geometric_minimum_world_conditional_m'])
            tag='LSp' if name.startswith('lsp') else 'CCoSp' if name.startswith('ccosp') else 'StSq'
            families[tag].append(row['outside_candidate_band_px'])
    return dict(cost=float(val['residual']@val['residual']/2),parts=val['weighted_parts'],
        gate=val['gate'],source=source,native_bands=bands,
        band_median_px={k:float(np.median(v)) for k,v in families.items()},
        body=body_groups(problem,val),turns=turn_episode_image_metrics(problem,base.turn_factors),
        step={k:v for k,v in b.items() if 'steps_' in k and isinstance(v,dict) and 'across_runner_mps' in v})

def checks(first,now,fixed):
    bad=[]
    for k,(p95,r) in first['source'].items():
        past,r0=fixed['source'].get(k,(p95,r))
        ceiling=max(p95,past+.25*r0)
        if now['source'][k][0]>ceiling+.001:bad.append(k)
    turn=turn_episode_image_checks(fixed.get('turn_episodes',first['turns']),now['turns'])
    torso=first['body']['torso']['pixel_p95']
    gate=now['gate'];original=first['gate']
    return dict(steel=bad,turns=[k for k,v in turn.items() if v['rejected']],
        torso=now['body']['torso']['pixel_p95']<=torso+max(2.,.1*torso),
        structure_max=gate['max_violation_m']<=original['max_violation_m']+1.e-6,
        unknown_collar=gate['collar_unknown_sections']<=original['collar_unknown_sections'],
        full_decrease=now['cost']<first['cost'],
        source_band_no_group_regression=all(now['band_median_px'][k]<=v+1. for k,v in first['band_median_px'].items()))

def store_candidate(p,ref,camera,info,doc,mass,saved,x,first,after,steps,log_summary,t,baseline,archived):
    state=p.body.state
    parent_steps=json.loads((PARENT/'run_summary.json').read_text('utf8'))['retained_step']
    shape=dict(saved)
    shape.update(state.packet(x[17:]))
    shape.update(archived)
    shape.update(x=x,source_evidence_sha256=np.array(digest(PRE/'source_evidence.json')),
        objective_block_weights_sha256=np.array(digest(OUT/'objective_block_weights.npz')),
        objective_policy_sha256=np.array(digest(OUT/'objective_policy.json')),
        objective_phase=np.array('physics'),parent_accepted_steps=np.array(parent_steps),
        sequence_contact_modes=p.action_factors.modes.modes,
        sequence_contact_status=p.action_factors.modes.status,
        sequence_contact_association_json=np.array(json.dumps(p.action_factors.modes.record,sort_keys=True)))
    temp=OUT/'solve/candidate.pending.npz'
    np.savez_compressed(temp,**shape)
    os.replace(temp,OUT/'solve/candidate.npz')
    write_json(OUT/'solve/candidate_selection.json',dict(retained_step=steps,attempt_step=steps,
        available=True,quality_failed=False,whole_programme_objective=True))
    write_json(OUT/'primary_quality_baseline.json',dict(source={
        **first['source'],**baseline['source']},torso=baseline['torso'],
        turn_episodes=baseline.get('turn_episodes',first['turns']),stage_reset_allowed=False))
    zref=absolute_reference(ref,state.current_curve(x[17:]),root=PRE)
    zref.relative_skate_curve=state.shoe_current_curve(x[17:])
    zref.binding=copy.copy(ref.binding);zref.binding.reference=zref
    z=export_candidate(PRE,OUT,zref,camera,info,doc.get('supports',[]))
    c=zref.at(zref.times)
    native=fk_state(zref,c,np.zeros((len(t),69)))
    C,_=mass.evaluate(c,np.zeros((len(t),69)))
    z.update(com_world_mid=mass.com(C),native_body_local_R_mid=native['local_R'],
        native_body_FK_world_R_mid=native['Q'][:,None]@native['frames'],
        source_action_ledger_sha256=np.array(digest(PRE/'source_evidence.json')),
        P_velocity_world_mps=zref.root_derivative(zref.times,1)*ref.binding.ratio,
        source_shot_ids=ref.p['shot_ids'],trajectory_evidence_status=np.where(p.body.obs['use'].sum(1)>=4,
           'source_constrained_estimate','temporal_inference'),
        position_uncertainty_status=np.full(len(t),'conditional single-camera depth; pending source review'),
        ice_plane_known=np.array(ref.ice_plane_known),sequence_contact_modes=p.action_factors.modes.modes,
        sequence_contact_status=p.action_factors.modes.status,shared_betas=ref.p['shared_betas'],
        shared_body_scale=ref.p['body_scale'],source_body_model_joint_ids=body_joint_assignment(p.body.obs))
    if 'body_joint_assignment_provenance_json' in p.body.obs:
        z['body_joint_assignment_provenance_json']=p.body.obs['body_joint_assignment_provenance_json']
    with np.load(PARENT/'fitted/result.npz',allow_pickle=False) as old:
        for k in ('pts','time_base','metres_per_scene_unit','shared_body_scale'):
            np.testing.assert_array_equal(z[k],old[k])
        untouched=~in_ranges(t)
        for k in ('P_world_mid','Q_world_mid','native22_world_mid','skate_R_world_mid'):
            np.testing.assert_allclose(z[k][untouched],old[k][untouched],rtol=0,atol=3e-8)
        shift=np.linalg.norm(z['P_world_mid']-old['P_world_mid'],axis=1)
    np.savez_compressed(OUT/'fitted/result.npz',**z)
    log_summary.update(native_exposures=len(t),whole_seconds=t[[0,-1]].tolist(),
        root_max_change_m=float(shift.max()),root_max_change_pts=int(z['pts'][np.argmax(shift)]),
        prepared=str(PRE),frozen_camera=True,retained_step=parent_steps+steps,additional_accepted_steps=steps,
        accuracy_accepted=False,source_visual_review='pending',parameter=str(OUT/'solve/candidate.npz'),
        result=str(OUT/'fitted/result.npz'),objective_during_acceptance='same weighted full programme rows at each step')
    write_json(OUT/'run_summary.json',log_summary)
    write_json(OUT/'replay_comparison.json',dict(before=first,after=after))
    return log_summary

def run():
    OUT.mkdir(exist_ok=True)
    (OUT/'solve').mkdir(exist_ok=True)
    if (OUT/'solve/candidate.npz').exists():raise FileExistsError(OUT)
    tick=time.monotonic()
    with (OUT/'progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        def emit(event,**data):
            v=dict(event=event,elapsed_s=time.monotonic()-tick,**data)
            log.write(json.dumps(v)+'\n');print(json.dumps(v),flush=True)
        emit('begin',maximum_accepted=MAX_ACCEPTED,wall_seconds=MAX_SECONDS)
        saved=load(PARENT/'solve/candidate.npz')
        ref,camera,info,doc,seed=load_reference(PROJECT,PRE)
        validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
        for key,file in [('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz')]:
            assert str(saved[key])==digest(PRE/file)
        ref.absolute_curve=curve_from_packet(saved,seed=True)
        ref.relative_skate_curve=shoe_curve_from_packet(saved,seed=True)
        x=saved['x'].copy()
        start=x.copy()
        execution=BodyExecution(BodyExecutionConfig(workers=8,blas_threads=1,gpu_batch_size=512,
            compatibility_backend='cuda_resident'))
        execution.attach(ref)
        with execution.limits():
            t=source_seconds(ref.p)
            policy=IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))
            p=SequenceJoint(ref,camera,PRE/'observations.npz',(t[0],t[-1]),doc.get('supports',[]),
                OUT/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=policy)
            p.al.multiplier[:]=saved['multiplier'];p.al.penalty=float(saved['penalty'])
            mass=ContinuousMassGeometry(ref)
            base=PrimaryActions(p,ref,mass,doc,warm=x[17:])
            p.action_factors=SequenceActions(p,ref,mass,doc,x[17:],base_factors=base)
            objective=primary_policy(p)
            weights=load(PARENT/'objective_block_weights.npz')
            for f in base.factors:
                if not isinstance(f,EpisodeScopedFactor):continue
                for b in f.evaluate(x[17:],False)['weighted_blocks']:
                    if b.name not in weights:
                        family=b.name.rsplit('/',1)[-1]
                        weights[b.name]=np.array({'native_source_support_contact':2.,
                            'step_runner_lateral':1.,'step_runner_travel':.5,'episode_support':1.}[family])
            p.objective_block_weights=weights
            write_json(OUT/'objective_policy.json',dict(objective,
                parent_objective_sha256=str(saved['objective_policy_sha256']),
                source_revision_sha256=digest(PRE/'source_evidence.json'),
                added_weights={k:float(v) for k,v in weights.items() if k.startswith('episode/steps_')}))
            np.savez_compressed(OUT/'objective_block_weights.npz',**weights)
            archived=load_associations(p,saved,x)
            np.savez_compressed(OUT/'source_associations.npz',**archived)
            baseline=json.loads((PARENT/'primary_quality_baseline.json').read_text('utf8'))
            first_path=OUT/'initial_metrics.json'
            if first_path.exists():
                first=json.loads(first_path.read_text('utf8'))
            else:
                val=p.evaluate(x,False)
                first=metrics(p,base,val)
            selected=first
            write_json(OUT/'initial_metrics.json',first)
            cols,bounds,frames,local,compact_execution=local_problem(p,base,t,x,archived,first)
            emit('programme_start',frames=len(t),variables=len(x),proposal_active_columns=len(cols),
                compact_source_frames=len(frames),cost=first['cost'],bands=first['band_median_px'],
                collision=first['parts']['self_collision'],step_rows={k:len(v['source_pts'])for k,v in first['step'].items()})
            accepted=0;trials=[]
            for k in range(MAX_ACCEPTED):
                if time.monotonic()-tick>MAX_SECONDS:break
                check_memory_budget()
                start_J=time.monotonic()
                r,J,costs,local_gate=local(x,True)
                column_scale=bounds/.2
                direction=column_scale*lsmr(J@diags(column_scale),-r,damp=1.1,atol=2.e-6,btol=2.e-6,maxiter=240)[0]
                direction*=min(1.,1./max(np.max(np.abs(direction)/bounds),1.e-12))
                direction,guards=protect_structure_direction(direction,first['gate']['max_violation_m'],
                    local_gate['margin'],local_gate['J'])
                direction*=min(1.,1./max(np.max(np.abs(direction)/bounds),1.e-12))
                slope=J@direction
                signs={}
                for name in ('self_collision','temporal_contact_mode'):
                    signs[name]=costs[name]
                # Verify the directional derivative against these actual compact
                # rows, with unchanged observation/association and local branch.
                if k==0:
                    a=x.copy();b=x.copy()
                    a[cols]+=1e-5*direction;b[cols]-=1e-5*direction
                    fd=(local(a,False)[0]-local(b,False)[0])/(2e-5)
                    write_json(OUT/'direction_check.json',dict(rows=len(r),columns=len(cols),
                        relative_error=float(np.linalg.norm(fd-slope)/max(np.linalg.norm(slope),1.e-12)),
                        absolute_max=float(np.max(np.abs(fd-slope))),scope='actual compact weighted residual; includes source, step, mode, and collision rows'))
                emit('direction',iteration=k,seconds=time.monotonic()-start_J,rows=len(r),nnz=J.nnz,
                    predicted_slope=float(r@slope),parts=costs,structure_projections=guards)
                del J
                found=False
                for alpha in (1.,.5,.25,.125,.0625):
                    if time.monotonic()-tick>MAX_SECONDS:break
                    y=x.copy();y[cols]+=alpha*direction
                    try:
                        rr,_,parts,test_gate=local(y,False)
                        if test_gate['unknown'] or test_gate['max_m']>first['gate']['max_violation_m']+1e-6:
                            emit('structure_screen',iteration=k,fraction=alpha,
                                 max_violation_m=test_gate['max_m'],unknown=test_gate['unknown'])
                            continue
                        if rr@rr>=r@r:
                            emit('compact_reject',iteration=k,fraction=alpha,objective=float(rr@rr/2))
                            continue
                        trial=p.evaluate(y,False)
                        m=metrics(p,base,trial)
                        gate=checks(first,m,baseline)
                        ok=(not gate['steel'] and not gate['turns'] and
                            all(v for key,v in gate.items()if key not in ('steel','turns'))
                            and m['cost']<selected['cost']-.005)
                        rec=dict(iteration=k,fraction=alpha,accept=ok,cost=m['cost'],checks=gate,
                            bands=m['band_median_px'],collision=m['parts']['self_collision'])
                        trials.append(rec);emit('global_trial',**rec)
                        write_json(OUT/'accepted_trials.json',trials)
                        if ok:
                            found=True;accepted+=1;selected=m;x=y;val=trial
                            temp=OUT/'solve/last_accepted_controls.pending.npz'
                            np.savez_compressed(temp,x=x,full_cost=m['cost'],accepted=accepted)
                            os.replace(temp,OUT/'solve/last_accepted_controls.npz')
                            break
                    except (ValueError,ArithmeticError) as err:
                        emit('domain_rejection',iteration=k,fraction=alpha,detail=str(err))
                if not found:break
            if not accepted:
                write_json(OUT/'failed_status.json',dict(stop='no_guarded_update',accepted=accepted))
                emit('held_for_review',accepted=accepted)
                return
            val=p.evaluate(x,False)
            write_json(OUT/'weighted_registry.json',verify(val['weighted_blocks'],val['residual']))
            record_action_factors(p,val,OUT/'final_action_factors.json',jacobian=False)
            emit('exporting',accepted=accepted,cost=selected['cost'])
            summary=store_candidate(p,ref,camera,info,doc,mass,saved,x,first,selected,accepted,
                dict(stop='bounded_joint_action_refinement',budget_s=MAX_SECONDS,elapsed_before_export_s=time.monotonic()-tick),
                t,baseline,archived)
            emit('candidate_saved',**summary)
            compact_execution.close()
        execution.close()

if __name__=='__main__':
    memory=ProcessMemoryGuard(ROOT/'solve_v2_resources',22.)
    try:run()
    finally:memory.close()
