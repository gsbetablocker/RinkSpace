"""One continuous skate-evidence refinement with a complete production replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from calibration.program_tracking.body_tracking.programme_skate_replay import (
    SkateReplay, common_witnesses, directions, refresh_surface, save, surface_events, write)
from calibration.program_tracking.body_tracking.programme_skate_evidence import ProgrammeSkateSourceFactors
from calibration.program_tracking.body_tracking.programme_joint_limits import NAMES
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.weighted_blocks import signature


SOURCES=('manual_native_leg','ordered_native_skate','reviewed_native_support')


def serial(value):
    return {k:v for k,v in value.items() if k not in ('value','anatomy','cavity')}


def groups(replay,value):
    metrics=value['value']['action_metrics']
    result=defaultdict(list)
    for name in SOURCES:
        for row in metrics.get(name,[]):
            k=row['event_id'],row['foot'],row.get('semantic',name)
            raw=row.get('distance_px',row.get('outside_candidate_band_px'))
            radius=row.get('radius_px',row.get('source_radius_px'))
            result[k].append((raw,radius,row))
    if replay.ev.get('tracked_sequence_policy',{}).get('legacy_source_guard'):
        for row in metrics.get('visible_runner_contours',[]):
            if row['role']=='fit':
                key='legacy_forward',row['foot'],row['structure_feature']
                result[key].append((row['distance_px'],row['pixel_uncertainty_radius_px'],row))
        for row in metrics.get('complete_runner_extents',[]):
            distance=np.asarray(row['model_to_source_distance_px'])
            mask=np.asarray(row.get('model_visible_mask',np.ones_like(distance)),bool)
            key='legacy_extent',row['foot'],'runner_lower_edge'
            result[key].append((distance[mask],row['pixel_uncertainty_radius_px'],row))
        for row in metrics.get('complete_visible_structure_extents',[]):
            key='legacy_structure_extent',row['foot'],row['structure_feature']
            result[key].append((row['distance_px'],row['pixel_uncertainty_radius_px'],row))
        for row in metrics.get('boot_axis',[]):
            key='boot_axis',row['foot'],'heel_to_toe'
            result[key].append(([row['axis_error_degrees']],row['angular_tolerance_degrees'],row))
    return result


def source_guard(replay,before,after):
    a,b=groups(replay,before),groups(replay,after)
    out=[]
    for k,rows in a.items():
        for (v,s,row),(u,_,following) in zip(rows,b[k],strict=True):
            if row['id']!=following['id']:raise ValueError('Source group correspondence changed')
            if u is None or v is None:
                if v is not None and u is None:out.append(dict(group=k,pts=row['pts'],reason='lost_visible_model_path'))
                continue
            p0,p1=(float(np.percentile(x,95)) for x in (v,u))
            if p1>max(s,p0)+.5*s:
                if k[0]=='boot_axis':
                    out.append(dict(group=k,pts=row['pts'],before_degrees=p0,after_degrees=p1,radius_degrees=s))
                else:
                    out.append(dict(group=k,pts=row['pts'],before_px=p0,after_px=p1,radius_px=s))
    return out


def anatomy_change(replay,before,after):
    delta=after['anatomy']-before['anatomy']
    out={}
    scopes=[*replay.ev['events'],dict(id='whole_programme',start_pts=int(replay.pts[0]),
                                     end_pts=int(replay.pts[-1]))]
    for e in scopes:
        mask=(replay.pts>=e['start_pts'])&(replay.pts<=e['end_pts'])
        v0,v1=before['anatomy'][mask],after['anatomy'][mask]
        increase=np.maximum(delta[mask],0)
        excursions=(v0<.25)&(v1>.50)&(increase>.25)
        q=np.unravel_index(np.argmax(increase),increase.shape)
        critical=np.unravel_index(np.argmax(np.where(excursions,increase,-1)),increase.shape)
        out[e['id']]=dict(worst_same_component_increase=float(increase[q]),
            worst_pts=int(replay.pts[mask][q[0]]),joint=NAMES[q[1]],component=int(q[2]),
            before=float(v0[q]),after=float(v1[q]),
            newly_outside_components=int(np.sum((v0<=0)&(v1>1e-10))),
            new_excursions=int(excursions.sum()),
            new_excursion_worst_pts=int(replay.pts[mask][critical[0]]) if excursions.any() else None,
            new_excursion_joint=NAMES[critical[1]] if excursions.any() else None,
            squared_violation_before=np.sum(v0*v0,axis=(0,2)).tolist(),
            squared_violation_after=np.sum(v1*v1,axis=(0,2)).tolist())
    return out


def evaluate_guard(replay,before,after):
    failures=[]
    if after['cost']>=before['cost']:failures.append('full_weighted_descent')
    for e,v in before['torso_by_event'].items():
        if after['torso_by_event'][e]['p95_px']>v['p95_px']+v['median_sigma_px']/2:
            failures.append('torso:'+e)
    new=source_guard(replay,before,after)
    if new:failures.append('source_rows_outside_finite_allowance')
    for k,v in before['weighted'].items():
        # These are untouched reviewed events. An actual difference beyond
        # arithmetic noise would mean the local controls leaked.
        if '/flight_' in k and abs(after['weighted'][k]-v)>1.e-7*max(1.,v):
            failures.append('protected_flight:'+k)
    anatomy=anatomy_change(replay,before,after)
    wear={}
    for e in replay.ev['events']:
        m=(replay.pts>=e['start_pts'])&(replay.pts<=e['end_pts'])
        a,b=before['cavity'][m],after['cavity'][m]
        tails=np.percentile(a,95,axis=0),np.percentile(b,95,axis=0)
        wear[e['id']]=dict(before_p95_m=tails[0].tolist(),after_p95_m=tails[1].tolist(),
                          worst_increase_m=np.maximum(b-a,0).max(0).tolist(),
                          before_squared=np.sum(a*a,axis=0).tolist(),after_squared=np.sum(b*b,axis=0).tolist())
        if np.any(tails[1]>tails[0]*(1.05)+1e-10):failures.append('cavity_tail:'+e['id'])
        # Compare the existing joint domain locally at the same exposure and
        # component. The declared domain cost always acts on the full value.
    for scope,comparison in anatomy.items():
        if comparison['new_excursions']:
            failures.append('new_anatomical_excursion:'+scope)
    return dict(failures=failures,source_rows=new,per_event_cavity=wear,
                pointwise_anatomical_comparison=anatomy)


def source_derivative(replay,move,at=None):
    f=replay.problem.action_factors.reviewed_skates
    x=(replay.x if at is None else at)[17:];d=move[17:];eps=2.e-5
    before=f.evaluate(x,True)
    plus=f.evaluate(x+eps*d,False)
    minus=f.evaluate(x-eps*d,False)
    out={}
    for b,p,m in zip(before['weighted_blocks'],plus['weighted_blocks'],minus['weighted_blocks']):
        exact=b.jacobian@move
        diff=(p.residual-m.residual)/(2*eps)
        out[b.name]=dict(relative_error=float(np.linalg.norm(exact-diff)/max(1.e-9,np.linalg.norm(exact),np.linalg.norm(diff))),
                         norm=float(np.linalg.norm(exact)),rows=len(exact))
    f.evaluate(x,False)
    return out


def association_record(f):
    return f.association_record()


def finalize(r,args,x,precheck,*,surface_parent=None):
    r.emit('selected_for_structure_replay',name=precheck['name'],fraction=precheck['fraction'],
           first_stage_failures=precheck['failures'])
    parent_scan,parent_x=(r.s0,r.x) if surface_parent is None else surface_parent
    scan,coverage=refresh_surface(r.problem,parent_scan,parent_x,x,r.document,r.emit)
    save(args.output/'actual_surface.npz',**scan)
    events=[dict(e,_face_parts=r.geometry.parts) for e in r.ev['events']]
    events.append(dict(id='whole_programme',start_pts=int(r.pts[0]),end_pts=int(r.pts[-1]),
                       _face_parts=r.geometry.parts))
    scans={name:surface_events(p,events,r.tb[0]/r.tb[1]) for name,p in (
        ('s09_parent',r.s0),('lsp_parent',r.sold),('candidate',scan))}
    for name,data in scans['candidate'].items():
        q=scans['s09_parent'][name]['pairs']
        data['added_semantic_pairs']=sorted(set(data['pairs'])-set(q))
        data['removed_semantic_pairs']=sorted(set(q)-set(data['pairs']))
    write(args.output/'surface_comparison.json',dict(coverage=coverage,results=scans))
    initial_visible={a['id'] for a in association_record(r.ordered) if a['match'] is not None}
    r.refresh(x,common_witnesses([r.s0,r.sold,scan],r.geometry))
    lost_paths=sorted(initial_visible-{a['id'] for a in association_record(r.ordered) if a['match'] is not None})
    final_parent=r.full(r.x,'s09_parent_refreshed')
    final_old=r.full(r.xold,'lsp_parent_refreshed')
    final=r.full(x,'candidate_refreshed')
    if not (args.output/'production_registry.json').exists():
        write(args.output/'production_registry.json',final['registry'])
    guard=evaluate_guard(r,final_parent,final)
    old_guard=evaluate_guard(r,final_old,final)
    guard['lost_visible_paths_after_refresh']=lost_paths
    if lost_paths:guard['failures'].append('lost_visible_paths_after_refresh')
    final['coverage'].update(surface_checked_samples=coverage['checked_samples'])
    if final['weighted']['surface_intersection']>final_parent['weighted']['surface_intersection']*1.01+1e-10:
        guard['failures'].append('refreshed_actual_surface_response')
    for e in r.ev['events']:
        event=e['id']
        if scans['candidate'][event]['added_semantic_pairs']:
            guard['failures'].append('new_surface_pair:'+event)
    # Stronger historical structural comparison is reported and used
    # to withhold promotion, never as a per-exposure free allowance.
    if any(k.startswith('cavity_tail:') for k in old_guard['failures']):
        guard['failures'].append('cavity_regression_vs_pre_s09')
    quality='full_replay_engineering_step' if not guard['failures'] else 'review_only'
    write(args.output/'ordered_associations.json',association_record(r.ordered))
    assoc_sha=digest(args.output/'ordered_associations.json')
    p=r.problem
    for k in ('prepared_input_sha256','prepared_observations_sha256','programme_context_json',
              'parent_accepted_steps','source_association_origin'):
        p.checkpoint_extra[k]=r.latest[k].copy()
    for k in r.latest:
        if k.startswith('source_association_') and k not in p.checkpoint_extra:p.checkpoint_extra[k]=r.latest[k].copy()
    p.checkpoint_extra.update(objective_phase=np.array('programme_skate_evidence'),
        objective_policy_sha256=np.array(digest(args.output/'objective_weights.json')),
        objective_block_weights_sha256=np.array(r.weights_record()['weight_digest']),
        programme_structure_json=np.array(json.dumps(r.document['programme_structure'],sort_keys=True)),
        source_evidence_sha256=np.array(digest(r.source_path)),ordered_source_association_sha256=np.array(assoc_sha),
        ordered_source_association_json=np.array(json.dumps(association_record(r.ordered),sort_keys=True)),
        source_replay_status=np.array(quality),parent_parameter_sha256=np.array(digest(args.latest)),
        anatomical_basis=r.joints.chart.basis,anatomical_rest_R=r.joints.chart.rest,
        anatomical_reference_twist=r.joints.chart.reference_twist,
        anatomical_policy_json=np.array(json.dumps(r.joints.configuration,sort_keys=True)))
    candidate=p.checkpoint_fields(x)
    np.testing.assert_allclose(candidate['absolute_rotation_controls'][:,0],
                                  r.latest['absolute_rotation_controls'][:,0],rtol=0,atol=2.e-12)
    # Lifted SU(2) root log reference is part of the preserved path.
    np.testing.assert_allclose(candidate['absolute_rotation_edge_reference_quaternion'][:,0],
                                  r.latest['absolute_rotation_edge_reference_quaternion'][:,0],rtol=0,atol=2.e-12)
    for k in ('sequence_contact_modes','sequence_contact_status'):
        np.testing.assert_array_equal(candidate[k],r.latest[k])
    cols=np.flatnonzero(x!=r.x)
    if not len(cols) or not np.isin(cols,r.selection()[1]).all():raise ValueError('Unapproved control change')
    save(args.output/'candidate.npz',**candidate)
    # Freeze check associations on the parent and never use these
    # observations to choose a direction, weights or candidate.
    check=ProgrammeSkateSourceFactors(p,r.ev,role='check')
    check.ordered.associate(r.x[17:])
    rows={}
    for label,z,v in [('s09_parent',r.x,final_parent),('lsp_parent',r.xold,final_old),('candidate',x,final)]:
        rows[label]={'fit':{k:v['value']['action_metrics'][k] for k in SOURCES},
                     'check':check.evaluate(z[17:],False)['metrics']}
    write(args.output/'source_comparison.json',rows)
    write(args.output/'acceptance.json',dict(schema='rinkspace.programme-skate-replay.v1',
        acceptance=quality,engineering_step_accepted=not guard['failures'],user_accepted=False,accuracy_accepted=False,
        same_camera_sha256=r.info['state_sha256'],source_evidence=str(r.source_path),
        source_evidence_sha256=digest(r.source_path),weight_digest=r.weights_record()['weight_digest'],
        association_sha256=assoc_sha,shared_geometry_sha256=signature(x[:17]),
        programme_clock_sha256=signature(r.pts,np.asarray(r.tb)),coverage=final['coverage'],
        parent_sha256=digest(args.latest),previous_sha256=digest(args.previous),
        parameter_sha256=digest(args.output/'candidate.npz'),root_lifted_path_preserved=True,
        final_guard=guard,comparison_vs_lsp=old_guard,proposal=precheck,
        actual_weighted_same_version=dict(s09=serial(final_parent),lsp=serial(final_old),candidate=serial(final)),
        whole_programme_joint_refit=False,local_calculation_is_proposal_only=True,
        model_height_is_not_actual_ice_contact=True))
    r.emit('candidate_saved',status=quality,candidate=str(args.output/'candidate.npz'),
           reasons=guard['failures'])


def separated_proposals(r,args):
    """Test proximal motion while leaving the fitted shank/boot relation intact."""
    old=args.distal_from
    def read_local(name):return json.loads((old/name).read_text(encoding='utf-8'))
    receipt=read_local('acceptance.json')
    path=old/'candidate.npz'
    if (digest(path)!=receipt['parameter_sha256'] or digest(args.latest)!=receipt['parent_sha256']
            or digest(args.previous)!=receipt['previous_sha256'] or receipt['coverage']['native_exposures']!=8329):
        raise ValueError('Separated-motion proposal requires the verified full seed from these parents')
    saved=load(path);seed=saved['x']
    np.testing.assert_array_equal(saved['absolute_seed_rotation_controls'],r.latest['absolute_seed_rotation_controls'])
    np.testing.assert_array_equal(saved['absolute_seed_position_controls'],r.latest['absolute_seed_position_controls'])
    r.ordered.restore(json.loads(str(saved['ordered_source_association_json'])))
    write(args.output/'ordered_initial_associations.json',association_record(r.ordered))
    active=r.selection()[1]
    state=r.problem.body.state
    slot=(active-17)%69
    proximal=active[(active<17+state.body_size)&np.isin(slot,np.r_[6:12,15:21])]
    move=np.zeros_like(seed)
    move[proximal]=r.x[proximal]-seed[proximal]
    if not np.any(move):raise ValueError('The proximal trial has no motion to replay')
    np.testing.assert_array_equal(move[17+state.body_size:],0.)
    from calibration.program_tracking.body_tracking.rebuild_prepare import digest as sha
    source=r.problem.action_factors.reviewed_skates.evaluate(seed[17:],True)
    directional={b.name:float(np.asarray(r.problem.objective_block_weights[b.name])*
                    (b.residual@(b.jacobian@move))) for b in source['weighted_blocks']}
    write(args.output/'directions.json',dict(
        separated_hip_and_knee=dict(native_weighted_rTJd=directional,active_columns=len(proximal),
            excludes='ankle, toe, relative shoe, pelvis, root Q and upper body'),
        seed_full_directions=dict(path=str(old/'directions.json'),sha256=sha(old/'directions.json'),
                                  data=read_local('directions.json'))))
    derivative=source_derivative(r,move,seed)
    if any(a['relative_error']>1.e-3 for a in derivative.values()):
        raise ValueError('Separated-motion native directional derivative did not verify')
    write(args.output/'native_source_jacobian_check.json',derivative)
    write(args.output/'diagnostic_rigid_task_rank.json',r.ordered.pose_task_probe(seed[17:],
                                  family_weight=r.problem.objective_block_weights['ordered_native_skate']))
    save(args.output/'proposals.npz',x=seed,separated_proximal=move)
    r.emit('separated_proximal_direction',seed_sha256=digest(path),columns=len(proximal),source_rTJd=directional)
    base=r.full(seed,'previous_review_seed')
    latest=r.full(r.x,'s09_parent')
    earlier=r.full(r.xold,'lsp_parent')
    write(args.output/'initial_parents.json',dict(seed=serial(base),s09=serial(latest),lsp=serial(earlier)))
    trials=[];candidates=[]
    for alpha in (1.,.65):
        x=seed+alpha*move
        q=r.full(x,'separated_proximal_'+str(alpha),verify_rows=False)
        baseline_guard=evaluate_guard(r,base,q)
        recent_guard=evaluate_guard(r,latest,q)
        previous_guard=evaluate_guard(r,earlier,q)
        guard=dict(name='separated_proximal',fraction=alpha,weighted=q['weighted'],cost=q['cost'],
            actual_delta=q['cost']-base['cost'],failures=recent_guard['failures'],
            comparison_vs_review_seed=baseline_guard,comparison_vs_prior=previous_guard,comparison_vs_latest=recent_guard)
        trials.append(guard)
        r.emit('programme_trial',name='separated_proximal',fraction=alpha,actual_delta=guard['actual_delta'],
               failures=guard['failures'],source_weighted={b:q['weighted'][b] for b in SOURCES},
               older_failures=previous_guard['failures'])
        source_loss=sum(q['weighted'][b] for b in SOURCES)
        # Guard per-event fit, anatomy and wear first; do not rank with check rows.
        structural=sum(k.startswith(('cavity_tail','new_anatomical','torso','protected_flight'))
                       for k in recent_guard['failures']+previous_guard['failures'])
        score=(structural,len(recent_guard['source_rows']),source_loss)
        candidates.append((score,x,guard))
        if not recent_guard['failures'] and not structural:break
    write(args.output/'full_trials.json',trials)
    _,x,guard=min(candidates,key=lambda z:z[0])
    return x,guard


def retract_new_pair_support(r,args):
    """Revert only the spline controls spanning newly intersecting samples."""
    proposed,clean=args.repair_new_pairs_from,args.distal_from
    if clean is None:raise ValueError('A checked alternate state is required')
    def info(stage):
        rr=json.loads((stage/'acceptance.json').read_text(encoding='utf8'))
        q=stage/'candidate.npz'
        if (rr['parameter_sha256']!=digest(q) or rr['parent_sha256']!=digest(args.latest)
                or rr['previous_sha256']!=digest(args.previous) or rr['coverage']['surface_checked_samples']!=16657):
            raise ValueError('Surface retraction needs the verified same-parent complete replay')
        return rr,load(q)
    sr,seed=info(proposed);cr,fallback=info(clean)
    for k in ('absolute_seed_rotation_controls','absolute_seed_position_controls','relative_skate_rotation_controls'):
        np.testing.assert_array_equal(seed[k],fallback[k],err_msg=k)
    scan=load(proposed/'actual_surface.npz')
    np.testing.assert_array_equal(scan['source_pts'],r.pts)
    events=json.loads((proposed/'surface_comparison.json').read_text(encoding='utf8'))['results']['candidate']
    clean_events=json.loads((clean/'surface_comparison.json').read_text(encoding='utf8'))['results']['candidate']
    sample=scan['pair_counts'];selected=[];parts=scan['parts'].tolist();rows=[]
    for e in r.ev['events']:
        key=e['id']
        for label in events[key]['added_semantic_pairs']:
            if label in clean_events[key]['added_semantic_pairs']:continue
            i,j=sorted(parts.index(p) for p in label.split('/'))
            seconds=scan['source_seconds'][sample[:,0]]
            pick=(sample[:,1]==i)&(sample[:,2]==j)&(seconds>=e['start_pts']/12800)&(seconds<=e['end_pts']/12800)
            selected.extend(sample[pick,0].tolist())
            rows.append(dict(event=key,pair=label,times_s=seconds[pick].tolist()))
    if not selected:raise ValueError('No new sampled pair to retract')
    s=r.problem.body.state
    # Include both neighbouring half-exposure structure samples. Changing
    # controls preserves the existing continuous curve, never a frame patch.
    sample_ids=np.clip(np.asarray(selected)[:,None]+np.arange(-2,3),0,len(scan['times'])-1)
    spans=np.clip(np.searchsorted(s.knots,scan['times'][sample_ids],side='right')-4,0,s.m-4)
    controls=np.unique(spans[...,None]+np.arange(4))
    columns=(17+69*controls[:,None]+np.r_[6:12,15:21]).ravel()
    columns=np.intersect1d(columns,r.selection()[1])
    x=seed['x'].copy();x[columns]=fallback['x'][columns]
    if np.array_equal(x,seed['x']):raise ValueError('No proximal spline difference at new pair samples')
    r.ordered.restore(json.loads(str(seed['ordered_source_association_json'])))
    write(args.output/'ordered_initial_associations.json',association_record(r.ordered))
    upstream=json.loads((proposed/'directions.json').read_text(encoding='utf8'))
    write(args.output/'directions.json',dict(prior_true_direction_audit=upstream,
        prior_sha256=digest(proposed/'directions.json'),retracted_columns=int(np.count_nonzero(x!=seed['x'])),
        triggering_new_pairs=rows))
    check=source_derivative(r,x-seed['x'],seed['x'])
    if any(a['relative_error']>1.e-3 for a in check.values()):raise ValueError('Retraction direction did not verify')
    write(args.output/'native_source_jacobian_check.json',check)
    write(args.output/'diagnostic_rigid_task_rank.json',r.ordered.pose_task_probe(seed['x'][17:],
                                  family_weight=r.problem.objective_block_weights['ordered_native_skate']))
    trial=dict(name='new_pair_spline_retraction',fraction=1.,failures=[],changed_columns=int(np.count_nonzero(x!=seed['x'])),
        triggering_new_pairs=rows,prior_parameter_sha256=sr['parameter_sha256'],alternate_parameter_sha256=cr['parameter_sha256'])
    write(args.output/'full_trials.json',[trial])
    r.emit('spline_retraction_proposal',**trial)
    return x,trial,(scan,seed['x'])


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ('project','prepared','source','latest','previous','latest-surface','previous-surface',
              'annotations','plane','contours','output'):
        a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--repair-new-pairs-from',type=Path,help='Retract newly crossed sampled pairs on their existing spline support')
    a.add_argument('--distal-from',type=Path,help='Verified full candidate to retain as the distal/shoe state')
    a.add_argument('--wear-seed',action='store_true',help='Use preceding attached leg state as a numerical seed; keep latest P and lifted Q')
    args=a.parse_args()
    args.output.mkdir(parents=True)
    log_path=args.output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path))),flush=True)
    with log_path.open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(args,log)
        write(args.output/'command.json',dict(command=sys.argv,inputs={k:str(v) for k,v in vars(args).items()}))
        write(args.output/'objective_weights.json',r.weights_record())
        if args.distal_from is not None:
            with r.execution.limits():
                if args.repair_new_pairs_from is None:
                    x,proposal=separated_proposals(r,args)
                    finalize(r,args,x,proposal)
                else:
                    x,proposal,parent=retract_new_pair_support(r,args)
                    finalize(r,args,x,proposal,surface_parent=parent)
            r.execution.close()
            return
        with r.execution.limits():
            # Check the source chain before allocating the full-programme Jacobian.
            rng=np.random.default_rng(1109)
            columns=r.selection()[1]
            probe=np.zeros_like(r.x);probe[columns]=rng.normal(size=len(columns))
            probe/=np.linalg.norm(probe)
            numerical_seed=r.x.copy()
            if args.wear_seed:
                bs=r.problem.body.state.body_size
                legs=columns[(columns<17+bs)&((columns-17)%69>=6)]
                numerical_seed[legs]=r.xold[legs]
                r.emit('attached_wear_seed',changed_leg_controls=int(np.count_nonzero(numerical_seed!=r.x)),
                       latest_P_and_Q_retained=True)
            if args.wear_seed:r.ordered.associate(numerical_seed[17:])
            write(args.output/'ordered_initial_associations.json',association_record(r.ordered))
            preflight=source_derivative(r,probe,numerical_seed)
            write(args.output/'source_jacobian_preflight.json',preflight)
            if any(d['relative_error']>1.e-3 for d in preflight.values()):
                raise ValueError('Native source directional derivative does not match its objective: '+str(preflight))
            seed_name='attached_wear_seed' if args.wear_seed else 's09_parent'
            baseline=r.full(numerical_seed,seed_name,jacobian=True)
            proposals,steps=directions(r,baseline,wear_seed=args.wear_seed)
            write(args.output/'production_registry.json',baseline['registry'])
            write(args.output/'directions.json',steps)
            save(args.output/'proposals.npz',x=numerical_seed,**proposals)
            r.emit('directional_ledger_written',directions=list(steps),
                   rTJd={k:v['total_rTJd'] for k,v in steps.items()})
            check=source_derivative(r,proposals['B_attached_joint'],numerical_seed)
            write(args.output/'native_source_jacobian_check.json',check)
            if any(d['relative_error']>1.e-3 for d in check.values()):
                raise ValueError('Native source derivative fails in the proposed direction: '+str(check))
            write(args.output/'diagnostic_rigid_task_rank.json',r.ordered.pose_task_probe(numerical_seed[17:],
                                        family_weight=r.problem.objective_block_weights['ordered_native_skate']))
            preceding=r.full(r.xold,'lsp_parent')
            initial=dict(lsp=serial(preceding))
            if args.wear_seed:
                initial.update(seed=serial(baseline),s09=serial(r.full(r.x,'s09_parent')))
            else:initial['s09']=serial(baseline)
            write(args.output/'initial_parents.json',initial)
            attempts=[];winner=None;fallback=None
            # Both proposals are evaluated on the same complete nonlinear
            # objective. No local ftol ever accepts a programme candidate.
            for name in ('B_attached_joint','B_structural_correction'):
                for fraction in ((.6,.24,.09) if args.wear_seed else (1.,.5,.25)):
                    x=numerical_seed+fraction*proposals[name]
                    if np.array_equal(x,r.x):continue
                    label=name+'_'+str(fraction)
                    q=r.full(x,label,verify_rows=False)
                    guard=evaluate_guard(r,baseline,q)
                    actual=q['cost']-baseline['cost']
                    trial=dict(name=name,fraction=fraction,cost=q['cost'],weighted=q['weighted'],
                               weighted_delta={k:q['weighted'][k]-v for k,v in baseline['weighted'].items()},
                               actual_delta=actual,linear_directional_slope=fraction*steps[name]['total_rTJd'],**guard)
                    attempts.append(trial)
                    r.emit('programme_trial',name=name,fraction=fraction,actual_delta=actual,
                           failures=guard['failures'],weighted=q['weighted'])
                    score=(len(guard['failures']),q['cost'])
                    if fallback is None or score<fallback[0]:fallback=score,x,q,trial
                    if not guard['failures']:
                        if winner is None or q['cost']<winner[2]['cost']:winner=score,x,q,trial
                        break
            if fallback is None:raise RuntimeError('No nonzero control proposal')
            chosen=winner or fallback
            x=chosen[1];precheck=chosen[3]
            write(args.output/'full_trials.json',attempts)
            finalize(r,args,x,precheck)
        r.execution.close()


if __name__=='__main__':main()
