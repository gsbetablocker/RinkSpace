"""Continuous source-led programme solve with a complete native replay."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr

from calibration.program_tracking.body_tracking.ordered_native_skate import OrderedNativeSkateFactors
from calibration.program_tracking.body_tracking.programme_skate_evidence import ProgrammeSkateSourceFactors
from calibration.program_tracking.body_tracking.programme_skate_replay import (
    SkateReplay, common_witnesses, refresh_surface, save, surface_events, write)
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from scripts.refit_programme_skates import SOURCES, evaluate_guard, serial, source_guard


IMAGE={'manual_native_leg','ordered_native_skate','reviewed_native_support',
       'full_source_skate_structure','source_pixel_edge_profile','complete_runner_extent',
       'complete_visible_structure_extent','boot_axis'}


def protected(r):
    shot=r.ref.p['shot_ids']=='s01'
    result=[(float(r.clock[shot][0])-.02,float(r.clock[shot][-1])+.02),
            (77.58,79.02),(156.30,166.00)]
    for e in r.document['source_episode_review']['episodes']:
        if e['kind']=='flight_visible_candidate':
            result.append((e['start_pts']*r.tb[0]/r.tb[1]-.02,e['end_pts_inclusive']*r.tb[0]/r.tb[1]+.02))
    return result


def scopes(r,columns):
    state=r.problem.body.state
    body=columns<17+state.body_size
    start=np.empty(len(columns));end=start.copy();feet=np.empty(len(columns),int)
    for mask,offset,size,knots in ((body,17,69,state.knots),
            (~body,17+state.body_size,6,state.shoe_curve.knots)):
        k=(columns[mask]-offset)//size
        start[mask]=knots[k]+r.clock[0];end[mask]=knots[k+4]+r.clock[0]
        axes=(columns[mask]-offset)%size
        feet[mask]=((axes-6)//3)%3 if size==69 else axes//3
    return start,end,feet


def active(r,blocks):
    state=r.problem.body.state
    j=[b.jacobian.indices for b in blocks if b.name in IMAGE and b.jacobian is not None]
    cols=np.unique(np.concatenate(j))
    body=cols<17+state.body_size
    cols=cols[(~body)|(np.isin((cols-17)%69,np.r_[6:12,15:21,24:30])&(cols>=17))]
    a,b,_=scopes(r,cols)
    fixed=np.zeros(len(cols),bool)
    for low,high in protected(r):
        fixed|=(a<=high)&(b>=low)
    return cols[~fixed]


def directions(r,baseline):
    blocks=baseline['value']['weighted_blocks']
    columns=active(r,blocks)
    sparse=[]
    for b in blocks:
        M=b.jacobian[:,columns].tocsr()
        take=np.diff(M.indptr)>0
        if take.any():
            sparse.append((b.name,M[take],b.residual[take]))
    result={}
    for label,multiplier,trust in [('source_led',2.8,.29),('coupled',1.45,.24),('production',1.,.21)]:
        source=[multiplier if key in IMAGE else 1. for key,_,_ in sparse]
        J=vstack([m*w for (_,m,_),w in zip(sparse,source,strict=True)],format='csr')
        values=np.concatenate([v*w for (_,_,v),w in zip(sparse,source,strict=True)])
        diagonal=1/np.maximum(1.,np.sqrt(np.asarray(J.power(2).sum(0)).ravel()))
        solution=lsmr(J@diags(diagonal),-values,damp=.20,atol=4e-5,btol=4e-5,maxiter=120)
        d=np.zeros_like(r.x);d[columns]=solution[0]*diagonal
        n=r.problem.body.state.body_size
        for a,cap in ((d[17:17+n].reshape(-1,23,3),trust),(d[17+n:].reshape(-1,2,3),trust*.68)):
            length=np.linalg.norm(a,axis=2)
            a*=np.minimum(1.,cap/np.maximum(length,1e-12))[...,None]
        slopes={name:float(v@(m@d[columns])) for name,m,v in sparse}
        r.emit('continuous_direction',name=label,columns=len(columns),rows=J.shape[0],nonzeros=J.nnz,
               iterations=int(solution[2]),slopes=slopes)
        result[label]=d
    r.emit('protected_intervals',source_seconds=protected(r))
    save(r.args.output/'initial_directions.npz',x=r.x,active_columns=columns,**result)
    return result


def measured_sources(r,x):
    actions=r.problem.action_factors
    factors=[actions.contour,actions.extent,actions.structure_extent,actions.reviewed_skates,actions.boot_axes]
    metrics={}
    for factor in factors:
        if factor is not None:
            metrics.update(factor.evaluate(x[17:],False)['metrics'])
    return dict(value=dict(action_metrics=metrics))


def changed_ordered(r,x,initial):
    return r.ordered.associate_changed(x[17:],r.x[17:],initial)

def statistics(r,first,second):
    m0=first['value']['action_metrics']['ordered_native_skate']
    m1=second['value']['action_metrics']['ordered_native_skate']
    buckets=defaultdict(list)
    for a,b in zip(m0,m1,strict=True):
        assert a['id']==b['id']
        if 'distance_px' not in a or 'distance_px' not in b:continue
        stamp=a['pts']*r.tb[0]/r.tb[1]
        values=[float(np.percentile(v['distance_px'],95)) for v in (a,b)]
        buckets[a['semantic']].append(values)
        if 76.<=stamp<=95. and a['event_id']=='temporal_skate_s05':
            buckets['s05_'+a['semantic']].append(values)
    return {k:dict(rows=len(v),median_before_after=np.median(v,axis=0).tolist(),
                   p95_before_after=np.percentile(v,95,axis=0).tolist()) for k,v in buckets.items()}


summary=statistics


def soften(r,step,bad):
    c=np.flatnonzero(step)
    low,high,side=scopes(r,c)
    cut=np.zeros(len(c),bool)
    for row in bad:
        at=row['pts']*r.tb[0]/r.tb[1]
        cut|=(low<=at+.019)&(high>=at-.019)&(side==row['group'][1])
    step[c[cut]]*=.29
    return int(cut.sum())


def run(r):
    out=r.args.output
    write(out/'objective_weights.json',r.weights_record())
    baseline=r.full(r.x,'whole_programme_parent',jacobian=True)
    write(out/'production_registry.json',baseline['registry'])
    r.emit('whole_parent_verified',cost=baseline['cost'],coverage=baseline['coverage'],
           boot_axis_rows=len(r.problem.action_factors.boot_axes.axes),weighted=baseline['weighted'])
    if not any(b.name=='boot_axis' for b in baseline['value']['weighted_blocks']):
        raise ValueError('Direct boot axis was not assembled in the continuous objective')
    candidates=directions(r,baseline)
    native=next(b for b in baseline['value']['weighted_blocks'] if b.name=='ordered_native_skate')
    initial_matches=r.ordered.matches
    d=candidates['coupled']
    epsilon=2e-5
    difference=np.diff([r.ordered.evaluate(r.x[17:]+f*epsilon*d[17:],False)['residual']
                        for f in (-1,1)],axis=0)[0]/(2*epsilon)
    measured=difference*np.sqrt(float(r.problem.objective_block_weights['ordered_native_skate']))
    exact=native.jacobian@d
    error=float(np.linalg.norm(measured-exact)/max(1e-12,np.linalg.norm(measured),np.linalg.norm(exact)))
    r.emit('real_source_derivative',rows=len(exact),relative_error=error,norm=float(np.linalg.norm(exact)))
    if error>.03 or not np.any(exact):raise ValueError('Source Jacobian direction check failed')
    baseline['value']['jac']=None;baseline['value']['weighted_blocks']=[]
    r.problem.cache=None;r.problem.cache_x=None
    records=[]
    picked=None
    for name,move in candidates.items():
        step=move.copy()
        for trial in range(5):
            x=r.x+step
            n=changed_ordered(r,x,initial_matches)
            value=measured_sources(r,x)
            bad=source_guard(r,baseline,value)
            by_part=statistics(r,baseline,value)
            r.emit('actual_source_trial',direction=name,trial=trial,changed_paths=n,failures=len(bad),raw=by_part)
            if not bad:break
            cut=soften(r,step,bad)
            if not cut:break
        for fraction in (1.,.55):
            x=r.x+fraction*step
            changed_ordered(r,x,initial_matches)
            actual=r.full(x,f'{name}_{fraction}_whole',verify_rows=False)
            guard=evaluate_guard(r,baseline,actual)
            raw=statistics(r,baseline,actual)
            steel=raw.get('s05_runner_lower_edge',raw.get('runner_lower_edge',{})).get('p95_before_after')
            row=dict(direction=name,fraction=fraction,failures=guard['failures'],cost=actual['cost'],
                     source_conflicts=guard['source_rows'][:20],raw=raw)
            records.append(row);r.emit('whole_programme_trial',weighted=actual['weighted'],**row)
            if not guard['failures'] and steel and steel[1]<steel[0]:
                picked=x,row
                break
        if picked is not None:break
    write(out/'proposal_trials.json',records)
    if picked is None:
        raise ValueError('No source-led move passed the full programme guard')
    x,row=picked
    verify_candidate(r,x,row,scan_seed=(r.s0,r.x))


def checkpoint(r,x,status):
    path=r.args.output
    p=r.problem
    for k in ('prepared_input_sha256','prepared_observations_sha256','programme_context_json',
              'parent_accepted_steps','source_association_origin'):
        p.checkpoint_extra[k]=r.latest[k].copy()
    for k in r.latest:
        if k.startswith('source_association_') and k not in p.checkpoint_extra:
            p.checkpoint_extra[k]=r.latest[k].copy()
    assoc=r.ordered.association_record()
    write(path/'ordered_associations.json',assoc)
    p.checkpoint_extra.update(objective_phase=np.array('programme_blade_trajectory'),
        objective_policy_sha256=np.array(digest(path/'objective_weights.json')),
        objective_block_weights_sha256=np.array(r.weights_record()['weight_digest']),
        programme_structure_json=np.array(json.dumps(r.document['programme_structure'],sort_keys=True)),
        source_evidence_sha256=np.array(digest(r.source_path)),
        ordered_source_association_sha256=np.array(digest(path/'ordered_associations.json')),
        ordered_source_association_json=np.array(json.dumps(assoc,sort_keys=True)),
        source_replay_status=np.array(status),parent_parameter_sha256=np.array(digest(r.args.latest)),
        anatomical_basis=r.joints.chart.basis,anatomical_rest_R=r.joints.chart.rest,
        anatomical_reference_twist=r.joints.chart.reference_twist,
        anatomical_policy_json=np.array(json.dumps(r.joints.configuration,sort_keys=True)))
    candidate=p.checkpoint_fields(x)
    np.testing.assert_array_equal(candidate['absolute_position_controls'],r.latest['absolute_position_controls'])
    for k in ('absolute_rotation_controls','absolute_rotation_edge_reference_quaternion'):
        np.testing.assert_allclose(candidate[k][:,0],r.latest[k][:,0],rtol=0,atol=2e-12)
    for k in ('sequence_contact_modes','sequence_contact_status'):
        np.testing.assert_array_equal(candidate[k],r.latest[k])
    save(path/'candidate.npz',**candidate)
    return path/'candidate.npz'


def verify_candidate(r,x,selected,scan_seed=None,initial_matches=None):
    output=r.args.output
    save(output/'selected_unverified.npz',x=x)
    # Actual native and midpoint surface scan. Exact spline complements use
    # the bound cached parent geometry, with fresh unaffected probes.
    cached,previous=(r.s0,r.x) if scan_seed is None else scan_seed
    try:
        s,coverage=refresh_surface(r.problem,cached,previous,x,r.document,r.emit,
            exact_saved_pose=scan_seed is None)
    except ValueError as error:
        if not str(error).startswith('Surface cache'):raise
        r.emit('surface_cache_rejected',reason=str(error))
        try:
            s,coverage=refresh_surface(r.problem,cached,previous,x,r.document,r.emit,exact_saved_pose=True)
        except ValueError as error:
            if not str(error).startswith('Surface cache'):raise
            r.emit('surface_cache_rejected',reason=str(error))
            r.s0,_=refresh_surface(r.problem,r.s0,r.x,r.x,r.document,r.emit,exact_saved_pose=True,rescan_all=True)
            save(output/'fresh_parent_surface.npz',**r.s0)
            s,coverage=refresh_surface(r.problem,r.s0,r.x,x,r.document,r.emit,exact_saved_pose=True)
    sfile=output/'surface_actual.npz';save(sfile,**s)
    surface_scope=[dict(id='whole_programme',start_pts=int(r.pts[0]),end_pts=int(r.pts[-1]),_face_parts=r.geometry.parts),
        *[dict(e,_face_parts=r.geometry.parts) for e in r.ev['events'] if e['id'].startswith('blade_')]]
    scan_table={k:surface_events(t,surface_scope,r.tb[0]/r.tb[1]) for k,t in (('parent',r.s0),('candidate',s))}
    write(output/'surface_comparison.json',dict(coverage=coverage,**scan_table))
    r.refresh(x,common_witnesses([r.s0,s],r.geometry),'actual')
    if initial_matches is not None:
        r.ordered.matches=initial_matches
    parent=r.full(r.x,'parent_with_actual_structure')
    if initial_matches is not None:
        changed_ordered(r,x,initial_matches)
    final=r.full(x,'candidate_with_actual_structure')
    guard=evaluate_guard(r,parent,final)
    raw=summary(r,parent,final)
    new_pairs={e:sorted(set(scan_table['candidate'][e]['pairs'])-set(p['pairs']))
               for e,p in scan_table['parent'].items()}
    if any(new_pairs.values()):guard['failures'].append('new_semantic_surface_pair')
    if final['weighted']['surface_intersection']>parent['weighted']['surface_intersection']*1.01+1e-10:
        guard['failures'].append('refreshed_actual_surface_response')
    status='review_only' if guard['failures'] else 'full_replay_engineering_step'
    candidate=checkpoint(r,x,status)
    check=ProgrammeSkateSourceFactors(r.problem,r.ev,role='check')
    check.ordered.associate(r.x[17:])
    source_comparison={label:dict(fit={k:v['value']['action_metrics'][k] for k in SOURCES},
                                  check=check.evaluate(z[17:],False)['metrics'])
        for label,z,v in [('parent',r.x,parent),('candidate',x,final)]}
    write(output/'source_comparison.json',source_comparison)
    write(output/'acceptance.json',dict(status=status,user_accepted=False,accuracy_accepted=False,
        root_and_lifted_rotation_preserved=True,camera_sha256=r.info['state_sha256'],
        source_evidence=str(r.source_path),source_evidence_sha256=digest(r.source_path),
        parameter_sha256=digest(candidate),parent_sha256=digest(r.args.latest),
        candidate_surface=str(sfile),candidate_surface_sha256=digest(sfile),
        coverage=final['coverage'],surface_coverage=coverage,guard=guard,
        new_semantic_surface_pairs=new_pairs,fit_raw_source_px=raw,selected=selected,
        production=dict(parent=serial(parent),candidate=serial(final))))
    write(sfile.with_suffix('.json'),dict(parameter_sha256=digest(candidate),camera_sha256=r.info['state_sha256'],
                                      input_sha256=digest(r.args.prepared/'input.npz'),coverage=coverage))
    r.emit('candidate_saved',status=status,failures=guard['failures'],path=str(candidate),source=raw,
           before=parent['cost'],after=final['cost'])


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','native-review','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.previous=a.latest;a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf8',buffering=1) as log:
        r=SkateReplay(a,log)
        run(r)


if __name__=='__main__':
    main()
