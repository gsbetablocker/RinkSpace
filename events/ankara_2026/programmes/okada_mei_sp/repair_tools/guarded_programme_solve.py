from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr

from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
from calibration.program_tracking.body_tracking.programme_skate_evidence import ProgrammeSkateSourceFactors
from calibration.program_tracking.body_tracking.programme_skate_replay import (
    SkateReplay, common_witnesses, refresh_surface, save, surface_events, write)
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from scripts.refit_programme_skates import SOURCES, evaluate_guard, serial


def block_family(name):
    if name.startswith('episode/'):
        name=name.rsplit('/',1)[-1]
    if name in SOURCES or name in ('full_source_skate_structure','source_pixel_edge_profile',
                                   'complete_runner_extent','complete_visible_structure_extent'):
        return 'source'
    if name.startswith('flight_') or name in ('programme_parent_root_trust','rotation_direction_path',
                                                'rotation_path_impulse','source_facing_cone','body_pixels'):
        return 'body_and_trajectory'
    if name in ('finite_ice','cavity_AL','dynamic_foot_compatibility','surface_intersection',
                'anatomical_joint_domain','source_supported_grasp','episode_support',
                'signed_joint_relation','self_collision'):
        return 'structure'
    return 'other'


def active_columns(r,J):
    state=r.problem.body.state
    origin=r.clock[0]
    # Only material leg-chain columns observed in the actual source Jacobian.
    possible=np.unique(J.indices)
    body=possible[(possible>=17)&(possible<17+state.body_size)]-17
    shoe=possible[possible>=17+state.body_size]-17-state.body_size
    def protected(k,knots):
        t=knots[k]+origin;end=knots[k+4]+origin
        return (t<=79.02)&(end>=77.58)
    axes=np.r_[6:12,15:21,24:30]
    body=body[np.isin(body%69,axes)&~protected(body//69,state.knots)]
    shoe=shoe[~protected(shoe//6,state.shoe_curve.knots)]
    return np.r_[17+body,17+state.body_size+shoe]


def solve_directions(r,baseline):
    blocks=baseline['value']['weighted_blocks']
    ordered=next(b for b in blocks if b.name=='ordered_native_skate')
    active=active_columns(r,ordered.jacobian)
    if not len(active):raise ValueError('The admitted steel and shoe rows have no active leg controls')
    index={row['raw_group_id']:row for row in r.ordered.rows}
    metrics=baseline['value']['action_metrics']['ordered_native_skate']
    semantics=[];events=[]
    for m in metrics:
        if m['status']!='evaluated':continue
        row=index[m['id']]
        n=2*(len(m['distance_px']))
        semantics.extend([m['semantic']]*n);events.extend([row['event_id']]*n)
    assert len(semantics)==len(ordered.residual)
    steel=np.isin(semantics,['runner_lower_edge','runner_upper_edge'])
    zero=np.zeros_like(r.x)
    groups=defaultdict(lambda:defaultdict(float))
    direction={}
    for name,source_mult,steel_mult in [('balanced',1.,1.),('steel',2.,4.)]:
        rows=[];values=[];pieces=[]
        for b in blocks:
            part=b.jacobian[:,active].tocsr()
            if not part.nnz:continue
            relevant=np.diff(part.indptr)>0
            part=part[relevant]
            residue=b.residual[relevant]
            wt=1.
            if b.name=='ordered_native_skate':
                wt=np.where(steel[relevant],steel_mult,source_mult)
                part=part.multiply(wt[:,None]).tocsr()
            rows.append(part);values.append(wt*residue)
            pieces.append((b.name,len(residue),relevant))
        J=vstack(rows,format='csr')
        residual=np.concatenate(values)
        norms=np.sqrt(np.asarray(J.power(2).sum(0)).ravel())
        scales=1/np.maximum(norms,1.)
        scaled=J@diags(scales)
        result=lsmr(scaled,-residual,damp=.18,atol=5e-5,btol=5e-5,maxiter=120)
        dx=result[0]*scales
        limit=.20 if name=='steel' else .16
        mx=max(np.linalg.norm(np.bincount((active[active<17+r.problem.body.state.body_size]-17)//3,
                weights=dx[active<17+r.problem.body.state.body_size]**2)**.5,ord=np.inf),np.max(abs(dx)))
        dx*=min(1.,limit/max(mx,1e-12))
        d=zero.copy();d[active]=dx
        direction[name]=d
        for b in blocks:
            y=b.jacobian[:,active]@dx
            group=groups[name][block_family(b.name)]
            groups[name][block_family(b.name)]=group+float(b.residual@y)
        dr=ordered.jacobian[:,active]@dx
        per_event={}
        for e in dict.fromkeys(events):
            region=np.array(events)==e
            if region.any():
                per_event[e]=dict(slope=float(ordered.residual[region]@dr[region]),
                    steel_slope=float(ordered.residual[region&steel]@dr[region&steel]))
        r.emit('actual_linear_direction',name=name,columns=len(active),rows=J.shape[0],
            nonzeros=J.nnz,solver_iterations=int(result[2]),max_radian=float(np.max(abs(dx))),
            source_linear_slope=float(ordered.residual@dr),production_group_slopes=dict(groups[name]),
            source_by_event=per_event)
    write(r.args.output/'directions.json',dict(production_group_slopes=groups,control_count=len(active),
        actual_ordered_source_rows=len(ordered.residual), steel_coordinate_rows=int(steel.sum()),
        root_and_shared_geometry_fixed=True))
    save(r.args.output/'directions.npz',x=r.x,active_columns=active,**direction)
    return direction,ordered


def numeric_derivative(r,ordered,d):
    x=r.x[17:];h=2e-5
    f=r.ordered
    # The production block is scaled by the declared family weight. Compare
    # the exact same fixed association and projector before changing any match.
    w=np.sqrt(float(r.problem.objective_block_weights[ordered.name]))
    b=f.evaluate(x+h*d[17:],False)['residual']
    a=f.evaluate(x-h*d[17:],False)['residual']
    measured=w*(b-a)/(2*h)
    exact=ordered.jacobian@d
    report=dict(rows=len(exact), norm=float(np.linalg.norm(exact)),
        relative_difference=float(np.linalg.norm(exact-measured)/max(1e-9,np.linalg.norm(exact),np.linalg.norm(measured))))
    write(r.args.output/'actual_source_derivative.json',report);r.emit('actual_source_derivative',**report)
    if report['norm']<=0 or report['relative_difference']>.03:
        raise ValueError('Actual source Jacobian did not pass the direction check')


def summary(r,before,after):
    m0=before['value']['action_metrics']['ordered_native_skate']
    m1=after['value']['action_metrics']['ordered_native_skate']
    family=defaultdict(list)
    for a,b in zip(m0,m1,strict=True):
        assert a['id']==b['id']
        if 'distance_px' in a and 'distance_px' in b:
            key=a['semantic']
            family[key].append([float(np.percentile(q['distance_px'],95)) for q in (a,b)])
    return {key:dict(rows=len(pair),median_before_after=np.median(pair,axis=0).tolist(),
                    p95_before_after=np.percentile(pair,95,axis=0).tolist())
            for key,pair in family.items()}


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


from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem


def frozen_columns(r,columns,intervals):
    state=r.problem.body.state
    body=columns<17+state.body_size
    start=np.empty(len(columns));end=start.copy()
    for subset,offset,shape,knots in ((body,17,69,state.knots),
        (~body,17+state.body_size,6,state.shoe_curve.knots)):
        k=(columns[subset]-offset)//shape
        start[subset]=knots[k]+r.clock[0];end[subset]=knots[k+4]+r.clock[0]
    fixed=np.zeros(len(columns),bool)
    for a,b in intervals:fixed|=(start<=b)&(end>=a)
    return fixed


def limit_step(r,d):
    n=r.problem.body.state.body_size
    for block,maximum in ((d[17:17+n].reshape(-1,23,3),.36),
                          (d[17+n:].reshape(-1,2,3),.24)):
        length=np.linalg.norm(block,axis=-1)
        block*=np.minimum(1.,maximum/np.maximum(length,1e-12))[...,None]
    return d


def local_direction(r,old):
    p=r.problem;state=p.body.state
    native=r.ordered.evaluate(r.x[17:],True)
    source=native['weighted_blocks'][0]
    cols=active_columns(r,source.jacobian)
    scale=r.tb[0]/r.tb[1]
    frozen=[(77.58,79.02),(156.30,166.00)]
    frozen.extend((e['start_pts']*scale-.02,e['end_pts_inclusive']*scale+.02)
        for e in r.document['source_episode_review']['episodes'] if e['kind']=='flight_visible_candidate')
    cols=cols[~frozen_columns(r,cols,frozen)]
    steel=np.concatenate([np.full(2*len(m['distance_px']),m['semantic'].startswith('runner_'),dtype=bool)
                          for m in native['metrics'][source.name] if 'distance_px' in m])
    assert len(steel)==len(source.residual)
    u=np.where(steel,6.,2.)
    Jsource=source.jacobian[:,cols].multiply(u[:,None]).tocsr()
    q=np.zeros(len(r.clock),bool)
    for t in {row['pts'] for row in r.ordered.rows}:
        j=np.searchsorted(r.pts,t)
        q[max(0,j-11):min(j+12,len(q))]=True
    frames=np.flatnonzero(q)[::2]
    body=FullBodyProblem(r.ref,r.camera,r.observations_path,tuple(r.clock[[0,-1]]),'full',
        state_factory=lambda *unused:state,source_frames=frames,execution=r.execution)
    b=body.evaluate(r.x[17:],True)['weighted_blocks'][0]
    use=np.repeat(np.isin(body.oj,[11,12,13,14,15,16]),2)
    Jbody=b.jacobian[use][:,cols-17]*.10
    warm=np.zeros_like(old);warm[cols]=old[cols]
    regular=np.where(cols<17+state.body_size,.55,.80)
    J=vstack((Jsource,Jbody,diags(regular)),format='csr')
    v=np.r_[source.residual*u,b.residual[use]*.10,-regular*warm[cols]]
    diag=1/np.maximum(np.sqrt(np.asarray(J.power(2).sum(0)).ravel()),.5)
    linear=lsmr(J@diags(diag),-v,damp=.08,atol=2e-5,btol=2e-5,maxiter=160)
    d=np.zeros_like(old);d[cols]=linear[0]*diag
    d=limit_step(r,d)
    r.emit('scoped_attached_source_step',controls=len(cols),native_body_frames=len(frames),
        J_source_nnz=Jsource.nnz,solver_iterations=int(linear[2]),
        raw_steel_slope=float(source.residual[steel]@(source.jacobian@d)[steel]),frozen_source_seconds=frozen)
    return d,limit_step(r,2.4*warm),source


def source_merit(r,baseline,d,source):
    for _ in range(3):
        value=r.ordered.evaluate(r.x[17:]+d[17:],False)
        bad=[]
        for before,after in zip(baseline,value['metrics'][source.name],strict=True):
            if 'distance_px' not in before or 'distance_px' not in after:continue
            a,b=[float(np.percentile(row['distance_px'],95)) for row in (before,after)]
            s=before['radius_px'];t=before['pts']*r.tb[0]/r.tb[1]
            if b>max(a,s)+.40*s:bad.append((t-.022,t+.022))
        if not bad:break
        active=np.flatnonzero(d)
        d[active[frozen_columns(r,active,bad)]]*=.40
    value=r.ordered.evaluate(r.x[17:]+d[17:],False)
    return summary(r,dict(value=dict(action_metrics={source.name:baseline})),
        dict(value=dict(action_metrics={source.name:value['metrics'][source.name]})))


def run_guarded(r):
    output=r.args.output
    write(output/'objective_weights.json',r.weights_record())
    parent=r.full(r.x,'full_parent_for_guarded_trials')
    r.emit('full_parent_ready',seconds=parent['elapsed_s'],cost=parent['cost'],weighted=parent['weighted'])
    write(output/'production_registry.json',parent['registry'])
    with np.load(r.args.proposal_cache,allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['x'],r.x)
        d,prior,source=local_direction(r,cache['steel'])
    base=parent['value']['action_metrics']['ordered_native_skate']
    proposed=[]
    for name,raw in [('mixed',.72*d+.78*prior),('steel_local',d),('production_warm',prior)]:
        step=limit_step(r,raw)
        merit=source_merit(r,base,step,source)
        r.emit('scoped_source_projection_check',name=name,merit=merit,nonzero_controls=int(np.count_nonzero(step)))
        proposed.append((merit['runner_lower_edge']['p95_before_after'][1],name,step))
    proposed.sort(key=lambda q:q[0])
    # Reuse the exact source Jacobian just evaluated for the proposal.
    x=r.x[17:];h=2e-5;move=proposed[0][2]
    plus=r.ordered.evaluate(x+h*move[17:],False)['residual']
    minus=r.ordered.evaluate(x-h*move[17:],False)['residual']
    diff=(plus-minus)/(2*h);exact=source.jacobian@move
    report=dict(rows=len(exact),norm=float(np.linalg.norm(exact)),relative_difference=float(
        np.linalg.norm(exact-diff)/max(1e-9,np.linalg.norm(exact),np.linalg.norm(diff))))
    write(output/'actual_source_derivative.json',report);r.emit('actual_source_derivative',**report)
    if report['norm']<=0 or report['relative_difference']>.03:raise ValueError('Guarded source derivative failed')
    save(output/'guarded_directions.npz',x=r.x,**{name:step for _,name,step in proposed})
    reports=[];selected=None;fallback=None
    for _,name,move in proposed:
        if not np.any(move):continue
        for factor in (1.,.62):
            x=r.x+move*factor
            trial=r.full(x,f'{name}_{factor}',verify_rows=False)
            guard=evaluate_guard(r,parent,trial)
            raw=summary(r,parent,trial);steel=raw['runner_lower_edge']['p95_before_after']
            report=dict(direction=name,fraction=factor,failures=guard['failures'],raw_source=raw,
                cost=trial['cost'],elapsed_s=trial['elapsed_s'])
            reports.append(report);r.emit('whole_objective_done',weighted=trial['weighted'],**report)
            if trial['cost']<parent['cost'] and steel[1]<steel[0]:
                if not guard['failures']:
                    selected=x,report;break
                rank=(len(guard['failures']),steel[1])
                if fallback is None or rank<fallback[0]:fallback=(rank,x,report)
        if selected is not None:break
    write(output/'proposal_trials.json',reports)
    if selected is None:
        if fallback is None:raise ValueError('No guarded proposal improved both the full objective and the steel tail')
        _,x,report=fallback
    else:x,report=selected
    verify_candidate(r,x,report)


def verify_candidate(r,x,selected,scan_seed=None):
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
    parent=r.full(r.x,'parent_with_actual_structure')
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
    for k in ('project','prepared','source','latest','latest-surface','native-review','proposal-cache','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.previous=a.latest;a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf8',buffering=1) as f:
        r=SkateReplay(a,f)
        run_guarded(r)

if __name__=='__main__':main()
