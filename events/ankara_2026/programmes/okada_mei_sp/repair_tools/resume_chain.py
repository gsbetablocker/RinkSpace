"""Build, screen, and replay continuous ankle-attached chain proposals."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay,save,write
from calibration.program_tracking.body_tracking.skate_chain_proposal import profiled_leg_proposal
from scripts.refit_programme_skates import evaluate_guard,source_guard
from .solve_tracked_programme import measured_sources,scopes,statistics
from .replay_shoe_pressure import exact_source,project_trial,residual_summary
from .chain_verify import verify_candidate


def anatomy(r,x):
    joints=r.joints
    return joints.values(joints.state.increments(x[17:],joints.times))[0]


def native_score(value, factor, baseline_spans):
    metrics=value['metrics'][factor.name]
    return sum(int('model_uv' in m and np.linalg.norm(np.diff(np.asarray(m['model_uv'])[[0,-1]],axis=0)) < .58*span)
        for m,span in zip(metrics,baseline_spans,strict=True) if span>0.)


def cut_columns(r,delta,src,excursions):
    idx=np.flatnonzero(delta)
    lo,hi,foot=scopes(r,idx)
    a=(idx-17)%69
    root=a<3
    discard=np.zeros(len(idx),bool)
    for v in src:
        time=v['pts']*r.tb[0]/r.tb[1]
        discard|=(lo<=time+.015)&(hi>=time-.015)&(root|(foot==v['group'][1]))
    for frame,joint,component in np.argwhere(excursions):
        j=r.joints.joints[joint]
        axis=6+3*(j-1)
        at=r.clock[frame]
        discard|=(lo<=at)&(hi>=at)&(a>=axis)&(a<axis+3)
    delta[idx[discard]]=0.
    return int(discard.sum())


def guard_trial(r,f,initial,fassoc,context,x,tag):
    delta=x-r.x
    failures=[]
    while np.any(delta):
        x=r.x+delta
        r.ordered.associate_changed(x[17:],r.x[17:],initial)
        actual=measured_sources(r,x)
        failures=source_guard(r,context['source'],actual)
        v=anatomy(r,x)
        exc=(context['anatomy']<.25)&(v>.5)&(v-context['anatomy']>.25)
        if not failures and not exc.any():
            value,cost,missing=project_trial(f,r.x,x,fassoc)
            # Real visibility as determined by the model, alongside the
            # source rows. Do not reward a contour disappearing in a trial.
            fragments=native_score(value,f,context['source_spans'])
            if not missing and not fragments and cost < context['pixel_cost']:
                r.emit('source_and_anatomy_trial',tag=tag,passed=True,cost=cost,
                    by_part=residual_summary(f,value),source_conflicts=0,new_joint_excursions=0)
                save(r.args.output/('screened_'+tag+'.npz'),x=x)
                return x,cost
            r.emit('source_and_anatomy_trial',tag=tag,passed=False,cost=cost,
                   missing_paths=missing,new_fragments=fragments)
            return None
        count=cut_columns(r,delta,failures,exc)
        r.emit('retired_conflicting_supports',tag=tag,columns=count,source_conflicts=len(failures),
               new_joint_excursions=int(exc.sum()),first=[q['pts']/12800 for q in failures[:6]])
        if count==0:
            break
    return None


def solve_local(J,residual,S,src,d,pressure,columns,body_size,ratio,root_enabled):
    A=J; B=S[:,columns]
    M=(A.T@A+pressure*(B.T@B)).tocsc()
    p=(columns-17)%69
    root=p<3
    diagonal=.025*np.maximum(M.diagonal(),.001)
    # Per-sample source resolves only the observable depth. For proposal
    # damping, condition translation and rotation in physical task units.
    diagonal[root] += .20*(ratio/.08)**2
    g=A.T@(residual+A@d)+pressure*(B.T@src)
    step=spsolve(M+diags(diagonal,format='csc'),-g)
    if not root_enabled:
        step[root]=0.
    cap=np.where(root,.10/ratio,.38)
    size=float(np.max(np.abs(step)/cap))
    return step/max(1.,size)


def run(r,archive):
    out=r.args.output
    x0=r.x
    matrix=np.load(archive,allow_pickle=False)
    np.testing.assert_array_equal(matrix['x'],x0)
    Jall=csr_matrix((matrix['data'],matrix['indices'],matrix['indptr']),shape=tuple(matrix['shape']))
    all_cols=matrix['columns']
    b=matrix['residual']
    n=r.problem.body.state.body_size
    initial=r.ordered.matches
    f=exact_source(r)
    f0=f.matches
    raw,cost,_=project_trial(f,x0,x0,f0,jacobian=True)
    phase=(all_cols<17+n)
    # A source-driven move must have actual influence on an observed foot,
    # not merely reside in a wide episode. This keeps the other leg from
    # changing to buy unrelated objective descent.
    phase &= np.asarray(raw['jac'][:,all_cols].getnnz(axis=0)).ravel()>0
    columns=all_cols[phase]
    J=Jall[:,phase].tocsr()
    base_metrics=measured_sources(r,x0)
    native_spans=[]
    for m in raw['metrics'][f.name]:
        span=np.linalg.norm(np.asarray(m['model_uv'])[-1]-np.asarray(m['model_uv'])[0]) if 'model_uv' in m else 0.
        native_spans.append(span if m['semantic']=='runner_lower_edge' and m['complete_visible'] else 0.)
    context=dict(source=base_metrics,anatomy=anatomy(r,x0),pixel_cost=cost,source_spans=native_spans)
    r.emit('reachable_chain_start',fit_contours=len(f.rows),initial_cost=cost,columns=len(columns),
           pelvis_columns=int(np.sum((columns-17)%69<3)),native_jacobian_rows=J.shape[0])
    semantics={}
    for q in f.rows:
        semantics.setdefault((q['pts'],q['foot']),set()).add(q['semantic'])
    f.matches=f0
    tasks=[p for p in f.pose_task_probe(x0[17:]) if 'runner_lower_edge' in semantics[p['pts'],p['foot']]
           and len(semantics[p['pts'],p['foot']])>=2 and p['weighted_rank']>=3]
    proposals=[]
    root_index=(columns-17)%69<3
    for permit in (True,):
        active=np.ones(len(columns),bool) if permit else ~root_index
        cols=columns[active]
        A=J[:,active].tocsr()
        seeds=[('saved',np.zeros(len(cols)))]
        if tasks and r.args.include_profile:
            try:
                seed,receipt=profiled_leg_proposal(r,tasks,cols,permit_root=permit,fraction=.8)
                r.emit('source_depth_profile',**receipt)
                seeds.append(('ray_ik',seed))
            except ValueError as error:
                r.emit('source_depth_profile_failed',reason=str(error),root_enabled=permit)
        for origin,d in seeds:
            xt=x0.copy()
            xt[cols]+=d
            if np.any(d):
                for scale in (.75,.35):
                    z=x0.copy();z[cols]+=scale*d
                    checked=guard_trial(r,f,initial,f0,context,z,f'{origin}_{permit}_{scale}')
                    if checked:
                        proposals.append((checked[1],checked[0],f'{origin}_{permit}_{scale}'))
                if not any(tag.startswith(f'{origin}_{permit}_') for _,_,tag in proposals):
                    xt=x0.copy()
            for pressure in (80.,384.,1536.):
                current=xt
                for iteration in range(2):
                    value,now,missing=project_trial(f,x0,current,f0,jacobian=True)
                    if missing:
                        current=x0.copy()
                        value,now,_=project_trial(f,x0,current,f0,jacobian=True)
                    s=solve_local(A,b,value['jac'],value['residual'],current[cols]-x0[cols],
                        pressure,cols,n,r.problem.binding.ratio,permit)
                    best=None
                    for alpha in (1.,.5,.25):
                        z=current.copy();z[cols]+=alpha*s
                        tag=f'{origin}_{permit}_{pressure}_{iteration}_{alpha}'
                        checked=guard_trial(r,f,initial,f0,context,z,tag)
                        if checked and checked[1]<now:
                            proposals.append((checked[1],checked[0],tag))
                            best=checked[0]
                            break
                    if best is None:
                        break
                    current=best
                # A fully screened branch seeds the next pressure level.
                xt=current
    if not proposals:
        raise ValueError('No attached proposal improved true source with pointwise checks')
    proposals.sort(key=lambda q:q[0])
    write(out/'objective_weights.json',r.weights_record())
    # Preserve useful alternatives without replaying duplicate directions.
    choices=[]
    for score,x,label in proposals:
        has_root=bool(np.any(x[columns[root_index]]!=x0[columns[root_index]]))
        delta=x[columns]-x0[columns]
        pred=J@delta
        estimate=float(b@pred+.5*(pred@pred))
        if not choices or (estimate<0 and not any(q[4]<0 for q in choices)):
            choices.append((score,x,label,has_root,estimate))
    save(out/'proposals.npz',x=x0,**{label:x for _,x,label,_,_ in choices})
    r.ordered.matches=initial
    before=r.full(x0,'protected_chain_parent')
    for score,x,label,root,predicted_delta in choices:
        r.ordered.associate_changed(x[17:],x0[17:],initial)
        after=r.full(x,label,verify_rows=False)
        guard=evaluate_guard(r,before,after)
        report=dict(label=label,root_motion=root,predicted_whole_delta=predicted_delta,source_cost=[cost,score],
                    whole_cost=[before['cost'],after['cost']],failures=guard['failures'],
                    source_rows=guard['source_rows'],raw=statistics(r,before,after),
                    body=[before['body'],after['body']],torso=[before['torso_by_event'],after['torso_by_event']])
        r.emit('complete_chain_proposal',**report)
        write(out/('trial_'+label+'.json'),report)
        if not guard['failures']:
            verify_candidate(r,x,report,scan_seed=(r.s0,x0),initial_matches=initial,allowed_root=columns[root_index])
            return
    raise ValueError('All source-led chain directions failed the complete replay')


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','output','archive'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--include-profile',action='store_true')
    a=p.parse_args()
    a.previous=a.latest
    a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(a,log)
        run(r,a.archive)


if __name__=='__main__':
    main()
