"""Stabilize the two-foot continuous proposal before complete source and structure replay."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.inter_exposure_runner_guard import InterExposureRunnerGuard
from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay,load_scan,save,write
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_chain_proposal import preserve_free_skate
from .chain_verify import verify_candidate
from .replay_shoe_pressure import exact_source,project_trial,residual_summary
from .resume_chain import anatomy,guard_trial,solve_local
from scripts.refit_programme_skates import evaluate_guard
from .solve_tracked_programme import measured_sources,scopes,statistics


def guard_visibility(r,visibility,x,tag):
    previous=None
    d=x-r.x
    while True:
        hidden=visibility.newly_hidden(r.x+d)
        r.emit('inter_exposure_visibility',tag=tag,new_fragments=len(hidden),
               examples=hidden[:12])
        if not hidden:
            return r.x+d
        columns=np.flatnonzero(d)
        lo,hi,feet=scopes(r,columns)
        root=(columns-17)%69<3
        retire=np.zeros(len(columns),bool)
        for h in hidden:
            t=h['pts']*r.tb[0]/r.tb[1]
            retire|=(lo<=t+.012)&(hi>=t-.012)&(~root)&(feet==h['foot'])
        if not retire.any() or len(hidden)==previous:
            for h in hidden:
                t=h['pts']*r.tb[0]/r.tb[1]
                retire|=(lo<=t+.012)&(hi>=t-.012)&root
        if not retire.any():
            return None
        previous=len(hidden)
        d[columns[retire]]=0.


def targets(r,cols,foot,pts):
    lo,hi,side=scopes(r,cols)
    axis=(cols-17)%69
    chosen=np.zeros(len(cols),bool)
    for f,t in zip(foot,pts,strict=True):
        stamp=t*r.tb[0]/r.tb[1]
        chosen|=(lo<=stamp+.018)&(hi>=stamp-.018)&((axis<3)|(side==f))
    return chosen


def run(r):
    out=r.args.output
    arc=load(r.args.archive);x0=r.x
    np.testing.assert_array_equal(arc['x'],x0)
    all_cols=arc['columns'];N=r.problem.body.state.body_size
    eligible=all_cols<17+N
    J=csr_matrix((arc['data'],arc['indices'],arc['indptr']),shape=tuple(arc['shape']))
    seed=load(r.args.seed/'candidate.npz')['x']
    target_doc=json.loads(r.args.native_review.read_text(encoding='utf8'))
    direct=target_doc['ordered_contours']
    phase=eligible & targets(r,all_cols,[e['foot'] for e in direct],[e['pts'] for e in direct])
    cols=all_cols[phase]
    A=J[:,phase]
    b=arc['residual']+J@(seed[all_cols]-x0[all_cols])
    base_matches=r.ordered.matches
    f=exact_source(r);f0=f.matches
    v0,cost0,_=project_trial(f,x0,x0,f0)
    context=dict(source=measured_sources(r,x0),anatomy=anatomy(r,x0),
                 pixel_cost=cost0,source_spans=np.zeros(len(f.rows)))
    fresh=seed.copy()
    for pressure in (192.,640.):
        value,pixels,missing=project_trial(f,x0,fresh,f0,jacobian=True)
        move=solve_local(A,b,value['jac'],value['residual'],fresh[cols]-seed[cols],pressure,cols,N,r.problem.binding.ratio,True)
        for scale in (1.,.6,.3):
            candidate=fresh.copy()
            candidate[cols]+=scale*move
            result=guard_trial(r,f,base_matches,f0,context,candidate,f'both_{pressure:g}_{scale:g}')
            if result and result[1]<pixels:
                fresh=result[0]
                break
    # Source-identified grasp interval. Preserve the free-skate continuation
    # only in gaps where it has no direct fit trace nearby.
    r.emit('guard_unobserved_blade_start',added_visible_source_rows=len(direct))
    zone=np.flatnonzero((r.clock>=79.02)&(r.clock<=95.))
    vis=InterExposureRunnerGuard(r,zone)
    grasp=np.zeros(len(r.pts),bool)
    for e in r.document['source_episode_review']['episodes']:
        if e['kind']=='hand_to_raised_shoe_grasp_candidate' and e['foot']==0:
            grasp|=(r.pts>=e['start_pts'])&(r.pts<=e['end_pts_inclusive'])
    for r0 in f.rows:
        if r0['foot']==0:
            grasp&=abs(r.pts-r0['pts'])>.16*12800
    lo,hi,ft=scopes(r,all_cols)
    root=(all_cols-17)%69<3
    pose0=eligible&(~root)&(ft==0)
    with_prior=np.flatnonzero(grasp)
    # Existing spline phase only. Drop columns outside the original full-J
    # support and keep the dominant corrections of the traced support foot.
    membership=np.any((lo[:,None]<=r.clock[with_prior])&(hi[:,None]>=r.clock[with_prior]),axis=1)
    other=all_cols[pose0&membership]
    hold=preserve_free_skate(r,fresh,other,with_prior,0)
    proposals=[]
    for variant in ('stabilized','both_traces','verified_seed'):
        candidate=fresh.copy() if variant!='verified_seed' else seed.copy()
        if variant=='stabilized':
            candidate[other]+=hold
        candidate=guard_visibility(r,vis,candidate,variant)
        if candidate is None:
            continue
        accept=guard_trial(r,f,base_matches,f0,context,candidate,variant)
        if accept is None:
            continue
        x,pixels=accept
        # A support retired by the original source guard can alter a
        # nearby visibility branch. Recheck after all modifications.
        if vis.newly_hidden(x):
            continue
        proposals.append((pixels,variant,x))
    if not proposals:
        r.emit('no_visibility_safe_proposal')
        return
    proposals.sort(key=lambda p:p[0])
    write(out/'objective_weights.json',r.weights_record())
    r.ordered.matches=base_matches
    before=r.full(x0,'two_skates_parent')
    for score,name,x in proposals:
        r.ordered.associate_changed(x[17:],x0[17:],base_matches)
        after=r.full(x,name,verify_rows=False)
        g=evaluate_guard(r,before,after)
        receipt=dict(label=name,source_cost=[cost0,score],full_cost=[before['cost'],after['cost']],
                     failures=g['failures'],source_rows=g['source_rows'],raw=statistics(r,before,after),
                     body=[before['body'],after['body']],torso=[before['torso_by_event'],after['torso_by_event']])
        r.emit('two_skates_full_replay',**receipt)
        write(out/('trial_'+name+'.json'),receipt)
        if g['failures']:
            continue
        saved=json.loads((r.args.seed/'acceptance.json').read_text(encoding='utf8'))
        scan=load_scan(saved['candidate_surface'],r.args.seed/'candidate.npz',r.info['state_sha256'],r.args.prepared)
        verify_candidate(r,x,receipt,scan_seed=(scan,seed),initial_matches=base_matches,
                         allowed_root=all_cols[eligible&root])
        return


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','output','archive','seed','native-review'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.previous=a.latest;a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(a,log)
        run(r)


if __name__=='__main__':
    main()
