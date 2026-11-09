"""Restore the verified supports of the particular torso/wear events that failed."""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np

from calibration.program_tracking.body_tracking.inter_exposure_runner_guard import InterExposureRunnerGuard
from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay,load_scan,save,write
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from .chain_verify import verify_candidate
from .replay_shoe_pressure import exact_source,project_trial
from .resume_chain import anatomy,guard_trial
from scripts.refit_programme_skates import evaluate_guard
from .solve_tracked_programme import measured_sources,scopes,statistics
from .stabilize_two_skates import guard_visibility


def run(r):
    out=r.args.output
    full=r.args.attempt
    fields=load(full/'screened_both_traces.npz')
    trial=json.loads((full/'trial_both_traces.json').read_text(encoding='utf8'))
    broken={v.split(':',1)[-1] for v in trial['failures'] if v.startswith(('torso:','cavity_tail:'))}
    if not broken:
        raise ValueError('No observed physical-event failure to restore')
    x0=r.x
    safe=load(r.args.safe/'candidate.npz')['x']
    possible=(fields['x']!=x0)|(safe!=x0)
    col=np.flatnonzero(possible)
    lo,hi,_=scopes(r,col)
    restore=np.zeros(len(col),bool)
    for e in r.ev['events']:
        if e['id'] in broken:
            low,high=(e[k]*r.tb[0]/r.tb[1] for k in ('start_pts','end_pts'))
            restore|=(lo<=high+.012)&(hi>=low-.012)
    if not restore.any():
        raise ValueError('No spline controls contribute to the flagged events')
    f=exact_source(r);f0=f.matches
    initial=r.ordered.matches
    _,cost0,_=project_trial(f,x0,x0,f0)
    context=dict(source=measured_sources(r,x0),anatomy=anatomy(r,x0),
                 pixel_cost=cost0,source_spans=np.zeros(len(f.rows)))
    # This guard consumes model geometry only; fit/check observations are
    # never selected or changed on the basis of the held-out pixel values.
    zone=np.flatnonzero((r.clock>=79.02)&(r.clock<=95.))
    vis=InterExposureRunnerGuard(r,zone)
    r.emit('restore_failed_event_support',events=sorted(broken),columns=int(restore.sum()))
    trials=[]
    for label,target in [('restored_verified_events',safe),('restored_parent_events',x0)]:
        candidate=fields['x'].copy()
        candidate[col[restore]]=target[col[restore]]
        candidate=guard_visibility(r,vis,candidate,label)
        if candidate is None:
            continue
        v=guard_trial(r,f,initial,f0,context,candidate,label)
        if v is None:
            continue
        x,score=v
        if vis.newly_hidden(x):
            continue
        trials.append((label,x,score))
    if not trials:
        r.emit('no_visibility_safe_event_recovery')
        return
    write(out/'objective_weights.json',r.weights_record())
    r.ordered.matches=initial
    base=r.full(x0,'restored_event_parent')
    for label,x,score in trials:
        r.ordered.associate_changed(x[17:],x0[17:],initial)
        final=r.full(x,label,verify_rows=False)
        g=evaluate_guard(r,base,final)
        outrow=dict(label=label,retired_events=sorted(broken),failures=g['failures'],
                    source_rows=g['source_rows'],raw=statistics(r,base,final),source_cost=[cost0,score],
                    full_cost=[base['cost'],final['cost']])
        r.emit('recovered_event_full_check',**outrow)
        write(out/('trial_'+label+'.json'),outrow)
        if g['failures']:
            continue
        fresh=json.loads((r.args.safe/'acceptance.json').read_text(encoding='utf8'))
        packet=load_scan(fresh['candidate_surface'],r.args.safe/'candidate.npz',r.info['state_sha256'],r.args.prepared)
        cols=load(r.args.archive)['columns']
        root=cols[(cols<17+r.problem.body.state.body_size)&((cols-17)%69<3)]
        verify_candidate(r,x,outrow,scan_seed=(packet,safe),initial_matches=initial,allowed_root=root)
        return


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','output','archive','attempt','safe','native-review'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.previous=a.latest;a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(a,log)
        run(r)


if __name__=='__main__':
    main()
