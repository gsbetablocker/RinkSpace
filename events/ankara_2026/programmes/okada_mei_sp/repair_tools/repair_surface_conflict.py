"""Remove only the continuous controls touching a measured new collision."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay, load_json, write
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from .guarded_programme_solve import frozen_columns, verify_candidate


def run(r,previous):
    old=load_json(previous/'acceptance.json')
    scan_file=previous/'surface_actual.npz'
    x_old=load(previous/'candidate.npz')['x']
    if (old['parent_sha256']!=digest(r.args.latest)
            or old['parameter_sha256']!=digest(previous/'candidate.npz')
            or old['candidate_surface_sha256']!=digest(scan_file)):
        raise ValueError('Saved full candidate or surface receipt changed')
    if old['guard']['failures']!=['new_semantic_surface_pair']:
        raise ValueError('Unexpected failure in the preceding full replay')
    packet=load(scan_file)
    times=packet['times']+r.clock[0];counts=packet['pair_counts']
    parts=packet['parts'].tolist()
    stamp=[]
    scopes={e['id']:e for e in r.ev['events']}
    for name,pairs in old['new_semantic_surface_pairs'].items():
        if not pairs:continue
        e=scopes[name]
        bound=np.array([e['start_pts'],e['end_pts']])*r.tb[0]/r.tb[1]
        for p in pairs:
            a,b=sorted(parts.index(k) for k in p.split('/'))
            support=times[counts[(counts[:,1]==a)&(counts[:,2]==b),0]]
            stamp.extend(support[(support>=bound[0])&(support<=bound[1])].tolist())
    if not stamp:raise ValueError('Failed to locate the actual new model intersection')
    active=np.flatnonzero(x_old!=r.x)
    keep=frozen_columns(r,active,[(t-.018,t+.018) for t in stamp])
    x=x_old.copy();x[active[keep]]=r.x[active[keep]]
    if not keep.any() or np.array_equal(x,r.x):raise ValueError('Wrong time support for the local fallback')
    report=dict(method='preserve the previous attached steel proposal outside the measured structure conflict',
        source_intersection_seconds=sorted(set(stamp)),disabled_controls=int(keep.sum()),
        unchanged_controls_outside_witness=True,reused_surface_candidate_sha256=old['candidate_surface_sha256'])
    r.emit('pruned_measured_surface_conflict',**report)
    write(r.args.output/'objective_weights.json',r.weights_record())
    verify_candidate(r,x,report,scan_seed=(packet,x_old))


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','previous-run','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--native-review',type=Path)
    a=p.parse_args();a.previous=a.latest;a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf8',buffering=1) as f:
        r=SkateReplay(a,f)
        run(r,a.previous_run)


if __name__=='__main__':main()
