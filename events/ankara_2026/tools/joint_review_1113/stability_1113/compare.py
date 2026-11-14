"""Score candidate optics, full original trajectories and untouched image observations."""
from argparse import ArgumentParser
from time import monotonic
from camera_audit import HERE, PREP, ji, sj, vj, setup, native_scene, audit
from measure import predictions, summarize, stat, frame_errors
import numpy as np

WINDOWS={
's01':[(32.70,33.14)], 's03':[(39.48,41.68)], 's04':[(42.18,42.70),(46.98,47.42)],
's06':[(52.72,57.46)], 's07':[(57.48,58.12),(62.92,63.25)], 's09':[(72.40,72.98),(77.16,78.24)],
's10':[(78.26,78.70),(83.30,85.20),(88.84,89.14)], 's12':[(104.50,104.94)],
's13':[(105.38,106.08),(112.30,113.62)], 's14':[(120.38,122.18),(122.66,123.04),(123.10,123.98)],
's15':[(124.40,124.90)], 's16':[(127.62,128.36)], 's17':[(132.90,133.46)],
's18':[(136.78,137.96)], 's19':[(142.98,143.60)],
's21':[(147.02,148.20),(148.86,149.08)], 's22':[(149.18,149.95),(153.12,155.08)],
's24':[(171.08,172.14),(172.16,178.50),(183.04,183.50),(194.50,195.24)]
}
LAYERS=('rail','ice','ads')
MOTIONS=('board_adjacent','board_jitter','board_lag8','ice_adjacent','ice_jitter','ice_lag8')


def compare_positions(d,a,b0,b1,mask):
    f=a['family']
    by=np.where(np.char.startswith(f,'Ad'),'ads',np.where(np.char.startswith(f,'board'),'rail','ice'))
    q={}
    named=set(f[np.flatnonzero(mask)])
    minrows=12
    for k in list(LAYERS)+sorted(named):
        g=mask&(by==k if k in LAYERS else f==k)
        common=g&np.isfinite(b0)&np.isfinite(b1)
        if common.sum()<minrows:
            continue
        # Use equally weighted exposure medians, then summarize in native px.
        before=frame_errors(a['idx'],b0,common,len(d.pts),minimum=1)
        after=frame_errors(a['idx'],b1,common,len(d.pts),minimum=1)
        q[k]=dict(points=int(common.sum()),missing_before=int(np.sum(g&~np.isfinite(b0))),
            missing_after=int(np.sum(g&~np.isfinite(b1))),before=stat(before),after=stat(after))
    return q


def measure_uv(native,d,a,cur,cache):
    sel=(a['fold']==2)|(a['fold']==3)
    if np.count_nonzero(sel)<40:
        sel=a['fold']!=0
    if cache.exists():
        old=np.load(cache)['error']
    else:
        old,_=vj.eval_one(native,d,a,{d.name+'_controls':d.controls,d.name+'_rig':d.rigs},sel)
        sj.atomic_arrays(cache,error=old)
    e,_=vj.eval_one(native,d,a,cur,sel)
    pos=compare_positions(d,a,old,e,sel)
    by_window={}
    for low,high in WINDOWS.get(d.id,[]):
        s=a['pts']/12800.
        by_window[f'{low:.3f}–{high:.3f}']=compare_positions(d,a,old,e,sel&(s>=low)&(s<=high))
    return pos,by_window,old,e


def compare(d,core,native,a,t,origin,cur,out,cache):
    name=d.name
    pre=vj.load_checkpoint(HERE/'baseline_motion'/(name+'.npz'))
    b=summarize(d,t,predictions(core,native,d,cur[name+'_controls'],cur[name+'_rig'],t))[1]
    summary={}
    by_window={}
    ts=d.pts/12800.
    for role in MOTIONS:
        for fold in ('held','all'):
            k=role+'_'+fold
            a0,a1=pre[k],b[k]
            valid=np.isfinite(a0)&np.isfinite(a1)
            summary[k]={'before':stat(a0[valid]),'after':stat(a1[valid]),
                         'worse_max':float(np.max(a1[valid]-a0[valid])) if valid.any() else None}
            for lo,hi in WINDOWS.get(d.id,[]):
                w=valid&(ts>=lo)&(ts<=hi)
                by_window.setdefault(f'{lo:.3f}–{hi:.3f}',{})[k]={'before':stat(a0[w]),'after':stat(a1[w])}
    p,pw,e0,e1=measure_uv(native,d,a,cur,cache)
    cam,cam_values=audit(core,d,cur[name+'_controls'],cur[name+'_rig'])
    sj.atomic_arrays(out/(name+'_evaluation.npz'),pts=d.pts,**b,before_abs=e0,after_abs=e1)
    return dict(motion=summary,absolute=p,window_motion=by_window,window_absolute=pw,camera=cam)


def main():
    args=ArgumentParser(description=__doc__)
    args.add_argument('candidate')
    args.add_argument('--shots',nargs='+')
    q=args.parse_args()
    m,core,base,state=setup()
    native=native_scene(m,core,state)
    out=HERE/q.candidate
    res={}
    t0=monotonic()
    folder=HERE/'baseline_abs'
    folder.mkdir(exist_ok=True)
    for source in sorted(out.glob('jin_s??.npz')):
        name=source.stem
        if q.shots and name[4:] not in q.shots:
            continue
        d=vj.shot_from_inputs(name,m,base)
        d.controls=state[name+'_controls'];d.rigs=state[name+'_rig']
        r=vj.load_checkpoint(source)
        np.testing.assert_array_equal(r['pts'],d.pts)
        np.testing.assert_array_equal(r['scene_values'],state['scene_values'])
        t=vj.load_checkpoint(HERE/'source_tracks'/(name+'.npz'))
        a=vj.load_checkpoint(PREP/(name+'.npz'))
        data=compare(d,core,native,a,t,state,{name+'_controls':r['controls'],name+'_rig':r['rig']},
            out,folder/(name+'.npz'))
        res[name]=data
        sj.write_json(out/(name+'_evaluation.json'),data)
        def fmt(q):
            return ':'.join(str(round(q[s].get('p90',-1.),2)) for s in ('before','after'))
        print(name,[fmt(data['motion'][k+'_held']) for k in MOTIONS[:4]],
              {k:fmt(data['absolute'][k]) for k in LAYERS if k in data['absolute']},
              round(monotonic()-t0),flush=True)
    sj.write_json(out/'validation.json',res)

if __name__=='__main__':
    main()
