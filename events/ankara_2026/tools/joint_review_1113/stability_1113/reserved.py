"""Independent source-pixel checks absent from the stability solve and selection."""
from collections import defaultdict
from camera_audit import HERE,PREP,ji,sj,vj,setup,native_scene
from audit_residuals import localize
import numpy as np


def evaluate(d,native,pose,groups):
    return localize(d,native,groups,cs=pose[d.name+'_controls'],rig=pose[d.name+'_rig'])


def stats(data):
    v=np.asarray(data)
    if not len(v):
        return {'points':0}
    return dict(points=len(v),median_px=float(np.median(v)),p90_px=float(np.quantile(v,.9)),
                rms_px=float(np.sqrt(np.mean(v*v))))


def main():
    m,c,history,baseline=setup()
    state=vj.load_checkpoint(HERE/'selected.npz')
    native=native_scene(m,c,state)
    changes=set(ji.read(HERE/'selection.json')['changes'])
    combined={}
    by_program=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    for k in m['shots']:
        paths=[HERE.parent/root/(k+'.json') for root in ('reserved_source','final_checks','independent_checks')]
        if not any(p.exists() for p in paths):continue
        d=vj.shot_from_inputs(k,m,history)
        # All of these PTS were selected as a separate check before this pass.
        rows=defaultdict(dict)
        sources=[]
        inputs=vj.load_checkpoint(PREP/(k+'.npz'))
        named=set(inputs['pts'][inputs['fold']==0])
        for p in paths:
            if not p.exists():continue
            q=ji.read(p)
            if q.get('rows'):sources.append(ji.pin(p))
            for a in q.get('rows',[]):
                pt=a['pts']
                path=a['path']
                if pt in named or path not in native.paths:continue
                uv=np.asarray(a['uv'],float).reshape(-1,2)
                if len(uv):
                    rows[pt].setdefault(path,[]).extend(uv.tolist())
        if not rows:continue
        groups=[]
        for p,h in sorted(rows.items()):
            j=(p-d.pts[0])//256
            if d.pts[j]!=p:raise ValueError((d.name,p))
            groups.append((int(j),p,d.times[j],[dict(k=path,uv=np.unique(uv,axis=0)) for path,uv in h.items()]))
        a=evaluate(d,native,baseline,groups)
        b=evaluate(d,native,state,groups) if k in changes else a
        lookup={(r['pts'],r['k']):r for r in b}
        result=defaultdict(lambda:[[],[]])
        missing=0
        for l in a:
            r=lookup.get((l['pts'],l['k']))
            if r is None:
                missing+=len(l['uv'])
                continue
            f=l['family']
            layer='ads' if f.startswith('Ad') else 'rail' if f.startswith('board.') else 'ice'
            for key in ('all',layer,f):
                result[key][0].extend(l['error'].tolist())
                result[key][1].extend(r['error'].tolist())
                by_program[d.program][key][0].extend(l['error'].tolist())
                by_program[d.program][key][1].extend(r['error'].tolist())
        combined[k]=dict(frames=len(rows),lost=missing,observations=sources,
            scores={f:dict(before=stats(x),after=stats(y)) for f,(x,y) in result.items()})
        print(k,len(rows),missing,combined[k]['scores'].get('all'),flush=True)
    summary={p:{f:dict(before=stats(x),after=stats(y)) for f,(x,y) in a.items() if f in ('all','ads','rail','ice')}
             for p,a in by_program.items()}
    sj.write_json(HERE/'reserved_checks.json',dict(summary=summary,shots=combined,
        scope='Direct source UV from reserved-source registries; excluded named training exposures; fresh movement folds independently assessed'))
    print(summary,flush=True)

if __name__=='__main__':main()
