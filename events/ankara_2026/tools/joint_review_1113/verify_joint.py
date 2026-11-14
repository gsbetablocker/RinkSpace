"""Profile identical held original UVs on each candidate and the published cameras."""
from argparse import ArgumentParser
from collections import defaultdict, Counter
from pathlib import Path
from time import monotonic
import json
import hashlib

import joint_inputs as ji
from joint_inputs import np, OUT
import solve_joint as sj
from audit_residuals import localize, TRAIN, AUDIT, DEVELOP, GUARD

VARIANTS=('baseline','camera','joint')
FOLDS={0:'train',1:'development',2:'audit',3:'guard'}


def load_checkpoint(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def shot_from_inputs(name,setup,b):
    d=setup['shots'][name]
    class Bound:
        pass
    s=Bound()
    for k,v in d.items():setattr(s,k,v)
    s.id=d['shot'];s.name=name
    s.controls=b[name+'_controls'];s.rigs=b[name+'_rig'];s.pts=b[name+'_pts'];s.times=b[name+'_times']
    return s


def pack(a,selection):
    by_frame=defaultdict(lambda:defaultdict(list))
    for r in np.flatnonzero(selection):
        by_frame[(a['idx'][r],a['pts'][r])][a['path'][r]].append(r)
    return by_frame


def eval_one(native,shot,a,state,sel):
    pack_=pack(a,sel)
    groups=[]
    for (i,p),look in pack_.items():
        obs=[]
        for k,indices in look.items():
            obs.append(dict(k=k,uv=a['uv'][indices],row=indices))
        groups.append((int(i),int(p),shot.times[i],obs))
    out=localize(shot,native,groups,cs=state[shot.name+'_controls'],rig=state[shot.name+'_rig'])
    errors=np.full(len(a['pts']),np.nan)
    signed=np.full(len(a['pts']),np.nan)
    for row in out:
        errors[row['row']]=row['error'];signed[row['row']]=row['residual']
    return errors,signed


def family_group(v):
    if v.startswith('Ad'):return 'ads'
    if v.startswith('board.'):return 'board'
    return 'ice'


def collect(a,errors,ss,sel):
    by_frame=defaultdict(list)
    for i in np.flatnonzero(sel):
        d=dict(p=int(a['pts'][i]),e=float(errors[i]),w=float(a['weight'][i]),fresh=bool(a['fresh'][i]),sgn=float(ss[i]))
        f=a['family'][i]
        by_frame[f].append(d)
        by_frame[family_group(f)].append(d)
        by_frame['all'].append(d)
        if d['fresh']:by_frame['fresh_only'].append(d)
    return by_frame


def distribution(a):
    if not a:return {}
    err=np.asarray([r['e'] for r in a])
    w=np.asarray([r['w'] for r in a])**2
    valid=np.isfinite(err)
    e=err[valid]
    w=w[valid]
    if not len(e):return dict(rows=len(err),missing=len(err),frames=len({r['p'] for r in a}))
    return {**ji.stats(err),'frames':len({r['p'] for r in a}),
            'balanced_rms':float(np.sqrt(np.dot(w,e**2)/w.sum())),
            'balanced_p90':weighted_percentile(e,w,.9),
            'signed_median':float(np.median([r['sgn'] for r in a if np.isfinite(r['sgn'])]))}


def weighted_percentile(e,w,f):
    i=np.argsort(e,kind='stable')
    e=e[i];cw=np.cumsum(w[i]);j=min(len(e)-1,np.searchsorted(cw,cw[-1]*f))
    return float(e[j])


def summarise(data):
    return {v:distribution(w) for v,w in data.items()}


def protect_report(shot,a,errs):
    records={}
    is_held=(a['fold']==AUDIT)|(a['fold']==DEVELOP)
    # Checks use matched observed features on the same source exposures.
    for f in ('board.top','board.bottom','lane1','lane2','lane3','lane4',
              'goal_line_left','goal_line_right','C','B1','B2','A4','F1','F2','R1','R2','Ad4','Ad17','Ad18','Ad19','Ad26'):
        u=(a['family']==f)&is_held
        if u.sum()<12:continue
        records[f]={variant:ji.stats(e[0][u]) for variant,e in errs.items()}
    return records


def motion_report(native,shot,base,new):
    key=shot.name
    i=np.arange(len(shot.times))
    native_time=shot.times
    def camera(z):
        cs=z[key+'_controls'];r=z[key+'_rig']
        t=shot.times
        # Both centre times and in-between exposure times use the same cubic.
        q=native.core.evaluate_motion_kinematics(cs,shot.first,shot.dt,np.sort(np.r_[t,(t[1:]+t[:-1])*.5]))
        return q
    vb=camera(base);va=camera(new)
    diff={}
    for k in ('position','velocity','acceleration','jerk'):
        if k not in vb:continue
        bb=np.asarray(vb[k]);cc=np.asarray(va[k]);err=cc-bb
        diff[k]=dict(delta_max=np.max(abs(err),axis=0).tolist(),max_before=np.max(abs(bb),axis=0).tolist(),
                     max_after=np.max(abs(cc),axis=0).tolist())
    # Record actual projected pan/tilt/zoom corrections; cuts have separate clocks.
    q=new[key+'_controls']-base[key+'_controls']
    v=new[key+'_rig']-base[key+'_rig']
    return dict(channels=diff,max_control_shift=np.max(abs(q),axis=0).tolist(),optics_delta=v[0].tolist())


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--fit',default='fit')
    ap.add_argument('--input',default='prepared_joint')
    ap.add_argument('--out',default='verification')
    ap.add_argument('--variants',nargs='+',default=list(VARIANTS))
    ap.add_argument('--only',nargs='*',default=[])
    ap.add_argument('--include-train',action='store_true')
    v=ap.parse_args()
    start=monotonic()
    sj.INPUT=OUT/v.input
    setup,core,b,series=sj.load()
    dest=OUT/v.out
    dest.mkdir(exist_ok=True)
    state={'baseline':b}
    for k in v.variants:
        if k!='baseline':state[k]=load_checkpoint(OUT/v.fit/(k+'.npz'))
    native={k:sj.NativeSceneGPU(setup['scene_path'],core,parameter_state=sj.ParameterState(setup['scene_layout'],
                                    np.ascontiguousarray(a['scene_values']))) for k,a in state.items()}
    summed=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    all_protected={};per_shot={};motions={}
    for name,a in series.items():
        if v.only and name not in v.only:continue
        shot=shot_from_inputs(name,setup,b)
        focus=ji.JIN_FOCUS if shot.program=='jin' else ji.OKADA_FOCUS
        at=np.zeros(len(a['pts']),bool)
        for t in focus.get(shot.id,()):
            at|=abs(a['pts']-round(t*12800))<=256*10
        check=(a['fold']==AUDIT)|(a['fold']==DEVELOP)|at
        if v.include_train:check|=a['fold']==TRAIN
        errs={};dd={}
        for variant,z in state.items():
            errors,signed=eval_one(native[variant],shot,a,z,check)
            errs[variant]=(errors,signed)
            dd[variant]={}
            for k in (DEVELOP,AUDIT):
                z0=collect(a,errors,signed,check&(a['fold']==k))
                dd[variant][FOLDS[k]]=summarise(z0)
                for f,rows in z0.items():
                    summed[variant][shot.program][FOLDS[k]][f].extend(rows)
            if at.any():
                dd[variant]['reported_windows']=summarise(collect(a,errors,signed,at))
            if v.include_train:
                dd[variant]['train']=summarise(collect(a,errors,signed,check&(a['fold']==TRAIN)))
        per_shot[name]=dd
        all_protected[name]=protect_report(shot,a,errs)
        motions[name]={k:motion_report(native[k],shot,b,z) for k,z in state.items() if k!='baseline'}
        ji.write(dest/(name+'.json'),dd)
        np.savez_compressed(dest/(name+'.npz'),pts=a['pts'],path=a['path'],family=a['family'],fold=a['fold'],
                   uv=a['uv'],weight=a['weight'],fresh=a['fresh'],
                   **{v+suffix:q for v,values in errs.items() for suffix,q in zip(('_e','_signed'),values)})
        def display(s):
            return tuple(round(dd[k].get(s,{}).get('all',{}).get('balanced_rms',0.),3) for k in state)
        print(name,'development',display('development'),'audit',display('audit'),'reported',display('reported_windows'),
              'elapsed',round(monotonic()-start,1),flush=True)
    tree={k:{prog:{fold:summarise(d) for fold,d in f.items()} for prog,f in p.items()} for k,p in summed.items()}
    report=dict(variants=list(state),folds=FOLDS,
              compared_same_exact_uv=True,shared_geometry_per_variant=True,source_distance='nearest_on_native_rolling_row_curve',
              aggregate=tree,shot_reports=per_shot,protected=all_protected,motion=motions)
    sj.write_json(dest/'comparison.json',report)
    for k,programmes in tree.items():
        for prog,folds in programmes.items():
            for fold,val in folds.items():
                print(k,prog,fold,{f:round(row.get('balanced_rms',0.),3)
                        for f,row in val.items() if f in ('all','board','ice','ads','fresh_only')},flush=True)

if __name__=='__main__':main()
