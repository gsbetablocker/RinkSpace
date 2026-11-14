"""Estimate auditable paired-source residuals and qualification candidates."""
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from time import monotonic

import joint_inputs as ji
from joint_inputs import np, cv2, OUT, H, TB, TOP, BOTTOM
from scipy.spatial.distance import cdist

OKROOT=OUT/'okada_survey_native'
JROOT=OUT/'jin_survey_qualified'
# Historical evidence remains on its original clock and outside the source ledger.
GOAL=H/'s04_goal_tail_1111/source_goal_tail_midline.jsonl'
A4=H/'q871_C_rail_A4_1112/a4_cross_programme.json'
CHOICE=('s03','s04','s06','s08','s09','s10','s12','s13','s14','s15','s16','s18','s19','s20','s21','s22','s24')
TRAIN=0
DEVELOP=1
AUDIT=2
GUARD=3
NEW_MATERIAL={'measured_paired_paint_edges','measured_paired_annulus','blue_cap_ice_side',
              'blue_cap_far','signed_print','printed_luma','signed_logo','white_letter','panel_edge',
              'original_s04_confirmed_red_center','original_okada_confirmed_red_A4'}


def split(shot,pts):
    # Reserve whole source intervals with a support gap. These checks were
    # already available to the baseline and are only holdouts for this pass.
    idx=shot.at(pts)
    q=(idx+19*(int(shot.id[1:])%6))%174
    v=np.full(len(idx),TRAIN,np.int8)
    v[((q>=52)&(q<77))|((q>=112)&(q<136))]=GUARD
    v[(q>=57)&(q<70)]=DEVELOP
    v[(q>=117)&(q<129)]=AUDIT
    return v


def cap_group(g, count):
    uv=np.asarray(g.get('uv',[]),float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not len(uv) or not np.isfinite(uv).all():
        return None
    if (uv<0).any() or (uv>=[1920,1080]).any():return None
    indices=np.arange(len(uv))
    if len(uv)>count:
        distances=cdist(uv,uv)
        sel=[int(np.argmin(uv[:,0]+.15*uv[:,1]))]
        for i in range(count-1):
            sel.append(int(np.argmax(distances[sel].min(axis=0))))
        indices=np.sort(np.unique(sel))
    return uv[indices]


def old_groups(shot,frame):
    return frame.get('previous') or []


def load_extra(shot):
    if shot.name=='jin_s04':
        data={}
        for q in ji.lines(GOAL):
            pts=q['pts']; h=q.get('source_BGR_sha256')
            group=defaultdict(list)
            for r in q['rows']:
                if r['path_id']=='demo_initial/goal_line_left:0':
                    group[r['path_id']].append(r['uv'])
            if group:
                data[pts]=(h,[dict(path=k,uv=v,material='original_s04_confirmed_red_center') for k,v in group.items()])
        return data
    return {}


def preferred_fresh(shot, frame, g, native):
    material=g.get('material')
    k=g['path']; role=ji.family(k,native)
    if material=='original_s04_confirmed_red_center':return True
    if material=='measured_paired_paint_edges':
        # Existing source reader verifies both paint flanks, continuity and
        # colour. Dark stripes must be unambiguously isolated from glare.
        details=g.get('detail',[])
        if role.startswith('lane'):
            return len(details)>=5 and float(np.median([v['contrast'] for v in details]))>=5.
        if role.startswith('A') and len(role)==2 and role[1].isdigit():
            return len(details)>=6 and float(np.median([v['contrast'] for v in details]))>=3.0
        if role in ('goal_line_left','goal_line_right','trapezoid_left','trapezoid_right','B1','B2','C'):
            return len(details)>=4 and float(np.median([v['contrast'] for v in details]))>=3.0
        return False
    if material=='measured_paired_annulus':
        d=g.get('detail',[])
        return len(d)>=5 and (role.startswith('F') or all(r['tone']=='red' for r in d))
    # Blue-lip polarity and paint identity are necessary but they are checked
    # visually below before accepting any replacement on a judging-side rim.
    if material in ('blue_cap_ice_side','blue_cap_far'):
        return len(g.get('uv',[]))>=4
    if role.startswith('Ad'):
        return (any(x in k for x in ('outer','outline','border','left_vertical','right_vertical','top_edge','bottom_edge',
                                     'ANKARA.native','JUNIOR','badge','ring','WYNDHAM','ISU','text'))
                and len(g.get('uv',[]))>=3)
    return False


def generate(shot,native,frames,source_supplement=True):
    rtimes=shot.times
    extra=load_extra(shot) if source_supplement else {}
    frame_map={r['pts']:r for r in frames}
    groups=[]
    for p in sorted(frame_map):
        row=frame_map[p]
        idx=int(shot.at(p))
        if p in extra:
            h,g=extra[p]
            if h!=row.get('source_BGR_sha256'):raise ValueError('Source-proven red stripe does not match this exposure')
            row['fresh'].extend(g)
        obs=[]
        for g in old_groups(shot,row):
            k=g['path']
            if k not in native.paths:continue
            uv=cap_group(g,12 if k in (TOP,BOTTOM) else 7)
            if uv is not None:
                obs.append(dict(k=k,uv=uv,origin='saved',material=g.get('material','unspecified')))
        for g in row.get('fresh',[]):
            k=g['path']
            if k not in native.paths or not preferred_fresh(shot,row,g,native):continue
            uv=cap_group(g,10)
            if uv is not None:
                obs.append(dict(k=k,uv=uv,origin='fresh',material=g['material'],detail=g.get('detail',[])))
        if obs:groups.append((idx,p,rtimes[idx],obs))
    return groups


def localize(shot,native,groups,cs=None,rig=None):
    names={g['k'] for i,p,t,rows in groups for g in rows}
    world=ji.native_paths(native,sorted(names),count=143)
    result=[]
    for offset in range(0,len(groups),10):
        window=groups[offset:offset+10]
        q=list(dict.fromkeys((i,g['k'],t) for i,p,t,rows in window for g in rows))
        if not q:continue
        projected=ji.project_groups(native,shot.rigs if rig is None else rig,
                    shot.controls if cs is None else cs,shot.first,shot.dt,q,{k:v[0] for k,v in world.items()})
        by={(i,k):x for (i,k,t),x in zip(q,projected,strict=True)}
        for i,p,t,rows in window:
            for g in rows:
                k=g['k']
                coord,grid=world[k]
                uv,status=by[i,k]
                ans=ji.nearest(uv,status,g['uv'])
                if ans is None:continue
                seg,frac,n,err,edge=ans
                result.append(dict(g,idx=i,pts=p,t=t,station=grid[seg]+frac*(grid[seg+1]-grid[seg]),
                                   normal=n,error=np.linalg.norm(err,axis=1),endpoint=edge,
                                   residual=np.einsum('ni,ni->n',err,n),family=ji.family(k,native)))
    return result


def summarize(shot,groups):
    features=defaultdict(lambda:defaultdict(list))
    coverage=defaultdict(Counter)
    by_pts=defaultdict(lambda:defaultdict(list))
    for g in groups:
        role=g['family']
        v=g['error']
        features[role][g['origin']].extend(v.tolist())
        coverage[role][g['origin']]+=1
        by_pts[g['pts']][role].append(float(np.median(v)))
    summary={}
    for key,a in features.items():
        z={f:ji.stats(v) for f,v in a.items()}
        summary[key]={**z,'curve_exposures':coverage[key]}
    keys=[]
    for role,item in summary.items():
        if role in ji.KEY_FEATURES or role.startswith('trapezoid'):
            for origin in item.keys()-{'curve_exposures'}:
                if item[origin]['rows']>=9:
                    keys.append((item[origin]['p90'],role,origin,item[origin]['median'],item[origin]['rows']))
    return summary,sorted(keys,reverse=True),by_pts


def use_frames(shot,dir0,stride):
    rows=list(ji.lines(dir0/(shot.name+'.jsonl')))
    # Uniform sampling in time, with extra neighbors around reviewed events.
    # If the source reader used a sparse pass, this does not invent frames.
    if stride>1:
        refs=set(ji.sparse_pts(shot,stride,True))
        return [v for v in rows if v['pts'] in refs]
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('jin','okada'))
    p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--chunks',type=int,default=1)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--root',default=None)
    p.add_argument('--out',default='residual_audit')
    v=p.parse_args()
    a,native,shots,_=ji.inputs()
    scans=OUT/v.root if v.root else (JROOT if v.part=='jin' else OKROOT)
    target=OUT/v.out
    target.mkdir(exist_ok=True)
    clock=monotonic()
    for key,s in [x for x in shots.items() if x[1].program==v.part][v.chunk::v.chunks]:
        rows=use_frames(s,scans,v.stride)
        if not rows:
            raise ValueError('No sampled coverage at '+key)
        print(key,'bind observations at',len(rows),'source exposures',flush=True)
        groups=generate(s,native,rows)
        observations=localize(s,native,groups)
        summary,large,by_pts=summarize(s,observations)
        by_pts={str(p):{k:max(v) for k,v in value.items()} for p,value in by_pts.items()}
        d=dict(features=summary,large=large,by_pts=by_pts,frames=len(rows),
               exact_path_model_sha=ji.numeric_hash(native.parameter_state.values))
        ji.write(target/(key+'.json'),d)
        with (target/(key+'.jsonl')).open('x',encoding='utf-8') as fp:
            for q in observations:
                row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in q.items() if k!='detail'}
                fp.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        print(key,'done',len(observations),[(round(v,1),k,o,round(m,1),n) for v,k,o,m,n in large[:7]],
              'elapsed',round(monotonic()-clock,1),flush=True)


if __name__=='__main__':main()
