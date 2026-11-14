"""Targeted original reads for blue annuli and paint bands after the broad survey."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
import json
import sys

import joint_inputs as ji
ji.enable_historical_source_readers()
from joint_inputs import OUT, H, TOP, BOTTOM, TB, STEP, W, HIMG, np, cv2
import read_current_material as mat
from read_ring_secondary import annulus
from read_material_871 import red_line
from survey import packed_projection
from qualify import raw_pixels, segment_gap
from audit_residuals import JROOT

TARGET=OUT/'supplement'
BLUE_RINGS=('F1','F2','R1','R2')
EXTRA_SHOTS=('s03','s04','s06','s08','s09','s10','s12','s13','s14','s15','s16','s18','s19','s20','s21','s22','s23','s24')
CARDS={'s03':(39.50,40.94),'s10':(78.50,83.52),'s12':(104.72,),'s14':(121.80,123.18),
       's19':(143.56,),'s20':(144.94,),'s21':(147.66,),'s22':(153.92,155.04)}

def around(frame,q,n):
    unit=n/max(np.linalg.norm(n),1.e-8)
    return unit

def blue_pair(raw,mask,outer,middle):
    # Distinct opening and closing colour transitions on the blue annulus;
    # a dark trough alone cannot qualify. Existing detector locates the
    # outer rim; use its measured paint width to check the opposite rim.
    rows=mat.outer_house(raw,mask,outer,middle,lambda q:True)
    keep=[]
    for g in rows:
        p=np.asarray(g['uv'])
        n=np.asarray(g['source_normal'])
        width=float(g['actual_annulus_width_px'])
        q=p-width*n
        if not ((q>[4,115])&(q<[1915,1072])).all():continue
        u=np.vstack((p+5*n,p-min(5.,width*.28)*n,q+min(5.,width*.28)*n,q-5*n))
        rgb=raw_pixels(raw,u)
        blue=rgb@mat.BLUE
        diff=min(blue[1],blue[2])-max(blue[0],blue[3])
        if g['blue_contrast']<1.8 or diff<.45:continue
        keep.append(dict(g,inner_uv=q.tolist(),chroma_pair=float(diff)))
    if len(keep)<5:return []
    n=np.asarray([g['source_normal'] for g in keep])
    p=np.asarray([g['uv'] for g in keep])
    if np.ptp(n,axis=0).max()<.28 or np.ptp(p,axis=0).max()<112.:return []
    return keep


def board_gap(p, q):
    u=q['uv'];s=q['status']
    p=np.asarray(p,float)
    v=~(s!=0)
    u=u[v]
    u=u[(u[:,0]>-20)&(u[:,0]<1940)&(u[:,1]>130)&(u[:,1]<1079)]
    return segment_gap(p,u,adjacent=45.)


def separate_goal(raw, p, rail):
    u=np.array([g['uv'] for g in p]); n=np.array([g['source_normal'] for g in p])
    d=board_gap(u,rail)
    yellow=np.vstack([raw_pixels(raw,u+n*v)@np.array([-1.,.48,.52]) for v in (-7.,-3.,0.,3.,7.)])
    # Exclude the overlapping kickplate even at a native ice/board junction.
    yes=(d>=14.)&((yellow>15.).sum(0)<=1)
    return [g for g,v in zip(p,yes,strict=True) if v]


def scan_shot(native,shot,world,groups,mask_map):
    key=shot.id
    frame_range=shot.pts
    selected=set(frame_range[::6].tolist())|{int(frame_range[-1])}
    for sec in CARDS.get(key,()):
        p=round(sec*12800)
        selected.update(int(q) for q in p+STEP*np.arange(-12,13,3) if frame_range[0]<=q<=frame_range[-1])
    selected=np.array(sorted(selected),np.int64)
    hist=list(ji.lines(shot.evidence['path']))
    cached={}
    shots=[]
    for i,(p,raw) in enumerate(ji.source_reader().exact_frames(shot.source,selected.tolist(),time_base=TB)):
        idx=int((p-frame_range[0])//STEP)
        h=ji.sha256(raw.tobytes()).hexdigest()
        if h!=hist[idx]['source_BGR_sha256']:raise ValueError((key,p,'source checksum'))
        if p not in cached:
            cached,slices,_=packed_projection(shot,native,selected[i:i+6],world,world)
        z=cached.pop(p)
        if p not in mask_map:
            shots.append(dict(pts=p,source_BGR_sha256=h,groups=[]))
            continue
        mask=mat.mask_frame(mask_map,p)
        get=lambda k:mat.subproj(z,slices[k])
        foot=get(BOTTOM)
        gate=mat.rink_gate(z,slices,shot.rigs[idx])
        good=[]
        for name in BLUE_RINGS:
            group=groups[name];ko=group['outer'];km=group['middle']
            pair=blue_pair(raw,mask,get(ko),get(km))
            if not pair:
                continue
            # F1/S19 looked like a displaced reflection in the broad pass.
            # Re-read blue flanks but keep it out until paired appearance
            # and temporal association can independently identify the arc.
            if key=='s19' and name=='F1':
                continue
            span=float(np.linalg.norm(np.ptp(np.asarray([r['uv'] for r in pair]),axis=0)))
            if span<125.:continue
            good.append(mat.used_rows(ko,'ice.'+name+':outer',name,pair,'paired_blue_annulus_outer_source'))
        for name in ('B1','B2','C','goal_line_left','goal_line_right'):
            k=groups[name]['single']
            if name.startswith('goal_') and key not in ('s04','s14','s19','s20','s21','s22','s23'):continue
            try:
                q=mat.blue_ice_read(raw,mask,get(k),name,gate)
            except (ValueError,IndexError):
                continue
            if name.startswith('goal_') and len(q):
                q=separate_goal(raw,q,foot)
            if len(q)<4:continue
            if np.linalg.norm(np.ptp(np.asarray([g['uv'] for g in q]),axis=0))<155.:continue
            good.append(mat.used_rows(k,'ice.'+name,name,q,'direct_coloured_paint_flanks'))
        if key=='s03':
            for k in groups['Ad9']:
                q=mat.badge(raw,mask,get(k))
                if len(q)>=5:
                    good.append(mat.used_rows(k,'ad_panel','Ad9',q,'direct_red_badge_ink'))
        shots.append(dict(pts=p,source_BGR_sha256=h,groups=good))
    channels=defaultdict(list)
    for f in shots:
        for g in f['groups']:
            v=np.median([x['offset'] for x in g['detail']])
            channels[g['path'],g['material']].append((f['pts'],float(v)))
    for f in shots:
        keep=[]
        p=f['pts']
        for g in f['groups']:
            centre=float(np.median([x['offset'] for x in g['detail']]))
            close=[abs(q-p) for q,v in channels[g['path'],g['material']]
                   if 0<abs(q-p)<=STEP*24 and abs(v-centre)<8.]
            if len(close)>=2:keep.append(g)
        f['groups']=keep
    return shots


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--part',type=int,default=0)
    ap.add_argument('--parts',type=int,default=1)
    args=ap.parse_args()
    cv2.setNumThreads(1)
    _,native,shots,_=ji.inputs()
    keys=defaultdict(dict)
    for k,p in native.paths.items():
        family=ji.family(k,native)
        if family in BLUE_RINGS:
            z=p['label'].rsplit(':',1)[-1]
            if z in ('outer','middle'):keys[family][z]=k
        if family in ('B1','B2','C','goal_line_left','goal_line_right'):
            keys[family]['single']=k
    keys['Ad9']=[k for k in native.paths if k.startswith('print_instance/Ad9::') and 'red_field' in k]
    names={TOP,BOTTOM,*keys['Ad9'],*(k for f,x in keys.items() if f!='Ad9' for k in x.values())}
    world={k:v[0] for k,v in ji.native_paths(native,sorted(names),count=221).items()}
    masks=dict(ji.source_reader().mask_index())
    TARGET.mkdir(exist_ok=True)
    t0=monotonic()
    for sid in EXTRA_SHOTS[args.part::args.parts]:
        name='jin_'+sid
        row=scan_shot(native,shots[name],world,keys,masks)
        file=TARGET/(name+'.jsonl')
        with file.open('x',encoding='utf8',buffering=1) as f:
            for r in row:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
        print(name,len(row),Counter(g['mark'] for f in row for g in f['groups']),
               round(monotonic()-t0,1),flush=True)

if __name__=='__main__':main()
