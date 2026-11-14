"""Verify new exact source exposures in protected intervals and measure cross-programme A4."""
from collections import defaultdict, Counter
from argparse import ArgumentParser
from pathlib import Path
import json
from time import monotonic

import joint_inputs as ji
ji.enable_historical_source_readers()
from joint_inputs import OUT,TB,STEP,TOP,BOTTOM,cv2,np
from survey import packed_projection
import read_current_material as mat
from supplement import blue_pair,separate_goal
from audit_residuals import localize
from prepare_guarded import role
import solve_joint as sj
import verify_joint as verify

PREP=OUT/'prepared_guarded'
CAND=OUT/'candidate/review.npz'
DEST=OUT/'reserved_source'
MASK_DIR=ji.DATA/'runs/native_programme_20261031/source_bootstrap/person_masks_refined/full/frames.jsonl'
A4=ji.H/'q871_C_rail_A4_1112/a4_cross_programme.json'
A4_VALID=ji.H/'q871_C_rail_A4_1112/a4_verified_cross_programme.json'
JIN=('s03','s04','s06','s10','s12','s13','s14','s15','s16','s18','s19','s20','s21','s22','s24')
OKADA=('s07','s08','s09')


def wanted(shot,known):
    pts=[]
    for i,p in enumerate(shot.pts.tolist()):
        if role(shot,p)==0 or p in known or i%2!=1:continue
        # Keep a full 0.16-second gap to the training frames.
        l=max(0,i-6);r=min(len(shot.pts),i+7)
        if any(role(shot,int(z))==0 for z in shot.pts[l:r]):continue
        pts.append(p)
    return np.asarray(pts,np.int64)


def pack_reference(native):
    keys={}
    for k,p in native.paths.items():
        f=ji.family(k,native)
        end=p['label'].rsplit(':',1)[-1]
        if f in ('B1','B2','C','goal_line_left','goal_line_right'):keys[f]=k
        if f in ('R1','R2','F1','F2') and end in ('outer','middle'):
            keys[f+':'+end]=k
    return keys


def collect(native,shot,keys,selected,masks,ok=False):
    pp=mat.print_configs(native,(4,9,13,14,17,18,19,26))
    world={k:v[0] for k,v in ji.native_paths(native,{TOP,BOTTOM,*keys.values(),*pp},count=181).items()}
    cache={}
    for j,(p,raw) in enumerate(ji.source_reader().exact_frames(shot.source,selected.tolist(),time_base=TB)):
        m=masks.get(p)
        if m is None:continue
        if ok:
            if ji.sha256(np.ascontiguousarray(raw[:,:,::-1]).tobytes()).hexdigest()!=m['decoded_native_RGB_sha256']:
                raise ValueError('Okada native RGB evidence changed')
            im=cv2.imdecode(np.fromfile(m['mask_path'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if im.shape!=raw.shape[:2]:raise ValueError('Okada mask has a different image domain')
            mask=cv2.dilate(np.uint8(im>0),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
        else:
            mask=mat.mask_frame(masks,p)
        h=ji.sha256(raw.tobytes()).hexdigest()
        if p not in cache:
            cache,sl,_=packed_projection(shot,native,selected[j:j+6],world,world)
        z=cache.pop(p)
        get=lambda k:mat.subproj(z,sl[k])
        i=int(shot.at(p))
        gate=mat.rink_gate(z,sl,shot.rigs[i])
        g=[]
        def add(k,q,origin):
            if q:
                g.append(dict(k=k,uv=np.asarray([r['uv'] for r in q]),
                              detail=q,origin=origin))
        for f in ('B1','B2','C','goal_line_left','goal_line_right'):
            k=keys[f]
            t=mat.blue_ice_read(raw,mask,get(k),f,gate)
            if f.startswith('goal_') and t:
                t=separate_goal(raw,t,get(BOTTOM))
                if len(t)<4:t=[]
            add(k,t,'paired_paint_flanks')
        for f in ('R1','R2','F1','F2'):
            if f=='F1' and shot.name=='jin_s19':continue
            k=keys[f+':outer']
            add(k,blue_pair(raw,mask,get(k),get(keys[f+':middle'])),'blue_annulus')
        c={k:v for k,v in pp.items() if (get(k)['status']==0).sum()>=8}
        if c:
            for r in mat.read_prints(raw,mask,z,sl,c):
                if len(r['uv'])<4:continue
                if r['material'] not in ('panel_edge','red_badge_outer_print_stroke','white_letter','signed_logo'):
                    continue
                if not any(x in r['path'] for x in ('edge','vertical','badge','outer_circle','ANKARA','JUNIOR','red_field','ISU','native3976')):
                    continue
                g.append(dict(k=r['path'],uv=np.asarray(r['uv']),detail=r['detail'],origin=r['material']))
        yield (i,p,shot.times[i],g,h)


def gate_rows(rows):
    seq=defaultdict(list)
    for i,p,t,groups,h in rows:
        for g in groups:
            offsets=[z.get('offset',z.get('normal_shift_px')) for z in g['detail']]
            good=[x for x in offsets if x is not None]
            a=float(np.median(good)) if good else None
            seq[g['k'],g['origin']].append((p,a))
    for i,p,t,groups,h in rows:
        gg=[]
        for g in groups:
            data=seq[g['k'],g['origin']]
            own=next(z for q,z in data if q==p)
            if not any(0<abs(q-p)<=STEP*24 and (own is None or z is None or abs(own-z)<9.) for q,z in data):
                continue
            uv=g['uv']
            from qualify import cover_indices
            ix=cover_indices(uv,8,12.)
            if len(ix)>=3:gg.append(dict(g,uv=uv[ix]))
        yield i,p,t,gg,h


def measure(native,shot,state,rows):
    points=[(i,p,t,g) for i,p,t,g,h in rows if g]
    return localize(shot,native,points,cs=state[shot.name+'_controls'],rig=state[shot.name+'_rig'])


def a4_check(shots,native,base,review):
    z=ji.read(A4); p=ji.read(A4_VALID)
    wanted_=set(p['checked_okada_source_pts'])
    source=[r for r in z['rows'] if r.get('programme',r.get('program','')).lower()=='okada' and r['pts'] in wanted_]
    if not source:source=[r for r in z['rows'] if r['pts'] in wanted_]
    matched=[]
    for g in source:
        tag='okada_'+g['shot']
        shot=shots[tag];pt=g['pts'];i=int(shot.at(pt))
        # The original Okada A4 readings were previously visually qualified.
        uv=np.asarray(g['uv'],float)
        h=next(ji.source_reader().exact_frames(shot.source,[pt],time_base=TB))[1]
        digest=g.get('source_BGR_sha256')
        if digest and ji.sha256(h.tobytes()).hexdigest()!=digest:raise ValueError('A4 source hash')
        row=(i,pt,shot.times[i],[dict(k=z['path'],uv=uv)],digest)
        a=measure(native['baseline'],shot,base,[row])[0]
        b=measure(native['review'],shot,review,[row])[0]
        matched.append(dict(shot=shot.id,pts=pt,points=len(uv),
                        baseline=ji.stats(a['error']),review=ji.stats(b['error'])))
    return matched


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--part',choices=('jin','okada'))
    args=ap.parse_args()
    cv2.setNumThreads(1)
    a,old,shots,_=ji.inputs()
    s=verify.load_checkpoint(CAND)
    base=verify.load_checkpoint(PREP/'baseline.npz')
    n=dict(baseline=old,review=ji.NativeSceneGPU(old.scene_path,old.core,
                   parameter_state=ji.ParameterState(old.parameter_state.layout,s['scene_values'])))
    k=pack_reference(old)
    DEST.mkdir(exist_ok=True)
    if args.part=='jin':
        masks=dict(ji.source_reader().mask_index())
    else:
        masks={r['source_binding']['source_pts']:r for r in ji.lines(MASK_DIR)}
    output={};t0=monotonic()
    for sid in JIN if args.part=='jin' else OKADA:
        name=args.part+'_'+sid;shot=shots[name]
        known={r['pts'] for r in ji.lines(PREP/(name+'_decisions.jsonl'))}
        pts=wanted(shot,known)
        if args.part=='okada':
            pts=np.array([p for p in pts if p in masks],np.int64)
        evidence=list(gate_rows(list(collect(old,shot,k,pts,masks,args.part=='okada')))) if len(pts) else []
        d={}
        aa=measure(n['baseline'],shot,base,evidence);bb=measure(n['review'],shot,s,evidence)
        lookup={(r['pts'],r['k']):r for r in bb}
        out=[];all_=[]
        for r in aa:
            q=lookup.get((r['pts'],r['k']))
            if q is None:raise ValueError('Candidate no longer projects reserved source geometry')
            m=r['family']
            for item in zip(r['error'],q['error'],strict=True):
                all_.append((m,*map(float,item)))
            out.append(dict(pts=r['pts'],path=r['k'],origin=r['origin'],family=m,
                        uv=r['uv'].tolist(),baseline=r['error'].tolist(),review=q['error'].tolist()))
        def paired_errors(rows):
            x=np.asarray(rows,float)
            return dict(baseline=ji.stats(x[:,0]),review=ji.stats(x[:,1])) if len(x) else {}
        output[name]=dict(scheduled=len(pts),actual_qualified_frames=len({r['pts'] for r in aa}),
                          all=paired_errors([(x,y) for _,x,y in all_]),
                          by_family={m:paired_errors([(x,y) for k,x,y in all_ if k==m]) for m in set(k for k,x,y in all_)})
        ji.write(DEST/(name+'.json'),dict(summary=output[name],source_hashes={p:h for i,p,t,g,h in evidence},rows=out))
        print(name,len(pts),len(out),output[name]['all'],round(monotonic()-t0,1),flush=True)
    if args.part=='okada':output['A4_visual_legacy_verified']=a4_check(shots,n,base,s)
    ji.write(DEST/(args.part+'.json'),output)
if __name__=='__main__':main()
