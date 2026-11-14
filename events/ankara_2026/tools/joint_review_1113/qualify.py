"""Prepare balanced, source-qualified observations for one common Ankara fit."""
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
from itertools import groupby
import json

import joint_inputs as ji
from joint_inputs import OUT, np, cv2, STEP, TB, TOP, BOTTOM
from audit_residuals import split, TRAIN, GUARD, JROOT, OKROOT
from scipy.spatial.distance import cdist

AUD=OUT/'residual_audit'
TARGET=OUT/'prepared_joint'
RINK_GHOSTS={'goal_line_left','goal_line_right','trapezoid_left','trapezoid_right'}
RINGS={'F1','F2','R1','R2'}
PRIOR_PROVEN={'blue_band_upper_boundary_direct_source','yellow_kickplate_lower_edge_direct_source',
 'signed_original_print_transition','actual_exposed_letter_outline','two_sided_original_paint_band',
 'paint_two_sides','colored_annulus_both_edges','red_hockey_arc_two_flanks','blue_cap_ice_side_long_edge',
 'blue_cap_ice_side','blue_cap_far','original_source_rail_ice_side','original_s04_confirmed_red_center'}
EDGE_WORDS=('edge','boundary','vertical','ring','outline','border','red_field','native3976','qr')
QUALITY={'board.top':1.6,'board.bottom':1.45,'goal_line_left':1.12,'goal_line_right':1.1,
         'C':1.13,'B1':1.13,'B2':1.13}
MATERIAL_FRESH={'measured_paired_paint_edges','measured_paired_annulus','blue_cap_far','blue_cap_ice_side',
                'original_s04_confirmed_red_center'}
CHECK_JIN=set(ji.JIN_FOCUS)|{'s08','s11','s23'}


def cover_indices(uv,count,min_dist):
    uv=np.asarray(uv,float)
    if not len(uv):
        return np.zeros(0,dtype=int)
    dst=cdist(uv,uv)
    i=int(np.argmin(uv[:,0]+.23*uv[:,1]))
    chosen=[i]
    while len(chosen)<count:
        near=dst[chosen].min(0)
        j=int(np.argmax(near))
        if near[j]<min_dist:break
        chosen.append(j)
    return np.array(chosen,int)


def trim(g,use):
    return {**g,**{k:np.asarray(g[k])[use] for k in ('uv','normal','error','residual','endpoint','station')}}


def feature_groups(groups):
    tags=defaultdict(list)
    for g in groups:
        if g['family'] not in ('near','far'):
            tags[g['family']].append(g)
    return tags


def raw_pixels(frame,uv):
    x=np.ascontiguousarray(uv[:,0].astype('f4')[None,:])
    y=np.ascontiguousarray(uv[:,1].astype('f4')[None,:])
    return cv2.remap(frame,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0].astype(float)


def segment_gap(uv,others, *, adjacent=120.):
    if not len(others):return np.full(len(uv),np.inf)
    # Compare within the observed span of each rail segment and keep branch
    # separation on the near side instead of interpolating across an occlusion.
    delta=np.diff(others,axis=0);norm=np.linalg.norm(delta,axis=1)
    keep=(norm>1.)&(norm<=adjacent)
    if not keep.any():return cdist(uv,others).min(1)
    first=others[:-1][keep];delta=delta[keep];norm=norm[keep]
    shift=uv[:,None,:]-first[None,:,:]
    a=np.clip(np.einsum('nki,ki->nk',shift,delta)/(norm**2),0.,1.)
    d=np.linalg.norm(shift-a[:,:,None]*delta,axis=2)
    return np.minimum(d.min(1),cdist(uv,others).min(1))


def bottom_candidates(groups):
    strips=[]
    for g in groups:
        if g['family']!='board.bottom' or len(g['uv'])<4:continue
        uv=np.asarray(g['uv'])
        strips.append(uv[np.argsort(uv[:,0],kind='stable')])
    return strips


def yellow_strip(frame,g,strips):
    p,n=g['uv'],g['normal']
    proximity=np.min([segment_gap(p,q) for q in strips],axis=0) if strips else np.full(len(p),np.inf)
    if frame is None:
        return proximity<=4.3
    yellow=[]
    for r in (-9,-5,-2,0,2,5,9):
        b,green,red=raw_pixels(frame,p+r*n).T
        yellow.append((.49*green+.51*red-b)>17.)
    on_yellow=np.sum(yellow[1:-1],axis=0)>=2
    next_to_yellow=np.any(yellow,axis=0)
    return (proximity<=5.5) | (on_yellow & (proximity<=23.)) | (next_to_yellow & (proximity<=10.5))


def direct_rail(g,frame):
    # Inspect the blue material on either side of the sampled contour. The
    # matching branch must have a blue interior and a different outer flank.
    q,n=g['uv'],g['normal']
    if frame is None:return np.ones(len(q),bool)
    sig=[]
    for v in (-6.,-3.,-1.,1.,3.,6.):
        b,green,r=raw_pixels(frame,q+v*n).T
        sig.append(b-.53*green-.47*r)
    a=np.median(sig[:3],axis=0);b=np.median(sig[3:],axis=0)
    blue=np.maximum(a,b);cross=np.abs(a-b)
    return (blue>8.)&(cross>5.5)


def shape_overlap(g,others, limit):
    if not others:return np.zeros(len(g['uv']),bool)
    return np.min([cdist(g['uv'],r['uv']).min(1) for r in others],axis=0)<limit


def native_array(g):
    return trim(g,np.ones(len(g['uv']),bool))


def eligible(frame, groups, shot, p, recurring, changes):
    tags=feature_groups(groups)
    strips=bottom_candidates(groups)
    keep=[]
    for row in groups:
        g=native_array(row)
        mat=g['material']; f=g['family']; k=g['k']; new=g['origin']=='fresh'
        error=g['error']; signed=g['residual']
        use=(~g['endpoint']) | (error<6.)
        code=np.full(len(use),'',dtype='<U64')
        if shot.program=='jin':
            if f in RINK_GHOSTS:
                if mat!='original_s04_confirmed_red_center':
                    bad=yellow_strip(frame,g,strips)
                    code[bad]='goal_or_trapezoid_on_kickplate'
                    use&=~bad
            if shot.id=='s14' and f=='A4':
                if new or mat not in PRIOR_PROVEN:
                    bad=shape_overlap(g,tags['F2'],5.5)
                    code[bad]='ambiguous_overlap_with_F2'
                    use&=~bad
                else:
                    bad=shape_overlap(g,tags['F2'],3.0)
                    code[bad]='ambiguous_overlap_with_F2'
                    use&=~bad
            if shot.id=='s14' and f=='F1' and 122.94<=p/12800<=123.42 and k.endswith('146/path0'):
                # The lower arc in this window meets the shadow, and several
                # prior detections describe a different contour than the paired
                # print boundaries. Retain the exposed upper material edge.
                bad=np.zeros(len(use),bool)
                if frame is not None:
                    q,n=g['uv'],g['normal']
                    b1,g1,r1=raw_pixels(frame,q+4*n).T
                    b2,g2,r2=raw_pixels(frame,q-4*n).T
                    blue=(b1-r1)-(b2-r2)
                    bad=(q[:,1]>np.median(q[:,1])+7.) & (np.abs(blue)<1.5)
                code[bad]='F1_ambiguous_lower_shadow_arc'
                use&=~bad
            if f=='board.top' and mat not in {'blue_band_upper_boundary_direct_source','blue_cap_ice_side_long_edge'}:
                # Existing direct blue-band extraction gets priority on S21
                # and S22, where the bench and blue trim create nearby edges.
                if (shot.id in {'s14','s19','s20','s21','s22'} and new) or (
                        shot.id=='s14' and mat=='reviewed_prior_source'):
                    bad=~direct_rail(g,frame)
                    code[bad]='no_blue_lip_transition'
                    use&=~bad
        if new:
            if mat=='original_s04_confirmed_red_center':
                pass
            elif not recurring(g) or (f in RINGS and mat=='measured_paired_annulus' and
                        f=='F1' and shot.id=='s14' and np.median(error)>17.):
                code[use]='weak_or_uncorroborated_source_candidate'
                use[:]=False
            elif f in RINK_GHOSTS:
                # Explicit red reads already verified across the shot tail.
                # Generic pale streaks near the board remain quarantined.
                code[use]='ambiguous_generic_goal_or_trapezoid'
                use[:]=False
            elif f in ('A1','A2','A3','A4') and (len(use)<6 or np.median(error)>24.):
                code[use]='uncertain_red_arc_candidate'
                use[:]=False
        # High error is a review reason and is not itself a mislabel. Groups
        # with no stable association at a finite curve end cannot guide a
        # small correction. All originally measured coordinates stay in audit.
        code[(~use)&(code=='')]='ambiguous_finite_endpoint'
        for reason in set(code[~use])-{''}:
            bad=code==reason
            changes.append(dict(pts=p,path=k,family=f,origin=g['origin'],material=mat,
                                reason=reason,uv=g['uv'][bad].tolist()))
        if use.any():keep.append(trim(g,use))
    return keep


def path_budget(g):
    f=g['family'];k=g['k']
    if f.startswith('Ad'):
        return ('structure' if any(x in k for x in EDGE_WORDS) and
                not any(x in k for x in ('glyph','component','can_c','can_a','can_n','letter')) else 'letters')
    if f in ('board.top','board.bottom'):
        return ('near' if np.median(g['uv'][:,1])>650 else 'far')
    if f.startswith('lane'):
        return k
    if f in RINGS:
        return k
    return f


def take_family(f, groups, max_f, version):
    sub=defaultdict(list)
    for g in groups:
        sub[path_budget(g)].append(g)
    # Equalize coverage across both Lane edges, annular layers, and board
    # branches. Prefer material boundaries over many glyphs from one panel.
    if f.startswith('Ad'):
        budgets={'structure':max_f//2+1,'letters':max_f-(max_f//2+1)}
    else:
        budgets={k:max(2,int(np.ceil(max_f/len(sub)))) for k in sub}
    taken=[]
    for label,a in sub.items():
        preference=lambda g:(int(g['material']=='original_s04_confirmed_red_center')*4+
                             int(g['origin']=='fresh' and g['material'] in MATERIAL_FRESH)*2+
                             int(g['material'] in PRIOR_PROVEN))
        a=sorted(a,key=preference,reverse=True)
        # Do not count the same pixel from the old and new detectors twice.
        ranges=[];flat=[]
        for g in a:
            p=g['uv']
            used=np.vstack(flat) if flat else np.empty((0,2))
            w=(cdist(p,used).min(1)>4.) if len(used) else np.ones(len(p),bool)
            if w.any():
                h=trim(g,w);flat.append(h['uv']);ranges.append(h)
        if not ranges:continue
        pool=np.vstack(flat)
        out=cover_indices(pool,budgets.get(label,max_f) if len(sub)>1 else max_f,12. if not f.startswith('Ad') else 5.)
        # Keep the spatial design uniform and ensure the verified material
        # replacement is used when the old group did not cover it.
        for g in ranges:
            m=len(g['uv']);use=out[(out>=0)&(out<m)]
            if len(use):taken.append(trim(g,use))
            out=out-m
    return taken


def run():
    start=monotonic()
    cv2.setNumThreads(1)
    _,native,shots,_=ji.inputs()
    TARGET.mkdir(exist_ok=True)
    summary={}
    for name,shot in shots.items():
        raw_root=JROOT if shot.program=='jin' else OKROOT
        source={r['pts']:r for r in ji.lines(raw_root/(name+'.jsonl'))}
        rows=defaultdict(list)
        for g in ji.lines(AUD/(name+'.jsonl')):
            rows[g['pts']].append(g)
        temporal=defaultdict(list)
        for p,b in rows.items():
            for g in b:
                if g['origin']=='fresh':
                    temporal[g['k'],g['material']].append((p,np.median(g['residual']),len(g['uv'])))
        pts=np.array(sorted(rows),np.int64)
        roles=dict(zip(pts.tolist(),split(shot,pts).tolist()))
        raw_images=ji.source_reader().exact_frames(shot.source,pts.tolist(),time_base=TB) if shot.program=='jin' and shot.id in CHECK_JIN else ()
        records=Counter();removed=Counter();frows=Counter();uv=[];times=[];idx=[];keys=[];stations=[];weights=[];folds=[];labels=[];fresh=[]
        audit_file=TARGET/(name+'_decisions.jsonl')
        with audit_file.open('x',encoding='utf8') as af:
            stream=iter(raw_images)
            for p in pts.tolist():
                frame=None
                if shot.program=='jin' and shot.id in CHECK_JIN:
                    v,frame=next(stream)
                    if v!=p or ji.sha256(frame.tobytes()).hexdigest()!=source[p]['source_BGR_sha256']:
                        raise ValueError((name,p,'native source changed'))
                support=source[p].get('source_BGR_sha256')
                def recurring(g):
                    nearby=temporal[g['k'],g['material']]
                    centre=float(np.median(g['residual']))
                    return any(0<abs(q-p)<=STEP*22 and abs(s-centre)<(11. if g['family'].startswith('Ad') else 7.) and n>=3
                               for q,s,n in nearby)
                changes=[]
                keep=eligible(frame,rows[p],shot,p,recurring,changes)
                used=[]
                for f,g in feature_groups(keep).items():
                    max_f=9 if f.startswith('board.') else 7 if not f.startswith('Ad') else 6
                    a=take_family(f,g,max_f,name)
                    if not a:continue
                    frows[f]+=sum(len(x['uv']) for x in a)
                    # Do not let a foreground rectangle dominate all ice or
                    # vice versa just because more candidate samples existed.
                    mass=QUALITY.get(f,1.02 if f.startswith('Ad') else 1.)
                    total=sum(len(x['uv']) for x in a)
                    w=np.sqrt(mass/total)
                    for g in a:
                        n=len(g['uv'])
                        uv.extend(g['uv']);stations.extend(g['station']);keys.extend([g['k']]*n)
                        idx.extend([g['idx']]*n);times.extend([p]*n);weights.extend([w]*n);folds.extend([roles[p]]*n)
                        labels.extend([f]*n);fresh.extend([g['origin']=='fresh']*n)
                        records[g['origin']]+=n
                        if g['origin']=='fresh':
                            records['new_'+f]+=n
                        used.append(dict(path=g['k'],origin=g['origin'],material=g['material'],uv=g['uv'].tolist()))
                removed.update({k:sum(len(x['uv']) for x in changes if x['reason']==k and x['origin']=='saved')
                                for k in set(x['reason'] for x in changes)})
                af.write(json.dumps(dict(pts=p,source_BGR_sha256=support,fold=roles[p],groups=used,excluded=changes),
                                    separators=(',',':'))+'\n')
        np.savez_compressed(TARGET/(name+'.npz'),uv=np.asarray(uv),pts=np.asarray(times,np.int64),idx=np.asarray(idx,np.int32),
                path=np.asarray(keys),family=np.asarray(labels),station=np.asarray(stations),weight=np.asarray(weights),
                fold=np.asarray(folds,np.int8),fresh=np.asarray(fresh,bool))
        summary[name]=dict(frames=len(pts),fit_frames=sum(v==TRAIN for v in roles.values()),rows=len(uv),
                by_fold=dict(Counter(folds)),kept=dict(records),dropped_saved=dict(removed),by_feature=dict(frows))
        print(name,len(pts),len(uv),records['fresh'],dict(removed),round(monotonic()-start,1),flush=True)
    ji.write(TARGET/'input_summary.json',summary)
    # Package the already verified current camera and common scene for the
    # solver independently of the historical input-production chain.
    arrays={k+'_controls':s.controls for k,s in shots.items()}
    arrays.update({k+'_rig':s.rigs for k,s in shots.items()})
    arrays.update({k+'_pts':s.pts for k,s in shots.items()})
    arrays.update({k+'_times':s.times for k,s in shots.items()})
    arrays['scene_values']=native.parameter_state.values
    np.savez_compressed(TARGET/'baseline.npz',**arrays)
    setup=dict(scene_path=str(native.scene_path),scene_layout=native.parameter_state.layout,
               geometry_sha256=ji.numeric_hash(native.parameter_state.values),
               core=str(native.core._lib._name),
               shots={k:dict(program=s.program,shot=s.id,camera_index=s.shot['camera_index'],first=s.first,dt=s.dt,source=s.source)
                      for k,s in shots.items()})
    ji.write(TARGET/'setup.json',setup)

if __name__=='__main__':run()
