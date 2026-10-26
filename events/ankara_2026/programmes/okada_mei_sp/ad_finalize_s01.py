"""Consolidated source-image review; never changes native model parameters."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_source_verify_s01 as v
import collections, hashlib, json, time
import cv2,numpy as np
from scipy.spatial import cKDTree

VERSION='source_entity_final_v3'
OUT=b.OUT


def corner_chart_hypotheses(data,ats):
    charts=collections.defaultdict(list)
    for group,prefix,rowname in [('top_word','Ad7.top_','Ad7.row1_'),('bottom_word','Ad7.bottom_','Ad7.row2_')]:
        a=next(x for x in ats if x.get('entity_group')==group)
        shapes={p['path_id']:np.asarray(p['pixels'],float) for p in a['paths']}
        for pt in (444928,445440,445952):
            rows=np.flatnonzero((data['pts']==pt)&np.char.startswith(data['labels'],rowname))
            src=[];dst=[];ids=[]
            for row in rows:
                label=str(data['labels'][row]);entity,corner=label.split('.')[1:]
                native=prefix+entity.split('_',1)[1]+'_part00';p=shapes[native]
                left=p[:,0].min();ix=np.flatnonzero(np.abs(p[:,0]-left)<1e-4)
                j=ix[np.argmin(p[ix,1]) if corner=='top_left' else np.argmax(p[ix,1])]
                src.append(p[j]);dst.append(data['uv'][row]);ids.append(str(data['ids'][row]))
            if len(src)==4:
                H,_=cv2.findHomography(np.asarray(src),np.asarray(dst),0)
                charts[(7,pt)].append(dict(atlas_id=a['id'],H=H.tolist(),p95=0.,inliers=4,method='same_named_letter_corners_source_chart_navigation_only',source_observation_ids=ids,chart_points_are_not_independent_identity_evidence=True))
    return charts


def exclusion_contours(refs,ats):
    """Additional original-artwork entities serve as negative identity examples."""
    result={};aa=b.assets()
    source=b.read(b.SOURCE/'ad3_binding/typed_source_composition.json')
    layout=b.read(source['frozen_layout_reference']);H=np.asarray(layout['evidence']['Japanese_source_pixel_to_native_matrix'],float)
    neg=[]
    for p in source['contours']:
        if 'Ad3.print.'+p['id'] not in aa[3]['paths']:
            q=cv2.perspectiveTransform(np.asarray(p['xy_composition'],float)[None],H)[0]
            if not np.allclose(q[0],q[-1]):q=np.vstack([q,q[0]])
            neg.append(b.dense_poly(q)[0])
    if neg:result[70]=np.vstack(neg)
    for n in (8,9,10):
        a=aa[n];art=b.read(next(iter(a['paths'].values()))['shape_evidence']['path']);r=art['raster']
        allowed={p['original_shape_key'] for p in a['paths'].values()}
        ps=[]
        for c in art['components']:
            for g in c['geometry']:
                gid=g['id']
                if any(gid==key or gid.endswith('.'+key) for key in allowed):continue
                q=np.asarray(g.get('points_uv',g.get('uv',g.get('points'))),float)*[r['width'],r['height']]
                if q.ndim!=2 or len(q)<2:continue
                if not np.allclose(q[0],q[-1]):q=np.vstack([q,q[0]])
                ps.append(b.dense_poly(q)[0])
        if ps:
            aid=next(x['id'] for x in ats if x['ad']==n and Path(x['image'])==Path(r['path']))
            result[aid]=np.vstack(ps)
    return result


def verify_final(ads=None):
    cv2.setNumThreads(2);data=b.load();ats=b.read(OUT/'atlases.json')['atlases'];atlas_sha=b.sha(OUT/'atlases.json')
    index=b.read(OUT/'source_evidence_index.json')
    hints={int(k):x for k,x in index['hints'].items()}
    hard={int(k):x for k,x in index['withdrawals'].items()}
    retired={int(k):x for k,x in index['retired'].items()}
    vis={r['pts']:r for r in b.read(OUT/'visibility.json')['frames']}
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012);matcher=cv2.BFMatcher(cv2.NORM_L2)
    refs={a['id']:b.prepare_reference(a,sift) for a in ats if ads is None or a['ad'] in ads}
    negatives=exclusion_contours(refs,ats)
    prior={};anchors=collections.defaultdict(list)
    for p in (OUT/'verified_entities').glob('*.json'):
        n=int(p.stem.split('_')[0][2:])
        if ads and n not in ads:continue
        d=b.read(p);prior[(n,d['pts'])]=d
        if len(d['confirmed'])>=5:anchors[n].append(d['pts'])
    charts=corner_chart_hypotheses(data,ats)
    selected=np.flatnonzero((data['ad']>0)&(np.isin(data['ad'],ads) if ads else True))
    wanted=sorted(set(map(int,data['pts'][selected])))
    totals=collections.Counter();start=time.monotonic()
    b.write(OUT/'final_verification_recipe.json',dict(version=VERSION,script=str(Path(__file__).resolve()),script_sha256=b.sha(__file__),entity_metrics_script=str(Path(v.__file__)),entity_metrics_sha256=b.sha(v.__file__),atlas_sha256=atlas_sha,negative_source_entities={k:len(x) for k,x in negatives.items()},source_corner_chart_hypotheses={str(k):x for k,x in charts.items()},reused_source_observations_are_navigation_not_new_measurements=True))
    for frame_no,pt in enumerate(wanted):
        rows=selected[data['pts'][selected]==pt]
        image=vis[pt];bgr=cv2.imread(image['image'])
        if image.get('source_BGR_sha256') and hashlib.sha256(bgr.tobytes()).hexdigest()!=image['source_BGR_sha256']:raise ValueError('Source frame hash mismatch')
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);person=cv2.imread(image['mask_path'],0)
        for n in sorted(set(map(int,data['ad'][rows]))):
            rr=rows[data['ad'][rows]==n];dest=OUT/'verified_final'/f'ad{n}_{pt}.json'
            if dest.exists():
                d=b.read(dest)
                if d['version']!=VERSION:raise ValueError('Final evidence version mismatch')
                totals.update(d['counts']);continue
            if (n,pt) not in prior:raise ValueError('Second-pass source frame not fully reviewed')
            previous=prior[(n,pt)]
            veto={int(r):(hard.get(int(r)) or retired.get(int(r))) for r in rr if int(r) in hard or int(r) in retired}
            active=np.asarray([r for r in rr if int(r) not in veto],int)
            confirmed=[];conflicts=[];fail={};regs=[]
            if len(active):
                uv=data['uv'][active]
                regs=list(previous['registrations'])+charts.get((n,pt),[])
                # Recover partial clear glyphs using actual neighboring exposures
                # that now have independently verified word/letter identities.
                need=any(x['reason']=='no_verified_source_image_entity_registration' for x in previous['excluded'])
                if need:
                    mask=np.zeros_like(gray);lo=np.maximum(np.floor(uv.min(0)-40).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)+41).astype(int),[1920,1080])
                    if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=255
                    mask[person>0]=0
                    kp,des=sift.detectAndCompute(gray,mask);kp=np.asarray([k.pt for k in kp])
                    near=sorted(anchors[n],key=lambda a:abs(a-pt))
                    for apt in [a for a in near if 0<abs(a-pt)<=7680][:2]:
                        anchor=prior[(n,apt)];ar=np.array([x['row'] for x in anchor['confirmed']])
                        vi=vis[apt];ref=v.anchor_keypoints(apt,n,vi['image'],tuple(data['uv'][ar].ravel()),vi['mask_path'])
                        links=v.image_candidates(ref,kp,des,matcher)
                        anchor_reg_ids=collections.Counter(x['registration_index'] for x in anchor['confirmed']).most_common(3)
                        for link in links[:1]:
                            for ri,_ in anchor_reg_ids:
                                old=anchor['registrations'][ri]
                                regs.append(dict(atlas_id=old['atlas_id'],H=(np.asarray(link['H'])@np.asarray(old['H'])).tolist(),p95=max(link['p95'],old.get('p95',0)),inliers=min(link['inliers'],old.get('inliers',0)),method='verified_named_entity_neighbor_source_registration',anchor_pts=apt,anchor_evidence=str(OUT/'verified_entities'/f'ad{n}_{apt}.json')))
                votes=collections.defaultdict(list)
                for ri,reg in enumerate(regs):
                    ref=refs.get(reg['atlas_id'])
                    if ref is None:continue
                    try:matches,failed=v.local_evidence(ref,reg,gray,uv)
                    except (cv2.error,np.linalg.LinAlgError):continue
                    for k,reason in failed.items():fail[int(active[k])]=reason
                    negative=negatives.get(reg['atlas_id'])
                    tree=None
                    if negative is not None and matches:
                        xy=cv2.perspectiveTransform(negative[None],np.asarray(reg['H']))[0]
                        finite=np.isfinite(xy).all(1)&(np.abs(xy).max(1)<10000)
                        if finite.any():tree=cKDTree(xy[finite])
                    for x in matches:
                        k=x.pop('local_row');row=int(active[k])
                        if tree is not None:
                            dist=float(tree.query(uv[k]-np.asarray(x['local_flow']))[0])
                            if dist<x['source_reference_distance']+.4:
                                fail[row]='source_pixel_matches_omitted_fine_entity_or_its_boundary';continue
                        x['registration_index']=ri;votes[row].append(x)
                for row,choices in votes.items():
                    labels={x['path_id'] for x in choices}
                    if len(labels)>1:
                        # Repeated O, R, I and N require the neighboring word
                        # sequence, not just a locally similar stroke or ring.
                        bylabel=collections.defaultdict(list)
                        for candidate in choices:
                            reg=regs[candidate['registration_index']];ref=refs[candidate['atlas_id']]
                            score=float(v.patch_ncc(ref['gray'],gray,np.asarray(reg['H']),data['uv'][row:row+1],39,np.asarray(candidate['local_flow'])[None])[0])
                            candidate['word_context_correlation']=score;bylabel[candidate['path_id']].append(candidate)
                        ranked=sorted([(max(x['word_context_correlation'] for x in vv),label) for label,vv in bylabel.items()],reverse=True)
                        if ranked[0][0]>=.65 and ranked[0][0]-ranked[1][0]>=.08:choices=bylabel[ranked[0][1]]
                        else:conflicts.append(dict(row=row,choices=sorted(labels),word_context_scores={label:score for score,label in ranked}));continue
                    best=max(choices,key=lambda x:x.get('word_context_correlation',x['glyph_context_correlation'])+x['local_correlation'])
                    if n==10:
                        reg=regs[best['registration_index']];ref=refs[best['atlas_id']]
                        word=float(v.patch_ncc(ref['gray'],gray,np.asarray(reg['H']),data['uv'][row:row+1],39,np.asarray(best['local_flow'])[None])[0])
                        if word<.65:
                            fail[row]='repeated_SPOR_TOTO_letter_word_context_unconfirmed';continue
                        best['word_context_correlation']=word
                    confirmed.append(dict(row=row,stable_id=str(data['ids'][row]),historical_entity_hint=hints.get(row),**best))
            ok={x['row'] for x in confirmed};ambiguous={x['row'] for x in conflicts};excluded=[]
            for row in rr:
                row=int(row)
                if row in ok:continue
                reason=veto.get(row)
                if reason is None:reason=dict(reason='conflicting_actual_entity_matches' if row in ambiguous else fail.get(row,'no_verified_source_image_entity_registration_after_direct_and_neighbor_checks'))
                excluded.append(dict(row=row,stable_id=str(data['ids'][row]),**reason))
            counts=collections.Counter(x['reason'] for x in excluded);counts.update(confirmed=len(confirmed),checked=len(rr));totals.update(counts)
            b.write(dest,dict(schema='rinkspace.ad_source_entity_final.v3',version=VERSION,ad=n,pts=pt,source_frame=image,confirmed=confirmed,excluded=excluded,registrations=regs,conflicts=conflicts,counts=dict(counts),input_atlas_sha256=atlas_sha,source_identity_does_not_use_native_parameters=True))
        if frame_no%40==0:b.emit('final_entity_review_progress',frames=frame_no+1,total_frames=len(wanted),counts=dict(totals),seconds=round(time.monotonic()-start,1))
    b.emit('final_entity_review_complete',counts=dict(totals),seconds=round(time.monotonic()-start,1))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ads',type=int,nargs='+');a=p.parse_args();verify_final(a.ads)
