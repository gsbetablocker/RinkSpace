"""Recover missing pattern charts by actual-image registration from overlapping source charts."""
from pathlib import Path
import sys,json,time,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
b=r.b;OUT=r.OUT

def refine(ref,H,pt,allow_partial=False):
    im,gray,mask=r.current(pt);p=cv2.perspectiveTransform(np.array([[ref['lo'],[ref['hi'][0],ref['lo'][1]],ref['hi'],[ref['lo'][0],ref['hi'][1]]]],float),H)[0]
    lo=np.maximum(np.floor(p.min(0)-15).astype(int),[0,185]);hi=np.minimum(np.ceil(p.max(0)+16).astype(int),[1920,1080])
    if min(hi-lo)<15:return None
    T=np.eye(3);T[:2,2]=-lo;local=T@H
    warped=cv2.warpPerspective(ref['gray'],local,tuple(hi-lo),flags=cv2.INTER_LINEAR)
    valid=np.zeros(gray.shape,np.uint8);cv2.fillPoly(valid,[np.rint(p).astype(int)],255);valid[cv2.dilate(mask,np.ones((9,9),np.uint8))>0]=0;valid=valid[lo[1]:hi[1],lo[0]:hi[0]]
    actual=gray[lo[1]:hi[1],lo[0]:hi[0]]
    if np.count_nonzero(valid)<300:return None
    best=None
    for blur in [3,9]:
        a=cv2.GaussianBlur(warped,(blur,3),0).astype(np.float32)/255.;c=actual.astype(np.float32)/255.;W=np.eye(2,3,dtype=np.float32)
        try:cc,W=cv2.findTransformECC(a,c,W,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-5),valid,5)
        except cv2.error:continue
        HH=np.linalg.inv(T)@np.vstack([W,[0,0,1]])@local
        if r.geometry(HH,ref) is None:continue
        q=ref['points'][ref['probe']];displacement=np.linalg.norm(cv2.perspectiveTransform(q[None],HH)[0]-cv2.perspectiveTransform(q[None],H)[0],axis=1)
        if np.median(displacement)>25:continue
        score=r.full_pattern_score(ref,HH,pt,allow_partial=allow_partial)
        if score is None:continue
        rank=score['pattern_correlation_median']
        if best is None or rank>best['rank']:best=dict(H=HH.tolist(),rank=rank,ecc_correlation=float(cc),navigation_refinement_median_px=float(np.median(displacement)),registration_reference_horizontal_smoothing=blur,full_pattern_score=score)
    return best

def run():
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];orig={a:dict(v) for a,v in doc['records'].items()};data=b.load();proof=[];added=collections.Counter();start=time.monotonic()
    pairs=[(4,6),(71,16),(76,16),(11,16),(12,16),(67,73),(67,72)]
    samples={(4,575488),(71,507136),(76,507136),(67,465920)} if '--sample' in sys.argv else None
    for aa,bb in pairs:
        A=orig.get(str(aa),{});B=orig.get(str(bb),{});common=sorted(set(A)&set(B))
        if not common:continue
        ref=r.reference(ats[aa]);hyp=[]
        for pt in common[::max(1,len(common)//9)]:
            H=np.linalg.inv(np.asarray(B[pt]['H']))@np.asarray(A[pt]['H'])
            if r.geometry(H,ref) is not None:hyp.append((pt,H))
        for pt,row in B.items():
            if pt in doc['records'][str(aa)]:continue
            ptint=int(pt)
            if samples and (aa,ptint) not in samples:continue
            # Original row timestamps locate investigation intervals, not entity identity.
            if np.min(abs(data['pts'][data['ad']==ats[aa]['ad']]-ptint))>2560:continue
            best=None
            for seed,H in hyp:
                first=np.asarray(row['H'])@H
                if r.geometry(first,ref) is None:continue
                direct=r.full_pattern_score(ref,first,ptint)
                candidate=dict(H=first.tolist(),rank=direct['pattern_correlation_median'],full_pattern_score=direct,refinement='not_needed') if direct else None
                if candidate is None:candidate=refine(ref,first,ptint)
                if candidate is not None and (best is None or candidate['rank']>best['rank']):best=dict(candidate,atlas_id=aa,pts=ptint,method='actual_current_pattern_registered_from_source_chart_overlap',via_atlas=bb,overlap_seed_pts=int(seed),source_reference_sha256=ats[aa]['image_sha256'],source_BGR_sha256=r.VIS[ptint]['source_BGR_sha256'],native_geometry_used=False)
            if best:
                doc['records'][str(aa)][pt]=best;proof.append(best);added[aa]+=1
                if samples:
                    im=r.current(ptint)[0].copy()
                    for p in ats[aa]['paths']:
                        q=cv2.perspectiveTransform(np.asarray(p['pixels'],float)[None],np.asarray(best['H']))[0];cv2.polylines(im,[np.rint(q).astype(int)],False,(0,0,255),1)
                    cv2.imwrite(str(OUT/f'ecc_{aa}_{pt}.jpg'),im)
            if samples:print(json.dumps(dict(atlas=aa,pts=ptint,accepted=best is not None,rank=best['rank'] if best else None)),flush=True)
        print(json.dumps(dict(stage='source_overlap_pattern_recovery',atlas=aa,via=bb,new_anchors=added[aa],seconds=round(time.monotonic()-start,1))),flush=True)
    tag='sample_' if samples else ''
    b.write(OUT/(tag+'source_overlap_recovery.json'),dict(new_anchors=dict(added),proof=proof,native_geometry_used=False))
    if not samples:b.write(OUT/'source_identity_charts.json',doc)
if __name__=='__main__':run()
