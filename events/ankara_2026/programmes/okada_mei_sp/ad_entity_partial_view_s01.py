"""Recheck partial entering/leaving print patterns using existing source overlap."""
from pathlib import Path
import sys,json,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_ecc_s01 as e
r=e.r;b=r.b;OUT=r.OUT

def run():
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];proof=[]
    for aa,bb in [(6,7),(76,12),(71,16)]:
        A=doc['records'][str(aa)];B=doc['records'][str(bb)];common=sorted(set(A)&set(B),key=int);ref=r.reference(ats[aa]);count=0
        for pt,c in list(B.items()):
            if pt in A or int(pt)<283136:continue
            seeds=sorted(common,key=lambda s:abs(int(s)-int(pt)))[:4];best=None
            for seed in seeds:
                H=np.asarray(c['H'])@np.linalg.inv(np.asarray(B[seed]['H']))@np.asarray(A[seed]['H'])
                if r.geometry(H,ref) is None:continue
                score=r.full_pattern_score(ref,H,int(pt),allow_partial=True)
                candidate=dict(H=H.tolist(),rank=score['pattern_correlation_median'],full_pattern_score=score) if score else e.refine(ref,H,int(pt),allow_partial=True)
                if candidate and (best is None or candidate['rank']>best['rank']):best=dict(candidate,atlas_id=aa,pts=int(pt),method='partial_visible_source_pattern_with_known_same_instance_overlap',via_atlas=bb,overlap_seed_pts=int(seed),source_BGR_sha256=r.VIS[int(pt)]['source_BGR_sha256'],native_geometry_used=False,partial_view_not_whole_frame_visibility=True)
            if best:A[pt]=best;count+=1;proof.append(best)
        print(json.dumps(dict(atlas=aa,new_chart_frames=count)),flush=True)
    for pt,c in doc['records']['6'].items():
        if pt not in doc['records']['81']:doc['records']['81'][pt]=dict(c,atlas_id=81,method='authenticated_partial_left_boundary_via_same_exposure_reference')
    b.write(OUT/'source_identity_charts.json',doc);b.write(OUT/'partial_view_recovery.json',dict(proof=proof,charts_are_not_observations=True))
    for pt in [552960]:
        c=doc['records']['6'].get(str(pt))
        if c:
            im=r.current(pt)[0].copy()
            for p in ats[6]['paths']:
                q=cv2.perspectiveTransform(np.asarray(p['pixels'],float)[None],np.asarray(c['H']))[0];cv2.polylines(im,[np.rint(q).astype(int)],False,(0,0,255),1)
            cv2.imwrite(str(OUT/f'qa/partial_{pt}.jpg'),im[350:650])
if __name__=='__main__':run()
