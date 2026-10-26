"""Join source charts only after checking the actual reference photograph."""
from pathlib import Path
import sys,json,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
b=r.b;OUT=r.OUT

def run():
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];proof=[];orig={a:dict(v) for a,v in doc['records'].items()}
    pairs=[(1,3),(3,1),(70,77),(77,70),(4,6),(6,4),(9,75),(75,9),(76,11),(76,12),(71,16),(71,17),(72,19),(73,44)]
    for aa,bb in pairs:
        A=orig.get(str(aa),{});B=orig.get(str(bb),{});common=sorted(set(A)&set(B))
        if not common:continue
        ref=r.reference(ats[aa]);photo=r.rgba_read(ats[bb]['image']);source=(photo,cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY),np.zeros(photo.shape[:2],np.uint8));best=None
        for pt in common[::max(1,len(common)//25)]:
            H=np.linalg.inv(np.asarray(B[pt]['H']))@np.asarray(A[pt]['H'])
            if r.geometry(H,ref) is None:continue
            score=r.full_pattern_score(ref,H,0,source_image=source)
            if score is None:continue
            rank=score['pattern_correlation_median']
            if best is None or rank>best['rank']:best=dict(source_atlas=aa,target_atlas=bb,H=H.tolist(),shared_actual_exposure_pts=int(pt),rank=rank,reference_photo_check=score,source_reference_sha256=ats[aa]['image_sha256'],target_reference_sha256=ats[bb]['image_sha256'],native_geometry_used=False)
        if best is None:
            print(json.dumps(dict(join=[aa,bb],status='source_photograph_correspondence_unresolved')),flush=True);continue
        proof.append(best);h=np.asarray(best['H']);canvas=photo.copy()
        for p in ats[aa]['paths']:
            q=cv2.perspectiveTransform(np.asarray(p['pixels'],float)[None],h)[0];cv2.polylines(canvas,[np.rint(q).astype(int)],False,(0,0,255),1,cv2.LINE_AA)
        q=cv2.perspectiveTransform(ref['points'][None],h)[0];lo=np.maximum(np.floor(q.min(0)-12).astype(int),0);hi=np.minimum(np.ceil(q.max(0)+13).astype(int),[1920,1080])
        if (hi>lo).all():cv2.imwrite(str(OUT/f'join_{aa}_{bb}.jpg'),canvas[lo[1]:hi[1],lo[0]:hi[0]])
        n=0
        for pt,row in B.items():
            if pt in doc['records'][str(aa)]:continue
            H=np.asarray(row['H'])@h
            if r.geometry(H,ref) is None or row.get('source_track_conflict_px',0)>8:continue
            doc['records'][str(aa)][pt]=dict(H=H.tolist(),atlas_id=aa,pts=int(pt),method='same_ad_source_charts_joined_with_actual_reference_pattern_check',source_reference_join=best,via_atlas=bb,via_pts=int(pt),source_BGR_sha256=r.VIS[int(pt)]['source_BGR_sha256'],native_geometry_used=False)
            n+=1
        print(json.dumps(dict(join=[aa,bb],source_photo_pattern_correlation=best['rank'],added_chart_frames=n,total=len(doc['records'][str(aa)]))),flush=True)
    b.write(OUT/'source_chart_join_proofs.json',dict(joins=proof));b.write(OUT/'source_identity_charts.json',doc)
if __name__=='__main__':run()
