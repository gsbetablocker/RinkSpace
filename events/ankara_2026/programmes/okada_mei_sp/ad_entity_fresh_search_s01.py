"""Search missing source-chart coverage directly in authentic S01 exposures."""
from pathlib import Path
import sys,json,collections,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
import ad_source_verify_s01 as old
b=r.b;OUT=r.OUT
PRIMARY=[1,3,70,77,4,6,7,9,75,76,11,12,71,72,73,67,68]
def run():
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];data=b.load();sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008);matcher=cv2.BFMatcher(cv2.NORM_L2);refs={};features={};needed=collections.defaultdict(list);added=collections.Counter();start=time.monotonic()
    for ai in PRIMARY:
        ref=r.reference(ats[ai]);refs[ai]=ref;gray=ref['gray'];roi=np.zeros_like(gray)
        if ai in [71,72,73,76]:roi[:]=255
        else:
            for path in ats[ai]['paths']:
                q=np.asarray(path['pixels'],np.int32);cv2.polylines(roi,[q],False,255,25)
        key,desc=sift.detectAndCompute(gray,roi)
        if desc is None:continue
        features[ai]=dict(keypoints=np.array([k.pt for k in key]),descriptors=desc)
        present=set(map(int,doc['records'].get(str(ai),{})))
        pts=set(map(int,data['pts'][data['ad']==ats[ai]['ad']]))
        for p in list(pts):
            for dt in range(-2560,2561,256):pts.add(p+dt)
        for pt in pts:
            if pt in r.VIS and pt not in present:needed[pt].append(ai)
    for fi,pt in enumerate(sorted(needed)):
        im,gray,mask=r.current(pt);roi=np.ones_like(gray,dtype=np.uint8)*255;roi[:185]=0;roi[cv2.dilate(mask,np.ones((9,9),np.uint8))>0]=0;key,desc=sift.detectAndCompute(gray,roi)
        if desc is None:continue
        xy=np.array([k.pt for k in key])
        for ai in needed[pt]:
            best=None
            for c in old.image_candidates(features[ai],xy,desc,matcher):
                if c['inliers']<8 or r.geometry(np.asarray(c['H']),refs[ai]) is None:continue
                score=r.full_pattern_score(refs[ai],np.asarray(c['H']),pt)
                if score is None:continue
                rank=score['pattern_correlation_median']+min(c['inliers'],100)*.001
                if best is None or rank>best['rank']:best=dict(c,rank=rank,full_pattern_score=score,atlas_id=ai,pts=pt,method='fresh_actual_source_full_pattern_correspondence',source_reference_sha256=ats[ai]['image_sha256'],source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used=False)
            if best:doc['records'].setdefault(str(ai),{})[str(pt)]=best;added[ai]+=1
        if fi%60==0:print(json.dumps(dict(stage='fresh_missing_source_search',frames=fi+1,total=len(needed),new_anchors=dict(added),seconds=round(time.monotonic()-start,1))),flush=True)
    b.write(OUT/'fresh_source_search.json',dict(new_anchors=dict(added),frames=len(needed),native_geometry_used=False));b.write(OUT/'source_identity_charts.json',doc)
    print(json.dumps(dict(stage='fresh_source_search_done',new_anchors=dict(added),seconds=round(time.monotonic()-start,1))),flush=True)
if __name__=='__main__':run()
