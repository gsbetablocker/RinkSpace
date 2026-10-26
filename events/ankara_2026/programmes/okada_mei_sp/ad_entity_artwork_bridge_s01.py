"""Join original artwork to photographed patterns, then to current exposures."""
from pathlib import Path
import sys,json,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
import ad_source_verify_s01 as old
b=r.b;OUT=r.OUT

def run():
    cv2.setNumThreads(2);ats=b.read(b.OUT/'atlases.json')['atlases'];charts=b.read(OUT/'source_identity_charts.json');sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008);matcher=cv2.BFMatcher(cv2.NORM_L2);bridges=[];added=collections.Counter()
    for official,photos in [(71,list(range(13,18))),(73,list(range(41,67)))]:
        ref=r.reference(ats[official]);kp,des=sift.detectAndCompute(ref['gray'],None);reference=dict(keypoints=np.array([p.pt for p in kp]),descriptors=des);found=[]
        for photo in photos:
            im=r.rgba_read(ats[photo]['image']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ap=np.concatenate([np.asarray(p['pixels']) for p in ats[photo]['paths']]);lo=np.maximum(np.floor(ap.min(0)-30).astype(int),0);hi=np.minimum(np.ceil(ap.max(0)+31).astype(int),[1920,1080]);roi=np.zeros_like(gray);roi[lo[1]:hi[1],lo[0]:hi[0]]=255;key,desc=sift.detectAndCompute(gray,roi)
            if desc is None:continue
            best=None
            for reg in old.image_candidates(reference,np.array([p.pt for p in key]),desc,matcher):
                shape=r.geometry(np.asarray(reg['H']),ref)
                if shape is None or reg['inliers']<10:continue
                hull=np.asarray(reg.get('source_inlier_hull',[]))
                if len(hull)<3 or np.any(np.ptp(hull,axis=0)<(ref['hi']-ref['lo'])*np.array([.30,.22])):continue
                q=cv2.perspectiveTransform(ref['points'][ref['probe']][None],np.asarray(reg['H']))[0]
                if np.count_nonzero((q[:,0]>=lo[0])&(q[:,0]<hi[0])&(q[:,1]>=lo[1])&(q[:,1]<hi[1]))<len(q)*.65:continue
                score=reg['inliers']/(1+reg['p95'])
                if best is None or score>best['score']:best=dict(official_atlas=official,photo_atlas=photo,H=reg['H'],score=score,source_feature_evidence=reg,source_photo=ats[photo]['image'],source_photo_sha256=ats[photo]['image_sha256'],original_artwork_sha256=ats[official]['image_sha256'],geometry_is_source_navigation_only=True)
            if best is not None:found.append(best)
        if not found:continue
        bridges.extend(found);charts['records'][str(official)]={};best=max(found,key=lambda x:x['score']);im=r.rgba_read(ats[best['photo_atlas']]['image']);H=np.asarray(best['H'])
        for p in ref['atlas']['paths']:
            q=cv2.perspectiveTransform(np.asarray(p['pixels'],np.float32)[None],H)[0];cv2.polylines(im,[np.rint(q).astype(int)],False,(0,0,255),1,cv2.LINE_AA)
        q=cv2.perspectiveTransform(ref['points'][None],H)[0];lo=np.maximum(np.floor(q.min(0)-15).astype(int),0);hi=np.minimum(np.ceil(q.max(0)+16).astype(int),[1920,1080]);cv2.imwrite(str(OUT/f'artwork_bridge_{official}.png'),im[lo[1]:hi[1],lo[0]:hi[0]])
        for bridge in sorted(found,key=lambda v:v['score']):
            photo=bridge['photo_atlas'];H=np.asarray(bridge['H'])
            for pt,source in charts['records'].get(str(photo),{}).items():
                pt=int(pt);composed=np.asarray(source['H'])@H
                if r.geometry(composed,ref) is None or source.get('source_track_conflict_px',0)>8:continue
                score=r.full_pattern_score(ref,composed,pt)
                # A full-word clarity gate cannot remove clear local contours.
                # The measured current-source chart is a navigation authority;
                # per-boundary material measurements and identity checks follow.
                existing=charts['records'][str(official)].get(str(pt))
                rank=(score['pattern_correlation_median'] if score else 0)+bridge['score']*.001
                if existing and existing['bridge_rank']>=rank:continue
                charts['records'][str(official)][str(pt)]=dict(H=composed.tolist(),pts=pt,atlas_id=official,method='original_artwork_via_actual_photograph_and_current_source_chart',photo_atlas=photo,photo_chart_pts=pt,source_bridge=bridge,full_pattern_score=score,bridge_rank=rank,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],source_reference_artwork_not_observation=True)
        added[official]=len(charts['records'][str(official)])
        print(json.dumps(dict(stage='artwork_bridge',atlas=official,photographic_bridges=len(found),current_source_frames=added[official])),flush=True)
    b.write(OUT/'artwork_source_bridges.json',dict(bridges=bridges,matched_frames=dict(added),native_geometry_used=False));b.write(OUT/'source_identity_charts.json',charts)
if __name__=='__main__':run()
