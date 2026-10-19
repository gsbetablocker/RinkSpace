"""Image-measured evidence for the 69 camera-evidenced low-end frames.

No geometric projection is used as a measurement. This script keeps matching
diagnostics and all failures, and only constructs conditional pose initializers.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = _resolve_asset(ROOT, ROOT / 'outputs/calibration/ankara_2026_v015_joint')
POOL = BASE / 'women_sp_training_frames_1000'
OUT = BASE / 'joint_279/low'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def low_frames():
    manifest = load(POOL / 'training_input_manifest.json')
    return [f for f in manifest['frames'] if f['camera_id'] == 'low_end']


def image_for(frame):
    path = POOL / frame['native_image_path'] if 'native_image_path' in frame else Path(frame['image_path'])
    image = cv2.imread(str(path))
    if image is None or image.shape[:2] != (1080, 1920):
        raise ValueError(f'Native source image unavailable: {path}')
    return image


def sift(gray, polygon=None, board_band=False):
    mask = np.zeros(gray.shape, np.uint8)
    if polygon is not None:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    elif board_band:
        mask[170:780] = 255
        mask[:350, :520] = 0  # broadcast graphics never provide scene matches
    else:
        mask[:] = 255
    detector = cv2.SIFT_create(nfeatures=4500, contrastThreshold=.014, edgeThreshold=15, sigma=1.4)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    return np.array([kp.pt for kp in keypoints], np.float32), descriptors


def homography_match(source, destination):
    a, da = source; b, db = destination
    if da is None or db is None or len(da) < 6 or len(db) < 6:
        return None
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < .72 * n.distance]
    if len(good) < 8:
        return None
    # Distinct target keypoints are needed: repeated text cannot vote twice.
    dedup = {}
    for m in sorted(good, key=lambda m: m.distance):
        dedup.setdefault(m.trainIdx, m)
    good = list(dedup.values())
    if len(good) < 8:
        return None
    p = np.float32([a[m.queryIdx] for m in good]); q = np.float32([b[m.trainIdx] for m in good])
    h, mask = cv2.findHomography(p, q, cv2.RANSAC, 2.5, maxIters=5000, confidence=.999)
    if h is None:
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < 8 or keep.mean() < .45:
        return None
    p, q = p[keep], q[keep]
    pred = cv2.perspectiveTransform(p.reshape(-1, 1, 2), h).reshape(-1, 2)
    error = np.linalg.norm(pred-q, axis=1)
    if np.percentile(error, 90) > 2.5 or np.ptp(p[:, 0]) < 40 or np.ptp(p[:, 1]) < 10:
        return None
    return {'H': h.tolist(), 'matched_source_uv': p.tolist(), 'matched_target_uv': q.tolist(),
            'inliers': int(len(p)), 'candidate_matches': len(good), 'median_px': float(np.median(error)),
            'p90_px': float(np.percentile(error, 90))}


def match_existing_advertisements():
    problem = load(BASE / 'fits/joint_ads111_start0/problem.json')
    old_frames = {f['id']: f for f in problem['frames']}
    definitions = [
        ('advert_ppt_object_115', 'main_69427200', [[372,324],[783,324],[783,410],[372,410]]),
        ('advert_ppt_object_116', 'main_69427200', [[1046,312],[1456,312],[1456,396],[1046,396]]),
        ('advert_ppt_object_111', 'main_64458240', [[1173,456],[1562,459],[1562,518],[1169,515]]),
    ]
    templates = []
    by_landmark = {p['id']: p for p in problem['board_landmarks']}
    for instance, fid, polygon in definitions:
        gray = cv2.cvtColor(image_for(old_frames[fid]), cv2.COLOR_BGR2GRAY)
        observations = [o for o in problem['observations'] if o['frame_id']==fid and
                        o['target']['kind']=='board_landmark' and
                        by_landmark[o['target']['landmark_id']]['advertisement_instance_id']==instance]
        templates.append((instance, fid, polygon, sift(gray, polygon), observations))
    results = []
    for frame in low_frames():
        gray = cv2.cvtColor(image_for(frame), cv2.COLOR_BGR2GRAY)
        dest = sift(gray, board_band=True)
        for instance, fid, polygon, source, observations in templates:
            result = homography_match(source, dest)
            if result is None:
                continue
            h = np.array(result['H'])
            quad = cv2.perspectiveTransform(np.float32(polygon).reshape(-1,1,2),h).reshape(-1,2)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 700:
                continue
            if quad[:,0].min() < -100 or quad[:,0].max()>2020 or quad[:,1].min()<160 or quad[:,1].max()>800:
                continue
            tracked=[]
            for o in observations:
                uv=cv2.perspectiveTransform(np.float32(o['pixels_uv']).reshape(-1,1,2),h).reshape(2)
                tracked.append({'landmark_id':o['target']['landmark_id'],'pixels_uv':uv.tolist(),
                                'reference_observation_id':o['id']})
            results.append(dict(frame_id=frame['id'],ordinal=frame['ordinal'],source_time=frame['source_time'],
                advertisement_instance_id=instance,reference_frame_id=fid,
                projected_image_patch_corners=quad.tolist(),mapped_old_details=tracked,**result))
        print(frame['ordinal'], 'matched',len([r for r in results if r['frame_id']==frame['id']]),flush=True)
    save(OUT/'existing_advertisement_image_matches.json',{'matches':results,
        'method':'Native-image SIFT correspondences and planar image-patch homography; no scene projection.',
        'status':'candidate_image_matches_require_instance_and_overlay_review'})


def remeasure():
    from extract_279_main_observations import image_features, match, measure_observation, target_key
    from rinkspace.calibration_companion import Scene, Camera, look_at_rvec
    from rinkspace.calibration_companion.joint_scene import SharedGeometry
    from scipy.optimize import least_squares
    cv2.setNumThreads(2);cv2.setRNGSeed(91)
    problem=load(BASE/'fits/joint_ads111_start0/problem.json')
    fitted=load(BASE/'fits/joint_ads111_start0/fit.json')
    estim={p['id']:p for p in fitted['board_landmarks']}
    fitcam={f['id']:f['camera'] for f in fitted['frames']}
    by=defaultdict(list)
    for o in problem['observations']:
        if o['target'].get('landmark_id','').startswith('advert_ppt_object_108'):continue
        by[o['frame_id']].append(o)
    seeds=[]
    for f in problem['frames']:
        f=copy.deepcopy(f);f['measurements']=by[f['id']];f['camera']=fitcam[f['id']];seeds.append(f)
    timeline=load(_resolve_asset(ROOT, ROOT/'outputs/camera_classification/trial_010800_013600/result.json'))['intervals']
    frames=[]
    for row in low_frames():
        path=POOL/row['native_image_path']
        shot=next((s['id'] for s in timeline if s['start']<=row['source_seconds']<s['end']),'user_low_905_947')
        f=dict(id=f"low279_{row['source_pts']}",source_id=row['source_id'],media_track_id=row['media_track_id'],
            source_path=str(ROOT/row['source_path']),source_sha256=row['source_sha256'],time_base='1/15360',stream_index=0,
            source_pts=row['source_pts'],source_seconds=row['source_seconds'],source_time=row['source_time'],
            image_path=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            image_size_wh=[1920,1080],principal_point_px=[959.5,539.5],fy_over_fx=1.,distortion_opencv=[0.]*5,
            focal_bounds_px=[450.,30000.],shot_id=shot,camera_id='camera_low_end',physical_camera_id='camera_low_end',installation_id='women_sp_low',
            camera_assignment_authority=row['camera_identity_authority'],camera_identity_source=row['camera_identity_source'],
            pixel_frame_id=f"{row['media_track_id']}:pts:{row['source_pts']}:1920x1080",pool_ordinal=row['ordinal'],
            previously_heldout_promoted=shot in ['S074','S085','S087','S089'])
        frames.append(f)
    with ThreadPoolExecutor(max_workers=4) as executor:
        src=list(executor.map(image_features,seeds));dest=list(executor.map(image_features,frames))
    observations=[];status=[];correspondences=[]
    old_landmarks={p['id']:p for p in problem['board_landmarks']}
    for index,t in enumerate(dest):
        frame=t['entry'];best={};logs=[]
        for s in src:
            proposal=match(s,t)
            if proposal is None or proposal['inliers']<16:continue
            logs.append(dict(seed_frame_id=s['entry']['id'],inliers=proposal['inliers'],H_half=proposal['H'].tolist(),
                             span_half_px=proposal['span_half_px'],p95_half_px=proposal['p95_half_px']))
            for obs in s['entry']['measurements']:
                result=measure_observation(s,t,obs,proposal)
                if result is None:continue
                points,quality=result;key=target_key(obs)
                score=len(points)+quality.get('ncc_min',quality.get('ncc_median',0))*10
                if key not in best or score>best[key]['score']:
                    best[key]=dict(points=points,quality=quality,obs=obs,seed=s['entry'],score=score)
        local=[]
        for j,item in enumerate(best.values()):
            old=item['obs'];target=old['target']
            # Repeated ISU logos produced visually confirmed false matches in
            # this batch. Their 111 identities cannot be inherited here.
            if target.get('landmark_id','').startswith('advert_ppt_object_111'):continue
            row=dict(id=f"{frame['id']}_measured_{j:03d}",frame_id=frame['id'],type=old['type'],target=copy.deepcopy(target),
                physical_group_id=old.get('physical_group_id',target_key(old)),use='fit',
                pixels_uv=np.round(item['points'],4).tolist(),sigma_px=max(float(old.get('sigma_px',3)),4. if old['type']=='point' else 6.),
                measurement_origin='actual_image_remeasurement_from_traceable_seed_identity',
                method='mutual RootSIFT image correspondence followed by signed edge or bidirectional unique patch remeasurement',
                source_seed_observation_id=old['id'],source_seed_frame_id=item['seed']['id'],
                source_seed_pts=item['seed']['source_pts'],measurement_diagnostics=item['quality'],
                synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                correlation_group_id=f"{frame['shot_id']}:{old.get('physical_group_id',target_key(old))}")
            if target['kind']=='board_landmark':row['advertisement_instance_id']=old_landmarks[target['landmark_id']]['advertisement_instance_id']
            local.append(row)
        observations.extend(local)
        status.append(dict(frame_id=frame['id'],ordinal=frame['pool_ordinal'],source_time=frame['source_time'],shot_id=frame['shot_id'],
            measured_observation_groups=len(local),targets=[o['target'] for o in local],source_matches=len(logs),
            previous_holdout_promoted=frame['previously_heldout_promoted'],
            status='measured_pending_visual_review' if local else 'no_reliable_measurement'))
        correspondences.append(dict(frame_id=frame['id'],proposals=logs))
        print(index+1,frame['pool_ordinal'],'groups',len(local),'matches',len(logs),flush=True)
    save(OUT/'remeasured_candidates.json',dict(frames=frames,observations=observations,board_landmarks=problem['board_landmarks'],
        status='pending_visual_identity_and_edge_review',source_manifest=str(POOL/'training_input_manifest.json')))
    save(OUT/'frame_status.json',status);save(OUT/'crossview_image_correspondences.json',correspondences)
    print('measured',sum(s['measured_observation_groups']>0 for s in status),'groups',len(observations),flush=True)


def color_candidates():
    records=[];panels=[]
    for frame in low_frames():
        image=image_for(frame);hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        found=[]
        for color,lo,hi in [('purple',130,179),('blue',95,137)]:
            mask=((hsv[:,:,0]>=lo)&(hsv[:,:,0]<=hi)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)&(hsv[:,:,2]<245)).astype(np.uint8)*255
            mask[:210]=0;mask[820:]=0;mask[:325,:520]=0
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,13),np.uint8))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,w,h=cv2.boundingRect(contour)
                if w<110 or h<22 or h>280 or w/h<2.5 or cv2.contourArea(contour)<w*h*.48:continue
                hull=cv2.convexHull(contour);poly=cv2.approxPolyDP(hull,.022*cv2.arcLength(hull,True),True).reshape(-1,2)
                found.append(dict(color=color,bbox_xywh=[x,y,w,h],contour=contour.reshape(-1,2).tolist(),polygon=poly.tolist()))
        found.sort(key=lambda r:(r['bbox_xywh'][0],r['color']))
        for j,c in enumerate(found):
            c.update(ordinal=frame['ordinal'],candidate_index=j,source_time=frame['source_time'],frame_id=frame['id'])
            x,y,w,h=c['bbox_xywh'];crop=image[max(0,y-8):min(1080,y+h+8),max(0,x-8):min(1920,x+w+8)]
            factor=min(390/crop.shape[1],100/crop.shape[0]);small=cv2.resize(crop,None,fx=factor,fy=factor)
            panel=np.zeros((140,400,3),np.uint8);panel[:small.shape[0],:small.shape[1]]=small
            cv2.putText(panel,f"{frame['ordinal']}:{j} {c['color']} verts={len(c['polygon'])}",(5,125),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            panels.append(panel);records.append(c)
    save(OUT/'color_candidates.json',records)
    for page in range((len(panels)+35)//36):
        rows=panels[page*36:(page+1)*36]
        while len(rows)%4:rows.append(np.zeros((140,400,3),np.uint8))
        cv2.imwrite(str(OUT/f'color_candidates_{page+1}.jpg'),np.vstack([np.hstack(rows[i:i+4]) for i in range(0,len(rows),4)]))
    print('candidates',len(records),flush=True)


# Visually reviewed source regions; these are image ROIs and surface identities,
# never model-coordinate measurements. The podium + APP/official-logo sequence
# distinguishes the right end from repeated logos at the opposite end.
UPPER_ROIS={108:(50,850),113:(650,1750),114:(800,1800),901:(200,1700),902:(50,800),
    916:(300,1800),918:(100,850),933:(100,1600),934:(100,1450),952:(100,1400),953:(200,1400),
    1004:(150,1600),1005:(100,1500),1007:(650,1900),1028:(50,850),1029:(100,1400),1030:(100,1200),
    1031:(100,1600),1046:(20,650),1047:(400,1100),1095:(200,1600),1135:(150,650),1136:(300,1350),
    1179:(100,430),1182:(100,1400),1206:(200,1650),1214:(200,1900),1215:(100,1500),1216:(100,1500),
    1224:(50,1150),1225:(100,1600),1277:(100,1500),1278:(800,1800),1287:(50,1650),1288:(500,1700),
    1289:(100,750),1290:(100,720),1291:(100,720),1301:(100,1700),1320:(40,500)}
RIGHT_ROIS={109:(150,1750),110:(400,1500),111:(750,1850),112:(100,850),115:(400,1650),116:(350,1600),
    903:(200,1700),904:(150,1700),915:(400,1750),1000:(100,1700),1006:(100,1150),1096:(100,1150),
    1118:(300,1750),1119:(350,1750),1134:(600,1850),1205:(800,1850),1222:(700,1800),1276:(100,1150),1286:(1000,1820),
    917:(780,1260),930:(50,1000)}
# Judging-side ISU is a different physical instance from 111. These small
# visible cap segments are assigned to the right/lower turn by the observed
# APP -> official logos -> SPORTOTO -> can -> WYNDHAM -> ISU -> judges order.
# They supply surface curves only, never same-name advertisement point matches.
LOWER_RIGHT_ROIS={928:(1100,1850),929:(400,1000),1003:(1610,1900),1117:(1350,1850),1285:(650,1700),1300:(1180,1570)}
# (pool ordinal, segmented region index, physical advert, actual visible corners)
COLOR_REVIEW=[(108,0,116,[0,1,2,3]),(108,1,117,[1,2]),(111,0,117,[1,2]),
    (113,0,110,[1,2]),(113,1,109,[0,1,2,3]),(117,0,117,[0,3]),(901,0,109,[1,2]),
    (902,0,116,[0,3]),(902,2,117,[0,1,2,3]),(915,0,117,[1,2]),(918,0,116,[0,1,2,3]),
    (918,1,117,[1,2]),(934,1,116,[0,1,2,3]),(1007,1,110,[1,2]),
    (1028,0,115,[0,1,2,3]),(1028,1,116,[0,1,2,3]),(1028,2,117,[1,2]),
    (1030,0,109,[0,3]),(1046,0,117,[1,2]),(1047,1,116,[0,3]),(1047,2,117,[0]),
    (1134,0,117,[1]),(1214,0,109,[0,1,2,3]),(1215,0,109,[0,1,2,3]),
    (1216,0,109,[0,1,2,3]),(1222,0,117,[1,2]),(1224,1,117,[0]),
    (1277,0,110,[0,1,2,3]),(1278,0,109,[1,2]),(1287,0,109,[0,1,2,3]),
    (1288,1,110,[0,1,2,3]),(1289,0,116,[0,1,2,3]),(1290,0,116,[0,1,2,3]),
    (1291,0,116,[0,1,2,3]),(1301,3,116,[0,1,2,3]),(1183,1,111,[1,2])]


def ordered_quad(contour):
    p=np.asarray(contour,np.float32).reshape(-1,2)
    return p[[np.argmin(p.sum(1)),np.argmax(p[:,0]-p[:,1]),np.argmax(p.sum(1)),np.argmax(p[:,1]-p[:,0])]]


def refine_printed_quad(image,record,advert):
    if advert==110:return ordered_quad(record['contour'])
    x,y,w,h=record['bbox_xywh'];x0=max(0,x-14);x1=min(1920,x+w+15);y0=max(0,y-12);y1=min(1080,y+h+95)
    hsv=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
    hue,sat,value=cv2.split(hsv);lo=120 if advert==116 else 90
    mask=((hue>=lo)&(hue<=179)&(sat>14)&(value>30)&(value<247)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    eligible=[]
    for c in contours:
        xx,yy,ww,hh=cv2.boundingRect(c)
        if ww<w*.55 or hh<h*.5:continue
        eligible.append(c)
    if not eligible:return ordered_quad(record['contour'])
    contour=max(eligible,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
    return ordered_quad(contour)


def direct_board_edges(image,roi):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);x0,x1=roi
    blue=((hsv[:,:,0]>=90)&(hsv[:,:,0]<=150)&(hsv[:,:,1]>45)&(hsv[:,:,2]>30)&(hsv[:,:,2]<225)).astype(np.uint8)*255
    blue[:80]=0;blue[850:]=0;blue[:,:x0]=0;blue[:,x1:]=0
    lines=cv2.HoughLinesP(blue,1,np.pi/1800,threshold=60,minLineLength=max(140,int((x1-x0)*.45)),maxLineGap=40)
    eligible=[]
    if lines is not None:
        for a,b,c,d in lines.reshape(-1,4):
            if abs(c-a)<100 or abs(d-b)/abs(c-a)>.13:continue
            y=(b+d)/2.;eligible.append((-abs(c-a),y,[a,b,c,d]))
    if not eligible:return {}
    candidate_tops=[]
    for _,_,line in sorted(eligible)[:100]:
        a,b,c,d=line;slope=(d-b)/(c-a);points=[]
        for x in np.linspace(x0+8,x1-8,50).astype(int):
            expected=b+slope*(x-a);lo=max(80,int(expected)-32);hi=min(850,int(expected)+33)
            ys=np.flatnonzero(blue[lo:hi,x])
            if not len(ys):continue
            runs=np.split(ys,np.where(np.diff(ys)>1)[0]+1)
            runs=[r for r in runs if r[0]>0 and r[-1]<hi-lo-1 and len(r)<=42]
            if not runs:continue  # same-colour podium is not the board-cap edge
            ys=min(runs,key=lambda r:abs(lo+(r[0]+r[-1])/2-expected))
            points.append([float(x),float(ys[0]+lo)])
        if points:candidate_tops.append(points)
    if not candidate_tops:return {}
    # A long filled advertisement block is not a cap: it has no compact
    # colour band at the sampled columns. Select by actual compact support.
    top=max(candidate_tops,key=lambda p:(len(p),np.ptp(np.asarray(p)[:,0])))
    bottom=[]
    for x,y in top:
        x=int(x);y=int(y)
        yy=np.arange(min(y+25,1060),min(y+470,1061))
        if not len(yy):continue
        h,s,v=hsv[yy,x].T
        gold=(h>=10)&(h<=43)&(s>=27)&(v>=60)
        runs=np.split(np.flatnonzero(gold),np.where(np.diff(np.flatnonzero(gold))>1)[0]+1)
        runs=[r for r in runs if 2<=len(r)<=22]
        if runs:
            r=max(runs,key=lambda r:float(np.mean(s[r])))
            bottom.append([float(x),float(yy[r[-1]]+1)])
    result={}
    for edge,points in [('top',top),('contact',bottom)]:
        p=np.asarray(points,np.float32)
        if len(p)<12:continue
        # Robust native-pixel line support filters people, printed logos, and
        # ice reflections. No predicted model line enters this measurement.
        vx,vy,x,y=cv2.fitLine(p,cv2.DIST_HUBER,0,.01,.01).ravel()
        error=np.abs((p[:,0]-x)*vy-(p[:,1]-y)*vx)
        keep=error<4.5
        if keep.sum()<12 or np.ptp(p[keep,0])<(x1-x0)*.4:continue
        result[edge]=p[keep]
    return result


def package():
    from rinkspace.calibration_companion import Scene, Camera, look_at_rvec
    from rinkspace.calibration_companion.board_intersection import intersect_board_surface
    from scipy.optimize import least_squares
    cv2.setNumThreads(2)
    data=load(OUT/'remeasured_candidates.json');frames=data['frames'];byord={f['pool_ordinal']:f for f in frames}
    oldproblem=load(BASE/'fits/joint_ads111_start0/problem.json');oldfit=load(BASE/'fits/joint_ads111_start0/fit.json')
    scene=load(BASE/'fits/joint_ads111_start0/scene_semantic.json');pars=scene['parameters']
    oldframes={f['id']:f for f in oldproblem['frames']};oldcameras={f['id']:f['camera'] for f in oldfit['frames']}
    knownxyz={p['id']:p['estimated_xyz'] for p in oldfit['board_landmarks']}
    landmarks={p['id']:copy.deepcopy(p) for p in data['board_landmarks']}
    observations=[];review_rejections=[]
    # Only printed details survived visual review. Two weak circle proposals
    # and inherited end-side labels were not strong enough to retain.
    for o in data['observations']:
        if o['target']['kind']=='board_landmark':observations.append(o)
        else:review_rejections.append(dict(observation_id=o['id'],target=o['target'],reason='ice arc too weak or repeated-end source identity; directly remeasure identified board instead'))
    records={(r['ordinal'],r['candidate_index']):r for r in load(OUT/'color_candidates.json')}
    corner_pixels=[]
    for ordinal,index,advert,corners in COLOR_REVIEW:
        r=records[(ordinal,index)];quad=refine_printed_quad(cv2.imread(byord[ordinal]['image_path']),r,advert)
        for corner in corners:
            uv=quad[corner]
            if min(uv[0],1919-uv[0])<4:continue
            corner_pixels.append((byord[ordinal],advert,corner,uv,{'segmented_candidate_index':index,'color':r['color']}))
    # Existing main-camera exposure gives direct cross-camera observations of
    # the same advertisement corners. Image-coordinate ROIs are hand reviewed.
    anchor=copy.deepcopy(oldframes['main_69427200'])
    anchor['physical_camera_id']='camera_main_long_side';anchor['camera_id']='camera_main_long_side'
    native=cv2.imread(anchor['image_path']);hsv=cv2.cvtColor(native,cv2.COLOR_BGR2HSV)
    for advert,roi,color,corners in [(115,(365,315,790,415),'blue',[0,1,2,3]),(116,(1035,305,1465,400),'purple',[0,1,2,3]),
                                    (117,(1660,308,1920,390),'blue',[0,3])]:
        x0,y0,x1,y1=roi;h,s,v=cv2.split(hsv[y0:y1,x0:x1])
        mask=((h>=(120 if advert==116 else 95))&(h<=178)&(s>14)&(v>35)&(v<247)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        contour=max(contours,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
        quad=ordered_quad(contour)
        for corner in corners:corner_pixels.append((anchor,advert,corner,quad[corner],{'native_image_color_boundary_anchor':True}))
    frame_ids={f['id'] for f in frames}
    if corner_pixels:frames.append(anchor);frame_ids.add(anchor['id'])
    # The left-corner ISU in 1183 is identified by its translucent wall and
    # spectator-side neighbourhood. ISU beside the judging table is distinct.
    left_anchor=copy.deepcopy(oldframes['main_64458240'])
    left_anchor.update(physical_camera_id='camera_main_long_side',camera_id='camera_main_long_side')
    im=cv2.imread(left_anchor['image_path']);hsv=cv2.cvtColor(im[449:528,1160:1575],cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,0]>=95)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>30)).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        quad=ordered_quad(max(contours,key=cv2.contourArea).reshape(-1,2)+[1160,449])
        for corner in range(4):corner_pixels.append((left_anchor,111,corner,quad[corner],{'native_image_color_boundary_anchor':True}))
        frames.append(left_anchor)
    along_centers={109:-37.,110:-65.,111:-90.,115:46.,116:70.,117:92.}
    for frame,advert,corner,uv,diagnostic in corner_pixels:
        instance=f'advert_ppt_object_{advert}';name='blue_corner' if advert==109 else 'color_corner';lid=f'{instance}_{name}_{corner}'
        along=along_centers[advert]+([-2.,2.,2.,-2.][corner] if advert==109 else [-9.,9.,9.,-9.][corner])
        if advert==117:along=min(along,97.)
        frac=.75 if corner<2 else .2
        if lid not in landmarks:
            spec=dict(id=lid,advertisement_instance_id=instance,side='upper_long_side',initial_along=along,
                bounds_along=[-999.,999.],initial_z_fraction=frac,bounds_z_fraction=[.01,.99],fit=True,evidence_observation_ids=[])
            if advert in (116,117):spec['allow_corner']='upper_right'
            if advert==111:spec['allow_corner']='upper_left'
            # Initial height/along guesses do not constrain their fitted values.
            spec['provenance']={'identity':'user-numbered physical instance plus visually reviewed neighbouring graphics/background',
                'initialization':'schematic numerical guess only; no XYZ observation or coordinate prior',
                'corner_order':'graphic top-left, top-right, bottom-right, bottom-left'}
            landmarks[lid]=spec
        if frame['id'] in oldcameras:
            c=oldcameras[frame['id']];camera=Camera(c['center_xyz'],c['rvec_world_to_camera'],c['K_px'],c.get('distortion_opencv',[]))
            hit=intersect_board_surface(camera,uv,'upper_long_side',pars,allow_corner=landmarks[lid].get('allow_corner'))
            if hit['valid']:
                landmarks[lid]['initial_along']=float(hit['xyz'][0]);landmarks[lid]['initial_z_fraction']=float(np.clip(hit['xyz'][2]/pars['board_height'],.01,.99))
                knownxyz[lid]=hit['xyz'];landmarks[lid]['provenance']['conditional_ray_initializer']=hit
        oid=f"{frame['id']}_{lid}"
        observations.append(dict(id=oid,frame_id=frame['id'],type='point',use='fit',target={'kind':'board_landmark','landmark_id':lid},
            physical_group_id=instance,advertisement_instance_id=instance,pixels_uv=[np.asarray(uv,float).tolist()],sigma_px=5.0,
            measurement_origin='direct_native_image_printed_color_boundary',synthetic_or_model_projected=False,
            measurement_diagnostics=diagnostic,correlation_group_id=f"low279:{instance}:color_corners"))
    board_counts=Counter()
    for frame in frames:
        ordinal=frame.get('pool_ordinal')
        side=('upper_long_side' if ordinal in UPPER_ROIS else 'right_end' if ordinal in RIGHT_ROIS
              else 'lower_right' if ordinal in LOWER_RIGHT_ROIS else None)
        if side is None:continue
        roi=({'upper_long_side':UPPER_ROIS,'right_end':RIGHT_ROIS,'lower_right':LOWER_RIGHT_ROIS}[side])[ordinal]
        for edge,points in direct_board_edges(cv2.imread(frame['image_path']),roi).items():
            if ordinal in LOWER_RIGHT_ROIS and edge=='contact':
                review_rejections.append(dict(frame_id=frame['id'],target={'kind':'board_side','side':side,'edge':edge},
                    reason='independent native-image review: gold colour continues into ice reflection, contact boundary uncertain'))
                continue
            target={'kind':'board_side','side':side,'edge':edge}
            if side=='lower_right':target.update(side='right_end',continue_corner='lower_right')
            observations.append(dict(id=f"{frame['id']}_direct_{side}_{edge}",frame_id=frame['id'],type='curve',use='fit',
                target=target,physical_group_id=f'{side}_{edge}',pixels_uv=points.tolist(),sigma_px=6. if edge=='top' else 10.,
                measurement_origin='native_image_color_edge_measurement_no_geometry_projection',
                method='blue board cap upper boundary / yellow kickplate lower boundary; native color segmentation and robust local line support',
                identity_evidence={'reviewed_source_roi_x':list(roi),'context':'spectator-side sequence, APP/official-logo right-end sequence, or distinct judging-side right/lower turn',
                    'specific_straight_vs_arc_segment_resolved':side!='lower_right',
                    'continuous_region_only':side=='lower_right'},
                correlation_group_id=f"{frame['shot_id']}:{side}:{edge}"))
            board_counts[edge]+=1
    by=defaultdict(list)
    for o in observations:by[o['frame_id']].append(o)
    # Conditional initialization only. The joint optimizer will freely change
    # this low-camera installation center, scene and per-frame optics.
    center=np.array([-120.,-32.,4.]);K=np.array([[5000.,0.,959.5],[0.,5000.,539.5],[0.,0.,1.]])
    for frame in frames:
        if frame['id'] in oldcameras:continue
        local=by[frame['id']];controls=[]
        for o in local:
            lid=o['target'].get('landmark_id')
            if not lid:continue
            p=landmarks[lid]
            xyz=knownxyz.get(lid,[p['initial_along'],50.,p['initial_z_fraction']*pars['board_height']])
            controls.append((xyz,o['pixels_uv'][0]))
        side=('right_end' if frame['pool_ordinal'] in RIGHT_ROIS else 'lower_right' if frame['pool_ordinal'] in LOWER_RIGHT_ROIS else 'upper_long_side')
        aim=np.mean([p[0] for p in controls],axis=0) if controls else np.array({'right_end':[100.,10.,1.7],
            'lower_right':[90.,-40.,1.7],'upper_long_side':[0.,50.,1.7]}[side])
        rot=look_at_rvec(center,aim)
        edges={o['target'].get('edge'):np.asarray(o['pixels_uv']) for o in local if o['target']['kind']=='board_side'}
        focal=6000.
        if 'top' in edges and 'contact' in edges:
            gap=np.median(edges['contact'][:,1])-np.median(edges['top'][:,1])
            if gap>10:focal=float(np.clip(gap*np.linalg.norm(aim-center)/pars['board_height'],450,30000))
        initial=np.r_[rot,np.log(focal)]
        if controls:
            xyz=np.asarray([p[0] for p in controls]);uv=np.asarray([p[1] for p in controls])
            def error(x):
                k=K.copy();k[0,0]=k[1,1]=np.exp(x[3]);cam=Camera(center,x[:3],k,[])
                projected,depth=cam.project(xyz)
                return np.r_[(projected-uv).ravel()/5.,np.minimum(depth,0)*100]
            result=least_squares(error,initial,bounds=([-np.inf]*3+[np.log(450)], [np.inf]*3+[np.log(30000)]),max_nfev=70,loss='soft_l1')
            initial=result.x
        else:
            # Aim the observed board's midpoint at its real image location.
            if edges:
                mean=np.mean(np.concatenate(list(edges.values())),axis=0)
                from fit_scene3d_real_keyframes import align_z_to
                R=align_z_to(np.r_[(mean-[959.5,539.5])/focal,1.])@cv2.Rodrigues(rot)[0]
                initial[:3]=cv2.Rodrigues(R)[0].ravel()
        frame.update(initial_rvec_world_to_camera=initial[:3].tolist(),initial_focal_px=float(np.exp(initial[3])),
            initialization_diagnostics={'method':'conditional board/ad geometry and free low-camera numerical center; initializer only',
                'initial_center_xyz':center.tolist(),'not_measured_camera_pose':True,'control_count':len(controls),
                'weak_initialization':len(controls)<4},fit_status='actual_measurements_available' if local else 'no_reliable_measurement_yet')
    for lid,p in landmarks.items():p['evidence_observation_ids']=[o['id'] for o in observations if o['target'].get('landmark_id')==lid]
    used_lids={o['target']['landmark_id'] for o in observations if o['target']['kind']=='board_landmark'}
    out=dict(schema_version='rinkspace.actual_low279_observations.v1',frames=frames,observations=observations,
        board_landmarks=[p for lid,p in landmarks.items() if lid in used_lids],
        installations=[dict(id='women_sp_low',physical_camera_id='camera_low_end',initial_center_xyz=center.tolist(),fit_center=True,
            center_bounds_xyz=[[-500.,-200.,.1],[-100.001,200.,100.]],initialization_status='conditional_free_guess_not_measured')],
        source_manifest=str(POOL/'training_input_manifest.json'),measurement_status='actual_pixels_with_visual_surface_review; not a fit result',
        rejected_candidates=review_rejections,summary={'selected_low_frames':69,'low_frames_with_actual_observations':sum(bool(by[f['id']]) for f in frames if f.get('pool_ordinal')),
            'observation_groups':len(observations),'board_edge_frame_counts':dict(board_counts),'model_fit_run':False})
    save(OUT/'observations.json',out);save(OUT/'frame_status.json',[dict(frame_id=f['id'],ordinal=f.get('pool_ordinal'),source_seconds=f['source_seconds'],
        observations=len(by[f['id']]),status=f.get('fit_status'),targets=[o['target'] for o in by[f['id']]]) for f in frames])
    (OUT/'overlays').mkdir(exist_ok=True);panels=[]
    for frame in frames:
        if not frame.get('pool_ordinal'):continue
        image=cv2.imread(frame['image_path'])
        for j,o in enumerate(by[frame['id']]):
            col=[(0,255,255),(80,255,80),(255,100,230),(255,200,30)][j%4]
            for p in o['pixels_uv']:cv2.circle(image,tuple(np.rint(p).astype(int)),4,col,-1)
        cv2.rectangle(image,(0,0),(1920,55),(15,15,15),-1)
        cv2.putText(image,f"LOW #{frame['pool_ordinal']} {frame['source_time']} groups={len(by[frame['id']])}",(15,40),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{frame['pool_ordinal']:05d}.jpg"),image)
        panels.append(cv2.resize(image,(480,270)))
    for page in range((len(panels)+11)//12):
        rows=panels[page*12:(page+1)*12]
        while len(rows)%3:rows.append(np.zeros((270,480,3),np.uint8))
        cv2.imwrite(str(OUT/f'measurements_{page+1:02d}.jpg'),np.vstack([np.hstack(rows[i:i+3]) for i in range(0,len(rows),3)]))
    print(json.dumps(out['summary']),flush=True)


def proof_images():
    """Traceable same-detail native-image panels, never projected geometry."""
    from PIL import Image, ImageDraw, ImageFont
    data=load(OUT/'observations.json');frames={f['id']:f for f in data['frames']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    byord={f['pool_ordinal']:f for f in data['frames'] if f.get('pool_ordinal')}
    colors=[(255,220,0),(50,245,100),(30,205,255),(255,70,205)]
    for advert,low_ordinal,main_id,label in [(116,108,'main_69427200','用户编号2：ISU网站'),
        (115,1028,'main_69427200','用户编号3：买取大吉'),(111,1183,'main_64458240','用户编号7：左上角ISU'),
        (117,902,'main_69427200','用户编号1：ISU APP')]:
        canvas=Image.new('RGB',(1920,940),(20,20,24));draw=ImageDraw.Draw(canvas)
        draw.text((20,8),label+'｜同色字母表示同一块广告上的同一印刷角点',font=font,fill='white')
        pair=[frames[main_id],byord[low_ordinal]]
        for col,frame in enumerate(pair):
            im=cv2.imread(frame['image_path']);points=[]
            for o in data['observations']:
                lid=o['target'].get('landmark_id','')
                if o['frame_id']!=frame['id'] or not lid.startswith(f'advert_ppt_object_{advert}_color_corner_'):continue
                index=int(lid.rsplit('_',1)[1]);uv=np.asarray(o['pixels_uv'][0]);points.append(uv)
                cv2.circle(im,tuple(np.rint(uv).astype(int)),9,colors[index][::-1],2)
                cv2.putText(im,chr(65+index),tuple(np.rint(uv+[10,-10]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1.,colors[index][::-1],3)
            canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(960,540)),cv2.COLOR_BGR2RGB)),(col*960,95))
            who='长边主机位' if col==0 else '低位端部机位'
            seconds=float(frame['source_seconds']);time_text=f'{int(seconds//3600):02d}:{int(seconds%3600//60):02d}:{seconds%60:06.3f}'
            draw.text((col*960+20,52),who+'  源时间 '+time_text,font=font,fill='white')
            p=np.array(points);x0,y0=np.maximum(p.min(0)-80,[0,0]).astype(int);x1,y1=np.minimum(p.max(0)+80,[1920,1080]).astype(int)
            crop=Image.fromarray(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));crop.thumbnail((920,245));canvas.paste(crop,(col*960+(960-crop.width)//2,650))
        draw.text((20,905),'彩色点直接量自原图，作为拟合输入；这张图不是模型拟合结果。遮挡的角点没有补画。',font=small,fill=(215,215,215))
        canvas.save(OUT/f'crossview_advert_{advert}_main_low.jpg',quality=94)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['match','remeasure','color','package','proof'],default='match')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='match':match_existing_advertisements()
    if args.phase=='remeasure':remeasure()
    if args.phase=='color':color_candidates()
    if args.phase=='package':package()
    if args.phase=='proof':proof_images()


if __name__=='__main__':
    main()
