"""Image-only measurements for the explicitly identified main-camera pool.

Saved geometry projections are never used as pixel observations. Existing
measured curves/details provide identities; RootSIFT proposes a correspondence,
then local pixels must independently support the proposed edge or printed detail.
"""
from rinkspace.storage import resolve_asset as _resolve_asset
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib, json, sys
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from track_scene3d_observations import normals, sample, edge_resnap, warp, motion, moving_occlusion_reject
from extract_scene3d_board_edges import measure_edge
from prepare_scene3d_calibration_observations import extract_ring,extract_line
BASE=_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v015_joint')
POOL=BASE/'women_sp_training_frames_1000'
OUT=BASE/'joint_279/main'
OLD=BASE/'fits/joint_ads111_start0'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def absolute(p):
    p=Path(p);return p if p.is_absolute() else ROOT/p

def image_features(entry):
    im=cv2.imread(entry['image_path']);gray=cv2.cvtColor(cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    mask=np.full(gray.shape,255,np.uint8)
    mask[:12]=0;mask[-12:]=0;mask[:,:12]=0;mask[:,-12:]=0
    mask[:155,:320]=0;mask[472:,810:]=0
    detector=cv2.SIFT_create(nfeatures=1700,contrastThreshold=.017,edgeThreshold=12)
    kp,desc=detector.detectAndCompute(gray,mask)
    if desc is not None:desc=np.sqrt(desc/(desc.sum(1,keepdims=True)+1e-8)).astype(np.float32)
    return dict(entry=entry,gray=gray,mask=mask,kp=np.float32([k.pt for k in kp]),desc=desc)

def match(a,b):
    if a['desc'] is None or b['desc'] is None:return None
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=3),dict(checks=48))
    forward=matcher.knnMatch(a['desc'],b['desc'],k=2)
    backward=matcher.knnMatch(b['desc'],a['desc'],k=2)
    rev={x.queryIdx:x.trainIdx for pair in backward if len(pair)==2 for x,y in [pair] if x.distance<.75*y.distance}
    good=[x for pair in forward if len(pair)==2 for x,y in [pair] if x.distance<.75*y.distance and rev.get(x.trainIdx)==x.queryIdx]
    if len(good)<12:return None
    p=np.float32([a['kp'][x.queryIdx] for x in good]);q=np.float32([b['kp'][x.trainIdx] for x in good])
    H,inlier=cv2.findHomography(p,q,cv2.USAC_MAGSAC,2.0,maxIters=1800,confidence=.995)
    if H is None:return None
    ok=inlier.ravel().astype(bool);n=int(ok.sum())
    if n<12 or n/len(good)<.35:return None
    span=np.ptp(q[ok],axis=0)
    if span[0]<90 or span[1]<24:return None
    residual=np.linalg.norm(warp(p[ok],H)-q[ok],axis=1)
    if np.percentile(residual,95)>2.4:return None
    # Source clock/scoreboards are masked and geometrically compact person-only
    # matches rejected. A broad match is still only a pixel search proposal.
    return dict(H=H/H[2,2],inliers=n,ratio=n/len(good),p95_half_px=float(np.percentile(residual,95)),
                matched_source=p[ok],matched_target=q[ok],span_half_px=span.tolist())

def local_patch_correlation(a,b,points_a,points_b,H,point=False):
    warped=cv2.warpPerspective(a,H,(960,540),flags=cv2.INTER_LINEAR)
    values=[];measured=[]
    for x,y in points_b:
        if x<13 or x>=947 or y<13 or y>=527:
            values.append(-1.);measured.append([x,y]);continue
        patch=cv2.getRectSubPix(warped,(13,13),(float(x),float(y)))
        if patch.std()<2.:
            values.append(-1.);measured.append([x,y]);continue
        search=cv2.getRectSubPix(b,(21,21),(float(x),float(y)))
        corr=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
        _,val,_,loc=cv2.minMaxLoc(corr)
        delta=np.array(loc,dtype=float)-4
        # The winning match must be inside the local search; repeated stripes
        # are not accepted as unique printed details.
        if min(loc)==0 or max(loc)==8:val=-1
        if point:
            other=corr.copy();xx,yy=loc;other[max(0,yy-1):yy+2,max(0,xx-1):xx+2]=-1
            if val-float(other.max())<.035:val=-1
        values.append(float(val));measured.append([x+delta[0],y+delta[1]])
    return np.array(values),np.float32(measured)

def measure_observation(source,target,obs,proposal):
    P=np.float32(obs['pixels_uv'])/2
    if len(P)>80:P=P[np.linspace(0,len(P)-1,80).astype(int)]
    H=proposal['H'];Q=warp(P,H)
    finite=np.isfinite(Q).all(1)
    inside=finite&(Q[:,0]>14)&(Q[:,0]<946)&(Q[:,1]>14)&(Q[:,1]<526)
    inside &= ~((Q[:,0]<320)&(Q[:,1]<155))
    inside &= ~((Q[:,0]>810)&(Q[:,1]>472))
    if obs['target']['kind']=='board_landmark':
        if not inside.all():return None
        ncc,measured=local_patch_correlation(source['gray'],target['gray'],P,Q,H,True)
        if ncc[0]<.78:return None
        # Forward/backward patch validation after alignment prevents exporting
        # a raw homography location as a printed-detail measurement.
        back=warp(measured,np.linalg.inv(H))
        nc_back,measured_back=local_patch_correlation(target['gray'],source['gray'],measured,back,np.linalg.inv(H),True)
        if nc_back[0]<.72 or np.linalg.norm(measured_back[0]-P[0])>1.4:return None
        return measured*2,dict(ncc_min=float(ncc.min()),forward_backward_native_px=float(np.linalg.norm(measured_back[0]-P[0])*2),kind='printed_detail_patch_remeasurement')
    if inside.sum()<8:return None
    N=normals(P,obs['target']);n=warp(P+N,H)-Q;n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-6)
    contrast=(sample(source['gray'],(P+N)[:,None])-sample(source['gray'],(P-N)[:,None]))[:,0]
    measured,ok,strength=edge_resnap(target['gray'],Q,n,np.where(contrast>=0,1,-1),np.abs(contrast))
    ncc,_=local_patch_correlation(source['gray'],target['gray'],P,Q,H)
    ok &= inside&(ncc>.42)&(np.abs(contrast)>1.0)
    if ok.sum()<8 or np.linalg.norm(np.ptp(measured[ok],axis=0))<18:return None
    return measured[ok]*2,dict(measured_points=int(ok.sum()),proposed_points=len(P),ncc_median=float(np.median(ncc[ok])),
                             gradient_median=float(np.median(strength[ok])),kind='signed_edge_remeasurement')

def pose_from_motion(seed,H):
    camera=seed.get('camera')
    if not camera:return {}
    S=np.diag([.5,.5,1.]);full=np.linalg.inv(S)@H@S
    P=full@np.array(camera['K_px'])@np.array(camera['R_world_to_camera'])
    T=np.array([[1,0,-959.5],[0,1,-539.5],[0,0,1.]])
    A=T@P
    scale=np.linalg.norm(A[2]);f=np.sqrt((np.dot(A[0],A[0])+np.dot(A[1],A[1]))/2)/scale
    approximate=np.diag([1/f,1/f,1.])@A/scale
    u,s,v=np.linalg.svd(approximate);R=u@v
    if np.linalg.det(R)<0:R=-R
    rvec=cv2.Rodrigues(R)[0].ravel().tolist()
    return dict(initial_rvec_world_to_camera=rvec,initial_focal_px=float(np.clip(f,450,30000)),
                initialization_diagnostics=dict(method='image-only motion applied to old conditional camera; initializer only',
                    seed_frame_id=seed['id'],rotation_singular_values=s.tolist(),not_a_measured_camera_pose=True))

def target_key(obs):return json.dumps(obs['target'],sort_keys=True)

def native_frame(row,shot):
    path=POOL/row['native_image_path']
    return dict(id=f"main279_{row['source_pts']}",source_id=row['source_id'],media_track_id=row['media_track_id'],
        source_path=str(absolute(row['source_path'])),source_sha256=row['source_sha256'],time_base='1/15360',stream_index=0,
        source_pts=row['source_pts'],source_seconds=row['source_seconds'],image_path=str(path),image_sha256=digest(path),
        image_size_wh=[1920,1080],principal_point_px=[959.5,539.5],fy_over_fx=1.,distortion_opencv=[0.,0.,0.,0.,0.],
        focal_bounds_px=[450.,30000.],shot_id=shot,camera_id='main_long_side',installation_id='women_sp_main',
        camera_assignment_authority=row['camera_identity_authority'],camera_identity_source=row['camera_identity_source'],
        pixel_frame_id=f"{row['media_track_id']}:pts:{row['source_pts']}:1920x1080",pool_ordinal=row['ordinal'],
        source_hash_verified_this_run=False,source_hash_verification='Inherited verified manifest; image bytes verified now',
        previously_heldout_promoted=shot in ['S074','S085'],fit_status='pending_actual_measurements')

def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'overlays').mkdir(exist_ok=True)
    cv2.setNumThreads(2);cv2.setRNGSeed(43)
    problem=load(OLD/'problem.json');fit=load(OLD/'fit.json');fitted={f['id']:f for f in fit['frames']}
    seeds=[];obs_by_id=defaultdict(list)
    for obs in problem['observations']:
        if obs.get('advertisement_instance_id')=='advert_ppt_object_108':continue
        obs_by_id[obs['frame_id']].append(obs)
    for frame in problem['frames']:
        if frame['installation_id']!='women_sp_main':continue
        f=copy.deepcopy(frame);f['camera']=fitted[f['id']]['camera'];f['seed_observation_file']=str(OLD/'problem.json')
        f['measurements']=obs_by_id[f['id']];seeds.append(f)
    # User's all-279 scope promotes these former holdout shots. Only their
    # original measured pixels are reused; failed fitted projections are not.
    for shot in ['S074','S085']:
        path=BASE/f'holdout/{shot}/observations.json';data=load(path)
        by=defaultdict(list)
        for obs in data['observations']:
            if obs['target']['kind'] in ['circle_outline','line_segment','board_side','board_corner']:
                by[obs['frame_id']].append(obs)
        for frame in data['frames']:
            if not by[frame['id']]:continue
            f=copy.deepcopy(frame);f['measurements']=by[f['id']];f['seed_observation_file']=str(path);f['promoted_old_holdout']=True;seeds.append(f)
    manifest=load(POOL/'training_input_manifest.json');selected=[f for f in manifest['frames'] if f['camera_id']=='main_long_side']
    timeline=load(_resolve_asset(ROOT, ROOT/'outputs/camera_classification/trial_010800_013600/result.json'))['intervals']
    newframes=[]
    for row in selected:
        shot=next((s['id'] for s in timeline if s['start']<=row['source_seconds']<s['end']),'user_main_850_905')
        newframes.append(native_frame(row,shot))
    print(f'Preparing {len(seeds)} measured seed frames and {len(newframes)} target frames',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_features=list(pool.map(image_features,seeds));target_features=list(pool.map(image_features,newframes))
    observations=[];statuses=[];matching_log=[];frame_obs={}
    for index,t in enumerate(target_features):
        frame=t['entry'];proposals=[]
        for i,s in enumerate(source_features):
            proposal=match(s,t)
            if proposal:proposals.append((i,proposal))
        proposals.sort(key=lambda x:x[1]['inliers'],reverse=True)
        best={};diagnostics=[]
        for si,proposal in proposals:
            seed=source_features[si]
            diagnostics.append(dict(seed_frame_id=seed['entry']['id'],inliers=proposal['inliers'],ratio=proposal['ratio'],
                p95_half_px=proposal['p95_half_px'],H_half=proposal['H'].tolist(),span_half_px=proposal['span_half_px']))
            for obs in seed['entry']['measurements']:
                key=target_key(obs)
                measured=measure_observation(seed,t,obs,proposal)
                if measured is None:continue
                points,quality=measured
                score=len(points)+quality.get('ncc_min',quality.get('ncc_median',0))*10
                if key not in best or score>best[key]['score']:
                    best[key]=dict(score=score,points=points,obs=obs,quality=quality,seed=seed['entry'],proposal=proposal)
        if proposals:
            si,proposal=proposals[0]
            frame.update(pose_from_motion(source_features[si]['entry'],proposal['H']))
        local=[]
        for j,item in enumerate(best.values()):
            obs=item['obs'];seed=item['seed'];target=copy.deepcopy(obs['target'])
            row={k:copy.deepcopy(obs[k]) for k in ['feature_id','physical_group_id','type','surface_class','identity_status','xyz_status','advertisement_instance_id'] if k in obs}
            row.update(id=f"{frame['id']}_measured_{j:03d}",frame_id=frame['id'],target=target,use='fit',pixels_uv=np.round(item['points'],4).tolist(),
                sigma_px=max(3.5 if target['kind']=='board_landmark' else 5.0,float(obs.get('sigma_px',3))),
                measurement_origin='actual_image_remeasurement_from_traceable_seed_identity',
                method='mutual RootSIFT broad static-image correspondence; signed local edge or bidirectional unique printed-detail patch remeasurement',
                synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                source_seed_observation_id=obs['id'],source_seed_frame_id=seed['id'],source_seed_pts=seed['source_pts'],
                source_seed_observation_file=seed['seed_observation_file'],measurement_diagnostics=item['quality'],
                correlation_group_id=f"{frame['shot_id']}_{obs.get('physical_group_id',target_key(obs))}",
                source_frame=dict(source_id=frame['source_id'],source_pts=frame['source_pts'],time_base='1/15360',image_path=frame['image_path']),
                previously_heldout_seed_promoted=bool(seed.get('promoted_old_holdout')),
                independence='development observations linked by physical identity; not heldout validation')
            local.append(row)
        frame['fit_status']='measured' if local else 'no_reliable_measurement'
        frame_obs[frame['id']]=local;observations.extend(local)
        statuses.append(dict(frame_id=frame['id'],ordinal=frame['pool_ordinal'],source_pts=frame['source_pts'],source_seconds=frame['source_seconds'],shot_id=frame['shot_id'],
            status=frame['fit_status'],measured_observation_groups=len(local),pixel_count=sum(len(o['pixels_uv']) for o in local),
            source_matches=len(proposals),targets=[o['target'] for o in local],image_path=frame['image_path'],
            previous_holdout_promoted=frame['previously_heldout_promoted'],
            failure_reason=None if local else 'No broad instance-consistent seed match with locally remeasured visible pixels; no projected substitute exported'))
        matching_log.append(dict(frame_id=frame['id'],proposals=diagnostics))
        if (index+1)%10==0:print(f"{index+1}/{len(newframes)} frames; measured {sum(x['status']=='measured' for x in statuses)}, observations {len(observations)}",flush=True)
        if (index+1)%20==0:save(OUT/'partial_status.json',statuses)
    # Visual evidence includes losses as well as successful matches, and every
    # selected frame receives a full-size overlay for individual inspection.
    panels=[]
    for i,f in enumerate(newframes):
        im=cv2.imread(f['image_path']);local=frame_obs[f['id']]
        for j,o in enumerate(local):
            color=[(0,220,255),(100,255,80),(255,100,220),(255,180,60)][j%4]
            for p in o['pixels_uv']:cv2.circle(im,tuple(np.rint(p).astype(int)),3,color,-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,55),(12,12,12),-1)
        cv2.putText(im,f"#{f['pool_ordinal']} {f['shot_id']} {f['source_seconds']:.3f}s  actual groups={len(local)}",(12,38),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
        path=OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg";cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,91])
        panels.append(cv2.resize(im,(480,270)))
    for page in range((len(panels)+11)//12):
        pp=panels[page*12:(page+1)*12]
        while len(pp)%3:pp.append(np.zeros((270,480,3),np.uint8))
        sheet=np.vstack([np.hstack(pp[j:j+3]) for j in range(0,len(pp),3)])
        cv2.imwrite(str(OUT/f'contact_{page+1:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
    summary=dict(selected_frames=len(newframes),measured_frames=sum(bool(frame_obs[f['id']]) for f in newframes),
        observation_groups=len(observations),observed_pixels=sum(len(o['pixels_uv']) for o in observations),
        by_kind=dict(Counter(o['target']['kind'] for o in observations)),
        previous_holdout_frames_promoted=sum(f['previously_heldout_promoted'] for f in newframes),
        identity_seed_frames=len(seeds),model_fitting_run=False,synthetic_or_model_projected_observations=False,
        limitations=['Frames with no reliable correspondence remain explicit and contribute no invented residual.',
                    'Direct cross-shot correspondences do not establish continuous coverage between the exported frames.',
                    'Actual frame camera labels come from the selected manifest, including the nine user corrections.',
                    'Old S074/S085 have been promoted to development and cannot be called independent validation.'])
    result=dict(schema_version='rinkspace.actual_main279_observations.v1',frames=newframes,observations=observations,
        board_landmarks=problem['board_landmarks'],summary=summary,source_manifest=str(POOL/'training_input_manifest.json'),
        source_manifest_sha256=digest(POOL/'training_input_manifest.json'),source_seed_problem_sha256=digest(OLD/'problem.json'))
    save(OUT/'observations.json',result);save(OUT/'frame_status.json',statuses);save(OUT/'image_correspondences.json',matching_log);save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True),flush=True)

def direct_visible_features():
    """Add visibly present features when texture-template matching failed.

    The following guides were read from native source images, without loading a
    fitted scene. They delimit searches, not the coordinates exported as data.
    Straight-board extents exclude visually identified rounded-corner portions.
    """
    result=load(OUT/'observations.json');statuses=load(OUT/'frame_status.json')
    if not (OUT/'observations_before_direct_guides.json').exists():save(OUT/'observations_before_direct_guides.json',result)
    result['observations']=[o for o in result['observations'] if o.get('measurement_origin')!='direct_native_visible_feature_edge_measurement']
    board={
      95:('upper_long_side',30,1880,470,849),96:('upper_long_side',1700,1900,497,828),
      98:('left_end',30,1460,492,780),101:('right_end',30,1850,474,750),
      102:('right_end',1320,1890,430,674),103:('right_end',30,1850,528,837),
      105:('upper_long_side',30,1850,462,700),106:('left_end',30,1850,612,843),
      107:('left_end',30,1510,441,729),911:('upper_long_side',50,1850,219,420),
      925:('upper_long_side',50,1850,372,702),1002:('right_end',30,1850,462,684),
      1010:('upper_long_side',40,1850,395,700),1091:('upper_long_side',50,1850,225,295),
      1120:('upper_long_side',700,1850,191,248),1124:('upper_long_side',50,1850,342,558),
      1126:('upper_long_side',40,1850,354,648),1180:('left_end',40,1480,327,435),
      1188:('right_end',50,1600,334,512),1210:('right_end',1280,1890,409,575),
      1212:('upper_long_side',50,1850,354,630),1283:('right_end',30,1750,460,622),
      1284:('right_end',30,1850,504,726),1294:('upper_long_side',40,1850,396,627),
      1305:('right_end',900,1890,408,546),
    }
    ring={1188:('R1',[610,912,780,95,-1.]),1283:('R1',[755,975,687,91,0.]),
          1210:('R2',[1000,916,691,78,-.5]),1305:('R2',[711,821,566,67,0.])}
    line={1091:[[1380,290],[1150,1050]],1120:[[852,280],[714,535]]}
    additions=[];failures=[];all_measurements=[]
    for frame in result['frames']:
        ordinal=frame['pool_ordinal'];im=cv2.imread(frame['image_path']);entries=[]
        if ordinal in board:
            side,x0,x1,top,bottom=board[ordinal]
            # Conservative central subject exclusion; it is not a detected pose.
            exclusion=[[600,80,1320,1060]]
            if ordinal in [96,102,1210,1305]:exclusion=[]
            if ordinal==1120:exclusion=[[0,0,700,1080]]
            for edge,y in [('top',top),('contact',bottom)]:
                band=12 if edge=='top' and ordinal in [1091,1120] else 34
                roi=[x0,max(0,y-band),x1,min(1080,y+band)]
                try:
                    measured=measure_edge(im,roi=roi,edge='blue_rail_upper' if edge=='top' else 'yellow_kickplate_lower',exclude_rectangles=exclusion)
                    points=measured['pixel_points']
                    if len(points)>90:points=np.array(points)[np.linspace(0,len(points)-1,90).astype(int)].tolist()
                    entries.append(dict(target=dict(kind='board_side',side=side,edge=edge),physical_group_id=f'boards_{side}_{edge}',
                        type='curve',surface_class='board_elevated' if edge=='top' else 'board_ice_contact',pixels_uv=points,
                        sigma_px=max(6.,measured['clarity']['median_uncertainty_px']),direct_measurement=measured,
                        identity_evidence='Native source visually reviewed: architectural background and adjacent views identify the physical end; advertisement brands/order alone cannot distinguish centrally repeated installations. Rounded portions and people excluded.'))
                except ValueError as e:failures.append(dict(ordinal=ordinal,target=f'{side}_{edge}',error=str(e)))
        if ordinal in ring:
            fid,ellipse=ring[ordinal]
            exclusions={1188:[[1030,180,1350,905]],1283:[[820,350,1140,910]],1210:[[820,300,1180,850]],1305:[[520,350,780,780]]}[ordinal]
            spec=dict(feature_id=fid,seed_ellipse_cx_cy_a_b_angle=ellipse,exclude_boxes=exclusions,min_gradient_strength=2.5)
            if ordinal==1188:spec['partial_no_ellipse_refit']=True
            try:
                measured=extract_ring(im,spec)
                entries.append(dict(target=dict(kind='circle_outline',feature_id=fid),physical_group_id=fid,feature_id=fid,
                    type='curve',surface_class='ice_marking',pixels_uv=measured['pixels_uv'],sigma_px=max(5.,measured['sigma_px']),direct_measurement=measured,
                    identity_evidence='Right-end grey-door/window wall: R1 grey ring at lower end by SPORTOTO/can/WYNDHAM; R2 red ring between gray R3 and R1 in uninterrupted 5312.5-5316s source views. Architectural context and adjacent source frames distinguish repeated opposite-end advertisements.'))
            except ValueError as e:failures.append(dict(ordinal=ordinal,target=fid,error=str(e)))
        if ordinal in line:
            spec=dict(feature_id='C',search_axis_endpoints=line[ordinal],half_band_px=95,exclude_boxes=[])
            try:
                measured=extract_line(im,spec)
                entries.append(dict(target=dict(kind='line_segment',feature_id='C'),physical_group_id='C',feature_id='C',
                    type='curve',surface_class='ice_marking',pixels_uv=measured['pixels_uv'],sigma_px=max(7.,measured['sigma_px']),direct_measurement=measured,
                    identity_evidence='Native image central red line through the event banner/centre-circle region; search interval excludes the skater.'))
            except ValueError as e:failures.append(dict(ordinal=ordinal,target='C',error=str(e)))
        if ordinal in [96,1099]:
            # Right upper/lower corners of user7/object111. Left edge is clipped;
            # do not pretend it is visible or generate a four-corner rectangle.
            xmid,ytop,ybottom=(756,520,765) if ordinal==96 else (1554,450,731)
            B,G,R=cv2.split(im.astype(np.float32));blue=cv2.GaussianBlur(B-(R+G)/2,(0,0),1.4)
            vertical=[]
            for y in range(ytop+28,ybottom-28,9):
                p=blue[y-2:y+3,xmid-28:xmid+29].mean(0);g=-np.gradient(p);ix=int(np.argmax(g))
                if 2<ix<len(g)-3 and g[ix]>1.2:vertical.append([xmid-28+ix,y])
            if len(vertical)>=8:
                vx,vy,x,y=cv2.fitLine(np.float32(vertical),cv2.DIST_HUBER,0,.01,.01).ravel();vertical_line=np.array([-vy,vx,vy*x-vx*y])
                for corner,yy,sign in [(1,ytop,1),(2,ybottom,-1)]:
                    horizontal=[]
                    for xx in range(xmid-170,xmid-25,7):
                        p=blue[yy-24:yy+25,xx-2:xx+3].mean(1);g=np.gradient(p)*sign;iy=int(np.argmax(g))
                        if 2<iy<len(g)-3 and g[iy]>1.2:horizontal.append([xx,yy-24+iy])
                    if len(horizontal)<8:continue
                    vx,vy,x,y=cv2.fitLine(np.float32(horizontal),cv2.DIST_HUBER,0,.01,.01).ravel();horizontal_line=np.array([-vy,vx,vy*x-vx*y])
                    meet=np.cross(vertical_line,horizontal_line);uv=meet[:2]/meet[2]
                    lid=f'advert_ppt_object_111_printed_corner_{corner}'
                    entries.append(dict(target=dict(kind='board_landmark',landmark_id=lid),physical_group_id='advert_ppt_object_111',advertisement_instance_id='advert_ppt_object_111',
                        type='point',surface_class='board_elevated',pixels_uv=[uv.tolist()],sigma_px=7.,
                        direct_measurement=dict(vertical_edge_pixels=vertical,horizontal_edge_pixels=horizontal),
                        identity_evidence='User7 blue ISU object111 before red katakana object110; right printed colour boundary corners, not panel seams.'))
        for j,entry in enumerate(entries):
            measurement=entry.pop('direct_measurement')
            entry.update(id=f"{frame['id']}_direct_native_{j:02d}",frame_id=frame['id'],use='fit',
                measurement_origin='direct_native_visible_feature_edge_measurement',synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                source_frame=dict(source_id=frame['source_id'],source_pts=frame['source_pts'],time_base='1/15360',image_path=frame['image_path']),
                correlation_group_id=f"{frame['shot_id']}_{entry['physical_group_id']}",
                method='Native image chromatic/intensity transition measurement; visually identified search band, no scene projection read',
                identity_status='assistant_visual_association_to_user_layout',xyz_status='jointly_estimated_not_surveyed')
            additions.append(entry);all_measurements.append(dict(observation_id=entry['id'],ordinal=ordinal,measurement=measurement))
        if entries:
            frame['fit_status']='measured_direct_visible_features'
    result['observations'].extend(additions)
    byobs=defaultdict(list)
    for o in result['observations']:byobs[o['frame_id']].append(o)
    for status in statuses:
        rows=byobs[status['frame_id']]
        if rows:
            status.update(status='measured',measured_observation_groups=len(rows),pixel_count=sum(len(o['pixels_uv'])for o in rows),targets=[o['target']for o in rows],failure_reason=None)
    result['summary'].update(measured_frames=sum(bool(byobs[f['id']])for f in result['frames']),observation_groups=len(result['observations']),
        observed_pixels=sum(len(o['pixels_uv'])for o in result['observations']),direct_new_native_groups=len(additions),
        by_kind=dict(Counter(o['target']['kind']for o in result['observations'])))
    save(OUT/'observations.json',result);save(OUT/'frame_status.json',statuses);save(OUT/'summary.json',result['summary'])
    save(OUT/'direct_native_measurements.json',dict(measurements=all_measurements,failed_searches=failures))
    for f in result['frames']:
        if not any(o['measurement_origin']=='direct_native_visible_feature_edge_measurement'for o in byobs[f['id']]):continue
        im=cv2.imread(f['image_path'])
        for j,o in enumerate(byobs[f['id']]):
            color=[(0,220,255),(100,255,80),(255,100,220),(255,180,60)][j%4]
            for p in o['pixels_uv']:cv2.circle(im,tuple(np.rint(p).astype(int)),3,color,-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,55),(12,12,12),-1)
        cv2.putText(im,f"#{f['pool_ordinal']} {f['shot_id']} {f['source_seconds']:.3f}s  actual measured groups={len(byobs[f['id']])}",(12,38),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,91])
    print(json.dumps(result['summary'],ensure_ascii=True));print('Failed searches',failures)


def ensure_initializers():
    """Attach conditional starting poses only; do not change a measured pixel."""
    result=load(OUT/'observations.json');old=load(OLD/'fit.json')
    cameras={f['id']:f for f in old['frames']}
    S=np.diag([.5,.5,1.])
    for shot in ['S074','S085']:
        poses=load(BASE/f'holdout/{shot}/results/pose_fit.json')['poses']
        native={p['source_pts']:p['selected']['camera'] for p in poses if p.get('selected')}
        projected=load(BASE/f'holdout/{shot}/results/projection_frames.json')['frames']
        data=load(BASE/f'holdout/{shot}/observations.json')
        for frame in data['frames']:
            if frame['source_pts'] in native:
                cameras[frame['id']]={**frame,'camera':native[frame['source_pts']]};continue
            candidates=[p for p in projected if p.get('H_current_from_anchor') is not None and p.get('anchor_source_pts') in native]
            if not candidates:continue
            near=min(candidates,key=lambda p:abs(p['source_pts']-frame['source_pts']))
            seed=dict(id=f"conditional_{near['anchor_source_pts']}",camera=native[near['anchor_source_pts']])
            inferred=pose_from_motion(seed,S@np.array(near['H_current_from_anchor'])@np.linalg.inv(S))
            rv=inferred['initial_rvec_world_to_camera'];focal=inferred['initial_focal_px']
            cameras[frame['id']]={**frame,'camera':dict(K_px=[[focal,0,959.5],[0,focal,539.5],[0,0,1]],
                R_world_to_camera=cv2.Rodrigues(np.array(rv))[0].tolist()),'initializer_caution':'old conditional motion chain, not verified pose'}
    proposals={r['frame_id']:r['proposals'] for r in load(OUT/'image_correspondences.json')}
    used={o['frame_id'] for o in result['observations']};log=[]
    for frame in result['frames']:
        if frame['id'] not in used or 'initial_focal_px' in frame:continue
        choices=[p for p in proposals.get(frame['id'],[]) if p['seed_frame_id'] in cameras]
        if choices:
            p=choices[0];frame.update(pose_from_motion(cameras[p['seed_frame_id']],np.array(p['H_half'])))
            frame['initialization_diagnostics']['old_holdout_initialization_only']=p['seed_frame_id'].startswith('holdout_')
        else:
            # A sparse board-only frame may have no identifiable zoom/pose.
            # Keep its real measurements; an explicitly uncertain same-camera
            # start is permissible and must not be called a calibrated result.
            oldframes=[f for f in old['frames'] if f['installation_id']=='women_sp_main']
            near=min(oldframes,key=lambda f:abs(f['source_pts']-frame['source_pts']))
            frame.update(initial_rvec_world_to_camera=near['camera']['rvec_world_to_camera'],
                initial_focal_px=near['camera']['K_px'][0][0],initialization_diagnostics=dict(
                    method='uncertain same-installation fallback; initial value only, pose not identified by sparse observations',seed_frame_id=near['id']))
        log.append(dict(frame_id=frame['id'],diagnostics=frame['initialization_diagnostics']))
    save(OUT/'observations.json',result);save(OUT/'initialization_completion.json',log)
    print('Initializers added',len(log),'missing with observations',sum(f['id'] in used and 'initial_focal_px'not in f for f in result['frames']),flush=True)


def repair_backward_initializers():
    sys.path.insert(0,str(ROOT/'src'))
    from rinkspace.calibration_companion import Scene,look_at_rvec
    from fit_scene3d_real_keyframes import initialize_frame
    scene=Scene(_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.15b-advert-joint/scene.json'))
    result=load(OUT/'observations.json');measurements={m['observation_id']:m['measurement'] for m in load(OUT/'direct_native_measurements.json')['measurements']}
    old=load(OLD/'fit.json');center=np.array(next(f for f in old['frames'] if f['installation_id']=='women_sp_main')['camera']['center_xyz'])
    log=[]
    for frame in result['frames']:
        if frame['pool_ordinal']not in [102,1210,1305]:continue
        previous=dict(rvec=frame.get('initial_rvec_world_to_camera'),focal=frame.get('initial_focal_px'))
        if frame['pool_ordinal']==102:
            right=any(o['frame_id']==frame['id'] and o['target'].get('side')=='right_end' for o in result['observations'])
            target=np.array([100. if right else -100.,12. if right else -12.,1.64])
            frame['initial_rvec_world_to_camera']=look_at_rvec(center,target).tolist();frame['initial_focal_px']=8000.
            method='Aim at currently identified board side; numerical start only'
        else:
            ring=copy.deepcopy(next(o for o in result['observations'] if o['frame_id']==frame['id'] and o['target']['kind']=='circle_outline'))
            ring['ellipse_cv2_image_only']=measurements[ring['id']]['ellipse_cv2_image_only']
            rv,focal=initialize_frame(scene,frame,[ring],center,1.)
            frame['initial_rvec_world_to_camera']=rv;frame['initial_focal_px']=focal
            method='Aim at observed F2 and image-only measured ellipse with old conditional scene scale; numerical start only'
        frame['initialization_diagnostics']=dict(method=method,not_calibrated=True,previous_backward_initialization_replaced=previous)
        log.append(dict(frame_id=frame['id'],ordinal=frame['pool_ordinal'],initial_focal_px=frame['initial_focal_px']))
    save(OUT/'observations.json',result);save(OUT/'backward_initialization_repair.json',log)
    print('Repaired backward initializers',log)


def correct_right_end_identities():
    """Correct nine confirmed physical identities without changing pixel evidence."""
    sys.path.insert(0,str(ROOT/'src'))
    from rinkspace.calibration_companion import Scene,look_at_rvec
    from fit_scene3d_real_keyframes import initialize_frame,align_z_to
    scene=Scene(_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.15b-advert-joint/scene.json'))
    old=load(OLD/'fit.json');center=np.array(next(f for f in old['frames'] if f['installation_id']=='women_sp_main')['camera']['center_xyz'])
    result=load(OUT/'observations.json');original=copy.deepcopy(result)
    if not (OUT/'observations_before_right_end_identity_correction.json').exists():save(OUT/'observations_before_right_end_identity_correction.json',original)
    selected={101,102,103,1002,1188,1210,1283,1284,1305};circles={1188:'R1',1283:'R1',1210:'R2',1305:'R2'}
    frames={f['id']:f for f in result['frames']};byobs=defaultdict(list);changed=[]
    measurements={m['observation_id']:m['measurement'] for m in load(OUT/'direct_native_measurements.json')['measurements']}
    for o in result['observations']:
        f=frames[o['frame_id']];byobs[f['id']].append(o)
        if f['pool_ordinal']not in selected:continue
        before=copy.deepcopy(o['target']);target=o['target']
        if target['kind']=='board_side':target['side']='right_end';o['physical_group_id']=f"boards_right_end_{target['edge']}"
        elif target['kind']=='circle_outline':target['feature_id']=circles[f['pool_ordinal']];o['feature_id']=target['feature_id'];o['physical_group_id']=target['feature_id']
        else:raise ValueError('Unexpected target in limited identity correction')
        if before==target:continue
        o['previous_target']=before;o['previous_correlation_group_id']=o['correlation_group_id'];o['correlation_group_id']=f"{f['shot_id']}_{o['physical_group_id']}"
        o['identity_correction']={
            'reason':'Repeated opposite-end advertisement sequence was incorrectly identified as left end. Grey doors and broad upper glass windows identify the right end; left-end reference has continuous tall translucent rink panels.',
            'evidence_images':['../main_fit_diagnostic/identity1283.jpg','../main_fit_diagnostic/identity1188.jpg','../main_fit_diagnostic/red_circle_context.jpg'],
            'independent_right_end_reference_ordinal':109,'left_end_reference_ordinals':[1194,1039],
            'review':'Root independently confirmed nine right-end frames and R1/R2 circle IDs from adjacent views, APP1 entrance-stage association, and building background.',
            'user_information':'User confirms advertisement content and order are centrally symmetric around the rink. This is used only to reject brand/order-only instance identification; no geometric symmetry constraint is added.',
            'pixels_changed':False,'model_projection_used_for_identity':False}
        o['identity_status']='right_end_identity_corrected_from_building_context_and_adjacent_views'
        o['identity_evidence']='Right-end grey door/window wall, independently matched to ordinal109; user confirms repeated opposite-side advertisement sequence. Circle R2 is between gray R3 and R1 in continuous native source views; grey R1 at the lower end. See identity_correction evidence.'
        changed.append(dict(observation_id=o['id'],frame_id=f['id'],ordinal=f['pool_ordinal'],source_seconds=f['source_seconds'],previous_target=before,new_target=copy.deepcopy(target),pixel_count=len(o['pixels_uv'])))
    pose_changes=[]
    for f in result['frames']:
        ordinal=f['pool_ordinal']
        if ordinal not in selected:continue
        previous=dict(rvec=f['initial_rvec_world_to_camera'],focal=f['initial_focal_px'])
        ring=next((copy.deepcopy(o) for o in byobs[f['id']] if o['target']['kind']=='circle_outline'),None)
        if ring:
            ring['ellipse_cv2_image_only']=measurements[ring['id']]['ellipse_cv2_image_only']
            rv,focal=initialize_frame(scene,f,[ring],center,1.)
        else:
            boards=byobs[f['id']];ob=boards[0];uv=np.mean(ob['pixels_uv'],axis=0)
            target=np.array([scene.data['parameters']['bounds'][2],15. if ordinal in [101,102,1002] else -15.,scene.data['parameters']['board_height'] if ob['target']['edge']=='top' else 0.])
            focal=6500.
            if len(boards)>1:
                heights=[np.median(np.asarray(o['pixels_uv'])[:,1]) for o in boards]
                focal=float(np.clip(abs(heights[0]-heights[1])*np.linalg.norm(target-center)/scene.data['parameters']['board_height'],1000.,18000.))
            base=cv2.Rodrigues(look_at_rvec(center,target))[0];ray=np.r_[(uv-np.asarray(f['principal_point_px']))/focal,1.]
            rv=cv2.Rodrigues(align_z_to(ray)@base)[0].ravel().tolist()
        f['initial_rvec_world_to_camera']=rv;f['initial_focal_px']=float(focal)
        f['initialization_diagnostics']=dict(method='Corrected right-end target aiming; free numerical initializer only, not measured pose',previous_left_end_pose_replaced=previous,identity_correction_applied=True,not_calibrated=True)
        pose_changes.append(dict(frame_id=f['id'],ordinal=ordinal,previous=previous,initial_rvec_world_to_camera=rv,initial_focal_px=float(focal)))
    assert len(changed)==20, len(changed)
    assert all(a['pixels_uv']==b['pixels_uv'] for a,b in zip(original['observations'],result['observations']))
    unaffected=set(frames)-{f['id'] for f in result['frames'] if f['pool_ordinal']in selected}
    assert all(a==b for a,b in zip(original['observations'],result['observations']) if a['frame_id']in unaffected)
    result['summary']['identity_correction']=dict(frames=9,observation_groups=20,pixel_changes=0,reason='Opposite-end repeated advertisements; grey door/window building context corrects left to right, F3 to R1, F2 to R2')
    save(OUT/'observations.json',result);save(OUT/'summary.json',result['summary'])
    statuses=load(OUT/'frame_status.json')
    for s in statuses:s['targets']=[o['target'] for o in byobs[s['frame_id']]]
    save(OUT/'frame_status.json',statuses)
    report=dict(observations_sha256=digest(OUT/'observations.json'),changed=changed,pose_changes=pose_changes,pixels_unchanged=True,other_132_frames_and_all_their_observations_unchanged=True,geometric_symmetry_added=False)
    save(OUT/'right_end_identity_correction.json',report)
    for f in result['frames']:
        if f['pool_ordinal']not in selected:continue
        im=cv2.imread(f['image_path'])
        for j,o in enumerate(byobs[f['id']]):
            for q in o['pixels_uv']:cv2.circle(im,tuple(np.rint(q).astype(int)),3,[(0,220,255),(100,255,80),(255,100,220)][j%3],-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,80),(10,10,10),-1)
        text=f"#{f['pool_ordinal']} {f['source_seconds']:.3f}s RIGHT END | circle {circles.get(f['pool_ordinal'],'none')} | identity corrected, measured pixels unchanged"
        cv2.putText(im,text,(12,43),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,94])
    print(json.dumps(dict(frames=9,changed_groups=len(changed),pixels_unchanged=True,sha256=report['observations_sha256']),ensure_ascii=True),flush=True)


def correct_red_row_from_sequence():
    """Resolve the red target's row from gray neighbours in uninterrupted video."""
    sys.path.insert(0,str(ROOT/'src'))
    from rinkspace.calibration_companion import Scene
    from fit_scene3d_real_keyframes import initialize_frame
    scene=Scene(_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.15b-advert-joint/scene.json'))
    old=load(OLD/'fit.json');center=np.array(next(f for f in old['frames'] if f['installation_id']=='women_sp_main')['camera']['center_xyz'])
    result=load(OUT/'observations.json');previous=copy.deepcopy(result)
    if not (OUT/'observations_before_R2_sequence_correction.json').exists():save(OUT/'observations_before_R2_sequence_correction.json',result)
    selected={1210,1305};frames={f['id']:f for f in result['frames']};changes=[]
    measurements={m['observation_id']:m['measurement'] for m in load(OUT/'direct_native_measurements.json')['measurements']}
    for o in result['observations']:
        f=frames[o['frame_id']]
        if f['pool_ordinal']not in selected or o['target']['kind']!='circle_outline':continue
        prev=copy.deepcopy(o['target']);o['target']['feature_id']='R2';o['feature_id']='R2';o['physical_group_id']='R2'
        o.setdefault('identity_correction_history',[]).append(dict(previous_target=prev,corrected_target=copy.deepcopy(o['target']),reason='Direct uninterrupted source sequence: target red circle lies between known gray R3 and gray R1; target is R2, not R4. Position in front of APP/TC advertisement did not establish row identity.',evidence=[dict(source_seconds=5312.5,source_pts=81600000,relationship='Known gray R3 left/far, target red circle right/near, both visible'),dict(source_seconds=5313.5,source_pts=81615360,relationship='Same target red circle left/far, known gray R1 right/near, both visible'),dict(source_seconds=5316.,source_pts=81653760,relationship='Continuous pan returns to the same red circle in frame1210'),dict(source_seconds=5695.,source_pts=87475200,relationship='Same right-end red circle, with preceding gray R3 visible at left; continuous shot reaches frame1305 at5696s')],pixels_changed=False,model_fit_used_to_choose_identity=False))
        o['correlation_group_id']=f"{f['shot_id']}_R2";o['identity_status']='R2_row_confirmed_by_continuous_source_views_between_R3_and_R1'
        o['identity_evidence']='Continuous native 5312.5s grayR3+targetred,5313.5s targetred+grayR1 establishes R3-R2-R1 order;5316s is same red circle. Independent programme5693-5696s same ordering and physical ring. No model residual used for identity.'
        o['identity_correction']['review']='Right-end identity retained; subsequent continuous-video row check supersedes R4 claim. Final red circle identity R2.'
        ring=copy.deepcopy(o);ring['ellipse_cv2_image_only']=measurements[o['id']]['ellipse_cv2_image_only']
        rv,focal=initialize_frame(scene,f,[ring],center,1.)
        previous_pose=dict(rvec=f['initial_rvec_world_to_camera'],focal=f['initial_focal_px'])
        f['initial_rvec_world_to_camera']=rv;f['initial_focal_px']=float(focal);f['initialization_diagnostics']=dict(method='Aim at sequence-confirmed R2 and measured ellipse; free numerical initializer only',previous_R4_pose_replaced=previous_pose,not_calibrated=True)
        changes.append(dict(frame_id=f['id'],ordinal=f['pool_ordinal'],source_pts=f['source_pts'],source_seconds=f['source_seconds'],observation_id=o['id'],previous_target=prev,new_target=o['target'],initial_rvec_world_to_camera=rv,initial_focal_px=float(focal)))
    assert len(changes)==2
    assert all(a['pixels_uv']==b['pixels_uv']for a,b in zip(previous['observations'],result['observations']))
    assert all(a==b for a,b in zip(previous['observations'],result['observations']) if a['id']not in {c['observation_id']for c in changes})
    result['summary']['identity_correction']['reason']='Opposite-end repeated advertisements corrected using building background; gray circles F3 to R1; red circles F2 then erroneous R4 to sequence-confirmed R2.'
    result['summary']['red_row_sequence_correction']=dict(frames=2,observation_groups=2,pixels_changed=False,final_feature_id='R2')
    save(OUT/'observations.json',result);save(OUT/'summary.json',result['summary'])
    statuses=load(OUT/'frame_status.json')
    for s in statuses:
        if frames[s['frame_id']]['pool_ordinal']in selected:s['targets']=[o['target']for o in result['observations']if o['frame_id']==s['frame_id']]
    save(OUT/'frame_status.json',statuses)
    report=dict(changes=changes,observations_sha256=digest(OUT/'observations.json'),pixels_unchanged=True,all_other_893_observation_groups_unchanged=True,evidence_directory='../main_fit_diagnostic/red_ring_sequence',geometric_symmetry_added=False)
    save(OUT/'R2_sequence_identity_correction.json',report)
    for f in result['frames']:
        if f['pool_ordinal']not in selected:continue
        im=cv2.imread(f['image_path']);rows=[o for o in result['observations']if o['frame_id']==f['id']]
        for j,o in enumerate(rows):
            for q in o['pixels_uv']:cv2.circle(im,tuple(np.rint(q).astype(int)),3,[(0,220,255),(100,255,80),(255,100,220)][j%3],-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,80),(10,10,10),-1)
        cv2.putText(im,f"#{f['pool_ordinal']} {f['source_seconds']:.3f}s RIGHT END | circle R2 | confirmed between R3 and R1 in continuous source video",(12,43),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,94])
    print(json.dumps(report,ensure_ascii=True),flush=True)


def finalize_qa():
    """Freeze measured input after visual correction, and refresh review pages."""
    result=load(OUT/'observations.json');frames=result['frames'];obs=result['observations']
    manifest=load(POOL/'training_input_manifest.json')
    selected={f['ordinal']:f for f in manifest['frames'] if f.get('camera_id')=='main_long_side'}
    byobs=defaultdict(list)
    for o in obs:
        byobs[o['frame_id']].append(o)
        oldstatus=o.get('xyz_status','')
        if 'fixed scene' in oldstatus:
            o['source_seed_xyz_status']=oldstatus
            o['xyz_status']='jointly_estimated_board_geometry; seed schematic XYZ is not a measurement'
    checks={
        'all_selected_main_ordinals_present':set(selected)=={f['pool_ordinal'] for f in frames},
        'unique_frame_ids':len({f['id'] for f in frames})==len(frames),
        'unique_native_pts':len({f['source_pts'] for f in frames})==len(frames),
        'all_frames_have_real_observations':all(byobs[f['id']] for f in frames),
        'unique_observation_ids':len({o['id'] for o in obs})==len(obs),
        'all_observation_frames_resolve':all(o['frame_id'] in {f['id'] for f in frames} for o in obs),
        'all_pixel_points_finite_inside_native_image':all(np.isfinite(o['pixels_uv']).all() and np.all((np.asarray(o['pixels_uv'])>=0)&(np.asarray(o['pixels_uv'])<[1920,1080])) for o in obs),
        'all_initializers_finite_and_focal_bounded':all(len(f.get('initial_rvec_world_to_camera',[]))==3 and np.isfinite(f['initial_rvec_world_to_camera']).all() and np.isfinite(f.get('initial_focal_px',np.nan)) and f['focal_bounds_px'][0]<=f['initial_focal_px']<=f['focal_bounds_px'][1] for f in frames),
        'native_image_hashes_match_manifest':all(digest(f['image_path'])==f['image_sha256']==selected[f['pool_ordinal']]['native_image_sha256'] for f in frames),
        'pts_and_camera_match_manifest':all(f['source_pts']==selected[f['pool_ordinal']]['source_pts'] and f['camera_id']==selected[f['pool_ordinal']]['camera_id'] for f in frames),
        'all_landmark_targets_resolve':all(o['target'].get('landmark_id') in {b['id'] for b in result['board_landmarks']} for o in obs if o['target']['kind']=='board_landmark'),
        'ad108_excluded_from_all_observations':all('advert_ppt_object_108' not in json.dumps(o['target']) for o in obs),
        'no_model_projected_or_pure_transfer_observations':all(not o.get('synthetic_or_model_projected',False) and not o.get('pure_transfer_points_exported',False) for o in obs),
    }
    assert all(checks.values()),checks
    result['summary']['limitations']=[s for s in result['summary']['limitations'] if not s.startswith('Frames with no reliable correspondence')]
    limitation='All 141 frames have measurements; a board-only frame may still provide weak camera or scene constraints. Some motion-blurred measurements carry broad pixel uncertainty.'
    if limitation not in result['summary']['limitations']:result['summary']['limitations'].append(limitation)
    save(OUT/'observations.json',result);save(OUT/'summary.json',result['summary'])
    def pages(items,prefix,n,width,height,columns):
        panels=[cv2.resize(cv2.imread(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg")),(width,height)) for f in items]
        for page in range((len(panels)+n-1)//n):
            pp=panels[page*n:(page+1)*n]
            while len(pp)%columns:pp.append(np.zeros((height,width,3),np.uint8))
            cv2.imwrite(str(OUT/f'{prefix}_{page+1:02d}.jpg'),np.vstack([np.hstack(pp[j:j+columns]) for j in range(0,len(pp),columns)]),[cv2.IMWRITE_JPEG_QUALITY,93])
    pages(frames,'contact',12,480,270,3)
    directframes={o['frame_id'] for o in obs if o.get('measurement_origin')=='direct_native_visible_feature_edge_measurement'}
    pages([f for f in frames if f['id'] in directframes],'direct_final',8,640,360,2)
    qa=dict(checks=checks,all_passed=all(checks.values()),observations_sha256=digest(OUT/'observations.json'),
        frames=len(frames),observation_groups=len(obs),pixel_samples=sum(len(o['pixels_uv']) for o in obs),
        visual_review='All 26 direct-addition frames reviewed; pixel-search corrections retained. Subsequent root audit overturned nine end identities: right-end grey-window architecture differs from left tall rink panels despite centrally repeated advertisements. See right_end_identity_correction.json; four circles corrected to R1/R2 without pixel edits.',
        initial_pose_repairs=load(OUT/'backward_initialization_repair.json'),
        later_identity_and_pose_correction='right_end_identity_correction.json',
        latest_red_row_correction='R2_sequence_identity_correction.json',
        representative_new_evidence=['overlays/01283.jpg','overlays/01305.jpg','overlays/01120.jpg'],
        validation_limit='Input integrity and measurement review only; not geometric accuracy, independent validation, or successful continuous coverage. Root joint fit still required.')
    save(OUT/'final_QA.json',qa);print(json.dumps(qa,ensure_ascii=True),flush=True)


def bridge_missing():
    """Recover missing selected frames through native within-shot pixel tracks."""
    cv2.setNumThreads(2)
    result=load(OUT/'observations.json');statuses=load(OUT/'frame_status.json')
    if not (OUT/'observations_direct.json').exists():save(OUT/'observations_direct.json',result)
    byframe={f['id']:f for f in result['frames']};byobs=defaultdict(list)
    for o in result['observations']:byobs[o['frame_id']].append(o)
    goodframes=[f for f in result['frames'] if byobs[f['id']]]
    additions=[];audits=[];total_processed=0
    for status in statuses:
        if status['status']=='measured':continue
        target=byframe[status['frame_id']]
        choices=sorted([f for f in goodframes if f['shot_id']==target['shot_id']],key=lambda f:abs(f['source_pts']-target['source_pts']))
        if not choices:
            audits.append(dict(frame_id=target['id'],status='no_measured_seed_in_same_shot'));continue
        # Try distinct actual measured anchors, nearest first, until a genuine
        # uninterrupted local track succeeds. No cross-cut track is attempted.
        candidates=[]
        for seed in choices:
            pts0=seed['source_pts'];pts1=target['source_pts'];lo=min(pts0,pts1);hi=max(pts0,pts1)
            pictures=[];decoded=0
            with av.open(seed['source_path']) as container:
                stream=container.streams.video[0];container.seek(lo,stream=stream,backward=True)
                for vf in container.decode(stream):
                    if vf.pts is None or vf.pts<lo:continue
                    if vf.pts>hi:break
                    decoded+=1
                    if vf.pts==lo or vf.pts==hi or (vf.pts-lo)%1024==0:
                        gray=cv2.resize(vf.to_ndarray(format='gray'),(960,540),interpolation=cv2.INTER_AREA)
                        pictures.append((vf.pts,gray))
            if not pictures or pictures[0][0]!=lo or pictures[-1][0]!=hi:continue
            if pts1<pts0:pictures.reverse()
            states=[]
            for obs in byobs[seed['id']]:
                P=np.float32(obs['pixels_uv'])/2
                if len(P)>64:P=P[np.linspace(0,len(P)-1,64).astype(int)]
                point=obs['target']['kind']=='board_landmark'
                if point:
                    N=np.zeros_like(P);contrast=np.zeros(len(P))
                else:
                    N=normals(P,obs['target'])
                    contrast=(sample(pictures[0][1],(P+N)[:,None])-sample(pictures[0][1],(P-N)[:,None]))[:,0]
                states.append(dict(obs=obs,P=P,N=N,point=point,polarity=np.where(contrast>=0,1,-1),strength=np.abs(contrast),
                    valid=np.ones(len(P),bool),missing=np.zeros(len(P),int)))
            trace=[];accumulated=np.eye(3);complete=True
            for i in range(1,len(pictures)):
                prev=pictures[i-1][1];curr=pictures[i][1]
                H,diag=motion(prev,curr);total_processed+=1
                if H is None:
                    complete=False;trace.append(dict(pts=pictures[i][0],motion=diag,measured_points=0));break
                accumulated=H@accumulated
                measured_count=0
                for state in states:
                    P=state['P'];proposal=warp(P,H)
                    inside=(proposal[:,0]>14)&(proposal[:,0]<946)&(proposal[:,1]>14)&(proposal[:,1]<526)
                    inside &= ~((proposal[:,0]<320)&(proposal[:,1]<155))
                    inside &= ~((proposal[:,0]>810)&(proposal[:,1]>472))
                    if state['point']:
                        q,st,err=cv2.calcOpticalFlowPyrLK(prev,curr,P[:,None],proposal[:,None].copy(),winSize=(21,21),maxLevel=2,
                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
                        back,sb,_=cv2.calcOpticalFlowPyrLK(curr,prev,q,None,winSize=(21,21),maxLevel=2)
                        measured=q[:,0];fb=np.linalg.norm(back[:,0]-P,axis=1)
                        ok=(st[:,0]==1)&(sb[:,0]==1)&(fb<.6)&(err[:,0]<22)&(np.linalg.norm(measured-proposal,axis=1)<1.5)
                    else:
                        N=warp(P+state['N'],H)-proposal;N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-6)
                        measured,ok,strength=edge_resnap(curr,proposal,N,state['polarity'],state['strength'])
                        ok &= ~moving_occlusion_reject(prev,curr,P,proposal)
                        state['N']=N
                    ok &= inside & (state['missing']<3)
                    state['P']=np.where(ok[:,None],measured,proposal)
                    state['valid']=ok;state['missing']=np.where(ok,0,state['missing']+1)
                    measured_count+=int(ok.sum())
                trace.append(dict(pts=pictures[i][0],motion=diag,measured_points=measured_count))
                if not any(np.any(s['missing']<3) for s in states):complete=False;break
            rows=[]
            if complete:
                for j,state in enumerate(states):
                    ok=state['valid'];obs=state['obs']
                    if (state['point'] and ok.sum()!=1) or (not state['point'] and (ok.sum()<8 or np.linalg.norm(np.ptp(state['P'][ok],axis=0))<18)):continue
                    o=copy.deepcopy(obs);o.update(id=f"{target['id']}_continuous_{j:03d}",frame_id=target['id'],pixels_uv=np.round(state['P'][ok]*2,4).tolist(),
                        sigma_px=max(float(obs['sigma_px']),5.0),method='native within-shot 15fps forward/backward LK detail tracking or signed edge remeasurement with static-motion occlusion rejection',
                        source_seed_frame_id=seed['id'],source_seed_pts=seed['source_pts'],source_seed_observation_id=obs['id'],
                        source_seed_observation_file=str(OUT/'observations_direct.json'),
                        measurement_origin='continuous_native_image_remeasurement',
                        measurement_diagnostics=dict(native_decoded_frames=decoded,tracking_samples=len(pictures),pure_transfer_points_exported=False),
                        source_frame=dict(source_id=target['source_id'],source_pts=target['source_pts'],time_base='1/15360',image_path=target['image_path']))
                    rows.append(o)
            audit=dict(frame_id=target['id'],seed_frame_id=seed['id'],shot_id=seed['shot_id'],first_pts=pts0,last_pts=pts1,
                       native_decoded_frames=decoded,tracking_samples=len(pictures),complete_chain=complete,
                       recovered_groups=len(rows),steps=trace)
            audits.append(audit)
            if rows:
                candidates=rows;break
        if candidates:
            additions.extend(candidates);byobs[target['id']].extend(candidates)
            target['fit_status']='measured_by_continuous_tracking'
            # Initial pose is deliberately separate from pixel evidence.
            if 'initial_rvec_world_to_camera' in seed:
                camera=dict(K_px=[[seed['initial_focal_px'],0,959.5],[0,seed['initial_focal_px'],539.5],[0,0,1]],
                    R_world_to_camera=cv2.Rodrigues(np.array(seed['initial_rvec_world_to_camera']))[0].tolist())
                target.update(pose_from_motion(dict(id=seed['id'],camera=camera),accumulated))
            status.update(status='measured_by_continuous_tracking',measured_observation_groups=len(candidates),
                pixel_count=sum(len(o['pixels_uv']) for o in candidates),targets=[o['target'] for o in candidates],failure_reason=None)
        print(f"bridge #{target['pool_ordinal']} recovered={len(candidates)}",flush=True)
    # Preserve the pre-tracking measured input as exact provenance for chains.
    if not (OUT/'observations_direct.json').exists():save(OUT/'observations_direct.json',result)
    result['observations'].extend(additions)
    result['summary'].update(measured_frames=sum(bool(byobs[f['id']]) for f in result['frames']),
        observation_groups=len(result['observations']),observed_pixels=sum(len(o['pixels_uv']) for o in result['observations']),
        by_kind=dict(Counter(o['target']['kind'] for o in result['observations'])),
        continuous_tracking_added_groups=len(additions),continuous_tracking_processed_pairs=total_processed,
        continuous_tracking_recovered_frames=sum(x['status']=='measured_by_continuous_tracking' for x in statuses))
    save(OUT/'observations.json',result);save(OUT/'frame_status.json',statuses);save(OUT/'continuous_tracking.json',audits);save(OUT/'summary.json',result['summary'])
    for f in result['frames']:
        if f['fit_status']!='measured_by_continuous_tracking':continue
        im=cv2.imread(f['image_path'])
        for j,o in enumerate(byobs[f['id']]):
            color=[(0,220,255),(100,255,80),(255,100,220),(255,180,60)][j%4]
            for p in o['pixels_uv']:cv2.circle(im,tuple(np.rint(p).astype(int)),3,color,-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,55),(12,12,12),-1)
        cv2.putText(im,f"#{f['pool_ordinal']} {f['shot_id']} {f['source_seconds']:.3f}s  continuous measured groups={len(byobs[f['id']])}",(12,38),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,91])
    panels=[cv2.resize(cv2.imread(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg")),(480,270)) for f in result['frames']]
    for page in range((len(panels)+11)//12):
        pp=panels[page*12:(page+1)*12]
        while len(pp)%3:pp.append(np.zeros((270,480,3),np.uint8))
        cv2.imwrite(str(OUT/f'contact_{page+1:02d}.jpg'),np.vstack([np.hstack(pp[j:j+3]) for j in range(0,len(pp),3)]),[cv2.IMWRITE_JPEG_QUALITY,93])
    print(json.dumps(result['summary'],ensure_ascii=True),flush=True)

if __name__=='__main__':
    if '--bridge' in sys.argv:bridge_missing()
    elif '--initialize' in sys.argv:ensure_initializers()
    elif '--direct-visible' in sys.argv:direct_visible_features()
    elif '--repair-initializers' in sys.argv:repair_backward_initializers()
    elif '--finalize' in sys.argv:finalize_qa()
    elif '--correct-right-end' in sys.argv:correct_right_end_identities()
    elif '--correct-R2-sequence' in sys.argv:correct_red_row_from_sequence()
    else:run()
