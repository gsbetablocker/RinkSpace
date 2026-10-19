"""Image measurements for all 69 explicitly identified secondary-camera frames.

Instance appearance masks are visual review decisions. Homographies only locate
search patches; all exported pixels are locally remeasured. No scene projection
is exported as evidence. Conditional camera/landmark values are initializers.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import copy, hashlib, json, sys, argparse
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from extract_279_main_observations import image_features, match, measure_observation
from prepare_scene3d_calibration_observations import extract_ring
BASE=_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v015_joint')
POOL=BASE/'women_sp_training_frames_1000'
OUT=BASE/'joint_279/secondary'
OLD=BASE/'fits/joint_ads111_start0'

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(p,H):return cv2.perspectiveTransform(np.float32(p).reshape(-1,1,2),H).reshape(-1,2)

# Native contact sheets were reviewed before matching. These are occurrences of
# the physical upper-side instances, established by neighbours and architecture.
# Unresolved repeated short-end brands deliberately do not appear in this table.
VISIBLE={
 115:[927,937,938,1033,1034,1094,1103,1104,1105,1106,1107,1112,1116,1186,1187,1280,1281,1295,1296,1298,1313,1314,1315,1316],
 116:[927,937,1017,1034,1093,1111,1280,1296,1309,1310,1311,1312,1313,1314,1315,1316],
 111:[1086,1110,1175,1265,1268,1269],
 110:[1032,1036,1085,1086,1110,1175,1176,1177,1185,1198,1265,1266,1268],
 109:[939,1032,1101,1174,1176,1177,1185,1198,1262,1263,1264,1266],
 117:[1017,1093,1102,1111,1199,1200,1201,1202,1297,1309,1310,1311,1312,1316],
 108:[913,938,939,1032,1035,1101,1107,1112,1113,1114,1115,1116,1185,1198,1267,1280,1295],
}

def native_frame(row):
    p=POOL/row['native_image_path']
    assert digest(p)==row['native_image_sha256']
    return dict(id=f"secondary279_{row['source_pts']}",pool_ordinal=row['ordinal'],source_id=row['source_id'],
        source_pts=row['source_pts'],source_seconds=row['source_seconds'],source_time=row['source_time'],time_base='1/15360',
        source_path=str(ROOT/row['source_path']),source_sha256=row['source_sha256'],media_track_id=row['media_track_id'],
        pixel_frame_id=f"{row['media_track_id']}:pts:{row['source_pts']}:1920x1080",image_path=str(p),image_sha256=digest(p),
        image_size_wh=[1920,1080],principal_point_px=[959.5,539.5],fy_over_fx=1.,distortion_opencv=[0.,0.,0.,0.,0.],
        focal_bounds_px=[450.,30000.],shot_id=row['shot_id'],camera_id='camera_secondary_long_side',installation_id='women_sp_secondary',
        camera_identity_authority=row['camera_identity_authority'],camera_identity_source=row['camera_identity_source'],
        source_hash_verified_this_run=False,source_hash_verification='Inherited verified native manifest; image hash verified this run',
        previously_heldout_promoted=row['shot_id']=='S087',fit_status='pending_actual_measurements',
        visible_advertisement_instances_reviewed=[f'advert_ppt_object_{k}' for k,v in VISIBLE.items() if row['ordinal'] in v])

def sift(im,quad=None):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) if im.ndim==3 else im
    mask=np.zeros(gray.shape,np.uint8)
    if quad is not None:cv2.fillPoly(mask,[np.int32(quad)],255)
    else:
        mask[220:900]=255;mask[:330,:610]=0;mask[970:]=0
    kp,de=cv2.SIFT_create(nfeatures=5500,contrastThreshold=.013,edgeThreshold=15,sigma=1.3).detectAndCompute(gray,mask)
    if de is not None:de=np.sqrt(de/(de.sum(1,keepdims=True)+1e-8)).astype(np.float32)
    return np.float32([p.pt for p in kp]),de

def instance_match(a,b,quad):
    p,da=a;q,db=b
    if da is None or db is None or min(len(da),len(db))<8:return None
    bf=cv2.BFMatcher();fw=bf.knnMatch(da,db,k=2);bw=bf.knnMatch(db,da,k=2)
    rev={m.queryIdx:m.trainIdx for pair in bw if len(pair)==2 for m,n in [pair] if m.distance<.78*n.distance}
    good=[m for pair in fw if len(pair)==2 for m,n in [pair] if m.distance<.76*n.distance and rev.get(m.trainIdx)==m.queryIdx]
    if len(good)<7:return None
    pp=np.float32([p[m.queryIdx] for m in good]);qq=np.float32([q[m.trainIdx] for m in good])
    H,mask=cv2.findHomography(pp,qq,cv2.USAC_MAGSAC,3.0,maxIters=3000,confidence=.999)
    if H is None:return None
    ok=mask.ravel().astype(bool)
    if ok.sum()<7 or ok.mean()<.5:return None
    err=np.linalg.norm(transform(pp[ok],H)-qq[ok],axis=1)
    if np.ptp(pp[ok],axis=0)[0]<35 or np.ptp(pp[ok],axis=0)[1]<12 or np.percentile(err,90)>3.:return None
    corners=transform(quad,H)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners) or cv2.contourArea(corners)<500:return None
    if corners[:,0].min()<-450 or corners[:,0].max()>2370 or corners[:,1].min()<170 or corners[:,1].max()>940:return None
    # Orientation and local magnification must remain sensible, rejecting
    # folded fits to repeated ISU letters or a small repeated glyph.
    if corners[1,0]<=corners[0,0] or corners[2,1]<=corners[1,1]:return None
    return dict(H=H,quad=corners,inliers=int(ok.sum()),p90_px=float(np.percentile(err,90)),
        matched_source=pp[ok].tolist(),matched_target=qq[ok].tolist())

def patch_measure(source,target,uv,H):
    p=np.array(uv,float);q=transform([p],H)[0]
    if not (22<q[0]<1897 and 22<q[1]<1057):return None
    warped=cv2.warpPerspective(source,H,(1920,1080))
    a=cv2.getRectSubPix(warped,(17,17),tuple(map(float,q)))
    b=cv2.getRectSubPix(target,(29,29),tuple(map(float,q)))
    if a.std()<3:return None
    corr=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)
    _,value,_,loc=cv2.minMaxLoc(corr)
    if value<.80 or min(loc)<=0 or max(loc)>=12:return None
    rival=corr.copy();x,y=loc;rival[max(0,y-2):y+3,max(0,x-2):x+3]=-1
    if value-float(rival.max())<.018:return None
    measured=q+np.array(loc)-6
    # Actual reverse pixel match is required, not merely inverse(H)*q.
    inv=np.linalg.inv(H);backguess=transform([measured],inv)[0]
    reverse=cv2.warpPerspective(target,inv,(1920,1080))
    pa=cv2.getRectSubPix(reverse,(17,17),tuple(map(float,backguess)))
    pb=cv2.getRectSubPix(source,(29,29),tuple(map(float,backguess)))
    _,backscore,_,backloc=cv2.minMaxLoc(cv2.matchTemplate(pb,pa,cv2.TM_CCOEFF_NORMED))
    back=backguess+np.array(backloc)-6
    if backscore<.76 or min(backloc)<=0 or max(backloc)>=12 or np.linalg.norm(back-p)>2.5:return None
    return measured,dict(ncc=float(value),reverse_ncc=float(backscore),forward_backward_px=float(np.linalg.norm(back-p)))

def obs_point(frame,iid,landmark,uv,method,quality,seed_id):
    return dict(id=f"{frame['id']}_{landmark}",frame_id=frame['id'],type='point',
        target=dict(kind='board_landmark',landmark_id=landmark),physical_group_id=iid,advertisement_instance_id=iid,
        use='check' if iid.endswith('_108') else 'fit',pixels_uv=[np.round(uv,4).tolist()],sigma_px=3.,
        surface_class='board_surface_detail',measurement_origin='actual_image_measurement',method=method,
        synthetic_or_model_projected=False,pure_transfer_points_exported=False,source_seed_frame_id=seed_id,
        source_pts=frame['source_pts'],time_base='1/15360',measurement_diagnostics=quality,
        correlation_group_id=f"{frame['shot_id']}_{iid}",identity_evidence='user numbered-instance mapping, visible neighbouring instances and stand/window architecture; per-frame reviewed occurrence table',
        independent_validation=False)

def color_corners(im,rough,color):
    """Intersections of independently measured printed colour boundary lines."""
    im=im.astype(np.float32)
    field=(255-cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) if color=='dark' else
           im[:,:,2]-.5*(im[:,:,0]+im[:,:,1]) if color=='red' else im[:,:,0]-.5*(im[:,:,1]+im[:,:,2]))
    field=cv2.GaussianBlur(field,(0,0),.8)
    rough=np.array(rough,float);lines=[];counts=[]
    for i in range(4):
        a,b=rough[i],rough[(i+1)%4];d=b-a;n=np.array([d[1],-d[0]])/np.linalg.norm(d)
        base=a+np.linspace(.16,.84,34)[:,None]*d;ofs=np.linspace(-7,7,57)
        xy=base[:,None]+ofs[None,:,None]*n
        def sample(p):return cv2.remap(field,p[:,:,0].astype(np.float32),p[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        g=sample(xy-n*1.5)-sample(xy+n*1.5)
        idx=np.argmax(g-.13*np.abs(ofs),axis=1);strength=g[np.arange(len(idx)),idx]
        keep=(idx>1)&(idx<len(ofs)-2)&(strength>3.)
        points=xy[np.arange(len(idx)),idx][keep]
        if len(points)<16:return None
        vx,vy,x,y=cv2.fitLine(points.astype(np.float32),cv2.DIST_WELSCH,0,.01,.01).ravel()
        n=np.array([-vy,vx]);n/=np.linalg.norm(n);line=np.r_[n,-np.dot(n,[x,y])]
        distance=np.abs(points@line[:2]+line[2]);points=points[distance<2.5]
        if len(points)<16 or np.ptp(points,axis=0).max()<np.linalg.norm(d)*.40:return None
        lines.append(line);counts.append(dict(count=len(points),edge_pixels_uv=points.tolist(),median_gradient=float(np.median(strength[keep]))))
    corners=[]
    for i in range(4):
        p=np.cross(lines[(i-1)%4],lines[i])
        if abs(p[2])<1e-5:return None
        corners.append(p[:2]/p[2])
    corners=np.array(corners)
    if np.max(np.linalg.norm(corners-rough,axis=1))>18:return None
    return corners,dict(method='intersections of four independently measured colour-edge lines',edges=counts)

def initial_pose(frame,observations,landmarks):
    byid={p['id']:p for p in landmarks};xyz=[];uv=[]
    for o in observations:
        if o['use']!='fit' or o['target']['kind']!='board_landmark':continue
        p=byid.get(o['target']['landmark_id'])
        if p is None or p.get('side')!='upper_long_side' or p.get('allow_corner'):continue
        xyz.append([p.get('along',p.get('initial_along',0)),50.,p.get('z_fraction',p.get('initial_z_fraction',.5))*3.3]);uv.append(o['pixels_uv'][0])
    # Enough common measured points to seed pan/tilt/f, while centre stays a
    # conditional guess. Parent joint fit frees each installation centre.
    if len(xyz)<4:return None
    xyz=np.array(xyz);uv=np.array(uv);center=np.array([53.,-80.,8.]);target=np.mean(xyz,axis=0)
    forward=target-center;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    R=np.array([right,down,forward]);r=cv2.Rodrigues(R)[0].ravel()
    def residual(x):
        rot=cv2.Rodrigues(x[:3])[0];cam=(xyz-center)@rot.T
        f=np.exp(x[3]);pred=cam[:,:2]/np.maximum(cam[:,2,None],1e-3)*f+[959.5,539.5]
        return (pred-uv).ravel()
    best=None
    for focal in [1800.,3500.,7000.]:
        result=least_squares(residual,np.r_[r,np.log(focal)],bounds=(np.r_[[-7]*3,np.log(450)],np.r_[[7]*3,np.log(30000)]),max_nfev=140,loss='soft_l1',f_scale=4)
        rms=float(np.sqrt(np.mean(residual(result.x)**2)))
        if best is None or rms<best[0]:best=(rms,result)
    rms,result=best
    if rms>35:return None
    return dict(initial_rvec_world_to_camera=result.x[:3].tolist(),initial_focal_px=float(np.exp(result.x[3])),
        initialization_diagnostics=dict(method='conditional R/f initialization from real board pixels and previous uncertain point estimates',
            center_guess=[53,-80,8],fitted_pixel_rms=rms,points=len(xyz),physical_accuracy_verified=False,check_pixels_used=False))

def fallback_pose(frame,observations,landmarks):
    """A deliberately coarse look direction, never a camera position claim."""
    center=np.array([53.,-80.,8.]);byid={p['id']:p for p in landmarks};targets=[];pixels=[]
    for o in observations:
        if o.get('use')!='fit':continue
        t=o['target'];point=None
        if t['kind']=='board_landmark' and t['landmark_id'] in byid:
            p=byid[t['landmark_id']];along=p.get('along',p.get('initial_along',0));height=p.get('z_fraction',p.get('initial_z_fraction',.5))*3.3
            if p['side']=='upper_long_side':point=[along,50,height]
        elif t['kind']=='board_side':
            side=t['side'];height=3.3 if t.get('edge')=='top' else 0.
            point={'upper_long_side':[0,50,height],'lower_long_side':[0,-50,height],'left_end':[-100,0,height],'right_end':[100,0,height]}.get(side)
        if point is not None:targets.append(point);pixels.append(np.median(o['pixels_uv'],axis=0))
    if not targets:
        number=next((k for k in [115,116,109,110,111,117,108] if frame['pool_ordinal'] in VISIBLE[k]),None)
        targets=[{115:[48,50,2],116:[78,50,2],109:[-35,50,2],110:[-63,50,2],111:[-91,47,2],117:[95,43,2],108:[0,50,2]}.get(number,[0,20,0])]
        pixels=[[959.5,450.]]
    aim=np.mean(targets,axis=0);pixel=np.mean(pixels,axis=0);forward=aim-center;distance=np.linalg.norm(forward);forward/=distance
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.array([right,down,forward]);focal=3600.
    top=[o for o in observations if o['target']['kind']=='board_side' and o['target'].get('edge')=='top']
    bottom=[o for o in observations if o['target']['kind']=='board_side' and o['target'].get('edge')=='contact']
    if top and bottom:
        delta=float(abs(np.median(top[0]['pixels_uv'],axis=0)[1]-np.median(bottom[0]['pixels_uv'],axis=0)[1]));focal=float(np.clip(delta*distance/3.3,900,18000))
    ray=np.r_[(pixel-[959.5,539.5])/focal,1.];ray/=np.linalg.norm(ray)
    axis=np.cross([0.,0.,1.],ray);norm=np.linalg.norm(axis)
    if norm>1e-8:R=cv2.Rodrigues(axis/norm*np.arctan2(norm,ray[2]))[0]@R
    return dict(initial_rvec_world_to_camera=cv2.Rodrigues(R)[0].ravel().tolist(),initial_focal_px=focal,
        initialization_diagnostics=dict(method='coarse view direction toward observed instance/side; numerical initializer only',
            center_guess=center.tolist(),aim_guess_xyz=aim.tolist(),pixel_centroid=pixel.tolist(),focal_guess_px=focal,
            fitted_to_pixels=False,physical_accuracy_verified=False,check_pixels_used=False,unconstrained_camera_and_field_remain_free=True))

def add_initializers():
    data=load(OUT/'observations.json');by=defaultdict(list)
    for o in data['observations']:by[o['frame_id']].append(o)
    for f in data['frames']:
        if 'initial_rvec_world_to_camera' not in f:f.update(fallback_pose(f,by[f['id']],data['board_landmarks']))
    data['summary']['frames_with_pose_initializer']=len(data['frames'])
    data['summary']['coarse_view_direction_initializers']=sum(not f['initialization_diagnostics'].get('fitted_pixel_rms') for f in data['frames'])
    for st in data['frame_status']:st['pose_initializer_available']=True
    save(OUT/'observations.json',data);save(OUT/'summary.json',data['summary'])
    print(json.dumps(data['summary']),flush=True)

def direct_board_edge(image,edge,xlimits=(16,1904)):
    """Measure the actual upper blue cap / lower yellow kickplate boundary."""
    im=image.astype(np.float32)
    field=im[:,:,0]-.5*(im[:,:,1]+im[:,:,2]) if edge=='top' else .5*(im[:,:,1]+im[:,:,2])-im[:,:,0]
    field=cv2.GaussianBlur(field,(0,0),1.)
    mask=np.uint8((field>(12 if edge=='top' else 11))*255)
    mask[:210]=0;mask[950:]=0;mask[:330,:610]=0
    mask[:,:xlimits[0]]=0;mask[:,xlimits[1]:]=0
    lines=cv2.HoughLinesP(mask,1,np.pi/1800,150,minLineLength=450,maxLineGap=110)
    if lines is None:return None
    candidates=[]
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(x2-x1)<450 or abs(y2-y1)>abs(x2-x1)*.14:continue
        m=(y2-y1)/(x2-x1);b=y1-m*x1
        if edge=='top' and b>720:continue
        # Global support across independent columns identifies the structural
        # strip rather than an isolated ad rectangle or moving clothing.
        xx=np.arange(xlimits[0],xlimits[1],20,dtype=float);yy=m*xx+b
        sampled=cv2.remap(field,xx[None].astype(np.float32),yy[None].astype(np.float32),cv2.INTER_LINEAR)[0]
        support=sampled>12
        if support.sum()<28 or np.ptp(xx[support])<600:continue
        candidates.append((support.sum()-.008*b,m,b,xx))
    if not candidates:return None
    _,m,b,xx=max(candidates,key=lambda a:a[0]);points=[];strength=[]
    for x in xx:
        y=m*x+b
        if x<610 and y<335:continue
        # Search outer boundary of the coloured strip, with known polarity.
        offsets=np.arange(-17,18,.5);yy=y+offsets;xxs=np.full(yy.shape,x)
        def smp(ys):return cv2.remap(field,xxs[None].astype(np.float32),ys[None].astype(np.float32),cv2.INTER_LINEAR)[0]
        gradient=smp(yy+2)-smp(yy-2) if edge=='top' else smp(yy-2)-smp(yy+2)
        k=int(np.argmax(gradient))
        if k<2 or k>len(offsets)-3 or gradient[k]<6:continue
        points.append([x,yy[k]]);strength.append(float(gradient[k]))
    if len(points)<24:return None
    points=np.array(points);vx,vy,x,y=cv2.fitLine(points.astype(np.float32),cv2.DIST_WELSCH,0,.01,.01).ravel()
    distance=np.abs((points-[x,y])@np.array([-vy,vx]));keep=distance<2.8
    if keep.sum()<24 or np.ptp(points[keep,0])<600:return None
    return points[keep],dict(method='direct structural colour-strip Hough search plus signed local gradient maxima; no projected samples',
        measured_colour_gradient_median=float(np.median(np.array(strength)[keep])),line_distance_p95_px=float(np.percentile(distance[keep],95)),
        endpoints_are_image_crop_not_physical_correspondences=True,edge_definition='upper edge of blue board cap' if edge=='top' else 'bottom edge of yellow kickplate at ice')

def augment_direct():
    data=load(OUT/'observations.json');by=defaultdict(list);changes=[]
    for o in data['observations']:by[o['frame_id']].append(o)
    # Adjacency was established on native frames. These ranges only include
    # confirmed straight upper-side structure; curved/end-only frames omitted.
    upper={n for k in [109,110,115,108] for n in VISIBLE[k]}
    # Native-image QA found horizontal reflection/line texture above the real
    # grey outer ring in 1104 and 1106. Exclude those image regions independently
    # of any model residual; retain the visible paint arcs elsewhere.
    ring_specs={1104:([572,864,852,78,-1],[[900,580,1230,1000],[0,770,930,805]]),
                1105:([260,836,830,72,-1],[[600,430,1190,950]]),
                1106:([185,877,900,86,0],[[920,40,1280,1030],[0,789,700,818]])}
    for f in data['frames']:
        im=cv2.imread(f['image_path']);n=f['pool_ordinal'];added=[]
        if n in upper:
            # Exclude nearby upper-left transition when its blue panel is in
            # frame. Geometry beyond the observed range remains unconstrained.
            xlimits=(650,1904) if n in [1086,1110,1175,1265,1268] else (16,1904)
            for edge in ['top','contact']:
                if any(o['target'].get('kind')=='board_side' and o['target'].get('side')=='upper_long_side' and o['target'].get('edge')==edge for o in by[f['id']]):continue
                measured=direct_board_edge(im,edge,xlimits)
                if measured is None:continue
                points,quality=measured
                target=dict(kind='board_side',side='upper_long_side',edge=edge)
                if n==1316:target['continue_corner']='upper_right'
                o=dict(id=f"{f['id']}_direct_board_{edge}",frame_id=f['id'],type='curve',target=target,
                    use='fit',pixels_uv=np.round(points,4).tolist(),sigma_px=3.,physical_group_id='upper_long_side_'+edge,
                    measurement_origin='actual_image_measurement',method=quality['method'],measurement_diagnostics=quality,
                    source_pts=f['source_pts'],time_base='1/15360',synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                    correlation_group_id=f"{f['shot_id']}_upper_long_side_{edge}",identity_evidence='straight stand-side boards identified by upper-ad instance sequence and architecture')
                added.append(o)
        if n in ring_specs:
            ellipse,boxes=ring_specs[n]
            spec=dict(feature_id='R3',seed_ellipse_cx_cy_a_b_angle=ellipse,exclude_boxes=boxes,
                partial_no_ellipse_refit=True,min_gradient_strength=2.5)
            try:
                o=extract_ring(im,spec)
                if len(o['pixels_uv'])>=18:
                    o.update(id=f"{f['id']}_direct_R3",frame_id=f['id'],type='curve',target=dict(kind='circle_outline',feature_id='R3'),
                        feature_id='R3',physical_group_id='R3',use='fit',sigma_px=max(6.,o['sigma_px']),source_pts=f['source_pts'],time_base='1/15360',
                        measurement_origin='actual_image_measurement',synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                        correlation_group_id=f"{f['shot_id']}_R3",identity_evidence='same grey/red-centre house under user3 advert115 as old main S027 R3; nearby red house and blue-line order distinguish row')
                    added.append(o)
            except ValueError:pass
        if added:
            existing={o['id'] for o in data['observations']};added=[o for o in added if o['id'] not in existing]
            data['observations']+=added;by[f['id']]+=added
            changes.append(dict(ordinal=n,added_groups=len(added),added_pixels=sum(len(o['pixels_uv']) for o in added),targets=[o['target'] for o in added]))
    for f,st in zip(data['frames'],data['frame_status']):
        local=by[f['id']];f['fit_status']='measured' if local else 'no_reliable_measurement'
        st.update(status=f['fit_status'],observations=len(local),targets=[o['target'] for o in local])
        # Better focal guess becomes available when direct top/contact exists.
        if not f.get('initialization_diagnostics',{}).get('fitted_pixel_rms'):f.update(fallback_pose(f,local,data['board_landmarks']))
    oo=data['observations'];data['summary'].update(measured_frames=sum(bool(by[f['id']]) for f in data['frames']),observation_groups=len(oo),
        observed_pixels=sum(len(o['pixels_uv']) for o in oo),by_kind=dict(Counter(o['target']['kind'] for o in oo)),
        by_instance=dict(Counter(o.get('advertisement_instance_id','ice_or_board') for o in oo)))
    data['direct_measurement_additions']=changes
    save(OUT/'observations.json',data);save(OUT/'summary.json',data['summary']);save(OUT/'direct_additions.json',changes)
    panels=[]
    for f in data['frames']:
        im=cv2.imread(f['image_path']);local=by[f['id']]
        for j,o in enumerate(local):
            color=(60,240,70) if o['target']['kind']=='circle_outline' else [(0,255,255),(255,150,80),(80,255,70),(255,80,220)][j%4]
            for uv in o['pixels_uv']:cv2.circle(im,tuple(np.rint(uv).astype(int)),3,color,-1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,52),(15,15,15),-1);cv2.putText(im,f"#{f['pool_ordinal']} {f['shot_id']} {f['source_time']} actual groups {len(local)}",(12,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
        cv2.imwrite(str(OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,92]);panels.append(cv2.resize(im,(480,270)))
    for start in range(0,len(panels),12):
        pp=panels[start:start+12]
        while len(pp)%3:pp.append(np.zeros((270,480,3),np.uint8))
        cv2.imwrite(str(OUT/f'contact_{start//12+1:02d}.jpg'),np.vstack([np.hstack(pp[k:k+3]) for k in range(0,len(pp),3)]),[cv2.IMWRITE_JPEG_QUALITY,92])
    print(json.dumps(data['summary']),flush=True)

def extra_corner_instances():
    """Add independently observed print details on upper corner instances."""
    data=load(OUT/'observations.json');frames={f['pool_ordinal']:f for f in data['frames']};added=[];landmarks=data['board_landmarks']
    # User7 belongs to the upper-left transition, witnessed next to user8.
    f=frames[1269];im=cv2.imread(f['image_path']);result=color_corners(im,[[737,403],[1531,403],[1531,530],[737,525]],'dark')
    if result is not None:
        corners,quality=result
        old={p['id']:p for p in load(OLD/'fit.json')['board_landmarks']}
        for j,p in enumerate(corners):
            name=f'advert_ppt_object_111_printed_corner_{j}'
            added.append(obs_point(f,'advert_ppt_object_111',name,p,'independent native-pixel printed colour-edge measurement',quality,f['id']))
            if not any(l['id']==name for l in landmarks):
                landmark=copy.deepcopy(old[name]);landmark['allow_corner']='upper_left';landmarks.append(landmark)
    # User1: ALL SKATING / ONE APP detail patch, below the distinctive purple
    # entrance platform. Other APP panels are not searched or merged.
    seed=frames[1297];im=cv2.imread(seed['image_path']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    polygon=[[145,541],[642,541],[642,704],[145,704]];a=sift(im,polygon)
    points=[]
    for p in a[0]:
        if any(np.linalg.norm(p-q)<18 for q in points):continue
        if not (160<p[0]<625 and 551<p[1]<693):continue
        points.append(p)
        if len(points)>=18:break
    definitions=[]
    for j,p in enumerate(points):
        lid=f'advert_ppt_object_117_secondary1297_detail_{j:03d}'
        definitions.append((lid,p))
        if not any(l['id']==lid for l in landmarks):
            fraction=float(.88-(p[0]-160)/465*.18)
            # Preserve the former rough quarter-circle start geometry while
            # allowing the point to move onto its adjacent upper straight.
            # User1 identifies this vicinity, not the exact tangent point.
            radius_guess=20.833
            along_guess=100-radius_guess+radius_guess*np.cos(fraction*np.pi/2)
            landmarks.append(dict(id=lid,advertisement_instance_id='advert_ppt_object_117',side='upper_long_side',allow_corner='upper_right',
                along_parameterization='world_x',initial_along=float(along_guess),bounds_along=[-99.9,99.9],
                initial_z_fraction=float(np.clip(.8-(p[1]-475)/255*.6,.05,.95)),bounds_z_fraction=[.01,.99],
                initialization_only=True,xyz_measured=False,initialization_basis='same-instance upper-right vicinity; tangent membership unknown, adjoining straight and corner both allowed; coordinate/height guesses only'))
    audit=[]
    for ordinal in VISIBLE[117]:
        f=frames.get(ordinal)
        if f is None:continue
        target=cv2.imread(f['image_path']);gg=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
        proposal=dict(H=np.eye(3),inliers=0) if ordinal==1297 else instance_match(a,sift(target),polygon)
        if proposal is None:
            audit.append(dict(ordinal=ordinal,status='no_same_detail_search'));continue
        count=0
        for name,p in definitions:
            result=(p,dict(direct_native_SIFT_keypoint=True)) if ordinal==1297 else patch_measure(gray,gg,p,proposal['H'])
            if result is None:continue
            uv,quality=result
            added.append(obs_point(f,'advert_ppt_object_117',name,uv,'native printed-detail keypoint or same-instance search plus bidirectional patch measurement',quality,seed['id']));count+=1
        audit.append(dict(ordinal=ordinal,status='measured' if count else 'no_local_pixel_support',measurements=count,inliers=proposal['inliers']))
    existing={o['id'] for o in data['observations']};added=[o for o in added if o['id'] not in existing]
    data['observations']+=added
    used={o['target']['landmark_id'] for o in data['observations'] if o['target']['kind']=='board_landmark'}
    data['board_landmarks']=[p for p in landmarks if p['id'] in used]
    data['corner_instance_additions']=dict(added_groups=len(added),app117_reidentification=audit)
    save(OUT/'observations.json',data)
    # This recomputes the ledger and every overlay, and adds no duplicate edge.
    augment_direct()
    print('corner_instance_additions',len(added),flush=True)

def partial_bright_corners(image,rough):
    field=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.)
    q=np.array(rough,float);lines={};diagnostics={}
    for i in range(4):
        a,b=q[i],q[(i+1)%4];d=b-a;n=np.array([d[1],-d[0]])/np.linalg.norm(d)
        base=a+np.linspace(.12,.88,max(40,int(np.linalg.norm(d)/4)))[:,None]*d;ofs=np.arange(-10.,10.01,.5);xy=base[:,None]+ofs[None,:,None]*n
        valid=(base[:,0]>14)&(base[:,0]<1905)&(base[:,1]>14)&(base[:,1]<1065)
        def samp(p):return cv2.remap(field,p[:,:,0].astype(np.float32),p[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        gradient=samp(xy-n*2)-samp(xy+n*2);idx=np.argmax(gradient-.06*np.abs(ofs),axis=1);strength=gradient[np.arange(len(idx)),idx]
        keep=valid&(idx>1)&(idx<len(ofs)-2)&(strength>5.)
        points=xy[np.arange(len(idx)),idx][keep]
        if len(points)<16:continue
        vx,vy,x,y=cv2.fitLine(points.astype(np.float32),cv2.DIST_WELSCH,0,.01,.01).ravel();nn=np.array([-vy,vx]);nn/=np.linalg.norm(nn);line=np.r_[nn,-np.dot(nn,[x,y])]
        distance=np.abs(points@line[:2]+line[2]);points=points[distance<2.5]
        if len(points)<16:continue
        lines[i]=line;diagnostics[i]=dict(count=len(points),edge_pixels_uv=points.tolist())
    result=[]
    for i in range(4):
        if i not in lines or (i-1)%4 not in lines:continue
        p=np.cross(lines[(i-1)%4],lines[i]);p=p[:2]/p[2]
        if not(8<p[0]<1910 and 8<p[1]<1070) or np.linalg.norm(p-q[i])>22:continue
        result.append((i,p,dict(adjacent_edges=[diagnostics[(i-1)%4],diagnostics[i]],partial_corner_visible=True)))
    return result

def fill_visible_remaining():
    data=load(OUT/'observations.json');frames={f['pool_ordinal']:f for f in data['frames']};added=[]
    # The advertisement sequence repeats at the opposite end. Architecture
    # distinguishes 1016 (window wall and clock, right) from 1184 (continuous
    # translucent protection panels, left); names/order alone are insufficient.
    for ordinal,xlimits in [(1016,(16,1016)),(1184,(16,1904))]:
        f=frames[ordinal];im=cv2.imread(f['image_path'])
        side='right_end' if ordinal==1016 else 'left_end'
        for edge in ['top','contact']:
            result=direct_board_edge(im,edge,xlimits)
            if result is None:continue
            p,quality=result
            added.append(dict(id=f"{f['id']}_direct_left_{edge}",frame_id=f['id'],type='curve',target=dict(kind='board_side',side=side,edge=edge),
                pixels_uv=np.round(p,4).tolist(),sigma_px=4.,use='fit',physical_group_id=side+'_'+edge,
                source_pts=f['source_pts'],time_base='1/15360',method=quality['method'],measurement_diagnostics=quality,
                identity_evidence='1016: window-wall clock and red cabinet distinguish right end. 1184: continuous translucent panels distinguish left end. Repeated advertisement order is not a side identifier.',
                measurement_origin='actual_image_measurement',synthetic_or_model_projected=False,pure_transfer_points_exported=False,
                correlation_group_id=f"{f['shot_id']}_{side}_{edge}"))
    # Only the QR's white printed outer border, never a predicted rectangle.
    # Partial crop1093 can supply its two left corners independently.
    search={1017:[[1618,496],[1900,496],[1900,731],[1618,731]],
            1093:[[1760,348],[1960,348],[1960,500],[1760,500]],
            1111:[[865,477],[980,477],[980,596],[865,596]],
            1199:[[56,437],[167,437],[167,535],[56,535]],
            1202:[[467,354],[670,354],[670,559],[467,559]],
            1201:[[432,344],[634,344],[634,550],[432,550]]}
    for ordinal,quad in search.items():
        f=frames[ordinal];im=cv2.imread(f['image_path']);measured=partial_bright_corners(im,quad)
        for j,p,quality in measured:
            name=f'advert_ppt_object_117_qr_white_corner_{j}'
            observation=obs_point(f,'advert_ppt_object_117',name,p,'intersection of separately measured QR white-border edges; cropped corners omitted',quality,f['id'])
            observation['sigma_px']=8.0  # broadcast pan blur broadens the white edge
            added.append(observation)
            if not any(l['id']==name for l in data['board_landmarks']):
                data['board_landmarks'].append(dict(id=name,advertisement_instance_id='advert_ppt_object_117',side='upper_long_side',allow_corner='upper_right',
                    initial_along=87. if j in [0,3] else 89.,bounds_along=[50.,99.9],initial_z_fraction=.81 if j<2 else .16,bounds_z_fraction=[.01,.99],
                    initialization_only=True,xyz_measured=False,initialization_basis='user1 upper-right board instance; QR span guessed only; can lie on adjoining straight/corner surface'))
    # Last frame1102 exposes two actual right corners of the same APP print;
    # reuse low-camera IDs/parameterization instead of inventing new points.
    f=frames[1102];im=cv2.imread(f['image_path'])
    measured=partial_bright_corners(255-im,[[-600,473],[245,475],[245,640],[-600,640]])
    lowpath=BASE/'joint_279/low/observations.json'
    lowdefs={p['id']:p for p in load(lowpath)['board_landmarks']} if lowpath.exists() else {}
    for j,p,quality in measured:
        if j not in [1,2]:continue
        name=f'advert_ppt_object_117_color_corner_{j}'
        if name not in lowdefs:continue
        o=obs_point(f,'advert_ppt_object_117',name,p,'partial APP printed dark/white border edge intersection',quality,f['id']);o['sigma_px']=5.
        added.append(o)
        if not any(l['id']==name for l in data['board_landmarks']):data['board_landmarks'].append(copy.deepcopy(lowdefs[name]))
    existing={o['id'] for o in data['observations']};added=[o for o in added if o['id'] not in existing]
    data['observations']+=added
    for o in data['observations']:
        if '_qr_white_corner_' in o['target'].get('landmark_id',''):o['sigma_px']=8.
    data['remaining_visible_additions']=[dict(id=o['id'],frame_id=o['frame_id'],target=o['target']) for o in added]
    save(OUT/'observations.json',data);augment_direct();print('remaining additions',len(added),flush=True)

def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'overlays').mkdir(exist_ok=True)
    cv2.setNumThreads(2);cv2.setRNGSeed(17)
    rows=[r for r in load(POOL/'training_input_manifest.json')['frames'] if r.get('camera_id')=='secondary_long_side']
    assert len(rows)==69
    frames=[native_frame(r) for r in rows];byordinal={f['pool_ordinal']:f for f in frames}
    problem=load(OLD/'problem.json');oldfit=load(OLD/'fit.json');oldframes={f['id']:f for f in problem['frames']}
    landmarks=copy.deepcopy(oldfit['board_landmarks']);landmark_ids={p['id'] for p in landmarks}
    images={f['id']:cv2.imread(f['image_path']) for f in frames};templates=[]
    # Existing true printed details bridge secondary to already measured main
    # and high views. Their IDs and common unknown 3D positions stay shared.
    for number,fid,quad in [(115,'main_69427200',[[371,328],[782,323],[783,400],[373,410]]),
                            (116,'main_69427200',[[1045,319],[1454,312],[1455,381],[1045,392]]),
                            (111,'main_64458240',[[1172,457],[1565,459],[1564,518],[1176,513]])]:
        frame=oldframes[fid];im=cv2.imread(frame['image_path']);iid=f'advert_ppt_object_{number}'
        points=[(o['target']['landmark_id'],o['pixels_uv'][0]) for o in problem['observations'] if o['frame_id']==fid and o['target']['kind']=='board_landmark' and o.get('advertisement_instance_id')==iid]
        templates.append(dict(number=number,frame=frame,image=im,gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),quad=quad,
            features=sift(im,quad),points=points,color='blue'))
    # New physical print edges; all corners remain independent board-surface
    # latent points. Initial along ranges come from schematic identity only.
    for number,ordinal,quad,color,along in [(110,1032,[[208,419],[555,423],[555,522],[207,513]],'red',[-74,-52]),
                                           (109,1263,[[48,550],[373,553],[372,621],[48,618]],'blue',[-39,-31])]:
        frame=byordinal[ordinal];im=images[frame['id']];measured=color_corners(im,quad,color)
        if measured is None:raise RuntimeError(f'New colour seed {number} not measurable')
        corners,quality=measured;points=[];suffix='blue_corner' if number==109 else 'color_corner'
        for i,uv in enumerate(corners):
            name=f'advert_ppt_object_{number}_{suffix}_{i}';points.append((name,uv.tolist()))
            landmarks.append(dict(id=name,advertisement_instance_id=f'advert_ppt_object_{number}',side='upper_long_side',
                initial_along=float(along[0 if i in [0,3] else 1]),bounds_along=[-99.,99.],initial_z_fraction=.78 if i<2 else .28,
                bounds_z_fraction=[.01,.99],along_prior_sigma=None,z_fraction_prior_sigma=None,
                initialization_only=True,xyz_measured=False,initialization_basis='PPT instance order and rough span, never a measured coordinate'))
        templates.append(dict(number=number,frame=frame,image=im,gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),quad=corners.tolist(),
            features=sift(im,[[0,535],[680,535],[680,790],[0,790]] if number==109 else quad),points=points,color=color,new=True,seed_quality=quality))
    observations=[];status=[];logs=[];perframe={}
    for n,frame in enumerate(frames):
        im=images[frame['id']];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);dest=sift(im);local=[];attempts=[]
        for t in templates:
            number=t['number'];iid=f'advert_ppt_object_{number}'
            if frame['pool_ordinal'] not in VISIBLE[number]:continue
            if frame['id']==t['frame']['id']:
                m=dict(H=np.eye(3),quad=np.array(t['quad']),inliers=0,p90_px=0.,seed_direct=True)
            else:m=instance_match(t['features'],dest,t['quad'])
            if m is None:
                attempts.append(dict(instance=iid,status='no_reliable_same_instance_homography'));continue
            count=0
            # Full colour boundary corners need their own independently measured
            # edges. A homography corner alone is NEVER a pixel observation.
            cm=color_corners(im,m['quad'],t['color'])
            if cm is not None and (t.get('new') or number in [115,116,111]):
                corners,quality=cm
                suffix='blue_corner' if number==109 else ('printed_corner' if number==111 else 'color_corner')
                for j,uv in enumerate(corners):
                    # A line intersection outside the image is extrapolation,
                    # even if the two edge fragments were themselves visible.
                    if not (1<=uv[0]<1919 and 1<=uv[1]<1079):continue
                    name=f'{iid}_{suffix}_{j}'
                    local.append(obs_point(frame,iid,name,uv,'measured colour boundary intersections',quality,t['frame']['id']));count+=1
                    if name not in landmark_ids and not any(p['id']==name for p in landmarks):
                        # 115/116 physical corners are also measured by the low
                        # extractor; use exactly the agreed cross-camera IDs.
                        bounds=[-99.,99.];rough=(38.,62.) if number==115 else ((-96.,-80.) if number==111 else (70.,83.))
                        landmarks.append(dict(id=name,advertisement_instance_id=iid,side='upper_long_side',
                            initial_along=rough[0 if j in [0,3] else 1],bounds_along=bounds,
                            initial_z_fraction=.82 if j<2 else .18,bounds_z_fraction=[.01,.99],
                            initialization_only=True,xyz_measured=False,
                            **({'allow_corner':'upper_left'} if number==111 else {})))
            # Original detail IDs only. Color corner points are handled above.
            if not t.get('new'):
                for lid,p in t['points']:
                    if number==111:continue
                    pm=patch_measure(t['gray'],gray,p,m['H'])
                    if pm is None:continue
                    uv,quality=pm;local.append(obs_point(frame,iid,lid,uv,'same-instance mutual RootSIFT search plus bidirectional unique image-patch remeasurement',quality,t['frame']['id']));count+=1
            attempts.append(dict(instance=iid,status='measured' if count else 'search_only_no_local_pixel_support',
                inliers=m['inliers'],p90_px=m['p90_px'],observations=count,search_quad=m['quad'].tolist()))
        local={o['id']:o for o in local};local=list(local.values())
        perframe[frame['id']]=local;observations.extend(local)
        status.append(dict(frame_id=frame['id'],ordinal=frame['pool_ordinal'],source_time=frame['source_time'],shot_id=frame['shot_id'],
            observations=len(local),status='measured' if local else 'no_reliable_measurement',attempts=attempts,
            failure_reason=None if local else 'Visible pixels lacked accepted same-instance matching / unoccluded colour edges; unknown repeated short-end instances not guessed'))
        logs.append(dict(frame_id=frame['id'],attempts=attempts))
        if (n+1)%10==0:
            print(f'{n+1}/69 measured {sum(bool(perframe[f["id"]]) for f in frames[:n+1])} obs {len(observations)}',flush=True)
            save(OUT/'partial_status.json',status)
    # Reuse broad image matching only to propose locally observed existing ice
    # markings / physical board edges. No projected geometry supplies pixels.
    seeds=[]
    for f in problem['frames']:
        if f.get('installation_id')!='women_sp_main':continue
        oo=[o for o in problem['observations'] if o['frame_id']==f['id'] and o['target']['kind'] in ['circle_outline','line_segment','board_side','board_corner']]
        if oo:seeds.append((f,oo))
    sf=[image_features(f) for f,oo in seeds]
    for n,frame in enumerate(frames):
        target=image_features(frame);best={}
        for (seed,oo),source in zip(seeds,sf):
            proposal=match(source,target)
            if proposal is None:continue
            for original in oo:
                result=measure_observation(source,target,original,proposal)
                if result is None:continue
                points,quality=result;key=json.dumps(original['target'],sort_keys=True)
                if key not in best or len(points)>len(best[key][0]):best[key]=(points,quality,original,seed)
        for k,(points,quality,original,seed) in enumerate(best.values()):
            observed_target=copy.deepcopy(original['target'])
            if frame['pool_ordinal']==1316 and observed_target.get('kind')=='board_side' and observed_target.get('side')=='upper_long_side':
                observed_target['continue_corner']='upper_right'
            o=dict(id=f"{frame['id']}_edge_{k}",frame_id=frame['id'],type='curve',target=observed_target,
                feature_id=original.get('feature_id'),physical_group_id=original.get('physical_group_id'),use='fit',pixels_uv=np.round(points,4).tolist(),
                sigma_px=5.,measurement_origin='actual_image_remeasurement',method='static image correspondence followed by signed local edge remeasurement',
                source_seed_frame_id=seed['id'],source_seed_observation_id=original['id'],measurement_diagnostics=quality,
                synthetic_or_model_projected=False,pure_transfer_points_exported=False,correlation_group_id=f"{frame['shot_id']}_{key}")
            observations.append(o);perframe[frame['id']].append(o)
        if (n+1)%15==0:print('curves',n+1,'of69',flush=True)
    # Seed per-frame R/f from fit-only observed points, not check advertisements.
    for frame,st in zip(frames,status):
        local=perframe[frame['id']];pose=initial_pose(frame,local,landmarks)
        if pose:frame.update(pose)
        frame['fit_status']='measured' if local else 'no_reliable_measurement'
        st.update(status=frame['fit_status'],observations=len(local),pose_initializer_available=bool(pose),targets=[o['target'] for o in local])
    used={o['target']['landmark_id'] for o in observations if o['target']['kind']=='board_landmark'}
    landmarks=[p for p in landmarks if p['id'] in used]
    # Every input including failure receives a visible overlay.
    panels=[]
    for f in frames:
        im=images[f['id']].copy();local=perframe[f['id']]
        for j,o in enumerate(local):
            color=[(0,255,255),(255,150,80),(80,255,70),(255,80,220)][j%4]
            for uv in o['pixels_uv']:cv2.circle(im,tuple(np.rint(uv).astype(int)),4,color,1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(1920,52),(15,15,15),-1)
        cv2.putText(im,f"#{f['pool_ordinal']} {f['shot_id']} {f['source_time']} measured groups {len(local)}",(12,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
        path=OUT/'overlays'/f"{f['pool_ordinal']:05d}.jpg";cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,92]);panels.append(cv2.resize(im,(480,270)))
    for start in range(0,len(panels),12):
        pp=panels[start:start+12]
        while len(pp)%3:pp.append(np.zeros((270,480,3),np.uint8))
        cv2.imwrite(str(OUT/f'contact_{start//12+1:02d}.jpg'),np.vstack([np.hstack(pp[k:k+3]) for k in range(0,len(pp),3)]),[cv2.IMWRITE_JPEG_QUALITY,92])
    summary=dict(selected_frames=len(frames),attempted_frames=len(frames),measured_frames=sum(bool(perframe[f['id']]) for f in frames),
        observation_groups=len(observations),observed_pixels=sum(len(o['pixels_uv']) for o in observations),
        by_kind=dict(Counter(o['target']['kind'] for o in observations)),by_instance=dict(Counter(o.get('advertisement_instance_id','ice_or_board') for o in observations)),
        frames_with_pose_initializer=sum('initial_rvec_world_to_camera' in f for f in frames),model_fit_run=False,
        previously_heldout_promoted_shots=['S087'],retained_check_advertisement_instances=['advert_ppt_object_108'])
    result=dict(schema_version='rinkspace.measured-secondary279.v1',frames=frames,observations=observations,board_landmarks=landmarks,
        installations=[dict(id='women_sp_secondary',physical_camera_id='camera_secondary_long_side',camera_id='camera_secondary_long_side',initial_center_xyz=[53.,-80.,8.],fit_center=True,center_bounds_xyz=[[-200,-500,.25],[300,-50.001,200]],center_fixed=False)],
        summary=summary,frame_status=status,synthetic_or_model_projected_observations=False,
        limitations=['Initial camera centre and advertisement coordinates are conditional guesses, not measured positions.',
                    'Ambiguous repeated short-end instances and unsupported pixel matches omitted, never synthesized.',
                    'S087 is now development input if measured; it cannot serve as independent validation.'])
    save(OUT/'observations.json',result);save(OUT/'measurement_audit.json',logs);save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initializers-only',action='store_true');ap.add_argument('--augment-direct',action='store_true');ap.add_argument('--extra-corner-instances',action='store_true');ap.add_argument('--fill-visible-remaining',action='store_true');args=ap.parse_args()
    if args.initializers_only:add_initializers()
    elif args.augment_direct:augment_direct()
    elif args.extra_corner_instances:extra_corner_instances()
    elif args.fill_visible_remaining:fill_visible_remaining()
    else:run();add_initializers()
