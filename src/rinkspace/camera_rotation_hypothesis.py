"""Isolated camera-image hypothesis; NEVER an ice-plane registration backend.

An uncalibrated background homography does not prove a fixed optical centre.
It merely makes a falsifiable prediction for ice pixels that are wholly absent
from training. Background points have unverified depth/plane membership.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import cv2
import numpy as np

from .continuous_registration import ContinuousConfig, ContinuousRegistrar
from .local_registration import RegistrationConfig, _regions_mask
from .geometry import project


ALGORITHM_VERSION = "background-only-camera-hypothesis-3-explicit-building-polygons"


def source_background_mask(image, correction, people=(), ice_buffer_px=96):
    """Source-only conservative boards mask; crowds/upper officials excluded.

    Locate observed blue top-rail segments within the source-reviewed non-ice
    region. Only the strip below an observed rail and above the ice buffer can
    train. Missing rail support stays missing; no mask is inferred from H.
    """
    height,width=image.shape[:2]
    ice=_regions_mask((height,width),[{"polygon":p} for p in correction.get("include_polygons",[])])
    if not np.any(ice):
        raise ValueError("A complete source-only ice polygon is required")
    ice_guard=cv2.dilate(ice,np.ones((2*ice_buffer_px+1,2*ice_buffer_px+1),np.uint8))
    excluded=np.zeros_like(ice)
    # The first polygon is the ice-fit's non-ice complement, not a person.
    # Identify that complement by its full-width top edge, never drop other
    # polygons merely because they are large.
    for p in correction.get("exclude_polygons",[]):
        q=np.asarray(p)
        complement=np.any(np.all(q==[0,0],axis=1)) and np.any(np.all(q==[width-1,0],axis=1))
        if not complement:
            excluded |= _regions_mask((height,width),[{"polygon":p}])
    for box in people:
        xyxy=box.get("expanded_xyxy",box.get("xyxy")) if isinstance(box,dict) else box
        x0,y0,x1,y1=np.rint(xyxy).astype(int)
        excluded[max(0,y0):min(height,y1+1),max(0,x0):min(width,x1+1)]=255
    # Score/clock rectangles are already source annotations, but zero them and
    # the entire ice before any rail detection, too.
    isolated=image.copy(); isolated[(ice_guard>0)|(excluded>0)]=0
    hsv=cv2.cvtColor(isolated,cv2.COLOR_BGR2HSV)
    blue=((hsv[:,:,0]>=90)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>30)&(hsv[:,:,2]>40)).astype(np.uint8)*255
    # Observe only long near-horizontal rail segments, not isolated letters.
    lines=cv2.HoughLinesP(blue,1,np.pi/360,threshold=25,minLineLength=80,maxLineGap=20)
    top=np.full(width,np.nan)
    selected=[]
    boundary=np.argmax(ice>0,axis=0)
    if lines is not None:
        for raw in lines[:,0]:
            x0,y0,x1,y1=map(int,raw)
            if x0==x1 or abs(y1-y0)>abs(x1-x0)*.4:continue
            if x1<x0:x0,y0,x1,y1=x1,y1,x0,y0
            xx=np.arange(x0,x1+1);yy=y0+(y1-y0)*(xx-x0)/(x1-x0)
            # Exclude unrelated high railings in the audience area. A rail too
            # near ice leaves no usable strip rather than entering the ice.
            good=(yy<boundary[xx]-28)&(yy>boundary[xx]-150)
            xx,yy=xx[good],yy[good]
            if len(xx)<80:continue
            selected.append([x0,y0,x1,y1])
            # Lowest observed rail is conservative when several blue structures
            # coexist; no undocumented extrapolation fills unobserved columns.
            top[xx]=np.fmax(top[xx],yy)
    yy=np.arange(height)[:,None]
    allowed=np.isfinite(top)[None,:]&(yy>=top[None,:]+4)&(yy<boundary[None,:]-ice_buffer_px)
    allowed &= (ice_guard==0)&(excluded==0)
    mask=allowed.astype(np.uint8)*255
    return dict(training=mask,heldout_ice=ice,ice_guard=ice_guard,people_and_graphics=excluded,
                observed_rail_segments=selected,plane="unverified_off_plane_background",
                caveat="Conservative background candidate beyond 96px buffer of inset ice polygon; may remove nearly all usable boards. Spectators excluded wholesale. Surface membership is not a calibration.")


class _BackgroundEdges(ContinuousRegistrar):
    def _safe(self,gray,usable):
        # 9px LK patches at two scales, plus filtering/interpolation allowance.
        isolated=cv2.GaussianBlur(np.where(usable>0,gray,0).astype(np.uint8),(5,5),1.)
        safe=cv2.erode(usable,np.ones((41,41),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        return isolated,safe


class CameraRotationHypothesis:
    def __init__(self):
        self.config=ContinuousConfig(keyframe_interval_s=.4,max_keyframes=40,patch_radius_px=4,
            pyramid_levels=1,feature_count=2200)
        # A narrow background strip can provide a scene-H candidate but cannot
        # validate its large extrapolation onto ice; that is the experiment.
        self.fit_config=RegistrationConfig(min_support_fraction=.01,min_support_height_fraction=.04,
            direct_anchor_every=1000000,dense_anchor_every=1000000)
        self.engine=_BackgroundEdges(self.config,self.fit_config)

    def push(self,image,sample,training_mask,force_break=False):
        isolated=np.where(training_mask[:,:,None]>0,image,0).astype(np.uint8)
        masks=dict(usable=training_mask,mask_status="source_only_static_background_candidate_not_ice",
                   ice_fraction=0.,usable_fraction=float(np.mean(training_mask>0)))
        row=self.engine.push(isolated,sample,masks,force_break)
        row["H_scene_reference_from_current"]=row.pop("H_reference_from_current")
        row["fit_scene_reference_xy"]=row.pop("fit_reference_xy")
        row.update(geometry_level="scene_projective_reference",unit="reference_image_px",
            matrix_direction="scene_reference_from_current",route="camera_image_hypothesis_only",
            training_plane="unverified_off_plane_background",ice_plane_mapping_established=False,
            independent_validation=False,formal_acceptance=False,
            interpretation="Background homography predicts withheld ice only under an unverified shared-camera-motion hypothesis",
            fitting_patch_policy=dict(raw_outside_mask_constant=0,preprocessing_kernel=[5,5],
                center_erosion_px=20,lk_patch_width=9,pyramid_levels=1,
                all_declared_ice_pixels_excluded=True,declared_excluded_raw_values_unavailable_to_estimator=True,
                actual_surface_membership_depends_on_source_annotations=True,
                iterative_LK_path_not_exported=True,protected_raw_values_unavailable_to_all_iterations=True))
        if row.get("graph_component_id") is not None:row["graph_component_id"]=f"camera_scene_{self.engine.component:03d}"
        row["reason"]=row.get("reason","").replace("ice_plane","background_scene")
        return row

    def configuration(self):
        return {"observed_keyframe":asdict(self.config),"background_fit":asdict(self.fit_config),
                "safe_erosion_px":20,"automatic_rail_source_ice_polygon_buffer_px":96,"source_polygon_warning":"Input ice-fit polygon is inset into ice; its complement is not an observed boards mask. Added 96px exclusion is conservative and may eliminate all support.","rail_detection":"observed_blue_segments_no_gap_extrapolation",
                "training_support_warning":"Narrow unverified boards support; predictions on the wholly withheld ice are extrapolation."}


class _RotationZoomEdges(_BackgroundEdges):
    """Constrained camera edge. Width/height gate is replaced by Jacobian rank.

    Focal length is an explicit assumed reference focal, not calibration. Each
    edge solves three rotation components and current/key focal ratio. It uses
    only observed background correspondences; zero ice residual enters fitting.
    """
    def __init__(self, config, fit_config, assumed_focal_px, flow_initialization="last_observation", sift_reobservation=False):
        super().__init__(config,fit_config)
        self.assumed_focal_px=float(assumed_focal_px)
        self.focal_by_sample={}
        self.flow_initialization=flow_initialization
        self.observed_pose_history=[]
        self.initial_current_h=None
        self.sift_reobservation=sift_reobservation
        self.sift=cv2.SIFT_create(nfeatures=2200,contrastThreshold=.005,edgeThreshold=12) if sift_reobservation else None
        self.current_sift=None

    def _key(self,gray,usable,sample,h):
        key=super()._key(gray,usable,sample,h)
        if key is not None:
            # Large pan can approach the first reference chart horizon. Keep
            # projective targets in float64 until mapped back to observed pixels.
            key['target']=project(h,key['points'])
            key['focal_px']=self.focal_by_sample.get(sample['sample_index'],self.assumed_focal_px)
            if self.sift_reobservation:key['sift']=self._sift(gray,usable)
        return key

    def _edge(self,key,gray,isolated,safe):
        edge=self._camera_edge(key,gray,isolated,safe)
        if edge['status']!='candidate' and self.sift_reobservation:
            edge=self._sift_edge(key,gray.shape,edge)
        self.last_camera_edge_audit.append({'observed_keyframe_sample_index':key['sample_index'],**edge})
        return edge


    def _sift(self,gray,usable):
        isolated=np.where(usable>0,gray,0).astype(np.uint8)
        points,descriptors=self.sift.detectAndCompute(isolated,usable)
        if descriptors is None:return np.empty((0,2),np.float32),np.empty((0,128),np.float32),[]
        distance=cv2.distanceTransform(usable,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        keep=[]
        for i,k in enumerate(points):
            x,y=np.rint(k.pt).astype(int)
            if 0<=x<gray.shape[1] and 0<=y<gray.shape[0] and distance[y,x]>=8*k.size+4:keep.append(i)
        return np.asarray([points[i].pt for i in keep],np.float32).reshape(-1,2),descriptors[keep], [float(points[i].size) for i in keep]

    def _sift_edge(self,key,shape,lk_failure):
        x,da,ka=key['sift'];y,db,kb=self.current_sift
        if min(len(x),len(y))<self.fit_config.min_tracks:
            return {**lk_failure,'sift_attempt_reason':'too_few_mask_safe_descriptors','sift_key_count':len(x),'sift_current_count':len(y)}
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        forward=matcher.knnMatch(da,db,k=2);reverse=matcher.knnMatch(db,da,k=2)
        back={a.queryIdx:a.trainIdx for a,b in reverse if a.distance<.75*b.distance}
        matches=[a for a,b in forward if a.distance<.75*b.distance and back.get(a.trainIdx)==a.queryIdx]
        xx=np.asarray([x[m.queryIdx] for m in matches],np.float32).reshape(-1,2)
        yy=np.asarray([y[m.trainIdx] for m in matches],np.float32).reshape(-1,2)
        target=project(key['H_reference_from_keyframe'],xx)
        edge=self._fit_camera_pairs(key,xx,yy,target,shape)
        edge.update(pair_algorithm='mask_isolated_SIFT_mutual_ratio_then_camera_rotation_zoom_4DOF_robust_fit',
            fallback_from_LK_reason=lk_failure.get('reason'),sift_key_descriptor_sizes=[ka[m.queryIdx] for m in matches],
            sift_current_descriptor_sizes=[kb[m.trainIdx] for m in matches],sift_descriptor_boundary_margin_is_heuristic=True,
            all_descriptor_inputs_outside_training_mask_are_constant_zero=True)
        return edge

    def _camera_edge(self,key,gray,isolated,safe):
        from scipy.optimize import least_squares
        target=key['target']
        guess=project(np.linalg.inv(self.initial_current_h if self.initial_current_h is not None else self.last_h),target).astype(np.float32)
        use=self._inside(guess,safe)
        x,target,guess=key['points'][use],target[use],guess[use]
        if len(x)<self.fit_config.min_tracks:
            return dict(status='unavailable',reason='no_visible_background_keyframe_support')
        radius=self.config.patch_radius_px
        params=dict(winSize=(radius*2+1,radius*2+1),maxLevel=self.config.pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005),minEigThreshold=1e-6)
        y,sf,_=cv2.calcOpticalFlowPyrLK(key['gray'],isolated,x[:,None],guess[:,None].copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**params)
        z,sb,_=cv2.calcOpticalFlowPyrLK(isolated,key['gray'],y,x[:,None].copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**params)
        y,z=y[:,0],z[:,0]
        ok=(sf[:,0]>0)&(sb[:,0]>0)&self._inside(y,safe)&(np.linalg.norm(z-x,axis=1)<self.config.forward_backward_px)
        correlations=np.zeros(len(x))
        for i in np.flatnonzero(ok):
            a=cv2.getRectSubPix(key['raw'],(15,15),tuple(map(float,x[i]))).astype(float)
            b=cv2.getRectSubPix(gray,(15,15),tuple(map(float,y[i]))).astype(float)
            if min(a.std(),b.std())<self.config.min_patch_std:ok[i]=False;continue
            a-=a.mean();b-=b.mean();correlations[i]=np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8)
        ok &= correlations>=self.config.min_patch_correlation
        x,y,target=x[ok],y[ok],target[ok]
        return self._fit_camera_pairs(key,x,y,target,gray.shape)

    def _fit_camera_pairs(self,key,x,y,target,shape):
        from scipy.optimize import least_squares
        base=dict(status='unavailable',correspondence_count_before_fit=len(x),
            edge_keyframe_sample_index=key['sample_index'],edge_keyframe_time_s=key['time'])
        if len(x)<self.fit_config.min_tracks:return dict(base,reason='too_few_background_correspondences')
        height,width=shape;fk=key['focal_px'];cx,cy=width/2,height/2
        k=np.array([[fk,0,cx],[0,fk,cy],[0,0,1.]])
        def matrix(p):
            fc=fk*np.exp(p[3]);kc=np.array([[fc,0,cx],[0,fc,cy],[0,0,1.]])
            h=k@cv2.Rodrigues(p[:3])[0]@np.linalg.inv(kc)
            return h/h[2,2]
        def residual(p,points,observed):return (project(matrix(p),points)-observed).ravel()
        # Start from fresh observed flow, with no calibration from ice marks.
        init=np.zeros(4);delta=np.median(x-y,axis=0);init[:2]=[-delta[1]/fk,delta[0]/fk]
        result=least_squares(residual,init,args=(y,x),loss='soft_l1',f_scale=1.,max_nfev=80,
            bounds=([-1.2,-1.2,-1.2,-1.1],[1.2,1.2,1.2,1.1]))
        error=np.linalg.norm(residual(result.x,y,x).reshape(-1,2),axis=1)
        flags=error<self.fit_config.ransac_threshold_px
        if flags.sum()<self.fit_config.min_inliers or flags.mean()<.45:
            return dict(base,reason='camera_model_disagrees_with_background',train_inliers=int(flags.sum()),
                background_residual_p95_px=float(np.percentile(error,95)))
        result=least_squares(residual,result.x,args=(y[flags],x[flags]),loss='soft_l1',f_scale=1.,max_nfev=60,
            bounds=([-1.2,-1.2,-1.2,-1.1],[1.2,1.2,1.2,1.1]))
        edge=matrix(result.x);error=np.linalg.norm(project(edge,y[flags])-x[flags],axis=1)
        singular=np.linalg.svd(result.jac,compute_uv=False);condition=float(singular[0]/max(singular[-1],1e-15))
        if singular[-1]<1e-5 or condition>1e6:return dict(base,reason='camera_parameter_rank_unobservable',jacobian_singular_values=singular.tolist())
        if np.percentile(error,95)>self.fit_config.max_inlier_p95_px:return dict(base,reason='camera_background_residual_too_large')
        combined=key['H_reference_from_keyframe']@edge;combined/=combined[2,2]
        # A real rotation may map an image across the reference horizon. A
        # finite single canvas cannot represent that: leave this frame missing.
        corners=np.array([[0,0],[width,0],[width,height],[0,height]],float)
        denominator=np.c_[corners,np.ones(4)]@combined[2]
        chart_crosses_horizon=bool(denominator.min()*denominator.max()<=0)
        hull=cv2.convexHull(y[flags].astype(np.float32)).reshape(-1,2)
        covariance=np.linalg.pinv(result.jac.T@result.jac)*max(np.mean(error**2),.01)
        return dict(base,status='candidate',reason='camera_only_hypothesis_ice_validation_pending',
            H_reference_from_current=combined.tolist(),H_keyframe_from_current=edge.tolist(),
            fit_current_xy=y[flags].tolist(),fit_keyframe_xy=x[flags].tolist(),fit_reference_xy=target[flags].tolist(),
            train_inliers=int(flags.sum()),support_polygon=hull.tolist(),support_fraction=float(cv2.contourArea(hull)/(height*width)),
            inlier_median_reference_px=float(np.median(error)),inlier_p95_reference_px=float(np.percentile(error,95)),
            fitting_residual_coordinate_space='observed_background_keyframe_pixels',
            pair_algorithm='isolated_LK_FB_NCC_then_camera_rotation_zoom_4DOF_robust_fit',
            assumed_reference_focal_px=self.assumed_focal_px,keyframe_focal_px=fk,current_focal_px=float(fk*np.exp(result.x[3])),
            first_reference_chart_horizon_crosses_image=chart_crosses_horizon,
            chart_warning='Camera orientations can be valid while a chosen perspective chart crosses a horizon; whole-shot display gauge must be assessed separately.',
            camera_edge_parameters=result.x.tolist(),camera_parameter_covariance=covariance.tolist(),
            camera_parameter_names=['rotation_x_rad','rotation_y_rad','rotation_z_rad','log_current_over_key_focal'],
            covariance_is_local_linearized_fit_diagnostic=True,
            optimizer_status=int(result.status),optimizer_function_evaluations=int(result.nfev),
            jacobian_singular_values=singular.tolist(),jacobian_condition=condition,
            parameter_observability_warning='Local Jacobian rank does not establish global focal calibration; extrapolation may remain uncertain',
            prefit_current_xy=y.tolist(),prefit_keyframe_xy=x.tolist(),fit_observation_flags=flags.tolist())

    def push(self,image,sample,masks,force_break=False):
        # ContinuousRegistrar builds its new key inside push, so focal must be
        # assigned to that key immediately after the successful edge is known.
        self.last_camera_edge_audit=[]
        time=float(int(sample['pts'])*Fraction(sample['time_base']))
        if abs(time-sample['source_time_s'])>1e-5:raise ValueError('Source PTS mismatch')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.current_sift=self._sift(gray,masks['usable']) if self.sift_reobservation else None
        gap=None if self.last_time is None else time-self.last_time
        broken=force_break or (gap is not None and (gap<=0 or gap>self.config.max_recovery_gap_s)) or (self.shape is not None and gray.shape!=self.shape)
        if broken:
            self.keys=[];self.previous_key=None;self.last_h=None;self.last_good_time=None;self.observed_pose_history=[]
        self.initial_current_h=self.last_h
        initialization='last_observation'
        if self.flow_initialization=='constant_velocity' and len(self.observed_pose_history)==2:
            (ta,ha),(tb,hb)=self.observed_pose_history
            if abs((time-tb)-(tb-ta))<1e-5 and 0<time-tb<=.06:
                predicted=hb@np.linalg.inv(ha)@hb
                if np.isfinite(predicted).all() and abs(predicted[2,2])>1e-9:
                    self.initial_current_h=predicted/predicted[2,2];initialization='constant_velocity_from_two_observed_poses_only_for_LK_initialization'
        if not self.keys:
            key=self._key(gray,masks['usable'],sample,np.eye(3))
            row={**sample,'status':'unavailable','H_reference_from_current':None,'fit_current_xy':[],
                'fit_reference_xy':[],'support_polygon':[],'graph_component_id':None,
                'mask_status':masks['mask_status'],'usable_fraction':masks['usable_fraction'],'ice_fraction':0.,
                'reason':'insufficient_background_camera_reference_support','independent_validation':False,
                'map_usable_for_metric_output':False,'dense_optimization_used':False}
            if key is not None:
                h,w=gray.shape;xy=key['points']-np.array([w/2,h/2]);xx,yy=xy.T;f=self.assumed_focal_px
                jac=np.stack([np.stack([-xx*yy/f,f+xx*xx/f,-yy,-xx],axis=1),np.stack([-f-yy*yy/f,xx*yy/f,xx,-yy],axis=1)],axis=1).reshape(-1,4)
                singular=np.linalg.svd(jac,compute_uv=False)
                if singular[-1]>1e-5 and singular[0]/singular[-1]<1e6:
                    self.component+=1;self.reference_sample_index=sample['sample_index'];self.keys=[key]
                    self.previous_key=key;self.last_h=np.eye(3);self.last_good_time=time
                    hull=cv2.convexHull(key['points']).reshape(-1,2)
                    row.update(status='candidate',H_reference_from_current=np.eye(3).tolist(),reference_frame_only=True,
                        reason='first_observed_reference_only',fit_current_xy=key['points'].tolist(),fit_reference_xy=key['points'].tolist(),
                        support_polygon=hull.tolist(),train_inliers=len(key['points']),support_fraction=float(cv2.contourArea(hull)/(h*w)),
                        graph_component_id=f'camera_{self.component:03d}',reference_sample_index=self.reference_sample_index,
                        jacobian_singular_values=singular.tolist(),jacobian_condition=float(singular[0]/singular[-1]),
                        reference_observability_model='rotation_zoom_4DOF; no 8DOF homography support gate')
                else:row['reason']='camera_reference_parameter_rank_unobservable'
            if broken:row['break_reason']='declared_shot_boundary' if force_break else 'pts_gap_or_resolution_change'
            self.last_time=time;self.shape=gray.shape
        else:row=super().push(image,sample,masks,False)
        row['flow_initialization']=initialization
        row['H_flow_initialization_only']=self.initial_current_h.tolist() if self.initial_current_h is not None else None
        if row['status']=='candidate':
            self.observed_pose_history.append((time,np.asarray(row['H_reference_from_current']).copy()));self.observed_pose_history=self.observed_pose_history[-2:]
            focal=row.get('current_focal_px',self.assumed_focal_px)
            self.focal_by_sample[sample['sample_index']]=focal
            if self.previous_key is not None and self.previous_key['sample_index']==sample['sample_index']:
                self.previous_key['focal_px']=focal
            for key in self.keys:
                if key['sample_index']==sample['sample_index']:key['focal_px']=focal
        return row


class CameraRotationZoomHypothesis(CameraRotationHypothesis):
    def __init__(self,assumed_focal_px=960.,flow_initialization="last_observation",sift_reobservation=False):
        if not np.isfinite(assumed_focal_px) or assumed_focal_px<=0:raise ValueError('Assumed focal length must be finite and positive')
        super().__init__()
        self.engine=_RotationZoomEdges(self.config,self.fit_config,assumed_focal_px,flow_initialization,sift_reobservation)
        self.assumed_focal_px=float(assumed_focal_px)

    def configuration(self):
        return {**super().configuration(),'motion_model':'K_key R K_current^-1, 3 rotation + 1 focal ratio',
            'sift_reobservation_when_LK_fails':self.engine.sift_reobservation,
            'sift_descriptor_ratio':.75,'sift_margin_policy':'distance_to_excluded >= 8*keypoint.size+4; heuristic boundary avoidance, raw outside mask is always zero before all SIFT processing',
            'flow_initialization':self.engine.flow_initialization,'prediction_never_used_as_measurement':True,
            'assumed_reference_focal_px':self.assumed_focal_px,'principal_point':'decoded image centre, assumed',
            'square_pixels':True,'focal_calibrated':False,'zero_translation_and_static_background':'unverified hypothesis',
            'support_policy':'min18 observed matches, min14 consistent; 4DOF Jacobian rank/condition<=1e6; no 8DOF height gate'}


def source_building_mask(image,correction,people,annotation):
    """Explicit source-reviewed static building areas; no geometry enters masks.

    Manual source polygons define the non-ice surface scope. The original ice
    fit mask was inset and cannot be complemented to obtain these annotations.
    Unreviewed/interpolated source annotations remain explicitly labelled.
    """
    base=source_background_mask(image,correction,people)
    training=_regions_mask(image.shape[:2],[{'polygon':p} for p in annotation.get('include_polygons',[])])
    extra=_regions_mask(image.shape[:2],[{'polygon':p} for p in annotation.get('exclude_polygons',[])])
    # Explicit building polygons, rather than the complement of an inset ice
    # polygon, establish the non-ice scope. Never reintroduce known ice/persons.
    excluded=base['people_and_graphics']|extra|base['heldout_ice']
    training[excluded>0]=0
    base.update(training=training,people_and_graphics=base['people_and_graphics']|extra,
        observed_rail_segments=[],annotation_status=annotation.get('review_status','unreviewed'),
        caveat='Explicit source-only building polygons; interpolated ranges remain candidate masks, not geometric calibration.')
    return base
