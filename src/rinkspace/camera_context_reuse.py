"""Offline fixed-centre context hypotheses with explicit, separate ice checks.

Pair tracking is image evidence, not permission to publish a global grid. Named
ice contours must separately qualify an anchor or propagated candidate.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from .fixed_camera import fit_rotation_zoom
from .multisurface_camera import world_rays,project_camera
from .landmark_camera import GroundCurve,model_equation,curve_residual
from .multisurface_camera import Camera
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from dataclasses import replace


def fit_context_increment(reference_camera, reference_xy, current_xy, *, max_nfev=30):
    """Small adjacent-frame R/f update from measured pixels, never pose interpolation.

    One fifth of actual image correspondences are excluded from optimization.
    Those are static-image checks, not an independent ground-geometry check.
    """
    left=np.asarray(reference_xy,float);right=np.asarray(current_xy,float)
    if len(left)<12:return None,{'reason':'insufficient_actual_pixel_tracks','count':len(left)}
    rays,valid=world_rays(reference_camera,left)
    fit=np.arange(len(left))%5!=0;fit&=valid;check=(~fit)&valid
    def decode(x):
        return replace(reference_camera,rotation=Rotation.from_rotvec(x[:3]).as_matrix()@reference_camera.rotation,
                       focal=reference_camera.focal*np.exp(x[3]))
    def residual(x):
        prediction,front=project_camera(decode(x),rays[fit],rays=True)
        delta=prediction-right[fit];delta[~front]=1e4
        return delta.ravel()
    result=least_squares(residual,np.zeros(4),bounds=([-.12]*3+[-.12],[.12]*3+[.12]),
                         loss='soft_l1',f_scale=1.,max_nfev=max_nfev,x_scale='jac')
    camera=decode(result.x);prediction,front=project_camera(camera,rays,rays=True)
    errors=np.linalg.norm(prediction-right,axis=1);errors[~front]=1e6
    train=float(np.percentile(errors[fit],95));held=float(np.percentile(errors[check],95)) if check.any() else None
    accepted=train<=3. and held is not None and held<=5.
    return camera if accepted else None,{'reason':'measured_adjacent_image_link' if accepted else 'static_image_link_rejected',
        'count':len(left),'fit_count':int(fit.sum()),'heldout_count':int(check.sum()),'fit_p95_sample_px':train,
        'heldout_p95_sample_px':held,'image_link_accepted':accepted,'geometry_validation':False,
        'nfev':int(result.nfev),'rotation_and_zoom_from_actual_pixels':True,'fixed_optical_centre':True}


@dataclass(frozen=True)
class ContextConfig:
    min_points:int=12
    max_fb_px:float=.9
    max_static_p95_px:float=3.
    max_ice_p95_px:float=5.
    min_ice_points:int=4
    max_gap_s:float=.15
    keyframe_every_s:float=1/60


def temporal_correspondences(grays,valid_masks,start,end,*,min_points=12):
    """Measure every intervening frame before a wider graph edge is created.

    Direct 0.1 s LK can jump by one repeating seat/advertisement during a fast
    pan. This never bridges a frame with no surviving actual pixel track.
    """
    points=cv2.goodFeaturesToTrack(grays[start],1000,.003,5,mask=valid_masks[start],blockSize=5)
    if points is None:return None,None,{'reason':'no_initial_features'}
    points=points.reshape(-1,2);original=points.copy();previous=start;steps=[]
    direction=1 if end>start else -1
    for index in range(start+direction,end+direction,direction):
        b,ok,_=cv2.calcOpticalFlowPyrLK(grays[previous],grays[index],points[:,None],None,winSize=(25,25),maxLevel=3)
        if b is None:return None,None,{'reason':'missing_forward_flow','failed_frame':index}
        back,bk,_=cv2.calcOpticalFlowPyrLK(grays[index],grays[previous],b,None,winSize=(25,25),maxLevel=3)
        if back is None:return None,None,{'reason':'missing_reverse_flow','failed_frame':index}
        b=b[:,0];back=back[:,0];q=np.rint(b).astype(int)
        good=ok[:,0].astype(bool)&bk[:,0].astype(bool)&(np.linalg.norm(back-points,axis=1)<.8)
        good&=(q[:,0]>=0)&(q[:,0]<grays[index].shape[1])&(q[:,1]>=0)&(q[:,1]<grays[index].shape[0])
        selected=np.flatnonzero(good);good[selected]&=valid_masks[index][q[selected,1],q[selected,0]]>0
        points=b[good];original=original[good];previous=index;steps.append({'frame':index,'count':len(points)})
        if len(points)<min_points:return None,None,{'reason':'intervening_image_track_missing','steps':steps}
    _,mask=cv2.findHomography(original,points,cv2.USAC_MAGSAC,2.5,maxIters=1500,confidence=.999)
    if mask is None or mask.sum()<min_points:return None,None,{'reason':'no_static_consensus','steps':steps}
    selected=mask.ravel()>0
    return original[selected],points[selected],{'reason':'all_intervening_frames_measured','steps':steps,'count':int(selected.sum())}


def track_pair(reference_gray,current_gray,reference_camera,reference_valid,current_valid,reference_ice,config=None):
    config=config or ContextConfig()
    a=cv2.goodFeaturesToTrack(reference_gray,1000,.003,5,mask=reference_valid,blockSize=5)
    failure={'passed':False,'reason':'insufficient_features','stage':'context_image_pair_only'}
    if a is None or len(a)<config.min_points:return None,failure,None,None
    b,okay,_=cv2.calcOpticalFlowPyrLK(reference_gray,current_gray,a,None,winSize=(25,25),maxLevel=4)
    if b is None:return None,{**failure,'reason':'forward_flow_missing'},None,None
    back,back_okay,_=cv2.calcOpticalFlowPyrLK(current_gray,reference_gray,b,None,winSize=(25,25),maxLevel=4)
    if back is None:return None,{**failure,'reason':'reverse_flow_missing'},None,None
    a,b,back=a.reshape(-1,2),b.reshape(-1,2),back.reshape(-1,2)
    good=okay.ravel().astype(bool)&back_okay.ravel().astype(bool)&(np.linalg.norm(back-a,axis=1)<config.max_fb_px)
    xy=np.rint(b).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<current_valid.shape[1])&(xy[:,1]>=0)&(xy[:,1]<current_valid.shape[0])
    good&=inside
    selected=np.flatnonzero(good);good[selected]&=current_valid[xy[selected,1],xy[selected,0]]>0
    left,right=a[good],b[good]
    if len(left)<config.min_points:return None,{**failure,'reason':'forward_backward_flow_support_insufficient','count':len(left)},left,right
    _,inliers=cv2.findHomography(left,right,cv2.USAC_MAGSAC,2.5,maxIters=1500,confidence=.999)
    if inliers is None or inliers.sum()<config.min_points:return None,{**failure,'reason':'static_consensus_missing','count':len(left)},left,right
    left,right=left[inliers.ravel()>0],right[inliers.ravel()>0]
    xy=np.rint(left).astype(int);is_ice=reference_ice[xy[:,1],xy[:,0]]>0
    # Prefer independent ice as a complete surface holdout. With no visible
    # ice points the tracker can preserve a camera hypothesis, but cannot
    # upgrade it into a geometry-qualified result.
    if np.count_nonzero(~is_ice)>=config.min_points:
        fit=~is_ice;check=is_ice;kind='all_ice_excluded_from_pair_fit'
    else:
        order=np.lexsort((left[:,1],left[:,0]));check=np.zeros(len(left),bool);check[order[::5]]=True;fit=~check
        kind='interleaved_image_points_not_independent_ice'
    if fit.sum()<8:return None,{**failure,'reason':'not_enough_fit_points'},left,right
    current,diagnostic=fit_rotation_zoom(reference_camera,left,right,max_nfev=35,fit_mask=fit,
                                       initial_camera=reference_camera)
    if current is None:return None,{**failure,'reason':'rotation_zoom_failed','solver':diagnostic},left,right
    rays,valid=world_rays(reference_camera,left);prediction,front=project_camera(current,rays,rays=True)
    error=np.linalg.norm(prediction-right,axis=1);error[~(valid&front)]=1e6
    fit_p95=float(np.percentile(error[fit],95));check_p95=float(np.percentile(error[check],95)) if check.any() else None
    passed=fit_p95<=config.max_static_p95_px and (check.sum()<config.min_ice_points or check_p95<=config.max_ice_p95_px)
    ice_pass=kind=='all_ice_excluded_from_pair_fit' and check.sum()>=config.min_ice_points and check_p95<=config.max_ice_p95_px
    report={'passed':bool(passed),'reason':'image_pair_supported' if passed else 'image_pair_residual_failed',
        'stage':'context_image_pair_only','fit_count':int(fit.sum()),'check_count':int(check.sum()),
        'fit_p95_px':fit_p95,'check_p95_px':check_p95,'check_kind':kind,'unused_ice_pair_passed':bool(ice_pass),
        'fit_mask':fit.tolist(),'check_mask':check.tolist(),'errors_px':error.tolist(),
        'geometry_qualified':False,'geometry_independent_validation':False,'rank':diagnostic.get('data_jacobian_rank')}
    return current if passed else None,report,left,right


def named_ice_check(camera,observations,model,*,max_p95_px=7.,minimum_physical_groups=2):
    primitives={p['id']:p for p in model['structure']['primitives']};scores=[]
    for ob in observations:
        ident=ob.get('selected_model_id',ob.get('feature_id'))
        if ident not in primitives or ob.get('association_status') in {'conflict','masked_out','not_detected'}:continue
        points=ob.get('points_image')
        if points is None and ob.get('points_source_px') is not None:
            points=np.asarray(ob['points_source_px'])*np.asarray(camera.size)/np.asarray(ob.get('source_size',[1920,1080]))
        if points is None or len(points)<3:continue
        kind,equation=model_equation(primitives[ident]);curve=GroundCurve(0,ident,kind,equation,np.asarray(points),ob.get('sigma_image_px',2.5))
        errors=np.abs(curve_residual(camera,curve));scores.append({'physical_id':ident,'count':len(errors),
            'median_px':float(np.median(errors)),'p95_px':float(np.percentile(errors,95)),
            'observation_id':ob.get('id'),'observation_role':ob.get('role'),
            'used_to_fit_this_pose':ob.get('used_to_fit_this_pose',False)})
    unused=[s for s in scores if not s['used_to_fit_this_pose']]
    distinct={s['physical_id'] for s in unused if s['p95_px']<=max_p95_px}
    failed=[s for s in unused if s['p95_px']>max_p95_px]
    return {'scores':scores,'accepted':bool(len(distinct)>=minimum_physical_groups and not failed),
        'passing_physical_group_ids':sorted(distinct),'failed_physical_group_ids':[s['physical_id'] for s in failed],
        'minimum_physical_groups':minimum_physical_groups,'max_p95_px':max_p95_px,
        'scope':'Current-image named contours; prior geometry remains conditional, not independent rink validation.'}


def joint_fixed_center_geometry(model,initial_cameras,edges,fit_groups,identity,*,max_nfev=180,fit_radial=False,fit_shared_metric=False,fit_right_row=False):
    """Test a named F-row hypothesis with one C and per-view rotation/focal.

    Only F_row_x is changed. Shared radius, row spacing and corresponding
    transverse coordinates are hard-preserved by the existing parameterization.
    Stored old F observations are never implicitly consumed.
    """
    count=len(initial_cameras);base=initial_cameras[0]
    primitives={p['id']:p for p in model['structure']['primitives']}
    equations={k:model_equation(p) for k,p in primitives.items() if p['kind'] in {'circle','axis_line'}}
    old_f=float(model['parameter_estimates']['F_row_x']['value'])
    physical_multiplicity={(g['frame'],g['physical_id']):sum(h['frame']==g['frame'] and h['physical_id']==g['physical_id'] for h in fit_groups) for g in fit_groups}
    x0=np.r_[base.centre,old_f,*[np.r_[Rotation.from_matrix(c.rotation).as_rotvec(),np.log(c.focal)] for c in initial_cameras]]
    lower=np.r_[[-180.,-300.,.5],-190.,*[np.r_[Rotation.from_matrix(c.rotation).as_rotvec()-.8,np.log(150.)] for c in initial_cameras]]
    upper=np.r_[[180.,-50.01,100.],-20.,*[np.r_[Rotation.from_matrix(c.rotation).as_rotvec()+.8,np.log(8000.)] for c in initial_cameras]]
    radial_index=None;metric_indices=[];right_row_index=None
    if fit_radial:
        radial_index=len(x0);x0=np.r_[x0,0.];lower=np.r_[lower,-.5];upper=np.r_[upper,.5]
    old_radius=float(model['parameter_estimates']['target_radius']['value'])
    old_spacing=float(model['parameter_estimates']['ring_half_spacing']['value'])
    mid_y=float(model['parameter_estimates']['ring_row_mid_y']['value'])
    if fit_shared_metric:
        metric_indices=[len(x0),len(x0)+1]
        x0=np.r_[x0,old_radius,old_spacing];lower=np.r_[lower,1.,4.];upper=np.r_[upper,15.,16.5]
    old_right=float(model['parameter_estimates']['R_row_x']['value'])
    if fit_right_row:
        right_row_index=len(x0);x0=np.r_[x0,old_right];lower=np.r_[lower,20.];upper=np.r_[upper,180.]
    # Mild declared regularizers stabilize weak directions; they are not camera
    # surveys or measured rink dimensions. Fit/holdout scores exclude priors.
    def decode(x):
        return [Camera(Rotation.from_rotvec(x[4+4*i:7+4*i]).as_matrix(),x[:3],float(np.exp(x[7+4*i])),
                       initial_cameras[i].principal,initial_cameras[i].size,float(x[radial_index]) if fit_radial else 0.) for i in range(count)]
    row_count=sum(len(e['reference_xy'])*2 for e in edges)+sum(len(g['points_image']) for g in fit_groups)+4+int(fit_radial)+2*int(fit_shared_metric)+int(fit_right_row)
    sparsity=lil_matrix((row_count,len(x0)),dtype=int);cursor=0
    for edge in edges:
        n=2*len(edge['reference_xy']);columns=list(range(4+edge['a']*4,8+edge['a']*4))+list(range(4+edge['b']*4,8+edge['b']*4))
        if fit_radial:columns.append(radial_index)
        sparsity[cursor:cursor+n,columns]=1;cursor+=n
    for group in fit_groups:
        n=len(group['points_image']);columns=[0,1,2,*range(4+group['frame']*4,8+group['frame']*4)]
        if group['physical_id']=='$F':columns.append(3)
        if fit_radial:columns.append(radial_index)
        if fit_shared_metric:columns.extend(metric_indices)
        if fit_right_row and group['physical_id'] in {'R1','R2','R3','R4'}:columns.append(right_row_index)
        sparsity[cursor:cursor+n,columns]=1;cursor+=n
    for j in range(4):sparsity[cursor+j,j]=1
    if fit_radial:sparsity[cursor+4,radial_index]=1
    if fit_shared_metric:
        for j,col in enumerate(metric_indices):sparsity[cursor+4+int(fit_radial)+j,col]=1
    if fit_right_row:sparsity[-1,right_row_index]=1
    def group_curve(group,x):
        physical_id=identity if group['physical_id']=='$F' else group['physical_id']
        kind,equation=equations[physical_id];equation=equation.copy()
        if group['physical_id']=='$F' or ((fit_shared_metric or fit_right_row) and physical_id in {'R1','R2','R3','R4'}):
            p=np.asarray(primitives[physical_id]['points_model']);centre=(p.min(0)+p.max(0))/2
            radius=np.median(np.linalg.norm(p-centre,axis=1));a,b=x[3] if group['physical_id']=='$F' else centre[0],centre[1]
            if fit_shared_metric:
                radius=float(x[metric_indices[0]]);spacing=float(x[metric_indices[1]])
                coefficient={'F0':-3,'F1':-1,'F2':1,'F3':3,'R1':-3,'R2':-1,'R3':1,'R4':3}[physical_id]
                b=mid_y+coefficient*spacing
            if fit_right_row and physical_id in {'R1','R2','R3','R4'}:a=float(x[right_row_index])
            equation=np.array([[1.,0.,-a],[0.,1.,-b],[-a,-b,a*a+b*b-radius*radius]])
        return GroundCurve(group['frame'],physical_id,kind,equation,np.asarray(group['points_image']),group.get('sigma_px',2.5))
    def data_residuals(x,with_priors=True):
        cameras=decode(x);parts=[]
        for edge in edges:
            left=np.asarray(edge['reference_xy']);right=np.asarray(edge['current_xy'])
            rays,valid=world_rays(cameras[edge['a']],left);pred,front=project_camera(cameras[edge['b']],rays,rays=True)
            delta=pred-right;delta[~(valid&front)]=1e4
            parts.append(delta.ravel()*np.sqrt(16/max(len(left),1))/1.5)
        for group in fit_groups:
            curve=group_curve(group,x);residual=curve_residual(cameras[group['frame']],curve)
            multiplicity=physical_multiplicity[(group['frame'],group['physical_id'])]
            parts.append(residual*np.sqrt(12/max(len(residual)*multiplicity,1))/curve.sigma_px)
        if with_priors:
            parts.append(np.r_[(x[:3]-base.centre)/100.,(x[3]-old_f)/80.])
            if fit_radial:parts.append(np.asarray([x[radial_index]/.5]))
            if fit_shared_metric:parts.append(np.asarray([(x[metric_indices[0]]-old_radius)/30.,(x[metric_indices[1]]-old_spacing)/30.]))
            if fit_right_row:parts.append(np.asarray([(x[right_row_index]-old_right)/100.]))
        return np.concatenate(parts)
    result=least_squares(data_residuals,x0,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),
                         loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=max_nfev,
                         ftol=1e-7,xtol=1e-7,gtol=1e-7)
    cameras=decode(result.x);scores=[]
    for group in fit_groups:
        curve=group_curve(group,result.x);err=np.abs(curve_residual(cameras[group['frame']],curve))
        scores.append({'frame':group['frame'],'physical_id':curve.physical_id,'count':len(err),
                       'median_px':float(np.median(err)),'p95_px':float(np.percentile(err,95)),
                       'role':'fit_not_validation'})
    edge_scores=[]
    for edge in edges:
        rays,valid=world_rays(cameras[edge['a']],np.asarray(edge['reference_xy']));pred,front=project_camera(cameras[edge['b']],rays,rays=True)
        errors=np.linalg.norm(pred-np.asarray(edge['current_xy']),axis=1)
        edge_scores.append({'a':edge['a'],'b':edge['b'],'count':len(errors),
                            'median_px':float(np.median(errors)),'p95_px':float(np.percentile(errors,95))})
    diagnostic={'identity_hypothesis':identity,'success':bool(result.success),'message':result.message,
        'nfev':result.nfev,'robust_cost':float(result.cost),'F_row_x':float(result.x[3]),
        'shared_center_model':result.x[:3].tolist(),'fit_curve_scores':scores,'static_edge_scores':edge_scores,
        'parameters_changed':['shared_installation_C','per_view_R_f','F_row_x'],
        'hard_preserved':['all_eight_ring_radii','equal_row_spacing','paired_transverse_coordinates','central_circle_and_blue_line_relationships'],
        'old_misidentified_F_observations_used':False,'independent_validation':False,
        'declared_regularizers':{'centre_sigma_model_units':100.,'F_row_x_sigma_model_units':80.},
        'initial_F_row_x':old_f,'initial_center_model':base.centre.tolist()}
    diagnostic['shared_radial_ray_k1']=float(result.x[radial_index]) if fit_radial else 0.
    diagnostic['shared_radial_fit_enabled']=fit_radial
    if fit_radial:
        diagnostic['parameters_changed'].append('shared_radial_ray_k1')
        diagnostic['radial_curve_error_domain']='approximately sample px after inverse radial correction'
    if fit_shared_metric:
        diagnostic['parameters_changed']+=['target_radius_shared_all_eight','ring_half_spacing_shared_both_rows']
        diagnostic['hard_preserved']=['all_eight_equal_radii','equal_row_spacing','paired_transverse_coordinates','central_circle_and_blue_line_relationships']
        diagnostic['shared_metric_parameters']={'target_radius':float(result.x[metric_indices[0]]),'ring_half_spacing':float(result.x[metric_indices[1]])}
        diagnostic['declared_regularizers']['shared_radius_spacing_sigma_model_units']=30.
        diagnostic['R_row_x_fixed_as_gauge_not_measured']=float(model['parameter_estimates']['R_row_x']['value'])
    if fit_right_row:
        diagnostic.setdefault('shared_metric_parameters',{})['R_row_x']=float(result.x[right_row_index])
        diagnostic['parameters_changed'].append('R_row_x_relative_to_observed_centre_line')
        diagnostic.pop('R_row_x_fixed_as_gauge_not_measured',None)
        diagnostic['declared_regularizers']['R_row_x_sigma_model_units']=100.
        diagnostic['coordinate_scale']='conditional width100 drawing convention; no independently surveyed metre scale'
    return cameras,float(result.x[3]),diagnostic
