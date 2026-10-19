"""Static scene search above an actually observed rink kickplate.

This is an image-domain region proposal, not a rink pose or plane transform.
Its sloping boundary retains the advertising band in low-angle broadcast views.
"""
import cv2
import numpy as np


def above_observed_kickplate(image,person_mask=None):
    height,width=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    yellow=(((hsv[:,:,0]>=15)&(hsv[:,:,0]<=43))&(hsv[:,:,1]>40)&(hsv[:,:,2]>85)).astype(np.uint8)*255
    yellow[:int(height*.22)]=0;yellow[int(height*.86):]=0
    yellow=cv2.morphologyEx(yellow,cv2.MORPH_CLOSE,np.ones((3,13),np.uint8))
    lines=cv2.HoughLinesP(yellow,1,np.pi/720,threshold=80,minLineLength=width*.28,maxLineGap=width*.08)
    candidates=[]
    for x0,y0,x1,y1 in ([] if lines is None else lines[:,0]):
        if abs(x1-x0)<width*.28:continue
        slope=(y1-y0)/(x1-x0)
        if abs(slope)>.45:continue
        candidates.append((abs(x1-x0),float(slope),float(y0-slope*x0),[int(x0),int(y0),int(x1),int(y1)]))
    if not candidates:return np.zeros((height,width),np.uint8),{'status':'missing_observed_kickplate','candidate_count':0}
    longest=max(candidates,key=lambda c:c[0]);slope,intercept=longest[1:3]
    boundary=slope*np.arange(width)+intercept-8
    yy=np.arange(height)[:,None]
    valid=((yy<boundary[None,:])&(yy>=max(18,height//25))).astype(np.uint8)*255
    if person_mask is not None:
        if person_mask.shape!=(height,width):raise ValueError('Person mask differs from static sample image')
        valid[cv2.dilate(person_mask,np.ones((15,15),np.uint8))>0]=0
    valid[:,:5]=0;valid[:,-5:]=0
    return valid,{'status':'observed_kickplate_proposal','candidate_count':len(candidates),
        'line_segment_sample_px':longest[3],'slope':slope,'intercept':intercept,
        'margin_sample_px':8,'source_only_no_model_projection':True,
        'person_mask_used':person_mask is not None,'mask_fraction':float(np.mean(valid>0))}


def scene_pyramid(image,mask,*,reference=False):
    """Multi-resolution image descriptors; source coordinates never change.

    Reference-only horizontal box blur approximates exposure during broadcast
    pans. It changes a descriptor's support, not the recorded camera or image
    provenance. Returned keypoints are mapped back to the actual sample grid.
    """
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    result=[]
    for width in [960,640,480]:
        scale=width/image.shape[1];height=round(image.shape[0]*scale)
        small=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
        valid=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
        for blur in ([0,5,9] if reference else [0]):
            view=cv2.blur(small,(blur,1)) if blur else small
            keys,desc=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.006,edgeThreshold=14).detectAndCompute(view,valid)
            xy=np.asarray([k.pt for k in keys],np.float32).reshape(-1,2)/scale
            if desc is not None:desc=np.sqrt(desc/(desc.sum(axis=1,keepdims=True)+1e-12))
            result.append({'width':width,'blur':blur,'xy':xy,'desc':desc})
    return result


def match_scene_pyramids(reference,current,*,min_matches=14):
    """Merge distinct real-image points across scales before spatial gating."""
    pairs=[];stages=[];matcher=cv2.BFMatcher(cv2.NORM_L2)
    for ref in reference:
        for cur in current:
            if ref['width']!=cur['width'] or ref['desc'] is None or cur['desc'] is None:continue
            if min(len(ref['desc']),len(cur['desc']))<2:continue
            backward={p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(cur['desc'],ref['desc'],k=2)
                if len(p)==2 and p[0].distance<.76*p[1].distance}
            selected=[p[0] for p in matcher.knnMatch(ref['desc'],cur['desc'],k=2)
                if len(p)==2 and p[0].distance<.76*p[1].distance and backward.get(p[0].trainIdx)==p[0].queryIdx]
            stages.append({'width':ref['width'],'reference_horizontal_blur_width':ref['blur'],'reciprocal_count':len(selected)})
            pairs.extend((float(p.distance),ref['xy'][p.queryIdx],cur['xy'][p.trainIdx]) for p in selected)
    left=[];right=[]
    for _,a,b in sorted(pairs,key=lambda p:p[0]):
        if left and (np.min(np.linalg.norm(np.asarray(left)-a,axis=1))<3 or np.min(np.linalg.norm(np.asarray(right)-b,axis=1))<3):continue
        left.append(a);right.append(b)
    diagnostic={'pyramid_stages':stages,'all_reciprocal_count':len(pairs),'unique_source_pairs':len(left),
        'thresholds':{'min_matches':min_matches,'ratio':.76,'homography_px':3.,'span_x':90.,'span_y':22.},
        'reference_preprocessing_is_derived_image_support_not_new_observations':True}
    if len(left)<min_matches:return None,None,{**diagnostic,'reason':'not_enough_distinct_source_points'}
    a=np.asarray(left);b=np.asarray(right)
    _,inliers=cv2.findHomography(a,b,cv2.USAC_MAGSAC,3.,maxIters=3000,confidence=.999)
    if inliers is None:return None,None,{**diagnostic,'reason':'no_static_image_consensus'}
    keep=inliers.ravel()>0;a=a[keep];b=b[keep];diagnostic['consensus_count']=len(a)
    if len(a)<min_matches:return None,None,{**diagnostic,'reason':'static_consensus_below_threshold'}
    diagnostic['span_sample_px']=np.ptp(b,axis=0).tolist()
    if np.ptp(b[:,0])<90 or np.ptp(b[:,1])<22:return None,None,{**diagnostic,'reason':'static_spatial_support_insufficient'}
    return a,b,{**diagnostic,'reason':'static_source_proposal_requires_separate_camera_and_ice_checks'}
