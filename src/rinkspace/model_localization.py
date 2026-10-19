"""Frozen shared-model localization with camera proposals and ice re-observation.

Background features propose a camera/view hypothesis only. A global output needs
fresh masked ice re-observation or bounded propagation from such an anchor.
Local tracking alone is deliberately unable to emit a global grid.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import time

import av
import cv2
import numpy as np

from .continuous_registration import ContinuousRegistrar, ContinuousConfig
from .curve_registration import refine_model_projection, CurveRefinementConfig
from .event_model import load_model, polygon_mask, ellipse_points
from .geometry import project
from .io import dump_json, load_json, sha256
from .local_registration import ice_mask, RegistrationConfig
from .foreground_rows import row_foreground_exclusion
from .rink_reference import RinkReference, validate_frame_map
from .workspace import source_path, load_index

ALGORITHM_VERSION='model-atlas-multihypothesis+masked-print-reobserve+bounded-local-v2'


def require_anchor(model, anchor_id, source_hash=None):
    matches=[v for v in model.get('reference_views',[]) if v.get('id')==anchor_id]
    if len(matches)!=1: raise ValueError('Anchor missing from this exact model revision')
    v=matches[0]
    if not v.get('H_model_from_sample') or not v.get('physical_ids') or not v.get('observations'):
        raise ValueError('Anchor has no model geometry/physical observation chain')
    physical={p['physical_id'] for p in model.get('landmarks',[])}
    if not set(v['physical_ids']).issubset(physical):
        raise ValueError('Anchor names unknown physical groups')
    if source_hash is not None and v['source_sha256']!=source_hash:
        raise ValueError('Anchor source identity mismatch')
    return v


def _features(image):
    # Camera proposals only. In particular, the fixed programme timer and
    # skater silhouettes must not form a convincing but screen-fixed match.
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask=np.full(gray.shape,255,np.uint8)
    h,w=gray.shape
    mask[:round(h*.325),:round(w*.365)]=0
    mask[round(h*.805):,round(w*.8125):]=0
    foreground=ice_mask(image)
    for x1,y1,x2,y2 in foreground['foreground_boxes']:
        # Very large heuristic components can be entire printed/board areas.
        # The bounded boxes remain coarse exclusions, never person segmentation.
        if x2-x1 < .7*w and y2-y1 < h:
            pad=max(3,round(w/192))
            mask[max(0,int(y1)-pad):min(h,int(y2)+pad),
                 max(0,int(x1)-pad):min(w,int(x2)+pad)]=0
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.008,edgeThreshold=15)
    keys=[];descriptors=[]
    # Horizontal motion blur in a following shot can erase sharp-reference
    # letter features. Explicit variants retain their own proposal identities.
    # Duplicates across variants never count as additional physical support.
    for length in (0,7,11):
        view=gray if not length else cv2.filter2D(gray,-1,np.ones((1,length),np.float32)/length)
        key,desc=sift.detectAndCompute(view,mask)
        if desc is None:continue
        for point in key:point.class_id=length
        keys.extend(key);descriptors.append(desc)
    return keys,np.vstack(descriptors) if descriptors else None


def _proposal(ref_key,ref_desc,current_key,current_desc):
    if ref_desc is None or current_desc is None:return None
    matcher=cv2.BFMatcher();hypotheses=[]
    variants=sorted({point.class_id for point in ref_key})
    current_variants=sorted({point.class_id for point in current_key})
    # Compare a blurred reference to the actual query, plus the inverse case;
    # never match the concatenated variant banks as if they were new controls.
    pairs=[(v,0 if 0 in current_variants else current_variants[0]) for v in variants]
    pairs.extend((0,v) for v in current_variants if v!=0 and 0 in variants)
    for reference_variant,current_variant in pairs:
        ri=[i for i,k in enumerate(ref_key) if k.class_id==reference_variant]
        qi=[i for i,k in enumerate(current_key) if k.class_id==current_variant]
        if min(len(ri),len(qi))<6:continue
        rd=ref_desc[ri];qd=current_desc[qi]
        # RootSIFT improves descriptor comparison across the observed contrast
        # difference. The image coordinates and source identity do not change.
        rd=np.sqrt(rd/(rd.sum(axis=1,keepdims=True)+1e-12))
        qd=np.sqrt(qd/(qd.sum(axis=1,keepdims=True)+1e-12))
        forward=matcher.knnMatch(rd,qd,k=2)
        reverse=matcher.match(qd,rd)
        back={m.queryIdx:m.trainIdx for m in reverse}
        for ratio in (.76,.84,.90):
            matches=[m[0] for m in forward if len(m)==2 and m[0].distance<ratio*m[1].distance
                     and back.get(m[0].trainIdx)==m[0].queryIdx]
            if len(matches)<6:continue
            x=np.float32([ref_key[ri[m.queryIdx]].pt for m in matches])
            y=np.float32([current_key[qi[m.trainIdx]].pt for m in matches])
            cv2.setRNGSeed(20260920)
            a,flags=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=4.,
                                             maxIters=4000,confidence=.999)
            if a is None or flags is None:continue
            good=np.flatnonzero(flags.ravel()>0)
            # Multiple orientations at one SIFT location do not provide more
            # geometric information. Count one per paired 4-pixel cell.
            unique={}
            for j in good:
                cell=tuple(np.rint(np.r_[x[j],y[j]]/4).astype(int))
                unique.setdefault(cell,j)
            selected=np.asarray(list(unique.values()),dtype=int)
            if len(selected)<6:continue
            xx,yy=x[selected],y[selected]
            area=float(cv2.contourArea(cv2.convexHull(xx)))
            span=np.ptp(xx,axis=0)
            if area<250 or np.max(span)<60 or np.min(span)<7:continue
            scale=float(np.hypot(a[0,0],a[1,0]));angle=float(np.degrees(np.arctan2(a[1,0],a[0,0])))
            if not .35<=scale<=3.5 or abs(angle)>35:continue
            h=np.vstack([a,[0,0,1]])
            try:
                hull=np.array([[0,240],[959,240],[959,539],[0,539]],float)
                q=project(h,hull)
                if not cv2.isContourConvex(q.astype(np.float32)) or cv2.contourArea(q.astype(np.float32),oriented=True)<=0:continue
                if np.max(np.abs(q))>8000:continue
                residual=np.linalg.norm(project(h,xx)-yy,axis=1)
            except (ValueError,np.linalg.LinAlgError):continue
            hypotheses.append({'H':h,'matches':len(selected),'raw_inliers':int(flags.sum()),
                'reference_blur_px':reference_variant,'query_blur_px':current_variant,
                'descriptor':'RootSIFT','ratio_threshold':ratio,'scale':scale,'angle_deg':angle,
                'reference_hull_area_px2':area,'reference_span_px':span.tolist(),
                'median_camera_residual_px':float(np.median(residual)),
                'reference_points':xx.tolist(),'query_points':yy.tolist(),
                'proposal_role':'background_camera_hypothesis_only_not_ice_correspondence',
                'plane_geometry_supported':False})
    if not hypotheses:return None
    hypotheses.sort(key=lambda h:(-h['matches'],h['median_camera_residual_px'],h['ratio_threshold']))
    result=hypotheses[0]
    # Retain distinct alternatives for diagnostic/refinement callers. The
    # current pipeline still must pass every selected proposal through ice
    # refinement; none of these matrices can be directly emitted as a grid.
    alternatives=[]
    controls=np.array([[0,250],[480,250],[959,250],[480,539]],float)
    for candidate in hypotheses[1:]:
        distance=np.linalg.norm(project(candidate['H'],controls)-project(result['H'],controls),axis=1)
        if float(np.max(distance))<8:continue
        if any(float(np.max(np.linalg.norm(project(candidate['H'],controls)-project(a['H'],controls),axis=1)))<8 for a in alternatives):continue
        alternatives.append(candidate)
        if len(alternatives)==3:break
    result['alternatives']=alternatives
    return result


class ModelLocalizer:
    def __init__(self, model_path, *, remove_anchors=False, reobserve_every=10, refinement_motion_model='affine'):
        if isinstance(reobserve_every,bool) or not isinstance(reobserve_every,int) or reobserve_every<1:
            raise ValueError('reobserve_every must be a positive source-frame count')
        self.model_path=Path(model_path)
        self.model=load_model(model_path)
        self.refs=[]
        self.reobserve_every=reobserve_every
        self.max_prediction_s=.6
        self.refinement_config=CurveRefinementConfig(motion_model=refinement_motion_model)
        self.local=ContinuousRegistrar(ContinuousConfig(keyframe_interval_s=.2,max_keyframes=5),
             RegistrationConfig(min_tracks=12,min_inliers=9,min_support_fraction=.003,
                                min_support_width_fraction=.1,min_support_height_fraction=.035))
        self.last_global=None;self.last_anchor_time=None;self.last_component=None;self.bridge=None
        self.last_render=None
        if remove_anchors:return
        for v in self.model['reference_views']:
            if not v.get('H_model_from_sample'):continue
            require_anchor(self.model,v['id'])
            im=cv2.imread(str(self.model_path.parent/v['image_path']))
            if im is None:raise FileNotFoundError(v['image_path'])
            im=cv2.resize(im,(960,540))
            plane=polygon_mask(im.shape,v['ice_polygon_sample'])
            reference_foreground=row_foreground_exclusion(im,plane)
            mask=reference_foreground['usable']
            # Only source-observed printed groups enter cross-day re-observation.
            prints=np.zeros_like(mask)
            for c in v['observations']:
                contour=np.rint(ellipse_points(c['ellipse'])).astype(np.int32)
                cv2.fillPoly(prints,[contour],255)
            prints=cv2.dilate(prints,np.ones((45,45),np.uint8))
            mask=cv2.bitwise_and(mask,prints)
            key,desc=_features(im)
            self.refs.append({'view':v,'image':im,'gray':cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),
                              'mask':mask,'key':key,'desc':desc})

    def reset(self):
        self.last_global=None;self.last_anchor_time=None;self.last_component=None;self.bridge=None
        self.last_render=None

    def push(self,image,sample,force_break=False):
        if force_break:self.reset()
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        masks=ice_mask(image)
        foreground=row_foreground_exclusion(image,masks['ice'])
        masks['usable']=foreground['usable'];masks['foreground']=foreground['foreground']
        # Serialize the actual row-based exclusion contours, not a fixed
        # central rectangle which would hide valid ice prints in the viewer.
        contours,_=cv2.findContours(foreground['foreground'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        foreground_polygons=[cv2.approxPolyDP(c,1.,True).reshape(-1,2) for c in contours if cv2.contourArea(c)>=4]
        raw_scale=np.array([sample['width']/960,sample['height']/540])
        local=self.local.push(image,sample,masks,force_break=force_break)
        row={**sample,'model_id':self.model['model_id'],'model_revision':self.model['revision'],
             'matrix_direction':'H_model_from_image','image_domain':'raw_px_distortion_unchecked',
             'status':'unavailable','H_model_from_image':None,'anchor_id':None,
             'physical_ids':[],'support_polygon_image':[],
             'person_boxes_image':[],
             'foreground_polygons_image':[(p*raw_scale).tolist() for p in foreground_polygons],
             'image_width':sample['width'],'image_height':sample['height'],
             'independent_accuracy_validated':False,'geometry_level':'projective',
             'unit':'reference_px','algorithm_version':ALGORITHM_VERSION,
             'foreground_method':foreground['method'],
             'foreground_config':foreground['config'],
             'local_status':local['status'],'local_component_id':local.get('graph_component_id'),
             'error_summary':None,'reason':'no_model_ice_anchor','camera_group_id':None,
             'propagated_from':None,'reobservation_attempts':[]}
        ts=sample['source_time_s'];direct=None
        photo_prediction=None
        if (self.last_render is not None and self.last_global is not None
                and 0<ts-self.last_render['source_time_s']<=.08
                and ts-self.last_anchor_time<=self.max_prediction_s
                and (local.get('H_reference_from_current') is None or self.bridge is None
                     or local.get('graph_component_id')!=self.last_component)):
            previous=self.last_render
            search=cv2.dilate(previous['support_mask'],np.ones((37,37),np.uint8))
            cm=cv2.bitwise_and(search,masks['usable'])
            pair=refine_model_projection(previous['gray'],gray,previous['support_mask'],cm,
                np.eye(3),config=self.refinement_config)
            row['bounded_photometric_propagation']={'status':pair['status'],'reason':pair['reason'],
                'from_source_time_s':previous['source_time_s'],'max_pair_gap_s':.08,
                'anchor_age_s':ts-self.last_anchor_time,'max_anchor_age_s':self.max_prediction_s,
                'final_curve_residual':pair.get('final_curve_residual'),
                'role':'local_propagation_only_not_a_new_model_observation'}
            if pair['status']=='candidate':
                photo_prediction=previous['H_model_from_sample']@np.linalg.inv(np.asarray(pair['H_current_from_reference']))
        # A cut always performs fresh retrieval; no matrix can cross it.
        attempt=force_break or sample['sample_index']%self.reobserve_every==0
        if attempt and self.refs:
            key,desc=_features(image)
            proposals=[]
            if photo_prediction is not None:
                target=next((v for v in self.refs if v['view']['id']==self.last_global['anchor_id']),None)
                if target is not None:
                    hprior=np.linalg.inv(photo_prediction)@np.asarray(target['view']['H_model_from_sample'])
                    proposals.append((110000,target,{'H':hprior,'matches':0,
                        'proposal_role':'bounded_ice_photometric_motion_to_fixed_model_reobservation'}))
            # Reobserve the fixed model view using bounded local motion as an
            # initial guess. This is an actual drift correction, not a reset
            # of the shared map or an unbounded copy of the last transform.
            lh_now=local.get('H_reference_from_current')
            if (self.last_global is not None and self.bridge is not None and lh_now is not None
                    and local.get('graph_component_id')==self.last_component
                    and ts-self.last_anchor_time<=self.max_prediction_s):
                target=next((v for v in self.refs if v['view']['id']==self.last_global['anchor_id']),None)
                if target is not None:
                    predicted=self.bridge@np.asarray(lh_now)
                    hprior=np.linalg.inv(predicted)@np.asarray(target['view']['H_model_from_sample'])
                    proposals.append((100000,target,{'H':hprior,'matches':0,
                        'proposal_role':'bounded_local_motion_to_fixed_model_reobservation'}))
            elif self.last_global is not None and 0<=ts-self.last_anchor_time<=.25:
                target=next((v for v in self.refs if v['view']['id']==self.last_global['anchor_id']),None)
                if target is not None:
                    # A stationary initial guess is never emitted. It only
                    # seeds a fresh bounded, masked ice fit when local feature
                    # tracks are missing in the intervening low-texture frame.
                    hprior=np.linalg.inv(self.last_global['H_model_from_sample'])@np.asarray(target['view']['H_model_from_sample'])
                    proposals.append((90000,target,{'H':hprior,'matches':0,
                        'proposal_role':'last_model_observation_initial_guess_requires_fresh_ice_fit'}))
            for ref in self.refs:
                proposal=_proposal(ref['key'],ref['desc'],key,desc)
                if proposal:
                    proposals.append((proposal['matches'],ref,proposal))
                    for alternative in proposal.get('alternatives',[]):
                        proposals.append((alternative['matches'],ref,alternative))
            for _,ref,proposal in sorted(proposals,key=lambda r:-r[0])[:8]:
                h=proposal['H']
                # Reference ice extent protects against bright boards leaking
                # through the legacy heuristic mask.
                warp=cv2.warpPerspective(ref['mask'],h,(960,540),flags=cv2.INTER_NEAREST)
                current_mask=cv2.bitwise_and(warp,masks['usable'])
                fit=refine_model_projection(ref['gray'],gray,ref['mask'],current_mask,h,config=self.refinement_config)
                row['reobservation_attempts'].append({'anchor_id':ref['view']['id'],'camera_proposal_inliers':proposal['matches'],
                    'proposal_role':proposal['proposal_role'],'status':fit['status'],'reason':fit['reason'],
                    'initial_curve_residual':fit.get('initial_curve_residual'),
                    'final_curve_residual':fit.get('final_curve_residual'),
                    'initial_support':fit.get('initial_support'),
                    'initial_affine_normal_sensitivity':fit.get('initial_affine_normal_sensitivity'),
                    'initial_texture_std':fit.get('initial_texture_std')})
                if fit['status']!='candidate':continue
                hcur=np.asarray(fit['H_current_from_reference'])
                hmodel=np.asarray(ref['view']['H_model_from_sample'])@np.linalg.inv(hcur)
                support=fit['support_current_polygon']
                if len(support)<3:continue
                visible=[]
                for curve in ref['view']['observations']:
                    xy=project(hcur,ellipse_points(curve['ellipse']))
                    q=np.rint(xy).astype(int)
                    inside=(q[:,0]>=0)&(q[:,0]<960)&(q[:,1]>=0)&(q[:,1]<540)
                    ii=np.flatnonzero(inside)
                    inside[ii]&=current_mask[q[ii,1],q[ii,0]]>0
                    visible.append({'physical_id':curve['physical_id'],'curve_samples_in_support':int(inside.sum()),
                                    'curve_sample_count':len(q),'independent_validation':False})
                direct={'H_model_from_sample':hmodel,'anchor_id':ref['view']['id'],
                        'physical_ids':ref['view']['physical_ids'],
                        'support_model':project(hmodel,support).tolist(),
                        'camera_group_id':ref['view'].get('camera_group_id','candidate'),
                        'error_summary':fit.get('final_curve_residual'),
                        'anchor_source_pts':ref['view']['pts'],'anchor_source_time_base':ref['view']['time_base'],
                        'visible_curve_support':visible,
                        'is_reference_self_frame':sample['source_sha256']==ref['view']['source_sha256'] and sample['pts']*Fraction(sample['time_base'])==ref['view']['pts']*Fraction(ref['view']['time_base'])}
                break
        lh=local.get('H_reference_from_current')
        component=local.get('graph_component_id')
        if direct:
            self.last_global=direct;self.last_anchor_time=ts
            self.last_component=component
            self.bridge=direct['H_model_from_sample']@np.linalg.inv(lh) if lh is not None else None
            h=direct['H_model_from_sample'];state='candidate';why='direct_fixed_model_print_reobservation_geometry_candidate'
        elif (self.last_global is not None and self.bridge is not None and lh is not None and
              component==self.last_component and ts-self.last_anchor_time<=self.max_prediction_s):
            h=self.bridge@np.asarray(lh);state='predicted';why='bounded_local_ice_propagation_from_model_anchor'
        elif photo_prediction is not None:
            h=photo_prediction;state='predicted';why='bounded_masked_ice_photometric_propagation_from_model_anchor'
        else:
            return row
        a=self.last_global
        try:support=project(np.linalg.inv(h),a['support_model']).tolist()
        except (ValueError,np.linalg.LinAlgError):return row
        scale_xy=np.array([960/sample['width'],540/sample['height']])
        row.update(status=state,H_model_from_image=(h@np.diag([*scale_xy,1])).tolist(),
            anchor_id=a['anchor_id'],physical_ids=a['physical_ids'],
            support_polygon_image=(np.asarray(support)/scale_xy).tolist(),
            camera_group_id=a['camera_group_id'],error_summary=a['error_summary'],reason=why,
            anchor_physical_ids=a['physical_ids'],visible_curve_support=a['visible_curve_support'],
            reobserved_physical_ids=[v['physical_id'] for v in a['visible_curve_support'] if v['curve_samples_in_support']>=8] if direct else [],
            is_reference_self_frame=a['is_reference_self_frame'] if direct else False,
            anchor_source_pts=a['anchor_source_pts'],anchor_source_time_base=a['anchor_source_time_base'],
            propagated_from=None if direct else {'anchor_source_time_s':self.last_anchor_time,
                'local_component_id':component,'max_gap_s':self.max_prediction_s,'future_anchor_used':False,
                'method':why})
        support_mask=cv2.bitwise_and(polygon_mask(gray.shape,support),masks['usable'])
        self.last_render={'gray':gray.copy(),'H_model_from_sample':h.copy(),
            'source_time_s':ts,'support_mask':support_mask}
        return row


def localize_window(root,model_path,source_id,start_s,end_s,output,*,selection_basis,
                    cut_times=(),remove_anchors=False,evaluation_lock=None,
                    refinement_motion_model='affine',reobserve_every=10):
    root,output=Path(root).resolve(),Path(output).resolve()
    evaluation={}
    if evaluation_lock:
        if remove_anchors:raise ValueError('Anchor ablation is a separate control, not the frozen evaluation')
        from .model_evaluation import check_evaluation
        evaluation=check_evaluation(root,evaluation_lock,model_path,source_id,start_s,end_s,cut_times,
            runtime_config={'refinement_motion_model':refinement_motion_model,'reobserve_every':reobserve_every})
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4)
    index=load_index(root,source_id)
    chosen=[r for r in index['frames'] if start_s<=r['source_time_s']<end_s]
    if not chosen:raise ValueError('No frames in selected full source interval')
    lookup={(r['pts'],r['time_base']):r for r in chosen}
    model=load_model(model_path)
    identity=RinkReference(model['model_id'],model['revision'],model['geometry_level'],model['unit'],model['coordinate_definition'])
    localizer=ModelLocalizer(model_path,remove_anchors=remove_anchors,
        refinement_motion_model=refinement_motion_model,reobserve_every=reobserve_every)
    frames=[];mapping=[]
    proxy=output/'source.mp4'
    encoded=av.open(str(proxy),'w')
    writer=encoded.add_stream('libx264',rate=Fraction(1,Fraction(chosen[1]['time_base'])*(chosen[1]['pts']-chosen[0]['pts'])))
    writer.width=640;writer.height=360;writer.pix_fmt='yuv420p';writer.options={'crf':'29','preset':'fast'}
    writer.time_base=Fraction(chosen[0]['time_base'])
    first_time=chosen[0]['source_time_s'];last_time=None;shot=0
    print(f'localizing {source_id}: {len(chosen)} full source frames',flush=True)
    with av.open(str(source_path(root,source_id))) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        container.seek(int(Fraction(chosen[0]['pts'])*Fraction(chosen[0]['time_base'])/stream.time_base),stream=stream)
        for f in container.decode(stream):
            if f.pts is None:continue
            ts=float(f.pts*f.time_base)
            if ts>=end_s:break
            original=lookup.get((int(f.pts),str(f.time_base)))
            if original is None:continue
            i=len(frames)
            broken=i==0 or any(last_time<t<=ts for t in cut_times)
            if i and broken:shot+=1
            sample={**original,'sample_index':i,'source_sha256':index['video_id'],
                'source_id':source_id,'source_pts':int(f.pts),'source_time_base':str(f.time_base),
                'source_time_s':ts,'width':f.width,'height':f.height,'shot_id':f'{source_id}_selected_{shot}',
                'source_frame_index':original['source_frame_index']}
            image=cv2.resize(f.to_ndarray(format='bgr24'),(960,540),interpolation=cv2.INTER_AREA)
            row=localizer.push(image,sample,force_break=broken)
            validate_frame_map(row,identity)
            frames.append(row)
            small=cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            vf=av.VideoFrame.from_ndarray(small,format='bgr24')
            vf.pts=int(f.pts)-chosen[0]['pts'];vf.time_base=f.time_base
            for packet in writer.encode(vf):encoded.mux(packet)
            mapping.append({'pts':int(f.pts),'time_base':str(f.time_base),'source_frame_index':original['source_frame_index']})
            if i in (0,len(chosen)//2,len(chosen)-1):cv2.imwrite(str(output/f'source_{i:05d}.jpg'),image)
            if i%100==0:print(i,dict(Counter(r['status'] for r in frames)),flush=True)
            last_time=ts
    for packet in writer.encode():encoded.mux(packet)
    encoded.close()
    if len(frames)!=len(chosen):raise RuntimeError('Decoded full source window lost frames')
    with av.open(str(proxy)) as c:
        dec=list(c.decode(video=0))
        if len(dec)!=len(mapping):raise RuntimeError('Proxy frame count mismatch')
        for p,f in zip(mapping,dec):p.update(proxy_pts=int(f.pts),proxy_time_base=str(f.time_base))
    record={'schema_version':'rinkspace.model-localization.v1','algorithm_version':ALGORITHM_VERSION,
        **evaluation,
        'source_id':source_id,'source_sha256':index['video_id'],'model_id':model['model_id'],
        'model_revision':model['revision'],'model_sha256':sha256(model_path),
        'image_width':frames[0]['width'],'image_height':frames[0]['height'],
        'media_time_offset_s':0,'proxy_source_time_offset_s':first_time,
        'frames':frames,'selected_window':{'first_pts':frames[0]['pts'],'last_pts':frames[-1]['pts'],
            'time_base':frames[0]['time_base'],'frame_count':len(frames),'start_s':start_s,'end_s':end_s,
            'selection_basis':selection_basis},'cut_times_s':list(cut_times),
        'status_counts':dict(Counter(r['status'] for r in frames)),
        'all_selected_frames_in_denominator':True,'independent_validation_passed':False,
        'anchor_ablation':remove_anchors,'reobserve_every_source_frames':reobserve_every,
        'config':asdict(localizer.refinement_config)}
    dump_json(output/'localization.json',record)
    dump_json(output/'source_pts_map.json',{'source_sha256':index['video_id'],'proxy_sha256':sha256(proxy),'frames':mapping})
    return record
