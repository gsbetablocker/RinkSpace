"""Candidate-local visible ice support inside a coarse primary-person box.

The historical all-person union remains unchanged. A named source material
needs two real RGB images, actual measured person contours and an unambiguous
material interface in both images before its small support window is cleared.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .vision_ice_temporal_identity import _point_polyline_distance,_measured_material_descriptors
from .vision_ice_identity_graph import _point_source_flow
from .vision_programme_ice_review import _actual_profile_candidates,_batched_actual_profiles
from .vision_source_shapes import _MaterialProfileFrame


def primary_person_box_components(person_masks,pts):
    """Separate one uniquely SAM2-associated detector box from all other boxes."""
    person,status=person_masks.get(int(pts),require_sam2=True)
    person=np.asarray(person,bool)
    boxes=person_masks.boxes[int(pts)]
    rectangles=[];coverage=[]
    for box in boxes:
        x1,y1,x2,y2=np.rint(np.asarray(box[:4])*2).astype(int)
        rect=(max(0,x1-28),max(0,y1-28),min(person.shape[1],x2+29),min(person.shape[0],y2+29))
        rectangles.append(rect)
        l,t,r,b=rect
        coverage.append(float(person[t:b,l:r].sum())/max(1,int(person.sum())))
    matches=np.flatnonzero(np.asarray(coverage)>=.90)
    primary=np.zeros_like(person);others=np.zeros_like(person)
    selected=int(matches[0]) if len(matches)==1 else None
    for index,(l,t,r,b) in enumerate(rectangles):
        (primary if index==selected else others)[t:b,l:r]=True
    # Existing SAM2 has already been dilated in PersonMasks.get. Additional
    # measured contour context is kept during source material verification.
    hard=person|others
    return dict(primary_box_mask=primary,measured_person_mask=person,
        other_person_boxes_mask=others,hard_person_exclusion_mask=hard,
        primary_box_index=selected,primary_box_SAM2_coverage=None if selected is None else coverage[selected],
        measured_person_status=status)


def _context_pixels(point,normal,extent=24.):
    tangent=np.array([-normal[1],normal[0]])
    return np.rint(point[None,None]+normal[None,None]*np.arange(-extent,extent+1)[None,:,None]
        +tangent[None,None]*np.arange(-3,4)[:,None,None]).astype(int).reshape(-1,2)


def _read_one_material(frame,point,normal,role,extent):
    offsets,profiles,smooth,valid=_batched_actual_profiles(frame,np.asarray(point)[None],np.asarray(normal)[None],
        np.array([[-extent,extent]]),np.array([role['material_signal']]))
    take=valid[0]
    candidates=_actual_profile_candidates(point,normal,offsets[0,take],profiles[0,take],smooth[0,take],
        operator=role['operator'],signal=role['material_signal'],transition=role['material_transition'],with_quantiles=True)
    matches=[v for v in candidates if not v[2] and np.linalg.norm(v[0]-point)<=3.]
    return matches[0] if len(matches)==1 else None


class KnownIceMaterialBoxClearance:
    """Only actual source-anchored, two-image static material support is cleared."""
    def __init__(self,inventory,*,source_sha256,time_base,seed_registry=()):
        self.inventory=[dict(v) for v in inventory]
        self.source_sha256,self.time_base=source_sha256,str(time_base)
        self.seeds={}
        records=seed_registry.get('records',()) if isinstance(seed_registry,dict) else seed_registry
        for seed in records:self.seeds.setdefault(int(seed['actual_pts']),[]).append(seed)
        self.previous=None
        self.confirmed_support=[]

    def remember_current_source_rows(self,rows,measurement):
        """Retain confirmed actual material supports, never unknown box pixels."""
        if isinstance(rows,dict):rows=rows['rows']
        pts=int(measurement['source_binding']['source_pts'])
        if self.previous is None or self.previous['pts']!=pts:raise ValueError('Remember only this just-consumed original frame')
        self.confirmed_support=[]
        for row in rows:
            if row.get('actual_pts')!=pts or row.get('actual_source_RGB_measured') is not True:
                continue
            if row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity'):continue
            ids=np.asarray(row.get('original_candidate_indices',()),int)
            if len(ids)<2:continue
            curve=next(v for v in self.inventory if v['curve_id']==row['curve_id'])
            from .vision_ice_temporal_identity import exact_selected_source_readouts
            uv,sides=exact_selected_source_readouts(row,measurement,operator=curve['operator'])
            normal=np.asarray(measurement['query_normal'],float)[np.asarray(measurement['candidate_query'])[ids]].copy()
            width=np.linalg.norm(sides[:,1]-sides[:,0],axis=1)
            paired=width>1e-6
            normal[paired]=(sides[paired,1]-sides[paired,0])/width[paired,None]
            self.confirmed_support.append(dict(uv=uv,sides=sides,normal=normal,
                seed=dict(row['source_seed_reference'],curve_id=row['curve_id'],authority=row['source_seed_authority'])))

    def consume(self,rgb,measurement,conservative_exclusion_mask,*,source_ice_scope_mask,
                primary_box_mask,measured_person_mask,other_person_boxes_mask,seeds=(),known_source_rows=()):
        binding=measurement['source_binding'];pts=int(binding['source_pts'])
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        if (digest!=measurement['decoded_native_RGB_sha256'] or binding['source_sha256']!=self.source_sha256
                or str(binding['time_base'])!=self.time_base):raise ValueError('Local clearance needs exact original RGB clock')
        bgr_digest=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest()
        primary=np.asarray(primary_box_mask,bool)
        hard=np.asarray(measured_person_mask,bool)|np.asarray(other_person_boxes_mask,bool)
        conservative=np.asarray(conservative_exclusion_mask,bool)
        unrelated=conservative&~primary
        hard|=unrelated
        # No visibility inference comes from a low mask probability. Full RGB
        # material context and two-image source readouts remain required below.
        padded=cv2.dilate(hard.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
        profile_frame=_MaterialProfileFrame(rgb,binding,digest)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        current=dict(frame=profile_frame,gray=gray,hard=padded,primary=primary,pts=pts,
            binding=dict(binding),BGR_sha256=bgr_digest)
        effective_mask=conservative.copy();scope=np.asarray(source_ice_scope_mask,bool).copy()
        result=dict(measurement);flags=np.asarray(measurement['candidate_flags']).copy()
        proof_by_candidate={};decisions=[]
        uv=np.asarray(measurement['candidate_uv'],float);sides=np.asarray(measurement['candidate_sides'],float)
        q=np.asarray(measurement['candidate_query']);curve=np.asarray(measurement['query_curve'])[q]
        normals=np.asarray(measurement['query_normal'],float)[q]
        anchors=[*self.seeds.get(pts,()),*seeds]
        for row in known_source_rows:
            anchors.append(dict(row['source_seed_reference'],curve_id=row['curve_id'],actual_pts=pts,
                navigation_uv=row['uv'],decoded_native_RGB_sha256=digest))
        previous=self.previous
        if previous is not None and previous['binding']['shot_id']==binding['shot_id'] and previous['pts']<pts:
            for confirmed in self.confirmed_support:
                navigation,valid,errors=_point_source_flow(previous['gray'],gray,confirmed,previous['hard'],padded)
                if valid.sum()<2:continue
                anchors.append(dict(confirmed['seed'],actual_pts=pts,navigation_uv=navigation[valid].tolist(),
                    decoded_native_RGB_sha256=digest,source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
                    previous_actual_source_navigation_proof=dict(actual_pts=previous['pts'],decoded_native_BGR_sha256=previous['BGR_sha256'],
                        source_material_LK_forward_backward_error_px=errors[valid].tolist(),projected_UV_used_for_identity=False)))
            for seed in anchors:
                if (seed.get('surface_and_physical_identity_confirmed') is not True or seed.get('source_material_class')!='static_painted_ice'
                        or not seed.get('authority') or seed['source_sha256']!=self.source_sha256 or seed['actual_pts']!=pts):continue
                if str(seed.get('time_base'))!=self.time_base or seed.get('shot_id')!=binding['shot_id']:continue
                if seed.get('decoded_native_RGB_sha256') not in (None,digest):continue
                name=seed['curve_id'];ci=next((i for i,v in enumerate(self.inventory) if v['curve_id']==name),None)
                if ci is None or self.inventory[ci]['operator']=='unresolved':continue
                role=self.inventory[ci]
                ids=np.flatnonzero((curve==ci)&((flags&1)==0))
                distance=_point_polyline_distance(uv[ids],seed['navigation_uv'])
                ids=ids[distance<=12.]
                admitted=[]
                for candidate in ids:
                    point=uv[candidate];normal=normals[candidate]
                    width=float(np.linalg.norm(sides[candidate,1]-sides[candidate,0]))
                    extent=max(24.,width*.5+12.)
                    pixels=_context_pixels(point,normal,extent)
                    inside=(pixels>=0).all(1)&(pixels<[rgb.shape[1],rgb.shape[0]]).all(1)
                    if not inside.all() or padded[pixels[:,1],pixels[:,0]].any():continue
                    box_context=bool(primary[pixels[:,1],pixels[:,0]].any())
                    missing_surface_scope=bool((~scope[pixels[:,1],pixels[:,0]]).any())
                    if not box_context and not missing_surface_scope:continue
                    actual=_read_one_material(profile_frame,point,normal,role,extent)
                    if actual is None:continue
                    support=np.concatenate([actual[0][None],actual[1]],axis=0).astype(np.float32)[:,None]
                    # Read backwards to the adjacent source image, then back to
                    # the current source: the known current identity can prove
                    # an earlier static material instance without model motion.
                    back,good,_=cv2.calcOpticalFlowPyrLK(gray,previous['gray'],support,None,winSize=(31,31),maxLevel=4,
                        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.01))
                    if back is None:continue
                    forward,reverse,_=cv2.calcOpticalFlowPyrLK(previous['gray'],gray,back,None,winSize=(31,31),maxLevel=4,
                        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.01))
                    if forward is None or not good.all() or not reverse.all():continue
                    fb=np.linalg.norm(forward[:,0]-support[:,0],axis=1)
                    if not np.isfinite(back).all() or fb.max()>2.:continue
                    previous_point=back[0,0].astype(float)
                    previous_pixels=_context_pixels(previous_point,normal,extent)
                    inside=(previous_pixels>=0).all(1)&(previous_pixels<[rgb.shape[1],rgb.shape[0]]).all(1)
                    if not inside.all() or previous['hard'][previous_pixels[:,1],previous_pixels[:,0]].any():continue
                    before=_read_one_material(previous['frame'],previous_point,normal,role,extent)
                    if before is None:continue
                    current_descriptor,strength=_measured_material_descriptors(rgb,actual[0][None],actual[1][None],normal[None])
                    previous_descriptor,old_strength=_measured_material_descriptors(previous['frame'].rgb,before[0][None],before[1][None],normal[None])
                    correlation=float(current_descriptor[0]@previous_descriptor[0])
                    if strength[0]<=0 or old_strength[0]<=0 or correlation<.90:continue
                    current_width=np.linalg.norm(actual[1][1]-actual[1][0]);previous_width=np.linalg.norm(before[1][1]-before[1][0])
                    if role['operator']=='paint_center' and not .65<=current_width/max(previous_width,1e-6)<=1.55:continue
                    proof=dict(kind='known_static_ice_material_visible_inside_primary_detector_box',curve_id=name,
                        original_candidate_index=int(candidate),original_candidate_flags=int(flags[candidate]),
                        primary_detector_box_context_present=box_context,original_source_ice_scope_was_unknown=missing_surface_scope,
                        actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),shot_id=binding['shot_id'],
                        source_sha256=self.source_sha256,time_base=self.time_base,decoded_native_BGR_sha256=bgr_digest,
                        source_identity_authority=seed['authority'],source_identity_confirmed=True,
                        source_identity_navigation_continuity=seed.get('previous_actual_source_navigation_proof'),
                        previous_actual_pts=previous['pts'],previous_decoded_BGR_sha256=previous['BGR_sha256'],
                        current_actual_RGB_interface_uv=actual[0].tolist(),current_actual_RGB_sides_uv=actual[1].tolist(),
                        previous_actual_RGB_interface_uv=before[0].tolist(),previous_actual_RGB_sides_uv=before[1].tolist(),
                        actual_material_RGB_context_correlation=correlation,source_forward_backward_LK_error_px=fb.tolist(),
                        current_complete_material_context=dict(source_center_uv=point.tolist(),source_normal=normal.tolist(),
                            normal_interval_px=[-extent,extent],tangent_interval_px=[-3,3],
                            actual_native_context_pixel_count=len(pixels),
                            actual_native_context_pixel_indices_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()),
                        measured_person_dilated_contour_and_other_person_boxes_excluded=True,
                        only_primary_box_context_exclusion_removed=True,
                        actual_source_painted_ice_membership_confirmed=True,
                        model_projection_or_fit_residual_used_for_identity=False,mask_probability_used_as_visibility_truth=False)
                    proof_by_candidate[int(candidate)]=proof;admitted.append(int(candidate))
                    effective_mask[pixels[:,1],pixels[:,0]]=False
                    scope[pixels[:,1],pixels[:,0]]=True
                    flags[candidate]&=np.uint8(253)
                decisions.append(dict(curve_id=name,actual_pts=pts,source_local_clearance_candidates=len(admitted)))
        result['candidate_flags']=flags
        result['candidate_local_source_clearance']=proof_by_candidate
        self.previous=current
        return dict(measurement=result,exclusion_mask=effective_mask,source_ice_scope_mask=scope,
            candidate_local_source_clearance=proof_by_candidate,decisions=decisions,
            historical_person_union_modified=False,unknown_material_cleared=False)
