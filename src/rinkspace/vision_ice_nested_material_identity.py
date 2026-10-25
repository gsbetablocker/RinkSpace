"""Identify nested house layers from an already named actual material boundary.

The known source entity supplies its name. The current image supplies the
inward material direction and every alternating interface. Camera coordinates,
model radii and projected centers do not decide material identities or UVs.
Only independently measured current candidate UVs can be returned to callers.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .vision_ice_temporal_identity import _current_readout_uncertainty


LAYERS = ('outer', 'middle', 'inner', 'center')


def _source_nested_profile(rgb, mask, point, direction,*,native_rgb_field=None,native_luminance_field=None,
                           first_sign=-1,max_layers=4):
    """Actual RGB signed transitions on one native ray, including both flanks."""
    h, w = mask.shape
    tangent = np.array([-direction[1], direction[0]])
    ends = []
    for axis, limit in enumerate((w-4, h-4)):
        if abs(direction[axis]) > 1e-9:
            ends.append(((limit if direction[axis] > 0 else 3)-point[axis])/direction[axis])
    end = int(np.floor(min(ends))) if ends else 0
    if end < 30:
        return []
    offsets = np.arange(-16., end+1.)
    samples = point[None,None]+direction[None,None]*offsets[None,:,None]+tangent[None,None]*np.arange(-2.,3.)[:,None,None]
    xy = np.rint(samples.astype(np.float32)*32.)/32.
    pix = np.rint(xy).astype(int)
    valid = (pix>=0).all(2)&(pix<[w,h]).all(2)
    clear = valid.copy()
    clear[valid] &= ~mask[pix[valid,1],pix[valid,0]]
    # Do not cross an unknown surface, person or image edge and resume beyond
    # it: that would bridge missing source evidence to a different material.
    clear = clear.all(0)
    anchor = int(np.searchsorted(offsets, 0.))
    if not clear[max(0,anchor-6):anchor+7].all():
        return []
    left = anchor
    while left and clear[left-1]:left-=1
    right=anchor+1
    while right<len(clear) and clear[right]:right+=1
    if offsets[right-1]<24. or anchor-left<6:
        return []
    xy=xy[:,left:right];offsets=offsets[left:right]
    rgb_field=rgb.astype(np.float32) if native_rgb_field is None else native_rgb_field
    values = cv2.remap(rgb_field,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
    colours = np.median(values,axis=0).astype(float)
    # Match the established native uint8 OpenCV luminance measurement rather
    # than redefining its rounding in this identity check.
    luminance=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32) if native_luminance_field is None else native_luminance_field
    lum=np.median(cv2.remap(luminance,
        xy[...,0],xy[...,1],cv2.INTER_LINEAR),axis=0).astype(float)
    smooth = cv2.GaussianBlur(lum[None],(5,1),1.,borderType=cv2.BORDER_REPLICATE)[0]
    gradient=np.diff(smooth)
    second=np.diff(smooth,n=2)
    noise=float(1.4826*np.median(np.abs(second-np.median(second))))
    active=np.abs(gradient)>max(.25,noise)
    signs=np.sign(gradient).astype(int)
    boundaries=[]
    for sign in (-1,1):
        take=active&(signs==sign)
        starts=np.flatnonzero(take&~np.r_[False,take[:-1]])
        ends=np.flatnonzero(take&~np.r_[take[1:],False])+1
        merged=[]
        for a,b in zip(starts,ends):
            if merged and a-merged[-1][1]<5 and not (sign*gradient[merged[-1][1]:a]<-max(1.,3*noise)).any():
                merged[-1][1]=int(b)
            else:merged.append([int(a),int(b)])
        for a,b in merged:
            lo,hi=max(0,a-3),min(len(gradient),b+3)
            mass=np.maximum(sign*gradient[lo:hi],0.)
            amount=float(mass.sum())
            if amount<max(4.,3*noise) or lo==0 or hi==len(gradient):continue
            qs=np.interp([.25,.5,.75],np.r_[0.,np.cumsum(mass)]/amount,offsets[lo:hi+1])
            boundaries.append(dict(offset=float(qs[1]),quantiles=qs,sign=sign,contrast=amount,
                uv=point+direction*qs[1],direction=direction))
    boundaries.sort(key=lambda v:v['offset'])
    near=[(i,v) for i,v in enumerate(boundaries) if abs(v['offset'])<=3. and v['sign']==first_sign]
    if len(near)!=1:return []
    start=near[0][0]
    chain=[boundaries[start]]
    for v in boundaries[start+1:]:
        if len(chain)==max_layers:break
        # A crossing narrow paint stroke creates an extra close transition;
        # such a sequence is not evidence for a nested filled house layer.
        if v['sign']!=-chain[-1]['sign'] or v['offset']-chain[-1]['offset']<6.:break
        between=(offsets>chain[-1]['quantiles'][2]+1.)&(offsets<v['quantiles'][0]-1.)
        if between.sum()<3:break
        current_colour=np.median(colours[between],axis=0)
        chain.append({**v,'preceding_material_RGB':current_colour.tolist()})
    # Repeated painted regions must actually have a compatible RGB material,
    # rather than a dark reflected advertisement followed by another stripe.
    if len(chain)>3:
        a=np.asarray(chain[1]['preceding_material_RGB'])
        b=np.asarray(chain[3]['preceding_material_RGB'])
        if np.linalg.norm((a-a.mean())-(b-b.mean()))>12.:
            return chain[:2]
    return chain


class SourceHouseLayerIdentityExpander:
    """Supplement real nested interfaces without assigning any projected UV."""
    def __init__(self,inventory,*,source_sha256,time_base,profile_backend='cpu'):
        self.inventory=[dict(v) for v in inventory]
        self.by_name={v['curve_id']:i for i,v in enumerate(inventory)}
        self.source_sha256,self.time_base=source_sha256,str(time_base)
        self.profile_backend=profile_backend

    def _append_observed_interface_searches(self,rgb,measurement,mask,definition,known,measured_rays,layer):
        """Reread actually found interfaces missed by frozen query stations."""
        from .vision_ice_source_guides import _append_guided_readouts
        if any(np.asarray(measurement[k]).dtype!=np.dtype(np.float64) for k in ('query_uv','query_normal','query_interval')):
            return {}
        points=[];normals=[];intervals=[];proofs=[];query_for_source={}
        olduv=np.asarray(measurement['candidate_uv'],float);oldquery=np.asarray(measurement['candidate_query'])
        oldcurve=np.asarray(measurement['query_curve'])[oldquery];oldnormal=np.asarray(measurement['query_normal'],float)[oldquery]
        oldflags=np.asarray(measurement['candidate_flags']);ci=self.by_name[definition['curve_id']]
        desired=1 if definition['material_transition']=='dark_to_light' else -1
        for source_candidate,ray,labels in measured_rays:
            if layer not in labels:continue
            level=labels.index(layer)
            if not level or len(ray)<=level:continue
            physical=ray[level];point=np.asarray(physical['uv'],float)
            eligible=np.flatnonzero((oldcurve==ci)&((oldflags&3)==0)&(np.linalg.norm(olduv-point,axis=1)<=2.)
                &(np.abs(oldnormal@physical['direction'])>=.90))
            if len(eligible)==1:continue
            if any(np.linalg.norm(point-v)<1. for v in points):continue
            normal=np.asarray(physical['direction'],float)*(1. if physical['sign']==desired else -1.)
            window=np.array([-128.,128.])
            for axis,limit in ((0,rgb.shape[1]-1),(1,rgb.shape[0]-1)):
                if abs(normal[axis])<1e-12:continue
                a,b=sorted(((4-point[axis])/normal[axis],(limit-4-point[axis])/normal[axis]))
                window[0]=max(window[0],a);window[1]=min(window[1],b)
            if window[1]-window[0]<20.:continue
            query_for_source[source_candidate]=len(measurement['query_curve'])+len(points)
            points.append(point);normals.append(normal);intervals.append(window)
            proofs.append(dict(known_source_candidate_index=source_candidate,actual_source_layer_order=list(labels[:level+1]),
                actual_source_interface_uv=[v['uv'].tolist() for v in ray[:level+1]],
                actual_source_transition_signs=[v['sign'] for v in ray[:level+1]],
                actual_source_contrasts=[v['contrast'] for v in ray[:level+1]],model_projection_used_for_identity=False))
        if not points:return {}
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        first=len(measurement['query_curve'])
        guide=dict(curve_id=definition['curve_id'],actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),
            source_sha256=self.source_sha256,time_base=self.time_base,shot_id=binding['shot_id'],
            decoded_native_RGB_sha256=measurement['decoded_native_RGB_sha256'],
            decoded_native_BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),
            authority=known['source_seed_authority'],original_source_image_actually_reviewed=False,
            physical_identity_from_model_projection=False,guide_origin='actual_named_house_nested_material_interface_navigation',
            actual_current_RGB_nested_material_identity_evidence=proofs,known_named_source_house_reference=known['source_seed_reference'],
            navigation_uv=np.asarray(points).tolist(),navigation_UV_used_as_measurement=False)
        overlay=_append_guided_readouts(rgb,measurement,mask,self.inventory,np.asarray(points,float),np.asarray(normals,float),
            np.asarray(intervals,float),np.full(len(points),ci,np.int16),np.zeros(len(points),np.int32),
            [dict(guide,first_appended_query=first,appended_query_count=len(points))],guides=[guide],profile_backend=self.profile_backend)
        # Preserve old arrays/indices and the immutable cache. Only this caller's
        # current-source dictionary receives the explicit appended overlay.
        measurement.clear();measurement.update(overlay)
        return query_for_source

    def consume(self,rgb,measurement,exclusion_mask,*,source_ice_scope_mask,selected_rows=(),current_measured_curve_ids=()):
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        if (measurement['decoded_native_RGB_sha256']!=digest or binding['source_sha256']!=self.source_sha256
                or str(binding['time_base'])!=self.time_base):
            raise ValueError('Nested source identity requires the exact original RGB clock')
        mask=np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        uv=np.asarray(measurement['candidate_uv'],float)
        sides=np.asarray(measurement['candidate_sides'],float)
        queries=np.asarray(measurement['candidate_query'])
        curve=np.asarray(measurement['query_curve'])[queries]
        normals=np.asarray(measurement['query_normal'],float)[queries]
        flags=np.asarray(measurement['candidate_flags'])
        selected=selected_rows.get('rows',()) if isinstance(selected_rows,dict) else selected_rows
        # A current RGB readout already represented in the active source guides
        # must not become a new nested navigation group on each exposure.
        already={r['curve_id'] for r in selected}|set(current_measured_curve_ids)
        rows=[];decisions=[]
        native_rgb_field=native_luminance_field=None
        for known in selected:
            family,role=known['curve_id'].rsplit(':',1)
            if role not in LAYERS or family[:1] not in ('F','R') or family[1:] not in ('1','2','3','4'):continue
            if native_rgb_field is None:
                native_rgb_field=rgb.astype(np.float32)
                native_luminance_field=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
            level_known=LAYERS.index(role)
            first_inward_sign=-1 if role in ('outer','inner') else 1
            measured_rays=[]
            for candidate in known.get('original_candidate_indices',()):
                point=uv[candidate];normal=normals[candidate]
                for step in (1,-1):
                    labels=LAYERS[level_known:] if step==1 else tuple(reversed(LAYERS[:level_known+1]))
                    if len(labels)<2:continue
                    first_sign=first_inward_sign*step
                    # The image, rather than a projected center, chooses the
                    # direction with the required actually observed interface.
                    chains=[_source_nested_profile(rgb,mask,point,sign*normal,native_rgb_field=native_rgb_field,
                        native_luminance_field=native_luminance_field,first_sign=first_sign,max_layers=len(labels)) for sign in (1.,-1.)]
                    viable=[c for c in chains if len(c)>1]
                    if len(viable)==1:measured_rays.append((int(candidate),viable[0],labels))
            for layer in LAYERS:
                name=family+':'+layer
                if name not in self.by_name or name in already:continue
                ci=self.by_name[name];definition=self.inventory[ci]
                source_queries=self._append_observed_interface_searches(rgb,measurement,mask,definition,known,measured_rays,layer)
                if source_queries:
                    uv=np.asarray(measurement['candidate_uv'],float);sides=np.asarray(measurement['candidate_sides'],float)
                    queries=np.asarray(measurement['candidate_query']);curve=np.asarray(measurement['query_curve'])[queries]
                    normals=np.asarray(measurement['query_normal'],float)[queries];flags=np.asarray(measurement['candidate_flags'])
                candidates=np.flatnonzero((curve==ci)&((flags&3)==0))
                evidence={};matched=[]
                for outer_candidate,ray,labels in measured_rays:
                    if layer not in labels:continue
                    level=labels.index(layer)
                    if not level or len(ray)<=level:continue
                    physical=ray[level]
                    distance=np.linalg.norm(uv[candidates]-physical['uv'],axis=1)
                    compatible=candidates[(distance<=2.)&(np.abs(normals[candidates]@physical['direction'])>=.90)]
                    if outer_candidate in source_queries:
                        compatible=compatible[queries[compatible]==source_queries[outer_candidate]]
                    if len(compatible)!=1:continue
                    candidate=int(compatible[0]);pixel=np.rint(uv[candidate]).astype(int)
                    if mask[pixel[1],pixel[0]]:continue
                    if candidate in evidence:continue
                    matched.append(candidate)
                    evidence[candidate]=dict(outer_candidate_index=outer_candidate,
                        current_actual_RGB_nested_interface_uv=[v['uv'].tolist() for v in ray[:level+1]],
                        actual_inward_material_transition_signs=[v['sign'] for v in ray[:level+1]],
                        actual_interface_contrasts=[v['contrast'] for v in ray[:level+1]],
                        source_normal_layer_order=list(labels[:level+1]),
                        independent_source_interface_to_candidate_distance_px=float(np.linalg.norm(uv[candidate]-physical['uv'])),
                        model_projection_or_fit_residual_used_for_identity=False)
                indices=np.array(matched,int)
                if len(indices)<2:
                    decisions.append(dict(curve_id=name,actual_pts=pts,status='nested_original_material_layers_not_uniquely_resolved',source_matches=len(indices)))
                    continue
                # Current pixel support and an existing named physical house
                # jointly establish a new source anchor for this same house.
                seed=dict(known['source_seed_reference'],curve_id=name,physical_id=family,actual_pts=pts,native_pts=pts,
                    decoded_native_RGB_sha256=digest,navigation_uv=uv[indices].tolist(),
                    source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
                    authority=known['source_seed_authority'],
                    physical_identity_basis='same_actual_named_house_and_current_RGB_alternating_nested_material_sequence',
                    same_source_house_layer_identity_evidence=[evidence[i] for i in indices],
                    model_projection_or_fit_residual_used_for_identity=False,is_exact_centerline_pixel_ground_truth=False)
                temporal=dict(kind='confirmed_original_source_identity_anchor',physical_material_id=name,
                    shot_id=binding['shot_id'],current_actual_pts=pts,same_physical_entity_confirmed=True,
                    current_frame_RGB_measured=True,source_identity_confirmed=True,
                    source_identity_authority=known['source_seed_authority'],
                    source_nested_material_identity_evidence=seed['same_source_house_layer_identity_evidence'],
                    source_house_identity_reference=known['source_seed_reference'],
                    model_projection_or_fit_residual_used_for_identity=False)
                row=dict(known,curve_id=name,feature_id=family,world_curve_role=layer,physical_material_id=name,
                    actual_pts=pts,native_pts=pts,uv=uv[indices].tolist(),raw_candidate_uv=uv[indices].tolist(),
                    candidate_sides=sides[indices].tolist(),source_sides_uv=sides[indices].tolist(),
                    source_candidate_indices=indices.tolist(),original_candidate_indices=indices.tolist(),
                    source_query_indices=queries[indices].tolist(),source_seed_reference=seed,temporal_evidence=temporal,
                    source_nested_material_identity_evidence=seed['same_source_house_layer_identity_evidence'],
                    admission_status='admitted_source_physical_boundary_temporal_identity',
                    physical_edge_role='confirmed_material_interface',center_definition=None)
                row.pop('source_readout_uncertainty',None);row.pop('sigma_px',None)
                row.pop('candidate_local_source_clearance',None)
                row.pop('original_source_exclusion_flag_was_cleared_only_by_explicit_local_material_proof',None)
                row['continuity_evidence']=dict(current_actual_pts=pts,
                    current_original_RGB_named_house_nested_material_layers_confirmed=True,
                    physical_identity_from_model_projection=False,
                    current_nested_material_identity_evidence=seed['same_source_house_layer_identity_evidence'])
                if known['source_seed_reference'].get('source_identity_navigation_evidence') is not None:
                    from .vision_ice_model_context_identity import attach_grid_context_evidence
                    seed.pop('model_projection_or_fit_residual_used_for_identity',None)
                    attach_grid_context_evidence(row,seed,source_binding=dict(source_sha256=self.source_sha256,
                        time_base=self.time_base,actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),
                        decoded_native_BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),shot_id=binding['shot_id']),
                        source_motion=row['continuity_evidence'])
                    row['continuity_evidence'].pop('physical_identity_from_model_projection',None)
                    row['source_identity_navigation_evidence']['current_actual_source_nested_material_structure']=seed['same_source_house_layer_identity_evidence']
                clearance=measurement.get('candidate_local_source_clearance',{})
                proofs={int(i):clearance[int(i)] for i in indices if int(i) in clearance}
                if proofs:
                    row['candidate_local_source_clearance']=proofs
                    row['original_source_exclusion_flag_was_cleared_only_by_explicit_local_material_proof']=True
                uncertainty=_current_readout_uncertainty(measurement,indices,uv[indices],normals[indices])
                if uncertainty is not None:
                    row['source_readout_uncertainty']=uncertainty
                    row['sigma_px']=np.max(np.abs(uncertainty['normal_interval_px']),axis=1).tolist()
                rows.append(row);already.add(name)
                decisions.append(dict(curve_id=name,actual_pts=pts,status=row['admission_status'],source_matches=len(indices)))
        return dict(rows=rows,decisions=decisions,visual_review_complete=False,
            source_nested_material_identity_used=True,model_projection_used_for_identity=False)
