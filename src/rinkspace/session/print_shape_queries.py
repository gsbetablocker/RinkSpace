"""Sparse queries for visible shape structure before unchanged RGB profile reads.

The photographed reference supplies query navigation only. Corners, isolated
support-direction extrema and inflections describe distinct shape structure.
No regularly spaced dense arc grid is added. Nearby redundant stations are
suppressed within one path; holes and distinct paths retain their identities.
Registration, material readout, source masks and eligibility gates are inherited
or byte-for-byte equivalent to PrintPattern. No solver weights are introduced.
"""
from __future__ import annotations

import cv2
import numpy as np
from rinkspace.session.print_patterns import (
    PrintPattern, project, sample, registration_geometry_check,
    read_source_material_transition_profiles,
)

KEY_SHAPE_POLICY = dict(
    schema='rinkspace.photographed_shape_query_policy.v1',
    query_coordinates_are_navigation_only=True,
    regular_dense_arc_sampling=False,
    approximation_error_px=2.,
    minimum_corner_turn_deg=40.,
    minimum_same_path_key_arc_separation_px=12.,
    corner_side_inset_px=3.,
    support_directions=8,
    open_endpoint_inset_px=1.,
    corners_use_two_distinct_side_normals=True,
    distinct_holes_and_paths_preserved=True,
    weights_or_residuals_used=False,
)


def key_shape_stations(points, *, closed=None, horizontal_blur_px=1.):
    """Plan sparse source-image arc stations; output is never measured UV.

    Corner pairs sample the two incident material sides, rather than the
    ambiguous bisector at a sharp vertex. Flat extrema have no unique key and
    add no points. A smooth closed loop receives its isolated compass extrema;
    a straight open path receives its two separated ends. Inflection stations
    preserve a change of bend without filling an arc with neighboring pixels.
    """
    points=np.asarray(points,np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('Finite photographed polyline required for shape navigation')
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    total=float(length[-1])
    if total<4:return np.empty(0),dict(policy=KEY_SHAPE_POLICY,arc_length_px=total,keys=[],stations=[])
    if closed is None:closed=bool(np.linalg.norm(points[0]-points[-1])<1e-4)
    blur=float(horizontal_blur_px)
    if not np.isfinite(blur)or blur<0:raise ValueError('Finite actual registration blur required')
    separation=max(KEY_SHAPE_POLICY['minimum_same_path_key_arc_separation_px'],blur/3.)
    extent=float(np.linalg.norm(np.ptp(points,axis=0)))
    epsilon=max(KEY_SHAPE_POLICY['approximation_error_px'],extent*.008)
    simplified=cv2.approxPolyDP(points.astype(np.float32).reshape(-1,1,2),epsilon,bool(closed)).reshape(-1,2)
    vertices=[]
    for vertex in simplified:
        i=int(np.argmin(np.linalg.norm(points-vertex,axis=1)))
        if not vertices or i!=vertices[-1]:vertices.append(i)
    if len(vertices)>1 and vertices[0]==vertices[-1]:vertices.pop()
    # Arc separation is cyclic only for an explicitly closed photographed path.
    def distance(a,b):
        d=abs(a-b)
        return min(d,total-d)if closed else d
    corner_candidates=[];turns=[]
    for j,i in enumerate(vertices):
        if not closed and j in (0,len(vertices)-1):continue
        left=points[vertices[(j-1)%len(vertices)]];right=points[vertices[(j+1)%len(vertices)]]
        incoming=points[i]-left;outgoing=right-points[i]
        norms=np.linalg.norm(incoming)*np.linalg.norm(outgoing)
        if norms<1e-12:continue
        turn=float(np.degrees(np.arctan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0],np.dot(incoming,outgoing))))
        station=float(length[i]);turns.append((station,turn))
        if abs(turn)>=KEY_SHAPE_POLICY['minimum_corner_turn_deg']:
            inset=min(KEY_SHAPE_POLICY['corner_side_inset_px'],np.linalg.norm(incoming)/4.,np.linalg.norm(outgoing)/4.)
            corner_candidates.append(dict(kind='corner_sides',arc=station,turn_deg=turn,inset_px=float(inset),priority=abs(turn)))
    chosen=[]
    for item in sorted(corner_candidates,key=lambda c:(-c['priority'],c['arc'])):
        if all(distance(item['arc'],c['arc'])>=separation for c in chosen):chosen.append(item)
    other=[]
    if not closed:
        other.extend([dict(kind='open_endpoint',arc=1.,priority=3.),dict(kind='open_endpoint',arc=total-1.,priority=3.)])
    ordered=sorted(turns)
    for (a,ta),(b,tb)in zip(ordered,ordered[1:]):
        if abs(ta)>=10 and abs(tb)>=10 and ta*tb<0:
            other.append(dict(kind='inflection_between_bends',arc=(a+b)/2.,priority=2.))
    angles=np.arange(KEY_SHAPE_POLICY['support_directions'])*2*np.pi/KEY_SHAPE_POLICY['support_directions']
    for angle in angles:
        direction=np.array([np.cos(angle),np.sin(angle)]);projection=points@direction
        peak=float(projection.max());near=np.flatnonzero(peak-projection<=max(1e-7,extent*1e-8))
        if not len(near):continue
        arcs=length[near]
        if closed and np.ptp(arcs)>total/2:arcs=np.where(arcs<total/2,arcs+total,arcs)
        # A long flat edge is not an isolated geometric extremum.
        if float(np.ptp(arcs))>2.:continue
        station=float(arcs.mean()%total)
        other.append(dict(kind='isolated_support_extremum',arc=station,priority=1.,direction=direction.tolist()))
    for item in sorted(other,key=lambda c:(-c['priority'],c['arc'])):
        if all(distance(item['arc'],c['arc'])>=separation for c in chosen):chosen.append(item)
    # Short or degenerate paths still get their most separated source support.
    if not chosen:
        chosen=[dict(kind='short_path_endpoint',arc=1.),dict(kind='short_path_endpoint',arc=total-1.)]
    stations=[]
    for item in chosen:
        offsets=(-item['inset_px'],item['inset_px'])if item['kind']=='corner_sides'else(0.,)
        for offset in offsets:
            value=item['arc']+offset
            if closed:value%=total
            value=float(np.clip(value,1.,total-1.))
            if not any(abs(value-old)<.05 for old in stations):stations.append(value)
    stations=np.asarray(sorted(stations),np.float64)
    return stations,dict(policy=KEY_SHAPE_POLICY,closed=bool(closed),arc_length_px=total,
        approximation_error_px=epsilon,same_path_key_separation_px=separation,
        simplified_vertices=len(vertices),keys=sorted(chosen,key=lambda c:c['arc']),stations=stations.tolist())


class KeyShapePrintPattern(PrintPattern):
    """Use shape stations; retain the original complete-pattern and RGB gates."""

    def read_paths(self,image,exclusion,registration,binding,spacing=None):
        if registration['status']!='source_pattern_registered':return [],dict(reason=registration['status'])
        H=np.asarray(registration['H'])
        # Archived registration status is not a geometry certificate. Recheck
        # the finite reference at the readout boundary before sampling RGB.
        geometry=registration_geometry_check(H,np.eye(3),self.points,
            warp_roi=(self.points.min(0),self.points.max(0)))
        if not geometry['valid'] or registration.get('registration_geometry',{}).get('valid') is False:
            return [],dict(reason='current_registration_geometry_invalid',geometry=geometry)
        inverse=np.linalg.inv(H);query=[];normals=[];labels=[];key_plans=[]
        resolved,structure=self.resolved_glyphs(image,exclusion,H,
            horizontal_blur_px=registration['shared_horizontal_blur_px'])
        reference_groups=set(self.reference_path_components.values())
        resolved_groups={self.reference_path_components[name] for name in resolved}
        required_groups=min(3,len(reference_groups))
        if not required_groups or len(resolved_groups)<required_groups:
            return [],dict(reason='current_source_semantic_identity_unresolved',
                required_visible_material_groups=required_groups,
                resolved_material_groups=len(resolved_groups),structure=structure)
        exclusion=cv2.dilate(exclusion,np.ones((7,7),np.uint8))
        for p in self.paths:
            if p['path_id'] not in resolved:continue
            points=project(H,p['pixels']);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            if length[-1]<4:continue
            t,key_plan=key_shape_stations(points,closed=p.get('source_closed',p.get('closed')),horizontal_blur_px=registration['shared_horizontal_blur_px'])
            key_plans.append(dict(path_id=p['path_id'],**key_plan))
            q=np.column_stack([np.interp(t,length,points[:,k]) for k in (0,1)])
            l=np.column_stack([np.interp(t-.8,length,points[:,k]) for k in (0,1)])
            r=np.column_stack([np.interp(t+.8,length,points[:,k]) for k in (0,1)])
            n=np.c_[-(r-l)[:,1],(r-l)[:,0]];n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
            delta=sample(self.ink,project(inverse,q+n*1.25))-sample(self.ink,project(inverse,q-n*1.25))
            n[delta<0]*=-1
            if self.ink_polarity=='dark':n*=-1
            valid=(q[:,0]>5)&(q[:,0]<image.shape[1]-6)&(q[:,1]>5)&(q[:,1]<image.shape[0]-6)&(abs(delta)>2)
            q=q[valid];n=n[valid];xy=np.rint(q).astype(int);visible=exclusion[xy[:,1],xy[:,0]]==0
            query.extend(q[visible]);normals.extend(n[visible]);labels.extend([p['path_id']]*int(visible.sum()))
        if not query:return [],dict(reason='no_resolved_visible_glyphs',structure=structure)
        query=np.array(query);normals=np.array(normals)
        radius=np.maximum(4.,registration['shared_horizontal_blur_px']*abs(normals[:,0])/2+3)
        readings=read_source_material_transition_profiles(np.ascontiguousarray(image[:,:,::-1]),binding=binding,
            search_points_uv=query,normals=normals,search_intervals_px=np.c_[-radius,radius],transition='dark_to_light',
            tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
        # One exposure-wide ink/background split, not a self-justifying split of
        # a mislabeled local patch. Foreground and out-of-frame pixels are absent.
        mask=cv2.warpPerspective(self.mask,H,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        values=self.ink_signal(image)[(mask>0)&(exclusion==0)]
        if len(values)<20:return [],dict(reason='current_ink_material_unresolved')
        boundary,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        accepted=[];reasons={};signal=self.ink_signal(image).astype(np.float32)
        for i,(r,q,n,name) in enumerate(zip(readings,query,normals,labels)):
            flags=r['source_flags'];offsets=np.asarray(r['source_q10_q50_q90_normal_offsets_px'])
            reason='source_profile_unresolved' if any(flags.values()) or not np.isfinite(offsets).all() else None
            if reason is None:
                xy=q+np.linspace(offsets[0]-1,offsets[2]+1,33)[:,None]*n
                if np.any(xy<0) or np.any(xy>[image.shape[1]-1,image.shape[0]-1]):reason='profile_outside_source'
                elif np.any(sample(exclusion,xy)>0):reason='profile_foreground'
                else:
                    sides=sample(signal,q+np.array([offsets[0]-1,offsets[2]+1])[:,None]*n)
                    lower,upper=sides[::-1] if self.ink_polarity=='dark' else sides
                    if not lower<boundary<upper:reason='profile_is_not_current_ink_boundary'
            if reason:
                reasons[reason]=reasons.get(reason,0)+1;continue
            accepted.append(dict(path_id='print_instance/'+name.split('.')[0]+'::'+name,candidate=i,
                uv=np.asarray(r['actual_source_uv']).tolist(),search_uv=q.tolist(),normal=n.tolist(),actual_normal_quantiles=offsets.tolist(),
                actual_current_material_sides=sides.tolist(),current_ink_class_boundary=float(boundary),
                source_profile_flags=flags,current_source_RGB_measured=True,model_UV_used_as_measurements=False))
        return accepted,dict(queries=len(query),accepted=len(accepted),reasons=reasons,current_ink_class_boundary=float(boundary),structure=structure,query_policy=KEY_SHAPE_POLICY,key_query_plans=key_plans)
