"""Same-state grid context plus two-exposure actual RGB material identity.

The grid supplies named nearby search context and field order. It never
supplies output pixels. All candidate centers/sides are current native RGB
readouts, and a competing physical material interpretation remains pending.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .vision_ice_temporal_identity import (
    _point_polyline_distance,_curve_local_navigation,_measured_material_descriptors,
    final_current_source_track_support)
from .vision_programme_ice_review import ice_curve_inventory,_candidate_context_excluded


GRID_CONTEXT_KIND='actual_RGB_material_field_order_and_continuity_with_model_navigation'
GRID_CONTEXT_FLAGS=dict(model_projection_used_as_navigation_and_identity_context=True,
    projected_UV_used_as_source_pixel_truth=False,source_identity_not_confirmed_from_projection_alone=True,
    fit_residual_used_for_identity=False)


def _actual_paint_colour_context(rgb,points,sides,normals,signal,*,native_rgb_field=None):
    """Current raster samples of paint and both exterior ice materials."""
    tangent=np.c_[-normals[:,1],normals[:,0]]
    positions=np.stack([sides[:,0]-3.*normals,points,sides[:,1]+3.*normals],axis=1)
    samples=positions[:,:,None,:]+tangent[:,None,None,:]*np.arange(-2.,3.)[None,None,:,None]
    xy=np.rint(samples.astype(np.float32)*32.)/32.
    field=rgb.astype(np.float32) if native_rgb_field is None else native_rgb_field
    values=cv2.remap(field,xy[...,0].reshape(len(points),-1),
        xy[...,1].reshape(len(points),-1),cv2.INTER_LINEAR).reshape(len(points),3,5,3)
    colours=np.median(values,axis=2).astype(np.float64)
    if signal=='red':field=colours[:,:,0]-colours[:,:,1];contrast=field[:,1,None]-field[:,[0,2]]
    elif signal=='blue':field=colours[:,:,2]-colours[:,:,0];contrast=field[:,1,None]-field[:,[0,2]]
    else:
        field=colours@np.array([.299,.587,.114]);contrast=field[:,[0,2]]-field[:,1,None]
    # Two independent flanks must display the modeled paint colour. This is
    # actual contrast support, not inferred static material from strength>0.
    readable=(contrast>.5).all(1)
    return readable,colours,contrast,positions


def actual_paint_path_coherence(measurement,indices,*,maximum_source_neighbor_gap_px=80.):
    """Hold local jumps in measured paint sides; never synthesize a center.

    Native source neighbors support a local linear navigation check. Their
    physical stripe widths are checked for an entity switch, while actual
    gradient blur and sampling quantify the center readout comparison. No
    world path, projected UV or camera fit enters this check.
    """
    indices=np.asarray(indices,int);sides=np.asarray(measurement['candidate_sides'],np.float64)[indices]
    points=sides.mean(1);keep=np.ones(len(indices),bool);decisions=[]
    if len(indices)<5:return keep,decisions
    tangent=np.linalg.svd(points-points.mean(0),full_matrices=False)[2][0]
    normal=np.array([-tangent[1],tangent[0]]);station=points@tangent
    widths=np.linalg.norm(sides[:,1]-sides[:,0],axis=1)
    quantiles=measurement.get('candidate_side_quantiles_uv');blur=np.full(len(indices),np.nan)
    step=float(measurement.get('source_readout_quantization_step_px',1.))
    if quantiles is not None:
        q=np.asarray(quantiles,np.float64)[indices]
        available=np.isfinite(q).all((1,2,3))
        centers=q.mean(1)
        support=np.abs(np.einsum('nki,i->nk',centers[:,[0,2]]-points[:,None],normal))
        blur[available]=support[available].max(1)
    for i in range(len(indices)):
        delta=station-station[i]
        neighbors=np.flatnonzero((np.abs(delta)>3.)&(np.abs(delta)<=maximum_source_neighbor_gap_px))
        # Avoid navigating through a missing source stretch. If both sides
        # are visible, use both; a one-sided endpoint needs three neighbors.
        left=neighbors[delta[neighbors]<0];right=neighbors[delta[neighbors]>0]
        if len(left) and len(right):
            neighbors=np.r_[left[np.argsort(np.abs(delta[left]))[:3]],right[np.argsort(np.abs(delta[right]))[:3]]]
        else:neighbors=neighbors[np.argsort(np.abs(delta[neighbors]))[:4]]
        if len(neighbors)<3:continue
        x=delta[neighbors];design=np.c_[np.ones(len(x)),x]
        observed=points[neighbors]@normal
        beta=np.linalg.lstsq(design,observed,rcond=None)[0]
        # Robustly exclude a competing neighbor texture from navigation;
        # the retained source measurements themselves are never altered.
        residual=observed-design@beta;scale=1.4826*np.median(np.abs(residual-np.median(residual)))
        coherent=np.abs(residual-np.median(residual))<=max(1.,3.*scale)
        if coherent.sum()<3:continue
        neighbors=neighbors[coherent];design=design[coherent]
        beta=np.linalg.lstsq(design,observed[coherent],rcond=None)[0]
        current_error=abs(float(points[i]@normal-beta[0]))
        actual_local_variation=float(np.max(np.abs(observed[coherent]-design@beta)))
        supported_blur=blur[neighbors][np.isfinite(blur[neighbors])]
        boundary_support=float(np.median(supported_blur)) if len(supported_blur) else step*.5
        center_limit=step+boundary_support+actual_local_variation
        ratio=widths[i]/max(float(np.median(widths[neighbors])),1e-9)
        normal_jump=current_error>center_limit
        material_width_switch=ratio<.65 or ratio>1.55
        if normal_jump or material_width_switch:
            keep[i]=False;decisions.append(dict(original_candidate_index=int(indices[i]),
                status='current_actual_paint_side_path_discontinuity_requires_source_reread',
                observed_UV=points[i].tolist(),observed_two_sides_UV=sides[i].tolist(),
                actual_source_neighbor_candidate_indices=indices[neighbors].tolist(),
                source_normal_step_px=current_error,actual_neighbor_center_support_px=center_limit,
                current_source_width_to_neighbor_width_ratio=float(ratio),
                center_support_uses_paint_width=False,source_neighbor_navigation_used_as_output_UV=False,
                model_projection_or_fit_residual_used_for_identity=False))
    return keep,decisions


def source_paint_pixel_accuracy_check(rgb,measurement,indices,definition,*,check_path=True,native_rgb_field=None):
    """Current native paint readout check usable without temporal state.

    Returns ``accepted_candidate_indices`` and ``pending``. Neither flags nor
    candidate UVs are changed. Where available, each candidate's actual q50
    sides provide the check coordinates; raw saved sides are used explicitly
    otherwise. Aliases must pass their own RGB and own native measurement.
    """
    binding=measurement['source_binding'];rgb=np.asarray(rgb);indices=np.asarray(indices,int)
    if (hashlib.sha256(rgb.tobytes()).hexdigest()!=measurement['decoded_native_RGB_sha256']
            or binding.get('decoded_native_BGR_sha256',hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest())
                !=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()):
        raise ValueError('Paint accuracy check requires the exact own-native original RGB')
    if definition['operator']!='paint_center':return dict(accepted_candidate_indices=indices,pending=[],pixel_values_changed=False)
    sides=np.asarray(measurement['candidate_sides'],np.float64).copy();fine=measurement.get('candidate_side_quantiles_uv')
    domain=np.full(len(sides),'raw_saved_candidate_readout',dtype='<U32')
    if fine is not None:
        q50=np.asarray(fine,np.float64)[:,:,1]
        available=np.isfinite(q50).all((1,2));sides[available]=q50[available];domain[available]='actual_gradient_q50_float64'
    current=sides[indices].mean(1);width=np.linalg.norm(sides[indices,1]-sides[indices,0],axis=1)
    normals=(sides[indices,1]-sides[indices,0])/np.maximum(width[:,None],1e-9)
    feature=definition['feature_id'];signal=definition.get('material_signal',
        'blue' if feature in ('B1','B2') else 'red' if feature=='C' or feature.startswith('goal_line') else 'dark')
    readable,colours,contrast,locations=_actual_paint_colour_context(rgb,current,sides[indices],normals,signal,native_rgb_field=native_rgb_field)
    pending=[dict(original_candidate_index=int(indices[i]),status='actual_paint_and_two_exterior_RGB_material_support_missing',
        actual_readout_coordinate_domain=str(domain[indices[i]]),actual_center_UV=current[i].tolist(),
        actual_paint_and_two_exterior_RGB=colours[i].tolist(),actual_contrast_to_each_exterior_material=contrast[i].tolist())
        for i in np.flatnonzero(~readable)]
    accepted=indices[readable]
    if check_path and feature in ('C','B1','B2','goal_line_left','goal_line_right'):
        current_measurement=dict(measurement,candidate_sides=sides)
        keep,held=actual_paint_path_coherence(current_measurement,accepted)
        for record in held:record['actual_readout_coordinate_domain']=str(domain[record['original_candidate_index']])
        pending.extend(held);accepted=accepted[keep]
    return dict(accepted_candidate_indices=accepted,pending=pending,
        actual_readout_coordinate_domains=domain[indices].tolist(),candidate_flags_changed=False,
        own_native_source_binding=dict(binding),pixel_values_changed=False,source_clock_changed=False)


def _actual_curved_paint_structure(points,query_indices):
    """A partial nearly straight coloured streak does not prove a red circle."""
    order=np.argsort(query_indices,kind='stable');points=np.asarray(points)[order]
    if len(points)<7:return False,dict(status='actual_curved_stroke_extent_missing')
    tangents=[];support=[]
    for i in range(2,len(points)-2):
        neighborhood=points[i-2:i+3];steps=np.linalg.norm(np.diff(neighborhood,axis=0),axis=1)
        if (steps>80.).any() or (steps<2.).any():continue
        vector=neighborhood[-1]-neighborhood[0];length=np.linalg.norm(vector)
        if length<24.:continue
        direction=vector/length;normal=np.array([-direction[1],direction[0]])
        deviation=np.abs((neighborhood-neighborhood.mean(0))@normal)
        if np.median(deviation)>3.:continue
        tangents.append(np.arctan2(direction[1],direction[0]));support.append(int(i))
    if len(tangents)<4:return False,dict(status='current_RGB_curved_paint_tangent_support_missing')
    # A reversed traversal of the same straight texture is not curvature.
    # Use unoriented line tangents so a 180-degree query-order reversal cannot
    # turn an advertisement reflection into a circle's observed arc.
    angle=np.unwrap(2.*np.asarray(tangents))*.5;span=float(np.ptp(angle))
    return span>=np.deg2rad(35.),dict(status='actual_curved_paint_structure' if span>=np.deg2rad(35.) else 'current_RGB_coloured_streak_does_not_identify_circle',
        actual_source_tangent_support_indices=support,actual_source_tangent_span_rad=span,
        measured_center_coordinates_replaced_by_curve_fit=False,source_structure_used_as_exact_pixel_truth=False)


def attach_grid_context_evidence(row,seed,*,source_binding,source_motion=None):
    """Preserve the explicit context branch without a false legacy flag."""
    original=seed.get('source_identity_navigation_evidence')
    if original is None:return row
    evidence=dict(original)
    flags={key:original.get(key,value) for key,value in GRID_CONTEXT_FLAGS.items()}
    if original.get('verified_source_guides_used_as_navigation'):
        flags['verified_source_guides_used_as_navigation']=True
    evidence.update(source_binding=dict(source_binding),
        current_actual_candidate_indices=list(row['original_candidate_indices']),
        current_actual_query_indices=list(row.get('source_query_indices',())),
        current_actual_two_sides_uv=row['source_sides_uv'],
        current_actual_measured_UV_sha256=hashlib.sha256(np.ascontiguousarray(row['uv'],np.float64).tobytes()).hexdigest(),
        current_feature_semantics_remeasured=True,source_identity_confirmed=True,**flags)
    temporal=row['temporal_evidence']
    if temporal['kind']=='confirmed_original_source_identity_anchor':
        temporal['kind']=original.get('kind',GRID_CONTEXT_KIND)
        prior=original['previous_source_binding']
        temporal.update(previous_actual_pts=prior['actual_pts'],previous_shot_id=prior['shot_id'],
            previous_decoded_BGR_sha256=prior['decoded_native_BGR_sha256'],previous_frame_RGB_measured=True,
            source_RGB_tracking_confirmed=True,current_feature_semantics_remeasured=True,source_identity_confirmed=True)
    else:
        evidence['previous_source_binding']=dict(source_sha256=source_binding['source_sha256'],time_base=source_binding['time_base'],
            actual_pts=temporal['previous_actual_pts'],shot_id=temporal['previous_shot_id'],
            decoded_native_BGR_sha256=temporal['previous_decoded_BGR_sha256'])
    if source_motion is not None:evidence['current_actual_material_continuity_evidence']=source_motion
    row.pop('model_projection_or_fit_residual_used_for_identity',None)
    temporal.pop('model_projection_or_fit_residual_used_for_identity',None)
    row.update(flags,source_identity_navigation_evidence=evidence)
    temporal.update(flags,source_identity_navigation_evidence=evidence)
    return row


class SameStateIceModelContextIdentity:
    """Produce source-checked identity seeds for an ordered shared RGB stream.

    ``state_identity_path`` is exactly the state shown in the current video;
    ``navigation_authority`` records the user's permission to use that grid.
    It is permission for navigation, not a claim that the user reviewed each
    candidate. Current actual correspondence is recorded separately in the
    returned seed's ``source_identity_navigation_evidence``.
    """
    def __init__(self,inventory,*,state_identity_path,navigation_authority,source_sha256,time_base,
            actual_source_negative_materials=()):
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        self.inventory=[dict(v) for v in inventory];self.source_sha256=source_sha256
        self.time_base=str(Fraction(time_base));path=Path(state_identity_path).resolve()
        self.state_reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        identity=json.loads(path.read_text(encoding='utf8'))
        scene_path=path.parent/'scene_UNACCEPTED.json'
        self.scene_reference=dict(path=str(scene_path),sha256=identity['scene_sha256'])
        self.camera_reference=dict(path=identity['camera_path'],sha256=identity['camera_sha256'])
        for reference in (self.state_reference,self.scene_reference,self.camera_reference,navigation_authority):
            authenticate_source_reference(reference)
        self.authority=dict(navigation_authority)
        self.source_negative_materials=list(actual_source_negative_materials)
        scene=json.loads(scene_path.read_text(encoding='utf8'));self.paths=ice_curve_inventory(scene)
        self.feature_kind={feature['id']:feature['kind'] for feature in scene['ice_features']}
        if [v['curve_id'] for v in self.paths]!=[v['curve_id'] for v in self.inventory]:
            raise ValueError('Same-state grid and source material inventory differ')
        with np.load(identity['camera_path'],allow_pickle=False) as source:
            self.cameras=source['camera25'];self.pts=source['pts']
            meta=json.loads(str(source['metadata_json']))
        if (meta['source_sha256']!=source_sha256 or str(meta['source_time_base'])!=self.time_base):
            raise ValueError('Model navigation camera belongs to another original source')
        self.camera_lookup={int(p):i for i,p in enumerate(self.pts)}
        self.previous_gray=self.previous_mask=self.previous_binding=None;self.previous_tracks={}

    def navigation(self,measurement):
        from calibration.program_tracking.projection import project_fast
        binding=measurement['source_binding'];native=int(binding.get('native_pts',binding['source_pts']))
        camera=self.cameras[self.camera_lookup[native]];output=[]
        for path in self.paths:
            uv,depth=project_fast(camera,path['xyz']);valid=np.isfinite(uv).all(1)&(depth>.05)
            # Keep separated valid runs; no projection across unavailable rays.
            indices=np.flatnonzero(valid);runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            output.append([uv[run] for run in runs if len(run)>=2])
        return output

    def _source_context_output_curve_allowed(self,definition):
        return True

    def _source_field_order_competitors(self,rgb,measurement,mask,ci,chosen,neighbor_identities):
        """Default physical-name competition; successor support rules stay explicit."""
        return neighbor_identities!=ci

    def _source_field_order_identity_evidence(self,measurement,ci,indices):
        return {}

    def retain_final_source_nodes(self,rgb,measurement,node_ids,*,selected):
        """Keep final material nodes while a real two-image identity is pending."""
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest()
        if (self.previous_binding is None or self.previous_binding['actual_pts']!=pts
                or self.previous_binding['shot_id']!=binding['shot_id']
                or self.previous_binding['decoded_native_BGR_sha256']!=bgr
                or measurement['decoded_native_RGB_sha256']!=digest):
            raise ValueError('Final grid-context support must use its own just-consumed RGB')
        node_ids=np.asarray(node_ids,np.int64)
        if (node_ids.ndim!=1 or len(np.unique(node_ids))!=len(node_ids)
                or np.any(node_ids<0) or np.any(node_ids>=len(measurement['candidate_uv']))):
            raise ValueError('Final graph nodes need unique exact current candidate locators')
        rows=selected.get('rows',()) if isinstance(selected,dict) else selected
        selected_ids={}
        for row in rows:selected_ids.setdefault(row['curve_id'],[]).extend(row['original_candidate_indices'])
        definitions={definition['curve_id']:definition for definition in self.inventory}
        retained={}
        for name,track in self.previous_tracks.items():
            definition=definitions[name]
            if definition['operator'] not in ('paint_center','material_interface'):continue
            allowed=node_ids if track.get('source_context_identity_pending') is True else np.asarray(selected_ids.get(name,()),np.int64)
            ids=np.asarray(track['source_candidate_indices'],np.int64)
            ids=ids[np.isin(ids,allowed)&np.isin(ids,node_ids)]
            if len(ids)<2:continue
            ci=list(definitions).index(name)
            if np.any(np.asarray(measurement['query_curve'])[np.asarray(measurement['candidate_query'])[ids]]!=ci):
                raise ValueError('Final grid-context track cannot rename another current material curve')
            if np.any(np.asarray(measurement['candidate_flags'])[ids]&3):
                raise ValueError('Final grid-context track still contains excluded source material')
            support=final_current_source_track_support(rgb,measurement,ids,operator=definition['operator'])
            if support is not None:retained[name]={**track,**support}
        self.previous_tracks=retained
        return dict(actual_pts=pts,retained_material_preparation_tracks=len(retained),
            pending_two_image_identity_tracks=sum(track['source_context_identity_pending'] for track in retained.values()),
            source_identity_created_by_filter=False)

    def consume(self,rgb,measurement,exclusion_mask,*,source_ice_scope_mask):
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        native=int(binding.get('native_pts',pts));rgb=np.asarray(rgb)
        digest=hashlib.sha256(rgb.tobytes()).hexdigest();bgr=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()
        if (binding['source_sha256']!=self.source_sha256 or str(Fraction(binding['time_base']))!=self.time_base
                or digest!=measurement['decoded_native_RGB_sha256']
                or binding.get('decoded_native_BGR_sha256',bgr)!=bgr):
            raise ValueError('Grid context requires its own exact current source RGB')
        mask=np.asarray(exclusion_mask,bool).copy();scope=np.asarray(source_ice_scope_mask,bool)
        if mask.shape!=rgb.shape[:2] or scope.shape!=mask.shape:raise ValueError('Actual current ice scope and source exclusion required')
        mask |= ~scope;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        current_binding=dict(source_sha256=self.source_sha256,time_base=self.time_base,actual_pts=pts,native_pts=native,
            decoded_native_BGR_sha256=bgr,shot_id=binding['shot_id'])
        # Known source reflections / model-external material exclusions are
        # per-native original evidence. They never become a generic inferred
        # segmentation or a copied exclusion in another exposure.
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        negative_references=[]
        for negative in self.source_negative_materials:
            if negative['source_binding']!=current_binding:continue
            authenticate_source_reference(negative['authority'])
            region=np.asarray(negative['actual_source_material_mask'],bool)
            if region.shape!=mask.shape:raise ValueError('Known negative material mask differs from its source native image')
            mask |= region;negative_references.append(negative['authority'])
        reset=self.previous_binding is None or self.previous_binding['shot_id']!=binding['shot_id']
        if not reset and pts<=self.previous_binding['actual_pts']:raise ValueError('Grid source identity must follow native exposure order')
        if reset:self.previous_tracks={}
        model=self.navigation(measurement)
        from scipy.spatial import cKDTree
        # Dense bounded model-navigation samples support a fast field-order
        # comparison. These coordinates never enter source pixel outputs.
        order_trees={};h,w=mask.shape
        for signal in {v['material_signal'] for v in self.inventory}:
            sampled=[];identities=[]
            for ci,record in enumerate(self.inventory):
                if record['material_signal']!=signal:continue
                pieces=[]
                for path in model[ci]:
                    for a,b in zip(path[:-1],path[1:]):
                        if (min(a[0],b[0])>w+24 or max(a[0],b[0])< -24
                                or min(a[1],b[1])>h+24 or max(a[1],b[1])< -24):continue
                        length=np.linalg.norm(b-a)
                        count=int(np.ceil(min(length,4.*(h+w))))+1
                        points=a+(b-a)*np.linspace(0.,1.,max(2,count))[:,None]
                        good=(points>= [-24,-24]).all(1)&(points<= [w+24,h+24]).all(1)
                        pieces.append(points[good])
                if not pieces:continue
                points=np.vstack(pieces)
                _,unique=np.unique(np.rint(points*2.),axis=0,return_index=True);points=points[unique]
                sampled.append(points);identities.extend([ci]*len(points))
            if sampled:order_trees[signal]=(cKDTree(np.vstack(sampled)),np.asarray(identities,np.int16))
        native_rgb_field=rgb.astype(np.float32);native_luminance_field=gray.astype(np.float32)
        qcurve=np.asarray(measurement['query_curve']);q=np.asarray(measurement['candidate_query'])
        raw_uv=np.asarray(measurement['candidate_uv'],float);sides=np.asarray(measurement['candidate_sides'],float)
        flags=np.asarray(measurement['candidate_flags']);normal=np.asarray(measurement['query_normal'],float)[q]
        uv=raw_uv.copy();fine=measurement.get('candidate_side_quantiles_uv')
        if fine is not None:
            fine=np.asarray(fine,float);available=np.isfinite(fine[:,:,1]).all((1,2))
            sides=sides.copy();sides[available]=fine[available,:,1]
        seeds=[];decisions=[];next_tracks={}
        for ci,definition in enumerate(self.inventory):
            name=definition['curve_id'];operator=definition['operator']
            if not self._source_context_output_curve_allowed(definition):
                decisions.append(dict(curve_id=name,actual_pts=pts,status='model_display_role_has_no_independent_source_output'));continue
            if operator=='paint_center':uv[qcurve[q]==ci]=sides[qcurve[q]==ci].mean(1)
            candidates=np.flatnonzero((qcurve[q]==ci)&((flags&3)==0))
            if not model[ci] or not len(candidates) or operator=='unresolved':
                decisions.append(dict(curve_id=name,actual_pts=pts,status='current_actual_material_or_role_unresolved'));continue
            distance=np.min(np.stack([_point_polyline_distance(uv[candidates],path) for path in model[ci]]),axis=0)
            # This is only a neighborhood used to inspect physical identity.
            # It is never a center error allowance or a fitting acceptance.
            near=candidates[distance<=24.]
            clear=[]
            for index in near:
                if any(_candidate_context_excluded(mask,p,normal[index]) for p in [uv[index],*sides[index]]):continue
                clear.append(int(index))
            near=np.asarray(clear,int)
            # Preserve competing material interpretations at a station. A
            # nearest-to-grid candidate alone never establishes an identity.
            chosen=[]
            for query in np.unique(q[near]):
                matches=near[q[near]==query]
                if len(matches)==1:chosen.append(int(matches[0]))
            chosen=np.asarray(chosen,int)
            if len(chosen)<2:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='current_material_station_competes_or_missing'));continue
            # Actual field order must distinguish other modeled materials of
            # the same appearance. Near-touching house layers stay pending.
            own=np.min(np.stack([_point_polyline_distance(uv[chosen],path) for path in model[ci]]),axis=0)
            tree,identity=order_trees[definition['material_signal']]
            distance,neighbor=tree.query(uv[chosen],k=min(128,len(identity)),workers=1)
            if distance.ndim==1:distance=distance[:,None];neighbor=neighbor[:,None]
            competing=self._source_field_order_competitors(rgb,measurement,mask,ci,chosen,identity[neighbor])
            if np.asarray(competing).shape!=distance.shape:raise ValueError('Source field-order competitor shape differs')
            other_distance=np.min(np.where(competing,distance,np.inf),axis=1)
            no_other=~np.isfinite(other_distance)
            # The farthest returned tree neighbor is a lower bound if no
            # alternative is returned; never treat a truncated query as inf.
            other_distance[no_other]=distance[no_other,-1]
            distinct=other_distance>own+2.5  # Includes half-pixel navigation discretization.
            chosen=chosen[distinct];own=own[distinct];other_distance=other_distance[distinct]
            if len(chosen)<2:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_field_material_order_ambiguous'));continue
            actual=uv[chosen];actual_sides=sides[chosen]
            width=np.linalg.norm(actual_sides[:,1]-actual_sides[:,0],axis=1)
            normals=normal[chosen].copy()
            if operator=='paint_center':normals=(actual_sides[:,1]-actual_sides[:,0])/np.maximum(width[:,None],1e-9)
            descriptor,strength=_measured_material_descriptors(rgb,actual,actual_sides,normals)
            readable=strength>0
            chosen,actual,actual_sides,width,normals,descriptor=([v[readable] for v in (chosen,actual,actual_sides,width,normals,descriptor)])
            if len(chosen)<2:continue
            raster_material=None
            if operator=='paint_center':
                material,colours,contrast,positions=_actual_paint_colour_context(rgb,actual,actual_sides,normals,definition['material_signal'],native_rgb_field=native_rgb_field)
                chosen,actual,actual_sides,width,normals,descriptor=([v[material] for v in (chosen,actual,actual_sides,width,normals,descriptor)])
                raster_material=dict(operator='paint_center',material_signal=definition['material_signal'],
                    paint_and_two_exterior_material_RGB=colours[material].tolist(),
                    actual_paint_contrast_to_each_exterior_material=contrast[material].tolist(),
                    actual_source_RGB_sample_locations=positions[material].tolist(),
                    actual_two_signed_interfaces_measured=True)
            else:
                # The visible outer interface plus actual inward material
                # order supports a named house. Occluded or blurred inner
                # layers do not invalidate an otherwise clear outer segment.
                if definition['world_curve_role']!='outer':
                    decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_house_layer_requires_complete_source_nested_sequence'));continue
                from .vision_ice_nested_material_identity import _source_nested_profile
                rgb_field=native_rgb_field;luminance=native_luminance_field;nested=[];actual_layers=[]
                for i,(point,direction) in enumerate(zip(actual,-normals)):
                    ray=_source_nested_profile(rgb,mask,point,direction,native_rgb_field=rgb_field,native_luminance_field=luminance)
                    if not ray:continue
                    nested.append(i);actual_layers.append(dict(actual_source_boundary_uv=[r['uv'].tolist() for r in ray],
                        actual_inward_transition_signs=[r['sign'] for r in ray],
                        actual_material_RGB=[r.get('preceding_material_RGB') for r in ray],
                        actual_gradient_quantiles=[r['quantiles'].tolist() for r in ray]))
                nested=np.asarray(nested,int)
                chosen,actual,actual_sides,width,normals,descriptor=([v[nested] for v in (chosen,actual,actual_sides,width,normals,descriptor)])
                raster_material=dict(operator='material_interface',actual_visible_nested_material_sequence=actual_layers,
                    current_outer_material_role_confirmed_by_actual_visible_interface_order=True,
                    all_four_layers_required_for_outer_readout=False)
            if len(chosen)<2:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_paint_two_flanks_or_nested_material_structure_missing'));continue
            if operator=='paint_center' and self.feature_kind.get(definition['feature_id'])=='circle':
                curved,structure=_actual_curved_paint_structure(actual,q[chosen])
                if not curved:
                    decisions.append(dict(curve_id=name,actual_pts=pts,status='current_circle_identity_requires_actual_curved_material_structure',
                        actual_source_structure=structure));continue
                raster_material['actual_source_curved_paint_structure']=structure
            track=dict(actual_uv=actual,candidate_sides=actual_sides,actual_pts=pts,navigation_pts=pts,
                navigation_uv=actual,material_descriptors=descriptor,source_RGB_sha256=digest,
                source_BGR_sha256=bgr,source_query_indices=q[chosen],source_candidate_indices=chosen,
                source_context_identity_pending=True)
            next_tracks[name]=track
            previous=self.previous_tracks.get(name)
            if reset or previous is None:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_material_context_read_waiting_for_next_source_exposure',
                    actual_source_readouts=len(chosen)));continue
            predicted,motion=_curve_local_navigation(self.previous_gray,gray,self.previous_mask,mask,previous)
            if motion.get('source_RGB_tracking_confirmed') is not True:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_material_RGB_continuity_requires_review',source_motion=motion));continue
            distance=_point_polyline_distance(actual,predicted)
            nearest=np.linalg.norm(actual[:,None]-np.asarray(predicted)[None],axis=2).argmin(1)
            previous_desc=np.asarray(previous['material_descriptors'])[nearest]
            correlation=np.einsum('ij,ij->i',descriptor,previous_desc)
            old_sides=np.asarray(previous['candidate_sides'])[nearest]
            old_width=np.linalg.norm(old_sides[:,1]-old_sides[:,0],axis=1)
            ratio=width/np.maximum(old_width,1e-9)
            matched=(distance<=6.)&(correlation>=.7)
            if operator=='paint_center':matched &= (ratio>=.65)&(ratio<=1.55)
            indices=chosen[matched]
            if len(indices)<2:
                decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_material_current_interface_or_entity_switch'));continue
            context_flags=getattr(self,'navigation_flags',GRID_CONTEXT_FLAGS)
            evidence=dict(kind=getattr(self,'navigation_kind',GRID_CONTEXT_KIND),source_binding=current_binding,
                previous_source_binding=dict(self.previous_binding),
                actual_material_correspondence_confirmed=True,actual_field_order_confirmed=True,
                physical_model_scope_confirmed=True,same_shot_source_continuity_confirmed=True,
                source_identity_authority=self.authority,source_model_navigation_reference=self.state_reference,
                same_scene_reference=self.scene_reference,same_native_camera_reference=self.camera_reference,
                current_actual_candidate_indices=indices.tolist(),current_actual_query_indices=q[indices].tolist(),
                current_actual_two_sides_uv=sides[indices].tolist(),
                current_actual_measured_UV_sha256=hashlib.sha256(np.ascontiguousarray(uv[indices]).tobytes()).hexdigest(),
                actual_material_local_LK=motion,current_RGB_material_context_correlations=correlation[matched].tolist(),
                actual_material_width_ratios=ratio[matched].tolist(),
                current_named_curve_distance_px=np.min(np.stack([_point_polyline_distance(uv[indices],path) for path in model[ci]]),axis=0).tolist(),
                actual_named_material_scope='ice',source_identity_confirmed=True,current_feature_semantics_remeasured=True,
                current_actual_source_material_context={key:([value[i] for i in np.flatnonzero(matched)] if isinstance(value,list) else value)
                    for key,value in raster_material.items()},
                known_actual_source_negative_material_references=negative_references,
                competing_actual_material_interpretations_rejected=True,
                field_order_test=getattr(self,'navigation_field_order_test','named_same_state_curve_neighborhood_and_distinct_actual_material_support'),
                model_neighborhood_px=24.,model_neighborhood_is_center_accuracy_tolerance=False,
                **self._source_field_order_identity_evidence(measurement,ci,indices),
                **context_flags)
            seed=dict(curve_id=name,physical_id=definition['feature_id'],actual_pts=pts,native_pts=native,
                source_sha256=self.source_sha256,time_base=self.time_base,shot_id=binding['shot_id'],
                decoded_native_RGB_sha256=digest,decoded_native_BGR_sha256=bgr,navigation_uv=uv[indices].tolist(),
                source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
                authority=self.authority,is_exact_centerline_pixel_ground_truth=False,
                source_identity_navigation_evidence=evidence,**context_flags)
            track['source_context_identity_pending']=False
            seeds.append(seed);decisions.append(dict(curve_id=name,actual_pts=pts,status='actual_RGB_model_context_material_correspondence',
                actual_source_readouts=len(indices)))
        self.previous_gray=gray;self.previous_mask=mask;self.previous_binding=current_binding;self.previous_tracks=next_tracks
        return dict(seeds=seeds,decisions=decisions,model_UV_used_as_observation=False)
