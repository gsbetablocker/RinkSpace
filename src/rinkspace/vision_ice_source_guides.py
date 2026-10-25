"""Additional RGB searches guided by actually reviewed source paint locations.

The frozen camera queries can miss a clearly visible physical marker. A source
guide supplies independent search locations, never observed UVs or fit geometry.
Appended candidates retain complete native profiles, source exclusion and all
competing interfaces, and remain pending until the identity selector admits them.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .vision_programme_ice_review import (
    _source_queries,_batched_actual_profiles,_actual_profile_candidates,
    _candidate_context_excluded,STATUS)
from .vision_source_shapes import _MaterialProfileFrame


def append_actual_source_guided_queries(rgb,measurement,exclusion_mask,inventory,guides,*,profile_backend='cpu'):
    """Append source-profile readouts without replacing frozen queries.

    Each guide requires curve_id, navigation_uv, actual_pts, native_pts,
    shot_id, source_sha256, time_base, both decoded_native_RGB/BGR_sha256,
    authority {path, sha256}, original_source_image_actually_reviewed=True,
    physical_identity_from_model_projection=False. Its polyline is navigation
    only. Optional source_interior_navigation_uv orients named disc interfaces
    using a visible source interior; it is never a world/model circle center.

    Sampling spacing and normal_search_half_extent_px affect only source search.
    Restore the original frozen query prefix to its exact reader float64
    precision before appending (restore_exact_navigation_queries). Appending
    must not silently declare a float32 cache prefix to be an exact query.
    The caller must preserve appended arrays/indices as a measurement overlay.
    No seed identity, surface membership or exclusion clearance is granted here.
    """
    guides=list(guides)
    if not guides:return dict(measurement)
    if any(np.asarray(measurement[name]).dtype!=np.dtype(np.float64) for name in ('query_uv','query_normal','query_interval')):
        raise ValueError('Restore the exact frozen float64 source-navigation query prefix before appending a guide')
    from calibration.program_tracking.strict_model_scope import authenticate_source_reference
    binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
    digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
    bgr=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest()
    if measurement['decoded_native_RGB_sha256']!=digest:raise ValueError('Source guides need the exact measured original RGB')
    names={v['curve_id']:i for i,v in enumerate(inventory)}
    mask=np.asarray(exclusion_mask,bool)
    if mask.shape!=np.asarray(rgb).shape[:2]:raise ValueError('Source guide exclusions must match the original native pixels')
    oldq=len(measurement['query_curve'])
    q_uv=[];q_normal=[];q_interval=[];q_curve=[];q_guide=[];references=[]
    for gi,guide in enumerate(guides):
        if (guide.get('actual_pts')!=pts or guide.get('native_pts')!=int(binding.get('native_pts',pts))
                or guide.get('shot_id')!=binding['shot_id'] or guide.get('source_sha256')!=binding['source_sha256']
                or str(guide.get('time_base'))!=str(binding['time_base'])
                or guide.get('decoded_native_RGB_sha256')!=digest or guide.get('decoded_native_BGR_sha256')!=bgr
                or guide.get('original_source_image_actually_reviewed') is not True
                or guide.get('physical_identity_from_model_projection') is not False):
            raise ValueError('A source guide requires its own reviewed original RGB clock and identity basis')
        authenticate_source_reference(guide['authority'])
        if guide['curve_id'] not in names:raise ValueError('A source guide cannot add an unmodeled physical feature')
        polyline=np.asarray(guide['navigation_uv'],np.float64)
        if polyline.ndim!=2 or polyline.shape[1]!=2 or len(polyline)<2 or not np.isfinite(polyline).all():
            raise ValueError('A source guide needs a finite original-image navigation polyline')
        extent=float(guide.get('normal_search_half_extent_px',128.));spacing=float(guide.get('sampling_spacing_px',8.))
        if extent<10 or spacing<=0:raise ValueError('Source guide search context and sampling must be positive')
        # Reuse only the public image-polyline sampling/clipping arithmetic.
        # No scene, camera, world depth or projected model center is accessed.
        points,normals,intervals=_source_queries({},polyline,np.ones(len(polyline)),rgb.shape[:2],spacing,extent)
        ci=names[guide['curve_id']];definition=inventory[ci]
        interior=guide.get('source_interior_navigation_uv')
        if definition['operator']=='material_interface':
            if interior is None:raise ValueError('Named disc interfaces need an actually reviewed source interior direction')
            interior=np.asarray(interior,np.float64)
            if interior.shape!=(2,) or not np.isfinite(interior).all():raise ValueError('Finite visible-source interior guide required')
            reverse=np.einsum('ij,ij->i',points-interior,normals)<0
            normals[reverse]*=-1;intervals[reverse]=-intervals[reverse,::-1]
        first=len(q_uv);q_uv.extend(points);q_normal.extend(normals);q_interval.extend(intervals)
        q_curve.extend([ci]*len(points));q_guide.extend([gi]*len(points))
        references.append(dict(curve_id=guide['curve_id'],actual_pts=pts,authority=guide['authority'],
            original_source_navigation_polyline=polyline.tolist(),first_appended_query=oldq+first,appended_query_count=len(points),
            model_projection_used_for_navigation=False,navigation_UV_used_as_measurement=False))
    points=np.asarray(q_uv,np.float64).reshape(-1,2);normals=np.asarray(q_normal,np.float64).reshape(-1,2)
    intervals=np.asarray(q_interval,np.float64).reshape(-1,2);curves=np.asarray(q_curve,np.int16)
    return _append_guided_readouts(rgb,measurement,mask,inventory,points,normals,intervals,curves,q_guide,references,
        guides=guides,profile_backend=profile_backend)


def remeasure_actual_source_guided_queries(rgb,measurement,exclusion_mask,inventory,source_guided_queries,*,profile_backend='cpu'):
    """Read the same guide stations in another native presentation's own RGB.

    Canonical source guides are reused solely as search navigation. Each alias
    receives fresh candidates and gradient quantiles from its own decoded RGB;
    no canonical observed UV, visibility, clearance or identity is copied.
    """
    from calibration.program_tracking.strict_model_scope import authenticate_source_reference
    receipt=source_guided_queries;binding=measurement['source_binding'];old=receipt['source_binding']
    if receipt.get('schema')!='rinkspace.original-source-guided-query-overlay.v1':raise ValueError('Unknown source guide receipt')
    for key in ('source_sha256','time_base','shot_id'):
        if str(binding[key])!=str(old[key]):raise ValueError('Source guide stations belong to another source shot/clock')
    if int(binding.get('actual_pts',binding['source_pts']))!=int(old.get('actual_pts',old['source_pts'])):
        raise ValueError('Canonical navigation stations can only be remeasured for the same actual exposure')
    if len(measurement['query_curve'])!=receipt['original_query_prefix_length']:
        raise ValueError('Alias original-query prefix differs from the saved guide overlay')
    for guide in receipt['guides']:authenticate_source_reference(guide['authority'])
    points=np.asarray(receipt['query_uv'],np.float64).reshape(-1,2)
    normals=np.asarray(receipt['query_normal'],np.float64).reshape(-1,2)
    intervals=np.asarray(receipt['query_interval'],np.float64).reshape(-1,2)
    curves=np.asarray(receipt['query_curve'],np.int16)
    if any(inventory[int(ci)]['curve_id']!=name for ci,name in zip(curves,receipt['query_curve_ids'])):
        raise ValueError('Alias guide inventory physical identities differ')
    result=_append_guided_readouts(rgb,measurement,np.asarray(exclusion_mask,bool),inventory,points,normals,intervals,curves,
        receipt['query_guide_indices'],receipt['guide_evidence'],guides=receipt['guides'],profile_backend=profile_backend)
    result['source_guided_queries']['canonical_navigation_source_binding']=old
    result['source_guided_queries']['own_native_RGB_profiles_read']=True
    result['source_guided_queries']['canonical_observed_UV_copied']=False
    return result


class ActualIceSourceGuideTracker:
    """Actual-material LK navigation for subsequent guide-profile searches.

    ``consume`` runs before selection and only emits pending RGB candidates.
    ``remember_selected_source_rows`` runs after actual source identity gates.
    It retains one source image and confirmed material sides, not a frame bank.
    Canonical exposures alone advance this state; aliases use the receipt reader.
    """
    def __init__(self,inventory,*,source_sha256,time_base,profile_backend='cpu'):
        self.inventory=[dict(v) for v in inventory]
        self.source_sha256,self.time_base=source_sha256,str(time_base)
        self.backend=profile_backend;self.previous=None;self.current=None;self.supports=[]

    def _source_support_profile_half_extent_px(self,support):
        """Original search context; explicit successors may use measured bands."""
        return 128.

    def consume(self,rgb,measurement,exclusion_mask,*,source_ice_scope_mask,guides=()):
        from .vision_ice_identity_graph import _point_source_flow
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        if (measurement['decoded_native_RGB_sha256']!=digest or binding['source_sha256']!=self.source_sha256
                or str(binding['time_base'])!=self.time_base):raise ValueError('Source-guide tracking needs this exact original RGB clock')
        if any(np.asarray(measurement[name]).dtype!=np.dtype(np.float64) for name in ('query_uv','query_normal','query_interval')):
            raise ValueError('Restore exact frozen source-navigation prefix before source-guide tracking')
        mask=np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if self.previous is not None and self.previous['binding']['shot_id']!=binding['shot_id']:
            self.previous=None;self.supports=[]
        if self.previous is not None and pts<=self.previous['pts']:raise ValueError('Source-guide tracker advances only ordered canonical exposures')
        result=append_actual_source_guided_queries(rgb,measurement,mask,self.inventory,guides,profile_backend=self.backend) if guides else dict(measurement)
        decisions=[];points=[];normals=[];intervals=[];curves=[];refs=[];ids=[];tracked_guides=[]
        if self.previous is not None:
            for support in self.supports:
                support_pts=support.get('actual_pts',self.previous['pts'])
                if support_pts==self.previous['pts']:
                    predicted,good,errors=_point_source_flow(self.previous['gray'],gray,support,self.previous['mask'],mask)
                else:
                    predicted=np.asarray(support['uv'],float);good=np.zeros(len(predicted),bool);errors=np.full(len(predicted),np.nan)
                selected=np.flatnonzero(good)
                measured_motion=len(selected)>=2
                if len(selected)<2:
                    # An aperture-limited thin stripe can have no reliable 2D
                    # LK correspondence while it remains visible. Keep its
                    # last actual source positions as search navigation only.
                    # Current RGB profiles, unique material matches and the
                    # selector's width/context gates still decide admission.
                    # Never flow stale coordinates in the previous image.
                    selected=np.arange(len(support['uv']));predicted=np.asarray(support['uv'],float)
                    decisions.append(dict(curve_id=support['curve_id'],actual_pts=pts,
                        status='last_confirmed_source_material_navigation_current_RGB_reread_required',
                        last_confirmed_actual_pts=support_pts,source_identity_admitted_by_guide=False))
                # Actual LK supplies normal navigation only. Every output UV
                # still comes from both fresh profile sides or a fresh q50 edge.
                chosen=predicted[selected];normal=support['normal'][selected]
                extent=float(self._source_support_profile_half_extent_px(support))
                if not np.isfinite(extent) or extent<10.:raise ValueError('Complete source guide search context required')
                window=np.tile([-extent,extent],(len(chosen),1))
                for axis,limit in ((0,rgb.shape[1]-1),(1,rgb.shape[0]-1)):
                    moving=np.abs(normal[:,axis])>1e-12
                    a=np.divide(4-chosen[:,axis],normal[:,axis],out=np.full(len(chosen),-np.inf),where=moving)
                    b=np.divide(limit-4-chosen[:,axis],normal[:,axis],out=np.full(len(chosen),np.inf),where=moving)
                    window[:,0]=np.maximum(window[:,0],np.minimum(a,b));window[:,1]=np.minimum(window[:,1],np.maximum(a,b))
                enough=window[:,1]-window[:,0]>=20.;chosen,normal,window=chosen[enough],normal[enough],window[enough]
                if len(chosen)<2:continue
                gi=len(refs);first=len(result['query_curve'])+len(points)
                points.extend(chosen);normals.extend(normal);intervals.extend(window);curves.extend([support['curve_index']]*len(chosen));ids.extend([gi]*len(chosen))
                proof=dict(curve_id=support['curve_id'],actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),
                    shot_id=binding['shot_id'],source_sha256=self.source_sha256,time_base=self.time_base,
                    decoded_native_RGB_sha256=digest,decoded_native_BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),
                    authority=support['authority'],original_source_image_actually_reviewed=False,
                    physical_identity_from_model_projection=False,navigation_UV_used_as_measurement=False,
                    guide_origin='confirmed_material_actual_RGB_LK_navigation_only' if measured_motion else 'last_confirmed_actual_source_material_search_navigation_only',
                    previous_actual_pts=support_pts,previous_decoded_BGR_sha256=support.get('decoded_BGR_sha256',self.previous['BGR_sha256']),
                    current_RGB_tracking_confirmed_by_guide=measured_motion,
                    source_forward_backward_LK_error_px=errors[selected][enough].tolist() if measured_motion else [])
                tracked_guides.append(proof)
                refs.append(dict(proof,first_appended_query=first,appended_query_count=len(chosen),
                    source_identity_admitted_by_guide=False))
                decisions.append(dict(curve_id=support['curve_id'],actual_pts=pts,
                    status='actual_material_LK_navigation_and_current_RGB_profiles' if measured_motion else 'last_source_material_navigation_and_current_RGB_profiles',
                    source_queries=len(chosen),source_identity_admitted_by_guide=False))
        if points:
            result=_append_guided_readouts(rgb,result,mask,self.inventory,np.asarray(points,float),np.asarray(normals,float),
                np.asarray(intervals,float),np.asarray(curves,np.int16),ids,refs,guides=tracked_guides,profile_backend=self.backend)
        self.current=dict(pts=pts,binding=dict(binding),RGB_sha256=digest,BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),
            gray=gray,mask=mask,measurement=result,
            readout_prefix={key:np.asarray(result[key]) for key in ('query_curve','query_uv','query_normal','query_interval',
                'candidate_uv','candidate_sides','candidate_query')})
        return dict(measurement=result,decisions=decisions,model_projection_used_as_measurements=False)

    def remember_selected_source_rows(self,rows,measurement=None):
        """Only current measured/static/admitted source rows become LK support."""
        from calibration.program_tracking.strict_absolute_source import _temporal_identity
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        if self.current is None:raise ValueError('Remember only the just-consumed native source exposure')
        selected=rows.get('rows',()) if isinstance(rows,dict) else rows;supports=[]
        original=self.current['measurement'];binding=self.current['binding']
        if measurement is None:measurement=original
        else:
            # Later source readers may append visible nested-material queries.
            # Keep their exact current readouts while verifying that the guide
            # reader's existing source prefix was neither replaced nor moved.
            final_binding=measurement['source_binding']
            for key in ('source_sha256','time_base','source_pts','actual_pts','native_pts','shot_id',
                        'decoded_native_BGR_sha256'):
                if final_binding.get(key)!=binding.get(key):
                    raise ValueError('Final source-guide overlay has a different native source clock or BGR')
            if measurement.get('decoded_native_RGB_sha256')!=self.current['RGB_sha256']:
                raise ValueError('Final source-guide overlay has a different original RGB')
            for key in ('query_curve','query_uv','query_normal','query_interval',
                        'candidate_uv','candidate_sides','candidate_query'):
                before=self.current['readout_prefix'][key];after=np.asarray(measurement[key])
                if len(after)<len(before) or not np.array_equal(after[:len(before)],before,equal_nan=True):
                    raise ValueError('Final source-guide overlay changed an existing actual source readout prefix')
        names={v['curve_id']:i for i,v in enumerate(self.inventory)}
        for row in selected:
            from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context
            if (row.get('actual_pts')!=self.current['pts'] or row.get('decoded_native_RGB_sha256')!=self.current['RGB_sha256']
                    or row.get('actual_source_RGB_measured') is not True or row.get('static_material_confirmed') is not True
                    or row.get('pixel_location_confirmed') is not True or not admissible_source_identity_context(row)
                    or row.get('admission_status') not in ('admitted_source_material_center_temporal_identity','admitted_source_physical_boundary_temporal_identity')
                    or row.get('curve_id') not in names or not _temporal_identity(row)):continue
            authenticate_source_reference(row['source_seed_authority'])
            idx=np.asarray(row['original_candidate_indices'],int)
            if len(idx)<2:continue
            if np.any(idx<0) or np.any(idx>=len(measurement['candidate_uv'])):
                raise ValueError('Source-guide row indices do not belong to the final current source overlay')
            ci=names[row['curve_id']]
            from .vision_ice_temporal_identity import exact_selected_source_readouts
            uv,sides=exact_selected_source_readouts(row,measurement,operator=self.inventory[ci]['operator'])
            width=np.linalg.norm(sides[:,1]-sides[:,0],axis=1);normal=np.zeros_like(uv)
            paired=width>1e-6;normal[paired]=(sides[paired,1]-sides[paired,0])/width[paired,None]
            # A material-interface normal comes from its observed local source
            # arc, rather than a projected camera/world circle center.
            tangent=np.gradient(uv,axis=0);source_normal=np.c_[-tangent[:,1],tangent[:,0]]
            lengths=np.linalg.norm(source_normal,axis=1);nonzero=lengths>1e-6
            normal[~paired&nonzero]=source_normal[~paired&nonzero]/lengths[~paired&nonzero,None]
            valid=paired|nonzero
            if valid.sum()<2:continue
            positions=np.flatnonzero(valid);cells=np.rint(uv[positions]).astype(np.int64)
            # Several measured queries may land on the same native material
            # pixel. Reuse one of those actual supports for LK navigation;
            # retaining every duplicate would grow the next frame's searches
            # recursively. All measured output rows/readouts remain untouched.
            _,unique=np.unique(cells,axis=0,return_index=True);positions=positions[np.sort(unique)]
            supports.append(dict(uv=uv[positions],sides=sides[positions],normal=normal[positions],curve_id=row['curve_id'],curve_index=ci,
                authority=row['source_seed_authority'],actual_pts=self.current['pts'],decoded_BGR_sha256=self.current['BGR_sha256']))
        admitted_names={v['curve_id'] for v in supports}
        supports.extend(v for v in self.supports if v['curve_id'] not in admitted_names)
        self.supports=supports;self.previous={k:v for k,v in self.current.items() if k not in ('measurement','readout_prefix')};self.current=None


def _append_guided_readouts(rgb,measurement,mask,inventory,points,normals,intervals,curves,q_guide,references,*,guides,profile_backend):
    result=dict(measurement);oldq=len(measurement['query_curve']);oldc=len(measurement['candidate_uv'])
    binding=measurement['source_binding'];digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
    if measurement['decoded_native_RGB_sha256']!=digest:raise ValueError('Guide overlay must read this native presentation RGB')
    if mask.shape!=np.asarray(rgb).shape[:2]:raise ValueError('Native guide profile mask dimensions differ')
    statuses=np.full(len(points),STATUS['physical_role_unresolved'],np.uint8)
    added={key:[] for key in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv')}
    if len(points):
        frame=_MaterialProfileFrame(rgb,binding,digest)
        signals=np.asarray([inventory[int(ci)]['material_signal'] for ci in curves])
        if profile_backend=='cuda':
            from .vision_source_cuda_profiles import batched_actual_profiles_cuda
            offsets,profiles,smoothed,valid=batched_actual_profiles_cuda(frame,points,normals,intervals,signals)
        elif profile_backend=='cpu':offsets,profiles,smoothed,valid=_batched_actual_profiles(frame,points,normals,intervals,signals)
        else:raise ValueError('Source guide profile backend must be cpu or cuda')
        for qi in range(len(points)):
            role=inventory[int(curves[qi])];take=valid[qi]
            operator='paint_center' if role['operator']=='unresolved' else role['operator']
            values=_actual_profile_candidates(points[qi],normals[qi],offsets[qi,take],profiles[qi,take],smoothed[qi,take],
                operator=operator,signal=role['material_signal'],transition=role.get('material_transition'),with_quantiles=True)
            clean=0
            for uv,sides,bad,quantiles in values:
                excluded=any(_candidate_context_excluded(mask,p,normals[qi]) for p in (uv,*sides))
                added['candidate_uv'].append(uv);added['candidate_sides'].append(sides)
                added['candidate_query'].append(oldq+qi);added['candidate_flags'].append(int(bad)|(int(excluded)<<1)|(int(len(values)>1)<<2)|8)
                added['candidate_side_quantiles_uv'].append(quantiles);clean+=not bad and not excluded
            if role['operator']!='unresolved':statuses[qi]=STATUS['source_material_missing'] if not values else (
                STATUS['actual_RGB_candidate_pending_review'] if clean==len(values)==1 else STATUS['competing_or_unresolved_material'])
    for name,values in [('query_curve',curves),('query_uv',points),('query_normal',normals),('query_interval',intervals),('query_status',statuses)]:
        old=result.get(name,np.zeros(oldq,np.uint8) if name=='query_status' else None)
        result[name]=np.concatenate([np.asarray(old),values])
    shapes={'candidate_uv':(-1,2),'candidate_sides':(-1,2,2),'candidate_side_quantiles_uv':(-1,2,3,2),'candidate_query':(-1,),'candidate_flags':(-1,)}
    for name,items in added.items():
        old=result.get(name,np.full((oldc,2,3,2),np.nan))
        dtype=np.float64 if name=='candidate_side_quantiles_uv' else np.asarray(old).dtype
        result[name]=np.concatenate([old,np.asarray(items,dtype=dtype).reshape(shapes[name])])
    previous_receipt=measurement.get('source_guided_queries')
    previous_guides=previous_receipt['guides'] if previous_receipt else []
    previous_references=previous_receipt['guide_evidence'] if previous_receipt else []
    guide_index=np.asarray(q_guide,np.int32)+len(previous_guides)
    result['query_source_guide_indices']=np.r_[measurement.get('query_source_guide_indices',np.full(oldq,-1,np.int32)),guide_index]
    result['source_guided_query_evidence']=[*previous_references,*references]
    def combined(name,value):
        if previous_receipt is None:return np.asarray(value).tolist()
        value=np.asarray(value)
        before=np.asarray(previous_receipt[name],dtype=value.dtype).reshape((-1,*value.shape[1:]))
        return np.concatenate([before,value]).tolist()
    original_q=previous_receipt['original_query_prefix_length'] if previous_receipt else oldq
    original_c=previous_receipt['original_candidate_prefix_length'] if previous_receipt else oldc
    result['source_guided_queries']=dict(schema='rinkspace.original-source-guided-query-overlay.v1',source_binding=dict(binding),
        decoded_native_RGB_sha256=digest,decoded_native_BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),
        original_query_prefix_length=original_q,original_candidate_prefix_length=original_c,guides=[*previous_guides,*list(guides)],
        guide_evidence=[*previous_references,*references],query_uv=combined('query_uv',points),query_normal=combined('query_normal',normals),
        query_interval=combined('query_interval',intervals),query_curve=combined('query_curve',curves),query_status=combined('query_status',statuses),
        query_curve_ids=[*(previous_receipt['query_curve_ids'] if previous_receipt else []),*[inventory[int(ci)]['curve_id'] for ci in curves]],
        query_guide_indices=[*(previous_receipt['query_guide_indices'] if previous_receipt else []),*guide_index.tolist()],
        navigation_UV_used_as_measurement=False,model_projection_used_for_navigation=False,
        appended_actual_candidate_count=len(result['candidate_uv'])-original_c)
    if 'curve_status' in measurement:
        status=np.asarray(measurement['curve_status']).copy()
        for ci in np.unique(curves):
            if inventory[int(ci)]['operator']=='unresolved':continue
            status[int(ci)]=max(status[int(ci)],int(statuses[curves==ci].max()))
        result['curve_status']=status
    result.update(source_guided_actual_RGB_remeasured=True,source_guided_navigation_UV_admitted=False,
        source_readout_quantization_step_px=1.,source_interpolation_grid_px=1./32.)
    return result


def read_actual_source_query_profiles(rgb, binding, inventory, points, normals, intervals, curves,
        exclusion_mask, *, query_guide_indices=None, profile_backend='cpu'):
    """Public adapter for native-bound image queries; reuse the existing q50 reader.

    The inventory supplies measurement semantics only. Queries are current image
    navigation, never observations. No old scene or camera is loaded here.
    """
    points=np.asarray(points,np.float64).reshape(-1,2)
    normals=np.asarray(normals,np.float64).reshape(-1,2)
    intervals=np.asarray(intervals,np.float64).reshape(-1,2)
    curves=np.asarray(curves,np.int16)
    if (normals.shape!=points.shape or intervals.shape!=points.shape or curves.shape!=(len(points),)
            or not np.isfinite(np.r_[points.ravel(),normals.ravel(),intervals.ravel()]).all()
            or np.any(curves<0) or np.any(curves>=len(inventory))):
        raise ValueError('Every native source query needs a finite normal, interval and retained path')
    if len(points) and (not np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-5)
            or np.any(intervals[:,1]<=intervals[:,0])):
        raise ValueError('Unit source normals and ordered material search intervals required')
    empty=dict(source_binding=dict(binding),decoded_native_RGB_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
        query_uv=np.empty((0,2),np.float64),query_normal=np.empty((0,2),np.float64),query_interval=np.empty((0,2),np.float64),
        query_curve=np.empty(0,np.int16),query_status=np.empty(0,np.uint8),curve_status=np.zeros(len(inventory),np.uint8),
        candidate_uv=np.empty((0,2),np.float32),candidate_sides=np.empty((0,2,2),np.float32),
        candidate_query=np.empty(0,np.int32),candidate_flags=np.empty(0,np.uint8),
        candidate_side_quantiles_uv=np.empty((0,2,3,2),np.float64),
        source_measurement_method='existing_native_RGB_material_profiles_q25_q50_q75',
        source_RGB_measured=True,projected_UV_used_as_measurements=False)
    query_guide_indices=np.full(len(points),-1,np.int32) if query_guide_indices is None else np.asarray(query_guide_indices,np.int32)
    if query_guide_indices.shape!=(len(points),):raise ValueError('Source query provenance has a different length')
    return _append_guided_readouts(rgb,empty,np.asarray(exclusion_mask,bool),inventory,points,normals,intervals,curves,
        query_guide_indices,[],guides=(),profile_backend=profile_backend)
