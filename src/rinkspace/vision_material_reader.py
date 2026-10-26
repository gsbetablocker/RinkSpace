"""Wire existing source-material, temporal-identity and precision readers to native.

This module owns no world geometry or new pixel detector. Its inventory comes
from NativeSourceGeometry; measurements and continuity reuse the existing code.
"""
from __future__ import annotations
import hashlib
import numpy as np


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def _mask(value, shape):
    if value is None:return np.zeros(shape,bool)
    value=_array(value)
    if value.shape==(1,1,*shape):value=value[0,0]
    if value.shape!=tuple(shape) or value.dtype!=np.bool_:
        raise ValueError('Expected a boolean current native exclusion mask')
    return value


def _stored_source_identity(guide,binding):
    """The dispatcher authenticated a source anchor; flow can navigate it later."""
    from .vision_geometry import _same_frame
    authority=guide.get('identity_authority') or {}
    if authority.get('kind')!='authenticated_existing_source_material_identity':return False
    if not (authority.get('source_records') and authority.get('independent_identity_basis') and
            authority.get('current_native_path_id')==guide.get('native_geometry',{}).get('path_id') and
            authority.get('identity_from_projection_alone') is False):return False
    source=authority.get('source_binding') or {}
    if source.get('source_pts')==binding['source_pts']:
        _same_frame(binding,source);return True
    previous=guide.get('hint_previous_source_binding') or {}
    flow=guide.get('temporal_flow_supported')
    return bool(guide.get('origin')=='temporal_source_search_hint' and flow is not None and _array(flow).any() and
        all(source.get(k)==previous.get(k)==binding.get(k) for k in
            ('source_sha256','time_base','shot_id','event_id','rig_id','installation_id')) and
        int(source.get('source_pts',-1))<=int(previous.get('source_pts',-1))<int(binding['source_pts']))


def material_readout_identity(reading, binding, rgb_sha256, index, point):
    """Return only the proof for this exact current material readout."""
    from .vision_geometry import _same_frame
    evidence=reading.get('support',{}).get('actual_readout_evidence') or {}
    member=evidence.get('identity_by_source_index',{}).get(str(int(index)))
    if member is None:return None
    proof=evidence.get('identity_authorities',{}).get(member.get('authority_key'),{})
    try:
        _same_frame(binding,proof.get('source_binding',{}))
        valid=(proof.get('kind')=='current_RGB_material_graph_identity' and
            proof.get('native_path_id')==reading.get('native_path_id') and proof.get('source_seed_authority') and
            proof.get('current_RGB_measured') is True and proof.get('identity_from_projection_alone') is False and
            proof.get('decoded_native_RGB_sha256')==rgb_sha256 and
            np.array_equal(np.asarray(member.get('actual_source_uv')),np.asarray(point)))
    except (ValueError,KeyError,TypeError):valid=False
    return proof if valid else None


def paired_material_support(member):
    """Validate this measured center's original two sides, not a painted-edge alias."""
    try:
        pair=np.asarray(member.get('actual_source_sides_uv'),float)
        quantiles=np.asarray(member.get('actual_transition_quantiles_uv'),float)
        signs=np.asarray(member.get('actual_transition_luminance_signs',[]),float)
        point=np.asarray(member['actual_source_uv'],float)
        origin=np.asarray(member.get('actual_profile_origin_uv'),float)
        interval=np.asarray(member.get('actual_profile_interval_px'),float)
        normal=np.asarray(member.get('actual_profile_normal'),float)
        if (member.get('measurement_operator')!='paint_center' or pair.shape!=(2,2)
                or quantiles.shape!=(2,3,2) or signs.shape!=(2,)
                or origin.shape!=(2,) or interval.shape!=(2,) or normal.shape!=(2,)
                or not np.isfinite(np.r_[origin,interval,normal]).all() or interval[1]<=interval[0]
                or np.linalg.norm(normal)<=0
                or not np.isfinite(np.r_[pair.ravel(),quantiles.ravel(),signs]).all()
                or not np.array_equal(pair.mean(0),point) or np.linalg.norm(pair[1]-pair[0])<=0
                or not np.all(np.abs(signs)==1) or signs[0]!=-signs[1]):return None
        return pair,quantiles,signs
    except (ValueError,KeyError,TypeError):return None


def retain_material_identity(guide,reading,binding,rgb_sha256):
    """Carry point-scoped machine identity, without confirming the whole guide."""
    indices=_array(reading.get('source_indices',[])).astype(int)
    points=_array(reading.get('actual_source_uv',[])).reshape(-1,2)
    members={};authorities={};keys={}
    for index,point in zip(indices,points):
        proof=material_readout_identity(reading,binding,rgb_sha256,index,point)
        if proof is None:continue
        key=keys.setdefault(id(proof),str(len(keys)));authorities[key]=proof
        members[str(int(index))]=dict(authority_key=key,original_query_index=int(index))
    return dict(source_binding=dict(binding),native_path_id=guide.get('native_geometry',{}).get('path_id'),
        query_id=reading.get('query_id'),members=members,authorities=authorities,human_pixel_acceptance=False)


def _tracked_material_identity(guide,binding):
    """A current flow query can carry only its measured predecessor's identity."""
    from .vision_geometry import _same_frame
    support=guide.get('material_identity_continuation') or {}
    source=support.get('source_binding') or {};previous=guide.get('hint_previous_source_binding') or {}
    flow=guide.get('temporal_flow_supported')
    empty=(np.empty(0,int),None)
    if guide.get('origin')!='temporal_source_search_hint' or flow is None or not support.get('members'):return empty
    if (support.get('native_path_id')!=guide.get('native_geometry',{}).get('path_id') or
        any(source.get(k)!=binding.get(k) for k in ('source_sha256','time_base','shot_id','event_id','rig_id','installation_id')) or
        int(source.get('source_pts',-1))>=int(binding['source_pts'])):return empty
    try:_same_frame(source,previous)
    except (ValueError,KeyError,TypeError):return empty
    flow=_array(flow).astype(bool);indices=[];proofs={}
    for local,member in support['members'].items():
        key=member['authority_key']
        index=int(local);proof=support.get('authorities',{}).get(key,{})
        if index<0 or index>=len(flow) or not flow[index]:continue
        try:_same_frame(source,proof.get('source_binding',{}))
        except (ValueError,KeyError,TypeError):continue
        if (proof.get('native_path_id')!=support['native_path_id'] or not proof.get('source_seed_authority') or
            proof.get('current_RGB_measured') is not True or proof.get('identity_from_projection_alone') is not False):continue
        indices.append(index)
        if key not in proofs:
            import json
            encoded=json.dumps(proof,sort_keys=True,separators=(',',':'),default=lambda value:_array(value).tolist())
            seed=proof['source_seed_authority']
            proofs[key]=dict(source_binding=source,query_id=support.get('query_id'),
                native_path_id=support['native_path_id'],material_readout_proof_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                decoded_native_RGB_sha256=proof.get('decoded_native_RGB_sha256'),
                original_seed_reference={k:seed[k] for k in ('path','sha256','kind') if k in seed},
                original_query_indices=[])
        proofs[key]['original_query_indices'].append(member['original_query_index'])
    if not indices:return empty
    authority=dict(guide['native_geometry']['scene'],kind='current_RGB_material_identity_continuation',
        source_binding=dict(binding),previous_source_binding=source,current_native_path_id=support['native_path_id'],
        original_query_indices=indices,original_readout_proof_references=proofs,
        flow_is_measurement=False,current_RGB_reread_required=True,human_pixel_acceptance=False)
    return np.asarray(indices,int),authority


def install_readout(row, guide, indices, uv, *, method, evidence=None):
    """Replace only the automatic readout; preserve accepted UV and provenance."""
    import torch
    device=row['actual_source_uv'].device
    indices=torch.as_tensor(indices,device=device,dtype=torch.long)
    uv=torch.as_tensor(uv if hasattr(uv,'detach') else np.asarray(uv),device=device,dtype=torch.float64).reshape(-1,2)
    order=torch.argsort(indices,stable=True);indices,uv=indices[order],uv[order]
    if len(indices)!=len(torch.unique(indices)):
        raise ValueError('One current source query cannot silently emit two material branches')
    segments=torch.as_tensor(guide.get('segment_ids',np.zeros(len(guide['points_uv']),int)),device=device)[indices]
    runs=[];start=0
    for end in range(1,len(indices)+1):
        if end==len(indices) or indices[end]!=indices[end-1]+1 or segments[end]!=segments[end-1]:
            runs.append(dict(actual_source_uv=uv[start:end],source_indices=indices[start:end],segment_id=int(segments[start])))
            start=end
    if 'pre_material_readout' not in row:
        row['pre_material_readout']=dict(actual_source_uv=row['actual_source_uv'],source_indices=row['source_indices'],
            method=row.get('support',{}).get('method'),is_current_output=False)
    quality=torch.zeros(len(guide['points_uv']),device=device,dtype=torch.bool);quality[indices]=True
    row.update(actual_source_uv=uv,source_indices=indices,segment_ids=segments,measured_runs=runs,
        coherent_source_uv=uv,coherent_source_indices=indices,coherent_measured_runs=runs,
        source_quality_mask=quality,source_quality_weights=quality.float(),source_peak_candidates=None,
        shape_characterization=None,coherent_flags=dict(status=method),
        support=dict(method=method,actual_readout_evidence=evidence))
    row['metadata'].update(actual_measured_count=len(indices),position_rule=method,
        original_accepted_UV_rewritten=False,missing_sections_extrapolated=False)



class CurrentNativeMaterialReader:
    def __init__(self, geometry, *, device='cuda'):
        self.geometry=geometry;self.device=str(device)
        self.base_inventory=geometry.ice_inventory()
        self.selector=self.graph=None;self.scope=None
        self.material_shape_reader=None;self.last=None
        self._ice_frame_before=None

    def reset(self, binding):
        from .vision_ice_gray_source_reentry_selector import CurrentSourceGrayReentrySelector
        from .vision_ice_final_precision_graph import PreciseSourceIceMaterialIdentityGraph
        from .vision_source_shapes import SourceMaterialShapeReader
        self.inventory=[dict(row) for row in self.base_inventory]
        self.source_context=CurrentNativeSourceMaterialContext(self.geometry,self.inventory,binding)
        self.selector=CurrentSourceGrayReentrySelector(self.inventory,source_sha256=binding['source_sha256'],
            time_base=binding['time_base'],model_context_identity=self.source_context)
        self.graph=PreciseSourceIceMaterialIdentityGraph(self.inventory,source_sha256=binding['source_sha256'],
            time_base=binding['time_base'],device=self.device)
        self.material_shape_reader=SourceMaterialShapeReader()
        from .vision_ice_nested_material_identity import SourceHouseLayerIdentityExpander
        self.house_expander=SourceHouseLayerIdentityExpander(self.inventory,
            source_sha256=binding['source_sha256'],time_base=binding['time_base'],
            profile_backend='cuda' if self.device.startswith('cuda') else 'cpu')
        from .vision_ice_local_clearance import KnownIceMaterialBoxClearance
        self.local_clearance=KnownIceMaterialBoxClearance(self.inventory,
            source_sha256=binding['source_sha256'],time_base=binding['time_base'])
        self.scope=tuple(binding[k] for k in ('source_sha256','time_base','shot_id','event_id','rig_id','installation_id') if k in binding)
        self.last=None
        self._ice_frame_before=None
        self.output_queries={}

    def delayed_readouts(self, graph, binding):
        revisions=[]
        for row in graph['rows']:
            pts=int(row['actual_pts'])
            if pts>=int(binding['source_pts']):continue
            saved=self.output_queries.get(pts)
            if saved is None:raise ValueError('Delayed source row has no original query correspondence')
            original=saved['source_binding']
            for key in ('source_sha256','time_base','shot_id','event_id','rig_id','installation_id'):
                if row['source_binding'].get(key)!=original.get(key) or binding.get(key)!=original.get(key):
                    raise ValueError('Delayed source identity cannot cross a source or shot')
            if row['decoded_native_RGB_sha256']!=saved['RGB_sha256']:
                raise ValueError('Delayed source readout changed original RGB')
            by_guide={}
            for candidate,uv in zip(row['original_candidate_indices'],row['uv']):
                qi=int(saved['candidate_query'][int(candidate)])
                gi=int(saved['owners'][qi]);local=int(saved['local_indices'][qi])
                descriptor=saved['guides'][gi]
                if descriptor['curve_id']!=row['curve_id']:raise ValueError('Delayed source row changed native path')
                by_guide.setdefault(gi,[]).append((local,uv,int(candidate)))
            for gi,values in by_guide.items():
                revisions.append(dict(source_binding=original,decoded_native_RGB_sha256=saved['RGB_sha256'],
                    evidence_source_binding=dict(binding),guide_index=gi,**saved['guides'][gi],
                    source_indices=[v[0] for v in values],actual_source_uv=[v[1] for v in values],
                    original_candidate_indices=[v[2] for v in values],temporal_evidence=row['temporal_evidence'],
                    original_source_UV_rewritten=False,source_seed_authority=row['source_seed_authority']))
        return revisions

    def _ice_state(self):
        from copy import deepcopy
        names=('inventory','selector','source_context','house_expander','local_clearance')
        components={name:getattr(self,name) for name in names if hasattr(self,name)}
        graph_state=None
        if hasattr(self.graph,'source_frame_state'):
            graph_state=(self.graph.source_frame_state(),getattr(self.graph,'_source_frame_checkpoint',None))
        else:
            components['graph']=self.graph
        return dict(components=deepcopy(components,{id(self.geometry):self.geometry}),
            graph_state=graph_state,output_queries=dict(self.output_queries),
            ice_scope=getattr(self,'ice_scope',None))

    def _restore_ice_state(self,state):
        from copy import deepcopy
        self.__dict__.update(deepcopy(state['components'],{id(self.geometry):self.geometry}))
        if state['graph_state'] is not None:
            self.graph.restore_source_frame_state(*state['graph_state'])
        self.output_queries=dict(state['output_queries'])
        self.ice_scope=state['ice_scope']

    def refine_ice(self,rgb,binding,guides,guided,*,near_board,global_exclusion=None,source_instances=None,person_boxes=(),reset=False):
        """Replace this exposure transactionally when its queries are reread."""
        from .vision_geometry import _same_frame
        key=tuple(binding[k] for k in ('source_sha256','time_base','shot_id','event_id','rig_id','installation_id') if k in binding)
        repeated=bool(self.scope==key and self.last and self.last['source_binding']['source_pts']==binding['source_pts'])
        if (reset and not repeated) or self.scope!=key:self.reset(binding)
        if repeated:
            _same_frame(self.last['source_binding'],binding)
            if self.last['decoded_native_RGB_sha256']!=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest():
                raise ValueError('Same PTS changed RGB')
        committed=self._ice_state()
        previous_checkpoint=self._ice_frame_before
        if repeated:
            self._restore_ice_state(self._ice_frame_before)
        else:
            self._ice_frame_before=committed
        if hasattr(self.graph,'checkpoint_source_frame'):
            self.graph.checkpoint_source_frame()
        working_guides=list(guides)
        working=dict(guided,readings=[dict(row,metadata=dict(row.get('metadata',{}))) for row in guided['readings']])
        previous_result=self.last
        try:
            result=self._refine_ice_current(rgb,binding,working_guides,working,
                near_board=near_board,global_exclusion=global_exclusion,
                source_instances=source_instances,person_boxes=person_boxes)
        except Exception:
            self._restore_ice_state(committed)
            self._ice_frame_before=previous_checkpoint
            self.last=previous_result
            raise
        # Commit automatic readouts only after every identity/precision gate
        # succeeds. Keep the caller's existing row objects and accepted UV.
        old_count=len(guided['readings'])
        for original,updated in zip(guided['readings'],working['readings']):
            original.clear();original.update(updated)
        guided['readings'].extend(working['readings'][old_count:])
        guides[:]=working_guides
        return result

    def _refine_ice_current(self,rgb,binding,guides,guided,*,near_board,global_exclusion=None,source_instances=None,person_boxes=()):
        from .vision_ice_source_guides import read_actual_source_query_profiles, apply_query_exclusions
        from .vision_geometry import _same_frame
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr=hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()
        binding=dict(binding,decoded_native_BGR_sha256=bgr)
        shape=rgb.shape[:2];unassigned_exclusion=_mask(global_exclusion,shape).copy()
        excluded=unassigned_exclusion.copy();guide_exclusions={};guide_protected={};curve_guide_owners={}
        names={row['native_path_id']:ci for ci,row in enumerate(self.inventory)}
        points=[];normals=[];intervals=[];curves=[];owners=[];local_indices=[];seeds_by_curve={};missing=[]
        measured_channels={}
        for guide in guides:
            evidence=guide.get('source_material_definition_evidence') or {}
            if not (evidence.get('same_exposure_saved_RGB_profiles') and evidence.get('source_records')):continue
            source=evidence['source_binding']
            if source.get('source_pts')==binding['source_pts']:_same_frame(binding,source)
            elif not (guide.get('origin')=='temporal_source_search_hint' and
                    guide.get('temporal_flow_supported') is not None and _array(guide['temporal_flow_supported']).any() and
                    all(source.get(k)==binding.get(k) for k in ('source_sha256','time_base','shot_id','installation_id')) and
                    int(source.get('source_pts',-1))<int(binding['source_pts'])):continue
            signal={'dark_band':'dark','red_excess':'red','blue_excess':'blue'}.get(guide.get('signal'))
            if signal:measured_channels.setdefault(guide.get('native_geometry',{}).get('path_id'),set()).add(signal)
        for gi,(guide,reading) in enumerate(zip(guides,guided['readings'])):
            if guide.get('kind')!='ice_paint':continue
            _same_frame(binding,guide['source_binding'])
            ci=names.get(guide.get('native_geometry',{}).get('path_id'))
            if ci is None:raise ValueError('Ice reader received a path outside current native')
            definition=self.inventory[ci]
            signal={'dark_band':'dark','red_excess':'red','blue_excess':'blue'}.get(guide.get('signal'))
            measured=measured_channels.get(definition['native_path_id'],set())
            # Geometry owns the operator; actual RGB owns its material channel.
            # A later projection query must not overwrite a measured grey/red
            # channel on the same native path with the display's default colour.
            if len(measured)==1:signal=next(iter(measured))
            if signal:definition['material_signal']=signal
            transition=guide.get('material_outward_transition')
            if transition in ('dark_to_light','light_to_dark'):definition['material_transition']=transition
            polarity=guide.get('rgb_transition_polarity')
            if definition['operator']=='material_interface' and definition['material_transition'] is None:
                if polarity:definition['material_transition']='dark_to_light'
                else:
                    missing.append(dict(id=guide['id'],reason='current_source_material_polarity_unresolved'))
                    continue
            uv=_array(reading['guide']['uv']).astype(np.float64)
            normal=_array(reading['guide']['normals']).astype(np.float64)
            length=np.linalg.norm(normal,axis=1)
            good=np.isfinite(uv).all(1)&np.isfinite(normal).all(1)&(length>1e-9)
            if guide.get('temporal_flow_supported') is not None:
                good&=_array(guide['temporal_flow_supported']).astype(bool)
            normal=normal/np.maximum(length[:,None],1e-9)
            if definition['operator']=='material_interface' and polarity:
                positive=polarity=='lightening_along_normal'
                if positive!=(definition['material_transition']=='dark_to_light'):normal=-normal
            current_mask=(_mask(guide['source_exclusion_mask'],shape)
                if guide.get('source_exclusion_mask') is not None else unassigned_exclusion)
            if guide.get('source_exclusion_mask') is not None:_same_frame(binding,guide['source_exclusion_binding'])
            guide_exclusions[gi]=current_mask
            # Without explicit component provenance retain the full mask;
            # set subtraction cannot recover overlapping guide-only vetoes.
            guide_protected[gi]=(_mask(guide['source_exclusion_protected_mask'],shape)
                if guide.get('source_exclusion_protected_mask') is not None else current_mask)
            curve_guide_owners.setdefault(ci,[]).append(gi)
            # SourceRinkReader already built each final mask, including any
            # same-frame reviewed exception. The shared processing mask may
            # veto only pixels excluded for every guide; per-query flags below
            # preserve each owner's stricter mask before identity propagation.
            excluded&=current_mask
            ids=np.flatnonzero(good)
            # A tracked source query already has a local material window. The
            # broad 128px model-bootstrap interval would merge neighbouring rings
            # into a grey-line band and lose the existing fine source readout.
            extent=float(guide.get('material_profile_half_extent_px',guide.get('search_radius_px',18.)))
            extents=_array(guide.get('material_profile_half_extent_px_per_point',np.full(len(uv),extent))).astype(float)
            if extents.shape!=(len(uv),) or not np.isfinite(extents).all() or (extents<=0).any():
                raise ValueError('Material profile extent lost its original source query')
            windows=_array(guide.get('material_profile_intervals_px',np.c_[-extents,extents])).astype(float)
            if (windows.shape!=(len(uv),2) or not np.isfinite(windows).all()
                    or np.any(windows[:,1]<=windows[:,0])):
                raise ValueError('Material profile interval lost its original source query')
            if (definition['operator']=='material_interface' and polarity
                    and (polarity=='lightening_along_normal')!=(definition['material_transition']=='dark_to_light')):
                windows=-windows[:,::-1]
            points.extend(uv[ids]);normals.extend(normal[ids]);intervals.extend(windows[ids])
            curves.extend([ci]*len(ids));owners.extend([gi]*len(ids));local_indices.extend(ids)
            authority=guide.get('identity_authority')
            stored_identity=_stored_source_identity(guide,binding)
            confirmed=bool(authority and guide.get('identity_confirmed') and (guide.get('origin')=='user_confirmed_visual_supervision' or
                    authority.get('kind') in ('current_native_ice_source_correspondence', 'current_RGB_candidate_admission') or
                    stored_identity or
                    (guide.get('source_reference_navigation_only') and authority.get('kind')=='user_confirmed_visual_supervision')))
            seed_ids=ids if confirmed else np.empty(0,int)
            if not confirmed:
                tracked,authority=_tracked_material_identity(guide,binding)
                seed_ids=np.intersect1d(ids,tracked)
            current_points={}
            proof_reading=dict(reading,native_path_id=definition['native_path_id'])
            for index,point in zip(_array(reading.get('source_indices',[])),_array(reading.get('actual_source_uv',[]))):
                proof=material_readout_identity(proof_reading,binding,digest,index,point)
                if proof is not None and int(index) in ids:current_points[int(index)]=point
            if current_points:
                seed_ids=np.union1d(seed_ids,list(current_points))
                if authority is None:
                    authority=dict(kind='current_RGB_existing_readout_identity',source_binding=dict(binding),
                        current_native_scene=self.geometry.refs[0],native_path_id=definition['native_path_id'],
                        decoded_native_RGB_sha256=digest,current_RGB_measured=True,human_pixel_acceptance=False)
            if len(seed_ids):
                seed=seeds_by_curve.setdefault(ci,dict(curve_id=definition['curve_id'],physical_id=definition['feature_id'],
                    actual_pts=int(binding['source_pts']),native_pts=int(binding['source_pts']),
                    source_sha256=binding['source_sha256'],time_base=binding['time_base'],shot_id=binding['shot_id'],
                    decoded_native_RGB_sha256=digest,navigation_uv=[],source_material_class='static_painted_ice',
                    surface_and_physical_identity_confirmed=True,is_exact_centerline_pixel_ground_truth=False,
                    authority=authority))
                seed['navigation_uv'].extend(np.asarray([current_points.get(int(i),uv[i]) for i in seed_ids]).tolist())
        # Keep material channels synchronized; the native measurement operator is immutable.
        self.selector.inventory={r['curve_id']:dict(r) for r in self.inventory}
        self.graph.inventory=[dict(r) for r in self.inventory]
        measurement=read_actual_source_query_profiles(rgb,binding,self.inventory,points,normals,intervals,curves,unassigned_exclusion,
            query_guide_indices=owners,guide_exclusion_masks=guide_exclusions,
            profile_backend='cuda' if str(self.device).startswith('cuda') else 'cpu')
        measurement['query_original_guide_indices']=np.asarray(owners,np.int32)
        measurement['query_original_point_indices']=np.asarray(local_indices,np.int32)
        ice_scope=near_board.get('qualified_ice_visible_mask')
        scope_evidence=None
        if ice_scope is None and shape==(1080,1920):
            from calibration.program_tracking.source_board_measurements import measure_board_source_frame
            from .vision_programme_board_review import conservative_ice_exclusion
            board=near_board.get('current_source_board_measurement')
            if board is None:
                board=measure_board_source_frame(np.ascontiguousarray(rgb[:,:,::-1]),excluded,
                    dict(binding,actual_pts=int(binding['source_pts']),native_pts=int(binding['source_pts']),decoded_native_BGR_sha256=bgr))
            ice_scope=~conservative_ice_exclusion(excluded,board['ice_scope_boundary_candidates'])
            scope_evidence=dict(method='existing_current_RGB_yellow_foot_conservative_ice_scope',
                band_count=len(board['ice_scope_boundary_candidates']),source_binding=binding,
                mask_is_not_a_measured_curve=True)
        elif ice_scope is not None:
            scope_evidence=dict(method='existing_qualified_near_board_ice_scope',reference=near_board.get('source_reference'))
        else:ice_scope=np.zeros(shape,bool)
        ice_scope=np.asarray(ice_scope,bool)&~excluded
        clearance=None;cleared_global=None
        masks=[_array(row['region_mask']).astype(bool) for row in (source_instances or {}).get('instances',[])
            if row.get('kind')=='occluder' and row.get('region_mask') is not None and bool(_array(row['region_mask']).any())]
        if masks and person_boxes:
            from types import SimpleNamespace
            from .vision_ice_local_clearance import primary_person_box_components
            primary=max(masks,key=lambda m:int(m.sum()))
            other=np.zeros(shape,bool)
            for mask in masks:
                if mask is not primary:other|=mask
            # The existing helper's detector coordinates are half-native;
            # adapt current native boxes without changing any source location.
            source_masks=SimpleNamespace(boxes={int(binding['source_pts']):
                [np.asarray(row['box_xyxy'],float)/2 for row in person_boxes]},
                get=lambda pts,require_sam2:(primary,dict(source_binding=binding,model_sha256=source_instances.get('model_sha256'))))
            context=primary_person_box_components(source_masks,int(binding['source_pts']))
            if context['primary_box_index'] is not None:
                self.local_clearance.inventory=[dict(row) for row in self.inventory]
                clearance=self.local_clearance.consume(rgb,measurement,excluded,source_ice_scope_mask=ice_scope,
                    primary_box_mask=context['primary_box_mask'],measured_person_mask=primary|other,
                    other_person_boxes_mask=context['other_person_boxes_mask'],seeds=list(seeds_by_curve.values()))
                measurement=clearance['measurement']
                excluded=clearance['exclusion_mask']
                ice_scope=clearance['source_ice_scope_mask']
                cleared_global=unassigned_exclusion&~excluded
                apply_query_exclusions(measurement,excluded,query_guide_indices=owners,
                    guide_exclusion_masks=guide_exclusions,guide_protected_masks=guide_protected,
                    cleared_global_mask=cleared_global)
        self.source_context.inventory=[dict(r) for r in self.inventory]
        self.source_context.prepare(guides,binding)
        selected=self.selector.consume(rgb,measurement,excluded,seeds=list(seeds_by_curve.values()),source_ice_scope_mask=ice_scope)
        if clearance is not None:self.local_clearance.remember_current_source_rows(selected,measurement)
        # Reuse the existing actual alternating-material reader. Only native
        # retained layers can be expanded, and no projected radius is supplied.
        self.house_expander.inventory=[dict(r) for r in self.inventory]
        for definition in self.house_expander.inventory:
            if definition['material_transition'] is None and definition['world_curve_role'] in ('outer','middle','inner','center'):
                definition['material_transition']='dark_to_light' if definition['world_curve_role'] in ('outer','inner') else 'light_to_dark'
        represented={self.inventory[names[g['native_geometry']['path_id']]]['curve_id']
            for g,r in zip(guides,guided['readings']) if g.get('kind')=='ice_paint'
            and len(r.get('coherent_source_uv',r['actual_source_uv']))}
        nested=self.house_expander.consume(rgb,measurement,unassigned_exclusion,
            source_ice_scope_mask=ice_scope&~unassigned_exclusion,
            selected_rows=selected,current_measured_curve_ids=represented)
        if len(measurement['query_curve'])>len(owners):
            import torch
            from .vision_ice_final_precision_graph import _subset_source_row
            appended_by_curve={}
            for qi in range(len(owners),len(measurement['query_curve'])):
                ci=int(measurement['query_curve'][qi]);appended_by_curve.setdefault(ci,[]).append(qi)
            nested_masks={}
            for ci in appended_by_curve:
                nested_mask=unassigned_exclusion
                if ci in curve_guide_owners:
                    nested_mask=nested_mask.copy()
                    for gi in curve_guide_owners[ci]:nested_mask|=guide_exclusions[gi]
                nested_masks[len(guides)+ci]=nested_mask
            temporary_owners=[*owners,*[len(guides)+int(ci)
                for ci in measurement['query_curve'][len(owners):]]]
            apply_query_exclusions(measurement,unassigned_exclusion,query_guide_indices=temporary_owners,
                guide_exclusion_masks={**guide_exclusions,**nested_masks},
                guide_protected_masks=guide_protected,cleared_global_mask=cleared_global)
            # Filter before creating a new source-identity authority. Waiting
            # until graph.consume would leave a masked nested row in the new
            # guide's reusable authority even though no graph node survived.
            qualified=[];held=[]
            for row in nested['rows']:
                indices=np.asarray(row['original_candidate_indices'],int)
                keep=(np.asarray(measurement['candidate_flags'])[indices]&3)==0
                held.extend(indices[~keep].tolist())
                if keep.sum()>=2:qualified.append(_subset_source_row(row,keep))
                else:held.extend(indices[keep].tolist())
            nested=dict(nested,rows=qualified,guide_exclusion_rejected_candidate_indices=held)
            additions={}
            for ci,indices in appended_by_curve.items():
                definition=self.inventory[ci]
                # Appending guides changes len(guides); use the initial owner
                # encoded above rather than deriving it from the new length.
                nested_exclusion=nested_masks[temporary_owners[indices[0]]]
                evidence=[r for r in nested['rows'] if r['curve_id']==definition['curve_id']]
                authority=dict(kind='current_RGB_named_house_nested_material',source_binding=dict(binding),
                    current_native_scene=self.geometry.refs[0],source_rows=evidence,
                    model_projection_used_for_identity=False,human_pixel_acceptance=False)
                guide=dict(id=definition['native_path_id'],kind='ice_paint',
                    points_uv=np.asarray(measurement['query_uv'])[indices],normals=np.asarray(measurement['query_normal'])[indices],
                    material_profile_intervals_px=np.asarray(measurement['query_interval'])[indices].copy(),
                    segment_ids=np.arange(len(indices)),source_binding=dict(binding),origin='source_annotation_search_hint',
                    identity_confirmed=bool(evidence),identity_authority=authority,
                    source_group_id='nested_actual_material:'+definition['curve_id'],
                    source_exclusion_mask=torch.as_tensor(nested_exclusion,device=self.device)[None,None],source_exclusion_binding=dict(binding),search_radius_px=18.,
                    material_outward_transition=self.house_expander.inventory[ci]['material_transition'])
                bound,_=self.geometry.bind([guide]);guide=bound[0];gi=len(guides);guides.append(guide)
                guide_exclusions[gi]=nested_exclusion
                row=dict(expected_id=guide['id'],physical_id=definition['feature_id'],identity_authority=authority,
                    origin=guide['origin'],kind='ice_paint',primitive=guide['primitive'],physical_role=guide['physical_role'],
                    actual_source_uv=torch.empty((0,2),device=self.device),source_indices=torch.empty(0,device=self.device,dtype=torch.long),
                    guide=dict(uv=torch.as_tensor(guide['points_uv'],device=self.device),
                        normals=torch.as_tensor(guide['normals'],device=self.device),
                        segment_ids=torch.as_tensor(guide['segment_ids'],device=self.device,dtype=torch.long),
                        closed=bool(guide.get('closed',False)),source_binding=dict(binding)),
                    metadata=dict(source_point_count=len(indices)),confirmed_source_uv=None)
                guided['readings'].append(row)
                additions.update({qi:(gi,local) for local,qi in enumerate(indices)})
            for qi in range(len(owners),len(measurement['query_curve'])):
                gi,local=additions[qi];owners.append(gi);local_indices.append(local)
            measurement['query_original_guide_indices']=np.asarray(owners,np.int32)
            measurement['query_original_point_indices']=np.asarray(local_indices,np.int32)
        selected=dict(selected,rows=list(selected['rows'])+nested['rows'])
        graph=self.graph.consume(rgb,measurement,excluded,source_ice_scope_mask=ice_scope,selected_rows=selected)
        final_support=self.selector.retain_final_source_rows(rgb,measurement,selected,final_source_node_ids=self.graph.frames[-1]['ids'])
        # Current and delayed rows share exact candidate locators, never guessed UV.
        current=list(selected['rows'])+[r for r in graph['rows'] if r['actual_pts']==binding['source_pts']]
        by_guide={};seen=set();identity_proofs={};identity_members={}
        profile_rgb=np.asarray(rgb,np.float32)
        from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context, source_identity_context_fields
        for row_ordinal,row in enumerate(current):
            source_proof=None
            if (row.get('static_material_confirmed') is True and row.get('actual_source_RGB_measured') is True
                    and row.get('pixel_location_confirmed') is True and row.get('source_seed_authority')
                    and admissible_source_identity_context(row)
                    and row.get('model_projection_used_as_measurement') is False
                    and row.get('source_center_smoothed_or_interpolated') is False
                    and row.get('decoded_native_RGB_sha256')==digest
                    and row.get('source_sha256')==binding['source_sha256']
                    and row.get('actual_pts')==binding['source_pts'] and row.get('time_base')==binding['time_base']
                    and row.get('shot_id')==binding['shot_id']):
                source_proof=dict(kind='current_RGB_material_graph_identity',source_binding=dict(binding),
                    decoded_native_RGB_sha256=digest,curve_id=row['curve_id'],
                    source_seed_authority=row['source_seed_authority'],
                    temporal_evidence=row.get('temporal_evidence'),continuity_evidence=row.get('continuity_evidence'),
                    material_identity_context=source_identity_context_fields(row),
                    current_RGB_measured=True,identity_from_projection_alone=False,human_pixel_acceptance=False)
            actual_sides=np.asarray(row.get('source_sides_uv',row.get('candidate_sides',[])),float)
            for ri,(candidate,uv) in enumerate(zip(row['original_candidate_indices'],row['uv'])):
                if candidate in seen:continue
                seen.add(candidate);qi=int(measurement['candidate_query'][candidate]);gi=owners[qi];local=local_indices[qi]
                if self.inventory[int(measurement['query_curve'][qi])]['curve_id']!=row['curve_id']:
                    raise ValueError('Material graph readout changed its current native query identity')
                by_guide.setdefault(gi,[]).append((local,uv,candidate))
                if source_proof is not None:
                    import cv2
                    quantiles=np.asarray(measurement['candidate_side_quantiles_uv'][candidate],float)
                    positions=quantiles[:,[0,2]].astype(np.float32)
                    signs=np.zeros(2,int)
                    if np.isfinite(positions).all():
                        samples=cv2.remap(profile_rgb,positions[:,:,0],positions[:,:,1],cv2.INTER_LINEAR)
                        luminance=samples@np.array([.299,.587,.114])
                        signs=np.sign(luminance[:,1]-luminance[:,0]).astype(int)
                    key=row['curve_id']+':source_row:'+str(row_ordinal)
                    identity_proofs.setdefault(gi,{})[key]=dict(source_proof,
                        native_path_id=guides[gi]['native_geometry']['path_id'])
                    identity_members.setdefault(gi,{})[str(int(local))]=dict(authority_key=key,
                        original_candidate_index=int(candidate),actual_source_uv=np.asarray(uv).tolist(),
                        actual_transition_quantiles_uv=quantiles.tolist(),
                        actual_transition_luminance_signs=signs.tolist(),
                        measurement_operator=self.inventory[int(measurement['query_curve'][qi])]['operator'],
                        material_signal=self.inventory[int(measurement['query_curve'][qi])]['material_signal'],
                        actual_source_sides_uv=(actual_sides[ri].tolist() if actual_sides.shape==(len(row['uv']),2,2) else None),
                        actual_profile_origin_uv=np.asarray(measurement['query_uv'][qi]).tolist(),
                        actual_profile_interval_px=np.asarray(measurement['query_interval'][qi]).tolist(),
                        actual_profile_normal=np.asarray(measurement['query_normal'][qi]).tolist(),
                        material_transition=self.inventory[int(measurement['query_curve'][qi])]['material_transition'])
        for gi,(guide,reading) in enumerate(zip(guides,guided['readings'])):
            if guide.get('kind')!='ice_paint':continue
            values=by_guide.get(gi,[])
            precise={int(r[0]):r[1] for r in values}
            base_indices=_array(reading.get('coherent_source_indices',reading['source_indices']))
            base_uv=_array(reading.get('coherent_source_uv',reading['actual_source_uv']))
            # An additional proof may be unavailable without contradicting the
            # current RGB reader. Preserve that already measured progress; never
            # copy old-frame UV, or erase it just because a new graph lacks nodes.
            retained={int(i):p for i,p in zip(base_indices,base_uv) if int(i) not in precise}
            combined={**retained,**precise}
            indices=sorted(combined)
            install_readout(reading,guide,indices,[combined[i] for i in indices],
                method='current_native_RGB_with_per_query_precision_evidence',
                evidence=dict(candidate_indices=[r[2] for r in values],precise_source_indices=sorted(precise),
                    retained_current_RGB_source_indices=sorted(retained),
                    retained_current_RGB_is_additional_graph_proof=False,
                    identity_authorities=identity_proofs.get(gi,{}),identity_by_source_index=identity_members.get(gi,{}),
                    native_path_id=guide['native_geometry']['path_id'],source_binding=binding,
                    precision_reader='PreciseSourceIceMaterialIdentityGraph'))
        delayed=self.delayed_readouts(graph,binding)
        self.output_queries[int(binding['source_pts'])]=dict(source_binding=dict(binding),RGB_sha256=digest,
            candidate_query=np.asarray(measurement['candidate_query']).copy(),owners=np.asarray(owners),
            local_indices=np.asarray(local_indices),guides={gi:dict(expected_id=guide['id'],
                native_path_id=guide['native_geometry']['path_id'],native_scene=self.geometry.refs[0],
                curve_id=self.inventory[names[guide['native_geometry']['path_id']]]['curve_id'])
                for gi,guide in enumerate(guides) if guide['kind']=='ice_paint'})
        result=dict(source_binding=dict(binding),decoded_native_RGB_sha256=digest,inventory=self.inventory,
            delayed_source_readouts=delayed,current_source_context=dict(
                reader='CurrentSourceGrayReentrySelector + SameStateIceModelContextIdentity',
                navigation_kind=self.source_context.navigation_kind,
                navigation_flags=dict(getattr(self.source_context,'navigation_flags',{})),
                current_native_scene=self.geometry.refs[0],old_scene_or_camera_loaded=False,
                reentry=self.selector.latest_current_source_context_reentry,
                decisions=getattr(self.source_context,'latest_source_decisions',[])),
            measurement=measurement,selected=selected,nested_material_identity=nested,temporal_graph=graph,retained_final_support=final_support,
            missing_source_material_definitions=missing,ice_scope_evidence=scope_evidence,
            local_person_box_clearance=None if clearance is None else {k:v for k,v in clearance.items() if k!='measurement'},
            current_named_readout_count=sum(len(row['uv']) for row in current),
            final_ice_readout_count=sum(len(row['actual_source_uv'])
                for row in guided['readings'] if row.get('kind')=='ice_paint'),
            native_geometry_only=True,old_geometry_imported=False,accuracy_accepted=False)
        self.ice_scope=ice_scope&~unassigned_exclusion;self.last=result
        return result

    def refine_print(self, rgb, binding, guides, guided, *, near_board):
        """Reuse the original ordered q25/q50/q75 reader on retained native print.

        Accepted/transported source UV only navigate the current image. Existing
        source groups and parent/hole definitions are never reconstructed.
        """
        from .vision_instances import _add_existing_geometry_dependencies
        _add_existing_geometry_dependencies()
        from calibration.program_tracking.extract_first_shot_board_source import profile_read
        from .vision_board_reader import _unexcluded_pixels
        from .vision_geometry import _same_frame
        bgr=np.ascontiguousarray(rgb[:,:,::-1]);shape=rgb.shape[:2]
        face=near_board.get('vertical_face_admission_mask')
        reports=[]
        for guide, reading in zip(guides,guided['readings']):
            if guide.get('kind')!='ad_boundary':continue
            definition=self.geometry.lookup(guide['id'])
            if definition is None or definition['layer']!='advertisement':
                raise ValueError('Print readout must bind to a retained native path')
            _same_frame(binding,guide['source_binding'])
            uv=_array(guide['points_uv']).astype(float)
            segments=_array(guide.get('segment_ids',np.zeros(len(uv),int)))
            excluded=_mask(guide.get('source_exclusion_mask'),shape).copy()
            if guide.get('source_exclusion_mask') is not None:
                _same_frame(binding,guide['source_exclusion_binding'])
            if face is not None:excluded|=~np.asarray(face,bool)
            precision={};evidence=[];measured_normals={}
            if shape==(1080,1920):
                for segment in np.unique(segments):
                    local=np.flatnonzero(segments==segment);path=uv[local]
                    if len(path)<2:continue
                    values,_=profile_read(bgr,excluded,path,[1/3]*3,6,
                        maximum_blur=4.,minimum_amplitude=25.,navigation_tolerance=3.5,
                        require_clear_end_plateaus=False)
                    tangent=np.roll(path,-1,axis=0)-np.roll(path,1,axis=0)
                    tangent[0],tangent[-1]=path[1]-path[0],path[-1]-path[-2]
                    normal=np.c_[-tangent[:,1],tangent[:,0]]
                    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),.01)
                    flow=guide.get('temporal_flow_supported')
                    if flow is not None:flow=_array(flow)
                    for k,value in enumerate(values):
                        if value is None:continue
                        i=int(local[k]);point,width,contrast=value
                        if flow is not None and not flow[i]:continue
                        support=point+normal[k]*np.linspace(-max(2,width),max(2,width),17)[:,None]
                        if not _unexcluded_pixels(excluded,np.vstack([uv[i],point,support])).all():continue
                        precision[i]=point
                        measured_normals[i]=normal[k].copy()
                        evidence.append(dict(source_query_index=i,actual_source_uv=point,
                            transition_iqr_px=width,material_contrast=contrast,source_support_uv=support,
                            profile_normal=normal[k].copy()))
            indices=_array(reading.get('coherent_source_indices',reading['source_indices']))
            points=_array(reading.get('coherent_source_uv',reading['actual_source_uv']))
            retained={int(i):p for i,p in zip(indices,points) if int(i) not in precision}
            combined={**retained,**precision};order=sorted(combined)
            receipt=dict(source_binding=binding,native_path_id=definition['id'],
                native_role=definition.get('role'),parent_contour_key=definition.get('parent_contour_key'),
                precise_source_indices=sorted(precision),retained_current_RGB_source_indices=sorted(retained),
                measurements=evidence,existing_reader='extract_first_shot_board_source.profile_read',
                original_source_UV_rewritten=False,old_geometry_imported=False)
            install_readout(reading,guide,order,[combined[i] for i in order],
                method='current_native_print_RGB_half_transition',evidence=receipt)
            if measured_normals:
                import torch
                # The reused print profiler computes its own directions from
                # each original source run. Export those actual directions;
                # a reference guide may legitimately carry zero placeholders.
                # Keep untouched directions for any retained generic readouts.
                current=dict(reading.get('guide',{}))
                values=_array(current.get('normals',guide.get('normals',np.zeros_like(uv)))).astype(float).copy()
                if values.shape!=uv.shape:raise ValueError('Print source query normals differ')
                for i,normal in measured_normals.items():values[i]=normal
                current.setdefault('uv',torch.as_tensor(uv,device=self.device,dtype=torch.float64))
                current.setdefault('segment_ids',torch.as_tensor(segments,device=self.device,dtype=torch.long))
                current.setdefault('source_binding',dict(binding))
                current['normals']=torch.as_tensor(values,device=self.device,dtype=torch.float64)
                reading['guide']=current
                receipt['source_profile_directions_exported']=True
            reports.append(receipt)
        return dict(readings=reports,source_binding=binding,current_native_retained_paths_only=True)

    def read_unassigned_material(self,rgb,binding,prediction,source,*,flow_pair=None,reset=False,guides=(),guided=None):
        """Learned shape heads search RGB; unnamed fragments never enter named output."""
        if self.material_shape_reader is None:self.reset(binding)
        packet=dict(source_binding=binding,RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            classes=prediction['classes'],class_probabilities=prediction['class_probabilities'],source=source)
        confirmed=[dict(guide,points_uv=_array(guide['points_uv'])) for guide in guides
            if guide.get('kind')=='ice_paint' and guide.get('origin')=='user_confirmed_visual_supervision'
            and guide.get('identity_confirmed') and guide.get('identity_authority')]
        # Newly established machine identity can seed a newly visible shape in
        # any exposure. Only the exact RGB readouts with point-scoped graph
        # proof enter this bridge, never the whole projected/search curve.
        inventory={row['native_path_id']:row for row in self.inventory}
        for guide,reading in zip(guides,(guided or {}).get('readings',[])):
            definition=inventory.get(guide.get('native_geometry',{}).get('path_id'))
            if definition is None or not (definition['curve_id'].endswith(':outer') or definition['operator']=='paint_center'):continue
            if reading.get('native_path_id') not in (None,definition['native_path_id']):
                raise ValueError('Material shape identity changed its bound native path')
            proof_reading=dict(reading,native_path_id=definition['native_path_id'])
            points=[];source_indices=[];members={};authorities={}
            original=reading.get('support',{}).get('actual_readout_evidence') or {}
            for i,p in zip(_array(reading.get('source_indices',[])),_array(reading.get('actual_source_uv',[]))):
                proof=material_readout_identity(proof_reading,binding,packet['RGB_sha256'],i,p)
                if proof is None:continue
                member=original['identity_by_source_index'][str(int(i))]
                members[str(len(points))]=dict(member)
                authorities[member['authority_key']]=proof;points.append(p);source_indices.append(int(i))
            if len(points)<2:continue
            authority=dict(kind='current_RGB_verified_material_identity_for_shape',source_binding=dict(binding),
                current_native_scene=dict(self.geometry.refs[0]),native_path_id=definition['native_path_id'],
                decoded_native_RGB_sha256=packet['RGB_sha256'],human_pixel_acceptance=False,
                source_readout_evidence=dict(identity_by_source_index=members,identity_authorities=authorities))
            confirmed.append(dict(id='ice.'+definition['curve_id'],kind='ice_paint',
                physical_role=('painted_stroke_center_axis' if definition['operator']=='paint_center' else 'filled_disc_boundary'),
                points_uv=np.asarray(points),source_binding=dict(binding),
                origin='current_RGB_verified_material_identity',identity_confirmed=True,identity_authority=authority,
                source_group_id=guide.get('source_group_id',guide.get('query_id',guide['id'])),
                segment_ids=np.cumsum(np.r_[0,np.diff(source_indices)!=1]),
                search_radius_px=guide.get('search_radius_px',18.)))
        roles={'ice.'+row['curve_id']:dict(physical_role='painted_stroke_center_axis'
            if row['operator']=='paint_center' else 'filled_disc_boundary') for row in self.inventory}
        result=self.material_shape_reader.read(rgb,packet,flow_pair=flow_pair,reset=reset,
            primitive_probabilities=prediction['primitive_probabilities'],primitive_source_binding=binding,
            source_ice_search_mask=self.ice_scope,source_ice_search_binding=binding,
            confirmed_source_guide_packet=dict(source_binding=binding,guides=confirmed) if confirmed else None,
            role_definitions=roles)
        for row in result.get('source_shape_readings',[]):
            definition=self.geometry.lookup(row.get('physical_curve_id'))
            if definition is not None and definition['layer']=='ice':
                expected='painted_stroke_center_axis' if definition['measurement']=='centreline' else 'filled_disc_boundary'
                row['native_geometry']=dict(scene=self.geometry.refs[0],path_id=definition['id'],measurement=definition['measurement'])
                row['native_material_role_matches']=row.get('physical_role')==expected
                row['source_role_requires_material_reread']=row.get('physical_role')!=expected
            elif row.get('physical_curve_id'):
                row['native_identity_pending']=dict(original_physical_curve_id=row['physical_curve_id'],reason='not_retained_by_current_native')
                row['physical_identity_verified']=False
        result['current_native_retained_identity_gate']=True
        result['shape_readouts_are_additional_candidates_not_independent_landmarks']=True
        return result


class CurrentNativeSourceMaterialContext:
    """Reuse the existing two-RGB context checks with authenticated source guides.

    The adapter supplies no old camera/scene projection, no new material detector,
    and no new physical identity. Its named scope is the current native inventory.
    """
    def __new__(cls,geometry,inventory,source_binding):
        from .vision_ice_model_context_identity import SameStateIceModelContextIdentity
        from fractions import Fraction
        class SourceContext(SameStateIceModelContextIdentity):
            def __init__(self):
                self.geometry=geometry;self.inventory=[dict(v) for v in inventory]
                self.source_sha256=source_binding['source_sha256'];self.time_base=str(Fraction(source_binding['time_base']))
                self.scene_reference=geometry.refs[0];self.state_reference=geometry.refs[0]
                self.camera_reference=None;self.source_negative_materials=[]
                self.feature_kind={r['feature_id']:'circle' if r.get('native_kind')=='ice_circle' else 'line' for r in inventory}
                self.previous_gray=self.previous_mask=self.previous_binding=None;self.previous_tracks={}
                self.navigation_flags=dict(model_projection_used_as_navigation_and_identity_context=False,
                    model_projection_used_for_navigation=False,
                    model_projection_used_for_identity_confirmation=False,
                    projected_UV_used_as_source_pixel_truth=False,source_identity_not_confirmed_from_projection_alone=True,
                    fit_residual_used_for_identity=False,verified_source_guides_used_as_navigation=False)
                self.navigation_kind='actual_RGB_material_field_order_and_continuity_with_native_source_guides'
                self.navigation_field_order_test='current_native_named_source_guides_and_distinct_actual_material_support'
                self.navigation_paths=[[] for _ in inventory];self.authorized={};self.navigation_binding=None
                self.projected_candidates=set()
                self.authority={}
            def prepare(self,guides,binding):
                from .vision_geometry import _same_frame
                self.navigation_paths=[[] for _ in self.inventory];self.authorized={};self.projected_candidates=set()
                projection_paths=[];verified_source_navigation=False
                names={r['native_path_id']:i for i,r in enumerate(self.inventory)}
                for guide in guides:
                    if guide.get('kind')!='ice_paint':continue
                    _same_frame(binding,guide['source_binding'])
                    native=guide.get('native_geometry',{})
                    if native.get('scene')!=self.scene_reference or native.get('path_id') not in names:
                        raise ValueError('Material context requires a retained current native source guide')
                    ci=names[native['path_id']];uv=_array(guide['points_uv']).astype(float)
                    segments=_array(guide.get('segment_ids',np.zeros(len(uv),int)))
                    valid=np.isfinite(uv).all(1)
                    if guide.get('temporal_flow_supported') is not None:valid&=_array(guide['temporal_flow_supported']).astype(bool)
                    authority=guide.get('identity_authority')
                    authorized=False
                    authorized_indices=np.flatnonzero(valid)
                    if authority and guide.get('identity_confirmed'):
                        if (guide.get('origin')=='user_confirmed_visual_supervision'
                                or authority.get('kind') in ('current_native_ice_source_correspondence', 'current_RGB_candidate_admission')
                                or _stored_source_identity(guide,binding)
                                or (guide.get('source_reference_navigation_only') and authority.get('kind')=='user_confirmed_visual_supervision')
                                or (guide.get('origin')=='temporal_source_search_hint' and guide.get('temporal_flow_supported') is not None)):
                            self.authorized[self.inventory[ci]['curve_id']]=authority
                            authorized=True
                    if not authorized:
                        tracked,authority=_tracked_material_identity(guide,binding)
                        authorized_indices=np.intersect1d(authorized_indices,tracked)
                        if len(authorized_indices):
                            self.authorized[self.inventory[ci]['curve_id']]=authority;authorized=True
                    navigation=guide.get('native_identity_navigation')
                    if navigation is None and guide.get('model_search_projection'):
                        navigation=dict(native_path_id=native['path_id'],source_binding=guide['source_binding'],
                            native_scene=native['scene'],model_search_projection=guide['model_search_projection'],
                            points_uv=uv,segment_ids=segments,projected_UV_used_as_source_pixel_truth=False)
                    if navigation is not None:
                        _same_frame(binding,navigation['source_binding'])
                        projection=navigation['model_search_projection']
                        if (navigation.get('native_scene')!=self.scene_reference
                                or navigation.get('native_path_id')!=native['path_id']
                                or projection.get('model_revision')!=self.scene_reference['sha256']
                                or navigation.get('projected_UV_used_as_source_pixel_truth') is not False):
                            raise ValueError('Material identity context changed the current native path')
                        if projection.get('source_binding'):
                            _same_frame(binding,projection['source_binding'])
                        elif projection.get('native_pts')!=binding['source_pts']:
                            raise ValueError('Material identity context changed the exposure')
                        projected_uv=_array(navigation['points_uv']).astype(float)
                        projected_segments=_array(navigation['segment_ids'])
                        projected_ids=np.flatnonzero(np.isfinite(projected_uv).all(1))
                        queried=False
                        for group in np.split(projected_ids,np.flatnonzero((np.diff(projected_ids)>1)|
                                (np.diff(projected_segments[projected_ids])!=0))+1):
                            if len(group)>1:
                                self.navigation_paths[ci].append(projected_uv[group]);queried=True
                        if queried:
                            projection_paths.append(native['path_id'])
                            self.projected_candidates.add(self.inventory[ci]['curve_id'])
                    # Saved unknown names locate RGB profiles, but must not
                    # replace the independent native field-order navigation.
                    if authorized:
                        ids=authorized_indices
                        for group in np.split(ids,np.flatnonzero((np.diff(ids)>1)|(np.diff(segments[ids])!=0))+1):
                            if len(group)>1:
                                self.navigation_paths[ci].append(uv[group]);verified_source_navigation=True
                projected=bool(projection_paths)
                self.navigation_flags.update(model_projection_used_for_navigation=projected,
                    model_projection_used_as_navigation_and_identity_context=projected,
                    model_projection_used_for_identity_confirmation=False,
                    verified_source_guides_used_as_navigation=bool(verified_source_navigation and not projected))
                self.navigation_kind=('actual_RGB_material_field_order_and_continuity_with_model_navigation'
                    if projected else 'actual_RGB_material_field_order_and_continuity_with_native_source_guides')
                self.navigation_binding=dict(binding)
                self.authority=dict(self.scene_reference,kind='current_native_source_identity_context',reference=self.scene_reference,
                    original_feature_identity_authorities=self.authorized,source_binding=dict(binding),
                    original_accepted_UV_rewritten=False,model_projection_used=projected,
                    model_projection_used_for_navigation=projected,model_projection_used_for_identity_confirmation=False,
                    projected_navigation_path_ids=sorted(set(projection_paths)))
            def navigation(self,measurement):
                from .vision_geometry import _same_frame
                _same_frame(self.navigation_binding,measurement['source_binding'])
                return self.navigation_paths
            def consume(self,*args,**kwargs):
                result=super().consume(*args,**kwargs)
                self.latest_source_decisions=result['decisions']
                return result
            def _source_context_output_curve_allowed(self,definition):
                # This enables the existing two-RGB material/field-order/LK
                # checks. It does not issue an identity seed by itself.
                return definition['curve_id'] in self.authorized or definition['curve_id'] in self.projected_candidates
        return SourceContext()
