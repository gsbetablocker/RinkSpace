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
        self.scope=(binding['source_sha256'],binding['time_base'],binding['shot_id'])
        self.last=None
        self.output_queries={}

    def delayed_readouts(self, graph, binding):
        revisions=[]
        for row in graph['rows']:
            pts=int(row['actual_pts'])
            if pts>=int(binding['source_pts']):continue
            saved=self.output_queries.get(pts)
            if saved is None:raise ValueError('Delayed source row has no original query correspondence')
            original=saved['source_binding']
            for key in ('source_sha256','time_base','shot_id'):
                if row['source_binding'][key]!=original[key] or binding[key]!=original[key]:
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

    def refine_ice(self,rgb,binding,guides,guided,*,near_board,global_exclusion=None,source_instances=None,person_boxes=(),reset=False):
        from .vision_ice_source_guides import read_actual_source_query_profiles
        from .vision_geometry import _same_frame
        key=(binding['source_sha256'],binding['time_base'],binding['shot_id'])
        if reset or self.scope!=key:self.reset(binding)
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr=hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()
        binding=dict(binding,decoded_native_BGR_sha256=bgr)
        if self.last and self.last['source_binding']['source_pts']==binding['source_pts']:
            if self.last['decoded_native_RGB_sha256']!=digest:raise ValueError('Same PTS changed RGB')
            return self.last
        shape=rgb.shape[:2];excluded=_mask(global_exclusion,shape).copy()
        names={row['native_path_id']:ci for ci,row in enumerate(self.inventory)}
        points=[];normals=[];intervals=[];curves=[];owners=[];local_indices=[];seeds_by_curve={};missing=[]
        for gi,(guide,reading) in enumerate(zip(guides,guided['readings'])):
            if guide.get('kind')!='ice_paint':continue
            _same_frame(binding,guide['source_binding'])
            ci=names.get(guide.get('native_geometry',{}).get('path_id'))
            if ci is None:raise ValueError('Ice reader received a path outside current native')
            definition=self.inventory[ci]
            signal={'dark_band':'dark','red_excess':'red','blue_excess':'blue'}.get(guide.get('signal'))
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
            current_mask=_mask(guide.get('source_exclusion_mask'),shape)
            if guide.get('source_exclusion_mask') is not None:_same_frame(binding,guide['source_exclusion_binding'])
            excluded|=current_mask
            ids=np.flatnonzero(good)
            # A tracked source query already has a local material window. The
            # broad 128px model-bootstrap interval would merge neighbouring rings
            # into a grey-line band and lose the existing fine source readout.
            extent=float(guide.get('material_profile_half_extent_px',guide.get('search_radius_px',18.)))
            points.extend(uv[ids]);normals.extend(normal[ids]);intervals.extend([[-extent,extent]]*len(ids))
            curves.extend([ci]*len(ids));owners.extend([gi]*len(ids));local_indices.extend(ids)
            authority=guide.get('identity_authority')
            if authority and guide.get('identity_confirmed') and (guide.get('origin')=='user_confirmed_visual_supervision' or
                    authority.get('kind')=='current_native_ice_source_correspondence'):
                seed=seeds_by_curve.setdefault(ci,dict(curve_id=definition['curve_id'],physical_id=definition['feature_id'],
                    actual_pts=int(binding['source_pts']),native_pts=int(binding['source_pts']),
                    source_sha256=binding['source_sha256'],time_base=binding['time_base'],shot_id=binding['shot_id'],
                    decoded_native_RGB_sha256=digest,navigation_uv=[],source_material_class='static_painted_ice',
                    surface_and_physical_identity_confirmed=True,is_exact_centerline_pixel_ground_truth=False,
                    authority=authority))
                seed['navigation_uv'].extend(uv[ids].tolist())
        # Keep material channels synchronized; the native measurement operator is immutable.
        self.selector.inventory={r['curve_id']:dict(r) for r in self.inventory}
        self.graph.inventory=[dict(r) for r in self.inventory]
        measurement=read_actual_source_query_profiles(rgb,binding,self.inventory,points,normals,intervals,curves,excluded,
            query_guide_indices=owners,profile_backend='cuda' if str(self.device).startswith('cuda') else 'cpu')
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
        clearance=None
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
        nested=self.house_expander.consume(rgb,measurement,excluded,source_ice_scope_mask=ice_scope,
            selected_rows=selected,current_measured_curve_ids=represented)
        selected=dict(selected,rows=list(selected['rows'])+nested['rows'])
        if len(measurement['query_curve'])>len(owners):
            import torch
            appended_by_curve={}
            for qi in range(len(owners),len(measurement['query_curve'])):
                ci=int(measurement['query_curve'][qi]);appended_by_curve.setdefault(ci,[]).append(qi)
            additions={}
            for ci,indices in appended_by_curve.items():
                definition=self.inventory[ci]
                evidence=[r for r in nested['rows'] if r['curve_id']==definition['curve_id']]
                authority=dict(kind='current_RGB_named_house_nested_material',source_binding=dict(binding),
                    current_native_scene=self.geometry.refs[0],source_rows=evidence,
                    model_projection_used_for_identity=False,human_pixel_acceptance=False)
                guide=dict(id=definition['native_path_id'],kind='ice_paint',
                    points_uv=np.asarray(measurement['query_uv'])[indices],normals=np.asarray(measurement['query_normal'])[indices],
                    segment_ids=np.arange(len(indices)),source_binding=dict(binding),origin='source_annotation_search_hint',
                    identity_confirmed=bool(evidence),identity_authority=authority,
                    source_group_id='nested_actual_material:'+definition['curve_id'],
                    source_exclusion_mask=torch.as_tensor(excluded,device=self.device)[None,None],source_exclusion_binding=dict(binding),search_radius_px=18.,
                    material_outward_transition=self.house_expander.inventory[ci]['material_transition'])
                bound,_=self.geometry.bind([guide]);guide=bound[0];gi=len(guides);guides.append(guide)
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
        graph=self.graph.consume(rgb,measurement,excluded,source_ice_scope_mask=ice_scope,selected_rows=selected)
        final_support=self.selector.retain_final_source_rows(rgb,measurement,selected,final_source_node_ids=self.graph.frames[-1]['ids'])
        # Current and delayed rows share exact candidate locators, never guessed UV.
        current=list(selected['rows'])+[r for r in graph['rows'] if r['actual_pts']==binding['source_pts']]
        by_guide={};seen=set()
        for row in current:
            for candidate,uv in zip(row['original_candidate_indices'],row['uv']):
                if candidate in seen:continue
                seen.add(candidate);qi=int(measurement['candidate_query'][candidate]);gi=owners[qi];local=local_indices[qi]
                by_guide.setdefault(gi,[]).append((local,uv,candidate))
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
                current_native_scene=self.geometry.refs[0],old_scene_or_camera_loaded=False,
                reentry=self.selector.latest_current_source_context_reentry),
            measurement=measurement,selected=selected,nested_material_identity=nested,temporal_graph=graph,retained_final_support=final_support,
            missing_source_material_definitions=missing,ice_scope_evidence=scope_evidence,
            local_person_box_clearance=None if clearance is None else {k:v for k,v in clearance.items() if k!='measurement'},
            current_named_readout_count=sum(len(row['uv']) for row in current),
            native_geometry_only=True,old_geometry_imported=False,accuracy_accepted=False)
        self.ice_scope=ice_scope;self.last=result
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
            precision={};evidence=[]
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
                        evidence.append(dict(source_query_index=i,actual_source_uv=point,
                            transition_iqr_px=width,material_contrast=contrast,source_support_uv=support))
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
            reports.append(receipt)
        return dict(readings=reports,source_binding=binding,current_native_retained_paths_only=True)

    def read_unassigned_material(self,rgb,binding,prediction,source,*,flow_pair=None,reset=False,guides=()):
        """Learned shape heads search RGB; unnamed fragments never enter named output."""
        if self.material_shape_reader is None:self.reset(binding)
        packet=dict(source_binding=binding,RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            classes=prediction['classes'],class_probabilities=prediction['class_probabilities'],source=source)
        confirmed=[dict(guide,points_uv=_array(guide['points_uv'])) for guide in guides
            if guide.get('kind')=='ice_paint' and guide.get('origin')=='user_confirmed_visual_supervision'
            and guide.get('identity_confirmed') and guide.get('identity_authority')] if reset else []
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
                    projected_UV_used_as_source_pixel_truth=False,source_identity_not_confirmed_from_projection_alone=True,
                    fit_residual_used_for_identity=False,verified_source_guides_used_as_navigation=True)
                self.navigation_kind='actual_RGB_material_field_order_and_continuity_with_native_source_guides'
                self.navigation_field_order_test='current_native_named_source_guides_and_distinct_actual_material_support'
                self.navigation_paths=[[] for _ in inventory];self.authorized={};self.navigation_binding=None
                self.authority={}
            def prepare(self,guides,binding):
                from .vision_geometry import _same_frame
                self.navigation_paths=[[] for _ in self.inventory];self.authorized={}
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
                    ids=np.flatnonzero(valid)
                    for group in np.split(ids,np.flatnonzero((np.diff(ids)>1)|(np.diff(segments[ids])!=0))+1):
                        if len(group)>1:self.navigation_paths[ci].append(uv[group])
                    authority=guide.get('identity_authority')
                    if authority and guide.get('identity_confirmed'):
                        if (guide.get('origin')=='user_confirmed_visual_supervision'
                                or authority.get('kind')=='current_native_ice_source_correspondence'
                                or (guide.get('origin')=='temporal_source_search_hint' and guide.get('temporal_flow_supported') is not None)):
                            self.authorized[self.inventory[ci]['curve_id']]=authority
                self.navigation_binding=dict(binding)
                self.authority=dict(self.scene_reference,kind='current_native_source_identity_context',reference=self.scene_reference,
                    original_feature_identity_authorities=self.authorized,source_binding=dict(binding),
                    original_accepted_UV_rewritten=False,model_projection_used=False)
            def navigation(self,measurement):
                from .vision_geometry import _same_frame
                _same_frame(self.navigation_binding,measurement['source_binding'])
                return self.navigation_paths
            def _source_context_output_curve_allowed(self,definition):
                return definition['curve_id'] in self.authorized
        return SourceContext()
