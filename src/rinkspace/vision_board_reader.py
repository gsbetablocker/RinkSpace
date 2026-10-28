"""Use the existing ordered material readers for both physical board boundaries."""
import hashlib
import numpy as np


def source_front_face_top(rgb, binding, observed_top_uv, exclusion):
    """Read the inside edge of the blue cap for front-face ad searches.

    This is a search-domain boundary, not a new board observation. The top
    geometry remains the upper cap edge. A guide at that upper edge must not
    let an advertisement reader accept the same blue material as its panel.
    """
    from .vision_source_topology import read_source_board_rail_material_profiles
    image = np.ascontiguousarray(rgb)
    hints = np.asarray(observed_top_uv, float).reshape(-1, 2)
    if not len(hints):
        return np.empty((0, 2)), dict(status='no_observed_front_face_top')
    packet = read_source_board_rail_material_profiles(image, dict(
        source_binding=binding, RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        actual_source_top_uv=hints), source_instances=[dict(
            object_id='current_source_exclusion', kind='occluder',
            source_binding=binding, region_mask=np.asarray(exclusion,bool))], sample_spacing_px=1,
        exit_requires_current_blue_material=True)
    good = np.asarray(packet['primary_role_samples'],bool)
    lower = np.asarray(packet['actual_source_blue_to_white_exit_uv'])[good]
    return lower, dict(status='current_RGB_cap_exit_search_boundary',
        source_binding=binding, lower_blue_cap_uv=lower.tolist(),
        source_material_role_evidence=[e for e,k in zip(packet['source_material_role_evidence'],good) if k],
        creates_observations=False)


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _bound_exclusion_mask(guide, binding, native_hw):
    from .vision_geometry import _same_frame
    mask = guide.get('source_exclusion_mask')
    if mask is None:
        return None
    if guide.get('source_exclusion_binding') is None:
        raise ValueError('Board source exclusion requires its source frame binding')
    _same_frame(binding, guide['source_exclusion_binding'])
    mask = _numpy(mask)
    if mask.shape == (1, 1, *native_hw):
        mask = mask[0, 0]
    if mask.shape != tuple(native_hw) or mask.dtype != np.bool_:
        raise ValueError('Board source exclusion must be a boolean mask in the native RGB domain')
    return mask


def _unexcluded_pixels(mask, uv):
    """Require every contributing bilinear source pixel to be visible."""
    uv = np.asarray(uv).reshape(-1, 2)
    h, w = mask.shape
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    allowed = np.zeros(len(uv), bool)
    lo, hi = np.floor(uv[inside]).astype(int), np.ceil(uv[inside]).astype(int)
    allowed[inside] = ~(mask[lo[:, 1], lo[:, 0]] | mask[lo[:, 1], hi[:, 0]] |
                        mask[hi[:, 1], lo[:, 0]] | mask[hi[:, 1], hi[:, 0]])
    return allowed


def refine_board_readings(rgb, binding, guides, guided, *, authority, feedback, prediction, classes, device, near_board=None):
    import torch
    from .vision_source_topology import read_source_board_rail_material_profiles, read_board_foot_material_profiles
    front_authorized = authority['source_sha256'] == binding['source_sha256']
    authority = feedback.material_authority(authority) if front_authorized else authority
    packets = {}
    for guide in guides:
        proof=guide.get('source_board_view_evidence') or {}
        if proof.get('kind')=='authenticated_original_board_material_frame':
            from .vision_geometry import _same_frame
            source=proof['source_binding']
            # Stored same-exposure semantics must refer to the actual decoded
            # source. Older transported views continue through their RGB flow.
            if source.get('source_pts')==binding['source_pts']:
                _same_frame(binding,source)
                if proof['decoded_native_BGR_sha256']!=hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest():
                    raise ValueError('Stored board material frame differs from the current original RGB')
            else:
                previous=guide.get('hint_previous_source_binding') or {}
                flow=guide.get('temporal_flow_supported')
                valid=(guide.get('origin')=='temporal_source_search_hint' and flow is not None and _numpy(flow).any() and
                    all(source.get(k)==previous.get(k)==binding.get(k) for k in
                        ('source_sha256','time_base','shot_id','installation_id')) and
                    int(source.get('source_pts',-1))<=int(previous.get('source_pts',-1))<int(binding['source_pts']))
                if not valid:raise ValueError('Stored board view needs its own exposure or verified same-shot flow')
    for i, guide in enumerate(guides):
        if guide.get('kind') not in ('board_top','ice_contact') or (front_authorized and guide.get('source_board_view') == 'front_face'):
            continue
        # Do not apply front-face blue/yellow rules to a top deck or an unresolved view.
        # Reviewed ice-side deck readers retain their own native-source evidence.
        row = guided['readings'][i]
        row['unqualified_generic_edge_diagnostic'] = dict(actual_source_uv=row['actual_source_uv'],
            not_physical_board_boundary=True)
        empty = torch.empty((0,2),device=device)
        indices = torch.empty((0,),device=device,dtype=torch.long)
        row.update(actual_source_uv=empty,source_indices=indices,segment_ids=indices,measured_runs=[],
            coherent_source_uv=empty,coherent_source_indices=indices,coherent_measured_runs=[],
            coherent_flags=dict(status='board_view_requires_bound_ice_side_reader',
                source_board_view=guide.get('source_board_view'), generic_blue_band_rule_applied=False))
        row['source_quality_mask'] = torch.zeros(len(guide['points_uv']),device=device,dtype=torch.bool)
        row['source_peak_candidates'] = None
        row['shape_characterization'] = None
        row['metadata'].update(actual_measured_count=0,position_rule='view_specific_material_evidence_required')
        if guide.get('kind') == 'board_top' and guide.get('source_board_view') == 'top_face_only':
            row['view_specific_material_readings'] = (near_board or {}).get('readings',[])
            row['coherent_flags']['status'] = 'top_face_material_intervals_and_points_in_view_specific_readings'
    for kind in ('board_top', 'ice_contact'):
        selected = [(i,g) for i,g in enumerate(guides) if front_authorized and g.get('kind') == kind and g.get('source_board_view') == 'front_face']
        if not selected:
            continue
        points, spans = [], []
        for i, guide in selected:
            uv = guide['points_uv']
            uv = uv.detach().cpu().numpy() if hasattr(uv, 'detach') else np.asarray(uv)
            spans.append((i,len(points),len(points)+len(uv)))
            points.extend(uv)
        points = np.asarray(points, np.float32)
        exclusions = {i: _bound_exclusion_mask(guides[i], binding, rgb.shape[:2]) for i, _, _ in spans}
        material_support = []
        if kind == 'board_top':
            # The existing paired-interface reader treats these as navigation.
            # No old UV becomes an output; blue entry and blue/white exit are read separately.
            packet = read_source_board_rail_material_profiles(rgb,
                dict(source_binding=binding, RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                     actual_source_top_uv=points), sample_spacing_px=1)
            anchor = packet['actual_original_RGB_profiles']['anchors_native_uv']
            evidence = packet['source_material_role_evidence']
            local = np.asarray([e['source_query_index'] for e in evidence], int)
            primary = np.asarray(packet['primary_role_samples'], bool)
            output_uv = packet['actual_source_blue_rail_entry_uv'][primary]
            # Each anchor was selected from an unchanged original source query.
            indices = np.asarray([int(np.argmin(np.linalg.norm(points-anchor[j],axis=1))) for j in local[primary]], dtype=np.int64)
            samples = packet['actual_original_RGB_profiles']['sampled_native_uv']
            for e in (e for e, accept in zip(evidence, primary) if accept):
                entry = e['entry_observed_same_signed_material_transition']['source_indices']
                exit = e['exit_observed_same_signed_material_transition']['source_indices']
                # Each derivative uses two adjacent original RGB samples. Include
                # the selected blue band between both observed transitions.
                material_support.append(samples[e['source_query_index'], int(entry[0]):int(exit[-1])+2])
            method = 'ordered_blue_rail_upper_material_entry'
        else:
            packet = read_board_foot_material_profiles(rgb, dict(source_binding=binding, instances=[]),
                dict(source_binding=binding), material_role_authority=authority, prediction=prediction,
                classes=classes, device=device, source_query_points_uv=points)
            lower = packet['readings'][0]
            output_uv = lower['primary_source_uv'].detach().cpu().numpy()
            indices = lower['primary_source_search_indices'].detach().cpu().numpy()
            samples = _numpy(packet['original_RGB_profiles']['sampled_native_uv'])
            pairs = packet['same_material_band_pair_evidence']
            entry, exit = _numpy(pairs['entry_index']), _numpy(pairs['exit_index'])
            material_support = [samples[j, int(entry[j]):int(exit[j])+2] for j in indices]
            method = 'ordered_yellow_foot_lower_material_exit'
        packet['navigation_UV_are_search_only'] = True
        packet['historical_or_projected_UV_are_new_measurements'] = False
        packet['source_exclusion_admission'] = []
        packets[kind] = packet
        for i, begin, end in spans:
            reading, guide = guided['readings'][i], guides[i]
            selected_rows = np.flatnonzero((indices>=begin)&(indices<end))
            ix, uv = indices[selected_rows]-begin, output_uv[selected_rows]
            mask = exclusions[i]
            excluded = np.zeros(end-begin, bool)
            if mask is not None:
                query_ok = _unexcluded_pixels(mask, points[begin+ix])
                output_ok = _unexcluded_pixels(mask, uv)
                support_ok = np.asarray([len(material_support[j]) > 0 and
                    _unexcluded_pixels(mask, material_support[j]).all() for j in selected_rows], bool)
                keep = query_ok & output_ok & support_ok
                excluded[ix[~keep]] = True
                reading['source_excluded_material_candidates'] = dict(
                    actual_source_uv=uv[~keep].copy(), source_indices=ix[~keep].copy(),
                    query_excluded=~query_ok, output_excluded=~output_ok, material_support_excluded=~support_ok,
                    candidate_source_indices=ix.copy(), raw_optical_candidates_retained=True,
                    excluded_candidates_are_physical_boundaries=False)
                packet['source_exclusion_admission'].append(dict(guide_id=guide.get('id'),
                    source_binding=guide['source_exclusion_binding'], candidate_count=len(ix),
                    excluded_count=int((~keep).sum()), query_excluded_count=int((~query_ok).sum()),
                    output_excluded_count=int((~output_ok).sum()), material_support_excluded_count=int((~support_ok).sum())))
                ix, uv = ix[keep], uv[keep]
            valid = guide.get('temporal_flow_supported')
            if valid is not None:
                valid = valid.detach().cpu().numpy() if hasattr(valid,'detach') else np.asarray(valid)
                keep = valid[ix]; ix,uv = ix[keep],uv[keep]
            order = np.argsort(ix); ix,uv = ix[order],uv[order]
            ix = torch.as_tensor(ix,device=device,dtype=torch.long)
            uv = torch.as_tensor(uv,device=device,dtype=torch.float32)
            segments = torch.as_tensor(guide.get('segment_ids',np.zeros(end-begin,int)),device=device,dtype=torch.long)[ix]
            runs=[]; start=0
            for stop in range(1,len(ix)+1):
                if stop==len(ix) or int(ix[stop])!=int(ix[stop-1])+1 or int(segments[stop])!=int(segments[stop-1]):
                    runs.append(dict(actual_source_uv=uv[start:stop],source_indices=ix[start:stop],segment_id=int(segments[start])))
                    start=stop
            reading['unqualified_generic_edge_diagnostic'] = dict(actual_source_uv=reading['actual_source_uv'],
                coherent_source_uv=reading.get('coherent_source_uv'), not_physical_board_boundary=True)
            reading.update(actual_source_uv=uv, source_indices=ix, segment_ids=segments, measured_runs=runs,
                coherent_source_uv=uv,coherent_source_indices=ix,coherent_measured_runs=runs,
                coherent_flags=dict(status=method, world_geometry_modified=False, reflection_tail_is_not_contact=True),
                support=dict(method=method, material_packet_kind=kind, source_role_authority=authority['reference']))
            reading['source_peak_candidates'] = None
            reading['shape_characterization'] = None
            quality = torch.zeros(end-begin,device=device,dtype=torch.bool); quality[ix] = True
            reading['source_quality_mask'] = quality
            reading['source_quality_weights'] = None
            reading['flags'] = dict(ordered_material_role_unresolved=~quality,
                source_exclusion_conflict=torch.as_tensor(excluded,device=device))
            reading['metadata'].update(actual_measured_count=len(ix), position_rule=method,
                generic_gradient_or_semantic_peak_is_physical_boundary=False,
                source_exclusion_checked=mask is not None, source_excluded_material_count=int(excluded.sum()))
    return dict(status='ordered_current_RGB_board_materials_read',material_packets=packets,
        blue_white_internal_edge_is_top=False,yellow_upper_edge_is_contact=False,
        reflection_tail_is_not_contact=True,source_pixels_measured=True,geometry_modified=False)


def board_face_admission(guides, guided, native_hw, *, device):
    """Same conservative visible-run interior policy as native_board_material_support.

    A top deck alone encloses no vertical ad face. Gaps are never bridged,
    and no far-side ad can be admitted from the near top edge.
    """
    import torch
    mask=np.zeros(native_hw,bool); boundaries={'board_top':[],'ice_contact':[]}
    for guide,row in zip(guides,guided['readings']):
        kind=guide.get('kind')
        if kind not in boundaries or guide.get('source_board_view')!='front_face':continue
        for run in row.get('coherent_measured_runs',row.get('measured_runs',[])):
            uv=run['actual_source_uv']
            uv=uv.detach().cpu().numpy() if hasattr(uv,'detach') else np.asarray(uv)
            if len(uv)<2:continue
            if not (np.all(np.diff(uv[:,0])>0) or np.all(np.diff(uv[:,0])<0)):continue
            boundaries[kind].append(uv)
    regions=[]
    for top in boundaries['board_top']:
        for foot in boundaries['ice_contact']:
            x0=max(top[:,0].min(),foot[:,0].min());x1=min(top[:,0].max(),foot[:,0].max())
            y0=top[:,1].max();y1=foot[:,1].min()
            if x0>=x1 or y0>=y1:continue
            # Strict interior only. Border sample pixels do not establish ad material.
            left=max(0,int(np.floor(x0))+1);right=min(native_hw[1],int(np.ceil(x1)))
            upper=max(0,int(np.floor(y0))+1);lower=min(native_hw[0],int(np.ceil(y1)))
            mask[upper:lower,left:right]=True
            regions.append([left,upper,right,lower])
    return torch.as_tensor(mask,device=device)[None,None],dict(
        visible_front_face_interiors_xyxy=regions,unknown_or_top_face_only_admission=False,
        policy='visible_run_intersection_no_gap_bridge_no_extrapolation',
        search_mask_is_not_geometry_or_new_UV=True)


def filter_ad_readings(guides,guided,allowed):
    import torch
    h,w=allowed.shape[-2:]
    for guide,row in zip(guides,guided['readings']):
        if guide.get('kind')!='ad_boundary':continue
        uv=row.get('coherent_source_uv',row['actual_source_uv'])
        index=row.get('coherent_source_indices',row['source_indices'])
        ij=uv.round().long()
        inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        keep=torch.zeros(len(uv),device=uv.device,dtype=torch.bool)
        keep[inside]=allowed[0,0,ij[inside,1],ij[inside,0]]
        row['unverified_board_face_candidate_uv']=uv[~keep]
        uv,index=uv[keep],index[keep]
        segment=torch.as_tensor(guide.get('segment_ids',np.zeros(len(guide['points_uv']),int)),device=uv.device)[index]
        runs=[];a=0
        for b in range(1,len(index)+1):
            if b==len(index) or index[b]!=index[b-1]+1 or segment[b]!=segment[b-1]:
                runs.append(dict(actual_source_uv=uv[a:b],source_indices=index[a:b],segment_id=int(segment[a])))
                a=b
        row.update(actual_source_uv=uv,source_indices=index,segment_ids=segment,measured_runs=runs,
            coherent_source_uv=uv,coherent_source_indices=index,coherent_measured_runs=runs,
            source_peak_candidates=None)
        quality=torch.zeros(len(guide['points_uv']),device=uv.device,dtype=torch.bool)
        quality[index]=True
        row['source_quality_mask']=quality
        weights=row.get('source_quality_weights')
        if weights is not None and len(weights)==len(quality):weights[~quality]=0
        row.setdefault('flags',{})['vertical_face_unverified']=~quality
        row['shape_characterization']=None
        row['metadata'].update(actual_measured_count=len(uv),current_vertical_board_face_required=True)


def dense_front_face_admission(rgb, binding, guides, guided, exclusion, *, device):
    """Connect existing paired RGB bands to already identified front-face reads.

    Sparse named guide queries must not erase an otherwise visible board face.
    The existing dense reader measures both materials again on this exposure;
    matching named boundary reads supply identity. Neither creates output UV.
    """
    import torch
    from calibration.program_tracking.source_board_measurements import measure_board_source_frame
    mask = np.zeros(rgb.shape[:2], bool)
    named = {'board_top': [], 'ice_contact': []}
    for guide, row in zip(guides, guided['readings']):
        kind = guide.get('kind')
        if kind not in named or guide.get('source_board_view') != 'front_face':
            continue
        uv = _numpy(row.get('coherent_source_uv', row['actual_source_uv'])).reshape(-1, 2)
        if len(uv):
            named[kind].append((guide['id'], uv, float(guide.get('search_radius_px', 18.))))
    if rgb.shape[:2] != (1080, 1920) or not all(named.values()):
        return torch.as_tensor(mask, device=device)[None,None], dict(status='no_paired_named_front_face'), None
    bgr = np.ascontiguousarray(rgb[:,:,::-1])
    packet = measure_board_source_frame(bgr, exclusion, dict(binding,
        actual_pts=int(binding['source_pts']), native_pts=int(binding['source_pts']),
        decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
    regions = []
    for band in packet['boundary_candidates']:
        top, foot = band['blue_upper_uv'], band['yellow_lower_uv']
        matched = {}
        for kind, boundary in (('board_top',top),('ice_contact',foot)):
            for name, uv, radius in named[kind]:
                in_span = (uv[:,0] >= boundary[0,0]) & (uv[:,0] <= boundary[-1,0])
                local = uv[in_span]
                # The dense band already contains no >29px gaps. This
                # interpolation checks correspondence, never supplies readout UV.
                if len(local) and np.any(np.abs(local[:,1]-np.interp(local[:,0],boundary[:,0],boundary[:,1])) <= radius):
                    matched.setdefault(kind,[]).append(name)
        if len(matched) != 2:
            continue
        x0, x1 = float(top[:,0].min()), float(top[:,0].max())
        upper = np.max(top[:,1]+np.max(np.abs(band['blue_transition_interval_y_px']),axis=1))
        lower = np.min(foot[:,1]-np.max(np.abs(band['yellow_transition_interval_y_px']),axis=1))
        left,right=max(0,int(np.floor(x0))+1),min(rgb.shape[1],int(np.ceil(x1)))
        first,last=max(0,int(np.floor(upper))+1),min(rgb.shape[0],int(np.ceil(lower)))
        if left >= right or first >= last:
            continue
        mask[first:last,left:right] = True
        regions.append(dict(candidate_id=band['candidate_id'], rectangle_xyxy=[left,first,right,last],
            named_source_boundaries=matched))
    mask &= ~exclusion
    return torch.as_tensor(mask,device=device)[None,None], dict(
        status='current_RGB_paired_bands_bound_to_named_front_face', regions=regions,
        source_binding=binding, gaps_bridged=False, source_UV_created=False,
        detector_candidates_alone_certify_identity=False), packet


def read_connected_board_gaps(rgb, binding, guides, guided, *, source_packet, face_receipt,
        geometry, authority, feedback, prediction, classes, device, exclusion):
    """Read missing parts of an actually paired and already identified face.

    The existing source detector supplies blue/yellow material searches. Both
    sides must correspond to current named physical readouts in the same run;
    an unverified projection or an isolated blue strip cannot authorize a face.
    Every appended output is freshly read by the ordered material readers.
    """
    import torch
    from .vision_geometry import _same_frame
    from .vision_navigation import _uncovered_query
    report=dict(added_query_count=0,added_actual_point_count=0,
        source_binding=dict(binding),front_face_inferred_from_projection=False,
        existing_readings_rewritten=False,source_query_UV_used_as_output=False)
    if source_packet is None or geometry is None:return report
    source=source_packet.get('source',{})
    _same_frame(binding,dict(source,source_pts=source.get('source_pts',source.get('native_pts'))))
    if source.get('decoded_BGR_sha256') != hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest():
        raise ValueError('Connected board searches must come from this exact original RGB')
    eligible={g['id']:g for g,r in zip(guides,guided['readings'])
        if g.get('source_board_view')=='front_face' and g.get('identity_confirmed') is True
        and g.get('identity_authority') and r.get('physical_id') and len(r['actual_source_uv'])}
    paired={r['candidate_id']:r for r in face_receipt.get('regions',[])
        if all(any(name in eligible for name in r['named_source_boundaries'].get(kind,[]))
               for kind in ('board_top','ice_contact'))}
    retained=[]
    for g,r in zip(guides,guided['readings']):
        points=np.asarray(_numpy(g['points_uv']),float).copy()
        indices=np.asarray(_numpy(r['source_indices']),int)
        points[indices]=_numpy(r['actual_source_uv'])
        retained.append(dict(g,points_uv=points,last_measured_query_indices=indices))
    additions=[]
    for band in source_packet.get('boundary_candidates',[]):
        anchor=paired.get(band['candidate_id'])
        if anchor is None:continue
        for kind,field,identity in (('board_top','blue_upper_uv','board.top'),
                                    ('ice_contact','yellow_lower_uv','board.bottom')):
            definition=geometry.lookup(identity)
            if definition is None or definition.get('layer')!='board':continue
            uv=np.asarray(band[field],float).reshape(-1,2)
            if len(uv)<2:continue
            tangent=np.gradient(uv,axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]]
            normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
            proof=dict(kind='current_RGB_connected_board_material_pair',source_binding=dict(binding),
                source_material_reference=authority['reference'],source_candidate_id=band['candidate_id'],
                current_named_boundaries=anchor['named_source_boundaries'],
                source_frame_anchor_reading_indices=[i for i,g in enumerate(guides)
                    if g['id'] in anchor['named_source_boundaries']['board_top']
                    or g['id'] in anchor['named_source_boundaries']['ice_contact']],
                decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                physical_identity_from_model_projection=False,human_pixel_acceptance=False)
            query=dict(id=definition['id'],kind=kind,points_uv=uv,normals=normal,
                segment_ids=np.zeros(len(uv),int),source_binding=dict(binding),
                query_id=definition['id']+'/source_material_run/'+band['candidate_id'],
                origin='source_annotation_search_hint',identity_confirmed=True,identity_authority=proof,
                source_board_view='front_face',source_board_view_evidence=proof,
                source_exclusion_mask=np.asarray(exclusion,bool).copy(),source_exclusion_binding=dict(binding),
                search_radius_px=18.,sample_spacing_px=28.,candidate_navigation_only=True,
                source_group_id='current_board_material:'+band['candidate_id']+':'+kind)
            query=_uncovered_query(query,retained,geometry=geometry)
            if query is not None:
                additions.append(query);retained.append(query)
    additions,native_binding=geometry.bind(additions)
    if native_binding.get('unresolved'):raise ValueError('Connected board query lost its current native path')
    fresh=dict(readings=[dict(expected_id=g['id'],physical_id=g['id'],kind=g['kind'],
        primitive=g['primitive'],physical_role=g['physical_role'],identity_authority=g['identity_authority'],
        origin=g['origin'],actual_source_uv=torch.empty((0,2),device=device),
        source_indices=torch.empty(0,device=device,dtype=torch.long),
        guide=dict(uv=torch.as_tensor(g['points_uv'],device=device),normals=torch.as_tensor(g['normals'],device=device),
            segment_ids=torch.as_tensor(g['segment_ids'],device=device,dtype=torch.long),
            closed=False,source_binding=dict(binding)),metadata=dict(source_point_count=len(g['points_uv']),
                guide_position_is_measurement=False,localization_accuracy_validated=False)) for g in additions])
    if additions:
        report['material_reread']=refine_board_readings(rgb,binding,additions,fresh,authority=authority,feedback=feedback,
            prediction=prediction,classes=classes,device=device)
        guides.extend(additions);guided['readings'].extend(fresh['readings'])
    report.update(added_query_count=len(additions),
        added_actual_point_count=sum(len(r['actual_source_uv']) for r in fresh['readings']),
        native_binding=native_binding)
    return report


def continue_front_face_admission(current, previous, binding, previous_binding, flow_pair, exclusion):
    """Transport a verified face search region, never old measurement pixels."""
    from .vision_geometry import _same_frame, warp
    report=dict(status='no_previous_verified_front_face',source_binding=binding,
        source_UV_created=False,previous_UV_used_as_current_measurements=False,
        current_RGB_reread_required=True)
    if previous is None or flow_pair is None or previous_binding is None:
        return current, report
    pair=flow_pair.metadata.get('source_bindings')
    if not pair or not flow_pair.metadata.get('source_alignment_verified'):
        raise ValueError('Front-face continuity requires authenticated actual RGB flow')
    _same_frame(previous_binding,pair[0]);_same_frame(binding,pair[1])
    if any(binding[k]!=previous_binding[k] for k in ('source_sha256','time_base','shot_id')) or binding['source_pts']<=previous_binding['source_pts']:
        raise ValueError('Front-face continuity must stay within the same source shot')
    transported,inside=warp(previous.float(),flow_pair.backward)
    # Every bilinear contributor must belong to the previously measured face.
    # Unknown flow, new occlusions and a current top-face exclusion stay closed.
    support=(transported>=1.-1e-6)&inside&flow_pair.valid_backward&~exclusion
    report.update(status='actual_RGB_flow_continues_verified_front_face_search',
        previous_source_binding=previous_binding,flow_source_bindings=pair,
        flow_weights_sha256=flow_pair.metadata.get('weights_sha256'),
        carried_search_pixel_count=int(support.sum()),added_search_pixel_count=int((support&~current).sum()),
        current_exclusions_reapplied=True)
    return (current|support)&~exclusion,report
