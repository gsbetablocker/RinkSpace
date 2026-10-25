"""Use the existing ordered material readers for both physical board boundaries."""
import hashlib
import numpy as np


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
