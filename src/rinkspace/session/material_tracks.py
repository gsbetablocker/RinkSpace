"""Competing source-material fragments and bidirectional image links.

Curve names organize searches, not identities. Every fragment stores indices
into unchanged RGB readouts. Geometry never supplies measurement coordinates.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rinkspace.vision_ice_temporal_identity import _measured_material_descriptors


def assemble_fragments(measurement, inventory, eligible_candidates=None):
    """Retain coherent competing branches, including ones far from the guide.

    A short-gap DAG joins source profiles. The query strip supplies order and
    a local normal only. A long material branch beats an isolated closer
    reading; all other supported branches remain as alternatives.
    """
    uv = np.asarray(measurement['candidate_uv'], float)
    query = np.asarray(measurement['candidate_query'], int)
    quv = np.asarray(measurement['query_uv'], float)
    qnormal = np.asarray(measurement['query_normal'], float)
    curve = np.asarray(measurement['query_curve'], int)[query]
    sides = np.asarray(measurement['candidate_sides'], float)
    width = np.linalg.norm(sides[:, 1]-sides[:, 0], axis=1)
    clean = (np.asarray(measurement['candidate_flags']) & 3) == 0
    if eligible_candidates is not None:
        eligibility=np.asarray(eligible_candidates,bool)
        if eligibility.shape!=clean.shape:raise ValueError('Candidate eligibility shape mismatch')
        clean &= eligibility
    displacement = np.einsum('ij,ij->i', uv-quv[query], qnormal[query])
    fragments = []
    for ci in np.unique(curve[clean]):
        curve_queries = np.flatnonzero(np.asarray(measurement['query_curve']) == ci)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(quv[curve_queries], axis=0), axis=1))]
        candidate_arc = np.zeros(len(uv))
        curve_candidates = np.flatnonzero(curve == ci)
        candidate_arc[curve_candidates] = arc[np.searchsorted(curve_queries, query[curve_candidates])]
        remaining = np.flatnonzero(clean & (curve == ci))
        remaining = remaining[np.argsort(query[remaining], kind='stable')]
        while len(remaining) >= 4:
            score = np.ones(len(remaining))
            parent = np.full(len(remaining), -1, int)
            for j, item in enumerate(remaining):
                gap = candidate_arc[item]-candidate_arc[remaining[:j]]
                earlier = np.flatnonzero((gap > 0)&(gap <= 105.))
                if not len(earlier): continue
                old = remaining[earlier]
                distance = np.linalg.norm(quv[query[old]]-quv[query[item]], axis=1)
                offset = np.abs(displacement[old]-displacement[item])
                ratio = width[item]/np.maximum(width[old], 1e-6)
                compatible = ((width[item] <= 1e-6) & (width[old] <= 1e-6)) | ((ratio >= .5) & (ratio <= 2.))
                good = (distance <= 105.) & (offset <= 3.+.08*distance) & compatible
                if not good.any(): continue
                earlier, distance, offset = earlier[good], distance[good], offset[good]
                # Smoothness is for fragment organization, never a fit weight.
                proposed = score[earlier]+1.-.025*offset-.003*np.maximum(distance-35., 0.)
                best = int(np.argmax(proposed))
                if proposed[best] > score[j]:
                    score[j] = proposed[best]; parent[j] = earlier[best]
            at = int(np.argmax(score)); path = []
            while at >= 0:
                path.append(int(remaining[at])); at = int(parent[at])
            path = np.asarray(path[::-1], int)
            if len(path) < 4: break
            span = float(np.linalg.norm(np.diff(quv[query[path]], axis=0), axis=1).sum())
            if span < 70.: break
            fragments.append(dict(curve=int(ci), candidate_ids=path.tolist(),
                span_px=span, normal_offset_median=float(np.median(displacement[path])),
                query_first=int(query[path[0]]), query_last=int(query[path[-1]]),
                physical_identity=None))
            remaining = remaining[~np.isin(remaining, path)]
    return fragments


def qualify_line_candidates(measurement, curve_indices, source_exclusion_mask=None, *,
                            require_paired_boundaries=False, native_shape_guided=False,
                            use_source_quantiles=False, source_support_queries=None,
                            source_band_unmasked=None):
    """Reuse whole-source line-role organization without changing measured UV.

    The caller must establish that each selected physical primitive is a line.
    This step uses actual alternative readouts across the visible source span;
    it supplies neither physical identity nor native model residual filtering.
    """
    from rinkspace.vision_source_roles import organize_source_line_material_branches
    query=np.asarray(measurement['candidate_query'],int)
    curves=np.asarray(measurement['query_curve'],int)
    clean=(np.asarray(measurement['candidate_flags'])&3)==0
    uv=np.asarray(measurement['candidate_uv'])
    if require_paired_boundaries:
        sides=np.asarray(measurement['candidate_sides'],float)
        if sides.shape!=(len(uv),2,2):raise ValueError('Paired source boundaries have wrong shape')
        clean &= np.isfinite(sides).all(axis=(1,2))
        clean &= np.linalg.norm(sides[:,1]-sides[:,0],axis=1)>1e-6
        clean &= np.linalg.norm(sides.mean(1)-uv,axis=1)<1e-3
    quantiles=None
    if use_source_quantiles:
        quantiles=np.asarray(measurement['candidate_side_quantiles_uv'],float)
        if quantiles.shape!=(len(uv),2,3,2) or not np.isfinite(quantiles).all():
            raise ValueError('Actual source boundary q25/q50/q75 required')
    hidden=np.zeros(len(curves),bool)
    support=np.ones(len(curves),bool)
    if source_support_queries is not None:
        support=np.asarray(source_support_queries,bool)
        if support.shape!=(len(curves),):raise ValueError('Source support queries must match query rows')
    if source_exclusion_mask is not None and len(curves):
        mask=np.asarray(source_exclusion_mask,bool)
        if mask.ndim!=2:raise ValueError('Source exclusion mask must be an image')
        if require_paired_boundaries and use_source_quantiles:
            from rinkspace.session.source_shape_context import measured_band_unmasked
            # A clear midpoint cannot qualify a band whose measured ink or
            # outer transitions intersect the source exclusion mask.
            if source_band_unmasked is None:
                band_unmasked=measured_band_unmasked(quantiles,mask)
            else:
                band_unmasked=np.asarray(source_band_unmasked)
                if band_unmasked.shape!=(len(uv),) or band_unmasked.dtype!=np.bool_:
                    raise ValueError('Cached source band mask must be boolean per candidate')
            clean &= band_unmasked
        intervals=np.asarray(measurement['query_interval'],float)
        offsets=np.arange(np.floor(intervals.min()),np.ceil(intervals.max())+1.)
        locations=(np.asarray(measurement['query_uv'])[:,None]+
                   np.asarray(measurement['query_normal'])[:,None]*offsets[None,:,None])
        xy=np.rint(locations).astype(int)
        valid=(offsets[None]>=intervals[:,0,None])&(offsets[None]<=intervals[:,1,None])
        valid&=(xy[:,:,0]>=0)&(xy[:,:,0]<mask.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<mask.shape[0])
        excluded=mask[np.clip(xy[:,:,1],0,mask.shape[0]-1),np.clip(xy[:,:,0],0,mask.shape[1]-1)]
        hidden=valid.any(1)&np.all(~valid|excluded,axis=1)
    selected=[];reports=[]
    for curve in curve_indices:
        queries=np.flatnonzero((curves==curve)&support)
        occluded=queries[hidden[queries]]
        queries=queries[~hidden[queries]]
        if not len(queries):
            reports.append(dict(curve=int(curve),disposition='no_visible_source_queries',selected_candidate_ids=[]))
            continue
        positions=np.asarray(measurement['query_uv'])[queries]
        normal=np.asarray(measurement['query_normal'])[queries]
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
        def source_position(value,local):
            if not native_shape_guided:return np.asarray(value).tolist()
            delta=np.asarray(value)-positions[local]
            tangent=np.array([normal[local,1],-normal[local,0]])
            return [float(arc[local]+delta@tangent),float(delta@normal[local])]
        def source_item(i,local,side=None):
            value=uv[i] if side is None else sides[i,side]
            item=dict(actual_source_uv=source_position(value,local),disposition='measured_material_transition')
            # A broad transition is useful for associating its two material
            # sides, but does not permit a midpoint to wander off a straight
            # mark. The center always keeps the source geometry tolerance.
            if quantiles is not None and side is not None:
                q=quantiles[i].mean(0) if side is None else quantiles[i,side]
                offsets=(q[[0,2]]-value)@normal[local]
                item.update(observed_normal_interval_px=np.sort(offsets).tolist(),
                    source_search_normal=[0.,1.] if native_shape_guided else normal[local].tolist())
            return item
        search_positions=(np.c_[arc,np.zeros(len(arc))] if native_shape_guided else positions)
        navigation=(np.tile([1.,0.],(len(queries),1)) if native_shape_guided else np.c_[-normal[:,1],normal[:,0]])
        rows=[];ids=[]
        for local_query,qi in enumerate(queries):
            candidates=np.flatnonzero(clean & (query==qi));ids.append(candidates)
            rows.append(dict(source_search_point_uv=search_positions[local_query].tolist(),
                all_actual_source_material_candidates=[source_item(i,local_query) for i in candidates]))
        # A failed material reading is negative/missing source support, not
        # evidence of occlusion. Count its actual visible length. Only a fully
        # excluded search profile can split the physical support denominator.
        starts=np.r_[0,np.flatnonzero(np.diff(queries)!=1)+1]
        stops=np.r_[starts[1:],len(queries)]
        visible=[list(range(int(a),int(b))) for a,b in zip(starts,stops)]
        result=organize_source_line_material_branches(rows,
            navigation_tangent_uv=navigation,source_visible_paths=visible)
        accepted=[int(ids[int(qi)][int(branch)]) for qi,branch in
                  zip(result['source_query_indices'],result['source_candidate_indices'])]
        boundary_reports=[]
        if require_paired_boundaries and accepted:
            # A plausible midpoint can conceal edges belonging to different
            # painted bands. Require each measured side to support the same
            # whole line. Missing/ambiguous samples still count in its visible
            # denominator; source fits select pixels but never replace them.
            sides=np.asarray(measurement['candidate_sides'],float)
            if sides.shape!=(len(uv),2,2) or not np.isfinite(sides).all():
                raise ValueError('Paired source boundaries must be finite 2D pixels')
            center_ids=set(accepted)
            supported=set(accepted)
            for side in range(2):
                edge_rows=[];edge_ids=[]
                for local_query,qi in enumerate(queries):
                    candidates=np.asarray([i for i in ids[local_query] if i in center_ids],int)
                    edge_ids.append(candidates)
                    edge_rows.append(dict(source_search_point_uv=search_positions[local_query].tolist(),
                        all_actual_source_material_candidates=[source_item(i,local_query,side) for i in candidates]))
                edge_result=organize_source_line_material_branches(edge_rows,
                    navigation_tangent_uv=navigation,source_visible_paths=visible)
                side_ids={int(edge_ids[int(qi)][int(branch)]) for qi,branch in
                          zip(edge_result['source_query_indices'],edge_result['source_candidate_indices'])}
                supported &= side_ids
                boundary_reports.append(dict(side=side,disposition=edge_result['disposition'],
                    source_line_support_qualified=edge_result['source_line_support_qualified'],
                    selected_candidate_ids=sorted(side_ids),
                    source_organizer=edge_result.get('actual_source_line_organizer')))
            accepted=[i for i in accepted if i in supported]
            joint_rows=[];joint_ids=[]
            for local_query,qi in enumerate(queries):
                candidates=np.asarray([i for i in ids[local_query] if i in supported],int)
                joint_ids.append(candidates)
                joint_rows.append(dict(source_search_point_uv=search_positions[local_query].tolist(),
                    all_actual_source_material_candidates=[source_item(i,local_query) for i in candidates]))
            joint=organize_source_line_material_branches(joint_rows,
                navigation_tangent_uv=navigation,source_visible_paths=visible)
            accepted=[int(joint_ids[int(qi)][int(branch)]) for qi,branch in
                      zip(joint['source_query_indices'],joint['source_candidate_indices'])]
        qualified=bool(result['source_line_support_qualified'] and accepted)
        disposition=result['disposition'] if qualified or not result['source_line_support_qualified'] else 'paired_source_boundaries_not_supported'
        selected.extend(accepted)
        reports.append(dict(curve=int(curve),disposition=disposition,
            selected_candidate_ids=accepted,source_line_support_qualified=qualified,
            paired_source_boundaries_required=require_paired_boundaries,boundary_reports=boundary_reports,
            source_organizer=result.get('actual_source_line_organizer'),
            visible_source_queries=queries.tolist(),fully_excluded_source_queries=occluded.tolist(),
            missing_readouts_count_as_visible_source_length=True,
            source_support_restricted_to_proposed_intervals=source_support_queries is not None,
            source_geometry_tolerance_px=result['centerline_role_tolerance_px'],
            native_fit_residual_used=False,fitted_line_used_as_observation=False))
        reports[-1]['source_geometry_domain']='native_normal_strip' if native_shape_guided else 'source_image'
        reports[-1]['source_uv_unchanged']=True
        reports[-1]['actual_boundary_quantiles_used_for_source_organization']=use_source_quantiles
    return np.asarray(sorted(set(selected)),np.int64),reports


def source_line_fragments(measurement, reports):
    """Represent each unique observed line as one entity across visible gaps.

    The source-role organizer has already compared whole-line alternatives.
    Occlusion breaks remain explicit; only its unchanged actual readouts become
    nodes. A line identity may connect visible pieces without inventing material
    points between them or a tangent correspondence across a gap.
    """
    result=[]
    query=np.asarray(measurement['candidate_query'],int)
    qcurve=np.asarray(measurement['query_curve'],int)
    quv=np.asarray(measurement['query_uv'])
    uv=np.asarray(measurement['candidate_uv'])
    normal=np.asarray(measurement['query_normal'])
    for report in reports:
        if not report.get('source_line_support_qualified'):continue
        ids=np.asarray(report['selected_candidate_ids'],int)
        if not len(ids):continue
        if not np.all(qcurve[query[ids]]==report['curve']):
            raise ValueError('Whole source line mixes query primitives')
        ids=ids[np.argsort(query[ids],kind='stable')];queries=query[ids]
        starts=np.r_[0,np.flatnonzero(np.diff(queries)!=1)+1];stops=np.r_[starts[1:],len(ids)]
        pieces=[ids[a:b].tolist() for a,b in zip(starts,stops)]
        span=sum(float(np.linalg.norm(np.diff(quv[query[p]],axis=0),axis=1).sum()) for p in pieces)
        displacement=np.einsum('ij,ij->i',uv[ids]-quv[queries],normal[queries])
        result.append(dict(curve=int(report['curve']),candidate_ids=ids.tolist(),
            source_visible_candidate_pieces=pieces,span_px=span,
            query_first=int(queries[0]),query_last=int(queries[-1]),
            normal_offset_median=float(np.median(displacement)),physical_identity=None,
            organization='unique_whole_source_line_family',unseen_source_pixels_created=False))
    return result


def qualify_visible_line_fragments(measurement, curve_indices, source_exclusion_mask=None, *,
                                   fragments=None):
    """Keep locally supported paired paint spans when a whole line is incomplete.

    A candidate chain proposes an observed interval, never fills its gaps. All
    competing RGB candidates inside that interval participate in qualification.
    Overlapping intervals that select different bands at one query stay unknown.
    Identity is still supplied by the source-image temporal graph, not this test.
    Explicit fragments reuse the caller's unchanged full-frame assembly.
    """
    options=dict(require_paired_boundaries=True,native_shape_guided=True,use_source_quantiles=True)
    band_unmasked=None
    if source_exclusion_mask is not None and len(measurement['query_curve']):
        from rinkspace.session.source_shape_context import measured_band_unmasked
        # Every local proposal retains original candidate row indices. Cache
        # the same measured-band check once, then only slice its result.
        band_unmasked=measured_band_unmasked(measurement['candidate_side_quantiles_uv'],source_exclusion_mask)
    selected,reports=qualify_line_candidates(measurement,curve_indices,source_exclusion_mask,
        source_band_unmasked=band_unmasked,**options)
    whole=source_line_fragments(measurement,reports)
    resolved={f['curve'] for f in whole}
    query=np.asarray(measurement['candidate_query'],int)
    qcurve=np.asarray(measurement['query_curve'],int)
    proposals=(assemble_fragments(measurement,[{}]*(int(qcurve.max())+1 if len(qcurve) else 0))
        if fragments is None else fragments)
    local_results=[]; selected_by_curve=defaultdict(list)
    for fragment in proposals:
        ci=fragment['curve']
        if ci not in curve_indices or ci in resolved:continue
        begin,stop=fragment['query_first'],fragment['query_last']+1
        rows=np.flatnonzero((query>=begin)&(query<stop))
        local={}
        for key,value in measurement.items():
            if not isinstance(value,np.ndarray):continue
            if key.startswith('query_'):local[key]=value[begin:stop]
            elif key.startswith('candidate_'):local[key]=value[rows]
            else:local[key]=value
        local['candidate_query']=local['candidate_query']-begin
        picked,qualification=qualify_line_candidates(local,[ci],source_exclusion_mask,
            source_band_unmasked=None if band_unmasked is None else band_unmasked[rows],**options)
        original=rows[picked]
        if len(original)<4:continue
        span=np.linalg.norm(np.diff(measurement['query_uv'][query[original]],axis=0),axis=1).sum()
        if span<70.:continue
        selected_by_curve[ci].append(set(map(int,original)))
        local_results.append(dict(curve=ci,query_interval=[begin,stop],selected_candidate_ids=original.tolist(),
            qualification=qualification))
    coherent_reports=[]
    for ci,groups in selected_by_curve.items():
        # Local intervals establish visible material support, not distinct
        # permission to attach any straight fragment to this physical line.
        # Compare every RGB alternative on their union in one common source
        # geometry. Separate offsets/slopes cannot inherit the same identity
        # merely because each short piece is locally straight.
        support=np.zeros(len(qcurve),bool)
        for item in local_results:
            if item['curve']==ci:
                begin,stop=item['query_interval'];support[begin:stop]=True
        _,qualification=qualify_line_candidates(measurement,[ci],source_exclusion_mask,
            source_support_queries=support,source_band_unmasked=band_unmasked,**options)
        coherent_reports.extend(qualification)
        for fragment in source_line_fragments(measurement,qualification):
            # Distributed intervals and all RGB alternatives already qualified.
            # This display span excludes true gaps between accepted pieces.
            if len(fragment['candidate_ids'])<4:continue
            fragment['organization']='one_source_line_across_supported_intervals'
            whole.append(fragment)
    return whole,dict(whole_line_reports=reports,local_source_intervals=local_results,
        coherent_source_interval_reports=coherent_reports,
        missing_readouts_not_filled=True,source_uv_unchanged=True)


def source_line_reference_support(reference_uv, measured_uv, query_indices):
    """Match a known contour's source support without equating tangent samples.

    Both inputs are actual source pixels. Only consecutive measured queries
    create finite segments; unobserved gaps and endpoint extensions provide no
    support. The same finite source segments also represent observed arcs;
    this establishes an identity overlap, never a new measured UV.
    """
    from rinkspace.session.native_shape_guidance import point_to_partial_polyline
    reference=np.asarray(reference_uv,float).reshape(-1,2)
    measured=np.asarray(measured_uv,float).reshape(-1,2)
    queries=np.asarray(query_indices,int)
    if queries.shape!=(len(measured),) or not np.isfinite(reference).all() or not np.isfinite(measured).all():
        raise ValueError('Finite source pixels and matching source queries required')
    reference=np.unique(reference,axis=0)
    empty=dict(count=0,span_px=0.,qualified=False,matched_reference_uv=[])
    if len(reference)<3 or len(measured)<2:return empty
    order=np.argsort(queries,kind='stable');queries=queries[order];measured=measured[order]
    if len(np.unique(queries))!=len(queries):raise ValueError('Competing candidates cannot seed one source contour')
    pieces=[];has_segment=False
    for i,point in enumerate(measured):
        if i and queries[i]!=queries[i-1]+1:pieces.append([np.nan,np.nan])
        elif i:has_segment|=bool(np.linalg.norm(point-measured[i-1])>0)
        pieces.append(point)
    if not has_segment:return empty
    match=point_to_partial_polyline(reference,np.asarray(pieces))
    supported=reference[match.distance_px<=3.]
    span=float(np.linalg.norm(np.ptp(supported,axis=0))) if len(supported) else 0.
    return dict(count=len(supported),span_px=span,qualified=len(supported)>=3 and span>=56.,
                matched_reference_uv=supported.tolist())


def source_nodes(rgb, measurement, fragments, inventory):
    """One node per exact original candidate, carrying real RGB descriptors."""
    ids = np.unique([i for f in fragments for i in f['candidate_ids']]).astype(int)
    query = np.asarray(measurement['candidate_query'], int)[ids]
    uv = np.asarray(measurement['candidate_uv'], float)[ids]
    sides = np.asarray(measurement['candidate_sides'], float)[ids]
    normal = np.asarray(measurement['query_normal'], float)[query]
    curves = np.asarray(measurement['query_curve'], int)[query]
    width = np.linalg.norm(sides[:, 1]-sides[:, 0], axis=1)
    descriptor, strength = _measured_material_descriptors(rgb, uv, sides, normal)
    family_names = sorted({(p['operator'], p['material_signal'], str(p.get('material_transition'))) for p in inventory})
    family = np.array([family_names.index((inventory[c]['operator'], inventory[c]['material_signal'],
        str(inventory[c].get('material_transition')))) for c in curves], np.int16)
    groups = np.full(len(ids), -1, np.int32)
    for fi, f in enumerate(fragments): groups[np.searchsorted(ids, f['candidate_ids'])] = fi
    return dict(ids=ids, uv=uv, sides=sides, normal=normal, width=width,
        descriptor=descriptor.astype(np.float32), strength=strength,
        family=family, curve=curves, fragment=groups)


def same_frame_conflicts(nodes, fragments):
    """Expose competing curve names that query the same measured source line."""
    if not len(nodes['uv']): return []
    pairs = cKDTree(nodes['uv']).query_pairs(14., output_type='ndarray')
    votes = defaultdict(set)
    for a, b in pairs:
        fa, fb = int(nodes['fragment'][a]), int(nodes['fragment'][b])
        if fa == fb or nodes['curve'][a] == nodes['curve'][b]: continue
        if nodes['family'][a] != nodes['family'][b]: continue
        delta = nodes['uv'][a]-nodes['uv'][b]
        if abs(float(delta@nodes['normal'][a])) > 2.: continue
        if abs(float(nodes['normal'][a]@nodes['normal'][b])) < .97: continue
        if nodes['descriptor'][a]@nodes['descriptor'][b] < .85: continue
        votes[tuple(sorted((fa, fb)))].add(int(a))
    return [dict(a=a, b=b, supporting_readouts=len(v),
                 reason='same_source_material_queried_under_competing_names')
            for (a, b), v in votes.items() if len(v) >= 3]


def _warp(h, uv):
    homogeneous = np.c_[uv, np.ones(len(uv))]@np.asarray(h).T
    return homogeneous[:, :2]/homogeneous[:, 2:]


def normal_profile_flow(target_rgb, old, initial_h, target_mask, radius=24):
    """One-dimensional image registration; no fictitious tangent constraint.

    A geometric map initializes a finite search. Real RGB profile correlation
    supplies normal displacement. Distinct near-equal peaks remain ambiguous.
    Returned positions are search matches, not measurement coordinates.
    """
    uv, normals = old['uv'], old['normal']
    if not len(uv): return uv.copy(), np.zeros(0, bool), np.zeros(0), np.zeros(0)
    seed = _warp(initial_h, uv)
    tangent = np.c_[-normals[:, 1], normals[:, 0]]
    mapped_tangent = _warp(initial_h, uv+tangent*5.)-_warp(initial_h, uv-tangent*5.)
    mapped_tangent /= np.maximum(np.linalg.norm(mapped_tangent, axis=1)[:, None], 1e-9)
    normal = np.c_[mapped_tangent[:, 1], -mapped_tangent[:, 0]]
    mapped_normal = _warp(initial_h, uv+normals)-seed
    sign = np.where(np.einsum('ij,ij->i', mapped_normal, normal) >= 0, 1., -1.)
    normal *= sign[:, None]
    scale = np.abs(np.einsum('ij,ij->i', mapped_normal, normal))
    # A different camera can compress or enlarge a material far beyond 2x.
    # The native map initializes scale; actual RGB still supplies the match.
    extent = np.maximum(2., np.maximum(12., old['width']*.5+6.)*scale)
    shifts = np.arange(-radius, radius+1, dtype=float)
    offsets = np.linspace(-1., 1., 41)
    prediction = seed.copy(); quality = np.zeros(len(uv)); ambiguity = np.ones(len(uv))
    valid = np.zeros(len(uv), bool)
    rgb = np.asarray(target_rgb, np.float32)
    for start in range(0, len(uv), 128):
        end = min(start+128, len(uv)); count = end-start
        center = seed[start:end, None]+normal[start:end, None]*shifts[None, :, None]
        xy = center[:, :, None]+normal[start:end, None, None]*extent[start:end, None, None, None]*offsets[None, None, :, None]
        bounded = ((xy[..., 0] >= 1)&(xy[..., 0] < rgb.shape[1]-1)&
                   (xy[..., 1] >= 1)&(xy[..., 1] < rgb.shape[0]-1)).all(2)
        xy = np.rint(xy.astype(np.float32)*32.)/32.
        sampled = cv2.remap(rgb, xy[..., 0].reshape(count*len(shifts), 41),
            xy[..., 1].reshape(count*len(shifts), 41), cv2.INTER_LINEAR)
        background = .5*(sampled[:, :4].mean(1)+sampled[:, -4:].mean(1))
        values = (sampled-background[:, None]).reshape(count, len(shifts), -1)
        strength = np.linalg.norm(values, axis=2)
        values /= np.maximum(strength[:, :, None], 1e-6)
        corr = np.einsum('nki,ni->nk', values, old['descriptor'][start:end])
        px = np.rint(center).astype(int)
        in_mask = (px[..., 0] >= 0)&(px[..., 0] < target_mask.shape[1])&(px[..., 1] >= 0)&(px[..., 1] < target_mask.shape[0])
        masked = np.ones_like(in_mask)
        masked[in_mask] = target_mask[px[..., 1][in_mask], px[..., 0][in_mask]]
        corr[~bounded | masked | (strength <= 0)] = -1.
        best = corr.argmax(1); rows = np.arange(count)
        peak = corr[rows, best]
        separated = np.abs(np.arange(len(shifts))[None]-best[:, None]) >= 7
        # A broad peak from motion blur is one match, not several competing
        # materials. Only another local maximum competes for identity.
        peaks = np.zeros_like(corr, bool)
        peaks[:, 1:-1] = (corr[:, 1:-1]>=corr[:, :-2]) & (corr[:, 1:-1]>corr[:, 2:])
        peaks[:, 0] = corr[:, 0]>corr[:, 1]
        peaks[:, -1] = corr[:, -1]>corr[:, -2]
        second = np.max(np.where(separated & peaks, corr, -1.), axis=1)
        displacement = shifts[best].copy()
        interior = (best > 0)&(best < len(shifts)-1)
        ix = rows[interior]; j = best[interior]
        denominator = corr[ix, j-1]-2.*corr[ix, j]+corr[ix, j+1]
        delta = .5*(corr[ix, j-1]-corr[ix, j+1])/np.minimum(denominator, -1e-9)
        displacement[interior] += np.clip(delta, -.5, .5)
        prediction[start:end] = seed[start:end]+normal[start:end]*displacement[:, None]
        quality[start:end] = peak; ambiguity[start:end] = second
        valid[start:end] = interior & (peak >= .80) & (peak-second >= .025) & (old['strength'][start:end] > 0)
    return prediction, valid, quality, ambiguity


def link_fragments(previous_rgb, current_rgb, old, now, previous_mask, current_mask, initial_h):
    """Real two-image links across curve names; retain ambiguous alternatives.

    The line tangent is unobservable locally, so profile registration only predicts the normal.
    Descriptors and width test actual paint support. Fragment votes aggregate
    multiple separated profiles. Neither a curve ID nor model residual is an
    eligibility gate, and a competing link remains visible in the graph.
    """
    predicted, good, forward_correlation, _ = normal_profile_flow(current_rgb, old, initial_h, current_mask)
    reversed_uv, reverse_good, reverse_correlation, _ = normal_profile_flow(previous_rgb, now, np.linalg.inv(initial_h), previous_mask)
    old_tangent=np.c_[-old['normal'][:,1],old['normal'][:,0]]
    moved_tangent=_warp(initial_h,old['uv']+old_tangent)-_warp(initial_h,old['uv']-old_tangent)
    moved_tangent/=np.maximum(np.linalg.norm(moved_tangent,axis=1)[:,None],1e-9)
    moved_normal=np.c_[moved_tangent[:,1],-moved_tangent[:,0]]
    moved_step=_warp(initial_h,old['uv']+old['normal'])-_warp(initial_h,old['uv'])
    normal_scale=np.abs(np.einsum('ij,ij->i',moved_step,moved_normal))
    votes = defaultdict(list)
    for family in np.intersect1d(old['family'][good], now['family']):
        before = np.flatnonzero(good & (old['family'] == family))
        after = np.flatnonzero((now['family'] == family) & reverse_good)
        if not len(before) or not len(after): continue
        tree = cKDTree(predicted[before])
        neighborhoods = tree.query_ball_point(now['uv'][after], 42.)
        for node, neighbors in zip(after, neighborhoods):
            candidates = before[neighbors]
            if not len(candidates): continue
            delta = now['uv'][node]-predicted[candidates]
            normals = old['normal'][candidates]
            normal_distance = np.abs(delta@now['normal'][node])
            reverse_distance = np.abs(np.einsum('ij,ij->i', reversed_uv[node]-old['uv'][candidates], normals))
            tangent_distance = np.abs(delta@np.array([-now['normal'][node,1],now['normal'][node,0]]))
            corr = old['descriptor'][candidates]@now['descriptor'][node]
            ratio = now['width'][node]/np.maximum(old['width'][candidates]*normal_scale[candidates], 1e-6)
            interfaces = (now['width'][node] <= 1e-6) & (old['width'][candidates] <= 1e-6)
            eligible = (normal_distance <= 3.) & (reverse_distance <= 3.) & (tangent_distance <= 40.) & (corr >= .75)
            eligible &= (np.abs(moved_normal[candidates]@now['normal'][node]) >= .95)
            eligible &= interfaces | ((ratio >= .6) & (ratio <= 1.67))
            score = normal_distance+.025*tangent_distance+2.*(1.-corr)
            for fragment in np.unique(old['fragment'][candidates[eligible]]):
                picks = np.flatnonzero(eligible & (old['fragment'][candidates] == fragment))
                pick = picks[np.argmin(score[picks])]; parent = int(candidates[pick])
                votes[(int(fragment), int(now['fragment'][node]))].append(
                    (parent, int(node), float(reverse_distance[pick]), float(corr[pick]), float(normal_distance[pick])))
    edges = []
    for (a, b), evidence in votes.items():
        pairs = np.asarray(evidence)
        current = np.unique(pairs[:, 1].astype(int))
        previous = np.unique(pairs[:, 0].astype(int))
        if min(len(current), len(previous)) < 3: continue
        span = np.ptp(now['uv'][current], axis=0)
        if float(np.linalg.norm(span)) < 56.: continue
        edges.append(dict(a=a, b=b, support=len(current), previous_support=len(previous),
            backward_normal_median_px=float(np.median(pairs[:, 2])), correlation_median=float(np.median(pairs[:, 3])),
            normal_median_px=float(np.median(pairs[:, 4])),
            pairs=pairs[:, :2].astype(int).tolist(),
            source_only=True, identity_assumed_from_curve_name=False))
    return qualify_fragment_links(edges)


def qualify_fragment_links(edges):
    """Keep disjoint source spans connected across segmentation splits.

    A split or merge is not an alternative correspondence when its measured
    support is disjoint. Reuse of even one matched node remains competition;
    the existing dominance threshold and all RGB matching evidence stay intact.
    """
    source = [{int(a) for a, _ in e['pairs']} for e in edges]
    target = [{int(b) for _, b in e['pairs']} for e in edges]
    for i, e in enumerate(edges):
        rivals = [r for j, r in enumerate(edges) if j != i and
                  ((r['a'] == e['a'] and source[i] & source[j]) or
                   (r['b'] == e['b'] and target[i] & target[j]))]
        rival = max((r['support'] for r in rivals), default=0)
        e['unambiguous'] = rival <= .4*e['support']
        e['strongest_competing_support'] = rival
        e['competition_basis'] = 'shared_measured_source_or_target_nodes'
    return edges


def disconnected_source_boundaries(frames, edge_sets, reference_edges=()):
    """Schedule RGB checks across missing exposures, without assigning identity.

    Query curves only locate successive source opportunities. A scheduled pair
    cannot propagate a label until its actual RGB correspondence is qualified.
    Already connected components and already tested adjacent frames are omitted.
    """
    parent={(fi,j):(fi,j) for fi,f in enumerate(frames) for j in range(len(f['fragments']))}
    def find(node):
        while parent[node]!=node:
            parent[node]=parent[parent[node]];node=parent[node]
        return node
    def join(a,b):
        parent[find(a)]=find(b)
    for fi,edges in enumerate(edge_sets):
        for edge in edges:
            if edge['unambiguous']:join((fi,edge['a']),(fi+1,edge['b']))
    for record in reference_edges:
        for edge in record['edges']:
            if edge['unambiguous']:
                join((record['reference'],edge['a']),(record['current'],edge['b']))
    previous={};pairs=set()
    for fi,frame in enumerate(frames):
        current=defaultdict(list)
        for j,fragment in enumerate(frame['fragments']):current[fragment['curve']].append((fi,j))
        for curve,nodes in current.items():
            old=previous.get(curve,[])
            if old and fi-old[0][0]>1 and any(find(a)!=find(b) for a in old for b in nodes):
                pairs.add((old[0][0],fi))
            previous[curve]=nodes
    return sorted(pairs)


def propagate_identities(frames, edge_sets, anchors, reference_edges=()):
    """Propagate anchor hypotheses both ways, never erase label conflicts."""
    adjacency = defaultdict(list)
    for i, edges in enumerate(edge_sets):
        for ei, edge in enumerate(edges):
            if not edge['unambiguous']: continue
            a, b = (i, edge['a']), (i+1, edge['b'])
            adjacency[a].append((b, i, ei)); adjacency[b].append((a, i, ei))
    for ri, record in enumerate(reference_edges):
        for ei, edge in enumerate(record['edges']):
            if not edge['unambiguous']: continue
            a, b = (record['reference'], edge['a']), (record['current'], edge['b'])
            adjacency[a].append((b, 'reference:'+str(ri), ei))
            adjacency[b].append((a, 'reference:'+str(ri), ei))
    by_pts = {int(f['pts']): i for i, f in enumerate(frames)}
    labels = defaultdict(dict)
    for ai, anchor in enumerate(anchors):
        start = (by_pts[int(anchor['pts'])], int(anchor['fragment']))
        queue = [start]; visited = {start}; labels[start][ai] = None
        for at in queue:
            for other, frame, edge in adjacency[at]:
                if other in visited: continue
                visited.add(other); queue.append(other)
                labels[other][ai] = dict(parent=list(at), edge_frame=frame, edge=edge)
    result = []
    for (fi, fragment), proofs in sorted(labels.items()):
        identities = sorted({anchors[ai]['identity'] for ai in proofs})
        result.append(dict(frame=fi, pts=frames[fi]['pts'], fragment=fragment,
            identities=identities, status='anchored' if len(identities) == 1 else 'identity_conflict',
            proofs={str(ai): path for ai, path in proofs.items()}))
    return result


def qualify_reference_links(edges, identities):
    """Separate actual identity rivals from disjoint fragments of one line.

    Generic fragment matching treats any split as competition. With a human
    identity reference, two disjoint source spans support the same identity.
    Reuse of the same source support for multiple branches remains ambiguous.
    """
    by_target = defaultdict(set)
    for edge in edges:
        by_target[edge['b']].add(identities[edge['a']])
    for edge in edges:
        source_nodes_here = {int(a) for a, _ in edge['pairs']}
        rivals = []
        for other in edges:
            if other is edge or other['a'] != edge['a'] or other['b'] == edge['b']:
                continue
            shared = source_nodes_here & {int(a) for a, _ in other['pairs']}
            if len(shared) >= max(3, .4*len(source_nodes_here)):
                rivals.append(other['b'])
        edge['identity_unambiguous'] = len(by_target[edge['b']]) == 1 and not rivals
        edge['identity_competing_branches'] = sorted(set(rivals))
    return edges



def propagate_readout_identities(candidate_counts, links, anchors, qualified_groups=()):
    """Propagate only measured node pairs and explicitly qualified whole lines.

    Local fragment membership is deliberately absent: one matching part of a
    mixed fragment must not label its other material. Connected components
    preserve every anchor conflict and can be reconstructed from saved links.
    """
    counts=np.asarray(candidate_counts,np.int64)
    if counts.ndim!=1 or (counts<0).any():raise ValueError('Nonnegative frame candidate counts required')
    offsets=np.r_[0,np.cumsum(counts)];parent=np.arange(offsets[-1]);rank=np.zeros(len(parent),np.int8)
    def node(frame,candidate):
        frame=int(frame);candidate=int(candidate)
        if not 0<=frame<len(counts) or not 0<=candidate<counts[frame]:
            raise ValueError('Source correspondence references a nonexistent readout')
        return int(offsets[frame]+candidate)
    def find(at):
        while parent[at]!=at:
            parent[at]=parent[parent[at]];at=int(parent[at])
        return at
    def join(a,b):
        a=find(a);b=find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    for group in qualified_groups:
        if group.get('source_line_support_qualified') is not True:
            raise ValueError('Unqualified local fragments cannot share identity')
        ids=[node(group['frame'],i) for i in group['candidate_ids']]
        for other in ids[1:]:join(ids[0],other)
    for link in links:
        if not link.get('unambiguous'):continue
        if link.get('source_only') is not True:raise ValueError('Actual RGB correspondence required')
        for first,last in link['pairs']:
            join(node(link['previous'],first),node(link['current'],last))
    component=np.asarray([find(i) for i in range(len(parent))],np.int64)
    labels=defaultdict(list)
    for ai,anchor in enumerate(anchors):labels[int(component[node(anchor['frame'],anchor['candidate'])])].append(ai)
    result=[]
    for ci,source in sorted(labels.items()):
        identities=sorted({anchors[i]['identity'] for i in source})
        result.append(dict(component=ci,identities=identities,anchor_indices=source,
            status='anchored' if len(identities)==1 else 'identity_conflict'))
    return offsets,component,result


def check_matched_line_geometry(measurement, candidate_ids, visible_reports):
    """Check source shape after independent node correspondence supplied identity.

    Sparse matches do not become whole-line qualifications or identity anchors.
    Missing visible length remains in the organizer's original denominator;
    this check only rejects inconsistent or ambiguous existing RGB readouts.
    """
    from rinkspace.vision_source_roles import organize_source_line_material_branches
    ids=np.asarray(candidate_ids,int);query=np.asarray(measurement['candidate_query'],int)
    curves=np.asarray(measurement['query_curve'],int);uv=np.asarray(measurement['candidate_uv'])
    selected=[];diagnostics=[]
    for curve in np.unique(curves[query[ids]]):
        report=next(r for r in visible_reports if r['curve']==curve)
        queries=np.asarray(report.get('visible_source_queries',[]),int)
        candidates=ids[curves[query[ids]]==curve]
        if not len(queries):
            diagnostics.append(dict(curve=int(curve),disposition='no_visible_source_queries'));continue
        rows=[];pools=[]
        for qi in queries:
            pool=candidates[query[candidates]==qi];pools.append(pool)
            rows.append(dict(source_search_point_uv=measurement['query_uv'][qi].tolist(),
                all_actual_source_material_candidates=[dict(actual_source_uv=uv[k].tolist(),
                    disposition='measured_material_transition') for k in pool]))
        starts=np.r_[0,np.flatnonzero(np.diff(queries)!=1)+1];stops=np.r_[starts[1:],len(queries)]
        paths=[list(range(int(a),int(b))) for a,b in zip(starts,stops)]
        normal=np.asarray(measurement['query_normal'])[queries]
        result=organize_source_line_material_branches(rows,navigation_tangent_uv=np.c_[-normal[:,1],normal[:,0]],
            source_visible_paths=paths)
        organizer=result.get('actual_source_line_organizer');keep=[]
        unique=(organizer is not None and organizer['unique_source_support_length_px']>=
            result['competing_support_ratio']*organizer['competing_family_support_length_px'])
        if unique:
            for pool in pools:
                if not len(pool):continue
                station=uv[pool]@organizer['basis']-organizer['station_center']
                delta=station*organizer['coefficients'][0]+organizer['coefficients'][1]-uv[pool]@organizer['normal']
                consistent=pool[np.abs(delta)<=result['centerline_role_tolerance_px']]
                if len(consistent)==1:keep.append(int(consistent[0]))
            if len(keep)<3 or np.linalg.norm(np.ptp(uv[keep],axis=0))<56.:keep=[]
        selected.extend(keep)
        diagnostics.append(dict(curve=int(curve),input_candidate_ids=candidates.tolist(),
            selected_candidate_ids=keep,whole_visible_line_disposition=result['disposition'],
            source_line_support_qualified=result['source_line_support_qualified'],
            independent_node_identity_still_required=True,source_organizer=organizer,
            source_geometry_tolerance_px=result['centerline_role_tolerance_px'],
            visible_length_denominator_unchanged=True,native_fit_residual_used=False,source_UV_changed=False))
    return np.asarray(sorted(set(selected)),np.int64),diagnostics
