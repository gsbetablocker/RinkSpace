"""Actual RGB exterior edges of an independently image-registered print.

Print matching belongs to the caller. Source component envelopes and source
rectangles navigate/topologically guard searches only; each output UV remains
an original RGB interface q50, never a contour, rectangle or fitted line point.
"""
from copy import deepcopy
from itertools import permutations

import cv2
import numpy as np

from rinkspace.session.projected_curve_queries import projected_curve_queries
from rinkspace.session.material_tracks import qualify_line_candidates
from rinkspace.session.panel_boundaries import _color_filled_components, _source_supported_visibility
from rinkspace.vision_programme_ice_review import _batched_actual_profiles, _actual_profile_candidates
from rinkspace.vision_source_shapes import _MaterialProfileFrame


def _source_topology_once(rgb, paths, exclusion, radius):
    """Reuse the public panel reader's hole-filled RGB component envelope.

    Reliable ROI-background connectivity additionally distinguishes board white
    from enclosed white glyphs. Neither mask supplies an observed coordinate.
    """
    height, width = exclusion.shape
    all_uv = np.vstack([p[np.isfinite(p).all(1)] for p in paths])
    lo = np.maximum(np.floor(all_uv.min(0)-radius).astype(int), [0, 0])
    hi = np.minimum(np.ceil(all_uv.max(0)+radius).astype(int), [width, height])
    if np.any(hi-lo < 8):
        return None, dict(status='outside_source')
    patch = rgb[lo[1]:hi[1], lo[0]:hi[0]]
    signal = patch.min(2)
    available = ~exclusion[lo[1]:hi[1], lo[0]:hi[0]]
    values = signal[available]
    if len(values) < 20 or np.ptp(values) < 4:
        return None, dict(status='panel_material_unresolved')
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = ((signal <= threshold) & available).astype('uint8')
    expected = np.zeros(binary.shape, 'uint8')
    cv2.fillConvexPoly(expected, np.rint(cv2.convexHull((all_uv-lo).astype('float32'))).astype('int32'), 1)
    # A one-dimensional rail or a narrow foreground bridge can be dark and
    # lie wholly inside the registered search footprint. Remove thin appendages
    # before filling glyph holes; filling the original connected contour first
    # would also fill board white enclosed by a rail/foreground bridge.
    # This chart is source-search scale only, never a measured boundary.
    _, _, axes = np.linalg.svd(all_uv-all_uv.mean(0), full_matrices=False)
    short_axis = axes[-1]
    yy, xx = np.nonzero(expected)
    if not len(xx):
        return None, dict(status='no_registered_source_footprint_in_image')
    cross_span = float(np.ptp((np.c_[xx, yy]+lo)@short_axis))
    core_radius = max(1, int(round(cross_span/16.)))
    diameter = 2*core_radius+1
    core = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    core[~available] = 0
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for number, contour in enumerate(contours):
        region = np.zeros(binary.shape, 'uint8')
        cv2.fillPoly(region, [contour], 1)
        overlap = int(np.count_nonzero(region & expected))
        area = int(np.count_nonzero(region))
        cross = (contour[:, 0]+lo)@short_axis
        candidates.append(dict(component=number, overlap=overlap, area=area,
            contour=contour, region=region, cross=[float(cross.min()), float(cross.max())]))
    overlapping = [c for c in candidates if c['overlap']]
    if not overlapping:
        return None, dict(status='no_registered_two_dimensional_panel_core',
            source_core_radius_px=core_radius)
    primary = max(overlapping, key=lambda c: (c['overlap'], c['area']))
    low, high = primary['cross']; thickness = high-low
    if thickness < .5*cross_span:
        return None, dict(status='source_core_does_not_span_registered_panel_cross_section',
            source_core_radius_px=core_radius, primary_core_cross_span_px=thickness,
            visible_registered_cross_span_px=cross_span)
    pieces = []
    component_reports = []
    for item in candidates:
        a, b = item['cross']; shared = max(0., min(high, b)-max(low, a))
        # Occlusion may split the same print into a narrow left/right piece.
        # Its fraction inside a tilted registration is not identity evidence:
        # preserve the same two-dimensional cross-panel extent instead. Thin
        # parallel rails and small interior patches cannot enlarge the envelope.
        selected = bool(item['overlap'] and b-a >= .5*thickness
                        and shared >= .5*min(thickness, b-a))
        if selected: pieces.append((item['contour'], item['region']))
        component_reports.append(dict(component=item['component'],
            area_px=item['area'], registered_overlap_px=item['overlap'],
            cross_panel_interval_px=item['cross'], selected=selected,
            disposition='same_panel_cross_section' if selected else 'not_same_two_dimensional_panel_extent'))
    hull = cv2.convexHull(np.concatenate([c for c, _ in pieces], axis=0))
    envelope = np.zeros(binary.shape, 'uint8')
    cv2.fillConvexPoly(envelope, hull, 1)
    # Only original RGB pixels inside the coherent panel envelope supply
    # material. Opening, glyph filling and the hull never produce source UV or
    # painted pixels, and cannot turn an internal reflection into board white.
    material = envelope & available & (signal <= threshold)
    # Connectivity belongs to this selected print's exterior, rather than all
    # dark marks in the board ROI: another print/border may enclose a real white
    # board strip. The filled print envelope already excludes all glyph holes.
    filled = _color_filled_components(envelope, available, lo, hi, (height, width))
    if filled is None:
        return None, dict(status='reliable_source_board_background_unresolved')
    board = (filled == 0) & available & (signal > threshold) & (envelope == 0)
    info = dict(status='source_topology_resolved', threshold=float(threshold),
        roi_xyxy_exclusive=[*lo.tolist(), *hi.tolist()], source_component_count=len(pieces),
        material_pixel_count=int(material.sum()), exterior_white_pixel_count=int(board.sum()),
        method='two_dimensional_source_material_core_and_connected_exterior_light_board',
        source_core_radius_px=core_radius, primary_source_core=primary['component'],
        source_core_components=component_reports,
        source_component_used_for_topology_only=True, contour_or_rectangle_used_as_observation=False)
    arrays = dict(lo=lo, hi=hi, source_RGB=patch, source_exclusion=~available,
        dark_binary=binary, source_material_core=core, filled_component_envelope=envelope, actual_material=material,
        connected_white_board=board, expected_navigation_footprint=expected, source_hull=hull[:, 0]+lo)
    return arrays, info


def _source_topology(rgb, paths, exclusion, radius):
    # Expand only source context, until the measured component has a complete
    # profile-radius halo at artificial ROI edges. True frame edges stay clipped
    # and can never supply missing outside source pixels.
    attempts = []
    margin = float(radius)+3.
    previous_roi = None
    while True:
        topology, info = _source_topology_once(rgb, paths, exclusion, radius)
        attempts.append(dict(info, context_radius_px=float(radius)))
        if topology is None:
            info['context_attempts'] = attempts
            return topology, info
        lo, hi = topology['lo'], topology['hi']
        low = topology['source_hull'].min(0)
        high = topology['source_hull'].max(0)
        clipped = (((lo > 0) & (low-lo < margin)).any()
            or ((hi < np.array(exclusion.shape[::-1])) & (hi-1-high < margin)).any())
        roi = tuple(np.r_[lo, hi])
        if not clipped or roi == previous_roi:
            info['context_attempts'] = attempts
            return topology, info
        previous_roi = roi
        radius *= 2.


def _source_search_paths(topology, paths):
    """Orient four source-envelope search sides by the known semantic sides.

    A min-area rectangle is an intentionally coarse search chart, including when
    a panel is clipped by the source image. It is never a boundary measurement;
    profiles and their complete mask/exterior guards still decide every q50.
    """
    hull = topology['source_hull'].astype('float32')
    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
    source_center = box.mean(0)
    edge = [np.array([box[i], box[(i+1) % 4]]) for i in range(4)]
    directions = np.array([p.mean(0)-source_center for p in edge])
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
    all_uv = np.vstack([p[np.isfinite(p).all(1)] for p in paths])
    center = all_uv.mean(0)
    outward = []
    for path in paths:
        finite = path[np.isfinite(path).all(1)]
        if len(finite) < 2:
            return None, dict(status='incomplete_registered_semantic_sides')
        tangent = finite[-1]-finite[0]
        n = np.array([-tangent[1], tangent[0]])
        if n @ (finite.mean(0)-center) < 0:
            n = -n
        outward.append(n/max(float(np.linalg.norm(n)), 1e-12))
    assignment = max(permutations(range(4)), key=lambda p: sum(outward[i]@directions[p[i]] for i in range(4)))
    result = []
    for ci, j in enumerate(assignment):
        side = edge[j]
        finite = paths[ci][np.isfinite(paths[ci]).all(1)]
        if (side[-1]-side[0]) @ (finite[-1]-finite[0]) < 0:
            side = side[::-1]
        result.append(side.copy())
    return result, dict(status='source_envelope_search_chart', source_rectangle=box.tolist(),
        semantic_side_assignment=list(assignment), paths=[p.tolist() for p in result],
        source_chart_is_navigation_only=True, native_fit_residual_used=False,
        source_contour_UV_used_as_measurement=False)


def read_registered_panel_edges(rgb, projected_paths, exclusion, *, binding,
        search_radius=20., spacing=24., candidate_material_validator=None, proof_sink=None):
    """Read four actual RGB exterior edges after independent print verification.

    The optional ``proof_sink(stage, arrays)`` receives native source topology and
    sampled RGB profiles for a diagnostic artifact. It has no decision role.
    All original-navigation and relocated-search candidates/rejections are kept.
    """
    if binding.get('independent_source_pattern_match_verified') is not True:
        raise ValueError('Independent source pattern identity/navigation verification is required')
    if candidate_material_validator is not None and not callable(candidate_material_validator):
        raise TypeError('candidate_material_validator must be callable')
    if proof_sink is not None and not callable(proof_sink):
        raise TypeError('proof_sink must be callable')
    frame = _MaterialProfileFrame(np.ascontiguousarray(rgb), binding)
    height, width = frame.rgb.shape[:2]
    exclusion = np.asarray(exclusion, bool)
    if exclusion.shape != (height, width):
        raise ValueError('Source exclusion mask must match the native RGB image')
    paths = []
    for path in projected_paths:
        path = np.asarray(path, float)
        if not path.size: path = np.empty((0, 2))
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError('Projected panel paths must have shape (N, 2)')
        paths.append(path)
    if len(paths) != 4:
        raise ValueError('Four registered panel paths are required in top/right/bottom/left order')
    if not np.isfinite(search_radius) or search_radius <= 0 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Search radius and spacing must be finite and positive')
    record = dict(status='source_exterior_not_resolved', candidate_count=0, query_count=0,
        reports=[], candidate_records=[], pending_candidates=[], profile_debug=[], selected_candidate_ids=[],
        source_UV_unchanged=True, model_UV_synthesized=False,
        source_RGB_sha256=frame.digest, independent_source_pattern_match_verified=True,
        physical_identity_determined_by_this_reader=False,
        source_component_method='two_dimensional_source_material_core_and_connected_exterior_light_board',
        source_component_used_for_topology_only=True,
        source_material_support_method='actual_dark_material_and_exterior_connected_board_white',
        native_shape_guided=True, native_shape_guidance_role='semantic sides and source-line organization only',
        native_fit_residual_used=False, fitted_line_used_as_observation=False,
        source_UV_guard='actual q25-q75 span plus 3px, all pixel contributors in-image and unexcluded; inner actual material; outer connected board white',
        edge_order=['top', 'right', 'bottom', 'left'], query_sampling=[])
    if not any(np.isfinite(p).all(1).any() for p in paths):
        record['status'] = 'outside_source'
        return [], record
    topology, topology_info = _source_topology(frame.rgb, paths, exclusion, search_radius)
    record['source_topology'] = topology_info
    source_paths, chart = _source_search_paths(topology, paths) if topology is not None else (None, None)
    record['source_search_chart'] = chart
    if topology is not None and proof_sink is not None:
        proof_sink('source_topology', topology)
    stages = [('registered_navigation_audit', paths)]
    if source_paths is not None:
        stages.append(('actual_source_envelope_search', source_paths))
    candidate_rows = []
    stage_measurements = []
    for stage, stage_paths in stages:
        finite = [p[np.isfinite(p).all(1)] for p in stage_paths]
        center = np.vstack([p for p in finite if len(p)]).mean(0)
        q, normal, interval, curve = [], [], [], []
        for ci, path in enumerate(stage_paths):
            valid_path = np.isfinite(path).all(1)
            segment_valid = valid_path[:-1] & valid_path[1:]
            path_length = float(np.linalg.norm(np.diff(path, axis=0)[segment_valid], axis=1).sum())
            short_edge = 0. < path_length <= 6.*spacing
            query_spacing = spacing/3. if short_edge else spacing
            queries = projected_curve_queries(path, valid_path, (height, width),
                query_spacing, search_radius, center_uv=center, closed=False)
            record['query_sampling'].append(dict(stage=stage, curve=ci, projected_path_length_px=path_length,
                requested_spacing_px=float(spacing), actual_search_spacing_px=float(query_spacing),
                short_edge_dense_search=short_edge, query_count=len(queries['points_uv']),
                source_qualification_criteria_changed=False, final_source_uv_spacing_owned_by_caller=True))
            q.extend(queries['points_uv']); normal.extend(queries['normals'])
            interval.extend(queries['intervals']); curve.extend([ci]*len(queries['points_uv']))
        if not q: continue
        q, normal, interval, curve = map(np.asarray, (q, normal, interval, curve))
        query_base = record['query_count']; record['query_count'] += len(q)
        offsets, profiles, smooth, valid = _batched_actual_profiles(frame, q, normal, interval,
                                                                   np.full(len(q), 'dark'))
        sides, quantiles, candidate_query, flags, global_ids = [], [], [], [], []
        for qi in range(len(q)):
            actual = _actual_profile_candidates(q[qi], normal[qi], offsets[qi, valid[qi]],
                profiles[qi, valid[qi]], smooth[qi, valid[qi]], operator='material_interface',
                signal='dark', transition='dark_to_light', with_quantiles=True, separate_bands=True)
            query_debug = dict(stage=stage, curve=int(curve[qi]), query=q[qi].tolist(), normal=normal[qi].tolist(),
                               interval=interval[qi].tolist(), candidate_ids=[])
            for point, pair, bad, qq in actual:
                point, qq = np.asarray(point), np.asarray(qq)
                candidate_id = len(candidate_rows)
                row = dict(curve=int(curve[qi]), uv=point.tolist(), normal=normal[qi].tolist(),
                    actual_source_quantiles=qq.tolist(), model_UV_synthesized=False,
                    measurement_operator='source_outer_panel_material_interface')
                reasons = []; flag = 0; span = None; topology_probes = None
                if bad:
                    flag |= 1; reasons.append('actual_source_profile_truncated_or_reversed')
                finite_source = (point.shape == (2,) and qq.shape == (2, 3, 2)
                                 and np.isfinite(point).all() and np.isfinite(qq).all())
                if not finite_source:
                    flag |= 1; reasons.append('nonfinite_or_invalid_actual_source_quantiles')
                else:
                    quantile_offsets = (qq[:, [0, 2]].reshape(-1, 2)-point)@normal[qi]
                    lo, hi = float(quantile_offsets.min()-3.), float(quantile_offsets.max()+3.)
                    span = [lo, hi]
                    segment = point+np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)*4.))+1))[:, None]*normal[qi]
                    if np.any(segment < 0) or np.any(segment > [width-1, height-1]):
                        flag |= 2; reasons.append('actual_transition_span_crosses_image_boundary')
                    else:
                        lower, upper = np.floor(segment).astype(int), np.ceil(segment).astype(int)
                        if any(exclusion[yy[:, 1], xx[:, 0]].any() for yy in (lower, upper) for xx in (lower, upper)):
                            flag |= 2; reasons.append('actual_transition_span_crosses_source_exclusion')
                    probes = point+np.array([lo, hi])[:, None]*normal[qi]
                    ij = np.rint(probes).astype(int)
                    topology_probes = dict(uv=probes.tolist(), roles=['inner_actual_material', 'outer_connected_board_white'])
                    if topology is None:
                        flag |= 1; reasons.append('actual_source_exterior_topology_unresolved')
                    else:
                        local = ij-topology['lo']
                        inside = (local >= 0).all(1) & (local < topology['hi']-topology['lo']).all(1)
                        inner = bool(inside[0] and topology['actual_material'][local[0, 1], local[0, 0]])
                        outer = bool(inside[1] and topology['connected_white_board'][local[1, 1], local[1, 0]])
                        topology_probes.update(inner_actual_material_supported=inner, outer_connected_board_white_supported=outer,
                            source_RGB=[frame.rgb[y, x].tolist() if 0 <= x < width and 0 <= y < height else None for x, y in ij])
                        if not inner:
                            flag |= 1; reasons.append('inner_quantile_guard_not_actual_source_material')
                        if not outer:
                            flag |= 1; reasons.append('outer_quantile_guard_not_component_exterior_board_white')
                material_supported = None
                if candidate_material_validator is not None:
                    checked = candidate_material_validator(deepcopy(row))
                    if not isinstance(checked, (bool, np.bool_)):
                        raise TypeError('candidate_material_validator must return a boolean')
                    material_supported = bool(checked)
                    if not material_supported:
                        flag |= 1; reasons.append('actual_print_material_not_supported')
                candidate_rows.append(row); sides.append(pair); quantiles.append(qq)
                candidate_query.append(qi); flags.append(flag); global_ids.append(candidate_id)
                query_debug['candidate_ids'].append(candidate_id)
                record['candidate_records'].append(dict(row, candidate_id=candidate_id, stage=stage, query=query_base+qi,
                    stage_query=qi, actual_source_sides=np.asarray(pair).tolist(),
                    source_profile_truncated_or_reversed=bool(bad), actual_guard_normal_span_px=span,
                    source_topology_probes=topology_probes, candidate_material_supported=material_supported,
                    candidate_flags=flag, pending_reasons=reasons,
                    disposition='pending' if reasons else 'awaiting_source_line_organization'))
            record['profile_debug'].append(query_debug)
        if proof_sink is not None:
            proof_sink(stage, dict(query_uv=q, query_normal=normal, query_interval=interval, query_curve=curve,
                offsets=offsets, actual_luma_profiles=profiles, smoothed_luma_profiles=smooth, valid=valid,
                global_candidate_ids=np.array(global_ids), candidate_uv=np.array([candidate_rows[i]['uv'] for i in global_ids]),
                candidate_query=np.array(candidate_query), candidate_side_quantiles_uv=np.array(quantiles),
                candidate_sides=np.array(sides), candidate_flags=np.array(flags)))
        if not global_ids: continue
        measurement = dict(query_uv=q, query_normal=normal, query_interval=interval, query_curve=curve,
            candidate_uv=np.asarray([candidate_rows[i]['uv'] for i in global_ids]), candidate_sides=np.asarray(sides),
            candidate_side_quantiles_uv=np.asarray(quantiles), candidate_query=np.asarray(candidate_query),
            candidate_flags=np.asarray(flags))
        stage_measurements.append((stage, measurement, np.asarray(global_ids)))
    selected_ids = set()
    for stage, measurement, global_ids in stage_measurements:
        # The original tilted registration is audit-only: a crossing of its
        # left normal with the actual bottom silhouette must not become a left
        # edge. Semantic source-envelope sides organize the emitted candidates.
        if stage != 'actual_source_envelope_search': continue
        selected, reports = qualify_line_candidates(measurement, range(4), exclusion,
            native_shape_guided=True, use_source_quantiles=False)
        support, inferred_occlusion = _source_supported_visibility(measurement, reports, exclusion)
        if inferred_occlusion:
            original_reports = deepcopy(reports)
            selected, reports = qualify_line_candidates(measurement, range(4), exclusion,
                native_shape_guided=True, use_source_quantiles=False, source_support_queries=support)
            for report in reports:
                report['source_line_masked_queries'] = inferred_occlusion.get(int(report['curve']), [])
                report['visibility_basis'] = 'actual source-line organizer; complete geometry-tolerance halo is excluded; no clean candidate removed and no new UV'
                report['source_visibility_initial_report'] = next(r for r in original_reports if r['curve']==report['curve'])
        selected_ids.update(map(int, global_ids[selected]))
        for report in reports:
            report['stage'] = stage
            report['selected_candidate_ids'] = [int(global_ids[i]) for i in report.get('selected_candidate_ids', [])]
            record['reports'].append(report)
    for candidate in record['candidate_records']:
        if candidate['candidate_id'] in selected_ids:
            candidate['disposition'] = 'source_line_organized_actual_measurement'
        else:
            if not candidate['pending_reasons']:
                candidate['pending_reasons'].append('registered_navigation_is_audit_only' if candidate['stage']=='registered_navigation_audit'
                    else 'source_line_organization_not_selected')
            candidate['disposition'] = 'pending'
    selected = sorted(selected_ids)
    rows = [candidate_rows[i] for i in selected]
    record.update(status='source_panel_exterior_read' if rows else 'source_exterior_not_resolved',
        candidate_count=len(candidate_rows), selected_candidate_ids=selected,
        pending_candidates=[r for r in record['candidate_records'] if r['candidate_id'] not in selected_ids])
    return rows, record
