"""Organize measured RGB branches by a declared source geometry role.

Navigation provides a direction and a search neighborhood, not measurements.
The source line below is fitted only to the reader's actual RGB coordinates;
it organizes branches and is never emitted as a replacement UV or ground truth.
Physical identity and FIT/HELD authority remain the caller's responsibility.
"""
from __future__ import annotations

import numpy as np


def organize_source_line_material_branches(readings, *, navigation_tangent_uv,
        source_visible_paths, direction_tolerance_rad=np.deg2rad(8.),
        source_support_margin_px=2., competing_support_ratio=1.15,
        candidate_collection_key='all_actual_source_material_candidates',
        candidate_disposition='measured_material_transition',
        centerline_role_tolerance_px=3., minimum_source_span_fraction=.5):
    """Require global source-line support, instead of local edge dominance.

    ``readings`` retain all local RGB alternatives from the connected reader.
    Explicit paths separate occlusion/identity breaks. The organizing line may
    relate visible portions across those breaks, but outputs retain each break
    and contain only unchanged measured UV. A source ellipse that briefly
    enters a line ROI cannot acquire its role from stronger local contrast.
    Quantization/blur support is an engineering interval, not a covariance.
    Competing long parallel branches remain unknown rather than becoming the
    nearest navigation curve. This is a LINE-family adapter; circular roles
    must use the source-conic adapter, not this function.
    """
    if (not np.isfinite(direction_tolerance_rad) or not 0 < direction_tolerance_rad < np.pi/4
            or not np.isfinite(source_support_margin_px) or source_support_margin_px < 0
            or not np.isfinite(centerline_role_tolerance_px) or centerline_role_tolerance_px <= 0
            or not np.isfinite(minimum_source_span_fraction) or not 0 < minimum_source_span_fraction <= 1
            or not np.isfinite(competing_support_ratio) or competing_support_ratio <= 1):
        raise ValueError('Explicit source role engineering settings required')
    tangent = np.asarray(navigation_tangent_uv, float)
    if tangent.shape == (2,):
        tangent = np.tile(tangent, (len(readings), 1))
    if tangent.shape != (len(readings), 2) or not np.isfinite(tangent).all():
        raise ValueError('Matching navigation tangent directions required')
    lengths = np.linalg.norm(tangent, axis=1)
    if (lengths <= 1e-9).any():
        raise ValueError('Nonzero navigation tangent required')
    tangent = tangent / lengths[:, None]
    tangent *= np.where(tangent @ tangent[0] < 0, -1., 1.)[:, None]
    basis = tangent.mean(0); basis /= np.linalg.norm(basis)
    normal = np.array([-basis[1], basis[0]])
    search = np.asarray([r['source_search_point_uv'] for r in readings], float)
    if search.shape != (len(readings), 2) or not np.isfinite(search).all():
        raise ValueError('Original finite source search positions required')
    station = search @ basis
    center = .5 * (station.min() + station.max())
    station -= center
    pools = []
    for query, row in enumerate(readings):
        pool = []
        candidates = row.get(candidate_collection_key, [row])
        for branch, item in enumerate(candidates):
            uv = np.asarray(item.get('actual_source_uv', []), float)
            if item.get('disposition') != candidate_disposition or uv.shape != (2,):
                continue
            if 'observed_normal_interval_px' in item:
                observed = np.asarray(item['observed_normal_interval_px'], float)
                direction = np.asarray(item['source_search_normal'], float)
                direction = direction/np.linalg.norm(direction)
                projected = np.sort(observed * float(direction @ normal))
                interval = projected + np.array([-source_support_margin_px, source_support_margin_px])
            else:
                # A paint band's occupied footprint is not its center's
                # covariance. Center roles use a separately declared source
                # geometry tolerance; no packet or solver sigma is inferred.
                interval = np.array([-centerline_role_tolerance_px, centerline_role_tolerance_px])
            pool.append(dict(query=query, branch=branch, original=item, uv=uv,
                station=float(uv @ basis-center), value=float(uv @ normal), interval=interval))
        pools.append(pool)
    covered = set(); seen_edges = set(); weights = np.zeros(len(readings), float)
    for path in source_visible_paths:
        if (not isinstance(path, (list, tuple)) or not path
                or any(type(i) is not int or not 0 <= i < len(readings) for i in path)):
            raise ValueError('Explicit in-range source visible query paths required')
        covered.update(path)
        for first, last in zip(path[:-1], path[1:]):
            edge=tuple(sorted((first,last)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            distance = .5 * abs(station[last]-station[first])
            weights[first] += distance; weights[last] += distance
        if len(path) == 1:
            weights[path[0]] = 1.
    if covered != set(range(len(readings))):
        raise ValueError('Every source query requires explicit visible path coverage')
    eligible = [i for i, pool in enumerate(pools) if pool]
    empty = dict(schema='rinkspace.source-line-material-role-organization.v1',
        source_geometry_role='line', actual_source_uv=np.empty((0, 2)),
        source_query_indices=np.empty(0, np.int64), source_candidate_indices=np.empty(0, np.int64),
        source_visible_paths=[], source_line_support_qualified=False,
        physical_edge_role_verified=False, physical_identity_authority_required=True,
        model_projection_used_as_measurement=False, fitted_geometry_used_as_measurement=False,
        source_interpolation_used_as_measurement=False, source_pixels_unchanged=True,
        direction_tolerance_rad=float(direction_tolerance_rad),
        source_support_margin_px=float(source_support_margin_px),
        competing_support_ratio=float(competing_support_ratio))
    empty.update(candidate_collection_key=candidate_collection_key,
        centerline_role_tolerance_px=float(centerline_role_tolerance_px),
        minimum_source_span_fraction=float(minimum_source_span_fraction),
        geometry_tolerance_is_not_solver_sigma_or_calibrated_covariance=True)
    if len(eligible) < 2 or np.ptp(station[eligible]) <= 1e-9:
        return dict(empty, disposition='insufficient_distinct_actual_source_support', readings=[])
    # Hypotheses are actual RGB alternatives at source-span quantiles. No
    # model UV is an anchor, and amplitude or sampling density cannot create
    # additional length support for a physical curve.
    levels = np.quantile(station[eligible], [0., .25, .5, .75, 1.])
    anchors = sorted({min(eligible, key=lambda i: abs(station[i]-level)) for level in levels})
    hypotheses = []
    min_span = .5 * np.ptp(station[eligible])
    for a in anchors:
        for b in anchors:
            if station[b]-station[a] < min_span:
                continue
            for first in pools[a]:
                for last in pools[b]:
                    span = last['station']-first['station']
                    if abs(span) <= 1e-9:
                        continue
                    slope = (last['value']-first['value'])/span
                    if abs(np.arctan(slope)) <= direction_tolerance_rad:
                        hypotheses.append(np.array([slope, first['value']-slope*first['station']]))
    if not hypotheses:
        return dict(empty, disposition='actual_branches_do_not_support_navigation_line_direction', readings=[])

    def assignment(coef):
        selected, ambiguous = {}, []
        for query in eligible:
            compatible = []
            for item in pools[query]:
                distance = coef[0]*item['station']+coef[1]-item['value']
                if item['interval'][0] <= distance <= item['interval'][1]:
                    compatible.append(item)
            if len(compatible) == 1:
                selected[query] = compatible[0]
            elif compatible:
                ambiguous.append(query)
        return selected, ambiguous

    models = {}
    for initial in hypotheses:
        coef = initial.copy()
        # Re-estimate geometry from source candidates only. This fit does not
        # emit coordinates or manufacture support through an occlusion.
        for _ in range(3):
            selected, ambiguous = assignment(coef)
            if len(selected) < 2:
                break
            items = list(selected.values())
            matrix = np.asarray([[v['station'], 1.] for v in items])
            target = np.asarray([v['value'] for v in items])
            weight = np.sqrt(np.maximum(weights[list(selected)], 1e-9))
            new = np.linalg.lstsq(matrix*weight[:, None], target*weight, rcond=None)[0]
            if abs(np.arctan(new[0])) > direction_tolerance_rad:
                break
            coef = new
        selected, ambiguous = assignment(coef)
        key = tuple((i, v['branch']) for i, v in sorted(selected.items()))
        score = float(weights[list(selected)].sum())
        if score > 0 and key:
            models[key] = dict(coef=coef, selected=selected, ambiguous=ambiguous, support=score)
    ranked = sorted(models.values(), key=lambda r: r['support'], reverse=True)
    if not ranked:
        return dict(empty, disposition='no_unique_actual_source_line_branch', readings=[])
    best = ranked[0]
    # Nearby estimates of the same branch family are not independent rivals.
    alternatives = []
    best_keys = {(i, v['branch']) for i, v in best['selected'].items()}
    for row in ranked[1:]:
        keys = {(i, v['branch']) for i, v in row['selected'].items()}
        overlap = sum(weights[i] for i, branch in keys & best_keys)
        if overlap < .5*min(best['support'], row['support']):
            alternatives.append(row)
    rival = alternatives[0]['support'] if alternatives else 0.
    enough_span=(best['support'] >= minimum_source_span_fraction*weights.sum()
        and np.ptp(station[list(best['selected'])]) >= minimum_source_span_fraction*np.ptp(station))
    family_unique = best['support'] >= competing_support_ratio*rival and enough_span
    selected = best['selected'] if family_unique else {}
    rows, output, indices, branches, paths = [], [], [], [], []
    for query, original in enumerate(readings):
        item = selected.get(query)
        reason = ('actual_RGB_branch_supported_by_cross_visible_source_line' if item else
            'insufficient_distributed_source_line_span' if not enough_span else
            'competing_source_line_family' if not family_unique else
            'multiple_material_branches_in_source_line_support' if query in best['ambiguous'] else
            'material_branch_does_not_support_global_source_line_role')
        rows.append(dict(source_query_index=query, disposition=reason,
            selected_actual_candidate_index=item['branch'] if item else None,
            actual_source_uv=item['uv'].copy() if item else np.empty((0, 2)),
            all_actual_source_material_candidates=original.get(candidate_collection_key, [original])))
        if item:
            output.append(item['uv'].copy()); indices.append(query); branches.append(item['branch'])
    reindex = {query: i for i, query in enumerate(indices)}
    for original_path in source_visible_paths:
        run = []
        for query in original_path:
            if query in reindex:
                run.append(reindex[query])
            elif run:
                paths.append(run); run=[]
        if run:
            paths.append(run)
    return dict(empty, disposition='source_line_family_supported' if family_unique else
        'insufficient_distributed_source_line_span' if not enough_span else 'competing_source_line_families_unknown',
        actual_source_uv=np.asarray(output, float).reshape(-1, 2),
        source_query_indices=np.asarray(indices, np.int64),
        source_candidate_indices=np.asarray(branches, np.int64), source_visible_paths=paths,
        source_line_support_qualified=family_unique, readings=rows,
        actual_source_line_organizer=dict(basis=basis, normal=normal, station_center=center,
            coefficients=best['coef'], unique_source_support_length_px=best['support'],
            competing_family_support_length_px=rival, organizing_curve_is_not_output_measurement=True),
        global_source_line_organizer_ignores_local_contrast_dominance=True)
