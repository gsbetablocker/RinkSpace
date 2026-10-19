"""Source-only spatial representatives; complete audit packs remain immutable.

This is a preparation view, not a solver admission or accuracy certificate.
Every actual exposure is retained. A missing visibility/adjacency declaration
uses isolated anchors rather than a guessed path. HELD is never compressed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from rinkspace.source_sampling import source_support_masses
from .strict_model_scope import file_sha
from .strict_source_pack_merge import _slice_row


SCHEMA = 'rinkspace.strict-source-spatial-representatives.v1'


def coarsen_measured_path(uv, paths, *, quantization_step_px, protected=()):
    """Retain measured UV, endpoints and source shape within half a native step.

    The tolerance comes from the source reader's declared native quantization;
    blur and paint width do not increase it. Integration mass is reassigned to
    the nearest retained station along each explicit path, without crossing
    visibility gaps. The original per-point source support is unchanged.
    """
    uv = np.asarray(uv, np.float64)
    if (uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv)
            or not np.isfinite(uv).all()):
        raise ValueError('Finite measured source path required')
    if (isinstance(quantization_step_px, bool)
            or not np.isfinite(quantization_step_px) or quantization_step_px <= 0):
        raise ValueError('Declared positive native source quantization required')
    protected = set(protected)
    if any(type(i) is not int or not 0 <= i < len(uv) for i in protected):
        raise ValueError('Protected measured source indices are out of range')
    paths = [list(path) for path in paths]
    # Require a disjoint partition: overlapping/reversed paths need an upstream
    # exact-source merge, not a new interpretation of their support here.
    flat = [i for path in paths for i in path]
    if (not flat or any(type(i) is not int for i in flat)
            or sorted(flat) != list(range(len(uv)))):
        raise ValueError('Explicit disjoint visible paths must cover each anchor')
    original_mass = source_support_masses(uv, paths)
    _, inverse, copies = np.unique(uv, axis=0, return_inverse=True, return_counts=True)
    original_mass /= copies[inverse]
    tolerance = .5 * float(quantization_step_px)
    keep = set(protected)
    maximum_deviation = 0.
    for path in paths:
        endpoints = {0, len(path)-1} | {j for j, i in enumerate(path) if i in protected}
        endpoints = sorted(endpoints)
        keep.update(path[j] for j in endpoints)
        stack = list(zip(endpoints[:-1], endpoints[1:]))
        while stack:
            lo, hi = stack.pop()
            if hi-lo <= 1:
                continue
            a, b = uv[path[lo]], uv[path[hi]]
            points = uv[np.asarray(path[lo+1:hi])]
            vector = b-a
            length2 = float(vector @ vector)
            fractions = (np.zeros(len(points)) if length2 == 0 else
                np.clip((points-a) @ vector / length2, 0., 1.))
            deviations = np.linalg.norm(points-(a+fractions[:, None]*vector), axis=1)
            at = int(np.argmax(deviations))
            if deviations[at] > tolerance:
                middle = lo+1+at
                keep.add(path[middle])
                stack.extend(((lo, middle), (middle, hi)))
            else:
                maximum_deviation = max(maximum_deviation, float(deviations[at]))
    retained = np.asarray(sorted(keep), np.int64)
    retained_mass = np.zeros(len(uv), float)
    assignments = np.empty(len(uv), np.int64)
    for path in paths:
        station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(uv[path], axis=0), axis=1))]
        chosen = np.asarray([j for j, i in enumerate(path) if i in keep], np.int64)
        # Each original station contributes once, only inside its own visible
        # path. No interpolated UV or observed extent endpoint is generated.
        for j, i in enumerate(path):
            target = path[int(chosen[np.argmin(abs(station[chosen]-station[j]))])]
            assignments[i] = target
            retained_mass[target] += original_mass[i]
    if not np.isclose(retained_mass.sum(), original_mass.sum(), rtol=1e-12, atol=1e-10):
        raise ValueError('Source support integration mass changed')
    return dict(kept_indices=retained, original_mass=original_mass,
        retained_mass=retained_mass[retained], original_to_retained=assignments,
        maximum_chord_deviation_px=maximum_deviation,
        native_quantization_step_px=float(quantization_step_px),
        shape_tolerance_px=tolerance)


def _declared_paths(row):
    """Use producer-declared visible paths or verified adjacent source stations."""
    n = len(row['uv'])
    paths = row.get('source_visible_paths')
    if paths is not None:
        return [list(path) for path in paths], 'explicit_visible_source_paths'
    # Ice readouts with a source query gap or competing query are separated.
    # Board adjacency additionally requires current neighbouring source pixels.
    name = 'source_query_indices' if row['scope'] == 'ice' else 'anchor_point_indices'
    values = np.asarray(row.get(name, []))
    confirmed = (row.get('final_current_RGB_pixel_precision_checked') is True
        if row['scope'] == 'ice' else row.get('source_neighbour_adjacency_confirmed') is True)
    if (not confirmed or values.shape != (n,) or values.dtype.kind not in 'iu'
            or n > 1 and np.any(np.diff(values) <= 0)):
        return [[i] for i in range(n)], 'no_declared_continuous_path_isolated_anchors'
    cuts = np.r_[0, np.flatnonzero(np.diff(values) != 1)+1, n]
    return [list(range(int(a), int(b))) for a, b in zip(cuts[:-1], cuts[1:])], name


def compile_strict_representatives(pack, *, source_reference, protected_points=None):
    """Prepare an exact measured subset and separate physical-curve weights.

    Missing source working support remains pending and receives no invented
    weight. All HELD pixels and caller-protected FIT anchors remain exact.
    Quantization shape checks alone do not certify solver information or speed.
    """
    path = Path(source_reference['path']).resolve()
    if file_sha(path) != source_reference['sha256']:
        raise ValueError('Representative source pack bytes changed')
    protected_points = {} if protected_points is None else protected_points
    selected, pending, records, rows, group_mass = [], [], [], [], defaultdict(float)
    record_by_row = {}
    fit_before = held = 0
    for ri, header in enumerate(pack.rows):
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        row = dict(header, uv=pack.values['uv'][lo:hi], sigma_px=pack.values['sigma_px'][lo:hi])
        n = len(row['uv'])
        if header['split'] == 'held':
            held += n
            selected.extend(range(lo, hi))
            record = dict(source_row_index=ri, split='held', kept_indices=list(range(n)), reason='held_exact')
            records.append(record); record_by_row[ri] = record
            continue
        fit_before += n
        valid = np.isfinite(row['sigma_px']) & (row['sigma_px'] > 0)
        pending.extend((lo+np.flatnonzero(~valid)).tolist())
        paths, basis = _declared_paths(row)
        # Splitting at unavailable support is necessary before any source path
        # quadrature; no short chord may bridge an unsupported original station.
        valid_paths = []
        for part in paths:
            current = []
            for i in part:
                if valid[i]:
                    current.append(i)
                elif current:
                    valid_paths.append(current); current=[]
            if current:
                valid_paths.append(current)
        local = np.flatnonzero(valid)
        if not len(local):
            continue
        back = {int(i):j for j,i in enumerate(local)}
        translated = [[back[i] for i in part] for part in valid_paths]
        uncertainty = header.get('source_readout_uncertainty', {})
        step = uncertainty.get('quantization_step_px', uncertainty.get('source_native_pixel_quantization_step_px'))
        if not isinstance(step, (int, float)) or isinstance(step, bool) or not np.isfinite(step) or step <= 0:
            # No invented quantization or spatial compression. Isolated source
            # anchors still share their physical curve's existing loss budget.
            basis = 'no_quantization_declaration_isolated_anchors'
            result = dict(kept_indices=np.arange(len(local)),
                retained_mass=np.ones(len(local)), original_mass=np.ones(len(local)),
                maximum_chord_deviation_px=0., shape_tolerance_px=None)
        else:
            protected = [back[i] for i in protected_points.get(ri, ()) if i in back]
            result = coarsen_measured_path(row['uv'][local], translated,
                quantization_step_px=step, protected=protected)
        keep = local[result['kept_indices']]
        selected.extend((lo+keep).tolist())
        group = (header['shot_id'], int(header['actual_pts']), header['scope'], header['physical_material_id'])
        group_mass[group] += float(result['retained_mass'].sum())
        rows.append((ri, keep, group, result['retained_mass']))
        record = dict(source_row_index=ri, split='fit', kept_indices=keep.tolist(),
            visible_path_basis=basis, visible_path_count=len(valid_paths),
            maximum_chord_deviation_px=result['maximum_chord_deviation_px'],
            source_shape_tolerance_px=result['shape_tolerance_px'],
            original_source_support_mass=float(result['original_mass'].sum()),
            retained_source_support_mass=float(result['retained_mass'].sum()))
        records.append(record); record_by_row[ri] = record
    for ri, keep, group, mass in rows:
        record = record_by_row[ri]
        record['point_integration_weights'] = (mass/group_mass[group]).tolist()
    fit_after = len(selected)-held
    return dict(schema=SCHEMA, source_reference=dict(source_reference),
        selected_global_point_indices=np.asarray(sorted(selected), np.int64),
        pending_support_global_point_indices=np.asarray(sorted(pending), np.int64), records=records,
        fit_points_before=fit_before, fit_points_after=fit_after, held_points=held,
        actual_exposures_with_fit_before=len({r['actual_pts'] for r in pack.rows if r['split']=='fit'}),
        actual_exposures_with_finite_fit_after=len({pack.rows[ri]['actual_pts'] for ri,_,_,_ in rows}),
        time_frames_subsampled=False, interpolated_observations_created=False,
        original_source_pack_modified=False, weights_applied_outside_huber=True,
        physical_curve_exposure_budget=1.,
        solver_information_verified=False, solver_speed_verified=False, solver_admission_verified=False,
        source_processing_is_not_manual_accuracy_acceptance=True, solver_started=False)


def representative_rows(pack, plan):
    """Materialize only exact source subsets for a diagnostic/public adapter."""
    if plan['schema'] != SCHEMA:
        raise ValueError('Exact-source representative plan required')
    if file_sha(plan['source_reference']['path']) != plan['source_reference']['sha256']:
        raise ValueError('Representative source pack bytes changed')
    if any(plan[key] is not False for key in ('solver_information_verified',
            'solver_speed_verified','solver_admission_verified','solver_started')):
        raise ValueError('Source shape preparation alone cannot authorize a solver')
    result = []
    seen = set()
    held_rows = {i for i,row in enumerate(pack.rows) if row['split']=='held'}
    budgets = defaultdict(float)
    selected = []
    for record in plan['records']:
        ri = record['source_row_index']; lo, hi = map(int,pack.values['row_offsets'][ri:ri+2])
        if ri in seen or record['split'] != pack.rows[ri]['split']:
            raise ValueError('Representative plan changed source row or FIT/HELD identity')
        seen.add(ri)
        row = dict(pack.rows[ri], **{name:pack.values[name][lo:hi] for name in
            ('uv','sigma_px','source_sides_uv','raw_candidate_uv','source_side_quantiles_uv','original_point_indices')})
        keep = np.asarray(record['kept_indices'], np.int64)
        if (keep.ndim != 1 or not len(keep) or np.any(keep<0) or np.any(keep>=hi-lo)
                or len(keep)>1 and np.any(np.diff(keep)<=0)):
            raise ValueError('Exact ordered measured source indices required')
        if record['split']=='held' and not np.array_equal(keep,np.arange(hi-lo)):
            raise ValueError('HELD source pixels must remain exact and complete')
        selected.extend((lo+keep).tolist())
        row = _slice_row(row, keep)
        uncertainty = row.get('source_readout_uncertainty', {})
        if 'actual_side_gradient_quantiles_uv' in uncertainty:
            values = np.asarray(uncertainty['actual_side_gradient_quantiles_uv'])
            if values.ndim and len(values) == hi-lo:
                uncertainty['actual_side_gradient_quantiles_uv'] = values[keep].copy()
        row['point_count'] = len(keep)
        row['source_representative_locator'] = dict(source_pack=plan['source_reference'],
            source_row_index=ri, kept_original_row_point_indices=keep.tolist())
        if record['split'] == 'fit':
            weight = np.asarray(record['point_integration_weights'],float)
            if (weight.shape!=(len(keep),) or not np.isfinite(weight).all() or np.any(weight<=0)
                    or not np.isfinite(row['sigma_px']).all()):
                raise ValueError('Measured working support and separate finite integration weights required')
            row['source_support_integration_weights'] = weight
            budgets[(row['shot_id'],row['actual_pts'],row['scope'],row['physical_material_id'])] += float(weight.sum())
        result.append(row)
    if (not held_rows.issubset(seen) or not np.array_equal(sorted(selected),plan['selected_global_point_indices'])
            or any(not np.isclose(v,1.,rtol=1e-12,atol=1e-12) for v in budgets.values())):
        raise ValueError('Representative coverage or physical curve budget changed')
    return result
