"""Small source-evidence helpers for the next RinkSpace candidate.

These functions do not fit, smooth, relocate or accept a body trajectory. Pixel
regions are engineering uncertainty sets, NOT calibrated covariance matrices.
No ice distance, velocity or angular-velocity threshold is introduced here.
"""
from __future__ import annotations

import numpy as np


def source_region_shape(row):
    """Keep the source region's extent; never derive precision from flow score.

    An explicit, separately source-reviewed override takes precedence. A source
    rectangle is circumscribed by an ellipse, matching support_refinement.py.
    The original row is not modified. This does not validate material identity.
    """
    selected = row.get('source_region_override', row)
    if 'region_shape_px2' in selected:
        shape = np.asarray(selected['region_shape_px2'], float)
    elif 'region_half_extent_uv' in selected:
        half = np.asarray(selected['region_half_extent_uv'], float)
        if half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError('Positive source rectangle half extents required')
        shape = np.diag(2 * half ** 2)
    else:
        raise ValueError('Source region shape is missing; flow score is not a substitute')
    if shape.shape != (2, 2) or not np.isfinite(shape).all():
        raise ValueError('Source region shape must be a finite 2x2 array')
    if not np.allclose(shape, shape.T, rtol=0, atol=1e-12):
        raise ValueError('Source region shape must be symmetric')
    if np.linalg.eigvalsh(shape)[0] <= 0:
        raise ValueError('Source region shape must be positive definite')
    return shape.copy()


def propagated_region_shape(source_shape, affine_linear, flow_spread_shape):
    """Conservative outer envelope for propagated source uncertainty.

    Shape matrices describe ellipses, not probabilities. With unknown error
    correlation, 2*(A S A.T + F) encloses the Minkowski sum of the transported
    source ellipse and the flow-error ellipse. The caller must supply A/F from
    source-image tracking. A saved scalar confidence cannot reconstruct them.
    """
    S = np.asarray(source_shape, float)
    A = np.asarray(affine_linear, float)
    F = np.asarray(flow_spread_shape, float)
    for label, value in [('source', S), ('affine', A), ('flow', F)]:
        if value.shape != (2, 2) or not np.isfinite(value).all():
            raise ValueError(label + ' must be finite 2x2')
    if not np.allclose(S, S.T) or np.linalg.eigvalsh(S)[0] <= 0:
        raise ValueError('Source shape must be positive definite')
    if not np.allclose(F, F.T) or np.linalg.eigvalsh(F)[0] < -1e-12:
        raise ValueError('Flow spread must be positive semidefinite')
    return 2 * (A @ S @ A.T + F)


def exposure_cell_weights(pts, valid, selected, *, time_base, native_step_pts):
    """Assign observed exposure cells to retained samples without filling holes.

    Each valid exposure owns exactly one native frame period. Its duration is
    assigned to the nearest selected sample in the SAME contiguous valid run.
    Unknown frames and excluded review frames split runs. Runs without any
    selected sample are reported, never donated to an unrelated anchor.

    Returns weights for selected samples in source order plus an audit record.
    These are integration weights, not independent-observation counts.
    """
    pts = np.asarray(pts)
    valid, selected = np.asarray(valid, bool), np.asarray(selected, bool)
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or len(pts) == 0:
        raise ValueError('Nonempty integer native PTS required')
    if valid.shape != pts.shape or selected.shape != pts.shape:
        raise ValueError('PTS and masks must have the same shape')
    if np.any(np.diff(pts) <= 0):
        raise ValueError('Strictly increasing PTS required')
    if isinstance(native_step_pts, bool) or int(native_step_pts) != native_step_pts or native_step_pts <= 0:
        raise ValueError('Explicit positive native frame step required')
    if len(time_base) != 2 or min(time_base) <= 0:
        raise ValueError('Positive source time base required')
    if np.any(selected & ~valid):
        raise ValueError('Unknown exposures cannot enter the source objective')
    selected_indices = np.flatnonzero(selected)
    if not len(selected_indices):
        raise ValueError('No selected source measurements')
    duration = int(native_step_pts) * float(time_base[0]) / float(time_base[1])
    weights = np.zeros(len(selected_indices))
    positions = {int(i): j for j, i in enumerate(selected_indices)}
    valid_indices = np.flatnonzero(valid)
    splits = np.flatnonzero((np.diff(valid_indices) != 1) |
                           (np.diff(pts[valid_indices]) != native_step_pts)) + 1
    runs = np.split(valid_indices, splits)
    records = []
    unrepresented = []
    for run in runs:
        if not len(run):
            continue
        representatives = run[selected[run]]
        record = dict(first_pts=int(pts[run[0]]), last_pts=int(pts[run[-1]]),
                      valid_exposures=len(run), selected_exposures=len(representatives))
        if not len(representatives):
            unrepresented.extend(map(int, pts[run])); record['assigned_seconds'] = 0.
        else:
            distances = abs(pts[run, None] - pts[representatives][None, :])
            owners = representatives[np.argmin(distances, axis=1)]
            for owner in owners:
                weights[positions[int(owner)]] += duration
            record['assigned_seconds'] = len(run) * duration
        records.append(record)
    audit = dict(schema='rinkspace.source-exposure-cell-quadrature.v1',
                 native_period_seconds=duration, valid_exposures=int(valid.sum()),
                 selected_exposures=len(selected_indices),
                 valid_exposure_seconds=float(valid.sum() * duration),
                 assigned_seconds=float(weights.sum()),
                 unrepresented_valid_pts=unrepresented, runs=records,
                 unknown_gaps_integrated=False, independent_observation_count=False)
    return weights, audit


def summarize_source_regions(details):
    """Report all source kinds and times, including disabled historical rows.

    Useful alongside the existing fixed 88--92s comparison, which should retain
    its original scope. This helper is descriptive and cannot accept a result.
    """
    groups = {}
    for role in sorted({r['role'] for r in details}):
        for kind in sorted({r['kind'] for r in details}):
            for foot in (0, 1):
                rows = [r for r in details if (r['role'], r['kind'], r['foot']) == (role, kind, foot)]
                if not rows:
                    continue
                error = np.array([r['outside_region_px'] for r in rows], float)
                if not np.isfinite(error).all():
                    raise ValueError('Nonfinite region error cannot be summarized as a pass')
                groups[f'{role}/{kind}/foot{foot}'] = dict(
                    count=len(rows), unique_exposures=len({r['pts'] for r in rows}),
                    first_pts=min(r['pts'] for r in rows), last_pts=max(r['pts'] for r in rows),
                    median_outside_px=float(np.median(error)), p95_outside_px=float(np.quantile(error, .95)),
                    maximum_outside_px=float(error.max()),
                    objective_enabled=sum(bool(r.get('optimization_enabled', True)) for r in rows if role == 'fit'),
                    region_interpretation='region_shape_px2 is a finite set, not a calibrated probability')
    return dict(schema='rinkspace.source-region-kind-summary.v1', groups=groups,
                scope='all supplied rows and source PTS', accuracy_accepted=False)


from .actual_observation_ledger import source_ledger_export


SOURCE_SEMANTICS = frozenset(('boot_material_region', 'blade_endpoint_region',
    'blade_midpoint_region', 'coarse_steel_shadow_region', 'unknown'))


def select_exposure_representatives(pts, valid, selected, *, native_step_pts):
    """Retain real valid selections and add one real exposure per missed run."""
    pts = np.asarray(pts)
    valid, selected = np.asarray(valid, bool), np.asarray(selected, bool)
    if (pts.ndim != 1 or pts.dtype.kind not in 'iu' or valid.shape != pts.shape
            or selected.shape != pts.shape or np.any(np.diff(pts) <= 0)):
        raise ValueError('Increasing integer PTS and matching masks required')
    if (isinstance(native_step_pts, (bool, np.bool_))
            or not isinstance(native_step_pts, (int, np.integer)) or native_step_pts <= 0):
        raise ValueError('Explicit positive integer native frame step required')
    selected = selected.copy() & valid
    indices = np.flatnonzero(valid)
    splits = np.flatnonzero((np.diff(indices) != 1)
        | (np.diff(pts[indices]) != native_step_pts)) + 1
    added = []
    for run in np.split(indices, splits):
        if len(run) and not selected[run].any():
            index = int(run[(len(run) - 1) // 2])
            selected[index] = True
            added.append(int(pts[index]))
    return selected, added


def prepare_track_evidence(track, *, excluded_review_pts=(), native_step_pts=256,
                           time_base=(1,12800), select_every=5):
    """Reuse a saved flow track, with gaps/protected exposures cutting time runs.

    This does not rerun tracking or imply a fixed blade-point identity. Returned
    optimization_use/weight_seconds are quadrature selection only; a consuming
    image factor must also pass independent source-semantic admission.
    """
    if isinstance(select_every, bool) or not isinstance(select_every, int) or select_every <= 0:
        raise ValueError('Positive integer source sampling stride required')
    pts, uv = np.asarray(track['pts']), np.asarray(track['uv'])
    valid = np.asarray(track['valid'], bool).copy()
    if uv.shape != (len(pts), 2) or valid.shape != pts.shape:
        raise ValueError('Track PTS/UV/valid shapes differ')
    if 'time_base' in track and tuple(map(int,track['time_base'])) != tuple(time_base):
        raise ValueError('Saved track and requested source time base differ')
    valid &= np.isfinite(uv).all(axis=1)
    valid &= ~np.isin(pts, list(excluded_review_pts))
    requested = np.asarray(track.get('optimization_use', np.zeros(len(pts),bool)),bool)
    if requested.shape != pts.shape:
        raise ValueError('Saved track selection shape differs')
    requested = requested | ((np.arange(len(pts)) % select_every) == 0)
    selected, added = select_exposure_representatives(pts,valid,requested,native_step_pts=native_step_pts)
    if valid.any():
        weights, audit = exposure_cell_weights(pts,valid,selected,
            time_base=time_base,native_step_pts=native_step_pts)
    else:
        weights = np.empty(0)
        audit = dict(schema='rinkspace.source-exposure-cell-quadrature.v1',
            native_period_seconds=native_step_pts*time_base[0]/time_base[1],
            valid_exposures=0,selected_exposures=0,valid_exposure_seconds=0.,
            assigned_seconds=0.,unrepresented_valid_pts=[],runs=[],
            unknown_gaps_integrated=False,independent_observation_count=False)
    full_weights = np.zeros(len(pts))
    full_weights[selected] = weights
    audit.update(added_representative_pts=added,
        selection_is_not_material_point_admission=True,
        excluded_review_pts=sorted(map(int,excluded_review_pts)))
    result = {k: np.array(v,copy=True) for k,v in track.items()}
    result.update(valid=valid,optimization_use=selected,
        exposure_weight_seconds=full_weights,source_exposure_audit=audit)
    return result


def apply_seed_review_overrides(seed_rows, review_overrides=None, *, source_sha256=None):
    """Preserve historical seeds and apply explicit per-seed source review only.

    Unreviewed steel/shadow seeds remain coarse and disabled. A midpoint name or
    flow confidence does not establish material identity. Review regions retain
    their rectangle extent unless a new source-reviewed region is supplied.
    """
    import copy
    rows = copy.deepcopy(list(seed_rows))
    if source_sha256 is None:
        hashes={r.get('source_sha256') for r in rows if r.get('source_sha256')}
        if len(hashes)>1:raise ValueError('Source seed hashes differ')
        source_sha256=next(iter(hashes),None)
    if review_overrides is None:
        overrides = []
    elif isinstance(review_overrides, dict):
        if source_sha256 is not None and review_overrides.get('source_sha256') != source_sha256:
            raise ValueError('Review and track source SHA256 differ')
        if tuple(review_overrides.get('time_base', [1,12800])) != (1,12800):
            raise ValueError('Seed review source time base differs')
        overrides = review_overrides.get('rows', [])
    else:
        overrides = list(review_overrides)
    by_pts = {}
    seen = set()
    for override in overrides:
        pt = override['pts']
        if isinstance(pt,bool) or not isinstance(pt,(int,np.integer)):
            raise ValueError('Integer per-seed review PTS required')
        key = (int(pt),override.get('source_semantic','unknown'),override.get('blade_endpoint'))
        if key in seen:
            raise ValueError('Duplicate source feature at one review exposure')
        seen.add(key)
        by_pts.setdefault(int(pt),[]).append(override)
    seed_pts = {int(row['pts']) for row in rows}
    for pt in sorted(set(by_pts) - seed_pts):
        # Additional direct source exposures are allowed; their own image
        # identity and region are required, never filled from saved flow.
        candidate = by_pts[pt][0]
        for key in ('source_frame','source_frame_sha256','centre_uv','foot','provenance'):
            if key not in candidate:
                raise ValueError('Additional source review requires '+key)
        source_region_shape(candidate)
        rows.append(dict(candidate,source_evidence_origin='additional_direct_source_review'))
    expanded = [(copy.deepcopy(row),override) for row in rows
        for override in by_pts.get(int(row['pts']),[None])]
    rows = []
    for row,override in expanded:
        rows.append(row)
        original = copy.deepcopy(row)
        row.update(source_semantic='coarse_steel_shadow_region',
            material_identity_confirmed=False,optimization_enabled=False,
            source_visibility=row.get('visibility','unknown'),
            source_uncertainty=row.get('observation','historical coarse steel/shadow region'))
        if override is not None:
            if not override.get('provenance'):
                raise ValueError('New seed review needs explicit source provenance')
            semantic = override.get('source_semantic','unknown')
            if semantic not in SOURCE_SEMANTICS:
                raise ValueError('Unknown source semantic')
            if row.get('role') == 'review' and override.get('role','review') != 'review':
                raise ValueError('Protected review seed cannot enter fit')
            if override.get('foot',row.get('foot')) != row.get('foot'):
                raise ValueError('A seed review cannot silently change anatomical association')
            if override.get('source_sha256',source_sha256) != source_sha256:
                raise ValueError('Reviewed seed source SHA differs')
            for key,value in override.items():
                if key not in ('pts','foot','source_sha256'):
                    row[key] = copy.deepcopy(value)
            # Region override is indivisible: old ellipse cannot shadow a newly
            # reviewed rectangle, and old rectangle cannot shrink a new ellipse.
            if 'region_half_extent_uv' in override and 'region_shape_px2' not in override:
                row.pop('region_shape_px2',None)
            row['source_visibility'] = override.get('visibility',row['source_visibility'])
            fixed_kind = semantic in ('blade_midpoint_region','blade_endpoint_region','boot_material_region')
            row['optimization_enabled'] = bool(row.get('optimization_enabled',False)
                and fixed_kind and row.get('material_identity_confirmed',False)
                and row.get('role','fit') == 'fit'
                and row['source_visibility'] in ('visible','partially_visible','visible_blurred'))
            if row['optimization_enabled']:
                if semantic=='blade_endpoint_region':
                    if 'centre_uv' not in override:raise ValueError('Endpoint review requires its own source centre')
                    source_region_shape(override)
                for field in ('source_frame','source_frame_sha256'):
                    if not row.get(field):
                        raise ValueError('Admitted direct seed requires source frame identity')
                if semantic == 'blade_endpoint_region' and row.get('blade_endpoint') not in ('front','rear'):
                    raise ValueError('Reviewed endpoint requires front/rear identity')
                if semantic == 'boot_material_region' and not row.get('adapter_point_ids'):
                    raise ValueError('Reviewed boot material requires explicit fixed adapter point IDs')
        row['region_shape_px2'] = source_region_shape(row).tolist()
        row['source_seed_original'] = original
        row['source_evidence_origin'] = 'direct_source_review' if override else 'historical_coarse_seed'
    return rows


def build_track_region_document(document, track, seed_rows, *, review_overrides=None,
                                excluded_review_pts=(), native_step_pts=256):
    """Build a new source document from a saved track plus real seed reviews.

    All old rows remain for historical reporting. Legacy flow-as-midpoint rows
    are disabled; saved flow has no affine/error envelope, so it is kept in the
    evidence ledger instead of being silently admitted as fixed material points.
    Returns (new_document, prepared_track). No input is changed, no files written.
    """
    import copy
    out = copy.deepcopy(document)
    reviewed = apply_seed_review_overrides(seed_rows,review_overrides,
        source_sha256=document['source_sha256'])
    protected = set(map(int,excluded_review_pts))
    protected.update(int(r['pts'])+delta for r in document['observations'] if r['role'] == 'review'
        for delta in (-native_step_pts,0,native_step_pts))
    prepared = prepare_track_evidence(track,excluded_review_pts=protected,
        native_step_pts=native_step_pts,time_base=tuple(document['time_base']))
    for row in out['observations']:
        if (row['kind'] == 'support_contact_region' or row['id'].startswith('derived_support_middle_')
                or row['id'].startswith('direct_blade_')):
            row.update(optimization_enabled=False,
                replaced_in_optimization_by='explicit per-seed source semantic review; historical row retained')
    known = {r['id'] for r in out['observations']}
    added = []
    for row in reviewed:
        if not row['optimization_enabled'] or row['pts'] in protected:
            continue
        suffix='_'+row['blade_endpoint'] if row['source_semantic']=='blade_endpoint_region' else ''
        rid = 'reviewed_source_seed_' + str(row['pts']) + suffix
        if rid in known:
            raise ValueError('Reviewed source seed already present; do not duplicate evidence')
        semantic = row['source_semantic']
        centre = np.asarray(row['centre_uv'],float)
        if centre.shape != (2,) or not np.isfinite(centre).all():
            raise ValueError('Reviewed direct source centre must be finite UV')
        shape = np.asarray(row['region_shape_px2'],float)
        # One directly reviewed exposure owns its own native cell. Neighbouring
        # flow cells do not acquire its material identity by proximity in time.
        record = dict(id=rid,source_id=document['source_id'],source_sha256=document['source_sha256'],
            shot_id=document['shot_id'],pts=int(row['pts']),time_base=list(document['time_base']),
            foot=row['foot'],foot_track='support_track',
            kind='boot_marker_region' if semantic == 'boot_material_region' else semantic,
            centre_uv=centre.tolist(),region_shape_px2=shape.tolist(),visibility='visible',
            contact_state=row.get('contact_state','unknown'),role='fit',
            reliability=float(row.get('reliability',1.)),
            objective_weight=native_step_pts*document['time_base'][0]/document['time_base'][1],
            provenance=row['provenance'],source_semantic=semantic,
            material_identity_confirmed=True,optimization_enabled=True,
            source_visibility=row['source_visibility'],source_uncertainty=row.get('source_uncertainty','finite source region'),
            source_evidence_origin='direct_source_review',dependency_seed_pts=[int(row['pts'])],
            image_footprint=dict(source_frame=row['source_frame'],source_frame_sha256=row['source_frame_sha256']),
            human_confirmed=row.get('human_confirmed',False))
        for key in ('blade_endpoint','adapter_point_ids'):
            if key in row: record[key] = copy.deepcopy(row[key])
        out['observations'].append(record)
        added.append(rid)
    out['source_evidence_rows'] = reviewed
    out['source_exposure_quadrature'] = dict(prepared['source_exposure_audit'],
        scope='saved source-flow valid cells; direct review admission is recorded separately')
    out['source_evidence_admission'] = dict(schema='rinkspace.source-semantic-admission.v1',
        added_direct_rows=added,
        direct_unique_exposures=len({r['pts'] for r in out['observations'] if r['id'] in added}),
        direct_single_exposure_seconds=native_step_pts*document['time_base'][0]/document['time_base'][1],
        saved_flow_material_rows_admitted=0,
        propagation_shape_missing=True,confidence_is_probability=False,
        missing='Saved flow contains no per-exposure affine and flow-error ellipse; it cannot reconstruct source uncertainty or fixed material identity.')
    return out, prepared
