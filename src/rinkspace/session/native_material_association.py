"""Local association of measured annulus interfaces to known native paths."""
from __future__ import annotations

import numpy as np

from .material_tracks import assemble_fragments
from .native_shape_guidance import point_to_polyline


def source_interface_material_support(rgb, measurement, inventory, selected_ids, source_exclusion_mask=None):
    """Read visible dark-paint support from unchanged source RGB interfaces.

    q25 minus 3/6/9 pixels and q75 plus 3/6/9 pixels are sampled along the
    existing query normal. Luminance is the RGB channel mean in 0..255 units.
    For middle/light_to_dark the positive flank is paint; for outer/dark_to_light
    the negative flank is paint. Contrast is background median minus paint
    median. Flank spread is max-minus-min over the three source samples.

    Each candidate is qualified independently: all three background samples
    must exceed all three paint samples, and their median contrast must exceed
    both flank spreads. Other candidates, geometry and fitted error do not set
    a threshold. This local support check does not establish physical identity
    or distinguish a material edge from illumination. Weak/invalid support
    remains unresolved: it does not assert that a physical boundary is absent.
    Nothing is written or removed.
    Return a full-candidate boolean mask and JSON-safe per-candidate receipt;
    missing source samples use None, never projected or substituted pixels.
    """
    import cv2

    if len(inventory) != 2 or {p.get('world_curve_role') for p in inventory} != {'middle', 'outer'}:
        raise ValueError('Material support requires middle and outer interface roles')
    for path in inventory:
        transition = 'light_to_dark' if path['world_curve_role'] == 'middle' else 'dark_to_light'
        if (path.get('operator') != 'material_interface' or path.get('material_signal') != 'dark' or
                path.get('material_transition') != transition):
            raise ValueError('Material support requires the declared dark interface transition')
    image = np.asarray(rgb)
    if (image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1] or
            image.dtype.kind not in 'iuf' or not np.isfinite(image).all() or
            image.min() < 0 or image.max() > 255):
        raise ValueError('rgb must be a finite source RGB image in 0..255 units')
    uv = np.asarray(measurement['candidate_uv'])
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError('candidate_uv must have shape (N, 2)')
    count = len(uv)
    ids = np.asarray(selected_ids)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in 'iu'):
        raise ValueError('selected_ids must be original integer candidate IDs')
    ids = np.unique(ids.astype(np.int64))
    if (ids < 0).any() or (ids >= count).any():
        raise ValueError('selected_ids are outside the source candidate set')
    selected = np.zeros(count, bool); selected[ids] = True
    support = np.zeros(count, bool)
    paint = np.full(count, np.nan); background = np.full(count, np.nan)
    paint_spread = np.full(count, np.nan); background_spread = np.full(count, np.nan)
    source_valid = np.zeros(count, bool)
    contrast = np.full(count, np.nan); separation = np.full(count, np.nan)
    flanks_separated = np.zeros(count, bool)
    if source_exclusion_mask is not None:
        exclusion = np.asarray(source_exclusion_mask, bool)
        if exclusion.shape != image.shape[:2]:
            raise ValueError('Source exclusion mask must match the RGB image')
    else:
        exclusion = np.zeros(image.shape[:2], bool)
    if count:
        query = np.asarray(measurement['candidate_query'])
        qcurve = np.asarray(measurement['query_curve'])
        qnormal = np.asarray(measurement['query_normal'], float)
        quantiles = np.asarray(measurement['candidate_side_quantiles_uv'], float)
        if (query.shape != (count,) or query.dtype.kind not in 'iu' or qcurve.ndim != 1 or
                qcurve.dtype.kind not in 'iu' or qnormal.shape != (len(qcurve), 2) or
                quantiles.shape != (count, 2, 3, 2) or (query < 0).any() or
                (query >= len(qcurve)).any() or not np.isin(qcurve, [0, 1]).all()):
            raise ValueError('Invalid source interface/query arrays')
        normals = qnormal[query]
        q25, q75 = quantiles[:, 0, 0], quantiles[:, 0, 2]
        offsets = np.array([3., 6., 9.])
        negative = q25[:, None]-normals[:, None]*offsets[None, :, None]
        positive = q75[:, None]+normals[:, None]*offsets[None, :, None]
        flank_xy = np.stack([negative, positive], axis=1).reshape(count, 6, 2)
        # Include the actual interface and its transition support in the mask
        # check; bilinear sampling must not mix an excluded neighbour's RGB.
        support_xy = np.concatenate([np.asarray(uv, float)[:, None], q25[:, None],
                                     q75[:, None], flank_xy], axis=1)
        finite = np.isfinite(support_xy).all(axis=(1, 2)) & np.isfinite(normals).all(1)
        finite &= np.linalg.norm(normals, axis=1) > 0.
        height, width = image.shape[:2]
        inside = ((support_xy[:, :, 0] >= 0) & (support_xy[:, :, 0] <= width-1) &
                  (support_xy[:, :, 1] >= 0) & (support_xy[:, :, 1] <= height-1)).all(1)
        safe_xy = np.where(np.isfinite(support_xy), support_xy, -1.).astype(np.float32)
        excluded = cv2.remap(exclusion.astype(np.float32), safe_xy[:, :, 0], safe_xy[:, :, 1],
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.)
        source_valid = finite & inside & ~(excluded > 0.).any(1)
        flags = np.asarray(measurement.get('candidate_flags', np.zeros(count, np.uint8)))
        if flags.shape != (count,) or flags.dtype.kind not in 'iu':
            raise ValueError('Invalid candidate source flags')
        source_valid &= (flags & 3) == 0
        safe_flanks = safe_xy[:, 3:]
        pixels = cv2.remap(image.astype(np.float32), safe_flanks[:, :, 0], safe_flanks[:, :, 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
        luma = pixels.mean(axis=2).reshape(count, 2, 3)
        medians = np.median(luma, axis=2); spreads = np.ptp(luma, axis=2)
        paint_side = np.array([p['world_curve_role'] == 'middle' for p in inventory], int)[qcurve[query]]
        rows = np.arange(count)
        sampleable = finite & inside
        paint[sampleable] = medians[rows, paint_side][sampleable]
        background[sampleable] = medians[rows, 1-paint_side][sampleable]
        paint_spread[sampleable] = spreads[rows, paint_side][sampleable]
        background_spread[sampleable] = spreads[rows, 1-paint_side][sampleable]
        contrast = background-paint
        paint_samples = luma[rows, paint_side]
        background_samples = luma[rows, 1-paint_side]
        separation[sampleable] = (background_samples.min(axis=1)-paint_samples.max(axis=1))[sampleable]
        flanks_separated = sampleable & (separation > 0.)
        support = (selected & source_valid & flanks_separated &
                   (contrast > np.maximum(paint_spread, background_spread)))
    def nullable(values):
        return [float(value) if np.isfinite(value) else None for value in values]
    status = np.full(count, 'not_selected', dtype='<U32')
    status[selected] = 'unresolved'; status[support] = 'visible_material_supported'
    receipt = dict(selected_ids=ids.tolist(), supported_ids=np.flatnonzero(support).tolist(),
        unresolved_ids=np.flatnonzero(selected & ~support).tolist(), support_status=status.tolist(),
        paint_median_luma=nullable(paint), background_median_luma=nullable(background),
        contrast=nullable(contrast), paint_flank_spread=nullable(paint_spread),
        background_flank_spread=nullable(background_spread), source_valid=source_valid.tolist(),
        flanks_separated=flanks_separated.tolist(), source_separation_luma=nullable(separation),
        otsu_threshold_luma=None, threshold_method='local_source_flank_separation',
        flank_offsets_px=[3, 6, 9],
        luminance='RGB_channel_mean_0_255', flank_spread_definition='max_minus_min_of_three_source_samples',
        source_uv_unchanged=True, visible_material_qualification_only=True,
        physical_line_absence_asserted=False)
    return support, receipt


def associate_annulus_interfaces(measurement, inventory, native_polylines, eligible_candidates=None):
    """Return original candidate IDs, measured fragments and a small receipt.

    ``inventory`` contains exactly the middle/outer material_interface paths;
    ``native_polylines`` contains their finite native projections in that same
    order and source pixel domain. No path receives a separate transform.
    Physical identities must already be known and the native initial geometry
    reasonably close. This local search association does not independently
    establish physical identity, reject reflections or validate calibration.

    A clean, caller-eligible RGB candidate must be strictly nearer its expected
    native boundary than the other boundary. Its native distance cannot exceed
    the local inter-boundary distance at its closest native position. This is
    a source search association domain, not a fit tolerance or a weight.
    Per query, select the legal candidate nearest its expected complete native
    boundary, then retain only IDs belonging to existing assemble_fragments
    continuous arcs. Original query UV remains RGB search provenance even when
    the caller updates both native curves by one common source registration.
    Equal-distance per-query ties use the first original candidate ID; native
    role ties remain unselected. All candidate records and UV stay untouched.
    """
    if len(inventory) != 2 or len(native_polylines) != 2:
        raise ValueError("Annulus association needs exactly two native interface paths")
    roles = [path.get('world_curve_role') for path in inventory]
    if set(roles) != {'middle', 'outer'} or any(path.get('operator') != 'material_interface' for path in inventory):
        raise ValueError("Inventory must contain middle/outer material_interface paths")
    curves = [np.asarray(polyline) for polyline in native_polylines]
    # Validate both finite supports even when the source has no candidates.
    for curve in curves:
        point_to_polyline(np.empty((0, 2)), curve)
    uv = np.asarray(measurement['candidate_uv'])
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.dtype.kind not in 'iuf' or not np.isfinite(uv).all():
        raise ValueError("candidate_uv must be finite original (N, 2) source coordinates")
    count = len(uv)
    eligible = np.ones(count, bool) if eligible_candidates is None else np.asarray(eligible_candidates, bool)
    if eligible.shape != (count,):
        raise ValueError("Candidate eligibility shape mismatch")
    receipt = dict(status='no_source_candidates', candidates=count, roles=roles,
        clean_and_user_eligible=0, rejected_native_role_or_tie=0, rejected_local_domain=0,
        queries_with_legal_rgb_candidate=0, selected=0, source_uv_unchanged=True,
        selected_by_role={role:0 for role in roles},
        physical_identity_confirmed=False, association_scope='local_known_identity_native_seed')
    if not count:
        return [], [], receipt
    query = np.asarray(measurement['candidate_query'])
    quv = np.asarray(measurement['query_uv'])
    qcurve = np.asarray(measurement['query_curve'])
    flags = np.asarray(measurement['candidate_flags'])
    if (query.shape != (count,) or query.dtype.kind not in 'iu' or
        quv.ndim != 2 or quv.shape[1] != 2 or not np.isfinite(quv).all() or
        qcurve.shape != (len(quv),) or qcurve.dtype.kind not in 'iu' or
        flags.shape != (count,) or flags.dtype.kind not in 'iu'):
        raise ValueError("Invalid source query, curve or candidate flag arrays")
    if (query >= len(quv)).any() or (query < 0).any() or not np.isin(qcurve, [0, 1]).all():
        raise ValueError("Source queries must reference the supplied two native paths")
    expected = qcurve[query]
    clean = ((flags & 3) == 0) & eligible
    matches = [point_to_polyline(uv, curve) for curve in curves]
    distance = np.column_stack([match.distance_px for match in matches])
    own = distance[np.arange(count), expected]
    nearest_expected = own < distance[np.arange(count), 1-expected]
    gap = np.empty(count)
    for index in (0, 1):
        rows = np.flatnonzero(expected == index)
        gap[rows] = point_to_polyline(matches[index].closest_uv[rows], curves[1-index]).distance_px
    in_domain = own <= gap
    legal = clean & nearest_expected & in_domain
    receipt.update(clean_and_user_eligible=int(clean.sum()),
        rejected_native_role_or_tie=int((clean & ~nearest_expected).sum()),
        rejected_local_domain=int((clean & nearest_expected & ~in_domain).sum()))
    candidates = np.flatnonzero(legal)
    chosen = np.zeros(count, bool)
    if len(candidates):
        order = np.lexsort((candidates, own[candidates], query[candidates]))
        ordered = candidates[order]
        first = np.r_[True, np.diff(query[ordered]) != 0]
        chosen[ordered[first]] = True
    fragments = assemble_fragments(measurement, inventory, eligible_candidates=chosen)
    selected = sorted({int(cid) for fragment in fragments for cid in fragment['candidate_ids']})
    receipt.update(status='continuous_native_interface_association' if selected else 'no_continuous_native_interface',
                   queries_with_legal_rgb_candidate=int(chosen.sum()), selected=len(selected),
                   selected_by_role={roles[index]:int(np.sum(expected[selected] == index)) for index in (0, 1)})
    return selected, fragments, receipt
