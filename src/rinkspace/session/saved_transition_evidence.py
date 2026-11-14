"""Pure exact checks of saved signed-luminance transition arrays.

No RGB, file access, sampling, ink/guard check, identity, qualification or fit.
The caller authenticates the original row, call/query/return and array-file pins.
"""
from __future__ import annotations

import numpy as np


PROFILE_FLAGS = frozenset({
    'source_image_border_truncated', 'no_signed_material_support',
    'competing_material_reversal', 'separated_same_sign_transitions',
    'search_interval_truncated', 'sided_material_direction_unverified',
})


def _exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes() == right.tobytes())


def _exact_json(value, original):
    value = np.asarray(value)
    return (value.shape == original.shape and value.dtype.kind in 'fiu'
            and np.array_equal(value, original)
            and value.astype(original.dtype).tobytes() == original.tobytes())


def inspect_saved_transition(profile, reading, row, *, native_size):
    """Check original numeric arrays and return detailed results, never a grant.

    ``profile`` contains point/normal/offsets/smoothed/signed/normal_valid/q/
    candidate arrays. ``reading`` is the decoded original public return record.
    ``row`` is the unchanged original source row, and native_size is (width,height).
    Signed mass and interval-endpoint q10/q50/q90 follow the original public API
    and source-contour evidence block, including their operation order. Exact
    byte comparisons have no added tolerance. No input object is modified.
    """
    checks = {}
    result = dict(
        schema='rinkspace.saved_transition_numeric_evidence.v1', checks=checks,
        saved_numeric_pass=False, validates_ink=False, validates_guard=False,
        validates_entity=False, physical_identity_qualified=False,
        qualification_granted=False, new_source_measurement_created=False,
        inputs_modified=False, fit_executed=False,
    )

    def check(name, value):
        checks[name] = bool(value)
        return checks[name]

    def finish():
        result['failed_checks'] = [name for name, value in checks.items() if not value]
        result['saved_numeric_pass'] = bool(checks) and not result['failed_checks']
        return result

    try:
        width, height = native_size
        if not check('native_dimensions', type(width) is int and type(height) is int
                     and width > 0 and height > 0):
            return finish()
        arrays = {name: np.asarray(profile[name]) for name in (
            'point', 'normal', 'offsets', 'smoothed', 'signed', 'normal_valid',
            'q', 'candidate')}
        point, normal = arrays['point'], arrays['normal']
        offsets, smoothed = arrays['offsets'], arrays['smoothed']
        signed, normal_valid = arrays['signed'], arrays['normal_valid']
        q, candidate = arrays['q'], arrays['candidate']
        shape_ok = (point.shape == normal.shape == candidate.shape == (2,)
                    and q.shape == (3,) and offsets.ndim == 1 and len(offsets) >= 2
                    and smoothed.shape == normal_valid.shape == offsets.shape
                    and signed.shape == (len(offsets)-1,))
        dtype_ok = (normal_valid.dtype == np.dtype('bool')
                    and all(value.dtype == np.dtype('float64')
                            for name, value in arrays.items() if name != 'normal_valid'))
        check('original_array_shapes', shape_ok)
        check('original_array_dtypes', dtype_ok)
        if not shape_ok or not dtype_ok:
            return finish()
        finite = all(np.isfinite(value).all() for name, value in arrays.items()
                     if name not in ('normal_valid', 'signed'))
        check('finite_saved_values', finite)
        check('ordered_source_offsets', np.all(np.diff(offsets) > 0))
        flags = reading['source_flags']
        check('original_return_and_row_flags_zero',
              set(flags) == PROFILE_FLAGS and all(value is False for value in flags.values())
              and row['source_profile_flags'] == flags)
        check('normal_valid_consistent_with_clear_border_flag', normal_valid.all())
        raw_normal = np.asarray(row['normal'], np.float64)
        search = np.asarray(row['search_uv'], np.float64)
        row_shapes = raw_normal.shape == search.shape == (2,)
        check('row_search_and_normal_shapes', row_shapes)
        if not finite or not row_shapes:
            return finish()
        length = np.linalg.norm(raw_normal[None], axis=1)[0]
        if not check('finite_nonzero_original_normal',
                     np.isfinite(raw_normal).all() and np.isfinite(search).all()
                     and np.isfinite(length) and length > 1e-9):
            return finish()
        normalized = raw_normal/length
        check('original_point_and_normal_exact',
              _exact_json(row['search_uv'], point) and _exact(normal, normalized)
              and _exact(reading['source_search_point_uv'], point)
              and _exact(reading['source_search_normal'], normal))
        transition = reading['material_transition']
        if not check('original_signed_transition',
                     transition in ('dark_to_light', 'light_to_dark')):
            return finish()
        sign = 1 if transition == 'dark_to_light' else -1
        pair = normal_valid[:-1] & normal_valid[1:]
        expected_signed = np.diff(smoothed)*sign
        expected_signed[~pair] = np.nan
        check('original_signed_profile_exact', _exact(signed, expected_signed))
        check('finite_valid_signed_values', np.isfinite(signed[pair]).all())
        mass = np.where(pair, np.maximum(signed, 0), 0)
        total = float(mass.sum())
        result['original_positive_signed_mass'] = total if np.isfinite(total) else None
        if not check('positive_finite_signed_mass', np.isfinite(total) and total > 0):
            return finish()
        expected_q = np.interp([.1, .5, .9], np.r_[0., np.cumsum(mass)]/total, offsets)
        returned_q = np.asarray(reading['source_q10_q50_q90_normal_offsets_px'])
        check('original_interval_endpoint_quantiles_exact',
              _exact(returned_q, expected_q) and _exact(q, returned_q)
              and _exact_json(row['actual_normal_quantiles'], returned_q))
        check('ordered_original_quantiles', q[0] <= q[1] <= q[2])
        uv = point+normal*returned_q[1]
        check('original_q50_candidate_and_UV_bytes_exact',
              _exact(candidate, uv)
              and _exact(reading['actual_source_transition_candidate_uv'], uv)
              and _exact(reading['actual_source_uv'], uv)
              and _exact_json(row['uv'], uv))
        check('actual_UV_in_native_domain',
              np.isfinite(uv).all() and np.all(uv >= 0) and np.all(uv < [width, height]))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        checks['well_formed_saved_numeric_contract'] = False
        result['contract_error'] = type(error).__name__+': '+str(error)
    return finish()
