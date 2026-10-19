"""Migrate an absolute native-time initializer through its saved canonical chart.

A tangent update added to an already expanded physical vector is useful for
initializer evaluation, but its final floating point bits need not equal the
saved canonical coefficients expanded by the running model. The canonical q
and all original chart auxiliaries remain authoritative. The owning original
model must still compare its actual effective q byte for byte with this result
and run its complete guard before any source evaluation.
"""
import hashlib
import numpy as np

from .first_shot_time_detail_stage import CenteredNativeTimeDetailPlan


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise ValueError('Frozen native initialization differs: ' + label)


def expand_absolute_initializer(*, canonical_parent, canonical_candidate,
                                physical_parent, auxiliary, programme_count,
                                historical_physical_candidate=None):
    """Return the exact expansion of one absolute, time-only canonical proposal.

    No increment is applied to a later accepted q. No coordinates are fixed in
    the joint solver. This function neither creates an owner nor evaluates a
    source, Jacobian, or guard. Its physical vector is an expectation that the
    original full model must independently authenticate before installation.
    """
    parent = np.asarray(canonical_parent)
    candidate = np.asarray(canonical_candidate)
    physical = np.asarray(physical_parent)
    if (parent.dtype != np.float64 or candidate.dtype != parent.dtype
            or candidate.shape != parent.shape or physical.dtype != parent.dtype
            or physical.shape != parent.shape or not np.isfinite(parent).all()
            or not np.isfinite(candidate).all() or not np.isfinite(physical).all()
            or not 0 < programme_count <= len(parent)):
        raise ValueError('Complete finite matching canonical and physical arrays required')
    plan = CenteredNativeTimeDetailPlan.from_parent(dict(q=parent, aux=auxiliary))
    active = np.asarray(auxiliary['time_detail_active_indices'])
    controls = np.r_[plan.slots8.reshape(-1), plan.slots.reshape(-1)[active]]
    controls = np.unique(controls)
    native = plan.native.reshape(-1)
    if np.any(native >= programme_count):
        raise ValueError('Native camera coordinates leave the programme vector')
    fixed = np.ones(len(parent), bool)
    fixed[controls] = False
    _same(candidate[fixed], parent[fixed], 'non-temporal canonical coordinates')
    initial_expanded = plan.expand(parent)
    _same(initial_expanded[native], physical[native], 'parent native physical coordinates')
    expanded = plan.expand(candidate)
    expected = physical.copy()
    expected[native] = expanded[native]
    result = dict(schema='rinkspace.absolute-native-time-initialization-expansion.v1',
        canonical_parent_sha256=_sha(parent), canonical_candidate_sha256=_sha(candidate),
        expanded_candidate_sha256=_sha(expanded), physical_candidate_sha256=_sha(expected),
        physical_programme_candidate_sha256=_sha(expected[:programme_count]),
        original_native_pts_sha256=_sha(plan.pts), active_time_control_count=len(controls),
        changed_canonical_count=int(np.sum(parent.view(np.uint64) != candidate.view(np.uint64))),
        original_non_temporal_canonical_byte_exact=True, parent_native_physical_byte_exact=True,
        expected_physical_uses_original_plan_expand=True, absolute_candidate_loaded=True,
        later_accepted_q_not_used_in_expansion=True, initializer_anchors_added=False,
        original_full_model_authentication_still_required=True,
        source_or_J_evaluated=False, guard_relaxed=False)
    if historical_physical_candidate is not None:
        historical = np.asarray(historical_physical_candidate)
        if historical.dtype != np.float64 or historical.shape != (programme_count,) or not np.isfinite(historical).all():
            raise ValueError('Complete finite historical physical programme vector required')
        fixed_physical = np.ones(programme_count, bool)
        fixed_physical[native] = False
        _same(historical[fixed_physical], expected[:programme_count][fixed_physical],
              'historical non-temporal physical coordinates')
        a, b = expected[:programme_count], historical
        mismatch = np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))
        result['historical_tangent_physical_comparison'] = dict(
            mismatch_count=len(mismatch), all_mismatches_native=bool(np.isin(mismatch,native).all()),
            max_abs=float(np.max(np.abs(a-b), initial=0.)),
            max_same_sign_ulp=max((abs(int(a.view(np.uint64)[i])-int(b.view(np.uint64)[i]))
                for i in mismatch if np.signbit(a[i]) == np.signbit(b[i])), default=0),
            sign_difference_count=int(np.sum(np.signbit(a) != np.signbit(b))),
            historical_physical_sha256=_sha(historical),
            historical_tangent_vector_preserved_but_not_installed=True,
            no_tolerance_used_for_runtime_identity=True)
    return expanded, expected, result
