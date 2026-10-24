"""Diagnose yellow returns in the current exposure; no physical promotion.

The user's red-line reference includes the faint, nearly neutral lower strip.
A strong return from saturated yellow can still be internal to that strip.
This incomplete reader is therefore explicitly withdrawn from fit admission.
Its numerical profiles remain reproducible diagnostics for repairing it.
"""
import numpy as np
from scipy.signal import find_peaks

from .source_worker import _channels, _sample
from .native_print_track_readout import batch_interfaces, clear_profile


CONFIG = dict(
    version='same_native_solid_yellow_to_ice_interface_v1',
    search_normal_offsets=[-40., 40.], search_step_px=.5,
    tangent_offsets_px=[-2., 0., 2.], minimum_true_yellow_8bit=8.,
    minimum_material_return_8bit=8.,
    old_source_curve_used_only_for_navigation=True,
    model_projection_used=False, source_time_or_sigma_changed=False,
    physical_lower_strip_boundary_verified=False,
    source_promotion_status='withdrawn_pale_lower_strip_extent_not_resolved',
)


def read_contact_interfaces(rgb, mask, uv, normal=None):
    """Return diagnostic RGB returns; all physical eligibility stays unresolved.

    The caller owns original-video binding and the identity of this board
    instance.  The current saturated-yellow return does not establish the
    lower extent of the pale board strip in the user's boundary definition.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, bool)
    uv = np.asarray(uv, float).reshape(-1, 2)
    if rgb.shape != (*mask.shape, 3):
        raise ValueError('Current native RGB and exclusion mask differ')
    n = np.tile([0., 1.], (len(uv), 1)) if normal is None else np.asarray(normal, float).reshape(-1, 2).copy()
    if n.shape != uv.shape or not np.isfinite(n).all():
        raise ValueError('Finite source navigation normals required')
    length = np.linalg.norm(n, axis=1)
    if np.any(length < 1e-8):
        raise ValueError('Zero source navigation normal')
    n /= length[:, None]
    n[n[:, 1] < 0] *= -1
    channels = _channels(rgb)
    off = np.arange(-40., 40.0001, .5)
    tangent = np.c_[n[:, 1], -n[:, 0]]
    toff = np.asarray(CONFIG['tangent_offsets_px'])
    query = uv[:, None, None, :] + off[None, None, :, None] * n[:, None, None, :] + toff[None, :, None, None] * tangent[:, None, None, :]
    values = _sample(channels, query.reshape(-1, 2)).reshape(len(uv), len(toff), len(off), 3)
    raw = _sample(np.moveaxis(rgb.astype(np.float32), 2, 0), query.reshape(-1, 2)).reshape(len(uv), len(toff), len(off), 3)
    colour = np.median(raw, axis=1)
    chroma = np.median(values[..., 2], axis=1)
    yellow = np.minimum(colour[..., 0], colour[..., 1]) - colour[..., 2]
    imbalance = np.abs(colour[..., 0] - colour[..., 1])
    true_yellow = (yellow >= 8.) & (imbalance <= yellow)
    gradient = np.gradient(chroma, .5, axis=1)
    nav = uv.copy()
    eligible = np.zeros(len(uv), bool)
    evidence = [None] * len(uv)
    reasons = ['unresolved_solid_yellow_to_ice_transition'] * len(uv)
    h, w = mask.shape
    for i in range(len(uv)):
        core_signal = np.where(true_yellow[i] & (off >= -32) & (off <= 16), yellow[i], -np.inf)
        core = int(np.argmax(core_signal))
        if not np.isfinite(core_signal[core]):
            reasons[i] = 'no_actual_true_yellow_strip_core'
            continue
        # The whole material return must leave this strong yellow plateau.
        # This excludes both an internal core ripple and a later ice reflection.
        peaks, _ = find_peaks(gradient[i], height=.75, prominence=.25)
        candidates = []
        for k in peaks:
            if k <= core + 1 or k < 16 or k + 16 >= len(off):
                continue
            before = float(np.median(chroma[i, k-16:k-6]))
            after = float(np.median(chroma[i, k+6:k+16]))
            ybefore = float(np.median(yellow[i, k-16:k-6]))
            yafter = float(np.median(yellow[i, k+6:k+16]))
            contrast = after - before
            if contrast < 8. or ybefore < max(8., .45 * yellow[i, core]) or yafter > .65 * yellow[i, core]:
                continue
            if before > .55 * chroma[i, core] or after < .65 * chroma[i, core]:
                continue
            candidates.append((contrast, int(k), before, after, ybefore, yafter))
        if not candidates:
            reasons[i] = 'yellow_core_has_no_supported_iceward_material_return'
            continue
        candidates.sort(reverse=True)
        best = candidates[0]
        if any(a[0] >= .85 * best[0] and abs(off[a[1]] - off[best[1]]) > 8 for a in candidates[1:]):
            reasons[i] = 'competing_lower_material_returns'
            continue
        _, k, before, after, ybefore, yafter = best
        # Check the actual core, transition, and following ice on all tangents.
        q = query[i, :, max(0, core-4):k+13]
        qi = np.rint(q).astype(int)
        if (q[..., 0] < 4).any() or (q[..., 0] >= w-4).any() or (q[..., 1] < 4).any() or (q[..., 1] >= h-4).any():
            reasons[i] = 'material_support_outside_native_exposure'
            continue
        if mask[qi[..., 1], qi[..., 0]].any():
            reasons[i] = 'material_core_or_interface_occluded'
            continue
        nav[i] = uv[i] + off[k] * n[i]
        eligible[i] = True
        evidence[i] = dict(
            operator_version=CONFIG['version'], source_query_uv=uv[i].tolist(),
            source_navigation_only=True, yellow_core_uv=(uv[i]+off[core]*n[i]).tolist(),
            yellow_core_RGB=colour[i, core].tolist(), yellow_core_chroma=float(chroma[i, core]),
            yellow_core_min_RG_minus_B=float(yellow[i, core]),
            lower_transition_navigation_uv=nav[i].tolist(),
            before_chroma=before, after_chroma=after,
            before_min_RG_minus_B=ybefore, after_min_RG_minus_B=yafter,
            material_return_contrast=best[0],
            signed_transition='yellow_to_adjacent_ice; B-(R+G)/2 increasing',
            RGB_profile_sample_offsets_px=[-8., -4., 0., 4., 8.],
            tangent_offsets=toff.tolist(),
            actual_RGB_profile_samples=raw[i][:, k+np.array([-16, -8, 0, 8, 16]), :].tolist(),
            core_to_ice_material_span_mask_checked=True,
            physical_identity_from_source_colour_and_order=False,
            pale_lower_strip_extent_verified=False,
        )
    measured = batch_interfaces(nav, n, channels)
    valid = eligible & measured['valid'] & clear_profile(nav, n, mask) & clear_profile(measured['uv'], n, mask)
    # The final native operator may not switch from yellow's chroma return to
    # a neighbouring luminance/shadow edge in its narrow replay window.
    valid &= (measured['profile'][:, 3] == 2) & (measured['profile'][:, 4] == 1)
    for i in np.flatnonzero(eligible):
        if valid[i]:
            reasons[i] = 'unaccepted_yellow_return_pale_lower_strip_unresolved'
        else:
            reasons[i] = 'narrow_source_operator_did_not_replay_yellow_to_ice'
    return {**measured, 'numeric_valid': valid, 'valid': np.zeros(len(uv), bool),
            'navigation_uv': nav, 'evidence': evidence,
            'reasons': reasons, 'source_model_projection_used': False,
            'source_time_changed': False}
