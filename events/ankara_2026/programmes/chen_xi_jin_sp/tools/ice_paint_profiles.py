"""Original-pixel inside/outside profiles for rink paint width measurements."""

from __future__ import annotations

import cv2
import numpy as np


def _sample_cross_sections(raw, mask, points, normal, radius):
    """Read colours on short parallel cuts so a single ice scratch has little influence."""
    s = np.arange(-radius, radius + 1, dtype=np.float32)
    tangent = np.c_[-normal[:, 1], normal[:, 0]]
    d = np.array([-3.0, 0.0, 3.0], np.float32)
    u = (
        points[:, None, None, :]
        + s[None, :, None, None] * normal[:, None, None, :]
        + d[None, None, :, None] * tangent[:, None, None, :]
    )
    maps = np.ascontiguousarray(u.reshape(len(points), -1, 2), dtype=np.float32)
    values = cv2.remap(raw, maps, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    covered = cv2.remap(
        mask, maps, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1
    )
    b, g, r = np.moveaxis(values.reshape(len(points), len(s), 3, 3).astype(np.float32), -1, 0)
    signal = np.stack(
        (-(0.114 * b + 0.587 * g + 0.299 * r), b - 0.42 * g - 0.58 * r, r - 0.48 * b - 0.52 * g),
        axis=-1,
    )
    signal = np.median(signal, axis=2)
    signal = cv2.GaussianBlur(signal, (7, 1), 1.10, borderType=cv2.BORDER_REPLICATE)
    valid = (covered.reshape(len(points), len(s), 3) == 0).all(axis=2)
    return s, signal, valid


def _remove_ice_baseline(signal, left, right, s):
    distance = max(float(s[right].mean() - s[left].mean()), 1.0)
    a, b = (float(np.quantile(signal[m], 0.34)) for m in (left, right))
    z = a + np.clip((s - s[left].mean()) / distance, -0.22, 1.22) * (b - a)
    return signal - z


def _interpolate_crossing(y, i, j, h):
    return i + (j - i) * (h - y[i]) / (y[j] - y[i])


def _outer_crossings(level, valid, s, width, threshold, *, new=False):
    """Fit the two half-contrast crossings of one resolved band."""
    room = (0.60 if new else 0.57) * width + 2.6
    search = (abs(s) <= min(room, 40.0)) & valid
    if search.sum() < 6:
        return None
    # Small local gradients can split a paint top but cannot create a new
    # outside edge. A sustained maximum selects the same material on both sides.
    w = max(3, min(11, int(width * 0.21) // 2 * 2 + 1))
    core = cv2.blur(level[None, :], (w, 1), borderType=cv2.BORDER_REPLICATE)[0]
    value = np.where(search, core - 0.018 * abs(s), -np.inf)
    peak = int(np.argmax(value))
    half = 0.50 * float(core[peak])
    if core[peak] < threshold:
        return None
    inside = level >= half
    if not inside[peak]:
        return None
    holes = (~inside[1:-1]) & inside[:-2] & inside[2:] & valid[1:-1]
    inside[1:-1] |= holes
    a = peak
    b = peak
    while a > 0 and inside[a - 1] and valid[a - 1]:
        a -= 1
    while b < len(s) - 1 and inside[b + 1] and valid[b + 1]:
        b += 1
    if a < 3 or b >= len(s) - 3 or not (valid[a - 3 : b + 4]).all():
        return None
    if (
        level[a - 1] >= half
        or level[b + 1] >= half
        or level[a] == level[a - 1]
        or level[b] == level[b + 1]
    ):
        return None
    left = _interpolate_crossing(level, a - 1, a, half) + s[0]
    right = _interpolate_crossing(level, b, b + 1, half) + s[0]
    measured = right - left
    lower = max(2.6, width * (0.36 if new else 0.39))
    if not lower <= measured <= max(7.6, 2.02 * width + 4.0):
        return None
    if abs((left + right) * 0.5) > max(5.4, min(34.0, 0.64 * width)):
        return None
    span = int(np.clip(measured * 0.22 + 2, 5, 25))
    lo = np.arange(max(0, a - span - 1), a - 1)
    hi = np.arange(b + 2, min(len(s), b + span + 2))
    lo = lo[valid[lo]]
    hi = hi[valid[hi]]
    if min(len(lo), len(hi)) < 3:
        return None
    inner = level[a + max(0, int(measured * 0.16)) : b + 1 - max(0, int(measured * 0.16))]
    if len(inner) < 2:
        return None
    interior = float(np.quantile(inner, 0.38))
    flank = np.array([np.quantile(level[lo], 0.56), np.quantile(level[hi], 0.56)])
    contrast = interior - flank
    slope = np.array(
        [np.max(np.diff(level[a - 3 : a + 3])), np.max(-np.diff(level[b - 2 : b + 4]))]
    )
    if min(contrast) < 0.49 * threshold or min(slope) < 0.20 * threshold * 0.28:
        return None
    if max(flank) > 0.64 * core[peak] or min(slope) < 0.14 * max(slope):
        return None
    return left, right, contrast, slope


def _confirm_saved_pair(colour, valid, s, width, tone):
    """Recognize an already documented pair if the photo confirms its two edges."""
    a, b = np.clip(np.rint(np.array([-width, width]) * 0.5 - s[0]).astype(int), 0, len(s) - 1)
    if a < 5 or b >= len(s) - 5 or not valid[a - 4 : b + 5].all():
        return False
    gate = int(np.clip(width * 0.1 + 1, 2, 5))
    depth = int(np.clip(width * 0.20 + 2, 3, 16))
    for index in ((0,) if tone == "dark" else ((2,) if tone == "red" else (1, 0))):
        v = colour[:, index]
        rise = np.gradient(v)
        lo = a - gate + int(np.argmax(rise[a - gate : a + gate + 1]))
        hi = b - gate + int(np.argmax(-rise[b - gate : b + gate + 1]))
        if abs(lo - a) > gate - 1 or abs(hi - b) > gate - 1:
            continue
        aa = np.arange(max(0, a - depth - 2), a - 2)
        bb = np.arange(b + 3, min(len(s), b + 3 + depth))
        aa = aa[valid[aa]]
        bb = bb[valid[bb]]
        if min(len(aa), len(bb)) < 3:
            continue
        cut = int(0.21 * width)
        inner = np.median(v[a + cut : b - cut + 1])
        contrast = inner - np.array([np.median(v[aa]), np.median(v[bb])])
        minimum = 4.8 if index == 0 else 2.3
        if min(contrast) >= minimum and min(rise[lo], -rise[hi]) >= 0.19 * minimum:
            if tone != "blue" or index == 1 or np.median(colour[a + cut : b - cut + 1, 1]) >= 1.2:
                return True
    return False


def _select_paint_pair(colour, valid, s, width, tone, *, new=False):
    outer = float(np.clip(0.72 * width + 2, 7.0, 122.0))
    leave = min(170.0, max(22.0, 0.43 * width + 14.0))
    sides = [
        valid & (s >= outer) & (s <= outer + leave),
        valid & (s <= -outer) & (s >= -outer - leave),
    ]
    # The complete cross-section and both neighbouring ice flanks must be visible.
    if min(side.sum() for side in sides) < 4:
        start = max(4.0, 0.56 * width)
        sides = [
            valid & (s >= start) & (s <= outer + leave),
            valid & (s <= -start) & (s >= -outer - leave),
        ]
    if min(side.sum() for side in sides) < 4:
        return None
    a, b = sides[1], sides[0]
    # Local clean-ice brightness slope is removed separately for each channel.
    dark, blue, red = (_remove_ice_baseline(colour[:, k], a, b, s) for k in range(3))
    if tone == "blue":
        recipes = ((dark + 0.70 * blue, 6.0, "luma_blue"), (blue, 3.20, "blue_chroma"))
    elif tone == "red":
        recipes = ((red, 3.9, "red_chroma"), (red + 0.26 * dark, 5.7, "red_luma"))
    else:
        recipes = ((dark, 6.0, "luma"),)
    choices = []
    for v, lim, name in recipes:
        hit = _outer_crossings(v, valid, s, width, lim, new=new)
        if hit is None:
            continue
        e0, e1, contrast, slope = hit
        interior = valid & (s >= e0 + 0.20 * (e1 - e0)) & (s <= e1 - 0.20 * (e1 - e0))
        if interior.sum() < 2:
            continue
        if tone == "blue":
            local = np.quantile(blue[interior], 0.51)
            competing = np.quantile(red[interior], 0.58)
            if local < (1.65 if new else 1.20) or local < 0.54 * competing:
                continue
            if name == "luma_blue" and np.median(dark[interior]) < 4.5:
                continue
        elif tone == "red":
            if np.median(red[interior]) < 2.7 or np.median(red[interior]) < 0.62 * np.median(
                blue[interior]
            ):
                continue
        elif np.median(red[interior]) >= 4.3 or np.median(blue[interior]) >= 5.7:
            continue
        score = min(contrast) / lim + min(slope) / lim
        # Complete luma transitions dominate when colour is a weaker, displaced
        # fringe. The blue chroma-only branch is for strokes with little luma.
        if name == "luma_blue":
            if np.median(dark[interior]) > 9.0:
                score += 4.2
        elif name == "red_chroma":
            score += 0.35
        choices.append((score, e0, e1, contrast, slope, name))
    return max(choices, key=lambda z: z[0]) if choices else None


def read_paint_profiles(raw, mask, pts, normal, width, tones, *, new=False):
    """Return signed outer flanks and checks of any saved edge pairs.

    The projected line only orients the cross-section. Interpolation is
    between adjacent original pixels at each detected paint–ice transition.
    """
    if not len(pts):
        return [], np.empty(0, bool)
    radius = int(np.clip(np.ceil(np.max(width) * 1.20 + 16.0), 21, 290))
    s, colours, valid = _sample_cross_sections(raw, mask, pts, normal, radius)
    result = []
    old = []
    for v, m, w, tone in zip(colours, valid, width, tones, strict=True):
        result.append(_select_paint_pair(v, m, s, w, tone, new=new))
        old.append(False if new else _confirm_saved_pair(v, m, s, w, tone))
    return result, np.asarray(old)
