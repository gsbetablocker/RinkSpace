"""Measure painted stripe centres from real colour profiles, not Hough edges."""
from __future__ import annotations

import cv2
import numpy as np


def recenter_painted_band(image, points, tangent, color, exclusion=None,
                         search_radius=128):
    """Return measured UV centres and strengths for a proposed stripe segment.

    Hough only supplies orientation and a search location. Each centre is the
    midpoint of two observed colour-band edges in an image-normal profile.
    Profiles lacking either edge, or touching an exclusion, are omitted.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    tangent = np.asarray(tangent, float)
    normal = np.array([-tangent[1], tangent[0]], float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.)
    b, g, r = cv2.split(smoothed)
    signal = b - .5*(r+g) if color == 'blue' else r - .5*(b+g)
    offsets = np.arange(-search_radius, search_radius+1, dtype=float)
    height, width = signal.shape
    measured, strengths = [], []
    for point in points:
        xy = point + offsets[:, None]*normal
        x, y = xy[:, 0], xy[:, 1]
        if np.any((x < 1) | (x >= width-1) | (y < 1) | (y >= height-1)):
            continue
        if exclusion is not None and np.any(exclusion[np.rint(y).astype(int), np.rint(x).astype(int)]):
            continue
        unit = tangent/max(np.linalg.norm(tangent), 1e-12)
        # A thin crossing line occupies only a minority of a long
        # along-stripe profile stack; the actual stripe persists throughout.
        span = 60 if color == 'blue' else 9
        strip = np.arange(-span, span+1, 3, dtype=float)[:, None]
        sx = (x[None]+strip*unit[0]).astype(np.float32)
        sy = (y[None]+strip*unit[1]).astype(np.float32)
        profile = np.median(cv2.remap(signal, sx, sy, cv2.INTER_LINEAR), axis=0)
        profile = cv2.GaussianBlur(profile[None], (19, 1), 3.).ravel()
        baseline = float(np.percentile(profile, 15))
        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])
        contrast = peak-baseline
        if peak < (4.5 if color == 'blue' else 6.5) or contrast < 3.5:
            continue
        level = baseline + .24*contrast
        # Blue hockey markings can cross a thin blue logo rule. A long
        # closing kernel joins the two painted bands and invents a centre
        # between them. Keep blue components separate and favour the full
        # stripe's integrated colour support over a narrow dark peak.
        closing_width = 3 if color == 'blue' else 19
        band = cv2.morphologyEx((profile >= level).astype(np.uint8)[None],
                                cv2.MORPH_CLOSE, np.ones((1, closing_width), np.uint8)).ravel()>0
        if color == 'blue':
            edges = np.diff(np.r_[False, band, False].astype(np.int8))
            candidates = []
            for lo, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                hi = stop-1
                if lo == 0 or hi == len(profile)-1 or hi-lo < 3:
                    continue
                middle = .5*(lo+hi)-search_radius
                support = float(np.maximum(profile[lo:stop]-baseline, 0).sum())
                support *= np.exp(-.5*(middle/max(search_radius, 1))**2)
                candidates.append((support, int(lo+np.argmax(profile[lo:stop]))))
            if not candidates:
                continue
            peak_index = max(candidates)[1]
        left = peak_index
        right = peak_index
        while left > 0 and band[left-1]:
            left -= 1
        while right < len(profile)-1 and band[right+1]:
            right += 1
        if left == 0 or right == len(profile)-1 or not 3 <= right-left <= 215:
            continue
        left_edge = left-1+(level-profile[left-1])/max(profile[left]-profile[left-1], 1e-6)
        right_edge = right+(profile[right]-level)/max(profile[right]-profile[right+1], 1e-6)
        center_offset = .5*(left_edge+right_edge)-search_radius
        if abs(center_offset) > 100:
            continue
        measured.append(point+center_offset*normal)
        strengths.append(contrast)
    measured = np.asarray(measured, float).reshape(-1, 2)
    strengths = np.asarray(strengths, float)
    if color == 'blue' and len(measured) >= 8:
        # Retain the actual measured pixels, never replace them with a
        # model projection. Crossing-rule mixtures form a second path or
        # abrupt steps, while one painted stripe has a coherent direction.
        line = cv2.fitLine(measured.astype(np.float32), cv2.DIST_WELSCH, 3., .01, .01).ravel()
        normal_fit = np.array([-line[1], line[0]])
        deviation = np.abs((measured-line[2:]) @ normal_fit)
        good = deviation <= 4.
        measured, strengths = measured[good], strengths[good]
    return measured, strengths
