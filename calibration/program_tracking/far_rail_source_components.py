"""Separate source blue-strip components before assigning a board identity.

This is a source-only candidate extractor. A blue connected component and its
upper colour transition are not, by themselves, a physical board annotation.
In particular, printed advertisement edges must remain separate candidates.
No camera, projected curve, world height, or fitted residual enters extraction.
"""
from __future__ import annotations

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np


def source_exclusion(image, person_mask=None):
    exclude = (np.zeros(image.shape[:2], bool) if person_mask is None
               else np.asarray(person_mask, bool).copy())
    hsv = cv2.cvtColor(image[25:65, 80:500], cv2.COLOR_BGR2HSV)
    graphic = ((hsv[..., 0] >= 130) & (hsv[..., 0] <= 175)
               & (hsv[..., 1] >= 65))
    if np.count_nonzero(graphic) > 110:
        exclude[24:254, 25:555] = True
    exclude[:40] = True
    exclude[905:, 1660:] = True
    return exclude


def blue_strip_candidates(image, person_mask=None, *, split_connected_columns=False):
    """Return measured components; keep the physical identity unassigned.

    Span and thin-column support rank inspection candidates only. They do not
    certify blue strips as rails. Thick or multiply connected columns (e.g. a
    banner attached to a rail at the picture edge) provide no upper-edge UV.
    """
    exclude = source_exclusion(image, person_mask)
    smooth = cv2.GaussianBlur(image, (5, 5), .9).astype(np.float32)
    b, g, r = cv2.split(smooth)
    score = b - .5 * (r + g)
    mask = ((score > 13) & (b > 45) & (b > g + 8) & (b > r + 10)
            & ~exclude).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x0, y0, width, height, area = map(int, stats[label])
        if width < 250:
            continue
        supported = []
        all_thickness = []
        columns = np.arange(max(16, x0), min(image.shape[1]-16, x0+width))
        for x in columns:
            ys = np.flatnonzero(labels[y0:y0+height, x] == label) + y0
            if not len(ys):
                continue
            all_thickness.append(int(len(ys)))
            # A single short run is a measurable thin colour strip. Reject
            # disconnected print fragments rather than bridging their pixels.
            if len(ys) > 30 or ys[-1]-ys[0]+1 != len(ys):
                if not split_connected_columns:
                    continue
                # A rail can touch printed blue ink elsewhere in the image.
                # Keep an actual local thin run separate from that component's
                # disconnected lower lettering; never bridge its missing pixels.
                runs = np.split(ys, np.flatnonzero(np.diff(ys)>1)+1)
                runs = [run for run in runs if 2 <= len(run) <= 30]
                if not runs:
                    continue
                ys = runs[0]
            supported.append((int(x), int(ys[0]), int(ys[-1])))
        if not supported:
            continue
        support = np.asarray(supported, int)
        uv, bands, strengths = [], [], []
        for x, upper, lower in support:
            if (x-16) % 28:
                continue
            peak_y = upper + int(np.argmax(score[upper:lower+1, x]))
            peak = float(score[peak_y, x])
            lo = max(1, upper-12)
            profile = score[lo:peak_y+1, x].astype(float)
            baseline = float(np.min(profile[:min(4, len(profile))]))
            amplitude = peak-baseline
            if amplitude < 12 or peak_y-lo < 2:
                continue
            crossings = []
            for fraction in (.1, .5, .9):
                threshold = baseline+fraction*amplitude
                below = np.flatnonzero(profile[:-1] <= threshold)
                if not len(below):
                    break
                i = int(below[-1])
                if profile[i+1] <= profile[i]:
                    break
                crossings.append(float(lo+i+(threshold-profile[i]) /
                                       (profile[i+1]-profile[i])))
            if len(crossings) != 3 or not (crossings[0] <= crossings[1] <= crossings[2]):
                continue
            if exclude[lo:peak_y+1, x].any():
                continue
            uv.append((float(x), crossings[1]))
            bands.append((crossings[0]-crossings[1], crossings[2]-crossings[1]))
            strengths.append(peak)
        if not uv:
            continue
        fraction = len(supported) / max(1, len(columns))
        candidates.append(dict(component_id=label,
            component_bbox_xywh=[x0, y0, width, height], component_area=area,
            thin_column_fraction=float(fraction),
            thickness_p50_p95=np.percentile(all_thickness, [50,95]).tolist(),
            source_support_span_px=int(np.ptp(support[:, 0])),
            source_ranking_score=float(np.ptp(support[:, 0])*fraction),
            observed_uv=np.asarray(uv, float),
            vertical_transition_interval_px=np.asarray(bands, float),
            color_peak_score=np.asarray(strengths, float),
            physical_identity='unassigned_blue_component_upper_transition',
            status='source_candidate_requires_physical_identity_review',
            interval_semantics='10-to-90 percent source colour transition, not a statistical confidence interval'))
    return sorted(candidates, key=lambda row: -row['source_ranking_score'])
