"""Display only: all geometry/projection values are supplied by the native core."""
from __future__ import annotations
import re
import cv2
import numpy as np

COLORS = {"ice": (30, 230, 245), "board": (255, 175, 30), "advertisement": (70, 245, 80)}
PANEL_WIDTH = 640


def text(image, message, xy, scale=.42, color=(240, 240, 240)):
    at = tuple(map(int, xy))
    cv2.putText(image, str(message), at, cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 8, 8), 3, cv2.LINE_AA)
    cv2.putText(image, str(message), at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def top_down(world, cuts, height=1080, *, missing_count=0):
    panel = np.full((height, PANEL_WIDTH, 3), 22, np.uint8)
    xy = np.asarray(world)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = min((PANEL_WIDTH-70)/max(hi[0]-lo[0], 1e-6), max(1, height-220)/max(hi[1]-lo[1], 1e-6))
    center = (hi+lo)/2
    points = np.rint((xy-center)*[scale, -scale]+[PANEL_WIDTH/2, height/2]).astype(np.int32)
    ad_labels = {}
    for item in cuts:
        pp = points[item['begin']:item['end']]
        color = COLORS[item['layer']]
        if len(pp)>1:
            cv2.polylines(panel, [pp], item.get('closed', False), color, 1, cv2.LINE_AA)
        elif len(pp):
            cv2.circle(panel, tuple(pp[0]), 2, color, 1, cv2.LINE_AA)
        ad = re.search(r'Ad\d+', item['label'], re.IGNORECASE)
        if ad and len(pp):
            ad_labels.setdefault(ad.group(), []).append(pp[len(pp)//2])
    for label, positions in ad_labels.items():
        text(panel, label, np.median(positions, axis=0), .45, (255, 255, 255))
    text(panel, 'SAME MODEL / TOP VIEW', (18, 30), .59)
    text(panel, f'{len(cuts)} paths | missing / unlocalized: {missing_count}', (18, 57), .45)
    text(panel, 'Current state replay; no optimization', (18, 84), .45, (60, 210, 255))
    text(panel, 'Red: active UV at this exact source PTS', (18, height-42), .42, (100, 130, 255))
    text(panel, 'Geometry coordinates: rink-length units', (18, height-18), .40)
    return panel


def overlay(raw, projected, cuts, overview, pts, time_base, measured_uv, shot_id, frame_index, total):
    canvas = raw.copy()
    h, w = raw.shape[:2]
    uv = np.asarray(projected['uv'])
    status = np.asarray(projected['status'])
    good = (status == 0) & np.isfinite(uv).all(axis=1)
    in_image = good & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    # Large offscreen coordinates are clipped for display only. The native
    # projection output and its validity status are never modified.
    pixels = np.rint(np.clip(np.nan_to_num(uv, nan=0., posinf=1e7, neginf=-1e7), -1e7, 1e7)).astype(np.int32)
    labels = {}
    for item in cuts:
        begin, end = item['begin'], item['end']
        pp = pixels[begin:end]; gg = good[begin:end]
        color = COLORS[item['layer']]
        transitions = np.flatnonzero(np.diff(np.r_[False, gg, False]))
        for start, stop in transitions.reshape(-1, 2):
            if stop-start > 1:
                cv2.polylines(canvas, [pp[start:stop]], bool(item.get('closed') and start == 0 and stop == len(pp)), color, 1, cv2.LINE_AA)
            elif in_image[begin+start]:
                cv2.circle(canvas, tuple(pp[start]), 2, color, 1, cv2.LINE_AA)
        visible = np.flatnonzero(in_image[begin:end])
        if len(visible):
            ad = re.search(r'Ad\d+', item['label'], re.IGNORECASE)
            label = ad.group() if ad else item['label'].removeprefix('ice.').split(':')[0]
            if len(label) <= 16:
                labels.setdefault(label, []).append(pp[visible[len(visible)//2]])
    for label, positions in labels.items():
        at = np.median(positions, axis=0)
        at = np.clip(at, [5, 90], [w-90, h-24])
        text(canvas, label, at, .46)
    for point in measured_uv:
        if np.isfinite(point).all():
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (0, 0, 255), -1, cv2.LINE_AA)
    canvas[:65] = 18
    seconds = float(pts*time_base)
    text(canvas, f'{shot_id.upper()} | NATIVE SOURCE / CURRENT STATE REVIEW | NO NEW FIT', (15, 25), .62, (55, 215, 255))
    text(canvas, f'Source {seconds:.3f}s | PTS {pts} | frame {frame_index+1}/{total} | all ice / board / ads + IDs | UV {len(measured_uv)}', (15, 52), .48)
    side = overview.copy()
    text(side, f'Source PTS {pts}', (18, h-68), .46)
    return np.concatenate((canvas, side), axis=1), {
        'native_pts': int(pts), 'visible_samples': int(in_image.sum()),
        'projected_samples': int(good.sum()), 'failed_samples': int((~good).sum()),
        'active_source_points': len(measured_uv),
        'status_histogram': {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
        'maximum_root_equality_px': float(np.max(np.abs(np.asarray(projected['root_equality_px'])[good]))) if good.any() else None,
    }
