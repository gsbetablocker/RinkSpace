"""Source-pixel advertisement corners for the fixed event rink.

The released advertisement XYZ is only a broad search hint.  Observations
come from native-frame color/edge intersections; invisible corners are absent.
The output remains an observation set, not an accepted world calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .motion import match_sift_descriptors
    from .projection import project_fast
    from .render_result import (DEFAULT_MASKS, DEFAULT_PEOPLE, DEFAULT_SHOTS,
                                PersonMasks, decode_selected)
except ImportError:
    from motion import match_sift_descriptors
    from projection import project_fast
    from render_result import (DEFAULT_MASKS, DEFAULT_PEOPLE, DEFAULT_SHOTS,
                               PersonMasks, decode_selected)


ROOT = Path(__file__).resolve().parents[2]
TEMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
ROLES = ("left_top", "left_bottom", "right_top", "right_bottom")


def _robust_line(x: np.ndarray, y: np.ndarray, threshold: float = 5.) -> tuple[np.ndarray, int] | None:
    """Fit an observed image edge; never fit a projected scene edge."""
    if len(x) < 6 or np.ptp(x) < 18:
        return None
    good = np.ones(len(x), bool)
    for _ in range(5):
        if np.count_nonzero(good) < 6:
            return None
        coef = np.polyfit(x[good], y[good], 1)
        residual = np.abs(y-np.polyval(coef, x))
        next_good = residual < threshold
        if np.array_equal(good, next_good):
            break
        good = next_good
    if np.count_nonzero(good) < 6 or np.ptp(x[good]) < 18:
        return None
    return np.polyfit(x[good], y[good], 1), int(np.count_nonzero(good))


def _edge_step(frame: np.ndarray, x: int, y: int, axis: int,
               radius: int = 10) -> tuple[int, float] | None:
    """Read the strongest actual RGB step near a *candidate* border pixel."""
    height, width = frame.shape[:2]
    if axis == 0:
        if not (4 <= y < height-4):
            return None
        positions = np.arange(max(4, x-radius), min(width-4, x+radius+1))
        if not len(positions):
            return None
        a = frame[y, positions+2].astype(np.float32)
        b = frame[y, positions-2].astype(np.float32)
    else:
        if not (4 <= x < width-4):
            return None
        positions = np.arange(max(4, y-radius), min(height-4, y+radius+1))
        if not len(positions):
            return None
        a = frame[positions+2, x].astype(np.float32)
        b = frame[positions-2, x].astype(np.float32)
    strengths = np.linalg.norm(a-b, axis=1)
    best = int(np.argmax(strengths))
    return int(positions[best]), float(strengths[best])


def _white_context(frame: np.ndarray, person: np.ndarray, x: int, ylo: int,
                   yhi: int, side: str) -> bool:
    height, width = frame.shape[:2]
    outside_x = x-14 if side == "left" else x+14
    if outside_x < 5 or outside_x >= width-5:
        return False
    ys = np.linspace(max(10, ylo+12), min(height-11, yhi-12), 25).astype(int)
    samples = frame[ys, outside_x]
    visible = ~person[ys, outside_x]
    if np.count_nonzero(visible) < 9:
        return False
    brightness = np.mean(samples[visible], axis=1)
    saturation = cv2.cvtColor(samples[visible, None, :], cv2.COLOR_BGR2HSV)[:, 0, 1]
    return float(np.mean((brightness > 145) & (saturation < 75))) > .54


def _color_rectangle(frame: np.ndarray, person: np.ndarray,
                     projected: dict[str, np.ndarray],
                     color: str, max_left_shift: float = 350.
                     ) -> dict[str, dict] | None:
    """Read an outer color rectangle; pixels and final lines come from source."""
    height, width = frame.shape[:2]
    left_guess = np.mean([projected["left_top"][0], projected["left_bottom"][0]])
    xs = [float(x[0]) for x in projected.values()]
    ys = [float(x[1]) for x in projected.values()]
    # A verified same-print SIFT match provides a much tighter *search*
    # region than the old scene geometry.  Do not join an adjacent ad or the
    # long blue dasher rail into the color component.
    x_margin = 85 if max_left_shift <= 150 else 320
    y_margin = 70 if max_left_shift <= 150 else 140
    x0 = max(0, int(min(xs)-x_margin))
    x1 = min(width, int(max(xs)+x_margin))
    y0 = max(0, int(min(ys)-y_margin))
    y1 = min(height, int(max(ys)+y_margin))
    if x1-x0 < 80 or y1-y0 < 70:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "purple":
        colored = cv2.inRange(hsv, (118, 55, 35), (172, 255, 230))
    elif color == "navy":
        colored = cv2.inRange(hsv, (106, 55, 30), (129, 255, 220))
    elif color == "blue":
        colored = cv2.inRange(hsv, (97, 42, 40), (129, 255, 230))
    elif color == "red":
        colored = (cv2.inRange(hsv, (0, 65, 45), (15, 255, 240))
                   | cv2.inRange(hsv, (165, 65, 45), (179, 255, 240)))
    else:
        raise ValueError(color)
    roi = colored[y0:y1, x0:x1]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (15, 13)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
    candidates = []
    for i in range(1, count):
        cx, cy, cw, ch, area = map(int, stats[i])
        if area < 1000 or cw < 45 or ch < 35:
            continue
        candidates.append((i, cx+x0, cy+y0, cw, ch, area))
    if not candidates:
        return None
    largest = max(item[5] for item in candidates)
    eligible = [item for item in candidates if item[5] >= max(1000, largest*.14)]
    selected = min(eligible, key=lambda item: abs(item[1]-left_guess))
    if abs(selected[1]-left_guess) > max_left_shift:
        return None
    chosen = [selected]
    sx, sy, sw, sh = selected[1:5]
    for other in candidates:
        if other is selected:
            continue
        ox, oy, ow, oh = other[1:5]
        overlap = max(0, min(sy+sh, oy+oh)-max(sy, oy))
        gap = ox-(sx+sw)
        bridge_person = False
        if 0 <= gap < 390:
            yy0, yy1 = max(sy, oy), min(sy+sh, oy+oh)
            bridge_person = bool(yy1 > yy0 and ox > sx+sw and
                                 person[yy0:yy1, sx+sw:ox].mean() > .22)
        if (0 <= gap < max(70, .16*(max(xs)-min(xs)), 390 if bridge_person else 0)
                and overlap > .55*min(sh, oh)
                and other[5] > .08*selected[5]):
            chosen.append(other)
    component = np.zeros_like(roi, dtype=np.uint8)
    for item in chosen:
        component[labels == item[0]] = 1
    py, px = np.nonzero(component)
    if len(px) < 1000:
        return None
    px = px+x0
    py = py+y0
    lo_x, hi_x = int(px.min()), int(px.max())
    lo_y, hi_y = int(py.min()), int(py.max())
    if hi_x-lo_x < 75 or hi_y-lo_y < 50:
        return None
    # Pixel extrema in many columns/rows supply source-only edge evidence.
    horizontal = {"top": [], "bottom": []}
    for x in range(lo_x+4, hi_x-3, max(4, (hi_x-lo_x)//90)):
        at = py[px == x]
        if len(at) < max(5, .17*(hi_y-lo_y)):
            continue
        for side, guess in (("top", int(at.min())), ("bottom", int(at.max()))):
            if person[max(0, guess-7):min(height, guess+8), max(0, x-7):min(width, x+8)].mean() > .2:
                continue
            refined = _edge_step(frame, x, guess, axis=1, radius=9)
            # The first/last color pixel is also a measured source boundary.
            # A stronger nearby blue rail or lettering must not move it.
            if color == "red":
                measured_y = (refined[0] if refined is not None and refined[1] >= 25
                              and abs(refined[0]-guess) <= 5 else guess)
                horizontal[side].append((x, measured_y, refined[1] if refined else 0.))
            elif refined is not None and refined[1] >= 25:
                horizontal[side].append((x, refined[0], refined[1]))
    vertical = {"left": [], "right": []}
    for y in range(lo_y+6, hi_y-5, max(3, (hi_y-lo_y)//70)):
        at = px[py == y]
        if len(at) < max(5, .12*(hi_x-lo_x)):
            continue
        for side, guess in (("left", int(at.min())), ("right", int(at.max()))):
            if person[max(0, y-7):min(height, y+8), max(0, guess-7):min(width, guess+8)].mean() > .2:
                continue
            refined = _edge_step(frame, guess, y, axis=0, radius=9)
            measured_x = (refined[0] if refined is not None and refined[1] >= 30
                          and abs(refined[0]-guess) <= 9 else guess)
            vertical[side].append((y, measured_x, refined[1] if refined else 0.))
    fits = {}
    for side, items in horizontal.items():
        if items:
            q = np.asarray(items)
            fits[side] = _robust_line(q[:, 0], q[:, 1], threshold=5.)
    for side, items in vertical.items():
        if items:
            q = np.asarray(items)
            fits[side] = _robust_line(q[:, 0], q[:, 1], threshold=5.)
    result = {}
    for side in ("left", "right"):
        if (fits.get(side) is None or not _white_context(frame, person,
                                                        lo_x if side == "left" else hi_x,
                                                        lo_y, hi_y, side)):
            continue
        if side == "right" and projected["right_top"][0] > width+350:
            continue
        for height_side in ("top", "bottom"):
            if fits.get(height_side) is None:
                continue
            # x = ax*y+b, y = ay*x+b; their crossing is the measured corner.
            vx, _ = fits[side]
            hy, _ = fits[height_side]
            denom = 1-vx[0]*hy[0]
            if abs(denom) < .15:
                continue
            x = (vx[0]*hy[1]+vx[1])/denom
            y = hy[0]*x+hy[1]
            if not (8 <= x < width-8 and 8 <= y < height-8):
                continue
            if abs(x-(lo_x if side == "left" else hi_x)) > 18:
                continue
            # An adjacent fragment can contain purple decoration above the
            # printed rectangle; assess the crossing against the primary
            # source component, not the union's global extremum.
            role = f"{side}_{height_side}"
            primary_y = (float(projected[role][1]) if max_left_shift <= 150
                         else sy if height_side == "top" else sy+sh-1)
            if abs(y-primary_y) > max(30, .2*sh):
                continue
            if person[int(y)-7:int(y)+8, int(x)-7:int(x)+8].mean() > .1:
                continue
            result[role] = {"uv": [round(float(x), 2), round(float(y), 2)],
                                                  "support": min(fits[side][1], fits[height_side][1]),
                                                  "sigma_px": 3.0,
                                                  "measurement_origin": "actual_source_edge_intersection"}
    return result or None


def _purple_ad2(frame: np.ndarray, person: np.ndarray,
                projected: dict[str, np.ndarray]) -> dict[str, dict] | None:
    return _color_rectangle(frame, person, projected, "purple")


def _printed_color_mask(frame: np.ndarray, color: str) -> np.ndarray:
    """Source-color support for a proposed printed rectangle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "navy":
        mask = cv2.inRange(hsv, (106, 55, 30), (129, 255, 220))
    elif color == "blue":
        mask = cv2.inRange(hsv, (97, 42, 40), (129, 255, 230))
    elif color == "red":
        mask = (cv2.inRange(hsv, (0, 65, 45), (15, 255, 240))
                | cv2.inRange(hsv, (165, 65, 45), (179, 255, 240)))
    else:
        raise ValueError(color)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) > 0


def _incident_edge_samples(mask: np.ndarray, person: np.ndarray,
                           white: np.ndarray, score: np.ndarray,
                           uv: np.ndarray, side: str,
                           role: str) -> dict | None:
    """Require a visible source border close to this corner, not an extrapolated line."""
    vertical = side in ("left", "right")
    inside = 1 if side in ("left", "top") else -1
    along = 1 if (role.endswith("top") if vertical else role.startswith("left")) else -1
    normal = float(uv[0] if vertical else uv[1])
    tangent = float(uv[1] if vertical else uv[0])
    height, width = mask.shape
    support, widths = [], []
    for distance in (7, 13, 19, 25, 31, 37, 43):
        t = int(round(tangent + along*distance))
        if not (25 <= t < (height if vertical else width)-25):
            continue
        candidates = []
        for n in range(int(round(normal))-18, int(round(normal))+19):
            if not (17 <= n < (width if vertical else height)-17):
                continue
            xs = [n + inside*k for k in (3, 5, 7)]
            xo = [n - inside*k for k in (3, 5, 7)]
            white_outside = [n - inside*k for k in (9, 11, 13)]
            if (min(xs+xo+white_outside) < 2 or
                    max(xs+xo+white_outside) >= (width if vertical else height)-2):
                continue
            if vertical:
                inside_color = np.count_nonzero(mask[t, xs])
                outside_color = np.count_nonzero(mask[t, xo])
                outside_white = np.count_nonzero(white[t, white_outside])
                blocked = person[t-3:t+4, n-3:n+4].mean() > .1
            else:
                inside_color = np.count_nonzero(mask[xs, t])
                outside_color = np.count_nonzero(mask[xo, t])
                outside_white = np.count_nonzero(white[white_outside, t])
                blocked = person[n-3:n+4, t-3:t+4].mean() > .1
            if (inside_color >= 2 and outside_color == 0 and
                    outside_white >= 2 and not blocked):
                candidates.append(n)
        if not candidates:
            continue
        n = min(candidates, key=lambda candidate: abs(candidate-normal))
        support.append((distance, n))
        positions = n + inside*np.arange(-12, 13)
        profile = score[t, positions] if vertical else score[positions, t]
        lo, hi = float(np.median(profile[:4])), float(np.median(profile[-4:]))
        if hi-lo > 8:
            level = (profile-lo)/(hi-lo)
            ten, ninety = np.flatnonzero(level >= .1), np.flatnonzero(level >= .9)
            if len(ten) and len(ninety) and ninety[0] >= ten[0]:
                widths.append(float(ninety[0]-ten[0]))
    if len(support) < 3 or sum(distance <= 19 for distance, _ in support) < 2:
        return None
    distances, positions = np.asarray(support, float).T
    line = np.polyfit(distances, positions, 1)
    scatter = float(1.4826*np.median(np.abs(positions-np.polyval(line, distances))))
    return {"support": len(support), "nearest_px": int(distances.min()),
            "blur_width_px": float(np.median(widths)) if widths else None,
            "line_scatter_px": scatter,
            "corner_offset_px": float(abs(line[1]-normal))}


def _source_quality_color_corners(frame: np.ndarray, person: np.ndarray,
                                  measured: dict[str, dict], color: str) -> dict[str, dict]:
    """Keep only directly visible two-edge corners and scale source uncertainty."""
    mask = _printed_color_mask(frame, color)
    bgr = frame.astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 80) & (bgr.mean(axis=2) > 140)
    if color == "red":
        score = bgr[:, :, 2] - .5*(bgr[:, :, 0]+bgr[:, :, 1])
    else:
        score = bgr[:, :, 0] - bgr[:, :, 2]
    result = {}
    for role, evidence in measured.items():
        if role not in ROLES:
            continue
        uv = np.asarray(evidence["uv"], float)
        if not np.isfinite(uv).all():
            continue
        sides = role.split("_")
        local = {side: _incident_edge_samples(mask, person, white, score,
                                              uv, side, role)
                 for side in sides}
        if any(value is None for value in local.values()):
            continue
        widths = [value["blur_width_px"] for value in local.values()
                  if value["blur_width_px"] is not None]
        blur = max(widths) if widths else 12.
        scatter = max(value["line_scatter_px"] for value in local.values())
        offset = max(value["corner_offset_px"] for value in local.values())
        sigma = float(max(float(evidence.get('sigma_px', 3.)),
                          np.clip(.4*blur + 1.5*scatter + .35*offset, 3., 20.)))
        result[role] = {**evidence, "sigma_px": round(sigma, 2),
                        "source_edge_quality": {"scale_method": "image_edge_heuristic_not_statistically_calibrated",
                                                "incident_sides": local,
                                                "blur_width_px": round(blur, 2),
                                                "line_scatter_px": round(scatter, 2),
                                                "corner_offset_px": round(offset, 2)}}
    return result


def _color_quad_from_proposal(frame: np.ndarray, person: np.ndarray,
                              quad: np.ndarray, color: str) -> dict[str, dict]:
    """Read printed color boundaries near a same-print SIFT proposal.

    Each accepted side is fit from multiple *native source* color transitions.
    The homography only assigns the physical ad identity and a search window.
    """
    mask = _printed_color_mask(frame, color)
    h, w = frame.shape[:2]
    top_samples, bottom_samples = [], []
    left_samples, right_samples = [], []
    for t in np.linspace(.12, .88, 38):
        top = (1-t)*quad[0]+t*quad[1]
        bottom = (1-t)*quad[3]+t*quad[2]
        x = int(round((top[0]+bottom[0])/2))
        if x < 10 or x >= w-10:
            continue
        for name, aim, inside_sign, samples in (
                ("top", float(top[1]), 1, top_samples),
                ("bottom", float(bottom[1]), -1, bottom_samples)):
            yi = int(round(aim))
            if yi < 38 or yi >= h-38:
                continue
            if person[max(0, yi-20):min(h, yi+21), x-6:x+7].mean() > .14:
                continue
            candidates = []
            for y in range(yi-27, yi+28):
                if not mask[y, x]:
                    continue
                outside = y-3*inside_sign
                inside = y+3*inside_sign
                if outside < 0 or inside < 0 or outside >= h or inside >= h:
                    continue
                if mask[outside, x] == 0 and np.count_nonzero(mask[
                        min(y, inside):max(y, inside)+1, x]) >= 3:
                    candidates.append(y)
            if candidates:
                y = min(candidates, key=lambda item: abs(item-aim))
                if abs(y-aim) <= 19:
                    samples.append((x, y))
    for t in np.linspace(.16, .84, 30):
        left = (1-t)*quad[0]+t*quad[3]
        right = (1-t)*quad[1]+t*quad[2]
        y = int(round((left[1]+right[1])/2))
        if y < 10 or y >= h-10:
            continue
        for aim, inside_sign, samples in ((float(left[0]), 1, left_samples),
                                          (float(right[0]), -1, right_samples)):
            xi = int(round(aim))
            if xi < 38 or xi >= w-38:
                continue
            if person[y-6:y+7, xi-20:xi+21].mean() > .14:
                continue
            candidates = []
            for x in range(xi-27, xi+28):
                if not mask[y, x]:
                    continue
                outside = x-3*inside_sign
                inside = x+3*inside_sign
                if outside < 0 or inside < 0 or outside >= w or inside >= w:
                    continue
                if mask[y, outside] == 0 and np.count_nonzero(mask[
                        y, min(x, inside):max(x, inside)+1]) >= 3:
                    candidates.append(x)
            if candidates:
                x = min(candidates, key=lambda item: abs(item-aim))
                if abs(x-aim) <= 19:
                    samples.append((y, x))
    side_lines = {}
    for name, samples in (("top", top_samples), ("bottom", bottom_samples),
                          ("left", left_samples), ("right", right_samples)):
        if len(samples) >= 7:
            arr = np.asarray(samples, float)
            fit = _robust_line(arr[:, 0], arr[:, 1], threshold=4.)
            if fit is not None and fit[1] >= 7:
                side_lines[name] = fit
    result = {}
    for role, (vertical, horizontal, guess) in {
            "left_top": ("left", "top", quad[0]),
            "right_top": ("right", "top", quad[1]),
            "right_bottom": ("right", "bottom", quad[2]),
            "left_bottom": ("left", "bottom", quad[3]),
    }.items():
        if vertical not in side_lines or horizontal not in side_lines:
            continue
        vx, vn = side_lines[vertical]
        hy, hn = side_lines[horizontal]
        denom = 1-vx[0]*hy[0]
        if abs(denom) < .15:
            continue
        x = (vx[0]*hy[1]+vx[1])/denom
        y = hy[0]*x+hy[1]
        if not (8 <= x < w-8 and 8 <= y < h-8):
            continue
        if np.linalg.norm(np.array([x, y])-guess) > 23:
            continue
        if person[int(y)-7:int(y)+8, int(x)-7:int(x)+8].mean() > .1:
            continue
        result[role] = {"uv": [round(float(x), 2), round(float(y), 2)],
                        "support": min(vn, hn), "sigma_px": 3.,
                        "measurement_origin": "actual_source_color_edge_intersection"}
    return result


def _qr_white_frame(frame: np.ndarray, person: np.ndarray,
                    projected_ad1: dict[str, np.ndarray] | None) -> dict[str, dict]:
    """Detect an internal printed QR white frame, never an outer ad corner."""
    if projected_ad1 is None:
        return {}
    # The QR is a local landmark on ad 1. Detection supplies real finder
    # pattern pixels; the surrounding white print frame is refined separately.
    detector = cv2.QRCodeDetector()
    found, points = detector.detect(frame)
    if not found or points is None:
        return {}
    quad = points.reshape(-1, 2)
    if len(quad) != 4 or not np.isfinite(quad).all():
        return {}
    center = np.mean(quad, axis=0)
    guess = np.mean(np.stack(list(projected_ad1.values())), axis=0)
    if np.linalg.norm(center-guess) > 350:
        return {}
    xlo, ylo = np.min(quad, axis=0)
    xhi, yhi = np.max(quad, axis=0)
    if not (25 <= xhi-xlo <= 420 and 25 <= yhi-ylo <= 420):
        return {}
    if person[max(0, int(ylo)-5):min(frame.shape[0], int(yhi)+6),
              max(0, int(xlo)-5):min(frame.shape[1], int(xhi)+6)].mean() > .02:
        return {}
    # QRCodeDetector returns the code border, not the larger white print box.
    # Therefore expose it explicitly as a *code* quadrilateral; the world
    # coordinates and larger white-box corners are intentionally unknown.
    roles = ("qr_code_left_top", "qr_code_right_top",
             "qr_code_right_bottom", "qr_code_left_bottom")
    return {role: {"uv": [round(float(x), 2), round(float(y), 2)],
                   "support": 1, "sigma_px": 3.0,
                   "measurement_origin": "actual_source_qr_detector",
                   "coordinate_status": "internal_print_unknown_world"}
            for role, (x, y) in zip(roles, quad)}


def _root_sift(desc: np.ndarray) -> np.ndarray:
    values = np.asarray(desc, np.float32)
    return np.sqrt(values/(values.sum(axis=1, keepdims=True)+1e-12))


def _historic_quads(scene: dict) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[int, int], dict[str, list[float]]] = {}
    for point in scene["advertisement_3d_corners"]:
        grouped.setdefault((int(point["source_pts"]), int(point["user_number"])), {})[
            point["role"]] = point["source_uv"]
    quads = {}
    for key, corners in grouped.items():
        if (len(corners) == 3 and "right_top" not in corners
                and all(name in corners for name in ("left_top", "left_bottom", "right_bottom"))):
            # This missing corner is *only* a template mask boundary.  It is
            # never emitted as a historical or programme observation.
            corners["right_top"] = (np.asarray(corners["right_bottom"])
                                    + np.asarray(corners["left_top"])
                                    - np.asarray(corners["left_bottom"])).tolist()
        if all(name in corners for name in ROLES):
            quads[key] = np.asarray([corners[k] for k in
                                     ("left_top", "right_top", "right_bottom", "left_bottom")],
                                    np.float32)
    return quads


def _build_templates(scene: dict, frames: dict[int, np.ndarray],
                     reviewed: dict | None) -> dict[int | str, dict]:
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.014)
    choices: dict[int | str, list[dict]] = {}
    for (pts, ad_id), quad in _historic_quads(scene).items():
        frame = frames.get(pts)
        if frame is None:
            continue
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        keys, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
        if descriptors is not None and len(keys) >= 8:
            choices.setdefault(ad_id, []).append({"pts": pts, "quad": quad,
                                                    "xy": np.asarray([k.pt for k in keys], np.float32),
                                                    "descriptors": _root_sift(descriptors)})
    templates = {ad_id: max(options, key=lambda item: len(item["xy"]))
                 for ad_id, options in choices.items()}
    if reviewed is not None:
        qr = [row for row in reviewed.get("internal_observations", [])
              if int(row["pts"]) == 229679360 and int(row["user_number"]) == 1]
        aliases = {row["role"]: row["uv"] for row in qr}
        required = ("qr_white_left_top", "qr_white_right_top",
                    "qr_white_right_bottom", "qr_white_left_bottom")
        if all(k in aliases for k in required) and 229679360 in frames:
            quad = np.asarray([aliases[k] for k in required], np.float32)
            mask = np.zeros(frames[229679360].shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
            keys, descriptors = sift.detectAndCompute(
                cv2.cvtColor(frames[229679360], cv2.COLOR_BGR2GRAY), mask)
            if descriptors is not None and len(keys) >= 15:
                templates["ad1_qr_white"] = {
                    "pts": 229679360, "quad": quad,
                    "xy": np.asarray([k.pt for k in keys], np.float32),
                    "descriptors": _root_sift(descriptors)}
    return templates


def _alternative_templates(scene: dict, frames: dict[int, np.ndarray],
                           primary: dict[int | str, dict]) -> dict[int, list[dict]]:
    """Keep other views of the same print when the largest SIFT template fails.

    A close-up reference can contain a skater, while a smaller high view shows
    the actual print unobscured.  The template homography proposes identity and
    a search region only; accepted corner pixels still come from target edges.
    """
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.014)
    alternatives: dict[int, list[dict]] = {}
    quads = _historic_quads(scene)
    for (pts, ad_id), quad in quads.items():
        if pts not in frames or ad_id not in primary or primary[ad_id]["pts"] == pts:
            continue
        mask = np.zeros(frames[pts].shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        keys, descriptors = sift.detectAndCompute(
            cv2.cvtColor(frames[pts], cv2.COLOR_BGR2GRAY), mask)
        if descriptors is None or len(keys) < 8:
            continue
        alternatives.setdefault(ad_id, []).append({
            "pts": pts, "quad": quad,
            "xy": np.asarray([key.pt for key in keys], np.float32),
            "descriptors": _root_sift(descriptors),
            "identity_region": "historic_print_rectangle",
        })
    # The small blue/green MEIWA bars are nearly textureless.  The 40-years
    # symbol and Japanese print beside them are on the same physical board
    # print and can establish identity without redefining the bar corners.
    pts, ad_id = 104693760, 5
    if pts in frames and (pts, ad_id) in quads and ad_id in primary:
        quad = quads[pts, ad_id]
        height, width = frames[pts].shape[:2]
        x0 = max(0, int(np.floor(quad[:, 0].min())) - 115)
        x1 = min(width, int(np.ceil(quad[:, 0].max())) + 140)
        y0 = max(0, int(np.floor(quad[:, 1].min())) - 40)
        y1 = min(height, int(np.ceil(quad[:, 1].max())) + 35)
        mask = np.zeros((height, width), np.uint8)
        mask[y0:y1, x0:x1] = 255
        keys, descriptors = sift.detectAndCompute(
            cv2.cvtColor(frames[pts], cv2.COLOR_BGR2GRAY), mask)
        if descriptors is not None and len(keys) >= 15:
            alternatives.setdefault(ad_id, []).append({
                "pts": pts, "quad": quad,
                "xy": np.asarray([key.pt for key in keys], np.float32),
                "descriptors": _root_sift(descriptors),
                "identity_region": "same_meiwa_print_context",
            })
    return alternatives


def _live_templates(observations: dict, frames: dict[int, np.ndarray],
                    people: PersonMasks) -> dict[int, list[dict]]:
    """Same-shot references from four already measured corners, not a camera.

    Local print appearance survives motion blur and zoom changes better than a
    distant historical view. These templates only establish print identity and
    the search area. Every output corner must still be read from target edges.
    """
    groups = {}
    for row in observations['observations']:
        if row.get('status') == 'measured' and row['role'] in ROLES:
            groups.setdefault((int(row['pts']), int(row['user_number'])), {})[row['role']] = row['uv']
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.014)
    result = {}
    for (pts, number), corners in sorted(groups.items()):
        if not all(role in corners for role in ROLES) or pts not in frames:
            continue
        frame = frames[pts]
        quad = np.asarray([corners[role] for role in
                           ('left_top', 'right_top', 'right_bottom', 'left_bottom')], np.float32)
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
            raise ValueError('Invalid measured reference quadrilateral')
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        person, _ = people.get(pts)
        mask[person] = 0
        keys, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
        if descriptors is None or len(keys) < 8:
            continue
        result.setdefault(number, []).append({'pts': pts, 'quad': quad,
            'xy': np.asarray([key.pt for key in keys], np.float32),
            'descriptors': _root_sift(descriptors),
            'identity_region': 'same_shot_measured_print_rectangle'})
    return result


class _ObservedMotionProposals:
    """Carry a verified print identity through actual within-shot image motion.

    The transported quad is only a source-edge search proposal. It is never
    emitted as an observation, and it cannot cross a camera cut. Limiting the
    proposal to nearby measured references prevents unbounded flow drift.
    """

    def __init__(self, motion_index: Path, shots: list[int], maximum_steps: int = 60):
        manifest = json.loads(motion_index.read_text(encoding='utf8'))
        self.maximum_steps = maximum_steps
        self.groups = []
        for shot in shots:
            with np.load(manifest['shots'][shot]['motion_npz'], allow_pickle=False) as data:
                self.groups.append(({name: data[name].copy() for name in data.files}, {}))

    def quad(self, reference: dict, target_pts: int):
        start = int(reference['pts'])
        steps = abs(target_pts-start)//256
        if not 0 < steps <= self.maximum_steps:
            return None
        for data, cache in self.groups:
            pairs = data['pair_pts']
            if not (pairs[0, 0] <= min(start, target_pts) and max(start, target_pts) <= pairs[-1, 1]):
                continue
            if (start-int(pairs[0, 0])) % 256 or (target_pts-start) % 256:
                return None
            lo, hi = (min(start, target_pts)-int(pairs[0, 0]))//256, (max(start, target_pts)-int(pairs[0, 0]))//256
            matrix = np.eye(3)
            for index in range(lo, hi):
                if index not in cache:
                    a, b = data['offsets'][index:index+2]
                    good = data['homography_inlier'][a:b].astype(bool) & (data['fb_error_px'][a:b] < 2.5)
                    p, q = data['xy0'][a:b][good], data['xy1'][a:b][good]
                    h, inlier = (cv2.findHomography(p, q, cv2.USAC_MAGSAC, 2.5)
                                 if len(p) >= 12 else (None, None))
                    cache[index] = (h if h is not None and inlier is not None
                                    and int(inlier.sum()) >= 12 else None)
                if cache[index] is None:
                    return None
                matrix = cache[index] @ matrix
                if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
                    return None
                matrix /= matrix[2, 2]
            if target_pts < start:
                try:
                    matrix = np.linalg.inv(matrix)
                except np.linalg.LinAlgError:
                    return None
            quad = cv2.perspectiveTransform(reference['quad'][None], matrix)[0]
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
                return None
            return quad, steps
        return None


def _meiwa_bar_envelope(frame: np.ndarray, person: np.ndarray,
                        quad: np.ndarray) -> dict[str, dict]:
    """Measure the blue top and green bottom of one identified MEIWA print."""
    height, width = frame.shape[:2]
    x0 = max(0, int(np.floor(quad[:, 0].min())) - 35)
    x1 = min(width, int(np.ceil(quad[:, 0].max())) + 35)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - 32)
    y1 = min(height, int(np.ceil(quad[:, 1].max())) + 32)
    if x1-x0 < 45 or y1-y0 < 45:
        return {}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = {}
    bounds = {}
    for name, low, high, horizontal_side in (
        ("blue", (95, 45, 35), (130, 255, 255), "top"),
        ("green", (32, 35, 35), (91, 255, 255), "bottom"),
    ):
        colored = cv2.inRange(hsv[y0:y1, x0:x1], low, high)
        colored[person[y0:y1, x0:x1]] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(colored)
        if count < 2:
            return {}
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        yy, xx = np.nonzero(labels == component)
        xx, yy = xx+x0, yy+y0
        if len(xx) < 150 or np.ptp(xx) < 35 or np.ptp(yy) < 8:
            return {}
        bounds[name] = (int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max()))
        horizontal, vertical = [], {"left": [], "right": []}
        for x in range(int(xx.min())+15, int(xx.max())-14, 4):
            at = yy[xx == x]
            if len(at) > 8:
                horizontal.append((x, int(at.min()) if name == "blue" else int(at.max())))
        for y in range(int(yy.min())+5, int(yy.max())-4, 2):
            at = xx[yy == y]
            if len(at) > 25:
                vertical["left"].append((y, int(at.min())))
                vertical["right"].append((y, int(at.max())))
        if len(horizontal) < 7 or any(len(v) < 7 for v in vertical.values()):
            return {}
        horizontal = np.asarray(horizontal)
        top_or_bottom = _robust_line(horizontal[:, 0], horizontal[:, 1], 4.)
        if top_or_bottom is None:
            return {}
        for side in ("left", "right"):
            samples = np.asarray(vertical[side])
            side_line = _robust_line(samples[:, 0], samples[:, 1], 4.)
            if side_line is None:
                return {}
            vx, hy = side_line[0], top_or_bottom[0]
            denom = 1-vx[0]*hy[0]
            if abs(denom) < .15:
                return {}
            x = (vx[0]*hy[1]+vx[1])/denom
            y = hy[0]*x+hy[1]
            if not (8 <= x < width-8 and 8 <= y < height-8):
                return {}
            if person[max(0, int(y)-8):min(height, int(y)+9),
                      max(0, int(x)-8):min(width, int(x)+9)].mean() > .1:
                continue
            found[f"{side}_{horizontal_side}"] = {
                "uv": [round(float(x), 2), round(float(y), 2)],
                "support": min(side_line[1], top_or_bottom[1]),
                "sigma_px": 6.,
                "measurement_origin": "actual_source_blue_green_bar_edge_intersection",
            }
    blue, green = bounds["blue"], bounds["green"]
    if (green[2]-blue[3] < 20 or green[2]-blue[3] > 70
            or abs(blue[0]-green[0]) > 22 or abs(blue[1]-green[1]) > 22
            or min(blue[1]-blue[0], green[1]-green[0]) < 45):
        return {}
    return found


def _cached_sift(shots: list[int], cache_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    features = {}
    for shot in shots:
        path = cache_dir/f"sift_shot_{shot:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            for i, stamp in enumerate(data["keyframe_pts"]):
                lo, hi = map(int, data["offsets"][i:i+2])
                features[int(stamp)] = (data["xy_native"][lo:hi].astype(np.float32),
                                        _root_sift(data["descriptors"][lo:hi]))
    return features


def _match_template(template: dict, target_xy: np.ndarray, target_desc: np.ndarray,
                    min_inliers: int = 8) -> tuple[np.ndarray, int] | None:
    src_ids, tgt_ids, _ = match_sift_descriptors(template["descriptors"], target_desc,
                                                ratio=.80, device="cuda")
    if len(src_ids) < min_inliers:
        return None
    source_xy = template["xy"][src_ids]
    target_xy = target_xy[tgt_ids]
    matrix, inliers = cv2.findHomography(source_xy, target_xy, cv2.USAC_MAGSAC,
                                        5., maxIters=3000, confidence=.995)
    if matrix is None or inliers is None:
        return None
    count = int(inliers.sum())
    if count < min_inliers or count < .55*len(src_ids):
        return None
    quad = cv2.perspectiveTransform(template["quad"][None], matrix)[0]
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
        return None
    width = max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3]))
    height = max(np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1]))
    if not (35 < width < 2400 and 20 < height < 800):
        return None
    return quad, count


def _proposal_near_scene(quad: np.ndarray, scene_corners: dict[str, np.ndarray],
                         max_centre_delta: float = 350.) -> bool:
    scene_xy = np.stack(list(scene_corners.values()))
    scene_center = np.median(scene_xy, axis=0)
    true_center = np.median(quad, axis=0)
    return bool(np.linalg.norm(scene_center-true_center) < max_centre_delta)


def _edge_quad(frame: np.ndarray, person: np.ndarray, quad: np.ndarray,
               radius: int = 19) -> dict[str, dict]:
    """Refine a feature-match proposal to four actual line intersections."""
    height, width = frame.shape[:2]
    edges = {}
    definition = {"top": (quad[0], quad[1], 1),
                  "bottom": (quad[3], quad[2], 1),
                  "left": (quad[0], quad[3], 0),
                  "right": (quad[1], quad[2], 0)}
    for side, (a, b, axis) in definition.items():
        samples = []
        for t in np.linspace(.1, .9, 26):
            x, y = np.rint((1-t)*a+t*b).astype(int)
            if not (radius+5 <= x < width-radius-5 and radius+5 <= y < height-radius-5):
                continue
            if person[y-8:y+9, x-8:x+9].mean() > .2:
                continue
            candidate = _edge_step(frame, x, y, axis=axis, radius=radius)
            if candidate is None or candidate[1] < 34:
                continue
            point = (x, candidate[0]) if axis == 1 else (y, candidate[0])
            samples.append(point)
        if len(samples) < 8:
            continue
        arr = np.asarray(samples, float)
        fit = _robust_line(arr[:, 0], arr[:, 1], threshold=5.)
        if fit is not None and fit[1] >= 8:
            edges[side] = fit
    output = {}
    names = {"left_top": ("left", "top", quad[0]),
             "right_top": ("right", "top", quad[1]),
             "right_bottom": ("right", "bottom", quad[2]),
             "left_bottom": ("left", "bottom", quad[3])}
    for role, (vertical, horizontal, proposal) in names.items():
        if vertical not in edges or horizontal not in edges:
            continue
        vx, vn = edges[vertical]
        hy, hn = edges[horizontal]
        denom = 1-vx[0]*hy[0]
        if abs(denom) < .15:
            continue
        x = (vx[0]*hy[1]+vx[1])/denom
        y = hy[0]*x+hy[1]
        if not (8 <= x < width-8 and 8 <= y < height-8):
            continue
        if np.linalg.norm(np.array([x, y])-proposal) > radius+9:
            continue
        if person[int(y)-8:int(y)+9, int(x)-8:int(x)+9].mean() > .1:
            continue
        output[role] = {"uv": [round(float(x), 2), round(float(y), 2)],
                        "support": min(vn, hn), "sigma_px": 4.,
                        "measurement_origin": "actual_source_edge_intersection"}
    return output


def run(keyframes_dir: Path, output: Path, shots: list[int],
        source_shots: Path = DEFAULT_SHOTS,
        masks_path: Path = DEFAULT_MASKS,
        people_path: Path = DEFAULT_PEOPLE,
        sift_dir: Path = TEMP/"motion",
        reviewed_seeds: Path | None = None,
        feature_source: str = "motion_cache", live_references: Path | None = None,
        frame_stride: int = 1, exclude_observations: list[Path] | None = None,
        exclude_pts: list[int] | None = None, motion_proposals: bool = False,
        start_pts: int | None = None, end_pts: int | None = None) -> dict:
    if feature_source not in ("motion_cache", "native"):
        raise ValueError("feature_source must be motion_cache or native")
    if not output.resolve().is_relative_to(TEMP.resolve()):
        raise ValueError("Advertisement cache must be outside OneDrive")
    output.mkdir(parents=True, exist_ok=True)
    scene_path = ROOT/"calibration/deliverables/scene.json"
    scene_bytes = scene_path.read_bytes()
    scene = json.loads(scene_bytes)
    source = json.loads(source_shots.read_text(encoding="utf-8"))
    source_hash = source["source_sha256_from_timeline"]
    model_hash = hashlib.sha256(scene_bytes).hexdigest()
    if frame_stride < 1:
        raise ValueError('frame_stride must be positive')
    if (start_pts is not None and end_pts is not None
            and start_pts > end_pts):
        raise ValueError('start_pts must not exceed end_pts')
    excluded = set(exclude_pts or [])
    exclusion_hashes = {}
    for path in exclude_observations or []:
        payload = path.read_bytes()
        frozen = json.loads(payload)
        if frozen['source_sha256'] != source_hash:
            raise ValueError('Excluded validation pixels belong to another source')
        excluded.update(int(row['pts']) for row in frozen['observations'])
        exclusion_hashes[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
    if reviewed_seeds is None:
        reviewed_seeds = output/"reviewed_seeds.json"
    reviewed = None
    if reviewed_seeds.is_file():
        reviewed = json.loads(reviewed_seeds.read_text(encoding="utf-8"))
        if reviewed.get("source_sha256") != source_hash:
            raise ValueError("Reviewed source corners belong to a different video")
    ads = {a["user_number"]: a for a in scene["advertisements"]}
    stamps: dict[int, np.ndarray] = {}
    for shot in shots:
        with np.load(keyframes_dir/f"shot{shot:02d}_keyframes.npz", allow_pickle=False) as data:
            if (str(data["source_sha256"]) != source_hash
                    or str(data["model_sha256"]) != model_hash):
                raise ValueError(f"Shot {shot} has wrong native source or scene")
            stamps.update((int(p), c.astype(float)) for p, c in
                          zip(data["pts"][::frame_stride], data["camera25"][::frame_stride])
                          if int(p) not in excluded
                          and (start_pts is None or int(p) >= start_pts)
                          and (end_pts is None or int(p) <= end_pts))
    people = PersonMasks(masks_path, people_path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This ad association requires the existing CUDA matcher")
    print(json.dumps({"descriptor_device": "cuda", "gpu": torch.cuda.get_device_name(0)}), flush=True)
    needed_refs = {stamp for stamp, _ in _historic_quads(scene)} | {229679360}
    live = None
    if live_references is not None:
        live = json.loads(live_references.read_bytes())
        if live['source_sha256'] != source_hash:
            raise ValueError('Live print reference source differs')
        if any(int(row['pts']) in excluded for row in live['observations']):
            raise ValueError('Validation frames cannot be live reference templates')
        needed_refs.update(int(row['pts']) for row in live['observations'])
    reference_frames = dict(decode_selected(Path(source["source_path"]), sorted(needed_refs)))
    templates = _build_templates(scene, reference_frames, reviewed)
    alternatives = _alternative_templates(scene, reference_frames, templates)
    local_templates = {}
    if live is not None:
        local_templates = _live_templates(live, reference_frames, people)
        for number, options in local_templates.items():
            if number not in templates:
                templates[number] = options[0]
            alternatives.setdefault(number, []).extend(options)
    proposal_motion = (_ObservedMotionProposals(Path(__file__).with_name('motion_index.json'), shots)
                       if motion_proposals else None)
    # Motion-cache features were selected for pairwise camera motion; that
    # coverage can miss a clearly visible printed advertisement. A fresh
    # source-image pass finds features throughout the unoccluded board area.
    target_features = _cached_sift(shots, sift_dir) if feature_source == "motion_cache" else {}
    fallback_sift = cv2.SIFT_create(nfeatures=1900 if feature_source == "native" else 900,
                                   contrastThreshold=.025)
    observations = []
    internal = []
    statuses = []
    for index, (pts, frame) in enumerate(decode_selected(Path(source["source_path"]), list(stamps))):
        person, mask_status = people.get(pts)
        projected = {}
        for n, ad in ads.items():
            corners = ad["corners_xyz_scene_units"]
            world = np.asarray([corners[k] for k in corners], float)
            uv, depth = project_fast(stamps[pts], world)
            if np.all(depth > 0) and np.isfinite(uv).all():
                projected[n] = dict(zip(corners, uv))
        if pts in target_features:
            target_xy, target_desc = target_features[pts]
        else:
            small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            keep = cv2.resize((~person).astype(np.uint8)*255, (960, 540),
                              interpolation=cv2.INTER_NEAREST)
            keys, descriptors = fallback_sift.detectAndCompute(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), keep)
            target_xy = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)*2
            target_desc = (_root_sift(descriptors) if descriptors is not None
                           else np.empty((0, 128), np.float32))
        found = _purple_ad2(frame, person, projected[2]) if 2 in projected else None
        if found:
            for role, evidence in found.items():
                observations.append({"pts": pts, "user_number": 2, "role": role,
                                     **evidence, "status": "measured",
                                     "mask_status": mask_status})
        ad_seen = {2: list(found or {})}
        identity_attempts = {}
        for ad_id in (1, 3, 4, 5, 6, 7, 20):
            if ad_id not in templates or ad_id not in projected:
                continue
            candidates = [{**templates[ad_id],
                           "identity_region": "historic_print_rectangle"}]
            candidates.extend(alternatives.get(ad_id, []))
            attempts = []
            valid = []
            for template in candidates:
                attempt = {"reference_pts": int(template["pts"]),
                           "identity_region": template["identity_region"]}
                matched = _match_template(template, target_xy, target_desc,
                                          min_inliers=11 if ad_id == 4 else 8)
                if matched is None:
                    attempt["status"] = "no_verified_same_print_match"
                else:
                    candidate_quad, candidate_inliers = matched
                    attempt["inliers"] = candidate_inliers
                    if _proposal_near_scene(candidate_quad, projected[ad_id],
                                            max_centre_delta=520.):
                        attempt["status"] = "verified_same_print_match"
                        valid.append((candidate_inliers, template, candidate_quad))
                    else:
                        attempt["status"] = "match_outside_broad_scene_roi"
                attempts.append(attempt)
            identity_attempts[ad_id] = attempts
            proposal_kind = 'same_print_sift_cuda'
            motion_steps = None
            if not valid and proposal_motion is not None:
                for reference in sorted(local_templates.get(ad_id, []),
                                        key=lambda row: abs(row['pts']-pts)):
                    transported = proposal_motion.quad(reference, pts)
                    if transported is None:
                        continue
                    candidate_quad, motion_steps = transported
                    if _proposal_near_scene(candidate_quad, projected[ad_id], max_centre_delta=520.):
                        valid.append((0, reference, candidate_quad))
                        proposal_kind = 'verified_print_identity_continuous_source_motion'
                        attempts.append({'reference_pts': reference['pts'], 'status': 'source_motion_identity_proposal',
                                         'native_steps': motion_steps})
                        break
            if not valid:
                continue
            inliers, template, quad = max(valid, key=lambda row: row[0])
            proposal = dict(zip(("left_top", "right_top", "right_bottom", "left_bottom"), quad))
            color = {1: "blue", 3: "navy", 6: "red", 7: "blue", 20: "blue"}.get(ad_id)
            measured = (_meiwa_bar_envelope(frame, person, quad) if ad_id == 5
                        else _color_quad_from_proposal(frame, person, quad, color)
                        if color else None)
            if color:
                component_measured = _color_rectangle(
                    frame, person, proposal, color, max_left_shift=110.) or {}
                measured = measured or {}
                for role, evidence in component_measured.items():
                    # Keep a second, independent source-boundary method only
                    # near the verified printed-instance proposal.
                    if (role not in measured and
                            np.linalg.norm(np.asarray(evidence["uv"])-proposal[role]) < 25):
                        measured[role] = evidence
            if not measured and ad_id == 4 and inliers >= 11:
                measured = _edge_quad(frame, person, quad, radius=16)
                if len(measured) < 2:
                    measured = None
            if measured and color:
                # A full-edge line can cross behind an occluder. Both incident
                # borders must be visible in this native frame before emission.
                measured = _source_quality_color_corners(frame, person, measured, color)
            if measured:
                ad_seen[ad_id] = list(measured)
                for role, evidence in measured.items():
                    observations.append({"pts": pts, "user_number": ad_id, "role": role,
                                         **evidence, "status": "measured",
                                         "mask_status": mask_status,
                                         **({"measurement_component": "meiwa_color_bar_envelope"}
                                            if ad_id == 5 else {}),
                                         "identity_evidence": {"kind": proposal_kind,
                                                               "inliers": inliers,
                                                               "reference_pts": template["pts"],
                                                               "native_motion_steps": motion_steps,
                                                               "identity_region": template["identity_region"]}})
        qr = {}
        if "ad1_qr_white" in templates and 1 in projected:
            matched = _match_template(templates["ad1_qr_white"], target_xy,
                                      target_desc, min_inliers=11)
            if matched is not None:
                quad, inliers = matched
                if _proposal_near_scene(quad, projected[1], max_centre_delta=460.):
                    found_white = _edge_quad(frame, person, quad, radius=12)
                    for role, evidence in found_white.items():
                        qr[f"qr_white_{role}"] = {**evidence,
                                                "coordinate_status": "internal_print_unknown_world",
                                                "identity_evidence": {"kind": "qr_print_sift_cuda",
                                                                      "inliers": inliers,
                                                                      "reference_pts": 229679360}}
        # A decoded QR code can still give a separate internal finder-pattern
        # landmark when the white print border is not measurable.
        if not qr:
            qr = _qr_white_frame(frame, person, projected.get(1))
        for role, evidence in qr.items():
            internal.append({"pts": pts, "user_number": 1, "role": role,
                             **evidence, "status": "measured",
                             "mask_status": mask_status})
        statuses.append({"pts": pts, "outer_corner_ids": ad_seen,
                         "internal_corner_ids": list(qr), "mask_status": mask_status,
                         "identity_attempts": identity_attempts})
        if index % 50 == 0:
            print(json.dumps({"processed": index+1, "total": len(stamps),
                              "pts": pts, "outer": len(observations),
                              "internal": len(internal)}), flush=True)
    if reviewed is not None:
        def merge_rows(rows: list[dict], reviewed_rows: list[dict]) -> list[dict]:
            by_key = {(int(row["pts"]), int(row["user_number"]), row["role"]): row
                      for row in rows}
            for row in reviewed_rows:
                if int(row["pts"]) in stamps:
                    by_key[(int(row["pts"]), int(row["user_number"]), row["role"])] = row
            return [by_key[key] for key in sorted(by_key)]
        observations = merge_rows(observations, reviewed.get("observations", []))
        internal = merge_rows(internal, reviewed.get("internal_observations", []))
    manifest = {"schema_version": "rinkspace.programme-ad-source-corners.v1",
                "source_sha256": source_hash, "model_sha256": model_hash,
                "feature_source": feature_source,
                "frame_stride": frame_stride,
                "source_pts_interval_inclusive": [start_pts, end_pts],
                "observed_motion_identity_proposals": motion_proposals,
                "excluded_validation_input_sha256": exclusion_hashes,
                "excluded_pts": sorted(excluded),
                "live_reference_sha256": (hashlib.sha256(live_references.read_bytes()).hexdigest()
                                           if live_references is not None else None),
                "observations": observations, "internal_observations": internal,
                "frame_status": statuses,
                "measurement_policy": "scene is broad ROI only; no projected pixel is an observation"}
    (output/"corners.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(stamps), "outer": len(observations),
                      "internal": len(internal)}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot", action="append", type=int, required=True)
    parser.add_argument("--reviewed-seeds", type=Path)
    parser.add_argument("--feature-source", choices=("motion_cache", "native"),
                        default="motion_cache")
    parser.add_argument('--live-references', type=Path)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--start-pts', type=int)
    parser.add_argument('--end-pts', type=int)
    parser.add_argument('--exclude-observations', type=Path, action='append')
    parser.add_argument('--exclude-pts', type=int, action='append')
    parser.add_argument('--motion-proposals', action='store_true')
    args = parser.parse_args()
    run(args.keyframes, args.output, args.shot, reviewed_seeds=args.reviewed_seeds,
        feature_source=args.feature_source, live_references=args.live_references,
        frame_stride=args.frame_stride, exclude_observations=args.exclude_observations,
        exclude_pts=args.exclude_pts, motion_proposals=args.motion_proposals,
        start_pts=args.start_pts, end_pts=args.end_pts)
