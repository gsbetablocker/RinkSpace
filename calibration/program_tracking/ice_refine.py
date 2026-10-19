"""Read real source-pixel rink marks and refine physical keyframe candidates.

Only image extrema become measurements.  A candidate camera proposes a search
window and physical identity; it never supplies an observed pixel.  Ambiguous or
occluded markings are omitted.  Outputs remain outside the project tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import av
import cv2
import numpy as np

try:
    from .projection import ROOT, from_state, project_central_valid, project_fast, to_state
    from .reference_board_pair import BoardIntersectionError
    from .paint_centerline import recenter_painted_band
except ImportError:
    from projection import ROOT, from_state, project_central_valid, project_fast, to_state
    from reference_board_pair import BoardIntersectionError
    from paint_centerline import recenter_painted_band


try:
    from .acceleration import nearest_segments
except ImportError:
    from acceleration import nearest_segments


TEMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
SCENE_PATH = ROOT / "calibration/deliverables/scene.json"
SHOTS_PATH = ROOT / "calibration/program_tracking/source_shots.json"
MASK_DIR = TEMP / "people/masks"
OBS_DIR = TEMP / "ice_observations"
FRAME_SIZE = (1920, 1080)
PTS_PER_FRAME = 256


def source_frame(source: Path, pts: int) -> np.ndarray:
    """Decode the original image at its exact native presentation timestamp."""
    with av.open(str(source)) as container:
        video = container.streams.video[0]
        container.seek(max(0, pts - 2 * 15360), stream=video, backward=True)
        for frame in container.decode(video):
            if frame.pts == pts:
                image = frame.to_ndarray(format="bgr24")
                if image.shape[:2] != (1080, 1920):
                    raise ValueError(f"Source resolution changed: {image.shape}")
                return image
            if frame.pts is not None and frame.pts > pts:
                break
    raise ValueError(f"Exact native PTS {pts} is missing from {source}")


def person_mask(pts: int, first_pts: int) -> np.ndarray:
    """Return the actual SAM2 mask, dilated to cover blur and its near reflection."""
    chunk = (pts - first_pts) // (120 * PTS_PER_FRAME)
    filename = MASK_DIR / f"{chunk:05d}.npz"
    if not filename.exists():
        raise FileNotFoundError(filename)
    with np.load(filename) as data:
        indices = np.flatnonzero(data["pts"] == pts)
        if not len(indices):
            raise KeyError(f"SAM2 mask has no PTS {pts}: {filename}")
        bits = data["masks"][indices[0]]
        low = np.unpackbits(bits, axis=1, count=960).astype(np.uint8)
    full = cv2.resize(low, FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(full, np.ones((25, 25), np.uint8)) > 0


def _signals(image: np.ndarray) -> dict[str, np.ndarray]:
    smooth = cv2.GaussianBlur(image.astype(np.float32), (5, 5), .9)
    b, g, r = cv2.split(smooth)
    return {
        "blue": b - .5 * (r + g),
        "red": r - .5 * (g + b),
        "yellow": .5 * (r + g) - b,
        "gray": cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), .9),
    }


def _graphics_mask() -> np.ndarray:
    mask = np.zeros((1080, 1920), bool)
    mask[:270, :570] = True
    mask[905:, 1660:] = True
    return mask


def detect_contact(image: np.ndarray, signal: dict[str, np.ndarray],
                   exclusion: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Find the continuous yellow board/ice stripe directly in source pixels."""
    blue = (signal["blue"] > 7.5).astype(np.uint8)
    blue[:180] = False
    blue[800:] = False
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((1, 41), np.uint8))
    rail_lines = cv2.HoughLinesP(blue, 1, np.pi / 1800, threshold=100,
                                minLineLength=500, maxLineGap=100)
    rail_y = None
    if rail_lines is not None:
        plausible = [(min(y0, y1), max(y0, y1), abs(x1-x0))
                     for x0, y0, x1, y1 in np.asarray(rail_lines).reshape(-1, 4)
                     if abs(x1-x0) > 500 and abs(y1-y0) < .13*abs(x1-x0)]
        if plausible:
            # The physical rail crosses nearly the full view.  An upper blue
            # arena banner can start earlier in y but spans only part of it.
            rail_y = float(max(plausible, key=lambda x: x[2])[0])
    b, g, r = cv2.split(image)
    yellow = signal["yellow"]
    # The blue rail's white lower edge has a weak positive yellow score of
    # 3--10, while the true kickplate stripe is strongly yellow.  A low
    # threshold mistakenly called the rail bottom the ice contact.
    m = ((yellow > 13.) & (r > 105) & (g > 105) & (b > 65)
         & (np.abs(r.astype(np.int16)-g.astype(np.int16)) < 42)
         & ~exclusion)
    m[:270] = False
    m[940:] = False
    narrow = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((1, 31), np.uint8))
    segments = cv2.HoughLinesP(narrow, 1, np.pi / 1800, threshold=85,
                               minLineLength=170, maxLineGap=65)
    if segments is None:
        return None, np.empty((0, 2))
    candidates = []
    for x0, y0, x1, y1 in np.asarray(segments).reshape(-1, 4):
        dx = x1 - x0
        if abs(dx) < 380 or abs((y1 - y0) / dx) > .18:
            continue
        x = np.linspace(x0, x1, max(5, abs(dx) // 12)).astype(int)
        y = np.rint(np.linspace(y0, y1, len(x))).astype(int)
        strength = np.maximum(yellow[y, x], 0)
        support = (strength > 13.).sum()
        # Very long faint stripes can be real; short bright ad text cannot.
        middle_y = .5 * (y0+y1)
        if rail_y is not None and not (rail_y+55 < middle_y < rail_y+430):
            continue
        if support < 30:
            continue
        # A broad, warm reflection can form a slightly longer Hough segment
        # than the narrow physical kickplate.  The real painted stripe has
        # much stronger yellow chroma along its length.  Compare only long
        # segments, then use their integrated source-image color support.
        color_support = float(np.maximum(yellow[y, x], 0).mean())
        candidates.append((abs(dx), abs(dx) * color_support,
                           (x0, y0, x1, y1)))
    if not candidates:
        return None, np.empty((0, 2))
    # Reject short ad/logo strokes even if bright, then prefer the most
    # strongly supported yellow stripe rather than the longest reflection.
    longest = max(item[0] for item in candidates)
    best = np.asarray(max((item for item in candidates
                           if item[0] >= .6 * longest),
                          key=lambda item: (item[1], item[0]))[2], float)
    bx0, by0, bx1, by1 = best
    axis = np.arange(1920)
    trace = by0 + (axis - bx0) * (by1 - by0) / (bx1 - bx0)
    # Every recorded sample is independently re-read on the actual yellow
    # stripe; the Hough segment supplies only its coarse path.
    observed = []
    for x in range(max(16, int(min(bx0, bx1))), min(1904, int(max(bx0, bx1))), 28):
        ys = np.arange(max(280, int(trace[x]) - 13), min(935, int(trace[x]) + 14))
        if len(ys) < 5:
            continue
        values = yellow[ys, x].copy()
        values[exclusion[ys, x]] = -np.inf
        good = np.flatnonzero(values > 13.)
        if len(good) < 2 or not np.isfinite(values).any():
            continue
        # The yellow stripe can cast a long warm reflection into the ice.
        # Measure the first ice-side half-height crossing of the real stripe,
        # rather than the last weakly yellow reflection pixel.
        peak = int(np.argmax(values))
        if values[peak] < 20:
            continue
        threshold = max(12., .5*float(values[peak]))
        falling = np.flatnonzero((np.arange(len(ys)) > peak) &
                                (values < threshold))
        if not len(falling):
            continue
        y = int(ys[falling[0]])
        observed.append((float(x), float(y)))
    observed = np.asarray(observed, float).reshape(-1, 2)
    if len(observed) < 9 or np.ptp(observed[:, 0]) < 350:
        return None, np.empty((0, 2))
    # A bright yellow reflection may take over the profile search halfway
    # across the board.  A single polynomial fit can absorb that step and
    # retain pixels from both stripes.  A real continuous kickplate edge
    # cannot jump by 14 pixels over one 28-pixel horizontal sample.
    delta = np.diff(observed, axis=0)
    discontinuity = np.flatnonzero(
        np.abs(delta[:, 1]) > 3. + .12 * np.abs(delta[:, 0])
    )
    continuous = [run for run in np.split(observed, discontinuity + 1)
                  if len(run) >= 9 and np.ptp(run[:, 0]) >= 350]
    if not continuous:
        return None, np.empty((0, 2))
    observed = max(continuous, key=lambda run: (np.ptp(run[:, 0]), len(run)))
    coeff = np.polyfit(observed[:, 0], observed[:, 1], 1)
    fit = np.polyval(coeff, observed[:, 0])
    good = np.abs(fit - observed[:, 1]) < 8
    if good.sum() < 9:
        return None, np.empty((0, 2))
    coeff = np.polyfit(observed[good, 0], observed[good, 1], 1)
    return np.polyval(coeff, axis).astype(np.float32), observed[good]


def _ice_roi(contact_y: np.ndarray | None, exclusion: np.ndarray) -> np.ndarray:
    roi = ~exclusion.copy()
    if contact_y is None:
        # Unknown board/ice boundary: do not turn ad or background colors into
        # an ice observation.  Contact can still be measured independently.
        return np.zeros_like(roi)
    roi &= np.arange(1080)[:, None] > contact_y[None, :] + 14
    roi[:280] = False
    return roi


def _image_lines(signal: np.ndarray, image: np.ndarray, roi: np.ndarray,
                 color: str) -> list[dict]:
    """Native-pixel Hough segments with independent local chroma support."""
    b, g, r = cv2.split(image)
    if color == "blue":
        foreground = (signal > 4.5) & (b.astype(np.int16) > r.astype(np.int16) + 2)
    else:
        foreground = (signal > 6.5) & (r.astype(np.int16) > g.astype(np.int16) + 3)
    foreground &= roi
    binary = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    raw = cv2.HoughLinesP(binary, 1, np.pi / 1440, threshold=48,
                          minLineLength=100, maxLineGap=85)
    if raw is None:
        return []
    result = []
    for x0, y0, x1, y1 in np.asarray(raw).reshape(-1, 4):
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 110:
            continue
        n = max(5, int(length / 10))
        xy = np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)]
        xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, 1919)
        yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, 1079)
        valid = roi[yi, xi] & (signal[yi, xi] > (4.5 if color == "blue" else 6.5))
        if valid.sum() < max(5, int(.42 * n)):
            continue
        strength = float(np.median(signal[yi[valid], xi[valid]]))
        result.append({"segment": np.asarray([x0, y0, x1, y1], float),
                       "support": xy[valid], "strength": strength,
                       "length": length, "color": color})
    return sorted(result, key=lambda x: x["length"] * x["strength"], reverse=True)


def _closed_ring(world: np.ndarray) -> np.ndarray:
    """Include the final edge of a physical perimeter without changing scene data."""
    world = np.asarray(world, np.float64)
    if np.allclose(world[0], world[-1], rtol=0., atol=1e-9):
        return world
    return np.vstack((world, world[:1]))


def _world_curves(scene: dict) -> dict[str, np.ndarray]:
    curves = {"ice_board_contact": _closed_ring(scene["ice_boundary"]["points_xyz"])}
    for feature in scene["ice_features"]:
        name = feature["id"]
        if name in ("B1", "B2", "C", "goal_line_left", "goal_line_right"):
            path = np.asarray(feature["segments_xyz"][0], float)
            curves[name] = np.linspace(path[0], path[-1], 220)
        elif feature["kind"] == "circle":
            rings = feature.get("rings") or [feature]
            ring = next((r for r in rings if r.get("role", "outer") == "outer"), None)
            if ring is None:
                continue
            center = np.asarray(ring["center_xyz"], float)
            t = np.linspace(0, 2 * np.pi, 145)
            curves[f"{name}_outer"] = center + float(ring["radius"]) * np.c_[
                np.cos(t), np.sin(t), np.zeros(len(t))]
    return curves


def _segment_distance(observed: np.ndarray, predicted: np.ndarray,
                      min_cosine: float = .985) -> float:
    """Symmetric coarse curve distance, without claiming point correspondence."""
    if len(predicted) < 2:
        return np.inf
    a = observed.reshape(2, 2)
    d = a[1] - a[0]
    norm = np.linalg.norm(d)
    if norm < 1:
        return np.inf
    p = predicted[np.isfinite(predicted).all(axis=1)]
    if len(p) < 2:
        return np.inf
    p = p[(p[:, 0] > -100) & (p[:, 0] < 2020) & (p[:, 1] > 200) & (p[:, 1] < 1180)]
    if len(p) < 2:
        return np.inf
    delta = p-a[0]
    distance = np.abs((d[0]*delta[:, 1]-d[1]*delta[:, 0])/norm)
    # Candidate must overlap the image segment, not merely its infinite line.
    along = (p - a[0]) @ d / norm**2
    near = (along > -.4) & (along < 1.4)
    if near.sum() < 2:
        return np.inf
    local = p[near]
    tangent = local[-1] - local[0]
    cosine = abs(np.dot(d, tangent)) / (norm * max(np.linalg.norm(tangent), 1e-6))
    # A real rink transverse line has the same local perspective direction.
    # The former .68 cutoff admitted board-ad reflections as blue B lines.
    if cosine < min_cosine:
        return np.inf
    return float(np.percentile(distance[near], 35) + 220 * (1 - cosine))


def _extended_line_points(segment: np.ndarray, search_radius: int,
                          spacing: float = 20.) -> np.ndarray:
    """Continue a measured image line through the visible source-frame area.

    These points are search centres, never observations.  The painted-band
    reader below must find two real colour edges at each returned centre.
    """
    endpoints = np.asarray(segment, float).reshape(2, 2)
    midpoint = endpoints.mean(axis=0)
    direction = endpoints[1] - endpoints[0]
    direction /= max(np.linalg.norm(direction), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    # Leave room for the full normal colour profile inside the source image.
    x_margin = search_radius*abs(normal[0])+12.
    y_margin = search_radius*abs(normal[1])+12.
    limits = ((x_margin, 1920.-x_margin),
              (max(280., y_margin), 1080.-y_margin))
    lower, upper = -np.inf, np.inf
    for axis, (start, stop) in enumerate(limits):
        if abs(direction[axis]) < 1e-8:
            if not start <= midpoint[axis] <= stop:
                return np.empty((0, 2))
            continue
        a, b = ((start-midpoint[axis])/direction[axis],
                (stop-midpoint[axis])/direction[axis])
        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
    if upper-lower < 80.:
        return np.empty((0, 2))
    along = np.arange(lower+spacing/2, upper, spacing)
    return midpoint[None, :] + along[:, None]*direction[None, :]


def associate_color_lines(lines: list[dict], camera: np.ndarray,
                          curves: dict[str, np.ndarray], color: str,
                          image: np.ndarray, exclusion: np.ndarray) -> list[dict]:
    names = (["B1", "B2"] if color == "blue"
             else ["C", "goal_line_left", "goal_line_right"])
    projections = {n: project_fast(camera, curves[n])[0] for n in names}
    # Jointly rank every line-to-world-ID proposal.  Hough strength determines
    # whether a real source stripe exists; geometric cost chooses its identity.
    proposals = []
    for index, line in enumerate(lines):
        scores = sorted((_segment_distance(line["segment"], projections[n],
                                           .985 if color == "blue" else .94), n)
                        for n in names)
        best, name = scores[0]
        if (np.isfinite(best) and best <= 130 and
                (len(scores) == 1 or scores[1][0]-best >= 35)):
            proposals.append((best, index, name))
    proposals.sort()
    groups: dict[str, list[tuple[float, dict]]] = {}
    accepted_segments: dict[str, np.ndarray] = {}
    measurements = []
    for best, index, name in proposals:
        line = lines[index]
        segment = line["segment"].reshape(2, 2)
        direction = segment[1]-segment[0]
        if name in groups:
            # Multiple fragments can belong to one curved image stripe.  Do
            # not discard its lower half just because an upper fragment won.
            prior = accepted_segments[name].reshape(2, 2)
            other_direction = prior[1]-prior[0]
            cosine = abs(float(np.dot(direction, other_direction))) / max(
                np.linalg.norm(direction)*np.linalg.norm(other_direction), 1e-9)
            normal = np.array([-other_direction[1], other_direction[0]])
            normal /= max(np.linalg.norm(normal), 1e-9)
            separation = abs(float(np.dot(segment.mean(axis=0)-prior.mean(axis=0), normal)))
            if cosine > .965 and separation < 65:
                groups[name].append((best, line))
            continue
        # Hough frequently returns both edges of the same painted stripe.
        # One physical stripe cannot become two different B-line identities.
        duplicate = False
        for prior_name, prior in accepted_segments.items():
            if prior_name == name:
                continue
            other = prior.reshape(2, 2)
            other_direction = other[1]-other[0]
            cosine = abs(float(np.dot(direction, other_direction))) / max(
                np.linalg.norm(direction)*np.linalg.norm(other_direction), 1e-9)
            normal = np.array([-other_direction[1], other_direction[0]])
            normal /= max(np.linalg.norm(normal), 1e-9)
            separation = abs(float(np.dot(segment.mean(axis=0)-other.mean(axis=0), normal)))
            along = np.dot(segment-other[0], other_direction) / max(
                np.dot(other_direction, other_direction), 1e-9)
            if cosine > .95 and separation < 55 and along.max() > -.25 and along.min() < 1.25:
                duplicate = True
                break
        if duplicate:
            continue
        groups[name] = [(best, line)]
        accepted_segments[name] = line["segment"]

    for name, fragments in groups.items():
        fragments = sorted(fragments, key=lambda item: item[0])[:6]
        # Fit only the geometry of mutually consistent source Hough fragments;
        # their pixels do not become measurements until real colour-band edges
        # are independently read below.
        endpoints = np.concatenate([part["segment"].reshape(2, 2)
                                    for _, part in fragments])
        fit = cv2.fitLine(endpoints.astype(np.float32), cv2.DIST_WELSCH,
                          0, .01, .01).ravel()
        tangent = fit[:2].astype(float)
        anchor = fit[2:].astype(float)
        segment = np.vstack((anchor-200*tangent, anchor+200*tangent))
        # A near-field red centre stripe can be over 150px wide in the source
        # image; a 78px normal window clips one edge around the lower frame.
        radius = 52 if color == "blue" else 150
        points = _extended_line_points(segment, radius)
        points, strengths = recenter_painted_band(
            image, points, tangent, color, exclusion=exclusion,
            search_radius=radius)
        if len(points) < 10:
            continue
        # A distant painted line crossing the extrapolated search path is not
        # evidence for this world ID.  Use the projection only to gate source
        # pixels; it never supplies their measured coordinates.
        distances = np.abs(_curve_error(camera, curves[name], points))
        # The current physical candidate can be off by >150px at the near
        # image edge even when the source red stripe is plainly continuous.
        # The proposal's identity/direction and observed two-edge colour band
        # carry this association; a tight projection gate would silently drop
        # precisely the lower marks needed to correct that camera error.
        keep = np.isfinite(distances) & (distances < (225. if color == "red" else 130.))
        points, strengths = points[keep], strengths[keep]
        if len(points) < 10 or np.ptp(points[:, 1]) < 70.:
            continue
        best = float(min(cost for cost, _ in fragments))
        for p, strength in zip(points, strengths):
            measurements.append({"uv": p, "physical_id": name,
                                 "quality": float(strength), "kind": color,
                                 "assignment_cost_px": best})
    return measurements


def measure_outer_rings(signal: np.ndarray, roi: np.ndarray, contact_y: np.ndarray,
                        camera: np.ndarray, curves: dict[str, np.ndarray]) -> list[dict]:
    """Take outward gray-edge extrema around candidate outer rings only."""
    measurements = []
    for name, world in curves.items():
        if not name.endswith("_outer"):
            continue
        predicted, depth = project_fast(camera, world)
        tangent = np.roll(predicted, -1, axis=0) - np.roll(predicted, 1, axis=0)
        normals = np.c_[-tangent[:, 1], tangent[:, 0]]
        norms = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norms[:, None], 1e-9)
        center = np.nanmean(predicted, axis=0)
        outward = np.sum(normals * (predicted - center), axis=1) < 0
        normals[outward] *= -1
        sampled = []
        for i in range(0, len(world) - 1, 2):
            p = predicted[i]
            if (depth[i] <= 0 or not np.isfinite(p).all() or
                    p[0] < 35 or p[0] > 1884 or p[1] < 320 or p[1] > 1039):
                continue
            # A yellow board edge and the dark ice immediately below it are
            # not an outer ring.  Keep a visible gap from the measured edge.
            if p[1] < contact_y[int(p[0])] + 80:
                continue
            # The board-texture-only initializer can miss a visible outer
            # circle by many pixels.  Read a wide profile, then demand a
            # coherent source-image arc before assigning its physical ID.
            offsets = np.arange(-105, 106, 1, dtype=float)
            profile_xy = p[None] + offsets[:, None] * normals[i]
            xi = np.rint(profile_xy[:, 0]).astype(int)
            yi = np.rint(profile_xy[:, 1]).astype(int)
            inside = (xi >= 0) & (xi < 1920) & (yi >= 0) & (yi < 1080)
            if inside.sum() < 110:
                continue
            valid = np.zeros(len(xi), bool)
            valid[inside] = (roi[yi[inside], xi[inside]] &
                             (yi[inside] > contact_y[xi[inside]]+75))
            if valid.sum() < 90:
                continue
            values = np.full(len(xi), np.nan)
            values[inside] = signal[yi[inside], xi[inside]]
            gradient = np.roll(values, -3) - np.roll(values, 3)
            gradient[~valid] = -np.inf
            gradient[:4] = gradient[-4:] = -np.inf
            k = int(np.nanargmax(gradient))
            if not np.isfinite(gradient[k]) or gradient[k] < 6.0:
                continue
            inner = values[max(k - 10, 0)]
            outer = values[min(k + 10, len(values) - 1)]
            if not np.isfinite(inner + outer) or outer - inner < 6:
                continue
            sampled.append((i, profile_xy[k], float(outer - inner), float(offsets[k])))
        # One or two isolated rink scratches are not a circle observation.
        if len(sampled) < 10:
            continue
        offsets = np.asarray([x[3] for x in sampled])
        if np.median(np.abs(offsets)) > 92:
            continue
        # Retain coherent arcs only.  Isolated texture gradients and the board
        # shadow often satisfy a single profile but not a curved run.
        runs = []
        run = [sampled[0]]
        for item in sampled[1:]:
            if item[0]-run[-1][0] <= 4 and abs(item[3]-run[-1][3]) <= 15:
                run.append(item)
            else:
                if len(run) >= 6:
                    runs.extend(run)
                run = [item]
        if len(run) >= 6:
            runs.extend(run)
        if len(runs) < 10:
            continue
        for i, p, strength, offset in runs:
            measurements.append({"uv": p, "physical_id": name,
                                 "quality": strength / 2, "kind": "gray_outer_edge",
                                 "assignment_cost_px": abs(offset)})
    return measurements


def measure_frame(image: np.ndarray, pts: int, first_pts: int,
                  scene: dict, candidate: np.ndarray) -> tuple[list[dict], dict]:
    exclude = person_mask(pts, first_pts) | _graphics_mask()
    signals = _signals(image)
    contact, contact_points = detect_contact(image, signals, exclude)
    roi = _ice_roi(contact, exclude)
    curves = _world_curves(scene)
    measurements = []
    for p in contact_points:
        measurements.append({"uv": p, "physical_id": "ice_board_contact",
                             "quality": 5., "kind": "yellow_contact",
                             "assignment_cost_px": 0.})
    for color in ("blue", "red"):
        # Blue/red spill from the far board is most common 20--35 px below
        # contact.  A painted ice line needs meaningful ice-side separation.
        margin = 45 if color == "blue" else 35
        line_roi = (roi & (np.arange(1080)[:, None] >
                           contact[None, :]+margin)) if contact is not None else roi
        lines = _image_lines(signals[color], image, line_roi, color)
        profile_exclusion = (exclude | (np.arange(1080)[:, None] <=
                                         contact[None, :]+14)) if contact is not None else exclude
        measurements.extend(associate_color_lines(
            lines, candidate, curves, color, image, profile_exclusion))
    if contact is not None:
        measurements.extend(measure_outer_rings(signals["gray"], roi, contact,
                                                candidate, curves))
    diagnostics = {"contact_detected": contact is not None,
                   "contact_samples": len(contact_points),
                   "ice_pixel_fraction": float(np.mean(roi)),
                   "by_id": {name: sum(x["physical_id"] == name for x in measurements)
                             for name in sorted(set(x["physical_id"] for x in measurements))}}
    return measurements, diagnostics


def save_measurements(output: Path, pts: int, camera_id: str,
                      measurements: list[dict], diagnostics: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, pts=np.asarray(pts, np.int64), camera_id=np.asarray(camera_id),
        observed_uv=np.asarray([m["uv"] for m in measurements], np.float32).reshape(-1, 2),
        physical_id=np.asarray([m["physical_id"] for m in measurements], dtype="<U32"),
        quality=np.asarray([m["quality"] for m in measurements], np.float32),
        kind=np.asarray([m["kind"] for m in measurements], dtype="<U24"),
        assignment_cost_px=np.asarray([m["assignment_cost_px"] for m in measurements], np.float32),
        source=np.asarray("original_decoded_mp4_native_pts"),
        diagnostics=np.asarray(json.dumps(diagnostics)),
    )


def draw_measurements(image: np.ndarray, measurements: list[dict], pts: int,
                      output: Path) -> None:
    result = image.copy()
    colors = {"yellow_contact": (0, 235, 255), "blue": (255, 70, 60),
              "red": (60, 80, 255), "gray_outer_edge": (255, 30, 235)}
    for m in measurements:
        p = tuple(np.rint(m["uv"]).astype(int))
        cv2.circle(result, p, 4, colors[m["kind"]], -1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (1180, 48), (15, 20, 30), -1)
    cv2.putText(result, f"SOURCE PIXEL MEASUREMENTS  PTS {pts}  N={len(measurements)}",
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(output)


def _curve_error(camera: np.ndarray, world: np.ndarray,
                 observed: np.ndarray, with_jacobian: bool = False):
    """Signed source-pixel distance to the nearest visible projected segment.

    The Jacobian differentiates the image segment itself.  Association to its
    nearest segment is piecewise constant, as in standard contour ICP.  No
    camera-projected pixel enters the observed side of this residual.
    """
    if with_jacobian:
        projected, depth, raw_j = project_fast(camera, world, with_jacobian=True)
        jac = raw_j.reshape(len(world), 2, 11)
    else:
        projected, depth = project_fast(camera, world)
    start = projected[:-1]
    finish = projected[1:]
    d = finish - start
    length2 = np.sum(d*d, axis=1)
    valid = (np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[:-1] > 0) & (depth[1:] > 0)
             & (length2 > 1e-4))
    if not valid.any():
        residual = np.full(len(observed), 1e3)
        return (residual, np.zeros((len(observed), 11))) if with_jacobian else residual
    delta = observed[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum("nmi,mi->nm", delta, d) /
                np.maximum(length2[None, :], 1e-9), 0, 1)
    nearest = start[None, :, :] + t[:, :, None]*d[None, :, :]
    distances = np.sum((nearest - observed[:, None, :])**2, axis=2)
    distances[:, ~valid] = np.inf
    index = np.argmin(distances, axis=1)
    row = np.arange(len(observed))
    a = start[index]
    direction = d[index]
    norm = np.sqrt(length2[index])
    u = a-observed
    cross = direction[:, 0]*u[:, 1] - direction[:, 1]*u[:, 0]
    residual = cross/norm
    if not with_jacobian:
        return residual
    j0 = jac[index]
    jd = jac[index+1] - j0
    dc = (jd[:, 0]*u[:, 1, None] - jd[:, 1]*u[:, 0, None]
          + direction[:, 0, None]*j0[:, 1] - direction[:, 1, None]*j0[:, 0])
    dn = np.sum(direction[:, :, None]*jd, axis=1)/norm[:, None]
    derivative = dc/norm[:, None] - cross[:, None]*dn/norm[:, None]**2
    return residual, derivative


def _curve_vector_error(camera: np.ndarray, world: np.ndarray,
                        observed: np.ndarray, with_jacobian: bool = False):
    """Vector from each source pixel to its nearest finite projected segment.

    The nearest-segment association is piecewise constant for differentiation.
    Unlike a signed supporting-line distance, the 2D residual includes endpoint
    distance and has continuous squared norm when neighboring segments meet.
    The Jacobian differentiates the projection parameter on segment interiors;
    on a clamped endpoint its parameter derivative is zero.
    """
    observed = np.asarray(observed, np.float64).reshape(-1, 2)
    if with_jacobian:
        projected, depth, raw_j, central = project_central_valid(
            camera, world, with_jacobian=True)
        jac = raw_j.reshape(len(world), 2, 11)
    else:
        projected, depth, central = project_central_valid(camera, world)
    start, finish = projected[:-1], projected[1:]
    direction = finish - start
    length2 = np.sum(direction*direction, axis=1)
    midpoint_central = project_central_valid(
        camera, .5 * (world[:-1] + world[1:]))[-1]
    valid = (central[:-1] & central[1:] & midpoint_central
             & np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[:-1] > 0) & (depth[1:] > 0)
             & (length2 > 1e-4))
    if not valid.any():
        raise BoardIntersectionError('ice_curve_no_central_segment',
                                     np.arange(len(observed)))

    index, selected_t, clipped_t, nearest = nearest_segments(
        observed, start, direction, length2, valid)
    residual = (nearest - observed).ravel()
    if not with_jacobian:
        return residual

    a = start[index]
    d = direction[index]
    u = observed - a
    length2_selected = length2[index]
    ja = jac[index]
    jd = jac[index + 1] - ja
    # t = (u·d)/(d·d), u = observed-a. Its derivative is active only when
    # the nearest point lies in the open segment interior.
    numerator_j = (-np.einsum('nij,ni->nj', ja, d)
                   + np.einsum('ni,nij->nj', u, jd))
    length2_j = 2.*np.einsum('ni,nij->nj', d, jd)
    t_j = (numerator_j - selected_t[:, None]*length2_j) / length2_selected[:, None]
    t_j[(selected_t <= 0.) | (selected_t >= 1.)] = 0.
    nearest_j = ja + clipped_t[:, None, None]*jd + d[:, :, None]*t_j[:, None, :]
    return residual, nearest_j.reshape(-1, 11)


def finite_curve_distance(camera: np.ndarray, world: np.ndarray,
                          observed: np.ndarray) -> np.ndarray:
    """Euclidean pixel distance to the nearest *finite* projected segment.

    The optimizer's signed ``_curve_error`` is a point-to-supporting-line
    residual: its nearest-segment search is clipped, but the final cross
    product does not penalize travel beyond an endpoint. This independent
    metric measures the actual nearest point after clipping. It retains long
    projected segments: rejecting segments longer than 1414 pixels creates a
    false gap at high zoom. The optimizer remains unchanged by this function.
    """
    observed = np.asarray(observed, np.float64).reshape(-1, 2)
    if len(observed) == 0:
        return np.empty(0, np.float64)
    projected, depth = project_fast(camera, world)
    start, finish = projected[:-1], projected[1:]
    direction = finish-start
    length2 = np.sum(direction*direction, axis=1)
    valid = (np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[:-1] > 0) & (depth[1:] > 0)
             & (length2 > 1e-4))
    if not valid.any():
        return np.full(len(observed), 1e3, np.float64)
    delta = observed[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum('nmi,mi->nm', delta, direction)
                / np.maximum(length2[None, :], 1e-9), 0., 1.)
    nearest = start[None, :, :] + t[:, :, None]*direction[None, :, :]
    squared = np.sum((nearest-observed[:, None, :])**2, axis=2)
    squared[:, ~valid] = np.inf
    return np.sqrt(np.min(squared, axis=1))


def _stratified(measurements: list[dict], limit: int) -> list[dict]:
    if len(measurements) <= limit:
        return measurements
    selected = np.linspace(0, len(measurements)-1, limit).astype(int)
    return [measurements[i] for i in selected]


def _load_board_texture(frame_file: Path, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not frame_file.exists():
        return np.empty((0, 3)), np.empty((0, 2))
    with np.load(frame_file) as data:
        if "xyz" not in data.files or "uv" not in data.files:
            return np.empty((0, 3)), np.empty((0, 2))
        xyz = np.asarray(data["xyz"], float)
        uv = np.asarray(data["uv"], float)
        surface = np.asarray(data["surface"]) if "surface" in data.files else None
    predicted, depth = project_fast(camera, xyz)
    good = (np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1)
            & (depth > 0) & (np.linalg.norm(predicted-uv, axis=1) < 12))
    if surface is not None:
        good &= surface == "board_surface"
    return xyz[good], uv[good]


def refine_camera_gtsam(initial: np.ndarray, measurements: list[dict],
                        scene: dict, texture_xyz: np.ndarray,
                        texture_uv: np.ndarray,
                        ring_sigma: float = 9., contact_sigma: float = 4.,
                        texture_sigma: float = 3.) -> tuple[np.ndarray, dict]:
    """GTSAM camera resection from image contours plus existing 3D texture.

    All outputs are candidates pending a full source-frame visual check.  This
    local LM initialization feeds the separate real iSAM2 temporal backend.
    """
    import gtsam

    curves = _world_curves(scene)
    by_id = {}
    for item in measurements:
        if item["physical_id"] in curves:
            by_id.setdefault(item["physical_id"], []).append(item)
    non_contact = sum(len(items) for name, items in by_id.items()
                      if name != "ice_board_contact")
    if non_contact < 12 or len(by_id) < 2:
        return initial.copy(), {"status": "insufficient_ice_marks",
                                "non_contact_pixels": non_contact,
                                "physical_ids": sorted(by_id)}
    key = gtsam.symbol("x", 0)
    graph = gtsam.NonlinearFactorGraph()
    full_initial_state = to_state(initial)
    # The 3D installation centre is removed from the per-frame variable
    # altogether.  It can only change in a future shared multi-frame solve.
    prior_state = full_initial_state[3:]

    def camera_from_local(local_state):
        state = full_initial_state.copy()
        state[3:] = local_state
        return from_state(state, initial)

    prior_sigma = np.array([.32, .32, .32, .20, .20, .13, .13, .035])
    graph.add(gtsam.PriorFactorVector(
        key, prior_state, gtsam.noiseModel.Diagonal.Sigmas(prior_sigma)))

    def aspect_error(_factor, values, jacobians):
        q = values.atVector(key)
        if jacobians is not None:
            derivative = np.zeros((1, 8))
            derivative[0, 3], derivative[0, 4] = 1., -1.
            jacobians[0] = np.asfortranarray(derivative)
        return np.array([q[3]-q[4]])

    graph.add(gtsam.CustomFactor(
        gtsam.noiseModel.Isotropic.Sigma(1, .025), [key], aspect_error))
    total_curve = 0
    for name, items in by_id.items():
        items = _stratified(items, 44 if name == "ice_board_contact" else 35)
        world = curves[name]
        sigma = (contact_sigma if name == "ice_board_contact" else
                 7. if name in ("B1", "B2", "C", "goal_line_left", "goal_line_right")
                 else ring_sigma)
        for begin in range(0, len(items), 7):
            pixels = np.asarray([m["uv"] for m in items[begin:begin+7]], float)
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(1.5),
                gtsam.noiseModel.Isotropic.Sigma(len(pixels), sigma))

            def curve_factor(_factor, values, jacobians,
                             world=world, pixels=pixels):
                camera = camera_from_local(values.atVector(key))
                if jacobians is None:
                    return _curve_error(camera, world, pixels)
                residual, derivative = _curve_error(camera, world, pixels, True)
                jacobians[0] = np.asfortranarray(derivative[:, 3:])
                return residual

            graph.add(gtsam.CustomFactor(noise, [key], curve_factor))
            total_curve += len(pixels)
    # Existing board texture is an independent image observation tied to an
    # atlas world point.  It stabilizes the camera while ice curves correct it.
    texture_xyz = np.asarray(texture_xyz, float)
    texture_uv = np.asarray(texture_uv, float)
    for begin in range(0, len(texture_xyz), 6):
        world = texture_xyz[begin:begin+6]
        pixels = texture_uv[begin:begin+6]
        if not len(world):
            continue
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.5),
            gtsam.noiseModel.Isotropic.Sigma(2*len(world), texture_sigma))

        def texture_factor(_factor, values, jacobians,
                           world=world, pixels=pixels):
            camera = camera_from_local(values.atVector(key))
            projected, depth, derivative = project_fast(camera, world, True)
            residual = (projected-pixels).ravel()
            if jacobians is not None:
                jacobians[0] = np.asfortranarray(derivative[:, 3:])
            return residual

        graph.add(gtsam.CustomFactor(noise, [key], texture_factor))

    values = gtsam.Values()
    values.insert(key, prior_state)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(35)
    params.setRelativeErrorTol(1e-6)
    try:
        solution = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
        refined = camera_from_local(solution.atVector(key))
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return initial.copy(), {"status": "optimizer_failed", "error": str(error)[:240]}

    before = []
    after = []
    for name, items in by_id.items():
        pixels = np.asarray([m["uv"] for m in items], float)
        before.extend(np.abs(_curve_error(initial, curves[name], pixels)))
        after.extend(np.abs(_curve_error(refined, curves[name], pixels)))
    old_texture = (np.linalg.norm(project_fast(initial, texture_xyz)[0]-texture_uv, axis=1)
                   if len(texture_xyz) else np.empty(0))
    new_texture = (np.linalg.norm(project_fast(refined, texture_xyz)[0]-texture_uv, axis=1)
                   if len(texture_xyz) else np.empty(0))
    metrics = {"status": "ice_refined_candidate_unvalidated",
               "source_curve_pixels": total_curve,
               "texture_pixels": len(texture_xyz),
               "physical_ids": sorted(by_id),
               "curve_median_before_px": float(np.median(before)),
               "curve_median_after_px": float(np.median(after)),
               "texture_median_before_px": (float(np.median(old_texture))
                                             if len(old_texture) else None),
               "texture_median_after_px": (float(np.median(new_texture))
                                            if len(new_texture) else None),
               "camera_centre_shift_scene_units": float(np.linalg.norm(refined[:3]-initial[:3]))}
    if (not np.isfinite(refined).all() or refined[2] < 1 or
            abs(refined[6]-refined[7]) > .12 or
            metrics["curve_median_after_px"] > metrics["curve_median_before_px"]+.5 or
            (len(new_texture) >= 5 and np.median(new_texture) >
             max(12., 1.8*np.median(old_texture)))):
        metrics["status"] = "refinement_rejected_keep_initial"
        return initial.copy(), metrics
    return refined, metrics


def draw_refined(image: np.ndarray, measurements: list[dict],
                 camera: np.ndarray, scene: dict, pts: int, output: Path) -> None:
    result = image.copy()
    curves = _world_curves(scene)
    names = {m["physical_id"] for m in measurements}
    for name in names:
        if name not in curves:
            continue
        projected, depth = project_fast(camera, curves[name])
        valid = ((depth > 0) & np.isfinite(projected).all(axis=1)
                 & (projected[:, 0] >= 0) & (projected[:, 0] < 1920)
                 & (projected[:, 1] >= 0) & (projected[:, 1] < 1080))
        for j in range(len(projected)-1):
            if valid[j] and valid[j+1] and np.linalg.norm(projected[j+1]-projected[j]) < 80:
                cv2.line(result, tuple(np.rint(projected[j]).astype(int)),
                         tuple(np.rint(projected[j+1]).astype(int)),
                         (10, 225, 20), 2, cv2.LINE_AA)
    for m in measurements:
        cv2.circle(result, tuple(np.rint(m["uv"]).astype(int)), 3,
                   (225, 15, 235), -1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (1050, 48), (15, 20, 30), -1)
    cv2.putText(result, f"GTSAM CANDIDATE / REAL PIXELS  PTS {pts}", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(output)


def _keyframe_paths(keyframes: Path) -> tuple[Path, Path]:
    keyframes = Path(keyframes)
    return ((keyframes / "cameras.json", keyframes) if keyframes.is_dir()
            else (keyframes, keyframes.parent))


def run_measure(pts_values: list[int], keyframes: Path,
                output_dir: Path = OBS_DIR,
                review_dir: Path | None = None,
                refined_dir: Path | None = None) -> list[dict]:
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    shots = json.loads(SHOTS_PATH.read_text(encoding="utf-8"))
    camera_file, frame_dir = _keyframe_paths(keyframes)
    rows = {int(x["pts"]): x for x in json.loads(
        camera_file.read_text(encoding="utf-8"))["frames"]}
    first_pts = int(shots["window_pts"][0])
    source = Path(shots["source_path"])
    manifest = []
    for pts in pts_values:
        row = rows.get(pts)
        if row is None or "camera_vector25" not in row:
            manifest.append({"pts": pts, "status": "no_physical_camera_candidate"})
            continue
        image = source_frame(source, pts)
        camera = np.asarray(row["camera_vector25"], float)
        measured, diagnostics = measure_frame(image, pts, first_pts, scene, camera)
        save_measurements(output_dir / f"{pts}.npz", pts, row["camera_id"],
                          measured, diagnostics)
        if review_dir is not None:
            draw_measurements(image, measured, pts, review_dir / f"{pts}.jpg")
        report = {"pts": pts, "camera_id": row["camera_id"],
                  "status": "source_pixels_measured_unvalidated",
                  "keyframe_source": str(camera_file), **diagnostics}
        if refined_dir is not None:
            texture_xyz, texture_uv = _load_board_texture(frame_dir / f"{pts}.npz", camera)
            refined, result = refine_camera_gtsam(camera, measured, scene,
                                                  texture_xyz, texture_uv)
            refined_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(refined_dir / f"{pts}.npz", pts=np.asarray(pts, np.int64),
                                camera25=refined, initial_camera25=camera,
                                metrics=np.asarray(json.dumps(result)),
                                source_observations=np.asarray(str(output_dir/f"{pts}.npz")),
                                keyframes_source=np.asarray(str(camera_file)))
            if review_dir is not None:
                draw_refined(image, measured, refined, scene, pts,
                             review_dir / f"{pts}_refined.jpg")
            report["refinement"] = result
        manifest.append(report)
        print(json.dumps(manifest[-1]), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pts", type=int, nargs="+", required=True)
    parser.add_argument("--keyframes", type=Path, required=True,
                        help="Directory containing cameras.json and PTS.npz, or that JSON file")
    parser.add_argument("--output", type=Path, default=OBS_DIR)
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--refined-output", type=Path, default=None,
                        help="Run GTSAM contour + texture candidate refinement into this directory")
    args = parser.parse_args()
    run_measure(args.pts, args.keyframes, args.output, args.review,
                args.refined_output)


if __name__ == "__main__":
    main()
