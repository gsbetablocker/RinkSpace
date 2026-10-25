"""Existing source-only near-board material readers, restored from project history.

Blue-cap algorithm: main_nearboard_blue_ice_source.py (2026-09-28).
Grey/ice interval: secondary_nearboard_top_source._transition_interval.
These readers identify material evidence, never alter native board geometry.
"""
import hashlib
import cv2
import numpy as np

IMAGE_SIZE = (1080, 1920)

def _blue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, _, r = cv2.split(image.astype(np.int16))
    blue = ((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 128)
            & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 15)
            & (b-r >= 13))
    return cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((3, 3), np.uint8))


def _outer_blue_components(mask: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        # Conservative source-image context: the near-board cap is foreground
        # entering the bottom viewport.  A blue advertisement can touch a
        # side image border and lie next to white panel/ice-colored pixels;
        # such a component must stay unknown, never be called near board.
        # A real near cap clipped solely by a side border also stays unknown.
        if (area < 1300 or min(width, height) < 42 or y+height < 560
                or y+height < IMAGE_SIZE[0]-2):
            continue
        component = (labels[y:y+height, x:x+width] == label).astype(np.uint8)
        result.append((component, dict(x=x, y=y, width=width, height=height,
                                       area=area, touches_left=x <= 2,
                                       touches_right=x+width >= IMAGE_SIZE[1]-2,
                                       touches_bottom=y+height >= IMAGE_SIZE[0]-2)))
    return sorted(result, key=lambda item: -item[1]["area"])


def _edge_points(image: np.ndarray, hsv: np.ndarray, component: np.ndarray,
                 box: dict, exclude: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    """Return measured q50 pixels, blue-to-ice normal, q10/q90, and contrast."""
    x0, y0, width, height = (box[key] for key in ("x", "y", "width", "height"))
    # Need a pad for both plateaux and normal estimation.  Do not invent a
    # contour past the image; border pixels therefore remain unmeasured.
    pad = 26
    xa, ya = max(0, x0-pad), max(0, y0-pad)
    xb, yb = min(IMAGE_SIZE[1], x0+width+pad), min(IMAGE_SIZE[0], y0+height+pad)
    local = np.zeros((yb-ya, xb-xa), np.uint8)
    local[y0-ya:y0-ya+height, x0-xa:x0-xa+width] = component
    smoothed = cv2.GaussianBlur(local.astype(np.float32), (9, 9), 1.6)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (np.empty((0, 2)), np.empty((0, 2)),
                np.empty((0, 2)), np.empty(0))
    boundary = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    if len(boundary) > 5000:
        boundary = boundary[::max(1, len(boundary)//5000)]
    xx, yy = boundary[:, 0], boundary[:, 1]
    norm = np.hypot(gx[yy, xx], gy[yy, xx])
    good = norm > .15
    xy = boundary[good].astype(np.float32) + (xa, ya)
    normals = -np.column_stack((gx[yy[good], xx[good]],
                                 gy[yy[good], xx[good]])) / norm[good, None]
    # Arc-length thinning bounds processing time while retaining arbitrary
    # orientation, including near-vertical and curved fragments.
    xy, normals = xy[::3], normals[::3]
    if not len(xy):
        return (np.empty((0, 2)), np.empty((0, 2)),
                np.empty((0, 2)), np.empty(0))
    steps = np.arange(-20., 21., dtype=np.float32)
    sx = xy[:, None, 0] + normals[:, None, 0]*steps
    sy = xy[:, None, 1] + normals[:, None, 1]*steps
    in_image = ((sx[:, 0] >= 1) & (sx[:, -1] < IMAGE_SIZE[1]-1)
                & (sy[:, 0] >= 1) & (sy[:, -1] < IMAGE_SIZE[0]-1))
    xy, normals, sx, sy = xy[in_image], normals[in_image], sx[in_image], sy[in_image]
    if not len(xy):
        return (np.empty((0, 2)), np.empty((0, 2)),
                np.empty((0, 2)), np.empty(0))
    maps = (sx.astype(np.float32), sy.astype(np.float32))
    colors = cv2.remap(image, *maps, cv2.INTER_LINEAR).astype(np.float32)
    hsv_colors = cv2.remap(hsv, *maps, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(colors.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blue_luma = np.median(gray[:, :7], axis=1)
    ice_luma = np.median(gray[:, -7:], axis=1)
    contrast = ice_luma-blue_luma
    blue_h = np.median(hsv_colors[:, :7, 0], axis=1)
    blue_s = np.median(hsv_colors[:, :7, 1], axis=1)
    ice_s = np.median(hsv_colors[:, -7:, 1], axis=1)
    # The blue/ice order is determined by the component's outward normal;
    # no screen-up, screen-right, slope, or candidate camera is assumed.
    reliable = ((contrast >= 34) & (ice_luma >= 145) & (ice_s <= 62)
                & (blue_h >= 94) & (blue_h <= 132) & (blue_s >= 40))
    reliable &= ~exclude[np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)]
    if not np.any(reliable):
        return (np.empty((0, 2)), np.empty((0, 2)),
                np.empty((0, 2)), np.empty(0))
    xy, normals, gray, contrast = (array[reliable] for array in
                                   (xy, normals, gray, contrast))
    low = blue_luma[reliable]
    amp = contrast
    # Monotone rising flank about the connected blue component.  Broken or
    # multiple crossings are unknown, not smoothed into a fabricated edge.
    offsets = np.empty((len(xy), 3), np.float32)
    valid = np.ones(len(xy), bool)
    for j, fraction in enumerate((.1, .5, .9)):
        level = low + fraction*amp
        crossings = (gray[:, :-1] < level[:, None]) & (gray[:, 1:] >= level[:, None])
        closeness = np.abs(steps[:-1][None, :])
        score = np.where(crossings, closeness, np.inf)
        index = np.argmin(score, axis=1)
        valid &= np.isfinite(score[np.arange(len(xy)), index])
        rise = gray[np.arange(len(xy)), index+1]-gray[np.arange(len(xy)), index]
        offsets[:, j] = steps[index] + np.divide(level-gray[np.arange(len(xy)), index],
                                                  rise, out=np.zeros_like(level),
                                                  where=np.abs(rise) > 1e-4)
    width = offsets[:, 2]-offsets[:, 0]
    valid &= ((offsets[:, 0] < offsets[:, 1]) & (offsets[:, 1] < offsets[:, 2])
              & (width >= .8) & (width <= 24) & (np.abs(offsets[:, 1]) <= 10))
    xy, normals, offsets, contrast = (array[valid] for array in
                                      (xy, normals, offsets, contrast))
    observed = xy + offsets[:, 1, None]*normals
    interval = offsets[:, (0, 2)] - offsets[:, 1, None]
    return observed, normals, interval, contrast


def _dominant_fragment(uv: np.ndarray, normals: np.ndarray,
                       intervals: np.ndarray, contrast: np.ndarray):
    if len(uv) < 12:
        return None
    # RANSAC is used only for candidate identity/fragment continuity.  All
    # returned UV remain measured q50 crossings on the original image.
    rng = np.random.default_rng(20260928)
    best = np.empty(0, np.int64)
    for _ in range(180):
        i, j = rng.choice(len(uv), 2, replace=False)
        delta = uv[j]-uv[i]
        span = float(np.linalg.norm(delta))
        if span < 80:
            continue
        tangent = delta/span
        normal = np.array((-tangent[1], tangent[0]))
        dist = np.abs((uv-uv[i]) @ normal)
        aligned = np.abs(normals @ normal) > .65
        chosen = np.flatnonzero((dist <= 8) & aligned)
        if len(chosen) > len(best):
            best = chosen
    if len(best) < 12:
        return None
    subset = uv[best]
    center = subset.mean(axis=0)
    _, _, vt = np.linalg.svd(subset-center, full_matrices=False)
    tangent = vt[0]
    projection = (subset-center) @ tangent
    if np.ptp(projection) < 85:
        return None
    order = np.argsort(projection)
    best = best[order]
    # Preserve a spread of actual crossings rather than counting thousands
    # of correlated pixels as independent evidence.
    projection = projection[order]
    chosen = []
    last = -np.inf
    for index, location in zip(best, projection):
        if location-last >= 10:
            chosen.append(int(index))
            last = location
    if len(chosen) < 8:
        return None
    chosen = np.asarray(chosen, int)
    uv, normals, intervals, contrast = (array[chosen] for array in
                                        (uv, normals, intervals, contrast))
    return uv, normals, intervals, contrast


def _connected_ice(image: np.ndarray, hsv: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Find the pale ice component connected to the broad image bottom.

    The yellow kickplate and colored advertisements break its connection to
    similarly pale *board* panels. This is a source-pixel context test only;
    missing or occluded ice yields no qualified candidate.
    """
    pale = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 130)
            & (image.min(axis=2) > 120)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale, 8)
    possible = [index for index in range(1, count)
                if stats[index, cv2.CC_STAT_AREA] >= 100_000
                and stats[index, cv2.CC_STAT_TOP]
                + stats[index, cv2.CC_STAT_HEIGHT] >= IMAGE_SIZE[0]-5
                and stats[index, cv2.CC_STAT_WIDTH] >= 400]
    if not possible:
        return labels, -1, 0
    chosen = max(possible, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    return labels, int(chosen), int(stats[chosen, cv2.CC_STAT_AREA])


def _ice_side_fraction(uv: np.ndarray, normals: np.ndarray,
                       labels: np.ndarray, ice_label: int) -> float:
    if ice_label < 0:
        return 0.
    hits = np.zeros(len(uv), bool)
    for distance in (20., 30.):
        sample = np.rint(uv + distance*normals).astype(int)
        inside = ((sample[:, 0] >= 0) & (sample[:, 0] < IMAGE_SIZE[1])
                  & (sample[:, 1] >= 0) & (sample[:, 1] < IMAGE_SIZE[0]))
        hits[inside] |= (labels[sample[inside, 1], sample[inside, 0]] == ice_label)
    return float(np.mean(hits))


def extract_candidate(image: np.ndarray, pts: int,
                      person_mask: np.ndarray | None = None) -> list[dict]:
    """Find broad blue-cap/ice candidate fragments from raw native pixels."""
    if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
        raise ValueError("Expected native 1080x1920 BGR uint8 source frame")
    exclude = np.zeros(IMAGE_SIZE, bool) if person_mask is None else np.asarray(person_mask, bool)
    if exclude.shape != IMAGE_SIZE:
        raise ValueError("Person mask has wrong native shape")
    blue = _blue_mask(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ice_labels, ice_label, ice_area = _connected_ice(image, hsv)
    output = []
    for component, box in _outer_blue_components(blue):
        uv, normals, interval, contrast = _edge_points(image, hsv, component, box, exclude)
        fragment = _dominant_fragment(uv, normals, interval, contrast)
        if fragment is None:
            continue
        uv, normals, interval, contrast = fragment
        ice_fraction = _ice_side_fraction(uv, normals, ice_labels, ice_label)
        if ice_fraction < .75:
            continue
        output.append(dict(pts=int(pts), observed_uv=uv.astype(np.float32),
                           outward_normal=normals.astype(np.float32),
                           normal_interval_px=interval.astype(np.float32),
                           color_contrast=contrast.astype(np.float32),
                           component=box,
                           source_context=dict(ice_connected_fraction=ice_fraction,
                                               connected_ice_area_px=ice_area,
                                               blue_component_area_px=box["area"]),
                           status="candidate_bluecap_ice_edge_unassigned",
                           source_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    return output



def _transition_interval(gray: np.ndarray, y: int, seed_x: int
                         ) -> tuple[list[int] | None, str, float | None]:
    """Bracket the raw dark-to-ice rise; endpoints are outward whole pixels.

    The seven-row average incorporates motion blur and local line slope.  A
    q10/q90 luminance rise is visual support, not a statistical confidence
    interval or a subpixel claim about the physical board edge.
    """
    if not (3 <= y < 1077 and 33 <= seed_x < 468):
        return None, "image_border_no_two_plateaux", None
    line = gray[y - 3:y + 4, :500].astype(np.float64).mean(axis=0)
    dark = float(np.median(line[seed_x - 30:seed_x - 16]))
    ice = float(np.median(line[seed_x + 16:seed_x + 31]))
    amplitude = ice - dark
    if amplitude < 18.:
        return None, "weak_grey_ice_contrast", amplitude
    xx = np.arange(seed_x - 15, seed_x + 16)
    values = line[xx]
    crossings = []
    for fraction in (.1, .5, .9):
        level = dark + fraction * amplitude
        candidates = np.flatnonzero((values[:-1] < level) &
                                    (values[1:] >= level))
        if not len(candidates):
            return None, "no_local_rising_transition", amplitude
        index = int(candidates[np.argmin(np.abs(xx[candidates] - seed_x))])
        crossing = (float(xx[index]) +
                    (level - values[index]) / (values[index + 1] - values[index]))
        crossings.append(crossing)
    q10, q50, q90 = crossings
    if not (q10 < q50 < q90 and 1. <= q90 - q10 <= 30.
            and abs(q50 - seed_x) <= 8.):
        return None, "nonunique_or_broad_transition", amplitude
    return [int(np.floor(q10)), int(np.ceil(q90))], "source_interval", amplitude
