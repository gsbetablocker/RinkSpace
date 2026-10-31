"""Locate searches on a plane from independently identified straight source marks.

Line correspondences need no invented point correspondence along a stripe.
The resulting homography is navigation only: it cannot output observations,
change a native camera or establish the identity of a newly encountered line.
"""
from collections.abc import Mapping
import numpy as np


def _line(points, label):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Finite 2D source/world line support required: ' + label)
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
    if singular[0] <= np.finfo(float).eps*max(1., np.linalg.norm(center)):
        raise ValueError('Coincident points cannot locate a line: ' + label)
    normal = axes[-1]
    return np.r_[normal, -normal@center]


def fit_planar_line_navigation(world_paths: Mapping, source_lines: Mapping,
                               *, image_size=(1920, 1080)):
    """Fit a source search map using complete named finite straight marks.

    Caller supplies independently established material identities. All supplied
    lines participate, with no residual-based removal. A deficient line layout
    is rejected rather than resolved by the current native projection.
    """
    width, height = map(float, image_size)
    if min(width, height) <= 0 or not np.isfinite([width, height]).all():
        raise ValueError('Positive finite source image dimensions required')
    names = tuple(sorted(source_lines))
    if len(names) < 4 or any(name not in world_paths for name in names):
        raise ValueError('At least four named world/source line correspondences required')
    scale = max(width, height)/2.
    normalizer = np.array([[1/scale, 0., -(width-1)/(2*scale)],
                           [0., 1/scale, -(height-1)/(2*scale)], [0., 0., 1.]])
    equations = []; world_lines = {}; image_lines = {}; observed = {}
    for name in names:
        world = np.asarray(world_paths[name], float)
        if world.ndim != 2 or world.shape[1] != 2 or not np.isfinite(world).all():
            raise ValueError('World paths must be finite ground-plane XY: ' + name)
        world_line = _line(world, name)
        # A world arc must not silently become a navigation line. This is a
        # floating arithmetic test on the native primitive, not a pixel gate.
        slack = np.finfo(float).eps*128*max(1., float(np.linalg.norm(world, axis=1).max()))
        if np.max(np.abs(world@world_line[:2]+world_line[2])) > slack:
            raise ValueError('Native path is not straight: ' + name)
        uv = np.asarray(source_lines[name], float).copy()
        normalized = uv/scale+normalizer[:2, 2]
        image_line = _line(normalized, name)
        u, v, w = image_line
        cross = np.array([[0., -w, v], [w, 0., -u], [-v, u, 0.]])
        block = np.zeros((3, 9))
        for j in range(3):
            block[j, j*3:j*3+3] = world_line
        equations.extend(cross@block)
        world_lines[name], image_lines[name], observed[name] = world_line, image_line, uv
    matrix = np.asarray(equations)
    _, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps*max(matrix.shape)*singular[0]
    if np.count_nonzero(singular > tolerance) < 8:
        raise ValueError('Line layout does not determine a planar search map')
    dual = axes[-1].reshape(3, 3)
    if abs(np.linalg.det(dual)) <= np.finfo(float).eps*max(1., np.linalg.norm(dual)**3):
        raise ValueError('Singular source line navigation')
    homography = np.linalg.inv(normalizer)@np.linalg.inv(dual).T
    homography /= np.linalg.norm(homography)
    if homography[2, 2] < 0:
        homography *= -1
    errors = {}
    for name in names:
        image_line = np.linalg.solve(homography.T, world_lines[name])
        image_line /= np.linalg.norm(image_line[:2])
        errors[name] = np.abs(observed[name]@image_line[:2]+image_line[2])
    return dict(homography=homography, source_line_ids=names,
        source_uv={key: value.copy() for key, value in observed.items()},
        source_line_normal_distances_px=errors, singular_values=singular,
        fit_eligible=False, navigation_only=True, source_UV_changed=False,
        native_camera_changed=False, correspondence_along_lines_invented=False)


def project_search_paths(homography, world_paths):
    """Return named search paths; no returned coordinate is a source reading."""
    h = np.asarray(homography, float)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError('Finite 3x3 source search homography required')
    result = {}
    for name, values in world_paths.items():
        xy = np.asarray(values, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Finite plane XY path required')
        projected = np.c_[xy, np.ones(len(xy))]@h.T
        valid = projected[:, 2] > np.finfo(float).eps*max(1., np.linalg.norm(h))
        uv = np.full((len(xy), 2), np.nan)
        uv[valid] = projected[valid, :2]/projected[valid, 2, None]
        result[name] = dict(uv=uv, valid=valid, navigation_only=True)
    return result
