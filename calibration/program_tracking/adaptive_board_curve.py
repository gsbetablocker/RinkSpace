"""Camera-dependent adaptive subdivision of paired rink board contours.

Both top and ice-contact rings retain every original world endpoint, the same
insertion fractions, and a closed loop. Subdivision uses actual Brown25
projection and a finite projected-chord error test at 1/4, 1/2 and 3/4 of
each leaf. It changes only the representation of a fixed 3D segment, never
its shape, source observations or camera. The returned error is sampled, not
a formal supremum over the continuous Brown curve; unresolved leaves are
explicit instead of being silently treated as accurate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import numpy as np

WORK = Path(__file__).resolve().parents[1] / "work"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
from physical_projection import safe_project  # noqa: E402


@dataclass(frozen=True)
class RefinedBoardPaths:
    top_xyz: np.ndarray
    contact_xyz: np.ndarray
    top_uv: np.ndarray
    contact_uv: np.ndarray
    top_valid: np.ndarray
    contact_valid: np.ndarray
    segment_offsets: np.ndarray
    original_segment: np.ndarray
    original_fraction: np.ndarray
    segment_report: tuple[dict, ...]
    tolerance_px: float


@dataclass(frozen=True)
class RefinedVerticalEdge:
    fractions: np.ndarray
    uv: np.ndarray
    valid: np.ndarray
    unresolved_leaves: int
    skipped_offscreen_leaves: int


def _closed_ring(vertices, name: str) -> np.ndarray:
    points = np.asarray(vertices, np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError(f"{name} must be a finite Nx3 closed-ring vertex sequence")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("Board ring has too few distinct vertices")
    return points


def _finite_chord_distance(a: np.ndarray, b: np.ndarray,
                           sample: np.ndarray) -> float:
    direction = b-a
    length2 = float(direction @ direction)
    if length2 <= 1e-20:
        return float(np.linalg.norm(sample-a))
    t = float(np.clip((sample-a) @ direction/length2, 0., 1.))
    return float(np.linalg.norm(sample-(a+t*direction)))


def _in_sensor(uv: np.ndarray, image_size_wh: tuple[int, int]) -> bool:
    width, height = image_size_wh
    return bool(np.isfinite(uv).all() and 0 <= uv[0] <= width-1
                and 0 <= uv[1] <= height-1)


def _chord_intersects_sensor(a: np.ndarray, b: np.ndarray,
                             image_size_wh: tuple[int, int]) -> bool:
    width, height = image_size_wh
    delta = b-a
    lower, upper = 0., 1.
    for p, q in ((-delta[0], a[0]), (delta[0], width-1-a[0]),
                 (-delta[1], a[1]), (delta[1], height-1-a[1])):
        if abs(p) <= 1e-14:
            if q < 0:
                return False
        elif p < 0:
            lower = max(lower, q/p)
        else:
            upper = min(upper, q/p)
    return lower <= upper


def refine_board_paths(camera25, top_vertices, contact_vertices, *,
                       tolerance_px: float = .125, max_depth: int = 16,
                       max_nodes_per_segment: int = 1024,
                       image_size_wh: tuple[int, int] = (1920, 1080),
                       segment_indices: set[int] | None = None) -> RefinedBoardPaths:
    """Refine paired world segments until sampled visible chord error is small.

    ``segment_indices`` can isolate a diagnostic; unselected original segments
    remain one segment. Any selected leaf that reaches ``max_depth`` without
    meeting the requested tolerance is reported as unresolved. Invalid optical
    transitions are subdivided rather than bridged. The caller must separately
    validate dense samples when it needs a stronger empirical error claim.
    """
    camera = np.asarray(camera25, np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("Camera must be one finite Brown25 vector")
    top = _closed_ring(top_vertices, "top")
    contact = _closed_ring(contact_vertices, "contact")
    if (top.shape != contact.shape
            or not np.allclose(top[:, :2], contact[:, :2], rtol=0., atol=1e-10)):
        raise ValueError("Top/contact rings must share original XY vertices")
    if (not np.isfinite(tolerance_px) or tolerance_px <= 0 or max_depth < 1
            or max_nodes_per_segment < 5):
        raise ValueError("Refinement needs positive tolerance and depth")
    if len(image_size_wh) != 2 or min(image_size_wh) <= 0:
        raise ValueError("Invalid original-sensor size")
    selected = set(range(len(top))) if segment_indices is None else set(segment_indices)
    if selected - set(range(len(top))):
        raise ValueError("Unknown original board segment index")

    endpoint_world = np.stack((top, contact), axis=1)
    endpoint_uv, _, endpoint_valid = safe_project(camera, endpoint_world.reshape(-1, 3))
    endpoint_uv = endpoint_uv.reshape(len(top), 2, 2)
    endpoint_valid = endpoint_valid.reshape(len(top), 2)
    initial_fractions = np.asarray([.25, .5, .75])
    selected_order = sorted(selected)
    if selected_order:
        sampled_top = (top[selected_order, None, :]
                       + initial_fractions[None, :, None]
                       * (np.roll(top, -1, axis=0)[selected_order]-top[selected_order])[:, None, :])
        sampled_contact = (contact[selected_order, None, :]
                           + initial_fractions[None, :, None]
                           * (np.roll(contact, -1, axis=0)[selected_order]
                              -contact[selected_order])[:, None, :])
        sampled_world = np.stack((sampled_top, sampled_contact), axis=2)
        sampled_uv, _, sampled_valid = safe_project(camera, sampled_world.reshape(-1, 3))
        sampled_uv = sampled_uv.reshape(len(selected_order), 3, 2, 2)
        sampled_valid = sampled_valid.reshape(len(selected_order), 3, 2)
        selected_position = {segment: row for row, segment in enumerate(selected_order)}
    else:
        sampled_uv = np.empty((0, 3, 2, 2))
        sampled_valid = np.empty((0, 3, 2), bool)
        selected_position = {}

    world_top = [top[0].copy()]
    world_contact = [contact[0].copy()]
    projected_top = []
    projected_contact = []
    validity_top = []
    validity_contact = []
    segment_origin = [0]
    fractions = [0.]
    offsets = [0]
    reports = []

    for segment in range(len(top)):
        a_top, b_top = top[segment], top[(segment+1) % len(top)]
        a_contact, b_contact = contact[segment], contact[(segment+1) % len(top)]
        next_index = (segment+1) % len(top)
        cache: dict[float, tuple[np.ndarray, np.ndarray]] = {
            0.: (endpoint_uv[segment], endpoint_valid[segment]),
            1.: (endpoint_uv[next_index], endpoint_valid[next_index]),
        }
        if segment in selected_position:
            row = selected_position[segment]
            for j, fraction in enumerate(initial_fractions):
                cache[float(fraction)] = (sampled_uv[row, j], sampled_valid[row, j])
        unresolved = 0
        skipped_offscreen = 0
        leaf_errors = []
        visited_nodes = 0

        def evaluate(fraction: float):
            if fraction not in cache:
                positions = np.stack((a_top+fraction*(b_top-a_top),
                                      a_contact+fraction*(b_contact-a_contact)))
                uv, _, valid = safe_project(camera, positions)
                cache[fraction] = (uv, valid)
            return cache[fraction]

        def recur(left: float, right: float, depth: int) -> list[float]:
            nonlocal unresolved, skipped_offscreen, visited_nodes
            visited_nodes += 1
            test = (left, left+.25*(right-left), .5*(left+right),
                    left+.75*(right-left), right)
            measurements = [evaluate(t) for t in test]
            uv = np.asarray([row[0] for row in measurements])  # [5, top/contact, xy]
            valid = np.asarray([row[1] for row in measurements])
            active = np.asarray([
                (any(_in_sensor(uv[i, k], image_size_wh)
                     for i in range(5) if valid[i, k])
                 or (valid[0, k] and valid[-1, k]
                     and _chord_intersects_sensor(uv[0, k], uv[-1, k], image_size_wh)))
                for k in range(2)], bool)
            if not active.any():
                # A renderer has no original-sensor samples to connect here.
                # Record that this leaf was not given an in-frame error claim.
                skipped_offscreen += 1
                return [right]
            if np.all(valid[:, active]):
                maximum = max(_finite_chord_distance(uv[0, k], uv[-1, k], uv[i, k])
                              for k in np.flatnonzero(active) for i in (1, 2, 3))
                if maximum <= tolerance_px:
                    leaf_errors.append(maximum)
                    return [right]
            if depth >= max_depth or visited_nodes >= max_nodes_per_segment:
                unresolved += 1
                leaf_errors.append(np.nan)
                # A sampled optical-domain transition must interrupt drawing.
                # Keep the three tested interior vertices when any active
                # contour mixes valid and invalid states on this leaf.
                return ([test[1], test[2], test[3], right]
                        if not np.all(valid[:, active]) else [right])
            middle = .5*(left+right)
            return recur(left, middle, depth+1)+recur(middle, right, depth+1)

        evaluate(0.)
        if not projected_top:
            first, good = evaluate(0.)
            projected_top.append(first[0].copy())
            projected_contact.append(first[1].copy())
            validity_top.append(bool(good[0]))
            validity_contact.append(bool(good[1]))
        leaves = recur(0., 1., 0) if segment in selected else [1.]
        for t in leaves:
            uv, good = evaluate(t)
            world_top.append(a_top+t*(b_top-a_top))
            world_contact.append(a_contact+t*(b_contact-a_contact))
            projected_top.append(uv[0].copy())
            projected_contact.append(uv[1].copy())
            validity_top.append(bool(good[0]))
            validity_contact.append(bool(good[1]))
            segment_origin.append(segment)
            fractions.append(t)
        offsets.append(len(world_top)-1)
        finite_error = np.asarray(leaf_errors, float)
        finite_error = finite_error[np.isfinite(finite_error)]
        reports.append({"segment_index": segment, "selected": segment in selected,
                        "leaf_count": len(leaves), "inserted_points": len(leaves)-1,
                        "unresolved_leaf_count": unresolved,
                        "offscreen_or_invalid_leaf_count": skipped_offscreen,
                        "max_sampled_leaf_chord_error_px": (float(np.max(finite_error))
                                                              if len(finite_error) else None)})

    result = RefinedBoardPaths(
        top_xyz=np.asarray(world_top), contact_xyz=np.asarray(world_contact),
        top_uv=np.asarray(projected_top), contact_uv=np.asarray(projected_contact),
        top_valid=np.asarray(validity_top), contact_valid=np.asarray(validity_contact),
        segment_offsets=np.asarray(offsets, np.int32),
        original_segment=np.asarray(segment_origin, np.int32),
        original_fraction=np.asarray(fractions),
        segment_report=tuple(reports), tolerance_px=float(tolerance_px))
    if (not np.array_equal(result.top_xyz[0], result.top_xyz[-1])
            or not np.array_equal(result.contact_xyz[0], result.contact_xyz[-1])
            or not np.allclose(result.top_xyz[:, :2], result.contact_xyz[:, :2],
                               rtol=0., atol=1e-10)):
        raise AssertionError("Refinement broke loop closure or top/contact correspondence")
    return result


def refine_vertical_edges(camera25, board: RefinedBoardPaths, *,
                          tolerance_px: float = .125, max_depth: int = 16,
                          max_nodes_per_edge: int = 256,
                          image_size_wh: tuple[int, int] = (1920, 1080)
                          ) -> tuple[RefinedVerticalEdge, ...]:
    """Refine the projected vertical sides shared by adjacent wall facets.

    A side is computed once per paired world vertex and is reused in reverse
    order by its neighboring polygon. Thus their common boundary has exactly
    the same rasterized vertices, even when different edges need different
    vertical fractions. An invalid optical sample makes a facet ineligible
    for filling; no polygon is bridged through an invalid Brown domain.
    """
    camera = np.asarray(camera25, np.float64)
    if (camera.shape != (25,) or not np.isfinite(camera).all()
            or not np.isfinite(tolerance_px) or tolerance_px <= 0
            or max_depth < 1 or max_nodes_per_edge < 5):
        raise ValueError("Invalid camera or vertical refinement controls")
    if (not np.array_equal(board.top_xyz[0], board.top_xyz[-1])
            or not np.array_equal(board.contact_xyz[0], board.contact_xyz[-1])):
        raise ValueError("Paired board contour must be closed")
    n = len(board.top_xyz)-1
    initial_fractions = np.asarray([0., .25, .5, .75, 1.])
    world = (board.top_xyz[:n, None, :]
             + initial_fractions[None, :, None]
             * (board.contact_xyz[:n, None, :]-board.top_xyz[:n, None, :]))
    projected, _, supported = safe_project(camera, world.reshape(-1, 3))
    projected = projected.reshape(n, len(initial_fractions), 2)
    supported = supported.reshape(n, len(initial_fractions))
    paths: list[RefinedVerticalEdge] = []
    for vertex in range(n):
        start, end = board.top_xyz[vertex], board.contact_xyz[vertex]
        cache: dict[float, tuple[np.ndarray, bool]] = {
            float(t): (projected[vertex, j], bool(supported[vertex, j]))
            for j, t in enumerate(initial_fractions)}
        unresolved, skipped, visited = 0, 0, 0

        def evaluate(fraction: float):
            if fraction not in cache:
                uv, _, valid = safe_project(camera, (start+fraction*(end-start))[None, :])
                cache[fraction] = (uv[0], bool(valid[0]))
            return cache[fraction]

        def recur(left: float, right: float, depth: int) -> list[float]:
            nonlocal unresolved, skipped, visited
            visited += 1
            samples = (left, left+.25*(right-left), .5*(left+right),
                       left+.75*(right-left), right)
            rows = [evaluate(t) for t in samples]
            uv = np.asarray([row[0] for row in rows])
            valid = np.asarray([row[1] for row in rows], bool)
            active = (any(_in_sensor(uv[i], image_size_wh)
                          for i in range(5) if valid[i])
                      or (valid[0] and valid[-1]
                          and _chord_intersects_sensor(uv[0], uv[-1], image_size_wh)))
            if not active:
                skipped += 1
                return [right]
            if valid.all():
                error = max(_finite_chord_distance(uv[0], uv[-1], uv[i])
                            for i in (1, 2, 3))
                if error <= tolerance_px:
                    return [right]
            if depth >= max_depth or visited >= max_nodes_per_edge:
                unresolved += 1
                return ([samples[1], samples[2], samples[3], right]
                        if not valid.all() else [right])
            middle = .5*(left+right)
            return recur(left, middle, depth+1)+recur(middle, right, depth+1)

        fractions = np.asarray([0., *recur(0., 1., 0)])
        rows = [evaluate(float(t)) for t in fractions]
        paths.append(RefinedVerticalEdge(
            fractions=fractions,
            uv=np.asarray([row[0] for row in rows]),
            valid=np.asarray([row[1] for row in rows], bool),
            unresolved_leaves=unresolved,
            skipped_offscreen_leaves=skipped))
    return (*paths, paths[0])
