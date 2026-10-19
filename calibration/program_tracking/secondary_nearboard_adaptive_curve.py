"""Shared opt-in adaptive projection for source-backed physical curves.

The confirmed secondary near-board top uses horizontal source intervals;
same-event reference curves use source points.  Their source measurements,
physical world vertices, lens model, and graph variables are unchanged.  A
sampled world polyline finds centrally valid candidate regions, then a direct
one-dimensional search uses the actual projected world curve for each nearest
point.  A coarse projected chord is never itself a physical-domain rejection.

All centrally valid coarse segments that can compete with any source interval
are refined, not just the initially nearest segment.  Mixed valid/invalid
leaves are split and never bridged.  A single physical Jacobian uses the union
of subdivisions needed by the base state and all its +/- perturbations, then
evaluates every residual on that *same* world-node topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Mapping

import numpy as np

try:
    from . import secondary_nearboard_top_source as source
    from .adaptive_curve_parallel import AdaptiveCurveJob
    from .continuous_brown_adapter import adapt_factor
    from .projection import from_state, project_central_valid
    from .reference_board_pair import BoardIntersectionError
    from .robust_observations import pointwise_huber
except ImportError:
    import secondary_nearboard_top_source as source
    from adaptive_curve_parallel import AdaptiveCurveJob
    from continuous_brown_adapter import adapt_factor
    from projection import from_state, project_central_valid
    from reference_board_pair import BoardIntersectionError
    from robust_observations import pointwise_huber


TOLERANCE_PX = .01
COMPETITION_MARGIN_PX = 1.0
MAX_INITIAL_WORLD_STEP = .25
MAX_SUBDIVISION_DEPTH = 14
MAX_WORLD_NODES = 16000
CONTINUOUS_SEARCH_STEPS = 45


class AdaptiveCurveResolutionError(RuntimeError):
    """The numerical approximation budget failed; not a physical domain error."""


@dataclass(frozen=True)
class AdaptiveWorldPolyline:
    """Unchanged world segments parameterized by cumulative chord length."""

    base_x: np.ndarray
    original_arc: np.ndarray
    vertices_xyz: np.ndarray

    @classmethod
    def from_world(cls, world_xyz: np.ndarray,
                   *, max_initial_world_step: float = MAX_INITIAL_WORLD_STEP
                   ) -> "AdaptiveWorldPolyline":
        xyz = np.asarray(world_xyz, np.float64)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
                or not np.isfinite(xyz).all()):
            raise ValueError("Adaptive projection needs a finite world polyline")
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if np.any(step <= 1e-12) or not np.isfinite(max_initial_world_step) or max_initial_world_step <= 0:
            raise ValueError("World polyline has repeated or degenerate vertices")
        arc = np.r_[0., np.cumsum(step)]
        seeded = [0.]
        for left, right in zip(arc[:-1], arc[1:]):
            count = int(np.ceil((right - left) / max_initial_world_step))
            seeded.extend(np.linspace(left, right, count + 1)[1:])
        return cls(np.asarray(seeded, np.float64), arc, xyz.copy())

    def world(self, x: np.ndarray) -> np.ndarray:
        arc = np.asarray(x, np.float64).reshape(-1)
        if np.any(~np.isfinite(arc)) or np.any(arc < self.original_arc[0]) or np.any(arc > self.original_arc[-1]):
            raise ValueError("Adaptive point outside original world polyline")
        index = np.clip(np.searchsorted(self.original_arc, arc, side="right") - 1,
                        0, len(self.original_arc) - 2)
        fraction = ((arc - self.original_arc[index])
                    / (self.original_arc[index + 1] - self.original_arc[index]))
        return (self.vertices_xyz[index]
                + fraction[:, None]
                * (self.vertices_xyz[index + 1] - self.vertices_xyz[index]))


class _ProjectionCache:
    def __init__(self, project_x: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]):
        self.project_x = project_x
        self.cache: dict[float, int] = {}
        self._uv = np.empty((0, 2), np.float64)
        self._valid = np.empty(0, bool)
        self._size = 0

    def sample(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(x, np.float64).reshape(-1)
        # Search only the requested coordinates. Subtracting dict_keys scans
        # the growing history on every continuous-search step.
        coordinates = requested.tolist()
        missing = sorted({coordinate for coordinate in coordinates
                          if coordinate not in self.cache})
        if missing:
            uv, valid = self.project_x(np.asarray(missing, np.float64))
            uv = np.asarray(uv, np.float64).reshape(len(missing), 2)
            valid = np.asarray(valid, bool).reshape(len(missing))
            needed = self._size + len(missing)
            if needed > len(self._valid):
                capacity = max(needed, max(128, 2 * len(self._valid)))
                new_uv = np.empty((capacity, 2), np.float64)
                new_valid = np.empty(capacity, bool)
                new_uv[:self._size] = self._uv[:self._size]
                new_valid[:self._size] = self._valid[:self._size]
                self._uv, self._valid = new_uv, new_valid
            self._uv[self._size:needed] = uv
            self._valid[self._size:needed] = valid
            self.cache.update(zip(missing, range(self._size, needed)))
            self._size = needed
        index = np.fromiter((self.cache[coordinate] for coordinate in coordinates),
                            dtype=np.intp, count=len(coordinates))
        return self._uv[index], self._valid[index]


def _physical_projector(camera25: np.ndarray,
                        line: AdaptiveWorldPolyline) -> _ProjectionCache:
    def project(x):
        uv, _, valid = project_central_valid(camera25, line.world(x))
        return uv, valid
    return _ProjectionCache(project)


def _interval_distance_by_segment(a: np.ndarray, b: np.ndarray,
                                  low: float, high: float, y: float
                                  ) -> np.ndarray:
    """Distances of every projected chord to one observed horizontal band."""
    direction = b - a
    length2 = np.maximum(np.sum(direction * direction, axis=1), 1e-12)
    candidates = []
    for endpoint in (a, b):
        dx = endpoint[:, 0] - np.clip(endpoint[:, 0], low, high)
        dy = endpoint[:, 1] - y
        candidates.append(dx * dx + dy * dy)
    for qx in (low, high):
        q = np.array([qx, y])
        t = np.clip(np.sum((q[None, :] - a) * direction, axis=1) / length2, 0., 1.)
        r = a + t[:, None] * direction - q
        candidates.append(np.sum(r * r, axis=1))
    squared = np.min(np.stack(candidates), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y - a[:, 1]) / direction[:, 1]
    x = a[:, 0] + t * direction[:, 0]
    crossing = (np.isfinite(t) & (t >= 0.) & (t <= 1.)
                & (x >= low) & (x <= high))
    squared[crossing] = 0.
    return np.sqrt(squared)


def _interval_distance_matrix(a: np.ndarray, b: np.ndarray,
                              observations) -> np.ndarray:
    """The same chord distances, batching intervals with bounded scratch space."""
    bands = np.asarray(observations, np.float64).reshape(-1, 3)
    result = np.empty((len(a), len(bands)), np.float64)
    direction = b - a
    length2 = np.maximum(np.sum(direction * direction, axis=1), 1e-12)
    chunk = max(1, 131072 // max(1, len(a)))
    # Original reference-curve observations are points (low == high).  The
    # general interval formula constructs four candidate matrices and several
    # NxMx2 arrays; a point needs only its clamped chord projection.  Retain
    # endpoint candidates for the length2 floor on degenerate chords.
    if np.all(bands[:, 0] == bands[:, 1]):
        for start in range(0, len(bands), chunk):
            point = bands[start:start + chunk]
            rx = point[None, :, 0] - a[:, None, 0]
            ry = point[None, :, 2] - a[:, None, 1]
            t = np.clip((rx * direction[:, None, 0]
                         + ry * direction[:, None, 1]) / length2[:, None],
                        0., 1.)
            dx = t * direction[:, None, 0] - rx
            dy = t * direction[:, None, 1] - ry
            squared = dx * dx + dy * dy
            end_dx = direction[:, None, 0] - rx
            end_dy = direction[:, None, 1] - ry
            np.minimum(squared, rx * rx + ry * ry, out=squared)
            np.minimum(squared, end_dx * end_dx + end_dy * end_dy,
                       out=squared)
            with np.errstate(divide="ignore", invalid="ignore"):
                crossing_t = ry / direction[:, None, 1]
            crossing_x = a[:, None, 0] + crossing_t * direction[:, None, 0]
            crossing = (np.isfinite(crossing_t) & (crossing_t >= 0.)
                        & (crossing_t <= 1.) & (crossing_x == point[None, :, 0]))
            squared[crossing] = 0.
            result[:, start:start + len(point)] = np.sqrt(squared)
        return result
    for start in range(0, len(bands), chunk):
        low, high, y = bands[start:start + chunk].T
        candidates = []
        for endpoint in (a, b):
            dx = endpoint[:, 0, None] - np.clip(endpoint[:, 0, None], low, high)
            dy = endpoint[:, 1, None] - y
            candidates.append(dx * dx + dy * dy)
        for qx in (low, high):
            q = np.column_stack((qx, y))
            t = np.clip(np.sum((q[None, :, :] - a[:, None, :])
                               * direction[:, None, :], axis=2)
                        / length2[:, None], 0., 1.)
            r = a[:, None, :] + t[:, :, None] * direction[:, None, :] - q
            candidates.append(np.sum(r * r, axis=2))
        squared = np.min(np.stack(candidates), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (y - a[:, 1, None]) / direction[:, 1, None]
        x = a[:, 0, None] + t * direction[:, 0, None]
        crossing = (np.isfinite(t) & (t >= 0.) & (t <= 1.)
                    & (x >= low) & (x <= high))
        squared[crossing] = 0.
        result[:, start:start + len(low)] = np.sqrt(squared)
    return result


def _samples(nodes: np.ndarray, projector: _ProjectionCache):
    a, b = nodes[:-1], nodes[1:]
    fractions = np.asarray([0., .25, .5, .75, 1.])
    x = a[:, None] + (b - a)[:, None] * fractions[None, :]
    uv, valid = projector.sample(x.reshape(-1))
    return uv.reshape(len(a), 5, 2), valid.reshape(len(a), 5)


def _screen_chord_error(samples: np.ndarray, *, geometric: bool) -> np.ndarray:
    """Bound projected-curve shape, not world-parameter image speed.

    The original parametric rule is retained exactly for existing factors.
    The opt-in measures each interior projection against its closest point on
    the finite image chord. Perspective maps a straight world segment to a
    straight screen line but does not preserve its quarter/midpoint timing.
    """
    start, finish = samples[:, 0], samples[:, 4]
    interior = samples[:, 1:4]
    if not geometric:
        chord = (start[:, None, :]
                 + np.asarray([.25, .5, .75])[None, :, None]
                 * (finish-start)[:, None, :])
        return np.max(np.linalg.norm(interior-chord, axis=2), axis=1)
    direction = finish-start
    length2 = np.maximum(np.sum(direction*direction, axis=1), 1e-30)
    fraction = np.clip(np.sum((interior-start[:, None, :])
                              * direction[:, None, :], axis=2)
                       / length2[:, None], 0., 1.)
    chord = start[:, None, :] + fraction[:, :, None]*direction[:, None, :]
    return np.max(np.linalg.norm(interior-chord, axis=2), axis=1)


def _refine_one(nodes: np.ndarray, projector: _ProjectionCache,
                observations: tuple[tuple[float, float, float], ...],
                *, tolerance_px: float, competition_margin_px: float,
                min_world_span: float, max_world_nodes: int,
                geometric_chord_error: bool = False) -> np.ndarray:
    """Refine every empirical competitor and every central-domain transition."""
    current = np.asarray(nodes, np.float64)
    while True:
        samples, valid = _samples(current, projector)
        full = np.all(valid, axis=1)
        mixed = np.any(valid, axis=1) & ~full
        error = np.full(len(full), np.inf)
        if np.any(full):
            error[full] = _screen_chord_error(
                samples[full], geometric=geometric_chord_error)
        candidates = np.zeros(len(full), bool)
        if np.any(full):
            a, b = samples[full, 0], samples[full, 4]
            distances = _interval_distance_matrix(a, b, observations)
            best = np.min(distances, axis=0)
            # A curved candidate may approach the interval more closely than
            # its chord.  The measured sample deviation is used generously in
            # screening, and the selection is repeated after every split.
            candidates[full] = np.any(
                distances - 2. * error[full, None]
                <= best[None, :] + competition_margin_px, axis=1)
        span = np.diff(current)
        # A mixed-validity leaf is a real optical-domain boundary.  At the
        # numerical resolution floor it remains disconnected; it is never
        # bridged or called an underresolved physical camera.
        split = (mixed & (span > min_world_span)) | (full & candidates &
                (error > tolerance_px))
        if not np.any(split):
            return current
        unresolved_valid = full & candidates & (error > tolerance_px)
        if np.any(unresolved_valid & (span <= min_world_span)):
            raise AdaptiveCurveResolutionError(
                "Valid near-board curve exhausted numerical subdivision; "
                "no physical camera/domain conclusion follows")
        if len(current) + int(split.sum()) > max_world_nodes:
            raise AdaptiveCurveResolutionError(
                "Adaptive near-board curve exceeded the numerical node budget")
        mids = .5 * (current[:-1][split] + current[1:][split])
        current = np.sort(np.r_[current, mids])


def _fixed_topology_residual(nodes: np.ndarray, projector: _ProjectionCache,
                             observations: tuple[tuple[float, float, float], ...],
                             source_keys: tuple[int, ...], factor_family: str,
                             no_central_reason: str,
                             competition_margin_px: float = COMPETITION_MARGIN_PX,
                             geometric_chord_error: bool = False
                             ) -> tuple[np.ndarray, dict]:
    samples, valid = _samples(nodes, projector)
    full = np.all(valid, axis=1)
    selected = np.flatnonzero(full)
    if not len(selected):
        error = BoardIntersectionError(no_central_reason, np.arange(len(nodes) - 1))
        error.factor_family = factor_family
        error.source_keys = source_keys
        raise error
    sampled_error = _screen_chord_error(
        samples[selected], geometric=geometric_chord_error)
    a, b = samples[selected, 0], samples[selected, 4]
    intervals = np.asarray(observations, np.float64)
    interval_distances = _interval_distance_matrix(a, b, observations)
    brackets = []
    for obs_index, (low, high, y) in enumerate(observations):
        distances = interval_distances[:, obs_index]
        best = float(np.min(distances))
        plausible = selected[distances - 2. * sampled_error
                             <= best + competition_margin_px]
        if not len(plausible):
            raise AdaptiveCurveResolutionError("No numerical curve candidates")
        cuts = np.r_[0, np.flatnonzero(np.diff(plausible) > 1) + 1,
                     len(plausible)]
        for begin, end in zip(cuts[:-1], cuts[1:]):
            group = plausible[begin:end]
            first, last = int(group[0]), int(group[-1])
            brackets.append((obs_index, float(nodes[first]),
                             float(nodes[last + 1])))
            # A long contiguous group can contain more than one local valley.
            # Also search around its best sampled chord, without splitting the
            # original physical segment or crossing an invalid optical gap.
            winner = int(group[np.argmin(distances[np.searchsorted(selected, group)])])
            local_left, local_right = max(first, winner - 1), min(last + 1, winner + 2)
            if local_left > first or local_right < last + 1:
                brackets.append((obs_index, float(nodes[local_left]),
                                 float(nodes[local_right])))

    bracket_obs = np.asarray([item[0] for item in brackets], np.int64)
    original_left = np.asarray([item[1] for item in brackets], np.float64)
    original_right = np.asarray([item[2] for item in brackets], np.float64)
    golden = (np.sqrt(5.) - 1.) / 2.

    def evaluate(x: np.ndarray, obs: np.ndarray):
        # Search coordinates are new on almost every iteration.  The node
        # cache only helps the repeated topology samples; inserting every
        # bisection coordinate into it grows the dict and copies the results.
        uv, central = projector.project_x(x)
        if not np.all(central):
            raise AdaptiveCurveResolutionError(
                "Continuous curve search sampled an unresolved optical-domain gap")
        band = intervals[obs]
        residual = uv - np.column_stack((np.clip(uv[:, 0], band[:, 0], band[:, 1]),
                                         band[:, 2]))
        return np.sum(residual * residual, axis=1), residual

    def gradient(x: np.ndarray):
        # Minimize the actual continuous projection.  Differencing the large
        # squared residual itself is inaccurate at a flat nearest point;
        # r dot d(uv)/d(world arc) remains well scaled there.
        width = np.minimum(1e-4, .25 * (original_right - original_left))
        x_low = np.maximum(original_left, x - width)
        x_high = np.minimum(original_right, x + width)
        all_uv, all_central = projector.project_x(np.r_[x, x_low, x_high])
        uv, uv_low, uv_high = np.split(all_uv, 3)
        central, central_low, central_high = np.split(all_central, 3)
        if not np.all(central & central_low & central_high):
            raise AdaptiveCurveResolutionError(
                "Continuous curve tangent met an unresolved optical-domain gap")
        tangent = (uv_high - uv_low) / (x_high - x_low)[:, None]
        band = intervals[bracket_obs]
        residual = uv - np.column_stack((np.clip(uv[:, 0], band[:, 0], band[:, 1]),
                                         band[:, 2]))
        return np.sum(residual * tangent, axis=1)

    left_gradient, right_gradient = gradient(original_left), gradient(original_right)
    root_exists = (left_gradient <= 0.) & (right_gradient >= 0.)
    root_left, root_right = original_left.copy(), original_right.copy()
    for _ in range(CONTINUOUS_SEARCH_STEPS):
        middle = .5 * (root_left + root_right)
        negative = gradient(middle) < 0.
        root_left = np.where(root_exists & negative, middle, root_left)
        root_right = np.where(root_exists & ~negative, middle, root_right)
    root_x = .5 * (root_left + root_right)
    local_x = root_x.copy()
    root_score, _ = evaluate(root_x, bracket_obs)
    end_x = np.column_stack((original_left, original_right))
    end_score, _ = evaluate(end_x.reshape(-1), np.repeat(bracket_obs, 2))
    end_score = end_score.reshape(-1, 2)
    better_end = root_exists & (np.min(end_score, axis=1) < root_score - 1e-7)
    end_choice = np.argmin(end_score, axis=1)
    local_x[better_end] = end_x[np.arange(len(brackets)), end_choice][better_end]

    fallback = np.flatnonzero(~root_exists)
    if len(fallback):
        left, right = original_left[fallback].copy(), original_right[fallback].copy()
        fallback_obs = bracket_obs[fallback]
        for _ in range(CONTINUOUS_SEARCH_STEPS):
            x_left = right - golden * (right - left)
            x_right = left + golden * (right - left)
            pair_score, _ = evaluate(np.r_[x_left, x_right],
                                     np.r_[fallback_obs, fallback_obs])
            score_left, score_right = np.split(pair_score, 2)
            left_wins = score_left <= score_right
            right = np.where(left_wins, x_right, right)
            left = np.where(left_wins, left, x_left)
        candidates = np.column_stack((left, .5 * (left + right), right,
                                      original_left[fallback], original_right[fallback]))
        score, _ = evaluate(candidates.reshape(-1),
                            np.repeat(fallback_obs, candidates.shape[1]))
        choice = np.argmin(score.reshape(len(fallback), -1), axis=1)
        local_x[fallback] = candidates[np.arange(len(fallback)), choice]

    local_score, local_residual = evaluate(local_x, bracket_obs)
    values, winning, winning_arc = [], [], []
    for obs_index in range(len(observations)):
        options = np.flatnonzero(bracket_obs == obs_index)
        best = int(options[np.argmin(local_score[options])])
        values.extend(local_residual[best])
        winning_arc.append(float(local_x[best]))
        winning.append(int(selected[np.argmin(np.abs(
            .5 * (nodes[selected] + nodes[selected + 1]) - local_x[best]))]))
    answer = np.asarray(values, np.float64)
    if not np.isfinite(answer).all():
        raise AdaptiveCurveResolutionError("Adaptive physical-curve residual became nonfinite")
    max_winning_error = float(max(sampled_error[np.searchsorted(selected, winning)]))
    return answer, dict(world_nodes=len(nodes), central_segments=len(selected),
                        max_winning_sampled_chord_error_px=max_winning_error,
                        winning_segments=winning,
                        winning_world_arc=winning_arc,
                        continuous_search_brackets=len(brackets))


class AdaptiveProjectedPolyline:
    def __init__(self, world_xyz: np.ndarray, measurements: list[dict],
                 *, tolerance_px: float = TOLERANCE_PX,
                 competition_margin_px: float = COMPETITION_MARGIN_PX,
                 max_subdivision_depth: int = MAX_SUBDIVISION_DEPTH,
                 max_world_nodes: int = MAX_WORLD_NODES,
                 max_initial_world_step: float = MAX_INITIAL_WORLD_STEP,
                 closed: bool = False,
                 geometric_chord_error: bool = False,
                 independent_fd_topology: bool = False,
                 factor_family: str = "original_secondary_lower_board_top",
                 no_central_reason: str = "nearboard_no_central_curve"):
        vertices = np.asarray(world_xyz, np.float64)
        if closed:
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
                raise ValueError("Closed physical curve needs at least three vertices")
            if np.linalg.norm(vertices[0] - vertices[-1]) > 1e-12:
                vertices = np.vstack((vertices, vertices[0]))
        self.line = AdaptiveWorldPolyline.from_world(
            vertices, max_initial_world_step=max_initial_world_step)
        self.observations = tuple((float(item["x_interval_px"][0]),
                                   float(item["x_interval_px"][1]),
                                   float(item["y"])) for item in measurements)
        if (not self.observations or any(not (0 <= lo <= hi < 1920 and 0 <= y < 1080)
                                         for lo, hi, y in self.observations)):
            raise ValueError("Need finite original-sensor horizontal intervals")
        if (not np.isfinite(tolerance_px) or tolerance_px <= 0
                or not np.isfinite(competition_margin_px) or competition_margin_px < 0
                or max_subdivision_depth < 1 or max_world_nodes < len(self.line.base_x)):
            raise ValueError("Invalid adaptive numerical tolerance or node budget")
        self.tolerance_px = float(tolerance_px)
        self.competition_margin_px = float(competition_margin_px)
        self.min_world_span = (float(np.min(np.diff(self.line.base_x)))
                               / 2.**int(max_subdivision_depth))
        self.max_world_nodes = int(max_world_nodes)
        self.geometric_chord_error = bool(geometric_chord_error)
        self.independent_fd_topology = bool(independent_fd_topology)
        self.factor_family = factor_family
        self.no_central_reason = no_central_reason

    def _refine(self, nodes: np.ndarray, projector: _ProjectionCache) -> np.ndarray:
        return _refine_one(nodes, projector, self.observations,
                           tolerance_px=self.tolerance_px,
                           competition_margin_px=self.competition_margin_px,
                           min_world_span=self.min_world_span,
                           max_world_nodes=self.max_world_nodes,
                           geometric_chord_error=self.geometric_chord_error)

    def residual(self, camera25: np.ndarray, *, source_keys=()
                 ) -> tuple[np.ndarray, dict]:
        projector = _physical_projector(camera25, self.line)
        nodes = self._refine(self.line.base_x, projector)
        answer, diagnostic = _fixed_topology_residual(
            nodes, projector, self.observations, tuple(source_keys),
            self.factor_family, self.no_central_reason,
            self.competition_margin_px,
            geometric_chord_error=self.geometric_chord_error)
        if diagnostic["max_winning_sampled_chord_error_px"] > self.tolerance_px + 1e-9:
            raise AdaptiveCurveResolutionError("Winning curve segment was not refined")
        return answer, diagnostic

    def value_and_physical_jacobian(self, state: np.ndarray,
                                    template25: np.ndarray, *, source_keys=(),
                                    step_scale: float = 1e-6
                                    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """One union topology for base and all 2*physical_dim FD cameras."""
        physical = np.asarray(state, np.float64)
        if physical.shape not in ((11,), (15,)) or not np.isfinite(physical).all():
            raise ValueError("Need one finite physical camera state")
        if not np.isfinite(step_scale) or step_scale <= 0:
            raise ValueError("Positive FD scale required")
        states = [physical.copy()]
        widths = []
        for column in range(len(physical)):
            width = step_scale * max(1., abs(float(physical[column])))
            high, low = physical.copy(), physical.copy()
            high[column] += width
            low[column] -= width
            states.extend((high, low))
            widths.append(width)
        projectors = [_physical_projector(from_state(s, template25), self.line)
                      for s in states]
        if self.independent_fd_topology:
            # Opt-in for a curve crossing a moving central-domain horizon.
            # The true factor is nearest distance to the continuous projected
            # physical curve, not distance to a sampled chord. Each camera is
            # independently refined to that same target. Requiring every
            # +/- perturbation's optical-boundary vertices in one common mesh
            # can grow without improving any of those individual minima.
            evaluated = [self.residual(from_state(s, template25),
                                       source_keys=source_keys)
                         for s in states]
            residual = evaluated[0][0]
            jac = np.empty((len(residual), len(physical)), np.float64)
            for column, width in enumerate(widths):
                jac[:, column] = (evaluated[1+2*column][0]
                                  - evaluated[2+2*column][0])/(2.*width)
            if not np.isfinite(jac).all():
                raise AdaptiveCurveResolutionError(
                    "Independent physical-curve Jacobian became nonfinite")
            return residual, jac, dict(
                evaluated[0][1],
                jacobian_topology='independently_refined_same_continuous_curve',
                max_independent_world_nodes=max(item[1]['world_nodes']
                                                for item in evaluated),
                max_perturbed_winning_chord_error_px=max(
                    item[1]['max_winning_sampled_chord_error_px']
                    for item in evaluated))
        nodes = self.line.base_x.copy()
        # Each projector may reveal a new competitor after another state's
        # subdivisions are unioned.  Iterate to a common fixed topology.
        while True:
            before = len(nodes)
            for projector in projectors:
                refined = self._refine(nodes, projector)
                if len(refined) != len(nodes):
                    nodes = refined
            if len(nodes) == before:
                break
        evaluated = [_fixed_topology_residual(nodes, projector,
                                              self.observations, tuple(source_keys),
                                              self.factor_family, self.no_central_reason,
                                              self.competition_margin_px,
                                              geometric_chord_error=self.geometric_chord_error)
                     for projector in projectors]
        residual = evaluated[0][0]
        jac = np.empty((len(residual), len(physical)), np.float64)
        for column, width in enumerate(widths):
            jac[:, column] = (evaluated[1 + 2 * column][0]
                              - evaluated[2 + 2 * column][0]) / (2. * width)
        if not np.isfinite(jac).all():
            raise AdaptiveCurveResolutionError("Adaptive physical Jacobian nonfinite")
        diagnostic = dict(evaluated[0][1],
                          jacobian_union_world_nodes=len(nodes),
                          max_perturbed_winning_chord_error_px=max(
                              item[1]["max_winning_sampled_chord_error_px"]
                              for item in evaluated))
        if diagnostic["max_perturbed_winning_chord_error_px"] > self.tolerance_px + 1e-9:
            raise AdaptiveCurveResolutionError("Union topology missed a perturbed winner")
        return residual, jac, diagnostic


# A 2-D source point is represented by lo==hi==pixel x.  The same numerical
# engine serves reference curves and the near-board interval factor without
# changing source identities or factor likelihoods.
AdaptiveNearboardCurve = AdaptiveProjectedPolyline


def adaptive_curve_callback(curve: AdaptiveProjectedPolyline,
                            template25: np.ndarray,
                            source_keys: tuple[int, ...], *, with_prime: bool = False):
    """Use the same continuous target with a cheap value-only trial path."""
    lock = RLock()
    last_value_state: bytes | None = None
    last_value: np.ndarray | None = None
    last_bundle_state: bytes | None = None
    last_bundle: tuple[np.ndarray, np.ndarray, dict] | None = None

    def unpack(states):
        if len(states) != 1:
            raise ValueError("Adaptive curve factor expects one derived camera")
        state = np.asarray(states[0], np.float64)
        return state, state.tobytes()

    def residual(states):
        nonlocal last_value_state, last_value
        try:
            state, key = unpack(states)
            with lock:
                if key == last_bundle_state:
                    return last_bundle[0].copy()
                if key != last_value_state:
                    camera = from_state(state, template25)
                    last_value = curve.residual(camera, source_keys=source_keys)[0]
                    last_value_state = key
                return last_value.copy()
        except AdaptiveCurveResolutionError as exc:
            exc.evaluation_stage = "residual"
            exc.factor_family = curve.factor_family
            exc.source_keys = source_keys
            raise

    def derivative(states):
        nonlocal last_bundle_state, last_bundle
        try:
            state, key = unpack(states)
            with lock:
                if key != last_bundle_state:
                    candidate = curve.value_and_physical_jacobian(
                        state, template25, source_keys=source_keys)
                    if (key == last_value_state and
                            np.max(np.abs(candidate[0] - last_value)) > 1e-3):
                        raise AdaptiveCurveResolutionError(
                            "Value/Jacobian curve representations differ by over 0.001px")
                    last_bundle = candidate
                    last_bundle_state = key
                return [last_bundle[1].copy()]
        except AdaptiveCurveResolutionError as exc:
            exc.evaluation_stage = "jacobian"
            exc.factor_family = curve.factor_family
            exc.source_keys = source_keys
            raise

    def prime(state, bundle):
        """Install an exact-state bundle computed for this curve in a worker."""
        nonlocal last_bundle_state, last_bundle
        _, key = unpack([state])
        value, jacobian, diagnostic = bundle
        value = np.asarray(value, np.float64).reshape(-1).copy()
        jacobian = np.asarray(jacobian, np.float64).copy()
        if (value.shape != (2 * len(curve.observations),)
                or jacobian.shape != (len(value), len(np.asarray(state)))
                or not np.isfinite(value).all() or not np.isfinite(jacobian).all()):
            raise AdaptiveCurveResolutionError("Invalid prepared physical curve bundle")
        with lock:
            if (key == last_value_state and
                    np.max(np.abs(value - last_value)) > 1e-3):
                error = AdaptiveCurveResolutionError(
                    "Value/Jacobian curve representations differ by over 0.001px")
                error.evaluation_stage = "jacobian"
                error.factor_family = curve.factor_family
                error.source_keys = source_keys
                raise error
            last_bundle = value, jacobian, diagnostic
            last_bundle_state = key

    return (residual, derivative, prime) if with_prime else (residual, derivative)


def attach_secondary_nearboard_adaptive_factors(
    graph, data: dict, model, initial: Mapping[int, np.ndarray], rows: list[dict],
    *, tolerance_px: float = TOLERANCE_PX
) -> dict:
    """Opt-in replacement for only the new near-board factor family."""
    if data["camera_id"] != "secondary_long_side":
        raise ValueError("Adaptive near-board factors require the secondary group")
    active = set(map(int, initial))
    if not hasattr(graph, "_adaptive_curve_jobs"):
        graph._adaptive_curve_jobs = []
    for row in rows:
        pts = int(row["representative_pts"])
        if pts not in model.frames:
            raise ValueError("Near-board PTS missing from shared rig model")
        source_keys = tuple(map(int, row["reviewed_native_pts"]))
        curve = AdaptiveNearboardCurve(row["world_xyz"], row["measurements"],
                                      tolerance_px=tolerance_px)
        template = model.frames[pts].template25
        raw, jac, prime = adaptive_curve_callback(
            curve, template, source_keys, with_prime=True)
        raw, jac = pointwise_huber(raw, jac, group_size=2,
                                   delta=2. * source.EDGE_SIGMA_PX)
        adapted = adapt_factor(
            (pts,), raw, jac,
            dict(residual_dim=2 * len(row["measurements"]),
                 sigma=source.EDGE_SIGMA_PX * np.sqrt(len(row["measurements"]))),
            active, data["states"], model)
        if adapted is None:
            raise ValueError("Near-board factor has no active camera dependency")
        keys, residual, derivative, options = adapted

        def checked_residual(q, *, fn=residual, native=source_keys):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = "original_secondary_lower_board_top"
                exc.source_keys = native
                raise

        def checked_derivative(q, *, fn=derivative, native=source_keys):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                exc.factor_family = "original_secondary_lower_board_top"
                exc.source_keys = native
                raise

        graph.add_observation(keys, checked_residual,
                              jac_fn=checked_derivative, **options)
        graph._adaptive_curve_jobs.append(AdaptiveCurveJob(
            curve, np.asarray(template, np.float64).copy(), model,
            pts, source_keys, prime))
    return dict(factor_family="original_secondary_lower_board_top",
                source_exposure_factors=len(rows),
                representation="adaptive_competing_central_segments",
                sampled_chord_tolerance_px=tolerance_px,
                jacobian_topology="union_base_plus_minus_all_physical_columns",
                source_uv_modified=False,
                physical_world_modified=False,
                existing_factors_modified=False,
                solve_started=False)

