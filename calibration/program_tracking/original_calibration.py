"""Attach the three original, source-annotated cameras to a shared GTSAM graph.

The three camera states are independent of programme camera mounts.  All
observed native pixels stay as annotated.  Only explicitly named circular
markers receive shared centre/radius increments. Advertisements optionally
share the programme's AdGeometry states; other world geometry stays fixed.
The original camera Brown coefficients remain in their 25-component templates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys

from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import numpy as np

from ice_refine import _curve_error, _curve_vector_error, finite_curve_distance
from projection import from_state, project_fast, to_state
from robust_observations import pointwise_huber
from shared_markers import RingPixels, add_shared_ring_factor, ring_world


ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / "calibration/deliverables/fit.json"
ORIGINAL = ROOT / "calibration/inputs/step02_observations.json"
ORIGINAL_PTS = (56218880, 104693760, 205992960)
SHARED_CIRCLES = frozenset(("R3", "R4", "F2", "C0"))
# Graph camera states scale world position by 1/100 and principal point/skew
# by 1/1000.  These are deliberately weak prior *standard deviations*.
DEFAULT_CAMERA_PRIOR_SIGMA = np.array(
    (.10, .10, .03, .10, .10, .10, .30, .30, .30, .30, .30), dtype=float)


@dataclass(frozen=True)
class OriginalFactor:
    source_id: str
    kind: str
    keys: tuple[int, ...]
    residual: object = field(repr=False)
    jacobians: object = field(repr=False)


@dataclass(frozen=True)
class OriginalObservation:
    source_id: str
    kind: str
    frame_pts: int
    uv: np.ndarray = field(repr=False)
    world0: np.ndarray = field(repr=False)
    sigma: np.ndarray = field(repr=False)
    group: RingPixels | None = field(default=None, repr=False)
    ad_number: int | None = None
    ad_role: str | None = None
    shared_ad_key: int | None = None


@dataclass(frozen=True)
class OriginalCalibration:
    camera_templates: dict[int, np.ndarray]
    camera_initial_states: dict[int, np.ndarray]
    camera_keys: dict[int, int]
    original_groups: tuple[RingPixels, ...]
    diagnostics: dict
    factor_specs: tuple[OriginalFactor, ...] = field(repr=False)
    feature_keys: dict[str, int] = field(repr=False)
    observations: tuple[OriginalObservation, ...] = field(repr=False)
    ad_geometries: dict[int, object] = field(repr=False)
    ad_keys: dict[int, int] = field(repr=False)
    finite_vector: bool = False

    def evaluate(self, states: Mapping[int, Sequence[float]], *,
                 include_details: bool = False) -> dict:
        """Score every original UV at supplied camera and shared-ring states.

        This is a source reprojection check of fitted inputs, not an independent
        holdout. Curve values use finite segment distances when enabled,
        otherwise absolute signed distances; ad values are Euclidean native-
        pixel corner errors. No factors are added.
        """
        scores = []
        frame_values = {pts: {"curves_px": [], "ads_px": []}
                        for pts in self.camera_keys}
        for obs in self.observations:
            pts = obs.frame_pts
            q = np.asarray(states[self.camera_keys[pts]], float).reshape(-1)
            if q.shape != (11,) or not np.isfinite(q).all():
                raise ValueError(f"Missing finite original camera state at {pts}")
            camera = from_state(q, self.camera_templates[pts])
            if obs.kind == "advertisement":
                world = obs.world0
                if obs.shared_ad_key is not None:
                    ad_state = np.asarray(states[obs.shared_ad_key], float).reshape(-1)
                    world = np.asarray(self.ad_geometries[obs.ad_number].corner(
                        ad_state, obs.ad_role), float).reshape(1, 3)
                predicted, depth = project_fast(camera, world)
                if depth[0] <= 0 or not np.isfinite(predicted).all():
                    raise ValueError(f"Invalid original ad projection: {obs.source_id}")
                error = np.linalg.norm(predicted-obs.uv, axis=1)
                frame_values[pts]["ads_px"].extend(error.tolist())
            else:
                world = obs.world0
                if obs.group is not None:
                    key = self.feature_keys[obs.group.feature_id]
                    feature = np.asarray(states[key], float).reshape(-1)
                    if feature.shape != (11,) or not np.isfinite(feature).all():
                        raise ValueError(f"Missing finite shared marker state {key}")
                    world = ring_world(obs.group, feature)
                error = (finite_curve_distance(camera, world, obs.uv)
                         if self.finite_vector else
                         np.abs(_curve_error(camera, world, obs.uv)))
                frame_values[pts]["curves_px"].extend(error.tolist())
            if not np.isfinite(error).all():
                raise ValueError(f"Nonfinite original source residual: {obs.source_id}")
            row = {"source_id": obs.source_id, "frame_pts": pts,
                   "kind": obs.kind,
                   "feature_id": obs.group.feature_id if obs.group else None,
                   "ad_number": obs.ad_number,
                   "ad_role": obs.ad_role,
                   "shared_ad_key": obs.shared_ad_key,
                   **_stats_px(error)}
            if include_details:
                row["per_point_error_px"] = error.tolist()
            scores.append(row)
        summary = {
            str(pts): {name: _stats_px(frame_values[pts][name])
                       for name in ("curves_px", "ads_px")}
            for pts in self.camera_keys}
        return {
            "scope": "original_source_fit_pixels_not_independent_holdout",
            "curve_error_metric": ("finite_curve_distance_px" if self.finite_vector
                                   else "absolute_signed_supporting_line_distance_px"),
            "per_frame": summary,
            "group_count": len(scores),
            "worst_groups_by_p95": sorted(scores,
                key=lambda row: row["p95_px"], reverse=True)[:12],
            **({"by_group": scores} if include_details else {}),
        }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigma_vector(value: float | Sequence[float], count: int, label: str) -> np.ndarray:
    sigma = np.asarray(value, float)
    sigma = (np.full(count, float(sigma)) if sigma.ndim == 0 else sigma.reshape(-1))
    if sigma.shape != (count,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError(f"Invalid native-pixel sigma for {label}")
    return sigma.copy()


def _stats_px(values: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or nonfinite source errors")
    return {"count": int(len(values)), "median_px": float(np.median(values)),
            "p95_px": float(np.percentile(values, 95)),
            "max_px": float(np.max(values))}


def _feature_meta(scene: dict, name: str, role: str) -> tuple[np.ndarray, float, float]:
    feature = next((f for f in scene["ice_features"] if f["id"] == name), None)
    if feature is None or feature.get("kind") != "circle":
        raise ValueError(f"Shared original marker {name} is not a scene circle")
    rings = {r["role"]: r for r in feature.get("rings", [])}
    if rings:
        if role not in rings or "outer" not in rings or "middle" not in rings:
            raise ValueError(f"Missing ring definition for {name}:{role}")
        return (np.asarray(feature["center_xyz"][:2], float),
                float(rings["outer"]["radius"]), float(rings["middle"]["radius"]))
    if name != "C0" or role != "outer":
        raise ValueError(f"Only C0 may use a single-circle marker: {name}:{role}")
    # C0 has one painted circle and no middle ring.  Its physical guard may
    # use zero as the absent middle radius; no fictional middle pixels exist.
    return np.asarray(feature["center_xyz"][:2], float), float(feature["radius"]), 0.


def _attach_curve(graph, frame_key: int, template: np.ndarray, world: np.ndarray,
                  uv: np.ndarray, sigma: np.ndarray, source_id: str,
                  *, finite_vector: bool = False, source_normal_precision=None) -> OriginalFactor:
    curve = _curve_vector_error if finite_vector else _curve_error
    source_sigma = np.repeat(sigma, 2) if finite_vector else sigma

    def raw(states):
        return curve(from_state(states[0], template), world, uv) / source_sigma

    def jac(states):
        camera = from_state(states[0], template)
        return [curve(camera, world, uv, True)[1] / source_sigma[:, None]]

    if source_normal_precision is not None:
        if not finite_vector:
            raise ValueError('Original source normal precision requires finite vectors')
        from source_precision import SourcePixelPrecision
        raw, jac = SourcePixelPrecision.wrap_curve(raw, jac, sigma, *source_normal_precision)
    residual, derivative = pointwise_huber(
        raw, jac, group_size=2 if finite_vector else 1, delta=2.)
    graph.add_observation([frame_key], residual,
                          residual_dim=len(uv) * (2 if finite_vector else 1),
                          sigma=1., jac_fn=derivative)
    return OriginalFactor(source_id, "fixed_curve", (frame_key,), residual, derivative)


def _attach_ad(graph, frame_key: int, template: np.ndarray, world: np.ndarray,
               uv: np.ndarray, sigma: float, source_id: str) -> OriginalFactor:
    sigma_uv = np.broadcast_to(np.asarray(sigma, float), (2,)).copy()
    def raw(states):
        return (project_fast(from_state(states[0], template), world)[0][0] - uv) / sigma_uv

    def jac(states):
        return [project_fast(from_state(states[0], template), world, True)[2] / sigma_uv[:, None]]

    residual, derivative = pointwise_huber(raw, jac, group_size=2, delta=2.)
    graph.add_observation([frame_key], residual, residual_dim=2,
                          sigma=1., jac_fn=derivative)
    return OriginalFactor(source_id, "fixed_ad_corner", (frame_key,), residual, derivative)


def add_original_calibration(
    graph,
    scene: dict,
    feature_keys: Mapping[str, int],
    guesses: Mapping[int, Sequence[float]] | None = None,
    *,
    fit_path: Path = FIT,
    observation_path: Path = ORIGINAL,
    camera_prior_sigma: Sequence[float] | None = None,
    ad_sigma_px: float = 3.,
    ad_geometries: Mapping[int, object] | None = None,
    ad_keys: Mapping[int, int] | None = None,
    finite_vector: bool = False,
    source_quality=None,
    source_precision=None,
) -> OriginalCalibration:
    """Add independent original cameras, all original pixels, and shared rings.

    ``feature_keys`` maps any subset of R3/R4/F2/C0 to the caller's shared
    11-vector keys. The caller owns each feature's prior and initial state.
    ``guesses`` may override *initial* conditioned 11-vectors for original
    cameras; priors always target the saved three-frame fit. No original
    camera is attached to a programme mount. Original ads have no calibrated
    source sigma; ``ad_sigma_px`` is an explicit modelling choice for them.
    When both ad mappings are supplied, scene-existing roles use the caller's
    programme AdGeometry node, while other original ad corners remain fixed.
    ``finite_vector`` opts into 2D nearest-finite-segment curve residuals.
    """
    if getattr(graph, "state_dim", 11) != 11:
        raise ValueError("Original calibration requires an 11-state graph")
    if set(feature_keys) - SHARED_CIRCLES:
        raise ValueError(f"Unsupported shared marker(s): {sorted(set(feature_keys)-SHARED_CIRCLES)}")
    if (len(set(feature_keys.values())) != len(feature_keys)
            or set(feature_keys.values()) & set(ORIGINAL_PTS)):
        raise ValueError("Shared marker keys must be distinct from original camera keys")
    if (ad_geometries is None) != (ad_keys is None):
        raise ValueError("ad_geometries and ad_keys must be supplied together")
    ad_geometries = {int(number): model for number, model in (ad_geometries or {}).items()}
    ad_keys = {int(number): int(key) for number, key in (ad_keys or {}).items()}
    if ad_geometries.keys() != ad_keys.keys():
        raise ValueError("ad_geometries and ad_keys must have matching user numbers")
    if (len(set(ad_keys.values())) != len(ad_keys)
            or set(ad_keys.values()) & (set(ORIGINAL_PTS) | set(feature_keys.values()))):
        raise ValueError("Shared ad keys must be distinct from original camera and marker keys")
    for number, model in ad_geometries.items():
        if (int(getattr(model, "user_number", -1)) != number
                or not callable(getattr(model, "factor", None))
                or not callable(getattr(model, "corner", None))):
            raise ValueError(f"Invalid shared AdGeometry for user number {number}")
    if not np.isfinite(ad_sigma_px) or ad_sigma_px <= 0:
        raise ValueError("ad_sigma_px must be finite and positive")
    prior_sigma = _sigma_vector(
        DEFAULT_CAMERA_PRIOR_SIGMA if camera_prior_sigma is None else camera_prior_sigma,
        11, "original camera prior")
    fit_path, observation_path = Path(fit_path), Path(observation_path)
    fit = json.loads(fit_path.read_text(encoding="utf8"))
    source_hash, fit_hash = _hash(observation_path), _hash(fit_path)
    if (fit.get("observation_sha256") != source_hash
            or scene.get("source", {}).get("selected_fit_sha256") != fit_hash
            or scene.get("source", {}).get("observation_sha256") != source_hash):
        raise ValueError("Original fit, manual observations, and shared scene differ")
    work = ROOT / "calibration/work"
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    from fit_joint import problem_for_fit  # imported only when this adapter is used

    problem = problem_for_fit(fit)
    x = np.asarray(fit["parameter_values"], float)
    if (problem.names != fit["parameter_names"]
            or problem.input_sha256 != source_hash
            or not np.isfinite(x).all()):
        raise ValueError("Original three-frame Problem differs from the saved fit")
    geometry = problem.geometry(x)
    pts_values = {int(row["frame_pts"]) for row in problem.fit_obs}
    pts_values.update(int(row["pts"]) for row in problem.ad_obs)
    if pts_values != set(ORIGINAL_PTS):
        raise ValueError(f"Unexpected original source frames: {sorted(pts_values)}")
    if guesses is not None and set(guesses) - pts_values:
        raise ValueError("Original camera guess has an unknown PTS")

    templates, initial_states = {}, {}
    for pts in ORIGINAL_PTS:
        template = np.asarray(x[problem.slices[("camera", pts)]], float).copy()
        if template.shape != (25,) or not np.isfinite(template).all():
            raise ValueError(f"Invalid original camera template at {pts}")
        prior = to_state(template)
        initial = np.asarray(guesses[pts], float).reshape(-1).copy() if guesses and pts in guesses else prior.copy()
        if initial.shape != (11,) or not np.isfinite(initial).all():
            raise ValueError(f"Original camera guess at {pts} must be a finite 11-vector")
        graph.add_prior(pts, prior, prior_sigma)
        templates[pts], initial_states[pts] = template, initial

    groups: list[RingPixels] = []
    factors: list[OriginalFactor] = []
    observations: list[OriginalObservation] = []
    by_frame = defaultdict(lambda: Counter())
    baseline = defaultdict(lambda: {"curves_px": [], "ads_px": []})
    for obs in problem.fit_obs:
        pts = int(obs["frame_pts"])
        uv = np.asarray(obs["uv"], float).copy()
        sigma = _sigma_vector(obs["sigma_px"], len(uv), obs["id"])
        if source_quality is not None:
            multipliers = source_quality.scales(pts, uv, 'original_curve')
            if source_precision is None:
                sigma = sigma*multipliers
        physical_id = obs['physical_group_id']
        if physical_id == 'board_contact':
            physical_id = 'ice_board_contact'
        normal_precision = (source_precision.curve(pts, physical_id, uv, sigma)
                            if source_precision is not None else None)
        world = np.asarray(problem.curve_world(geometry, obs), float).copy()
        if uv.ndim != 2 or uv.shape[1] != 2 or world.ndim != 2 or world.shape[1] != 3 or len(world) < 2:
            raise ValueError(f"Malformed original source curve {obs['id']}")
        if not np.isfinite(uv).all() or not np.isfinite(world).all():
            raise ValueError(f"Nonfinite original source curve {obs['id']}")
        name = obs["target"].get("feature_id")
        if name in feature_keys:
            if obs["target"].get("kind") != "circle_outline":
                raise ValueError(f"Shared marker {name} is not a circular source curve")
            role = obs["target"].get("ring_role") or "outer"
            center, outer, middle = _feature_meta(scene, name, role)
            group = RingPixels(name, role, pts, "original3", obs["id"],
                               str(observation_path), templates[pts], world,
                               center, outer, middle, uv, sigma,
                               float(np.sum(np.asarray(obs["sqrt_weights"], float)**2)))
            add_shared_ring_factor(graph, pts, int(feature_keys[name]), templates[pts],
                                   group, finite_vector=finite_vector,
                                   source_normal_precision=normal_precision)
            groups.append(group)
            observations.append(OriginalObservation(obs["id"], "shared_curve", pts,
                                                    uv, world, sigma, group))
            by_frame[pts]["shared_curve_groups"] += 1
        else:
            factors.append(_attach_curve(graph, pts, templates[pts], world, uv,
                                         sigma, obs["id"], finite_vector=finite_vector,
                                         source_normal_precision=normal_precision))
            observations.append(OriginalObservation(obs["id"], "fixed_curve", pts,
                                                    uv, world, sigma))
            by_frame[pts]["fixed_curve_groups"] += 1
        by_frame[pts]["curve_pixels"] += len(uv)
        baseline_curve = (finite_curve_distance(templates[pts], world, uv)
                          if finite_vector else np.abs(_curve_error(templates[pts], world, uv)))
        baseline[pts]["curves_px"].extend(baseline_curve.tolist())

    frame_pts = {frame["id"]: int(frame["pts"]) for frame in problem.input["frames"]}
    original_ads = {}
    for obs in problem.current_observations:
        if obs["category"] != "ads":
            continue
        target = obs["target"]
        key = (frame_pts[obs["frame_id"]], int(target["user_ad_number"]), target["corner_role"])
        if key in original_ads:
            raise ValueError(f"Duplicate original advertisement corner {key}")
        original_ads[key] = obs
    missing_ad_sigma = 0
    shared_ad_corners = 0
    fixed_ad_corners = 0
    for row in problem.ad_obs:
        pts, number, role = int(row["pts"]), int(row["number"]), row["role"]
        source = original_ads.pop((pts, number, role))
        uv = np.asarray(row["uv"], float).copy()
        source_uv = np.asarray(source["pixels_uv"], float).reshape(1, 2)[0]
        if not np.array_equal(uv, source_uv):
            raise ValueError(f"Original advertisement UV changed: {pts}:{number}:{role}")
        reported_sigma = source.get("uncertainty", {}).get("sigma_px")
        if reported_sigma is None:
            missing_ad_sigma += 1
        sigma = float(ad_sigma_px if reported_sigma is None else reported_sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid original advertisement sigma at {pts}:{number}:{role}")
        if source_quality is not None:
            multiplier = source_quality.scales(pts, [uv], 'original_ad_corner')[0]
            if source_precision is None:
                sigma *= multiplier
        if source_precision is not None:
            sigma = source_precision.ad_sigma(pts, number, role, uv, sigma)
        world = np.asarray(problem.ad_world(x, geometry, number, role), float).reshape(1, 3).copy()
        shared_key = None
        if number in ad_geometries and role in ad_geometries[number].roles:
            shared_key = ad_keys[number]
            ad_geometries[number].factor(
                graph, pts, shared_key, templates[pts], role, uv, sigma_px=sigma)
            shared_ad_corners += 1
            by_frame[pts]["shared_ad_corners"] += 1
        else:
            factors.append(_attach_ad(graph, pts, templates[pts], world, uv,
                                      sigma, source["id"]))
            fixed_ad_corners += 1
            by_frame[pts]["fixed_ad_corners"] += 1
        observations.append(OriginalObservation(source["id"], "advertisement", pts,
                                                uv.reshape(1, 2), world,
                                                np.asarray([sigma]),
                                                ad_number=number, ad_role=role,
                                                shared_ad_key=shared_key))
        by_frame[pts]["advertisement_corners"] += 1
        baseline[pts]["ads_px"].append(float(np.linalg.norm(
            project_fast(templates[pts], world)[0][0] - uv)))
    if original_ads:
        raise ValueError(f"Unmatched original advertisement corners: {len(original_ads)}")

    diagnostics = {
        "fit_sha256": fit_hash, "original_observations_sha256": source_hash,
        "camera_policy": "three independent original camera states; no programme mount equality",
        "camera_prior_target": "saved original three-frame fit, not initialization guesses",
        "camera_prior_sigma_conditioned": prior_sigma.tolist(),
        "brown_policy": "all 14 template coefficients fixed at the saved original fit values",
        "geometry_policy": "feature_keys circles share centre/radius states; mapped scene-existing ad roles share programme AdGeometry states; all other original world geometry fixed",
        "source_weighting": ("original curve sigma_px repeated per XY component, Huber-2 per 2D source point"
                             if finite_vector else
                             "original curve sigma_px per pixel, Huber-2 per scalar distance")
                            + "; fixed ads 2D Huber-2 per corner; shared ads use AdGeometry.factor 2D Huber-3 per corner; no original group-budget or 1/N attenuation",
        "curve_error_metric": ("finite_curve_distance_px" if finite_vector
                               else "absolute_signed_supporting_line_distance_px"),
        "original_ad_sigma_missing_count": missing_ad_sigma,
        "ad_sigma_fallback_px": float(ad_sigma_px),
        "source_pixel_precision": source_precision.report() if source_precision is not None else None,
        "original_ad_shared_corners": shared_ad_corners,
        "original_ad_fixed_corners": fixed_ad_corners,
        "shared_ad_user_numbers": sorted({obs.ad_number for obs in observations
                                          if obs.shared_ad_key is not None}),
        "shared_features": sorted(feature_keys),
        "per_frame": {str(pts): dict(by_frame[pts]) for pts in ORIGINAL_PTS},
        "saved_fit_baseline_px": {
            str(pts): {kind: _stats_px(baseline[pts][kind])
                       for kind in ("curves_px", "ads_px")}
            for pts in ORIGINAL_PTS},
    }
    return OriginalCalibration(templates, initial_states,
                               {pts: pts for pts in ORIGINAL_PTS},
                               tuple(groups), diagnostics, tuple(factors),
                               {name: int(key) for name, key in feature_keys.items()},
                               tuple(observations), ad_geometries, ad_keys,
                               bool(finite_vector))
