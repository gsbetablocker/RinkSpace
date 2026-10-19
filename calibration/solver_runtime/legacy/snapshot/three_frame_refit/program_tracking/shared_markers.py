"""Source-backed shared painted-ring factors and a fixed-camera diagnostic.

The reusable graph API uses one 11-vector per physical ring: conditioned
dx/100, dy/100 and outer-radius delta/100, with eight fixed trailing values.
The diagnostic changes no released geometry, camera or source observation.
"""

from __future__ import annotations
from calibration.solver_runtime.config import data_module_dir as _runtime_data_module_dir

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2
import gtsam
import numpy as np

from ice_refine import _curve_error, _curve_vector_error, finite_curve_distance, source_frame
from projection import from_state, project_fast
from render_result import clip_segments
from robust_observations import pointwise_huber


ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
SCENE = ROOT / "three_frame_refit/deliverables/scene.json"
FIT = ROOT / "three_frame_refit/deliverables/fit.json"
ORIGINAL = ROOT / "three_frame_refit/inputs/step02_observations.json"
SHOTS = _runtime_data_module_dir(__file__) / 'source_shots.json'
FEATURES = ("R3", "R4")


@dataclass(frozen=True)
class RingPixels:
    feature_id: str
    role: str
    pts: int
    source_kind: str
    source_id: str
    source_file: str
    camera: np.ndarray
    world0: np.ndarray
    center_xy: np.ndarray
    outer_radius: float
    middle_radius: float
    uv: np.ndarray
    sigma: np.ndarray
    original_quadrature_sum: float | None = None


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ring_world(group: RingPixels, feature11: np.ndarray) -> np.ndarray:
    """Move every ring's centre together; change only the outer radius."""
    q = np.asarray(feature11, float).reshape(11)
    result = group.world0.copy()
    result[:, :2] += q[:2] * 100.
    if group.role == "outer":
        radial = (group.world0[:, :2] - group.center_xy) / group.outer_radius
        result[:, :2] += (100. * q[2]) * radial
    return result


def _raw(camera: np.ndarray, group: RingPixels, feature11: np.ndarray,
         finite_vector: bool = False) -> np.ndarray:
    curve = _curve_vector_error if finite_vector else _curve_error
    sigma = np.repeat(group.sigma, 2) if finite_vector else group.sigma
    return curve(camera, ring_world(group, feature11), group.uv) / sigma


def _feature_jac(camera: np.ndarray, group: RingPixels,
                 feature11: np.ndarray, finite_vector: bool = False) -> np.ndarray:
    result = np.zeros((len(group.uv) * (2 if finite_vector else 1), 11), float)
    step = 1e-5  # 0.001 conditional scene units per centre/radius step.
    for axis in range(3):
        high = feature11.copy(); low = feature11.copy()
        high[axis] += step; low[axis] -= step
        result[:, axis] = (_raw(camera, group, high, finite_vector)
                           - _raw(camera, group, low, finite_vector)) / (2*step)
    return result


def add_shared_ring_prior(graph, feature_key: int, outer_radius: float,
                          middle_radius: float, *, allow_outer_radius: bool = True,
                          center_sigma_scene: float = 1.,
                          radius_sigma_scene: float = .3) -> None:
    """Add a conditioned 11-state prior and the outer>middle guard.

    `graph` is the program's IncrementalCameraGraph; caller adds one state
    per feature and passes the same `feature_key` on every source frame.
    """
    if outer_radius <= middle_radius or center_sigma_scene <= 0 or radius_sigma_scene <= 0:
        raise ValueError("Invalid ring geometry or prior")
    sigma = np.r_[[center_sigma_scene/100.]*2,
                  [radius_sigma_scene/100. if allow_outer_radius else 1e-9],
                  np.full(8, 1e-9)]
    graph.add_prior(feature_key, np.zeros(11), sigma)

    def guard(states):
        radius = outer_radius + 100.*states[0][2]
        return np.array([max(0., middle_radius + .05 - radius)])

    def derivative(states):
        jac = np.zeros((1, 11))
        if outer_radius + 100.*states[0][2] < middle_radius + .05:
            jac[0, 2] = -100.
        return [jac]

    graph.add_observation([feature_key], guard, residual_dim=1,
                          sigma=.001, jac_fn=derivative)


def add_shared_ring_factor(graph, camera_key: int, feature_key: int,
                           camera_template: np.ndarray,
                           group: RingPixels, *, finite_vector: bool = False,
                           source_normal_precision=None) -> None:
    """Attach actual pixels to camera+shared feature in an 11-state graph.

    Camera Jacobian is analytic. Feature's three active columns use central
    differences; source UV remains immutable. Every pixel has its source sigma
    and its own Huber-2 residual. ``finite_vector`` uses the true nearest
    finite-segment 2D vector and a per-pixel two-component Huber group.
    No whole-group division by point count.
    """
    template = np.asarray(camera_template, float).copy()
    curve = _curve_vector_error if finite_vector else _curve_error
    sigma = np.repeat(group.sigma, 2) if finite_vector else group.sigma

    def raw(states):
        return _raw(from_state(states[0], template), group, states[1], finite_vector)

    def jac(states):
        camera = from_state(states[0], template)
        world = ring_world(group, states[1])
        camera_jac = curve(camera, world, group.uv, True)[1] / sigma[:, None]
        return [camera_jac, _feature_jac(camera, group, states[1], finite_vector)]

    weighted_raw, weighted_jac = raw, jac
    if source_normal_precision is not None:
        if not finite_vector:
            raise ValueError('Directional source precision requires finite curve vectors')
        from source_precision import SourcePixelPrecision
        effective, normals = source_normal_precision
        weighted_raw, weighted_jac = SourcePixelPrecision.wrap_curve(
            raw, jac, group.sigma, effective, normals)
    robust, derivative = pointwise_huber(
        weighted_raw, weighted_jac, group_size=2 if finite_vector else 1, delta=2.)
    graph.add_observation([camera_key, feature_key], robust,
                          residual_dim=len(group.uv) * (2 if finite_vector else 1),
                          sigma=1., jac_fn=derivative)


def _feature(scene: dict, name: str) -> dict:
    return next(f for f in scene["ice_features"] if f["id"] == name)


def _ring_meta(feature: dict, role: str) -> tuple[np.ndarray, float, float]:
    rings = {r["role"]: r for r in feature["rings"]}
    ring = rings[role]
    return (np.asarray(ring["center_xyz"][:2], float),
            float(rings["outer"]["radius"]), float(rings["middle"]["radius"]))


def load_groups(candidate: Path, config_path: Path) -> tuple[list[RingPixels], dict]:
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    diagnostic = json.loads((candidate / "diagnostics.json").read_text(encoding="utf-8"))
    if (scene["source"]["selected_fit_sha256"] != _hash(FIT)
            or scene["source"]["observation_sha256"] != _hash(ORIGINAL)
            or fit["observation_sha256"] != _hash(ORIGINAL)
            or diagnostic["model_sha256"] != _hash(SCENE)
            or diagnostic["configuration_sha256"] != _hash(config_path)
            or diagnostic["source_sha256"] != shots["source_sha256_from_timeline"]):
        raise ValueError("The frozen fit, scene, source or v3 configuration changed")
    work = ROOT / "three_frame_refit/work"
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    from fit_joint import problem_for_fit  # noqa: E402

    problem = problem_for_fit(fit)
    x = np.asarray(fit["parameter_values"], float)
    if problem.names != fit["parameter_names"] or problem.input_sha256 != _hash(ORIGINAL):
        raise ValueError("Original three-frame Problem changed")
    geometry = problem.geometry(x)
    groups: list[RingPixels] = []
    for obs in problem.fit_obs:
        name = obs["target"].get("feature_id")
        if name not in FEATURES:
            continue
        role = obs["target"].get("ring_role", "outer")
        pts = int(obs["frame_pts"])
        camera = x[problem.slices[("camera", pts)]].copy()
        base_world = problem.curve_world(geometry, obs).copy()
        center, outer_radius, middle_radius = _ring_meta(_feature(scene, name), role)
        uv = np.asarray(obs["uv"], float).copy()
        sigma = np.full(len(uv), float(obs["sigma_px"]))
        groups.append(RingPixels(name, role, pts, "original3", obs["id"],
                                 str(ORIGINAL), camera, base_world, center,
                                 outer_radius, middle_radius, uv, sigma,
                                 float(np.sum(np.asarray(obs["sqrt_weights"])**2))))
    if {g.feature_id for g in groups} != set(FEATURES):
        raise ValueError("Original manual ring source groups are missing")

    # Use only the reviewed source files already configured in the v3 graph.
    # The v3 candidate cameras are read as fixed values, never optimized here.
    seen: set[tuple[int, str]] = set()
    for shot in config["shots"]:
        index = int(shot["shot_index"])
        camera_path = candidate / f"shot{index:02d}_keyframes.npz"
        with np.load(camera_path, allow_pickle=False) as data:
            if (str(data["model_sha256"]) != _hash(SCENE)
                    or str(data["source_sha256"]) != diagnostic["source_sha256"]):
                raise ValueError(f"Fixed camera sidecar differs: {camera_path}")
            cameras = {int(pts): camera.copy() for pts, camera in
                       zip(data["pts"], data["camera25"])}
        for spec in shot["curves"]:
            allowed = set(spec["ids"]) & {f"{name}_outer" for name in FEATURES}
            if not allowed:
                continue
            for pts, camera in cameras.items():
                path = Path(spec.get("overrides", {}).get(str(pts),
                            str(Path(spec["directory"]) / f"{pts}.npz")))
                if not path.exists():
                    continue
                expected = diagnostic["observation_input_sha256"].get(str(path.resolve()))
                if expected != _hash(path):
                    raise ValueError(f"Programme source file differs from v3 graph: {path}")
                with np.load(path, allow_pickle=False) as data:
                    if int(data["pts"]) != pts or str(data["source"]) != "original_decoded_mp4_native_pts":
                        raise ValueError(f"Not the original native source: {path}")
                    if "source_sha256" in data and str(data["source_sha256"]) != diagnostic["source_sha256"]:
                        raise ValueError(f"Source hash differs: {path}")
                    if "model_sha256" in data and str(data["model_sha256"]) != _hash(SCENE):
                        raise ValueError(f"Scene hash differs: {path}")
                    names = data["physical_id"].astype(str)
                    for name in sorted(allowed & set(names)):
                        selected = names == name
                        if selected.sum() < 6:
                            continue
                        key = (pts, name)
                        if key in seen:
                            raise ValueError(f"Duplicate source group {key}")
                        seen.add(key)
                        feature_id = name.removesuffix("_outer")
                        center, outer_radius, middle_radius = _ring_meta(
                            _feature(scene, feature_id), "outer")
                        angles = np.linspace(0., 2*np.pi, 145)
                        world = np.column_stack((
                            center[0] + outer_radius*np.cos(angles),
                            center[1] + outer_radius*np.sin(angles),
                            np.zeros(len(angles))))
                        uv = data["observed_uv"][selected].astype(float).copy()
                        sigma = (data["point_sigma_px"][selected].astype(float)
                                 if "point_sigma_px" in data else
                                 np.full(len(uv), float(config.get("marking_sigma_px", 2.))))
                        groups.append(RingPixels(feature_id, "outer", pts,
                                                 "programme", name, str(path),
                                                 camera, world, center, outer_radius,
                                                 middle_radius, uv, sigma))
    programme = [g for g in groups if g.source_kind == "programme"]
    if len(programme) < 5 or {g.feature_id for g in programme} != set(FEATURES):
        raise ValueError("Reviewed R3/R4 programme coverage is incomplete")
    provenance = {"scene_sha256": _hash(SCENE), "fit_sha256": _hash(FIT),
                  "original_observations_sha256": _hash(ORIGINAL),
                  "programme_diagnostics_sha256": _hash(candidate / "diagnostics.json"),
                  "programme_config_sha256": _hash(config_path),
                  "source_video_sha256": diagnostic["source_sha256"],
                  "programme_groups": len(programme),
                  "original_groups": len(groups) - len(programme),
                  "weighting": "each actual source pixel divided only by its source sigma; Huber-2 per pixel; no /N or legacy quadrature group attenuation",
                  "camera_policy": "all v3 programme and original-three camera25 fixed"}
    return groups, provenance


def _feature_only_solve(groups: list[RingPixels], *, include_original: bool,
                        allow_outer_radius: bool, start: dict[str, np.ndarray] | None = None
                        ) -> tuple[dict[str, np.ndarray], dict]:
    selected = [g for g in groups if include_original or g.source_kind == "programme"]
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    callbacks = []
    keys = {name: gtsam.symbol("r", index) for index, name in enumerate(FEATURES, 3)}
    for name in FEATURES:
        first = next(g for g in groups if g.feature_id == name)
        key = keys[name]
        q0 = np.zeros(11) if start is None else start[name].copy()
        if not allow_outer_radius:
            q0[2] = 0.
        initial.insert(key, q0)
        sigma = np.r_[[.01, .01], [.003 if allow_outer_radius else 1e-9],
                      np.full(8, 1e-9)]
        graph.add(gtsam.PriorFactorVector(key, np.zeros(11),
                  gtsam.noiseModel.Diagonal.Sigmas(sigma)))

        def guard(_factor, values, jacobians, key=key, first=first):
            q = np.asarray(values.atVector(key), float)
            radius = first.outer_radius + 100.*q[2]
            violation = max(0., first.middle_radius + .05 - radius)
            if jacobians is not None:
                jac = np.zeros((1, 11))
                if violation > 0:
                    jac[0, 2] = -100.
                jacobians[0] = np.asfortranarray(jac)
            return np.array([violation])
        callbacks.append(guard)
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, .001),
                                     [key], guard))
    for group in selected:
        key = keys[group.feature_id]
        def raw(states, group=group):
            return _raw(group.camera, group, states[0])
        def jac(states, group=group):
            return [_feature_jac(group.camera, group, states[0])]
        robust, robust_jac = pointwise_huber(raw, jac, group_size=1, delta=2.)
        def evaluate(_factor, values, jacobians, key=key,
                     robust=robust, robust_jac=robust_jac):
            q = np.asarray(values.atVector(key), float)
            if jacobians is not None:
                jacobians[0] = np.asfortranarray(robust_jac([q])[0])
            return robust([q])
        callbacks.append(evaluate)
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(len(group.uv), 1.),
                                     [key], evaluate))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(80)
    params.setRelativeErrorTol(1e-9)
    solved = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
    state = {name: np.asarray(solved.atVector(key), float).copy()
             for name, key in keys.items()}
    return state, {"graph_error_before": round(float(graph.error(initial)), 4),
                   "graph_error_after": round(float(graph.error(solved)), 4),
                   "factors": graph.size(), "source_groups": len(selected)}


def _score(group: RingPixels, state: np.ndarray) -> dict:
    world = ring_world(group, state)
    error = finite_curve_distance(group.camera, world, group.uv)
    return {"n": len(error), "median_px": round(float(np.median(error)), 2),
            "p95_px": round(float(np.percentile(error, 95)), 2),
            "max_px": round(float(np.max(error)), 2)}


def _scores(groups: list[RingPixels], state: dict[str, np.ndarray]) -> dict:
    by_group = {f"{g.source_kind}:{g.pts}:{g.source_id}": _score(g, state[g.feature_id])
                for g in groups}
    aggregate = {}
    for name in FEATURES:
        for source in ("programme", "original3"):
            subset = [g for g in groups if g.feature_id == name and
                      g.source_kind == source and g.role == "outer"]
            if not subset:
                continue
            errors = np.concatenate([finite_curve_distance(
                g.camera, ring_world(g, state[name]), g.uv) for g in subset])
            aggregate[f"{name}_{source}_outer"] = {
                "n": len(errors), "median_px": round(float(np.median(errors)), 2),
                "p95_px": round(float(np.percentile(errors, 95)), 2),
                "max_px": round(float(np.max(errors)), 2)}
    return {"aggregate": aggregate, "by_group": by_group}


def _state_summary(state: dict[str, np.ndarray], groups: list[RingPixels]) -> dict:
    result = {}
    for name in FEATURES:
        group = next(g for g in groups if g.feature_id == name)
        delta = state[name][:3]*100.
        result[name] = {"delta_center_xy_scene": np.round(delta[:2], 5).tolist(),
                        "delta_outer_radius_scene": round(float(delta[2]), 5),
                        "outer_radius_after": round(float(group.outer_radius+delta[2]), 5),
                        "middle_radius": round(group.middle_radius, 5),
                        "prior_norm_sigma": round(float(np.linalg.norm(delta / [1., 1., .3])), 3)}
    return result


def _draw_path(image: np.ndarray, camera: np.ndarray, world: np.ndarray,
               color: tuple[int, int, int]) -> None:
    uv, depth = project_fast(camera, world)
    good = (np.isfinite(uv[:-1]).all(axis=1) & np.isfinite(uv[1:]).all(axis=1)
            & (depth[:-1] > .05) & (depth[1:] > .05))
    if good.any():
        a, b, _ = clip_segments(uv[:-1][good], uv[1:][good], 1920, 1080)
        if len(a):
            cv2.polylines(image, np.rint(np.stack((a, b), axis=1)).astype(np.int32),
                          False, color, 1, cv2.LINE_AA)


def _draw(groups: list[RingPixels], before: dict[str, np.ndarray],
          programme_only: dict[str, np.ndarray], joint: dict[str, np.ndarray],
          output: Path) -> None:
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    views = [(229666560, Path(shots["source_path"]), "programme"),
             (56218880, ROOT / "outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/survey/survey_3660.png", "original3")]
    for pts, source, kind in views:
        if kind == "programme":
            image = source_frame(source, pts)
        else:
            image = cv2.imread(str(source))
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError("Original high-view source image unavailable")
        chosen = [g for g in groups if g.pts == pts and g.role == "outer"]
        for group in chosen:
            name = group.feature_id
            for state, color in ((before, (40, 60, 225)),
                                 (programme_only, (0, 225, 255)),
                                 (joint, (40, 240, 60))):
                _draw_path(image, group.camera, ring_world(group, state[name]), color)
            for x, y in group.uv:
                cv2.circle(image, (round(float(x)), round(float(y))), 2,
                           (255, 0, 255), -1, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (1460, 67), (20, 20, 20), -1)
        cv2.putText(image, f"Original source PTS {pts} | shared R3/R4 geometry test",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "source magenta dots | released red | programme-only yellow | original+programme green | all cameras fixed",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
        path = output / f"{pts}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, default=TMP / "joint_programme_correlated_v3")
    p.add_argument("--config", type=Path, default=TMP / "joint_programme_correlated_v3_config.json")
    p.add_argument("--output", type=Path, default=TMP / "shared_markers_v3")
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(TMP.resolve()):
        raise ValueError("Diagnostic output must stay in temporary storage")
    args.output.mkdir(parents=True, exist_ok=True)
    groups, provenance = load_groups(args.candidate, args.config)
    zero = {name: np.zeros(11) for name in FEATURES}
    trials = {}
    center_programme_start = None
    for allow_radius in (False, True):
        mode = "center_radius" if allow_radius else "center_only"
        p_state, p_solve = _feature_only_solve(
            groups, include_original=False, allow_outer_radius=allow_radius,
            start=center_programme_start if allow_radius else None)
        if not allow_radius:
            center_programme_start = p_state
        j_state, j_solve = _feature_only_solve(
            groups, include_original=True, allow_outer_radius=allow_radius,
            start=p_state)
        trials[mode] = {
            "programme_only": {"delta": _state_summary(p_state, groups),
                               "solver": p_solve, "scores": _scores(groups, p_state)},
            "original_plus_programme": {"delta": _state_summary(j_state, groups),
                                        "solver": j_solve, "scores": _scores(groups, j_state)}}
        if allow_radius:
            _draw(groups, zero, p_state, j_state, args.output)
    result = {"purpose": "fixed-camera shared physical marker diagnostic; no scene release",
              "provenance": provenance, "baseline": _scores(groups, zero),
              "trials": trials,
              "images": {str(pts): str(args.output / f"{pts}.jpg")
                         for pts in (229666560, 56218880)}}
    path = args.output / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(path),
                      "baseline": result["baseline"]["aggregate"],
                      "trials": {name: {stage: {"delta": part["delta"],
                                                "scores": part["scores"]["aggregate"]}
                                        for stage, part in trial.items()}
                                 for name, trial in trials.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
