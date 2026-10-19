"""Three physical camera rigs, nine independent shot clocks and splines.

This is a routing/evaluation adapter, not a fitted nine-shot solution.  An
uninitialised rig/shot is an explicit error.  Camera25/legacy60 arrays are never
interpreted as native times or as fixed-head calibration parameters.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "rinkfit.joint_program.v1"
SHOT_TO_RIG = {"s01": "main", "s02": "secondary", "s03": "end",
               "s04": "main", "s05": "end", "s06": "secondary",
               "s07": "main", "s08": "end", "s09": "main"}
REFERENCE_SHOTS = {"main": "s01", "secondary": "s02", "end": "s03"}


class GraphError(ValueError):
    pass


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GraphError(f"{name} must be an integer native PTS, not seconds")
    return int(value)


def initial_graph(shot_scopes: Mapping[str, Mapping[str, Any]],
                  source_sha256: str) -> dict[str, Any]:
    """Build only verified identities/scopes.  No camera numbers are invented."""
    if set(shot_scopes) != set(SHOT_TO_RIG):
        raise GraphError("exactly s01..s09 scopes are required")
    rigs = {rid: {"status": "uninitialised", "reference_shot": ref,
                  "shared_head": None, "optics": None,
                  "rolling_shutter": None,
                  "shared_parameters_remain_optimisable": True}
            for rid, ref in REFERENCE_SHOTS.items()}
    shots = {}
    for sid, rid in SHOT_TO_RIG.items():
        scope = copy.deepcopy(dict(shot_scopes[sid]))
        shots[sid] = {"rig_id": rid, "scope": scope,
                      "time_origin_pts": scope["first_pts"],
                      "reference_angles": [0.0, 0.0] if sid == REFERENCE_SHOTS[rid] else None,
                      "camera_controls": None, "status": "uninitialised",
                      "continuity_across_cut": False}
    graph = {"schema": SCHEMA, "source_clock": {
                 "source_video_sha256": source_sha256, "time_base": [1, 12800],
                 "frame_step_pts": 256, "image_size": [1920, 1080],
                 "pts_semantics": "native_frame_representative_time"},
             "coordinate_gauge": {"length_L": 1.0, "ice_z": 0.0,
                 "handedness": "right", "scale_verified": False,
                 "metres_per_L": None},
             "shared_scene": {"status": "awaiting_latest_s01_checkpoint",
                 "event_model_ref": None, "geometry_is_frozen": False},
             "rigs": rigs, "shots": shots,
             "cache_generation": 0,
             "certificates": {"joint_fit_complete": False,
                              "all_source_replay_complete": False}}
    CameraGraph(graph).validate()
    return graph


class CameraGraph:
    """Read-only snapshot: mutations create a new graph and invalidate caches."""
    def __init__(self, graph: Mapping[str, Any]):
        self.graph = copy.deepcopy(dict(graph))
        self._cameras: dict[str, Any] = {}

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraGraph":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, *, require_ready: bool = False) -> dict[str, Any]:
        g = self.graph
        if g.get("schema") != SCHEMA:
            raise GraphError(f"unsupported joint schema: {g.get('schema')}")
        c = g["source_clock"]
        if c.get("time_base") != [1, 12800] or c.get("frame_step_pts") != 256:
            raise GraphError("this programme must use official 50 fps native 1/12800 PTS")
        if c.get("image_size") != [1920, 1080]:
            raise GraphError("native evaluator uses row range 0..1079; resizing needs a new adapter")
        sha = c.get("source_video_sha256", "")
        if len(sha) != 64 or any(v not in "0123456789abcdef" for v in sha):
            raise GraphError("source video SHA-256 is required")
        if set(g["rigs"]) != set(REFERENCE_SHOTS) or set(g["shots"]) != set(SHOT_TO_RIG):
            raise GraphError("expected three physical rigs and nine shot records")
        incomplete = []
        previous_stop = None
        for sid, expected_rig in SHOT_TO_RIG.items():
            s = g["shots"][sid]; scope = s["scope"]
            if s["rig_id"] != expected_rig:
                raise GraphError(f"{sid}: rig assignment changed; revise identity evidence explicitly")
            first = _integer(scope["first_pts"], f"{sid}.first_pts")
            last = _integer(scope["last_pts"], f"{sid}.last_pts")
            stop = _integer(scope["stop_pts_exclusive"], f"{sid}.stop_pts_exclusive")
            if first > last or stop != last + 256 or (last - first) % 256:
                raise GraphError(f"{sid}: invalid inclusive/exclusive native frame interval")
            if previous_stop is not None and first != previous_stop:
                raise GraphError(f"{sid}: missing or overlapping programme exposures at cut")
            previous_stop = stop
            if _integer(s["time_origin_pts"], f"{sid}.time_origin_pts") != first:
                raise GraphError(f"{sid}: this adapter fixes spline t=0 at first native exposure")
            if s.get("continuity_across_cut") is not False:
                raise GraphError(f"{sid}: no spline/flow factor may cross an edit cut")
            if s.get("camera_controls") is None or s.get("reference_angles") is None:
                incomplete.append(sid)
                continue
            camera = self.native_camera(sid)
            allowed_tau = float(g["rigs"][s["rig_id"]].get("tau_bound", .02))
            if not np.isfinite(allowed_tau) or allowed_tau <= 0 or abs(camera.tau) > allowed_tau:
                raise GraphError(f"{sid}: invalid immutable full readout bound")
            half = allowed_tau / 2.0
            duration = (last - first) / 12800.0
            if camera.domain[0] > -half + 1e-12 or camera.domain[1] < duration + half - 1e-12:
                raise GraphError(f"{sid}: cubic support omits first/last rolling shutter rows")
            if not camera.optical_domain_certificate()["passed"]:
                raise GraphError(f"{sid}: Brown zoom/ray domain is not certified")
        for rid, ref in REFERENCE_SHOTS.items():
            rig = g["rigs"][rid]
            if rig.get("reference_shot") != ref:
                raise GraphError(f"{rid}: changed rig angle gauge without an explicit migration")
            a = g["shots"][ref].get("reference_angles")
            if a is not None and not np.array_equal(np.asarray(a, float), [0.0, 0.0]):
                raise GraphError(f"{rid}: reference shot angle offsets must be fixed to zero")
        if require_ready and incomplete:
            raise GraphError("camera initialisation incomplete: " + ", ".join(incomplete))
        return {"schema_valid": True, "initialised_shots": 9 - len(incomplete),
                "uninitialised_shots": incomplete, "fit_accuracy_accepted": False}

    def local_seconds(self, shot_id: str, native_pts: Any) -> np.ndarray:
        if shot_id not in self.graph["shots"]:
            raise GraphError(f"unknown shot {shot_id}")
        s = self.graph["shots"][shot_id]
        p = np.asarray(native_pts)
        if p.dtype.kind not in "iu":
            raise GraphError("native_pts must be an integer array; legacy60/time seconds are rejected")
        first, last = s["scope"]["first_pts"], s["scope"]["last_pts"]
        if np.any((p < first) | (p > last)) or np.any((p - first) % 256):
            raise GraphError(f"{shot_id}: PTS outside shot or off native 50 fps exposure lattice")
        return (p.astype(np.int64) - s["time_origin_pts"]) / 12800.0

    def native_camera(self, shot_id: str):
        if shot_id in self._cameras:
            return self._cameras[shot_id]
        from rinkfit_start.native_camera import NativeCamera, rotation
        s = self.graph["shots"][shot_id]
        rig = self.graph["rigs"][s["rig_id"]]
        if any(rig.get(k) is None for k in ("shared_head", "optics", "rolling_shutter")):
            raise GraphError(f"{shot_id}: rig not initialised; camera25 cannot fill it implicitly")
        if s.get("camera_controls") is None or s.get("reference_angles") is None:
            raise GraphError(f"{shot_id}: shot-local orientation/zoom trajectory not initialised")
        forbidden = {"optics", "shared_head", "rolling_shutter", "camera_xyz"}.intersection(s)
        if forbidden:
            raise GraphError(f"{shot_id}: illegal per-shot copies of rig properties: {sorted(forbidden)}")
        head = copy.deepcopy(rig["shared_head"])
        for key, shape in (("R0", (3, 3)), ("n", (3,)), ("b", (3,)), ("P", (3,)), ("d", (3,))):
            value = np.asarray(head[key], float)
            if value.shape != shape or not np.isfinite(value).all():
                raise GraphError(f"{shot_id}: invalid shared head {key}")
        for key in ("a", "pupil_linear", "pupil_quadratic"):
            if not np.isfinite(float(head.get(key, 0.0))):
                raise GraphError(f"{shot_id}: nonfinite shared head {key}")
        for key, value in rig["optics"].items():
            if not np.isfinite(np.asarray(value, float)).all():
                raise GraphError(f"{shot_id}: nonfinite optical parameter {key}")
        if float(rig["optics"]["f_ref"]) <= 0:
            raise GraphError(f"{shot_id}: f_ref must be positive and fixed within a rig")
        if not np.isfinite(float(rig["rolling_shutter"]["signed_readout_seconds"])):
            raise GraphError(f"{shot_id}: nonfinite readout time")
        offsets = np.asarray(s["reference_angles"], float)
        if offsets.shape != (2,) or not np.isfinite(offsets).all():
            raise GraphError(f"{shot_id}: expected two finite reference angles")
        controls = s["camera_controls"]
        if "controls" in controls and np.asarray(controls["controls"]).shape != (int(controls["basis"]["n_controls"]), 3):
            raise GraphError(f"{shot_id}: complete control count differs from cubic basis")
        alpha, beta = offsets
        head["R0"] = (rotation(head["b"], beta) @ np.asarray(head["R0"], float)
                      @ rotation(head["n"], alpha)).tolist()
        camera = NativeCamera({"shared_head": head, "optics": copy.deepcopy(rig["optics"]),
            "rolling_shutter": copy.deepcopy(rig["rolling_shutter"]),
            "camera_controls": copy.deepcopy(s["camera_controls"])})
        self._cameras[shot_id] = camera
        return camera

    def project(self, shot_id: str, world_xyz: Any, native_pts: Any) -> dict[str, Any]:
        return self.native_camera(shot_id).project_rs(world_xyz, self.local_seconds(shot_id, native_pts))

    def project_rows(self, shot_ids: Any, native_pts: Any, world_xyz: Any) -> dict[str, Any]:
        ids = np.asarray(shot_ids); pts = np.asarray(native_pts); xyz = np.asarray(world_xyz, float)
        if ids.ndim != 1 or pts.shape != ids.shape or xyz.shape != (len(ids), 3):
            raise GraphError("row arrays must have shapes [N], [N], [N,3]")
        unknown = set(ids.tolist()) - set(self.graph["shots"])
        if unknown:
            raise GraphError(f"unrouted shot rows: {sorted(unknown)}")
        result = {"uv": np.full((len(ids), 2), np.nan), "valid": np.zeros(len(ids), bool),
                  "row": np.full(len(ids), np.nan), "root_equality_px": np.full(len(ids), np.nan)}
        for sid in np.unique(ids):
            rows = np.flatnonzero(ids == sid)
            out = self.project(str(sid), xyz[rows], pts[rows])
            for key in result:
                result[key][rows] = out[key]
        return result

    def parameter_dependencies(self, shot_id: str, material_key: str) -> list[str]:
        """Factor ownership for the later sparse solver; no shot-specific scene."""
        rig = self.graph["shots"][shot_id]["rig_id"]
        return [f"shared_scene/materials/{material_key}", "shared_scene/rink_and_board",
                f"rigs/{rig}/shared_head", f"rigs/{rig}/optics",
                f"rigs/{rig}/rolling_shutter", f"shots/{shot_id}/reference_angles",
                f"shots/{shot_id}/camera_controls"]


def with_s01_event(graph: Mapping[str, Any], event_model: Mapping[str, Any],
                   *, model_ref: str, model_sha256: str, import_receipt: str) -> dict[str, Any]:
    """Return a NEW candidate graph; the live s01 repair state is never written."""
    from rinkfit_start.native_scene import camera_from_event
    if event_model.get("schema") != "rinkfit.event_model.v1.1":
        raise GraphError("latest s01 schema needs an explicit native adapter; refusing old-seed fallback")
    result = copy.deepcopy(dict(graph))
    current = CameraGraph(result)
    current.validate()
    if any(result["shots"][sid].get("camera_controls") is not None for sid in SHOT_TO_RIG if sid != "s01"):
        raise GraphError("initial import only: a changed shared scene requires explicit joint-state rebase")
    camera = camera_from_event(event_model).source
    result["rigs"]["main"].update(status="s01_warm_start_not_frozen",
        shared_head=copy.deepcopy(camera["shared_head"]), optics=copy.deepcopy(camera["optics"]),
        rolling_shutter=copy.deepcopy(camera["rolling_shutter"]), tau_bound=float(event_model["camera"]["tau_bound"]))
    result["shots"]["s01"].update(status="imported_complete_warm_state",
        reference_angles=[0.0, 0.0], camera_controls=copy.deepcopy(camera["camera_controls"]))
    result["shared_scene"].update(status="latest_s01_warm_start_not_frozen",
        event_model_ref=model_ref, event_model_sha256=model_sha256,
        source_import_receipt=import_receipt, geometry_is_frozen=False)
    result["cache_generation"] = int(result.get("cache_generation", 0)) + 1
    result["certificates"] = {"joint_fit_complete": False, "all_source_replay_complete": False,
                              "reason": "import or shared-state change invalidates all joint caches"}
    CameraGraph(result).validate()
    return result
