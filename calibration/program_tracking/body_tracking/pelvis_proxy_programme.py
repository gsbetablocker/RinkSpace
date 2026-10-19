"""Bind one source hip-midpoint proxy to a shared rink at a declared fixed h.

This candidate entry does not infer skeletal pose, pelvis height or blade
contact.  Missing source observations and cuts break the displayed path.
Camera/scene inputs remain replaceable and content-bound.  The complete model
inventory and actual finite advertisement borders are shared by both views.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from ..runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import cv2
from PIL import Image, ImageDraw, ImageFont
from rinkspace.metric_scale import AssumedMetricScale, DEFAULT_SCALE

from .camera_bridge import height_conditioned_ground, project_world
from ..render_result import (
    advertisement_review_paths, DEFAULT_MASKS, DEFAULT_PEOPLE,
    fixed_scene_paths, PersonMasks, SceneRenderer,
)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def json_read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n", encoding="utf-8")


def source_time(pts: int) -> str:
    seconds = pts/15360
    return f"{int(seconds)//3600:02d}:{int(seconds)%3600//60:02d}:{seconds%60:06.3f}"


def select_scope(pts: np.ndarray, start: int | None, end: int | None) -> np.ndarray:
    ids = np.flatnonzero((pts >= (pts[0] if start is None else start))
                         & (pts <= (pts[-1] if end is None else end)))
    if (not len(ids) or (start is not None and pts[ids[0]] != start)
            or (end is not None and pts[ids[-1]] != end)
            or np.any(np.diff(pts[ids]) != 256)):
        raise ValueError("Scope endpoints must be present consecutive native source PTS")
    return ids


def validate_camera_packet(path: Path, packet: dict, shots: dict, *, scene_path=None) -> dict:
    """Replay typed Z190 exports without turning their status into acceptance."""
    if "metadata_json" not in packet:
        result = dict(format="native_camera25", accuracy_accepted=False,
                      camera_source_metadata_present=False)
        if "source_sha256" in packet:
            if str(packet["source_sha256"]) != shots["source_sha256_from_timeline"]:
                raise ValueError("Native camera packet belongs to another source video")
            result.update(camera_source_metadata_present=True,
                          source_sha256=str(packet["source_sha256"]))
        if "model_sha256" in packet:
            if scene_path is not None and str(packet["model_sha256"]) != digest(scene_path):
                raise ValueError("Native camera packet belongs to another fixed scene")
            result["scene_sha256"] = str(packet["model_sha256"])
        if "status" in packet:
            result["source_camera_status_counts"] = dict(Counter(map(str, packet["status"])))
        return result
    from ..z190_physical_camera import PhysicalCameraModel
    # The physical loader rejects altered states, aliases, source metadata or
    # camera25 values instead of recovering a state from the numeric camera.
    PhysicalCameraModel.load_npz(path)
    metadata = json.loads(str(packet["metadata_json"]))
    if (metadata["source_sha256"] != shots["source_sha256_from_timeline"]
            or metadata["source_time_base"] != shots["source_time_base"]):
        raise ValueError("Camera package and source hips belong to different source clocks")
    result = dict(format=metadata["schema_version"], accuracy_accepted=False,
                camera_source_metadata_present=True, canonical_replay_verified=True,
                exposure_policy=metadata["exposure_policy"],
                source_sha256=metadata["source_sha256"],
                source_time_base=metadata["source_time_base"])
    binding_path = path.with_suffix(".binding.json")
    if binding_path.exists():
        binding = json_read(binding_path)
        if binding.get("exported_camera_sha256") != digest(path):
            raise ValueError("Source-exposure camera bridge binding changed")
        if scene_path is not None and binding["scene_sha256"] != digest(scene_path):
            raise ValueError("Camera bridge belongs to another fixed scene")
        result.update(source_exposure_bridge_binding_path=str(binding_path.resolve()),
                      source_exposure_bridge_binding_sha256=digest(binding_path),
                      actual_state_rekeying=binding["rekeyed_actual_source_keys"])
    return result


def validate_source_hip_clock(path: Path, hips: dict, shots_path: Path) -> dict:
    """Check the original evidence mapping, preserving all 60 Hz presentations."""
    from .exposure_observations import confirmed_representatives, load_confirmed_evidence
    report_path = path.with_name("source_observations_report.json")
    report = json_read(report_path)
    shots = json_read(shots_path)
    if report["source_binding"]["sha256"] != shots["source_sha256_from_timeline"]:
        raise ValueError("Source hip evidence belongs to another video")
    evidence = report["input_bindings"]
    groups_path = next(Path(p) for p in evidence if Path(p).name == "exposure_groups.json")
    mapping_path = next(Path(p) for p in evidence if Path(p).name == "native60_to_official50.jsonl")
    for p in (groups_path, mapping_path):
        if digest(p) != evidence[str(p)]:
            raise ValueError("Confirmed source exposure evidence changed")
    groups, mapping, hashes = load_confirmed_evidence(groups_path, mapping_path, shots_path)
    reps, proof = confirmed_representatives(
        groups, mapping, hips["pts"], hips["shot_index"], shots["source_sha256_from_timeline"])
    if not np.array_equal(reps, hips["source_observation_pts"]):
        raise ValueError("Source hips no longer preserve confirmed exposure representatives")
    if not np.array_equal(hips["source_observation_alias"], reps != hips["pts"]):
        raise ValueError("Source hip alias flags changed confirmed exposure identity")
    return dict(report_path=str(report_path.resolve()), report_sha256=digest(report_path),
                exposure_proof=proof, exposure_manifest_hashes=hashes)


def bind_fixed_height_proxy(cameras_path: Path, hips_path: Path, scene_path: Path,
                            shots_path: Path, output_dir: Path,
                            height_scene: float | None = None, *, height_m=None,
                            metric_scale_path=DEFAULT_SCALE, start_pts=None,
                            end_pts=None, conditional_diagnostic=False) -> dict:
    """Intersect the hip-midpoint source ray with caller-declared z=h only."""
    started = time.perf_counter()
    scale = AssumedMetricScale.load(metric_scale_path, scene_path=scene_path)
    if (height_scene is None) == (height_m is None):
        raise ValueError("Declare exactly one fixed pelvis height in scene units or assumed m")
    height_input_unit = "metric_assumed_m" if height_m is not None else "conditional_scene_units"
    if height_m is not None:
        height_scene = float(height_m)/scale.metres_per_scene_unit
    if not np.isfinite(height_scene) or height_scene <= 0:
        raise ValueError("An explicit positive conditional body height is required")
    hips, cameras = read_npz(hips_path), read_npz(cameras_path)
    scene, manifest = json_read(scene_path), json_read(shots_path)
    camera_metadata = validate_camera_packet(cameras_path, cameras, manifest, scene_path=scene_path)
    hip_clock = validate_source_hip_clock(hips_path, hips, shots_path)
    coordinate = scene["coordinate_system"]
    if (coordinate["handedness"] != "right" or coordinate["z_axis"] != "above ice"
            or coordinate["ice_z"] != 0. or not coordinate.get("id")
            or coordinate["unit"] != "conditional_scene_units"):
        raise ValueError("Expected the declared shared right-handed Z-up scene")
    expected = np.arange(manifest["analysis_scope"]["start_pts"],
                         manifest["analysis_scope"]["end_pts_exclusive"], 256)
    if (not np.array_equal(hips["pts"], expected)
            or tuple(hips["native_size_wh"]) != (1920, 1080)
            or str(hips["source_time_base"]) != "1/15360"):
        raise ValueError("Source hip packet changes native programme geometry/timeline")
    camera_pts, camera25 = cameras["pts"], cameras["camera25"]
    if (camera_pts.dtype.kind not in "iu" or np.any(np.diff(camera_pts) <= 0)
            or camera25.shape != (len(camera_pts), 25)
            or not np.isfinite(camera25).all()):
        raise ValueError("Ordered exact native camera PTS and finite camera25 required")
    ids = select_scope(hips["pts"], start_pts, end_pts)
    pts = hips["pts"][ids]
    ci = np.searchsorted(camera_pts, pts)
    if np.any(ci >= len(camera_pts)) or not np.array_equal(camera_pts[ci], pts):
        raise ValueError("Camera candidate does not cover the entire requested source scope")
    cam = camera25[ci]
    reps = hips["source_observation_pts"][ids]
    if "exposure_representative_pts" in cameras:
        if not np.array_equal(cameras["exposure_representative_pts"][ci], reps):
            raise ValueError("Camera and source hips disagree about confirmed exposure aliases")
    for p in np.unique(reps):
        rows = np.flatnonzero(reps == p)
        if not np.allclose(cam[rows], cam[rows[0]], rtol=0, atol=1e-10):
            raise ValueError("Repeated confirmed exposure has different background cameras")
    shot = hips["shot_index"][ids]
    for p, s in zip(pts, shot):
        row = manifest["shots"][int(s)]
        if not row["start_pts"] <= p < row["end_pts"]:
            raise ValueError("Source hip packet assigns a frame to the wrong cut")
    n = len(pts)
    P = np.full((n, 3), np.nan); G = P.copy(); derivative = P.copy()
    root_uv = np.full((n, 2), np.nan); ground_uv = root_uv.copy()
    valid = np.zeros(n, bool); inside = np.zeros(n, bool)
    closure = np.full(n, np.nan)
    status = hips["source_hip_status"][ids].astype("<U96")
    source_uv = hips["source_hip_midpoint_proxy_uv"][ids]
    source_valid = hips["source_hip_proxy_candidate_valid"][ids]
    perimeter = np.asarray(scene["ice_boundary"]["points_xyz"], float)[:, :2]
    polygon = perimeter.astype(np.float32)
    cache = {}
    for j in range(n):
        if not source_valid[j]:
            continue
        key = int(reps[j])
        if key not in cache:
            point, dh, good = height_conditioned_ground(
                source_uv[j:j+1], cam[j], height_scene)
            record = dict(valid=False, status="conditional_source_ray_height_unavailable")
            if good[0]:
                p = point[0].copy(); p[2] = height_scene
                uv, optical = project_world(np.array([p, point[0]]), cam[j])
                if optical[0]:
                    record = dict(valid=True, status="source_hip_proxy_at_declared_fixed_height",
                                  P=p, G=point[0], dh=dh[0], Puv=uv[0],
                                  Guv=uv[1] if optical[1] else np.full(2, np.nan),
                                  closure=float(np.linalg.norm(uv[0]-source_uv[j])),
                                  inside=cv2.pointPolygonTest(polygon, tuple(map(float, p[:2])), False) >= 0)
            cache[key] = record
        row = cache[key]; status[j] = row["status"]
        if not row["valid"]:
            continue
        valid[j] = True; P[j] = row["P"]; G[j] = row["G"]
        derivative[j] = row["dh"]; root_uv[j] = row["Puv"]
        ground_uv[j] = row["Guv"]; inside[j] = row["inside"]
        closure[j] = row["closure"]
    if np.any(closure[valid] > 1e-4):
        raise ValueError("Conditional source pixel inverse fails native round trip")
    connect = np.zeros(n, bool)
    # A caller-declared fixed h is not source body-height evidence. Keeping its
    # ray intersection is useful for explicit diagnostics, but it must not
    # silently become a usable world trajectory after a camera replacement.
    if conditional_diagnostic:
        connect[1:] = valid[:-1] & valid[1:] & (shot[:-1] == shot[1:]) & (np.diff(pts) == 256)
    world_status = np.where(source_valid,
        "pelvis_height_unobserved_world_position_unknown", "source_hip_unknown").astype("<U64")
    output_dir.mkdir(parents=True, exist_ok=False)
    packet = output_dir/"fixed_height_pelvis_proxy_trace.npz"
    np.savez_compressed(packet, pts=pts, shot_index=shot, source_observation_pts=reps,
        camera25=cam, source_H_hip_pair_uv=hips["source_H_hip_pair_uv"][ids],
        camera25_assumed_m=scale.camera25_to_metres(cam),
        source_hip_midpoint_proxy_uv=source_uv, source_hip_proxy_candidate_valid=source_valid,
        conditional_proxy_P_world_xyz=P, conditional_vertical_G_world_xyz=G,
        conditional_proxy_P_assumed_m_xyz=scale.points_to_metres(P),
        conditional_vertical_G_assumed_m_xyz=scale.points_to_metres(G),
        conditional_proxy_P_native_uv=root_uv, conditional_vertical_G_native_uv=ground_uv,
        conditional_projection_valid=valid, conditional_projection_status=status,
        pelvis_world_position_known=np.zeros(n, bool),
        pelvis_world_G_assumed_m_xyz=np.full((n, 3), np.nan),
        pelvis_world_position_status=world_status,
        pelvis_height_source_evidence_valid=np.zeros(n, bool),
        world_trajectory_usable=np.array(False),
        conditional_diagnostic_explicitly_requested=np.array(bool(conditional_diagnostic)),
        G_xy_dheight_scene=derivative, G_xy_inside_rink=inside,
        G_xyz_dheight_assumed_m=derivative,
        native_source_inverse_closure_px=closure,
        connect_from_previous_native_source=connect,
        declared_fixed_height_scene=np.array(height_scene),
        declared_fixed_height_assumed_m=np.array(height_scene*scale.metres_per_scene_unit),
        metres_per_scene_unit=np.array(scale.metres_per_scene_unit),
        metric_geometry_level=np.array("metric_assumed"),
        verified_metre_scale=np.array(False),
        blade_contact_status=np.full(n, "not_estimated", dtype="<U32"),
        exact_native_joint0_source_valid=hips["source_native_joint0_valid"][ids])
    bindings = dict(cameras=dict(path=str(cameras_path.resolve()), sha256=digest(cameras_path)),
                    source_hips=dict(path=str(hips_path.resolve()), sha256=digest(hips_path)),
                    scene=dict(path=str(scene_path.resolve()), sha256=digest(scene_path)),
                    shots=dict(path=str(shots_path.resolve()), sha256=digest(shots_path)),
                    metric_scale=dict(path=str(Path(metric_scale_path).resolve()),
                                      sha256=digest(Path(metric_scale_path))))
    report = dict(schema="rinkspace.fixed-height-source-pelvis-proxy.v1",
        accuracy_accepted=False, height_measured=False, exact_native_joint0_measured=False,
        world_trajectory_usable=False, known_world_pelvis_frames=0,
        world_height_unknown_frames=n,
        height_authority="caller_declared_fixed_assumption_no_current_camera_body_height_evidence",
        conditional_diagnostic_explicitly_requested=bool(conditional_diagnostic),
        default_world_trajectory="unknown; conditional P/G arrays are numeric diagnostics only",
        condition="Source COCO11/12 image midpoint treated as pelvis proxy; declared fixed z=h",
        height_scene=height_scene, world_coordinate_system=coordinate,
        height_assumed_m=height_scene*scale.metres_per_scene_unit,
        height_input_unit=height_input_unit,
        metric_conversion=dict(scale_id=scale.manifest["scale_id"],
            metres_per_scene_unit=scale.metres_per_scene_unit,
            geometry_level="metric_assumed", verified_metre_scale=False,
            native_solver_and_validity_thresholds_unchanged=True,
            anatomical_accuracy_or_camera_alignment_promoted=False),
        camera_package_metadata=camera_metadata,
        source_hip_clock=hip_clock,
        G_semantics="P=(x,y,h); G=(x,y,0), vertical pelvis-proxy projection, not blade contact",
        no_complex_pose_or_all17_fit=True, no_XY_lock_or_temporal_smoothing=True,
        no_outside_rink_clamping=True, no_cut_or_missing_links=True,
        source_binding=dict(path=manifest["source_path"], sha256=manifest["source_sha256_from_timeline"]),
        input_bindings=bindings, native_frames=n, actual_exposures=len(np.unique(reps)),
        conditional_valid_frames=int(valid.sum()), unknown_frames=int((~valid).sum()),
        candidate_G_outside_rink_frames=int((valid & ~inside).sum()),
        native_inverse_closure_px_max=float(np.max(closure[valid])) if valid.any() else None,
        status_counts=dict(Counter(map(str, status))),
        trace_sha256=digest(packet), runtime_seconds=time.perf_counter()-started)
    save_json(output_dir/"projection_binding.json", report)
    return report


class PlanView:
    """Same world inventory and finite advertisements, with a broken source path."""
    def __init__(self, scene: dict, height: float, *, height_m=None, board_paths=None):
        self.base = np.full((1080, 660, 3), 248, np.uint8)
        boundary = np.asarray(scene["ice_boundary"]["points_xyz"], float)[:, :2]
        low, high = boundary.min(axis=0), boundary.max(axis=0)
        self.center = (low+high)/2
        self.gain = min(604/(high[0]-low[0]), 350/(high[1]-low[1]))
        paths = list(fixed_scene_paths(scene, draw_grid=True))
        if board_paths is not None:
            for kind, world in board_paths:
                world = np.asarray(world, float)
                if (kind not in ('board_horizontal','board_vertical')
                        or world.ndim != 2 or world.shape[1:] != (3,)
                        or len(world) < 2 or not np.isfinite(world).all()):
                    raise ValueError('Explicit same-scene finite board-grid paths required')
                paths.append((kind,world.copy()))
        self.model_paths = tuple(paths)
        self.path_counts = dict(Counter(kind for kind, _ in paths))
        for kind, path in paths:
            if kind == "board_top":
                continue
            color = {"grid":(215, 236, 215), "marking":(200, 120, 190), "contact":(70, 70, 70),
                     "board_horizontal":(220,175,90), "board_vertical":(220,175,90)}[kind]
            cv2.polylines(self.base, [self.mapxy(path[:, :2])], False, color,
                          1 if kind == "grid" else 2, cv2.LINE_AA)
        labels = {}
        self.shared_board_pattern_ids = []
        for path in advertisement_review_paths(scene):
            uv = self.mapxy(path["xyz"][:, :2])
            if path.get('primitive') == 'points':
                for point in uv:
                    cv2.drawMarker(self.base, tuple(map(int, point)), (210, 160, 20), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            else:
                cv2.polylines(self.base, [uv], False, (210, 160, 20), 2, cv2.LINE_AA)
            if 'pattern_kind' in path:
                self.shared_board_pattern_ids.append(path['number'])
            labels.setdefault(path["number"], uv.mean(axis=0))
        for number, uv in labels.items():
            cv2.putText(self.base, f"ad{number}", tuple(np.rint(uv).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (160, 115, 15), 1, cv2.LINE_AA)
        self.advertisement_ids = sorted(labels)
        self.font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
        self.small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
        self.base = self.text(self.base, (24, 22), "同一冰场：全部已有冰线与广告位置")
        height_label = (f"固定 h={height_m:.6f} m（条件米制）" if height_m is not None
                        else f"固定 h={height:.6f} scene")
        self.base = self.text(self.base, (24, 530), "骨盆 2D 代理；"+height_label)
        self.base = self.text(self.base, (24, 578), "绿点 P；蓝线 G：骨盆垂直投影")
        self.base = self.text(self.base, (24, 626), "切镜、缺失断线；冰刀接触未估计", small=True)
        self.base = self.text(self.base, (24, 670), "高度和解剖中心仍是条件值", small=True)
        self.base = self.text(self.base, (24, 714), "当前用于连续查错，尚未确认准确", color=(155, 60, 45), small=True)
        self.trail = self.base.copy()

    def mapxy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.rint(np.column_stack(((xy[:, 0]-self.center[0])*self.gain+330,
                                        305-(xy[:, 1]-self.center[1])*self.gain))).astype(np.int32)

    def text(self, image, pos, value, color=(40,40,40), small=False):
        bitmap = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(bitmap).text(pos, value, font=self.small if small else self.font, fill=color)
        return cv2.cvtColor(np.asarray(bitmap), cv2.COLOR_RGB2BGR)

    def frame(self, G, previous, connect, valid, inside):
        if valid and connect and previous is not None:
            uv = self.mapxy(np.array([previous[:2], G[:2]]))
            ok, a, b = cv2.clipLine((0, 0, 660, 510), tuple(uv[0]), tuple(uv[1]))
            if ok:
                cv2.line(self.trail, a, b, (210, 120, 40), 2, cv2.LINE_AA)
        right = self.trail.copy()
        if valid:
            uv = self.mapxy(G[None, :2])[0]
            if 0 <= uv[0] < 660 and 0 <= uv[1] < 510:
                cv2.circle(right, tuple(uv), 5, (210, 120, 40) if inside else (20, 20, 230), -1)
            label = "当前 G：场内条件值" if inside else "当前 G：条件投影在场外，需核查"
        else:
            label = "当前骨盆代理未知；没有补轨迹"
        return self.text(right, (24, 778), label, small=True)


def render_proxy_review(trace_dir: Path, scene_path: Path, output: Path,
                        *, start_pts=None, end_pts=None,
                        masks_path=DEFAULT_MASKS, people_path=DEFAULT_PEOPLE,
                        conditional_diagnostic=False) -> dict:
    """Decode actual consecutive source frames; never open or autoplay output."""
    started = time.perf_counter()
    data = read_npz(trace_dir/"fixed_height_pelvis_proxy_trace.npz")
    binding = json_read(trace_dir/"projection_binding.json")
    if not (conditional_diagnostic or binding.get("conditional_diagnostic_explicitly_requested", False)):
        raise ValueError("Body height has no matched source evidence. Use an explicit conditional diagnostic, or the height-range entry; world trajectory remains unknown.")
    if (digest(scene_path) != binding["input_bindings"]["scene"]["sha256"]
            or digest(trace_dir/"fixed_height_pelvis_proxy_trace.npz") != binding["trace_sha256"]):
        raise ValueError("Trace geometry/binding changed; reproject with the new scene first")
    scene = json_read(scene_path)
    source = Path(binding["source_binding"]["path"])
    if digest(source) != binding["source_binding"]["sha256"]:
        raise ValueError("Actual source video identity differs from the observations")
    chosen = select_scope(data["pts"], start_pts, end_pts)
    pts = data["pts"][chosen]
    renderer = SceneRenderer(scene, graphics_regions=((28,24,554,253),(1690,944,1865,1000)))
    plan = PlanView(scene, float(data["declared_fixed_height_scene"]),
                    height_m=binding.get("height_assumed_m"))
    if plan.path_counts != renderer.path_counts:
        raise ValueError("Camera and plan views differ in fixed model inventory")
    masks = PersonMasks(masks_path, people_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Preserve existing diagnostic output")
    video_output = av.open(str(output), "w")
    encoding = video_output.add_stream("libx264", rate=60)
    encoding.width, encoding.height = 2580, 1120
    encoding.pix_fmt = "yuv420p"; encoding.options = {"crf":"20", "preset":"fast"}
    facts = []; previous = None
    with av.open(str(source)) as movie:
        stream = movie.streams.video[0]; stream.thread_type = "AUTO"
        movie.seek(int(pts[0]), stream=stream, backward=True, any_frame=False)
        for original in movie.decode(stream):
            if original.pts is None or original.pts < pts[0]:
                continue
            if original.pts > pts[-1]:
                break
            q = len(facts)
            if q >= len(pts) or int(original.pts) != int(pts[q]):
                raise ValueError("Continuous source decoder changed native PTS/cadence")
            j = chosen[q]; native = original.to_ndarray(format="bgr24")
            mask, mask_status = masks.get(int(original.pts), require_sam2=True)
            image, metrics = renderer.draw(native, data["camera25"][j], mask)
            for uv in data["source_H_hip_pair_uv"][j]:
                if np.isfinite(uv).all() and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                    cv2.drawMarker(image, tuple(np.rint(uv).astype(int)), (230,200,25), cv2.MARKER_CROSS, 16, 2)
            valid = bool(data["conditional_projection_valid"][j])
            if valid:
                uv = np.array([data["conditional_proxy_P_native_uv"][j], data["conditional_vertical_G_native_uv"][j]])
                if np.isfinite(uv).all() and np.max(np.abs(uv)) < 1e7:
                    a, b = map(tuple, np.rint(uv).astype(int))
                    ok, aa, bb = cv2.clipLine((0,0,1920,1080), a, b)
                    if ok:
                        cv2.line(image, aa, bb, (210,120,40), 2, cv2.LINE_AA)
                    for point, color in ((a,(45,210,55)), (b,(210,120,40))):
                        if 0 <= point[0] < 1920 and 0 <= point[1] < 1080:
                            cv2.drawMarker(image, point, color, cv2.MARKER_DIAMOND, 20, 3)
            G = data["conditional_vertical_G_world_xyz"][j]
            right = plan.frame(G, previous, bool(data["connect_from_previous_native_source"][j]) and q>0,
                               valid, bool(data["G_xy_inside_rink"][j]))
            previous = G if valid else None
            canvas = np.zeros((1120,2580,3), np.uint8)
            canvas[:1080,:1920] = image; canvas[:1080,1920:] = right
            label = f"原片 {source_time(int(original.pts))} | PTS {original.pts} | 固定高度未测；骨盆条件诊断、世界定位未知；原生60帧/重复曝光保留"
            canvas = plan.text(canvas, (20,1084), label, color=(235,235,235))
            frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            frame.pts = q; frame.time_base = Fraction(1,60)
            for packet in encoding.encode(frame):
                video_output.mux(packet)
            if q in (0,len(pts)//2,len(pts)-1):
                cv2.imwrite(str(output.with_name(output.stem+f"_{int(original.pts)}.jpg")), canvas)
            facts.append(dict(source_pts=int(original.pts), source_observation_pts=int(data["source_observation_pts"][j]),
                              shot=int(data["shot_index"][j]), projection_valid=valid,
                              condition_status=str(data["conditional_projection_status"][j]),
                              connect_from_previous=bool(data["connect_from_previous_native_source"][j]) and q>0,
                              mask_status=mask_status, drawn_model_paths=metrics["fixed_model_path_counts"],
                              projected_advertisement_ids=metrics["advertisements"]["projected_ids"],
                              confidence_gating=metrics["confidence_gating"]))
    if len(facts) != len(pts):
        raise ValueError("Native source review is incomplete")
    for packet in encoding.encode():
        video_output.mux(packet)
    video_output.close()
    count = 0
    with av.open(str(output)) as check:
        stream = check.streams.video[0]
        if float(stream.average_rate) != 60.:
            raise ValueError("Diagnostic encoding changed native presentation rate")
        for frame in check.decode(stream):
            if abs(float(frame.pts*stream.time_base)-count/60) > 1e-8:
                raise ValueError("Diagnostic encoding dropped or retimed a source frame")
            count += 1
    if count != len(pts):
        raise ValueError("Diagnostic encoding changed source frame count")
    result = dict(schema="rinkspace.fixed-height-source-pelvis-proxy-review.v1",
        video=str(output.resolve()), accuracy_accepted=False,
        world_trajectory_usable=False, height_measured=False,
        conditional_diagnostic_explicitly_requested=True,
        projection_trace_sha256=binding["trace_sha256"], scene_sha256=digest(scene_path),
        source_sha256=binding["source_binding"]["sha256"], video_sha256=digest(output),
        native_source_first_pts=int(pts[0]), native_source_last_pts=int(pts[-1]),
        encoded_native_source_frames=count, presentation_rate="60/1",
        fixed_scene_inventory=renderer.path_counts, advertisement_ids=plan.advertisement_ids,
        h_scene=float(data["declared_fixed_height_scene"]),
        h_assumed_m=binding.get("height_assumed_m"),
        metric_conversion=binding.get("metric_conversion"),
        trail="Vertical source pelvis-proxy projection; cuts/unknown break lines; no contact inference",
        source_frames=facts, runtime_seconds=time.perf_counter()-started)
    save_json(output.with_suffix(".binding.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    project = actions.add_parser("project")
    for name in ("cameras", "source-hips", "scene", "shots", "output-dir"):
        project.add_argument("--"+name, type=Path, required=True)
    height = project.add_mutually_exclusive_group(required=True)
    height.add_argument("--height-scene", type=float)
    height.add_argument("--height-m", type=float, help="Declared fixed pelvis height in assumed metres")
    project.add_argument("--metric-scale", type=Path, default=DEFAULT_SCALE)
    render = actions.add_parser("render")
    for name in ("trace-dir", "scene", "output"):
        render.add_argument("--"+name, type=Path, required=True)
    for action in (project, render):
        action.add_argument("--conditional-diagnostic", action="store_true",
                            help="Explicitly show the declared-height condition; does not establish a known world trajectory")
        action.add_argument("--start-pts", type=int)
        action.add_argument("--end-pts", type=int)
    args = parser.parse_args(); cv2.setNumThreads(1)
    if args.action == "project":
        result = bind_fixed_height_proxy(args.cameras, args.source_hips, args.scene,
            args.shots, args.output_dir, args.height_scene, height_m=args.height_m,
            metric_scale_path=args.metric_scale, start_pts=args.start_pts, end_pts=args.end_pts,
            conditional_diagnostic=args.conditional_diagnostic)
        keys = ("native_frames", "conditional_valid_frames", "unknown_frames", "known_world_pelvis_frames",
                "world_trajectory_usable", "world_height_unknown_frames", "candidate_G_outside_rink_frames",
                "native_inverse_closure_px_max", "runtime_seconds")
    else:
        result = render_proxy_review(args.trace_dir, args.scene, args.output,
                                     start_pts=args.start_pts, end_pts=args.end_pts,
                                     conditional_diagnostic=args.conditional_diagnostic)
        keys = ("video", "encoded_native_source_frames", "fixed_scene_inventory", "advertisement_ids", "runtime_seconds")
    print(json.dumps({key:result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
