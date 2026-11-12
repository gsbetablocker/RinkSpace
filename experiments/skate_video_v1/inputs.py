"""Bind the experiment to immutable programme inputs and the native pixel clock."""

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]

from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.review.render import source_frames
from rinkspace.storage import configured_data_root

PROGRAMME = PROJECT / "events/ankara_2026/programmes/okada_mei_sp"
ROOT = configured_data_root(PROJECT)
RUN = ROOT / "runs/body_s01_20261029/structure_camera_1108"
PARENT = PROGRAMME / "programme_skate_chain_v1.json"
PREPARED = RUN / "prepared_repair_lsp_arm_identity"
WEIGHTS = ROOT / "cache/pretrained/sam2.1-hiera-tiny"
SIZE = (1920, 1080)
BOOT = {"boot_shell", "heel_shell", "toe_shell", "outsole"}
STEEL = {"runner_lower_edge", "runner_upper_edge"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    pending = path.with_suffix(".pending.npz")
    np.savez_compressed(pending, **arrays)
    pending.replace(path)


def atomic_json(path, doc):
    path = Path(path)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def distribute(curve, size, closed=False):
    points = np.asarray(curve, dtype=np.float64)
    if closed:
        points = np.vstack((points, points[0]))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.column_stack([np.interp(np.linspace(0., arc[-1], size, endpoint=not closed), arc, c)
                            for c in points.T]).astype(np.float32)


def masked_distance(points, curve, closed=False):
    curve = np.asarray(curve)
    if closed:
        curve = np.vstack((curve, curve[0]))
    v = np.diff(curve, axis=0)
    d = np.asarray(points)[:, None] - curve[:-1]
    station = np.clip((d * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-8), 0., 1.)
    return np.linalg.norm(d - station[..., None] * v, axis=-1).min(axis=1)


class Inputs:
    def __init__(self, *, verify_media=False):
        scope = read(PROGRAMME / "rebuild_scope.json")
        registered = read(PARENT)
        parent = registered
        recovery_path = os.environ.get("RINKSPACE_SKATE_RECOVERY_BINDING")
        self.recovery_path = Path(recovery_path) if recovery_path else None
        if recovery_path:
            parent = read(recovery_path)
            if (parent.get("schema") != "rinkspace.skate-video-recovery-binding.v1"
                    or parent["candidate_id"] != registered["candidate_id"]
                    or parent["parameter"] != registered["parameter"]
                    or parent["camera_sha256"] != registered["camera_sha256"]
                    or not parent["exact_motion_verified_against_archive"]):
                raise ValueError("Recovery does not match the registered programme state")
            if digest(parent["result"]["path"]) != parent["result"]["sha256"]:
                raise ValueError("Recovered body result changed")
        self.parent = parent
        self.prepared = Path(parent.get("prepared", PREPARED))
        self.pts, self.shots, self.tb = required_clock(scope)
        self.times = self.pts * float(Fraction(*self.tb))
        self.index = {int(pts): i for i, pts in enumerate(self.pts)}
        self.media = Path(scope["source"]["registered_path"])
        self.media_sha = scope["source"]["sha256"]
        self.source = dict(width=SIZE[0], height=SIZE[1], time_base_num=int(self.tb[0]),
                           time_base_den=int(self.tb[1]))
        if verify_media and digest(self.media) != self.media_sha:
            raise ValueError("Different source media bytes")
        self.parent_path = Path(parent["result"]["path"])
        self.parent_sha = parent["result"]["sha256"]
        self.document_path = Path(parent["source"]["path"])
        self.document = read(self.document_path)
        self.evidence = self.document["programme_skate_evidence"]
        if (digest(self.document_path) != parent["source"]["sha256"]
                or self.evidence["source_video_sha256"] != self.media_sha
                or self.evidence["camera_sha256"] != parent["camera_sha256"]):
            raise ValueError("Programme evidence binding changed")
        self.camera_sha = parent["camera_sha256"]
        self.camera_record = Path(self.document["programme_camera_candidate"]["record"])
        if digest(self.camera_record) != self.document["programme_camera_candidate"]["record_sha256"]:
            raise ValueError("Programme camera record changed")
        self.rows = self.evidence["ordered_contours"]
        self.checks = [row for row in self.rows if row["role"] == "check"]
        self.withheld = np.zeros(len(self.pts), dtype=bool)
        # Include the check source frame and a local protection interval.
        # Correlated old labels are still development checks, never independent GT.
        check_pts = {row["pts"] for row in self.checks}
        check_groups = {(row.get("raw_group_id") or row.get("id")) for row in self.checks} - {None}
        for stamp in check_pts:
            i = self.index[stamp]
            self.withheld[max(0, i - 2):i + 3] = True
        self.fit = defaultdict(list)
        for row in self.rows:
            i = self.index[row["pts"]]
            prov = row.get("provenance") or {}
            # Direct source reads and named upstream anchor PTS stay separate.
            origins = set()
            for key in ("anchor_pts", "source_pts", "start_pts", "end_pts", "left_anchor_pts", "right_anchor_pts"):
                value = prov.get(key) if isinstance(prov, dict) else None
                if isinstance(value, int):
                    origins.add(value)
                elif isinstance(value, list):
                    origins.update(v for v in value if isinstance(v, int))
            if (row["role"] == "fit" and not self.withheld[i] and not origins & check_pts
                    and (row.get("raw_group_id") or row.get("id")) not in check_groups
                    and row["foot"] in (0, 1) and len(row["polyline_uv"]) >= 2):
                self.fit[(i, row["foot"])].append(row)
        with np.load(self.prepared / "input.npz", allow_pickle=False) as packet:
            np.testing.assert_array_equal(packet["pts"], self.pts)
            np.testing.assert_array_equal(packet["shot_ids"], self.shots)
            self.camera_packet = {k: packet[k] for k in ("pts", "image_height", "metres_per_scene_unit")}
        with np.load(self.parent_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["pts"], self.pts)
            np.testing.assert_array_equal(saved["time_base"], self.tb)
            self.old_gear = saved["equipment_uv"]
            self.old_coco = saved["COCO_uv"]
        self.action = read(PROGRAMME / "programme_action_ledger_1107.json")["episodes"]

    def source_keypoints(self):
        """Original 2D anatomical correspondences and per-joint admission."""
        from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path

        path = programme_observation_path(self.prepared, self.document)
        with np.load(path, allow_pickle=False) as packet:
            np.testing.assert_array_equal(packet["pts"], self.pts)
            uv, joint, use = packet["uv"], packet["model_joint_ids"], packet["use"]
        output = self.old_coco.copy()
        direct = np.zeros(output.shape[:-1], dtype=bool)
        finite = np.isfinite(uv).all(axis=-1)
        for label in range(output.shape[1]):
            matched = (joint == label) & use & finite
            good = np.flatnonzero(matched.sum(axis=1) == 1)
            output[good, label] = uv[good, matched[good].argmax(axis=1)]
            direct[good, label] = True
        return output, direct

    def navigation(self):
        joints, direct = self.source_keypoints()
        order = [13, 15, 14, 16]
        return joints[:, order].reshape(-1, 2, 2, 2), direct[:, order].reshape(-1, 2, 2)

    def frames(self, start, stop):
        return source_frames(self.media, self.source, self.pts[start:stop])

    def ranges(self):
        cuts = np.r_[0, np.flatnonzero(self.shots[1:] != self.shots[:-1]) + 1, len(self.pts)]
        return zip(cuts[:-1], cuts[1:])


def geometry(inputs):
    """Recover rigid lower geometry and an explicitly approximate fixed boot shell."""
    with np.load(inputs.parent_path, allow_pickle=False) as f:
        origin, rotation = f["skate_origin_world_mid"], f["skate_R_world_mid"]
        lower = (f["skate_lower_world_mid"][0] - origin[0, :, None]) @ rotation[0]
        edges = ((f["skate_edge_world_mid"][0] - origin[0, :, None, None]) @ rotation[0, :, None])
        sample = np.arange(0, len(origin), 73)
        upper = np.median((f["skate_upper_world_mid"][sample] - origin[sample, :, None])
                          @ rotation[sample], axis=0)
        shape = dict(lower=lower, lower_faces=f["skate_lower_faces"], parts=f["skate_lower_face_parts"],
                     edges=edges, upper=upper, upper_faces=f["skate_upper_faces"],
                     upper_region=f["skate_upper_region_ids"], ratio=f["body_length_ratio"])
    identity = hashlib.sha256()
    for name in ("lower", "lower_faces", "parts", "edges", "upper", "upper_faces", "upper_region"):
        array = np.ascontiguousarray(shape[name])
        identity.update(name.encode() + array.dtype.str.encode() + str(array.shape).encode())
        identity.update(array.tobytes())
    shape["template_sha256"] = np.asarray(identity.hexdigest())
    return shape, origin, rotation


def direct_source(row):
    provenance = row.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    if provenance.get("track_source_anchor_id"):
        return False
    return (row.get("evidence_status") == "observed" or
            (row.get("observation_status") == "observed" and provenance.get("kind") == "direct source visual read"))


def direct_rows(rows, group):
    return [r for r in rows if r["semantic"] in group and direct_source(r)]


def tracked_rows(rows, group):
    return [r for r in rows if r["semantic"] in group and not direct_source(r)]
