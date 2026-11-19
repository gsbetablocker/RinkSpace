"""Source and frozen model bindings for Ankara programme fitting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from fractions import Fraction

import av
import numpy as np

DEFAULT_PROGRAMME = Path(__file__).resolve().parents[1]
SCOPE_PATH = Path(os.environ.get("RINKSPACE_ANKARA_PROGRAMME_SCOPE", DEFAULT_PROGRAMME / "rebuild_scope.json"))
PROGRAMME = SCOPE_PATH.parent
EVENT = DEFAULT_PROGRAMME.parents[1]
ROOT = EVENT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rinkspace.storage import configured_data_root

DATA = configured_data_root(ROOT)
OUTPUT = Path(os.environ.get("RINKSPACE_ANKARA_PROGRAMME_OUTPUT",
              DATA / "events/ankara_2026/programmes/chen_xi_jin_sp/fit_1115"))
JOINT = EVENT / "tools/joint_review_1113"
TB = Fraction(1, 12800)
STEP = 256
WIDTH, HEIGHT = 1920, 1080


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, item):
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def save(destination, **data):
    temp = destination.with_suffix(".writing.npz")
    np.savez_compressed(temp, **data)
    temp.replace(destination)


def visibility_packet(folder, shot_id):
    """Resolve a packed source silhouette, including a manifest selection."""
    index=folder/"binding.json"
    if index.is_file():
        return Path(read(index)["shots"][shot_id]["packet"])
    return folder/f"{shot_id}.npz"


def observation_packet(folder, shot_id):
    """Read a revised shot packet or the explicitly bound upstream packet."""
    own=folder/f"{shot_id}.npz"
    if own.is_file():
        return own
    references=folder/"shot_packets.json"
    if references.is_file():
        return Path(read(references)[shot_id])
    return own


def scope():
    record = read(SCOPE_PATH)
    required = set(record["required_shot_ids"])
    record["shots"] = [s for s in record["shots"] if s["id"] in required]
    assert record["required_shot_ids"] == [s["id"] for s in record["shots"]]
    assert all(s["camera_assignment_human_confirmed"] for s in record["shots"])
    return record


def source_pts(shot):
    span = shot["processing"]
    pts = np.arange(span["first_pts"], span["stop_pts_exclusive"], STEP, dtype=np.int64)
    assert len(pts) == span["native_frame_count"] and pts[-1] == span["last_pts"]
    return pts


def source_frames(path, points):
    """Decode the selected real integer PTS, preserving presentation order."""
    points = np.asarray(points, dtype=np.int64)
    assert len(points) and np.all(np.diff(points) > 0)
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if stream.time_base != TB or (stream.width, stream.height) != (WIDTH, HEIGHT):
            raise ValueError("Unregistered source dimensions or time base")
        stream.thread_type = "AUTO"
        stream.thread_count = 4
        video.seek(int(points[0]), stream=stream, backward=True)
        i = 0
        for frame in video.decode(stream):
            target = points[i]
            if frame.pts < target:
                continue
            if frame.pts != target or frame.time_base != TB:
                raise ValueError(f"Source PTS {target} missing before {frame.pts}")
            yield int(frame.pts), frame
            i += 1
            if i == len(points):
                return
        raise ValueError(f"Only decoded {i} of {len(points)} requested PTS")


def frozen_scene(record=None, *, verify=False):
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import ParameterState, SceneParameterModel
    from rinkspace.session.gpu_geometry import NativeSceneGPU

    link = (record or scope())["geometry_binding"]
    scene_path = ROOT / link["structure"]["path"]
    state_path = ROOT / link["snapshot"]["path"]
    library = Path(link["native_library"]["path"])
    if verify:
        for name, path in (("structure", scene_path), ("snapshot", state_path), ("native_library", library)):
            if sha(path) != link[name]["sha256"]:
                raise ValueError(f"Frozen {name} SHA-256 differs")
    document = read(scene_path)
    model = SceneParameterModel(document, scene_path, optimize_scene=True)
    with np.load(state_path, allow_pickle=False) as arrays:
        if (arrays["layout_revision"].item() != model.initial_state.layout["layout_revision"]
                or arrays["scene_values_sha256"].item() != link["value_sha256"]):
            raise ValueError("Frozen layout or numeric revision differs")
        values = arrays["scene_values"]
    if hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest() != link["value_sha256"]:
        raise ValueError("Frozen numeric values changed")
    state = ParameterState(model.initial_state.layout, values)
    native = NativeSceneGPU(scene_path, NativeCore(library), parameter_state=state)
    if len(native.paths) != 922 or native.unlocalized:
        raise ValueError("The frozen scene inventory differs")
    return native
