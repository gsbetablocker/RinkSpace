"""Authenticate a historical camera against its original observations and clock."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import numpy as np
from ..session.dataset import digest, read_json, require


def resolve_d78_source_binding(model_path, source_binding_path, *, scope_path, shot_id):
    model_path = Path(model_path).resolve(strict=True)
    owner = Path(source_binding_path).resolve(strict=True)
    model, binding, scope = map(read_json, (model_path, owner, scope_path))
    require(binding.get("schema") == "rinkspace.d78_source_binding.v1", "Explicit d78 source binding required")
    require(binding.get("model_sha256") == digest(model_path), "d78 binding model identity differs")
    hashes = model.get("source_hashes", {})
    require(model.get("source_identity") == hashes.get("observations.npz") and bool(model.get("source_identity")),
            "d78 source identity must authenticate the original observations")
    proofs = {}
    def asset(key, expected):
        record = binding.get(key, {})
        require(isinstance(record, dict) and bool(record.get("path")) and record.get("sha256") == expected,
                "Missing or conflicting d78 source proof", proof=key)
        target = (owner.parent / record["path"]).resolve(strict=True)
        require(digest(target) == expected, "Changed d78 source proof", proof=key, path=str(target))
        proofs[key] = {"path": str(target), "sha256": expected}
        return target
    observations_path = asset("observations", model["source_identity"])
    media = read_json(asset("media", hashes.get("media.json")))
    original = read_json(asset("input_manifest", hashes.get("input_manifest.json")))
    matches = [s for s in scope["shots"] if s["id"] == shot_id]
    require(len(matches) == 1 and shot_id in scope["required_shot_ids"], "Unknown d78 scope shot")
    shot = matches[0]
    historical_id = original["shot_id"]
    canonical_id = f'{scope["event_id"]}/{scope["programme_id"]}/shot{int(shot_id[1:])}' if shot_id.startswith("s") and shot_id[1:].isdigit() else shot_id
    require(historical_id in (shot_id, canonical_id), "Original d78 shot identity differs")
    require(binding.get("shot_id") == shot_id and binding.get("rig_id") == shot["rig_id"], "d78 shot/rig binding differs")
    require(binding.get("source_id") == scope["source"]["id"], "d78 source ID differs")
    tb = Fraction(*original["time_base"])
    clock = scope["source_clock"]
    require(tb > 0 and tb == Fraction(clock["time_base_num"], clock["time_base_den"]), "d78 source timebase differs")
    width, height = original["image_wh"]
    require([width, height] == [scope["source"]["width"], scope["source"]["height"]], "d78 source dimensions differ")
    frames = media.get("frames", [])
    require(bool(frames), "d78 original media has no exposure evidence")
    video_sha = scope["source"]["sha256"]
    pts = []
    for frame in frames:
        value = frame["native_pts"]
        require(type(value) is int and value == frame["actual_pts"], "d78 media native/actual PTS differs")
        require(frame["source_video_sha256"] == video_sha and Fraction(frame["time_base"]) == tb,
                "d78 original video/timebase differs")
        pts.append(value)
    require(all(b > a for a, b in zip(pts, pts[1:])), "d78 original PTS are not strictly ordered")
    reference = original["pts_origin"]
    require(type(reference) is int and binding.get("original_reference_pts") == reference and reference == pts[0],
            "d78 original camera time origin differs")
    first, stop = shot["processing"]["first_pts"], shot["processing"]["stop_pts_exclusive"]
    step = original["pts_step"]
    require(type(step) is int and step > 0 and all(b-a == step for a,b in zip(pts,pts[1:])), "d78 source PTS step differs")
    require(first in pts and stop <= pts[-1]+step and stop > first and (stop-first) % step == 0,
            "Selected d78 window is outside original source support")
    with np.load(observations_path, allow_pickle=False) as arrays:
        for key, desc in original.get("arrays", {}).items():
            require(key in arrays, "d78 original observation array missing", array=key)
            array = arrays[key]
            import hashlib
            require(array.dtype.str == desc["dtype_str"] and list(array.shape) == desc["shape"] and
                    hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == desc["sha256_c_contiguous_array_bytes"],
                    "d78 original observation array differs", array=key)
        require(np.isin(arrays["native_pts"], pts).all(), "d78 observation PTS absent from original media")
    video = asset("source_video", video_sha)
    import av
    with av.open(str(video)) as container:
        require(len(container.streams.video) == 1, "Ambiguous d78 source video stream")
        stream = container.streams.video[0]
        require((stream.width, stream.height) == (width, height) and Fraction(stream.time_base) == tb,
                "Actual d78 source dimensions/timebase differs")
    return {"schema": "rinkspace.authenticated_d78_source.v1", "verified": True,
            "model_sha256": digest(model_path), "observations_sha256": model["source_identity"],
            "source_id": binding["source_id"], "source_video_sha256": video_sha, "shot_id": shot_id,
            "rig_id": shot["rig_id"], "original_shot_id": historical_id, "original_reference_pts": reference,
            "width": width, "height": height, "time_base_num": tb.numerator, "time_base_den": tb.denominator,
            "original_first_pts": pts[0], "original_stop_pts_exclusive": pts[-1]+step,
            "binding_sha256": digest(owner), "proofs": proofs}
