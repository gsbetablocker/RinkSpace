"""Use the authenticated native PTS inventory, never selected-frame endpoints."""
from __future__ import annotations
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from .camera_graph import SHOT_TO_RIG, initial_graph
from .io_utils import file_sha256, write_json, read_json
from .validation import build_frame_schedule


def programme_spec(inventory, selected_frames):
    source = inventory["source"]
    if source["time_base"] != "1/12800" or source["native_fps"] != "50/1":
        raise ValueError("Unsupported source clock; do not convert legacy 60 fps indices")
    by_shot = {sid: [] for sid in SHOT_TO_RIG}
    for f in selected_frames: by_shot[f["shot_id"]].append(int(f["native_pts"]))
    shots = []
    for raw in inventory["shots"]:
        sid = f's{int(raw["shot_index"]):02d}'
        first, last = int(raw["processing_first_pts"]), int(raw["processing_last_pts"])
        shots.append({"id": sid, "rig_id": SHOT_TO_RIG[sid], "first_pts": first,
            "last_pts": last, "stop_pts_exclusive": last + 256,
            "physical_last_pts": int(raw["last_pts"]), "existing_pts": by_shot[sid],
            "source_seconds": [first / 12800, last / 12800],
            "programme_frames": [(first - inventory["programme_first_pts"]) // 256 + 1,
                                 (last - inventory["programme_first_pts"]) // 256 + 1],
            "native_frame_count": (last - first) // 256 + 1})
    graph = initial_graph({s["id"]: {k:s[k] for k in ("first_pts", "last_pts", "stop_pts_exclusive")}
                           for s in shots}, source["sha256"])
    schedule = build_frame_schedule(shots)
    for row in schedule:
        row["programme_frame_index_1based"] = (row["native_pts"] - inventory["programme_first_pts"]) // 256 + 1
        row["native_video_frame_index_0based"] = row["native_pts"] // 256
        if row["shot_id"] == "s01":
            row["processing_status"] = "awaiting_latest_s01_full_replay"
            row["task"] = "import_latest_s01_and_replay_same_source_rows"
    excluded = [s["explicit_user_excluded_tail"] for s in inventory["shots"]
                if s.get("explicit_user_excluded_tail")]
    return {"schema": "rinkfit.joint_scope.v1", "source": source, "shots": shots,
        "included_frame_count": len(schedule), "excluded_tail": excluded,
        "scope_authority": "official inventory processing_* plus explicit user-excluded tail",
        "physical_cut_is_not_processing_stop": True}, graph, schedule


def verify_video(video, scope):
    """Hash compressed source once; independently verify all video packet PTS."""
    video = Path(video).resolve()
    if not shutil.which("ffprobe"): raise RuntimeError("ffprobe is required on PATH")
    expected = scope["source"]
    actual_sha = file_sha256(video)
    if actual_sha != expected["sha256"]: raise ValueError("Different source video SHA-256")
    args = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_packets",
            "-show_entries", "stream=codec_name,width,height,time_base,avg_frame_rate,nb_frames,duration_ts:packet=pts,duration",
            "-of", "json", str(video)]
    data = json.loads(subprocess.check_output(args, text=True))
    stream = data["streams"][0]
    for key, value in {"width":1920, "height":1080, "time_base":"1/12800", "avg_frame_rate":"50/1"}.items():
        if stream[key] != value: raise ValueError(f"Video {key} mismatch: {stream[key]}")
    pts = [int(p["pts"]) for p in data["packets"]]
    wanted = list(range(0, int(expected["native_frame_count"]) * 256, 256))
    if sorted(pts) != wanted: raise ValueError("Packet PTS missing/duplicated/off lattice")
    included = [p for s in scope["shots"] for p in range(s["first_pts"], s["last_pts"]+1,256)]
    if not set(included).issubset(pts): raise ValueError("Requested programme exposure missing in source")
    return {"schema":"rinkfit.video_audit.v1", "passed":True, "video_path":str(video),
        "sha256":actual_sha, "bytes":video.stat().st_size, "video_stream":stream,
        "native_packet_count":len(pts), "included_native_frames":len(included),
        "packet_pts_verified":True, "decoded_frame_pixels_verified":False}


def control_plan(scope, dt=.4, tau_bound=.02):
    """Allocation proposal, not invented orientation/zoom values."""
    if dt < .02 or not math.isfinite(dt): raise ValueError("Invalid control spacing")
    if not math.isfinite(tau_bound) or tau_bound < 0: raise ValueError("Invalid readout bound")
    plans = []
    for s in scope["shots"]:
        if s["id"] == "s01":
            plans.append({"shot_id":"s01", "action":"import_latest_complete_controls_before_any_refinement",
                          "dt":None,"n_controls":None}); continue
        duration = (s["last_pts"] - s["first_pts"]) / 12800
        # first=-2dt -> valid cubic domain begins -dt and covers RS halo.
        if dt < tau_bound / 2: raise ValueError("Control halo insufficient")
        count = math.ceil((duration + tau_bound / 2) / dt) + 4
        plans.append({"shot_id":s["id"], "dt":dt, "first_control_time":-2*dt,
            "n_controls":count,"valid_domain_seconds":[-dt,(count-4)*dt],
            "values_status":"to_initialise_from_this_shot_native_source",
            "angle_gauge":"local_angle(0)=0; add two shot reference_angles; rig reference shot offsets=0",
            "refinement_trigger":"supported native residual motion/acceleration, never every-frame freedom by default"})
    return {"schema":"rinkfit.control_allocation.v1","rig_f_ref_is_shared":True,"shots":plans}


def decode_selected(video, frames, destination, audit):
    """Sequential native PTS decode; large FFmpeg select trees are unnecessary."""
    import av
    import cv2
    from fractions import Fraction
    destination = Path(destination)
    if destination.exists(): raise ValueError("Choose a new native-frame output directory")
    if not audit.get("passed") or file_sha256(video) != audit["sha256"]:
        raise ValueError("Native source must pass verify-video before decoding")
    count=int(audit["native_packet_count"])
    for f in frames:
        pts=f.get("native_pts")
        if isinstance(pts,bool) or not isinstance(pts,int) or pts%256 or not 0<=pts<count*256:
            raise ValueError("Decode jobs require exact in-range native PTS")
    selected=[{**f,"native_png":f"{i:06d}.png"} for i,f in enumerate(sorted(frames,key=lambda f:f["native_pts"]),1)]
    wanted={f["native_pts"]:f for f in selected}
    if len(wanted)!=len(selected) or not selected: raise ValueError("Duplicate or empty frame jobs")
    destination.mkdir(parents=True)
    identities={}; found=set(); previous=None; inspected=0
    with av.open(str(video)) as source:
        stream=source.streams.video[0]; stream.codec_context.thread_count=4
        if stream.time_base!=Fraction(1,12800): raise ValueError("Unexpected native video time base")
        for frame in source.decode(stream):
            pts=frame.pts
            if pts is None or (previous is not None and pts-previous!=256):
                raise ValueError("Decoded native PTS sequence is not continuous")
            previous=pts; inspected+=1
            if pts in wanted:
                if frame.width!=1920 or frame.height!=1080: raise ValueError("Unexpected native dimensions")
                f=wanted[pts]; pixels=frame.to_ndarray(format="bgr24")
                if not cv2.imwrite(str(destination/f["native_png"]),pixels,[cv2.IMWRITE_PNG_COMPRESSION,2]):
                    raise IOError("PNG write failed")
                f["native_png_sha256"]=file_sha256(destination/f["native_png"])
                f["native_BGR_sha256"]=hashlib.sha256(pixels.tobytes()).hexdigest()
                identities[f"{audit['sha256']}:{pts}"]={"source_video_sha256":audit["sha256"],
                    "native_pts":pts,"native_png_sha256":f["native_png_sha256"],
                    "native_BGR_sha256":f["native_BGR_sha256"],"decode_sequence_verified":True,
                    "decoder":"PyAV sequential exact native PTS"}
                found.add(pts)
            if pts>=selected[-1]["native_pts"]: break
    if found!=set(wanted): raise ValueError("Missing native frame jobs")
    result={"source_video_sha256":audit["sha256"],"frames":selected,
        "pixel_geometry":"native 1920x1080; no resampling",
        "decoded_source_frames_inspected":inspected,"decoder":"PyAV sequential exact native PTS"}
    write_json(destination/"manifest.json",result)
    write_json(destination/"native_frame_identities.json",identities)
    return {"decoded_frames":len(found),"manifest":str(destination/"manifest.json")}
