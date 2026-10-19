"""FFmpeg-backed audit and PTS-preserving frame extraction.

Packet timestamps are an intake diagnostic, NOT a decoded frame index.
A failed decode/index never silently falls back to frame_number / average_fps.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any
import json
import numpy as np
from .io import dump_json, sha256


def _run(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    if not shutil.which(args[0]):
        raise RuntimeError(f"Missing executable: {args[0]}. Install FFmpeg including ffprobe and add it to PATH.")
    try:
        p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out after {timeout}s: {args[0]}; no completed result was recorded.") from exc
    if p.returncode:
        raise RuntimeError(f"{args[0]} failed ({p.returncode}):\n{p.stderr[-5000:]}")
    return p


def timestamp_summary(times: list[float]) -> dict[str, Any]:
    t = np.asarray(times, dtype=float)
    if not len(t):
        return {"count": 0, "monotonic": None}
    if not np.all(np.isfinite(t)):
        raise ValueError("Timestamps contain missing/nonfinite values")
    delta = np.diff(t)
    vals, counts = np.unique(np.round(delta, 6), return_counts=True)
    return {"count": len(t), "first_s": float(t[0]), "last_s": float(t[-1]),
            "monotonic": bool(np.all(delta > 0)),
            "nonpositive_deltas": int(np.sum(delta <= 0)),
            "median_delta_s": float(np.median(delta)) if len(delta) else None,
            "delta_histogram": [{"delta_s": float(v), "count": int(n)} for v, n in zip(vals, counts)]}


def probe(path: str | Path, packets: bool = False) -> dict[str, Any]:
    p = Path(path).expanduser().resolve(strict=True)
    data = json.loads(_run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)]).stdout)
    video = [s for s in data["streams"] if s.get("codec_type") == "video"]
    if not video:
        raise ValueError("No video stream")
    result = {"schema_version": "1.0", "filename": p.name, "sha256": sha256(p),
              "size_bytes": p.stat().st_size, "probe": data, "packet_diagnostic": None,
              "decoded_frame_index_complete": False}
    if packets:
        packet_data = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
            "-show_entries", "packet=pts,pts_time,dts_time,duration_time,flags", "-of", "json", str(p)]).stdout)
        packet_list = packet_data.get("packets", [])
        ts = [float(x["pts_time"]) for x in packet_list if "pts_time" in x]
        result["packet_diagnostic"] = {**timestamp_summary(ts), "packets_without_pts": len(packet_list) - len(ts),
                                       "not_a_decoded_frame_count": True}
    return result


def index_frames(path: str | Path, out: str | Path, timeout: float = 1800) -> dict[str, Any]:
    p = Path(path).resolve(strict=True)
    meta = probe(p)
    stream = next(s for s in meta["probe"]["streams"] if s["codec_type"] == "video")
    tb = Fraction(stream["time_base"])
    decoded = json.loads(_run(["ffprobe", "-v", "error", "-threads", "2", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=pts,best_effort_timestamp,key_frame,width,height,duration_time",
        "-of", "json", str(p)], timeout).stdout).get("frames", [])
    rows = []
    for n, f in enumerate(decoded):
        field = "pts" if "pts" in f else "best_effort_timestamp"
        if field not in f:
            raise ValueError(f"Decoded frame {n} lacks PTS and best-effort timestamp")
        value = int(f[field])
        rows.append({"source_frame_index": n, "pts": value, "time_base": str(tb),
                     "source_time_s": float(value * tb), "timestamp_source": field,
                     "key_frame": bool(f.get("key_frame")), "width": f.get("width"), "height": f.get("height")})
    summary = timestamp_summary([x["source_time_s"] for x in rows])
    result = {"source_sha256": meta["sha256"], "decoded_frame_index_complete": True,
              "summary": summary, "frames": rows}
    dump_json(out, result)
    return {k: v for k, v in result.items() if k != "frames"}


def sample_window(path: str | Path, out: str | Path, start: float, duration: float,
                  every_n: int = 1, width: int = 960, timeout: float = 600) -> dict[str, Any]:
    """Decode selected source frames. Preserve actual PTS; do not interpolate.

    source_frame_index intentionally stays null until joined to a full decoded index.
    FFmpeg's showinfo `n` here is an OUTPUT sample ordinal, not a source frame number.
    """
    if start < 0 or duration <= 0 or every_n < 1 or width < 64 or width % 2:
        raise ValueError("Require start>=0, duration>0, every_n>=1 and even width>=64")
    source = Path(path).resolve(strict=True)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # -copyts keeps source PTS after input seeking. trim terminates this window;
    # no -r or fps filter is used, so source frames are not duplicated/interpolated.
    vf = f"trim=start={start}:end={start+duration},select='not(mod(n,{every_n}))',scale={width}:-2,showinfo"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-threads", "2", "-ss", str(start), "-copyts",
               "-t", str(duration), "-i", str(source), "-an", "-vf", vf, "-fps_mode", "vfr", "-threads", "1", "-q:v", "3",
               str(out / "frame_%06d.jpg")]
    process = _run(command, timeout)
    (out / "ffmpeg.log").write_text(process.stderr, encoding="utf-8")
    matches = re.findall(r" n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\d.eE]+)", process.stderr)
    files = sorted(out.glob("frame_*.jpg"))
    if len(matches) != len(files) or not files:
        raise RuntimeError(f"PTS/image count mismatch or empty extraction: {len(matches)} vs {len(files)}")
    # Read the filter's actual rational time base rather than trusting rounded pts_time text.
    bases = re.findall(r"config in time_base:\s*(\d+/\d+)", process.stderr)
    if not bases:
        raise RuntimeError("Decoder did not expose filter time_base")
    tb = Fraction(bases[-1])
    samples = [{"sample_index": int(m[0]), "source_frame_index": None,
                "pts": int(m[1]), "time_base": str(tb), "source_time_s": float(int(m[1]) * tb),
                "image": f.name} for m, f in zip(matches, files)]
    record = {"schema_version": "1.0", "source_filename": source.name, "source_sha256": sha256(source),
              "source_frame_index_complete": False, "extraction": "decoded_frames_with_original_pts",
              "every_n": every_n, "requested_start_s": start, "requested_duration_s": duration,
              "samples": samples}
    dump_json(out / "samples.json", record)
    return record
