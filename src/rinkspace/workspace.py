"""Local source registry, decoded PTS index and explicitly mapped browser proxies.

All source files are opened read-only. Project artifacts live under outputs/workspace.
The PyAV dependency supplies AV1 decoding and H.264 encoding without a
system FFmpeg executable. Automatic shot/role results are candidates, not labels.
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import dump_json, load_json, sha256
from .media import timestamp_summary
from .storage import AssetStore


def _av():
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Decoded PTS indexing requires the declared PyAV dependency: pip install av") from exc
    return av


def load_config(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    local = root / "configs/local.json"
    path = local if local.exists() else root / "configs/project.example.json"
    config = load_json(path)
    config["config_path"] = str(path)
    return config


def _key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key):
        raise ValueError("Source key must contain only letters, numbers, underscores or hyphens")
    return key


def source_directory(root: str | Path, key: str) -> Path:
    key = _key(key)
    return AssetStore(root).directory(f"outputs/workspace/{key}", f"cache/workspace/{key}")


def source_path(root: str | Path, key: str) -> Path:
    root = Path(root).resolve()
    item = load_config(root).get("sources", {}).get(_key(key))
    if not item:
        raise KeyError(f"Unknown source: {key}")
    return AssetStore(root).source(item["path"])


def import_source(root: str | Path, path: str | Path, url: str | None = None,
                  key: str | None = None, title: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    path = AssetStore(root).source(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local source does not exist: {path}")
    key = _key(key or re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:100])
    config = load_config(root)
    config.pop("config_path", None)
    try:
        stored_path = path.relative_to(root).as_posix()
    except ValueError:
        stored_path = str(path)
    config.setdefault("sources", {})[key] = {
        "path": stored_path, "role": "broadcast_with_unreviewed_roles",
        "read_only": True, "url": url, "title": title or path.stem,
    }
    config["configuration_status"] = "active_local_workspace"
    dump_json(root / "configs/local.json", config)
    return {"key": key, **config["sources"][key]}


def load_sources(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    sources = []
    for key, entry in load_config(root).get("sources", {}).items():
        path = source_path(root, key)
        meta_path = source_directory(root, key) / "metadata.json"
        metadata = load_json(meta_path) if meta_path.exists() else None
        metadata_source = AssetStore(root).source(metadata["source_path"]) if metadata and metadata.get("source_path") else None
        if metadata and (not path.is_file() or metadata_source != path
                         or metadata.get("size_bytes") != path.stat().st_size
                         or metadata.get("source_mtime_ns", path.stat().st_mtime_ns) != path.stat().st_mtime_ns):
            metadata = None
        shots_path = source_directory(root, key) / "shots.automatic.json"
        shots = load_shots(root, key)["shots"] if shots_path.exists() and metadata else []
        sources.append({"key": key, **entry, "absolute_path": str(path), "shots": shots,
                        "exists": path.is_file(), "prepared": bool(metadata), "metadata": metadata})
    return sources


def load_index(root: str | Path, key: str) -> dict[str, Any]:
    metadata_path = source_directory(root, key) / "metadata.json"
    if metadata_path.exists():
        meta = load_json(metadata_path)
        path = source_path(root, key)
        metadata_source = AssetStore(root).source(meta["source_path"]) if meta.get("source_path") else None
        if (not path.is_file() or metadata_source != path
                or meta.get("size_bytes") != path.stat().st_size
                or meta.get("source_mtime_ns", path.stat().st_mtime_ns) != path.stat().st_mtime_ns):
            raise ValueError("Source identity changed; rebuild its decoded index before browsing")
    return load_json(source_directory(root, key) / "frame_index.json")


def source_frame_meta(root: str | Path, key: str, source_frame_index: int) -> dict[str, Any]:
    index = load_index(root, key)
    if not 0 <= source_frame_index < len(index["frames"]):
        raise IndexError("Source frame index out of range")
    meta = load_json(source_directory(root, key) / "metadata.json")
    return {**index["frames"][source_frame_index], "video_id": index["video_id"],
            "fps": meta["declared_average_rate"], "fps_is_stream_declaration": True,
            "frame_count": len(index["frames"])}


def _role_candidates(ice_fraction: float) -> tuple[str, list[dict[str, Any]]]:
    # There is no trained broadcast semantic model in this delivery. Ice occupancy
    # is useful for triage, but cannot identify live vs replay by itself.
    if ice_fraction >= .23:
        primary = "programme_candidate"
        scores = [(.60, "programme"), (.25, "replay"), (.15, "nonprogramme")]
    else:
        primary = "nonprogramme_candidate"
        scores = [(.65, "nonprogramme"), (.20, "replay"), (.15, "programme")]
    return primary, [{"role": role, "heuristic_score": score,
                      "reason": "ice_occupancy_only; replay requires human review"}
                     for score, role in scores]


def _make_shots(rows: list[dict], cuts: list[dict], ice_values: list[float]) -> list[dict]:
    starts = [0] + [c["source_frame_index"] for c in cuts]
    shots = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(rows)
        ice = float(np.median(ice_values[start:end]))
        role, candidates = _role_candidates(ice)
        shots.append({"shot_id": f"shot_{i:04d}", "start_frame": start,
                      "end_frame_exclusive": end, "start_s": rows[start]["source_time_s"],
                      "end_s": rows[end]["source_time_s"] if end < len(rows) else rows[-1]["source_time_s"],
                      "start_pts": rows[start]["pts"], "role": role,
                      "role_candidates": candidates, "review_status": "unreviewed",
                      "ice_fraction_median": ice,
                      "boundary_status": "automatic_cut_candidate" if i else "source_start",
                      "cut_evidence": cuts[i-1] if i else None})
    return shots


def prepare_source(root: str | Path, key: str, make_proxy: bool = True,
                   proxy_width: int = 960, proxy_every_n: int = 2,
                   force: bool = False, progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Decode the COMPLETE source, index true display-order PTS, detect cut candidates.

    Proxy every_n is recorded per output frame; browser time is never used as a
    source frame number. Displayed source PTS remain those of the original file.
    """
    av = _av()
    root, path = Path(root).resolve(), source_path(root, key)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    if proxy_every_n < 1 or proxy_width < 64 or proxy_width % 2:
        raise ValueError("Require proxy_every_n >= 1 and even proxy_width >= 64")
    out = source_directory(root, key)
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(path)
    if (out / "metadata.json").exists() and not force:
        previous = load_json(out / "metadata.json")
        if (previous.get("video_id") == digest and previous.get("decoded_frame_index_complete")
                and (not make_proxy or (out / "proxy.mp4").is_file())):
            previous.update(source_path=str(path), source_mtime_ns=path.stat().st_mtime_ns)
            dump_json(out / "metadata.json", previous)
            return previous
    rows, cuts, ice_values, mapping = [], [], [], []
    previous_gray = previous_hist = None
    last_cut = -1000.0
    proxy_container = proxy_stream = None
    first_time = None
    thumbs = []
    next_thumbnail = 0.0
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        nominal_rate = stream.average_rate or stream.base_rate or Fraction(25)
        metadata = {"schema_version": "1.1", "video_id": digest, "source_sha256": digest,
                    "filename": path.name, "source_path": str(path), "size_bytes": path.stat().st_size,
                    "source_mtime_ns": path.stat().st_mtime_ns,
                    "codec": stream.codec_context.name, "width": stream.width, "height": stream.height,
                    "stream_time_base": str(stream.time_base), "declared_average_rate": str(nominal_rate),
                    "declared_frame_count": stream.frames, "source_read_only": True,
                    "decoded_frame_index_complete": False,
                    "cut_detector": "adjacent grayscale change + HSV histogram, candidate only",
                    "role_detector": "ice occupancy triage; live/replay cannot be distinguished automatically"}
        if make_proxy:
            proxy_container = av.open(str(out / "proxy.part.mp4"), mode="w", options={"movflags": "+faststart"})
            proxy_stream = proxy_container.add_stream("libx264", rate=nominal_rate / proxy_every_n)
            proxy_stream.width = min(proxy_width, stream.width)
            proxy_stream.height = int(round(stream.height * proxy_stream.width / stream.width / 2)) * 2
            proxy_stream.pix_fmt = "yuv420p"
            proxy_stream.time_base = Fraction(1, 90000)
            proxy_stream.codec_context.time_base = Fraction(1, 90000)
            proxy_stream.options = {"crf": "24", "preset": "veryfast"}
        try:
            for frame_n, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError(f"Source decoded frame {frame_n} has no PTS; index cannot be completed")
                frame_tb = Fraction(frame.time_base)
                exact_time = frame.pts * frame_tb
                time_s = float(exact_time)
                if first_time is None:
                    first_time = exact_time
                row = {"source_frame_index": frame_n, "pts": int(frame.pts), "time_base": str(frame_tb),
                       "source_time_s": time_s, "timestamp_source": "decoded_frame_pts",
                       "key_frame": bool(frame.key_frame), "width": frame.width, "height": frame.height}
                rows.append(row)
                tiny = frame.reformat(width=160, height=90, format="rgb24").to_ndarray()
                gray = cv2.cvtColor(tiny, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(tiny, cv2.COLOR_RGB2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                hist /= max(float(hist.sum()), 1)
                ice_values.append(float(np.mean((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 145))))
                if previous_gray is not None:
                    difference = float(np.mean(cv2.absdiff(previous_gray, gray)) / 255)
                    histogram = float(np.sum(np.abs(previous_hist - hist)) / 2)
                    if ((difference > .055 and histogram > .16) or difference > .31) and time_s-last_cut > .35:
                        cuts.append({"source_frame_index": frame_n, "source_time_s": time_s,
                                     "previous_source_frame_index": frame_n-1, "mean_difference": difference,
                                     "histogram_distance": histogram,
                                     "transition_interval_s": [rows[-2]["source_time_s"], time_s],
                                     "status": "candidate"})
                        last_cut = time_s
                previous_gray, previous_hist = gray, hist
                if time_s >= next_thumbnail:
                    thumb = frame.reformat(width=320, height=180, format="rgb24").to_ndarray()
                    thumbs.append((thumb, row))
                    next_thumbnail = time_s + 10
                if make_proxy and frame_n % proxy_every_n == 0:
                    proxy_frame = frame.reformat(width=proxy_stream.width, height=proxy_stream.height, format="yuv420p")
                    proxy_frame.pts = round((exact_time - first_time) * 90000)
                    proxy_frame.time_base = Fraction(1, 90000)
                    mapping.append({"proxy_frame_index": len(mapping), "expected_proxy_pts": proxy_frame.pts,
                                    "proxy_time_s": float(proxy_frame.pts / 90000),
                                    "source_frame_index": frame_n, "source_pts": row["pts"],
                                    "source_time_base": str(frame_tb), "source_time_s": time_s})
                    for packet in proxy_stream.encode(proxy_frame):
                        proxy_container.mux(packet)
                if progress and frame_n % 1000 == 0:
                    progress({"source": key, "decoded_frames": frame_n+1, "source_time_s": time_s})
            if make_proxy:
                for packet in proxy_stream.encode():
                    proxy_container.mux(packet)
        finally:
            if proxy_container is not None:
                proxy_container.close()
    if not rows:
        raise ValueError("No decoded video frames")
    summary = timestamp_summary([r["source_time_s"] for r in rows])
    dump_json(out / "frame_index.json", {"source_sha256": digest, "video_id": digest,
              "decoded_frame_index_complete": True, "summary": summary, "frames": rows})
    shots = _make_shots(rows, cuts, ice_values)
    dump_json(out / "shots.automatic.json", {"video_id": digest, "shots": shots,
              "cuts": cuts, "status": "candidates_require_review"})
    if make_proxy:
        encoded_rows = []
        with av.open(str(out / "proxy.part.mp4")) as encoded:
            for n, frame in enumerate(encoded.decode(video=0)):
                if n >= len(mapping):
                    raise RuntimeError("Encoded proxy inserted extra frames")
                t = float(frame.pts * frame.time_base)
                if abs(t-mapping[n]["proxy_time_s"]) > 1/90000 + 1e-6:
                    raise RuntimeError(f"Proxy timestamp drift at decoded frame {n}")
                encoded_rows.append({**mapping[n], "proxy_pts": frame.pts,
                                     "proxy_time_base": str(frame.time_base), "proxy_time_s": t})
        if len(encoded_rows) != len(mapping):
            raise RuntimeError("Encoded proxy dropped frames")
        (out / "proxy.part.mp4").replace(out / "proxy.mp4")
        dump_json(out / "proxy_mapping.json", {"video_id": digest, "mapping_complete": True,
                  "source_frame_stride": proxy_every_n, "proxy_is_not_source": True,
                  "audio": "omitted; visual review proxy", "frames": encoded_rows})
    columns, cell_w, cell_h = 4, 320, 208
    sheet = Image.new("RGB", (columns*cell_w, math.ceil(len(thumbs)/columns)*cell_h), "#111827")
    draw = ImageDraw.Draw(sheet)
    for n, (thumb, row) in enumerate(thumbs):
        x, y = (n % columns)*cell_w, (n // columns)*cell_h
        sheet.paste(Image.fromarray(thumb), (x, y))
        draw.text((x+5, y+184), f"source {row['source_time_s']:.3f}s | frame {row['source_frame_index']}", fill="white")
    sheet.save(out / "contact_sheet.jpg", quality=88)
    metadata.update({"decoded_frame_index_complete": True, "frame_count": len(rows),
                     "timing": summary, "duration_s": summary["last_s"]-summary["first_s"],
                     "shot_count": len(shots), "proxy": "proxy.mp4" if make_proxy else None,
                     "proxy_frame_count": len(mapping) if make_proxy else 0,
                     "proxy_mapping": "proxy_mapping.json" if make_proxy else None,
                     "frame_index": "frame_index.json", "contact_sheet": "contact_sheet.jpg"})
    dump_json(out / "metadata.json", metadata)
    if make_proxy:
        refine_shot_candidates(root, key)
        metadata = load_json(out / "metadata.json")
    return metadata


def get_frame(root: str | Path, key: str, source_frame_index: int, width: int = 960) -> Path:
    """Seek to a source keyframe and match the requested decoded PTS exactly."""
    av = _av()
    index = load_index(root, key)
    if not 0 <= source_frame_index < len(index["frames"]):
        raise IndexError("Source frame index out of range")
    row = index["frames"][source_frame_index]
    output = source_directory(root, key) / "frames" / f"source_{source_frame_index:07d}_w{width}.jpg"
    if (output.exists() and output.with_suffix(".json").exists()
            and load_json(output.with_suffix(".json")).get("video_id") == index["video_id"]):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    exact = row["pts"] * Fraction(row["time_base"])
    path = source_path(root, key)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Duplicate PTS are legitimate source identities; seek cannot disambiguate
        # them, so decode from source start if the input is not strictly monotonic.
        monotonic = index["summary"].get("monotonic", False)
        if monotonic:
            container.seek(int(exact / stream.time_base), stream=stream, backward=True, any_frame=False)
        for n, frame in enumerate(container.decode(stream)):
            match = frame.pts is not None and frame.pts * Fraction(frame.time_base) == exact
            if (monotonic and match) or (not monotonic and n == source_frame_index):
                image = frame.to_image()
                if width and width < image.width:
                    image = image.resize((width, round(image.height*width/image.width)), Image.Resampling.LANCZOS)
                image.save(output, quality=94)
                dump_json(output.with_suffix(".json"), {"video_id": index["video_id"], **row,
                          "output_image": output.name, "output_size": list(image.size),
                          "coordinate_space": "scaled_image_px", "scale_from_source": image.width / frame.width})
                return output
    raise RuntimeError(f"Exact source frame/PTS not recovered: {source_frame_index} / {row['pts']}")


def load_shots(root: str | Path, key: str) -> dict[str, Any]:
    out = source_directory(root, key)
    corrected = out / "shots.corrected.json"
    automatic = load_json(out / "shots.automatic.json")
    if corrected.exists():
        candidate = load_json(corrected)
        if candidate.get("video_id") == automatic.get("video_id"):
            return candidate
    return automatic


def refine_shot_candidates(root: str | Path, key: str, observations: list[dict] | None = None) -> dict[str, Any]:
    """Group transitions and detect same-colour camera cuts using the mapped proxy.

    Candidate time precision is explicitly the interval between adjacent proxy
    samples. No proxy identity is substituted for a source identity. Thresholds
    were fixed on the JIN development video, never on the OKADA validation clip.
    """
    out = source_directory(root, key)
    index = load_index(root, key)
    source_rows = index["frames"]
    mapping = load_json(out / "proxy_mapping.json")["frames"]
    if observations is None:
        av = _av()
        observations = []
        previous = None
        with av.open(str(out / "proxy.mp4")) as container:
            for n, frame in enumerate(container.decode(video=0)):
                rgb = frame.reformat(width=160, height=90, format="rgb24").to_ndarray()
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                histogram /= max(float(histogram.sum()), 1)
                diff = 0. if previous is None else float(np.mean(cv2.absdiff(previous[0], gray))/255)
                hist = 0. if previous is None else float(np.abs(previous[1]-histogram).sum()/2)
                observations.append({"t": mapping[n]["source_time_s"], "diff": diff, "hist": hist,
                    "ice": float(np.mean((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 145))),
                    "overlay": float(np.mean(hsv[:14, :40, 1] > 80))})
                previous = gray, histogram
    if len(observations) != len(mapping):
        raise ValueError("Scene analysis and proxy mapping lengths disagree")
    candidates = [n for n, r in enumerate(observations) if (r["diff"] > .055 and r["hist"] > .16) or r["diff"] > .31]
    groups: list[list[int]] = []
    for n in candidates:
        if groups and observations[n]["t"]-observations[groups[-1][-1]]["t"] < .8:
            groups[-1].append(n)
        else:
            groups.append([n])
    cuts = []
    for group in groups:
        strongest = max(group, key=lambda n: observations[n]["diff"])
        r, m = observations[strongest], mapping[strongest]
        lower = max(0, group[0]-1)
        upper = group[-1]
        cuts.append({"source_frame_index": m["source_frame_index"], "source_time_s": m["source_time_s"],
                     "previous_source_frame_index": mapping[max(0,strongest-1)]["source_frame_index"],
                     "mean_difference": r["diff"], "histogram_distance": r["hist"],
                     "transition_interval_s": [mapping[lower]["source_time_s"],mapping[upper]["source_time_s"]],
                     "transition_source_frames": [mapping[lower]["source_frame_index"],mapping[upper]["source_frame_index"]],
                     "confidence": .9 if r["diff"] > .15 else .65, "status": "candidate",
                     "boundary_type": "multi_frame_transition_candidate" if len(group)>1 else "hard_cut_candidate",
                     "precision": "bounded_by_mapped_proxy_samples"})
    times = np.asarray([r["t"] for r in observations])
    ice = np.interp([r["source_time_s"] for r in source_rows], times, [r["ice"] for r in observations]).tolist()
    shots = _make_shots(source_rows, cuts, ice)
    # Saturated top-left score graphics are a transparent broadcast-specific
    # heuristic. Their disappearance can only propose a replay, never confirm it.
    step = float(np.median(np.diff(times))) if len(times)>1 else .04
    smooth_size = max(1, int(4/step))
    overlay = cv2.blur(np.asarray([r["overlay"] for r in observations],np.float32).reshape(-1,1), (1,smooth_size)).ravel()
    graphic_present = (overlay > .15) & (np.asarray([r["ice"] for r in observations]) > .30)
    runs = []
    begin = None
    for n, present in enumerate(np.r_[graphic_present, False]):
        if present and begin is None:
            begin = n
        elif not present and begin is not None:
            if times[n-1]-times[begin] >= 30:
                runs.append((times[begin],times[n-1]))
            begin = None
    graphics_span = [float(min(r[0] for r in runs)),float(max(r[1] for r in runs))] if runs else None
    for shot in shots:
        shot["confidence"] = .5
        if graphics_span and shot["start_s"] > graphics_span[1]+8 and shot["ice_fraction_median"] >= .23:
            shot["role"] = "replay_candidate"
            shot["role_candidates"] = [{"role":"replay","heuristic_score":.55,
                 "reason":"ice visible after sustained score graphics disappear; may include post-programme material"},
                 {"role":"nonprogramme","heuristic_score":.30,"reason":"post-programme skating is ambiguous"},
                 {"role":"programme","heuristic_score":.15,"reason":"no semantic classifier"}]
        elif graphics_span and shot["end_s"] < graphics_span[0]-3:
            shot["role"] = "nonprogramme_candidate"
    result = {"video_id": index["video_id"], "shots": shots, "cuts": cuts,
              "status": "candidates_require_review", "graphics_span_candidate_s": graphics_span,
              "method": {"mean_difference_min":.055,"histogram_distance_min":.16,
                         "transition_group_gap_s":.8,"role_semantics":"heuristics; human correction required",
                         "development_source":"jin", "frozen_before_okada_evaluation":True}}
    dump_json(out / "shots.automatic.json", result)
    dump_json(out / "scene_observations.json", {"video_id":index["video_id"],"proxy_mapping":"proxy_mapping.json",
                                               "observations":observations})
    meta = load_json(out / "metadata.json")
    meta.update({"shot_count":len(shots), "cut_detector":"mapped-proxy adjacent differences + histogram with grouped transitions",
                 "role_detector":"ice occupancy and sustained score-graphic sequence; candidates only"})
    dump_json(out / "metadata.json",meta)
    return result


def save_shot_labels(root: str | Path, key: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    index = load_index(root, key)
    rows = index["frames"]
    allowed = {"programme", "replay", "nonprogramme", "unknown", "programme_candidate", "replay_candidate", "nonprogramme_candidate"}
    previous_end = 0
    cleaned = []
    for n, shot in enumerate(shots):
        start, end = int(shot["start_frame"]), int(shot["end_frame_exclusive"])
        if start != previous_end or not start < end <= len(rows):
            raise ValueError("Corrected shots must partition all source frames without gaps or overlaps")
        if shot.get("role") not in allowed:
            raise ValueError("Unknown programme role")
        cleaned.append({**shot, "shot_id": f"shot_{n:04d}", "start_frame": start,
                        "end_frame_exclusive": end, "start_s": rows[start]["source_time_s"],
                        "end_s": rows[end]["source_time_s"] if end < len(rows) else rows[-1]["source_time_s"],
                        "review_status": "user_corrected", "provenance": {"type": "user"}})
        previous_end = end
    if previous_end != len(rows):
        raise ValueError("Corrected shots must cover the full source")
    result = {"video_id": index["video_id"], "shots": cleaned, "status": "user_corrected",
              "automatic_source": "shots.automatic.json"}
    dump_json(source_directory(root, key) / "shots.corrected.json", result)
    return result


def extract_window(root: str | Path, key: str, out: str | Path, start_s: float,
                   end_s: float, every_n: int = 5, width: int = 960) -> dict[str, Any]:
    """Create an exact-index-joined sample directory for registration, no seek guesses."""
    av = _av()
    if end_s <= start_s or every_n < 1:
        raise ValueError("Require end_s > start_s and every_n >= 1")
    index = load_index(root, key)
    selected = [r for r in index["frames"] if start_s <= r["source_time_s"] < end_s][::every_n]
    if not selected:
        raise ValueError("Requested time interval contains no source frames")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("frame_*.jpg")):
        raise FileExistsError(f"Sample directory already contains frames: {out}")
    lookup = {(r["pts"], r["time_base"]): r for r in selected}
    by_source_index = {r["source_frame_index"]: r for r in selected}
    monotonic = index["summary"].get("monotonic", False)
    samples = []
    with av.open(str(source_path(root, key))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if monotonic:
            container.seek(int(Fraction(selected[0]["pts"]) * Fraction(selected[0]["time_base"]) / stream.time_base),
                           stream=stream, backward=True, any_frame=False)
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                continue
            time_s = float(frame.pts * frame.time_base)
            if (monotonic and time_s >= end_s) or (not monotonic and decoded_index > selected[-1]["source_frame_index"]):
                break
            match = lookup.get((int(frame.pts), str(frame.time_base))) if monotonic else by_source_index.get(decoded_index)
            if match is None:
                continue
            image = frame.to_image()
            if width and image.width != width:
                image = image.resize((width, round(image.height*width/image.width)), Image.Resampling.LANCZOS)
            name = f"frame_{len(samples):06d}.jpg"
            image.save(out / name, quality=95)
            samples.append({"sample_index": len(samples), **match, "image": name,
                            "image_width": image.width, "image_height": image.height,
                            "image_scale_from_source": image.width/frame.width})
    if len(samples) != len(selected):
        raise RuntimeError(f"Only recovered {len(samples)}/{len(selected)} requested source identities")
    record = {"schema_version": "1.1", "source_filename": source_path(root, key).name,
              "source_sha256": index["video_id"], "source_key": key,
              "source_frame_index_complete": True, "extraction": "decoded_frames_joined_to_full_pts_index",
              "requested_start_s": start_s, "requested_duration_s": end_s-start_s,
              "every_n": every_n, "samples": samples}
    dump_json(out / "samples.json", record)
    return record


def sample_source_window(root: str | Path, key: str, start: float, end: float,
                         every_n: int, width: int, out: str | Path) -> dict[str, Any]:
    return extract_window(root, key, out, start, end, every_n, width)
