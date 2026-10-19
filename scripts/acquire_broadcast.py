"""Acquire and inspect the explicitly authorized ISU full broadcast.

Original format 399 is retained byte-for-byte. Scout timestamps come from
decoded source PTS after a backward keyframe seek, never an FPS frame guess.
This script does not claim a complete decoded-frame index or calibration.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import concurrent.futures
import shutil
import time
import urllib.request
import io

import av
import numpy as np
from PIL import Image, ImageDraw

from rinkspace.io import dump_json, sha256

VIDEO_ID = "oxXu1avhrvs"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
FORMAT_SIZES = {"399": 3392134310, "134": 452991174}


class CachedSourceReader(io.RawIOBase):
    """Read only already validated original byte ranges; absent ranges fail."""
    def __init__(self, root: Path, format_id: str):
        self.position = 0
        self.total = FORMAT_SIZES[format_id]
        self.chunk = 4 * 1024 * 1024
        self.parts = _resolve_asset(root, root / "outputs/broadcast_scout" / f"download_f{format_id}_parts")
        self.used: set[int] = set()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.total + offset
        return self.position

    def read(self, size=-1):
        end = min(self.total, self.position + (size if size >= 0 else self.total))
        pieces = []
        while self.position < end:
            base = self.position // self.chunk * self.chunk
            path = self.parts / f"{base:012d}.part"
            expected = min(self.chunk, self.total - base)
            if not path.exists() or path.stat().st_size != expected:
                raise OSError(f"Original byte range not downloaded yet: {base}")
            self.used.add(base)
            data = path.read_bytes()
            count = min(end - self.position, len(data) - (self.position - base))
            pieces.append(data[self.position - base:self.position - base + count])
            self.position += count
        return b"".join(pieces)

    def provenance(self):
        return [{"offset": base, "length": (self.parts / f"{base:012d}.part").stat().st_size,
                 "sha256": sha256(self.parts / f"{base:012d}.part")} for base in sorted(self.used)]


def source_path(root: Path, format_id: str) -> Path:
    suffix = "" if format_id == "399" else f".f{format_id}"
    return _resolve_asset(root, root / "data/input" / f"{VIDEO_ID}{suffix}.mp4")


def local_media_facts(source: Path) -> dict:
    facts = {"size_bytes": source.stat().st_size, "sha256": sha256(source), "download_complete": True}
    with av.open(str(source)) as media:
        stream = media.streams.video[0]
        first = next(media.decode(stream))
        duration = float(stream.duration * stream.time_base)
        media.seek(int(max(0, duration - 1) / stream.time_base), stream=stream, backward=True)
        last = None
        for last in media.decode(stream):
            pass
        if last is None:
            raise RuntimeError("Could not decode final source span")
        facts["container_validation"] = {
            "time_base": str(stream.time_base), "duration_seconds": duration,
            "width": stream.width, "height": stream.height,
            "average_rate": str(stream.average_rate), "first_decoded_pts": first.pts,
            "last_decoded_pts": last.pts,
            "last_decoded_time_seconds": float(last.pts * last.time_base),
            "decode_scope": "First frame and the final keyframe-to-end span; not a complete frame index"}
    return facts


def ranged_download(root: Path, source: Path, fmt: dict, headers: dict, workers: int) -> None:
    """Bounded public-byte range downloader; temporary signed URL stays in RAM."""
    if source.exists():
        return
    total = int(fmt["filesize"])
    chunk_size = 4 * 1024 * 1024
    parts = _resolve_asset(root, root / "outputs/broadcast_scout" / f"download_f{fmt['format_id']}_parts")
    parts.mkdir(parents=True, exist_ok=True)

    def chunk(start: int) -> Path:
        end = min(start + chunk_size, total) - 1
        path = parts / f"{start:012d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(6):
            try:
                request = urllib.request.Request(fmt["url"], headers={**headers, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read(end - start + 2)
                    content_range = response.headers.get("Content-Range", "")
                if len(body) != end - start + 1 or content_range != f"bytes {start}-{end}/{total}":
                    raise RuntimeError(f"Unexpected byte range length={len(body)} range={content_range}")
                path.write_bytes(body)
                return path
            except Exception as exc:
                if attempt == 5:
                    # Never print the signed media URL contained in network errors.
                    raise RuntimeError(f"Byte range {start}-{end} failed after six attempts: {type(exc).__name__}") from None
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    starts = list(range(0, total, chunk_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(chunk, start): start for start in starts}
        for complete, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if complete % 5 == 0 or complete == len(starts):
                print(f"RANGES format={fmt['format_id']} {complete}/{len(starts)} ({100 * complete / len(starts):.1f}%)", flush=True)
    pending = source.with_suffix(source.suffix + ".assembled")
    with pending.open("wb") as writer:
        for start in starts:
            with (parts / f"{start:012d}.part").open("rb") as reader:
                shutil.copyfileobj(reader, writer)
    if pending.stat().st_size != total:
        raise RuntimeError("Assembled original byte count differs from official format size")
    pending.replace(source)


def inspect(root: Path, download: bool, format_id: str, workers: int) -> dict:
    import yt_dlp

    source = source_path(root, format_id)
    source.parent.mkdir(parents=True, exist_ok=True)
    opts = {"format": format_id, "noplaylist": True, "continuedl": True,
            "retries": 5, "fragment_retries": 5, "overwrites": False,
            "outtmpl": str(source)}
    with yt_dlp.YoutubeDL(opts) as downloader:
        info = downloader.extract_info(URL, download=download and not source.exists() and workers == 1)
    selected = next(fmt for fmt in info.get("formats", []) if fmt["format_id"] == format_id)
    if download and workers > 1:
        ranged_download(root, source, selected, info.get("http_headers", {}), workers)
    record = {"source_url": URL, "youtube_id": VIDEO_ID,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "authorization": "User explicitly authorized this official ISU video for local noncommercial research",
              "format_id": format_id, "representation": f"Official format {format_id} original video-only bytes; no audio acquired",
              "format": {key: selected.get(key) for key in ("width", "height", "fps", "vcodec", "filesize")},
              "local_path": str(source.relative_to(root)).replace("\\", "/"),
              "source_file_modified": False,
              **{key: info.get(key) for key in ("title", "channel", "channel_id", "upload_date", "duration", "live_status")}}
    record["available_1080p_formats"] = [{key: fmt.get(key) for key in (
        "format_id", "ext", "width", "height", "fps", "vcodec", "acodec", "filesize", "filesize_approx")}
        for fmt in info.get("formats", []) if fmt.get("height") == 1080]
    if source.exists():
        if selected.get("filesize") and source.stat().st_size != int(selected["filesize"]):
            raise RuntimeError("Existing source size differs from the official selected representation; original was not overwritten")
        record.update(local_media_facts(source))
    else:
        record["download_complete"] = False
    suffix = "" if format_id == "399" else f"_f{format_id}"
    dump_json(root / f"evidence/input/ankara2026_broadcast_acquisition{suffix}.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def hms(value: float) -> str:
    hour = int(value // 3600)
    minute = int(value % 3600 // 60)
    return f"{hour:02}:{minute:02}:{value % 60:06.3f}"


def reviewed_epoch(start: float, end: float) -> str:
    for number, (first, last) in enumerate([(3610, 3670), (8640, 8700), (14340, 14400)], 1):
        if first <= start and end <= last:
            return f"2026-09-17_SP_after_resurface_{number}"
    return "multi_epoch_broadcast_or_unreviewed_window"


def scout(root: Path, start: float, end: float | None, step: float, width: int, label: str, format_id: str, cached: bool = False) -> None:
    source = source_path(root, format_id)
    out = _resolve_asset(root, root / "outputs/broadcast_scout" / label)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    source_input = CachedSourceReader(root, format_id) if cached else str(source)
    with av.open(source_input, format="mp4") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
        stop = min(end or duration, duration)
        timebase = str(stream.time_base)
        for wanted in np.arange(start, stop, step):
            container.seek(int(wanted / stream.time_base), stream=stream, backward=True, any_frame=False)
            frame = next((item for item in container.decode(stream) if item.pts is not None and float(item.pts * item.time_base) >= wanted), None)
            if frame is None:
                continue
            actual = float(frame.pts * frame.time_base)
            image = frame.to_image()
            image.thumbnail((width, width))
            path = out / f"pts_{frame.pts:012d}.jpg"
            image.save(path, quality=90)
            rgb = np.asarray(image)
            # Ranking aid only: bright/low-chroma pixels are not a validated ice mask.
            brightness = rgb.mean(axis=2)
            chroma = rgb.max(axis=2).astype(float) - rgb.min(axis=2).astype(float)
            bright_fraction = float(((brightness > 170) & (chroma < 45)).mean())
            rows.append({"requested_seconds": float(wanted), "source_pts": frame.pts,
                         "time_base": str(frame.time_base), "source_time_seconds": actual,
                         "source_time_label": hms(actual), "source_decoded_frame_id": None,
                         "frame_id_note": "Complete source-frame index has not been generated; identity is source SHA256 + PTS + time_base",
                         "image": path.name, "bright_neutral_fraction_ranking_only": bright_fraction})
            print(f"SCOUT {len(rows):04d} {hms(actual)} {bright_fraction:.3f}", flush=True)
    for sheetno, offset in enumerate(range(0, len(rows), 48)):
        selected = rows[offset:offset + 48]
        tile_w, tile_h, columns = 240, 160, 6
        canvas = Image.new("RGB", (tile_w * columns, tile_h * ((len(selected) + columns - 1) // columns)), "#111923")
        draw = ImageDraw.Draw(canvas)
        for idx, row in enumerate(selected):
            im = Image.open(out / row["image"])
            im.thumbnail((tile_w, tile_h - 23))
            x, y = (idx % columns) * tile_w, (idx // columns) * tile_h
            canvas.paste(im, (x, y))
            draw.text((x + 5, y + tile_h - 20), f'{offset + idx:03d} | {row["source_time_label"]}', fill="#FFFFFF")
        canvas.save(out / f"contact_sheet_{sheetno:02d}.jpg", quality=93)
    dump_json(out / "scout.json", {"source_url": URL, "source": str(source.relative_to(root)).replace("\\", "/"),
              "format_id": format_id, "source_time_base": timebase, "duration_seconds": duration,
              "full_source_download_complete": source.exists(),
              "validated_cached_ranges": source_input.provenance() if cached else None,
              "background_session": "Ankara 2026 Junior Women Short Program, 2026-09-17; not the 2026-09-18 free-skating session",
              "background_epoch": reviewed_epoch(start, stop),
              "sampling": {"start_seconds": start, "end_seconds": stop, "step_seconds": step},
              "scope": "Actual source-decoded frames, sparse scout only; no full-frame index, geometric registration or calibration claimed",
              "rows": rows})


def export_clip(root: Path, start: float, end: float, width: int, label: str, format_id: str, cached: bool = False) -> None:
    source = source_path(root, format_id)
    out = _resolve_asset(root, root / "outputs/broadcast_scout" / label)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "clip.mp4"
    rows = []
    next_sample = start
    output_tb = Fraction(1, 1_000_000)
    source_input = CachedSourceReader(root, format_id) if cached else str(source)
    with av.open(source_input, format="mp4") as container, av.open(str(destination), "w", options={"movflags": "+faststart"}) as writer:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        encoded = writer.add_stream("libx264", rate=15)
        encoded.width = width
        encoded.height = round(stream.height * width / stream.width / 2) * 2
        encoded.pix_fmt = "yuv420p"
        encoded.time_base = output_tb
        encoded.codec_context.time_base = output_tb
        encoded.options = {"crf": "24", "preset": "fast", "bf": "0"}
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            source_time = float(frame.pts * frame.time_base)
            if source_time >= end:
                break
            if source_time + 1e-9 < next_sample:
                continue
            next_sample += 1 / 15
            if not rows:
                frame.to_image().resize((width, encoded.height)).save(out / "poster.jpg", quality=93)
                anchor_time = source_time
            source_pts, source_time_base = frame.pts, str(frame.time_base)
            preview = frame.reformat(width=width, height=encoded.height, format="yuv420p")
            preview.pts = round((source_time - anchor_time) / output_tb)
            preview.time_base = output_tb
            rows.append({"preview_frame_index": len(rows), "source_pts": source_pts,
                         "source_time_base": source_time_base, "source_time_seconds": source_time,
                         "source_decoded_frame_id": None, "expected_preview_pts": preview.pts,
                         "expected_preview_time_base": str(output_tb)})
            for packet in encoded.encode(preview):
                writer.mux(packet)
        for packet in encoded.encode():
            writer.mux(packet)
    with av.open(str(destination)) as validation:
        actual_count = 0
        for actual_count, frame in enumerate(validation.decode(video=0), 1):
            if actual_count > len(rows):
                raise RuntimeError("Preview has more decoded frames than the source mapping")
            row = rows[actual_count - 1]
            actual_time = float(frame.pts * frame.time_base)
            expected_time = row["expected_preview_pts"] * float(output_tb)
            if abs(actual_time - expected_time) > 1e-6:
                raise RuntimeError("Preview encoded timestamp does not match source mapping")
            row["preview_pts"] = frame.pts
            row["preview_time_base"] = str(frame.time_base)
        if actual_count != len(rows):
            raise RuntimeError(f"Preview decode count mismatch: {actual_count} versus {len(rows)}")
    dump_json(out / "pts_map.json", {"source_url": URL, "source_file": source.name,
              "source_sha256": sha256(source) if source.exists() else None, "format_id": format_id,
              "full_source_download_complete": source.exists(),
              "validated_cached_ranges": source_input.provenance() if cached else None,
              "background_session": "Ankara 2026 Junior Women Short Program 2026-09-17",
              "background_epoch": reviewed_epoch(start, end),
              "requested_window_seconds": [start, end], "preview": destination.name,
              "preview_sha256": sha256(destination), "preview_sampling": "15 source-PTS samples per second, H264 derived preview",
              "preview_dimensions": [encoded.width, encoded.height],
              "preview_encoding": {"codec": "libx264", "crf": 24, "pixel_format": "yuv420p"},
              "preview_role": "Playback illustration only; pixel measurements must identify the original source representation",
              "complete_source_frame_index_available": False,
              "decoded_preview_validation": "All decoded output PTS verified against source mapping within 1 microsecond",
              "rows": rows})
    print(f"CLIP {destination} rows={len(rows)} size={destination.stat().st_size}", flush=True)


def build_review(root: Path, format_id: str) -> None:
    base = _resolve_asset(root, root / "outputs/broadcast_scout")
    out = base / "review"
    out.mkdir(exist_ok=True)
    files: list[str] = []

    def copy(source: Path, relative: str):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append(relative)

    suffix = "" if format_id == "399" else f"_f{format_id}"
    acquisition_path = root / f"evidence/input/ankara2026_broadcast_acquisition{suffix}.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if not acquisition.get("download_complete"):
        raise RuntimeError("Review manifest requires a completed source representation")
    copy(acquisition_path, "acquisition.json")
    copy(_resolve_asset(root, root / "evidence/input/ankara2026_broadcast_acquisition_f134.json"), "scout_source_f134.json")
    copy(_resolve_asset(root, root / "evidence/input/ankara2026_broadcast_visual_review.json"), "visual_review.json")
    if format_id == "399":
        copy(_resolve_asset(root, root / "evidence/input/ankara2026_broadcast_hd_validation.json"), "hd_source_validation.json")
    copy(root / "scripts/acquire_broadcast.py", "reproduce/acquire_broadcast.py.txt")
    for filename in ["wide_geometry_inventory_360p.jpg", "wide_geometry_inventory_360p.json"]:
        if not (base / filename).exists():
            raise RuntimeError("Geometry source inventory must exist before preparing the broadcast review")
        copy(base / filename, filename)
    copy(root / "scripts/inventory_broadcast_wide.py", "reproduce/inventory_broadcast_wide.py.txt")
    copy(base / "overview/resurfacing_to_empty.jpg", "resurfacing_to_empty.jpg")
    copy(base / "overview/overview_pts.json", "overview_pts.json")
    copy(base / "coarse_60s_360p/scout.json", "full_broadcast_scout_pts.json")
    for page in sorted((base / "coarse_60s_360p").glob("contact_sheet_*.jpg")):
        relative = f"full_scout/{page.name}"
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(page) as picture:
            picture.save(target, quality=65)
        files.append(relative)
    clips = []
    for number, (start, end) in enumerate([(3610, 3670), (8640, 8700), (14340, 14400)], 1):
        source_folder = base / f"clip_clean_{number}_{'1080_package' if format_id == '399' else '360p'}"
        mapping = json.loads((source_folder / "pts_map.json").read_text(encoding="utf-8"))
        if mapping["source_sha256"] != acquisition["sha256"]:
            raise RuntimeError("Clip's source hash must match the fully downloaded acquisition")
        for filename in ["clip.mp4", "pts_map.json", "poster.jpg"]:
            copy(source_folder / filename, f"clean_{number}/{filename}")
        clips.append({"title": f"清冰后连续空场广角 {number} · {hms(start)}–{hms(end)}",
                      "start_s": start, "end_s": end, "video": f"clean_{number}/clip.mp4",
                      "pts_map": f"clean_{number}/pts_map.json", "poster": f"clean_{number}/poster.jpg",
                      "background_epoch": reviewed_epoch(start, end),
                      "purpose": "原片空场广角取材；可观察冰面固定标记、边界和反光，作为后续几何建模输入",
                      "geometry_status": "原始透视视图，未进行跨机位或米制标定；不能直接当两个自由滑节目的同日冰纹"})
    manifest = {"schema_version": "rinkspace.broadcast-review.v1", "source_media_included": False,
                "title": "ISU Ankara 全场转播与清冰后广角取材", "video_id": VIDEO_ID,
                "source_url": URL, "source_sha256": acquisition["sha256"], "source_duration_s": 19560,
                "source_scope": f"完整官方 format {format_id} 视频轨；三段各60秒连续实片。全场每60秒粗筛联系表及清冰前后概览来自独立的官方360p format134取材轨，绝不作为1080p像素测量。",
                "source_format_id": format_id, "contact_sheet": "resurfacing_to_empty.jpg",
                "packaging_policy": "保留完整326张粗筛的7页联系表及3×60秒/2700帧连续预览和PTS映射。小包播放片为640×360、15fps、H264 CRF24；联系表JPEG quality65用于浏览。" + ("完整1080p源轨和960宽预览留在本地，原片未改写。" if format_id == "399" else "本清单只声明360p完整取材轨；1080p是否下载完成，以单独的来源记录为准。") + "缩小和有损编码均不得当作原像素测量输入。",
                "geometry_inventory_image": "wide_geometry_inventory_360p.jpg",
                "geometry_inventory_json": "wide_geometry_inventory_360p.json",
                "contact_sheets": [{"image": f"full_scout/contact_sheet_{page:02d}.jpg",
                                    "title": f"全场第 {page + 1}/7 页 · {hms(page * 48 * 60)} 起（每60秒实际解码一帧，360p粗筛）"}
                                   for page in range(7)],
                "full_source_player_url": "http://127.0.0.1:8765/api/media?video_id=" + ("broadcast" if format_id == "399" else "broadcast_scout"),
                "files": files, "clips": clips,
                "observations": ["完整5小时26分转播已按实际解码PTS每60秒抽样，共326张；7页联系表已逐页目视检查。",
                                 "三段60秒窗口先由360p逐秒取样目视检查，冰面上未见运动员或清冰车；栏外观众仍有活动。",
                                 "前置上下文实际出现清冰车，所列三个空场窗口分别位于三次清冰之后。",
                                 "派生播放片为15fps，每一帧都有源PTS、时基及编码后实际PTS映射，回读误差不超过1微秒。",
                                 "小包为保持全部时段覆盖，采用640×360播放预览和JPEG quality65联系表；已完成的本地原轨保持不变，压缩预览不能充当几何测量源。"],
                "limitations": ["该全场是2026-09-17女单短节目；重点节目为2026-09-18自由滑，背景冰纹属于不同时间段，不能直接跨日拟合。",
                                "广角取材完整度不等于全场几何配准、欧氏标定或米制标定；相机畸变和跨机位连接仍需独立验证。",
                                "全场视频无音频轨；未建立全场全部解码帧索引，取材帧以原视频SHA256、源PTS和时基定位。",
                                "离线小包只含有证据出处的联系表和连续派生片，完整多GB原片保留本地，未装入小包。"]}
    dump_json(out / "manifest.json", manifest)
    print(f"REVIEW_MANIFEST {out / 'manifest.json'} assets={len(files)} bytes={sum((out / file).stat().st_size for file in files)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--verify-local", action="store_true", help="Offline SHA256 and first/tail decode verification")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--format", dest="format_id", default="399", choices=["399", "134"])
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 17))
    parser.add_argument("--scout", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--cached", action="store_true", help="Read only existing verified download ranges; fail on missing bytes")
    parser.add_argument("--review-manifest", action="store_true", help="Package existing inspected scouts and all three mapped clips")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--step", type=float, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--label", default="every_60s")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.step <= 0 or args.start < 0:
        parser.error("Require step > 0 and start >= 0")
    if args.inspect or args.download:
        inspect(root, args.download, args.format_id, args.workers)
    if args.verify_local:
        suffix = "" if args.format_id == "399" else f"_f{args.format_id}"
        provenance = root / f"evidence/input/ankara2026_broadcast_acquisition{suffix}.json"
        record = json.loads(provenance.read_text(encoding="utf-8"))
        facts = local_media_facts(source_path(root, args.format_id))
        if record.get("sha256") and record["sha256"] != facts["sha256"]:
            raise RuntimeError("Original source SHA256 changed; existing provenance was not overwritten")
        record.update(facts)
        dump_json(provenance, record)
        print(json.dumps(record["container_validation"]), flush=True)
    if args.scout:
        scout(root, args.start, args.end, args.step, args.width, args.label, args.format_id, args.cached)
    if args.clip:
        if args.end is None or args.end <= args.start:
            parser.error("--clip requires --end greater than --start")
        export_clip(root, args.start, args.end, args.width, args.label, args.format_id, args.cached)
    if args.review_manifest:
        build_review(root, args.format_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
