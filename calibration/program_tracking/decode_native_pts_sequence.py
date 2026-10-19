"""Decode an authenticated consecutive original PTS sequence without resampling.

Native PNGs and an explicit source clock manifest belong in the Data root.
No model, camera, tracked point or exposure alias is loaded by this decoder.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.programme_source_stream import file_sha

DEFAULT_SELECTION = ROOT / "events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json"
DEFAULT_OUTPUT = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/"
    "frame_by_frame_review_20261013/opening_100_50fps")


def decode(selection_path, output, start_pts, step_pts, count):
    selection_path, output = Path(selection_path), Path(output).resolve()
    if not output.is_relative_to(Path("D:/CodexData/RinkSpace_Data").resolve()):
        raise ValueError("Decoded original PNGs belong in the existing Data root")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    source = selection["primary_source"]
    source_path = Path(source["path"]).resolve()
    source_stat = source_path.stat()
    source_sha = file_sha(source_path)
    if source_sha != source["sha256"]:
        raise ValueError("Primary source bytes changed")
    if count <= 0 or step_pts <= 0:
        raise ValueError("Positive native PTS step/count required")
    clock = Fraction(source["time_base"])
    if Fraction(source["native_fps"]) * clock * step_pts != 1:
        raise ValueError("Requested PTS step does not cover every native exposure")
    targets = [int(start_pts)+i*int(step_pts) for i in range(int(count))]
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = set(map(int, np.linspace(1, count, 20).round().astype(int)))
    samples, rows = {}, []
    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if (Fraction(stream.time_base) != clock or stream.average_rate != Fraction(source["native_fps"])
                or (stream.width, stream.height) != (source["width"], source["height"])):
            raise ValueError("Primary native stream clock/rate/pixel domain changed")
        stream_description = dict(codec=stream.codec_context.name, video_stream_index=stream.index,
            pixel_format=stream.codec_context.format.name,
            native_fps=str(stream.average_rate), time_base=str(stream.time_base),
            native_width_height=[stream.width, stream.height])
        container.seek(targets[0], stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < targets[0]:
                continue
            if len(rows) == count:
                break
            target = targets[len(rows)]
            if int(frame.pts) != target or Fraction(frame.time_base) != clock:
                raise ValueError(f"Original consecutive PTS missing or repeated: wanted={target}, decoded={frame.pts}")
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.dtype != np.uint8 or bgr.shape != (source["height"], source["width"], 3):
                raise ValueError("Original native pixels changed shape/type")
            bgr_sha = hashlib.sha256(bgr.tobytes()).hexdigest()
            rgb_sha = hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            index = len(rows)+1
            path = frames_dir / f"{index:06d}.png"
            if path.exists():
                decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if decoded is None or not np.array_equal(decoded, bgr):
                    raise ValueError("Existing frame file has different original pixels: "+str(path))
                png_sha = file_sha(path)
            else:
                ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), bgr):
                    raise ValueError("Original PNG lossless roundtrip failed")
                data = encoded.tobytes()
                path.write_bytes(data)
                png_sha = hashlib.sha256(data).hexdigest()
            row = dict(index=index, native_pts=target, source_pts=target,
                source_time_base=str(clock), source_time_seconds=float(target*clock),
                source_time=f"00:00:{float(target*clock):06.3f}",
                native_frame_index_0based=int(target/step_pts),
                path=str(path.resolve()), png_sha256=png_sha,
                decoded_native_BGR_sha256=bgr_sha, decoded_native_RGB_sha256=rgb_sha,
                native_hw=[source["height"], source["width"]],
                exact_native_PTS_and_lossless_PNG_verified=True)
            rows.append(row)
            if index in sample_indices:
                samples[index] = cv2.resize(bgr, (384, 216), interpolation=cv2.INTER_AREA)
            if index % 20 == 0:
                print(json.dumps(dict(stage="decoded_original_frames", completed=index, total=count)), flush=True)
    if [r["native_pts"] for r in rows] != targets:
        raise ValueError(f"Original consecutive sequence incomplete: {len(rows)}/{count}")
    final_stat = source_path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Primary source changed while decoding")
    known = selection.get("representation_verification", {})
    matching_first = known.get("checked_official_pts") == rows[0]["native_pts"]
    if matching_first and rows[0]["decoded_native_BGR_sha256"] != known["checked_frame_decoded_BGR_sha256"]:
        raise ValueError("First official exposure differs from the previously verified original")
    tile_w, tile_h, label_h = 384, 216, 26
    sheet = np.full((4*(tile_h+label_h), 5*tile_w, 3), 18, np.uint8)
    for ordinal, index in enumerate(sorted(samples)):
        x, y = (ordinal % 5)*tile_w, (ordinal // 5)*(tile_h+label_h)
        sheet[y:y+tile_h, x:x+tile_w] = samples[index]
        row = rows[index-1]
        cv2.putText(sheet, f"{index:03d} | {row['source_time']} | PTS {row['native_pts']}",
            (x+5, y+tile_h+18), cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240), 1, cv2.LINE_AA)
    sheet_path = output / "contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(sheet_path)
    manifest = dict(schema="rinkspace.exact-native-consecutive-source-frames.v1",
        event_id=selection["event_id"], programme_id=selection["programme_id"],
        source=dict(source, path=str(source_path), file_sha256_authenticated=source_sha),
        source_selection_reference=dict(path=str(selection_path.resolve()), sha256=file_sha(selection_path)),
        decoder=dict(library="PyAV", version=av.__version__, av_libraries={k:list(v) for k,v in av.library_versions.items()},
            **stream_description, cpu_threads=1, seek_once_to_prior_keyframe=True,
            consecutive_decode=True, temporal_resampling=False, spatial_resizing_of_source_PNGs=False,
            frame_pts_selected_as_integer=True, PNG_compression=1),
        start_pts=targets[0], last_pts=targets[-1], stop_pts_exclusive=targets[-1]+step_pts,
        native_pts_step=step_pts, frame_count=count, frames=rows,
        native_exposures_all_decoded=True, original_PTS_order_verified=True,
        PNG_roundtrip_lossless_verified=True,
        previously_verified_first_official_exposure_hash_checked=matching_first,
        first_frame_accepted_points_modified=False, points_extracted=False, model_or_solver_used=False,
        contact_sheet=dict(path=str(sheet_path.resolve()), sha256=file_sha(sheet_path),
            layout_columns_rows=[5,4], sample_indices_1based=sorted(samples),
            thumbnail_only_not_measurement_pixels=True))
    manifest_path = output / "frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(stage="complete", frame_count=count, output=str(output),
        first_pts=targets[0], last_pts=targets[-1], manifest=str(manifest_path)), ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-pts", type=int, default=256256)
    parser.add_argument("--step-pts", type=int, default=256)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    decode(args.selection, args.output, args.start_pts, args.step_pts, args.count)
