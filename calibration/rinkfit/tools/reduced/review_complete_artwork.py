"""Continuous S01 working review using the original exact d78 RS pipeline.

Only current complete model/source data are consumed. The original review's
RS certificates, bracket, fixed-point acceleration, bisection fallback, drawing,
native frame iterator, encoding and clock verification are reused directly.
No solver or historical qualification replay is started.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import av
import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "calibration" / "rinkfit" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rinkfit_start.native_scene import NativeScene, camera_from_event
from rinkfit_start.gpu_camera import validate_then_install
from review_original_ad import paths_from_model

FIRST, LAST, STEP, FRAMES = 256256, 575488, 256, 1248


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4194304), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(model):
    """Retain every actual source vertex through the current native review API."""
    scene = NativeScene(model)
    paths = paths_from_model(model)
    lookup = {p.get("id", f"{mi}/{pi}"): (mi, pi, p) for mi, material in enumerate(model["materials"])
              for pi, p in enumerate(material["paths"]) if not material.get("unused_legacy_witness", False)}
    records, arrays = [], []
    offset = 0

    def add(xyz, mi, pi, path_id, variant_id, label, color):
        nonlocal offset
        xyz = np.asarray(xyz, float)
        if not np.isfinite(xyz).all():
            raise ValueError("Nonfinite current native scene path")
        records.append(dict(material_id=mi, path_index=pi, path_id=path_id, variant_id=variant_id,
                            label=label, begin=offset, end=offset + len(xyz), color=color))
        arrays.append(xyz)
        offset += len(xyz)

    for path in paths:
        mi, pi, definition = lookup[path["path_id"]]
        ad = re.search(r"Ad\d+", path["label"])
        color = (40, 245, 70) if ad else (30, 235, 245) if path["kind"] in (0, 1) else (255, 190, 30)
        add(path["xyz"], mi, pi, path["path_id"], definition.get("variant_id", path["path_id"]), path["label"], color)
    physical_path_count = len(records)
    for height, name in ((0., "continuous_board_bottom"), (scene.height, "continuous_board_top")):
        add(scene.board(np.linspace(0., scene.perimeter, 4097), height), -1, -1, name, name, name, (255, 160, 20))
    for ad in range(2, 13):
        candidates = [p for p in paths if p["label"].startswith(f"Ad{ad}.")]
        if not candidates:
            raise ValueError(f"Current complete scene is missing Ad{ad}")
        candidates.sort(key=lambda p: len(p["xyz"]), reverse=True)
        sample = candidates[0]["xyz"]
        xyz = sample[[len(sample) // 2]]
        add(xyz, -2-ad, -1, f"number_Ad{ad}", f"Ad{ad}", f"Ad{ad}", (255, 255, 255))
    return np.concatenate(arrays), records, physical_path_count



PANEL_WIDTH = 640
OUTPUT_WIDTH = 1920 + PANEL_WIDTH


def current_observations(args, incoming, model=None):
    packet_path = getattr(args, 'packet', None)
    state_report = incoming.parent / 'report.json'
    if packet_path is None and state_report.is_file():
        report = json.loads(state_report.read_text(encoding='utf8'))
        packet_path = report.get('packet', {}).get('path')
    if packet_path is None:
        return {}, None
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text(encoding='utf8'))
    semantic_path = Path(packet['semantic_packet']['manifest'])
    if sha(semantic_path) != packet['semantic_packet']['sha256']:
        raise ValueError('Current observation packet/semantic binding differs')
    semantic = json.loads(semantic_path.read_text(encoding='utf8'))
    if state_report.is_file():
        expected = json.loads(state_report.read_text(encoding='utf8')).get('packet')
        if expected is not None and expected != ref(packet_path):
            raise ValueError('Review observations differ from this complete solver state')
    if model is not None:
        declared = model.get('current_input_scope', {}).get('statistical_active_mask', {})
        if declared.get('sha256') != semantic['statistical_active_mask']['sha256']:
            raise ValueError('Review model/source selection differs')
    count = int(packet['observations'])
    if semantic['observations'] != count or packet['native_time_base'] != [1, 12800] or packet['native_image_wh'] != [1920, 1080]:
        raise ValueError('Current observation native source clock/layout differs')
    def array(desc):
        path = Path(desc['file'])
        if not path.is_absolute():
            path = packet_path.parent / path
        if sha(path) != desc['sha256']:
            raise ValueError('Current observation array differs: ' + str(path))
        return np.fromfile(path, desc['dtype']).reshape(desc['shape'])
    uv = array(packet['arrays']['native_uv'])
    pts = array(packet['arrays']['native_pts'])
    active = array(semantic['active_source_mask']).astype(bool)
    if uv.shape != (count, 2) or pts.shape != (count,) or active.shape != (count,):
        raise ValueError('Current observation row layout differs')
    if not np.isfinite(uv[active]).all() or np.any((pts[active] < FIRST) | (pts[active] > LAST) | ((pts[active]-FIRST) % STEP != 0)):
        raise ValueError('Current observations are outside the exact native exposure clock')
    rows = {int(pt): np.flatnonzero(active & (pts == pt)) for pt in np.unique(pts[active])}
    values = {pt: (rr, uv[rr]) for pt, rr in rows.items()}
    identity = dict(packet=ref(packet_path), semantic=ref(semantic_path),
                    active_source_mask=semantic['active_source_mask'], native_uv=packet['arrays']['native_uv'],
                    native_pts=packet['arrays']['native_pts'], raw_rows=count, active_rows=int(active.sum()),
                    only_current_exposure_active_sources=True, inactive_or_historical_points_drawn=False,
                    source_point_numbers_drawn=False)
    return values, identity


def top_down(world, records):
    panel = np.full((1080, PANEL_WIDTH, 3), 24, np.uint8)
    xy = np.asarray(world)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi-lo, 1.e-6)
    scale = min((PANEL_WIDTH-72)/span[0], 850/span[1])
    center = (hi+lo)/2
    def pixel(points):
        result = (points-center)*scale
        return np.rint(np.column_stack([PANEL_WIDTH/2+result[:, 0], 535-result[:, 1]])).astype(np.int32)
    for item in records:
        points = pixel(xy[item['begin']:item['end']])
        if item['material_id'] <= -2:
            if len(points):
                cv2.putText(panel, item['label'], tuple(points[0]+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (255,255,255), 1, cv2.LINE_AA)
            continue
        if len(points)>1:
            cv2.polylines(panel, [points], False, item['color'], 1, cv2.LINE_AA)
        elif len(points):
            cv2.circle(panel, tuple(points[0]), 2, item['color'], 1, cv2.LINE_AA)
    cv2.putText(panel, 'Same complete model - top view', (20, 32), cv2.FONT_HERSHEY_SIMPLEX, .57, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(panel, 'All ice / board / advertisement paths', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .47, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Red dots at left: current active source UV', (20, 1020), cv2.FONT_HERSHEY_SIMPLEX, .44, (100,140,255), 1, cv2.LINE_AA)
    return panel

def encoder(ffmpeg, path):
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", str(OUTPUT_WIDTH)+"x1080", "-framerate", "50", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-video_track_timescale", "12800",
            "-movflags", "+faststart", str(path)]


def setup(args):
    out = Path(args.output).resolve()
    if "RinkSpace_Data" not in out.parts:
        raise ValueError("Video and diagnostic products must be stored in Data")
    out.mkdir(parents=True, exist_ok=True)
    incoming = Path(args.model).resolve()
    frozen = out / "model.json"
    if frozen.exists():
        if sha(incoming) != sha(frozen):
            raise ValueError("The review output already belongs to a different exact model")
    else:
        shutil.copyfile(incoming, frozen)
    model = json.loads(frozen.read_text(encoding="utf8"))
    native = imported("original_d78_continuous_review", args.native_review)
    sys.path.insert(0, str(Path(args.compiled_dir).resolve()))
    from bound_camera_sensor import bind_camera_from_model
    camera = bind_camera_from_model(ref(frozen), model=model)
    camera = validate_then_install(camera, args.gpu_dll, args.gpu_validation, model_ref=ref(frozen),
                                   output_dir=out / "gpu_camera_shadow")
    if not camera.gpu_acceleration.get("enabled"):
        raise RuntimeError("Existing GPU camera binding failed: " + json.dumps(camera.gpu_acceleration))
    world, records, count = inventory(model)
    # Reuse the original drawing routine while retaining Ad numbers. Per-source
    # material IDs remain in the inventory rather than covering large letters.
    original_text = native.text

    def review_text(canvas, label, xy, scale=.38, color=(245, 245, 245)):
        if re.fullmatch(r"m\d+", label):
            return
        if re.fullmatch(r"Ad\d+", label):
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            xy = np.clip(xy, [4., height+4.], [1916.-width, 1076.-baseline])
        original_text(canvas, label, xy, scale, color)

    native.text = review_text
    native.REVIEW_LABEL = "current source artwork"
    native.MODEL_MATERIALS = len(model["materials"])
    native.MODEL_PATHS = count

    # Cache the static banner once. Camera/model hashes stay in the manifest;
    # source time, frame count and advertisement numbers remain in native.draw.
    banner = np.full((67, 1920, 3), 15, dtype=np.uint8)
    chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese_font.exists():
        from PIL import Image, ImageDraw, ImageFont
        heading = Image.fromarray(banner[..., ::-1])
        pen = ImageDraw.Draw(heading)
        pen.text((16, 3), "第一镜头 · 当前完整拟合检查", font=ImageFont.truetype(str(chinese_font), 24),
                 fill=(255, 210, 70))
        pen.text((16, 37), "冰场线、广告边界与编号全部显示 · 待整段核对", font=ImageFont.truetype(str(chinese_font), 18),
                 fill=(245, 245, 245))
        banner = np.asarray(heading)[..., ::-1].copy()
    else:
        original_text(banner, "Current complete scene/camera - working review", (16, 25), .59, (50, 210, 255))
        original_text(banner, "All ice lines, artwork boundaries and advertisement numbers; not accepted", (16, 51), .43)

    source_points, source_identity = current_observations(args, incoming, model)
    overview = top_down(world, records)

    def draw(raw, prediction, pts):
        canvas, record = native.draw(raw, prediction, records, pts, dict(maximum_error_px=0., quantiles_px=[0.]))
        rr, measured = source_points.get(int(pts), (np.empty(0, np.int64), np.empty((0,2), float)))
        for point in measured:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (0,0,255), -1, cv2.LINE_AA)
        canvas[:67] = banner
        right = overview.copy()
        cv2.putText(right, 'PTS '+str(pts)+' | source points '+str(len(rr)), (20,1054), cv2.FONT_HERSHEY_SIMPLEX, .48, (230,230,230), 1, cv2.LINE_AA)
        record.update(current_active_source_points=len(rr),
                      current_source_rows_sha256=hashlib.sha256(rr.astype('<u8').tobytes()).hexdigest(),
                      current_source_UV_sha256=hashlib.sha256(measured.astype('<f8').tobytes()).hexdigest())
        return np.concatenate([canvas, right], axis=1), record

    context = dict(schema="rinkfit.current_complete_artwork_review_context.v1", model=ref(frozen),
                   source=str(Path(args.source).resolve()), original_native_review=ref(args.native_review),
                   current_model_path_inventory=ref(Path(__file__).with_name("review_original_ad.py")),
                   current_continuous_caller=ref(__file__), compiled_binding=ref(Path(args.compiled_dir) / "bound_camera_sensor.py"),
                   GPU_camera=camera.gpu_acceleration, GPU_dll=ref(args.gpu_dll),
                   all_actual_material_paths=count, continuous_board_added_paths=2, advertisement_numbers=list(range(2, 13)),
                   total_world_samples=len(world), source_vertices_preserved=True, projection_approximation=False,
                   current_observations=source_identity, source_native_image_wh=[1920,1080], review_output_wh=[OUTPUT_WIDTH,1080],
                   same_complete_model_top_down=True,
                   strict_original_RS_certificate_bracket_and_1e8_root_equation=True,
                   source_PTS=dict(first=FIRST, last=LAST, step=STEP, frames=FRAMES, time_base="1/12800"),
                   model_time="(source_PTS-256256)/12800", no_Solve=True, whole_shot_acceptance=False)
    write(out / "context.json", context)
    write(out / "all_paths.json", records)
    return out, frozen, model, native, camera, world, records, draw, context


def preflight(args, runtime):
    out, frozen, model, native, camera, world, records, draw, context = runtime
    if context["current_observations"] is None:
        raise ValueError("Complete review needs the current packet (--packet); source points must not be omitted")
    report_path = out / "preflight.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf8"))
        implementation = {key: context[key] for key in ("original_native_review", "current_model_path_inventory", "current_continuous_caller", "compiled_binding", "GPU_dll")}
        if report["model"] != ref(frozen) or not report["passed"] or report.get("implementation_binding") != implementation:
            raise ValueError("Saved preflight does not bind the exact current review model")
        return report
    started = time.perf_counter()
    frames = native.source_frames(args.source)
    measurements = []
    qas = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clip = out / "preflight.mp4"
    if clip.exists():
        raise FileExistsError("Refuse to replace an earlier preflight clip")
    with (out / "preflight_encoding.log").open("wb") as log:
        proc = subprocess.Popen(encoder(ffmpeg, clip), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for i in range(5):
                pts, raw = next(frames)
                t = (pts-FIRST) / 12800.
                begin = time.perf_counter()
                prediction = native.project_RS(camera, world, t)
                projection_time = time.perf_counter()-begin
                begin = time.perf_counter()
                canvas, record = draw(raw, prediction, pts)
                draw_time = time.perf_counter()-begin
                if i == 0:
                    check = native.QA(camera_from_event(model), world, t, prediction)
                    check.update(native_pts=pts, model=ref(frozen))
                    qas.append(check)
                    if not cv2.imwrite(str(out / "preflight_first.png"), canvas):
                        raise OSError("Could not save preflight frame")
                proc.stdin.write(memoryview(canvas).cast("B"))
                measurements.append(dict(index=i+1, native_pts=pts, projection_seconds=projection_time,
                                         draw_seconds=draw_time, **record))
                print(json.dumps(dict(stage="preflight", frame=i+1, native_pts=pts,
                                      projection_seconds=projection_time, draw_seconds=draw_time)), flush=True)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("Preflight encoder failed")
        except BaseException:
            if proc.poll() is None:
                proc.kill(); proc.wait()
            raise
        finally:
            frames.close()
    encoded = av.open(str(clip))
    video = encoded.streams.video[0]
    ticks = [Fraction(frame.pts) * frame.time_base for frame in encoded.decode(video)]
    encoded.close()
    if ticks != [Fraction(i, 50) for i in range(5)]:
        raise ValueError("Preflight encoded 50 fps clock mismatch")
    projection_mean = float(np.mean([item["projection_seconds"] for item in measurements]))
    report = dict(schema="rinkfit.current_artwork_actual_native_preflight.v1", passed=True,
                  model=ref(frozen), clip=ref(clip), source_PTS=[item["native_pts"] for item in measurements],
                  implementation_binding={key: context[key] for key in ("original_native_review", "current_model_path_inventory", "current_continuous_caller", "compiled_binding", "GPU_dll")},
                  native_fps="50/1", exact_native_consecutive_exposures=True,
                  every_actual_path_enabled=True, all_source_vertices_preserved=True,
                  independent_original_NumPy_same_model_RS=qas, GPU_stats=dict(camera.at.stats),
                  projection_seconds_mean=projection_mean,
                  estimated_full_projection_seconds=projection_mean*FRAMES,
                  measurements=measurements, elapsed_seconds=time.perf_counter()-started,
                  whole_shot_acceptance=False)
    write(report_path, report)
    print(json.dumps(dict(stage="preflight_passed", report=str(report_path),
                          projection_seconds_mean=projection_mean, GPU_stats=report["GPU_stats"])), flush=True)
    return report


def complete(args, runtime, initial):
    out, frozen, model, native, camera, world, records, draw, context = runtime
    final = out / "s01_all_artwork_working_review.mp4"
    silent = out / "s01_all_artwork.silent.partial.mp4"
    if final.exists() or silent.exists():
        raise FileExistsError("Refuse to replace an earlier continuous video")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("Original encoded-clock verifier requires ffprobe")
    started = time.perf_counter()
    qas, keyframes, visible_counts = [], [], []
    projection_total = draw_total = 0.
    command = encoder(ffmpeg, silent)
    ledger = out / "source_PTS_coverage.jsonl"
    count = 0
    with (out / "encoding.log").open("wb") as log, ledger.open("x", encoding="utf8") as coverage:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for pts, raw in native.source_frames(args.source):
                i = (pts-FIRST)//STEP+1
                t = (pts-FIRST)/12800.
                before = time.perf_counter()
                prediction = native.project_RS(camera, world, t)
                projection_total += time.perf_counter()-before
                before = time.perf_counter()
                canvas, record = draw(raw, prediction, pts)
                draw_total += time.perf_counter()-before
                if i in (1, 625, 1248):
                    check = native.QA(camera_from_event(model), world, t, prediction)
                    check.update(frame_index=i, native_pts=pts)
                    qas.append(check)
                    image = out / f"frame_{i:06d}.png"
                    if not cv2.imwrite(str(image), canvas):
                        raise OSError("Could not save continuous review keyframe")
                    keyframes.append(ref(image))
                proc.stdin.write(memoryview(canvas).cast("B"))
                coverage.write(json.dumps(dict(index=i, native_pts=pts, source_time_base="1/12800",
                                               source_seconds=pts/12800., model_sha256=context["model"]["sha256"],
                                               source_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), **record))+"\n")
                count += 1
                visible_counts.append(record["visible_model_paths"])
                if i == 1 or i % 25 == 0:
                    camera.at.assert_bound_model(ref(frozen))
                    print(json.dumps(dict(stage="continuous", rendered_frames=count, total=FRAMES,
                                          native_pts=pts, elapsed_seconds=time.perf_counter()-started,
                                          projection_seconds=projection_total, draw_seconds=draw_total,
                                          GPU_stats=dict(camera.at.stats))), flush=True)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("Original encoder pipeline failed")
        except BaseException:
            if proc.poll() is None:
                proc.kill(); proc.wait()
            raise
    if count != FRAMES:
        raise ValueError("Full native source frame count changed")
    source = av.open(args.source)
    has_audio = bool(source.streams.audio)
    source.close()
    mux = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(silent)]
    if has_audio:
        mux += ["-ss", str(FIRST/12800.), "-t", "24.96", "-i", args.source, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", "24.96"]
    else:
        mux += ["-map", "0:v:0", "-c:v", "copy"]
    mux += ["-video_track_timescale", "12800", "-movflags", "+faststart", str(final)]
    result = subprocess.run(mux, capture_output=True, text=True, encoding="utf8")
    if result.returncode:
        raise RuntimeError(result.stderr)
    source_wh = native.WH
    try:
        native.WH = (OUTPUT_WIDTH,1080)
        verification = native.verify_video(final, ffprobe)
    finally:
        native.WH = source_wh
    actual_pts = [json.loads(line)["native_pts"] for line in ledger.read_text(encoding="utf8").splitlines()]
    if actual_pts != list(range(FIRST, LAST+STEP, STEP)):
        raise ValueError("Full source native PTS ledger changed")
    camera.at.assert_bound_model(ref(frozen))
    report = dict(schema="rinkfit.current_complete_artwork_working_review.v1", status="complete_NOT_ACCEPTED",
                  model=ref(frozen), context=ref(out / "context.json"), video=ref(final), frames=count,
                  source=args.source, source_PTS_first=FIRST, source_PTS_last=LAST, source_time_base="1/12800",
                  source_coverage=ref(ledger), source_PTS_coverage_exactly_all1248_once=True,
                  source_PTS_array_sha256=hashlib.sha256(np.asarray(actual_pts, dtype="<i8").tobytes()).hexdigest(),
                  all_actual_material_paths=context["all_actual_material_paths"], continuous_board_paths=2,
                  advertisement_numbers=list(range(2, 13)), all_actual_source_vertices_preserved=True,
                  same_model_original_exact_RS=True, independent_original_NumPy_keyframe_checks=qas,
                  keyframes=keyframes, visible_path_count_range=[min(visible_counts), max(visible_counts)],
                  original_encoded_clock_verification=verification, original_audio_included=has_audio,
                  GPU_stats=dict(camera.at.stats), preflight=ref(out / "preflight.json"),
                  projection_seconds=projection_total, draw_seconds=draw_total,
                  elapsed_seconds=time.perf_counter()-started, encoding_command=command, audio_mux_command=mux,
                  current_observations=context["current_observations"], same_complete_model_top_down=True,
                  review_output_wh=[OUTPUT_WIDTH,1080], no_Solve=True, no_new_observations=True, whole_shot_acceptance=False)
    write(out / "manifest.json", report)
    print(json.dumps(dict(stage="complete", video=str(final), manifest=str(out / "manifest.json"), frames=count,
                          elapsed_seconds=report["elapsed_seconds"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--packet", help="Current solver packet; inferred only from this complete state report when present")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiled-dir", required=True)
    parser.add_argument("--gpu-dll", required=True)
    parser.add_argument("--gpu-validation", required=True)
    parser.add_argument("--native-review", default=str(ROOT / "events/ankara_2026/programmes/okada_mei_sp/fit_v11/review_native.py"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    runtime = setup(args)
    try:
        checked = preflight(args, runtime)
        if not args.preflight_only:
            complete(args, runtime, checked)
    finally:
        runtime[4].at.close()


if __name__ == "__main__":
    main()
