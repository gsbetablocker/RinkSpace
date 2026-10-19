"""Render only the statistics ending and a short source-context preview.

No main-programme frame or current full demo is overwritten. The ending-only
file starts with an IDR and can be joined after the retained programme later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src"), str(ROOT/"calibration/program_tracking")]

import av
import cv2
import numpy as np
from calibration.program_tracking.demo_outro import PlaneTexture
from calibration.program_tracking.demo_stats_layout import StatsLayout, LeftPanelTransform
from calibration.program_tracking.demo_outro_stats import load_stats, summary_layers, OUT
from calibration.program_tracking.demo_v3_sample import V3Sample
from calibration.program_tracking.render_demo_full import FIRST, LAST, MAIN_FRAMES
from calibration.program_tracking.render_demo_last_shot import decode, digest, write_json
from calibration.program_tracking.demo_particle_pass import _encoder_quality
from calibration.program_tracking.render_result import _ffmpeg_executable, _available_encoder
from calibration.program_tracking.demo_source import check_output_path

FPS = 50
WH = (1920, 1080)
CONTEXT_FRAMES = 100
OUTRO_FRAMES = 650
CONTEXT_FIRST = LAST-(CONTEXT_FRAMES-1)*256


def make_layout(stats):
    display = {key: stats[key] for key in
               ("programme_seconds", "active_seconds", "active_distance_m",
                "average_speed_kmh", "max_speed_kmh", "coverage_percent")}
    display.update(stats["athlete"])
    event = stats["event"]; nation = stats["athlete"]
    display["event_line"] = (f'{nation["nation_zh"]} {nation["nation_code"]}  ·  '
                             f'{event["display_name"]}  ·  {event["discipline"]}')
    display["coverage_note"] = (f'σ = {stats["coverage"]["sigma_m"]:g} m'
                                f' · 阈值 {stats["coverage"]["threshold"]:.2f}')
    curve = stats["speed_curve"]
    return StatsLayout(display, curve["programme_time_seconds"],
                       curve["speed_kmh"], curve["excluded_mask"],
                       stats["exclusion_intervals"])


class StatsEnding:
    def __init__(self, sample, stats):
        self.sample = sample
        self.stats = stats
        self.layout = make_layout(stats)
        self.native = next(decode(str(sample.packet["source_path"]), LAST, LAST))[1]
        self.final, self.base, self.texture, self.foreground = sample.draw(self.native, LAST)
        self.transform = LeftPanelTransform(sample.packet["camera25"][-1],
                                            sample.layers.bounds_xy)
        # The particle overlay uses the same camera as the ice during its .55s fade.
        sample.outro = self.transform
        sample.particles.last_foreground = self.foreground
        layers = summary_layers(sample)
        self.summary = layers.texture_prepared(len(layers.times_s)-1, summary=True)
        if self.summary.wh != self.texture.wh:
            raise ValueError("Summary texture shape changed the continuity domain")
        self._hold = None
        zero = self.transform.frame(self.base, self.texture, 0.,
                                    foreground_mask=self.foreground)
        zero = sample.particles.frame(zero, MAIN_FRAMES-1, foreground=self.foreground)
        self.continuity_max_byte_error = int(np.max(np.abs(
            zero.astype(np.int16)-self.final.astype(np.int16))))
        if self.continuity_max_byte_error:
            raise ValueError("Ending does not start at the same final competition image")

    def frame(self, elapsed):
        elapsed = float(elapsed)
        if elapsed >= 4.0 and self._hold is not None:
            return self._hold
        u = np.clip(elapsed/.55, 0., 1.); u = u*u*(3-2*u)
        texture = (self.summary if u >= 1 else PlaneTexture(
            self.texture.premultiplied*(1-u)+self.summary.premultiplied*u))
        frame = self.transform.frame(self.base, texture, elapsed,
                                     foreground_mask=self.foreground)
        if 0 < elapsed < .55:
            index = MAIN_FRAMES+round(elapsed*FPS)-1
            frame = self.sample.particles.frame(frame, index)
        frame = self.layout.composite(frame, elapsed)
        if elapsed >= 4.0: self._hold = frame
        return frame


def previews(stats_path=OUT/"stats.json", output_dir=OUT):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sample = V3Sample(); stats = load_stats(stats_path)
    ending = StatsEnding(sample, stats)
    static = ending.frame(4.)
    page = output_dir/"stats_page.png"
    cv2.imwrite(str(page), static)
    times = [0., .2, .55, 1., 1.7, 2.4, 3., 4.]
    tiles = []
    for elapsed in times:
        # Do not use frame(0) here: that omits the final decorative particles.
        frame = ending.final if elapsed == 0 else ending.frame(elapsed)
        tile = cv2.resize(frame, (480, 270))
        cv2.putText(tile, f"ending +{elapsed:.2f}s", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 245, 250), 1, cv2.LINE_AA)
        tiles.append(tile)
    strip = output_dir/"outro_transition.jpg"
    cv2.imwrite(str(strip), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    geometry = ending.transform.selfcheck(sample.layers.texture_wh)
    report = dict(status="preview_complete", stats_page=str(page),
                  transition_preview=str(strip),
                  start_composite_max_byte_error=ending.continuity_max_byte_error,
                  presentation_geometry=geometry,
                  summary_window=stats["clock"], layout_map_rect=[74, 219, 1234, 725])
    write_json(output_dir/"layout_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def _video_encoder(target, frames, output_dir, label, ffmpeg):
    args, name = _available_encoder(ffmpeg)
    args = _encoder_quality(args, name)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "1920x1080",
               "-framerate", "50", "-i", "pipe:0", *args, "-pix_fmt", "yuv420p",
               "-an", "-frames:v", str(frames), "-video_track_timescale", "12800",
               "-movflags", "+faststart", str(target)]
    write_json(output_dir/f"{label}_command.json", command)
    log = (output_dir/f"{label}_encode.stderr.log").open("wb")
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=log,
                               creationflags=0x08000000)
    return process, log, name


def _mux_audio(video, target, output_dir, ffmpeg, *, source=None):
    duration = (CONTEXT_FRAMES+OUTRO_FRAMES)/FPS if source else OUTRO_FRAMES/FPS
    if source:
        input_args = ["-ss", str(CONTEXT_FIRST/12800.), "-i", str(source)]
        audio_filter = ("atrim=duration=2,afade=t=out:st=1.55:d=0.45,"
                        f"apad=whole_dur={duration},aresample=48000,"
                        "asetnsamples=n=1024:p=1,asetpts=N/SR/TB")
    else:
        input_args = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_filter = "asetnsamples=n=1024:p=1,asetpts=N/SR/TB"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(video), *input_args, "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
               "-ar", "48000", "-t", str(duration), "-video_track_timescale", "12800",
               "-movflags", "+faststart", str(target)]
    write_json(output_dir/f"{target.stem}_mux_command.json", command)
    result = subprocess.run(command, capture_output=True, creationflags=0x08000000)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf8", errors="replace"))


def verify_clock(path, frames):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != WH or stream.average_rate != FPS:
            raise ValueError("Sample format is not 1920x1080 native50")
        timestamps = []
        for frame in container.decode(stream):
            timestamps.append(float(frame.pts*frame.time_base))
        if len(timestamps) != frames or not np.allclose(
                timestamps, np.arange(frames)/FPS, atol=1e-9, rtol=0):
            raise ValueError("Sample decode is not complete continuous native50 CFR")
        video_time_base = str(stream.time_base)
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        pts = np.array([p.pts for p in container.demux(audio)
                        if p.pts is not None and p.size > 0], dtype=np.int64)
        if not np.all(np.diff(pts) == 1024):
            raise ValueError("Audio contains a sparse or discontinuous AAC timestamp")
    return dict(path=str(path), sha256=digest(path), bytes=Path(path).stat().st_size,
                video_frames=frames, decoded_video_frames=len(timestamps), fps=50,
                width=1920, height=1080, duration_seconds=frames/FPS,
                video_time_base=video_time_base, audio_packets=len(pts),
                audio_pts_step=1024, audio_sample_rate=48000)


def render(stats_path=OUT/"stats.json", output_dir=OUT):
    cv2.setNumThreads(1)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(stats_path); stats = load_stats(stats_path)
    started = time.perf_counter()
    sample = V3Sample()
    # Render only the final 100 source frames for viewing context. The main file
    # remains untouched and is not traversed or re-encoded.
    ffmpeg = _ffmpeg_executable()
    preview_raw = check_output_path(output_dir/"preview_video_only.mp4")
    outro_raw = check_output_path(output_dir/"outro_video_only.mp4")
    preview_proc, preview_log, encoder = _video_encoder(
        preview_raw, CONTEXT_FRAMES+OUTRO_FRAMES, output_dir, "preview", ffmpeg)
    outro_proc, outro_log, _ = _video_encoder(
        outro_raw, OUTRO_FRAMES, output_dir, "outro", ffmpeg)
    preview_count = 0; outro_count = 0; last_context = None
    try:
        for pts, native in decode(str(sample.packet["source_path"]), CONTEXT_FIRST, LAST):
            last_context = sample.draw(native, pts)[0]
            preview_proc.stdin.write(last_context.tobytes()); preview_count += 1
        ending = StatsEnding(sample, stats)
        if not np.array_equal(last_context, ending.final):
            raise ValueError("Context and ending do not use the same final state")
        for j in range(OUTRO_FRAMES):
            frame = ending.frame((j+1)/FPS)
            data = frame.tobytes()
            preview_proc.stdin.write(data); preview_count += 1
            outro_proc.stdin.write(data); outro_count += 1
            if (j+1) % 100 == 0:
                write_json(output_dir/"render_status.json", dict(
                    status="running", stage="outro", frames=j+1,
                    total_frames=OUTRO_FRAMES, elapsed_seconds=time.perf_counter()-started))
        if preview_count != CONTEXT_FRAMES+OUTRO_FRAMES or outro_count != OUTRO_FRAMES:
            raise ValueError("Short sample frame count changed")
    finally:
        preview_proc.stdin.close(); outro_proc.stdin.close()
        preview_return = preview_proc.wait(); outro_return = outro_proc.wait()
        preview_log.close(); outro_log.close()
    if preview_return or outro_return: raise RuntimeError("Short sample video encode failed")
    preview = check_output_path(output_dir/"RinkSpace_stats_outro_preview.mp4")
    outro = check_output_path(output_dir/"RinkSpace_stats_outro_only.mp4")
    _mux_audio(preview_raw, preview, output_dir, ffmpeg, source=sample.packet["source_path"])
    _mux_audio(outro_raw, outro, output_dir, ffmpeg)
    report = dict(status="complete", sample_only=True, main_video_unchanged=True,
                  preview=verify_clock(preview, CONTEXT_FRAMES+OUTRO_FRAMES),
                  ending_only=verify_clock(outro, OUTRO_FRAMES),
                  context_source_first_pts=CONTEXT_FIRST, context_source_last_pts=LAST,
                  context_source_time_seconds=[CONTEXT_FIRST/12800., LAST/12800.],
                  source_time_base="1/12800", context_frames=CONTEXT_FRAMES,
                  outro_frames=OUTRO_FRAMES, outro_seconds=OUTRO_FRAMES/FPS,
                  turn_seconds=3., chart_draw_seconds=[3., 4.],
                  full_page_hold_seconds=9.,
                  t0_max_byte_error=ending.continuity_max_byte_error,
                  statistics=dict(path=str(stats_path), sha256=digest(stats_path)),
                  summary_window=stats["clock"],
                  encoder=encoder, elapsed_seconds=time.perf_counter()-started)
    write_json(output_dir/"render_report.json", report)
    write_json(output_dir/"render_status.json", dict(status="complete", **{
        k: report[k] for k in ("outro_frames", "outro_seconds", "elapsed_seconds")}))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--stats", type=Path, default=OUT/"stats.json")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.previews: previews(args.stats, args.output_dir)
    else: render(args.stats, args.output_dir)


if __name__ == "__main__":
    main()
