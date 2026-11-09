"""Render a frozen-camera, shoe-only programme candidate on original exposures."""

from argparse import ArgumentParser
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np

from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.review.render import source_frames, validate_video
from scripts.fit_programme_shoes import PROGRAMME, SIZE, projected_world, read, emit


COLOURS = [(241, 220, 36), (60, 191, 255)]  # BGR: cyan / amber


class FootOverlay:
    def __init__(self, result):
        upper = result["upper_local_m"]
        lower = result["lower_local_m"]
        faces = result["lower_faces"]
        parts = result["lower_face_parts"]
        vertices = [upper]
        self.upper_faces = result["upper_faces"]
        self.parts = []
        for part in sorted(set(parts.tolist()) - {"steel_runner"}):
            ids = np.unique(faces[parts == part])
            first = sum(map(len, vertices))
            self.parts.append((part, slice(first, first+len(ids))))
            vertices.append(lower[ids])
        self.runner = []
        web = np.array([0., 0., float(result["blade_web_height_m"])])
        for runner in result["runner_local_m"]:
            path = runner[::4]
            first = sum(map(len, vertices))
            self.runner.append(slice(first, first+2*len(path)))
            vertices.extend((path, path+web))
        self.vertices = np.concatenate(vertices)

    def paint(self, raw, uv, side, weak, *, above):
        bound0 = np.clip(np.floor(uv.min(axis=0)-3).astype(int), 0, SIZE-1)
        bound1 = np.clip(np.ceil(uv.max(axis=0)+4).astype(int), 1, SIZE)
        w, h = bound1-bound0
        if min(w, h) < 4:
            return
        pts = np.rint(uv-bound0).astype(np.int32)
        shell = np.zeros((h, w), dtype=np.uint8)
        steel = np.zeros_like(shell)
        for tri in pts[self.upper_faces]:
            cv2.fillConvexPoly(shell, tri, 255)
        for part, ix in self.parts:
            layer = shell if part in {"outsole", "heel_block"} else steel
            cv2.fillConvexPoly(layer, cv2.convexHull(pts[ix]), 255)
        for ix in self.runner:
            lower, upper = np.split(pts[ix], 2, axis=0)
            for a, b, c, d in zip(lower[:-1], lower[1:], upper[1:], upper[:-1], strict=True):
                cv2.fillConvexPoly(steel, np.array([a,b,c,d]), 255)
        if above:
            steel[shell > 0] = 0
        else:
            shell[steel > 0] = 0
        x0, y0 = bound0
        x1, y1 = bound1
        crop = raw[y0:y1, x0:x1]
        colour = COLOURS[side]
        line = tuple(int(.64*x + .36*150) for x in colour) if weak else colour
        for mask, opacity in [(shell, .12), (steel, .29)]:
            hit = mask > 0
            crop[hit] = (crop[hit]*(1-opacity) + np.asarray(line)*opacity).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, contours, -1, line, 1, cv2.LINE_AA)
        if weak and (shell.any() or steel.any()):
            xy = (max(5,x0), max(25,y0-4))
            cv2.putText(raw, "?L" if side == 0 else "?R", xy, cv2.FONT_HERSHEY_SIMPLEX, .47, line, 1, cv2.LINE_AA)


def audio_encoder(ffmpeg, video, target, error, time_base, first_pts, count, step_pts):
    offset = first_pts*time_base
    duration = count*step_pts*time_base
    rate = 1/(step_pts*time_base)
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "1920x1080",
               "-framerate", str(rate), "-i", "pipe:0", "-ss", f"{float(offset):.6f}", "-i", str(video),
               "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1", "-map_chapters", "-1",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-b:v", "0",
               "-fps_mode", "passthrough", "-video_track_timescale", str(time_base.denominator),
               "-af", f"atrim=duration={float(duration):.6f},asetpts=PTS-STARTPTS",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def review_events(pts, shots, weak, overlap):
    events = []
    for side, name in enumerate(("left", "right")):
        take = np.flatnonzero(weak[:, side])
        for run in np.split(take, np.flatnonzero((np.diff(take) != 1) | (shots[take[1:]] != shots[take[:-1]]))+1):
            if not len(run):
                continue
            a, b = run[0], run[-1]
            events.append(dict(foot=name, shot=str(shots[a]), first_pts=int(pts[a]), last_pts=int(pts[b]),
                               first_video_frame=int(a), last_video_frame=int(b),
                               exposures=len(run), overlap_exposures=int(overlap[run].sum())))
    return sorted(events, key=lambda x: (x["first_pts"], x["foot"]))


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("project", "candidate", "ffmpeg"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    out = args.candidate.parent
    target = out/"okada_9shots_two_skates.mp4"
    pending = out/"okada_9shots_two_skates.pending.mp4"
    receipt = out/"video_receipt.json"
    if target.exists() or pending.exists() or receipt.exists():
        raise FileExistsError("Use a new candidate directory for a new export")
    pts, shots, tb = required_clock(read(args.project/PROGRAMME/"rebuild_scope.json"))
    time_base = Fraction(*tb)
    with np.load(args.candidate, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["source_pts"], pts)
        np.testing.assert_array_equal(result["shot_ids"], shots)
        np.testing.assert_array_equal(result["time_base"], tb)
        origin, rotation = result["origin_world_m"], result["rotation_world"]
        weak, overlap = result["inferred"], result["identity_overlap"]
        media = Path(str(result["source_video_path"]))
        media_sha = str(result["source_video_sha256"])
        camera_sha = str(result["camera_sha256"])
        record = str(result["camera_record"])
        ratio = float(result["length_ratio"])
        scale = float(result["solver_metres_per_scene_unit"])
        geometry = FootOverlay(result)
    if len(pts) != 8329 or len(set(shots.tolist())) != 9 or digest(media) != media_sha:
        raise ValueError("Require the complete exact source and its registered bytes")
    cv2.setNumThreads(1)
    packet = dict(pts=pts, image_height=int(SIZE[1]), metres_per_scene_unit=scale)
    camera = ProgrammeCamera(args.project, packet, shots, record_path=record)
    if camera.info["state_sha256"] != camera_sha:
        raise ValueError("Frozen camera changed")
    adjacent_cosine = (np.sum(rotation[1:]*rotation[:-1], axis=(-1, -2))-1.)*.5
    jumps = ((shots[1:] == shots[:-1])[:, None]) & (adjacent_cosine < .5)
    weak = weak.copy()
    weak[1:] |= jumps
    weak[:-1] |= jumps
    # Every source exposure is projected from the saved rigid pose using
    # the actual native camera. No body mesh is loaded for this render.
    started = time.perf_counter()
    source = dict(width=int(SIZE[0]), height=int(SIZE[1]), time_base_num=int(tb[0]), time_base_den=int(tb[1]))
    reader = iter(source_frames(media, source, pts))
    camera.core.motion_projection_threads = 8
    centres = camera.midpoint["centre"] * camera.info["metres_per_scene_unit"]
    above = np.sum((centres[:, None] - origin) * rotation[:, :, :, 2], axis=-1) > 0
    emit("render_start", path=str(target), exposures=len(pts))
    with (out/"encoder.log").open("xb") as errors:
        encoder = audio_encoder(args.ffmpeg, media, pending, errors, time_base, int(pts[0]), len(pts), int(pts[1]-pts[0]))
        try:
            for i in range(0, len(pts), 128):
                j = min(i+128, len(pts))
                uv = projected_world(camera, geometry.vertices, origin[i:j], rotation[i:j], np.arange(i, j), ratio)
                for k in range(i, j):
                    stamp, frame = next(reader)
                    if stamp != pts[k]:
                        raise ValueError("Decoded source frame differs from candidate")
                    for foot in range(2):
                        geometry.paint(frame, uv[k-i, foot], foot, bool(weak[k, foot]), above=above[k, foot])
                    seconds = float(stamp*time_base)
                    time_label = f"{int(seconds//60):02d}:{seconds%60:06.3f}"
                    cv2.rectangle(frame, (12, 1018), (545, 1063), (29, 35, 44), -1)
                    cv2.putText(frame, "L", (28,1048), cv2.FONT_HERSHEY_SIMPLEX, .62, COLOURS[0], 2, cv2.LINE_AA)
                    cv2.putText(frame, "R", (65,1048), cv2.FONT_HERSHEY_SIMPLEX, .62, COLOURS[1], 2, cv2.LINE_AA)
                    label = f"? uncertain   {shots[k].upper()}   SOURCE {time_label}"
                    cv2.putText(frame, label, (105,1047), cv2.FONT_HERSHEY_SIMPLEX, .54, (239,242,248), 1, cv2.LINE_AA)
                    encoder.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                if i % 512 == 0 or j == len(pts):
                    emit("render", exposures=j, total=len(pts), elapsed_s=round(time.perf_counter()-started, 1))
            if next(reader, None) is not None:
                raise RuntimeError("Unexpected additional frame")
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed. See encoder.log")
        finally:
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
    exact = validate_video(pending, pts, time_base)
    with av.open(str(pending)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("Missing source audio")
    pending.rename(target)
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(dict(schema="rinkspace.independent-skates-video.v1", video=str(target), video_sha256=digest(target),
                       candidate=str(args.candidate), candidate_sha256=digest(args.candidate),
                       source=str(media), source_sha256=media_sha, camera_sha256=camera_sha,
                       first_source_pts=int(pts[0]), last_source_pts=int(pts[-1]), time_base=list(tb),
                       native_clock_check=exact, pose_interpolation="within shot: linear origin and SLERP rotation",
                       shoe_pose_time="native exposure, rigid during the shoe readout",
                       weak_foot_exposures=int(weak.sum()), abrupt_same_shot_turns=int(jumps.sum()), weak_events=review_events(pts, shots, weak, overlap)), f, indent=2)
    emit("render_saved", path=str(target), exact_frames=len(pts), elapsed_s=round(time.perf_counter()-started, 1))


if __name__ == "__main__":
    main()
