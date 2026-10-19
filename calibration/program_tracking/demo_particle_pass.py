"""Append visible decorative particle billboards to the completed v2 sample.

Only particle centres use the unchanged ice-plane projection. Small screen
sprites remain round/diamond-shaped so shallow camera views cannot crush them
into a continuous white line. No heat texture is recomputed, no model changes.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]

import av
import cv2
import numpy as np
from calibration.program_tracking.render_demo_last_shot import (
    Sample, BASE_OUT, FIRST, LAST, FPS, WH, digest, write_json, decode)
from calibration.program_tracking.demo_outro import plane_homography, _ease
from calibration.program_tracking.render_result import _ffmpeg_executable, _available_encoder
from calibration.program_tracking.demo_source import check_output_path

DEFAULT_BASE = BASE_OUT/'v2/RinkSpace_last_shot_demo_v2.mp4'
DEFAULT_TARGET = BASE_OUT/'v2/RinkSpace_last_shot_demo_v2_final.mp4'
RADIUS_LIMITS = (1.2, 3.0)
MAX_PARTICLES = 128
ENDING_FADE = .55


def _smoothstep(value):
    u = float(np.clip(value, 0., 1.))
    return u*u*(3.-2.*u)


def _point(value):
    return tuple(np.rint(np.asarray(value)*16).astype(np.int32))


class ParticleOverlay:
    """Particle-only API. frame() modifies only a small BGR frame ROI in place.

    Encoded frame index0 corresponds to first_pts, at native50fps.
    No Sample.draw, texture, heat update or model-fitting function is called.
    """
    def __init__(self, sample=None, first_pts=FIRST, last_pts=LAST):
        self.first_pts, self.last_pts = int(first_pts), int(last_pts)
        if (self.last_pts < self.first_pts or self.first_pts % 256
                or self.last_pts % 256):
            raise ValueError('Particle range must be increasing official native50 PTS on the256-tick grid')
        self.sample = Sample(version='v2') if sample is None else sample
        self.layers = self.sample.layers
        self.native_frames = (self.last_pts-self.first_pts)//256+1
        wanted = np.arange(self.first_pts, self.last_pts+256, 256, dtype=np.int64)
        self.indices = np.asarray([self.sample.lookup[int(p)] for p in wanted], dtype=np.int64)
        if not np.array_equal(self.sample.packet['pts'][self.indices], wanted):
            raise ValueError('Particle pass source packet does not bind every official native50 PTS')
        self.last_index = int(self.indices[-1])
        self.last_time = float(self.sample.times[self.last_index])
        self.last_foreground = None
        self.frames_modified = 0
        self.maximum_alive = 0
        self.maximum_screen_particles = 0

    def _sprites(self, camera, now, strength):
        layer = self.layers
        begin = int(np.searchsorted(layer._particle_times, now-1.8, side='left'))
        end = int(np.searchsorted(layer._particle_times, now, side='right'))
        indices = np.arange(begin, end)
        if not len(indices) or strength <= 0:
            return []
        remaining = 1.-(now-layer._particle_times[indices])/layer._particle_lifetimes[indices]
        gain = layer._particle_gains[indices]*np.maximum(remaining, 0.)**1.2*strength
        keep = (remaining > 0.) & (gain > 1./255.)
        indices, gain = indices[keep], gain[keep]
        self.maximum_alive = max(self.maximum_alive, len(indices))
        if len(indices) > MAX_PARTICLES:
            selected = np.sort(np.argsort(gain)[-MAX_PARTICLES:])
            indices, gain = indices[selected], gain[selected]
        if not len(indices):
            return []
        points = layer._particle_uv[indices]
        h = plane_homography(camera, layer.bounds_xy, layer.texture_wh)
        projected = np.column_stack((points, np.ones(len(points)))) @ h.T
        denom = projected[:, 2]
        finite = np.isfinite(projected).all(axis=1) & (np.abs(denom) > 1e-9)
        screen = np.divide(projected[:, :2], denom[:, None],
                           out=np.full((len(points), 2), np.nan), where=np.abs(denom[:, None]) > 1e-9)
        # Keep the same plane-side visibility convention as the main renderer.
        xmin, ymin, xmax, ymax = layer.bounds_xy
        tw, th = layer.texture_wh
        world = np.column_stack((xmin+points[:, 0]*(xmax-xmin)/(tw-1),
                                 ymax-points[:, 1]*(ymax-ymin)/(th-1), np.zeros(len(points))))
        rotation = cv2.Rodrigues(np.asarray(camera[3:6], float))[0]
        finite &= ((world-camera[:3]) @ rotation[2] > .05)
        finite &= (screen[:, 0] >= -12) & (screen[:, 0] < WH[0]+12)
        finite &= (screen[:, 1] >= -12) & (screen[:, 1] < WH[1]+12)
        result = []
        for j in np.flatnonzero(finite):
            q = projected[j]
            jacobian = (h[:2, :2]*q[2]-q[:2, None]*h[2, :2])/(q[2]*q[2])
            scale = math.sqrt(abs(float(np.linalg.det(jacobian))))
            radius = float(np.clip(float(layer._particle_radii[indices[j]])/16.*scale,
                                   *RADIUS_LIMITS))
            result.append((screen[j], radius, float(gain[j]),
                           bool(layer._particle_sliver[indices[j]]),
                           float(layer._particle_angles[indices[j]]), int(indices[j])))
        self.maximum_screen_particles = max(self.maximum_screen_particles, len(result))
        return result

    def frame(self, frame, frame_index, foreground=None):
        if frame.shape != (WH[1], WH[0], 3) or frame.dtype != np.uint8:
            raise ValueError('Particle overlay expects uint8 BGR1920x1080')
        index = int(frame_index)
        if index < 0:
            raise ValueError('Encoded frame index must be nonnegative')
        if index < self.native_frames:
            source_index = int(self.indices[index])
            now = float(self.sample.times[source_index])
            camera = self.sample.packet['camera25'][source_index]
            strength, visibility = 1., 1.
            if foreground is None:
                foreground = self.sample.foreground(source_index)
        else:
            elapsed = (index-self.native_frames+1)/FPS
            if elapsed >= ENDING_FADE:
                return frame
            now = self.last_time
            camera = self.sample.outro.camera_at(elapsed)
            strength = 1.-_smoothstep(elapsed/ENDING_FADE)
            visibility = 1.-_ease(elapsed/self.sample.outro.fade_seconds)
            if self.last_foreground is None:
                self.last_foreground = self.sample.foreground(self.last_index)
            foreground = self.last_foreground
        sprites = self._sprites(camera, now, strength)
        if not sprites:
            return frame
        centers = np.asarray([item[0] for item in sprites])
        margin = math.ceil(RADIUS_LIMITS[1]*2.4)+2
        low = np.maximum(np.floor(centers.min(axis=0)).astype(int)-margin, (0, 0))
        high = np.minimum(np.ceil(centers.max(axis=0)).astype(int)+margin+1, WH)
        x0, y0, x1, y1 = int(low[0]), int(low[1]), int(high[0]), int(high[1])
        if x1 <= x0 or y1 <= y0:
            return frame
        shape = (y1-y0, x1-x0)
        halo, rim, core = [np.zeros(shape, np.uint8) for _ in range(3)]
        offset = np.array([x0, y0])
        for center, radius, gain, sliver, angle, particle in sprites:
            local = center-offset
            cv2.circle(halo, _point(local), round(radius*2.3*16), round(gain*.20*255),
                       -1, cv2.LINE_AA, shift=4)
            cv2.circle(rim, _point(local), round(radius*1.25*16), round(gain*.48*255),
                       -1, cv2.LINE_AA, shift=4)
            value = round(gain*.98*255)
            if sliver:
                delta = np.array([math.cos(angle), math.sin(angle)])*(1.+particle%3*.5)
                cv2.line(core, _point(local-delta), _point(local+delta), value,
                         1, cv2.LINE_AA, shift=4)
            elif particle % 4 == 0:
                polygon = np.array([local+[0, -radius], local+[radius, 0],
                                    local+[0, radius], local+[-radius, 0]])
                cv2.fillConvexPoly(core, np.rint(polygon*16).astype(np.int32), value,
                                   lineType=cv2.LINE_AA, shift=4)
            else:
                cv2.circle(core, _point(local), round(radius*16), value,
                           -1, cv2.LINE_AA, shift=4)
        visible = 1.-np.asarray(foreground[y0:y1, x0:x1], np.float32)*visibility
        result = frame[y0:y1, x0:x1].astype(np.float32)
        for alpha8, colour in ((halo, np.array([255., 210., 75.], np.float32)),
                               (rim, np.array([255., 230., 145.], np.float32)),
                               (core, np.array([255., 249., 226.], np.float32))):
            alpha = (alpha8.astype(np.float32)/255.*visible)[..., None]
            result *= 1.-alpha
            result += colour[None, None, :]*alpha
        frame[y0:y1, x0:x1] = np.clip(result+.5, 0., 255.).astype(np.uint8)
        self.frames_modified += 1
        return frame

    def parameters(self):
        return dict(style='ice-plane centres with screen-space round/diamond/sliver billboards',
                    seed=20261016, particle_births=len(self.layers._particle_times),
                    screen_core_radius_px=list(RADIUS_LIMITS), maximum_alive=MAX_PARTICLES,
                    halo_radius_multiple=2.3, halo_gain=.20, rim_gain=.48, core_gain=.98,
                    core_bgr=[255, 249, 226], halo_bgr=[255, 210, 75],
                    sprite_scaling='homography local Jacobian determinant geometric-mean, then clamp',
                    preserves_seeded_particle_times_positions_lifetimes=True,
                    foreground='Sample.foreground person/broadcast protections',
                    summary_particle_fade_seconds=ENDING_FADE,
                    summary_particle_strength='1 minus cubic smoothstep(elapsed/.55)',
                    summary_foreground_strength='1 minus OutroTransform quintic ease(elapsed/.55)',
                    summary_after_fade='Base pixels passed through without any particle drawing')


class UniformEnding:
    """Rebuild only the ending from the last official native source frame.

    Keeps one final raw/base/texture/foreground state in memory. The uniform
    summary texture is generated lazily once by Sample.ending_texture.
    """
    def __init__(self, sample):
        self.sample = sample
        self.state = None

    def frame(self, elapsed):
        if self.state is None:
            native = next(decode(str(self.sample.packet['source_path']), LAST, LAST))[1]
            _, base, texture, foreground = self.sample.draw(native, LAST)
            self.state = (base, texture, foreground)
        base, texture, foreground = self.state
        return self.sample.outro.frame(base, self.sample.ending_texture(texture, elapsed),
                                       elapsed, foreground_mask=foreground)


def _encoder_quality(args, name):
    args = list(args)
    option = '-cq' if 'nvenc' in name else '-crf'
    if option in args:
        args[args.index(option)+1] = '17'
    else:
        args += [option, '17']
    return args


def _verify_encoded_clock(target, expected_frames):
    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != FPS or (stream.width, stream.height) != WH:
            raise ValueError('Output is not1920x1080 at50fps')
        packet_times = [Fraction(packet.pts)*stream.time_base for packet in container.demux(stream)
                        if packet.pts is not None and packet.size > 0]
        packet_times.sort()
        if len(packet_times) != expected_frames or any(t != Fraction(i, FPS) for i, t in enumerate(packet_times)):
            raise ValueError('Encoded output packet clock lost native50 CFR frame count/order')
        audio_streams = len(container.streams.audio)
    return dict(video_packets=len(packet_times), first_seconds=0.,
                last_seconds=(expected_frames-1)/FPS, interval_seconds=1/FPS,
                audio_streams=audio_streams)


def render_particle_pass(base=DEFAULT_BASE, target=DEFAULT_TARGET, progress_every=100):
    base, target = Path(base), Path(target)
    if base.resolve() == target.resolve():
        raise ValueError('Particle pass must preserve its base video')
    target = check_output_path(target)
    report_path = check_output_path(target.parent/'particle_pass.json')
    log_path = check_output_path(target.parent/'particle_pass.stderr.log')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    overlay = ParticleOverlay()
    ending = UniformEnding(overlay.sample)
    input_sha = digest(base)
    ffmpeg = _ffmpeg_executable()
    args, name = _available_encoder(ffmpeg)
    args = _encoder_quality(args, name)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1920x1080',
               '-framerate', '50', '-i', 'pipe:0', '-i', str(base),
               '-map', '0:v:0', '-map', '1:a:0?', *args, '-pix_fmt', 'yuv420p',
               '-c:a', 'copy', '-movflags', '+faststart', str(target)]
    report = dict(status='running', input=dict(path=str(base), sha256=input_sha),
                  output=str(target), parameters=overlay.parameters(), encoder=name,
                  encoder_args=args, command=command,
                  native_clock=dict(source_time_base='1/12800', first_pts=FIRST, last_pts=LAST,
                                    source_pts_step=256, fps=50, encoded_interval_seconds=.02),
                  original_model_unmodified=True, native_heat_recomputed=False, outro_final_heat_recomputed_once=True,
                  base_video_unmodified=True, audio_codec='copy',
                  outro_regenerated_with_uniform_summary=True,
                  outro_rebuilt_frames=round(overlay.sample.outro.total_seconds*FPS),
                  outro_rebuild='Only ending rebuilt from last official native source with Sample.ending_texture; main2255 frames retain encoded base plus billboards')
    write_json(report_path, report)
    count, source_count = 0, 0
    with log_path.open('wb') as log, av.open(str(base)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != FPS or (stream.width, stream.height) != WH:
            raise ValueError('Base video must be1920x1080 native50 CFR')
        expected_frames = overlay.native_frames + round(overlay.sample.outro.total_seconds*FPS)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=log, creationflags=0x08000000)
        try:
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or Fraction(frame.pts)*frame.time_base != Fraction(index, FPS):
                    raise ValueError(f'Base video native50 PTS mismatch at decoded frame{index}')
                if index >= expected_frames:
                    raise ValueError('Base has unexpected extra frames')
                if index < overlay.native_frames:
                    source_pts = FIRST+256*index
                    if int(overlay.sample.packet['pts'][overlay.indices[index]]) != source_pts:
                        raise ValueError('Official packet/source PTS binding mismatch')
                    source_count += 1
                if index < overlay.native_frames:
                    native = frame.to_ndarray(format='bgr24')
                else:
                    elapsed = (index-overlay.native_frames+1)/FPS
                    native = ending.frame(elapsed)
                rendered = overlay.frame(native, index)
                process.stdin.write(rendered.tobytes())
                count += 1
                if count % max(1, int(progress_every)) == 0:
                    progress = dict(status='running', stage='particle_append', frames=count,
                                    total=expected_frames, elapsed_seconds=time.perf_counter()-started)
                    write_json(report_path, {**report, **progress})
                    print(json.dumps(progress), flush=True)
            if count != expected_frames or source_count != overlay.native_frames:
                raise ValueError('Base frame count does not match bound source plus ending')
        except BaseException as error:
            process.stdin.close()
            process.wait()
            write_json(report_path, {**report, 'status': 'failed', 'frames': count, 'error': str(error)})
            raise
        else:
            process.stdin.close()
            return_code = process.wait()
            if return_code:
                error = 'ffmpeg failed: '+log_path.read_text(encoding='utf8', errors='replace')[-3000:]
                write_json(report_path, {**report, 'status': 'failed', 'frames': count, 'error': error})
                raise RuntimeError(error)
    clock = _verify_encoded_clock(target, count)
    report.update(status='complete', frames=count, native_source_frames=source_count,
                  frames_modified=overlay.frames_modified, maximum_alive_seen=overlay.maximum_alive,
                  maximum_screen_particles_seen=overlay.maximum_screen_particles,
                  encoded_clock=clock, output_sha256=digest(target), output_bytes=target.stat().st_size,
                  wall_seconds=time.perf_counter()-started)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def preview_particle_pass(base=DEFAULT_BASE, frame_index=450, output=None):
    """Save one actual encoded-base preview; never start the full append pass."""
    base = Path(base)
    output = check_output_path(base.parent/'particle_pass_preview.jpg' if output is None else output)
    overlay = ParticleOverlay()
    wanted = Fraction(int(frame_index), FPS)
    original = None
    with av.open(str(base)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != FPS:
            raise ValueError('Preview base must be50fps')
        container.seek(int(wanted/stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = Fraction(frame.pts)*frame.time_base
            if stamp < wanted:
                continue
            if stamp != wanted:
                raise ValueError('Requested preview frame is absent from native50 base')
            original = frame.to_ndarray(format='bgr24')
            break
    if original is None:
        raise ValueError('Preview endpoint absent')
    if int(frame_index) >= overlay.native_frames:
        elapsed = (int(frame_index)-overlay.native_frames+1)/FPS
        original = UniformEnding(overlay.sample).frame(elapsed)
    enhanced = overlay.frame(original.copy(), int(frame_index))
    changed = np.any(enhanced != original, axis=2).astype(np.uint8)
    if not cv2.imwrite(str(output), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise IOError('Could not save particle preview')
    report = dict(preview=str(output), frame_index=int(frame_index), encoded_seconds=float(wanted),
                  official_pts=FIRST+256*int(frame_index),
                  particles=overlay.maximum_screen_particles, changed_pixels=int(changed.sum()),
                  changed_bbox_xywh=list(map(int, cv2.boundingRect(changed))),
                  no_heat_recomputed=True, foreground_applied=True, full_pass_started=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--preview-frame', type=int, default=450)
    parser.add_argument('--preview-output', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.preview:
        preview_particle_pass(args.base, args.preview_frame, args.preview_output)
    else:
        render_particle_pass(args.base, args.output, args.progress_every)
