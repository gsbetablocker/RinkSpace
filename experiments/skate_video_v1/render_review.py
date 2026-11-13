"""Render source evidence and the source / old / new comparison at the original cadence."""

from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from evidence import colour_status
from inputs import Inputs, emit, geometry, read
from model import SkateModel
from projection import Native, PoseCurve, project_available
from rinkspace.review.render import validate_video

FOOT_COLOUR = ((69, 237, 99), (248, 177, 56))
STEEL_COLOUR = ((40, 252, 250), (224, 166, 255))
ALTERNATIVE_COLOUR = (252, 102, 128)
SKEL = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def lines(img, xy, colour, *, close=False, thick=1):
    if xy is None or len(xy) < 2 or not np.isfinite(xy).all():
        return
    cv2.polylines(img, [np.rint(np.clip(xy, -4000, 8000)).astype(np.int32)], close, colour, thick, cv2.LINE_AA)


def text(frame, label, origin, scale=.65, colour=(245, 245, 245)):
    cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def image_marks(frame, packet, i):
    for foot, colour in enumerate(FOOT_COLOUR):
        lines(frame, packet["boot"][i, foot], colour, close=True, thick=2)
        lines(frame, packet["boot_alt"][i, foot], ALTERNATIVE_COLOUR, close=True)
        lines(frame, packet["steel"][i, foot], STEEL_COLOUR[foot], thick=2)
        lines(frame, packet["steel_alt"][i, foot], ALTERNATIVE_COLOUR)
        ends = packet["heel_toe"][i, foot]
        if np.isfinite(ends).all():
            uncertain = "" if packet["heel_toe_ordered"][i, foot] else "?"
            for label, point in zip(("H", "T"), ends):
                point = tuple(np.rint(point).astype(int))
                cv2.circle(frame, point, 3, colour, 1, cv2.LINE_AA)
                text(frame, label + uncertain, (point[0] + 4, point[1] - 5), .42, colour)


def zoom(frame, xy, width, height, *, outline=None):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    points = np.asarray(xy).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=-1)]
    if not len(points):
        text(canvas, "no associated region", (18, height // 2), .55)
        return canvas
    lo = np.maximum(0, np.floor(points.min(axis=0) - 18)).astype(int)
    hi = np.minimum(frame.shape[1::-1], np.ceil(points.max(axis=0) + 18)).astype(int)
    x, y = lo
    r, b = hi
    if r <= x or b <= y:
        return canvas
    chip = frame[y:b, x:r]
    if outline is not None:
        polygon, colour = outline
        chip = chip.copy()
        lines(chip, polygon-lo, colour, close=True, thick=2)
    scale = min((width - 16) / chip.shape[1], (height - 14) / chip.shape[0], 5.)
    resized = cv2.resize(chip, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    oy, ox = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def stamp(inputs, i):
    return f"{inputs.shots[i].upper()}  SOURCE {inputs.times[i]:07.3f}s  PTS {inputs.pts[i]}"


def image_frame(inputs, src, packet, i):
    frame = src.copy()
    image_marks(frame, packet, i)
    panel = np.full((1080, 480, 3), 24, np.uint8)
    text(panel, stamp(inputs, i), (18, 32), .58)
    for f in range(2):
        y = 74 + f * 455
        label = "L" if f == 0 else "R"
        boot, steel = (packet[name][i, f] for name in ("boot_status", "steel_status"))
        boot_state = colour_status(boot) if boot else "unassociated"
        if boot and "repair_method" in packet and packet["repair_method"][i, f] == 4:
            anchor = packet["repair_anchor_pts"][i, f, 0]
            boot_state = "source prompt" if anchor == inputs.pts[i] else "source track"
        text(panel, label + "  Boot: " + boot_state, (19, y), .78, FOOT_COLOUR[f])
        if "steel_anchor_pts" in packet:
            state = "unconfirmed" if not steel else ("source read" if steel == 2 else "source track")
            text(panel, "Cutting edge: " + state, (19, y + 32), .70, STEEL_COLOUR[f])
        else:
            text(panel, "Steel: " + (colour_status(steel) if steel else "unassociated"), (19, y + 32), .70, STEEL_COLOUR[f])
        poly = np.concatenate((packet["boot"][i, f], packet["steel"][i, f]), axis=0)
        if not boot and not steel:
            poly = inputs.old_gear[i, f, :14]
        panel[y + 48:y + 402] = zoom(frame, poly, 480, 354)
        if packet["roi_partial"][i, f] or packet["overlap"][i]:
            text(panel, "Partial / overlap possible", (19, y + 426), .61)
        elif not boot and not steel:
            text(panel, "Search crop", (19, y + 426), .61)
    text(panel, "H? T?  Both directions retained", (18, 1012), .61)
    text(panel, "Purple: competing pixel hypothesis", (18, 1045), .59, ALTERNATIVE_COLOUR)
    return np.hstack((frame, panel))


def chapters(inputs, output, stem):
    path = output / "navigation.json"
    document = read(path) if path.is_file() else {"cuts": [], "events": []}
    entries = {int(inputs.pts[0]): "Programme start"}
    for row in document["cuts"]:
        entries[row["pts"]] = row["label"]
    for row in document["events"]:
        stamp = row["start_pts"]
        entries[stamp] = entries.get(stamp, "") + (" | " if stamp in entries else "") + row["label"]
    starts = sorted(entries)
    starts.append(int(inputs.pts[-1] + (inputs.pts[-1] - inputs.pts[-2])))
    path = output / f"{stem}_chapters.ffmeta"
    chunks = [";FFMETADATA1"]
    for a, b in zip(starts[:-1], starts[1:]):
        label = entries[a].replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
        chunks.append(f"[CHAPTER]\nTIMEBASE={inputs.tb[0]}/{inputs.tb[1]}\nSTART={a - starts[0]}\nEND={b - starts[0]}\ntitle={label}")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return path


def render(inputs, output, filename, frames, size, *, refresh=False):
    target = output / filename
    pending = target.with_suffix(".pending.mp4")
    if target.exists() and not refresh:
        receipt = validate_video(target, inputs.pts, Fraction(*inputs.tb))
        emit("video_already_verified", path=str(target), **receipt)
        return
    if pending.exists():
        raise FileExistsError(f"An incomplete export is preserved at {pending}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg")
    duration = len(inputs.pts) * (inputs.times[1] - inputs.times[0])
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
           "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{size[0]}x{size[1]}",
           "-framerate", "50", "-i", "pipe:0", "-ss", f"{inputs.times[0]:.6f}", "-i", str(inputs.media),
           "-f", "ffmetadata", "-i", str(chapters(inputs, output, target.stem)), "-map", "0:v:0",
           "-map", "1:a:0?", "-map_metadata:g", "-1", "-map_chapters", "2",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0",
           "-fps_mode", "passthrough", "-video_track_timescale", str(inputs.tb[1]),
           "-af", f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(pending)]
    with (output / (target.stem + "_encode.log")).open("ab") as error:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as encoder:
            try:
                count = 0
                for count, frame in enumerate(frames, start=1):
                    if frame.shape != (size[1], size[0], 3):
                        raise ValueError(f"Wrong video output size: {frame.shape}")
                    encoder.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                    if count % 1000 == 0:
                        emit("encoded", video=filename, exposures=count, pts=int(inputs.pts[count - 1]))
                encoder.stdin.close()
                if encoder.wait() or count != len(inputs.pts):
                    raise RuntimeError(f"Failed video encoding at {count} exposures")
            finally:
                if encoder.poll() is None:
                    encoder.terminate()
    receipt = validate_video(pending, inputs.pts, Fraction(*inputs.tb))
    pending.replace(target)
    emit("video_verified", path=str(target), start_pts=int(inputs.pts[0]), end_pts=int(inputs.pts[-1]), **receipt)


def original_review(inputs, output, data, refresh):
    frames = (image_frame(inputs, frame, data, i) for i, (_, frame) in enumerate(inputs.frames(0, len(inputs.pts))))
    render(inputs, output, "image_evidence_review.mp4", frames, (2400, 1080), refresh=refresh)


def draw_pose(frame, model, uv, origin, rotation, centre, joints, *, attached):
    if attached:
        for a, b in SKEL:
            lines(frame, joints[[a, b]], (130, 208, 250), thick=2)
    for f, colour in enumerate(FOOT_COLOUR):
        if not np.isfinite(uv[f]).all():
            text(frame, "Native row domain unresolved", (25, 94 + 30*f), .60, colour)
            continue
        outline = model.source_outline(uv[f, :model.point_count])
        lines(frame, outline, colour, close=True, thick=2)
        view = (centre - origin[f]) @ rotation[f]
        side = int(view[1] >= 0)
        runner = uv[f, model.point_count:].reshape(*model.runner.shape[1:-1], 2)
        lines(frame, runner[0, side], STEEL_COLOUR[f], thick=2)
        lines(frame, runner[1, side], STEEL_COLOUR[f])
        if not attached and outline is not None:
            # Crosses are parent ankle locations. The independent shoe proposal
            # is not rendered as part of the body mesh.
            ankle = tuple(np.rint(joints[15 + f]).astype(int))
            cv2.drawMarker(frame, ankle, colour, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)


def comparison_frames(inputs, model, native, new, old_p, old_r, image, fused, feasibility):
    n = len(inputs.pts)
    new_p, new_r = new["origin_world_m"], new["rotation_world"]
    old_curve = PoseCurve(inputs.times, old_p, old_r)
    new_curve = PoseCurve(inputs.times, new_p, new_r)
    reader = iter(inputs.frames(0, n))
    pivot_curve = PoseCurve(inputs.times, feasibility["pivot_trial_origin_m"], new_r) if feasibility is not None and fused is None else None
    for start in range(0, n, 128):
        index = np.arange(start, min(start + 128, n))
        old_uv = project_available(native, old_curve, model.points, index)
        new_uv = project_available(native, new_curve, model.points, index)
        pivot_uv = None if pivot_curve is None else project_available(native, pivot_curve, model.points, index)
        for j, i in enumerate(index):
            _, src = next(reader)
            old = src.copy()
            result = src.copy()
            draw_pose(old, model, old_uv[j], old_p[i], old_r[i], native.centres[i], inputs.old_coco[i], attached=True)
            if fused is None:
                draw_pose(result, model, new_uv[j], new_p[i], new_r[i], native.centres[i],
                          inputs.old_coco[i], attached=False)
                title = "NEW | visual SE(3), detached diagnostic"
                if pivot_uv is not None:
                    for a, b in ((11, 13), (12, 14), (13, 15), (14, 16)):
                        lines(result, inputs.old_coco[i, [a, b]], (145, 170, 190))
                    for f in range(2):
                        if np.isfinite(pivot_uv[j, f]).all():
                            outline = model.source_outline(pivot_uv[j, f, :model.point_count])
                            lines(result, outline, (60, 60, 245), close=True, thick=2)
                    text(result, "Red: same-body ankle-pivot probe (unaccepted)", (25, 121), .70)
            else:
                draw_pose(result, model, new_uv[j], new_p[i], new_r[i], native.centres[i], fused["COCO_uv"][i], attached=True)
                title = "NEW | shared-body candidate"
            panels = []
            for a, (frame, label) in enumerate(((src, "SOURCE | unmodified frame"),
                                               (old, "OLD | saved attached body and skates"),
                                               (result, title))):
                panel = np.full((880, 1280, 3), 23, np.uint8)
                panel[:720] = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
                text(panel, label, (18, 29), .78)
                text(panel, stamp(inputs, i), (18, 58), .65)
                for f in range(2):
                    roi = image["boot"][i, f]
                    if a == 1 and not np.isfinite(roi).all():
                        roi = inputs.old_gear[i, f, :14]
                    if a == 2:
                        roi = np.concatenate((roi, new_uv[j, f, :model.point_count]), axis=0)
                    panel[720:, 640 * f:640 * (f + 1)] = zoom(frame, roi, 640, 160)
                    if a == 2:
                        mode = colour_status(new["observation_status"][i, f])
                        extra = "" if feasibility is None else f' | ankle gap {feasibility["ankle_pivot_gap_m"][i, f]:.2f} m'
                        text(panel, ("L" if f == 0 else "R") + " " + mode + extra, (640*f + 12, 866), .51, FOOT_COLOUR[f])
                panels.append(panel)
            yield np.hstack(panels)


def main(output, compare, refresh):
    cv2.setNumThreads(1)
    inputs = Inputs()
    with np.load(output / "image_evidence.npz", allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    np.testing.assert_array_equal(data["source_pts"], inputs.pts)
    if not compare:
        original_review(inputs, output, data, refresh)
        return
    shape, old_p, old_r = geometry(inputs)
    model = SkateModel(shape)
    native = Native(inputs)
    with np.load(output / "visual_shoes.npz", allow_pickle=False) as f:
        new = {key: f[key] for key in f.files}
    np.testing.assert_array_equal(new["source_pts"], inputs.pts)
    fused, feasibility = None, None
    path = output / "fusion_check.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as p:
            np.testing.assert_array_equal(p["source_pts"], inputs.pts)
            feasibility = {key: p[key] for key in ("ankle_pivot_gap_m", "pivot_trial_origin_m")}
    if (output / "body_shoes_fused.npz").is_file():
        with np.load(output / "body_shoes_fused.npz", allow_pickle=False) as f:
            fused = {key: f[key] for key in ("source_pts", "COCO_uv", "skate_origin_world_mid", "skate_R_world_mid")}
        np.testing.assert_array_equal(fused["source_pts"], inputs.pts)
        new["origin_world_m"], new["rotation_world"] = fused["skate_origin_world_mid"], fused["skate_R_world_mid"]
    frames = comparison_frames(inputs, model, native, new, old_p, old_r, data, fused, feasibility)
    render(inputs, output, "source_old_new_review.mp4", frames, (3840, 880), refresh=refresh)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    main(args.output, args.compare, args.refresh)
