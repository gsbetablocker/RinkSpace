"""Extract source-only S09 orientation contact sheets on the native PTS grid."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rinkspace.path_policy import guard_project_write
from rinkspace.review.render import source_frames
from rinkspace.storage import AssetStore
from rinkspace.vision_native import file_ref


PROJECT = Path(__file__).resolve().parents[1]
PROGRAMME = PROJECT / "events/ankara_2026/programmes/okada_mei_sp"
RUN = Path("runs/body_s01_20261029/structure_camera_1108")
EPISODES = (
    ("LSp", "156.50", "165.30", ("165.14", "165.30")),
    ("CCoSp", "172.08", "184.52", ("184.52", "184.62")),
)
TILE = (220, 380)
COLUMNS, ROWS = 4, 5
GAP, HEADER, CAPTION = 12, 48, 30
STRIDE_SECONDS = Fraction("0.12")


def exact_pts(seconds, time_base, step):
    value = Fraction(seconds) / time_base
    if value.denominator != 1 or value.numerator % step:
        raise ValueError(f"{seconds}s is outside the registered native PTS grid")
    return value.numerator


def crop_axis(lower, upper, span, limit):
    extent = min(limit, math.ceil(span) + 1)
    first = min(limit - extent, max(0, math.floor((lower + upper - extent) / 2)))
    return first, first + extent


def source_crop(uv, image_size):
    if uv.shape != (17, 2) or not np.isfinite(uv).all() or np.any((uv < 0) | (uv >= image_size)):
        raise ValueError("A source exposure must carry all 17 original in-image UV points")
    lo, hi = uv.min(0), uv.max(0)
    if np.any(hi - lo <= 0):
        raise ValueError("Degenerate crop-navigation UV bounds")
    width, height = image_size
    body_height = hi[1] - lo[1]
    side_pad = max(24., .18 * body_height)
    head_pad = max(24., .13 * body_height)
    shoe_pad = max(30., .20 * body_height)
    left, right = lo[0] - side_pad, hi[0] + side_pad
    top, bottom = lo[1] - head_pad, hi[1] + shoe_pad
    aspect = TILE[0] / TILE[1]
    target_height = max(bottom - top, (right - left) / aspect)
    x0, x1 = crop_axis(left, right, target_height * aspect, width)
    y0, y1 = crop_axis(top, bottom, target_height, height)
    if not (x0 <= lo[0] <= hi[0] < x1 and y0 <= lo[1] <= hi[1] < y1):
        raise ValueError("Expanded crop omitted a saved source UV point")
    return [x0, y0, x1, y1], [*map(float, lo), *map(float, hi)]


def exposure_plan(label, start, end, extra, clock, native_step, shot, points, uv, image_size):
    a, b = (exact_pts(t, clock, native_step) for t in (start, end))
    special = {exact_pts(t, clock, native_step) for t in extra}
    requested = sorted(set(range(a, b + 1, exact_pts(STRIDE_SECONDS, clock, native_step))) | special)
    if requested[0] < shot["first_pts"] or requested[-1] > shot["last_pts"]:
        raise ValueError(f"{label} exposures extend beyond the registered S09 processing range")
    indices = np.searchsorted(points, requested)
    if np.any(indices >= len(points)) or not np.array_equal(points[indices], requested):
        raise ValueError(f"Saved crop observations do not cover all {label} source PTS")
    frames = []
    for stamp, index in zip(requested, indices, strict=True):
        box, navigation = source_crop(uv[index], image_size)
        frames.append(dict(source_pts=stamp, source_seconds=float(stamp * clock), shot="s09",
                           endpoint_review=stamp in special, crop_xyxy_exclusive=box,
                           navigation_uv_bbox_xyxy=navigation))
    return frames


def source_sheet(output, label, page, total, entries, source_images):
    width, height = TILE
    canvas = Image.new("RGB", (COLUMNS * width + (COLUMNS + 1) * GAP,
                               HEADER + ROWS * (height + CAPTION + GAP)), "#14212b")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(22)
    caption = ImageFont.load_default(15)
    draw.text((GAP, 10), f"S09  {label}  |  Source crop sequence   {page}/{total}", fill="white", font=title)
    for index, (record, image) in enumerate(zip(entries, source_images, strict=True)):
        row, col = divmod(index, COLUMNS)
        x = GAP + col * (width + GAP)
        y = HEADER + row * (height + CAPTION + GAP)
        ox, oy = x + (width - image.width) // 2, y + (height - image.height) // 2
        canvas.paste(image, (ox, oy))
        text = f"{record['source_seconds']:.2f}s  |  PTS {record['source_pts']}"
        fill = "#ffcf81" if record["endpoint_review"] else "#d9e7f2"
        draw.text((x + 1, y + height + 5), text, fill=fill, font=caption)
        if record["endpoint_review"]:
            draw.text((x + width - 27, y + height + 5), "END", fill=fill, font=caption)
        record.update(page=page, row=row + 1, column=col + 1,
                      display_crop_xyxy_exclusive=[ox, oy, ox + image.width, oy + image.height])
    path = output / f"{label}_{page:02d}.png"
    canvas.save(path)
    return str(path)


def render_episode(output, label, frames, video, source, native_step):
    stamps = {r["source_pts"] for r in frames}
    clock = range(frames[0]["source_pts"], frames[-1]["source_pts"] + 1, native_step)
    capacity = COLUMNS * ROWS
    total = math.ceil(len(frames) / capacity)
    pages, images = [], []
    for stamp, image in source_frames(video, source, clock):
        if stamp not in stamps:
            continue
        page = len(pages) + 1
        first = (page - 1) * capacity
        record = frames[first + len(images)]
        if record["source_pts"] != stamp:
            raise ValueError("Decoded crop order differs from the requested source PTS")
        x0, y0, x1, y1 = record["crop_xyxy_exclusive"]
        crop = Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1, ::-1]))
        crop.thumbnail(TILE, Image.Resampling.LANCZOS)
        images.append(crop)
        if len(images) == capacity or stamp == frames[-1]["source_pts"]:
            selected = frames[first:first + len(images)]
            page_path = source_sheet(output, label, page, total, selected, images)
            pages.append(dict(page=page, path=page_path, exposures=len(selected),
                              source_pts=[r["source_pts"] for r in selected]))
            images.clear()
    if len(pages) != total or sum(r["exposures"] for r in pages) != len(frames):
        raise ValueError("Contact sheets did not capture the complete requested exposure list")
    return dict(episode=label, sample_interval_seconds=float(STRIDE_SECONDS),
                selected_exposures=len(frames), pages=pages, frames=frames)


def main():
    store = AssetStore(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=store.data / RUN / "prepared_structure3")
    parser.add_argument("--out", type=Path, default=store.data / RUN / "out/s09_orientation_source")
    args = parser.parse_args()
    out = guard_project_write(args.out)
    if not store.configured or not out.is_relative_to(store.data.resolve()) or out.exists():
        raise ValueError("Choose a new output directory inside the configured Data root")
    source_path = PROGRAMME / "primary_source_selection.json"
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))["primary_source"]
    video = store.source(source["path"]).resolve(strict=True)
    original = file_ref(video)
    if original["sha256"] != source["sha256"]:
        raise ValueError("Local source video does not match the registered original SHA256")
    time_base = Fraction(source["time_base"])
    native_step = source["native_pts_step"]
    if native_step * time_base != Fraction(1, 50):
        raise ValueError("This contact sheet requires the registered 50 Hz exposure grid")
    binding = dict(source, time_base_num=time_base.numerator, time_base_den=time_base.denominator)
    scope = json.loads((PROGRAMME / "rebuild_scope.json").read_text(encoding="utf-8-sig"))
    shot = next(s["processing"] for s in scope["shots"] if s["id"] == "s09")
    observation_path = (args.prepared / "observations.npz").resolve(strict=True)
    with np.load(observation_path, allow_pickle=False) as observations:
        pts, uv = observations["pts"], observations["uv"]
    if pts.ndim != 1 or pts.dtype.kind not in "iu" or np.any(np.diff(pts) <= 0) or uv.shape != (len(pts), 17, 2):
        raise ValueError("Invalid saved crop-navigation clock/UV array")
    source_size = (source["width"], source["height"])
    plans = {label: exposure_plan(label, start, end, extra, time_base, native_step, shot,
                                  pts, uv, source_size)
             for label, start, end, extra in EPISODES}
    out.mkdir(parents=True, exist_ok=False)
    episodes = []
    for label, frames in plans.items():
        episode = render_episode(out, label, frames, video, binding, native_step)
        episodes.append(episode)
        print(f"{label} {len(frames)} exposures, {len(episode['pages'])} pages", flush=True)
    report = dict(
        schema="rinkspace.s09-source-orientation-contact-sheets.v1", source=original,
        source_selection=file_ref(source_path), source_size=list(source_size),
        time_base=[time_base.numerator, time_base.denominator], native_pts_step=native_step,
        crop_navigation_observations=file_ref(observation_path), grid=[COLUMNS, ROWS], tile_size=list(TILE),
        crop_policy="Raw source UV envelope, padded for head, arms and shoe margins. Crop is resized for the sheet, with all labels outside the image.",
        review_scope="Sampled orientation progression. Gold END labels identify requested extra endpoint checks.",
        source_pictures_with_model_overlays=False, episodes=episodes,
    )
    target = out / "orientation_source.json"
    with target.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write(chr(10))
    print(target, flush=True)


if __name__ == "__main__":
    main()
