"""Make every-frame contact pages from a completed same-model review video.

The native source PTS comes from the render ledger. Every encoded exposure is
decoded once, in order, without seeking, frame selection or a second render.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4194304), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def verify(reference):
    path = Path(reference["path"]).resolve()
    if sha(path) != reference["sha256"]:
        raise ValueError("Current review file changed: " + str(path))
    return path


def run(review_manifest, output):
    manifest_path = Path(review_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("status") != "complete_NOT_ACCEPTED":
        raise ValueError("Contact pages require the completed working review")
    video = verify(manifest["video"])
    ledger_path = verify(manifest["source_coverage"])
    verify(manifest["model"])
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf8").splitlines()]
    total = int(manifest["frames"])
    if len(ledger) != total or any(row["index"] != i+1 or row["model_sha256"] != manifest["model"]["sha256"] for i, row in enumerate(ledger)):
        raise ValueError("Frame ledger differs from the exact rendered model/order")
    pts = np.asarray([row["native_pts"] for row in ledger], dtype="<i8")
    if hashlib.sha256(pts.tobytes()).hexdigest() != manifest["source_PTS_array_sha256"]:
        raise ValueError("Source PTS array differs from completed review")
    out = Path(output).resolve()
    if "RinkSpace_Data" not in out.parts:
        raise ValueError("Contact diagnostics must be written in Data")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refuse to replace existing contact pages")
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = 5, 10
    tile_w, image_h, label_h, header_h = 320, 180, 20, 32
    page_h = header_h + rows*(image_h+label_h)
    page_w = cols*tile_w
    pages = []
    page = None
    tiles = []
    count = 0

    def finish():
        if not tiles:
            return
        number = len(pages)+1
        path = out / f"page_{number:02d}.jpg"
        if not cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(path)
        pages.append(dict(page=number, image=ref(path), first_frame=tiles[0]["frame_index"],
                          last_frame=tiles[-1]["frame_index"], first_source_PTS=tiles[0]["native_pts"],
                          last_source_PTS=tiles[-1]["native_pts"], tiles=list(tiles)))

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != Fraction(50, 1):
            raise ValueError("Expected the exact 50 fps review video")
        for frame in container.decode(stream):
            if count >= total:
                raise ValueError("Encoded video has extra exposures")
            if Fraction(frame.pts)*frame.time_base != Fraction(count, 50):
                raise ValueError("Encoded exposure clock differs from its exact index/50")
            cell = count % (cols*rows)
            if cell == 0:
                finish()
                tiles = []
                page = np.full((page_h, page_w, 3), (20, 25, 30), np.uint8)
                label = f"S01 ALL FRAMES | page {len(pages)+1}/{(total+49)//50} | model {manifest['model']['sha256'][:12]} | working review, NOT ACCEPTED"
                cv2.putText(page, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .47, (245, 245, 245), 1, cv2.LINE_AA)
            row = ledger[count]
            x = (cell % cols)*tile_w
            y = header_h + (cell // cols)*(image_h+label_h)
            image = cv2.resize(frame.to_ndarray(format="bgr24"), (tile_w, image_h), interpolation=cv2.INTER_AREA)
            page[y:y+image_h, x:x+tile_w] = image
            text = f"{count+1:04d}/{total}  src {row['native_pts']/12800.:.3f}s  PTS {row['native_pts']}"
            cv2.putText(page, text, (x+4, y+image_h+14), cv2.FONT_HERSHEY_SIMPLEX, .35, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(dict(frame_index=count+1, native_pts=int(row["native_pts"]),
                              source_seconds=row["native_pts"]/12800., encoded_pts=int(frame.pts),
                              encoded_time_base=str(frame.time_base), cell=cell, row=cell//cols, column=cell%cols,
                              image_rect_xywh=[x, y, tile_w, image_h]))
            count += 1
    finish()
    if count != total or sum(len(item["tiles"]) for item in pages) != total:
        raise ValueError("Contact coverage is not exactly every encoded exposure")
    report = dict(schema="rinkfit.same_model_every_frame_contact_pages.v1", review_manifest=ref(manifest_path),
                  video=manifest["video"], model=manifest["model"], source_coverage=manifest["source_coverage"],
                  frame_count=count, pages=pages, page_count=len(pages), columns=cols, rows=rows,
                  tile_image_wh=[tile_w, image_h], tile_label_height=label_h, page_image_wh=[page_w, page_h],
                  encoded_frames_decoded_exactly_once=True, frame_stride=1, no_frame_omitted=True,
                  another_video_rendered=False, whole_shot_acceptance=False,
                  purpose="Continuous overview of every frame; inspect original-size frames for precise advertisement boundaries")
    target = out / "manifest.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(manifest=str(target), frames=count, pages=len(pages), page_image_wh=[page_w, page_h]), ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.review_manifest, args.output)
