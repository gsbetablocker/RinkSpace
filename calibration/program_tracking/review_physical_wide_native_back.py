"""Read-only native-scale QA views of changed physical-wide proposals.

This changes only derived visualizations and their snapshot manifest. It never
generates, fits, or edits observations. The stored affine camera-plane strip
uses one native pixel per station in both directions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .extract_first_shot_ice import DATA

RD = DATA / "repair_20261014"
OUT = RD / "wide_line_diagnostics/physical_native_strips_back"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_changed() -> list[int]:
    manifest_path = OUT / "manifest.json"
    manifest = read(manifest_path)
    changed = []
    for record in manifest["records"]:
        i = record["index"]
        proposal = RD / f"physical_wide_proposals/{i:06d}.json"
        if sha(proposal) == record["candidate_sha256"]:
            continue
        packet = read(proposal)
        raw = cv2.imread(packet["source"]["image_path"])
        transforms = {row["curve_id"]: row for row in record["transforms_for_visualization_only"]}
        cells, new_transforms = [], []
        for feature in packet["features"]:
            curve_id = feature["world_curve"]["id"] if isinstance(feature["world_curve"], dict) else feature["world_curve"]
            if curve_id not in transforms:
                curve_id = feature["id"]
            transform = transforms[curve_id]
            origin = np.asarray(transform["origin_native_uv"], dtype=float)
            tangent = np.asarray(transform["unit_tangent"], dtype=float)
            normal = np.asarray(transform["unit_normal"], dtype=float)
            rows = feature["source_line_fit"]["direct_readouts"]
            sides = np.vstack([row["source_evidence"]["actual_source_sides_uv"] for row in rows])
            mids = np.array([row["uv"] for row in rows])
            reps = np.vstack([segment["uv"] for segment in feature["segments"]])
            support = np.vstack((sides, mids, reps))
            station = (support - origin) @ tangent
            offset = (support - origin) @ normal
            lo = min(transform["station_range_px"][0], int(np.floor(station.min())) - 25)
            hi = max(transform["station_range_px"][1], int(np.ceil(station.max())) + 25)
            nlo = min(transform["normal_range_px"][0], int(np.floor(offset.min())) - 25)
            nhi = max(transform["normal_range_px"][1], int(np.ceil(offset.max())) + 25)
            xx, yy = np.meshgrid(np.arange(lo, hi + 1), np.arange(nlo, nhi + 1))
            grid = origin + xx[..., None] * tangent + yy[..., None] * normal
            strip = cv2.remap(raw, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            marked = strip.copy()

            def xy(uv):
                delta = np.asarray(uv, dtype=float) - origin
                return np.rint(np.column_stack((delta @ tangent - lo, delta @ normal - nlo))).astype(int)

            for point in xy(sides):
                cv2.circle(marked, tuple(point), 1, (0, 220, 255), -1)
            for point in xy(mids):
                cv2.circle(marked, tuple(point), 1, (0, 0, 255), -1)
            for segment in feature["segments"]:
                points = xy(segment["uv"])
                cv2.polylines(marked, [points], False, (255, 0, 255), 1, cv2.LINE_AA)
                for point in points:
                    cv2.circle(marked, tuple(point), 2, (255, 0, 255), 1, cv2.LINE_AA)
            title = np.zeros((24, strip.shape[1], 3), np.uint8)
            cv2.putText(title, f"{i} {curve_id} native 1px affine: RAW / SIDES yellow; MID red; REP purple", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
            cells.append(np.vstack((title, strip, marked)))
            new_transforms.append(dict(transform, station_range_px=[int(lo), int(hi)], normal_range_px=[int(nlo), int(nhi)], all_physical_sides_included_in_view=True))
        width = max(cell.shape[1] for cell in cells)
        height = sum(cell.shape[0] for cell in cells)
        canvas = np.zeros((height, width, 3), np.uint8)
        y = 0
        for cell in cells:
            canvas[y:y + cell.shape[0], :cell.shape[1]] = cell
            y += cell.shape[0]
        dest = OUT / f"{i:06d}.png"
        cv2.imwrite(str(dest), canvas)
        record.update(candidate_sha256=sha(proposal), source_png_sha256=sha(Path(packet["source"]["image_path"])), strip_path=str(dest), strip_sha256=sha(dest), transforms_for_visualization_only=new_transforms)
        changed.append(i)
    for page in manifest["pages"]:
        if not set(page["indices"]) & set(changed):
            continue
        pics = [cv2.imread(str(OUT / f"{i:06d}.png")) for i in page["indices"]]
        width = max(pic.shape[1] for pic in pics)
        height = sum(pic.shape[0] for pic in pics)
        canvas = np.zeros((height, width, 3), np.uint8)
        y = 0
        for pic in pics:
            canvas[y:y + pic.shape[0], :pic.shape[1]] = pic
            y += pic.shape[0]
        dest = Path(page["path"])
        cv2.imwrite(str(dest), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])
        page["sha256"] = sha(dest)
    manifest["last_rerender_changed_indices"] = changed
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(changed=changed)))
    return changed


if __name__ == "__main__":
    render_changed()
