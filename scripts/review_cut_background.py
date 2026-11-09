"""Review original static UV residuals at fixed stations on eight camera cuts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rinkspace.core.native import NativeCore
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.storage import AssetStore

from scripts.review_structure_cuts import (
    PROGRAMME,
    PROJECT,
    add_cases,
    load_camera,
    named_path,
    read_json,
    sha256,
)


DIAGNOSTIC = "Fixed original curve station diagnostic"
RUN = Path("runs/body_s01_20261029/structure_camera_1108")
# The verified source contact sheets from review_structure_cuts use this layout.
TILE_WIDTH, TILE_HEIGHT, GAP, TITLE, LABEL = 960, 540, 16, 60, 34
ARROW_SCALE = 6
DISPLAY_GRID = np.array([8, 5])


def material_groups(paths, scene):
    entities = {path["id"]: path["entity_id"] for path in scene["paths"]}
    return np.array([path.split("::", 1)[0] if "::" in path else entities[path] for path in paths])


def project_rows(native, camera, camera_definition, paths, stations, pts, uv, shot_sections):
    """Run the frozen native implicit projection on independently timed shot sections."""
    timeline, controls, rigs = camera
    shots = {s["id"]: s for s in timeline.shots}
    times, camera_ids = np.empty(len(pts)), np.empty(len(pts), dtype=np.int32)
    for shot_id, rows in shot_sections:
        shot = shots[shot_id]
        if np.any((pts[rows] < shot["first_pts"]) | (pts[rows] > shot["last_pts"])):
            raise ValueError(f"Source rows outside independent shot {shot['id']}")
        times[rows] = timeline.exposure_time(shot, pts[rows])
        camera_ids[rows] = shot["camera_index"]
        halo = abs(rigs[shot["camera_index"], 37]) / 2
        edge_times = np.array([times[rows].min() - halo, times[rows].max() + halo])
        controls_used, _ = cubic_samples(timeline.count, timeline.first, timeline.dt, edge_times)
        if controls_used.min() < shot["control_begin"] or controls_used.max() >= shot["control_stop"]:
            raise ValueError(f"Readout spans multiple motion blocks for {shot['id']}")
    definition = dict(camera_definition, first_control_time=timeline.first, control_dt=timeline.dt)
    with native.resident(
        camera=definition, controls=controls, rigs=rigs, camera_ids=camera_ids,
        path_ids=paths, stations=stations, pts=pts, observed=uv, motion_times=times,
        scene_directions=np.zeros((len(native.parameter_state.values), 1)),
    ) as problem:
        problem.configure_projection_time_support(half_seconds=timeline.max_readout() / 2)
        projected = problem.evaluate("cuda")
    pred = projected["values"][:, 3:5].copy()
    statuses = projected["status"].copy()
    if not np.isfinite(pred[statuses == 0]).all():
        raise ValueError("Native projector returned a non-finite UV with success status")
    return pred, statuses


def projection(native, camera, camera_definition, paths, stations, pts, uv, cut, split):
    sections = ((cut["left"], slice(None, split)), (cut["right"], slice(split, None)))
    predicted, status = project_rows(native, camera, camera_definition, paths, stations, pts, uv, sections)
    return predicted - uv, status


def stats(residual, status):
    valid = status == 0
    selected = residual[valid]
    errors = np.linalg.norm(selected, axis=1)
    return dict(
        rows=len(status), valid=int(valid.sum()), invalid=int((~valid).sum()),
        invalid_native_status_counts=dict(Counter(map(int, status[~valid]))),
        rms_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        p95_px=float(np.quantile(errors, .95)) if len(errors) else None,
        mean_signed_uv_px=selected.mean(axis=0).tolist() if len(errors) else None,
    )


def grouped_stats(residual, status, labels):
    return {label: stats(residual[labels == label], status[labels == label])
            for label in np.unique(labels)}


def display_rows(uv, status, source_size):
    within = np.all((uv >= 0) & (uv < source_size), axis=1)
    rows = np.flatnonzero((status == 0) & within)
    cells = uv[rows] / source_size * DISPLAY_GRID
    grid = np.floor(cells).astype(int)
    distance = np.linalg.norm(cells - grid - .5, axis=1)
    order = np.argsort(distance, kind="stable")
    _, indices = np.unique(grid[order], axis=0, return_index=True)
    return rows[order[indices]]


def draw_arrow(draw, start, offset):
    end = start + offset
    start = tuple(start)
    length = np.linalg.norm(offset)
    if length >= 1:
        draw.line([start, tuple(end)], fill="#19120d", width=5)
        draw.line([start, tuple(end)], fill="#ffad55", width=2)
        unit = offset / length
        tangent = np.array([-unit[1], unit[0]])
        head = end - min(5, length * .6) * unit
        spread = min(3, length * .4) * tangent
        draw.polygon([tuple(end), tuple(head + spread), tuple(head - spread)], fill="#ffad55")
    x, y = start
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#65e5ff", outline="#062731")


def overlay(source_sheet, output, name, source_size, exposures, pts, uv, residual, status):
    expected = (2 * TILE_WIDTH + 3 * GAP, TITLE + 5 * (TILE_HEIGHT + LABEL + GAP))
    with Image.open(source_sheet) as source:
        if source.size != expected:
            raise ValueError(f"Unrecognized source contact sheet size: {source_sheet}")
        canvas = source.convert("RGB")
    font = ImageFont.load_default(16)
    caption = ImageDraw.Draw(canvas)
    caption.text((GAP, 37),
                 f"{name}  |  {DIAGNOSTIC}  |  Source dot to predicted UV, arrows x{ARROW_SCALE}",
                 font=font, fill="#f4d2aa")
    scale = np.array([TILE_WIDTH, TILE_HEIGHT]) / source_size
    rows_per_frame = {}
    for index, exposure in enumerate(exposures):
        side, row = divmod(index, 5)
        x = GAP + side * (TILE_WIDTH + GAP)
        y = TITLE + row * (TILE_HEIGHT + LABEL + GAP)
        selected = pts == exposure
        source_uv = uv[selected]
        frame_residual = residual[selected]
        chosen = display_rows(source_uv, status[selected], source_size)
        frame = canvas.crop((x, y, x + TILE_WIDTH, y + TILE_HEIGHT))
        draw = ImageDraw.Draw(frame)
        for i in chosen:
            draw_arrow(draw, source_uv[i] * scale, frame_residual[i] * scale * ARROW_SCALE)
        canvas.paste(frame, (x, y))
        caption.text((x + TILE_WIDTH - 208, y + TILE_HEIGHT + 8),
                     f"{len(chosen)} shown / {selected.sum()} rows", font=font, fill="#cdd5df")
        rows_per_frame[str(exposure)] = len(chosen)
    canvas.save(output)
    return rows_per_frame


def main():
    store = AssetStore(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-review", type=Path, default=store.data / RUN / "cut_review/cut_review.json")
    parser.add_argument("--out", type=Path, default=store.data / RUN / "cut_background_baseline")
    parser.add_argument("--camera", type=named_path, action="append", default=[], metavar="NAME=NPZ")
    args = parser.parse_args()

    out = guard_project_write(args.out)
    if not store.configured or not out.is_relative_to(store.data.resolve()):
        raise ValueError("Static background outputs belong in the registered Data root")
    if out.exists():
        raise FileExistsError(f"Preserve existing review output: {out}")
    review_path = args.cut_review.resolve(strict=True)
    review = read_json(review_path)
    primary = read_json(PROGRAMME / "primary_source_selection.json")["primary_source"]
    source_size = np.array([primary["width"], primary["height"]])
    if (review.get("schema") != "rinkspace.structure_cut_review.v1" or
            review["original_source_sha256"] != primary["sha256"] or
            not np.array_equal(review["source_size"], source_size)):
        raise ValueError("Original source review does not match the programme source binding")

    record = read_json(PROGRAMME / "fit_v11/camera_solver_refined_1102.json")
    prepared = store.source(record["prepared"])
    manifest = read_json(prepared / "joint_manifest.json")
    points_path = prepared / "point_observations.npz"
    mapping_path = prepared / "mapping.json"
    scene_path = store.source(manifest["scene_path"])
    state_path = store.source(record["state"]["path"])
    compact_path = store.source(record["state"]["compact_path"])
    core = NativeCore(record["native_library"]["path"])
    for path, expected in (
        (points_path, record["point_input_sha256"]), (scene_path, record["scene_sha256"]),
        (state_path, record["state"]["sha256"]), (compact_path, record["state"]["compact_sha256"]),
        (core.library_path, record["native_library"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input differs from its programme binding: {path}")
    paths = np.asarray(read_json(mapping_path)["path_ids"])
    path_digest = hashlib.sha256(json.dumps(paths.tolist(), separators=(",", ":")).encode()).hexdigest()
    with np.load(points_path, allow_pickle=False) as points, np.load(state_path, allow_pickle=False) as state:
        deps = json.loads(state["joint_numeric_dependencies_json"].item())
        if (deps["paths_sha256"] != path_digest or deps["shots"] != manifest["shots"] or
                state["point_input_sha256"].item() != record["point_input_sha256"] or
                not np.array_equal(points["source_rows"], state["source_rows"])):
            raise ValueError("Source row identities and frozen fitted stations do not align")
        pts, uv, path_codes = points["pts"], points["uv"], points["path_indices"]
        stations, scene_values = state["stations"], state["scene_values"]
        baseline_controls, baseline_rigs = state["controls"], state["rig"]
    scene = read_json(scene_path)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    if any(model.initial_state.layout[key] != deps[bound] for key, bound in
           (("layout_revision", "layout_revision"), ("scene_content_revision", "scene_revision"))):
        raise ValueError("Scene parameter state does not match the frozen layout")
    native = NativeSceneGPU(scene_path, core, parameter_state=ParameterState(model.initial_state.layout, scene_values))
    groups = material_groups(paths, native.scene)
    camera_def = read_json(store.source(manifest["cameras"][0]["camera_definition"]))
    camera_paths = add_cases(compact_path, args.camera)
    cameras = {name: load_camera(path, manifest["shots"]) for name, path in camera_paths.items()}
    if not (np.array_equal(cameras["baseline"][1], baseline_controls) and
            np.array_equal(cameras["baseline"][2], baseline_rigs)):
        raise ValueError("Compact camera is not equal to the frozen full-state camera")
    for path in camera_paths.values():
        with np.load(path, allow_pickle=False) as state:
            candidate_deps = json.loads(state["joint_numeric_dependencies_json"].item())
            if (not np.array_equal(state["scene_values"], scene_values) or
                    state["point_input_sha256"].item() != record["point_input_sha256"] or
                    candidate_deps["cameras"] != deps["cameras"] or
                    [s["camera_index"] for s in candidate_deps["shots"]] !=
                    [s["camera_index"] for s in deps["shots"]]):
                raise ValueError(f"Candidate must bind the same scene, input rows and physical installations: {path}")

    shots = manifest["shots"]
    if len(review["cuts"]) != len(shots) - 1 or len(shots) != 9:
        raise ValueError("Review must contain all eight cuts")
    total = {name: [] for name in cameras}
    cuts = []
    out.mkdir(parents=True, exist_ok=False)
    for i, cut in enumerate(review["cuts"]):
        left, right = shots[i:i + 2]
        step = primary["native_pts_step"]
        before = list(range(left["last_pts"] - 4 * step, left["last_pts"] + step, step))
        after = list(range(right["first_pts"], right["first_pts"] + 5 * step, step))
        if (cut["left"], cut["right"], cut["left_pts"], cut["right_pts"]) != (
            left["id"], right["id"], before, after
        ):
            raise ValueError("Source contact sheet does not contain the expected exact cut exposures")
        split = np.count_nonzero(np.isin(pts[left["row_begin"]:left["row_stop"]], before))
        rows = np.flatnonzero(np.isin(pts, before + after))
        if (not split or split == len(rows) or not np.array_equal(np.unique(pts[rows]), before + after)):
            raise ValueError("Missing current static rows at a required cut exposure")
        source_sheet = Path(cut["source_sheet"]).resolve(strict=True)
        entry = dict(
            left=left["id"], right=right["id"], cut_pts=right["first_pts"],
            source_seconds=cut["source_seconds"], left_pts=before, right_pts=after,
            base_contact_sheet=str(source_sheet), base_contact_sheet_sha256=sha256(source_sheet),
            cameras={},
        )
        for case, (name, camera) in enumerate(cameras.items()):
            r, status = projection(native, camera, camera_def, paths[path_codes[rows]],
                                   stations[rows], pts[rows], uv[rows], cut, split)
            labels = groups[path_codes[rows]]
            total[name].append((r, status, labels))
            figure = out / f"case_{case:02}_cut_{i + 1:02}_{cut['source_seconds']:.2f}.png"
            display = overlay(source_sheet, figure, name, source_size, before + after,
                              pts[rows], uv[rows], r, status)
            details = dict(figure=str(figure), display_rows_by_pts=display, **stats(r, status))
            for side, section in (("left", slice(None, split)), ("right", slice(split, None))):
                details[side] = dict(
                    **stats(r[section], status[section]),
                    entities=grouped_stats(r[section], status[section], labels[section]),
                )
            details["exposures"] = {int(exposure): stats(r[pts[rows] == exposure], status[pts[rows] == exposure])
                                    for exposure in before + after}
            entry["cameras"][name] = details
        cuts.append(entry)
        print(f"{cut['source_seconds']:.2f}s, {len(rows)} source rows", flush=True)
    summary = {}
    for name, all_windows in total.items():
        r, status, labels = [np.concatenate(field) for field in zip(*all_windows, strict=True)]
        summary[name] = dict(path=str(camera_paths[name]), sha256=sha256(camera_paths[name]),
                             **stats(r, status), entities=grouped_stats(r, status, labels))
    report = dict(
        schema="rinkspace.fixed_station_cut_background.v1",
        diagnostic=DIAGNOSTIC, independently_measured_world_points=False, fit_executed=False,
        source_path=review["original_source_path"], source_sha256=primary["sha256"],
        source_time_base=review["time_base"], source_size=source_size.tolist(),
        input_rows=dict(path=str(points_path), sha256=record["point_input_sha256"]),
        material_paths=dict(path=str(mapping_path), path_id_list_sha256=path_digest),
        station_state=dict(path=str(state_path), sha256=record["state"]["sha256"]),
        scene=dict(path=str(scene_path), sha256=record["scene_sha256"]),
        source_cut_review=dict(path=str(review_path), sha256=sha256(review_path)),
        native_library=dict(path=str(core.library_path), sha256=record["native_library"]["sha256"]),
        residual_definition="predicted native UV minus original measured UV, in 1920x1080 source pixels",
        station_policy="Saved full-state station fixed for every input row, including candidate comparisons.",
        display_policy=f"One valid in-image row per {DISPLAY_GRID[0]}x{DISPLAY_GRID[1]} image grid cell; arrows x{ARROW_SCALE}. Statistics include all original rows and report invalid projections.",
        cameras=summary, cuts=cuts,
    )
    target = out / "cut_background.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + chr(10), encoding="utf8")
    print(str(target), flush=True)


if __name__ == "__main__":
    main()
