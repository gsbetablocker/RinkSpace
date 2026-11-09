"""Export saved programme camera evidence at native exposure and row times."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.storage import AssetStore
from scripts.review_cut_background import material_groups, project_rows, projection, stats
from scripts.review_structure_cuts import load_camera, read_json, sha256


PROGRAMME = Path("events/ankara_2026/programmes/okada_mei_sp")
ROW_FRACTIONS = np.linspace(0., 1., 5)
BACKGROUND_BATCH_ROWS = 32_768
CAMERA_FIELDS = {
    "motion_values": (4,),
    "R_world_to_camera": (3, 3),
    "C_scene_units": (3,),
    "K_px": (3, 3),
    "brown": (4,),
    "control_indices": (4,),
    "control_basis": (4,),
    "camera_status": (),
}


def identity(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen identity differs: {path}")
    return dict(path=str(path), sha256=digest)


def camera_samples(core, camera, shot_ids, times):
    """Sample each shot's own native controls at its explicit storage-clock times."""
    timeline, controls, rigs = camera
    motion_times = np.asarray(times)
    shots = np.broadcast_to(np.asarray(shot_ids).reshape((-1,) + (1,) * (motion_times.ndim - 1)),
                            motion_times.shape).ravel()
    flat = motion_times.ravel()
    samples = {key: np.full((len(flat), *shape), np.nan) for key, shape in CAMERA_FIELDS.items()}
    for key in ("control_indices", "camera_status"):
        samples[key] = np.full(samples[key].shape, -1, dtype=np.int32)
    for shot in timeline.shots:
        use = np.flatnonzero((shots == shot["id"]) & np.isfinite(flat))
        if not len(use):
            continue
        t = flat[use]
        indices, basis = cubic_samples(timeline.count, timeline.first, timeline.dt, t)
        if indices.min() < shot["control_begin"] or indices.max() >= shot["control_stop"]:
            raise ValueError(f"Camera samples cross the independent control block for {shot['id']}")
        state = core.evaluate_motion_camera(rigs[shot["camera_index"]], controls, timeline.first, timeline.dt, t)
        values, status = state["values"], state["status"]
        valid = status == 0
        if not np.isfinite(values[valid]).all():
            raise ValueError("Native camera returned a non-finite successful sample")
        reconstructed = np.einsum("ni,nij->nj", basis, controls[indices])
        np.testing.assert_allclose(reconstructed[valid], values[valid, :3], rtol=0., atol=1e-10)
        K = np.zeros((len(t), 3, 3))
        K[:, 0, 0], K[:, 1, 1] = state["fx"], state["fy"]
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = state["cx"], state["cy"], 1.
        K[~valid] = np.nan
        samples["motion_values"][use] = values[:, :4]
        samples["R_world_to_camera"][use] = state["rotation"]
        samples["C_scene_units"][use] = state["centre"]
        samples["K_px"][use] = K
        samples["brown"][use] = values[:, 20:24]
        samples["control_indices"][use] = indices
        samples["control_basis"][use] = basis
        samples["camera_status"][use] = status
    if np.any((samples["camera_status"] == -1) & np.isfinite(flat)):
        raise ValueError("A finite camera time has no registered shot")
    return {key: value.reshape((*motion_times.shape, *CAMERA_FIELDS[key])) for key, value in samples.items()}


def exposure_samples(dispatch, camera, pts, shot_ids, time_base, camera_sha, scales):
    timeline, controls, rigs = camera
    midpoint = np.asarray(dispatch.p["motion_times"])
    readout = dispatch.readout_times(np.arange(len(pts)))
    camera_ids = np.full(len(pts), -1, dtype=np.int32)
    for shot in timeline.shots:
        use = shot_ids == shot["id"]
        np.testing.assert_array_equal(midpoint[use], timeline.exposure_time(shot, pts[use]))
        camera_ids[use] = shot["camera_index"]
    np.testing.assert_array_equal(readout, rigs[camera_ids, 37])
    times = midpoint[:, None] + readout[:, None] * (ROW_FRACTIONS - .5)
    samples = camera_samples(dispatch.core, camera, shot_ids, times)
    np.testing.assert_array_equal(samples["camera_status"][:, 2], dispatch.midpoint["status"])
    np.testing.assert_allclose(samples["R_world_to_camera"][:, 2], dispatch.midpoint["rotation"],
                               rtol=0., atol=1e-12)
    np.testing.assert_allclose(samples["C_scene_units"][:, 2], dispatch.midpoint["centre"],
                               rtol=0., atol=1e-12)
    return dict(source_pts=pts, shot_id=shot_ids, camera_index=camera_ids, time_base=np.asarray(time_base),
                camera_state_sha256=np.array(camera_sha), readout_signed_s=readout,
                midpoint_motion_time_s=midpoint,
                node_image_row_fraction=ROW_FRACTIONS, node_motion_time_s=times,
                node_chronological_order=np.argsort(times, axis=1, kind="stable"),
                native_controls=controls, native_rigs=rigs,
                first_control_time_s=np.array(timeline.first), control_dt_s=np.array(timeline.dt), **scales, **samples)


def row_camera_evidence(native, camera, exposures, stamps, predicted, status, height):
    frame_ids = np.searchsorted(exposures["source_pts"], stamps)
    if np.any(frame_ids == len(exposures["source_pts"])):
        raise ValueError("Background row is outside the original source exposure clock")
    np.testing.assert_array_equal(exposures["source_pts"][frame_ids], stamps)
    shot_ids = exposures["shot_id"][frame_ids]
    readout = exposures["readout_signed_s"][frame_ids]
    midpoint = exposures["midpoint_motion_time_s"][frame_ids]
    # The native solver continues the same equation outside the raster.
    # Failed projections have no solved row and no corresponding instant.
    solved = status == 0
    motion_times = np.full(len(stamps), np.nan)
    motion_times[solved] = midpoint[solved] + readout[solved] * (predicted[solved, 1] / (height - 1) - .5)
    sampled = camera_samples(native.core, camera, shot_ids, motion_times)
    if np.any(sampled["camera_status"][solved]):
        raise ValueError("Native camera failed at a successful implicit projection's row time")
    return dict(source_pts=stamps, shot_id=shot_ids, camera_index=exposures["camera_index"][frame_ids],
                projection_status=status, readout_signed_s=readout, midpoint_motion_time_s=midpoint,
                row_motion_time_s=motion_times, **sampled)


def background_rows(native, camera, definition, row_data, source_data, exposures, cuts, height):
    timeline, _, _ = camera
    point_pts, uv, path_codes, source_rows, stations = row_data
    paths, groups = source_data
    saved, summaries = [], []
    for cut_index, (cut, left, right) in enumerate(zip(cuts, timeline.shots[:-1], timeline.shots[1:], strict=True)):
        before = exposures["source_pts"][exposures["shot_id"] == left["id"]][-5:]
        after = exposures["source_pts"][exposures["shot_id"] == right["id"]][:5]
        if (cut["left"], cut["right"], cut["left_pts"], cut["right_pts"], cut["cut_pts"]) != (
            left["id"], right["id"], before.tolist(), after.tolist(), int(after[0])
        ):
            raise ValueError("Cut review differs from the native source exposures")
        selected = []
        for shot, stamps in ((left, before), (right, after)):
            begin, stop = shot["row_begin"], shot["row_stop"]
            rows = np.flatnonzero(np.isin(point_pts[begin:stop], stamps)) + begin
            np.testing.assert_array_equal(np.unique(point_pts[rows]), stamps)
            selected.append(rows)
        rows = np.concatenate(selected)
        split = len(selected[0])
        stamps, observed = point_pts[rows], uv[rows]
        path_ids, group_ids = paths[path_codes[rows]], groups[path_codes[rows]]
        residual, status = projection(native, camera, definition, path_ids, stations[rows],
                                      stamps, observed, cut, split)
        predicted = observed + residual
        evidence = row_camera_evidence(native, camera, exposures, stamps, predicted, status, height)
        saved.append(dict(cut_index=np.full(len(rows), cut_index, dtype=np.int32),
                          cut_pts=np.full(len(rows), cut["cut_pts"], dtype=np.int64),
                          cut_side=np.repeat(["left", "right"], [split, len(rows) - split]),
                          source_pts=evidence.pop("source_pts"), shot_id=evidence.pop("shot_id"),
                          camera_index=evidence.pop("camera_index"), point_input_row=rows, source_row=source_rows[rows], path_index=path_codes[rows],
                          path_id=path_ids, material_group=group_ids, station=stations[rows],
                          observed_uv_px=observed, predicted_uv_px=predicted, residual_uv_px=residual,
                          **evidence))
        summaries.append(dict(cut_index=cut_index, cut_pts=cut["cut_pts"], source_seconds=cut["source_seconds"],
                              left=left["id"], right=right["id"], left_pts=before.tolist(), right_pts=after.tolist(),
                              **stats(residual, status)))
        print(f"{cut['source_seconds']:.2f}s: {len(rows)} original background rows", flush=True)
    result = {key: np.concatenate([cut[key] for cut in saved]) for key in saved[0]}
    return result, summaries


def all_background_rows(native, camera, definition, row_data, source_data, exposures, height, out):
    """Write one shot at a time in bounded native projection batches."""
    timeline, _, _ = camera
    point_pts, uv, path_codes, source_rows, stations = row_data
    paths, groups = source_data
    if any(len(values) != len(point_pts) for values in row_data):
        raise ValueError("Frozen background arrays differ in row count")
    if len(path_codes) and (path_codes.min() < 0 or path_codes.max() >= len(paths)):
        raise ValueError("Frozen background row refers to an unknown material path")
    clock, clock_shots = exposures["source_pts"], exposures["shot_id"]
    by_exposure = dict(
        source_pts=clock, shot_id=clock_shots, camera_index=exposures["camera_index"],
        rows=np.zeros(len(clock), dtype=np.int32), valid=np.zeros(len(clock), dtype=np.int32),
        invalid=np.zeros(len(clock), dtype=np.int32), rms_px=np.full(len(clock), np.nan),
        p95_px=np.full(len(clock), np.nan))
    metadata = {key: exposures[key] for key in
                ("time_base", "camera_state_sha256", "metres_per_scene_unit",
                 "solver_metres_per_scene_unit", "length_export_ratio")}
    shards, shot_stats = [], {}
    # Only one scalar error per admitted row is retained across shots for exact programme p95.
    errors = np.empty(len(point_pts))
    n_valid, expected_begin = 0, 0
    status_counts = Counter()
    signed_sum = np.zeros(2)
    shard_dir = out / "background_shards"
    shard_dir.mkdir()
    for shot in timeline.shots:
        shot_id = shot["id"]
        begin, stop = shot["row_begin"], shot["row_stop"]
        if begin != expected_begin or not begin <= stop <= len(point_pts):
            raise ValueError(f"Invalid original row partition for {shot_id}")
        expected_begin = stop
        shot_pts = point_pts[begin:stop]
        frame_ids = np.searchsorted(clock, shot_pts)
        if np.any(frame_ids == len(clock)):
            raise ValueError(f"Background row outside the native source clock in {shot_id}")
        np.testing.assert_array_equal(clock[frame_ids], shot_pts)
        if np.any(clock_shots[frame_ids] != shot_id):
            raise ValueError(f"Background rows do not bind to their declared shot {shot_id}")
        residuals = np.empty((stop - begin, 2))
        statuses = np.empty(stop - begin, dtype=np.int32)
        for batch, start in enumerate(range(begin, stop, BACKGROUND_BATCH_ROWS)):
            end = min(start + BACKGROUND_BATCH_ROWS, stop)
            rows = slice(start, end)
            stamps, observed, path_indices = point_pts[rows], uv[rows], path_codes[rows]
            predicted, status = project_rows(
                native, camera, definition, paths[path_indices], stations[rows], stamps, observed,
                ((shot_id, slice(None)),))
            residual = predicted - observed
            evidence = row_camera_evidence(native, camera, exposures, stamps, predicted, status, height)
            target = shard_dir / f"{shot_id}_{batch:04d}.npz"
            np.savez_compressed(
                target, point_input_row=np.arange(start, end, dtype=np.int64), source_row=source_rows[rows],
                path_index=path_indices, station=stations[rows], observed_uv_px=observed,
                predicted_uv_px=predicted, residual_uv_px=residual, **metadata, **evidence)
            shards.append(dict(shot_id=shot_id, camera_index=shot["camera_index"], rows=end-start,
                               point_input_row_range=[start, end], **identity(target)))
            local = slice(start - begin, end - begin)
            residuals[local], statuses[local] = residual, status
            del evidence

        valid = statuses == 0
        shot_error = np.linalg.norm(residuals[valid], axis=1)
        count = len(shot_error)
        errors[n_valid:n_valid + count] = shot_error
        n_valid += count
        status_counts.update(map(int, statuses[~valid]))
        signed_sum += residuals[valid].sum(axis=0)
        shot_frames = np.flatnonzero(clock_shots == shot_id)
        # Original row order is preserved in the files. Group by true frame only for statistics.
        order = np.argsort(frame_ids, kind="stable")
        bounds = np.searchsorted(frame_ids[order], np.r_[shot_frames, shot_frames[-1] + 1])
        for frame, first, last in zip(shot_frames, bounds[:-1], bounds[1:], strict=True):
            rows = order[first:last]
            summary = stats(residuals[rows], statuses[rows])
            for key in ("rows", "valid", "invalid"):
                by_exposure[key][frame] = summary[key]
            if summary["valid"]:
                by_exposure["rms_px"][frame] = summary["rms_px"]
                by_exposure["p95_px"][frame] = summary["p95_px"]
        shot_stats[shot_id] = dict(
            camera_index=shot["camera_index"], exposures=len(shot_frames),
            exposures_with_rows=int(np.count_nonzero(by_exposure["rows"][shot_frames])),
            exposures_with_valid_rows=int(np.count_nonzero(by_exposure["valid"][shot_frames])),
            point_input_row_range=[begin, stop], **stats(residuals, statuses))
        print(f"{shot_id}: {stop-begin} original rows, {count} valid", flush=True)

    if expected_begin != len(point_pts) or by_exposure["rows"].sum() != len(point_pts):
        raise ValueError("Full background export did not account for every frozen input row")
    exposure_path = out / "background_by_exposure.npz"
    np.savez_compressed(exposure_path, **by_exposure, **metadata)
    summary = dict(
        rows=len(point_pts), valid=n_valid, invalid=len(point_pts)-n_valid,
        invalid_native_status_counts=dict(status_counts),
        rms_px=float(np.sqrt(np.mean(errors[:n_valid]**2))) if n_valid else None,
        p95_px=float(np.quantile(errors[:n_valid], .95)) if n_valid else None,
        mean_signed_uv_px=(signed_sum / n_valid).tolist() if n_valid else None,
        exposures=len(clock), exposures_with_rows=int(np.count_nonzero(by_exposure["rows"])),
        exposures_with_valid_rows=int(np.count_nonzero(by_exposure["valid"])))
    return dict(background=summary, background_by_shot=shot_stats, background_mode="all",
                material_path_table=dict(path_id=paths.tolist(), material_group=groups.tolist()),
                outputs=dict(background_shards=shards, background_by_exposure=identity(exposure_path)))


def export(project, prepared, camera_record, cut_review, output, *, all_background=False):
    store = AssetStore(project)
    out = guard_project_write(output, project_root=project)
    if not store.configured or not out.is_relative_to(store.data.resolve()):
        raise ValueError("Evidence output belongs in the registered Data root")
    if out.exists():
        raise FileExistsError(f"Preserve existing camera evidence: {out}")
    exposure_path, background_path, receipt_path = (out / name for name in
                                                     ("exposures.npz", "background_rows.npz", "receipt.json"))
    if any(len(str(path).encode("utf-16-le")) // 2 >= 260 for path in
           (exposure_path, background_path, receipt_path)):
        raise ValueError(f"Windows evidence path is too long: {out}")
    scope = project / PROGRAMME / "rebuild_scope.json"
    source_path = project / PROGRAMME / "primary_source_selection.json"
    primary = read_json(source_path)["primary_source"]
    review = read_json(cut_review)
    pts, shots, time_base = required_clock(read_json(scope))
    source_size = [primary["width"], primary["height"]]
    if (review.get("schema") != "rinkspace.structure_cut_review.v1"
            or review["original_source_sha256"] != primary["sha256"]
            or review["source_size"] != source_size or tuple(review["time_base"]) != time_base
            or review["programme_exposures"] != len(pts) or review["reviewed_exposures"] != 80
            or len(review["cuts"]) != 8):
        raise ValueError("Cut review must bind the eight registered cuts and 80 native exposures")

    record = read_json(camera_record)
    station_root = store.source(record["prepared"])
    manifest_path = station_root / "joint_manifest.json"
    mapping_path = station_root / "mapping.json"
    points_path = station_root / "point_observations.npz"
    manifest, mapping = read_json(manifest_path), read_json(mapping_path)
    state_path = store.source(record["state"]["path"])
    compact_path = store.source(record["state"]["compact_path"])
    scene_path = store.source(manifest["scene_path"])
    camera_definition_path = store.source(manifest["cameras"][0]["camera_definition"])
    definition = read_json(camera_definition_path)
    if definition["image_height"] != primary["height"]:
        raise ValueError("The native row equation must use the original raster height")
    inputs = dict(source_selection=identity(source_path), scope=identity(scope), cut_review=identity(cut_review),
                  original_source=identity(store.source(primary["path"]), primary["sha256"]),
                  body_prepared_input=identity(prepared / "input.npz"), camera_record=identity(camera_record),
                  point_input=identity(points_path, record["point_input_sha256"]), mapping=identity(mapping_path),
                  station_state=identity(state_path, record["state"]["sha256"]),
                  camera_state=identity(compact_path, record["state"]["compact_sha256"]),
                  scene=identity(scene_path, record["scene_sha256"]), manifest=identity(manifest_path),
                  native_camera_definition=identity(camera_definition_path))
    camera_sha = inputs["camera_state"]["sha256"]
    paths = np.asarray(mapping["path_ids"])
    path_digest = hashlib.sha256(json.dumps(paths.tolist(), separators=(",", ":")).encode()).hexdigest()
    with (np.load(points_path, allow_pickle=False) as points, np.load(state_path, allow_pickle=False) as state,
          np.load(compact_path, allow_pickle=False) as compact, np.load(prepared / "input.npz", allow_pickle=False) as body):
        deps = json.loads(state["joint_numeric_dependencies_json"].item())
        compact_deps = json.loads(compact["joint_numeric_dependencies_json"].item())
        if (deps != compact_deps or deps["paths_sha256"] != path_digest or deps["shots"] != manifest["shots"]
                or mapping["point_input_sha256"] != record["point_input_sha256"]
                or state["point_input_sha256"].item() != record["point_input_sha256"]
                or compact["point_input_sha256"].item() != record["point_input_sha256"]):
            raise ValueError("Camera dependencies, source paths and original point rows do not align")
        source_rows = points["source_rows"]
        np.testing.assert_array_equal(source_rows, state["source_rows"])
        scene_values = state["scene_values"]
        for key in ("scene_values", "controls", "rig"):
            np.testing.assert_array_equal(state[key], compact[key])
        row_data = (points["pts"], points["uv"], points["path_indices"], source_rows, state["stations"])
        np.testing.assert_array_equal(body["pts"], pts)
        np.testing.assert_array_equal(body["shot_ids"], shots)
        np.testing.assert_array_equal(body["time_base"], time_base)
        packet = {key: body[key] for key in ("pts", "metres_per_scene_unit")}
    camera = load_camera(compact_path, manifest["shots"])
    dispatch = ProgrammeCamera(project, packet, shots, record_path=camera_record)
    core = dispatch.core
    inputs["native_library"] = identity(core.library_path, record["native_library"]["sha256"])
    if any(row["state_sha256"] != camera_sha or row["native_library_sha256"] != inputs["native_library"]["sha256"]
           for row in dispatch.info["shots"].values()):
        raise ValueError("ProgrammeCamera differs from the declared frozen state or binary")
    scales = {key: dispatch.info[key] for key in
              ("metres_per_scene_unit", "solver_metres_per_scene_unit", "length_export_ratio")}
    exposures = exposure_samples(dispatch, camera, pts, shots, time_base, camera_sha, scales)
    print(f"{len(pts)} original exposures × {len(ROW_FRACTIONS)} native camera nodes evaluated", flush=True)

    scene = read_json(scene_path)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    if any(model.initial_state.layout[key] != deps[bound] for key, bound in
           (("layout_revision", "layout_revision"), ("scene_content_revision", "scene_revision"))):
        raise ValueError("Scene parameter state does not match the frozen layout")
    native = NativeSceneGPU(scene_path, core, parameter_state=ParameterState(model.initial_state.layout, scene_values))
    groups = material_groups(paths, native.scene)
    if all_background:
        shard_path = out / "background_shards" / f"{camera[0].shots[-1]['id']}_0000.npz"
        exposure_rows_path = out / "background_by_exposure.npz"
        if any(len(str(path).encode("utf-16-le")) // 2 >= 260 for path in (shard_path, exposure_rows_path)):
            raise ValueError(f"Windows full background path is too long: {out}")
        recorded_rows = record.get("rows", manifest["shots"][-1]["row_stop"])
        if len(row_data[0]) != recorded_rows:
            raise ValueError("Full background population differs from the frozen record")
        out.mkdir(parents=True, exist_ok=False)
        background_report = all_background_rows(native, camera, definition, row_data, (paths, groups),
                                                exposures, source_size[1], out)
    else:
        background, cuts = background_rows(native, camera, definition, row_data, (paths, groups), exposures,
                                           review["cuts"], source_size[1])
        background.update(time_base=np.asarray(time_base), camera_state_sha256=np.array(camera_sha), **scales)
        out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(background_path, **background)
        background_report = dict(
            outputs=dict(background_rows=identity(background_path)), cut_exposures=review["reviewed_exposures"],
            background=stats(background["residual_uv_px"], background["projection_status"]), cuts=cuts)
    np.savez_compressed(exposure_path, **exposures)
    background_report["outputs"] = dict(exposures=identity(exposure_path), **background_report["outputs"])
    receipt = dict(schema="rinkspace.structure_camera_evidence.v1", inputs=inputs,
                   source_time_base=list(time_base), source_size_px=source_size, original_exposures=len(pts),
                   nodes_per_exposure=len(ROW_FRACTIONS),
                   native_camera_status_counts=dict(Counter(map(int, exposures["camera_status"].ravel()))),
                   scene_content_revision=native.scene_revision, **background_report,
                   centre_units="frozen scene units", **scales,
                   motion_value_order=["pan_rad", "tilt_rad", "log_f_px", "zoom_log_ratio"],
                   brown_order=["k1", "k2", "p1", "p2"], camera_status_not_sampled=-1,
                   shot_layout=camera[0].shots, physical_cameras=manifest["cameras"],
                   semantics=[
                       "Node order follows image rows, with signed readout: midpoint + tau * (row_fraction - 0.5).",
                       "Chronological indices reorder those same five times. Native times use packed independent shot blocks.",
                       "Background projection uses frozen scene and fitted stations with the native CUDA implicit predicted-row equation.",
                       "Row times use predicted y/(height-1), without replacing it by measured y or clamping outside the image.",
                       "A failed projection has no solved row time or instantaneous row camera.",
                       "Residual is predicted minus original observed UV in native source pixels.",
                       "Valid background rows have native projection status zero. Missing source rows are not synthesized.",
                       "The conditional metre scale is the common uniform export conversion. The solver scale is its input convention.",
                       "Fitted stations and conditional scene mapping are model diagnostics.",
                   ],
                   fit_executed=False, independently_measured_world_points=False, accuracy_accepted=False)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(str(receipt_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "prepared", "camera-record", "cut-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--all-background", action="store_true",
                        help="Replay all frozen background rows in per-shot shards instead of the cut subset")
    args = parser.parse_args()
    export(args.project.resolve(), args.prepared.resolve(), args.camera_record.resolve(),
           args.cut_review.resolve(), args.output, all_background=args.all_background)


if __name__ == "__main__":
    main()
