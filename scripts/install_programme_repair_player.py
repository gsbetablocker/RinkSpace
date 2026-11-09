"""Register a finalized full-programme repair in a separate results player."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT)]

from calibration.program_tracking.body_tracking.programme_contract import required_clock  # noqa: E402
from rinkspace.path_policy import guard_project_write  # noqa: E402
from rinkspace.storage import AssetStore  # noqa: E402
from rinkspace.viewer3d.data import digest, motion_diagnostics_state  # noqa: E402


TEMPLATE_STEM = "START_STRUCTURE_CAMERA_1108"
DEFAULT_STEM = "START_PROGRAMME_REPAIR_1108"
SESSION_ID = "programme_repair_1108"
COMPARISONS = ("programme_action_finish_1107", "programme_structure_1108", "programme_structure_camera_1108")
BASE_CONFIG = "events/ankara_2026/programmes/okada_mei_sp/viewer3d_structure_camera_1108.json"
COM_FIELDS = ("com_world_mid",)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_clock(packet, pts, time_base, path):
    keys = [key for key in ("pts", "source_pts") if key in packet]
    if not keys:
        raise ValueError(f"Missing source PTS: {path}")
    for key in keys:
        values = packet[key]
        if values.dtype.kind not in "iu" or not np.array_equal(values, pts):
            raise ValueError(f"Source PTS differ in {key}: {path}")
    clock = packet.get("time_base")
    if clock is None or clock.dtype.kind not in "iu" or not np.array_equal(clock, time_base):
        raise ValueError(f"Source time base differs: {path}")


def verify_base(store, base, base_path, output_path, pts, time_base):
    project, data = store.project, store.data
    source_root = data / base["source_run"]
    scene = data / base.get("scene_root", base["source_run"])
    with np.load(require_file(source_root / "input.npz"), allow_pickle=False) as packet:
        check_clock(packet, pts, time_base, source_root / "input.npz")
    for path in (source_root / "compact_smplx.npz", scene / "review_scene.npz", scene / "review_scene.json",
                 data / base["camera_snapshot"] if "camera_snapshot" in base else scene / "camera_snapshot.npz"):
        require_file(path)
    store.resolve(project / base["model_asset"], must_exist=True)
    if base.get("metric_binding"):
        metric_root = data if base.get("metric_binding_base") == "data" else project
        require_file(metric_root / base["metric_binding"])
    for key in ("source_video", "support_observations", "support_world_regions"):
        if base.get(key):
            require_file(project / base[key])
    if base.get("blade_review_asset"):
        require_file(store.resolve(base["blade_review_asset"], must_exist=True))
    if base.get("source_navigation"):
        navigation = require_file(store.resolve(base["source_navigation"], base=base_path.parent, must_exist=True))
        if output_path.parent != base_path.parent and not Path(base["source_navigation"]).is_absolute():
            base["source_navigation"] = str(navigation)
    if base.get("source_time_base", list(time_base)) != list(time_base) or base["initial_source_pts"] not in pts:
        raise ValueError("Base player does not use the registered source clock")
    for variant in base["variants"]:
        result = require_file(source_root / variant["result"])
        with np.load(result, allow_pickle=False) as packet:
            check_clock(packet, pts, time_base, result)
        if "base" in variant:
            reference = require_file(source_root / variant["base"])
            with np.load(reference, allow_pickle=False) as packet:
                check_clock(packet, pts, time_base, reference)
        for key in ("parameter", "diagnostic_record"):
            if key in variant:
                require_file(store.resolve(variant[key], must_exist=True))
        for binding in ("camera_binding", "projection_runtime", "legacy_grid_reuse"):
            metadata = variant.get(binding, {})
            for reference in (metadata, *metadata.values()):
                if isinstance(reference, dict) and "path" in reference:
                    require_file(store.resolve(reference["path"], must_exist=True))


def bound_file(store, path, expected_hash=None):
    physical = require_file(store.resolve(path, must_exist=True))
    sha = digest(physical)
    if expected_hash is not None and sha != expected_hash:
        raise ValueError(f"SHA-256 differs: {physical}")
    return {"path": str(physical), "sha256": sha}


def candidate_variant(store, record_path, record, pts, shots, time_base):
    parameter = bound_file(store, record["parameter"], record["parameter_sha256"])
    result = bound_file(store, record["result"], record["result_sha256"])
    context = record["programme_context"]
    with np.load(result["path"], allow_pickle=False) as packet:
        check_clock(packet, pts, time_base, result["path"])
        motion_diagnostics_state(packet, len(pts), 1., COM_FIELDS)
        if json.loads(packet["programme_context_json"].item()) != context:
            raise ValueError("Saved result programme context differs from the formal record")
        camera = json.loads(packet["camera_binding_json"].item())
        saved_projection = "COCO_uv" in packet and "equipment_uv" in packet
    if context["continuous_state_registry"]["parameter_sha256"] != parameter["sha256"]:
        raise ValueError("Formal programme context points to a different parameter")
    shot_hashes = {row["shot_id"]: row["frozen_camera_state_sha256"] for row in context["shots"]}
    result_hashes = {shot: entry["state_sha256"] for shot, entry in camera["shots"].items()}
    if set(shot_hashes) != set(shots) or result_hashes != shot_hashes:
        raise ValueError("Saved camera shots differ from the formal programme context")
    state = bound_file(store, camera["state_path"], camera["state_sha256"])
    camera_record = bound_file(store, camera["record"])
    declaration = read_json(Path(camera_record["path"]))["state"]
    if (set(shot_hashes.values()) != {state["sha256"]} or
            declaration["compact_sha256"] != state["sha256"] or
            store.resolve(declaration["compact_path"], must_exist=True) != Path(state["path"])):
        raise ValueError("The formal camera record does not bind the saved camera state")
    variant = dict(
        id=record.get("variant_id", record["candidate_id"]), candidate_id=record["candidate_id"],
        label="全节目 COM 与足鞋修复候选（待复核）", kind="fullbody", result=result["path"],
        result_sha256=result["sha256"], shell_topology="saved", required_saved_diagnostics=list(COM_FIELDS),
        body_reference_label="programme_initial",
        body_refinement_intervals_seconds=[(pts[[0, -1]] * time_base[0] / time_base[1]).tolist()],
        diagnostic_record=str(record_path), diagnostic_record_sha256=digest(record_path),
        parameter=parameter["path"], parameter_sha256=parameter["sha256"],
        user_accepted=False, accuracy_accepted=False, blade_review_available=False,
        camera_binding=dict(record=camera_record, state=state, shot_state_sha256=shot_hashes,
                            frozen_camera_sha256=context["frozen_camera_sha256"]),
    )
    if saved_projection:
        variant["source_projection_origin"] = "own saved COCO_uv and equipment_uv"
    return variant


def verify_blade_review(store, asset, base, variant, pts, time_base):
    path = require_file(store.resolve(asset, must_exist=True))
    document = read_json(path)
    if document["schema"] != "rinkspace.blade-player-review.v1" or document["time_base"] != list(time_base):
        raise ValueError("Blade review asset has a different schema or source time base")
    variant_id = variant["id"]
    prefix = document["variants"].get(variant_id)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Blade review asset has no projection for " + variant_id)
    if base.get("blade_review_asset"):
        old_path = store.resolve(base["blade_review_asset"], must_exist=True)
        old_variants = read_json(old_path)["variants"]
        if any(document["variants"].get(key) != value for key, value in old_variants.items()):
            raise ValueError("Blade review asset would omit an existing comparison projection")
    binding = (document["prediction_bindings"][variant_id] if "prediction_bindings" in document
               else document["prediction_binding"])
    paths = {}
    for field in ("parameter", "result"):
        expected_hash = variant[field + "_sha256"]
        physical = require_file(store.resolve(binding[field], must_exist=True))
        if (binding[field + "_sha256"] != expected_hash or
                (physical != Path(variant[field]) and digest(physical) != expected_hash)):
            raise ValueError(f"Blade review {field} SHA-256 differs from the formal candidate")
        paths[field] = physical
    expected_camera = variant["camera_binding"]
    expected_sha = expected_camera["state"]["sha256"]
    with np.load(paths["parameter"], allow_pickle=False) as parameter:
        camera = json.loads(parameter["camera_binding_json"].item())
    shot_hashes = {shot: entry["state_sha256"] for shot, entry in camera["shots"].items()}
    if (binding.get("camera_state_sha256", expected_sha) != expected_sha or
            camera["state_sha256"] != expected_sha or shot_hashes != expected_camera["shot_state_sha256"]):
        raise ValueError("Blade review camera SHA-256 differs from the formal candidate")
    projection = require_file(store.resolve(document["projection_npz"], base=path.parent, must_exist=True))
    with np.load(projection, allow_pickle=False) as packet:
        check_clock(packet, pts, time_base, projection)
        for suffix in ("blade_uv", "boot_uv", "axis_rate_rad_s", "COCO_uv", "runner_edges_uv", "runner_upper_edges_uv"):
            if f"{prefix}_{suffix}" not in packet:
                raise ValueError(f"Blade review candidate projection is missing {suffix}")
    return path


def launcher_contents(project, output_config, variant_id, launcher_stem):
    directory = project / "launchers"
    powershell = (directory / f"{TEMPLATE_STEM}.ps1").read_text(encoding="utf-8-sig")
    command = (directory / f"{TEMPLATE_STEM}.cmd").read_text(encoding="ascii")
    config_arg = output_config.relative_to(project).as_posix().replace("'", "''")
    for old, new in ((f"'{COMPARISONS[-1]}'", f"'{variant_id}'"), (f"'{BASE_CONFIG}'", f"'{config_arg}'")):
        if powershell.count(old) != 1:
            raise ValueError("Structure camera PowerShell template has changed")
        powershell = powershell.replace(old, new)
    if command.count(TEMPLATE_STEM) != 1:
        raise ValueError("Structure camera command template has changed")
    command = command.replace(TEMPLATE_STEM, launcher_stem).replace("\n", "\r\n")
    return powershell.encode("utf-8-sig"), command.encode("ascii")


def install(project, base_config, candidate_record, output_config, launcher_stem=DEFAULT_STEM, *, blade_review_asset=None):
    project = Path(project).resolve()
    if re.fullmatch(r"START_[A-Za-z0-9_]+", launcher_stem) is None:
        raise ValueError("Launcher stem must start with START_ and use ASCII letters, digits or underscores")
    if Path(output_config).suffix.lower() != ".json":
        raise ValueError("The new player config must have a .json filename")
    requested = (project / output_config, project / "launchers" / f"{launcher_stem}.ps1",
                 project / "launchers" / f"{launcher_stem}.cmd")
    existing = [str(path) for path in requested if os.path.lexists(path)]
    if existing:
        raise FileExistsError("Preserve existing player config and launchers: " + ", ".join(existing))
    destinations = [guard_project_write(path, project_root=project) for path in requested]
    if any(not path.is_relative_to(project) for path in destinations):
        raise ValueError("Player outputs must stay inside the project")
    for path in destinations:
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    output_config = destinations[0]

    base_path, record_path = ((project / path).resolve() for path in (base_config, candidate_record))
    base, record = read_json(base_path), read_json(record_path)
    variant_id = record.get("variant_id", record["candidate_id"])
    for identifier in (record["candidate_id"], variant_id):
        if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9_]+", identifier) is None:
            raise ValueError("Invalid formal candidate identifier")
    if (project / record["viewer_config"]).resolve() != output_config:
        raise ValueError("Output config differs from the formal record's viewer_config")
    ids = [variant["id"] for variant in base["variants"]]
    if len(ids) != len(set(ids)) or variant_id in ids or not {base["default_variant"], *COMPARISONS}.issubset(ids):
        raise ValueError("Expected unique existing variants, a preserved default and a new candidate ID")
    if base.get("session_id") == SESSION_ID:
        raise ValueError("The repair player requires an independent session")

    store = AssetStore(project)
    scope_path = require_file(store.resolve(base["source_programme_scope"], must_exist=True))
    if digest(scope_path) != record["scope_sha256"]:
        raise ValueError("Formal programme scope SHA-256 differs")
    pts, shots, time_base = required_clock(read_json(scope_path))
    if (len(pts) != 8329 or len(set(shots)) != 9 or time_base != (1, 12800) or
            record["source_frames"] != len(pts) or record["source_time_base"] != list(time_base)):
        raise ValueError("Expected the full 8329-exposure native programme")
    variant = candidate_variant(store, record_path, record, pts, shots, time_base)
    verify_base(store, base, base_path, output_config, pts, time_base)
    if blade_review_asset is not None:
        base["blade_review_asset"] = str(verify_blade_review(store, blade_review_asset, base, variant, pts, time_base))
        variant.update(blade_review_available=True, source_projection_origin="own saved COCO_uv and native five-node shoe/blade UV")
    base["variants"].append(variant)
    base.update(trajectory_reference="com", session_id=SESSION_ID, latest_review_variant=variant_id,
                comparison_variants=[*COMPARISONS, variant_id],
                review_status="全节目 8329 个源曝光对照，新增修复候选的轨迹、人体、鞋刀和接触准确性待复核。")
    base.setdefault("initial_visible", {}).update(com=True, com_projection=True, trajectory=True)
    base.setdefault("diagnostic_semantics", {})["com"] = "形状体积分配与鞋质量近似的模型 COM，非实测人体质心"
    config = (json.dumps(base, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    payloads = (config, *launcher_contents(project, output_config, variant_id, launcher_stem))

    created = []
    try:
        for path, contents in zip(destinations, payloads, strict=True):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(contents)
    except OSError:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return dict(variant=variant_id, config=str(output_config), launchers=[str(path) for path in destinations[1:]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--launcher-stem", default=DEFAULT_STEM)
    parser.add_argument("--blade-review-asset", type=Path, help="Combined exact-PTS source panel projection asset")
    args = parser.parse_args(argv)
    try:
        installed = install(args.project, args.base_config, args.candidate_record, args.output_config,
                            args.launcher_stem, blade_review_asset=args.blade_review_asset)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(installed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
