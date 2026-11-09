"""Exercise publication in an isolated project with no model or GUI."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.path_policy import OneDrivePathError
from scripts import install_programme_repair_player as installer


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def repair(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data = tmp_path / "data"
    monkeypatch.setenv("RINKSPACE_DATA_ROOT", str(data))
    for directory in (project / "launchers", project / "viewer", data / "source", data / "scene"):
        directory.mkdir(parents=True)
    for suffix in ("ps1", "cmd"):
        name = f"{installer.TEMPLATE_STEM}.{suffix}"
        shutil.copyfile(installer.PROJECT / "launchers" / name, project / "launchers" / name)

    scope_path = project / "viewer" / "scope.json"
    real_scope = installer.PROJECT / Path(installer.BASE_CONFIG).parent / "rebuild_scope.json"
    shutil.copyfile(real_scope, scope_path)
    scope = installer.read_json(scope_path)
    pts, shots, time_base = installer.required_clock(scope)
    source = data / "source"
    clock = dict(pts=pts, source_pts=pts, time_base=time_base)
    np.savez(source / "input.npz", **clock)
    for filename in ("source/compact_smplx.npz", "scene/review_scene.npz", "scene/review_scene.json",
                     "scene/legacy_camera.npz", "model.npz"):
        (data / filename).touch()
    (project / "viewer" / "navigation.json").touch()
    variant_ids = [*installer.COMPARISONS, "old_default"]
    variants = []
    for index, variant_id in enumerate(variant_ids):
        result = f"{variant_id}.npz"
        np.savez(source / result, **clock)
        variants.append(dict(id=variant_id, result=result, kind="fullbody", label=f"old candidate {index}"))
    base = dict(source_run="source", scene_root="scene", camera_snapshot="scene/legacy_camera.npz",
                model_asset=str(data / "model.npz"), source_navigation="navigation.json",
                source_programme_scope=str(scope_path), default_variant="old_default",
                session_id="old_session", initial_source_pts=int(pts[0]), variants=variants,
                initial_visible=dict(surface=False, com=False, com_projection=False, trajectory=False),
                diagnostic_semantics=dict(com="old COM description", blade_material="saved marker"),
                review_status="Three full states")
    base_path = project / "viewer" / "base.json"
    write_json(base_path, base)

    camera_path = project / "camera_state.npz"
    np.savez(camera_path, controls=np.array([1., 2., 3.]))
    camera_hash = installer.digest(camera_path)
    camera_record = project / "camera.json"
    write_json(camera_record, {"state": {"compact_path": str(camera_path), "compact_sha256": camera_hash}})
    camera = dict(record=str(camera_record), state_path=str(camera_path), state_sha256=camera_hash,
                  shots={str(shot): dict(state_sha256=camera_hash) for shot in np.unique(shots)})
    parameter = project / "repair.npz"
    np.savez(parameter, state_kind=np.array("absolute_motion_v1"), camera_binding_json=np.array(json.dumps(camera)))
    parameter_hash = installer.digest(parameter)
    context = dict(frozen_camera_sha256=camera_hash,
                   shots=[dict(shot_id=str(shot), frozen_camera_state_sha256=camera_hash) for shot in np.unique(shots)],
                   continuous_state_registry=dict(parameter_sha256=parameter_hash))
    arrays = dict(**clock, com_world_mid=np.ones((len(pts), 3)),
                  programme_context_json=np.array(json.dumps(context)), camera_binding_json=np.array(json.dumps(camera)))
    result = data / "repair_result.npz"
    np.savez(result, **arrays)
    output = project / "viewer" / "repair's COM.json"
    record_path = project / "repair.json"
    record = dict(candidate_id="repair_candidate", variant_id="repair_candidate",
                  result=str(result), result_sha256=installer.digest(result),
                  parameter=str(parameter), parameter_sha256=parameter_hash, scope_sha256=installer.digest(scope_path),
                  source_frames=len(pts), source_time_base=list(time_base), programme_context=context,
                  viewer_config=output.relative_to(project).as_posix())
    write_json(record_path, record)
    stem = "START_TEST_REPAIR"
    outputs = [output, project / "launchers" / (stem + ".ps1"), project / "launchers" / (stem + ".cmd")]
    return SimpleNamespace(project=project, data=data, base=base, base_path=base_path, record=record,
                           record_path=record_path, arrays=arrays, outputs=outputs, stem=stem)


def install_fixture(repair, **options):
    return installer.install(repair.project, repair.base_path, repair.record_path, repair.outputs[0], repair.stem, **options)


@pytest.fixture
def blade_review(repair):
    asset = repair.data / "blade_review.json"
    projection = asset.with_suffix(".npz")
    n = len(repair.arrays["pts"])
    np.savez(projection, pts=repair.arrays["pts"], time_base=repair.arrays["time_base"],
             repair_blade_uv=np.zeros((n, 2, 3, 2)), repair_boot_uv=np.zeros((n, 2, 2, 2)),
             repair_axis_rate_rad_s=np.zeros((n, 2)), repair_COCO_uv=np.zeros((n, 17, 2)),
             repair_runner_edges_uv=np.zeros((n, 2, 2, 2, 2)), repair_runner_upper_edges_uv=np.zeros((n, 2, 2, 2, 2)))
    binding = {key: repair.record[key] for key in ("parameter", "parameter_sha256", "result", "result_sha256")}
    stage_parameter = repair.data / "stage_parameter.npz"
    shutil.copyfile(binding["parameter"], stage_parameter)
    binding["parameter"] = str(stage_parameter)
    binding["camera_state_sha256"] = repair.record["programme_context"]["frozen_camera_sha256"]
    old_asset = repair.data / "old_blade_review.json"
    old_variants = {"old_default": "old"}
    write_json(old_asset, dict(variants=old_variants))
    repair.base["blade_review_asset"] = str(old_asset)
    write_json(repair.base_path, repair.base)
    document = dict(schema="rinkspace.blade-player-review.v1", time_base=repair.record["source_time_base"],
                    projection_npz=projection.name, variants=dict(old_variants, repair_candidate="repair"),
                    prediction_bindings={"repair_candidate": binding})
    write_json(asset, document)
    return asset, old_asset, document


def test_installs_an_explicitly_bound_blade_review_asset(repair, blade_review):
    asset, old_asset, _ = blade_review
    original_config = repair.base_path.read_bytes()
    original_asset = old_asset.read_bytes()
    install_fixture(repair, blade_review_asset=asset)
    config = installer.read_json(repair.outputs[0])
    assert config["blade_review_asset"] == str(asset)
    assert config["variants"][-1]["blade_review_available"] is True
    assert repair.base_path.read_bytes() == original_config
    assert old_asset.read_bytes() == original_asset


@pytest.mark.parametrize("field", ["parameter", "result", "camera_state"])
def test_refuses_blade_review_bound_to_another_saved_state(repair, blade_review, field):
    asset, _, document = blade_review
    document["prediction_bindings"]["repair_candidate"][field + "_sha256"] = "0" * 64
    write_json(asset, document)
    with pytest.raises(ValueError, match="Blade review .* SHA-256 differs"):
        install_fixture(repair, blade_review_asset=asset)
    assert all(not path.exists() for path in repair.outputs)


def test_refuses_blade_review_if_the_replayed_parameter_copy_changed(repair, blade_review):
    asset, _, document = blade_review
    stage_parameter = Path(document["prediction_bindings"]["repair_candidate"]["parameter"])
    stage_parameter.write_bytes(b"changed after replay")
    with pytest.raises(ValueError, match="Blade review parameter SHA-256 differs"):
        install_fixture(repair, blade_review_asset=asset)
    assert all(not path.exists() for path in repair.outputs)


def test_refuses_blade_review_without_new_candidate(repair, blade_review):
    asset, _, document = blade_review
    del document["variants"]["repair_candidate"]
    write_json(asset, document)
    with pytest.raises(ValueError, match="no projection for repair_candidate"):
        install_fixture(repair, blade_review_asset=asset)
    assert all(not path.exists() for path in repair.outputs)


def test_registers_new_candidate_and_preserves_default_and_base_variants(repair):
    before = repair.base_path.read_bytes()
    output, ps, cmd = repair.outputs
    installed = install_fixture(repair)
    config = installer.read_json(output)
    assert installed == dict(variant="repair_candidate", config=str(output), launchers=[str(ps), str(cmd)])
    assert repair.base_path.read_bytes() == before
    for key in ("default_variant", "camera_snapshot", "source_navigation"):
        assert config[key] == repair.base[key]
    assert config["variants"][:-1] == repair.base["variants"]
    assert config["trajectory_reference"] == "com"
    assert config["session_id"] == "programme_repair_1108"
    assert config["latest_review_variant"] == "repair_candidate"
    assert config["comparison_variants"] == [*installer.COMPARISONS, "repair_candidate"]
    assert config["initial_visible"] == dict(surface=False, com=True, com_projection=True, trajectory=True)
    assert config["diagnostic_semantics"] == dict(
        com="形状体积分配与鞋质量近似的模型 COM，非实测人体质心", blade_material="saved marker")
    assert "8329" in config["review_status"] and config["review_status"] != repair.base["review_status"]
    new = config["variants"][-1]
    assert new["id"] == "repair_candidate" and new["kind"] == "fullbody"
    assert new["body_reference_label"] == "programme_initial"
    assert new["body_refinement_intervals_seconds"] == [[22.12, 188.68]]
    assert new["result_sha256"] == repair.record["result_sha256"]
    assert new["parameter_sha256"] == repair.record["parameter_sha256"]
    assert new["camera_binding"]["state"]["sha256"] == repair.record["programme_context"]["frozen_camera_sha256"]
    assert new["required_saved_diagnostics"] == ["com_world_mid"]
    powershell = ps.read_text(encoding="utf-8-sig")
    assert "$Variant='repair_candidate'" in powershell
    assert "'viewer/repair''s COM.json'" in powershell
    assert "-Config $reviewConfig -Variant $Variant -Fresh:$Fresh -Wait:$Wait" in powershell
    assert f'%~dp0{repair.stem}.ps1' in cmd.read_text(encoding="ascii")


@pytest.mark.parametrize("index", range(3))
def test_refuses_any_existing_output_before_writing(repair, index):
    repair.outputs[index].write_bytes(b"previous file")
    with pytest.raises(FileExistsError):
        install_fixture(repair)
    assert repair.outputs[index].read_bytes() == b"previous file"
    assert all(not path.exists() for i, path in enumerate(repair.outputs) if i != index)


def test_rejects_long_onedrive_output_path(repair):
    long_output = repair.project / ("x" * 245 + ".json")
    with pytest.raises(OneDrivePathError):
        installer.install(repair.project, repair.base_path, repair.record_path, long_output, repair.stem)
    assert not long_output.exists()
    assert all(not path.exists() for path in repair.outputs)


def test_rejects_validly_hashed_candidate_without_com(repair):
    del repair.arrays["com_world_mid"]
    result = Path(repair.record["result"])
    np.savez(result, **repair.arrays)
    repair.record["result_sha256"] = installer.digest(result)
    write_json(repair.record_path, repair.record)
    with pytest.raises(ValueError, match="com_world_mid"):
        install_fixture(repair)
    assert all(not path.exists() for path in repair.outputs)


def test_rejects_result_with_changed_hash(repair):
    with Path(repair.record["result"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        install_fixture(repair)
    assert all(not path.exists() for path in repair.outputs)


def test_rejects_old_variant_with_a_different_source_clock(repair):
    old = repair.data / "source" / repair.base["variants"][0]["result"]
    np.savez(old, pts=repair.arrays["pts"] + 1, time_base=repair.arrays["time_base"])
    with pytest.raises(ValueError, match="Source PTS differ"):
        install_fixture(repair)
    assert all(not path.exists() for path in repair.outputs)
