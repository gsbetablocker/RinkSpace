"""Synthetic finalizer run: optional panel publication cannot interrupt delivery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from calibration.program_tracking import finalize_shot2_review as finalizer
from rinkspace.storage import AssetStore


def prepare(project, data, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    dataset = project / "events/example/programmes/synthetic/source_datasets/batch"
    model = project / "events/example/models/synthetic"
    data.mkdir(parents=True)
    write(project / "configs/storage.json", dict(schema_version="rinkspace.storage.v1", data_root=str(data)))
    for name, value in (("feature_registry.json", {"model_id": "synthetic"}),
                        ("scene.json", {}), ("ad2_material_template.json", {}), ("model.json", {})):
        write(model / name, value)
    (project / "START_POINT_EDITOR.ps1").write_text("# synthetic launcher", encoding="utf-8")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    mask = np.zeros((1080, 1920), np.uint8)
    records = []
    for index in (1, 2):
        image, exclusion = data / f"frame{index}.png", data / f"mask{index}.png"
        assert cv2.imwrite(str(image), frame) and cv2.imwrite(str(exclusion), mask)
        source = dict(native_pts=256256 + (index-1)*6400, time_base="1/12800",
            programme_frame_index_1based=index, source_video_sha256="a"*64, image_path=str(image),
            png_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            decoded_BGR_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(exclusion.read_bytes()).hexdigest(), source_time=f"00:00:20.{20+(index-1)*500:03}")
        records.append(dict(index=index, source=source, image_path=str(image), mask_path=str(exclusion)))
        features = [dict(id="board_"+role, target_id="board."+role,
            segments=[dict(uv=[[0, y], [1919, y]])]) for role, y in (("top", 100), ("contact", 200))]
        write(data / f"board_candidates/{index:06d}.json", dict(source=source, features=features))
        write(data / f"ice_candidates/{index:06d}.json", dict(source=source, features=[]))
    write(data / "frame_manifest.json", dict(event_id="example", programme_id="synthetic",
          shot_scope=dict(shot_index=2, native_frame_count=2), frames=records))
    write(data / "selection.json", {})
    np.savez(data / "camera_seeds.npz", programme_indices=[1, 2], camera25=np.zeros((2, 25)))
    for key, value in dict(ROOT=project, DATA=data, OUT=dataset, MODEL=model, SOURCE_MEDIA=data,
                           EXPECTED_SHOT=2).items():
        monkeypatch.setattr(finalizer, key, value)
    monkeypatch.setattr(finalizer, "make_binding", lambda packet, *a: dict(observations=[
        dict(target_id=f["target_id"], source_vertex_indices=[0, 1]) for f in packet["features"]]))
    monkeypatch.setattr(finalizer, "assemble", lambda *a: ([], {}, []))
    monkeypatch.setattr(finalizer, "evaluate_rows", lambda *a, **k: (np.zeros(4), np.zeros((4, 2))))
    return dataset


def test_no_publish_finishes_outputs_without_any_panel(tmp_path, monkeypatch):
    project, data = tmp_path / "project", tmp_path / "data"
    dataset = prepare(project, data, monkeypatch)
    monkeypatch.setattr(finalizer, "PUBLISH_ENTRY", False)
    finalizer.run()
    assert (dataset / "frames/000002.json").is_file()
    assert (dataset / "batch_manifest.json").is_file()
    receipt = json.loads((data / "assembly_validation.json").read_text())
    assert receipt["frame_count"] == 2 and Path(receipt["review_html"]).is_file()
    assert not AssetStore(project).project_entry().exists()


def test_required_panel_fails_before_observations_change(tmp_path, monkeypatch):
    project, data = tmp_path / "project", tmp_path / "data"
    dataset = prepare(project, data, monkeypatch)
    before = (finalizer.MODEL / "model.json").read_bytes()
    monkeypatch.setattr(finalizer, "PUBLISH_ENTRY", True)
    with pytest.raises(FileNotFoundError):
        finalizer.run()
    assert not dataset.exists()
    assert (finalizer.MODEL / "model.json").read_bytes() == before


def test_missing_old_shortcut_uses_available_generic_launcher(tmp_path, monkeypatch):
    project, data = tmp_path / "project", tmp_path / "data"
    prepare(project, data, monkeypatch)
    panel = AssetStore(project).project_entry()
    panel.parent.mkdir(parents=True)
    panel.write_text("<!-- CURRENT_TASK_BEGIN --><!-- CURRENT_TASK_END -->", encoding="utf-8")
    monkeypatch.setattr(finalizer, "PUBLISH_ENTRY", True)
    finalizer.run()
    text = panel.read_text(encoding="utf-8")
    assert "START_SHOT2_REVIEW.cmd" not in text
    assert "START_POINT_EDITOR.ps1" in text
    assert (data / "assembly_validation.json").is_file()
