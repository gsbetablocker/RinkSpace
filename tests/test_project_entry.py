"""Generated-panel links use the current document as the hash-navigation base."""
from pathlib import Path
import importlib.util
import json
import re
import shutil
from urllib.parse import urljoin, urlsplit
import pytest

from rinkspace.storage import AssetStore

ROOT = Path(__file__).resolve().parents[1]


def build_panel(project, data):
    def write(relative, value):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    write("configs/storage.json", dict(schema_version="rinkspace.storage.v1", data_root=str(data),
          public_project_paths=["README.md", "events", "calibration", "launchers/CHECK_PATHS.cmd"]))
    write("configs/project.example.json", {"sources": {}})
    event = Path("events/ankara_2026")
    scene_dir = event / "scene"
    pointer = json.loads((ROOT / scene_dir / "current_fit.json").read_bytes())
    release = pointer["registration"]["path"]
    (project / scene_dir).mkdir(parents=True)
    for name in ("current_fit.json", release, pointer["previous_designation"]):
        shutil.copyfile(ROOT / scene_dir / name, project / scene_dir / name)
    for key in pointer["programme_names"]:
        directory = event / "programmes" / key
        (project / directory).mkdir(parents=True)
        shutil.copyfile(ROOT / directory / release, project / directory / release)
    scene = "events/ankara_2026/models/synthetic/scene.json"
    write(scene, dict(ice_boundary=dict(points_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
                      ice_features=[], advertisements=[]))
    write("events/ankara_2026/manifest.json", dict(
        working_model=dict(q_sha256="a" * 64),
        resources=[dict(id=name, project_path=scene) for name in ("working_scene", "registered_v025_scene")]))
    (project / "README.md").write_text("# Synthetic README", encoding="utf-8")
    entry = dict(full_video_path=str(data / "not-a-video.mp4"), source_time_base="1/12800", shots=[],
                 current_video_delivery=dict(frozen_inputs_path=str(data / "frozen.json")))
    legacy = ('<html><head><meta charset="utf-8"><base href="file:///old/"></head><body>'
              '<!-- CURRENT_TASK_BEGIN --><p>Synthetic state</p><!-- CURRENT_TASK_END -->'
              '<script id="entry-data" type="application/json">' + json.dumps(entry) + '</script></body></html>')
    (project / "START_HERE.html").write_text(legacy, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("refresh_fixture", ROOT / "scripts/refresh_project_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = project
    return module


def test_real_generated_page_hash_links_stay_on_document_and_file_links_are_explicit(tmp_path):
    project, data = tmp_path / "project", tmp_path / "data"
    project.mkdir()
    data.mkdir()
    module = build_panel(project, data)
    module.refresh()
    panel = AssetStore(project).project_entry()
    text = panel.read_text(encoding="utf-8")
    assert "<base" not in text.lower()
    hashes = re.findall(r'href="(#[^"]*)"', text)
    assert "#/events/ankara_2026" in hashes
    assert "#/events/ankara_2026/programmes/okada_mei_sp" in hashes
    for key in ("jin_shuxian_sp", "chen_xi_jin_sp", "nayeon_ko_sp", "okada_mei_sp"):
        assert f"#/events/ankara_2026/programmes/{key}" in hashes
        assert f'data-view="{key}"' in text
    assert text.count("本地缺少该版整片回放文件") == 4
    assert text.count("用户整片视觉验收：尚未确认") == 5
    assert "lane4_side_1118" in text and "s14_boards_1118" in text
    assert "20.800–188.420" in text
    assert 'href="events/ankara_2026/scene/provisional_final_1117.json"' in text
    for current in (panel.as_uri(), "http://127.0.0.1:8765/"):
        for target in hashes:
            assert urlsplit(urljoin(current, target)).path == urlsplit(current).path
    assert json.dumps(project.as_uri()) in text
    assert "new URL(h,projectRoot)" in text
    assert "__PROJECT_URI__" not in text and "__DATA_URI__" not in text


def test_bad_designated_registration_cannot_replace_existing_panel(tmp_path):
    project, data = tmp_path / "project", tmp_path / "data"
    module = build_panel(project, data)
    module.refresh()
    panel = AssetStore(project).project_entry()
    previous = panel.read_bytes()
    _, _, _, programmes = module.designated_fit()
    record, _ = programmes["nayeon_ko_sp"]
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        module.refresh()
    assert panel.read_bytes() == previous


def test_designated_programmes_match_source_scopes_and_geometry():
    import hashlib

    event = ROOT / "events/ankara_2026"
    pointer = json.loads((event / "scene/current_fit.json").read_bytes())
    joint = json.loads((event / "scene" / pointer["registration"]["path"]).read_bytes())
    for reference in joint["programmes"].values():
        folder, filename = reference["path"].replace("\\", "/").split("/")[-2:]
        directory = event / "programmes" / folder
        scope = json.loads((directory / "rebuild_scope.json").read_bytes())
        current = scope["current_result"]
        assert ROOT / current["path"] == directory / filename
        raw = (ROOT / current["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == current["sha256"] == reference["sha256"]
        fit = json.loads(raw)
        assert current["designation"] == pointer["designation"]["status"]
        assert current["geometry_version_id"] == joint["geometry_binding"]["version_id"]
        assert current["version_id"] == joint["version_id"] == fit["version_id"]
        assert current["user_visual_acceptance"] == fit["user_visual_acceptance"]
        assert [shot.rsplit("_", 1)[-1] for shot in fit["shots"]] == scope["required_shot_ids"]
        review = fit["full_review"]
        assert scope["processing_scope"]["native_frame_count"] == review["frames"]
        numerator, denominator = review["native_time_base"]
        assert [scope["processing_scope"][key] * numerator / denominator for key in ("first_pts", "last_pts")] == review["source_seconds"]
