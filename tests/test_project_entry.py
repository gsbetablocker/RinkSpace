"""Generated-panel links use the current document as the hash-navigation base."""
from pathlib import Path
import importlib.util
import json
import re
from urllib.parse import urljoin, urlsplit

from rinkspace.storage import AssetStore

ROOT = Path(__file__).resolve().parents[1]


def build_panel(project, data):
    def write(relative, value):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    write("configs/storage.json", dict(schema_version="rinkspace.storage.v1", data_root=str(data),
          public_project_paths=["README.md", "events", "calibration", "CHECK_PATHS.cmd"]))
    write("configs/project.example.json", {"sources": {}})
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
    module.refresh()
    return AssetStore(project).project_entry()


def test_real_generated_page_hash_links_stay_on_document_and_file_links_are_explicit(tmp_path):
    project, data = tmp_path / "project", tmp_path / "data"
    project.mkdir()
    data.mkdir()
    panel = build_panel(project, data)
    text = panel.read_text(encoding="utf-8")
    assert "<base" not in text.lower()
    hashes = re.findall(r'href="(#[^"]*)"', text)
    assert "#/events/ankara_2026" in hashes
    assert "#/events/ankara_2026/programmes/okada_mei_sp" in hashes
    for current in (panel.as_uri(), "http://127.0.0.1:8765/"):
        for target in hashes:
            assert urlsplit(urljoin(current, target)).path == urlsplit(current).path
    assert json.dumps(project.as_uri()) in text
    assert "new URL(h,projectRoot)" in text
    assert "__PROJECT_URI__" not in text and "__DATA_URI__" not in text
