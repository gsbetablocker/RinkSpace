"""Migration preserves source/model identity and bounds Web file access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rinkspace.storage import AssetStore
from rinkspace.workspace import load_sources, source_directory, source_path


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("legacy,target", [
    (".venv-gpu", "runtime_tools/vision_cuda"),
    (".venv-body", "runtime_tools/body_cuda"),
    ("environments/body", "runtime_tools/body"),
    ("models/cache", "cache/pretrained"),
])
def test_gpu_resources_prefer_configured_data_copy(tmp_path, legacy, target):
    project, data = tmp_path / "project", tmp_path / "data"
    (project / "configs").mkdir(parents=True)
    config = json.loads((ROOT / "configs/storage.defaults.json").read_text("utf8"))
    config.update(data_root=str(data), legacy_root=str(project))
    (project / "configs/storage.defaults.json").write_text(json.dumps(config), "utf8")
    old, current = project / legacy / "resource.bin", data / target / "resource.bin"
    for path, value in ((old, b"old"), (current, b"verified")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    store = AssetStore(project)
    assert store.resolve(old, must_exist=True) == current
    old.unlink()
    assert store.resolve(legacy + "/resource.bin", must_exist=True) == current


@pytest.fixture
def stored_project(tmp_path):
    project, data = tmp_path / "project", tmp_path / "data"
    (project / "configs").mkdir(parents=True)
    data.mkdir()
    config = {
        "schema_version": "rinkspace.storage.v1",
        "data_root": str(data), "event_root": "events/ankara_2026",
        "legacy_root": str(project),
        "legacy_paths": {
            "data/input": "events/ankara_2026/media/originals",
            "outputs": "events/ankara_2026/history/legacy_outputs",
            "evidence": "events/ankara_2026/evidence",
        },
        "fallback_paths": {"three_frame_refit": "events/ankara_2026/history/legacy_project/three_frame_refit"},
        "public_project_paths": ["outputs", "evidence", "three_frame_refit/deliverables"],
    }
    (project / "configs/storage.json").write_text(json.dumps(config), encoding="utf-8")
    (project / "configs/project.example.json").write_text(json.dumps({"sources": {}}), encoding="utf-8")
    return project, data


def test_legacy_relative_and_absolute_assets_use_canonical_data(stored_project):
    project, data = stored_project
    target = data / "events/ankara_2026/media/originals/source.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserved original")
    # A stale local file cannot override the registered canonical asset.
    old = project / "data/input/source.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"stale copy")
    store = AssetStore(project)
    assert store.resolve("data/input/source.mp4") == target
    assert store.resolve(old) == target
    assert store.output("runs/new") == data / "events/ankara_2026/runs/new"
    with pytest.raises(ValueError, match="outside registered"):
        store.resolve(project.parent / "unregistered.txt")


def test_moved_source_index_keeps_original_metadata_and_pts(stored_project):
    project, data = stored_project
    source = data / "events/ankara_2026/media/originals/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original video bytes")
    (project / "configs/project.example.json").write_text(json.dumps({
        "sources": {"programme": {"path": "data/input/source.mp4", "read_only": True}}
    }), encoding="utf-8")
    indexed = data / "events/ankara_2026/history/legacy_outputs/workspace/programme"
    indexed.mkdir(parents=True)
    metadata = {
        "source_path": str(project / "data/input/source.mp4"),
        "size_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
        "decoded_frame_index_complete": True, "video_id": "original-sha",
    }
    record = indexed / "metadata.json"
    record.write_text(json.dumps(metadata), encoding="utf-8")
    before = record.read_bytes()
    (indexed / "shots.automatic.json").write_text(json.dumps({"video_id": "original-sha", "shots": []}), encoding="utf-8")
    assert source_path(project, "programme") == source
    assert source_directory(project, "programme") == indexed
    assert load_sources(project)[0]["prepared"] is True
    assert record.read_bytes() == before
    assert source_directory(project, "unprocessed").is_relative_to(data)
    assert not (project / "outputs").exists()


def test_current_model_wins_and_moved_review_uses_fallback(stored_project):
    project, data = stored_project
    current = project / "three_frame_refit/deliverables/scene.json"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"formal model bytes")
    archived = data / "events/ankara_2026/history/legacy_project/three_frame_refit/deliverables/scene.json"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"historical model bytes")
    image = archived.with_name("high_56218880.png")
    image.write_bytes(b"original review image")
    store = AssetStore(project)
    assert store.resolve("three_frame_refit/deliverables/scene.json") == current
    assert store.resolve("three_frame_refit/deliverables/high_56218880.png") == image
    assert current.read_bytes() == b"formal model bytes"


def test_renamed_code_model_keeps_legacy_document_paths(stored_project):
    project, data = stored_project
    config_path = project / "configs/storage.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["project_paths"] = {"three_frame_refit": "calibration"}
    config["fallback_paths"]["calibration"] = config["fallback_paths"]["three_frame_refit"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    current = project / "calibration/deliverables/scene.json"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"unchanged formal model")
    image = data / "events/ankara_2026/history/legacy_project/three_frame_refit/deliverables/high.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"original evidence")
    store = AssetStore(project)
    assert store.resolve("three_frame_refit/deliverables/scene.json") == current
    assert store.resolve("three_frame_refit/deliverables/high.png") == image
    assert store.resolve("calibration/deliverables/high.png") == image


def test_web_routes_reject_escape_and_do_not_expose_code(stored_project):
    from rinkspace.web import Workbench
    project, data = stored_project
    app = Workbench(project)
    app.persist()
    assert app.path.is_relative_to(data)
    assert app.localization_cache.directory.is_relative_to(data)
    assert not (project / "outputs").exists()
    evidence = data / "events/ankara_2026/evidence/frame.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    assert app.asset_url(evidence) == "/assets/data/events/ankara_2026/evidence/frame.json"
    assert app.storage.public_asset("evidence/frame.json") == evidence
    assert app.storage.public_asset("data/events/ankara_2026/evidence/frame.json") == evidence
    for requested in ("../secret", "data/../../secret", "/absolute", "C:/Windows/win.ini", "project/configs/storage.json"):
        with pytest.raises(ValueError):
            app.storage.public_asset(requested)


def test_configured_data_symlink_cannot_escape_registered_root(stored_project):
    project, data = stored_project
    outside = project.parent / "private.txt"
    outside.write_text("private", encoding="utf-8")
    link = data / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Host does not permit test symlinks")
    with pytest.raises(ValueError, match="escapes"):
        AssetStore(project).public_asset("data/escape.txt")


def test_current_formal_model_hash_is_unchanged():
    from rinkspace.calibration_companion.active_scene import active_scene_path, load_active_scene
    config = json.loads((ROOT / "configs/rink_model/active_shared_scene3d.json").read_text(encoding="utf-8"))
    path = active_scene_path(ROOT)
    before = path.read_bytes()
    scene = load_active_scene(ROOT)
    assert config["revision"] == "v0.25-three-frame-refit"
    assert hashlib.sha256(before).hexdigest() == config["model_sha256"] == scene.sha256
    assert path.read_bytes() == before


def test_exact_project_relocation_preserves_old_references_and_missing_checks(stored_project):
    project, data = stored_project
    config_path = project / "configs/storage.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    legacy = project.parent / "original-checkout"
    config["legacy_root"] = str(legacy)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    old_relative = "events/example/observations/long-source-packet/original.json"
    new_relative = "events/example/observations/s1.json"
    target = project / new_relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable original evidence")
    aliases = {"schema": "rinkspace.project-path-aliases.v1", "aliases": [{
        "old_path": old_relative, "new_path": new_relative,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}
    (project / "configs/path_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")
    store = AssetStore(project)
    for reference in (old_relative, project / old_relative, legacy / old_relative):
        assert store.resolve(reference, must_exist=True) == target
    assert store.resolve("original.json", base=project / Path(old_relative).parent, must_exist=True) == target
    # Exact file aliases cannot relocate a sibling or turn a file into a prefix.
    assert store.resolve(Path(old_relative).with_name("other.json")) == project / Path(old_relative).with_name("other.json")
    target.unlink()
    assert store.resolve(old_relative) == target
    with pytest.raises(FileNotFoundError):
        store.resolve(old_relative, must_exist=True)


@pytest.mark.parametrize("field,value", [("old_path", "../outside.json"), ("new_path", "../outside.json"),
                                         ("old_path", "C:/outside.json"), ("new_path", "bad\x00path"),
                                         ("sha256", "missing-identity")])
def test_project_relocation_rejects_unsafe_or_unhashed_aliases(stored_project, field, value):
    project, _ = stored_project
    row = {"old_path": "old.json", "new_path": "short.json", "sha256": "a" * 64}
    row[field] = value
    (project / "configs/path_aliases.json").write_text(json.dumps({
        "schema": "rinkspace.project-path-aliases.v1", "aliases": [row]}), encoding="utf-8")
    with pytest.raises(ValueError):
        AssetStore(project)


def test_new_outputs_enforce_checkout_limit_and_keep_data_long_path_support(stored_project, tmp_path):
    from rinkspace.path_policy import OneDrivePathError
    project, data = stored_project
    long_relative = Path("a" * 80) / ("b" * 80) / ("c" * 80) / "result.json"
    unconfigured = tmp_path / "unconfigured"
    unconfigured.mkdir()
    store = AssetStore(unconfigured)
    with pytest.raises(OneDrivePathError):
        store.output(long_relative)
    # Historical references remain readable; only new output paths are guarded.
    assert store.resolve(long_relative) == unconfigured / long_relative
    configured = AssetStore(project)
    assert configured.output(long_relative).is_relative_to(data)


def test_review_renderer_writes_data_and_preserves_source_packets(stored_project, monkeypatch):
    from types import SimpleNamespace
    import cv2
    import numpy as np
    from calibration.program_tracking import render_opening_frame_batch as renderer

    project, data = stored_project
    dataset = project / "events/example/programmes/synthetic/source_datasets/one"
    media = data / "events/example/programmes/synthetic/results/one"
    (dataset / "frames").mkdir(parents=True)
    (media / "frames").mkdir(parents=True)
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(media / "frames/000001.png"), source)
    packet = dict(point_count=1, source=dict(native_pts=256256, time_base="1/12800",
                  decoded_BGR_sha256=hashlib.sha256(source.tobytes()).hexdigest()),
                  features=[dict(id="ice.synthetic", segments=[dict(uv=[[100, 100]])])])
    point = dataset / "frames/000001.json"
    point.write_text(json.dumps(packet), encoding="utf-8")
    legacy = dataset / "review.html"
    legacy.write_bytes(b"retained old local page")
    before = point.read_bytes()
    monkeypatch.setattr(renderer, "ROOT", project)
    monkeypatch.setattr(renderer, "DATA_ROOT", data)
    result = renderer.run(SimpleNamespace(dataset=dataset, media=media, html=None,
                                         expected_frames=1, skip_overlays=False))
    page = Path(result["review_html"])
    assert page.is_relative_to(data) and page.is_file()
    assert result["frames"] == 1 and result["total_points"] == 1
    assert "00:00:20.020" in page.read_text(encoding="utf-8")
    assert point.read_bytes() == before
    assert legacy.read_bytes() == b"retained old local page"
    assert (media / "overlays/000001.png").is_file()


def test_workbench_serves_external_panel_and_falls_back_to_generic_ui(stored_project, monkeypatch):
    import http.client
    import threading
    from rinkspace import web

    project, data = stored_project
    (project / "START_HERE.html").write_bytes(b"legacy panel must not become the public homepage")
    panel = AssetStore(project).project_entry()
    panel.parent.mkdir(parents=True)
    from test_project_entry import build_panel
    panel = build_panel(project, data)
    original = web.ThreadingHTTPServer
    ready = threading.Event()
    servers = []

    def bind(address, handler):
        instance = original(address, handler)
        servers.append(instance)
        ready.set()
        return instance

    monkeypatch.setattr(web, "ThreadingHTTPServer", bind)
    worker = threading.Thread(target=web.serve, args=(project, 0, False), daemon=True)
    worker.start()
    assert ready.wait(5)
    instance = servers[0]

    def get(path):
        connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        assert get("/") == (200, panel.read_bytes())
        assert get("/START_HERE.html") == (200, panel.read_bytes())
        panel.unlink()
        status, body = get("/")
        assert status == 200 and b"legacy panel" not in body
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(5)


def test_shared_storage_defaults_do_not_select_a_machine_root(tmp_path, monkeypatch):
    from rinkspace.storage import storage_cache_root
    import tempfile
    monkeypatch.delenv("RINKSPACE_DATA_ROOT", raising=False)
    project = tmp_path / "clone"
    (project / "configs").mkdir(parents=True)
    (project / "configs/storage.defaults.json").write_text(json.dumps(dict(
        schema_version="rinkspace.storage.v1", event_root="events/example",
        project_paths={"old-model": "events/example/models"},
        legacy_paths={"outputs": "events/example/history"})), encoding="utf-8")
    store = AssetStore(project)
    assert not store.configured
    assert store.output("runs").is_relative_to(project / "outputs")
    assert storage_cache_root(project) == Path(tempfile.gettempdir()) / "RinkSpace_Data"


def test_explicit_and_environment_data_roots_work_without_local_config(tmp_path, monkeypatch):
    from rinkspace.storage import storage_cache_root
    from rinkspace.web import Workbench
    project = tmp_path / "clone"
    (project / "configs").mkdir(parents=True)
    shared = project / "configs/storage.defaults.json"
    shared.write_text(json.dumps(dict(schema_version="rinkspace.storage.v1",
                      event_root="events/example")), encoding="utf-8")
    original = shared.read_bytes()
    data, override = tmp_path / "data", tmp_path / "override"
    monkeypatch.setenv("RINKSPACE_DATA_ROOT", str(data))
    store = AssetStore(project)
    assert store.data == data and store.event == data / "events/example"
    assert storage_cache_root(project) == data
    assert Workbench(project).path.is_relative_to(data)
    assert AssetStore(project, data_root=override).data == override
    assert shared.read_bytes() == original and not (project / "configs/storage.json").exists()


def test_shared_history_alias_survives_local_roots_and_legacy_settings(stored_project):
    project, data = stored_project
    old = "events/example/scene/old.json"
    shared = {"schema_version": "rinkspace.storage.v1",
              "legacy_paths": {old: "history/preserved.json", "outputs": "shared/outputs"}}
    (project / "configs/storage.defaults.json").write_text(json.dumps(shared), encoding="utf-8")
    target = data / "history/preserved.json"
    target.parent.mkdir(parents=True)
    original = b'{"uv":[11.25,7.5],"pts":320000,"quality":"pending"}\n'
    target.write_bytes(original)
    store = AssetStore(project)
    assert store.resolve(old, must_exist=True) == target
    assert store.resolve(project / old, must_exist=True) == target
    assert store.resolve("old.json", base=project / "events/example/scene", must_exist=True) == target
    assert store.resolve("outputs/file.json") == data / "events/ankara_2026/history/legacy_outputs/file.json"
    assert target.read_bytes() == original


def test_local_exact_history_override_still_takes_precedence(stored_project):
    project, data = stored_project
    old = "events/example/scene/old.json"
    shared = {"schema_version": "rinkspace.storage.v1", "legacy_paths": {old: "shared/original.json"}}
    (project / "configs/storage.defaults.json").write_text(json.dumps(shared), encoding="utf-8")
    config_path = project / "configs/storage.json"
    local = json.loads(config_path.read_text(encoding="utf-8"))
    local["legacy_paths"][old] = "local/original.json"
    config_path.write_text(json.dumps(local), encoding="utf-8")
    assert AssetStore(project).resolve(old) == data / "local/original.json"
