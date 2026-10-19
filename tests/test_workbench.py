"""Workbench contract checks using disposable projects, never original media.

These tests protect provenance and rejection paths; they are not real-video
validation and do not establish any claimed geometric accuracy.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import threading
from pathlib import Path
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from rinkspace.io import dump_json, load_json
from rinkspace.web import Workbench


@pytest.fixture()
def project(tmp_path):
    """A source identity/index fixture; source bytes are never decoded here."""
    root = tmp_path
    source = root / "data/input/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"read-only source identity fixture")
    dump_json(root / "configs/project.example.json", {"sources": {
        "test": {"path": "data/input/source.mp4", "title": "Disposable test source", "read_only": True}}})
    directory = root / "outputs/workspace/test"
    frames = [{"source_frame_index": n, "pts": n * 256, "time_base": "1/12800",
               "source_time_s": n / 50, "width": 1920, "height": 1080}
              for n in range(401)]
    dump_json(directory / "frame_index.json", {"video_id": "a" * 64,
        "source_sha256": "a" * 64, "decoded_frame_index_complete": True, "frames": frames})
    dump_json(directory / "metadata.json", {"video_id": "a" * 64, "source_sha256": "a" * 64,
        "source_path": str(source.resolve()), "size_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "declared_average_rate": "50", "frame_count": len(frames), "duration_s": 8,
        "width": 1920, "height": 1080, "decoded_frame_index_complete": True})
    return root


def test_project_save_reload_and_undo_preserve_source_and_versions(project):
    app = Workbench(project)
    original = (project / "data/input/source.mp4").read_bytes()
    selected = app.edit({"action": "select_video", "video_id": "test"})
    assert selected["project"]["revision"] == 1
    marked = app.edit({"action": "add_mask", "video_id": "test", "source_frame_index": 100,
        "pts": 25600, "time_base": "1/12800", "coordinate_space": "image_px",
        "shape": "polygon", "points": [[10, 20], [50, 20], [50, 80]], "reason": "person"})
    annotation = marked["project"]["annotations"][0]
    assert annotation["points"] == [[10, 20], [50, 20], [50, 80]]
    assert annotation["source_frame_index"] == 100
    assert annotation["pts"] == 25600
    assert annotation["revision"] == 2
    app.edit({"action": "save"})
    loaded = Workbench(project)
    assert loaded.project["annotations"][0] == annotation
    undone = loaded.edit({"action": "undo"})["project"]
    assert undone["annotations"] == []
    assert undone["selected_video_id"] == "test"
    assert undone["revision"] == 3
    assert (project / "data/input/source.mp4").read_bytes() == original
    assert Workbench(project).project["annotations"] == []


def _write_run(root, run_id, source_reference, source_current=300):
    directory = root / "outputs/runs" / run_id
    rows = [
        {"sample_index": 0, "source_frame_index": source_reference,
         "source_time_s": source_reference / 50, "reference_sample_index": 0,
         "reference_frame_only": True, "status": "candidate"},
        {"sample_index": 1, "source_frame_index": source_current,
         "source_time_s": source_current / 50, "reference_sample_index": 0,
         "status": "candidate", "fixed_image": "fixed/000001.png", "mask_image": "overlays/000001.jpg"},
    ]
    dump_json(directory / "run.json", {"id": run_id, "video_id": "test", "status": "candidate"})
    dump_json(directory / "registration.json", {"frames": rows})
    return directory


def test_run_selection_uses_its_actual_reference_and_never_reuses_neighbor_map(project):
    app = Workbench(project)
    app.edit({"action": "add_keyframe", "video_id": "test", "source_frame_index": 17})
    _write_run(project, "20260919-new", 200)
    _write_run(project, "20260918-old", 100)
    row = app.frame_meta("test", 300, "20260918-old")
    assert row["source_frame_index"] == 300
    assert row["pts"] == 76800
    assert row["reference_frame_index"] == 100  # not user keyframe 17 or newer run 200
    assert "index=100" in row["reference_url"]
    assert "20260918-old" in row["fixed_url"]
    assert row["previous_evidence_frame"] == 100
    assert row["next_evidence_frame"] is None
    missing = app.frame_meta("test", 299, "20260918-old")
    assert missing["source_frame_index"] == 299
    assert missing["registration"]["status"] == "unavailable"
    assert "fixed_url" not in missing
    assert "mask_url" not in missing
    assert missing["previous_evidence_frame"] == 100
    assert missing["next_evidence_frame"] == 300


def test_rejected_run_review_reaches_frame_metadata_without_rewriting_geometry(project):
    app = Workbench(project)
    rejected = _write_run(project, "rejected-baseline", 100)
    _write_run(project, "unreviewed-comparison", 200)
    info = load_json(rejected / "run.json")
    info.update(review_disposition="rejected_plane_contamination",
                review_note="The fitted support includes boards and screen graphics.")
    dump_json(rejected / "run.json", info)
    before = (rejected / "registration.json").read_bytes()

    row = app.frame_meta("test", 300, "rejected-baseline")
    assert row["run_id"] == "rejected-baseline"
    assert row["review_disposition"] == "rejected_plane_contamination"
    assert row["registration"]["review_disposition"] == row["review_disposition"]
    assert row["registration"]["review_note"] == info["review_note"]
    assert row["registration"]["status"] == "candidate"  # original solver result preserved
    assert row["diagnostic_preview_only"] is True
    assert "失败诊断" in row["fixed_reason"]
    assert "不能作为独立验收通过" in row["error_reason"]
    assert "rejected-baseline" in row["fixed_url"]  # failed evidence remains inspectable
    assert row["reference_frame_index"] == 100
    assert (rejected / "registration.json").read_bytes() == before

    other = app.frame_meta("test", 300, "unreviewed-comparison")
    assert "review_disposition" not in other
    assert "review_disposition" not in other["registration"]
    assert "diagnostic_preview_only" not in other
    assert other["reference_frame_index"] == 200


@pytest.mark.parametrize("backend", [None, "photometric_affine", "baseline"])
def test_corrected_shot_boundary_and_source_pixel_protection_reach_local_run(project, monkeypatch, backend):
    pytest.importorskip("av")
    from rinkspace import local_registration, photometric_sequence, review, workspace
    directory = project / "outputs/workspace/test"
    automatic = {"shots": [
        {"shot_id": "shot_0000", "start_frame": 0, "end_frame_exclusive": 200,
         "start_s": 0, "end_s": 4, "role": "programme_candidate"},
        {"shot_id": "shot_0001", "start_frame": 200, "end_frame_exclusive": 401,
         "start_s": 4, "end_s": 8, "role": "programme_candidate"}]}
    dump_json(directory / "shots.automatic.json", automatic)
    original = (directory / "shots.automatic.json").read_bytes()
    app = Workbench(project)
    app.edit({"action": "edit_shot", "video_id": "test", "shot_id": "test:shot_0000",
        "label": "programme", "start_frame": 0, "end_frame": 204})
    app.edit({"action": "add_mask", "video_id": "test", "source_frame_index": 200,
        "coordinate_space": "image_px", "reason": "heldout_protection", "physical_landmark_id": "logo-A",
        "points": [[200, 100], [400, 100], [400, 300], [200, 300]]})
    captured = {}

    def sample_fixture(root, key, start, end, every_n, width, destination):
        captured['every_n'] = every_n
        rows = [{"sample_index": n, "source_frame_index": source_index, "pts": source_index * 256,
            "time_base": "1/12800", "source_time_s": source_index / 50,
            "image_scale_from_source": .5, "image": f"frame_{n:06d}.jpg"}
            for n, source_index in enumerate([190, 200, 210])]
        value = {"source_sha256": "a" * 64, "samples": rows}
        dump_json(destination / "samples.json", value)
        return value

    def register_fixture(source, out, **kwargs):
        captured.update(kwargs)
        result = {"frames": [], "decoded_dimensions": [960, 540]}
        dump_json(out / "registration.json", result)
        return result

    monkeypatch.setattr(workspace, "sample_source_window", sample_fixture)
    if backend == "baseline":
        monkeypatch.setattr(local_registration, "register_sequence", register_fixture)
    else:
        monkeypatch.setattr(photometric_sequence, "register_photometric_sequence", register_fixture)
    monkeypatch.setattr(review, "build_run_report", lambda out: load_json(out / "run.json"))
    request = {"video_id": "test", "start_s": 3.8, "end_s": 4.3,
               "config": {"exclusions": {"*": []}}}
    if backend is not None:
        request["backend"] = backend
    if backend == "baseline":
        request["every_n"] = 10
    run = app.run(request)
    assert run["backend"] == (backend or "photometric_affine")
    assert captured["every_n"] == run["every_n"] == (10 if backend == "baseline" else 1)
    assert captured["shot_breaks"] == [2]  # corrected cut is source 205, not original 200
    protected = captured["exclusions"]["1"][0]
    assert protected["physical_landmark_id"] == "logo-A"
    assert protected["polygon"] == [[100, 50], [200, 50], [200, 150], [100, 150]]
    assert captured["config"]["corrections"] == {}  # independent protection is not an ordinary mask edit
    assert "exclusions" not in captured["config"]  # request cannot erase project protection
    assert run["first_frame_index"] == 190
    assert run["project_revision"] == 2
    assert (directory / "shots.automatic.json").read_bytes() == original


def test_unknown_backend_rejected_before_source_sampling(project, monkeypatch):
    from rinkspace import workspace
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid backend must not sample source media")
    monkeypatch.setattr(workspace,"sample_source_window",forbidden)
    with pytest.raises(ValueError,match="Unknown registration backend"):
        Workbench(project).run({"video_id":"test","start_s":0,"end_s":1,"backend":"unregistered"})


def test_default_affine_run_writes_playable_report_and_local_audit(project, monkeypatch):
    av = pytest.importorskip("av")
    import cv2
    from rinkspace import workspace
    rng = np.random.default_rng(553)
    base = cv2.GaussianBlur(rng.integers(170,240,(180,320),dtype=np.uint8),(5,5),1.)

    def actual_samples(root, key, start, end, every_n, width, destination):
        assert every_n == 1
        destination.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i in range(5):
            im=cv2.warpAffine(base,np.float32([[1,0,i],[0,1,0]]),(320,180))
            name=f"frame_{i:06d}.png"
            cv2.imwrite(str(destination/name),cv2.cvtColor(im,cv2.COLOR_GRAY2BGR))
            n=190+i
            rows.append(dict(sample_index=i,source_frame_index=n,pts=n*256,time_base="1/12800",
                source_time_s=n/50,image=name,image_width=320,image_height=180,image_scale_from_source=1/6))
        dump_json(destination/"samples.json",dict(source_sha256="a"*64,source_filename="synthetic_workbench_fixture",
            every_n=1,samples=rows))

    monkeypatch.setattr(workspace,"sample_source_window",actual_samples)
    app=Workbench(project)
    info=app.run({"video_id":"test","start_s":3.8,"end_s":3.9})
    out=project/"outputs/runs"/info["id"]
    assert info["backend"] == "photometric_affine" and info["every_n"] == 1
    assert info["review_disposition"] == "candidate_not_independently_validated"
    assert (out/"review.zip").is_file() and (out/"index.html").is_file()
    with av.open(str(out/"comparison.mp4")) as media:
        assert len(list(media.decode(video=0))) == 5
    metadata=app.frame_meta("test",191,info["id"])
    assert metadata["registration"]["motion_model"] == "affine"
    assert metadata["registration"]["independent_validation"] is False
    assert metadata["photometric_audit_url"].endswith("footprints/000001.json.gz")
    assert metadata["photometric_audit_manifest_url"].endswith("footprints/manifest.json")
    assert metadata["registration"]["fit_points_full_count"] > 200
    assert len(load_json(out/"preview_map.json")["frames"]) == 5


def audit_fixture():
    points = [[180., 130.], [480., 120.], [780., 150.],
              [170., 410.], [470., 390.], [790., 400.]]
    groups = [{"physical_landmark_id": f"logo-{i}", "xy": xy, "radius_px": 25.}
              for i, xy in enumerate(points)]
    rows = []
    for sample in range(3):
        rows.append({"sample_index": sample, "source_frame_index": sample * 5,
            "source_time_s": sample / 10, "reference_sample_index": 0,
            "reference_frame_only": sample == 0, "H_reference_from_current": np.eye(3).tolist(),
            "status": "candidate", "protected_regions": copy.deepcopy(groups),
            "support_polygon": [[80, 60], [880, 60], [880, 490], [80, 490]],
            "fit_current_xy": [[100, 80], [850, 80], [850, 470], [100, 470]],
            "fit_reference_xy": [[100, 80], [850, 80], [850, 470], [100, 470]]})
    observations = [{"sample_index": sample, "reference_sample_index": 0,
        "point_id": f"point-{i}", "physical_landmark_id": f"logo-{i}",
        "source_xy": xy, "reference_xy": xy, "visible": True,
        "provenance": {"type": "manual_source_observation"}}
        for sample in range(3) for i, xy in enumerate(points)]
    return {"decoded_dimensions": [960, 540], "frames": rows}, observations


def _audit(registration, observations):
    pytest.importorskip("av")
    from rinkspace.review import independent_audit
    return independent_audit(registration, observations)


def test_audit_requires_protection_in_every_fitting_frame_and_handles_missing_checks():
    registration, observations = audit_fixture()
    result = _audit(registration, observations)
    assert result["accepted"] is True
    assert result["median_1920_px"] == 0
    registration["frames"][1]["protected_regions"].pop()
    result = _audit(registration, observations)
    assert result["accepted"] is False
    assert "logo-5" in result["missing_protection_groups"]
    empty = _audit(registration, [])
    assert empty["accepted"] is False
    assert empty["median_1920_px"] is None


def test_audit_rejects_same_id_protection_that_misses_the_observed_landmark():
    registration, observations = audit_fixture()
    for row in registration["frames"]:
        for region in row["protected_regions"]:
            region["xy"] = [20., 20.]
    result = _audit(registration, observations)
    assert result["accepted"] is False, "matching group labels cannot substitute for actual protection geometry"


def test_audit_separates_unprotected_candidate_errors_without_hiding_bad_matches():
    """A false NCC match must remain visible without polluting protected statistics."""
    registration, observations = audit_fixture()
    outside = next(p for p in observations if p["sample_index"] == 1 and p["point_id"] == "point-0")
    outside["source_xy"] = [480., 130.]  # 300 px away, outside its physical group's protection.
    footprint = next(p for p in observations if p["sample_index"] == 1 and p["point_id"] == "point-1")
    footprint["source_xy"] = [486., 120.]
    footprint["protected_from_fitting"] = False  # Centre can be inside while its patch footprint is not.

    result = _audit(registration, observations)
    assert result["accepted"] is False
    assert result["observation_count"] == 10
    assert result["ineligible_candidate_count"] == 2
    assert result["median_1920_px"] == pytest.approx(0, abs=1e-9)
    assert result["p95_1920_px"] == pytest.approx(0, abs=1e-9)
    assert result["all_candidate_error_summary"]["count"] == 12  # Reference identities remain excluded.
    assert result["all_candidate_error_summary"]["p95_1920_px"] == pytest.approx(276.6)
    checks = {(p["sample_index"], p["point_id"]): p for p in result["checks"]}
    assert len(checks) == len(observations)
    assert checks[(1, "point-0")]["error_1920_equivalent_px"] == pytest.approx(600)
    assert checks[(1, "point-1")]["error_1920_equivalent_px"] == pytest.approx(12)
    for point_id in ("point-0", "point-1"):
        assert checks[(1, point_id)]["measurement_eligible"] is False
        assert checks[(1, point_id)]["ineligible_reason"]
    assert any(v["reason"] == "observed_point_outside_its_protected_group" for v in result["protection_violations"])


def test_audit_does_not_count_never_observed_points_toward_six_landmarks():
    registration, observations = audit_fixture()
    for point in observations:
        if point["physical_landmark_id"] in {"logo-4", "logo-5"}:
            point["visible"] = False
    result = _audit(registration, observations)
    assert result["accepted"] is False, "four observed points plus two absent points are not six observed landmarks"


def test_audit_does_not_count_six_points_on_one_logo_as_six_physical_landmarks():
    registration, observations = audit_fixture()
    for row in registration['frames']:
        for region in row['protected_regions']:
            region['physical_landmark_id'] = 'one-large-logo'
    for point in observations:
        point['physical_landmark_id'] = 'one-large-logo'
    result = _audit(registration, observations)
    assert result['unique_visible_check_points'] == 6
    assert result['unique_physical_landmarks'] == 1
    assert result['frames_with_four_visible_checks'] == 0
    assert result['accepted'] is False


def test_audit_rejects_checks_entirely_outside_the_fitting_support():
    registration, observations = audit_fixture()
    for row in registration["frames"]:
        row["support_polygon"] = [[430, 220], [530, 220], [530, 320], [430, 320]]
        row["fit_current_xy"] = [[430, 220], [530, 220], [530, 320], [430, 320]]
        row["fit_reference_xy"] = copy.deepcopy(row["fit_current_xy"])
    result = _audit(registration, observations)
    assert result["accepted"] is False, "zero reprojection error cannot turn an entirely extrapolated check set into a supported local audit"


@pytest.mark.parametrize("coordinate_field", ["fit_current_xy", "fit_reference_xy"])
def test_audit_rejects_actual_training_support_inside_a_protected_landmark(coordinate_field):
    registration, observations = audit_fixture()
    registration["frames"][1][coordinate_field].append([180., 130.])
    result = _audit(registration, observations)
    assert result["accepted"] is False, "training leakage invalidates independent acceptance even when error is zero"


def test_offline_report_preserves_preview_timing_and_displays_independent_check_image(tmp_path):
    av = pytest.importorskip("av")
    from PIL import Image
    from rinkspace.review import build_run_report

    out = tmp_path / "run"
    source = out / "source"
    source.mkdir(parents=True)
    (out / "overlays").mkdir()
    rows = []
    for n, time_s in enumerate([1.0, 1.09, 1.22]):
        name = f"frame_{n:06d}.jpg"
        Image.new("RGB", (160, 90), (70 + n * 40, 100, 140)).save(source / name)
        Image.new("RGB", (160, 90), (60, 60, 60)).save(out / "overlays" / name)
        rows.append({"sample_index": n, "image": name, "source_frame_index": 50 + n * 5,
            "pts": round(time_s * 1000), "time_base": "1/1000", "source_time_s": time_s,
            "status": "unavailable", "reason": "synthetic_fixture_no_geometry",
            "mask_image": f"overlays/{name}", "graph_component_id": None})
    Image.new("RGB", (160, 90), (240, 160, 30)).save(out / "check-000001.jpg")
    dump_json(out / "registration.json", {"frames": rows, "sample_directory": str(source),
        "source_sha256": "b" * 64, "algorithm": "synthetic-test-only",
        "configuration_sha256": "c" * 64, "decoded_dimensions": [160, 90]})
    dump_json(out / "run.json", {"id": "synthetic-report", "role": "failure_control",
        "status": "candidate", "start_s": 1.0, "end_s": 1.3})
    dump_json(out / "independent_audit.json", {"accepted": False,
        "status": "independent_checks_incomplete_or_failed", "median_1920_px": None,
        "p95_1920_px": None, "frames_with_four_visible_checks": 0,
        "nonreference_frame_count": 3, "checks": []})
    build_run_report(out)
    mapping = load_json(out / "preview_map.json")
    assert mapping["source_sha256"] == "b" * 64
    assert [r["source_frame_index"] for r in mapping["frames"]] == [50, 55, 60]
    with av.open(str(out / "comparison.mp4")) as encoded:
        actual_times = [float(frame.pts * frame.time_base) for frame in encoded.decode(video=0)]
    assert actual_times == pytest.approx([0, .09, .22], abs=1e-5)
    report = (out / "index.html").read_text(encoding="utf-8")
    assert "check-000001.jpg" in report, "the entry point must show or directly link independent check imagery"
    with zipfile.ZipFile(out / "review.zip") as bundle:
        names = set(bundle.namelist())
    assert {"index.html", "comparison.mp4", "preview_map.json", "independent_audit.json", "check-000001.jpg"} <= names
    assert not any(name.startswith("source/") for name in names)


@pytest.fixture()
def http_workbench(project, monkeypatch):
    import rinkspace.web as web
    actual_server = web.ThreadingHTTPServer
    holder = {}
    ready = threading.Event()

    def create_server(address, handler):
        server = actual_server(address, handler)
        holder["server"] = server
        ready.set()
        return server

    monkeypatch.setattr(web, "ThreadingHTTPServer", create_server)
    worker = threading.Thread(target=web.serve, kwargs={"root": project, "port": 0}, daemon=True)
    worker.start()
    assert ready.wait(5), "local fixture server did not start"
    server = holder["server"]
    try:
        yield f"http://127.0.0.1:{server.server_port}", project
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_http_serves_derived_assets_and_rejects_path_escape(http_workbench):
    base, root = http_workbench
    (root / "private.txt").write_text("PRIVATE_FIXTURE_MARKER", encoding="utf-8")
    (root / "outputs/safe.txt").write_text("derived public evidence", encoding="utf-8")
    with urlopen(base + "/assets/outputs/safe.txt", timeout=5) as response:
        assert response.read() == b"derived public evidence"
    with urlopen(base + "/", timeout=5) as response:
        assert "冰面参照工作台" in response.read().decode("utf-8")
    for path in ("/assets/outputs/%2e%2e/private.txt", "/assets/%2e%2e/private.txt",
                 "/static/%2e%2e/web.py", "/assets/outputs/%2e%2e%5cprivate.txt"):
        with pytest.raises(HTTPError) as rejected:
            urlopen(base + path, timeout=5)
        assert rejected.value.code == 400
        assert b"PRIVATE_FIXTURE_MARKER" not in rejected.value.read()


def test_http_cross_origin_mutation_is_rejected_without_changing_project(http_workbench):
    base, root = http_workbench
    request = Request(base + "/api/project", data=json.dumps({"action": "select_video", "video_id": "test"}).encode(),
        headers={"Content-Type": "application/json", "Origin": "https://example.invalid"}, method="POST")
    with pytest.raises(HTTPError) as rejected:
        urlopen(request, timeout=5)
    assert rejected.value.code == 400
    assert not (root / "outputs/workbench/project.json").exists()
    with urlopen(base + "/api/state", timeout=5) as response:
        state = json.load(response)
    assert state["project"]["revision"] == 0
    assert state["project"]["selected_video_id"] is None


def test_registered_original_media_can_seek_without_index_and_never_accepts_a_path(http_workbench):
    base, root = http_workbench
    original = (root / 'data/input/source.mp4').read_bytes()
    # Media seeking is independent of the expensive full decoded-frame index.
    (root / 'outputs/workspace/test/metadata.json').unlink()
    with urlopen(base + '/api/state') as response:
        source = json.load(response)['project']['videos'][0]
    assert source['indexed'] is False
    assert source['media_url'] == '/api/media?video_id=test'
    for header, expected in [('bytes=3-8', original[3:9]), ('bytes=-6', original[-6:]),
                             ('bytes=8-', original[8:])]:
        request = Request(base + source['media_url'], headers={'Range': header})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == expected
            assert response.headers['Accept-Ranges'] == 'bytes'
            assert response.headers['Content-Range'].endswith('/' + str(len(original)))
    with urlopen(Request(base + source['media_url'], method='HEAD')) as response:
        assert int(response.headers['Content-Length']) == len(original)
        assert response.read() == b''
    for header in ('bytes=99999-', 'bytes=0-4,10-20', 'bytes=-0', 'nonsense'):
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(base + source['media_url'], headers={'Range': header}))
        assert rejected.value.code == 416
        assert rejected.value.headers['Content-Range'] == f'bytes */{len(original)}'
    with pytest.raises(HTTPError) as rejected:
        urlopen(base + '/api/media?video_id=../private.txt')
    assert rejected.value.code == 400
    assert (root / 'data/input/source.mp4').read_bytes() == original


def test_model_catalog_keeps_provenance_null_poses_and_model_revisions(http_workbench):
    base, root = http_workbench
    directory = root / 'outputs/rink_model/ankara/v0'
    dump_json(directory / 'rink_model.json', {'model_id': 'ankara', 'revision': 'v0',
        'status': 'candidate', 'geometry_level': 'projective', 'overview_path': 'overview.jpg',
        'reference_views': [{'id': 'wide', 'image_path': 'wide.jpg', 'pts': 901, 'time_base': '1/50'}],
        'landmarks': [{'physical_id': 'ring', 'observations': [{'source_id': 'test',
            'pts': 901, 'time_base': '1/50', 'image_path': '../../../../private.txt'}]}]})
    (directory / 'overview.jpg').write_bytes(b'derived preview fixture')
    (directory / 'wide.jpg').write_bytes(b'derived source crop fixture')
    (root / 'private.txt').write_text('private fixture', encoding='utf-8')
    frames = [{'source_frame_index': 11, 'source_pts': 2816, 'source_time_base': '1/12800',
               'source_time_s': .22, 'H_model_from_image': None, 'status': 'unavailable'}]
    dump_json(root / 'outputs/model_localization/test-run/localization.json', {
        'model_id': 'ankara', 'model_revision': 'v0', 'source_id': 'test', 'frames': frames})
    with urlopen(base + '/api/rink-models') as response:
        model = json.load(response)[0]
    assert model['status'] == 'candidate'
    assert model['reference_views'][0]['pts'] == 901
    assert model['reference_views'][0]['image_url'].endswith('/wide.jpg')
    assert model['landmarks'][0]['observations'][0]['image_url'] is None
    assert model['overview_url'].endswith('/overview.jpg')
    with urlopen(base + '/api/model-localizations?video_id=test') as response:
        listing = json.load(response)
    assert len(listing) == 1
    assert listing[0]['frame_count'] == 1
    assert listing[0]['start_s'] == .22
    assert 'frames' not in listing[0]
    with urlopen(base + '/api/model-localizations?video_id=other') as response:
        assert json.load(response) == []
    with urlopen(base + '/api/model-localization?run_id=test-run') as response:
        assert json.load(response)['frames'] == frames
    with pytest.raises(HTTPError) as rejected:
        urlopen(base + '/api/model-localization?run_id=../../private')
    assert rejected.value.code == 400


def test_whole_rink_catalog_uses_existing_api_without_overwriting_legacy_runs(http_workbench):
    base, root = http_workbench
    directory = root / 'outputs/rink_model/ankara/whole-v1'
    image = root / 'outputs/reference-whole.jpg'
    image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(b'derived reference fixture')
    dump_json(directory / 'rink_model.json', {'schema_version': 'rinkspace.whole-rink.v1',
        'model_id': 'ankara', 'revision': 'whole-v1', 'view_poses': [{'id': 'fixed',
        'image': 'outputs/reference-whole.jpg', 'source_time_s': 148., 'pts': 1894400,
        'time_base': '1/12800'}]})
    (directory / 'whole_rink.png').write_bytes(b'whole overview fixture')
    for folder in ('model_localization', 'whole_rink_localization'):
        dump_json(root / 'outputs' / folder / 'same-name/localization.json', {'source_id': 'test',
            'model_id': 'ankara', 'model_revision': 'whole-v1',
            'frames': [{'source_time_s': 148., 'status': 'candidate', 'H_model_from_image': None,
                        'projection_chain': {'reference_radial_distortion': {'k1': -.1}}}]})
    with urlopen(base + '/api/rink-models') as response:
        model = json.load(response)[0]
    assert model['overview_url'].endswith('/whole_rink.png')
    assert model['view_poses'][0]['image_url'].endswith('/reference-whole.jpg')
    with urlopen(base + '/api/model-localizations?video_id=test') as response:
        listing = json.load(response)
    assert {r['run_id'] for r in listing} == {'same-name', 'whole:same-name'}
    with urlopen(base + '/api/model-localization?run_id=whole%3Asame-name') as response:
        result = json.load(response)
    assert result['frames'][0]['H_model_from_image'] is None
    assert result['frames'][0]['projection_chain']['reference_radial_distortion']['k1'] == -.1
    before = (directory / 'rink_model.json').read_bytes()
    dump_json(root / 'configs/rink_model/active_delivery.json', {
        'model_path': 'outputs/rink_model/ankara/whole-v1/rink_model.json',
        'preferred_run_ids': {'test': 'whole:same-name'}})
    dump_json(directory / 'review_status.json', {'status': 'development_candidate', 'label_zh': '已审查的条件草模候选'})
    with urlopen(base + '/api/rink-models') as response:
        preferred = json.load(response)[0]
    assert preferred['preferred_delivery'] is True
    assert preferred['preferred_run_ids'] == {'test': 'whole:same-name'}
    assert preferred['review_label_zh'] == '已审查的条件草模候选'
    assert (directory / 'rink_model.json').read_bytes() == before


def test_browser_grid_rejects_wrong_model_hash_domain_and_sparse_time_gaps():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable for browser geometry contract check')
    script = Path(__file__).parents[1] / 'src/rinkspace/static/model_grid.js'
    check = r'''
const assert = require('node:assert/strict');
const grid = require(process.argv[1]);
const H=[[2,0,10],[0,3,20],[0,0,1]];
const restored=grid.project(grid.invert(H),grid.project(H,[7,11]));
assert.ok(Math.hypot(restored[0]-7,restored[1]-11)<1e-10);
assert.equal(grid.invert([[0,0,0],[0,0,0],[0,0,0]]),null);
const model={model_id:'m',revision:'v0'};
const run={model_id:'m',model_revision:'v0',source_sha256:'original'};
const frame={status:'observed',source_time_s:1,H_model_from_image:H,
  support_polygon_image:[[0,0],[100,0],[100,100]],image_domain:'original_image_px'};
assert.equal(grid.drawable(frame,model,run,'original'),null);
assert.match(grid.drawable(frame,{...model,revision:'v1'},run,'original'),/版本/);
assert.match(grid.drawable(frame,model,run,'other'),/SHA-256/);
assert.match(grid.drawable({...frame,H_model_from_image:null},model,run,'original'),/矩阵/);
assert.match(grid.drawable({...frame,support_polygon_image:[]},model,run,'original'),/支持范围/);
assert.match(grid.drawable({...frame,image_domain:'undistorted'},model,run,'original'),/坐标域/);
assert.match(grid.drawable({...frame,status:'unavailable'},model,run,'original'),/无法定位/);
const frames=[{...frame,source_time_s:1},{...frame,source_time_s:1.02},
 {...frame,source_time_s:3,status:'unavailable',H_model_from_image:null}];
assert.equal(grid.frameAtTime(frames,1.004),frames[0]);
assert.equal(grid.frameAtTime(frames,1.019),frames[0]);
assert.equal(grid.frameAtTime(frames,1.020),frames[1]);
assert.equal(grid.frameAtTime(frames,2),null);
assert.equal(grid.frameAtTime(frames,3),frames[2]);
assert.equal(grid.frameAtTime(frames,.5,{media_time_offset_s:.5}),frames[0]);
assert.equal(grid.frameAtTime([],0),null);
const horizon=grid.projectedLine([[1,0,0],[0,1,0],[1,0,0]],[[-2,1],[2,1]]);
assert.equal(horizon.length,2);
assert.ok(horizon[0].every(p=>p[1]<0));
assert.ok(horizon[1].every(p=>p[1]>0));
assert.deepEqual(grid.landmarkPoint({curve_model:[[0,0],[10,0],[10,10],[0,10]]}),[5,5]);
assert.equal(grid.landmarkPoint({curve_model:[[0,0],[NaN,10]]}),null);
// Separate subtraction passes keep overlapping person boxes excluded.
let passes=0;const clip={canvas:{width:100,height:100},beginPath(){},rect(){},moveTo(){},lineTo(){},closePath(){},clip(rule){assert.equal(rule,'evenodd');passes++;}};
grid.clipForeground(clip,{person_boxes_image:[[10,10,40,40],[20,20,50,50]],foreground_polygons_image:[[[20,20],[70,20],[40,60]]]});
assert.equal(passes,3);
assert.match(grid.reasonText('no_model_ice_anchor'),/没有找到足够的冰面图案对应/);
assert.match(grid.reasonText('bounded_masked_ice_photometric_propagation_from_model_anchor'),/按冰面纹理连续推算/);
assert.match(grid.reasonText('new_unknown_internal_error'),/来源诊断/);
assert.equal(grid.landmarkLabel({physical_id:'R1_gray_near',label:'R1_gray_near'}),'R1 近侧灰靶');
assert.equal(grid.landmarkLabel({physical_id:'R4_red_far'}),'R4 远侧红靶');
const proxyFrames=[{media_time_s:0,source_pts:10000},{media_time_s:.02,source_pts:10020},{media_time_s:.04,source_pts:10040}];
assert.equal(grid.frameAtTime(proxyFrames,grid.seekFrameTime(proxyFrames,1)),proxyFrames[1]);
assert.equal(grid.presentedClockTime(.039,.02),.02); // paused clock cannot advance displayed PTS
assert.equal(grid.presentedClockTime(.04,.02,true),.04); // seek discards old presented frame
assert.equal(grid.presentedClockTime(.06,.02,false,true,.04),.04); // ended keeps final source frame
'''
    result = subprocess.run([node, '-e', check, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
