"""Check export planning only: no native library, fit, media or historical points."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "events/ankara_2026/programmes/okada_mei_sp"
sys.path.insert(0, str(HERE))
import export_native_review as generic
import export_native_six_review as historical
import render_native_joint_shots as renderer
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline


@pytest.fixture
def scope():
    return json.loads((HERE / "rebuild_scope.json").read_text(encoding="utf-8-sig"))


def source_case(scope, ids):
    shots, pts, stop = [], [], 0
    for shot in scope["shots"]:
        if shot["id"] not in ids:
            continue
        processing = shot["processing"]
        # Multiple measured points in an exposure are valid, not extra frames.
        selection = np.repeat(np.arange(processing["first_pts"],
                                        processing["stop_pts_exclusive"], 256), 2)
        shots.append(dict(id=shot["id"], row_begin=stop, row_stop=stop + len(selection),
                          **processing))
        pts.extend(selection)
        stop += len(selection)
    return dict(shots=shots), dict(pts=np.asarray(pts, dtype=np.int64))


def selections(manifest, points, scope):
    return historical._source_selections(manifest, points, scope, legacy_six=False)


@pytest.mark.parametrize("ids", [
    ["s01"], ["s03", "s04"], ["s07", "s08", "s09"],
    ["s01", "s02", "s03", "s04", "s05", "s06"],
    [f"s{i:02d}" for i in range(1, 10)],
])
def test_all_selected_shot_rows_and_native_exposures_are_complete(scope, ids):
    manifest, points = source_case(scope, ids)
    found = selections(manifest, points, scope)
    assert len(found) == len(ids)
    for shot, pts in zip(manifest["shots"], found):
        np.testing.assert_array_equal(pts, points["pts"][shot["row_begin"]:shot["row_stop"]:2])
        assert len(pts) == shot["native_frame_count"]
    if len(ids) == 9:
        combined = np.concatenate(found)
        assert combined[0] == 283136 and combined[-1] == 2415104
        assert len(combined) == scope["processing_scope"]["native_frame_count"] == 8329
        np.testing.assert_array_equal(np.diff(combined), 256)
        assert float(combined[0] * historical.TIME_BASE) == 22.120
        assert float(combined[-1] * historical.TIME_BASE) == 188.680


@pytest.mark.parametrize("change", [
    "unknown_shot", "duplicate_shot", "reverse_order", "row_gap", "row_overlap",
    "extra_rows", "missing_exposure", "off_grid_exposure", "wrong_stop",
    "s01_physical_start", "s09_excluded_tail", "shortened_shot",
])
def test_changed_ranges_or_incomplete_current_exposures_fail(scope, change):
    manifest, points = source_case(scope, scope["required_shot_ids"])
    if change == "unknown_shot":
        manifest["shots"][0]["id"] = "s10"
    elif change == "duplicate_shot":
        manifest["shots"][1]["id"] = "s01"
    elif change == "reverse_order":
        manifest["shots"].reverse()
    elif change == "row_gap":
        manifest["shots"][1]["row_begin"] += 1
    elif change == "row_overlap":
        manifest["shots"][1]["row_begin"] -= 1
    elif change == "extra_rows":
        points["pts"] = np.r_[points["pts"], points["pts"][-1]]
    elif change == "missing_exposure":
        points["pts"][2:4] = points["pts"][0]
    elif change == "off_grid_exposure":
        points["pts"][2] += 1
    elif change == "wrong_stop":
        manifest["shots"][-1]["stop_pts_exclusive"] += 256
    elif change == "s01_physical_start":
        manifest["shots"][0]["first_pts"] = scope["shots"][0]["physical"]["first_pts"]
    elif change == "s09_excluded_tail":
        manifest["shots"][-1].update({k: scope["shots"][-1]["physical"][k]
                                    for k in ("last_pts", "stop_pts_exclusive")})
    elif change == "shortened_shot":
        manifest["shots"][-1]["last_pts"] -= 256
    with pytest.raises(ValueError):
        selections(manifest, points, scope)


def test_empty_or_discontinuous_selection_is_not_a_complete_concat(scope):
    with pytest.raises(ValueError):
        selections(dict(shots=[]), dict(pts=np.empty(0, dtype=np.int64)), scope)
    manifest, points = source_case(scope, ["s01", "s03"])
    with pytest.raises(ValueError, match="contiguous"):
        selections(manifest, points, scope)


def test_old_six_entry_and_product_names_remain_compatible(scope, monkeypatch):
    ids = scope["required_shot_ids"][:6]
    manifest, points = source_case(scope, ids)
    assert len(historical._source_selections(manifest, points)) == 6
    manifest9, points9 = source_case(scope, scope["required_shot_ids"])
    with pytest.raises(ValueError, match="six distinct"):
        historical._source_selections(manifest9, points9)
    assert historical._review_names("candidate", legacy_six=True) == (
        "concat_six.txt", "candidate_six_complete.mp4")
    assert historical._review_names("candidate", legacy_six=False) == (
        "concat_selected.txt", "candidate_complete.mp4")
    calls = []
    monkeypatch.setattr(historical, "export", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["export_native_six_review.py", "--run-root", "input",
                                   "--state", "state.npz", "--output", "review", "--label", "candidate"])
    historical.main()
    assert calls[-1][1] == dict(legacy_six=True)
    generic.main()
    assert calls[-1][1] == dict(legacy_six=False)


def test_generic_python_entry_uses_shared_export(monkeypatch):
    calls = []
    monkeypatch.setattr(generic, "_export", lambda *args, **kwargs: calls.append((args, kwargs)))
    generic.export("run", "state.npz", "review", "candidate", "compact.npz")
    assert calls == [(("run", "state.npz", "review", "candidate", "compact.npz"),
                      dict(legacy_six=False))]


def sparse_selections(manifest, points, scope):
    return historical._source_selections(manifest, points, scope, legacy_six=False,
                                         allow_sparse_observations=True)


def test_missing_observations_keep_every_original_frame_and_count_gaps(scope):
    manifest, points = source_case(scope, scope['required_shot_ids'])
    expected = selections(manifest, points, scope)
    missing = [int(expected[0][0]), int(expected[6][len(expected[6])//2]), int(expected[8][-1])]
    old_pts = points['pts']
    keep = ~np.isin(old_pts, missing)
    points['pts'] = old_pts[keep]
    for shot in manifest['shots']:
        begin, stop = shot['row_begin'], shot['row_stop']
        shot['row_begin'] = int(np.count_nonzero(keep[:begin]))
        shot['row_stop'] = int(np.count_nonzero(keep[:stop]))
    with pytest.raises(ValueError, match='Incomplete current exposure'):
        selections(manifest, points, scope)
    found = sparse_selections(manifest, points, scope)
    np.testing.assert_array_equal(np.concatenate(found), np.concatenate(expected))
    assert sum(map(len, found)) == 8329
    coverage = historical._observation_coverage(manifest, points, found)
    assert sum(c['frames_without_source_observations'] for c in coverage) == 3
    assert sum(c['frames_with_source_observations'] for c in coverage) == 8326
    assert [pts for c in coverage for pts in c['empty_source_pts']] == missing


@pytest.mark.parametrize('change', ['off_grid', 'other_shot', 'row_gap', 'short_range', 'float_pts'])
def test_sparse_mode_still_rejects_unregistered_pts_rows_and_ranges(scope, change):
    manifest, points = source_case(scope, ['s08', 's09'])
    if change == 'off_grid':
        points['pts'][0] += 1
    elif change == 'other_shot':
        points['pts'][0] = manifest['shots'][1]['first_pts']
    elif change == 'row_gap':
        manifest['shots'][1]['row_begin'] += 1
    elif change == 'short_range':
        manifest['shots'][1]['last_pts'] -= 256
    else:
        points['pts'] = points['pts'].astype(float)
    with pytest.raises(ValueError):
        sparse_selections(manifest, points, scope)


def test_empty_shot_keeps_full_frames_and_metrics_are_missing(scope):
    manifest, points = source_case(scope, ['s09'])
    manifest['shots'][0].update(row_begin=0, row_stop=0)
    points['pts'] = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError, match='Invalid current observation rows'):
        selections(manifest, points, scope)
    selection, = sparse_selections(manifest, points, scope)
    coverage, = historical._observation_coverage(manifest, points, [selection])
    assert coverage['frames_without_source_observations'] == len(selection)
    assert coverage['frames_with_source_observations'] == 0
    rows = np.empty(0, dtype=int)
    assert renderer._key_frames(selection, rows, points['pts'], np.empty(0)) == {
        0, len(selection)//2, len(selection)-1}
    metrics = renderer._source_metrics(np.empty(0), rows)
    assert metrics == dict(source_observation_count=0, has_source_observations=False,
                           max_px=None, median_px=None, p95_px=None, above_20px=None)
    assert renderer._source_metric_lines(metrics) == ['No source observations', 'Metrics N/A']
    assert len(renderer._observation_times(ShotMotionTimeline(manifest['shots'], .02), points['pts'])) == 0


def test_observation_metrics_use_only_current_frame_rows():
    metrics = renderer._source_metrics(np.array([1., 100., 21.]), np.array([0, 2]))
    assert metrics['source_observation_count'] == 2
    assert metrics['has_source_observations'] is True
    assert metrics['max_px'] == 21. and metrics['median_px'] == 11.
    assert metrics['above_20px'] == 1
    assert renderer._source_metric_lines(metrics)[0] == 'Active source points 2 | invalid 0'


def test_empty_shot_does_not_change_other_observation_motion_times():
    def shot(begin, stop, control_begin):
        return dict(row_begin=begin, row_stop=stop, first_pts=512, last_pts=1024,
                    stop_pts_exclusive=1280, reference_pts=512,
                    time_base_num=1, time_base_den=12800,
                    control_begin=control_begin, first_control_time=-.02)
    shots = [shot(0, 1, 0), shot(1, 1, 8), shot(1, 2, 16)]
    pts = np.array([512, 1024], dtype=np.int64)
    timeline = ShotMotionTimeline(shots, .02, 2.)
    found = renderer._observation_times(timeline, pts)
    np.testing.assert_array_equal(found, [2.02, 2.38])


def test_sparse_and_library_flags_are_explicit_and_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(historical, 'export', lambda *a, **kw: calls.append((a, kw)))
    argv = ['export_native_review.py', '--run-root', 'run', '--state', 'state.npz',
            '--output', 'review', '--label', 'candidate', '--allow-sparse-observations',
            '--native-library', 'bound.dll']
    monkeypatch.setattr(sys, 'argv', argv)
    generic.main()
    assert calls[-1][1] == dict(legacy_six=False, allow_sparse_observations=True,
                               native_library=Path('bound.dll'))
    with pytest.raises(SystemExit):
        historical.main()
    monkeypatch.setattr(generic, '_export', lambda *a, **kw: calls.append((a, kw)))
    generic.export('run', 'state', 'out', 'candidate', allow_sparse_observations=True,
                   native_library=Path('bound.dll'))
    assert calls[-1][1] == dict(legacy_six=False, allow_sparse_observations=True,
                               native_library=Path('bound.dll'))


def test_explicit_library_override_preserves_prepared_manifest_and_state_binding(tmp_path):
    old, new = tmp_path/'prepared.dll', tmp_path/'bound.dll'
    old.write_bytes(b'prepared build')
    new.write_bytes(b'state build')
    manifest = dict(native_library=str(old), native_library_sha256=historical._digest(old),
                    shots=[dict(id='s09')], point_input_sha256='unchanged point binding')
    state = dict(native_library_provenance_sha256=np.array(historical._digest(new)))
    selected = historical._manifest_for_library(manifest, new)
    assert selected['native_library'] == str(new.resolve())
    assert manifest['native_library'] == str(old)
    assert selected['shots'] == manifest['shots']
    assert selected['point_input_sha256'] == manifest['point_input_sha256']
    historical._verify_native_library_binding(selected, state)
    with pytest.raises(ValueError, match='pass --native-library'):
        historical._verify_native_library_binding(manifest, state)
    with pytest.raises(ValueError, match='lacks native_library_provenance'):
        historical._verify_native_library_binding(selected, {})
    with pytest.raises(ValueError):
        historical._verify_native_library_binding(selected,
            dict(native_library_provenance_sha256=np.array(historical._digest(old))))


def test_sparse_export_freezes_binding_and_commands_without_rendering(tmp_path, monkeypatch):
    # Three registered native frames, two real observations, no GPU/media execution.
    root, here = tmp_path/'prepared', tmp_path/'programme'
    root.mkdir(); here.mkdir()
    out = tmp_path/'review'
    old, bound, scene = tmp_path/'prepared.dll', tmp_path/'state.dll', tmp_path/'scene.json'
    old.write_bytes(b'old DLL'); bound.write_bytes(b'state DLL'); scene.write_text('{}')
    np.savez(root/'point_observations.npz', pts=np.array([512, 1024], dtype=np.int64))
    state_path = tmp_path/'state.npz'
    np.savez(state_path, rig=np.zeros((1, 43)),
             native_library_provenance_sha256=np.array(historical._digest(bound)))
    processing = dict(first_pts=512, last_pts=1024, stop_pts_exclusive=1280, native_frame_count=3)
    manifest = dict(shots=[dict(id='s09', row_begin=0, row_stop=2, **processing)],
                    native_library=str(old), native_library_sha256=historical._digest(old),
                    scene_path=str(scene), scene_file_sha256=historical._digest(scene),
                    point_input_sha256=historical._digest(root/'point_observations.npz'))
    scope = dict(required_shot_ids=['s09'], shots=[dict(id='s09', processing=processing)],
                 source_clock=dict(time_base_num=1, time_base_den=12800,
                                   native_pts_step=256, native_fps_num=50, native_fps_den=1))
    historical._write(root/'joint_manifest.json', manifest)
    historical._write(root/'mapping.json', {})
    historical._write(here/'rebuild_scope.json', scope)
    monkeypatch.setattr(historical, 'DATA', tmp_path)
    monkeypatch.setattr(historical, 'HERE', here)
    validated = []
    def load_npz(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    prepared = SimpleNamespace(npz=load_npz,
        validate_snapshot=lambda *args: validated.append(args),
        old=SimpleNamespace(validate_joint_snapshot=lambda *args: pytest.fail('Wrong validator')))
    monkeypatch.setitem(sys.modules, 'prepare_native_multicamera', prepared)
    import rinkspace.domain.parameters as parameters
    import rinkspace.review.render as review_render
    monkeypatch.setattr(parameters, 'SceneParameterModel',
        lambda *a, **kw: SimpleNamespace(initial_state=SimpleNamespace(layout='checked layout')))
    verification = dict(decoded_frames=3, native_PTS_equal=True)
    checked_pts = []
    def verify_video(video, pts, time_base):
        checked_pts.append((np.asarray(pts).tolist(), time_base))
        return verification
    monkeypatch.setattr(review_render, 'validate_video', verify_video)
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        if '--snapshot' in command:
            frozen_state = Path(command[command.index('--snapshot')+1])
            report = dict(shot='s09', video=str(out/'candidate_s09.mp4'), frames=3,
                verification=verification, snapshot_sha256=historical._digest(frozen_state),
                input_sha256=manifest['point_input_sha256'],
                native_library_sha256=historical._digest(bound),
                frames_with_source_observations=2, frames_without_source_observations=1,
                empty_source_pts=[768])
            historical._write(out/'candidate_videos.json', [report])
        else:
            Path(command[-1]).write_bytes(b'mocked concat only')
    monkeypatch.setattr(historical.subprocess, 'run', fake_run)
    record = generic.export(root, state_path, out, 'candidate',
                            allow_sparse_observations=True, native_library=bound)
    assert len(validated) == 1
    assert validated[0][3]['native_library_sha256'] == historical._digest(bound)
    assert commands[0][commands[0].index('--library')+1] == str(bound.resolve())
    assert '--allow-sparse-observations' in commands[0]
    assert commands[1][commands[1].index('-c')+1] == 'copy'
    assert checked_pts == [([512, 768, 1024], historical.TIME_BASE)]
    assert record['source_exposures'] == 3
    assert record['frames_without_source_observations'] == 1
    assert record['observation_coverage'][0]['empty_source_pts'] == [768]
    assert record['native_library_override_explicit'] is True
    assert historical._read(out/'frozen_problem'/'prepared_joint_manifest.json') == manifest
    assert historical._read(out/'frozen_problem'/'joint_manifest.json')['native_library'] == str(bound.resolve())
    assert historical._digest(state_path) == historical._digest(out/'frozen_state.npz')
    assert (out/'concat_selected.txt').read_text() == "file 'candidate_s09.mp4'\n"
