"""Portable report engineering fixtures, never real rink validation."""
from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import av
import numpy as np
import pytest

from rinkspace.io import dump_json, load_json, sha256
from rinkspace.model_report import build_model_delivery


@pytest.fixture
def records(tmp_path):
    model_dir, run = tmp_path / 'model', tmp_path / 'run'
    run.mkdir()
    video = run / 'source.mp4'
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 320, 180, 'yuv420p'
        for index in range(4):
            image = np.full((180, 320, 3), 40 + index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        proxy = [{'proxy_pts': frame.pts, 'proxy_time_base': str(frame.time_base)} for frame in container.decode(video=0)]
    rows = []
    for index, point in enumerate(proxy):
        tb = Fraction(point['proxy_time_base'])
        source_pts = int(Fraction(10) / tb) + point['proxy_pts']
        point.update(source_pts=source_pts, source_time_base=str(tb))
        rows.append({'source_pts': source_pts, 'source_time_base': str(tb),
            'source_time_s': float(source_pts * tb), 'source_frame_index': 250 + index,
            'status': 'unavailable' if index == 2 else 'candidate',
            'H_model_from_image': None if index == 2 else [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'anchor_id': None if index == 2 else 'fixture-reference',
            'support_polygon_image': [[0, 50], [640, 50], [640, 360], [0, 360]],
            'person_boxes_image': [[280, 80, 340, 260]],
            'image_domain': 'raw_px_distortion_unchecked'})
    model = {'model_id': 'synthetic-ui-fixture', 'revision': 'v0', 'geometry_level': 'projective',
        'unit': 'reference_px', 'status': 'synthetic_fixture_not_real_rink', 'grid': {
            'bounds': [0, 0, 640, 360], 'x_values': [0, 160, 320, 480, 640], 'y_values': [0, 90, 180, 270, 360]},
        'sources': [{'source_id': 'fixture', 'local_path': 'data/input/never-copy.mp4'}],
        'reference_views': [{'id': 'fixture-reference', 'image_path': 'missing.jpg',
                            'source_id': 'fixture', 'pts': 500, 'time_base': '1/50', 'source_time_s': 10}],
        'landmarks': [{'physical_id': 'F1', 'label': '</script><script>wrong()</script>', 'model_xy': [160, 180]}]}
    dump_json(model_dir / 'rink_model.json', model)
    dump_json(run / 'localization.json', {'run_id': 'fixture-run', 'label': '合成界面测试，非真实冰场',
        'model_id': model['model_id'], 'model_revision': model['revision'],
        'source_sha256': 'a' * 64, 'image_width': 640, 'image_height': 360,
        'selected_window': {'selection_basis': 'Complete four-frame disposable engineering fixture'}, 'frames': rows})
    dump_json(run / 'source_pts_map.json', {'source_sha256': 'a' * 64, 'proxy_sha256': sha256(video), 'frames': proxy})
    (run / 'unlisted-original.mp4').write_bytes(b'PRIVATE ORIGINAL FIXTURE')
    return model_dir, run, tmp_path / 'review'


def test_portable_report_preserves_all_rows_and_exact_source_model(records):
    model, run, out = records
    before = (run / 'localization.json').read_bytes()
    result = build_model_delivery(model, [run], out)
    assert result['programmes'] == 1
    assert result['missing_reference_images'] == ['missing.jpg']
    manifest = load_json(out / 'review_manifest.json')
    rows = load_json(out / 'programmes/01/localization.json')['frames']
    assert len(rows) == 4 and rows[2]['H_model_from_image'] is None
    assert rows[2]['status'] == 'unavailable'
    assert all(row['model_revision'] == 'v0' and row['source_sha256'] == 'a' * 64 for row in rows)
    assert all(row['pts'] == row['source_pts'] for row in rows)
    assert (run / 'localization.json').read_bytes() == before
    assert (out / 'model/rink_model.json').read_bytes() == (model / 'rink_model.json').read_bytes()
    assert not any('unlisted' in path for path in manifest['files'])
    assert manifest['programmes'][0]['selected_window']['frame_count'] == 4
    page = (out / 'index.html').read_text(encoding='utf-8')
    assert '<script>wrong()' not in page
    embedded = page.split('<script id="review-data" type="application/json">')[1].split('</script>')[0]
    data = json.loads(embedded)
    assert data['model']['landmarks'][0]['label'] == '</script><script>wrong()</script>'
    assert data['runs'][0]['frames'][0]['media_time_s'] == 0
    assert data['runs'][0]['frames'][0]['source_time_s'] == 10
    spec = importlib.util.spec_from_file_location('export_model_review', Path(__file__).parents[1] / 'scripts/export_model_delivery.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    _, checks = exporter.validate_run(out, decode=True)
    assert checks['proxy_decoded_and_pts_verified']
    assert checks['programmes'][0]['status_counts']['unavailable'] == 1
    assert not checks['independent_rink_accuracy_validated']


def test_report_rejects_proxy_replacement_and_original_media(records):
    model, run, out = records
    (run / 'source.mp4').write_bytes(b'replaced')
    with pytest.raises(ValueError, match='Proxy SHA-256'):
        build_model_delivery(model, [run], out)
    mapping = load_json(run / 'source_pts_map.json')
    record = load_json(run / 'localization.json')
    mapping['source_sha256'] = record['source_sha256'] = sha256(run / 'source.mp4')
    dump_json(run / 'source_pts_map.json', mapping)
    dump_json(run / 'localization.json', record)
    with pytest.raises(ValueError, match='Original source media'):
        build_model_delivery(model, [run], out)


def test_compact_html_keeps_complete_linked_diagnostics_pts_and_drawable_masks(records):
    model,run,out = records
    original = load_json(run/'localization.json')
    for index,row in enumerate(original['frames']):
        row['foreground_config'] = {'margin_px': 18}
        row['foreground_polygons_image'] = [[[10,20],[30,20],[30,90],[10,90]]]
        row['error_summary'] = {'maximum_curve_p90_px': index+.25}
        row['reobservation_attempts'] = [{'anchor_id': 'fixture-reference', 'status': 'unavailable',
            'reason': 'fixture_refusal_kept', 'camera_proposal_inliers': 7, 'proposal_role': 'camera_only',
            'initial_affine_normal_sensitivity': {'large_diagnostic_array': list(range(50))},
            'initial_curve_residual': {'p90_px': 17.5}}]
    dump_json(run/'localization.json',original)
    before = (run/'localization.json').read_bytes()
    build_model_delivery(model,[run],out)
    assert (run/'localization.json').read_bytes() == before
    complete = load_json(out/'programmes/01/localization.json')
    embedded = (out/'index.html').read_text(encoding='utf-8').split('<script id="review-data" type="application/json">')[1].split('</script>')[0]
    display = json.loads(embedded)['runs'][0]
    assert display['frames_file'] == 'programmes/01/localization.json'
    assert display['foreground_configs'] == [{'margin_px':18}]
    assert len(display['frames']) == len(complete['frames']) == len(original['frames'])
    for before_row,full,shown in zip(original['frames'],complete['frames'],display['frames']):
        assert full['reobservation_attempts'] == before_row['reobservation_attempts']
        assert full['foreground_polygons_image'] == before_row['foreground_polygons_image']
        for key in ('source_pts','source_time_base','status','H_model_from_image','anchor_id','error_summary','source_sha256','model_id','model_revision'):
            assert shown[key] == full[key]
        assert shown['reobservation_attempts'][0]['reason'] == 'fixture_refusal_kept'
        assert 'initial_curve_residual' not in shown['reobservation_attempts'][0]
        if shown['status']=='unavailable':
            assert shown['H_model_from_image'] is None
            assert 'foreground_polygons_image' not in shown and 'person_boxes_image' not in shown
        else:
            assert shown['foreground_polygons_image'] == full['foreground_polygons_image']
            assert shown['person_boxes_image'] == full['person_boxes_image']
    assert 'run.frames_file' in (out/'report.js').read_text(encoding='utf-8')


@pytest.mark.parametrize('change', ['frame_count', 'timing', 'model', 'selection'])
def test_report_rejects_incomplete_or_incompatible_inputs(records, change):
    model, run, out = records
    record = load_json(run / 'localization.json')
    if change == 'frame_count':
        record['frames'].pop()
    elif change == 'timing':
        record['frames'][1]['source_pts'] += 1
    elif change == 'model':
        record['frames'][1]['model_revision'] = 'different'
    else:
        record['selected_window'] = {}
    dump_json(run / 'localization.json', record)
    with pytest.raises(ValueError):
        build_model_delivery(model, [run], out)
