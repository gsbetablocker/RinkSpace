import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rinkspace.camera_viewer import CameraCatalog, _has_boundary_issue, make_server


@pytest.fixture
def root(tmp_path):
    (tmp_path / 'clip.mp4').write_bytes(b'0123456789')
    (tmp_path / 'unreferenced.txt').write_text('private unrelated file')
    output = tmp_path / 'outputs/camera_classification'
    output.mkdir(parents=True)
    rows = [
        {'shot_id': 's:0', 'source_id': 's', 'start_seconds': 0, 'end_seconds': 3,
         'boundary_end': {'status': 'hard_cut', 'uncertainty_interval_seconds': [2.96, 3]},
         'classification': {'status': 'supported', 'physical_camera_id': 'camera:a', 'installation_id': 'install:a'},
         'content': {'kind': 'rink', 'replay_status': 'unresolved'}},
        {'shot_id': 's:1', 'source_id': 's', 'start_seconds': 3, 'end_seconds': 6,
         'boundary_end': {'status': 'unresolved_transition'},
         'classification': {'status': 'multiple_candidates', 'candidate_installation_ids': ['install:a', 'install:b'],
                            'appearance_group_ids': ['appearance:a']},
         'content': {'kind': 'rink', 'replay_status': 'replay'}, 'duplicate_links': [{'target_shot_id': 's:0'}]},
        {'shot_id': 't:0', 'source_id': 't', 'start_seconds': 0, 'end_seconds': 10,
         'classification': {'status': 'unknown'}, 'content': {'kind': 'interview'}}]
    catalog = {'shots': rows, 'sources': [{'source_id': 's', 'track': {'source_path': 'clip.mp4'}}],
               'installations': [{'installation_id': 'install:a', 'physical_camera_id': 'camera:a'},
                                 {'installation_id': 'install:b', 'physical_camera_id': 'camera:b'}],
               'validation_path': 'outputs/camera_classification/validation.json'}
    (output / 'catalog.json').write_text(json.dumps(catalog), encoding='utf-8')
    (output / 'validation.json').write_text(json.dumps({'automatic': 'passed', 'visual': 'not_reviewed'}))
    return tmp_path


def test_catalog_filters_keep_candidates_and_boundary_uncertainty_distinct(root):
    catalog = CameraCatalog(root)
    result = catalog.query({'installation_id': 'install:a'})
    assert result['total'] == 2
    assert result['duration_seconds'] == 6
    physical = catalog.query({'physical_camera_id': 'camera:a'})
    assert physical['total'] == 2
    assert physical['shots'][1]['classification']['status'] == 'multiple_candidates'
    assert catalog.query({'physical_camera_id': 'camera:b'})['shots'][0]['shot_id'] == 's:1'
    assert catalog.query({'special': 'boundary'})['shots'][0]['shot_id'] == 's:1'
    assert catalog.query({'special': 'boundary'})['total'] == 1
    assert catalog.query({'special': 'replay'})['total'] == 1
    assert catalog.query({'special': 'replay_unknown'})['total'] == 2
    assert catalog.query({'source_id': 's', 'at_seconds': '3'})['shots'][0]['shot_id'] == 's:1'
    assert catalog.query({'appearance_group_id': 'appearance:a'})['total'] == 1
    assert catalog.query({'source_id': 's', 'offset': '1', 'limit': '1'})['shots'][0]['shot_id'] == 's:1'
    assert catalog.metadata()['viewer']['status_counts']['unknown']['duration_seconds'] == 10


def test_http_range_and_allowlist(root):
    server = make_server(root, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        request = Request(url + '/api/file?path=clip.mp4', headers={'Range': 'bytes=3-5'})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == 'bytes 3-5/10'
            assert response.read() == b'345'
        with urlopen(Request(url + '/api/file?path=clip.mp4', headers={'Range': 'bytes=-2'})) as response:
            assert response.read() == b'89'
        with urlopen(Request(url + '/api/file?path=clip.mp4', method='HEAD')) as response:
            assert response.headers['Content-Length'] == '10'
            assert response.read() == b''
        for path in ('unreferenced.txt', '../outside.txt', 'outputs/camera_classification/../../unreferenced.txt'):
            with pytest.raises(HTTPError) as error:
                urlopen(url + '/api/file?path=' + path)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + '/api/file?path=clip.mp4', headers={'Range': 'bytes=20-30'}))
        assert error.value.code == 416
        with urlopen(url + '/api/validation') as response:
            assert json.load(response)['visual'] == 'not_reviewed'
        with urlopen(url + '/api/timeline?source_id=s') as response:
            assert len(json.load(response)) == 2
        with urlopen(url + '/api/references') as response:
            assert json.load(response)['references'] == []
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_reference_manifest_reload_preserves_old_pts_image_access(root):
    catalog = CameraCatalog(root)
    image = root / 'reviewed_old_pts.jpg'
    image.write_bytes(b'reviewed image evidence')
    reference_path = root / 'outputs/camera_classification/reference_evidence.json'
    reference_path.write_text(json.dumps({'references': [{
        'reference_id': 's:review:1',
        'observation': {'image_path': image.name, 'source_pts': 1234, 'source_id': 's'},
    }]}), encoding='utf-8')
    catalog.reload()
    assert catalog.references[0]['observation']['source_pts'] == 1234
    assert catalog.file(image.name) == image
    with pytest.raises(FileNotFoundError):
        catalog.file('unreferenced.txt')


def test_explicit_unresolved_intervals_are_visible_without_fabricated_cut_boundaries():
    assert _has_boundary_issue({'interval_type': 'boundary_unresolved', 'boundary_status': 'unresolved_range'})
    assert _has_boundary_issue({'interval_type': 'transition', 'boundary_status': 'unresolved_range'})
    assert not _has_boundary_issue({'interval_type': 'shot', 'boundary_status': 'after_unresolved_interval'})
    assert not _has_boundary_issue({'boundary_start': {'status': 'machine_supported_cut', 'uncertainty_interval_seconds': [1, 1.033]}})
