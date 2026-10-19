"""Regression checks use the original RGB-reviewed packets, not fit residuals."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from calibration.program_tracking.native_circle_source_admission import (
    ReviewedCircleAdmission, reference, install_circle_source_admission)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/source_datasets/s1_wide/c0_gate_1017.json'
POLICY_SHA = 'f6f700ce74e6b912c4f6acd3c393ab612ba5c2a149e6731c38f5dcc7b9f9045f'
SOURCE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014')
FRAMES = (526, 582, 583, 587, 1147, 1148, 1149, 1150, 1151, 1152, 1153)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@pytest.fixture(scope='module')
def gate():
    if not SOURCE.exists():
        pytest.skip('Native source packets are stored in the registered local Data root')
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from rinkspace.storage import AssetStore
    return ReviewedCircleAdmission(dict(path=str(POLICY), sha256=POLICY_SHA),
        resolve=AssetStore(ROOT).resolve)


@pytest.mark.parametrize('frame', FRAMES)
@pytest.mark.parametrize('folder', ('circle_candidates', 'ice_candidates'))
def test_all_reviewed_complete_estimators_stay_out_of_FIT(gate, frame, folder):
    path = SOURCE / folder / f'{frame:06d}.json'
    before_bytes = path.read_bytes()
    packet = json.loads(before_bytes)
    original = deepcopy(packet)
    features = (list(packet['features_by_curve'].values()) if folder == 'circle_candidates'
        else packet['features'])
    failed = next(f for f in features if f['target_id'] == 'ice.C0:outer')
    revised = gate.revise_packet(packet)
    active = (list(revised['features_by_curve'].values()) if folder == 'circle_candidates'
        else revised['features'])
    assert not any(f['target_id'] == 'ice.C0:outer' and f['point_count'] for f in active)
    assert len(revised['inactive_source_estimators']) == 1
    archive = revised['inactive_source_estimators'][0]
    assert archive['complete_original_feature'] == failed
    assert archive['review']['operator_matrix_sha256']
    assert archive['no_correlated_operator_rows_or_columns_sliced'] is True
    assert revised['source'] == packet['source']
    assert revised['source_admission_revision']['native_pts'] == packet['source']['native_pts']
    for feature in features:
        if feature['target_id'] != 'ice.C0:outer':
            assert next(f for f in active if f['target_id'] == feature['target_id']) == feature
    assert packet == original
    assert path.read_bytes() == before_bytes


def test_regenerated_sparse_stations_cannot_clear_physical_denial(gate):
    packet = read(SOURCE / 'circle_candidates/001147.json')
    feature = packet['features_by_curve']['C0:outer']
    feature['source_evidence_group_id'] = 'regenerated_with_different_stations'
    feature['segments'][0]['uv'][0] = [123., 456.]
    feature['segments'][0]['sigma_px'][0] = 17.
    revised = gate.revise_packet(packet)
    assert revised['features_by_curve']['C0:outer']['point_count'] == 0
    assert revised['inactive_source_estimators'][0]['complete_original_feature'] == feature


def test_other_video_target_and_native_clock_are_not_relabelled(gate):
    packet = read(SOURCE / 'circle_candidates/001147.json')
    other_video = deepcopy(packet)
    other_video['source']['source_video_sha256'] = 'a' * 64
    revised = gate.revise_packet(other_video)
    assert revised['features_by_curve'] == other_video['features_by_curve']
    assert not revised['inactive_source_estimators']
    other_pts = deepcopy(packet)
    other_pts['source']['native_pts'] = 549376
    other_pts['source']['programme_frame_index_1based'] = 1146
    assert gate.revise_packet(other_pts)['features_by_curve'] == other_pts['features_by_curve']
    conflict = deepcopy(packet)
    conflict['source']['programme_frame_index_1based'] = 1146
    with pytest.raises(ValueError, match='PTS and programme index'):
        gate.revise_packet(conflict)


def test_cached_and_new_return_paths_use_the_same_admission(gate):
    packet = read(SOURCE / 'circle_candidates/001147.json')
    observed = []
    extractor = SimpleNamespace(rows={1147:dict(source=packet['source'])}, use_circle_cache=True)
    def original(index, native):
        observed.append(extractor.use_circle_cache)
        return deepcopy(packet['features_by_curve']), dict(path='cache' if extractor.use_circle_cache else 'new')
    extractor.nested_house_features = original
    install_circle_source_admission(extractor, gate)
    results = []
    for cached in (True, False):
        extractor.use_circle_cache = cached
        features, diagnostic = extractor.nested_house_features(1147, None)
        results.append(features)
        assert features['C0:outer']['point_count'] == 0
        assert extractor.source_admission_archives[1147][0]['complete_original_feature'] == packet['features_by_curve']['C0:outer']
        final = gate.revise_packet(dict(source=packet['source'], features_by_curve=features),
            additional_archives=extractor.source_admission_archives[1147])
        assert final['source_admission_revision']['complete_estimator_denials_applied'] == 1
    assert observed == [True, False]
    assert results[0] == results[1]


def test_changed_authenticated_source_facts_fail_closed(tmp_path):
    source = dict(event_id='e', programme_id='p', source_video_sha256='b'*64,
        time_base='1/12800', native_pts=256, programme_frame_index_1based=1)
    feature = dict(target_id='ice.C0:outer', point_count=1, segments=[dict(uv=[[1, 2]])],
        source_circle_fit=dict(direct_readouts=[dict(uv=[1, 2])]))
    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return reference(path)
    evidence = put('packet.json', dict(source=source, features=[feature]))
    review = put('review.json', dict(reviewed=True))
    groups = put('groups.json', dict(complete_W=[[1, 2], [3, 4]]))
    fact = put('fact.json', dict(schema='rinkspace.reviewed-complete-circle-estimator-quarantine.v1',
        review_complete=True, authority='agent_rgb_review', user_confirmed=False,
        source_identity_review=review, actual_original_group_export=groups,
        specifications=[dict(frame=1, native_pts=256,
            source_point_ids=['ice.C0:outer/segment/0/vertex/0'],
            operator_matrix_sha256='c'*64, operator_matrix_shape=[2, 2],
            material_identity_confirmed=False, reason='Actual native RGB wrong material',
            evidence_references=[review, evidence])]))
    policy = put('policy.json', dict(schema='rinkspace.native-circle-source-admission.v1',
        source={k:source[k] for k in ('event_id','programme_id','source_video_sha256','time_base')},
        reviewed_quarantine_manifests=[fact]))
    admission = ReviewedCircleAdmission(policy)
    assert admission.revise_packet(dict(source=source, features=[feature]))['point_count'] == 0
    Path(fact['path']).write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='evidence changed'):
        admission.revise_packet(dict(source=source, features=[feature]))
    with pytest.raises(ValueError, match='evidence changed'):
        ReviewedCircleAdmission(policy)


def test_denial_cannot_split_an_undeclared_shared_estimator(gate):
    packet = read(SOURCE / 'circle_candidates/001147.json')
    failed = packet['features_by_curve']['C0:outer']
    other = deepcopy(failed)
    other['target_id'] = 'ice.another:outer'
    packet['features_by_curve']['another:outer'] = other
    with pytest.raises(ValueError, match='split a shared estimator'):
        gate.revise_packet(packet)


def test_revising_a_packet_is_idempotent(gate):
    packet = read(SOURCE / 'ice_candidates/001147.json')
    once = gate.revise_packet(packet)
    assert gate.revise_packet(once) == once
