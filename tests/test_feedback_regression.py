"""Regression bookkeeping checks; these fixtures establish no visual accuracy."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import pytest

from rink_feedback.core import Store
import rinkspace.feedback_regression as regression


SOURCE = dict(video_sha256='a' * 64, pts=621312, time_base='1/12800', coordinate_domain='native_pixels')


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    database = tmp_path / 'feedback.sqlite'

    @contextmanager
    def store_for_test(project, event_id):
        with Store(database) as store:
            yield store

    monkeypatch.setattr(regression, 'feedback_store', store_for_test)
    monkeypatch.setattr(regression, 'verify_manual_regressions', lambda *args: {'test_fixture_only': True})
    monkeypatch.setattr(regression, 'replay_source_installation_outbox', lambda *args: {'test_fixture_only': True})

    def add_case(case_id, subjects, membership, *, missing_finding=False, review_source=None):
        evidence = _json(tmp_path / case_id / 'review.json', dict(
            source_binding=review_source or SOURCE,
            physical_membership_rows=[dict(source_row_id=row_id, status=status, current_uv=[1., 2.])
                                      for row_id, status in membership]))
        with Store(database) as store:
            if missing_finding:
                finding_id = 'finding_not_in_derived_index'
            else:
                finding = store.record_finding(dict(source=SOURCE, event_id='test_event',
                    signal_type='operator_conflict', entity_ids=['board.contact'],
                    observation_ids=subjects, evidence_refs=[evidence]))
                finding_id = finding['finding_id']
            store.upsert_case(dict(case_id=case_id, version=1, status='active', kind='recipe',
                title='Fixture ordered material recheck', scope=dict(event_id='test_event',
                    source_video_sha256=SOURCE['video_sha256'], entity_ids=['board.contact']),
                tags=['material_boundary'], assertions=dict(episode_source=SOURCE,
                    lesson=dict(reader_check='ordered_board_material_flanks'),
                    verified_dimensions=['measurement_operator'],
                    unresolved_dimensions=['physical_instance', 'position', 'visibility']),
                provenance=dict(finding_id=finding_id), evidence_refs=[evidence]))
        return finding_id

    def check():
        return regression.verify_feedback_regressions('test_event', project=tmp_path,
                                                     output=tmp_path / 'report.json')

    return add_case, check, database


def test_partial_original_subject_coverage_cannot_pass(isolated_ledger):
    add_case, check, _ = isolated_ledger
    add_case('partial', ['actual_geometry:chunk:s02::native/a', 'actual_geometry:chunk:s02::native/b'],
             [('native/a', 'wrong_role')])
    row = check()['learned_procedure_regression'][0]
    assert row['status'] == 'unknown_original_subject_or_expected_check'
    assert row['original_subject_coverage'] == dict(expected_native_ids=['native/a', 'native/b'],
                                                   matched_native_ids=['native/a'], complete=False)


@pytest.mark.parametrize('second_status,expected', [
    ('wrong_role', 'passed_original_wrong_role_preserved'),
    ('unresolved', 'unknown_original_subject_or_expected_check'),
])
def test_complete_coverage_also_requires_every_subject_wrong_role(isolated_ledger, second_status, expected):
    add_case, check, _ = isolated_ledger
    add_case('complete', ['actual_geometry:chunk:s02::native/a', 'native/b'],
             [('native/a', 'wrong_role'), ('native/b', second_status)])
    row = check()['learned_procedure_regression'][0]
    assert row['status'] == expected
    assert row['original_subject_coverage']['complete'] is True


def test_missing_finding_remains_unknown_and_does_not_abort_other_cases(isolated_ledger):
    add_case, check, database = isolated_ledger
    add_case('missing', ['native/a'], [('native/a', 'wrong_role')], missing_finding=True)
    add_case('intact', ['native/b'], [('native/b', 'wrong_role')])
    rows = {r['case_id']: r for r in check()['learned_procedure_regression']}
    assert rows['missing']['status'] == 'unknown_missing_finding'
    assert rows['missing']['finding_history_available'] is False
    assert rows['missing']['finding_history_reconstructed'] is False
    assert 'immutable discovery history' in rows['missing']['missing_history_reason']
    assert rows['intact']['status'] == 'passed_original_wrong_role_preserved'
    with Store(database) as store:
        assert store.db.execute('SELECT COUNT(*) FROM findings').fetchone()[0] == 1
        assert store.db.execute('SELECT COUNT(*) FROM uses').fetchone()[0] == 0


def test_same_video_wrong_native_frame_is_rejected(isolated_ledger):
    add_case, check, _ = isolated_ledger
    add_case('wrong_frame', ['native/a'], [('native/a', 'wrong_role')],
             review_source=dict(SOURCE, pts=621568))
    with pytest.raises(ValueError, match='different native source/frame'):
        check()
