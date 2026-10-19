"""Freeze and source-window semantics with disposable files, never real video."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rinkspace import model_evaluation
from rinkspace.io import dump_json, load_json, sha256


@pytest.fixture
def frozen_case(tmp_path, monkeypatch):
    root = tmp_path
    media = root / 'data/input/heldout.mp4'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'disposable source-byte identity fixture, not playable video')
    for name in model_evaluation.CORE_FILES:
        path = root / 'src/rinkspace' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# disposable frozen code: ' + name, encoding='utf-8')
    model_path = root / 'outputs/model/rink_model.json'
    image_path = model_path.parent / 'views/anchor.jpg'
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b'disposable reference JPEG hash fixture')
    model = {'model_id': 'fixture', 'revision': 'v0', 'reference_views': [
        {'id': 'reference', 'source_sha256': 'a' * 64, 'image_path': 'views/anchor.jpg'}]}
    dump_json(model_path, model)
    window = {'source_id': 'heldout', 'start_s': 10.0, 'end_s': 20.0, 'cuts': [15.5],
              'role': 'new_source_inference_after_freeze', 'reason': 'Entire synthetic shot selected before inference'}
    windows = root / 'configs/rink_model/evaluation_windows.json'
    dump_json(windows, {'selected_before_full_localization': True, 'windows': [window]})
    source_digest = sha256(media)
    monkeypatch.setattr(model_evaluation, 'load_index', lambda root, source: {'video_id': source_digest})
    monkeypatch.setattr(model_evaluation, 'source_path', lambda root, source: media)
    lock_path = root / 'outputs/evaluation_lock.json'
    lock = model_evaluation.freeze_evaluation(root, model_path, windows, lock_path)
    return root, model_path, windows, media, image_path, lock_path, lock


def check(case, *, source='heldout', start=10.0, end=20.0, cuts=(15.5,)):
    root, model, _, _, _, lock, _ = case
    return model_evaluation.check_evaluation(root, lock, model, source, start, end, cuts)


def test_freeze_checks_exact_model_code_source_and_window_without_accuracy_promotion(frozen_case):
    result = check(frozen_case)
    assert result['model_and_code_checked_before_inference']
    assert not result['independent_geometry_validation']
    assert frozen_case[-1]['evaluation_outputs_seen_at_freeze'] is False
    assert set(frozen_case[-1]['core_file_sha256']) == {'src/rinkspace/' + name for name in model_evaluation.CORE_FILES}


@pytest.mark.parametrize('target,match', [('model', 'model bytes'), ('code', 'code changed'), ('source', 'source bytes')])
def test_tampering_is_rejected_before_inference(frozen_case, target, match):
    root, model, _, media, _, _, _ = frozen_case
    path = {'model': model, 'code': root / 'src/rinkspace' / model_evaluation.CORE_FILES[0], 'source': media}[target]
    path.write_bytes(path.read_bytes() + b'\nchanged after freeze')
    with pytest.raises(ValueError, match=match):
        check(frozen_case)


@pytest.mark.parametrize('change', [dict(start=10.02), dict(end=19.98), dict(cuts=()),
                                    dict(cuts=(15.52,)), dict(source='different_upload')])
def test_changed_window_cut_or_source_cannot_reuse_frozen_acceptance(frozen_case, change):
    with pytest.raises(ValueError, match='Window or cuts differ'):
        check(frozen_case, **change)


def test_same_source_as_model_reference_is_not_a_new_source_holdout(frozen_case):
    root, model, windows, media, _, _, _ = frozen_case
    data = load_json(model)
    data['reference_views'][0]['source_sha256'] = sha256(media)
    dump_json(model, data)
    with pytest.raises(ValueError, match='already occurs in model'):
        model_evaluation.freeze_evaluation(root, model, windows, root / 'outputs/invalid_lock.json')


def test_index_source_mismatch_rejects_freeze(frozen_case, monkeypatch):
    root, model, windows, _, _, _, _ = frozen_case
    monkeypatch.setattr(model_evaluation, 'load_index', lambda root, source: {'video_id': '0' * 64})
    with pytest.raises(ValueError, match='changed since indexing'):
        model_evaluation.freeze_evaluation(root, model, windows, root / 'outputs/invalid_lock.json')


def test_development_windows_cannot_be_relabelled_implicitly_as_new_holdout(frozen_case):
    root, model, windows, _, _, _, _ = frozen_case
    data = load_json(windows)
    data['windows'][0]['role'] = 'development_regression'
    dump_json(windows, data)
    with pytest.raises(ValueError, match='No preselected new-source'):
        model_evaluation.freeze_evaluation(root, model, windows, root / 'outputs/invalid_lock.json')


def test_duplicate_locked_window_is_ambiguous_not_double_counted(frozen_case):
    _, _, _, _, _, lock_path, lock = frozen_case
    lock['windows'].append(copy.deepcopy(lock['windows'][0]))
    dump_json(lock_path, lock)
    with pytest.raises(ValueError, match='Window or cuts differ'):
        check(frozen_case)


def test_reference_image_tampering_must_invalidate_frozen_model_input(frozen_case):
    _, _, _, _, image_path, _, lock = frozen_case
    assert lock['reference_asset_sha256']
    image_path.write_bytes(image_path.read_bytes() + b'changed model template pixels')
    with pytest.raises(ValueError, match='[Rr]eference|[Aa]sset|[Ii]mage'):
        check(frozen_case)


@pytest.mark.parametrize('config', [
    {'refinement_motion_model': 'similarity', 'reobserve_every': 10},
    {'refinement_motion_model': 'affine', 'reobserve_every': 5},
    {'refinement_motion_model': 'similarity', 'reobserve_every': 5},
])
def test_changed_runtime_motion_model_or_frequency_cannot_reuse_frozen_evaluation(frozen_case, config):
    root, model, _, _, _, lock, _ = frozen_case
    with pytest.raises(ValueError, match='runtime configuration changed'):
        model_evaluation.check_evaluation(root, lock, model, 'heldout', 10., 20., (15.5,), runtime_config=config)


def test_explicit_similarity_configuration_is_frozen_and_checked(frozen_case):
    root, model, windows, _, _, _, _ = frozen_case
    config = {'refinement_motion_model': 'similarity', 'reobserve_every': 5}
    path = root / 'outputs/similarity_lock.json'
    lock = model_evaluation.freeze_evaluation(root, model, windows, path, runtime_config=config)
    assert lock['runtime_config'] == config
    result = model_evaluation.check_evaluation(root, path, model, 'heldout', 10., 20., (15.5,), runtime_config=config)
    assert result['model_and_code_checked_before_inference']
    assert not result['independent_geometry_validation']
    with pytest.raises(ValueError, match='runtime configuration changed'):
        model_evaluation.check_evaluation(root, path, model, 'heldout', 10., 20., (15.5,))
