import json

import numpy as np
import pytest

from calibration.program_tracking import strict_source_dataset_points as points
from calibration.program_tracking.current_source_points import CurrentSourcePoints, save_current_source_point_binding


def source_packet(tmp_path, monkeypatch, *, reported_count=3):
    source = dict(video_sha256='a' * 64, time_base='1/15360', pixel_domain='native1920x1080')
    dataset_path = tmp_path / 'dataset.json'
    dataset_path.write_text(json.dumps(dict(schema=points.SCHEMA, source=source,
        expected_actual_pts=[100, 103], fit_points=reported_count, held_points=1,
        pending_support_points=1, pending_named_points=1)))
    reference = dict(path=str(dataset_path), sha256=points.file_sha(dataset_path))
    camera = tmp_path / 'camera.npz'
    np.savez(camera, pts=np.array([100, 101, 103], np.int64),
        exposure_representative_pts=np.array([100, 100, 103], np.int64),
        metadata_json=json.dumps(dict(source_sha256=source['video_sha256'],
                                     source_time_base=source['time_base'])))
    manifest = tmp_path / 'solver.json'
    manifest.write_text('{}')

    def rows(ref, *, split):
        assert ref == reference and split in ('fit', 'held')
        if split == 'fit':
            yield dict(split='fit', scope='ice', actual_pts=100, point_count=3,
                uv=np.array([[12.25, 30.5], [12.25, 30.5], [52., 71.]]), sigma_px=np.ones(3))
        else:
            yield dict(split='held', scope='board', actual_pts=103, point_count=1,
                uv=np.array([[150.25, 240.5]]), sigma_px=np.array([np.nan]))

    monkeypatch.setattr(points, 'iter_dataset_rows', rows, raising=False)

    def read_once(ref):
        result = {}
        for role in ('fit', 'held'):
            result[role] = [dict(row, sigma=row['sigma_px'], points=row['point_count'],
                source_locators=[dict(group_point_range=[0, row['point_count']])])
                for row in rows(ref, split=role)]
        receipt = dict(schema='rinkspace.single-pass-strict-dataset-read.v1',
            full_source_pack_passes=1, fit_points=reported_count, held_points=1,
            every_original_row_proof_UV_sigma_clock_and_role_checked=True,
            source_pixels_sigma_roles_locators_and_groups_unchanged=True)
        return result['fit'], result['held'], receipt

    monkeypatch.setattr(points, 'load_strict_dataset_rows_once', read_once)
    return reference, camera, manifest


def test_every_current_fit_pixel_and_empty_exposure_survive_binding(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    shared = points.freeze_dataset_source_point_inputs(tmp_path / 'shared.npz', ref,
        input_manifest=manifest, camera_path=camera)
    binding = tmp_path / 'bound.npz'
    save_current_source_point_binding(binding, camera, shared, q_sha256='b' * 64,
                                     input_manifest=manifest)
    view = CurrentSourcePoints(binding, camera, q_sha256='b' * 64)
    np.testing.assert_array_equal(view.points(100), [[12.25, 30.5], [52., 71.]])
    assert view.points(103).shape == (0, 2)
    receipt = view.metadata['family_receipts'][0]
    assert receipt['fit_points'] == 3
    assert receipt['held_points_not_drawn'] == receipt['pending_named_points_not_drawn'] == 1
    assert receipt['display_roles'] == ['fit']


def test_incomplete_fit_consumer_cannot_seal_archive(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch, reported_count=4)
    shared = tmp_path / 'shared.npz'
    with pytest.raises(ValueError, match='Every admitted'):
        points.freeze_dataset_source_point_inputs(shared, ref, input_manifest=manifest,
                                                 camera_path=camera)
    assert not shared.exists()


def test_other_source_camera_cannot_bind_dataset_pixels(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    np.savez(camera, exposure_representative_pts=np.array([100, 103]),
        metadata_json=json.dumps(dict(source_sha256='c' * 64, source_time_base='1/15360')))
    with pytest.raises(ValueError, match='original source/exposure clock'):
        points.freeze_dataset_source_point_inputs(tmp_path / 'shared.npz', ref,
            input_manifest=manifest, camera_path=camera)


def test_explicit_held_display_preserves_missing_sigma_UV_and_separate_row_denominators(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    shared = points.freeze_dataset_source_point_inputs(tmp_path / 'shared.npz', ref,
        input_manifest=manifest, camera_path=camera, include_held=True)
    binding = tmp_path / 'bound.npz'
    save_current_source_point_binding(binding, camera, shared, q_sha256='b' * 64,
                                     input_manifest=manifest)
    view = CurrentSourcePoints(binding, camera, q_sha256='b' * 64)
    np.testing.assert_array_equal(view.points(100), [[12.25, 30.5], [52., 71.]])
    np.testing.assert_array_equal(view.points(103), [[150.25, 240.5]])
    receipt = view.metadata['family_receipts'][0]
    assert receipt['display_roles'] == ['fit', 'held']
    assert receipt['display_original_points_by_role'] == dict(fit=3, held=1)
    assert receipt['displayed_source_rows_by_role'] == dict(fit=1, held=1)
    assert receipt['held_missing_sigma_displayed_points'] == 1
    assert receipt['held_source_sigma_not_filled'] is True
    assert receipt['held_used_in_fit'] is False
    assert receipt['held_points_not_drawn'] == 0
    assert receipt['pending_or_historical_points_included'] is False
    assert shared['original_source_points'] == 4 and shared['points'] == 3


def test_pending_row_cannot_be_smuggled_into_display_held_stream(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    original_reader = points.load_strict_dataset_rows_once

    def wrong_rows(reference):
        fit, held, receipt = original_reader(reference)
        return fit, [dict(row, split='pending') for row in held], receipt

    monkeypatch.setattr(points, 'load_strict_dataset_rows_once', wrong_rows)
    target = tmp_path / 'shared.npz'
    with pytest.raises(ValueError, match='current admitted'):
        points.freeze_dataset_source_point_inputs(target, ref, input_manifest=manifest,
            camera_path=camera, include_held=True)
    assert not target.exists()


def test_every_held_point_must_be_enumerated_independently_of_fit_count(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    original_reader = points.load_strict_dataset_rows_once

    def missing_held(reference):
        fit, held, receipt = original_reader(reference)
        return fit, [], receipt

    monkeypatch.setattr(points, 'load_strict_dataset_rows_once', missing_held)
    target = tmp_path / 'shared.npz'
    with pytest.raises(ValueError, match='Every admitted dataset HELD point'):
        points.freeze_dataset_source_point_inputs(target, ref, input_manifest=manifest,
            camera_path=camera, include_held=True)
    assert not target.exists()


@pytest.mark.parametrize('include_held', [False, True])
def test_selected_roles_share_one_read_without_legacy_proof_expansion_and_keep_original_rows(tmp_path, monkeypatch, include_held):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    original_reader = points.load_strict_dataset_rows_once
    calls = []

    def once(reference):
        calls.append(reference)
        fit, held, receipt = original_reader(reference)
        fit[0]['source_locators'] = [dict(group_point_range=[0, 1]), dict(group_point_range=[1, 3])]
        return fit, held, receipt

    def forbidden(*args, **kwargs):
        raise AssertionError('All-points display must not expand original pixel proofs twice')

    monkeypatch.setattr(points, 'load_strict_dataset_rows_once', once)
    monkeypatch.setattr(points, 'iter_dataset_rows', forbidden, raising=False)
    shared = points.freeze_dataset_source_point_inputs(tmp_path / 'shared.npz', ref,
        input_manifest=manifest, camera_path=camera, include_held=include_held)
    assert calls == [ref]
    with np.load(shared['path'], allow_pickle=False) as archive:
        receipt = json.loads(str(archive['metadata_json'].item()))['family_receipts'][0]
        expected = [[12.25, 30.5], [52., 71.]] + ([[150.25, 240.5]] if include_held else [])
        np.testing.assert_array_equal(archive['uv'], expected)
    assert receipt['displayed_source_rows_by_role'] == (dict(fit=2, held=1) if include_held else dict(fit=2))
    assert receipt['displayed_groups_by_role'] == (dict(fit=1, held=1) if include_held else dict(fit=1))
    assert receipt['held_points_not_drawn'] == (0 if include_held else 1)
    assert receipt['held_missing_sigma_displayed_points'] == (1 if include_held else 0)
    assert receipt['single_pass_source_validation']['full_source_pack_passes'] == 1


def test_compact_groups_need_complete_original_source_row_ranges(tmp_path, monkeypatch):
    ref, camera, manifest = source_packet(tmp_path, monkeypatch)
    original_reader = points.load_strict_dataset_rows_once

    def lost_row(reference):
        fit, held, receipt = original_reader(reference)
        fit[0]['source_locators'] = [dict(group_point_range=[0, 2])]
        return fit, held, receipt

    monkeypatch.setattr(points, 'load_strict_dataset_rows_once', lost_row)
    target = tmp_path / 'shared.npz'
    with pytest.raises(ValueError, match='source row locator coverage'):
        points.freeze_dataset_source_point_inputs(target, ref, input_manifest=manifest,
            camera_path=camera, include_held=True)
    assert not target.exists()


@pytest.mark.parametrize('include_held', [False, True])
def test_real_authenticated_reader_selected_archive_matches_original_role_selection(tmp_path, monkeypatch, include_held):
    from calibration.program_tracking.test_strict_source_disjoint_merge import _fixture_pack
    from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
    from calibration.program_tracking.strict_source_disjoint_merge import merge_disjoint_strict_absolute_source_packs
    from calibration.program_tracking.strict_source_fit_dataset import partition_support, reference
    from calibration.program_tracking.strict_source_fit_dataset_stream import iter_dataset_rows
    from calibration.solver_runtime import strict_dataset_read_runtime as runtime

    fit = _fixture_pack(tmp_path / 'fit')
    held = _fixture_pack(tmp_path / 'held', pts=200, split='held', board=True)
    empty = _fixture_pack(tmp_path / 'empty', pts=300, empty=True)
    merged = merge_disjoint_strict_absolute_source_packs([fit, held, empty], tmp_path / 'merged.npz')
    pack = StrictAbsoluteSourcePack(merged['path'], expected_sha256=merged['sha256'])
    selection = partition_support(pack.rows, pack.values['row_offsets'], pack.values['sigma_px'])
    indices = tmp_path / 'indices.npz'
    np.savez_compressed(indices, row_offsets=pack.values['row_offsets'], **selection)
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps(dict(schema=points.SCHEMA, source=pack.metadata['source'],
        expected_actual_pts=[100, 200, 300], fit_points=2, held_points=3,
        pending_support_points=1, pending_named_points=0,
        cuts=[dict(source_pack=reference(merged['path']), support_index=reference(indices))])), encoding='utf8')
    ref = reference(dataset)
    roles = ('fit', 'held') if include_held else ('fit',)
    original = [row for role in roles for row in iter_dataset_rows(ref, split=role)]
    camera = tmp_path / 'camera.npz'
    np.savez(camera, pts=np.array([100, 101, 200, 300], np.int64),
        exposure_representative_pts=np.array([100, 100, 200, 300], np.int64),
        metadata_json=json.dumps(dict(source_sha256=pack.metadata['source']['video_sha256'],
                                     source_time_base=pack.metadata['source']['time_base'])))
    manifest = tmp_path / 'solver.json'
    manifest.write_text('{}')
    calls = []
    validated_pack = runtime.StrictDatasetValidatedPack

    def observe(*args, **kwargs):
        calls.append(args[0])
        return validated_pack(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Final all-points archive cannot re-expand legacy proof records')

    monkeypatch.setattr(runtime, 'StrictDatasetValidatedPack', observe)
    monkeypatch.setattr(points, 'iter_dataset_rows', forbidden, raising=False)
    shared = points.freeze_dataset_source_point_inputs(tmp_path / 'shared.npz', ref,
        input_manifest=manifest, camera_path=camera, include_held=include_held)
    assert calls == [merged['path']]
    with np.load(shared['path'], allow_pickle=False) as archive:
        for i, stamp in enumerate(archive['actual_pts']):
            incoming = [row['uv'] for row in original if row['actual_pts'] == stamp]
            expected = np.unique(np.concatenate(incoming), axis=0) if incoming else np.empty((0, 2))
            actual = archive['uv'][archive['offsets'][i]:archive['offsets'][i + 1]]
            assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        receipt = json.loads(str(archive['metadata_json'].item()))['family_receipts'][0]
    assert receipt['display_original_points_by_role'] == (dict(fit=2, held=3) if include_held else dict(fit=2))
    assert receipt['displayed_source_rows_by_role'] == (dict(fit=1, held=1) if include_held else dict(fit=1))
    assert receipt['held_missing_sigma_displayed_points'] == (3 if include_held else 0)
    assert receipt['held_points_not_drawn'] == (0 if include_held else 3)
    assert receipt['missing_support_points_not_drawn'] == 1
