from copy import deepcopy

import pytest
import numpy as np

from calibration.program_tracking.strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack, normalize_selected_ice_rows
from calibration.program_tracking.strict_source_pack_merge import merge_strict_absolute_source_packs
from calibration.program_tracking.test_strict_absolute_source import fixture_inputs


def test_board_local_clearance_dictionary_keeps_exact_remaining_anchor_indices():
    from calibration.program_tracking.strict_source_pack_merge import _slice_row
    original=dict(uv=np.asarray([[1.,2.],[3.,4.],[5.,6.]]),original_point_indices=[8,9,10],
        candidate_local_source_clearance={'8':{'actual':8},'9':{'actual':9},'10':{'actual':10}})
    result=_slice_row(original,[0,2])
    assert result['candidate_local_source_clearance']=={'8':{'actual':8},'10':{'actual':10}}
    assert original['candidate_local_source_clearance']['9']=={'actual':9}


def _pack(directory, *, pts=100, split='fit', shift=0.):
    directory.mkdir()
    binding, row, measurement, inventory = fixture_inputs(directory)
    binding.update(actual_pts=pts, native_pts=pts+10)
    measurement['source_binding'] = deepcopy(binding)
    measurement['candidate_sides'] += shift
    measurement['candidate_uv'] += shift
    row['actual_pts'] = pts
    row['temporal_evidence']['current_actual_pts'] = pts
    normalized = normalize_selected_ice_rows([row], measurement,
        source_binding=binding, curve_inventory=inventory, split=split)
    writer = StrictAbsolutePackWriter(directory/'pack.npz', source=normalized['review'].source,
        expected_actual_pts=[100, 200], legacy_pixel_families=['ice', 'ads'])
    writer.add_row(normalized['rows'][0], normalized['review'])
    writer.exclusions.append(dict(actual_pts=pts, reason='source_competing_material_pending',
        original_candidate_indices=[7, 19], source_pixel_identity_preserved=True))
    writer.mark_exposure_processed(pts)
    return writer.finish()


def test_actual_partial_clocks_merge_but_explicit_held_never_becomes_fit(tmp_path):
    first = _pack(tmp_path/'first')
    held_delta = _pack(tmp_path/'held_delta', split='held')
    later = _pack(tmp_path/'later', pts=200)
    receipt = merge_strict_absolute_source_packs([first, held_delta, later], tmp_path/'merged.npz')
    result = StrictAbsoluteSourcePack(receipt['path'], expected_sha256=receipt['sha256'])
    assert receipt['points'] == 2
    assert receipt['exact_duplicate_points_removed'] == 1
    assert result.metadata['all_exposures_processed'] is True
    assert result.metadata['all_exposures_exact_native_BGR_bound'] is True
    assert [r['actual_pts'] for r in result.fit_rows] == [200]
    assert [r['actual_pts'] for r in result.held_rows] == [100]
    assert result.metadata['legacy_family_fallback'] == 'forbidden'
    assert result.metadata['source_measurement_processing_is_not_manual_visual_completion'] is True
    assert len(result.metadata['original_references']) == 3
    inputs = [StrictAbsoluteSourcePack(r['path'], expected_sha256=r['sha256'])
        for r in (first, held_delta, later)]
    assert result.exclusions == [item for pack in inputs for item in pack.exclusions]
    assert receipt['original_input_exclusions_preserved'] is True


def test_changed_exact_source_candidate_is_not_hidden_as_duplicate(tmp_path):
    first = _pack(tmp_path/'first')
    conflict = _pack(tmp_path/'conflict', shift=1.)
    with pytest.raises(ValueError, match='competing actual UV'):
        merge_strict_absolute_source_packs([first, conflict], tmp_path/'rejected.npz')
    assert not (tmp_path/'rejected.npz').exists()
