import json
from pathlib import Path

from calibration.program_tracking import strict_source_dataset_append as common
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_source_fit_dataset import reference
from calibration.program_tracking.strict_source_dataset_partial_delta import append_partial_reviewed_deltas
from test_strict_source_dataset_append import _inputs


def test_partial_original_catalogue_is_not_claimed_as_complete(tmp_path):
    base, _, audit = _inputs(tmp_path)
    partial = reference(tmp_path/'delta/pack.npz')
    packet = json.loads(Path(audit['path']).read_text(encoding='utf8'))
    packet['records'][0]['inputs'][1]['source_reference'] = partial
    Path(audit['path']).write_text(json.dumps(packet), encoding='utf8')
    before = StrictAbsoluteSourcePack(partial['path'], expected_sha256=partial['sha256'])
    assert before.metadata['all_exposures_processed'] is False
    original_reader = common.StrictAbsoluteSourcePack
    result = append_partial_reviewed_deltas(base, [partial], reference(audit['path']), tmp_path/'final')
    after = json.loads(Path(result['dataset_reference']['path']).read_text(encoding='utf8'))
    assert after['expected_actual_pts'] == [100,200,300]
    assert after['actual_exposures_with_finite_fit'] == 2
    view = after['appended_delta_processed_scope_views'][0]
    assert view['original_declared_actual_exposures'] == 3
    assert view['original_all_exposures_processed'] is False
    assert view['local_processed_actual_pts'] == [200]
    assert after['cuts'][1]['source_pack'] == partial
    assert reference(partial['path']) == partial
    assert common.StrictAbsoluteSourcePack is original_reader
    unchanged = StrictAbsoluteSourcePack(partial['path'], expected_sha256=partial['sha256'])
    assert unchanged.metadata == before.metadata
