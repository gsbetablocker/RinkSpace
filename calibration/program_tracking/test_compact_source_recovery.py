import json

import numpy as np
import pytest

from calibration.program_tracking.compact_source_json import pack_source_json, unpack_source_json
from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack, normalize_selected_ice_rows)
from calibration.program_tracking.strict_source_recovery import restore_source_admission_journal
from calibration.program_tracking.test_strict_absolute_source import fixture_inputs


def test_large_repeated_exact_evidence_is_shared_and_lossless():
    evidence = dict(source_text='真实原片证据'*100000, same_source_uv=[[1.25,2.5],[3.75,4.]])
    rows = [dict(actual_pts=i, evidence=evidence) for i in range(64)]
    encoded = pack_source_json(rows)
    restored = unpack_source_json(encoded)
    assert restored == rows
    assert restored[0]['evidence'] is restored[-1]['evidence']
    assert encoded.nbytes < len(evidence['source_text'].encode('utf8'))+20000
    assert unpack_source_json(np.asarray(json.dumps(rows[:1],ensure_ascii=False))) == rows[:1]


def test_committed_actual_rows_restore_after_failed_final_save(tmp_path,monkeypatch):
    binding,row,measurement,inventory = fixture_inputs(tmp_path)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    path = tmp_path/'interrupted.npz'
    journal = tmp_path/'source_recovery.sqlite'
    writer = StrictAbsolutePackWriter(path,source=normalized['review'].source,
        expected_actual_pts=[100,200],recovery_journal=journal,legacy_pixel_families=['old_ice'])
    writer.add_row(normalized['rows'][0],normalized['review'])
    writer.mark_exposure_processed(100)
    original_save = np.savez_compressed
    def fail(*args,**kwargs):raise OSError('Simulated interrupted final serialization')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError,match='interrupted'):
        writer.finish()
    assert not path.exists()
    writer.recovery_journal.close()
    monkeypatch.setattr(np,'savez_compressed',original_save)
    resumed = restore_source_admission_journal(journal,tmp_path/'recovered.npz')
    receipt = resumed.finish()
    pack = StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert np.array_equal(pack.values['uv'],normalized['rows'][0]['uv'])
    assert pack.metadata['processed_actual_pts'] == [100]
    assert pack.metadata['all_exposures_processed'] is False
    assert pack.metadata['legacy_family_fallback'] == 'forbidden'
