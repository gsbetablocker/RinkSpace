import gc
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from calibration.solver_runtime.prepared_oob_cache_runtime import PreparedOutOfBandCache


def test_original_pickle_arrays_alias_readonly_noncontiguous_and_COW_isolation(tmp_path):
    base = np.arange(48, dtype=np.float64).reshape(6,8)
    readonly = np.arange(9, dtype=np.int16)
    readonly.flags.writeable = False
    original = dict(array=base, alias=base, noncontiguous=base[::-1,::2],
        fortran=np.asfortranarray(base), readonly=readonly, empty=np.empty((0,3)),
        numeric=np.arange(7, dtype='>u4'))
    path = tmp_path/'original_prepared.pickle'
    path.write_bytes(pickle.dumps(original, protocol=5))
    baseline = pickle.loads(path.read_bytes())
    controller = PreparedOutOfBandCache(tmp_path/'cache', {'original_fixture_producer': 'unchanged'})
    first = controller.load(path)
    assert controller.events[-1]['natural_cache_saved']
    receipt_path = next(controller.directory.rglob('prepared_oob_receipt.json'))
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    assert receipt['buffer_count'] >= 5
    assert all(row['offset'] % 64 == 0 for row in receipt['buffers'])
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in receipt_path.parent.iterdir()}
    left = controller.load(path)
    assert controller.events[-1]['action'] == 'authenticated_OOB_hit'
    gc.collect()  # Only original ndarray buffers own the mmap lifetime.
    right = controller.load(path)
    assert left['array'] is left['alias'] and right['array'] is right['alias']
    for key, expected in baseline.items():
        for actual in (first[key], left[key], right[key]):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.strides == expected.strides and actual.flags.writeable == expected.flags.writeable
            assert actual.flags.c_contiguous == expected.flags.c_contiguous
            assert actual.flags.f_contiguous == expected.flags.f_contiguous
            assert actual.flags.aligned == expected.flags.aligned
            assert actual.tobytes(order='A') == expected.tobytes(order='A')
    assert np.shares_memory(left['array'], left['noncontiguous']) == np.shares_memory(baseline['array'], baseline['noncontiguous'])
    left['array'][0,0] = -91.
    assert left['alias'][0,0] == -91.
    assert right['array'][0,0] == baseline['array'][0,0] == 0.
    assert np.shares_memory(left['array'], right['array']) is False
    after = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in receipt_path.parent.iterdir()}
    assert before == after and path.read_bytes() == pickle.dumps(original, protocol=5)
    # A bad derived artifact returns the original load; an invalid original
    # still raises its original unpickle error and is never hidden by fallback.
    metadata = receipt_path.parent/'metadata.pickle'
    metadata.write_bytes(b'bad-derived-metadata')
    fallback = controller.load(path)
    assert controller.events[-1]['action'] == 'original_prepared_load_miss'
    assert np.array_equal(fallback['array'], baseline['array'])
    path.write_bytes(b'not-a-pickle')
    try:
        controller.load(path)
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError('Original corrupt pickle failure was swallowed')
