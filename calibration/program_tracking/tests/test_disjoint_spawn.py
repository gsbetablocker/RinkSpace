"""Regress old native-function identity loss across a spawn pickle boundary."""
from functools import partial
import pickle
import sys
from types import ModuleType, SimpleNamespace
import pytest
from calibration.program_tracking import first_shot_disjoint_print as core
from calibration.program_tracking import first_shot_disjoint_spawn as adapter
from calibration.program_tracking import first_shot_transverse_execution as context


def _entry(task):
    return task


def test_native_raw_capture_is_unpicklable_but_explicit_adapter_is_stable(monkeypatch):
    native = ModuleType('native_transverse_bundle')
    native.__file__ = __file__
    namespace = {'__name__':native.__name__}
    exec(compile('def raw_chunk(task):\n    return task, {"original": True}\n', __file__, 'exec'), namespace)
    original = namespace['raw_chunk']
    native.raw_chunk = original
    native.init_worker = native.source_chunk = _entry
    monkeypatch.setitem(sys.modules, native.__name__, native)
    native.raw_chunk = partial(core.raw_chunk, original_sampler=original)
    with pytest.raises(pickle.PicklingError, match='not the same object'):
        pickle.dumps(native.raw_chunk)
    other = (native.init_worker,native.source_chunk,context._init_context_worker,context._context_sample)
    programme = SimpleNamespace(pool=None, bundle=SimpleNamespace(_source_pool=None,_geometry_pool=None),
        raw_retention_receipt={},disjoint_print_receipt={},source_producers=[],checkpoint_identity=lambda:{'old':True})
    receipt = adapter.install(programme)
    assert pickle.loads(pickle.dumps(native.raw_chunk)) is adapter.raw_chunk
    assert other == (native.init_worker,native.source_chunk,context._init_context_worker,context._context_sample)
    assert programme.checkpoint_identity()['old'] is True
    assert receipt['original_raw_retention_shadow_and_bitwise_verification_unchanged'] is True
    # A fresh child sees its untouched imported native producer.
    native.raw_chunk = original
    monkeypatch.setattr(core, '_CONFIG', {'receipt':{'disjoint':True}})
    values, proof = adapter.raw_chunk([1.,2.])
    assert values == [1.,2.]
    assert proof['original'] is True and proof['original_native_raw_reimported_in_child'] is True
