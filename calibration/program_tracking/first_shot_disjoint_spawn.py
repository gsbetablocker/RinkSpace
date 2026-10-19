"""Explicit spawn adapter for a frozen disjoint-print successor.

The frozen successor captures the native raw function in a partial, then
replaces that function's module attribute. Python cannot pickle the captured
function after that replacement. This adapter sends a stable public entry
point instead; the fresh child imports and calls the original native function.
No residual, row clock, whitening, retention proof or source equation changes.
"""
from copy import deepcopy
from functools import partial
import hashlib
from pathlib import Path
import pickle
from types import MethodType


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def raw_chunk(task):
    """Run the true original raw producer imported by this fresh child."""
    import native_transverse_bundle as native
    from . import first_shot_disjoint_print as disjoint
    original = native.raw_chunk
    if (getattr(original, '__module__', None) != native.__name__
            or getattr(original, '__name__', None) != 'raw_chunk'
            or Path(original.__code__.co_filename).resolve() != Path(native.__file__).resolve()):
        raise RuntimeError('Fresh spawn must import the original native raw producer')
    values, receipt = disjoint.raw_chunk(task, original_sampler=original)
    return values, dict(receipt, disjoint_spawn_adapter=reference(__file__),
        original_native_raw_reimported_in_child=True)


def install(programme, *, entry_reference=None):
    """After frozen disjoint install; before any source/geometry pool opens."""
    import native_transverse_bundle as native
    from . import first_shot_disjoint_print as disjoint
    from . import first_shot_transverse_execution as context
    if (programme.pool is not None or programme.bundle._source_pool is not None
            or programme.bundle._geometry_pool is not None):
        raise ValueError('Install explicit spawn adapter before pools open')
    if hasattr(programme, 'disjoint_spawn_receipt'):
        raise ValueError('Disjoint spawn adapter already installed')
    if not hasattr(programme, 'raw_retention_receipt') or not hasattr(programme, 'disjoint_print_receipt'):
        raise ValueError('Original raw retention and explicit material successor must be installed')
    captured = native.raw_chunk
    if not isinstance(captured, partial) or captured.func is not disjoint.raw_chunk or captured.args:
        raise ValueError('Expected frozen disjoint raw partial before adapter')
    original = captured.keywords.get('original_sampler')
    if (set(captured.keywords) != {'original_sampler'}
            or getattr(original, '__module__', None) != native.__name__
            or getattr(original, '__name__', None) != 'raw_chunk'
            or Path(original.__code__.co_filename).resolve() != Path(native.__file__).resolve()):
        raise ValueError('Frozen disjoint partial does not retain original native raw')
    # Every other captured worker entry must independently remain importable.
    # Do not turn a later unrelated pickle failure into a silent substitution.
    for function in (native.init_worker, native.source_chunk,
            context._init_context_worker, context._context_sample, raw_chunk):
        pickle.dumps(function)
    producers = [reference(__file__)]
    if entry_reference is not None:
        if reference(entry_reference['path']) != entry_reference:
            raise ValueError('Explicit spawn adapter entry changed')
        producers.append(deepcopy(entry_reference))
    for producer in producers:
        if producer not in programme.source_producers:
            programme.source_producers.append(deepcopy(producer))
    receipt = dict(schema='rinkspace.disjoint-print-native-raw-spawn-adapter.v1',
        producer=producers[0], entry=deepcopy(entry_reference),
        frozen_material_runtime=reference(disjoint.__file__),
        original_native_raw_producer=reference(original.__code__.co_filename),
        child_imports_original_native_raw=True, captured_native_function_not_serialized=True,
        source_and_geometry_other_worker_entries_pickle_preflight_passed=True,
        original_raw_retention_shadow_and_bitwise_verification_unchanged=True,
        source_residual_J_W_row_clock_and_guard_unchanged=True)
    previous_identity = programme.checkpoint_identity
    def identity(adapter):
        return dict(previous_identity(), disjoint_spawn_execution=deepcopy(receipt))
    native.raw_chunk = raw_chunk
    programme.checkpoint_identity = MethodType(identity, programme)
    programme.disjoint_spawn_receipt = receipt
    return deepcopy(receipt)
