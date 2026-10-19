import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix


def fixture_cache():
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal
    from calibration.solver_runtime.source_loss import SourceLossWeights
    q = np.array([1., 2., 3., 4.])
    blocks = [
        SourceRowBlock(np.array([2., -4.]), csr_matrix([[1., 2., 0., 0.], [0., 3., -2., 0.]]),
                       np.array([False, True]), np.array([1., 3.]),
                       dict(family='original', original_indices=np.array([4, 7]))),
        SourceRowBlock(np.array([1., -2.]), csr_matrix([[0., 0., 2., 1.], [4., 0., 0., 1.]]),
                       np.array([False, False]), np.array([2., 4.]), dict(family='weighted'),
                       SourceLossWeights([.2, 2.], [.5, 1.4]))]
    normal = accumulate_source_normal(lambda: iter(blocks), len(q), expected_row_count=4,
        frozen_profile_token='actual-fixture-profile', retain_native_replay=True, source_state=q)
    return q, normal, blocks


def capture(tmp_path, *, lagged):
    from calibration.solver_runtime.failed_native_reference_capture import capture_failed_native_reference
    q, accumulated, blocks = fixture_cache()
    gn = np.array([.1, -.2, .3, -.4]); reference = gn + [.001, -.002, .003, -.004]
    receipt = capture_failed_native_reference(tmp_path, q=q, normal=accumulated.normal,
        gradient=accumulated.gradient, diagonal=np.maximum(accumulated.normal.diagonal(), 1.),
        gn=gn, reference_step=reference, shared_indices=[0, 2], fixed_indices=[],
        cache=accumulated.native_replay_cache,
        metadata=dict(derivative_model='lagged_J' if lagged else 'fresh_source_J',
                      lagged_J=lagged, original_threshold_native_px=.001))
    return Path(receipt['path']), q, accumulated, blocks, gn, reference


@pytest.mark.parametrize('lagged', [False, True])
def test_real_frozen_native_cache_capture_preserves_every_array_and_replays_original_gate(tmp_path, lagged):
    from calibration.solver_runtime.failed_native_reference_capture import (
        load_failed_native_reference, iter_failed_native_reference_blocks, replay_failed_native_reference)
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import replay_native_step_maxima
    path, q, accumulated, blocks, gn, reference = capture(tmp_path, lagged=lagged)
    metadata, numeric = load_failed_native_reference(path)
    assert metadata['lagged_J'] is lagged
    assert np.array_equal(numeric['q'], q)
    assert np.array_equal(numeric['gradient'], accumulated.gradient)
    assert np.array_equal(numeric['normal'].data, accumulated.normal.data)
    restored = list(iter_failed_native_reference_blocks(path))
    for original, loaded in zip(blocks, restored):
        for name in ('r', 'quadratic', 'scales'):
            a, b = np.asarray(getattr(original, name)), np.asarray(getattr(loaded, name))
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        for name in ('data', 'indices', 'indptr'):
            a, b = getattr(original.J, name), getattr(loaded.J, name)
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    expected = replay_native_step_maxima(lambda: iter(blocks), np.stack((gn, reference-gn)),
        accumulated.contract, frozen_profile_token='actual-fixture-profile')
    actual = replay_failed_native_reference(path)
    assert actual['maximum_GN_native_step_px'] == expected[0]
    assert actual['reference_error_native_px'] == expected[1]
    assert actual['derivative_model'] == ('lagged_J' if lagged else 'fresh_source_J')


@pytest.mark.parametrize('name', ['numerical_state.npz', 'source_blocks/001.npz', 'source_blocks/000.ledger.json'])
def test_offline_arrays_and_source_ledger_must_retain_original_file_hash(tmp_path, name):
    from calibration.solver_runtime.failed_native_reference_capture import load_failed_native_reference
    path, *_ = capture(tmp_path, lagged=True)
    with (path / name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_failed_native_reference(path)


def test_registered_solver_capture_is_before_cache_release_and_uses_defined_iteration():
    path = Path('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_joint_source_solver.py')
    tree = ast.parse(path.read_text(encoding='utf8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'capture_failed_native_reference']
    assert len(calls) == 1
    call = calls[0]
    metadata = next(keyword.value for keyword in call.keywords if keyword.arg == 'metadata')
    iteration = next(keyword.value for keyword in metadata.keywords if keyword.arg == 'iteration')
    assert ast.unparse(iteration) == 'len(history)'
    cache = next(keyword.value for keyword in call.keywords if keyword.arg == 'cache')
    assert ast.unparse(cache) == 'accumulated.native_replay_cache'
    guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Compare)
               and ast.unparse(node.test.left) == 'reference_error'
               and isinstance(node.test.ops[0], ast.Gt)
               and node.test.comparators[0].value == .001
               and any(inner is call for inner in ast.walk(node))]
    assert len(guarded) == 1
    owners = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
              and any(inner is call for body in node.body for inner in ast.walk(body))
              and any(isinstance(inner, ast.Assign)
                      and ast.unparse(inner.targets[0]) == 'accumulated.native_replay_cache'
                      and ast.unparse(inner.value) == 'None'
                      for body in node.finalbody for inner in ast.walk(body))]
    assert len(owners) == 1
