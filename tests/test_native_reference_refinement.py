import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix, diags

from calibration.solver_runtime.native_reference_refinement import refine_undamped_native_reference


def test_original_equation_two_independent_routes_and_unchanged_arrays():
    normal = csc_matrix([[4., 1., 0.], [1., 3., 0.], [0., 0., 1.]])
    gradient = np.array([2., -1., 0.]); diagonal = np.maximum(normal.diagonal(), 1.)
    snapshots = [v.copy() for v in (normal.data, gradient, diagonal)]
    equation = normal + diags(1e-8 * diagonal)
    actual = np.linalg.solve(equation.toarray(), -gradient)
    sparse_initial = actual + [.003, -.002, 0.]
    schur_initial = actual + [-.001, .005, 0.]
    replayed = []
    def replay(steps):
        replayed.append(np.array(steps))
        return np.max(abs(np.array(steps)), axis=1)
    gn, reference, maximum, error, report = refine_undamped_native_reference(
        normal, gradient, diagonal, sparse_initial, schur_initial, np.array([0]), np.array([2]),
        native_replay=replay, initial_native_error=.01)
    assert np.max(abs(gn-actual)) < 1e-12
    assert np.max(abs(reference-actual)) < 1e-12
    assert error < 1e-12
    assert gn[2] == reference[2] == 0.
    assert len(replayed) == 1
    assert np.array_equal(replayed[0], np.array([gn, reference-gn]))
    assert report['independent_full_sparse_LU'] is not report['independent_global_Schur']
    assert report['independent_full_sparse_LU']['equation_identity'] == report['independent_global_Schur']['equation_identity']
    assert report['original_ridge_factor'] == 1e-8
    assert report['original_native_reference_error_px'] == .01
    assert len(report['implementation_sources']) == 2
    for entry in report['implementation_sources']:
        assert Path(entry['path']).exists() and len(entry['sha256']) == 64
    for before, after in zip(snapshots, (normal.data, gradient, diagonal)):
        assert before.tobytes() == after.tobytes()


def test_original_diagonal_and_finite_replay_are_required():
    normal = diags([2., 3.]).tocsc(); gradient = np.array([1., 2.])
    kwargs = dict(native_replay=lambda steps: [1., np.nan], initial_native_error=.01)
    with pytest.raises(ValueError, match='Original complete'):
        refine_undamped_native_reference(normal, gradient, [2., 4.], [0., 0.], [0., 0.], [0], [], **kwargs)
    with pytest.raises(ValueError, match='native replay must be finite'):
        refine_undamped_native_reference(normal, gradient, [2., 3.], [0., 0.], [0., 0.], [0], [], **kwargs)


def test_production_refinement_is_failure_only_before_capture_and_cache_release():
    path = Path('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_joint_source_solver.py')
    text = path.read_text(encoding='utf8'); tree = ast.parse(text)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'refine_undamped_native_reference')
    guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'reference_error > 0.001'
               and any(inner is call for inner in ast.walk(node))]
    assert len(guarded) == 1
    assert text.index('corrected_gn,corrected_reference,corrected_gn_step') < text.index('captured=capture_failed_native_reference(')
    assert 'if corrected_error<=1e-3:' in text
    assert "_write_json(output/('native_reference_refinement_%06d.json'%len(history))" in text
    assert 'normal,gradient,diagonal,gn,reference.step,' in text
    assert 'numerical_ridge_diagonal=1e-8*diagonal' in text
