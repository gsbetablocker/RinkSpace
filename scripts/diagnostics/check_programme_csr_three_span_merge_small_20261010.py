"""Compare native CSR storage with the three real original functions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['RINKSPACE_SCATTER_THREADS'] = '3'
import numpy as np
from scipy.sparse import csr_matrix
from calibration.solver_runtime import install
install()
from calibration.solver_runtime import absolute_material_value_runtime as material
from calibration.solver_runtime import camera_absolute_reuse_runtime as camera
from calibration.solver_runtime import programme_csr_three_span_merge_runtime as merge
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock


def equal(left, right):
    assert left.shape == right.shape
    for key in ('data', 'indices', 'indptr'):
        a, b = getattr(left, key), getattr(right, key)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
    assert left.has_sorted_indices == right.has_sorted_indices
    assert left.has_canonical_format == right.has_canonical_format


def original_three(matrix, lens, mount, selected, course, columns, motion_rows):
    result = merge._ORIGINAL_MOUNT(matrix, lens, (34207, 34208, 34209), 34204)
    result = merge._ORIGINAL_MOUNT(result, mount, (34201, 34202, 34203), 34195)
    return merge._ORIGINAL_SELECTED(result, selected, course,
        motion_rows=motion_rows, geometry_columns=columns)


def data_fixture(dtype=np.float64, large_columns=False):
    source_rows, rows = 17, 19
    columns = 2**31+200 if large_columns else 34340
    row_columns = np.array([5, 18, 18, 34213, columns-1], np.int64)
    data = np.tile(np.array([1.25, -0., -3.5, 1e-300, -1e200]), rows).astype(dtype)
    matrix = csr_matrix((data, np.tile(row_columns, rows), np.arange(rows+1)*5), shape=(rows, columns))
    lens = np.tile(np.array([1.5, -0., -.125]), (source_rows, 1))
    mount = np.tile(np.array([0., -.75, 1e-250]), (source_rows, 1))
    selected = np.array([0, 4, 9, 16])
    course = np.array([[.25, 0., -.5, 1e-200], [0., .125, -0., 2.],
                       [-.25, .5, .75, 0.], [1e100, -.5, -0., 1.]])
    course_columns = np.array([34217, 1, 34214, 4])
    return matrix, lens, mount, selected, course, course_columns, source_rows


@contextmanager
def direct_scope():
    owner = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    owner.name = 'small_original_CSR_calls'
    owner.optics_chart = SimpleNamespace(indices=(34207,34208,34209), mount_indices=(34201,34202,34203))
    scope = dict(owner=owner, phase=0, pending=None, merged=False, fallback=None,
                 selected_motion_rows=0, result_rows=0, result_nnz=0)
    token = merge._SCOPE.set(scope)
    try: yield owner, scope
    finally: merge._SCOPE.reset(token)


def actual_three(fixture, motion_rows=None):
    matrix,lens,mount,selected,course,columns,source_rows = fixture
    with direct_scope() as (owner, scope):
        result = merge._insert_mount(matrix, lens, owner.optics_chart.indices, 34204)
        result = merge._insert_mount(result, mount, owner.optics_chart.mount_indices, 34195)
        result = merge._insert_selected(result, selected, course,
            motion_rows=source_rows if motion_rows is None else motion_rows, geometry_columns=columns)
        return result, scope['merged'], scope['fallback']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    material._install_runtime()
    camera.install_camera_absolute_reuse_runtime()
    current_evaluate = StructuredProgrammeBlock._evaluate
    assert current_evaluate.__code__.co_freevars
    receipt = merge.install_programme_csr_three_span_merge_runtime()
    assert receipt['native_backend']['available']
    cases = []
    for name, fixture in [('i32', data_fixture()), ('i64', data_fixture(large_columns=True)),
                          ('unsupported_float32', data_fixture(np.float32))]:
        reference = original_three(*fixture[:-1], fixture[-1])
        actual, merged, reason = actual_three(fixture)
        equal(actual, reference)
        assert merged == (name != 'unsupported_float32')
        cases.append(dict(case=name, storage_bytes_equal=True, native_merged=merged, fallback_reason=reason))

    # Exercise the actual registered _evaluate body and installed outer
    # material wrapper with numeric span producers only, without a source graph.
    matrix,lens,mount,selected,course,columns,n = data_fixture()
    owner = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    owner.name, owner.groups, owner.geometry = 'small_registered_programme', [], SimpleNamespace()
    owner.optics_chart = SimpleNamespace(indices=(34207,34208,34209), mount_indices=(34201,34202,34203), physical_steps=())
    owner.bundle = SimpleNamespace(source_residual_rows=n, motion_rows=n, profile_weights=None,
        motion_sigma=np.ones(n), current_calibration=lambda local:SimpleNamespace())
    owner.evaluate_camera_geometry = lambda q,jacobian:(np.zeros(matrix.shape[0]), matrix)
    owner.local = lambda q:np.zeros(4)
    owner.geometry_values = lambda q:np.zeros(0)
    owner.motion_columns_provider = lambda *a,**k:lens.copy()
    owner.course_motion_columns = lambda local,cal:(selected,course)
    owner.course_current_motion_geometry_parameter_indices = columns
    globals_ = __import__('inspect').unwrap(current_evaluate).__globals__
    saved_motion = globals_['motion_mount_columns']
    globals_['motion_mount_columns'] = lambda *a,**k:mount.copy()
    try:
        reference = current_evaluate(owner, np.zeros(1), True)
        actual = StructuredProgrammeBlock._evaluate(owner, np.zeros(1), True)
        assert actual[0].tobytes() == reference[0].tobytes()
        equal(actual[1], reference[1])
        assert hasattr(owner, '_rinkspace_last_material_value_scope')
    finally:
        globals_['motion_mount_columns'] = saved_motion

    failures=[]
    for name, outside in [('course_overlap',False), ('overlap_precedes_outside_matrix',True)]:
        values=list(data_fixture())
        if outside:
            values[3] = np.array([0, values[0].shape[0]+2])
            values[4] = np.ones((2,4))
            values[5] = np.array([5,1,34214,4])
        else:
            values[5] = np.array([34207,1,34214,4])
        motion_rows = values[0].shape[0]+5 if outside else values[-1]
        errors=[]
        for action in (lambda:original_three(*values[:-1],motion_rows),
                       lambda:actual_three(values,motion_rows)):
            try: action()
            except Exception as error: errors.append((type(error), error.args))
            else: raise AssertionError('Original expected error omitted')
        assert errors[0] == errors[1]
        assert errors[0][0] is ValueError
        assert errors[0][1] == ('Selected finite geometry entries must be inserted exactly once',)
        failures.append(name)
    report=dict(schema='rinkspace.small-original-three-CSR-spans-merge-check.v1',
        cases=cases, actual_registered_body_and_existing_material_wrapper_passed=True,
        original_first_error_type_and_args_equal=failures,
        no_full_source_graph_process_pool_or_complete_solver_run=True,
        helper_sha256=hashlib.sha256(Path(merge.__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(merge.SOURCE.read_bytes()).hexdigest(),
        seconds=time.perf_counter()-started)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
