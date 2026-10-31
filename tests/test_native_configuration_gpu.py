"""Synthetic state/forward checks through an explicit real CPU/CUDA library."""
import os

import numpy as np
import pytest

from rinkspace.core.gpu import ResidentResidualProblem

pytestmark = pytest.mark.skipif(
    not (os.environ.get('RINKSPACE_G2_LIBRARY') and os.environ.get('RINKSPACE_G3_LIBRARY')),
    reason='explicit native CUDA library required',
)


class HiddenSymbol:
    """Inject a missing ABI capability; all other calls reach the real DLL."""
    def __init__(self, library, name):
        self.library, self.name = library, name

    def __getattr__(self, name):
        if name == self.name:
            raise AttributeError(name)
        return getattr(self.library, name)


def snapshot(problem):
    # These reads are available before compiling a factor plan or attaching
    # a solver, so the configuration under test remains legitimately mutable.
    result = problem.export_model()
    result.update({'profile_'+key: value for key, value in problem.resolve_profile_roots('cpu').items()})
    if hasattr(problem, 'scene_parameter_count'):
        result['scene_values'] = problem.export_scene_parameters()
    for backend in ('cpu', 'cuda'):
        result.update({backend+'_'+key: value for key, value in problem.evaluate(backend).items()
                       if key in ('values', 'status')})
    return result


def assert_same_state_forward(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize('error', ['mode', 'wrapped_zero', 'wrapped_none',
                                  'fractional_ids', 'missing_setter'])
def test_profile_error_keeps_native_state_and_allows_same_handle_retry(error):
    from test_gpu_profile_queries import ProfileQueriesGPU
    fixture = ProfileQueriesGPU()
    fixture.setUp()
    with fixture.f.make() as problem:
        before = snapshot(problem)
        bad = fixture.profile.copy()
        library = problem._lib
        if error == 'mode':
            bad['mode'] = 'typo'
        elif error.startswith('wrapped'):
            ids = bad['row_ids'].astype(np.uint64)
            ids[0] = 2**32 if error == 'wrapped_zero' else 2**32-1
            bad['row_ids'] = ids
        elif error == 'fractional_ids':
            bad['row_ids'] = bad['row_ids']+.5
        else:
            problem._lib = HiddenSymbol(library, 'rs_gpu_set_profile_mode')
        with pytest.raises((ValueError, AttributeError)):
            problem.configure_profiles(**bad)
        problem._lib = library
        assert_same_state_forward(snapshot(problem), before)
        assert not hasattr(problem, 'profile_count')
        problem.configure_profiles(**fixture.profile)
        actual = snapshot(problem)
        for backend in ('cpu', 'cuda'):
            roots = problem.resolve_profile_roots(backend)
            np.testing.assert_array_equal(roots['status'], 0)
            np.testing.assert_allclose(roots['station'], fixture.base_kw['stations'], atol=1e-12)
    with fixture.f.make() as baseline:
        baseline.configure_profiles(**fixture.profile)
        assert_same_state_forward(actual, snapshot(baseline))


def scene_problem(fixture):
    state = fixture.state
    problem = ResidentResidualProblem(fixture.core,
        **{**fixture.kw, 'board': state[:6], 'layout': state[6:41]})
    paths = []
    for code in (100, 410):
        kind, parameters = fixture.core.shared_geometry(state[:6], state[6:41], code, options=0)
        paths.append(dict(kind=kind, parameters=parameters))
    paths.append(dict(kind=7, parameters=[1.]))
    paths.extend(dict(kind=5, parameters=[.81, .005, .03, .12, .008],
                      vertices=fixture.data[:10]) for _ in range(3))
    problem.configure_geometry(paths)
    return problem


@pytest.mark.parametrize('error', ['chart_shape', 'fractional_chart', 'wrapped_chart',
    'chart_domain', 'ordered_chart', 'fractional_record', 'wrapped_record', 'missing_setter'])
def test_scene_error_keeps_native_state_and_allows_same_handle_retry(error):
    from test_gpu_joint_scene import JointSceneGPU
    fixture = JointSceneGPU()
    fixture.setUp()
    # Exact integer float metadata remains accepted by the real boundary.
    valid = dict(values=fixture.state, records=fixture.records.astype(float),
        data=fixture.data, directions=np.zeros((len(fixture.state), 1)),
        charts=np.zeros(len(fixture.state)))
    with scene_problem(fixture) as problem:
        before = snapshot(problem)
        bad = valid.copy()
        library = problem._lib
        if error == 'chart_shape':
            bad['charts'] = bad['charts'][:-1]
        elif error == 'fractional_chart':
            bad['charts'] = bad['charts'].copy();bad['charts'][0] = .5
        elif error == 'wrapped_chart':
            bad['charts'] = bad['charts'].astype(np.uint64);bad['charts'][0] = 2**32
        elif error == 'chart_domain':
            bad['charts'] = bad['charts'].copy();bad['charts'][0] = 2
        elif error == 'ordered_chart':
            bad['charts'] = bad['charts'].copy();bad['charts'][11:14] = [3, 4, 0]
        elif error == 'fractional_record':
            bad['records'] = bad['records']+.5
        elif error == 'wrapped_record':
            bad['records'] = bad['records'].astype(np.int64);bad['records'][0, 0] = 2**32
        else:
            problem._lib = HiddenSymbol(library, 'rs_gpu_set_scene_chart')
        with pytest.raises((ValueError, AttributeError)):
            problem.configure_scene_parameters(**bad)
        problem._lib = library
        assert_same_state_forward(snapshot(problem), before)
        assert not hasattr(problem, 'scene_parameter_count')
        problem.configure_scene_parameters(**valid)
        actual = snapshot(problem)
    with scene_problem(fixture) as baseline:
        baseline.configure_scene_parameters(**valid)
        assert_same_state_forward(actual, snapshot(baseline))
        np.testing.assert_array_equal(actual['scene_values'], fixture.state)
        np.testing.assert_array_equal(actual['cuda_status'], 0)
        np.testing.assert_allclose(actual['cpu_values'], actual['cuda_values'], atol=2e-10)
