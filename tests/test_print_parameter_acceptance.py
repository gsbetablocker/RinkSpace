"""Current advertisement station regressions; geometry evaluation only."""
import os
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.print_assets import PrintBindingError, evaluate_print_path
from rinkspace.domain.scene_assets import snapshot_scene


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    source = Path(__file__).resolve().parents[1]/'events/ankara_2026/scene/field/model.json'
    root = tmp_path_factory.mktemp('print_acceptance')/'scene'
    scene, _ = snapshot_scene(source, root)
    return SceneParameterModel(scene, root/'scene.json')


@pytest.fixture(scope='module')
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit RINKSPACE_CORE_LIBRARY required')
    return NativeCore(library)


def test_every_localized_path_accepts_original_station_domain_in_both_modes(model, core):
    # The 201-point grid previously rejected 42 current instance paths, even
    # though the default review grid passed. Include the exact audit example
    # and representable values just inside each endpoint as well.
    stations = np.unique(np.r_[np.linspace(0., 1., 201), .035,
                               np.nextafter(0., 1.), np.nextafter(1., 0.)])
    checked, unlocalized = set(), set()
    state = model.initial_state
    for feature, (assets, instance_id, path_id) in model.paths.items():
        placement = model.dependencies(feature)[0]
        if state[placement] is None:
            unlocalized.add(instance_id)
            continue
        results = []
        for jacobian in (False, True):
            try:
                result = evaluate_print_path(assets, instance_id, path_id,
                    model.scene['board'], core, stations,
                    parameter_state=state, jacobian=jacobian)
            except PrintBindingError as exc:
                pytest.fail(f'{feature}, jacobian={jacobian}: {exc}, {exc.context}')
            assert not result['status'].any(), feature
            for key in ('material_uv', 'sz', 'xyz'):
                assert np.isfinite(result[key]).all(), (feature, key)
            if jacobian:
                for block, values in result['jacobians_by_block'].items():
                    assert np.isfinite(values).all(), (feature, block)
                assert np.isfinite(result['station_jacobian']).all(), feature
            results.append(result)
        for key in ('material_uv', 'sz', 'xyz', 'status'):
            np.testing.assert_array_equal(results[0][key], results[1][key],
                                          err_msg=feature+'/'+key)
        checked.add(feature)
    assert 'print_instance/Ad1::Ad1.print.ALL_L1' in checked
    assert 'print_instance/Ad14::Ad1.print.ALL_L1' in checked
    assert unlocalized == {'print_instance/Ad15', 'print_instance/Ad16'}
    assert len(checked) == sum(instance not in unlocalized
                              for _, instance, _ in model.paths.values())


@pytest.mark.parametrize('jacobian', [False, True])
def test_roundoff_fix_does_not_admit_outside_original_station_domain(model, core, jacobian):
    feature = 'print_instance/Ad1::Ad1.print.ALL_L1'
    for station in (np.nextafter(0., -np.inf), np.nextafter(1., np.inf)):
        with pytest.raises(PrintBindingError, match='native_print_path_domain_failure') as raised:
            model.evaluate_print(feature, [station], core, jacobian=jacobian)
        assert raised.value.context['status'] == 3
