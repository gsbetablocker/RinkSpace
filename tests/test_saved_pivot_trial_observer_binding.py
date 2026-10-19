"""The saved-direction observer reaches the actual installed main namespace."""
from types import FunctionType, SimpleNamespace

import pytest

from scripts.diagnostics.run_saved_pivot_trial_comparison import bind_live_driver_solver_observer


def test_cloned_main_reaches_observer_then_same_original_solver():
    seen = []
    def original(value):
        seen.append(('original', value))
        return value + 1
    namespace = {'solve_joint_source': original}
    def template(value):
        return solve_joint_source(value)
    main = FunctionType(template.__code__, namespace)
    driver = SimpleNamespace(main=main, solve_joint_source=original)
    def observer(value):
        seen.append(('observer', value))
        return original(value)
    receipt = bind_live_driver_solver_observer(driver, original, observer)
    assert driver.main(3) == 4
    assert seen == [('observer', 3), ('original', 3)]
    assert namespace['solve_joint_source'] is driver.solve_joint_source is observer
    assert receipt['live_fresh_J_main_namespace_bound']


def test_mismatched_live_binding_is_fatal_and_not_replaced():
    def original(value):
        return value
    namespace = {'solve_joint_source': lambda value: value}
    def template(value):
        return solve_joint_source(value)
    driver = SimpleNamespace(main=FunctionType(template.__code__, namespace), solve_joint_source=original)
    previous = dict(namespace)
    with pytest.raises(RuntimeError, match='bindings differ'):
        bind_live_driver_solver_observer(driver, original, lambda value: value)
    assert namespace == previous
    assert driver.solve_joint_source is original
