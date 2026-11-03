from types import SimpleNamespace
import numpy as np
from calibration.program_tracking.body_tracking.skate_integration import (
    ContinuousShotWindow,HaloWindow,base_layer_packet,restore_checkpoint_layers,
)


def test_whole_shot_has_no_internal_phase_gate_or_missing_body_dof():
    ref=SimpleNamespace(p={'pts':np.array([round(22.12*12800)])})
    state=ContinuousShotWindow(ref,(22.11,44.97))
    np.testing.assert_array_equal(state.active,np.arange(69))
    times=np.array([22.12,28.12,34.8,35.2,35.6,36.8,37.2,43.8,44.96])-22.12
    np.testing.assert_allclose(state.basis(times).sum(1),1,atol=1e-14)
    # Every retained local coordinate, including toe and wrist joints, can
    # respond at the old artificial core boundary without a frozen prefix.
    controls=np.ones((state.m,state.d))*.01
    np.testing.assert_allclose(state.increments(controls.ravel(),times),.01,atol=1e-14)


def test_one_shot_curve_is_c2_at_interior_knots():
    ref=SimpleNamespace(p={'pts':np.array([round(22.12*12800)])})
    state=ContinuousShotWindow(ref,(22.11,44.97))
    control=np.random.default_rng(5).normal(size=(state.m,state.d))
    knots=np.unique(state.knots[4:-4])
    eps=np.min(np.diff(np.unique(state.knots)))*1e-10
    for order in (0,1,2):
        left=state.basis(knots-eps,order)@control
        right=state.basis(knots+eps,order)@control
        np.testing.assert_allclose(left,right,atol=2e-6,rtol=1e-8)


def test_preserved_candidate_layers_round_trip_without_spline_remapping():
    def reference():
        return SimpleNamespace(p={'pts':np.array([round(22.12*12800)])},new_refinements=[])
    original=reference();old=HaloWindow(original,(28.12,43.8));whole=ContinuousShotWindow(original,(22.11,44.97))
    rng=np.random.default_rng(1103)
    original.new_refinements=[(old,rng.normal(0,.02,old.size)),(whole,rng.normal(0,.001,whole.size))]
    packet=base_layer_packet(original);restored=reference();restore_checkpoint_layers(restored,packet)
    times=np.linspace(-.01,22.85,1001)
    for (before,x),(after,y) in zip(original.new_refinements,restored.new_refinements):
        np.testing.assert_array_equal(x,y)
        for order in (0,1,2):
            np.testing.assert_array_equal(before.basis(times,order),after.basis(times,order))
        np.testing.assert_array_equal(before.increments(x,times),after.increments(y,times))
