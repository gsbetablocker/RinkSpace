"""Regress the disconnected material target at the raw projection boundary."""
import numpy as np
import pytest
from collections import OrderedDict
from types import SimpleNamespace
from calibration.program_tracking import first_shot_disjoint_print as disjoint
from calibration.program_tracking.first_shot_source_kernel import evaluate_projected_targets


def test_disjoint_target_rejects_fake_gap_bridge_and_has_correct_world_J():
    camera = np.zeros(25)
    camera[6:8] = np.log([500., 500.])
    camera[8:10] = [320., 240.]
    world = np.array([[-1., -1., 10.], [-1., 1., 10.], [1., 1., 10.], [1., -1., 10.]])
    target = dict(kind=disjoint.KIND, advertisement_id=7, contour_key="test_union",
        curve_topology=dict(schema=disjoint.TOPOLOGY, path_ids=["left", "right"],
            offsets=[0, 2, 4], closed=[False, False]))
    row = dict(target=target, _target_key="union", point_correspondence=False,
        uv=np.array([[320., 290.], [278., 246.]]))
    D = np.zeros(world.shape+(1,)); D[:, 0, 0] = 1.
    r, _, J = disjoint._evaluate_disjoint(camera, [row], {"union":world}, {"union":D}, jacobian=True)
    old = evaluate_projected_targets(camera, [row], {"union":world}, jacobian=False)[0]
    assert np.linalg.norm(old[:2]) == 0.
    assert np.linalg.norm(r[:2]) == 50.
    # Use the second point, away from a tie between physical components.
    h = 1e-6
    plus = disjoint._evaluate_disjoint(camera, [row], {"union":world+h*D[...,0]}, None, jacobian=False)[0]
    minus = disjoint._evaluate_disjoint(camera, [row], {"union":world-h*D[...,0]}, None, jacobian=False)[0]
    np.testing.assert_allclose(J[2:,0], ((plus-minus)/(2*h))[2:], atol=1e-7, rtol=1e-7)


def test_historical_template_context_restores_successor_after_exception(monkeypatch):
    from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
    old = dict(advertisements={"7":dict(advertisement_number=7,
        contours={"old_component":dict(advertisement_id=7, material_uv=[[.2,.2],[.4,.4]], closed=False)})})
    material = dict(advertisement_number=7,
        contours={"real_path":dict(advertisement_id=7, material_uv=[[.2,.2],[.4,.4]], closed=False)},
        curve_aliases={"old_component":["real_path"]})
    monkeypatch.setattr(disjoint, "_CONFIG", dict(reference={"test":"material-successor"},
        ids=(7,), templates=dict(advertisements={"7":material})))
    state = FirstShotPhysicalState.__new__(FirstShotPhysicalState)
    state.templates = old
    state._world_cache = OrderedDict()
    disjoint.install_material_state(state)
    successor = state.templates
    programme = SimpleNamespace(state=state, pool=None, transverse_world_cache={}, guard_cache={},
        bundle=SimpleNamespace(state=state, _source_pool=None, _geometry_pool=None,
            _world_cache={}, _finite_coverage_cache={}),
        disjoint_print_receipt=dict(source_target_phase="pending_original_source_delta_authentication"))
    with pytest.raises(RuntimeError, match="authentication failed"):
        with disjoint.historical_source_templates(programme):
            assert state.templates is old
            state._world_cache["historical"] = "must not survive"
            raise RuntimeError("authentication failed")
    assert state.templates is successor
    assert not state._world_cache
    assert programme._disjoint_historical_template_depth == 0
