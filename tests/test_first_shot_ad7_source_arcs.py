import numpy as np
import pytest

from calibration.program_tracking import first_shot_ad7_source_arcs as arcs
from calibration.program_tracking import first_shot_ad7_source_paths as paths
from calibration.program_tracking import first_shot_disjoint_print as disjoint


def _target(monkeypatch):
    material_ref = dict(path="explicit-material", sha256="material-sha")
    target = dict(kind="ad_print_curve", advertisement_id=7, contour_key="component000",
        curve_topology=dict(schema=disjoint.TOPOLOGY, path_ids=list(paths.SOURCE_PATHS),
            offsets=[0, 2, 4, 6, 8], closed=[True]*4, material_reference=material_ref))
    t = np.linspace(0., 2*np.pi, 185)
    uv = np.c_[np.cos(t), np.sin(t)]
    uv[-1] = uv[0]
    material = dict(contours={"logo_part03":dict(closed=True, material_uv=uv.tolist())})
    monkeypatch.setattr(disjoint, "_CONFIG", dict(manifest=dict(material_override=material_ref),
        templates=dict(advertisements={"7":material})))
    return target


def test_true_exterior_arc_excludes_opposite_side_and_closure(monkeypatch):
    target = arcs.reviewed_target(_target(monkeypatch), "logo_part03")
    topology = disjoint.PathUnion.from_target(target)
    assert topology.offsets == (0, 127)
    assert topology.closed == (False,)
    assert target[arcs.ARC]["vertex_slice"] == [1, 128]
    full = arcs._full_target(target)
    assert full["curve_topology"]["offsets"] == [0, 185]
    assert full["curve_topology"]["closed"] == [True]
    assert arcs.ARC not in full


def test_world_values_and_all_J_columns_select_identical_real_vertices(monkeypatch):
    target = arcs.reviewed_target(_target(monkeypatch), "logo_part03")
    world = np.arange(185*3, dtype=float).reshape(185, 3)**1.2
    J = np.arange(185*3*8, dtype=float).reshape(185, 3, 8)**1.3
    seen = []
    def original_world(scene, full, templates):
        seen.append(full)
        assert full["curve_topology"]["closed"] == [True]
        return world
    def original_J(state, geometry, descriptors):
        assert arcs.ARC not in descriptors[0]
        return (J,)
    monkeypatch.setitem(arcs._ORIGINAL, "feature_world", original_world)
    monkeypatch.setitem(arcs._ORIGINAL, "print_layout_world_jacobians", original_J)
    actual = arcs.feature_world({}, target, {})
    derivative = arcs.print_layout_world_jacobians(object(), np.zeros(1), [target])[0]
    assert np.array_equal(actual, world[1:128])
    assert np.array_equal(derivative, J[1:128])
    assert not np.shares_memory(actual, world)
    assert not np.shares_memory(derivative, J)


def test_forged_closed_arc_or_other_material_vertex_range_is_rejected(monkeypatch):
    target = arcs.reviewed_target(_target(monkeypatch), "logo_part03")
    target[arcs.ARC]["vertex_slice"] = [0, 185]
    with pytest.raises(ValueError, match="true exterior SVG arc"):
        arcs._full_target(target)
