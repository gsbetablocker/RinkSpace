"""Source-reviewed exterior lower-S arcs, through the same physical print bank.

The closed lower-S ink piece has two opposite sides. These observations were
verified on its exterior bottom curve, so the upper side is not an eligible
closest point. Every parameter remains a curve query, with free position along
this explicit real SVG arc; no per-frame material point is introduced.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import partial
import json
from types import MethodType

import numpy as np

from . import first_shot_ad7_source_paths as base
from .first_shot_source_transition import read, reference

SCHEMA = "rinkspace.source-reviewed-Ad7-exterior-arcs.v1"
ARC = "source_reviewed_material_arc"
VERTEX_SLICE = (1, 128)
BASE_VERTEX_COUNT = 185
_ORIGINAL = {}
_INSTALLED = False


def _full_target(target):
    if ARC not in target:
        return target
    from .first_shot_disjoint_print import PathUnion
    arc = target[ARC]
    topology = PathUnion.from_target(target)
    if (arc.get("schema") != SCHEMA or arc.get("base_path_id") != "logo_part03"
            or arc.get("base_vertex_count") != BASE_VERTEX_COUNT
            or tuple(arc.get("vertex_slice", [])) != VERTEX_SLICE
            or arc.get("source_boundary_side") != "exterior_bottom_arc"
            or topology.ids != ("logo_part03",)
            or topology.offsets != (0, VERTEX_SLICE[1]-VERTEX_SLICE[0])
            or topology.closed != (False,)):
        raise ValueError("Explicit source-reviewed true exterior SVG arc required")
    result = deepcopy(target)
    del result[ARC]
    result["curve_topology"] = dict(result["curve_topology"],
        offsets=[0, BASE_VERTEX_COUNT], closed=[True])
    return result


def reviewed_target(target, true_path_id):
    full = base.reviewed_target(target, true_path_id)
    if full["curve_topology"]["offsets"][-1] != BASE_VERTEX_COUNT:
        raise ValueError("Reviewed SVG path inventory changed")
    result = deepcopy(full)
    result["curve_topology"] = dict(result["curve_topology"],
        offsets=[0, VERTEX_SLICE[1]-VERTEX_SLICE[0]], closed=[False])
    result[ARC] = dict(schema=SCHEMA, base_path_id="logo_part03",
        base_vertex_count=BASE_VERTEX_COUNT, vertex_slice=list(VERTEX_SLICE),
        source_boundary_side="exterior_bottom_arc")
    _full_target(result)
    return result


def feature_world(scene, target_descriptor, templates=None):
    full = _full_target(target_descriptor)
    world = _ORIGINAL["feature_world"](scene, full, templates)
    if ARC not in target_descriptor:
        return world
    if len(world) != BASE_VERTEX_COUNT:
        raise ValueError("Actual same-model full path inventory changed")
    return world[slice(*VERTEX_SLICE)].copy()


def print_layout_world_jacobians(self, geometry, descriptors):
    full = [_full_target(target) for target in descriptors]
    result = _ORIGINAL["print_layout_world_jacobians"](self, geometry, full)
    return tuple(value[slice(*VERTEX_SLICE)].copy() if ARC in target else value
        for target, value in zip(descriptors, result, strict=True))


def install_execution():
    """Process-local opt-in after original/disjoint/grouped authentication."""
    global _INSTALLED
    if _INSTALLED:
        return reference(__file__)
    from . import framewise_model
    from . import first_shot_disjoint_print as disjoint
    from .first_shot_physical_state import FirstShotPhysicalState
    if disjoint._CONFIG is None:
        raise ValueError("Original disjoint material execution must precede true arcs")
    _ORIGINAL["feature_world"] = framewise_model.feature_world
    _ORIGINAL["print_layout_world_jacobians"] = FirstShotPhysicalState.print_layout_world_jacobians
    framewise_model.feature_world = feature_world
    FirstShotPhysicalState.print_layout_world_jacobians = print_layout_world_jacobians
    _INSTALLED = True
    return reference(__file__)


def worker_init(*args, original_initializer):
    original_initializer(*args)
    install_execution()


def source_chunk(task, *, original_sampler):
    values, jacobian, receipt = original_sampler(task)
    if not _INSTALLED:
        raise ValueError("Source worker did not install reviewed true arcs")
    return values, jacobian, dict(receipt, ad7_true_exterior_arc_execution=reference(__file__))


def raw_chunk(task, *, original_sampler):
    values, receipt = original_sampler(task)
    if not _INSTALLED:
        raise ValueError("Raw source worker did not install reviewed true arcs")
    return values, dict(receipt, ad7_true_exterior_arc_execution=reference(__file__))


def geometry_sample(task, *, original_sampler):
    result, receipt = original_sampler(task)
    if not _INSTALLED:
        raise ValueError("Geometry worker did not install reviewed true arcs")
    return result, dict(receipt, ad7_true_exterior_arc_execution=reference(__file__))


def install_successor(owner, *, mapping_reference):
    """Exact source identities only, before first worker or fresh source J."""
    value, lookup = base._load(mapping_reference)
    if (value.get("arc_execution_producer") != reference(__file__)
            or value.get("arc_schema") != SCHEMA
            or any(item.get("source_boundary_side") != "exterior_bottom_arc" for item in lookup.values())):
        raise ValueError("Frozen reviewed exterior-arc execution required")
    programme = owner.programme
    bundle = programme.bundle
    if any(pool is not None for pool in (programme.pool, bundle._geometry_pool, bundle._source_pool)):
        raise ValueError("Install source-reviewed arcs before execution pools")
    if getattr(owner, "ad7_source_arc_receipt", None) is not None:
        raise ValueError("Reviewed source arcs must be installed once")
    before_q = np.asarray(owner.start).copy()
    before_aux = {k:np.asarray(v).copy() for k,v in owner.checkpoint_arrays().items()}
    install_execution()
    frames, transition = base.transition_frames(bundle.frames, lookup=lookup,
        mapping_reference=mapping_reference, target_builder=reviewed_target)
    bundle.frames = frames
    bundle.descriptors = OrderedDict()
    for frame in frames:
        for row in frame["rows"]:
            bundle.descriptors.setdefault(row["_target_key"], deepcopy(row["target"]))
    bundle.keys = tuple(bundle.descriptors)
    bundle.target_descriptors = tuple(bundle.descriptors.values())
    from .first_shot_disjoint_print import _clear_caches
    _clear_caches(programme)
    import native_transverse_bundle as runtime
    from . import first_shot_transverse_execution as context
    runtime.init_worker = partial(worker_init, original_initializer=runtime.init_worker)
    runtime.source_chunk = partial(source_chunk, original_sampler=runtime.source_chunk)
    runtime.raw_chunk = partial(raw_chunk, original_sampler=runtime.raw_chunk)
    context._init_context_worker = partial(worker_init, original_initializer=context._init_context_worker)
    context._context_sample = partial(geometry_sample, original_sampler=context._context_sample)
    receipt = dict(schema=SCHEMA, producer=reference(__file__), mapping=deepcopy(mapping_reference),
        transition=transition, actual_material_bank_unchanged=True,
        source_UV_ID_sigma_W_clock_and_parameters_unchanged=True,
        same_closed_ink_piece_opposite_side_excluded=True,
        original_source_labels_and_target_history_retained=True,
        free_curve_station_retained=True, point_correspondences_added=False,
        fresh_source_J_required=True, old_source_or_world_J_reusable=False)
    previous_identity, previous_arrays = owner.checkpoint_identity, owner.checkpoint_arrays
    archive = "ad7_source_paths_"+mapping_reference["sha256"][:16]+"_json"
    metadata = np.asarray(json.dumps(receipt, sort_keys=True))
    if archive in before_aux and before_aux[archive].tobytes() != metadata.tobytes():
        raise ValueError("Inherited source arc identity conflicts")
    def identity(this):
        return dict(previous_identity(), ad7_source_exterior_arcs=deepcopy(receipt))
    def arrays(this):
        return dict(previous_arrays(), **{archive:metadata.copy()})
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.checkpoint_arrays = MethodType(arrays, owner)
    for ref in (reference(__file__), reference(base.__file__), deepcopy(mapping_reference)):
        for obj, name in ((owner,"source_producers"), (owner,"_references"), (programme,"source_producers")):
            values = getattr(obj, name)
            if ref not in values:
                values.append(deepcopy(ref))
    if owner.start.tobytes() != before_q.tobytes():
        raise ValueError("Source arc installation changed full parameters")
    after_aux = owner.checkpoint_arrays()
    for key, expected in before_aux.items():
        actual = after_aux[key]
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("Source arc installation changed inherited auxiliary: "+key)
    owner.ad7_source_arc_receipt = deepcopy(receipt)
    return deepcopy(receipt)
