"""Candidate exact TOP producer for the existing WideStrokeTruth query.

The complete wide source/guard remains untouched. Only its manual TOP query
avoids constructing unrelated curve/corner outputs and a deep scene copy.
The same world validator, perimeter and venue-paint qualifier still run.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path
from types import MethodType
import numpy as np


def qualified_wide_top_world(factor, world):
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    from structured_programme_source_candidate.paired_location_programme_geometry import PairedLocationProgrammeGeometry
    if factor.qualified_scene is not PairedLocationProgrammeGeometry.qualified_scene:
        raise TypeError("TOP-only candidate requires the original audited venue qualifier")
    geometry = factor.geometry
    geometry.validate_parameters(world)
    view = factor.__dict__.get('_wide_ice_material_view')
    if view is None:
        view = CurrentIceMaterialView(geometry)
        factor._wide_ice_material_view = view
    course, cq, _ = geometry.print_course_binding(world)
    bottom, top, _ = course.sample_perimeter(cq, with_jacobian=False)
    # The qualifier reads original ice inventory and current Lane extents.
    # Retain all inherited metadata by shallow reference; replace exactly the
    # same dynamic fields it consumes. No shared scene/list/array is mutated.
    scene = dict(course.base)
    scene['ice_features'] = list(view.features(world).values())
    scene['ice_boundary'] = dict(course.base['ice_boundary'], points_xyz=bottom.tolist())
    scene['board_surface'] = dict(course.base['board_surface'],
                                 bottom_edge_xyz=bottom.tolist(), top_edge_xyz=top.tolist())
    extent_state = getattr(geometry, 'ice_extent_state', None)
    if callable(extent_state):
        scene.update(extent_state(world))
    scene = factor.qualified_scene(scene)
    top = np.asarray(scene['board_surface']['top_edge_xyz'], float)
    if not np.array_equal(top[0], top[-1]):
        top = np.vstack((top, top[0]))
    return top


def _queries_for_records_top_only(self, q, records):
    from structured_programme_source_candidate.human_stroke_truth import (
        _truth_record_input_bytes, _truth_record_memo, finite_stroke_queries,
    )
    q = np.asarray(q, float)
    if q.shape != (self.parameter_count,):
        raise ValueError('Existing complete q required for original wide human guard')
    world = q[self.world_indices]
    camera = self.factor.camera(q[self.camera_indices])
    cache = getattr(self, '_constraint_exact_world_cache', None)
    if cache is None:
        top = qualified_wide_top_world(self.factor, world)
    else:
        self.factor.geometry.validate_parameters(world)
        key = world.tobytes()
        if key not in cache:
            cache[key] = qualified_wide_top_world(self.factor, world)
        top = cache[key]
    rows = []
    for record in records:
        key = ('wide-finite-query', _truth_record_input_bytes((camera, top,
               record['uv'], record['normals'], record['bounds'], record.get('tangent_bounds'))))
        query = _truth_record_memo(self, record, key, q,
            lambda record=record: finite_stroke_queries(camera, top, record['uv'],
                record['normals'], record['bounds'], association='nearest_finite_segment',
                tangent_bounds=record.get('tangent_bounds')))
        rows.append(dict(id=record['record']['id'], query=query))
    return rows


def install_wide_top_only(truth):
    """Install on one owned truth instance; no global/runtime source edits."""
    from structured_programme_source_candidate.human_stroke_truth import WideStrokeTruth
    from structured_programme_source_candidate.paired_location_programme_geometry import PairedLocationProgrammeGeometry
    if type(truth) is not WideStrokeTruth:
        raise TypeError('Original complete WideStrokeTruth instance required')
    if truth.factor.qualified_scene is not PairedLocationProgrammeGeometry.qualified_scene:
        raise TypeError('Original audited venue-paint qualifier required')
    if '_queries_for_records' in vars(truth):
        raise ValueError('Truth query already has a local override')
    truth._queries_for_records = MethodType(_queries_for_records_top_only, truth)
    return dict(execution_only=True, complete_source_and_guard_unchanged=True,
                original_validator_perimeter_and_qualifier_retained=True,
                complete_scene_deepcopy_and_unrelated_outputs_omitted=True,
                input_source_pixels_bounds_rows_and_FD_stencils_unchanged=True,
                accuracy_accepted=False)


def install_wide_top_only_runtime(namespace=None):
    """Run the original constructor before one owned instance installation.

    The driver imports the existing class inside main(). Preserve that class,
    its arguments, and all constructor work instead of replacing the class.
    """
    from structured_programme_source_candidate.human_stroke_truth import WideStrokeTruth
    cls = getattr(namespace, 'WideStrokeTruth', WideStrokeTruth)
    if cls is not WideStrokeTruth:
        raise TypeError('Only the original source-owned WideStrokeTruth class is supported')
    existing = getattr(cls, '_rinkspace_wide_top_only_execution_receipt', None)
    if existing is not None:
        return existing
    original = cls.__init__
    helper = Path(__file__).resolve()
    receipt = dict(original_constructor_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_query_function_sha256=hashlib.sha256(inspect.getsource(cls._queries_for_records).encode()).hexdigest(),
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        original_class_and_constructor_arguments_retained=True,
        original_constructor_runs_before_instance_installation=True,
        original_source_contract_and_all_other_methods_retained=True,
        registered_source_files_unchanged=True, accuracy_accepted=False)

    @functools.wraps(original)
    def construct(self, *args, **kwargs):
        original(self, *args, **kwargs)
        install_wide_top_only(self)

    cls.__init__ = construct
    cls._rinkspace_wide_top_only_execution_receipt = receipt
    return receipt
