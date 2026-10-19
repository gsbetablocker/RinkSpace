"""Reuse an already-installed exact station cache for direct perimeter queries.

No new cache, budget, or startup policy is introduced. The existing station
producer creates the same perimeter/J arrays; callers receive fresh copies
with the original mutability. Installation is per owned board-factor instance.
"""
from __future__ import annotations

from copy import deepcopy
import functools
from types import MethodType
import numpy as np


def existing_cached_perimeter(course, q=None, *, with_jacobian=False):
    from structured_programme_source_candidate.course_station_cache import ExactCourseStationCache
    from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry
    cache = course.station
    if (not isinstance(cache, ExactCourseStationCache)
            or type(course).sample_perimeter is not SharedBoardCourseNumericGeometry.sample_perimeter):
        return course.sample_perimeter(q, with_jacobian=with_jacobian)
    # Preserve the released q=None and per-call physical-validator behavior.
    value = course.initial if q is None else course.validate_parameters(q)
    perimeter = cache._inputs(value)['perimeter']
    if with_jacobian:
        return (*[array.copy() for array in perimeter[:4]], deepcopy(perimeter[4]))
    return perimeter[0].copy(), perimeter[1].copy(), deepcopy(perimeter[4])


def install_board_perimeter_bridge(factor):
    """Bridge only the two original complete physical-perimeter branches."""
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    if type(factor) is not NativeBoardEdgeFactor:
        raise TypeError('Original NativeBoardEdgeFactor required')
    if '_world' in vars(factor):
        raise ValueError('Board world producer already has an instance override')
    original = factor._world

    @functools.wraps(original)
    def world(self, q, curve=None):
        effective = None if curve is None else self.effective_curve(curve)
        if effective is None or effective.get('kind') not in ('board_top', 'ice_board_contact'):
            return original(q, curve)
        if effective.get('source_structure') not in ('far_visible_board', 'physical_course_facet'):
            raise ValueError('An explicit original far-board or physical facet source structure is required')
        physical = np.asarray(q, float)[self.world_indices]
        self.geometry.validate_parameters(physical)
        course, cq, _ = self.geometry.print_course_binding(physical)
        level = 1 if effective['kind'] == 'board_top' else 0
        perimeter = existing_cached_perimeter(course, cq, with_jacobian=False)
        result = np.asarray(perimeter[level], float)
        if not np.array_equal(result[0], result[-1]):
            result = np.vstack((result, result[0]))
        if result.ndim != 2 or result.shape[1] != 3 or len(result) < 3 or not np.isfinite(result).all():
            raise ValueError('Current complete finite physical board top/ice contact required')
        return result

    factor._world = MethodType(world, factor)
    return dict(existing_station_cache_only=True, cache_budget_and_activation_unchanged=True,
                physical_validator_retained_per_call=True, source_pixels_rows_roles_and_FD_unchanged=True,
                caller_outputs_remain_independently_mutable=True, accuracy_accepted=False)
