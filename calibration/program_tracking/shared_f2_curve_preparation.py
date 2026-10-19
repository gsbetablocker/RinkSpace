"""Scene-aware use of the existing adaptive reference-curve worker cache.

The stock cache token contains only physical camera state. A released F2
circle additionally depends on graph node 2001, so its worker job must be
replaced and its cached bundle invalidated when that node changes. Unchanged
companion curves use the stock jobs directly. This module does not alter the
original worker, callback, or solver source files.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

from adaptive_curve_parallel import prepare_adaptive_curve_jobs
from shared_f2_nested_factor import F2_SCENE_KEY


def prepare_joint_shared_f2_curves(graph, states):
    """Prime all original/new jobs using their exact accepted camera+scene."""
    dynamic = getattr(graph, '_shared_f2_dynamic_jobs', ())
    if dynamic:
        scene = np.asarray(states[F2_SCENE_KEY], np.float64)
        token = scene.tobytes()
        jobs = graph._adaptive_curve_jobs
        manager = getattr(graph, '_adaptive_curve_manager', None)
        for index, factor in dynamic:
            if getattr(factor, '_prepared_scene_token', None) != token:
                jobs[index] = factor.adaptive_job(scene)
                factor._prepared_scene_token = token
                if manager is not None:
                    manager.cached.pop(index, None)
    return prepare_adaptive_curve_jobs(graph, states)


@contextmanager
def install_joint_shared_f2_curve_preparation(graph):
    """Route the unchanged CompleteFactorSession solver's preparation hook.

    The installed wrapper is scoped to one graph/process. Its non-F2 calls
    still invoke the stock worker code. Use around the future joint solve;
    never patch a concurrently running programme solver.
    """
    import continuous_camera_solver as solver
    previous = solver.prepare_adaptive_curve_jobs
    def dispatch(target_graph, states):
        if target_graph is graph:
            return prepare_joint_shared_f2_curves(graph, states)
        return previous(target_graph, states)
    solver.prepare_adaptive_curve_jobs = dispatch
    try:
        yield
    finally:
        solver.prepare_adaptive_curve_jobs = previous
