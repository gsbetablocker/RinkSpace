"""Chain captured physical-camera factors through shared Brown4 graph nodes.

Graph nodes stay Vector11. Selected programme-camera inputs may be physical15
(``C,R,K,k1,k2,p1,p2``); external reference and scene inputs stay physical11.
Captured callbacks may supply complete physical15 Jacobians, or exact core-11
Jacobians at the current camera. In the latter case the four Brown columns
are central differences of the *complete captured residual*, including its
fixed source weights and robust transform. No source
pixel, factor callback, or template is mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


GRAPH_DIM = 11
EXTENDED_DIM = 15
BROWN_START = 11


def _state11(value, label: str) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape != (GRAPH_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite Vector11")
    return result


def _physical(value, label: str) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape not in ((GRAPH_DIM,), (EXTENDED_DIM,)) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite physical11/15 camera")
    return result


def adapt_factor(original_keys: Sequence[int], raw_residual, raw_jacobian,
                 options: Mapping, active_keys: set[int],
                 fixed_states: Mapping[int, np.ndarray], model,
                 *, derived_frame_keys: set[int] | None = None,
                 brown_fd_step: float = 1e-6):
    """Return a Vector11 factor whose selected camera inputs may be Brown15.

    The interface matches ``continuous_camera_model.adapt_factor``. Each
    occurrence of an original camera input is differentiated independently;
    when two occurrences alias the same exposure or Brown node, their graph
    Jacobian contributions are summed. Fixed external inputs are not expanded.
    A factor with no active dependency returns ``None``.
    """
    if not np.isfinite(brown_fd_step) or brown_fd_step <= 0:
        raise ValueError("Brown central-difference step must be positive")
    original_keys = tuple(map(int, original_keys))
    active = set(map(int, active_keys))
    if derived_frame_keys is None:
        derived = {key for key in original_keys if key in model.frames
                   and model.frames[key].frame_key in active}
    else:
        derived = set(map(int, derived_frame_keys))
    if derived - (set(model.frames) & set(original_keys)):
        raise ValueError("Derived camera keys must occur in this factor and model")

    kept: list[int] = []
    fixed_needed: set[int] = set()
    for key in original_keys:
        dependencies = model.dependencies(key) if key in derived else (key,)
        for dependency in dependencies:
            if dependency in active:
                if dependency not in kept:
                    kept.append(dependency)
            else:
                if dependency not in fixed_states:
                    raise KeyError(f"Missing fixed graph state {dependency}")
                fixed_needed.add(dependency)
    if not kept:
        return None
    kept = tuple(kept)
    options = dict(options)
    rows = int(options["residual_dim"])
    if rows < 1:
        raise ValueError("Adapted factor must have positive residual dimension")
    fixed_values = {key: _state11(fixed_states[key], "fixed factor state").copy()
                    for key in fixed_needed}

    def expand(q, with_jacobian=False):
        if len(q) != len(kept):
            raise ValueError("Adapted factor received the wrong number of graph states")
        values = fixed_values.copy()
        values.update(zip(kept, (_state11(value, "active factor state") for value in q)))
        expanded = []
        chains = [] if with_jacobian else None
        for key in original_keys:
            if key in derived:
                if with_jacobian:
                    state, derivatives = model.physical_frame(key, values)
                    state = _physical(state, "derived camera")
                    for dependency, matrix in derivatives.items():
                        matrix = np.asarray(matrix, np.float64)
                        if matrix.shape != (len(state), GRAPH_DIM) or not np.isfinite(matrix).all():
                            raise ValueError(f"Invalid physical-camera chain for {dependency}")
                    expanded.append(state)
                    chains.append(derivatives)
                else:
                    expanded.append(_physical(model.physical_state(key, values),
                                              "derived camera"))
            else:
                expanded.append(_state11(values[key], "external factor state"))
                if with_jacobian:
                    chains.append({key: np.eye(GRAPH_DIM)})
        return expanded, chains

    def _residual_at(states):
        value = np.asarray(raw_residual(states), np.float64).reshape(-1)
        if value.shape != (rows,) or not np.isfinite(value).all():
            raise ValueError("Captured residual changed size or became nonfinite")
        return value

    def residual(q):
        return _residual_at(expand(q)[0])

    def jacobian(q):
        expanded, chains = expand(q, with_jacobian=True)
        originals = raw_jacobian(expanded)
        if len(originals) != len(original_keys):
            raise ValueError("Raw factor omitted an original Jacobian block")
        result = {key: np.zeros((rows, GRAPH_DIM)) for key in kept}
        for index, (matrix, chain) in enumerate(zip(originals, chains)):
            matrix = np.asarray(matrix, np.float64)
            physical_dim = len(expanded[index])
            if not np.isfinite(matrix).all():
                raise ValueError("Raw factor Jacobian became nonfinite")
            if physical_dim == GRAPH_DIM:
                if matrix.shape != (rows, GRAPH_DIM):
                    raise ValueError("External or physical11 Jacobian must have 11 columns")
            elif matrix.shape == (rows, GRAPH_DIM):
                brown = np.empty((rows, EXTENDED_DIM-GRAPH_DIM), np.float64)
                for local in range(EXTENDED_DIM-GRAPH_DIM):
                    axis = BROWN_START+local
                    step = brown_fd_step*max(1., abs(float(expanded[index][axis])))
                    high, low = list(expanded), list(expanded)
                    high_state = expanded[index].copy()
                    low_state = expanded[index].copy()
                    high_state[axis] += step
                    low_state[axis] -= step
                    high[index], low[index] = high_state, low_state
                    brown[:, local] = (_residual_at(high)-_residual_at(low))/(2*step)
                matrix = np.column_stack((matrix, brown))
            elif matrix.shape != (rows, EXTENDED_DIM):
                raise ValueError("Physical15 Jacobian must have 11 core or 15 full columns")
            for key, derivative in chain.items():
                if key in result:
                    result[key] += matrix @ derivative
        return [result[key] for key in kept]

    return kept, residual, jacobian if raw_jacobian is not None else None, options
