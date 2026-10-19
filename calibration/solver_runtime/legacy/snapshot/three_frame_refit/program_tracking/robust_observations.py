"""Pointwise robust observations without dropping an entire image match set.

GTSAM's Python Huber model applies block reweighting to a CustomFactor. For a
factor containing many independent image points that can suppress all good
points together. This equivalent square-root Huber residual robustifies each
1D curve distance or 2D feature pair and retains analytic derivatives.
"""
from __future__ import annotations

import os
from importlib import import_module
from threading import RLock

import numpy as np


def _acceleration_mode() -> str:
    """Use the solver-wide mode; tolerate the module's staged installation."""
    name = f"{__package__}.acceleration" if __package__ else "acceleration"
    try:
        return str(import_module(name).MODE)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return os.environ.get("RINKSPACE_SOLVER_ACCELERATION", "auto")


def _pointwise_huber_uncached(residual_fn, jacobian_fn, *, group_size: int, delta: float):
    """Original implementation, retained for a strict uncached baseline."""
    if group_size < 1 or delta <= 0:
        raise ValueError('Positive robust group size and pixel delta required')

    def transform(states):
        raw = np.asarray(residual_fn(states), float).reshape(-1, group_size)
        norm = np.linalg.norm(raw, axis=1)
        scale = np.ones(len(raw))
        derivative = np.zeros(len(raw))
        tail = norm > delta
        q = norm[tail]
        rho_root = np.sqrt(2.*delta*q-delta*delta)
        scale[tail] = rho_root/q
        derivative[tail] = delta/(rho_root*q*q)-rho_root/(q*q*q)
        return raw, scale, derivative

    def residual(states):
        raw, scale, _ = transform(states)
        return (raw*scale[:, None]).ravel()

    def jacobians(states):
        raw, scale, derivative = transform(states)
        matrices = []
        for value in jacobian_fn(states):
            jac = np.asarray(value, float).reshape(len(raw), group_size, -1)
            radial = np.einsum('ni,nij->nj', raw, jac)
            result = scale[:, None, None]*jac
            result += derivative[:, None, None]*raw[:, :, None]*radial[:, None, :]
            matrices.append(result.reshape(len(raw)*group_size, -1))
        return matrices

    return residual, jacobians


def pointwise_huber(residual_fn, jacobian_fn, *, group_size: int, delta: float,
                    cache: bool | None = None, cache_raw: bool | None = None):
    """Apply the same pointwise Huber transform, caching a factor's last state.

    Cache entries belong to this returned closure only. State bytes are copied
    and compared on every callback, so GTSAM's in-place state updates invalidate
    them. Observations and weights captured by ``residual_fn`` must remain fixed
    after factor construction, and callers must not mutate a state during one
    callback. A lock serializes callbacks to the same factor's cache.

    The solver-wide acceleration MODE='baseline' always selects the original
    uncached implementation. ``cache=False`` or RINKSPACE_HUBER_CACHE=0 also
    disables it explicitly. ``cache_raw`` is an alias for ``cache``.
    """
    if cache is not None and cache_raw is not None and cache != cache_raw:
        raise ValueError('Conflicting cache and cache_raw settings')
    if cache_raw is not None:
        cache = cache_raw
    cache = (_acceleration_mode() != "baseline" and
             os.environ.get("RINKSPACE_HUBER_CACHE", "1") != "0" and
             cache is not False)
    if not cache:
        return _pointwise_huber_uncached(
            residual_fn, jacobian_fn, group_size=group_size, delta=delta)
    if group_size < 1 or delta <= 0:
        raise ValueError('Positive robust group size and pixel delta required')

    previous: tuple[np.ndarray, ...] | None = None
    transformed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    lock = RLock()

    def transform(states):
        nonlocal previous, transformed
        with lock:
            # Value comparison alone treats +0 and -0 as equal, although raw
            # geometry can distinguish their signs (for example in atan2).
            same = (previous is not None and len(previous) == len(states) and
                    all(saved.shape == np.asarray(current).shape and
                        saved.dtype == np.asarray(current).dtype and
                        saved.tobytes() == np.asarray(current).tobytes()
                        for saved, current in zip(previous, states)))
            if same:
                return transformed

            # Own the raw values: residual_fn may reuse its output buffer, and
            # caller-visible robust residuals must never expose cached arrays.
            raw = np.array(np.asarray(residual_fn(states), float).reshape(-1, group_size),
                           copy=True)
            norm = np.linalg.norm(raw, axis=1)
            scale = np.ones(len(raw))
            derivative = np.zeros(len(raw))
            tail = norm > delta
            q = norm[tail]
            rho_root = np.sqrt(2.*delta*q-delta*delta)
            scale[tail] = rho_root/q
            derivative[tail] = delta/(rho_root*q*q)-rho_root/(q*q*q)

            snapshot = tuple(np.array(value, copy=True) for value in states)
            for value in (*snapshot, raw, scale, derivative):
                value.flags.writeable = False
            previous = snapshot
            transformed = (raw, scale, derivative)
            return transformed

    def residual(states):
        with lock:
            raw, scale, _ = transform(states)
            return (raw*scale[:, None]).ravel()

    def jacobians(states):
        with lock:
            raw, scale, derivative = transform(states)
            matrices = []
            for value in jacobian_fn(states):
                jac = np.asarray(value, float).reshape(len(raw), group_size, -1)
                radial = np.einsum('ni,nij->nj', raw, jac)
                result = scale[:, None, None]*jac
                result += derivative[:, None, None]*raw[:, :, None]*radial[:, None, :]
                matrices.append(result.reshape(len(raw)*group_size, -1))
            return matrices

    return residual, jacobians
