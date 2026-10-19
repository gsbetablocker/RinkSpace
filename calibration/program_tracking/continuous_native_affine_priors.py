"""Opt-in native equivalents of the continuous rig's affine regularizers.

The caller supplies an already loaded ``NativeEdgeManager``.  This module
changes neither the regularizer targets nor the default Python graph path.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class NativeAffinePriors:
    """Append exact Vector11 affine factors through round6's native bridge."""

    def __init__(self, native_edge_manager):
        self.native = native_edge_manager.native
        self.gtsam = native_edge_manager.gtsam
        if not hasattr(self.native, "add_linear_factor"):
            raise RuntimeError("Loaded round6 bridge has no native linear factor")
        self.counts = Counter()

    def _add(self, graph, keys, matrices, target, sigma, family):
        keys = tuple(graph._frame_id(key) for key in keys)
        graph._check_live(keys)
        target = np.asarray(target, np.float64).reshape(-1).copy()
        if not keys or not len(target) or not np.isfinite(target).all():
            raise ValueError("Native affine target must be a finite nonempty vector")
        matrices = tuple(np.ascontiguousarray(matrix, dtype=np.float64)
                         for matrix in matrices)
        if (len(matrices) != len(keys) or
                any(matrix.shape != (len(target), 11) or
                    not np.isfinite(matrix).all() for matrix in matrices)):
            raise ValueError("Native affine matrices must have target rows and 11 columns")
        noise = self.gtsam.noiseModel.Diagonal.Sigmas(
            graph._sigma(sigma, len(target)))
        symbol_keys = tuple(graph._key(key) for key in keys)
        holder = self.gtsam.NonlinearFactorGraph()
        index = self.native.add_linear_factor(
            holder, noise, list(symbol_keys), list(matrices), target)
        if index != 0 or holder.size() != 1:
            raise RuntimeError("Native affine factor construction changed factor count")
        factor = holder.at(0)
        if tuple(map(int, factor.keys())) != symbol_keys:
            raise RuntimeError("Native affine factor changed key order")
        graph._pending.append(factor)
        graph._pending_ids.update(keys)
        self.counts[family] += 1

    def add_selected_prior(self, graph, key, columns, target, sigma):
        columns = tuple(columns)
        selection = np.eye(11)[list(columns)]
        self._add(graph, (key,), (selection,), target, sigma, "selected_prior")

    def add_dummy_factor(self, graph, key, live_columns):
        columns = tuple(i for i in range(11) if i not in live_columns)
        selection = np.eye(11)[list(columns)]
        self._add(graph, (key,), (selection,), np.zeros(len(columns)), 1.,
                  "dummy_factor")

    def add_acceleration_factor(self, graph, keys, times, columns, sigma):
        dt0, dt1 = np.diff(times)
        mid = .5 * (dt0 + dt1)
        coefficients = np.asarray([1/dt0, -1/dt0-1/dt1, 1/dt1]) / mid
        selection = np.eye(11)[list(columns)]
        matrices = tuple(coefficient * selection for coefficient in coefficients)
        self._add(graph, keys, matrices, np.zeros(len(columns)),
                  np.asarray(sigma) / np.sqrt(mid), "acceleration_factor")

    def metadata(self):
        return dict(enabled=True, backend="round6 native_source_edge_prod.add_linear_factor",
                    counts=dict(self.counts),
                    scope="exact affine priors only; original targets and per-axis noise")
