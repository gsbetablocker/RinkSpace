"""Source-supported tolerance intervals along a frozen image-edge normal.

This is an opt-in observation model, not an estimated standard deviation or
a probability interval. It never relocates the stored UV. The caller must
qualify the same physical edge from source pixels before enabling a row.
Apply to native pixel residuals before the existing quality and Huber steps.
"""
from __future__ import annotations

import numpy as np


class SourceNormalIntervals:
    """Keep tangential/endpoint errors; penalize normal errors outside support.

    Input residuals are projected-nearest-point minus original observed UV.
    Normals and bounds are fixed source quantities, independent of a camera.
    Unknown rows pass through exactly. At a support boundary the derivative
    takes the exterior branch; the piecewise objective has no unique second
    derivative there.
    """

    def __init__(self, normals, bounds, eligible):
        self.normals = np.asarray(normals, np.float64).copy()
        self.bounds = np.asarray(bounds, np.float64).copy()
        self.eligible = np.asarray(eligible, bool).copy()
        count = len(self.eligible)
        if (self.eligible.ndim != 1 or self.normals.shape != (count, 2)
                or self.bounds.shape != (count, 2)):
            raise ValueError("Edge support needs N normals, N bounds, and N flags")
        normal = self.normals[self.eligible]
        limits = self.bounds[self.eligible]
        if (not np.isfinite(normal).all() or not np.isfinite(limits).all()
                or np.any(np.abs(np.linalg.norm(normal, axis=1)-1.) > 1e-8)
                or np.any(limits[:, 0] >= limits[:, 1])
                or np.any(limits[:, 0] > 0.) or np.any(limits[:, 1] < 0.)):
            raise ValueError("Qualified support must be finite, unit-normal, and contain original UV")
        for value in (self.normals, self.bounds, self.eligible):
            value.flags.writeable = False

    def apply(self, residual, jacobians=None):
        raw = np.asarray(residual, np.float64)
        if raw.shape != (2*len(self.eligible),) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite native two-component residual per original UV")
        result = raw.reshape(-1, 2).copy()
        normal = self.normals[self.eligible]
        limits = self.bounds[self.eligible]
        signed = np.einsum("ni,ni->n", result[self.eligible], normal)
        result[self.eligible] -= normal*np.clip(signed, limits[:, 0], limits[:, 1])[:, None]
        if jacobians is None:
            return result.ravel()
        inside = self.eligible.copy()
        inside[self.eligible] = (signed > limits[:, 0]) & (signed < limits[:, 1])
        matrices = []
        for matrix in jacobians:
            value = np.asarray(matrix, np.float64)
            if (value.ndim != 2 or value.shape[0] != len(raw)
                    or not np.isfinite(value).all()):
                raise ValueError("Edge Jacobian must have the original 2N residual rows")
            mapped = value.reshape(len(self.eligible), 2, -1).copy()
            selected = self.normals[inside]
            component = np.einsum("ni,nij->nj", selected, mapped[inside])
            mapped[inside] -= selected[:, :, None]*component[:, None, :]
            matrices.append(mapped.reshape(value.shape))
        return result.ravel(), matrices

