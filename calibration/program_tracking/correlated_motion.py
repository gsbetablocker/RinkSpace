"""Optional fixed, low-rank pixel noise for a source-image correspondence set.

The four latent modes are common horizontal/vertical translation and a small
scale/rotation displacement around (960, 540).  They describe an explicit
noise hypothesis, not a fitted or empirically calibrated uncertainty.  Apply
this wrapper *after* the existing pointwise Huber transform, then give the
factor unit sigma.  This robustified correlated objective is not identical to
eliminating a latent similarity transform under a raw-pixel Huber likelihood.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


ResidualFn = Callable[[list[np.ndarray]], np.ndarray]
JacobianFn = Callable[[list[np.ndarray]], list[np.ndarray]]

_CENTER_PX = (960.0, 540.0)
_RADIUS_PX = 960.0
_MODE_NAMES = ("translation_u", "translation_v", "scale", "rotation")


class SimilarityNoiseWhitener:
    """Apply ``S^-1/2`` where ``S = sigma_point^2 I + B diag(sigma_modes^2) B.T``.

    For target pixel ``(u, v)``, let ``x=(u-960)/960`` and ``y=(v-540)/960``.
    Its two rows of ``B`` are ``[1,0,x,-y]`` and ``[0,1,y,x]``.  Only a thin
    SVD of the four-column scaled mode matrix is retained, so applying the
    whitener to residuals or Jacobians is linear in the number of pixels.
    """

    def __init__(
        self,
        xy1: np.ndarray,
        *,
        sigma_point_px: float,
        sigma_modes_px: Sequence[float],
    ) -> None:
        pixels = np.asarray(xy1, dtype=np.float64)
        point_sigma = float(sigma_point_px)
        mode_sigmas = np.asarray(sigma_modes_px, dtype=np.float64).reshape(-1)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) == 0:
            raise ValueError("xy1 must contain one or more target image (u, v) points")
        if not np.isfinite(pixels).all():
            raise ValueError("xy1 must be finite")
        if not np.isfinite(point_sigma) or point_sigma <= 0:
            raise ValueError("sigma_point_px must be finite and positive")
        if mode_sigmas.shape != (4,) or not np.isfinite(mode_sigmas).all() or np.any(mode_sigmas < 0):
            raise ValueError("sigma_modes_px must contain four finite nonnegative values")

        x = (pixels[:, 0] - _CENTER_PX[0]) / _RADIUS_PX
        y = (pixels[:, 1] - _CENTER_PX[1]) / _RADIUS_PX
        basis = np.zeros((2 * len(pixels), 4), dtype=np.float64)
        basis[0::2, 0] = 1.0
        basis[1::2, 1] = 1.0
        basis[0::2, 2] = x
        basis[1::2, 2] = y
        basis[0::2, 3] = -y
        basis[1::2, 3] = x
        scaled_modes = basis * mode_sigmas[None, :]
        left, singular, _ = np.linalg.svd(scaled_modes, full_matrices=False)

        self.point_count = len(pixels)
        self.sigma_point_px = point_sigma
        self.sigma_modes_px = tuple(float(value) for value in mode_sigmas)
        self._left = left
        self._attenuation = 1.0 / np.sqrt(1.0 + (singular / point_sigma) ** 2) - 1.0

    def whiten(self, value: np.ndarray) -> np.ndarray:
        """Whiten a 2N residual or a 2N-by-D Jacobian without forming ``S``."""
        array = np.asarray(value, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != 2 * self.point_count:
            raise ValueError("input must have 2N rows")
        if not np.isfinite(array).all():
            raise ValueError("input to correlated whitener must be finite")
        projected = self._left.T @ array
        if array.ndim == 1:
            correction = self._left @ (self._attenuation * projected)
        else:
            correction = self._left @ (self._attenuation[:, None] * projected)
        return (array + correction) / self.sigma_point_px

    def wrap(self, residual_fn: ResidualFn, jac_fn: JacobianFn) -> tuple[ResidualFn, JacobianFn]:
        """Return callbacks with the same shape/order as the input callbacks."""
        if not callable(residual_fn) or not callable(jac_fn):
            raise TypeError("residual_fn and jac_fn must be callable")

        def residual(states: list[np.ndarray]) -> np.ndarray:
            return self.whiten(residual_fn(states))

        def jacobians(states: list[np.ndarray]) -> list[np.ndarray]:
            return [self.whiten(matrix) for matrix in jac_fn(states)]

        return residual, jacobians

    def metadata(self) -> dict:
        """Return the stated pixel-noise assumptions for candidate provenance."""
        return {
            "model": "fixed_low_rank_similarity_noise",
            "calibration_status": "explicit_hypothesis_not_empirically_calibrated",
            "point_count": self.point_count,
            "sigma_point_px": self.sigma_point_px,
            "sigma_modes_px": dict(zip(_MODE_NAMES, self.sigma_modes_px)),
            "mode_center_px": list(_CENTER_PX),
            "mode_radius_px": _RADIUS_PX,
            "application_order": "after_pointwise_huber",
        }


def wrap_correlated_pair(
    residual_fn: ResidualFn,
    jac_fn: JacobianFn,
    xy1: np.ndarray,
    *,
    sigma_point_px: float,
    sigma_modes_px: Sequence[float],
) -> tuple[ResidualFn, JacobianFn, dict]:
    """Build the opt-in whitener and return callbacks plus raw-mode metadata."""
    whitener = SimilarityNoiseWhitener(
        xy1, sigma_point_px=sigma_point_px, sigma_modes_px=sigma_modes_px,
    )
    residual, jacobians = whitener.wrap(residual_fn, jac_fn)
    return residual, jacobians, whitener.metadata()
