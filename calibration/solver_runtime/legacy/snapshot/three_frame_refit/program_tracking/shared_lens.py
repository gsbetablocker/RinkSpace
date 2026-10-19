"""Optional shared physical-lens factors for one fixed camera position.

Frame states retain their free pan, tilt and common zoom. A separate 11-vector
per camera stores only log(fx/fy), cx, cy and skew; its unused entries are held
at zero. All principal-point and skew values use the camera state's /1000
normalization. These are soft calibration factors, not image observations.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULTS = {
    "ratio_sigma_log": 0.005,
    "principal_sigma_px": (5.0, 5.0),
    "skew_sigma_px": 5.0,
    "ratio_prior_sigma_log": 0.1,
    "principal_prior_sigma_px": (200.0, 200.0),
    "skew_prior_sigma_px": 50.0,
}


def _settings(config: Mapping | None) -> dict:
    options = dict(DEFAULTS)
    if config is not None:
        options.update(config)
    for name in ("ratio_sigma_log", "skew_sigma_px", "ratio_prior_sigma_log",
                 "skew_prior_sigma_px"):
        value = float(options[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        options[name] = value
    for name in ("principal_sigma_px", "principal_prior_sigma_px"):
        value = np.asarray(options[name], np.float64).reshape(-1)
        if value.shape != (2,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name} must contain two finite positive pixel sigmas")
        options[name] = value
    return options


def lens_initial() -> np.ndarray:
    """Return one 11-state lens node: ratio 1, principal (960,540), skew 0."""
    state = np.zeros(11, np.float64)
    state[8:10] = (0.960, 0.540)
    return state


def add_lens_prior(graph, key: int, initial: np.ndarray,
                   config: Mapping | None = None) -> None:
    """Give the shared node a broad physical prior and fix unused entries."""
    options = _settings(config)
    target = np.asarray(initial, np.float64).reshape(-1).copy()
    if target.shape != (11,) or not np.isfinite(target).all():
        raise ValueError("shared lens initial state must be 11 finite values")
    if np.max(np.abs(target[[0, 1, 2, 3, 4, 5, 7]])) > 1e-12:
        raise ValueError("shared lens unused state entries must stay zero")
    sigma = np.full(11, 1e-6)
    sigma[6] = options["ratio_prior_sigma_log"]
    sigma[8:10] = options["principal_prior_sigma_px"] / 1000.
    sigma[10] = options["skew_prior_sigma_px"] / 1000.
    graph.add_prior(key, target, sigma)


def add_frame_lens(graph, frame_key: int, lens_key: int,
                   config: Mapping | None = None) -> None:
    """Tie frame ratio/principal/skew softly to the camera's shared lens.

    Both Jacobians are exact in the graph's Euclidean camera-state coordinates.
    The ratio is log(fx/fy); it leaves frame log zoom free.
    """
    options = _settings(config)
    sigma = np.r_[options["ratio_sigma_log"],
                  options["principal_sigma_px"] / 1000.,
                  options["skew_sigma_px"] / 1000.]
    frame_jac = np.zeros((4, 11), np.float64)
    lens_jac = np.zeros((4, 11), np.float64)
    frame_jac[0, 6:8] = (1., -1.)
    lens_jac[0, 6] = -1.
    for row, column in enumerate((8, 9, 10), start=1):
        frame_jac[row, column] = 1.
        lens_jac[row, column] = -1.

    def residual(states: list[np.ndarray]) -> np.ndarray:
        frame, lens = states
        return np.array((frame[6] - frame[7] - lens[6],
                         frame[8] - lens[8], frame[9] - lens[9],
                         frame[10] - lens[10]), np.float64)

    def jacobians(_states: list[np.ndarray]) -> list[np.ndarray]:
        return [frame_jac, lens_jac]

    graph.add_observation([frame_key, lens_key], residual, residual_dim=4,
                          sigma=sigma, jac_fn=jacobians)
