"""GTSAM iSAM2 fixed-lag graph for camera states at source-frame timestamps.

Each caller-defined state is a Euclidean vector.  Camera projection belongs in
the observation callback; this module only handles graph optimization.  A new
graph is constructed for every camera shot.  GTSAM's
``IncrementalFixedLagSmoother`` uses iSAM2 and marginalizes old states as the
window advances.  Cross-window links to archived frames are rejected instead
of being silently treated as current observations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
import sys
import time

import gtsam
import gtsam_unstable
import numpy as np

from .callback_factory import select_factory


State = np.ndarray
ResidualFn = Callable[[list[State]], np.ndarray]
JacobianFn = Callable[[list[State]], list[np.ndarray]]


@dataclass
class _Observation:
    frame_ids: tuple[int, ...]
    residual_fn: ResidualFn
    jac_fn: JacobianFn | None
    noise: object
    residual_dim: int
    fd_step: float


class IncrementalCameraGraph:
    """Optimize caller-defined camera vectors with actual GTSAM iSAM2 updates.

    ``frame_id`` is a nonnegative unique integer within a shot (source PTS is
    suitable).  ``add_observation`` accepts one or more states; its residual
    callback receives their vectors in ``keys`` order and returns a fixed
    length raw residual vector.  Its optional Jacobian callback returns one
    matrix per state, each shaped ``(residual_dim, state_dim)``.  Without an
    analytic callback a central numerical derivative is used.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        window_size: int = 120,
        relinearize_threshold: float = 0.01,
        relinearize_skip: int = 1,
        refinement_updates: int = 0,
        optimization: str = "gauss_newton",
        dogleg_initial_delta: float = 0.1,
        diagnose_cost: bool | None = None,
    ) -> None:
        if state_dim < 1 or window_size < 2:
            raise ValueError("state_dim must be positive and window_size >= 2")
        if relinearize_threshold <= 0 or relinearize_skip < 1:
            raise ValueError("invalid iSAM2 relinearization settings")
        if refinement_updates < 0:
            raise ValueError("invalid refinement update count")
        if optimization not in ("gauss_newton", "dogleg"):
            raise ValueError("optimization must be gauss_newton or dogleg")
        if not np.isfinite(dogleg_initial_delta) or dogleg_initial_delta <= 0:
            raise ValueError("dogleg_initial_delta must be finite and positive")
        self.state_dim = int(state_dim)
        self._custom_factor, self.callback_metadata = select_factory()
        self.window_size = int(window_size)
        self.refinement_updates = int(refinement_updates)
        self._optimization = optimization
        self._dogleg_initial_delta = float(dogleg_initial_delta)
        self._diagnose_cost = (os.getenv("RINKSPACE_GTSAM_DIAGNOSTICS") == "1"
                               if diagnose_cost is None else bool(diagnose_cost))
        # Read only. Enable for a necessary solve with RINKSPACE_ISAM_PHASE_PROFILE=1.
        # In particular, never request an extra estimate here: getDelta() is lazy
        # and an extra calculateEstimate() can change when Dogleg work happens.
        self._phase_profile = os.getenv("RINKSPACE_ISAM_PHASE_PROFILE") == "1"
        self.last_update_phase_timings: list[dict[str, float | int | str]] = []
        self.last_update_diagnostics: list[dict[str, float | int | str | None]] = []
        self.last_update_summary: dict[str, float | int | None] = {}
        self.last_batch_diagnostics: dict[str, float | int | bool | str | None] = {}
        self._relinearize_threshold = float(relinearize_threshold)
        self._relinearize_skip = int(relinearize_skip)
        self._smoother = gtsam_unstable.IncrementalFixedLagSmoother(
            float(window_size - 1), self._new_isam_params()
        )
        self._active: list[int] = []
        self._latest: dict[int, State] = {}
        self._archived: dict[int, State] = {}
        self.last_retired_estimates: dict[int, State] = {}
        self._pending: list[object] = []
        self._pending_ids: set[int] = set()
        self._prior_initials: dict[int, State] = {}
        self._callbacks: list[tuple[tuple[int, ...], Callable]] = []
        self._window_count = 0
        self._next_time = 0

    def _new_isam_params(self) -> gtsam.ISAM2Params:
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(self._relinearize_threshold)
        params.relinearizeSkip = self._relinearize_skip
        params.findUnusedFactorSlots = True
        if self._optimization == "dogleg":
            dogleg = gtsam.ISAM2DoglegParams()
            dogleg.setInitialDelta(self._dogleg_initial_delta)
            params.setOptimizationParams(dogleg)
        return params

    def _vector(self, vector: Sequence[float], *, name: str = "state") -> State:
        result = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
        if result.shape != (self.state_dim,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length {self.state_dim}")
        return result

    @staticmethod
    def _frame_id(frame_id: int) -> int:
        if not isinstance(frame_id, (int, np.integer)) or not 0 <= frame_id < 2**56:
            raise ValueError("frame_id must be a nonnegative integer below 2**56")
        return int(frame_id)

    @staticmethod
    def _key(frame_id: int) -> int:
        return gtsam.symbol("x", frame_id)

    @staticmethod
    def _sigma(sigma: float | Sequence[float], size: int) -> np.ndarray:
        values = np.asarray(sigma, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(size, float(values))
        else:
            values = values.reshape(-1)
        if values.shape != (size,) or np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"sigma must contain {size} finite positive values")
        return values

    def _check_live(self, frame_ids: tuple[int, ...]) -> None:
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("one factor cannot repeat a frame_id")
        archived = set(frame_ids).intersection(self._archived)
        if archived:
            raise ValueError(f"cross-window link to archived frame(s): {sorted(archived)}")

    def add_prior(
        self, frame_id: int, state: Sequence[float], sigma: float | Sequence[float]
    ) -> None:
        frame_id = self._frame_id(frame_id)
        self._check_live((frame_id,))
        measurement = self._vector(state, name="prior state")
        noise = gtsam.noiseModel.Diagonal.Sigmas(self._sigma(sigma, self.state_dim))
        self._pending.append(gtsam.PriorFactorVector(self._key(frame_id), measurement, noise))
        self._pending_ids.add(frame_id)
        self._prior_initials.setdefault(frame_id, measurement)

    def add_between(
        self,
        previous_frame_id: int,
        frame_id: int,
        delta: Sequence[float],
        sigma: float | Sequence[float],
    ) -> None:
        pair = (self._frame_id(previous_frame_id), self._frame_id(frame_id))
        self._check_live(pair)
        delta_vector = self._vector(delta, name="between delta")
        noise = gtsam.noiseModel.Diagonal.Sigmas(self._sigma(sigma, self.state_dim))
        self._pending.append(
            gtsam.BetweenFactorVector(self._key(pair[0]), self._key(pair[1]), delta_vector, noise)
        )
        self._pending_ids.update(pair)

    def add_observation(
        self,
        keys: Sequence[int],
        residual_fn: ResidualFn,
        *,
        residual_dim: int,
        sigma: float | Sequence[float],
        jac_fn: JacobianFn | None = None,
        huber_k: float | None = None,
        fd_step: float = 1e-5,
    ) -> None:
        ids = tuple(self._frame_id(item) for item in keys)
        if not ids or residual_dim < 1 or fd_step <= 0:
            raise ValueError("observation needs keys, residual_dim > 0 and fd_step > 0")
        self._check_live(ids)
        if not callable(residual_fn) or (jac_fn is not None and not callable(jac_fn)):
            raise TypeError("residual_fn and jac_fn must be callable")
        base = gtsam.noiseModel.Diagonal.Sigmas(self._sigma(sigma, residual_dim))
        if huber_k is not None:
            if not np.isfinite(huber_k) or huber_k <= 0:
                raise ValueError("huber_k must be finite and positive")
            base = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(float(huber_k)), base
            )
        spec = _Observation(ids, residual_fn, jac_fn, base, int(residual_dim), float(fd_step))
        keys = [self._key(item) for item in ids]
        state_dim = self.state_dim

        def evaluate(
            _factor: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians: list[np.ndarray] | None,
        ) -> np.ndarray:
            states = [np.asarray(values.atVector(key), dtype=np.float64).copy() for key in keys]
            residual = np.asarray(spec.residual_fn(states), dtype=np.float64).reshape(-1)
            if residual.shape != (spec.residual_dim,) or not np.all(np.isfinite(residual)):
                raise ValueError("observation residual changed size or is nonfinite")
            if jacobians is not None:
                if spec.jac_fn is not None:
                    matrices = spec.jac_fn(states)
                else:
                    matrices = IncrementalCameraGraph._finite_difference(spec, states, state_dim)
                if len(matrices) != len(keys):
                    raise ValueError("one Jacobian matrix is required per observation state")
                for index, matrix in enumerate(matrices):
                    matrix = np.asarray(matrix, dtype=np.float64)
                    if matrix.shape != (spec.residual_dim, state_dim) or not np.all(
                        np.isfinite(matrix)
                    ):
                        raise ValueError("observation Jacobian has wrong shape or nonfinite values")
                    jacobians[index] = np.asfortranarray(matrix)
            return residual

        self._callbacks.append((ids, evaluate))  # Keep callbacks for active custom factors alive.
        self._pending.append(self._custom_factor(spec.noise, keys, evaluate))
        self._pending_ids.update(ids)

    @staticmethod
    def _finite_difference(
        spec: _Observation, states: list[State], state_dim: int
    ) -> list[np.ndarray]:
        matrices: list[np.ndarray] = []
        for state_index, state in enumerate(states):
            jacobian = np.empty((spec.residual_dim, state_dim), dtype=np.float64)
            for component in range(state_dim):
                step = spec.fd_step * max(1.0, abs(float(state[component])))
                plus = [item.copy() for item in states]
                minus = [item.copy() for item in states]
                plus[state_index][component] += step
                minus[state_index][component] -= step
                high = np.asarray(spec.residual_fn(plus), dtype=np.float64).reshape(-1)
                low = np.asarray(spec.residual_fn(minus), dtype=np.float64).reshape(-1)
                if high.shape != (spec.residual_dim,) or low.shape != (spec.residual_dim,):
                    raise ValueError("observation residual changed size during finite differences")
                jacobian[:, component] = (high - low) / (2 * step)
            matrices.append(jacobian)
        return matrices

    def _phase_start(self) -> tuple[float, float] | None:
        return (time.perf_counter(), time.process_time()) if self._phase_profile else None

    def _phase_end(self, name: str, start: tuple[float, float] | None) -> None:
        if start is None:
            return
        row: dict[str, float | int | str] = {
            "phase": name,
            "wall_seconds": time.perf_counter() - start[0],
            "cpu_seconds": time.process_time() - start[1],
        }
        self.last_update_phase_timings.append(row)
        print(json.dumps({"gtsam_isam_phase": row}, separators=(",", ":")),
              file=sys.stderr, flush=True)

    def _print_tree_stats(self, stage: str) -> None:
        if not self._phase_profile:
            return
        # GTSAM writes printStats() to C++ stdout. Redirect only that call to
        # stderr; printStats reports clique-size statistics, not the whole tree.
        marker: dict[str, int | str] = {"stage": stage, "pid": os.getpid()}
        isam = self._smoother.getISAM2()
        try:
            marker["linearization_point_variables"] = int(isam.getLinearizationPoint().size())
            marker["factor_slots"] = int(isam.getFactorsUnsafe().size())
        except Exception as exc:
            marker["counts_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"gtsam_isam_tree_stats_begin": marker}, separators=(",", ":")),
              file=sys.stderr, flush=True)
        saved_stdout = None
        try:
            sys.stdout.flush()
            saved_stdout = os.dup(1)
            os.dup2(2, 1)
            isam.printStats()
            sys.stdout.flush()
        except Exception as exc:
            print(json.dumps({"gtsam_isam_tree_stats_error": {
                **marker, "error": f"{type(exc).__name__}: {exc}"}}, separators=(",", ":")),
                  file=sys.stderr, flush=True)
        finally:
            if saved_stdout is not None:
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)
            print(json.dumps({"gtsam_isam_tree_stats_end": marker}, separators=(",", ":")),
                  file=sys.stderr, flush=True)

    def update(self, initial_states: Mapping[int, Sequence[float]] | None = None) -> dict[int, State]:
        """Submit pending factors and new state guesses to GTSAM iSAM2.

        Call once per source frame.  The returned mapping contains the active
        estimates; archived estimates remain accessible through ``estimate``.
        """
        initial_states = initial_states or {}
        guesses = {self._frame_id(k): self._vector(v) for k, v in initial_states.items()}
        unseen = self._pending_ids.difference(self._active)
        extras = set(guesses).difference(unseen)
        if extras:
            raise ValueError(f"initial guesses supplied for existing/unconnected frames: {sorted(extras)}")
        missing = unseen.difference(guesses).difference(self._prior_initials)
        if missing:
            raise ValueError(f"missing initial guesses for frames: {sorted(missing)}")
        if not self._pending and not guesses:
            return self.active_estimates()
        self.last_retired_estimates = {}
        graph = gtsam.NonlinearFactorGraph()
        for factor in self._pending:
            graph.add(factor)
        values = gtsam.Values()
        timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
        new_ids = sorted(unseen)
        # A batch can move the fixed-lag boundary by dozens of frames. GTSAM
        # marginalizes those old keys inside update(), before Python can read
        # their newly optimized estimates. Stage new keys at the current time
        # first, then advance their clocks in an empty update after capturing
        # the estimate. A one-frame update needs the same treatment once the
        # window is full, though callers normally batch native video frames.
        latest_new_time = self._next_time + len(new_ids) - 1
        will_trim = bool(new_ids) and latest_new_time >= self.window_size
        provisional_time = float(max(self._next_time - 1, 0))
        for offset, frame_id in enumerate(new_ids):
            key = self._key(frame_id)
            values.insert(key, guesses.get(frame_id, self._prior_initials.get(frame_id)))
            timestamps.insert((key, provisional_time if will_trim else float(self._next_time + offset)))
        self.last_update_diagnostics = []
        self.last_update_summary = {}
        self.last_batch_diagnostics = {}
        self.last_update_phase_timings = []
        if self._diagnose_cost and not self._active:
            phase_start = self._phase_start()
            self.last_update_diagnostics.append({
                "refinement": -1, "nonlinear_cost": float(graph.error(values)),
                "max_state_change_l2": None, "max_rotation_change_rad": None,
                "max_logf_change": None,
            })
            self._phase_end("initial_graph_error", phase_start)
            print(json.dumps({"gtsam_refinement_progress": self.last_update_diagnostics[-1]},
                             separators=(",", ":")), file=sys.stderr, flush=True)
        phase_start = self._phase_start()
        self._smoother.update(graph, values, timestamps)
        self._phase_end("initial_smoother_update", phase_start)
        self._print_tree_stats("after_initial_update")
        # Include the penultimate update so the last observed state step and
        # cost change are genuinely consecutive, rather than e.g. 16 -> 30.
        sample_indices = {0, 1, 2, 4, 8, 16,
                          max(0, self.refinement_updates - 1), self.refinement_updates}
        previous_diagnostic = None

        def record(refinement: int) -> None:
            nonlocal previous_diagnostic
            if not self._diagnose_cost or refinement not in sample_indices:
                return
            phase_start = self._phase_start()
            estimate = self._smoother.calculateEstimate()
            self._phase_end(f"record_{refinement}_calculate_estimate", phase_start)
            self._print_tree_stats(f"after_record_{refinement}_estimate")
            factors = self._smoother.getISAM2().getFactorsUnsafe()
            common = [] if previous_diagnostic is None else [
                frame_id for frame_id in self._active + sorted(unseen)
                if estimate.exists(self._key(frame_id))
                and previous_diagnostic.exists(self._key(frame_id))
            ]
            if common:
                differences = [np.asarray(estimate.atVector(self._key(frame_id)))
                               - np.asarray(previous_diagnostic.atVector(self._key(frame_id)))
                               for frame_id in common]
                max_step = float(max(np.linalg.norm(item) for item in differences))
                max_rotation = (float(max(np.linalg.norm(item[3:6]) for item in differences))
                                if self.state_dim >= 6 else None)
                max_focal = (float(max(np.max(np.abs(item[6:8])) for item in differences))
                             if self.state_dim >= 8 else None)
            else:
                max_step = max_rotation = max_focal = None
            phase_start = self._phase_start()
            cost = float(factors.error(estimate))
            self._phase_end(f"record_{refinement}_factor_error", phase_start)
            self.last_update_diagnostics.append({
                "refinement": refinement,
                "nonlinear_cost": cost,
                "max_state_change_l2": max_step,
                "max_rotation_change_rad": max_rotation,
                "max_logf_change": max_focal,
            })
            print(json.dumps({"gtsam_refinement_progress": self.last_update_diagnostics[-1]},
                             separators=(",", ":")), file=sys.stderr, flush=True)
            previous_diagnostic = estimate

        record(0)
        for refinement in range(1, self.refinement_updates + 1):
            phase_start = self._phase_start()
            self._smoother.update(
                gtsam.NonlinearFactorGraph(),
                gtsam.Values(),
                gtsam_unstable.FixedLagSmootherKeyTimestampMap(),
            )
            self._phase_end(f"refinement_{refinement}_smoother_update", phase_start)
            self._print_tree_stats(f"after_refinement_{refinement}_update")
            record(refinement)
        phase_start = self._phase_start()
        pretrim_values = self._smoother.calculateEstimate()
        self._phase_end("pretrim_calculate_estimate", phase_start)
        self._print_tree_stats("after_pretrim_estimate")
        if will_trim:
            real_timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
            for offset, frame_id in enumerate(new_ids):
                real_timestamps.insert((self._key(frame_id), float(self._next_time + offset)))
            phase_start = self._phase_start()
            self._smoother.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                                  real_timestamps)
            self._phase_end("trim_smoother_update", phase_start)
            self._print_tree_stats("after_trim_update")
        self._next_time += len(new_ids)
        if self._diagnose_cost:
            samples = self.last_update_diagnostics
            refinements = [row for row in samples if row["refinement"] >= 0]
            first = refinements[0]
            final = refinements[-1]
            previous = refinements[-2] if len(refinements) >= 2 else None
            initial = samples[0] if samples[0]["refinement"] == -1 else None
            initial_cost = None if initial is None else float(initial["nonlinear_cost"])
            first_cost = float(first["nonlinear_cost"])
            final_cost = float(final["nonlinear_cost"])
            previous_cost = None if previous is None else float(previous["nonlinear_cost"])
            self.last_update_summary = {
                "initial_nonlinear_cost": initial_cost,
                "post_insert_nonlinear_cost": first_cost,
                "final_nonlinear_cost": final_cost,
                "refinement_updates": self.refinement_updates,
                "relative_reduction_from_insert":
                    (first_cost - final_cost) / max(abs(first_cost), 1e-12),
                "last_step_relative_cost_reduction":
                    None if previous_cost is None else
                    (previous_cost - final_cost) / max(abs(previous_cost), 1e-12),
                "last_step_max_state_change_l2": final["max_state_change_l2"],
            }
            print(json.dumps({"gtsam_refinement": self.last_update_summary},
                             separators=(",", ":")), file=sys.stderr, flush=True)
        phase_start = self._phase_start()
        current_values = self._smoother.calculateEstimate()
        self._phase_end("current_calculate_estimate", phase_start)
        self._print_tree_stats("after_current_estimate")
        phase_start = self._phase_start()
        all_ids = self._active + new_ids
        marginalised = [frame_id for frame_id in self._active if not current_values.exists(self._key(frame_id))]
        self.last_retired_estimates = {
            frame_id: np.asarray(pretrim_values.atVector(self._key(frame_id))).copy()
            for frame_id in marginalised
        }
        self._archived.update(self.last_retired_estimates)
        self._active = [frame_id for frame_id in all_ids if current_values.exists(self._key(frame_id))]
        self._latest = {
            frame_id: np.asarray(current_values.atVector(self._key(frame_id))).copy()
            for frame_id in self._active
        }
        self._callbacks = [
            (ids, callback)
            for ids, callback in self._callbacks
            if all(frame_id in self._active for frame_id in ids)
        ]
        if marginalised:
            self._window_count += 1
        self._pending.clear()
        self._pending_ids.clear()
        self._prior_initials.clear()
        result = self.active_estimates()
        self._phase_end("python_finalize", phase_start)
        return result

    def batch_polish(
        self,
        *,
        max_iterations: int = 20,
        relative_error_tol: float = 1e-6,
        diagonal_damping: bool = False,
        lambda_upper_bound: float | None = None,
    ) -> dict[int, State]:
        """Optionally refine the current iSAM2 estimate with GTSAM LM.

        This reads the exact active iSAM2 factor graph, including its priors,
        custom source-pixel factors and robust noise models.  It does not
        mutate the smoother; callers explicitly choose whether to use the
        returned states.  A failed or worse batch result returns the original
        iSAM2 estimates and records the reason in ``last_batch_diagnostics``.
        Intended for a completed, untrimmed joint graph, not a replay window.
        """
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(relative_error_tol) or relative_error_tol <= 0:
            raise ValueError("relative_error_tol must be finite and positive")
        if not isinstance(diagonal_damping, (bool, np.bool_)):
            raise TypeError("diagonal_damping must be boolean")
        if (lambda_upper_bound is not None
                and (not np.isfinite(lambda_upper_bound) or lambda_upper_bound <= 0)):
            raise ValueError("lambda_upper_bound must be finite and positive")
        if not self._active:
            raise ValueError("batch_polish requires an updated graph")
        if self._window_count or self._archived or self._next_time != len(self._active):
            raise ValueError("batch_polish requires an untrimmed graph")

        factors = self._smoother.getISAM2().getFactorsUnsafe()
        initial = self._smoother.calculateEstimate()
        initial_cost = float(factors.error(initial))
        if not np.isfinite(initial_cost):
            raise ValueError("iSAM2 estimate has nonfinite graph cost")
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        params.setRelativeErrorTol(float(relative_error_tol))
        params.setDiagonalDamping(bool(diagonal_damping))
        if lambda_upper_bound is not None:
            if lambda_upper_bound < params.getlambdaInitial():
                raise ValueError("lambda_upper_bound must be at least the initial lambda")
            params.setlambdaUpperBound(float(lambda_upper_bound))
        diagnostics: dict[str, float | int | bool | str | None] = {
            "method": "gtsam_levenberg_marquardt",
            "factor_count": int(factors.nrFactors()),
            "state_count": len(self._active),
            "initial_nonlinear_cost": initial_cost,
            "final_nonlinear_cost": None,
            "iterations": 0,
            "diagonal_damping": bool(params.getDiagonalDamping()),
            "lambda_initial": float(params.getlambdaInitial()),
            "lambda_lower_bound": float(params.getlambdaLowerBound()),
            "lambda_upper_bound": float(params.getlambdaUpperBound()),
            "lambda_final": None,
            "accepted": False,
            "status": "not_run",
            "max_state_change_l2": None,
        }
        self.last_batch_diagnostics = diagnostics
        optimizer = None
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(factors, initial, params)
            polished = optimizer.optimize()
            diagnostics["iterations"] = int(optimizer.iterations())
            diagnostics["lambda_final"] = float(optimizer.lambda_())
            final_cost = float(factors.error(polished))
            diagnostics["final_nonlinear_cost"] = final_cost if np.isfinite(final_cost) else None
            states = {
                frame_id: self._vector(polished.atVector(self._key(frame_id)))
                for frame_id in self._active
            }
            diagnostics["max_state_change_l2"] = float(max(
                np.linalg.norm(state - np.asarray(initial.atVector(self._key(frame_id))))
                for frame_id, state in states.items()
            ))
        except Exception as exc:
            if optimizer is not None:
                diagnostics["iterations"] = int(optimizer.iterations())
                diagnostics["lambda_final"] = float(optimizer.lambda_())
            diagnostics["status"] = "optimizer_error"
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return self.active_estimates()

        if not np.isfinite(final_cost):
            diagnostics["status"] = "nonfinite_cost"
            return self.active_estimates()
        if final_cost > initial_cost + 1e-9 * max(1.0, abs(initial_cost)):
            diagnostics["status"] = "cost_increased"
            return self.active_estimates()
        if final_cost >= initial_cost:
            diagnostics["status"] = "unchanged"
            return self.active_estimates()
        diagnostics["accepted"] = True
        diagnostics["status"] = "improved"
        return states

    def estimate(self, frame_id: int) -> State:
        frame_id = self._frame_id(frame_id)
        if frame_id in self._archived:
            return self._archived[frame_id].copy()
        if frame_id not in self._active:
            raise KeyError(frame_id)
        return self._latest[frame_id].copy()

    def active_estimates(self) -> dict[int, State]:
        return {frame_id: value.copy() for frame_id, value in self._latest.items()}

    @property
    def active_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._active)

    @property
    def archived_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._archived)

    @property
    def window_count(self) -> int:
        return self._window_count

    def marginal_covariance(self, frame_id: int) -> np.ndarray:
        frame_id = self._frame_id(frame_id)
        if frame_id not in self._active:
            raise KeyError("covariance is available only for active frames")
        return np.asarray(self._smoother.getISAM2().marginalCovariance(self._key(frame_id))).copy()
