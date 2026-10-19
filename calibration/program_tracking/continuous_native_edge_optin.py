"""Opt-in production bridge for complete native source-edge factors.

Only construction and unsupported-factor replacement run in Python. The
registered C++ factor evaluates residuals/Jacobians without a Python callback.
The solver must rebuild its GTSAM graph and optimizer after replacement.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
import hashlib
import importlib
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np


DEFAULT_BINARY_DIR = Path(
    "D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/round6_source_edge_prod"
)


class NativeEdgeUnsupported(RuntimeError):
    """Status 2: retry this entire factor via its original Python callbacks."""

    def __init__(self, factor_id: int, source_pts: int, stage: str, status: int = 2):
        self.factor_id = int(factor_id)
        self.source_pts = int(source_pts)
        self.evaluation_stage = str(stage)
        self.factor_family = "source_edge"
        self.source_keys = (self.source_pts,)
        self.status = int(status)
        self.reason = "native_status_2"
        super().__init__(
            f"native source-edge status {status}: factor_id={factor_id}, "
            f"source_pts={source_pts}, stage={stage}"
        )


@dataclass
class _NativeRecord:
    factor_id: int
    source_pts: int
    keys: tuple[int, ...]
    residual: object
    derivative: object
    options: dict
    factor: object
    spec: dict
    noise: object
    observed_sha256: str
    quality_sha256: str
    world_sha256: str
    fallback: bool = False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class NativeEdgeManager:
    """Own the opt-in plugin and stable pending-factor identities.

    ``states`` contains the construction-time Vector11 values for every rig
    dependency; ``active_keys`` is exactly the graph's initial key set. The
    caller keeps original adapted callbacks in the captured specification.
    """

    def __init__(self, model, states, active_keys, *, threads: int = 2,
                 cache: bool = False, binary_dir: Path | str | None = None,
                 module_name: str = 'native_source_edge_prod'):
        if int(threads) != threads or threads < 1:
            raise ValueError("Native source-edge threads must be a positive integer")
        self.model = model
        self.states = states
        self.active_keys = set(map(int, active_keys))
        self.threads = int(threads)
        self.cache = bool(cache)
        self.binary_dir = Path(binary_dir or DEFAULT_BINARY_DIR).resolve()
        if module_name not in ('native_source_edge_prod','native_source_edge_trial_prod'):
            raise ValueError('Unknown native source-edge module')
        self.module_name = module_name
        self._owner_thread = threading.get_ident()
        self._records: dict[int, _NativeRecord] = {}
        self._fallbacks: list[dict] = []
        self._counts = Counter()
        self._dll_handle = None
        self._closed = False
        module_file = self.binary_dir / (self.module_name + '.pyd')
        kernel_file = self.binary_dir / "source_edge_native.dll"
        if not module_file.is_file() or not kernel_file.is_file():
            raise FileNotFoundError("Production source-edge PYD/DLL not compiled")
        self.binary_sha256 = _sha(module_file)
        self.kernel_sha256 = _sha(kernel_file)
        import gtsam  # Register GTSAM's Boost types before importing pybind factor.
        self.gtsam = gtsam
        if sys.platform == "win32":
            self._dll_handle = os.add_dll_directory(str(self.binary_dir))
        if str(self.binary_dir) not in sys.path:
            sys.path.insert(0, str(self.binary_dir))
        self.native = importlib.import_module(self.module_name)
        if Path(self.native.__file__).resolve() != module_file.resolve():
            raise RuntimeError("Loaded native source-edge PYD differs from hashed binary")
        self.native.set_threads(self.threads)

    def _main_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Native source-edge fallback must run on the owner Python thread")
        if self._closed:
            raise RuntimeError("Native source-edge manager has been closed")

    def _eligible(self, captured_spec, adapted, source_keys):
        import continuous_central_edge_optin as edge

        interval = edge.interval_replacement_for_spec(captured_spec)
        if interval is not None:
            self._counts["ineligible_source_interval_factors"] += 1
            self._counts["ineligible_source_interval_uv"] += interval["original_uv_count"]
            return None
        guard = edge.guard_for_spec(captured_spec)
        if guard is None:
            return None
        pts = int(guard.source_pts)
        keys, residual, derivative, options = adapted
        source_keys = tuple(map(int, source_keys))
        if (source_keys != (pts,) or tuple(map(int, captured_spec["keys"])) != (pts,)
                or derivative is None):
            self._counts["ineligible_keys_or_jacobian"] += 1
            return None
        dependencies = tuple(map(int, self.model.dependencies(pts)))
        expected = tuple(key for key in dependencies if key in self.active_keys)
        if tuple(map(int, keys)) != expected or not expected:
            self._counts["ineligible_dependency_order"] += 1
            return None
        observed = np.asarray(guard.observed, np.float64)
        quality = np.asarray(guard.scales, np.float64)
        if (observed.ndim != 2 or observed.shape[1] != 2 or not len(observed)
                or quality.shape != (len(observed),)
                or _array_sha(observed) != guard.observed_sha256
                or _array_sha(quality) != guard.quality_sha256):
            raise ValueError("Registered source UV or quality provenance changed")
        if (options.get("residual_dim") != 2 * len(observed)
                or options.get("huber_k") is not None
                or set(options) - {"residual_dim", "sigma", "huber_k", "fd_step"}):
            self._counts["ineligible_noise_options"] += 1
            return None
        try:
            sigma = float(options["sigma"])
        except (TypeError, ValueError, KeyError):
            self._counts["ineligible_noise_options"] += 1
            return None
        if (not np.isfinite(sigma) or sigma <= 0
                or sigma != float(guard.sigma)
                or np.any(quality < 1.0) or not np.isfinite(quality).all()):
            self._counts["ineligible_noise_options"] += 1
            return None
        if not hasattr(self.model, "brown_keys") or not self.model.brown_keys:
            self._counts["ineligible_no_brown15"] += 1
            return None
        if captured_spec.get("kind") != "observation":
            self._counts["ineligible_capture_kind"] += 1
            return None
        return guard

    def _spec(self, pts, guard, factor_id):
        gtsam = self.gtsam
        model = self.model
        binding = model.frames[pts]
        group = binding.camera_id
        dependencies = tuple(map(int, model.dependencies(pts)))
        index = {key: i for i, key in enumerate(dependencies)}
        if len(index) != len(dependencies):
            raise ValueError("Rig dependencies contain duplicate keys")
        frozen = np.ascontiguousarray(np.stack(
            [np.asarray(self.states[key], np.float64) for key in dependencies]
        ))
        if frozen.shape != (len(dependencies), 11) or not np.isfinite(frozen).all():
            raise ValueError("Native edge requires finite frozen Vector11 states")
        spline = model.shot_splines[binding.shot_id]
        canonical = binding.pts if binding.spline_pts is None else binding.spline_pts
        roles = {
            "frame": index[binding.frame_key],
            "controls": [(index[key], float(weight))
                         for key, weight in spline.basis_pairs(canonical)],
            "lens_base": index[model.lens_base_keys[group]],
            "lens_slope": index[model.lens_slope_keys[group]]
                if group in model.lens_slope_keys else -1,
            "mount": index[model.mount_keys[group]],
            "offset": index[model.offset_keys[group]],
            "offset_slope": index[model.offset_slope_keys[group]],
            "brown": index[model.brown_keys[group]],
            "brown_slope": index[model.brown_slope_keys[group]]
                if group in model.brown_slope_keys else -1,
        }
        kept = tuple(key for key in dependencies if key in self.active_keys)
        return dict(
            factor_id=int(factor_id), source_pts=int(pts),
            dependency_keys=[gtsam.symbol("x", key) for key in dependencies],
            active_keys=[gtsam.symbol("x", key) for key in kept],
            frozen_states=frozen, roles=roles,
            zoom_reference=float(model.lens_zoom_reference[group]),
            world_xyz=np.ascontiguousarray(guard.world, dtype=np.float64),
            observed_uv=np.ascontiguousarray(guard.observed, dtype=np.float64),
            quality_scales=np.ascontiguousarray(guard.scales, dtype=np.float64),
            frozen_segments=np.ascontiguousarray(guard.frozen, dtype=np.int64),
            sigma=float(guard.sigma), dynamic_candidates=bool(guard.dynamic),
            fixed_mount=False,
            fixed_center_xyz=np.ascontiguousarray(guard.template[:3], dtype=np.float64),
            central_domain=True, cache=self.cache,
        )

    def maybe_add(self, graph, captured_spec, adapted, source_keys) -> bool:
        """Append a native factor only when exact source/rig semantics are known."""
        self._main_thread()
        guard = self._eligible(captured_spec, adapted, source_keys)
        if guard is None:
            return False
        keys, residual, derivative, options = adapted
        keys = tuple(map(int, keys))
        graph._check_live(keys)
        factor_id = len(graph._pending)
        spec = self._spec(int(guard.source_pts), guard, factor_id)
        noise = self.gtsam.noiseModel.Diagonal.Sigmas(
            graph._sigma(options["sigma"], options["residual_dim"]))
        holder = self.gtsam.NonlinearFactorGraph()
        returned = self.native.add_factor(holder, noise, spec)
        if returned != 0 or holder.size() != 1:
            raise RuntimeError("Native source-edge construction changed factor count")
        factor = holder.at(0)
        if tuple(int(key) for key in factor.keys()) != tuple(spec["active_keys"]):
            raise RuntimeError("Native source-edge key order differs from adapted factor")
        graph._pending.append(factor)
        graph._pending_ids.update(keys)
        self._records[factor_id] = _NativeRecord(
            factor_id, int(guard.source_pts), keys, residual, derivative,
            dict(options), factor, spec, noise, guard.observed_sha256,
            guard.quality_sha256, _array_sha(np.asarray(guard.world, np.float64)),
        )
        self._counts["registered_native"] += 1
        self._counts["registered_observed_uv"] += len(guard.observed)
        return True

    def replace_unsupported(self, graph, error: NativeEdgeUnsupported) -> bool:
        """Replace one complete factor in the same pending slot on main thread."""
        self._main_thread()
        if not isinstance(error, NativeEdgeUnsupported) or error.status != 2:
            raise TypeError("Only native status 2 supports Python fallback")
        record = self._records.get(error.factor_id)
        if (record is None or record.fallback or record.source_pts != error.source_pts
                or error.evaluation_stage not in ("residual", "jacobian")
                or error.source_keys != (record.source_pts,)):
            raise RuntimeError("Unknown, repeated or inconsistent native factor fallback") from error
        before = len(graph._pending)
        if error.factor_id >= before or graph._pending[error.factor_id] is not record.factor:
            raise RuntimeError("Native factor slot changed before fallback") from error
        graph.add_observation(record.keys, record.residual,
                              jac_fn=record.derivative, **record.options)
        if len(graph._pending) != before + 1:
            raise RuntimeError("Python fallback did not add exactly one factor")
        replacement = graph._pending.pop()
        if tuple(int(key) for key in replacement.keys()) != tuple(
                graph._key(key) for key in record.keys):
            raise RuntimeError("Python fallback changed factor key order")
        graph._pending[error.factor_id] = replacement
        record.factor = None
        record.fallback = True
        self._fallbacks.append(dict(
            factor_id=error.factor_id, source_pts=record.source_pts,
            source_keys=list(error.source_keys), stage=error.evaluation_stage,
            observed_sha256=record.observed_sha256,
            quality_sha256=record.quality_sha256,
            world_sha256=record.world_sha256,
            reason=str(error.reason),
        ))
        self._counts["fallback_complete_factor"] += 1
        return True

    def verify_accepted(self, graph, states, *, residual_atol: float = 1e-6,
                        residual_rtol: float = 1e-9):
        """Compare *every* live native factor against original Python residual.

        Call on an accepted candidate before committing it. Any disagreement,
        including Python's physical-domain rejection, returns a status-2
        request to replace the entire factor in the owner thread. A native
        status-4 physical rejection is deliberately allowed to propagate.
        No residual or Jacobian from this verification is cached in the graph.
        """
        self._main_thread()
        from reference_board_pair import BoardIntersectionError
        from projection_reuse import ProjectionWorld, reuse_verified_projection
        import continuous_central_edge_optin as edge

        if residual_atol < 0 or residual_rtol < 0:
            raise ValueError("Parity tolerances must be nonnegative")
        start = time.perf_counter()
        values = self.gtsam.Values()
        live = [r for r in self._records.values() if not r.fallback]
        required = {key for record in live for key in record.keys}
        for key in sorted(required):
            vector = np.asarray(states[key], np.float64).reshape(-1)
            if vector.shape != (11,) or not np.isfinite(vector).all():
                raise ValueError(f"Accepted state {key} is not finite Vector11")
            values.insert(graph._key(key), vector)
        failures = []
        counts = Counter()
        maximum_difference = 0.0
        maximum_projection_difference = 0.0
        projection_points = 0
        ambiguous_unselected_points = 0
        geometry_cache = {}
        projection_cache = {}
        reused_projection_factors = 0
        last_source_pts = None
        physical = camera = None
        for record in live:
            factor_id = record.factor_id
            if (factor_id >= len(graph._pending)
                    or graph._pending[factor_id] is not record.factor):
                raise RuntimeError("Accepted parity factor slot changed")
            # The final curve residual alone cannot certify non-winning world
            # segments or the <=1e-5 px OpenCV-vs-safe projection predicate.
            # Audit every captured endpoint and midpoint at this exact state.
            world = record.spec["world_xyz"]
            geometry_key = (world.shape, np.ascontiguousarray(world, np.float64).tobytes())
            geometry = geometry_cache.get(geometry_key)
            if geometry is None:
                geometry = ProjectionWorld(np.vstack((
                    world, .5 * (world[:-1] + world[1:]))))
                # Different source factors normally share two world curves.
                # Bound retention even if a caller supplies many unique curves.
                if len(geometry_cache) >= 8:
                    geometry_cache.pop(next(iter(geometry_cache)))
                geometry_cache[geometry_key] = geometry
            all_world = geometry.world
            projection_points += len(all_world)
            try:
                if record.source_pts != last_source_pts:
                    physical = self.model.physical_state(record.source_pts, states)
                    camera = self.model.camera25(record.source_pts, states)
                    last_source_pts = record.source_pts
                signature = (np.asarray(physical, np.float64).tobytes(),
                             np.asarray(camera, np.float64).tobytes(), geometry_key)
                projected = projection_cache.get(signature)
                if projected is None:
                    native_uv, native_depth, native_status = self.native.project_probe(
                        np.ascontiguousarray(physical, np.float64), all_world)
                    python_uv, python_depth = edge._project15(camera, all_world)
                    python_mask = edge._central_mask(
                        camera, all_world, raw_uv=python_uv,
                        raw_depth=python_depth)
                    projected = (native_uv, native_depth, native_status,
                                 python_uv, python_depth, python_mask)
                    if len(projection_cache) >= 8:
                        projection_cache.pop(next(iter(projection_cache)))
                    projection_cache[signature] = projected
                else:
                    reused_projection_factors += 1
                (native_uv, native_depth, native_status,
                 python_uv, python_depth, python_mask) = projected
                native_uv = np.asarray(native_uv, np.float64)
                native_depth = np.asarray(native_depth, np.float64)
                native_status = np.asarray(native_status, np.int32)
                ambiguous_unselected_points += int(np.sum(native_status == 2))
                # Probe status 2 is deliberately undecided. Its provisional UV
                # is not certified and must not reject an otherwise equivalent
                # factor when this world point is outside its candidate set.
                # rs_edge_eval rejects the complete factor if such a point is
                # used; the complete original residual is also checked below.
                comparable = python_mask & (native_status == 1)
                projection_difference = np.linalg.norm(
                    native_uv[comparable] - python_uv[comparable], axis=1)
                if len(projection_difference):
                    maximum_projection_difference = max(
                        maximum_projection_difference,
                        float(projection_difference.max()))
                projection_ok = (
                    native_status.shape == (len(all_world),)
                    # Status 2 on a world segment not considered by the
                    # factor is immaterial at this state; if it enters the
                    # candidate set, rs_edge_eval itself returns status 2.
                    and np.array_equal((native_status == 1)[native_status != 2],
                                       python_mask[native_status != 2])
                    and np.all(np.isfinite(native_uv[comparable]))
                    and np.all(np.isfinite(native_depth[comparable]))
                    and np.allclose(native_depth[comparable],
                                    python_depth[comparable], atol=1e-8, rtol=1e-10)
                    and np.all(projection_difference <= 1e-5)
                )
            except (ValueError, FloatingPointError, OverflowError,
                    edge.cv2.error, BoardIntersectionError):
                projection_ok = False
            if not projection_ok:
                reason = "projection_or_central_mask_mismatch"
                counts[reason] += 1
                exc = NativeEdgeUnsupported(factor_id, record.source_pts,
                                            "residual")
                exc.reason = reason
                failures.append(exc)
                continue
            try:
                with reuse_verified_projection(camera, geometry, python_uv,
                                               python_depth, python_mask):
                    original = np.asarray(record.residual(
                        [np.asarray(states[key], np.float64).copy() for key in record.keys]
                    ), np.float64).reshape(-1)
            except BoardIntersectionError:
                reason = "python_physical_domain_rejection_native_still_present"
                counts[reason] += 1
                exc = NativeEdgeUnsupported(factor_id, record.source_pts,
                                            "residual")
                exc.reason = reason
                failures.append(exc)
                continue
            native_failed = None
            try:
                got, _ = self.native.evaluate_factor(
                    record.noise, record.spec, values, False)
            except NativeEdgeUnsupported as error:
                # This routine already owns Values; do not retain a full copy
                # on each implementation-fallback item in a large batch.
                error.trial_values = None
                error.__traceback__ = None
                native_failed = error
            if native_failed is not None:
                counts["native_status_2"] += 1
                failures.append(native_failed)
                continue
            got = np.asarray(got, np.float64).reshape(-1)
            if (original.shape != got.shape or not np.isfinite(original).all()
                    or not np.isfinite(got).all()):
                reason = "nonfinite_or_shape_mismatch"
            else:
                difference = np.abs(original - got)
                maximum_difference = max(maximum_difference,
                                         float(difference.max(initial=0.0)))
                reason = ("residual_mismatch" if not np.allclose(
                    original, got, atol=residual_atol, rtol=residual_rtol)
                          else "")
            if reason:
                counts[reason] += 1
                exc = NativeEdgeUnsupported(factor_id, record.source_pts,
                                            "residual")
                exc.reason = reason
                failures.append(exc)
            else:
                counts["parity_match"] += 1
        result = dict(
            evaluated_native_factors=len(live),
            observed_uv=sum(len(r.spec["observed_uv"]) for r in live),
            reasons=dict(counts), maximum_residual_abs_difference=maximum_difference,
            checked_world_endpoints_and_midpoints=projection_points,
            probe_status2_on_non_candidate_world_points=ambiguous_unselected_points,
            maximum_native_vs_opencv_uv_px=maximum_projection_difference,
            exact_projection_reuse_factors=reused_projection_factors,
            elapsed_seconds=time.perf_counter() - start,
            tolerance=dict(atol=residual_atol, rtol=residual_rtol),
        )
        self._counts["accepted_parity_passes"] += 1
        self._counts["accepted_parity_factor_checks"] += len(live)
        self._counts["accepted_parity_fallback_requests"] += len(failures)
        return failures, result

    def metadata(self) -> dict:
        self._main_thread()
        source = Path(__file__).resolve()
        own_cpp = source.parent / "native" / (
            "round6_source_edge_trial" if self.module_name ==
            "native_source_edge_trial_prod" else "round6_source_edge")
        return dict(
            schema="rinkspace.native-source-edge-optin.v1",
            source_sha256=_sha(source),
            cpp_sha256={name: _sha(own_cpp / name) for name in (
                "native_source_edge_prod.cpp", "source_edge_native_prod.cpp",
                "brown15_projection.hpp")},
            module_name=self.module_name,
            binary_dir=str(self.binary_dir), pybind_sha256=self.binary_sha256,
            kernel_sha256=self.kernel_sha256, threads=self.threads,
            physical15_cache_enabled=self.cache,
            counts=dict(self._counts), native_stats=dict(self.native.stats()),
            fallback_events=list(self._fallbacks),
            world_sha256_by_factor={str(k): v.world_sha256
                                    for k, v in self._records.items()},
            domain_limit=(
                "C++ central Brown4 predicate uses conservative numeric gates; "
                "project_probe status 2 on an unused world point is undecided, "
                "not a valid projection. verify_accepted checks original "
                "all original endpoints/midpoints: definite statuses must "
                "match the OpenCV/safe central mask, and status-1 UV must "
                "agree within 1e-5 px; status-2 provisional UV is excluded. "
                "It then compares the complete original residual. "
                "A point entering a candidate segment with status 2 makes "
                "rs_edge_eval replace the complete original Python factor."
            ),
        )

    def close_after_release(self) -> None:
        """Call only after releasing optimizer, graph, linearization and Factors."""
        self._main_thread()
        self._records.clear()
        gc.collect()
        self.native.clear_threads()
        if self._dll_handle is not None:
            self._dll_handle.close()
            self._dll_handle = None
        self._closed = True
