"""New Ceres block-sparse bridge with one bulk Python/GPU evaluation per point.

This layer knows no camera, scene, whitening, robust loss, constraint or source
clock. The caller supplies the current model's residuals and exact block
Jacobians. No retired solver module or binary is imported. This is nonlinear
least squares; hard equalities require an explicit admissible parametrization
or an outer constrained method, not an arbitrary large residual weight.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import ctypes as C
import json
import os
import threading
from typing import Callable
import numpy as np

ABI_VERSION = 1
LINEAR_SOLVERS = {"dense_qr": 0, "sparse_normal_cholesky": 1, "sparse_schur": 2,
                  "iterative_schur": 3, "cgnr": 4}
CERES_LINEAR_SOLVERS = {0: "dense_normal_cholesky", 1: "dense_qr", 2: "sparse_normal_cholesky",
                         3: "dense_schur", 4: "sparse_schur", 5: "iterative_schur", 6: "cgnr"}
TERMINATIONS = {0: "convergence", 1: "no_convergence", 2: "failure", 3: "user_success", 4: "user_failure"}
I32P, I64P, U8P, DP = C.POINTER(C.c_int32), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_double)


def _ro(a, dtype):
    value = np.array(a, dtype=dtype, copy=True, order="C")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class BatchLayout:
    """Stable IDs and local block sparsity, independent of physical semantics.

    factor_blocks lists native block indices, once per factor. The Jacobian
    buffer concatenates each factor, then each listed parameter block, each
    matrix row-major [residual_size, block_size]. Logical schema blocks may be
    split into sparse views (e.g. one 3-value spline control); the caller owns
    the semantic_block_id/offset/size mapping used to pack and restore them.
    """
    block_ids: tuple[str, ...]
    block_sizes: tuple[int, ...]
    factor_ids: tuple[str, ...]
    residual_sizes: tuple[int, ...]
    factor_blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        names = ("block_ids", "block_sizes", "factor_ids", "residual_sizes", "factor_blocks")
        for name in names:
            value = getattr(self, name)
            value = tuple(tuple(x) for x in value) if name == "factor_blocks" else tuple(value)
            object.__setattr__(self, name, value)
        nb, nf = len(self.block_ids), len(self.factor_ids)
        if not nb or not nf or len(self.block_sizes) != nb or len(self.residual_sizes) != nf or len(self.factor_blocks) != nf:
            raise ValueError("block/factor layout lengths must match and be nonempty")
        for ids in (self.block_ids, self.factor_ids):
            if len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v for v in ids):
                raise ValueError("stable IDs must be unique nonempty strings")
        for field in ("block_sizes", "residual_sizes"):
            sizes = getattr(self, field)
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 < v <= 2147483647 for v in sizes):
                raise ValueError("block and residual sizes must be positive int32 values")
            object.__setattr__(self, field, tuple(int(v) for v in sizes))
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for bs in self.factor_blocks for v in bs):
            raise ValueError("factor indices must be exact integers")
        object.__setattr__(self, "factor_blocks", tuple(tuple(int(v) for v in bs) for bs in self.factor_blocks))
        edge_offsets, edges, jac_offsets = [0], [], [0]
        for r, blocks in zip(self.residual_sizes, self.factor_blocks):
            if not blocks or len(set(blocks)) != len(blocks):
                raise ValueError("each factor needs distinct parameter block indices")
            for b in blocks:
                if type(b) is not int or not 0 <= b < nb:
                    raise ValueError("factor parameter index outside layout")
                edges.append(b)
                jac_offsets.append(jac_offsets[-1] + r * self.block_sizes[b])
            edge_offsets.append(len(edges))
        object.__setattr__(self, "parameter_offsets", _ro([0, *np.cumsum(self.block_sizes, dtype=np.int64)], np.int64))
        object.__setattr__(self, "residual_offsets", _ro([0, *np.cumsum(self.residual_sizes, dtype=np.int64)], np.int64))
        object.__setattr__(self, "factor_edge_offsets", _ro(edge_offsets, np.int64))
        object.__setattr__(self, "edge_blocks", _ro(edges, np.int32))
        object.__setattr__(self, "jacobian_offsets", _ro(jac_offsets, np.int64))
        object.__setattr__(self, "_sizes", _ro(self.block_sizes, np.int32))
        object.__setattr__(self, "_residual_sizes", _ro(self.residual_sizes, np.int32))

    @property
    def parameter_count(self): return int(self.parameter_offsets[-1])
    @property
    def residual_count(self): return int(self.residual_offsets[-1])
    @property
    def jacobian_value_count(self): return int(self.jacobian_offsets[-1])

    def jacobian_block(self, flat, factor_index, local_parameter_index):
        """Return an output matrix view; values are not copied."""
        f, k = factor_index, local_parameter_index
        if not 0 <= f < len(self.factor_ids) or not 0 <= k < len(self.factor_blocks[f]):
            raise IndexError("factor/local parameter index")
        edge = int(self.factor_edge_offsets[f]) + k
        b = self.factor_blocks[f][k]
        return flat[self.jacobian_offsets[edge]:self.jacobian_offsets[edge+1]].reshape(self.residual_sizes[f], self.block_sizes[b])

    def describe(self):
        return {"block_count": len(self.block_ids), "factor_count": len(self.factor_ids),
                "parameter_count": self.parameter_count, "residual_count": self.residual_count,
                "edge_count": len(self.edge_blocks), "jacobian_values": self.jacobian_value_count,
                "jacobian_layout": "factor, listed_parameter_block, row_major(residual,parameter)"}


@dataclass(frozen=True)
class SolveOptions:
    max_iterations: int = 100
    num_threads: int = max(1, os.cpu_count() or 1)
    linear_solver: str = "sparse_normal_cholesky"
    max_time_seconds: float = 3600.0
    function_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-12
    parameter_tolerance: float = 1e-10
    initial_trust_region_radius: float = 1e4
    max_trust_region_radius: float = 1e16
    progress_to_stdout: bool = False

    def __post_init__(self):
        if type(self.max_iterations) is not int or self.max_iterations <= 0 or type(self.num_threads) is not int or self.num_threads <= 0:
            raise ValueError("positive integer iteration/thread count required")
        if self.linear_solver not in LINEAR_SOLVERS:
            raise ValueError("unknown linear solver")
        for key in ("max_time_seconds", "function_tolerance", "gradient_tolerance", "parameter_tolerance", "initial_trust_region_radius", "max_trust_region_radius"):
            if not np.isfinite(getattr(self, key)):
                raise ValueError("solver options must be finite")
        if self.max_time_seconds <= 0 or min(self.function_tolerance, self.gradient_tolerance, self.parameter_tolerance) < 0:
            raise ValueError("invalid solver tolerance/time")
        if not 0 < self.initial_trust_region_radius <= self.max_trust_region_radius:
            raise ValueError("invalid trust region radii")


@dataclass(frozen=True)
class SolveResult:
    parameters: np.ndarray
    summary: dict
    iterations: tuple[dict, ...]
    callback_errors: tuple[str, ...]
    report: str

    @property
    def usable(self): return bool(self.summary["solution_usable"])


class _Problem(C.Structure):
    _fields_ = [("abi_version", C.c_int32), ("nblocks", C.c_int32), ("nfactors", C.c_int32), ("reserved", C.c_int32),
        ("nparameters", C.c_int64), ("nedges", C.c_int64), ("parameters", DP), ("block_sizes", I32P),
        ("residual_sizes", I32P), ("factor_edge_offsets", I64P), ("edge_blocks", I32P), ("constant_blocks", U8P),
        ("lower_bounds", DP), ("upper_bounds", DP), ("ordering_groups", I32P)]

class _Options(C.Structure):
    _fields_ = [(k, C.c_int32) for k in ("abi_version", "max_iterations", "num_threads", "linear_solver", "minimizer_progress", "reserved0", "reserved1", "reserved2")] + [
        (k, C.c_double) for k in ("max_time_seconds", "function_tolerance", "gradient_tolerance", "parameter_tolerance", "initial_trust_region_radius", "max_trust_region_radius")]

class _Result(C.Structure):
    _fields_ = [(k, C.c_int32) for k in ("abi_version", "termination_type", "solution_usable", "linear_solver_used", "iterations", "successful_steps", "unsuccessful_steps", "residual_evaluations", "jacobian_evaluations", "batch_evaluations", "batch_jacobian_evaluations", "callback_failures")] + [
        (k, C.c_double) for k in ("initial_cost", "final_cost", "total_seconds", "residual_seconds", "jacobian_seconds", "linear_solver_seconds")]

_EVAL = C.CFUNCTYPE(C.c_int, C.c_void_p, DP, C.c_int64, C.c_int, C.c_int, DP, C.c_int64, DP, C.c_int64)
_ITER = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double, C.c_double, C.c_double, C.c_double, DP, C.c_int64)


def _pointer(array, pointer_type):
    return array.ctypes.data_as(pointer_type) if array is not None else pointer_type()


def _vector(value, size, dtype, name, *, finite=True):
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.shape != (size,) or (finite and not np.isfinite(array).all()):
        raise ValueError(f"{name} must have shape ({size},) and valid values")
    return array


class NativeBackend:
    """Load only an explicitly built new bridge and its public Ceres dependencies.

    No automatic compilation or historical data qualification occurs on load.
    A backend instance is serialized; independent instances can run separately.
    """
    def __init__(self, build_manifest):
        self.manifest_path = Path(build_manifest).resolve()
        metadata = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        if metadata.get("schema") != "rinksolve.native_build.v1" or metadata.get("status") != "built" or metadata.get("uses_legacy_solver") is not False:
            raise ValueError("expected current standalone native build manifest")
        self.metadata = metadata
        self._dll_dirs = [os.add_dll_directory(str(Path(p).resolve())) for p in metadata["dll_directories"] if Path(p).is_dir()]
        self._lib = C.CDLL(str(Path(metadata["binary"]).resolve()))
        self._lib.rs_batch_abi_version.restype = C.c_int
        self._lib.rs_ceres_version.restype = C.c_char_p
        if self._lib.rs_batch_abi_version() != ABI_VERSION:
            raise RuntimeError("native bridge ABI mismatch")
        self.ceres_version = self._lib.rs_ceres_version().decode("ascii")
        self._lib.rs_batch_solve.argtypes = [C.POINTER(_Problem), C.POINTER(_Options), _EVAL, _ITER, C.c_void_p, C.POINTER(_Result), C.c_char_p, C.c_int]
        self._lib.rs_batch_solve.restype = C.c_int
        self.has_unit_chart = hasattr(self._lib, "rs_batch_solve_scaled") and hasattr(self._lib, "rs_unit_chart_abi_version")
        if self.has_unit_chart:
            self._lib.rs_unit_chart_abi_version.restype=C.c_int
            if self._lib.rs_unit_chart_abi_version()!=1:raise RuntimeError("unit chart ABI mismatch")
            self._lib.rs_batch_solve_scaled.argtypes=[C.POINTER(_Problem),C.POINTER(_Options),DP,C.c_int64,_EVAL,_ITER,C.c_void_p,C.POINTER(_Result),C.c_char_p,C.c_int]
            self._lib.rs_batch_solve_scaled.restype=C.c_int
        self._lock = threading.Lock()
        self._local = threading.local()
        self.has_native_transpose = hasattr(self._lib, "rs_layout_gradient")
        if self.has_native_transpose:
            self._lib.rs_layout_gradient.argtypes = [C.POINTER(_Problem), DP, C.c_int64, DP, C.c_int64, DP, C.c_int64, C.c_char_p, C.c_int]
            self._lib.rs_layout_gradient.restype = C.c_int
        self.has_jacobian_product = hasattr(self._lib, 'rs_layout_product')
        if self.has_jacobian_product:
            self._lib.rs_layout_product.argtypes=[C.c_int64,I64P,I64P,I64P,I64P,I32P,I32P,DP,DP,DP,C.c_int64,C.c_int]
            self._lib.rs_layout_product.restype=C.c_int
        self.has_row_combinations = hasattr(self._lib, "rs_combine_rows")
        if self.has_row_combinations:
            self._lib.rs_combine_rows.argtypes = [C.c_int64,I64P,I64P,I64P,I32P,DP,DP,I64P,I64P,DP,DP,C.c_int64,C.c_int]
            self._lib.rs_combine_rows.restype = C.c_int


    def jacobian_product(self,layout,jacobian,direction,*,num_threads=20):
        if not self.has_jacobian_product:raise ValueError('current native build needs rs_layout_product')
        j=np.ascontiguousarray(jacobian,dtype=np.float64);v=np.ascontiguousarray(direction,dtype=np.float64)
        if j.shape!=(layout.jacobian_value_count,) or v.shape!=(layout.parameter_count,) or not np.isfinite(j).all() or not np.isfinite(v).all():raise ValueError('Jv input shape/finite')
        if type(num_threads)is not int or num_threads<=0:raise ValueError('Jv positive explicit threads')
        out=np.empty(layout.residual_count,dtype=np.float64)
        rc=self._lib.rs_layout_product(len(layout.factor_ids),_pointer(layout.residual_offsets,I64P),_pointer(layout.factor_edge_offsets,I64P),
            _pointer(layout.parameter_offsets,I64P),_pointer(layout.jacobian_offsets,I64P),_pointer(layout.edge_blocks,I32P),_pointer(layout._sizes,I32P),
            _pointer(j,DP),_pointer(v,DP),_pointer(out,DP),len(out),num_threads)
        if rc:raise ValueError('native Jv rejected '+str(rc))
        return out

    def transpose_product(self, layout: BatchLayout, jacobian, weights):
        """Compute sparse J.T @ weights without a dense global Jacobian."""
        j=np.asarray(jacobian);w=np.asarray(weights)
        if j.dtype!=np.float64 or w.dtype!=np.float64 or j.shape!=(layout.jacobian_value_count,) or w.shape!=(layout.residual_count,) or not j.flags.c_contiguous or not w.flags.c_contiguous or not np.isfinite(j).all() or not np.isfinite(w).all():
            raise ValueError("gradient requires current contiguous finite float64 layout buffers")
        result=np.zeros(layout.parameter_count)
        if self.has_native_transpose:
            p=_Problem(ABI_VERSION,len(layout.block_ids),len(layout.factor_ids),0,layout.parameter_count,len(layout.edge_blocks),DP(),
                _pointer(layout._sizes,I32P),_pointer(layout._residual_sizes,I32P),_pointer(layout.factor_edge_offsets,I64P),_pointer(layout.edge_blocks,I32P),U8P(),DP(),DP(),I32P())
            report=C.create_string_buffer(2048)
            rc=self._lib.rs_layout_gradient(C.byref(p),_pointer(j,DP),len(j),_pointer(w,DP),len(w),_pointer(result,DP),len(result),report,len(report))
            if rc:raise ValueError("native sparse gradient rejected layout: "+report.value.decode("utf8",errors="replace"))
        else:
            for f,blocks in enumerate(layout.factor_blocks):
                row=w[layout.residual_offsets[f]:layout.residual_offsets[f+1]]
                for k,b in enumerate(blocks):
                    e=layout.factor_edge_offsets[f]+k;lo,hi=layout.jacobian_offsets[e:e+2]
                    result[layout.parameter_offsets[b]:layout.parameter_offsets[b+1]]+=j[lo:hi].reshape(layout.residual_sizes[f],layout.block_sizes[b]).T@row
        return result

    def solve(self, layout: BatchLayout, initial_parameters, evaluate: Callable, *, options=None,
              constant_blocks=None, lower_bounds=None, upper_bounds=None, ordering_groups=None,
              on_iteration=None, parameter_scales=None) -> SolveResult:
        """Run actual Ceres nonlinear iterations against the caller's batch kernel.

        evaluate(x, residual_out, jacobian_out_or_None, new_point) writes every
        requested value and returns None/True for success, False for a domain
        failure. x is a borrowed read-only view. Output buffers and x are valid
        only during the callback; asynchronous GPU work must synchronize before
        return. All residuals/whitening and all derivatives are caller-defined.
        One call covers ALL factors, never one Python callback per factor.

        on_iteration(event, x) may persist current accepted parameters. Return
        0/None to continue, +1 for caller success, -1 to abort. Iteration zero is
        Ceres' initial state and is explicitly marked. Result.usable and the
        termination reason must be checked; return does not imply fit validity.
        Input arrays are copied; immutable source arrays are never optimized.
        """
        if getattr(self._local, "solving", False):
            raise RuntimeError("recursive solve on the same backend is forbidden inside callbacks")
        options = options or SolveOptions()
        if not callable(evaluate): raise TypeError("evaluate must be callable")
        x = _vector(initial_parameters, layout.parameter_count, np.float64, "initial_parameters")
        scales = _vector(parameter_scales, len(x), np.float64, "parameter_scales") if parameter_scales is not None else None
        if scales is not None:
            if np.any(scales<=0):raise ValueError("parameter_scales must be strictly positive")
            if not self.has_unit_chart:raise ValueError("current native build lacks explicit unit chart identity LM metric")
        nb = len(layout.block_ids)
        if constant_blocks is not None and (np.asarray(constant_blocks).shape != (nb,) or not np.isin(constant_blocks, [0, 1]).all()):
            raise ValueError("constant block values must be exactly 0 or 1")
        const = _vector(constant_blocks, nb, np.uint8, "constant_blocks") if constant_blocks is not None else None
        if const is not None and np.any(const > 1): raise ValueError("constant block values must be 0 or 1")
        lo = _vector(lower_bounds, len(x), np.float64, "lower_bounds", finite=False) if lower_bounds is not None else None
        hi = _vector(upper_bounds, len(x), np.float64, "upper_bounds", finite=False) if upper_bounds is not None else None
        low, high = lo if lo is not None else -np.inf, hi if hi is not None else np.inf
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high) or np.any(x < low) or np.any(x > high):
            raise ValueError("invalid bounds or initial bound violation")
        if ordering_groups is not None:
            raw_groups = np.asarray(ordering_groups)
            if raw_groups.shape != (nb,) or not np.isfinite(raw_groups).all() or np.any(raw_groups != np.floor(raw_groups)) or np.any(raw_groups < 0) or np.any(raw_groups > 2147483647):
                raise ValueError("ordering groups must be exact nonnegative int32 values")
        groups = _vector(ordering_groups, nb, np.int32, "ordering_groups") if ordering_groups is not None else None
        if groups is not None:
            if np.any(groups < 0): raise ValueError("ordering groups must be nonnegative")
            if options.linear_solver in ("sparse_schur", "iterative_schur"):
                free = np.ones(nb, dtype=bool) if const is None else const == 0
                eliminated = int(groups[free].min()) if np.any(free) else -1
                if any(sum(groups[b] == eliminated and free[b] for b in bs) > 1 for bs in layout.factor_blocks):
                    raise ValueError("Schur first group is not an independent parameter set")
        p = _Problem(ABI_VERSION, nb, len(layout.factor_ids), 0, len(x), len(layout.edge_blocks),
            _pointer(x, DP), _pointer(layout._sizes, I32P), _pointer(layout._residual_sizes, I32P),
            _pointer(layout.factor_edge_offsets, I64P), _pointer(layout.edge_blocks, I32P), _pointer(const, U8P),
            _pointer(lo, DP), _pointer(hi, DP), _pointer(groups, I32P))
        o = _Options(ABI_VERSION, options.max_iterations, options.num_threads, LINEAR_SOLVERS[options.linear_solver],
            int(options.progress_to_stdout), 0, 0, 0, options.max_time_seconds, options.function_tolerance,
            options.gradient_tolerance, options.parameter_tolerance, options.initial_trust_region_radius, options.max_trust_region_radius)
        errors, events = [], []
        def remember(text):
            if len(errors) < 64: errors.append(text)
        @_EVAL
        def callback(_context, xp, nx, want_j, new_point, rp, nr, jp, nj):
            try:
                if (nx, nr, nj) != (layout.parameter_count, layout.residual_count, layout.jacobian_value_count if want_j else 0):
                    raise RuntimeError("native evaluation layout disagreement")
                xv = np.ctypeslib.as_array(xp, shape=(nx,)).view(); xv.setflags(write=False)
                rv = np.ctypeslib.as_array(rp, shape=(nr,))
                jv = np.ctypeslib.as_array(jp, shape=(nj,)) if want_j else None
                status = evaluate(xv, rv, jv, bool(new_point))
                if status is False: return 0
                if status is not None and status is not True:
                    raise TypeError("batch evaluator returns only None, True or False")
                return 1
            except BaseException as exc:
                remember(f"evaluate: {type(exc).__name__}: {exc}")
                return 0
        @_ITER
        def iteration(_context, number, valid, successful, cost, change, gradient, step, radius, xp, nx):
            try:
                event = {"iteration": number, "initial_state": number == 0, "step_valid": bool(valid),
                    "step_successful": bool(successful), "cost": cost, "cost_change": change,
                    "gradient_max_norm": gradient, "step_norm": step, "trust_region_radius": radius}
                events.append(event)
                if on_iteration is not None:
                    xv = np.ctypeslib.as_array(xp, shape=(nx,)).view(); xv.setflags(write=False)
                    rc = on_iteration(dict(event), xv)
                    if rc not in (None, -1, 0, 1): raise ValueError("iteration callback returns -1, 0, +1 or None")
                    return int(rc or 0)
                return 0
            except BaseException as exc:
                remember(f"iteration: {type(exc).__name__}: {exc}")
                return -1
        native_result, report = _Result(), C.create_string_buffer(32768)
        with self._lock:
            self._local.solving = True
            try:
                if scales is None:
                    rc = self._lib.rs_batch_solve(C.byref(p), C.byref(o), callback, iteration, None, C.byref(native_result), report, len(report))
                else:
                    rc = self._lib.rs_batch_solve_scaled(C.byref(p), C.byref(o), _pointer(scales,DP), len(scales), callback, iteration, None, C.byref(native_result), report, len(report))
            finally:
                self._local.solving = False
        text = report.value.decode("utf-8", errors="replace")
        if rc != 0: raise ValueError(f"native bridge rejected problem ({rc}): {text}")
        summary = {name: getattr(native_result, name) for name, _ in _Result._fields_}
        summary["termination"] = TERMINATIONS.get(summary["termination_type"], "unknown")
        summary["linear_solver"] = CERES_LINEAR_SOLVERS.get(summary["linear_solver_used"], "unknown")
        summary["ceres_version"] = self.ceres_version
        summary["legacy_solver_used"] = False
        summary["numerical_metric"] = {"implementation":"full_rank_unit_tangent_identity_lm" if scales is not None else "public_ceres_default", "unit_chart_abi":1 if scales is not None else None,"ambient_bounds":True,"reduced_degrees_of_freedom":0,"jacobi_scaling":False if scales is not None else True,"min_lm_diagonal":1.0 if scales is not None else None,"max_lm_diagonal":1.0 if scales is not None else None,"parameter_scales_sha256":__import__("hashlib").sha256(scales.tobytes()).hexdigest() if scales is not None else None}
        summary["compute_backend"] = "caller_supplied_bulk_evaluator"
        summary["gpu_execution"] = "not_measured_by_native_bridge"
        x.setflags(write=False)
        return SolveResult(x, summary, tuple(events), tuple(errors), text)


class RowCombinationPlan:
    """Fixed-layout a*u+b*v Jacobian chain, with optional direct scalar terms.

    Covers generic scalar constraints on a delivered 2-vector. Coefficients,
    physical meanings and direct derivative positions are caller-supplied.
    """
    def __init__(self, backend:NativeBackend, raw_layout:BatchLayout, rows, *, appended_blocks=()):
        if any(v!=2 for v in raw_layout.residual_sizes):raise ValueError("raw factors must each be 2-vectors")
        indices=np.asarray(rows)
        if indices.ndim!=1 or indices.dtype.kind not in "iu" or np.any(indices<0) or np.any(indices>=len(raw_layout.factor_ids)):
            raise ValueError("combination row indices outside raw layout")
        self.backend,self.raw_layout=backend,raw_layout
        self.appended_blocks=tuple(appended_blocks)
        if len(set(self.appended_blocks))!=len(self.appended_blocks) or any(type(b)is not int or not 0<=b<len(raw_layout.block_ids) for b in self.appended_blocks):raise ValueError("invalid appended derivative blocks")
        if any(set(raw_layout.factor_blocks[i])&set(self.appended_blocks) for i in indices):raise ValueError("appended block already in raw row")
        self.rows=_ro(indices,np.int64)
        self.edge_widths=_ro(raw_layout._sizes[raw_layout.edge_blocks],np.int32)
        widths=np.array([(raw_layout.jacobian_offsets[raw_layout.factor_edge_offsets[i+1]]-raw_layout.jacobian_offsets[raw_layout.factor_edge_offsets[i]])//2 for i in indices],dtype=np.int64)
        widths+=sum(raw_layout.block_sizes[b] for b in self.appended_blocks)
        self.out_offsets=_ro(np.r_[0,np.cumsum(widths)],np.int64)
        self.flat_count=int(self.out_offsets[-1])

    def check_output_layout(self,layout):
        expected=tuple(self.raw_layout.factor_blocks[i]+self.appended_blocks for i in self.rows)
        if (layout.block_ids,layout.block_sizes)!=(self.raw_layout.block_ids,self.raw_layout.block_sizes) or layout.residual_sizes!=(1,)*len(self.rows) or layout.factor_blocks!=expected or layout.jacobian_value_count!=self.flat_count:
            raise ValueError("scalar constraint layout differs from declared 2-vector chain")

    def combine(self,raw_jacobian,coefficients,*,out=None,direct_positions=None,direct_values=None,num_threads=20):
        j=np.asarray(raw_jacobian);c=np.asarray(coefficients)
        if j.dtype!=np.float64 or j.shape!=(self.raw_layout.jacobian_value_count,) or not j.flags.c_contiguous or not np.isfinite(j).all():
            raise ValueError("raw combination Jacobian ABI")
        if c.dtype!=np.float64 or c.shape!=(len(self.rows),2) or not c.flags.c_contiguous or not np.isfinite(c).all():
            raise ValueError("combination coefficient ABI")
        result=np.empty(self.flat_count) if out is None else np.asarray(out)
        if result.dtype!=np.float64 or result.shape!=(self.flat_count,) or not result.flags.c_contiguous or not result.flags.writeable:
            raise ValueError("combination output ABI")
        positions=values=None
        if direct_positions is not None:
            raw=np.asarray(direct_positions)
            if raw.shape!=(len(self.rows),) or raw.dtype.kind not in "iu":raise ValueError("direct derivative positions require exact integer row vector")
            positions=np.ascontiguousarray(raw,dtype=np.int64)
            active=positions>=0
            if np.any(active&((positions<self.out_offsets[:-1])|(positions>=self.out_offsets[1:]))):raise ValueError("direct derivative position outside its output factor")
            values=np.asarray(direct_values)
            if values.dtype!=np.float64 or values.shape!=positions.shape or not values.flags.c_contiguous or not np.isfinite(values).all():raise ValueError("direct derivative values ABI")
        if type(num_threads) is not int or num_threads<=0:raise ValueError("positive explicit thread count required")
        result.fill(0)
        if self.backend.has_row_combinations:
            rc=self.backend._lib.rs_combine_rows(len(self.rows),_pointer(self.rows,I64P),_pointer(self.raw_layout.factor_edge_offsets,I64P),
                _pointer(self.raw_layout.jacobian_offsets,I64P),_pointer(self.edge_widths,I32P),_pointer(j,DP),_pointer(c,DP),
                _pointer(self.out_offsets,I64P),_pointer(positions,I64P),_pointer(values,DP),_pointer(result,DP),len(result),num_threads)
            if rc:raise ValueError(f"native row combination failed ({rc})")
        else:
            for i,row in enumerate(self.rows):
                target=self.out_offsets[i]
                for k,b in enumerate(self.raw_layout.factor_blocks[row]):
                    matrix=self.raw_layout.jacobian_block(j,row,k);width=matrix.shape[1]
                    result[target:target+width]=c[i]@matrix;target+=width
                if positions is not None and positions[i]>=0:result[positions[i]]+=values[i]
            if not np.isfinite(result).all():raise ValueError("nonfinite row combination")
        return result
