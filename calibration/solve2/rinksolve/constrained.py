"""Explicit-unit Powell-Hestenes-Rockafellar outer method for current kernels.

Only normalized mathematical equality h=0 and inequality g<=0 enter this layer.
It defines no physics, measurement, statistical sigma or engineering budget.
Finite penalty limits and incomplete inner solves are resumable numerical
states, never declarations that a scene has been fitted.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Callable
import hashlib
import json
import os
import uuid
import threading
from functools import wraps
import numpy as np
from .native_backend import BatchLayout, NativeBackend, SolveOptions
from .atomic_io import atomic_replace,read_text_shared

VERSION = "rinksolve.phr.v1"


def readonly(value, dtype=np.float64):
    a = np.array(value, dtype=dtype, copy=True, order="C")
    a.setflags(write=False)
    return a


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def layout_digest(layout):
    h = hashlib.sha256()
    for ids in (layout.block_ids, layout.factor_ids):
        for v in ids:
            b = v.encode("utf8"); h.update(len(b).to_bytes(8,"little")); h.update(b)
    for a in (layout._sizes, layout._residual_sizes, layout.factor_edge_offsets, layout.edge_blocks):
        h.update(a.dtype.str.encode("ascii")); h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ConstraintFamily:
    """A same-unit family of constraint factors; scale is an explicit unit chart.

    normalization_scale may be one positive value or one value per component
    of the family's factors, in factor_indices order. It is not a sigma.
    authority_refs and scale_source preserve the caller's authority/engineering
    declarations; this layer never upgrades them into confirmed physical facts.
    """
    id: str
    sense: str                         # equality or less_equal
    factor_indices: tuple[int, ...]
    unit: str
    normalization_scale: float | tuple[float, ...]
    scale_source: str
    authority_refs: tuple[str, ...]

    def __post_init__(self):
        if not self.id or self.sense not in ("equality", "less_equal") or not self.unit or not self.scale_source:
            raise ValueError("constraint ID/sense/unit/scale source must be explicit")
        indices = tuple(self.factor_indices)
        if not indices or len(set(indices)) != len(indices) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v < 0 for v in indices):
            raise ValueError("family factor indices must be distinct nonnegative integers")
        object.__setattr__(self, "factor_indices", tuple(int(v) for v in indices))
        object.__setattr__(self, "authority_refs", tuple(self.authority_refs))
        if any(not isinstance(v,str) or not v for v in self.authority_refs):
            raise ValueError("constraint authority references must be nonempty strings")
        scale = np.asarray(self.normalization_scale,dtype=np.float64)
        if scale.ndim > 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("constraint normalization scale must be finite and positive")
        object.__setattr__(self, "normalization_scale", float(scale) if scale.ndim == 0 else tuple(float(v) for v in scale))


@dataclass(frozen=True)
class ConstraintSet:
    layout: BatchLayout
    families: tuple[ConstraintFamily, ...]

    def __post_init__(self):
        object.__setattr__(self,"families",tuple(self.families))
        if not self.families or len({f.id for f in self.families}) != len(self.families):
            raise ValueError("nonempty distinct constraint families required")
        covered = np.full(len(self.layout.factor_ids),-1,dtype=np.int32)
        row_families = np.empty(self.layout.residual_count,dtype=np.int32)
        scales = np.empty(self.layout.residual_count)
        equality = np.empty(self.layout.residual_count,dtype=bool)
        for family_index,family in enumerate(self.families):
            scale = np.asarray(family.normalization_scale)
            required = sum(self.layout.residual_sizes[i] for i in family.factor_indices if i < len(covered))
            if scale.ndim and len(scale) != required:
                raise ValueError("family scale array must match its scalar components")
            cursor = 0
            for f in family.factor_indices:
                if f >= len(covered) or covered[f] != -1:
                    raise ValueError("constraint factors must belong to exactly one family")
                covered[f] = family_index
                start,end = self.layout.residual_offsets[f:f+2]
                row_families[start:end] = family_index
                equality[start:end] = family.sense == "equality"
                scales[start:end] = scale if scale.ndim == 0 else scale[cursor:cursor+end-start]
                cursor += end-start
        if np.any(covered < 0): raise ValueError("constraint factor missing a family")
        object.__setattr__(self,"row_families",readonly(row_families,np.int32))
        object.__setattr__(self,"scales",readonly(scales))
        object.__setattr__(self,"equality",readonly(equality,bool))
        # One row index per Jacobian scalar, reusable for the PHR row chain.
        jac_rows = np.empty(self.layout.jacobian_value_count,dtype=np.int64)
        for f,blocks in enumerate(self.layout.factor_blocks):
            first = self.layout.residual_offsets[f]
            rows = np.arange(first,self.layout.residual_offsets[f+1])
            edge = self.layout.factor_edge_offsets[f]
            for k,b in enumerate(blocks):
                lo,hi = self.layout.jacobian_offsets[edge+k:edge+k+2]
                jac_rows[lo:hi] = np.repeat(rows,self.layout.block_sizes[b])
        object.__setattr__(self,"jacobian_rows",readonly(jac_rows,np.int64))

    def digest(self):
        h = hashlib.sha256(layout_digest(self.layout).encode("ascii"))
        h.update(_json([asdict(v) for v in self.families]).encode("utf8"))
        h.update(self.scales.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class ConstrainedProblem:
    identity: dict
    block_ids: tuple[str,...]
    block_sizes: tuple[int,...]
    parameter_scales: np.ndarray
    objective_layout: BatchLayout | None = None
    constraints: ConstraintSet | None = None
    constant_blocks: np.ndarray | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    ordering_groups: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self,"block_ids",tuple(self.block_ids))
        object.__setattr__(self,"block_sizes",tuple(self.block_sizes))
        identity = json.loads(_json(self.identity))
        if not isinstance(identity,dict) or not identity:
            raise ValueError("current input/model/constraint identity must be explicit")
        object.__setattr__(self,"identity",identity)
        zero = BatchLayout(self.block_ids,self.block_sizes,("objective/zero",),(1,),((0,),))
        objective = self.objective_layout or zero
        if self.constraints is not None:
            overlap = set(objective.factor_ids)&set(self.constraints.layout.factor_ids)
            if overlap: raise ValueError("objective and constraint factor IDs must be distinct")
        layouts = [objective]+([self.constraints.layout] if self.constraints else [])
        for v in layouts:
            if (v.block_ids,v.block_sizes) != (self.block_ids,self.block_sizes):
                raise ValueError("all objective/constraint layouts must use the same native parameter views")
        scale = readonly(self.parameter_scales)
        if scale.shape != (zero.parameter_count,) or not np.isfinite(scale).all() or np.any(scale<=0):
            raise ValueError("each native parameter component requires an explicit positive normalization scale")
        object.__setattr__(self,"parameter_scales",scale)
        nb,np_ = len(self.block_ids),zero.parameter_count
        for name,size,dtype in (("constant_blocks",nb,np.uint8),("lower_bounds",np_,np.float64),("upper_bounds",np_,np.float64),("ordering_groups",nb,np.int32)):
            value = getattr(self,name)
            if value is None: continue
            raw = np.asarray(value)
            if raw.shape != (size,): raise ValueError(f"{name} shape mismatch")
            if name == "constant_blocks" and not np.isin(raw,[0,1]).all(): raise ValueError("constant flags must be 0 or 1")
            if name == "ordering_groups" and (not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)) or np.any(raw<0) or np.any(raw>2147483647)):
                raise ValueError("ordering groups require nonnegative exact int32")
            if name.endswith("bounds") and np.isnan(raw).any(): raise ValueError("NaN parameter bound")
            object.__setattr__(self,name,readonly(raw,dtype))
        if self.lower_bounds is not None and self.upper_bounds is not None and np.any(self.lower_bounds>self.upper_bounds):
            raise ValueError("lower parameter bound exceeds upper bound")
        object.__setattr__(self,"_objective",objective)
        combined = objective if self.constraints is None else BatchLayout(self.block_ids,self.block_sizes,
            objective.factor_ids+self.constraints.layout.factor_ids,objective.residual_sizes+self.constraints.layout.residual_sizes,
            objective.factor_blocks+self.constraints.layout.factor_blocks)
        object.__setattr__(self,"combined_layout",combined)
        h = hashlib.sha256(_json(identity).encode("utf8"))
        h.update(layout_digest(objective).encode("ascii"))
        h.update((self.constraints.digest() if self.constraints else "no_constraints").encode("ascii"))
        for v in (scale,self.constant_blocks,self.lower_bounds,self.upper_bounds,self.ordering_groups):
            if v is not None: h.update(v.dtype.str.encode("ascii"));h.update(v.tobytes())
            else:h.update(b"none")
        object.__setattr__(self,"digest",h.hexdigest())


@dataclass(frozen=True)
class ALOptions:
    # Required caller-visible numerical policy; no hidden model assumptions.
    initial_penalty: float
    penalty_growth: float
    penalty_limit: float
    required_violation_reduction: float
    feasibility_tolerance: float
    stationarity_tolerance: float
    complementarity_tolerance: float
    inner_stationarity_tolerance: float
    inner_options: SolveOptions
    inner_parameter_metric: str = "ceres_default"
    continuous_inner: bool = False

    def __post_init__(self):
        values = [getattr(self,n) for n in ("initial_penalty","penalty_growth","penalty_limit","required_violation_reduction","feasibility_tolerance","stationarity_tolerance","complementarity_tolerance","inner_stationarity_tolerance")]
        if not np.isfinite(values).all(): raise ValueError("AL policy must be finite")
        if not 0 < self.initial_penalty <= self.penalty_limit or self.penalty_growth <= 1 or not 0 < self.required_violation_reduction < 1:
            raise ValueError("invalid finite penalty/progress policy")
        if min(self.feasibility_tolerance,self.stationarity_tolerance,self.complementarity_tolerance,self.inner_stationarity_tolerance)<=0:
            raise ValueError("positive explicit convergence tolerances required")
        if not isinstance(self.inner_options,SolveOptions): raise TypeError("inner_options must be SolveOptions")
        if self.inner_parameter_metric not in ("ceres_default","unit_tangent_identity_lm","full_uv_unit_tangent_identity_lm"):raise ValueError("explicit supported inner parameter metric required")
        if type(self.continuous_inner)is not bool:raise TypeError('continuous_inner must be bool')
        if self.continuous_inner:
            native=self.inner_options
            if native.max_iterations!=np.iinfo(np.int32).max or native.max_time_seconds!=np.finfo(np.float64).max:
                raise ValueError('continuous inner uses only the native API representational ceilings, not iteration/time scheduling chunks')
            if any(getattr(native,k)!=0 for k in ('function_tolerance','gradient_tolerance','parameter_tolerance')):
                raise ValueError('continuous inner termination uses the original declared-unit augmented stationarity callback')


@dataclass(frozen=True)
class ALState:
    problem_digest: str
    identity: dict
    parameters: np.ndarray
    multipliers: np.ndarray              # normalized constraint convention
    penalties: np.ndarray                # one finite rho per family
    previous_violation: np.ndarray       # one shifted PHR measure per family
    outer_iteration: int
    phase: str
    policy: dict
    diagnostics: dict
    last_inner: dict | None = None
    history: tuple[dict,...] = ()
    version: str = VERSION

    def __post_init__(self):
        for n in ("parameters","multipliers","penalties","previous_violation"):
            a=readonly(getattr(self,n))
            if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError("finite flat AL state arrays required")
            object.__setattr__(self,n,a)
        if self.version!=VERSION or self.outer_iteration<0 or np.any(self.penalties<=0) or np.any(self.previous_violation<0):
            raise ValueError("invalid AL continuation state")
        for n in ("identity","policy","diagnostics","last_inner"):
            value=getattr(self,n)
            if value is not None:object.__setattr__(self,n,json.loads(_json(value)))
        object.__setattr__(self,"history",tuple(json.loads(_json(v)) for v in self.history))

    def save(self,directory,*,data_root):
        """New array generation then atomic manifest; previous arrays are retained."""
        root=Path(data_root).resolve();directory=Path(directory).resolve()
        if not directory.is_relative_to(root): raise ValueError("checkpoints must stay inside the declared Data root")
        directory.mkdir(parents=True,exist_ok=True)
        generation=uuid.uuid4().hex
        arrays=directory/f"state_{generation}.npz"
        with arrays.open("xb") as stream:
            np.savez(stream,parameters=self.parameters,multipliers=self.multipliers,penalties=self.penalties,previous_violation=self.previous_violation)
            stream.flush();os.fsync(stream.fileno())
        metadata={n:getattr(self,n) for n in ("problem_digest","identity","outer_iteration","phase","policy","diagnostics","last_inner","history","version")}
        metadata.update(schema="rinksolve.al_state.v1",generation=generation,arrays=arrays.name)
        temporary=directory/f"manifest_{generation}.tmp"
        with temporary.open("x",encoding="utf8") as stream:
            stream.write(_json(metadata));stream.flush();os.fsync(stream.fileno())
        manifest=directory/"state.json";atomic_replace(temporary,manifest)
        return manifest

    @classmethod
    def load(cls,manifest,*,data_root):
        root=Path(data_root).resolve();manifest=Path(manifest).resolve()
        if not manifest.is_relative_to(root): raise ValueError("checkpoint outside declared Data root")
        metadata=json.loads(read_text_shared(manifest))
        if metadata.pop("schema")!="rinksolve.al_state.v1":raise ValueError("unknown AL state schema")
        generation=metadata.pop("generation")
        arrays=(manifest.parent/metadata.pop("arrays")).resolve()
        if not arrays.is_relative_to(manifest.parent) or arrays.name!=f"state_{generation}.npz":raise ValueError("checkpoint generation mismatch")
        with np.load(arrays,allow_pickle=False) as a:
            return cls(**metadata,**{n:a[n] for n in ("parameters","multipliers","penalties","previous_violation")})


@dataclass(frozen=True)
class ALResult:
    state: ALState
    status: str
    trace: tuple[dict,...]

    @property
    def numerically_converged(self):return self.status=="numerically_feasible_stationary"


class EvaluationFailure(RuntimeError):pass


class KernelDomainRejection(EvaluationFailure):
    """An expected inadmissible trial; Ceres may shrink its trust region."""
    pass


class CheckpointCallbackFailure(RuntimeError):
    def __init__(self,state,error):
        super().__init__(f"checkpoint callback failed: {error}");self.state=state


def _fill(callback,x,r,j,new_point,kind):
    r.fill(np.nan)
    if j is not None:j.fill(np.nan)
    xr=x.view();xr.setflags(write=False)
    status=callback(xr,r,j,new_point)
    if status is False: raise KernelDomainRejection(f"{kind} kernel rejected current state")
    if status is not None and status is not True:raise EvaluationFailure(f"{kind} callback must return None/True/False")
    if not np.isfinite(r).all() or (j is not None and not np.isfinite(j).all()):raise EvaluationFailure(f"{kind} kernel left nonfinite values")


def transpose_product(layout,jacobian,weights):
    """Exact sparse J.T @ weights, no dense observation-by-parameter matrix."""
    result=np.zeros(layout.parameter_count)
    for f,blocks in enumerate(layout.factor_blocks):
        row=weights[layout.residual_offsets[f]:layout.residual_offsets[f+1]]
        edge=layout.factor_edge_offsets[f]
        for k,b in enumerate(blocks):
            lo,hi=layout.jacobian_offsets[edge+k:edge+k+2]
            matrix=jacobian[lo:hi].reshape(layout.residual_sizes[f],layout.block_sizes[b])
            start,end=layout.parameter_offsets[b:b+2]
            result[start:end]+=matrix.T@row
    return result


def _exclusive(method):
    @wraps(method)
    def guarded(self,*args,**kwargs):
        owner=threading.get_ident()
        if self._owner==owner:raise RuntimeError("recursive AL evaluation/run is forbidden inside callbacks")
        with self._lock:
            self._owner=owner
            try:return method(self,*args,**kwargs)
            finally:self._owner=None
    return guarded


class AugmentedLagrangian:
    def __init__(self,backend:NativeBackend,problem:ConstrainedProblem,evaluate_objective:Callable|None,evaluate_constraints:Callable|None,*,inner_scale_callback:Callable|None=None):
        self.backend,self.problem=backend,problem
        self.inner_scale_callback=inner_scale_callback
        self.projected_stationarity_callback=None
        self.inner_solve_callback=None
        self._lock=threading.Lock();self._owner=None
        if problem.objective_layout is not None and not callable(evaluate_objective):raise TypeError("objective callback required")
        if problem.constraints is not None and not callable(evaluate_constraints):raise TypeError("constraint callback required")
        self.objective_callback=evaluate_objective
        self.constraint_callback=evaluate_constraints
        self._or=np.empty(problem._objective.residual_count)
        self._oj=np.empty(problem._objective.jacobian_value_count)
        self._cr=np.empty(problem.constraints.layout.residual_count if problem.constraints else 0)
        self._cj=np.empty(problem.constraints.layout.jacobian_value_count if problem.constraints else 0)
        self._constant=np.zeros(problem._objective.parameter_count,dtype=bool)
        if problem.constant_blocks is not None:
            for b,v in enumerate(problem.constant_blocks):
                if v:self._constant[problem._objective.parameter_offsets[b]:problem._objective.parameter_offsets[b+1]]=True

    def _evaluate(self,x,need_jacobian=True):
        if self.problem.objective_layout is None:
            self._or.fill(0);self._oj.fill(0)
        else:_fill(self.objective_callback,x,self._or,self._oj if need_jacobian else None,True,"objective")
        if self.problem.constraints:_fill(self.constraint_callback,x,self._cr,self._cj if need_jacobian else None,True,"constraint")
        return self._or,self._oj,self._cr,self._cj

    def _rho_rows(self,penalties):
        c=self.problem.constraints
        return penalties[c.row_families] if c else np.empty(0)

    def _measure(self,q,multipliers,penalties):
        c=self.problem.constraints
        if c is None:return np.empty(0)
        v=np.where(c.equality,np.abs(q),np.abs(np.maximum(q,-multipliers/self._rho_rows(penalties))))
        measures=np.zeros(len(c.families));np.maximum.at(measures,c.row_families,v)
        return measures

    def _projected_gradient(self,x,gradient):
        p=self.problem;scale=p.parameter_scales;y=x/scale;g=gradient*scale
        lo=p.lower_bounds/scale if p.lower_bounds is not None else -np.inf
        hi=p.upper_bounds/scale if p.upper_bounds is not None else np.inf
        projected=y-np.minimum(np.maximum(y-g,lo),hi)
        projected[self._constant]=0
        return projected

    def _stationarity(self,x,gradient,objective_residual,constraint_multipliers):
        callback=self.projected_stationarity_callback
        return callback(x,gradient,objective_residual,constraint_multipliers) if callable(callback) else self._projected_gradient(x,gradient)

    def _diagnostics(self,x,multipliers,penalties,*,jacobian=True):
        r,j,cvalues,cj=self._evaluate(x,jacobian)
        c=self.problem.constraints
        q=cvalues/c.scales if c else np.empty(0)
        equality=c.equality if c else np.empty(0,dtype=bool)
        inequality=~equality
        feasibility=float(max(np.max(np.abs(q[equality]),initial=0),np.max(np.maximum(q[inequality],0),initial=0)))
        dual=float(np.max(np.maximum(-multipliers[inequality],0),initial=0))
        comp=float(np.max(np.abs(multipliers[inequality]*q[inequality]),initial=0))
        objective=.5*float(r@r)
        families=[]
        if c:
            for i,f in enumerate(c.families):
                mask=c.row_families==i;raw=cvalues[mask];qn=q[mask]
                raw_violation=np.abs(raw) if f.sense=="equality" else np.maximum(raw,0)
                norm_violation=np.abs(qn) if f.sense=="equality" else np.maximum(qn,0)
                families.append({"id":f.id,"sense":f.sense,"unit":f.unit,"raw_violation_inf":float(np.max(raw_violation,initial=0)),
                    "normalized_violation_inf":float(np.max(norm_violation,initial=0)),"raw_min":float(np.min(raw)),"raw_max":float(np.max(raw)),
                    "normalization_scale_min":float(np.min(c.scales[mask])),"normalization_scale_max":float(np.max(c.scales[mask])),
                    "scale_source":f.scale_source,"authority_refs":list(f.authority_refs),"penalty":float(penalties[i])})
        diagnostics={"objective":objective,"normalized_feasibility_inf":feasibility,"normalized_complementarity_inf":comp,
            "normalized_dual_violation_inf":dual,"families":families,"objective_semantics":"caller_supplied_sum_of_squared_residuals_over_two",
            "budget_is_sigma":False,"parameter_scales_explicit":True}
        if jacobian:
            gradient=self.backend.transpose_product(self.problem._objective,j,r)
            if c:gradient+=self.backend.transpose_product(c.layout,cj,multipliers/c.scales)
            diagnostics["normalized_projected_stationarity_inf"]=float(np.max(np.abs(self._stationarity(x,gradient,r,multipliers)),initial=0))
        return diagnostics,q.copy()

    @staticmethod
    def _converged(d,options):
        return d["normalized_feasibility_inf"]<=options.feasibility_tolerance and d["normalized_complementarity_inf"]<=options.complementarity_tolerance and d["normalized_dual_violation_inf"]<=options.complementarity_tolerance and d["normalized_projected_stationarity_inf"]<=options.stationarity_tolerance

    @_exclusive
    def initialize(self,parameters,options:ALOptions):
        p=self.problem;x=readonly(parameters)
        if x.shape!=(p._objective.parameter_count,) or not np.isfinite(x).all():raise ValueError("initial parameter shape/values")
        if p.lower_bounds is not None and np.any(x<p.lower_bounds):raise ValueError("initial lower bound violation")
        if p.upper_bounds is not None and np.any(x>p.upper_bounds):raise ValueError("initial upper bound violation")
        multipliers=np.zeros(len(self._cr));penalties=np.full(len(p.constraints.families) if p.constraints else 0,options.initial_penalty)
        diagnostics,q=self._diagnostics(x,multipliers,penalties)
        return ALState(p.digest,p.identity,x,multipliers,penalties,self._measure(q,multipliers,penalties),0,"ready",asdict(options),diagnostics)

    @_exclusive
    def run(self,state:ALState,options:ALOptions,*,outer_iterations:int,on_checkpoint=None,on_inner_iteration=None,policy_change_reason=None):
        """Execute a caller-declared chunk; exhausting it is a resumable state.

        An inner iteration/time limit never triggers multiplier updates unless
        its actual projected augmented gradient meets the explicit inner target.
        All current source/model identities remain bound across continuation.
        """
        p=self.problem;c=p.constraints
        if type(outer_iterations) is not int or outer_iterations<=0:raise ValueError("caller must declare positive outer_iterations")
        if state.problem_digest!=p.digest or state.identity!=p.identity:raise ValueError("AL state belongs to different current inputs/model/constraints")
        if state.parameters.shape!=(p._objective.parameter_count,) or state.multipliers.shape!=(len(self._cr),) or state.penalties.shape!=(len(c.families) if c else 0,) or state.previous_violation.shape!=state.penalties.shape:
            raise ValueError("AL state array dimensions differ from current problem")
        if c and np.any(state.multipliers[~c.equality]<0):raise ValueError("inequality multipliers must be nonnegative")
        if np.any(state.penalties>options.penalty_limit):raise ValueError("resume policy penalty limit below existing state")
        policy=asdict(options);trace=[]
        if state.policy!=policy:
            if not policy_change_reason:raise ValueError("changed numerical policy requires an explicit recorded reason")
            event={"kind":"numerical_policy_changed","reason":policy_change_reason,"before":state.policy,"after":policy}
            state=replace(state,policy=policy,history=state.history+(event,));trace.append(event)
        def checkpoint(current,event):
            if on_checkpoint is not None:
                try:on_checkpoint(current,event)
                except BaseException as exc:raise CheckpointCallbackFailure(current,exc) from exc
        for _ in range(outer_iterations):
            try:diagnostics,q=self._diagnostics(state.parameters,state.multipliers,state.penalties)
            except BaseException as exc:
                event={"kind":"current_evaluation_failure","error":f"{type(exc).__name__}: {exc}"}
                state=replace(state,phase="evaluation_failure",diagnostics={**state.diagnostics,"failure":event},history=state.history+(event,))
                checkpoint(state,event);return ALResult(state,"evaluation_failure",tuple(trace+[event]))
            if self._converged(diagnostics,options):
                state=replace(state,phase="numerically_feasible_stationary",diagnostics=diagnostics)
                checkpoint(state,{"kind":"numerically_feasible_stationary"})
                return ALResult(state,"numerically_feasible_stationary",tuple(trace))
            lam=state.multipliers;rho_rows=self._rho_rows(state.penalties)
            nr=p._objective.residual_count;nj=p._objective.jacobian_value_count
            def inner_kernel(x,r,j,new_point):
                if p.objective_layout is None:r[:nr]=0
                else:
                    try:_fill(self.objective_callback,x,r[:nr],j[:nj] if j is not None else None,new_point,"objective")
                    except KernelDomainRejection:return False
                if j is not None and p.objective_layout is None:j[:nj]=0
                if c is None:return
                try:_fill(self.constraint_callback,x,self._cr,self._cj if j is not None else None,new_point,"constraint")
                except KernelDomainRejection:return False
                qraw=self._cr/c.scales;shift=qraw+lam/rho_rows
                active=c.equality|(shift>0)
                r[nr:]=np.sqrt(rho_rows)*np.where(c.equality,shift,np.maximum(shift,0))
                if j is not None:
                    row_chain=np.sqrt(rho_rows)*active/c.scales
                    j[nj:]=self._cj*row_chain[c.jacobian_rows]
            # One native Solve owns its chart and all private LM state until
            # this exact AL inner problem is solved or explicitly interrupted.
            # The callback persists progress without restarting Ceres.
            inner_start_sha=hashlib.sha256(state.parameters.tobytes()).hexdigest()
            stationarity_cache={}
            def continuous_callback(event,parameters):
                event=dict(event)
                if event['initial_state'] or event['step_successful']:
                    r,j,cv,cj=self._evaluate(parameters,True)
                    qcurrent=cv/c.scales if c else np.empty(0)
                    aug=np.where(c.equality,lam+rho_rows*qcurrent,np.maximum(0,lam+rho_rows*qcurrent)) if c else np.empty(0)
                    gradient=self.backend.transpose_product(p._objective,j,r)
                    if c:gradient+=self.backend.transpose_product(c.layout,cj,aug/c.scales)
                    stationarity_cache['value']=float(np.max(np.abs(self._stationarity(parameters,gradient,r,aug)),initial=0))
                value=stationarity_cache.get('value')
                event.update(inner_continuation=event.get('inner_continuation','single_ceres_solve'),inner_start_parameters_sha256=inner_start_sha,
                    normalized_augmented_stationarity_inf=value,inner_stationarity_tolerance=options.inner_stationarity_tolerance,
                    stationarity_units='original_declared_parameter_scales',lambda_rho_fixed_within_inner=True,
                    chart_fixed_within_each_native_context=True,chart_and_lambda_rho_fixed_within_inner=not callable(self.inner_solve_callback))
                rc=on_inner_iteration(event,parameters) if on_inner_iteration is not None else 0
                if rc:return rc
                return 1 if value is not None and value<=options.inner_stationarity_tolerance else 0
            try:
                inner_scales=None
                if options.inner_parameter_metric=='unit_tangent_identity_lm':inner_scales=p.parameter_scales
                elif options.inner_parameter_metric=='full_uv_unit_tangent_identity_lm':
                    if not callable(self.inner_scale_callback):raise ValueError('full UV numerical metric requires an explicit current-J scale provider')
                    inner_scales=self.inner_scale_callback(state.parameters)
                inner_solver=self.inner_solve_callback if callable(self.inner_solve_callback) else self.backend.solve
                inner=inner_solver(p.combined_layout,state.parameters,inner_kernel,options=options.inner_options,
                    constant_blocks=p.constant_blocks,lower_bounds=p.lower_bounds,upper_bounds=p.upper_bounds,ordering_groups=p.ordering_groups,
                    on_iteration=continuous_callback if options.continuous_inner else on_inner_iteration,
                    parameter_scales=inner_scales)
            except BaseException as exc:
                event={"kind":"inner_setup_failure","error":f"{type(exc).__name__}: {exc}"}
                state=replace(state,phase="inner_setup_failure",last_inner={"error":event},history=state.history+(event,))
                checkpoint(state,event);return ALResult(state,"inner_setup_failure",tuple(trace+[event]))
            inner_record={"summary":inner.summary,"callback_errors":list(inner.callback_errors),"report":inner.report,
                "actual_accepted_steps":sum(int(e["step_successful"] and not e["initial_state"]) for e in inner.iterations)}
            if options.continuous_inner:
                inner_record['continuation']={'implementation':'verified_finite_corner_working_sets' if 'finite_corner_native_contexts' in inner.summary else 'single_ceres_solve',
                    'native_context_count':inner.summary.get('finite_corner_native_contexts',1),'start_parameters_sha256':inner_start_sha,
                    'native_private_LM_state_persisted_within_each_unchanged_chart':True,
                    'chart_reconstruction_only_for_verified_corner_active_set_changes':'finite_corner_native_contexts' in inner.summary,'chart_lambda_rho_fixed_within_this_call':True,
                    'termination_criterion':'original declared-unit augmented projected stationarity',
                    'native_iteration_time_ceilings_are_API_representation_only':True,
                    'process_recovery_semantics':'warm restart; native private LM rejection history is not serialized'}
            try:
                new_d,new_q=self._diagnostics(inner.parameters,lam,state.penalties)
                # Re-evaluate true augmented gradient. Ceres maxiter/function
                # tolerance alone does not certify this required inner solve.
                augmented_multiplier=np.where(c.equality,lam+rho_rows*new_q,np.maximum(0,lam+rho_rows*new_q)) if c else np.empty(0)
                g=self.backend.transpose_product(p._objective,self._oj,self._or)
                if c:g+=self.backend.transpose_product(c.layout,self._cj,augmented_multiplier/c.scales)
                inner_stationarity=float(np.max(np.abs(self._stationarity(inner.parameters,g,self._or,augmented_multiplier)),initial=0))
                inner_record["normalized_augmented_stationarity_inf"]=inner_stationarity
            except BaseException as exc:
                event={"kind":"inner_output_evaluation_failure","error":f"{type(exc).__name__}: {exc}"}
                state=replace(state,phase="inner_failure",last_inner=inner_record,history=state.history+(event,))
                checkpoint(state,event);return ALResult(state,"inner_failure",tuple(trace+[event]))
            if not inner.usable or inner.callback_errors:
                event={"kind":"inner_failure","termination":inner.summary["termination"],"actual_accepted_steps":inner_record["actual_accepted_steps"]}
                state=replace(state,parameters=inner.parameters,phase="inner_failure",diagnostics=new_d,last_inner=inner_record,history=state.history+(event,))
                checkpoint(state,event);return ALResult(state,"inner_failure",tuple(trace+[event]))
            if inner_stationarity>options.inner_stationarity_tolerance:
                event={"kind":"inner_incomplete","normalized_augmented_stationarity_inf":inner_stationarity,"actual_accepted_steps":inner_record["actual_accepted_steps"]}
                state=replace(state,parameters=inner.parameters,phase="inner_incomplete",diagnostics=new_d,last_inner=inner_record,history=state.history+(event,))
                checkpoint(state,event);return ALResult(state,"inner_incomplete",tuple(trace+[event]))
            new_lam=augmented_multiplier
            new_d,_=self._diagnostics(inner.parameters,new_lam,state.penalties)
            measure=self._measure(new_q,lam,state.penalties)
            improve=measure<=options.required_violation_reduction*state.previous_violation
            need_growth=(~improve)&(measure>options.feasibility_tolerance)
            penalty_limited=bool(np.any(need_growth&(state.penalties>=options.penalty_limit)))
            penalties=state.penalties.copy()
            for i in np.flatnonzero(need_growth):
                penalties[i]=min(options.penalty_limit,float(penalties[i])*options.penalty_growth)
            event={"kind":"outer_update","outer_iteration":state.outer_iteration+1,"actual_accepted_steps":inner_record["actual_accepted_steps"],
                "normalized_feasibility_inf":new_d["normalized_feasibility_inf"],"normalized_projected_stationarity_inf":new_d["normalized_projected_stationarity_inf"],
                "penalty_increased_families":[c.families[i].id for i in np.flatnonzero(penalties>state.penalties)] if c else [],"penalty_limited":penalty_limited}
            trace.append(event)
            state=replace(state,parameters=inner.parameters,multipliers=new_lam,penalties=penalties,previous_violation=measure,
                outer_iteration=state.outer_iteration+1,phase="ready",diagnostics=new_d,last_inner=inner_record,history=state.history+(event,))
            if self._converged(new_d,options):
                state=replace(state,phase="numerically_feasible_stationary")
                checkpoint(state,event);return ALResult(state,"numerically_feasible_stationary",tuple(trace))
            if penalty_limited:
                state=replace(state,phase="penalty_limit_requires_review")
                checkpoint(state,event);return ALResult(state,"penalty_limit_requires_review",tuple(trace))
            checkpoint(state,event)
        state=replace(state,phase="outer_chunk_exhausted")
        checkpoint(state,{"kind":"outer_chunk_exhausted","meaning":"resumable_not_converged"})
        return ALResult(state,"outer_chunk_exhausted",tuple(trace))
