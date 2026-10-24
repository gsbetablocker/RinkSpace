"""Current numerical factor semantics; independent of all retired solvers.

Statistical residuals use the delivered full whitening matrices or the original
independent scalar sigma. Engineering pixel budgets are separate inequalities.
No robust loss, residual normalization, or inferred covariance is added here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.sparse import csr_matrix

VERSION = "rinksolve.joint_factors.v1"


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RationalProfile:
    """A source Query parameter mapped into native image pixels.

    Alpha is the delivered Query coordinate, not a world station. Its unit is
    not inferred from a drawing normal. Polynomial coefficients are ascending.
    """
    id: str
    observation_row: int
    numerator_x: tuple[float, ...]
    numerator_y: tuple[float, ...]
    denominator: tuple[float, ...]
    alpha_origin: float
    alpha_scale: float
    alpha_bounds: tuple[float, float]

    def __post_init__(self):
        values = [*self.numerator_x, *self.numerator_y, *self.denominator,
                  self.alpha_origin, self.alpha_scale, *self.alpha_bounds]
        if not np.isfinite(values).all() or self.alpha_scale <= 0:
            raise ValueError("nonfinite Query or nonpositive alpha scale")
        if len(self.denominator) == 0 or len(self.numerator_x) == 0 or len(self.numerator_y) == 0:
            raise ValueError("empty rational Query polynomial")
        if self.alpha_bounds[0] > self.alpha_bounds[1]:
            raise ValueError("reversed Query alpha bounds")
        # A rational chart cannot cross a pole inside its declared source range.
        roots = np.polynomial.polynomial.polyroots(self.denominator)
        lo, hi = [(x-self.alpha_origin)/self.alpha_scale for x in self.alpha_bounds]
        if any(abs(r.imag) <= 1e-12 and lo <= r.real <= hi for r in roots):
            raise ValueError("Query denominator has a pole in its source domain")

    def evaluate(self, alpha, *, derivative=False):
        a = np.asarray(alpha, dtype=np.float64)
        if not np.isfinite(a).all() or np.any(a < self.alpha_bounds[0]) or np.any(a > self.alpha_bounds[1]):
            raise ValueError("Query alpha outside current source domain; no clipping")
        z = (a-self.alpha_origin)/self.alpha_scale
        den = np.polynomial.polynomial.polyval(z,self.denominator)
        if np.any(den == 0) or not np.isfinite(den).all():
            raise ValueError("Query rational denominator invalid")
        numer = np.stack([np.polynomial.polynomial.polyval(z,c) for c in (self.numerator_x,self.numerator_y)],axis=-1)
        value = numer/den[...,None]
        if not derivative:
            return value
        dn = np.stack([np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c)) for c in (self.numerator_x,self.numerator_y)],axis=-1)
        dd = np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(self.denominator))
        jac = (dn*den[...,None]-numer*dd[...,None])/(den[...,None]**2*self.alpha_scale)
        return value,jac

    @classmethod
    def from_current(cls, query, row, bounds):
        return cls("query:"+str(query["row_index"]),int(row),tuple(query["numerator_x"]),
            tuple(query["numerator_y"]),tuple(query["denominator"]),float(query["alpha_origin"]),
            float(query["alpha_scale"]),tuple(map(float,bounds)))


@dataclass(frozen=True)
class GroupWhitening:
    """One indivisible current source group, W columns interleaved u,v.

    Rectangular and rank-deficient source support is preserved. Rank zero means
    no statistical residual, never an identity covariance or a unit weight.
    """
    id: str
    rows: np.ndarray
    matrix: np.ndarray
    statistic_kind: str = "original"
    source_group_id: str | None = None

    def __post_init__(self):
        rows = np.asarray(self.rows)
        w = np.asarray(self.matrix,dtype=np.float64)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows) or np.any(rows<0):
            raise ValueError("group rows require unique nonnegative integer addresses")
        if w.size == 0 and w.shape != (0,2*len(rows)):
            w = w.reshape(0,2*len(rows))
        if w.ndim != 2 or w.shape[1] != 2*len(rows) or not np.isfinite(w).all():
            raise ValueError("group W dimensions or values differ from full source support")
        if w.shape[0] == 0 and self.statistic_kind != "constraint_only_unknown_noise":
            raise ValueError("rank-zero group must explicitly declare unknown noise")
        object.__setattr__(self,"rows",_readonly(rows,np.int64))
        object.__setattr__(self,"matrix",_readonly(w))

    @property
    def rank(self):
        return self.matrix.shape[0]

    def residual(self, native_error):
        e=np.asarray(native_error,dtype=np.float64)
        if e.shape != (len(self.rows),2) or not np.isfinite(e).all():
            raise ValueError("group error must cover all original members exactly once")
        return self.matrix@e.reshape(-1)

    def jacobian(self, native_jacobian):
        j=np.asarray(native_jacobian,dtype=np.float64)
        if j.ndim<3 or j.shape[:2] != (len(self.rows),2) or not np.isfinite(j).all():
            raise ValueError("group Jacobian must cover complete native u,v support")
        return np.tensordot(self.matrix,j.reshape((2*len(self.rows),)+j.shape[2:]),axes=(1,0))

    def with_rows(self, rows):
        if not np.array_equal(np.asarray(rows),self.rows):
            raise ValueError("cannot drop/reorder a member and retain the original W")
        return self


@dataclass(frozen=True)
class StatisticalFactor:
    id: str
    rows: np.ndarray
    residual_start: int
    residual_count: int
    kind: str


class WhiteningPlan:
    """Sparse exact linear whitening shared by batch residuals and Jacobians.

    All input rows have one native 2-vector. A factor may use many dependent
    observations. The returned CSR can be multiplied on the GPU after an exact
    transfer; no per-member Python callbacks are necessary.
    """
    def __init__(self, row_count:int, groups:Sequence[GroupWhitening], independent_rows, sigma, *, independent_ids=None):
        self.row_count=int(row_count)
        if self.row_count<0:
            raise ValueError("negative row count")
        independent_rows=np.asarray(independent_rows,dtype=np.int64)
        sigma=np.asarray(sigma,dtype=np.float64)
        if independent_rows.ndim!=1 or sigma.shape!=independent_rows.shape or not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise ValueError("independent rows require their original positive scalar sigma")
        if independent_ids is None:
            independent_ids=["independent_local_factor:"+str(i) for i in range(len(independent_rows))]
        if len(independent_ids)!=len(independent_rows) or len(set(independent_ids))!=len(independent_ids):
            raise ValueError("independent factor identities must be complete and unique")
        claimed=np.zeros(self.row_count,dtype=np.uint8)
        rr=[];cc=[];vv=[];factors=[];offset=0
        for g in groups:
            if np.any(g.rows>=self.row_count):
                raise ValueError("group row outside current prepared table")
            if np.any(claimed[g.rows]):
                raise ValueError("statistical source support counted more than once")
            claimed[g.rows]=1
            if g.rank==0:
                continue
            r,c=np.nonzero(g.matrix)
            rr.extend((r+offset).tolist());cc.extend((2*g.rows[c//2]+c%2).tolist());vv.extend(g.matrix[r,c].tolist())
            factors.append(StatisticalFactor(g.id,g.rows,offset,g.rank,g.statistic_kind));offset+=g.rank
        if len(np.unique(independent_rows))!=len(independent_rows) or np.any(independent_rows<0) or np.any(independent_rows>=self.row_count):
            raise ValueError("invalid/duplicate independent source support")
        if np.any(claimed[independent_rows]):
            raise ValueError("independent row is already in a source group")
        for row,s,identifier in zip(independent_rows,sigma,independent_ids):
            rr.extend([offset,offset+1]);cc.extend([2*int(row),2*int(row)+1]);vv.extend([1/float(s)]*2)
            factors.append(StatisticalFactor(identifier,_readonly([row],np.int64),offset,2,"original_independent_scalar_sigma"));offset+=2
        self.operator=csr_matrix((vv,(rr,cc)),shape=(offset,2*self.row_count),dtype=np.float64)
        self.factors=tuple(factors)
        self.used_rows=_readonly(np.unique(self.operator.indices//2),np.int64)
        self.independent_rows=_readonly(independent_rows,np.int64)
        self.sigma=_readonly(sigma)

    @property
    def residual_count(self):
        return self.operator.shape[0]

    def apply(self, native_error):
        e=np.asarray(native_error,dtype=np.float64)
        if e.shape!=(self.row_count,2):
            raise ValueError("native batch error shape mismatch")
        # Unused rows may be NaN; each active factor must have a finite support.
        used=self.used_rows
        if not np.isfinite(e[used]).all():
            raise ValueError("nonfinite active factor support: reject evaluation, do not silently change objective")
        clean=np.zeros_like(e);clean[used]=e[used]
        return np.asarray(self.operator@clean.reshape(-1))

    def jacobian(self, native_jacobian):
        j=np.asarray(native_jacobian,dtype=np.float64)
        if j.ndim!=3 or j.shape[:2]!=(self.row_count,2):
            raise ValueError("native batch Jacobian shape mismatch")
        used=self.used_rows
        if not np.isfinite(j[used]).all():
            raise ValueError("nonfinite active Jacobian support")
        clean=np.zeros_like(j);clean[used]=j[used]
        return np.asarray(self.operator@clean.reshape(2*self.row_count,-1))

    def pullback(self, whitened_gradient):
        v=np.asarray(whitened_gradient,dtype=np.float64)
        if v.shape!=(self.residual_count,) or not np.isfinite(v).all():
            raise ValueError("whitened gradient shape/value mismatch")
        return np.asarray(self.operator.T@v).reshape(self.row_count,2)


def native_errors(predicted_uv, immutable_uv, *, profiles=(), profile_alpha=None):
    """Original native pixel error for statistics, for EVERY semantic kind.

    The former profile substitution was wrong: Q(alpha) is a separate source
    query equality, not the target of W or independent-sigma residuals. Optional
    old keyword arguments remain accepted but never change this statistic.
    """
    p,q=np.asarray(predicted_uv,dtype=np.float64),np.asarray(immutable_uv,dtype=np.float64)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1]!=2:
        raise ValueError("native image arrays require matching (N,2)")
    return p-q,{}


def profile_query_equalities(predicted_uv,profiles,profile_alpha):
    """Two exact native-image query equations, independent of W statistics."""
    p=np.asarray(predicted_uv,dtype=np.float64);alpha=np.asarray(profile_alpha,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or alpha.shape!=(len(profiles),):
        raise ValueError("one current Query alpha and native UV table required")
    rows=[q.observation_row for q in profiles]
    if len(rows)!=len(set(rows)) or any(row<0 or row>=len(p) for row in rows):
        raise ValueError("duplicate/out-of-range source Query observation")
    values=np.empty((len(profiles),2));alpha_jac=np.empty_like(values)
    for i,q in enumerate(profiles):
        target,derivative=q.evaluate(alpha[i],derivative=True)
        values[i]=p[q.observation_row]-target;alpha_jac[i]=-derivative
    return values,alpha_jac


def source_budget_values(native_error,budget_px,semantic_kind,profile_normal,*,budget_sq=1.,jacobian=None):
    """Original all-source q-b <= 0, with ONE shared global budget scalar.

    Ordinary finite-curve/corner q=||pred-raw||^2/E^2. Genuine-profile q uses
    the original source normal projection of that same raw error, while its
    independent two-component Q(alpha) equality remains installed separately.
    E is an engineering native-pixel scale; it is never a source sigma.
    """
    e=np.asarray(native_error,dtype=np.float64);E=np.asarray(budget_px,dtype=np.float64)
    kinds=np.asarray(semantic_kind);normal=np.asarray(profile_normal,dtype=np.float64)
    if e.ndim!=2 or e.shape[1]!=2 or E.shape!=(len(e),) or kinds.shape!=(len(e),) or normal.shape!=e.shape:
        raise ValueError("current all-source budget arrays have inconsistent shape")
    if not np.isfinite(e).all() or not np.isfinite(E).all() or np.any(E<=0) or not np.isin(kinds,[0,1,2]).all():
        raise ValueError("current source errors, positive E or semantic kind invalid")
    b=np.asarray(budget_sq,dtype=np.float64)
    if b.shape!=() or not np.isfinite(b) or b<0:
        raise ValueError("budget_sq is one nonnegative finite global scalar")
    profile=kinds==2
    if not np.isfinite(normal[profile]).all() or np.any(np.abs(np.linalg.norm(normal[profile],axis=1)-1.)>1e-8):
        raise ValueError("genuine profile requires its original unit source normal")
    gradient=2*e/(E[:,None]**2);q=np.sum(e*e,axis=1)/(E*E)
    projection=np.einsum('ni,ni->n',normal[profile],e[profile])
    q[profile]=projection**2/(E[profile]**2)
    gradient[profile]=2*projection[:,None]*normal[profile]/(E[profile,None]**2)
    if jacobian is None:
        return q-float(b)
    j=np.asarray(jacobian,dtype=np.float64)
    if j.ndim!=3 or j.shape[:2]!=e.shape or not np.isfinite(j).all():
        raise ValueError("current source native Jacobian shape/value mismatch")
    return q-float(b),np.einsum('ni,nip->np',gradient,j)


def engineering_budget_values(native_error, budget_px, *, jacobian=None):
    """g=||native error||^2/E^2-1 <= 0; E is a budget, never sigma.

    No statistical factor or covariance is created by this operation. Group and
    shared-pixel dependence stays in preparation metadata for review and repair.
    """
    e=np.asarray(native_error,dtype=np.float64);E=np.asarray(budget_px,dtype=np.float64)
    if e.ndim!=2 or e.shape[1]!=2 or E.shape!=(len(e),) or not np.isfinite(e).all() or not np.isfinite(E).all() or np.any(E<=0):
        raise ValueError("finite native errors and positive current pixel budgets required")
    value=np.sum(e*e,axis=1)/(E*E)-1
    if jacobian is None:
        return value
    j=np.asarray(jacobian,dtype=np.float64)
    if j.ndim!=3 or j.shape[:2]!=e.shape or not np.isfinite(j).all():
        raise ValueError("budget Jacobian shape/value mismatch")
    return value,2*np.einsum('ni,nip->np',e,j)/(E[:,None]**2)
