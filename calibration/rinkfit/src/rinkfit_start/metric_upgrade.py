"""S3 initializer: one projective camera from explicit metric plane upgrades.

For q=Q_j xi and physical plane embedding X_h=M_j xi, solve
H_j Q_j ~ P M_j with one shared P and one scale per plane.  This is a
conditional initialization primitive, never an additional final data factor.
Unknown Q is not replaced by identity; one plane cannot recover general P.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class MetricUpgradeStatus(str, Enum):
    INITIALIZER_READY = "initializer_ready"
    WEAK_INITIALIZER = "weak_initializer"
    DEGENERATE = "degenerate"
    MISSING_METRIC_UPGRADE = "missing_metric_upgrade"
    INVALID_INPUT = "invalid_input"


def _array(value, shape, name):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class PlaneHomographyInput:
    """One genuine plane's same-frame atlas H and caller-supplied upgrade.

    metric_chart_kind is either 'independent_metric_chart' or
    'conditional_world_seed'.  The latter is required when Q=I merely
    reuses an old proposed world; identity is not evidence of metric truth.
    frame_key must identify one source clock/native frame, shared by all inputs.
    """

    plane_id: str
    frame_key: str
    M: np.ndarray
    H_atlas_to_native: np.ndarray
    Q_metric_to_atlas: Optional[np.ndarray]
    metric_chart_provenance: str
    genuine_plane: bool
    metric_chart_kind: str = "conditional_world_seed"

    def __post_init__(self):
        object.__setattr__(self, "M", _array(self.M, (4, 3), "physical plane M"))
        object.__setattr__(self, "H_atlas_to_native", _array(self.H_atlas_to_native, (3, 3), "atlas-to-native H"))
        if self.Q_metric_to_atlas is not None:
            object.__setattr__(self, "Q_metric_to_atlas", _array(self.Q_metric_to_atlas, (3, 3), "explicit Q"))
        if not self.plane_id.strip() or not self.frame_key.strip():
            raise ValueError("plane ID and actual same-frame key are required")
        if self.metric_chart_kind not in ("independent_metric_chart", "conditional_world_seed"):
            raise ValueError("unknown metric_chart_kind; do not silently mark an unupgraded atlas as metric")


@dataclass(frozen=True)
class PlaneUpgradeReplay:
    plane_id: str
    fitted_homography_scale: float
    relative_matrix_residual: float
    metric_chart_kind: str
    metric_chart_provenance: str


@dataclass(frozen=True)
class MetricUpgradeResult:
    status: MetricUpgradeStatus
    P: Optional[np.ndarray] = None
    frame_key: Optional[str] = None
    metric_status: str = "not_upgraded"
    design_shape: tuple[int, int] = (0, 0)
    normalized_singular_values: tuple[float, ...] = ()
    numerical_rank: int = 0
    numerical_nullity: int = 0
    expected_projective_rank: int = 0
    nullspace_gap: Optional[float] = None
    identifiable_subspace_condition: Optional[float] = None
    normalized_algebraic_residual: Optional[float] = None
    plane_replay: tuple[PlaneUpgradeReplay, ...] = ()
    row_scales: tuple[float, ...] = ()
    column_scales: tuple[float, ...] = ()
    rank_relative_tolerance: float = 1e-10
    minimum_nullspace_gap: float = 10.0
    maximum_identifiable_condition: float = 1e8
    initializer_only: bool = True
    final_physical_camera_recovered: bool = False
    message: str = ""

    def __post_init__(self):
        if self.P is not None:
            object.__setattr__(self, "P", _array(self.P, (3, 4), "shared projection P"))

    @property
    def ready(self):
        return self.status is MetricUpgradeStatus.INITIALIZER_READY

    def require_projection(self):
        if not self.ready or self.P is None:
            raise ValueError(f"{self.status.value}: {self.message}")
        return self.P


def common_projection_from_planes(
    planes: Sequence[PlaneHomographyInput], *, rank_relative_tolerance=1e-10,
    minimum_nullspace_gap=10.0, maximum_identifiable_condition=1e8,
) -> MetricUpgradeResult:
    """Joint scale DLT with honest rank/weakness and conditional-world labels.

    A_j=H_j Q_j.  Each of nine equations per plane is
    sum_l P[r,l]*M_j[l,c] - lambda_j*A_j[r,c] = 0.
    P has 12 coefficients and each plane one nuisance scale.  Input block
    Frobenius scaling, row scaling and column equilibration improve numerical
    conditioning; the reported spectrum is for this *declared* normalized
    design, not for the final pixel likelihood/information matrix.

    Full SVD preserves all null vectors in an underdetermined design.  A
    projective camera needs a unique smallest direction: rank >= n_unknown-1
    and a declared smallest-direction gap.  With noisy H, algebraic rank may
    equal n_unknown; the smallest singular vector is then an approximate DLT
    initializer, with its residual and gap reported, not an exact solution.
    """
    planes = tuple(planes)
    if (not np.isfinite(rank_relative_tolerance) or not 0 < rank_relative_tolerance < 1 or
            not np.isfinite(minimum_nullspace_gap) or minimum_nullspace_gap <= 1 or
            not np.isfinite(maximum_identifiable_condition) or maximum_identifiable_condition <= 1):
        raise ValueError("freeze finite rank tolerance in (0,1), minimum gap >1 and maximum condition >1")
    settings = dict(rank_relative_tolerance=float(rank_relative_tolerance),
                    minimum_nullspace_gap=float(minimum_nullspace_gap),
                    maximum_identifiable_condition=float(maximum_identifiable_condition))
    if not planes:
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **settings,
                                   message="at least two same-frame genuine planes with explicit metric upgrades are needed")
    if not all(isinstance(p, PlaneHomographyInput) for p in planes):
        raise TypeError("all inputs must be PlaneHomographyInput")
    frame_key = planes[0].frame_key
    common = dict(frame_key=frame_key, **settings)
    if any(p.frame_key != frame_key for p in planes):
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,
                                   message="zoom/motion-varying frame homographies cannot share one P")
    if len({p.plane_id for p in planes}) != len(planes):
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,
                                   message="duplicate plane IDs cannot create independent plane support")
    if any(not p.genuine_plane for p in planes):
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,
                                   message="curved board sheets and arbitrary patches are not genuine planes")
    missing = [p.plane_id for p in planes
               if p.Q_metric_to_atlas is None or not p.metric_chart_provenance.strip()]
    if missing:
        return MetricUpgradeResult(MetricUpgradeStatus.MISSING_METRIC_UPGRADE, **common,
                                   message="explicit Q and metric-chart provenance missing for: "+", ".join(missing))
    for p in planes:
        for name, matrix, expected in (("M",p.M,3), ("H",p.H_atlas_to_native,3), ("Q",p.Q_metric_to_atlas,3)):
            if np.linalg.matrix_rank(matrix, tol=rank_relative_tolerance*np.linalg.norm(matrix,2)) != expected:
                return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,
                                           message=f"{p.plane_id}: {name} is singular/degenerate at the frozen tolerance")
        # M must be an affine Euclidean chart: X_h=M*[xi1,xi2,1].
        # An unconstrained physical 4x3 projective block is not an upgrade.
        if not np.allclose(p.M[3,:], [0.,0.,1.], atol=1e-12, rtol=0.):
            return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,
                                       message=f"{p.plane_id}: M must be an explicit affine physical embedding with last row [0,0,1]")
    metric_status = ("conditional_world_seed" if any(p.metric_chart_kind == "conditional_world_seed" for p in planes)
                     else "caller_supplied_metric_charts")
    n_unknown = 12+len(planes)
    design = np.zeros((9*len(planes),n_unknown),dtype=np.float64)
    for j,p in enumerate(planes):
        with np.errstate(over="ignore",invalid="ignore"):
            A = p.H_atlas_to_native @ p.Q_metric_to_atlas
            a_norm = np.linalg.norm(A)
        if not np.all(np.isfinite(A)) or not np.isfinite(a_norm) or a_norm <= 0:
            return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,metric_status=metric_status,
                                       message=f"{p.plane_id}: H Q is outside finite nonzero arithmetic")
        A = A/a_norm
        M = p.M/np.linalg.norm(p.M)
        for r in range(3):
            for c in range(3):
                equation = 9*j+3*r+c
                design[equation,4*r:4*r+4] = M[:,c]
                design[equation,12+j] = -A[r,c]
    row_norms = np.linalg.norm(design,axis=1)
    if np.any(row_norms == 0):
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,metric_status=metric_status,
                                   message="zero design equation from degenerate material support")
    row_scaled = design/row_norms[:,None]
    column_norms = np.linalg.norm(row_scaled,axis=0)
    # Zero columns represent actual missing support and are preserved as zero;
    # unit numerical scaling is not a constraint or a regularization prior.
    column_norms[column_norms == 0] = 1.0
    normalized = row_scaled/column_norms[None,:]
    try:
        _,s,Vt = np.linalg.svd(normalized,full_matrices=True)
    except np.linalg.LinAlgError as exc:
        return MetricUpgradeResult(MetricUpgradeStatus.INVALID_INPUT, **common,metric_status=metric_status,
                                   message="normalized DLT factorization failed: "+str(exc))
    singular = np.zeros(n_unknown)
    singular[:len(s)] = s
    cutoff = rank_relative_tolerance*singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    nullity = n_unknown-rank
    gap = float(singular[-2]/max(singular[-1],np.finfo(float).eps*singular[0]))
    condition = (float(singular[0]/singular[-2]) if singular[-2] > 0 else float("inf"))
    diagnostics = dict(metric_status=metric_status,design_shape=design.shape,
                       normalized_singular_values=tuple(float(v) for v in singular),
                       numerical_rank=rank,numerical_nullity=nullity,
                       expected_projective_rank=n_unknown-1,nullspace_gap=gap,
                       identifiable_subspace_condition=condition,
                       row_scales=tuple(float(v) for v in row_norms),
                       column_scales=tuple(float(v) for v in column_norms))
    if len(planes) < 2 or rank < n_unknown-1:
        return MetricUpgradeResult(MetricUpgradeStatus.DEGENERATE, **common,**diagnostics,
                                   message="plane support leaves multiple projective directions; a single plane cannot recover a general 3D camera")
    normalized_vector = Vt[-1]
    unscaled = normalized_vector/column_norms
    P = unscaled[:12].reshape(3,4)
    norm = float(np.linalg.norm(P))
    if norm <= np.finfo(float).eps:
        return MetricUpgradeResult(MetricUpgradeStatus.DEGENERATE, **common,**diagnostics,
                                   message="smallest algebraic direction contains no usable camera component")
    P = P/norm
    # Deterministic projective sign only.  It is not a positive-depth claim.
    largest = np.unravel_index(np.argmax(np.abs(P)),P.shape)
    if P[largest] < 0:
        P = -P
    if np.linalg.matrix_rank(P, tol=rank_relative_tolerance*np.linalg.norm(P,2)) < 3:
        return MetricUpgradeResult(MetricUpgradeStatus.DEGENERATE, **common,**diagnostics,
                                   message="recovered shared projection is rank deficient")
    replays = []
    for p in planes:
        A = p.H_atlas_to_native @ p.Q_metric_to_atlas
        predicted = P @ p.M
        scale = float(np.sum(predicted*A)/np.sum(A*A))
        relative = float(np.linalg.norm(predicted-scale*A)/np.linalg.norm(predicted))
        replays.append(PlaneUpgradeReplay(p.plane_id,scale,relative,p.metric_chart_kind,p.metric_chart_provenance))
    algebraic = float(np.linalg.norm(normalized@normalized_vector))
    weak = gap < minimum_nullspace_gap or condition > maximum_identifiable_condition
    return MetricUpgradeResult(MetricUpgradeStatus.WEAK_INITIALIZER if weak else MetricUpgradeStatus.INITIALIZER_READY,
                               P=P,**common,**diagnostics,normalized_algebraic_residual=algebraic,
                               plane_replay=tuple(replays),
                               message=("smallest singular direction is weak; retain candidate and seek independent geometry" if weak else
                                        "shared projective initialization only; recover/check native optics, depth and fixed head before physical acceptance"))


# IMPLEMENTED: explicit Q/M provenance, same-frame genuine-plane validation,
# joint common-P/per-plane-scale DLT, declared normalization, full rank/nullity
# and weak-direction diagnostics, conditional metric labels and plane replay.
# NOT IMPLEMENTED: estimating an unknown Q, RQ K/R/C decomposition, positive-
# depth/orientation certification, rolling-shutter homography equivalence,
# fixed-head retraction, final physical constraints or any real-shot fit.
