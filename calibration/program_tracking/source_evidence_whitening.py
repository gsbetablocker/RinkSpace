"""Whiten actual source evidence while preserving every original FIT address.

The input is an unscaled, interleaved ``[rx, ry, ...]`` pixel residual and its
Jacobian.  If the existing residual builder has divided by a row's scales,
multiply both outputs by ``repeat(scales, 2)`` before applying an operator.
No source UV, camera, material identity, finite-curve guard or source clock is
changed here.  The dataset consumer must enforce those guards first.

A measured straight line has two uncertain coefficients, however many visible
representatives it has.  Whitening uses the measured source normal, rather
than treating the representatives as independent pixels.  Intersections and
the readouts that generated them can share the same original noise variables.
The retained point IDs are the FIT membership, not a count of independent
measurements.  Missing covariance or readout scales is an error, never sigma=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class MissingSourceEvidence(ValueError):
    """An authenticated source measurement lacks the required uncertainty."""


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result.copy()


def _ids(values):
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x for x in result) or len(set(result)) != len(result):
        raise ValueError("Every original source point requires one unique nonempty ID")
    return result


def _psd_factor(value, name):
    covariance = np.asarray(value, dtype=np.float64)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not len(covariance) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)):
        raise ValueError(f"{name} must be a finite symmetric covariance")
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    tolerance = np.finfo(float).eps * max(1, len(covariance)) * np.max(np.abs(eigenvalues))
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    keep = eigenvalues > tolerance
    if not keep.any():
        raise MissingSourceEvidence(f"{name} has no positive source uncertainty")
    return vectors[:, keep] * np.sqrt(eigenvalues[keep])


@dataclass(frozen=True)
class SourceWhiteningOperator:
    """Low rank source operator; all member addresses enter its raw residual."""

    matrix: np.ndarray
    source_point_ids: tuple[str, ...]
    group_id: str
    covariance_origin: str
    numerical_rank_evidence: Mapping | None = None

    def __post_init__(self):
        ids = _ids(self.source_point_ids)
        matrix = np.asarray(self.matrix, float).copy()
        if (matrix.ndim != 2 or matrix.shape[1] != 2 * len(ids)
                or not matrix.shape[0] or not np.isfinite(matrix).all()
                or not self.group_id or not self.covariance_origin):
            raise ValueError("Invalid explicit source whitening operator")
        # Every retained FIT member must influence the actual whitened residual.
        if (np.linalg.norm(matrix.reshape(matrix.shape[0], len(ids), 2), axis=(0, 2)) == 0).any():
            raise ValueError("A FIT member would be decorative rather than consumed")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source_point_ids", ids)

    @property
    def rank(self):
        return self.matrix.shape[0]

    @property
    def consumed_point_count(self):
        return len(self.source_point_ids)

    def apply(self, pixel_residual, pixel_jacobian=None):
        residual = _finite(pixel_residual, (self.matrix.shape[1],), "raw pixel residual")
        whitened = self.matrix @ residual
        if pixel_jacobian is None:
            return whitened
        jacobian = np.asarray(pixel_jacobian, float)
        if (jacobian.ndim != 2 or jacobian.shape[0] != len(residual)
                or not np.isfinite(jacobian).all()):
            raise ValueError("Raw pixel Jacobian has an incompatible source inventory")
        return whitened, self.matrix @ jacobian

    def metadata(self):
        return dict(schema="rinkspace.source-evidence-whitening.v1", group_id=self.group_id,
                    source_point_ids=list(self.source_point_ids), consumed_point_count=self.consumed_point_count,
                    raw_pixel_residual_dimension=self.matrix.shape[1], covariance_rank=self.rank,
                    covariance_origin=self.covariance_origin, all_members_consumed=True,
                    original_uv_modified=False, representative_count_increases_source_information=False)

    def to_record(self):
        """A complete JSON record; store larger matrices in the existing Data area."""
        result = dict(self.metadata(), whitening_matrix=self.matrix.tolist())
        if self.numerical_rank_evidence is not None:
            result["numerical_rank_evidence"] = dict(self.numerical_rank_evidence)
        return result

    @classmethod
    def from_record(cls, record):
        if record.get("schema") != "rinkspace.source-evidence-whitening.v1":
            raise ValueError("Explicit source whitening record required")
        operator = cls(record["whitening_matrix"], tuple(record["source_point_ids"]),
                       record["group_id"], record["covariance_origin"], record.get("numerical_rank_evidence"))
        if (record.get("consumed_point_count") != operator.consumed_point_count
                or record.get("covariance_rank") != operator.rank
                or record.get("raw_pixel_residual_dimension") != operator.matrix.shape[1]):
            raise ValueError("Stored operator membership or residual dimension differs")
        return operator


def latent_whitening_operator(source_point_ids, sensitivity, latent_covariance=None, *, group_id,
                              covariance_origin, explicit_noise_factor=None, physical_basis=None):
    """Whiten C = B Cov(latent) B.T, without a diagonal floor or fake noise.

    With ``explicit_noise_factor``, preserve the original readout influence
    times its native sigma directly; do not square its condition number by
    constructing/factoring a Gram covariance. ``physical_basis`` is an
    orthonormal row map from the 2N pixel residual to the physically measured
    scalars (one normal per curve point, two coordinates per corner). In that
    case ``sensitivity`` is already assembled in this physical scalar basis.

    The numerical rank threshold covers the upstream B@L multiplication's
    backward error as well as SVD error. Tiny cancellation residuals cannot
    become additional, overwhelmingly confident source measurements.
    """
    ids = _ids(source_point_ids)
    factor_from_covariance = explicit_noise_factor is None
    if factor_from_covariance:
        factor = _psd_factor(latent_covariance, "latent source covariance")
    else:
        factor = np.asarray(explicit_noise_factor, float)
        if factor.ndim != 2 or not all(factor.shape) or not np.isfinite(factor).all():
            raise ValueError("An explicit finite native readout noise factor is required")
    if physical_basis is None:
        basis = np.eye(2*len(ids))
    else:
        basis = np.asarray(physical_basis, float)
        if (basis.ndim != 2 or basis.shape[1] != 2*len(ids) or not basis.shape[0]
                or not np.isfinite(basis).all()):
            raise ValueError("The physical source residual basis has incompatible dimensions")
        orthogonality_error = np.linalg.norm(basis @ basis.T-np.eye(len(basis)), ord=2)
        if orthogonality_error > 8*np.finfo(float).eps*max(basis.shape):
            raise ValueError("The declared physical source residual rows must be orthonormal")
    sensitivity = _finite(sensitivity, (basis.shape[0], factor.shape[0]), "source sensitivity")
    propagated = sensitivity @ factor
    u, singular, _ = np.linalg.svd(propagated, full_matrices=False)
    eps = np.finfo(float).eps
    terms = 2*factor.shape[0]+2
    gamma = terms*eps/(1-terms*eps)
    multiplication_error = gamma*np.linalg.norm(np.abs(sensitivity) @ np.abs(factor), ord="fro")
    svd_error = eps*max(propagated.shape)*np.linalg.norm(propagated, ord="fro")
    covariance_factor_error = 0.
    if factor_from_covariance:
        covariance = np.asarray(latent_covariance, float)
        # A covariance supplied without its original factor has lost that
        # factor's lineage. Propagate its eigensolver/Gram rounding envelope;
        # callers with readout influence should use the explicit factor path.
        covariance_error = eps*max(covariance.shape)*np.linalg.norm(covariance, ord=2)
        covariance_factor_error = np.linalg.norm(sensitivity, ord=2)*np.sqrt(covariance_error)
    tolerance = multiplication_error+svd_error+covariance_factor_error
    keep = singular > tolerance
    if not keep.any():
        raise MissingSourceEvidence("Source evidence supports no residual mode")
    audit = dict(method="physical_source_factor_SVD_with_propagation_backward_error",
        original_readout_noise_factor_used=not factor_from_covariance,
        physical_residual_dimension=basis.shape[0], latent_dimension=factor.shape[0],
        native_noise_dimension=factor.shape[1], singular_values=singular.tolist(),
        multiplication_backward_error_bound=float(multiplication_error),
        SVD_backward_error_bound=float(svd_error),
        covariance_factor_error_bound=float(covariance_factor_error),
        numerical_rank_threshold=float(tolerance), retained_rank=int(keep.sum()),
        unresolvable_modes_removed=int((~keep).sum()), source_point_membership_reduced=False)
    return SourceWhiteningOperator(((u[:, keep] / singular[keep]).T @ basis), ids,
                                   group_id, covariance_origin, audit)


@dataclass(frozen=True)
class SourceLineEvidence:
    """dependent coordinate = a * independent coordinate + b.

    Axis 0 means y=a*x+b; axis 1 means x=a*y+b.  Optional readout lineage is
    required when a corner and its original individual edge readouts share a
    joint group.  A provided coefficient covariance alone suffices for a group
    consisting only of line representatives and derived intersections.
    """

    coefficients: np.ndarray
    covariance_ab: np.ndarray
    axis: int
    group_id: str
    covariance_origin: str
    readout_ids: tuple[str, ...] = ()
    readout_sigmas: np.ndarray | None = None
    coefficient_influence: np.ndarray | None = None

    def __post_init__(self):
        if self.axis not in (0, 1) or not self.group_id or not self.covariance_origin:
            raise ValueError("A source line requires explicit axis and identity")
        object.__setattr__(self, "coefficients", _finite(self.coefficients, (2,), "source line coefficients"))
        cov = _finite(self.covariance_ab, (2, 2), "source coefficient covariance")
        _psd_factor(cov, "source coefficient covariance")
        object.__setattr__(self, "covariance_ab", cov)
        if self.readout_ids:
            ids = _ids(self.readout_ids)
            sigma = _finite(self.readout_sigmas, (len(ids),), "source readout scales")
            influence = _finite(self.coefficient_influence, (2, len(ids)), "source fit influence")
            if (sigma <= 0).any():
                raise MissingSourceEvidence("Positive measured readout scales required")
            propagated = (influence * sigma) @ (influence * sigma).T
            if not np.allclose(propagated, cov, rtol=1e-8, atol=1e-12):
                raise ValueError("Declared coefficient covariance differs from its readout lineage")
            object.__setattr__(self, "readout_ids", ids)
            object.__setattr__(self, "readout_sigmas", sigma)
            object.__setattr__(self, "coefficient_influence", influence)
        elif self.readout_sigmas is not None or self.coefficient_influence is not None:
            raise ValueError("Partial source readout lineage is not allowed")

    @property
    def normal(self):
        a = self.coefficients[0]
        n = np.array([-a, 1.]) if self.axis == 0 else np.array([1., -a])
        return n / np.linalg.norm(n)

    def metadata(self):
        result = dict(schema="rinkspace.source-line-uncertainty.v1", axis=self.axis,
                      coefficients_ab=self.coefficients.tolist(), covariance_ab=self.covariance_ab.tolist(),
                      group_id=self.group_id, covariance_origin=self.covariance_origin,
                      source_normal_xy=self.normal.tolist())
        if self.readout_ids:
            result.update(readout_ids=list(self.readout_ids), readout_sigmas=self.readout_sigmas.tolist(),
                          coefficient_influence=self.coefficient_influence.tolist())
        return result

    @classmethod
    def from_record(cls, record):
        if record.get("schema") != "rinkspace.source-line-uncertainty.v1":
            raise ValueError("Explicit source line uncertainty record required")
        return cls(record["coefficients_ab"], record["covariance_ab"], record["axis"],
                   record["group_id"], record["covariance_origin"], tuple(record.get("readout_ids", ())),
                   record.get("readout_sigmas"), record.get("coefficient_influence"))


def line_evidence_from_readouts(uv, sigma, coefficients, *, axis, group_id, estimator,
                               readout_ids=None, soft_l1_f_scale=None):
    """Propagate the original readout noise through the recorded line estimator.

    The observed coefficients are retained. ``weighted_linear`` uses supplied
    localization scales. ``weighted_soft_l1`` uses residual/sigma and requires
    an explicit loss scale; ``unweighted_soft_l1`` requires its pixel loss scale.
    No remeasurement or optimization is performed.
    """
    points = np.asarray(uv, float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2 or not np.isfinite(points).all():
        raise MissingSourceEvidence("At least two finite original source readouts required")
    scales = _finite(sigma, (len(points),), "source localization scales")
    if (scales <= 0).any() or axis not in (0, 1):
        raise MissingSourceEvidence("Explicit positive source scales and axis required")
    coefficient = _finite(coefficients, (2,), "original source line coefficients")
    x, y = points[:, axis], points[:, 1-axis]
    center = float(np.mean(x)); span = float(np.ptp(x))
    if span <= 0:
        raise MissingSourceEvidence("Source readouts cannot identify a line slope")
    design = np.column_stack(((x-center)/span, np.ones(len(x))))
    residual = coefficient[0]*x+coefficient[1]-y
    if estimator == "weighted_linear":
        weights = 1 / scales**2
    elif estimator in ("weighted_soft_l1", "unweighted_soft_l1"):
        if soft_l1_f_scale is None or not np.isfinite(soft_l1_f_scale) or soft_l1_f_scale <= 0:
            raise MissingSourceEvidence("Recorded soft_l1 loss scale required")
        normalized = residual / (scales if estimator == "weighted_soft_l1" else 1) / soft_l1_f_scale
        weights = (1+normalized**2)**(-1.5)
        if estimator == "weighted_soft_l1":
            weights /= scales**2
    else:
        raise ValueError("Unsupported explicitly declared source line estimator")
    information = design.T @ (weights[:, None]*design)
    if np.linalg.matrix_rank(information) != 2:
        raise MissingSourceEvidence("Original source readouts have singular line information")
    influence_centered = np.linalg.solve(information, design.T*weights)
    transform = np.array([[1/span, 0.], [-center/span, 1.]])
    influence = transform @ influence_centered
    covariance = (influence*scales) @ (influence*scales).T
    ids = tuple(readout_ids) if readout_ids is not None else tuple(f"{group_id}/original_readout/{i}" for i in range(len(points)))
    return SourceLineEvidence(coefficient, covariance, axis, group_id,
                              f"propagated_original_readout_scales_through_{estimator}", ids, scales, influence)


def line_evidence_from_feature(feature, *, group_id=None):
    """Read the actual accepted200 line fields; refuse missing uncertainty.

    The board-top estimator's recorded implementation uses scipy soft_l1 with
    loss scale 1 in *normalized* residual units.  This is a loss parameter,
    not an invented one-pixel measurement scale.
    """
    identity = group_id or feature.get("source_evidence_group_id") or feature.get("id")
    fitted = feature.get("source_line_fit", {})
    if "coefficient_y_equals_ax_plus_b" in fitted and "covariance_ab" in fitted:
        if (fitted.get("current_frame_only") is not True
                or fitted.get("old_model_projection_used_as_output_points") is not False):
            raise ValueError("Source covariance must belong to current-frame material readouts")
        return SourceLineEvidence(fitted["coefficient_y_equals_ax_plus_b"], fitted["covariance_ab"], 0,
                                  identity, "original_source_line_fit.covariance_ab")
    record = feature.get("source_straight_line_estimate", {})
    if record:
        if record.get("model_projection_used") is not False:
            raise ValueError("Source line must explicitly exclude projected measurement pixels")
        if record.get("estimator") != "source_uncertainty_weighted_soft_l1_line":
            raise MissingSourceEvidence("Unrecognized recorded source line estimator")
        return line_evidence_from_readouts(record["evidence_uv"], record["evidence_localization_sigma_px"],
            [record["y_per_x"], record["y_intercept_px"]], axis=0, group_id=identity,
            estimator="weighted_soft_l1", soft_l1_f_scale=1.)
    raise MissingSourceEvidence("Source line coefficients alone do not supply covariance")


def line_whitening_operator(uv, source_point_ids, line):
    """All visible representatives of this one source line enter one rank-2 FIT."""
    ids = _ids(source_point_ids)
    points = _finite(uv, (len(ids), 2), "unchanged original line UV")
    n = line.normal
    # Coefficient perturbations yield a normal displacement (a*x+b)/sqrt(1+a²).
    # Both rx and ry contribute along the actual measured source normal.
    basis = np.column_stack((points[:, line.axis], np.ones(len(points)))) * n[1-line.axis]
    sensitivity = (n[None, :, None] * basis[:, None, :]).reshape(2*len(points), 2)
    return latent_whitening_operator(ids, sensitivity, line.covariance_ab,
                                     group_id=line.group_id, covariance_origin=line.covariance_origin)


def intersection_coordinates_and_jacobian(horizontal, vertical):
    """Return the original source-line intersection and analytic d(x,y)/d(a,b,c,d)."""
    if horizontal.axis != 0 or vertical.axis != 1:
        raise ValueError("Intersection requires y=a*x+b and x=c*y+d source lines")
    a,b = horizontal.coefficients; c,d = vertical.coefficients
    implicit = np.array([[-a, 1.], [1., -c]])
    if np.linalg.matrix_rank(implicit) != 2:
        raise MissingSourceEvidence("Source lines do not identify a finite intersection")
    xy = np.linalg.solve(implicit, [b,d])
    derivative = np.array([[-xy[0], -1., 0., 0.], [0., 0., -xy[1], -1.]])
    return xy, -np.linalg.solve(implicit, derivative)


def joint_line_corner_operator(members: Sequence[Mapping], lines: Mapping[str, SourceLineEvidence], *,
                               group_id, independent_readouts: Mapping[str, float] | None = None):
    """Join line representatives, derived corners and original readout noise.

    Members retain their original ``point_id`` and ``uv``. Kinds are:
    ``line_representative`` with ``line_id``; ``derived_intersection`` with
    ``horizontal_line_id``/``vertical_line_id``; and ``direct_readout`` with
    ``readout_id``/explicit two-component ``displacement_xy_per_unit_noise``.

    With readout lineage, corners and those same edge readouts share latent
    variables, so adding the derived corner cannot duplicate edge information.
    Explicit additional source scales can be supplied in independent_readouts;
    they are measured noise, never a numerical floor. A covariance-only line
    may join representatives/corners, but cannot be asserted to share a direct
    readout until its coefficient influence has actually been supplied.
    """
    ids = _ids([member["point_id"] for member in members])
    noise = dict(independent_readouts or {})
    for line in lines.values():
        for rid, sigma in zip(line.readout_ids, line.readout_sigmas if line.readout_ids else ()):
            if rid in noise and not np.isclose(noise[rid], sigma, rtol=1e-12, atol=0):
                raise ValueError("Shared source readout has contradictory scales")
            noise[rid] = float(sigma)
    if any(not np.isfinite(s) or s <= 0 for s in noise.values()):
        raise MissingSourceEvidence("Every source readout requires an explicit positive scale")
    readout_index = {rid:i for i,rid in enumerate(noise)}
    offsets = {}; dimension = len(noise)
    for key,line in lines.items():
        if not line.readout_ids:
            offsets[key] = dimension; dimension += 2
    if not dimension:
        raise MissingSourceEvidence("Joint group has no declared source uncertainty")
    covariance = np.zeros((dimension,dimension))
    for rid,i in readout_index.items():covariance[i,i] = noise[rid]**2
    for key,offset in offsets.items():covariance[offset:offset+2,offset:offset+2] = lines[key].covariance_ab
    coefficient_maps = {}
    for key,line in lines.items():
        mapping = np.zeros((2,dimension))
        if line.readout_ids:
            mapping[:,[readout_index[rid] for rid in line.readout_ids]] = line.coefficient_influence
        else:
            mapping[:,offsets[key]:offsets[key]+2] = np.eye(2)
        coefficient_maps[key] = mapping
    sensitivity = np.zeros((2*len(members),dimension))
    for i,member in enumerate(members):
        uv = _finite(member["uv"], (2,), "unchanged original member UV")
        block = sensitivity[2*i:2*i+2]
        kind = member["kind"]
        if kind == "line_representative":
            key = member["line_id"]; line = lines[key]; n = line.normal
            derivative = n[:,None] * (np.array([uv[line.axis],1.])*n[1-line.axis])[None,:]
            block[:] = derivative @ coefficient_maps[key]
        elif kind == "derived_intersection":
            h,v = member["horizontal_line_id"],member["vertical_line_id"]
            _,derivative = intersection_coordinates_and_jacobian(lines[h],lines[v])
            block[:] = derivative @ np.vstack((coefficient_maps[h],coefficient_maps[v]))
        elif kind == "direct_readout":
            rid = member["readout_id"]
            if rid not in readout_index:
                raise MissingSourceEvidence("Direct FIT readout has no authenticated source scale")
            block[:,readout_index[rid]] = _finite(member["displacement_xy_per_unit_noise"],(2,),"source readout displacement")
        else:
            raise ValueError("Unknown explicit joint source member kind")
    return latent_whitening_operator(ids, sensitivity, covariance, group_id=group_id,
                                     covariance_origin="joint_original_readout_noise_and_declared_source_line_covariance")
