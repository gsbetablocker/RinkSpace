"""Small, explicit factors for a source-supported hand/skate relationship.

These functions do not infer contact, visibility, foot identity, or thresholds.
They consume fixed source hypotheses and the ONE current articulated state.
All lengths use one caller-declared metric world. No numerical default is an
accuracy gate. A wrist joint MUST NOT be silently used as a palm surface point.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _finite(a, shape, name):
    a = np.asarray(a, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite shape {shape}")
    return a


def _positive(x, name, *, zero=False):
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
        raise ValueError(name + " must be a finite scalar")
    x = float(x)
    if not np.isfinite(x) or (x < 0 if zero else x <= 0):
        raise ValueError(name + " has an invalid value")
    return x


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _robust_vector(e, delta):
    """Residual and dr/de whose half squared norm is radial pseudo-Huber."""
    d = _positive(delta, "robust_delta")
    u = np.sqrt(1. + float(e @ e) / d**2)
    k = np.sqrt(2. / (u + 1.))
    dk = -k / (4. * d**2 * u * (u + 1.))
    return k * e, k * np.eye(len(e)) + 2. * dk * np.outer(e, e)


@dataclass(frozen=True)
class GripInterval:
    """An explicit event hypothesis, separate from per-frame visual visibility.

    Boundary/source decisions are made outside the inner optimizer. `held`
    means the caller admitted a sustained relation based on source evidence;
    it does not mean every included frame was individually observed.
    """
    source_id: str
    shot_id: str
    start_pts: int
    end_pts: int
    time_base: tuple[int, int]
    hand: str
    foot: int
    state: str
    role: str
    evidence_pts: tuple[int, ...]
    provenance: str

    def __post_init__(self):
        if not self.source_id or not self.shot_id or not self.provenance:
            raise ValueError("Source identity and provenance are required")
        ints = (self.start_pts, self.end_pts, *self.time_base, *self.evidence_pts)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in ints):
            raise ValueError("Use original integer PTS and rational time base")
        if len(self.time_base) != 2 or min(self.time_base) <= 0 or self.end_pts < self.start_pts:
            raise ValueError("Invalid interval/time base")
        if (self.hand not in ("left", "right") or isinstance(self.foot, (bool, np.bool_))
                or not isinstance(self.foot, (int, np.integer)) or self.foot not in (0, 1)):
            raise ValueError("Anatomical hand and foot identity must be explicit")
        if self.state not in ("held", "released", "unknown") or self.role not in ("fit", "review"):
            raise ValueError("Invalid relation state/role")
        if self.state == "held" and not self.evidence_pts:
            raise ValueError("A held hypothesis needs actual source evidence")
        if any(not self.start_pts <= t <= self.end_pts for t in self.evidence_pts):
            raise ValueError("Evidence PTS outside this interval")

    def admitted_pts(self, pts, *, source_id, shot_id, time_base):
        """Call before optimization; never use current geometry to enable rows."""
        p = np.asarray(pts)
        if p.ndim != 1 or p.dtype.kind not in "iu" or np.any(np.diff(p.astype(np.int64)) <= 0):
            raise ValueError("Expected unique increasing integer source PTS")
        if (source_id, shot_id, tuple(time_base)) != (self.source_id, self.shot_id, self.time_base):
            raise ValueError("Source/shot/time base mismatch")
        if self.role != "fit" or self.state != "held":
            return np.zeros(len(p), dtype=bool)
        return (p >= self.start_pts) & (p <= self.end_pts)


def observation_visibility_policy(state, *, source_region_present):
    """A detector score is not a visibility label.

    The relation prior is deliberately NOT switched off here. A hidden hand
    can remain in an admitted grasp interval even when its 2D point is absent.
    Decisions must be based on source evidence outside an inner solve.
    """
    allowed = {"visible", "partially_visible", "self_occluded", "externally_occluded", "out_of_frame", "uncertain"}
    if state not in allowed:
        raise ValueError("Unknown source visibility state")
    active = state in {"visible", "partially_visible"} and bool(source_region_present)
    return dict(image_factor_enabled=active,
                image_semantics="source_region" if active else "no_direct_image_measurement",
                relation_state_unchanged=True,
                observed_3d_truth=False)


def palm_anchor(vertices_world, vertex_ids, weights, *, provenance):
    """A fixed semantic palm patch on the current skinned hand mesh.

    The caller must supply verified model vertex IDs and nonnegative weights,
    after propagating the hand pose to FK/LBS. This helper invents no IDs,
    bone lengths, wrist offsets or hand pose. It returns the linear gather
    weights needed for dH/dstate. A palm centre has finite thickness; its
    contact tube must reflect that geometry rather than assert zero skin gap.
    """
    V = np.asarray(vertices_world, dtype=float)
    ids = np.asarray(vertex_ids)
    w = np.asarray(weights, dtype=float)
    if V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all():
        raise ValueError("Expected finite (V,3) current hand vertices")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(ids < 0) or np.any(ids >= len(V)):
        raise ValueError("Invalid palm vertex IDs")
    if len(np.unique(ids)) != len(ids) or w.shape != ids.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid palm patch weights")
    if not provenance:
        raise ValueError("Declare how this palm patch was selected")
    w = w / w.sum()
    return dict(point_world=w @ V[ids], vertex_ids=ids.copy(), normalized_weights=w,
                provenance=provenance, feature="semantic_palm_patch")


def grip_to_segment_factor(palm_world, shoe_rotation, shoe_origin,
                           segment_start_local, segment_end_local, *,
                           contact_radius_m, residual_scale_m,
                           objective_weight, quadrature_seconds, robust_delta):
    """Soft distance from a real palm patch to a source-selected grip segment.

    The segment belongs to a declared GRASPABLE shoe/blade patch. It is not
    the ice's lowest point, and must not be selected from the whole body by
    nearest distance. Longitudinal grip location may move within this patch.
    The closest-point derivative is included. All parameters/row identities
    stay fixed inside a solve; hidden images do not erase a held relation.
    This positional factor does not by itself constrain palm orientation,
    finger wrapping, or surface interpenetration. Those require separately
    supported geometry; a correct position alone is not a certified grasp.

    Jacobians use R_new = R @ Exp(eta), O and H independent. The caller chains
    BOTH palm and shoe transforms to the SAME body controls. World rigid
    motion cancels; this factor cannot locate the person on the ice by itself.
    """
    H = _finite(palm_world, (3,), "palm_world")
    O = _finite(shoe_origin, (3,), "shoe_origin")
    R = _finite(shoe_rotation, (3, 3), "shoe_rotation")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0) or np.linalg.det(R) <= 0:
        raise ValueError("shoe_rotation must be SO(3)")
    a = _finite(segment_start_local, (3,), "segment_start_local")
    b = _finite(segment_end_local, (3,), "segment_end_local")
    radius = _positive(contact_radius_m, "contact_radius_m", zero=True)
    scale = _positive(residual_scale_m, "residual_scale_m")
    weight = _positive(objective_weight, "objective_weight")
    dt = _positive(quadrature_seconds, "quadrature_seconds")
    v = b - a
    vv = float(v @ v)
    if vv <= 0:
        raise ValueError("Grip segment must have positive length")
    h = R.T @ (H - O)
    raw_u = float((h-a) @ v / vv)
    u = float(np.clip(raw_u, 0., 1.))
    closest = a + u * v
    e = h - closest
    D = np.eye(3) - (np.outer(v, v) / vv if 0. < raw_u < 1. else 0.)
    distance = float(np.linalg.norm(e))
    if distance <= radius:
        outside, Joutside = np.zeros(3), np.zeros((3, 3))
    else:
        outside = (1. - radius / distance) * e
        Joutside = ((1. - radius/distance) * np.eye(3)
                    + radius * np.outer(e, e) / distance**3) @ D
    r, Dr = _robust_vector(outside / scale, robust_delta)
    factor = np.sqrt(weight * dt)
    Jlocal = factor * Dr @ Joutside / scale
    return dict(residual=factor*r,
                jacobian_palm_world=Jlocal @ R.T,
                jacobian_shoe_origin=-Jlocal @ R.T,
                jacobian_shoe_right_rotation=Jlocal @ _skew(h),
                closest_grip_point_world=O + R @ closest,
                closest_segment_coordinate=u,
                palm_to_segment_m=distance,
                excess_outside_contact_tube_m=max(distance-radius, 0.),
                cost=float((factor*r) @ (factor*r) / 2.),
                endpoint_transition=raw_u in (0., 1.),
                measured_accuracy=False)


def body_centric_points(points_world, root_world, root_rotation):
    """Remove global root translation/rotation before a LOCAL pose prior."""
    X = np.asarray(points_world, dtype=float)
    P = np.asarray(root_world, dtype=float)
    Q = np.asarray(root_rotation, dtype=float)
    if X.ndim != 3 or X.shape[-1] != 3 or P.shape != (len(X),3) or Q.shape != (len(X),3,3):
        raise ValueError("Expected points(N,K,3), roots(N,3), rotations(N,3,3)")
    if not all(np.isfinite(v).all() for v in (X,P,Q)):
        raise ValueError("Nonfinite body trajectory")
    if not np.allclose(Q.transpose(0,2,1) @ Q, np.eye(3), atol=1e-7, rtol=0) or np.any(np.linalg.det(Q) <= 0):
        raise ValueError("Root rotations must be SO(3)")
    return np.einsum('nji,nkj->nki', Q, X-P[:,None])


def actual_local_acceleration_factor(local_points, times_seconds, *,
                                     scale_m_s2, objective_weight,
                                     fixed_point_weights, robust_delta):
    """A robust prior on actual body-centric FK, not on spline corrections.

    Nonuniform source times are supported. A constant pose rotating rapidly
    in the world has zero local acceleration; changing poses are allowed by
    the finite robust cost. Do not apply one constant-pose template through
    entry, grasp, extension, release, or a change of supporting foot.

    Returns D2 and per-vector dr/d(acceleration) for sparse chain assembly.
    All source visibility/phase weights are frozen before the inner solve.
    """
    X = np.asarray(local_points, dtype=float)
    t = np.asarray(times_seconds, dtype=float)
    if X.ndim != 3 or X.shape[-1] != 3 or len(X) < 3 or t.shape != (len(X),):
        raise ValueError("Expected N>=3 actual local points and times")
    if not np.isfinite(X).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be finite and strictly increasing")
    scale = _positive(scale_m_s2, "scale_m_s2")
    weight = _positive(objective_weight, "objective_weight")
    pw = _finite(fixed_point_weights, (X.shape[1],), "fixed_point_weights")
    if np.any(pw < 0):raise ValueError("Negative point weight")
    h0, h1 = np.diff(t)[:-1], np.diff(t)[1:]
    D2 = np.zeros((len(t)-2,len(t)))
    k = np.arange(len(t)-2)
    D2[k,k] = 2./(h0*(h0+h1))
    D2[k,k+1] = -2./(h0*h1)
    D2[k,k+2] = 2./(h1*(h0+h1))
    acc = np.einsum('tn,nkc->tkc', D2, X)
    measure = (h0+h1)/2.
    r = np.empty_like(acc);J = np.empty((*acc.shape[:2],3,3))
    for i in range(len(acc)):
        for j in range(X.shape[1]):
            rr,DD = _robust_vector(acc[i,j]/scale,robust_delta)
            factor=np.sqrt(weight*measure[i]*pw[j])
            r[i,j]=factor*rr;J[i,j]=factor*DD/scale
    return dict(residual=r.ravel(),jacobian_per_acceleration=J,
                second_difference_operator=D2,actual_local_acceleration=acc,
                time_measure_seconds=measure,cost=float(r.ravel()@r.ravel()/2.),
                is_motion_prior_not_observation=True)
