"""Joint rotation/zoom adjustment from actual typed image edges.

No pose smoothness or matrix interpolation residual is used. Elevated board
pixels are rays at unknown depth; ice pixels are intersected with Z=0. Only
components connected by measured edges to a fixed source camera can output.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .multisurface_camera import Camera
from .landmark_camera import curve_residual_h,curve_direction_residual_h


@dataclass
class ImageEdge:
    previous: int
    current: int
    previous_xy: np.ndarray
    current_xy: np.ndarray
    surface: str
    kind: str = 'adjacent_actual_image_pair'


@dataclass
class GroundLine:
    frame: int
    endpoints_xyz: np.ndarray
    observed_xy: np.ndarray
    surface: str
    physical_id: str
    sigma_px: float = 3.


def spatial_subset(points, count=16):
    """Deterministic farthest-point subset in measured image coordinates."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) <= count:
        return np.arange(len(points))
    chosen = [int(np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(count - 1):
        k = int(np.argmax(distance)); chosen.append(k)
        distance = np.minimum(distance, np.linalg.norm(points - points[k], axis=1))
        distance[chosen] = -1
    return np.asarray(chosen)


def connected_frames(count, fixed, edges):
    neighbours = [set() for _ in range(count)]
    for edge in edges:
        if len(edge.previous_xy) >= 4:
            neighbours[edge.previous].add(edge.current)
            neighbours[edge.current].add(edge.previous)
    reached, pending = set(fixed), list(fixed)
    while pending:
        current = pending.pop()
        for other in neighbours[current] - reached:
            reached.add(other); pending.append(other)
    return reached


def bundle_adjust(cameras, fixed, edges, *, ground_lines=(), max_nfev=80, points_per_edge=16,
                  landmark_curves=(), landmark_anchor_frames=(), balance_image_groups=False):
    """Adjust all connected frame cameras, fixing original reference frames.

    Camera centres/principal points/lens coefficients are conditional constants.
    Return finite limited-iteration candidates separately from convergence.
    """
    cameras = list(cameras); fixed = set(fixed)
    usable = []
    for edge in edges:
        if edge.surface not in {'ice_marking', 'board_surface'}:
            raise ValueError('Unknown geometric surface')
        if not 0 <= edge.previous < len(cameras) or not 0 <= edge.current < len(cameras):
            raise ValueError('Edge frame outside source list')
        a, b = np.asarray(edge.previous_xy, float), np.asarray(edge.current_xy, float)
        if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 2:
            raise ValueError('Image edge must have matched Nx2 endpoints')
        good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[good], b[good]
        if len(a) < 4:
            continue
        take = spatial_subset(a, points_per_edge)
        usable.append(ImageEdge(edge.previous, edge.current, a[take], b[take], edge.surface, edge.kind))
    landmark_roots = set(landmark_anchor_frames)
    if not landmark_roots.issubset({c.frame for c in landmark_curves}):
        raise ValueError('A landmark root needs actual identified contour observations')
    reached = connected_frames(len(cameras), fixed | landmark_roots, usable)
    usable = [e for e in usable if e.previous in reached and e.current in reached]
    for i in reached:
        if cameras[i] is None:
            raise ValueError('Connected frame needs an explicit numerical camera initializer')
        if cameras[i].radial_ray_k1 != 0:
            raise ValueError('This joint solver explicitly supports zero radial distortion only')
    for e in usable:
        if e.surface == 'board_surface' and not np.allclose(cameras[e.previous].centre, cameras[e.current].centre, atol=1e-7, rtol=0):
            raise ValueError('Elevated board rays require a common conditional optical centre')
    variable = sorted(reached - fixed)
    if (not usable and not landmark_curves) or not variable:
        return {i: cameras[i] for i in reached}, dict(solver_converged=True, reason='fixed_references_only', connected_frame_count=len(reached), edges=[])
    vlookup = {i: j for j, i in enumerate(variable)}
    # Placeholder constants for disconnected frames are never evaluated/output.
    fallback = cameras[min(reached)]
    all_cameras = [c if c is not None else fallback for c in cameras]
    base_R = np.asarray([c.rotation for c in all_cameras]); centres = np.asarray([c.centre for c in all_cameras])
    principal = np.asarray([c.principal for c in all_cameras]); focal0 = np.asarray([c.focal for c in all_cameras])
    radial = np.asarray([c.radial_ray_k1 for c in all_cameras])
    left = np.concatenate([np.full(len(e.previous_xy), e.previous, int) for e in usable]) if usable else np.empty(0,int)
    right = np.concatenate([np.full(len(e.current_xy), e.current, int) for e in usable]) if usable else np.empty(0,int)
    old = np.concatenate([e.previous_xy for e in usable]) if usable else np.empty((0,2))
    observed = np.concatenate([e.current_xy for e in usable]) if usable else np.empty((0,2))
    ice = np.concatenate([np.full(len(e.previous_xy), e.surface == 'ice_marking', bool) for e in usable]) if usable else np.empty(0,bool)
    image_weights = np.concatenate([np.full(len(e.previous_xy), np.sqrt(4/len(e.previous_xy)) if balance_image_groups else 1.) for e in usable]) if usable else np.empty(0)
    lengths = [len(e.previous_xy) for e in usable]; offsets = np.r_[0, np.cumsum(lengths)]
    curves = []
    for line in ground_lines:
        if line.frame not in reached:
            continue
        world = np.asarray(line.endpoints_xyz, float); points = np.asarray(line.observed_xy, float)
        if line.surface not in {'ice_marking', 'ice_board_contact'} or world.shape != (2, 3) or not np.allclose(world[:, 2], 0):
            raise ValueError('Only bounded Z=0 lines can constrain ground curves')
        if len(points):
            take = spatial_subset(points, 8)
            curves.append(GroundLine(line.frame, world, points[take], line.surface, line.physical_id, line.sigma_px))
    curve_count = sum(len(c.observed_xy) for c in curves)
    curve_frames = np.concatenate([np.full(len(c.observed_xy), c.frame, int) for c in curves]) if curves else np.empty(0, int)
    curve_world = np.concatenate([np.repeat(c.endpoints_xyz[None], len(c.observed_xy), axis=0) for c in curves]) if curves else np.empty((0, 2, 3))
    curve_observed = np.concatenate([c.observed_xy for c in curves]) if curves else np.empty((0, 2))
    curve_sigma = np.concatenate([np.full(len(c.observed_xy), max(c.sigma_px, 1.)) for c in curves]) if curves else np.empty(0)
    named = [c for c in landmark_curves if c.frame in reached]
    for c in named:
        if len(c.observed_xy)<3 or not np.isfinite(c.observed_xy).all() or c.sigma_px<=0 or c.group_weight<=0:
            raise ValueError('Invalid named measured contour')
    named_count = sum(len(c.observed_xy)*(2 if c.tangents_xy is not None else 1) for c in named)
    # Two pixel errors plus two cheirality penalties per measured point.
    point_residual_count = len(old) * 4
    sparse = lil_matrix((point_residual_count + curve_count + named_count, len(variable) * 4), dtype=np.int8)
    for edge, start, stop in zip(usable, offsets[:-1], offsets[1:]):
        for i in (edge.previous, edge.current):
            if i in vlookup:
                j = vlookup[i] * 4
                sparse[start * 2:stop * 2, j:j + 4] = 1
                sparse[len(old) * 2 + start * 2:len(old) * 2 + stop * 2, j:j + 4] = 1
    cursor = point_residual_count
    for curve in curves:
        if curve.frame in vlookup:
            j = vlookup[curve.frame] * 4
            sparse[cursor:cursor + len(curve.observed_xy), j:j + 4] = 1
        cursor += len(curve.observed_xy)
    for curve in named:
        size=len(curve.observed_xy)*(2 if curve.tangents_xy is not None else 1)
        if curve.frame in vlookup:
            j = vlookup[curve.frame] * 4
            sparse[cursor:cursor + size, j:j+4] = 1
        cursor += size
    sparse = sparse.tocsr()

    def unpack(x):
        x = x.reshape(-1, 4); rotations = base_R.copy(); focal = focal0.copy()
        rotations[variable] = Rotation.from_rotvec(x[:, :3]).as_matrix() @ base_R[variable]
        focal[variable] = focal0[variable] * np.exp(x[:, 3])
        return rotations, focal

    def line_errors(rotations, focal):
        if not curves:
            return np.empty(0)
        qline = np.einsum('nij,nkj->nki', rotations[curve_frames], curve_world - centres[curve_frames, None])
        both_hidden = np.max(qline[:, :, 2], axis=1) <= 1e-6
        for side in (0, 1):
            other = 1 - side
            clip = (qline[:, side, 2] <= 1e-6) & (qline[:, other, 2] > 1e-6)
            fraction = (1e-6 - qline[clip, side, 2]) / (qline[clip, other, 2] - qline[clip, side, 2])
            qline[clip, side] += fraction[:, None] * (qline[clip, other] - qline[clip, side])
        z = np.maximum(qline[:, :, 2], 1e-6)
        ends = qline[:, :, :2] / z[:, :, None] * focal[curve_frames, None, None] + principal[curve_frames, None]
        direction = ends[:, 1] - ends[:, 0]
        alpha = np.clip(np.sum((curve_observed - ends[:, 0]) * direction, axis=1) / np.maximum(np.sum(direction ** 2, axis=1), 1e-9), 0, 1)
        errors = np.linalg.norm(curve_observed - ends[:, 0] - alpha[:, None] * direction, axis=1)
        errors[both_hidden] = 640. + np.maximum(-qline[both_hidden, :, 2].mean(axis=1), 0)
        return errors

    def named_errors(rotations, focal, weighted=True):
        values=[]
        for c in named:
            i=c.frame
            K=np.array([[focal[i],0,principal[i,0]],[0,focal[i],principal[i,1]],[0,0,1.]])
            h=K @ np.column_stack((rotations[i,:,:2], -rotations[i] @ centres[i]))
            error=curve_residual_h(h,c.kind,c.equation_model,c.observed_xy)
            if weighted:
                error=error/c.sigma_px*np.sqrt(16*c.group_weight/len(c.observed_xy))
            values.append(error)
            if weighted and c.tangents_xy is not None:
                values.append(curve_direction_residual_h(h,c)/c.tangent_sigma_rad*np.sqrt(16*c.group_weight/len(c.observed_xy)))
        return np.concatenate(values) if values else np.empty(0)

    def residual(x, raw=False):
        rotations, focal = unpack(x)
        normalized = (old - principal[left]) / focal[left, None]
        # Invert the original camera's radial ray equation, not a homography.
        distorted_radius = np.linalg.norm(normalized, axis=1); radius = distorted_radius.copy()
        k = radial[left]
        for _ in range(6):
            radius -= (radius * (1 + k * radius ** 2) - distorted_radius) / np.maximum(1 + 3 * k * radius ** 2, .05)
        normalized *= np.divide(radius, distorted_radius, out=np.ones_like(radius), where=distorted_radius > 1e-12)[:, None]
        local = np.column_stack((normalized, np.ones(len(old))))
        world_ray = np.einsum('ni,nij->nj', local, rotations[left])
        vector = world_ray.copy(); valid = np.isfinite(world_ray).all(axis=1)
        # Board vectors remain unknown-depth rays. Only ice gets Z=0 depth.
        depth = -centres[left, 2] / np.where(np.abs(world_ray[:, 2]) > 1e-9, world_ray[:, 2], -1e-9)
        vector[ice] = (centres[left] + depth[:, None] * world_ray - centres[right])[ice]
        valid &= (~ice) | (depth > 0)
        q = np.einsum('nij,nj->ni', rotations[right], vector)
        valid &= q[:, 2] > 1e-7
        # The signed guarded denominator keeps a gradient near invalid starts.
        z = np.where(np.abs(q[:, 2]) > 1e-7, q[:, 2], 1e-7)
        prediction = q[:, :2] / z[:, None]
        r2 = np.sum(prediction ** 2, axis=1)
        prediction *= (1 + radial[right] * r2)[:, None]
        prediction = prediction * focal[right, None] + principal[right]
        delta = np.nan_to_num(prediction - observed, nan=1e4, posinf=1e4, neginf=-1e4)
        if raw:
            return delta, valid
        # Physical branch penalties cannot be evaded by a signed homography
        # yielding the same 2-D point from behind a camera or above the ice.
        ground_penalty = np.maximum(world_ray[:, 2] + 1e-5, 0) * ice * 640
        front_penalty = np.maximum(1e-5 - q[:, 2] / np.maximum(np.linalg.norm(q, axis=1), 1e-9), 0) * 640
        values = [(delta / 1.5 * image_weights[:,None]).ravel(), np.column_stack((ground_penalty, front_penalty)).ravel()]
        if curves:
            values.append(line_errors(rotations, focal) / curve_sigma)
        if named:
            values.append(named_errors(rotations,focal))
        return np.concatenate(values)

    zero = np.zeros(len(variable) * 4)
    initial_delta, initial_valid = residual(zero, True)
    # A broad robust scale initially permits a displaced partition to move.
    # This is an image residual objective, not a smoothness/age rule.
    fit = least_squares(residual, zero, jac_sparsity=sparse, method='trf', tr_solver='lsmr',
                        loss='soft_l1', f_scale=6., max_nfev=max_nfev,
                        bounds=(-np.tile([1.5, 1.5, 1.5, 1.2], len(variable)), np.tile([1.5, 1.5, 1.5, 1.2], len(variable))),
                        ftol=1e-6, xtol=1e-6, gtol=1e-5, x_scale='jac')
    rotations, focals = unpack(fit.x); delta, valid = residual(fit.x, True)
    curve_errors = line_errors(rotations, focals)
    named_before=named_errors(base_R,focal0,False);named_after=named_errors(rotations,focals,False)
    named_report=[];offset=0
    for c in named:
        size=len(c.observed_xy);before=named_before[offset:offset+size];after=named_after[offset:offset+size];offset+=size
        named_report.append(dict(frame=c.frame,physical_id=c.physical_id,observation_id=c.observation_id,
            used_by_solver=c.frame in vlookup, fixed_reference_check_only=c.frame in fixed,
            sigma_sample_px=c.sigma_px,group_weight=c.group_weight,measured_sample_count=size,
            initial_median_px=float(np.median(np.abs(before))),median_px=float(np.median(np.abs(after))),
            p95_px=float(np.percentile(np.abs(after),95))))
    output = {i: Camera(rotations[i], all_cameras[i].centre.copy(), float(focals[i]), all_cameras[i].principal.copy(), all_cameras[i].size, all_cameras[i].radial_ray_k1) for i in reached}
    diagnostics = []
    for edge, start, stop in zip(usable, offsets[:-1], offsets[1:]):
        errors = np.linalg.norm(delta[start:stop], axis=1); before = np.linalg.norm(initial_delta[start:stop], axis=1)
        diagnostics.append(dict(previous_index=edge.previous, current_index=edge.current, surface=edge.surface,
                                kind=edge.kind, point_count=int(stop - start), visible_count=int(valid[start:stop].sum()),
                                initial_median_px=float(np.median(before)), median_px=float(np.median(errors)), p95_px=float(np.percentile(errors, 95))))
    return output, dict(solver_converged=bool(fit.success), solver_status=int(fit.status), solver_message=str(fit.message),
                        nfev=int(fit.nfev), initial_squared_error=float(np.sum(initial_delta ** 2)),
                        final_squared_error=float(np.sum(delta ** 2)), final_robust_cost=float(fit.cost),
                        connected_frame_count=len(reached), fixed_reference_indices=sorted(fixed),
                        missing_disconnected_indices=sorted(set(range(len(cameras))) - reached),
                        bound_parameter_count=int(np.count_nonzero(fit.active_mask)), finite_candidate=bool(np.isfinite(fit.x).all()),
                        invalid_point_projection_count=int(np.count_nonzero(~valid)), zero_radial_approximation_explicit=True,
                        ground_line_point_count=len(curve_errors), ground_line_median_sample_px=float(np.median(curve_errors)) if len(curve_errors) else None,
                        used_ground_lines=[dict(frame=c.frame, surface=c.surface, physical_id=c.physical_id,
                            observed_sample_xy=c.observed_xy.tolist(), endpoints_model_xyz=c.endpoints_xyz.tolist(), sigma_sample_px=c.sigma_px) for c in curves],
                        board_points_used_as_Z0=False, pose_smoothness_prior_used=False,
                        residuals_from_actual_images_only=True, independent_accuracy_validated=False, edges=diagnostics,
                        used_landmark_curves=named_report,landmark_anchor_frames=sorted(landmark_roots),
                        image_groups_balanced=balance_image_groups,
                        landmark_group_normalization='16_equivalent_samples_per_physical_group',
                        optimality=float(fit.optimality), parameter_step_norm=float(np.linalg.norm(fit.x)),
                        parameter_scaling='jacobian', robust_loss='soft_l1', robust_scale=6.)
