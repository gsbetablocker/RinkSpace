"""Finite source support-location factors for one continuous body/shoe state.

Camera-only ice-interface footprints are conditional on the frozen camera,
Z=0 and scene scale. They share source evidence with complete 2-D shoe/blade
curves: part of that frame budget is allocated here, not independent 3-D data.
Ambiguous interface boundaries keep their full supplied uncertainty range.
Only explicit current continuous-source support-identity admission enables rows.

The prediction is the exact lowest point on the actual finite steel runner
arcs. This moving geometry witness is neither a fixed material point nor a
source midpoint. Its motion is differentiated along the existing body and
ankle-pivoted relative-shoe curves. Ice Z softness remains in the existing ice
policy: this factor has no Z residual or new feasibility threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.spatial import ConvexHull

from .fullbody_fit import right_jacobian, skew
from .skate_physics import fk_state


SCHEMA = 'rinkspace.source-support-locations.v1'


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must remain an integer')
    return int(value)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


def _runner_witness(binding, rotation, origin):
    """Exact finite-arc minima and d(local witness)/d(local world-up normal).

    The chosen side/arc is piecewise smooth; ties and endpoint/side changes are
    reported. At a switch the selected one-sided derivative is used. Neither
    shoe shape nor attachment changes to make this witness.
    """
    normal = rotation[:, 2, :]
    arcs = [a for a in binding.packet()['template']['contact_arcs']
            if a['semantic'] == 'runner']
    if not arcs:
        raise ValueError('The actual template has no finite steel runner arcs')
    points, derivatives, angles, interiors = [], [], [], []
    for arc in arcs:
        a, b = np.asarray(arc['axis_cos']), np.asarray(arc['axis_sin'])
        A, B = normal @ a, normal @ b
        lo, hi = arc['theta_start_rad'], arc['theta_end_rad']
        theta = np.arctan2(-B, -A)
        theta += 2*np.pi*np.round(((lo+hi)/2-theta)/(2*np.pi))
        candidates = np.stack((np.full_like(A, lo), np.full_like(A, hi),
                               np.clip(theta, lo, hi)), axis=-1)
        height = A[:, None]*np.cos(candidates)+B[:, None]*np.sin(candidates)
        which = np.argmin(height, axis=-1)
        selected = candidates[np.arange(len(normal)), which]
        interior = (which == 2) & (theta > lo) & (theta < hi)
        q = (np.asarray(arc['center']) + arc['radius_m'] *
             (np.cos(selected)[:, None]*a+np.sin(selected)[:, None]*b))
        denominator = A*A+B*B
        dtheta = np.zeros_like(normal)
        good = interior & (denominator > np.finfo(float).tiny)
        dtheta[good] = (A[good, None]*b-B[good, None]*a)/denominator[good, None]
        tangent = arc['radius_m']*(-np.sin(selected)[:, None]*a+np.cos(selected)[:, None]*b)
        points.append(q)
        derivatives.append(tangent[:, :, None]*dtheta[:, None, :])
        angles.append(selected)
        interiors.append(interior)
    q = np.stack(points, axis=1)
    height = origin[:, 2, None]+np.einsum('ni,nai->na', normal, q)
    choice = np.argmin(height, axis=1)
    each = np.arange(len(normal))
    point = q[each, choice]
    world = origin+np.einsum('nij,nj->ni', rotation, point)
    ordered = np.sort(height, axis=1)
    return dict(world=world, local=point,
                derivative_normal=np.stack(derivatives, axis=1)[each, choice],
                feature_index=choice, feature_ids=[a['id'] for a in arcs],
                theta=np.stack(angles, axis=1)[each, choice],
                interior=np.stack(interiors, axis=1)[each, choice],
                runner_heights_solver=height,
                next_feature_height_gap_solver=ordered[:, 1]-ordered[:, 0]
                    if len(arcs) > 1 else np.full(len(normal), np.nan))


def _polygon_terms(point, polygon):
    """Vector distance to a fixed closed convex source range and its Jacobian.

    Squared distance is C1 at the boundary; edge/vertex changes have explicit
    piecewise residual derivatives. This finite range is not a feasibility gate.
    """
    edge = np.roll(polygon, -1, axis=0)-polygon
    offset = point-polygon
    cross = edge[:, 0]*offset[:, 1]-edge[:, 1]*offset[:, 0]
    if np.all(cross >= 0):
        return np.zeros(2), np.zeros((2, 2)), point.copy(), 'inside'
    length2 = np.sum(edge*edge, axis=1)
    fraction = np.clip(np.sum(offset*edge, axis=1)/length2, 0., 1.)
    closest = polygon+fraction[:, None]*edge
    k = int(np.argmin(np.sum((point-closest)**2, axis=1)))
    derivative = np.eye(2)
    if 0 < fraction[k] < 1:
        derivative -= np.outer(edge[k], edge[k])/length2[k]
    return point-closest[k], derivative, closest[k], 'edge' if 0 < fraction[k] < 1 else 'vertex'


class SourceSupportLocationFactors:
    """Finite XY factors from current explicit continuous-source admission.

    support_evidence rows have integer pts/foot, current role='fit' and
    optimization_enabled=True, nonempty support_identity_evidence and phase,
    and context_source_pts spanning this native source frame. Such current
    records may explicitly admit historical development/review frames; old
    flags remain in diagnostics and the original document. Missing admission
    creates no inferred support labels.

    evaluate accepts the body/shoe vector used by SequenceActions, or the full
    frozen-shared problem vector. Jacobians always have problem.size columns.
    Shared shoe shape/attachment must remain frozen.
    """
    def __init__(self, problem, document, *, support_evidence,
                 frame_weight_budget=.02, location_budget_fraction=.25):
        self.problem, self.document = problem, document
        self.body, self.ref, self.binding = problem.body, problem.reference, problem.binding
        if getattr(problem, 'optimize_shared', False):
            raise ValueError('Source support locations require frozen shape/attachment')
        if problem.size != self.body.size+17:
            raise ValueError('Expected 17 shared columns plus one continuous state')
        if not hasattr(self.body.state, 'shoe_chain'):
            raise ValueError('Source support locations require the relative-skate continuous state')
        if document.get('schema') != 'rinkspace.source-skate-plane.v1' or document.get('time_base') != [1, 12800]:
            raise ValueError('Expected camera-only source plane document and native clock')
        if document.get('diagnostic_footprint_only', False) or not document.get('no_body_pose_used', False):
            raise ValueError('Only camera-only ice_interface source mapping is eligible')
        self.frame_weight_budget = _positive(frame_weight_budget, 'frame_weight_budget')
        self.location_budget_fraction = float(location_budget_fraction)
        if not np.isfinite(self.location_budget_fraction) or not 0 < self.location_budget_fraction < 1:
            raise ValueError('A finite strict fraction must remain for complete 2-D curves')
        self.weight = self.frame_weight_budget*self.location_budget_fraction
        self.curve_frame_weight = self.frame_weight_budget-self.weight
        info = document['camera_binding']
        source_scale = _positive(document['metres_per_scene_unit'], 'source export scale')
        solver_scale = _positive(problem.camera.scale, 'actual camera solver scale')
        ratio = _positive(self.binding.ratio, 'fitted length export ratio')
        values = [source_scale, solver_scale, ratio]
        expected = [info['metres_per_scene_unit'], info['solver_metres_per_scene_unit'], info['length_export_ratio']]
        if not np.allclose(values, expected, rtol=1e-12, atol=0) or not np.isclose(source_scale/solver_scale, ratio, rtol=1e-12, atol=0):
            raise ValueError('Camera mapping and fitted solver/export length units differ')
        self.ratio = ratio
        self.units = dict(metres_per_scene_unit=source_scale,
                          solver_metres_per_scene_unit=solver_scale, length_export_ratio=ratio,
                          model_conversion='(O + R*q) * length_export_ratio exactly once')
        admission = {}
        evidence = support_evidence.get('rows', support_evidence.get('supports', [])) if isinstance(support_evidence, dict) else support_evidence
        for record in evidence:
            key = (_integer(record['pts'], 'Support PTS'), _integer(record['foot'], 'Support foot'))
            if key in admission or key[1] not in (0, 1):
                raise ValueError('Duplicate or invalid current support admission')
            admission[key] = record
        lookup = {_integer(p, 'Reference PTS'): i for i, p in enumerate(self.ref.p['pts'])}
        if len(lookup) != len(self.ref.p['pts']):
            raise ValueError('Reference PTS must be unique')
        self.rows, self.skipped = [], []
        seen = set()
        for frame in document['frames']:
            key = (_integer(frame['pts'], 'Source PTS'), _integer(frame['foot'], 'Source foot'))
            if key in seen:
                raise ValueError('Duplicate source PTS/foot interface')
            seen.add(key)
            feature = frame.get('features', {}).get('ice_interface')
            record = admission.get(key)
            if record is None or record.get('role') != 'fit' or record.get('optimization_enabled') is not True:
                self.skipped.append(dict(pts=key[0], foot=key[1], reason='no_current_fit_support_admission'))
                continue
            if not isinstance(record.get('support_identity_evidence'), str) or not record['support_identity_evidence'].strip() or not str(record.get('phase', '')).strip():
                raise ValueError('Support admission needs continuous-source identity evidence and phase')
            context = np.asarray([_integer(p, 'Context PTS') for p in record.get('context_source_pts', [])], dtype=np.int64)
            if len(context) < 3 or key[0] not in context or not (context.min() < key[0] < context.max()):
                raise ValueError('Continuous-source support identity needs surrounding native PTS')
            if key[0] not in lookup or key[1] not in (0, 1):
                raise ValueError('Admitted support is outside the continuous reference or physical feet')
            if not feature or feature.get('mapping_role') != 'source_ice_interface_candidate' or feature.get('source_status') not in ('observed', 'ambiguous'):
                raise ValueError('Admitted support needs observed/ambiguous ice_interface mapping')
            mapping = feature['mapping']
            uv, xyz = np.asarray(mapping['points_uv'], float), np.asarray(mapping['xyz_conditional_m'], float)
            radius = np.asarray(mapping['pixel_uncertainty_radius_px'], float)
            jac = np.asarray(mapping['xy_jacobian_conditional_m_per_px'], float)
            boundary = np.asarray(mapping['sampled_pixel_disk_boundary_xy_conditional_m'], float)
            valid, bvalid = np.asarray(mapping['valid'], bool), np.asarray(mapping['sampled_pixel_disk_boundary_valid'], bool)
            n = len(uv)
            if (not n or uv.shape != (n, 2) or xyz.shape != (n, 3) or jac.shape != (n, 2, 2)
                    or radius.shape != (n,) or boundary.ndim != 3 or boundary.shape[0] != n or boundary.shape[2] != 2
                    or valid.shape != (n,) or bvalid.shape != boundary.shape[:2]
                    or not valid.all() or not bvalid.all() or not np.isfinite(np.r_[uv.ravel(), xyz.ravel(), jac.ravel(), boundary.ravel(), radius]).all()
                    or np.any(radius <= 0) or np.any(xyz[:, 2] != 0)):
                raise ValueError('Complete finite native XY mapping and pixel ranges are required')
            if not np.array_equal(uv, np.asarray(feature['source_annotation']['points_uv'], float)):
                raise ValueError('Mapped native UV must equal the untouched source annotation')
            # Fixed engineering coordinates, not a noise covariance. The full
            # nonlinear sampled source envelope defines the finite range.
            centre = xyz[:, :2].mean(axis=0)
            transform = np.linalg.inv(jac.mean(axis=0))/np.median(radius)
            vertices = np.r_[xyz[:, :2], boundary.reshape(-1, 2)]
            equivalent = (vertices-centre) @ transform.T
            hull = ConvexHull(equivalent)
            polygon = equivalent[hull.vertices]
            fr = lookup[key[0]]
            if 'frames' in mapping and not np.all(np.asarray(mapping['frames']) == fr):
                raise ValueError('Source mapping frame differs from the continuous reference')
            row_times = np.asarray(mapping['camera_row_time_seconds'], float)
            if row_times.shape != (n,) or not np.isfinite(row_times).all():
                raise ValueError('Mapped native camera row times are required')
            if n == 1:
                measures = np.ones(1)
            else:
                segments = np.linalg.norm(np.diff(uv, axis=0), axis=-1)
                measures = np.r_[segments, 0.]+np.r_[0., segments]
                if measures.sum() <= 0:
                    raise ValueError('Native interface polyline must have finite source extent')
            row_time = float(np.average(row_times, weights=measures))
            # Reuse stored native row-camera times; existing camera midpoints
            # translate them to body time. No second camera formula is used.
            offset = row_time-float(problem.camera.p['motion_times'][fr])
            self.rows.append(dict(pts=key[0], foot=key[1], frame=fr, feature=feature,
                admission=record, centre=centre, transform=transform, polygon=polygon,
                source_hull_xy_conditional_m=vertices[hull.vertices],
                time=float(self.ref.times[fr])+offset, row_time=row_time,
                row_time_span=[float(row_times.min()), float(row_times.max())]))
        unmatched = set(admission)-seen
        if unmatched:
            raise ValueError('Support admission has no current source interface: '+str(sorted(unmatched)))
        self.rows.sort(key=lambda r: (r['pts'], r['foot']))
        counts = {}
        for row in self.rows:
            counts[row['pts']] = counts.get(row['pts'], 0)+1
        for row in self.rows:
            row['objective_weight'] = self.weight/counts[row['pts']]
        self.times = np.array([r['time'] for r in self.rows])
        self.feet = np.array([r['foot'] for r in self.rows], int)
        self.base = self.ref.at(self.times) if self.rows else None
        self.last = None

    def _local_jacobians(self, delta, state, witness, rotation, D):
        n = len(self.rows)
        point = witness['world']
        local = np.zeros((n, 3, 69))
        local[:, :, :3] = np.eye(3)
        dq = np.einsum('nij,njk->nik', rotation, witness['derivative_normal'])
        def apply(axes, pivot):
            fixed = -skew(point-pivot) @ axes
            dn = np.einsum('nji,njk->nik', rotation, skew(np.array([0., 0., 1.])) @ axes)
            return fixed+dq @ dn
        local[:, :, 3:6] = apply(state['Q'] @ right_jacobian(np.asarray(delta)[:, 3:6]), state['P'])
        for foot, ancestors in ((0, (1, 4, 7)), (1, (2, 5, 8))):
            selected = self.feet == foot
            for joint in ancestors:
                start = 6+(joint-1)*3
                axes = state['Q'] @ state['frames'][:, joint] @ right_jacobian(np.asarray(delta)[:, start:start+3])
                value = apply(axes, state['native_world'][:, joint])
                local[selected, :, start:start+3] = value[selected]
        RFB, tFB, _ = self.binding.parameters()
        ankle = -np.einsum('fi,fij->fj', tFB, RFB)
        normal = rotation[:, 2, :]
        right = -rotation @ skew(witness['local']-ankle[self.feet])+dq @ skew(normal)
        shoe = np.zeros((n, 3, 6))
        for foot in (0, 1):
            selected = self.feet == foot
            shoe[selected, :, foot*3:foot*3+3] = right[selected]
        local = np.einsum('nij,njk->nik', D, local*self.ratio)
        shoe = np.einsum('nij,njk->nik', D, shoe*self.ratio)
        J = delta.chain(local)+delta.owner.shoe_chain(shoe, delta.parameters, delta.sample_times)
        return hstack((csr_matrix((2*n, 17)), J), format='csr')

    def evaluate(self, x, jacobian=False):
        x = np.asarray(x, float)
        if x.shape == (self.problem.size,):
            if not np.array_equal(x[:17], self.binding.x):
                raise ValueError('Shared shoe shape/attachment is frozen for this factor')
            x = x[17:]
        if x.shape != (self.body.size,):
            raise ValueError('Expected one complete continuous body/relative-shoe state')
        if not self.rows:
            return dict(residual=np.empty(0), jac=csr_matrix((0, self.problem.size)) if jacobian else None,
                        parts={'source_support_location_xy': 0.}, image_cost=0.,
                        metrics={'source_support_locations': [], 'source_support_location_skipped': self.skipped})
        delta = self.body.state.increments(x, self.times)
        state = fk_state(self.ref, self.base, delta)
        each = np.arange(len(self.rows))
        rotation, origin = state['R'][each, self.feet], state['O'][each, self.feet]
        witness = _runner_witness(self.binding, rotation, origin)
        prediction = witness['world']*self.ratio
        residual, derivatives, metrics = [], [], []
        previous = {}
        for i, row in enumerate(self.rows):
            position = row['transform'] @ (prediction[i, :2]-row['centre'])
            r, dr, closest, boundary_kind = _polygon_terms(position, row['polygon'])
            scale = np.sqrt(row['objective_weight'])
            residual.append(scale*r)
            derivatives.append(np.c_[scale*dr @ row['transform'], np.zeros(2)])
            feature = row['feature']; mapping = feature['mapping']
            feature_id = witness['feature_ids'][witness['feature_index'][i]]
            old = previous.get(row['foot']); previous[row['foot']] = feature_id
            closest_xy = row['centre']+np.linalg.solve(row['transform'], closest)
            metrics.append(dict(pts=row['pts'], foot=row['foot'], role='fit',
                source_status=feature['source_status'], source_annotation=feature['source_annotation'],
                source_frame=feature['source_frame'], current_support_admission=row['admission'],
                predicted_xy_conditional_m=prediction[i, :2].tolist(),
                predicted_runner_height_conditional_m=float(prediction[i, 2]),
                closest_source_range_xy_conditional_m=closest_xy.tolist(),
                outside_source_range_xy_conditional_m=(prediction[i, :2]-closest_xy).tolist(),
                outside_range_normalized_source_pixels=float(np.linalg.norm(r)), residual=(scale*r).tolist(),
                source_polyline_xy_conditional_m=np.asarray(mapping['xyz_conditional_m'])[:, :2].tolist(),
                source_sampled_xy_aabb_conditional_m=mapping['sampled_xy_aabb_conditional_m'],
                source_pixel_uncertainty_radius_px=mapping['pixel_uncertainty_radius_px'],
                source_sampled_pixel_disk_boundary_xy_conditional_m=mapping['sampled_pixel_disk_boundary_xy_conditional_m'],
                source_range_convex_hull_xy_conditional_m=row['source_hull_xy_conditional_m'].tolist(),
                source_range_model='convex hull of complete supplied sampled pixel disks; finite sampled convexification, not a proven camera-error bound',
                source_boundary_projection=boundary_kind, model_body_time_seconds=row['time'],
                native_camera_row_time_seconds=row['row_time'], native_camera_row_time_span_seconds=row['row_time_span'],
                witness_feature=feature_id, witness_feature_changed_since_previous_source_sample=old is not None and old != feature_id,
                witness_theta_rad=float(witness['theta'][i]), witness_stationary_interior=bool(witness['interior'][i]),
                witness_local_solver=witness['local'][i].tolist(),
                next_runner_feature_height_gap_conditional_m=float(witness['next_feature_height_gap_solver'][i]*self.ratio),
                witness_material_correspondence='moving_lowest_exact_steel_geometry_not_fixed_material_or_source_midpoint',
                actual_contact_material_confirmed=False, independent_3d_observation=False,
                frame_weight_budget=self.frame_weight_budget, location_budget_fraction=self.location_budget_fraction,
                support_location_weight=row['objective_weight'],
                support_location_frame_weight=self.weight, reserved_complete_2d_curve_weight=self.curve_frame_weight))
        residual = np.asarray(residual).ravel()
        cost = float(residual @ residual/2)
        self.last = dict(residual=residual,
            jac=self._local_jacobians(delta, state, witness, rotation, np.asarray(derivatives)) if jacobian else None,
            parts={'source_support_location_xy': cost}, image_cost=cost,
            metrics=dict(source_support_locations=metrics, source_support_location_skipped=self.skipped,
                source_support_location_units=self.units,
                source_support_location_correlation='Same native source evidence as complete 2-D curves; frame budget is split, no independent 3-D count',
                source_support_location_normalization='Mean mapped dXY/dUV and median pixel radius are fixed engineering coordinates, not covariance or an accuracy gate',
                source_support_location_z_policy='No Z residual here; existing ice_policy finite soft terms remain authoritative',
                source_support_location_source_frames=len(set(r['pts'] for r in self.rows)),
                source_support_location_foot_rows=len(self.rows)))
        return self.last
