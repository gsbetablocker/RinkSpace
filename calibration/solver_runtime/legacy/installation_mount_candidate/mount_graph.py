"""Explicit shared nonzero camera-local mount re-estimation candidate.

One lever XYZ per physical installation, never a frame translation or a zoom
term. Original source pixels, loss roles, finite material and camera pivots are
unchanged. No hardware measurement or independently known lever is asserted.
"""
from __future__ import annotations
import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.sparse import csr_matrix, vstack
from all_installation_optics_candidate.all_installation_optics_graph import (
    InstallationOpticsChart, AllInstallationOpticsBlock, AllInstallationOpticsPatterns,
    install_all_installation_optics, bind_external_main_optics, evaluate_external_with_optics)
from secondary_optics_candidate.z190_motion_optics_extension import iter_motion_optics_inputs
from z190_motion_native import native_motion
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_shared_board_pattern_source import fixed_pivot_jacobian


def insert_mount_csr(matrix, values, indices, mount_begin):
    """Insert the new mount span before optics; preserve every old CSR entry."""
    m = matrix.tocsr(copy=False)
    values = np.asarray(values, float)
    slots = np.asarray(indices, dtype=m.indices.dtype)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) > m.shape[0] or
            not np.isfinite(values).all() or slots.shape != (3,) or
            not np.array_equal(slots, np.arange(slots[0], slots[0]+3)) or
            slots[0] < mount_begin or slots[-1] >= mount_begin+9 or
            np.any((m.indices >= mount_begin) & (m.indices < mount_begin+9)) or
            not m.has_sorted_indices):
        raise ValueError('Original sorted CSR and one explicit three-of-nine mount span required')
    active = values != 0.
    counts = np.zeros(m.shape[0], dtype=m.indptr.dtype)
    counts[:len(values)] = active.sum(axis=1)
    ptr = m.indptr + np.r_[0, np.cumsum(counts, dtype=m.indptr.dtype)]
    data = np.empty(int(ptr[-1]), m.data.dtype)
    columns = np.empty(len(data), m.indices.dtype)
    for lo in range(0, m.shape[0], 65536):
        hi = min(lo+65536, m.shape[0])
        left, right = map(int, m.indptr[[lo, hi]])
        sizes = np.diff(m.indptr[lo:hi+1])
        rows = np.repeat(np.arange(lo, hi, dtype=m.indptr.dtype), sizes)
        inner = np.arange(right-left, dtype=m.indptr.dtype) - np.repeat(m.indptr[lo:hi]-left, sizes)
        old_cols = m.indices[left:right]
        tail = old_cols >= mount_begin+9
        dest = ptr[rows] + inner + counts[rows]*tail
        data[dest], columns[dest] = m.data[left:right], old_cols
        stop = min(hi, len(values))
        if stop > lo:
            tail_counts = np.bincount(rows[tail]-lo, minlength=hi-lo)
            rr, cc = np.nonzero(active[lo:stop])
            rank = np.cumsum(active[lo:stop], axis=1)-1
            original_prefix = sizes[:stop-lo]-tail_counts[:stop-lo]
            dest = ptr[lo+rr] + original_prefix[rr] + rank[rr,cc]
            data[dest], columns[dest] = values[lo+rr, cc], slots[cc]
    result = csr_matrix((data, columns, ptr), shape=m.shape, copy=False)
    result.has_sorted_indices = True
    result.has_canonical_format = m.has_canonical_format
    return result


@dataclass(frozen=True)
class MountOpticsChart(InstallationOpticsChart):
    mount_indices: tuple[int, int, int] = (0, 0, 0)
    def calibration(self, base, q):
        current = super().calibration(base, q)
        delta = np.asarray(q)[np.asarray(self.mount_indices)]
        if np.all(delta == 0.):
            return current
        return replace(current, lever_xyz=tuple(np.asarray(base.lever_xyz)+delta/SCENE_UNIT_M))


def motion_mount_columns(bundle, local, cal):
    """Reuse exact bounded/profiled native shared derivatives for lever metres."""
    out = np.empty((bundle.motion_rows, 3))
    kernel = bundle.motion_kernel
    for lo, hi, k in iter_motion_optics_inputs(bundle, local, cal):
        n = hi-lo
        ds_a, ds_b = np.zeros((n,3,4)), np.zeros((n,3,4))
        ds_a[:,:,:3] = np.transpose(k['ra'],(0,2,1))/SCENE_UNIT_M
        ds_b[:,:,:3] = np.transpose(k['rb'],(0,2,1))/SCENE_UNIT_M
        k.update(jacobian=True, da=np.zeros((3,3,3)), db=np.zeros((3,3,3)),
            dca_shared=ds_a, dcb_shared=ds_b, dca_view=np.zeros((3,4)), dcb_view=np.zeros((3,4)))
        result, _ = native_motion(**k, minimum_source_depth_scene_units=kernel.minimum_source_depth_scene_units,
            schur=kernel.schur)
        out[2*lo:2*hi] = result[3][:,:3]
    return out


class MountOpticsBlock(AllInstallationOpticsBlock):
    def evaluate(self, q, jacobian=True):
        original = super().evaluate(q, jacobian)
        if not jacobian:
            return original
        residual, matrix = original
        local = self.local(q)
        cal = self.bundle.current_calibration(local)
        values = np.zeros((self.bundle.source_residual_rows,3))
        values[:self.bundle.motion_rows] = motion_mount_columns(self.bundle,local,cal)/self.bundle.motion_sigma[:,None]
        cursor = self.bundle.motion_rows
        geometry = self.geometry_values(q)
        for p, old_rows in self.groups:
            cam = self.bundle.camera(local,p)
            moved = [self.geometry.rebind(row,geometry) for row in old_rows]
            _, raw = self.absolute(cam,moved,True)
            rotation = cv2.Rodrigues(np.asarray(cam[3:6]))[0]
            count = sum(2*len(row['uv']) for row in old_rows)
            values[cursor:cursor+count] = np.asarray(raw)[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
            cursor += count
        if cursor != len(values):
            raise ValueError('Complete unchanged absolute mount source inventory required')
        return residual, insert_mount_csr(matrix,values,self.optics_chart.mount_indices,self.mount_begin)


class MountOpticsPatterns(AllInstallationOpticsPatterns):
    def camera(self,q,frame,*,held=False):
        if frame['observer_state_owner']=='programme':
            p=int(frame['source_actual_pts'])
            block=self.state.bypts[p]
            from structured_programme_source_candidate.native_camera_local import native_camera_local
            local=native_camera_local(block,q)
            cal=block.bundle.current_calibration(local)
            cam=cal.camera25(block.bundle.view(local,p))
            return cam,block.xyz_slots,block.start+4*block.bundle.index[p],np.asarray(cal.lever_xyz)*SCENE_UNIT_M
        if held and frame['observer_state_owner']=='external':
            with bind_external_main_mount(self.state,q):
                return super().camera(q,frame,held=held)
        return super().camera(q,frame,held=held)

    def evaluate(self, q, jacobian=True, *, held=False):
        if not jacobian or held:
            return super().evaluate(q,jacobian,held=held)
        q9 = q[self.start:self.start+9]
        self.geometry.guard(q9)
        values, matrices = [], []
        from z190_parallel_pattern_evaluation import take_pattern_frames
        pending = take_pattern_frames(self,q,True)
        for ordinal, frame in enumerate(self.fit):
            if pending is None:
                cam, slots, view, lever = self.camera(q,frame)
                result = self.frame_factor(frame).evaluate(q9,cam,frame,with_jacobian=True)
            else:
                _, (cam,slots,view,lever), future = pending[ordinal]
                result = future.result()
            values.append(result.residual)
            jxyz,jview = fixed_pivot_jacobian(result.camera_state11_jacobian,cam,lever,SCENE_UNIT_M)
            matrix = np.zeros((len(result.residual),len(q)))
            matrix[:,slots],matrix[:,view:view+4] = jxyz,jview
            matrix[:,self.start:self.start+9] = result.shared_q9_jacobian
            chart = self.chart_for_frame(frame)
            matrix[:,chart.indices] = result.optics_jacobian
            rotation = cv2.Rodrigues(np.asarray(cam[3:6]))[0]
            matrix[:,chart.mount_indices] = np.asarray(result.camera_state11_jacobian)[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
            matrices.append(csr_matrix(matrix))
        return np.concatenate(values),vstack(matrices,format='csr')


@contextmanager
def bind_external_main_mount(state, q):
    external = state.external
    if external is None:
        yield None
        return
    old = external.lever_m
    chart = state.installation_optics[state.ids[0]]
    cal = chart.calibration(state.external_optics_base_calibration,q)
    external.lever_m = np.asarray(cal.lever_xyz)*SCENE_UNIT_M
    try:
        with bind_external_main_optics(state,q):
            yield external
    finally:
        external.lever_m = old


def evaluate_external_with_mount(state,q,*,jacobian=True,held=False,local_override=None,evaluator=None):
    def original(value,jac):
        with bind_external_main_mount(state,value):
            return evaluate_external_with_optics(state,value,jacobian=jac,held=held,
                local_override=local_override,evaluator=evaluator)
    result = original(q,jacobian)
    if not jacobian:
        return result
    chart = state.installation_optics[state.ids[0]]
    columns = np.empty((len(result.residual),3))
    expected = [row['id'] for row in result.source_row_ledger]
    for j,index in enumerate(chart.mount_indices):
        plus,minus = q.copy(),q.copy()
        plus[index] += 1e-7
        minus[index] -= 1e-7
        a,b = original(plus,False),original(minus,False)
        if [r['id'] for r in a.source_row_ledger] != expected or [r['id'] for r in b.source_row_ledger] != expected:
            raise ValueError('Mount FD changed external source inventory')
        columns[:,j] = (a.residual-b.residual)/(2e-7)
    return SimpleNamespace(**vars(result),mount_jacobian=columns,mount_parameter_indices=np.asarray(chart.mount_indices))


def _mount_chart_q(chart, q):
    """Use the actual chart's declared core; preserve a caller-owned full tail."""
    value = np.asarray(q, float)
    coordinates = getattr(chart, 'coordinate_chart', None)
    count = (int(coordinates.core_count) if coordinates is not None
             else int(chart.original_parameter_count) + 9)
    if value.ndim != 1 or len(value) < count or not np.isfinite(value).all():
        raise ValueError('Complete finite state containing the actual mount/optics chart required')
    return value[:count]


def _mount_members(state, name):
    members = [block for block in state.blocks
               if block.bundle.calibration.installation_id == name]
    if not members:
        raise ValueError('Shared installation mount has no actual programme member: ' + name)
    return members


def _current_mount_lever(chart, member, q):
    return (np.asarray(chart.calibration(member.optics_base_calibration,
                                        _mount_chart_q(chart, q)).lever_xyz, float)
            * SCENE_UNIT_M)


def _mount_inside_working_range(lever, entry, tolerance):
    return (np.isfinite(lever).all() and np.linalg.norm(lever) > 1e-8
            and np.all(lever >= np.asarray(entry['absolute_lever_lower_m']) - tolerance)
            and np.all(lever <= np.asarray(entry['absolute_lever_upper_m']) + tolerance))


def install_mount_working_constraints(state, facts_path):
    """Install explicit user-adopted shared mount ranges without changing q.

    Absolute camera-local work-metre ranges are converted to the actual
    chart's metre increments from its authenticated base lever.  These photo
    ranges are user working limits, never surveyed hardware dimensions.  An
    existing out-of-range checkpoint is left intact and will fail validation
    until the caller explicitly invokes its separate initializer.
    """
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Install working mount constraints before source workers')
    path = Path(facts_path).resolve()
    payload = path.read_bytes()
    facts = json.loads(payload.decode('utf8'))
    mount = facts.get('rigid_mount_offset', {})
    if (facts.get('schema_version') != 'rinkspace.user-camera-facts.v1'
            or mount.get('authority') != 'user_adopted_photo_based_initialization_and_bounds_from_dot'
            or mount.get('coordinate_frame') != 'camera-local OpenCV x right, y down, z forward'
            or mount.get('bounds_are_user_working_search_ranges_not_measurements') is not True
            or mount.get('shared_across_same_installation_exposures') is not True
            or mount.get('separate_offset_for_each_installation') is not True
            or mount.get('first_version_zoom_dependent_offset_enabled') is not False
            or mount.get('per_frame_free_optical_centre_enabled') is not False):
        raise ValueError('Explicit v3-compatible user working shared rigid-mount facts required')
    if (not np.isfinite(facts.get('metres_per_original_scene_unit', np.nan))
            or abs(float(facts['metres_per_original_scene_unit']) - SCENE_UNIT_M) > 1e-15):
        raise ValueError('Mount working ranges require the same declared uniform scene scale')
    initial = np.asarray(mount.get('initial_camera_xyz_m'), float)
    lower = np.asarray(mount.get('lower_camera_xyz_m'), float)
    upper = np.asarray(mount.get('upper_camera_xyz_m'), float)
    if (any(value.shape != (3,) or not np.isfinite(value).all()
            for value in (initial, lower, upper))
            or np.any(lower >= upper) or np.any(initial < lower)
            or np.any(initial > upper) or np.linalg.norm(initial) <= 1e-8):
        raise ValueError('Finite ordered user mount XYZ ranges and nonzero in-range initializer required')
    q = np.asarray(state.q, float)
    tolerance = 1e-12
    installations, indices, delta_lower, delta_upper = {}, [], [], []
    for name, chart in state.installation_optics.items():
        members = _mount_members(state, name)
        slots = np.asarray(chart.mount_indices, int)
        if (slots.shape != (3,) or len(np.unique(slots)) != 3
                or np.any(slots < 0) or np.any(slots >= len(q))):
            raise ValueError('Actual explicit three shared mount increment slots required')
        lever = _current_mount_lever(chart, members[0], q)
        zero = q.copy()
        zero[slots] = 0.
        base = _current_mount_lever(chart, members[0], zero)
        if (not np.isfinite(base).all()
                or np.max(np.abs(lever - q[slots] - base)) > tolerance):
            raise ValueError('Actual mount chart is not the explicit metre increment from its base lever')
        for member in members:
            if (not np.array_equal(np.asarray(member.optics_chart.mount_indices), slots)
                    or np.max(np.abs(_current_mount_lever(chart, member, q) - lever)) > tolerance
                    or np.max(np.abs(_current_mount_lever(chart, member, zero) - base)) > tolerance):
                raise ValueError('One current base lever and shared mount triplet per physical installation required')
        entry = dict(
            mount_parameter_indices=slots.tolist(), base_lever_m=base.tolist(),
            absolute_lever_initial_m=initial.tolist(),
            absolute_lever_lower_m=lower.tolist(), absolute_lever_upper_m=upper.tolist(),
            delta_initial_m=(initial - base).tolist(),
            delta_lower_m=(lower - base).tolist(), delta_upper_m=(upper - base).tolist(),
            programme_members=[member.name for member in members])
        installations[name] = entry
        indices.extend(slots.tolist())
        delta_lower.extend(entry['delta_lower_m'])
        delta_upper.extend(entry['delta_upper_m'])
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Distinct explicit shared mount slots for all installations required')
    contract = dict(
        schema='rinkspace.user-working-shared-rigid-mount-constraints.v1',
        facts=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                   schema_version=facts['schema_version'], authority=mount['authority'],
                   programme_source_sha256=facts.get('programme_source_sha256')),
        shared_mount_parameter_indices=indices, delta_lower_m=delta_lower,
        delta_upper_m=delta_upper, by_installation=installations,
        coordinate_frame=mount['coordinate_frame'], scene_unit_working_m=SCENE_UNIT_M,
        bounds_are_user_working_search_ranges_not_measurements=True,
        numeric_comparison_tolerance_working_m=tolerance,
        one_lever_shared_across_own_installation_exposures=True,
        independent_lever_per_installation=True, zoom_dependent_offset=False,
        per_frame_free_optical_centre=False, nominal_q_changed=False,
        range_installation_silently_clipped_state=False,
        optical_height_prior_or_new_source_rows_added=False,
        source_pixels_sigmas_roles_profiles_and_held_unchanged=True,
        measured_hardware_or_metric_calibration_claim=False, accuracy_accepted=False)
    state.mount_working_constraints = contract
    state.mount_working_parameter_bounds = {
        int(index): (float(low), float(high))
        for index, low, high in zip(indices, delta_lower, delta_upper)}
    return contract


def validate_mounts(state, q, *, check_pivot_exterior=True):
    constraints = getattr(state, 'mount_working_constraints', None)
    for name, chart in state.installation_optics.items():
        member = _mount_members(state, name)[0]
        lever = _current_mount_lever(chart, member, q)
        if not np.isfinite(lever).all() or np.linalg.norm(lever) <= 1e-8:
            raise ValueError('One finite nonzero physical lever per installation required')
        if constraints is not None:
            entry = constraints['by_installation'].get(name)
            if (entry is None or not np.array_equal(np.asarray(chart.mount_indices),
                                                    entry['mount_parameter_indices'])):
                raise ValueError('Installed working range does not belong to the actual shared mount chart')
            if not _mount_inside_working_range(
                    lever, entry, constraints['numeric_comparison_tolerance_working_m']):
                raise ValueError('Camera-local lever exceeds the user working mount range: ' + name)
    if (check_pivot_exterior
            and getattr(state, 'current_user_pivot_outside_constraint_contract', None) is not None):
        report = pivot_outside_domain(state, q)
        invalid = [row['installation_id'] for row in report['installations']
                   if not row['valid']]
        if invalid:
            raise ValueError('Fixed tripod pivot is inside the current actual board course: '
                             + ', '.join(invalid))


def _pivot_course(state, q, contract):
    """Use the actual chart and current producer consumed by source and scene."""
    value = np.asarray(q, float)
    slots = np.asarray(contract['world_parameter_indices'], int)
    if (value.ndim != 1 or not np.isfinite(value).all()
            or np.any(slots < 0) or np.any(slots >= len(value))):
        raise ValueError('Complete finite current state and explicit world slots required')
    producer = state.geometry.current
    course, course_q, chain = producer.print_course_binding(value[slots])
    points = np.asarray(course.sample_perimeter(course_q)[0], float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or not np.isfinite(points).all()):
        raise ValueError('Complete current actual board course required')
    polygon = points[:, :2] * SCENE_UNIT_M
    if np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 3:
        raise ValueError('Nondegenerate current closed board course required')
    area = .5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                       - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if not np.isfinite(area) or abs(area) <= 1e-12:
        raise ValueError('Current actual board course has no finite oriented area')
    return polygon, float(area)


def _pivot_outward_margin(polygon, area, xy):
    """Double-precision signed distance; positive means outside, zero is legal."""
    xy = np.asarray(xy, float)
    start = polygon
    delta = np.roll(polygon, -1, axis=0) - start
    length2 = np.einsum('ij,ij->i', delta, delta)
    valid = length2 > 0.
    if not np.any(valid):
        raise ValueError('Actual board course has no nonzero segment')
    original_indices = np.flatnonzero(valid)
    start, delta, length2 = start[valid], delta[valid], length2[valid]
    station = np.clip(np.einsum('ij,ij->i', xy - start, delta) / length2, 0., 1.)
    closest = start + station[:, None] * delta
    differences = xy - closest
    distances2 = np.einsum('ij,ij->i', differences, differences)
    index = int(np.argmin(distances2))
    distance = float(np.sqrt(distances2[index]))
    # Half-open ray crossings avoid a vertex being counted twice. Unlike
    # OpenCV pointPolygonTest, this does not first round the course to float32.
    crosses = (start[:, 1] > xy[1]) != ((start + delta)[:, 1] > xy[1])
    crossing_x = start[crosses, 0] + ((xy[1] - start[crosses, 1])
                                    * delta[crosses, 0] / delta[crosses, 1])
    inside = bool(np.count_nonzero(crossing_x > xy[0]) % 2)
    roundoff = 64. * np.finfo(float).eps * max(
        1., float(np.max(abs(polygon))), float(np.max(abs(xy))))
    if distance <= roundoff:
        margin = 0.
        d = delta[index] / np.sqrt(length2[index])
        outward = np.array([d[1], -d[0]]) * (1. if area > 0. else -1.)
        side = 'on'
    else:
        margin = -distance if inside else distance
        outward = differences[index] / distance * (-1. if inside else 1.)
        side = 'inside' if inside else 'outside'
    return dict(outward_margin_working_m=float(margin), side=side,
                valid=bool(margin >= 0.), nearest_course_xy_working_m=closest[index].tolist(),
                nearest_segment_index=int(original_indices[index]),
                nearest_segment_station=float(station[index]),
                outward_normal_xy=np.asarray(outward).tolist(),
                numerical_boundary_roundoff_working_m=roundoff)


def install_pivot_outside_constraints(state, facts_path):
    """Install only an explicit user outside relation, never a marked XY box."""
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Install explicit pivot domains before source workers')
    path = Path(facts_path).resolve()
    payload = path.read_bytes()
    facts = json.loads(payload.decode('utf8'))
    if (facts.get('schema') != 'rinkspace.user-tripod-position-evidence.v1'
            or facts.get('authority') != 'current_user_direct_annotation'
            or facts.get('no_new_user_tolerance_inferred') is not True
            or not np.isfinite(facts.get('metres_per_scene_unit', np.nan))
            or abs(float(facts['metres_per_scene_unit']) - SCENE_UNIT_M) > 1e-15):
        raise ValueError('Explicit immutable user tripod outside facts in the current working scale required')
    charts = tuple(state.installation_optics.values())
    if not charts or any(getattr(chart, 'coordinate_chart', None) is None for chart in charts):
        raise ValueError('Actual explicit current physical camera/world chart required')
    world_slots = tuple(map(int, charts[0].coordinate_chart.world_indices))
    if any(tuple(map(int, chart.coordinate_chart.world_indices)) != world_slots
           for chart in charts):
        raise ValueError('One current world chart across installations required')
    installations = {}
    for key in ('low_end', 'long_side_main', 'long_side_secondary'):
        source = facts.get(key, {})
        if source.get('outside_ice_boundary') is not True:
            continue
        ordinal = source.get('installation_id')
        if (isinstance(ordinal, bool) or not isinstance(ordinal, int)
                or ordinal < 0 or ordinal >= len(state.ids)):
            raise ValueError('Explicit user installation identity for outside pivot required')
        name = state.ids[ordinal]
        slots = np.asarray(state.slots[name], int)
        if slots.shape != (3,) or len(set(slots.tolist())) != 3:
            raise ValueError('Actual distinct pivot XYZ slots required')
        if any(not np.array_equal(member.xyz_slots, slots)
               for member in _mount_members(state, name)):
            raise ValueError('Outside pivot slots disagree with actual installation members')
        installations[name] = dict(pivot_parameter_indices=slots.tolist(),
                                   user_fact_field=key + '.outside_ice_boundary',
                                   relation='pivot_outside_or_on_current_actual_board_course')
    if not installations:
        raise ValueError('No explicit user outside-pivot relation was supplied')
    contract = dict(schema='rinkspace.explicit-user-fixed-tripod-outside-current-course.v1',
                    facts=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                               authority=facts['authority'], source_sha256=facts.get('source_sha256')),
                    by_installation=installations, world_parameter_indices=list(world_slots),
                    current_course_provider='actual geometry.current.print_course_binding/sample_perimeter',
                    scene_unit_working_m=SCENE_UNIT_M, outside_margin_positive=True,
                    on_boundary_allowed=True, hard_inside_rejection=True,
                    marked_nominal_XY_or_tolerance_not_a_constraint=True,
                    optical_centre_not_constrained_outside=True,
                    numerical_world_derivative_step=1e-6,
                    new_source_or_prior_rows=0, nominal_q_changed=False,
                    measured_offset_or_metric_calibration_claim=False, accuracy_accepted=False)
    state.current_user_pivot_outside_constraint_contract = contract
    return contract


def pivot_outside_domain(state, q):
    contract = getattr(state, 'current_user_pivot_outside_constraint_contract', None)
    if contract is None:
        return dict(schema='rinkspace.current-pivot-outside-domain-report.v1', installed=False,
                    valid=True, installations=[])
    polygon, area = _pivot_course(state, q, contract)
    value = np.asarray(q, float)
    rows = []
    for name, entry in contract['by_installation'].items():
        slots = np.asarray(entry['pivot_parameter_indices'], int)
        row = _pivot_outward_margin(polygon, area, value[slots[:2]])
        row.update(installation_id=name, pivot_parameter_indices=slots.tolist(),
                   pivot_xyz_working_m=value[slots].tolist())
        rows.append(row)
    return dict(schema='rinkspace.current-pivot-outside-domain-report.v1', installed=True,
                valid=all(row['valid'] for row in rows), installations=rows,
                actual_course_sha256=hashlib.sha256(polygon.tobytes()).hexdigest(),
                actual_course_vertices=len(polygon), working_metres_not_surveyed=True)


def pivot_outside_linear_step_constraints(state, q, shared_indices):
    """A @ delta_shared >= -margin; an inequality, never an objective row.

    Camera pivot derivatives are exact signed-distance gradients. World
    derivatives use the original 1e-6 symmetric step on the same actual
    current-course value, including nearest-segment recomputation. The
    nonlinear guard must still validate every trial state.
    """
    report = pivot_outside_domain(state, q)
    selected = np.asarray(shared_indices, int).reshape(-1)
    value = np.asarray(q, float)
    if (len(set(selected.tolist())) != len(selected) or np.any(selected < 0)
            or np.any(selected >= len(value))):
        raise ValueError('Distinct actual shared parameter indices required')
    position = {int(index): i for i, index in enumerate(selected)}
    rows = report['installations']
    matrix = np.zeros((len(rows), len(selected)))
    lower = -np.asarray([row['outward_margin_working_m'] for row in rows])
    if not report['installed']:
        return dict(matrix=matrix, lower=lower, report=report)
    contract = state.current_user_pivot_outside_constraint_contract
    for i, row in enumerate(rows):
        for column, derivative in zip(row['pivot_parameter_indices'][:2], row['outward_normal_xy']):
            if column not in position:
                raise ValueError('Constrained pivot XY is absent from shared solve coordinates')
            matrix[i, position[column]] = derivative
    h = float(contract['numerical_world_derivative_step'])
    producer = state.geometry.current
    world_slots = np.asarray(contract['world_parameter_indices'], int)
    _, _, chain = producer.print_course_binding(value[world_slots])
    chain = np.asarray(chain, float)
    if chain.ndim != 2 or chain.shape[1] != len(world_slots) or not np.isfinite(chain).all():
        raise ValueError('Explicit finite current-course/world derivative chain required')
    course_world_columns = world_slots[np.any(chain != 0., axis=0)]
    branches = []
    for column in course_world_columns:
        plus, minus = value.copy(), value.copy()
        plus[column] += h
        minus[column] -= h
        pp, pa = _pivot_course(state, plus, contract)
        mp, ma = _pivot_course(state, minus, contract)
        for i, row in enumerate(rows):
            xy = value[np.asarray(row['pivot_parameter_indices'][:2], int)]
            rp = _pivot_outward_margin(pp, pa, xy)
            rm = _pivot_outward_margin(mp, ma, xy)
            derivative = (rp['outward_margin_working_m'] - rm['outward_margin_working_m']) / (2. * h)
            if column in position:
                matrix[i, position[column]] = derivative
            elif derivative != 0.:
                raise ValueError('Current course derivative is absent from shared solve coordinates')
            if derivative != 0.:
                branches.append(dict(installation_id=row['installation_id'], column=int(column),
                                     derivative=float(derivative), plus_segment=rp['nearest_segment_index'],
                                     minus_segment=rm['nearest_segment_index']))
    resolution = h * np.maximum(1., np.sum(abs(matrix), axis=1))
    active = np.array([0. <= row['outward_margin_working_m'] <= limit
                       for row, limit in zip(rows, resolution)], bool)
    lower[active] = 0.
    report = dict(report, derivative_world_step=h, world_derivative_branches=branches,
                  current_course_world_parameter_indices=course_world_columns.tolist(),
                  numerical_margin_resolution_working_m=resolution.tolist(),
                  numerical_resolution_is_not_a_physical_offset=True,
                  active_tangent_face=active.tolist(),
                  nominal_outward_margin_working_m=[row['outward_margin_working_m'] for row in rows],
                  step_lower_working_m=lower.tolist(),
                  shared_parameter_indices=selected.tolist(),
                  nonlinear_trial_guard_still_required=True, objective_rows_added=0)
    return dict(matrix=matrix, lower=lower, report=report)


def reinitialize_invalid_pivots_outside(state, q, *, require_coordinate_fd_margin=True):
    """Nearest-course Pxy feasibility, with an explicit initialization mode.

    Startup initialization retains its original coordinate-FD reserve. A
    nonlinear trial needs only the declared outside-or-on domain: retract it
    to the same nearest course point, without adding a physical offset or
    moving it by the unrelated derivative stencil size.
    """
    if not isinstance(require_coordinate_fd_margin, bool):
        raise ValueError('Explicit boolean coordinate-FD initialization mode required')
    parent = np.asarray(q, float)
    before = pivot_outside_domain(state, parent)
    if not before['installed']:
        raise ValueError('Install explicit outside-pivot facts before initialization')
    initialized = parent.copy()
    contract = state.current_user_pivot_outside_constraint_contract
    selected = np.unique(np.r_[contract['world_parameter_indices'],
                               *[row['pivot_parameter_indices'][:2]
                                 for row in before['installations']]])
    linear = (pivot_outside_linear_step_constraints(state, parent, selected)
              if require_coordinate_fd_margin else None)
    changes = []
    h = float(contract['numerical_world_derivative_step'])
    for i, row in enumerate(before['installations']):
        if row['valid']:
            continue
        slots = np.asarray(row['pivot_parameter_indices'], int)
        normal = np.asarray(row['outward_normal_xy'], float)
        nearest = np.asarray(row['nearest_course_xy_working_m'], float)
        # This is solely a numerical interior margin for original h-sized
        # coordinate/directional FD. It is not a physical tripod offset.
        epsilon = (32. * h * max(1., float(np.sum(abs(linear['matrix'][i]))))
                   if require_coordinate_fd_margin else 0.)
        initialized[slots[:2]] = nearest + epsilon * normal
        polygon, area = _pivot_course(state, initialized, contract)
        after = _pivot_outward_margin(polygon, area, initialized[slots[:2]])
        if not after['valid'] or (require_coordinate_fd_margin
                                 and after['outward_margin_working_m'] <= 0.):
            raise ValueError('Nearest-course outward pivot initializer is not strictly domain feasible')
        minimum_fd_margin = after['outward_margin_working_m'] if require_coordinate_fd_margin else None
        if require_coordinate_fd_margin:
            for column in selected:
                for sign in (-1., 1.):
                    trial = initialized.copy()
                    trial[column] += sign * h
                    poly, trial_area = _pivot_course(state, trial, contract)
                    margin = _pivot_outward_margin(poly, trial_area, trial[slots[:2]])['outward_margin_working_m']
                    minimum_fd_margin = min(minimum_fd_margin, margin)
        if require_coordinate_fd_margin and minimum_fd_margin <= 0.:
            raise ValueError('Outward initializer has insufficient original FD numerical margin')
        movement = initialized[slots] - parent[slots]
        changes.append(dict(installation_id=row['installation_id'], changed_columns=slots[:2].tolist(),
                            before_pivot_xyz_working_m=parent[slots].tolist(),
                            after_pivot_xyz_working_m=initialized[slots].tolist(),
                            before_outward_margin_working_m=row['outward_margin_working_m'],
                            after_outward_margin_working_m=after['outward_margin_working_m'],
                            numerical_initializer_margin_working_m=epsilon,
                            original_domain_boundary_retraction=not require_coordinate_fd_margin,
                            minimum_original_coordinate_FD_margin_working_m=minimum_fd_margin,
                            every_exposure_optical_C_translation_working_m=movement.tolist(),
                            every_exposure_optical_C_translation_norm_working_m=float(np.linalg.norm(movement)),
                            original_RF_or_source_values_reinitialized=False))
    after = pivot_outside_domain(state, initialized)
    if not after['valid']:
        raise ValueError('Explicit outside pivot initialization failed its actual nonlinear domain')
    validate_mounts(state, initialized)
    columns = np.flatnonzero(initialized != parent).tolist()
    allowed = {column for row in before['installations'] for column in row['pivot_parameter_indices'][:2]}
    if not set(columns).issubset(allowed):
        raise ValueError('Outside initializer changed a nonpivot coordinate')
    report = dict(schema='rinkspace.explicit-nearest-current-course-pivot-initializer.v1',
                  applied=bool(changes), changed_columns=columns, changes=changes,
                  before_domain=before, after_domain=after,
                  parent_q_sha256=hashlib.sha256(parent.tobytes()).hexdigest(),
                  initialized_q_sha256=hashlib.sha256(initialized.tobytes()).hexdigest(),
                  parent_state_readonly=True, nominal_state_assigned=False,
                  original_source_domain_revalidated=False,
                  initializer_is_not_accepted_fit_or_observation_equivalence=True,
                  world_RF_lever_Pz_optics_profiles_and_suffix_unchanged=True,
                  numerical_margin_is_not_a_measured_physical_offset=True,
                  original_coordinate_fd_margin_required=require_coordinate_fd_margin,
                  original_domain_boundary_retraction=not require_coordinate_fd_margin,
                  accuracy_accepted=False)
    return initialized, report


def reinitialize_invalid_mounts(state, q):
    """Explicitly replace invalid levers while closing only each reference C.

    Returns a full q copy and a small report.  No state.q, profile, world,
    Rodrigues/logf, or material suffix is modified.  Keeping one reference
    optical centre does not preserve all source projections or observation
    equivalence; every exposure's induced centre movement is reported.
    """
    constraints = getattr(state, 'mount_working_constraints', None)
    if constraints is None:
        raise ValueError('Explicit installed user working mount constraints required before initialization')
    parent = np.asarray(q, float)
    if parent.ndim != 1 or not np.isfinite(parent).all():
        raise ValueError('Complete finite parent state required for explicit mount initialization')
    original = parent.copy()
    initialized = parent.copy()
    tolerance = float(constraints['numeric_comparison_tolerance_working_m'])
    reports, permitted = [], []
    for name, chart in state.installation_optics.items():
        members = _mount_members(state, name)
        entry = constraints['by_installation'][name]
        slots = np.asarray(chart.mount_indices, int)
        if not np.array_equal(slots, entry['mount_parameter_indices']):
            raise ValueError('Explicit initializer must use the actual installed mount triplet')
        old_lever = _current_mount_lever(chart, members[0], parent)
        if _mount_inside_working_range(old_lever, entry, tolerance):
            reports.append(dict(installation_id=name, changed=False,
                                lever_m=old_lever.tolist(), reason='already within user working range'))
            continue
        reference = members[0]
        ref_pts = int(reference.bundle.reference_pts)
        if ref_pts not in reference.bundle.index:
            raise ValueError('Existing installation reference exposure is absent from its actual bundle')
        ref_start = reference.start + 4 * reference.bundle.index[ref_pts]
        ref_rotation = cv2.Rodrigues(parent[ref_start:ref_start+3])[0]
        pivot_slots = np.asarray(reference.xyz_slots, int)
        if (pivot_slots.shape != (3,) or any(not np.array_equal(member.xyz_slots, pivot_slots)
                                           for member in members)
                or np.intersect1d(pivot_slots, slots).size):
            raise ValueError('One distinct shared fixed-pivot XYZ triplet per installation required')
        old_pivot = parent[pivot_slots].copy()
        new_lever = np.asarray(entry['absolute_lever_initial_m'], float)
        new_pivot = old_pivot + ref_rotation.T @ (old_lever - new_lever)
        if not np.isfinite(new_pivot).all() or new_pivot[2] <= 0.:
            raise ValueError('Reference-preserving explicit mount initializer leaves the positive fixed-pivot domain')
        initialized[slots] = new_lever - np.asarray(entry['base_lever_m'])
        initialized[pivot_slots] = new_pivot
        actual_new_lever = _current_mount_lever(chart, reference, initialized)
        old_reference_c = old_pivot + ref_rotation.T @ old_lever
        new_reference_c = new_pivot + ref_rotation.T @ actual_new_lever
        closure = float(np.max(np.abs(new_reference_c - old_reference_c)))
        if closure > tolerance:
            raise ValueError('Actual mount chart failed the preserved reference optical-centre closure')
        bypts = getattr(state, 'bypts', None)
        frames = ([(int(pts), block) for pts, block in bypts.items()
                   if block.bundle.calibration.installation_id == name] if bypts is not None
                  else [(int(pts), block) for block in members for pts in block.bundle.stamps])
        changes, times = [], []
        for pts, block in frames:
            start = block.start + 4 * block.bundle.index[pts]
            rotation = cv2.Rodrigues(parent[start:start+3])[0]
            changes.append(new_pivot - old_pivot + rotation.T @ (actual_new_lever - old_lever))
            times.append(pts)
        changes = np.asarray(changes, float)
        if not len(changes) or not np.isfinite(changes).all():
            raise ValueError('All actual installation exposures required for initializer centre-change report')
        distance = np.linalg.norm(changes, axis=1)
        permitted.extend(slots.tolist() + pivot_slots.tolist())
        reports.append(dict(
            installation_id=name, changed=True, reference_member=reference.name,
            reference_pts=ref_pts, mount_parameter_indices=slots.tolist(),
            pivot_parameter_indices=pivot_slots.tolist(),
            before_pivot_m=old_pivot.tolist(), after_pivot_m=new_pivot.tolist(),
            before_lever_m=old_lever.tolist(), after_lever_m=actual_new_lever.tolist(),
            before_reference_optical_centre_m=old_reference_c.tolist(),
            after_reference_optical_centre_m=new_reference_c.tolist(),
            reference_C_max_absolute_closure_working_m=closure,
            actual_exposure_count=len(times), first_actual_pts=min(times), last_actual_pts=max(times),
            per_frame_C_change_xyz_min_m=changes.min(axis=0).tolist(),
            per_frame_C_change_xyz_max_m=changes.max(axis=0).tolist(),
            per_frame_C_change_norm_min_m=float(distance.min()),
            per_frame_C_change_norm_median_m=float(np.median(distance)),
            per_frame_C_change_norm_p90_m=float(np.percentile(distance, 90)),
            per_frame_C_change_norm_max_m=float(distance.max()),
            max_C_change_actual_pts=times[int(np.argmax(distance))]))
    changed = np.flatnonzero(initialized != original)
    if (not np.array_equal(parent, original)
            or not set(changed.tolist()).issubset(set(permitted))):
        raise ValueError('Explicit mount initializer changed protected parent state coordinates')
    validate_mounts(state, initialized, check_pivot_exterior=False)
    report = dict(
        schema='rinkspace.explicit-invalid-mount-reference-centre-initialization.v1',
        parent_q_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
        initialized_q_sha256=hashlib.sha256(initialized.tobytes()).hexdigest(),
        facts=constraints['facts'], changed_parameter_indices=changed.tolist(),
        initialization_stage_scope='shared lever working range only; exterior stage follows',
        pivot_exterior_validated=False,
        installations=reports, parent_q_readonly=True, state_q_not_assigned=True,
        only_invalid_installation_mount_and_shared_pivot_changed=True,
        reference_centres_preserved_only=True, all_frame_observation_equivalence_claimed=False,
        world_R_logf_material_suffix_and_profiles_changed=False,
        source_UV_sigmas_FIT_HELD_and_original_operators_changed=False,
        hardware_or_metric_measurement_claimed=False, accuracy_accepted=False)
    return initialized, report


def install_mount_and_optics(state):
    if getattr(state,'source_actors',None) is not None or hasattr(state,'installation_optics'):
        raise ValueError('Mount chart requires separate original graph before workers and optics')
    begin = len(state.q)
    original_levers = {name:np.asarray(next(b for b in state.blocks if b.bundle.calibration.installation_id == name).bundle.calibration.lever_xyz)*SCENE_UNIT_M for name in state.ids}
    state.q = np.r_[state.q,np.zeros(9)]
    install_all_installation_optics(state)
    charts = {name:MountOpticsChart(chart.installation_id,chart.indices,chart.original_parameter_count,
        chart.physical_steps,tuple(range(begin+3*i,begin+3*i+3)))
        for i,(name,chart) in enumerate(state.installation_optics.items())}
    replacements = {}
    for old in state.blocks:
        new = MountOpticsBlock.__new__(MountOpticsBlock)
        new.__dict__.update(vars(old))
        new.optics_chart = charts[old.bundle.calibration.installation_id]
        new.mount_begin = begin
        new.actor_output_capacity = dict(old.actor_output_capacity,
            nnz=old.actor_output_capacity['nnz']+3*old.bundle.source_residual_rows)
        replacements[id(old)] = new
    state.blocks = [replacements[id(b)] for b in state.blocks]
    state.bypts = {p:replacements[id(b)] for p,b in state.bypts.items()}
    state.installation_optics = charts
    patterns = MountOpticsPatterns.__new__(MountOpticsPatterns)
    patterns.__dict__.update(vars(state.patterns))
    patterns.installation_optics = charts
    state.patterns = patterns
    state.mount_parameter_indices = np.arange(begin,begin+9)
    state.mount_contract = dict(schema='rinkspace.shared-constant-nonzero-installation-mount-candidate.v1',
        parameter_indices=state.mount_parameter_indices.tolist(),parameter_units='metre increments from authenticated base camera-local lever',
        original_parameter_count=begin,installation_order=list(state.ids),
        original_levers_m={name:lever.tolist() for name,lever in original_levers.items()},
        parameter_indices_by_installation={name:list(c.mount_indices) for name,c in charts.items()},
        one_lever_shared_over_all_cuts=True,zoom_dependence=False,per_frame_translation=False,
        prior_hardware_working_search_ranges_not_measurements=True,
        additional_prior_rows=0,original_source_pixels_sigmas_loss_roles_and_held_unchanged=True,
        extra_physical_shared_parameters=9,measured_lever_claim=False,accuracy_accepted=False)
    state.optics_contract = dict(state.optics_contract, mount_contract=state.mount_contract)
    return state.mount_contract
