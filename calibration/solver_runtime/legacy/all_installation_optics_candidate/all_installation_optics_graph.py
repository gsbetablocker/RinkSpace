"""Nine constant installation lens increments on the original complete source.

Old 29 shared indices, q11 geometry, XYZ, nonzero levers, source rows and
weights remain intact.  Installation is explicit and happens before workers.
No accepted snapshot, kernel, source definition, or production default is patched.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix, vstack
from secondary_optics_candidate.secondary_optics_graph import (
    SecondaryOpticsChart, SecondaryOpticsFixedXYZRinkBlock,
    OpticsPatternFactor, PHYSICAL_COLUMN_SCALES, _steps)
from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock
from z190_joint_shared_board_pattern_adapter import JointSharedBoardPatterns
from z190_shared_board_pattern_source import fixed_pivot_jacobian
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_physical_camera import ViewState


@dataclass(frozen=True)
class InstallationOpticsChart(SecondaryOpticsChart):
    def delta(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.original_parameter_count+9,) or not np.isfinite(q).all():
            raise ValueError('Complete original finite state plus nine constant installation lens coordinates required')
        return q[np.asarray(self.indices)]*np.asarray(PHYSICAL_COLUMN_SCALES)

    def manifest(self, base):
        result = super().manifest(base)
        result.update(schema='rinkspace.installation-shared-optics-candidate.v1',
            parameter_names=[self.installation_id+':delta_'+name for name in
                             ('PPx_native_px', 'PPy_native_px', 'k1_div_0.001')],
            total_new_shared_parameters=9)
        return result


def _lens_signature(calibration):
    # Pivot values may differ in historical block initializations; actual XYZ
    # is already one shared global triplet.  Compare the physical lens/lever
    # function, not an irrelevant constant-lens q_reference or pivot seed.
    return tuple(getattr(calibration, name) for name in (
        'installation_id', 'principal_x_fraction', 'principal_y_fraction', 'k1',
        'lever_xyz', 'output_mode', 'aspect_fx_over_fy', 'k2',
        'tangential_p1_p2', 'axial', 'allow_effective_pp_offset'))


def append_installation_optics_csr(matrix, optics, indices, original_parameter_count):
    """Copy every old CSR entry, including duplicates, before three lens columns.

    Native original scatter sorts but does not coalesce repeated view columns.
    The original normal accumulator owns that later canonicalization.  Adding
    lens columns must preserve those entries and their arithmetic order.
    """
    matrix = matrix.tocsr(copy=False)
    optics = np.asarray(optics, float)
    slots = np.asarray(indices, dtype=matrix.indices.dtype)
    old_count = int(original_parameter_count)
    if (matrix.shape[1] != old_count+9 or slots.shape != (3,) or
            not np.array_equal(slots, np.arange(slots[0], slots[0]+3)) or
            slots[0] < old_count or slots[-1] >= old_count+9 or
            (matrix.nnz and int(matrix.indices.max()) >= old_count)):
        raise ValueError('Original CSR and explicit three-of-nine optics span required: '+
            str(dict(shape=matrix.shape, indices=slots.tolist(), original_count=old_count,
                old_max=None if not matrix.nnz else int(matrix.indices.max()))))
    if (not matrix.has_sorted_indices or optics.ndim != 2 or optics.shape[1] != 3 or
            len(optics) > matrix.shape[0] or not np.isfinite(optics).all()):
        raise ValueError('Original sorted CSR and finite three-column source prefix required: '+
            str(dict(shape=matrix.shape, sorted=matrix.has_sorted_indices,
                canonical=matrix.has_canonical_format, optics_shape=optics.shape)))
    rows = matrix.shape[0]
    active = optics != 0.
    counts = np.zeros(rows, dtype=matrix.indptr.dtype)
    counts[:len(optics)] = active.sum(axis=1)
    pointer = matrix.indptr + np.r_[0, np.cumsum(counts, dtype=matrix.indptr.dtype)]
    data = np.empty(int(pointer[-1]), dtype=matrix.data.dtype)
    columns = np.empty(len(data), dtype=matrix.indices.dtype)
    for lo in range(0, rows, 65536):
        hi = min(lo + 65536, rows)
        left, right = map(int, matrix.indptr[[lo, hi]])
        old_counts = np.diff(matrix.indptr[lo:hi + 1])
        row_ids = np.repeat(np.arange(lo, hi, dtype=matrix.indptr.dtype), old_counts)
        inner = np.arange(right - left, dtype=matrix.indptr.dtype)
        inner -= np.repeat(matrix.indptr[lo:hi] - left, old_counts)
        destination = pointer[row_ids] + inner
        data[destination] = matrix.data[left:right]
        columns[destination] = matrix.indices[left:right]
        stop = min(hi, len(optics))
        if stop > lo:
            rr, cc = np.nonzero(active[lo:stop])
            if len(rr):
                old_counts = np.diff(matrix.indptr[lo:stop + 1])
                inner_optics = np.cumsum(active[lo:stop], axis=1) - 1
                destination = pointer[lo + rr] + old_counts[rr] + inner_optics[rr, cc]
                data[destination] = optics[lo + rr, cc]
                columns[destination] = slots[cc]
    result = csr_matrix((data, columns, pointer), shape=matrix.shape, copy=False)
    result.has_sorted_indices = matrix.has_sorted_indices
    result.has_canonical_format = matrix.has_canonical_format
    return result


class AllInstallationOpticsBlock(SecondaryOpticsFixedXYZRinkBlock):
    def evaluate(self, q, jacobian=True):
        # Bypass only the secondary-specific last-three CSR routine.  All
        # original geometry/source calculations remain in the old base class.
        result = FixedXYZRinkBlock.evaluate(self, q, jacobian)
        if not jacobian:
            return result
        residual, matrix = result
        local = self.local(q)
        calibration = self.bundle.current_calibration(local)
        optics = np.zeros((int(self.bundle.source_residual_rows), 3))
        motion = np.asarray(self.motion_columns_provider(self.bundle, local, calibration,
            self.bundle.profile_weights, steps=self.optics_chart.physical_steps,
            physical_column_scales=PHYSICAL_COLUMN_SCALES), float)
        if motion.shape != (self.bundle.motion_rows, 3):
            raise ValueError('Original complete native motion lens column inventory changed')
        np.divide(motion, self.bundle.motion_sigma[:, None], out=optics[:self.bundle.motion_rows])
        cursor = int(self.bundle.motion_rows)
        geometry = self.geometry_values(q)
        for p, old_rows in self.groups:
            moved = [self.geometry.rebind(row, geometry) for row in old_rows]
            _, raw = self.absolute(self.bundle.camera(local, p), moved, True)
            count = sum(2*len(row['uv']) for row in old_rows)
            raw = np.asarray(raw, float)
            if raw.shape != (count, 15):
                raise ValueError('Original normalized state15 absolute lens Jacobian required')
            optics[cursor:cursor+count] = raw[:, [8, 9, 11]]*.001
            cursor += count
        if cursor != len(optics):
            raise ValueError('Original complete absolute source inventory changed')
        return residual, append_installation_optics_csr(matrix, optics,
            self.optics_chart.indices, self.optics_chart.original_parameter_count)


@contextmanager
def bind_external_main_optics(state, q):
    """Bind every cached main lens field together, and restore on any exit.

    The old external evaluate/shared_points/guard/held path reads pp and k1
    directly, in addition to calibration.  Only updating calibration would
    miss inverse anchor rays and their depth coupling.  No lever or source is
    changed.  Nested uses are safe; concurrent mutation of one external owner
    is intentionally rejected by the caller's original sequential protocol.
    """
    external = state.external
    if external is None:
        yield None
        return
    chart = state.installation_optics[state.ids[0]]
    base = state.external_optics_base_calibration
    current = chart.calibration(base, q)
    previous = external.calibration, external.pp, external.k1
    external.calibration = current
    external.pp = np.array([1920*current.principal_x_fraction[0],
                            1080*current.principal_y_fraction[0]])
    external.k1 = float(current.k1[0])
    try:
        yield external
    finally:
        external.calibration, external.pp, external.k1 = previous


def evaluate_external_with_optics(state, q, *, jacobian=True, held=False,
                                  local_override=None, evaluator=None):
    """Original full external source plus its complete three main-lens columns.

    Each finite difference replays the original family including inverse
    anchor ray -> shared positive optical depth -> unknown world point ->
    forward target projection, true top branch, heads and actual Ad14 corners.
    Explicit local R/logf/logdepth coordinates are fixed for a lens derivative.
    Existing local/logdepth/XYZ columns are evaluated under current lens.
    """
    if evaluator is None:
        from z190_joint_intercamera_fixed_xyz_source import evaluate_external_state
        evaluator = evaluate_external_state
    def original(qq, with_jacobian):
        with bind_external_main_optics(state, qq):
            return evaluator(state, qq, jacobian=with_jacobian, held=held,
                             local_override=local_override)
    result = original(q, jacobian)
    if not jacobian:
        return result
    chart = state.installation_optics[state.ids[0]]
    optics = np.empty((len(result.residual), 3))
    ids = [row['id'] for row in result.source_row_ledger]
    for col, (index, width, scale) in enumerate(zip(chart.indices,
            chart.physical_steps, PHYSICAL_COLUMN_SCALES)):
        plus, minus = np.asarray(q, float).copy(), np.asarray(q, float).copy()
        plus[index] += width/scale
        minus[index] -= width/scale
        rp, rm = original(plus, False), original(minus, False)
        if (rp.residual.shape != result.residual.shape or rm.residual.shape != result.residual.shape or
                [row['id'] for row in rp.source_row_ledger] != ids or
                [row['id'] for row in rm.source_row_ledger] != ids):
            raise ValueError('Main lens perturbation changed original external source inventory')
        optics[:, col] = (rp.residual-rm.residual)*(scale/(2*width))
    return SimpleNamespace(**vars(result), optics_jacobian=optics,
        optics_parameter_indices=np.asarray(chart.indices, int),
        optics_inverse_anchor_forward_and_logdepth_chain_included=True)


class AllInstallationOpticsPatterns(JointSharedBoardPatterns):
    @classmethod
    def from_owner(cls, owner, charts):
        result = cls.__new__(cls)
        result.__dict__.update(vars(owner))
        result.installation_optics = dict(charts)
        steps = next(iter(charts.values())).physical_steps
        result.factor = OpticsPatternFactor(owner.factor, steps)
        result.external_factor = None if owner.external_factor is None else OpticsPatternFactor(owner.external_factor, steps)
        result.manifest = dict(owner.manifest,
            installation_optics_columns={name:list(chart.indices) for name,chart in charts.items()},
            optics_profile_chain='exact original complete native pattern residual central derivative',
            optics_derivative_physical_steps=list(steps))
        return result

    def chart_for_frame(self, frame):
        if frame['observer_state_owner'] == 'programme':
            block = self.state.bypts[int(frame['source_actual_pts'])]
            return self.installation_optics[block.bundle.calibration.installation_id]
        if frame['observer_state_owner'] == 'external':
            return self.installation_optics[self.state.ids[0]]
        raise ValueError('Original explicit pattern observer installation required')

    def camera(self, q, frame, *, held=False):
        if frame['observer_state_owner'] == 'programme':
            return super().camera(q, frame, held=held)
        state = self.state
        p = int(frame['source_actual_pts'])
        eq = q[state.external_start:state.external_stop]
        if held:
            with bind_external_main_optics(state, q):
                eq, _ = state.external.infer_held_view_from_neighbours(eq)
        index = int(state.external.index[p])
        view = eq[4*index:4*index+4]
        chart = self.chart_for_frame(frame)
        cal = chart.calibration(state.external_optics_base_calibration, q)
        cal = replace(cal, pivot_xyz=tuple(q[state.slots[state.ids[0]]]/SCENE_UNIT_M))
        camera = cal.camera25_and_jacobians(ViewState(tuple(view[:3]), float(view[3]-np.log(1920.))))[0]
        return camera, state.slots[state.ids[0]], state.external_start+4*index, np.asarray(cal.lever_xyz)*SCENE_UNIT_M

    def evaluate(self, q, jacobian=True, *, held=False):
        q9 = q[self.start:self.start+9]
        self.geometry.guard(q9)
        values, matrices, reports = [], [], []
        from z190_parallel_pattern_evaluation import take_pattern_frames, submit_pattern_frames
        pending = submit_pattern_frames(self,q,jacobian,held=True) if held else take_pattern_frames(self,q,jacobian)
        for ordinal, frame in enumerate(self.held if held else self.fit):
            if pending is None:
                cam, slots, view, lever = self.camera(q,frame,held=held)
                result = self.frame_factor(frame).evaluate(q9,cam,frame,with_jacobian=jacobian)
            else:
                _, (cam,slots,view,lever), future = pending[ordinal]
                result = future.result()
            values.append(result.residual)
            reports.append(dict(physical_ad_id=frame['physical_ad_id'],
                source_actual_pts=frame['source_actual_pts'], source_native_pts=frame['source_native_pts'],
                split=frame['split'], observer_state_owner=frame['observer_state_owner'],
                source_reports=result.source_reports, source_pixels_roles_sigma_unchanged=True))
            if jacobian:
                jxyz, jview = fixed_pivot_jacobian(result.camera_state11_jacobian,cam,lever,SCENE_UNIT_M)
                matrix = np.zeros((len(result.residual),len(q)))
                matrix[:,slots] = jxyz
                matrix[:,view:view+4] = jview
                matrix[:,self.start:self.start+9] = result.shared_q9_jacobian
                matrix[:,self.chart_for_frame(frame).indices] = result.optics_jacobian
                matrices.append(csr_matrix(matrix))
        residual = np.concatenate(values)
        if held:
            return SimpleNamespace(residual=residual, source_reports=reports)
        return (residual,vstack(matrices,format='csr')) if jacobian else residual


def calibration_for_global(block, q):
    return block.calibration_for_global(q) if hasattr(block,'calibration_for_global') else block.bundle.calibration


def optics_diagnostics(state, q):
    result = {}
    for name, chart in state.installation_optics.items():
        block = next(b for b in state.blocks if b.bundle.calibration.installation_id == name)
        current = block.bundle.current_calibration(block.local(q))
        result[name] = dict(parameter_indices=list(chart.indices), delta_physical=chart.delta(q).tolist(),
            principal_native_px=[1920*current.principal_x_fraction[0],1080*current.principal_y_fraction[0]],
            k1=current.k1[0],current_calibration=current.manifest())
    return result


def install_all_installation_optics(state, *, motion_columns_provider=None,
                                    physical_steps=(1e-4,1e-4,1e-6)):
    if hasattr(state,'installation_optics') or hasattr(state,'secondary_optics'):
        raise ValueError('Install nine lens coordinates on a separately loaded original 29-shared graph')
    if getattr(state,'source_actors',None) is not None:
        raise ValueError('Install all lens wrappers before constructing source actors')
    if len(state.ids)!=3 or len(state.geometry_parameter_indices)!=11 or state.patterns is None:
        raise ValueError('Original three installations, q11 geometry and nine shared material coordinates required')
    old_shared = np.r_[np.arange(14),np.arange(state.patterns.start,state.patterns.start+9),state.geometry_parameter_indices[5:]]
    if len(set(old_shared.tolist())) != 29:
        raise ValueError('All original 29 shared parameter indices are required')
    if motion_columns_provider is None:
        from secondary_optics_candidate.z190_motion_optics_extension import motion_optics_columns
        motion_columns_provider = motion_optics_columns
    steps = _steps(physical_steps)
    old_q = np.asarray(state.q,float)
    count = len(old_q)
    q = np.r_[old_q,np.zeros(9)]
    charts, bases, signatures = {}, {}, {}
    for ordinal,name in enumerate(state.ids):
        members = [b for b in state.blocks if b.bundle.calibration.installation_id == name]
        if not members:
            raise ValueError('Every physical installation must retain its complete original blocks')
        base = members[0].bundle.calibration
        signature = _lens_signature(base)
        if any(_lens_signature(b.bundle.calibration) != signature for b in members):
            raise ValueError('Same-installation PP/k1/lever/output mapping differs across original source blocks: '+name)
        chart = InstallationOpticsChart(str(name),tuple(range(count+3*ordinal,count+3*ordinal+3)),count,steps)
        chart.calibration(base,q)
        charts[name],bases[name],signatures[name] = chart,base,signature
    if any(b.bundle.calibration.installation_id not in charts for b in state.blocks):
        raise ValueError('An original source block has an unknown physical lens owner')
    external_base = None if state.external is None else state.external.calibration
    if external_base is not None and _lens_signature(external_base) != signatures[state.ids[0]]:
        raise ValueError('External original observer lens/lever/output mapping differs from its physical main installation')
    replacements = {id(b):AllInstallationOpticsBlock.from_block(b,charts[b.bundle.calibration.installation_id],motion_columns_provider) for b in state.blocks}
    state.blocks = [replacements[id(b)] for b in state.blocks]
    state.bypts = {p:replacements[id(b)] for p,b in state.bypts.items()}
    state.q = q
    state.installation_optics = charts
    state.optics_parameter_indices = np.arange(count,count+9,dtype=int)
    state.external_optics_base_calibration = external_base
    # The external object is private to this separately loaded candidate state.
    # Clone its owner so lens binding cannot mutate another state's object.
    if state.external is not None:
        state.external = copy.copy(state.external)
    state.optics_contract = dict(schema='rinkspace.all-installation-shared-optics-candidate.v1',
        parameter_indices=state.optics_parameter_indices.tolist(), original_parameter_count=count,
        installation_order=list(state.ids),
        installations={name:charts[name].manifest(bases[name]) for name in state.ids},
        external_observer_installation=state.ids[0],
        external_base_calibration=None if external_base is None else external_base.manifest(),
        all_29_shared_and_q11_preserved=True,source_rows_sigma_roles_and_held_unchanged=True,
        physical_numeric_survey_claim=False,unaccepted_candidate=True)
    state.patterns = AllInstallationOpticsPatterns.from_owner(state.patterns,charts)
    return state.optics_contract
