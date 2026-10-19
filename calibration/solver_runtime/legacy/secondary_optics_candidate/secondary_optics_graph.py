"""Three installation-wide lens increments on the unchanged complete source graph.

This module owns no source admission, objective, prior, or solve loop.  It is
installed explicitly on a separately loaded candidate state before actors start.
All old parameter indices remain unchanged.  The last three coordinates are
native-pixel PPx, native-pixel PPy, and k1/0.001 increments from the authenticated
seed calibration.  It never patches classes in the accepted solver snapshot.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix, vstack
from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock
from z190_joint_shared_board_pattern_adapter import JointSharedBoardPatterns
from z190_shared_board_pattern_source import fixed_pivot_jacobian
from z190_continuous_source_bundle import SCENE_UNIT_M


PHYSICAL_COLUMN_SCALES = (1., 1., .001)


def _steps(value):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Three positive finite native physical optics FD widths required')
    return tuple(float(x) for x in result)


@dataclass(frozen=True)
class SecondaryOpticsChart:
    installation_id: str
    indices: tuple[int, int, int]
    original_parameter_count: int
    physical_steps: tuple[float, float, float] = (1e-4, 1e-4, 1e-6)

    def delta(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.original_parameter_count+3,) or not np.isfinite(q).all():
            raise ValueError('Complete original finite state plus exactly three shared optics coordinates required')
        return q[np.asarray(self.indices)] * np.asarray(PHYSICAL_COLUMN_SCALES)

    def calibration(self, base, q):
        if base.installation_id != self.installation_id:
            raise ValueError('Lens increment must belong to its original physical installation')
        dx, dy, dk = self.delta(q)
        # This moving-source bundle currently interprets PP in native 1920x1080
        # coordinates.  Reject an incompatible output mapping instead of
        # applying A a second time or silently changing its projection model.
        if (tuple(base.output_mode.domain_wh) != (1920, 1080) or
                tuple(base.output_mode.final_wh) != (1920, 1080) or
                not np.array_equal(base.output_mode.A, np.eye(3))):
            raise ValueError('Candidate moving-source optics require the original native identity output mapping')
        if any(len(getattr(base, key)) != 1 for key in
               ('principal_x_fraction', 'principal_y_fraction', 'k1')):
            raise ValueError('This candidate adds constant installation optics, not a zoom polynomial')
        # Keep zero-delta calibration byte-for-byte at the old seed.
        if dx == 0. and dy == 0. and dk == 0.:
            return base
        return replace(base,
            principal_x_fraction=(base.principal_x_fraction[0]+dx/1920.,),
            principal_y_fraction=(base.principal_y_fraction[0]+dy/1080.,),
            k1=(base.k1[0]+dk,))

    def manifest(self, base):
        return dict(schema='rinkspace.secondary-shared-optics-candidate.v1',
            installation_id=self.installation_id,
            parameter_indices=list(self.indices),
            parameter_names=['secondary_delta_PPx_native_px',
                             'secondary_delta_PPy_native_px', 'secondary_delta_k1_div_0.001'],
            physical_column_scales=list(PHYSICAL_COLUMN_SCALES),
            physical_derivative_steps=list(self.physical_steps),
            state_derivative_steps=(np.asarray(self.physical_steps)/PHYSICAL_COLUMN_SCALES).tolist(),
            original_parameter_count=self.original_parameter_count,
            base_calibration=base.manifest(),
            zero_delta_preserves_seed=True,installation_shared_not_per_frame=True,
            source_rows_sigma_roles_and_held_unchanged=True,
            physical_numeric_survey_claim=False,unaccepted_candidate=True)


def append_shared_optics_csr(matrix, optics, indices):
    """Append three tail columns without a rows-by-global-columns dense array.

    Existing row data and column order are copied exactly.  Tail values use
    only the original source prefix; roll priors acquire no new entries.
    Chunked copying bounds transient index allocations.  Explicit zero optics
    entries are omitted, while original CSR storage remains unchanged.
    """
    matrix = matrix.tocsr(copy=False)
    optics = np.asarray(optics, float)
    slots = np.asarray(indices, dtype=matrix.indices.dtype)
    if (optics.ndim != 2 or optics.shape[1] != 3 or optics.shape[0] > matrix.shape[0]
            or not np.isfinite(optics).all() or slots.shape != (3,)
            or not np.array_equal(slots, np.arange(matrix.shape[1]-3, matrix.shape[1]))):
        raise ValueError('Finite original source prefix and three explicit tail optics columns required')
    if (not matrix.has_sorted_indices or not matrix.has_canonical_format or
            (matrix.nnz and int(matrix.indices.max()) >= int(slots[0]))):
        raise ValueError('Old canonical CSR must contain only original parameter columns')
    rows = matrix.shape[0]
    active = optics != 0.
    counts = np.zeros(rows, dtype=matrix.indptr.dtype)
    counts[:len(optics)] = active.sum(axis=1)
    pointer = matrix.indptr + np.r_[0, np.cumsum(counts, dtype=matrix.indptr.dtype)]
    data = np.empty(int(pointer[-1]), dtype=matrix.data.dtype)
    columns = np.empty(len(data), dtype=matrix.indices.dtype)
    for lo in range(0, rows, 65536):
        hi = min(lo+65536, rows)
        left, right = map(int, matrix.indptr[[lo, hi]])
        old_counts = np.diff(matrix.indptr[lo:hi+1])
        row_ids = np.repeat(np.arange(lo, hi, dtype=matrix.indptr.dtype), old_counts)
        inner = np.arange(right-left, dtype=matrix.indptr.dtype)
        inner -= np.repeat(matrix.indptr[lo:hi]-left, old_counts)
        destination = pointer[row_ids]+inner
        data[destination] = matrix.data[left:right]
        columns[destination] = matrix.indices[left:right]
        stop = min(hi, len(optics))
        if stop > lo:
            rr, cc = np.nonzero(active[lo:stop])
            if len(rr):
                local_counts = np.diff(matrix.indptr[lo:stop+1])
                inner_opt = np.cumsum(active[lo:stop], axis=1)-1
                position = pointer[lo+rr]+local_counts[rr]+inner_opt[rr, cc]
                data[position] = optics[lo+rr, cc]
                columns[position] = slots[cc]
    result = csr_matrix((data, columns, pointer), shape=matrix.shape)
    result.has_sorted_indices = True
    result.has_canonical_format = True
    return result


class SecondaryOpticsFixedXYZRinkBlock(FixedXYZRinkBlock):
    @classmethod
    def from_block(cls, block, chart, provider):
        result = cls.__new__(cls)
        result.__dict__.update(vars(block))
        # Preserve large immutable source-array identity during actor pickle.
        # No reference to the full joint state or old wrapper is retained.
        result.bundle = copy.copy(block.bundle)
        if block.prepared is not None:
            result.prepared = copy.copy(block.prepared)
            result.prepared.bundle = result.bundle
        result.optics_chart = chart
        result.optics_base_calibration = block.bundle.calibration
        result.motion_columns_provider = provider
        absolute_rows = int(result.bundle.source_residual_rows-result.bundle.motion_rows)
        old_capacity = (int(result.bundle.pointer[-1])+3*len(result.bundle.stamps)
            -int(result.bundle.source_residual_rows)
            +absolute_rows*result.geometry_parameter_count)
        result.actor_output_capacity = dict(rows=len(result.quadratic),
            motion_rows=int(result.bundle.motion_rows),
            nnz=old_capacity+3*int(result.bundle.source_residual_rows))
        return result

    def calibration_for_global(self, q):
        return self.optics_chart.calibration(self.optics_base_calibration, q)

    def local(self, q):
        # Actor profile_motion_residual receives only local q; bind its lens
        # here so profile updates, finite guards, reviews and export agree.
        self.bundle.calibration = self.calibration_for_global(q)
        return super().local(q)

    def evaluate(self, q, jacobian=True):
        result = super().evaluate(q, jacobian)
        if not jacobian:
            return result
        residual, matrix = result
        local = self.local(q)
        cal = self.bundle.current_calibration(local)
        rows = int(self.bundle.source_residual_rows)
        optics = np.zeros((rows, 3))
        motion = np.asarray(self.motion_columns_provider(self.bundle, local, cal,
            self.bundle.profile_weights, steps=self.optics_chart.physical_steps,
            physical_column_scales=PHYSICAL_COLUMN_SCALES), float)
        if motion.shape != (self.bundle.motion_rows, 3):
            raise ValueError('Original native motion optics column inventory changed')
        np.divide(motion, self.bundle.motion_sigma[:, None], out=optics[:self.bundle.motion_rows])
        cursor = int(self.bundle.motion_rows)
        geometry = self.geometry_values(q)
        for p, old_rows in self.groups:
            moved = [self.geometry.rebind(row, geometry) for row in old_rows]
            _, raw = self.absolute(self.bundle.camera(local, p), moved, True)
            count = sum(2*len(row['uv']) for row in old_rows)
            raw = np.asarray(raw, float)
            if raw.shape != (count, 15):
                raise ValueError('Original normalized camera-state15 absolute derivative required')
            # State15 PP coordinates are PP_native/1000; Brown k1 is raw.
            # Evaluator already divides by the original absolute sigma.
            optics[cursor:cursor+count] = raw[:, [8, 9, 11]]*.001
            cursor += count
        if cursor != rows:
            raise ValueError('Original complete absolute source inventory changed')
        return residual, append_shared_optics_csr(matrix, optics, self.optics_chart.indices)


class OpticsPatternFactor:
    """Original complete profile factor plus three independently checked columns."""
    def __init__(self, factor, steps):
        self.factor = factor
        self.steps = _steps(steps)

    def evaluate(self, q9, camera25, frame, with_jacobian=True):
        original = self.factor.evaluate(q9, camera25, frame, with_jacobian=with_jacobian)
        if not with_jacobian:
            return original
        camera = np.asarray(camera25, float)
        optics = np.empty((len(original.residual), 3))
        for col, (physical, step, scale) in enumerate(zip((8, 9, 11), self.steps, PHYSICAL_COLUMN_SCALES)):
            plus, minus = camera.copy(), camera.copy()
            plus[physical] += step
            minus[physical] -= step
            rp = self.factor.evaluate(q9, plus, frame, with_jacobian=False).residual
            rm = self.factor.evaluate(q9, minus, frame, with_jacobian=False).residual
            if rp.shape != original.residual.shape or rm.shape != original.residual.shape:
                raise ValueError('Pattern optics perturbation changed original source rows')
            optics[:, col] = (rp-rm)*(scale/(2*step))
        return SimpleNamespace(**vars(original), optics_jacobian=optics)


class SecondaryOpticsPatterns(JointSharedBoardPatterns):
    @classmethod
    def from_owner(cls, owner, chart):
        result = cls.__new__(cls)
        result.__dict__.update(vars(owner))
        result.optics_chart = chart
        result.factor = OpticsPatternFactor(owner.factor, chart.physical_steps)
        # External observer is the main installation; no secondary lens column.
        result.external_factor = owner.external_factor
        result.manifest = dict(owner.manifest, secondary_optics_columns=list(chart.indices),
            optics_profile_chain='exact original complete native pattern residual central derivative',
            optics_derivative_physical_steps=list(chart.physical_steps))
        return result

    def evaluate(self, q, jacobian=True, *, held=False):
        q9 = q[self.start:self.start+9]
        self.geometry.guard(q9)
        values, matrices, reports = [], [], []
        from z190_parallel_pattern_evaluation import take_pattern_frames, submit_pattern_frames
        pending = submit_pattern_frames(self, q, jacobian, held=True) if held else take_pattern_frames(self, q, jacobian)
        for ordinal, frame in enumerate(self.held if held else self.fit):
            if pending is None:
                cam, slots, view, lever = self.camera(q, frame, held=held)
                result = self.frame_factor(frame).evaluate(q9, cam, frame, with_jacobian=jacobian)
            else:
                _, (cam, slots, view, lever), future = pending[ordinal]
                result = future.result()
            values.append(result.residual)
            reports.append(dict(physical_ad_id=frame['physical_ad_id'],
                source_actual_pts=frame['source_actual_pts'], source_native_pts=frame['source_native_pts'],
                split=frame['split'], observer_state_owner=frame['observer_state_owner'],
                source_reports=result.source_reports, source_pixels_roles_sigma_unchanged=True))
            if jacobian:
                jxyz, jview = fixed_pivot_jacobian(result.camera_state11_jacobian, cam, lever, SCENE_UNIT_M)
                matrix = np.zeros((len(result.residual), len(q)))
                matrix[:, slots] = jxyz
                matrix[:, view:view+4] = jview
                matrix[:, self.start:self.start+9] = result.shared_q9_jacobian
                if frame['observer_state_owner'] == 'programme':
                    block = self.state.bypts[int(frame['source_actual_pts'])]
                    if block.bundle.calibration.installation_id != self.optics_chart.installation_id:
                        raise ValueError('Pattern lens columns must belong to the secondary installation')
                    matrix[:, self.optics_chart.indices] = result.optics_jacobian
                matrices.append(csr_matrix(matrix))
        residual = np.concatenate(values)
        if held:
            return SimpleNamespace(residual=residual, source_reports=reports)
        return (residual, vstack(matrices, format='csr')) if jacobian else residual


def calibration_for_global(block, q):
    """Current immutable lens for direct factors; does not rely on last evaluation."""
    if hasattr(block, 'calibration_for_global'):
        return block.calibration_for_global(q)
    return block.bundle.calibration


def optics_diagnostics(state, q):
    chart = state.secondary_optics
    block = next(b for b in state.blocks if isinstance(b, SecondaryOpticsFixedXYZRinkBlock))
    cal = block.bundle.current_calibration(block.local(q))
    return dict(installation_id=chart.installation_id, parameter_indices=list(chart.indices),
        delta_physical=chart.delta(q).tolist(),
        principal_native_px=[1920*cal.principal_x_fraction[0], 1080*cal.principal_y_fraction[0]],
        k1=cal.k1[0], current_calibration=cal.manifest(),
        source_inventory_and_nonzero_lever_preserved=True)


def install_secondary_optics(state, *, installation_id=None, motion_columns_provider=None,
                             physical_steps=(1e-4, 1e-4, 1e-6)):
    """Opt in once, before persistent actors or pattern pools are constructed."""
    if hasattr(state, 'secondary_optics'):
        raise ValueError('Secondary optics candidate is already installed')
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Install candidate optics before constructing source actors')
    if len(state.ids) != 3 or len(state.geometry_parameter_indices) != 11 or state.patterns is None:
        raise ValueError('Candidate requires the complete original three-installation q11 and nine-pattern source graph')
    old_indices = np.r_[np.arange(14), np.arange(state.patterns.start, state.patterns.start+9),
                        state.geometry_parameter_indices[5:]]
    if len(set(old_indices.tolist())) != 29:
        raise ValueError('All 29 original explicit shared parameter indices are required')
    if installation_id is None:
        installation_id = state.ids[1]
    selected = [b for b in state.blocks if b.bundle.calibration.installation_id == installation_id]
    if not selected or not any(b.name == 'secondary' for b in selected):
        raise ValueError('Explicit secondary physical installation and complete source block required')
    base = selected[0].bundle.calibration
    if any(b.bundle.calibration != base for b in selected):
        raise ValueError('The same installation must begin with one original shared calibration')
    if motion_columns_provider is None:
        from secondary_optics_candidate.z190_motion_optics_extension import motion_optics_columns
        motion_columns_provider = motion_optics_columns
    old_q = np.asarray(state.q, float)
    count = len(old_q)
    chart = SecondaryOpticsChart(str(installation_id), (count, count+1, count+2), count, _steps(physical_steps))
    candidate_q = np.r_[old_q, np.zeros(3)]
    chart.calibration(base, candidate_q)  # Validate output mapping and constant lens before mutation.
    replacements = {id(b): SecondaryOpticsFixedXYZRinkBlock.from_block(b, chart, motion_columns_provider) for b in selected}
    state.blocks = [replacements.get(id(b), b) for b in state.blocks]
    state.bypts = {p: replacements.get(id(b), b) for p, b in state.bypts.items()}
    state.q = candidate_q
    state.optics_parameter_indices = np.asarray(chart.indices, dtype=int)
    state.secondary_optics = chart
    state.optics_contract = chart.manifest(base)
    state.patterns = SecondaryOpticsPatterns.from_owner(state.patterns, chart)
    return state.optics_contract
