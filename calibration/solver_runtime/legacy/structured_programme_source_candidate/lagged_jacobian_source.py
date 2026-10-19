"""Reuse one RAM Jacobian as a modified-Newton derivative model.

Current residuals, profiles, Huber weights and every nonlinear trial cost are
evaluated from the admitted source. The retained J is explicitly approximate
away from its anchor; it is never represented as an exact current Jacobian.
"""
from __future__ import annotations

import hashlib
import numpy as np
from z190_streaming_source_normal import (
    SourceRowBlock, FrozenNativeReplayCache, _checked_block, _check_replay,
    _finish_replay,
)


def _q_digest(q):
    value = np.ascontiguousarray(q, dtype=np.float64)
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


class LaggedZeroJacobianActivation(RuntimeError):
    """An original zero-residual/zero-J source row became active at this q."""
    def __init__(self, report):
        self.report = dict(report)
        super().__init__('Original zero-J source support became active; exact same-q/profile J required')


def _zero_residual_zero_numeric_J_rows(residual, matrix):
    """Exact numeric zeros, including explicit CSR zero entries; bounded scratch."""
    matrix = matrix.tocsr(copy=False)
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    zero = np.asarray(residual) == 0.
    for lo in range(0, len(zero), 65536):
        hi = min(lo + 65536, len(zero))
        if not zero[lo:hi].any():
            continue
        first, last = map(int, matrix.indptr[[lo, hi]])
        prefix = np.empty(last - first + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(matrix.data[first:last] != 0., dtype=np.int64, out=prefix[1:])
        offsets = matrix.indptr[lo:hi + 1] - first
        zero[lo:hi] &= prefix[offsets[1:]] == prefix[offsets[:-1]]
    return np.flatnonzero(zero)


class LaggedJacobianSource:
    """Single retained J, refreshed when its real trial prediction is poor."""
    def __init__(self, refresh_reduction_ratio=0.):
        self.refresh_reduction_ratio = float(refresh_reduction_ratio)
        if not np.isfinite(self.refresh_reduction_ratio) or self.refresh_reduction_ratio < 0:
            raise ValueError('Finite nonnegative optional derivative prediction refresh ratio required')
        self.contract = None
        self.jacobians = ()
        self.zero_support_rows = ()
        self.anchor_q = None
        self.anchor_profile_token = None
        self.age = 0
        self.force_fresh = True
        self.refresh_reason = 'initial exact source linearization'
        self.reusing = False
        self.current_report = {}

    def begin(self, q, profile_token):
        self.reusing = bool(self.jacobians) and not self.force_fresh
        if self.reusing:
            self.age += 1
        else:
            # Release the preceding J before generating its replacement.
            self.jacobians = ()
            self.zero_support_rows = ()
            self.contract = None
            self.anchor_q = _q_digest(q)
            self.anchor_profile_token = profile_token
            self.age = 0
        self.current_report = dict(
            schema='rinkspace.modified-newton-source-J-reuse.v1',
            derivative_model='lagged_J' if self.reusing else 'fresh_source_J',
            exact_current_J=not self.reusing,
            J_anchor_q_sha256=self.anchor_q,
            J_anchor_profile_token=self.anchor_profile_token,
            current_q_sha256=_q_digest(q), current_profile_token=profile_token,
            accepted_state_reuse_age=self.age,
            optional_prediction_refresh_ratio=self.refresh_reduction_ratio,
            refresh_reason=None if self.reusing else self.refresh_reason,
            current_residuals_and_Huber_H_g_recomputed=True,
            original_profile_update_and_full_nonlinear_trial_cost=True,
            source_rows_roles_sigma_masks_scales_and_parameters_unchanged=True,
            retained_J_copies=1, persistent_matrix_files=0,
            historical_exact_same_q_speedup_is_not_a_claim_for_this_mode=True,
        )
        return self.reusing

    def capture(self, cache):
        if not isinstance(cache, FrozenNativeReplayCache):
            raise ValueError('Modified Newton requires the generated source J replay cache')
        self.contract = cache.contract
        self.jacobians = tuple(block.J for block in cache.blocks)
        self.zero_support_rows = tuple(
            _zero_residual_zero_numeric_J_rows(block.r, block.J)
            for block in cache.blocks)
        self.current_report['retained_zero_support_row_indices_bytes'] = sum(
            rows.nbytes for rows in self.zero_support_rows)
        self.force_fresh = False
        self.current_report['retained_CSR_payload_bytes'] = sum(
            j.data.nbytes + j.indices.nbytes + j.indptr.nbytes
            for j in self.jacobians)

    def blocks(self, current_residual_factory, *, fresh_activated_block=None, fresh_activated_rows=None):
        if not self.reusing or self.contract is None:
            raise RuntimeError('No retained derivative model for this iteration')
        count = rows = 0
        for ordinal, supplied in enumerate(current_residual_factory()):
            r, _, mask, scales, signature = _checked_block(
                supplied, self.contract.parameter_count, False)
            _check_replay(signature, ordinal, self.contract, False)
            ledger = supplied.row_ledger if isinstance(supplied, SourceRowBlock) else supplied['row_ledger']
            candidates = self.zero_support_rows[ordinal]
            activated = candidates[r[candidates] != 0.]
            if len(activated):
                row_patch=fresh_activated_rows(supplied,activated)if fresh_activated_rows is not None else None
                if row_patch is not None:
                    from .activated_source_row_subset import SourceJacobianRowPatch,replace_csr_rows
                    if not isinstance(row_patch,SourceJacobianRowPatch):raise TypeError('Exact original source-row derivative patch required')
                    updated=replace_csr_rows(self.jacobians[ordinal],row_patch,activated,r,signature.inventory_sha256)
                    replaced=list(self.jacobians);replaced[ordinal]=updated;self.jacobians=tuple(replaced)
                    # Recheck exactly the refreshed rows; unchanged old zeros
                    # keep their original witness and future activation gate.
                    zeros=list(self.zero_support_rows)
                    keep=zeros[ordinal][~np.isin(zeros[ordinal],row_patch.row_indices)]
                    fresh_zero=row_patch.row_indices[_zero_residual_zero_numeric_J_rows(r[row_patch.row_indices],row_patch.matrix)]
                    zeros[ordinal]=np.sort(np.r_[keep,fresh_zero]);self.zero_support_rows=tuple(zeros)
                    self.current_report['retained_CSR_payload_bytes']=sum(
                        j.data.nbytes+j.indices.nbytes+j.indptr.nbytes for j in self.jacobians)
                    self.current_report['retained_zero_support_row_indices_bytes']=sum(
                        row.nbytes for row in self.zero_support_rows)
                    self.current_report.setdefault('exact_activated_source_row_derivative_refreshes',[]).append(dict(
                        block_ordinal=ordinal,family=ledger.get('family'),activated_local_rows=activated.tolist(),
                        refreshed_scalar_row_count=len(row_patch.row_indices),complete_family_rows=len(r),
                        complete_parameter_columns=updated.shape[1],
                        current_q_sha256=self.current_report['current_q_sha256'],
                        current_profile_token=self.current_report['current_profile_token'],
                        public_operator_detail=row_patch.detail,
                        original_current_all_residuals_masks_scales_and_inventory_unchanged=True,
                        untouched_rows_preserve_modified_Newton_CSR_bytes=True))
                    fresh=None
                else:
                    fresh = fresh_activated_block(supplied) if fresh_activated_block is not None else None
                if fresh is not None:
                    fresh_r, fresh_J, fresh_mask, fresh_scales, fresh_signature = _checked_block(
                        fresh, self.contract.parameter_count, True)
                    # Check original inventory; an exact current derivative has
                    # different r/J bytes from the retained anchor by design.
                    _check_replay(fresh_signature, ordinal, self.contract, False)
                    if (not np.array_equal(fresh_r, r) or not np.array_equal(fresh_mask, mask)
                            or not np.array_equal(fresh_scales, scales)):
                        raise ValueError('Activated family fresh derivative changed current residual/mask/native scales')
                    replaced = list(self.jacobians); replaced[ordinal] = fresh_J
                    self.jacobians = tuple(replaced)
                    zeros = list(self.zero_support_rows)
                    zeros[ordinal] = _zero_residual_zero_numeric_J_rows(fresh_r, fresh_J)
                    self.zero_support_rows = tuple(zeros)
                    self.current_report.setdefault('exact_activated_family_derivative_refreshes', []).append(dict(
                        block_ordinal=ordinal, family=ledger.get('family'),
                        activated_row_count=len(activated),
                        current_q_sha256=self.current_report['current_q_sha256'],
                        current_profile_token=self.current_report['current_profile_token'],
                        complete_family_rows=len(r), complete_parameter_columns=fresh_J.shape[1],
                        original_current_residual_mask_scales_and_ledger_unchanged=True,
                        other_family_J_remains_explicit_modified_Newton=True))
                    self.current_report['retained_CSR_payload_bytes'] = sum(
                        j.data.nbytes+j.indices.nbytes+j.indptr.nbytes for j in self.jacobians)
                    self.current_report['retained_zero_support_row_indices_bytes'] = sum(
                        row.nbytes for row in self.zero_support_rows)
                elif row_patch is None:
                    sample = activated[:32]
                    raise LaggedZeroJacobianActivation(dict(
                        schema='rinkspace.original-zero-J-support-activation-refresh.v1',
                        block_ordinal=ordinal, family=ledger.get('family') if isinstance(ledger, dict) else None,
                        J_anchor_q_sha256=self.anchor_q,
                        current_q_sha256=self.current_report['current_q_sha256'],
                        current_profile_token=self.current_report['current_profile_token'],
                        activated_row_count=len(activated), local_rows=sample.tolist(),
                        global_rows=(rows + sample).tolist(),
                        current_normalized_residual=r[sample].tolist(),
                        unchanged_native_scales=scales[sample].tolist(),
                        current_native_residual=(r[sample] * scales[sample]).tolist(),
                        anchor_residual_and_actual_numeric_J_were_exact_zero=True,
                        source_sigma_bounds_rows_roles_and_objective_unchanged=True))
            integration = supplied.loss_weights if isinstance(supplied, SourceRowBlock) else supplied.get('loss_weights')
            yield SourceRowBlock(r, self.jacobians[ordinal], mask, scales, ledger, integration)
            count += 1
            rows += len(r)
        _finish_replay(count, rows, self.contract)

    def refresh(self, reason):
        self.force_fresh = True
        self.refresh_reason = str(reason)
        self.current_report['next_linearization_refresh_reason'] = self.refresh_reason

    def accepted(self, before, after, step, normal, gradient, *, source_objective_acceptance=True):
        predicted = float(-gradient @ step - .5 * step @ (normal @ step))
        actual = float(before - after)
        valid_prediction = np.isfinite(predicted) and (predicted > 0 if source_objective_acceptance else predicted != 0)
        ratio = actual / predicted if valid_prediction else None
        self.current_report.update(predicted_source_reduction=predicted,
            actual_source_reduction=actual, actual_to_predicted_reduction=ratio if source_objective_acceptance else None)
        if not source_objective_acceptance:
            self.current_report.update(actual_to_predicted_source_change_for_restoration=ratio,
                source_change_model_agreement_is_not_ordinary_LM_acceptance_rho=True)
            # Necessary truth restoration can legitimately increase source
            # cost. Its signed ratio may then be large and positive despite
            # a poor model, so the ordinary one-sided reduction test cannot
            # decide whether this J remains reusable. Reuse exactly the same
            # symmetric agreement criterion as restoration LM control.
            from z190_joint_source_solver import restoration_source_model_agreement
            agreement=restoration_source_model_agreement(predicted,actual)
            self.current_report['restoration_source_model_agreement']=agreement
            if not agreement['reliable_for_restoration_lambda_decrease']:
                self.refresh('accepted truth-restoration source model is unreliable at the original symmetric relative-error criterion; regenerate exact current q/profile J')
                return
        if ratio is None or not np.isfinite(ratio) or ratio < self.refresh_reduction_ratio:
            self.refresh('accepted nonlinear reduction has an invalid prediction or is below the explicitly selected refresh ratio')

    def report(self):
        return dict(self.current_report)
