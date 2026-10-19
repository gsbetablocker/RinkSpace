"""Same-state actual v1-to-v2 new-cue normalization check, no legacy graph."""
from __future__ import annotations
import hashlib,time
import numpy as np
from .native_ice_curve_source_mean_normalized_v1 import NativeIceCurveFactor as MeanV1
from .native_ice_curve_source import NativeIceCurveFactor as CurrentGeometry

class MeanV1OnCurrentGeometry(CurrentGeometry):
    """Replay v1 normalization on new finite segments, in diagnostics only."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scales=np.concatenate([np.repeat(row['sigma'],2)*np.sqrt(len(row['uv']))for row in self.rows])
        self.contract=dict(self.contract,schema='rinkspace.current-native-ice-geometry-v1-normalization-diagnostic.v1')
    def _residual(self,q,rows,*,cameras=None):
        values=CurrentGeometry._residual(self,q,rows,cameras=cameras)
        multiplier=np.concatenate([np.full(2*len(row['uv']),np.sqrt(len(row['uv'])))for row in rows])
        return values/multiplier

def _active_original_scalar_indices(extra):
    """Select admitted FIT slots without changing the original sampling N."""
    qualification=getattr(extra,'source_point_qualification',None)
    masks=[]
    for row in extra.rows:
        mask=(np.ones(len(row['uv']),bool) if qualification is None
              else np.asarray(qualification.point_mask(row)))
        if mask.shape!=(len(row['uv']),) or mask.dtype!=np.bool_:
            raise ValueError('Native ice qualification has an invalid original point mask')
        masks.append(np.repeat(mask,2))
    return np.flatnonzero(np.concatenate(masks))

def check_raw_sigma_revision(extra,q):
    t=time.perf_counter()
    current_segments=any(row['curve']['kind']=='current_ice_segment'for row in extra.rows+extra.held)
    optical_support=any(row.get('_source_optical_support')is not None for row in extra.rows+extra.held)
    qualification=getattr(extra,'source_point_qualification',None)
    old_type=MeanV1OnCurrentGeometry if current_segments or optical_support or qualification is not None else MeanV1
    old=old_type(extra.state,extra.packet_path,extra.parameter_count)
    if len(old.rows)!=len(extra.rows)or len(old.held)!=len(extra.held):raise ValueError('Normalization revision changed source group inventory')
    for a,b in zip(old.rows+old.held,extra.rows+extra.held):
        if (a['id']!=b['id']or a['split']!=b['split']or a['actual_pts']!=b['actual_pts']or
                not np.array_equal(a['uv'],b['uv'])or not np.array_equal(a['sigma'],b['sigma'])):
            raise ValueError('Normalization revision changed source UV/sigma/split/identity')
    selected=_active_original_scalar_indices(extra)
    original_scalars=sum(2*len(row['uv'])for row in extra.rows)
    # This historical diagnostic evaluates the original full source only, then
    # selects exactly the production qualification. Its sqrt(N) means original
    # source N, never the smaller active count. The production view remains
    # untouched and continues to omit retired UV before residual/J evaluation.
    if qualification is not None:
        old.source_point_qualification=None
        old._active_fit_view_cache=None
        old.row_count=len(old.scales)
        old.quadratic=np.ones(old.row_count,bool)
    mult=np.concatenate([np.full(2*len(row['uv']),np.sqrt(len(row['uv'])))for row in extra.rows])[selected]
    before=old.block(q,True);after=extra.block(q,True)
    if len(before.r)!=original_scalars or len(after.r)!=len(selected):
        raise ValueError('Normalization revision does not match the qualified active scalar layout')
    if not np.array_equal(before.quadratic[selected],after.quadratic):raise ValueError('New cue loss mask changed during normalization revision')
    if not np.array_equal(after.scales,np.concatenate([np.repeat(row['sigma'],2)for row in extra.rows])[selected]):
        raise ValueError('Raw per-UV sigma was not used directly')
    expected_r=before.r[selected]*mult;expected_j=before.J[selected].multiply(mult[:,None]).tocsr()
    dr=float(np.max(np.abs(after.r-expected_r)))
    delta=(after.J-expected_j).tocsr();dj=0. if not delta.nnz else float(np.max(np.abs(delta.data)))
    # The two actual FD evaluations perform division/subtraction in a different
    # order. This bound is solely their finite arithmetic agreement, never a
    # fitted pixel tolerance or a covariance adjustment.
    jr=2e-7+2e-9*float(np.max(np.abs(after.J.data)))
    if dr>2e-13 or dj>jr:raise ValueError('Actual normalization-only residual/J transformation failed')
    return dict(schema='rinkspace.actual-native-ice-raw-sigma-revision-check.v1',
        before_factor_schema=old.contract['schema'],after_factor_schema=extra.contract['schema'],
        current_geometry_extensions_preserved=current_segments,
        historical_v1_full_geometry_operator_replayed=not(current_segments or optical_support or qualification is not None),
        explicit_optical_support_retained_during_normalization_check=optical_support,
        v1_normalization_arithmetic_replayed=True,diagnostic_factor_not_used_in_solver=True,
        complete_q_sha256=hashlib.sha256(np.asarray(q,dtype=np.float64).tobytes()).hexdigest(),
        packet_sha256=extra.contract['packet_sha256'],native_fit_UV=len(selected)//2,
        original_native_fit_UV=sum(len(r['uv'])for r in extra.rows),
        native_held_UV=sum(len(r['uv'])for r in extra.held),original_UV_sigma_split_aliases_byte_exact=True,
        scalar_fit_rows=extra.row_count,quadratic_mask_exact=True,raw_sigma_direct_exact=True,
        actual_r_vs_v1_times_sqrt_N_max_absolute=dr,actual_J_vs_v1_times_sqrt_N_max_absolute=dj,
        finite_arithmetic_J_bound=jr,all_scalar_rows_preserved=len(selected)==original_scalars,
        qualified_active_scalar_rows=len(selected),original_scalar_rows=original_scalars,
        source_only_qualified_scalar_rows_removed=original_scalars-len(selected),
        active_FIT_matches_original_qualified_scalar_selection=True,
        historical_group_sqrt_N_uses_original_source_count=True,
        group_multipliers=[dict(id=r['id'],UV_count=len(r['uv']),residual_and_J_multiplier=float(np.sqrt(len(r['uv']))),
            normal_gradient_and_cost_multiplier=len(r['uv']))for r in extra.rows],
        legacy_sigma_correlation_density_and_loss_not_changed=True,seconds=time.perf_counter()-t,passed=True)
