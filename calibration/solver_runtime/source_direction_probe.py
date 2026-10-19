"""Persist a real failed directional source gate without a programme graph."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import types

import numpy as np


def capture_source_direction_record(camera, world, uv, sigma, residual, *,
        world_columns=None, world_jacobian=None, camera_columns=None,
        camera_chain=None, local_camera_jacobian=None):
    from structured_programme_source_candidate.prepared_finite_curve_world_columns import prepare_curve_current_world_columns
    prepared=prepare_curve_current_world_columns(camera,world,uv)
    result=dict(camera25=np.array(camera,copy=True),world=np.array(world,copy=True),
        uv=np.array(uv,copy=True),sigma=np.array(sigma,copy=True),residual=np.array(residual,copy=True),
        selected_segment=prepared.selected_index,selected_raw_t=prepared.selected_raw_t,
        clipped_t=prepared.clipped_t,projected_knots=prepared.projected,
        valid_segments=prepared.valid_segments)
    for name,value in dict(world_columns=world_columns,world_jacobian=world_jacobian,
            camera_columns=camera_columns,camera_chain=camera_chain,
            local_camera_jacobian=local_camera_jacobian).items():
        if value is not None:result[name]=np.array(value,copy=True)
    if local_camera_jacobian is not None:
        import z190_source_camera_diagnostic as diagnostic
        original=diagnostic._curve_vector_error15
        selected=[];nearest=original.__globals__['nearest_segments']
        def capture_nearest(*args,**kwargs):
            value=nearest(*args,**kwargs);selected.append(value);return value
        globals_=dict(original.__globals__);globals_['nearest_segments']=capture_nearest
        replay=types.FunctionType(original.__code__,globals_,original.__name__,original.__defaults__,original.__closure__)
        replay(camera,world,uv)
        if (len(selected)!=1 or not np.array_equal(selected[0][0],prepared.selected_index)
                or not np.array_equal(selected[0][1],prepared.selected_raw_t)):
            raise ValueError('Original camera15 and prepared world selected different nominal finite branches')
        result['native_camera_selected_segment']=np.array(selected[0][0],copy=True)
        result['native_camera_selected_raw_t']=np.array(selected[0][1],copy=True)
    return result


def qualify_source_direction_records(records,matrix,direction,h,*,original_group_point_indices=None):
    """Qualify the original branch J and the true trials wherever differentiable.

    A finite polyline vertex has no single derivative when the active segment
    or endpoint clamp changes. The original native ice replay independently
    differentiates the declared nominal branch. Unrestricted trial residuals
    are still computed normally, and their stable rows must pass the same gate.
    """
    from structured_programme_source_candidate.native_ice_curve_source import _curve_error_on_association
    nominal,plus,minus=(records[k] for k in ('nominal','plus','minus'))
    uv,sigma=nominal['uv'],nominal['sigma'];scale=np.repeat(sigma,2)
    predicted=matrix@direction
    association=(nominal['selected_segment'],nominal['selected_raw_t'])
    nominal_replay,original_association=_curve_error_on_association(nominal['camera25'],nominal['world'],uv)
    if (not np.array_equal(nominal_replay/scale,nominal['residual'])
            or not np.array_equal(original_association[0],association[0])
            or not np.array_equal(original_association[1],association[1])):
        raise ValueError('Independent original source replay changed nominal residual bytes or finite branch')
    frozen=[]
    for record in (plus,minus):
        value,_=_curve_error_on_association(record['camera25'],record['world'],uv,association)
        frozen.append(value/scale)
    measured=(frozen[0]-frozen[1])/(2*h)
    maximum=float(np.abs(measured-predicted).max())
    relative=maximum/max(1.,float(np.abs(predicted).max()))
    def regime(record):
        t=record['selected_raw_t'];return np.where(t<=0.,-1,np.where(t>=1.,1,0))
    nominal_regime=regime(nominal);plus_regime=regime(plus);minus_regime=regime(minus)
    stable=((nominal['selected_segment']==plus['selected_segment'])
        &(nominal['selected_segment']==minus['selected_segment'])
        &(nominal_regime==plus_regime)&(nominal_regime==minus_regime))
    scalar_stable=np.repeat(stable,2)
    unrestricted=(plus['residual']-minus['residual'])/(2*h)
    stable_maximum=(float(np.abs(unrestricted[scalar_stable]-predicted[scalar_stable]).max())
        if stable.any() else None)
    stable_relative=(stable_maximum/max(1.,float(np.abs(predicted[scalar_stable]).max()))
        if stable.any() else None)
    indices=np.arange(len(uv)) if original_group_point_indices is None else np.asarray(original_group_point_indices)
    changed=[]
    for i in np.flatnonzero(~stable):
        entry=dict(local_query_index=int(i),original_group_point_index=int(indices[i]),source_uv=uv[i].tolist())
        for label,record,clamp in (('nominal',nominal,nominal_regime),('plus',plus,plus_regime),('minus',minus,minus_regime)):
            entry[label]=dict(segment=int(record['selected_segment'][i]),raw_t=float(record['selected_raw_t'][i]),
                clamp_regime=int(clamp[i]),original_raw_native_residual_px=(record['residual'][2*i:2*i+2]*sigma[i]).tolist())
        changed.append(entry)
    return dict(original_nominal_segment_and_clamp_fixed_FD_maximum_error=maximum,
        original_nominal_segment_and_clamp_fixed_FD_relative_error=relative,
        complete_original_branch_J_passed=relative<=1e-4,
        unrestricted_stable_queries=int(stable.sum()),unrestricted_changed_queries=int((~stable).sum()),
        unrestricted_stable_FD_maximum_error=stable_maximum,unrestricted_stable_FD_relative_error=stable_relative,
        unrestricted_stable_FD_available=bool(stable.any()),
        unrestricted_stable_queries_passed=None if stable_relative is None else stable_relative<=1e-4,
        unrestricted_all_query_FD_maximum_difference=float(np.abs(unrestricted-predicted).max()),
        changed_segment_or_clamp_queries=changed,source_FIT_pixels_removed=0,
        all_FIT_source_residuals_J_and_nonlinear_trials_unchanged=True,
        independently_replayed_nominal_source_residual_bytes_and_association_identical=True,
        original_relative_threshold=1e-4,passed=relative<=1e-4 and (stable_relative is None or stable_relative<=1e-4))


def save_source_direction_probe(q,direction,h,matrix,records,metadata):
    from z190_source_camera_diagnostic import absolute_value_jac15
    token=hashlib.sha256(np.asarray(q).tobytes()+json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    root=Path(os.environ.get('RINKSPACE_SOURCE_RUNTIME_DIR',
        'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/strict_direction_probes'))
    root.mkdir(parents=True,exist_ok=True);path=root/('failed_strict_direction_'+token[:20]+'.npz')
    arrays=dict(q=np.asarray(q),direction=np.asarray(direction),step=np.array(h),
        matrix_data=matrix.data,matrix_indices=matrix.indices,matrix_indptr=matrix.indptr,
        matrix_shape=np.array(matrix.shape),metadata_json=np.array(json.dumps(metadata,ensure_ascii=False)))
    for label,record in records.items():
        arrays.update({label+'__'+key:value for key,value in record.items()})
    nominal=records['nominal'];plus=records['plus'];minus=records['minus']
    def residual(camera,world):
        return absolute_value_jac15(camera,[dict(world=world,uv=nominal['uv'],sigma=nominal['sigma'])],False)
    # Same nonlinear source operator, perturbing one route at a time.
    arrays['world_only_FD']=(residual(nominal['camera25'],plus['world'])-
        residual(nominal['camera25'],minus['world']))/(2*h)
    arrays['camera_only_FD']=(residual(plus['camera25'],nominal['world'])-
        residual(minus['camera25'],nominal['world']))/(2*h)
    arrays['actual_camera25_FD']=(plus['camera25']-minus['camera25'])/(2*h)
    arrays['actual_world_FD']=(plus['world']-minus['world'])/(2*h)
    np.savez_compressed(path,**arrays)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        failed_original_directional_gate_preserved=True,complete_q_and_local_source_operator_arrays_preserved=True,
        original_threshold_unchanged=True)
