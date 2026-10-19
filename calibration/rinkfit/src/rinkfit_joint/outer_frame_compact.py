"""Exact existing source frame arithmetic with bounded RGB work buffers.

A native RGB64 conversion is reused; every original full RGB uniform draw is
unchanged, and the binary addition is written into that draw. The estimator,
normal covariance and numerical supported whitening are unchanged.
"""
from pathlib import Path
import importlib.util,time
import numpy as np
from scipy.ndimage import gaussian_filter,spline_filter
from .io_utils import file_sha256,object_sha256,write_json
from .source_statistics import supported_factor_whitening,StatisticsUnavailable

def _load_frozen(path):
    spec=importlib.util.spec_from_file_location('frozen_source_replay',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def _one_channel(rgb,ch):
    a=rgb.astype(np.float32)
    v=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2])if ch==0 else(a[:,:,0]-a[:,:,1]if ch==1 else a[:,:,2]-.5*(a[:,:,0]+a[:,:,1]))
    return gaussian_filter(v,sigma=.8)

def _measure(rgb,rows,operator):
    coefficients={2:spline_filter(_one_channel(rgb,2),order=3)}
    shifts=[];bound=[];flat=[]
    for r in rows:
        q=operator.replay_outer_cubic(rgb,r,coefficients=coefficients)
        if not q['numeric_valid']:raise ValueError('frozen outer estimator has no finite numerical definition')
        shifts.append(q['normal_offset_px']);bound.append(q['synthetic_window_boundary_count']);flat.append(len(q['flat_peak_intervals']))
    return np.asarray(shifts),np.asarray(bound),np.asarray(flat)


def _frame(job):
    started=time.perf_counter();timings={};out=Path(job['out']);out.mkdir();rows=job['rows'];rgb=job['rgb'];frozen=Path(job['frozen']);operator=_load_frozen(frozen)
    channel_error=float(np.max(abs(operator._channels(rgb)[2]-_one_channel(rgb,2))))
    if channel_error!=0:raise ValueError('shared channel pipeline differs from actual frozen mean')
    base,bound,flat=_measure(rgb,rows,operator);normals=np.asarray([r['normal']for r in rows]);seed=np.asarray([r['navigation_seed_uv']for r in rows]);mean=seed+base[:,None]*normals;actual=np.asarray([r['current_uv']for r in rows]);error=np.linalg.norm(mean-actual,axis=1)
    if error.max(initial=0)>1e-9:raise ValueError('actual continuous cubic mean differs from exact frozen estimator')
    if any(r['decoded_native_BGR_sha256']!=job['native_receipt']['decoded_native_BGR_sha256']for r in rows):raise ValueError('outer statistics actual exposure SHA mismatch')
    timings['initialization_and_actual_mean_check']=time.perf_counter()-started;n=len(rows);trials=job['trials'];cast_start=time.perf_counter();rgb64=rgb.astype(np.float64);timings['RGB64_cast_once']=time.perf_counter()-cast_start;draw_seconds=0.;measure_seconds=0.;rng=np.random.default_rng(np.random.SeedSequence([job['seed'],int(rows[0]['native_pts'])]));samples=np.empty((trials,n));boundary=np.zeros(n,np.int64);plateau=np.zeros(n,np.int64)
    for k in range(trials):
        t=time.perf_counter();perturbed=rng.uniform(-.5,.5,rgb.shape);np.add(rgb64,perturbed,out=perturbed);draw_seconds+=time.perf_counter()-t;t=time.perf_counter();position,b,f=_measure(perturbed,rows,operator);measure_seconds+=time.perf_counter()-t;samples[k]=position-base;boundary+=b;plateau+=f
    del perturbed,rgb64
    timings['whole_RGB_256_draws_and_addition']=draw_seconds;timings['measurement_including_GPU_filters_and_profiles']=measure_seconds;cov_start=time.perf_counter()
    factor=(samples-samples.mean(0)).T/np.sqrt(trials-1);C=factor@factor.T;B=np.zeros((n,2*n))
    for i,normal in enumerate(normals):B[i,2*i:2*i+2]=normal
    W=None;proof=None;status='usable_conditional_joint_native_outer_W'
    try:
        Wn,proof=supported_factor_whitening(factor);W=Wn@B;W.astype('<f8').tofile(out/'W.f64')
    except StatisticsUnavailable as exc:status='no_positive_supported_covariance_not_infinite_precision';proof=dict(reason=str(exc))
    timings['normal_covariance_and_whitening_W']=time.perf_counter()-cov_start;save_start=time.perf_counter()
    np.savez_compressed(out/'joint_outer_replay.npz',source_row_ids=np.asarray([r['source_row_id']for r in rows]),normal_shift_delta_px=samples,profile_normals=normals,actual_mean_uv=actual,normal_covariance_px2=C,synthetic_window_boundary_counts=boundary,synthetic_flat_peak_counts=plateau,actual_normal_offsets_px=base)
    timings['replay_array_serialization']=time.perf_counter()-save_start;meta_start=time.perf_counter()
    source_hash={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=r['source_row_id']))for r in rows}
    current_hash={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=r['source_row_id'],measurement_operator=r['measurement_operator_resolved']))for r in rows}
    std=np.sqrt(np.diag(C));eigen=np.linalg.eigvalsh(C);bias=samples.mean(0)
    group=dict(source_group_id='actual_outer_exposure:'+rows[0]['shot_id']+':'+str(rows[0]['native_pts']),row_ids=[r['source_row_id']for r in rows],source_family_ids=sorted({r['source_family_id']for r in rows}),native_pts=rows[0]['native_pts'],shot_id=rows[0]['shot_id'],status=status,rank=len(W)if W is not None else 0,W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape))if W is not None else None,
        statistic_kind='native_joint_readout',original_numerical_scheme='whole_native_exposure_shared_RGB_exact_continuous_cubic_nonlinear_replay',numeric_whitening_verified=W is not None,source_measurements_sha256=source_hash,current_measurements_sha256=current_hash,source_receipts_sha256=object_sha256(rows),actual_mean_replay_max_px=float(error.max(initial=0)),float32_channel_pipeline_error=channel_error,trials=trials,seed=job['seed'],native_receipt=job['native_receipt'],frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)),replay_arrays=dict(file=str(out/'joint_outer_replay.npz'),sha256=file_sha256(out/'joint_outer_replay.npz')),
        normal_std_px_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),normal_covariance_eigenvalue_quantiles=np.quantile(eigen,[0,.1,.5,.9,1]).tolist(),MC_estimator_bias_px_quantiles=np.quantile(bias,[0,.1,.5,.9,1]).tolist(),MC_estimator_max_abs_bias_px=float(abs(bias).max(initial=0)),synthetic_window_boundary_count=int(boundary.sum()),synthetic_flat_peak_interval_count=int(plateau.sum()),base_flat_peak_interval_rows=int(np.sum(flat>0)),physical_total_error_calibrated=False,shared_full_native_RGB_domain=True,independent_rows_or_PSD_floor_added=False,tangent_information_from_navigation_seed_claimed=False,
        noise_calibration_status='usable conditional encoded-RGB quantization covariance; same continuous cubic source mean; broader total physical noise and selection uncertainty uncalibrated',conditional_model=dict(encoded_RGB_noise='independent uniform[-.5,+.5] per native encoded pixel/channel, full same-exposure draw shared by top/contact/all rows, no clipping',material_selection='frozen seed/channel/polarity/window, no synthetic quality censoring or source selection',covariance=f'finite{trials} joint nonlinear Monte Carlo, empirical numerical support only; no variance floor or PSD ridge',fixed_tangential_query='source normal localization only; no tangent precision invented'),conditional_model_is_total_error_calibration=False)
    timings['source_hash_and_covariance_diagnostics']=time.perf_counter()-meta_start;group['timings_seconds']=timings;group['source_decode_seconds']=job.get('source_decode_seconds');group['parent_queue_seconds']=job.get('worker_queue_seconds');group['source_MC_elapsed_seconds']=time.perf_counter()-started;group['MC_trials_per_second']=trials/group['source_MC_elapsed_seconds'];
    if not job.get('defer_final_receipt'):write_json(out/'receipt.json',group)
    return group

