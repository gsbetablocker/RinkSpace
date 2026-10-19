"""Exact continuous-cubic outer means under shared native-exposure RGB noise.

This is an explicitly conditional encoded-RGB model, not a calibrated total
physical-error claim. No independent representative noise, ridge or floor.
"""
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,json,time
import numpy as np
from scipy.ndimage import spline_filter
from .paint_statistics import _load_frozen,_one_channel
from .native_source import iter_training_native_rgb
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening,StatisticsUnavailable


def _measure(rgb,rows,operator):
    coefficients={2:spline_filter(_one_channel(rgb,2),order=3)}
    shifts=[];bound=[];flat=[]
    for r in rows:
        q=operator.replay_outer_cubic(rgb,r,coefficients=coefficients)
        if not q['numeric_valid']:raise ValueError('frozen outer estimator has no finite numerical definition')
        shifts.append(q['normal_offset_px']);bound.append(q['synthetic_window_boundary_count']);flat.append(len(q['flat_peak_intervals']))
    return np.asarray(shifts),np.asarray(bound),np.asarray(flat)


def _frame(job):
    started=time.perf_counter();out=Path(job['out']);out.mkdir();rows=job['rows'];rgb=job['rgb'];frozen=Path(job['frozen']);operator=_load_frozen(frozen)
    channel_error=float(np.max(abs(operator._channels(rgb)[2]-_one_channel(rgb,2))))
    if channel_error!=0:raise ValueError('shared channel pipeline differs from actual frozen mean')
    base,bound,flat=_measure(rgb,rows,operator);normals=np.asarray([r['normal']for r in rows]);seed=np.asarray([r['navigation_seed_uv']for r in rows]);mean=seed+base[:,None]*normals;actual=np.asarray([r['current_uv']for r in rows]);error=np.linalg.norm(mean-actual,axis=1)
    if error.max(initial=0)>1e-9:raise ValueError('actual continuous cubic mean differs from exact frozen estimator')
    if any(r['decoded_native_BGR_sha256']!=job['native_receipt']['decoded_native_BGR_sha256']for r in rows):raise ValueError('outer statistics actual exposure SHA mismatch')
    n=len(rows);trials=job['trials'];rng=np.random.default_rng(np.random.SeedSequence([job['seed'],int(rows[0]['native_pts'])]));samples=np.empty((trials,n));boundary=np.zeros(n,np.int64);plateau=np.zeros(n,np.int64)
    for k in range(trials):
        perturbed=rgb.astype(np.float64)+rng.uniform(-.5,.5,rgb.shape);position,b,f=_measure(perturbed,rows,operator);samples[k]=position-base;boundary+=b;plateau+=f
    factor=(samples-samples.mean(0)).T/np.sqrt(trials-1);C=factor@factor.T;B=np.zeros((n,2*n))
    for i,normal in enumerate(normals):B[i,2*i:2*i+2]=normal
    W=None;proof=None;status='usable_conditional_joint_native_outer_W'
    try:
        Wn,proof=supported_factor_whitening(factor);W=Wn@B;W.astype('<f8').tofile(out/'W.f64')
    except StatisticsUnavailable as exc:status='no_positive_supported_covariance_not_infinite_precision';proof=dict(reason=str(exc))
    np.savez_compressed(out/'joint_outer_replay.npz',source_row_ids=np.asarray([r['source_row_id']for r in rows]),normal_shift_delta_px=samples,profile_normals=normals,actual_mean_uv=actual,normal_covariance_px2=C,full_uv_covariance_px2=B.T@C@B,synthetic_window_boundary_counts=boundary,synthetic_flat_peak_counts=plateau,actual_normal_offsets_px=base)
    source_hash={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=r['source_row_id']))for r in rows}
    current_hash={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=r['source_row_id'],measurement_operator=r['measurement_operator_resolved']))for r in rows}
    std=np.sqrt(np.diag(C));eigen=np.linalg.eigvalsh(C);bias=samples.mean(0)
    group=dict(source_group_id='actual_outer_exposure:'+rows[0]['shot_id']+':'+str(rows[0]['native_pts']),row_ids=[r['source_row_id']for r in rows],source_family_ids=sorted({r['source_family_id']for r in rows}),native_pts=rows[0]['native_pts'],shot_id=rows[0]['shot_id'],status=status,rank=len(W)if W is not None else 0,W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape))if W is not None else None,
        statistic_kind='native_joint_readout',original_numerical_scheme='whole_native_exposure_shared_RGB_exact_continuous_cubic_nonlinear_replay',numeric_whitening_verified=W is not None,source_measurements_sha256=source_hash,current_measurements_sha256=current_hash,source_receipts_sha256=object_sha256(rows),actual_mean_replay_max_px=float(error.max(initial=0)),float32_channel_pipeline_error=channel_error,trials=trials,seed=job['seed'],native_receipt=job['native_receipt'],frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)),replay_arrays=dict(file=str(out/'joint_outer_replay.npz'),sha256=file_sha256(out/'joint_outer_replay.npz')),
        normal_std_px_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),normal_covariance_eigenvalue_quantiles=np.quantile(eigen,[0,.1,.5,.9,1]).tolist(),MC_estimator_bias_px_quantiles=np.quantile(bias,[0,.1,.5,.9,1]).tolist(),MC_estimator_max_abs_bias_px=float(abs(bias).max(initial=0)),synthetic_window_boundary_count=int(boundary.sum()),synthetic_flat_peak_interval_count=int(plateau.sum()),base_flat_peak_interval_rows=int(np.sum(flat>0)),physical_total_error_calibrated=False,shared_full_native_RGB_domain=True,independent_rows_or_PSD_floor_added=False,tangent_information_from_navigation_seed_claimed=False,
        noise_calibration_status='usable conditional encoded-RGB quantization covariance; same continuous cubic source mean; broader total physical noise and selection uncertainty uncalibrated',conditional_model=dict(encoded_RGB_noise='independent uniform[-.5,+.5] per native encoded pixel/channel, full same-exposure draw shared by top/contact/all rows, no clipping',material_selection='frozen seed/channel/polarity/window, no synthetic quality censoring or source selection',covariance=f'finite{trials} joint nonlinear Monte Carlo, empirical numerical support only; no variance floor or PSD ridge',fixed_tangential_query='source normal localization only; no tangent precision invented'),conditional_model_is_total_error_calibration=False)
    group['source_MC_elapsed_seconds']=time.perf_counter()-started;group['MC_trials_per_second']=trials/group['source_MC_elapsed_seconds'];write_json(out/'receipt.json',group);return group


def compile_outer_statistics(run,source_rows,frozen_operator,output,*,trials=128,seed=20261018,workers=2):
    run=Path(run);source=Path(source_rows);frozen=Path(frozen_operator).resolve();out=Path(output)
    if out.exists():raise ValueError('new immutable statistics generation required')
    frames=defaultdict(list);keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','navigation_seed_uv','normal','measurement_operator_resolved','decoded_native_BGR_sha256','joint_material_id','joint_material_index')
    for original in read_jsonl(source):
        if original.get('source_geometry_eligible')is not True:continue
        row={k:original[k]for k in keep};row['source_measurement_audit']={'profile':original['source_measurement_audit']['profile']};frames[int(row['native_pts'])].append(row)
    if not frames:raise ValueError('no physically source-qualified actual outer means')
    out.mkdir(parents=True);records=[];priority=[]
    for sid in sorted({rs[0]['shot_id']for rs in frames.values()}):priority.append(min(p for p,rs in frames.items()if rs[0]['shot_id']==sid))
    priority=set(priority);stages=[priority,set(frames)-priority];pending=[];seen=set()
    def accept(future):
        rec=future.result();records.append(rec);print(json.dumps(dict(completed_exposures=len(records),source_rows=sum(len(x['row_ids'])for x in records),shots=sorted({x['shot_id']for x in records}),native_pts=rec['native_pts'],rank=rec['rank'],actual_base_replay_max_px=rec['actual_mean_replay_max_px'],source_MC_elapsed_seconds=rec.get('source_MC_elapsed_seconds'))),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for wanted in stages:
            for pts,rgb,receipt in iter_training_native_rgb(run,wanted):
                seen.add(pts);pending.append(pool.submit(_frame,dict(rows=frames[pts],rgb=rgb,native_receipt=receipt,out=str(out/(frames[pts][0]['shot_id']+'_'+str(pts))),frozen=str(frozen),trials=trials,seed=seed)))
                if len(pending)>=2*workers:accept(pending.pop(0))
            while pending:accept(pending.pop(0))
    if seen!=set(frames):raise ValueError('not all actual source exposures replayed')
    registry={r['source_group_id']:r for r in records if r.get('W')};write_json(out/'statistics_registry.json',registry);report=dict(schema='rinkfit.actual_continuous_cubic_outer_conditional_statistics.v1',source_rows=sum(len(r['row_ids'])for r in records),exposures=len(records),shots=sorted({r['shot_id']for r in records}),rank=sum(r['rank']for r in records),trials=trials,original_UV_and_current_source_mean_preserved=True,old_bilinear_W_reused=False,conditional_model_explicit=True,physical_total_noise_accuracy_claimed=False,statistics_registry=dict(file=str(out/'statistics_registry.json'),sha256=file_sha256(out/'statistics_registry.json')),source_generation=dict(file=str(source),sha256=file_sha256(source)),frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)))
    write_json(out/'manifest.json',report);return report
