"""Same-estimator whole-exposure conditional MC with exact CUDA filters.

The CPU source estimator and noise model are unchanged. Only Gaussian and cubic
prefilter arithmetic use the separately shadowed NVRTC implementation.
"""
from pathlib import Path
import copy
from .source_statistics_storage import compact_normal_replay,encode_rng_state
from .source_worker_memory import snapshot as memory_snapshot
import json,time
import numpy as np
from scipy.ndimage import spline_filter,gaussian_filter
from . import outer_frame_compact as cpu
from .ptx_source_filter import PTXSourceFilterCUDA
from .io_utils import write_json,file_sha256

_original_frame=cpu._frame

def _frame(job,*,gpu_ptx,gpu_full256_shadow):
    started=time.perf_counter();CPU_started=time.process_time();memory_start=memory_snapshot()
    job={**job,"gpu_ptx":str(gpu_ptx),"GPU_full256_shadow":gpu_full256_shadow}
    backend=PTXSourceFilterCUDA(job["rgb"].shape[0],job["rgb"].shape[1],
        ptx_file=job["gpu_ptx"])
    memory_context=memory_snapshot()
    original_measure=cpu._measure;original_save=np.savez_compressed
    verification=[];calls=[0];rng_records=[];filter_seconds=0.;profile_seconds=0.
    original_rng=np.random.default_rng
    def exact_rng(seed=None):
        generator=original_rng(seed);rng_records.append((generator,copy.deepcopy(generator.bit_generator.state)))
        return generator
    def measure(rgb,rows,operator):
        nonlocal filter_seconds,profile_seconds
        filter_start=time.perf_counter()
        a=rgb.astype(np.float32);channel=a[:,:,2]-.5*(a[:,:,0]+a[:,:,1])
        if calls[0]<2:g,c=backend(channel)
        else:c=backend(channel,include_gaussian=False)
        if calls[0] < 2:
            refg=gaussian_filter(channel,.8);refc=spline_filter(refg,order=3)
            error=(float(np.max(abs(g-refg))),float(np.max(abs(c-refc))))
            if error!=(0.,0.):raise ValueError("CUDA filter differs from same-exposure CPU arithmetic")
            verification.append(dict(call=calls[0],gaussian_max=error[0],prefilter_max=error[1]))
        filter_seconds+=time.perf_counter()-filter_start;profile_start=time.perf_counter()
        calls[0]+=1
        shifts=[];bound=[];flat=[]
        for row in rows:
            q=operator.replay_outer_cubic(rgb,row,coefficients={2:c})
            if not q["numeric_valid"]:raise ValueError("frozen source estimator has no numerical definition")
            shifts.append(q["normal_offset_px"]);bound.append(q["synthetic_window_boundary_count"]);flat.append(len(q["flat_peak_intervals"]))
        profile_seconds+=time.perf_counter()-profile_start
        return np.asarray(shifts),np.asarray(bound),np.asarray(flat)
    def compact_save(file,*args,**arrays):
        if Path(file).name=="joint_outer_replay.npz":
            arrays=compact_normal_replay(arrays)
        return original_save(file,*args,**arrays)
    cpu._measure=measure;np.savez_compressed=compact_save;np.random.default_rng=exact_rng
    job["defer_final_receipt"]=True
    job["worker_queue_seconds"]=time.time()-job["submitted_wall_time"]if job.get("submitted_wall_time")else None
    try:record=_original_frame(job)
    finally:
        cpu._measure=original_measure;np.savez_compressed=original_save;np.random.default_rng=original_rng;backend.close()
    if len(rng_records)!=1:raise ValueError("one exact whole-exposure RNG stream required")
    generator,initial_state=rng_records[0]
    record["RNG_replay_contract"]=dict(numpy_version=np.__version__,bit_generator=generator.bit_generator.__class__.__name__,seed_sequence_entropy=[job["seed"],int(job["rows"][0]["native_pts"])],state_integer_encoding="hexadecimal unsigned128 strings; avoids JSON floating conversion",initial_state=encode_rng_state(initial_state),final_state=encode_rng_state(generator.bit_generator.state),draw_method="Generator.uniform(-.5,+.5,shape=RGB.shape)",draw_dtype="float64",image_shape=list(job["rgb"].shape),image_order="C row-major RGB",trials=job["trials"],same_exposure_stream_shared_by_all_source_rows=True)
    record["timings_seconds"].update(GPU_filter_and_channel=filter_seconds,continuous_profile_estimation=profile_seconds)
    record["memory_counters"]=dict(before_context=memory_start,context_loaded=memory_context,after_frame_and_context_close=memory_snapshot())
    record["source_projection"]=job.get("source_projection")
    record["compute_backend_GPU"]=True
    record["compute_backend"]="CUDA exact Gaussian/cubic prefilter; original CPU full-RGB draws, continuous estimator and covariance/whitening"
    record["GPU_worker_elapsed_seconds"]=time.perf_counter()-started
    record["GPU_worker_process_CPU_seconds"]=time.process_time()-CPU_started
    record["GPU_filter"]=dict(PTX=dict(file=str(Path(job["gpu_ptx"]).resolve()),sha256=file_sha256(job["gpu_ptx"])),
        base_and_first_shared_perturbation_full_coefficient_shadow=verification,
        full256_same_mean_covariance_W_shadow=job["GPU_full256_shadow"],
        exact_SciPy_reflect_Gaussian_and_mirror_cubic_recurrence=True,
        RGB_noise_draws_and_continuous_source_estimator_unchanged=True,
        device_context_CPU_wait="blocking_sync",NVRTC_compiler_DLLs_not_loaded=True,unused_Gaussian_D2H_omitted_after_two_exact_shadows=True)
    record["replay_arrays_layout"]="normal covariance, mean UV, profile normals, normal shift bias/extrema/digest and necessary nonregular diagnostic columns; full UV covariance B.T@C@B; exact full256 regeneration uses frozen operator/native RGB/RNG state"
    record["full_normal_trials_stored"]=False
    record["full_MC_draw_stream_exactly_replayable"]=True
    record["storage_policy"]="no full RGB draws, duplicate full UV covariance or WtW; normal draws retained only for actual nonregular/window/zero-variance diagnostic sources"
    write_json(Path(job["out"])/"receipt.json",record)
    return record

def compile_outer_statistics_gpu(run,source_rows,frozen_operator,output,*,gpu_ptx,gpu_full256_shadow,trials=256,seed=20261018,workers=2):
    """Preserve the CPU source scheduler/role guard; select GPU in new workers."""
    from functools import partial
    from .streaming_source_statistics import compile_streaming_statistics
    frame=partial(_frame,gpu_ptx=str(gpu_ptx),gpu_full256_shadow=gpu_full256_shadow)
    result=compile_streaming_statistics(run,source_rows,frozen_operator,output,frame_callable=frame,trials=trials,seed=seed,workers=workers)
    result["GPU_filter"]=dict(PTX=dict(file=str(Path(gpu_ptx).resolve()),sha256=file_sha256(gpu_ptx)),full256_shadow=gpu_full256_shadow,source_estimator_noise_or_trial_count_changed=False,each_exposure_base_and_first_noise_CPU_coefficient_shadow_required=True)
    result["compact_derived_UV_covariance_not_duplicated"]=True
    result["complete_normal_trials_not_stored"]=True
    result["full_RGB_RNG_state_operator_and_native_hash_replay_retained"]=True
    write_json(Path(output)/"manifest.json",result);return result
