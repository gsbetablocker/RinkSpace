"""Same-estimator whole-exposure conditional MC with exact CUDA filters.

The CPU source estimator and noise model are unchanged. Only Gaussian and cubic
prefilter arithmetic use the separately shadowed NVRTC implementation.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.ndimage import spline_filter,gaussian_filter
from . import outer_statistics as cpu
from .nvrtc_source_filter import NVRTCSourceFilterCUDA
from .io_utils import write_json,file_sha256

_original_frame=cpu._frame

def _frame(job,*,gpu_ptx,gpu_full256_shadow):
    started=time.perf_counter();CPU_started=time.process_time()
    job={**job,"gpu_ptx":str(gpu_ptx),"GPU_full256_shadow":gpu_full256_shadow}
    backend=NVRTCSourceFilterCUDA(job["rgb"].shape[0],job["rgb"].shape[1],
        ptx_file=job["gpu_ptx"])
    original_measure=cpu._measure;original_save=np.savez_compressed
    verification=[];calls=[0]
    def measure(rgb,rows,operator):
        a=rgb.astype(np.float32);channel=a[:,:,2]-.5*(a[:,:,0]+a[:,:,1])
        g,c=backend(channel)
        if calls[0] < 2:
            refg=gaussian_filter(channel,.8);refc=spline_filter(refg,order=3)
            error=(float(np.max(abs(g-refg))),float(np.max(abs(c-refc))))
            if error!=(0.,0.):raise ValueError("CUDA filter differs from same-exposure CPU arithmetic")
            verification.append(dict(call=calls[0],gaussian_max=error[0],prefilter_max=error[1]))
        calls[0]+=1
        shifts=[];bound=[];flat=[]
        for row in rows:
            q=operator.replay_outer_cubic(rgb,row,coefficients={2:c})
            if not q["numeric_valid"]:raise ValueError("frozen source estimator has no numerical definition")
            shifts.append(q["normal_offset_px"]);bound.append(q["synthetic_window_boundary_count"]);flat.append(len(q["flat_peak_intervals"]))
        return np.asarray(shifts),np.asarray(bound),np.asarray(flat)
    def compact_save(file,*args,**arrays):
        if Path(file).name=="joint_outer_replay.npz":
            arrays.pop("full_uv_covariance_px2",None)
        return original_save(file,*args,**arrays)
    cpu._measure=measure;np.savez_compressed=compact_save
    try:record=_original_frame(job)
    finally:
        cpu._measure=original_measure;np.savez_compressed=original_save;backend.close()
    record["compute_backend_GPU"]=True
    record["compute_backend"]="CUDA exact Gaussian/cubic prefilter; original CPU full-RGB draws, continuous estimator and covariance/whitening"
    record["GPU_worker_elapsed_seconds"]=time.perf_counter()-started
    record["GPU_worker_process_CPU_seconds"]=time.process_time()-CPU_started
    record["GPU_filter"]=dict(PTX=dict(file=str(Path(job["gpu_ptx"]).resolve()),sha256=file_sha256(job["gpu_ptx"])),
        base_and_first_shared_perturbation_full_coefficient_shadow=verification,
        full256_same_mean_covariance_W_shadow=job["GPU_full256_shadow"],
        exact_SciPy_reflect_Gaussian_and_mirror_cubic_recurrence=True,
        RGB_noise_draws_and_continuous_source_estimator_unchanged=True,
        device_context_CPU_wait="blocking_sync")
    record["replay_arrays_layout"]="normal-shift samples, normal covariance, profile normals and mean UV; full UV covariance is B.T@C@B, derived without a duplicate stored matrix"
    write_json(Path(job["out"])/"receipt.json",record)
    return record

def compile_outer_statistics_gpu(run,source_rows,frozen_operator,output,*,gpu_ptx,gpu_full256_shadow,trials=256,seed=20261018,workers=2):
    """Preserve the CPU source scheduler/role guard; select GPU in new workers."""
    original=cpu._frame
    from functools import partial
    cpu._frame=partial(_frame,gpu_ptx=str(gpu_ptx),gpu_full256_shadow=gpu_full256_shadow)
    try:result=cpu.compile_outer_statistics(run,source_rows,frozen_operator,output,trials=trials,seed=seed,workers=workers)
    finally:cpu._frame=original
    result["GPU_filter"]=dict(PTX=dict(file=str(Path(gpu_ptx).resolve()),sha256=file_sha256(gpu_ptx)),full256_shadow=gpu_full256_shadow,source_estimator_noise_or_trial_count_changed=False,each_exposure_base_and_first_noise_CPU_coefficient_shadow_required=True)
    result["compact_derived_UV_covariance_not_duplicated"]=True
    write_json(Path(output)/"manifest.json",result);return result
