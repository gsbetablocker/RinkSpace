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
from .ptx_source_filter_roi import PTXSourceFilterROI,frozen_normal_y_ranges
from .io_utils import write_json,file_sha256

_original_frame=cpu._frame

def _frame(job,*,gpu_ptx,gpu_full256_shadow):
    started=time.perf_counter();CPU_started=time.process_time();memory_start=memory_snapshot()
    job={**job,"gpu_ptx":str(gpu_ptx),"GPU_full256_shadow":gpu_full256_shadow}
    source_operator=cpu._load_frozen(job["frozen"])
    support=frozen_normal_y_ranges(job["rows"],source_operator.CONFIG["normal_offsets"],*job["rgb"].shape[:2])
    backend=PTXSourceFilterROI(job["rgb"].shape[0],job["rgb"].shape[1],
        ptx_file=job["gpu_ptx"],coefficient_y_ranges=support["ranges"])
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
            error=(float(np.max(abs(g-refg))),float(np.max(abs(backend.full_shadow_coefficients-refc))))
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
    record["GPU_filter"]["coefficient_readback_ROI"]=dict(support=support,bytes_per_call=backend.rows_downloaded*backend.w*8,full_native_coefficients_computed=True,host_shape_and_global_indices_unchanged=True,non_downloaded_host_rows_poisoned_NaN=True,base_and_first_draw_full_coefficient_and_poison_shadows=backend.readback_shadows,first_two_full_readbacks_are_QA_only=True)
    record["GPU_filter"]["ROI_producer"]=dict(file=str(Path(__file__).resolve()),sha256=file_sha256(__file__))
    record["replay_arrays_layout"]="normal covariance, mean UV, profile normals, normal shift bias/extrema/digest and necessary nonregular diagnostic columns; full UV covariance B.T@C@B; exact full256 regeneration uses frozen operator/native RGB/RNG state"
    record["full_normal_trials_stored"]=False
    record["full_MC_draw_stream_exactly_replayable"]=True
    record["storage_policy"]="no full RGB draws, duplicate full UV covariance or WtW; normal draws retained only for actual nonregular/window/zero-variance diagnostic sources"
    write_json(Path(job["out"])/"receipt.json",record)
    return record

