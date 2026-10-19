"""Whole-native-exposure conditional statistics for mixed actual finite sources.

All points/flanks share one encoded-RGB draw. Frozen scalar source operators are
replayed unchanged; strict same-stream board MC samples may be reused.
"""
from pathlib import Path
from collections import defaultdict
import json,time
import numpy as np
from .paint_statistics import _prepare,_load_frozen
from .paint_pair_statistics import _measure_pair
from .nvrtc_source_filter import NVRTCSourceFilterCUDA
from .native_source import iter_training_native_rgb
from .source_statistics import supported_factor_whitening
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256

def _raw_channel(rgb,ch):
    a=rgb.astype(np.float32)
    return (.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2])if ch==0 else(a[:,:,0]-a[:,:,1]if ch==1 else a[:,:,2]-.5*(a[:,:,0]+a[:,:,1]))

def _frame(rgb,native,rows,operator,backend,out,trials,seed,board_cache=None):
    started=time.perf_counter();out=Path(out);out.mkdir();n=len(rows)
    normals=np.asarray([r['source_normal']for r in rows]);means=np.asarray([r['current_uv']for r in rows])
    pairs={};point=[]
    for i,r in enumerate(rows):
        if r['operator_kind']=='continuous_cubic_point':point.append(i)
        elif r['operator_kind']=='paint_flank':pairs.setdefault(r['source_pair_id'],[]).append(i)
        else:raise ValueError("unknown frozen actual source operator")
    pair_keys=list(pairs);prepared=[]
    for key in pair_keys:
        ix=pairs[key]
        if [rows[i]['flank_side_index']for i in ix]!=[0,1]:raise ValueError("two-flank column order is not source0/source1")
        prepared+=_prepare([dict(source_measurement_audit=dict(source_material_receipt=rows[ix[0]]['source_material_receipt']))])
    channels=sorted({int(rows[i]['frozen_point_operator']['source_measurement_audit']['profile']['channel'])for i in point}|{p['ch']for p in prepared})
    proof=[]
    def coefficients(image,chs,shadow=False):
        raw=np.stack([_raw_channel(image,ch)for ch in chs]);g,c=backend(raw);result={ch:c[j]for j,ch in enumerate(chs)}
        if shadow:
            original=operator.prepare_cubic_channels(image)
            error=max(float(abs(result[ch]-original[ch]).max())for ch in chs)
            if error!=0:raise ValueError("actual mixed source GPU coefficients differ from CPU")
            proof.append(dict(channels=chs,full_native_coefficient_error=error))
        return result
    def measure(image,coeff,selected):
        shifts=np.empty(n);bound=np.zeros(n,np.int64)
        for i in point:
            if i not in selected:continue
            q=operator.replay_outer_cubic(image,rows[i]['frozen_point_operator'],coefficients=coeff)
            if not q['numeric_valid']:raise ValueError("finite cubic source operator has no numeric definition")
            shifts[i]=q['normal_offset_px'];bound[i]=q['synthetic_window_boundary_count']
        if prepared:
            values,b,_=_measure_pair(image,prepared,coefficients=coeff)
            for k,key in enumerate(pair_keys):
                for side,i in enumerate(pairs[key]):shifts[i]=values[k,side];bound[i]=b[k,side]
        return shifts,bound
    coeff=coefficients(rgb,channels,True);base,bound=measure(rgb,coeff,set(range(n)));replay=[]
    for i,r in enumerate(rows):
        origin=(r['frozen_point_operator']['navigation_seed_uv']if r['operator_kind']=='continuous_cubic_point'else r['source_material_receipt']['original_navigation_seed_uv'])
        replay.append(np.asarray(origin)+base[i]*normals[i])
    base_error=float(abs(np.asarray(replay)-means).max())
    if base_error>1e-9:raise ValueError("mixed source mean is not its exact frozen estimator "+str(base_error))
    cached={};reuse_proof=None;boundary=np.zeros(n,np.int64)
    if board_cache:
        cache=Path(board_cache);rec=json.loads(cache.read_text(encoding='utf8'))
        if rec['seed']!=seed or rec['trials']!=trials or rec['native_receipt']['decoded_native_BGR_sha256']!=native['decoded_native_BGR_sha256']or rec['actual_mean_replay_max_px']!=0:raise ValueError("cached board is another native RNG domain")
        if rec['frozen_operator']['sha256']!=file_sha256(operator.__file__):raise ValueError("cached source estimator differs")
        if file_sha256(rec['replay_arrays']['file'])!=rec['replay_arrays']['sha256']:raise ValueError("cached source samples changed")
        with np.load(rec['replay_arrays']['file'])as z:
            old=list(z['source_row_ids']);samples=(z['normal_shift_delta_px']if 'normal_shift_delta_px'in z else None);oldmeans=z['actual_mean_uv'];oldnormals=z['profile_normals'];oldbounds=z['synthetic_window_boundary_counts']
            for i,r in enumerate(rows):
                # A compact covariance cache has no joint trial columns to reuse.
                # Regenerate every source under the same RGB stream instead.
                if samples is None:break
                if r['operator_kind']!='continuous_cubic_point'or r['source_row_id']not in old:continue
                j=old.index(r['source_row_id'])
                current=object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=r['source_row_id']))
                updated=object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=r['source_row_id'],measurement_operator=r['measurement_operator_resolved']))
                if rec['source_measurements_sha256'][r['source_row_id']]!=current or rec['current_measurements_sha256'][r['source_row_id']]!=updated or not np.array_equal(oldmeans[j],means[i])or not np.array_equal(oldnormals[j],normals[i]):raise ValueError("cached board source mean/operator/normal differs")
                cached[i]=samples[:,j].copy();boundary[i]=oldbounds[j]
        reuse_proof=dict(parent_receipt=dict(file=str(cache),sha256=file_sha256(cache)),rows=len(cached),same_exact_RNG_sequence='SeedSequence([20261018,nativePTS]), full 1080x1920x3 float64 uniform draw once per trial; same official RGB and NumPy pipeline',covariance_cross_terms_computed_from_joint_trial_columns=True)
    selected=set(range(n))-set(cached);needed_channels=sorted({int(rows[i]['frozen_point_operator']['source_measurement_audit']['profile']['channel'])for i in point if i in selected}|{p['ch']for p in prepared})
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(native['native_pts'])]));D=np.empty((trials,n))
    for i,a in cached.items():D[:,i]=a
    for k in range(trials):
        perturbed=rgb.astype(np.float64)+rng.uniform(-.5,.5,rgb.shape)
        coeff=coefficients(perturbed,needed_channels,k==0);values,b=measure(perturbed,coeff,selected)
        for i in selected:D[k,i]=values[i]-base[i];boundary[i]+=b[i]
    factor=(D-D.mean(0)).T/np.sqrt(trials-1);C=factor@factor.T;Wn,white=supported_factor_whitening(factor);B=np.zeros((n,2*n))
    for i,normal in enumerate(normals):B[i,2*i:2*i+2]=normal
    W=Wn@B;W.astype('<f8').tofile(out/'W.f64')
    np.savez_compressed(out/'joint_normal_replay.npz',source_row_ids=np.asarray([r['source_row_id']for r in rows]),normal_shift_delta_px=D,profile_normals=normals,actual_mean_uv=means,normal_covariance_px2=C,synthetic_window_boundary_counts=boundary)
    def hashrow(r,current):
        return object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=r['source_row_id'],measurement_operator=r['measurement_operator_resolved']))if current else object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=r['source_row_id']))
    pairhash={key:object_sha256(dict(native_pts=native['native_pts'],source_row_id=key,actual_flank_uv=[rows[i]['current_uv']for i in pairs[key]],measurement_operator='finite_width_paint_material_flank_pair'))for key in pair_keys}
    eigen=np.linalg.eigvalsh(C)
    rec=dict(source_group_id='actual_mixed_exposure:'+rows[0]['shot_id']+':'+str(native['native_pts']),row_ids=[r['source_row_id']for r in rows],source_family_ids=sorted({r['source_family_id']for r in rows}),native_pts=native['native_pts'],shot_id=rows[0]['shot_id'],status='usable_conditional_joint_native_mixed_source_W',rank=len(W),W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape)),statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',numeric_whitening_verified=True,actual_mean_replay_max_px=base_error,float32_channel_pipeline_error=0.,source_measurements_sha256={r['source_row_id']:hashrow(r,False)for r in rows},current_measurements_sha256={r['source_row_id']:hashrow(r,True)for r in rows},current_pair_measurements_sha256=pairhash,native_receipt=native,frozen_operator=dict(file=str(Path(operator.__file__).resolve()),sha256=file_sha256(operator.__file__)),trials=trials,seed=seed,original_numerical_scheme='whole_native_exposure_joint_board_print_target_and_two_paint_flank_frozen_operators_nonlinear_RGB_replay',conditional_model=dict(encoded_RGB_noise='independent uniform[-.5,+.5] per encoded native RGB pixel/channel; one draw shared by every actual source readout in this exposure; no clipping',selection='actual material/channel/polarity/seed/window frozen, no synthetic source quality truncation',covariance='full empirical MC256; numerical supported factor, no ridge/floor or independent representative noise',tangent='source normal localization covariance only; deterministic finite frozen query must be preserved'),noise_calibration_status='explicit conditional encoded-RGB model; total physical accuracy/source selection/compression nuisance not calibrated',physical_total_error_calibrated=False,shared_full_native_RGB_domain=True,independent_rows_or_PSD_floor_added=False,compute_backend_GPU=True,GPU_coefficient_shadow=proof,reused_same_noise_domain_board= reuse_proof,source_receipts_sha256=object_sha256(rows),replay_arrays=dict(file=str(out/'joint_normal_replay.npz'),sha256=file_sha256(out/'joint_normal_replay.npz')),normal_covariance_eigenvalue_quantiles=np.quantile(eigen,[0,.1,.5,.9,1]).tolist(),normal_std_px_quantiles=np.quantile(np.sqrt(np.diag(C)),[0,.1,.5,.9,1]).tolist(),MC_estimator_bias_px_quantiles=np.quantile(D.mean(0),[0,.1,.5,.9,1]).tolist(),synthetic_window_boundary_count=int(boundary.sum()),proof=white,source_MC_elapsed_seconds=time.perf_counter()-started)
    write_json(out/'receipt.json',rec);return rec

def compile_joint_native_statistics(run,factor_rows,frozen_operator,output,*,gpu_ptx,trials=256,seed=20261018,board_cache_by_pts=None):
    out=Path(output)
    if out.exists():raise ValueError("new immutable joint statistics generation required")
    out.mkdir();frames=defaultdict(list)
    for r in read_jsonl(factor_rows):frames[int(r['native_pts'])].append(r)
    operator=_load_frozen(frozen_operator);records=[];cache=board_cache_by_pts or{}
    with NVRTCSourceFilterCUDA(1080,1920,batch=3,ptx_file=gpu_ptx)as backend:
        for pts,rgb,native in iter_training_native_rgb(run,set(frames)):
            rec=_frame(rgb,native,frames[pts],operator,backend,out/(frames[pts][0]['shot_id']+'_'+str(pts)),trials,seed,cache.get(str(pts)));records.append(rec)
            print(json.dumps(dict(completed_exposures=len(records),native_pts=pts,rows=len(rec['row_ids']),rank=rec['rank'],mean_replay_max=rec['actual_mean_replay_max_px'],seconds=rec['source_MC_elapsed_seconds'],reused_board_rows=(rec['reused_same_noise_domain_board']or{}).get('rows',0))),flush=True)
    write_json(out/'statistics_registry.json',{r['source_group_id']:r for r in records})
    report=dict(schema='rinkfit.mixed_actual_native_source_conditional_statistics.v1',exposures=len(records),factor_rows=sum(len(r['row_ids'])for r in records),rank=sum(r['rank']for r in records),trials=trials,compute_backend_GPU=True,all_sources_share_full_exposure_RGB_noise=True,statistical_admission_or_material_identity_added=False,source_factor_rows=dict(file=str(Path(factor_rows).resolve()),sha256=file_sha256(factor_rows)),statistics_registry=dict(file=str(out/'statistics_registry.json'),sha256=file_sha256(out/'statistics_registry.json')))
    write_json(out/'manifest.json',report);return report
