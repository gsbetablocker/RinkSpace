"""Joint two-flank native-exposure covariance for actual finite-width paint.

The two material interfaces are measured; an image arithmetic midpoint is only
navigation. Conditional encoded-RGB perturbations share the entire exposure.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,concurrent.futures
import numpy as np
from PIL import Image
from scipy.ndimage import spline_filter,map_coordinates
from .paint_statistics import _load_frozen,_one_channel,_prepare
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening,StatisticsUnavailable

OPERATOR='finite_width_paint_material_flank'
def _measure_pair(rgb,prepared,*,coefficients=None):
    profiles=[None]*len(prepared)
    for ch in sorted({p['ch']for p in prepared}):
        indices=[i for i,p in enumerate(prepared)if p['ch']==ch];q=np.concatenate([prepared[i]['coords']for i in indices]);c=coefficients[ch] if coefficients is not None else spline_filter(_one_channel(rgb,ch),order=3)
        sampled=map_coordinates(c,[q[:,1],q[:,0]],order=3,mode='nearest',prefilter=False);at=0
        for i in indices:n=len(prepared[i]['off']);profiles[i]=sampled[at:at+n];at+=n
    positions=[];boundaries=[];peaks=[]
    for p,values in zip(prepared,profiles):
        g=np.gradient(values,.25);flanks=[];boundary=[];peak=[]
        for ix,polarity in p['windows']:
            signed=polarity*g;k=int(ix[np.argmax(signed[ix])]);boundary.append(int(k in (ix[0],ix[-1])));peak.append(k);shift=0.
            if 0<k<len(p['off'])-1:
                v=signed[k-1:k+2];den=v[0]-2*v[1]+v[2]
                if abs(den)>1e-10:shift=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
            flanks.append(float(p['off'][k]+shift))
        positions.append(flanks);boundaries.append(boundary);peaks.append(peak)
    return np.asarray(positions),np.asarray(boundaries),np.asarray(peaks)

def _frame(job):
    rows=job['rows'];out=Path(job['out']);out.mkdir();frozen=Path(job['frozen']);operator=_load_frozen(frozen);png=Path(rows[0]['native_png'])
    if file_sha256(png)!=rows[0]['native_png_sha256']:raise ValueError('changed actual native PNG')
    rgb=np.asarray(Image.open(png).convert('RGB'));prepared=_prepare(rows);base,bound,base_peaks=_measure_pair(rgb,prepared)
    uv=np.asarray([[p['seed']+value*p['n']for value in a]for p,a in zip(prepared,base)]);means=np.asarray([r['source_measurement_audit']['flank_uv']for r in rows]);delta=np.linalg.norm(uv-means,axis=2)
    middle=np.asarray([p['seed']+.5*sum(a)*p['n']for p,a in zip(prepared,base)]);mid_error=float(np.max(np.linalg.norm(middle-np.asarray([r['current_uv']for r in rows]),axis=1)))
    replay=operator.replay_paint_midline(rgb,rows[0]['source_measurement_audit']['source_material_receipt']);direct_error=float(np.linalg.norm(np.asarray(replay['current_uv'])-middle[0]));channel_error=max(float(np.max(abs(operator._channels(rgb)[ch]-_one_channel(rgb,ch))))for ch in {p['ch']for p in prepared})
    if delta.max(initial=0)>1e-9 or mid_error>1e-9 or direct_error>1e-12 or channel_error!=0:raise ValueError('paired actual source mean/operator replay mismatch')
    n=len(rows);trials=job['trials'];rng=np.random.default_rng(np.random.SeedSequence([job['seed'],int(rows[0]['native_pts'])]));samples=np.empty((trials,n,2));all_peaks=np.empty((trials,n,2),np.uint16);boundary_counts=np.zeros((n,2),np.int64)
    for trial in range(trials):
        perturbed=rgb.astype(np.float64)+rng.uniform(-.5,.5,rgb.shape);position,bound,peaks=_measure_pair(perturbed,prepared);samples[trial]=position-base;all_peaks[trial]=peaks;boundary_counts+=bound
    scalar=samples.reshape(trials,2*n);factor=(scalar-scalar.mean(0)).T/np.sqrt(trials-1);C=factor@factor.T;B=np.zeros((2*n,4*n))
    for i,p in enumerate(prepared):
        for side in range(2):B[2*i+side,4*i+2*side:4*i+2*side+2]=p['n']
    status='usable_conditional_joint_native_two_flank_W';W=None;proof=None
    try:Wn,proof=supported_factor_whitening(factor);W=Wn@B;W.astype('<f8').tofile(out/'W.f64')
    except StatisticsUnavailable as exc:status='no_positive_supported_covariance_not_infinite_precision';proof=dict(reason=str(exc))
    midpoint=samples.mean(axis=2);midfactor=(midpoint-midpoint.mean(0)).T/np.sqrt(trials-1);midcov=midfactor@midfactor.T
    np.savez_compressed(out/'joint_flank_replay.npz',source_pair_row_ids=np.asarray([r['source_row_id']for r in rows]),normal_shift_delta_px=samples,profile_normals=np.asarray([p['n']for p in prepared]),flank_mean_uv=means,normal_covariance_px2=C,full_uv_covariance_4d_px2=B.T@C@B,midpoint_normal_covariance_px2=midcov,synthetic_window_boundary_counts=boundary_counts,synthetic_peak_indices=all_peaks,base_peak_indices=base_peaks)
    factor_ids=[r['source_row_id']+'/native_flank/'+str(side)for r in rows for side in range(2)];source_hash={};current_hash={};pair_hash={};factor_rows=[]
    for i,r in enumerate(rows):
        pair_hash[r['source_row_id']]=object_sha256(dict(native_pts=r['native_pts'],source_row_id=r['source_row_id'],actual_flank_uv=means[i].tolist(),measurement_operator='finite_width_paint_material_flank_pair'))
        for side in range(2):
            rid=factor_ids[2*i+side];q=means[i,side].tolist();source_hash[rid]=object_sha256(dict(native_pts=r['native_pts'],source_uv=q,source_row_id=rid));current_hash[rid]=object_sha256(dict(native_pts=r['native_pts'],current_uv=q,source_row_id=rid,measurement_operator=OPERATOR))
            factor_rows.append(dict(source_row_id=rid,original_source_row_id=r['source_row_id'],original_source_uv=r['source_uv'],native_pts=r['native_pts'],shot_id=r['shot_id'],source_family_id=r['source_family_id'],target_id=r['target_id'],joint_material_id=r.get('joint_material_id'),source_uv=q,current_uv=q,measurement_operator_resolved=OPERATOR,flank_side_index=side,source_normal=prepared[i]['n'].tolist(),pair_is_unoriented_same_stroke_flanks=True,side_index_is_world_inner_outer_sign=False,native_png_sha256=r['native_png_sha256'],source_pair_measurement_sha256=pair_hash[r['source_row_id']],source_material_receipt_sha256=object_sha256(r['source_measurement_audit']['source_material_receipt']),physical_width_is_source_nuisance_not_engineering_E=True))
    with (out/'factor_rows.jsonl').open('w',encoding='utf8')as f:
        for r in factor_rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    gid='physical_paint_pair_exposure:'+rows[0]['shot_id']+':'+str(rows[0]['native_pts']);std=np.sqrt(np.diag(C));switch=np.sum(all_peaks!=base_peaks[None,:,:],axis=0)
    rec=dict(source_group_id=gid,row_ids=factor_ids,source_pair_row_ids=[r['source_row_id']for r in rows],source_family_ids=sorted({r['source_family_id']for r in rows}),native_pts=rows[0]['native_pts'],shot_id=rows[0]['shot_id'],status=status,rank=len(W)if W is not None else 0,
        W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape))if W is not None else None,
        statistic_kind='native_joint_readout',original_numerical_scheme='full_native_exposure_shared_RGB_frozen_two_actual_material_flanks_nonlinear_replay',numeric_whitening_verified=W is not None,original_W_claimed=False,
        source_measurements_sha256=source_hash,current_measurements_sha256=current_hash,current_pair_measurements_sha256=pair_hash,original_source_family_sha256=object_sha256([dict(source_row_id=r['source_row_id'],source_uv=r['source_uv'],native_pts=r['native_pts'])for r in rows]),proof=proof,
        actual_two_flank_base_replay_max_px=float(delta.max(initial=0)),midpoint_base_replay_max_px=mid_error,frozen_original_midpoint_reference_error_px=direct_error,float32_channel_pipeline_error=channel_error,trials=trials,
        normal_std_px_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),synthetic_window_boundary_count=int(boundary_counts.sum()),synthetic_peak_grid_index_switch_count=int(switch.sum()),
        width_is_observation_source_nuisance=True,old_midpoint_rows_must_not_be_counted_together=True,side_index_is_not_world_inner_outer_sign=True,physical_total_error_calibrated=False,
        shared_full_native_RGB_domain=True,independent_rows_or_PSD_floor_added=False,tangent_information_from_navigation_seed_claimed=False,
        noise_calibration_status='usable conditional encoded-RGB quantization covariance; actual nonlinear same-mean operator; broader physical error/model bias/selection uncertainty not calibrated',
        conditional_model=dict(RGB_perturbation='independent uniform[-.5,+.5] per encoded native pixel/channel; one same full-frame draw shared by all actual source flanks, no clipping',material_seed_channel_polarity_window_selection='frozen to actual native source; no synthetic reselection/quality censoring',MC_covariance='128 joint trials; numerical support only, absent modes carry no precision',normal_only='fixed query tangent is not assigned source localization precision'),
        factor_rows=dict(file=str(out/'factor_rows.jsonl'),sha256=file_sha256(out/'factor_rows.jsonl')),replay_arrays=dict(file=str(out/'joint_flank_replay.npz'),sha256=file_sha256(out/'joint_flank_replay.npz')),
        source_receipts_sha256=object_sha256([r['source_measurement_audit']['source_material_receipt']for r in rows]),frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)))
    write_json(out/'receipt.json',rec);return rec

def compile_paint_pair_statistics(audit_dir,frozen_operator,output,roles_file,*,trials=128,seed=20261018,workers=2):
    audit=Path(audit_dir);frozen=Path(frozen_operator).resolve();out=Path(output);rolespath=Path(roles_file).resolve();roles=json.loads(rolespath.read_text())['frame_roles']
    if out.exists():raise ValueError('new immutable generation required')
    out.mkdir(parents=True);frames=defaultdict(list)
    for r in read_jsonl(audit/'relocated_rows.jsonl.gz'):
        if r.get('semantic_local_material_operator_passed')and r.get('measurement_operator_resolved')=='paint_center'and not r.get('manual_reference_preserved'):
            role=roles.get(str(r['native_pts']),{});receipt=(r.get('source_measurement_audit')or{}).get('source_material_receipt')
            if not receipt or role.get('role')not in ('development_candidate','development')or role.get('shot_id')!=r['shot_id']:continue
            if receipt['measured_source_mean']!=r['current_uv']:raise ValueError('actual source mean/receipt mismatch')
            frames[(r['shot_id'],int(r['native_pts']))].append(r)
    byshot=defaultdict(list)
    for key,rows in sorted(frames.items()):byshot[key[0]].append((key,rows))
    ordered=[byshot[s][j]for j in range(max(map(len,byshot.values())))for s in sorted(byshot)if j<len(byshot[s])]
    jobs=[dict(rows=rows,out=str(out/(shot+'_'+str(native))),frozen=str(frozen),trials=trials,seed=seed)for(shot,native),rows in ordered];records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for rec in pool.map(_frame,jobs):records.append(rec);print(json.dumps(dict(completed_exposures=len(records),source_pairs=sum(len(x['source_pair_row_ids'])for x in records),shots=sorted({x['shot_id']for x in records}),native_pts=rec['native_pts'],rank=rec['rank'])),flush=True)
    registry={r['source_group_id']:r for r in records if r.get('W')};write_json(out/'statistics_registry.json',registry)
    report=dict(schema='rinkfit.joint_native_exposure_two_flank_conditional_statistics.v1',source_generation=dict(file=str(audit/'relocated_rows.jsonl.gz'),sha256=file_sha256(audit/'relocated_rows.jsonl.gz')),frame_roles=dict(file=str(rolespath),sha256=file_sha256(rolespath)),source_pairs=sum(len(r['source_pair_row_ids'])for r in records),factor_rows=sum(len(r['row_ids'])for r in records),exposures=len(records),rank=sum(r['rank']for r in records),trials=trials,old_source_UV_or_midpoint_W_changed=False,midpoint_W_duplicated_with_two_flanks=False,conditional_native_joint_statistics_usable=True,physical_total_noise_accuracy_claimed=False,statistics_registry_sha256=file_sha256(out/'statistics_registry.json'))
    write_json(out/'manifest.json',report);return report
