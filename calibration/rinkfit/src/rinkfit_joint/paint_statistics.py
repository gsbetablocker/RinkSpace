"""Same-mean frozen physical paint operator, joint native-exposure replay.

Finite Monte Carlo quantization covariance is conditional initialization evidence;
source selection, compression and broader physical noise remain uncalibrated.
"""
from pathlib import Path
from collections import defaultdict,Counter
import importlib.util,json,concurrent.futures,time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,spline_filter,map_coordinates
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening,StatisticsUnavailable

def _load_frozen(path):
    spec=importlib.util.spec_from_file_location('frozen_paint_replay',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def _one_channel(rgb,ch):
    a=rgb.astype(np.float32)
    v=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]) if ch==0 else (a[:,:,0]-a[:,:,1] if ch==1 else a[:,:,2]-.5*(a[:,:,0]+a[:,:,1]))
    return gaussian_filter(v,sigma=.8)

def _prepare(rows):
    prepared=[]
    for row in rows:
        re=row['source_measurement_audit']['source_material_receipt'];off=np.asarray(re['normal_offsets'],float);seed=np.asarray(re['original_navigation_seed_uv'],float);n=np.asarray(re['normal'],float)
        if abs(np.linalg.norm(n)-1)>1e-9:raise ValueError('source profile normal is not unit')
        windows=[]
        for window in re['frozen_flank_windows'][2]:
            lo,hi=window['normal_window_offsets'];ix=np.flatnonzero((off>=lo-1e-8)&(off<=hi+1e-8));windows.append((ix,int(window['polarity'])))
        prepared.append(dict(seed=seed,n=n,off=off,ch=int(re['channel']),coords=seed+off[:,None]*n,windows=windows))
    return prepared

def _measure(rgb,prepared):
    profiles=[None]*len(prepared)
    for ch in sorted({p['ch']for p in prepared}):
        indices=[i for i,p in enumerate(prepared)if p['ch']==ch];q=np.concatenate([prepared[i]['coords']for i in indices]);c=spline_filter(_one_channel(rgb,ch),order=3)
        sampled=map_coordinates(c,[q[:,1],q[:,0]],order=3,mode='nearest',prefilter=False);at=0
        for i in indices:n=len(prepared[i]['off']);profiles[i]=sampled[at:at+n];at+=n
    positions=[];boundaries=[]
    for p,values in zip(prepared,profiles):
        g=np.gradient(values,.25);flanks=[];boundary=0
        for ix,polarity in p['windows']:
            signed=polarity*g;k=int(ix[np.argmax(signed[ix])]);boundary+=int(k in (ix[0],ix[-1]));shift=0.
            if 0<k<len(p['off'])-1:
                v=signed[k-1:k+2];den=v[0]-2*v[1]+v[2]
                if abs(den)>1e-10:shift=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
            flanks.append(float(p['off'][k]+shift))
        positions.append(.5*sum(flanks));boundaries.append(boundary)
    return np.asarray(positions),np.asarray(boundaries)

def _frame(job):
    rows=job['rows'];out=Path(job['out']);out.mkdir();frozen=Path(job['frozen']);operator=_load_frozen(frozen);png=Path(rows[0]['native_png'])
    if job.get('reuse_receipt'):
        previous=Path(job['reuse_receipt']);record=json.loads(previous.read_text(encoding='utf8'))
        expected=[r['source_row_id']for r in rows]
        receipts=object_sha256([r['source_measurement_audit']['source_material_receipt']for r in rows])
        if record['row_ids']!=expected or record['source_receipts_sha256']!=receipts or record['frozen_operator']['sha256']!=file_sha256(frozen):raise ValueError('replay reuse differs from current actual source operator/mean')
        if record.get('trials')!=job['trials']:raise ValueError('replay trial count differs')
        if file_sha256(png)!=rows[0]['native_png_sha256'] or (record.get('W')and file_sha256(record['W']['file'])!=record['W']['sha256']):raise ValueError('reused exact exposure artifact changed')
        record={**record,'reused_immutable_exposure_receipt':dict(file=str(previous),sha256=file_sha256(previous))}
        write_json(out/'receipt.json',record);return record
    if file_sha256(png)!=rows[0]['native_png_sha256']:raise ValueError('changed actual native PNG')
    if len({(r['native_png_sha256'],r['native_pts']) for r in rows})!=1:raise ValueError('mixed native exposure')
    rgb=np.asarray(Image.open(png).convert('RGB'));prepared=_prepare(rows);base,bound=_measure(rgb,prepared);uv=np.asarray([p['seed']+a*p['n']for p,a in zip(prepared,base)]);means=np.asarray([r['current_uv']for r in rows]);delta=np.linalg.norm(uv-means,axis=1)
    replay=operator.replay_paint_midline(rgb,rows[0]['source_measurement_audit']['source_material_receipt']);direct_error=float(np.linalg.norm(np.asarray(replay['current_uv'])-uv[0]));channel_error=max(float(np.max(abs(operator._channels(rgb)[ch]-_one_channel(rgb,ch)))) for ch in {p['ch']for p in prepared})
    if float(delta.max(initial=0.))>1e-9 or direct_error>1e-12 or channel_error!=0:raise ValueError('frozen original-seed source mean not bit-consistent '+str((delta.max(),direct_error,channel_error)))
    n=len(rows);trials=job['trials'];rng=np.random.default_rng(np.random.SeedSequence([job['seed'],int(rows[0]['native_pts'])]));samples=np.empty((trials,n));boundary_counts=np.zeros(n,np.int64)
    for trial in range(trials):
        perturbed=rgb.astype(np.float64)+rng.uniform(-.5,.5,rgb.shape);position,bound=_measure(perturbed,prepared);samples[trial]=position-base;boundary_counts+=bound
    factor=(samples-samples.mean(0)).T/np.sqrt(trials-1);C=factor@factor.T;diagonal=np.diag(C);std=np.sqrt(diagonal);status='conditional_same_mean_quantization_initialization_W';W=None;proof=None
    try:
        Wn,proof=supported_factor_whitening(factor);B=np.zeros((n,2*n))
        for i,p in enumerate(prepared):B[i,2*i:2*i+2]=p['n']
        W=Wn@B;W.astype('<f8').tofile(out/'W.f64')
    except StatisticsUnavailable as exc:status='no_positive_quantization_variance_support_not_infinite_information';proof=dict(reason=str(exc))
    np.savez_compressed(out/'shared_native_rgb_replay.npz',row_ids=np.asarray([r['source_row_id']for r in rows]),normal_shift_delta_px=samples,profile_normals=np.asarray([p['n']for p in prepared]),mean_uv=means,normal_covariance_px2=C,synthetic_window_boundary_counts=boundary_counts)
    sd=np.sqrt(np.outer(diagonal,diagonal));corr=np.divide(C,sd,out=np.zeros_like(C),where=sd>0);different=[];same=[]
    for i in range(n):
      for j in range(i):
        (same if rows[i]['source_family_id']==rows[j]['source_family_id'] else different).append(abs(float(corr[i,j])))
    gid='physical_paint_exposure:'+rows[0]['shot_id']+':'+str(rows[0]['native_pts']);hashes={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=r['source_row_id'],measurement_operator=r['measurement_operator_resolved']))for r in rows}
    rec=dict(source_group_id=gid,row_ids=[r['source_row_id']for r in rows],native_pts=rows[0]['native_pts'],shot_id=rows[0]['shot_id'],source_family_ids=sorted({r['source_family_id']for r in rows}),status=status,rank=len(W) if W is not None else 0,
        W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape))if W is not None else None,
        statistic_kind='initialization_surrogate',original_numerical_scheme='joint_whole_native_exposure_frozen_physical_paint_operator_RGB_quantization_replay',numeric_whitening_verified=W is not None,original_W_claimed=False,
        source_measurements_sha256={r['source_row_id']:object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=r['source_row_id']))for r in rows},current_measurements_sha256=hashes,
        original_source_family_sha256=object_sha256([dict(source_row_id=r['source_row_id'],native_pts=r['native_pts'],source_uv=r['source_uv'])for r in rows]),proof=proof,
        original_seed_mean_replay_max_px=float(delta.max(initial=0.)),direct_reference_replay_error_px=direct_error,float32_channel_pipeline_max_error=channel_error,
        trials=trials,source_mean_unchanged=True,normal_std_px_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),synthetic_window_boundary_count=int(boundary_counts.sum()),
        shared_RGB_same_family_abs_correlation_max=max(same,default=0),shared_RGB_cross_family_abs_correlation_max=max(different,default=0),shared_RGB_cross_family_pair_count=len(different),
        shared_frame_pixel_noise_preserved=True,shared_two_flank_pixel_noise_preserved=True,PSD_floor_or_ridge_used=False,tangential_information_from_navigation_seed_claimed=False,
        noise_calibration_status='conditional_quantization_model_only; source_selection/compression/photometric/shared_nuisance_not_calibrated; no formal noise accuracy claim',
        assumptions=dict(native_encoded_RGB_perturbation='independent uniform [-0.5,+0.5] per source pixel/channel, same draw shared by all measured rows of this exposure; no clipping',source_material_and_window_selection='conditioned on actual source; no synthetic material reselection or quality truncation',cubic_spline_support='full native exposure; no independent local-bbox approximation',finite_sample_rank='measured covariance support only; no zero mode is treated as exact high confidence'),
        physical_material_identity_promoted_by_statistics=False,source_receipts_sha256=object_sha256([r['source_measurement_audit']['source_material_receipt']for r in rows]),frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)))
    write_json(out/'receipt.json',rec);return rec

def compile_paint_statistics(audit_dir,frozen_operator,output,*,trials=128,seed=20261018,workers=2,roles_file=None,reuse_generations=()):
    audit=Path(audit_dir);frozen=Path(frozen_operator).resolve();out=Path(output)
    if out.exists():raise ValueError('new immutable generation required')
    if roles_file is None:raise ValueError('explicit immutable native frame-role contract is required before RGB access')
    rolepath=Path(roles_file).resolve();roles=json.loads(rolepath.read_text(encoding='utf8'))['frame_roles']
    out.mkdir(parents=True);families=defaultdict(list);excluded=Counter()
    for r in read_jsonl(audit/'relocated_rows.jsonl.gz'):
        if r.get('semantic_local_material_operator_passed') and r.get('measurement_operator_resolved')=='paint_center' and not r.get('manual_reference_preserved'):
            role=roles.get(str(r['native_pts']),{})
            if role.get('role') not in ('development_candidate','development') or role.get('shot_id')!=r['shot_id']:
                excluded['protected_or_unresolved_native_frame_role']+=1;continue
            re=(r.get('source_measurement_audit')or{}).get('source_material_receipt')
            if not re:excluded['passed_flag_has_no_same_mean_frozen_receipt']+=1;continue
            if re['measured_source_mean']!=r['current_uv']:raise ValueError('actual frozen receipt/current mean mismatch')
            if re.get('source_row_id')!=r['source_row_id']or re.get('native_pts')!=r['native_pts']:raise ValueError('source receipt row identity differs')
            families[(r['shot_id'],int(r['native_pts']))].append(r)
    reusable={}
    for generation in map(Path,reuse_generations):
        for ref in generation.glob('*/receipt.json'):
            record=json.loads(ref.read_text(encoding='utf8'));key=(record['shot_id'],int(record['native_pts']))
            if key in families and record['row_ids']==[r['source_row_id']for r in families[key]]:reusable[key]=str(ref.resolve())
    jobs=[dict(rows=rows,out=str(out/(shot+'_'+str(native))),frozen=str(frozen),trials=trials,seed=seed,reuse_receipt=reusable.get((shot,native)))for(shot,native),rows in sorted(families.items())];records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for record in pool.map(_frame,jobs):
            records.append(record);print(json.dumps(dict(completed_exposures=len(records),reused_exact_exposures=sum('reused_immutable_exposure_receipt'in r for r in records),rows=sum(len(r['row_ids'])for r in records),native_pts=record['native_pts'],rank=record['rank'])),flush=True)
    registry={r['source_group_id']:r for r in records if r['W']is not None};write_json(out/'statistics_registry.json',registry)
    report=dict(schema='rinkfit.whole_native_exposure_same_mean_physical_paint_quantization_replay.v1',input=dict(file=str(audit/'relocated_rows.jsonl.gz'),sha256=file_sha256(audit/'relocated_rows.jsonl.gz')),frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)),exposures=len(records),rows=sum(len(r['row_ids'])for r in records),rank=sum(r['rank']for r in records),trials=trials,excluded=dict(excluded),frame_roles=dict(file=str(rolepath),sha256=file_sha256(rolepath)),source_mean_max_replay_px=max((r['original_seed_mean_replay_max_px']for r in records),default=0),source_UV_or_current_mean_changed=False,full_frame_shared_pixel_draws=True,physical_source_selection_noise_or_formal_accuracy_claimed=False,statistics_registry_sha256=file_sha256(out/'statistics_registry.json'))
    write_json(out/'manifest.json',report);return report
