"""Replay the frozen source profile operator with common native RGB perturbations.

This numerical calibration preserves the original seeds and current means. It
is a quantization hypothesis check, never a physical accuracy certificate.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening,StatisticsUnavailable
OFF=np.arange(-3.,3.0001,.25)

def _reflect(a,n):
    a=np.asarray(a);a=a%(2*n);return np.where(a<n,a,2*n-1-a)

def prepare_parts(rows,rgb):
    seeds=[];normals=[];owners=[];weights=[]
    for i,row in enumerate(rows):
        audit=row['source_measurement_audit'];normal=np.asarray(audit['normal'],float);normal/=np.linalg.norm(normal)
        if row['measurement_operator_resolved']=='paint_center':
            if len(audit.get('flank_uv',[]))!=2:raise ValueError('missing frozen actual two flank source navigation')
            qs=[np.asarray(q)-float(pr['offset_px'])*normal for q,pr in zip(audit['flank_uv'],audit['flank_profiles'])]
        else:qs=[np.asarray(row['source_uv'],float)]
        for q in qs:seeds.append(q);normals.append(normal);owners.append(i);weights.append(1/len(qs))
    seeds=np.asarray(seeds);normals=np.asarray(normals);origin=np.floor(seeds).astype(int)-7
    xx=_reflect(origin[:,0,None]+np.arange(15),rgb.shape[1]);yy=_reflect(origin[:,1,None]+np.arange(15),rgb.shape[0])
    pixel_ids=((yy[:,:,None,None]*rgb.shape[1]+xx[:,None,:,None])*3+np.arange(3)[None,None,None,:])
    unique,inverse=np.unique(pixel_ids,return_inverse=True)
    base=rgb.reshape(-1)[pixel_ids];q=seeds[:,None,:]-origin[:,None,:]+OFF[None,:,None]*normals[:,None,:]
    ij=np.floor(q).astype(int);d=q-ij
    indices=np.stack([ij[:,:,1]*15+ij[:,:,0],ij[:,:,1]*15+ij[:,:,0]+1,(ij[:,:,1]+1)*15+ij[:,:,0],(ij[:,:,1]+1)*15+ij[:,:,0]+1],axis=-1)
    bilinear=np.stack([(1-d[:,:,0])*(1-d[:,:,1]),d[:,:,0]*(1-d[:,:,1]),(1-d[:,:,0])*d[:,:,1],d[:,:,0]*d[:,:,1]],axis=-1)
    return {'seeds':seeds,'normals':normals,'owners':np.asarray(owners),'weights':np.asarray(weights),
        'base':base,'unique_pixel_ids':unique,'inverse':inverse.reshape(pixel_ids.shape),'indices':indices,'bilinear':bilinear,'row_count':len(rows)}

def replay(part,noise=None):
    if noise is None:a=part['base'][None].astype(np.float32)
    else:a=(part['base'][None]+noise[:,part['inverse']]).astype(np.float32)
    # Exactly the source worker float32 channel expressions and Gaussian kernel.
    ch=np.stack((.2126*a[:,:,:,:,0]+.7152*a[:,:,:,:,1]+.0722*a[:,:,:,:,2],a[:,:,:,:,0]-a[:,:,:,:,1],a[:,:,:,:,2]-.5*(a[:,:,:,:,0]+a[:,:,:,:,1])),axis=2)
    ch=gaussian_filter(ch,sigma=(0,0,0,.8,.8));T,P=ch.shape[:2]
    flat=ch.reshape(T,P,3,225)[:,:,:,None,:]
    values=np.take_along_axis(flat,part['indices'][None,:,None,:,:],axis=-1)
    intensity=(values*part['bilinear'][None,:,None,:,:]).sum(-1).astype(np.float32)
    gradient=np.gradient(intensity,.25,axis=-1);channel=np.max(abs(gradient),axis=-1).argmax(axis=2)
    idx=channel[:,:,None,None];g=np.take_along_axis(gradient,idx,axis=2)[:,:,0,:];a=np.take_along_axis(intensity,idx,axis=2)[:,:,0,:]
    v=abs(g);k=v.argmax(axis=-1);ks=np.clip(k,1,23)
    trio=np.take_along_axis(v,ks[:,:,None]+np.arange(-1,2)[None,None,:],axis=-1)
    den=trio[:,:,0]-2*trio[:,:,1]+trio[:,:,2];num=.5*(trio[:,:,0]-trio[:,:,2]);fraction=np.divide(num,den,out=np.zeros_like(num),where=abs(den)>1e-9)
    shift=OFF[k]+.25*np.clip(fraction,-1,1)
    peak=np.take_along_axis(v,k[:,:,None],axis=-1)[:,:,0]
    valid=(peak>=.75)&(np.ptp(a,axis=-1)>=4)&(k!=0)&(k!=24)
    points=part['seeds'][None]+shift[:,:,None]*part['normals'][None]
    uv=np.zeros((T,part['row_count'],2));ok=np.ones((T,part['row_count']),bool)
    for j,(owner,w) in enumerate(zip(part['owners'],part['weights'])):uv[:,owner]+=w*points[:,j];ok[:,owner]&=valid[:,j]
    return uv,ok,{'peak_indices':k,'channels':channel}

def compile_nonlinear_profile_statistics(source_audit,parent_packet,output,*,trials=128,seed=20261018,limit=None):
    source=Path(source_audit);packet_path=Path(parent_packet);packet=json.loads(packet_path.read_text(encoding='utf8'));out=Path(output)
    if out.exists():raise ValueError('new nonlinear statistics generation required')
    out.mkdir(parents=True);lookup={r['source_row_id']:r for r in read_jsonl(source)}
    groups=[g for g in packet['groups'] if g['statistic_kind']!='original']
    if limit is not None:groups=groups[:limit]
    registry={};decisions=[];counts=Counter();cached=None;rgb=None;sample_refs=[]
    for ordinal,group in enumerate(groups):
        ids=[packet['source_row_ids'][i] for i in group['rows']];rows=[lookup[i] for i in ids];gid=group['source_group_id'];record={'source_group_id':gid,'row_ids':ids}
        if any(r.get('manual_reference_preserved') or r.get('point_correspondence') for r in rows):
            record.update(status='pending',reason='manual or entity operator excluded from profile likelihood');decisions.append(record);counts['non_profile_operators']+=1;continue
        png=Path(rows[0]['native_png'])
        if len({r['native_png_sha256'] for r in rows})!=1:raise ValueError('family spans native exposures')
        if cached!=png:
            if file_sha256(png)!=rows[0]['native_png_sha256']:raise ValueError('native source changed')
            rgb=np.asarray(Image.open(png).convert('RGB'));cached=png
        try:part=prepare_parts(rows,rgb);base,good,base_profile=replay(part)
        except (ValueError,KeyError) as e:
            record.update(status='pending',reason='frozen source navigation unavailable: '+str(e));decisions.append(record);counts['navigation_unavailable']+=1;continue
        mean=np.asarray([r['current_uv'] for r in rows]);error=np.linalg.norm(base[0]-mean,axis=1);maxerr=float(error.max(initial=0))
        record.update(native_source_mean_replay_max_error_px=maxerr,native_rgb_unique_samples=len(part['unique_pixel_ids']))
        if not good.all() or maxerr>1e-6:
            record.update(status='pending',reason='frozen native source operator does not replay current mean; no W substitution');decisions.append(record);counts['source_mean_replay_mismatch']+=1;continue
        rng=np.random.default_rng(int.from_bytes(bytes.fromhex(object_sha256({'gid':gid,'seed':seed}))[:8],'little'))
        samples=[];valid=[];switch=0
        for first in range(0,trials,16):
            noise=rng.uniform(-.5,.5,(min(16,trials-first),len(part['unique_pixel_ids'])))
            uv,ok,pr=replay(part,noise);samples.append(uv-mean);valid.append(ok)
            switch+=int(np.sum(pr['peak_indices']!=base_profile['peak_indices']))
        delta=np.concatenate(samples);valid=np.concatenate(valid);record.update(trials=trials,all_valid_trials=int(np.sum(valid.all(axis=1))),peak_branch_switches=switch,
            native_RGB_noise='same source-family actual native RGB Uniform[-0.5,+0.5] LSB before exact frozen original-seed operator',
            current_measurement_means_changed=False,linearized_currentUV_seed_used=False,shared_native_pixels_common=True,
            broader_physical_noise_calibration_complete=False)
        name=f'g{ordinal:05d}_replay.npz';np.savez_compressed(out/name,delta_uv=delta,valid=valid)
        record['nonlinear_samples']={'file':name,'sha256':file_sha256(out/name)}
        if not valid.all():
            record.update(status='pending',reason='nonlinear source family resampling crosses measurement domain');decisions.append(record);counts['nonlinear_domain_crossing']+=1;continue
        normals=np.asarray([r['source_measurement_audit']['normal'] for r in rows],float);normals/=np.linalg.norm(normals,axis=1)[:,None]
        X=np.einsum('tij,ij->ti',delta,normals);factor=(X-X.mean(axis=0)).T/np.sqrt(trials-1)
        try:
            Wn,proof=supported_factor_whitening(factor)
            W=np.empty((len(Wn),2*len(rows)));W[:,0::2]=Wn*normals[:,0];W[:,1::2]=Wn*normals[:,1]
        except (ValueError,StatisticsUnavailable) as e:
            record.update(status='pending',reason=str(e));decisions.append(record);counts['unsupported_covariance']+=1;continue
        wp=out/f'g{ordinal:05d}_W.f64';W.astype('<f8').tofile(wp)
        record.update(status='nonlinear_quantization_initialization_W_ready',rank=len(W),W={'file':str(wp.resolve()),'sha256':file_sha256(wp),'dtype':'<f8','shape':list(W.shape)},
            statistic_kind='initialization_surrogate',original_numerical_scheme='same_family_shared_RGB_exact_source_operator_resampling',
            numeric_whitening_verified=True,proof=proof,original_W_claimed=False,PSD_floor_or_ridge_used=False,
            source_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'source_uv':r['source_uv'],'source_row_id':r['source_row_id']}) for r in rows},
            current_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'current_uv':r['current_uv'],'source_row_id':r['source_row_id'],'measurement_operator':r['measurement_operator_resolved']}) for r in rows},
            native_png={'file':str(png),'sha256':rows[0]['native_png_sha256']},bootstrap_mean_delta_px=delta.mean(axis=0).tolist(),
            old_linearized_W=group['W'],noise_calibration_status='actual frozen seed nonlinear quantization completed; physical source-noise calibration pending')
        registry[gid]=record;decisions.append(record);counts['groups_ready']+=1;counts['rows_ready']+=len(rows);counts['rank']+=len(W)
        if ordinal%25==0:write_json(out/'progress.json',{'groups_done':ordinal+1,'groups_total':len(groups),'counts':dict(counts)})
    write_json(out/'statistics_registry.json',registry);write_json(out/'decisions.json',decisions)
    result={'schema':'rinkfit.native_frozen_operator_nonlinear_quantization.v1','counts':dict(counts),'groups_total':len(groups),'trials':trials,
        'source_audit':{'file':str(source),'sha256':file_sha256(source)},'parent_packet':{'file':str(packet_path),'sha256':file_sha256(packet_path)},
        'physical_noise_accuracy_or_formal_statistical_optimum_claimed':False,'shared_between_family_RGB_covariance_complete':False,
        'source_UV_or_current_mean_changed':False,'PSD_floor_or_ridge_used':False}
    write_json(out/'manifest.json',result);return result


def recompile_normal_statistics(parent_generation,source_audit,output):
    """Use the same joint RGB trials, retaining curve normal support only."""
    parent=Path(parent_generation);out=Path(output)
    if out.exists():raise ValueError('new normal-support generation required')
    out.mkdir(parents=True);lookup={r['source_row_id']:r for r in read_jsonl(source_audit)}
    decisions=json.loads((parent/'decisions.json').read_text(encoding='utf8'));registry={};records=[];counts=Counter()
    for ordinal,old in enumerate(decisions):
        record={k:v for k,v in old.items() if k not in {'rank','W','proof','numeric_whitening_verified'}}
        spec=old.get('nonlinear_samples')
        if not spec:
            records.append(record);counts['missing_replayed_operator']+=1;continue
        sample=Path(spec['file']);sample=sample if sample.is_absolute() else parent/sample
        if file_sha256(sample)!=spec['sha256']:raise ValueError('nonlinear sample batch changed')
        record['nonlinear_samples']={**spec,'file':str(sample)}
        with np.load(sample,allow_pickle=False) as z:delta=z['delta_uv'];valid=z['valid']
        if not valid.all():
            records.append(record);counts['nonlinear_domain_crossing']+=1;continue
        ids=record['row_ids'];rows=[lookup[rid] for rid in ids];normals=np.asarray([r['source_measurement_audit']['normal'] for r in rows],float);normals/=np.linalg.norm(normals,axis=1)[:,None]
        X=np.einsum('tij,ij->ti',delta,normals);factor=(X-X.mean(axis=0)).T/np.sqrt(len(X)-1)
        sd=np.std(X,axis=0,ddof=1);record['normal_standard_deviation_px']=sd.tolist()
        record['source_operator_resolution']={'native_profile_sampling_step_px':.25,'bilinear_original_pixel_lattice_px':1.,'peak_fraction_clip':[-1.,1.],
            'diagnostic_finite_normal_bin_hypothesis_px':[-.125,.125],
            'finite_bin_hypothesis_used_as_W_or_covariance_floor':False,
            'meaning':'grid-locked machine output is not an infinitely precise physical material boundary; a finite-bin physical model remains explicit and uncalibrated'}
        record['normal_only_support']=True;record['unmeasured_tangential_roundoff_whitened']=False
        try:
            Wn,proof=supported_factor_whitening(factor);W=np.empty((len(Wn),2*len(rows)));W[:,0::2]=Wn*normals[:,0];W[:,1::2]=Wn*normals[:,1]
        except (ValueError,StatisticsUnavailable) as e:
            record.update(status='pending',reason=str(e));records.append(record);counts['unsupported_normal_covariance']+=1;continue
        path=out/f'g{ordinal:05d}_W.f64';W.astype('<f8').tofile(path)
        record.update(status='nonlinear_quantization_initialization_W_ready',rank=len(W),W={'file':str(path.resolve()),'sha256':file_sha256(path),'dtype':'<f8','shape':list(W.shape)},proof=proof,
            source_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'source_uv':r['source_uv'],'source_row_id':r['source_row_id']}) for r in rows},
            current_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'current_uv':r['current_uv'],'source_row_id':r['source_row_id'],'measurement_operator':r['measurement_operator_resolved']}) for r in rows},
            native_png={'file':rows[0]['native_png'],'sha256':rows[0]['native_png_sha256']},numeric_whitening_verified=True,
            statistic_kind='initialization_surrogate',original_numerical_scheme='same_family_shared_RGB_exact_original_seed_normal_operator_resampling',
            original_W_claimed=False,PSD_floor_or_ridge_used=False,noise_calibration_status='normal-only exact native operator nonlinear quantization replay; physical noise/finite-bin and shared-family nuisance pending')
        registry[record['source_group_id']]=record;records.append(record);counts['groups_ready']+=1;counts['rows_ready']+=len(ids);counts['rank']+=len(W)
    write_json(out/'statistics_registry.json',registry);write_json(out/'decisions.json',records)
    result={'schema':'rinkfit.native_nonlinear_normal_support_replay.v1','counts':dict(counts),'groups_total':len(decisions),
        'source_nonlinear_generation':{'file':str(parent/'manifest.json'),'sha256':file_sha256(parent/'manifest.json')},
        'source_audit':{'file':str(source_audit),'sha256':file_sha256(source_audit)},'actual_RGB_resampling_repeated':False,
        'same_samples_reused':True,'unmeasured_tangential_float_roundoff_support_discarded':True,
        'PSD_floor_or_ridge_used':False,'source_mean_changed':False,'formal_physical_noise_or_accuracy_claimed':False}
    write_json(out/'manifest.json',result);return result
