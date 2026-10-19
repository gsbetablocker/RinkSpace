"""Joint two-edge native resampling for fixed entity corner uncertainty.

The whole RGB patch is perturbed once per trial; both incident material edges
are remeasured together. A scalar normal W can never replace a 2-D entity.
"""
from pathlib import Path
from collections import Counter
import argparse,json,gzip
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .entity_corners import remeasure_corner_from_patch
from .source_statistics import supported_whitening,StatisticsUnavailable

def compile_corner_statistics(source_audit,entity_dir,output,*,trials=128,seed=20261018,operator="two_edge"):
    if operator=="two_edge":measure=remeasure_corner_from_patch
    elif operator=="reacquired":
        from .entity_reacquire import remeasure_reacquired_corner
        measure=remeasure_reacquired_corner
    elif operator=="fixed":
        from .fixed_entity_source import remeasure_fixed_entity
        measure=remeasure_fixed_entity
    else:raise ValueError("unknown source corner operator")
    audit=Path(source_audit);entity=Path(entity_dir);out=Path(output)
    if out.exists():raise ValueError('new corner statistics generation required')
    out.mkdir(parents=True)
    rows={r['source_row_id']:r for r in read_jsonl(audit/'relocated_rows.jsonl.gz')}
    metadata=json.loads((entity/'patch_index.json').read_text(encoding='utf8'))
    with np.load(entity/'corner_patches.npz',allow_pickle=False) as z:RGB=z['RGB'];masks=z['mask']
    base_admission={r['source_row_id']:r.get('source_admitted_before_synthetic_sampling') for r in read_jsonl(entity/'fit_receipts.jsonl.gz')} if operator=='fixed' else {}
    registry={};decisions=[];counts=Counter();delta_arrays=[];valid_arrays=[];row_ids=[]
    for receipt in metadata:
        rid=receipt['source_row_id'];row=rows[rid];index=receipt['patch_index'];record={'source_row_id':rid,'native_pts':row['native_pts'],'source_family_id':row['source_family_id']}
        if operator=='fixed' and base_admission.get(rid) is not True:
            record.update(status='pending',reason='fixed material source window not admitted before synthetic sampling');decisions.append(record);counts['fixed_base_source_window_pending']+=1;continue
        if receipt['manual_reference_preserved']:
            record.update(status='pending',reason='manual entity click operator has no automatic-intersection covariance; original UV preserved');decisions.append(record);counts['manual_entities_preserved']+=1;continue
        if row.get('current_uv') is None or row['source_measurement_audit']['status']!='entity_corner_two_direction_support':
            record.update(status='pending',reason='no accepted two-edge current source mean');decisions.append(record);counts['missing_two_edge_source']+=1;continue
        rgb=RGB[index];mask=masks[index];base=measure(rgb,receipt,mask)
        if not base['valid']:
            record.update(status='pending',reason='saved native patch does not replay source operator');decisions.append(record);counts['patch_replay_failures']+=1;continue
        origin=np.asarray(receipt['native_origin_uv']);base_uv=origin+base['current_intersection_uv'];mean_uv=np.asarray(row['current_uv'])
        difference=float(np.linalg.norm(base_uv-mean_uv))
        if difference>1e-6:
            record.update(status='pending',reason='patch replay and current source mean differ',difference_px=difference);decisions.append(record);counts['source_mean_replay_mismatch']+=1;continue
        stable_seed=int.from_bytes(bytes.fromhex(object_sha256({'row_id':rid,'seed':seed}))[:8],'little');rng=np.random.default_rng(stable_seed)
        samples=np.full((trials,2),np.nan);valid=np.zeros(trials,bool);boundary_hits=0;max_numeric_iterations=0;max_numeric_residual=0.
        for trial in range(trials):
            perturbed=rgb.astype(float)+rng.uniform(-.5,.5,rgb.shape)
            answer=measure(perturbed,receipt,mask)
            boundary_hits+=int(answer.get('profile_window_boundary_count',0));max_numeric_iterations=max(max_numeric_iterations,int(answer.get('numeric_iterations',0)))
            residual=answer.get('numeric_equation_residual_px')
            if residual is not None:max_numeric_residual=max(max_numeric_residual,float(residual))
            if answer['valid']:
                samples[trial]=origin+answer['current_intersection_uv']-mean_uv;valid[trial]=True
        row_ids.append(rid);delta_arrays.append(samples);valid_arrays.append(valid)
        successes=int(valid.sum());record.update(native_patch_replay_difference_px=difference,requested_trials=trials,valid_trials=successes,
            synthetic_fixed_profile_window_boundary_hits=boundary_hits,maximum_numeric_iterations=max_numeric_iterations,maximum_final_equation_residual_px=max_numeric_residual,
            source_operator_configuration=receipt.get('source_operator_configuration',receipt.get('fixed_source_operator_configuration')),source_corner_mean_unchanged=True,
            noise_model='independent original 8-bit RGB quantization Uniform[-0.5,+0.5] LSB before the exact frozen source operator',
            two_material_edges_remeasured_jointly=True,one_direction_corner_statistic_used=False,
            conditional_localization_only=operator=='fixed',material_selection_uncertainty_included=False,
            physical_accuracy_or_formal_noise_calibration_claimed=False)
        if successes<trials:
            record.update(status='pending',reason='nonlinear source resampling crosses finite source/domain boundary; censored samples not promoted');decisions.append(record);counts['nonlinear_domain_crossing']+=1;continue
        covariance=np.cov(samples,rowvar=False,ddof=1)
        try:
            W,proof=supported_whitening(covariance)
            if len(W)!=2:raise StatisticsUnavailable('native corner covariance is not two dimensional')
        except (ValueError,StatisticsUnavailable) as e:
            record.update(status='pending',reason=str(e));decisions.append(record);counts['covariance_not_two_dimensional']+=1;continue
        first=np.cov(samples[:trials//2],rowvar=False,ddof=1);second=np.cov(samples[trials//2:],rowvar=False,ddof=1)
        drift=float(np.linalg.norm(first-second)/np.linalg.norm(covariance));gid='entity:'+rid;name=f'corner_{index:04d}_W.f64';W.astype('<f8').tofile(out/name)
        source_hash=object_sha256({'native_pts':row['native_pts'],'source_uv':row['source_uv'],'source_row_id':rid})
        current_hash=object_sha256({'native_pts':row['native_pts'],'current_uv':row['current_uv'],'source_row_id':rid,'measurement_operator':row['measurement_operator_resolved']})
        record.update(status='numeric_two_direction_initialization_W_ready',row_ids=[rid],rank=2,
            W={'file':str((out/name).resolve()),'sha256':file_sha256(out/name),'dtype':'<f8','shape':[2,2]},
            statistic_kind='initialization_surrogate',numeric_whitening_verified=True,
            source_measurements_sha256={rid:source_hash},current_measurements_sha256={rid:current_hash},
            original_source_family_sha256=source_hash,source_group_id=gid,
            covariance_px2=covariance.tolist(),bootstrap_mean_delta_px=samples.mean(0).tolist(),
            split_half_covariance_relative_difference=drift,proof=proof,
            native_png={'file':row['native_png'],'sha256':row['native_png_sha256']},
            native_patch_generation={'file':str(entity/'manifest.json'),'sha256':file_sha256(entity/'manifest.json')},
            noise_calibration_status='nonlinear_two_edge_quantization_replay_completed; shared-family nuisance and broader source-noise calibration pending',
            original_W_claimed=False,PSD_floor_or_ridge_used=False)
        registry[gid]=record;decisions.append(record);counts['two_direction_W_ready']+=1
        if len(decisions)%20==0:write_json(out/'progress.json',{'source_records':len(decisions),'counts':dict(counts)})
    write_json(out/'statistics_registry.json',registry);write_json(out/'decisions.json',decisions)
    np.savez_compressed(out/'joint_edge_resampling.npz',source_row_ids=np.asarray(row_ids),delta_uv=np.asarray(delta_arrays),valid=np.asarray(valid_arrays))
    result={'schema':'rinkfit.native_entity_joint_resampling_statistics.v1','counts':dict(counts),'trials_per_entity':trials,'source_seed':seed,'source_operator':operator,
        'source_audit_rows':{'file':str(audit/'relocated_rows.jsonl.gz'),'sha256':file_sha256(audit/'relocated_rows.jsonl.gz')},
        'native_patch_generation':{'file':str(entity/'manifest.json'),'sha256':file_sha256(entity/'manifest.json')},
        'all_existing_entity_UV_or_binding_changed':False,'entity_station_freed':False,'one_dimensional_corner_W_used':False,
        'shared_source_family_covariance_complete':False,'formal_statistical_optimum_claimed':False}
    write_json(out/'manifest.json',result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True);ap.add_argument('--entity',required=True);ap.add_argument('--out',required=True);ap.add_argument('--trials',type=int,default=128);ap.add_argument('--operator',choices=['two_edge','reacquired','fixed'],default='two_edge');a=ap.parse_args()
    print(json.dumps(compile_corner_statistics(a.audit,a.entity,a.out,trials=a.trials,operator=a.operator)))
