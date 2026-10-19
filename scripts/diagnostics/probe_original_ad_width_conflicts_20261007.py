"""Read-only current named66 width/station attribution on original sources.

These are explanatory finite differences, never a state initializer or fit.
"""
from pathlib import Path
import argparse, json, pickle, time
import numpy as np
from audit_all_original_absolute_sources_20261007 import (
    accepted, read, sha, DATA, PACKET, PACKET_SHA256, load_camera,
    load_original_absolute_evaluators, current_complete_source_geometry,
    install_source_ice_material_view)


def main():
    p=argparse.ArgumentParser();p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);args=p.parse_args();start=time.perf_counter()
    identity=accepted(args.state_dir);options=read(args.config)['source_options']
    provider,world,proof=current_complete_source_geometry(args.state_dir/'complete_state_UNACCEPTED.npz',options['--world-init'])
    install_source_ice_material_view(provider)
    if sha(PACKET)!=PACKET_SHA256:raise ValueError('Original source changed')
    source=pickle.loads(PACKET.read_bytes())['programme_absolute_rows']
    calls,authority=load_original_absolute_evaluators();archive=load_camera(identity['camera_path'])
    cameras={int(t):archive['camera25'][i] for i,t in enumerate(archive['exposure_representative_pts'])}
    audited=read(DATA/'diagnostics/external_expert_20261007/all_original_absolute_source_audit'/f'{identity["q_sha256"][:12]}_all_absolute_source_audit.json')
    valid_ads={r['original_packet_row_index'] for r in audited['full_source_rows'] if r['status']=='evaluated' and r.get('material_kind')=='ad'}
    selected=[]
    for index in sorted(valid_ads):
        item=source[index];row=item['source_row'];at=int(item['group_actual_pts'])
        provider.register_row(row,actual_pts=at,allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
        ids=provider._binding(row)[1]['ids']
        if not any(n in (1,3,14,16) for n,_ in ids):continue
        selected.append(dict(index=index,item=item,ids=ids))

    def residual(record,q):
        item=record['item'];row=item['source_row'];bound=provider.rebind(row,q)
        return np.asarray(calls[item['block']](cameras[int(item['group_actual_pts'])],[bound],False)).reshape(-1)

    def summary(record,q):
        row=record['item']['source_row'];r=residual(record,q).reshape(-1,2)*np.asarray(row['sigma'])[:,None]
        d=np.linalg.norm(r,axis=1)
        return dict(original_packet_row_index=record['index'],id=row['id'],pts=int(row['pts']),actual_pts=int(record['item']['group_actual_pts']),role=record['item']['collection_role'],physical_ids=record['ids'],mean_outside_support_native_px=float(d.mean()),p95_outside_support_native_px=float(np.quantile(d,.95)),mean_model_minus_source_XY_native_px=r.mean(axis=0).tolist())

    before=[summary(r,world) for r in selected];trials=[]
    definitions=[
        ('Ad1_four_actual_external_corners_229624064',(1,14),['Ad1_14_station_center_m','Ad1_14_width_m'],lambda r:r['index'] in (5253,5254,5255,5256)),
        ('Ad1_all_original_FIT_primary_sources',(1,14),['Ad1_14_station_center_m','Ad1_14_width_m'],lambda r:r['item']['collection_role']=='FIT' and any(n==1 for n,_ in r['ids'])),
        ('Ad3_original_named_cross_cut_extremes',(3,16),['Ad3_independent_station_center_m','Ad3_independent_arc_width_m'],lambda r:r['item']['collection_role']=='FIT' and int(r['item']['group_actual_pts']) in (229217024,229528320,229531904) and any(n==3 for n,_ in r['ids'])),
        ('Ad3_all_original_FIT_primary_sources',(3,16),['Ad3_independent_station_center_m','Ad3_independent_arc_width_m'],lambda r:r['item']['collection_role']=='FIT' and any(n==3 for n,_ in r['ids'])),
    ]
    names=tuple(provider.parameter_names);h=1e-6
    for label,pair,parameters,selector in definitions:
        cols=[names.index(n) for n in parameters];training=[r for r in selected if selector(r)]
        if not training:raise ValueError('No real original training source for '+label)
        value=np.concatenate([residual(r,world) for r in training]);jac=[]
        for col in cols:
            plus=world.copy();minus=world.copy();plus[col]+=h;minus[col]-=h
            jac.append(np.concatenate([(residual(r,plus)-residual(r,minus))/(2*h) for r in training]))
        J=np.column_stack(jac);delta=np.linalg.lstsq(J,-value,rcond=None)[0]
        candidate=world.copy();candidate[cols]+=delta
        record=dict(label=label,world_parameters=parameters,world_parameter_indices=cols,complete_q_parameter_indices=[proof['actual_world_parameter_indices'][k] for k in cols],parameter_values_before=world[cols].tolist(),explanatory_delta_working_metres=delta.tolist(),training_original_indices=[r['index'] for r in training],training_native_clock=[int(r['item']['group_actual_pts']) for r in training],singular_values=np.linalg.svd(J,compute_uv=False).tolist(),FD_width=h,actual_camera_all_exposures_unchanged=True,no_source_or_state_initializer_written=True)
        try:
            provider._q(candidate)
            after=[summary(r,candidate) for r in selected]
            group=[]
            for n in pair:
                for role in ('FIT','HELD'):
                    members=[(a,b) for r,a,b in zip(selected,before,after) if any(i==n for i,_ in r['ids']) and r['item']['collection_role']==role]
                    if not members:continue
                    group.append(dict(ad=n,role=role,rows=len(members),source_row_mean_before=float(np.mean([a['mean_outside_support_native_px'] for a,b in members])),source_row_mean_after=float(np.mean([b['mean_outside_support_native_px'] for a,b in members])),source_row_worst_p95_before=max(a['p95_outside_support_native_px'] for a,b in members),source_row_worst_p95_after=max(b['p95_outside_support_native_px'] for a,b in members)))
            records=[dict(b,source_mean_before_px=a['mean_outside_support_native_px'],delta_mean_px=b['mean_outside_support_native_px']-a['mean_outside_support_native_px']) for a,b in zip(before,after) if any(n in pair for n,_ in a['physical_ids'])]
            record.update(nonlinear_existing_geometry_guard_passed=True,all_original_pair_sources=group,greatest_original_source_regressions=sorted(records,key=lambda r:r['delta_mean_px'],reverse=True)[:8],training_sources_after=[b for r,b in zip(selected,after) if selector(r)])
        except ValueError as error:
            record.update(nonlinear_existing_geometry_guard_passed=False,existing_guard_error=str(error),no_unchecked_trial_claim=True)
        trials.append(record)
    report=dict(schema='rinkspace.original-advertisement-current-width-station-read-only-attribution.v1',state=identity,world_proof=proof,source_evaluators=authority,original_source_packet_sha256=PACKET_SHA256,baseline_original_pair_sources=before,diagnostic_trials=trials,elapsed_seconds=time.perf_counter()-start,FD_rebinds_actual_owned_current_material=True,whole_joint_cost_not_evaluated=True,diagnostic_not_a_camera_or_world_fit=True,source_or_state_not_modified=True,accuracy_accepted=False)
    out=DATA/'diagnostics/external_expert_20261007/all_original_absolute_source_audit';out.mkdir(parents=True,exist_ok=True)
    path=out/f'{identity["q_sha256"][:12]}_original_ad_width_station_attribution.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(path),seconds=report['elapsed_seconds'],trials=[{k:v for k,v in r.items() if k not in ('greatest_original_source_regressions','training_sources_after')}for r in trials]),ensure_ascii=False))


if __name__=='__main__':main()
