"""Read-only same-camera wide5808/original225 value check of local A release."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import argparse
import hashlib
import json
import os
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(Path(__file__).parent)]
sys.dont_write_bytecode = True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from diagnose_current_goal_transverse_release_20261008 import TransverseReleaseView


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def numeric_geometry(row):
    return {k:np.asarray(row[k],float) for k in ('world','world_xyz','world_xyz_scene_units') if k in row}
def difference(a,b):
    assert a.keys()==b.keys()
    return max((float(abs(a[k]-b[k]).max()) for k in a), default=0.)
def stats(raw):
    d=np.linalg.norm(np.asarray(raw,float).reshape(-1,2),axis=1)
    return dict(mean_px=float(d.mean()),median_px=float(np.median(d)),p95_px=float(np.quantile(d,.95)),max_px=float(d.max()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();ab=read(args.experiment);state_path=Path(ab['state_path'])
    identity=ab['state_identity'];arrays=dict(np.load(state_path,allow_pickle=False))
    provider,world,proof=current_complete_source_geometry(state_path,ab['world_proof']['auxiliary_path'])
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    state,camera_error=state_camera_adapter(arrays,cameras,provider);q=state.evaluation_q(arrays['q']).copy()
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    from structured_programme_source_candidate.current_wide_source import CurrentWideSource
    from wide_anchor_candidate.wide_high_source_factor import WORLD_INDICES as OLD_WORLD_INDICES
    from wide_anchor_candidate.wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor
    from z190_streaming_source_normal import _block_cost
    optics=json.loads(str(arrays['aux__all_installation_optics_contract_json'].item()))
    ids=json.loads(str(arrays['aux__correlated_roll_prior_contract_json'].item()))['installation_order']
    secondary=ids[1]
    original_state=SimpleNamespace(q=arrays['q'][:34225],ids=ids,slots={secondary:np.array([8,9,10])},
        geometry=SimpleNamespace(parameter_count=23,scene=lambda unused:provider.current.scene(world)),
        geometry_parameter_indices=OLD_WORLD_INDICES,
        installation_optics={secondary:SimpleNamespace(mount_indices=np.array([34198,34199,34200]))},optics_contract=optics)
    oldwide=WideHighFreeOpticsSourceFactor(original_state)
    trace=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json')
    wide=CurrentWideSource(oldwide,state,read(trace))
    view=TransverseReleaseView(CurrentIceMaterialView(provider.current),'A_left_crease_only',provider.scene_unit_m)
    wide._wide_ice_material_view=view;provider._source_ice_material_view=view
    core_count=state.structured_programme_coordinate_chart.core_count
    private=q[np.asarray(state.structured_programme_coordinate_chart.high_indices,int)]
    savedwide=read(state_path.parent/'source_review_UNACCEPTED.json')['wide']
    baseline=wide.evaluate(q[:core_count],private,False)[0]
    baseline_error=float(abs(baseline-np.asarray(savedwide['source_normalized_residual'])).max())
    assert baseline_error<1e-10,baseline_error
    before_wide=wide.review(q[:core_count],private)
    assert len(baseline)==5808
    replication_counts=np.rint(4*wide.cost_weights).astype(int)
    assert np.array_equal(replication_counts,4*wide.cost_weights)
    replication_indices=np.repeat(np.arange(len(baseline)),replication_counts)
    print(json.dumps(dict(stage='wide_baseline_verified',rows=len(baseline),saved_r_max=baseline_error)),flush=True)
    from verify_original225_derived_center_20261008 import numeric_data_loads
    from structured_programme_source_candidate import original_review_source as original
    from structured_programme_source_candidate import ad14_source_role_certificate
    safe=SimpleNamespace(loads=numeric_data_loads);original.pickle=safe;ad14_source_role_certificate.pickle=safe
    rows=original.load_original_review_rows();calls,authorities=original.load_original_absolute_evaluators()
    signatures=[original.observation_signature(x['source_row']) for x in rows]
    for item in rows:
        provider.register_row(item['source_row'],actual_pts=int(item['group_actual_pts']),
            allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
    before_rows=[provider.rebind(x['source_row'],world) for x in rows]
    original_cameras={int(x['group_actual_pts']):state.bypts[int(x['group_actual_pts'])].bundle.camera(q,int(x['group_actual_pts'])) for x in rows}
    saved225=read(state_path.parent/'original225_review_UNACCEPTED.json')
    savedkeys={(r['block'],r['group_order'],r['row_order'],r['id']):r for r in saved225['rows']}
    assert len(savedkeys)==225
    report=dict(schema='rinkspace.local-goal-transverse-release-same-state-wide-original-source-values.v1',
        frozen_A_B_report=dict(path=str(args.experiment),sha256=sha(args.experiment)),state_identity=identity,
        world_proof=proof,script_sha256=sha(__file__),source_rows_UV_sigma_roles_and_camera_unchanged=True,
        wide5808_nominal_saved_residual_max=baseline_error,actual_camera25_archive_reproduce_max=camera_error,
        original225_numeric_only_restricted_source_decoder=True,original225_source_packet_sha256=original.PACKET_SHA256,
        no_J_full_motion_or_source_target_loaded=True,whole_model_adoption_or_accuracy_acceptance=False,experiments=[])
    choices=[e for e in ab['experiments'] if e['mode']=='A_left_crease_only' and e['experiment'] in
        ('offset_only','offset_plus_existing_shared_radius9')]
    for entry in choices:
        trial=q.copy();trial[state.geometry_parameter_indices[9]]=entry['radius9_working_m']
        trialworld=trial[state.geometry_parameter_indices];view.offset_m=entry['offset_working_m']
        wide._cache.clear()
        after_wide=wide.review(trial[:core_count],private);wide_r=np.asarray(after_wide['source_normalized_residual'])
        oldgroups={r['id']:r for r in before_wide['raw_source_native_pixel_error_groups']}
        changes=[dict(id=r['id'],before=oldgroups[r['id']],after=r,
            delta_mean_px=r['mean_native_px']-oldgroups[r['id']]['mean_native_px'])
            for r in after_wide['raw_source_native_pixel_error_groups']]
        originals=[];changed_count=0
        for item,before,signature in zip(rows,before_rows,signatures):
            row=item['source_row'];after=provider.rebind(row,trialworld)
            assert original.observation_signature(before)==original.observation_signature(after)==signature
            dw=difference(numeric_geometry(before),numeric_geometry(after))
            key=(item['block'],int(item['group_order']),int(item['row_order']),row['id']);record=deepcopy(savedkeys[key])
            record['experimental_material_world_max_delta_scene_units']=dw
            if dw:
                changed_count+=1;camera=original_cameras[int(item['group_actual_pts'])]
                rb=np.asarray(calls[item['block']](camera,[before],False));ra=np.asarray(calls[item['block']](camera,[after],False))
                assert np.isfinite(ra).all() and rb.shape==ra.shape==(2*len(row['uv']),)
                saved_rms=record['normalized_operator_rms'];rms=float(np.sqrt(np.mean(rb*rb)))
                assert abs(rms-saved_rms)<1e-10,(row['id'],rms,saved_rms)
                scale=np.repeat(np.asarray(row['sigma'],float),2)
                record.update(before_same_actual_operator=stats(rb*scale),after_same_actual_operator=stats(ra*scale),
                    normalized_r_max_delta=float(abs(ra-rb).max()),source_operator_evaluated=True,
                    before_saved_normalized_rms_max_error=abs(rms-saved_rms))
            else:
                record.update(source_operator_proven_bitexact_same_camera_and_world=True,normalized_r_max_delta=0.)
            originals.append(record)
        result=dict(A_local_experiment=entry['experiment'],offset_working_m=entry['offset_working_m'],
            radius9_delta_working_m=entry['radius9_delta_working_m'],wide5808_baseline_unweighted_unique_cost=float(_block_cost(baseline,wide.quadratic)),
            wide5808_after_unweighted_unique_cost=float(_block_cost(wide_r,wide.quadratic)),
            actual_weighted_fourfold_wide_FIT_cost_before=float(_block_cost(baseline[replication_indices],wide.quadratic[replication_indices])),
            actual_weighted_fourfold_wide_FIT_cost_after=float(_block_cost(wide_r[replication_indices],wide.quadratic[replication_indices])),
            wide_fourfold_and_original_component_weights_preserved=True,wide_groups=changes,
            original225_changed_material_rows=changed_count,original225_unchanged_material_rows=225-changed_count,
            original225_all_rows=originals,original225_actual_operator_authorities=authorities)
        report['experiments'].append(result)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(stage='done',experiment=result['A_local_experiment'],
            wide_cost_before=result['actual_weighted_fourfold_wide_FIT_cost_before'],wide_cost_after=result['actual_weighted_fourfold_wide_FIT_cost_after'],
            changed_original225=changed_count,goal_wide=[r for r in changes if 'goal' in r['id']],
            changed_original_ids=[r['id'] for r in originals if r['experimental_material_world_max_delta_scene_units']]),ensure_ascii=False),flush=True)
    view.offset_m=0.;wide._cache.clear()
    np.testing.assert_array_equal(wide.evaluate(q[:core_count],private,False)[0],baseline)
    for item,old in zip(rows,before_rows):
        assert difference(numeric_geometry(old),numeric_geometry(provider.rebind(item['source_row'],world)))==0.
    report['baseline_geometry_and_wide_restored_bitexact']=True
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),complete=True)),flush=True)


if __name__=='__main__':main()
