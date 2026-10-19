"""Reproduce one real failed source gate from immutable model/source inputs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np


def main():
    from calibration.solver_runtime import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    from calibration.program_tracking.strict_model_scope import authenticate_source_reference
    from calibration.program_tracking.compact_source_json import unpack_source_json
    from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout,PairedProgrammeMountOpticsChart
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from z190_physical_camera import SharedCalibration
    from z190_continuous_source_bundle import SCENE_UNIT_M
    from calibration.solver_runtime.strict_source_dataset_adapter import StrictDatasetFactor
    from calibration.solver_runtime.strict_dataset_camera_runtime import compare_strict_dataset_camera_mapping
    from calibration.solver_runtime.source_direction_probe import save_source_direction_probe
    r=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    a=r/'okada_q32_reviewed_source_pixels_current_points_ready_20261012/accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891'
    identity=json.loads((a/'state_identity.json').read_text(encoding='utf8'))
    camera_path=Path(identity['camera_path']);authenticate_source_reference(dict(path=str(camera_path),sha256=identity['camera_sha256']))
    state_path=a/'complete_state_UNACCEPTED.npz';authenticate_source_reference(dict(path=str(state_path),sha256=identity['state_npz_sha256']))
    with np.load(camera_path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata_json'].item()));actual=z['exposure_representative_pts'];cameras=z['camera25']
        camera_lookup={int(k):v for k,v in zip(actual,cameras)}
    cal=SharedCalibration.from_manifest(metadata['calibrations']['okada_sp_installation_2_conditional_z190'])
    with np.load(state_path,allow_pickle=False) as z:
        canonical=z['q'].copy();pts=z['aux__canonical_pts'].copy();world_can=z['aux__current_shared_world_geometry_parameter_indices'].copy()
    layout=CompleteProgrammeParameterLayout(34306,34);q=layout.evaluation_q(canonical);wi=layout.canonical_to_evaluation[world_can]
    manifest=json.loads((r/'okada_strict_v6_8a28_fast_v4_20261013/input_manifest.json').read_text(encoding='utf8'))
    private_ref=manifest['absolute_geometry_fd_parallelism']['material_owner'];authenticate_source_reference(private_ref)
    owner=pickle.loads(Path(private_ref['path']).read_bytes())
    base=replace(cal,principal_x_fraction=(cal.principal_x_fraction[0]-q[34210]/1920.,),
        principal_y_fraction=(cal.principal_y_fraction[0]-q[34211]/1080.,),k1=(cal.k1[0]-.001*q[34212],),
        lever_xyz=tuple(np.asarray(cal.lever_xyz)-q[34201:34204]/SCENE_UNIT_M))
    chart=PairedProgrammeMountOpticsChart(cal.installation_id,(34210,34211,34212),34204,
        mount_indices=(34201,34202,34203),additional_world_parameter_count=34)
    bundle=FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle);bundle.calibration=base
    stamps=pts[(14330-14)//4:(21734-14)//4];bundle.index={int(k):j for j,k in enumerate(stamps)}
    block=StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    block.name='low';block.bundle=bundle;block.start=14330;block.xyz_slots=np.array([11,12,13])
    block.optics_chart=chart;block.optics_base_calibration=base;block.q0=np.zeros(7408)
    block.geometry_parameter_indices=wi;block.geometry=owner
    factor=StrictDatasetFactor.__new__(StrictDatasetFactor)
    factor.parameter_count=len(q);factor.core_count=34302;factor.world_indices=wi
    factor.state=SimpleNamespace(bypts={int(k):block for k in stamps});factor.geometry=owner.current
    factor.ice=CurrentIceMaterialView(owner.current)
    factor._evaluation_q=factor._world_cache_q=factor._camera_cache_q=None
    factor._world_cache={};factor._derivative_cache={};factor._camera_cache={}
    chain_receipt=compare_strict_dataset_camera_mapping(factor,q,[229672192])
    nominal=factor._camera(q,229672192,True)[0]
    if nominal.tobytes()!=camera_lookup[229672192].tobytes():raise ValueError('Reconstructed actual block differs from original same-q camera export')
    target=dict(kind='current_scene_ice_curve',curve_id='lane4:edge2',feature_id='lane4',world_curve_role='edge2')
    dataset_path=Path('events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/specific_inner_review_v6/fit_dataset.json')
    dataset=json.loads(dataset_path.read_text(encoding='utf8'));uv_parts=[];sigma_parts=[];locators=[]
    for cut in dataset['cuts']:
        with np.load(cut['source_pack']['path'],allow_pickle=False) as pack,np.load(cut['support_index']['path'],allow_pickle=False) as support:
            headers=unpack_source_json(pack['rows_json']);offsets=pack['row_offsets'];stations=support['fit_point_indices']
            row_ids=np.searchsorted(offsets,stations,side='right')-1
            chosen=[int(i) for i in np.unique(row_ids) if headers[i]['actual_pts']==229672192 and headers[i]['model_target']==target]
            if not chosen:continue
            authenticate_source_reference(cut['source_pack']);authenticate_source_reference(cut['support_index'])
            uv=pack['uv'];sigma=pack['sigma_px']
            for i in chosen:
                selected=stations[row_ids==i];uv_parts.append(uv[selected]);sigma_parts.append(sigma[selected])
                locators.append(dict(source_pack=cut['source_pack'],source_row_index=i,local_point_indices=(selected-offsets[i]).tolist()))
    uv=np.concatenate(uv_parts);sigma=np.concatenate(sigma_parts)
    take=np.unique(np.linspace(0,len(uv)-1,min(16,len(uv))).astype(int))
    row=dict(actual_pts=229672192,model_target=target,uv=uv[take],sigma=sigma[take])
    factor._preflight_probe_records={};factor._preflight_probe_label='nominal';factor._prepare_evaluation(q)
    residual,matrix=factor._evaluate_row(q,row,True)
    columns=np.unique(matrix.indices);direction=np.zeros(len(q));direction[columns]=np.random.default_rng(70213).normal(size=len(columns))
    predicted=matrix@direction;native=predicted*np.repeat(row['sigma'],2)
    h=min(1e-6,.001/max(1.,float(np.abs(native).max())))
    factor._preflight_probe_label='plus';plus=q+h*direction;factor._prepare_evaluation(plus);rp=factor._evaluate_row(plus,row,False)[0]
    factor._preflight_probe_label='minus';minus=q-h*direction;factor._prepare_evaluation(minus);rm=factor._evaluate_row(minus,row,False)[0]
    measured=(rp-rm)/(2*h);maximum=float(np.abs(measured-predicted).max());relative=maximum/max(1.,float(np.abs(predicted).max()))
    print(json.dumps(dict(stage='actual_parent_world_source_direction',relative=relative,maximum=maximum,
        full_group_points=len(uv),actual_pts=229672192,model_target=target)),flush=True)
    probe=save_source_direction_probe(q,direction,h,matrix,factor._preflight_probe_records,
        dict(kind=target['kind'],installation=cal.installation_id,actual_pts=229672192,model_target=target,
            selected_original_group_point_indices=take.tolist(),source_locators=locators,maximum=maximum,relative=relative,
            original_camera25_bytes_identical=True,parent_serial_world_FD=True))
    receipt=dict(probe=probe,parent_serial_world_FD_directional_error=relative,maximum=maximum,full_group_points=len(uv),
        camera_chain=chain_receipt,passed_original_threshold=relative<=1e-4)
    report=r/'strict_direction_probes/actual_low_source_direction_parent_world.json'
    report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
