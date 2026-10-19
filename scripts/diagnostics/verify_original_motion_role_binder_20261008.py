"""Real production binder/method/pickle interfaces on a small pure-NPZ bundle.

This creates only one actual block owner with original source arrays. It does
not load historical pickles, prepare the programme graph or assemble global J.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from copy import copy,deepcopy
import argparse,hashlib,json,os,pickle,sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeMountOpticsChart
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
from structured_programme_source_candidate.original_motion_far_board_roles import (
    OriginalMotionFarBoardRoleBinding,install_original_motion_far_board_roles,PAIR)
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cli=argparse.ArgumentParser()
    for name in ('state','camera-identity','auxiliary','packet','ledger','output'):cli.add_argument('--'+name,required=True,type=Path)
    a=cli.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names','aux__installation_xyz_slots',
        'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices',
        'aux__correlated_roll_prior_contract_json','aux__current_complete_parameter_layout_json')
    with np.load(a.state,allow_pickle=False)as z:
        small={k:z[k].copy()for k in keys}
        weights=z['profile__weights_2'].copy()
        depth_contract=json.loads(str(z['aux__unknown_depth_physical_domain_contract_json'].item()))
    provider,_,_=current_complete_source_geometry(a.state,a.auxiliary)
    identity=read(a.camera_identity)
    if sha(identity['camera_path'])!=identity['camera_sha256']:raise ValueError('Archived camera source changed')
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    state,camera_error=state_camera_adapter(small,cameras,provider);q=state.evaluation_q(small['q']);core=q[:state.structured_programme_coordinate_chart.core_count]
    fixture=state.bypts[PAIR[0]];role=OriginalMotionFarBoardRoleBinding(a.packet)
    # Use actual original 39,679 UV rows and original clock; no synthetic rows.
    with np.load(role.packet['original_motion_graph']['path'],allow_pickle=False)as z:
        original={k:z[k].copy()for k in z.files}
    bundle=FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle)
    bundle.stamps=fixture.bundle.stamps.copy();bundle.index=dict(fixture.bundle.index)
    counts=np.diff(original['offset']);pairs=original['pair_pts']
    bundle.ia=np.repeat([bundle.index[int(p)]for p in pairs[:,0]],counts)
    bundle.ib=np.repeat([bundle.index[int(p)]for p in pairs[:,1]],counts)
    bundle.a=original['source_uv'];bundle.b=original['target_uv'];bundle.motion_sigma=np.full(2*len(bundle.a),2.)
    bundle.finite_records=deepcopy(read(a.ledger)['source_plane_binding'])
    assert len(bundle.finite_records)==6 and all(r['surface_kind']=='ice_plane'for r in bundle.finite_records)
    bundle.finite_mask=np.zeros(len(bundle.a),bool);bundle.finite_planes=np.zeros((len(bundle.a),4))
    for record in bundle.finite_records:
        i=int(record['global_bundle_index']);bundle.finite_mask[i]=True;bundle.finite_planes[i]=[0.,0.,1.,0.]
    bundle.scene=deepcopy(provider.current.base)
    chart=PairedProgrammeMountOpticsChart(fixture.bundle.calibration.installation_id,tuple(map(int,fixture.optics_chart.indices)),
        34195,mount_indices=tuple(map(int,fixture.optics_chart.mount_indices)),additional_world_parameter_count=provider.parameter_count-66)
    from z190_physical_camera import SharedCalibration
    cal=SharedCalibration.from_manifest(fixture.bundle.calibration.manifest())
    dm=q[list(chart.mount_indices)]/provider.scene_unit_m;do=q[list(chart.indices)]
    base_cal=replace(cal,lever_xyz=tuple(np.asarray(cal.lever_xyz)-dm),
        principal_x_fraction=(cal.principal_x_fraction[0]-do[0]/1920.,),
        principal_y_fraction=(cal.principal_y_fraction[0]-do[1]/1080.,),k1=(cal.k1[0]-.001*do[2],))
    block=StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    block.name='secondary';block.start=fixture.start;block.xyz_slots=fixture.xyz_slots.copy();block.bundle=bundle
    block.optics_chart=chart;block.optics_base_calibration=base_cal;block.q0=np.zeros(4+4*len(bundle.stamps))
    block.geometry=provider;block.geometry_parameter_indices=state.geometry_parameter_indices.copy();block.geometry_parameter_count=provider.parameter_count
    block.course_board_indices=np.empty(0,int);block.course_board_record_positions=np.empty(0,int)
    block.course_original_board_records=[];block.course_original_board_planes=np.empty((0,4))
    block._course_current_geometry_key=block._course_camera_cache_key=None
    block.actor_output_capacity=dict(rows=2*len(bundle.a),motion_rows=2*len(bundle.a),nnz=100)
    reference={p:fixture.bundle.camera(core,p)for p in PAIR}
    from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock
    local=SecondaryOpticsFixedXYZRinkBlock.local(block,core)
    production_camera_error=max(float(abs(bundle.camera(local,p)-reference[p]).max())for p in PAIR)
    assert production_camera_error<1e-11,production_camera_error
    state=SimpleNamespace(original=SimpleNamespace(aliases={p:p for p in PAIR}),geometry=provider,bypts={p:block for p in PAIR},
        structured_programme_coordinate_chart=state.structured_programme_coordinate_chart,geometry_parameter_indices=state.geometry_parameter_indices)
    before={k:deepcopy(getattr(bundle,k))for k in('a','b','ia','ib','motion_sigma','finite_records','finite_planes','finite_mask')}
    contract=install_original_motion_far_board_roles(state,a.packet,q)
    assert int(bundle.finite_mask.sum())==21 and len(bundle.finite_records)==21 and len(block.course_board_indices)==15
    assert bundle.finite_records[:6]==before['finite_records'] and np.array_equal(bundle.finite_planes[before['finite_mask']],before['finite_planes'][before['finite_mask']])
    assert all(np.array_equal(getattr(bundle,k),before[k])for k in('a','b','ia','ib','motion_sigma'))
    assert block.actor_output_capacity['nnz']==280 and block.actor_output_capacity['rows']==2*len(bundle.a)
    bound=provider.finite_board_records(block.course_original_board_records,core[state.geometry_parameter_indices],original_planes=block.course_original_board_planes)
    assert bound['extent_world'].shape==(15,2,3)and bound['extent_world_jacobian'].shape==(15,2,3,98)
    assert np.array_equal(bound['planes'],bundle.finite_planes[role.indices])
    changed=core.copy();changed[state.geometry_parameter_indices[44]]+=1e-4
    old=bundle.finite_planes[role.indices].copy();block.local(changed)
    world_changed_max=float(abs(bundle.finite_planes[role.indices]-old).max());assert world_changed_max>0.
    block.local(core);signature=state.original_motion_far_board_roles.source_camera_signature
    camera_changed=core.copy();camera_changed[block.start+4*bundle.index[PAIR[0]]]+=1e-7;block.local(camera_changed)
    assert state.original_motion_far_board_roles.source_camera_signature!=signature
    block.local(core)
    # Serialize only this object constructed here, not an external pickle.
    from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    encoded=pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL);restored=pickle.loads(encoded)
    rb=restored.bypts[PAIR[0]];assert rb is restored.bypts[PAIR[1]]
    assert rb._original_motion_far_board_role_binding is restored.original_motion_far_board_roles
    assert restored.geometry._original_motion_far_board_role_binding is restored.original_motion_far_board_roles
    assert rb.synchronize_course.args[0] is rb and restored.geometry.finite_board_records.args[0] is restored.geometry
    rb.local(changed);rb.local(core)
    assert np.array_equal(rb.bundle.finite_planes,bundle.finite_planes)
    from z190_process_source_evaluation import _minimal_pickle
    actor_payload,dynamic_modules=_minimal_pickle(block);actor_block=pickle.loads(actor_payload)
    assert actor_block.synchronize_course.args[0] is actor_block
    assert actor_block.geometry.finite_board_records.args[0] is actor_block.geometry
    assert actor_block._original_motion_far_board_role_binding is actor_block.geometry._original_motion_far_board_role_binding
    actor_block.local(changed);actor_block.local(core)
    assert np.array_equal(actor_block.bundle.finite_planes,bundle.finite_planes)
    # The production residual-only path makes an additional shallow parent
    # clone before its delegate takes the same minimal-pickle route.
    residual_parent_clone=copy(block)
    residual_payload,_=_minimal_pickle(residual_parent_clone);residual_worker=pickle.loads(residual_payload)
    assert residual_worker.synchronize_course.args[0] is residual_worker
    residual_worker.local(changed);residual_worker.local(core)
    assert np.array_equal(residual_worker.bundle.finite_planes,bundle.finite_planes)
    # Read-only depth-domain count over this one real source producer. This
    # uses archived current cameras and frozen original profile weights.
    import cv2
    from z190_motion_native import native_motion
    lookup={int(p):i for i,p in enumerate(cameras['exposure_representative_pts'])}
    current=np.asarray([cameras['camera25'][lookup[int(p)]]for p in bundle.stamps])
    rotation=np.asarray([cv2.Rodrigues(c[3:6])[0]for c in current])
    ca,cb=current[bundle.ia],current[bundle.ib]
    kernel=dict(a=bundle.a,b=bundle.b,fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
        ra=rotation[bundle.ia],rb=rotation[bundle.ib],ca=ca[:,:3],cb=cb[:,:3],pp=ca[0,8:10],k1=ca[0,11],
        sigma=np.full((len(bundle.a),2),2.),profile_weight=weights,jacobian=False,
        da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None)
    depth=[];depth_arrays={}
    for label,planes in (('original_unknown15',before['finite_planes']),('admitted_finite15',bundle.finite_planes)):
        residual,checks,trace=native_motion(**kernel,planes=planes,minimum_source_depth_scene_units=1./provider.scene_unit_m,schur=True,return_trace=True)
        axial=np.divide(1.-trace[:,1],trace[:,1],out=np.full(len(trace),np.inf),where=trace[:,1]!=0)*provider.scene_unit_m
        unknown=trace[:,12]==0;floor=unknown&np.isclose(axial,1.,atol=1e-9,rtol=0.)
        depth_arrays.update({label+'_axial_depth_working_m':axial,label+'_unknown_mask':unknown,
            label+'_one_metre_endpoint_mask':floor,label+'_interior_mask':trace[:,9]!=0,
            label+'_native_residual_px':residual.reshape(-1,2)})
        depth.append(dict(label=label,checks=checks,unknown_rows=int(unknown.sum()),
            actual_one_metre_endpoint_rows=int(floor.sum()),fraction_unknown_at_one_metre=float(floor.sum()/unknown.sum()),
            finite_rows=int((~unknown).sum()),scope='one actual secondary producer, all 39,679 original FIT points',
            other_five_programme_blocks_not_reprofiled=True))
    report=dict(schema='rinkspace.original-motion-source-role-production-binder-method-roundtrip-check.v1',packet=contract,
        logical_world_count=provider.parameter_count,actual_source_UV_rows=len(bundle.a),original_secondary_finite_ice_rows_preserved=6,
        newly_typed_original_motion_rows=15,new_scalar_rows=0,actual_camera_adapter_error=camera_error,
        production_base_local_camera_replay_error=production_camera_error,dynamic_world_plane_changed_max=world_changed_max,
        current_extent_world_shape=list(bound['extent_world'].shape),current_extent_J_shape=list(bound['extent_world_jacobian'].shape),
        source_UV_PTS_sigma_all_rows_bitexact=True,source_camera_invalidates_course_cache=True,
        self_constructed_production_object_pickle_roundtrip_passed=True,pickle_bytes=len(encoded),
        actual_actor_minimal_pickle_roundtrip_passed=True,actor_hook_owner_is_actual_worker_block=True,actor_payload_bytes=len(actor_payload),
        residual_only_double_shallow_copy_actor_roundtrip_passed=True,
        unknown_depth_contract=depth_contract,actual_secondary_depth_domain_counts=depth,
        whole_graph_not_constructed=True,external_pickle_not_loaded=True)
    depth_path=a.output/'secondary_all_original_depth_bound_values_v1.npz'
    np.savez_compressed(depth_path,**depth_arrays,pair_pts=original['pair_pts'],pair_offsets=original['offset'])
    report['secondary_depth_values']=dict(path=str(depth_path),sha256=sha(depth_path),
        operator_values_only=True,source_roles_or_point_qualification_not_inferred_from_numbers=True)
    (a.output/'original_motion_role_production_binder_check_v1.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
