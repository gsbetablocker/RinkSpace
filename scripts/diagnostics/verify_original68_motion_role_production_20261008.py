"""Two actual numeric producers: admitted68 roles, live J and worker owners.

Reconstructs only existing authenticated low/secondary numeric motion owners.
No historical pickle, full programme source graph or full global J is read.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from copy import deepcopy,copy
import argparse,hashlib,json,os,pickle,sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[n]='1'
os.environ['RINKSPACE_MOTION_BACKEND']='native'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)];sys.dont_write_bytecode=True
import cv2,numpy as np
from scipy.sparse import csr_matrix
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeMountOpticsChart
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding,install_original_motion_far_board_roles
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
from z190_physical_camera import SharedCalibration
from z190_profiled_motion_schur import SchurMotionKernel
from z190_motion_native import native_motion
from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock
from secondary_optics_candidate.z190_motion_optics_extension import native_motion_with_optics
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
PREP=DATA/'calibration/runs/solver_acceleration_20261003/prepared_full8503_1791035775270492900/complete_original_sources'
ADMISSION=DATA/'history/work_sessions/20261002_01a0fa4f-6592-7ab2-bc33-42ff983aff59/z190_low_fixed_mount_foreground_rf_v1/original_motion_foreground_admission.npz'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def stats(x):
    x=np.asarray(x);return dict(count=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))

def producer_state(path,auxiliary,packet):
    saved=arrays(path);identity=read(path.parent/'state_identity.json');assert sha(identity['camera_path'])==identity['camera_sha256']
    cameras=arrays(identity['camera_path']);provider,_,proof=current_complete_source_geometry(path,auxiliary)
    state,error=state_camera_adapter(saved,cameras,provider);q=state.evaluation_q(saved['q']);core=q[:state.structured_programme_coordinate_chart.core_count]
    role=OriginalMotionFarBoardRoleBinding(packet);children=[read(r['path'])for r in role.packet['source_packets']]
    original_secondary=arrays(children[0]['original_motion_graph']['path']);low=arrays(ADMISSION);active=low['active_original_global_index']
    lowrecords=deepcopy(read(PREP/'low_source/low_source_factory/finite_material_source_binding.json'))
    persistent=read(PREP/'low_source/input_manifest.json')['persistent_ice_source_opt_in']['added_source_ledger']
    assert len(persistent)==194 and len(lowrecords)==274
    secondaryrecords=deepcopy(read(PREP/'secondary_source/core_source_admission_ledger.json')['source_plane_binding'])
    blocks={};old={};camera_error=0.
    for name,bi in (('secondary',2),('low',3)):
        fixture=next(b for b in state.blocks if b.name==name);bundle=FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle)
        bundle.stamps=fixture.bundle.stamps.copy();bundle.index=dict(fixture.bundle.index)
        if name=='secondary':
            counts=np.diff(original_secondary['offset']);pairs=original_secondary['pair_pts']
            bundle.a=original_secondary['source_uv'];bundle.b=original_secondary['target_uv']
            bundle.ia=np.repeat([bundle.index[int(p)]for p in pairs[:,0]],counts);bundle.ib=np.repeat([bundle.index[int(p)]for p in pairs[:,1]],counts)
            bundle.motion_sigma=np.full(2*len(bundle.a),2.);records=secondaryrecords
        else:
            bundle.a=np.concatenate((low['source_uv_a'][active],np.array([r['source_uv']for r in persistent])))
            bundle.b=np.concatenate((low['source_uv_b'][active],np.array([r['target_uv']for r in persistent])))
            bundle.ia=np.r_[low['source_endpoint_index_a'][active],[bundle.index[int(r['source_actual_pts'])]for r in persistent]]
            bundle.ib=np.r_[low['source_endpoint_index_b'][active],[bundle.index[int(r['target_actual_pts'])]for r in persistent]]
            bundle.motion_sigma=np.r_[low['original_sigma_xy'][active].ravel(),np.array([r['sigma_engineering_px']for r in persistent]).ravel()]
            records=lowrecords+persistent
            ice=json.loads(str(saved['aux__qualified_original_motion_ice_contract_json'].item()))
            records+= [dict(global_bundle_index=int(i),surface_kind='ice_plane',plane=[0.,0.,1.,0.])for i in ice['global_bundle_indices']]
        bundle.profile_weights=saved['profile__weights_'+str(bi)].copy();assert bundle.profile_weights.shape==(len(bundle.a),2)
        bundle.motion_rows=2*len(bundle.a)
        bundle.finite_records=deepcopy(records);bundle.finite_mask=np.zeros(len(bundle.a),bool);bundle.finite_planes=np.zeros((len(bundle.a),4))
        for r in records:
            i=int(r['global_bundle_index']);bundle.finite_mask[i]=True
            if r['surface_kind']=='ice_plane':bundle.finite_planes[i]=[0.,0.,1.,0.]
            else:
                u,v=np.array(r['extent_start_xy']),np.array(r['extent_end_xy']);edge=v-u;normal=np.array([edge[1],-edge[0]])/np.linalg.norm(edge)
                bundle.finite_planes[i]=[normal[0],normal[1],0.,-normal@u]
        assert int(bundle.finite_mask.sum())==(6 if name=='secondary'else 471)
        bundle.scene=deepcopy(provider.current.base);bundle.motion_kernel=SchurMotionKernel(minimum_source_depth_scene_units=1./provider.scene_unit_m,schur=True)
        chart=PairedProgrammeMountOpticsChart(fixture.bundle.calibration.installation_id,tuple(map(int,fixture.optics_chart.indices)),
            34195,mount_indices=tuple(map(int,fixture.optics_chart.mount_indices)),additional_world_parameter_count=provider.parameter_count-66)
        cal=SharedCalibration.from_manifest(fixture.bundle.calibration.manifest());dm=q[list(chart.mount_indices)]/provider.scene_unit_m;do=q[list(chart.indices)]
        basecal=replace(cal,lever_xyz=tuple(np.array(cal.lever_xyz)-dm),principal_x_fraction=(cal.principal_x_fraction[0]-do[0]/1920.,),
            principal_y_fraction=(cal.principal_y_fraction[0]-do[1]/1080.,),k1=(cal.k1[0]-.001*do[2],))
        b=StructuredProgrammeBlock.__new__(StructuredProgrammeBlock);b.name=name;b.start=fixture.start;b.xyz_slots=fixture.xyz_slots.copy();b.bundle=bundle
        b.optics_chart=chart;b.optics_base_calibration=basecal;b.q0=np.zeros(4+4*len(bundle.stamps));b.geometry=provider
        b.geometry_parameter_indices=state.geometry_parameter_indices.copy();b.geometry_parameter_count=provider.parameter_count
        positions=sorted([i for i,r in enumerate(records)if r['surface_kind']=='finite_board_face'],key=lambda i:records[i]['global_bundle_index'])
        b.course_board_record_positions=np.asarray(positions,int);b.course_original_board_records=[deepcopy(records[i])for i in positions]
        b.course_board_indices=np.array([r['global_bundle_index']for r in b.course_original_board_records],int)
        b.course_original_board_planes=bundle.finite_planes[b.course_board_indices].copy();b._course_current_geometry_key=b._course_camera_cache_key=None
        b.actor_output_capacity=dict(rows=2*len(bundle.a),motion_rows=2*len(bundle.a),nnz=100)
        local=SecondaryOpticsFixedXYZRinkBlock.local(b,core)
        for row in role.records:
            if row['source_role_block']!=name:continue
            for stamp in (row['source_pts'],row['target_pts']):camera_error=max(camera_error,float(abs(bundle.camera(local,stamp)-fixture.bundle.camera(core,stamp)).max()))
        b.synchronize_course(core)
        old[name]={k:deepcopy(getattr(bundle,k))for k in ('a','b','ia','ib','motion_sigma','profile_weights','finite_mask','finite_planes','finite_records')}
        blocks[name]=b
    assert camera_error<1e-11,camera_error
    state.bypts={int(p):b for b in blocks.values()for p in b.bundle.stamps};state.blocks=list(blocks.values())
    contract=install_original_motion_far_board_roles(state,packet,q)
    for name,b in blocks.items():
        assert all(np.array_equal(getattr(b.bundle,k),old[name][k])for k in ('a','b','ia','ib','motion_sigma','profile_weights'))
        assert np.array_equal(b.bundle.finite_planes[old[name]['finite_mask']],old[name]['finite_planes'][old[name]['finite_mask']])
        expected=int(old[name]['finite_mask'].sum())+sum(r['source_role_block']==name for r in role.records)
        assert int(b.bundle.finite_mask.sum())==expected
    return state,q,saved,old,error,camera_error,contract,proof

def queries(state,q,role,group,cached_local=None):
    name,pa,pb=group;b=next(b for b in state.blocks if b.name==name)
    local=b.local(q[:state.structured_programme_coordinate_chart.core_count])if cached_local is None else cached_local
    rr=[r for r in role.records if (r['source_role_block'],r['source_pts'],r['target_pts'])==group];idx=np.array([r['original_global_bundle_index']for r in rr],int)
    ca,cb=[b.bundle.camera(local,p)for p in (pa,pb)];sigma=b.bundle.motion_sigma.reshape(-1,2)[idx]
    k=dict(a=b.bundle.a[idx],b=b.bundle.b[idx],fa=np.exp(ca[6]),fb=np.exp(cb[6]),ra=cv2.Rodrigues(ca[3:6])[0],rb=cv2.Rodrigues(cb[3:6])[0],
        ca=ca[:3],cb=cb[:3],pp=ca[8:10],k1=ca[11],planes=b.bundle.finite_planes[idx],sigma=sigma,profile_weight=b.bundle.profile_weights[idx],
        da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,jacobian=False)
    return b,idx,k,sigma,ca,cb,local

def actual_values(state,q,role,groups):
    result=[];locals_by_block={b.name:b.local(q[:state.structured_programme_coordinate_chart.core_count])for b in state.blocks if any(g[0]==b.name for g in groups)}
    for group in groups:
        b,idx,k,sigma,ca,cb,local=queries(state,q,role,group,locals_by_block[group[0]])
        r=b.bundle.motion_kernel(**k);result.append(r/sigma.ravel())
    return np.concatenate(result)

def full_motion_values(block,q,core_count,planes=None):
    local=block.local(q[:core_count]);bundle=block.bundle
    original_planes=bundle.finite_planes;original_records=bundle.finite_records
    if planes is not None:
        bundle.finite_planes=planes
        bundle.finite_records=[r for r in original_records if np.any(planes[int(r['global_bundle_index'])])]
    try:
        # Actual profiling entry includes the production finite extent and
        # positive-depth guards, and its original65536 native ray batches.
        normalized=bundle.profile_motion_residual(local)
        return (normalized*bundle.motion_sigma).reshape(-1,2)
    finally:bundle.finite_planes=original_planes;bundle.finite_records=original_records

def actual_J(state,q,role,groups):
    rows=[];cols=[];values=[];offset=0;reports=[]
    locals_by_block={b.name:b.local(q[:state.structured_programme_coordinate_chart.core_count])for b in state.blocks}
    for group in groups:
        b,idx,k,sigma,ca,cb,local=queries(state,q,role,group,locals_by_block[group[0]]);div=sigma.ravel();width=len(div)
        geometric_rows,jg=b.course_motion_columns(local,b.bundle.current_calibration(local));lookup={int(r):i for i,r in enumerate(geometric_rows)}
        block_j=jg[[lookup[int(r)]for r in (2*idx[:,None]+[0,1]).ravel()]]
        columns=list(map(int,state.geometry_parameter_indices[44:50]));parts=[block_j[:,i]for i in range(6)]
        ra,da=cv2.Rodrigues(ca[3:6]);rb,db=cv2.Rodrigues(cb[3:6]);da=da.reshape(3,3,3);db=db.reshape(3,3,3)
        lever=np.asarray(b.bundle.calibration.lever_xyz);dva=np.zeros((3,4));dvb=np.zeros((3,4));dva[:,:3]=np.einsum('aij,i->ja',da,lever);dvb[:,:3]=np.einsum('aij,i->ja',db,lever)
        derivative=dict(k,jacobian=True,da=da,db=db,dca_view=dva,dcb_view=dvb,dca_shared=np.c_[np.eye(3)/state.geometry.scene_unit_m,np.zeros(3)],dcb_shared=np.c_[np.eye(3)/state.geometry.scene_unit_m,np.zeros(3)])
        result,_,_=native_motion_with_optics(**derivative,minimum_source_depth_scene_units=1./state.geometry.scene_unit_m,schur=True)
        raw,ja,jb,jpivot,joptics=result
        for i,col in enumerate(b.xyz_slots):columns.append(int(col));parts.append(jpivot[:,i]/div)
        for stamp,matrix in zip(group[1:],(ja,jb)):
            begin=b.start+4*b.bundle.index[int(stamp)]
            for i in range(4):columns.append(begin+i);parts.append(matrix[:,i]/div)
        derivative.update(dca_shared=np.c_[ra.T/state.geometry.scene_unit_m,np.zeros(3)],dcb_shared=np.c_[rb.T/state.geometry.scene_unit_m,np.zeros(3)])
        mounts,_=native_motion(**derivative,minimum_source_depth_scene_units=1./state.geometry.scene_unit_m,schur=True)
        for i,col in enumerate(b.optics_chart.mount_indices):columns.append(int(col));parts.append(mounts[3][:,i]/div)
        for i,col in enumerate(b.optics_chart.indices):columns.append(int(col));parts.append(joptics[:,i]/div)
        compact=np.column_stack(parts);ix,iy=np.nonzero(compact);rows.extend(offset+ix);cols.extend(np.asarray(columns)[iy]);values.extend(compact[ix,iy]);offset+=width
        reports.append(dict(group=group,production_course_J_audit=b.course_latest_motion_column_audit,raw_call_r_max=float(abs(raw/div-b.bundle.motion_kernel(**k)/div).max())))
    return csr_matrix((values,(rows,cols)),shape=(offset,len(q))),reports

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,action='append',required=True)
    ap.add_argument('--reuse-world-layout-proof',type=Path,help='Existing same producer all98 individual-column proof; new points still get full98/world-camera mixed FD and six course columns')
    for key in ('auxiliary','packet','output'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);reports=[]
    for path in a.state:
        state,q,saved,old,camera_error,production_camera_error,contract,proof=producer_state(path,a.auxiliary,a.packet)
        role=state.original_motion_far_board_roles;groups=sorted({(r['source_role_block'],r['source_pts'],r['target_pts'])for r in role.records})
        r=actual_values(state,q,role,groups)
        full_replay=[]
        for b in state.blocks:
            idx=np.array([row['original_global_bundle_index']for row in role.records if row['source_role_block']==b.name],int)
            typed=full_motion_values(b,q,state.structured_programme_coordinate_chart.core_count)
            legacyplanes=b.bundle.finite_planes.copy()
            keep15={27476,27480,27483,27484,27485,27487,27490,27491,27493,27494,27496,27498,27499,27502,27503}
            newidx=np.array([i for i in idx if b.name=='low'or int(i)not in keep15],int)
            legacyplanes[newidx]=0.
            previous=full_motion_values(b,q,state.structured_programme_coordinate_chart.core_count,legacyplanes)
            other=np.ones(len(typed),bool);other[newidx]=False
            assert np.array_equal(typed[other],previous[other])
            if b.name=='secondary':assert np.array_equal(typed[list(keep15)],previous[list(keep15)])
            full_replay.append(dict(block=b.name,all_actual_motion_points=len(typed),all_other_original_motion_values_bitexact=True,
                actual_profile_motion_residual_finite_extent_guard_passed=True,
                original_secondary15_values_preserved=b.name=='secondary',typed_native_norm_px=stats(np.linalg.norm(typed[idx],axis=1)),
                old_target_same_q_native_norm_px=stats(np.linalg.norm(previous[idx],axis=1))))
        print(json.dumps(dict(stage='actual_producer_full_values_and_finite_extent_guard_passed',q_sha256=hashlib.sha256(saved['q'].tobytes()).hexdigest(),blocks=full_replay)),flush=True)
        matrix,production_J_audit=actual_J(state,q,role,groups)
        nominal=role.current_binding(state.geometry,q[state.geometry_parameter_indices])['selected_course_facets']
        def branch_step(d,h):
            steps=[]
            while True:
                changed=0
                for sign in (1.,-1.):
                    value=q+sign*h*d;actual_values(state,value,role,groups);bound=role.current_binding(state.geometry,value[state.geometry_parameter_indices]);changed+=int((bound['selected_course_facets']!=nominal).sum())
                steps.append(dict(h=h,changed_facets=changed))
                if not changed:return h,steps
                h*=.1
                if not np.any(q+h*d!=q):raise ValueError('No representable same-facet step')
        columns=np.unique(np.r_[state.geometry_parameter_indices,matrix.indices]);d=np.zeros(len(q));d[columns]=np.random.default_rng(6853).normal(size=len(columns));d/=max(float(abs(matrix@d).max()),1.)
        h,steps=branch_step(d,1e-4);fd=(actual_values(state,q+h*d,role,groups)-actual_values(state,q-h*d,role,groups))/(2*h)
        error=float(abs(fd-matrix@d).max());assert error<2e-3+2e-5*float(abs(fd).max()),error
        checks=[]
        columns_to_check=range(len(state.geometry_parameter_indices))if a.reuse_world_layout_proof is None else range(44,50)
        for logical in columns_to_check:
            col=state.geometry_parameter_indices[logical]
            direction=np.zeros(len(q));direction[col]=1.;hh,trials=branch_step(direction,1e-6)
            ff=(actual_values(state,q+hh*direction,role,groups)-actual_values(state,q-hh*direction,role,groups))/(2*hh);predicted=matrix[:,col].toarray().ravel();err=float(abs(ff-predicted).max())
            assert err<2e-3+2e-5*float(abs(ff).max()),(logical,err)
            checks.append(dict(world_column=logical,actual_FD_max=float(abs(ff).max()),returned_J_max=float(abs(predicted).max()),error=err,h=hh,branch_trials=trials))
        actual_values(state,q,role,groups)
        from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
        from z190_process_source_evaluation import _minimal_pickle
        install_mappingproxy_pickle();install_geometry_unpickle();worker_checks=[]
        for b in state.blocks:
            before=b.bundle.finite_planes.copy();payload,_=_minimal_pickle(copy(b));worker=pickle.loads(payload)
            assert worker.synchronize_course.args[0]is worker and worker.geometry.finite_board_records.args[0]is worker.geometry
            assert worker._original_motion_far_board_role_binding is worker.geometry._original_motion_far_board_role_binding
            worker.local(q[:state.structured_programme_coordinate_chart.core_count]);assert np.array_equal(worker.bundle.finite_planes,before)
            changed=q.copy();changed[b.start+4*b.bundle.index[next(r['source_pts']for r in role.records if r['source_role_block']==b.name)]]+=1e-7
            previous=worker._original_motion_far_board_role_binding.source_camera_signatures[b.name];worker.local(changed[:state.structured_programme_coordinate_chart.core_count])
            assert previous!=worker._original_motion_far_board_role_binding.source_camera_signatures[b.name]
            worker.local(q[:state.structured_programme_coordinate_chart.core_count]);assert np.array_equal(worker.bundle.finite_planes,before)
            worker_checks.append(dict(block=b.name,actual_minimal_pickle_and_residual_doublecopy_owner_correct=True,payload_bytes=len(payload),
                source_camera_updates_and_nominal_restore_bitexact=True,actual_typed_count=int(worker.bundle.finite_mask.sum())))
        per=[]
        for group in groups:
            b,idx,k,sigma,ca,cb,local=queries(state,q,role,group);typed=b.bundle.motion_kernel(**k);oldk=dict(k,planes=np.zeros_like(k['planes']));untyped=b.bundle.motion_kernel(**oldk)
            per.append(dict(group=group,actual_points=len(idx),typed_native_norm_px=stats(np.linalg.norm(typed.reshape(-1,2),axis=1)),
                untyped_same_q_native_norm_px=stats(np.linalg.norm(untyped.reshape(-1,2),axis=1)),actual_original_sigma=sigma.tolist(),
                frozen_saved_profile_buffers_sha256=hashlib.sha256(b.bundle.profile_weights.tobytes()).hexdigest()))
        reports.append(dict(state=str(path),q_sha256=hashlib.sha256(saved['q'].tobytes()).hexdigest(),producer_motion_point_counts={b.name:len(b.bundle.a)for b in state.blocks},
            source_arrays_UV_PTS_sigma_and_complete_frozen_profile_buffers_bitexact=True,actual_camera_replay_max=camera_error,production_camera_replay_max=production_camera_error,
            production_J_audit=production_J_audit,mixed_full98_world_and_actual_camera_FD=dict(error=error,h=h,branch_steps=steps),
            full_actual_producer_saved_profile_value_replay=full_replay,
            world_uncached_actual_residual_FD=checks,world_column_max_error=max(c['error']for c in checks),worker_checks=worker_checks,
            unchanged_producer_full98_world_layout_proof=None if a.reuse_world_layout_proof is None else dict(path=str(a.reuse_world_layout_proof),sha256=sha(a.reuse_world_layout_proof)),
            per_actual_source_pair=per,new_scalar_rows=0,source_r_invariance_not_claimed=True))
        print(json.dumps(dict(stage='actual68_producer_check_passed',q_sha256=reports[-1]['q_sha256'],mixed_error=error,world_max=reports[-1]['world_column_max_error'],workers=worker_checks)),flush=True)
        stage=dict(schema='rinkspace.actual-original-role-single-state-production-check.v1',packet=dict(path=str(a.packet),sha256=sha(a.packet)),
            role_module_sha256=sha(ROOT/'calibration/solver_runtime/legacy/structured_programme_source_candidate/original_motion_far_board_roles.py'),state=reports[-1])
        stage_path=a.output/('actual_'+reports[-1]['q_sha256']+'_production_check_v1.json');stage_path.write_text(json.dumps(stage,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps(dict(stage='immutable_single_state_proof_saved',path=str(stage_path),sha256=sha(stage_path))),flush=True)
    report=dict(schema='rinkspace.actual-two-producer-original68-finite-role-fullsavedprofile-J-worker-check.v1',packet=dict(path=str(a.packet),sha256=sha(a.packet)),
        role_module_sha256=sha(ROOT/'calibration/solver_runtime/legacy/structured_programme_source_candidate/original_motion_far_board_roles.py'),states=reports,
        original_secondary15_preserved=True,low_original53_material_operator_explicitly_changed=True,full_programme_graph_J_solver_or_video_not_started=True)
    target=a.output/'actual_original68_production_binder_fullsavedprofile_FD_worker_check_v1.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(target),sha256=sha(target),status='passed')),flush=True)
if __name__=='__main__':main()
