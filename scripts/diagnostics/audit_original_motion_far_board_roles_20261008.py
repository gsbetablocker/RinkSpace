"""Real original FIT rows: current plane/extent/actual-q J replay."""
from pathlib import Path
import argparse,hashlib,json,os,sys,importlib.util
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import cv2,numpy as np
from scipy.sparse import csr_matrix
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.original_motion_far_board_roles import (
    OriginalMotionFarBoardRoleBinding,freeze_role_admission,PAIR)
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from z190_motion_native import native_motion
from secondary_optics_candidate.z190_motion_optics_extension import native_motion_with_optics
from shared_board_course_candidate.finite_plane_motion_columns import prepare_finite_plane_motion_cache,finite_plane_motion_columns


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,float);return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x)for x in v]
    return v


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--state',required=True,type=Path);cli.add_argument('--camera-identity',required=True,type=Path)
    cli.add_argument('--auxiliary',required=True,type=Path);cli.add_argument('--candidate',required=True,type=Path)
    cli.add_argument('--admitted',required=True,type=Path);cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--pair',nargs=2,type=int,default=PAIR,help='Exact native source/target PTS of the original FIT queries')
    cli.add_argument('--all-admitted-pairs',action='store_true',help='Replay every admitted pair from the same physical installation in one small numeric check')
    cli.add_argument('--authority',default='Parent root authorized necessary actual-RGB-supported original material-role correction on 2026-10-08; original fifteen FIT source rows retained')
    cli.add_argument('--source-role-api',type=Path,help='Authenticated Data-only public role API draft; camera/world/native producers remain public')
    cli.add_argument('--reuse-admitted',action='store_true',help='Authenticate an already explicitly admitted immutable packet, preserving its authority')
    a=cli.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pair=tuple(a.pair)
    role_class=OriginalMotionFarBoardRoleBinding;freeze=freeze_role_admission
    if a.source_role_api:
        spec=importlib.util.spec_from_file_location('structured_programme_source_candidate.original_motion_far_board_roles_Data_audit',a.source_role_api)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        role_class=module.OriginalMotionFarBoardRoleBinding;freeze=module.freeze_role_admission
    if a.reuse_admitted:
        packet=read(a.admitted)
        if packet['admission']['parent_candidate']!={'path':str(a.candidate.resolve()),'sha256':sha(a.candidate)}:
            raise ValueError('Existing immutable role packet source qualification changed')
        admitted=dict(path=str(a.admitted.resolve()),sha256=sha(a.admitted),original_typed_points=packet['source_qualified_far_board_count'],new_scalar_rows=0)
    else:admitted=freeze(a.candidate,a.admitted,authority=a.authority)
    binding=role_class(a.admitted)
    if not a.all_admitted_pairs and any((r['source_pts'],r['target_pts'])!=pair for r in binding.records):
        raise ValueError('Small same-pair replay requires every admitted row on the requested exact native pair')
    source_pts=np.asarray([r['source_pts']for r in binding.records],np.int64)
    target_pts=np.asarray([r['target_pts']for r in binding.records],np.int64)
    all_stamps=np.unique(np.r_[source_pts,target_pts])
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names','aux__installation_xyz_slots',
        'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices',
        'aux__correlated_roll_prior_contract_json','aux__current_complete_parameter_layout_json')
    with np.load(a.state,allow_pickle=False)as z:
        small={k:z[k].copy()for k in keys}
        names={r['source_role_block']for r in binding.records}
        if len(names)!=1:raise ValueError('Small value/J replay requires one original source block')
        block_name=next(iter(names));bi=small['aux__block_names'].tolist().index(block_name)
        ids=np.array([r.get('active_global_bundle_index',r.get('current_compact_bundle_index',r['original_global_bundle_index']))for r in binding.records],int)
        for i,row in enumerate(binding.records):
            if ids[i]<0:ids[i]=int(row['current_runtime_equivalent_compact_bundle_index'])
        weights=z['profile__weights_'+str(bi)][ids].copy()
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    identity=read(a.camera_identity);cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    if sha(identity['camera_path'])!=identity['camera_sha256']:raise ValueError('Immutable real camera archive changed')
    state,camera_error=state_camera_adapter(small,cameras,provider);q=state.evaluation_q(small['q'])
    block=state.bypts[int(source_pts[0])];unit=provider.scene_unit_m
    if any(state.bypts[int(stamp)]is not block for stamp in all_stamps):
        raise ValueError('Small multi-pair check requires one actual physical camera block')
    p=np.asarray([r['source_uv']for r in binding.records]);b=np.asarray([r['target_uv']for r in binding.records]);sigma=np.asarray([r['source_sigma_px']for r in binding.records],float)
    def inputs(value):
        local=block.local(value[:state.structured_programme_coordinate_chart.core_count])
        actual={int(stamp):block.bundle.camera(local,int(stamp))for stamp in all_stamps}
        ca=np.asarray([actual[int(stamp)]for stamp in source_pts])
        cb=np.asarray([actual[int(stamp)]for stamp in target_pts])
        rotations={stamp:cv2.Rodrigues(camera[3:6])[0]for stamp,camera in actual.items()}
        binding.source_cameras={int(stamp):actual[int(stamp)]for stamp in np.unique(source_pts)}
        bound=binding.current_binding(provider,value[state.geometry_parameter_indices])
        k=dict(a=p,b=b,fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),ra=np.asarray([rotations[int(stamp)]for stamp in source_pts]),rb=np.asarray([rotations[int(stamp)]for stamp in target_pts]),
            ca=ca[:,:3],cb=cb[:,:3],pp=ca[0,8:10],k1=ca[0,11],planes=bound['planes'],sigma=sigma,profile_weight=weights,
            da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,jacobian=False)
        return k,bound,ca,cb
    def evaluate(value):
        k,bound,_,_=inputs(value);r,checks,trace=native_motion(**k,minimum_source_depth_scene_units=1./unit,schur=True,return_trace=True)
        return r/sigma.reshape(-1),bound,trace
    r,bound,trace=evaluate(q);k,_,ca,cb=inputs(q)
    # The production current-course J helper consumes exactly these six columns.
    ng=provider.parameter_count;other=np.setdiff1d(np.arange(ng),np.arange(44,50))
    if np.any(bound['plane_jacobian'][:,:,other]!=0.):raise ValueError('Current plane has noncourse material dependence')
    cache=prepare_finite_plane_motion_cache(k,np.ones(len(p),bool))
    _,jg,course_audit=finite_plane_motion_columns(cache,bound['planes'],bound['plane_jacobian'][:,:,44:50],sigma_normalized=True)
    columns=[];values=[]
    for i,col in enumerate(state.geometry_parameter_indices[44:50]):columns.append(int(col));values.append(jg[:,i])
    ra,rb=k['ra'],k['rb']
    da=np.asarray([cv2.Rodrigues(camera[3:6])[1].reshape(3,3,3)for camera in ca])
    db=np.asarray([cv2.Rodrigues(camera[3:6])[1].reshape(3,3,3)for camera in cb])
    lever=np.asarray(block.bundle.calibration.lever_xyz)
    dva=np.zeros((len(p),3,4));dvb=np.zeros((len(p),3,4))
    dva[:,:,:3]=np.einsum('naij,i->nja',da,lever);dvb[:,:,:3]=np.einsum('naij,i->nja',db,lever)
    derivative=dict(k,jacobian=True,da=da,db=db,dca_view=dva,dcb_view=dvb,
        dca_shared=np.c_[np.eye(3)/unit,np.zeros(3)],dcb_shared=np.c_[np.eye(3)/unit,np.zeros(3)])
    result,_,_=native_motion_with_optics(**derivative,minimum_source_depth_scene_units=1./unit,schur=True)
    rr,ja,jb,jpivot,joptics=result
    if np.max(abs(rr/sigma.reshape(-1)-r))>1e-12:raise ValueError('Analytic camera derivative call changed source r')
    for i,col in enumerate(block.xyz_slots):columns.append(int(col));values.append(jpivot[:,i]/sigma.reshape(-1))
    for stamp in all_stamps:
        camera_columns=ja*np.repeat(source_pts==stamp,2)[:,None]+jb*np.repeat(target_pts==stamp,2)[:,None]
        view=block.start+4*block.bundle.index[int(stamp)]
        for i in range(4):columns.append(view+i);values.append(camera_columns[:,i]/sigma.reshape(-1))
    derivative.update(dca_shared=np.concatenate((ra.transpose(0,2,1)/unit,np.zeros((len(p),3,1))),axis=2),
        dcb_shared=np.concatenate((rb.transpose(0,2,1)/unit,np.zeros((len(p),3,1))),axis=2))
    mounts,_=native_motion(**derivative,minimum_source_depth_scene_units=1./unit,schur=True)
    for i,col in enumerate(block.optics_chart.mount_indices):columns.append(int(col));values.append(mounts[3][:,i]/sigma.reshape(-1))
    for i,col in enumerate(block.optics_chart.indices):columns.append(int(col));values.append(joptics[:,i]/sigma.reshape(-1))
    compact=np.column_stack(values);ix,iy=np.nonzero(compact)
    matrix=csr_matrix((compact[ix,iy],(ix,np.asarray(columns)[iy])),shape=(len(r),len(q)))
    def step(direction,h):
        trials=[]
        while True:
            plus,minus=q+h*direction,q-h*direction
            if not np.any(plus!=q)or not np.any(minus!=q):raise ValueError('No representable same-facet validation step')
            kp,bp,_,_=inputs(plus);km,bm,_,_=inputs(minus)
            changed=(bp['selected_course_facets']!=bound['selected_course_facets'])|(bm['selected_course_facets']!=bound['selected_course_facets'])
            trials.append(dict(h=h,changed_course_facets=int(changed.sum())))
            if not np.any(changed):return h,trials
            h*=.1
    mixed=np.zeros(len(q));cols=np.unique(np.r_[state.geometry_parameter_indices,columns]);mixed[cols]=np.random.default_rng(15230509824).normal(size=len(cols))
    mixed/=max(float(abs(matrix@mixed).max()),1.)
    h,trials=step(mixed,1e-4);mixed_h=h;fd=(evaluate(q+h*mixed)[0]-evaluate(q-h*mixed)[0])/(2*h)
    error=float(abs(fd-matrix@mixed).max());allowed=2e-3+2e-5*float(abs(fd).max())
    if error>allowed:raise AssertionError(('Actual mixed q camera/world finite-role FD failed',error,allowed))
    checks=[]
    for logical,col in enumerate(state.geometry_parameter_indices):
        d=np.zeros(len(q));d[int(col)]=1.;h,tr=step(d,1e-6);fd=(evaluate(q+h*d)[0]-evaluate(q-h*d)[0])/(2*h)
        predicted=matrix[:,int(col)].toarray().ravel();err=float(abs(fd-predicted).max());tol=2e-3+2e-5*float(abs(fd).max())
        rec=dict(logical_world_column=logical,h=h,branch_trials=tr,actual_uncached_FD_max=float(abs(fd).max()),
            returned_J_max=float(abs(predicted).max()),error=err,allowed=tol,passed=err<=tol)
        checks.append(rec)
        if not rec['passed']:raise AssertionError(rec)
    axial=(1.-trace[:,1])/trace[:,1]*unit
    untyped=dict(k,planes=np.zeros_like(k['planes']));old,_=native_motion(**untyped,minimum_source_depth_scene_units=1./unit,schur=True)
    report=dict(schema='rinkspace.actual-original-motion-source-role-current-binding-FD-check.v2',admitted_packet=admitted,
        original_native_pair_pts=sorted(set(zip(source_pts.tolist(),target_pts.tolist()))),actual_original_FIT_point_count=len(p),
        state=str(a.state),state_q_sha256=hashlib.sha256(np.asarray(small['q']).tobytes()).hexdigest(),world_count=ng,
        q_evaluation_shape=list(q.shape),actual_camera_archive_reproduce_max=camera_error,
        typed_native_residual_px=stats(np.linalg.norm((r*sigma.reshape(-1)).reshape(-1,2),axis=1)),
        old_untyped_same_q_native_residual_px=stats(np.linalg.norm(old.reshape(-1,2),axis=1)),
        typed_source_axial_depth_working_m=stats(axial),typed_minimum_one_metre_floor_not_active=bool(np.all(axial>1.)),
        all_typed_native_trace_known=bool(np.all(trace[:,12])),typed_depth_schur_or_unknown_profile_absent=bool(not np.any(trace[:,9])),
        current_course_selected_facets=bound['selected_course_facets'],course_J_helper=course_audit,
        mixed_full_world_and_actual_camera_FD=dict(h=mixed_h,error=error,allowed=allowed,branch_trials=trials),
        all_world_column_checks=checks,all_world_max_error=max(c['error']for c in checks),
        old_r_equal_not_claimed=True,original_global_indices=binding.indices,UV_PTS_sigma_rows_and_frozen_weights_preserved=True,
        full_motion_graph_constructed=False,full_global_source_J_constructed=False,source_scalar_rows_added=0,accuracy_accepted=False)
    report.update(source_role_block=block_name,actual_frozen_profile_weight_point_indices=ids,
        authenticated_Data_source_role_API=dict(path=str(a.source_role_api),sha256=sha(a.source_role_api))if a.source_role_api else None,
        original_exact_sigma_from_admitted_source_records=True,
        original_history_duplicate_queries_in_diagnostic=int(sum(r.get('current_compact_bundle_index',0)<0 for r in binding.records)),
        historical_duplicate_diagnostic_rows_do_not_claim_runtime_double_counting=True)
    target=a.output/'original_motion_far_board_roles_all_world_FD_check_v1.json';target.write_text(json.dumps(serial(report),indent=2)+'\n',encoding='utf8')
    np.savez_compressed(a.output/'original_motion_far_board_role_arrays_v1.npz',native_typed_residual=r*sigma.reshape(-1),native_untyped_residual=old,
        planes=bound['planes'],plane_world_jacobian=bound['plane_jacobian'],extent_world_jacobian=bound['extent_world_jacobian'],native_trace=trace)
    print(json.dumps(serial({k:report[k]for k in ('admitted_packet','typed_native_residual_px','old_untyped_same_q_native_residual_px',
        'typed_source_axial_depth_working_m','mixed_full_world_and_actual_camera_FD','all_world_max_error')})),flush=True)


if __name__=='__main__':main()
