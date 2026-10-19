"""Actual complete-state check of continuous board material and source routes.

No motion graph, full normal matrix or new observations are created.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,pickle,sys,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor,straight_board_circle_world
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth
from structured_programme_source_candidate.current_wide_source import CurrentWideSource
from rinkspace.vision_camera_initialization import frozen_scene_curve


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--camera-identity',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--owner-runtime-report',type=Path,required=True)
    ap.add_argument('--authority-directory',type=Path,required=True)
    ap.add_argument('--location-envelope',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();started=time.perf_counter()
    if args.output.exists():raise FileExistsError('Preserve the existing check')
    arrays=dict(np.load(args.state,allow_pickle=False));identity=json.loads(args.camera_identity.read_text(encoding='utf8'))
    assert hashlib.sha256(arrays['q'].tobytes()).hexdigest()==identity['q_sha256']
    assert sha(identity['camera_path'])==identity['camera_sha256']
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    owner=json.loads(args.owner_runtime_report.read_text(encoding='utf8'))['material_owner']
    assert sha(owner['path'])==owner['sha256']
    install_mappingproxy_pickle();install_geometry_unpickle()
    with Path(owner['path']).open('rb')as stream:provider=pickle.load(stream)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    q=state.evaluation_q(arrays['q']).copy();manifest=json.loads(args.manifest.read_text(encoding='utf8'))
    auth=args.authority_directory
    scope=auth/'user20_existing_lower_long_side_connected_lower_right_scope_authority_v1.json'
    state.current_user_physical_course_scope_source=scope
    board=NativeBoardEdgeFactor(state,manifest['additional_real_native_board_edges']['packet_path'],len(q))
    ice=NativeIceCurveFactor(state,manifest['additional_real_native_ice_curves']['packet_path'],len(q))
    cref=manifest['additional_real_native_board_circles']
    circle=NativeBoardCircleFactor(state,cref['packet_path'],int(cref['prefix_parameter_count']))
    qc=q[:circle.parameter_count];cq=circle._course(qc)
    course,_,_=circle.geometry.print_course_binding(qc[circle.world_indices])
    scene=provider.scene(q[state.geometry_parameter_indices])
    scene=circle.export_scene(scene,qc)
    world_checks=[]
    for row in board.rows+board.held:
        curve=dict(board.effective_curve(row['curve']))
        if curve['kind']=='advertisement_boundary':curve['extent_scope']=row['actual_modeled_extent_scope']
        actual=board._world(q,row['curve']);exported=frozen_scene_curve(scene,curve)
        assert actual.shape==exported.shape,(row['id'],actual.shape,exported.shape)
        difference=float(abs(actual-exported).max());assert difference<5e-14,(row['id'],difference)
        world_checks.append(dict(id=row['id'],shape=list(actual.shape),export_max_difference=difference))
    circle_checks=[]
    for number in (11,24):
        material=circle._material(qc,number);a,b,rx,ry,p,h=cq;unit=circle.unit
        theta=np.linspace(0.,2*np.pi,361);co=np.cos(theta);si=np.sin(theta);co[-1]=co[0];si[-1]=si[0]
        old=np.column_stack((np.full(len(theta),(-1 if number==11 else 1)*a/unit),
            (-b+ry+material[0]+material[2]*co)/unit,(material[1]+material[2]*si)/unit))
        actual=circle.world(qc,number)
        inside=(material[0]-material[2]>=0. and material[0]+material[2]<=2*(b-ry))
        old_difference=float(abs(actual-old).max())
        if inside:assert old_difference<1e-11,(number,old_difference)
        exported=frozen_scene_curve(scene,dict(kind='board_circle',advertisement_id=number,edge='outermost_visible_circular_print_stroke'))
        assert np.array_equal(actual,exported)
        crossings=[]
        for tangent in (0.,2*(b-ry)):
            moved=material.copy();moved[0]=tangent
            current=straight_board_circle_world(cq,moved,unit,number,course=course)
            assert np.array_equal(current[0],current[-1])
            step=1e-6;plus=moved.copy();minus=moved.copy();plus[0]+=step;minus[0]-=step
            fp=straight_board_circle_world(cq,plus,unit,number,course=course)
            fm=straight_board_circle_world(cq,minus,unit,number,course=course)
            assert np.isfinite((fp-fm)/(2*step)).all()
            continuity=float(abs(fp-fm).max());assert continuity<=2.001e-6/unit
            crossings.append(dict(legacy_station_at_tangent_m=tangent,closed=True,station_delta_2um_world_max=continuity))
        circle_checks.append(dict(advertisement_number=number,current_material=material.tolist(),
            current_circle_still_inside_legacy_straight=bool(inside),legacy_straight_world_max_difference=old_difference,
            source_export_world_bitexact=True,crosses_both_computational_tangents=crossings))
    cb=circle.block(qc,True);d=np.zeros(len(qc))
    d[circle.circle_indices[:5]]=[.31,-.17,.08,-.23,.13]
    d/=max(float(abs(cb.J@d).max()),1.);step=1e-4
    fd=(circle.block(qc+step*d,False).r-circle.block(qc-step*d,False).r)/(2*step)
    je=float(abs(fd-cb.J@d).max());assert je<2e-3,je
    print(json.dumps(dict(stage='continuous_circle_actual_J_and_export_verified',directional_error=je)),flush=True)
    bb=board.block(q,True);bd=np.zeros(len(q));cols=board.active
    excluded=state.geometry_parameter_indices[[47,56,57,71]]
    cols=cols[~np.isin(cols,excluded)];bd[cols]=np.random.default_rng(230024192).normal(size=len(cols))
    bd/=max(float(abs(bb.J@bd).max()),1.)
    step,support=board.local_support_directional_step(q,bd,1e-4)
    assert support['same_support_branch_step_found']
    fd=(board.block(q+step*bd,False).r-board.block(q-step*bd,False).r)/(2*step)
    be=float(abs(fd-bb.J@bd).max());assert be<2e-3,(be,step)
    truth=HumanStrokeTruth(state,auth/'human_registered_hard_truth_authorities_v1.json',board,ice)
    truth.bind_full_stroke_guard(auth/'human_registered_full_stroke_hard_guard_v1.json')
    truth.bind_stroke_cap_support(auth/'human_registered_stroke_end_2D_source_footprint_support_v2.json')
    truth.bind_practical_location_envelope(args.location_envelope)
    human=truth.report(q)
    old_scene=json.loads((args.camera_identity.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    exported=[r['camera25']for r in old_scene['frame_cameras']if r.get('source_pts')==56218880]
    assert len(exported)==1
    wauth=auth/'human_original_wide_top_contours_hard_authority_v1.json'
    wmeta=json.loads(wauth.read_text(encoding='utf8'))
    trace=json.loads(Path(wmeta['trace_packet_reference']['path']).read_text(encoding='utf8'))
    wide=CurrentWideSource.__new__(CurrentWideSource);wide.geometry=provider.current;wide.qualified_scene=provider.qualified_scene
    wide.geometry_parameter_indices=state.geometry_parameter_indices;wide.observations=[];wide.ad_packet={'rows':[]}
    wide.template_camera25=np.asarray(exported[0],float);wide.trace_packet=trace
    wide.trace_uv=[np.asarray(r['observed_centres_source_px'],float)for r in trace['contours']]
    wi=state.canonical_parameter_layout.canonical_to_evaluation[np.asarray(arrays['aux__current_wide_camera_parameter_indices'],int)]
    assert np.array_equal(wide.camera(q[wi]),exported[0])
    wt=WideStrokeTruth(wide,wauth,len(q),wi);wt.bind_practical_location_envelope(args.location_envelope)
    wr=wt.report(q)
    report=dict(schema='rinkspace.actual-continuous-board-source-material-check.v1',
        complete_state=str(args.state),complete_state_sha256=sha(args.state),canonical_q_sha256=identity['q_sha256'],
        actual_camera_archive=str(identity['camera_path']),camera_reproduce_max=camera_error,
        material_owner=owner,world_checks=world_checks,circle_checks=circle_checks,
        circle_FIT_scalar_rows=len(cb.r),circle_directional_J_error=je,board_FIT_scalar_rows=len(bb.r),
        board_directional_J_error=be,board_support_directional_step=step,native_human_location_guard=human,
        original_wide_human_location_guard=wr,source_packets_UV_sigma_profiles_roles_and_q_unchanged=True,
        whole_motion_graph_or_normal_loaded=False,new_geometry_or_solver_added=False,
        old_sameq_J_not_claimed_reusable_after_source_math_revision=True,
        accuracy_accepted=False,elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report_path=str(args.output),native_manual_feasible=human['feasible'],wide_manual_feasible=wr['feasible'],
        circle_directional_J_error=je,board_directional_J_error=be,seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
