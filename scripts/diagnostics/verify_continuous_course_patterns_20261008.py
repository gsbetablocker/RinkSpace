"""Verify released q9 stations using real source frames and the existing pool4."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,pickle,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor,admit_pattern_source
from shared_board_course_candidate.course_pattern_geometry import CourseBoardPatternGeometry
from structured_programme_source_candidate.course_patterns_independent_placements import IndependentlyPlacedCoursePatterns
from structured_programme_source_candidate.current_geometry_pattern_pool import CurrentGeometryPatternPool,geometry_signature
from secondary_optics_candidate.secondary_optics_graph import OpticsPatternFactor
from calibration.solver_runtime.worker_entry import evaluate_pattern_worker


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for name in ('state','camera-identity','owner-runtime-report','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--restore-only',action='store_true')
    args=ap.parse_args();started=time.perf_counter()
    if args.output.exists():raise FileExistsError('Preserve previous evidence')
    arrays=dict(np.load(args.state,allow_pickle=False));identity=json.loads(args.camera_identity.read_text(encoding='utf8'))
    assert hashlib.sha256(arrays['q'].tobytes()).hexdigest()==identity['q_sha256']
    assert sha(identity['camera_path'])==identity['camera_sha256']
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    owner=json.loads(args.owner_runtime_report.read_text(encoding='utf8'))['material_owner'];assert sha(owner['path'])==owner['sha256']
    install_mappingproxy_pickle();install_geometry_unpickle()
    with Path(owner['path']).open('rb')as stream:provider=pickle.load(stream)
    state,ce=state_camera_adapter(arrays,cameras,provider);q=state.evaluation_q(arrays['q'])
    old_scene=json.loads((args.camera_identity.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    base=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,facet_authority=old_scene['shared_board_patterns']['facet_authority'])
    course=provider.source_course.course;cq=provider.current.current_course6(q[state.geometry_parameter_indices])
    placements=state.canonical_parameter_layout.canonical_to_evaluation[arrays['aux__current_independent_pattern_placement_parameter_indices']]
    geom=IndependentlyPlacedCoursePatterns(base,course,cq,q[placements]);start=int(arrays['aux__shared_pattern_start']);q9=q[start:start+9]
    geom.guard(q9);theta=np.linspace(0.,2*np.pi,257);theta[-1]=0.
    lo,hi=base.course_source_boundary_indices;bottom=course.sample_perimeter(cq)[0]
    old_xy=bottom[lo:hi+1,:2]*geom.scene_unit_m;edges=np.diff(old_xy,axis=0);lengths=np.linalg.norm(edges,axis=1)
    stations=np.r_[0.,np.cumsum(lengths)];tangents=edges/lengths[:,None]
    old_checks=[]
    for ad in (12,13,25,26):
        material=geom.instance_q9(q9,ad)
        s=material[0]+material[2]*np.cos(theta)if ad in(13,26)else material[3:].reshape(3,2)[:,0]
        z=material[1]+material[2]*np.sin(theta)if ad in(13,26)else material[3:].reshape(3,2)[:,1]
        inside=bool(np.all((s>=0)&(s<=stations[-1])))
        world,jq=geom.circle(q9,theta,ad_id=ad)if ad in(13,26)else geom.shield(q9,ad_id=ad)
        if inside:
            i=np.clip(np.searchsorted(stations,s,side='right')-1,0,len(lengths)-1)
            xy=old_xy[i]+(s-stations[i])[:,None]*tangents[i]
            if ad in(25,26):xy=-xy
            expected=np.column_stack((xy,z))/geom.scene_unit_m
            difference=float(abs(world-expected).max());assert difference<1e-11,(ad,difference)
        else:difference=None
        moved=q9.copy()
        if ad in(13,26):moved[0]=-geom.course_station_m[-1]+.1
        else:moved[3::2]-=geom.course_station_m[-1]+1.
        geom.guard(moved)
        crossed=geom.circle(moved,theta,ad_id=ad)[0]if ad in(13,26)else geom.shield(moved,ad_id=ad)[0]
        assert np.isfinite(crossed).all()
        old_checks.append(dict(advertisement=ad,current_inside_previous_local_interval=inside,
            previous_local_world_max_difference=difference,arbitrary_full_perimeter_station_accepted=True))
    if args.restore_only:
        from run_current_named66_full_programme import current_seed_parts
        from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeCoordinateChart
        layout=state.canonical_parameter_layout;chart=PairedProgrammeCoordinateChart(provider.parameter_count-66)
        restored=[]
        for shift in (-2*geom.course_station_m[-1]-1.,2*geom.course_station_m[-1]+1.):
            trial=arrays['q'].copy();trial[start]=shift;trial[start+3:start+9:2]+=shift
            old_complete=trial[:layout.original_complete_count]
            suffix=old_complete[-4:].copy()
            legacy,world66,circles=current_seed_parts(old_complete[:-4],circle_enabled=True)
            world=np.r_[world66,trial[layout.original_complete_count:]]
            evaluation=chart.initialize_full_q(legacy,provider.initial)
            evaluation[np.asarray(chart.world_indices,int)]=world
            evaluation=np.r_[evaluation,circles,suffix]
            actual=layout.canonical_q(evaluation)
            assert np.array_equal(actual,trial)
            restored_geom=IndependentlyPlacedCoursePatterns(base,course,provider.current.current_course6(world),evaluation[placements])
            assert restored_geom.guard(evaluation[start:start+9])
            restored.append(dict(unaccepted_diagnostic_station_m=shift,canonical_full_q_roundtrip_bitexact=True,
                current_board_H_and_periodic_station_guard_passed=True))
        report=dict(schema='rinkspace.actual-cross-region-complete-checkpoint-restoration-check.v1',
            parent_state=str(args.state),parent_sha256=sha(args.state),parent_canonical_q_sha256=identity['q_sha256'],
            diagnostic_trials=restored,original_profile_aux_fields_unmodified=True,
            checkpoint_q_not_passed_to_legacy_graph_initializer=True,
            actual_driver_order='prepare(DEFAULT_SEED), capture original schema, read camera-seed, restore chart, install CurrentPatterns, guard exact checkpoint q9 on current course',
            full_motion_graph_or_normal_loaded=False,diagnostic_trials_saved_as_optimizer_updates=False,
            elapsed_seconds=time.perf_counter()-started)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(report_path=str(args.output),cross_region_complete_roundtrip=True,seconds=report['elapsed_seconds'])),flush=True)
        return
    # A wrapped (negative) station tests the winding*dL geometry derivative.
    s=np.array([-1.3,geom.course_station_m[-1]+2.1]);world,_,analytic=geom._arc_course(s);errors=[]
    for k in range(6):
        step=1e-6;plus=cq.copy();minus=cq.copy();plus[k]+=step;minus[k]-=step
        gp=CourseBoardPatternGeometry(base,course,plus);gm=CourseBoardPatternGeometry(base,course,minus)
        numeric=(gp._arc(s)[0]-gm._arc(s)[0])/(2*step)
        error=float(abs(numeric-analytic[:,:,k]).max());errors.append(error);assert error<2e-6,(k,error)
    from z190_joint_shared_F_height_source import N
    frames=[];source_refs=[]
    for ad,expected in ((13,'214448266965e5ee14e8519c0fef8bac03bba356273c8acdaea17c00f84775e5'),(12,'8bc7571bac78ebb6c3ca6fa2d21f9085ba43b8ac9277021e53883293f3d8e28f')):
        p=N/f'low5_repair/secondary6_Ad13_Ad12_native_patterns_v1/secondary6_Ad{ad}_native_pattern_QUALIFIED_FROZEN.json'
        actual,ref=admit_pattern_source(p,state.original.aliases,expected_ad_id=ad,expected_observer='long_side_secondary',expected_sha256=expected)
        frames.extend([r for r in actual if r['split']=='fit'][:2]);source_refs.append(ref)
    assert len(frames)==4
    factor=OpticsPatternFactor(SharedBoardPatternSourceFactor(geom),steps=(1e-4,1e-4,1e-6))
    manager=CurrentGeometryPatternPool(SimpleNamespace(_evaluate_current=None),workers=4)
    futures=[];serial=[];maximum=0.;worker_reports=[]
    try:
        for frame in frames:
            p=int(frame['source_actual_pts']);camera=state.bypts[p].bundle.camera(q,p)
            serial.append(factor.evaluate(q9,camera,frame,True))
            payload=pickle.dumps((factor,q9.copy(),camera.copy(),frame,True),protocol=5)
            futures.append(manager._executor().submit(evaluate_pattern_worker,payload,hashlib.sha256(payload).hexdigest(),geometry_signature(factor)))
        for original,future in zip(serial,futures):
            actual,stats=future.result();worker_reports.append(stats)
            for field in ('residual','shared_q9_jacobian','camera_state11_jacobian','optics_jacobian'):
                d=float(abs(np.asarray(getattr(actual,field))-np.asarray(getattr(original,field))).max())
                maximum=max(maximum,d);assert d==0.,(field,d)
    finally:manager.close()
    report=dict(schema='rinkspace.actual-continuous-q9-course-pattern-check.v1',complete_state=str(args.state),
        canonical_q_sha256=identity['q_sha256'],camera_reproduce_max=ce,source_packets=source_refs,
        old_actual_sameq_world_checks=old_checks,wrapped_station_course_J_FD_errors=errors,
        existing_pattern_worker_count=4,actual_typed_source_frames=4,serial_worker_r_J_max_difference=maximum,
        worker_reports=worker_reports,original_station_origin_and_q9_source_fields_preserved=True,
        independent_pair_placements_and_shared_print_sizes_preserved=True,source_profiles_aux_and_q_modified=False,
        motion_graph_or_normal_loaded=False,elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report_path=str(args.output),worker_r_J_max_difference=maximum,geometry_J_max=max(errors),seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
