"""Export a canonical complete initializer through existing material/camera APIs.

Loads the authenticated material owner, not the programme motion graph or J.
This export is visual evidence; it does not accept the complete fitted target.
"""
from pathlib import Path
from types import MethodType,SimpleNamespace
from dataclasses import replace
import argparse, copy, hashlib, json, os, pickle, shutil, sys, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from calibration.program_tracking.z190_physical_camera import PhysicalCameraModel,CanonicalView,SharedCalibration,ViewState
from structured_programme_source_candidate.current_full_source import CurrentPatterns
from structured_programme_source_candidate.course_patterns_independent_placements import _bind_independent_current_geometry
from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.current_wide_source import CurrentWideSource
from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor
from rinkspace.user_contours import apply_user_visible_markings
from rinkspace.vision_camera_initialization import frozen_scene_curve
from calibration.program_tracking.render_result import advertisement_review_paths,fixed_scene_paths
from z190_joint_source_solver import _frozen_profile_token


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def qsha(value):return hashlib.sha256(np.asarray(value,dtype=np.float64).tobytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        def numeric(item):
            if isinstance(item,np.generic):return item.item()
            if isinstance(item,np.ndarray):return item.tolist()
            raise TypeError('Unsupported scene value: '+type(item).__name__)
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False,default=numeric);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state','parent-identity','owner-runtime-report','input-manifest','physical-course-scope-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    if args.output.exists():raise ValueError('Preserve prior evidence; output must be new')
    parent=read(args.parent_identity)
    parent_state=args.parent_identity.parent/'complete_state_UNACCEPTED.npz'
    parent_scene_path=args.parent_identity.parent/'scene_UNACCEPTED.json'
    for path,expected in ((parent_state,parent['state_npz_sha256']),
                          (parent_scene_path,parent['scene_sha256']),
                          (parent['camera_path'],parent['camera_sha256']),
                          (args.input_manifest,parent['input_manifest_sha256'])):
        if sha(path)!=expected:raise ValueError('Parent identity changed: '+str(path))
    archived=dict(np.load(parent_state,allow_pickle=False));arrays=dict(np.load(args.state,allow_pickle=False))
    phase_identity=json.loads(str(arrays['identity_json'].item()))
    phase_parent=phase_identity.get('parent',{})
    if phase_parent.get('q_sha256')!=parent['q_sha256'] or sha(phase_parent['path'])!=phase_parent['sha256']:
        raise ValueError('Initializer actual complete parent identity differs')
    old=dict(np.load(phase_parent['path'],allow_pickle=False))
    if not np.array_equal(old['q'],archived['q']):raise ValueError('Actual and archived parent canonical q differ')
    for key in old:
        if key.startswith('aux__') and not np.array_equal(old[key],archived[key]):
            raise ValueError('Actual and archived parent auxiliary differs: '+key)
    if set(old)!=set(arrays):raise ValueError('Complete initializer array schema differs from parent')
    for key in old:
        if key not in ('q','identity_json','history_json','elapsed_seconds','next_line_search_fraction') and not np.array_equal(old[key],arrays[key]):
            raise ValueError('Original auxiliary/profile changed: '+key)
    if qsha(old['q'])!=parent['q_sha256']:raise ValueError('Parent canonical q identity differs')
    canonical=np.asarray(arrays['q'],float);canonical_hash=qsha(canonical)
    def profile_snapshot(values):
        return SimpleNamespace(iter_checkpoint_arrays=lambda:((key[9:],value) for key,value in values.items() if key.startswith('profile__')))
    if _frozen_profile_token(profile_snapshot(archived))!=parent['frozen_profile_token']:
        raise ValueError('Saved profile iterator does not reproduce original profile token')
    profile_token=_frozen_profile_token(profile_snapshot(arrays))
    cameras=dict(np.load(parent['camera_path'],allow_pickle=False));metadata=json.loads(str(cameras['metadata_json'].item()))
    manifest=read(args.input_manifest);parent_scene=read(parent_scene_path)
    owner=read(args.owner_runtime_report)['material_owner']
    if sha(owner['path'])!=owner['sha256']:raise ValueError('Actual material owner identity changed')
    install_mappingproxy_pickle();install_geometry_unpickle()
    provider=pickle.loads(Path(owner['path']).read_bytes())
    state,parent_camera_error=state_camera_adapter(old,cameras,provider)
    state.current_user_physical_course_scope_source=args.physical_course_scope_source
    layout=state.canonical_parameter_layout
    if layout is None or canonical.shape!=(layout.count,):raise ValueError('Complete canonical parameter layout required')
    evaluation=state.evaluation_q(canonical);reference=state.evaluation_q(old['q'])
    if not np.array_equal(layout.canonical_q(evaluation),canonical):raise ValueError('Canonical/evaluation round-trip differs')
    core_count=state.structured_programme_coordinate_chart.core_count
    core=evaluation[:core_count];world=evaluation[state.geometry_parameter_indices]
    provider.validate_parameters(world)
    packet_refs={key:manifest[key] for key in ('additional_real_native_board_edges',
        'additional_real_native_board_material_points','additional_real_native_board_circles',
        'additional_native_static_board_tracks')}
    for ref in packet_refs.values():
        if sha(ref['packet_path'])!=ref['packet_sha256']:raise ValueError('Actual source packet identity changed')

    # The same installed course geometry and independently placed pattern class
    # used by CurrentPatterns; no fitting rows or sparse derivatives are built.
    scene=provider.scene(world)
    scene=apply_user_visible_markings(scene,pairs=[('F1','R1'),('F2','R2'),('F3','R3'),('F4','R4')],
        authority='D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json')
    base=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,
        facet_authority=parent_scene['shared_board_patterns']['facet_authority'])
    original=SharedBoardPatternSourceFactor(base)
    patterns=CurrentPatterns.__new__(CurrentPatterns)
    patterns.start=int(old['aux__shared_pattern_start']);patterns.state=state
    patterns.geometry=base;patterns.factor=original;patterns.external_factor=original
    patterns._original_course_pattern_geometry=base
    patterns._original_course_pattern_factor=original;patterns._original_course_external_pattern_factor=original
    patterns.course_world_geometry=provider
    placement_canonical=np.asarray(old['aux__current_independent_pattern_placement_parameter_indices'],int)
    placement_evaluation=layout.canonical_to_evaluation[placement_canonical]
    saved_placement=json.loads(str(old['aux__current_independent_pattern_placement_contract_json'].item()))
    if not np.array_equal(placement_evaluation,saved_placement['placement_parameter_indices']):
        raise ValueError('Independent placement canonical/evaluation mapping differs')
    authority=saved_placement['user_authority']
    if sha(authority['path'])!=authority['sha256']:raise ValueError('Independent placement authority changed')
    patterns._independent_pair_offsets=evaluation[placement_evaluation].copy()
    patterns.bind_current_geometry=MethodType(_bind_independent_current_geometry,patterns)
    state.patterns=patterns
    scene['shared_board_patterns']=patterns.descriptor(core)
    scene['shared_board_patterns']['independent_placement_contract']=copy.deepcopy(saved_placement)
    # Initial-state identities omit this optional field; the authenticated
    # same-parent input manifest already records the identical canonical chart.
    circle_canonical=np.asarray(parent.get('independent_circle_material_parameter_indices',
        manifest['independent_circle_material_parameter_indices']),int)
    if circle_canonical.shape!=(6,) or len(np.unique(circle_canonical))!=6:
        raise ValueError('Explicit six original canonical circle slots required')
    circle_evaluation=layout.canonical_to_evaluation[circle_canonical]
    circle_ref=packet_refs['additional_real_native_board_circles']
    if not np.array_equal(circle_evaluation,circle_ref['circle_indices']):raise ValueError('Circle suffix mapping differs')
    circles=NativeBoardCircleFactor(state,circle_ref['packet_path'],int(circle_evaluation[0]))
    scene=circles.export_scene(scene,evaluation[:circles.parameter_count])
    tracks=NativeBoardTrackFactor(state,packet_refs['additional_native_static_board_tracks']['packet_path'],len(evaluation))
    scene=tracks.export_print_scene(scene,evaluation)
    descriptor=scene.get('source_conditioned_board_prints')
    if descriptor:
        descriptor['evaluation_q_sha256']=descriptor['q_sha256'];descriptor['q_sha256']=canonical_hash
    edges=NativeBoardEdgeFactor(state,packet_refs['additional_real_native_board_edges']['packet_path'],len(evaluation))
    physical_checks=[];seen=set()
    for row in edges.rows+edges.held:
        curve=edges.effective_curve(row['curve']);token=json.dumps(curve,sort_keys=True)
        if token in seen:continue
        seen.add(token)
        actual=edges._world(evaluation,row['curve'])
        if curve['kind']=='advertisement_boundary':
            # The installed edge factor authenticates historical Ad5 scope
            # aliases. Compare its actual material, exactly as production's
            # check_current_board_edge_exports, without rewriting source scope.
            ad=next(item for item in scene['advertisements'] if item['user_number']==curve['advertisement_number'])
            field={'top_edge':'top_edge_xyz','bottom_edge':'bottom_edge_xyz',
                   'left_vertical':'left_vertical_xyz','right_vertical':'right_vertical_xyz'}[curve['edge']]
            exported=np.asarray(ad['fitted_boundary_curves_xyz'][field],float)
        else:exported=frozen_scene_curve(scene,curve)
        if actual.shape!=exported.shape:raise ValueError('Actual source and exported finite material shape differ')
        difference=float(np.max(abs(actual-exported)))
        if difference!=0.:raise ValueError('Actual source and scene finite world producer differ: '+token)
        physical_checks.append(dict(curve=curve,world_shape=list(actual.shape),export_world_max_difference=difference))
    scene['source_physical_course_scope']=copy.deepcopy(edges.physical_course_scope_authority)

    # The original wide camera25 parameter chart is retained and evaluated from
    # canonical-mapped private camera columns, independently of programme RF.
    wide=CurrentWideSource.__new__(CurrentWideSource)
    wide.template_camera25=np.asarray(next(r['camera25'] for r in parent_scene['frame_cameras'] if r['source_pts']==56218880),float)
    wide_indices=layout.canonical_to_evaluation[np.asarray(old['aux__current_wide_camera_parameter_indices'],int)]
    if not np.array_equal(wide.camera(reference[wide_indices]),wide.template_camera25):raise ValueError('Original wide camera producer differs')
    scene['frame_cameras']=[dict(source_pts=56218880,camera25=wide.camera(evaluation[wide_indices]).tolist(),role='geometry evidence wide observer')]
    scene['source_event_state_contract']=dict(q_sha256=canonical_hash,q_shape=list(canonical.shape),
        frozen_profile_sha256=profile_token)
    scene['full_programme_source_binding']=dict(path=str(args.input_manifest.resolve()),source_pts=[229114112,231705600],
        native_presentations=len(cameras['pts']),actual_exposures=len(state.bypts))

    calibrations={key:SharedCalibration.from_manifest(value) for key,value in metadata['calibrations'].items()}
    order=json.loads(str(old['aux__correlated_roll_prior_contract_json'].item()))['installation_order']
    pivots=np.asarray(old['aux__installation_xyz_slots'],int)
    mounts=np.asarray(old['aux__current_shared_mount_parameter_indices'],int).reshape(3,3)
    optics=np.asarray(old['aux__current_all_installation_optics_parameter_indices'],int).reshape(3,3)
    for ii,key in enumerate(order):
        cal=calibrations[key];dx=evaluation[pivots[ii]]-reference[pivots[ii]]
        dm=evaluation[mounts[ii]]-reference[mounts[ii]];do=evaluation[optics[ii]]-reference[optics[ii]]
        settings=dict(cal.field_settings)
        settings['current_user_mount_working_constraints']=json.loads(str(old['aux__current_camera_mount_working_constraints_json'].item()))
        settings['current_user_pivot_outside_constraints']=json.loads(str(old['aux__current_camera_pivot_exterior_constraints_json'].item()))
        settings['physical_source_core_q_sha256']=qsha(core)
        calibrations[key]=replace(cal,pivot_xyz=tuple(np.asarray(cal.pivot_xyz)+dx/provider.scene_unit_m),
            lever_xyz=tuple(np.asarray(cal.lever_xyz)+dm/provider.scene_unit_m),
            principal_x_fraction=(cal.principal_x_fraction[0]+do[0]/1920.,),
            principal_y_fraction=(cal.principal_y_fraction[0]+do[1]/1080.,),
            k1=(cal.k1[0]+.001*do[2],),field_settings=settings)
    views={}
    for block in state.blocks:
        for actual,index in block.bundle.index.items():
            value=evaluation[block.start+4*index:block.start+4*index+4]
            views[int(actual)]=CanonicalView(block.bundle.calibration.installation_id,
                ViewState(tuple(value[:3]),float(value[3]-np.log(1920.))))
    args.output.mkdir(parents=True)
    camera_path=args.output/'complete_physical_camera_model_UNACCEPTED.npz'
    PhysicalCameraModel(calibrations).save_npz(camera_path,cameras['pts'],state.original.aliases,views,
        source_sha256=metadata['source_sha256'],time_base=metadata['source_time_base'])
    actual_camera=dict(np.load(camera_path,allow_pickle=False))
    expected=np.asarray([state.bypts[state.original.aliases[int(p)]].bundle.camera(evaluation,state.original.aliases[int(p)])
                         for p in cameras['pts']])
    camera_difference=float(abs(expected-actual_camera['camera25']).max())
    if camera_difference!=0.:raise ValueError('Same-q actual camera producer and exported camera25 differ')
    for key in ('pts','exposure_representative_pts','installation_id'):
        if not np.array_equal(cameras[key],actual_camera[key]):raise ValueError('Original source clock or installation identity changed')
    ads=advertisement_review_paths(scene)
    ad_ids=sorted({int(path['number']) for path in ads})
    paths=fixed_scene_paths(scene,draw_grid=True)
    counts={kind:sum(k==kind for k,_ in paths) for kind in ('grid','marking','contact','board_top')}
    if ad_ids!=list(range(1,27)) or counts['marking']!=65:raise ValueError('Full modeled review inventory differs')
    scene_path=args.output/'scene_UNACCEPTED.json';write(scene_path,scene)
    state_path=args.output/'complete_state_UNACCEPTED.npz';shutil.copy2(args.state,state_path)
    proof=dict(schema='rinkspace.same-canonical-complete-phase-I-review-export.v1',
        canonical_q_sha256=canonical_hash,evaluation_q_sha256=qsha(evaluation),complete_parameter_count=len(canonical),
        original_auxiliary_arrays_preserved=sum(k.startswith('aux__') for k in arrays),
        original_profile_arrays_preserved=sum(k.startswith('profile_') for k in arrays),
        parent_identity=dict(path=str(args.parent_identity.resolve()),sha256=sha(args.parent_identity)),
        actual_complete_parent=phase_parent,frozen_profile_token=profile_token,
        archived_parent_profile_token=parent['frozen_profile_token'],
        initializer=dict(path=str(args.state.resolve()),sha256=sha(args.state)),material_owner=owner,
        parent_camera_reproduction_max_difference=parent_camera_error,
        actual_same_q_camera25_max_difference=camera_difference,
        canonical_evaluation_round_trip_byte_exact=True,
        world_parameter_count=len(world),scene_world_producer_checks=physical_checks,
        source_physical_course_scope=scene['source_physical_course_scope'],
        circle_canonical_indices=circle_canonical.tolist(),circle_evaluation_indices=circle_evaluation.tolist(),
        placement_canonical_indices=placement_canonical.tolist(),placement_evaluation_indices=placement_evaluation.tolist(),
        native_presentations=len(cameras['pts']),actual_exposures=len(views),
        all_source_clocks_aliases_installations_preserved=True,model_path_counts=counts,modeled_advertisement_numbers=ad_ids,
        display_camera_smoothing=False,complete_motion_graph_or_J_loaded=False,
        complete_target_guard_or_cost_acceptance_claimed=False,accuracy_accepted=False,
        elapsed_seconds=time.perf_counter()-started)
    write(args.output/'same_q_export_receipt.json',proof)
    write(args.output/'state_identity.json',dict(q_sha256=canonical_hash,frozen_profile_token=profile_token,
        parameter_count=len(canonical),state_npz_sha256=sha(state_path),scene_sha256=sha(scene_path),
        camera_path=str(camera_path.resolve()),camera_sha256=sha(camera_path),input_manifest_sha256=sha(args.input_manifest),
        source_review_sha256=sha(args.output/'same_q_export_receipt.json'),
        state_kind='user_stopped_Phase_I_initializer_pending_complete_target_and_visual_review',accuracy_accepted=False))
    print(json.dumps(dict(output=str(args.output),q_sha256=canonical_hash,camera25_max_difference=camera_difference,
        model_path_counts=counts,advertisement_numbers=ad_ids,elapsed_seconds=proof['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
