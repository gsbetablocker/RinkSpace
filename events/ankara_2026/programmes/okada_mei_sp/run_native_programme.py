"""Single programme workspace for image registration and native joint fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from rinkspace.session.source_registration import SourceRegistration, RegistrationOptions, track_pair, write_json
from ice_points_s04 import source_domain


HERE = Path(__file__).resolve().parent
DATA = Path('D:/CodexData/RinkSpace_Data')
DEFAULT_RUN = DATA/'runs/native_programme_20261028'


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def material_directory(args):
    return getattr(args, 'material_root', None) or args.run_root/'registration/materials'


def material_navigation_file(args, shot):
    if getattr(args, 'material_root', None) is not None:
        return material_directory(args)/shot/'guided/projection_seed.npz'
    return args.run_root/'registration'/shot/'guided/projection_seed.npz'


def source_frames(shot, *, selected_pts=None):
    # Current source caches have decoded-pixel hashes and masks for every PTS.
    if shot == 's01':
        folder = DATA/'runs/native_ad_replacement_s01_20261026'
        visibility = read(folder/'visibility.json')
        if visibility.get('schema') != 'rinkspace.s01_existing_source_sam2_exclusion.v1':
            raise ValueError('S01 source visibility schema changed')
    elif shot == 's04':
        folder = DATA/'runs/native_ad_points_s04_20261026'
        visibility = read(folder/'visibility.json')
    else:
        date='20261028' if shot in ('s03','s05') else '20261027'
        folder = DATA/'runs'/f'native_points_{shot}_{date}'
        visibility = read(folder/'evidence/visibility.json')
    frames = visibility['frames']
    scope = read(HERE/'rebuild_scope.json')
    shot_scope = next(s for s in scope['shots'] if s['id']==shot)
    expected = shot_scope['processing']
    if shot == 's01':
        physical = shot_scope['physical']
        if [f['pts'] for f in frames] != list(range(physical['first_pts'], physical['stop_pts_exclusive'], 256)):
            raise ValueError('A full physical S01 cache is required')
        frames = [f for f in frames if expected['first_pts'] <= f['pts'] < expected['stop_pts_exclusive']]
        visibility = dict(visibility, source_video_sha256=scope['source']['sha256'])
    if [f['pts'] for f in frames] != list(range(expected['first_pts'], expected['stop_pts_exclusive'], 256)):
        raise ValueError('A full shot is required; source cache is incomplete: '+shot)
    if selected_pts is not None:
        selected_pts = set(selected_pts)
        frames = [frame for frame in frames if frame['pts'] in selected_pts]
    authority_hashes = {}
    for frame in frames:
        if shot == 's01':
            if frame['source_video_sha256'] != scope['source']['sha256']:
                raise ValueError('S01 cached frame belongs to another source')
            dependencies = [dict(path=frame['mask_path'], sha256=digest(frame['mask_path']))]
            for name in ('authority', 'mask_revision'):
                if frame.get(name):
                    path = frame[name]
                    if path not in authority_hashes: authority_hashes[path] = digest(path)
                    if authority_hashes[path] != frame[name+'_sha256']:
                        raise ValueError('S01 source authority changed: '+path)
                    dependencies.append(dict(path=path, sha256=authority_hashes[path]))
            if dependencies[0]['sha256'] != frame['mask_sha256']:
                raise ValueError('Source mask changed: '+frame['mask_path'])
            frame['_surface_roles_available'] = ['unoccluded_source']
            frame['_surface_dependencies'] = dependencies
            frame['surface_input_revision'] = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
            continue
        board = frame.get('board_source')
        if board is None:
            board_folder = 'anchored_frames' if shot == 's04' else 'qualified_frames'
            board = DATA/'runs/okada_rec_1019/recover_v2/dense_board_v1'/board_folder/f'{frame["pts"]}.json'
        frame['_board_path'] = str(board)
        dependencies = [dict(path=frame['mask_path'], sha256=digest(frame['mask_path'])),
                        dict(path=str(board), sha256=digest(board))]
        if frame.get('near_outside_ref'):
            near = frame['near_outside_ref']['file']
            dependencies.append(dict(path=near, sha256=digest(near)))
        if dependencies[0]['sha256'] != frame['mask_sha256']:
            raise ValueError('Source mask changed: '+frame['mask_path'])
        frame['_surface_dependencies'] = dependencies
        frame['surface_input_revision'] = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    return visibility, frames


def surfaces(frame, image):
    mask = cv2.imread(frame['mask_path'], 0)
    if mask is None: raise ValueError('Source exclusion mask is unavailable')
    if frame.get('_surface_roles_available') == ['unoccluded_source']:
        from rinkspace.vision_near_board_qualified import current_graphics
        if mask.shape != image.shape[:2]: raise ValueError('Source exclusion mask dimensions changed')
        graphics, _ = current_graphics(image)
        return dict(unoccluded_source=(mask == 0) & ~graphics, ice=None, board=None,
            surface_roles_available=['unoccluded_source']), frame['_surface_dependencies']
    if frame.get('near_outside_ref'):
        mask = np.maximum(mask, cv2.imread(frame['near_outside_ref']['file'], 0))
    board = read(frame['_board_path'])
    # S04 already has a complete source trace, in column arrays.  These traces
    # delimit the search surface; they never become ice observations.
    for role in board['roles'].values():
        if 'trace_uv' not in role:
            trace = np.column_stack((role['full_native_x'], role['full_native_y']))
            role['trace_uv'] = trace[np.isfinite(trace).all(1)].tolist()
            direct = np.asarray(role['full_direct'], bool) & np.isfinite(trace).all(1)
            runs = np.flatnonzero(np.diff(np.r_[False, direct, False])).reshape(-1, 2)
            role['spans'] = [dict(x_range=[float(trace[a, 0]), float(trace[b-1, 0])])
                             for a, b in runs]
    graphics, bottom, distance, _ = source_domain(image, mask, board)
    top_trace = np.asarray(board['roles']['board.top']['trace_uv'])
    top_trace = top_trace[np.isfinite(top_trace).all(1)]
    top_trace = top_trace[np.argsort(top_trace[:, 0])]
    top = np.interp(np.arange(image.shape[1]), top_trace[:, 0], top_trace[:, 1])
    y = np.arange(image.shape[0])[:, None]
    valid = (~graphics) & (distance > 8)
    # Source visibility and surface navigation are distinct. An advertisement
    # may really touch the board-top strip; that search boundary is not an
    # occluder and must not suppress its measured material edge.
    return dict(unoccluded_source=valid,ice=valid & (y > bottom[None]+8),
                board=valid & (y > top[None]+8) & (y < bottom[None]-8)), frame['_surface_dependencies']


def register(args):
    cv2.setNumThreads(2)
    root = args.run_root/'registration'
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print('MAIN LOG: '+str(root/'registration.jsonl'), flush=True)
    with (root/'registration.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for shot in args.shots:
            def emit(record):
                record = dict(shot=shot, elapsed_total_s=time.monotonic()-started, **record)
                line = json.dumps(record, ensure_ascii=False)
                log.write(line+'\n'); print(line, flush=True)
            visibility, frames = source_frames(shot)
            graph = SourceRegistration(root/shot, frames, visibility['source_video_sha256'])
            quality = graph.prepare(surfaces, emit)
            keys = graph.select_keyframes(quality)
            graph.build(keys, emit)
    write_json(args.run_root/'programme_state.json', dict(stage='source_registration',
        source_graphs={shot:str(root/shot/'graph.json') for shot in args.shots},
        source_registration_is_not_entity_acceptance=True, fit_completed=False))


def references(args):
    """Place the frozen native state beside image-selected reference exposures."""
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import SceneParameterModel, ParameterState
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.review.scene import sample_review_scene
    from rinkspace.review.point_connections import entity_color
    import prepare_native_multicamera as prepared
    baseline = read(args.run_root/'baseline.json')['inputs']
    manifest = read(baseline['manifest']['path'])
    mapping = read(baseline['mapping']['path'])
    values = dict(np.load(baseline['snapshot']['path'], allow_pickle=False))
    points = dict(np.load(baseline['points']['path'], allow_pickle=False))
    authority = read(args.run_root/'authority.json')
    scene = Path(authority['scene_path'])
    model = SceneParameterModel(read(scene), scene, optimize_scene=True)
    prepared.validate_snapshot(values, model.initial_state.layout, mapping, manifest, points)
    state = ParameterState(model.initial_state.layout, values['scene_values'])
    core = NativeCore(manifest['native_library'])
    samples = sample_review_scene(read(scene), core, scene, parameter_state=state,
                                 observed_path_ids={mapping['path_ids'][i] for i in np.unique(points['path_indices'])})
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'])
    destination = args.run_root/'registration/references'
    destination.mkdir(exist_ok=True)
    records = []
    for shot_id in args.shots:
        shot = next(s for s in manifest['shots'] if s['id']==shot_id)
        keys = read(args.run_root/'registration'/shot_id/'keyframes.json')['ice']
        selection = [keys[i] for i in np.unique(np.linspace(0, len(keys)-1, 8).round().astype(int))]
        folder = DATA/'runs'/f'native_points_{shot_id}_20261027'
        for pt in selection:
            raw = cv2.imread(str(folder/'source_frames'/f'{pt}.png'))
            result = raw.copy()
            clock = (pt-shot['reference_pts'])/12800+timeline.offset(shot)
            projection = core.project_motion(values['rig'][shot['camera_index']], values['controls'],
                timeline.first, timeline.dt, 1080, clock, samples['xyz'])
            labels = []
            for cut in samples['cuts']:
                if cut['layer'] not in ('ice', 'board'): continue
                a, b = cut['begin'], cut['end']
                xy = projection['uv'][a:b]
                valid = (projection['status'][a:b]==0) & np.isfinite(xy).all(1)
                inside = valid & (xy[:, 0]>10) & (xy[:, 0]<1910) & (xy[:, 1]>45) & (xy[:, 1]<1055)
                if not inside.any(): continue
                color = entity_color(cut['id'])
                for lo, hi in np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2):
                    if hi-lo>1:
                        poly = np.rint(np.clip(xy[lo:hi], -50000, 50000)).astype(np.int32)
                        cv2.polylines(result, [poly], False, color, 1, cv2.LINE_AA)
                at = xy[np.flatnonzero(inside)[len(np.flatnonzero(inside))//2]]
                name = cut['label']
                labels.append(dict(path_id=cut['id'], label=name, at=at.tolist(), visible_samples=int(inside.sum())))
                label_point = tuple(np.rint(at).astype(int))
                cv2.putText(result, name, label_point, 0, .42, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(result, name, label_point, 0, .42, color, 1, cv2.LINE_AA)
            cv2.rectangle(result, (0,0), (1920,35), (0,0,0), -1)
            title = f'{shot_id.upper()} SOURCE {pt/12800:.3f}s PTS {pt} | FROZEN BASELINE HYPOTHESIS'
            cv2.putText(result, title, (12,25), 0, .65, (255,255,255), 1, cv2.LINE_AA)
            target = destination/f'{shot_id}_{pt}.jpg'
            cv2.imwrite(str(target), result, [cv2.IMWRITE_JPEG_QUALITY, 96])
            records.append(dict(shot=shot_id, pts=pt, image=str(target),
                raw_image=str(folder/'source_frames'/f'{pt}.png'), projected_identities=labels,
                projection_is_hypothesis=True, source_labels_not_accepted=True))
    write_json(destination/'index.json', dict(baseline=baseline['snapshot'], references=records))
    print(json.dumps(dict(event='reference_images', images=len(records), index=str(destination/'index.json'))), flush=True)


def guided_registration(args):
    """Compare image-only and native-initialized searches on the same source."""
    from functools import lru_cache
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.geometry import project
    baseline = read(args.run_root/'baseline.json')['inputs']
    manifest = read(baseline['manifest']['path'])
    values = dict(np.load(baseline['snapshot']['path'], allow_pickle=False))
    core = NativeCore(manifest['native_library'])
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'])
    xx, yy = np.meshgrid(np.linspace(-.52,.52,81), np.linspace(-.27,.27,45))
    world = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    started = time.monotonic()
    logpath = args.run_root/'registration/guided_registration.jsonl'
    print('MAIN LOG: '+str(logpath), flush=True)
    with logpath.open('a', encoding='utf-8', buffering=1) as log:
        def emit(record):
            line = json.dumps(dict(elapsed_s=time.monotonic()-started, atlas_revision=args.atlas_name, **record))
            log.write(line+'\n'); print(line, flush=True)
        for shot_id in args.shots:
            root = args.run_root/'registration'/shot_id
            shot = next(s for s in manifest['shots'] if s['id']==shot_id)
            metadata = read(root/'identity.json')
            options = RegistrationOptions(**metadata['options'])
            frames = metadata['frames']
            output = root/'guided'; output.mkdir(exist_ok=True)
            projection_file = output/'projection_seed.npz'
            receipt = dict(snapshot=baseline['snapshot'], native=baseline['native'],
                           purpose='search_initialization_only_no_model_pixels_as_observations')
            if (output/'identity.json').exists():
                if read(output/'identity.json')!=receipt: raise ValueError('Search seed changed')
            else: write_json(output/'identity.json', receipt)
            if not projection_file.exists():
                uv = []; status = []
                for i, frame in enumerate(frames):
                    clock = (frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                    p = core.project_motion(values['rig'][shot['camera_index']], values['controls'],
                        timeline.first, timeline.dt, 1080, clock, world)
                    uv.append(p['uv']); status.append(p['status'])
                np.savez_compressed(projection_file, uv=np.asarray(uv), status=np.asarray(status))
            projection = dict(np.load(projection_file))
            @lru_cache(maxsize=12)
            def features(index):
                with np.load(root/'features'/f'{frames[index]["pts"]}.npz') as value:
                    return {k:value[k] for k in value.files}
            pure = read(root/'graph.json')['edges']
            results = []
            for ni, old in enumerate(pure):
                if old['surface']!='ice': continue
                a, b = old['reference_index'], old['current_index']
                path = output/f'ice_{frames[a]["pts"]}_{frames[b]["pts"]}.json'
                if path.exists():
                    result = read(path)
                else:
                    pa, pb = projection['uv'][a], projection['uv'][b]
                    valid = (projection['status'][a]==0) & (projection['status'][b]==0)
                    valid &= np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
                    for q in [pa, pb]:
                        valid &= (q[:,0]>-150)&(q[:,0]<2070)&(q[:,1]>100)&(q[:,1]<1230)
                    if valid.sum()<12:
                        result = dict(old, guided_reason='insufficient_seed_overlap', selected_search='image_only')
                    else:
                        h, _ = cv2.findHomography(pa[valid], pb[valid], 0)
                        if h is None: raise ValueError('Degenerate native search map')
                        seed_error = np.linalg.norm(project(h, pa[valid])-pb[valid],axis=1)
                        result, qa, qb, inside = track_pair(features(a), features(b), 'ice', options, h)
                        result.update(reference_index=a,current_index=b,reference_pts=frames[a]['pts'],
                            current_pts=frames[b]['pts'], source_only_status=old['status'],
                            native_map_approximation_p95_px=float(np.percentile(seed_error,95)),
                            model_uv_used_as_measurement=False)
                        np.savez_compressed(path.with_suffix('.npz'),reference_uv=qa,current_uv=qb,inliers=inside)
                    write_json(path,result)
                results.append(result)
                if ni%200==0: emit(dict(event='guided_edges',shot=shot_id,edges=len(results)))
            usable = [e for e in results if e['status']=='navigation']
            covered = set(i for e in usable for i in (e['reference_index'],e['current_index']))
            summary = dict(shot=shot_id,frames=len(frames),navigation_edges=len(usable),
                frames_with_edges=len(covered),uncovered_pts=[f['pts'] for i,f in enumerate(frames) if i not in covered],
                observations_created=0,scene_or_camera_parameters_changed=False)
            write_json(output/'graph.json',dict(summary=summary,edges=results))
            emit(dict(event='guided_ready',**{k:v for k,v in summary.items() if k!='uncovered_pts'}))


def materials(args):
    """Read competing material profiles over complete selected shots."""
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import SceneParameterModel, ParameterState
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.compat.d78_entities import sample_entities
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
    cv2.setNumThreads(2)
    if args.state is None:
        baseline = read(args.run_root/'baseline.json')['inputs']
        authority = read(args.run_root/'authority.json')
        manifest = read(baseline['manifest']['path'])
    else:
        manifest=read(args.run_root/'joint_manifest.json')
        baseline=dict(snapshot=dict(path=str(args.state),sha256=digest(args.state)))
        authority=dict(scene_path=manifest['scene_path'],scene_file_sha256=manifest['scene_file_sha256'],
            native_library=manifest['native_library'],native_library_sha256=manifest['native_library_sha256'])
    values = dict(np.load(baseline['snapshot']['path'], allow_pickle=False))
    scene_path = Path(authority['scene_path']); scene = read(scene_path)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    state = ParameterState(model.initial_state.layout, values['scene_values'])
    core = NativeCore(manifest['native_library'])
    samples = sample_entities(state.apply_scene(scene), core)
    definitions = {p['id']:p for p in scene['paths']}
    paths = []
    for cut in samples['cuts']:
        if cut['layer']!='ice': continue
        definition = definitions[cut['id']]
        name = cut['label'].removeprefix('ice.')
        feature, _, role = name.partition(':')
        xyz = samples['xyz'][cut['begin']:cut['end']]
        contour = definition['measurement']=='contour'
        signal = ('blue' if feature in ('B1','B2') else 'red'
            if feature in ('C','C0','A1','A2','A3','A4') or feature.startswith(('goal_line','goal_trapezoid')) else 'dark')
        row = dict(curve_id=name,feature_id=feature,world_curve_role=role,native_path_id=cut['id'],
            xyz=xyz,operator='material_interface' if contour else 'paint_center',material_signal=signal,
            material_transition=('light_to_dark' if role in ('middle','center') else 'dark_to_light') if contour else None,
            physical_identity_confirmed_by_navigation=False)
        if contour:
            row['center_xyz'] = (xyz[0]+xyz[-1])/2 if feature.startswith('goal_crease') else xyz[:-1].mean(0)
        paths.append(row)
    selected = getattr(args, 'material_features', None)
    if selected:
        missing = set(selected)-{p['feature_id'] for p in paths}
        if missing:raise ValueError('Unknown material features: '+str(sorted(missing)))
        paths = [p for p in paths if p['feature_id'] in selected]
    signal = getattr(args, 'material_signal', None)
    if (selected or signal) and getattr(args,'material_root',None) is None:
        raise ValueError('Changed material search requires a separate material root')
    if signal:
        for path in paths:path['material_signal'] = signal
    source_offsets=np.cumsum([0]+[len(p['xyz']) for p in paths])
    source_xyz=np.vstack([p['xyz'] for p in paths])
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'])
    root = material_directory(args); root.mkdir(parents=True,exist_ok=True)
    receipt = dict(snapshot=baseline['snapshot'],authority=authority,profile_spacing_px=28.,search_radius_px=48.,
        source_candidates_admitted=False,semantics='All candidates remain pending; no source UV comes from projection.')
    if getattr(args, 'material_root', None) is not None:
        receipt.update(selected_features=selected,material_signal_override=signal,
            source_reader_sha256=digest(Path(__import__('rinkspace.vision_programme_ice_review',fromlist=['x']).__file__)))
        if getattr(args, 'separate_material_bands', False):
            receipt['band_pairing'] = 'separate_source_components_with_disjoint_edge_integration'
    inventory = [{k:v for k,v in p.items() if k not in ('xyz','center_xyz')} for p in paths]
    if (root/'inventory.json').exists() and read(root/'inventory.json')!=inventory:
        raise ValueError('Material operators changed; retain the old cache and prepare a separate affected-source revision')
    if (root/'identity.json').exists() and read(root/'identity.json')!=receipt:
        raise ValueError('Material source cache belongs to a different numeric search state')
    write_json(root/'identity.json',receipt)
    write_json(root/'inventory.json',inventory)
    started = time.monotonic()
    print('MAIN LOG: '+str(root/'materials.jsonl'), flush=True)
    with (root/'materials.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        for shot_id in args.shots:
            visibility, frames = source_frames(shot_id)
            shot = next(s for s in manifest['shots'] if s['id']==shot_id)
            reader = IceProgrammeSourceReader(None,source_sha256=visibility['source_video_sha256'],time_base='1/12800',
                sample_spacing_px=28.,search_radius_px=48.,native_paths=paths,
                separate_bands=getattr(args, 'separate_material_bands', False),
                prepared_source_context=dict(source_sha256=visibility['source_video_sha256'],time_base='1/12800',
                    authority=[dict(path=str(scene_path),sha256=digest(scene_path)),
                               dict(path=str(root/'inventory.json'),sha256=digest(root/'inventory.json'))])
                    if getattr(args, 'material_root', None) is not None else None)
            folder = root/shot_id; folder.mkdir(exist_ok=True)
            records = []
            centres = np.array([p.get('center_xyz',[0.,0.,0.]) for p in paths])
            # The same frozen native geometry initializes material-to-material
            # searches. RGB profiles, not these model pixels, supply readouts.
            seed_folder=material_navigation_file(args,shot_id).parent;seed_folder.mkdir(parents=True,exist_ok=True)
            seed_receipt=dict(snapshot=baseline['snapshot'],purpose='search_initialization_only_no_model_pixels_as_observations',
                native_library_sha256=manifest['native_library_sha256'],source_pts=[f['pts'] for f in frames])
            seed_file=seed_folder/'projection_seed.npz'
            if seed_file.exists():
                identity_file=seed_folder/'material_seed_identity.json'
                if identity_file.exists():
                    if read(identity_file)!=seed_receipt:raise ValueError('Material projection search seed changed')
                else:
                    legacy=read(seed_folder/'identity.json')
                    if legacy.get('snapshot')!=baseline['snapshot']:raise ValueError('Existing guided projection uses a different snapshot')
                    with np.load(seed_file) as old_seed:
                        if len(old_seed['uv'])!=len(frames):raise ValueError('Existing guided projection has a different source clock')
                    write_json(identity_file,seed_receipt)
            else:
                xx,yy=np.meshgrid(np.linspace(-.52,.52,81),np.linspace(-.27,.27,45))
                world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];uv=[];status=[]
                for frame in frames:
                    clock=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                    projected=core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,timeline.dt,1080,clock,world)
                    uv.append(projected['uv']);status.append(projected['status'])
                np.savez_compressed(seed_file,uv=np.asarray(uv),status=np.asarray(status))
                write_json(seed_folder/'material_seed_identity.json',seed_receipt)
            for i, frame in enumerate(frames):
                target = folder/f'{frame["pts"]}.npz'
                if target.exists():
                    with np.load(target) as z: info=json.loads(str(z['metadata_json']))
                else:
                    image = cv2.imread(frame['image'])
                    if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source pixels changed')
                    masks,_ = surfaces(frame,image)
                    clock=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                    projection=core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,
                        timeline.dt,1080,clock,source_xyz)
                    cp=core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,timeline.dt,1080,clock,centres)
                    projected={}
                    for ci,path in enumerate(paths):
                        a,b=source_offsets[ci:ci+2]
                        projected[path['native_path_id']]=dict(uv=projection['uv'][a:b],valid=projection['status'][a:b]==0,centre_uv=cp['uv'][ci])
                    result=reader.read(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),frame['pts'],None,~masks['ice'],
                        shot_id=shot_id,native_projections=projected)
                    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                    info=dict(pts=frame['pts'],source_BGR_sha256=frame['source_BGR_sha256'],
                        queries=len(result['query_uv']),candidates=len(result['candidate_uv']),
                        clean_candidates=int(np.count_nonzero((result['candidate_flags']&3)==0)),
                        source_binding=result['source_binding'],all_candidates_pending_identity=True,
                        consumed_case_sha256=result.get('consumed_case_sha256',[]))
                    np.savez_compressed(target,**arrays,metadata_json=np.array(json.dumps(info)))
                records.append(info)
                if i%25==0 or i==len(frames)-1:
                    line=json.dumps(dict(event='material_profiles',shot=shot_id,frames=i+1,total=len(frames),elapsed_s=time.monotonic()-started,
                        candidates=sum(r['candidates'] for r in records),clean_candidates=sum(r['clean_candidates'] for r in records)))
                    log.write(line+'\n');print(line,flush=True)
            write_json(folder/'frames.json',records)


def material_line_roles(args):
    """Qualify whole physical line families from existing actual source UV."""
    from rinkspace.session.material_tracks import qualify_line_candidates
    from rinkspace import vision_source_roles
    root=material_directory(args)
    if not getattr(args,'material_role_name',None):raise ValueError('Explicit new source role directory required')
    manifest=read(args.run_root/'joint_manifest.json');scene=read(manifest['scene_path'])
    definitions={p['id']:p for p in scene['paths']};inventory=read(root/'inventory.json')
    if any(p['operator']!='paint_center' or
           definitions[p['native_path_id']].get('finite_support_role')!='actual_generated_line_between_boundary_intersections'
           for p in inventory):
        raise ValueError('This adapter requires explicit finite physical line primitives')
    receipt=dict(source_identity_sha256=digest(root/'identity.json'),inventory_sha256=digest(root/'inventory.json'),
        source_organizer_sha256=digest(Path(vision_source_roles.__file__)),
        source_qualification_sha256=digest(Path(qualify_line_candidates.__code__.co_filename)),
        visibility_basis='only_fully_excluded_source_normal_profiles',
        missing_readouts_count_as_visible_source_length=True,
        whole_source_line_role=True,native_fit_residual_used=False,source_uv_changed=False,
        paired_source_boundaries_required=bool(getattr(args,'paired_line_boundaries',False)))
    started=time.monotonic()
    for shot in args.shots:
        output=root/shot/args.material_role_name;output.mkdir(parents=True,exist_ok=True)
        if (output/'identity.json').exists() and read(output/'identity.json')!=receipt:
            raise ValueError('Source line qualification dependencies changed')
        write_json(output/'identity.json',receipt)
        frames=read(root/shot/'frames.json');summary=[]
        _,source=source_frames(shot);source_by_pts={f['pts']:f for f in source}
        for fi,frame in enumerate(frames):
            path=root/shot/f'{frame["pts"]}.npz';target=output/f'{frame["pts"]}.json'
            if target.exists():record=read(target)
            else:
                with np.load(path) as z:measurement=dict(z)
                source_frame=source_by_pts[frame['pts']]
                image=cv2.imread(source_frame['image'])
                if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                    raise ValueError('Line qualification source pixels changed')
                masks,dependencies=surfaces(source_frame,image)
                exclusion=~masks['ice']
                ids,reports=qualify_line_candidates(measurement,range(len(inventory)),exclusion,
                    require_paired_boundaries=bool(getattr(args,'paired_line_boundaries',False)))
                record=dict(pts=frame['pts'],source_readout_sha256=digest(path),selected_candidate_ids=ids.tolist(),curves=reports,
                    source_BGR_sha256=frame['source_BGR_sha256'],source_surface_dependencies=dependencies,
                    source_exclusion_mask_sha256=hashlib.sha256(exclusion.tobytes()).hexdigest())
                record=json.loads(json.dumps(record,default=lambda x:x.tolist()))
                write_json(target,record)
            summary.append(dict(pts=frame['pts'],selected_readouts=len(record['selected_candidate_ids'])))
            if fi%100==0 or fi==len(frames)-1:
                print(json.dumps(dict(event='whole_source_line_roles',shot=shot,frames=fi+1,total=len(frames),
                    source_rows=sum(x['selected_readouts'] for x in summary),elapsed_s=time.monotonic()-started)),flush=True)
        write_json(output/'frames.json',summary)


def material_atlas(args):
    """Assemble measured fragments and link them in both temporal directions."""
    from rinkspace.session.material_tracks import (assemble_fragments, source_line_fragments, source_nodes,
        same_frame_conflicts, link_fragments, propagate_identities)
    cv2.setNumThreads(2)
    root = material_directory(args)
    inventory = read(root/'inventory.json')
    started = time.monotonic()
    log_path = root/'atlas.jsonl'
    print('MAIN LOG: '+str(log_path), flush=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        def emit(record):
            line = json.dumps(dict(elapsed_s=time.monotonic()-started, **record))
            log.write(line+'\n'); print(line, flush=True)
        for shot_id in args.shots:
            _, source = source_frames(shot_id)
            folder = root/shot_id/args.atlas_name; folder.mkdir(exist_ok=True)
            role_name=getattr(args,'material_role_name',None)
            qualification=dict(role_directory=role_name,raw_source_identity_sha256=digest(root/'identity.json'))
            if role_name:qualification['fragment_grouping']='unique_whole_source_line_family'
            if (folder/'qualification.json').exists() and read(folder/'qualification.json')!=qualification:
                raise ValueError('Material atlas qualification changed; use a new atlas output')
            if role_name and (folder/'graph.json').exists() and not (folder/'qualification.json').exists():
                raise ValueError('Preserve the previous unqualified source atlas')
            write_json(folder/'qualification.json',qualification)
            frames, edges = [], []
            old = old_rgb = old_mask = None
            with np.load(material_navigation_file(args,shot_id)) as z:
                projected = dict(z)
            for i, frame in enumerate(source):
                pts = frame['pts']; path = folder/f'{pts}.json'
                image = cv2.imread(frame['image'])
                if hashlib.sha256(image.tobytes()).hexdigest() != frame['source_BGR_sha256']:
                    raise ValueError('Atlas source pixels changed')
                masks, _ = surfaces(frame, image)
                mask = ~masks['ice']; rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if path.exists():
                    info = read(path)
                    with np.load(path.with_suffix('.npz')) as z: now = dict(z)
                else:
                    with np.load(root/shot_id/f'{pts}.npz') as z: measurement = dict(z)
                    eligibility=None
                    if role_name:
                        role=read(root/shot_id/role_name/f'{pts}.json')
                        if role['source_readout_sha256']!=digest(root/shot_id/f'{pts}.npz'):
                            raise ValueError('Source readouts changed after line-role qualification')
                        eligibility=np.zeros(len(measurement['candidate_uv']),bool)
                        eligibility[role['selected_candidate_ids']]=True
                    fragments = (source_line_fragments(measurement,role['curves']) if role_name else
                                 assemble_fragments(measurement, inventory,eligibility))
                    now = source_nodes(rgb, measurement, fragments, inventory)
                    info = dict(pts=pts, source_BGR_sha256=frame['source_BGR_sha256'],
                        fragments=fragments, competing_names=same_frame_conflicts(now, fragments),
                        source_readouts_unchanged=True, observations_admitted=0)
                    np.savez_compressed(path.with_suffix('.npz'), **now)
                    write_json(path, info)
                frames.append(info)
                if i:
                    link_path = folder/f'{source[i-1]["pts"]}_{pts}_profile_links.json'
                    if link_path.exists(): linked = read(link_path)
                    else:
                        pa, pb = projected['uv'][i-1], projected['uv'][i]
                        visible = (projected['status'][i-1] == 0)&(projected['status'][i] == 0)
                        visible &= np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
                        for xy in (pa, pb):
                            visible &= (xy[:, 0] > -150)&(xy[:, 0] < 2070)&(xy[:, 1] > 100)&(xy[:, 1] < 1230)
                        if visible.sum() < 12: raise ValueError('Missing source search initialization')
                        h, _ = cv2.findHomography(pa[visible], pb[visible], 0)
                        linked = link_fragments(old_rgb, rgb, old, now, old_mask, mask, h)
                        write_json(link_path, linked)
                    edges.append(linked)
                old, old_rgb, old_mask = now, rgb, mask
                if i%25 == 0 or i == len(source)-1:
                    emit(dict(event='material_atlas', shot=shot_id, frames=i+1, total=len(source),
                        fragments=sum(len(f['fragments']) for f in frames),
                        temporal_links=sum(len(e) for e in edges),
                        unambiguous_links=sum(sum(e['unambiguous'] for e in es) for es in edges)))
            anchor_path = folder/'anchors.json'
            anchors = read(anchor_path) if anchor_path.exists() else []
            labels = propagate_identities(frames, edges, anchors)
            graph = dict(shot=shot_id, frames=frames, edges=edges, anchors=anchors, labels=labels,
                source_candidates_not_automatically_admitted=True,
                temporal_operator='bidirectional_normal_RGB_profile_registration')
            write_json(folder/'graph.json', graph)


def material_gap_links(args):
    """Revisit disconnected source entities on the two sides of missing frames."""
    from functools import lru_cache
    from rinkspace.session.material_tracks import disconnected_source_boundaries,link_fragments
    cv2.setNumThreads(2)
    if args.atlas_graph_name=='graph.json' or Path(args.atlas_graph_name).name!=args.atlas_graph_name:
        raise ValueError('Gap links require a separate graph filename to preserve the original graph')
    root=material_directory(args)
    for shot in args.shots:
        _,frames=source_frames(shot)
        folder=root/shot/args.atlas_name
        original=folder/'graph.json';graph=read(original)
        pairs=disconnected_source_boundaries(graph['frames'],graph['edges'],graph.get('reference_edges',[]))
        with np.load(material_navigation_file(args,shot)) as z:projected=dict(z)
        @lru_cache(maxsize=4)
        def cached(i):
            frame=frames[i];image=cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                raise ValueError('Gap registration source pixels changed')
            masks,_=surfaces(frame,image)
            with np.load(folder/f'{frame["pts"]}.npz') as z:nodes=dict(z)
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),~masks['ice'],nodes
        records=list(graph.get('reference_edges',[]))
        for ordinal,(a,b) in enumerate(pairs):
            path=folder/f'{frames[a]["pts"]}_{frames[b]["pts"]}_gap_links.json'
            if path.exists():record=read(path)
            else:
                pa,pb=projected['uv'][a],projected['uv'][b]
                visible=(projected['status'][a]==0)&(projected['status'][b]==0)
                visible &= np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
                for xy in (pa,pb):
                    visible &= (xy[:,0]>-150)&(xy[:,0]<2070)&(xy[:,1]>100)&(xy[:,1]<1230)
                linked=[]
                if visible.sum()>=12:
                    h,_=cv2.findHomography(pa[visible],pb[visible],0)
                    argb,amask,old=cached(a);brgb,bmask,now=cached(b)
                    linked=link_fragments(argb,brgb,old,now,amask,bmask,h)
                record=dict(reference=a,current=b,reference_pts=frames[a]['pts'],current_pts=frames[b]['pts'],
                    edges=linked,candidate_UV_unchanged=True,missing_exposures_not_interpolated=True,
                    source_identity_not_assumed_from_query_curve=True)
                write_json(path,record)
            records.append(record)
            if ordinal%10==0 or ordinal==len(pairs)-1:
                print(json.dumps(dict(event='source_gap_RGB_links',shot=shot,pairs=ordinal+1,total=len(pairs),
                    unambiguous_links=sum(sum(e['unambiguous'] for e in r['edges']) for r in records))),flush=True)
        graph['reference_edges']=records
        graph['gap_source_graph']=str(original);graph['gap_source_graph_sha256']=digest(original)
        graph['labels']=[]
        write_json(folder/args.atlas_graph_name,graph)


def reference_material_links(args):
    """Direct clear-reference links span broken adjacent-exposure chains."""
    from functools import lru_cache
    from rinkspace.session.material_tracks import link_fragments, propagate_identities
    cv2.setNumThreads(2)
    root = args.run_root/'registration/materials'
    started = time.monotonic()
    log_path = root/'reference_links.jsonl'
    print('MAIN LOG: '+str(log_path), flush=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        def emit(record):
            line = json.dumps(dict(elapsed_s=time.monotonic()-started, **record))
            log.write(line+'\n'); print(line, flush=True)
        for shot_id in args.shots:
            _, frames = source_frames(shot_id)
            folder = root/shot_id/args.atlas_name
            graph = read(folder/'graph.json')
            anchors = read(folder/'anchors.json')
            by_pts = {f['pts']: i for i, f in enumerate(frames)}
            keys = read(args.run_root/'registration'/shot_id/'keyframes.json')['ice']
            keys = sorted(set(by_pts[p] for p in keys) | {by_pts[a['pts']] for a in anchors})
            with np.load(args.run_root/'registration'/shot_id/'guided/projection_seed.npz') as z:
                projected = dict(z)
            @lru_cache(maxsize=8)
            def cached(i):
                frame = frames[i]
                image = cv2.imread(frame['image'])
                if hashlib.sha256(image.tobytes()).hexdigest() != frame['source_BGR_sha256']:
                    raise ValueError('Reference source pixels changed')
                masks, _ = surfaces(frame, image)
                with np.load(folder/f'{frame["pts"]}.npz') as z: nodes = dict(z)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ~masks['ice'], nodes
            pairs = set()
            for i in range(len(frames)):
                past = [k for k in keys if k < i]; future = [k for k in keys if k > i]
                for k in ([past[-1]] if past else [])+([future[0]] if future else []):
                    if abs(k-i)>1: pairs.add((k, i))
            records = []
            for ordinal, (a, b) in enumerate(sorted(pairs)):
                target = folder/f'{frames[a]["pts"]}_{frames[b]["pts"]}_reference_links.json'
                if target.exists(): record = read(target)
                else:
                    pa, pb = projected['uv'][a], projected['uv'][b]
                    visible = (projected['status'][a] == 0)&(projected['status'][b] == 0)
                    visible &= np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
                    for xy in (pa, pb):
                        visible &= (xy[:, 0] > -150)&(xy[:, 0] < 2070)&(xy[:, 1] > 100)&(xy[:, 1] < 1230)
                    if visible.sum() < 12:
                        linked = []
                    else:
                        h, _ = cv2.findHomography(pa[visible], pb[visible], 0)
                        argb, amask, old = cached(a); brgb, bmask, now = cached(b)
                        linked = link_fragments(argb, brgb, old, now, amask, bmask, h)
                    record = dict(reference=a, current=b, reference_pts=frames[a]['pts'],
                        current_pts=frames[b]['pts'], edges=linked, candidate_UV_unchanged=True)
                    write_json(target, record)
                records.append(record)
                if ordinal%100 == 0 or ordinal == len(pairs)-1:
                    emit(dict(event='reference_material_links', shot=shot_id, pairs=ordinal+1, total=len(pairs),
                        links=sum(len(r['edges']) for r in records),
                        unambiguous=sum(sum(e['unambiguous'] for e in r['edges']) for r in records)))
            graph['reference_edges'] = records
            graph['anchors'] = anchors
            graph['labels'] = propagate_identities(graph['frames'], graph['edges'], anchors, records)
            write_json(folder/'graph.json', graph)
            covered = {r['pts'] for r in graph['labels'] if r['status']=='anchored'}
            emit(dict(event='reference_identity_coverage', shot=shot_id, anchored_frames=len(covered),
                anchored_fragments=sum(r['status']=='anchored' for r in graph['labels']),
                conflicts=sum(r['status']!='anchored' for r in graph['labels']), observations_admitted=0))


def verify_material_atlas(args):
    """Check current RGB support for anchored readouts, leaving gaps explicit."""
    from collections import defaultdict
    from rinkspace.vision_ice_model_context_identity import source_paint_pixel_accuracy_check
    root = args.run_root/'registration/materials'
    inventory = read(root/'inventory.json')
    by_name = {r['curve_id']: r for r in inventory}
    summary = {}
    summary_path = root/('verification.json' if args.atlas_name=='atlas' else args.atlas_name+'_verification.json')
    print('MAIN LOG: '+str(summary_path), flush=True)
    for shot_id in args.shots:
        graph_path = root/shot_id/args.atlas_name/'graph.json'
        graph = read(graph_path)
        _, source = source_frames(shot_id)
        labels = defaultdict(list)
        for label in graph['labels']:
            if label['status']=='anchored': labels[int(label['pts'])].append(label)
        accepted = []; held = []; all_rows = 0
        for fi, frame in enumerate(source):
            pts = frame['pts']
            if pts not in labels: continue
            rgb = cv2.cvtColor(cv2.imread(frame['image']), cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb[:, :, ::-1].tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                raise ValueError('Verification source image changed')
            with np.load(root/shot_id/f'{pts}.npz') as z: measurement = dict(z)
            info = json.loads(str(measurement['metadata_json']))
            measurement['source_binding'] = info['source_binding']
            measurement['decoded_native_RGB_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
            groups = defaultdict(set); provenance = defaultdict(list)
            for label in labels[pts]:
                name = label['identities'][0]
                fragment = graph['frames'][fi]['fragments'][label['fragment']]
                groups[name].update(fragment['candidate_ids'])
                for ci in fragment['candidate_ids']: provenance[(name, ci)].append(label['fragment'])
            for name, ids in groups.items():
                ids = np.array(sorted(ids), int); all_rows += len(ids)
                result = source_paint_pixel_accuracy_check(rgb, measurement, ids, by_name[name])
                passed = result['accepted_candidate_indices']
                for pending in result['pending']:
                    held.append(dict(pts=pts, identity=name, **pending))
                for ci in passed:
                    q = measurement['candidate_side_quantiles_uv'][ci]
                    if not np.isfinite(q).all():
                        held.append(dict(pts=pts, identity=name, original_candidate_index=int(ci), status='source_gradient_quantiles_missing'))
                        continue
                    uv = q[:, 1].mean(0)
                    accepted.append(dict(pts=pts, identity=name, path_id=by_name[name]['native_path_id'],
                        candidate=int(ci), uv=uv.tolist(), saved_float32_uv=measurement['candidate_uv'][ci].tolist(),
                        actual_source_quantiles=q.tolist(), fragments=provenance[(name, ci)],
                        source_BGR_sha256=frame['source_BGR_sha256'], evidence_image=frame['image'],
                        source_material_identity_from_bidirectional_graph=True,
                        source_UV_is_current_RGB_readout=True, native_fit_not_run=True))
        stats = {}
        for name in sorted({r['identity'] for r in accepted}):
            rows = [r for r in accepted if r['identity']==name]
            ps = sorted({r['pts'] for r in rows})
            stats[name] = dict(rows=len(rows), frames=len(ps), first_pts=ps[0], last_pts=ps[-1])
        report = dict(shot=shot_id, full_shot_frames=len(source), anchored_readouts=all_rows,
            source_verified_readouts=len(accepted), held_readouts=len(held), by_entity=stats,
            graph=dict(path=str(graph_path), sha256=digest(graph_path)),
            accepted=accepted, held=held, user_visual_acceptance=False,
            source_registration_stage_only=True, fit_observations_modified=False)
        write_json(root/shot_id/args.atlas_name/'source_verified.json', report)
        summary[shot_id] = {k:v for k,v in report.items() if k not in ('accepted', 'held')}
        print(json.dumps(dict(event='source_material_verified', **summary[shot_id])), flush=True)
    write_json(summary_path, summary)


def prepare_joint_atlas(args):
    """Stage three: source-backed group rereads in the existing four shots."""
    import copy
    import shutil
    from collections import defaultdict
    from scipy.spatial import cKDTree
    import prepare_native_multicamera as prepared
    from apply_native_multicamera_review import apply
    baseline = read(args.run_root/'baseline.json')['inputs']
    authority = read(args.run_root/'authority.json')
    root = args.run_root
    if (root/'joint_manifest.json').exists():
        raise FileExistsError('Prepared four-shot revision already exists; reuse its state')
    for key in ('points', 'mapping', 'snapshot', 'manifest', 'native'):
        if digest(baseline[key]['path']) != baseline[key]['sha256']:
            raise ValueError('Frozen baseline input changed: '+key)
    manifest = read(baseline['manifest']['path'])
    mapping = read(baseline['mapping']['path'])
    if [s['id'] for s in manifest['shots']] != ['s01', 's02', 's04', 's06']:
        raise ValueError('Stage three uses the verified four-shot baseline only')
    with np.load(baseline['points']['path']) as z: original = dict(z)
    groups = defaultdict(list); receipts = []
    for shot_id in ('s02', 's06'):
        path = root/'registration/materials'/shot_id/'atlas/source_verified.json'
        report = read(path)
        if digest(report['graph']['path']) != report['graph']['sha256']:
            raise ValueError('Verified material graph changed')
        receipts.append(dict(path=str(path), sha256=digest(path)))
        for row in report['accepted']:
            groups[(int(row['pts']), row['path_id'])].append(dict(row, shot=shot_id))
    decisions = []; additions = []
    # Replacement is limited to the actual source footprint. Other visible
    # portions of the same curve and all other entities retain their source.
    for (pts, path), rows in groups.items():
        path_index = mapping['path_ids'].index(path)
        selected = np.flatnonzero((original['pts']==pts)&(original['path_indices']==path_index))
        tree = cKDTree(np.array([r['uv'] for r in rows]))
        distance, _ = tree.query(original['uv'][selected])
        for i in selected[distance <= 40.]:
            decisions.append(dict(observation_id=str(original['observation_ids'][i]), pts=pts,
                expected_path_id=path, uv=original['uv'][i].tolist(),
                decision='exclude_preserving_source', evidence_image=rows[0]['evidence_image'],
                reason='Superseded within the same source-material footprint by current RGB atlas readouts; original is retained, not declared false.'))
        for row in rows:
            additions.append(dict(row, observation_id=f'atlas_v1:{row["shot"]}:{pts}:{path_index}:{row["candidate"]}',
                reason='Current original RGB gradient q50 readout, source material checked, identity linked to a reviewed reference through bidirectional source-profile matches.',
                kind=1))
    report = dict(input_sha256=baseline['points']['sha256'], source_receipts=receipts,
        decisions=decisions, added_observations=additions, source_pixels_synthesized=False,
        model_projection_only_used_for_registration_initialization=True,
        complete_four_shot_baseline_preserved=True, S03_S05_connected=False)
    write_json(root/'atlas_source_revision.json', report)
    shutil.copy2(baseline['points']['path'], root/'point_observations.npz')
    write_json(root/'mapping.json', mapping)
    manifest = copy.deepcopy(manifest)
    manifest['scene_path'] = authority['scene_path']
    manifest['scene_file_sha256'] = digest(authority['scene_path'])
    write_json(root/'joint_manifest.json', manifest)
    prepared.RUN = root
    apply(root/'atlas_source_revision.json', Path(baseline['snapshot']['path']),
          root/'atlas_source_seed.npz', root/'atlas_source_receipt')
    write_json(root/'programme_state.json', dict(stage=3, stage_name='four_shot_joint_comparison',
        stage_1='baseline_verified', stage_2='complete_sequences_registered_with_explicit_unresolved_materials',
        prepared_source_revision=str(root/'atlas_source_revision.json'),
        source_replaced_rows=len(decisions), source_new_readouts=len(additions),
        native_fit_completed=False, S03_S05_connected=False,
        stage_4='missing_entities_and_blur_constraints_pending', stage_5='six_shot_connection_pending'))


def comparison_exposure_coverage(manifest, mapping, observations, errors, path_layers):
    """Keep every source exposure, including missing constraints, in comparison."""
    def summary(values):
        finite=values[np.isfinite(values)]
        return dict(rows=len(values),invalid=len(values)-len(finite),
            rms=float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
            p95=float(np.percentile(finite,95)) if len(finite) else None,
            maximum=float(finite.max()) if len(finite) else None)
    pts=observations['pts'];order=np.argsort(pts,kind='stable');sorted_pts=pts[order]
    split=np.flatnonzero(np.diff(sorted_pts))+1
    groups={int(pts[ix[0]]):ix for ix in np.split(order,split) if len(ix)}
    layers=np.array([path_layers[name] for name in mapping['path_ids']])[observations['path_indices']]
    result=[];shot_summary={}
    for shot in manifest['shots']:
        all_rows=np.arange(shot['row_begin'],shot['row_stop']);by_layer={}
        for layer in sorted(set(layers[all_rows])):
            selected=all_rows[layers[all_rows]==layer]
            by_layer[str(layer)]=dict(before=summary(errors[0][selected]),after=summary(errors[1][selected]))
        shot_summary[shot['id']]=dict(before=summary(errors[0][all_rows]),after=summary(errors[1][all_rows]),layers=by_layer)
        for pt in range(shot['first_pts'],shot['last_pts']+1,256):
            rows=groups.get(pt,np.array([],dtype=int));ice=rows[layers[rows]=='ice']
            result.append(dict(shot=shot['id'],pts=pt,source_seconds=pt/12800,
                before=summary(errors[0][rows]),after=summary(errors[1][rows]),
                ice_before=summary(errors[0][ice]),ice_after=summary(errors[1][ice]),
                actual_source_ice_rows=len(ice),actual_source_ice_paths=sorted({mapping['path_ids'][i] for i in observations['path_indices'][ice]})))
    return dict(source_exposures=len(result),shots=shot_summary,frames=result,
        full_video_visual_acceptance=False,missing_constraints_are_included_in_exposure_denominator=True)


def compare_joint_atlas(args):
    """Evaluate a frozen native checkpoint on exactly the same source rows."""
    import shutil
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import SceneParameterModel, ParameterState
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    from rinkspace.review.scene import sample_native_review_scene
    from rinkspace.review.point_connections import entity_color
    import prepare_native_multicamera as prepared
    root = args.run_root
    stage=getattr(args,'stage',3)
    folder = args.output or root/f'stage{stage}_comparison'; folder.mkdir(parents=True,exist_ok=True)
    checkpoint = args.state or root/f'stage{stage}_joint/accepted_pixels_latest.npz'
    frozen = folder/'numeric_state.npz'
    shutil.copy2(checkpoint, frozen)
    manifest = read(root/'joint_manifest.json'); mapping = read(root/'mapping.json')
    z = dict(np.load(root/'point_observations.npz', allow_pickle=False))
    scene_path = Path(manifest['scene_path']); scene = read(scene_path)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    core = NativeCore(manifest['native_library'])
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    times = timeline.row_times(z['pts'])
    camera_ids = np.zeros(len(z['pts']), np.int32)
    for shot in manifest['shots']: camera_ids[shot['row_begin']:shot['row_stop']] = shot['camera_index']
    paths = [mapping['path_ids'][i] for i in z['path_indices']]
    before=args.baseline_state or root/('atlas_source_seed.npz' if stage==3 else f'stage{stage}_source_seed.npz')
    states = [dict(np.load(before, allow_pickle=False)), dict(np.load(frozen, allow_pickle=False))]
    errors = []; scenes = []; station_profiles = []
    camera = {**read(manifest['cameras'][0]['camera_definition']),
        'first_control_time':timeline.first, 'control_dt':timeline.dt}
    print('MAIN LOG: '+str(folder/'comparison.json'), flush=True)
    batch_size=getattr(args,'comparison_batch_size',None) or len(paths)
    if batch_size<=0:raise ValueError('Comparison allocation batch size must be positive')
    for state_index,state in enumerate(states):
        prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, z)
        parameter = ParameterState(model.initial_state.layout, state['scene_values'])
        native = NativeSceneGPU(scene_path, core, parameter_state=parameter)
        state_errors=np.empty(len(paths))
        for begin in range(0,len(paths),batch_size):
            end=min(len(paths),begin+batch_size);selection=slice(begin,end);batch_paths=paths[selection]
            with native.resident(camera=camera, rigs=state['rig'], camera_ids=camera_ids[selection],
                    controls=state['controls'], path_ids=batch_paths, stations=state['stations'][selection],
                    pts=z['pts'][selection], observed=z['uv'][selection], motion_times=times[selection],
                    scene_directions=np.zeros((len(state['scene_values']), 1))) as resident:
                # Each ordinary curve coordinate is independent at frozen geometry.
                # Batching changes allocation only; every original row is compared.
                curve=np.flatnonzero(z['kind'][selection]==1);local=np.full(len(batch_paths),-1,np.int32)
                local[curve]=np.arange(len(curve));periodic=np.array([native.paths[batch_paths[i]]['kind']==7 for i in curve])
                resident.configure_pixel_linearization(directions=np.zeros((len(state['controls']),3,1)),
                    local_ids=local,local_scales=np.full(len(curve),.001))
                import ctypes
                class MemoryStatus(ctypes.Structure):
                    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
                memory=MemoryStatus();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
                resident.configure_solver(lower=np.where(periodic,-np.inf,0.),upper=np.where(periodic,np.inf,1.),
                    device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=memory.avail_phys,
                    damping_metric='jacobi_image',initial_lambda=0.,profile_trial_stations=True)
                profile=resident.refine_curve_stations(mode='global')
                station_profiles.append(dict(profile,state_index=state_index,row_begin=begin,row_stop=end))
                state_errors[selection]=resident.evaluate_errors()
            if end==len(paths) or (begin//batch_size)%8==0:
                progress=dict(event='same_input_comparison',state_index=state_index,compared_rows=end,total_rows=len(paths))
                with (folder/'comparison_progress.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
        errors.append(state_errors)
        scenes.append(None if getattr(args,'metrics_only',False) else sample_native_review_scene(scene, core, scene_path,
            parameter_state=parameter,observed_path_ids=set(paths),native=native))
    def stats(e):
        finite = e[np.isfinite(e)]
        return dict(rows=len(e), invalid=int(len(e)-len(finite)),
            rms=float(np.sqrt(np.mean(finite*finite))), median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95)), maximum=float(finite.max())) if len(finite) else dict(rows=len(e), invalid=len(e))
    new = np.char.startswith(z['observation_ids'], 'atlas_v1:' if stage==3 else f'stage{stage}:')
    if stage==5:
        new=np.zeros(len(z['pts']),bool)
        for shot in manifest['shots']:
            if shot['id'] in ('s03','s05'):new[shot['row_begin']:shot['row_stop']]=True
    report = dict(checkpoint=dict(path=str(frozen), sha256=digest(frozen)),
        input_sha256=digest(root/'point_observations.npz'), same_rows_before_after=True,
        optimization_finished=(checkpoint.parent/'result.json').exists(),
        source_identity_not_inferred_from_fit=True, user_visual_acceptance=False,
        station_comparison='independent_global_native_curve_search_with_geometry_frozen',station_profiles=station_profiles,comparison_batch_size=batch_size,
        all_rows=[stats(e) for e in errors], retained_original_rows=[stats(e[~new]) for e in errors],
        current_RGB_atlas_rows=[stats(e[new]) for e in errors], entities={})
    if stage==5:
        report['introduced_shot_rows']=report.pop('current_RGB_atlas_rows')
        report['retained_four_shot_rows']=report.pop('retained_original_rows')
        report['baseline_state']=str(before)
    for shot in manifest['shots']:
        ix = np.arange(shot['row_begin'], shot['row_stop'])
        report['entities'][shot['id']] = {}
        for pi in np.unique(z['path_indices'][ix]):
            selected = ix[z['path_indices'][ix]==pi]
            path = mapping['path_ids'][pi]
            if native.paths[path]['layer']!='ice' and stage!=5: continue
            report['entities'][shot['id']][path] = dict(label=native.paths[path].get('label', path),
                before=stats(errors[0][selected]), after=stats(errors[1][selected]),
                frames=len(np.unique(z['pts'][selected])))
    np.savez_compressed(folder/'same_input_errors.npz', before=errors[0], after=errors[1], atlas_rows=new)
    coverage=comparison_exposure_coverage(manifest,mapping,z,errors,{name:native.paths[name]['layer'] for name in mapping['path_ids']})
    write_json(folder/'exposure_coverage.json',coverage)
    selected = dict(s02=[575744,616448,664832,703744],s06=[1334016,1375744,1414144,1425408])
    if stage==4:
        selected=dict(s01=[331776,573440],s02=[599040,601088,699136,720640],
            s04=[778496,872960,880128],s06=[1314304,1411584,1421056,1425408])
    if stage==5:
        selected={}
        for shot in manifest['shots']:
            ix=np.arange(shot['row_begin'],shot['row_stop'])
            pts=np.arange(shot['first_pts'],shot['stop_pts_exclusive'],256)
            worst=int(z['pts'][ix[np.argmax(errors[1][ix])]])
            selected[shot['id']]=sorted({int(pts[0]),int(pts[len(pts)//2]),int(pts[-1]),worst})
    if getattr(args,'metrics_only',False):selected={}
    pictures = []
    for shot_id, pts_list in selected.items():
        shot = next(s for s in manifest['shots'] if s['id']==shot_id)
        for pts in pts_list:
            path = DATA/'runs'/f'native_points_{shot_id}_20261027'/'source_frames'/f'{pts}.png'
            if shot_id=='s01':path=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/frames'/f'{pts//256-1000:06d}.png'
            elif shot_id=='s04':path=DATA/'runs/native_ad_points_s04_20261026/source_frames'/f'{pts}.png'
            elif shot_id in ('s03','s05'):path=DATA/'runs'/f'native_points_{shot_id}_20261028/source_frames'/f'{pts}.png'
            if shot_id=='s01' and not path.exists():
                path=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013'/f'frame_{pts//256-1000:06d}_50fps/source.png'
            if path.exists():raw=cv2.imread(str(path))
            else:
                from rinkspace.review.render import source_frames as original_frames
                scope=read(HERE/'rebuild_scope.json');source={**scope['source'],**scope['source_clock']}
                _,raw=next(original_frames(source['registered_path'],source,[int(pts)]))
                path=folder/f'{shot_id}_{pts}_source.png';cv2.imwrite(str(path),raw)
            panels = []
            for label, state, samples in zip(('BEFORE', 'CURRENT CHECKPOINT'), states, scenes):
                output = raw.copy()
                clock = (pts-shot['reference_pts'])/12800+timeline.offset(shot)
                projection = core.project_motion(state['rig'][shot['camera_index']], state['controls'],
                    timeline.first, timeline.dt, 1080, clock, samples['xyz'])
                for cut in samples['cuts']:
                    if stage!=5 and cut['layer'] not in ('ice','board'): continue
                    a,b = cut['begin'],cut['end']; uv = projection['uv'][a:b]
                    good = (projection['status'][a:b]==0)&np.isfinite(uv).all(1)
                    color = entity_color(cut['id'])
                    for lo,hi in np.flatnonzero(np.diff(np.r_[False,good,False])).reshape(-1,2):
                        if hi-lo>1:
                            poly = np.rint(np.clip(uv[lo:hi],-50000,50000)).astype(np.int32)
                            cv2.polylines(output,[poly],False,color,1,cv2.LINE_AA)
                if stage==5:
                    for uv in z['uv'][z['pts']==pts]:
                        cv2.circle(output,tuple(np.rint(uv).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
                cv2.rectangle(output,(0,0),(1920,35),(0,0,0),-1)
                cv2.putText(output,f'{shot_id.upper()} {pts/12800:.3f}s PTS {pts} | {label}',(12,25),0,.7,(255,255,255),1,cv2.LINE_AA)
                panels.append(output)
            target = folder/f'{shot_id}_{pts}.jpg'
            cv2.imwrite(str(target),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,96])
            pictures.append(dict(shot=shot_id,pts=pts,source_image=str(path),comparison=str(target)))
    report['images'] = pictures
    write_json(folder/'comparison.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('entities','images')}),flush=True)


def reread_source_anchors(args):
    """Stage four: source-reference queries driven by the updated native state."""
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import SceneParameterModel, ParameterState
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.compat.d78_entities import sample_entities
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
    from rinkspace.vision_ice_model_context_identity import source_paint_pixel_accuracy_check
    from rinkspace.session.material_tracks import assemble_fragments, source_nodes, link_fragments, _warp
    cv2.setNumThreads(2)
    root = args.run_root
    manifest = read(root/'joint_manifest.json'); scene_path = Path(manifest['scene_path']); scene = read(scene_path)
    state_path = args.state or root/'stage3_joint/accepted_pixels_latest.npz'
    values = dict(np.load(state_path, allow_pickle=False))
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    state = ParameterState(model.initial_state.layout, values['scene_values'])
    core = NativeCore(manifest['native_library'])
    samples = sample_entities(state.apply_scene(scene), core)
    cuts = {cut['id']:cut for cut in samples['cuts']}
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'])
    material_root = root/'registration/materials'
    inventory = read(material_root/'inventory.json')
    definitions = {p['curve_id']:p for p in inventory}
    xx, yy = np.meshgrid(np.linspace(-.52,.52,81),np.linspace(-.27,.27,45))
    world = np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    destination = root/'registration'/args.read_name; destination.mkdir(exist_ok=True)
    started = time.monotonic(); logpath = destination/'read.jsonl'
    print('MAIN LOG: '+str(logpath),flush=True)
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        def emit(record):
            line=json.dumps(dict(elapsed_s=time.monotonic()-started,**record));log.write(line+'\n');print(line,flush=True)
        for shot_id in args.shots:
            visibility, frames = source_frames(shot_id)
            shot = next(s for s in manifest['shots'] if s['id']==shot_id)
            anchor_folder = material_root/shot_id/args.atlas_name
            anchors = read(anchor_folder/'anchors.json')
            by_pts = {f['pts']:i for i,f in enumerate(frames)}
            projections = []
            for frame in frames:
                clock=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                projections.append(core.project_motion(values['rig'][shot['camera_index']],values['controls'],
                    timeline.first,timeline.dt,1080,clock,world))
            for ai, anchor in enumerate(anchors):
                definition = definitions[anchor['identity']]
                if definition['operator']!='paint_center' or definition['world_curve_role'] not in ('edge0','edge1','edge2','segment0'):
                    continue  # Curved references need their observed curved guide, never a line surrogate.
                index = by_pts[anchor['pts']]; reference = frames[index]
                cut=cuts[definition['native_path_id']]
                path=dict(definition,xyz=samples['xyz'][cut['begin']:cut['end']])
                reader=IceProgrammeSourceReader(None,source_sha256=visibility['source_video_sha256'],time_base='1/12800',
                    native_paths=[path],sample_spacing_px=12.,search_radius_px=48.)
                ref_image=cv2.imread(reference['image']); ref_rgb=cv2.cvtColor(ref_image,cv2.COLOR_BGR2RGB)
                ref_masks,_=surfaces(reference,ref_image)
                with np.load(anchor_folder/f'{anchor["pts"]}.npz') as z: source=dict(z)
                selected=np.isin(source['ids'],anchor['candidate_ids'])
                old={k:v[selected] for k,v in source.items()}
                old['fragment']=np.zeros(len(old['uv']),np.int32);old['family']=np.zeros(len(old['uv']),np.int16)
                line=cv2.fitLine(old['uv'].astype(np.float32),cv2.DIST_L2,0,.001,.001).ravel()
                tangent=line[:2].astype(float);origin=line[2:].astype(float)
                station=(old['uv']-origin)@tangent
                guide=origin+np.array([station.min(),station.max()])[:,None]*tangent
                folder=destination/shot_id/(anchor['identity'].replace(':','_'));folder.mkdir(parents=True,exist_ok=True)
                receipt=dict(anchor=anchor,source_graph=dict(path=str(anchor_folder/'graph.json'),sha256=digest(anchor_folder/'graph.json')),
                    numeric_state=dict(path=str(state_path),sha256=digest(state_path)),
                    search_guide_is_not_observed_UV=True,measurement='current_native_RGB_gradient_q50',
                    source_reference_first_pts=anchor['pts'],sample_spacing_px=12.,search_radius_px=48.,
                    profile_matcher='current_image_normals_and_matched_readouts_v3')
                prior_folder=root/'registration/anchor_reads2'/shot_id/(anchor['identity'].replace(':','_'))
                prior_identity=read(prior_folder/'identity.json') if (prior_folder/'identity.json').exists() else {}
                reuse=all(prior_identity.get(k)==receipt[k] for k in ('anchor','numeric_state','sample_spacing_px','search_radius_px'))
                if (folder/'identity.json').exists() and read(folder/'identity.json')!=receipt:
                    raise ValueError('Source-reference read cache belongs to a different state')
                write_json(folder/'identity.json',receipt)
                accepted=[]; frame_records=[]
                for i, frame in enumerate(frames):
                    target=folder/f'{frame["pts"]}.json'
                    if target.exists(): record=read(target)
                    else:
                        pa,pb=projections[index]['uv'],projections[i]['uv']
                        good=(projections[index]['status']==0)&(projections[i]['status']==0)
                        good &= np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
                        for uv in (pa,pb):good &= (uv[:,0]>-150)&(uv[:,0]<2070)&(uv[:,1]>100)&(uv[:,1]<1230)
                        record=dict(pts=frame['pts'],accepted=[],candidate_fragments=0,links=[])
                        if good.sum()>=12:
                            h,_=cv2.findHomography(pa[good],pb[good],0)
                            moved=_warp(h,guide)
                            if np.isfinite(moved).all() and not ((moved[:,0]<-60).all() or (moved[:,0]>1980).all() or
                                    (moved[:,1]<0).all() or (moved[:,1]>1140).all()):
                                image=cv2.imread(frame['image']);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                                if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Current source differs')
                                masks,_=surfaces(frame,image)
                                prior_measurement=prior_folder/f'{frame["pts"]}.npz'
                                if reuse and prior_measurement.exists():
                                    measurement=dict(np.load(prior_measurement,allow_pickle=False))
                                    with np.load(material_root/shot_id/f'{frame["pts"]}.npz') as original:
                                        binding=json.loads(str(original['metadata_json']))['source_binding']
                                    measurement['source_binding']=binding
                                    measurement['decoded_native_RGB_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
                                    record['source_measurement_reused_from']=str(prior_measurement)
                                else:
                                    measurement=reader.read(rgb,frame['pts'],None,~masks['ice'],shot_id=shot_id,
                                        native_projections={path['native_path_id']:dict(uv=moved,valid=np.ones(2,bool))})
                                fragments=assemble_fragments(measurement,[path]);now=source_nodes(rgb,measurement,fragments,[path])
                                links=link_fragments(ref_rgb,rgb,old,now,~ref_masks['ice'],~masks['ice'],h)
                                ids=sorted({int(now['ids'][pair[1]]) for link in links if link['unambiguous'] for pair in link['pairs']})
                                checked=(source_paint_pixel_accuracy_check(rgb,measurement,ids,path) if ids
                                    else dict(accepted_candidate_indices=[],pending=[]))
                                for ci in checked['accepted_candidate_indices']:
                                    q=measurement['candidate_side_quantiles_uv'][ci]
                                    if not np.isfinite(q).all():continue
                                    record['accepted'].append(dict(pts=frame['pts'],identity=anchor['identity'],path_id=path['native_path_id'],
                                        candidate=int(ci),uv=q[:,1].mean(0).tolist(),actual_source_quantiles=q.tolist(),
                                        evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                                        source_reference=anchor,source_identity_via_direct_bidirectional_material_match=True))
                                record.update(candidate_fragments=len(fragments),links=links,source_check_pending=checked['pending'],
                                    query_guide=moved.tolist(),actual_candidates=len(measurement['candidate_uv']))
                                np.savez_compressed(target.with_suffix('.npz'),**{k:v for k,v in measurement.items() if isinstance(v,np.ndarray)})
                        write_json(target,record)
                    accepted.extend(record['accepted']);frame_records.append({k:v for k,v in record.items() if k not in ('accepted','links','source_check_pending')})
                    if i%50==0 or i==len(frames)-1:emit(dict(event='source_reference_reread',shot=shot_id,identity=anchor['identity'],
                        frames=i+1,total=len(frames),source_verified_rows=len(accepted),source_verified_frames=len({r['pts'] for r in accepted})))
                write_json(folder/'source_verified.json',dict(shot=shot_id,identity=anchor['identity'],accepted=accepted,
                    frames=frame_records,receipt=receipt,fit_observations_modified=False,user_visual_acceptance=False))


def source_rings(args):
    """Whole-shot nested source contours; instance identities remain separate."""
    from rinkspace.session import material_rings
    cv2.setNumThreads(2)
    root=args.run_root/'registration/materials'
    inventory=read(root/'inventory.json')
    destination=args.run_root/'registration/ring_materials_v1'
    destination.mkdir(exist_ok=True)
    authority=dict(algorithm_sha256=digest(material_rings.__file__),
        inventory_sha256=digest(root/'inventory.json'),
        method='source_conic_consensus_and_material_nesting',
        output_points='unchanged_RGB_gradient_quantile_candidates',
        ellipse_vertices_never_exported_as_observations=True)
    if (destination/'identity.json').exists() and read(destination/'identity.json')!=authority:
        raise ValueError('Ring source cache belongs to another algorithm or inventory')
    write_json(destination/'identity.json',authority)
    logpath=destination/'read.jsonl';print('MAIN LOG: '+str(logpath),flush=True)
    started=time.monotonic()
    houses=sorted({p['feature_id'] for p in inventory if p['feature_id'][0] in 'FR' and p['operator']=='material_interface'})
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        for shot in args.shots:
            _,frames=source_frames(shot)
            folder=destination/shot;folder.mkdir(exist_ok=True)
            coverage={}
            for fi,frame in enumerate(frames):
                target=folder/f'{frame["pts"]}.json'
                if target.exists():result=read(target)
                else:
                    path=root/shot/f'{frame["pts"]}.npz'
                    measurement=dict(np.load(path,allow_pickle=False))
                    image=cv2.imread(frame['image'])
                    if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                        raise ValueError('Source ring image changed')
                    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    fragments=read(root/shot/'atlas'/f'{frame["pts"]}.json')['fragments']
                    result=dict(pts=frame['pts'],source_BGR_sha256=frame['source_BGR_sha256'],
                        measurement=dict(path=str(path),sha256=digest(path)),evidence_image=frame['image'],
                        houses={},fit_observations_modified=False,physical_instances_confirmed=False)
                    for house in houses:
                        hypotheses={p['world_curve_role']:material_rings.source_ring_hypotheses(rgb,measurement,ci,fragments)
                            for ci,p in enumerate(inventory) if p['feature_id']==house and p['operator']=='material_interface'}
                        if not any(hypotheses.values()):continue
                        result['houses'][house]=dict(hypotheses=hypotheses,stack=material_rings.nested_source_rings(hypotheses))
                    write_json(target,result)
                for house,value in result['houses'].items():
                    rows=sum(len(ids) for ids in value['stack']['selected'].values())
                    if rows:
                        entry=coverage.setdefault(house,dict(frames=0,readouts=0,first_pts=frame['pts'],last_pts=frame['pts']))
                        entry['frames']+=1;entry['readouts']+=rows;entry['last_pts']=frame['pts']
                if fi%50==0 or fi==len(frames)-1:
                    line=json.dumps(dict(event='source_ring_stack',shot=shot,frames=fi+1,total=len(frames),
                        elapsed_s=time.monotonic()-started,coverage=coverage))
                    log.write(line+'\n');log.flush();print(line,flush=True)
            write_json(folder/'coverage.json',dict(shot=shot,total_frames=len(frames),by_house=coverage,
                identity_from_query_name_is_not_confirmation=True,fit_observations_modified=False))


def source_ring_colors(args):
    """Separate actual coloured house layers from ellipse-shaped gray clutter."""
    from rinkspace.session.material_rings import source_ring_color_evidence
    root=args.run_root/'registration/ring_materials_v1'
    destination=root/getattr(args,'color_name','color_checked');destination.mkdir(exist_ok=True)
    logpath=destination/'read.jsonl';print('MAIN LOG: '+str(logpath),flush=True)
    started=time.monotonic()
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        for shot in args.shots:
            _,frames=source_frames(shot);folder=destination/shot;folder.mkdir(exist_ok=True)
            coverage={}
            for fi,frame in enumerate(frames):
                source=root/shot/f'{frame["pts"]}.json';result=read(source)
                joint_colors=getattr(args,'joint_color_stack',False)
                stacks=(result['houses'] if joint_colors else {h:v['stack']['selected'] for h,v in result['houses'].items() if v['stack']['selected']})
                output=dict(pts=frame['pts'],source_hypotheses=dict(path=str(source),sha256=digest(source)),houses={},
                    color_before_geometric_consensus=joint_colors)
                if stacks:
                    measurement=dict(np.load(result['measurement']['path'],allow_pickle=False))
                    rgb=cv2.cvtColor(cv2.imread(frame['image']),cv2.COLOR_BGR2RGB)
                    for house,selected in stacks.items():
                        if joint_colors:
                            from rinkspace.session.material_ring_qualification import color_constrained_ring_stack
                            checked=color_constrained_ring_stack(rgb,measurement,selected['hypotheses'],nesting=selected['stack'])
                        else:checked=source_ring_color_evidence(rgb,measurement,selected)
                        output['houses'][house]=checked
                        if checked['selected']:
                            entry=coverage.setdefault(house,dict(frames=0,readouts=0,first_pts=frame['pts'],last_pts=frame['pts']))
                            entry['frames']+=1;entry['readouts']+=sum(map(len,checked['selected'].values()));entry['last_pts']=frame['pts']
                write_json(folder/f'{frame["pts"]}.json',output)
                if fi%100==0 or fi==len(frames)-1:
                    line=json.dumps(dict(event='source_ring_colors',shot=shot,frames=fi+1,total=len(frames),
                        elapsed_s=time.monotonic()-started,coverage=coverage));log.write(line+'\n');print(line,flush=True)
            write_json(folder/'coverage.json',dict(shot=shot,total_frames=len(frames),by_house=coverage,
                physical_instance_identity_pending=True,fit_observations_modified=False))


def source_ring_links(args):
    """Transfer a coloured source reference to weak frames by actual RGB."""
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.session.material_tracks import source_nodes, link_fragments
    from collections import defaultdict
    cv2.setNumThreads(2)
    root=args.run_root;material_root=root/'registration/materials';ring_root=root/'registration/ring_materials_v1'
    destination=ring_root/'linked';destination.mkdir(exist_ok=True)
    inventory=read(material_root/'inventory.json');definitions={p['curve_id']:i for i,p in enumerate(inventory)}
    manifest=read(root/'joint_manifest.json');state_path=args.state or root/'stage3_joint/accepted_pixels_latest.npz'
    values=dict(np.load(state_path,allow_pickle=False));core=NativeCore(manifest['native_library'])
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'])
    xx,yy=np.meshgrid(np.linspace(-.52,.52,81),np.linspace(-.27,.27,45));world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    anchors=read(ring_root/'anchors.json')
    logpath=destination/'read.jsonl';print('MAIN LOG: '+str(logpath),flush=True);started=time.monotonic()
    def nodes(frame, selected):
        z=dict(np.load(material_root/frame['shot']/f'{frame["pts"]}.npz',allow_pickle=False))
        rgb=cv2.cvtColor(cv2.imread(frame['image']),cv2.COLOR_BGR2RGB)
        fragments=[];names=[]
        for house,layers in selected.items():
            for role,ids in layers.items():
                if not ids:continue
                names.append(dict(house=house,role=role))
                fragments.append(dict(curve=definitions[house+':'+role],candidate_ids=ids))
        points=source_nodes(rgb,z,fragments,inventory)
        points['family']=np.array([('inner','middle','outer').index(names[i]['role']) for i in points['fragment']],np.int16)
        return rgb,z,points,fragments,names
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        for shot_id in args.shots:
            _,frames=source_frames(shot_id);shot=next(s for s in manifest['shots'] if s['id']==shot_id)
            by_pts={f['pts']:f for f in frames}
            projections={};references=[]
            for frame in frames:
                frame['shot']=shot_id
                clock=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                projections[frame['pts']]=core.project_motion(values['rig'][shot['camera_index']],values['controls'],
                    timeline.first,timeline.dt,1080,clock,world)
            for anchor in anchors:
                if anchor['shot']!=shot_id:continue
                frame=by_pts[anchor['pts']]
                color=read(ring_root/'color_checked'/shot_id/f'{frame["pts"]}.json')['houses'][anchor['house']]
                if not color['selected']:raise ValueError('Ring anchor lacks source colour evidence')
                rgb,z,points,fragments,names=nodes(frame,{anchor['house']:color['selected']})
                mask=~surfaces(frame,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))[0]['ice']
                references.append((anchor,rgb,points,names,mask))
            folder=destination/shot_id;folder.mkdir(exist_ok=True);accepted=[]
            for fi,frame in enumerate(frames):
                result=read(ring_root/shot_id/f'{frame["pts"]}.json')
                selected={h:v['stack']['selected'] for h,v in result['houses'].items() if v['stack']['selected']}
                frame_links=[];identities=defaultdict(set)
                if selected:
                    rgb,z,now,fragments,names=nodes(frame,selected)
                    mask=~surfaces(frame,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))[0]['ice']
                    for anchor,old_rgb,old,old_names,old_mask in references:
                        pa,pb=projections[anchor['pts']],projections[frame['pts']]
                        good=(pa['status']==0)&(pb['status']==0)&np.isfinite(pa['uv']).all(1)&np.isfinite(pb['uv']).all(1)
                        for uv in (pa['uv'],pb['uv']):good&=(uv[:,0]>-150)&(uv[:,0]<2070)&(uv[:,1]>100)&(uv[:,1]<1230)
                        if good.sum()<12:continue
                        h,_=cv2.findHomography(pa['uv'][good],pb['uv'][good],0)
                        links=link_fragments(old_rgb,rgb,old,now,old_mask,mask,h)
                        # Identity propagates as the entire ordered material
                        # stack; a lone similar gray interface is insufficient.
                        grouped=defaultdict(list)
                        for link in links:
                            if link['unambiguous'] and old_names[link['a']]['role']==names[link['b']]['role']:
                                grouped[names[link['b']]['house']].append(link)
                        for house,edges in grouped.items():
                            if {names[e['b']]['role'] for e in edges}!={'inner','middle','outer'}:continue
                            for edge in edges:
                                name=anchor['house']+':'+names[edge['b']]['role']
                                identities[edge['b']].add(name)
                                frame_links.append(dict(reference=anchor,identity=name,current_fragment=edge['b'],**edge))
                    for fragment, labels in identities.items():
                        if len(labels)!=1:continue
                        name=next(iter(labels));definition=inventory[definitions[name]]
                        proofs=[i for i,e in enumerate(frame_links) if e['current_fragment']==fragment]
                        matched={int(now['ids'][pair[1]]) for i in proofs for pair in frame_links[i]['pairs']}
                        for ci in sorted(matched):
                            quantiles=z['candidate_side_quantiles_uv'][ci]
                            if not np.isfinite(quantiles).all():continue
                            accepted.append(dict(pts=frame['pts'],identity=name,path_id=definition['native_path_id'],
                                candidate=int(ci),uv=quantiles[:,1].mean(0).tolist(),actual_source_quantiles=quantiles.tolist(),
                                evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                                source_identity_via_ordered_ring_stack_and_bidirectional_RGB=True,
                                source_link_indices=proofs,source_link_file=str(folder/f'{frame["pts"]}.json'),
                                reference_self_read=any(frame_links[i]['reference']['pts']==frame['pts'] for i in proofs)))
                write_json(folder/f'{frame["pts"]}.json',dict(pts=frame['pts'],links=frame_links,
                    labels={str(k):sorted(v) for k,v in identities.items()},fit_observations_modified=False))
                if fi%50==0 or fi==len(frames)-1:
                    line=json.dumps(dict(event='source_ring_links',shot=shot_id,frames=fi+1,total=len(frames),
                        elapsed_s=time.monotonic()-started,readouts=len(accepted),linked_frames=len({r['pts'] for r in accepted})))
                    log.write(line+'\n');print(line,flush=True)
            write_json(folder/'source_verified.json',dict(shot=shot_id,accepted=accepted,
                source_references=[a for a in anchors if a['shot']==shot_id],
                numeric_state=dict(path=str(state_path),sha256=digest(state_path)),
                source_UV_is_current_RGB_readout=True,user_visual_acceptance=False,fit_observations_modified=False))


def cross_view_ring_links(args):
    """Associate complete source material families across physical cameras.

    A qualified three-layer reference establishes the material family. A weak
    current exposure can contribute two actually matched interfaces; the absent
    third interface is not manufactured or required in every blurred frame.
    """
    from collections import defaultdict
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.session.material_tracks import source_nodes, link_fragments, _warp
    cv2.setNumThreads(2)
    root=args.run_root;reference_root=args.reference_run
    if reference_root is None or args.state is None:raise ValueError('Cross-view reading requires explicit references and frozen geometry')
    material_root=root/'registration/materials';ring_root=root/'registration/ring_materials_v1'
    inventory=read(material_root/'inventory.json');definition={x['curve_id']:i for i,x in enumerate(inventory)}
    manifest=read(root/'joint_manifest.json');values=dict(np.load(args.state,allow_pickle=False))
    core=NativeCore(manifest['native_library']);timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    xx,yy=np.meshgrid(np.linspace(-.52,.52,81),np.linspace(-.27,.27,45));world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    anchors=read(reference_root/'registration/ring_materials_v1/anchors.json')
    destination=ring_root/args.link_name;destination.mkdir(exist_ok=True)
    receipt=dict(numeric_state=dict(path=str(args.state),sha256=digest(args.state)),reference_run=str(reference_root),
        matcher_sha256=digest(Path(__import__('rinkspace.session.material_tracks',fromlist=['__file__']).__file__)),
        reference_anchors=anchors,instance_seed='source_colour_qualified_three_interface_stack',
        current_frame_evidence='at_least_two_distinct_interfaces_with_unique_bidirectional_RGB_matches',
        current_missing_layers_never_synthesized=True,physical_identity_is_assistant_inference=True)
    if (destination/'identity.json').exists() and read(destination/'identity.json')!=receipt:raise ValueError('Cross-view reading has different input')
    write_json(destination/'identity.json',receipt)
    def project_grid(shot_id,pts):
        shot=next(s for s in manifest['shots'] if s['id']==shot_id)
        clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
        return core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,timeline.dt,1080,clock,world)
    def nodes(frame,folder,selected):
        measurement=dict(np.load(folder/f'{frame["pts"]}.npz',allow_pickle=False))
        image=cv2.imread(frame['image'])
        if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Material image pixels changed')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);fragments=[];names=[]
        for house,layers in selected.items():
            for role,ids in layers.items():
                if not ids:continue
                names.append(dict(house=house,role=role))
                fragments.append(dict(curve=definition[house+':'+role],candidate_ids=ids))
        result=source_nodes(rgb,measurement,fragments,inventory)
        result['family']=np.array([('inner','middle','outer').index(names[i]['role']) for i in result['fragment']],np.int16)
        return rgb,measurement,result,~surfaces(frame,image)[0]['ice'],names
    reference_frames={};references=[]
    for anchor in anchors:
        if anchor['shot'] not in reference_frames:
            _,frames=source_frames(anchor['shot']);reference_frames[anchor['shot']]={f['pts']:f for f in frames}
        frame=reference_frames[anchor['shot']][anchor['pts']]
        color_path=reference_root/'registration/ring_materials_v1/color_checked'/anchor['shot']/f'{anchor["pts"]}.json'
        colored=read(color_path)['houses'][anchor['house']]
        if set(colored['selected'])!={'inner','middle','outer'} or min(map(len,colored['selected'].values()))<6:
            raise ValueError('External reference does not establish the complete material family')
        rgb,z,points,mask,names=nodes(frame,reference_root/'registration/materials'/anchor['shot'],{anchor['house']:colored['selected']})
        references.append((dict(anchor,color_evidence=dict(path=str(color_path),sha256=digest(color_path))),rgb,points,mask,names,project_grid(anchor['shot'],anchor['pts'])))
    logpath=destination/'read.jsonl';print('MAIN LOG: '+str(logpath),flush=True);started=time.monotonic()
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        for shot_id in args.shots:
            _,frames=source_frames(shot_id);folder=destination/shot_id;folder.mkdir(exist_ok=True);accepted=[]
            for fi,frame in enumerate(frames):
                path=folder/f'{frame["pts"]}.json'
                if path.exists():record=read(path)
                else:
                    source_path=ring_root/shot_id/f'{frame["pts"]}.json';source=read(source_path);selected={}
                    for house,item in source['houses'].items():
                        alternatives=item['stack'].get('alternatives',[])
                        if not alternatives:continue
                        selected[house]={role:sorted({ci for a in alternatives for ci in item['hypotheses'][role][a['indices'][role]]['candidate_ids']}) for role in ('inner','middle','outer')}
                    record=dict(pts=frame['pts'],source_hypotheses=dict(path=str(source_path),sha256=digest(source_path)),links=[],accepted=[],pending_instances={})
                    if selected:
                        rgb,measurement,now,mask,names=nodes(frame,material_root/shot_id,selected);pb=project_grid(shot_id,frame['pts']);candidates=defaultdict(list)
                        for anchor,old_rgb,old,old_mask,old_names,pa in references:
                            valid=(pa['status']==0)&(pb['status']==0)&np.isfinite(pa['uv']).all(1)&np.isfinite(pb['uv']).all(1)
                            for uv in (pa['uv'],pb['uv']):valid&=(uv[:,0]>-150)&(uv[:,0]<2070)&(uv[:,1]>100)&(uv[:,1]<1230)
                            if valid.sum()<12:continue
                            h,_=cv2.findHomography(pa['uv'][valid],pb['uv'][valid],0)
                            moved=_warp(h,old['uv']);visible=np.isfinite(moved).all(1)&(moved[:,0]>-80)&(moved[:,0]<2000)&(moved[:,1]>-80)&(moved[:,1]<1160)
                            if not visible.any():continue
                            bounds=np.array([moved[visible].min(0)-80,moved[visible].max(0)+80])
                            relevant=(now['uv']>=bounds[0]).all(1)&(now['uv']<=bounds[1]).all(1)
                            if not relevant.any():continue
                            local={k:v[relevant] for k,v in now.items()};edges=link_fragments(old_rgb,rgb,old,local,old_mask,mask,h)
                            grouped=defaultdict(list)
                            for edge in edges:
                                if edge['unambiguous'] and old_names[edge['a']]['role']==names[edge['b']]['role']:
                                    grouped[names[edge['b']]['house']].append(edge)
                            for query_house,matched in grouped.items():
                                roles={names[e['b']]['role'] for e in matched}
                                if len(roles)<2:continue
                                rows=[]
                                for edge in matched:
                                    role=names[edge['b']]['role'];record['links'].append(dict(reference=anchor,query_house=query_house,role=role,**edge))
                                    for _,node in edge['pairs']:
                                        ci=int(local['ids'][node]);quantiles=measurement['candidate_side_quantiles_uv'][ci]
                                        if not np.isfinite(quantiles).all():continue
                                        rows.append(dict(pts=frame['pts'],identity=anchor['house']+':'+role,
                                            path_id=inventory[definition[anchor['house']+':'+role]]['native_path_id'],candidate=ci,
                                            uv=quantiles[:,1].mean(0).tolist(),actual_source_quantiles=quantiles.tolist(),
                                            evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],source_reference=anchor,
                                            source_identity_via_shared_plane_and_bidirectional_RGB=True,matched_material_roles=sorted(roles),
                                            missing_material_layers_not_filled=True))
                                candidates[query_house].append((anchor['house'],rows))
                        used=set()
                        for house,claims in candidates.items():
                            identities={claim[0] for claim in claims}
                            if len(identities)!=1:
                                record['pending_instances'][house]=sorted(identities);continue
                            for _,rows in claims:
                                for row in rows:
                                    key=(row['identity'],row['candidate'])
                                    if key not in used:record['accepted'].append(row);used.add(key)
                    write_json(path,record)
                accepted.extend(record['accepted'])
                if fi%50==0 or fi==len(frames)-1:
                    progress=dict(event='cross_view_material_match',shot=shot_id,frames=fi+1,total=len(frames),
                        readouts=len(accepted),supported_frames=len({x['pts'] for x in accepted}),elapsed_s=time.monotonic()-started)
                    line=json.dumps(progress);log.write(line+'\n');print(line,flush=True)
            write_json(folder/'source_verified.json',dict(shot=shot_id,accepted=accepted,receipt=receipt,
                user_visual_acceptance=False,source_UV_is_current_RGB_readout=True,fit_observations_modified=False))


def prepare_source_revision(args):
    """Assimilate the stage-four material evidence once, preserving its parent."""
    import os
    from collections import defaultdict
    from scipy.spatial import cKDTree
    import prepare_native_multicamera as prepared
    from apply_native_multicamera_review import apply
    root=args.run_root
    if (root/'stage4_source_seed.npz').exists():
        raise FileExistsError('Stage-four revision is already prepared; continue its native state')
    manifest=read(root/'joint_manifest.json');mapping=read(root/'mapping.json')
    if [s['id'] for s in manifest['shots']]!=['s01','s02','s04','s06']:
        raise ValueError('Stage four must precede the six-shot connection')
    sources=[('rings_'+shot,root/'registration/ring_materials_v1/linked'/shot/'source_verified.json') for shot in ('s02','s06')]
    sources += [(shot+'_'+name,root/'registration/anchor_reads3'/shot/name/'source_verified.json')
                for shot,name in [('s02','lane3_edge1'),('s06','lane2_edge0'),('s06','lane3_edge2')]]
    groups=defaultdict(list);receipts=[];pixels=set()
    for tag,path in sources:
        report=read(path);receipts.append(dict(path=str(path),sha256=digest(path),rows=len(report['accepted'])))
        for row in report['accepted']:
            key=(int(row['pts']),row['path_id'],*map(float,row['uv']))
            if key in pixels:continue
            pixels.add(key);groups[key[:2]].append(dict(row,source_packet=tag))
    original=dict(np.load(root/'point_observations.npz',allow_pickle=False));decisions=[];additions=[]
    for (pts,path),rows in groups.items():
        pi=mapping['path_ids'].index(path)
        selected=np.flatnonzero((original['pts']==pts)&(original['path_indices']==pi))
        distance,_=cKDTree(np.array([r['uv'] for r in rows])).query(original['uv'][selected])
        for i in selected[distance<=40.]:
            decisions.append(dict(observation_id=str(original['observation_ids'][i]),pts=pts,expected_path_id=path,
                uv=original['uv'][i].tolist(),decision='exclude_preserving_source',evidence_image=rows[0]['evidence_image'],
                reason='Replaced inside the same measured source-material footprint by bidirectionally matched current RGB readouts; not selected by native fit residual.'))
        for row in rows:
            additions.append(dict(row,observation_id=f'stage4:{row["source_packet"]}:{pts}:{pi}:{row["candidate"]}',kind=1,
                reason='Actual current-source gradient q50; source identity supported by an explicit user line reference or ordered coloured ring stack and bidirectional RGB correspondence. Physical instance inference and pixel uncertainty remain in the source receipt.'))
    if not additions:raise ValueError('No new source-supported material observations')
    before=digest(root/'point_observations.npz')
    if before!=manifest['point_input_sha256']:raise ValueError('Prepared input changed')
    archive=root/'stage4_input_archive';archive.mkdir()
    os.link(root/'point_observations.npz',archive/'point_observations.npz')
    write_json(archive/'joint_manifest.json',manifest);write_json(archive/'mapping.json',mapping)
    report=dict(input_sha256=before,source_receipts=receipts,decisions=decisions,added_observations=additions,
        source_pixels_synthesized=False,native_residual_used_for_selection=False,
        identity_uncertainty_is_not_pixel_accuracy=True,unresolved_F3_F4_ring_patterns_not_admitted=True,
        S03_S05_connected=False)
    write_json(root/'stage4_source_revision.json',report)
    prepared.RUN=root
    apply(root/'stage4_source_revision.json',root/'stage3_joint/accepted_pixels_latest.npz',
        root/'stage4_source_seed.npz',root/'stage4_source_receipt')
    state=read(root/'programme_state.json')
    state.update(stage=4,stage_name='source_gap_and_identity_joint_fit',stage_4='source_revision_prepared',
        source_revision=str(root/'stage4_source_revision.json'),source_replaced_rows=len(decisions),
        source_new_readouts=len(additions),current_numeric_state=str(root/'stage4_source_seed.npz'),
        native_fit_completed=False,numerical_convergence=False,S03_S05_connected=False)
    write_json(root/'programme_state.json',state)


def manual_references(args):
    from manual_material_references import prepare
    prepare(args)


def manual_graph(args):
    from manual_material_references import anchor_graph
    anchor_graph(args)


def manual_direct_links(args):
    from manual_material_references import direct_links
    direct_links(args)


def manual_export(args):
    from manual_material_references import export_source_readouts
    export_source_readouts(args)


def manual_revision(args):
    from manual_material_references import prepare_revision
    prepare_revision(args)


def source_composite(args):
    from source_material_composite import build
    build(args)


def temporal_ring_registration(args):
    from source_ring_registration import propagate
    propagate(args)


def main():
    from paired_ring_registration import run as ring_bands, link as ring_band_links, assimilate as ring_band_revision
    from source_ring_registration import complete_stacks
    from register_print_patterns import run as print_patterns, reread as print_readouts, prepare_revision as print_revision
    from manual_material_references import readout_graph,refine_readout_geometry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'references', 'guided', 'materials', 'line-roles', 'atlas', 'atlas-gap-links', 'atlas-links', 'atlas-verify', 'prepare-joint', 'compare-joint', 'anchor-read', 'rings', 'ring-colors', 'ring-links', 'prepare-revision','cross-ring-links','manual-references','manual-graph','readout-graph','readout-geometry','manual-links','manual-export','manual-revision','source-composite','ring-graph','ring-bands','ring-consensus','ring-band-links','ring-band-revision','apply-source-revision','print-patterns','print-readouts','print-revision'])
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--shots', nargs='+', choices=['s02','s03','s04','s05','s06'], default=['s02','s06'])
    parser.add_argument('--state', type=Path)
    parser.add_argument('--baseline-state', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metrics-only', action='store_true')
    parser.add_argument('--comparison-batch-size',type=int)
    parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--print-atlas-id',type=int)
    parser.add_argument('--reference-pts',type=int)
    parser.add_argument('--ring-house',default='F2')
    parser.add_argument('--ring-signal',choices=['red','dark'],default='red')
    parser.add_argument('--ring-roles',nargs='+',choices=['middle','outer'],default=['middle','outer'])
    parser.add_argument('--refine-source-origin',action='store_true')
    parser.add_argument('--source-packets',type=Path,nargs='+')
    parser.add_argument('--replace-source-prefixes',nargs='+')
    parser.add_argument('--replace-source-paths',nargs='+')
    parser.add_argument('--source-observation-prefix')
    parser.add_argument('--append-source-only',action='store_true')
    parser.add_argument('--color-name',default='color_checked')
    parser.add_argument('--link-name',default='linked_cross_view')
    parser.add_argument('--joint-color-stack',action='store_true')
    parser.add_argument('--atlas-name', default='atlas')
    parser.add_argument('--atlas-graph-name', default='graph.json')
    parser.add_argument('--material-root',type=Path)
    parser.add_argument('--material-features',nargs='+')
    parser.add_argument('--material-signal',choices=['red','blue','dark'])
    parser.add_argument('--material-role-name')
    parser.add_argument('--separate-material-bands',action='store_true')
    parser.add_argument('--paired-line-boundaries',action='store_true')
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--source-identity-graph',type=Path)
    parser.add_argument('--read-name', default='anchor_reads3')
    parser.add_argument('--stage',type=int,default=3)
    args = parser.parse_args()
    dict(register=register, references=references, guided=guided_registration, materials=materials,
         atlas=material_atlas, **{'atlas-links':reference_material_links, 'atlas-verify':verify_material_atlas,
             'line-roles':material_line_roles,'atlas-gap-links':material_gap_links,
             'prepare-joint':prepare_joint_atlas, 'compare-joint':compare_joint_atlas, 'anchor-read':reread_source_anchors,
             'rings':source_rings, 'ring-colors':source_ring_colors, 'ring-links':source_ring_links,
               'prepare-revision':prepare_source_revision,'cross-ring-links':cross_view_ring_links,'manual-references':manual_references,'manual-graph':manual_graph,'readout-graph':readout_graph,'readout-geometry':refine_readout_geometry,'manual-links':manual_direct_links,'manual-export':manual_export,'manual-revision':manual_revision,'source-composite':source_composite,'ring-graph':temporal_ring_registration,'ring-bands':ring_bands,'ring-consensus':complete_stacks,'ring-band-links':ring_band_links,'ring-band-revision':ring_band_revision,'apply-source-revision':ring_band_revision,'print-patterns':print_patterns,'print-readouts':print_readouts,'print-revision':print_revision})[args.action](args)


if __name__ == '__main__':
    main()
