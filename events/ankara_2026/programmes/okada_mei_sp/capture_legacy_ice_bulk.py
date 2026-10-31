"""Stream original RGB ice rereads using legacy UV only as search navigation.

Run through scripts/run_native_entry.py. No solver, packet admission or prepared
input mutation occurs. All RGB alternatives and partial arcs remain available.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, gzip, hashlib, json, time
import cv2
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.path_policy import guard_project_write
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import assemble_fragments, qualify_visible_line_fragments
from rinkspace.session.material_rings import ellipse_matrix, ellipse_vertices, source_ring_hypotheses, nested_source_rings
from rinkspace.session.material_ring_qualification import color_constrained_ring_stack
from capture_independent_materials import source_exclusion, spaced_indices
from capture_independent_ice import load_navigation_graph, navigation_between, capture_frames, timed_capture_frames, line_search_guide, ice_source_mask


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def read(path):
    path=Path(path)
    data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    return json.loads(data)


def write(path, value):
    path=guard_project_write(path)
    compact=path.name.split('.')[0].isdigit()
    data=(json.dumps(plain(value),ensure_ascii=False,indent=None if compact else 2,
        separators=(',',':') if compact else None)+'\n').encode('utf8')
    path.write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if path.suffix=='.gz' else data)


def native_inventory(cuts, xyz, labels):
    """All native ice paths, with the event's current material/midline rule."""
    paths = []
    for cut in cuts:
        if cut['layer'] != 'ice': continue
        name = labels.get(cut['id'], cut['label']).removeprefix('ice.'); feature, _, role = name.partition(':')
        if not role and feature.startswith('goal_crease'): role='boundary'
        if not role: raise ValueError('No authoritative source-material role: ' + str(cut))
        points = xyz[cut['begin']:cut['end']]
        annulus = len(feature) == 2 and feature[0] in 'FR' and feature[1] in '1234' and cut['closed'] and cut['kind']=='ice_circle'
        crease = feature.startswith('goal_crease')
        signal = ('dark' if annulus else 'blue' if feature in ('B1', 'B2') or crease else
                  'red' if feature in ('C', 'C0', 'A1', 'A2', 'A3', 'A4') or feature.startswith(('goal_line', 'goal_trapezoid')) else 'dark')
        path = dict(curve_id=f'{feature}:{role}', feature_id=feature, world_curve_role=role,
                    native_path_id=cut['id'], xyz=points, closed=bool(cut['closed']),
                    query_sampling='continuous_arc_length', operator='material_interface' if annulus or crease else 'paint_center',
                    material_signal=signal, material_transition=('dark_to_light' if role in ('outer', 'inner') else 'light_to_dark') if annulus else 'dark_to_light' if crease else None,
                    physical_identity_confirmed_by_navigation=False)
        if cut['closed'] and cut['kind'] == 'ice_circle': path['center_xyz'] = points[:-1].mean(0)
        paths.append(path)
    return paths


def source_search_selection(paths, feature_ids=None, paint_signal_overrides=None):
    """Select searches and explicit paint channels without changing path roles."""
    features = {p['feature_id'] for p in paths}
    if feature_ids is not None:
        if not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError('Select distinct, nonempty feature IDs')
        unknown = set(feature_ids)-features
        if unknown:
            raise ValueError('Unknown source feature IDs: '+', '.join(sorted(unknown)))
    selected = features if feature_ids is None else set(feature_ids)
    overrides = {} if paint_signal_overrides is None else paint_signal_overrides
    if not isinstance(overrides, dict):
        raise ValueError('Paint signal overrides must be a JSON feature-to-signal object')
    applied = {}
    for feature, signal in overrides.items():
        if not isinstance(feature, str) or feature not in features:
            raise ValueError('Unknown paint signal override feature: '+str(feature))
        if signal not in ('dark', 'red', 'blue'):
            raise ValueError('Paint signal override must be dark, red or blue')
        targets = [p for p in paths if p['feature_id'] == feature]
        if any(p['operator'] != 'paint_center' for p in targets):
            raise ValueError('Signal overrides only apply to paint_center paths: '+feature)
        applied[feature] = dict(original_signals=sorted({p['material_signal'] for p in targets}),
            selected_signal=signal, path_ids=[p['native_path_id'] for p in targets],
            measurement_operator_and_role_changed=False)
        for path in targets:
            path['material_signal'] = signal
    searched = {p['native_path_id'] for p in paths if p['feature_id'] in selected}
    return searched, applied


class LegacyNavigation:
    """Retain both old indexes; none of their coordinates is a new readout."""
    def __init__(self, folder, paths, source_sha):
        self.folder = folder
        self.references = defaultdict(dict)
        self.paths = {p['curve_id']: p for p in paths}
        correspondence=folder/'ice_path_correspondence.json'
        document=read(correspondence)
        self.aliases={row['old_label']:row['current_label'].removeprefix('ice.') for row in document['correspondence_rows'] if row.get('path_id') and row.get('current_label')}
        self.correspondence_sha256=digest(correspondence)
        self.inputs = {str(correspondence):self.correspondence_sha256}
        self.unmapped = Counter()
        records = defaultdict(lambda: defaultdict(list))
        full = folder / 'old_full9_s07_s09_source_index.npz'
        sparse = folder / 'old_sparse_source_index.npz'
        self.inputs[str(full)] = digest(full)
        self.inputs[str(sparse)] = digest(sparse)
        self.inputs[str(sparse.with_suffix('.json'))] = digest(sparse.with_suffix('.json'))
        z = dict(np.load(full, allow_pickle=False))
        if str(z['source_video_sha256']) != source_sha: raise ValueError('Legacy index source differs')
        labels = z['material_labels'][z['material_index']]
        for i in np.flatnonzero(np.char.startswith(labels, 'ice.')):
            name = self.aliases.get(str(labels[i]),str(labels[i]).removeprefix('ice.'))
            if name not in self.paths: self.unmapped[str(labels[i])] += 1; continue
            sid = str(z['shot_ids'][z['shot_index'][i]])
            records[(sid, name)][int(z['native_pts'][i])].append((z['uv'][i], 'full9_stored_UV_navigation', int(i)))
        z = dict(np.load(sparse, allow_pickle=False))
        meta = read(sparse.with_suffix('.json'))
        if meta['source_clock']['video_sha256'] != source_sha: raise ValueError('Sparse source differs')
        for i in np.flatnonzero(z['category'] == 0):
            old_label=str(z['label_values'][z['label_index'][i]])
            name = self.aliases.get(old_label,old_label.removeprefix('ice.'))
            if name not in self.paths: self.unmapped[str(z['label_values'][z['label_index'][i]])] += 1; continue
            # Actual readouts/quantiles are preferred for navigation, but remain
            # old evidence with their original admission and measurement roles.
            uv = z['actual_native_uv'][i]
            basis = 'sparse_actual_native_UV_navigation'
            if not np.isfinite(uv).all():
                quant = z['actual_source_side_quantiles_uv'][i, :, 1]
                if np.isfinite(quant).all():
                    uv = quant.mean(0); basis = 'sparse_two_q50_navigation_only_no_midline_admission'
                else: uv = z['uv'][i]; basis = 'sparse_stored_UV_navigation'
            records[(f's{int(z["shot_index"][i]):02d}', name)][int(z['pts'][i])].append((uv, basis, int(i)))
        for key, exposures in records.items():
            for pts, values in exposures.items():
                uv = np.array([v[0] for v in values], float)
                keep = np.isfinite(uv).all(1)
                self.references[key][pts] = dict(uv=np.unique(uv[keep], axis=0),
                    index_rows=[dict(kind=v[1], row=v[2]) for v in values],
                    original_admission_inherited=False)
        self.times = {key: np.array(sorted(v), np.int64) for key, v in self.references.items()}

    def guide(self, sid, pts, path, native, graph):
        key = (sid, path['curve_id']); times = self.times.get(key)
        note = dict(status='current_native_search_only', coordinates_are_observations=False)
        if times is None: return native, note
        insertion = int(np.searchsorted(times, pts))
        candidates = sorted(set([max(0, insertion-1), min(len(times)-1, insertion)]), key=lambda i: abs(int(times[i])-pts))
        for at in candidates:
            refpts = int(times[at]); reference = self.references[key][refpts]
            h, status = navigation_between(graph, refpts, pts)
            if h is None: continue
            xy = np.c_[reference['uv'], np.ones(len(reference['uv']))] @ h.T
            uv = xy[:, :2] / xy[:, 2:]
            if not np.isfinite(uv).all() or len(uv) < 3: continue
            try:
                if path['closed']:
                    # Keep partial source arcs as finite navigation support.
                    # One family transform is applied after all paths are read.
                    projection = native
                    organization = 'legacy_source_points_for_shared_family_navigation'
                elif np.linalg.matrix_rank(path['xyz']-path['xyz'][0]) == 1:
                    guide = line_search_guide(uv)
                    if guide is None: continue
                    projection = dict(uv=guide, valid=np.ones(len(guide), bool))
                    organization = 'legacy_source_straight_search_only'
                else:
                    # Partial nonclosed curves keep the finite native support;
                    # old actual UV supplies a broad search displacement only.
                    from rinkspace.session.native_shape_guidance import point_to_partial_polyline
                    matches = point_to_partial_polyline(uv, native['uv'])
                    shift = np.median(matches.residual_uv, axis=0)
                    projection = dict(native, uv=native['uv'] + shift)
                    organization = 'legacy_partial_curve_search_translation_only'
            except (ValueError, cv2.error, np.linalg.LinAlgError): continue
            note = dict(status=organization, reference_pts=refpts, source_navigation=status,
                H_current_from_reference=h.tolist(), old_index_row_count=len(reference['index_rows']), old_index_kinds=sorted({r['kind'] for r in reference['index_rows']}),
                old_coordinate_count=len(uv), coordinates_are_observations=False,
                physical_identity_confirmed=False)
            note['_source_navigation_uv']=uv
            return projection, note
        return native, dict(note, old_reference_unavailable_in_supported_source_graph=True)


def candidate_rows(z, paths, fragments, line_fragments, ring_results):
    membership = defaultdict(list)
    for fi, fragment in enumerate(fragments):
        for cid in fragment['candidate_ids']: membership[cid].append(fi)
    complete = set()
    for value in ring_results.values():
        for ids in value['qualification'].get('selected', {}).values(): complete.update(ids)
    straight = {i for fragment in line_fragments for i in fragment['candidate_ids']}
    rows = []
    # Thin each actual competing branch independently; keep the complete arrays
    # and branch membership in the NPZ and organization receipt.
    selected = set()
    for fragment in fragments:
        ids = np.asarray(fragment['candidate_ids'], int)
        selected.update(map(int, ids[spaced_indices(z['candidate_uv'][ids], 48.)]))
    selected.update(complete); selected.update(straight)
    for ci, path in enumerate(paths):
        ids = sorted(i for i in selected if int(z['query_curve'][z['candidate_query'][i]]) == ci)
        # Complete and line consensus can be dense; 48px thinning is within a
        # contour/branch, never across distinct entities or alternatives.
        by_branch = defaultdict(list)
        for i in ids: by_branch[tuple(membership[i])].append(i)
        for branch, values in by_branch.items():
            values = np.asarray(sorted(values, key=lambda i: int(z['candidate_query'][i])), int)
            for i in values[spaced_indices(z['candidate_uv'][values], 48.)]:
                i = int(i); qi = int(z['candidate_query'][i])
                rows.append(dict(path_id=path['native_path_id'], entity=path['curve_id'], candidate_id=i,
                    uv=z['candidate_uv'][i], query_index=qi,
                    fragment_ids=list(branch), source_flags=int(z['candidate_flags'][i]),
                    complete_color_stack_candidate=i in complete, visible_line_candidate=i in straight,
                    partial_arc_candidate=path['operator']=='material_interface' and i not in complete,
                    review_state='pending'))
    return rows


def capture(args):
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Shard index must belong to the positive shard count')
    if args.shard_count > 1 and (not args.worker_shot or args.pts):
        raise ValueError('Contiguous sharding requires one worker shot and no explicit PTS')
    root=args.root; index=read(root/'index.json'); manifest=read(args.prepared/'joint_manifest.json')
    source_sha=index['source']['actual_sha256']
    if digest(index['source']['resolved_path']) != source_sha: raise ValueError('Original video hash changed')
    state=dict(np.load(args.state, allow_pickle=False)); library=Path(manifest['native_library'])
    if digest(library) != str(state['native_library_provenance_sha256']): raise ValueError('State/library mismatch')
    scene_path=Path(manifest['scene_path'])
    if digest(scene_path) != manifest['scene_file_sha256']: raise ValueError('Scene changed')
    raw=read(scene_path); model=SceneParameterModel(raw, scene_path, optimize_scene=True)
    parameter_state=ParameterState(model.initial_state.layout, state['scene_values']); core=NativeCore(library)
    samples=sample_native_review_scene(raw, core, scene_path, parameter_state=parameter_state, observed_path_ids=set())
    labels={c['id']:c['label'] for c in read(root/'native_world.json')['cuts']}
    paths=native_inventory(samples['cuts'], samples['xyz'], labels); cuts={c['id']:c for c in samples['cuts']}
    searched_paths, signal_override_evidence = source_search_selection(paths,
        args.feature_ids, args.paint_signal_overrides)
    setup=time.perf_counter(); legacy=LegacyNavigation(args.legacy, paths, source_sha)
    print(json.dumps(dict(stage='legacy_navigation_loaded',seconds=time.perf_counter()-setup,paths=len(paths),legacy_source_groups=sum(map(len,legacy.references.values())))),flush=True)
    ice_chunks=[]; offset=0
    for path in paths:
        path['_projection_slice']=(offset,offset+len(path['xyz']));ice_chunks.append(path['xyz']);offset+=len(path['xyz'])
    for path in paths:
        if 'center_xyz' in path:
            path['_projection_center']=offset;ice_chunks.append(path['center_xyz'][None]);offset+=1
    ice_xyz=np.vstack(ice_chunks)
    timeline=ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    args.output=guard_project_write(args.output); args.output.mkdir(parents=True, exist_ok=True)
    from rinkspace.session import projected_curve_queries, native_shape_guidance, material_ring_qualification
    algorithm_files = {IceProgrammeSourceReader.read.__code__.co_filename,
        assemble_fragments.__code__.co_filename, source_ring_hypotheses.__code__.co_filename,
        source_exclusion.__code__.co_filename, ice_source_mask.__code__.co_filename,
        projected_curve_queries.__file__, native_shape_guidance.__file__, material_ring_qualification.__file__}
    cuda_profiles=Path(IceProgrammeSourceReader.read.__code__.co_filename).with_name('vision_source_cuda_profiles.py')
    if args.profile_backend=='cuda':algorithm_files.add(str(cuda_profiles))
    binding=dict(schema='rinkspace.legacy_navigated_original_ice_reread.v1', source_video_sha256=source_sha,
        source_index_sha256=digest(root/'index.json'), state_path=str(args.state), state_sha256=digest(args.state),
        prepared_manifest_path=str(args.prepared/'joint_manifest.json'), prepared_manifest_sha256=digest(args.prepared/'joint_manifest.json'),
        native_library_sha256=digest(library), scene_sha256=digest(scene_path), adapter_sha256=digest(__file__),
        algorithm_sha256={str(Path(p).resolve()):digest(p) for p in sorted(algorithm_files)},
        profile_backend=args.profile_backend,
        cuda_profile_implementation=dict(path=str(cuda_profiles.resolve()),sha256=digest(cuda_profiles),
            used=args.profile_backend=='cuda'),
        legacy_navigation_inputs=legacy.inputs, unmapped_old_labels=dict(legacy.unmapped),
        native_inventory=[{k:v for k,v in p.items() if k not in ('xyz','center_xyz','_projection_slice','_projection_center')} for p in paths],
        full_native_inventory_count=len(paths),requested_feature_ids=args.feature_ids,
        searched_native_path_ids=[p['native_path_id'] for p in paths if p['native_path_id'] in searched_paths],
        searched_curve_ids=[p['curve_id'] for p in paths if p['native_path_id'] in searched_paths],
        paint_signal_overrides=args.paint_signal_overrides or {},paint_signal_override_evidence=signal_override_evidence,
        query_curve_indexes_full_native_inventory=True,unrequested_paths_are_not_absence_evidence=True,
        query_spacing_px=args.spacing, search_radius_px=args.radius, output_spacing_px=48.,
        frame_metadata_compression='gzip_level1', source_time_base='1/12800', native_pts_step=256, pixel_domain='native_distorted', raw_quantiles_field='candidate_side_quantiles_uv', raw_sides_field='candidate_sides', rows_reference_raw_arrays_by_candidate_id=True,
        all_RGB_alternatives_retained=True, partial_arcs_retained_without_complete_stack=True,
        old_UV_and_native_projection_are_navigation_only=True, new_fits_started=0,
        prepared_modified=False, fit_eligible=False, requested_shots=args.shots)
    old_binding=args.output/'binding.json'
    if old_binding.exists() and read(old_binding)!=plain(binding):
        if list(args.output.glob('s??/[0-9]*.npz')): raise ValueError('Output binding differs')
        with (args.output/'failed_initial_bindings.jsonl').open('a',encoding='utf8') as failed:failed.write(json.dumps(read(old_binding),ensure_ascii=False)+'\n')
    if not old_binding.exists() or read(old_binding)!=plain(binding): write(old_binding,binding)
    cv2.setNumThreads(2); start=time.monotonic()
    reader=IceProgrammeSourceReader(None, source_sha256=source_sha, time_base='1/12800', native_paths=paths,
        sample_spacing_px=args.spacing, search_radius_px=args.radius, separate_bands=True,
        profile_backend=args.profile_backend,
        prepared_source_context=dict(source_sha256=source_sha,time_base='1/12800',authority=[dict(path=str(scene_path),sha256=digest(scene_path))]))
    for sid in args.shots:
        if args.worker_shot and sid!=args.worker_shot:continue
        shot=next(s for s in manifest['shots'] if s['id']==sid); source=next(s for s in index['shots'] if s['shot_id']==sid)
        graph, graph_note=load_navigation_graph(root,source); folder=args.output/sid; folder.mkdir(exist_ok=True)
        counts=Counter(); frame_summaries=[]
        if not (folder/'source_navigation.json').exists(): write(folder/'source_navigation.json',graph_note)
        selected_pts=args.pts
        if args.shard_count > 1:
            complete=np.arange(source['processing']['first_pts'],source['processing']['stop_pts_exclusive'],256,dtype=np.int64)
            selected_pts=np.array_split(complete,args.shard_count)[args.shard_index].tolist()
        for fi,(pts,image),decode_seconds in timed_capture_frames(index,source,selected_pts):
            phase=time.perf_counter(); timing=dict(decode_s=decode_seconds)
            target=folder/f'{pts}.json.gz'; cache=folder/f'{pts}.npz'
            if target.exists() and cache.exists() and not args.reread:
                result=read(target); counts.update(r['entity'] for r in result['rows']);frame_summaries.append(result['summary']);continue
            clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
            proj=core.project_motion(state['rig'][shot['camera_index']], state['controls'], timeline.first,timeline.dt,1080,clock,ice_xyz)
            native={}; guided={}; navigation={}
            for path in paths:
                begin,end=path['_projection_slice']; uv=proj['uv'][begin:end].copy(); valid=proj['status'][begin:end]==0;uv[~valid]=np.nan
                p=dict(uv=uv,valid=valid)
                if 'center_xyz' in path:
                    at=path['_projection_center'];center=proj['uv'][at]
                    if proj['status'][at] or not np.isfinite(center).all():
                        p['centre_uv']=np.zeros(2);p['valid']=np.zeros(len(valid),bool)
                    else:p['centre_uv']=center
                native[path['native_path_id']]=p
                if path['native_path_id'] in searched_paths:
                    guided[path['native_path_id']],navigation[path['curve_id']]=legacy.guide(sid,pts,path,p,graph)
                else:
                    guided[path['native_path_id']]=dict(p,valid=np.zeros(len(p['valid']),bool))
                    navigation[path['curve_id']]=dict(status='not_requested_in_source_subset',
                        coordinates_are_observations=False,physical_absence_asserted=False)
            # Original source arcs organize one common transform per physical
            # family. Partial arcs never independently extrapolate an ellipse.
            from rinkspace.session.native_shape_guidance import point_to_partial_polyline
            for family in sorted({p['feature_id'] for p in paths if p['closed'] and p['native_path_id'] in searched_paths}):
                members=[p for p in paths if p['feature_id']==family and p['closed'] and p['native_path_id'] in searched_paths]
                observed=[]; names=[]; polylines={}
                for p in members:
                    uv=navigation[p['curve_id']].get('_source_navigation_uv')
                    line=native[p['native_path_id']]['uv']
                    if uv is not None and len(uv)>=3 and np.isfinite(line).all():
                        observed.extend(uv); names.extend([p['curve_id']]*len(uv)); polylines[p['curve_id']]=line
                if not observed: continue
                # Search positioning does not need an iterative affine fit.
                # Partial nearly flat arcs leave affine directions weak and
                # could otherwise spend minutes refining an unobservable map.
                # One common displacement preserves every native ring shape;
                # the broad original-RGB profiles still measure alternatives.
                residual=[]
                for p in members:
                    uv=navigation[p['curve_id']].get('_source_navigation_uv')
                    if uv is not None and p['curve_id'] in polylines:
                        residual.extend(point_to_partial_polyline(uv,polylines[p['curve_id']]).residual_uv)
                if not residual:continue
                affine=np.eye(3);affine[:2,2]=np.median(residual,axis=0)
                method='direct_shared_source_family_translation_navigation_only'
                for p in members:
                    original=native[p['native_path_id']]; guide=dict(original)
                    guide['uv']=original['uv']@affine[:2,:2].T+affine[:2,2]
                    if 'centre_uv' in original:guide['centre_uv']=original['centre_uv']@affine[:2,:2].T+affine[:2,2]
                    guided[p['native_path_id']]=guide
                    navigation[p['curve_id']]['shared_family_navigation']=dict(method=method,family=family,affine=affine.tolist(),coordinates_are_observations=False)
            for note in navigation.values():note.pop('_source_navigation_uv',None)
            timing['native_projection_and_old_navigation_s']=time.perf_counter()-phase; phase=time.perf_counter()
            if args.pts:print(json.dumps(dict(stage='navigation_ready',pts=pts,timing=timing)),flush=True)
            invalid,mask_path=source_exclusion(root,sid,pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            invalid,floor_receipt=ice_source_mask(rgb,invalid)
            timing['source_mask_s']=time.perf_counter()-phase; phase=time.perf_counter()
            z=reader.read(rgb,pts,None,invalid,shot_id=sid,native_projections=guided)
            timing['actual_RGB_profiles_s']=time.perf_counter()-phase; phase=time.perf_counter()
            if args.pts:print(json.dumps(dict(stage='RGB_read',queries=len(z['query_uv']),candidates=len(z['candidate_uv']),timing=timing)),flush=True)
            fragments=assemble_fragments(z,paths)
            timing['assemble_fragments_s']=time.perf_counter()-phase; phase=time.perf_counter()
            if args.pts:print(json.dumps(dict(stage='fragments',count=len(fragments),timing=timing)),flush=True)
            straight_indices=[ci for ci,p in enumerate(paths) if p['operator']=='paint_center' and np.linalg.matrix_rank(p['xyz']-p['xyz'][0])==1]
            line_fragments,line_report=qualify_visible_line_fragments(z,straight_indices,invalid,fragments=fragments)
            timing['line_qualification_s']=time.perf_counter()-phase; phase=time.perf_counter()
            rings={}
            for house in sorted({p['feature_id'] for p in paths if len(p['feature_id'])==2 and p['feature_id'][0] in 'FR'}):
                hypotheses={p['world_curve_role']:source_ring_hypotheses(rgb,z,ci,fragments) for ci,p in enumerate(paths) if p['feature_id']==house}
                nesting=nested_source_rings(hypotheses)
                rings[house]=dict(hypotheses=hypotheses,nesting=nesting,qualification=color_constrained_ring_stack(rgb,z,hypotheses,nesting=nesting))
            timing['complete_ring_qualification_s']=time.perf_counter()-phase; phase=time.perf_counter()
            rows=candidate_rows(z,paths,fragments,line_fragments,rings)
            np.savez_compressed(cache,selected_candidate_ids=np.array([r['candidate_id'] for r in rows],np.int32),**{k:v for k,v in z.items() if isinstance(v,np.ndarray)})
            by_path={p['curve_id']:dict(search_requested=p['native_path_id'] in searched_paths,
                queries=int(sum(z['query_curve']==ci)),actual_RGB_candidates=int(sum(z['query_curve'][z['candidate_query']]==ci)),
                clean_RGB_candidates=int(sum((z['query_curve'][z['candidate_query']]==ci)&((z['candidate_flags']&3)==0))),
                partial_or_complete_fragments=sum(f['curve']==ci for f in fragments),rows=sum(r['entity']==p['curve_id'] for r in rows)) for ci,p in enumerate(paths)}
            summary=dict(pts=pts,rows=len(rows),actual_RGB_candidates=len(z['candidate_uv']),queries=len(z['query_uv']),by_path=by_path,
                complete_color_stack_houses=[h for h,v in rings.items() if v['qualification']['selected']],seconds=time.monotonic()-start)
            result=dict(pts=pts,source_seconds=pts/12800,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                source_RGB_sha256=z['decoded_native_RGB_sha256'],source_video_sha256=source_sha,
                ice_floor_guard=floor_receipt,mask_path=str(mask_path),mask_sha256=digest(mask_path),source_exclusion_sha256=hashlib.sha256(invalid.tobytes()).hexdigest(),
                raw_candidates=dict(path=str(cache),sha256=digest(cache)),summary=summary,rows=rows,
                fragments=fragments,line_fragments=line_fragments,line_qualification=line_report,rings=rings,navigation=navigation,
                fit_eligible=False,physical_identity_confirmed=False,prepared_modified=False)
            timing['rows_and_npz_write_s']=time.perf_counter()-phase; result['timing']=timing
            write(target,result);counts.update(r['entity'] for r in rows);frame_summaries.append(summary)
            if args.pts or fi%250==0 or pts==source['processing']['last_pts']:
                proof=image.copy()
                for ci,p in enumerate(paths):
                    uv=native[p['native_path_id']]['uv']; good=np.isfinite(uv).all(1)&(np.abs(uv)<20000).all(1)
                    segments=np.split(np.flatnonzero(good),np.flatnonzero(np.diff(np.flatnonzero(good))>1)+1)
                    for seg in segments:
                        if len(seg)>1:cv2.polylines(proof,[np.rint(uv[seg]).astype('int32')],False,(110,90,30),1,cv2.LINE_AA)
                for row in rows:
                    uv=np.asarray(row['uv']);cv2.circle(proof,tuple(np.rint(uv).astype(int)),3,(0,255,0) if row['complete_color_stack_candidate'] else (0,210,255),1)
                cv2.putText(proof,f'{sid} PTS {pts} | {len(rows)} RGB candidate rows | ALL PENDING | NO FIT',(20,35),0,.6,(0,220,255),2)
                cv2.imwrite(str(folder/f'{pts}_source_candidates.jpg'),proof)
            if args.pts or fi%10==0: print(json.dumps(dict(shot=sid,frame=fi+1,pts=pts,rows=len(rows),candidates=len(z['candidate_uv']),complete_stack=summary['complete_color_stack_houses'],timing=timing,seconds=time.monotonic()-start)),flush=True)
        coverage_name=f'coverage_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json' if args.shard_count>1 else 'coverage.json'
        write(folder/coverage_name,dict(shot=sid,frames=len(frame_summaries),expected_native_frames=source['processing']['native_frame_count'],
            shard_index=args.shard_index,shard_count=args.shard_count,
            actual_RGB_candidates=sum(s['actual_RGB_candidates'] for s in frame_summaries),rows=sum(s['rows'] for s in frame_summaries),
            by_path=dict(counts),frames_summary=frame_summaries,fit_eligible=False,prepared_modified=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shots',nargs='+',default=['s07','s08','s09'])
    p.add_argument('--worker-shot',choices=['s07','s08','s09']);p.add_argument('--pts',nargs='+',type=int);p.add_argument('--spacing',type=float,default=24.)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--radius',type=float,default=48.);p.add_argument('--reread',action='store_true')
    p.add_argument('--profile-backend',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--feature-ids',nargs='+',help='Only these exact native feature IDs generate source queries; the full inventory remains bound')
    p.add_argument('--paint-signal-overrides',type=json.loads,help='JSON feature-to-channel object, e.g. {"C0":"dark"}; paint_center paths only')
    capture(p.parse_args())


if __name__=='__main__': main()
