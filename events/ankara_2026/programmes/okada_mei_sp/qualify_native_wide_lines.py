"""Convert frozen actual wide-line readouts; no reading, geometry or fitting."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import argparse
import json

import cv2
import numpy as np

import run_native_programme as programme
from capture_independent_materials import spaced_indices
from prepare_native_source_update import authenticate_replacement_spans, in_remeasured_line_spans


RUN = Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031')
BASE = OUT = PREPARED = None
REPLACEMENT_PREPARED = None
APPROVED = {}
WITHHELD = {}
INHERIT_SOURCE_SEEDS = True
HASHES = {}


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def digest(path):
    path = Path(path)
    if path not in HASHES:
        with path.open('rb') as stream:
            HASHES[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return HASHES[path]


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def line(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'


def write(path, value):
    Path(path).write_text(line(value), encoding='utf-8')


def replacement_parent_binding(source_prepared, source_input_sha256, replacement_prepared=None):
    """Keep frozen measurement provenance while replacing the current dataset.

    A new numeric fit does not invalidate measured RGB. Reusing those readings
    must nevertheless compute explicit replacement IDs against the actual new
    parent, never relabel the old capture as if it came from the new state.
    """
    source_prepared=Path(source_prepared)
    target=Path(replacement_prepared or source_prepared)
    source_manifest=read(source_prepared/'joint_manifest.json')
    target_manifest=read(target/'joint_manifest.json')
    if (source_manifest['point_input_sha256']!=source_input_sha256 or
            digest(source_prepared/'point_observations.npz')!=source_input_sha256):
        raise ValueError('Source capture parent input differs')
    if digest(target/'point_observations.npz')!=target_manifest['point_input_sha256']:
        raise ValueError('Replacement parent input differs')
    if source_manifest['scene_file_sha256']!=target_manifest['scene_file_sha256']:
        raise ValueError('Replacement parent belongs to a different scene definition')
    fields=('id','first_pts','last_pts','reference_pts','camera_index')
    scope=lambda m:[tuple(s[k]for k in fields)for s in m['shots']]
    if scope(source_manifest)!=scope(target_manifest):
        raise ValueError('Replacement parent source-shot or camera binding differs')
    return target,target_manifest,dict(source_capture_prepared=str(source_prepared.resolve()),
        source_capture_input_sha256=source_input_sha256,
        replacement_prepared=str(target.resolve()),replacement_input_sha256=target_manifest['point_input_sha256'],
        original_capture_binding_preserved=True)


def measured_piece_members(old_uv, measurement, ids, tolerance=3.):
    """Retire only the same material actually remeasured along a visible piece.

    Finite actual UV segments supply tangential extent. Measured paint sides
    supply the normal footprint; a remote or opposite circle arc stays intact.
    """
    old_uv=np.asarray(old_uv,float).reshape(-1,2);inside=np.zeros(len(old_uv),bool)
    uv=measurement['candidate_uv'];bounds=measurement['candidate_side_quantiles_uv']
    if len(ids)==1:return np.all(old_uv==uv[ids[0]],axis=1)
    for first,last in zip(ids,ids[1:]):
        a,b=uv[[first,last]].astype(float);direction=b-a;length2=float(direction@direction)
        if length2<=1e-12:continue
        t=(old_uv-a)@direction/length2
        normal=np.array([-direction[1],direction[0]])/np.sqrt(length2)
        side_a=(bounds[first].reshape(-1,2)-a)@normal
        side_b=(bounds[last].reshape(-1,2)-b)@normal
        lo=(1-t)*side_a.min()+t*side_b.min()-tolerance
        hi=(1-t)*side_a.max()+t*side_b.max()+tolerance
        across=(old_uv-a)@normal
        inside |= (t>=0)&(t<=1)&(across>=lo)&(across<=hi)
    return inside


def bgr_ledger(sid, expected):
    if sid in ('s01', 's02', 's03', 's04', 's05', 's06'):
        visibility, frames = programme.source_frames(sid)
        if sid == 's01':
            path = programme.DATA/'runs/native_ad_replacement_s01_20261026/visibility.json'
        elif sid == 's04':
            path = programme.DATA/'runs/native_ad_points_s04_20261026/visibility.json'
        else:
            date = '20261028' if sid in ('s03', 's05') else '20261027'
            path = programme.DATA/'runs'/f'native_points_{sid}_{date}/evidence/visibility.json'
        original = read(path)['frames']
        offsets = {f['pts']: i for i, f in enumerate(original)}
        records = {f['pts']: dict(sha=f['source_BGR_sha256'],
            locator='/frames/'+str(offsets[f['pts']])) for f in frames}
    else:
        path = RUN/'legacy_comparison/registration_batch_lines/source_packets'/sid/'source_frames.json'
        frames = read(path)
        records = {f['pts']: dict(sha=f['source_BGR_sha256'], locator='/'+str(i))
                   for i, f in enumerate(frames)}
    if sorted(records) != expected:
        raise ValueError('Incomplete original BGR ledger: '+sid)
    if any(not isinstance(r['sha'], str) or len(r['sha']) != 64 for r in records.values()):
        raise ValueError('Missing original BGR pixels identity: '+sid)
    return records, ref(path)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    inventory = read(BASE/'all9_inventory.json')
    component_doc = read(BASE/'pending_components.json')
    components = {c['component_id']: c for c in component_doc['components']}
    scope = programme.read(programme.HERE/'rebuild_scope.json')
    source_sha = scope['source']['sha256']
    parent_prepared,manifest,parent_binding=replacement_parent_binding(PREPARED,inventory['input_sha256'],REPLACEMENT_PREPARED)
    parent_mapping=read(parent_prepared/'mapping.json')
    searched=set()
    for shot in inventory['shots'].values():
        searched.update(p['native_path_id'] for p in read(BASE/shot['batch']/'binding.json')['inventory'])
    with np.load(parent_prepared/'point_observations.npz',allow_pickle=False) as arrays:
        selected=np.flatnonzero(np.isin(arrays['path_indices'],[parent_mapping['path_ids'].index(p)for p in searched]))
        parent={key:arrays[key][selected]for key in ('pts','path_indices','uv','observation_ids')}
    parent_groups=defaultdict(list)
    for i,(pt,pi) in enumerate(zip(parent['pts'],parent['path_indices'])):
        parent_groups[int(pt),parent_mapping['path_ids'][int(pi)]].append(i)
    if digest(scope['source']['registered_path']) != source_sha:
        raise ValueError('Original source video identity changed')
    approved_indices = defaultdict(dict)
    image_reviews = []
    for identity, conclusion in APPROVED.items():
        component = components[identity]
        rep = component['representative']
        stem = f"{component['shot']}_{rep['pts']}"
        source = BASE/'pending_review'/f'{stem}_source.png'
        actual = BASE/'pending_review'/f'{stem}_actual.png'
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or hashlib.sha256(image.tobytes()).hexdigest() != rep['source_BGR_sha256']:
            raise ValueError('Reviewed original BGR pixels changed: '+stem)
        image_reviews.append(dict(component_id=identity, shot=component['shot'], pts=rep['pts'],
            source_BGR_sha256=rep['source_BGR_sha256'], source_image=ref(source),
            actual_readout_image=ref(actual), conclusion=conclusion,
            reviewer='Explicit current-session source-image review recorded in --reviews',
            review_scope='The declared source-linked component, retaining original source UV/PTS/identity',
            user_visual_acceptance=False))
        for index in component['pending_row_indices']:
            if index in approved_indices[component['shot']]:
                raise ValueError('Overlapping explicit component review')
            approved_indices[component['shot']][index] = identity
    OUT.mkdir()
    decisions = dict(schema='rinkspace.actual_wide_line_source_qualification.v1',
        source_video_sha256=source_sha, input_sha256=manifest['point_input_sha256'],
        parent_binding=parent_binding,
        state_sha256=inventory['state_sha256'], inventory=ref(BASE/'all9_inventory.json'),
        pending_components=ref(BASE/'pending_components.json'),
        existing_linked_source_policy=('Unchanged source whole-line organization; same-exposure frozen actual UV overlap seed; actual bidirectional RGB source links'
            if INHERIT_SOURCE_SEEDS else 'Withheld: this material family requires new explicit original-source identity review'),
        explicit_new_component_reviews=image_reviews,
        source_identity_scope='Only explicitly reviewed pending components plus inherited same-source seeds and verified RGB links',
        all_other_pending_disposition='Unchanged original pending records remain withheld',
        sparse_policy=dict(implementation=str(Path(spaced_indices.__code__.co_filename).resolve()),
            implementation_sha256=digest(spaced_indices.__code__.co_filename), spacing_px=48.,
            partition='Original fragment source_visible_candidate_pieces, split again at any ineligible original candidate',
            endpoints='Both endpoints retained as exact original measurements; no projected or interpolated UV',
            spans='Separate actual-source pieces only; singleton fallback when tangential spans would include withheld readouts'),
        fit_qualification_is_not_user_visual_acceptance=True, user_visual_acceptance=False,
        pixel_accuracy_accepted=False, UV_modified=False, weights_added=False,
        prepared_modified=False, packet_built=False, fit_executed=False)
    write(OUT/'review_decisions.json', decisions)
    review_ref = ref(OUT/'review_decisions.json')
    summary = dict(schema='rinkspace.qualified_wide_line_capture.v1', source_video_sha256=source_sha,
        input_sha256=manifest['point_input_sha256'], state_sha256=inventory['state_sha256'],
        parent_binding=parent_binding,
        review=review_ref, shots={}, fit_eligible=True, user_visual_acceptance=False,
        pixel_accuracy_accepted=False, UV_modified=False, weights_added=False,
        prepared_modified=False, packet_built=False, fit_executed=False)
    for sid, shot in inventory['shots'].items():
        capture = BASE/shot['batch']/sid
        source_path = capture/'source_pixels.json'
        if digest(source_path) != shot['source_pixels']['sha256']:
            raise ValueError('Original source rows changed: '+sid)
        source = read(source_path)
        source_ref = ref(source_path)
        expected = list(range(shot['first_pts'], shot['last_pts']+256, 256))
        bgr, bgr_ref = bgr_ledger(sid, expected)
        folder = OUT/sid
        folder.mkdir()
        eligible = defaultdict(list)
        withheld = defaultdict(list)
        counts = Counter({key: 0 for key in ('eligible_original_rows', 'original_linked_rows',
            'explicit_review_component_rows', 'pending_rows', 'qualified_rows',
            'sparse_omitted_rows', 'qualified_source_pieces', 'frames', 'frames_with_rows')})
        with (folder/'pending.jsonl').open('x', encoding='utf-8') as pending_out:
            for collection in ('accepted', 'pending'):
                for index, row in enumerate(source[collection]):
                    if row['source_BGR_sha256'] != bgr[row['pts']]['sha']:
                        raise ValueError('Source row and exact original BGR ledger differ')
                    locator = dict(source=source_ref, collection=collection, index=index,
                                   json_pointer=f'/{collection}/{index}')
                    component = approved_indices[sid].get(index) if collection == 'pending' else None
                    withheld_reason=WITHHELD.get((sid,int(row['pts']),row['path_id'],*map(float,row['uv'])))
                    if ((collection == 'accepted' and INHERIT_SOURCE_SEEDS) or component is not None) and withheld_reason is None:
                        eligible[row['pts']].append((row, locator, component))
                        counts['eligible_original_rows'] += 1
                        counts['original_linked_rows' if collection == 'accepted' else 'explicit_review_component_rows'] += 1
                    else:
                        pending_out.write(line(dict(row=row, source_row_locator=locator,
                            fit_eligible=False, reason=withheld_reason or 'original_component_not_source_qualified')))
                        withheld[(row['pts'], row['path_id'])].append(row['uv'])
                        counts['pending_rows'] += 1
        spans_all = [];replacement_all=[]
        kept_for_auth = []
        with (folder/'source_rows.jsonl').open('x', encoding='utf-8') as output, \
                (folder/'sparse_omitted.jsonl').open('x', encoding='utf-8') as omitted_out, \
                (folder/'source_piece_audit.jsonl').open('x', encoding='utf-8') as pieces_out:
            for stamp in expected:
                originals = eligible.get(stamp, [])
                rows = []
                spans_by_path = defaultdict(list)
                rows_by_path = defaultdict(list)
                replacing=defaultdict(set)
                covered_originals = set()
                if originals:
                    frame_path = capture/f'{stamp}.json'
                    frame = read(frame_path)
                    frame_ref = ref(frame_path)
                    by_fragment = defaultdict(dict)
                    for original, locator, component in originals:
                        key = int(original['candidate'])
                        group = by_fragment[int(original['fragment'])]
                        if key in group:
                            raise ValueError('Duplicate original candidate within a fragment')
                        group[key] = (original, locator, component)
                    with np.load(capture/f'{stamp}.npz', allow_pickle=False) as arrays:
                        original_uv = arrays['candidate_uv']
                        measurement={key:arrays[key]for key in ('candidate_uv','candidate_side_quantiles_uv')}
                        raw_ref = ref(capture/f'{stamp}.npz')
                        for fragment_index, candidates in by_fragment.items():
                            fragment = frame['fragments'][fragment_index]
                            pieces = fragment['source_visible_candidate_pieces']
                            if not set(candidates) <= set(fragment['candidate_ids']):
                                raise ValueError('Qualified row lost its original whole-line fragment')
                            runs = []
                            for piece_index, piece in enumerate(pieces):
                                run = []
                                for candidate in piece:
                                    if candidate in candidates:
                                        run.append(int(candidate))
                                    elif run:
                                        runs.append((piece_index, run)); run = []
                                if run:
                                    runs.append((piece_index, run))
                            for piece_index, ids in runs:
                                locations = [candidates[cid] for cid in ids]
                                points = np.asarray([r[0]['uv'] for r in locations])
                                if not np.array_equal(points, original_uv[ids]):
                                    raise ValueError('Qualified source pixels differ from original raw readouts')
                                kept = sorted(set(map(int, spaced_indices(points, spacing=48.))) | {0, len(ids)-1})
                                selected_ids = [ids[i] for i in kept]
                                endpoint = [locations[0][0]['uv'], locations[-1][0]['uv']]
                                path_id = locations[0][0]['path_id']
                                assert all(item[0]['path_id'] == path_id for item in locations)
                                prior=np.asarray(parent_groups.get((stamp,path_id),[]),int)
                                if len(prior):
                                    measured=measured_piece_members(parent['uv'][prior],measurement,ids)
                                    replacing[path_id].update(str(x)for x in parent['observation_ids'][prior[measured]])
                                spans_by_path[path_id].append(endpoint)
                                for local_index, (original, locator, component) in enumerate(locations):
                                    covered_originals.add((fragment_index, ids[local_index]))
                                    if local_index not in kept:
                                        omitted_out.write(line(dict(source_row_locator=locator,
                                            pts=stamp, path_id=path_id, candidate_id=ids[local_index],
                                            uv=original['uv'], qualification_retained=True,
                                            reason='not_selected_by_existing_48px_source_piece_spacing')))
                                        counts['sparse_omitted_rows'] += 1
                                        continue
                                    qualified = dict(original, entity=original['identity'],
                                        candidate_id=int(original['candidate']), capture=shot['batch'],
                                        measurement_sha256=raw_ref['sha256'], source_row_locator=locator,
                                        fit_eligible=True, review_state='source_qualified_not_user_accepted',
                                        source_review=review_ref, source_component_review=component,
                                        qualification_basis=('main_session_original_image_component_review'
                                            if component is not None else 'original_actual_source_seed_and_RGB_links'),
                                        source_piece=dict(frame_metadata=frame_ref, fragment=fragment_index,
                                            source_visible_piece=piece_index,
                                            original_piece_index=pieces[piece_index].index(ids[local_index]),
                                            original_candidate_id=ids[local_index]), user_visual_acceptance=False)
                                    for name, value in original.items():
                                        if qualified[name] != value:
                                            raise ValueError('An original measurement/identity field changed')
                                    rows.append(qualified)
                                    rows_by_path[path_id].append(qualified)
                                pieces_out.write(line(dict(pts=stamp, path_id=path_id, fragment=fragment_index,
                                    source_visible_piece=piece_index, original_candidate_ids=ids,
                                    kept_candidate_ids=selected_ids, kept_local_indices=kept,
                                    actual_original_endpoints=endpoint, frame_metadata=frame_ref,
                                    raw_measurement=raw_ref)))
                                counts['qualified_source_pieces'] += 1
                    if covered_originals != {(int(r['fragment']), int(r['candidate'])) for r, _, _ in originals}:
                        raise ValueError('Eligible original candidate is missing from visible source pieces')
                for path_id, spans in spans_by_path.items():
                    pending_uv = withheld.get((stamp, path_id), [])
                    retained = {tuple(r['uv']) for r in rows_by_path[path_id]}
                    safe = []
                    for span in spans:
                        if any(in_remeasured_line_spans(uv, np.asarray([span], float)) for uv in pending_uv):
                            safe.extend([[list(uv), list(uv)] for uv in sorted(retained)
                                if in_remeasured_line_spans(uv, np.asarray([span], float))])
                            counts['spans_reduced_to_singletons_to_preserve_pending'] += 1
                        else:
                            safe.append(span)
                    for uv in retained:
                        if not in_remeasured_line_spans(uv, np.asarray(safe, float)):
                            safe.append([list(uv), list(uv)])
                            counts['retained_outside_endpoint_projection_singletons'] += 1
                    unique = []
                    seen = set()
                    for span in safe:
                        key = tuple(map(tuple, span))
                        if key not in seen:
                            seen.add(key); unique.append(span)
                    if any(in_remeasured_line_spans(uv, np.asarray(unique, float)) for uv in pending_uv):
                        raise ValueError('A replacement span would include an unqualified original pixel')
                    if any(tuple(uv) not in retained for span in unique for uv in span):
                        raise ValueError('Span endpoint is not an actual retained original UV')
                    spans_all.append(dict(pts=stamp, path_id=path_id, spans=unique))
                    replacement_all.append(dict(pts=stamp,path_id=path_id,parent_observation_ids=sorted(replacing[path_id])))
                for row in rows:
                    kept_for_auth.append((stamp, row['path_id'], row['uv']))
                record = dict(shot=sid, shot_id=sid, pts=stamp, source_BGR_sha256=bgr[stamp]['sha'],
                    source_video_sha256=source_sha, time_base='1/12800',
                    source_BGR_ledger=dict(source=bgr_ref, json_pointer=bgr[stamp]['locator']),
                    rows=rows, fit_eligible=bool(rows), source_UV_unchanged=True,
                    review_state='source_qualified_not_user_accepted' if rows else 'no_qualified_rows_at_this_source_PTS',
                    source_review=review_ref, user_visual_acceptance=False)
                output.write(line(record))
                counts['frames'] += 1
                counts['frames_with_rows'] += bool(rows)
                counts['qualified_rows'] += len(rows)
        declaration = dict(policy='explicit_parent_observation_ids',parent_point_input_sha256=manifest['point_input_sha256'],groups=replacement_all)
        path_ids = list(dict.fromkeys(path for _, path, _ in kept_for_auth))
        points = dict(pts=np.asarray([pt for pt, _, _ in kept_for_auth], np.int64),
            path_indices=np.asarray([path_ids.index(path) for _, path, _ in kept_for_auth], np.int32),
            uv=np.asarray([uv for _, _, uv in kept_for_auth], float).reshape(-1, 2))
        authenticated = authenticate_replacement_spans(dict(path_ids=path_ids, source_replacement=dict(policy='partial_line_spans',groups=spans_all)), points)
        if len(authenticated) != len(spans_all):
            raise ValueError('Consumer partial-span authentication failed')
        if counts['eligible_original_rows'] != counts['qualified_rows']+counts['sparse_omitted_rows']:
            raise ValueError('Sparse selection lost original source indices')
        write(folder/'source_replacement.json', declaration)
        summary['shots'][sid] = dict(counts, source_rows=ref(folder/'source_rows.jsonl'),
            source_replacement=ref(folder/'source_replacement.json'), pending=ref(folder/'pending.jsonl'),
            sparse_omitted=ref(folder/'sparse_omitted.jsonl'), source_piece_audit=ref(folder/'source_piece_audit.jsonl'),
            original_source=source_ref, source_BGR_ledger=bgr_ref,
            actual_retained_endpoints_authenticated=True, original_UV_PTS_identity_unchanged=True,
            replacement_parent_rows=sum(len(g['parent_observation_ids'])for g in replacement_all),
            replacement_policy='same source PTS and native path; finite measured piece with measured paint-side normal footprint',
            no_packet_required=counts['qualified_rows'] == 0)
        print(line(dict(shot=sid, **dict(counts))), end='', flush=True)
    summary['total_frames'] = sum(s['frames'] for s in summary['shots'].values())
    summary['qualified_rows'] = sum(s['qualified_rows'] for s in summary['shots'].values())
    summary['remaining_pending_rows'] = sum(s['pending_rows'] for s in summary['shots'].values())
    summary['explicit_component_original_rows'] = sum(s['explicit_review_component_rows'] for s in summary['shots'].values())
    expected_frames=sum(s['native_source_frames'] for s in inventory['shots'].values())
    if summary['total_frames'] != expected_frames:
        raise ValueError('Exported native exposure coverage differs from source inventory')
    write(OUT/'qualification_summary.json', summary)
    print(line(dict(complete=True, qualified_rows=summary['qualified_rows'],
        remaining_pending_rows=summary['remaining_pending_rows'], output=str(OUT))), end='', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--replacement-prepared',type=Path,
        help='Current dataset to replace; frozen capture provenance remains bound to --prepared')
    args=parser.parse_args()
    BASE=args.source.resolve(); OUT=args.output.resolve()
    PREPARED=args.prepared.resolve()
    REPLACEMENT_PREPARED=args.replacement_prepared.resolve() if args.replacement_prepared else None
    from rinkspace.path_policy import guard_project_write
    OUT=guard_project_write(OUT)
    document=read(args.reviews)
    APPROVED=document['approved_components']
    INHERIT_SOURCE_SEEDS=document.get('include_inherited_seed_rows',True)
    if not isinstance(INHERIT_SOURCE_SEEDS,bool):
        raise ValueError('include_inherited_seed_rows must be a boolean')
    WITHHELD={(r['shot'],int(r['pts']),r['path_id'],*map(float,r['uv'])):r['reason']
        for r in document.get('withheld_source_uv',[])}
    if not isinstance(APPROVED,dict) or any(not isinstance(v,str) or not v.strip() for v in APPROVED.values()):
        raise ValueError('Every approved source component needs an explicit source-image conclusion')
    main()
