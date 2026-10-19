"""Evidence-linked event camera catalogue; image retrieval never asserts camera identity.

This module only writes the requested classification output. Original surveys and
geometric libraries are read-only inputs. Thresholds are versioned engineering
parameters, not project limits or an assumed number of cameras.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SOURCES = 'Rd66CneD9io 2U7RhoKchXA DXiFMHULDZ8 oxXu1avhrvs uaod0sM-9Hw mEoZsjFK4r0 M_2a8gDUyrQ yS-iLdqoHtg'.split()
VERSION = 'camera-classification-2026-09-21-v1'
PARAMETERS = {'sift_ratio': .72, 'reprojection_px_at_426': 2.5,
              'candidate_min_inliers': 12, 'candidate_min_x_span': .25,
              'appearance_rms_threshold': .105, 'background_height_fraction': .62,
              'rationale': 'Appearance distance is a retrieval bucket only. SIFT is confined to visible context; single-plane matches remain installation candidates. High overview support additionally requires named end-wall and grandstand regions and repeated observations.'}
# This CPU cache signature has the same descriptor/matching semantics. The only
# compatible change is the measured OpenCL BF backend (nearest indices checked).
CPU_EQUIVALENT_CACHE_SIGNATURE = '037ad39a9ffb0ba38a95fe803426373b194a885dd86c922b018bb82df8d18398'
HIGH = {'Rd66CneD9io': [3, 11], '2U7RhoKchXA': [9], 'DXiFMHULDZ8': [14],
        'oxXu1avhrvs': [15], 'uaod0sM-9Hw': [5], 'mEoZsjFK4r0': [4],
        'M_2a8gDUyrQ': [11], 'yS-iLdqoHtg': [6]}
NON_TARGET = {'2U7RhoKchXA': [8], 'oxXu1avhrvs': [7, 14],
              'uaod0sM-9Hw': [4], 'M_2a8gDUyrQ': [10]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def compact_observation(row):
    keys = ['source_id', 'image_path', 'image_sha256', 'source_pts', 'source_time_seconds',
            'time_base', 'source_path', 'source_sha256', 'media_track_id', 'selection_reason']
    return {k: row[k] for k in keys if k in row}


def feature(path, high=False):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = cv2.resize(im, (426, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    mask[:149] = 255
    # Broadcast graphics cannot establish venue/camera correspondence.
    mask[:57, :145] = 0
    if high:
        mask[:] = 255
        mask[215:, 330:] = 0
    detector = cv2.SIFT_create(nfeatures=600, contrastThreshold=.025)
    kp, desc = detector.detectAndCompute(gray, mask)
    points = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2)
    thumbnail = cv2.resize(im[:149], (16, 6), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    thumbnail[:2, :5] = np.mean(thumbnail[2:], axis=(0, 1))
    return {'points': points, 'desc': desc, 'appearance': thumbnail.ravel(),
            'gray_mean': float(gray.mean()), 'gray_std': float(gray.std())}


def match_background(query, reference):
    """Return measured matches, never a physical-camera decision."""
    if query is None or reference is None or query['desc'] is None or reference['desc'] is None:
        return None
    matches = descriptor_matches(query['desc'], reference['desc'])
    good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < PARAMETERS['sift_ratio'] * b.distance]
    if len(good) < 8:
        return None
    q = np.array([query['points'][m.queryIdx] for m in good], np.float32)
    r = np.array([reference['points'][m.trainIdx] for m in good], np.float32)
    h, keep = cv2.findHomography(q, r, cv2.RANSAC, PARAMETERS['reprojection_px_at_426'])
    if h is None or keep is None:
        return None
    keep = keep.ravel().astype(bool)
    q, r = q[keep], r[keep]
    if len(q) < 8:
        return None
    projected = cv2.perspectiveTransform(q[None], h)[0]
    error = np.linalg.norm(projected - r, axis=1)
    xspan = float(np.ptp(r[:, 0]) / 426)
    regions = {
        'end_wall_windows': int(np.sum((r[:, 0] < 128) & (r[:, 1] < 62))),
        'grandstand_structure': int(np.sum((r[:, 0] >= 128) & (r[:, 1] < 80))),
        'rink_surface_context': int(np.sum(r[:, 1] >= 80)),
    }
    return {'inliers': len(q), 'tentative_matches': len(good), 'inlier_fraction': round(len(q)/len(good), 4),
            'x_span': round(xspan, 4), 'median_error_px': round(float(np.median(error)), 4),
            'regions': regions, 'homography_query_to_reference': h.tolist(),
            'query_points_xy': q.round(2).tolist(), 'reference_points_xy': r.round(2).tolist(),
            'coordinate_basis': 'both images resized to 426x240; feature mask excludes usual top-left score overlay',
            'interpretation': 'background correspondence; homography alone does not determine a physical installation'}


def descriptor_matches(query, reference):
    """Use the installed OpenCL device; retain the exact CPU fallback."""
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            try:
                return cv2.BFMatcher().knnMatch(cv2.UMat(query), cv2.UMat(reference), k=2)
            except cv2.error:
                pass
    return cv2.BFMatcher().knnMatch(query, reference, k=2)


def high_region_support(match):
    return bool(match and match['inliers'] >= 20 and match['x_span'] >= .5
                and match['regions']['end_wall_windows'] >= 4
                and match['regions']['grandstand_structure'] >= 6
                and match['regions']['rink_surface_context'] >= 4)


def load_references(root):
    references = []
    for sid in SOURCES:
        path = root / 'outputs/event_survey_v11/all_source_review_v2' / f'{sid}_review.json'
        for row in read(path)['selected_views']:
            index = row['panel_index']
            kind = 'high_overview' if index in HIGH[sid] else 'non_target' if index in NON_TARGET.get(sid, []) else 'context_candidate'
            groups = row.get('candidate_library_group_ids', [])
            if sid == 'Rd66CneD9io' and kind == 'context_candidate' and index not in [15, 16]:
                groups = ['long_side_or_end_unresolved']
            references.append({
                'reference_id': f'{sid}:review:{index}', 'kind': kind,
                'observation': compact_observation(row), 'source_id': sid,
                'legacy_candidate_group_ids': groups, 'historical_record': str(path.relative_to(root).as_posix()),
                'panel_index': index, 'use': 'classification_reference_not_independent_check',
                'reviewer_kind': 'codex_model_visual_review',
                'visual_reason': ('高位斜俯视；左侧端墙窗格、对侧红白看台阶梯、围挡转角及冰面透视同时可见。'
                                  if kind == 'high_overview' else '外场宣传素材：看台、建筑或冰面标志与本赛事场馆不同。'
                                  if kind == 'non_target' else '同场馆背景检索参考；取景方向不等于安装位置，旧标签仅作候选。'),
                'opposing_evidence': [] if kind == 'high_overview' else ['人物、广告板同形和取景方向不足以排除附近另一安装位置。'],
                'features': feature(root / row['image_path'], high=kind == 'high_overview'),
            })
    return references


def observation_matches(root, observation, refs):
    path = root / observation['image_path']
    f = feature(path)
    if f is None:
        return {'image_path': observation['image_path'], 'status': 'media_missing', 'matches': []}
    # Every reviewed reference in the session is considered. No cross-session
    # identity is inherited just because the view resembles an earlier event day.
    matches = []
    high_features = None
    for reference in refs:
        if reference['kind'] == 'high_overview':
            if high_features is None:
                high_features = feature(path, high=True)
            m = match_background(high_features, reference['features'])
        else:
            m = match_background(f, reference['features'])
        if m and m['inliers'] >= PARAMETERS['candidate_min_inliers'] and m['x_span'] >= PARAMETERS['candidate_min_x_span']:
            m.update({'reference_id': reference['reference_id'], 'kind': reference['kind'],
                      'candidate_group_ids': reference['legacy_candidate_group_ids'],
                      'high_multiregion_support': reference['kind'] == 'high_overview' and high_region_support(m)})
            matches.append(m)
    matches.sort(key=lambda x: x['inliers'], reverse=True)
    return {'image_path': observation['image_path'], 'image_sha256': sha256(path),
            'source_pts': observation.get('source_pts'), 'source_time_seconds': observation.get('source_time_seconds'),
            'status': 'processed', 'matches': matches, 'appearance': f['appearance'].tolist(),
            'blank_candidate': f['gray_mean'] < 8 and f['gray_std'] < 8}


def classify_observations(source_id, observations, results, duration):
    """Conservative assignment: a candidate is not an affirmative installation."""
    usable = [r for r in results if r['status'] == 'processed']
    base = {'physical_camera_id': None, 'installation_id': None,
            'candidate_installation_ids': [], 'appearance_group_ids': [],
            'supporting_evidence': [], 'opposing_evidence': [],
            'motion_type': 'unknown', 'assignment_basis': 'direct_observations'}
    if not usable:
        return dict(base, status='unprocessed', reason='没有可读取观察图，无法执行背景证据判断。')
    high = [r for r in usable if any(m['high_multiregion_support'] for m in r['matches'])]
    other = [r for r in usable if any(m['kind'] == 'non_target' for m in r['matches'])]
    times = [r.get('source_time_seconds') for r in high if r.get('source_time_seconds') is not None]
    repeated = len(set(times)) >= 2 and max(times)-min(times) >= min(.5, duration*.3) if times else False
    if repeated and len(high) / len(usable) >= .8 and not other:
        base.update(status='supported', physical_camera_id=f'position:{source_id}:high_overview',
                    installation_id=f'install:{source_id}:high_overview', motion_type='fixed_supported',
                    reason='同镜头多时刻同时对应端墙窗格、红白看台结构及冰面/围挡上下文，支持该播出来源中的高位全景位置。实际拍摄场次未决；跨场同一位置另存候选关联。',
                    supporting_evidence=[{'image_path': r['image_path'], 'source_pts': r.get('source_pts'),
                                          'matches': [m for m in r['matches'] if m['high_multiregion_support']]} for r in high],
                    opposing_evidence=['未证明硬件相机身份；低清分辨率不能排除几乎重合的邻近安装。未匹配的观察帧仍可见，完整镜头质量需另检。'])
        return base
    if len(other) == len(usable) and (len(other) >= 2 or duration < 1):
        return dict(base, status='not_applicable', reason='全部可用观察对应已复核外场宣传素材；保留源时间轴。',
                    supporting_evidence=[{'image_path': r['image_path'], 'matches': [m for m in r['matches'] if m['kind'] == 'non_target']} for r in other])
    candidates = set()
    for r in usable:
        for m in r['matches']:
            if m['kind'] == 'high_overview':
                candidates.add(f'install:{source_id}:high_overview')
            elif m['kind'] == 'context_candidate':
                for g in m['candidate_group_ids']:
                    candidates.add(f'candidate:{source_id}:{g}')
    if candidates:
        base.update(status='multiple_candidates', candidate_installation_ids=sorted(candidates),
                    reason='静态背景匹配提供安装候选；主要为围挡或局部背景，尚未排除邻近机位/隐藏切镜，单候选也不升级为支持归属。',
                    supporting_evidence=[{'image_path': r['image_path'], 'source_pts': r.get('source_pts'), 'matches': r['matches']} for r in usable if r['matches']],
                    opposing_evidence=['单平面单应、相似构图与旧标签不能证明物理机位相同。', '未获多深度连续观察链排他证据，允许已有候选之外的未知安装。'])
        return base
    return dict(base, status='unknown', reason='已执行背景检索，当前参考无足够静态结构匹配；可能为特写、计分/采访、移动视角或未建参考的固定机位。',
                opposing_evidence=['没有足够证据时不把外观分组当作物理安装。'])


def group_appearances(rows, source_id):
    """Variable-count radius buckets for browsing only; never physical identities."""
    centers, groups = [], []
    for row in rows:
        if 'appearance' not in row:
            continue
        vector = np.asarray(row['appearance'], np.float32)
        if centers:
            distances = np.sqrt(np.mean((np.asarray(centers)-vector)**2, axis=1))
            which = int(np.argmin(distances))
        else:
            distances, which = [float('inf')], 0
        if not centers or distances[which] > PARAMETERS['appearance_rms_threshold']:
            which = len(centers)
            centers.append(vector)
            groups.append({'appearance_group_id': f'appearance:{source_id}:{which:04d}', 'source_id': source_id,
                           'meaning': 'appearance_retrieval_only_not_camera_count', 'representative_observations': [], 'image_paths': []})
        g = groups[which]
        g['image_paths'].append(row['image_path'])
        if not g['representative_observations']:
            g['representative_observations'].append({k: row[k] for k in ['image_path','source_pts','source_time_seconds'] if k in row})
        row['appearance_group_id'] = g['appearance_group_id']
    return groups


def build_image_evidence(root, observations_by_source, output, workers=4):
    cv2.setNumThreads(1)
    references = load_references(root)
    import inspect
    signature = hashlib.sha256((''.join(inspect.getsource(f) for f in
        (feature, match_background, high_region_support, observation_matches, load_references))
        + json.dumps([PARAMETERS,HIGH,NON_TARGET],sort_keys=True)).encode()).hexdigest()
    all_results, groups = {}, []
    for sid in SOURCES:
        unique = {o['image_path']: o for o in observations_by_source.get(sid, []) if o.get('image_path')}
        refs = [r for r in references if r['source_id'] == sid]
        cache_path = output / 'cache' / f'{sid}_background_matches.json'
        cached = read(cache_path) if cache_path.exists() else {}
        compatible = cached.get('implementation_sha256') in {signature,CPU_EQUIVALENT_CACHE_SIGNATURE}
        old = {r['image_path']: r for r in cached.get('rows', [])} if compatible else {}
        # Cache validates image bytes, so same path with changed content is invalidated.
        pending = [o for p,o in unique.items() if p not in old or not (root/p).exists() or old[p].get('image_sha256') != sha256(root/p)]
        print(f'{sid}: background observations={len(unique)} compute={len(pending)}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(lambda o: observation_matches(root,o,refs), pending):
                old[result['image_path']] = result
        results = [old[p] for p in unique]
        groups.extend(group_appearances(results,sid))
        write(cache_path, {'implementation_sha256': signature, 'parameters': PARAMETERS,
                          'backend':'OpenCL BF when available; SIFT CPU; compatible CPU cache retained', 'rows': list(old.values())})
        all_results.update({r['image_path']:r for r in results})
    serial_refs = [{k:v for k,v in r.items() if k != 'features'} for r in references]
    write(output/'reference_evidence.json', {'references':serial_refs, 'geometry_usage':'none', 'method_version':VERSION})
    return all_results, groups, references


def validate_catalog(catalog):
    failures, sources = [], []
    shot_ids=[s['shot_id'] for s in catalog['shots']]
    if len(shot_ids)!=len(set(shot_ids)):
        failures.append({'kind':'duplicate_shot_id'})
    known_sources={s['source_id'] for s in catalog['sources']}
    for s in catalog['shots']:
        if s['source_id'] not in known_sources:failures.append({'kind':'unknown_source','shot_id':s['shot_id']})
    for source in catalog['sources']:
        processing_scope = source.get('processing_scope')
        shots = [s for s in catalog['shots'] if s['source_id'] == source['source_id']]
        shots.sort(key=lambda s:s['start_pts'])
        if processing_scope:
            from .camera_scope import merge_intervals
            actual = merge_intervals([{'start_pts':s['start_pts'],'end_pts_exclusive':s['end_pts']} for s in shots])
            if actual != merge_intervals(processing_scope['target_intervals']):
                failures.append({'source_id':source['source_id'],'kind':'target_scope_coverage_mismatch'})
        elif not shots:failures.append({'source_id':source['source_id'],'kind':'no_segments'})
        for a,b in zip(shots,shots[1:]):
            if a['end_pts'] > b['start_pts'] or (not processing_scope and a['end_pts'] != b['start_pts']):
                failures.append({'source_id':source['source_id'],'kind':'gap_or_overlap','left':a['shot_id'],'right':b['shot_id']})
        if not processing_scope and shots and (abs(shots[0]['start_seconds']) > 1e-6 or abs(shots[-1]['end_seconds']-source['duration_seconds']) > .04):
            failures.append({'source_id':source['source_id'],'kind':'endpoints'})
        counts = Counter(s['classification']['status'] for s in shots)
        durations = defaultdict(float)
        for s in shots:
            durations[s['classification']['status']] += s['end_seconds']-s['start_seconds']
            if s['end_pts']<=s['start_pts'] or s['end_seconds']<=s['start_seconds']:
                failures.append({'shot_id':s['shot_id'],'kind':'empty_or_reversed_interval'})
            if 'start_frame_index' in s and s['end_frame_index_exclusive']-s['start_frame_index']!=s['frame_count']:
                failures.append({'shot_id':s['shot_id'],'kind':'frame_count_mismatch'})
            if s['classification']['status'] not in {'supported','multiple_candidates','unknown','not_applicable','unprocessed'}:
                failures.append({'shot_id':s['shot_id'],'kind':'invalid_classification_status'})
            if s['classification']['status'] == 'supported' and not s['classification']['supporting_evidence']:
                failures.append({'shot_id':s['shot_id'],'kind':'support_without_evidence'})
            if s['classification']['status']=='supported' and s.get('interval_type','shot')!='shot':
                failures.append({'shot_id':s['shot_id'],'kind':'uncertain_interval_assigned_one_installation'})
            if s.get('boundary_start',{}).get('status')=='source_start' and s.get('start_frame_index',0)!=0:
                failures.append({'shot_id':s['shot_id'],'kind':'interior_interval_labeled_source_start'})
            if s.get('boundary_end',{}).get('status')=='source_end' and s.get('end_frame_index_exclusive')!=source.get('frame_count'):
                failures.append({'shot_id':s['shot_id'],'kind':'interior_interval_labeled_source_end'})
            for o in s.get('observations',[]):
                if not s['start_pts']<=o['source_pts']<s['end_pts']:
                    failures.append({'shot_id':s['shot_id'],'kind':'observation_outside_interval','source_pts':o['source_pts']})
        expected_frames = processing_scope['target_frame_count'] if processing_scope else source.get('frame_count')
        if expected_frames is not None and sum(s.get('frame_count',0) for s in shots)!=expected_frames:
            failures.append({'source_id':source['source_id'],'kind':'source_frame_count_mismatch'})
        if not processing_scope and shots and 'start_frame_index' in shots[0]:
            if shots[0]['start_frame_index']!=0 or shots[-1]['end_frame_index_exclusive']!=source['frame_count']:
                failures.append({'source_id':source['source_id'],'kind':'source_frame_endpoints'})
            if any(a['end_frame_index_exclusive']!=b['start_frame_index'] for a,b in zip(shots,shots[1:])):
                failures.append({'source_id':source['source_id'],'kind':'frame_gap_or_overlap'})
        sources.append({'source_id':source['source_id'],'segment_count':len(shots),'status_counts':dict(counts),
                        'status_duration_seconds':{k:round(v,6) for k,v in durations.items()},
                        'frame_count':sum(s.get('frame_count',0) for s in shots)})
    return {'schema_version':1,'method_version':VERSION,'structural_failures':failures,'structural_pass':not failures,
            'sources':sources,'physical_accuracy_established':False,
            'meaning':'structural coverage and evidence presence are not visual classification accuracy',
            'visual_quality_review':'see visual_validation.json; never inferred from passing program tests'}


def resolve_interval_boundary(segment, boundaries, source, side):
    """A partition endpoint without a cut ID is not the start/end of the source."""
    if side not in {'start','end'}:raise ValueError(side)
    existing=segment.get(f'boundary_{side}',{})
    endpoint_pts=segment['start_pts'] if side=='start' else segment['end_pts_exclusive']
    if existing.get('status')==f'scope_{side}' and existing.get('source_pts')==endpoint_pts:
        return dict(existing)
    boundary_id=segment.get(f'{side}_boundary_id')
    if boundary_id in boundaries:
        return boundaries[boundary_id]
    is_start=side=='start'
    frame_index=segment['start_frame_index'] if is_start else segment['end_frame_index_exclusive']
    at_source_edge=frame_index==0 if is_start else frame_index==source['frame_count']
    pts=segment['start_pts'] if is_start else segment['end_pts_exclusive']
    seconds=segment['start_seconds'] if is_start else segment['end_seconds']
    scope_edge=segment.get(f'scope_{side}',False)
    result={'boundary_id':None,'status':f'scope_{side}' if scope_edge else f'source_{side}' if at_source_edge else f'explicit_interval_{side}',
        'source_id':segment['source_id'],'source_pts':pts,'source_seconds':seconds,
        'source_time_seconds':seconds,'time_base':source['track']['time_base'],'frame_index':frame_index,
        'interval_type':segment.get('interval_type','shot'),'exact_cut_claim':False,
        'meaning':'target-rink scope endpoint, not an asserted edit frame' if scope_edge else 'source frame-allocation endpoint, not an asserted edit frame',
        'pts_semantics':'first_included_frame' if is_start else 'exclusive_end'}
    if segment.get('interval_type','shot')!='shot':
        result['uncertainty_interval_seconds']=[segment['start_seconds'],segment['end_seconds']]
        result['candidate_points']=segment.get('candidate_points',[])
    return result


def build_catalog(root, output, workers=4):
    root, output = Path(root).resolve(), Path(output).resolve()
    from .camera_scope import load_source_scope, intersections, clip_shot
    scopes = {sid:load_source_scope(root,sid,output) for sid in SOURCES}
    timelines = {}
    for sid in SOURCES:
        path=output/'rink_timelines'/f'{sid}.json'
        if not path.exists():path=output/'timelines'/f'{sid}.json'
        t=read(path);scope=scopes[sid]
        if t['track']['source_sha256']!=scope['source_sha256']:
            raise ValueError(f'{sid}: scene scope belongs to another source track')
        kept=[]
        for segment in t['segments']:
            segment.setdefault('end_pts',segment['end_pts_exclusive'])
            for a,b in intersections(segment['start_pts'],segment['end_pts_exclusive'],scope['target_intervals']):
                kept.append(clip_shot(segment,a,b,scope))
        t['segments']=kept
        t['_active_path']=path
        timelines[sid]=t
    registry = {s['source_id']:s for s in read(root/'configs/rink_model/source_registry.json')['sources']}
    observations = {}
    for sid,t in timelines.items():
        observations[sid] = []
        for segment in t['segments']:
            for o in segment['observations']:
                o['source_time_seconds'] = o.get('source_time_seconds',o.get('source_seconds'))
                observations[sid].append(o)
    results, groups, refs = build_image_evidence(root,observations,output,workers)
    sources, shots, installations = [], [], {}
    for sid,t in timelines.items():
        track = t['track']
        track['media_track_id'] = f'{sid}:f{track["format_id"]}:{track["source_sha256"][:12]}'
        source = {'source_id':sid,'title':t['title'],'session_id':registry[sid]['session_id'],
                  'duration_seconds':t['duration_seconds'],'track':track,
                  'frame_count':t['frame_count'],'shot_count':len(t['segments']),
                  'timeline_path':t['_active_path'].relative_to(root).as_posix(),
                  'timeline_sha256':sha256(t['_active_path']),
                  'processing_scope':{k:v for k,v in scopes[sid].items() if k!='decisions'},
                  'timeline_validation':t['timeline_validation'], 'chunk_audit':t['chunk_audit']}
        sources.append(source)
        boundaries = {b['boundary_id']:dict(b) for b in t['boundaries']}
        for b in boundaries.values():
            b['source_time_seconds'] = b.get('source_seconds')
            b['observations'] = b.get('neighbour_frames',[])
            for o in b['observations']:
                o['source_time_seconds'] = o.get('source_seconds')
            if b.get('uncertainty_interval_seconds'):
                b['uncertainty_start_seconds'],b['uncertainty_end_seconds']=b['uncertainty_interval_seconds']
        for segment in t['segments']:
            obs = segment['observations']
            rows = [results[o['image_path']] for o in obs if o.get('image_path') in results]
            duration=segment['end_seconds']-segment['start_seconds']
            assignment = classify_observations(sid,obs,rows,duration)
            if segment.get('interval_type','shot') != 'shot':
                assignment.update(status='unknown',physical_camera_id=None,installation_id=None,
                                  reason='该区间是转场或边界未决范围，内部可能包含多个拍摄画面，不作单一物理安装归属。')
            assignment['appearance_group_ids'] = sorted({r['appearance_group_id'] for r in rows if 'appearance_group_id' in r})
            normalized = []
            for o in obs:
                r=results.get(o['image_path'],{})
                normalized.append(dict(compact_observation(o), origin=o.get('origin'),
                    image_sha256=r.get('image_sha256'), media_track_id=track['media_track_id'],
                    appearance_group_id=r.get('appearance_group_id'),
                    background_match_cache=f'outputs/camera_classification/cache/{sid}_background_matches.json'))
            shot = {k:v for k,v in segment.items() if k not in ['observations','classification_status']}
            shot.update(end_pts=segment['end_pts_exclusive'], media_track_id=track['media_track_id'],
                observations=normalized, classification=assignment,
                boundary_start=resolve_interval_boundary(segment,boundaries,source,'start'),
                boundary_end=resolve_interval_boundary(segment,boundaries,source,'end'),
                content={'kind':'non_target_material' if assignment['status']=='not_applicable' else 'unresolved_content',
                         'venue_status':'same_venue_supported' if assignment['status']=='supported' else 'unresolved',
                         'replay_status':'unresolved','capture_session_id':None,
                         'broadcast_session_id':registry[sid]['session_id'],
                         'reason':'播出场次已知；画面拍摄场次/回放身份需内容对应，不能由播出时间自动赋值。'},
                duplicate_links=[],historical_links=[{'legacy_shot_id':x,'meaning':'historical survey candidate identifier, not final shot'} for x in segment.get('legacy_shot_ids',[])])
            shots.append(shot)
            ids=assignment['candidate_installation_ids']+([assignment['installation_id']] if assignment['installation_id'] else [])
            for iid in ids:
                if iid not in installations:
                    installations[iid]={'installation_id':iid,'source_id':sid,'session_id':registry[sid]['session_id'],
                        'physical_camera_id':f'position:{sid}:high_overview' if iid.endswith(':high_overview') else None,
                        'label':('高位斜俯全景' if iid.endswith(':high_overview') else '安装候选 '+iid.rsplit(':',1)[-1]),
                        'status':'supported_in_broadcast_scope' if assignment['installation_id']==iid else 'candidate',
                        'shot_ids':[],'supported_shot_ids':[],'candidate_shot_ids':[], 'representative_observations':[],
                        'supporting_evidence':[], 'opposing_evidence':['未验证的回放可能属于别的拍摄场次；来源内版本仅以播出来源为作用域。'],
                        'cross_session_relations':[], 'capture_session_status':'not_assumed_from_broadcast_session'}
                item=installations[iid]
                item['shot_ids'].append(shot['shot_id'])
                item['supported_shot_ids' if assignment['installation_id']==iid else 'candidate_shot_ids'].append(shot['shot_id'])
                if assignment['installation_id']==iid:
                    item['status']='supported_in_broadcast_scope'
                if not item['representative_observations']:
                    reference_ids={m['reference_id'] for r in rows for m in r['matches']
                                   if (m['kind']=='high_overview')==iid.endswith(':high_overview')}
                    item['representative_observations']=[r['observation'] for r in refs if r['reference_id'] in reference_ids]
                    item['supporting_evidence']=[{'reference_id':r['reference_id'],'reason':r['visual_reason']} for r in refs if r['reference_id'] in reference_ids]
    # Separate session installation records; proposed correspondence never collapses them.
    highs=[i for i in installations.values() if i['installation_id'].endswith(':high_overview')]
    for item in highs:
        item['cross_session_relations']=[{'installation_id':j['installation_id'],'relation':'corresponding_position_candidate',
            'reason':'均为相同角落斜俯视，端墙与红白看台排列一致；跨场安装位移仍需独立排他验证。'} for j in highs if j is not item]
    physical=[{'physical_camera_id':i['physical_camera_id'],'label':i['label']+' / '+registry[i['source_id']]['session_id'],
        'identity_status':i['status'],'motion_type':'fixed_supported' if i['supported_shot_ids'] else 'fixed_candidate',
               'installation_ids':[i['installation_id']],'representative_observations':i['representative_observations'],
               'meaning':'source-scoped position identity; cross-session candidates are not counted as distinct hardware'} for i in highs]
    for group in groups:
        gid=group['appearance_group_id']
        group['shot_ids']=[s['shot_id'] for s in shots if gid in s['classification']['appearance_group_ids']]
        group.pop('image_paths',None)
    catalog={'schema_version':'rinkspace.camera-classification.v1','method_version':VERSION,
        'implementation_sha256':sha256(Path(__file__)), 'parameters':PARAMETERS,
        'sources':sources,'media_tracks':[s['track'] for s in sources], 'shots':shots,
        'installations':list(installations.values()),'physical_cameras':physical,'appearance_groups':groups,
        'evidence_files':['outputs/camera_classification/reference_evidence.json'],
        'validation_path':'outputs/camera_classification/validation.json',
        'scope':{'timeline_processing_complete':False,'all_physical_cameras_confirmed':False,'target_rink_only':True,
                 'geometry_or_grid_work_performed':False,
                 'women_fs_usage':'classification catalogue/reference/check purposes recorded separately; no geometric model update'},
        'limitations':['机器支持的边界不是全场视觉确认；未决范围保留邻帧和原因。',
                       '同播出来源内的安装版本不证明回放拍摄场次；未对应回放仍显式未决。',
                       '外观组数量、安装候选数量均不是实际相机数量。']}
    integrate_evidence(catalog,root,output)
    scene_scope=read(output/'rink_scope.json')
    catalog['processing_scope']={'schema_version':scene_scope['schema_version'],'summary':scene_scope['summary'],
                                 'scene_screening_complete':scene_scope['scene_screening_complete']}
    synchronize_installation_membership(catalog)
    validation=validate_catalog(catalog)
    counts=Counter(s['classification']['status'] for s in shots)
    catalog['summary']={'source_count':len(sources),'segment_count':len(shots),'frame_count':sum(s['frame_count'] for s in shots),
        'duration_seconds':sum(s['end_seconds']-s['start_seconds'] for s in shots),'status_counts':dict(counts),
        'status_duration_seconds':{k:round(sum(s['end_seconds']-s['start_seconds'] for s in shots if s['classification']['status']==k),6) for k in counts},
        'boundary_status_counts':dict(Counter(b['status'] for t in timelines.values() for b in t['boundaries'])),
        'interval_type_counts':dict(Counter(s.get('interval_type','shot') for s in shots)),
        'boundary_unresolved_duration_seconds':round(sum(s['end_seconds']-s['start_seconds'] for s in shots if s.get('interval_type','shot')!='shot'),6),
        'appearance_group_count':len(groups),'installation_record_count':len(catalog['installations']),'actual_physical_camera_count':None,
        'human_confirmed_observation_count':len(catalog.get('confirmed_view_assignments',[]))}
    validation['timeline_validation']={sid:t['timeline_validation'] for sid,t in timelines.items()}
    visual=collect_visual_validation(catalog,root,output)
    write(output/'visual_validation.json',visual)
    validation['visual_validation_path']=(output/'visual_validation.json').relative_to(root).as_posix()
    validation['visual_review_summary']=visual['summary']
    for filename in ['timelines/integrity_validation.json','timelines/development_reviews/final_regression.json',
                     'evidence/evidence_integrity_audit.json','evidence/context_validation.json']:
        path=output/filename
        if path.exists():
            catalog['evidence_files'].append(path.relative_to(root).as_posix())
            validation[filename.replace('/','_').replace('.json','')]=read(path)
    write(output/'catalog.json',catalog)
    write(output/'validation.json',validation)
    print(json.dumps(catalog['summary'],ensure_ascii=False),flush=True)
    return catalog


def refresh_catalog_metadata(root, output):
    """Refresh saved boundary metadata and checks without decoding or matching."""
    root,output=Path(root).resolve(),Path(output).resolve()
    catalog=read(output/'catalog.json')
    old_hash=sha256(output/'catalog.json')
    if (output/'human_evidence.json').exists():
        catalog['human_evidence']=read(output/'human_evidence.json')
    apply_physical_identity_decisions(catalog)
    sources={s['source_id']:s for s in catalog['sources']}
    updates=[]
    for shot in catalog['shots']:
        source=sources[shot['source_id']]
        for side in ['start','end']:
            key=f'boundary_{side}'
            if shot.get(f'{side}_boundary_id') is not None:continue
            corrected=resolve_interval_boundary(shot,{},source,side)
            if shot.get(key)!=corrected:
                updates.append({'shot_id':shot['shot_id'],'field':key,
                    'previous_status':shot.get(key,{}).get('status'),'status':corrected['status']})
                shot[key]=corrected
    catalog['metadata_implementation_sha256']=sha256(Path(__file__))
    validation=read(output/'validation.json')
    validation.update(validate_catalog(catalog))
    validation['metadata_refresh']={'previous_catalog_sha256':old_hash,'changed_endpoint_count':len(updates),
        'decode_or_feature_computation_performed':False,'partition_or_assignment_status_changed':False,
        'physical_identity_decisions_applied':catalog.get('applied_physical_identity_decision_ids',[]),
        'endpoint_status_counts':dict(Counter(b['status'] for s in catalog['shots'] for b in [s['boundary_start'],s['boundary_end']]))}
    visual=collect_visual_validation(catalog,root,output)
    write(output/'visual_validation.json',visual)
    validation['visual_review_summary']=visual['summary']
    write(output/'catalog.json',catalog)
    validation['catalog_sha256']=sha256(output/'catalog.json')
    write(output/'validation.json',validation)
    write(output/'evidence/catalog_endpoint_refresh.json',{'previous_catalog_sha256':old_hash,
        'catalog_sha256':sha256(output/'catalog.json'),'updates':updates,'validation_pass':validation['structural_pass']})
    return validation


def integrate_evidence(catalog, root, output):
    """Attach actual human decisions, historical evidence and content mappings."""
    shots=catalog['shots']
    for filename in ['legacy_links.json','high_corner_multisession.json','evidence_use.json']:
        p=output/'evidence'/filename
        if p.exists():catalog['evidence_files'].append(p.relative_to(root).as_posix())
    for item in catalog['installations']:
        item['broadcast_scope_source_id']=item['source_id']
        item['capture_session_id']=None
        item['cross_session_evidence_path']='outputs/camera_classification/evidence/high_corner_multisession.json'
    human_path=output/'human_evidence.json'
    catalog['confirmed_view_assignments']=[]
    if human_path.exists():
        human=read(human_path)
        catalog['human_evidence']=human
        catalog['evidence_files'].append(human_path.relative_to(root).as_posix())
        for case in human['cases']:
            if case.get('status')!='human_confirmed':continue
            iid,pid=case['installation_id'],case['physical_camera_id']
            related=[];supported=[]
            for observation in case['observations']:
                sid,p=observation['source_id'],observation['source_pts']
                containing=next((s for s in shots if s['source_id']==sid and s['start_pts']<=p<s['end_pts']),None)
                if containing is None:continue
                related.append(containing['shot_id'])
                direct={'source_id':sid,'source_pts':p,'time_base':observation['time_base'],
                    'image_path':observation['image_path'],'physical_camera_id':pid,'installation_id':iid,
                    'status':'human_confirmed','evidence_path':human_path.relative_to(root).as_posix(),
                    'case_id':case['case_id'],'automatic_match_required':False,
                    'scope':'the displayed observation, not all members of any appearance group'}
                catalog['confirmed_view_assignments'].append(direct)
                c=containing['classification']
                c['human_confirmed_observations']=c.get('human_confirmed_observations',[])+[direct]
                if iid not in c['candidate_installation_ids']:c['candidate_installation_ids'].append(iid)
                c['supporting_evidence'].append(direct)
                if c['status']=='unknown':
                    c.update(status='multiple_candidates',reason='本段包含已获人工确认的观察帧；完整区间的连续性与边界另行核查，尚不整体传播。')
                # Entire-range visual context was separately inspected; graphics
                # and dissolution endpoints are excluded from continuity transfer.
                spans=[(2166.866666666667,2181.7),(9429.0,9442.766666666666)]
                continuity=next(((a,b) for a,b in spans if a-.001<=containing['start_seconds'] and containing['end_seconds']<=b+.001),None)
                if containing.get('interval_type','shot')=='shot' and continuity:
                    c.update(status='supported',physical_camera_id=pid,installation_id=iid,
                        motion_type='fixed_human_confirmed',assignment_basis='human_observation_plus_reviewed_within_shot_continuity',
                        reason='用户明确确认A/B同一固定安装。此段在已查看的完整上下文采样序列中连续摇摄；仅沿本段传播，不扩展整外观组。')
                    c['supporting_evidence'].append({'kind':'within_shot_visual_continuity',
                        'reviewer':'root_model','scope_seconds':continuity,
                        'context_paths':case.get('context_clips',[]),
                        'method':'one-second full-range sequence inspected; suspected 9437 change additionally checked at four-frame intervals by timeline reviewer'})
                    supported.append(containing['shot_id'])
            catalog['installations'].append({'installation_id':iid,'physical_camera_id':pid,'source_id':case['observations'][0]['source_id'],
                'session_id':'men_fs','broadcast_scope_source_id':'2U7RhoKchXA','capture_session_id':None,
                'label':'人工确认 A/B 同一固定机位','status':'human_confirmed_observations',
                'shot_ids':related,'supported_shot_ids':supported,'candidate_shot_ids':[s for s in related if s not in supported],
                'representative_observations':case['observations'],'supporting_evidence':[case],
                'opposing_evidence':['人工确认范围为两张图；其它镜头不由此自动确认。'],'cross_session_relations':[]})
            catalog['physical_cameras'].append({'physical_camera_id':pid,'label':'人工确认 A/B 同一固定机位',
                'identity_status':'human_confirmed_observations','motion_type':'fixed_human_confirmed',
                'installation_ids':[iid],'representative_observations':case['observations']})
    index_path=output/'evidence/duplicate_mappings.json'
    if index_path.exists():
        content=read(index_path)
        catalog['content_links']=content
        catalog['evidence_files'].append(index_path.relative_to(root).as_posix())
        for mapping in content['mappings']:
            data=read(root/mapping['path']); a,b=mapping['broadcast_interval_s']
            for shot in shots:
                if shot['source_id']!=mapping['broadcast_source_id'] or shot['start_seconds']>b or shot['end_seconds']<=a:continue
                anchors=[r for r in data['anchors'] if shot['start_seconds']<=r['target_time_s']<shot['end_seconds']]
                shot['duplicate_links'].append({'group_id':mapping['group_id'],'relation':'same_recording_upload_sampled_mapping',
                    'programme_source_id':mapping['programme_source_id'],'evidence_path':mapping['path'],
                    'overlap_seconds':[max(a,shot['start_seconds']),min(b,shot['end_seconds'])],
                    'verified_anchors':[r for r in anchors if r['status']=='pixel_duplicate_anchor'],
                    'unresolved_anchor_count':sum(r['status']!='pixel_duplicate_anchor' for r in anchors),
                    'full_interval_verified':False,'independent_evidence':False})
        cetin=read(root/content['mappings'][0]['path'])
        replay_path=output/'evidence/cetin_replay_mapping.json'
        if replay_path.exists():
            replay=read(replay_path)
            for segment in replay['segments']:
                lo,hi=segment['replay_interval_s']
                anchors=[a for a in cetin['anchors'] if lo<=a['query_time_s']<=hi and a['status']=='pixel_duplicate_anchor']
                if not anchors:continue
                a,b=min(x['target_time_s'] for x in anchors),max(x['target_time_s'] for x in anchors)
                for shot in shots:
                    if shot['source_id']=='Rd66CneD9io' and shot['start_seconds']<=b and shot['end_seconds']>a:
                        shot['content']['replay_status']='sampled_same_image_replay_supported'
                        shot['duplicate_links'].append({'group_id':'cetin_internal_replay','relation':'sampled_slow_or_normal_speed_replay',
                            'evidence_path':replay_path.relative_to(root).as_posix(),'programme_replay_segment':segment,
                            'broadcast_verified_anchors':anchors,'independent_evidence':False,'full_interval_verified':False})
    for path in (output/'evidence').glob('visual_review_*.json'):
        review=read(path)
        if not isinstance(review,dict) or 'cases' not in review:continue
        catalog['evidence_files'].append(path.relative_to(root).as_posix())
        for case in review.get('cases',[]):
            for shot in shots:
                if shot['source_id']==case['source_id'] and shot['start_seconds']>=case['start_seconds']-.001 and shot['end_seconds']<=case['end_seconds']+.001:
                    kind=case.get('content_kind',case.get('content_result'))
                    if kind:shot['content']['kind']=kind
                    shot.setdefault('visual_reviews',[]).append({'case_id':case['case_id'],'evidence_path':path.relative_to(root).as_posix(),
                        'finding':case.get('finding',case.get('findings')),
                        'usage':case.get('usage_after_review',case.get('classification_use'))})
                    if kind and any(x in kind for x in ['remote_interview','non_target']):
                        shot['classification'].update(status='not_applicable',physical_camera_id=None,installation_id=None,
                            reason='完整区间采样序列复核为远程访谈/非本场馆素材，保留时间轴。')
                    if 'replay' in str(kind):shot['content']['replay_status']='probable_visual_transition_evidence'
    if not catalog.get('scope',{}).get('target_rink_only'):
        integrate_context_evidence(catalog, root, output)
    link_human_reference_candidates(catalog,root,output)
    apply_physical_identity_decisions(catalog)


def integrate_context_evidence(catalog, root, output):
    path=output/'evidence/context_assignments.json'
    if not path.exists():return
    data=read(path)
    catalog['evidence_files'] += [path.relative_to(root).as_posix(),data['references_path']]
    for name in ['context_validation.json','context_visual_spotcheck_final.json']:
        evidence_path=output/'evidence'/name
        if evidence_path.exists():catalog['evidence_files'].append(evidence_path.relative_to(root).as_posix())
    index={(o['source_id'],o['source_pts']):o for o in data['observations']}
    items={}
    for shot in catalog['shots']:
        if shot.get('interval_type','shot')!='shot':continue
        matched=[index[(shot['source_id'],o['source_pts'])] for o in shot['observations'] if (shot['source_id'],o['source_pts']) in index]
        if not matched:continue
        sid=shot['source_id'];iid=f'candidate:{sid}:kiss_and_cry';pid=f'position:{sid}:kiss_and_cry'
        c=shot['classification']
        matched_pts={m['source_pts'] for m in matched}
        unmatched=[o for o in shot['observations'] if o['source_pts'] not in matched_pts]
        unresolved_boundary=any(str(shot.get(key,{}).get('status','')).startswith('unresolved')
                                for key in ['boundary_start','boundary_end'])
        content_conflict=any(word in shot.get('content',{}).get('kind','')
                             for word in ['non_target','remote_interview','graphic_transition'])
        c['supporting_evidence'].append({'kind':'static_context_region_matches','observations':matched,
            'unmatched_observations':unmatched,'unresolved_boundary':unresolved_boundary,
            'content_conflict':content_conflict,
            'reference_path':data['references_path'],'evidence_path':path.relative_to(root).as_posix()})
        if iid not in c['candidate_installation_ids']:c['candidate_installation_ids'].append(iid)
        ratio=len(matched)/len(shot['observations'])
        if ratio>=.8 and not content_conflict:
            shot['content']['kind']='off_rink_arena_kiss_and_cry'
        eligible=[m for m in matched if m.get('position_support_eligible')]
        times=[m['source_time_seconds'] for m in eligible]
        support=len(set(times))>=2 and max(times)-min(times)>=min(.5,(shot['end_seconds']-shot['start_seconds'])*.3) and ratio>=.8 if times else False
        support=support and not unresolved_boundary and not content_conflict
        if unresolved_boundary or content_conflict or unmatched:
            c['opposing_evidence'].append({'kind':'context_assignment_scope',
                'unresolved_boundary':unresolved_boundary,'content_conflict':content_conflict,
                'unmatched_observation_count':len(unmatched),
                'meaning':'Only matched observations establish visible context; unresolved boundaries/content conflicts prevent whole-shot installation support.'})
        if c['status'] not in ['not_applicable','supported']:
            if support:
                c.update(status='supported',physical_camera_id=pid,installation_id=iid,
                    reason='同播出来源参考与本段多时刻同时对应后幕、金色椅轮廓和花饰，支持等分区固定位置；拍摄场次及相邻重架设仍未确定。',
                    assignment_basis='multiple_observations_of_named_static_regions',motion_type='fixed_supported')
            else:
                c.update(status='multiple_candidates',reason='等分区静态布景提供场内安装候选；多深度或多时刻证据不足，不把同布景或跨来源匹配直接当同机位。')
        item=items.setdefault(iid,{'installation_id':iid,'physical_camera_id':pid,'source_id':sid,
            'broadcast_scope_source_id':sid,'capture_session_id':None,'label':'等分区固定构图候选',
            'status':'candidate','shot_ids':[],'supported_shot_ids':[],'candidate_shot_ids':[],
            'representative_observations':[],'supporting_evidence':[{'evidence_path':path.relative_to(root).as_posix(),'reference_path':data['references_path']}],
            'opposing_evidence':['相同布景不足以排除相邻或重新架设的机位；实际拍摄场次不从播出来源推定。']})
        item['shot_ids'].append(shot['shot_id'])
        item['supported_shot_ids' if c.get('installation_id')==iid else 'candidate_shot_ids'].append(shot['shot_id'])
        if c.get('installation_id')==iid:item['status']='supported_in_broadcast_scope'
        if not item['representative_observations']:
            item['representative_observations']=[compact_observation(matched[0])]
    for item in items.values():
        item['cross_session_relations']=[{'installation_id':other['installation_id'],'relation':'corresponding_position_candidate',
            'reason':'布景对应已测量；不同拍摄场次的安装关系未确认。'} for other in items.values() if other is not item]
        catalog['installations'].append(item)
        catalog['physical_cameras'].append({'physical_camera_id':item['physical_camera_id'],'label':item['label']+' / '+item['source_id'],
            'identity_status':item['status'],'motion_type':'fixed_supported' if item['supported_shot_ids'] else 'unknown',
            'installation_ids':[item['installation_id']],'representative_observations':item['representative_observations']})


def synchronize_installation_membership(catalog):
    """Build reverse links after human/context/content decisions, not before."""
    installations={item['installation_id']:item for item in catalog['installations']}
    for item in installations.values():
        item.update(shot_ids=[],supported_shot_ids=[],candidate_shot_ids=[])
    for shot in catalog['shots']:
        c=shot['classification']
        ids=set(c['candidate_installation_ids'])
        if c.get('installation_id'):ids.add(c['installation_id'])
        for iid in ids:
            if iid not in installations:continue
            item=installations[iid]
            item['shot_ids'].append(shot['shot_id'])
            supported=c['status']=='supported' and c.get('installation_id')==iid
            item['supported_shot_ids' if supported else 'candidate_shot_ids'].append(shot['shot_id'])
    for item in installations.values():
        if item.get('status')!='human_confirmed_observations':
            item['status']='supported_in_broadcast_scope' if item['supported_shot_ids'] else 'candidate'
    for physical in catalog['physical_cameras']:
        if physical.get('identity_status') in {'human_confirmed_observations','human_confirmed_position_group'}:continue
        supported=any(installations[i]['supported_shot_ids'] for i in physical['installation_ids'])
        physical['identity_status']='supported_in_broadcast_scope' if supported else 'candidate'
        physical['motion_type']='fixed_supported' if supported else 'fixed_candidate'


def collect_visual_validation(catalog, root, output):
    """Relink real visual checks to the final partition without claiming new review."""
    cases=[];reports=[]
    for path in sorted((output/'evidence').glob('visual_review_*.json')):
        report=read(path)
        if not isinstance(report,dict) or 'cases' not in report:continue
        reports.append({'path':path.relative_to(root).as_posix(),'sha256':sha256(path)})
        for case in report.get('cases',[]):
            lo,hi=case['start_seconds'],case['end_seconds']
            current=[s for s in catalog['shots'] if s['source_id']==case['source_id']
                     and s['start_seconds']<hi and s['end_seconds']>lo]
            cases.append({'case_id':case['case_id'],'source_id':case['source_id'],
                'source_interval_seconds':[lo,hi],'review_path':path.relative_to(root).as_posix(),
                'clip_path':case['clip_path'],'source_pts_mapping':case['source_pts_mapping'],
                'pages_read':len(case.get('strips',[])),
                'source_pts_samples':case.get('strip_sample_pts',[]),
                'current_fully_contained_segment_ids':[s['shot_id'] for s in current
                    if s['start_seconds']>=lo-.001 and s['end_seconds']<=hi+.001],
                'current_partially_overlapping_segment_ids':[s['shot_id'] for s in current
                    if s['start_seconds']<lo-.001 or s['end_seconds']>hi+.001],
                'current_classification_counts':dict(Counter(s['classification']['status'] for s in current)),
                'findings':case.get('findings',case.get('finding')),
                'usage':case.get('usage_after_review',case.get('classification_use')),
                'reference_overlap':case.get('reference_overlap',[]),
                'all_frames_visually_watched':False,
                'duplicate_group_independence':'No overlap with verified programme/replay groups in evidence_integrity_audit; undiscovered repeated content remains possible.',
                'statistical_independent_accuracy_claim':False})
    return {'schema_version':'rinkspace.camera-visual-validation.v1',
        'summary':{'complete_selected_ranges_reviewed':len(cases),
            'sources_reviewed':sorted({c['source_id'] for c in cases}),
            'range_duration_seconds':round(sum(c['source_interval_seconds'][1]-c['source_interval_seconds'][0] for c in cases),6),
            'contact_sheet_pages_read':sum(c['pages_read'] for c in cases),
            'all_source_frames_visually_watched':False,'global_physical_classification_accuracy_established':False,
            'method':'All listed pages covering complete selected provisional shot ranges: one-second samples plus candidate-boundary neighbours; dense sub-second windows where needed. Current segment links are computed, not an additional visual inspection.',
            'independence':'Ranges used for references or corrections are development evidence. Untouched reserved ranges are separately labelled. Known duplicate groups were excluded; complete repeated-content independence is unresolved.'},
        'reports':reports,'cases':cases,
        'boundary_correction_evidence':'outputs/camera_classification/timelines/development_reviews/boundary_reviews.json',
        'context_false_match_correction':'outputs/camera_classification/evidence/context_visual_spotcheck_final.json',
        'limitations':['Sampling may miss short cuts between samples.',
            'Whole-timeline processing is complete; remaining physical identities and uncertain boundaries remain unresolved.',
            'Program tests and frame allocation do not measure visual accuracy.']}


def link_human_reference_candidates(catalog, root, output):
    """Use confirmed seeds for retrieval; new views do not inherit confirmation."""
    references_path=output/'reference_evidence.json'
    if not references_path.exists():return
    refs=read(references_path)['references']
    for case in catalog.get('human_evidence',{}).get('cases',[]):
        if case.get('status')!='human_confirmed':continue
        anchor_keys={(o['source_id'],o['source_pts']) for o in case['observations']}
        ref_ids={r['reference_id'] for r in refs if
                 (r['source_id'],r['observation'].get('source_pts')) in anchor_keys}
        iid=case['installation_id']
        for shot in catalog['shots']:
            c=shot['classification']
            if c['status'] not in ['unknown','multiple_candidates'] or shot.get('interval_type','shot')!='shot':continue
            hits=[]
            for evidence in c['supporting_evidence']:
                matches=[m for m in evidence.get('matches',[]) if m.get('reference_id') in ref_ids]
                if matches:hits.append({'source_pts':evidence.get('source_pts'),
                    'image_path':evidence.get('image_path'),'reference_matches':matches})
            if not hits:continue
            if iid not in c['candidate_installation_ids']:c['candidate_installation_ids'].append(iid)
            c['status']='multiple_candidates'
            c['supporting_evidence'].append({'kind':'human_reference_retrieval',
                'case_id':case['case_id'],'candidate_installation_id':iid,'matches_to_confirmed_observations':hits,
                'meaning':'The two seed observations are directly human-confirmed. These additional views have only background retrieval evidence; no automatic propagation of human confirmation.'})
            c['opposing_evidence'].append('与人工确认图的局部背景相符，仍须排除相邻机位与隐藏切镜；该候选不是新的人工确认。')


def apply_physical_identity_decisions(catalog):
    """Adopt user position-group decisions without promoting candidate shots."""
    applied=catalog.setdefault('applied_physical_identity_decision_ids',[])
    for decision in catalog.get('human_evidence',{}).get('physical_identity_decisions',[]):
        if decision.get('status')!='human_confirmed':continue
        if decision['decision_id'] in applied:continue
        selected=set(decision['installation_ids'])
        installations=[i for i in catalog['installations'] if i['installation_id'] in selected]
        available={i['installation_id'] for i in installations}
        if available!=selected:
            scoped=bool(catalog.get('processing_scope') or catalog.get('scope',{}).get('target_rink_only')
                        or any(s.get('processing_scope') for s in catalog.get('sources',[])))
            if not scoped:
                raise ValueError('Human position decision references an unavailable installation')
            catalog.setdefault('physical_identity_decision_scope',{})[decision['decision_id']]={
                'available_installation_ids':sorted(available),
                'unavailable_in_current_scope_installation_ids':sorted(selected-available),
                'status':'applied_to_current_scope' if available else 'outside_current_scope',
                'original_decision_retained_in':'human_evidence.physical_identity_decisions'}
            if not installations:continue
        pid=decision['physical_camera_id']
        previous=[p for p in catalog['physical_cameras'] if set(p['installation_ids']).intersection(selected)]
        if any(not set(p['installation_ids']).issubset(selected) for p in previous):
            raise ValueError('Partial identity-group merge requires an explicit split')
        previous_ids={p['physical_camera_id'] for p in previous}
        catalog.setdefault('historical_physical_camera_records',[]).extend(previous)
        link={'decision_id':decision['decision_id'],'kind':'user_confirmed_position_identity',
              'evidence_path':'outputs/camera_classification/human_evidence.json',
              'scope':decision['scope'],'automatic_match_required':False}
        observations=[]
        for item in installations:
            item.setdefault('historical_physical_camera_ids',[]).append(item['physical_camera_id'])
            item['physical_camera_id']=pid
            item['physical_identity_evidence']=link
            item['historical_cross_session_relations']=item.get('cross_session_relations',[])
            item['cross_session_relations']=[{'installation_id':other['installation_id'],
                'relation':'same_position_user_confirmed','evidence':link}
                for other in installations if other is not item]
            observations.extend(item.get('representative_observations',[]))
        for shot in catalog['shots']:
            c=shot['classification']
            if c.get('physical_camera_id') in previous_ids:
                c['historical_physical_camera_id']=c['physical_camera_id']
                c['physical_camera_id']=pid
                c['physical_identity_evidence']=link
                if '跨场同一位置另存候选关联' in c.get('reason',''):
                    c['historical_assignment_reason']=c['reason']
                    c['reason']=c['reason'].replace('跨场同一位置另存候选关联','高位全景跨场位置归并采用用户确认')
        catalog['physical_cameras']=[p for p in catalog['physical_cameras'] if p['physical_camera_id'] not in previous_ids]
        catalog['physical_cameras'].insert(0,{'physical_camera_id':pid,'label':decision['label'],
            'identity_status':'human_confirmed_position_group','motion_type':'fixed_human_confirmed',
            'installation_ids':[iid for iid in decision['installation_ids'] if iid in available],
            'representative_observations':observations,
            'historical_physical_camera_ids':sorted(previous_ids),'identity_evidence':link,
            'meaning':'One physical camera position; per-broadcast installation records are subordinate versions, not additional cameras.'})
        applied.append(decision['decision_id'])
