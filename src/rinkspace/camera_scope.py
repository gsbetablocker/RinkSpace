"""Source-time scene scope, before shot detection or camera assignment.

Ignored and pending spans are not shots. Historical full-broadcast products are
read-only inputs; this module never changes source media or old timeline files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path


VERSION = 'rinkspace.target-rink-scope.v1'
EXCLUDED_KINDS = {'off_rink_arena_kiss_and_cry', 'off_rink_remote_interview',
                  'non_target_interview_composite', 'remote_interview',
                  'non_target_material', 'black'}
RINK_KINDS = {'arena_skating', 'arena_warmup', 'target_venue',
              'target_venue_warmup', 'target_venue_replay_candidate'}
# Scene-only visual review of the actual 8 x 16 existing contact-sheet images.
# These labels do not identify cameras and do not label neighbouring frames.
TARGET_PANELS = {
    'Rd66CneD9io': list(range(1,17)),
    '2U7RhoKchXA': [1,2,3,4,5,6,7,9,10,12,13,14,16],
    'DXiFMHULDZ8': [1,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'oxXu1avhrvs': [1,2,3,4,5,8,9,10,11,12,13,15,16],
    'uaod0sM-9Hw': [1,2,3,5,6,7,8,9,10,11,12,13,15,16],
    'mEoZsjFK4r0': [1,2,3,4,5,6,7,8,9,10,11,13,15,16],
    'M_2a8gDUyrQ': [1,2,4,5,6,7,8,11,12,13,14,15],
    'yS-iLdqoHtg': list(range(1,17)),
}
EXCLUDED_PANELS = {'2U7RhoKchXA':[8,11], 'DXiFMHULDZ8':[2,16],
                   'oxXu1avhrvs':[6,7,14], 'uaod0sM-9Hw':[4,14],
                   'mEoZsjFK4r0':[12,14], 'M_2a8gDUyrQ':[3,10,16]}
VISUAL_TARGET_REVIEWS = {
    'Rd66CneD9io:shot:8963072': {
        'start_pts': 8963072, 'end_pts_exclusive': 10098688,
        'observed_pts': [8964608,9024512,9262080,9423872,9530368,9884672,9937920,10096640],
        'finding': '本轮查看该连续镜头的八张已有原图：本场冰场全景和热身，字幕覆盖时冰場仍可见。场景范围沿用已有连续镜头，不声明逐帧已核验。',
    },
}
# Explicit user statement and existing visual-review findings, not source-wide
# assumptions. Endpoints below retain the first frame showing the target rink.
EXPLICIT_EXCLUSIONS = {
    'Rd66CneD9io': [(0, 583.5333333333333, '用户确认：首次拍到本场冰场之前全部忽略', '本次任务用户要求记录.md'),
                   (12369.4, 12450.933333333334, '等分区；之后切入本场冰场', 'evidence/visual_review_root.json')],
    '2U7RhoKchXA': [(3146.3, 3216.1, '等分区', 'evidence/visual_review_root.json'),
                   (9498.6, 9558.533333333333, '等分区', 'evidence/visual_review_root.json')],
    'mEoZsjFK4r0': [(4987.4, 5799.166666666667, '远程访谈，没有本场冰场', 'evidence/visual_review_evidence.json')],
    'yS-iLdqoHtg': [(3997.766666666667, 4478.666666666667, '远程访谈，没有本场冰场', 'evidence/visual_review_viewer.json')],
    'oxXu1avhrvs': [(10105.066666666667, 10180.366666666667, '等分区', 'evidence/visual_review_evidence.json')],
    'DXiFMHULDZ8': [(11322.5, 11367.5, '等分区；后续冰场叠化不在排除范围内', 'evidence/visual_review_viewer.json')],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def load_source_scope(root, source_id, output=None):
    output = Path(output) if output else Path(root)/'outputs/camera_classification'
    path = output/'rink_scope.json'
    if not path.exists():
        raise FileNotFoundError('缺少冰场范围筛选；先运行 scripts/classify_event_cameras.py scope')
    return read(path)['sources'][source_id]


def merge_intervals(intervals):
    result = []
    for row in sorted(intervals, key=lambda row: row['start_pts']):
        a, b = int(row['start_pts']), int(row['end_pts_exclusive'])
        if b <= a:
            continue
        if result and a <= result[-1]['end_pts_exclusive']:
            result[-1]['end_pts_exclusive'] = max(b, result[-1]['end_pts_exclusive'])
        else:
            result.append({'start_pts': a, 'end_pts_exclusive': b})
    return result


def intersections(a, b, intervals):
    for row in intervals:
        left, right = max(a, row['start_pts']), min(b, row['end_pts_exclusive'])
        if left < right:
            yield left, right


def _scene_decision(shot, ice_samples=None):
    content, c = shot.get('content', {}), shot.get('classification', {})
    kind = content.get('kind')
    review = VISUAL_TARGET_REVIEWS.get(shot['shot_id'])
    if review and shot['start_pts']==review['start_pts'] and shot.get('end_pts_exclusive',shot['end_pts'])==review['end_pts_exclusive']:
        return 'target', review['finding'], 'current_visual_scene_review'
    if kind in EXCLUDED_KINDS or c.get('status') == 'not_applicable':
        return 'ignored', '已有证据识别为无本场冰场的内容', kind
    if kind in RINK_KINDS:
        return 'target', '已有画面复核识别为本场冰场', kind
    if c.get('status') == 'supported' and not str(c.get('installation_id', '')).endswith(':kiss_and_cry'):
        return 'target', '已有本场冰场背景或人工观察证据', c.get('assignment_basis')
    # Reuse actual cached matches, not candidate names or the old generic
    # context_candidate flag (which incorrectly included some interview frames).
    votes=defaultdict(set)
    for evidence in c.get('supporting_evidence',[]):
        for match in evidence.get('matches',[]):
            ref=match.get('reference_id','').split(':review:')
            if len(ref)!=2 or ref[0]!=shot['source_id']:continue
            index=int(ref[1]);regions=match.get('regions',{})
            if index in EXCLUDED_PANELS.get(ref[0],[]) and match.get('inliers',0)>=30 and match.get('x_span',0)>=.45:
                votes['ignored'].add(evidence.get('source_pts'))
            if index in TARGET_PANELS.get(ref[0],[]) and match.get('inliers',0)>=20 and match.get('x_span',0)>=.3 and (regions.get('end_wall_windows',0)>=3 or regions.get('grandstand_structure',0)>=4):
                votes['target'].add(evidence.get('source_pts'))
    for points in votes.values():points.discard(None)
    observations=len(shot.get('observations',[]))
    if len(votes['ignored'])>=2 and len(votes['ignored'])>=.8*observations and not votes['target']:
        return 'ignored','多时刻对应本轮已查看的非目标场景参照','cached_non_target_scene_correspondence'
    if shot.get('interval_type','shot')=='shot' and ice_samples and len(votes['target'])>=2 and len(votes['target'])>=.5*observations and not votes['ignored']:
        pts,values=ice_samples
        lo,hi=bisect_left(pts,shot['start_pts']),bisect_left(pts,shot.get('end_pts_exclusive',shot['end_pts']))
        samples=values[lo:hi]
        if samples and min(samples)>=.3:
            return 'target','多时刻对应已查看的本场冰场结构，且缓存样本有冰面可见性佐证；自动场景筛选尚非逐帧人工确认','cached_target_scene_correspondence_with_ice_samples'
    return 'pending', '尚未判定是否拍到本场冰场；不进入分镜或机位分类', None


def build_scope(catalog, output, *, reference_path=None):
    """Reuse explicit scene evidence; do not turn unknown camera IDs into exclusions."""
    output = Path(output)
    result = {'schema_version': VERSION, 'sources': {},
              'rule': '只处理拍到这场比赛冰场的画面；其余忽略，不分镜、不分机位。',
              'scene_screening_complete': False, 'all_frames_visually_verified': False,
              'method': 'Recorded scene reviews plus repeated cached structural correspondence to visually checked rink references; brightness is only a secondary visibility cue.',
              'reference_scene_review':{'target_panel_indices':TARGET_PANELS,'excluded_panel_indices':EXCLUDED_PANELS,
                                        'reviewed_source_intervals':VISUAL_TARGET_REVIEWS,
                                        'source':'outputs/event_survey_v11/all_source_review_v2/*_16_views.jpg',
                                        'scope':'Actual reference images reviewed; not a claim about whole groups or neighbouring frames.'},
              'history_catalog': str(reference_path) if reference_path else None}
    shots_by_source = defaultdict(list)
    for shot in catalog['shots']:
        shots_by_source[shot['source_id']].append(shot)
    for source in catalog['sources']:
        sid = source['source_id']
        tb = Fraction(source['track']['time_base'])
        rows = sorted(shots_by_source[sid], key=lambda s: s['start_pts'])
        cached_samples=[]
        for path in sorted((output.parent/'event_survey_v11'/sid/'chunks').glob('*/one_second_features.json')):
            cached_samples.extend((r['source_pts'],r['features']['ice_candidate_fraction']) for r in read(path))
        cached_samples.sort()
        ice_samples=([p for p,v in cached_samples],[v for p,v in cached_samples]) if cached_samples else None
        step = round((rows[0].get('end_pts_exclusive', rows[0]['end_pts'])-rows[0]['start_pts'])/rows[0]['frame_count'])
        end = max(s.get('end_pts_exclusive', s['end_pts']) for s in rows)
        explicit = [{'start_pts': round(a/float(tb)/step)*step,
                     'end_pts_exclusive': round(b/float(tb)/step)*step,
                     'reason': reason, 'evidence_path': (output/path).as_posix()}
                    for a,b,reason,path in EXPLICIT_EXCLUSIONS.get(sid, [])]
        decisions = []
        for shot in rows:
            a,b = shot['start_pts'], shot.get('end_pts_exclusive',shot['end_pts'])
            cuts = sorted({a,b,*[p for r in explicit for p in (r['start_pts'],r['end_pts_exclusive']) if a<p<b]})
            default, reason, evidence_kind = _scene_decision(shot,ice_samples)
            for left,right in zip(cuts,cuts[1:]):
                exclusion = next((r for r in explicit if r['start_pts']<=left and right<=r['end_pts_exclusive']),None)
                decision = 'ignored' if exclusion else default
                decisions.append({'start_pts':left,'end_pts_exclusive':right,'decision':decision,
                                  'reason':exclusion['reason'] if exclusion else reason,
                                  'evidence':{'historical_shot_id':shot['shot_id'],
                                              'path':exclusion['evidence_path'] if exclusion else str(reference_path or ''),
                                              'content_kind':evidence_kind}})
        entry = {'source_id':sid,'source_sha256':source['track']['source_sha256'],
                 'time_base':str(tb),'frame_step_pts':step,'end_pts_exclusive':end,
                 'source_frame_count':source['frame_count'],'decisions':decisions}
        for decision in ('target','ignored','pending'):
            intervals=merge_intervals([d for d in decisions if d['decision']==decision])
            entry[decision+'_intervals']=intervals
            entry[decision+'_frame_count']=sum((r['end_pts_exclusive']-r['start_pts'])//step for r in intervals)
            entry[decision+'_seconds']=sum((r['end_pts_exclusive']-r['start_pts'])*float(tb) for r in intervals)
        assert sum(entry[d+'_frame_count'] for d in ('target','ignored','pending'))==source['frame_count']
        result['sources'][sid]=entry
    result['summary']={key:round(sum(s[key] for s in result['sources'].values()),6)
                       for key in ('target_seconds','ignored_seconds','pending_seconds')}
    result['summary'].update({key:sum(s[key] for s in result['sources'].values())
                              for key in ('target_frame_count','ignored_frame_count','pending_frame_count')})
    return result


def clip_shot(shot, left, right, source_scope):
    if left==shot['start_pts'] and right==shot.get('end_pts_exclusive',shot['end_pts']):
        return deepcopy(shot)
    row=deepcopy(shot)
    step=source_scope['frame_step_pts'];tb=Fraction(source_scope['time_base'])
    row.update(shot_id=f"{shot['shot_id']}:scope:{left}-{right}", historical_shot_id=shot['shot_id'],
               start_pts=left,end_pts=right,end_pts_exclusive=right,
               start_seconds=float(left*tb),end_seconds=float(right*tb),
               start_frame_index=left//step,end_frame_index_exclusive=right//step,
               frame_count=(right-left)//step)
    row['observations']=[o for o in row.get('observations',[]) if left<=o['source_pts']<right]
    for side,p in [('start',left),('end',right)]:
        row[f'{side}_boundary_id']=None
        row[f'boundary_{side}']={'status':f'scope_{side}','source_pts':p,'source_seconds':float(p*tb),
                               'meaning':'场景筛选范围端点，不声明为切镜','exact_cut_claim':False}
    row['boundary_status']='scope_start'
    c=row.get('classification',{})
    evidence=[]
    for original in c.get('supporting_evidence',[]):
        e=deepcopy(original)
        if e.get('source_pts') is not None and not left<=e['source_pts']<right:
            continue
        for key in ('observations','matches_to_confirmed_observations'):
            if key in e:e[key]=[o for o in e[key] if o.get('source_pts') is not None and left<=o['source_pts']<right]
        evidence.append(e)
    c['supporting_evidence']=evidence
    if c.get('status')=='supported':
        c['candidate_installation_ids']=list(dict.fromkeys(c.get('candidate_installation_ids',[])+[c['installation_id']]))
        c.update(status='multiple_candidates',physical_camera_id=None,installation_id=None,
                 reason='已按冰场范围裁切；保留原有机位线索，裁切后的归属尚未重新验证。')
    return row


def project_catalog(catalog, scope):
    """Migrate existing results without detecting shots in ignored/pending spans."""
    result={k:deepcopy(v) for k,v in catalog.items() if k not in ('shots','installations','physical_cameras','appearance_groups')}
    rows=[]
    for shot in catalog['shots']:
        source_scope=scope['sources'][shot['source_id']]
        for a,b in intersections(shot['start_pts'],shot.get('end_pts_exclusive',shot['end_pts']),source_scope['target_intervals']):
            item=clip_shot(shot,a,b,source_scope)
            item['classification']['candidate_installation_ids']=[i for i in item['classification'].get('candidate_installation_ids',[]) if not i.endswith(':kiss_and_cry')]
            rows.append(item)
    result['shots']=rows
    ids={s['shot_id'] for s in rows}; aliases=defaultdict(list)
    for s in rows:aliases[s.get('historical_shot_id',s['shot_id'])].append(s['shot_id'])
    installations=[]
    for original in catalog.get('installations',[]):
        if original['installation_id'].endswith(':kiss_and_cry'):continue
        item=deepcopy(original)
        for key in ('shot_ids','supported_shot_ids','candidate_shot_ids'):
            item[key]=[new for old in item.get(key,[]) for new in aliases.get(old,[]) if new in ids]
        if item['shot_ids']:installations.append(item)
    result['installations']=installations
    iids={i['installation_id'] for i in installations}
    result['physical_cameras']=[]
    for original in catalog.get('physical_cameras',[]):
        item=deepcopy(original);item['installation_ids']=[i for i in item.get('installation_ids',[]) if i in iids]
        if item['installation_ids']:result['physical_cameras'].append(item)
    result['appearance_groups']=[]
    for original in catalog.get('appearance_groups',[]):
        item=deepcopy(original);item['shot_ids']=[new for old in item.get('shot_ids',[]) for new in aliases.get(old,[]) if new in ids]
        if item['shot_ids']:result['appearance_groups'].append(item)
    for source in result['sources']:
        ss=scope['sources'][source['source_id']]
        source['processing_scope']={k:v for k,v in ss.items() if k!='decisions'}
        source['shot_count']=sum(s['source_id']==source['source_id'] for s in rows)
    result['processing_scope']={'schema_version':VERSION, 'summary':scope['summary'],
                                'scene_screening_complete':scope['scene_screening_complete'],
                                'history_catalog':scope.get('history_catalog'),
                                'migration':'复用已有分镜并按目标范围裁切；本轮没有对排除或待判定区域重新分镜。'}
    result['scope']={'target_rink_only':True,'scene_screening_complete':scope['scene_screening_complete'],
                     'timeline_processing_complete':False,'all_physical_cameras_confirmed':False,
                     'geometry_or_grid_work_performed':False}
    counts=Counter(s['classification']['status'] for s in rows)
    result['summary']={'source_count':len(result['sources']),'segment_count':len(rows),
                       'frame_count':sum(s['frame_count'] for s in rows),
                       'duration_seconds':sum(s['end_seconds']-s['start_seconds'] for s in rows),
                       'status_counts':dict(counts), 'status_duration_seconds':{state:sum(s['end_seconds']-s['start_seconds'] for s in rows if s['classification']['status']==state) for state in counts},
                       'ignored_and_pending_are_not_shots':True}
    result['limitations']=['尚未完成八场所有画面的场景判定；待判定时段不作为已忽略或已分镜。',
                           '复用的旧机位候选仍有错误；场景筛选不证明机位归属准确。']
    return result


def migrate_scope(root, output):
    """Keep one immutable historical catalogue and publish the scoped catalogue."""
    root,output=Path(root).resolve(),Path(output).resolve()
    current=output/'catalog.json';history=output/'catalog_before_rink_scope.json'
    if not history.exists():
        # A byte-for-byte historical copy, not a duplicate set of media assets.
        history.write_bytes(current.read_bytes())
    validation_history=output/'validation_before_rink_scope.json'
    if (output/'validation.json').exists() and not validation_history.exists():
        validation_history.write_bytes((output/'validation.json').read_bytes())
    catalog=read(history)
    scope=build_scope(catalog,output,reference_path=history.relative_to(root).as_posix())
    projected=project_catalog(catalog,scope)
    from .camera_classification import synchronize_installation_membership, validate_catalog
    synchronize_installation_membership(projected)
    validation=validate_catalog(projected)
    if not validation['structural_pass']:
        raise ValueError(validation['structural_failures'][:5])
    write(output/'rink_scope.json',scope)
    write(current,projected)
    validation['processing_scope']=projected['processing_scope']
    validation['catalog_sha256']=hashlib.sha256(current.read_bytes()).hexdigest()
    write(output/'validation.json',validation)
    return scope
