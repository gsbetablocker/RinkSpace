"""One-exposure source-reading audit. Candidates never enter prepared inputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..review.fonts import review_font
from ..vision_source_shapes import _MaterialProfileFrame, read_source_dark_stroke_profiles
from ..vision_source_roles import organize_source_line_material_branches
from ..vision_programme_ice_review import _batched_actual_profiles, _actual_profile_candidates
from .dataset import digest, read_json


@dataclass(frozen=True)
class ObservationAuditReceipt:
    case_path: str
    result_path: str
    artifact_paths: dict
    covered_source_pts: list
    reference_used_for_measurement: bool = False
    fit_executed: bool = False
    user_approval: str = 'not_reviewed'


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _verified_ref(ref):
    path = Path(ref['path']).resolve(strict=True)
    if digest(path) != ref['sha256']:
        raise ValueError('Referenced evidence SHA differs: '+str(path))
    return path


def _pixel_requests(requests):
    # Entity names/navigation/reference metadata belong to audit orchestration.
    # The actual reader receives only the declared material and broad rectangles.
    return [{k:r[k] for k in ('id','roi_xyxy','measurement','material_signal')} for r in requests]


def measure(frame, read_requests, reader_config):
    """Read RGB only. No camera, accepted UV, reference file or fitted state."""
    if set(reader_config)-{'profile_reader'} != {'scan_axis','sample_step_px','sampling_inset_px'} or reader_config['scan_axis'] != 'y':
        raise ValueError('This audit implements vertical profiles of a declared paint center only')
    step, inset = reader_config['sample_step_px'], reader_config['sampling_inset_px']
    if not np.isfinite([step,inset]).all() or step <= 0 or inset < 2:
        raise ValueError('Invalid profile sampling; full five-pixel tangent support is required')
    mode=reader_config.get('profile_reader','batched_band')
    if mode not in ('batched_band','signed_interfaces'):
        raise ValueError('Unknown RGB profile reader')
    height,width = frame.rgb.shape[:2]
    result = []
    seen = set()
    for request in read_requests:
        if set(request) != {'id','roi_xyxy','measurement','material_signal'}:
            raise ValueError('Pixel reader does not accept reference/identity/navigation fields')
        if request['id'] in seen:
            raise ValueError('Duplicate read request')
        seen.add(request['id'])
        if request['measurement'] != 'paint_center' or request['material_signal'] != 'dark':
            raise ValueError('Only the O0.1 dark paint-center measurement is implemented')
        roi = request['roi_xyxy']
        if len(roi)!=4 or any(type(v) is not int for v in roi):
            raise ValueError('ROI must use native integer pixels')
        x0,y0,x1,y1 = roi
        if not (0<=x0<x1<=width and 0<=y0<y1<height and y1-y0>=20 and x1-x0>2*inset):
            raise ValueError('Complete broad source ROI required')
        xs = np.arange(x0+inset,x1-inset+1,step,dtype=float)
        query = np.c_[xs,np.full(len(xs),(y0+y1)/2)]
        normals = np.tile([0.,1.],(len(xs),1))
        intervals = np.tile([-(y1-y0)/2,(y1-y0)/2],(len(xs),1))
        offsets, profiles, smoothed, valid = _batched_actual_profiles(
            frame,query,normals,intervals,np.array(['dark']*len(xs)))
        signed=None
        if mode=='signed_interfaces':
            signed=read_source_dark_stroke_profiles(frame.rgb,binding=frame.binding,
                search_points_uv=query,normals=normals,search_intervals_px=intervals,
                keep_profiles=False,_validated_frame=frame)['readings']
        queries = []
        for qi in range(len(xs)):
            keep = valid[qi]
            values = _actual_profile_candidates(query[qi],normals[qi],offsets[qi,keep],
                profiles[qi,keep],smoothed[qi,keep],operator='paint_center',signal='dark',
                transition=None,with_quantiles=True)
            quantile_probs=[.25,.5,.75]
            if signed is not None:
                values=[];quantile_probs=[.1,.5,.9]
                for item in signed[qi]['all_actual_source_dark_stroke_candidates']:
                    borders=[item['actual_RGB_entering_interface'],item['actual_RGB_leaving_interface']]
                    sides=np.asarray([b['actual_source_transition_candidate_uv'] for b in borders])
                    quantiles=np.asarray([np.asarray(b['source_search_point_uv'])[None]+
                        np.asarray(b['source_q10_q50_q90_normal_offsets_px'])[:,None]*
                        np.asarray(b['source_search_normal'])[None] for b in borders])
                    values.append((np.asarray(item['actual_source_dark_stroke_center_candidate_uv']),
                        sides,any(item['source_flags'].values()),quantiles))
            candidates = []
            for ci,(uv,sides,bad,quantiles) in enumerate(values):
                status = ('ambiguous_material' if bad or len(values)!=1 else 'candidate_not_reviewed')
                candidates.append({'evidence_id':request['id']+'/q'+str(qi)+'/c'+str(ci),
                    'uv_or_curve':uv.tolist(),'actual_sides_uv':sides.tolist(),
                    'side_transition_quantiles_uv':quantiles.tolist(),'quantile_probabilities':quantile_probs,
                    'measurement':'midpoint_of_two_measured_signed_interfaces',
                    'status':status,'truncated_material_support':bool(bad),
                    'competing_candidates':len(values)>1,'material_interfaces_resolved':not bool(bad),'visibility':'unresolved' if bad else 'pixel_support_present',
                    'uncertainty':{'kind':'interface_transition_quantile_span_not_independent_point_covariance',
                        'quantile_probabilities':quantile_probs,'side_span_px':np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1).tolist()},
                    'active_for_fit':False})
            status = ('no_pixel_support' if not values else 'candidate_not_reviewed'
                if len(values)==1 and not values[0][2] else 'ambiguous_material')
            queries.append({'query_id':request['id']+'/q'+str(qi),'search_uv':query[qi].tolist(),
                'normal':normals[qi].tolist(),'interval_px':intervals[qi].tolist(),
                'status':status,'candidates':candidates,
                'profile':{'offsets_px':offsets[qi,keep].tolist(),'luminance':profiles[qi,keep].tolist(),
                    'smoothed_luminance':smoothed[qi,keep].tolist()}})
        result.append({'request_id':request['id'],'roi_xyxy':roi,'queries':queries})
    return result



def organize_readings(batch, mask_ref):
    """Use measured branches and same-exposure exclusion only; no old UV."""
    from ..vision_programme_ice_review import _candidate_context_excluded
    mask=np.asarray(Image.open(_verified_ref(mask_ref)).convert('L'))!=0
    reports=[]
    for request in batch:
        rows=[]
        for query in request['queries']:
            pool=[]
            for candidate in query['candidates']:
                excluded=any(_candidate_context_excluded(mask,p,query['normal'])
                    for p in [candidate['uv_or_curve'],*candidate['actual_sides_uv']])
                candidate['excluded_by_same_exposure_mask']=excluded
                usable=candidate['material_interfaces_resolved'] and not excluded
                pool.append({'actual_source_uv':candidate['uv_or_curve'] if usable else [],
                    'disposition':'measured_dark_stroke_center_candidate' if usable else 'excluded'})
            rows.append({'source_search_point_uv':query['search_uv'],
                'all_actual_source_dark_stroke_candidates':pool})
        # Unsupported/occluded queries remain explicit breaks, with no UV output.
        paths=[];run=[]
        for i,row in enumerate(rows):
            if any(c['actual_source_uv'] for c in row['all_actual_source_dark_stroke_candidates']):run.append(i)
            else:
                if run:paths.append(run);run=[]
                paths.append([i])
        if run:paths.append(run)
        organized=organize_source_line_material_branches(rows,navigation_tangent_uv=[1.,0.],
            source_visible_paths=paths,candidate_collection_key='all_actual_source_dark_stroke_candidates',
            candidate_disposition='measured_dark_stroke_center_candidate')
        selected=dict(zip(map(int,organized['source_query_indices']),map(int,organized['source_candidate_indices'])))
        model=organized.get('actual_source_line_organizer')
        for qi,query in enumerate(request['queries']):
            for ci,candidate in enumerate(query['candidates']):
                candidate['status']=('candidate_not_reviewed' if selected.get(qi)==ci else
                    'excluded_visibility' if candidate['excluded_by_same_exposure_mask'] else 'not_selected_source_branch')
            query['selected_candidate_index']=selected.get(qi)
            if qi in selected:query['status']='candidate_not_reviewed'
            else:
                occluded=False
                if model is not None:
                    # This prediction tests the visibility mask only, never creates a measurement.
                    x=query['search_uv'][0];slope,intercept=model['coefficients']
                    y=slope*(x-model['station_center'])+intercept
                    occluded=_candidate_context_excluded(mask,[x,y],query['normal'])
                query['status']='occluded' if occluded else 'no_pixel_support'
                query['failure_reason']='same_exposure_exclusion_on_source_line' if occluded else 'no_resolved_measured_branch_for_target_line'
        reports.append({'request_id':request['request_id'],'disposition':organized['disposition'],
            'source_line_support_qualified':organized['source_line_support_qualified'],
            'organizer':None if model is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},
            'selected_query_indices':list(selected),'fitted_line_used_as_measurement':False,
            'reference_UV_used':False,'visibility_ref':mask_ref})
    return reports

def _load_reference(reference, case):
    """Called only after RGB measurement; retain the approved bytes and groups."""
    packet_path = _verified_ref(reference['packet'])
    binding_path = _verified_ref(reference['binding'])
    acceptance_path = _verified_ref(reference['acceptance'])
    packet, acceptance = read_json(packet_path), read_json(acceptance_path)
    match = [f for f in acceptance['frames'] if f['index']==reference['programme_frame_index_1based']]
    if (not acceptance.get('accepted') or len(match)!=1 or
            match[0]['packet_sha256']!=reference['packet']['sha256'] or
            match[0]['binding_sha256']!=reference['binding']['sha256'] or
            match[0]['native_pts']!=case['pts'] or packet['source_pts']!=case['pts'] or
            packet['source']['source_video_sha256']!=case['source_sha256']):
        raise ValueError('Reference is not the exact accepted exposure/version')
    features = [f for f in packet['features'] if f['id'] in reference['feature_ids']]
    if {f['id'] for f in features} != set(reference['feature_ids']):
        raise ValueError('Reference feature absent')
    if any(f.get('measurement_operator')!='paint_center' for f in features):
        raise ValueError('Historical edge coordinates cannot be renamed as center measurements')
    return {'packet':reference['packet'],'binding':reference['binding'],
        'acceptance':reference['acceptance'],'user_statement':acceptance['user_statement'],
        'accepted_at_local':acceptance['accepted_at_local'],'features':features,
        'source_correlated_groups':packet.get('source_correlated_groups',[]),
        'old_readout_semantics':'Approved same-frame joint straight-line estimates, not independent raw samples',
        'native_pts':case['pts'],'original_record_bytes_unchanged':True}


def _comparison(batch, reference):
    """Diagnostic displacement to frozen piecewise reference, never a reader input."""
    edges=[]
    for feature in reference['features']:
        for segment in feature['segments']:
            uv=np.asarray(segment['uv'],float)
            edges.extend(zip(uv[:-1],uv[1:]))
    differences=[]
    for request in batch:
        for query in request['queries']:
            if query['status']!='candidate_not_reviewed':continue
            point=np.asarray(next(c for c in query['candidates'] if c['status']=='candidate_not_reviewed')['uv_or_curve'])
            distances=[]
            for a,b in edges:
                delta=b-a;norm=float(delta@delta)
                if norm==0:continue
                fraction=float((point-a)@delta/norm)
                if 0<=fraction<=1:
                    distances.append(float(np.linalg.norm(point-a-fraction*delta)))
            if distances: differences.append(min(distances))
    return {'compared_single_candidates':len(differences),
        'distance_px_median':None if not differences else float(np.median(differences)),
        'distance_px_max':None if not differences else max(differences),
        'meaning':'Diagnostic distance to frozen approved segments; not independent accuracy evidence'}


def _draw_artifacts(rgb, case, batch, reference, out):
    roi=case['requests'][0]['roi_xyxy']
    if any(r['roi_xyxy']!=roi for r in case['requests']):
        raise ValueError('One comparison card requires the same crop for every request')
    x0,y0,x1,y1=roi
    display=case["display"]
    seconds=case["pts"]*case["source"]["time_base_num"]/case["source"]["time_base_den"]
    stamp=f"{int(seconds)//3600:02d}:{int(seconds)//60%60:02d}:{seconds%60:06.3f}"
    target=" / ".join(case["requests"][0]["entity_candidates"])
    raw=Image.fromarray(rgb).crop((x0,y0,x1,y1));raw.save(out/'crop_original.png')
    old=raw.copy();draw=ImageDraw.Draw(old)
    for feature in reference['features']:
        for segment in feature['segments']:
            for u,v in segment['uv']:
                x,y=u-x0,v-y0
                draw.ellipse((x-3,y-3,x+3,y+3),outline='#cf00ae',width=1)
    old.save(out/'crop_reference.png')
    current=raw.copy();draw=ImageDraw.Draw(current)
    def mark(point,color,rejected=False):
        x,y=np.asarray(point)-[x0,y0]
        if rejected:
            draw.line((x-3,y-3,x+3,y+3),fill=color,width=1)
            draw.line((x-3,y+3,x+3,y-3),fill=color,width=1)
        else:draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
    for request in batch:
        for query in request['queries']:
            # Missing readings have no invented UV; only show a margin tick.
            if not query['candidates']:
                x=query['search_uv'][0]-x0
                draw.line((x,2,x,8),fill='#d22f2f',width=2)
            for c in query['candidates']:
                mark(c['uv_or_curve'],'#008a55' if c['status']=='candidate_not_reviewed' else '#d86600',c['status']!='candidate_not_reviewed')
                for point,color in zip(c['actual_sides_uv'],('#009cd1','#e2a400')):mark(point,color)
    current.save(out/'crop_reread.png')
    thumb=Image.fromarray(rgb).resize((640,360));d=ImageDraw.Draw(thumb)
    d.rectangle(tuple(v/3 for v in roi),outline='#c500a8',width=2);thumb.save(out/'locator.png')
    card=Image.new('RGB',(1960,1140),'#f6f7f9');d=ImageDraw.Draw(card);font=review_font(24)
    d.text((20,12),case['shot_id'].upper()+' / PTS '+str(case['pts'])+' / '+stamp+' — '+display['target_label'],font=font,fill='#142f3c')
    card.paste(thumb,(20,55));d.text((690,75),'左：原全帧定位。下：同裁剪、同缩放。',font=font,fill='#142f3c')
    d.text((690,120),'身份沿用人工记录；重读不读取旧 UV 或相机。',font=font,fill='#142f3c')
    d.text((690,165),'保留缺失与竞争项；这张卡尚未获得新验收。',font=font,fill='#142f3c')
    d.text((690,210),'粉圈：旧参考；绿圈：单一候选；橙叉：未决/拒绝。',font=font,fill='#142f3c')
    d.text((690,255),'青/黄圈：实际两侧界面；上缘红刻线：未读出。',font=font,fill='#142f3c')
    for y,title,panel in [(430,'原片（无覆盖）',raw),(665,'旧已审点：'+target+'（共同直线估计）',old),(900,'本次 RGB 重读：双边中点、竞争候选及拒绝项',current)]:
        d.text((20,y),title,font=font,fill='#142f3c');card.paste(panel,(20,y+35))
    card.save(out/'card.png')
    names=['crop_original.png','crop_reference.png','crop_reread.png']
    titles=['纯原片','旧已审参考','本次独立重读']
    panels=''.join('<figure><figcaption>'+title+'</figcaption><div class="scroll"><img class="crop" src="'+name+'"></div></figure>' for name,title in zip(names,titles))
    summary='；'.join(k+'='+str(v) for k,v in Counter(q['status'] for r in batch for q in r['queries']).items())
    page='''<!doctype html><meta charset="utf-8"><title>O0.1 · S01 原片取点对照</title>
<style>body{font:18px system-ui;margin:24px;background:#f6f7f9;color:#142f3c}p{line-height:1.6;max-width:1150px}figure{margin:20px 0}figcaption{font-weight:600;margin:8px 0}.scroll{overflow:auto;border:1px solid #ccc}.crop{display:block;width:1920px;max-width:none}.locator{width:640px;max-width:100%}button{padding:8px 14px;margin-right:8px}details{max-width:1150px}a{color:#12628c}</style>
<h1>O0.1 · S01 原片取点对照</h1>
<p>原生 PTS <b>__PTS__</b> · __TIME__ · __SIZE__ · 官方 __FPS__。当前只检查 <b>__LABEL__</b>（历史实体 ID：__TARGET__）。原全帧的框只表示搜索区。</p>
<img class="locator" src="locator.png">
<p><b>粉圈</b>：旧已审参考；<b>绿圈</b>：本次单一候选；<b>青/黄圈</b>：测出的两侧界面；<b>橙叉</b>：竞争或支撑截断的候选；上缘红刻线表示该横截面未读出，没有补造 UV。</p>
<p><button onclick="zoom(1)">原生 100%</button><button onclick="zoom(1.5)">150%</button><button onclick="zoom(2)">200%</button>三个裁图使用相同缩放，可横向滚动；纯原图始终保留。</p>'''+panels+'''
<p><b>读数范围：</b>搜索区 __ROI__；全宽均匀横截面、间隔 __STEP__ px、纵向搜索 __SPAN__ px。区域来自原全帧目视定位；没有用旧参考点或相机投影逐点引导。</p>
<p><b>审核范围：</b>沿用原审核的目标身份；不测试自主身份识别。旧参考由该帧材料读数的共同直线估计产生，本次候选直接取两侧 RGB 界面的中点。历史曾用于开发，因此这里只是回归检查。旧坐标、相关组和批准版本保持原样。</p>
<p><b>当前发现：</b>__FINDING__没有读出不能解释成原图没有线；未决候选不能解释成可信观测。本次结果等待用户看图，未导入拟合。</p>
<details><summary>读取计数与证据文件</summary><p>'''+html.escape(summary)+'''</p><p><a href="result.json">完整结果</a> · <a href="reference.json">冻结旧参考</a> · <a href="evidence.json">本次全部候选与 RGB 剖面</a> · <a href="card.png">静态单卡</a></p></details>
<script>function zoom(s){document.querySelectorAll('.crop').forEach(e=>e.style.width=(1920*s)+'px')}</script>'''
    tokens={'PTS':case['pts'],'TIME':stamp,'SIZE':str(rgb.shape[1])+'×'+str(rgb.shape[0]),
        'FPS':case['source']['frame_rate'],'LABEL':display['target_label'],'TARGET':target,
        'ROI':roi,'STEP':case['reader_config']['sample_step_px'],'SPAN':y1-y0,'FINDING':display['agent_finding']}
    for key,value in tokens.items():page=page.replace('__'+key+'__',html.escape(str(value)))
    page=page.replace('width:1920px','width:'+str(raw.width)+'px').replace('(1920*s)','('+str(raw.width)+'*s)')
    (out/'index.html').write_text(page,encoding='utf-8')
    return {n:str((out/n).resolve()) for n in ['index.html','card.png','locator.png',*names]}


def audit_case(case_path, assets_path, out_dir):
    case_path,assets_path=Path(case_path).resolve(strict=True),Path(assets_path).resolve(strict=True)
    case,assets=read_json(case_path),read_json(assets_path)
    out=Path(out_dir).resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError('Source-image diagnostics belong in Data')
    if out.exists():raise FileExistsError(out)
    if case.get('schema')!='rinkspace.observation_audit_case.v1' or type(case.get('pts')) is not int:
        raise ValueError('Audit requires the declared case schema and native integer PTS')
    for key in ('case_id','revision','source_id','source_sha256','shot_id','purpose','requests','reference_ref'):
        if key not in case:raise ValueError('Missing audit field: '+key)
    scope=read_json(_verified_ref(case['scope_ref']))
    source,clock=scope['source'],scope['source_clock']
    shot=next((s for s in scope['shots'] if s['id']==case['shot_id']),None)
    if (source['id']!=case['source_id'] or source['sha256']!=case['source_sha256'] or shot is None or
            not shot['processing']['first_pts']<=case['pts']<shot['processing']['stop_pts_exclusive']):
        raise ValueError('Audit source/exposure differs from registered shot scope')
    for key in ('width','height'):
        if source[key]!=case['source'][key]:raise ValueError('Native source dimensions differ')
    for key in ('time_base_num','time_base_den'):
        if clock[key]!=case['source'][key]:raise ValueError('Native time base differs')
    supplied=assets['sources'][case['source_id']]
    video=Path(supplied['path'] if isinstance(supplied,dict) else supplied).resolve(strict=True)
    if digest(video)!=case['source_sha256']:raise ValueError('Original video SHA differs')
    from ..review.render import source_frames
    decoded=list(source_frames(video,{**source,'time_base_num':clock['time_base_num'],
        'time_base_den':clock['time_base_den']},[case['pts']]))
    if len(decoded)!=1 or decoded[0][0]!=case['pts']:raise ValueError('Exact native exposure missing')
    rgb=np.ascontiguousarray(decoded[0][1][:,:,::-1])
    binding={'source_sha256':case['source_sha256'],'source_pts':case['pts'],
        'time_base':str(clock['time_base_num'])+'/'+str(clock['time_base_den']),
        'native_hw':list(rgb.shape[:2]),'shot_id':case['shot_id']}
    frame=_MaterialProfileFrame(rgb,binding,case['source']['expected_decoded_RGB_sha256'])
    batch=measure(frame,_pixel_requests(case['requests']),case['reader_config'])
    organization=organize_readings(batch,case['visibility_ref']) if case.get('visibility_ref') else []
    # Loading reference files is deliberately after the complete measurement.
    reference=_load_reference(case['reference_ref'],case)
    reader_paths=[Path(__file__),Path(__file__).parents[1]/'vision_programme_ice_review.py',
        Path(__file__).parents[1]/'vision_source_shapes.py',Path(__file__).parents[1]/'vision_source_roles.py']
    revision={str(p.resolve()):digest(p) for p in reader_paths}
    out.mkdir(parents=True)
    for request in batch:
        for query in request['queries']:
            for candidate in query['candidates']:
                candidate.update(source_id=case['source_id'],source_sha256=case['source_sha256'],
                    pts=case['pts'],reader_revision=revision,
                    support_ref='evidence.json#'+query['query_id']+'/profile',
                    uncertainty_ref='evidence.json#'+candidate['evidence_id']+'/uncertainty')
    _json(out/'evidence.json',{'source':binding,'reader_revision':revision,'requests':batch})
    _json(out/'reference.json',reference)
    artifacts=_draw_artifacts(rgb,case,batch,reference,out)
    artifacts.update(evidence=str(out/'evidence.json'),reference=str(out/'reference.json'))
    counts=Counter(q['status'] for r in batch for q in r['queries'])
    result={'schema':'rinkspace.observation_audit_result.v1','case_id':case['case_id'],'case_revision':case['revision'],
        'case_ref':{'path':str(case_path),'sha256':digest(case_path)},'source':{**binding,
            'source_id':case['source_id'],'video_path':str(video),'decoded_RGB_sha256':frame.digest,
            'actual_native_pts_verified':True},'requests':case['requests'],'reader_config':case['reader_config'],
        'reader_revision':revision,'source_line_organization':organization,'reference_used_for_measurement':False,'camera_used_for_measurement':False,
        'reference_loaded_after_measurement':True,'reference_preserved':case['reference_ref'],
        'query_counts':dict(counts),'candidate_count':sum(len(q['candidates']) for r in batch for q in r['queries']),
        'diagnostic_comparison':_comparison(batch,reference),
        'identity_scope':'Human-approved target identity reused; each new pixel candidate remains unreviewed',
        'evaluation_role':'development_regression_not_unseen_validation',
        'actual_limits':['Person exclusion and unresolved reads remain gaps',
            'Comparison uses existing approved lines; it is not a new unseen-sample validation'],
        'uncovered_conditions':case['uncovered_conditions'],'required_shot_ids':case['required_shot_ids'],
        'artifacts':artifacts,'fit_executed':False,'formal_observations_written':False,'user_approval':'not_reviewed'}
    # Source evidence and approvals must remain byte-for-byte unchanged.
    for name in ('packet','binding','acceptance'):_verified_ref(case['reference_ref'][name])
    _json(out/'result.json',result)
    receipt=ObservationAuditReceipt(str(case_path),str(out/'result.json'),artifacts,[case['pts']])
    _json(out/'receipt.json',receipt.__dict__)
    return receipt
