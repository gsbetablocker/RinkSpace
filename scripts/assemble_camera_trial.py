"""Combine separately reviewed evidence without treating worker boundaries as cuts."""
from rinkspace.storage import resolve_asset as _resolve_asset
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=_resolve_asset(ROOT, ROOT/'outputs/camera_classification/trial_010800_013600')
LABELS={'main_long_side':'长边主机位','secondary_long_side':'长边副机位','low_end':'低位端部','high_oblique_overview':'高位斜俯视','unresolved':'机位待定','ignore':'忽略','transition':'转场混合'}


def pts(t): return round(t*60)*256
def sec(t): return pts(t)/15360
def stamp(t):return f'{int(t//3600):02}:{int(t%3600//60):02}:{t%60:06.3f}'
def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    original=[];parts=[];events=[];sourcepts=[]
    for tag in 'abc':
        path=OUT/f'part_{tag}/decisions.json';d=load(path);parts.append({'path':path.relative_to(ROOT).as_posix(),'review':d.get('review',{}),'status':d.get('status',d.get('authority'))})
        scan=load(path.with_name('scan.json'))
        with np.load(path.with_name('frame_metrics.npz')) as archive:sourcepts.extend(archive['pts'].tolist())
        for i,entry in enumerate(d['intervals']):
            r=copy.deepcopy(entry);r['start']=sec(r['start']);r['end']=sec(r['end']);r['raw_provenance']=[f'part_{tag}:{i+1}'];r['interval_type']='shot'
            r['candidate_camera_ids']=r.get('candidate_camera_ids',r.get('camera_candidates',[]))
            r['context_label']='回放' if r.get('mode',r.get('programme_context'))=='replay' else ''
            bp=r.get('boundary_pts',{})
            r['boundary_type']=r.get('boundary_type',bp.get('type','unspecified') if isinstance(bp,dict) else 'unspecified')
            if r.get('mixed_frame') or r['boundary_type']=='dissolve_start' or ('dissolve' in r['boundary_type'] and r['camera_id']=='unresolved'):
                if not r['candidate_camera_ids'] and r['camera_id']!='unresolved':
                    r['candidate_camera_ids']=[r['camera_id']]
                r['camera_id']='transition';r['interval_type']='transition'
            elif r['camera_id']=='ignore':
                r['interval_type']='ignored'
                r['ignored_components']=[{'start':r['start'],'end':r['end'],'reason':r['reason'],'raw_provenance':r['raw_provenance'][0],'authority':r.get('authority','assistant_independent_visual_assessment'),'scope_correction':r.get('scope_correction')}]
            meta=r.get('boundary_metadata',{})
            if meta.get('type')=='dissolve':
                start,end=map(sec,meta['observed_extent'])
                events.append({'start':start,'end':end,'camera_ids':[meta['from_camera_id'],meta['to_camera_id']],'source':r['raw_provenance'][0], 'uncertainty':meta.get('boundary_uncertainty_seconds',0.1)})
            original.append(r)
    original.sort(key=lambda r:r['start'])
    assert original[0]['start']==4080 and original[-1]['end']==5760
    for a,b in zip(original,original[1:]):assert abs(a['end']-b['start'])<1e-7,(a['end'],b['start'])
    # Slice only at observed dissolve extents; a mixed frame is not a new camera.
    cuts=sorted({p for r in original for p in [r['start'],r['end']]}|{p for e in events for p in [e['start'],e['end']]})
    rows=[]
    for start,end in zip(cuts,cuts[1:]):
        mid=(start+end)/2
        r=copy.deepcopy(next(r for r in original if r['start']<=mid<r['end']))
        r['start']=start;r['end']=end
        event=next((e for e in events if e['start']<=mid<e['end']),None)
        if event:
            r['camera_id']='transition';r['interval_type']='transition';r['boundary_type']='dissolve';r['candidate_camera_ids']=list(dict.fromkeys(event['camera_ids']));r['transition_event']=event['source'];r['boundary_uncertainty_seconds']=event['uncertainty'];r['context_label']='回放';r['reason']='回放叠化：'+ ' → '.join(LABELS[x] for x in event['camera_ids'])+'。两段画面在此范围重叠，不指定单一机位。'
        same_event=bool(rows and event and rows[-1].get('transition_event')==event['source'])
        same_original=bool(rows and not event and rows[-1]['raw_provenance']==r['raw_provenance'] and rows[-1]['camera_id']==r['camera_id'])
        chunk_join=bool(rows and start in [4640,5200] and r['boundary_type']=='range_start' and rows[-1]['camera_id']==r['camera_id'])
        both_ignore=bool(rows and r['camera_id']==rows[-1]['camera_id']=='ignore')
        if same_event or same_original or chunk_join or both_ignore:
            rows[-1]['end']=end
            rows[-1]['evidence_images']=list(dict.fromkeys(rows[-1].get('evidence_images',[])+r.get('evidence_images',[])))
            rows[-1]['raw_provenance']=list(dict.fromkeys(rows[-1]['raw_provenance']+r['raw_provenance']))
            if both_ignore:
                rows[-1]['ignored_components']+=r.get('ignored_components',[])
                rows[-1]['reason']='；'.join(dict.fromkeys(c['reason'] for c in rows[-1]['ignored_components']))
        else: rows.append(r)
    # Original full-size JPEGs are indexed by exact source PTS in their filenames.
    frames={}
    for path in sorted(OUT.glob('part_*/*/*.jpg')):
        match=re.fullmatch(r'(?:pts_)?(\d{10,})',path.stem)
        if match:
            key=int(match.group(1));frames.setdefault(key,{'pts':key,'seconds':key/15360,'image_path':path.relative_to(ROOT).as_posix()})
    counters=Counter();durations=defaultdict(float);camera_counts=Counter()
    for r in rows:
        key=r['camera_id'];prefix='I' if key=='ignore' else ('T' if key=='transition' else 'S');counters[prefix]+=1;r['id']=f'{prefix}{counters[prefix]:03}'
        r['start_pts']=pts(r['start']);r['end_pts']=pts(r['end']);r['authority']='assistant_independent_visual_assessment'
        within=[v for k,v in frames.items() if r['start_pts']<=k<r['end_pts']]
        chosen=[]
        for fraction in [.12,.5,.88]:
            if not within:break
            t=r['start']+(r['end']-r['start'])*fraction
            pick=min(within,key=lambda f:abs(f['seconds']-t))
            if pick not in chosen:chosen.append(pick)
        r['samples']=chosen
        if chosen:r['representative_image']=chosen[len(chosen)//2]['image_path']
        assert r.get('representative_image') and (ROOT/r['representative_image']).is_file(),r['id']
        for path in r.get('evidence_images',[]):assert (ROOT/path).is_file(),path
        if key=='secondary_long_side':r['review_note']='长边副机位：本次独立视觉判断，重点复核主、副机位区分。'
        if key=='transition':r['context_label']='回放 · 转场混合'
        if key=='ignore':
            r['context_label']='按用户指定范围合并忽略'
            r['candidate_camera_ids']=[]
        if key=='unresolved':r['context_label']='用户同意暂放 · 保留原候选' if r.get('user_review_status')=='deferred_by_user' else '具体候选见下方'
        durations[key]+=r['end']-r['start'];camera_counts[key]+=1
    unique=sorted({x for x in sourcepts if 4080*15360<=x<5760*15360})
    assert len(unique)==100800 and np.all(np.diff(unique)==256)
    assert abs(sum(durations.values())-1680)<1e-7
    summary=f"28 分钟 · {counters['S']} 个保留镜头（其中 {camera_counts['unresolved']} 个机位待定） · {counters['T']} 段转场混合 · {counters['I']} 段合并忽略"
    reps=[]
    preferred={'main_long_side':4270,'low_end':4145,'secondary_long_side':4130}
    for camera_id,t in preferred.items():
        r=next((r for r in rows if r['camera_id']==camera_id and r['start']<=t<r['end']),None)
        if r:
            v=min((v for v in frames.values() if r['start']<=v['seconds']<r['end']),key=lambda x:abs(x['seconds']-t));reps.append(dict(v,camera_id=camera_id))
    data={'schema_version':'rinkspace.bounded_camera_trial.v1','source_path':'data/input/oxXu1avhrvs.mp4','source_sha256':'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891','source_hash_verified_this_run':True,'start':4080,'end':5760,'time_base':'1/15360','frame_rate':60,'status':'assistant_complete_pending_user_review','summary_text':summary,'intervals':rows,'camera_representatives':reps,'statistics':{'counts':dict(camera_counts),'seconds':dict(durations),'unique_source_frames_scanned':len(unique),'timeline_gap_count':0},'part_reviews':parts,'method_notes':['对所选28分钟的100,800帧做了相邻帧变化扫描；实际视觉复核每5秒样本、全部候选切点前后图，并对回放/疑难段加密抽帧。不是逐帧人眼看完100,800帧。','硬切采用60fps源PTS；叠化单列重叠范围。A段叠化边界约0.1秒观察精度，其他边界精度见分段原始记录。浏览器播放跳转只供导航，精确时间以源PTS为准。','三个分段接合处已核对并合并同一连续镜头，人工分工边界不新增镜头。回放仍保留，时间是原文件时间而非动作发生时刻。','四机位位置及教学参照来自用户；本次新镜头归属为助手判断，尚未经用户验收。置信字段是主观判断，不是校准过的准确率。','本段未发现高位斜俯视。长边主/副机位是重点复核项；机位待定处显示具体候选。只有围挡/墙面而完全无冰面的近景按本次忽略口径处理；按用户最新补充，专门拍裁判的镜头即使带少量冰面也忽略。']}
    data['user_scope_knowledge_path']='configs/camera_classification/ankara_2026_user_confirmed.json'
    data['user_scope_refinement']='专门拍裁判的镜头，即使带到少量冰面也直接忽略；用户在查看本次结果时明确补充。'
    (OUT/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 01:08:00–01:36:00 镜头分组试验','',summary,'','[打开结果页](http://127.0.0.1:8771/) · [本地页面](index.html) · [完整JSON](result.json)','','本次结果由助手独立判断，等待用户验收；不宣称已验证的识别准确率。','']
    lines+=['| 机位/类别 | 段数 | 时长（秒） |','| --- | ---: | ---: |']
    lines += [f'| {LABELS[k]} | {camera_counts[k]} | {durations[k]:.3f} |' for k in LABELS if camera_counts[k]]
    lines+=['','## 待定与限制','']
    lines += [f"- {r['id']} {stamp(r['start'])}–{stamp(r['end'])}：{' / '.join(LABELS[x] for x in r['candidate_camera_ids'])}。{r['reason']}" for r in rows if r['camera_id']=='unresolved']
    lines+=['']+['- '+n for n in data['method_notes']]+['','页面服务若已关闭，在项目根目录运行 `.venv\\Scripts\\python.exe scripts\\serve_camera_trial.py`，然后打开上方结果页。','','## 完整镜头表','','区间均为左闭右开；忽略区间内部不再分镜。','','| 编号 | 开始 | 结束 | 机位/类别 | 原图 |','| --- | --- | --- | --- | --- |']
    for r in rows:
        relative=Path(r['representative_image']).relative_to(OUT.relative_to(ROOT)).as_posix()
        lines.append(f"| {r['id']} | {stamp(r['start'])} | {stamp(r['end'])} | {LABELS[r['camera_id']]} | [查看]({relative}) |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'counts':dict(camera_counts),'seconds':dict(durations)},ensure_ascii=False))


if __name__=='__main__':main()
