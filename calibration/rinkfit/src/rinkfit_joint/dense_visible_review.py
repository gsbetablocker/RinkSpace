"""Immutable six-shot before/after cards on original native exposures.

The numerical comparison ledger is a separate input. These images neither
change source qualification nor certify formal Query/physical feasibility.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext
import argparse, html, json, time
import cv2
import numpy as np
from .io_utils import read_json, write_json, file_sha256
from .joint_metrics import PacketArrays
from .joint_replay import render_snapshots

def ref(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': file_sha256(path)}

def checked(binding):
    path = Path(binding['file']).resolve()
    if file_sha256(path) != binding['sha256']:
        raise ValueError('Changed visible review dependency: ' + str(path))
    return path

def operator_bindings():
    root = Path(__file__).parent
    return {name: ref(root/name) for name in
            ('dense_visible_review.py', 'joint_review_camera.py', 'joint_replay.py',
             'joint_model.py', 'joint_display.py', 'joint_review_packet.py')}

def prepare(packet, baseline, scope, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('The immutable visible review plan already exists')
    model_file = Path(baseline).resolve()/'model.json'
    model = read_json(model_file)
    source = read_json(scope)
    loader = PacketArrays(packet)
    p = loader.packet
    arrays = {k: loader.load(k) for k in ('uv','admission','material_index','native_pts','shot_index','kind')}
    physical = arrays['admission'] == 1
    selected = {'s01':317696, 's03':729856, 's05':986368, 's06':1330944}
    fixed = []
    for sn, sid in enumerate(p['shot_ids']):
        if sid not in model['shots']:
            raise ValueError('Packet and baseline shot identities differ')
        ids = np.flatnonzero(physical & (arrays['shot_index'] == sn))
        if sid not in selected:
            pairs = np.unique(np.c_[arrays['native_pts'][ids],arrays['material_index'][ids]],axis=0)
            stamps, counts = np.unique(pairs[:,0],return_counts=True)
            selected[sid] = int(stamps[np.argmax(counts)])
        stamp = selected[sid]
        rows = ids[arrays['native_pts'][ids] == stamp]
        if not len(rows):
            raise ValueError('Fixed native exposure lacks current qualified source rows')
        kinds = arrays['kind'][rows]
        corner = rows[kinds == 1]
        witness = int(corner[0] if len(corner) else rows[0])
        fixed.append({'shot_id':sid,'native_pts':stamp,'source_seconds':stamp/12800.,
                      'selection_basis':'predeclared source context' if sid in ('s01','s03','s05','s06') else 'maximum distinct current physical materials; earliest tie; no model residual',
                      'raw_row':witness,'source_row_id':p['source_row_ids'][witness],
                      'source_uv':arrays['uv'][witness].tolist(),
                      'physical_source_rows_at_frame':len(rows),
                      'independent_validation':False})
    result = {'schema':'rinkfit.current_sixshot_visible_review_plan.v1',
              'packet':ref(packet),'baseline_model':ref(model_file),
              'baseline_state_directory':str(Path(baseline).resolve()),
              'baseline_binding':ref(Path(baseline)/'observed_row_initialization_state_binding.json'),
              'scope':ref(scope),'source':source['source'],
              'fixed_frames':fixed,'operator':operator_bindings(),
              'fixed_frames_may_be_changed_by_candidate':False,
              'selection_uses_residual_or_candidate':False,
              'development_comparison':True,'independent_validation':False,
              'no_source_or_weight_modifications':True,'quality_accepted':False}
    output.mkdir(parents=True)
    write_json(output/'plan.json',result)
    return result

def rebind_operator(plan_file, output, reason):
    """Explicitly update rendering code, preserving the original cohort/model."""
    if not reason.strip():raise ValueError('Rendering operator change needs an explicit reason')
    plan_file=Path(plan_file).resolve();old=read_json(plan_file)
    for key in ('scope','baseline_model','baseline_binding'):checked(old[key])
    for side in ('baseline','candidate'):
        checked(old.get(side+'_packet',old['packet']))
        if old.get(side+'_current_input_v1'):checked(old[side+'_current_input_v1'])
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('The immutable visible review plan already exists')
    result=dict(old,operator=operator_bindings(),previous_plan=ref(plan_file),
                operator_change_reason=reason,baseline_and_fixed_frames_unchanged=True)
    output.mkdir(parents=True);write_json(output/'plan.json',result)
    return result

def _validate_comparison(metrics, automatic, packet_ref, baseline_ref, candidate_ref):
    comparison=read_json(metrics)
    if comparison.get('schema')!='rinkfit.dense_visible_same_source_comparison.v1':
        raise ValueError('A complete same-source numerical comparison is required')
    if not comparison.get('full_raw_domain_compared') or comparison.get('quality_accepted') is True:
        raise ValueError('Visible candidate requires full raw domain; metrics cannot certify quality')
    before=read_json(checked(comparison['baseline']));after=read_json(checked(comparison['candidate']))
    for report,model_ref in ((before,baseline_ref),(after,candidate_ref)):
        if report.get('schema')!='rinkfit.dense_visible_metrics.v1':raise ValueError('Unknown numerical operator')
        for key in ('source_packet','model','selection'):checked(report[key])
        if report['source_packet']['sha256']!=packet_ref['sha256'] or report['model']['sha256']!=model_ref['sha256']:
            raise ValueError('Numerical comparison refers to another source or model')
        if not report['complete_raw_domain_processed'] or report['processed_rows']!=report['raw_rows']:
            raise ValueError('Numerical comparison omits raw rows')
        if report['raw_rows']!=comparison['raw_rows']:raise ValueError('Numerical comparison denominator differs')
        if not report.get('operator_source_files'):raise ValueError('Numerical operator source identity missing')
        for binding in report['operator_source_files'].values():checked(binding)
    if before['selection']['sha256']!=after['selection']['sha256']:
        raise ValueError('Numerical comparison changed qualification or representative selection')
    for field in ('sampling','operator_codes'):
        if before[field]!=after[field]:raise ValueError('Numerical comparison operator changed: '+field)
    operators=lambda r:{k:v['sha256'] for k,v in r['operator_source_files'].items()}
    if operators(before)!=operators(after):raise ValueError('Numerical comparison code changed')
    if comparison['source_packet']['sha256']!=packet_ref['sha256']:raise ValueError('Comparison source differs')
    if automatic['comparison']['sha256']!=file_sha256(metrics):raise ValueError('Automatic frames belong to another comparison')
    checked(automatic['comparison'])
    return comparison

def _bind_frame_sources(frames, packet, *, reader=None):
    inputs=reader if reader is not None else PacketArrays(packet);p=inputs.packet
    arrays={k:inputs.load(k) for k in ('native_pts','shot_index','uv','material_index')}
    count=len(arrays['native_pts'])
    for frame in frames:
        row=int(frame['raw_row'])
        if row<0 or row>=count:raise ValueError('Review source row outside raw domain')
        sid=p['shot_ids'][int(arrays['shot_index'][row])]
        if frame['shot_id']!=sid or int(frame['native_pts'])!=int(arrays['native_pts'][row]):
            raise ValueError('Review raw ID and native exposure do not match')
        uv=arrays['uv'][row].tolist()
        if 'source_uv' in frame and frame['source_uv']!=uv:raise ValueError('Review crop changed original UV')
        binding=inputs.source_binding(row) if hasattr(inputs,'source_binding') else dict(
            source_row_id=p['source_row_ids'][row],native_pts=int(arrays['native_pts'][row]),shot_id=sid,
            source_uv=uv,material_index=int(arrays['material_index'][row]),
            retired_material=int(arrays['material_index'][row])==int(np.iinfo(np.uint32).max))
        if 'source_row_id' in frame and frame['source_row_id']!=binding['source_row_id']:
            raise ValueError('Review fixed original source ID changed')
        frame.update(binding,source_seconds=int(frame['native_pts'])/12800.)
        if 'expected_native_crop_xyxy' in frame and _crop_bounds(frame)!=frame['expected_native_crop_xyxy']:
            raise ValueError('Fixed original native crop changed')


def _source_side_binding(frame):
    return {key:frame[key] for key in ('source_row_id','source_uv','native_pts','shot_id',
            'material_index','material_label','path','retired_material','qualification','current_input_v1','packet') if key in frame}


def _annotate_explicit_source_marker(image, subject):
    """Development annotation only; never an admitted or statistical source dot."""
    marker=subject.get('explicit_source_marker')
    if marker is None:return
    if marker.get('schema')!='rinkfit.fixed_development_source_marker.v1':
        raise ValueError('Unknown fixed development source marker')
    if marker.get('raw_row')!=subject['raw_row'] or marker.get('source_row_id')!=subject['source_row_id']:
        raise ValueError('Development marker changed its original raw identity')
    uv=np.asarray(subject['source_uv'],float)
    if not np.array_equal(marker.get('source_uv'),uv) or uv.shape!=(2,) or not np.isfinite(uv).all() or (uv<0).any() or (uv>[1919,1079]).any():
        raise ValueError('Development marker changed original source pixels')
    status=marker.get('decision')
    if status not in ('unknown','revoked','rejected','retained_source_measurement') or marker.get('annotation_only')is not True or marker.get('qualification_restored')is not False:
        raise ValueError('Development marker cannot issue or restore source qualification')
    color=(0,165,255) if status in ('revoked','rejected') else (255,210,60) if status=='unknown' else (210,210,210)
    xy=tuple(np.rint(uv).astype(int))
    cv2.circle(image,xy,7,(0,0,0),3,cv2.LINE_AA)
    cv2.drawMarker(image,xy,color,cv2.MARKER_TILTED_CROSS,11,1,cv2.LINE_AA)
    label='DEV SOURCE '+status.upper()+' raw'+str(subject['raw_row'])
    origin=(max(2,min(xy[0]+10,1920-560)),max(17,min(xy[1]-12,1060)))
    cv2.putText(image,label,origin,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,label,origin,cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)


def _crop_bounds(subject):
    uv = np.asarray(subject['source_uv'],float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError('Native crop needs original source UV')
    lo = np.floor(uv-[160,120]).astype(int)
    lo = np.maximum(0,np.minimum(lo,[1920-320,1080-240]))
    return [int(lo[0]),int(lo[1]),int(lo[0]+320),int(lo[1]+240)]

def render(plan_file, candidate, output, *, automatic=None, metrics=None,
           compiled_dir=None, gpu_dll=None, gpu_validation=None, baseline_only=False, fixed_preview=False):
    start=time.perf_counter()
    if fixed_preview and (baseline_only or automatic is not None or metrics is not None):
        raise ValueError("Fixed diagnostic preview cannot carry baseline-only or full-domain comparison claims")
    plan_file=Path(plan_file).resolve();plan=read_json(plan_file)
    if plan.get('schema')!='rinkfit.current_sixshot_visible_review_plan.v1':
        raise ValueError('Unknown current visible plan')
    for binding in plan['operator'].values():checked(binding)
    baseline_packet_ref=plan.get('baseline_packet',plan['packet'])
    candidate_packet_ref=plan.get('candidate_packet',plan['packet'])
    baseline_packet=checked(baseline_packet_ref);candidate_packet=checked(candidate_packet_ref)
    source_objective_changed=baseline_packet_ref['sha256']!=candidate_packet_ref['sha256']
    if source_objective_changed and plan.get('objective_changed')is not True:
        raise ValueError('Different review packets require an explicit changed-objective plan')
    if source_objective_changed and not (fixed_preview or baseline_only):
        raise ValueError('Changed-objective fixed cards cannot claim an existing same-source full-domain comparison')
    scope=checked(plan['scope']);baseline=checked(plan['baseline_model'])
    candidate_file=baseline if baseline_only else Path(candidate).resolve()/'model.json'
    candidate_ref=ref(candidate_file)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Immutable visible generation already exists')
    frames=[dict(x,role='fixed') for x in plan['fixed_frames']]
    automatic_document=None
    if automatic is not None:
        automatic_document=read_json(automatic)
        for x in automatic_document['frames']:
            if x['shot_id'] not in {v['shot_id'] for v in plan['fixed_frames']}:
                raise ValueError('Automatic frame is outside the fixed shot scope')
            frames.append(dict(x,role='automatic_worst_or_regression'))
    if not baseline_only and not fixed_preview and (automatic_document is None or metrics is None):
        raise ValueError('Candidate review requires complete metrics and automatic worst/regression frames')
    if not baseline_only and not fixed_preview:
        _validate_comparison(metrics,automatic_document,baseline_packet_ref,plan['baseline_model'],candidate_ref)
    focus=plan.get('fixed_focus_frames',[{'shot_id':'s03','native_pts':729856,'raw_row':187863,'role':'fixed_focus_Ad2_right_bottom'},
           {'shot_id':'s03','native_pts':737792,'raw_row':188110,'role':'fixed_focus_Ad1_left_top'}])
    frames.extend(focus)
    from .joint_review_packet import CurrentReviewPacket,source_point_reader
    required_rows=[int(x['raw_row'])for x in frames]
    def reader_for(side,path):
        transport=plan.get(side+'_current_input_v1')
        return CurrentReviewPacket(path,current_input=transport,required_rows=required_rows) if transport else PacketArrays(path)
    baseline_reader=reader_for('baseline',baseline_packet)
    candidate_reader=baseline_reader if baseline_only else reader_for('candidate',candidate_packet)
    baseline_frames=deepcopy(frames);candidate_frames=deepcopy(frames)
    _bind_frame_sources(baseline_frames,baseline_packet,reader=baseline_reader)
    _bind_frame_sources(candidate_frames,baseline_packet if baseline_only else candidate_packet,reader=candidate_reader)
    for subject,old_source,new_source in zip(frames,baseline_frames,candidate_frames):
        for key in ('source_row_id','source_uv','native_pts','shot_id'):
            if old_source[key]!=new_source[key]:raise ValueError('Before/after cards changed original source '+key)
        subject.update(source_uv=old_source['source_uv'],source_row_id=old_source['source_row_id'],
                       baseline_source=_source_side_binding(old_source),candidate_source=_source_side_binding(new_source))
        if 'qualification'in new_source:subject['candidate_current_qualification']=new_source['qualification']
    output.mkdir(parents=True)
    points=sorted({int(x['native_pts']) for x in frames})
    common=dict(pick_pts=points,shots=sorted({x['shot_id'] for x in frames}),
                compiled_dir=compiled_dir,gpu_dll=gpu_dll,gpu_validation=gpu_validation,
                top_view=False,only_pick_pts=True)
    with source_point_reader(baseline_reader) if isinstance(baseline_reader,CurrentReviewPacket) else nullcontext():
        before=render_snapshots(baseline,scope,output/'before',packet_file=baseline_packet,**common)
    if baseline_only:after=before
    else:
        with source_point_reader(candidate_reader) if isinstance(candidate_reader,CurrentReviewPacket) else nullcontext():
            after=render_snapshots(candidate_file,scope,output/'after',packet_file=candidate_packet,**common)
    old={(x['shot_id'],x['native_pts']):x for x in before['frames']}
    new={(x['shot_id'],x['native_pts']):x for x in after['frames']}
    baseline_label=plan.get('baseline_label','R9 BASELINE')
    baseline_description=plan.get('baseline_description','固定 r9 基线')
    cards=[]
    for index, subject in enumerate(frames):
        key=subject['shot_id'],int(subject['native_pts'])
        a,b=old[key],new[key]
        if a['decoded_native_BGR_sha256']!=b['decoded_native_BGR_sha256']:
            raise ValueError('Before and after images used different original pixels')
        ai=cv2.imread(a['path']);bi=cv2.imread(b['path'])
        if ai is None or bi is None or ai.shape!=(1080,1920,3) or bi.shape!=ai.shape:
            raise ValueError('The full original 1920x1080 raster is required')
        _annotate_explicit_source_marker(ai,baseline_frames[index])
        _annotate_explicit_source_marker(bi,candidate_frames[index])
        name=f"{index:02d}_{key[0]}_{key[1]}"
        caption=np.full((60,3840,3),24,np.uint8)
        for x,label in ((12,baseline_label),(1932,baseline_label if baseline_only else 'CURRENT CANDIDATE')):
            cv2.putText(caption,label+f"  {key[0]}  PTS {key[1]}",(x,38),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),1,cv2.LINE_AA)
        pair=np.vstack((caption,np.hstack((ai,bi))))
        pair_file=output/(name+'.png');cv2.imwrite(str(pair_file),pair,[cv2.IMWRITE_PNG_COMPRESSION,2])
        bounds=_crop_bounds(subject);x0,y0,x1,y1=bounds
        crop_file=output/(name+'_native_crop.png')
        cv2.imwrite(str(crop_file),np.hstack((ai[y0:y1,x0:x1],bi[y0:y1,x0:x1])),[cv2.IMWRITE_PNG_COMPRESSION,2])
        cards.append(dict(subject,pair=ref(pair_file),native_crop=ref(crop_file),
                          crop_xyxy=bounds,native_crop_scale=1,
                          source_BGR_sha256=a['decoded_native_BGR_sha256'],
                          baseline_frame=a,candidate_frame=b))
    pieces=['<!doctype html><meta charset="utf-8"><title>六镜头固定对照</title>',
            '<style>body{font:16px sans-serif;margin:24px;background:#172028;color:#eee}img{max-width:100%;height:auto}article{margin-bottom:34px}p{line-height:1.5}</style>',
            '<h1>'+html.escape(plan.get('title','当前六镜头固定帧预览' if fixed_preview else '当前六镜头开发对照'))+'</h1>',
            '<p>'+('本页仅固化 '+html.escape(baseline_description)+'，尚无新候选改善。' if baseline_only else '左：'+html.escape(baseline_description)+'；右：本次完整候选。')+(' 仅覆盖本计划事先固定的画面与局部；全域比较与自动最坏帧尚未完成，不用于候选晋级。' if fixed_preview else '')+' 全标线、广告边界及编号使用各自完整模型；不代表独立验证或正式质量通过。</p>']
    if source_objective_changed:pieces.append('<p>目标与资格已改变：基线和候选分别消费各自绑定的数据包及材料映射。退出行仅作原始像素注记；不恢复拟合资格，不把本页当作同目标成本比较。</p>')
    for card in cards:
        candidate_source=card.get('candidate_source',{});qualification=candidate_source.get('qualification',{})
        if candidate_source.get('retired_material') or qualification.get('physical')is False:
            pieces.append('<p>该固定原始读数在候选中已退出'+('，所属模型材料已退休' if candidate_source.get('retired_material') else '')+'；保留原像素注记，不作为当前拟合点或独立验证。</p>')
        pieces.append('<article><h2>'+html.escape(f"{card['shot_id']} · PTS {card['native_pts']} · {card['role']}")+'</h2><img src="'+Path(card['pair']['file']).name+'"><p>原生 1:1 局部，同一裁切范围</p><img src="'+Path(card['native_crop']['file']).name+'"></article>')
    (output/'index.html').write_text('\n'.join(pieces),encoding='utf8')
    receipt={'schema':'rinkfit.current_sixshot_visible_comparison.v1','plan':ref(plan_file),
             'baseline_model':plan['baseline_model'],'candidate_model':candidate_ref,
             'baseline_packet':baseline_packet_ref,'candidate_packet':baseline_packet_ref if baseline_only else candidate_packet_ref,
             'baseline_current_input_v1':plan.get('baseline_current_input_v1'),'candidate_current_input_v1':plan.get('candidate_current_input_v1'),
             'objective_changed':source_objective_changed,'same_objective_cost_comparison':False if source_objective_changed else None,
             'baseline_only':baseline_only,'fixed_diagnostic_preview':fixed_preview,
             'full_raw_domain_compared':not baseline_only and not fixed_preview,
             'candidate_promotion_allowed':False,'cards':cards,
             'complete_comparison_metrics':ref(metrics) if metrics else None,
             'automatic_selection':ref(automatic) if automatic else None,
             'all_model_paths_labels_and_ice_lines_enabled':True,
             'same_original_source_raster_and_PTS':True,'fixed_frames_reselected':False,
             'independent_validation':False,'quality_accepted':False,
             'seconds':time.perf_counter()-start}
    write_json(output/'manifest.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','render'])
    for name in ('packet','baseline','scope','plan','candidate','out','automatic','metrics','compiled-dir','gpu-dll','gpu-validation'):
        p.add_argument('--'+name)
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--fixed-preview',action='store_true');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.packet,a.baseline,a.scope,a.out)
    else:r=render(a.plan,a.candidate,a.out,automatic=a.automatic,metrics=a.metrics,compiled_dir=a.compiled_dir,gpu_dll=a.gpu_dll,gpu_validation=a.gpu_validation,baseline_only=a.baseline_only,fixed_preview=a.fixed_preview)
    print(json.dumps({'output':a.out,'schema':r['schema'],'quality_accepted':False}))

if __name__=='__main__':main()
