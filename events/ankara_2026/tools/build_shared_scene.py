"""Build and inspect a scene-only structural candidate; no video or optimizer.

Uses the existing self-contained d78 bundle and the registered 8a28 missing-
inventory seed. All plotted geometry is evaluated by SceneModel, including the
unwrapped board chart; the drawings contain no replacement geometry.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from render_shared_scene import plot_top, plot_board

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'calibration/solve2'))
from rinksolve import schema as S
from rinksolve.model import (ParameterState,load_model_bundle,save_model_bundle,
    supplement_demo_initialization,structured_candidate,shared_advertisement_candidate,plain)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def scene_only(model):
    # Cameras and motion are not part of an event-level geometry review.
    specs=[b for b in model.state.blocks if b.owner.kind not in (S.OwnerKind.RIG,S.OwnerKind.SHOT)]
    # Historical per-vertex shapes remain static templates in this candidate;
    # they are not a new production shape optimization vector.
    specs=[replace(b,mode=S.VariableMode.FIXED) if b.owner.kind==S.OwnerKind.FEATURE else b for b in specs]
    keys={b.id for b in specs}
    values=tuple(v for v in model.state.values if v.variable_block_id in keys)
    state=ParameterState(model.state.id,model.state.revision,model.state.coordinate_system_id,tuple(specs),values)
    return replace(model,rigs=(),shots=(),state=state,landmarks=())


def sample_model(model):
    scene,state=model.scene,model.state
    board=scene.board(state)
    result=[]
    for curve in scene.curves:
        d=scene.resolved_definition(curve,state)
        points=d.get('unwrapped_station_height',d.get('normalized_points',d.get('points_xyz',())))
        stations=np.unique(np.r_[np.linspace(0,1,257),np.linspace(0,1,len(points))]) if len(points)>1 else np.linspace(0,1,257)
        xyz=scene.world_path(curve.id,stations,state,board=board)
        if not np.isfinite(xyz).all():raise ValueError('Nonfinite scene path '+curve.id)
        q=scene.unwrapped_path(curve.id,stations,state,board=board) if d['kind'] in ('board_material_polyline','installed_artwork_polyline','board_boundary') else None
        match=re.match(r'Ad(\d+)',curve.label)
        result.append(dict(id=curve.id,label=curve.label,kind=d['kind'],xyz=xyz,unwrapped=q,
            ad=int(match[1]) if match else None,definition=d,source=curve.definition.get('initialization_source','d78')))
    return result,board


def serial_paths(paths):
    return [{k:(v.tolist() if isinstance(v,np.ndarray) else plain(v)) for k,v in p.items()} for p in paths]


def checks(model):
    family=model.scene.structural_family;state=model.state;d=family.curves(state)
    radii=[d[f'ice.{side}{row}:outer']['radius'] for side in ['F','R'] for row in range(1,5)]
    offsets=[]
    for row in range(1,5):
        a=np.array(d[f'ice.lane{row}:edge0']['endpoints_xy']).reshape(2,2)
        b=np.array(d[f'ice.lane{row}:edge2']['endpoints_xy']).reshape(2,2)
        offsets.append(abs((a[0,1]+b[0,1])/2-d[f'ice.F{row}:outer']['center_xy'][1]))
    return dict(FR_outer_radius_range=float(np.ptp(radii)),Lane_center_max_offset=float(max(offsets)),
        C0_center=d['ice.C0:outer']['center_xy'],row_pitch=float(state.get(family.curling_block_id)[3]),
        no_independent_Lane_centers=len(state.get(family.lane_block_id))==10,
        no_independent_hockey_origin=len(state.get(family.hockey_block_id))==5,
        hockey_end_markings_shared=family.hockey_end_block_id is not None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--d78-bundle',type=Path,required=True);ap.add_argument('--demo-seed',type=Path,required=True)
    ap.add_argument('--model-out',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--replace-unreviewed',action='store_true')
    args=ap.parse_args()
    for path in [args.model_out/'model.json',args.model_out/'parameters.npz',args.model_out/'changes.json']:
        if len(str(path.resolve()).encode('utf-16-le'))//2>=260 or any(len(v.encode('utf-16-le'))//2>255 for v in path.resolve().parts):raise ValueError('OneDrive path too long')
    if (args.model_out/'model.json').exists():
        previous=json.loads((args.model_out/'changes.json').read_text(encoding='utf-8'))
        if not args.replace_unreviewed or previous.get('stage_state')!='awaiting_visual_review' or previous.get('next_stage_authorized') is not False:
            raise FileExistsError('Only this unreviewed candidate may be replaced explicitly')
    args.out.mkdir(parents=True,exist_ok=True)
    originals=[ROOT/'events/ankara_2026/programmes/okada_mei_sp/s01_original'/name for name in ['original_model.json','original_complete_state.json']]
    original_hashes={str(p):digest(p) for p in originals}
    baseline=load_model_bundle(args.d78_bundle)
    combined,composition=supplement_demo_initialization(baseline,args.demo_seed)
    candidate,migration=structured_candidate(combined)
    pair_path=ROOT/'events/ankara_2026/model_constraints/user_advertisement_pairs_20261005.json'
    candidate,ad_pairs=shared_advertisement_candidate(candidate,json.loads(pair_path.read_text(encoding='utf-8')))
    candidate=scene_only(candidate)
    candidate=replace(candidate,id='ankara_2026_shared_scene_20261024',provenance={**plain(candidate.provenance),
        'scope':'scene_structure_only','fitting_run':False,'video_replay_run':False,'stage_state':'awaiting_visual_review'})
    save_model_bundle(candidate,args.model_out)
    candidate=load_model_bundle(args.model_out)
    report=checks(candidate)
    if report['FR_outer_radius_range']!=0 or report['Lane_center_max_offset']>1e-14 or not np.array_equal(report['C0_center'],[0.,0.]):raise AssertionError(report)
    old_paths,old_board=sample_model(baseline);new_paths,board=sample_model(candidate)
    np.testing.assert_array_equal(baseline.state.get(baseline.scene.board_block_id),candidate.state.get(candidate.scene.board_block_id))
    old_curve_map={p['id']:p for p in old_paths};new_curve_map={p['id']:p for p in new_paths}
    ad_delta=max(float(np.max(np.abs(p['unwrapped']-new_curve_map[k]['unwrapped']))) for k,p in old_curve_map.items() if p['ad'] is not None)
    ad_preserved=ad_delta<1e-14
    assert ad_preserved,'Original d78 advertisement geometry changed'
    plot_top(old_paths,new_paths,board,args.out);plot_board(old_paths,new_paths,board,args.out)
    evidence=dict(schema='rinkspace.shared_scene_structure_review.v1',stage_state='awaiting_visual_review',next_stage_authorized=False,
        model=str((args.model_out/'model.json').resolve()),model_file_sha256=digest(args.model_out/'model.json'),parameter_sha256=candidate.state.digest(),
        source_originals=original_hashes,original_files_unchanged=original_hashes=={str(p):digest(p) for p in originals},
        source_d78_bundle=str(args.d78_bundle),source_demo_seed=str(args.demo_seed),composition=composition,migration=migration,
        advertisement_pairs=ad_pairs,advertisement_pair_facts=str(pair_path),
        original_d78_advertisement_max_change_scene_units=ad_delta,
        relations=report,original_d78_advertisement_curves_unchanged=ad_preserved,board_parameters_unchanged=True,
        old_paths=len(old_paths),new_paths=len(new_paths),old_ads=sorted({p['ad'] for p in old_paths if p['ad'] is not None}),new_ads=sorted({p['ad'] for p in new_paths if p['ad'] is not None}),
        fitting=False,replay=False,new_observations=0,metric_verified=False,
        limits=['仅修改结构；尺寸沿用现有模型作数值初值，尚未核对原片贴合。','d78缺少部分从已登记8a28库存及共享关系补入，不冒充新观测。','相差13的广告使用共同静态图案和尺寸；图案仍为已有初值，并非高清模板认证。','Ad25已有配对图案，当前输入缺少独立位置，未虚构位置。','展开图纵轴放大；显示分段不是物理广告分段。'])
    write(args.model_out/'changes.json',evidence);write(args.out/'facts.json',evidence)
    write(args.out/'geometry.json',dict(old=serial_paths(old_paths),new=serial_paths(new_paths),parameter_sha256=candidate.state.digest()))
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共同冰场 · 新旧实际模型</title><style>body{margin:0;background:#f6f8f5;color:#25464c;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1550px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 14px}p{max-width:1100px}.notice{background:#fff0d2;border-left:4px solid #ba842b;padding:13px 18px}.frame{background:white;padding:12px;margin:24px 0;border:1px solid #d9e2df;border-radius:10px;overflow:auto}.frame img{display:block;width:100%;height:auto;cursor:zoom-in}.frame.full img{width:auto;max-width:none;cursor:zoom-out}button{border:1px solid #a8b8b6;border-radius:5px;background:white;padding:7px 14px;cursor:pointer}code{overflow-wrap:anywhere}a{color:#216a83}</style><main><h1>共同冰场：新旧实际模型对照</h1><p class="notice">原版 d78 保留。这里只改模型结构，没有拟合、没有重放。尺寸是待核实初值，等你看完明确通过再往下做。</p><p>F/R八组圈共用半径和行距；Lane中心直接引用同一行圆心；C0固定在场地中心。蓝线、球门线、争球圈、半圆蓝区和梯形线使用共同冰球方向。围挡及 d78 已有广告模板保留；编号相差13的两面共用图案和尺寸，安装位置各自独立。</p><p>点击图可以切换到原始尺寸，再横向滚动看细节。所有曲线均来自保存的实际模型；没有另画一份替代几何。</p><div class="frame"><img src="top_comparison.png" alt="带编号新旧俯视图" onclick="this.parentElement.classList.toggle('full')"></div><div class="frame"><img src="board_comparison.png" alt="带编号整圈围挡展开图" onclick="this.parentElement.classList.toggle('full')"></div><p>新模型补足的部分来自已登记8a28几何库存与用户确认的结构，不是新增源观测。配对广告引用同一个模板，未虚构新的源观测。Ad25已有配对图案，但当前输入缺少独立安装位置，图上不虚构摆放。展开图纵轴放大以便阅读。</p><p>模型参数身份：<code>PARAMETER_HASH</code></p><p><a href="facts.json">改动与限制</a> · <a href="geometry.json">两版实际几何坐标</a></p><p><strong>现在只看新共同模型的结构和编号是否符合你的描述；没有开始下一步。</strong></p></main></html>'''
    (args.out/'index.html').write_text(content.replace('PARAMETER_HASH',candidate.state.digest()),encoding='utf-8')
    print(json.dumps({k:evidence[k] for k in ['model','parameter_sha256','relations','old_paths','new_paths','old_ads','new_ads','original_files_unchanged','original_d78_advertisement_curves_unchanged']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
