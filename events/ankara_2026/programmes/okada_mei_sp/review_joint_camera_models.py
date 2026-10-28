"""Source-PTS comparison stills using the same native scene and camera projection."""
from pathlib import Path
import argparse,collections,json,html
import numpy as np
import cv2
import prepare_native_joint_shots as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.review.scene import sample_review_scene
from rinkspace.review.render import source_frames
from rinkspace.review.draw import COLORS

DEFAULT_PTS=[283136,373760,491776,573440,575488,745984,789504,878080,948736]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*');parser.add_argument('--pts',nargs='*',type=int);args=parser.parse_args()
    root=prepared.RUN;out=root/'camera_models';summary=prepared.read(out/'comparison.json');manifest=prepared.read(root/'joint_manifest.json');points=prepared.npz(root/'point_observations.npz');mapping=prepared.read(root/'mapping.json')
    core=NativeCore(manifest['native_library']);raw_scene=prepared.read(manifest['scene_path']);model=SceneParameterModel(raw_scene,Path(manifest['scene_path']),optimize_scene=True);scope=prepared.read(prepared.HERE/'rebuild_scope.json');source={**scope['source'],**scope['source_clock']}
    selected=args.pts or DEFAULT_PTS
    for shot in manifest['shots']:
        candidates=[p for p in selected if shot['first_pts']<=p<=shot['last_pts']]
        if not candidates:continue
        for pt,raw in source_frames(source['registered_path'],source,np.arange(min(candidates),max(candidates)+256,256,dtype=np.int64)):
            pts=int(pt)
            if pts not in candidates:continue
            rows=np.flatnonzero(points['pts']==pts)
            for rec in summary['models']:
                key=rec['model']
                if args.models and key not in args.models and key!='accepted_baseline':continue
                folder=out/'visuals'/key;folder.mkdir(parents=True,exist_ok=True);dest=folder/f'{pts}.jpg'
                receipt=dest.with_suffix('.json')
                if dest.exists() and receipt.exists() and prepared.read(receipt)['snapshot_sha256']==rec['snapshot_sha256']:continue
                v=prepared.npz(rec['snapshot']);state=ParameterState(model.initial_state.layout,v['scene_values']);native=NativeSceneGPU(manifest['scene_path'],core,parameter_state=state)
                samples=sample_review_scene(raw_scene,core,Path(manifest['scene_path']),parameter_state=state,observed_path_ids=set(np.asarray(mapping['path_ids'])[points['path_indices']]))
                chunks=[];cuts=[];offset=0
                for cut in samples['cuts']:
                    path=native.paths.get(cut['id']);xyz=samples['xyz'][cut['begin']:cut['end']]
                    if path is not None and cut['layer']!='advertisement':
                        stations=np.linspace(0,core.board_perimeter(native.scene['board']),4097) if path['kind']==7 else np.linspace(*(path.get('support') or [0.,1.]),1025 if path['kind'] in (1,6) else 129)
                        xyz=np.asarray(core.sample_path(native.scene['board'],path['kind'],path['parameters'],path.get('vertices',[]),stations))
                    chunks.append(xyz);cuts.append({**cut,'begin':offset,'end':offset+len(xyz)});offset+=len(xyz)
                world=np.vstack(chunks);controls=v['controls'][shot['control_begin']:shot['control_stop']];projected=core.project_motion(v['rig'],controls,shot['first_control_time'],manifest['control_dt'],1080,(pts-shot['reference_pts'])/12800,world)
                good=(projected['status']==0)&np.isfinite(projected['uv']).all(1);pixels=np.rint(np.clip(np.nan_to_num(projected['uv']),-1e7,1e7)).astype(np.int32);picture=raw.copy()
                for cut in cuts:
                    a,b=cut['begin'],cut['end'];color=(220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
                    for lo,hi in np.flatnonzero(np.diff(np.r_[False,good[a:b],False])).reshape(-1,2):
                        if hi-lo>1:cv2.polylines(picture,[pixels[a+lo:a+hi]],bool(cut.get('closed') and lo==0 and hi==b-a),color,1,cv2.LINE_AA)
                ev=prepared.npz(root/'joint_repaired/evaluation_pixels.npz' if key=='accepted_baseline' else out/key/'evaluation_pixels.npz');fit=ev['values'][rows,3:5];err=np.linalg.norm(ev['values'][rows,9:11],axis=1)
                for measured,fitted in zip(points['uv'][rows],fit):
                    a=tuple(np.rint(measured).astype(int));b=tuple(np.rint(fitted).astype(int));cv2.circle(picture,a,2,(0,0,255),-1,cv2.LINE_AA);cv2.line(picture,a,b,(30,130,240),1,cv2.LINE_AA);cv2.drawMarker(picture,b,(60,255,70),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
                header=np.full((72,1920,3),18,np.uint8);cv2.putText(header,f'{key} | {shot["id"].upper()} source {pts/12800:.3f}s PTS {pts}',(16,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA);cv2.putText(header,f'RMS {np.sqrt(np.mean(err*err)):.3f}px | max {err.max():.2f}px | complete native grid + current source points',(16,56),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,220,255),1,cv2.LINE_AA)
                cv2.imwrite(str(dest),np.vstack((header,picture)),[cv2.IMWRITE_JPEG_QUALITY,94]);prepared.write(receipt,dict(snapshot_sha256=rec['snapshot_sha256'],source_pts=pts,source_sha256=source['sha256'],native_library_sha256=manifest['native_library_sha256'],complete_native_grid=True,only_current_source_points=True));print(key,pts,str(dest),flush=True)
    available={rec['model']:[p for p in selected if (out/'visuals'/rec['model']/f'{p}.jpg').exists()] for rec in summary['models']}
    models=[k for k,v in available.items() if v];data=json.dumps(dict(models=models,pts=selected),ensure_ascii=False)
    page='<!doctype html><meta charset="utf-8"><title>1＋4 相机模型原片对照</title><style>body{margin:20px;font:16px system-ui;background:#edf1f5;color:#123}select{padding:8px;margin:10px}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}a{color:#075ca3}p{line-height:1.6}</style><h1>同源曝光：保留基线与候选模型</h1><p>左侧为用户已认可的基线，右侧为候选。每幅均显示完整已建模网格、本曝光原始读点及拟合点。点击图片查看原分辨率。这里只是有目的的局部比较；完整1936曝光的数值检查见统计报告，不能以这些截图代替连续视觉验收。</p><a href="comparison.html">数值和参数对照</a><br><label>候选 <select id="model"></select></label><label>源时间 <select id="pts"></select></label><main><a id="leftlink" target="_blank"><img id="left"></a><a id="rightlink" target="_blank"><img id="right"></a></main><script>'+f'const data={data};'+"const m=document.querySelector('#model'),p=document.querySelector('#pts');data.models.filter(x=>x!=='accepted_baseline').forEach(x=>m.add(new Option(x,x)));data.pts.forEach(x=>p.add(new Option((x/12800).toFixed(3)+' s',x)));function update(){let a='visuals/accepted_baseline/'+p.value+'.jpg',b='visuals/'+m.value+'/'+p.value+'.jpg';left.src=a;leftlink.href=a;right.src=b;rightlink.href=b;}m.onchange=p.onchange=update;update();</script>"
    (out/'visual_comparison.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
