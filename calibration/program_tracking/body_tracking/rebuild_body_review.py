"""Frozen complete-scene source diagnostics for the S01 body ablation."""
from pathlib import Path
from fractions import Fraction
import argparse,json,collections
import numpy as np
import cv2
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.review.render import source_frames
from rinkspace.review.draw import COLORS,text
from .rebuild_prepare import load,digest
from .rebuild_body_fit import FixedNativeProjector
from .rebuild_body_audit import MODES

MODE_COLORS=((80,190,255),(90,240,100),(255,150,90))
LABELS=('A fixed pose','B local pose corrections','C pose + contact')
BONES=((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def scene_cache(root,meta):
    xyz_path=root/'review_scene.npz';json_path=root/'review_scene.json'
    if xyz_path.exists():return load(xyz_path)['xyz'],json.loads(json_path.read_text(encoding='utf8'))
    for name,sha in meta['frozen_inputs'].items():
        if digest(name)!=sha:raise ValueError('Frozen input changed before review: '+name)
    state=load(root/'camera_snapshot.npz');scene_path=Path(meta['scene_path']);raw=json.loads(scene_path.read_text(encoding='utf8'))
    core=NativeCore(meta['native_library']);model=SceneParameterModel(raw,scene_path,optimize_scene=True)
    param=ParameterState(model.initial_state.layout,state['scene_values']);native=NativeSceneGPU(scene_path,core,parameter_state=param)
    mapping=json.loads((Path(meta['run_root'])/'mapping.json').read_text(encoding='utf8'))
    samples=sample_native_review_scene(raw,core,scene_path,parameter_state=param,observed_path_ids=set(mapping['path_ids']),native=native)
    info=dict(cuts=samples['cuts'],inventory=samples['inventory'],unlocalized=samples['missing_entities'],board=np.asarray(native.scene['board']).tolist(),
        scene_sha256=digest(scene_path),camera_snapshot_sha256=digest(root/'camera_snapshot.npz'),unobserved_print_surface_extensions=samples['unobserved_print_surface_extensions'])
    np.savez_compressed(xyz_path,xyz=samples['xyz']);json_path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    return samples['xyz'],info


def review(root,project):
    root,project=Path(root),Path(project);p=load(root/'input.npz');meta=json.loads((root/'input.json').read_text(encoding='utf8'))
    source_spec=json.loads((project/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json').read_text(encoding='utf8'))
    source={**source_spec['source'],**source_spec['source_clock']};results={m:load(root/m/'result.npz') for m in MODES}
    scene,scene_info=scene_cache(root,meta);camera=FixedNativeProjector(p,meta['native_library'])
    points_path=Path(meta['run_root'])/'point_observations.npz'
    manifest=json.loads((Path(meta['run_root'])/'joint_manifest.json').read_text(encoding='utf8'))
    if digest(points_path)!=manifest['point_input_sha256']:raise ValueError('Scene point input changed')
    with np.load(points_path) as file:point_pts=file['pts'];point_uv=file['uv']
    wanted=[321280,346624,372224,410368,461056,512000]
    delta=np.linalg.norm(results['pose_contact']['G_world_m']-results['fixed']['G_world_m'],axis=1)
    delta[p['lower_body_inferred']]=-1;worst=int(p['pts'][delta.argmax()]);wanted=sorted(set(wanted+[worst,int(p['pts'][0]),int(p['pts'][-1])]))
    groups=collections.defaultdict(list)
    for cut in scene_info['cuts']:
        if cut['layer']=='advertisement':groups[cut['id'].split('::',1)[0]].extend(range(cut['begin'],cut['end']))
    if digest(source['registered_path'])!=source['sha256']:raise ValueError('Official source identity changed')
    records=[]
    for pts,raw in source_frames(source['registered_path'],source,p['pts']):
        if pts not in wanted:continue
        fi=int(np.searchsorted(p['pts'],pts));time=p['motion_times'][fi]
        projected=camera.core.project_motion(p['rig'],p['controls'],float(p['first_control_time']),float(p['control_dt']),1080,np.full(len(scene),time),scene)
        uv=projected['uv'];good=(projected['status']==0)&np.isfinite(uv).all(1);pixels=np.rint(np.clip(np.nan_to_num(uv),-1e7,1e7)).astype(np.int32)
        base=raw.copy()
        for cut in scene_info['cuts']:
            a,b=cut['begin'],cut['end'];color=(220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
            for lo,hi in np.flatnonzero(np.diff(np.r_[False,good[a:b],False])).reshape(-1,2):
                if hi-lo>1:cv2.polylines(base,[pixels[a+lo:a+hi]],bool(cut.get('closed') and lo==0 and hi==b-a),color,1,cv2.LINE_AA)
        for identity,ids in groups.items():
            ids=np.asarray(ids);visible=ids[good[ids]&(uv[ids,0]>=0)&(uv[ids,0]<1920)&(uv[ids,1]>=0)&(uv[ids,1]<1080)]
            if len(visible):text(base,identity.rsplit('/',1)[-1],np.rint(np.median(uv[visible],0)).astype(int),.38,COLORS['advertisement'])
        active=np.flatnonzero(point_pts==pts)
        for at in point_uv[active]:cv2.circle(base,tuple(np.rint(at).astype(int)),2,(0,0,255),-1,cv2.LINE_AA)
        panels=[]
        for mi,mode in enumerate(MODES):
            z=results[mode];color=MODE_COLORS[mi];img=base.copy();prediction=z['COCO_uv'][fi]
            for a,b in BONES:cv2.line(img,tuple(np.rint(prediction[a]).astype(int)),tuple(np.rint(prediction[b]).astype(int)),color,2,cv2.LINE_AA)
            for xy in p['source_uv'][fi,p['source_qualified'][fi]]:cv2.circle(img,tuple(np.rint(xy).astype(int)),4,(20,240,240),1,cv2.LINE_AA)
            for foot in range(2):cv2.polylines(img,[np.rint(z['blade_uv'][fi,foot]).astype(int)],False,color,3,cv2.LINE_AA)
            P=z['P_world_m'][fi];line=P[None]+np.linspace(0,1,33)[:,None]*np.array([0,0,-P[2]])
            projected_line=camera.project(line[None],np.array([fi]))[0]
            cv2.polylines(img,[np.rint(projected_line).astype(int)],False,color,3,cv2.LINE_AA)
            for xy,label in [(projected_line[0],'P'),(projected_line[-1],'G')]:
                at=tuple(np.rint(xy).astype(int));cv2.circle(img,at,8,color,2,cv2.LINE_AA);cv2.rectangle(img,tuple(np.asarray(at)+[10,-31]),tuple(np.asarray(at)+[46,0]),(12,12,12),-1);text(img,label,np.asarray(at)+[14,-7],.85,(255,255,255))
            display=cv2.resize(img,(960,540),interpolation=cv2.INTER_AREA)
            header=np.full((96,960,3),22,np.uint8);text(header,LABELS[mi],(14,28),.65,color)
            text(header,f'Source {pts/12800:.3f}s | PTS {pts} | pelvis height {P[2]:.3f} conditional m',(14,53),.49)
            status='LOWER BODY INFERRED' if p['lower_body_inferred'][fi] else 'contact unknown except six reviewed instants'
            text(header,status+' | yellow circles: detector UV',(14,79),.43)
            panels.append(np.vstack((header,display)))
        side=np.full((636,960,3),22,np.uint8);scale=min(700/(2*float(scene_info['board'][0])),410/(2*float(scene_info['board'][1])))
        def top(xyz):return np.rint(np.asarray(xyz)[...,:2]*[scale,-scale]+[480,285]).astype(np.int32)
        for cut in scene_info['cuts']:
            color=(220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
            cv2.polylines(side,[top(scene[cut['begin']:cut['end']])],bool(cut.get('closed')),color,1,cv2.LINE_AA)
        for identity,ids in groups.items():
            centre=scene[ids].mean(0);at=top(centre);text(side,identity.rsplit('/',1)[-1],at,.3,COLORS['advertisement'])
        for mi,mode in enumerate(MODES):
            z=results[mode];color=MODE_COLORS[mi];path=z['G_world_m']/float(p['metres_per_scene_unit']);cv2.polylines(side,[top(path)],False,tuple(int(v*.5) for v in color),1,cv2.LINE_AA)
            at=top(path[fi]);cv2.circle(side,tuple(at),6,color,2,cv2.LINE_AA);text(side,'ABC'[mi],at+[8,8],.5,color)
            text(side,f'{LABELS[mi]}: G=({z["G_world_m"][fi,0]:+.3f}, {z["G_world_m"][fi,1]:+.3f}) conditional m',(20,460+30*mi),.48,color)
        text(side,'Complete frozen scene / top view / three pelvis projections',(20,28),.56)
        text(side,f'C minus A horizontal distance: {max(delta[fi],0):.3f} conditional m',(20,585),.5)
        text(side,'60 m/L, body stature and skate shape assumed; not 3D ground truth',(20,614),.44)
        canvas=np.vstack((np.hstack(panels[:2]),np.hstack((panels[2],side))))
        output=root/f'review_{pts}.jpg';cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
        if pts in [410368,421888,461056,512000]:
            # Magnify the same rendered native pixels, keeping the full scene
            # counterpart above. This does not generate a new observation.
            xy=p['source_uv'][fi,p['source_qualified'][fi]]
            lo=np.maximum(np.floor(xy.min(0)/2).astype(int)-[65,50],[0,0]);hi=np.minimum(np.ceil(xy.max(0)/2).astype(int)+[75,65],[959,539])
            detail=[]
            for mi,panel in enumerate(panels):
                clip=panel[96+lo[1]:96+hi[1],lo[0]:hi[0]]
                box=np.full((850,600,3),22,np.uint8);factor=min(580/clip.shape[1],740/clip.shape[0]);enlarged=cv2.resize(clip,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
                box[65:65+enlarged.shape[0],10:10+enlarged.shape[1]]=enlarged
                text(box,LABELS[mi],(12,30),.64,MODE_COLORS[mi]);text(box,f'Source {pts/12800:.3f}s | P: pelvis | G: vertical projection',(12,54),.4)
                detail.append(box)
            cv2.imwrite(str(root/f'detail_{pts}.png'),np.hstack(detail))
        records.append(dict(source_pts=pts,source_seconds=pts/12800,output=str(output),active_scene_points=len(active),invalid_scene_projection_points=int((~good).sum()),lower_body_inferred=bool(p['lower_body_inferred'][fi])))
    (root/'review_images.json').write_text(json.dumps(dict(frames=records,scene_paths=len(scene_info['cuts']),scene_points=scene_info['inventory'],source_sha256=source['sha256'],camera_and_scene_frozen=True),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);a=ap.parse_args();review(a.root,a.project)
