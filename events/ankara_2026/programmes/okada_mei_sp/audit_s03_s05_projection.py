"""Auxiliary native camera review of S03/S05; source pixels and scene stay fixed.

The camera optimization delegates to the existing native GPU camera runner.
Its result only locates current geometry for source inspection.
"""
from pathlib import Path
import argparse, collections, io, json, shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import points_s03_s05 as source
import fit_camera_s02 as fit
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.vision_native import file_ref
from rinkspace.review.scene import sample_review_scene
from rinkspace.review.point_connections import polyline_order,entity_color


def configure(shot):
    source.configure(shot)
    fit.source=source.common.shared
    fit.OUT=source.OUT/'evidence/native_overlay_review/camera'
    return fit.OUT


def prepare():
    out=fit.OUT;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    peer=source.PEER;manifest=fit.read(peer/'joint_manifest.json')
    snapshot=peer/'goal_end_constraints/joint_fit_fixed/accepted_pixels_latest.npz'
    blob=snapshot.read_bytes();origin=file_ref(snapshot)
    if blob!=snapshot.read_bytes():raise ValueError('Peer snapshot changed during read')
    state=dict(np.load(io.BytesIO(blob),allow_pickle=False))
    if str(state['point_input_sha256'])!=manifest['point_input_sha256']:
        raise ValueError('Peer state and current input do not match')
    scene_path=Path(manifest['scene_path']);scene=fit.read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    deps=json.loads(str(state['joint_numeric_dependencies_json']))
    assert deps['scene_revision']==model.initial_state.layout['scene_content_revision']
    assert deps['layout_revision']==model.initial_state.layout['layout_revision']
    current=ParameterState(model.initial_state.layout,state['scene_values']).apply_scene(scene)
    assert np.array_equal(current['board'],state['board']) and np.array_equal(current['shared_layout'],state['layout'])
    pz=np.load(peer/'point_observations.npz');pm=fit.read(peer/'mapping.json')
    counts=collections.Counter(pm['path_ids'][int(i)] for i in pz['path_indices'])
    z=np.load(source.OUT/'points/point_observations.npz');mapping=fit.read(source.OUT/'points/mapping.json')
    coverage=[];selected=[]
    for i,pid in enumerate(mapping['path_ids']):
        ids=np.flatnonzero(z['path_indices']==i)
        eligible=not pid.startswith('print_instance/') and counts[pid]>0
        if eligible:selected.extend(ids.tolist())
        coverage.append(dict(path_id=pid,rows=len(ids),peer_direct_rows=counts[pid],camera_initialization=eligible,
            reason='observed_ice_or_board_navigation_anchor' if eligible else 'shown_for_review_not_used_as_camera_anchor'))
    selected=np.asarray(sorted(selected),np.int64)
    np.save(out/'camera_rows.npy',selected)
    np.savez(out/'geometry_seed.npz',scene_values=state['scene_values'],board=state['board'],layout=state['layout'])
    camera=fit.read(source.CAMERA);fit.write(out/'camera_seed.json',camera)
    np.save(out/'controls_seed.npy',np.load(source.CAMERA.parent/camera['controls_ref']))
    refs=[file_ref(p)for p in [scene_path,Path(manifest['native_library']),source.OUT/'points/point_observations.npz',
        source.CAMERA,source.CAMERA.parent/camera['controls_ref'],out/'geometry_seed.npz',out/'camera_rows.npy',out/'camera_seed.json',out/'controls_seed.npy']]
    refs.extend(file_ref(scene_path.parent/ref['path'])for ref,_ in model.assets)
    fit.write(out/'geometry_coverage.json',dict(paths=coverage,rows=len(selected),all_rows=len(z['pts']),
        all_source_rows_retained=True,navigation_anchors_not_accuracy_acceptance=True))
    fit.write(out/'manifest.json',dict(scene_path=str(scene_path),native_library=manifest['native_library'],
        native_library_sha256=file_ref(manifest['native_library'])['sha256'],geometry_origin=origin,
        peer_input_sha256=manifest['point_input_sha256'],layout_revision=deps['layout_revision'],
        point_input_sha256=mapping['point_input_sha256'],source_video=mapping['source_video'],
        source_pts=[source.FIRST,source.LAST],source_time_base='1/12800',protected=refs,
        geometry_frozen_during_camera_solve=True,auxiliary_only=True))
    print(json.dumps(dict(event='prepared',shot=source.SHOT,rows=len(selected),frames=len(np.unique(z['pts'][selected])))),flush=True)


def render(state_path,pts):
    m,z,mapping,g,core,native=fit.load()
    camera=fit.read(fit.OUT/'camera_seed.json')
    if state_path:
        state=dict(np.load(state_path,allow_pickle=False))
        assert np.array_equal(state['scene_values'],g['scene_values'])
    else:state=dict(rig=np.asarray(camera['rig']),controls=np.load(fit.OUT/'controls_seed.npy'))
    samples=sample_review_scene(fit.read(m['scene_path']),core,Path(m['scene_path']),parameter_state=native.parameter_state,observed_path_ids=set())
    target=fit.OUT.parent/('adjusted'if state_path else'current_geometry_seed');target.mkdir(exist_ok=True)
    for pt in pts:
        im=cv2.imread(str(source.OUT/f'source_frames/{pt}.png'))
        r=core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],1080,(pt-camera['reference_pts'])/12800,samples['xyz'])
        for path in samples['cuts']:
            q=r['uv'][path['begin']:path['end']];ok=(r['status'][path['begin']:path['end']]==0)&np.isfinite(q).all(1)&(abs(q).max(1)<1e5)
            ids=np.flatnonzero(ok)
            for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                if len(run)>1:cv2.polylines(im,[np.rint(q[run]).astype(np.int32)],False,(80,245,125),1,cv2.LINE_AA)
        for uv in z['uv'][z['pts']==pt]:cv2.circle(im,tuple(np.rint(uv).astype(int)),2,(0,100,255),-1)
        cv2.putText(im,f'{source.SHOT.upper()} {pt/12800:.3f}s PTS {pt}  CURRENT NATIVE / CAMERA ONLY',(15,1063),0,.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(target/f'{pt}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95])
    print(json.dumps(dict(event='rendered',folder=str(target),pts=pts)),flush=True)


def diagnose(state_path):
    m,z,mapping,g,core,native=fit.load();camera=fit.read(fit.OUT/'camera_seed.json')
    state=dict(np.load(state_path,allow_pickle=False));assert np.array_equal(state['scene_values'],g['scene_values'])
    scene=fit.read(m['scene_path']);samples=sample_review_scene(scene,core,Path(m['scene_path']),parameter_state=native.parameter_state,observed_path_ids=set())
    cuts={p['id']:p for p in samples['cuts']}
    entities={p['id']:p for p in fit.read(source.OUT/'evidence/native_entities.json')['entities']}
    dist=np.full(len(z['pts']),np.nan);order=dist.copy();records=[]
    by_frame=collections.defaultdict(list)
    for i,pt in enumerate(z['pts']):by_frame[int(pt)].append(i)
    for frame_no,(pt,ii) in enumerate(by_frame.items()):
        ii=np.asarray(ii);r=core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],1080,(pt-camera['reference_pts'])/12800,samples['xyz'])
        for pi in np.unique(z['path_indices'][ii]):
            ids=ii[z['path_indices'][ii]==pi];pid=mapping['path_ids'][pi];cut=cuts[pid]
            q=r['uv'][cut['begin']:cut['end']];ok=(r['status'][cut['begin']:cut['end']]==0)&np.isfinite(q).all(1)&(abs(q).max(1)<1e6)
            valid=np.flatnonzero(ok);runs=[v for v in np.split(valid,np.flatnonzero(np.diff(valid)>1)+1)if len(v)>1]
            if not runs:continue
            matches=[polyline_order(z['uv'][ids],q[run])for run in runs]
            ds=np.asarray([v[1]for v in matches]);best=ds.argmin(0);d=ds[best,np.arange(len(ids))]
            o=np.asarray([v[0]for v in matches])[best,np.arange(len(ids))];dist[ids]=d;order[ids]=o
            shape=entities[pid];entry=dict(pts=pt,path_id=pid,label=shape['label'],layer=shape['layer'],kind=shape['kind'],rows=len(ids),
                median=float(np.median(d)),p95=float(np.percentile(d,95)),maximum=float(d.max()),worst_row=int(ids[d.argmax()]))
            if shape['kind']=='ice_segment'and len(ids)>2:
                uv=z['uv'][ids];center=np.mean(uv,axis=0);_,_,v=np.linalg.svd(uv-center,full_matrices=False)
                normal=abs((uv-center)@v[-1]);entry.update(straightness_p95=float(np.percentile(normal,95)),straightness_maximum=float(normal.max()))
            records.append(entry)
        if frame_no%100==0:print(json.dumps(dict(event='diagnosed',shot=source.SHOT,frames=frame_no+1,total=len(by_frame))),flush=True)
    dest=fit.OUT.parent;np.savez_compressed(dest/'auxiliary_curve_distances.npz',distance=dist,order=order)
    summary=[]
    for pi,pid in enumerate(mapping['path_ids']):
        ids=np.flatnonzero(z['path_indices']==pi);good=ids[np.isfinite(dist[ids])];e=entities[pid]
        rr=[r for r in records if r['path_id']==pid]
        if not len(good):continue
        summary.append(dict(path_id=pid,label=e['label'],layer=e['layer'],kind=e['kind'],rows=len(ids),frames=len(rr),
            median=float(np.median(dist[good])),p95=float(np.percentile(dist[good],95)),maximum=float(dist[good].max()),
            worst_frames=sorted(rr,key=lambda r:r['maximum'],reverse=True)[:6]))
    fit.write(dest/'diagnostic.json',dict(point_input_sha256=mapping['point_input_sha256'],camera=file_ref(state_path),geometry=m['geometry_origin'],
        frames=len(by_frame),records=records,summary=summary,diagnostic_only=True,no_automatic_source_changes=True))
    print(json.dumps(dict(event='diagnostic_ready',shot=source.SHOT,ice=[{k:v for k,v in s.items()if k!='worst_frames'}for s in summary if s['layer']=='ice'])),flush=True)


def material_panels(state_path,pids,pts=None):
    m,z,mapping,g,core,native=fit.load();camera=fit.read(fit.OUT/'camera_seed.json');state=dict(np.load(state_path,allow_pickle=False))
    samples=sample_review_scene(fit.read(m['scene_path']),core,Path(m['scene_path']),parameter_state=native.parameter_state,observed_path_ids=set())
    cuts={p['id']:p for p in samples['cuts']};diag=fit.read(fit.OUT.parent/'diagnostic.json');summary={r['path_id']:r for r in diag['summary']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    target=fit.OUT.parent/'materials';target.mkdir(exist_ok=True)
    for pid in pids:
        pi=mapping['path_ids'].index(pid);allpts=np.unique(z['pts'][z['path_indices']==pi]);ss=summary[pid]
        selected=pts or sorted(set([int(allpts[0]),int(allpts[len(allpts)//2]),int(allpts[-1]),ss['worst_frames'][0]['pts']]))
        tiles=[]
        for pt in selected:
            im=cv2.imread(str(source.OUT/f'source_frames/{pt}.png'))
            r=core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],1080,(pt-camera['reference_pts'])/12800,samples['xyz'])
            for other in samples['cuts']:
                q=r['uv'][other['begin']:other['end']]
                if not np.isfinite(q).all()or abs(q).max()>1e5:continue
                cv2.polylines(im,[np.rint(q).astype(np.int32)],False,(60,120,60),1,cv2.LINE_AA)
            cut=cuts[pid];q=r['uv'][cut['begin']:cut['end']];uv=z['uv'][(z['pts']==pt)&(z['path_indices']==pi)]
            if np.isfinite(q).all()and abs(q).max()<1e5:cv2.polylines(im,[np.rint(q).astype(np.int32)],False,(0,255,255),2,cv2.LINE_AA)
            for v in uv:cv2.circle(im,tuple(np.rint(v).astype(int)),5,(10,80,255),-1,cv2.LINE_AA)
            # Cropping keeps the actual observed region, not a displaced mesh.
            if len(uv):
                lo=np.maximum(np.floor(uv.min(0)-[100,80]).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)+[100,80]).astype(int),[1920,1080]);box=im[lo[1]:hi[1],lo[0]:hi[0]]
            else:box=im
            scale=min(960/box.shape[1],450/box.shape[0]);resized=cv2.resize(box,None,fx=scale,fy=scale);tile=np.zeros((500,960,3),np.uint8);tile[50:50+resized.shape[0],:resized.shape[1]]=resized
            pic=Image.fromarray(cv2.cvtColor(tile,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pic);d.text((10,8),f'{ss["label"]} {pt/12800:.3f}s PTS {pt} | {len(uv)}点',font=font,fill='white');tiles.append(cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR))
        while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
        name=pid.replace('/','_').replace(':','_');cv2.imwrite(str(target/f'{name}.jpg'),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,len(tiles),2)]),[cv2.IMWRITE_JPEG_QUALITY,96])
    print(json.dumps(dict(event='material_panels',folder=str(target),paths=pids)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('shot',choices=['s03','s05']);p.add_argument('stage',choices=['prepare','solve','render','diagnose','materials'])
    p.add_argument('--phase',default='navigation',choices=['motion','pose','image_intrinsics','navigation'])
    p.add_argument('--chart',choices=['pose','image','all'],default='all')
    p.add_argument('--resume',type=Path);p.add_argument('--state',type=Path);p.add_argument('--pts',nargs='+',type=int)
    p.add_argument('--paths',nargs='+')
    args=p.parse_args();cv2.setNumThreads(2);configure(args.shot);prepare()
    if args.stage=='solve':
        if args.phase=='navigation' and args.chart!='all':
            args.camera_parameter_indices=[0,1,2,6,7,8]+([15,16,17]if args.chart=='image'else[])
            args.output_name='navigation_'+args.chart
        fit.solve(args)
    elif args.stage=='render':render(args.state,args.pts or [source.FIRST,source.FIRST+((source.LAST-source.FIRST)//512)*256,source.LAST])
    elif args.stage=='diagnose':diagnose(args.state)
    elif args.stage=='materials':material_panels(args.state,args.paths,args.pts)


if __name__=='__main__':main()
