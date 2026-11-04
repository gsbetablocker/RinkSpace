"""Render actual SMPL-X and attached skate surfaces onto every native source frame.

No per-model image alignment, PnP, camera adjustment or pose fitting is allowed
here. The only pixel transform is one shared source-defined comparison crop.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import cv2
import av
from PIL import Image,ImageDraw,ImageFont
from rinkspace.storage import AssetStore
from rinkspace.viewer3d.source import SourceVideo
from .support_root_placement import restore_parent,SupportLocatedReference
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection
from .footwear_binding import transform


def silhouette(uv,faces,shape):
    mask=np.zeros(shape,np.uint8)
    points=np.rint(np.clip(uv,-100000,100000)).astype(np.int32)
    triangles=points[faces];lo=triangles.min(1);hi=triangles.max(1)
    visible=(hi[:,0]>=0)&(hi[:,1]>=0)&(lo[:,0]<shape[1])&(lo[:,1]<shape[0])&((hi-lo)<10000).all(1)
    for tri in triangles[visible]:cv2.fillConvexPoly(mask,tri,255,lineType=cv2.LINE_8)
    return mask


def paint(raw,uv,faces,shoe_uv,shoe_faces,label,font,pts):
    a=np.asarray(raw).copy();mask=silhouette(uv,faces,a.shape[:2]);inside=mask>0
    a[inside]=(.64*a[inside]+.36*np.array([30,225,145])).astype(np.uint8)
    contour,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(a,contour,-1,(10,235,180),2)
    for points,triangles in zip(shoe_uv,shoe_faces):
        m=silhouette(points,triangles,a.shape[:2]);hit=m>0
        a[hit]=(.60*a[hit]+.40*np.array([255,145,25])).astype(np.uint8)
        c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(a,c,-1,(255,155,35),2)
    im=Image.fromarray(a);draw=ImageDraw.Draw(im);draw.rectangle((0,0,1920,43),fill=(12,16,23));draw.text((20,5),label+' | source '+format(pts/12800.,'.2f')+'s | PTS '+str(pts),font=font,fill='white')
    return im


def open_encoder(path,width,height):
    c=av.open(str(path),'w');s=c.add_stream('libx264',rate=50);s.width=width;s.height=height;s.pix_fmt='yuv420p';s.options={'crf':'19','preset':'fast'};s.codec_context.thread_count=4
    return c,s


def encode(c,s,image,i):
    f=av.VideoFrame.from_image(image);f.pts=i
    for packet in s.encode(f):c.mux(packet)


def render(project,base,output,seconds=(76.,95.),batch=4,candidate_name='support_midpoint'):
    tick=time.perf_counter();target=output/('native_camera_overlay_midpoint' if candidate_name=='support_midpoint' else 'native_camera_overlay');target.mkdir(parents=True,exist_ok=True)
    paths=[target/'support_native_overlay_76_95.mp4',target/'before_after_mesh_76_95.mp4']
    if any(p.exists() for p in paths):raise FileExistsError('Preserve existing actual-camera overlay; choose a new run')
    active=json.loads((output/'active_candidate.json').read_text(encoding='utf8'))
    parent,camera,info,saved=restore_parent(project,base,output,target,Path(active['parameter']))
    params=load(project/'events/ankara_2026/programmes/okada_mei_sp'/('spin_'+candidate_name+'_1104.npz'))
    document=json.loads(str(params['document_json']));located=SupportLocatedReference(parent,params['anchor_pts'],params['anchor_world_m'],document)
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))['source']['path']
    native=load(base/'action_integration_1103/source_cache/spin_s05.native50.npz');bbox=native['bbox'];good=np.isfinite(bbox).all(1)
    centres=np.interp(parent.p['pts'],native['pts'][good],(bbox[good,0]+bbox[good,2])/2)
    ids=np.flatnonzero((parent.p['pts']>=round(seconds[0]*12800))&(parent.p['pts']<=round(seconds[1]*12800)))
    surface=parent.binding.surface;faces=surface.faces;packet=parent.binding.packet();template=packet['template'];lower=np.asarray(template['vertices']);lower_faces=np.asarray(template['faces'],np.int32);upper=np.asarray(packet['upper']);upper_faces=parent.binding.faces
    sfont=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    encoders=[open_encoder(p,1920,1080) for p in paths];source_reader=SourceVideo(source,(1,12800));count=0;clock=[];proof={};node_count=0;max_coco_error=0.
    parent_result=load(active['result']);located_result=load(output/candidate_name/'fitted/result.npz')
    try:
        for first in range(0,len(ids),batch):
            frames=ids[first:first+batch];t=parent.times[frames];shutter=float(camera.p['rig'][37]);times=(t[:,None]+np.linspace(-.5,.5,5)*shutter).ravel();n=len(frames)
            original=parent.at(times);shifted=located.at(times);s=fk_state(parent,original,np.zeros((len(times),69)))
            kw={} if 'hand_local_R' not in s else {'hand_local_R':s['hand_local_R']}
            vertices=surface.sample_vertices(s['native'],s['frames'],double_precision=True,**kw)
            # One actual articulation for the before/after root comparison.
            # No poses are interpolated from screenshots or display caches.
            world=np.einsum('nij,nvj->nvi',s['Q'],vertices)+s['P'][:,None]
            delta=shifted['P']-original['P'];deform=parent.binding.deformation(s['native'],s['frames']);upper_local=parent.binding.deform(upper,deform)
            shoe=np.concatenate([transform(s['R'],s['O'],lower),transform(s['R'],s['O'],upper_local)],axis=2)
            node_count+=len(times);projections=[];shoe_projections=[]
            for di in (np.zeros_like(delta),delta):
                uv=moving_projection(camera,(world+di[:,None]).reshape(n,5,-1,3),frames)['uv'];projections.append(uv)
                su=moving_projection(camera,(shoe+di[:,None,None]).reshape(n,5,-1,3),frames)['uv'].reshape(n,2,len(lower)+len(upper),2);shoe_projections.append(su)
            for j,frame in enumerate(frames):
                pt=int(parent.p['pts'][frame]);raw=Image.fromarray(source_reader.read(pt));renders=[]
                for k,label in enumerate(['原联合候选（4步）','支撑刀中点定位（待审核）']):
                    su=shoe_projections[k][j];parts=[su[0,:len(lower)],su[0,len(lower):],su[1,:len(lower)],su[1,len(lower):]]
                    renders.append(paint(raw,projections[k][j],faces,parts,[lower_faces,upper_faces,lower_faces,upper_faces],label,font,pt))
                encode(*encoders[0],renders[1],count)
                common_crop=(round(centres[frame]-360),0,round(centres[frame]+360),1080);canvas=Image.new('RGB',(1920,1080),(12,16,23))
                for k,im in enumerate([raw,*renders]):
                    panel=im.crop(common_crop).resize((640,960));canvas.paste(panel,(k*640,50));draw=ImageDraw.Draw(canvas);draw.text((k*640+12,12),['原片','原联合候选（4步）','支撑刀中点定位（待审核）'][k],font=sfont,fill='white')
                draw=ImageDraw.Draw(canvas);draw.text((16,1020),'源 '+format(pt/12800.,'.2f')+' s  PTS '+str(pt)+'；三个画面使用同一原片裁切；网格按原相机直接回投',font=sfont,fill='white')
                encode(*encoders[1],canvas,count);clock.append(pt)
                if pt in {round(v*12800) for v in (81.2,84.4,85.2,86.8,89.56,92.46,92.8,93.2)}:
                    native_path=target/('native_'+str(pt)+'.png');compare_path=target/('compare_'+str(pt)+'.png');renders[1].save(native_path);canvas.save(compare_path)
                    proof[str(pt)]=dict(native=str(native_path),comparison=str(compare_path))
                count+=1
            if first==0 or first//batch%20==0:print(json.dumps(dict(event='native_surface_backprojection',frames=count,total=len(ids),seconds=time.perf_counter()-tick)),flush=True)
        for c,s in encoders:
            for p in s.encode():c.mux(p)
            c.close()
    finally:
        source_reader.close()
        for c,s in encoders:
            try:c.close()
            except Exception:pass
    checks=[]
    for path in paths:
        with av.open(str(path)) as c:decoded=[float(f.pts*f.time_base) for f in c.decode(video=0)]
        if len(decoded)!=len(ids) or not np.allclose(decoded,np.arange(len(ids))/50,atol=1e-12,rtol=0):raise ValueError('Encoded source clock mismatch')
        checks.append(dict(path=str(path),sha256=digest(path),frames=len(decoded)))
    report=dict(source=source,source_time_base=[1,12800],source_pts=clock,source_seconds=list(seconds),every_source_exposure=True,encoded=checks,proof=proof,
      body_vertices=len(surface.shaped),body_faces=len(faces),exact_exposure_nodes=node_count,
      projection='existing moving_projection: native Brown, per-exposure camera, implicit rolling row, five actual body samples',
      geometry='same full SMPL-X dynamic hands and attached lower/upper skate surfaces as player; no image pose alignment',
      display='translucent projected triangle-union silhouette; external contours; source occluders are not a segmentation truth',
      current_joint_result_sha256=digest(Path(active['result'])),support_located_result_sha256=digest(output/candidate_name/'fitted/result.npz'),
      no_new_fit=True,no_new_network=True,candidate=candidate_name,seconds=time.perf_counter()-tick)
    write_json(target/'projection_and_clock.json',report)
    print(json.dumps(dict(event='native_mesh_overlay_ready',videos=[str(p) for p in paths],frames=count,seconds=report['seconds'])),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate-name',choices=['support_located','support_midpoint'],default='support_midpoint');p.add_argument('--seconds',nargs=2,type=float,default=[76,95]);a=p.parse_args(argv)
    store=AssetStore(a.project);render(a.project,store.resolve(a.base_root,base=store.data,must_exist=True),store.resolve(a.output,base=store.data,must_exist=True),tuple(a.seconds),candidate_name=a.candidate_name)

if __name__=='__main__':main()
