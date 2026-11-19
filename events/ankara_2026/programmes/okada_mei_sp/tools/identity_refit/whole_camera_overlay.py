"""Draw the actual saved body/boots over the registered field review.

The row predictor only accelerates the starting guess. Every delivered vertex
uses the registered native projection at the converged moving-body row.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import bind
from motion_overlay import mesh_nodes
from source_review import digest, SourceVideo
from source_camera_rebind import write
from calibration.program_tracking.body_tracking.rebuild_exposure_body import row_weights


def row_project(cam, frame, nodes):
    cal = cam.cameras[cam.shot_ids[frame]]
    p = cal.p
    clock = p['motion_times'][frame-cam.first[cam.shot_ids[frame]]]
    first,dt,tau = float(p['first_control_time']),float(p['control_dt']),p['rig'][37]
    rig,ctrl = p['rig'],p['controls']
    world_nodes = nodes/cal.scale
    def at_row(y):
        weights,_ = row_weights(y,5)
        return np.einsum('vq,qvc->vc',weights,world_nodes)
    def native(xyz):
        z=cal.core.project_motion(rig,ctrl,first,dt,1080,clock,xyz)
        if z['status'].any():raise ValueError(('native mesh projection', frame,np.unique(z['status'])))
        return z
    # Bracket the predicted rows from the actual pose; include off-raster
    # vertices. The final native check handles unbounded-row continuation.
    mid=cam.midpoint
    q=(world_nodes[2]-mid['centre'][frame])@mid['rotation'][frame].T
    y=q[:,1]/q[:,2]*mid['fy'][frame]+mid['cy'][frame]
    rows=np.arange(min(0,np.floor(y.min())-96),max(1080,np.ceil(y.max())+96)+4,4.)
    table=cal.core.evaluate_motion_camera(rig,ctrl,first,dt,clock+(rows/1079.-.5)*tau)
    if not table['status'].any():
        for _ in range(16):
            at=np.clip((y-rows[0])/4,0.,len(rows)-1.000001)
            j=at.astype(int)
            v=table['values'][j]+(at-j)[:,None]*(table['values'][j+1]-table['values'][j])
            ray=np.einsum('nij,nj->ni',v[:,4:13].reshape(-1,3,3),at_row(y)-v[:,13:16])
            x,b=(ray[:,:2]/ray[:,2,None]).T
            r=x*x+b*b
            updated=v[:,17]*(b*(1+r*(v[:,20]+r*v[:,21]))+v[:,22]*(r+2*b*b)+2*v[:,23]*x*b)+v[:,19]
            delta=np.max(abs(updated-y));y=updated
            if delta<1e-7:break
    for _ in range(64):
        z=native(at_row(y))
        delta=np.max(abs(z['row']-y));y=z['row']
        if delta<1.e-7:
            return np.column_stack((z['uv'],z['depth'])),float(delta)
    raise ValueError(('unresolved moving body row',frame,delta))


class DepthMask:
    def __init__(self, faces, counts, viewer_packages):
        sys.path.append(str(viewer_packages))
        import moderngl
        ctx=moderngl.create_standalone_context()
        self.ctx=ctx
        self.fbo=ctx.framebuffer(ctx.texture((1920,1080),1),ctx.depth_renderbuffer((1920,1080)))
        self.program=ctx.program(vertex_shader='''#version 330
        in vec3 vertex; in float part; flat out float group_id;
        void main(){
            float z=vertex.z;
            vec2 ndc=(vertex.xy+0.5)/vec2(960.,540.)-1.;
            gl_Position=vec4(ndc.x*z,-ndc.y*z,z-0.002,z);
            group_id=part;
        }''',fragment_shader='''#version 330
        flat in float group_id; out float value; void main(){value=group_id/255.;}''')
        self.data=np.empty((sum(counts),4),'f4')
        self.data[:,3]=np.repeat([1,2,2,3,3],counts)
        starts=np.r_[0,np.cumsum(counts)[:-1]]
        indices=np.concatenate([f+a for f,a in zip(faces,starts)]).astype('i4')
        self.buffer=ctx.buffer(reserve=self.data.nbytes)
        self.vao=ctx.vertex_array(self.program,[(self.buffer,'3f 1f','vertex','part')],ctx.buffer(indices.tobytes()))
        ctx.enable(moderngl.DEPTH_TEST)

    def render(self,points):
        self.data[:,:3]=points
        self.buffer.write(self.data.tobytes())
        self.fbo.use();self.fbo.clear(depth=1.);self.vao.render()
        return np.frombuffer(self.fbo.read(components=1,alignment=1),np.uint8).reshape(1080,1920)[::-1]


def overlay(img,mask):
    for code,color in [(1,(38,218,159)),(2,(255,158,41)),(3,(255,158,41))]:
        layer=np.asarray(mask==code,np.uint8)
        x,y,w,h=cv2.boundingRect(layer)
        if w==0:continue
        # Tint only the visible surface, with original pixels readable.
        hit=layer[y:y+h,x:x+w].astype(bool)
        roi=img[y:y+h,x:x+w]
        roi[hit]=(.68*roi[hit]+.32*np.asarray(color)).astype(np.uint8)
        contours,_=cv2.findContours(layer,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,color,1,cv2.LINE_AA)


def detail(canvas, raw, points, box):
    x,y,w,h=box
    visible=points[(points[:,0]>=0)&(points[:,0]<1920)&(points[:,1]>=0)&(points[:,1]<1080)]
    if not len(visible):return False
    lo=np.maximum(0,np.floor(visible[:,:2].min(0)-25)).astype(int)
    hi=np.minimum([1920,1080],np.ceil(visible[:,:2].max(0)+26)).astype(int)
    crop=raw[lo[1]:hi[1],lo[0]:hi[0]]
    scale=min(w/crop.shape[1],h/crop.shape[0])
    tile=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),
        interpolation=cv2.INTER_AREA if scale<=1 else cv2.INTER_CUBIC)
    a=x+(w-tile.shape[1])//2;b=y+(h-tile.shape[0])//2
    canvas[b:b+tile.shape[0],a:a+tile.shape[1]]=tile
    return True


def initialize(prepared,output,registration,viewer_packages):
    global runtime
    cv2.setNumThreads(1)
    ref,cam,*_=bind(prepared,output/'source.json',output/'candidate.npz')
    ref._render_body_surface=copy.copy(ref.binding.surface)
    ref._render_body_surface._sample_double_cache=None
    r=json.loads(registration.read_text(encoding='utf8'))
    raw=SourceVideo(r['source']['path'],(1,12800));field=SourceVideo(r['full_review']['video']['path'],(1,12800))
    for reader in (raw,field):reader.stream.codec_context.thread_count=1
    runtime=(ref,cam,raw,field,output,viewer_packages)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    labels=Image.new('RGB',(640,38),(18,20,21))
    ImageDraw.Draw(labels).text((10,3),'原片局部 · 人体',fill='white',font=font)
    global hdr,shoehdr,caption
    hdr=np.asarray(labels).copy()
    labels=Image.new('RGB',(640,38),(18,20,21))
    ImageDraw.Draw(labels).text((10,3),'双鞋局部 · 原片 / 回投',fill='white',font=font)
    shoehdr=np.asarray(labels).copy()
    labels=Image.new('RGB',(1030,56),(16,18,18))
    ImageDraw.Draw(labels).text((8,10),'冈田芽衣  |  人体与鞋刀回投  ·  s14_boards_1118  /  lane4_side_1118',fill='white',font=font)
    caption=np.asarray(labels).copy()


def segment(bounds):
    ref,cam,source,field,output,packages=runtime
    start,end=bounds; pts=ref.p['pts']
    folder=output/'segments'; path=folder/f'{start:05}.mp4'
    pending=path.with_suffix('.pending.mp4')
    receipt=path.with_suffix('.json')
    if receipt.exists() and path.is_file() and json.loads(receipt.read_text())['sha256']==digest(path):
        return path
    enc=av.open(str(pending),'w');s=enc.add_stream('libx264',rate=50);s.width=2560;s.height=1144
    s.pix_fmt='yuv420p';s.options={'crf':'18','preset':'veryfast'};s.codec_context.thread_count=2
    raster=None;max_residual=0.
    try:
        for a in range(start,end,4):
            ids=np.arange(a,min(a+4,end))
            body,faces,shoe,nlow,shoe_faces=mesh_nodes(ref,cam,ids)
            for j,i in enumerate(ids):
                stamp=int(pts[i])
                nodes=np.concatenate((body[j],shoe[j].reshape(5,-1,3)),axis=1)
                points,residual=row_project(cam,i,nodes)
                max_residual=max(max_residual,residual)
                if raster is None:
                    nupper=shoe.shape[3]-nlow
                    raster=DepthMask([faces,*shoe_faces],[body.shape[2],nlow,nupper,nlow,nupper],packages)
                mask=raster.render(points)
                raw=source.read(stamp)
                canvas=field.read(stamp-int(pts[0])).copy()
                if canvas.shape!=(1144,2560,3):raise ValueError('Unexpected registered field layout')
                overlay(canvas[64:,:1920],mask)
                canvas[4:60,790:1820]=caption
                canvas[72:110,1920:]=hdr
                canvas[110:415,1920:]=18
                detail(canvas,raw,points,(1923,112,633,300))
                canvas[775:813,1920:]=shoehdr
                canvas[813:1043,1920:]=18
                foot=points[body.shape[2]:]
                detail(canvas,raw,foot,(1923,816,311,223))
                detail(canvas,canvas[64:,:1920],foot,(2243,816,311,223))
                frame=av.VideoFrame.from_ndarray(canvas,format='rgb24');frame.pts=i-start
                for packet in s.encode(frame):enc.mux(packet)
        for packet in s.encode():enc.mux(packet)
    finally:
        enc.close()
        if raster is not None:raster.ctx.release()
    with av.open(str(pending)) as c:
        assert c.streams.video[0].frames==end-start
    pending.replace(path)
    write(receipt,dict(frames=end-start,source_pts=[int(pts[start]),int(pts[end-1])],
        sha256=digest(path),maximum_native_row_fixed_point_error_px=max_residual))
    print('rendered',start,end,flush=True)
    return path


def run(args):
    r=json.loads(args.registration.read_text(encoding='utf8'))
    field=Path(r['full_review']['video']['path'])
    if digest(field)!=r['full_review']['video']['sha256']:raise ValueError('Registered field video changed')
    pts=np.load(r['camera']['path'])['source_pts']
    folder=args.output/'segments';folder.mkdir(exist_ok=True)
    identity=dict(candidate=digest(args.output/'candidate.npz'),scene_video=digest(field),
        renderer=digest(__file__),mesh=digest(Path(__file__).with_name('motion_overlay.py')))
    index=folder/'identity.json'
    if index.exists() and json.loads(index.read_text())!=identity:raise ValueError('Segment identity differs')
    write(index,identity)
    groups=[(i,min(i+256,len(pts))) for i in range(0,len(pts),256)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(
            args.prepared,args.output,args.registration,args.viewer_packages)) as pool:
        paths=list(pool.map(segment,groups))
    listing=folder/'concat.txt'
    listing.write_text(''.join("file '"+p.name+"'\n" for p in paths),encoding='ascii')
    dest=args.output/'Okada_body_reprojection.mp4'
    pending=dest.with_suffix('.pending.mp4')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),
        '-i',str(field),'-map','0:v:0','-map','1:a:0?','-c','copy','-movflags','+faststart',str(pending)],check=True)
    with av.open(str(pending)) as c:
        times=np.asarray([float(f.pts*f.time_base) for f in c.decode(video=0)])
        audio=len(c.streams.audio)
    np.testing.assert_allclose(times,(pts-pts[0])/12800.,atol=1e-10,rtol=0)
    pending.replace(dest)
    write(args.output/'video.json',dict(**identity,path=str(dest),sha256=digest(dest),source_pts=pts.tolist(),
        time_base=[1,12800],frames=len(pts),audio_streams=audio,segments=[
            json.loads(p.with_suffix('.json').read_text()) for p in paths]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('prepared','output','registration','viewer_packages'):
        parser.add_argument(name,type=lambda x:Path(x).resolve())
    parser.add_argument('--workers',type=int,default=6)
    run(parser.parse_args())
