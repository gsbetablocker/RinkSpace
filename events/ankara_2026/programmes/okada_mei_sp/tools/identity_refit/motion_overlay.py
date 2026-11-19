"""Local original-camera comparison of two saved continuous poses."""
import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration.program_tracking.body_tracking.footwear_binding import transform
from calibration.program_tracking.body_tracking.native_mesh_overlay import encode, open_encoder, paint
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from compare import bind, write
from motion_compare import states
from source_review import SourceVideo, digest, review_bounds


def mesh_nodes(ref, camera, frames, *, include_body=True):
    tau = camera.readout_times(frames)
    times = (ref.times[frames,None]+tau[:,None]*np.linspace(-.5,.5,5)).ravel()
    s = fk_state(ref,ref.at(times),np.zeros((len(times),69)))
    b = ref.binding
    surface = getattr(ref, '_render_body_surface', b.surface)
    model = b.packet()
    template = model['template']
    lower = np.asarray(template['vertices'])
    deformation = b.deformation(s['native'],s['frames'],skate_relative_R=s['skate_relative_R'])
    upper = b.deform(np.asarray(model['upper']),deformation)
    hand = {} if 'hand_local_R' not in s else dict(hand_local_R=s['hand_local_R'])
    shoe = np.concatenate((transform(s['R'],s['O'],lower),transform(s['R'],s['O'],upper)),axis=2)
    body = None
    if include_body:
        v = surface.sample_vertices(s['native'],s['frames'],double_precision=True,**hand)
        body = np.einsum('nij,nvj->nvi',s['Q'],v)+s['P'][:,None]
        body = body.reshape(len(frames),5,-1,3)
    return body, surface.faces, shoe.reshape(len(frames),5,2,-1,3), len(lower), [template['faces'],b.faces]*2


def project(ref, camera, frames, *, include_body=True):
    body, faces, shoe, lower_count, shoe_faces = mesh_nodes(ref,camera,frames,include_body=include_body)
    p = moving_projection(camera,body,frames)['uv'] if body is not None else None
    q = moving_projection(camera,shoe.reshape(len(frames),5,-1,3),frames)['uv'].reshape(len(frames),2,-1,2)
    shoes = [q[:,foot,span] for foot in range(2) for span in (slice(None,lower_count),slice(lower_count,None))]
    return p,faces,shoes,shoe_faces


def render(data,prepared,source,replay,fit,output,seconds,*,video=False):
    scope, _, a, r, n = states(data,replay,fit)
    if float(np.max(abs(n['COCO_uv']-r['COCO_uv'])))<1.:
        raise ValueError('A local pose comparison needs visible saved geometry movement')
    refs = [bind(prepared,source,s/'solve/candidate.npz')[:2] for s in (replay,fit)]
    pts = r['source_pts']
    lo,hi = [round(t*12800) for t in seconds]
    frames = np.flatnonzero((pts>=lo)&(pts<=hi))
    if not len(frames):
        raise ValueError('No native exposures in the requested range')
    output.mkdir(parents=True,exist_ok=False)
    labels = ('Clean source','R  Zero update','N  Refit')
    fonts = [ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size) for size in (19,23)]
    reader = SourceVideo(scope['source']['registered_path'],r['time_base'])
    path = output/'entry_81_R_N.mp4'
    stream = open_encoder(path,1536,720) if video else None
    proof = set(round(t*12800) for t in (81.14,81.16,81.18,81.20,81.22,81.34,81.36,81.38,81.40,81.42,81.50,81.60,81.62))
    saved = []
    try:
        for start in range(0,len(frames),3):
            chunk = frames[start:start+3]
            if video or any(stamp in proof for stamp in pts[chunk]):
                assets = [project(ref,camera,chunk) for ref,camera in refs]
                for j, i in enumerate(chunk):
                    stamp = int(pts[i])
                    if not video and stamp not in proof:
                        continue
                    rgb = reader.read(stamp)
                    raw = Image.fromarray(rgb)
                    x0,y0,x1,y1 = review_bounds(a,int(i),rgb.shape)
                    common=(max(0,x0-95),max(0,y0-32),min(rgb.shape[1],x1+95),min(rgb.shape[0],y1+38))
                    plots=[raw]
                    for k,(body,faces,shoes,shoe_faces) in enumerate(assets):
                        plots.append(paint(raw,body[j],faces,[q[j] for q in shoes],shoe_faces,labels[k+1],fonts[0],stamp))
                    canvas = Image.new('RGB',(1536,720),(19,24,30));draw = ImageDraw.Draw(canvas)
                    for k,plot in enumerate(plots):
                        tile=plot.crop(common);tile.thumbnail((505,579),Image.Resampling.LANCZOS)
                        canvas.paste(tile,(k*512+(512-tile.width)//2,36+(579-tile.height)//2))
                        draw.text((k*512+12,8),labels[k],font=fonts[1],fill='white')
                    raw.thumbnail((158,90));canvas.paste(raw,(12,624))
                    x0,y0,x1,y1 = common
                    draw.rectangle((12+x0*raw.width/1920,624+y0*raw.height/1080,
                                    12+x1*raw.width/1920,624+y1*raw.height/1080),outline='yellow')
                    draw.text((192,656),f'Original source {stamp/12800:.2f} s    PTS {stamp}    Same camera and original exposure',
                              font=fonts[1],fill='white')
                    if stamp in proof:
                        still = output/f'{stamp}.jpg';canvas.save(still,quality=94)
                        saved.append(dict(pts=stamp,path=str(still),sha256=digest(still)))
                    if stream:
                        encode(*stream,canvas,start+j)
            print(f'comparison complete through source PTS {pts[chunk[-1]]}',flush=True)
        if stream:
            for packet in stream[1].encode():
                stream[0].mux(packet)
    finally:
        reader.close()
        if stream:
            stream[0].close()
    video_info = None
    if video:
        with av.open(str(path)) as c:
            clock = np.array([float(f.pts*f.time_base) for f in c.decode(video=0)])
        np.testing.assert_allclose(clock,np.arange(len(frames))/50.,rtol=0.,atol=1.e-10)
        video_info = dict(path=str(path),sha256=digest(path),frames=len(clock),source_pts=pts[frames].tolist())
    write(output/'media.json',dict(source_sha256=scope['source']['sha256'],time_base=[1,12800],stills=saved,
        video=video_info,candidate_sha256=[digest(s/'solve/candidate.npz') for s in (replay,fit)],
        projection='Actual current body and boot surfaces at five readout nodes, projected by the registered camera',
        no_pose_or_display_alignment=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data','prepared','source','replay','fit','output'):
        p.add_argument(arg,type=lambda x:Path(x).resolve())
    p.add_argument('--seconds',type=float,nargs=2,default=(81.02,81.74))
    p.add_argument('--video',action='store_true')
    a = p.parse_args()
    render(a.data,a.prepared,a.source,a.replay,a.fit,a.output,a.seconds,video=a.video)


if __name__=='__main__':
    main()
