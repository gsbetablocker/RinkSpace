"""Render unchanged native source pixels beside saved motion candidates."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from compare import bind
from motion_overlay import project
from source_review import SourceVideo
from calibration.program_tracking.body_tracking.native_mesh_overlay import paint, silhouette, open_encoder, encode


_render_state = None


def initialize_renderer(args):
    global _render_state
    before_source = getattr(args,'before_source',None) or args.source
    controls = [bind(args.prepared,source,path) for source,path in
                ((before_source,args.before),(args.source,args.after))]
    np.testing.assert_array_equal(controls[0][0].p['pts'],controls[1][0].p['pts'])
    # Body and boot skinning alternate full/subset selections. Give this
    # read-only renderer separate immutable caches for their existing FP64
    # operations instead of rebuilding the full pose basis every three frames.
    for ref,*_ in controls:
        ref._render_body_surface=copy.copy(ref.binding.surface)
        ref._render_body_surface._sample_double_cache=None
    with np.load(args.before_result) as z:
        source_uv = z['source_uv_native']
    observations=getattr(args,'body_observations',None)
    if observations is not None:
        # Ankle detector row numbers can exchange physical identities at a
        # crossing. Use the reviewed assignments for the detail crop only;
        # neither the source image nor projected geometry is moved.
        with np.load(observations) as z:
            np.testing.assert_array_equal(z['pts'],controls[0][0].p['pts'])
            ids,use,raw=z['model_joint_ids'],z['use'],z['uv']
            source_uv=source_uv.copy()
            for foot in (0,1):
                match=(ids==15+foot)&use
                valid=match.sum(1)==1
                source_uv[:,15+foot]=np.nan
                frames=np.flatnonzero(valid)
                source_uv[frames,15+foot]=raw[frames,np.argmax(match[frames],axis=1)]
    registration = json.loads(args.registration.read_text(encoding='utf-8'))['primary_source']
    reader = SourceVideo(registration['path'],np.array([1,12800]))
    reader.stream.codec_context.thread_count=1
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    both=getattr(args,'both_feet_detail',False)
    _render_state = (controls,source_uv,reader,font,args.foot_detail or both,(0,1) if both else (0,))


def render_chunk(frames):
    controls,source_uv,video,font,detail,detail_feet = _render_state
    pts=controls[0][0].p['pts']
    height=1080 if detail else 800
    projected=[project(t[0],t[1],frames) for t in controls]
    rendered=[]
    for offset,frame in enumerate(frames):
        stamp=int(pts[frame]);original=Image.fromarray(video.read(stamp))
        near=source_uv[max(0,frame-2):frame+3].reshape(-1,2)
        near=near[np.isfinite(near).all(-1)]
        near=near[(near[:,0]>0)&(near[:,0]<1920)&(near[:,1]>0)&(near[:,1]<1080)]
        lo=np.maximum(0,np.floor(near.min(0)-[100,95])).astype(int)
        hi=np.minimum([1920,1080],np.ceil(near.max(0)+[100,100])).astype(int)
        # Include projected surfaces too; an off-source limb must remain visible.
        for body,_,shoes,_ in projected:
            px=np.concatenate([body[offset],*[q[offset] for q in shoes]])
            px=px[np.isfinite(px).all(-1)]
            lo=np.maximum(0,np.minimum(lo,np.floor(px.min(0)-25))).astype(int)
            hi=np.minimum([1920,1080],np.maximum(hi,np.ceil(px.max(0)+25))).astype(int)
        bounds=(*lo,*hi)
        panels=[original]
        for body,faces,shoes,shoe_faces in projected:
            panels.append(paint(original,body[offset],faces,[q[offset] for q in shoes],shoe_faces,'',font,stamp))
        main_height=590 if detail else 660
        canvas=Image.new('RGB',(1920,height),(19,24,30));draw=ImageDraw.Draw(canvas)
        for k,(label,panel) in enumerate(zip(('原片','修正前','修正后'),panels)):
            tile=panel.crop(bounds);tile.thumbnail((635,main_height),Image.Resampling.LANCZOS)
            canvas.paste(tile,(640*k+(640-tile.width)//2,35+(main_height-tile.height)//2))
            draw.text((640*k+12,8),label,font=font,fill='white')
        if detail:
            # One common crop contains the source ankle and BOTH saved
            # left shoes; an incorrect extent cannot be cropped away.
            width=640//len(detail_feet)
            for slot,foot in enumerate(detail_feet):
                ankle=source_uv[frame,15+foot]
                foot_points=[ankle[None]+[-80.,-45.],ankle[None]+[80.,85.]]
                details=[original]
                parts=slice(2*foot,2*foot+2)
                for _,_,shoes,shoe_faces in projected:
                    pixels=np.concatenate([q[offset] for q in shoes[parts]])
                    foot_points.append(pixels)
                    pixels_image=np.asarray(original).copy()
                    for q,faces in zip(shoes[parts],shoe_faces[parts]):
                        mask=silhouette(q[offset],faces,pixels_image.shape[:2])
                        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(pixels_image,contours,-1,(255,155,35),1)
                    details.append(Image.fromarray(pixels_image))
                px=np.concatenate(foot_points);px=px[np.isfinite(px).all(1)]
                f0=np.maximum(0,np.floor(px.min(0)-8)).astype(int)
                f1=np.minimum([1920,1080],np.ceil(px.max(0)+8)).astype(int)
                if np.any(f1<=f0):
                    for k in range(3):
                        x=640*k+width*slot
                        draw.text((x+12,633),('左脚' if foot==0 else '右脚')+'：鞋与刀',font=font,fill='white')
                        draw.text((x+12,785),'此范围在原片画面外',font=font,fill='white')
                    continue
                for k,panel in enumerate(details):
                    tile=panel.crop((*f0,*f1))
                    scale=min((width-15)/tile.width,330/tile.height)
                    tile=tile.resize((max(1,round(tile.width*scale)),max(1,round(tile.height*scale))),Image.Resampling.LANCZOS)
                    x=640*k+width*slot
                    canvas.paste(tile,(x+(width-tile.width)//2,660+(330-tile.height)//2))
                    draw.text((x+12,633),('左脚' if foot==0 else '右脚')+'：鞋与刀',font=font,fill='white')
            footer=height-82
        else:
            footer=704
        thumb=original.copy();thumb.thumbnail((130,73));canvas.paste(thumb,(12,footer))
        draw.text((165,footer+29),f'原片 {stamp/12800:.2f} 秒   PTS {stamp}   同一原片机位 · 保存的三维动作直接投影',font=font,fill='white')
        rendered.append((stamp,np.asarray(canvas).copy()))
    return rendered


def render(args):
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    import av
    from source_review import digest

    args.output.mkdir(parents=True,exist_ok=True)
    initialize_renderer(args)
    pts=_render_state[0][0][0].p['pts']
    registration=json.loads(args.registration.read_text(encoding='utf-8'))['primary_source']
    if args.video:
        selected=np.flatnonzero((pts>=round(args.seconds[0]*12800))&(pts<=round(args.seconds[-1]*12800)))
    else:
        stamps=np.rint(np.array(args.seconds)*12800).astype(np.int64)
        selected=np.searchsorted(pts,stamps)
        np.testing.assert_array_equal(pts[selected],stamps)
    height=1080 if args.foot_detail or getattr(args,'both_feet_detail',False) else 800
    final_path=args.output/'source_motion_comparison.mp4'
    replacing=args.video and final_path.exists()
    previous=None
    if replacing:
        if not getattr(args,'replace_video',False):
            raise FileExistsError(final_path)
        previous=dict(path=str(final_path),sha256=digest(final_path),
                      source_clock=json.loads((args.output/'source_clock.json').read_text(encoding='utf-8')))
    path=final_path.with_name('source_motion_comparison.pending.mp4') if replacing else final_path
    if replacing and path.exists():
        raise FileExistsError(path)
    stream=open_encoder(path,1920,height) if args.video else None
    chunks=[selected[first:first+3] for first in range(0,len(selected),3)]
    saved=[]
    pool=None
    try:
        if args.workers>1:
            pool=ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_renderer,initargs=(args,))
            pending=deque()
            chunk_iterator=iter(chunks)
            def batches():
                for _ in range(2*args.workers):
                    frames=next(chunk_iterator,None)
                    if frames is not None:pending.append(pool.submit(render_chunk,frames))
                while pending:
                    value=pending.popleft().result()
                    frames=next(chunk_iterator,None)
                    if frames is not None:pending.append(pool.submit(render_chunk,frames))
                    yield value
            output=batches()
        else:
            output=map(render_chunk,chunks)
        count=0
        for frames in output:
            for stamp,pixels in frames:
                canvas=Image.fromarray(pixels)
                if stream:
                    encode(*stream,canvas,count)
                else:
                    still=args.output/f'{stamp}.jpg';canvas.save(still,quality=94);saved.append(str(still))
                count+=1
            if count<=3 or count%150==0 or count==len(selected):
                print(json.dumps(dict(event='render_progress',frames=count,total=len(selected),source_pts=frames[-1][0])),flush=True)
        if stream:
            for packet in stream[1].encode():stream[0].mux(packet)
    finally:
        _render_state[2].close()
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        if stream:stream[0].close()
    if args.video:
        with av.open(str(path)) as video:
            times=np.array([float(f.pts*f.time_base) for f in video.decode(video=0)])
        np.testing.assert_allclose(times,np.arange(len(selected))/50.,rtol=0.,atol=1.e-10)
        if replacing:
            path.replace(final_path)
    (args.output/'source_clock.json').write_text(json.dumps(dict(source_pts=pts[selected].tolist(),
        time_base=[1,12800],source_video=registration['path'],before=str(args.before),after=str(args.after),
        before_sha256=digest(args.before),after_sha256=digest(args.after),
        body_observations=None if getattr(args,'body_observations',None) is None else str(args.body_observations),
        body_observations_sha256=None if getattr(args,'body_observations',None) is None else digest(args.body_observations),
        replaces_previous_delivery=previous,
        projection='Actual saved body and both skates, five native readout states; one common crop; no image alignment',
        native_frames_verified=len(selected),stills=saved,visual_acceptance='pending'),indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','registration','before','after','before-result','output'):
        p.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--seconds',nargs='+',type=float,required=True)
    p.add_argument('--video',action='store_true')
    p.add_argument('--foot-detail',action='store_true')
    p.add_argument('--both-feet-detail',action='store_true')
    p.add_argument('--body-observations',type=lambda s:Path(s).resolve(),
        help='Use reviewed physical ankle identities for source detail crops')
    p.add_argument('--before-source',type=lambda s:Path(s).resolve(),
        help='Explicit source/camera binding for the historical before model')
    p.add_argument('--replace-video',action='store_true',
        help='Replace this generated comparison only after complete decode/clock verification; retain its model and hash provenance')
    p.add_argument('--workers',type=int,default=1)
    render(p.parse_args())


if __name__=='__main__':main()
