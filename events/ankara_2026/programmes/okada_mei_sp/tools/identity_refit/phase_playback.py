"""Bind R/N/M to one native clock and project a local original-video comparison."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration.program_tracking.body_tracking.native_mesh_overlay import encode, open_encoder, paint
from compare import bind, write
from motion_overlay import project
from source_review import PROGRAMME, SourceVideo, baseline, digest, read, review_bounds


REGISTRY = PROGRAMME/'fit_v11/body_trusted_motion_1115.json'
IDS = ('okada_phase_1115_R','okada_phase_1115_N','okada_phase_motion_1115')
LABELS = ('R · 历史零更新','N · 1115 膝线修复','M · 相位重拟合')


def bound_viewer(data, stages, output, support, start, *, ids=IDS, labels=LABELS,
                 scope_label='S05 入旋完整动作 · R / N / M'):
    scope, paths, original = baseline(data,REGISTRY)
    config=json.loads(read(paths['playback_config']))
    variants=[]
    for i,(key,label,stage) in enumerate(zip(ids,labels,stages.values(),strict=True)):
        old = config['variants'][0]
        result,parameter = stage/'fitted/result.npz',stage/'solve/candidate.npz'
        variant=dict(old,id=key,candidate_id=key,label=label,result=str(result),
            result_sha256=digest(result),parameter=str(parameter),parameter_sha256=digest(parameter),
            body_reference_label=ids[0],candidate_scope_label='完整节目原生源 PTS · 待验收',
            blade_review_available=False)
        updates=support if i==len(ids)-1 else []
        variant['body_refinement_intervals_seconds']=[r['seconds'] for r in updates]
        variant['body_refinement_intervals_pts']=[[r['first_pts'],r['last_pts']] for r in updates]
        variants.append(variant)
    config.update(variants=variants,session_id=ids[-1],comparison_variants=list(ids),default_variant=ids[-1],
        latest_review_variant=ids[-1],initial_source_pts=start,body_scope_label=scope_label,
        source_review_model_body_default=False,source_review_model_runner_default=False)
    nav=json.loads(read(config['source_navigation']))
    nav['events'].append(dict(kind='reviewed_motion_scope',start_pts=1035264,end_pts=1058560,worst_pts=1039104,
                              label='80.88–82.70 秒 · 准备、腾空、落冰、稳定旋转'))
    nav['events'].append(dict(kind='reviewed_motion_scope',start_pts=993280,end_pts=1005568,worst_pts=998400,
                              label='2A 起落与腾空质量检查'))
    nav['events'].sort(key=lambda row:row['start_pts'])
    output.mkdir(parents=True,exist_ok=False)
    write(output/'navigation.json',nav)
    config['source_navigation']=str(output/'navigation.json')
    write(output/'viewer3d.json',config)
    with (output/'START_PHASE_MOTION_3D.cmd').open('xb') as stream:
        stream.write(paths['viewer'].read_bytes())
    return scope,original


def local_video(prepared,source,stages,scope,original,output,span,label,*,labels=(
        'Original exposure','R  Historical replay','N  Knee-line refit','M  Phase refit')):
    pts=original['source_pts']; time_base=original['time_base']
    lo,hi=[round(t*12800) for t in span]
    frames=np.flatnonzero((pts>=lo)&(pts<=hi))
    ready=[bind(prepared,source,v/'solve/candidate.npz')[:2] for v in stages.values()]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    panel_width=1920//len(labels)
    keyframes={round(t*12800) for t in (80.92,81.04,81.16,81.18,81.20,81.22,81.34,81.38,81.40,81.42,
                                      81.50,81.54,81.58,81.60,81.62,81.72,81.86,82.10,82.34,82.46,
                                      77.82,78.00,78.18,78.20)}
    reader=SourceVideo(scope['source']['registered_path'],time_base)
    path=output/(label+'.mp4')
    encoder=open_encoder(path,1920,720)
    stills=[]
    try:
        for start in range(0,len(frames),3):
            chunk=frames[start:start+3]
            poses=[project(ref,camera,chunk) for ref,camera in ready]
            for j,i in enumerate(chunk):
                stamp=int(pts[i])
                rgb=reader.read(stamp)
                original_frame=Image.fromarray(rgb)
                x,y,x1,y1=review_bounds(original,int(i),rgb.shape)
                bounds=(max(0,x-105),max(0,y-35),min(rgb.shape[1],x1+105),min(rgb.shape[0],y1+45))
                panels=[original_frame]
                for k,(body,faces,shoes,shoe_faces) in enumerate(poses):
                    panels.append(paint(original_frame,body[j],faces,[q[j] for q in shoes],shoe_faces,labels[k+1],font,stamp))
                canvas=Image.new('RGB',(1920,720),(19,24,30));draw=ImageDraw.Draw(canvas)
                for k,panel in enumerate(panels):
                    tile=panel.crop(bounds);tile.thumbnail((panel_width-5,570),Image.Resampling.LANCZOS)
                    canvas.paste(tile,(k*panel_width+(panel_width-tile.width)//2,34+(570-tile.height)//2))
                    draw.text((k*panel_width+12,8),labels[k],font=font,fill='white')
                original_frame.thumbnail((158,89));canvas.paste(original_frame,(12,622))
                x,y,x1,y1=bounds
                draw.rectangle((12+x*158/1920,622+y*89/1080,12+x1*158/1920,622+y1*89/1080),outline='yellow')
                draw.text((190,652),f'Original source {stamp/12800:.2f} s   PTS {stamp}   Same native camera and exposure',
                          font=font,fill='white')
                encode(*encoder,canvas,start+j)
                if stamp in keyframes:
                    still=output/f'{stamp}.jpg';canvas.save(still,quality=94)
                    stills.append(dict(pts=stamp,path=str(still),sha256=digest(still)))
        for packet in encoder[1].encode():
            encoder[0].mux(packet)
    finally:
        reader.close()
        encoder[0].close()
    with av.open(str(path)) as container:
        encoded=np.asarray([float(frame.pts*frame.time_base) for frame in container.decode(video=0)])
    np.testing.assert_allclose(encoded,np.arange(len(frames))/50.,rtol=0.,atol=1.e-10)
    return dict(path=str(path),sha256=digest(path),frames=len(frames),time_base=time_base.tolist(),
                source_pts=pts[frames].tolist(),stills=stills)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','prepared','source','R','N','M','comparison','output'):
        p.add_argument(name,type=lambda x:Path(x).resolve())
    p.add_argument('--video',action='store_true')
    p.add_argument('--jump-video',action='store_true')
    a=p.parse_args()
    stages={code:getattr(a,code) for code in ('R','N','M')}
    values=json.loads(read(a.comparison))
    scope,original=bound_viewer(a.data,stages,a.output,values['exact_native_support'],1035776)
    if a.video:
        videos=[local_video(a.prepared,a.source,stages,scope,original,a.output,(80.88,82.70),'entry_spin_R_N_M')]
        if a.jump_video:
            videos.append(local_video(a.prepared,a.source,stages,scope,original,a.output,(77.60,78.56),'2A_R_N_M'))
        write(a.output/'media.json',dict(source_sha256=scope['source']['sha256'],videos=videos,
              parameter_sha256={code:digest(run/'solve/candidate.npz') for code,run in stages.items()},
              common_native_camera=True,display_alignment=False))


if __name__=='__main__':
    main()
