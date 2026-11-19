"""Bidirectional source foreground for conservative footwear containment.

One observed person includes both legs and both shoes. It never assigns a
visible foreground boot to a hidden physical foot. Outside-distance is a weak
source exclusion; interior pixels are not fabricated point correspondences.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from source_review import SourceVideo,digest, whole_skate_observations


def qualify(args):
    from calibration.program_tracking.body_tracking.source_foot_foreground import field_sample
    packet=dict(np.load(args.output));pts=packet['pts'];active=packet['active'].copy()
    points_path=whole_skate_observations(Path(__file__).resolve().parents[2])
    source=json.loads(points_path.read_text(encoding='utf8'));bad=set();checks=[]
    rows=[r for r in source['samples'] if r['role']=='fit' and
          r.get('source_kind') in ('assistant_native_source_review','adjacent_source_flow') and 'pixel_box' not in r]
    for row in rows:
        i=int(np.searchsorted(pts,row['pts']))
        if i==len(pts) or pts[i]!=row['pts'] or not active[i]:continue
        value,_=field_sample(packet['distance'][i:i+1].astype(float),packet['bounds'][i:i+1],
                             np.asarray(row['uv'],float).reshape(1,1,2))
        outside=float(value[0,0]);allowance=float(row.get('normalization_px',row.get('sigma_px',10.)))
        if outside>allowance:
            bad.add(i);checks.append(dict(pts=int(pts[i]),source_point_id=row.get('id'),outside_px=outside,
                                         source_reading_allowance_px=allowance))
    rejected=np.zeros(len(pts),bool)
    for i in bad:
        ids=np.arange(max(0,i-4),min(len(pts),i+5));ids=ids[packet['shot_ids'][ids]==packet['shot_ids'][i]]
        rejected[ids]=True
    # Explicit original-source absence takes precedence over a detector's
    # extrapolated ankle and a segmentation foreground at the image border.
    absence=[]
    for row in source.get('shoe_regions',[]):
        if row.get('source_kind')=='assistant_source_absence_region':absence.append(row['pts'])
    for stamp in absence:rejected|=abs(pts-int(stamp))<=512
    packet['active']=active&~rejected
    packet['source_visibility_qualification_json']=np.array(json.dumps(dict(
        source_points_sha256=digest(points_path),direct_or_correlated_source_counterexamples=checks,
        removed_exposures=int(np.count_nonzero(active&rejected)),retained_exposures=int(packet['active'].sum()),
        reason='Do not penalize a real source boot excluded by a segmentation, or a known off-image boot.',
        independent_accuracy_evaluation=False)))
    pending=args.output.with_suffix('.pending.npz');np.savez_compressed(pending,**packet);pending.replace(args.output)
    print(str(packet['source_visibility_qualification_json']),flush=True)


def review_all(args):
    from PIL import Image,ImageDraw,ImageFont
    a=np.load(args.output);report=json.loads((args.work/'source_review/occlusion_review.json').read_text(encoding='utf8'))
    media=json.loads(args.registration.read_text(encoding='utf8'))['primary_source'];reader=SourceVideo(media['path'],[1,12800])
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    for first in range(0,len(report['rows']),4):
        page=Image.new('RGB',(2000,1440),(20,25,29));draw=ImageDraw.Draw(page)
        for rowpos,row in enumerate(report['rows'][first:first+4]):
            for col,stamp in enumerate(row['pts']):
                i=int(np.searchsorted(a['pts'],stamp));x,y,r,b=a['bounds'][i]
                raw=reader.read(int(stamp));used=bool(a['active'][i])
                if r<=x or b<=y:x,y,r,b=row['crop'];rgb=raw[y:b,x:r].copy()
                else:
                    rgb=raw[y:b,x:r].copy();q=cv2.resize(a['distance'][i].astype('float32'),(r-x,b-y))
                    cs,_=cv2.findContours((q<.5).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgb,cs,-1,(30,255,100) if used else (255,130,40),2)
                scale=min(396/(r-x),320/(b-y));im=Image.fromarray(rgb).resize((max(1,round((r-x)*scale)),max(1,round((b-y)*scale))))
                page.paste(im,(col*400+2,rowpos*360+38));draw.text((col*400+3,rowpos*360+3),
                    f'R{row["index"]} C{col} {stamp/12800:.2f} use={int(used)}',font=font,fill='white')
        page.save(args.work/f'source_review/occlusion_foreground_{first//4:02d}.jpg',quality=94)
    reader.close()


def crosscheck_outlines(args):
    from calibration.program_tracking.body_tracking.source_foot_foreground import field_sample
    from source_occlusion_review import polygon_overlap
    fg=dict(np.load(args.output));path=args.work/'input/shoe_occlusion_source.npz';p=dict(np.load(path))
    original=dict(np.load(args.work/'input/shoe_turn_source.npz'))
    valid=(p['source_boundary_contrast']>=5.)&~p['source_extended_line']&(original['role']==1)[...,None]
    for i in np.flatnonzero(p['source_pair_overlap']>.06):
        _,a,b=polygon_overlap(*p['region'][i]);valid[i,0]&=~a;valid[i,1]&=~b
    p['partial_boundary']=(p['source_pair_overlap'][:,None]>.06)|(valid.mean(-1)<.78)|p['source_extended_line'].any(-1)
    p['role']=np.where((valid.sum(-1)>=8)&(p['source_pair_overlap'][:,None]<.65),original['role'],0).astype(np.uint8)
    p['measure_s']=np.where(p['role']==1,original['measure_s'],0.)
    old=valid.copy();dropped=np.zeros_like(valid)
    ids=np.flatnonzero(fg['active'])
    for start in range(0,len(ids),128):
        frames=ids[start:start+128]
        for foot in (0,1):
            uv=p['region'][frames,foot];finite=np.isfinite(uv).all((1,2));group=frames[finite]
            if not len(group):continue
            distance,_=field_sample(fg['distance'][group].astype(float),fg['bounds'][group],uv[finite])
            # Fine steel may lie outside a person mask even when correct.
            # Only gross excursions beyond the boot-size-scaled allowance can
            # disqualify an old whole-assembly contour here.
            allowance=np.maximum(12.,.18*np.linalg.norm(np.ptp(uv[finite],axis=1),axis=1))
            outside=distance>allowance[:,None]
            dropped[group,foot]=valid[group,foot]&outside;valid[group,foot]&=~outside
    p['boundary_valid']=valid
    p['partial_boundary']|=dropped.any(-1)
    usable=(p['role']==1)&(valid.sum(-1)>=8)
    p['role']=np.where(usable,p['role'],0).astype(np.uint8);p['measure_s']=np.where(usable,p['measure_s'],0.)
    record=dict(foreground_sha256=digest(args.output),source_arcs_removed=int(dropped.sum()),
        source_foot_frames_affected=int(dropped.any(-1).sum()),
        meaning='Gross disagreement with source foreground disqualifies candidate outline arcs; fine steel retains a size-scaled allowance. Neither segmentation is independent truth.',
        source_pixels_not_relocated=True)
    p['foreground_outline_crosscheck_json']=np.array(json.dumps(record))
    pending=path.with_suffix('.pending.npz');np.savez_compressed(pending,**p);pending.replace(path)
    print(json.dumps(record),flush=True)


def review(args):
    from PIL import Image,ImageDraw,ImageFont
    a=np.load(args.output);b=np.load(args.work/'source_review/shoe_screen_after.npz')
    media=json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    reader=SourceVideo(media['path'],[1,12800]);font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
    times=args.only_seconds or [52.26,61.58,62.72,68.56,75.66]
    out=Image.new('RGB',(400*len(times),650),(20,25,29));draw=ImageDraw.Draw(out)
    for k,t in enumerate(times):
        i=int(np.searchsorted(a['pts'],round(t*12800)));x,y,r,bot=a['bounds'][i]
        rgb=reader.read(int(a['pts'][i]))[y:bot,x:r].copy()
        q=cv2.resize(a['distance'][i].astype('float32'),(r-x,bot-y))
        cs,_=cv2.findContours((q<.5).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb,cs,-1,(30,255,100),2)
        for foot in (0,1):cv2.polylines(rgb,[np.rint(b['contour'][i,foot]-[x,y]).astype('int32')],True,(255,150,20),1)
        scale=min(396/(r-x),610/(bot-y));im=Image.fromarray(rgb).resize((round((r-x)*scale),round((bot-y)*scale)))
        out.paste(im,(k*400+2,35));draw.text((k*400+4,5),f'{t:.2f} src / model use={a["active"][i]}',font=font)
    reader.close();out.save(args.work/'source_review/person_foreground.jpg',quality=95)


def run(args):
    project=Path(__file__).resolve().parents[6]
    sys.path.insert(0,str(project/'experiments/skate_video_v1'))
    from track import VideoMasks
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    media=json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    obs=dict(np.load(args.work/'input/body_track_source.npz'))
    shoe=dict(np.load(args.work/'input/shoe_turn_source.npz'))
    pts=obs['pts'];shots=shoe['shot_ids'];n=len(pts);size=96
    ordered=np.take_along_axis(obs['uv'],np.argsort(obs['model_joint_ids'],axis=1)[...,None],axis=1)
    ankles=ordered[:,15:17]
    in_image=np.isfinite(ankles).all((1,2))&np.all((ankles>[25.,25.])&(ankles<[1895.,1045.]),axis=(1,2))
    bounds=np.zeros((n,4),np.int32);fields=np.zeros((n,size,size),np.float16)
    active=np.zeros(n,bool);agreement=np.zeros(n,np.float32)
    reviewed={}
    for name in ('early','middle','late'):
        path=args.work/f'source_review/occlusion_reads_{name}.json'
        for row in json.loads(path.read_text(encoding='utf8'))['rows']:
            for point in row['observations']:
                reviewed.setdefault(point['pts'],{}).setdefault(point['track_id'],[]).append(point)
    reader=SourceVideo(media['path'],[1,12800])
    net=VideoMasks(args.weights,image_size=512)
    with (args.work/'whole_continuity/progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        def emit(**values):
            text=json.dumps(values);log.write(text+'\n');print(text,flush=True)
        emit(event='bidirectional_person_foreground_started',source_frames=n,source_sha256=media['sha256'],
             model_sha256=digest(args.weights/'model.safetensors'),device='cuda',image_size=512)
        for shot in dict.fromkeys(shots):
            frames=np.flatnonzero(shots==shot)
            frames=frames[in_image[frames]]
            if args.only_seconds:
                frames=frames[np.min(abs(pts[frames,None]/12800.-np.asarray(args.only_seconds)),axis=1)<=.40]
            if not len(frames):continue
            crops=[];prompts=[]
            for i in frames:
                xy=obs['uv'][i,np.argsort(obs['model_joint_ids'][i])]
                nav=np.r_[xy[11:17],shoe['region'][i].reshape(-1,2)];good=np.isfinite(nav).all(1)
                good&=(nav[:,0]>-120)&(nav[:,0]<2040)&(nav[:,1]>0)&(nav[:,1]<1200)
                if good.sum()<3:raise ValueError('Source body navigation missing')
                margin=max(36.,.20*np.linalg.norm(np.ptp(nav[good],axis=0)))
                lo=np.maximum([0,0],np.floor(nav[good].min(0)-margin)).astype(int)
                hi=np.minimum([1920,1080],np.ceil(nav[good].max(0)+margin)).astype(int)
                if np.min(hi-lo)<16:raise ValueError('Invalid source lower-body crop')
                bounds[i]=np.r_[lo,hi];rgb=reader.read(int(pts[i]))
                crops.append(np.ascontiguousarray(rgb[lo[1]:hi[1],lo[0]:hi[0],::-1]))
                positive=list(np.r_[xy[11:13].mean(0)[None],xy[13:15]])
                boot_seeds=[]
                # White boots are frequently excluded from a person mask.
                # Source boot interiors explicitly prompt the SAME foreground
                # object; they do not create left/right anatomical identities.
                for foot in (0,1):
                    poly=shoe['region'][i,foot]
                    if shoe['role'][i,foot]==1 and np.isfinite(poly).all():
                        m=cv2.moments(poly)
                        if m['m00']>1:
                            centre=np.array([m['m10']/m['m00'],m['m01']/m['m00']])
                            distance=np.linalg.norm(xy[15:17]-centre,axis=1)
                            shins=np.linalg.norm(xy[15:17]-xy[13:15],axis=1)
                            if np.any(distance<8.+.30*shins):boot_seeds.append(centre)
                if int(pts[i]) in reviewed:boot_seeds=[]
                for points in reviewed.get(int(pts[i]),{}).values():
                    known={r['feature']:np.array(r['uv']) for r in points}
                    if 'cuff_center' in known and ('heel' in known or 'toe' in known):
                        boot_seeds.append(np.mean(list(known.values()),axis=0))
                    elif 'heel' in known and 'toe' in known:
                        boot_seeds.append((known['heel']+known['toe'])/2.)
                    elif 'cuff_center' in known:
                        cuff=known['cuff_center'];knee=xy[13:15][np.linalg.norm(xy[13:15]-cuff,axis=1).argmin()]
                        v=cuff-knee;length=np.linalg.norm(v)
                        boot_seeds.append(cuff+v/max(length,1.)*np.clip(length*.12,4.,14.))
                positive=np.asarray(positive)-lo
                h,w=crops[-1].shape[:2]
                negative=np.array([[3,3],[w-4,3],[3,h-4],[w-4,h-4]])
                inside=(positive[:,0]>2)&(positive[:,0]<w-3)&(positive[:,1]>2)&(positive[:,1]<h-3)
                points=np.r_[positive[inside],negative]
                prompt={0:(points.tolist(),[1]*int(inside.sum())+[0]*4,2)}
                # Boot objects have separate segmentation memories, then are
                # UNIONED. These image object ids carry no anatomical identity.
                seeds=[]
                for seed in reversed(boot_seeds):
                    if all(np.linalg.norm(seed-q)>12. for q in seeds):seeds.append(seed)
                seeds=sorted(seeds,key=lambda q:q[0])[:2]
                for object_id,seed in enumerate(seeds,1):
                    p=seed-lo
                    if (p>2).all() and (p<[w-3,h-3]).all():
                        negative_boot=np.r_[xy[13:15]-lo,negative]
                        prompt[object_id]=(np.r_[p[None],negative_boot].tolist(),[1]+[0]*len(negative_boot),2)
                prompts.append(prompt)
            masks=[]
            for reverse in (False,True):
                net.reset();bank=[None]*len(frames)
                order=range(len(frames)-1,-1,-1) if reverse else range(len(frames))
                for count,k in enumerate(order):
                    if count and abs(int(frames[k])-int(frames[previous_k]))!=1:
                        net.reset()
                    reset=(count==0 or (count and abs(int(frames[k])-int(frames[previous_k]))!=1))
                    # Periodic native source prompts bound memory drift. They
                    # use detector navigation only, never a rendered silhouette.
                    result=net.step(crops[k],count,prompts[k] if reset or count%10==0 or int(pts[frames[k]]) in reviewed else {})
                    good=[]
                    for object_id,(mask,score) in result.items():
                        if score<=0:continue
                        if object_id and (mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any()):
                            # A boot memory leaking onto an ice stripe usually
                            # reaches the crop edge; it is not foreground proof.
                            continue
                        good.append(mask)
                    if good:bank[k]=np.bitwise_or.reduce(good)
                    previous_k=k
                    if count%250==0:emit(event='person_foreground_pass',shot=str(shot),reverse=reverse,
                        processed=count,pts=int(pts[frames[k]]))
                masks.append(bank)
            for k,i in enumerate(frames):
                a,b=masks[0][k],masks[1][k]
                if a is None or b is None:continue
                union=a|b;agreement[i]=np.count_nonzero(a&b)/max(1,np.count_nonzero(union))
                fraction=union.mean()
                if agreement[i]<.55 or not .025<fraction<.84:continue
                # Union preserves a real visible boot missed in one direction;
                # it never requires hidden geometry to coincide with the front boot.
                union=cv2.morphologyEx(union,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                distance=cv2.distanceTransform(1-union,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                # Eight native pixels accommodate blur, fine steel and uncertain
                # source boundary. This is an allowance, not measured precision.
                distance=np.maximum(distance-8.,0.)
                fields[i]=cv2.resize(distance,(size,size),interpolation=cv2.INTER_LINEAR)
                active[i]=True
            emit(event='person_foreground_shot_ready',shot=str(shot),frames=len(frames),active=int(active[frames].sum()))
    reader.close()
    pending=args.output.with_suffix('.pending.npz')
    np.savez_compressed(pending,pts=pts,shot_ids=shots,bounds=bounds,distance=fields,active=active,
        bidirectional_iou=agreement,source_video_sha256=np.array(media['sha256']),
        source_observations_sha256=np.array(digest(args.work/'input/body_track_source.npz')),
        interpretation=np.array('RGB foreground union with 8 native pixel allowance. Weak outside-only constraint; segmentation is not truth or an anatomical association.'))
    pending.replace(args.output)
    print(json.dumps(dict(event='person_foreground_saved',path=str(args.output),sha256=digest(args.output),
                         active=int(active.sum()),frames=n)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('work','registration','weights','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--only-seconds',nargs='+',type=float)
    p.add_argument('--review',action='store_true')
    p.add_argument('--review-all',action='store_true');p.add_argument('--qualify',action='store_true')
    p.add_argument('--crosscheck-outlines',action='store_true')
    args=p.parse_args()
    if args.crosscheck_outlines:crosscheck_outlines(args)
    elif args.qualify:qualify(args)
    elif args.review_all:review_all(args)
    elif args.review:review(args)
    else:run(args)
