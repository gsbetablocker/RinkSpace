"""Exact native-source diagnostics from the same ModelBundle/operator as fitting.

Selected stills are diagnostics, never whole-shot acceptance. This module does
not alter geometry, hide residuals, or launch a video player.
"""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
import json
import re
import numpy as np
import cv2
from .schema import SourceClock,SourceFrame
from .sources import NativeVideoReader

COLORS={'grid':(100,165,100),'ice':(220,180,30),'board':(0,200,230),
        'advertisement':(230,105,20),'advertisement_print':(200,80,180)}

def scene_paths(model,*,samples=128):
    inventory=model.inventory(samples=samples);rows=[]
    for c in inventory['curves']:
        rows.append({'id':c['id'],'label':c['label'],'group':c.get('group','ice'),
                     'xyz':np.asarray(c['xyz'],dtype=np.float64),'closed':c['kind']=='ice_circle'})
    for g in inventory['grid']:
        rows.append({'id':g['id'],'label':g['id'],'group':'grid','xyz':np.asarray(g['xyz'],dtype=np.float64),'closed':False})
    for suffix in ('bottom','top'):
        rows.append({'id':'board_'+suffix,'label':'board_'+suffix,'group':'board','xyz':np.asarray(inventory['board']['xyz_'+suffix],dtype=np.float64),'closed':True})
    for a in inventory['installations']:
        if a.get('boundary_xyz'):
            rows.append({'id':a['id']+'/boundary','label':a['id'].split('/')[-1],
                         'group':'advertisement','xyz':np.asarray(a['boundary_xyz'],dtype=np.float64),'closed':True})
    return rows

def draw_projection(image,paths,projected,valid):
    overlay=image.copy();height,width=image.shape[:2];offset=0;labels=set();drawn=[]
    for path in paths:
        count=len(path['xyz']);uv=projected[offset:offset+count];ok=valid[offset:offset+count];offset+=count
        group=path['group'];color=COLORS.get(group,(180,180,180))
        if path['label'].startswith('ice.C:'):color=(30,40,245)
        elif path['label'].startswith('ice.C0:'):color=(20,170,235)
        elif path['label'].startswith('ice.B'):color=(245,95,45)
        indices=list(range(count))
        if path['closed'] and count>2:indices.append(0)
        first_hit=None
        for a,b in zip(indices[:-1],indices[1:]):
            if not ok[a] or not ok[b] or np.max(np.abs(uv[[a,b]]))>1e6:continue
            p,q=tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int))
            hit,start,end=cv2.clipLine((0,0,width,height),p,q)
            if hit:
                cv2.line(overlay,start,end,color,1,cv2.LINE_AA)
                if first_hit is None:first_hit=start
        label=path['label']
        match=re.search(r'Ad[0-9]+',label)
        if match:label=match[0]
        elif group=='ice':label=label.split(':',1)[0]
        if first_hit is not None:
            drawn.append(path['id'])
            if group!='grid' and label not in labels:
                labels.add(label);cv2.putText(overlay,label,first_hit,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return overlay,drawn


def native_projection_preview(model,native_pts,bgr,*,shot_id='S01',samples=128):
    """One frozen current bundle, the formal RS operator and every model path.

    No model publication, observation selection, source production or solver call.
    The returned identity belongs to this exact still even if a later numerical
    step publishes another state while the read-only preview is being computed.
    """
    if type(native_pts) is not int:
        raise ValueError('preview requires exact integer native PTS')
    op=model.operator(shot_id)
    op.frame_time(np.asarray(native_pts,dtype=np.int64))
    if bgr.ndim!=3 or bgr.shape!=(1080,1920,3):
        raise ValueError('preview requires the unscaled 1920x1080 native source')
    paths=scene_paths(model,samples=samples)
    world=np.concatenate([path['xyz'] for path in paths])
    projected=op.project_rs(world,np.full(len(world),native_pts,dtype=np.int64))
    overlay,drawn=draw_projection(bgr,paths,projected['uv'],projected['status_code']==0)
    shot=next(s for s in model.shots if s.id==shot_id)
    seconds=native_pts*shot.time_base_num/shot.time_base_den
    title=f'{shot_id} nativePTS={native_pts} source={seconds:.3f}s | model r{model.revision} params r{model.state.revision} | NOT ACCEPTED'
    cv2.rectangle(overlay,(0,0),(1920,31),(0,0,0),-1)
    cv2.putText(overlay,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    statuses,counts=np.unique(projected['status_code'],return_counts=True)
    identity=dict(schema='rinksolve.native_projection_preview.v1',
        model_id=model.id,model_revision=model.revision,parameter_state_id=model.state.id,
        parameter_revision=model.state.revision,parameter_sha256=model.state.digest(),
        shot_id=shot_id,native_pts=native_pts,source_seconds=seconds,
        all_model_paths_enabled=True,all_path_count=len(paths),visible_path_count=len(drawn),
        projected_world_samples=len(world),
        formal_projection_status_counts={str(int(k)):int(n) for k,n in zip(statuses,counts)},
        certificate_policy_id=op.identity["certificate_policy_id"],
        partition_certificate_point_count=int(np.sum(projected["certificate"]["sensor_partition_count"]>1)),
        whole_shot_acceptance=False,person_mask_used=False,invalid_roots_clamped=False)
    return overlay,identity

def render_selected(model,source,pts,output,*,data_root,shot_id='S01',samples=128):
    shot=next(s for s in model.shots if s.id==shot_id);op=model.operator(shot_id)
    requested_pts=np.asarray(pts)
    if requested_pts.ndim!=1 or requested_pts.dtype.kind not in "iu":
        raise ValueError("selected review frames require exact integer native PTS")
    values=sorted(set(int(p) for p in requested_pts));op.frame_time(np.asarray(values,dtype=np.int64))
    clock=SourceClock(shot.clock_id,shot.time_base_num,shot.time_base_den,'current_official_video')
    requested=tuple(SourceFrame(f'{shot_id}:{p}',clock.id,p,1920,1080,'current_official_video') for p in values)
    reader=NativeVideoReader(source,data_root,clock,1920,1080,50,1)
    paths=scene_paths(model,samples=samples);world=np.concatenate([p['xyz'] for p in paths])
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frames=[]
    for frame in reader.frames(requested):
        projected=op.project_rs(world,np.full(len(world),frame.native_pts,dtype=np.int64))
        valid=projected['status_code']==0
        bgr=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
        overlay,drawn=draw_projection(bgr,paths,projected['uv'],valid)
        title=f'{shot_id} nativePTS={frame.native_pts} source={frame.source_seconds:.3f}s | model r{model.revision} params r{model.state.revision} | NOT ACCEPTED'
        cv2.rectangle(overlay,(0,0),(1920,31),(0,0,0),-1)
        cv2.putText(overlay,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        # Native-size paired image preserves both source and exact overlay pixels.
        pair=np.concatenate((bgr,overlay),axis=1)
        path=output/f'pts{frame.native_pts}_pair.jpg'
        if not cv2.imwrite(str(path),pair,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError(str(path))
        statuses,counts=np.unique(projected['status_code'],return_counts=True)
        frames.append({'native_pts':frame.native_pts,'source_seconds':frame.source_seconds,'path':str(path),
            'projected_world_samples':len(world),'formal_projection_status_counts':{str(int(k)):int(n) for k,n in zip(statuses,counts)},
            'visible_path_ids':drawn,'all_model_paths_enabled':True})
    report={'schema':'rinksolve.same_model_native_stills.v1','model_id':model.id,'model_revision':model.revision,
        'parameter_state_id':model.state.id,'parameter_revision':model.state.revision,'parameter_sha256':model.state.digest(),
        'operator_identity':op.identity,'source':str(source),'time_base':f'{clock.time_base_num}/{clock.time_base_den}',
        'frames':frames,'all_path_count':len(paths),'whole_shot_acceptance':False,
        'limitations':['Selected stills only, not continuous full-shot review.','No person mask or board occlusion removal; every enabled physical model path is shown.',
                       'Projection rows without an interior sensor root are not fabricated or clamped.','Grid is a model diagnostic, not observed paint.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
