"""Audit source footwear visibility before using any proposed silhouette.

The source RGB, not the fitted silhouette, qualifies boundary arcs. Both boots
remain separate physical objects; overlapping masks do not become two complete
observations. All native exposures are retained in the audit, including gaps.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from source_review import SourceVideo, digest, whole_skate_observations


def prepare_points(work):
    import copy
    programme=Path(__file__).resolve().parents[2]
    path=whole_skate_observations(programme)
    doc=json.loads(path.read_text(encoding='utf8'))
    doc['samples']=[r for r in doc['samples'] if r.get('source_review_batch')!='source_occlusion_1116']
    added=[];counts={};evidence=[]
    for name in ('early','middle','late'):
        source=work/f'source_review/occlusion_reads_{name}.json'
        annotations=json.loads(source.read_text(encoding='utf8'));counts[name]=0
        evidence.append(dict(path=str(source),sha256=digest(source)))
        for row in annotations['rows']:
            for number,p in enumerate(row['observations']):
                stamp=int(p['pts']);group=f"occlusion:R{row['row']}:{p['track_id']}"
                if p.get('track_link_confirmed') is False:group+=f':unlinked:{stamp}'
                item=dict(id=f'occlusion:R{row["row"]}:{number}:{stamp}:{p["feature"]}',
                    pts=stamp,foot=0 if p['foot'] is None else int(p['foot']),feature=p['feature'],uv=p['uv'],
                    role='fit',normalization_px=float(p['normalization_px']),weight=9.,
                    source_kind='assistant_native_source_review',source_review_batch='source_occlusion_1116',
                    source_image=str(work/f'source_review/occlusion_source_{row["row"]//4:02d}.jpg'),
                    source_annotation=str(source),source_review_note=row['observation_note']+' '+p.get('note',''),
                    anatomical_identity_confirmed=False,human_confirmed=False,
                    flow_before_seconds=.08,flow_after_seconds=.08)
                if p.get('pixel_box') is not None:
                    box=np.asarray(p['pixel_box'],float).reshape(2,2)
                    if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid reviewed native pixel box')
                    item['pixel_box']=box.tolist()
                if p['foot'] is None:
                    item.update(foot_candidates=[0,1],association_group=f'{group}:{stamp}',
                        association_track=group,association_exclusion_group=f'occlusion:R{row["row"]}:pair:{stamp}')
                added.append(item);counts[name]+=1
    doc['samples'].extend(added)
    outline=work/'input/shoe_occlusion_source.npz';foreground=work/'input/person_foreground.npz'
    stages=copy.deepcopy(doc['whole_source_fidelity_intervals'])
    for number,stage in enumerate(stages):
        stage['name']=f'whole_occlusion_source_{number:02d}'
        stage['outline_source'].update(path=str(outline),sha256=digest(outline))
        stage['source_foreground']=dict(path=str(foreground),sha256=digest(foreground))
        stage['foreground_weight']=432.
    doc['whole_occlusion_source_intervals']=stages
    doc['source_occlusion_revision']=dict(evidence=evidence,added_samples=counts,
        source_frames_screened=8329,physical_identity='latent whole source tracks; simultaneous distinct boots are exclusive',
        masks='RGB-qualified partial arcs, no full hidden outline completion',
        containment='bidirectional conservative source person+shoe foreground; outside-only weak constraint',
        sparse_review_not_whole_video_acceptance=True)
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(event='source_occlusion_points_prepared',new_points=counts,
        direct_points=len(added),path=str(path),sha256=digest(path),whole_programme_stages=len(stages))),flush=True)


def prepare_blade_points(work):
    """Add independently reread directions; retain rejected coordinates verbatim."""
    import copy
    programme=Path(__file__).resolve().parents[2]
    path=whole_skate_observations(programme)
    doc=json.loads(path.read_text(encoding='utf8'))
    doc['samples']=[r for r in doc['samples'] if r.get('source_review_batch')!='source_blade_1116']
    by_id={r.get('id'):r for r in doc['samples']}
    counts={};rejected=[];added=[];evidence=[];duplicates=[]
    for name in ('early','middle','late'):
        source=work/f'source_review/blade_reads_{name}.json'
        annotations=json.loads(source.read_text(encoding='utf8'))
        old=json.loads((work/f'source_review/occlusion_reads_{name}.json').read_text(encoding='utf8'))
        previous={}
        for row in old['rows']:
            for number,p in enumerate(row['observations']):
                key=f'occlusion:R{row["row"]}:{number}:{p["pts"]}:{p["feature"]}'
                if key in by_id:
                    previous.setdefault((row['row'],int(p['pts']),str(p['track_id'])),[]).append(by_id[key])
        evidence.append(dict(path=str(source),sha256=digest(source)));counts[name]=0
        for item in annotations.get('rejected_prior_observations',[]):
            matches=[r for r in doc['samples'] if r.get('source_kind')=='assistant_native_source_review'
                and r.get('source_review_batch')=='source_occlusion_1116'
                and r['id'].startswith(f'occlusion:R{item["row"]}:')
                and r['pts']==int(item['pts']) and r['feature']==item['feature']
                and np.linalg.norm(np.asarray(r['uv'])-item['uv'])<2.]
            if len(matches)!=1:raise ValueError(f'A rejected source annotation is not uniquely identified: {item}')
            parent=matches[0]
            for row in doc['samples']:
                if row.get('id')==parent['id'] or row.get('source_anchor_id')==parent['id']:
                    row['role']='rejected';row['source_review_rejection']=dict(reason=item['reason'],source=str(source))
            rejected.append(dict(id=parent['id'],**item))
        for row in annotations['rows']:
            for number,p in enumerate(row['observations']):
                stamp=int(p['pts']);feature=p['feature']
                if feature not in ('toe','heel','runner_front','runner_rear'):
                    raise ValueError('A direction review must identify a visible shoe/blade endpoint')
                prior=previous.get((row['row'],stamp,str(p.get('prior_track_id'))),[])
                usable=[r for r in prior if r['role']=='fit']
                if any(r['feature']==feature and np.linalg.norm(np.asarray(r['uv'])-p['uv'])<6. for r in usable):
                    duplicates.append(dict(row=row['row'],pts=stamp,feature=feature));continue
                group=f'blade:R{row["row"]}:{p["track_id"]}'
                if not p.get('track_link_confirmed',False):group+=f':unlinked:{stamp}'
                item=dict(id=f'blade:R{row["row"]}:{number}:{stamp}:{feature}',pts=stamp,
                    foot=0 if p.get('foot') is None else int(p['foot']),feature=feature,uv=p['uv'],
                    role='fit',normalization_px=float(p['normalization_px']),weight=9.,
                    source_kind='assistant_native_source_review',source_review_batch='source_blade_1116',
                    source_image=str(work/f'source_review/occlusion_source_{row["row"]//4:02d}.jpg'),
                    source_annotation=str(source),source_review_note=row['observation_note']+' '+p.get('note',''),
                    anatomical_identity_confirmed=False,human_confirmed=False,
                    flow_before_seconds=.08,flow_after_seconds=.08)
                if p.get('pixel_box') is not None:
                    box=np.asarray(p['pixel_box'],float).reshape(2,2)
                    if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid source endpoint box')
                    item['pixel_box']=box.tolist()
                if p.get('foot') is None:
                    item.update(foot_candidates=[0,1],association_group=f'{group}:{stamp}',
                        association_track=group,association_exclusion_group=f'blade:R{row["row"]}:pair:{stamp}')
                    if usable:
                        identities={(r.get('association_track'),r.get('association_exclusion_group')) for r in usable}
                        if len(identities)!=1:raise ValueError('Prior source boot identity is inconsistent')
                        for key in ('association_group','association_track','association_exclusion_group'):
                            item[key]=usable[0][key]
                        item['prior_source_boot_id']=p['prior_track_id']
                added.append(item);counts[name]+=1
    doc['samples'].extend(added)
    stages=copy.deepcopy(doc['whole_occlusion_source_intervals'])
    for number,stage in enumerate(stages):
        stage['name']=f'whole_direction_source_{number:02d}'
        stage['occlusion_anatomy_multipliers']=dict(partial=4.,missing=16.)
    doc['whole_direction_source_intervals']=stages
    doc['source_blade_direction_revision']=dict(evidence=evidence,added_samples=counts,
        rejected_prior_observations=rejected,already_observed_endpoints_not_duplicated=duplicates,
        semantics='Visible upper/outsole heel and toe; metal runner rear and curved front endpoints; hidden endpoints not invented',
        missing_direction_prior='Authored knee/ankle envelope retained more strongly for partial or missing source outlines')
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(event='source_blade_directions_prepared',new_points=counts,
        rejected=len(rejected),duplicate_observations_skipped=len(duplicates),sha256=digest(path))),flush=True)


def prepare_residual_points(work):
    programme=Path(__file__).resolve().parents[2]
    path=whole_skate_observations(programme)
    doc=json.loads(path.read_text(encoding='utf8'))
    doc['samples']=[r for r in doc['samples'] if r.get('source_review_batch')!='source_residual_1116']
    source=work/'source_review/direction_residual_reads.json'
    annotations=json.loads(source.read_text(encoding='utf8'));added=[]
    for row in annotations['rows']:
        for number,p in enumerate(row['observations']):
            stamp=int(p['pts']);group=f'residual:R{row["row"]}:{p["track_id"]}'
            if not p.get('track_link_confirmed',False):group+=f':unlinked:{stamp}'
            item=dict(id=f'residual:R{row["row"]}:{number}:{stamp}:{p["feature"]}',pts=stamp,
                foot=0 if p.get('foot') is None else int(p['foot']),feature=p['feature'],uv=p['uv'],
                role='fit',normalization_px=float(p['normalization_px']),weight=9.,
                source_kind='assistant_native_source_review',source_review_batch='source_residual_1116',
                source_image=str(work/f'source_review/direction_residual_{row["row"]//4:02d}.jpg'),
                source_annotation=str(source),source_review_note=row['observation_note']+' '+p.get('note',''),
                anatomical_identity_confirmed=False,human_confirmed=False,
                flow_before_seconds=.08,flow_after_seconds=.08)
            if p.get('pixel_box') is not None:
                box=np.asarray(p['pixel_box'],float).reshape(2,2)
                if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid source endpoint box')
                item['pixel_box']=box.tolist()
            if p.get('foot') is None:
                item.update(foot_candidates=[0,1],association_group=f'{group}:{stamp}',association_track=group,
                    association_exclusion_group=f'residual:R{row["row"]}:pair:{stamp}')
            added.append(item)
    doc['samples'].extend(added)
    for stage in doc['whole_direction_source_intervals']:
        stage['outline_source']['ambiguous_source_union']=True
    doc['source_residual_direction_revision']=dict(source_path=str(source),source_sha256=digest(source),
        chronological_windows=len(annotations['rows']),added_source_points=len(added),
        source_only_pixels=True,model_used_for_navigation_only=True)
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(event='source_residual_directions_prepared',source_points=len(added),sha256=digest(path))),flush=True)


def polygon_overlap(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0., np.zeros(len(a), bool), np.zeros(len(b), bool)
    lo=np.floor(np.minimum(a.min(0),b.min(0))-3).astype(int)
    hi=np.ceil(np.maximum(a.max(0),b.max(0))+3).astype(int)
    if np.max(hi-lo)>700:return 0.,np.zeros(len(a),bool),np.zeros(len(b),bool)
    masks=[]
    for p in (a,b):
        m=np.zeros(tuple((hi-lo)[::-1]),np.uint8)
        cv2.fillPoly(m,[np.rint(p-lo).astype(np.int32)],1);masks.append(m)
    fraction=float(np.count_nonzero(masks[0]&masks[1])/max(1,min(m.sum() for m in masks)))
    inside=[np.array([cv2.pointPolygonTest(np.asarray(q,np.float32),tuple(map(float,x)),True)>-2.
                     for x in p]) for p,q in ((a,b),(b,a))]
    return fraction,*inside


def qualify_arcs(rgb, polygons):
    """Exclude unsupported and extended background edges; do not fill occlusion."""
    valid=np.zeros((2,80),bool);line_hit=valid.copy();contrast=np.zeros((2,80),np.float32)
    overlap,inside_a,inside_b=polygon_overlap(*polygons)
    for foot,p in enumerate(polygons):
        if not np.isfinite(p).all():continue
        extent=np.ptp(p,axis=0);span=max(float(np.linalg.norm(extent)),12.)
        lo=np.maximum(0,np.floor(p.min(0)-span).astype(int))
        hi=np.minimum(rgb.shape[1::-1],np.ceil(p.max(0)+span).astype(int))
        if np.min(hi-lo)<8:continue
        gray=cv2.cvtColor(rgb[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_RGB2GRAY)
        gray=cv2.GaussianBlur(gray,(0,0),.65)
        tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
        normal=tangent[:,::-1]*[-1.,1.]
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1.e-6)
        def sample(x):
            q=(x-lo).astype(np.float32)
            return cv2.remap(gray,q[:,0,None],q[:,1,None],cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)[:,0].astype(float)
        contrast[foot]=abs(sample(p+normal*2.)-sample(p-normal*2.))
        # A rink stripe/edge extends well outside the shoe. Short blade edges
        # are never rejected merely for being straight.
        edges=cv2.Canny(gray,25,65)
        lines=cv2.HoughLinesP(edges,1,np.pi/360,threshold=max(18,int(span*.55)),
            minLineLength=max(30,int(span*1.65)),maxLineGap=5)
        if lines is not None:
            for x0,y0,x1,y1 in lines[:,0]:
                a=np.array([x0,y0])+lo;b=np.array([x1,y1])+lo;v=b-a
                length=np.linalg.norm(v);station=(p-a)@v/max(length**2,1.)
                distance=abs((p[:,0]-a[0])*v[1]-(p[:,1]-a[1])*v[0])/max(length,1.)
                parallel=abs(tangent@v)/np.maximum(np.linalg.norm(tangent,axis=1)*length,1.)
                line_hit[foot]|=(distance<3.)&(station>-.1)&(station<1.1)&(parallel>.85)
        valid[foot]=(contrast[foot]>=5.)&~line_hit[foot]
        # The interior boundary of another source boot is not this boot's
        # exterior. A duplicate detection has no separately identified arcs.
        if overlap>.06:valid[foot]&=~(inside_a if foot==0 else inside_b)
    return valid,line_hit,contrast,overlap


def audit(work,registration,output):
    packet=dict(np.load(work/'input/shoe_turn_source.npz'))
    body=dict(np.load(work/'input/body_track_source.npz'))
    current=dict(np.load(work/'source_review/shoe_screen_after.npz'))
    pts=packet['source_pts'];np.testing.assert_array_equal(pts,current['pts'])
    media=json.loads(registration.read_text(encoding='utf8'))['primary_source']
    reader=SourceVideo(media['path'],packet['time_base'])
    valid=np.zeros((len(pts),2,80),bool);lines=valid.copy();contrast=np.zeros_like(valid,dtype=np.float32)
    overlap=np.zeros(len(pts),np.float32);bounds=np.zeros((len(pts),4),int)
    # Only crop navigation may use current geometry when source tracking is
    # absent. No rendered/model pixel is admitted as an observation.
    for i,stamp in enumerate(pts):
        rgb=reader.read(int(stamp));polygons=packet['region'][i]
        valid[i],lines[i],contrast[i],overlap[i]=qualify_arcs(rgb,polygons)
        valid[i]&=(packet['role'][i]==1)[:,None]
        legs=body['uv'][i,np.isin(body['model_joint_ids'][i],[11,12,13,14,15,16])]
        nav=np.r_[legs,current['contour'][i].reshape(-1,2),polygons.reshape(-1,2)]
        nav=nav[np.isfinite(nav).all(1)]
        nav=nav[(nav[:,0]>-100)&(nav[:,0]<2020)&(nav[:,1]>0)&(nav[:,1]<1180)]
        if not len(nav):nav=np.array([[800,600],[1100,1079]])
        lo=np.maximum([0,0],np.floor(nav.min(0)-25)).astype(int)
        hi=np.minimum([1920,1080],np.ceil(nav.max(0)+25)).astype(int)
        bounds[i]=np.r_[lo,hi]
        if i%400==0:print(json.dumps(dict(event='source_occlusion_rgb_audit',frame=i,pts=int(stamp))),flush=True)
    reader.close()
    coverage=valid.mean(-1)
    # Partial boundaries are one-way source-to-model arcs. Full symmetric
    # silhouette matching is reserved for resolved, predominantly visible feet.
    partial=(overlap[:,None]>.06)|(coverage<.78)|lines.any(-1)
    usable=(packet['role']==1)&(valid.sum(-1)>=8)&(overlap[:,None]<.65)
    revised=dict(packet,boundary_valid=valid,partial_boundary=partial,
                 source_pair_overlap=overlap,source_extended_line=lines,
                 source_boundary_contrast=contrast)
    revised['role']=np.where(usable,packet['role'],0).astype(np.uint8)
    revised['measure_s']=np.where(usable,packet['measure_s'],0.)
    revised['occlusion_revision_json']=np.array(json.dumps(dict(
        parent_sha256=digest(work/'input/shoe_turn_source.npz'),source_rgb_sha256=media['sha256'],
        source_pixels_changed=False,partial_contours_not_completed=True,
        duplicate_masks_not_independent_feet=True,source_model_is_navigation_only=True)))
    np.savez_compressed(output,**revised)
    # Chronological 2 s bins include every part of the programme. Rank finds
    # source ambiguities, never removes them from the reported denominator.
    source_missing=(packet['role']!=1).sum(1)
    rank=overlap*4+(1-coverage).mean(1)+source_missing*.5+lines.mean((1,2))*4
    centres=[]
    for first in range(0,len(pts),100):
        ids=np.arange(first,min(len(pts),first+100));centres.append(int(ids[np.argmax(rank[ids])]))
    centres.extend(int(np.searchsorted(pts,round(t*12800))) for t in [52.26,61.58,62.72,68.56,75.66])
    centres=sorted(set(centres));rows=[]
    for k,i in enumerate(centres):
        frames=np.clip(i+np.array([-16,-8,0,8,16]),0,len(pts)-1)
        frames=np.where(packet['shot_ids'][frames]==packet['shot_ids'][i],frames,i)
        lo=bounds[frames,:2].min(0);hi=bounds[frames,2:].max(0)
        rows.append(dict(index=k,centre_index=i,frames=frames.tolist(),pts=pts[frames].tolist(),
            seconds=(pts[frames]/12800).tolist(),crop=np.r_[lo,hi].tolist(),
            overlap=float(overlap[i]),missing=int(source_missing[i]),coverage=coverage[i].tolist()))
    report=dict(schema='rinkspace.source_occlusion_review.v1',source_frames=len(pts),
        source_sha256=media['sha256'],baseline_sha256=str(current['parameter_sha256']),
        source_seconds=(pts[[0,-1]]/12800).tolist(),qualified_outline=str(output),
        qualified_outline_sha256=digest(output),full_symmetric_foot_frames=int((usable&~partial).sum()),
        partial_foot_frames=int((usable&partial).sum()),excluded_or_missing_foot_frames=int((~usable).sum()),
        overlapping_frames=int((overlap>.06).sum()),duplicate_frames=int((overlap>.65).sum()),
        line_affected_foot_frames=int(lines.any(-1).sum()),rows=rows,
        scope='all native exposures screened from RGB; sparse sequence boards are development review, not whole-video acceptance')
    (work/'source_review/occlusion_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


def boards(work,registration):
    report=json.loads((work/'source_review/occlusion_review.json').read_text(encoding='utf8'))
    media=json.loads(registration.read_text(encoding='utf8'))['primary_source']
    reader=SourceVideo(media['path'],[1,12800]);font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    for first in range(0,len(report['rows']),4):
        page=Image.new('RGB',(2000,1440),(20,25,29));draw=ImageDraw.Draw(page)
        for rowpos,row in enumerate(report['rows'][first:first+4]):
            x,y,r,b=row['crop'];scale=min(396/max(1,r-x),320/max(1,b-y))
            row['scale']=scale;row['page']=first//4;row['row']=rowpos
            row['image_origin']=[2,rowpos*360+38]
            for col,stamp in enumerate(row['pts']):
                crop=Image.fromarray(reader.read(int(stamp))[y:b,x:r])
                crop=crop.resize((max(1,round((r-x)*scale)),max(1,round((b-y)*scale))))
                page.paste(crop,(col*400+2,rowpos*360+38))
                draw.text((col*400+3,rowpos*360+3),f"R{row['index']:02d} C{col} {stamp/12800:.2f}s",font=font,fill='white')
        page.save(work/f'source_review/occlusion_source_{first//4:02d}.jpg',quality=96)
    reader.close();(work/'source_review/occlusion_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(event='source_occlusion_boards_ready',pages=(len(report['rows'])+3)//4,rows=len(report['rows']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True,type=Path);p.add_argument('--registration',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--boards',action='store_true')
    p.add_argument('--prepare-points',action='store_true')
    p.add_argument('--prepare-blade-points',action='store_true')
    p.add_argument('--prepare-residual-points',action='store_true')
    a=p.parse_args()
    if a.prepare_residual_points:prepare_residual_points(a.work)
    elif a.prepare_blade_points:prepare_blade_points(a.work)
    elif a.prepare_points:prepare_points(a.work)
    elif a.boards:boards(a.work,a.registration)
    else:audit(a.work,a.registration,a.output)
