"""Mechanical native-pixel source crops for an existing outlier audit.

No image interpretation, point extraction, native projection or fit is performed.
The left crop is untouched; the right crop marks only existing audited source UV.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib
import json

import cv2
import numpy as np

from programme_source_frames import iter_registration_source_frames
from rinkspace.path_policy import guard_project_write


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def source_crop_box(uv,width=1920,height=1080,margin=60,minimum=(180,100)):
    uv=np.asarray(uv,float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not len(uv)or not np.isfinite(uv).all():
        raise ValueError('A source crop needs actual finite UV[N,2]')
    if np.any(uv<0)or np.any(uv>[width-1,height-1]):raise ValueError('Audited source UV is outside the native frame')
    lo=np.floor(uv.min(0)-margin).astype(int);hi=np.ceil(uv.max(0)+margin+1).astype(int)
    for axis,limit in enumerate((width,height)):
        target=min(limit,max(int(hi[axis]-lo[axis]),minimum[axis]))
        center=(lo[axis]+hi[axis])/2
        lo[axis]=min(max(0,int(np.floor(center-target/2))),limit-target)
        hi[axis]=lo[axis]+target
    return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]


def family(path):
    return path.split('::',1)[0]if path.startswith('print_instance/')else path


def tier(error):
    return '01_gt80'if error>80 else'02_gt40_to80'if error>40 else'03_gt20_to40'


def label_positions(uv,width,height):
    occupied=[];result=[]
    for number,(x,y)in enumerate(uv,1):
        text=str(number);(w,h),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.38,1)
        candidates=[]
        for radius in (7,14,22,32):
            for dx,dy in ((1,-1),(1,1),(-1,-1),(-1,1),(0,-1),(0,1),(1,0),(-1,0)):
                a=int(np.clip(round(x+dx*radius-(w if dx<0 else 0)),1,max(1,width-w-2)))
                b=int(np.clip(round(y+dy*radius),h+2,max(h+2,height-baseline-2)))
                box=(a-1,b-h-1,a+w+1,b+baseline+1)
                overlap=sum(max(0,min(box[2],other[2])-max(box[0],other[0]))*
                    max(0,min(box[3],other[3])-max(box[1],other[1]))for other in occupied)
                candidates.append((overlap,radius,a,b,box))
        _,_,a,b,box=min(candidates)
        occupied.append(box);result.append((a,b))
    return result


def paint_case(image,rows,label_rows,crop,heading):
    x0,y0,x1,y1=crop
    original=image[y0:y1,x0:x1].copy();marked=original.copy()
    for row in rows:
        point=tuple(np.rint(np.asarray(row['uv'])-[x0,y0]).astype(int))
        cv2.circle(marked,point,2,(80,80,80),-1,cv2.LINE_AA)
    xy=np.array([row['uv']for row in label_rows])-np.array([x0,y0])
    for row,point,position in zip(label_rows,xy,label_positions(xy,original.shape[1],original.shape[0])):
        point=tuple(np.rint(point).astype(int));label=str(row['local_label'])
        cv2.line(marked,point,position,(210,210,210),1,cv2.LINE_AA)
        cv2.circle(marked,point,3,(0,0,0),-1,cv2.LINE_AA)
        cv2.circle(marked,point,2,(40,40,255),-1,cv2.LINE_AA)
        cv2.putText(marked,label,position,cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(marked,label,position,cv2.FONT_HERSHEY_SIMPLEX,.38,(245,245,245),1,cv2.LINE_AA)
    h,w=original.shape[:2];header=52;gap=8
    canvas=np.full((h+header,2*w+gap,3),245,np.uint8)
    canvas[header:,:w]=original;canvas[header:,w+gap:]=marked
    title_width=cv2.getTextSize(heading,cv2.FONT_HERSHEY_SIMPLEX,.42,1)[0][0]
    title_scale=min(.42,.42*(canvas.shape[1]-10)/max(1,title_width))
    cv2.putText(canvas,heading,(5,17),cv2.FONT_HERSHEY_SIMPLEX,title_scale,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(canvas,'ORIGINAL SOURCE | native pixels',(5,39),cv2.FONT_HERSHEY_SIMPLEX,.39,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(canvas,'SAME SOURCE + AUDIT UV',(w+gap+5,39),cv2.FONT_HERSHEY_SIMPLEX,.39,(25,25,25),1,cv2.LINE_AA)
    return canvas


def save_png(path,image):
    temporary=path.with_name(path.stem+'.writing.png')
    if not cv2.imwrite(str(temporary),image,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise OSError('Cannot save source panel')
    temporary.replace(path)


def page_rows(cases):
    result=[];compact=[]
    for case in cases:
        if case['width']>700:
            if compact:result.append(compact);compact=[]
            result.append([case])
        else:
            compact.append(case)
            if len(compact)==3:result.append(compact);compact=[]
    if compact:result.append(compact)
    return result


def build_pages(output,priority,cases):
    rows=page_rows(cases);pages=[]
    for start in range(0,len(rows),2):
        chosen=rows[start:start+2];margin=12;title=32
        width=max(sum(case['width']for case in row)+(len(row)+1)*margin for row in chosen)
        height=title+sum(max(case['height']for case in row)+margin for row in chosen)
        canvas=np.full((height,width,3),235,np.uint8);y=title
        number=len(pages)+1
        cv2.putText(canvas,f'{priority} | page {number:03d} | source evidence; no decisions',(12,21),
            cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
        ids=[]
        for row in chosen:
            x=margin;row_height=max(case['height']for case in row)
            for case in row:
                picture=cv2.imread(str(output/case['image']))
                if picture is None:raise ValueError('Completed source panel is missing')
                canvas[y:y+case['height'],x:x+case['width']]=picture
                ids.append(case['panel_id']);x+=case['width']+margin
            y+=row_height+margin
        path=output/'pages'/f'{priority}_{number:03d}.png';save_png(path,canvas)
        pages.append(dict(priority=priority,page=number,path=str(path),panel_ids=ids,
            width=width,height=height,source_pixel_scale=1.))
    return pages


def normalized_observations(path):
    """Read the audit contract; field mapping is intentionally explicit."""
    records=[]
    with path.open(encoding='utf8')as stream:
        for line_number,line in enumerate(stream,1):
            record=json.loads(line)
            for key in ('ledger_line','row','shot','pts','path_id','observation_id','uv','error_px','frame_id'):
                if key not in record:raise ValueError('Missing audit field: '+key)
            if record['ledger_line']!=line_number:raise ValueError('Audit ledger line numbering differs')
            if not isinstance(record['row'],int)or isinstance(record['row'],bool)or record['row']<0:
                raise ValueError('Audit global row index must be an integer')
            if not isinstance(record['pts'],int)or isinstance(record['pts'],bool):raise ValueError('Audit source PTS must be integer')
            if record.get('time_base')!=[1,12800]:raise ValueError('Audit source time base differs')
            if record['frame_id']!=f"{record['shot']}:{record['pts']}":raise ValueError('Audit frame identity differs')
            if not np.isfinite(record['error_px'])or record['error_px']<=20:raise ValueError('Audit row is not an existing >20px observation')
            source_crop_box([record['uv']])
            records.append(record)
    if len({r['row']for r in records})!=len(records)or len({r['observation_id']for r in records})!=len(records):
        raise ValueError('Audit observation identities must be unique')
    return records


def run(args):
    root=args.review_root.resolve();output=guard_project_write(args.output.resolve())
    if output.exists():raise FileExistsError(output)
    inputs={str(path):digest(path)for path in (root/'observations.jsonl',root/'groups.json',root/'summary.json')}
    observations=normalized_observations(root/'observations.jsonl');audit=read(root/'groups.json')
    if audit.get('decision_ledger')!='observations.jsonl':raise ValueError('Unexpected audit decision ledger')
    if set(row['row']for row in observations)!=set(i for f in audit['frames']for i in f['rows']):
        raise ValueError('Audit exposure ledger does not cover the observation rows exactly')
    grouped=defaultdict(list);frame_errors=defaultdict(float)
    for row in observations:
        grouped[row['shot'],row['pts'],family(row['path_id'])].append(row)
        key=row['shot'],row['pts'];frame_errors[key]=max(frame_errors[key],row['error_px'])
    if len(frame_errors)!=len(audit['frames']):raise ValueError('Audit frame count differs')
    output.mkdir(parents=True);(output/'cases').mkdir();(output/'pages').mkdir()
    plan=[]
    ordered=sorted(grouped,key=lambda k:(tier(frame_errors[k[:2]]),-max(r['error_px']for r in grouped[k]),k))
    for number,key in enumerate(ordered,1):
        rows=sorted(grouped[key],key=lambda r:r['ledger_line'])
        sid,pt,name=key
        plan.append(dict(group_number=number,shot=sid,pts=pt,family=name,priority=tier(frame_errors[sid,pt]),
            rows=rows,max_error_px=max(row['error_px']for row in rows),frame_max_error_px=frame_errors[sid,pt],
            crop_xyxy=source_crop_box([row['uv']for row in rows])))
    group_counts=Counter(p['priority']for p in plan)
    frame_counts=Counter(tier(value)for value in frame_errors.values())
    summary=dict(schema='rinkspace.source_audit_panels.v1',audit_rows=len(observations),
        audit_path_groups=len(audit['groups']),native_exposures=len(frame_errors),family_groups=len(plan),
        family_groups_by_priority=dict(group_counts),frames_by_priority=dict(frame_counts),
        priority_uses_frame_max_error=True,source_pixel_scale=1.,interpretation_or_decisions_performed=False,
        native_projection_drawn=False,observations_modified=False,input_sha256=inputs,
        source_adapter=dict(path=iter_registration_source_frames.__code__.co_filename,
            sha256=digest(iter_registration_source_frames.__code__.co_filename)))
    write(output/'plan.json',summary)
    print(json.dumps(dict(stage='plan',**summary),ensure_ascii=False),flush=True)
    all_cases=[];all_pages=[];frame_records=[]
    for priority in ('01_gt80','02_gt40_to80','03_gt20_to40'):
        selected=[p for p in plan if p['priority']==priority];by_shot=defaultdict(lambda:defaultdict(list))
        for group in selected:by_shot[group['shot']][group['pts']].append(group)
        generated={}
        for sid,by_pts in sorted(by_shot.items()):
            expected=set(by_pts);seen=set()
            for frame in iter_registration_source_frames(sid,bootstrap_root=args.bootstrap,selected_pts=sorted(expected)):
                pt=int(frame['pts'])
                if pt not in expected or pt in seen:raise ValueError('Source adapter returned an unrequested/duplicate frame')
                seen.add(pt)
                source=dict(shot=sid,pts=pt,source_seconds=pt/12800,time_base=[1,12800],
                    source_video_sha256=frame['source_video_sha256'],source_BGR_sha256=frame['source_BGR_sha256'],
                    image_size=[1920,1080],pixel_domain='native_distorted')
                frame_records.append(source)
                for group in by_pts[pt]:
                    cases=[];rows=group['rows']
                    for start in range(0,len(rows),99):
                        chunk=rows[start:start+99]
                        local=[dict(row,local_label=i+1)for i,row in enumerate(chunk)]
                        panel_id=f"g{group['group_number']:04d}"+(f'_b{start//99+1:02d}'if len(rows)>99 else'')
                        filename=f"{panel_id}_{sid}_p{pt}.png"
                        short=group['family'].replace('print_instance/','')
                        heading=f"{panel_id} {sid} p{pt} {pt/12800:.2f}s {short} n{len(rows)} e{group['max_error_px']:.1f}"
                        picture=paint_case(frame['image'],rows,local,group['crop_xyxy'],heading)
                        image_path=output/'cases'/filename;save_png(image_path,picture)
                        def point_record(row):
                            return {key:row[key]for key in ('ledger_line','row','observation_id','path_id','uv','kind','error_px','parent_old_vs_new')}
                        labels=[dict(point_record(row),local_label=row['local_label'])for row in local]
                        sidecar=dict(schema='rinkspace.source_outlier_review_panel.v1',panel_id=panel_id,source=source,
                            priority=priority,family=group['family'],crop_xyxy=group['crop_xyxy'],
                            source_pixel_scale=1.,left_crop_untouched=True,right_crop_is_same_source_plus_existing_audit_UV=True,
                            all_group_outlier_rows=[point_record(row)for row in rows],local_labels=labels,
                            unlabeled_gray_points_are_other_rows_in_same_family=len(local)!=len(rows),
                            image=dict(path=str(image_path),sha256=digest(image_path)),
                            audit_ledger=dict(path=str(root/'observations.jsonl'),sha256=inputs[str(root/'observations.jsonl')]),
                            decisions_performed=False,source_measurements_generated=False,native_projection_drawn=False)
                        sidecar_path=image_path.with_suffix('.json');write(sidecar_path,sidecar)
                        cases.append(dict(panel_id=panel_id,shot=sid,pts=pt,family=group['family'],priority=priority,
                            image=str(image_path.relative_to(output)),sidecar=str(sidecar_path),
                            width=picture.shape[1],height=picture.shape[0],labels=len(local),
                            all_group_rows=len(rows),max_error_px=group['max_error_px']))
                    generated[group['group_number']]=cases
            if seen!=expected:raise ValueError('Requested source exposure is missing')
        cases=[case for group in selected for case in generated[group['group_number']]]
        pages=build_pages(output,priority,cases)if cases else[]
        all_cases.extend(cases);all_pages.extend(pages)
        tier_index=dict(summary,priority=priority,cases=cases,pages=pages,
            priority_complete=True,all_tiers_complete=False)
        write(output/f'index_{priority}.json',tier_index)
        print(json.dumps(dict(stage='priority_complete',priority=priority,frames=frame_counts[priority],
            family_groups=len(selected),panels=len(cases),pages=len(pages),first_pages=[p['path']for p in pages[:3]])),flush=True)
    if len(frame_records)!=len(frame_errors)or sum(case['labels']for case in all_cases)!=len(observations):
        raise ValueError('Source panels do not label every audited observation exactly once')
    for path,expected in inputs.items():
        if digest(path)!=expected:raise ValueError('Input audit changed while drawing source panels')
    final=dict(summary,cases=all_cases,pages=all_pages,panel_count=len(all_cases),page_count=len(all_pages),
        source_frames=frame_records,all_tiers_complete=True)
    write(output/'index.json',final)
    html=['<!doctype html><meta charset="utf-8"><title>Source outlier audit panels</title>',
        '<style>body{font:16px system-ui;margin:20px;background:#eee}img{max-width:100%;height:auto}p{max-width:1000px}</style>',
        '<h1>Source evidence panels</h1><p>Native-scale original crop on the left; identical crop with existing audited source UV on the right. No interpretation or decisions. Open a case PNG for the native pixels; sidecars map each short label to the global audit row and observation ID.</p>']
    for page in all_pages:
        relative=Path(page['path']).relative_to(output).as_posix()
        html.append(f'<h2>{page["priority"]} / {page["page"]}</h2><a href="{relative}"><img loading="lazy" src="{relative}"></a>')
    (output/'index.html').write_text('\n'.join(html),encoding='utf8')
    print(json.dumps(dict(stage='complete',output=str(output),rows=len(observations),frames=len(frame_errors),
        family_groups=len(plan),panels=len(all_cases),pages=len(all_pages))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review-root','bootstrap','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
