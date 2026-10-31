"""Extract original-source review stills and draw exported rows only.

No image judgment, qualification, replacement decision, packet or fit occurs.
source.png and overlay.png retain native dimensions; review.png places the
separate black legend above the complete overlay without covering source pixels.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from capture_independent_ice import capture_frames
from propagate_ice_readout_identities import read,digest,write


SELECTED={
    's07':[1465600,1499392,1521152,1590784,1609472],
    's08':[1746944,1779712,1801728],
    's09':[1931264,1943296,1964800,1970944,1990912,2001664,2408704],
}
ROLE_RGB={'inner':[70,255,70],'middle':[255,215,0],'outer':[0,235,255],
          'center':[255,70,255],'centreline':[255,55,55]}
LINE_RGB=[[85,135,255],[255,140,35],[170,95,255],[255,100,175],[20,195,160],
          [180,150,85],[135,210,255],[255,170,195],[150,135,245],[175,230,80],
          [255,105,90],[100,185,125],[200,130,210],[225,225,120]]


def bgr_sha(image):return hashlib.sha256(image.tobytes()).hexdigest()


def save_image(path,image):
    path=guard_project_write(path)
    if not cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise OSError(path)
    return dict(path=str(path),sha256=digest(path),width=int(image.shape[1]),height=int(image.shape[0]),BGR_sha256=bgr_sha(image))


def span(uv):
    uv=np.asarray(uv,float).reshape(-1,2)
    if not len(uv):return dict(rows=0,bbox_uv=None,span_uv=[0.,0.],span_norm_px=0.)
    bounds=[uv.min(0).tolist(),uv.max(0).tolist()];extent=np.ptp(uv,axis=0)
    return dict(rows=len(uv),bbox_uv=bounds,span_uv=extent.tolist(),span_norm_px=float(np.linalg.norm(extent)))


def run(args):
    batch=args.batch;output=guard_project_write(batch/'review');output.mkdir(exist_ok=True)
    index=read(args.root/'index.json');metadata=read(batch/'metadata.json')
    binding=read(args.legacy/'bulk_ice_source/full_direct/binding.json');inventory={p['native_path_id']:p for p in binding['native_inventory']}
    line_entities=sorted({p['curve_id']for p in inventory.values()if not p['closed'] and
        any(s['paths'][p['native_path_id']]['output_uv']for s in metadata['shots'].values())})
    if len(line_entities)>len(LINE_RGB):raise ValueError('Explicit line color palette is too small')
    line_colors=dict(zip(line_entities,LINE_RGB));new_uv=defaultdict(list);selected_records={};rowfiles={}
    for sid,pts in SELECTED.items():
        file=batch/sid/'source_rows.jsonl'
        if digest(file)!=metadata['shots'][sid]['source_rows']['sha256']:raise ValueError('Exported source rows changed')
        rowfiles[sid]=dict(path=str(file),sha256=digest(file))
        with file.open(encoding='utf8')as stream:
            for line in stream:
                record=json.loads(line);pt=record['pts']
                for row in record['rows']:new_uv[sid,pt,row['path_id']].append(row['uv'])
                if pt in pts:selected_records[sid,pt]=record
        if any((sid,pt)not in selected_records for pt in pts):raise ValueError('Selected source exposure missing')
    if digest(index['source']['resolved_path'])!=index['source']['actual_sha256']:raise ValueError('Original source video changed')
    manifest=[]
    for sid,pts in SELECTED.items():
        shot=next(s for s in index['shots']if s['shot_id']==sid)
        for fi,(pt,image)in capture_frames(index,shot,pts):
            record=selected_records[sid,pt]
            if image.shape!=(1080,1920,3)or bgr_sha(image)!=record['source_BGR_sha256']:raise ValueError('Actual source BGR pixels differ')
            folder=output/sid/str(pt);folder.mkdir(parents=True,exist_ok=True)
            source=save_image(folder/'source.png',image)
            if bgr_sha(cv2.imread(source['path'],cv2.IMREAD_COLOR))!=record['source_BGR_sha256']:raise ValueError('Lossless PNG source roundtrip differs')
            overlay=image.copy();counts=Counter();pending=Counter();colors={}
            for row in record['rows']:
                path=inventory[row['path_id']]
                if path['closed']:rgb=ROLE_RGB[row['relative_role']]
                else:rgb=line_colors[row['entity']]
                colors[row['path_id']]=rgb;counts[row['path_id']]+=1;pending[row['path_id']]+=bool(row['material_pending'])
                center=tuple(np.rint(np.asarray(row['uv'])).astype(int));cv2.circle(overlay,center,2,tuple(reversed(rgb)),-1,cv2.LINE_8)
            path_entries=[dict(path_id=pid,entity=inventory[pid]['curve_id'],relative_role=inventory[pid]['world_curve_role'],
                points=counts[pid],material_pending_points=pending[pid],color_RGB=colors[pid],
                color_hex='#'+''.join(f'{v:02X}'for v in colors[pid]))for pid in sorted(counts,key=lambda p:inventory[p]['curve_id'])]
            rows=max(1,math.ceil(len(path_entries)/3));legend=np.zeros((52+rows*30,1920,3),np.uint8)
            header=f'{sid} | PTS {pt} | source {pt/12800:.3f}s | {len(record["rows"])} exported points | pending {sum(pending.values())} | r=2 | NO FIT'
            cv2.putText(legend,header,(16,28),cv2.FONT_HERSHEY_SIMPLEX,.64,(245,245,245),1,cv2.LINE_AA)
            for i,p in enumerate(path_entries):
                col=i%3;row=i//3;x=col*640+16;y=55+row*30;rgb=p['color_RGB']
                cv2.rectangle(legend,(x,y-12),(x+17,y+3),tuple(reversed(rgb)),-1)
                label=f'{p["entity"]}: {p["points"]} pts  {p["color_hex"]}'
                cv2.putText(legend,label,(x+27,y),cv2.FONT_HERSHEY_SIMPLEX,.51,(235,235,235),1,cv2.LINE_AA)
            overlay_file=save_image(folder/'overlay.png',overlay);legend_file=save_image(folder/'legend.png',legend)
            review_file=save_image(folder/'review.png',np.vstack((legend,overlay)))
            manifest.append(dict(shot=sid,pts=pt,source_seconds=pt/12800,time_base='1/12800',
                source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=record['source_BGR_sha256'],
                exported_points=len(record['rows']),material_pending_points=sum(pending.values()),paths=path_entries,
                source=source,overlay=overlay_file,legend=legend_file,review=review_file,review_source_y_offset=int(legend.shape[0]),
                source_rows=rowfiles[sid],only_exported_rows_drawn=True,point_radius_px=2,point_rounding='numpy.rint original UV',
                lines_drawn=False,image_judgment_performed=False,fit_eligible=False))
            print(json.dumps(dict(shot=sid,pts=pt,points=len(record['rows']),folder=str(folder))),flush=True)
    write(output/'manifest.json',dict(schema='rinkspace.registration_batch_original_source_review.v1',frames=manifest,
        source_and_overlay_native_hw=[1080,1920],legend_outside_original_image=True,source_BGR_roundtrip_exact=True,
        role_RGB=ROLE_RGB,line_entity_RGB=line_colors,image_judgment_performed=False,fit_eligible=False))
    # Compare only the existing upsert-plan replacement groups, using raw bbox spans.
    plan=[json.loads(line)for line in(batch/'group_update_plan.jsonl').read_text('utf8').splitlines()]
    replacements={(p['shot'],int(p['pts']),p['path_id']):p for p in plan if p['proposed_operation']=='replace_existing_group'}
    old_input=metadata['inputs']['current_prepared_points'];mapping_file=metadata['inputs']['current_mapping']
    if digest(old_input['path'])!=old_input['sha256']or digest(mapping_file['path'])!=mapping_file['sha256']:raise ValueError('Current prepared comparison source changed')
    mapping=read(mapping_file['path']);old_uv=defaultdict(list)
    with np.load(old_input['path'],allow_pickle=False)as z:original_pts=z['pts'];original_path=z['path_indices'];original_uv=z['uv']
    source_windows={s['shot_id']:s['processing']for s in index['shots']}
    for sid,window in source_windows.items():
        for i in np.flatnonzero((original_pts>=window['first_pts'])&(original_pts<=window['last_pts'])):
            key=(sid,int(original_pts[i]),mapping['path_ids'][int(original_path[i])])
            if key in replacements:old_uv[key].append(original_uv[i])
    comparisons=[]
    for key,group in sorted(replacements.items()):
        before=span(old_uv[key]);after=span(new_uv[key]);denominator=before['span_norm_px']
        comparisons.append(dict(shot=key[0],pts=key[1],source_seconds=key[1]/12800,path_id=key[2],entity=inventory[key[2]]['curve_id'],
            old=before,new=after,span_norm_ratio=after['span_norm_px']/denominator if denominator else None,
            span_norm_reduction_px=denominator-after['span_norm_px'],selection_or_replacement_decision_made=False))
    by_entity={}
    for sid,entity in sorted({(p['shot'],p['entity'])for p in comparisons}):
        values=[p for p in comparisons if p['shot']==sid and p['entity']==entity];ratios=[p['span_norm_ratio']for p in values if p['span_norm_ratio']is not None]
        by_entity[sid+'/'+entity]=dict(groups=len(values),ratio_p0_p25_p50_p75_p100=np.percentile(ratios,[0,25,50,75,100]).tolist()if ratios else None,
            ratio_below_half_diagnostic_only=sum(v<.5 for v in ratios),ratio_below_three_quarters_diagnostic_only=sum(v<.75 for v in ratios))
    ratios=[p['span_norm_ratio']for p in comparisons if p['span_norm_ratio']is not None]
    span_report=dict(schema='rinkspace.registration_replacement_UV_span_review.v1',replacement_groups=len(comparisons),
        definition='span_uv = max(UV)-min(UV); span_norm_px = Euclidean norm of that vector; no residual or qualification decision',
        ratio_p0_p25_p50_p75_p100=np.percentile(ratios,[0,25,50,75,100]).tolist()if ratios else None,
        ratio_below_half_diagnostic_only=sum(v<.5 for v in ratios),ratio_below_three_quarters_diagnostic_only=sum(v<.75 for v in ratios),
        by_shot_entity=by_entity,largest_absolute_span_reductions=sorted(comparisons,key=lambda p:p['span_norm_reduction_px'],reverse=True)[:30],
        groups=comparisons,original_prepared=old_input,exported_source_rows=rowfiles,
        diagnostic_bins_do_not_control_retention_or_replacement=True,rows_changed=False,prepared_modified=False,new_fit_started=False)
    write(output/'replace_span_report.json',span_report)
    print(json.dumps(dict(complete=True,frames=len(manifest),replace_groups=len(comparisons),ratio_below_half=span_report['ratio_below_half_diagnostic_only'],span_ratio_percentiles=span_report['ratio_p0_p25_p50_p75_p100'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy','root','batch'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
