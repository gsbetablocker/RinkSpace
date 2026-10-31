"""Source-image checks of anchor identity and adjacent-exposure blade motion."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from .rebuild_prepare import load,digest
from .blade_cotracker_review import overlap_error,longest_run
from .blade_temporal import image_flow,curve_distance
from .footwear_fit import point_polyline_distance


def anchor_checks(output,model_root,destination):
    output,model_root,destination=Path(output),Path(model_root),Path(destination)
    idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));rows={r['pts']:r for r in idx['rows']}
    z=load(model_root/'tracks.npz');config=json.loads((model_root/'config.json').read_text(encoding='utf8'));groups={};by={int(t):i for i,t in enumerate(z['pts'])}
    for j,q in enumerate(config['queries']):
        if q['kind']=='blade_boundary':groups.setdefault(q['seed_index'],dict(foot=q['foot'],pts=q['source_pts'],ids=[]))['ids'].append(j)
    checks=[];directed={}
    for a,ga in groups.items():
        for b,gb in groups.items():
            if ga['pts']==gb['pts']:continue
            i=by[gb['pts']];direction='forward' if ga['pts']<gb['pts'] else 'backward';q=z[direction+'_native_uv'][i,ga['ids']];scores=z[direction+'_visibility_score'][i,ga['ids']];xyxy=rows[gb['pts']]['crop_xyxy'];valid=(scores>.6)&(q[:,0]>xyxy[0]+2)&(q[:,0]<xyxy[2]-2)&(q[:,1]>xyxy[1]+2)&(q[:,1]<xyxy[3]-2);run=longest_run(valid);target=z['queries_native'][gb['ids'],1:];part=q[run] if run is not None else np.empty((0,2));resolved=bool(len(part)>=2 and np.linalg.norm(part[-1]-part[0])>=8)
            overlap=overlap_error(part,target) if resolved else None
            median=float(np.median(point_polyline_distance(part,target))) if resolved else None
            r=dict(source_group=a,target_group=b,source_foot_hypothesis=ga['foot'],target_foot_hypothesis=gb['foot'],source_pts=ga['pts'],target_pts=gb['pts'],source_seconds=ga['pts']/12800,target_seconds=gb['pts']/12800,visible_query_points=int(valid.sum()),model_visibility_median=float(np.median(scores)),visible_fragment_resolved=resolved,overlap_error_px=overlap,median_query_to_source_curve_px=median,self_initialization=False)
            checks.append(r);directed[a,b]=r
    links=[]
    for a,ga in groups.items():
        for b,gb in groups.items():
            if a>=b or ga['pts']==gb['pts']:continue
            x,y=directed[a,b],directed[b,a];ex,ey=x['overlap_error_px'],y['overlap_error_px']
            if ex is not None and ey is not None and max(ex,ey)<=6:
                links.append(dict(groups=[a,b],source_seconds=[ga['pts']/12800,gb['pts']/12800],foot_hypotheses=[ga['foot'],gb['foot']],reciprocal_overlap_error_px=[ex,ey],crosses_existing_foot_hypothesis=bool(ga['foot']!=gb['foot']),physical_identity_confirmed=False))
    record=dict(schema='rinkspace.blade-source-cross-anchor-checks.v1',self_initializations_excluded=True,body_or_camera_projection_used=False,source_identity_automatically_rewritten=False,source_curve_points_are_development_reads=True,model_visibility_threshold=.6,matching_threshold_px=6,threshold_is_development_setting=True,source_groups={str(k):dict(foot_hypothesis=g['foot'],source_seconds=g['pts']/12800) for k,g in groups.items()},reciprocal_source_links=links,directed_checks=checks)
    (destination/'anchor_identity_checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');return record


def motion_checks(root,output,destination):
    root,output,destination=Path(root),Path(output),Path(destination);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));track=json.loads((destination/'tracks.json').read_text(encoding='utf8'));rows=track['rows'];wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');start=time.perf_counter();checks=[];counts={};cv2.setNumThreads(2)
    if [r['pts'] for r in rows]!=[r['pts'] for r in idx['rows']]:raise ValueError('Source exposures differ')
    previous=cv2.imread(idx['rows'][0]['file'])
    with (destination/'motion_check_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for i in range(1,len(rows)):
            current=cv2.imread(idx['rows'][i]['file']);feet=[f for f in [0,1] if rows[i-1]['feet'][f]['curve_uv'] is not None];curves=[dict(curve=np.asarray(rows[i-1]['feet'][f]['curve_uv'],float)) for f in feet]
            motion=image_flow(previous,current,idx['rows'][i-1],idx['rows'][i],curves,wb['kp2d_coco23'][i-1,[15,16],:2]) if feet else [];previous=current
            for f,m in zip(feet,motion):
                target=rows[i]['feet'][f];curve=target['curve_uv'];error=overlap_error(m['prediction'],curve) if curve is not None else None
                if curve is None:state='target_not_located'
                elif not m['image_motion_supported']:state='adjacent_image_motion_unresolved'
                elif error is None:state='visible_fragments_do_not_overlap'
                elif error>5:state='adjacent_image_motion_disagrees'
                else:state='adjacent_image_motion_agrees'
                counts[state]=counts.get(state,0)+1
                checks.append(dict(previous_pts=rows[i-1]['pts'],pts=rows[i]['pts'],source_seconds=rows[i]['source_seconds'],track='AB'[f],state=state,current_status=target['status'],image_motion_supported=m['image_motion_supported'],overlap_error_px=error,full_curve_distance_px=curve_distance(m['prediction'],curve) if curve is not None else None,valid_material_points=m['valid_material_points'],forward_backward_pixel_error=m['fb_median']))
            if i%100==0:log.write(json.dumps(dict(frames=i+1,seconds=time.perf_counter()-start))+'\n')
        record=dict(schema='rinkspace.blade-adjacent-source-motion-checks.v1',source_exposures=len(rows),counts=counts,checks=checks,curves_or_source_points_changed=False,source_ice_motion_allowed=True,body_or_camera_projection_used=False,threshold_px=5,threshold_is_development_setting=True,source_index_sha256=digest(output/'source_index.json'),tracks_sha256=digest(destination/'tracks.json'))
        (destination/'adjacent_motion_checks.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf8');log.write(json.dumps(dict(event='complete',counts=counts,seconds=time.perf_counter()-start))+'\n')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--model-root',required=True);p.add_argument('--destination',required=True);a=p.parse_args();ac=anchor_checks(a.output,a.model_root,a.destination);mc=motion_checks(a.root,a.output,a.destination);print(json.dumps(dict(reciprocal_source_links=ac['reciprocal_source_links'],motion_counts=mc['counts'])),flush=True)
