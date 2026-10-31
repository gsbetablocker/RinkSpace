"""Turn video point tracks into source-supported visible blade fragments."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from .rebuild_prepare import load,digest
from .blade_temporal import resample_curve,edge_snap,visible_segment
from .footwear_fit import point_polyline_distance


def source_to_model(xy,transform):
    xy=np.asarray(xy,float);tr=np.asarray(transform,float)
    while tr.ndim<xy.ndim:tr=np.expand_dims(tr,-2)
    return (xy-tr[...,:2]+.5)*tr[...,2:4]-.5+tr[...,4:6]


def model_to_source(xy,transform):
    xy=np.asarray(xy,float);tr=np.asarray(transform,float)
    while tr.ndim<xy.ndim:tr=np.expand_dims(tr,-2)
    return (xy+.5-tr[...,4:6])/tr[...,2:4]-.5+tr[...,:2]


def overlap_error(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);combined=np.r_[a,b];_,_,v=np.linalg.svd(combined-combined.mean(0),full_matrices=False);axis=v[0];pa=a@axis;pb=b@axis;lo=max(pa.min(),pb.min());hi=min(pa.max(),pb.max())
    if hi-lo<4:return None
    aa=a[(pa>=lo-1e-6)&(pa<=hi+1e-6)];bb=b[(pb>=lo-1e-6)&(pb<=hi+1e-6)]
    if not len(aa) or not len(bb):return None
    return float(np.median(np.r_[point_polyline_distance(aa,b),point_polyline_distance(bb,a)]))


def longest_run(mask):
    mask=np.asarray(mask,bool);starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask&~np.r_[mask[1:],False])+1
    if not len(starts):return None
    k=int(np.argmax(ends-starts));return slice(int(starts[k]),int(ends[k]))


def build(output,model_root,destination):
    output,model_root,destination=Path(output),Path(model_root),Path(destination);destination.mkdir(exist_ok=False);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));source_by={r['pts']:r for r in idx['rows']};config=json.loads((model_root/'config.json').read_text(encoding='utf8'));z=load(model_root/'tracks.npz');meta=config['queries'];groups={}
    for j,q in enumerate(meta):groups.setdefault(q['seed_index'],dict(foot=q['foot'],source_pts=q['source_pts'],blade=[],boot=[]))['blade' if q['kind']=='blade_boundary' else 'boot'].append(j)
    raw=json.loads((output/'candidates.json').read_text(encoding='utf8'));raw_by={r['pts']:r for r in raw['records']};rows=[];start=time.perf_counter();cv2.setNumThreads(2)
    with (destination/'postprocess_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for i,pts in enumerate(z['pts']):
            row=source_by[int(pts)];image=cv2.imread(row['file']);feet=[]
            for foot in [0,1]:
                candidates=[];boot_seen=False;image_guides=[]
                for group_id,g in groups.items():
                    if g['foot']!=foot:continue
                    direction='forward' if pts>=g['source_pts'] else 'backward';points=z[direction+'_native_uv'][i,g['blade']];scores=z[direction+'_visibility_score'][i,g['blade']];inside=(points[:,0]>=row['crop_xyxy'][0]+2)&(points[:,0]<row['crop_xyxy'][2]-2)&(points[:,1]>=row['crop_xyxy'][1]+2)&(points[:,1]<row['crop_xyxy'][3]-2);supported=(scores>.6)&inside
                    if g['boot']:
                        bs=z[direction+'_visibility_score'][i,g['boot']];valid_support=bs>.6;boot_seen|=bool(np.sum(valid_support)>=3)
                        if valid_support.sum()>=3:
                            qi=np.asarray(g['boot'])[valid_support];original=z['queries_native'][qi,1:];current=z[direction+'_native_uv'][i,qi];matrix,inliers=cv2.estimateAffinePartial2D(original.astype(np.float32),current.astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=3.)
                            if matrix is not None and int(inliers.sum())>=3:
                                singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
                                if singular.min()>.25 and singular.max()<4:
                                    source_blade=z['queries_native'][g['blade'],1:];guide=np.c_[source_blade,np.ones(len(source_blade))]@matrix.T;image_guides.append(guide)
                    if pts==g['source_pts']:supported=inside.copy()
                    run=longest_run(supported)
                    if run is None or run.stop-run.start<2:continue
                    part=points[run]
                    if np.linalg.norm(part[-1]-part[0])<8:continue
                    curve=resample_curve(part);refined,support,contrast=edge_snap(image,row,curve,1);segment=visible_segment(refined,support)
                    if segment is None:continue
                    candidates.append(dict(curve=segment,group_id=group_id,source_pts=g['source_pts'],age=abs(int(pts)-g['source_pts'])//256,direction=direction,model_visible_fraction=float(supported.mean()),model_visibility_score=float(np.median(scores[supported])),native_edge_support=float(support.mean()),native_edge_contrast=float(np.median(contrast[support])),is_source_annotation=bool(pts==g['source_pts'])))
                if not candidates and image_guides:
                    best_edge=None;best_distance=float('inf')
                    for edge in [q for pair in raw_by[int(pts)]['candidates'] for q in pair]:
                        curve=np.asarray(edge['curve']);distances=[float(np.median(point_polyline_distance(curve,guide))) for guide in image_guides];distance=min(distances)
                        if distance<best_distance and distance<=6 and edge['contrast']>=8 and edge.get('exterior_variation',0)<14:best_edge=edge;best_distance=distance
                    if best_edge is not None:feet.append(dict(status='image_candidate_unconfirmed',curve_uv=best_edge['curve'],reason='source image edge redetected near tracked equipment texture; blade-point identity unconfirmed',native_edge_contrast=best_edge['contrast'],guide_distance_px=best_distance,support_material_tracked=boot_seen,physical_foot_identity='unconfirmed'));continue
                if not candidates:feet.append(dict(status='not_located',curve_uv=None,reason='nearby texture tracked but blade fragment unresolved' if boot_seen else 'no supported source blade points',support_material_tracked=boot_seen,physical_foot_identity='unconfirmed'));continue
                candidates.sort(key=lambda q:(q['age'],-q['native_edge_support']));best=candidates[0];past=[q for q in candidates if q['source_pts']<pts];future=[q for q in candidates if q['source_pts']>pts];agreement=overlap_error(past[0]['curve'],future[0]['curve']) if past and future else None;status='image_candidate_unconfirmed';reason='visible source edge from one temporal side only'
                if best['is_source_annotation']:status='source_annotation';reason='source initialization; not an independent accuracy check'
                elif past and future:
                    if agreement is not None and agreement<=5:status='tracked_candidate';reason='past and future source fragments agree on overlapping visible boundary'
                    elif agreement is None:reason='past/future visible fragments do not overlap enough to check'
                    else:reason='past and future source fragments disagree'
                feet.append(dict(status=status,curve_uv=best['curve'].tolist(),reason=reason,source_anchor_pts=best['source_pts'],source_anchor_age_frames=best['age'],native_edge_contrast=best['native_edge_contrast'],native_edge_support=best['native_edge_support'],model_visibility_score=best['model_visibility_score'],past_future_agreement_px=agreement,available_source_groups=len(candidates),physical_foot_identity='unconfirmed',support_material_tracked=boot_seen))
            if all(q['curve_uv'] is not None for q in feet):
                error=overlap_error(feet[0]['curve_uv'],feet[1]['curve_uv'])
                if error is not None and error<3:
                    for q in feet:q.update(status='image_candidate_unconfirmed',reason='two track identities converge onto the same visible blade')
            rows.append(dict(pts=int(pts),source_seconds=int(pts)/12800,feet=feet,human_reviewed=False,contact_state='unknown',used_for_body_or_camera=False))
            if len(rows)%100==0:log.write(json.dumps(dict(event='native_edge_check',frames=len(rows),seconds=time.perf_counter()-start))+'\n')
        states=sorted({q['status'] for r in rows for q in r['feet']});counts={s:[sum(r['feet'][f]['status']==s for r in rows) for f in [0,1]] for s in states};record=dict(schema='rinkspace.cotracker-native-blade-fragments.v1',source_exposures=len(rows),source_time_base=[1,12800],source_first_last_pts=[rows[0]['pts'],rows[-1]['pts']],model_report_sha256=digest(model_root/'report.json'),model_config_sha256=digest(model_root/'config.json'),source_index_sha256=digest(output/'source_index.json'),counts=counts,model_visibility_threshold=.6,model_visibility_is_not_calibrated_accuracy=True,native_edge_gate_gray=5,maximum_past_future_overlap_error_px=5,thresholds_are_engineering_development_settings=True,physical_left_right_human_confirmed=False,camera_or_body_projection_used=False,contact_feedback=False,accuracy_accepted=False,rows=rows);(destination/'tracks.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf8');log.write(json.dumps(dict(event='complete',frames=len(rows),counts=counts,seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=len(rows),counts=counts,destination=str(destination))),flush=True);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model-root',required=True);p.add_argument('--destination',required=True);a=p.parse_args();build(a.output,a.model_root,a.destination)
