"""Short source-anchored image tracks; no camera, body or contact feedback."""
from pathlib import Path
from functools import lru_cache
import argparse,json,time,itertools
import cv2,numpy as np
from .rebuild_prepare import load,digest
from .blade_temporal import image_flow,resample_curve,edge_snap,visible_segment
from .blade_cotracker_review import overlap_error
from .footwear_fit import point_polyline_distance


def same_visible_blade(a,b,navigation_centres=None):
    a,b=np.asarray(a,float),np.asarray(b,float)
    error=overlap_error(a,b)
    if error is not None and error<3:return True
    ab=np.r_[a,b];_,_,v=np.linalg.svd(ab-ab.mean(0),full_matrices=False);axis=v[0];normal=v[1];pa,pb=a@axis,b@axis;gap=max(pa.min(),pb.min())-min(pa.max(),pb.max());normal_separation=abs(np.median(a@normal)-np.median(b@normal));da,db=a[-1]-a[0],b[-1]-b[0];parallel=abs(da@db)/(max(np.linalg.norm(da)*np.linalg.norm(db),1e-8))
    if normal_separation<3 and gap<12 and parallel>.8:return True
    if navigation_centres is not None:
        nav=np.asarray(navigation_centres);d=np.linalg.norm(np.stack([a.mean(0),b.mean(0)])[:,None]-nav[None],axis=-1);which=np.argmin(d,axis=1)
        if which[0]==which[1] and np.linalg.norm(nav[0]-nav[1])>35 and np.all(np.max(d,axis=1)-np.min(d,axis=1)>18):return True
    return False


def run(root,output,source_path,destination,max_age_frames=20):
    root,output,dest=Path(root),Path(output),Path(destination);dest.mkdir(exist_ok=False);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));source=json.loads(Path(source_path).read_text(encoding='utf8'));wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');pts=np.asarray([r['pts'] for r in idx['rows']]);by={int(p):i for i,p in enumerate(pts)};collected=[[[],[]] for _ in pts];start=time.perf_counter();cv2.setNumThreads(2)
    @lru_cache(maxsize=128)
    def image(i):return cv2.imread(idx['rows'][i]['file'])
    (dest/'source_anchors_at_start.json').write_bytes(Path(source_path).read_bytes())
    seeds=source['blade_curves'];seed_records=[]
    with (dest/'tracking_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for si,seed in enumerate(seeds):
            initial=by[seed['pts']];curve=resample_curve(seed['uv']);foot=seed['foot'];xyxy=idx['rows'][initial]['crop_xyxy'];inside=(curve[:,0]>=xyxy[0])&(curve[:,0]<xyxy[2])&(curve[:,1]>=xyxy[1])&(curve[:,1]<xyxy[3])
            if not inside.all():raise ValueError(f'Source curve outside original crop: {seed["pts"]}, {foot}')
            collected[initial][foot].append(dict(curve=np.asarray(seed['uv'],float),age=0,source_pts=seed['pts'],seed_id=si,direction=0,native_edge_contrast=None,material_motion_fb_px=None,visible_edge_fraction=1.))
            for direction in [-1,1]:
                current_curve=curve.copy();previous=initial;accepted=0;stop_reason='short_window_limit'
                for age in range(1,max_age_frames+1):
                    i=initial+direction*age
                    if i<0 or i>=len(pts):stop_reason='shot_end';break
                    m=image_flow(image(previous),image(i),idx['rows'][previous],idx['rows'][i],[dict(curve=current_curve)],wb['kp2d_coco23'][previous,[15,16],:2])[0]
                    if not m['image_motion_supported']:stop_reason='source_material_motion_unresolved';break
                    predicted=m['prediction'];box=idx['rows'][i]['crop_xyxy'];inside=(predicted[:,0]>=box[0]+2)&(predicted[:,0]<box[2]-2)&(predicted[:,1]>=box[1]+2)&(predicted[:,1]<box[3]-2)
                    refined,support,contrast=edge_snap(image(i),idx['rows'][i],predicted,1);support&=inside;visible=visible_segment(refined,support)
                    if visible is not None:
                        collected[i][foot].append(dict(curve=visible,age=age,source_pts=seed['pts'],seed_id=si,direction=direction,native_edge_contrast=float(np.median(contrast[support])),material_motion_fb_px=m['fb_median'],visible_edge_fraction=float(support.mean())));accepted+=1
                    current_curve=refined if visible is not None else predicted;previous=i
                seed_records.append(dict(seed=si,pts=seed['pts'],foot_hypothesis=foot,direction=direction,visible_source_exposures=accepted,last_source_pts=int(pts[previous]),stop_reason=stop_reason))
            if (si+1)%5==0:log.write(json.dumps(dict(event='source_window',seeds=si+1,total=len(seeds),seconds=time.perf_counter()-start))+'\n')
        rows=[];conflicts=[]
        for i,pair in enumerate(collected):
            pools=[]
            for foot,cs in enumerate(pair):
                cs.sort(key=lambda q:q['age'])
                for c in cs:
                    other=[q for q in cs if q['direction']*c['direction']<0 and q['source_pts']!=c['source_pts']];agreement=overlap_error(c['curve'],other[0]['curve']) if other else None;c['agreement']=agreement;c['has_opposite_source']=bool(other);c['status']='source_annotation' if c['age']==0 else 'tracked_candidate' if agreement is not None and agreement<=5 else 'image_candidate_unconfirmed';c['cost']=c['age']/10+(0 if c['status'] in ['source_annotation','tracked_candidate'] else 2)+(0 if not other or agreement is None else min(agreement/5,8))
                pools.append(cs[:6])
            nav=np.stack([wb['kp2d_coco23'][i,[17,18,19],:2].mean(0),wb['kp2d_coco23'][i,[20,21,22],:2].mean(0)]);best=None;best_cost=float('inf');rejected=0
            for a,b in itertools.product(pools[0]+[None],pools[1]+[None]):
                if a is not None and b is not None and same_visible_blade(a['curve'],b['curve'],nav):rejected+=1;continue
                cost=sum(8 if q is None else q['cost'] for q in [a,b])
                if cost<best_cost:best=[a,b];best_cost=cost
            feet=[]
            for foot,c in enumerate(best):
                if c is None:feet.append(dict(status='not_located',curve_uv=None,reason='candidate identity conflict' if pools[foot] else 'no source-supported short-window blade fragment',physical_foot_identity='unconfirmed'));continue
                reason='explicit source initialization, not an independent check' if c['status']=='source_annotation' else 'different source times agree on overlapping visible boundary' if c['status']=='tracked_candidate' else 'one source side only or other-side disagreement'
                feet.append(dict(status=c['status'],curve_uv=c['curve'].tolist(),reason=reason,source_anchor_pts=c['source_pts'],source_anchor_age_frames=c['age'],source_group=c['seed_id'],past_future_agreement_px=c['agreement'],native_edge_contrast=c['native_edge_contrast'],native_edge_support=c['visible_edge_fraction'],material_motion_fb_px=c['material_motion_fb_px'],physical_foot_identity='unconfirmed',available_source_groups=len(pair[foot])))
            if rejected:conflicts.append(dict(pts=int(pts[i]),source_seconds=float(pts[i]/12800),rejected_same_blade_assignments=rejected))
            rows.append(dict(pts=int(pts[i]),source_seconds=float(pts[i]/12800),feet=feet,human_reviewed=False,contact_state='unknown',used_for_body_or_camera=False))
        states=sorted({q['status'] for r in rows for q in r['feet']});counts={s:[sum(r['feet'][f]['status']==s for r in rows) for f in [0,1]] for s in states};record=dict(schema='rinkspace.short-source-window-blade-candidates.v1',source_time_base=[1,12800],source_exposures=len(rows),source_anchor_curves=len(seeds),source_anchor_times=len(set(q['pts'] for q in seeds)),source_anchor_file=str((dest/'source_anchors_at_start.json').resolve()),source_anchor_origin_file=str(Path(source_path).resolve()),source_anchor_sha256=digest(source_path),source_index_sha256=digest(output/'source_index.json'),wholebody_sha256=digest(root/'next_round_20261029/wholebody/wholebody_coco23.npz'),counts=counts,maximum_source_distance_frames=max_age_frames,source_qualified_pixels_unchanged=True,source_previous_lower_body_flag_used_as_blade_visibility=False,contact_feedback=False,camera_or_body_projection_used=False,physical_foot_identity_confirmed=False,human_reviewed=False,accuracy_accepted=False,seed_windows=seed_records,same_blade_assignment_conflicts=conflicts,rows=rows)
        (dest/'tracks.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf8');log.write(json.dumps(dict(event='complete',frames=len(rows),counts=counts,seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=len(rows),counts=counts,identity_conflict_exposures=len(conflicts),seconds=time.perf_counter()-start)),flush=True);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--max-age-frames',type=int,default=20);a=p.parse_args();run(a.root,a.output,a.source,a.destination,a.max_age_frames)
