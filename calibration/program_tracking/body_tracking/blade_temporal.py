"""Image-flow association of source blade curves across every native exposure."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from .rebuild_prepare import load
from .footwear_fit import point_polyline_distance


def curve_distance(a,b):
    return float((np.median(point_polyline_distance(a,b))+np.median(point_polyline_distance(b,a)))/2)


def pool_pair(pair):
    result=[]
    for foot in [0,1]:
        for item in pair[foot]:
            item=dict(item,curve=np.asarray(item['curve'],float),navigation_labels=[foot]);match=next((r for r in result if curve_distance(item['curve'],r['curve'])<2.5),None)
            if match is not None:
                match['navigation_labels']=sorted(set(match['navigation_labels']+[foot]));continue
            result.append(item)
    return result


def image_flow(previous,current,previous_row,current_row,curves,ankles):
    """One-step flow and real reverse-image checks, not a DP self-check."""
    if not curves:return []
    lo=np.minimum(previous_row['crop_xyxy'][:2],current_row['crop_xyxy'][:2]);hi=np.maximum(previous_row['crop_xyxy'][2:],current_row['crop_xyxy'][2:]);shape=tuple((hi-lo)[::-1]);canvases=[]
    for im,row in [(previous,previous_row),(current,current_row)]:
        canvas=np.full(shape,128,np.uint8);xy=np.asarray(row['crop_xyxy'][:2])-lo;canvas[xy[1]:xy[1]+im.shape[0],xy[0]:xy[0]+im.shape[1]]=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);canvases.append(canvas)
    batches=[];ranges=[];start=0
    for item in curves:
        q=item['curve']-lo;centre=item['curve'].mean(0);ankle=ankles[np.argmin(np.linalg.norm(ankles-centre,axis=1))]-lo;mask=np.zeros(shape,np.uint8);poly=cv2.convexHull(np.r_[q,ankle[None]].astype(np.float32)).astype(int);cv2.fillConvexPoly(mask,poly,255);mask=cv2.dilate(mask,np.ones((17,17),np.uint8));features=cv2.goodFeaturesToTrack(canvases[0],maxCorners=24,qualityLevel=.004,minDistance=3,mask=mask);features=np.empty((0,2)) if features is None else features[:,0];points=np.r_[q,features].astype(np.float32);batches.append(points);ranges.append((start,start+len(points),len(q)));start+=len(points)
    points=np.concatenate(batches)[:,None];options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4);forward,status,photo=cv2.calcOpticalFlowPyrLK(canvases[0],canvases[1],points,None,**options);back,sback,_=cv2.calcOpticalFlowPyrLK(canvases[1],canvases[0],forward,None,**options);fb=np.linalg.norm(back[:,0]-points[:,0],axis=1);valid=status[:,0].astype(bool)&sback[:,0].astype(bool)&(fb<1.5)&(photo[:,0]<25);result=[]
    for k,(a,b,ncurve) in enumerate(ranges):
        keep=valid[a:b];src=points[a:b,0];dst=forward[a:b,0];good=np.flatnonzero(keep);affine=None;inliers=None
        if len(good)>=3:
            affine,inliers=cv2.estimateAffinePartial2D(src[good],dst[good],method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99)
            corners=good[good>=ncurve]
            if len(corners)>=6 and affine is not None:
                full,full_inliers=cv2.estimateAffine2D(src[corners],dst[corners],method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99)
                if full is not None:
                    sv=np.linalg.svd(full[:,:2],compute_uv=False);aerr=np.median(np.linalg.norm(np.c_[src[corners],np.ones(len(corners))]@affine.T-dst[corners],axis=1));ferr=np.median(np.linalg.norm(np.c_[src[corners],np.ones(len(corners))]@full.T-dst[corners],axis=1))
                    if sv.min()>.5 and sv.max()<1.7 and np.linalg.det(full[:,:2])>.2 and ferr<.7*aerr and int(full_inliers.sum())>=5:affine,inliers=full,full_inliers
        trusted=False
        if affine is not None:
            singular=np.linalg.svd(affine[:,:2],compute_uv=False);trusted=bool(singular.min()>.5 and singular.max()<1.7 and np.linalg.det(affine[:,:2])>.2 and int(inliers.sum())>=4);prediction=np.c_[batches[k][:ncurve],np.ones(ncurve)]@affine.T+lo
        elif len(good):prediction=curves[k]['curve']+np.median(dst[good]-src[good],axis=0)
        else:prediction=curves[k]['curve'].copy()
        result.append(dict(prediction=prediction,image_motion_supported=bool(trusted),fb_median=float(np.median(fb[a:b][keep])) if keep.any() else None,valid_material_points=int(keep.sum()),affine_inliers=int(inliers.sum()) if inliers is not None else 0))
    return result


def resample_curve(curve,count=17):
    curve=np.asarray(curve,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];q=np.linspace(0,s[-1],count);return np.stack([np.interp(q,s,curve[:,j]) for j in [0,1]],1)


def edge_snap(image,row,curve,sign):
    from .blade_tracking import sample_image
    xy=np.asarray(curve)-row['crop_xyxy'][:2];tangent=np.gradient(xy,axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]];normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-8)[:,None];normal*=sign;gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.5);offsets=np.arange(-3,4,dtype=float);q=xy[:,None]+normal[:,None]*offsets[None,:,None];contrast=sample_image(gray,q+normal[:,None]*1.5)-sample_image(gray,q-normal[:,None]*1.5);score=abs(contrast)-.7*offsets[None]**2;k=score.argmax(1);values=abs(contrast[np.arange(len(xy)),k]);refined=q[np.arange(len(xy)),k]+row['crop_xyxy'][:2];support=values>=5
    refined[~support]=np.asarray(curve)[~support]
    return refined,support,values


def visible_segment(curve,support):
    starts=np.flatnonzero(support&~np.r_[False,support[:-1]]);ends=np.flatnonzero(support&~np.r_[support[1:],False])+1
    if not len(starts):return None
    k=int(np.argmax(ends-starts));part=np.asarray(curve)[starts[k]:ends[k]]
    return part if len(part)>=3 and np.linalg.norm(part[-1]-part[0])>=8 else None


def run(root,output,review_path):
    root,output=Path(root),Path(output);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));source=json.loads(Path(review_path).read_text(encoding='utf8'));wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');pts=np.asarray([r['pts'] for r in idx['rows']]);by={int(t):i for i,t in enumerate(pts)};seeds=[dict(r,seed_id=f'source_{r["pts"]}_{r["foot"]}') for r in source['blade_curves']];collected=[[[],[]] for _ in pts];start=time.perf_counter();cv2.setNumThreads(2)
    with (output/'source_track_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for seed in seeds:
            frame=by[seed['pts']];curve=resample_curve(seed['uv']);seed_image=cv2.imread(idx['rows'][frame]['file']);tangent=np.gradient(curve,axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]];normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-9)[:,None];from .blade_tracking import sample_image
            gray=cv2.cvtColor(seed_image,cv2.COLOR_BGR2GRAY).astype(np.float32);local=curve-idx['rows'][frame]['crop_xyxy'][:2];sign=1 if np.median(sample_image(gray,(local+normal*2)[:,None])-sample_image(gray,(local-normal*2)[:,None]))>=0 else -1
            collected[frame][seed['foot']].append(dict(curve=curve,seed_id=seed['seed_id'],age=0,direction=0,contrast=20.,support_fraction=1.,fb=0.,material_points=0,visible_curve=curve))
            for direction in [-1,1]:
                current_curve=curve.copy();previous_image=seed_image;previous=frame;accepted=0;reason='shot_end'
                for i in range(frame+direction,len(pts) if direction>0 else -1,direction):
                    if idx['rows'][i]['lower_body_inferred']:reason='source_lower_body_unresolved';break
                    image=cv2.imread(idx['rows'][i]['file']);motion=image_flow(previous_image,image,idx['rows'][previous],idx['rows'][i],[dict(curve=current_curve)],wb['kp2d_coco23'][previous,[15,16],:2])[0]
                    if not motion['image_motion_supported']:reason='image_motion_not_supported';break
                    refined,support,contrast=edge_snap(image,idx['rows'][i],motion['prediction'],sign)
                    visible=visible_segment(refined,support)
                    if visible is None:refined=motion['prediction']
                    if np.linalg.norm(refined[-1]-refined[0])<8:reason='blade_segment_foreshortened_or_unresolved';break
                    collected[i][seed['foot']].append(dict(curve=refined,seed_id=seed['seed_id'],age=abs(i-frame),direction=direction,contrast=float(np.median(contrast[support])) if support.any() else 0.,support_fraction=float(support.mean()),fb=motion['fb_median'],material_points=motion['valid_material_points'],visible_curve=visible));current_curve=refined;previous_image=image;previous=i;accepted+=1
                log.write(json.dumps(dict(event='seed_direction_complete',seed_id=seed['seed_id'],direction=direction,tracked_exposures=accepted,stop_source_seconds=idx['rows'][previous]['source_seconds'],reason=reason,seconds=time.perf_counter()-start))+'\n')
        rows=[]
        for i,pair in enumerate(collected):
            feet=[]
            for foot,cs in enumerate(pair):
                if not cs:feet.append(dict(status='not_located',curve_uv=None,reason='no source-supported continuous track',physical_identity='source foot hypothesis, not human confirmed'));continue
                cs=sorted(cs,key=lambda x:(x['visible_curve'] is None,x['age'],-x['contrast']));best=cs[0];future=[q for q in cs if q['direction']<0 and q['visible_curve'] is not None];past=[q for q in cs if q['direction']>0 and q['visible_curve'] is not None];agreement=None
                if best['visible_curve'] is None:feet.append(dict(status='boot_tracked_blade_not_resolved',curve_uv=None,seed_id=best['seed_id'],age_frames=best['age'],reason='boot image motion retained; no resolved blade pixels',physical_identity='source foot hypothesis, not human confirmed'));continue
                if past and future:agreement=curve_distance(past[0]['visible_curve'],future[0]['visible_curve'])
                status='source_annotation' if best['age']==0 else 'tracked_one_direction'
                if agreement is not None:status='tracked_both_directions' if agreement<=5 else 'track_disagreement'
                feet.append(dict(status=status,curve_uv=best['visible_curve'].tolist(),seed_id=best['seed_id'],age_frames=best['age'],edge_contrast=best['contrast'],edge_support_fraction=best['support_fraction'],material_motion_fb_px=best['fb'],bidirectional_distance_px=agreement,source_track_candidates=len(cs),physical_identity='source foot hypothesis, not human confirmed'))
            if all(q['curve_uv'] is not None for q in feet) and curve_distance(feet[0]['curve_uv'],feet[1]['curve_uv'])<3:
                for q in feet:q['status']='identity_or_occlusion_ambiguous'
            rows.append(dict(pts=int(pts[i]),source_seconds=int(pts[i])/12800,feet=feet,contact_state='unknown',human_reviewed=False,used_for_body_or_camera=False))
        states=sorted({q['status'] for r in rows for q in r['feet']});counts={s:[sum(r['feet'][f]['status']==s for r in rows) for f in [0,1]] for s in states};result=dict(schema='rinkspace.source-anchored-continuous-blade-tracks.v1',source_exposures=len(rows),source_time_base=[1,12800],counts=counts,source_seed_count=len(seeds),seeds=seeds,camera_or_body_projection_used=False,contact_feedback=False,physical_left_right_human_confirmed=False,accuracy_accepted=False,rows=rows,seconds=time.perf_counter()-start);(output/'source_tracks.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8');log.write(json.dumps(dict(event='complete',counts=counts,seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=len(rows),counts=counts,seconds=time.perf_counter()-start)),flush=True);return result

def possible_states(pool):
    count=len(pool);states=np.asarray([(a,b) for a in range(count+1) for b in range(count+1) if a!=b or a==count],int);unary=[]
    for a,b in states:
        cost=0.
        for foot,k in enumerate([a,b]):
            if k<count:cost-=min(pool[k]['quality'],8.);cost+=0.35*(foot not in pool[k]['navigation_labels'])
        if a<count and b<count and curve_distance(pool[a]['curve'],pool[b]['curve'])<3:cost+=50.
        unary.append(cost)
    return states,np.asarray(unary)


def associate_candidates(root,output,review_path=None):
    root,output=Path(root),Path(output);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));candidate=json.loads((output/'candidates.json').read_text(encoding='utf8'));wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');pools=[pool_pair(r['candidates']) for r in candidate['records']];state_list=[];backpointers=[];links=[];start=time.perf_counter();cv2.setNumThreads(2)
    seed_rows={} if review_path is None else {}
    if review_path is not None:
        for seed in json.loads(Path(review_path).read_text(encoding='utf8'))['blade_curves']:seed_rows.setdefault(seed['pts'],[]).append(seed)
    def source_anchor_cost(i,pool,states,unary):
        result=unary.copy()
        for seed in seed_rows.get(idx['rows'][i]['pts'],[]):
            for k,state in enumerate(states):
                j=state[seed['foot']]
                result[k]+=15. if j==len(pool) else min(float(np.mean(point_polyline_distance(seed['uv'],pool[j]['curve'])))/3.,30.)
        return result
    with (output/'temporal_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        previous_image=cv2.imread(idx['rows'][0]['file']);states,unary=possible_states(pools[0]);unary=source_anchor_cost(0,pools[0],states,unary);state_list.append(states);dp=unary;backpointers.append(None);links.append(None)
        for i in range(1,len(pools)):
            current_image=cv2.imread(idx['rows'][i]['file']);motion=image_flow(previous_image,current_image,idx['rows'][i-1],idx['rows'][i],pools[i-1],wb['kp2d_coco23'][i-1,[15,16],:2]);previous_image=current_image;old=pools[i-1];cur=pools[i];distance=np.full((len(old),len(cur)),np.nan);supported=np.zeros(len(old),bool);matrix=np.full((len(old)+1,len(cur)+1),.8);matrix[-1,-1]=0
            for a,item in enumerate(motion):
                supported[a]=item['image_motion_supported']
                prediction=item['prediction']
                if not supported[a]:
                    centre=old[a]['curve'].mean(0);j=np.argmin(np.linalg.norm(wb['kp2d_coco23'][i-1,[15,16],:2]-centre,axis=1));prediction=old[a]['curve']+(wb['kp2d_coco23'][i,[15,16],:2]-wb['kp2d_coco23'][i-1,[15,16],:2])[j]
                for b,item2 in enumerate(cur):distance[a,b]=curve_distance(prediction,item2['curve']);matrix[a,b]=min(distance[a,b]/4,12)+(0 if supported[a] else 2.)
            links.append(dict(distance=distance,supported=supported,material_points=[x['valid_material_points'] for x in motion]));current_states,current_unary=possible_states(cur);current_unary=source_anchor_cost(i,cur,current_states,current_unary);cost=dp[:,None]+matrix[states[:,0,None],current_states[None,:,0]]+matrix[states[:,1,None],current_states[None,:,1]];which=np.argmin(cost,axis=0);dp=cost[which,np.arange(len(current_states))]+current_unary;states=current_states;state_list.append(states);backpointers.append(which)
            if i%100==0:log.write(json.dumps(dict(event='image_temporal_association',frames=i+1,seconds=time.perf_counter()-start))+'\n')
        selected=[None]*len(pools);k=int(np.argmin(dp))
        for i in range(len(pools)-1,-1,-1):selected[i]=state_list[i][k].copy();k=int(backpointers[i][k]) if i else 0
        rows=[];segments=[0,0];was_present=[False,False]
        for i,(pool,choice) in enumerate(zip(pools,selected)):
            feet=[]
            for foot,j in enumerate(choice):
                present=j<len(pool)
                if not present:feet.append(dict(status='not_located',curve_uv=None,reason='source lower body unresolved' if idx['rows'][i]['lower_body_inferred'] else 'no resolved source edge selected',physical_foot_identity='unconfirmed',segment=segments[foot]));was_present[foot]=False;continue
                if not was_present[foot]:segments[foot]+=1
                incoming=False;outgoing=False;incoming_distance=None;outgoing_distance=None
                if i and selected[i-1][foot]<len(pools[i-1]):
                    old_index=selected[i-1][foot];incoming_distance=float(links[i]['distance'][old_index,j]);incoming=bool(links[i]['supported'][old_index] and incoming_distance<=5.)
                if i+1<len(pools) and selected[i+1][foot]<len(pools[i+1]):
                    next_index=selected[i+1][foot];outgoing_distance=float(links[i+1]['distance'][j,next_index]);outgoing=bool(links[i+1]['supported'][j] and outgoing_distance<=5.)
                item=pool[j];tracked=incoming and outgoing and item['contrast']>=8;feet.append(dict(status='tracked_candidate' if tracked else 'image_candidate_unconfirmed',curve_uv=item['curve'].tolist(),source_edge_contrast=item['contrast'],image_flow_incoming=incoming,image_flow_outgoing=outgoing,incoming_distance_px=incoming_distance,outgoing_distance_px=outgoing_distance,quality=item['quality'],navigation_labels=item['navigation_labels'],physical_foot_identity='unconfirmed',segment=segments[foot],reacquired_after_gap=not was_present[foot]));was_present[foot]=True
            rows.append(dict(pts=idx['rows'][i]['pts'],source_seconds=idx['rows'][i]['source_seconds'],feet=feet,contact_state='unknown',used_for_body_or_camera=False,human_reviewed=False))
        counts={status:[sum(r['feet'][f]['status']==status for r in rows) for f in [0,1]] for status in ['tracked_candidate','image_candidate_unconfirmed','not_located']};result=dict(schema='rinkspace.native-source-two-blade-tracks.v1',source_exposures=len(rows),source_time_base=[1,12800],source_size=[1920,1080],counts=counts,track_ids='A/B image continuity hypotheses, not confirmed anatomical left/right',camera_or_body_projection_used=False,contact_feedback=False,human_reviewed=False,accuracy_accepted=False,rows=rows,seconds=time.perf_counter()-start);(output/'tracks.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8');log.write(json.dumps(dict(event='complete',counts=counts,seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=len(rows),counts=counts,seconds=time.perf_counter()-start)),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--review',required=True);a=p.parse_args();run(a.root,a.output,a.review)
