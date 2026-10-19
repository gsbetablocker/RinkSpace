"""Read-only forward/backward source-flow checks of same board materials."""
from __future__ import annotations
import cv2
import numpy as np
from .extract_first_shot_board_source import DATA, read, write


def closest_segment(points, segments):
    distance=np.full(len(points),np.inf)
    for segment in segments:
        path=np.asarray(segment.get('source_native_readout_uv',segment['uv']),float)
        if len(path)<2:continue
        a,b=path[:-1],path[1:];delta=b-a;length=np.linalg.norm(delta,axis=1)
        fraction=np.sum((points[:,None,:]-a)*delta,axis=2)/np.maximum(length**2,1.e-8)
        local=np.linalg.norm(points[:,None,:]-(a+np.clip(fraction,0,1)[:,:,None]*delta),axis=2)
        local[:,length>45]=np.inf
        # A disappearing run endpoint supplies no normal-distance evidence.
        local[(fraction<0)|(fraction>1)]=np.inf
        distance=np.minimum(distance,local.min(1))
    return distance


def audit():
    destination=DATA/'repair_20261014/board_temporal_diagnostics';destination.mkdir(parents=True,exist_ok=True)
    raw=cv2.imread(str(DATA/'frames/000201.png'));previous=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(960,540))
    old=read(DATA/'board_candidates/000201.json');findings=[];checked=0;valid_points=0
    for index in range(202,1249):
        raw=cv2.imread(str(DATA/f'frames/{index:06d}.png'));current=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(960,540))
        new=read(DATA/f'board_candidates/{index:06d}.json');by_id={f['target_id']:f for f in new['features']}
        entries=[(f['target_id'],p)for f in old['features']if f['target_id']in by_id and not f['target_id'].startswith('Ad11.')for s in f['segments']for p in s['uv']]
        if not entries:previous=current;old=new;continue
        points=np.asarray([p for _,p in entries],np.float32)[:,None,:]*.5
        forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
        backward,status_b,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
        good=(status[:,0]>0)&(status_b[:,0]>0)&(np.linalg.norm(backward[:,0]-points[:,0],axis=1)<.5)
        native=forward[:,0]*2
        for target in sorted({t for t,p in entries}):
            selected=np.array([j for j,(t,p)in enumerate(entries)if t==target and good[j]])
            if len(selected)<4:continue
            distances=closest_segment(native[selected],by_id[target]['segments']);valid=np.isfinite(distances)
            if valid.sum()<4:continue
            checked+=1;valid_points+=int(valid.sum());bad=np.flatnonzero(valid&(distances>5))
            if len(bad)<2:continue
            findings.append(dict(index=index,target_id=target,checked=int(valid.sum()),bad_count=len(bad),
                                 maximum_normal_distance_px=float(distances[bad].max()),median_normal_distance_px=float(np.median(distances[valid])),
                                 preceding_source_points=[entries[selected[k]][1]for k in bad],
                                 current_source_flow_uv=native[selected[bad]].tolist()))
        previous=current;old=new
    findings.sort(key=lambda r:(-r['bad_count'],-r['maximum_normal_distance_px']))
    write(destination/'source_flow_candidates.json',dict(schema='rinkspace.board-current-source-temporal-diagnostic.v1',
          programme_frame_range_1based=[201,1248],frame_pairs=1047,checked_entity_pairs=checked,forward_backward_confirmed_points=valid_points,
          original_delivery_modified=False,diagnostic_only=True,flow_or_interpolation_never_added_as_observations=True,
          findings=require_visual_review(findings)))
    print('Source-flow diagnostic',len(findings),'flagged entity pairs; first',findings[:8],flush=True)


def require_visual_review(findings):
    for row in findings:row['status']='needs_actual_source_identity_and_visibility_review_not_automatic_failure'
    return findings


if __name__=='__main__':audit()
