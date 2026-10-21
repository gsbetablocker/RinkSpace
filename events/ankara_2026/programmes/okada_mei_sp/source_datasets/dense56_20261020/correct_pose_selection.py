import json,gzip,hashlib,cv2,numpy as np
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1');base=O/'clean_parent52_v2/index.jsonl.gz';cl={r['native_pts']:r for r in map(json.loads,gzip.open(base,'rt'))};src=O/'pose_proposals.jsonl.gz';dst=O/'pose_selection_v2.jsonl.gz';changed=0;n=0
with gzip.open(src,'rt')as f,gzip.open(dst,'xt',encoding='utf8')as out:
 for line in f:
  r=json.loads(line);p=r['native_pts']
  if p in cl:
   q=cl[p];m=cv2.imread(q['mask_file'],0);nn,labels,stats,_=cv2.connectedComponentsWithStats((m>0).astype('u1'),8);main=labels==(1+int(np.argmax(stats[1:,4])));area=int(main.sum());scores=[]
   for pp in r['person_proposals']:
    x,y,X,Y=pp['box_xyxy'];x=max(0,int(x));y=max(0,int(y));X=min(1920,int(np.ceil(X)));Y=min(1080,int(np.ceil(Y)));a=int(main[y:Y,x:X].sum());scores.append(a/max(area+(X-x)*(Y-y)-a,1))
   selected=int(np.argmax(scores));changed+=r['selected_person_list_index']!=selected;r['selected_person_list_index']=selected;r['selected_keypoints_uv']=r['person_proposals'][selected]['keypoints_uv'];r['selected_keypoint_confidence']=r['person_proposals'][selected]['keypoint_confidence'];r['selection_revision_evidence']=dict(clean_person_mask_file=q['mask_file'],clean_person_mask_sha256=q['mask_sha256'],score_definition='actual_current_person_mask_main_component_overlap_navigation_only',candidate_scores=scores)
  assert r['selected_person_list_index']is not None;out.write(json.dumps(r,separators=(',',':'))+'\n');n+=1
print(json.dumps(dict(frames=n,reselected_frames=changed,file=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())))
