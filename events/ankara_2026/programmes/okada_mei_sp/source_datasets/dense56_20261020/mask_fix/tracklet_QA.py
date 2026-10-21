exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/audit_static.py',encoding='utf8').read().split('mode=sys.argv')[0])
O=O/'background_v1';a=rz(O/'instances.jsonl.gz');groups={}
for i,r in enumerate(a):r['_order']=i;groups.setdefault(r['tracklet_id'],[]).append(r)
(O/'tracklet_QA').mkdir(exist_ok=True);pages=[];cells=[];summary=[];cache={}
def tile(i):
 pg=i//12
 if pg not in cache:cache[pg]=cv2.imread(str(O/'QA'/f'page_{pg:03}.jpg'))
 k=i%12;return cache[pg][(k//4)*280:(k//4+1)*280,(k%4)*360:(k%4+1)*360].copy()
for tid,rs in groups.items():
 pair=np.hstack([tile(rs[0]['_order']),tile(rs[-1]['_order'])]);pair=np.vstack([np.full((25,720,3),255,'u1'),pair]);cv2.putText(pair,f'T{tid} {len(rs)} frames {rs[0]["native_pts"]}..{rs[-1]["native_pts"]}',(5,20),0,.6,(0,0,0),1);cells.append(pair)
 summary.append(dict(tracklet_id=tid,frames=len(rs),first_pts=rs[0]['native_pts'],last_pts=rs[-1]['native_pts'],shot_id=rs[0]['shot_id'],first_order=rs[0]['_order'],last_order=rs[-1]['_order'],min_confidence=min(r['proposal']['confidence']for r in rs),max_mask_area_over_proposal=max(r['pixel_count']/max(1,(r['proposal']['box_xyxy'][2]-r['proposal']['box_xyxy'][0])*(r['proposal']['box_xyxy'][3]-r['proposal']['box_xyxy'][1]))for r in rs)))
 if len(cells)==6:
  ff=O/'tracklet_QA'/f'page_{len(pages):02}.jpg';cv2.imwrite(str(ff),np.vstack([np.hstack(cells[k:k+2])for k in(0,2,4)]),[cv2.IMWRITE_JPEG_QUALITY,97]);pages.append(str(ff));cells=[]
if cells:
 cells +=[np.full((305,720,3),240,'u1')for _ in range(6-len(cells))];ff=O/'tracklet_QA'/f'page_{len(pages):02}.jpg';cv2.imwrite(str(ff),np.vstack([np.hstack(cells[k:k+2])for k in(0,2,4)]),[cv2.IMWRITE_JPEG_QUALITY,97]);pages.append(str(ff))
(O/'tracklet_summary.json').write_text(json.dumps(dict(complete=True,tracklets=len(groups),instances=len(a),rows=summary,QA=pages),indent=2),encoding='utf8');print(json.dumps(dict(tracklets=len(groups),QA_pages=len(pages),critical_1235200=[dict(object_index=r['object_index'],tracklet_id=r['tracklet_id'],bbox=r['proposal']['box_xyxy'])for r in a if r['native_pts']==1235200],large_mask=[r for r in summary if r['max_mask_area_over_proposal']>1.6])))
