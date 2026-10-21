import source_reader_anchored as a
import av,cv2,numpy as np,time
intervals=[(948992,950784),(986368,997632),(1007360,1015808),(1223424,1235200),(1280768,1290496),(1347584,1357568),(1357568,1368832),(1406720,1425408)]
chosen=[997632,1235200,1357568,1415168,1418240,1424896];records=[];done=0;t=time.time()
for lo,hi in intervals:
 prev=None
 with av.open(a.readj(a.J/'programme_scope.json')['source']['path'])as c:
  s=c.streams.video[0];c.seek(lo,stream=s,backward=True)
  for f in c.decode(s):
   p=int(f.pts)
   if p<lo:continue
   if p>hi:break
   im=f.to_ndarray(format='bgr24');r=a.read_frame(p,im,prev);rec=dict(native_pts=p,decoded_native_BGR_sha256=r['decoded_native_BGR_sha256'],roles={})
   if prev:
    for ro,rr in r['roles'].items():
     yy=np.array(rr['full_native_y']);old=np.array(prev[0]['roles'][ro]['full_native_y']);occ=np.array(rr['material_band']['full_material_occlusion'])|np.array(prev[0]['roles'][ro]['material_band']['full_material_occlusion']);d=abs(yy-old);rec['roles'][ro]=dict(median_same_x_change_px=float(np.median(d[~occ])),q95_same_x_change_px=float(np.percentile(d[~occ],95)))
   if p in chosen and prev is not None:rec['actual_adjacent_board_LK']=r['temporal_source_check'];a.write(a.O/(str(p)+'_sequential_probe.json'),r)
   rec['top_contact_min_gap_px']=float(np.min(np.array(r['roles']['board.contact']['full_native_y'])-np.array(r['roles']['board.top']['full_native_y'])));records.append(rec);prev=(r,im);done+=1
   if done%50==0:print(dict(done=done,last_pts=p,elapsed=time.time()-t),flush=True)
a.write(a.O/'sequential_probe_receipt.json',dict(complete=True,intervals=intervals,source_reader_sha256=a.sha(a.O/'source_reader_anchored.py'),tested_frame_reads=done,records=records,actual_source_supported=True,fit_started=False))
print('COMPLETE',done,flush=True)
