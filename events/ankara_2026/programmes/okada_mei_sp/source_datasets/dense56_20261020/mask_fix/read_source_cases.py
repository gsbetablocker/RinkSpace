exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/diagnose.py',encoding='utf8').read().split('with av.open')[0])
wanted={1029376,1029632,1202432,1214464,1217280,1213440,1219072,1213696,1213952};bd={q['native_pts']:q for q in rz(R/'dense56_v1/shoe_fix/full_v3/index.stream.jsonl.gz')}
with av.open(source['file'])as c:
 st=c.streams.video[0];c.seek(min(wanted),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(wanted):break
  if p not in wanted:continue
  bgr=fr.to_ndarray(format='bgr24');cv2.imwrite(str(O/f'actual_{p}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,99]);ov=bgr.copy();m=cv2.imread(mi[p]['mask_file'],0)>0;cs,_=cv2.findContours(m.astype('u1'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(0,255,0),1)
  if p in{1029376,1029632}:crop=np.hstack([bgr[260:480,1050:1230],ov[260:480,1050:1230]]);cv2.imwrite(str(O/f'T84_actual_old_{p}.jpg'),cv2.resize(crop,None,fx=2,fy=2),[cv2.IMWRITE_JPEG_QUALITY,99])
