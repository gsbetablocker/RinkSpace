import av,cv2,numpy as np,json,gzip,base64,hashlib,sys
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');F=O/'final_frames_v5';V=json.loads(Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/programme_scope.json').read_text(encoding='utf8'))['source']['path']
PTS=[int(x)for x in sys.argv[1:]]or[948992,950528,1015552,1124864,1300992,1301248,1363456,1425408]
for pts in PTS:
 fp=F/f'{pts}.json'
 if not fp.exists():continue
 j=json.loads(fp.read_text(encoding='utf8'));rows=[]
 for m in j['source_members']:
  with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);b=f.read(m['length_bytes'])
  assert hashlib.sha256(b).hexdigest()==m['sha256'];rr=[json.loads(s)for s in gzip.decompress(b).decode('utf8').splitlines()];assert len(rr)==m['row_count'];rows.extend(rr)
 with av.open(V)as c:
  st=c.streams.video[0];c.seek(pts,stream=st,backward=True)
  for frame in c.decode(st):
   if frame.pts==pts:rgb=frame.to_ndarray(format='rgb24');break
 assert hashlib.sha256(rgb.tobytes()).hexdigest()==j['decoded_native_RGB_sha256'];im=rgb[:,:,::-1].copy();show=im.copy();eligible=np.frombuffer(base64.b64decode(j['eligible_uint8_base64']),dtype='u1');status=np.frombuffer(base64.b64decode(j['status_uint8_base64']),dtype='u1')
 for i,r in enumerate(rows):
  if eligible[i]:
   x,y=np.rint(r['uv']).astype(int);color=(255,0,255)if status[i]==7 else(0,255,255);cv2.circle(show,(x,y),1,color,-1,cv2.LINE_AA)
 board=json.loads(Path(j['final_board_ref']['file']).read_text(encoding='utf8'));top=np.asarray(board['roles']['board.top']['full_native_y']);foot=np.asarray(board['roles']['board.contact']['full_native_y']);a=max(0,int(np.nanmin(top))-24);b=min(1080,int(np.nanmax(foot))+24);pair=cv2.vconcat([im[a:b],show[a:b]]);cv2.putText(pair,f'{pts}  eligible={int(eligible.sum())} source={len(rows)}  actual_RGB / qualified (semantic normal-only magenta)',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2);cv2.imwrite(str(O/f'{pts}_final_v5_QA.jpg'),pair,[cv2.IMWRITE_JPEG_QUALITY,95]);print(json.dumps(dict(pts=pts,source=len(rows),eligible=int(eligible.sum()),status={int(k):int(v)for k,v in zip(*np.unique(status,return_counts=True))}) ,default=int))
