import json,gzip,hashlib,base64,numpy as np,cv2
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');F=O/'final_frames_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(pts):
 p=F/f'{pts}.json';j=json.loads(p.read_text(encoding='utf8'));rows=[]
 for m in j['source_members']:
  with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);bb=f.read(m['length_bytes'])
  assert hashlib.sha256(bb).hexdigest()==m['sha256'];rows.extend(json.loads(s)for s in gzip.decompress(bb).decode('utf8').splitlines())
 return j,rows,np.frombuffer(base64.b64decode(j['eligible_uint8_base64']),dtype='u1'),np.frombuffer(base64.b64decode(j['status_uint8_base64']),dtype='u1')
result=[]
for pts in [1124864,1187584,1187840,1188096,1188352]+list(range(1411328,1414912+1,256)):
 j,rows,el,code=read(pts);xy=np.asarray([r['uv']for r in rows]);record=dict(native_pts=pts,final_frame_ref=dict(file=str(F/f'{pts}.json'),sha256=sha(F/f'{pts}.json')),source_RGB_sha256=j['decoded_native_RGB_sha256'],board_ref=j['final_board_ref'],person_mask_ref=j['mask_ref'],source_unknown_exclusion_ref=j.get('unknown_exclusion_ref'),eligible_rows=int(el.sum()),semantic_normal_only_rows=int((code==7).sum()))
 if pts==1124864:
  for name,sel in [('old_cap_glint_counterexample',(xy[:,0]>=1522)&(xy[:,0]<=1684)&(xy[:,1]>=239)&(xy[:,1]<=250)),('current_actual_technical_UI',(xy[:,0]>=60)&(xy[:,0]<400)&(xy[:,1]>=90)&(xy[:,1]<257))]:
   record[name]=dict(raw_rows=int(sel.sum()),eligible_rows=int(el[sel].sum()),status_counts={str(int(k)):int(v)for k,v in zip(*np.unique(code[sel],return_counts=True))});assert not el[sel].any()
 if pts in(1187584,1187840,1188096,1188352):
  prior=json.loads((O/'arm_occlusion_review/sleeve_interior_diagnostic.json').read_text(encoding='utf8'));poly=next(x['manually_reviewed_sleeve_interior_diagnostic_polygon']for x in prior if x['native_pts']==pts);sel=np.asarray([cv2.pointPolygonTest(np.asarray(poly,'f4'),tuple(uv),False)>=0 for uv in xy]);record['actual_source_sleeve_interior_diagnostic']=dict(raw_rows=int(sel.sum()),eligible_rows=int(el[sel].sum()),not_an_exclusion_mask=True,status_counts={str(int(k)):int(v)for k,v in zip(*np.unique(code[sel],return_counts=True))});assert not el[sel].any()
 if 1411328<=pts<=1414912:
  normal=np.frombuffer(base64.b64decode(j['normal_only_uint8_base64']),dtype='u1');texture=np.frombuffer(base64.b64decode(j['texture_2D_tracking_uint8_base64']),dtype='u1');types=np.frombuffer(base64.b64decode(j['observation_type_uint8_base64']),dtype='u1');assert (normal[code==7]==1).all()and(texture[code==7]==0).all()and(types[code==7]==1).all()
 result.append(record)
(O/'final_counterexample_audit.json').write_text(json.dumps(dict(complete=True,final_mask_index_sha256='0ab10c6b0cac0dfc239198a591eecf7eed2588f1b3c315ecda0cff8f125e3620',frames=result,raw_UV_not_moved=True,world_correspondence_not_issued=True),ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps([dict(pts=j['native_pts'],eligible=j['eligible_rows'],normal7=j['semantic_normal_only_rows'])for j in result]))
