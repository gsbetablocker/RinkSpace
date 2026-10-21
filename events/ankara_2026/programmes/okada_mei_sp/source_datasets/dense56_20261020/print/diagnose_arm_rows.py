import json,gzip,base64,numpy as np,cv2,hashlib
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');polys={1187584:[[753,296],[795,324],[787,336],[747,310]],1187840:[[740,341],[799,354],[792,366],[737,353]],1188096:[[647,356],[785,381],[781,395],[645,369]],1188352:[[570,389],[753,396],[748,410],[568,402]]};out=[]
for pts,poly in polys.items():
 j=json.loads((O/'final_frames_v4_final'/f'{pts}.json').read_text(encoding='utf8'));rows=[]
 for m in j['source_members']:
  with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);bb=f.read(m['length_bytes'])
  assert hashlib.sha256(bb).hexdigest()==m['sha256'];rows.extend(json.loads(s)for s in gzip.decompress(bb).decode('utf8').splitlines())
 contour=np.asarray(poly,np.float32);idx=[i for i,r in enumerate(rows)if cv2.pointPolygonTest(contour,tuple(r['uv']),False)>=0];code=np.frombuffer(base64.b64decode(j['status_uint8_base64']),dtype='u1');el=np.frombuffer(base64.b64decode(j['eligible_uint8_base64']),dtype='u1');out.append(dict(native_pts=pts,manually_reviewed_sleeve_interior_diagnostic_polygon=poly,diagnostic_not_an_exclusion_mask=True,raw_sleeve_interior_rows=len(idx),prior_eligible_sleeve_interior_rows=int(el[idx].sum()),status_counts={str(int(k)):int(v)for k,v in zip(*np.unique(code[idx],return_counts=True))},raw_rows=[dict(row_index=i,uv=rows[i]['uv'],point_id=rows[i]['point_id'],prior_status=int(code[i]))for i in idx]))
(O/'arm_occlusion_review/sleeve_interior_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps([{k:v for k,v in j.items()if k!='raw_rows'}for j in out]))
