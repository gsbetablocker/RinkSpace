import av,cv2,numpy as np,json,hashlib,importlib.util
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent
s=importlib.util.spec_from_file_location('reader',O/'patch_v5_native.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h);out=O/'semantic_interval_review';out.mkdir(exist_ok=True);tiles=[];proof=[]
with av.open(str(h.h.V))as c:
 st=c.streams.video[0];c.seek(1411328,stream=st,backward=True)
 for frame in c.decode(st):
  pts=int(frame.pts)
  if pts<1411328:continue
  if pts>1414912:break
  rgb=frame.to_ndarray(format='rgb24');board=h.load(R/'dense56_board_v1/material_v5/anchored_frames'/f'{pts}.json');actor=cv2.imread(str(R/'dense56_masks_v1/masks'/f'{pts}.png'),0)>0;gr,_=h.graphics.current_graphics(rgb[:,:,::-1].copy());valid,nav,rec=h.body.body_material_domain(board,rgb,actor,gr);sm,sr=h.semantics.current_static_print_semantics(rgb,valid,pts);ref=h.load(O/'source_v4_frames'/f'{pts}.json');rows=h.rowsfrom(ref);surface=h.profiles(rows,valid);xy=np.rint(np.asarray([r['uv']for r in rows])).astype(int);smd=cv2.dilate(sm,np.ones((7,7),'u1'))>0;good=surface&smd[np.clip(xy[:,1],0,1079),np.clip(xy[:,0],0,1919)]&np.array([r.get('pixel_support',{}).get('normal')is not None and r.get('pixel_support',{}).get('contrast_native_DN',0)>=12 for r in rows]);tile=cv2.resize(rgb[:,:,::-1][220:640],(640,140),interpolation=cv2.INTER_AREA);cv2.putText(tile,f'{pts}  {int(good.sum())} normal profiles',(6,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1);tiles.append(tile);rec.update(native_pts=pts,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),prototype_source_person_mask_not_final=True,all_surface_rows=int(surface.sum()),semantic_fullprofile_rows=int(good.sum()),recognizable_print_semantics=sr);proof.append(rec)
cv2.imwrite(str(out/'all15_source_RGB.jpg'),cv2.vconcat([cv2.hconcat(tiles[i:i+3])for i in range(0,len(tiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,94]);(out/'all15_source_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps([dict(pts=j['native_pts'],body=j['all_surface_rows'],normal=j['semantic_fullprofile_rows'])for j in proof]))
