import av,cv2,json,hashlib,importlib.util
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');s=importlib.util.spec_from_file_location('repair',O/'repair_direct_v5.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h);scope=h.load(O/'body_direct_delta_v5/manifest.json');targets={x['native_pts']for x in scope['changed']};h.REVIEWED_EXTRA=targets;T=O/'semantics_qualified_v5';F=O/'source_final_v5_frames';T.mkdir(exist_ok=True);F.mkdir(exist_ok=True);records=[]
with av.open(str(h.h.V))as c:
 st=c.streams.video[0];c.seek(min(targets),stream=st,backward=True)
 for frame in c.decode(st):
  pts=int(frame.pts)
  if pts>max(targets):break
  if pts not in targets:continue
  rgb=frame.to_ndarray(format='rgb24');base=h.load(O/'source_direct_v5_frames'/f'{pts}.json');assert hashlib.sha256(rgb.tobytes()).hexdigest()==base['decoded_native_RGB_sha256'];valid=cv2.imread(h.load(base['body_material_ref']['file'])['body_material_mask_ref']['file'],0);mask,receipt=h.semantic_current(rgb,valid,pts);mp=T/f'{pts}.png';cv2.imwrite(str(mp),mask);receipt.update(decoded_native_RGB_sha256=base['decoded_native_RGB_sha256'],mask_ref=h.ref(mp),normal_only_not_material_point_2D_tracking=True);rp=T/f'{pts}.json';h.write(rp,receipt);r=dict(base);r['semantic_normal_ref']=h.ref(rp);fp=F/f'{pts}.json';h.write(fp,r);records.append(dict(native_pts=pts,source_ref=h.ref(fp),normal_semantic_ref=h.ref(rp)))
h.write(O/'source_final_v5_manifest.json',dict(complete=True,frames=1862,semantic_source_reread_cases=len(records),primary_source_generation=h.ref(O/'source_v5_manifest.json'),qualified_direct_raw_source_repair=h.ref(O/'source_direct_v5_manifest.json'),extra_semantic_source_review=h.ref(O/'direct_component_source_review.json'),actual_semantic_current_RGB_reader=h.ref(Path(__file__)),actual_semantic_helper=h.ref(O/'repair_direct_v5.py'),source_frame_resolution_priority=[str(F),str(O/'source_direct_v5_frames'),str(O/'source_v5_frames')],records=records,all_old_raw_UV_and_ID_members_preserved=True));print(json.dumps(dict(semantic_current_RGB_frames=len(records))))
