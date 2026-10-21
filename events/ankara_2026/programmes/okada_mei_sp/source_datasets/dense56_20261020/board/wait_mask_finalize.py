from source_reader_anchored import *
import subprocess
mf=R/'dense56_footwear_v1/manifest.json'
write(O/'finalization_state.json',dict(stage='waiting_final_source_mask_accepted_freeze',source_native_frames_complete=1862,UV_arrays_complete=True,READY_issued=False))
print('Waiting accepted footwear/unknown source-mask freeze',flush=True)
while True:
 if mf.exists():
  z=readj(mf)
  if z.get('complete')and z.get('accepted_for_final_source_qualification'):break
 time.sleep(1.)
ip=Path(z['index_file']);assert sha(ip)==z['index_sha256'];write(O/'final_mask_binding.json',dict(file=str(ip),sha256=z['index_sha256'],manifest_file=str(mf),manifest_sha256=sha(mf),person_mask_sha_is_not_unknown_exclusion_sha=True,person_guard_px=8,unknown_guard_px=8,source_UV_not_moved=True))
for script in ['finalize_qualified.py','contract_audit.py','boundary_change_receipt.py','jump_source_receipt.py']:
 write(O/'finalization_state.json',dict(stage=script,source_native_frames_complete=1862,final_mask_index_sha256=z['index_sha256'],READY_issued=False));print('START '+script,flush=True);subprocess.run([sys.executable,'-B','-X','utf8',str(O/script)],check=True)
write(O/'finalization_state.json',dict(stage='qualified_complete_waiting_final_visual_review_and_release',source_native_frames_complete=1862,qualified_native_frames=1862,final_mask_index_sha256=z['index_sha256'],READY_issued=False));print('QUALIFIED/AUDIT/DELTA COMPLETE',flush=True)
