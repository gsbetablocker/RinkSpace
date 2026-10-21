from source_reader_anchored import *
import subprocess
while True:
 s=O/'finalization_state.json'
 if s.exists()and readj(s).get('stage')=='qualified_complete_waiting_final_visual_review_and_release':break
 time.sleep(1.)
for points,name in [('948992,986624,987648,1007616,1015552,1099776','accepted_start_arc_review.jpg'),('1124864,1235200,1290240,1357568,1368064,1368576','accepted_mid_failure_review.jpg'),('1415168,1418240,1424896,1425408','accepted_tail_jump_review.jpg'),('958464,1000192,1124096,1267968,1328128,1400576','accepted_heldout_review.jpg')]:subprocess.run([sys.executable,'-B','-X','utf8',str(O/'self_review.py'),points,name],check=True)
j=O/'known_jump_source_review.json'
if not j.exists()or not all(r.get('qualified_overlay')for r in readj(j)['records']):subprocess.run([sys.executable,'-B','-X','utf8',str(O/'jump_source_receipt.py')],check=True)
images=[str(O/n)for n in ['accepted_start_arc_review.jpg','accepted_mid_failure_review.jpg','accepted_tail_jump_review.jpg','accepted_heldout_review.jpg']]
write(O/'final_visual_artifacts_complete.json',dict(complete=True,images=[dict(file=p,sha256=sha(p))for p in images],source_arrays_and_final_qualified_rows_used=True,human_source_review_record_still_required=True))
print('FINAL QUALIFIED SOURCE REVIEW ARTIFACTS COMPLETE',flush=True)
