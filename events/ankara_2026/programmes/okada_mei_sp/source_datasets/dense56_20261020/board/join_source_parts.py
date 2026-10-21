from source_reader_anchored import *
parts=[('s05a',948992,1129216),('s05b',1129472,1300992),('s06',1301248,1425408)]
for label,lo,hi in parts:
 f=O/('anchored_complete_'+label+'.json')
 while not f.exists():time.sleep(.5)
 q=readj(f);assert q['complete']and q['reader_sha256']==sha(O/'source_reader_anchored.py')
for p in PTS:
 q=readj(O/'anchored_frames'/(str(p)+'.json'));assert q['native_pts']==p
write(O/'anchored_complete.json',dict(complete=True,native_frames=len(PTS),shots={'s05':1376,'s06':486},parts=parts,partition_points_are_original_full_KF_reset=True,reader_sha256=sha(O/'source_reader_anchored.py'),source_video_sha256=readj(J/'programme_scope.json')['source']['sha256'],quality_review_required_before_final_delivery=True,fitting_started=False))
print('ANCHORED COMPLETE 1862',flush=True)
