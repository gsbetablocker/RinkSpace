from source_reader_anchored import *
target=PTS if len(sys.argv)==1 else[p for p in PTS if int(sys.argv[1])<=p<=int(sys.argv[2])];label=''if len(sys.argv)==1 else'_'+sys.argv[3]
prev=None;flags=[];bad=[];weak=[];minimum=1e9;done=0
for p in target:
 path=O/'anchored_frames'/(str(p)+'.json')
 while not path.exists():time.sleep(.2)
 q=readj(path);ys={ro:np.array(z['full_native_y'])for ro,z in q['roles'].items()};gap=ys['board.contact']-ys['board.top'];minimum=min(minimum,float(gap.min()))
 if gap.min()<=0:bad.append(dict(native_pts=p,min_gap_px=float(gap.min())))
 if prev and sid(p)==prev[0]:
  for ro,y in ys.items():
   d=abs(y-prev[1][ro]);v=float(np.median(d))
   if v>15:flags.append(dict(native_pts=p,role=ro,median_change_px=v,q95_change_px=float(np.percentile(d,95))))
 for ro,z in q['roles'].items():
  if z['material_band']['current_material_columns']<192:weak.append(dict(native_pts=p,role=ro,current_material_columns=z['material_band']['current_material_columns']))
 prev=(sid(p),ys);done+=1
 if done%25==0 or done==len(target):
  r=dict(complete=done==len(target),native_frames=done,last_native_pts=p,min_top_contact_gap_px=minimum,bad_order=bad,median_temporal_jump_over15px=flags,low_material_support_frames=weak,all_current_source_arrays_checked=True,fit_started=False);write(O/('live_source_audit'+label+'.json'),r);print(json.dumps(r),flush=True)
