from pathlib import Path
from collections import Counter
import json,gzip,hashlib,sys
import numpy as np
import cv2,av
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(P/'calibration/rinkfit/src'),str(P/'src')]
from rinkspace.vision_source_shapes import read_source_chromatic_stroke_centers
R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'c_local_v1';O.mkdir(exist_ok=True)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ref(f):return dict(file=str(f),sha256=sha(f))
def write(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
F=O/'source_branch_policy.json';ROWS=O/'source_branch_rows.jsonl'
if F.exists()or ROWS.exists():raise FileExistsError('immutable policy generation exists')
source=R/'tail_C_source_paired_v1/source_rows.jsonl';rows=[json.loads(x) for x in source.read_text(encoding='utf8').splitlines()];assert len(rows)==6328
manifest=json.loads((R/'tail_c_v2/manifest.json').read_text(encoding='utf8'));profilepath=Path(manifest['candidates']['file']);assert sha(profilepath)==manifest['candidates']['sha256']
frames={x['source_binding']['source_pts']:x for x in map(json.loads,gzip.open(profilepath,'rt',encoding='utf8'))};assert len(frames)==80
profiles={};policyrows=[];counts=Counter();max_flank=0.;replay=[]
for pts,frame in frames.items():
 for reading in frame['curve_outputs'][0]['native_source_readings']['readings']:profiles[pts,reading['source_query_index']]=reading
for j,sr in enumerate(rows):
 pts=sr['native_pts'];qi=sr['original_candidate_reference']['query_index'];reading=profiles[pts,qi];offsets=np.asarray(reading['actual_source_normal_offsets_px']);profile=np.asarray(reading['actual_source_color_profile']);background=np.asarray(reading['actual_source_color_background_profile']);paint=np.maximum(profile-background,0.);smooth=cv2.GaussianBlur(paint[:,None],(1,5),1.).ravel();second=np.diff(smooth,n=2);noise=float(1.4826*np.median(np.abs(second-np.median(second))));floor=max(2.,3*noise,.1*float(smooth.max()));supported=smooth>floor;starts=np.flatnonzero(supported&~np.r_[False,supported[:-1]]);ends=np.flatnonzero(supported&~np.r_[supported[1:],False]);comps=[]
 for a,b in zip(starts,ends):
  if b-a<3:continue
  if comps and a-comps[-1][1]-1<=3:comps[-1][1]=int(b)
  else:comps.append([int(a),int(b)])
 ci=sr['actual_material_reading']['source_candidate_index'];a,b=comps[ci];lo,hi=max(0,a-10),min(len(profile)-1,b+10);mass=paint[lo:hi+1];mid=float(np.interp(.5,(np.cumsum(mass)-mass*.5)/mass.sum(),offsets[lo:hi+1]));side=sr['source_side_index'];sign=1 if side==0 else -1;gridmid=.5*(offsets[:-1]+offsets[1:]);region=((gridmid>=offsets[lo])&(gridmid<=mid)) if side==0 else ((gridmid>=mid)&(gridmid<=offsets[hi]));gradient=np.diff(smooth);gm=np.where(region,np.maximum(sign*gradient,0.),0.);support=gm>0;sa=np.flatnonzero(support&~np.r_[False,support[:-1]]);sb=np.flatnonzero(support&~np.r_[support[1:],False]);lobes=[[float(offsets[x]),float(offsets[y+1])] for x,y in zip(sa,sb)];measured=sr['source_measured_alpha'];q50=float(np.interp(.5,np.r_[0.,np.cumsum(gm)]/gm.sum(),offsets));max_flank=max(max_flank,abs(q50-measured));assert abs(q50-measured)<1e-9
 chosen=[x for x in lobes if x[0]-1e-10<=measured<=x[1]+1e-10];status='candidate_unique_source_signed_gradient_lobe' if len(chosen)==1 else 'source_branch_unresolved';counts[status]+=1
 row=dict(raw_row_index=910012+j,source_row_id=sr['source_row_id'],native_pts=pts,query_index=qi,source_side_index=side,material_transition='ice_to_colored_paint'if side==0 else'colored_paint_to_ice',source_measured_alpha=measured,source_uv=sr['source_uv'],source_RGB_sha256=sr['current_rgb_sha256'],source_qualification=sr['source_geometry_qualification'],source_component_alpha_extent=sr['decision']['actual_C_component_normal_extent_px'],source_estimator_component_index=ci,source_estimator_component_support=[float(offsets[a]),float(offsets[b])],source_estimator_boundary_integration_support=[float(offsets[lo]),mid]if side==0 else[mid,float(offsets[hi])],all_matching_signed_gradient_lobes=lobes,candidate_branch_alpha=chosen[0]if len(chosen)==1 else None,status=status,profile_signal_noise_scale=noise,component_selection_floor=floor,original_acquisition_alpha=sr['source_acquisition_query']['alpha'],measured_q50_replay=q50,branch_is_not_E_sigma_or_zero_loss_allowance=True,source_UV_ID_unchanged=True,algorithmic_policy_candidate_not_manual_physical_boundary_certification=True)
 policyrows.append(row)
# Re-read only the two requested real frames and the selected median/max profile.
selected=[(2402560,912470-910012),(2415104,916339-910012)];wanted={x[0] for x in selected};video=Path(manifest['source_video']['path'])
with av.open(str(video)) as con:
 st=con.streams.video[0];con.seek(min(wanted),stream=st)
 for fr in con.decode(st):
  pts=int(fr.pts)
  if pts>max(wanted):break
  if pts not in wanted:continue
  rgb=fr.to_ndarray(format='rgb24');digest=hashlib.sha256(rgb.tobytes()).hexdigest();frame=frames[pts];assert digest==frame['source_binding']['decoded_native_rgb_sha256']
  for selectedpts,j in selected:
   if selectedpts!=pts:continue
   sr=rows[j];qi=sr['original_candidate_reference']['query_index'];old=profiles[pts,qi];fresh=read_source_chromatic_stroke_centers(rgb,binding=frame['source_binding'],search_points_uv=[old['source_search_point_uv']],normals=[old['source_search_normal']],search_intervals_px=[old['source_search_normal_interval_px']],keep_profiles=True,expected_RGB_sha256=digest)['readings'][0]
   diff=max(float(np.max(np.abs(np.asarray(fresh[k])-np.asarray(old[k])))) for k in ['actual_source_normal_offsets_px','actual_source_color_profile','actual_source_color_background_profile']);assert diff==0
   replay.append(dict(native_pts=pts,source_row_id=sr['source_row_id'],RGB_sha256=digest,full_raw_profile_replay_max_abs_diff=diff,candidate_branch=policyrows[j]['candidate_branch_alpha'],actual_measured_alpha=sr['source_measured_alpha']))
with ROWS.open('w',encoding='utf8') as f:
 for x in policyrows:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
report=dict(schema='rinkfit.C_two_flank_source_signed_gradient_branch_policy_candidate.v1',raw_rows=6328,source_sections=3164,source_exposures=80,counts=dict(counts),policy_rows=ref(ROWS),frozen_source_rows=ref(source),frozen_source_profiles=ref(profilepath),source_generation=ref(R/'tail_c_v2/manifest.json'),parent_qualification=ref(R/'tail_c_review_v1/qualification.json'),source_estimator_reader=ref(P/'src/rinkspace/vision_source_shapes.py'),policy_builder=ref(Path(__file__)),source_UV_ID_or_packet_modified=False,model_or_screen_geometry_used_in_policy=False,source_q50_replay_max_abs_diff=max_flank,actual_RGB_replays=replay,rule='Replay original selected color component and each signed-gradient boundary integration support. Candidate branch is the unique connected positive signed-gradient support containing the existing observed q50; disconnected same-polarity lobes are retained as alternatives, not auto selected. Full acquisition ROI and q10/q90 are not solver support.',uncertainty_or_noise_sigma_issued=False,finite_local_material_branch_algorithmic_candidate=True,manual_physical_branch_acceptance_claimed=False,production_solver_integration=False,quality_accepted=False)
write(F,report);print(json.dumps(report,ensure_ascii=False),flush=True)
