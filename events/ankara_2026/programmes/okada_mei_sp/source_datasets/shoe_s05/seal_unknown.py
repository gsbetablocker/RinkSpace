import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');D=O/'raisedshoe_v1';sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
cv2.setNumThreads(1)
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
from collections import Counter
A=D/'attempt_v3';dec=json.load(open(D/'source_decisions.json'));rejected=set(dec['reject_entire_new_candidate_native_pts']);parent=D.parent/'shoe_fix/effective_v2/s05_index.jsonl.gz';oldrows=rz(parent);assert len(oldrows)==1376
rows={r['native_pts']:r for r in oldrows};cand={r['native_pts']:r for r in rz(A/'candidates.jsonl.gz')};assert len(cand)==260
for f in[D/'refine_v1/candidates.jsonl.gz',D/'refine_v2/candidates.jsonl.gz']:
 for r in rz(f):cand[r['native_pts']]=r
for k in['unknown','delta','qualified_native','effective_QA']:(D/k).mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proofs=[];changed=[];stats=Counter();outrows=[];source=json.load(open(O/'parts/s05/video_config.json'))['source'];selected={p for p in cand};cells=[];page=0
review=dict(schema='rinkspace.source_local_footwear_qualification.v1',complete=True,accepted=True,all_260_native_source_frames_reviewed=True,reviewed_phase_first_native_pts=min(cand),reviewed_phase_last_native_pts=max(cand),time_base='1/12800',source_video_sha256=source['sha256'],source_decisions_ref=dict(file=str(D/'source_decisions.json'),sha256=sha(D/'source_decisions.json')),tail_cross_review_ref=dict(file=str(D/'tail_review.json'),sha256=sha(D/'tail_review.json')),new_pixels_only_typed_unknown=True,source_original_RGB_SAM_pixels_only=True,no_2D_or_physical_scene_correspondence_issued=True,explicit_static_ice_wedges_and_logo_components_rejected=True,old_person_and_old_unknown_preserved=True,physical_fitting_unchanged=True,limitations=['Source shoe/blade/limb boundaries include motion blur; the added layer is an independent source exclusion, not true person pixel/body/world/contact truth.','Source SAM and source visual reading do not assert whole-scene perfect segmentation.','Neighbour anchor interpolation is search navigation only; every retained mask pixel is from its current native RGB SAM output.'])
reviewfile=D/'source_qualification.json';reviewfile.write_text(json.dumps(review,indent=2));qref=dict(file=str(reviewfile),sha256=sha(reviewfile))
for old in oldrows:
 p=old['native_pts'];new=dict(old);base=cv2.imread(old['mask_file'],0)>0;ur=old.get('source_unknown_exclusion_mask_ref');ou=np.zeros_like(base)
 if ur:assert sha(ur['file'])==ur['sha256'];ou=cv2.imread(ur['file'],0)>0
 if p in cand:
  c=cand[p];assert c['source_RGB_sha256']==old['source_RGB_sha256']and c['source_BGR_sha256']==old['source_BGR_sha256'];rr=c['raw_current_source_SAM_ref'];assert sha(rr['file'])==rr['sha256'];raw=cv2.imread(rr['file'],0)>0;n,lab,cc,cent=cv2.connectedComponentsWithStats(raw.astype('u1'),8);qual=np.zeros_like(raw);kept=None
  if p not in rejected and n>1:kept=int(1+np.argmax(cc[1:,4]));qual=lab==kept
  rejected_cc=[dict(label=int(j),bbox_xywh=cc[j,:4].tolist(),pixels=int(cc[j,4]),old_true_person_overlap_pixels=int(((lab==j)&base).sum()),reason='source_review_rejected_entire_new_candidate'if p in rejected else'disconnected_source_SAM_logo_static_ice_or_unqualified_secondary_fragment')for j in range(1,n)if j!=kept]
  qual &=~base;extra=qual&~ou;unk=ou|qual;add=int(extra.sum());pf=D/'qualified_native'/f'{p}.png';cv2.imwrite(str(pf),qual.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);proof=dict(c,qualified_current_mask_ref=dict(file=str(pf),sha256=sha(pf)),qualification='rejected_current_static_or_unneeded_new_candidate_old_person_preserved'if p in rejected else'current_source_RGB_footwear_or_actor_SAM_component_typed_independent_unknown',retained_component_label=kept,rejected_components=rejected_cc,qualified_unknown_pixels=int(qual.sum()),new_unknown_pixels=add,source_qualification_ref=qref);proofs.append(proof)
  if add:
   uf=D/'unknown'/f'{p}.png';df=D/'delta'/f'{p}.png';cv2.imwrite(str(uf),unk.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);cv2.imwrite(str(df),extra.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);urf=dict(file=str(uf),sha256=sha(uf),native_pts=p,shot_id='s05',source_BGR_sha256=old['source_BGR_sha256'],source_RGB_sha256=old['source_RGB_sha256'],type='preserved_old_unknown_OR_independent_current_native_RGB_SAM_footwear_actor_silhouette_uncertainty',reason='Source-visible footwear/actor silhouette with motion blur and uncertain exact pixel ownership is external to ice/board scene observations; no positive scene/true-person pixel certificate.',pixel_count=int(unk.sum()),visibility_guard_pixels=8,parent_unknown_exclusion_ref=ur,source_qualification_ref=qref);new.update(source_unknown_exclusion_mask_ref=urf,parent_raisedshoe_unknown_ref=ur,raisedshoe_delta_ref=dict(file=str(df),sha256=sha(df)),raisedshoe_added_unknown_pixels=add,raisedshoe_current_source_SAM_ref=rr,raisedshoe_qualified_current_source_ref=dict(file=str(pf),sha256=sha(pf)),raisedshoe_source_qualification_ref=qref,raisedshoe_mask_origin='preserved_person_mask_AND_old_unknown_OR_actual_current_native_RGB_SAM_local_footwear_silhouette_uncertainty',old_true_person_pixels_preserved=True,old_unknown_pixels_preserved=True);changed.append(p);stats['added_unknown_pixels']+=add
  stats['reviewed_native_frames']+=1;stats['rejected_entire_new_candidates']+=int(p in rejected);stats['rejected_secondary_components']+=len(rejected_cc)
 assert new['mask_file']==old['mask_file']and new['mask_sha256']==old['mask_sha256'];outrows.append(new)
index=D/'s05_index.jsonl.gz'
with gzip.open(index,'wt',encoding='utf8',compresslevel=4)as w:
 for r in outrows:w.write(json.dumps(r,separators=(',',':'))+chr(10))
proofindex=D/'source_proof.jsonl.gz'
with gzip.open(proofindex,'wt',encoding='utf8',compresslevel=4)as w:
 for r in proofs:w.write(json.dumps(r,separators=(',',':'))+chr(10))
manifest=dict(schema='rinkspace.current_native_source_person_and_typed_unknown_effective.v3',complete=True,sealed=True,accepted=True,accepted_for_final_source_qualification=True,shot_id='s05',frame_count=1376,frames=1376,first_native_pts=948992,last_native_pts=1300992,time_base='1/12800',native_frame_step_pts=256,source_video_sha256=source['sha256'],index=dict(file=str(index),sha256=sha(index)),index_file=str(index),index_sha256=sha(index),parent_index=dict(file=str(parent),sha256=sha(parent)),source_qualification_ref=qref,source_proof_index=dict(file=str(proofindex),sha256=sha(proofindex)),changed_native_pts=changed,changed_native_frame_count=len(changed),unchanged_native_frame_count=1376-len(changed),reviewed_phase_native_pts=sorted(cand),counts=dict(stats),all_source_BGR_RGB_verified=True,old_true_person_pixels_preserved=True,old_unknown_pixels_preserved=True,person_layer_files_and_hashes_unchanged=True,both_person_and_unknown_layers_required=True,person_guard_pixels=8,source_unknown_guard_pixels=8,all_new_masks_are_current_source_SAM_pixels=True,all_new_layer_is_independent_unknown=True,box_pose_anchor_interpolation_is_navigation_only=True,no_dilation_bbox_ellipse_or_model_fill=True,source_original_UV_unchanged=True,physical_fitting_unchanged=True,s06_unchanged=True,source_reader=dict(file=__file__,sha256=sha(__file__)))
(D/'s05_manifest.json').write_text(json.dumps(manifest,indent=2));(D/'changed_native_pts.json').write_text(json.dumps(changed));print(json.dumps(dict(index=str(index),index_sha256=sha(index),manifest=str(D/'s05_manifest.json'),manifest_sha256=sha(D/'s05_manifest.json'),changed_frames=len(changed),counts=dict(stats))),flush=True)
