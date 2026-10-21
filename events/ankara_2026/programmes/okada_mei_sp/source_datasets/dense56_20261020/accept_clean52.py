import json,gzip,hashlib
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/clean_parent52_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[json.loads(l)for l in gzip.open(O/'index_candidate.jsonl.gz','rt')]
for r in rows:
 assert r['mask_sha256']==r['current_SAM']['sha256'];r['source_review_status']='all52_current_RGB_full_body_visual_reviewed_footwear_refinement_required';r['legacy_actor_pixels_ORed_into_person_mask']=False;r['legacy_graphics_or_rectangular_footpads_ORed_into_person_mask']=False;r['current_full_silhouette_search_prompt_only']=True
with gzip.open(O/'index.jsonl.gz','xt',encoding='utf8')as f:
 for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
review=dict(all52_actual_source_viewed=True,pages=[dict(file=str(p),sha256=sha(p))for p in sorted((O/'QA').glob('parent_*.jpg'))],observations=['legacy original scene masks contained fixed scoreboard/timer blocks, separate reflections and rectangular shoe pads; original bytes retained','fresh detector-only SAM missed horizontal free legs and skirts; rejected as full person base','new full-silhouette SAM uses actor component bounds solely as current RGB prompts, never hard exclusion or legacy pixel union','v2 restores observed free legs/skirt in current RGB; residual white shoe failures must be refined by source-supported footwear SAM before all-consumer release'],known_white_shoe_review=[997632,1007616,1044992,1063424,1071616,1080576,1119232,1167360,1177344,1188352,1223424,1235200,1280768,1300736,1311488,1339136,1347584,1368832],physical_fit_changed=False)
(O/'source_visual_review.json').write_text(json.dumps(review,indent=2),encoding='utf8')
m=json.load(open(O/'manifest_candidate.json',encoding='utf8'));m.update(accepted_as_current_person_base=True,accepted_full_footwear_mask=False,index=dict(file=str(O/'index.jsonl.gz'),sha256=sha(O/'index.jsonl.gz')),source_visual_review=dict(file=str(O/'source_visual_review.json'),sha256=sha(O/'source_visual_review.json')),legacy_pixels_ORed=False);(O/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8');print(json.dumps(m['index']))
