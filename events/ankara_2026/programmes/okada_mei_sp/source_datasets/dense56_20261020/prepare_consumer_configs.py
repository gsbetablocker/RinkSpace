from pathlib import Path
import hashlib,json
D=Path(r'D:/CodexData/RinkSpace_Data');R=D/'runs/okada_rec_1019/recover_v2';O=R/'dense56_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p);assert p.is_file(),p;return dict(file=str(p),sha256=sha(p))
def read(p):return json.load(open(p,encoding='utf8'))
source=read(D/'runs/okada_joint9_1018a/programme_scope.json')['source'];source=dict(file=source['path'],sha256=source['sha256'],native_fps='50/1',time_base='1/12800',native_frame_step_pts=256,image_size_hw=[1080,1920]);assert sha(source['file'])==source['sha256']
bm=R/'dense56_board_v1/material_v5/qualified_manifest.json';b=read(bm);assert b['complete']and b['quality_accepted_for_raw_point_overlay']
am=R/'dense56_print_v1/final_manifest.json';a=read(am);assert a['complete']
mm=R/'dense56_footwear_v1/manifest.json';m=read(mm);assert m['accepted_for_final_source_qualification'];mi=ref(m['index_file']);assert mi['sha256']==m['index_sha256']
ice={};extra=[ref(R/'dense56_ice_v1/sealed_handoff.json'),ref(Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src/rinkfit_joint/dense_source_review.py')),ref(R/'dense56_footwear_v1/source_UV_disposition_receipt.json'),ref(R/'dense56_board_v1/material_v5/READY.json'),ref(R/'dense56_footwear_v1/typed_reason_clarification.json'),ref(O/'clean52_review_addendum.json'),ref(mm),ref(R/'dense56_board_v1/material_v5/material_uv_freeze.json'),ref(O/'historical_guidance_registered.json'),ref(R/'dense56_footwear_v1/clean_parent52_v2/source_visual_review.json')]
for s,n in [('s05',1376),('s06',486)]:
 base=R/'dense56_ice_v1/canonical'/s;mf=base/'manifest.json';j=read(mf);assert j['complete']and j['sealed'];ice[s]=dict(manifest=ref(mf),rows=[ref(base/'rows.jsonl.gz')],coverage=ref(base/'coverage.jsonl.gz'));extra.append(ref(base/'exclusions.jsonl.gz'))
base=dict(purpose='user_requested_source_points_review',board_generation='material_v5',source=source,board_manifest=ref(bm),advert_manifest=ref(am),mask_index=mi,ice_shots=ice,ice_exclusions=[],additional_parent_pins=extra)
for sid in ['s05','s06']:
 cfg=dict(base,shot_ids=[sid],output=str(O/'parts'/sid));p=O/('consumer_'+sid+'.json');assert not p.exists(),p;p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(complete=True,configurations=[str(O/('consumer_'+s+'.json'))for s in ['s05','s06']],mask_index=mi),ensure_ascii=False))
