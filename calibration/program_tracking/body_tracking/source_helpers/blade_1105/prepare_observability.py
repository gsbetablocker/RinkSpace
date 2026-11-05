from pathlib import Path
import json,hashlib,subprocess,zipfile,io,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(r'D:/OneDrive/codex_projs/RinkSpace'); D=Path(r'D:/CodexData/RinkSpace_Data'); B=D/'runs/body_s01_20261029'; O=B/'blade_observability_1105'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pack=D/'handoff/1105/RinkSpace_0078c8db_Body_Offline_Delta.zip'
with zipfile.ZipFile(pack) as z:
 m=json.loads(z.read('MANIFEST.json'))
 parents=[]
 expected=np.arange(972800,1216000+1,256,dtype=np.int64)
 for item in m['files']:
  if item['path'] not in ['numerics/A_saved_review.npz','numerics/B_saved_review.npz','numerics/C6_saved_review.npz','numerics/local_281_saved_review.npz']:continue
  f=D/item['source']['data_relative'];h=sha(f)
  with np.load(f,allow_pickle=False) as nz:
   pts=nz['pts'];tb=nz['time_base']; assert np.array_equal(pts,expected) and np.array_equal(tb,[1,12800])
  blob=z.read(item['path'])
  with np.load(io.BytesIO(blob),allow_pickle=False) as nz:assert np.array_equal(nz['pts'],expected) and np.array_equal(nz['time_base'],tb)
  parents.append(dict(candidate=item['path'].split('/')[1].split('_saved')[0],path=str(f),bytes=f.stat().st_size,sha256=h,expected_sha256=item['source']['parent_sha256'],sha_matches=h==item['source']['parent_sha256'],packet_sha_matches=hashlib.sha256(blob).hexdigest()==item['sha256'],PTS_matches=True,time_base=tb.tolist()))
source=read(B/'spin_continuous_1105/source_review/seed_visual_review.json')
video=Path(source['source_path']);vh=sha(video)
assert vh==m['source_video_sha256'];assert all(x['sha_matches'] and x['packet_sha_matches'] for x in parents)
identity=dict(schema='rinkspace.blade-audit-identity.v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=P,text=True).strip(),baseline=m['fixed_public_commit'],source_video=dict(path=str(video),bytes=video.stat().st_size,sha256=vh,expected_sha256=m['source_video_sha256'],sha_matches=True),scope=dict(frames=951,time_base=[1,12800],first_last_pts=[972800,1216000],native_pts_step=256,first_last_seconds=[76,95],unit='native exposure; not duplicate display frames'),parents=parents,archive=dict(path=str(pack),manifest_sha256=hashlib.sha256(json.dumps(m).encode()).hexdigest(),manifest_hash_definition='JSON serialization identity, original member hash below'))
with zipfile.ZipFile(pack) as z:identity['archive']['manifest_original_sha256']=hashlib.sha256(z.read('MANIFEST.json')).hexdigest()
identity['dirty_tree_before']=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=P,text=True,encoding='utf-8').splitlines()
(O/'identity_audit.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Source paths come only from saved source reviews, never projected model locations.
docs=[(B/'spin_continuous_1105/source_evidence.json'),(B/'spin_continuation_1105/source_evidence_v4.json')]
frame_refs={}; observations=[]
for f in docs:
 a=read(f)
 for key in ['observations','source_evidence_rows','additional_direct_review_unknowns']:
  for r in a.get(key,[]):
   if 'pts' not in r:continue
   pt=int(r['pts']); footprint=r.get('image_footprint',{})
   s=r.get('source_frame') or footprint.get('source_frame'); sh=r.get('source_frame_sha256') or footprint.get('source_frame_sha256')
   if s:frame_refs.setdefault(pt,[]).append(dict(path=s,sha256=sh,document=str(f),row_id=r.get('id')))
   if key=='observations':observations.append(r)
# Dense selected exposures include unknowns; retain their source imagery too.
for r in read(B/'spin_continuous_1105/source_review/dense_frame_manifest.json')['frames']:
 frame_refs.setdefault(int(r['pts']),[]).append(dict(path=r['source_frame'],sha256=r['source_frame_sha256'],document='dense_frame_manifest',row_id=None))
with np.load(B/'spin_local_280_1105/observations.npz',allow_pickle=False) as z:
 obspts=z['pts'].copy(); uv=z['uv'].copy()
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
records=[];hashcache={};tiles=[]
for pt,refs in sorted(frame_refs.items()):
 valid=[]
 for r in refs:
  f=Path(r['path'])
  if not f.exists():continue
  if str(f) not in hashcache:hashcache[str(f)]=sha(f)
  r=dict(r,exists=True,actual_sha256=hashcache[str(f)],sha_matches=(not r['sha256']) or hashcache[str(f)]==r['sha256'])
  valid.append(r)
 if not valid:
  records.append(dict(pts=pt,source_seconds=pt/12800,source_frame_missing=True,references=refs));continue
 # Prefer an explicitly hash-verified native frame.
 chosen=next((r for r in valid if r['sha_matches']),valid[0]); f=Path(chosen['path'])
 im=Image.open(f).convert('RGB'); assert im.size==(1920,1080), (f,im.size)
 tile=Image.new('RGB',(1040,260),'white');draw=ImageDraw.Draw(tile)
 draw.text((8,4),f'PTS {pt} | {pt/12800:.2f}s | source pixels only',fill='black',font=font)
 tile.paste(im.resize((360,203)),(0,49)); draw.text((4,29),'full-frame context',fill='black',font=small)
 cropdefs=[]
 for foot in [0,1]:
  rows=[r for r in observations if r['pts']==pt and r['foot']==foot and r.get('kind')=='boot_marker_region' and r.get('feature')=='shoe_body']
  if rows: centre=np.asarray(rows[-1]['centre_uv'],float);basis='saved source shoe region'
  else:
   ix=np.flatnonzero(obspts==pt)
   centre=np.asarray(uv[ix[0],15+foot],float) if len(ix) else np.array([960.,540.]);basis='detector ankle ROI only'
  x,y=centre;box=[int(round(x-150)),int(round(y-110)),int(round(x+150)),int(round(y+110))]
  crop=im.crop(box).resize((330,220)); off=370+foot*335;tile.paste(crop,(off,38))
  draw.text((off+2,23),f'foot{foot} ROI proposal; NOT identity',fill='black',font=small)
  cropdefs.append(dict(foot_proposal=foot,basis=basis,native_box=box,does_not_establish_anatomy=True))
 records.append(dict(pts=pt,source_seconds=pt/12800,source_frame=str(f),source_frame_sha256=chosen['actual_sha256'],frame_references=valid,crops=cropdefs))
 tiles.append((pt,tile))
sheets=[]
# Keep early and late image-review tasks separate, each fresh agent sees at most four small source sheets.
split=(len(tiles)+1)//2
for label,part in [('early',tiles[:split]),('late',tiles[split:])]:
 for i in range(0,len(part),9):
  chunk=part[i:i+9];sheet=Image.new('RGB',(1040,260*len(chunk)),(238,238,238))
  path=O/f'dual_source_{label}_{i//9+1:02d}.png'
  for j,(pt,im) in enumerate(chunk):
   sheet.paste(im,(0,j*260))
   rec=next(r for r in records if r['pts']==pt);rec['contact_sheet']=str(path);rec['contact_sheet_row']=j
  sheet.save(path)
  sheets.append(dict(path=str(path),group=label,pts=[q[0] for q in chunk]))
manifest=dict(schema='rinkspace.blade-dual-source-contact.v1',time_base=[1,12800],review_frames=len(tiles),scope='saved direct source shoe/blade review PTS and dense selected flow PTS; not all951 exposures',no_model_projection=True,no_new_source_video_decode=True,ROI_warning='Foot labels are inherited ROI proposals only. Use source full-frame context to identify visible objects; no anatomical identity is proven by a detector crop.',frames=records,sheets=sheets,source_sha256=vh)
(O/'dual_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(identity_ok=True,video_bytes=video.stat().st_size,parents=[dict(candidate=x['candidate'],sha_matches=x['sha_matches']) for x in parents],contact_frames=len(tiles),contact_sheets=sheets,source_reference_hash_mismatches=[dict(pts=r['pts'],path=q['path']) for r in records for q in r.get('frame_references',[]) if not q['sha_matches']]),ensure_ascii=False))
