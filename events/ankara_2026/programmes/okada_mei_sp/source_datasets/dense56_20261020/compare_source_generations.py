"""Compare actual source identities and unchanged UV; counts alone do not prove quality."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import zip_longest
import sys,json,gzip,hashlib,base64

def read(p):return json.load(open(p,encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def groups(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  batch=[];key=None
  for line in f:
   r=json.loads(line);k=(r['shot_id'],r['native_pts'])
   if key is not None and k!=key:yield key,batch;batch=[]
   key=k;batch.append(r)
  if key is not None:yield key,batch
class FrameCursor:
 def __init__(self,p):self.it=iter(groups(p));self.n=next(self.it,None)
 def get(self,k):
  if self.n and self.n[0]<k:raise ValueError('Unconsumed source frame')
  if self.n and self.n[0]==k:q=self.n[1];self.n=next(self.it,None);return q
  return []
def key(r):return (r['feature_id'],r.get('source_point_id')or tuple(r['uv']))
oldp,newp,out=map(Path,sys.argv[1:]);old,new=read(oldp),read(newp)
if out.exists():raise FileExistsError(out)
for m in(old,new):
 for name in('selected_rows.jsonl.gz','consumer_bindings.jsonl.gz'):
  ref=m['files'][name]
  if sha(ref['file'])!=ref['sha256']:raise ValueError('Source generation changed')
out.mkdir(parents=True);changes=gzip.open(out/'changed_overlay_rows.jsonl.gz','wt',encoding='utf8',compresslevel=4)
bind=FrameCursor(new['files']['consumer_bindings.jsonl.gz']['file']);excluded=FrameCursor(new['files']['excluded.jsonl.gz']['file'])
counts=Counter();byshot=defaultdict(Counter);reasons=Counter();type_changes=Counter();newnearframes=set();newnearrows=0;critical={p:dict(old=Counter(),new=Counter())for p in(1096192,1135104,1153792,1161216,1170944,1188352,1235200,1334016,1339648,1342208,1346048,1351680,1363456,1414912)}
for op,np in zip_longest(groups(old['files']['selected_rows.jsonl.gz']['file']),groups(new['files']['selected_rows.jsonl.gz']['file'])):
 if op is None or np is None or op[0]!=np[0]:raise ValueError('Source frame clocks differ')
 frame,a=op;b=np[1];sid,p=frame;aa={key(r):r for r in a};bb={key(r):r for r in b}
 if len(aa)!=len(a)or len(bb)!=len(b):raise ValueError('Ambiguous same-frame source identity')
 common=aa.keys()&bb.keys();removed=aa.keys()-bb.keys();added=bb.keys()-aa.keys()
 for k in common:
  if aa[k]['uv']!=bb[k]['uv']:raise ValueError('Original source UV moved: '+str(frame)+' '+str(k))
  oldtype=tuple(aa[k].get(z)for z in('observation_type_code','normal_only','texture_2D_tracking'));newtype=tuple(bb[k].get(z)for z in('observation_type_code','normal_only','texture_2D_tracking'))
  if oldtype!=newtype:type_changes[str(oldtype)+' -> '+str(newtype)]+=1
 counts.update(dict(frames=1,retained_rows=len(common),removed_old_display_rows=len(removed),added_new_display_rows=len(added)));byshot[sid].update(dict(retained_rows=len(common),removed_old_display_rows=len(removed),added_new_display_rows=len(added)))
 br=bind.get(frame)
 if len(br)!=1:raise ValueError('No unique actual consumer binding')
 ad=None;status=None;eligible=None
 ex={str(r['source_row_id']):r['reason']for r in excluded.get(frame)if r.get('source_row_id')is not None}
 for k in removed:
  r=aa[k];rid=str(r.get('source_point_id'));reason=ex.get(rid)
  if reason is None and r.get('source_qualification_status_code')is not None:
   if ad is None:
    ref=br[0]['advert'];ad=read(ref['file'])
    if sha(ref['file'])!=ref['sha256']:raise ValueError('Advert source qualification changed')
    status=base64.b64decode(ad['status_uint8_base64']);eligible=base64.b64decode(ad['eligible_uint8_base64'])
   i=r['source_pixel_or_annotation_ref']['qualification_row_0based']
   if not eligible[i]:reason='new_actual_advert_qualification_status_'+str(status[i])
  if reason is None:reason='display_selection_changed_within_existing_500_ceiling'
  reasons[reason]+=1;changes.write(json.dumps(dict(shot_id=sid,native_pts=p,change='removed_from_overlay',source_point_id=r.get('source_point_id'),feature_id=r['feature_id'],source_uv=r['uv'],old_selected_row_id=r['selected_row_id'],reason=reason),separators=(',',':'))+'\n')
 for k in added:
  r=bb[k];changes.write(json.dumps(dict(shot_id=sid,native_pts=p,change='added_to_overlay',source_point_id=r.get('source_point_id'),feature_id=r['feature_id'],source_branch_id=r.get('source_branch_id'),source_uv=r['uv'],new_selected_row_id=r['selected_row_id']),separators=(',',':'))+'\n')
 for r in b:
  if r.get('source_branch_id')=='near_lower_long_side':newnearframes.add(p);newnearrows+=1
 if p in critical:
  for label,rows in(('old',a),('new',b)):
   for r in rows:critical[p][label][r['feature_id'].split('.')[0]]+=1
changes.close()
report=dict(schema='rinkspace.actual_source_generation_change.v1',old_generation=dict(file=str(oldp),sha256=sha(oldp)),new_generation=dict(file=str(newp),sha256=sha(newp)),counts=dict(counts),by_shot={s:dict(v)for s,v in byshot.items()},removed_overlay_rows_by_actual_reason=dict(reasons),retained_source_UV_exactly_equal=True,retained_rows_with_actual_constraint_type_change=dict(type_changes),new_near_lower_long_side_upper_boundary=dict(visible_frames_with_selected_source_rows=len(newnearframes),selected_rows=newnearrows,first_pts=min(newnearframes)if newnearframes else None,last_pts=max(newnearframes)if newnearframes else None),critical_source_cases={p:{k:dict(v)for k,v in x.items()}for p,x in critical.items()},changes=dict(file=str(out/'changed_overlay_rows.jsonl.gz'),sha256=sha(out/'changed_overlay_rows.jsonl.gz')),same_total_count_does_not_certify_accuracy=True,physical_fit_or_live_packet_changed=False)
json.dump(report,open(out/'manifest.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
print(json.dumps(dict(counts=report['counts'],near=report['new_near_lower_long_side_upper_boundary'])))
