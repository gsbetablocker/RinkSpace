"""Strict source-only C branch overlay; original Query/UV objects stay unchanged.

Call validate_packet(..., allow_exact_source_prefix_successor=True) explicitly
when a later generation appends other sources. All C identities, UVs, native
PTS, materials, and full frozen Query objects must still match exactly.
"""
from pathlib import Path
import hashlib,json
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()
class SourceBranchRegistry:
 def __init__(self,path):
  self.path=Path(path);self.registry=json.loads(self.path.read_text(encoding='utf8'));r=self.registry
  if r['schema']!='rinkfit.C_source_branch_registry.v1':raise ValueError('unsupported source branch policy')
  self.rows={int(x['row_index']):x for x in r['alpha_domains']}
  if sorted(self.rows)!=list(range(910012,916340)):raise ValueError('incomplete C6328 branch denominator')
  if len({x['source_row_id'] for x in self.rows.values()})!=6328:raise ValueError('duplicate source branch identity')
  for x in self.rows.values():
   if x['status']!='source_identity_branch_qualified':raise ValueError('pending branch cannot enter production alpha domain')
   lo,hi=x['allowed_alpha'];a=x['source_measured_alpha']
   if not lo<hi or not lo<=a<=hi:raise ValueError('invalid own-flank branch')
  self.seeds={}
  for key,d in r['numerical_seed_overlay']['arrays'].items():
   if sha(d['file'])!=d['sha256']:raise ValueError('changed C numerical seed overlay '+key)
   self.seeds[key]=np.memmap(d['file'],dtype=d['dtype'],mode='r',shape=tuple(d['shape']))
 def validate_packet(self,packet_path,allow_exact_source_prefix_successor=False):
  p=json.loads(Path(packet_path).read_text(encoding='utf8'))
  if sha(packet_path)!=self.registry['base_packet']['sha256'] and not allow_exact_source_prefix_successor:raise ValueError('branch policy bound to a different packet')
  if p['row_count']<916340:raise ValueError('C source prefix truncated')
  query={int(x['row_index']):x for x in p['queries']};arrays={}
  for key in ['uv','native_pts','material_index','admission','fit_role','shot_index']:
   d=p['arrays'][key]
   if sha(d['file'])!=d['sha256']:raise ValueError('packet array hash mismatch '+key)
   arrays[key]=np.memmap(d['file'],dtype=d['dtype'],mode='r',offset=d.get('offset_bytes',0),shape=tuple(d['shape']))
  for i,x in self.rows.items():
   if p['source_row_ids'][i]!=x['source_row_id'] or int(arrays['native_pts'][i])!=x['native_pts'] or arrays['uv'][i].tolist()!=x['source_uv'] or int(arrays['material_index'][i])!=x['material_index'] or int(arrays['admission'][i])!=1 or int(arrays['fit_role'][i])!=1 or p['shot_ids'][int(arrays['shot_index'][i])]!='s09':raise ValueError('source identity/UV/native clock/material/admission changed '+str(i))
   if object_sha(query[i])!=x['original_full_Query_sha256']:raise ValueError('frozen acquisition Query changed '+str(i))
  return dict(C_rows=6328,all_source_identities_and_original_Query_exact=True,original_UV_or_Query_rewritten=False)
 def alpha_domain(self,row_index,source_row_id,original_bounds):
  x=self.rows.get(int(row_index))
  if x is None:return None
  if x['source_row_id']!=source_row_id:raise ValueError('C branch source ID mismatch')
  lo=max(float(original_bounds[0]),x['allowed_alpha'][0]);hi=min(float(original_bounds[1]),x['allowed_alpha'][1])
  if not lo<hi:raise ValueError('empty source material alpha domain')
  return lo,hi
 def numeric_seed(self,row_index,source_row_id):
  x=self.rows[int(row_index)]
  if x['source_row_id']!=source_row_id:raise ValueError('C seed source ID mismatch')
  j=int(row_index)-910012
  return {key:(int(a[j])if key=='path'else float(a[j]))for key,a in self.seeds.items()}
 def allows(self,row_index,source_row_id,alpha):
  domain=self.alpha_domain(row_index,source_row_id,[-float('inf'),float('inf')]);return bool(domain and domain[0]<=alpha<=domain[1])
