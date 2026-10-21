"""Strict exact-ID consumer for current source-uncertain exclusions.
A red candidate PNG alone NEVER authorizes exclusion. Only listed rows do.
No point coordinates are changed and no replacement/refill is permitted.
"""
import gzip,json,hashlib
from pathlib import Path
class SourceUncertainExclusions:
 def __init__(self,manifest_path):
  self.manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
  if self.manifest['review_status']!='qualified_source_uncertain_exclusions_visual_QA_complete':raise ValueError('Qualification not released')
  ref=self.manifest['exclusions'];h=hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest()
  if h!=ref['sha256']:raise ValueError('Exclusion SHA mismatch')
  self.rows={}
  with gzip.open(ref['file'],'rt',encoding='utf8') as f:
   for s in f:
    r=json.loads(s);key=r['source_row_id']
    if key in self.rows:raise ValueError('Duplicate exact source ID '+key)
    self.rows[key]=r
 def exclusion(self,row):
  key=row.get('source_point_id') or row.get('source_row_id');ex=self.rows.get(key)
  if ex is None:return None
  assert row['native_pts']==ex['native_pts'] and row['feature_id']==ex['feature_id'] and row['uv']==ex['uv'],('Different source measurement for same ID',key)
  return ex
 def keep(self,row):return self.exclusion(row) is None
