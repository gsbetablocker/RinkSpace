"""Rebase only new Ad7 text aliases into the actual preserved S01 print chart.

This is an approximate chart initializer obtained from named original vertices,
not a world calibration or new observation.  The frozen v1 compiler is retained.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np
from .io_utils import read_json,write_json,file_sha256

ART_SHA='6230fb34a20def98aef6e3e7cfaedb72e691e526257d1731135e11144aa31717'
S01_SHA='d78b7012c80664131f88134be0f9f868c1d6742ad7516155fc131b72ca3110e9'
NAMES=('logo_part00','logo_part01','logo_part04','logo_part05')
def _bytes_sha(q):
 return hashlib.sha256(np.asarray(q,dtype='<f8').tobytes()).hexdigest()
def _paths(model):
 return {p['id']:p for m in model['materials']for p in m['paths']if p['id'].startswith('w115/')}
def _fit(a,b):
 x,y=a.T;u,v=b.T;n=len(a);A=np.zeros((2*n,8));z=b.reshape(-1)
 A[::2,:3]=np.c_[x,y,np.ones(n)];A[1::2,3:6]=np.c_[x,y,np.ones(n)]
 A[::2,6:]=-np.c_[u*x,u*y];A[1::2,6:]=-np.c_[v*x,v*y]
 h,_,rank,s=np.linalg.lstsq(A,z,rcond=None)
 if rank!=8:raise ValueError('Actual original Ad7 chart correspondences are rank deficient')
 return np.r_[h,1.].reshape(3,3),float(s[0]/s[-1])
def _apply(a,H):
 z=np.c_[np.asarray(a,float),np.ones(len(a))]@H.T
 if not np.isfinite(z).all()or np.any(z[:,2]==0):raise ValueError('Chart initializer leaves its projective domain')
 return z[:,:2]/z[:,2:]
def rebase(model,project):
 project=Path(project);ap=project/'events/ankara_2026/models/ad7_shape_1016.json'
 if file_sha256(ap)!=ART_SHA:raise ValueError('Official Ad7 source chart SHA changed')
 sr=model['source_s01_model_ref'];sp=Path(sr if isinstance(sr,str)else sr.get('path',sr.get('file','')))
 declared=model.get('s01_model_sha256')if isinstance(sr,str)else sr.get('sha256')
 if declared!=S01_SHA or file_sha256(sp)!=S01_SHA:raise ValueError('Actual preserved S01 model identity changed')
 original=read_json(sp);art=read_json(ap);actual=_paths(model);frozen=_paths(original);pairs=[]
 for name in NAMES:
  key='w115/'+name;p=actual[key];q=frozen[key]
  qc=copy.deepcopy(q)
  for field in ('geometry_block','advertisement_block'):
   b=model['blocks'][p[field]];f=original['blocks'][q[field]]
   if b['name']!=f['name']or b['values']!=f['values']:raise ValueError('Preserved actual S01 chart block changed')
   qc[field]=p[field]
  if p!=qc:raise ValueError('Preserved actual S01 path changed: '+key)
  if p['kind']!=2 or p['geometry_block']!=92 or p['advertisement_block']!=12:raise ValueError('Unexpected actual Ad7 layout')
  a=np.asarray(art['contours'][name]['material_uv'],float);b=np.asarray(p['points'],float)
  if a.shape!=b.shape:raise ValueError('Original named vertex correspondence changed')
  pairs.append((name,a,b))
 A=np.concatenate([x[1]for x in pairs]);C=np.concatenate([x[2]for x in pairs]);H,cond=_fit(A,C);fit_error=float(np.max(abs(_apply(A,H)-C)));loo=[]
 for name,a,b in pairs:
  ht,_=_fit(np.concatenate([x[1]for x in pairs if x[0]!=name]),np.concatenate([x[2]for x in pairs if x[0]!=name]))
  loo.append(dict(left_out_original_path='w115/'+name,vertices=len(a),maximum_coordinate_error=float(np.max(abs(_apply(a,ht)-b)))))
 result=copy.deepcopy(model);changes=[]
 if 'new_Ad7_alias_chart_initializer_v2'in model:raise ValueError('Already chart-rebased: use its immutable output without applying twice')
 for i,m in enumerate(result['materials']):
  if not(m['label'].startswith('Ad7.print.r2_INTERNATIONAL_')or m['label'].startswith('Ad7.print.r2_SKATING_UNION_')):continue
  if len(m['paths'])!=1:raise ValueError('New text alias no longer has one original finite path')
  p=m['paths'][0];pr=json.loads(p['provenance']);key=pr['official_subpath'];definition=art['contours'][key];old=np.asarray(p['points'],float)
  if p['geometry_block']!=92 or p['advertisement_block']!=12 or not np.array_equal(old,np.asarray(definition['material_uv'])):raise ValueError('New alias is not the original raw-chart initializer: '+m['label'])
  new=_apply(old,H);p['points']=new.tolist()
  changes.append(dict(material_index=i,label=m['label'],path_id=p['id'],official_subpath=key,vertices=len(old),old_points_binary64_sha256=_bytes_sha(old),new_points_binary64_sha256=_bytes_sha(new),only_points_changed=True,closed_unchanged=p.get('closed'),geometry_block=92,advertisement_block=12))
 if not changes:raise ValueError('No visible qualified Ad7 text aliases to rebase')
 report=dict(schema='rinkfit.actual_S01_chart_new_Ad7_text_initializer.v2',status='approximate_chart_initialization_not_world_calibration',official_chart=dict(path=str(ap),sha256=ART_SHA),actual_S01_chart=dict(path=str(sp),sha256=S01_SHA),basis='four full original named w115 paths and corresponding original vertex indices; no source UV, station, angle or camera fitting',corresponding_vertices=len(A),source_vertex_binary64_sha256=_bytes_sha(A),actual_S01_vertex_binary64_sha256=_bytes_sha(C),H=H.tolist(),linear_rank=8,linear_condition=cond,maximum_coordinate_fit_error=fit_error,leave_one_original_path_out=loo,changes=changes,new_physical_parameters=0,old_materials_paths_and_all_parameters_unchanged=True,new_text_is_numeric_initializer_not_known_world_anchor=True,Ad8_original_nonlinear_transfer_untouched=True)
 for i,(old,new)in enumerate(zip(model['materials'],result['materials'])):
  if i not in {x['material_index']for x in changes}and old!=new:raise AssertionError('Unrelated material changed')
 if model['blocks']!=result['blocks']:raise AssertionError('Any physical parameter changed')
 result['new_Ad7_alias_chart_initializer_v2']=report
 return result,report

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('A new immutable chart candidate directory is required')
 model=read_json(a.model);candidate,report=rebase(model,a.project);out.mkdir(parents=True);write_json(out/'model.json',candidate);report['input_model']={'path':a.model,'sha256':file_sha256(a.model)};report['candidate_model']={'path':str(out/'model.json'),'sha256':file_sha256(out/'model.json')};report['consumer']={'path':str(Path(__file__)),'sha256':file_sha256(__file__)};write_json(out/'report.json',report);print(json.dumps(dict(changed_aliases=len(report['changes']),candidate=report['candidate_model'],maximum_coordinate_fit_error=report['maximum_coordinate_fit_error'])))
if __name__=='__main__':main()
