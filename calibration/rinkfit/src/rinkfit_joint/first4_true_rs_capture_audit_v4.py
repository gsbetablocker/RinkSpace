"""Strict complete-state consumer for finite-cell source-UV capture.

The true-RS/geometry calculation remains the frozen v1 implementation through
its frozen v3 domain wrapper. Cell state is independently checked as numerical
optimizer state; no new source qualification or physical calibration is issued.
"""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np
from .io_utils import file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import _quoted_fields

BASE_V3_SHA='ddce5bcd33f5eda6ac5657494608a6695c250e0db34b5afdbee50f91d27313b7'
OBJECTIVE='rinkfit.sourceUV_trueRS_finite_capture.v4'
POLICY='sourceUV_trueRS_capture_v4: strict sensor bracket; certified zoom hard domain; scene fixed; derived RS row; polyline station finite-cell active set with temporary vertex KKT; boundary-only pointwise merit rebind; original Query not imposed'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def _base():
 p=Path(__file__).with_name('first4_true_rs_capture_audit_v3.py')
 if file_sha256(p)!=BASE_V3_SHA:raise ValueError('Frozen v3 domain consumer changed')
 s=importlib.util.spec_from_file_location('rinkfit_joint._finite_cell_capture_v4_domain_base',p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.OBJECTIVE=OBJECTIVE;m.POLICY=POLICY
 return m

def _checked(r,root):
 p=Path(r['file']);p=p if p.is_absolute() else Path(root)/p
 if file_sha256(p)!=r['sha256']:raise ValueError('Changed cell state dependency: '+str(p))
 return p

def _arrays(manifest,root):
 out={};refs=[];meta={}
 for key,r in manifest['arrays'].items():
  p=_checked(r,root);name=p.name
  if name in meta:raise ValueError('Duplicate cell state basename')
  shape=tuple(r['shape']);dt=np.dtype(r['dtype'])
  if len(shape)!=1 or shape[0]<0 or p.stat().st_size!=int(np.prod(shape))*dt.itemsize:raise ValueError('Invalid cell array shape/size')
  meta[name]={k:r[k]for k in ('dtype','sha256','shape')}
  out[name]=np.memmap(p,mode='r',dtype=dt,shape=shape);refs.append(ref(p))
 return out,refs,meta

def _maps(cp):
 with Path(cp).open(encoding='utf8')as f:
  for _ in range(8):f.readline()
  count=int(f.readline())
  for _ in range(2*count):f.readline()
  result=[]
  for _ in range(2):
   result.append({})
   for _ in range(int(f.readline())):
    key,value=_quoted_fields(f.readline());result[-1][key]=value
 return result

def _contains_string(value,wanted):
 if isinstance(value,str):return value==wanted
 if isinstance(value,dict):return any(_contains_string(v,wanted)for v in value.values())
 if isinstance(value,list):return any(_contains_string(v,wanted)for v in value)
 return False

def verify_cells(packet,cpdir):
 pp=Path(packet).resolve();cpdir=Path(cpdir).resolve();p=read(pp);m=read(cpdir/'model.json')
 bp=cpdir/'source_uv_capture_station_cells_binding.json';b=read(bp)
 sb=read(cpdir/'source_uv_capture_state_binding.json');cpath=cpdir/'engineering_checkpoint.json'
 if b['schema']!='rinkfit.capture_polyline_cell_KKT_binding.v1' or b['objective_id']!=OBJECTIVE or b['state_token']!=sb['state_token'] or b['canonical_packet_sha256']!=file_sha256(pp):raise ValueError('Different cell/source/complete state identity')
 if b.get('complete_checkpoint_sha256') is not None and b['complete_checkpoint_sha256']!=file_sha256(cpath):raise ValueError('Cell binding complete CP differs')
 policy=_checked(b['policy_ref'],cpdir)
 pol=read(policy)
 if pol.get('schema')!='rinkfit.capture_polyline_cell_policy.v1' or pol.get('objective_id')!=OBJECTIVE or pol.get('RS_state_policy')!=POLICY or pol.get('canonical_packet_sha256')!=file_sha256(pp) or pol.get('temporary_numerical_active_set')is not True or pol.get('no_source_domain_or_sensor_domain_expansion')is not True or pol.get('formal_Query_and_original_W_consumed')is not False:raise ValueError('Different finite-cell policy')
 policy_deps=[]
 parent=pol['reference_complete_capture']
 for name in ('checkpoint','state_binding','nuisance_manifest'):
  fp=_checked(parent[name],policy.parent);policy_deps.append(ref(fp))
 pb=_checked(parent['state_binding'],policy.parent)
 if read(pb)['state_token']!=parent['state_token']:raise ValueError('Reference station primal token differs')
 nm=_checked(parent['nuisance_manifest'],policy.parent);nuisance=read(nm)
 if nuisance['packet_sha256']!=file_sha256(pp)or nuisance['raw_rows']!=p['row_count']:raise ValueError('Reference station source differs')
 for filename,record in nuisance['files'].items():
  fp=nm.parent/filename
  if fp.stat().st_size!=record['size_bytes']or file_sha256(fp)!=record['sha256']:raise ValueError('Changed reference auxiliary')
  policy_deps.append(ref(fp))
 gp=_checked(b['actual_graph_receipt'],cpdir);g=read(gp)
 if g['state_token']!=b['state_token'] or g['objective_id']!=OBJECTIVE:raise ValueError('Cell graph belongs to another complete state')
 A,ar,meta=_arrays(b,cpdir);G,gr,gm=_arrays(g,gp.parent)
 for k,v in gm.items():
  if k in meta:
   if meta[k]!=v:raise ValueError('Cell graph row arrays differ')
  else:meta[k]=v
 import hashlib
 digest=hashlib.sha256(json.dumps(meta,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
 discrete=_maps(cpath)[0]
 if discrete.get('station_cell_state_sha256')!=digest or discrete.get('station_cell_policy_sha256')!=file_sha256(policy):raise ValueError('Cell active-set state is not bound into complete CP FNV identity')
 expected={'row_index.u64':'<u8','cell_index.i32':'<i4','vertex_index.i32':'<i4','cell_lower.f64':'<f8','cell_upper.f64':'<f8','numeric_lower.f64':'<f8','numeric_upper.f64':'<f8','kkt_active.u8':'u1','left_exists.u8':'u1','right_exists.u8':'u1','left_status.u8':'u1','right_status.u8':'u1','gradient_left.f64':'<f8','gradient_right.f64':'<f8','station_at_binding.f64':'<f8','active_kind.u8':'u1','sensor_active.u8':'u1'}
 if set(A)!=set(expected):raise ValueError('Unexpected finite-cell array contract: '+str(set(A)^set(expected)))
 for k,dt in expected.items():
  if A[k].dtype!=np.dtype(dt):raise ValueError('Cell array dtype differs: '+k)
 ids=A['row_index.u64'];M=len(ids)
 if any(len(v)!=M for v in A.values())or (M and (np.any(ids[1:]<=ids[:-1])or ids[-1]>=p['row_count'])):raise ValueError('Incomplete/duplicate/unordered cell rows')
 needed={'actual_Ceres_constant.u8':'u1','actual_lower.f64':'<f8','actual_upper.f64':'<f8'}
 if set(G)-{'row_index.u64'}!=set(needed):raise ValueError('Unexpected actual Ceres cell receipt arrays')
 for k,dt in needed.items():
  if G[k].dtype!=np.dtype(dt)or len(G[k])!=M:raise ValueError('Actual Ceres cell array shape differs')
 if 'row_index.u64'in G and not np.array_equal(G['row_index.u64'],ids):raise ValueError('Actual graph source rows differ')
 seen=set();Q={k:array(p,k,pp.parent,seen)for k in ('admission','fit_role','kind','material_index')}
 N=p['row_count'];aux=cpdir/'explicit_nuisance';station=np.memmap(aux/'station.f64',mode='r',dtype='<f8',shape=(N,));paths=np.memmap(aux/'path.u32',mode='r',dtype='<u4',shape=(N,))
 active=(Q['admission']==1)&(Q['fit_role']==1)&(Q['kind']!=1)
 expected_ids=[];sizes={}
 for i in np.flatnonzero(active):
  path=m['materials'][int(Q['material_index'][i])]['paths'][int(paths[i])]
  size=len(path.get('points',[]))
  if path['kind']in(2,6,7)and size>1:expected_ids.append(int(i));sizes[int(i)]=size
 if not np.array_equal(ids,np.asarray(expected_ids,dtype='<u8')):raise ValueError('Cell state omits or invents actual active polyline source rows')
 if not np.array_equal(A['station_at_binding.f64'].view('<u8'),station[ids].view('<u8')):raise ValueError('Cell station differs bitwise from complete saved auxiliary')
 domains={int(d['row_index']):d for d in p.get('source_station_domains',{}).get('rows',[])}
 count={'free_cell':0,'knot_KKT':0,'finite_endpoint_KKT':0}
 for j,row in enumerate(ids):
  i=int(row);s=float(station[i]);path=m['materials'][int(Q['material_index'][i])]['paths'][int(paths[i])];n=sizes[i];intervals=path.get('allowed_station_intervals',[[0.,1.]])
  if i in domains:
   d=domains[i]
   if int(d['path_index'])!=int(paths[i]):raise ValueError('Cell source finite path differs')
   intervals=[[max(a[0],bb[0]),min(a[1],bb[1])]for a in intervals for bb in d['allowed_station_intervals']if max(a[0],bb[0])<=min(a[1],bb[1])]
  spans=[(float(lo),float(hi))for lo,hi in intervals if lo<=s<=hi]
  if not spans:raise ValueError('Cell state outside original finite source domain at row '+str(i))
  if int(A['sensor_active.u8'][j])!=0:raise ValueError('Polyline cell was not materialized at ordinary strict-RS point')
  flags=[int(A[k][j])for k in ('kkt_active.u8','left_exists.u8','right_exists.u8')]
  if any(x not in(0,1)for x in flags):raise ValueError('Invalid cell flag')
  left,right=int(A['left_status.u8'][j]),int(A['right_status.u8'][j])
  if left not in(0,1,2)or right not in(0,1,2)or flags[1:]!=[int(left!=0),int(right!=0)]:raise ValueError('Cell side existence/status differs')
  if left==2 or right==2:raise ValueError('Unresolved strict-RS side cannot masquerade as endpoint/KKT')
  if left==0 and any(a<s<=bb for a,bb in spans):raise ValueError('Existing finite left side was falsely declared absent')
  if right==0 and any(a<=s<bb for a,bb in spans):raise ValueError('Existing finite right side was falsely declared absent')
  kind=int(A['active_kind.u8'][j]);cell=int(A['cell_index.i32'][j]);vertex=int(A['vertex_index.i32'][j]);cl=float(A['cell_lower.f64'][j]);cu=float(A['cell_upper.f64'][j]);lo=float(A['numeric_lower.f64'][j]);hi=float(A['numeric_upper.f64'][j]);gl=float(A['gradient_left.f64'][j]);gg=float(A['gradient_right.f64'][j])
  if not np.isfinite([s,cl,cu,lo,hi,gl,gg]).all():raise ValueError('Nonfinite cell state')
  if kind==1:
   if flags[0]or vertex!=-1 or not 0<=cell<n-1 or not cl<lo<=s<=hi<cu or not any(cl==max(cell/(n-1),a)and cu==min((cell+1)/(n-1),bb)for a,bb in spans):raise ValueError('Free cell escapes its original finite piece at row '+str(i))
   count['free_cell']+=1
  elif kind in(2,3):
   if not flags[0]or cell!=-1 or not cl==cu==lo==hi==s or not(left or right)or(left and gl>0)or(right and gg<0):raise ValueError('Vertex/finite endpoint KKT is not valid at row '+str(i))
   if kind==2:
    if not 0<=vertex<n or s!=vertex/(n-1):raise ValueError('KKT is not the actual saved polyline vertex')
    count['knot_KKT']+=1
   else:
    if vertex!=-1 or not any(s==a or s==bb for a,bb in spans):raise ValueError('KKT endpoint is not an original finite boundary')
    count['finite_endpoint_KKT']+=1
  else:raise ValueError('Unknown numerical cell active kind')
  if int(G['actual_Ceres_constant.u8'][j])!=flags[0]or float(G['actual_lower.f64'][j])!=lo or float(G['actual_upper.f64'][j])!=hi:raise ValueError('Actual Ceres active cell constant/bounds differ')
 return dict(schema='rinkfit.independent_capture_finite_cell_state.v1',rows=M,counts=count,complete_CP_cell_state_digest=digest,all_saved_stations_bitwise_verified=True,all_original_finite_domains_and_cells_checked=True,all_actual_Ceres_constant_and_bounds_records_checked=True,KKT_half_squared_source_UV_derivative_record_signs_checked=True,KKT_derivatives_independently_reimplemented_for_all_rows=False,source_domain_expanded=False,cell_state_is_temporary_numerical_optimizer_state=True,binding=ref(bp),policy=ref(policy),actual_graph_receipt=ref(gp),dependencies=ar+gr+policy_deps+[ref(bp),ref(policy),ref(gp)])

def audit(packet_path,candidate,output,*,certified_domain,graph_domain_receipt,baseline_import=None,native_domain_receipt=None):
 cells=verify_cells(packet_path,candidate)
 base=_base();r=base.audit(packet_path,candidate,output,certified_domain=certified_domain,graph_domain_receipt=graph_domain_receipt,baseline_import=baseline_import,native_domain_receipt=native_domain_receipt)
 r.update(schema='rinkfit.first4_true_RS_capture_independent_audit.v4',independent_finite_cell_state=cells,producer=ref(__file__),unchanged_domain_consumer=ref(base.__file__))
 r['dependencies'].extend(cells['dependencies']+[ref(__file__),ref(base.__file__)])
 (Path(output)/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 return r
