"""Exact original group-W assembly with finite-cell station columns.

No projection, root recovery, optimization, source changes or W elimination.
Physical coordinates are original scaled free coordinates; station coordinates
are unscaled deltaS. Every active source remains in the original W partition.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse,hashlib,json,sys,time
import numpy as np
from scipy import sparse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file','')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Changed immutable input: '+str(p))
 return p.resolve()
def array(d,base,allow_nonfinite=False):
 a=np.fromfile(verify(d,base),np.dtype(d['dtype'])).reshape(d['shape'])
 if a.dtype.kind=='f'and not allow_nonfinite and not np.isfinite(a).all():raise ValueError('Nonfinite array')
 return a
def write_array(p,a):
 a=np.ascontiguousarray(a);p.write_bytes(a.tobytes());return dict(file=p.name,dtype=a.dtype.str,shape=list(a.shape),sha256=sha(p))

@dataclass
class GroupWStation:
 model:dict
 authority:dict
 free:np.ndarray
 scale:np.ndarray
 parent_values:np.ndarray
 UV:sparse.csr_matrix
 T:sparse.csr_matrix
 W:sparse.csr_matrix
 rW:np.ndarray
 Jphysical:sparse.csc_matrix
 Jstation:sparse.csc_matrix
 original_rW:np.ndarray
 station_seed:np.ndarray
 requested:np.ndarray
 selected:np.ndarray
 reasons:np.ndarray
 S:np.ndarray
 cell_lower:np.ndarray
 cell_upper:np.ndarray
 station_lower:np.ndarray
 station_upper:np.ndarray
 sensor_rows:np.ndarray
 sensor_row0:np.ndarray
 sensor_Jphysical:sparse.csr_matrix
 sensor_Jstation:sparse.csr_matrix
 error:np.ndarray
 groups:list
 group_rows:np.ndarray
 group_weights:np.ndarray
 lower_physical:np.ndarray
 upper_physical:np.ndarray
 def evaluate(self,physical_scaled,deltaS,strict=True):
  x=np.asarray(physical_scaled,dtype=float);s=np.asarray(deltaS,dtype=float)
  if x.shape!=(len(self.free),)or s.shape!=(len(self.selected),)or not np.isfinite(x).all()or not np.isfinite(s).all():raise ValueError('Complete finite physical and S coordinates required')
  if np.any(x<self.lower_physical)or np.any(x>self.upper_physical):raise ValueError('Original physical box violated')
  if np.any(s<self.station_lower)or np.any(s>self.station_upper):raise ValueError('Original finite cell / declared station trust violated')
  sensor=self.sensor_row0+self.sensor_Jphysical@x+self.sensor_Jstation@s
  if strict and (np.any(sensor<0)or np.any(sensor>1079)):raise ValueError('Original affine sensor domain violated')
  r=self.rW+self.Jphysical@x
  if np.any(s):r=r+self.Jstation@s
  return dict(residual=r,half_cost=float(.5*r@r),sensor_row=sensor,
   sensor_within_original_domain=bool(np.all(sensor>=0)and np.all(sensor<=1079)),finite_cell_satisfied=True,
   original_W_rank=len(r),no_joint_station_optimality_or_physical_acceptance=True)
 def direct_original_group_residual(self,x,s):
  # Independent dense per-group evaluation in ORIGINAL member/rank order.
  e=self.error.copy();e+=(self.UV@x+self.T@s).reshape(-1,2);parts=[]
  for g in self.groups:
   rr=self.group_rows[g['row_offset']:g['row_offset']+g['members']];rank=g['rank'];w=self.group_weights[g['weight_offset']:g['weight_offset']+rank*2*len(rr)].reshape(rank,2*len(rr));parts.append(w@e[rr].reshape(-1))
  return np.concatenate(parts)

def assemble(linearization,transport,selection=None,station_trust=None,dependencies=None,station_seed=None):
 Q=Path(linearization).resolve();Tpath=Path(transport).resolve();q=read(Q);tr=read(Tpath);mp=verify(q['model']);model=read(mp);blocks=model['blocks'];sizes=np.array([len(b['values'])for b in blocks]);offset=np.r_[0,np.cumsum(sizes)]
 scale=np.concatenate([b['scales']for b in blocks]);parent=np.concatenate([b['values']for b in blocks]);lo=np.concatenate([b['lower']for b in blocks]);hi=np.concatenate([b['upper']for b in blocks]);fixed=np.zeros(len(parent),bool)
 for k,b in enumerate(blocks):
  if b.get('constant',False):fixed[offset[k]:offset[k+1]]=True
  for j in b.get('fixed_indices',[]):fixed[offset[k]+j]=True
 free=np.flatnonzero(~fixed);mapping=np.full(len(parent),-1,np.int64);mapping[free]=np.arange(len(free))
 if np.any(scale<=0)or not np.isfinite(np.r_[scale,parent,lo,hi]).all()or np.any(parent<lo)or np.any(parent>hi):raise ValueError('Original physical coordinate box')
 dep=Path(dependencies or Path(__file__).parent/'groupW_S_deps').resolve();sys.path.insert(0,str(dep))
 import fixed_entity_norm_constraints as strict
 import whole_W_qp_core as oldcore
 # This original validator reconstructs selected refinement and then compares
 # every byte of S/path/V/alpha with the transport's complete CP.
 entities=strict.load_fixed_entity_norms(Q,Tpath,offset,scale,free)
 cpP=verify(tr['checkpoint']);cp=read(cpP);SP=verify(tr['semantic']);sem=read(SP);PK=verify(tr['packet']);pk=read(PK)
 N=tr['raw_rows'];active=array(sem['active_source_mask'],SP.parent).astype(bool);kind=array(sem['arrays']['kind'],SP.parent);status=array(tr['arrays']['status'],Tpath.parent);vertex=array(tr['arrays']['vertex'],Tpath.parent).astype(bool)
 if tr['original_groups_including_independent']!=97140 or not np.array_equal(np.flatnonzero(~active),[25537]):raise ValueError('Original source/statistical scope')
 if selection is None:requested=array(q['source_rows'],Q.parent).astype(np.int64);selection_ref=dict(origin='same-state qbar requested source rows',descriptor=q['source_rows'])
 else:
  selP=Path(selection).resolve();sel=read(selP)
  if sel.get('schema')!='rinkfit.station_column_selection.v1':raise ValueError('Station selection schema')
  for key,actual in [('transport',ref(Tpath)),('checkpoint',ref(cpP))]:
   verify(sel[key]);
   if sel[key]['sha256']!=actual['sha256']:raise ValueError('Stale station selection '+key)
  requested=array(sel['source_rows'],selP.parent).astype(np.int64);selection_ref=ref(selP)
 if requested.ndim!=1 or np.any(requested<0)or np.any(requested>=N)or len(np.unique(requested))!=len(requested)or np.any(np.diff(requested)<=0):raise ValueError('Sorted unique raw station selection')
 S=array(cp['row_state_files']['station'],cpP.parent);branch=array(cp['row_state_files']['branch'],cpP.parent);material=array(sem['resolved_material_index'],SP.parent)
 smooth=active&(kind==0)&(status==0)&~vertex&(S>1e-10)&(S<1-1e-10);reasons=np.full(len(requested),'released_smooth_ordinary',dtype='U48')
 for j,row in enumerate(requested):
  if not active[row]:reasons[j]='excluded_source';smooth[row]=False
  elif kind[row]!=0:reasons[j]='entity_or_original_profile'
  elif status[row]!=0:reasons[j]='transport_derivative_unverified'
  elif vertex[row]:reasons[j]='polyline_vertex'
  elif not(1e-10<S[row]<1-1e-10):reasons[j]='finite_endpoint'
  else:
   path=model['materials'][int(material[row])]['paths'][int(branch[row])]
   if path['kind']==2 and abs(S[row]*(len(path['points'])-1)-round(S[row]*(len(path['points'])-1)))<=1e-7:reasons[j]='polyline_near_vertex';smooth[row]=False
 selected=requested[smooth[requested]];all_cell_lo=array(tr['arrays']['station_delta_lower_right'],Tpath.parent);all_cell_hi=array(tr['arrays']['station_delta_upper_right'],Tpath.parent);cell_lo=all_cell_lo[selected];cell_hi=all_cell_hi[selected]
 if np.any(cell_lo>0)or np.any(cell_hi<0)or np.any(S[selected]+cell_lo<0)or np.any(S[selected]+cell_hi>1):raise ValueError('Original finite support/cell bounds')
 radius=np.inf if station_trust is None else float(station_trust)
 if np.isnan(radius)or radius<0:raise ValueError('Nonnegative explicit station trust')
 seed=np.zeros(N,dtype='<f8');seed_ref=None
 if station_seed is not None:
  seedP=Path(station_seed).resolve();sd=read(seedP)
  if sd.get('schema')!='rinkfit.station_seed_delta.v1':raise ValueError('Station seed schema')
  for key,actual in [('transport',ref(Tpath)),('checkpoint',ref(cpP))]:
   verify(sd[key]);
   if sd[key]['sha256']!=actual['sha256']:raise ValueError('Stale station seed '+key)
  seed=array(sd['station_delta'],seedP.parent)
  if seed.shape!=(N,)or seed.dtype!=np.dtype('<f8'):raise ValueError('Full raw-N original-unit station seed required')
  # The seed may retain any lawful smooth ordinary station, not only the
  # columns released by this secondary. It never changes entity/profile S.
  seed_smooth=active&(kind==0)&(status==0)&~vertex&(S>1e-10)&(S<1-1e-10)
  for row in np.flatnonzero(seed_smooth&(seed!=0)):
   path=model['materials'][int(material[row])]['paths'][int(branch[row])]
   if path['kind']==2 and abs(S[row]*(len(path['points'])-1)-round(S[row]*(len(path['points'])-1)))<=1e-7:seed_smooth[row]=False
  if np.any(seed[~seed_smooth]!=0):raise ValueError('Station seed attempts to move an entity/profile/vertex/unverified source')
  changed=seed!=0
  if np.any(seed[changed]<all_cell_lo[changed])or np.any(seed[changed]>all_cell_hi[changed])or np.any(abs(seed[changed])>radius):raise ValueError('Station seed leaves original finite cell / declared trust')
  seed_ref=ref(seedP)
 slo=np.maximum(cell_lo,-radius)-seed[selected];shi=np.minimum(cell_hi,radius)-seed[selected]
 def csr(desc,rows,base=Q.parent):
  ptr=array(desc['row_offsets'],base).astype(np.int64);bi=array(desc['block_id'],base).astype(np.int64);co=array(desc['component'],base).astype(np.int64);v=array(desc['values'],base)
  if len(ptr)!=rows+1 or ptr[0]!=0 or ptr[-1]!=len(v)or np.any(np.diff(ptr)<0)or len(bi)!=len(v)or len(co)!=len(v)or np.any(bi<0)or np.any(bi>=len(blocks))or np.any(co<0)or np.any(co>=sizes[bi]):raise ValueError('Physical UV CSR layout')
  amb=offset[bi]+co;keep=~fixed[amb];ri=np.repeat(np.arange(rows),np.diff(ptr));return sparse.csr_matrix((v[keep]*scale[amb[keep]],(ri[keep],mapping[amb[keep]])),shape=(rows,len(free)))
 UV=csr(tr['physical_UV_CSR'],2*N,Tpath.parent);tan=array(tr['arrays']['station_UV_right'],Tpath.parent).reshape(N,2)
 if np.any(np.linalg.norm(tan[selected],axis=1)==0):raise ValueError('Released S has zero material tangent')
 trows=np.ravel(np.c_[2*selected,2*selected+1]);tcols=np.repeat(np.arange(len(selected)),2);Ts=sparse.csr_matrix((tan[selected].ravel(),(trows,tcols)),shape=(2*N,len(selected)));Ts.eliminate_zeros()
 group_rows=array(tr['arrays']['group_rows'],Tpath.parent).astype(np.int64);weights=array(tr['arrays']['group_W'],Tpath.parent);ri=[];ci=[];vv=[];rank=0;partition=np.zeros(N,np.uint32)
 for g in tr['groups']:
  rr=group_rows[g['row_offset']:g['row_offset']+g['members']];r=g['rank'];w=weights[g['weight_offset']:g['weight_offset']+r*2*len(rr)].reshape(r,2*len(rr));ir,ic=np.nonzero(w);ri.append(rank+ir);ci.append(2*rr[ic//2]+ic%2);vv.append(w[ir,ic]);rank+=r;partition[rr]+=1
 if len(tr['groups'])!=97140 or rank!=214112 or not np.array_equal(partition,active.astype(partition.dtype)):raise ValueError('Original W group/rank/once-only membership')
 W=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(ri),np.concatenate(ci))),shape=(rank,2*N));original_rW=array(tr['arrays']['W_residual'],Tpath.parent);Jp=(W@UV).tocsc();Js=(W@Ts).tocsc()
 error=np.fromfile(verify(tr['arrays']['error_uv'],Tpath.parent),np.dtype(tr['arrays']['error_uv']['dtype'])).reshape(N,2);error[~active]=0
 seed_UV=tan*seed[:,None];error+=seed_UV;row0=array(tr['arrays']['true_rs_row'],Tpath.parent,allow_nonfinite=True)+seed_UV[:,1];arows=np.flatnonzero(active);sensor_rows=2*arows+1
 if not np.isfinite(error[active]).all()or not np.isfinite(row0[active]).all():raise ValueError('Active full-source UV/row arithmetic')
 rW=original_rW.copy()if not np.any(seed)else original_rW+W@seed_UV.ravel()
 # Preserve the previous physical-only operation as an independent exact
 # sparse-assembly reference. It uses the original same W and coordinates.
 old_t,old_J,old_r,old_info=oldcore.load_W_transport(Tpath,dict(model_path=mp,doc=q,report=dict(linearization=ref(Q)),csr=csr,n=len(free)))
 if not(Jp.shape==old_J.shape and np.array_equal(Jp.indptr,old_J.indptr)and np.array_equal(Jp.indices,old_J.indices)and Jp.data.tobytes()==old_J.data.tobytes()and original_rW.tobytes()==old_r.tobytes()):raise ValueError('Physical-only original operator changed')
 auth=dict(linearization=ref(Q),transport=ref(Tpath),checkpoint=ref(cpP),model=tr['model'],packet=tr['packet'],semantic=tr['semantic'],complete_auxiliary_sha256={k:cp['row_state_files'][k]['sha256']for k in ['station','branch','row','alpha']},selection=selection_ref,station_seed=seed_ref,station_TR=None if station_trust is None else radius,physical_only_reference=old_info,physical_only_operator_bitwise_identical=True,strict_same_state_metadata=entities.metadata)
 return GroupWStation(model,auth,free,scale,parent,UV,Ts,W,rW,Jp,Js,original_rW,seed,requested,selected,reasons,S,cell_lo,cell_hi,slo,shi,arows,row0[arows],UV[sensor_rows].tocsr(),Ts[sensor_rows].tocsr(),error,tr['groups'],group_rows,weights,(lo[free]-parent[free])/scale[free],(hi[free]-parent[free])/scale[free])

def save(chart,output):
 O=Path(output).resolve();O.mkdir(exist_ok=False);a={}
 for key,value in [('rW',chart.rW),('original_rW_at_zero_seed',chart.original_rW),('station_seed_delta_full',chart.station_seed),('physical_free_ambient_indices',chart.free.astype('<u8')),('physical_scales',chart.scale),('parent_physical_values',chart.parent_values),('requested_source_rows',chart.requested.astype('<u8')),('station_source_rows',chart.selected.astype('<u8')),('station0',chart.S[chart.selected]),('station_cell_lower',chart.cell_lower),('station_cell_upper',chart.cell_upper),('station_lower',chart.station_lower),('station_upper',chart.station_upper),('sensor_source_rows',chart.sensor_rows.astype('<u8')),('sensor_row0',chart.sensor_row0),('sensor_lower',-chart.sensor_row0),('sensor_upper',1079-chart.sensor_row0),('physical_lower',chart.lower_physical),('physical_upper',chart.upper_physical)]:a[key]=write_array(O/(key+'.bin'),value)
 for name,value in [('Jphysical',chart.Jphysical),('Jstation',chart.Jstation),('sensor_Jphysical',chart.sensor_Jphysical),('sensor_Jstation',chart.sensor_Jstation)]:sparse.save_npz(O/(name+'.npz'),value);a[name]=dict(**ref(O/(name+'.npz')),shape=list(value.shape),nnz=int(value.nnz),format=value.format)
 reasons=[dict(source_row=int(row),action=str(reason))for row,reason in zip(chart.requested,chart.reasons)if reason!='released_smooth_ordinary'];(O/'nonreleased.json').write_text(json.dumps(reasons,indent=2),encoding='utf8')
 d=dict(schema='rinkfit.original_group_W_finite_station_coupling.v1',authority=chart.authority,arrays=a,nonreleased=ref(O/'nonreleased.json'),requested_station_rows=len(chart.requested),released_station_rows=len(chart.selected),raw_rows=182017,active_rows=182016,groups_including_independent=97140,original_W_rank=214112,physical_columns=len(chart.free),station_column_unit='eta in original unscaled finite material station units; total deltaS=station_seed_delta_full+scatter(eta)',physical_column_unit='delta_theta[free] / original_scale[free]',residual='rW_at_seed + Jphysical @ delta_physical_scaled + Jstation @ eta',sensor_domain='0 <= sensor_row0_at_seed + sensor_Jphysical @ delta_physical_scaled + sensor_Jstation @ eta <= 1079',station_domain='station_lower <= eta <= station_upper; original total-deltaS finitecell bounds separately preserved',physical_only_deltaS_zero_bitwise_reference=True,profile_S_V_alpha_original_implicit_operator_not_released=True,entity_S_fixed=True,unselected_station_seed_preserved=True,unselected_station_fixed_only_to_its_seed_in_this_numerical_proposal=True,original_W_E_Q_physics_and_source_unchanged=True,not_a_joint_W_inner_optimum_or_nonlinear_acceptance_or_KKT=True,source=ref(__file__),dependencies=[ref(Path(__file__).parent/'groupW_S_deps'/x)for x in ['fixed_entity_norm_constraints.py','whole_W_qp_core.py']])
 (O/'manifest.json').write_text(json.dumps(d,indent=2),encoding='utf8');return ref(O/'manifest.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--linearization',required=True);p.add_argument('--transport',required=True);p.add_argument('--selection');p.add_argument('--station-trust',type=float);p.add_argument('--station-seed');p.add_argument('--output',required=True);args=p.parse_args();start=time.monotonic();c=assemble(args.linearization,args.transport,args.selection,args.station_trust,station_seed=args.station_seed);r=save(c,args.output);print(json.dumps(dict(manifest=r,released=len(c.selected),elapsed_seconds=time.monotonic()-start)))
