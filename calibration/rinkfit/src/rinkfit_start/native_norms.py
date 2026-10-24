"""Strict selected-scope native norm readers over a frozen original source.

Event compatibility headers provide immutable raw packet/statistical anchors.
Only the selected semantic active mask determines active entity/profile counts.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json
import numpy as np
from scipy import sparse

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file')))
 if not p.is_absolute():p=Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Immutable descriptor changed: '+str(p))
 return p
def array(d,base,mmap=False):
 p=verify(d,base);dt=np.dtype(d['dtype']);shape=tuple(d['shape'])
 if p.stat().st_size!=int(np.prod(shape))*dt.itemsize:raise ValueError('Array byte count')
 return np.memmap(p,mode='r',dtype=dt,shape=shape)if mmap else np.fromfile(p,dt).reshape(shape)

@dataclass
class FixedEntityNorms:
 rows:np.ndarray
 residual:np.ndarray
 J:sparse.csr_matrix
 metadata:dict
 def values(self,physical_scaled_step):
  x=np.asarray(physical_scaled_step);assert x.shape==(self.J.shape[1],)
  r=self.residual+(self.J@x).reshape(-1,2)
  if not np.isfinite(r).all():raise ValueError('Nonfinite entity affine residual')
  return np.einsum('ij,ij->i',r,r),r
 def primary_SOC4(self):
  """s=(b+1,2r_U,2r_V,b-1) in SOC4 iff ||r||²<=b."""
  k=len(self.rows);n=self.J.shape[1];coo=self.J.tocoo()
  rr=(coo.row//2)*4+1+(coo.row%2)
  ep=np.r_[4*np.arange(k),4*np.arange(k)+3]
  A=sparse.csr_matrix((np.r_[-2*coo.data,-np.ones(2*k)],
      (np.r_[rr,ep],np.r_[coo.col,np.full(2*k,n)])),shape=(4*k,n+1))
  b=np.zeros(4*k);b[::4]=1.;b[3::4]=-1.;b[1::4]=2*self.residual[:,0];b[2::4]=2*self.residual[:,1]
  return A,b
 def secondary_SOC3(self,primary_physical_step,trust,b_fixed):
  """For a centered WQP, delta=base+trust*d, all entity norms<=same b."""
  if not np.isfinite(b_fixed)or b_fixed<0 or not np.isfinite(trust)or trust<=0:raise ValueError('Finite b/trust')
  _,r=self.values(primary_physical_step);k=len(self.rows);coo=self.J.tocoo()
  rr=(coo.row//2)*3+1+(coo.row%2)
  A=sparse.csr_matrix((-trust*coo.data,(rr,coo.col)),shape=(3*k,self.J.shape[1]))
  b=np.zeros(3*k);b[::3]=np.sqrt(b_fixed);b[1::3]=r[:,0];b[2::3]=r[:,1]
  return A,b
 def supporting_cuts(self,physical_scaled_step,entity_indices=None):
  """Fenchel inequality 2y·(e+Jx)-||y||²<=b, valid for any y.

  Caller must still audit every entity norm and may not assume a finite set
  of cuts gives norm feasibility. A WQP also needs these cones/all-row audit.
  """
  _,y=self.values(physical_scaled_step)
  ids=np.arange(len(self.rows))if entity_indices is None else np.asarray(entity_indices,dtype=int)
  U,V=self.J[2*ids],self.J[2*ids+1]
  G=2*(U.multiply(y[ids,0,None])+V.multiply(y[ids,1,None]))
  C=sparse.hstack([G,-np.ones((len(ids),1))],format='csr')
  rhs=np.einsum('ij,ij->i',y[ids],y[ids])-2*np.einsum('ij,ij->i',y[ids],self.residual[ids])
  return C,rhs,dict(source_rows=self.rows[ids].tolist(),tangent_y=y[ids].tolist(),
    valid_global_affine_quadratic_support=True,finite_cut_set_is_not_norm_feasibility=True)
 def audit(self,physical_scaled_step,b):
  q,_=self.values(physical_scaled_step);worst=int(np.argmax(q-b))
  return dict(entities=len(self.rows),maximum_affine_q=float(q.max()),
    maximum_affine_q_minus_b=float(np.max(q-b)),worst_source_row=int(self.rows[worst]),
    violating_entity_count=int(np.count_nonzero(q>b)),all_entity_affine_norms_within_b=bool(np.all(q<=b)),
    nonlinear_physical_acceptance=False)



def _validate_current_semantic_scope(t, sem, pk, semantic_base):
 """Consume the current prepared input without replaying historical assets.

 Budget membership and statistical membership are distinct. This only reads
 their current manifests; the original norm and W algebra is unchanged.
 """
 N=int(pk['observations']);declared=pk['current_input_scope']
 if N<0 or sem.get('observations')!=N or t.get('raw_rows')!=N:
  raise ValueError('Current packet/semantic/transport row counts differ')
 if sem.get('current_input_scope')!=declared:
  raise ValueError('Current packet and semantic scope differ')
 source=pk['immutable_source_hashes']['observations.npz']
 for document in (pk,sem):
  export=document.get('current_prepared_export',{})
  # Early complete-current exports already carry this identity in the packet
  # source hash and in the explicit semantic export. Accept that same current
  # declaration without rewriting completed qbar/GPU numerical operations.
  if document is pk and not export and declared.get('full_physical_solver',False):
   export=sem.get('current_prepared_export',{})
  if export.get('source_identity')!=source:
   raise ValueError('Current prepared source identity differs')
 if pk.get('semantic_packet',{}).get('sha256')!=t['semantic']['sha256']:
  raise ValueError('Current packet selects another semantic input')
 mask=array(sem['active_source_mask'],semantic_base)
 stat=array(sem['statistical_active_mask'],semantic_base)
 kind=array(sem['arrays']['kind'],semantic_base)
 for name,value in [('budget',mask),('statistical',stat)]:
  if value.shape!=(N,)or value.dtype.kind not in 'bu'or value.dtype.itemsize!=1 or np.any((value!=0)&(value!=1)):
   raise ValueError('Current '+name+' mask must contain raw-N Boolean bytes')
 if kind.shape!=(N,)or kind.dtype.kind not in 'iu'or np.any((kind<0)|(kind>2)):
  raise ValueError('Current source kind layout')
 active=mask.astype(bool);stat=stat.astype(bool)
 if np.any(stat&~active):raise ValueError('Statistical source outside current budget membership')
 counts={i:int(np.count_nonzero(active&(kind==i)))for i in range(3)}
 na=int(active.sum());ns=int(stat.sum());excluded=np.flatnonzero(~active)
 if sem.get('active_observations')!=na or t.get('active_rows')!=na or t.get('entity_count')!=counts[1]or t.get('profile_count')!=counts[2]:
  raise ValueError('Current active source-kind counts differ')
 if not np.array_equal(np.asarray(t.get('excluded_rows'),dtype=np.int64),excluded):
  raise ValueError('Current transport exclusions differ')
 stats=read(verify(sem['active_statistics'],semantic_base))
 if stats.get('active_source_mask_sha256')!=sem['active_source_mask']['sha256']or stats.get('statistical_active_mask_sha256')!=sem['statistical_active_mask']['sha256']:
  raise ValueError('Current statistics belong to another membership mask')
 if stats.get('active_observations')!=na or stats.get('statistical_observations')!=ns or stats.get('budget_only_observations')!=na-ns:
  raise ValueError('Current budget/statistical source counts differ')
 rank=int(stats['formal_residual_rows'])
 if rank<0 or rank!=declared['formal_W_rank']or t.get('full_W_rank')!=rank:
  raise ValueError('Current statistical rank differs')
 groups=declared['group_count']+declared['independent_count']
 if len(t['groups'])!=groups or t.get('original_groups_including_independent')!=groups:
  raise ValueError('Current complete W assembly group count differs')
 if not stats.get('statistical_partition_exactly_once')or not t.get('statistical_partition_exactly_once'):
  raise ValueError('Current statistics require once-only statistical membership')
 if not t.get('raw_arrays_preserved')or not t.get('source_order_is_raw_array_order'):
  raise ValueError('Current raw source order or payload changed')
 gpu=t.get('gpu_rows')
 if gpu is not None:
  expected=dict(packet=t['packet']['sha256'],semantic=t['semantic']['sha256'],source=source,active_mask=sem['active_source_mask']['sha256'],statistics=sem['active_statistics']['sha256'])
  if gpu.get('raw_rows')!=N or gpu.get('active_rows')!=na or not np.array_equal(np.asarray(gpu.get('excluded_rows'),dtype=np.int64),excluded)or any(gpu['packet_identity'].get(k)!=v for k,v in expected.items()):
   raise ValueError('GPU partials belong to another current input')
 scope=dict(raw_rows=N,active_rows=na,ordinary_count=counts[0],entity_count=counts[1],profile_count=counts[2],excluded_source_rows=excluded.tolist(),full_W_rank=rank,groups_including_independent=groups,active_mask_sha256=sem['active_source_mask']['sha256'],statistical_active_mask=sem['statistical_active_mask'],statistical_rows=ns,budget_only_rows=na-ns,active_statistics=sem['active_statistics'],source_identity=source,packet_sha256=t['packet']['sha256'],original_packet_sha256=t['packet']['sha256'],original_statistical_assets_unchanged=True,current_prepared_input=True)
 return scope,active,kind


def validate_semantic_scope(t,sem,pk,scope_contract,semantic_base=None):
 """Validate the selected mask against frozen source and original W assets.

 Active membership and counts may change only through the selected semantic
 authority. Raw packet, source UV/PTS/frame identity and W asset anchors do not.
 """
 if pk.get('current_input_scope',{}).get('schema')=='rinkfit.current_advertisement_input.v1':
  return _validate_current_semantic_scope(t,sem,pk,semantic_base)
 if scope_contract is None:raise ValueError('An explicit frozen event source contract is required')
 N=int(scope_contract['raw_rows'])
 if t.get('raw_rows')!=N or pk.get('observations')!=N or sem.get('observations')!=N:raise ValueError('Frozen raw source count changed')
 if pk['immutable_source_hashes']['observations.npz']!=scope_contract['source_identity']:raise ValueError('Frozen packet/source identity changed')
 if t['packet']['sha256']!=scope_contract['packet_sha256']:
  parent=pk.get('source_scope_packet_resolution',{}).get('original_raw_packet')
  if parent is None or parent.get('sha256')!=scope_contract['packet_sha256']:raise ValueError('Derived packet lacks its frozen original raw packet')
  parent_path=verify(parent);original=read(parent_path);packet_base=Path(t['packet'].get('path',t['packet'].get('file'))).parent
  allowed={'arrays','assets','string_tables','semantic_packet','source_audit_revision_branch','source_scope_packet_resolution'}
  if any(pk.get(k)!=v for k,v in original.items()if k not in allowed):raise ValueError('Derived packet changed original source/clock metadata')
  if set(pk.get('assets',{}))!=set(original.get('assets',{})):raise ValueError('Derived packet changed original asset membership')
  for key,desc in original.get('assets',{}).items():
   current=pk['assets'][key]
   if {k:v for k,v in current.items()if k not in ('file','path')}!={k:v for k,v in desc.items()if k not in ('file','path')}:raise ValueError('Derived packet changed original asset descriptor: '+key)
   asset_desc=current if current.get('file',current.get('path'))is not None else dict(current,file=str(parent_path.parent/key))
   verify(asset_desc,packet_base)
  if set(pk['arrays'])!=set(original['arrays']):raise ValueError('Derived packet changed original raw array membership')
  for key,desc in original['arrays'].items():
   current=pk['arrays'][key]
   if {k:v for k,v in current.items()if k not in ('file','path')}!={k:v for k,v in desc.items()if k not in ('file','path')}:raise ValueError('Derived packet changed original raw descriptor: '+key)
   verify(current,packet_base)
  if {k:v for k,v in pk['string_tables'].items()if k not in ('file','path')}!={k:v for k,v in original['string_tables'].items()if k not in ('file','path')}:raise ValueError('Derived packet changed source labels/point IDs')
  verify(pk['string_tables'],packet_base)
  if pk.get('semantic_packet',{}).get('sha256')!=t['semantic']['sha256']:raise ValueError('Derived packet selects another semantic authority')
 for sk,pkkey in [('source_uv','native_uv'),('native_pts','native_pts'),('frame_indices','frame_index')]:
  source=sem['immutable_source_arrays'][sk];packet=pk['arrays'][pkkey]
  if source['sha256_c_contiguous_array_bytes']!=packet['sha256']or source['shape']!=packet['shape']:raise ValueError('Original UV/PTS/frame source identity changed: '+sk)
 mask=array(sem['active_source_mask'],semantic_base);kind=array(sem['arrays']['kind'],semantic_base)
 if mask.shape!=(N,)or mask.dtype.kind not in 'bu'or mask.dtype.itemsize!=1 or np.any((mask!=0)&(mask!=1)):raise ValueError('Selected active mask must be raw-N Boolean bytes')
 if kind.shape!=(N,)or kind.dtype.kind not in 'iu'or np.any((kind<0)|(kind>2)):raise ValueError('Selected raw source kind layout')
 active=mask.astype(bool);excluded=np.flatnonzero(~active);counts={i:int(np.count_nonzero(active&(kind==i)))for i in range(3)};na=int(active.sum())
 if sem.get('active_observations')!=na or t.get('active_rows')!=na or t.get('entity_count')!=counts[1]or t.get('profile_count')!=counts[2]:raise ValueError('Transport/selected semantic active-kind counts differ')
 if not np.array_equal(np.asarray(t.get('excluded_rows'),dtype=np.int64),excluded):raise ValueError('Transport exclusions differ from the selected semantic mask')
 stats=read(verify(sem['active_statistics'],semantic_base))
 if stats.get('active_source_mask_sha256')!=sem['active_source_mask']['sha256']or stats.get('active_observations')!=na:raise ValueError('Selected statistics are bound to another mask')
 if stats.get('original_groups_asset_sha256')!=scope_contract['original_groups_asset_sha256']or stats.get('original_weights_array_sha256')!=scope_contract['original_weights_array_sha256']or not stats.get('independent_statistics_unchanged'):raise ValueError('Original statistical assets changed')
 rank=int(stats['formal_residual_rows'])
 if rank<=0 or rank>scope_contract['original_formal_rank']or t.get('full_W_rank')!=rank or not t.get('statistical_partition_exactly_once'):raise ValueError('Selected original statistical rank/partition mismatch')
 if not t.get('raw_arrays_preserved')or not t.get('source_order_is_raw_array_order'):raise ValueError('Original raw source order/payload not preserved')
 if len(t['groups'])!=t.get('original_groups_including_independent'):raise ValueError('Declared original group assembly count differs')
 if 'source_exclusion'in sem:verify(sem['source_exclusion'],semantic_base)
 if 'source_audit_revision'in sem:
  revision=read(verify(sem['source_audit_revision'],semantic_base));selected_authority=sem['source_audit_revision']['sha256']
  revision_excluded=sorted(revision.get('historical_human_authorized_exclusions',[])+revision.get('new_audit_exclusions',[]))
  if revision.get('source_identity')!=scope_contract['source_identity']or revision.get('raw_observations')!=N or revision.get('active_observations')!=na or revision_excluded!=excluded.tolist()or not revision.get('raw_rows_UV_ID_labels_W_E_remain_immutable'):raise ValueError('Source audit revision differs from the selected original-source scope')
  if stats.get('source_audit_revision_sha256')!=selected_authority:raise ValueError('Selected statistics/source audit identity differs')
 else:selected_authority=sem.get('source_exclusion',{}).get('sha256')
 if selected_authority is not None and stats.get('source_exclusion_sha256')!=selected_authority:raise ValueError('Selected statistics/source exclusion identity differs')
 gpu=t.get('gpu_rows')
 if gpu is not None:
  identity=gpu['packet_identity'];expected=dict(packet=t['packet']['sha256'],semantic=t['semantic']['sha256'],source=scope_contract['source_identity'],active_mask=sem['active_source_mask']['sha256'],statistics=sem['active_statistics']['sha256'])
  if gpu.get('raw_rows')!=N or gpu.get('active_rows')!=na or not np.array_equal(np.asarray(gpu.get('excluded_rows'),dtype=np.int64),excluded)or any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('GPU partials belong to another selected source scope')
 scope=dict(raw_rows=N,active_rows=na,ordinary_count=counts[0],entity_count=counts[1],profile_count=counts[2],excluded_source_rows=excluded.tolist(),full_W_rank=rank,groups_including_independent=len(t['groups']),active_mask_sha256=sem['active_source_mask']['sha256'],active_statistics=sem['active_statistics'],source_identity=scope_contract['source_identity'],packet_sha256=t['packet']['sha256'],original_packet_sha256=scope_contract['packet_sha256'],original_statistical_assets_unchanged=True)
 return scope,active,kind

def load_fixed_entity_norms(qbar_manifest,transport_manifest,offset,scale,free,*,scope_contract=None):
 """Strict same-state reader. offset/scale/free are the original master chart."""
 Q=Path(qbar_manifest);q=read(Q);T=Path(transport_manifest);t=read(T)
 if t['schema']!='rinkfit.whole_original_W_native_UV_transport.v1':raise ValueError('Whole UV transport schema')
 for key in ['model','packet','semantic']:
  verify(t[key]);verify(q[key])
  if t[key]['sha256']!=q[key]['sha256']:raise ValueError('Different same-state '+key)
 S=verify(t['semantic'],T.parent);sem=read(S);P=verify(t['packet'],T.parent);pk=read(P);scope,active,kind=validate_semantic_scope(t,sem,pk,scope_contract,S.parent);N=scope['raw_rows']
 cp_path=verify(t['checkpoint']);cp=read(cp_path)
 if cp['model']['sha256']!=t['model']['sha256']:raise ValueError('Transport CP theta mismatch')
 verify(cp['model']);I=verify(q['input']);inp=read(I);selected=array(q['source_rows'],Q.parent).astype(np.int64)
 for key,ckey in [('station','station'),('path','branch'),('row','row'),('alpha','alpha')]:
  expected=array(inp['arrays'][key],I.parent).copy()
  if key in ['station','row']:expected[selected]=array(q['linearization_'+key],Q.parent)
  actual=array(cp['row_state_files'][ckey],cp_path.parent)
  if expected.dtype!=actual.dtype or expected.shape!=(N,)or expected.tobytes()!=actual.tobytes():raise ValueError('Different complete nuisance state: '+key)
 rows=np.flatnonzero(active&(kind==1)).astype('<u8')
 if len(rows)!=scope['entity_count']:raise ValueError('Fixed entity source kind membership')
 for key in ['station_UV_left','station_UV_right']:
  tang=array(t['arrays'][key],T.parent).reshape(N,2)
  if np.any(tang[rows]!=0):raise ValueError('Entity has a station tangent; not a fixed physical point')
 P=verify(t['packet']);pk=read(P);E=array(pk['arrays']['engineering_budget_px'],P.parent)[rows]
 if not np.isfinite(E).all()or np.any(E<=0):raise ValueError('Original source E')
 residual=array(t['arrays']['error_uv'],T.parent).reshape(N,2)[rows]/E[:,None]
 model=read(verify(t['model']));expected_offset=np.r_[0,np.cumsum([len(b['values'])for b in model['blocks']])]
 scale=np.asarray(scale);offset=np.asarray(offset,dtype=np.int64);free=np.asarray(free,dtype=np.int64)
 if not np.array_equal(offset,expected_offset)or not np.array_equal(scale,np.concatenate([b['scales']for b in model['blocks']])):raise ValueError('Original master scales/layout mismatch')
 if len(np.unique(free))!=len(free)or np.any(free<0)or np.any(free>=offset[-1]):raise ValueError('Invalid/duplicate free physical component')
 fixed=np.zeros(offset[-1],bool)
 for block,b in enumerate(model['blocks']):
  if b.get('constant',False):fixed[offset[block]:offset[block+1]]=True
  for c in b.get('fixed_indices',[]):fixed[offset[block]+c]=True
 if np.any(fixed[free]):raise ValueError('The master attempts to release an original constant or gauge component')
 mapping=np.full(offset[-1],-1,np.int64);mapping[free]=np.arange(len(free))
 csr=t['physical_UV_CSR'];ptr=array(csr['row_offsets'],T.parent,mmap=True);bi=array(csr['block_id'],T.parent,mmap=True)
 co=array(csr['component'],T.parent,mmap=True);val=array(csr['values'],T.parent,mmap=True)
 if len(ptr)!=2*N+1 or ptr[-1]!=len(bi)or len(bi)!=len(co)or len(co)!=len(val):raise ValueError('Whole UV CSR shape')
 rr=[];cc=[];vv=[]
 for j,row in enumerate(rows):
  for component in range(2):
   a,b=map(int,ptr[2*int(row)+component:2*int(row)+component+2]);ambient=offset[bi[a:b]]+co[a:b];keep=mapping[ambient]>=0
   rr.append(np.full(int(keep.sum()),2*j+component,np.int64));cc.append(mapping[ambient[keep]])
   vv.append(val[a:b][keep]*scale[ambient[keep]]/E[j])
 J=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(2*len(rows),len(free)))if len(rows)else sparse.csr_matrix((0,len(free)))
 if not np.isfinite(J.data).all()or not np.isfinite(residual).all():raise ValueError('Entity affine arithmetic')
 metadata=dict(schema='rinkfit.fixed_entity_native_UV_affine_norm_chart.v1',qbar=ref(Q),transport=ref(T),checkpoint=t['checkpoint'],
  model=t['model'],packet=t['packet'],semantic=t['semantic'],raw_rows=N,active_rows=scope['active_rows'],
  validated_scope=scope,native_norm_core=dict(path=str(Path(__file__).resolve()),sha256=CORE_SOURCE_SHA256),entity_count=len(rows),entity_source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
  complete_nuisance_sha256={k:cp['row_state_files'][k]['sha256']for k in ['station','branch','row','alpha']},
  original_E_unchanged=True,original_W_rank=scope['full_W_rank'],original_W_unchanged=True,
  entity_S_is_physically_fixed=True,physical_parameters_or_shape_DOF_added=False,
  normalized_residual='(trueRS_native_UV_error)/original_engineering_budget_px',
  primary_SOC='SOC4(b+1,2*rU,2*rV,b-1); original b coordinate retained',
  W_secondary_SOC='SOC3(sqrt(same_primary_b),rU(base+TR*d),rV(base+TR*d))',
  finite_cut_set_needs_all_entity_norm_audit=True,affine_operator_is_not_nonlinear_physical_certificate=True)
 return FixedEntityNorms(rows,residual,J,metadata)

def solve_entity_primary_soc(A,rhs,lower,upper,entities,trust,qref=None,threads=1):
 """Solve the same original linear master plus every fixed-entity norm.

 b is optimized anew. The numeric chart is delta=trust*d, b=qref*beta.
 No former LP complementary face is carried into this mixed conic problem.
 Returned primal is only eligible for a W secondary; no physical trial runs.
 """
 import clarabel,time
 start=time.monotonic();A=A.tocsr();rhs=np.asarray(rhs);lower=np.asarray(lower);upper=np.asarray(upper)
 n=entities.J.shape[1];assert A.shape[1]==n+1 and lower.shape==upper.shape==(n+1,)
 if not np.all(np.isfinite(lower))or not np.all(np.isfinite(upper[:-1]))or lower[-1]!=0 or upper[-1]!=np.inf:raise ValueError('Original finite physical box and b>=0 are required')
 if np.max(A[:,-1].data,initial=0.)>0:raise ValueError('Original master has an epigraph upper-side constraint')
 if trust<=0 or not np.isfinite(trust):raise ValueError('Positive physical TR')
 if qref is None:qref=max(1.,float(entities.values(np.zeros(n))[0].max()))
 if qref<=0 or not np.isfinite(qref):raise ValueError('Positive epigraph unit')
 units=np.r_[np.full(n,trust),qref];normalized=FixedEntityNorms(entities.rows,entities.residual/np.sqrt(qref),entities.J*(trust/np.sqrt(qref)),entities.metadata)
 Ac,bc=normalized.primary_SOC4();Al=A.multiply(units[None,:]).tocsr()
 rownorm=np.asarray(abs(Al).max(axis=1).toarray()).ravel();row_scale=np.ones(len(rhs));nonzero=rownorm>0
 row_scale[nonzero]=np.exp2(-np.rint(np.log2(rownorm[nonzero])))
 Al=Al.multiply(row_scale[:,None]).tocsr();bl=rhs*row_scale
 I=sparse.eye(n+1,format='csr');upper_ids=np.flatnonzero(np.isfinite(upper))
 B=sparse.vstack([Ac,Al,-I,I[upper_ids]],format='csc');bb=np.r_[bc,bl,-lower/units,upper[upper_ids]/units[upper_ids]]
 c=np.zeros(n+1);c[-1]=1.;settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=int(threads)
 settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
 cones=[clarabel.SecondOrderConeT(4)for _ in entities.rows]+[clarabel.NonnegativeConeT(len(bl)+n+1+len(upper_ids))]
 solver=clarabel.DefaultSolver(sparse.csc_matrix((n+1,n+1)),c,B,bb,cones,settings)
 result=solver.solve();numeric=np.asarray(result.x);raw=numeric*units;canonical=raw.copy()
 # The origin is the accepted physical state and lies inside the unchanged
 # physical box. Correct a numerical endpoint overshoot by a single convex
 # ray contraction of the entire physical step, never by coordinate clamps.
 if np.any(lower[:-1]>0)or np.any(upper[:-1]<0):raise ValueError('Accepted origin outside original physical box')
 eta=1.
 for j,xj in enumerate(raw[:-1]):
  if xj<lower[j]:eta=min(eta,float(lower[j]/xj))
  elif xj>upper[j]:eta=min(eta,float(upper[j]/xj))
 if eta<1.:eta=np.nextafter(eta,0.);canonical[:-1]=eta*raw[:-1]
 qe,_=entities.values(canonical[:-1])
 # Form a genuinely feasible epigraph upper of this newly solved primary.
 # This never freezes or pretends feasibility of the old LP b. A gap audit
 # below must pass before this upper is eligible for the W secondary.
 required=float(qe.max());bcol=np.asarray(A[:,-1].toarray()).ravel();physical=A[:,:-1]@canonical[:-1]
 mask=bcol<0
 if np.any(mask):required=max(required,float(np.max((physical[mask]-rhs[mask])/(-bcol[mask]))))
 canonical[-1]=np.nextafter(max(float(raw[-1]),required,0.),np.inf)
 residual=A@canonical-rhs;roundoff=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(canonical))
 box=bool(np.all(canonical>=lower)and np.all(canonical<=upper));entity_audit=entities.audit(canonical[:-1],canonical[-1])
 dual=np.asarray(result.z);dual_raw_cost=-float(bb@dual);stationarity=c+B.T@dual
 # Numeric dual-domain audit for the exact transformed conic program.
 zd=dual[:len(bc)].reshape(-1,4);soc_dual_violation=float(np.max(np.linalg.norm(zd[:,1:],axis=1)-zd[:,0],initial=0.))
 nonnegative_violation=float(np.max(-dual[len(bc):],initial=0.))
 gap=float(canonical[-1]-qref*dual_raw_cost);gap_tol=max(1e-8,1e-8*abs(float(canonical[-1])))
 numerical=bool(str(result.status)=='Solved'and np.max(abs(stationarity),initial=0.)<=1e-8
                and soc_dual_violation<=1e-8 and nonnegative_violation<=1e-8 and abs(gap)<=gap_tol)
 primal=bool(box and np.all(residual<=roundoff)and entity_audit['all_entity_affine_norms_within_b'])
 report=dict(schema='rinkfit.fixed_entity_mixed_SOC_primary_result.v1',solver_status=str(result.status),solver_iterations=result.iterations,
  entity_count=len(entities.rows),variables=n+1,original_linear_rows=A.shape[0],added_SOC4=len(entities.rows),
  original_A_rhs_box_and_TR_unchanged=True,old_LP_b_not_fixed=True,old_LP_complementary_face_not_used=True,
  qref=float(qref),physical_scaled_trust=float(trust),numeric_units={'physical':float(trust),'epigraph':float(qref)},
  raw_primary_b=float(raw[-1]),canonical_primary_b=float(canonical[-1]),canonical_epigraph_upper_lift=float(canonical[-1]-raw[-1]),
  physical_step_ray_factor=float(eta),physical_step_ray_correction_inf=float(np.max(abs(canonical[:-1]-raw[:-1]),initial=0.)),
  physical_step_not_coordinate_clamped=True,
  original_master_max_violation=float(np.max(residual,initial=0.)),original_master_beyond_IEEE_count=int(np.count_nonzero(residual>roundoff)),
  box_strictly_satisfied=box,entity_norm_audit=entity_audit,strict_original_primal_gate=primal,
  transformed_dual_stationarity_inf=float(np.max(abs(stationarity),initial=0.)),scaled_numeric_dual_cost=float(qref*dual_raw_cost),
  SOC_dual_cone_violation=soc_dual_violation,linear_dual_sign_violation=nonnegative_violation,numeric_primal_dual_gap=gap,
  numeric_gap_tolerance=gap_tol,original_numeric_gate=1e-8,internal_solve_accuracy=1e-9,
  numeric_conic_solver_gate=numerical,master_ready_for_W_secondary=primal and numerical,
  proposed=False,physical_proposal_authorized=False,requires_same_b_entity_SOC_W_secondary=True,
  all_original_W_and_source_obligations_unchanged=True,full_source_actual_nonlinear_trial_required=True,
  nonlinear_KKT_or_final_fit=False,numerical_duals_are_only_this_affine_conic_program=True,
  formal_interval_dual_lower_certificate=False,elapsed_seconds=time.monotonic()-start)
 at=len(bc);ineq=-qref*dual[at:at+len(rhs)]*row_scale;at+=len(rhs)
 lower_dual=qref*dual[at:at+n+1]/units;at+=n+1
 upper_dual=np.zeros(n+1);upper_dual[upper_ids]=-qref*dual[at:]/units[upper_ids]
 return dict(report=report,master_primal=canonical,raw_primal=raw,entity_cone_duals=zd,
  transformed_conic_dual=dual,transformed_stationarity=stationarity,
  master_inequality_dual=ineq,master_bound_lower_dual=lower_dual,master_bound_upper_dual=upper_dual,
  entity_q_multiplier_candidate=zd[:,0]+zd[:,3],row_scale=row_scale,numerical_units=units)

def solve_entity_W_secondary(A,rhs,lower,upper,entities,primary,J_W,r_W,trust,threads=1):
 """Same-b correlated-W convex QP with all entity SOC3 constraints.

 No old linear-master dual face is used. The complete original linear A and
 native parameter box are retained. The returned delta is audited again in
 original units, after all numerical processing, against every entity norm.
 A caller applying any later feasibility repair MUST audit its result again.
 """
 import clarabel,time
 start=time.monotonic();A=A.tocsr();rhs=np.asarray(rhs);primary=np.asarray(primary)
 lower=np.asarray(lower);upper=np.asarray(upper);J_W=J_W.tocsc();r_W=np.asarray(r_W)
 n=entities.J.shape[1];assert primary.shape==(n+1,)and A.shape[1]==n+1 and J_W.shape[1]==n
 bstar=float(primary[-1]);TR=float(trust)
 if TR<=0 or not np.isfinite(TR)or bstar<0 or not np.isfinite(bstar):raise ValueError('Finite trust and canonical primary b')
 initial_linear=A@primary-rhs;initial_error=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(primary))
 if np.any(initial_linear>initial_error)or np.any(primary<lower)or np.any(primary>upper)or not entities.audit(primary[:-1],bstar)['all_entity_affine_norms_within_b']:raise ValueError('Primary lacks original-unit feasibility')
 # Center at the known primary, eliminate b exactly, and keep the full
 # original face as inequalities. Cone curvature may imply a different
 # supporting face than the former scalar-q LP.
 Ac,bc=entities.secondary_SOC3(primary[:-1],TR,bstar)
 Bl=A[:,:n]*TR;bl=rhs-A@primary
 rownorm=np.asarray(abs(Bl).max(axis=1).toarray()).ravel();row_scale=np.ones(len(rhs));nonzero=rownorm>0
 row_scale[nonzero]=np.exp2(-np.rint(np.log2(rownorm[nonzero])))
 Bl=Bl.multiply(row_scale[:,None]).tocsr();bl=bl*row_scale
 ld=(lower[:-1]-primary[:-1])/TR;ud=(upper[:-1]-primary[:-1])/TR
 if not np.isfinite(ld).all()or not np.isfinite(ud).all():raise ValueError('Finite original physical box')
 I=sparse.eye(n,format='csr');B=sparse.vstack([Ac,Bl,-I,I],format='csc');bb=np.r_[bc,bl,-ld,ud]
 base_residual=r_W+J_W@primary[:-1];H=(J_W.T@J_W).tocsc();g=np.asarray(J_W.T@base_residual).ravel()
 objective_scale=max(1.,float(.5*r_W@r_W));P=sparse.triu(H*(TR*TR/objective_scale)).tocsc();c=g*(TR/objective_scale)
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=int(threads)
 settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
 cones=[clarabel.SecondOrderConeT(3)for _ in entities.rows]+[clarabel.NonnegativeConeT(len(rhs)+2*n)]
 solver=clarabel.DefaultSolver(P,c,B,bb,cones,settings);s=solver.solve();d=np.asarray(s.x)
 delta=primary[:-1]+TR*d;final=np.r_[delta,bstar]
 # Original native bound representations are exact, with a subsequent full
 # A/entity audit. This is representation canonicalization, never a clamp.
 represented=[]
 for j in range(n):
  if d[j]==ld[j]:delta[j]=lower[j];represented.append([j,'lower'])
  elif d[j]==ud[j]:delta[j]=upper[j];represented.append([j,'upper'])
 final[:-1]=delta;viol=A@final-rhs;error=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(final))
 box=bool(np.all(final>=lower)and np.all(final<=upper));audit=entities.audit(delta,bstar)
 primal=bool(np.all(viol<=error)and box and audit['all_entity_affine_norms_within_b'])
 dual=np.asarray(s.z);zd=dual[:len(bc)].reshape(-1,3)
 dual_residual=np.asarray((H*(TR*TR/objective_scale))@d+c+B.T@dual);dual_cone=float(np.max(np.linalg.norm(zd[:,1:],axis=1)-zd[:,0],initial=0.))
 dual_sign=float(np.max(-dual[len(bc):],initial=0.));pred=r_W+J_W@delta
 numeric_gap=abs(float(s.obj_val-s.obj_val_dual));numeric_gap_tol=max(1e-8,1e-8*abs(float(s.obj_val)))
 numerical=bool(str(s.status)=='Solved'and np.max(abs(dual_residual),initial=0.)<=1e-8 and dual_cone<=1e-8 and dual_sign<=1e-8 and numeric_gap<=numeric_gap_tol)
 report=dict(schema='rinkfit.same_primary_correlated_W_entity_SOC_secondary.v1',solver_status=str(s.status),solver_iterations=s.iterations,
  proposed=bool(primal and numerical),fixed_primary_b=bstar,canonical_primary_b_IEEE_unchanged=bool(final[-1].tobytes()==primary[-1].tobytes()),
  old_LP_dual_complementary_face_not_used=True,all_original_A_rhs_box_TR_unchanged=True,all_entity_norms_constrained_and_finally_audited=True,
  entity_norm_audit=audit,original_master_max_violation=float(np.max(viol,initial=0.)),original_master_beyond_IEEE_count=int(np.count_nonzero(viol>error)),
  original_box_strictly_satisfied=box,strict_original_primal_gate=primal,numeric_conic_solver_gate=numerical,
  transformed_dual_stationarity_inf=float(np.max(abs(dual_residual),initial=0.)),SOC_dual_cone_violation=dual_cone,linear_dual_sign_violation=dual_sign,
  numeric_primal_dual_gap=numeric_gap,numeric_gap_tolerance=numeric_gap_tol,
  baseline_W_half_cost=float(.5*r_W@r_W),primary_W_half_cost=float(.5*base_residual@base_residual),predicted_W_half_cost=float(.5*pred@pred),
  original_W_rank=int(J_W.shape[0]),original_W_unchanged=True,source_E_unchanged=True,positive_objective_scale=objective_scale,
  native_bound_representations=represented,added_entity_SOC3=len(entities.rows),original_linear_rows=len(rhs),H_nnz=int(H.nnz),
  any_later_original_A_repair_requires_new_all_entity_audit=True,no_entity_dual_is_nonlinear_PHR_KKT=True,
  full_source_nonlinear_trial_required=True,PhaseII_joint_station_optimum=False,final_fit_or_KKT=False,
  original_numeric_gate=1e-8,internal_solve_accuracy=1e-10,elapsed_seconds=time.monotonic()-start)
 return dict(report=report,master_primal=final,step_scaled=delta,centered_QP_primal=d,conic_dual=dual,
  entity_cone_duals=zd,transformed_stationarity=dual_residual,row_scale=row_scale)


@dataclass
class AllSourceAffineNorms:
 rows:np.ndarray
 kind:np.ndarray
 residual:np.ndarray
 J:sparse.csr_matrix
 metadata:dict
 def values(self,delta):
  r=self.residual+(self.J@delta).reshape(-1,2)
  if not np.isfinite(r).all():raise ValueError('Nonfinite full-source affine residual')
  return np.einsum('ij,ij->i',r,r),r
 def subset(self,source_rows):
  source_rows=np.asarray(source_rows,dtype=np.int64);idx=np.searchsorted(self.rows,source_rows)
  if np.any(idx>=len(self.rows))or not np.array_equal(self.rows[idx],source_rows)or len(np.unique(idx))!=len(idx):raise ValueError('Invalid/duplicate active norm membership')
  uv=np.ravel(np.column_stack([2*idx,2*idx+1]));meta=dict(self.metadata)
  meta.update(selected_norm_rows=len(idx),selected_kind_counts={str(k):int(np.count_nonzero(self.kind[idx]==k))for k in [0,1,2]})
  return FixedEntityNorms(source_rows.astype('<u8'),self.residual[idx],self.J[uv].tocsr(),meta)
 def audit(self,delta,b):
  q,r=self.values(delta);bad=np.flatnonzero(q>b);worst=int(np.argmax(q-b))
  return dict(active_rows=len(self.rows),ordinary_count=int(np.count_nonzero(self.kind==0)),entity_count=int(np.count_nonzero(self.kind==1)),
   profile_count=int(np.count_nonzero(self.kind==2)),maximum_affine_q=float(q.max()),maximum_affine_q_minus_b=float(np.max(q-b)),
   worst_source_row=int(self.rows[worst]),violating_source_count=len(bad),violating_source_rows=self.rows[bad].tolist(),
   violating_kind_counts={str(k):int(np.count_nonzero(self.kind[bad]==k))for k in [0,1,2]},all_source_affine_norms_within_b=bool(not len(bad)),
   ordinary_fixed_S_is_temporary_numerical_witness=True,profile_original_normal_1D=True,nonlinear_physical_acceptance=False)

def load_all_source_affine_norms(qbar,transport,offset,scale,free,*,scope_contract=None):
 # Reuse the completed strict theta, authority, all4aux and layout validator.
 entities=load_fixed_entity_norms(qbar,transport,offset,scale,free,scope_contract=scope_contract)
 T=Path(transport);t=read(T);S=verify(t['semantic']);sem=read(S);P=verify(t['packet']);pk=read(P)
 scope,active,kind=validate_semantic_scope(t,sem,pk,scope_contract,S.parent);N=scope['raw_rows']
 rows=np.flatnonzero(active).astype('<u8');E=array(pk['arrays']['engineering_budget_px'],P.parent)
 normal=array(sem['arrays']['profile_normal'],S.parent);profile=np.flatnonzero(active&(kind==2))
 if len(profile)!=scope['profile_count'] or not np.isfinite(normal[profile]).all():raise ValueError('Original profile normal coverage')
 offset=np.asarray(offset);scale=np.asarray(scale);free=np.asarray(free);mapping=np.full(offset[-1],-1,np.int64);mapping[free]=np.arange(len(free))
 c=t['physical_UV_CSR'];ptr=array(c['row_offsets'],T.parent);bi=array(c['block_id'],T.parent);co=array(c['component'],T.parent);v=array(c['values'],T.parent)
 ambient=offset[bi]+co;keep=mapping[ambient]>=0;ri=np.repeat(np.arange(2*N),np.diff(ptr).astype(np.int64))
 UV=sparse.csr_matrix((v[keep]*scale[ambient[keep]],(ri[keep],mapping[ambient[keep]])),shape=(2*N,len(free)))
 ordinary=np.flatnonzero(active&(kind!=2));diag=np.ravel(np.column_stack([2*ordinary,2*ordinary+1]));weights=np.repeat(1/E[ordinary],2)
 lr=np.r_[diag,2*profile,2*profile];lc=np.r_[diag,2*profile,2*profile+1];lv=np.r_[weights,normal[profile,0]/E[profile],normal[profile,1]/E[profile]]
 transform=sparse.csr_matrix((lv,(lr,lc)),shape=(2*N,2*N));J=(transform@UV).tocsr();J.eliminate_zeros()
 error=array(t['arrays']['error_uv'],T.parent).reshape(N,2).copy()
 if not np.isfinite(error[active]).all()or np.any(E[active]<=0):raise ValueError('All-source native primal arithmetic')
 r=np.zeros((N,2));r[ordinary]=error[ordinary]/E[ordinary,None]
 r[profile,0]=np.einsum('ij,ij->i',normal[profile],error[profile])/E[profile]
 pair=np.ravel(np.column_stack([2*rows,2*rows+1]));J=J[pair].tocsr()
 if not np.isfinite(J.data).all():raise ValueError('All-source normalized Jacobian arithmetic')
 meta=dict(entities.metadata);meta.update(schema='rinkfit.all_source_affine_native_norm_proposal_chart.v1',active_rows=len(rows),
  ordinary_entity_native_2D_count=len(ordinary),genuine_profile_normal_1D_count=len(profile),entities_always_included=scope['entity_count'],
  ordinary_station_fixed_only_for_this_numerical_proposal=True,final_PhaseII_joint_station_claim=False,unselected_source_not_deleted=True,
  all_source_affine_check_is_not_nonlinear_acceptance=True)
 return AllSourceAffineNorms(rows,kind[rows],r[rows],J,meta)


CORE_SOURCE_SHA256=sha(__file__)
