"""Issue explicit engineering source budgets for the first-four capture successor.

This is deliberately not a source-noise covariance or statistical W. Inherited
per-observation budgets come from the actual parent native ledger. New direct
normal profiles use one separately declared channel with the parent's existing
S02/board mass; they never dilute inherited fixed 2-D texture observations.
"""
from pathlib import Path
import argparse,json,hashlib,collections,struct
import numpy as np
from .io_utils import file_sha256
SCHEMA='rinkfit.explicit_engineering_source_weights.v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def key_digest(p,a,n=None):
 h=hashlib.sha256();n=p['row_count']if n is None else n
 for i in range(n):
  fields=[str(i),p['source_dataset_ids'][i],p['source_row_ids'][i],p['shot_ids'][int(a['shot_index'][i])],str(int(a['native_pts'][i]))]
  if any('\x1f'in x or'\n'in x or'\r'in x for x in fields):raise ValueError('source key contains delimiter')
  h.update(('\x1f'.join(fields)+'\n').encode('utf8'))
 return h.hexdigest()
def load(p,base):
 a={};seen=set()
 for k,s in p['arrays'].items():
  f=Path(s['file']);f=f if f.is_absolute()else Path(base)/f;key=(str(f),s['sha256'])
  if key not in seen:
   if file_sha256(f)!=s['sha256']:raise ValueError('changed source array '+k)
   seen.add(key)
  a[k]=np.memmap(f,dtype=s['dtype'],mode='r',shape=tuple(s['shape']),offset=s.get('offset_bytes',0))
 return a

def issue(parent_packet,parent_pixel_ledger,parent_candidate,next_packet,output):
 pp,lp,cp,npk,out=map(Path,(parent_packet,parent_pixel_ledger,parent_candidate,next_packet,output))
 if out.exists():raise FileExistsError('immutable explicit source-weight generation exists')
 p,q,ledger,binding=read(pp),read(npk),read(lp),read(cp/'source_uv_capture_state_binding.json');a=load(p,pp.parent);b=load(q,npk.parent);n,N=p['row_count'],q['row_count']
 if binding['packet_sha256']!=file_sha256(pp)or binding['model_sha256']!=file_sha256(cp/'model.json'):raise ValueError('parent complete state/source differs')
 if ledger['schema']!='rinkfit.engineering_pixel_contributions.v1'or ledger['active_rows']!=p['active_rows']or ledger['original_W_likelihood_consumed']:raise ValueError('not the actual engineering parent ledger')
 if p['shot_ids']!=q['shot_ids']or p['shot_ids']!=['s01','s02','s03','s04']or N<n:raise ValueError('source scope/prefix changed')
 for k in ['source_row_ids','source_dataset_ids','source_family_ids']:
  if p[k]!=q[k][:n]:raise ValueError('inherited identity changed '+k)
 for k in ['native_pts','shot_index','kind','material_index','E']:
  if a[k].tobytes()!=b[k][:n].tobytes():raise ValueError('inherited source identity or E changed '+k)
 A=(a['admission']==1)&(a['fit_role']==1);B=(b['admission']==1)&(b['fit_role']==1)
 if np.any((~A)&B[:n]):raise ValueError('inactive parent source revived')
 if key_digest(p,a)!=key_digest(q,b,n):raise ValueError('prefix source key registry differs')
 labels=[m['label']for m in p['material_binding']['materials']]
 family=lambda i:'board'if labels[int(a['material_index'][i])].startswith('board.')else'advertisement'if labels[int(a['material_index'][i])].startswith(('Ad','advertisement'))else'ice'
 units={int(i):g['id']for g in p['engineering_pixel_dependency_groups']['groups']for i in g['rows']};groups={g['id']:(j,g)for j,g in enumerate(ledger['groups'])};parent_weights=np.zeros(n);assignment=np.full(n,0xffffffff,dtype='<u4');counts=np.zeros(len(groups),int)
 for i in np.flatnonzero(A):
  sid=p['shot_ids'][int(a['shot_index'][i])];f=family(i);key=sid+'/'+f+'/'+str(int(a['native_pts'][i]))+'/'+units[int(i)]
  if key not in groups:raise ValueError('parent actual ledger omitted source')
  j,g=groups[key];w=float(g['per_row_weight'])
  if not np.isfinite(w)or w<=0:raise ValueError('nonpositive actual parent budget')
  parent_weights[i]=w;assignment[i]=j;counts[j]+=1
 if not np.array_equal(counts,[g['rows']for g in ledger['groups']]):raise ValueError('parent native group membership denominator differs')
 channel_budget=sum(g['rows']*g['per_row_weight']for g in ledger['groups']if g['shot']=='s02'and g['family']=='board')
 if not np.isfinite(channel_budget)or channel_budget<=0:raise ValueError('missing existing S02 board budget')
 weights=np.zeros(N);weights[:n]=np.where(B[:n],parent_weights,0.);classes=np.zeros(N,dtype='u1');classes[:n]=np.where(B[:n],np.where(a['kind']==1,1,2),0)
 new=np.flatnonzero(B&(np.arange(N)>=n));queries={int(x['row_index']):x for x in q['queries']};nextunits={int(i):g['id']for g in q['engineering_pixel_dependency_groups']['groups']for i in g['rows']};newlabels=[m['label']for m in q['material_binding']['materials']];channels=collections.defaultdict(list);exposures=set();roles=collections.defaultdict(set);deps=collections.defaultdict(set);role_proofs={}
 qual=q['source_only_board_successor']['qualification'];qf=Path(qual['file'])
 if file_sha256(qf)!=qual['sha256']:raise ValueError('board qualification changed')
 qa=read(qf)
 if qa['actual_49_sample_profile_mean_replay_max_px']!=0:raise ValueError('board exact frozen mean was not preserved')
 for i in new:
  sid=q['shot_ids'][int(b['shot_index'][i])];pt=int(b['native_pts'][i]);label=newlabels[int(b['material_index'][i])];cq=queries.get(int(i))
  if sid!='s02'or label not in ['board.top','board.contact']or b['kind'][i]==1 or cq is None or len(cq['alpha'])!=49 or len(cq['native_uv'])!=49:raise ValueError('new budget only for actual S02 direct normal board profiles')
  if b['E'][i]<=0 or not np.isfinite(b['E'][i]):raise ValueError('bad new original engineering E')
  role=q['source_family_ids'][i];rk=(sid,pt,label,role);unit=nextunits[int(i)];key=(sid,pt,label,role,unit);channels[key].append(int(i));exposures.add((sid,pt));roles[sid,pt].add((label,role));deps[rk].add(unit);classes[i]=3
 for key,rr in channels.items():
  sid,pt,label,role,unit=key;w=channel_budget/len(exposures)/len(roles[sid,pt])/len(deps[sid,pt,label,role])/len(rr);weights[rr]=w
  role_proofs[str(key)]=dict(shot=sid,native_pts=pt,physical_role=label,actual_source_family_id=role,source_dependency_id=unit,rows=len(rr),per_row_weight=w,role_order_or_noise_independence_claimed=False)
 if not np.array_equal(weights>0,B)or not np.isfinite(weights).all():raise ValueError('explicit budget active partition failed')
 if not np.array_equal(weights[:n][B[:n]],parent_weights[B[:n]]):raise ValueError('inherited active source budget drift')
 if not np.isclose(weights[new].sum(),channel_budget,rtol=1e-13,atol=1e-10):raise ValueError('new profile channel budget does not close')
 out.mkdir(parents=True);files={}
 for name,x in [('weights.f64',weights.astype('<f8')),('active.u8',B.astype('u1')),('information_class.u8',classes),('parent_group.u32',assignment)]:
  x.tofile(out/name);files[name]={**ref(out/name),'dtype':x.dtype.str,'shape':list(x.shape)}
 oldwithdraw=np.flatnonzero(A&~B[:n]);report=dict(schema=SCHEMA,parent_packet=ref(pp),source_packet=ref(npk),parent_pixel_ledger=ref(lp),parent_complete_binding=ref(cp/'source_uv_capture_state_binding.json'),parent_complete_checkpoint=ref(cp/'engineering_checkpoint.json'),parent_state_token=binding['state_token'],issuer=ref(__file__),source_board_qualification=qual,source_key_digest_rule='UTF8 join U+001F: raw_index,dataset_id,source_row_id,shot_id,native_pts; newline per raw row',source_key_registry_sha256=key_digest(q,b),parent_prefix_key_registry_sha256=key_digest(p,a),source_E_binary64_le_sha256=hashlib.sha256(np.asarray(b['E'],dtype='<f8').tobytes()).hexdigest(),parent_rows=n,raw_rows=N,active_rows=int(B.sum()),parent_active_rows=int(A.sum()),inherited_active_rows=int(B[:n].sum()),parent_withdrawn_to_pending_rows=oldwithdraw.tolist(),new_direct_normal_profile_rows=len(new),information_classes={'0':'inactive','1':'inherited fixed 2D material point','2':'inherited sliding finite curve/profile','3':'new actual direct native normal profile; separate engineering source channel'},files=files,parent_original_total_budget=float(parent_weights.sum()),retained_parent_budget=float(weights[:n].sum()),withdrawn_parent_budget=float(parent_weights[oldwithdraw].sum()),new_profile_channel_budget=float(channel_budget),new_profile_budget_authority='sum rows*per_row_weight of actual parent native S02/board ledger; no new multiplier',actual_total_engineering_budget=float(weights.sum()),new_profile_exposures=len(exposures),new_profile_units=list(role_proofs.values()),policy='preserve actual parent per-source L2 budgets; distribute one separate source-normal channel across actual new exposures/role-source-families/dependency units; unchanged original E',source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False,source_source_UV_Query_W_or_E_changed=False,old_2D_point_budget_diluted_by_new_normal_rows=False,physical_Solver_started=False,production_consumer='rinkfit/joint_capture_weights_v3.hpp::load_joint_capture_weights_v3',quality_accepted=False)
 (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-packet',required=True);p.add_argument('--parent-ledger',required=True);p.add_argument('--parent-candidate',required=True);p.add_argument('--packet',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=issue(a.parent_packet,a.parent_ledger,a.parent_candidate,a.packet,a.out);print(json.dumps({k:r[k]for k in ['parent_state_token','raw_rows','active_rows','new_direct_normal_profile_rows','retained_parent_budget','new_profile_channel_budget','actual_total_engineering_budget']},indent=2))
if __name__=='__main__':main()
