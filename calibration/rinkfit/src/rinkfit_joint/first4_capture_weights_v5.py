"""Source-only candidate budgets; final issue requires an actually selected full CP.
Engineering L2 only. No source sigma, W or noise independence is inferred.
"""
from pathlib import Path
from collections import defaultdict
import importlib.util,json,hashlib,tempfile
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .first4_capture_weights_v3 import ref,key_digest,load
from .board_middle_append_v1 import require

SCHEMA='rinkfit.explicit_engineering_source_weights.v5'
PREP='rinkfit.source_only_ice_shape_engineering_weight_preparation.v1'

def checked(r):
 p=Path(r['file']);require(p.is_absolute()and file_sha256(p)==r['sha256'],'Changed actual budget/source authority '+str(p));return p

def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def prefix_module():
 p=Path(__file__).with_name('capture_primal_prefix_v5.py');s=importlib.util.spec_from_file_location('rinkfit_joint._ice_shape_capture_prefix_v5_weight_consumer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(parent_packet,parent_explicit_weights,canonical_source_packet,authority_pixel_ledger,output):
 pp,wp,npk,ap,out=map(lambda p:Path(p).resolve(),(parent_packet,parent_explicit_weights,canonical_source_packet,authority_pixel_ledger,output));require(not out.exists(),'Immutable weight preparation exists')
 inspection=prefix_module().inspect_source_successor(npk,pp);p=read_json(pp);q=read_json(npk);a=load(p,pp.parent);b=load(q,npk.parent);n,N=p['row_count'],q['row_count'];A=(a['admission']==1)&(a['fit_role']==1);B=(b['admission']==1)&(b['fit_role']==1);new=np.flatnonzero(B&(np.arange(N)>=n));old=read_json(wp)
 require(old['schema']=='rinkfit.explicit_engineering_source_weights.v4'and old['source_packet']==ref(pp)and old['old_budget_bits_preserved']and old['old_information_class_bits_preserved'],'Actual v4 source budget required')
 vectors={}
 for k,dt in [('weights.f64','<f8'),('information_class.u8','u1'),('active.u8','u1')]:
  r=old['files'][k];checked(r);v=np.fromfile(r['file'],dtype=dt);require(v.shape==(n,),'Parent weight denominator differs');vectors[k]=v
 pw=vectors['weights.f64'];pc=vectors['information_class.u8'];pa=vectors['active.u8'];require(np.array_equal(pa,A)and np.array_equal(pw>0,A)and np.isfinite(pw).all(),'Actual inherited v4 weight partition differs')
 authority=read_json(ap);require(authority['schema']=='rinkfit.engineering_pixel_contributions.v1'and not authority['original_W_likelihood_consumed']and not authority['engineering_scale_is_sigma'],'Original engineering authority differs')
 inherited_authority=checked(old['budget_authority_pixel_ledger']);require(inherited_authority==ap,'New authority differs from original v4 authority')
 records=inspection['qualified_rows'];byrow={r['row_index']:r for r in records};units={};groups=q['engineering_pixel_dependency_groups']['groups']
 for j,g in enumerate(groups):
  for i in g['rows']:
   require(i not in units,'Duplicate actual RGB closure membership');units[i]=(j,g['id'])
 require(all(int(i)in units for i in new),'New source lacks actual RGB dependence group')
 channels=defaultdict(list)
 for i in new:
  r=byrow[int(i)];family='ice'if r['path_kind']==1 else'advertisement';require((family=='ice'and r['shot_id']in('s02','s04'))or(family=='advertisement'and r['shot_id']=='s02'and r['target_id']=='Ad3.print.component000'),'Unexpected source role/channel');channels[(r['shot_id'],family)].append(int(i))
 w=np.zeros(N,dtype='<f8');w[:n]=pw;cl=np.zeros(N,dtype='u1');cl[:n]=pc;dependency=np.full(N,0xffffffff,dtype='<u4');channel_id=np.zeros(N,dtype='u1');authority_proofs=[];descriptors=[];totals={}
 for ch,rr in sorted(channels.items()):
  sid,family=ch;ag=[dict(group_index=j,**g)for j,g in enumerate(authority['groups'])if g['shot']==sid and g['family']==family];require(ag and all(g['per_row_weight']is not None and np.isfinite(g['per_row_weight'])and g['per_row_weight']>0 for g in ag),'Relevant original engineering budget missing')
  frames=sorted(set(byrow[i]['native_pts']for i in rr));origframes=sorted(set(g['native_pts']for g in ag));full=sum(g['rows']*g['per_row_weight']for g in ag);mean=full/len(origframes);frame_budget={}
  for pt in frames:
   exactgroups=[g for g in ag if int(g['native_pts'])==pt];value=sum(g['rows']*g['per_row_weight']for g in exactgroups)if exactgroups else mean;frame_budget[pt]=value
   authority_proofs.append(dict(shot=sid,family=family,native_pts=pt,budget=value,method='exact_original_same_exposure_family_sum'if exactgroups else'original_same_shot_family_total_divided_by_actual_original_exposure_count',original_family_budget=full,original_exposure_count=len(origframes),original_exposure_mean=mean,authority_group_indices=[g['group_index']for g in(exactgroups or ag)],authority_groups=[dict(group_index=g['group_index'],id=g['id'],rows=g['rows'],native_pts=g['native_pts'],per_row_weight=g['per_row_weight'])for g in(exactgroups or ag)],arbitrary_multiplier_applied=False))
  c=5 if family=='ice'else 6;totals[sid+'/'+family]=sum(frame_budget.values());layers=defaultdict(set);deps=defaultdict(set);local=defaultdict(list)
  for i in rr:
   r=byrow[i];pt=r['native_pts'];role=r['target_id'];fam=r['source_family_id'];j,unit=units[i];layers[pt].add((role,fam));deps[pt,role,fam].add(unit);local[pt,role,fam,unit].append(i);cl[i]=c;channel_id[i]=c;dependency[i]=j
  for key,rows in sorted(local.items()):
   pt,role,fam,unit=key;weight=frame_budget[pt]/len(layers[pt])/len(deps[pt,role,fam])/len(rows);w[rows]=weight;descriptors.append(dict(shot=sid,native_pts=pt,source_role=role,family=family,source_family_id=fam,source_dependency_id=unit,source_dependency_index=int(dependency[rows[0]]),information_class=c,actual_row_indices=rows,rows=len(rows),per_row_weight=weight,source_uncertainty_calibrated=False,RGB_noise_independence_claimed=False))
 require(exact(w[:n],pw)and exact(cl[:n],pc)and np.array_equal(w>0,B)and np.isfinite(w).all(),'Inherited bits or complete active partition changed')
 require(all(np.isclose(w[[i for i in channels[ch]]].sum(),totals[ch[0]+'/'+ch[1]],rtol=1e-13,atol=1e-10)for ch in channels),'Actual exposure/channel budget does not close')
 out.mkdir(parents=True);files={}
 for name,x in [('weights.f64',w),('active.u8',B.astype('u1')),('information_class.u8',cl),('new_dependency.u32',dependency),('new_channel.u8',channel_id)]:
  x.tofile(out/name);files[name]=dict(**ref(out/name),dtype=x.dtype.str,shape=list(x.shape))
 report=dict(schema=PREP,eventual_production_schema=SCHEMA,issuer=ref(__file__),prefix_consumer=ref(Path(__file__).with_name('capture_primal_prefix_v5.py')),source_packet=ref(npk),parent_packet=ref(pp),parent_explicit_source_weights=ref(wp),budget_authority_pixel_ledger=ref(ap),parent_rows=n,raw_rows=N,parent_active_rows=int(A.sum()),active_rows=int(B.sum()),new_direct_normal_profile_rows=len(new),new_active_path_kind_counts=inspection['new_active_path_kind_counts'],source_prefix_inspection=inspection,source_key_registry_sha256=key_digest(q,b),parent_prefix_key_registry_sha256=key_digest(p,a),source_E_binary64_le_sha256=hashlib.sha256(np.asarray(b['E'],dtype='<f8').tobytes()).hexdigest(),source_UV_binary64_le_sha256=hashlib.sha256(np.asarray(b['uv'],dtype='<f8').tobytes()).hexdigest(),files=files,information_classes={**old['information_classes'],'5':'new actual ice ring/layer normal-profile source; engineering only','6':'recovered original Ad3 logo boundary source; engineering only'},retained_parent_budget=float(w[:n].sum()),new_channel_budgets=totals,new_engineering_budget=float(w[new].sum()),actual_total_engineering_budget=float(w.sum()),budget_authority_rules=authority_proofs,new_profile_units=descriptors,old_budget_bits_preserved=True,old_information_class_bits_preserved=True,old_2D_point_budget_diluted_by_new_normal_rows=False,original_same_RGB_closure_consumed_for_new_unit_partition=True,old_to_new_RGB_dependence_preserved_in_source_groups=True,original_W_likelihood_consumed=False,source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False,actual_blur_interval_is_not_sigma=True,source_source_UV_Query_W_or_E_changed=False,prototype_only=True,numeric_parent_bound=False,dispatchable=False,physical_Solver_started=False,quality_accepted=False)
 write_json(out/'preparation.json',report);return report

def validate_preparation(preparation_manifest):
 prep=read_json(preparation_manifest);require(prep['schema']==PREP and prep['issuer']==ref(__file__)and prep['prefix_consumer']==ref(Path(__file__).with_name('capture_primal_prefix_v5.py')),'Preparation consumer changed')
 for key in ('parent_packet','parent_explicit_source_weights','source_packet','budget_authority_pixel_ledger'):checked(prep[key])
 for r in prep['files'].values():checked(r)
 with tempfile.TemporaryDirectory(prefix='verify_weight_v5_',dir=Path(preparation_manifest).parent)as td:
  actual=prepare(prep['parent_packet']['file'],prep['parent_explicit_source_weights']['file'],prep['source_packet']['file'],prep['budget_authority_pixel_ledger']['file'],Path(td)/'recomputed')
  for name,r in actual['files'].items():
   stored=prep['files'][name];require(all(r[k]==stored[k]for k in('sha256','dtype','shape')),'Actual per-row budget/class/dependency bits differ '+name)
  expected={**actual,'files':prep['files']};require(expected==prep,'Actual engineering budget/exposure/RGB authority changed')
 return dict(schema='rinkfit.actual_ice_shape_engineering_weight_verification.v1',passed=True,preparation=ref(preparation_manifest),all_parent_weight_and_class_bits_preserved=True,actual_exposure_authority_recomputed=True,source_sigma_or_W_issued=False,numeric_parent_bound=False,quality_accepted=False)

def validate_parent_ledger(parent_pixel_ledger,parent_explicit_weights,parent_packet):
 ledger=read_json(parent_pixel_ledger);p=read_json(parent_packet);require(ledger['schema']=='rinkfit.engineering_pixel_contributions.v1'and ledger['active_rows']==p['active_rows']and not ledger['original_W_likelihood_consumed'],'Actual parent engineering ledger differs');lr=ledger.get('explicit_source_weights_v4',{}).get('binding');require(lr is not None and checked(lr).resolve()==Path(parent_explicit_weights).resolve()and lr['sha256']==file_sha256(parent_explicit_weights),'Parent ledger does not consume exact inherited v4 budget')
 pw=read_json(checked(lr));oldw=np.fromfile(checked(pw['files']['weights.f64']),'<f8');seen=set();reconstructed=np.zeros(p['row_count'],dtype='<f8');A=load(p,Path(parent_packet).parent);active=(A['admission']==1)&(A['fit_role']==1)
 for g in ledger['groups']:
  require(g['per_row_weight']is None and not g['uniform_group_formula_consumed']and len(g['actual_row_indices'])==g['rows']==len(g['actual_per_row_weight']),'Actual parent ledger arrays absent')
  for i,w in zip(g['actual_row_indices'],g['actual_per_row_weight']):
   require(i not in seen and 0<=i<len(reconstructed)and active[i]and np.isfinite(w)and w>0,'Parent actual weight ledger duplicated/invalid');seen.add(i);reconstructed[i]=w
 require(seen==set(np.flatnonzero(active).tolist())and exact(oldw,reconstructed),'Actual selected parent per-row budgets differ')
 return dict(schema='rinkfit.actual_v4_parent_engineering_ledger_bit_verification.v1',passed=True,parent_packet=ref(parent_packet),parent_explicit_weights=ref(parent_explicit_weights),actual_pixel_ledger=ref(parent_pixel_ledger),rows=p['row_count'],active_rows=int(active.sum()),all_per_row_weights_bitexact=True)

def issue(preparation_manifest,parent_pixel_ledger,parent_candidate,next_bound_packet,selection_manifest,output):
 preparation_verification=validate_preparation(preparation_manifest);prep=read_json(preparation_manifest);out=Path(output).resolve();require(not out.exists(),'Immutable actual issue exists')
 pm=prefix_module();parent_path=checked(prep['parent_packet']);nextpath=Path(next_bound_packet).resolve();cp=Path(parent_candidate).resolve();selected=pm.validate_selection(selection_manifest,parent_path,cp);verified=pm.assert_capture_prefix_current(nextpath);q=read_json(nextpath);canonical=read_json(checked(prep['source_packet']));require(q['source_capture_complete_state_binding']['state_token']==selected['state_token']and q['source_capture_complete_state_binding']['canonical_source_packet']==prep['source_packet'],'New weights do not bind selected real source prefix')
 a=load(q,nextpath.parent);b=load(canonical,Path(prep['source_packet']['file']).parent)
 for k in b:
  if k not in('station','path','row','alpha'):require(exact(a[k],b[k]),'Source preparation does not match actual bound source '+k)
 require(key_digest(q,a)==prep['source_key_registry_sha256'],'Actual source key registry changed')
 ledger_verification=validate_parent_ledger(parent_pixel_ledger,checked(prep['parent_explicit_source_weights']),parent_path)
 for r in prep['files'].values():checked(r)
 out.mkdir(parents=True);report={**prep,'schema':SCHEMA,'source_weight_preparation':ref(preparation_manifest),'source_packet':ref(nextpath),'canonical_source_packet':prep['source_packet'],'parent_pixel_ledger':ref(parent_pixel_ledger),'parent_complete_binding':selected['capture_state_binding'],'parent_complete_checkpoint':selected['checkpoint'],'parent_model':selected['model'],'parent_state_token':selected['state_token'],'root_selection':ref(selection_manifest),'bound_prefix_guard':verified,'source_weight_preparation_verification':preparation_verification,'actual_selected_parent_weight_ledger_verification':ledger_verification,'parent_complete_auxiliary_identity_delegated_to':'capture_primal_prefix_v5','prototype_only':False,'numeric_parent_bound':True,'dispatchable':True,'production_consumer':'rinkfit/joint_capture_weights_v5.hpp::load_joint_capture_weights_v5'}
 write_json(out/'manifest.json',report);return report
