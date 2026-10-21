"""Streaming standard native enriched-state verification, independent of Solve."""
from pathlib import Path
import json,math,struct
import numpy as np
from .accepted_aux_prefix import _quoted,_quoted_fields,_cpp_hex,OBJECTIVE
from .source_math_identity_v1 import validate_source_math_bindings,digest_text,require

POLICY='explicit bounded station/row/alpha; exact original RS and finite frozen query equalities; independent true RS on export'
def same(a,b):return struct.pack('<d',a)==struct.pack('<d',b)
def verify_source_enriched_checkpoint(path,meta,model,packet,aux,active,query_rows,station_rows=None,*,base=None,verified_dependency_cache=None):
 """Verify genuine initial/fullcandidate metadata; never rewrite a receipt or CP."""
 path=Path(path);base=Path(base or path.parent);raw=meta['source_packet_sha256']
 require(meta['objective_id']==OBJECTIVE,'checkpoint engineering family')
 math_id=validate_source_math_bindings(meta['source_math_bindings'],packet,model,base,raw_packet_sha256=raw,verified_dependency_cache=verified_dependency_cache)
 require(meta['mathematical_source_id']==math_id['mathematical_source_id'],'reported mathematical source identity')
 n=packet['row_count'];require(len(active)==n and all(len(aux[k])==n for k in ('station','row','alpha','path')),'complete explicit auxiliary denominator')
 active_rows=np.flatnonzero(active);
 if station_rows is None:
  from .r_inner_prefix_v1 import array
  kinds=array(packet,'kind',base,verified_dependency_cache);station_rows=[int(i)for i in active_rows if kinds[i]!=1]
 station_rows=list(station_rows);query_rows=set(query_rows);alpha_rows=[int(i)for i in active_rows if int(i)in query_rows]
 h=14695981039346656037
 def put(s):
  nonlocal h
  for v in s.encode('utf8'):h=((h^v)*1099511628211)&0xffffffffffffffff
 extras={};count_parameters=0
 with path.open(encoding='utf8')as f:
  def line():
   x=f.readline()
   if not x:raise ValueError('Truncated source-enriched checkpoint')
   return x.rstrip('\r\n')
  def count():
   x=line();v=int(x);require(str(v)==x and 0<=v<=20000000,'checkpoint count');return v
  require(line()=='RINKFIT_AL_CHECKPOINT_V1','checkpoint format');require(_quoted_fields(line())==[meta['configuration_sha256']],'checkpoint configuration')
  outer=line().split();require(len(outer)==7 and int(outer[0])==2 and float(outer[4])==0 and math.isfinite(float(outer[6]))and float(outer[6])>=0,'old AL phase/rho')
  require(line()=='0'and line()=='0'and count()==1,'old AL duals/current state')
  token=_quoted_fields(line());require(token==[meta['candidate_state_token']],'checkpoint token metadata');identity=_quoted_fields(line())
  require(len(identity)==6 and identity[0]==math_id['mathematical_source_id']and identity[5]==OBJECTIVE,'mathematical source/objective identity')
  require(identity[2]==digest_text('no_common_budget_in_engineering_recovery')and identity[3]==digest_text(raw+'engineering_not_likelihood|'+meta['configuration_sha256']),'engineering budget/statistics identity')
  put(' '.join(_quoted(v)for v in identity)+'\n')
  rigs=model['rigs'];rig_ids=list(rigs)if isinstance(rigs,dict)else[r['id']for r in rigs]
  # Native std::map rigs are loaded in JSON lexical order.
  if isinstance(rigs,dict):rig_ids.sort()
  expected=len(model['blocks'])+len(station_rows)+len(active_rows)+len(alpha_rows)+1+len(rig_ids);size=count();require(size==expected,'full registry denominator');put(str(size)+'\n')
  def parameter(expected_name,expected_values,size=None):
   nonlocal count_parameters
   text=line();name,end=json.JSONDecoder().raw_decode(text);flag=text[end:].strip();require(name==expected_name and flag=='0','parameter registry/order/flag '+expected_name)
   v=line().split();k=int(v[0]);values=[float(x)for x in v[1:]];require(k==len(values)and k>0 and all(math.isfinite(x)for x in values),'parameter finite values '+name)
   if expected_values is not None:require(k==len(expected_values)and all(same(a,float(b))for a,b in zip(values,expected_values)),'physical/explicit auxiliary changed '+name)
   else:require(k==size,'nonphysical parameter shape '+name);extras[name]=values
   put(_quoted(name)+' 0 '+str(k)+''.join(' '+_cpp_hex(x)for x in values)+'\n');count_parameters+=1
  for b in model['blocks']:parameter(b['name'],b['values'])
  for i in station_rows:parameter('station/'+str(i),[aux['station'][i]])
  for i in active_rows:parameter('row/'+str(i),[aux['row'][i]])
  for i in alpha_rows:parameter('alpha/'+str(i),[aux['alpha'][i]])
  parameter('budget_sq',None,1)
  for k in rig_ids:parameter('rig/'+k+'/numerical_zoom_enclosure',None,2)
  maps=[]
  for _ in range(2):
   size=count();put(str(size)+'\n');d={};last=None
   for j in range(size):
    fields=_quoted_fields(line());require(len(fields)==2 and fields[0]not in d and(last is None or last<fields[0]),'map registry/order');last=fields[0];d[last]=fields[1];put(_quoted(last)+' '+_quoted(fields[1])+'\n')
   maps.append(d)
  require(count()==0 and line()=='END_RINKFIT_AL_CHECKPOINT'and not f.read().strip(),'old best state/trailing bytes')
 require(f'{h:016x}'==token[0],'complete FNV state token')
 wanted=dict(math_id['discrete'],joint_packet_sha256=raw,RS_profile_policy=POLICY,fixed_path_table=packet['schema'])
 wanted['finite_material_paths_RLE_u32_v1']=maps[0].get('finite_material_paths_RLE_u32_v1','')
 require(maps[0]==wanted and not maps[1],'fixed complete discrete registry')
 pos=0
 for run in wanted['finite_material_paths_RLE_u32_v1'].split(';'):
  if not run:continue
  length,value=map(int,run.split(':'));require(length>0 and 0<=value<=0xffffffff and pos+length<=n and np.all(aux['path'][pos:pos+length]==value),'complete finite path registry');pos+=length
 require(pos==n,'complete path denominator')
 return dict(schema='rinkfit.source_enriched_complete_checkpoint_verification.v1',passed=True,state_token=token[0],source_packet_sha256=raw,mathematical_source_id=identity[0],complete_parameters=count_parameters,explicit_all_raw_rows=n,source_attachment_discrete=math_id['discrete'],nonphysical_state_verified_not_resumed=extras,old_AL_duals_resumed=False)
