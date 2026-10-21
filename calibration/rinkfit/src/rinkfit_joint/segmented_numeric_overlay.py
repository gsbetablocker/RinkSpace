"""Reference the genuinely fitted legacy prefix and explicit new numeric seed.

Raw source columns, eligibility, W and Query remain unchanged. Segment references
avoid another full copy of numeric arrays. Native layout closure is checked again
before constructing the new graph; no AL or physical model is imported.
"""
from pathlib import Path
import copy,json,math,struct,re
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .accepted_aux_prefix import _quoted,_quoted_fields,_cpp_hex
from .dense_packet_delta_io import write_dense_packet
def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p))
def _need(v,s):
 if not v:raise ValueError(s)
def _same(a,b):return struct.pack('<d',float(a))==struct.pack('<d',float(b))
def _verify_complete(path,binding,model,packet,aux,objective):
 n=int(packet['row_count']);base=Path(path).parent;aa={}
 for key in ('admission','fit_role','kind'):
  r=packet['arrays'][key];_need(file_sha256(r['file'])==r['sha256'],'Changed original source '+key);aa[key]=np.memmap(r['file'],dtype=r['dtype'],mode='r',shape=tuple(r['shape']))
 active=np.flatnonzero((aa['admission']!=0)&(aa['fit_role']!=0));stations=[int(i)for i in active if aa['kind'][i]!=1];qrows={int(q['row_index'])for q in packet.get('queries',[])};alphas=[int(i)for i in active if int(i)in qrows]
 h=14695981039346656037
 def put(s):
  nonlocal h
  for v in s.encode('utf8'):h=((h^v)*1099511628211)&0xffffffffffffffff
 with Path(path).open(encoding='utf8')as f:
  def line():
   s=f.readline();_need(bool(s),'Truncated complete observed checkpoint');return s.rstrip('\r\n')
  def count():
   s=line();v=int(s);_need(str(v)==s and 0<=v<=20000000,'Invalid checkpoint count');return v
  _need(line()=='RINKFIT_AL_CHECKPOINT_V1','Expected standard native checkpoint');configuration=_quoted_fields(line());_need(len(configuration)==1,'Missing configuration fingerprint')
  outer=line().split();_need(len(outer)==7 and int(outer[0])==2 and float(outer[4])==0,'No old AL phase/rho may be imported');_need(line()=='0'and line()=='0'and count()==1,'No old AL duals or best state may be imported')
  token=_quoted_fields(line());_need(token==[binding['state_token']],'Complete token mismatch');identity=_quoted_fields(line());_need(len(identity)==6 and identity[0]==binding['source_packet_sha256']and identity[5]==objective,'Observed complete source/objective mismatch');put(' '.join(_quoted(x)for x in identity)+'\n')
  rig_ids=sorted(model['rigs']);expected=len(model['blocks'])+len(stations)+len(active)+len(alphas)+1+len(rig_ids);_need(count()==expected,'Incomplete physical/numeric registry');put(str(expected)+'\n')
  def parameter(name,values=None,size=None):
   text=line();got,end=json.JSONDecoder().raw_decode(text);_need(got==name and text[end:].strip()=='0','Physical/numeric order/flag mismatch '+name)
   fields=line().split();v=[float(x)for x in fields[1:]];_need(int(fields[0])==len(v)and len(v)>0 and all(math.isfinite(x)for x in v),'Nonfinite or malformed checkpoint parameter '+name)
   if values is not None:_need(len(v)==len(values)and all(_same(a,b)for a,b in zip(v,values)),'Checkpoint differs from model/numeric binary '+name)
   else:_need(len(v)==size,'Wrong numerical parameter shape '+name)
   put(_quoted(name)+' 0 '+str(len(v))+''.join(' '+_cpp_hex(x)for x in v)+'\n')
  for b in model['blocks']:parameter(b['name'],b['values'])
  for i in stations:parameter('station/'+str(i),[aux['station'][i]])
  for i in active:parameter('row/'+str(i),[aux['row'][i]])
  for i in alphas:parameter('alpha/'+str(i),[aux['alpha'][i]])
  parameter('budget_sq',size=1)
  for rid in rig_ids:parameter('rig/'+rid+'/numerical_zoom_enclosure',size=2)
  maps=[]
  for _ in range(2):
   size=count();put(str(size)+'\n');d={};last=None
   for k in range(size):
    fields=_quoted_fields(line());_need(len(fields)==2 and fields[0]not in d and(last is None or fields[0]>last),'Invalid discrete registry');last=fields[0];d[last]=fields[1];put(_quoted(last)+' '+_quoted(fields[1])+'\n')
   maps.append(d)
  _need(count()==0 and line()=='END_RINKFIT_AL_CHECKPOINT'and not f.read().strip(),'Old best state or trailing checkpoint bytes')
 _need(f'{h:016x}'==binding['state_token'],'Full FNV physical/numeric/discrete token mismatch')
 _need(maps[0].get('joint_packet_sha256')==binding['source_packet_sha256']and maps[0].get('RS_profile_policy','').startswith('sourceUV_trueRS_capture_v4:') and maps[0].get('fixed_path_table')==packet['schema']and not maps[1],'Observed complete discrete identity mismatch')
 pos=0
 for run in maps[0]['finite_material_paths_RLE_u32_v1'].split(';'):
  if not run:continue
  length,value=map(int,run.split(':'));_need(length>0 and pos+length<=n and np.all(aux['path'][pos:pos+length]==value),'Complete numeric path array differs');pos+=length
 _need(pos==n,'Incomplete numeric path denominator')
 return dict(schema='rinkfit.actual_native_complete_numeric_checkpoint_verification.v1',passed=True,state_token=binding['state_token'],raw_rows=n,active_rows=int(len(active)),physical_blocks=len(model['blocks']),registered_parameters=expected,configuration_sha256=configuration[0],source_identity=identity[0],objective_id=objective,old_AL_duals_imported=False,true_RS_verified=False)

def prepare_segmented_numeric_packet(parent_packet,current_model,prefix_state_dir,prefix_packet,source_geometry_qualification,output):
 out=Path(output).resolve();_need(not out.exists(),'New immutable segmented numeric handoff required')
 packet=read_json(parent_packet);q=read_json(source_geometry_qualification);parent=_ref(parent_packet);_need(q['source_packet']==parent,'Current qualification packet differs')
 state=Path(prefix_state_dir).resolve();prior=read_json(prefix_packet);n0=int(prior['row_count']);n=int(packet['row_count']);_need(int(packet['dense_source_binding']['row_start'])==n0 and n>n0,'Exact legacy prefix/new-source suffix required')
 _need(packet['dense_joint_source_append']['parent_packet']==_ref(prefix_packet),'Legacy source prefix is not exact named raw parent')
 binding_path=state/'source_uv_capture_state_binding.json';binding=read_json(binding_path);_need(binding['schema']=='rinkfit.sourceUV_trueRS_capture_state_binding.v1'and binding['objective_id']=='rinkfit.sourceUV_trueRS_finite_capture.v4'and binding['packet_sha256']==file_sha256(prefix_packet),'Expected actual v21 full source state')
 old_model_path=state/'model.json';old_model=read_json(old_model_path);model=read_json(current_model);_need(file_sha256(old_model_path)==binding['model_sha256'],'Fitted prefix physical file SHA differs')
 _need(sorted(old_model['shots'])==['s01','s02','s03','s04'] and sorted(model['shots'])==['s01','s02','s03','s04','s05','s06'],'Expected first4 source prefix and six-shot target')
 for k in ('source_clock','rigs','materials','scene_block','linear_control_manifolds','measurement_widths'):
  _need(old_model.get(k)==model.get(k),'Prefix projection metadata differs: '+k)
 for sid in old_model['shots']:_need(old_model['shots'][sid]==model['shots'][sid],'Prefix shot metadata differs: '+sid)
 _need(len(old_model['blocks'])==len(model['blocks']),'Prefix/target physical registry differs')
 ignored=[]
 for i,(a,b)in enumerate(zip(old_model['blocks'],model['blocks'])):
  _need(a['name']==b['name'],'Physical semantic registry differs')
  if a!=b:
   _need(bool(re.match(r'^shot/s0[5-9]/',a['name'])),'Changed potentially used prefix projection block: '+a['name']);ignored.append(i)
 manifest_path=state/'explicit_nuisance/manifest.json';manifest=read_json(manifest_path);_need(manifest['packet_sha256']==binding['packet_sha256']and int(manifest['raw_rows'])==n0,'Fitted prefix aux identity differs');arrays={};aux={}
 for key,dtype in [('station','<f8'),('row','<f8'),('alpha','<f8'),('path','<u4')]:
  name=key+('.u32'if key=='path'else'.f64');p=manifest_path.parent/name;r=manifest['files'][name];_need(p.stat().st_size==n0*np.dtype(dtype).itemsize==r['size_bytes']and file_sha256(p)==r['sha256'],'Fitted prefix aux SHA/size differs: '+key)
  arrays[key]=dict(file=str(p),sha256=r['sha256'],dtype=dtype,shape=[n0]);aux[key]=np.memmap(p,dtype=dtype,mode='r',shape=(n0,));_need(np.isfinite(aux[key]).all(),'Nonfinite fitted prefix aux '+key)
 normalized=dict(state_token=binding['state_token'],source_packet_sha256=binding['packet_sha256']);verification=_verify_complete(state/'engineering_checkpoint.json',normalized,old_model,prior,aux,binding['objective_id'])
 out.mkdir(parents=True);write_json(out/'prefix_complete_state_verification.json',verification)
 closure=dict(schema='rinkfit.first4_projection_closure_segment_binding.v1',prefix_model=_ref(old_model_path),current_model=_ref(current_model),clock_rigs_materials_scene_and_first4_metadata_exact=True,prefix_unreferenced_changed_blocks=ignored,prefix_unreferenced_changed_names=[model['blocks'][i]['name']for i in ignored],all_other_blocks_exact=True,existing_forward_identity=_ref(Path(parent_packet).resolve().parents[1]/'model_activity_v1/forward_identity.json'),actual_native_layout_verification_required=True,physical_model_imported=False)
 write_json(out/'projection_closure.json',closure)
 pieces=[dict(kind='complete_native_checkpoint_prefix',start=0,count=n0,source_offset=0,source_packet=_ref(prefix_packet),physical_model=_ref(old_model_path),complete_checkpoint=_ref(state/'engineering_checkpoint.json'),state_binding=_ref(binding_path),explicit_nuisance_manifest=_ref(manifest_path),complete_state_token=binding['state_token'],complete_state_verification=_ref(out/'prefix_complete_state_verification.json'),arrays=arrays),dict(kind='raw_packet_numeric_seed',start=n0,count=n-n0,source_offset=n0,source_packet=parent,arrays={k:packet['arrays'][k]for k in ('station','row','alpha','path')})]
 overlay=dict(schema='rinkfit.segmented_native_numeric_auxiliary_overlay.v1',source_packet=parent,source_geometry_qualification=_ref(source_geometry_qualification),candidate_model=_ref(current_model),raw_rows=n,pieces=pieces,projection_closure=_ref(out/'projection_closure.json'),numerical_initialization_only=True,true_RS_verified=False,source_UV_ID_PTS_qualification_Query_W_unchanged=True,old_AL_duals_imported=False,full_current_checkpoint_claimed=False,issuer=_ref(__file__))
 write_json(out/'numeric_overlay.json',overlay);doc=copy.deepcopy(packet);doc['dense_parent_packet']=parent;doc['dense_numeric_auxiliary_overlay']=_ref(out/'numeric_overlay.json');write_dense_packet(out/'packet.json',doc)
 result=dict(schema='rinkfit.segmented_fitted_prefix_numeric_packet_handoff.v1',packet=_ref(out/'packet.json'),numeric_overlay=_ref(out/'numeric_overlay.json'),candidate_model=_ref(current_model),fitted_prefix_rows=n0,new_numeric_seed_rows=n-n0,raw_rows=n,prefix_complete_state_token=binding['state_token'],original_18_source_array_specs_exact=True,new_numeric_array_files=0,source_pixels_decoded=0,old_AL_or_physical_model_imported=False,current_complete_state_token_must_be_generated_by_native=True,source_qualification_and_weights_rebind_required=True,quality_accepted=False)
 write_json(out/'receipt.json',result);return result
