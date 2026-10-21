"""Immutable numeric-only handoff from complete observed-row state to fresh RS.

The source packet columns remain exact. Four existing numeric arrays are referenced,
not copied into another table. This is not a physical or source-quality certificate.
"""
from pathlib import Path
import copy,json,math,struct
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .accepted_aux_prefix import _quoted,_quoted_fields,_cpp_hex
from .dense_packet_delta_io import write_dense_packet
OBJECTIVE='rinkfit.sourceUV_observed_row_finite_initialization.v1'
POLICY='PROVISIONAL sourceUV measured-row initialization: source row fixed, finite station, true RS unverified; fresh v3 revalidation required'
def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p))
def _need(v,s):
 if not v:raise ValueError(s)
def _same(a,b):return struct.pack('<d',float(a))==struct.pack('<d',float(b))
def _verify_complete(path,binding,model,packet,aux):
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
  token=_quoted_fields(line());_need(token==[binding['state_token']],'Complete token mismatch');identity=_quoted_fields(line());_need(len(identity)==6 and identity[0]==binding['source_packet_sha256']and identity[5]==OBJECTIVE,'Observed complete source/objective mismatch');put(' '.join(_quoted(x)for x in identity)+'\n')
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
 _need(maps[0].get('joint_packet_sha256')==binding['source_packet_sha256']and maps[0].get('RS_profile_policy')==POLICY and maps[0].get('fixed_path_table')==packet['schema']and not maps[1],'Observed complete discrete identity mismatch')
 pos=0
 for run in maps[0]['finite_material_paths_RLE_u32_v1'].split(';'):
  if not run:continue
  length,value=map(int,run.split(':'));_need(length>0 and pos+length<=n and np.all(aux['path'][pos:pos+length]==value),'Complete numeric path array differs');pos+=length
 _need(pos==n,'Incomplete numeric path denominator')
 return dict(schema='rinkfit.actual_observed_complete_numeric_checkpoint_verification.v1',passed=True,state_token=binding['state_token'],raw_rows=n,active_rows=int(len(active)),physical_blocks=len(model['blocks']),registered_parameters=expected,configuration_sha256=configuration[0],source_identity=identity[0],objective_id=OBJECTIVE,old_AL_duals_imported=False,true_RS_verified=False)

def prepare_observed_numeric_packet(state_dir,parent_packet,parent_source_qualification,output):
 state=Path(state_dir).resolve();out=Path(output).resolve();_need(not out.exists(),'New immutable numeric handoff output required')
 packet=read_json(parent_packet);qual=read_json(parent_source_qualification);parent=_ref(parent_packet);_need(qual['source_packet']==parent,'Parent qualification binds different packet')
 bpath=state/'observed_row_initialization_state_binding.json';binding=read_json(bpath);_need(binding['schema']=='rinkfit.provisional_measured_row_initialization_state_binding.v1'and binding['objective_id']==OBJECTIVE and binding['source_packet_sha256']==parent['sha256']and binding['true_RS_verified']is False and binding['phase_II_allowed']is False,'Not a complete provisional observed state')
 n=int(packet['row_count']);model=read_json(state/'model.json');manifest_path=state/'explicit_nuisance/manifest.json';manifest=read_json(manifest_path);_need(manifest['packet_sha256']==parent['sha256']and int(manifest['raw_rows'])==n,'Complete four-aux manifest denominator/source mismatch')
 arrays={};aux={}
 for key,dtype in [('station','<f8'),('row','<f8'),('alpha','<f8'),('path','<u4')]:
  name=key+('.u32'if key=='path'else'.f64');p=manifest_path.parent/name;r=manifest['files'][name];_need(p.stat().st_size==n*np.dtype(dtype).itemsize==r['size_bytes']and file_sha256(p)==r['sha256'],'Numeric column SHA/size mismatch '+key)
  arrays[key]=dict(file=str(p),sha256=r['sha256'],dtype=dtype,shape=[n]);aux[key]=np.memmap(p,dtype=dtype,mode='r',shape=(n,));_need(np.isfinite(aux[key]).all(),'Nonfinite observed numeric '+key)
 verification=_verify_complete(state/'engineering_checkpoint.json',binding,model,packet,aux)
 out.mkdir(parents=True);write_json(out/'complete_state_verification.json',verification)
 overlay=dict(schema='rinkfit.observed_row_numeric_auxiliary_overlay.v1',source_packet=parent,source_geometry_qualification=_ref(parent_source_qualification),candidate_model=_ref(state/'model.json'),observed_checkpoint=_ref(state/'engineering_checkpoint.json'),observed_state_binding=_ref(bpath),explicit_nuisance_manifest=_ref(manifest_path),complete_state_token=binding['state_token'],raw_rows=n,arrays=arrays,numerical_initialization_only=True,true_RS_verified=False,source_UV_ID_PTS_qualification_Query_W_unchanged=True,old_AL_duals_imported=False,issuer=_ref(__file__),complete_state_verification=_ref(out/'complete_state_verification.json'))
 write_json(out/'numeric_overlay.json',overlay);doc=copy.deepcopy(packet);doc['dense_parent_packet']=parent;doc['dense_numeric_auxiliary_overlay']=_ref(out/'numeric_overlay.json');write_dense_packet(out/'packet.json',doc)
 result=dict(schema='rinkfit.observed_numeric_only_packet_handoff.v1',packet=_ref(out/'packet.json'),numeric_overlay=_ref(out/'numeric_overlay.json'),candidate_model=overlay['candidate_model'],source_geometry_qualification=overlay['source_geometry_qualification'],original_18_source_arrays_bitexact=True,source_raw_rows=n,complete_state_token=binding['state_token'],source_pixels_decoded=0,source_generation_changed=False,true_RS_verified=False,old_AL_duals_imported=False,qualification_and_weights_rebinding_required=True)
 write_json(out/'receipt.json',result);return result
