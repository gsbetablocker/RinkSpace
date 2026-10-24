"""Select actual-state primitive branches and run the compact native derivative probe.

Only small row fixtures and summaries are saved; source arrays are read through
memory maps. This is a numerical diagnostic, never source or visual acceptance.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def select(args):
 model=read(args.model);cfg=read(args.config);packet=read(args.packet);arrays={};chain=[];here=Path(args.packet)
 while True:
  j=read(here);chain.append((here,j))
  if 'dense_parent_packet' not in j:break
  ref=j['dense_parent_packet'];here=Path(ref['file']);assert digest(here)==ref['sha256']
 for location,j in reversed(chain):
  for name,ref in j.get('arrays',{}).items():
   d=dict(ref);f=Path(d['file']);d['file']=str(f if f.is_absolute() else location.parent/f);arrays[name]=d
 needed=('admission','fit_role','material_index','shot_index','native_pts','uv','E','kind');a={}
 for name in needed:
  d=arrays[name];assert digest(d['file'])==d['sha256'],name
  a[name]=np.memmap(d['file'],mode='r',dtype=d.get('dtype','<f8'),shape=tuple(d['shape']),offset=d.get('offset_bytes',0))
 n=len(a['kind']);state=Path(args.state)/'explicit_nuisance';st=np.memmap(state/'station.f64',mode='r',dtype='<f8',shape=(n,));row=np.memmap(state/'row.f64',mode='r',dtype='<f8',shape=(n,));path=np.memmap(state/'path.u32',mode='r',dtype='<u4',shape=(n,))
 materials=model['materials'];shape={(mi,pi):p for mi,m in enumerate(materials) for pi,p in enumerate(m['paths'])}
 # The same production restoration is replayed natively, without reseeding values.
 restore=cfg.get('runtime_shared_parameter_stage_v2',{}).get('restore_original_print_vertices')
 if restore:
  assert digest(restore['file'])==restore['sha256']
  for u in read(restore['file'])['material_path_updates']:shape[u['material_index'],u['path_index']]=u['expected_path']
 maxpaths=max(len(m['paths'])for m in materials);den=np.zeros((len(materials),maxpaths),dtype=np.int32);board=np.zeros_like(den,dtype=bool)
 for (mi,pi),p in shape.items():
  if p['kind']==2:den[mi,pi]=max(0,len(p['points'])-1)
  elif p['kind']==6:
   size=len(p.get('geometry_indices',[]))or len(model['blocks'][p['geometry_block']]['values'])-p.get('geometry_offset',0);count=size//2;den[mi,pi]=count if p.get('closed')else count-1
  board[mi,pi]=p['kind']==3
 selected={};dtype=[('material','<u4'),('path','<u4'),('shot','<u2'),('zone','u1'),('board_part','u1')];active_count=0
 for start in range(0,n,65536):
  stop=min(n,start+65536);keep=(a['admission'][start:stop]==1)&(a['fit_role'][start:stop]==1);ids=np.flatnonzero(keep)+start;active_count+=len(ids)
  mi=a['material_index'][ids];pi=path[ids];s=st[ids];d=den[mi,pi];x=s*d;zone=np.full(len(ids),3,dtype=np.uint8);zone[(d>0)&(np.abs(x-np.rint(x))<=16*np.spacing(np.maximum(1.,np.abs(x))))]=2;zone[s==0]=0;zone[s==1]=1
  parts=np.where(board[mi,pi],np.minimum(7,np.maximum(0,np.floor(s*8))).astype(np.uint8),0)
  keys=np.empty(len(ids),dtype=dtype);keys['material']=mi;keys['path']=pi;keys['shot']=a['shot_index'][ids];keys['zone']=zone;keys['board_part']=parts
  unique,first=np.unique(keys,return_index=True);_,last=np.unique(keys[::-1],return_index=True);last=len(keys)-1-last
  for k,lo,hi in zip(unique,first,last):
   key=tuple(int(k[name])for name,_ in dtype);left,right=int(ids[lo]),int(ids[hi]);old=selected.get(key);selected[key]=[min(left,old[0])if old else left,max(right,old[1])if old else right]
 shots=sorted(model['shots']);ids=sorted({i for pair in selected.values()for i in pair});rows=[]
 for i in ids:
  rows.append(dict(raw_row=i,shot=shots[int(a['shot_index'][i])],pts=int(a['native_pts'][i]),material=int(a['material_index'][i]),path=int(path[i]),kind=int(a['kind'][i]),uv=a['uv'][i].tolist(),E=float(a['E'][i]),station=float(st[i]),row=float(row[i])))
 fixture=dict(schema='rinkfit.actual_state_primitive_branch_fixture.v1',source_packet=str(Path(args.packet).resolve()),source_packet_sha256=digest(args.packet),state=str(Path(args.state).resolve()),raw_rows=n,active_rows=active_count,selection='First and last actual row of each material/path/shot/station interior-vertex-endpoint/continuous-board segment branch; no residual-based selection.',structural_branches=len(selected),material_paths=len({(r['material'],r['path'])for r in rows}),rows=rows)
 save(Path(args.out)/'fixture.json',fixture);print(json.dumps({k:v for k,v in fixture.items()if k!='rows'}),flush=True)

def build_and_run(args):
 project=Path(__file__).resolve().parents[3];recipe=read(Path(args.build)/'recipe.json');assert recipe['returncode']==0
 spec=importlib.util.spec_from_file_location('rinkfit_existing_build_environment',project/'calibration/rinkfit/cpp/gpu_rows/build.py');env_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(env_module);os.environ.setdefault('CUDA_PATH',r'D:/Program Files/CUDA');env=env_module.environment();env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env['PATH'];out=Path(args.out);env['TMP']=env['TEMP']=str(out)
 source=Path(__file__).with_name('joint_gpu_state_probe.cpp');obj=out/'probe.obj';exe=out/'probe.exe';cmd=next(c['command']for c in recipe['commands']if '/c'in c.get('command',[]));at=cmd.index('/c');cmd=cmd[:at]+['/c',str(source),'/Fo:'+str(obj)]
 r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/'build.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0,r.stdout+r.stderr
 link=next(c['command']for c in reversed(recipe['commands'])if any(x.startswith('/Fe:')for x in c.get('command',[])));link=[str(obj)if x.endswith('joint_fit.obj')else '/Fe:'+str(exe)if x.startswith('/Fe:')else x for x in link]
 r=subprocess.run(link,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/'link.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0,r.stdout+r.stderr
 assert obj.resolve().parent==out.resolve();obj.unlink()
 cmd=[str(exe),str(args.model),str(out/'fixture.json'),str(args.ptx),str(args.config),str(out)]
 r=subprocess.run(cmd,cwd=project,env=env,capture_output=True,text=True,errors='replace');(out/'run.log').write_text(r.stdout+r.stderr,encoding='utf-8');print(r.stdout[:2500],r.stderr[-2000:],flush=True);save(out/'execution.json',dict(returncode=r.returncode,command=cmd,source_sha256=digest(source),executable_sha256=digest(exe),object_removed=True,linked_current_recipe=str(Path(args.build)/'recipe.json')));return r.returncode

def main():
 p=argparse.ArgumentParser();[p.add_argument('--'+k,required=True,type=Path)for k in ('model','packet','state','config','ptx','build','out')];args=p.parse_args();args.out.mkdir(exist_ok=True);select(args);return build_and_run(args)
if __name__=='__main__':sys.exit(main())
