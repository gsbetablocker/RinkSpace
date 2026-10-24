"""Current-state sampled finite-material/strict-RS numerical station trust.
No source budget, material, statistical operator or physical bound is changed.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(r,base=None):
 p=Path(r.get('path',r.get('file','')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=r['sha256']:raise ValueError('Changed complete-state authority '+str(p))
 return p.resolve()
def array(r,base):return np.fromfile(verify(r,base),r['dtype']).reshape(r['shape'])

def run(transport,output,python_package,compiled_binding=None):
 begin=time.monotonic();T=Path(transport).resolve();tr=read(T);CP=verify(tr['checkpoint']);cp=read(CP);MP=verify(tr['model']);model=read(MP);SP=verify(tr['semantic']);sem=read(SP);PK=verify(tr['packet']);pk=read(PK)
 if cp['model']['sha256']!=sha(MP):raise ValueError('Mixed model and complete state')
 O=Path(output).resolve();O.mkdir(exist_ok=False);package=Path(python_package).resolve();sys.path.insert(0,str(package))
 from rinkfit_start.native_scene import NativeScene,camera_from_event
 from rinkfit_start.material_foot import _certified_candidates
 scene=NativeScene(model);binding_ref=None
 if compiled_binding:
  CB=Path(compiled_binding).resolve();sys.path.insert(0,str(CB.parent));from bound_camera_sensor import bind_camera_from_model
  camera=bind_camera_from_model(ref(MP),model=model);binding_ref=ref(CB)
  camera.at.snapshot(O/'camera_snapshot',dict(model=ref(MP),checkpoint=ref(CP),transport=ref(T),semantic=ref(SP),packet=ref(PK)))
 else:camera=camera_from_event(model)
 N=int(pk['observations']);active=array(sem['active_source_mask'],SP.parent).astype(bool);kind=array(sem['arrays']['kind'],SP.parent);mi=array(sem['resolved_material_index'],SP.parent);pts=array(pk['arrays']['native_pts'],PK.parent);times=(pts-256256)/12800.;E=array(pk['arrays']['engineering_budget_px'],PK.parent)
 if active.shape!=(N,) or cp['active_rows']!=int(active.sum()):raise ValueError('Current complete source ledger')
 S=array(cp['row_state_files']['station'],CP.parent);B=array(cp['row_state_files']['branch'],CP.parent)
 for name in ['row','alpha']:array(cp['row_state_files'][name],CP.parent)
 vertex=array(tr['arrays']['vertex'],T.parent).astype(bool);status=array(tr['arrays']['status'],T.parent);L=array(tr['arrays']['station_delta_lower_right'],T.parent);U=array(tr['arrays']['station_delta_upper_right'],T.parent);tan=array(tr['arrays']['station_UV_right'],T.parent).reshape(N,2);speed=np.linalg.norm(tan,axis=1)
 eligible=active&(kind==0)&(status==0)&~vertex&(U-L>1e-15)&(speed>0);ids=np.flatnonzero(eligible);ss=S[ids];lo=ss+L[ids];hi=ss+U[ids];h0=(hi-lo)/128.;base=_certified_candidates(scene,camera,mi[ids],B[ids],ss,times[ids]) if len(ids) else dict(valid=np.empty(0,bool));curvature=np.zeros(len(ids));vc=np.zeros(len(ids),np.uint8);nc=np.zeros(len(ids),np.uint8);nl=nc.copy();nr=nc.copy();history=[]
 for fraction in ([.25,.5,1.] if len(ids) else []):
  h=h0*fraction;rr={}
  for sign in [-2,-1,1,2]:
   st=ss+sign*h;inside=(st>=lo)&(st<=hi);st=np.where(inside,st,ss);z=_certified_candidates(scene,camera,mi[ids],B[ids],st,times[ids]);z['valid']&=inside&base['valid'];rr[sign]=z
  central=rr[-1]['valid']&rr[1]['valid'];left=~central&rr[-1]['valid']&rr[-2]['valid'];right=~central&~left&rr[1]['valid']&rr[2]['valid'];good=central|left|right;d2=np.zeros((len(ids),2))
  d2[central]=(rr[-1]['uv'][central]-2*base['uv'][central]+rr[1]['uv'][central])/(h[central,None]**2)
  d2[left]=(base['uv'][left]-2*rr[-1]['uv'][left]+rr[-2]['uv'][left])/(h[left,None]**2)
  d2[right]=(base['uv'][right]-2*rr[1]['uv'][right]+rr[2]['uv'][right])/(h[right,None]**2)
  norm=np.linalg.norm(d2,axis=1);good&=np.isfinite(norm);norm[~good]=0;curvature=np.maximum(curvature,norm);vc+=good;nc+=central&good;nl+=left&good;nr+=right&good
  rec=dict(fraction=fraction,legal_scales=int(good.sum()),central=int((central&good).sum()),left=int((left&good).sum()),right=int((right&good).sum()),elapsed_seconds=time.monotonic()-begin);history.append(rec);print('CURRENT_RS_STATION_CURVATURE',json.dumps(rec),flush=True)
 remainder=.05*E[ids];safety=2.;raw_radius=np.sqrt(np.divide(2*remainder,safety*curvature,out=np.full(len(ids),np.inf),where=curvature>0));radius=np.minimum(raw_radius,np.maximum(-L[ids],U[ids]));unknown=(vc<2)|(curvature<=0)|(~np.isfinite(radius));radius[unknown]=0;legacy_radius=np.minimum(radius,10./speed[ids]);fullLegacy=np.zeros(N);fullLegacy[ids]=legacy_radius;sampleStatus=np.full(N,1,np.uint8);sampleStatus[ids]=0;sampleStatus[ids[vc<2]]=2;sampleStatus[ids[(curvature<=0)|(~np.isfinite(raw_radius))]]=3
 fullradius=np.zeros(N);fullradius[ids]=radius;fullC=np.zeros(N);fullC[ids]=curvature;fullCount=np.zeros(N,np.uint8);fullCount[ids]=vc;arrays={}
 for name,value,dtype in [('radius',fullradius,'<f8'),('legacy_10px_radius',fullLegacy,'<f8'),('sampled_curvature_envelope',fullC,'<f8'),('valid_scales',fullCount,'|u1'),('sample_status',sampleStatus,'|u1')]:
  p=O/(name+'.bin');np.asarray(value,dtype=dtype).tofile(p);arrays[name]=dict(file=p.name,dtype=dtype,shape=[N],sha256=sha(p))
 modules=[ref(package/'rinkfit_start'/x)for x in ['native_scene.py','material_foot.py','native_projection.py']if(package/'rinkfit_start'/x).exists()]
 out=dict(schema='rinkfit.original_station_numerical_TR_native_RS_curvature.v3',checkpoint=ref(CP),transport=ref(T),model=ref(MP),packet=ref(PK),semantic=ref(SP),complete_auxiliary_sha256={k:cp['row_state_files'][k]['sha256']for k in ['station','branch','row','alpha']},ordinary_candidates=len(ids),radius_positive=int(np.count_nonzero(radius)),unknown_curvature_S_fixed=int(np.count_nonzero(radius==0)),at_least_2_legal_strict_RS_stencils_required=True,base_strict_RS_invalid_rows=ids[~base['valid']].tolist(),stencil_rule='Current same finite chart width/128 times .25,.5,1; strict RS central or one-sided three-point second derivative.',maximum_first_order_motion_px=None,legacy_10px_cap_diagnostic_only=True,main_radius_uses_only_current_C2_and_finite_cell=True,sample_status_codes={'0':'at_least_two_valid_scales_and_positive_finite_C2','1':'not_an_eligible_smooth_ordinary_chart','2':'fewer_than_two_legal_scales','3':'zero_or_nonfinite_sampled_C2_unknown_not_infinite'},zero_C2_or_insufficient_scales_radius_zero_is_temporary_numerical_fallback=True,numerical_remainder_fraction_of_E=.05,curvature_safety_factor=safety,sampled_curvature_is_not_continuous_global_certificate=True,original_E_W_source_unchanged=True,no_new_DOF_prior_or_permanent_physical_bound=True,total_deltaS_seed_plus_eta_must_obey_radius=True,actual_fullsource_nonlinear_gates_still_required=True,arrays=arrays,history=history,code=ref(__file__),python_modules=modules,compiled_binding=binding_ref,elapsed_seconds=time.monotonic()-begin,diagnostic_rows={str(i):dict(radius=float(fullradius[i]),curvature=float(fullC[i]),legal_scales=int(fullCount[i]),original_cell=[float(L[i]),float(U[i])])for i in [9,17263,17089,25270,17147] if i<N})
 (O/'manifest.json').write_text(json.dumps(out,indent=2));print('CURRENT_RS_CURVATURE_DONE',json.dumps(dict(manifest=ref(O/'manifest.json'),positive=out['radius_positive'],elapsed=out['elapsed_seconds'],diagnostic_rows=out['diagnostic_rows'])),flush=True);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--transport',required=True);p.add_argument('--output',required=True);p.add_argument('--python-package',required=True);p.add_argument('--compiled-binding');a=p.parse_args();run(a.transport,a.output,a.python_package,a.compiled_binding)
