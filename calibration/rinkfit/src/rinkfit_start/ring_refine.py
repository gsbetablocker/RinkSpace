from pathlib import Path
import sys,json,copy,time,argparse,hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import lstsq
from scipy.optimize import minimize
from rinkfit_start.regression import NativeRegression
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture
from rinkfit_start.cross_ring import load_frame,fit,distances
ap=argparse.ArgumentParser(description='Existing physical circle-block proposal against the unchanged original W, with strict sensor endpoint capture. Optional cross-view constraints are projective diagnostics, not physical true-RS cameras.');ap.add_argument('--run',required=True);ap.add_argument('--semantic',required=True);ap.add_argument('--model',required=True);ap.add_argument('--seed',required=True);ap.add_argument('--output',required=True);ap.add_argument('--blocks',nargs='+',required=True);ap.add_argument('--labels',nargs='+',required=True);ap.add_argument('--pts-origin',type=int,required=True);ap.add_argument('--pts-per-second',type=float,required=True);ap.add_argument('--cross-frames',nargs='+',default=[]);ap.add_argument('--ratio-block');ap.add_argument('--cross-target');ap.add_argument('--preserve-label-rms',action='store_true',help='Diagnostic acceptance constraints only: each requested label retains its baseline UV RMS; does not change the original W objective.');a=ap.parse_args()
if a.pts_per_second<=0:raise ValueError('Native PTS scale must be positive')
R=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
model=json.loads(Path(a.model).read_text());seed=Path(a.seed);sem=Path(a.semantic);reg=NativeRegression(R,sem)
sdoc=reg.sem
from rinkfit_start.regression import typed
material=typed(sdoc['resolved_material_index'],sem.parent)
N=reg.n;S=np.fromfile(seed/'station.f64','<f8');B=np.fromfile(seed/'path.u32','<u4');V=np.fromfile(seed/'initial_row.f64','<f8')
scene=NativeScene(model);cam=camera_from_event(model);times=(reg.pts-a.pts_origin)/a.pts_per_second
hit=reg.active&np.isin(reg.labels,a.labels);ids=np.flatnonzero(hit)
assert np.all(reg.kind[ids]==0)
obs=reg.uv[ids];mi=material[ids];br=B[ids];tt=times[ids];stations=S[ids].copy()
blocks=[i for i,b in enumerate(model['blocks'])if b['name']in a.blocks]
for i in blocks:scene.values[i]=scene.values[i].copy()
cols=[(i,j)for i in blocks for j in range(len(model['blocks'][i]['values']))];base=np.array([scene.values[i][j]for i,j in cols]);scale=np.full(len(cols),1e-4)
assert len(blocks)==len(a.blocks)and len(ids),'Requested existing circle blocks/labels absent'
assert Path(a.model).read_bytes()==(seed/'model.json').read_bytes(),'Complete seed and model identity must agree'
baseline_summary,_,_,_=reg.evaluate(seed);ceiling=baseline_summary['all']['max']
loc=np.full(N,-1,int);loc[ids]=np.arange(len(ids));eall=np.load(seed/'candidate_predictions.npz')['error_native_uv']
rows=[];cc=[];vv=[];offset=[];cursor=0
for gid,rr,W in reg.units:
 if not np.any(hit[rr]):continue
 touch=np.flatnonzero(hit[rr]);fixed=np.flatnonzero(~hit[rr]);off=W[:,np.ravel(np.c_[2*fixed,2*fixed+1])]@eall[rr[fixed]].reshape(-1)if len(fixed)else np.zeros(len(W));offset.extend(off)
 sub=W[:,np.ravel(np.c_[2*touch,2*touch+1])];nz=np.nonzero(sub);rows.extend((cursor+nz[0]).tolist());lc=np.ravel(np.c_[2*loc[rr[touch]],2*loc[rr[touch]]+1]);cc.extend(lc[nz[1]].tolist());vv.extend(sub[nz].tolist());cursor+=len(W)
for r in reg.independent[hit[reg.independent]]:
 for k in range(2):rows.append(cursor);cc.append(2*loc[r]+k);vv.append(1/reg.sigma[r]);offset.append(0.);cursor+=1
L=coo_matrix((vv,(rows,cc)),shape=(cursor,2*len(ids))).tocsr();offset=np.array(offset)
def install(x):
 for k,(i,j)in enumerate(cols):scene.values[i][j]=base[k]+x[k]*scale[k]

# Re-solve a real sensor-boundary intersection after physical geometry changes.
# No station or image-row clipping is used. Test all roots on the finite path.
def visibility_capture(local_ids,s):
 for i in local_ids:
  m=int(mi[i]);bi=int(br[i]);center=float(s[i]);time0=float(tt[i]);candidates=[]
  for vb in (0.,1079.):
   boundtime=time0+cam.tau*(vb/1079.-.5)
   nodes=np.unique(np.r_[np.linspace(0.,1.,257),center]);values=vb-cam.at(scene.world(m,bi,nodes),np.full(len(nodes),boundtime))['uv'][:,1]
   for edge in np.flatnonzero(values[:-1]*values[1:]<=0):
    left,right=nodes[edge:edge+2];fl=values[edge]
    for _ in range(56):
     mid=(left+right)*.5;fm=vb-float(cam.at(scene.world(m,bi,np.array([mid])),[boundtime])['uv'][0,1])
     if fl*fm<=0:right=mid
     else:left=mid;fl=fm
    candidates.extend([left,right,np.nextafter(left,-np.inf),np.nextafter(right,np.inf)])
  if candidates:
   candidates=np.array(candidates);cr=_certified_candidates(scene,cam,np.full(len(candidates),m),np.full(len(candidates),bi),candidates,np.full(len(candidates),time0));legal=np.flatnonzero(cr['valid'])
   if len(legal):s[i]=candidates[legal[np.argmin(np.sum((cr['uv'][legal]-obs[i])**2,axis=1))]]
 return s
visibility_seed=(V[ids]<1e-5)|(V[ids]>1079.-1e-5)

def evaluate(x,ss):
 install(x);ss=ss.copy();before=_certified_candidates(scene,cam,mi,br,ss,tt);visibility_capture(np.flatnonzero(visibility_seed|~before['valid']),ss);s=_implicit_capture(scene,cam,mi,br,ss,tt,obs,iterations=12)
 old=_certified_candidates(scene,cam,mi,br,ss,tt);r=_certified_candidates(scene,cam,mi,br,s,tt);take=r['valid']&(~old['valid']|(np.sum((r['uv']-obs)**2,axis=1)<np.sum((old['uv']-obs)**2,axis=1)))
 s=np.where(take,s,ss);bad=_certified_candidates(scene,cam,mi,br,s,tt);visibility_capture(np.flatnonzero(~bad['valid']),s);r=_certified_candidates(scene,cam,mi,br,s,tt);e=r['uv']-obs;res=L@e.ravel()+offset
 return .5*float(res@res),res,e,r,s

label_groups=[np.flatnonzero(reg.labels[ids]==label)for label in a.labels]if a.preserve_label_rms else []
label_limits=np.array([np.sum(eall[ids[j]]**2)for j in label_groups])
cross_paths=[Path(p)for p in a.cross_frames]
if cross_paths and not(a.ratio_block and a.cross_target):raise ValueError('Cross-view diagnostic requires its explicit physical ratio block and native material prefix')
cross_views=[load_frame(p,a.cross_target)for p in cross_paths]
cross_sigma=[np.array([r['sigma']for r in rr])for _,rr in cross_views]
cross_slices=[];offsets=0
for _,rr in cross_views:cross_slices.append(slice(offsets,offsets+len(rr)));offsets+=len(rr)
def ratios():
 if not cross_paths:return {}
 v=scene.values[next(i for i,b in enumerate(model['blocks'])if b['name']==a.ratio_block)]
 return dict(inner=v[2]/v[4],middle=v[3]/v[4],outer=1.)
def cross_residual(parameters):
 if not cross_paths:return np.zeros(0)
 rs=ratios();return np.concatenate([distances(parameters[7*i:7*i+7],rr,rs)/cross_sigma[i]for i,(_,rr)in enumerate(cross_views)])
def cross_diagnostic():
 if not cross_paths:return None,np.zeros(0),np.zeros(0)
 report=fit(cross_views,ratios());nuisance=np.array(report['parameters']);dr=cross_residual(nuisance);return report,nuisance,dr
cross_before,cross_nuisance,cross_e=cross_diagnostic();cross_limits=np.array([.5*float(cross_e[s]@cross_e[s])for s in cross_slices]);cross_initial_limits=cross_limits.copy()
print('CROSS_BASELINE',cross_limits.tolist(),ratios(),flush=True)

x=np.zeros(len(cols));cost,res,e,r,stations=evaluate(x,stations);start=time.monotonic();history=[];print('START',len(ids),cost,float(np.linalg.norm(e,axis=1).max()),flush=True)
it=-1
while True:
 it+=1
 print('J_BEGIN',it,time.monotonic()-start,flush=True)
 X=scene.world_rows(mi,br,stations)[0];rt=tt+cam.tau*(r['row']/1079-.5);dt=1e-6;ft=(cam.at(X,rt+dt)['uv']-cam.at(X,rt-dt)['uv'])/(2*dt);den=1-cam.tau/1079*ft[:,1]
 ds=1e-6;sp=np.minimum(1,stations+ds);sm=np.maximum(0,stations-ds);XP=scene.world_rows(mi,br,sp)[0];XM=scene.world_rows(mi,br,sm)[0];fix=(cam.at(XP,rt)['uv']-cam.at(XM,rt)['uv'])/(sp-sm)[:,None];tan=fix+ft*(cam.tau/1079*fix[:,1]/den)[:,None];tan_raw=tan.copy();sensor=(r['row']<1e-5)|(r['row']>1079.-1e-5);tan/=np.maximum(np.linalg.norm(tan,axis=1),1e-15)[:,None];normal=np.c_[-tan[:,1],tan[:,0]]
 J=np.empty((len(ids),2,len(cols)))
 for k,(bi,j)in enumerate(cols):
  h=1e-7;scene.values[bi][j]+=h;XP=scene.world_rows(mi,br,stations)[0];scene.values[bi][j]-=2*h;XM=scene.world_rows(mi,br,stations)[0];scene.values[bi][j]+=h
  fixed=(cam.at(XP,rt)['uv']-cam.at(XM,rt)['uv'])/(2*h);gj=fixed+ft*(cam.tau/1079*fixed[:,1]/den)[:,None];col=normal*np.einsum('ni,ni->n',normal,gj)[:,None];col[sensor]=gj[sensor]-tan_raw[sensor]*(gj[sensor,1]/tan_raw[sensor,1])[:,None];J[:,:,k]=col*scale[k]
 print('J_END',it,time.monotonic()-start,flush=True)
 A=L@J.reshape(2*len(ids),-1);step,_,rank,sv=lstsq(A,-res,cond=1e-10)
 # Convex linearized constrained W subproblem: retain the original global
 # maximum-error ceiling while reducing W, with all eight existing ring DOF.
 jf=J.reshape(len(ids),2,-1)
 # Both native secondary-view likelihoods must be no worse than their exact
 # baseline fixed-ratio diagnostic; no arbitrary relative likelihood weight.
 cross_J=np.empty((len(cross_e),len(cols)+len(cross_nuisance)))
 for k,(bi,j)in enumerate(cols):
  h=1e-7;scene.values[bi][j]+=h;rp=cross_residual(cross_nuisance);scene.values[bi][j]-=2*h;rm=cross_residual(cross_nuisance);scene.values[bi][j]+=h;cross_J[:,k]=(rp-rm)/(2*h)*scale[k]
 for k in range(len(cross_nuisance)):
  h=1e-6;np_=cross_nuisance.copy();nm_=cross_nuisance.copy();np_[k]+=h;nm_[k]-=h;cross_J[:,len(cols)+k]=(cross_residual(np_)-cross_residual(nm_))/(2*h)*1e-3
 na=len(cols)+len(cross_nuisance)
 def obj(p):
  z=A@p[:len(cols)]+res;return .5*float(z@z)/cost
 def grad(p):return np.r_[np.asarray(A.T@(A@p[:len(cols)]+res))/cost,np.zeros(len(cross_nuisance))]
 def con(p):
  dr=cross_e+cross_J@p
  pe=e+np.einsum('nik,k->ni',jf,p[:len(cols)])
  return np.r_[1-np.sum(pe**2,axis=1)/ceiling**2,[1-np.sum(pe[g]**2)/label_limits[j]for j,g in enumerate(label_groups)],[1-.5*float(dr[s]@dr[s])/cross_limits[j]for j,s in enumerate(cross_slices)]]
 def cj(p):
  first=-2*np.einsum('nik,ni->nk',jf,e+np.einsum('nik,k->ni',jf,p[:len(cols)]))/ceiling**2
  dr=cross_e+cross_J@p;second=np.array([-dr[s]@cross_J[s]/cross_limits[j]for j,s in enumerate(cross_slices)]).reshape(-1,na)
  pe=e+np.einsum('nik,k->ni',jf,p[:len(cols)]);label_J=np.array([-2*np.einsum('nik,ni->k',jf[g],pe[g])/label_limits[j]for j,g in enumerate(label_groups)]).reshape(-1,len(cols))
  return np.vstack([np.pad(first,((0,0),(0,len(cross_nuisance)))),np.pad(label_J,((0,0),(0,len(cross_nuisance)))),second])
 opt=minimize(obj,np.zeros(na),jac=grad,constraints=[dict(type='ineq',fun=con,jac=cj)],method='SLSQP',options=dict(maxiter=150,ftol=1e-10))
 # A numerically feasible intermediate SQP direction is only a proposal;
 # every acceptance still re-solves actual finite RS and both diagnostics.
 if opt.success or(np.isfinite(opt.x).all()and con(opt.x).min()>-1e-6 and obj(opt.x)<1.):step=opt.x[:len(cols)]
 print('CONSTRAINED_STEP',opt.success,opt.message,float(con(opt.x).min()),float(obj(opt.x)),flush=True);accepted=None
 for ls in range(14):
  print('LINE_BEGIN',it,ls,time.monotonic()-start,flush=True)
  f=2.**(-ls);xx=x+f*step;install(xx)
  domain=all(model['blocks'][i]['lower'][j]<=scene.values[i][j]<=model['blocks'][i]['upper'][j]for i,j in cols)
  if not domain:continue
  physics=scene.check_physics()
  if not physics['passed']:
   print('PHYSICS_REJECT',physics['violations'][:3],flush=True);continue
  nc,nres,ne,nr,ns=evaluate(xx,stations);cross_new,new_nuisance,new_cross_e=cross_diagnostic();new_cross_costs=np.array([.5*float(new_cross_e[s]@new_cross_e[s])for s in cross_slices]);cross_ok=bool(np.all(new_cross_costs<=cross_initial_limits+1e-8));label_sums=np.array([np.sum(ne[g]**2)for g in label_groups]);label_ok=bool(np.all(label_sums<=label_limits+1e-8))
  print('CROSS_LINE',new_cross_costs.tolist(),cross_ok,ratios(),'LABEL_RMS_RATIO',(label_sums/label_limits).tolist(),label_ok,flush=True)
  worst=int(np.argmax(np.linalg.norm(ne,axis=1)));print('LINE_RESULT',nc,float(np.linalg.norm(ne,axis=1).max()),int(nr['valid'].sum()),'worst',int(ids[worst]),'row',float(nr['row'][worst]),'seedrow',float(V[ids[worst]]),flush=True)
  if label_ok and cross_ok and nr['valid'].all()and nc<cost-1e-7 and np.linalg.norm(ne,axis=1).max()<=ceiling:accepted=(xx,nc,nres,ne,nr,ns,f,cross_new,new_nuisance,new_cross_e,new_cross_costs);break
 rec=dict(iteration=it,cost_before=cost,accepted=accepted is not None,rank=int(rank))
 if accepted is None:history.append(rec);break
 x,cost,res,e,r,stations,f,cross_current,cross_nuisance,cross_e,cross_costs=accepted;rec.update(cross_view_half_cost=cross_costs.tolist(),ratios=ratios(),cost=cost,max=float(np.linalg.norm(e,axis=1).max()),fraction=f);history.append(rec);print('STEP',json.dumps(rec),flush=True)
 if np.linalg.norm(f*step)<1e-5:break
install(x);candidate=copy.deepcopy(model)
for i in blocks:candidate['blocks'][i]['values']=scene.values[i].tolist()
candidate['ring_refinement_provenance']=dict(parent=a.model,baseline=base.tolist(),new=[float(scene.values[i][j])for i,j in cols],same_theta_camera=True,source_scope=dict(labels=a.labels,blocks=a.blocks,source_rows=ids.tolist(),objective='full original W touching groups; other rows retained as fixed residuals',maximum_error_ceiling=ceiling),initialization_only=True)
(out/'model.json').write_text(json.dumps(candidate,indent=2))
Sn=S.copy();Vn=V.copy();Sn[ids]=stations;Vn[ids]=r['row']
for name,arr,dtype in [('station.f64',Sn,'<f8'),('path.u32',B,'<u4'),('initial_row.f64',Vn,'<f8')]:np.asarray(arr,dtype=dtype).tofile(out/name)
(out/'initial_alpha.f64').write_bytes((seed/'initial_alpha.f64').read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'initializer_manifest.json').write_text(json.dumps(dict(semantic_manifest=dict(path=str(sem),sha256=sha(sem)),files={n:dict(sha256=sha(out/n))for n in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']}),indent=2))
cross_final,_,_=cross_diagnostic();(out/'cross_view_comparison.json').write_text(json.dumps(dict(baseline=cross_before,candidate=cross_final,baseline_per_view_half_cost=cross_initial_limits.tolist(),source_files=[dict(path=str(p),sha256=sha(p))for p in cross_paths],assumption='sigma-weighted projective normal-distance diagnostic; no physical trueRS new camera claim; legacy source selection not independent'),indent=2))
(out/'report.json').write_text(json.dumps(dict(history=history,source_rows=len(ids),baseline_whole_W_half_cost=baseline_summary['whole_original_W_half_cost'],maximum_error_ceiling=ceiling,initialization_only=True,complete_full_source_recalculation_required=True,label_RMS_acceptance_constraint_only=bool(a.preserve_label_rms),label_baseline_squared_UV_sum=label_limits.tolist(),cost=cost,max=float(np.linalg.norm(e,axis=1).max()),physics=scene.check_physics(),elapsed=time.monotonic()-start),indent=2));print('DONE',cost,flush=True)
