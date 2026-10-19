"""FIT-only fixed-world H8 versus H8+one radial versus current Brown25.

All45 original groups stay in the report. Only original Z0 groups enter
the plane probe; above-ice board tops and advertisements never enter H.
No result changes any complete programme/source/runtime or official role.
"""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import os,sys,json,hashlib,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
BASE=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from isolated_solver_bootstrap import install
install()
import cv2,numpy as np
from types import SimpleNamespace
from wide_anchor_candidate.native_wide_source import load_native_wide_source_packet,world_curve,SCENE_UNIT_M
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown,curve_vector_error_full_brown
from projection import vector_from_scene
from low_independent_held_component_audit.audit_b691_to_v3_held19 import FOLDERS,read_version

OUT=BASE/'simple_wide_perspective_reset_candidate'
CENTER=np.array([960.,540.]);PS=1000.;WS=30.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(r):
    v=np.linalg.norm(np.asarray(r).reshape(-1,2),axis=1)
    return dict(points=len(v),native_point_RMS_px=float(np.sqrt(np.mean(v*v))),median_px=float(np.median(v)),p95_px=float(np.percentile(v,95)),max_px=float(v.max()))
def cost(r):
    a=abs(r);return float(np.where(a<=2,.5*r*r,2*a-2).sum())
def homography_seed(camera):
    R=cv2.Rodrigues(camera[3:6])[0];K=np.array([[np.exp(camera[6]),camera[10],camera[8]],[0,np.exp(camera[7]),camera[9]],[0,0,1.]])
    T=np.array([[1/PS,0,-CENTER[0]/PS],[0,1/PS,-CENTER[1]/PS],[0,0,1.]])
    H=T@K@np.column_stack((R[:,:2]*(WS/SCENE_UNIT_M),-R@camera[:3]));H/=H[2,2]
    return H.ravel()[:8]
def inverse_map_forward(undistorted,state,jacobian=None):
    """Inverse of ROOT's image-only undistortion, evaluated back in nativepx."""
    c=1000*np.asarray(state[:2]);k1,k2=state[2:];v=(undistorted-c)/1000.;ru=np.linalg.norm(v,axis=1);rd=ru.copy()
    for _ in range(20):
        f=rd*(1+k1*rd**2+k2*rd**4)-ru;fp=1+3*k1*rd**2+5*k2*rd**4
        rd-=f/fp
    g=1+k1*rd**2+k2*rd**4;gr=1+3*k1*rd**2+5*k2*rd**4
    if np.any(g<=0)|np.any(gr<=0)|np.any(rd<0):raise ValueError('Independent inverse radial not one-to-one on used support')
    ratio=np.divide(rd,ru,out=np.ones_like(rd),where=ru>0);native=c+1000*v*ratio[:,None]
    recovered=c+(native-c)*(1+k1*rd**2+k2*rd**4)[:,None]
    if np.max(abs(recovered-undistorted))>1e-7:raise ValueError('Native forward inversion failed')
    if jacobian is None:return native,None
    direction=np.divide(v,ru[:,None],out=np.zeros_like(v),where=ru[:,None]>0)
    D=np.eye(2)[None,:,:]/g[:,None,None]+(1/gr-1/g)[:,None,None]*direction[:,:,None]*direction[:,None,:]
    return native,np.einsum('nij,njk->nik',D,jacobian)
def plane_project(q,xy,jac=False):
    H=np.r_[q[:8],1.].reshape(3,3);X=np.c_[np.asarray(xy)*SCENE_UNIT_M/WS,np.ones(len(xy))]
    a=X@H.T;den=a[:,2];valid=(den>1e-8)&np.isfinite(a).all(axis=1)
    z=a[:,:2]/den[:,None];out=z.copy()
    if jac:
        J=np.zeros((len(X),2,len(q)));J[:,0,:3]=X/den[:,None];J[:,1,3:6]=X/den[:,None]
        J[:,:,:8][:,:,6:8]=-z[:,:,None]*X[:,None,:2]/den[:,None,None]
    if len(q)==9:
        rr=np.sum(z*z,axis=1);scale=1+q[8]*rr
        valid&=(scale>0)&(1+3*q[8]*rr>0)
        out=z*scale[:,None]
        if jac:
            dz=scale[:,None,None]*np.eye(2)[None,:,:]+2*q[8]*z[:,:,None]*z[:,None,:]
            J[:,:,:8]=np.einsum('nij,njk->nik',dz,J[:,:,:8]);J[:,:,8]=z*rr[:,None]
    return out*PS+CENTER,valid,(J*PS if jac else None)
def nearest_curve(uv,p,j=None,valid=None):
    a=p[:-1];d=np.diff(p,axis=0);den=np.sum(d*d,axis=1)
    good=(den>1e-4)&np.isfinite(a).all(axis=1)&np.isfinite(p[1:]).all(axis=1)
    if valid is not None:good&=valid[:-1]&valid[1:]
    if not np.any(good):raise ValueError('No finite plane support')
    offset=uv[:,None,:]-a[None,:,:];t=np.einsum('nsi,si->ns',offset,d)/np.where(den>0,den,1)[None,:]
    tc=np.clip(t,0,1);foot=a[None,:,:]+tc[:,:,None]*d[None,:,:]
    distances=np.sum((foot-uv[:,None,:])**2,axis=2);distances[:,~good]=np.inf
    ix=np.argmin(distances,axis=1);ii=np.arange(len(uv));r=foot[ii,ix]-uv
    if j is None:return r,None,ix,tc[ii,ix]
    ts=t[ii,ix];cc=tc[ii,ix];dv=d[ix];ja=j[ix];jd=j[ix+1]-ja;uu=uv-a[ix]
    num=-np.einsum('nik,ni->nk',ja,dv)+np.einsum('ni,nik->nk',uu,jd)
    dden=2*np.einsum('ni,nik->nk',dv,jd);dt=(num-ts[:,None]*dden)/den[ix,None];dt[(ts<=0)|(ts>=1)]=0.
    J=ja+cc[:,None,None]*jd+dv[:,:,None]*dt[:,None,:]
    return r,J,ix,cc
class PlaneProbe:
    def __init__(self,rows,fixed_inverse=None):self.rows=rows;self.calls=0;self._key=None;self._last=None;self.fixed_inverse=fixed_inverse
    def evaluate(self,q,jac=True):
        key=(q.tobytes(),jac)
        if key==self._key:return self._last
        values=[];js=[];groups=[]
        for row in self.rows:
            p,v,j=plane_project(q,row['world'][:,:2],jac)
            if self.fixed_inverse is not None:p,j=inverse_map_forward(p,self.fixed_inverse,j)
            # Check every original finite knot; never silently discard a bad
            # used world ray to buy a lower source cost.
            if not v.all():raise ValueError('Plane/radial support domain lost')
            raw,J,ix,t=nearest_curve(row['uv'],p,j,v)
            s=np.repeat(row['sigma'],2);values.append(raw.ravel()/s)
            if jac:js.append(J.reshape(-1,len(q))/s[:,None])
            groups.append(dict(id=row['id'],target=row['target'],**stats(raw),endpoint_points=int(np.sum((t==0)|(t==1)))))
        self.calls+=1;self._key=key;self._last=(np.concatenate(values),np.vstack(js)if jac else None,groups);return self._last
def fit_guarded(probe,start,name,model):
    """Small guarded IRLS LM: invalid trials rejected, not penalized/dropped."""
    q=start.copy();damping=1e-3;history=[];stopped='iteration limit';last=time.perf_counter()
    for iteration in range(100):
        r,J,_=probe.evaluate(q);current=cost(r);w=np.minimum(1.,2/np.maximum(abs(r),1e-300));sw=np.sqrt(w)
        scale=np.maximum(np.linalg.norm(J*sw[:,None],axis=0),1e-12);A=J*sw[:,None]/scale;rr=r*sw
        H=A.T@A;g=A.T@rr;accepted=False;invalid=0
        for attempt in range(14):
            step=-np.linalg.solve(H+damping*np.eye(len(q)),g)/scale
            try:
                candidate=q+step;trial=cost(probe.evaluate(candidate)[0])
            except ValueError:
                invalid+=1;trial=float('inf')
            if trial<current:
                q=candidate;damping=max(1e-10,damping/3);accepted=True;break
            damping*=5
        history.append(dict(iteration=iteration,cost=current,accepted=accepted,invalid_trials=invalid,damping=damping))
        if time.perf_counter()-last>3:
            last=time.perf_counter();print(json.dumps(dict(world=name,model=model,iteration=iteration,cost=current,accepted=accepted)),flush=True)
        if not accepted:stopped='no accepted feasible LM trial';break
        if current-trial<1e-9*max(1.,current):stopped='relative cost stationary';break
    r,J,_=probe.evaluate(q);w=np.minimum(1.,2/np.maximum(abs(r),1e-300));gradient=J.T@(w*r)
    return SimpleNamespace(x=q,success=stopped=='relative cost stationary',status=stopped,message=stopped,nfev=probe.calls,optimality=float(abs(gradient).max()),history=history)
def draw_chain(image,p,color,thick=1):
    v=np.isfinite(p).all(axis=1)&(abs(p).max(axis=1)<1e6)
    for i in range(len(p)-1):
        if v[i]and v[i+1]:
            a,b=np.round(p[i]).astype(int),np.round(p[i+1]).astype(int)
            ok,a,b=cv2.clipLine((0,0,image.shape[1],image.shape[0]),tuple(a),tuple(b))
            if ok:cv2.line(image,a,b,color,thick,cv2.LINE_AA)
def all_ice_curves(scene):
    curves=[]
    for feature in scene['ice_features']:
        target=dict(feature_id=feature['id'])
        if feature['kind']=='circle':
            curves.append((feature['id'],world_curve(scene,dict(target,kind='circle_outline'))))
            for ring in feature.get('rings',[]):curves.append((feature['id']+'_'+ring['role'],world_curve(scene,dict(target,kind='circle_outline',ring_role=ring['role']))))
        elif feature['kind']=='polyline':
            for n,_ in enumerate(feature['segments_xyz']):curves.append((feature['id'],world_curve(scene,dict(target,kind='line_segment',segment=n))))
        elif feature['kind']=='polygon':curves.append((feature['id'],world_curve(scene,dict(target,kind='polygon_outline'))))
    for side in ['left_end','right_end','upper_long_side','lower_long_side','upper_left','upper_right','lower_left','lower_right']:
        curves.append(('board_contact_'+side,world_curve(scene,dict(kind='board_side',side=side,edge='contact'))))
    return curves
def overlay(original,scene,q,camera,label,rows,filename,fixed_inverse=None):
    image=original.copy()
    # All original ice features/rings/Lanes/course contacts, regardless of fit.
    for name,w in all_ice_curves(scene):
        p=project_central_valid_full_brown(camera,w)[0]if q is None else plane_project(q,w[:,:2])[0]
        if q is not None and fixed_inverse is not None:p=inverse_map_forward(p,fixed_inverse)[0]
        draw_chain(image,p,(0,210,255),1)
    # Non-planar ads have no H projection. Keep actual current Brown25
    # boundaries+IDs on screen in a separate colour, never used in this fit.
    for ad in scene.get('advertisements',[]):
        corners=ad.get('corners_xyz_scene_units',{})
        if not corners:continue
        points=np.asarray([corners[k]for k in ['left_bottom','right_bottom','right_top','left_top']],float)
        p=project_central_valid_full_brown(camera,np.vstack([points,points[0]]))[0];draw_chain(image,p,(220,120,220),1)
        if np.isfinite(p[:4]).all()and(abs(p[:4]).max()<1e6):
            center=np.round(p[:4].mean(axis=0)).astype(int)
            if 0<=center[0]<1920 and 0<=center[1]<1080:cv2.putText(image,'Ad'+str(ad['user_number']),tuple(center),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,120,220),1,cv2.LINE_AA)
    for row in rows:
        for point in row['uv']:cv2.circle(image,tuple(np.round(point).astype(int)),1,(230,80,20),-1)
    cv2.rectangle(image,(0,0),(1920,50),(0,0,0),-1)
    cv2.putText(image,label,(15,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,'yellow: all ice/current fixed world; blue: original FIT UV; purple ads: current Brown25 (not H fit)',(15,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1,cv2.LINE_AA)
    cv2.imwrite(str(filename),image)
def main():
    begun=time.perf_counter();OUT.mkdir(exist_ok=True)
    packet=load_native_wide_source_packet();frame=packet.frame;image=cv2.imread(frame['image_path']);assert image is not None
    v3=read_version(FOLDERS[1]);scene_current=json.loads((FOLDERS[1]/'latest_candidate_scene_UNACCEPTED.json').read_text(encoding='utf8'))
    meta=json.loads((FOLDERS[1]/'actual_terminal_four_source_images/actual_render_input_metadata.json').read_text(encoding='utf8'))
    currentcam=np.asarray(meta['high_camera25'],float)
    inverse_path=BASE/'simple_wide_physical_reset_candidate/raw_image_straight_lines_and_inverse_radial.json'
    inverse_report=json.loads(inverse_path.read_text(encoding='utf8'));inverse_state=np.asarray(inverse_report['state'],float)
    scene_b691=json.loads((FOLDERS[0]/'latest_candidate_scene_UNACCEPTED.json').read_text(encoding='utf8'))
    seed=vector_from_scene(next(x for x in scene_b691['frame_cameras']if x['source_pts']==56218880))
    results={};arrays={};pictures=[]
    for name,scene in [('current_v3_world43',scene_current),('old_b691_world',scene_b691)]:
        rows=[dict(id=x['id'],target=x['target'],uv=np.asarray(x['pixels_uv'],float),sigma=np.broadcast_to(np.asarray(x['sigma_px'],float),(len(x['pixels_uv']),)).copy(),world=world_curve(scene,x['target']))for x in packet.observations]
        planar=[r for r in rows if np.all(r['world'][:,2]==0)]
        nonplanar=[r for r in rows if not np.all(r['world'][:,2]==0)]
        assert len(rows)==45 and sum(len(x['uv'])for x in rows)==2144 and len(planar)==41 and sum(len(x['uv'])for x in planar)==1926
        brown=[]
        for row in rows:
            r=curve_vector_error_full_brown(currentcam,row['world'],row['uv'],False)
            brown.append(dict(id=row['id'],target=row['target'],plane_admitted=bool(np.all(row['world'][:,2]==0)),**stats(r)))
        results[name]=dict(current25Brown_all45=brown,plane_groups=41,plane_points=1926,above_ice_excluded_from_H_fit=[dict(id=r['id'],points=len(r['uv']),z_range_scene=[float(r['world'][:,2].min()),float(r['world'][:,2].max())])for r in nonplanar])
        probe=PlaneProbe(planar);p0=homography_seed(seed)
        original_start=p0.copy()
        for model in ['homography8','homography8_plus_single_radial','homography8_fixed_image_only_inverse_plumbline']:
            fixed_inverse=inverse_state if model=='homography8_fixed_image_only_inverse_plumbline'else None
            probe=PlaneProbe(planar,fixed_inverse)
            start=np.r_[p0,0.]if model=='homography8_plus_single_radial'else original_start.copy()
            initial=probe.evaluate(start)[0]
            fit=fit_guarded(probe,start,name,model)
            value,J,groups=probe.evaluate(fit.x);endpoint=sum(x['endpoint_points']for x in groups)
            result=dict(parameters=fit.x.tolist(),initial_cost=cost(initial),final_cost=cost(value),per_group_native=groups,
                aggregate_native=stats(np.concatenate([value[sum(2*len(x['uv'])for x in planar[:i]):sum(2*len(x['uv'])for x in planar[:i+1])]*np.repeat(r['sigma'],2)for i,r in enumerate(planar)])),
                optimizer=dict(success=bool(fit.success),status=fit.status,message=str(fit.message),nfev=int(fit.nfev),optimality=float(fit.optimality),history=fit.history),
                every_original_planar_UV_retained=True,endpoint_points=endpoint,one_radial_center_fixed_native_px=CENTER.tolist()if len(fit.x)==9 else None)
            results[name][model]=result;arrays[name+'_'+model]=fit.x
            png=OUT/(name+'_'+model+'_all_ice_and_original_UV.png');overlay(image,scene,fit.x,currentcam,name+' / '+model+' / FIT-only; no adoption',planar,png,fixed_inverse);pictures.append(str(png))
            if model=='homography8':p0=fit.x.copy()
        png=OUT/(name+'_current25Brown_all_ice_and_original_UV.png');overlay(image,scene,None,currentcam,name+' / current25Brown / FIT-only; no accuracy claim',planar,png);pictures.append(str(png))
        print(json.dumps(dict(world=name,H_RMS=results[name]['homography8']['aggregate_native']['native_point_RMS_px'],Hk1_RMS=results[name]['homography8_plus_single_radial']['aggregate_native']['native_point_RMS_px'])),flush=True)
    report=dict(schema='rinkspace.readonly-original45-wide-fixed-world-simple-perspective-reset.v1',source_packet_contract=packet.contract,
        source_image=dict(path=frame['image_path'],sha256=sha(frame['image_path']),pts=56218880,time='01:01:00.083'),
        current_complete_frozen_q_profile=v3['event'],current_checkpoint_sha256=sha(v3['checkpoint']),current_high_camera25=currentcam.tolist(),
        image_only_inverse_plumbline=dict(path=str(inverse_path),sha256=sha(inverse_path),state=inverse_state.tolist(),original_UV_stored_unchanged=True,reported_residuals_back_in_native_px=True),
        world_source_files=[dict(path=str(f/'latest_candidate_scene_UNACCEPTED.json'),sha256=sha(f/'latest_candidate_scene_UNACCEPTED.json'))for f in FOLDERS],
        all45groups2144_UV_retained_in_report=True,plane_only41groups1926_in_H_objective=True,
        original_sigmas_roles_UV_and_targets_unchanged=True,HELD_used_for_fit_or_model_selection=False,
        objective='component original-sigma normalized Huber2; original v5 group budget, no extra family reweighting',
        homography_is_unknown_plane_projection_not_3D_camera_calibration=True,
        one_radial_is_diagnostic_image_plane_radial_centered_960_540=True,not_identified_physical_Brown_k1=True,
        nonplanar_tops_and_ads_not_fitted_or_H_projected=True,ad_overlay_policy='current actual Brown25, same scene geometry, purple; cannot validate H with this overlay',
        full_programme_graph_loaded=False,production_adopted=False,geometry_changed=False,accuracy_accepted=False,
        runtime_file_sha256=sha(__file__),results=results,pictures=pictures,wall_seconds=time.perf_counter()-begun)
    (OUT/'actual_original45_fixed_world_simple_perspective_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    np.savez_compressed(OUT/'actual_H8_and_H8_k1_diagnostic_parameters.npz',**arrays,current_camera25=currentcam)
    print(json.dumps(dict(report=str(OUT/'actual_original45_fixed_world_simple_perspective_report.json'),seconds=report['wall_seconds'])),flush=True)
if __name__=='__main__':main()
