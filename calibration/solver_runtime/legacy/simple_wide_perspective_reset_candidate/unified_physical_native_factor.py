"""One physical perspective camera and one image-only inverse radial map.

Small FIT-only experiment, not a programme/production source implementation.
All original45/2144 curve UV and26 manual corners remain literal. The two
weight policies are explicit alternatives; sigmas never carry loss weights.
"""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import os,sys,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
BASE=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from isolated_solver_bootstrap import install
install()
import numpy as np,cv2
from projection import project_central_valid
from wide_anchor_candidate.native_wide_source import load_native_wide_source_packet,world_curve,SCENE_UNIT_M
from wide_anchor_candidate.wide_user_ad_and_secondary_source import load_wide_user_ad_source,bind_wide_ad_world
from simple_wide_perspective_reset_candidate.audit_simple_wide_perspective import inverse_map_forward,stats
from simple_wide_perspective_reset_candidate.exact_user_raster_shape import corrected,radial_disk_guard


def camera14_from_family(K,R,C,inverse4):
    K=np.asarray(K,float);R=np.asarray(R,float)
    if abs(K[0,1])>1e-8:raise ValueError('This explicit physical experiment owns orthogonal pixel axes/skew0')
    return np.r_[np.asarray(C,float)/100,cv2.Rodrigues(R)[0].ravel(),np.log(K[0,0]),np.log(K[1,1]),K[0,2]/1000,K[1,2]/1000,inverse4]


def pinhole_camera25(state14):
    q=np.asarray(state14,float)
    if q.shape!=(14,) or not np.isfinite(q).all():raise ValueError('Complete active physical camera14 required')
    if not np.all((-8<q[6:8])&(q[6:8]<18)):raise ValueError('Finite positive physical focal domain required')
    c=np.zeros(25);c[:3]=q[:3]*100;c[3:8]=q[3:8];c[8:10]=q[8:10]*1000
    if c[2]<=0:raise ValueError('Physical camera must remain above ice')
    return c


def project_native(state14,world,*,jacobian=False,world_jacobian=None):
    """All rails/ice/prints use exactly this same projection, without Brown.

    Columns are [Cscene/100,Rodrigues,logfx/logfy,PP/1000,inverse4,
    optional actual shared-world columns]. Native J is not sigma normalized.
    """
    c=pinhole_camera25(state14);world=np.asarray(world,float)
    if jacobian:
        und,depth,j11,valid=project_central_valid(c,world,with_jacobian=True)
        j11=j11.reshape(-1,2,11);native,jc=inverse_map_forward(und,state14[10:],j11[:,:,:10])
        _,ju=corrected(native,state14[10:],True)
        _,D=inverse_map_forward(und,state14[10:],np.broadcast_to(np.eye(2),(len(world),2,2)))
        ji=-np.einsum('nij,njk->nik',D,ju)
        J=np.concatenate([jc,ji],axis=2)
        if world_jacobian is not None:
            dx=-j11[:,:,:3]/100
            jw=np.einsum('nij,njk,nkl->nil',D,dx,np.asarray(world_jacobian,float))
            J=np.concatenate([J,jw],axis=2)
    else:
        und,depth,valid=project_central_valid(c,world);native,_=inverse_map_forward(und,state14[10:]);J=None
    valid&=np.isfinite(native).all(axis=1)
    radial_disk_guard(state14[10:],native[valid])
    return native,depth,valid,J


def nearest_finite(uv,projected,J,valid_segments):
    """Original finite Euclidean segment association, same two residuals."""
    uv=np.asarray(uv,float);a=projected[:-1];d=np.diff(projected,axis=0);den=np.sum(d*d,axis=1)
    good=np.asarray(valid_segments,bool)&(den>1e-4)&np.isfinite(a).all(axis=1)&np.isfinite(projected[1:]).all(axis=1)
    if not good.any():raise ValueError('No original finite positive-depth curve segment')
    offset=uv[:,None,:]-a[None,:,:];t=np.einsum('nsi,si->ns',offset,d)/np.where(den>0,den,1)[None,:]
    tc=np.clip(t,0,1);foot=a[None,:,:]+tc[:,:,None]*d[None,:,:]
    ds=np.sum((foot-uv[:,None,:])**2,axis=2);ds[:,~good]=np.inf
    ix=np.argmin(ds,axis=1);ii=np.arange(len(uv));raw=foot[ii,ix]-uv
    if J is None:return raw,None,ix,tc[ii,ix]
    ts=t[ii,ix];cc=tc[ii,ix];dv=d[ix];ja=J[ix];jd=J[ix+1]-ja;uu=uv-a[ix]
    num=-np.einsum('nik,ni->nk',ja,dv)+np.einsum('ni,nik->nk',uu,jd)
    dden=2*np.einsum('ni,nik->nk',dv,jd);dt=(num-ts[:,None]*dden)/den[ix,None];dt[(ts<=0)|(ts>=1)]=0
    jr=ja+cc[:,None,None]*jd+dv[:,:,None]*dt[:,None,:]
    return raw,jr,ix,cc


def physical_feature_id(target):
    kind=target['kind']
    if kind=='circle_outline':return 'circle:'+target['feature_id']+':'+str(target.get('ring_role','outer'))
    if kind=='line_segment':return 'paint:'+target['feature_id']+':'+str(target.get('segment',0))
    if kind=='polygon_outline':return 'polygon:'+target['feature_id']
    if kind=='polygon_supporting_line':return 'lane:'+target['feature_id']+':'+str(target['index'])
    if kind=='board_side':return 'board:'+str(target['side'])+':'+str(target['edge'])
    return json.dumps(target,sort_keys=True,separators=(',',':'))


class UnifiedPhysicalNativeFactor:
    """Camera14 + actualworld43/47; no graph, HELD, or implicit priors."""
    def __init__(self,geometry,*,budget='literal_reference',engineering_ad_sigma_px=4.,geometry_step=1e-6):
        self.g=geometry;self.world_count=int(geometry.parameter_count);self.parameter_count=14+self.world_count
        self.packet=load_native_wide_source_packet();self.ad_packet=load_wide_user_ad_source()
        self.observations=list(self.packet.observations);self.geometry_step=float(geometry_step)
        if budget not in ('literal_reference','physical_feature_normalized'):raise ValueError('Explicit budget experiment required')
        self.budget=budget;self.ad_sigma=float(engineering_ad_sigma_px)
        self.rows=[];totals={}
        for o in self.observations:
            uv=np.asarray(o['pixels_uv'],float);fid=physical_feature_id(o['target']);totals[fid]=totals.get(fid,0)+len(uv)
            self.rows.append(dict(id=o['id'],target=o['target'],uv=uv,sigma=np.broadcast_to(np.asarray(o['sigma_px'],float),(len(uv),)).copy(),physical_feature_id=fid,literal_weight=float(o.get('weight',1.))))
        self.weights=[];self.scales=[];self.ledger=[];offset=0
        for row in self.rows:
            count=len(row['uv']);w=row['literal_weight']
            if budget=='physical_feature_normalized':w=1./totals[row['physical_feature_id']]
            self.weights.extend([w]*(2*count));self.scales.extend(np.repeat(row['sigma'],2))
            self.ledger.append(dict(id=row['id'],physical_feature_id=row['physical_feature_id'],slice=[offset,offset+2*count],points=count,role='original_FIT',loss_weight_per_component=w,group_total_point_budget=w*count));offset+=2*count
        for row in self.ad_packet['rows']:
            integration=np.asarray(row['original_group_weight']['point_integration_weights'],float)
            positive=integration[integration>0]
            if len(positive)!=1:raise ValueError('Original corner integration authority changed')
            weight=float(positive[0])
            self.weights.extend([weight,weight]);self.scales.extend([self.ad_sigma]*2)
            self.ledger.append(dict(id=row['id'],physical_feature_id='Ad'+str(row['user_ad_number']),slice=[offset,offset+2],points=1,role='original_user_FIT',source_sigma_px=None,engineering_sigma_px=self.ad_sigma,loss_weight_per_component=weight,group_total_point_budget=weight));offset+=2
        self.weights=np.asarray(self.weights);self.scales=np.asarray(self.scales)
        if offset!=4340 or len(self.rows)!=45 or sum(len(r['uv'])for r in self.rows)!=2144:raise ValueError('All original45/2144 and26 corners required')
        self._world_cache={};self._scene_cache={};self.calls=0
        self.contract=dict(schema='unified-single-physical-camera14-original45-and26-source.v1',rows=4340,original_curve_groups=45,original_curve_points=2144,original_manual_ad_corners=26,
            physical_camera_parameters_active=['Cx','Cy','Cz','rvec_x','rvec_y','rvec_z','logfx','logfy','PPx','PPy','inverse_cx','inverse_cy','inverse_k1','inverse_k2'],skew_fixed_zero_as_explicit_orthogonal_axes_model=True,
            inherited_Brown14_not_used=True,same_camera_for_ice_tops_bottoms_ads=True,world_parameter_count=self.world_count,budget_policy=budget,
            native_component_Huber_cutoff_in_original_sigma_units=2.,loss_weights_applied_after_original_native_Huber=True,
            original_sigma_UV_roles_preserved=True,manual_ad_source_sigma_None_preserved=True,manual_ad_engineering_sigma_px=self.ad_sigma,
            feature_normalization_is_new_explicit_objective=budget=='physical_feature_normalized',physical_feature_budget=1. if budget=='physical_feature_normalized'else None,
            original_auto_dark_bands_on_FIT=False,HELD_loaded_or_optimized=False,accuracy_accepted=False)

    def source_scene(self,qw):
        """Owned immutable-input memo, shared by source/reference/constraints.

        This cache is private to the small experiment. Its geometry provider
        and base authority are never mutated. It changes no material binding,
        finite sample, world-J arithmetic or projection. Consumers read it;
        a renderer that edits metadata must use the provider's full scene.
        """
        qw=np.asarray(qw,float);key=qw.tobytes();scene=self._scene_cache.get(key)
        if scene is None:
            scene=self.g.source_scene(qw);self._scene_cache[key]=scene
            if len(self._scene_cache)>192:self._scene_cache.pop(next(iter(self._scene_cache)))
        return scene

    def _world_values(self,qw):
        self.g.validate_parameters(qw);scene=self.source_scene(qw)
        curves=[world_curve(scene,row['target'])for row in self.rows]
        # Reuses actual original print role/finite material source binding.
        by={int(a['user_number']):a for a in scene['advertisements']}
        ads=np.array([by[r['user_ad_number']]['corners_xyz_scene_units'][r['role']]for r in self.ad_packet['rows']])
        return scene,curves,ads

    def audit_named_ad_scene_binding(self,qw):
        """Must pass at actual zero/nonzero before scene batch is adopted."""
        _,_,ads=self._world_values(qw)
        bound=bind_wide_ad_world(self.ad_packet,self.g,qw)
        difference=float(np.max(abs(ads-bound['world'])))
        if difference>1e-10:raise ValueError('Scene print corners and actual original source binder disagree: '+str(difference))
        return dict(max_scene_coordinate_difference=difference,original26roles_and_scopes_preserved=True)

    def world_data(self,qw,jac):
        key=(qw.tobytes(),bool(jac));cached=self._world_cache.get(key)
        if cached is not None:return cached
        scene,curves,ads=self._world_values(qw);counts=[len(x)for x in curves];world=np.concatenate([*curves,ads])
        jw=None
        if jac:
            jw=np.empty((*world.shape,self.world_count));h=self.geometry_step
            for k in range(self.world_count):
                qp,qm=qw.copy(),qw.copy();qp[k]+=h;qm[k]-=h
                _,cp,ap=self._world_values(qp);_,cm,am=self._world_values(qm)
                if any(a.shape!=b.shape for a,b in zip(cp,curves))or any(a.shape!=b.shape for a,b in zip(cm,curves)):raise ValueError('Finite material knot/order changed')
                jw[:,:,k]=(np.concatenate([*cp,ap])-np.concatenate([*cm,am]))/(2*h)
        result=(scene,curves,ads,world,jw,counts)
        self._world_cache[key]=result
        if len(self._world_cache)>4:self._world_cache.pop(next(iter(self._world_cache)))
        return result

    def evaluate(self,q,jacobian=True):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,):raise ValueError('Complete actual camera/world local state required')
        scene,curves,ads,world,jw,counts=self.world_data(q[14:],jacobian)
        projected,depth,valid,jp=project_native(q[:14],world,jacobian=jacobian,world_jacobian=jw)
        mids=np.concatenate([.5*(w[:-1]+w[1:])for w in curves]);_,_,midvalid,_=project_native(q[:14],mids)
        values=[];js=[];reports=[];offset=0;mo=0
        for row,n in zip(self.rows,counts):
            p=projected[offset:offset+n];v=valid[offset:offset+n];j=None if jp is None else jp[offset:offset+n]
            legal=v[:-1]&v[1:]&midvalid[mo:mo+n-1]
            raw,J,ix,t=nearest_finite(row['uv'],p,j,legal)
            values.append(raw.ravel());
            if jacobian:js.append(J.reshape(-1,self.parameter_count))
            reports.append(dict(id=row['id'],target=row['target'],physical_feature_id=row['physical_feature_id'],**stats(raw),native_residual_xy_px=raw.tolist(),selected_finite_segment=ix.tolist(),material_segment_fraction=t.tolist(),endpoint_points=int(np.sum((t==0)|(t==1))),original_finite_segments=len(legal),currently_legal_segments=int(legal.sum())))
            offset+=n;mo+=n-1
        if not valid[offset:].all():raise ValueError('Actual manual print corner lost positive physical depth')
        aduv=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']]);adraw=projected[offset:]-aduv
        values.append(adraw.ravel())
        if jacobian:js.append(jp[offset:].reshape(-1,self.parameter_count))
        for row,raw in zip(self.ad_packet['rows'],adraw):reports.append(dict(id=row['id'],physical_feature_id='Ad'+str(row['user_ad_number']),target=row['original_observation_target'],**stats(raw[None,:]),native_residual_xy_px=raw[None,:].tolist()))
        native=np.concatenate(values);residual=native/self.scales;J=np.vstack(js)/self.scales[:,None]if jacobian else None
        self.calls+=1
        return residual,J,reports,scene,native

    def cost(self,residual):
        a=abs(residual);return float(self.weights@np.where(a<=2,.5*residual**2,2*a-2))

    def irls(self,residual):return self.weights*np.minimum(1.,2/np.maximum(abs(residual),1e-300))

