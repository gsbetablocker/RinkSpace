"""Private main9 source adapter; no production/helper file mutation.

C is a horizontal image scanline paint-support measurement. The observed
source y selects a scanline, not a nearest-curve tangent target. Preserve x
support only at the unique projected physical line crossing that scanline.
"""
from pathlib import Path
import sys
P=Path('D:/OneDrive/codex_projs/RinkSpace/three_frame_refit/program_tracking');sys.path.insert(0,str(P))
import numpy as np
from scipy.sparse import csr_matrix,vstack
from projection import project_central_valid
from reference_board_pair import BoardIntersectionError
from source_edge_intervals import SourceNormalIntervals
from calibration.solver_runtime.original_measured_location_mode import uses_original_normal_support
from z190_source_camera_diagnostic import absolute_value_jac15 as original_absolute,brown4_world_jac
from z190_low8_direct_source_rf_recovery import FixedCutBundle

def horizontal_unique_x_value_jac15(camera,row,jacobian=True):
    if (row.get('id')!='C' or row.get('scanline_fixed_axis')!='y' or row.get('scanline_branch')!='unique_x'
            or row.get('curve_observation_kind')!='axis_scanline_branch'):
        raise ValueError('Expected explicit source horizontal-scanline unique x observation')
    world=np.asarray(row['world'],float);uv=np.asarray(row['uv'],float)
    if len(world)<2 or len(uv)<1:raise ValueError('Unique line scan requires complete source-identified curve')
    if jacobian:
        projected,depth,j11,valid=project_central_valid(camera,world,with_jacobian=True)
        allj=np.c_[j11,brown4_world_jac(camera,world).reshape(-1,4)].reshape(-1,2,15)
    else:projected,depth,valid=project_central_valid(camera,world)
    guard=valid&(depth>0)&np.isfinite(projected).all(axis=1)
    if not np.all(guard):
        raise BoardIntersectionError('complete_horizontal_scanline_curve_not_central_forward',np.flatnonzero(~guard))
    start=projected[:-1];delta=np.diff(projected,axis=0);residual=np.zeros((len(uv),2));block=np.zeros((2*len(uv),15))if jacobian else None
    for k,point in enumerate(uv):
        cross=(abs(delta[:,1])>1e-12)&(np.minimum(start[:,1],projected[1:,1])<=point[1])&(np.maximum(start[:,1],projected[1:,1])>=point[1])
        ids=np.flatnonzero(cross)
        if not len(ids):raise BoardIntersectionError('source_horizontal_scanline_has_no_complete_curve_crossing',[k])
        tt=(point[1]-start[ids,1])/delta[ids,1];xx=start[ids,0]+tt*delta[ids,0]
        # Two adjacent segments may both contain an exact shared endpoint;
        # allow duplicate same-location crossings, reject physical ambiguity.
        if np.ptp(xx)>1e-7:raise ValueError('Source requires unique x, projected complete curve has multiple crossings')
        i=int(ids[0]);t=float(tt[0]);residual[k,0]=xx[0]-point[0]
        if jacobian:
            ja=allj[i];jd=allj[i+1]-ja;dt=(-ja[1]-t*jd[1])/delta[i,1]
            block[2*k]=ja[0]+t*jd[0]+delta[i,0]*dt
    return(residual.ravel(),block)if jacobian else residual.ravel()

def typed_absolute_value_jac15(camera,rows,jacobian=True):
    values=[];blocks=[]
    for row in rows:
        use_normal_support=uses_original_normal_support(row)
        if row.get('id')=='C' and row.get('scanline_fixed_axis')=='y' and row.get('scanline_branch')=='unique_x':
            if jacobian:residual,j=horizontal_unique_x_value_jac15(camera,row,True)
            else:residual=horizontal_unique_x_value_jac15(camera,row,False)
            if 'normal_interval'in row and use_normal_support:
                normal=np.asarray(row['normal'],float)
                if not np.all(normal==[1.,0.]):raise ValueError('Horizontal scanline source supports x only')
                support=SourceNormalIntervals(normal,row['normal_interval'],np.ones(len(row['uv']),bool))
                if jacobian:residual,matrices=support.apply(residual,[j]);j=matrices[0]
                else:residual=support.apply(residual)
            sigma=np.repeat(row['sigma'],2);values.append(residual/sigma)
            if jacobian:blocks.append(j/sigma[:,None])
        else:
            if jacobian:r,j=original_absolute(camera,[row],True);values.append(r);blocks.append(j)
            else:values.append(original_absolute(camera,[row],False))
    return(np.concatenate(values),np.vstack(blocks))if jacobian else np.concatenate(values)

class Main9TypedFixedCutBundle(FixedCutBundle):
    """Same complete finite/motion kernel and fixed-mount layout, private absolute adapter."""
    def evaluate(self,q,jacobian=True,include_priors=True):
        cache=self.cache(q,jacobian);vv,rr,dr,cc,ds,dv=cache;self.validate_finite_extents(cache)
        focal=np.exp(vv[:,3]);pp=np.array([1920*self.calibration.principal_x_fraction[0],1080*self.calibration.principal_y_fraction[0]])
        values=[];data=[]
        for lo in range(0,len(self.a),65536):
            hi=min(lo+65536,len(self.a));a,b=self.ia[lo:hi],self.ib[lo:hi]
            result=self.motion_kernel(fa=focal[a],fb=focal[b],ra=rr[a],rb=rr[b],
                da=None if not jacobian else dr[a],db=None if not jacobian else dr[b],
                ca=cc[a],cb=cc[b],dca_shared=None if not jacobian else ds[a],dcb_shared=None if not jacobian else ds[b],
                dca_view=None if not jacobian else dv[a],dcb_view=None if not jacobian else dv[b],
                a=self.a[lo:hi],b=self.b[lo:hi],pp=pp,k1=self.calibration.k1[0],planes=self.finite_planes[lo:hi],jacobian=jacobian,
                sigma=self.motion_sigma[2*lo:2*hi].reshape(-1,2),profile_weight=None if self.profile_weights is None else self.profile_weights[lo:hi])
            sigma=self.motion_sigma[2*lo:2*hi]
            if jacobian:
                residual,ja,jb,shared=result[:4];endpoints=np.c_[ja,jb];swap=self.swap_rows[2*lo:2*hi];endpoints[swap]=np.c_[jb[swap],ja[swap]]
                data.append((np.c_[shared,endpoints]/sigma[:,None]).ravel())
            else:residual=result[0]if isinstance(result,tuple)else result
            values.append(residual/sigma)
        cal=self.current_calibration(q);scale=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
        for p,rows in self.abs_groups:
            if jacobian:
                cam,sj,vj=cal.camera25_and_jacobians(self.view(q,p));residual,raw=typed_absolute_value_jac15(cam,rows,True)
                mapping=np.c_[sj[:15,self.mapping_names],vj[:15]]*scale[:,None];data.append((raw@mapping).ravel())
            else:residual=typed_absolute_value_jac15(self.camera(q,p),rows,False)
            values.append(residual)
        value=np.concatenate(values);prior,pj=self.priors(q,cache)if include_priors else(np.empty(0),None)
        if not jacobian:return np.r_[value,prior]
        jac=csr_matrix((np.concatenate(data),self.indices,self.pointer),shape=(self.source_residual_rows,len(q)))
        return(np.r_[value,prior],vstack((jac,pj),format='csr'))if len(prior)else(value,jac)
