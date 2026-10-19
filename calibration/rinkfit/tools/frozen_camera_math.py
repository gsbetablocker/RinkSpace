"""Independent NumPy/SciPy replay of the frozen current row camera.

This module does not load the production model or authentication loader.
The equations have been reconstructed from the source and agree with all
five saved sensor rows of all 1248 exposures to <= 1.14e-13 in camera25.
It is a read-only audit utility, not an authenticated production continuation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_RUN=Path(__file__).resolve().parents[2]/'review/data/audit_1017/motion_joint_body2'

def decode(value, arrays):
    if isinstance(value,dict):
        if set(value)=={'_safe_npz_array'}:return arrays[value['_safe_npz_array']].copy()
        if set(value)=={'_safe_tuple'}:return tuple(decode(w,arrays)for w in value['_safe_tuple'])
        return {k:decode(w,arrays)for k,w in value.items()}
    if isinstance(value,list):return [decode(w,arrays)for w in value]
    return value

def unit(x):
    x=np.asarray(x,float)
    return x/np.linalg.norm(x)

def exp_axis(axis,angle):
    x,y,z=axis;S=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    c=np.cos(angle)[...,None,None];s=np.sin(angle)[...,None,None]
    return c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*S

class FrozenCamera:
    def __init__(self, run=DEFAULT_RUN, *, q=None):
        run=Path(run)
        with np.load(run/'final_complete_state.npz',allow_pickle=False) as state:
            self.q=state['physical_programme_q_augmented'].copy() if q is None else np.asarray(q,float).copy()
            self.canonical_q_sha256=str(state['canonical_q_sha256'])
            self.axis_index=int(state['aux__two_axis_a_column'][0])
            self.head_indices=state['aux__shared_head_axes_columns'].astype(int)
            self.optics_indices=state['aux__shared_nonlinear_optics_columns'].astype(int)
        packet=json.loads((run/'reviews/final/selected/frames.json').read_text())
        with np.load(run/'reviews/final/selected/arrays.npz',allow_pickle=False) as arrays:
            m=decode(packet['payload']['camera_state'],arrays)
        q=self.q;self.metadata=m;self.initial_q=m['initial_q'];b=m['base_calibration']
        self.scene_unit_working_m=u=m['scene_unit_working_m'];d=q[118:127]-self.initial_q[118:127]
        o=q[self.optics_indices];z=q[m['zoom_parameter_indices']]
        self.qref=m['zoom_configuration']['reference_domain_log_focal_ratio']
        self.P=np.array(b['pivot_xyz'])+d[:3]/u
        self.lever=np.array(b['lever_xyz'])+d[3:6]/u
        self.ppx=np.array([1920*b['principal_x_fraction_coefficients'][0]+d[6],z[0],o[1]])
        self.ppy=np.array([1080*b['principal_y_fraction_coefficients'][0]+d[7],z[1],o[2]])
        self.k1=np.array([b['k1_coefficients'][0]+.001*d[8],.001*z[2],.001*o[3]])
        self.k2=.001*o[4:7];self.tangential=.001*o[7:9]+b['tangential_p1_p2']
        self.axial=np.array([0.,z[3]/u,o[0]/u]);self.axis=q[self.axis_index]/u
        self.readout_ms=float(q[m['row_readout_index']])
        self.views=q[127:5119].reshape(1248,4)
        s=q[self.head_indices]
        # Exact local unit-axis charts currently used: n0=(0,0,1), b0=(1,0,0).
        self.n=unit([s[0],s[1],1.]);self.b=unit([1.,s[2],s[3]])
        n,b=self.n,self.b
        t=unit(np.array([1.,0.,0.])-n*n[0]);c=unit(np.array([0.,1.,0.])-b*b[1])
        W=np.column_stack([t,np.cross(n,t),n]);V=np.column_stack([b,np.cross(c,b),c])
        self.R0=V@W.T

    def camera25(self, frame, row):
        """frame is 1-based programme exposure; row is original 1080p UV y.

        Inputs are broadcast, return shape = broadcast_shape + (25,).
        Includes the documented terminal half-exposure extrapolation.
        """
        frames,rows=np.broadcast_arrays(np.asarray(frame),np.asarray(row,float))
        if np.any(frames!=np.floor(frames)) or np.any(frames<1) or np.any(frames>1248):
            raise ValueError('programme exposure index must be 1..1248')
        shape=frames.shape;frames=frames.ravel();rows=rows.ravel()
        t=frames-1+self.readout_ms/20.*(rows/1079.-.5)
        if np.any(t<-.5-1e-10) or np.any(t>1247.5+1e-10):
            raise ValueError('row time outside supported first-shot clock')
        left=np.clip(np.floor(t).astype(int),0,1246);a=t-left
        v=(1-a[:,None])*self.views[left]+a[:,None]*self.views[left+1]
        R=exp_axis(self.b,v[:,1])@self.R0@exp_axis(self.n,v[:,0])
        logf=v[:,3];z=logf-self.qref
        d=np.tile(self.lever,(len(v),1));d[:,2]+=np.polynomial.polynomial.polyval(z,self.axial)
        C=self.P+np.einsum('mji,mj->mi',R,d)+self.axis*np.cross(self.n,np.einsum('mji,j->mi',R,self.b))
        cam=np.zeros((len(v),25));cam[:,:3]=C;cam[:,3:6]=Rotation.from_matrix(R).as_rotvec()
        cam[:,6:8]=(np.log(1920.)+logf)[:,None]
        cam[:,8]=np.polynomial.polynomial.polyval(z,self.ppx)
        cam[:,9]=np.polynomial.polynomial.polyval(z,self.ppy)
        cam[:,11]=np.polynomial.polynomial.polyval(z,self.k1)
        cam[:,12]=np.polynomial.polynomial.polyval(z,self.k2)
        cam[:,13:15]=self.tangential
        return cam.reshape(shape+(25,))

    def project(self, xyz, frame, row):
        return project(self.camera25(frame,row),xyz)

def project(camera25, xyz):
    """Brown4 forward projection, return UV and optical depth.

    No domain predicate or silhouette association is reproduced here.
    Callers must not interpret finite offscreen output as a visible target.
    A scalar camera can project N points; camera array and xyz broadcast.
    """
    c=np.asarray(camera25,float);x=np.asarray(xyz,float)
    shape=np.broadcast_shapes(c.shape[:-1],x.shape[:-1])
    c=np.broadcast_to(c,shape+(25,));x=np.broadcast_to(x,shape+(3,))
    R=Rotation.from_rotvec(np.array(c[...,3:6].reshape(-1,3),copy=True)).as_matrix().reshape(shape+(3,3))
    v=np.einsum('...ij,...j->...i',R,x-c[...,:3]);xx=v[...,0]/v[...,2];yy=v[...,1]/v[...,2]
    r2=xx*xx+yy*yy;k1,k2,p1,p2=np.moveaxis(c[...,11:15],-1,0)
    radial=1.+k1*r2+k2*r2*r2
    xd=xx*radial+2*p1*xx*yy+p2*(r2+2*xx*xx)
    yd=yy*radial+p1*(r2+2*yy*yy)+2*p2*xx*yy
    uv=np.stack([np.exp(c[...,6])*xd+c[...,10]*yd+c[...,8],np.exp(c[...,7])*yd+c[...,9]],axis=-1)
    return uv,v[...,2]

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN,help='Extracted motion_joint_body2 directory')
    args=parser.parse_args();model=FrozenCamera(args.run)
    with np.load(args.run/'final_source_ordered/cameras.npz',allow_pickle=False)as saved:
        value=model.camera25(np.arange(1,1249)[:,None],saved['image_rows'][None,:])
        error=float(np.max(abs(value-saved['camera25'])))
    print(json.dumps(dict(canonical_q_sha256=model.canonical_q_sha256,frames=1248,rows=5,
        camera25_max_abs_error=error,production_owner_replayed=False),indent=2))
