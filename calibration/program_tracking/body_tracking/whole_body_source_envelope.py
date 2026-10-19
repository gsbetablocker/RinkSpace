"""Whole-person source envelope constraints for one rigid body root.

The observations are existing detector/SAM2 image envelopes, not anatomical
joint measurements. The same saved local body moves only by one world R/t.
Directions are correlated summaries of one silhouette; normalization prevents
adding eight independent copies of its evidence. Source roles never depend on
the trial projection or its residual. No inference, pose update or contact term.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DIRECTIONS = np.asarray([[1,0],[-1,0],[0,1],[0,-1],
                         [1,1],[-1,1],[1,-1],[-1,-1]], np.float64)
DIRECTIONS[4:] /= np.sqrt(2.)


def projected_envelope(uv, directions=DIRECTIONS):
    """Exact convex supports; max derivative is the selected support vertex."""
    uv = np.asarray(uv, np.float64)
    direction = np.asarray(directions, np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError('Finite projected points for the entire supplied body required')
    values = uv @ direction.T
    ids = np.argmax(values, axis=0)
    return values[ids, np.arange(len(direction))], ids


def envelope_pixel_factor(uv, source_support_px, qualified, *,
                          directions=DIRECTIONS, tolerance_px=0., sigma_px=1.,
                          uv_jacobian=None, lower_bound_only=False):
    """Return interval residual/J; every source-admitted direction is retained.

    Tolerance is an explicit algorithmic clothing/segmentation allowance, not
    measured reading precision. Outside that interval the residual is ordinary
    native pixel error divided by sigma and sqrt(number of active directions).
    The derivative is exact except the usual support ties/interval corners.
    An upstream robust loss may be applied, but no residual gate deletes rows.
    """
    support, ids = projected_envelope(uv, directions)
    obs = np.asarray(source_support_px, np.float64)
    active = np.asarray(qualified, bool)
    tol = np.broadcast_to(np.asarray(tolerance_px, float), support.shape)
    sigma = np.broadcast_to(np.asarray(sigma_px, float), support.shape)
    lower = np.broadcast_to(np.asarray(lower_bound_only, bool), support.shape)
    if obs.shape != support.shape or active.shape != support.shape:
        raise ValueError('Source support/qualification differs from fixed directions')
    if np.any(active & ~np.isfinite(obs)) or np.any(tol < 0) or np.any(sigma <= 0):
        raise ValueError('Invalid fixed source envelope settings')
    error = support - np.where(active, obs, support)
    raw = np.where(lower,np.minimum(error+tol,0.),
                   np.sign(error)*np.maximum(np.abs(error)-tol,0.))
    gain = active.astype(float) / (sigma * np.sqrt(max(1, int(active.sum()))))
    derivative = active & np.where(lower,error < -tol,np.abs(error)>tol)
    result = dict(residual=raw*gain, raw_error_native_px=error,
                  interval_error_native_px=raw, predicted_support_native_px=support,
                  support_vertex_index=ids, source_qualified=active.copy(),
                  source_lower_bound_only=lower.copy(),
                  interval_derivative=derivative.astype(float),
                  residual_derivative_gain=gain*derivative,
                  normalization='one correlated whole-person source envelope')
    if uv_jacobian is not None:
        j = np.asarray(uv_jacobian, np.float64)
        if j.ndim != 3 or j.shape[:2] != np.asarray(uv).shape:
            raise ValueError('UV Jacobian must be body_points x2 x parameters')
        d = np.asarray(directions, np.float64)
        js = np.einsum('ki,kij->kj', d, j[ids])
        result['jacobian'] = js*(gain*derivative)[:,None]
    return result


@dataclass
class NativeEnvelopeEvaluation:
    residual: np.ndarray
    jacobian_root_P_m: np.ndarray | None
    jacobian_world_rotation_delta: np.ndarray | None
    projected_body_native_uv: np.ndarray
    predicted_support_native_px: np.ndarray
    raw_error_native_px: np.ndarray
    support_vertex_index: np.ndarray


class NativeBodyEnvelopeFactor:
    """Fixed-camera R/t factor, metre-local complete mesh/envelope samples.

    P is native SMPL-X joint0 in metres. Q rotates the saved already-scaled
    rootlocal points to world. The Jacobian's rotation uses left world increments
    exp(delta)@Q; the caller maps it into its sequence parameterization.
    No camera, scale, articulation or per-joint freedom is introduced.
    """
    def __init__(self, local_points_m, source_support_px, qualified, *,
                 metres_per_scene_unit, tolerance_px=0., sigma_px=1.,
                 directions=DIRECTIONS, source_pts=None,lower_bound_only=False):
        self.local = np.asarray(local_points_m, np.float64)
        self.source = np.asarray(source_support_px, np.float64)
        self.qualified = np.asarray(qualified, bool)
        self.directions = np.asarray(directions, np.float64)
        self.unit = float(metres_per_scene_unit)
        self.tolerance, self.sigma = tolerance_px, sigma_px
        self.lower=np.broadcast_to(np.asarray(lower_bound_only,bool),self.source.shape)
        self.source_pts = source_pts
        if self.local.ndim != 2 or self.local.shape[1] != 3 or not np.isfinite(self.local).all():
            raise ValueError('Same-body finite scaled metre-local envelope vertices required')
        if self.unit <= 0 or not np.isfinite(self.unit):
            raise ValueError('Positive metre/scene conversion required')

    def value_and_jacobians(self, camera25, root_P_m, Q_world, *, need_jacobian=True):
        try:
            from ..runtime.bootstrap import add_geometry_dependencies
            from ..projection import project_central_valid
        except ImportError:
            from runtime.bootstrap import add_geometry_dependencies
            from projection import project_central_valid
        add_geometry_dependencies()
        q = np.asarray(Q_world, np.float64)
        p = np.asarray(root_P_m, np.float64)
        if q.shape != (3,3) or p.shape != (3,) or not np.allclose(q@q.T,np.eye(3),atol=1e-7):
            raise ValueError('One finite body root and proper rigid rotation required')
        offsets = self.local @ q.T
        world = (offsets+p)/self.unit
        result = project_central_valid(camera25,world,with_jacobian=False)
        uv, valid = result[0], result[-1]
        if not np.all(valid):
            raise ValueError(f'Whole-body envelope leaves optical domain at {self.source_pts}')
        j = None
        if need_jacobian:
            _,support_ids=projected_envelope(uv,self.directions)
            selected_result=project_central_valid(camera25,world[support_ids],with_jacobian=True)
            if not np.all(selected_result[-1]):
                raise ValueError('A retained whole-body support has invalid optics')
            # project_fast parameterizes camera C/100; root translation is metres.
            jxyz = (-selected_result[2][:,:3]/(100.*self.unit)).reshape(-1,2,3)
            x,y,z = offsets[support_ids].T
            neg_skew = np.zeros((len(support_ids),3,3))
            neg_skew[:,0,1]=z;neg_skew[:,0,2]=-y
            neg_skew[:,1,0]=-z;neg_skew[:,1,2]=x
            neg_skew[:,2,0]=y;neg_skew[:,2,1]=-x
            jrot = np.einsum('nij,njk->nik',jxyz,neg_skew)
            # Keep only eight extreme-point Jacobians. Do not allocate or
            # differentiate all surface samples merely to discard them.
            support_j=np.einsum('ki,kij->kj',self.directions,np.concatenate((jxyz,jrot),axis=2))
        e = envelope_pixel_factor(uv,self.source,self.qualified,
            directions=self.directions,tolerance_px=self.tolerance,
            sigma_px=self.sigma,lower_bound_only=self.lower)
        if need_jacobian:
            j=support_j*e['residual_derivative_gain'][:,None]
        return NativeEnvelopeEvaluation(e['residual'],
            None if j is None else j[:,:3],
            None if j is None else j[:,3:],uv,
            e['predicted_support_native_px'],e['raw_error_native_px'],
            e['support_vertex_index'])


def load_source_envelope(path, actual_pts):
    """Exact source/canonical PTS lookup; an alias is never treated as new data."""
    with np.load(path,allow_pickle=False) as z:
        stamps=z['actual_pts'];requested=np.asarray(actual_pts,np.int64)
        ids=np.searchsorted(stamps,requested)
        if np.any(ids>=len(stamps)) or not np.array_equal(stamps[ids],requested):
            raise ValueError('Requested actual exposure has no exact source envelope')
        return {k:z[k][ids].copy() for k in ('source_support_native_px',
            'source_envelope_qualified','source_tolerance_native_px',
            'source_sigma_native_px','source_bbox_native_xyxy',
            'source_envelope_kind','source_selected_target_status',
            'source_envelope_lower_bound_only')}


class FrozenCameraEnvelopeBatch:
    """Vectorized full-source support factor for current square Brown-k1 Z190.

    Supports arbitrary fx/fy/PP; k1 is the sole nonzero distortion coefficient
    of this current installation. A different optical model is explicitly
    rejected rather than silently projected without its extra terms.
    """
    def __init__(self,camera25,local_points_m,source_support_px,qualified,*,
                 metres_per_scene_unit,tolerance_px=0.,sigma_px=1.,directions=DIRECTIONS,
                 lower_bound_only=False):
        try:
            from ..runtime.bootstrap import add_geometry_dependencies
        except ImportError:
            from runtime.bootstrap import add_geometry_dependencies
        add_geometry_dependencies()
        import cv2
        self.camera=np.asarray(camera25,np.float64)
        self.local=np.asarray(local_points_m,np.float64)
        self.obs=np.asarray(source_support_px,np.float64)
        self.active=np.asarray(qualified,bool)
        self.directions=np.asarray(directions,np.float64)
        self.unit=float(metres_per_scene_unit)
        self.tol=np.broadcast_to(np.asarray(tolerance_px,float),self.obs.shape)
        self.sigma=np.broadcast_to(np.asarray(sigma_px,float),self.obs.shape)
        self.lower=np.broadcast_to(np.asarray(lower_bound_only,bool),self.obs.shape)
        N=len(self.camera)
        if self.camera.shape!=(N,25)or self.local.ndim!=3 or self.local.shape[0]!=N or self.local.shape[2]!=3:
            raise ValueError('Per-exposure frozen cameras and same-body surface points required')
        if np.any(self.camera[:,12:]!=0):
            raise ValueError('Batch envelope requires current native Brown sole-k1 optical model')
        if np.any(self.sigma<=0)or np.any(self.tol<0)or self.unit<=0:
            raise ValueError('Invalid fixed source envelope allowances')
        self.R=np.stack([cv2.Rodrigues(r)[0]for r in self.camera[:,3:6]])
        self.f=np.exp(self.camera[:,6:8]);self.pp=self.camera[:,8:10];self.k1=self.camera[:,11]
        # Missing local poses only disable that source envelope. They do not
        # remove a global trajectory/root-control node or infer new articulation.
        self.pose_valid=np.isfinite(self.local).all(axis=(1,2))
        self.active=self.active&self.pose_valid[:,None]
        self.safe_local=np.nan_to_num(self.local)
        self.gain=self.active/(self.sigma*np.sqrt(np.maximum(1,self.active.sum(1)))[:,None])

    def value_and_jacobians(self,P_m,Q_world,*,need_jacobian=True,return_uv=False):
        p=np.asarray(P_m,np.float64);q=np.asarray(Q_world,np.float64);N=len(p)
        if p.shape!=(N,3)or q.shape!=(N,3,3)or not np.isfinite(p).all()or not np.isfinite(q).all():
            raise ValueError('Finite global body root/rotation required even at inferred source times')
        offset=np.einsum('nij,nvj->nvi',q,self.safe_local)
        world=(offset+p[:,None])/self.unit
        cam=np.einsum('nij,nvj->nvi',self.R,world-self.camera[:,None,:3])
        if np.any(self.pose_valid[:,None]&(cam[:,:,2]<=0)):
            raise ValueError('Retained whole-body envelope behind current optical centre')
        z=np.maximum(cam[:,:,2],1e-12);xy=cam[:,:,:2]/z[:,:,None]
        r2=np.sum(xy*xy,axis=2);radial=1+self.k1[:,None]*r2
        radial_derivative=1+3*self.k1[:,None]*r2
        if np.any(self.pose_valid[:,None]&((radial<=0)|(radial_derivative<=0))):
            raise ValueError('Retained whole-body envelope outside current central Brown domain')
        uv=xy*radial[:,:,None]*self.f[:,None]+self.pp[:,None]
        dots=np.einsum('nvi,ki->nvk',uv,self.directions)
        vi=np.argmax(dots,axis=1);ni=np.arange(N)[:,None];ki=np.arange(len(self.directions))[None]
        support=dots[ni,vi,ki]
        error=support-np.where(self.active,self.obs,support)
        raw=np.where(self.lower,np.minimum(error+self.tol,0.),
                     np.sign(error)*np.maximum(np.abs(error)-self.tol,0.))
        derivative=np.where(self.lower,error < -self.tol,np.abs(error)>self.tol)&self.active
        out=dict(residual=raw*self.gain,predicted_support_native_px=support,
                 raw_error_native_px=error,support_vertex_index=vi,
                 source_lower_bound_only=self.lower,interval_derivative=derivative.astype(float),
                 residual_derivative_gain=self.gain*derivative)
        if return_uv:out['projected_body_native_uv']=uv
        if need_jacobian:
            x=xy[ni,vi,0];y=xy[ni,vi,1];zs=z[ni,vi];k=self.k1[:,None]
            radial_selected=1+k*(x*x+y*y)
            normJ=np.zeros((N,len(self.directions),2,3))
            xx=radial_selected+2*k*x*x;yy=radial_selected+2*k*y*y;cross=2*k*x*y
            normJ[:,:,0,0]=self.f[:,0,None]*xx/zs
            normJ[:,:,0,1]=self.f[:,0,None]*cross/zs
            normJ[:,:,0,2]=-self.f[:,0,None]*(xx*x+cross*y)/zs
            normJ[:,:,1,0]=self.f[:,1,None]*cross/zs
            normJ[:,:,1,1]=self.f[:,1,None]*yy/zs
            normJ[:,:,1,2]=-self.f[:,1,None]*(cross*x+yy*y)/zs
            pointJ=np.einsum('nkij,njl->nkil',normJ,self.R)/self.unit
            rootJ=np.einsum('ki,nkij->nkj',self.directions,pointJ)
            selected_offset=offset[ni,vi];xx,yy,zz=selected_offset.transpose(2,0,1)
            negskew=np.zeros((N,len(self.directions),3,3))
            negskew[:,:,0,1]=zz;negskew[:,:,0,2]=-yy
            negskew[:,:,1,0]=-zz;negskew[:,:,1,2]=xx
            negskew[:,:,2,0]=yy;negskew[:,:,2,1]=-xx
            rotationJ=np.einsum('nki,nkij->nkj',rootJ,negskew)
            gain=out['residual_derivative_gain']
            out['jacobian_root_P_m']=rootJ*gain[:,:,None]
            out['jacobian_world_rotation_delta']=rotationJ*gain[:,:,None]
        return out
