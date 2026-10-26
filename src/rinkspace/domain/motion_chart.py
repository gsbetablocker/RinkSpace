"""Uniform cubic motion coordinates, exactly refined onto the native grid.

This changes the fitted trajectory space explicitly. It does not filter a
finished projection or change the installation, optics, observations or clock.
"""
from dataclasses import dataclass
import numpy as np


def cubic_samples(count,first,dt,times):
    """Indices and value weights in MotionSpline's centered cubic convention."""
    t=np.atleast_1d(np.asarray(times,float));x=(t-first)/dt
    if not np.isfinite(t).all() or np.any(x<1-1e-10) or np.any(x>count-2+1e-10):
        raise ValueError('Motion sample outside native cubic support')
    cell=np.clip(np.floor(x).astype(int),1,count-3);u=x-cell;v=1-u
    w=np.c_[v**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3]/6
    return cell[:,None]+np.arange(-1,3),w


@dataclass(frozen=True)
class UniformMotionChart:
    first: float
    dt: float
    stride: int
    refinement: np.ndarray
    coarse_first: float
    reference_time: float

    @classmethod
    def create(cls,count,first,dt,stride=1,reference_time=0.):
        if isinstance(stride,bool) or int(stride)!=stride or stride<1:
            raise ValueError('Motion knot stride must be a positive integer')
        if count<4 or not np.isfinite([first,dt,reference_time]).all() or dt<=0:
            raise ValueError('Invalid native motion grid')
        stride=int(stride)
        # Cubic cardinal refinement mask: (1+x+...+x^(m-1))^4/m^3.
        # It is centered on each coarse coefficient's native grid coordinate.
        mask=np.ones(1)
        for _ in range(4):mask=np.convolve(mask,np.ones(stride))
        mask/=stride**3;radius=2*(stride-1)
        begin=int(np.ceil(-radius/stride));end=(count-1+radius)//stride
        centres=stride*np.arange(begin,end+1);offset=np.arange(count)[:,None]-centres[None,:]+radius
        valid=(offset>=0)&(offset<len(mask));matrix=np.zeros(offset.shape)
        matrix[valid]=mask[offset[valid]]
        if np.any(np.max(matrix,axis=0)==0):raise ValueError('Unobservable outside-support motion column')
        chart=cls(float(first),float(dt),stride,matrix,float(first+begin*stride*dt),float(reference_time))
        chart.reference_weights() # Validate the reference time in native support.
        return chart

    def reference_weights(self):
        i,w=cubic_samples(len(self.refinement),self.first,self.dt,[self.reference_time])
        return w[0]@self.refinement[i[0]]

    def directions(self):
        """Columns for three moving channels, with two reference-angle gauges."""
        w=self.reference_weights();pivot=int(np.argmax(np.abs(w)));columns=[];labels=[]
        for k in range(self.refinement.shape[1]):
            for channel in range(3):
                if channel<2 and k==pivot:continue
                d=np.zeros((len(self.refinement),3));d[:,channel]=self.refinement[:,k]
                if channel<2:d[:,channel]-=self.refinement[:,pivot]*(w[k]/w[pivot])
                columns.append(d);labels.append((k,channel))
        return np.stack(columns,axis=2),labels

    def initialize(self,controls,*,bound_zoom=False):
        """Approximate an explicit old state once, then fit in the new space.

        Least squares here only chooses initial control coefficients. Its
        sample residuals and weights never enter the pixel fitting objective.
        The existing reference pan/tilt values are preserved exactly.
        """
        controls=np.asarray(controls,float)
        if controls.shape!=(len(self.refinement),3) or not np.isfinite(controls).all():
            raise ValueError('Motion controls do not match the native grid')
        if self.stride==1:return controls.copy()
        times=self.first+self.dt*np.arange(1,len(controls)-1)
        indices,weights=cubic_samples(len(controls),self.first,self.dt,times)
        A=np.einsum('ij,ijk->ik',weights,self.refinement[indices])
        target=np.einsum('ij,ijk->ik',weights,controls[indices])
        ref_i,ref_w=cubic_samples(len(controls),self.first,self.dt,[self.reference_time])
        reference=ref_w[0]@controls[ref_i[0]]
        w=self.reference_weights();pivot=int(np.argmax(np.abs(w)))
        free=np.delete(np.arange(len(w)),pivot);D=np.eye(len(w))[:,free]
        D[pivot]=-w[free]/w[pivot];C=np.empty((len(w),3))
        for channel in range(3):
            if channel==2:
                if bound_zoom:
                    # A convex-combination cubic with coefficients inside an
                    # already valid zoom range cannot overshoot that range.
                    # This is only an initial state, never a fitted bound.
                    from scipy.optimize import lsq_linear
                    lo=float(target[:,2].min());hi=float(target[:,2].max())
                    if lo==hi:C[:,channel]=lo
                    else:
                        result=lsq_linear(A,target[:,2],bounds=(lo,hi),tol=1e-12)
                        if not result.success:raise RuntimeError('Coarse zoom initializer did not solve: '+result.message)
                        C[:,channel]=result.x
                else:C[:,channel]=np.linalg.lstsq(A,target[:,channel],rcond=None)[0]
            else:
                offset=np.zeros(len(w));offset[pivot]=reference[channel]/w[pivot]
                C[:,channel]=offset+D@np.linalg.lstsq(A@D,target[:,channel]-A@offset,rcond=None)[0]
        return self.refinement@C
