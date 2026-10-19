"""Exact global CUDA filters with sparse full-width coefficient readback.

Only the final device-to-host transfer changes. The full-image Gaussian and
mirror cubic recurrences, RGB arithmetic, source queries, and RNG stay intact.
"""
import ctypes as C
import numpy as np
from .ptx_source_filter import PTXSourceFilterCUDA


def frozen_normal_y_ranges(rows, offsets, height=1080, width=1920):
    seed=np.asarray([r['navigation_seed_uv'] for r in rows],float)
    normal=np.asarray([r['normal'] for r in rows],float)
    xy=seed[:,None,:]+np.asarray(offsets)[None,:,None]*normal[:,None,:]
    if not np.isfinite(xy).all():raise ValueError('Nonfinite frozen source query')
    outside=bool(np.any((xy[:,:,0]<0)|(xy[:,:,0]>width-1)|(xy[:,:,1]<0)|(xy[:,:,1]>height-1)))
    if outside:
        return dict(ranges=[[0,height]],queried_rows=height,full_fallback='native_boundary_conservative_full_coefficients',source_queries=len(rows)*len(offsets))
    base=np.floor(xy[:,:,1]).astype(np.int64)
    ys=np.unique(np.clip((base[:,:,None]+np.asarray([-1,0,1,2])).ravel(),0,height-1))
    ranges=[];lo=previous=int(ys[0])
    for y in ys[1:]:
        y=int(y)
        if y!=previous+1:ranges.append([lo,previous+1]);lo=y
        previous=y
    ranges.append([lo,previous+1])
    return dict(ranges=ranges,queried_rows=len(ys),full_fallback=None,source_queries=len(rows)*len(offsets),
        interpolation_stencil='all four cubic coefficient rows, including zero-weight slots')


class PTXSourceFilterROI(PTXSourceFilterCUDA):
    def __init__(self,height,width,*,coefficient_y_ranges,coefficient_full_shadow_calls=2,**kwargs):
        super().__init__(height,width,**kwargs)
        self.ranges=[(int(a),int(b)) for a,b in coefficient_y_ranges]
        previous=0
        for a,b in self.ranges:
            if not 0<=a<b<=self.h or a<previous:raise ValueError('Invalid coefficient readback row intervals')
            previous=b
        if not self.ranges:raise ValueError('Empty coefficient readback support')
        self._host_coeff=np.full((1,self.h,self.w),np.nan,np.float64)
        self.full_shadow_calls=int(coefficient_full_shadow_calls);self.calls=0
        self.full_shadow_coefficients=None;self.readback_shadows=[]
        self.rows_downloaded=sum(b-a for a,b in self.ranges)
        self._row_mask=np.zeros(self.h,bool)
        for a,b in self.ranges:self._row_mask[a:b]=True

    def __call__(self,channel,*,include_gaussian=True):
        a=np.ascontiguousarray(channel,dtype=np.float32);single=a.ndim==2
        if single:a=a[None]
        if a.shape!=(1,self.h,self.w):raise ValueError('ROI candidate supports the existing one-channel board recipe only')
        b=len(a);count=a.size;src,tmp,g,c=self.memory;I=C.c_int;D=C.c_double;U=C.c_uint64
        self.check(self.driver.cuMemcpyHtoD_v2(src,a.ctypes.data,a.nbytes))
        scalars=[D(self.kernel[i]) for i in (3,4,5,6)]
        for axis,p,q in ((0,src,tmp),(1,tmp,g)):
            self.launch('gauss',(count+255)//256,256,[U(p.value),U(q.value),I(self.h),I(self.w),I(count),I(axis),*scalars])
        self.launch('convert',(count+255)//256,256,[U(g.value),U(c.value),I(count)])
        z=-0.267949192431122706472553658494127633;gain=(1-z)*(1-1/z)
        for axis,lines in ((0,b*self.w),(1,b*self.h)):
            self.launch('spline',(lines+127)//128,128,[U(c.value),I(self.h),I(self.w),I(b),I(axis),D(z),D(gain)])
        gaussian=np.empty(a.shape,np.float32) if include_gaussian else None
        if include_gaussian:self.check(self.driver.cuMemcpyDtoH_v2(gaussian.ctypes.data,g,gaussian.nbytes))
        self.full_shadow_coefficients=None
        if self.calls<self.full_shadow_calls:
            full=np.empty(a.shape,np.float64)
            self.check(self.driver.cuMemcpyDtoH_v2(full.ctypes.data,c,full.nbytes))
            self.full_shadow_coefficients=full[0]
        for lo,hi in self.ranges:
            byte_offset=lo*self.w*8;size=(hi-lo)*self.w*8
            self.check(self.driver.cuMemcpyDtoH_v2(self._host_coeff.ctypes.data+byte_offset,U(c.value+byte_offset),size))
        if self.calls<self.full_shadow_calls:
            error=float(abs(self._host_coeff[0,self._row_mask]-self.full_shadow_coefficients[self._row_mask]).max(initial=0.))
            poison=bool(np.isnan(self._host_coeff[0,~self._row_mask]).all())
            if error!=0 or not poison:raise ValueError('ROI copy changed coefficients or poisoned outside rows')
            self.readback_shadows.append(dict(call=self.calls,kept_coeff_max_abs=error,all_non_downloaded_rows_NaN=poison,
                full_coefficients_downloaded_for_exact_shadow_only=True))
        self.calls+=1
        coeff=self._host_coeff[0] if single else self._host_coeff
        if not include_gaussian:return coeff
        return (gaussian[0],coeff) if single else (gaussian,coeff)
