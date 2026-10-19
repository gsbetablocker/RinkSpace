"""Bounded exact-course cache; opt in on an owned course instance only.

The station arithmetic is the certified original implementation verbatim.
Only its immutable perimeter and cumulative arc inputs are memoized. This
file does not install itself, alter source pixels, or modify an active run.
"""
from collections import OrderedDict
import hashlib
import inspect
import threading
import numpy as np
from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry

ORIGINAL_STATION_SHA256 = 'cce47f4547361c6e1d496733539f1529ef50456c363ed44f7c8213dc1220561a'
SCHEMA = 'rinkspace.exact-course6-readonly-bounded-station-input-cache.v1'


class ExactCourseStationCache:
    def __init__(self, course, *, max_cache_bytes=8*1024**2):
        if isinstance(max_cache_bytes,bool) or not isinstance(max_cache_bytes,int) or max_cache_bytes<0:
            raise ValueError('Nonnegative explicit cache byte budget required')
        self.course=course;self.max_cache_bytes=max_cache_bytes
        self._verify_original();self._reset()

    def _verify_original(self):
        c=self.course
        if not isinstance(c,SharedBoardCourseNumericGeometry) or c.parameter_count!=6:
            raise TypeError('Released SharedBoardCourseNumericGeometry course6 required')
        original=inspect.getsource(type(c).station)
        if hashlib.sha256(original.encode()).hexdigest()!=ORIGINAL_STATION_SHA256:
            raise ValueError('Original station implementation changed; new equivalence check required')
        expected=self.__dict__.get('base_sha256')
        if expected is not None and expected!=c.base_sha256:
            raise ValueError('Original course base identity changed during restoration')
        self.base_sha256=c.base_sha256

    def _reset(self):
        self._cache=OrderedDict();self._lock=threading.RLock();self.cache_bytes=0
        self.hits=0;self.misses=0;self.evictions=0
        self._needs_restore_verification=False

    def __getstate__(self):
        # Actor spawn gets an empty bounded cache, never a copied parent cache.
        return dict(course=self.course,max_cache_bytes=self.max_cache_bytes,base_sha256=self.base_sha256)

    def __setstate__(self,state):
        # The course->cache->course pickle cycle finishes restoring the course
        # AFTER this hook, so verify the original once on the first call.
        self.__dict__.update(state);self._reset();self._needs_restore_verification=True

    def clear(self):
        with self._lock:
            self._cache.clear();self.cache_bytes=0

    def statistics(self):
        with self._lock:
            return dict(schema=SCHEMA,entries=len(self._cache),cache_accounted_bytes=self.cache_bytes,
                max_cache_bytes=self.max_cache_bytes,hits=self.hits,misses=self.misses,evictions=self.evictions,
                original_station_sha256=ORIGINAL_STATION_SHA256,
                budget_scope='array payload plus conservative per-array/entry overhead',
                engineering_budget_not_user_rule=True,source_or_solver_changed=False)

    def _inputs(self,q):
        c=self.course
        if self._needs_restore_verification:
            self._verify_original();self._needs_restore_verification=False
        if c.base_sha256!=self.base_sha256:
            raise ValueError('Original course base identity changed')
        key=(float(c.scene_unit_m),np.ascontiguousarray(c.initial).tobytes(),q.tobytes())
        with self._lock:
            if key in self._cache:
                self.hits+=1;self._cache.move_to_end(key);return self._cache[key][0]
            self.misses+=1
            b,t,jb,jt,ranges=c.sample_perimeter(q,with_jacobian=True)
            if b.shape!=(480,3) or jb.shape!=(480,3,6):
                raise ValueError('Original finite480 course order required')
            edge=np.roll(b,-1,axis=0)-b
            je=np.roll(jb,-1,axis=0)-jb
            lengths=np.linalg.norm(edge[:,:2],axis=1)
            jl=np.einsum('ni,nik->nk',edge[:,:2],je[:,:2])/lengths[:,None]
            cum=np.r_[0.,np.cumsum(lengths)]
            jcum=np.vstack((np.zeros((1,6)),np.cumsum(jl,axis=0)))
            arrays=(b,t,jb,jt,edge,je,lengths,jl,cum,jcum)
            for a in arrays:a.flags.writeable=False
            value=dict(perimeter=(b,t,jb,jt,ranges),b=b,jb=jb,edge=edge,je=je,
                lengths=lengths,jl=jl,cum=cum,jcum=jcum)
            size=sum(a.nbytes+128 for a in arrays)+1024
            if size<=self.max_cache_bytes:
                while self._cache and self.cache_bytes+size>self.max_cache_bytes:
                    _,(_,old)=self._cache.popitem(last=False);self.cache_bytes-=old;self.evictions+=1
                self._cache[key]=(value,size);self.cache_bytes+=size
            return value

    def __call__(self,station_m,q=None,*,z_m=0.,with_jacobian=False):
        c=self.course
        # Preserve the original validation, including its q=None behavior.
        q=c.initial if q is None else c.validate_parameters(q)
        s_m=np.asarray(station_m,float).reshape(-1)
        if not np.isfinite(s_m).all():raise ValueError('Finite station metres required')
        inputs=self._inputs(q)
        b,jb,edge,je,lengths,jl,cum,jcum=(inputs[n]for n in('b','jb','edge','je','lengths','jl','cum','jcum'))
        raw=s_m/c.scene_unit_m;wrap=np.floor(raw/cum[-1])
        s=raw % cum[-1]
        i=np.minimum(np.searchsorted(cum,s,side='right')-1,479)
        t=(s-cum[i])/lengths[i]
        ds=-wrap[:,None]*jcum[-1]
        jt=(ds-jcum[i]-t[:,None]*jl[i])/lengths[i,None]
        world=b[i]+t[:,None]*edge[i]
        jac=jb[i]+t[:,None,None]*je[i]+edge[i,:,None]*jt[:,None,:]
        c._z(world,jac,q,'absolute_z_m',z_m)
        station_jac=edge[i]/lengths[i,None]/c.scene_unit_m
        station_jac[:,2]=0.
        report=dict(segment_index=i,segment_t=t,wrapped_turns=wrap.astype(np.int64),
            exactly_on_station_knot=(s==cum[i]),finite_polyline_domain=True,
            derivative_is_selected_station_branch=True)
        return(world,jac,station_jac,report)if with_jacobian else world


def install_exact_course_station_cache(course, *, max_cache_bytes=8*1024**2):
    """Explicit caller opt-in; each reused course instance needs one install.

    The cache is pickle-safe for owned actors and returns fresh station results.
    Cached primitive arrays are read-only. The byte budget is configurable and
    bounded; outer runtime_memory guards the real total process allocation.
    """
    if isinstance(course.station,ExactCourseStationCache):
        if course.station.max_cache_bytes!=max_cache_bytes:
            raise ValueError('Existing station cache has a different explicit budget')
        return course.station
    cache=ExactCourseStationCache(course,max_cache_bytes=max_cache_bytes)
    course.station=cache
    return cache
