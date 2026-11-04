"""Optional CPU native22 execution of the unchanged foot/collar objective.

The caller owns the reference and fixes its 17 shared skate parameters for the
executor's lifetime. Workers read prepared geometry and full-surface tensors;
each local perturbation owns its FK, skinning, and section work buffers.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
from types import SimpleNamespace

import numpy as np

from .skate_physics import fk_state


class NativeCompatibilityExecutor:
    def __init__(self,reference,mid_ref,*,workers=4):
        if isinstance(workers,bool) or int(workers)!=workers or workers<1:
            raise ValueError('workers must be a positive integer')
        self.workers=int(workers);self._pool=None
        self._source_binding=reference.binding
        self._shared_x=np.array(reference.binding.x,dtype=float,copy=True)
        if self._shared_x.shape!=(17,):raise ValueError('Expected 17 fixed shared skate parameters')
        self._shared_x.setflags(write=False)
        # Private cache owners prevent preparation from changing the live
        # binding/surface. Completed immutable cache entries may be shared.
        self.binding=copy.copy(reference.binding)
        # SciPy Rotation requires a writable input buffer even when it only
        # reads it. This private parameter copy is never changed by workers.
        self.binding.x=self._shared_x.copy()
        self.binding.surface=copy.copy(reference.binding.surface)
        self.reference=SimpleNamespace(body=reference.body,binding=self.binding)
        self.mid_ref={}
        for key in ('P','Q','local_R','hand_local_R'):
            if key in mid_ref:
                value=np.array(mid_ref[key],dtype=float,copy=True)
                value.setflags(write=False);self.mid_ref[key]=value
        self.n=len(self.mid_ref['P'])
        if self.n==0:raise ValueError('Empty compatibility sample set')
        self.binding.packet()
        if self.binding.collar_constraints_enabled:
            from .skate_collar_factors import prepare
            prepare(self.binding)
        # One union skin sample preserves every foot and shank vertex, all
        # pose blend shapes, and the actual hand rotations. Passing it to
        # compatibility_rows also avoids reskinning the collar subset.
        first={key:value[:1] for key,value in self.mid_ref.items()}
        state=fk_state(self.reference,first,np.zeros((1,69)))
        self._samples(state)

    def _check_shared(self):
        if not np.array_equal(self._source_binding.x,self._shared_x):
            raise ValueError('Shared skate parameters changed; create a new compatibility executor')

    def _delta(self,delta):
        value=np.asarray(delta,dtype=float)
        if value.shape!=(self.n,69) or not np.isfinite(value).all():
            raise ValueError('Expected finite (sample_count,69) body increments')
        return value

    def _samples(self,state):
        kwargs={} if 'hand_local_R' not in state else dict(hand_local_R=state['hand_local_R'])
        return self.binding.surface.sample_vertices(state['native'],state['frames'],
            self.binding.sample_ids,double_precision=True,**kwargs)

    def _residual(self,delta):
        state=fk_state(self.reference,self.mid_ref,delta)
        samples=self._samples(state)
        kwargs={} if 'hand_local_R' not in state else dict(hand_local_R=state['hand_local_R'])
        return self.binding.compatibility_rows(state['native'],state['frames'],
            samples,**kwargs)/np.sqrt(2)

    def residual(self,delta):
        self._check_shared()
        result=self._residual(self._delta(delta))
        self._check_shared()
        return result

    def local_derivatives(self,delta,columns,*,step=1e-5):
        """Ordered (column, N x compatibility_width) central differences.

        All body-local columns 6..68, including both toe joints, are supported.
        Root P/Q columns 0..5 return exact zero, matching BootWindowProblem.
        This method is called by one owner; only its local columns run in parallel.
        """
        self._check_shared();delta=self._delta(delta).copy()
        columns=list(columns)
        if any(isinstance(c,(bool,np.bool_)) or int(c)!=c or not 0<=c<69 for c in columns):
            raise ValueError('Local column outside 0..68')
        columns=list(map(int,columns))
        if not np.isfinite(step) or step<=0:raise ValueError('step must be finite and positive')
        delta.setflags(write=False)

        def difference(column):
            if column<6:
                return column,np.zeros((self.n,self.binding.compatibility_width))
            plus=delta.copy();minus=delta.copy()
            plus[:,column]+=step;minus[:,column]-=step
            return column,(self._residual(plus)-self._residual(minus))/(2*step)

        try:
            from threadpoolctl import threadpool_limits
            limits=threadpool_limits(limits=1,user_api='blas')
        except ImportError:
            if self.workers>1:raise RuntimeError('Parallel compatibility requires threadpoolctl')
            limits=nullcontext()
        with limits:
            if self.workers==1 or len(columns)<=1:
                result=[difference(c) for c in columns]
            else:
                if self._pool is None:self._pool=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='rink-compat')
                result=list(self._pool.map(difference,columns))
        self._check_shared()
        return result

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True,cancel_futures=True);self._pool=None

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()
