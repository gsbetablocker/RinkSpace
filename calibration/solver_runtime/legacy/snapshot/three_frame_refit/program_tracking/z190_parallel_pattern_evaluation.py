"""Independent process evaluation of all original print-pattern frames."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import atexit
import multiprocessing
import os
import pickle
import sys

_factors=None


def _initialize(payload):
    global _factors
    task_root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(task_root))
    from isolated_solver_bootstrap import install
    install()
    os.environ['RINKSPACE_MOTION_THREADS']='1'
    os.environ['RINKSPACE_SCATTER_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    _factors=pickle.loads(payload)
    from continuous_central_brown4_optin import enable_projection_backend
    enable_projection_backend()


def _evaluate(job):
    ordinal,q9,camera,frame,jacobian=job
    return _factors[ordinal].evaluate(q9,camera,frame,with_jacobian=jacobian)


class PatternFramePool:
    def __init__(self,owner,workers):
        factors=tuple(owner.frame_factor(frame)for frame in owner.fit+owner.held)
        self.pool=ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,
            initargs=(pickle.dumps(factors,protocol=5),))
        atexit.register(self.close)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.pool=None

    def submit(self,owner,q,jacobian,*,held=False):
        q9=q[owner.start:owner.start+9].copy()
        owner.geometry.guard(q9)
        frames=owner.held if held else owner.fit
        cameras=[owner.camera(q,frame,held=held)for frame in frames]
        offset=len(owner.fit)if held else 0
        futures=[self.pool.submit(_evaluate,(offset+i,q9,cam[0],frame,jacobian))
                 for i,(frame,cam)in enumerate(zip(frames,cameras))]
        return [(frame,cam,future)for frame,cam,future in zip(frames,cameras,futures)]


_pools={}
_pending={}


def submit_pattern_frames(owner,q,jacobian,*,held=False):
    workers=int(os.environ.get('RINKSPACE_PATTERN_WORKERS','0'))
    if workers<1:return None
    identity=id(owner)
    if identity not in _pools:_pools[identity]=PatternFramePool(owner,workers)
    return _pools[identity].submit(owner,q,jacobian,held=held)


def prefetch_pattern_frames(owner,q,jacobian):
    if owner is None or int(os.environ.get('RINKSPACE_PATTERN_WORKERS','0'))<1:return
    _pending[id(owner)]=(q.copy(),jacobian,submit_pattern_frames(owner,q,jacobian))


def take_pattern_frames(owner,q,jacobian):
    previous=_pending.pop(id(owner),None)
    if previous is not None:
        import numpy as np
        if previous[1]==jacobian and np.array_equal(previous[0],q):return previous[2]
    return submit_pattern_frames(owner,q,jacobian)


def close_pattern_pools():
    for pool in _pools.values():pool.close()
    _pools.clear()
    _pending.clear()
