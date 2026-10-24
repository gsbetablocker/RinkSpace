"""Explicit GPU linear policy registration for the separate common Ceres bridge."""
from .native_backend import NativeBackend,LINEAR_SOLVERS
import numpy as np

# Registration is confined to a newly constructed worker process. Existing CPU policies retain their numeric values.
LINEAR_SOLVERS.setdefault('dense_schur',5)

class GpuLinearBackend(NativeBackend):
    def __init__(self,manifest):
        super().__init__(manifest)
        if set(self.metadata.get('available_gpu_policies',{}))!={'cgnr','dense_schur'}:
            raise ValueError('GPU policy backend requires the separately built explicit CUDA manifest')
    def solve(self,layout,initial_parameters,evaluate,*,options=None,**kwargs):
        if options is not None and options.linear_solver=='dense_schur' and kwargs.get('ordering_groups') is not None:
            groups=np.asarray(kwargs['ordering_groups']);constant=kwargs.get('constant_blocks')
            free=np.ones(len(layout.block_ids),dtype=bool) if constant is None else np.asarray(constant)==0
            eliminated=int(groups[free].min()) if np.any(free) else -1
            if any(sum(groups[b]==eliminated and free[b] for b in bs)>1 for bs in layout.factor_blocks):
                raise ValueError('dense Schur first group is not an independent parameter set')
        return super().solve(layout,initial_parameters,evaluate,options=options,**kwargs)
