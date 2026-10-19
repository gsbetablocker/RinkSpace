"""Append authenticated high-wide source to the unchanged complete programme.

Explicit integer row replicas implement exactly four times the original
weighted Huber2 loss. They are numerical quadrature, never new observations.
Original native sigma and Huber threshold are unchanged.
"""
import numpy as np
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock

class WideProfileUpdate:
    def __init__(self,base,count):self.base=base;self.count=count
    def __call__(self,q):return self.base(np.asarray(q)[:self.count])
    def iter_checkpoint_arrays(self):return self.base.iter_checkpoint_arrays()
    def checkpoint_arrays(self):return self.base.checkpoint_arrays()
    def restore_checkpoint(self,values):return self.base.restore_checkpoint(values)

class CompleteWideSource:
    def __init__(self,base,factor,multiplier=4):
        if multiplier!=4:raise ValueError('Explicit initial fourfold wide source objective required')
        self.base=base;self.factor=factor;self.base_count=base.count
        self.private_count=int(factor.private_camera_parameter_count)
        if self.private_count!=25:raise ValueError('All high camera11 plus Brown14 coordinates must remain active')
        self.camera_indices=np.arange(self.base_count,self.base_count+self.private_count)
        self.count=self.base_count+self.private_count;self.multiplier=multiplier
        self.updater=WideProfileUpdate(base.updater,self.base_count)
        weights=np.asarray(factor.cost_weights,float)
        counts=np.rint(multiplier*weights).astype(int)
        if weights.shape!=factor.scales.shape or not np.array_equal(counts,multiplier*weights) or np.any(counts<1):
            raise ValueError('Original source weights must have exact positive integer quadrature after strengthening')
        self.row_counts=counts
        self.row_indices=np.repeat(np.arange(len(weights)),counts)
        self.wide_quadratic=np.repeat(factor.quadratic,counts)
        self.wide_scales=np.repeat(factor.scales,counts)
        self.quadratic=np.r_[base.quadratic,self.wide_quadratic]
        self.scales=np.r_[base.scales,self.wide_scales]
        self.contract=dict(schema='rinkspace.full-programme-strong-high-wide-free-optics-source.v3',
            base_parameter_count=self.base_count,parameter_count=self.count,
            high_camera_parameter_indices=self.camera_indices.tolist(),
            high_camera_parameter_count=self.private_count,high_observer_is_distinct_from_secondary_target=True,
            wide_source_contract=factor.contract,wide_objective_multiplier=multiplier,
            objective_definition='base original cost + sum(4 * original_cost_weight[row] * Huber2(original_sigma_normalized_r[row]))',
            original_wide_residual_rows=len(weights),numerical_wide_residual_rows=len(self.row_indices),
            numerical_quadrature_counts=counts.tolist(),
            native_step_guard_scales_are_original_sigma_not_cost_weights=True,
            replicas_are_not_independent_observations=True,
            original_native_sigma_and_Huber_threshold_preserved=True,
            old_source_observations_profile_roles_priors_preserved=True,
            programme_actual_exposures=8503,programme_native_presentations=10123,
            wide_source_native_pts=56218880,high_camera_all25_parameters_active=True,
            high_camera_Brown14_tail_fixed=False,old_camera_is_only_initialization_not_measurement=True,
            accuracy_accepted=False)
    def initial(self,base_q,camera=None):
        return np.r_[base_q,self.factor.initial_camera_state25 if camera is None else camera]
    def unpack(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.count,)or not np.isfinite(q).all():raise ValueError('Complete base and active high-camera25 state required')
        return q[:self.base_count],q[self.camera_indices]
    def guard(self,q):
        base,camera=self.unpack(q);self.base.guard(base)
        self.factor.guard(base[:self.base.core_count],camera)
    def blocks(self,q,jacobian=True):
        base,camera=self.unpack(q)
        for block in self.base.blocks(base,jacobian):
            if jacobian:
                m=block.J.tocsr(copy=False)
                matrix=csr_matrix((m.data,m.indices,m.indptr),shape=(m.shape[0],self.count),copy=False)
            else:matrix=None
            yield SourceRowBlock(block.r,matrix,block.quadratic,block.scales,block.row_ledger,getattr(block,'loss_weights',None))
        residual,core_jac,camera_jac=self.factor.evaluate(base[:self.base.core_count],camera,jacobian)
        if jacobian:
            from scipy.sparse import hstack
            matrix=hstack((core_jac,csr_matrix((len(residual),3)),csr_matrix(camera_jac)),format='csr')
            if matrix.shape!=(len(residual),self.count):raise ValueError('Full physical/bias/high-camera25 column map changed')
        else:matrix=None
        if jacobian:matrix=matrix[self.row_indices].tocsr()
        ledger=dict(family='strong_high_wide_source',source_pts=56218880,
            exact_integer_cost_quadrature=True,replicas_are_not_new_measurements=True,
            original_start=len(self.base.scales),original_stop=len(self.scales),
            unique_source_inventory=self.factor.contract)
        yield SourceRowBlock(residual[self.row_indices],matrix,self.wide_quadratic,self.wide_scales,ledger)
