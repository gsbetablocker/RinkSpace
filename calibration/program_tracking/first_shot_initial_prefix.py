"""Borrow a verified same-q programme J prefix with fresh original source rows.

The cached matrices are already in the complete canonical coordinate chart.
No second time/geometry pullback, old normal, gradient, objective value or
mount-domain callback is reused. Every other state uses the public factory.
"""
from copy import deepcopy
import hashlib,json

import numpy as np
from scipy.sparse import load_npz,vstack

from .first_shot_source_transition import reference,read

SCHEMA='rinkspace.initial-programme-J-prefix.v1'


def capture_source_chunk(task, *, dispatch):
    """Diagnostic dispatch: retain original raw vectors from the existing pool.

    The original fused worker and whitening run unchanged. A temporary worker
    hook copies the raw XY values, then is restored even if evaluation fails.
    This helper neither creates a pool nor installs a new source equation.
    """
    from . import first_shot_row_source as row
    previous = row._raw_and_local
    captured = {}

    def record(state, q, frame, worlds, derivatives=None):
        result = previous(state, q, frame, worlds, derivatives)
        if derivatives is None:
            index = int(frame['index'])-1
            if index in captured:
                raise ValueError('Diagnostic raw-vector capture repeated a frame')
            captured[index] = np.asarray(result[0], dtype=float).copy()
        return result

    row._raw_and_local = record
    try:
        r, j, receipt = dispatch(task)
        if task[2] is None:
            frames = list(task[3])
            if list(captured) != frames:
                raise ValueError('Diagnostic raw-vector capture changed frame order')
            receipt['_diagnostic_raw_xy'] = np.concatenate([captured[i] for i in frames])
        return r, j, receipt
    finally:
        row._raw_and_local = previous

def same(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if (a.dtype.hasobject or b.dtype.hasobject or a.dtype!=b.dtype or
            a.shape!=b.shape or a.tobytes()!=b.tobytes()):
        raise ValueError('Initial programme prefix changed: '+label)

def checked(ref):
    if reference(ref['path'])!=ref:
        raise ValueError('Initial programme prefix reference changed')
    return ref

def digest(q):
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()

class InitialProgrammePrefix:
    def __init__(self,base,*,parent,configuration_reference,entry_reference):
        self.base=base
        self.configuration_reference=deepcopy(checked(configuration_reference))
        self.configuration=read(configuration_reference)
        config=self.configuration
        if (config.get('schema')!=SCHEMA or config.get('producer')!=reference(__file__)
                or config.get('parent_q_sha256')!=parent['q_sha256']
                or config.get('programme_rows')!=214133 or config.get('retired_rows')!=4278):
            raise ValueError('Explicit final-q6 programme-prefix configuration required')
        if len(base.start)!=7700 or len(base.scales)!=214133 or len(base.shared)!=154:
            raise ValueError('Complete programme-only owner required before prefix installation')
        same(base.start,parent['q'],'complete canonical start')
        physical,_=base.effective_q_and_chain(base.start)
        expanded,_=base.expanded_q_and_chain(base.start)
        same(physical,parent['physical_q'],'complete physical start')
        same(expanded,parent['expanded_q'],'complete expanded native start')
        for key,value in parent['aux'].items():same(base.checkpoint_arrays()[key],value,'parent auxiliary '+key)
        for name in ('source_equivalence','source_stencils','current_model_QA','programme_guard_QA'):
            if read(config[name]).get('passed')is not True:
                raise ValueError('Actual current model/source evidence required: '+name)
        model=read(config['current_model_QA'])
        if (model['parent_q_sha256']!=parent['q_sha256'] or
                model['parent_aux_count']!=len(parent['aux']) or
                model['configuration']!=config['grouped_configuration']):
            raise ValueError('Current-model evidence belongs to another complete parent')
        stencil=read(config['source_stencils'])
        if not stencil.get('every_old_admitted_geometry_FD_side_remains_admissible'):
            raise ValueError('The actual old derivative stencils must remain valid')
        self.source_manifest_reference=deepcopy(config['source_manifest'])
        self.source_manifest=read(self.source_manifest_reference)
        saved=self.source_manifest
        if (saved.get('qaug_sha256')!=parent['q_sha256'] or saved.get('J_shape')!=[218411,7700]
                or saved.get('source_J_before_Huber_weights')is not True):
            raise ValueError('Exact complete old canonical same-q source cache required')
        self.blocks_meta=[];cursor=0
        for block in saved['blocks']:
            if cursor>=214133:break
            if (block['block']!=len(self.blocks_meta) or block['row_start']!=cursor or
                    cursor+block['row_count']>214133 or block['J_shape']!=[block['row_count'],7700]):
                raise ValueError('Unchanged complete programme block partition required')
            self.blocks_meta.append(deepcopy(block));cursor+=block['row_count']
        if cursor!=214133 or len(self.blocks_meta)!=6:
            raise ValueError('All six original programme blocks required')
        with np.load(checked(saved['complete_current_state_r_scales_quadratic'])['path'],allow_pickle=False)as z:
            same(z['q_augmented'],parent['q'],'old cache canonical identity')
            same(z['physical_q_augmented'],parent['physical_q'],'old cache physical identity')
            same(z['native_time_expanded_base_canonical_q'],parent['expanded_q'],'old cache native identity')
            self.original_r=z['r'][:214133].copy()
            same(z['scales'][:214133],base.scales,'all current programme scales')
            same(z['quadratic'][:214133],base.quadratic,'all current programme masks')
        with np.load(checked(saved['block_loss_weights'])['path'],allow_pickle=False)as z:
            self.loss_arrays={key:z[key].copy()for key in z.files}
        self.parent_q=np.asarray(parent['q']).copy()
        self._matrices=None
        self.verified_shadow=None
        self.same_q_calls=0
        self.fresh_other_q_calls=0
        self.source_producers=deepcopy(base.source_producers)
        self.recovery_references=deepcopy(base.recovery_references)
        self._references=deepcopy(getattr(base,'_references',[]))
        self.receipt=deepcopy(base.receipt)
        self.prefix_receipt=dict(schema=SCHEMA,producer=reference(__file__),entry=deepcopy(checked(entry_reference)),
            configuration=deepcopy(configuration_reference),source_manifest=deepcopy(self.source_manifest_reference),
            parent_q_sha256=parent['q_sha256'],programme_rows=214133,retired_rows_excluded=4278,
            same_q_original_source_derivative_approximation_retained=True,
            original_geometry_stencils_current_model_valid=True,canonical_J_pulled_back_again=False,
            fresh_original_r_scales_quadratic_loss_and_ledger_verified_each_J_call=True,
            old_normal_gradient_weights_cost_and_mount_callback_reused=False,
            current_guard_recover_QP_and_fixed_columns_unmodified=True,
            every_other_q_uses_public_fresh_source_factory=True)
        for name in ('source_producers','recovery_references','_references'):
            refs=getattr(self,name)
            for ref in (reference(__file__),entry_reference,configuration_reference,self.source_manifest_reference):
                if ref not in refs:refs.append(deepcopy(ref))
        self._producer=reference(__file__)
        self._initial_complete_J_reuse=False

    def __getattr__(self,name):return getattr(self.base,name)

    def same_q(self,q):
        q=np.asarray(q)
        return q.dtype==self.parent_q.dtype and q.shape==self.parent_q.shape and q.tobytes()==self.parent_q.tobytes()

    def _check(self):
        checked(self._producer);checked(self.configuration_reference)

    def _load_matrices(self):
        if self._matrices is None:
            values=[]
            for meta in self.blocks_meta:
                matrix=load_npz(checked(meta['J'])['path']).tocsr()
                if (list(matrix.shape)!=meta['J_shape'] or matrix.nnz!=meta['J_nonzeros']
                        or not np.isfinite(matrix.data).all() or matrix[:,7687:].nnz):
                    raise ValueError('Saved programme canonical CSR changed')
                values.append(matrix)
            self._matrices=values

    def source_blocks(self,q,jacobian):
        self._check()
        if not jacobian or not self.same_q(q):
            if jacobian:self.fresh_other_q_calls+=1
            yield from self.base.source_blocks(q,bool(jacobian))
            return
        from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
        self.base.guard(q)
        live=list(self.base.source_blocks(q,False))
        if len(live)!=len(self.blocks_meta):raise ValueError('Live programme source block count changed')
        cursor=0;proof=[]
        for index,(block,meta)in enumerate(zip(live,self.blocks_meta,strict=True)):
            n=len(block.r)
            if n!=meta['row_count']:raise ValueError('Live programme block row count changed')
            same(block.r,self.original_r[cursor:cursor+n],'fresh residual block '+str(index))
            same(block.scales,self.base.scales[cursor:cursor+n],'fresh scales block '+str(index))
            same(block.quadratic,self.base.quadratic[cursor:cursor+n],'fresh quadratic block '+str(index))
            if json.dumps(block.row_ledger,sort_keys=True)!=json.dumps(meta['row_ledger'],sort_keys=True):
                raise ValueError('Actual original UV/PTS/full-W row ledger changed')
            actual=source_loss_checkpoint_arrays(block.loss_weights,n)
            expected={key.split('__block_')[0]:value for key,value in self.loss_arrays.items()
                if key.endswith('__block_'+str(index))}
            if set(actual)!=set(expected):raise ValueError('Current source loss inventory changed')
            for key in actual:same(actual[key],expected[key],'fresh loss block '+str(index)+'/'+key)
            proof.append(dict(block=index,rows=n,source_r_scales_quadratic_loss_ledger_exact=True));cursor+=n
        if cursor!=214133:raise ValueError('Live programme source prefix incomplete')
        self._load_matrices()
        self.same_q_calls+=1
        self.verified_shadow=dict(passed=True,q_sha256=digest(q),rows=cursor,blocks=proof,
            fresh_source_values=True,old_J_only=True,current_original_guard_called=True)
        for block,matrix in zip(live,self._matrices,strict=True):
            yield type(block)(block.r,matrix,block.quadratic,block.scales,block.row_ledger,block.loss_weights)

    def source_J_block_reference(self,q,index,block):
        self._check()
        if not self.same_q(q)or self.verified_shadow is None:return None
        if not 0<=int(index)<len(self.blocks_meta):raise ValueError('Unexpected programme J block')
        expected=self._matrices[int(index)]
        if block.J is not expected:return None
        return deepcopy(checked(self.blocks_meta[int(index)]['J']))

    def checkpoint_arrays(self):return self.base.checkpoint_arrays()

    def evaluate(self,q,jacobian=False):
        blocks=list(self.source_blocks(q,bool(jacobian)))
        r=np.concatenate([block.r for block in blocks])
        return (r,vstack([block.J for block in blocks],format='csr'))if jacobian else r

    def checkpoint_identity(self):
        result=deepcopy(self.base.checkpoint_identity())
        result.update(initial_programme_prefix=deepcopy(self.prefix_receipt),source_producers=deepcopy(self.source_producers))
        return result

    def complete_state_payload(self,q):
        result=self.base.complete_state_payload(q)
        result['initial_programme_prefix_identity_json']=np.asarray(json.dumps(self.prefix_receipt,sort_keys=True))
        return result

    def solver_kwargs(self):
        result=self.base.solver_kwargs()
        if result.get('update_profiles')is not None:raise ValueError('Stateful source profiles cannot borrow this prefix')
        result.update(source_block_factory=self.source_blocks,checkpoint_arrays=self.checkpoint_arrays(),
            checkpoint_identity=self.checkpoint_identity())
        return result

def install(owner,*,parent,configuration_reference,entry_reference):
    return InitialProgrammePrefix(owner,parent=parent,configuration_reference=configuration_reference,entry_reference=entry_reference)
