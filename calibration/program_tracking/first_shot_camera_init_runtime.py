"""Explicit execution revision of the frozen restricted camera initializer.

Numerical source r/CSR and producer/domain identities are strict contracts.
Worker addresses, scheduling, thread receipts and runtime statistics are
recorded separately; process metadata is not an equality of camera models.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from .first_shot_source_transition import reference
from . import first_shot_camera_initialization as core


def backend_numerical_contract(backend):
    """Select declared numerical producers/domains, excluding execution receipts."""
    if backend is None:
        return None
    library_fields = ('module', 'source_path', 'source_sha256', 'library_path',
                      'library_sha256', 'source_and_library_match_manifest')
    libraries = {name:{key:deepcopy(item[key]) for key in library_fields if key in item}
        for name,item in backend.get('actual_native_libraries',{}).items()}
    hook = backend.get('projection_hook')
    hook = None if hook is None else {key:deepcopy(value) for key,value in hook.items()
        if key != 'stats_at_enable'}
    return dict(schema=backend.get('schema'),
        required_environment=deepcopy(backend.get('required_environment')),
        actual_native_libraries=libraries, projection_hook=hook)


def compare_backends(original, cached):
    if backend_numerical_contract(original) != backend_numerical_contract(cached):
        raise ValueError('Actual cache changed a backend numerical producer/domain contract')


class CameraInitializationStage(core.CameraInitializationStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        execution = reference(__file__)
        self._references.append(execution)
        if execution not in self.source_producers:
            self.source_producers.append(deepcopy(execution))
        self.receipt['inherited_camera_initializer_core'] = deepcopy(self.receipt['producer'])
        self.receipt['camera_initializer_execution_adapter'] = execution
        self.receipt['backend_execution_receipts_not_mathematical_equality'] = True

    def certify_execution(self, output):
        if self.programme.pool is None:
            raise ValueError('Cache validation requires the existing source pool, never a replacement')
        output = Path(output)
        if output.exists():
            raise ValueError('Actual cache QA must not overwrite previous source evidence')
        from native_transverse_bundle import source_chunk
        physical, _ = self.effective_q_and_chain(self.start)
        prefix = physical[:self.programme_count]
        worlds, _ = self.programme.world(prefix, False)
        zero = {key:np.zeros(value.shape+(171,)) for key,value in worlds.items()}
        indices = [200,384,588,841,1238,1247]
        if [self.programme.bundle.frames[i]['index'] for i in indices] != [201,385,589,842,1239,1248]:
            raise ValueError('Actual source six-frame native indices changed')
        task = (prefix,worlds,zero,indices,core._sha(prefix))
        original = next(self.programme._pool().map(source_chunk,[task],chunksize=1))
        cached = next(self.programme._pool().map(core.camera_source_chunk,[(task,self.execution)],chunksize=1))
        def identical(a,b,label):
            a,b = np.asarray(a),np.asarray(b)
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError('Actual installation cache changed '+label)
        identical(original[0],cached[0],'source residual')
        if original[1].shape != cached[1].shape:
            raise ValueError('Actual installation cache changed complete camera J shape')
        for name in ('data','indices','indptr'):
            identical(getattr(original[1],name),getattr(cached[1],name),'CSR '+name)
        contract = ('augmented_q_sha256','old_q_sha256','PP_sha256','axis_a_working_m',
            'quadratic_b_working_per_logf_squared','frames_0based','world_derivatives_received',
            'full_parameter_count','source_sampler')
        for key in contract:
            if ((key in original[2]) != (key in cached[2])
                    or original[2].get(key) != cached[2].get(key)):
                raise ValueError('Actual cache source worker contract changed: '+key)
        compare_backends(original[2].get('backend'),cached[2].get('backend'))
        report = dict(schema='rinkspace.camera-initializer-actual-six-source-cache-QA.v2',passed=True,
            producer=reference(__file__),inherited_core=reference(core.__file__),execution=deepcopy(self.execution),
            q_sha256=core._sha(prefix),frames_0based=indices,source_rows=len(original[0]),
            J_shape=list(original[1].shape),J_nonzeros=original[1].nnz,
            r_dtype_shape_and_bytes_exact=True,CSR_data_indices_indptr_shape_and_bytes_exact=True,
            original_source_worker_numerical_contract_exact=True,
            backend_numerical_producer_and_domain_contract=backend_numerical_contract(original[2].get('backend')),
            original_worker_receipt=deepcopy(original[2]),cached_worker_receipt=deepcopy(cached[2]),
            execution_receipts_preserved_but_not_equated=[
                'pid','backend.stage','backend.restored_environment_before','backend.native_threads',
                'backend.pattern_workers','backend.pattern_worker_count_owned_by_caller',
                'backend.acceleration_loaded_module_aliases[*].module_object_id',
                'backend.acceleration_loaded_module_aliases (other execution history)',
                'backend.actual_native_libraries.*.metadata','backend.projection_hook.stats_at_enable',
                'backend.dispatch','same_value_spawn_execution'],
            original_persistent_pool_reused=True,world_values_only_local_zero171_not_in_real_Dcache=True,
            parent_cache_not_installed_before_comparison=True,full1248_shadow_or_new_owner_or_pool_constructed=False,
            source_contract={key:deepcopy(original[2][key]) for key in contract if key in original[2]})
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        return reference(output)
