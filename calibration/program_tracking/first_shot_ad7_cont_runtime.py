"""Data-only future grouped-parent successor runtime and fixed-column restore.

No fit entry is provided. Unsealed synthetic fixtures require an explicit
protocol-only opt-in; an actual installation first needs a completed accepted
parent binding. This module never edits the frozen chart or its registry.
"""
from copy import deepcopy
from contextlib import contextmanager
from functools import partial
import json
from types import MethodType

import numpy as np
from scipy.sparse import csr_matrix,vstack,issparse

from . import first_shot_ad7_continuation as proposal_api
from calibration.program_tracking import first_shot_ad7_shape_chart as old
from calibration.program_tracking.first_shot_source_transition import reference,read

SCHEMA='rinkspace.grouped-parent-seed-successor-runtime.v2'
_GLOBAL=None
_ORIGINAL={}
_CLASS=None
METHODS=('templates_at','scene','guard_geometry','print_layout_world_jacobians')


def _read(ref):
    return proposal_api.authenticated(ref)


def _complete_parent(config):
    """Only small checkpoint arrays are read; no source, r/J or model state."""
    from calibration.program_tracking.first_shot_time_successor import read_accepted_parent
    proof=_read(config['completed_parent_binding'])
    refs=config['complete_parent_references']
    if (proof.get('passed') is not True or proof.get('complete_parent_references')!=refs
            or proof.get('all_observed_owner_launcher_and_workers_absent') is not True
            or proof.get('original_owner_normal_pool_shutdown') is not True
            or proof.get('no_process_terminated') is not True):
        raise ValueError('Explicit completed accepted parent and normal shutdown binding required')
    parent=read_accepted_parent(refs)
    proposal_api.verify_configuration(_read(config['continuation']),parent)
    return parent


def build_configuration(parent,*,continuation_reference,entry_reference,
                        completed_parent_binding=None):
    value=_read(continuation_reference)
    proposal_api.verify_configuration(value,parent)
    if float(parent['damping'])<=0:
        raise ValueError('Native-time continuation requires strictly positive inherited damping')
    if reference(entry_reference['path'])!=entry_reference:
        raise ValueError('Explicit future execution entry reference required')
    result=dict(schema=SCHEMA,producer=reference(__file__),
        continuation=deepcopy(continuation_reference),proposal_producer=reference(proposal_api.__file__),
        legacy_math_producer=reference(old.__file__),entry=deepcopy(entry_reference),
        material=deepcopy(value['material']),restored_configuration=deepcopy(value['restored_configuration']),
        installation_scope='unsealed_data_protocol',fit_ready=False,
        old_H8_initializer_called=False,formal_sources_or_W_added=False,
        initial_source_or_world_J_reuse=False,complete_parent_references=None,
        completed_parent_binding=deepcopy(completed_parent_binding),
        current_whole_source_FD_and_spawn_QA_required=True)
    if completed_parent_binding is not None:
        if value.get('allow_unsealed_protocol_ancestry'):
            raise ValueError('Synthetic ancestry opt-in cannot be sealed as a formal completed-parent configuration')
        proof=_read(completed_parent_binding)
        result['complete_parent_references']=deepcopy(proof['complete_parent_references'])
        result['installation_scope']='completed_parent_pending_actual_startup_QA'
        actual=_complete_parent(result)
        if proposal_api.parent_contract(actual)!=proposal_api.parent_contract(parent):
            raise ValueError('Completed binding differs from supplied complete grouped parent')
    return result


class RuntimeChart(old.Ad7GroupedShapeChart):
    """Exact old grouped math, explicit new identity, fixed text seed override."""
    def __init__(self,config):
        value=_read(config['continuation']);original=_read(value['restored_configuration'])
        if (original.get('producer')!=reference(old.__file__)
                or value.get('producer')!=reference(proposal_api.__file__)
                or config['proposal_producer']!=value['producer']
                or config['legacy_math_producer']!=original['producer']
                or value['material']!=original['material']):
            raise ValueError('Explicit unchanged grouped kernel and material required')
        seed=np.asarray(value['evaluation_seed8'],float)
        expected=np.asarray(original['physical_seed8'],float)
        prior=value.get('restored_current_runtime')
        if prior is not None:
            previous=_read(prior);previous_value=_read(previous['continuation'])
            if (previous.get('schema')not in proposal_api.RUNTIME_SCHEMAS
                    or previous.get('restored_configuration')!=value['restored_configuration']
                    or previous_value.get('restored_configuration')!=value['restored_configuration']):
                raise ValueError('Inherited current runtime does not share the original grouped basis')
            expected=np.asarray(previous_value['evaluation_seed8'],float)
        if not proposal_api.same(np.asarray(value['restored_current_seed8'],float),expected):
            raise ValueError('Restored current seed differs from its authenticated previous runtime')
        if value['initializer'] is not None:
            replacement,_=proposal_api._candidate(value['initializer'],
                dict(configuration=original,reference=value['restored_configuration']))
            if prior is not None and not proposal_api.same(replacement,expected):
                raise ValueError('Root-only initializer cannot overwrite a different current-conditioned seed')
            expected=replacement
        if not proposal_api.same(seed,expected):
            raise ValueError('New seed is not the explicitly authenticated text initializer')
        if not proposal_api.same(np.asarray(value['original_anchor8'],float),np.asarray(original['parent_slot_anchor'],float)):
            raise ValueError('Inherited grouped anchor must remain byte exact')
        super().__init__(original)
        self.seed=seed.copy()
        self.configuration=deepcopy(value)


def _load(ref,allow_unsealed_protocol):
    config=_read(ref)
    if (config.get('schema')!=SCHEMA or config.get('producer')!=reference(__file__)
            or config.get('fit_ready') is not False
            or config.get('proposal_producer')!=reference(proposal_api.__file__)
            or config.get('legacy_math_producer')!=reference(old.__file__)
            or reference(config['entry']['path'])!=config['entry']):
        raise ValueError('Explicit unchanged unsealed successor runtime required')
    scope=config.get('installation_scope')
    if scope=='unsealed_data_protocol':
        if not allow_unsealed_protocol:
            raise ValueError('Synthetic/unsealed protocol cannot install a formal successor')
    elif scope=='completed_parent_pending_actual_startup_QA':
        _complete_parent(config)
    else:raise ValueError('Unknown grouped continuation installation scope')
    return config,RuntimeChart(config)


def _chart(state):
    ref=getattr(state,'ad7_continuation_reference',None)
    if ref is None:return None
    if _GLOBAL is None or ref!=_GLOBAL['reference']:
        raise ValueError('Matching grouped continuation runtime must be installed')
    return _GLOBAL['chart']


def templates_at(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['templates_at'](self,geometry)
    q=self._geometry(geometry)
    value=deepcopy(_ORIGINAL['templates_at'](self,old._neutral(self,q)))
    metrics,D=old.canvas_metrics(self,q)
    paths,_=chart.paths(q[self.print_layout_world_slices[7]],metrics,D)
    for key,uv in paths.items():value['advertisements']['7']['contours'][key]['material_uv']=uv.tolist()
    value['advertisements']['7']['grouped_shape_chart']=dict(configuration=deepcopy(_GLOBAL['reference']),
        producer=reference(__file__),parameters=chart.parameters(q[self.print_layout_world_slices[7]]).tolist(),
        original_grouped_configuration=deepcopy(_GLOBAL['config']['restored_configuration']),
        no_shared_H8_applied_to_this_advertisement=True)
    return value


def scene(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['scene'](self,geometry)
    q=self._geometry(geometry)
    value=deepcopy(_ORIGINAL['scene'](self,old._neutral(self,q)))
    value['ad7_grouped_shape_chart']=dict(schema=SCHEMA,configuration=deepcopy(_GLOBAL['reference']),
        producer=reference(__file__),original_grouped_configuration=deepcopy(_GLOBAL['config']['restored_configuration']),
        continuation=deepcopy(_GLOBAL['config']['continuation']),parameter_names=list(old.PARAMETERS),
        physical_parameters=chart.parameters(q[self.print_layout_world_slices[7]]).tolist(),
        source_and_review_use_same_templates_at=True,other_advertisements_unchanged=True,
        text_initializer_is_candidate_not_formal_source=True)
    return value


def guard_geometry(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['guard_geometry'](self,geometry)
    q=self._geometry(geometry)
    result=_ORIGINAL['guard_geometry'](self,old._neutral(self,q))
    metrics,D=old.canvas_metrics(self,q)
    proof=chart.guard(q[self.print_layout_world_slices[7]],metrics,D)
    domains=deepcopy(result['shared_print_layout_domain'])
    domains['7']=dict(proof,configuration=deepcopy(_GLOBAL['reference']),old_H8_not_evaluated_as_Ad7_ink=True)
    return dict(result,shared_print_layout_domain=domains,ad7_grouped_shape_domain=proof)


def print_layout_world_jacobians(self,geometry,descriptors):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['print_layout_world_jacobians'](self,geometry,descriptors)
    q=self._geometry(geometry);paths=self.world_paths(q,descriptors)
    result=[np.zeros(path.shape+(len(q)-self.geometry_count,))for path in paths]
    other=[i for i,target in enumerate(descriptors)if not old._ad7(target)]
    if other:
        previous=_ORIGINAL['print_layout_world_jacobians'](self,old._neutral(self,q),[descriptors[i]for i in other])
        for i,item in zip(other,previous):result[i]=item
    metrics,Dmetric=old.canvas_metrics(self,q)
    values,D=chart.paths(q[self.print_layout_world_slices[7]],metrics,Dmetric)
    h=float(self.provider.fd_step)
    for i,target in enumerate(descriptors):
        if not old._ad7(target):continue
        parts=[]
        for key in old._keys(target):
            uv=values[key];lift=np.empty((len(uv),3,2))
            for axis in range(2):
                lo=uv.copy();hi=uv.copy();lo[:,axis]-=h;hi[:,axis]+=h
                a=self.provider.current.material(7,lo[:,0],1-lo[:,1],q[:100])[0]
                b=self.provider.current.material(7,hi[:,0],1-hi[:,1],q[:100])[0]
                lift[:,:,axis]=(b-a)/(2*h)
            item=np.einsum('nci,nip->ncp',lift,D[key][...,-8:])
            if chart.closed[key] and not np.array_equal(uv[0],uv[-1]):item=np.concatenate((item,item[:1]))
            parts.append(item)
        item=np.concatenate(parts)
        if item.shape!=paths[i].shape+(8,):raise ValueError('Continuation path/J inventory mismatch')
        result[i][...,self.print_layout_tail_slices[7]]=item
    return tuple(result)


def install_execution(configuration_reference,*,allow_unsealed_protocol=False,_protocol_state_class=None):
    """Use a fresh process after core/spawn; never layer over old group install."""
    global _GLOBAL,_CLASS
    config,chart=_load(configuration_reference,allow_unsealed_protocol)
    if _GLOBAL is not None:
        if _GLOBAL['reference']!=configuration_reference:
            raise ValueError('Another grouped continuation is already installed')
        return deepcopy(_GLOBAL['receipt'])
    if old._GLOBAL is not None:
        raise ValueError('Fresh process required; cannot layer onto installed legacy grouped runtime')
    from calibration.program_tracking import first_shot_disjoint_print as core
    if core._CONFIG is None or core._CONFIG['manifest']['material_override']!=config['material']:
        raise ValueError('Same disjoint material execution must precede grouped continuation')
    if _protocol_state_class is not None and not allow_unsealed_protocol:
        raise ValueError('Custom state class is only for a declared lightweight protocol fixture')
    if _protocol_state_class is None:
        from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
        _protocol_state_class=FirstShotPhysicalState
    _CLASS=_protocol_state_class
    originals={name:getattr(_CLASS,name) for name in METHODS}
    if any(fn in globals().values() for fn in originals.values()):
        raise ValueError('Runtime physical methods must be installed exactly once')
    receipt=dict(schema=SCHEMA,producer=reference(__file__),configuration=deepcopy(configuration_reference),
        continuation=deepcopy(config['continuation']),original_grouped_configuration=deepcopy(config['restored_configuration']),
        material=deepcopy(config['material']),entry=deepcopy(config['entry']),
        parent_canonical_and_all_aux_bytes_retained=True,only_fixed_text5_initializer_changed=True,
        source_UV_W_ID_PTS_and_rowtime_unchanged=True,old_source_or_world_J_reused=False,
        physical_guard_and_nonlinear15_margins_retained=True,installation_scope=config['installation_scope'],
        fit_ready=False,actual_startup_source_FD_spawn_and_whole_shot_validation_pending=True)
    _ORIGINAL.update(originals)
    _GLOBAL=dict(reference=deepcopy(configuration_reference),config=config,chart=chart,receipt=receipt,
        allow_unsealed_protocol=allow_unsealed_protocol)
    for name in METHODS:setattr(_CLASS,name,globals()[name])
    return deepcopy(receipt)


def install_state(state):
    if _GLOBAL is None:raise ValueError('Install continuation execution first')
    from calibration.program_tracking.first_shot_disjoint_print import _physical_states
    states=list(_physical_states(state));changed=[]
    if not states and _GLOBAL['allow_unsealed_protocol'] and isinstance(state,_CLASS):states=[state]
    if not states:raise ValueError('No actual physical states for continuation')
    for actual in states:
        previous=getattr(actual,'ad7_continuation_reference',None)
        if previous==_GLOBAL['reference']:continue
        if previous is not None or getattr(actual,'ad7_grouped_shape_reference',None) is not None:
            raise ValueError('Fresh physical state required; cannot silently replace another installed chart')
        for key in _GLOBAL['chart'].keys:
            if not proposal_api.same(np.asarray(actual.templates['advertisements']['7']['contours'][key]['canonical_print_uv']),
                                     np.asarray(_GLOBAL['chart'].material['contours'][key]['canonical_print_uv'])):
                raise ValueError('Physical state uses a different canonical material bank')
        actual.ad7_continuation_reference=deepcopy(_GLOBAL['reference'])
        actual._scene_cache.clear();actual._world_cache.clear();changed.append(type(actual).__name__)
    return changed


def source_worker_init(*args,configuration_reference,original_initializer,allow_unsealed_protocol=False):
    original_initializer(*args)
    install_execution(configuration_reference,allow_unsealed_protocol=allow_unsealed_protocol)


def geometry_worker_init(*args,configuration_reference,original_initializer,allow_unsealed_protocol=False):
    original_initializer(*args)
    install_execution(configuration_reference,allow_unsealed_protocol=allow_unsealed_protocol)
    from calibration.program_tracking import first_shot_fit as fit
    install_state(fit._geometry_worker[0])


def _immutable_archive(existing,key,value):
    if key in existing and not proposal_api.same(existing[key],value):
        raise ValueError('Historical continuation auxiliary conflict: '+key)


def _refs(owner,refs):
    for ref in refs:
        if reference(ref['path'])!=ref:raise ValueError('Continuation producer changed')
        for obj,name in ((owner,'source_producers'),(owner,'_references'),(owner.programme,'source_producers')):
            values=getattr(obj,name)
            if ref not in values:values.append(deepcopy(ref))


def install_successor(owner,*,parent,configuration_reference,entry_reference,
                      allow_unsealed_protocol=False,install_worker_wrappers=True):
    """After original source restore/refresh+printQP, before native-time stage."""
    programme=owner.programme
    if (owner.parameter_count!=7700 or owner.programme_count!=7687
            or len(owner.shared)!=172 or len(owner.fixed)!=3784
            or getattr(owner,'native_time_receipt',None) is not None):
        raise ValueError('Fresh original uncompressed172-shared base required')
    if any(v is not None for v in (programme.pool,programme.bundle._geometry_pool,programme.bundle._source_pool)):
        raise ValueError('All pools must remain unopened')
    if getattr(programme,'initial_J_manifest',None) is not None or getattr(programme,'_initial_complete_J_reuse',False):
        raise ValueError('Old source/J replay is invalid for this explicit successor')
    if not install_worker_wrappers and not allow_unsealed_protocol:
        raise ValueError('Only lightweight protocol fixtures may skip worker wrappers')
    config,chart=_load(configuration_reference,allow_unsealed_protocol)
    if entry_reference!=config['entry']:raise ValueError('Actual new entry differs from runtime configuration')
    value=_read(config['continuation']);proposal_api.verify_configuration(value,parent)
    if float(parent['damping'])<=0:
        raise ValueError('Native-time continuation requires strictly positive inherited damping')
    if not hasattr(owner,'_print_H8_domain'):raise ValueError('Original print QP must be installed first')
    # This is the original uncompressed native base. The accepted canonical
    # B8/B4 coordinates belong to the subsequent time stage, while this base
    # evaluates their fully expanded native values. Keep its historical start.
    base_start=np.asarray(owner.start).copy()
    physical_parent,_=owner.effective_q_and_chain(parent['expanded_q'])
    if not proposal_api.same(physical_parent,parent['physical_q']):
        raise ValueError('Uncompressed base does not restore the complete parent physical state')
    before=proposal_api.parent_contract(parent)
    previous_arrays={k:np.asarray(v).copy()for k,v in owner.checkpoint_arrays().items()}
    archive='ad7_cont_'+configuration_reference['sha256'][:16]+'_json'
    if archive in previous_arrays:
        raise ValueError('Fresh successor cannot overwrite an existing continuation archive')
    domain=owner._print_H8_domain
    rows=[j for item in domain.ledger if item['advertisement_id']==7 for j in range(*item['row_slice'])]
    if not rows:raise ValueError('Original print QP has no explicit Ad7 rows')
    from calibration.program_tracking import first_shot_disjoint_print as core
    state=old.actual_physical_state(programme.state)
    physical=physical_parent
    geo=physical[state.world_parameter_indices];metrics,Dmetric=old.canvas_metrics(state,geo)
    chart.guard(geo[state.print_layout_world_slices[7]],metrics,Dmetric)
    support=old.audit_source_support(programme.bundle.frames)
    if any(bool(support['groups'][name]['assigned_target_rows'])!=(name=='logo') for name in old.GROUPS):
        raise ValueError('Only authenticated logo support may be fitted; text support must remain absent')
    receipt=install_execution(configuration_reference,allow_unsealed_protocol=allow_unsealed_protocol)
    changed=install_state(programme.state)+install_state(programme.bundle.state)
    receipt.update(changed_physical_states=changed,assigned_source_support=support,
        old_Ad7_H8_QP_rows_replaced=rows,parent_aux_count=len(parent['aux']))
    old_constraints=owner.linear_constraints;old_arrays=owner.checkpoint_arrays;old_identity=owner.checkpoint_identity
    metadata=np.asarray(json.dumps(receipt,sort_keys=True))
    _immutable_archive(previous_arrays,archive,metadata)
    def constraints(this,q,selected):
        original=old_constraints(q,selected);first,stop=original['report']['appended_print_H8_row_slice']
        if stop-first!=domain.matrix.shape[0]:raise ValueError('Original print QP row ledger changed')
        keep=np.ones(len(original['lower']),bool);keep[first+np.asarray(rows,int)]=False
        physical,chain=this.effective_q_and_chain(q)
        state=old.actual_physical_state(programme.state)
        geo=physical[state.world_parameter_indices];metrics,Dmetric=old.canvas_metrics(state,geo)
        margin,D,labels=chart.margins(geo[state.print_layout_world_slices[7]],metrics,Dmetric)
        pulled=(csr_matrix(D)@csr_matrix(chain)[state.world_parameter_indices]).tocsr();pulled.eliminate_zeros()
        outside=np.setdiff1d(np.arange(this.parameter_count),selected)
        if pulled[:,outside].nnz:raise ValueError('Grouped domain has undeclared nonshared dependency')
        return dict(original,matrix=vstack((csr_matrix(original['matrix'])[keep],pulled[:,selected]),format='csr'),
            lower=np.r_[original['lower'][keep],-margin],report=dict(original['report'],
                ad7_old_H8_rows_replaced=rows,ad7_grouped_domain_rows=labels,
                other_original_halfspaces_preserved=True,closed_QP_relaxation_actual_strict_guard_still_required=True,
                continuation_configuration=deepcopy(configuration_reference)))
    def arrays(this):
        result=old_arrays();_immutable_archive(result,archive,metadata)
        return dict(result,**{archive:metadata.copy()})
    def identity(this):
        result=dict(old_identity())
        # Historical legacy keys and parent archives are not overwritten. A
        # distinct live identity explicitly names the new physical model.
        result.update(ad7_grouped_continuation=deepcopy(receipt),ad7_continuation_aux_key=archive,
                      ad7_current_physical_model=deepcopy(configuration_reference))
        return result
    owner.linear_constraints=MethodType(constraints,owner)
    owner.checkpoint_arrays=MethodType(arrays,owner);owner.checkpoint_identity=MethodType(identity,owner)
    refs=[reference(__file__),reference(proposal_api.__file__),reference(old.__file__),
          deepcopy(configuration_reference),deepcopy(config['continuation']),deepcopy(entry_reference),
          deepcopy(config['material']),deepcopy(value['restored_configuration'])]
    if value['initializer'] is not None:refs.append(deepcopy(value['initializer']))
    _refs(owner,refs)
    if install_worker_wrappers:
        import native_transverse_bundle as runtime
        from calibration.program_tracking import first_shot_transverse_execution as context
        runtime.init_worker=partial(source_worker_init,configuration_reference=deepcopy(configuration_reference),
            original_initializer=runtime.init_worker,allow_unsealed_protocol=allow_unsealed_protocol)
        context._init_context_worker=partial(geometry_worker_init,configuration_reference=deepcopy(configuration_reference),
            original_initializer=context._init_context_worker,allow_unsealed_protocol=allow_unsealed_protocol)
    core._clear_caches(programme)
    if proposal_api.parent_contract(parent)!=before or not proposal_api.same(owner.start,base_start):
        raise ValueError('Successor installation changed the complete parent')
    for key,item in previous_arrays.items():
        if not proposal_api.same(owner.checkpoint_arrays()[key],item):raise ValueError('Base auxiliary changed: '+key)
    return deepcopy(receipt)


def dense_record(record):
    """Only the legacy delivery boundary changes type; CSR owner stays intact."""
    return dict(record,matrix=record['matrix'].toarray()) if issparse(record['matrix']) else record


def dense_constraints(q,selected,*,original):
    return dense_record(original(q,selected))


def install_solver_delivery(owner):
    """After fixed restore: same halfspaces delivered to legacy dense consumer."""
    if getattr(owner,'ad7_continuation_dense_delivery',False):return
    original=owner.solver_kwargs
    def kwargs(this):
        result=original();callback=result.get('shared_linear_constraints')
        if not callable(callback):raise ValueError('Existing shared-domain callback required')
        result['shared_linear_constraints']=partial(dense_constraints,original=callback)
        identity=deepcopy(result.get('checkpoint_identity',{}))
        identity['legacy_dense_constraint_delivery']=dict(producer=reference(__file__),
            owner_domain_callback_unchanged=True,solver_boundary_dense_only=True,
            same_all_halfspaces_and_lower_values=True,source_J_guard_and_thresholds_unchanged=True)
        result['checkpoint_identity']=identity
        return result
    owner.solver_kwargs=MethodType(kwargs,owner)
    owner.ad7_continuation_dense_delivery=True


@contextmanager
def dense_metadata_scope(owner):
    """Call the original metadata preflight inside; restore exact bound method."""
    existed='linear_constraints' in owner.__dict__
    instance=owner.__dict__.get('linear_constraints');original=owner.linear_constraints
    def temporary(this,q,selected):return dense_record(original(q,selected))
    owner.linear_constraints=MethodType(temporary,owner)
    try:yield
    finally:
        if existed:owner.linear_constraints=instance
        else:del owner.linear_constraints
    if existed and owner.__dict__.get('linear_constraints') is not instance:
        raise RuntimeError('Exact original instance constraint method was not restored')


def restore_fixed_columns(stage,*,parent,configuration_reference):
    """After NativeTime(active_details=None); preserve every old archive key."""
    if _GLOBAL is None or _GLOBAL['reference']!=configuration_reference:
        raise ValueError('Matching continuation execution required')
    value=_read(_GLOBAL['config']['continuation']);proposal_api.verify_configuration(value,parent)
    if not proposal_api.same(stage.start,parent['q']):raise ValueError('Complete stage canonical parent changed')
    for key,item in parent['aux'].items():
        if key not in stage._aux or not proposal_api.same(stage._aux[key],item):
            raise ValueError('Time-detail stage changed parent auxiliary: '+key)
    columns=np.asarray(value['source_free_fixed_policy']['fixed_global_slots'],int)
    if not np.array_equal(columns,np.arange(5162,5167)):raise ValueError('Exactly five text candidates must remain fixed')
    existing=getattr(stage,'ad7_continuation_fixed_receipt',None)
    if existing is not None:
        if existing['configuration']!=configuration_reference:raise ValueError('Another fixed-column restoration already exists')
        return deepcopy(existing)
    old_shared=stage.shared.copy();old_fixed=stage.fixed.copy()
    if (len(old_shared)!=172 or not np.isin(columns,old_shared).all()
            or np.isin(columns,old_fixed).any()):
        raise ValueError('Restore fixed text exactly once on the fresh172-shared native-time stage')
    fixed_local=np.arange(3,8)
    if not proposal_api.same(stage.start[columns],np.asarray(value['original_anchor8'],float)[fixed_local]):
        raise ValueError('Inherited fixed text canonical coordinates changed')
    shared=old_shared[~np.isin(old_shared,columns)]
    if len(shared)!=167:raise ValueError('Unchanged five-column reduction must leave shared167')
    old_identity=stage.checkpoint_identity
    receipt=dict(configuration=deepcopy(configuration_reference),parent_checkpoint=deepcopy(parent['checkpoint']),
        parameters=list(old.PARAMETERS[3:]),global_slots=columns.tolist(),shared_count=167,
        fixed_canonical_values=stage.start[columns].tolist(),
        physical_initialization_parameters=_GLOBAL['chart'].parameters(stage.start[np.arange(5159,5167)])[3:].tolist(),
        initialization_is_candidate_not_measured=True,not_fitted_by_loss_or_QP=True,
        all_parent_q_and_aux_bytes_retained=True,original_shared_order=old_shared.tolist(),
        complete_old_halfspace_rows_and_lower_retained=True,original_grouped_fixed_archive_preserved=True)
    archive='ad7_contfix_'+configuration_reference['sha256'][:16]+'_json'
    metadata=np.asarray(json.dumps(receipt,sort_keys=True))
    # Check archive conflicts BEFORE changing fixed/shared/methods.
    _immutable_archive(stage._aux,archive,metadata)
    def constraints(this,q,selected):
        if not np.array_equal(selected,this.shared):raise ValueError('Exact restored shared order required')
        expanded,chain=this.expanded_q_and_chain(q)
        original=this.base.linear_constraints(expanded,old_shared.copy())
        matrix=csr_matrix(original['matrix'])
        if matrix.shape!=(len(original['lower']),len(old_shared)):raise ValueError('Original shared constraint dimensions changed')
        pulled=(matrix@chain[old_shared]).tocsr();pulled.eliminate_zeros()
        outside=np.setdiff1d(np.arange(this.parameter_count),np.r_[this.shared,columns])
        if pulled[:,outside].nnz:raise ValueError('Fixed-column restoration has undeclared dependency')
        return dict(original,matrix=pulled[:,this.shared].tocsr(),report=dict(original.get('report',{}),
            grouped_initialization_only_fixed_columns=columns.tolist(),
            original_lower_and_all_halfspace_rows_retained=True,grouped_continuation_fixed_restore=True))
    def identity(this):
        return dict(old_identity(),ad7_continuation_fixed=deepcopy(receipt),
            ad7_current_physical_model=deepcopy(configuration_reference))
    stage.fixed=np.r_[old_fixed,columns];stage.shared=shared
    stage._fixed_bits=stage.start[stage.fixed].view(np.uint64).copy()
    stage.linear_constraints=MethodType(constraints,stage)
    stage.checkpoint_identity=MethodType(identity,stage)
    stage._aux.setdefault(archive,metadata.copy())
    stage.ad7_continuation_fixed_receipt=deepcopy(receipt)
    return deepcopy(receipt)
