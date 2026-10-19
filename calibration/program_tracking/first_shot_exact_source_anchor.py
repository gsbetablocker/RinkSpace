"""Reuse a certified complete source J only at the identical numerical anchor.

Fresh source values, W/loss fields and ledgers are checked before each block.
Changed q always invokes the real source producer. No H, gradient or trial cost
is reused; the public writer references the original verified CSR bytes.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import hashlib
import numpy as np
from scipy.sparse import load_npz
from .first_shot_source_transition import reference,read


def install_exact_source_anchor(owner,manifest_reference,*,transitions=(),source_prefix_owner=None):
    from .first_shot_joint_stage_runtime import sha
    from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
    from z190_streaming_source_normal import SourceRowBlock
    manifest=read(manifest_reference);parent=Path(manifest_reference['path']).parents[2]
    closed=read(reference(parent/'owner_closed.json'));qa=read(reference(parent/'full_source_directional_QA.json'))
    if (closed.get('passed')is not True or closed.get('normal_close_returned')is not True or qa.get('passed')is not True
            or manifest.get('source_J_before_Huber_weights')is not True
            or manifest.get('all_active_objective_rows_present')is not True
            or manifest['J_shape']!=[len(owner.scales),len(owner.start)]
            or qa.get('canonical_q_sha256')!=manifest['qaug_sha256']):
        raise ValueError('Closed complete same-state source J with passed full-source QA required')
    declared={}
    for item in transitions:
        old=item['historical_producer'];current=item['current_producer']
        if reference(old['path'])!=old or reference(current['path'])!=current or not item.get('reason'):
            raise ValueError('Exact old and current execution-transition bytes required')
        declared[(str(Path(item['historical_path']).resolve()),old['sha256'])]=item
    verified={};used=set()
    def verify(ref):
        key=str(Path(ref['path']).resolve()),ref['sha256']
        if key in verified:return verified[key]
        actual=reference(ref['path'])
        if actual!=ref:
            if key not in declared:raise ValueError('Undeclared cached source/producer change: '+str(ref))
            item=declared[key]
            if actual!=item['current_producer']:raise ValueError('Declared execution successor changed')
            actual=item['current_producer'];used.add(key)
        verified[key]=actual;return actual
    def references(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=set(value):verify({k:value[k]for k in ('path','sha256')})
            for v in value.values():references(v)
        elif isinstance(value,list):
            for v in value:references(v)
    references(manifest)
    # Run-local JSON provenance may move to a new output directory. Its source
    # and numeric content must remain exact after the declared file successors.
    relative_names={'execution_manifest.json','construction/source_restore.json','construction/camera_transition.json','combined_c0.json','source_successors.json'}
    generated=[]
    for ref in manifest['source_producers']:
        path=Path(ref['path'])
        if path.is_relative_to(parent)and path.relative_to(parent).as_posix()in relative_names:generated.append(ref)
    aliases={};provenance=[]
    if generated:
        candidates=[ref for ref in owner.source_producers if Path(ref['path']).name=='execution_manifest.json'and Path(ref['path']).parent.parent==parent.parent]
        if len(candidates)!=1:raise ValueError('One explicit current run provenance directory required')
        current_parent=Path(candidates[0]['path']).parent
        for old in generated:
            new=reference(current_parent/Path(old['path']).relative_to(parent))
            if new not in owner.source_producers:raise ValueError('Current source omitted a run-provenance successor')
            aliases[(str(Path(old['path']).resolve()),old['sha256'])]=new
            provenance.append(dict(previous=old,current=new))
    def equivalent(ref):
        actual=verify(ref);key=str(Path(ref['path']).resolve()),ref['sha256']
        return aliases.get(key,actual)
    def normalized_provenance(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=set(value):
                ref=equivalent({k:value[k]for k in ('path','sha256')})
                return dict(ref,**{k:normalized_provenance(v)for k,v in value.items()if k not in ('path','sha256')})
            return {k:normalized_provenance(v)for k,v in value.items()}
        if isinstance(value,list):return [normalized_provenance(v)for v in value]
        return value
    for pair in provenance:
        if normalized_provenance(read(pair['previous']))!=normalized_provenance(read(pair['current'])):
            raise ValueError('Run-provenance successor changed source, physics or numerical content: '+pair['previous']['path'])
    expected={tuple(sorted(equivalent(ref).items()))for ref in manifest['source_producers']}
    current={tuple(sorted(ref.items()))for ref in owner.source_producers}
    if expected!=current:raise ValueError('Same-state source numerical/execution producer inventory differs')
    with np.load(manifest['complete_current_state_r_scales_quadratic']['path'],allow_pickle=False)as saved:
        previous={k:saved[k].copy()for k in saved.files}
    payload=owner.complete_state_payload(owner.start)
    for key,value in payload.items():
        actual=np.asarray(value)
        if (key not in previous or actual.dtype!=previous[key].dtype or actual.shape!=previous[key].shape
                or actual.tobytes()!=previous[key].tobytes()):
            raise ValueError('Exact source anchor q/complete auxiliary payload differs: '+key)
    for key in ('scales','quadratic'):
        if not np.array_equal(previous[key],getattr(owner,key)):raise ValueError('Source anchor '+key+' differs')
    anchor=owner.start.copy()
    if sha(anchor)!=manifest['qaug_sha256']:raise ValueError('Source J anchor hash differs')
    with np.load(manifest['block_loss_weights']['path'],allow_pickle=False)as z:losses={k:z[k].copy()for k in z.files}
    def canonical(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=set(value):
                ref=equivalent({k:value[k]for k in ('path','sha256')})
                return dict(content_sha256=ref['sha256'],**{k:canonical(v)for k,v in value.items()if k not in ('path','sha256')})
            return {k:canonical(v)for k,v in value.items()}
        if isinstance(value,list):return [canonical(v)for v in value]
        return value
    original=owner.source_blocks;old_reference=getattr(owner,'source_J_block_reference',None);matrices={};matrix_refs={};validations=[]
    def validate_block(index,block,saved):
        left=saved['row_start'];stop=left+saved['row_count'];r=previous['r'][left:stop]
        if (len(block.r)!=saved['row_count']or canonical(block.row_ledger)!=canonical(saved['row_ledger'])
                or not np.array_equal(block.scales,previous['scales'][left:stop])
                or not np.array_equal(block.quadratic,previous['quadratic'][left:stop])
                or not np.allclose(block.r,r,rtol=0.,atol=2e-10)):
            raise ValueError('Current raw source/complete W/ledger differs from exact cached anchor')
        weights=source_loss_checkpoint_arrays(block.loss_weights,len(block.r))
        old={k.rsplit('__block_',1)[0]:v for k,v in losses.items()if k.endswith('__block_'+str(index))}
        if set(weights)!=set(old)or any(not np.array_equal(v,old[k])for k,v in weights.items()):
            raise ValueError('Original source loss/integration/normalization changed')
        if index not in matrices:
            verify(saved['J']);J=load_npz(saved['J']['path']).tocsr()
            if (list(J.shape)!=saved['J_shape']or J.nnz!=saved['J_nonzeros']or not np.isfinite(J.data).all()):
                raise ValueError('Certified original source CSR changed')
            matrices[index]=J;matrix_refs[id(J)]=saved['J']
        validations.append(dict(block=index,current_r_max_difference=float(np.max(abs(block.r-r),initial=0.))))
        return matrices[index]
    if source_prefix_owner is not None:
        verifier=source_prefix_owner.source_blocks.__func__
        cells=dict(zip(verifier.__code__.co_freevars,verifier.__closure__ or ()))
        if verifier.__name__!='verify_first_J'or not {'pre_verification_blocks','before','first'}<=set(cells):
            raise ValueError('Exact historical initial-J value/geometry verifier required')
        before=cells['before'].cell_contents
        if not np.array_equal(before['q'],anchor):raise ValueError('Initial-J verifier belongs to another numerical seed')
        prefix_rows=len(source_prefix_owner.scales);prefix_records=[];cursor=0
        for record in manifest['blocks']:
            if cursor>=prefix_rows:break
            cursor+=record['row_count'];prefix_records.append(record)
        if cursor!=prefix_rows:raise ValueError('Saved complete J cannot split the original verified source prefix')
        pre_source=cells['pre_verification_blocks'].cell_contents
        def cached_prefix(q,jacobian):
            if not jacobian or not np.array_equal(q,anchor):
                yield from pre_source(q,jacobian);return
            current=list(pre_source(q,False))
            if len(current)!=len(prefix_records):raise ValueError('Current verified source prefix partition changed')
            for i,(block,saved)in enumerate(zip(current,prefix_records,strict=True)):
                J=validate_block(i,block,saved)
                yield SourceRowBlock(block.r,J,block.quadratic,block.scales,block.row_ledger,loss_weights=block.loss_weights)
        cells['pre_verification_blocks'].cell_contents=cached_prefix
    def source(this,q,jacobian):
        if not jacobian or not np.array_equal(q,anchor):
            yield from original(q,jacobian);return
        # In the production chain the original initial-J verifier still runs.
        # Only its input source producer is served from authenticated exact-q J.
        actual_blocks=list(original(q,source_prefix_owner is not None))
        if len(actual_blocks)!=len(manifest['blocks']):raise ValueError('Current complete source block partition changed')
        for index,(block,saved)in enumerate(zip(actual_blocks,manifest['blocks'],strict=True)):
            cached=validate_block(index,block,saved)
            J=cached if source_prefix_owner is None else block.J
            if J is not None and (J is cached or (np.array_equal(J.data,cached.data)and np.array_equal(J.indices,cached.indices)and np.array_equal(J.indptr,cached.indptr))):
                matrix_refs[id(J)]=saved['J']
            else:matrix_refs.pop(id(J),None)
            yield SourceRowBlock(block.r,J,block.quadratic,block.scales,block.row_ledger,loss_weights=block.loss_weights)
    def block_reference(q,index,block):
        if np.array_equal(q,anchor)and id(block.J)in matrix_refs:return matrix_refs[id(block.J)]
        return old_reference(q,index,block)if callable(old_reference)else None
    owner.source_blocks=MethodType(source,owner);owner.source_J_block_reference=block_reference
    receipt=dict(schema='rinkspace.exact-anchor-complete-source-J-reuse.v1',manifest=deepcopy(manifest_reference),
        source_QA=reference(parent/'full_source_directional_QA.json'),closed_parent=reference(parent/'owner_closed.json'),
        canonical_q_sha256=sha(anchor),source_rows=len(owner.scales),columns=len(anchor),complete_payload_fields=len(payload),
        verified_execution_successors=[deepcopy(declared[k])for k in used],
        run_provenance_successors=deepcopy(provenance),all_run_JSON_content_equal_after_declared_reference_transitions=True,producer=reference(__file__),
        fresh_current_source_values_and_loss_fields_required_before_reuse=True,current_R_absolute_gate=2e-10,
        existing_first_fresh_J_R_gate_retained=True,changed_q_always_fresh_J=True,
        original_verified_CSR_referenced_without_duplicate_copy=True,H_gradient_and_trial_cost_not_reused=True,
        original_initial_J_R_and_geometry_verifier_retained=source_prefix_owner is not None,
        complete_source_directional_QA_still_required_before_trial=True)
    owner._initial_complete_J_reuse=True
    old_identity=owner.checkpoint_identity
    def identity(this):return dict(old_identity(),exact_source_anchor=deepcopy(receipt),initial_exact_complete_current_state_full_J_reuse=True)
    owner.checkpoint_identity=MethodType(identity,owner)
    if reference(__file__)not in owner.recovery_references:owner.recovery_references.append(reference(__file__))
    return receipt,validations
