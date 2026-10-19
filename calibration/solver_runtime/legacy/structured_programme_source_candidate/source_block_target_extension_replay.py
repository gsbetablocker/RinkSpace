"""Replay same-q/profile r/J while extending only real native ice sources.

The old H/g/cost are never loaded. The caller must accumulate a fresh complete
normal from this iterator and current fixed-column policy. This is one initial
linearization, never a cache for another q or another profile snapshot.
"""
from __future__ import annotations
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from copy import deepcopy,copy
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import time

BASE=_runtime_data_root()
if __name__ == '__main__':
    sys.path.insert(0, str(_runtime_source_root))
    from isolated_solver_bootstrap import install
    install()

import numpy as np
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import (
    SourceRowBlock, BlockSignature, SourceBlockContract, _checked_block, _check_replay,
)
from z190_joint_source_solver import _frozen_profile_token

ICE_SCHEMA = 'rinkspace.complete-source-plus-real-native-ice-curves.v1'
OUTER_SCHEMAS = {
    'rinkspace.complete-source-plus-six-independent-real-native-board-circle-materials.v1',
    'rinkspace.complete-source-plus-real-native-board-material-points.v1',
    'rinkspace.complete-source-plus-real-native-board-edges.v1',
}
BACKEND_KEYS = ('RINKSPACE_MOTION_BACKEND', 'RINKSPACE_PROJECTION_BACKEND',
                'RINKSPACE_SCATTER_BACKEND', 'RINKSPACE_PROJECTION_VALUE_BACKEND')
FD_KEYS = ('RINKSPACE_GEOMETRY_SPARSE_FD', 'RINKSPACE_GEOMETRY_FD_STEP_M')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                                else x.item() if isinstance(x, np.generic) else str(x)))


def _qtoken(q):
    return hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest()


def _without_extended_ice(contract):
    """Remove only declared ice extension and propagated outer row totals."""
    result = deepcopy(contract)
    schema = result.get('schema')
    if schema == ICE_SCHEMA:
        result.pop('additional_real_ice_curve_contract')
        result.pop('additional_scalar_rows')
        result.pop('total_rows')
        # The whole pre-ice contract, including wide/reference source, stays.
        return result
    if schema not in OUTER_SCHEMAS:
        raise ValueError('Expected the known outer board wrappers above one ice extension')
    result['original_source_contract'] = _without_extended_ice(result['original_source_contract'])
    result.pop('original_source_rows')
    result.pop('total_rows')
    return result


def _check_source_extension(old, current):
    from .complete_same_state_replay import _check_strict_identity
    plan=_native_append_plan(old,current,('additional_real_native_ice_curves',))
    shared=np.zeros(int(current['full_parameters']),bool);shared[np.asarray(current['shared_indices'],int)]=True
    _check_strict_identity(old,current,append_normalized_pair=_append_normalized_manifests(old,current,plan),
        fixed_parameter_indices=current['persistent_source_replay_identity']['fixed_parameter_indices'],shared_parameter_mask=shared)
    p=plan['additional_real_native_ice_curves']
    return dict(old_packet=p['old_bound'],new_packet=p['new_bound'],added_scalar_rows=p['added_scalar_rows'],
        old_source_rows_complete_prefix_byte_exact=True)


def _check_backend(old, current, evidence):
    previous, present = old['native_backend_requested'], current['native_backend_requested']
    for key in FD_KEYS:
        if previous.get(key) != present.get(key):
            raise ValueError('Geometry FD/activity policy differs: ' + key)
    from_backend = {key:previous.get(key) for key in BACKEND_KEYS}
    to_backend = {key:present.get(key) for key in BACKEND_KEYS}
    if from_backend == to_backend:
        return dict(implementation_backends_unchanged=True)
    if evidence is None:
        raise ValueError('Changed implementation backend requires actual value/J equivalence evidence')
    path = Path(evidence).resolve(); record = _read(path)
    if (record.get('schema') != 'rinkspace.same-target-backend-equivalence.v1' or
            record.get('passed') is not True or
            record.get('previous_backend') != from_backend or record.get('current_backend') != to_backend or
            record.get('source_operator_and_geometry_code_equivalent') is not True or
            record.get('residual_and_J_equivalence_verified') is not True or not record.get('evidence')):
        raise ValueError('Backend evidence does not authenticate the actual previous/current source value/J implementations')
    for item in record['evidence']:
        if _sha(item['path']) != item['sha256']:
            raise ValueError('Referenced backend equivalence evidence changed')
    return dict(implementation_backends_unchanged=False,
                equivalence_record=dict(path=str(path),sha256=_sha(path)),
                previous_backend=from_backend,current_backend=to_backend,
                no_source_geometry_FD_sigma_role_or_target_change_authorized_by_backend_record=True)


def _check_current_ice_ledger(ledger, manifest):
    bound=manifest['additional_real_native_ice_curves']
    packet=_read(bound['packet_path'])
    fitted=[row for row in packet['rows'] if row['split']=='fit']
    held=[row for row in packet['rows'] if row['split']=='holdout']
    expected=dict(family='additional_real_native_ice_curves',
        packet_sha256=bound['packet_sha256'],objective_schema=bound['schema'],
        group_sqrt_N_multiplier_applied=False,native_pts=[row['pts'] for row in fitted],
        actual_pts=[row['actual_pts'] for row in fitted],
        feature_ids=[row['feature_id'] for row in fitted],
        fit_observation_ids=[row['id'] for row in fitted],
        held_observation_ids=[row['id'] for row in held],
        plan_pixels_used=False,paint_endpoints_inferred=False)
    actual={key:value for key,value in ledger.items() if key not in ('original_start','original_stop')}
    if actual != expected:
        raise ValueError('Actual extended ice ledger differs from the authenticated source packet')


def _old_cache_identity(directory):
    directory = Path(directory).resolve()
    mp = directory / 'input_manifest.json'; old = _read(mp)
    with np.load(directory/'actual_initial_normal_state.npz',allow_pickle=False) as z:
        q = z['q'].copy(); token = str(z['frozen_profile_token'])
        if str(z['source_manifest_sha256']) != _sha(mp):
            raise ValueError('Initial state manifest hash changed')
    # NPZ arrays are lazy: load ONLY the contract JSON, never old H/g/cost.
    with np.load(directory/'first_complete_source_normal_numeric.npz',allow_pickle=False) as z:
        payload = json.loads(str(z['source_contract_json']))
    contract = SourceBlockContract(payload['parameter_count'],payload['row_count'],
                                   tuple(BlockSignature(**item) for item in payload['signatures']),
                                   payload['frozen_profile_token'])
    prefix = directory/'initial_normal_source_blocks'
    completion = _read(prefix/'completed_prefix.json')
    if (completion['source_manifest_sha256'] != _sha(mp) or completion['q_sha256'] != _qtoken(q) or
            completion['frozen_profile_token'] != token or contract.frozen_profile_token != token or
            contract.parameter_count != len(q) or len(completion['completed_blocks']) != len(contract.signatures) or
            completion['completed_original_rows'] != contract.row_count):
        raise ValueError('Complete old source blocks/state/profile inventory is not authenticated')
    return directory, old, q, token, contract, completion


def prepare_same_state_ice_extension_blocks(directory, q, source, manifest, profile_token, *,
                                           backend_equivalence_evidence=None,
                                           fixed_parameter_indices=None,shared_parameter_mask=None):
    """Return (fresh_iterator_factory, compact_report) for one initial normal.

    Restore actual_initial_frozen_profile_arrays into source.updater FIRST.
    Run required small new-source checks and current guards in the caller.
    The caller MUST accumulate a new normal with this new target's masks,
    row count, fixed indices and replay contract; do not load old H/g.
    """
    # Preserve the public historical ice API while using strict sealed replay.
    # Changed implementation backends cannot authorize reuse via a prose flag.
    if backend_equivalence_evidence is not None:
        raise ValueError('Changed backend requires a fresh source normal; backend evidence cannot unlock cached r/J')
    return prepare_same_state_source_extension_blocks(directory,q,source,manifest,profile_token,
        fixed_parameter_indices=fixed_parameter_indices,shared_parameter_mask=shared_parameter_mask,
        permitted_families=('additional_real_native_ice_curves',))


APPEND_NATIVE_FACTORS={
    'additional_real_native_ice_curves':('native_ice_factor','new_scalar_rows'),
    'additional_real_native_board_edges':('native_board_factor','scalar_rows'),
}
APPEND_CONTRACT_INVENTORY={
    'additional_real_native_ice_curves':{'packet_path','packet_sha256','new_fit_observation_ids',
        'held_observation_ids','new_scalar_rows','explicit_optical_normal_support_ids'},
    'additional_real_native_board_edges':{'packet_path','packet_sha256','fit_ids','held_ids','scalar_rows',
        'advertisement_numbers','far_visible_board_top_ids','far_visible_ice_board_contact_ids',
        'physical_course_facet_ids','print_extent_scopes','historical_scope_aliases'},
}


def _native_append_plan(old,current,permitted_families):
    """Authenticate all old typed row fields, not just UV or a role count."""
    plan={}
    for family in permitted_families:
        a,b=old.get(family),current.get(family)
        if a==b:continue
        if not isinstance(a,dict)or not isinstance(b,dict):
            raise ValueError('Append replay requires an existing explicit parent block: '+family)
        permitted=APPEND_CONTRACT_INVENTORY[family]
        if {k:v for k,v in a.items()if k not in permitted}!={k:v for k,v in b.items()if k not in permitted}:
            raise ValueError('Native source operator/layout/weight domain changed: '+family)
        for bound in (a,b):
            if _sha(bound['packet_path'])!=bound['packet_sha256']:raise ValueError('Bound append packet changed')
        op,npk=_read(a['packet_path']),_read(b['packet_path'])
        for field in ('schema','source_video_sha256','time_base'):
            if op[field]!=npk[field]:raise ValueError('Native packet source identity changed: '+field)
        for field in ('frame_size_wh','image_size_wh','source_measurement_candidate','legacy_internal_scope_alias_authority'):
            if op.get(field)!=npk.get(field):raise ValueError('Native packet support/scope authority changed: '+field)
        if npk['rows'][:len(op['rows'])]!=op['rows']or len(npk['rows'])<=len(op['rows']):
            raise ValueError('All original source fields/UV/sigma/roles/support/clock must be an exact parent prefix')
        ids=[r['id']for r in npk['rows']]
        if len(ids)!=len(set(ids)):raise ValueError('Append source IDs must be unique')
        appended=npk['rows'][len(op['rows']):]
        if any(r['split']not in ('fit','holdout')for r in appended):raise ValueError('Explicit FIT/HELD append roles required')
        for key in permitted-{'packet_path','packet_sha256','new_scalar_rows','scalar_rows'}:
            left,right=a.get(key),b.get(key)
            if isinstance(left,dict):
                if not isinstance(right,dict)or any(right.get(k)!=v for k,v in left.items()):raise ValueError('Old typed inventory identity changed: '+key)
            elif isinstance(left,list):
                # A newly appended FIT precedes historical HELD in some derived
                # summaries. Filter only new identities; old order stays exact.
                if key=='advertisement_numbers':
                    if not set(left).issubset(right):raise ValueError('Old advertisement identity changed')
                elif right[:len(left)]!=left and [v for v in right if v in left]!=left:
                    raise ValueError('Old typed FIT/HELD inventory changed: '+key)
            elif left!=right:raise ValueError('Old typed inventory field changed: '+key)
        count_key=APPEND_NATIVE_FACTORS[family][1]
        oldfit=[r for r in op['rows']if r['split']=='fit'];newfit=[r for r in appended if r['split']=='fit']
        old_scalars=sum(2*len(r['uv'])for r in oldfit);added=sum(2*len(r['uv'])for r in newfit)
        if a[count_key]!=old_scalars or b[count_key]!=old_scalars+added:
            raise ValueError('Native append scalar count differs from actual typed FIT rows')
        plan[family]=dict(old_bound=a,new_bound=b,old_fit_count=len(oldfit),old_scalar_rows=old_scalars,
            added_scalar_rows=added,appended_fit_ids=[r['id']for r in newfit],appended_ids=[r['id']for r in appended])
    if not plan:raise ValueError('An actual append-only source extension is required')
    return plan


def _append_normalized_manifests(old,current,plan):
    from .complete_same_state_replay import _semantic_manifest
    a,b=_semantic_manifest(old),_semantic_manifest(current)
    def pair(left,right):
        for family,p in plan.items():
            if left==p['old_bound']and right==p['new_bound']:
                return {'authenticated_parent_plus_append_family':family},{'authenticated_parent_plus_append_family':family},p['added_scalar_rows']
        if isinstance(left,dict)and isinstance(right,dict):
            if set(left)!=set(right):raise ValueError('Append replay source contract keys changed')
            x,y={},{};deltas={}
            counts={'source_rows','total_rows','original_source_rows','additional_scalar_rows'}
            for key in left.keys()-counts:
                x[key],y[key],deltas[key]=pair(left[key],right[key])
            delta=sum(deltas.values())
            for key in left.keys()&counts:
                allowed=deltas.get('original_source_contract',0)if key=='original_source_rows'else (
                    delta-deltas.get('original_source_contract',0)if key=='additional_scalar_rows'else delta)
                if left[key]!=right[key]:
                    if not isinstance(left[key],int)or right[key]-left[key]!=allowed:
                        raise ValueError('Append propagated source scalar inventory differs: '+key)
                    x[key]=y[key]={'unchanged_parent_count':left[key],'authenticated_added_scalars':allowed}
                else:x[key]=left[key];y[key]=right[key]
            return x,y,delta
        if left!=right:raise ValueError('Unchanged source/layout/domain/backend field differs')
        return left,right,0
    # Each factor contract occurs both in the top manifest and nested source
    # contract. Validate top totals once, rather than counting these twice.
    for family,p in plan.items():a[family]=b[family]={'authenticated_parent_plus_append_family':family}
    total=sum(p['added_scalar_rows']for p in plan.values())
    if b['source_rows']-a['source_rows']!=total:raise ValueError('Complete appended source row count differs')
    a['source_rows']=b['source_rows']={'parent_count':a['source_rows'],'added_scalars':total}
    a['source_contract'],b['source_contract'],delta=pair(a['source_contract'],b['source_contract'])
    if delta!=total:raise ValueError('Nested source contract omitted an appended native factor')
    return a,b


def _appended_native_block(source,factor,q,old_fit_count,old_scalars,original_start):
    """Evaluate only appended FIT rows, leaving the original actor immutable."""
    selected=copy(factor);selected.rows=factor.rows[old_fit_count:];selected._cache=OrderedDict()
    selected.scales=factor.scales[old_scalars:];selected.quadratic=factor.quadratic[old_scalars:]
    selected.row_count=len(selected.scales)
    selected.blocks={r['actual_pts']:factor.blocks[r['actual_pts']]for r in selected.rows}
    layout=getattr(factor,'canonical_parameter_layout',None)
    value=np.asarray(q,float)if layout is None else layout.evaluation_q(q)
    block=selected.block(value[:factor.parameter_count],True,original_start=original_start)
    J=block.J.tocsr(copy=False)
    J=csr_matrix((J.data,J.indices,J.indptr),shape=(len(block.r),len(q)),copy=False)
    if layout is not None:J=J[:,layout.canonical_to_evaluation].tocsr()
    return SourceRowBlock(block.r,J,block.quadratic,block.scales,block.row_ledger)


def prepare_same_state_source_extension_blocks(directory,q,source,manifest,profile_token,*,
        fixed_parameter_indices=None,shared_parameter_mask=None,permitted_families=tuple(APPEND_NATIVE_FACTORS)):
    from .complete_same_state_replay import _check_strict_identity,_sealed_inventory,_array_signature
    directory,old,oldq,token,contract,completion=_old_cache_identity(directory);current=_plain(manifest)
    if _array_signature(q)!=_array_signature(oldq)or profile_token!=token:
        raise ValueError('Cached original r/J cannot be reused at a different exact q/profile')
    plan=_native_append_plan(old,current,permitted_families)
    _check_strict_identity(old,current,append_normalized_pair=_append_normalized_manifests(old,current,plan),
        fixed_parameter_indices=fixed_parameter_indices,shared_parameter_mask=shared_parameter_mask)
    _sealed_inventory(directory,q,source,token,append=True)
    expected=contract.row_count+sum(p['added_scalar_rows']for p in plan.values())
    if len(source.scales)!=expected or source.quadratic.shape!=source.scales.shape:raise ValueError('Current appended source shape differs')
    for family,p in plan.items():
        factor=getattr(source,APPEND_NATIVE_FACTORS[family][0])
        if _plain(factor.contract)!=p['new_bound']:raise ValueError('Actual attached native factor differs from manifest')
    frozen=np.asarray(q).copy();frozen.flags.writeable=False
    report=dict(schema='rinkspace.strict-same-state-native-parent-plus-append-r-J-replay.v1',
        old_H_g_or_cost_loaded=False,new_H_g_and_replay_contract_required=True,exact_old_full_source_prefix_verified=True,
        source_directory=str(directory),full_q_sha256=_qtoken(q),frozen_profile_token=token,
        old_source_rows=contract.row_count,new_source_rows=expected,appended_families=list(plan),loaded_cached_blocks=[])
    def factory():
        if _array_signature(q)!=_array_signature(frozen)or _frozen_profile_token(source.updater)!=token:
            raise ValueError('Append replay q/profile changed')
        cursor=0;seen=set()
        for ordinal,signature in enumerate(contract.signatures):
            oldblock,_=_load_authenticated_old_block(directory,ordinal,contract,frozen,token)
            family=oldblock.row_ledger['family'];ledger=deepcopy(oldblock.row_ledger);nr=oldblock.r;nj=oldblock.J
            mask=oldblock.quadratic;scale=oldblock.scales;added=0
            if family in plan:
                p=plan[family];factor=getattr(source,APPEND_NATIVE_FACTORS[family][0]);seen.add(family)
                if len(oldblock.r)!=p['old_scalar_rows']:raise ValueError('Old typed factor cached row inventory differs')
                if not np.array_equal(factor.scales[:len(scale)],scale)or not np.array_equal(factor.quadratic[:len(mask)],mask):
                    raise ValueError('Old native factor raw sigma or quadratic mask differs')
                if p['added_scalar_rows']:
                    fresh=_appended_native_block(source,factor,frozen,p['old_fit_count'],len(nr),cursor+len(nr))
                    from scipy.sparse import vstack
                    nr=np.r_[nr,fresh.r];nj=vstack((nj,fresh.J),format='csr');mask=np.r_[mask,fresh.quadratic];scale=np.r_[scale,fresh.scales]
                ledger['packet_sha256']=p['new_bound']['packet_sha256']
                ledger['fit_observation_ids']=[r['id']for r in factor.rows];ledger['held_observation_ids']=[r['id']for r in factor.held]
                ledger['native_pts']=[r['pts']for r in factor.rows];ledger['actual_pts']=[r['actual_pts']for r in factor.rows]
                if 'feature_ids'in ledger:ledger['feature_ids']=[r['feature_id']for r in factor.rows]
                added=p['added_scalar_rows']
            stop=cursor+len(nr);ledger.update(original_start=cursor,original_stop=stop)
            if not np.array_equal(mask,source.quadratic[cursor:stop])or not np.array_equal(scale,source.scales[cursor:stop]):
                raise ValueError('Actual retained mask/raw sigma/layout differs at replay block')
            report['loaded_cached_blocks'].append(dict(ordinal=ordinal,family=family,parent_rows=len(oldblock.r),
                only_appended_scalar_rows_evaluated=added,original_r_J_signature=asdict(signature)))
            yield SourceRowBlock(nr,nj,mask,scale,ledger);cursor=stop
        if cursor!=expected or seen!=set(plan)or _array_signature(q)!=_array_signature(frozen)or _frozen_profile_token(source.updater)!=token:
            raise ValueError('Append replay complete inventory or final q/profile differs')
    return factory,report


def _load_authenticated_old_block(directory, ordinal, old_contract, q, token):
    path=directory/'initial_normal_source_blocks'/(str(ordinal).zfill(3)+'.npz')
    with np.load(path,allow_pickle=False) as z:
        if str(z['q_sha256']) != _qtoken(q) or str(z['frozen_profile_token']) != token:
            raise ValueError('Cached comparison block q/profile differs')
        block=SourceRowBlock(z['r'].copy(),
            csr_matrix((z['J_data'].copy(),z['J_indices'].copy(),z['J_indptr'].copy()),shape=tuple(z['J_shape'])),
            z['quadratic'].copy(),z['scales'].copy(),json.loads(str(z['ledger_json'])))
    r,j,mask,scales,signature=_checked_block(block,len(q),True)
    _check_replay(signature,ordinal,old_contract,True)
    return SourceRowBlock(r,j,mask,scales,block.row_ledger), signature


def _verified_actual_prefix(path,directory,old_contract,q,token,current,extension,tolerances):
    path=Path(path).resolve();saved=_read(path);old=_read(directory/'input_manifest.json')
    if (saved.get('schema')!='rinkspace.actual-same-q-profile-source-backend-and-ice-extension-audit.v1' or
            Path(saved.get('source_directory','')).resolve()!=directory or
            saved.get('old_source_manifest_sha256')!=_sha(directory/'input_manifest.json') or
            saved.get('full_q_sha256')!=_qtoken(q) or saved.get('frozen_profile_token')!=token or
            saved.get('previous_backend')!={k:old['native_backend_requested'].get(k) for k in BACKEND_KEYS} or
            saved.get('current_backend')!={k:current['native_backend_requested'].get(k) for k in BACKEND_KEYS} or
            saved.get('extension')!=extension or saved.get('new_source_rows')!=old_contract.row_count+extension['added_scalar_rows'] or
            saved.get('source_operator_and_geometry_contract_except_appended_ice_unchanged') is not True or
            saved.get('residual_tolerance')!=tolerances[0] or saved.get('J_tolerance')!=tolerances[1]):
        raise ValueError('Partial actual backend audit is not the identical q/profile/source/target/backend/tolerance')
    entries=saved.get('actual_new_source_blocks',[])
    if not entries or len(entries)>5:
        raise ValueError('Only the actual completed programme prefix of at most five blocks can resume')
    for ordinal,item in enumerate(entries):
        family=item.get('family','')
        if (item.get('ordinal')!=ordinal or not family.startswith('programme:') or
                item.get('passed') is not True or item.get('old_ice_prefix_only') is not False or
                item.get('r_entries_outside_declared_roundoff_gate')!=0 or
                item.get('J_entries_outside_declared_roundoff_gate')!=0 or
                item.get('original_source_masks_scales_roles_exact') is not True or
                item.get('actual_new_rows')!=old_contract.signatures[ordinal].rows or
                item.get('compared_original_rows')!=old_contract.signatures[ordinal].rows):
            raise ValueError('Every resumed original programme block must have an actual passed full r/J comparison')
        for key in ('normalized_r_max_absolute_difference','native_px_r_max_absolute_difference',
                    'J_max_absolute_difference','native_px_J_max_absolute_difference'):
            if not np.isfinite(item.get(key,float('nan'))) or item[key]<0:
                raise ValueError('Finite measured actual prefix differences required')
        old_block,signature=_load_authenticated_old_block(directory,ordinal,old_contract,q,token)
        if item.get('old_signature')!=asdict(signature):
            raise ValueError('Actual passed prefix does not authenticate this saved original r/J block')
        if family!=old_block.row_ledger['family']:
            raise ValueError('Actual passed programme prefix order changed')
        del old_block
    return dict(path=str(path),sha256=_sha(path),entries=deepcopy(entries),
        prior_overall_passed=saved.get('passed'),prior_complete=saved.get('complete'),
        complete_original_prefix_blocks=len(entries),current_native_J_roundoff_measured_not_recomputed=True)


@contextmanager
def _cached_programme_prefix_context(raw,prefix,directory,old_contract,q,token):
    """Restore cached raw programme rows before the ORIGINAL bias wrappers.

    Only the identical initial linearization enters this temporary route.
    Original prior values are computed from q and checked after the same bias
    subtraction; every untouched raw r/J entry comes from authenticated cache.
    The pool scheduler sees only remaining blocks, so omitted prefix blocks
    are never evaluated. All source wrappers still build their actual ledgers.
    """
    from correlated_roll_prior_candidate.explicit_bias_source import theta_and_gradient,SIGMA
    core_count=len(raw.state.q);count=prefix['complete_original_prefix_blocks']
    original_evaluations=raw.evaluations;actors=raw.actors
    if actors is None or actors._busy:
        raise ValueError('Explicit idle owned native actors required to resume this verified prefix')
    complete_blocks=actors.blocks;expected=[item['family'][10:] for item in prefix['entries']]
    if [block.name for block in raw.state.blocks[:count]]!=expected or [block.name for block in complete_blocks[:count]]!=expected:
        raise ValueError('Owned actor programme order differs from verified actual prefix')
    remaining=tuple(complete_blocks[count:])
    if not remaining:
        raise ValueError('Actual remaining programme block required for this partial resume')
    installations=list(raw.state.ids);bias=np.asarray(q)[core_count:core_count+3]
    if len(installations)!=3 or bias.shape!=(3,):
        raise ValueError('Original explicit hardware bias suffix required')
    had_override='evaluations' in raw.__dict__;override=raw.__dict__.get('evaluations')
    def evaluations(core,jacobian=True):
        core=np.asarray(core,float)
        if not jacobian or not np.array_equal(core,np.asarray(q)[:core_count]):
            raise ValueError('Cached native-equivalent prefix is only the exact initial q Jacobian pass')
        for ordinal,programme in enumerate(raw.state.blocks[:count]):
            block,_=_load_authenticated_old_block(directory,ordinal,old_contract,q,token)
            n=len(programme.bundle.stamps);selected=np.arange(len(block.r)-n,len(block.r))
            installation=installations.index(programme.bundle.calibration.installation_id)
            suffix=block.J[:,core_count:].tocoo();suffix.eliminate_zeros()
            if (suffix.nnz!=n or not np.array_equal(suffix.row,selected) or
                    not np.all(suffix.col==installation) or not np.all(suffix.data==-1./SIGMA)):
                raise ValueError('Cached programme suffix has changes beyond its exact original statistical bias column')
            views=core[programme.start:programme.start+4*n].reshape(-1,4)
            prior=theta_and_gradient(views[:,:3])[0]/SIGMA
            if not np.array_equal(prior-bias[installation]/SIGMA,block.r[selected]):
                raise ValueError('Actual q roll prior minus original bias is not byte-exact cached final r')
            residual=block.r.copy();residual[selected]=prior;matrix=block.J[:,:core_count].tocsr()
            del block,suffix,prior
            yield 'programme:'+programme.name,residual,matrix
            del residual,matrix
        yield from original_evaluations(core,True)
    try:
        actors.blocks=remaining;raw.evaluations=evaluations
        yield
    finally:
        actors.blocks=complete_blocks
        if had_override:raw.evaluations=override
        else:raw.__dict__.pop('evaluations',None)


def audit_same_state_ice_extension_blocks(directory,q,source,manifest,profile_token, *,
                                         residual_atol=1e-6,residual_rtol=1e-9,
                                         jacobian_atol=2e-5,jacobian_rtol=2e-7,
                                         verified_prefix_report=None,raw=None):
    """Return (factory, report) that computes every CURRENT source r/J.

    Each current original block is compared to the authenticated saved NumPy
    block at the same q/profile. The extended ice block compares only its old
    exact source prefix. All yielded blocks are ACTUAL newly computed blocks;
    the caller accumulates this pass into the NEW target normal. No equivalence
    is preclaimed. Tolerances are declared implementation-roundoff gates,
    never localization accuracy criteria or fitted source weights.
    With an authenticated partial actual report, only its already passed
    programme prefix uses cached old r/J; remaining blocks are actually
    recomputed. Pass its explicit CurrentProgrammeSource owner as raw=raw.
    The caller records every yielded final SourceRowBlock for continuation.
    """
    directory,old,oldq,token,old_contract,completion=_old_cache_identity(directory)
    current=_plain(manifest);q=np.asarray(q,float)
    if not np.array_equal(q,oldq) or q.shape != oldq.shape:
        raise ValueError('Actual native-vs-saved source comparison requires the identical complete q')
    if profile_token != token or _frozen_profile_token(source.updater) != token:
        raise ValueError('Actual source comparison requires the identical frozen profile arrays')
    from .complete_same_state_replay import _authenticate_profiles
    _authenticate_profiles(directory,source.updater,token)
    extension=_check_source_extension(old,current)
    for key in FD_KEYS:
        if old['native_backend_requested'].get(key) != current['native_backend_requested'].get(key):
            raise ValueError('An implementation comparison must preserve geometry FD policy: '+key)
    expected_rows=old_contract.row_count+extension['added_scalar_rows']
    if len(source.scales) != expected_rows:
        raise ValueError('Current target row count is not the declared ice extension')
    for value in (residual_atol,residual_rtol,jacobian_atol,jacobian_rtol):
        if not np.isfinite(value) or value<0:
            raise ValueError('Explicit finite nonnegative implementation tolerances required')
    prefix=None
    if verified_prefix_report is not None:
        prefix=_verified_actual_prefix(verified_prefix_report,directory,old_contract,q,token,current,extension,
            (dict(atol=residual_atol,rtol=residual_rtol),dict(atol=jacobian_atol,rtol=jacobian_rtol)))
        if raw is None:
            raise ValueError('Pass the explicit raw=CurrentProgrammeSource owner for verified programme prefix resume')
    report=dict(schema='rinkspace.actual-same-q-profile-source-backend-and-ice-extension-audit.v1',
        passed=None,complete=False,source_directory=str(directory),
        old_source_manifest_sha256=_sha(directory/'input_manifest.json'),
        full_q_sha256=_qtoken(q),frozen_profile_token=token,
        previous_backend={k:old['native_backend_requested'].get(k) for k in BACKEND_KEYS},
        current_backend={k:current['native_backend_requested'].get(k) for k in BACKEND_KEYS},
        source_operator_and_geometry_contract_except_appended_ice_unchanged=True,
        extension=extension,old_source_blocks=len(old_contract.signatures),
        new_source_rows=expected_rows,actual_new_source_blocks=[],
        residual_tolerance=dict(atol=residual_atol,rtol=residual_rtol),
        J_tolerance=dict(atol=jacobian_atol,rtol=jacobian_rtol),
        tolerances_are_implementation_roundoff_gates_not_source_weights_or_accuracy=True,
        old_H_g_or_cost_loaded=False,all_normal_blocks_actually_recomputed=prefix is None,
        verified_cached_prefix=prefix,remaining_blocks_actually_recomputed=True,
        require_accumulate_new_target_normal_from_this_actual_pass=True,accuracy_accepted=False)
    frozen=q.copy();frozen.flags.writeable=False

    def factory():
        if report['complete'] or report['actual_new_source_blocks']:
            raise ValueError('Actual source audit is a single pass; retain its new normal for solver replay')
        if not np.array_equal(q,frozen) or _frozen_profile_token(source.updater) != token:
            raise ValueError('Source audit q/profile changed before evaluation')
        started=time.perf_counter();previous=started;cursor=0;count=0
        try:
            def current_blocks():
                if prefix is None:
                    yield from source.blocks(frozen,True)
                else:
                    with _cached_programme_prefix_context(raw,prefix,directory,old_contract,frozen,token):
                        yield from source.blocks(frozen,True)
            for ordinal,new in enumerate(current_blocks()):
                ready=time.perf_counter();producer_wait=ready-previous
                if ordinal>=len(old_contract.signatures):
                    raise ValueError('Actual new source factory added an undeclared block')
                old_block,old_signature=_load_authenticated_old_block(directory,ordinal,old_contract,frozen,token)
                nr,nj,nmask,nscales,new_signature=_checked_block(new,len(frozen),True)
                stop=cursor+len(nr)
                if (new.row_ledger.get('original_start')!=cursor or
                        new.row_ledger.get('original_stop')!=stop):
                    raise ValueError('Actual new source ledger differs from the new target cursor')
                if (not np.array_equal(nmask,source.quadratic[cursor:stop]) or
                        not np.array_equal(nscales,source.scales[cursor:stop])):
                    raise ValueError('Actual current source factory mask/scales disagree with its manifest inventory')
                is_ice=old_block.row_ledger['family']=='additional_real_native_ice_curves'
                if is_ice:
                    _check_current_ice_ledger(new.row_ledger,current)
                ncompare=len(old_block.r)
                if (is_ice and len(nr)!=ncompare+extension['added_scalar_rows']) or (not is_ice and len(nr)!=ncompare):
                    raise ValueError('Only the declared ice block may extend its scalar rows')
                compare_ledger=deepcopy(new.row_ledger)
                compare_ledger['original_start']=old_block.row_ledger['original_start']
                compare_ledger['original_stop']=old_block.row_ledger['original_stop']
                if not is_ice and compare_ledger!=old_block.row_ledger:
                    raise ValueError('Actual unchanged source block identity/ledger differs')
                if (not np.array_equal(nmask[:ncompare],old_block.quadratic) or
                        not np.array_equal(nscales[:ncompare],old_block.scales)):
                    raise ValueError('Actual old source prefix mask/scales changed')
                rd=nr[:ncompare]-old_block.r
                rt=residual_atol+residual_rtol*np.maximum(np.abs(nr[:ncompare]),np.abs(old_block.r))
                rbad=int(np.count_nonzero(np.abs(rd)>rt))
                delta=(nj[:ncompare]-old_block.J).tocoo();delta.eliminate_zeros()
                old_values=np.asarray(old_block.J[delta.row,delta.col]).ravel()
                new_values=old_values+delta.data
                jt=jacobian_atol+jacobian_rtol*np.maximum(np.abs(old_values),np.abs(new_values))
                jbad=int(np.count_nonzero(np.abs(delta.data)>jt))
                entry=dict(ordinal=ordinal,family=new.row_ledger['family'],
                    actual_new_rows=len(nr),compared_original_rows=ncompare,
                    old_ice_prefix_only=is_ice,fresh_current_source_wait_seconds=producer_wait,
                    comparison_seconds=time.perf_counter()-ready,
                    normalized_r_max_absolute_difference=float(np.max(np.abs(rd),initial=0.)),
                    native_px_r_max_absolute_difference=float(np.max(np.abs(rd)*old_block.scales,initial=0.)),
                    J_max_absolute_difference=float(np.max(np.abs(delta.data),initial=0.)),
                    native_px_J_max_absolute_difference=float(np.max(np.abs(delta.data)*old_block.scales[delta.row],initial=0.)),
                    r_entries_outside_declared_roundoff_gate=rbad,J_entries_outside_declared_roundoff_gate=jbad,
                    old_signature=asdict(old_signature),actual_current_signature=asdict(new_signature),
                    original_source_masks_scales_roles_exact=True,passed=rbad==0 and jbad==0)
                cached=prefix is not None and ordinal<prefix['complete_original_prefix_blocks']
                entry.update(r_J_recomputed=not cached,
                    evaluation_origin='authenticated_saved_original_r_J_with_previous_actual_native_equivalence' if cached else 'actual_current_native_source_evaluation')
                if cached:
                    if new_signature!=old_signature:
                        raise ValueError('Original source wrappers did not reconstruct the authenticated cached final r/J byte-exactly')
                    entry['previous_actual_native_comparison']=deepcopy(prefix['entries'][ordinal])
                report['actual_new_source_blocks'].append(entry)
                if not entry['passed']:
                    raise ValueError('Actual source value/J implementation equivalence failed: '+entry['family'])
                # The NEW real block, never the old block, feeds the new normal.
                cursor=stop;count+=1
                del old_block,old_values,new_values,delta,rd,rt,jt
                # Retained replay must outlive an actor queued/evicted for a
                # different source. Detach current arrays without changing any
                # values; shared transport itself still follows the old protocol.
                yield SourceRowBlock(nr.copy(),nj.copy(),nmask.copy(),nscales.copy(),deepcopy(new.row_ledger))
                previous=time.perf_counter()
            if count!=len(old_contract.signatures) or cursor!=expected_rows:
                raise ValueError('Actual source audit omitted blocks or new target rows')
            if not np.array_equal(q,frozen) or _frozen_profile_token(source.updater) != token:
                raise ValueError('Same-state q/profile changed during actual source audit')
            report.update(passed=True,complete=True,elapsed_seconds=time.perf_counter()-started,
                          residual_and_J_equivalence_verified=True)
        except Exception as exc:
            report.update(passed=False,complete=False,error=dict(type=type(exc).__name__,message=str(exc)),
                          elapsed_seconds=time.perf_counter()-started)
            raise
    return factory,report


class _ProfileArrays:
    def __init__(self,values):self.values=values
    def iter_checkpoint_arrays(self):yield from self.values.items()


def make_initializer_only_seed(directory, checkpoint_template, output, *, report_path=None):
    """Create a fresh initializer-only seed from actual initial q/profiles.

    The historical checkpoint supplies only auxiliary chart metadata. No
    accepted history survives, and no q in this file is claimed optimized.
    """
    directory,old,q,token,contract,completion = _old_cache_identity(directory)
    output=Path(output).resolve();template=Path(checkpoint_template).resolve()
    if (output.exists() or output.suffix.lower()!='.npz' or
            not output.is_relative_to(BASE.resolve()) or output.is_relative_to(directory)):
        raise ValueError('Fresh separate Data initializer output required; never write the running directory')
    rp=Path(report_path).resolve() if report_path else output.with_suffix('.INITIALIZER_ONLY.json')
    if rp.exists() or not rp.is_relative_to(BASE.resolve()) or rp.is_relative_to(directory):
        raise ValueError('Fresh separate initializer provenance output required')
    with np.load(template,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files if k.startswith('aux__')}
    with np.load(directory/'actual_initial_frozen_profile_arrays.npz',allow_pickle=False) as z:
        profiles={k:z[k].copy() for k in z.files}
    if _frozen_profile_token(_ProfileArrays(profiles)) != token:
        raise ValueError('Frozen profile NPZ order/arrays differ from actual initial snapshot')
    masks=[];scales=[]
    for ordinal,expected in enumerate(contract.signatures):
        with np.load(directory/'initial_normal_source_blocks'/(str(ordinal).zfill(3)+'.npz'),allow_pickle=False) as z:
            if str(z['q_sha256']) != _qtoken(q) or str(z['frozen_profile_token']) != token:
                raise ValueError('Initializer source mask/scales state differs')
            mask=z['quadratic'].copy();scale=z['scales'].copy()
            if mask.shape != (expected.rows,) or scale.shape != mask.shape:
                raise ValueError('Initializer original source mask/scales inventory differs')
            masks.append(mask);scales.append(scale)
    arrays.update(schema=np.asarray('rinkspace.actual-initial-q-profiles-INITIALIZER_ONLY.v1'),
                  identity_json=np.asarray(json.dumps(dict(input_manifest=dict(path=str(directory/'input_manifest.json'),
                    sha256=_sha(directory/'input_manifest.json')),initializer_only=True,accuracy_accepted=False))),
                  q=q,initial_q=q.copy(),quadratic=np.concatenate(masks),scales=np.concatenate(scales),
                  damping=np.asarray(float(old['optimizer_damping_continuation']['value'])),
                  history_json=np.asarray('[]'),reviews_json=np.asarray('[]'),
                  elapsed_seconds=np.asarray(0.),initializer_only=np.asarray(True),
                  source_actual_initial_q_sha256=np.asarray(_qtoken(q)),
                  source_actual_frozen_profile_token=np.asarray(token))
    arrays.update({'profile__'+key:value for key,value in profiles.items()})
    if len(arrays['scales']) != contract.row_count:
        raise ValueError('Initializer source mask/scales omitted rows')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    report=dict(schema='rinkspace.actual-initial-q-profiles-initializer-only-export.v1',
                output=dict(path=str(output),sha256=_sha(output)),source_directory=str(directory),
                source_initial_q= dict(path=str(directory/'actual_initial_normal_state.npz'),
                    sha256=_sha(directory/'actual_initial_normal_state.npz'),q_sha256=_qtoken(q)),
                source_initial_profiles=dict(path=str(directory/'actual_initial_frozen_profile_arrays.npz'),
                    sha256=_sha(directory/'actual_initial_frozen_profile_arrays.npz'),frozen_profile_token=token),
                historical_auxiliary_template=dict(path=str(template),sha256=_sha(template)),
                q_parameter_count=len(q),source_mask_scale_rows=contract.row_count,
                original_aux_chart_metadata_copied=True,
                optimizer_history_reviews_elapsed_cleared=True,accepted_checkpoint_created=False,
                new_geometry_fit_or_camera_fit_performed=False,source_arrays_unchanged=True,
                initializer_only=True,accuracy_accepted=False)
    rp.parent.mkdir(parents=True,exist_ok=True)
    with rp.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-initializer-only-seed',action='store_true',required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--checkpoint-template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    print(json.dumps(make_initializer_only_seed(args.directory,args.checkpoint_template,args.output,
                                               report_path=args.report),ensure_ascii=False))
