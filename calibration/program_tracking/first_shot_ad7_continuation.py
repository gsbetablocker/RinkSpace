"""Data-only grouped-parent continuation proposal; no runtime installation.

This deliberately does not call the frozen H8 initializer or mutate its global
registry.  It authenticates an existing grouped parent's archived model and
reuses its mathematical class through an explicitly named subclass.  The
optional text proposal is an asset initializer, never an admitted observation.
No returned document is fit-ready; a real completed accepted parent, public
adapter and current-owner checks are still required before future adoption.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.first_shot_source_transition import reference, read
from calibration.program_tracking import first_shot_ad7_shape_chart as legacy

SCHEMA = 'rinkspace.grouped-parent-continuation-proposal.v2'
OLD_SCHEMA = 'rinkspace.grouped-parent-continuation-proposal.v1'
RUNTIME_SCHEMAS = ('rinkspace.grouped-parent-seed-successor-runtime.v1',
                   'rinkspace.grouped-parent-seed-successor-runtime.v2')
TEXT_SCHEMA = 'rinkspace.source-conditioned-ad7-text-initializer-proposal.v1'
FIXED_LOCAL = np.arange(3, 8)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def digest(a):
    a = np.asarray(a)
    if a.dtype.hasobject:
        raise ValueError('Object arrays are not a complete checkpoint contract')
    return dict(dtype=a.dtype.str, shape=list(a.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def authenticated(ref):
    if not isinstance(ref, dict) or reference(ref['path']) != ref:
        raise ValueError('Immutable model/evidence reference changed')
    return read(ref)


def parent_contract(parent):
    for key in ('q', 'physical_q', 'expanded_q'):
        value = np.asarray(parent[key])
        if value.dtype != np.float64 or value.shape != (7700,) or not np.isfinite(value).all():
            raise ValueError('Complete finite 7700-value parent array required: '+key)
    if not parent['aux'] or any(np.asarray(v).dtype.hasobject for v in parent['aux'].values()):
        raise ValueError('Every non-object parent auxiliary must be retained')
    damping, fraction = float(parent['damping']), float(parent['fraction'])
    if not np.isfinite(damping) or damping < 0 or not 0 < fraction <= 1:
        raise ValueError('Final parent numerical search state required')
    return dict(arrays={k:digest(parent[k]) for k in ('q', 'physical_q', 'expanded_q')},
                aux={k:digest(v) for k,v in sorted(parent['aux'].items())},
                damping=damping, fraction=fraction,
                damping_bits=digest(np.asarray(damping)), fraction_bits=digest(np.asarray(fraction)),
                checkpoint=deepcopy(parent['checkpoint']))


def restore_parent_chart(parent, *, allow_unsealed_protocol_ancestry=False):
    """Authenticate live identity plus its original, untouched aux archive."""
    contract = parent_contract(parent)
    identity = parent['identity']
    receipt = identity.get('ad7_grouped_shape_chart')
    key = identity.get('ad7_grouped_aux_key')
    if not isinstance(receipt, dict) or key not in parent['aux']:
        raise ValueError('A grouped parent needs its explicit live model and archived receipt')
    archived = json.loads(str(np.asarray(parent['aux'][key])))
    if archived != receipt:
        raise ValueError('Grouped identity and original auxiliary archive differ')
    ref = receipt['configuration']; config = authenticated(ref)
    if (config.get('schema') != legacy.SCHEMA or receipt.get('schema') != legacy.SCHEMA
            or config.get('producer') != reference(legacy.__file__)
            or receipt.get('producer') != config['producer']):
        raise ValueError('This prototype restores the authenticated frozen v1 chart only')
    if key != 'ad7_groups_'+ref['sha256'][:16]+'_json':
        raise ValueError('Original chart auxiliary key is not its authenticated hash key')
    for name in ('material', 'source_support_reference'):
        authenticated(config[name])
    support = authenticated(config['source_support_reference'])
    if (support.get('passed') is not True or support.get('residual_selection') is not False
            or support.get('model') != config['material']
            or config['assigned_source_points_by_group'] != {'logo':2192, 'top':0, 'bottom':0}):
        raise ValueError('Unchanged complete logo-only source contract required for this proposal')
    if reference(support['inventory']['path']) != support['inventory']:
        raise ValueError('Original formal source inventory changed')
    indices = np.asarray(config['parent_global_slots'], int)
    if not np.array_equal(indices, np.arange(5159,5167)):
        raise ValueError('Explicit unchanged eight canonical chart slots required')
    if not same(parent['q'][indices], parent['physical_q'][indices]):
        raise ValueError('Parent derived chart is not identity on the Ad7 coordinates')
    fixed = identity.get('ad7_initialization_only_fixed')
    fixed_key = 'ad7_initfixed_'+ref['sha256'][:16]+'_json'
    if (not isinstance(fixed,dict) or fixed_key not in parent['aux']
            or json.loads(str(np.asarray(parent['aux'][fixed_key]))) != fixed
            or fixed.get('configuration') != ref
            or fixed.get('global_slots') != indices[FIXED_LOCAL].tolist()
            or fixed.get('parameters') != list(legacy.PARAMETERS[3:])
            or fixed.get('not_fitted_by_loss_or_QP') is not True):
        raise ValueError('Complete original fixed-text contract and archive required')
    if (not same(parent['q'][indices[FIXED_LOCAL]], np.asarray(fixed['fixed_canonical_values'],float))
            or not same(parent['q'][indices[FIXED_LOCAL]], np.asarray(config['parent_slot_anchor'],float)[FIXED_LOCAL])):
        raise ValueError('Source-unsupported text coordinates changed in the parent')
    result = dict(configuration=deepcopy(config), reference=deepcopy(ref),
                receipt=deepcopy(receipt), archive_key=key, fixed_archive_key=fixed_key,
                fixed_receipt=deepcopy(fixed), contract=contract,
                effective_seed8=np.asarray(config['physical_seed8'],float).copy(),
                current_runtime_reference=None,current_archive_key=None,current_fixed_archive_key=None,
                runtime_chain=[])
    current,live,current_fixed=_current_identity(parent,ref)
    if live is None:
        return result
    expected_key='ad7_cont_'+current['sha256'][:16]+'_json'
    seed,chain=_runtime_seed(current,parent,result,set(),allow_unsealed_protocol_ancestry,
        expected_receipt=live,expected_fixed=current_fixed)
    result.update(effective_seed8=seed,current_runtime_reference=deepcopy(current),
        current_archive_key=expected_key,current_fixed_archive_key='ad7_contfix_'+current['sha256'][:16]+'_json',
        runtime_chain=chain)
    return result


def _archive(parent,key):
    if key not in parent['aux']:
        raise ValueError('Missing ancestor model archive: '+key)
    return json.loads(str(np.asarray(parent['aux'][key])))


def _current_identity(parent,root_reference):
    """A missing current label may never turn existing runtime archives into H8."""
    identity=parent['identity'];current=identity.get('ad7_current_physical_model')
    live=identity.get('ad7_grouped_continuation');fixed=identity.get('ad7_continuation_fixed')
    key=identity.get('ad7_continuation_aux_key')
    if live is None:
        orphan=key is not None or any(k.startswith(('ad7_cont_','ad7_contfix_'))for k in parent['aux'])
        if current not in (None,root_reference) or fixed is not None or orphan:
            raise ValueError('Missing current identity cannot hide existing runtime archives')
        return None,None,None
    if (not isinstance(live,dict) or not isinstance(current,dict)
            or not isinstance(fixed,dict) or live.get('configuration')!=current
            or fixed.get('configuration')!=current):
        raise ValueError('Complete live runtime and fixed identities must name the same current model')
    expected='ad7_cont_'+current['sha256'][:16]+'_json'
    if key!=expected or _archive(parent,expected)!=live:
        raise ValueError('Current runtime identity and archive key/value differ')
    if _archive(parent,'ad7_contfix_'+current['sha256'][:16]+'_json')!=fixed:
        raise ValueError('Current fixed identity and archive differ')
    return deepcopy(current),deepcopy(live),deepcopy(fixed)


def _installation_parent_current(proposal,ancestor,root_reference):
    current,live,fixed=_current_identity(ancestor,root_reference)
    if proposal.get('restored_current_runtime')!=current:
        raise ValueError('Proposal skips or changes the actual current model of its own installation parent')
    return current,live,fixed


def _producer(ref):
    if not isinstance(ref,dict) or reference(ref['path'])!=ref:
        raise ValueError('Historical model producer changed')


def _runtime_seed(ref,parent,root,seen,allow_protocol,*,expected_receipt=None,expected_fixed=None):
    """Read authenticated current/ancestor contracts without executing old code.

    Old proposal.parent is verified against ITS installation parent, never
    against the later accepted q. Every ancestor auxiliary stays in the new
    complete parent; no historical identity or proposal is rewritten.
    """
    identity=(ref['path'],ref['sha256'])
    if identity in seen:raise ValueError('Grouped runtime ancestry contains a cycle')
    seen=seen|{identity}
    config=authenticated(ref)
    receipt=_archive(parent,'ad7_cont_'+ref['sha256'][:16]+'_json')
    fixed=_archive(parent,'ad7_contfix_'+ref['sha256'][:16]+'_json')
    if (expected_receipt is not None and receipt!=expected_receipt) or (expected_fixed is not None and fixed!=expected_fixed):
        raise ValueError('Current runtime identity and its original archives differ')
    if (config.get('schema') not in RUNTIME_SCHEMAS or receipt.get('schema')!=config['schema']
            or receipt.get('configuration')!=ref or receipt.get('producer')!=config.get('producer')
            or receipt.get('continuation')!=config.get('continuation')
            or receipt.get('original_grouped_configuration')!=root['reference']
            or config.get('restored_configuration')!=root['reference']
            or receipt.get('material')!=root['configuration']['material']
            or config.get('material')!=root['configuration']['material']
            or config.get('legacy_math_producer')!=root['configuration']['producer']
            or receipt.get('entry')!=config.get('entry')):
        raise ValueError('Current runtime does not describe the authenticated original grouped model')
    for key in ('producer','proposal_producer','legacy_math_producer','entry'):_producer(config[key])
    proposal=authenticated(config['continuation'])
    if (proposal.get('schema') not in (OLD_SCHEMA,SCHEMA)
            or proposal.get('producer')!=config['proposal_producer']
            or proposal.get('restored_configuration')!=root['reference']
            or proposal.get('material')!=root['configuration']['material']
            or proposal.get('source_support_reference')!=root['configuration']['source_support_reference']
            or proposal.get('restored_aux_key')!=root['archive_key']
            or proposal.get('restored_fixed_aux_key')!=root['fixed_archive_key']
            or proposal.get('parent_global_slots')!=root['configuration']['parent_global_slots']
            or proposal.get('parameter_names')!=list(legacy.PARAMETERS)
            or not same(np.asarray(proposal['original_anchor8'],float),np.asarray(root['configuration']['parent_slot_anchor'],float))):
        raise ValueError('Ancestor proposal changed its original chart basis')
    policy=proposal['source_free_fixed_policy']
    if (policy.get('fixed_global_slots')!=list(range(5162,5167)) or policy.get('shared_count')!=167
            or policy.get('assigned_source_points_by_group')!={'logo':2192,'top':0,'bottom':0}):
        raise ValueError('Current model changed the declared fixed-text/source contract')
    # Preserve every ancestor aux fingerprint, even when latest accepted q is
    # different from the initial q stored in that model's old proposal.
    for key,d in proposal['parent']['aux'].items():
        if key not in parent['aux'] or digest(parent['aux'][key])!=d:
            raise ValueError('A complete ancestor auxiliary was lost or changed: '+key)
    scope=config.get('installation_scope');bound_prior=None
    if scope=='unsealed_data_protocol':
        if not allow_protocol:raise ValueError('Unsealed current ancestry is only a synthetic protocol fixture')
    elif scope=='completed_parent_pending_actual_startup_QA':
        if proposal.get('allow_unsealed_protocol_ancestry'):
            raise ValueError('A completed model cannot inherit synthetic ancestry opt-in')
        from calibration.program_tracking.first_shot_time_successor import read_accepted_parent
        proof=authenticated(config['completed_parent_binding']);refs=config['complete_parent_references']
        if (proof.get('passed') is not True or proof.get('complete_parent_references')!=refs
                or proof.get('all_observed_owner_launcher_and_workers_absent') is not True
                or proof.get('original_owner_normal_pool_shutdown') is not True
                or proof.get('no_process_terminated') is not True):
            raise ValueError('Ancestor runtime has no authenticated normal final-parent binding')
        ancestor=read_accepted_parent(refs)
        if (parent_contract(ancestor)!=proposal['parent'] or
                hashlib.sha256(json.dumps(ancestor['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=proposal['parent_identity_sha256']):
            raise ValueError('Old proposal does not bind its own historical installation parent')
        bound_prior=_installation_parent_current(proposal,ancestor,root['reference'])
    else:raise ValueError('Unsupported ancestor runtime installation scope')
    prior=proposal.get('restored_current_runtime')
    baseline=np.asarray(root['configuration']['physical_seed8'],float).copy();chain=[]
    if prior is not None:
        baseline,chain=_runtime_seed(prior,parent,root,seen,allow_protocol,
            expected_receipt=None if bound_prior is None else bound_prior[1],
            expected_fixed=None if bound_prior is None else bound_prior[2])
    if proposal['schema']==SCHEMA and not same(np.asarray(proposal['restored_current_seed8'],float),baseline):
        raise ValueError('Ancestor proposal baseline differs from its previous live seed')
    seed=baseline.copy()
    if proposal['initializer'] is not None:seed,_=_candidate(proposal['initializer'],root)
    if proposal['schema']==SCHEMA and prior is not None and not same(seed,baseline):
        raise ValueError('Root-only initializer cannot overwrite a different current-conditioned seed')
    if not same(np.asarray(proposal['evaluation_seed8'],float),seed):
        raise ValueError('Current seed was not inherited or explicitly initialized as declared')
    columns=np.asarray(root['configuration']['parent_global_slots'],int)
    if (fixed.get('configuration')!=ref or fixed.get('global_slots')!=columns[FIXED_LOCAL].tolist()
            or fixed.get('parameters')!=list(legacy.PARAMETERS[3:]) or fixed.get('shared_count')!=167
            or fixed.get('not_fitted_by_loss_or_QP') is not True
            or not same(np.asarray(fixed['fixed_canonical_values'],float),parent['q'][columns[FIXED_LOCAL]])
            or not same(np.asarray(fixed['physical_initialization_parameters'],float),seed[FIXED_LOCAL])):
        raise ValueError('Current fixed-text receipt does not match the live physical initializer')
    return seed,[dict(runtime=deepcopy(ref),continuation=deepcopy(config['continuation']),
                      seed8_sha256=digest(seed)['sha256'])]+chain


def _candidate(reference_value, restored):
    proposal = authenticated(reference_value); old = restored['configuration']
    if (proposal.get('schema') != TEXT_SCHEMA
            or proposal.get('status') != 'not_installed_not_formal_source_fit_not_accepted'
            or proposal.get('base_chart_configuration') != restored['reference']
            or proposal.get('base_chart_runtime') != old['producer']
            or proposal.get('logo_frozen') is not True
            or proposal.get('parameter_names') != list(legacy.PARAMETERS)
            or proposal.get('adjusted_parameters') != list(legacy.PARAMETERS[3:])):
        raise ValueError('Explicit same-base source-conditioned text initializer required')
    for key in ('original_physical_seed8','parent_slot_anchor','parent_global_slots'):
        expected = old[{'original_physical_seed8':'physical_seed8'}.get(key,key)]
        if not same(np.asarray(proposal[key]),np.asarray(expected)):
            raise ValueError('Initializer was made for a different original chart: '+key)
    seed = np.asarray(proposal['proposed_physical_seed8'],float)
    if seed.shape != (8,) or not np.isfinite(seed).all() or not same(seed[:3],np.asarray(old['physical_seed8'],float)[:3]):
        raise ValueError('Text initializer must leave all original logo seed bits unchanged')
    objective = proposal['objective']
    if (objective.get('W_constructed') is not False or objective.get('formal_W_changed') is not False
            or objective.get('independent_observation_count_claimed') is not False):
        raise ValueError('Initializer may not invent statistical weighting or formal observations')
    candidates = authenticated(proposal['source_native_candidates'])
    if not candidates.get('features') or any(f.get('allowed_for_fit') is not False for f in candidates['features']):
        raise ValueError('Every initializer candidate must remain excluded from formal fit')
    for key in ('producer','admission_protocol','noise_review','original_q1_scene',
                'source_camera_safe_packet','source_camera_model','raw_residuals','failed_wrong_edge_trial'):
        ref = proposal[key]
        if reference(ref['path']) != ref:
            raise ValueError('Initializer provenance changed: '+key)
    return seed, proposal


def build_configuration(parent, *, initializer_reference=None, fixture_kind='unsealed_parent_protocol',
                        allow_unsealed_protocol_ancestry=False):
    """Return an unsealed proposal without changing q, aux or frozen config.

    A future public adapter must implement this contract explicitly.  No
    frozen install_execution/install_successor/verify function is bypassed or
    called with a forged legacy configuration here.
    """
    restored = restore_parent_chart(parent,allow_unsealed_protocol_ancestry=allow_unsealed_protocol_ancestry)
    old = restored['configuration']
    seed = restored['effective_seed8'].copy(); candidate = None
    if initializer_reference is not None:
        seed,candidate = _candidate(initializer_reference,restored)
        if restored['current_runtime_reference'] is not None and not same(seed,restored['effective_seed8']):
            raise ValueError('A new text update needs a new explicitly current-conditioned initializer; the old root-only candidate cannot overwrite a different live seed')
    return dict(schema=SCHEMA,producer=reference(__file__),
        status='data_only_pending_actual_accepted_parent_and_public_adapter',allowed_for_fit=False,
        fixture_kind=fixture_kind,mode='restore_existing_grouped_chart' if candidate is None else 'restore_and_update_fixed_text_initializer',
        restored_configuration=restored['reference'],restored_aux_key=restored['archive_key'],
        restored_fixed_aux_key=restored['fixed_archive_key'],parent=restored['contract'],
        restored_current_runtime=deepcopy(restored['current_runtime_reference']),
        restored_current_aux_key=deepcopy(restored['current_archive_key']),
        restored_current_fixed_aux_key=deepcopy(restored['current_fixed_archive_key']),
        restored_current_seed8=restored['effective_seed8'].tolist(),
        restored_runtime_chain=deepcopy(restored['runtime_chain']),
        allow_unsealed_protocol_ancestry=bool(allow_unsealed_protocol_ancestry),
        parent_identity_sha256=hashlib.sha256(json.dumps(parent['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        material=deepcopy(old['material']),source_support_reference=deepcopy(old['source_support_reference']),
        parent_global_slots=deepcopy(old['parent_global_slots']),parameter_names=list(legacy.PARAMETERS),
        original_anchor8=deepcopy(old['parent_slot_anchor']),evaluation_seed8=seed.tolist(),
        initializer=deepcopy(initializer_reference),
        initializer_is_source_conditioned_asset_candidate=candidate is not None,
        initializer_is_formal_source=False,formal_source_or_W_changed=False,
        source_free_fixed_policy=dict(fixed_global_slots=deepcopy(old['fixed_initialization_global_slots']),
            assigned_source_points_by_group=deepcopy(old['assigned_source_points_by_group']),shared_count=167),
        preserved=dict(all_canonical_q=True,all_parent_aux=True,expanded_native_time=True,physical_parameter_array=True,
            B4_anchor_and_nonzero_details=True,damping_and_fraction=True,logo_original_anchor_seed_and_arithmetic=True),
        changed_predictions='none' if candidate is None else 'only top/bottom material paths; no formal-source text support',
        model_J_reuse_allowed=False,whole_shot_accuracy_verified=False,
        deferred=['normal-completed accepted grouped parent binding','explicit new public execution/stage/reviewer/worker adapter',
                  'actual complete-parent mapping and 1248x5 row-camera byte checks','actual whole-source current residual/J contract and FD',
                  'actual full old physical guard plus all 15 current grouped margins','whole-shot real heldout validation'])


def verify_configuration(configuration,parent):
    if configuration.get('schema') != SCHEMA or configuration.get('producer') != reference(__file__):
        raise ValueError('Explicit unchanged continuation prototype producer required')
    expected=build_configuration(parent,initializer_reference=configuration['initializer'],fixture_kind=configuration['fixture_kind'],
        allow_unsealed_protocol_ancestry=configuration['allow_unsealed_protocol_ancestry'])
    if configuration != expected:
        raise ValueError('Continuation proposal or complete parent contract changed')
    return dict(passed=True,scope='pure-array grouped-parent restoration/proposal contract',
                accepted_parent_or_fit_ready_claimed=False,all_parent_arrays_aux_and_numerics_bound=True)


class ContinuationChart(legacy.Ad7GroupedShapeChart):
    """Use frozen mathematical methods under explicit new proposal identity."""
    def __init__(self,configuration,parent):
        verify_configuration(configuration,parent)
        restored=restore_parent_chart(parent,allow_unsealed_protocol_ancestry=configuration['allow_unsealed_protocol_ancestry'])
        super().__init__(restored['configuration'])
        self.seed=np.asarray(configuration['evaluation_seed8'],float).copy()
        self.configuration=deepcopy(configuration)
        # Crucially keep all original anchor bits, including accepted free logo
        # increments: evaluating at the new parent must not erase those values.
        if not same(self.anchor,np.asarray(configuration['original_anchor8'],float)):
            raise ValueError('Original grouped arithmetic anchor changed')


def clone_parent_unchanged(parent):
    """Small-array continuation payload; never rewrites an existing archive."""
    cloned=deepcopy(parent)
    if parent_contract(cloned) != parent_contract(parent):
        raise ValueError('Complete parent clone changed bytes or numerical state')
    return cloned
