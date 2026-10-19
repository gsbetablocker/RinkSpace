"""Continue an exact accepted physical state under an RGB-verified source edit.

Old source values, J, cost and world caches are never installed. A mandatory
transition manifest authenticates every changed target and optional complete
operator change. Original pixels, IDs, clock, sigma, parameter chart and all
other source inputs must match the previous accepted source.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

SCHEMA = 'rinkspace.verified-source-identity-continuation.v1'
_BINDING_METADATA = {'source_revision', 'historical_summary', 'revision_role',
                     'preparation_status', 'historical_summary_is_current_verification'}
_IDENTITY_METADATA = {'target', '_target_key', 'source_identity_revision', 'original_model_target'}


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(ref):
    if reference(ref['path']) != {k:ref[k] for k in ('path', 'sha256')}:
        raise ValueError('Authenticated source transition reference changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))


def _digest(value):
    from .first_shot_source_value_replay import _hash_value
    h = hashlib.sha256()
    _hash_value(h, value)
    return h.hexdigest()


def _target(value):
    value = deepcopy(value)
    if value.get('kind') == 'board_edge' and 'association_policy' in value:
        value['association_policy'].pop('source_binding', None)
    return value


def _evidence(item):
    from .first_shot_revision import _evidence as existing_evidence
    existing_evidence(item)
    if any(item.get(k) is True for k in ('user_confirmed', 'human_confirmed', 'identity_confirmed_by_user')):
        raise ValueError('An agent RGB revision cannot claim human confirmation')


def validate_manifest(path):
    ref = reference(path)
    manifest = read(ref)
    if (manifest.get('schema') != SCHEMA or manifest.get('authority') != 'agent_rgb_review'
            or manifest.get('review_complete') is not True):
        raise ValueError('A completed explicitly RGB-verified transition manifest is required')
    if any(manifest.get(k) is True for k in ('user_confirmed', 'human_confirmed')):
        raise ValueError('The transition review is agent RGB evidence, not human confirmation')
    old = read(manifest['base_binding']);new = read(manifest['revised_binding'])
    before = read(manifest['base_association']);after = read(manifest['revised_association'])
    if before.get('source_binding') != manifest['base_binding'] or after.get('source_binding') != manifest['revised_binding']:
        raise ValueError('Source association does not bind the exact old/new source')
    old_policy,new_policy=deepcopy(before),deepcopy(after)
    old_policy.pop('source_binding',None);new_policy.pop('source_binding',None)
    if old_policy != new_policy:
        raise ValueError('Source identity correction cannot alter the physical board association policy')
    differences={key for key in set(old)|set(new) if old.get(key)!=new.get(key)}
    if not differences or not differences.issubset(_BINDING_METADATA) or 'source_revision' not in differences:
        raise ValueError('Binding differs beyond source identity revision and descriptive provenance: ' + str(sorted(differences)))
    if new['source_revision'] != manifest['source_revision']:
        raise ValueError('The new binding does not name the authenticated source revision')
    revision = read(manifest['source_revision'])
    from .first_shot_revision import SOURCE_SCHEMA
    if (revision.get('schema') != SOURCE_SCHEMA
            or revision.get('source_video_sha256') != old['source']['sha256']
            or revision.get('source_time_base') != old['source_clock']['time_base']
            or revision.get('original_source_UV_IDs_and_approvals_modified') is not False):
        raise ValueError('Existing source revision does not preserve the old native evidence')
    previous_revision=read(old['source_revision']) if old.get('source_revision') else {}
    for field in ('added_native_readouts','correlated_estimator_replacements'):
        if revision.get(field,[])!=previous_revision.get(field,[]):
            raise ValueError('Identity-only continuation cannot change original removals, additions or derived means: '+field)
    changes={}
    for item in manifest.get('verified_changes',[]):
        _evidence(item)
        key=int(item['native_pts']),str(item['source_point_id'])
        if key in changes or item['before_target']==item['after_target']:
            raise ValueError('Duplicate or ineffective verified identity change')
        changes[key]=item
    if not changes:
        raise ValueError('A source transition must contain explicit verified target changes')
    items={(int(item['native_pts']),str(item['source_point_id'])):item
           for item in revision.get('identity_corrections',[])}
    for key,item in changes.items():
        revised=items.get(key)
        if revised is None or _target(revised['target']) != _target(item['after_target']):
            raise ValueError('Verified target change is absent from the actual source revision')
        _evidence(revised)
        if (not np.array_equal(revised['original_UV'],item['original_UV'])
                or any(ref not in revised['evidence_references'] for ref in item['evidence_references'])):
            raise ValueError('Transition and source revision disagree on original UV/RGB proof')
    previous_items={(int(item['native_pts']),str(item['source_point_id'])):item
                    for item in previous_revision.get('identity_corrections',[])}
    if any(key not in items for key in previous_items):
        raise ValueError('A prior source identity correction was removed')
    for key,item in items.items():
        if previous_items.get(key)!=item and key not in changes:
            raise ValueError('An unreviewed identity item or prior evidence change entered the source revision')
    quarantines={}
    for item in manifest.get('verified_quarantines',[]):
        _evidence(item)
        key=int(item['native_pts']),str(item['source_point_id'])
        if (key in quarantines or key in changes or item.get('after_target') is not None
                or item.get('original_member_of_correlated_source_group') is not False):
            raise ValueError('Only distinct explicitly reviewed ordinary-point quarantines are allowed')
        quarantines[key]=item
    old_inactive=previous_revision.get('quarantined_correspondences',[])
    new_inactive=revision.get('quarantined_correspondences',[])
    if new_inactive[:len(old_inactive)]!=old_inactive:
        raise ValueError('Historical quarantine prefix/evidence changed')
    appended={(int(item['native_pts']),str(item['source_point_id'])):item
              for item in new_inactive[len(old_inactive):]}
    if len(appended)!=len(new_inactive)-len(old_inactive) or set(appended)!=set(quarantines):
        raise ValueError('New inactive scope differs from exact verified quarantines')
    for key,item in quarantines.items():
        actual=appended[key];_evidence(actual)
        if (not np.array_equal(actual['original_UV'],item['original_UV'])
                or _target(actual['original_target'])!=_target(item['before_target'])
                or any(ref not in actual['evidence_references'] for ref in item['evidence_references'])
                or actual.get('original_member_of_correlated_source_group') is not False):
            raise ValueError('Quarantine source revision differs from its exact native RGB proof')
    if manifest.get('verified_operator_changes'):
        raise ValueError('This identity/ordinary-quarantine transition cannot change correlated operators')
    operators={}
    for item in manifest.get('verified_operator_changes',[]):
        _evidence(item)
        key=int(item['native_pts']),str(item['group_id'])
        if key in operators:
            raise ValueError('Duplicate verified complete-operator change')
        operators[key]=item
    new_ops={(int(item['native_pts']),str(item['group_id'])):item
             for item in revision.get('correlated_operator_replacements',[])}
    previous_ops={(int(item['native_pts']),str(item['group_id'])):item
                  for item in previous_revision.get('correlated_operator_replacements',[])}
    if any(key not in new_ops for key in previous_ops):
        raise ValueError('A previous complete operator correction was removed')
    for key,item in new_ops.items():
        if previous_ops.get(key)!=item and key not in operators:
            raise ValueError('Unverified operator item or previous operator proof changed')
    for key,item in operators.items():
        actual=new_ops.get(key)
        if actual is None or any(ref not in actual['evidence_references'] for ref in item['evidence_references']):
            raise ValueError('Verified operator scope lacks the actual complete source-revision evidence')
        _evidence(actual)
    return ref,manifest,changes,operators,old,new,before,after,differences,quarantines


def _point(row,index):
    protected={key:deepcopy(value) for key,value in row.items() if key not in _IDENTITY_METADATA}
    vertices=row['original_vertex_indices']
    if len(vertices)==len(row['uv']):
        vertex_metadata=[vertices[index]]
    elif row.get('native_line_queries'):
        # Native replacement inherits the old representative lineage, while
        # its complete current query IDs/UV/metadata are already protected.
        vertex_metadata=deepcopy(vertices)
    else:
        raise ValueError('Original source vertex lineage does not match its points')
    protected.update(uv=np.asarray(row['uv'][index]).copy(),sigma=np.asarray(row['sigma'][index]).copy(),
        source_point_ids=[row['source_point_ids'][index]],original_vertex_indices=vertex_metadata)
    return protected


def compare_source_frames(old_frames,new_frames,changes,operators,quarantines=None):
    """Exact retained native/whitened row mapping; no camera/world evaluation."""
    from .first_shot_fit import FrozenWhitening
    if operators:raise ValueError('This source continuation preserves all correlated operators')
    quarantines={} if quarantines is None else quarantines
    used,used_quarantines=set(),set()
    old_count=new_count=old_formal_offset=0
    formal_indices=[];removed_formal=[];frame_receipts=[];quarantine_receipts=[]
    for old,new in zip(old_frames,new_frames,strict=True):
        for field in ('index','native_pts','source_packet'):
            if old[field]!=new[field]:raise ValueError('Source transition changed frame '+field)
        pts=int(old['native_pts'])
        old_points=[(row,i) for row in old['rows'] for i in range(len(row['uv']))]
        new_points=[(row,i) for row in new['rows'] for i in range(len(row['uv']))]
        if len(old_points)!=old['source_point_count'] or len(new_points)!=new['source_point_count']:
            raise ValueError('Source transition changed declared native point counts')
        grouped={int(i) for group in old['whitening_groups'] for i in group['point_indices']}
        keep=np.ones(len(old_points),bool)
        inactive={str(item['source_point_id']):item for item in new.get('inactive_source_correspondences',[])}
        old_inactive=old.get('inactive_source_correspondences',[])
        new_historical=[item for item in new.get('inactive_source_correspondences',[])
                        if (pts,str(item['source_point_id'])) not in quarantines]
        if _digest(old_inactive)!=_digest(new_historical):
            raise ValueError('Historical inactive source inventory/evidence changed')
        for index,(row,j) in enumerate(old_points):
            key=pts,str(row['source_point_ids'][j])
            item=quarantines.get(key)
            if item is None:continue
            if (index in grouped or _target(row['target'])!=_target(item['before_target'])
                    or not np.array_equal(row['uv'][j],item['original_UV'])):
                raise ValueError('Quarantine is grouped or differs from its exact original target/UV')
            archived=inactive.get(key[1])
            if (archived is None or not np.array_equal(archived['UV'],row['uv'][j])
                    or _target(archived['original_target'])!=_target(row['target'])
                    or float(archived['revision']['original_sigma'])!=float(row['sigma'][j])):
                raise ValueError('Quarantined source UV/ID/target/sigma was not preserved in inactive archive')
            keep[index]=False;used_quarantines.add(key)
            quarantine_receipts.append(dict(native_pts=pts,source_point_id=key[1],old_point_index=index,
                sigma_px=float(row['sigma'][j]),ordinary=True,scalar_rows_removed=2,
                original_UV_ID_target_sigma_and_evidence_archived=True))
        retained=np.flatnonzero(keep)
        if len(new_points)!=len(retained):
            raise ValueError('Source transition removed unverified points or added native points')
        target_changes=0
        for index,(b,j) in zip(retained,new_points,strict=True):
            a,i=old_points[int(index)]
            if _digest(_point(a,i))!=_digest(_point(b,j)):
                raise ValueError('Retained source UV/ID/sigma/clock/metadata/order changed at '+str(pts))
            key=pts,str(a['source_point_ids'][i]);aa,bb=_target(a['target']),_target(b['target'])
            if aa!=bb:
                item=changes.get(key)
                if (item is None or aa!=_target(item['before_target']) or bb!=_target(item['after_target'])
                        or not np.array_equal(a['uv'][i],item['original_UV'])):
                    raise ValueError('Unverified or mismatched source target difference: '+str(key))
                used.add(key);target_changes+=1
        point_map=np.full(len(old_points),-1,int);point_map[retained]=np.arange(len(retained))
        raw_keep=np.repeat(keep,2);raw_map=np.full(len(raw_keep),-1,int)
        raw_map[raw_keep]=np.arange(int(raw_keep.sum()))
        for a,b in zip(old['whitening_groups'],new['whitening_groups'],strict=True):
            indices=np.asarray(a['point_indices'],int)
            if (not keep[indices].all() or not np.array_equal(point_map[indices],b['point_indices'])
                    or _digest(a['operator'].to_record())!=_digest(b['operator'].to_record())):
                raise ValueError('A correlated W/operator/member order changed or was sliced')
            da={k:v for k,v in a.items() if k!='point_indices' and k!='operator'}
            db={k:v for k,v in b.items() if k!='point_indices' and k!='operator'}
            if _digest(da)!=_digest(db):raise ValueError('Retained correlated source evidence changed')
        ow,nw=FrozenWhitening(old),FrozenWhitening(new)
        if ow.ordinary_start!=nw.ordinary_start:
            raise ValueError('Correlated formal row inventory changed')
        ordinary_keep=raw_keep[ow.ordinary]
        if (not np.array_equal(raw_map[ow.ordinary[ordinary_keep]],nw.ordinary)
                or not np.array_equal(ow.ordinary_scales[ordinary_keep],nw.ordinary_scales)):
            raise ValueError('FrozenWhitening ordinary scalar positions/scales do not follow exact retained map')
        for (a,A,dest),(b,B,dest2) in zip(ow.groups,nw.groups,strict=True):
            if not np.array_equal(raw_map[a],b) or not np.array_equal(A,B) or dest!=dest2:
                raise ValueError('Actual FrozenWhitening correlated apply layout changed')
        rows=np.r_[np.arange(ow.ordinary_start),ow.ordinary_start+np.flatnonzero(ordinary_keep)].astype(int)
        if not np.array_equal(ow.scales[rows],nw.scales) or len(rows)!=nw.count:
            raise ValueError('Actual retained complete formal rows/scales changed')
        removed=np.setdiff1d(np.arange(ow.count),rows,assume_unique=True)
        if len(removed)!=2*int((~keep).sum()):
            raise ValueError('Only the reviewed ordinary two-coordinate rows may be removed')
        formal_indices.extend((old_formal_offset+rows).tolist())
        removed_formal.extend((old_formal_offset+removed).tolist())
        frame_receipts.append(dict(frame=int(old['index']),native_pts=pts,old_source_points=len(old_points),
            source_points=len(new_points),verified_target_changes=target_changes,
            ordinary_quarantines=int((~keep).sum()),old_formal_rows=ow.count,new_formal_rows=nw.count,
            correlated_W_values_and_member_order_byte_exact=True,actual_FrozenWhitening_retained_map_verified=True))
        old_formal_offset+=ow.count;old_count+=len(old_points);new_count+=len(new_points)
    if used!=set(changes) or used_quarantines!=set(quarantines):
        raise ValueError('Actual target/quarantine differences do not exactly equal the verified scope')
    formal_indices=np.asarray(formal_indices,int)
    return dict(native_frames=len(frame_receipts),old_source_points=old_count,source_points=new_count,
        verified_target_changes=len(used),verified_operator_changes=0,verified_ordinary_quarantines=len(used_quarantines),
        old_formal_rows=old_formal_offset,new_formal_rows=len(formal_indices),removed_formal_rows=removed_formal,
        retained_formal_indices_sha256=hashlib.sha256(formal_indices.tobytes()).hexdigest(),
        frames=frame_receipts,quarantines=quarantine_receipts,
        retained_UV_ID_sigma_clock_and_nonidentity_source_inputs_byte_exact=True,
        all_correlated_whitening_operators_unchanged_no_rows_or_columns_sliced=True,
        _retained_formal_indices=formal_indices)


def load_source_frames(binding_ref,policy):
    """Use the existing data pipeline without an old physical state/bundle."""
    from .first_shot_fit import _load_frames
    binding=read(binding_ref)
    frames,_=_load_frames(binding_ref,policy)
    for field,module,method in (
        ('source_revision','first_shot_revision','revise_frames'),
        ('source_localization_noise_revision','first_shot_noise_revision','revise_localization_noise'),
        ('source_native_line_revision','first_shot_native_line_revision','revise_native_lines')):
        if binding.get(field) is not None:
            import importlib
            fn=getattr(importlib.import_module('.'+module,__package__),method)
            result=fn(frames,binding[field],binding,policy=policy) if field=='source_revision' else fn(frames,binding[field],binding)
            frames=result[0]
    return frames


def checkpoint_reference_verifier(pool_transition):
    """Authenticate file identities while retaining enriched provenance fields.

    A reference may include authority and source metadata in addition to its
    exact path/SHA. The original verifier owns all byte/producer checks; only
    the two file-identity fields are passed to its equality comparison.
    """
    from .first_shot_resume_cache import CompleteSourceReplay
    verifier=object.__new__(CompleteSourceReplay)
    verifier.pool_transition=Path(pool_transition);verifier.transitions={};verifier.verified={}
    original=verifier.verify_reference
    def verify_reference(value):
        return original({key:value[key] for key in ('path','sha256')})
    verifier.verify_reference=verify_reference
    return verifier


class RevisedSourceStart:
    """Authenticate one accepted full q, then demand a fresh new-source r/J."""
    def __init__(self,adapter,*,transition_manifest,checkpoint,prior_report,pool_transition,**unused):
        from .first_shot_resume_cache import CompleteSourceReplay
        self.adapter=adapter
        ref,manifest,changes,operators,old,new,before,after,differences,quarantines=validate_manifest(transition_manifest)
        if (adapter.bundle.state.binding_reference!=manifest['revised_binding']
                or adapter.association_reference!=manifest['revised_association']):
            raise ValueError('Actual new adapter differs from verified transition binding/association')
        verifier=checkpoint_reference_verifier(pool_transition)
        report_ref,checkpoint_ref=reference(prior_report),reference(checkpoint)
        report=read(report_ref);verifier.verify_references(report)
        with np.load(checkpoint,allow_pickle=False) as saved:
            identity=json.loads(str(saved['identity_json']));verifier.verify_references(identity)
            q=saved['q'].copy()
            if (q.shape!=(7674,) or not np.array_equal(q,adapter.start)
                    or identity['binding']!=manifest['base_binding']
                    or identity['association']!=manifest['base_association']):
                raise ValueError('Accepted full state does not belong to verified old source basis')
            for name,value in adapter.checkpoint_arrays().items():
                if not np.array_equal(saved['aux__'+name],value):raise ValueError('Source transition changed chart/aux '+name)
            old_scales,old_mask=saved['scales'].copy(),saved['quadratic'].copy()
            self.damping=float(saved['damping']);self.fraction=float(saved['next_line_search_fraction'])
        token=hashlib.sha256(q.tobytes()).hexdigest()
        if report['qaug_sha256']!=token or report['binding']!=manifest['base_binding']:
            raise ValueError('Previous reviewed state differs from the accepted canonical state')
        if (adapter.seed_receipt['initial_damping']!=self.damping
                or adapter.seed_receipt['initial_line_search_fraction']!=self.fraction):
            raise ValueError('Source transition changed checkpoint damping/fraction')
        old_frames=getattr(adapter.bundle,'_authenticated_previous_frames',None)
        if old_frames is None:old_frames=load_source_frames(manifest['base_binding'],before)
        comparison=compare_source_frames(old_frames,adapter.bundle.frames,changes,operators,quarantines)
        formal_indices=comparison.pop('_retained_formal_indices')
        old_ids=np.asarray([pid for f in old_frames for r in f['rows'] for pid in r['source_point_ids']])
        old_pts=np.asarray([f['native_pts'] for f in old_frames for r in f['rows'] for pid in r['source_point_ids']],np.int64)
        with np.load(report['snapshot']['path'],allow_pickle=False)as snapshot:
            for name,value in adapter.complete_state_payload(q).items():
                if not np.array_equal(snapshot[name],value):raise ValueError('Previous reviewed snapshot changed complete state '+name)
            if not np.array_equal(snapshot['source_point_ids'],old_ids) or not np.array_equal(snapshot['native_pts_per_point'],old_pts):
                raise ValueError('Previous raw snapshot does not cover the exact old native IDs/PTS')
        if (len(old_mask)!=comparison['old_formal_rows']
                or not np.array_equal(old_mask[formal_indices],adapter.quadratic)
                or not np.array_equal(old_scales[formal_indices],adapter.scales)):
            raise ValueError('Source transition changed formal masks/scales beyond exact ordinary quarantine rows')
        if hasattr(adapter.bundle,'_authenticated_previous_frames'):
            del adapter.bundle._authenticated_previous_frames
        self.q=np.empty(0)
        self.receipt=dict(schema='rinkspace.verified-source-revision-fresh-full-q-start.v1',
            producer=reference(__file__),transition_manifest=ref,checkpoint=checkpoint_ref,prior_report=report_ref,
            old_binding=manifest['base_binding'],new_binding=manifest['revised_binding'],
            old_source_basis_prior_cost=report['source_objective_cost'],old_new_costs_are_different_objectives=True,
            cross_basis_cost_reduction_not_claimed=True,qaug_sha256=token,parameter_count=len(q),
            initial_source_J_reuse=False,initial_source_r_reuse=False,initial_raw_reuse=False,
            cached_world_or_J_loaded=False,all_complete_q_aux_and_chart_values_preserved=True,
            source_identity_review_authority='agent_rgb_review',human_confirmation_claimed=False,
            complete_source_transition_comparison=comparison,binding_metadata_differences=sorted(differences),
            initial_damping=self.damping,initial_line_search_fraction=self.fraction,
            actual_checkpoint_damping_and_fraction_preserved=True,retained_source_whitening_operators_identical=True,
            only_explicit_ordinary_quarantine_scalar_rows_removed=True,
            enriched_reference_metadata_preserved_file_identity_path_and_sha_exact=True)
        self._new_source_ref=ref

    def install(self):
        owner=self.adapter;previous=owner.checkpoint_identity;receipt=self.receipt
        def identity(adapter):
            result=previous();result['verified_source_revision_continuation']=deepcopy(receipt)
            result['initial_exact_complete_current_state_full_J_reuse']=False
            return result
        owner.checkpoint_identity=MethodType(identity,owner)
        owner._initial_complete_J_reuse=False
        owner.source_basis_transition_receipt=receipt
        for ref in (reference(__file__),self._new_source_ref):
            if ref not in owner.source_producers:owner.source_producers.append(ref)
        return self
