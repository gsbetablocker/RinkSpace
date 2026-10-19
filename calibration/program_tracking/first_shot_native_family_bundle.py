"""Verified native-family source delta after the complete authenticated v22 chain."""
from collections import OrderedDict
from copy import deepcopy
import hashlib,json
from types import MethodType
import numpy as np
from .first_shot_source_transition import reference,read,_digest,_target,compare_source_frames
from .first_shot_native_family_revision import rebase_family_candidate,replace_native_family,REVIEW_SCHEMA

SCHEMA='rinkspace.verified-native-finite-curve-family-transition.v1'
FIELD='post_native_source_family_revision'
_CONFIG=None


def read_transition(path):
    ref=reference(path);manifest=read(ref)
    if (manifest.get('schema')!=SCHEMA or manifest.get('review_complete')is not True
            or manifest.get('authority')!='agent_rgb_review' or manifest.get('user_confirmed')is not False):
        raise ValueError('Completed independent RGB native-family transition required')
    old=read(manifest['base_binding']);new=read(manifest['revised_binding'])
    before=read(manifest['base_association']);after=read(manifest['revised_association'])
    from .first_shot_group_rebuild import read_transition as original_transition
    _,previous,*_=original_transition(manifest['authenticated_v22_transition']['path'])
    if (previous['revised_binding']!=manifest['base_binding']
            or previous['revised_association']!=manifest['base_association']):
        raise ValueError('Native families must follow the exact authenticated v22 source chain')
    if {k for k in set(old)|set(new)if old.get(k)!=new.get(k)}!={FIELD}:
        raise ValueError('Native family binding changed outside its single source revision field')
    if (new[FIELD]!=manifest['family_revision'] or before.get('source_binding')!=manifest['base_binding']
            or after.get('source_binding')!=manifest['revised_binding']):
        raise ValueError('Native family source binding/association differs')
    a,b=deepcopy(before),deepcopy(after);a.pop('source_binding');b.pop('source_binding')
    if a!=b:raise ValueError('Native source correction changed the physical board association')
    revision=read(manifest['family_revision'])
    if (revision.get('schema')!='rinkspace.reviewed-native-finite-curve-family-revision.v1'
            or revision.get('base_binding')!=manifest['base_binding'] or not revision.get('groups')):
        raise ValueError('Native family revision has another base source')
    groups={};checked=set()
    for entry in revision['groups']:
        item=read(entry['candidate']);review=read(entry['identity_review'])
        if (item['base_binding']!=manifest['base_binding'] or item['frame']in groups or item.get('house_id')!='F2'
                or review.get('schema')!=REVIEW_SCHEMA
                or review.get('all_original_native_queries_independently_reviewed')is not True
                or review.get('world_camera_or_fit_residual_used_for_identity')is not False
                or review.get('native_support_sha256')!=item['native_support_sha256']
                or len(review.get('queries',[]))!=len(item['native_inventory'])):
            raise ValueError('Every original native query requires its exact completed RGB scope')
        for original,decision in zip(item['native_inventory'],review['queries'],strict=True):
            if (decision.get('native_inventory_entry')!=original or decision.get('authority')!='agent_rgb_review'
                    or decision.get('user_confirmed')is not False or decision.get('actually_viewed_original_RGB')is not True
                    or decision.get('disposition')not in ('retained','excluded') or not decision.get('reason')
                    or not decision.get('evidence_references')):
                raise ValueError('Incomplete or inexact native query RGB disposition')
            keep=decision['disposition']=='retained'
            if (decision.get('admit_to_fit')is not keep or keep and
                    (decision.get('independent_static_material_identity_confirmed')is not True
                     or decision.get('outside_person_broadcast_and_unconfirmed_material')is not True)):
                raise ValueError('Unconfirmed native material cannot enter source fitting')
            for evidence in decision['evidence_references']:
                key=evidence['path'],evidence['sha256']
                if key in checked:continue
                if reference(evidence['path'])!={k:evidence[k]for k in ('path','sha256')}:
                    raise ValueError('Native RGB evidence bytes changed')
                checked.add(key)
        groups[item['frame']]=dict(entry=entry,item=item,review=review)
    if set(groups)!={841,866,876}:raise ValueError('Only the three explicitly reviewed F2 families may change')
    return ref,manifest,revision,old,new,before,after,groups


def _record(frame):
    from .first_shot_fit import _target_key
    rows=[]
    for row in frame['rows']:
        if row['_target_key']!=_target_key(row['target']):raise ValueError('Source target key differs from its full descriptor')
        rows.append(dict({k:v for k,v in row.items()if k!='_target_key'},target=_target(row['target'])))
    return dict(frame,rows=rows,whitening_groups=[dict(g,operator=g['operator'].to_record())for g in frame['whitening_groups']])


def compare_transition(old_frames,new_frames,expected,proofs):
    from .first_shot_fit import FrozenWhitening
    normalized=[];old_offsets=[];new_offsets=[];old_cursor=new_cursor=0;mapping=[]
    for old,new in zip(old_frames,new_frames,strict=True):
        index=old['index'];ow,nw=FrozenWhitening(old),FrozenWhitening(new)
        if index in expected:
            if _digest(_record(new))!=_digest(_record(expected[index])):
                raise ValueError('Native family changed beyond exact deterministic whole-group replacement: '+str(index))
            normalized.append(old)
            for block in proofs[index]['formal_blocks']:
                if not block['replaced']:
                    a,b=block['old_local_formal_slice'];c,d=block['new_local_formal_slice']
                    old_offsets.extend(range(old_cursor+a,old_cursor+b));new_offsets.extend(range(new_cursor+c,new_cursor+d))
            old_offsets.extend(range(old_cursor+ow.ordinary_start,old_cursor+ow.count))
            new_offsets.extend(range(new_cursor+nw.ordinary_start,new_cursor+nw.count))
        else:
            normalized.append(new);old_offsets.extend(range(old_cursor,old_cursor+ow.count));new_offsets.extend(range(new_cursor,new_cursor+nw.count))
        mapping.append(dict(frame=index,native_pts=old['native_pts'],old_raw_points=old['source_point_count'],
            new_raw_points=new['source_point_count'],old_formal_start=old_cursor,new_formal_start=new_cursor,
            old_formal_rows=ow.count,new_formal_rows=nw.count,replaced=index in expected))
        old_cursor+=ow.count;new_cursor+=nw.count
    common=compare_source_frames(old_frames,normalized,{},{});common.pop('_retained_formal_indices');common.pop('frames')
    return dict(native_frames=len(mapping),old_source_points=sum(f['source_point_count']for f in old_frames),
        source_points=sum(f['source_point_count']for f in new_frames),old_formal_rows=old_cursor,new_formal_rows=new_cursor,
        exactly_reviewed_whole_families=len(expected),all_other_source_UV_ID_sigma_W_clock_target_metadata_exact=True,
        no_old_mean_and_new_raw_double_count=True,old_W_not_sliced=True,whole_frame_mapping=mapping,
        original_common_scope=common,_old_unchanged_formal_indices=np.asarray(old_offsets,int),
        _new_unchanged_formal_indices=np.asarray(new_offsets,int))


def install_native_family_bundle(path):
    global _CONFIG
    if _CONFIG is not None:raise ValueError('Native family source pipeline already installed')
    data=read_transition(path)
    from .first_shot_group_rebuild import install_group_rebuilt_bundle
    install_group_rebuilt_bundle(data[1]['authenticated_v22_transition']['path'])
    import native_transverse_bundle as runtime
    _CONFIG=dict(data=data,original=runtime.FirstShotSourceBundle)
    runtime.FirstShotSourceBundle=NativeFamilySourceBundle
    return dict(producer=reference(__file__),transition_manifest=data[0])


class NativeFamilySourceBundle:
    def __new__(cls,binding_path,policy,**kwargs):
        cfg=_CONFIG
        if cfg is None:raise ValueError('Explicit verified native-family pipeline required')
        ref,manifest,revision,old_binding,new_binding,before,after,groups=cfg['data']
        if reference(binding_path)!=manifest['revised_binding']or policy!=after:
            raise ValueError('Native family constructor received a foreign binding/association')
        bundle=cfg['original'](manifest['base_binding']['path'],before,**kwargs)
        try:
            if bundle._source_pool is not None or bundle._geometry_pool is not None:
                raise ValueError('Native source delta must precede every worker/evaluation')
            # v22 already authenticated and compared its complete v21->v22 delta.
            for name in ('_authenticated_group_previous_frames','_authenticated_group_previous_scales','_authenticated_group_previous_mask'):
                if hasattr(bundle,name):delattr(bundle,name)
            old_frames=bundle.frames;old_state=bundle.state;old_scales=bundle.scales.copy();old_mask=bundle.quadratic.copy()
            expected={};proofs={};changed=[];rebased_refs=[]
            for frame in old_frames:
                entry=groups.get(frame['index'])
                if entry is None:changed.append(frame);continue
                item=rebase_family_candidate(frame,entry['entry']['candidate'],manifest['base_binding'])
                result,proof=replace_native_family(frame,item,old_binding,identity_review_reference=entry['entry']['identity_review'])
                expected[frame['index']]=result;proofs[frame['index']]=proof;changed.append(result)
                rebased_refs.append(dict(frame=frame['index'],candidate=entry['entry']['candidate'],
                    identity_review=entry['entry']['identity_review'],current_old_point_indices=item['old_point_indices'],
                    current_metadata_rebased_from_authenticated_actual_source=True))
            from .first_shot_fit import FrozenWhitening,_target_key
            from .first_shot_physical_state import FirstShotPhysicalState
            state=FirstShotPhysicalState.from_binding(binding_path)
            for name in ('initial_q','parent_q','parent_shared_slots','pts','world_parameter_indices','global_shared_indices'):
                if not np.array_equal(getattr(old_state,name),getattr(state,name)):
                    raise ValueError('Native source replacement changed physical chart '+name)
            descriptors=OrderedDict();updated=[]
            for frame in changed:
                rows=[]
                for previous in frame['rows']:
                    row=dict(previous);target=deepcopy(row['target'])
                    if target.get('kind')=='board_edge'and'association_policy'in target:target['association_policy']=deepcopy(policy)
                    row['target']=target;row['_target_key']=_target_key(target);rows.append(row);descriptors.setdefault(row['_target_key'],target)
                updated.append(dict(frame,rows=rows))
            scope=compare_transition(old_frames,updated,expected,proofs)
            for key in ('old_source_points','source_points','old_formal_rows','new_formal_rows'):
                if scope[key]!=revision['expected_inventory'][key]:
                    raise ValueError('Actual admitted source inventory differs: '+key)
            old_keep=scope.pop('_old_unchanged_formal_indices');new_keep=scope.pop('_new_unchanged_formal_indices')
            whitening=[FrozenWhitening(f)for f in updated];scales=np.concatenate([w.scales for w in whitening]);mask=np.zeros(len(scales),bool)
            if not np.array_equal(old_scales[old_keep],scales[new_keep])or not np.array_equal(old_mask[old_keep],mask[new_keep]):
                raise ValueError('Native source delta changed unrelated formal scales/masks')
            # F2 rows have no movable straight-line endpoint. Every actual
            # selected coverage row is byte-identical; retain the already
            # authenticated v22 inventory including its 1248 support rebuild.
            def coverage_rows(frames):
                return [{k:v for k,v in r.items()}for f in frames for r in f['rows']if r.get('_continuous_world_line')]
            if _digest(coverage_rows(old_frames))!=_digest(coverage_rows(updated)):
                raise ValueError('Native F2 delta changed finite straight-line coverage source rows')
            bundle.state=state;bundle.policy=deepcopy(policy);bundle.frames=updated
            bundle.descriptors=descriptors;bundle.keys=tuple(descriptors);bundle.target_descriptors=tuple(descriptors.values())
            bundle.initial_target_proof=state.validate_initial_targets(bundle.target_descriptors)
            bundle.whitening=whitening;bundle.scales=scales;bundle.quadratic=mask;bundle.ledgers=[]
            for frame,w in zip(updated,whitening,strict=True):
                h=hashlib.sha256()
                for row in frame['rows']:
                    h.update(json.dumps(row['source_point_ids'],separators=(',',':')).encode());h.update(row['uv'].tobytes());h.update(row['sigma'].tobytes())
                for g in frame['whitening_groups']:h.update(g['operator'].matrix.tobytes())
                bundle.ledgers.append(dict(family='accepted_official_static_source',programme_index=frame['index'],native_pts=frame['native_pts'],
                    time_base='1/12800',source_points=frame['source_point_count'],formal_rows=w.count,source_inventory_sha256=h.hexdigest(),
                    source_packet=frame['source_packet'],association_policy=deepcopy(policy)))
            bundle._world_cache.clear();bundle._finite_coverage_cache.clear()
            bundle._native_family_previous_frames=old_frames;bundle._native_family_previous_state=old_state
            bundle._native_family_previous_scales=old_scales;bundle._native_family_previous_mask=old_mask
            receipt=dict(producer=reference(__file__),transition_manifest=ref,family_revision=manifest['family_revision'],
                source_scope=scope,families=[proofs[i]for i in sorted(proofs)],rebased_actual_metadata=rebased_refs,
                previous_v22_complete_pipeline_authenticated_without_bypass=True,finite_line_coverage_inventory_reused_after_exact_source_check=True,
                original_native_family_all_readouts_and_old_representatives_archived=True,new_basis_fresh_r_J_world_required=True,
                production_source_pipeline_installed=True)
            bundle.native_family_receipt=receipt
            # The existing review count protocol accepts whole-family deltas;
            # distinguish their ordinary finite-curve kernel explicitly.
            previous_native=deepcopy(bundle.source_native_line_revision_proof)
            native_proof=dict(schema='rinkspace.composed-native-readout-review-inventory.v1',
                reference=manifest['family_revision'],original_authenticated_native_line_revision=previous_native,
                old_representatives_removed_atomically=True,
                frames=deepcopy(previous_native['frames'])+[dict(frame=i,unchanged=False,
                    old_raw_points=proofs[i]['old_raw_points'],new_raw_points=proofs[i]['new_raw_points'],
                    source_prediction_kernel='existing_point_to_named_finite_ice_curve_observed_y_RS',whole_family=True)for i in sorted(proofs)],
                original_global_formal_mappings_remain_historical_stage_only=True,
                current_complete_formal_rows=scope['new_formal_rows'],current_complete_source_points=scope['source_points'])
            bundle.source_native_line_revision_proof=native_proof
            return bundle
        except BaseException:
            bundle.close();raise


def attach_native_family_pipeline(owner):
    receipt=owner.bundle.native_family_receipt
    from .first_shot_group_rebuild import attach_group_pipeline
    attach_group_pipeline(owner)
    refs=[reference(__file__),reference(__import__('calibration.program_tracking.first_shot_native_family_revision',fromlist=['x']).__file__),
        receipt['transition_manifest'],receipt['family_revision']]
    for name in ('first_shot_native_circle','source_evidence_whitening','first_shot_noise_revision'):
        refs.append(reference(__import__('calibration.program_tracking.'+name,fromlist=['x']).__file__))
    for item in receipt['rebased_actual_metadata']:refs.extend((item['candidate'],item['identity_review']))
    for ref in refs:
        if ref not in owner.source_producers:owner.source_producers.append(deepcopy(ref))
    previous=owner.checkpoint_identity
    def identity(adapter):
        result=previous();result['reviewed_native_finite_curve_family_replacement']=deepcopy(receipt);return result
    owner.checkpoint_identity=MethodType(identity,owner)
    return receipt
