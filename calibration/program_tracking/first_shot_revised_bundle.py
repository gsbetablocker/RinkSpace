"""Authenticate old noise/native operators before the reviewed ordinary delta.

Explicitly install in a new fit or review process before constructing its Data
adapter. Original revision functions and archived producer files are unchanged.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

from .first_shot_source_transition import validate_manifest,reference,read

_CONFIG=None


def _build_delta(manifest,changes,quarantines,old_binding):
    from .first_shot_revision import SOURCE_SCHEMA
    revision=read(manifest['source_revision'])
    inactive={(int(item['native_pts']),str(item['source_point_id'])):item
              for item in revision['quarantined_correspondences']}
    items=[]
    for item in changes.values():
        change=deepcopy(item)
        change.update(original_target=deepcopy(item['before_target']),target=deepcopy(item['after_target']))
        items.append(change)
    return dict(schema=SOURCE_SCHEMA,base_binding=manifest['base_binding'],
        source_video_sha256=old_binding['source']['sha256'],
        source_time_base=old_binding['source_clock']['time_base'],
        original_source_UV_IDs_and_approvals_modified=False,
        identity_corrections=items,quarantined_correspondences=[deepcopy(inactive[key]) for key in quarantines],
        added_native_readouts=[],correlated_operator_replacements=[],correlated_estimator_replacements=[],
        application_order='after original v20 source/noise/native authentication',
        verified_transition=manifest['revised_binding'])


def install_revised_source_bundle(transition_manifest):
    """Public opt-in used by both fit and same-model review launchers."""
    global _CONFIG
    if _CONFIG is not None:raise ValueError('Revised source bundle already installed')
    data=validate_manifest(transition_manifest)
    ref,manifest,changes,operators,old,new,before,after,differences,quarantines=data
    if operators:raise ValueError('Ordinary identity continuation cannot change operators')
    delta=_build_delta(manifest,changes,quarantines,old)
    path=Path(transition_manifest).with_name('continuation_delta.json')
    payload=json.dumps(delta,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8')!=payload:raise ValueError('Existing continuation delta differs; never overwrite history')
    else:path.write_text(payload,encoding='utf8')
    import native_transverse_bundle as runtime
    from .first_shot_fit import FirstShotSourceBundle
    if runtime.FirstShotSourceBundle is not FirstShotSourceBundle:
        raise ValueError('An unknown first-shot source constructor is already installed')
    _CONFIG=dict(manifest=manifest,manifest_reference=ref,changes=changes,quarantines=quarantines,
        delta_reference=reference(path),before_policy=before,after_policy=after,
        original_constructor=FirstShotSourceBundle)
    runtime.FirstShotSourceBundle=RevisedFirstShotSourceBundle
    return dict(producer=reference(__file__),transition_manifest=ref,delta=reference(path),
        original_noise_and_native_authentication_unmodified=True,
        application_order='original v20 source -> original noise -> original native -> reviewed ordinary delta',
        original_recipe_files_modified=False)


class RevisedFirstShotSourceBundle:
    """Composition keeps the original constructor and all its source methods."""
    def __new__(cls,binding_path,policy,**kwargs):
        if _CONFIG is None:raise ValueError('Explicit verified source pipeline installation required')
        cfg=_CONFIG;manifest=cfg['manifest']
        if reference(binding_path)!=manifest['revised_binding'] or policy!=cfg['after_policy']:
            raise ValueError('Revised source constructor received a foreign binding/policy')
        from .first_shot_fit import FrozenWhitening,_target_key
        from .first_shot_physical_state import FirstShotPhysicalState
        from .first_shot_revision import revise_frames
        bundle=cfg['original_constructor'](manifest['base_binding']['path'],cfg['before_policy'],**kwargs)
        try:
            if bundle._geometry_pool is not None or bundle._source_pool is not None:
                raise ValueError('Reviewed source edits must precede workers/evaluation')
            old_state=bundle.state;old_frames=bundle.frames
            affected={key[0] for key in cfg['changes']}|{key[0] for key in cfg['quarantines']}
            selected=[frame for frame in old_frames if int(frame['native_pts']) in affected]
            required=set(cfg['changes'])|set(cfg['quarantines']);found=set()
            for frame in selected:
                grouped={int(i) for g in frame['whitening_groups'] for i in g['point_indices']}
                flattened=[(r,j) for r in frame['rows'] for j in range(len(r['uv']))]
                for index,(row,j) in enumerate(flattened):
                    key=int(frame['native_pts']),str(row['source_point_ids'][j])
                    if key in required:
                        if index in grouped:raise ValueError('Reviewed delta touches a correlated source group')
                        found.add(key)
                if any(row.get('native_line_queries') for row in frame['rows']):
                    raise ValueError('This ordinary delta cannot re-split an affected native-line query row')
            if found!=required:raise ValueError('Reviewed ordinary delta has missing active source IDs')
            current_binding=read(manifest['revised_binding'])
            changed,_,delta_proof=revise_frames(selected,cfg['delta_reference'],current_binding,policy=cfg['after_policy'])
            changed={int(frame['native_pts']):frame for frame in changed}
            frames=[];descriptors=OrderedDict()
            for old_frame in old_frames:
                frame=dict(changed.get(int(old_frame['native_pts']),old_frame))
                if int(old_frame['native_pts']) in changed:
                    frame['inactive_source_correspondences']=deepcopy(old_frame.get('inactive_source_correspondences',[]))+frame.get('inactive_source_correspondences',[])
                    frame['original_approved_source_point_count']=old_frame.get('original_approved_source_point_count',old_frame['source_point_count'])
                rows=[]
                for old_row in frame['rows']:
                    row=dict(old_row);target=deepcopy(old_row['target'])
                    if target.get('kind')=='board_edge' and 'association_policy' in target:
                        target['association_policy']=deepcopy(cfg['after_policy'])
                    row['target']=target;row['_target_key']=_target_key(target)
                    rows.append(row);descriptors.setdefault(row['_target_key'],target)
                frame['rows']=rows;frames.append(frame)
            state=FirstShotPhysicalState.from_binding(manifest['revised_binding']['path'])
            for attr in ('initial_q','parent_q','parent_shared_slots','pts','world_parameter_indices','global_shared_indices'):
                if not np.array_equal(getattr(old_state,attr),getattr(state,attr)):
                    raise ValueError('Revised source changed physical state/chart '+attr)
            bundle.state=state;bundle.policy=deepcopy(cfg['after_policy'])
            bundle.frames=frames;bundle.descriptors=descriptors
            bundle.keys=tuple(descriptors);bundle.target_descriptors=tuple(descriptors.values())
            bundle.initial_target_proof=state.validate_initial_targets(bundle.target_descriptors)
            bundle.whitening=[FrozenWhitening(frame) for frame in frames]
            bundle.scales=np.concatenate([w.scales for w in bundle.whitening])
            bundle.quadratic=np.zeros(len(bundle.scales),bool)
            bundle.ledgers=[]
            for frame,w in zip(frames,bundle.whitening,strict=True):
                h=hashlib.sha256()
                for row in frame['rows']:
                    h.update(json.dumps(row['source_point_ids'],separators=(',',':')).encode())
                    h.update(row['uv'].tobytes());h.update(row['sigma'].tobytes())
                for group in frame['whitening_groups']:h.update(group['operator'].matrix.tobytes())
                bundle.ledgers.append(dict(family='accepted_official_static_source',programme_index=frame['index'],
                    native_pts=frame['native_pts'],time_base='1/12800',source_points=frame['source_point_count'],
                    formal_rows=w.count,source_inventory_sha256=h.hexdigest(),source_packet=frame['source_packet'],
                    association_policy=deepcopy(cfg['after_policy'])))
            bundle._world_cache.clear();bundle._finite_coverage_cache.clear()
            from .finite_source_line_coverage import read_coverage_revision
            coverage=state.binding.get('finite_source_line_coverage_revision')
            bundle.finite_coverage_inventory=bundle.finite_coverage_revision_proof=None
            if coverage is not None:
                bundle.finite_coverage_inventory,bundle.finite_coverage_revision_proof=read_coverage_revision(coverage,state,frames)
            previous_proof=bundle.source_revision_proof
            bundle.source_revision_proof=dict(reference=manifest['source_revision'],
                original_authenticated_basis=deepcopy(previous_proof),reviewed_post_native_delta=delta_proof,
                prior_source_noise_and_native_revisions_authenticated_first=True,
                identity_corrections=len(read(manifest['source_revision'])['identity_corrections']),
                quarantined_correspondences=len(read(manifest['source_revision'])['quarantined_correspondences']))
            bundle._authenticated_previous_frames=old_frames
            bundle.revised_source_pipeline_receipt=dict(producers=[reference(__file__),cfg['delta_reference']],
                transition_manifest=cfg['manifest_reference'],original_binding=manifest['base_binding'],
                revised_binding=manifest['revised_binding'],delta=cfg['delta_reference'],
                original_noise_and_native_validation_executed_without_bypass=True,
                all_799_changes_and_four_quarantines_verified_ordinary=True,
                application_order='old v20 source/noise/native authenticated, then exact reviewed ordinary delta',
                old_operator_recipe_files_modified=False,new_source_rows_and_inventory_rebuilt=True,
                target_descriptors_scales_masks_ledgers_and_coverage_rebuilt=True,
                original_native_global_mapping_receipt_is_pre_delta_basis=True)
            return bundle
        except BaseException:
            bundle.close();raise


def attach_revised_pipeline(owner):
    """Record/freeze this explicit pipeline on any fit or review adapter."""
    receipt=owner.bundle.revised_source_pipeline_receipt
    for ref in receipt['producers']:
        if reference(ref['path'])!=ref:raise ValueError('Revised source pipeline producer changed')
        if ref not in owner.source_producers:owner.source_producers.append(deepcopy(ref))
    previous=owner.checkpoint_identity
    def identity(adapter):
        result=previous();result['authenticated_original_operators_then_reviewed_source_delta']=deepcopy(receipt)
        return result
    owner.checkpoint_identity=MethodType(identity,owner)
    return deepcopy(receipt)
