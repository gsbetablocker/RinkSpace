"""Apply an ordinary identity delta after the authenticated native pipeline."""
from collections import OrderedDict
from copy import deepcopy
from types import MethodType
import hashlib
import json

import numpy as np

from .first_shot_source_transition import reference, read, _target, _evidence

SCHEMA = 'rinkspace.post-native-ordinary-identity-transition.v1'
FIELD = 'post_native_ordinary_revision'
_CONFIG = None


def read_transition(path):
    ref = reference(path)
    manifest = read(ref)
    if (manifest.get('schema') != SCHEMA or manifest.get('authority') != 'agent_rgb_review'
            or manifest.get('review_complete') is not True or manifest.get('user_confirmed') is not False):
        raise ValueError('A completed RGB ordinary identity transition is required')
    from .first_shot_native_family_bundle import read_transition as native_transition
    _, parent, *_ = native_transition(manifest['authenticated_native_transition']['path'])
    if (parent['revised_binding'] != manifest['base_binding']
            or parent['revised_association'] != manifest['base_association']):
        raise ValueError('Ordinary delta must follow the exact authenticated native source')
    old, new = read(manifest['base_binding']), read(manifest['revised_binding'])
    before, after = read(manifest['base_association']), read(manifest['revised_association'])
    if ({k for k in set(old) | set(new) if old.get(k) != new.get(k)} != {FIELD}
            or new[FIELD] != manifest['ordinary_revision']):
        raise ValueError('Binding changed beyond its explicit post-native identity delta')
    if (before.get('source_binding') != manifest['base_binding']
            or after.get('source_binding') != manifest['revised_binding']):
        raise ValueError('Ordinary transition association names another source')
    a, b = deepcopy(before), deepcopy(after)
    a.pop('source_binding'); b.pop('source_binding')
    if a != b:
        raise ValueError('Ordinary identity delta cannot change physical association')
    revision = read(manifest['ordinary_revision'])
    from .first_shot_revision import SOURCE_SCHEMA
    if (revision.get('schema') != SOURCE_SCHEMA or revision.get('base_binding') != manifest['base_binding']
            or revision.get('source_video_sha256') != old['source']['sha256']
            or revision.get('source_time_base') != old['source_clock']['time_base']
            or revision.get('original_source_UV_IDs_and_approvals_modified') is not False):
        raise ValueError('Ordinary revision must preserve this exact source clock and evidence')
    for field in ('added_native_readouts', 'quarantined_correspondences',
                  'correlated_operator_replacements', 'correlated_estimator_replacements'):
        if revision.get(field):
            raise ValueError('This identity-only continuation cannot change ' + field)
    changes = {}
    for item in revision.get('identity_corrections', []):
        _evidence(item)
        key = int(item['native_pts']), str(item['source_point_id'])
        if key in changes or _target(item['original_target']) == _target(item['target']):
            raise ValueError('Duplicate or ineffective ordinary target correction')
        changes[key] = item
    if not changes:
        raise ValueError('At least one evidenced ordinary target correction is required')
    return ref, manifest, revision, old, new, before, after, changes


def install_post_native_identity(path):
    global _CONFIG
    if _CONFIG is not None:
        raise ValueError('Post-native identity pipeline already installed')
    data = read_transition(path)
    from .first_shot_native_family_bundle import install_native_family_bundle
    install_native_family_bundle(data[1]['authenticated_native_transition']['path'])
    import native_transverse_bundle as runtime
    _CONFIG = dict(data=data, original=runtime.FirstShotSourceBundle)
    runtime.FirstShotSourceBundle = PostNativeIdentityBundle
    return dict(producer=reference(__file__), transition_manifest=data[0])


class PostNativeIdentityBundle:
    def __new__(cls, binding_path, policy, **kwargs):
        if _CONFIG is None:
            raise ValueError('Explicit post-native identity installation required')
        ref, manifest, revision, old, new, before, after, changes = _CONFIG['data']
        if reference(binding_path) != manifest['revised_binding'] or policy != after:
            raise ValueError('Post-native constructor received a foreign source')
        bundle = _CONFIG['original'](manifest['base_binding']['path'], before, **kwargs)
        try:
            if bundle._source_pool is not None or bundle._geometry_pool is not None:
                raise ValueError('Ordinary identity delta must precede every worker')
            # The native stage has already authenticated its predecessor.
            for name in ('frames', 'state', 'scales', 'mask'):
                attr = '_native_family_previous_' + name
                if hasattr(bundle, attr):
                    delattr(bundle, attr)
            old_frames, old_state = bundle.frames, bundle.state
            old_scales, old_mask = bundle.scales.copy(), bundle.quadratic.copy()
            from .first_shot_ordinary_delta import apply_ordinary_delta
            frames, proof = apply_ordinary_delta(old_frames, manifest['ordinary_revision'], new, policy=after)
            proof['complete_source_comparison'].pop('_retained_formal_indices', None)
            from .first_shot_fit import FrozenWhitening, _target_key
            from .first_shot_physical_state import FirstShotPhysicalState
            state = FirstShotPhysicalState.from_binding(binding_path)
            for name in ('initial_q', 'parent_q', 'parent_shared_slots', 'pts',
                         'world_parameter_indices', 'global_shared_indices'):
                if not np.array_equal(getattr(old_state, name), getattr(state, name)):
                    raise ValueError('Ordinary identity delta changed the physical chart: ' + name)
            descriptors, updated = OrderedDict(), []
            for frame in frames:
                rows = []
                for previous in frame['rows']:
                    target = deepcopy(previous['target'])
                    if target.get('kind') == 'board_edge' and 'association_policy' in target:
                        target['association_policy'] = deepcopy(after)
                    row = dict(previous, target=target, _target_key=_target_key(target))
                    rows.append(row); descriptors.setdefault(row['_target_key'], target)
                updated.append(dict(frame, rows=rows))
            from .first_shot_source_transition import compare_source_frames
            aliases = {key: dict(item, before_target=item['original_target'], after_target=item['target'])
                       for key, item in changes.items()}
            scope = compare_source_frames(old_frames, updated, aliases, {})
            scope.pop('_retained_formal_indices')
            whitening = [FrozenWhitening(f) for f in updated]
            scales = np.concatenate([w.scales for w in whitening])
            if not np.array_equal(scales, old_scales):
                raise ValueError('Ordinary identities changed formal row scales/order')
            # No source points or line supports changed, so this current
            # coverage inventory remains valid without editing its evidence.
            from .first_shot_source_transition import _digest
            def coverage(frames):
                return [r for f in frames for r in f['rows'] if r.get('_continuous_world_line')]
            if _digest(coverage(old_frames)) != _digest(coverage(updated)):
                raise ValueError('Ordinary identities changed finite-line support')
            bundle.state, bundle.policy, bundle.frames = state, deepcopy(after), updated
            bundle.descriptors, bundle.keys = descriptors, tuple(descriptors)
            bundle.target_descriptors = tuple(descriptors.values())
            bundle.initial_target_proof = state.validate_initial_targets(bundle.target_descriptors)
            bundle.whitening, bundle.scales, bundle.quadratic = whitening, scales, old_mask.copy()
            bundle.ledgers = []
            for frame, w in zip(updated, whitening, strict=True):
                h = hashlib.sha256()
                for row in frame['rows']:
                    h.update(json.dumps(row['source_point_ids'], separators=(',', ':')).encode())
                    h.update(row['uv'].tobytes()); h.update(row['sigma'].tobytes())
                for group in frame['whitening_groups']:
                    h.update(group['operator'].matrix.tobytes())
                bundle.ledgers.append(dict(family='accepted_official_static_source', programme_index=frame['index'],
                    native_pts=frame['native_pts'], time_base='1/12800', source_points=frame['source_point_count'],
                    formal_rows=w.count, source_inventory_sha256=h.hexdigest(), source_packet=frame['source_packet'],
                    association_policy=deepcopy(after)))
            bundle._world_cache.clear(); bundle._finite_coverage_cache.clear()
            bundle._post_identity_previous_frames, bundle._post_identity_previous_state = old_frames, old_state
            bundle._post_identity_previous_scales, bundle._post_identity_previous_mask = old_scales, old_mask
            bundle.post_native_identity_receipt = dict(producer=reference(__file__), transition_manifest=ref,
                ordinary_revision=manifest['ordinary_revision'], source_scope=scope,
                identity_corrections=len(changes), delta_proof=proof,
                previous_native_source_pipeline_authenticated_without_bypass=True,
                original_UV_sigma_W_clock_native_queries_and_parameter_chart_preserved=True,
                finite_line_coverage_reused_after_exact_support_check=True, new_basis_fresh_r_J_world_required=True)
            return bundle
        except BaseException:
            bundle.close()
            raise


def attach_post_native_identity(owner):
    receipt = owner.bundle.post_native_identity_receipt
    refs = [reference(__file__), reference(__import__('calibration.program_tracking.first_shot_ordinary_delta',
        fromlist=['x']).__file__), receipt['transition_manifest'], receipt['ordinary_revision']]
    for ref in refs:
        if ref not in owner.source_producers:
            owner.source_producers.append(deepcopy(ref))
    previous = owner.checkpoint_identity
    def identity(adapter):
        result = previous(); result['post_native_ordinary_identity_delta'] = deepcopy(receipt)
        return result
    owner.checkpoint_identity = MethodType(identity, owner)
    return receipt
