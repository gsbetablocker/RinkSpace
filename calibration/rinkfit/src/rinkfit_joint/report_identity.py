"""Bind replay statistics to the frozen source objective, separate from auxiliaries.

Legacy files can be inspected explicitly as unbound diagnostics. They cannot
be promoted to an original-target report by adding a sidecar after the run.
"""
from pathlib import Path
import numpy as np
from .io_utils import file_sha256
from .measurement_operator import canonical_tree_sha256, measurement_operator_binding, dependency_identity

SCHEMA = 'rinkfit.joint_replay_objective_binding.v1'
TARGET_SCHEMA = 'rinkfit.joint_frozen_W_E_target.v1'
STATE_SCHEMA = 'rinkfit.joint_profiled_replay_state.v1'


def frozen_target_binding(packet):
    """Original E/W and the measurement definition; never a station/row/alpha seed."""
    arrays = packet.get('arrays', {})
    value = dict(schema=TARGET_SCHEMA,
        measurement_operator=measurement_operator_binding(packet),
        shot_ids=packet.get('shot_ids', []), source_clock=packet.get('source_clock'),
        source_manifests=packet.get('source_manifests', []),
        arrays={key: arrays[key] for key in ('E', 'source_sigma') if key in arrays},
        groups=packet.get('groups', []), independent_rows=packet.get('independent_rows', []))
    digest = canonical_tree_sha256(dependency_identity(value))
    return dict(schema=TARGET_SCHEMA, sha256=digest,
        objective_id='original_frozen_W_E_v1:' + digest)


def actual_exclusion_ledger(packet, load):
    n = int(packet['row_count'])
    admission = load('admission').reshape(n)
    role = load('fit_role').reshape(n)
    ids = packet['source_row_ids']
    datasets = packet.get('source_dataset_ids', ['joint'] * n)
    return [dict(row_index=int(i), source_dataset_id=str(datasets[i]),
        source_row_id=str(ids[i]), fit_role=str(int(role[i])), active=False)
        for i in np.flatnonzero(~((admission == 1) & (role == 1)))]


def replay_state_identity(value):
    return canonical_tree_sha256(dict(schema=STATE_SCHEMA,
        model_sha256=value['model_sha256'], files=value['files'],
        replay_semantics=value.get('profiled_native_export_semantics'),
        complete_snapshot_attachment=value.get('complete_snapshot_attachment')))


def make_replay_binding(packet_file, packet, value, exclusions):
    """Producer helper for native parity fixtures; not a legacy repair operation."""
    return dict(schema=SCHEMA, source_packet_sha256=file_sha256(packet_file),
        frozen_target=frozen_target_binding(packet),
        evaluation_state_sha256=replay_state_identity(value),
        evaluation_state_kind=STATE_SCHEMA, exclusion_ledger=exclusions,
        explicit_checkpoint_state_claimed=False)


def require_replay_binding(packet_file, packet, value, state, *, load=None,
                           allow_legacy_diagnostic=False):
    binding = value.get('source_objective_binding')
    if binding is None:
        if allow_legacy_diagnostic:
            return dict(status='legacy_unbound_diagnostic', original_target_verified=False,
                objective_id='legacy_unbound_supplied_packet:' + file_sha256(packet_file),
                limitation='Historical replay has no producer source binding; supplied E/W/roles/Query are diagnostic only.')
        raise ValueError('Replay lacks producer source/objective identity. Original-target reporting and auxiliary rebinding are refused; use an explicit legacy-unbound diagnostic to inspect historical numbers, or create a new bound replay.')
    if binding.get('schema') != SCHEMA:
        raise ValueError('Unsupported replay source/objective identity schema')
    if binding.get('source_packet_sha256') != file_sha256(packet_file):
        raise ValueError('Replay source packet differs from the actual frozen solve/replay input')
    expected = frozen_target_binding(packet)
    if binding.get('frozen_target') != expected:
        raise ValueError('Replay frozen E/W, role, material, finite domain, Query or source order changed')
    if binding.get('evaluation_state_kind') != STATE_SCHEMA or binding.get('explicit_checkpoint_state_claimed') is not False:
        raise ValueError('Profiled replay is not an explicit complete-checkpoint state')
    if binding.get('evaluation_state_sha256') != replay_state_identity(value):
        raise ValueError('Replay model/output/semantics state identity changed')
    root = Path(state)
    attachment=value.get('complete_snapshot_attachment')
    if attachment is not None:
        from .io_utils import read_json
        if (attachment.get('schema')!='rinkfit.joint_replay_complete_snapshot_attachment.v1'
            or attachment.get('source_packet_sha256')!=binding['source_packet_sha256']
            or attachment.get('physical_model_sha256')!=value['model_sha256']
            or not attachment.get('state_token')):
            raise ValueError('Replay complete-snapshot attachment identity changed')
        descriptor=attachment['explicit_nuisance_manifest'];mp=root/descriptor['file']
        if mp.resolve()!= (root/'explicit_nuisance/manifest.json').resolve() or file_sha256(mp)!=descriptor['sha256']:
            raise ValueError('Replay explicit auxiliary manifest changed')
        manifest=read_json(mp)
        if manifest['packet_sha256']!=binding['source_packet_sha256'] or int(manifest['raw_rows'])!=int(packet['row_count']):
            raise ValueError('Replay explicit auxiliary source differs')
        for name,ref in manifest['files'].items():
            path=mp.parent/name
            if path.resolve().parent!=mp.parent.resolve() or path.stat().st_size!=ref['size_bytes'] or file_sha256(path)!=ref['sha256']:
                raise ValueError('Replay explicit auxiliary file changed: '+name)
    for name, descriptor in value['files'].items():
        path = root / name
        if path.resolve().parent != root.resolve():
            raise ValueError('Replay file ledger must contain local state filenames')
        if path.stat().st_size != descriptor['size_bytes'] or file_sha256(path) != descriptor['sha256']:
            raise ValueError('Replay state array changed: ' + name)
    if load is None:
        from .joint_metrics import PacketArrays
        load = PacketArrays(packet_file).load
    # Verify the frozen numerical dependencies before any caller writes statistics.
    # Reuse PacketArrays' per-call SHA cache; no extra array copies are made.
    owner=getattr(load,'__self__',None); cache=getattr(owner,'checked',{})
    base=Path(packet_file).parent
    descriptors=[d for k,d in packet.get('arrays',{}).items() if k not in {'station','row','alpha','path'}]
    descriptors.extend(g['W'] for g in packet.get('groups',[]) if isinstance(g.get('W'),dict))
    for descriptor in descriptors:
        path=Path(descriptor['file']);path=path if path.is_absolute() else base/path
        key=str(path)
        if key not in cache:cache[key]=file_sha256(path)
        if cache[key]!=descriptor['sha256']:raise ValueError('Frozen target numerical dependency changed: '+str(path))
    ledger = actual_exclusion_ledger(packet, load)
    if binding.get('exclusion_ledger') != ledger:
        raise ValueError('Replay exclusion ledger differs from the actual admission/role input')
    return dict(status='producer_bound_original_target', original_target_verified=True,
        source_packet_sha256=binding['source_packet_sha256'],
        frozen_target=expected, evaluation_state_sha256=binding['evaluation_state_sha256'],
        exclusion_ledger=ledger, explicit_checkpoint_state_claimed=False)


def mark_unbound_diagnostic(value):
    """Do not label the caller's unauthenticated budgets as original pass/fail."""
    names = {'all_original_E_passed': 'all_supplied_E_passed',
        'rows_above_original_E': 'rows_above_supplied_E', 'original_E_px': 'supplied_E_px',
        'all_operator_errors_within_original_E': 'all_operator_errors_within_supplied_E'}
    if isinstance(value, dict):
        return {names.get(k, k): mark_unbound_diagnostic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mark_unbound_diagnostic(v) for v in value]
    return value


def assert_frozen_source_prefix(old, current, old_load, current_load):
    """A typed append may add observations but cannot change the replay prefix."""
    n = int(old['row_count'])
    if int(current['row_count']) < n:
        raise ValueError('Replay source prefix is longer than the appended packet')
    for name in ('source_row_ids', 'source_dataset_ids', 'source_family_ids'):
        if name in old and current.get(name, [])[:n] != old[name]:
            raise ValueError('Appended packet changed frozen prefix ' + name)
    if old.get('shot_ids') != current.get('shot_ids', [])[:len(old.get('shot_ids', []))]:
        raise ValueError('Appended packet changed frozen shot registry')
    for name in old.get('arrays', {}):
        if name in {'station', 'path', 'row', 'alpha'}:
            continue
        if name not in current.get('arrays', {}):
            raise ValueError('Appended packet lost frozen prefix array ' + name)
        a, b = old_load(name), current_load(name)
        if a.shape[0] != n or b.shape[0] < n or not np.array_equal(a, b[:n], equal_nan=True):
            raise ValueError('Appended packet changed frozen prefix array ' + name)
    def source_value(value):
        if isinstance(value, dict):
            return {k: source_value(v) for k, v in value.items()
                if not (k in {'file', 'path'} and 'sha256' in value) and k != 'alpha0'}
        if isinstance(value, list):
            return [source_value(v) for v in value]
        return value
    def rows(packet, key):
        return [q for q in packet.get(key, []) if int(q['row_index']) < n]
    for key in ('queries',):
        if source_value(old.get(key, [])) != source_value(rows(current, key)):
            raise ValueError('Appended packet changed frozen prefix ' + key)
    for key in ('allowed_paths',):
        prefix={k:v for k,v in current.get(key, {}).items() if int(k)<n}
        if old.get(key, {}) != prefix:
            raise ValueError('Appended packet changed frozen finite source paths')
    for key in ('source_station_domains',):
        a=old.get(key, {}).get('rows', [])
        b=[q for q in current.get(key, {}).get('rows', []) if int(q['row_index'])<n]
        if source_value(a) != source_value(b):
            raise ValueError('Appended packet changed frozen source station domains')
    prefix_groups=[]
    for group in current.get('groups', []):
        rr=group['rows']
        if any(int(i)<n for i in rr):
            if not all(int(i)<n for i in rr):
                raise ValueError('Append regrouped an existing replay source unit')
            prefix_groups.append(group)
    if source_value(old.get('groups', [])) != source_value(prefix_groups):
        raise ValueError('Appended packet changed frozen prefix W/group objective')
    if old.get('independent_rows', []) != [i for i in current.get('independent_rows', []) if int(i)<n]:
        raise ValueError('Appended packet changed frozen independent source units')
