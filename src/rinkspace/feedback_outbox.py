"""Replay committed human saves into event feedback without changing save success.

The committed edit journal is the durable queue. Delivery receipts are separate
from the original transaction, and every consumed before/after version has an
immutable snapshot. Database failures leave the journal replayable.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .path_policy import guard_project_write
from .storage import AssetStore

SCHEMA = 'rinkspace.human-edit-feedback-delivery.v1'


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _ref(path, *, kind=None):
    path = Path(path).resolve()
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if kind: result['kind'] = kind
    return result


def _write(path, value, project):
    path = guard_project_write(path, project_root=project)
    temporary = guard_project_write(path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.tmp'),
                                    project_root=project)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_bytes(_bytes(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return _ref(path)


def _immutable(path, value, project):
    path = guard_project_write(path, project_root=project)
    if path.exists():
        if path.read_bytes() != _bytes(value):
            raise ValueError('Durable feedback version is immutable: ' + str(path))
        return _ref(path)
    return _write(path, value, project)


def restore_manual_cases(store, project, event_id):
    """Rebuild the derived index in numeric version order, including retractions."""
    directory = Path(project) / 'events' / event_id / 'feedback' / 'manual'
    cases = []
    for path in directory.glob('*.v*.json'):
        if path.name.endswith('.regression.json'): continue
        case = _read(path)
        if not case.get('case_id', '').startswith('human_edit:'):
            raise ValueError('Manual feedback journal contains a foreign case')
        cases.append(case)
    return [store.upsert_case(case) for case in sorted(cases, key=lambda row: (row['case_id'], row['version']))]


def _scope(source, dataset, project):
    """Resolve only within this explicit project's registered roots.

    An isolated project/data tree may itself be nested inside another event.
    Those outer directory names are not this project's event authority.
    """
    assets=AssetStore(project)
    event,programme=source.get('event_id'),source.get('programme_id')
    if any(source.get(key) for key in ('video_sha256','source_video_sha256','source_sha256')):
        from .feedback_runtime import native_source,resolve_scope
        known_event,known_programme=resolve_scope(native_source(source),project)
        if known_event:
            if event and event!=known_event:raise ValueError('Source event conflicts with its registered source')
            if programme and known_programme and programme!=known_programme:
                raise ValueError('Source programme conflicts with its registered source')
            event,programme=known_event,known_programme or programme
    path=Path(dataset).resolve()
    path_scopes=[]
    for root in sorted({assets.project,assets.data},key=lambda value:len(value.parts),reverse=True):
        if not path.is_relative_to(root):continue
        relative=path.relative_to(root).parts
        if len(relative)>=2 and relative[0]=='events':
            path_scopes.append((relative[1],relative[3] if len(relative)>=4 and relative[2]=='programmes' else None))
        # The most specific registered root owns the path; an outer data root
        # cannot reinterpret this project's own nested directory names.
        break
    for path_event,path_programme in path_scopes:
        if event and event!=path_event:raise ValueError('Source event conflicts with registered dataset path')
        if programme and path_programme and programme!=path_programme:
            raise ValueError('Source programme conflicts with registered dataset path')
        event,programme=path_event,path_programme or programme
    if not isinstance(event,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',event):
        raise ValueError('Feedback needs a registered event ID; no global event fallback')
    return event,programme


def registered_source_dataset(project,path,source):
    """Require the registered source and its event/programme dataset tree.

    Source SHA or a caller-supplied event ID alone cannot publish human facts.
    Registered legacy aliases are accepted only when their original path is
    inside this exact programme's source_datasets subtree.
    """
    project=Path(project).resolve();assets=AssetStore(project)
    from .feedback_runtime import native_source,resolve_scope
    native=native_source(source)
    event,programme=resolve_scope(native,project)
    if not event or not programme:
        return False,dict(reason='source_not_registered_to_one_programme',event_id=event,programme_id=programme)
    if source.get('event_id') not in (None,event) or source.get('programme_id') not in (None,programme):
        return False,dict(reason='source_scope_conflicts_with_registry',event_id=event,programme_id=programme)
    relative=Path('events')/event/'programmes'/programme/'source_datasets'
    roots=[assets.project/relative,assets.data/relative,assets.resolve(assets.project/relative)]
    for original,(target,*_) in assets.path_aliases.items():
        if original.is_relative_to(relative):roots.append(assets.project/target)
    resolved=Path(path).resolve()
    registered=any(resolved.is_relative_to(root.resolve()) for root in roots)
    return registered,dict(reason='registered_programme_source_dataset' if registered else 'outside_registered_programme_source_dataset_trees',
        event_id=event,programme_id=programme,registered_trees=sorted({str(root.resolve()) for root in roots}))


def _points(packet):
    result = {}
    for collection in ('features', 'unbound_static_materials'):
        for feature in packet.get(collection, []):
            for si, segment in enumerate(feature.get('segments', [])):
                ids = segment.get('point_ids', [f"{feature['id']}/segment/{si}/vertex/{i}"
                                              for i in range(len(segment['uv']))])
                if len(ids) != len(segment['uv']): raise ValueError('Point IDs and UV differ')
                bindings = segment.get('point_bindings', [])
                for vi, (pid, uv) in enumerate(zip(ids, segment['uv'])):
                    if pid in result: raise ValueError('Duplicate source point ID')
                    if len(uv) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                          or not math.isfinite(v) for v in uv):
                        raise ValueError('Finite native source UV required')
                    binding = bindings[vi] if vi < len(bindings) else {}
                    result[pid] = dict(point_id=pid, feature_id=feature['id'],
                        target_id=binding.get('physical_target', feature.get('target_id')),
                        uv=uv, allowed_for_fit=collection == 'features')
    if sum(p['allowed_for_fit'] for p in result.values()) != packet['point_count']:
        raise ValueError('Actual bound point inventory differs from packet count')
    return result


def _values(points):
    return Counter((p['feature_id'], p['target_id'], tuple(p['uv']), p['allowed_for_fit'])
                   for p in points)


def _make_event(transaction, before, after, binding, hashes, evidence, acceptance, event, programme):
    from rink_feedback.adapters import normalize_source, source_key
    source = normalize_source(after['source'])
    source.update(event_id=event)
    if programme: source['programme_id'] = programme
    if source_key(before['source']) != source_key(source):
        raise ValueError('Human edit before/after source identity differs')
    for packet in (before, after):
        if packet.get('source_pts') != source['pts'] or packet.get('programme_frame_index_1based') != transaction['index']:
            raise ValueError('Human edit source PTS or programme index differs')
    if (transaction.get('source_pts') != source['pts'] or binding.get('source_pts') != source['pts']
            or Fraction(binding['source_time_base']) != Fraction(source['time_base'])):
        raise ValueError('Human edit binding source clock differs')
    if (transaction['packet_after']['sha256'] != hashes['after_packet']
            or binding['source_packet']['sha256'] != hashes['after_packet']):
        raise ValueError('Committed transaction and binding do not bind exact after bytes')
    previous, current = _points(before), _points(after)
    if (transaction['point_count_before'] != before['point_count']
            or transaction['point_count_after'] != after['point_count']
            or binding['point_count'] != after['point_count']):
        raise ValueError('Human transaction point count differs')
    removed = [p for pid, p in previous.items() if pid not in current]
    added = [p for pid, p in current.items() if pid not in previous]
    changed = [pid for pid in previous.keys() & current.keys() if previous[pid] != current[pid]]
    action = transaction['action']
    operations = action.get('operations', [])
    if action.get('action') == 'edit':
        if not operations or any(op.get('type') not in ('delete', 'add') for op in operations):
            raise ValueError('Unsupported explicit edit operation; no inferred correction')
        deleted = [op['point_id'] for op in operations if op['type'] == 'delete']
        if len(set(deleted)) != len(deleted) or set(deleted) != {p['point_id'] for p in removed}:
            raise ValueError('Deleted source point set differs from explicit operations')
        requested = Counter((op['target_id'], tuple(round(float(v), 4) for v in op['uv']))
                            for op in operations if op['type'] == 'add')
        if requested != Counter((p['target_id'], tuple(p['uv'])) for p in added):
            raise ValueError('Added source points differ from explicit native clicks')
        for p in added:
            if not p['point_id'].startswith('manual:'):
                raise ValueError('Native click lacks human point identity')
    elif action.get('action') == 'undo':
        if not action.get('undo_of'): raise ValueError('Undo lacks original transaction ID')
    else:
        raise ValueError('Unsupported human action; no inferred correction')
    if changed or _values(p for pid, p in previous.items() if pid in current) != _values(p for pid, p in current.items() if pid in previous):
        raise ValueError('Unrecorded changes to retained source coordinates or identity')
    accepted = False
    acceptance_row = None
    if acceptance is not None:
        rows = [r for r in acceptance.get('frames', []) if r.get('index') == transaction['index']]
        if (acceptance.get('accepted') is not True or len(rows) != 1
                or acceptance.get('official_source', {}).get('sha256') != source['video_sha256']
                or Fraction(acceptance['source_time_base']) != Fraction(source['time_base'])):
            raise ValueError('Acceptance source, clock or exact frame differs')
        acceptance_row = rows[0]
        expected = dict(packet_sha256=hashes['after_packet'], binding_sha256=hashes['binding'],
                        native_pts=source['pts'], point_count=after['point_count'])
        if any(acceptance_row.get(key) != value for key, value in expected.items()):
            raise ValueError('Acceptance does not bind this committed observation version')
        accepted = True
    return dict(schema='rink-feedback.correction-event.v2', event_id='human_edit:' + transaction['id'],
        kind='restore_prior_source_observations' if action.get('action') == 'undo' else 'explicit_source_edit',
        source=source, programme_frame_index_1based=transaction['index'], transaction_id=transaction['id'],
        undo_of=action.get('undo_of'), deleted_points=removed, added_points=added,
        counts=dict(human_transaction_before=before['point_count'], after=after['point_count'],
                    human_deleted=len(removed), human_added=len(added)),
        reason='unknown', reason_authority='not_recorded_in_human_transaction',
        source_observations_accepted=accepted, model_accuracy_accepted=acceptance.get('model_accuracy_accepted') if acceptance else None,
        acceptance_row=acceptance_row, file_hashes=hashes, evidence_refs=evidence,
        exact_source_frame_and_point_ids_only=True, background_negative_created=False,
        measurement_uncertainty='unknown', original_uv_changed_by_adapter=False,
        geometry_accuracy_inferred_from_edit=False, production_integration_status='committed_editor_outbox')


def deliver_human_edit(project, transaction_path, *, dataset, after_packet_path=None,
                       binding_path=None, acceptance_path=None, source_image_path=None):
    """Deliver one committed journal; failure is retried by the editor's scan."""
    from rink_feedback.core import Store
    project, transaction_path = Path(project).resolve(), Path(transaction_path).resolve()
    transaction = _read(transaction_path)
    if transaction.get('status') != 'committed': return dict(status='not_committed', delivered=False)
    if (transaction.get('schema') != 'rinkspace.human-frame-point-edit.v1'
            or transaction.get('model_geometry_changed') is not False):
        raise ValueError('Only committed source-only human edit journals are consumed')
    folder = transaction_path.parent
    before_path = folder / 'before_packet.json'
    after_path = Path(after_packet_path or folder / 'after_packet.json')
    binding_path = Path(binding_path or folder / 'after_binding.json')
    before, after, binding = _read(before_path), _read(after_path), _read(binding_path)
    event, programme = _scope(after['source'], dataset, project)
    registered,registration=registered_source_dataset(project,dataset,after['source'])
    if not registered:
        return dict(schema=SCHEMA,status='diagnostic_dataset_not_published',delivered=False,
            source_observation_save_preserved=True,event_id=event,registration=registration,
            reason='Only the registered programme source dataset may publish human edit facts')
    paths = dict(transaction=transaction_path, before_packet=before_path,
                 after_packet=after_path, binding=binding_path)
    acceptance = _read(acceptance_path) if acceptance_path else None
    if acceptance_path:
        original_acceptance = _ref(acceptance_path)
        snapshot = folder / ('acceptance_' + original_acceptance['sha256'][:12] + '.json')
        snapshot = guard_project_write(snapshot, project_root=project)
        if not snapshot.exists():
            snapshot.write_bytes(Path(acceptance_path).read_bytes())
        if _ref(snapshot)['sha256'] != original_acceptance['sha256']:
            raise ValueError('Acceptance changed while creating immutable feedback evidence')
        paths['acceptance'] = snapshot
    if transaction.get('action',{}).get('action')=='undo':
        undo_id=transaction['action'].get('undo_of','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',undo_id):raise ValueError('Undo transaction identity is invalid')
        prior_folder=folder.parent/undo_id
        prior_transaction=_read(prior_folder/'transaction.json')
        restored=_read(prior_folder/'before_packet.json')
        from rink_feedback.adapters import source_key
        if (prior_transaction.get('status')!='committed' or prior_transaction.get('id')!=undo_id
                or source_key(restored['source'])!=source_key(after['source'])
                or _values(_points(restored).values())!=_values(_points(after).values())):
            raise ValueError('Undo does not exactly restore the referenced committed source inventory')
        paths.update(undo_transaction=prior_folder/'transaction.json',undo_restored_packet=prior_folder/'before_packet.json')
    hashes = {key: _ref(path)['sha256'] for key, path in paths.items()}
    evidence = [dict(_ref(path), kind=key) for key, path in paths.items()]
    if source_image_path:
        image = _ref(source_image_path, kind='native_image')
        if after['source'].get('png_sha256') and after['source']['png_sha256'] != image['sha256']:
            raise ValueError('Native source image SHA differs from committed packet')
        evidence.append(image)
    delivery_key = hashlib.sha256(_bytes(hashes)).hexdigest()
    delivery_path = folder / 'feedback_delivery.json'
    cached_receipt = _read(delivery_path) if delivery_path.exists() else None
    correction = _make_event(transaction, before, after, binding, hashes, evidence, acceptance, event, programme)
    # Scope is resolved from registered paths, never a global shared feedback DB.
    data = AssetStore(project).data / 'events' / event / 'feedback'
    data.mkdir(parents=True, exist_ok=True)
    store = Store(data / 'feedback.sqlite')
    try:
        restore_manual_cases(store, project, event)
        if cached_receipt and cached_receipt.get('delivery_key') == delivery_key and cached_receipt.get('delivered') is True:
            return cached_receipt
        case_id = correction['event_id']
        old = store.get_case(case_id)
        if old and old['status']=='retracted' and old.get('provenance',{}).get('delivery_key')==delivery_key:
            receipt=dict(schema=SCHEMA,delivery_key=delivery_key,delivered=True,status='superseded_by_undo',
                superseded_by_transaction_id=old['provenance'].get('undone_by'),
                case_ref={key:old[key] for key in ('case_id','version','content_sha256')},
                model_accuracy_inferred=False,new_frame_visual_improvement_verified=False)
            _write(delivery_path,receipt,project)
            return receipt
        acceptance_state = correction['source_observations_accepted']
        version = old['version'] if old and old.get('provenance', {}).get('delivery_key') == delivery_key else (old['version'] + 1 if old else 1)
        case = dict(case_id=case_id, version=version, status='active',
            title=f"原生第 {transaction['index']} 帧人工修改：{len(correction['deleted_points'])} 点撤回，{len(correction['added_points'])} 点加入",
            scope=dict(event_id=event, source_video_sha256=correction['source']['video_sha256'], camera_id=None,
                       entity_ids=sorted({p['target_id'] for p in correction['deleted_points'] + correction['added_points'] if p['target_id']})),
            tags=['manual_edit', 'native_source', 'reason_unknown'],
            assertions=dict(source_edit=correction, position=dict(pixel_ground_truth=False, measurement_uncertainty='unknown'),
                negative_supervision=dict(background_negative_created=False, whole_entity_rejected=False),
                acceptance=dict(source_observations_accepted=acceptance_state, geometry_accuracy_inferred=False),
                counterexample='Only these exact source point associations changed; no other frame or entity is rejected.'),
            provenance=dict(authority='human_editor_transaction', transaction_id=transaction['id'],
                source=correction['source'], delivery_key=delivery_key, derivation_is_new_human_pixel_review=False),
            evidence_refs=evidence, split_role='development',
            lineage_ids=['capture:' + correction['source']['video_sha256']])
        # Keep long-lived JSON and regression inputs alongside event facts.
        token = hashlib.sha256(transaction['id'].encode()).hexdigest()[:16]
        permanent = project / 'events' / event / 'feedback' / 'manual'
        case_path = permanent / f'{token}.v{version}.json'
        case_ref = _immutable(case_path, case, project)
        regression = dict(schema='rinkspace.feedback-regression.v1', case_id=case_id, version=version,
            source=correction['source'], evidence_refs=evidence,
            expected=dict(removed_point_ids=[p['point_id'] for p in correction['deleted_points']],
                          added_point_ids=[p['point_id'] for p in correction['added_points']],
                          retained_points_unchanged=True, source_observations_accepted=acceptance_state,
                          reason='unknown', background_negative_created=False, model_accuracy_inferred=False),
            original_problem_and_counterexample=True, new_frame_visual_improvement_verified=False)
        regression_ref = _immutable(permanent / f'{token}.v{version}.regression.json', regression, project)
        published = store.upsert_case(case)
        retracted = None
        if correction['undo_of']:
            parent = store.get_case('human_edit:' + correction['undo_of'])
            if parent and parent.get('provenance', {}).get('undone_by') != transaction['id']:
                for field in ('content_sha256', 'content_hash', 'latest_recorded_version'): parent.pop(field, None)
                parent.update(status='retracted', version=parent['version'] + 1)
                parent['provenance']['undone_by'] = transaction['id']
                parent['evidence_refs'] = [*parent['evidence_refs'], _ref(transaction_path)]
                retracted = store.upsert_case(parent)
                _immutable(permanent / f"{hashlib.sha256(correction['undo_of'].encode()).hexdigest()[:16]}.v{parent['version']}.json", parent, project)
        receipt = dict(schema=SCHEMA, delivery_key=delivery_key, delivered=True,
            transaction=paths['transaction'].as_posix(), case_ref={k: published[k] for k in ('case_id', 'version', 'content_sha256')},
            durable_case=case_ref, regression=regression_ref,
            rollback_case_ref={k: retracted[k] for k in ('case_id', 'version', 'content_sha256')} if retracted else None,
            source_observations_accepted=acceptance_state, model_accuracy_inferred=False,
            new_frame_visual_improvement_verified=False)
        _write(delivery_path, receipt, project)
        return receipt
    finally:
        store.close()


def apply_editor_role_labels(registry, model_path, project):
    """Apply current event role facts to labels without moving source UV."""
    try:
        event, _ = _scope({}, model_path, project)
    except ValueError:
        return
    for path in sorted((Path(project) / 'events' / event / 'model_constraints').glob('*.json')):
        facts = _read(path)
        if (facts.get('schema') != 'rinkspace.user-confirmed-ice-measurement-roles.v1'
                or facts.get('event_id') != event):
            continue
        prior = facts.get('prior_C0_confirmation')
        if prior:
            if _ref(AssetStore(project).resolve(prior['path'], must_exist=True))['sha256'] != prior['sha256']:
                raise ValueError('Current event role facts prior authority SHA changed')
        for definition in facts.get('definitions', {}).values():
            if definition.get('measurement_operator') != 'paint_center': continue
            identities = set(definition.get('ids', []))
            for descriptor in registry.get('targets', {}).values():
                if (descriptor.get('kind') == 'ice_curve'
                        and descriptor.get('curve_id', '').split(':')[0] in identities):
                    descriptor['effective_measurement_operator'] = 'paired_material_boundary_midline'
                    descriptor['effective_role_authority'] = _ref(path)


def restore_installation_cases(store, project, event_id):
    """Restore source installation receipts; these are not visual truth labels."""
    cases = [_read(path) for path in (Path(project) / 'events' / event_id / 'feedback' / 'installs').glob('*.case.json')]
    return [store.upsert_case(case) for case in sorted(cases, key=lambda row: (row['case_id'], row['version']))]


def record_source_installation(project, packet_path, contract):
    """Journal an existing successful runtime transaction after its own checks.

    This records installation state and exact evidence. It does not certify the
    candidate's pixel accuracy, qualification/compaction completion, or promote
    it to the programme's production dataset. Replay is keyed by packet SHA.
    """
    from rink_feedback.core import Store
    project=Path(project).resolve();packet_path=Path(packet_path).resolve()
    contract=deepcopy(contract);contract.pop('feedback_outbox',None)
    assets=AssetStore(project)
    packet=_read(packet_path)
    if (contract.get('packet_sha256')!=_ref(packet_path)['sha256']
            or contract.get('active_FIT_source_UV_and_optical_support_explicitly_revised') is not True):
        raise ValueError('Source installation contract does not bind the successful packet')
    rows=contract.get('corrected_rows',[])
    if not rows:raise ValueError('Successful source installation has no exact row history')
    sample=dict(video_sha256=contract['source_video_sha256'],pts=int(rows[0]['locator']['group_actual_pts']),
                time_base=contract['time_base'],coordinate_domain='native_pixels')
    registered,registration=registered_source_dataset(project,packet_path,sample)
    if not registered:
        return dict(status='diagnostic_packet_not_published',runtime_rows_installed=True,
                    observation_dataset_promoted=False,visual_accuracy_verified=False,
                    registration=registration,production_feedback_index_consulted=False)
    event,programme=registration['event_id'],registration['programme_id']
    digest=contract['packet_sha256']
    with Store(assets.data/'events'/event/'feedback'/'feedback.sqlite') as store:
        restore_installation_cases(store,project,event)
        previous=store.get_case('source_install:'+digest)
        if previous and previous['status']=='retracted':
            return dict(status='retracted_terminal_no_republish',runtime_rows_installed=True,
                case_ref={key:previous[key] for key in ('case_id','version','content_sha256')},
                observation_dataset_promoted=False,visual_accuracy_verified=False)
    folder=assets.data/'events'/event/'feedback'/'install_outbox'
    pending_path=folder/(digest+'.json')
    journal=dict(schema='rinkspace.source-installation-feedback-outbox.v1',
        status='runtime_rows_installed',packet_reference=_ref(packet_path),
        contract=contract,source=sample,event_id=event,programme_id=programme,
        source_installation_is_not_formal_dataset_promotion=True,visual_accuracy_verified=False)
    # This durable input exists before touching the fallible derived index.
    _immutable(pending_path,journal,project)
    evidence=[dict(_ref(packet_path),kind='installed_source_revision'),
              dict(contract['original_absolute_packet'],kind='original_observation_snapshot'),
              *[dict(ref,kind='native_image') for ref in contract['source_evidence']]]
    permanent=project/'events'/event/'feedback'/'installs'/(digest[:20]+'.v1.case.json')
    summaries=[dict(locator=row['locator'],point_indices=row['point_indices'],
                    physical_edge_role=row['physical_edge_role'],measurement_role=row.get('measurement_role'),
                    retired_original_point_indices=row.get('retired_original_point_indices',[]),
                    retired_points_require_verified_physical_input_compaction=row.get('retired_points_require_verified_physical_input_compaction',False))
               for row in rows]
    case=dict(case_id='source_install:'+digest,version=1,status='active',
        title='已认证原片观测修订已进入运行状态；精度和完整安装链另行验收',
        scope=dict(event_id=event,source_video_sha256=sample['video_sha256'],camera_id=None,entity_ids=[]),
        tags=['source_installation','exact_observation_version','visual_accuracy_unknown'],
        assertions=dict(installation=dict(packet_reference=_ref(packet_path),rows=summaries,
            runtime_rows_installed=True,qualification_and_compaction_completion_not_inferred=True,
            formal_dataset_promotion=False,applicability='exact_source_snapshot_and_locators_only'),
            position=dict(pixel_ground_truth=False,current_frame_positions_must_be_read_from_original_RGB=True),
            counterexample='Installing a bounded correction does not establish new-frame visual improvement or model accuracy.'),
        provenance=dict(authority='existing_source_observation_installer',source=sample,
            implementation='original_ice_pixel_correction.install_original_ice_pixel_correction'),
        evidence_refs=evidence,split_role='development',lineage_ids=['capture:'+sample['video_sha256']])
    case_ref=_immutable(permanent,case,project)
    regression=dict(schema='rinkspace.source-installation-regression.v1',case_id=case['case_id'],
        packet_reference=_ref(packet_path),rows=summaries,evidence_refs=evidence,
        required_checks=['immutable_original_snapshot','native_clock_and_exact_locators',
                         'same_registered_physical_material','qualified_retirement_and_compaction_if_declared'],
        actual_installer_must_be_reused=True,new_frame_visual_improvement_verified=False)
    regression_ref=_immutable(permanent.with_name(permanent.name.replace('.case.json','.regression.json')),regression,project)
    with Store(assets.data/'events'/event/'feedback'/'feedback.sqlite') as store:
        restore_installation_cases(store,project,event)
        saved=store.upsert_case(case)
    receipt=dict(status='feedback_indexed',runtime_rows_installed=True,
        case_ref={key:saved[key] for key in ('case_id','version','content_sha256')},
        durable_case=case_ref,regression=regression_ref,outbox=_ref(pending_path),
        observation_dataset_promoted=False,visual_accuracy_verified=False)
    _write(folder/(digest+'.receipt.json'),receipt,project)
    return receipt


def replay_source_installation_outbox(project,event_id):
    """Retry successful runtime journals after an index/publication outage."""
    directory=AssetStore(project).data/'events'/event_id/'feedback'/'install_outbox'
    results=[]
    from rink_feedback.core import Store
    with Store(AssetStore(project).data/'events'/event_id/'feedback'/'feedback.sqlite') as store:
        restore_installation_cases(store,project,event_id)
        for path in directory.glob('*.json'):
            journal=_read(path)
            if journal.get('schema')!='rinkspace.source-installation-feedback-outbox.v1':continue
            case=store.get_case('source_install:'+journal['packet_reference']['sha256'])
            if case and case['status']=='retracted':
                results.append(dict(status='retracted_terminal_no_republish',outbox=str(path),
                    case_ref={key:case[key] for key in ('case_id','version','content_sha256')},
                    runtime_rows_reinstalled=False,observation_dataset_promoted=False))
                continue
            try:
                reference=journal['packet_reference']
                if _ref(reference['path'])['sha256']!=reference['sha256']:
                    raise ValueError('Installed source packet bytes changed; original journal retained')
                results.append(record_source_installation(project,reference['path'],journal['contract']))
            except Exception as error:
                results.append(dict(status='retry_pending',outbox=str(path),error=type(error).__name__+': '+str(error),
                    runtime_rows_reinstalled=False,observation_dataset_promoted=False))
    return results


def verify_manual_regressions(project, event_id):
    """Recheck actual saved source deltas. This is not a new-frame vision test."""
    project=Path(project).resolve()
    permanent=project/'events'/event_id/'feedback'/'manual'
    latest={}
    for path in permanent.glob('*.v*.json'):
        if path.name.endswith('.regression.json'):continue
        case=_read(path)
        if case['version']>latest.get(case['case_id'],{}).get('version',0):latest[case['case_id']]=case
    results=[]
    for case in latest.values():
        if case['status']=='retracted':
            results.append(dict(case_id=case['case_id'],version=case['version'],status='retracted_excluded'))
            continue
        correction=case['assertions']['source_edit']
        refs={ref['kind']:ref for ref in case['evidence_refs']}
        for ref in refs.values():
            if _ref(ref['path'])['sha256']!=ref['sha256']:
                raise ValueError('Manual regression source evidence changed: '+ref['path'])
        before=_read(refs['before_packet']['path']);after=_read(refs['after_packet']['path'])
        rerun=_make_event(_read(refs['transaction']['path']),before,after,_read(refs['binding']['path']),
            correction['file_hashes'],case['evidence_refs'],_read(refs['acceptance']['path']) if 'acceptance' in refs else None,
            event_id,correction['source'].get('programme_id'))
        if rerun!=correction:raise ValueError('Manual regression no longer preserves the committed source edit')
        results.append(dict(case_id=case['case_id'],version=case['version'],status='passed',
            exact_snapshot_delta_rechecked=True,source=correction['source'],
            input_evidence_refs=case['evidence_refs'],new_frame_visual_improvement_verified=False))
    report=dict(schema='rinkspace.manual-feedback-regression-receipt.v1',results=results,
                checked_active_cases=sum(row['status']=='passed' for row in results),
                new_frame_visual_improvement_verified=False)
    path=AssetStore(project).data/'events'/event_id/'feedback'/'manual_regression_receipt.json'
    return dict(report,receipt=_write(path,report,project))
