"""Explicit current-native review of an existing production board frame task.

Reuse the native video reader, exact readout aliases, source journal and mask.
This creates review/resolution evidence, never a solver packet or a certificate.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

from .feedback_runtime import (ROOT, SourceFeedbackRead, authenticate_refs, atomic_json,
    feedback_store, file_sha, learn_source_review, native_source)


def _ref(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def _checked(reference):
    reference = dict(reference)
    reference['path'] = reference.get('path', reference.get('file'))
    authenticate_refs([reference])
    return Path(reference['path'])


def review_existing_board_frame(run, pts, output, *, device='cpu', predecessor_findings=()):
    import cv2
    import numpy as np
    from PIL import Image
    from rink_feedback.cycle import FeedbackCycle
    from rinkfit_joint.io_utils import read_json, read_jsonl, object_sha256
    from rinkfit_joint.native_source import iter_training_native_rgb
    from rinkfit_joint.board_source_requalification import BoardMaterialRoleRequalifier
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():raise ValueError('Use a new immutable feedback review directory')
    # This public reader is for existing frame tasks; it never starts a scope scan.
    assets = __import__('rinkspace.storage', fromlist=['AssetStore']).AssetStore(ROOT)
    if not output.is_relative_to(assets.data):raise ValueError('Runtime review evidence belongs on Data')
    scopepath = run / 'board_physical_requalification_scope_v1' / 'manifest.json'
    scope = read_json(scopepath)
    task = next((t for t in read_jsonl(_checked(scope['frame_tasks'])) if int(t['native_pts']) == pts), None)
    if task is None:raise ValueError('PTS is outside the existing explicit development review scope')
    journals = read_json(_checked(scope['source_journals']))
    journal = _checked(journals[task['source_journal_index']]['source_journal'])
    rows = [r for r in read_jsonl(journal) if int(r['native_pts']) == pts]
    if len(rows) != task['raw_rows']:raise ValueError('Exact current source journal row count differs')
    aliasindex = read_json(_checked(scope['source_alias_authority']))
    aliases = {a['source_row_id'] for a in read_jsonl(_checked(aliasindex['aliases'])) if int(a['native_pts']) == pts}
    canonical = [r for r in rows if r['source_row_id'] not in aliases]
    if len(canonical) != task['canonical_rows']:raise ValueError('Exact current readout aliases differ')
    proofpath = _checked(dict(path=task['source_packet'], sha256=task['source_packet_sha256']))
    proof = read_json(proofpath)
    if any(r['source_packet'] != str(proofpath) or r['source_packet_sha256'] != task['source_packet_sha256'] for r in rows):
        raise ValueError('One frame needs explicit matching source proof for every row')
    _, rgb, decode = next(iter_training_native_rgb(run, [pts]))
    if proof['native_pts'] != pts or proof['decoded_native_BGR_sha256'] != decode['decoded_native_BGR_sha256']:
        raise ValueError('Current decoder differs from original source frame proof')
    maskref = dict(proof['current_mask']); maskpath = _checked(maskref)
    mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != rgb.shape[:2]:raise ValueError('Current source mask dimensions differ')
    output.mkdir(parents=True)
    Image.fromarray(rgb).save(output / 'native.png')
    native = _ref(output / 'native.png')
    context = dict(native_video_decode_receipt=decode, current_mask_ref=dict(path=str(maskpath), sha256=file_sha(maskpath)),
        evidence_directory=str(output / 'profiles'), feedback_run_id=output.name)
    before = object_sha256(rows)
    engine = BoardMaterialRoleRequalifier(ROOT, device=device, use_frozen_visual_model=False)
    result = engine.evaluate(rgb, mask > 0, canonical, context)
    if before != object_sha256(rows):raise RuntimeError('Feedback review mutated original readouts')
    resultref = atomic_json(output / 'review.json', result)
    source = native_source(dict(decode, native_hw=list(rgb.shape[:2])))
    event, programme = result['source_feedback']['source'].get('event_id'), None
    from .feedback_runtime import resolve_scope
    event, programme = resolve_scope(source)
    native_evidence = dict(native, kind='native_pixels', source=source,
        independent_of_model=True, current_frame=True)
    resolutions, discoveries = [], []
    # Validate links before declaring a checkpoint-to-review chain. A matching
    # PTS alone cannot substitute for the original observation identity.
    from rink_feedback.core import source_key
    subjects = {}
    with feedback_store(ROOT, event) as store:
        ids = {r['source_row_id'] for r in canonical}
        for predecessor in predecessor_findings:
            previous = store.get_finding(predecessor)
            if source_key(previous['source']) != source_key(source):
                raise ValueError('Predecessor belongs to another native source/frame')
            linked = {name if name in ids else name.rpartition('::')[2]
                for name in previous.get('observation_ids', [])}
            linked &= ids
            if not linked:raise ValueError('Predecessor has no exact current observation subject')
            subjects[predecessor] = linked
    wrong = [r for r in result['physical_membership_rows'] if r['status'] == 'wrong_role']
    for row in wrong:
        # Source profile can establish a role conflict without establishing a
        # replacement UV or a complete physical-instance/visibility decision.
        finding = dict(source=source, event_id=event, programme_id=programme, shot_id=decode['shot_id'],
            entity_ids=[row['material_role']], observation_ids=[row['source_row_id']],
            signal_type='operator_conflict', problem_key='current_board_material_role_conflict',
            tags=['ice_contact', 'board_foot', 'material_boundary'],
            predecessor_finding_ids=[key for key, ids in subjects.items() if row['source_row_id'] in ids],
            evidence_refs=[native_evidence, resultref], metrics=dict(status='wrong_role', reasons=row['reason_codes']))
        with feedback_store(ROOT, event) as store:
            checkpoint = FeedbackCycle(store).checkpoint(output.name, [finding])
        discoveries.extend(checkpoint['tasks'])
        checks = dict(measurement_operator=dict(status='verified', value='ordered_board_material_flanks',
            evidence_refs=[native['sha256']], reasons=row['reason_codes']),
            semantic_class=dict(status='unknown', reason='A wrong boundary role does not by itself establish pixel material class'),
            physical_instance=dict(status='unknown'), visibility=dict(status='unknown'), position=dict(status='unknown'))
        related = [key for key, ids in subjects.items() if row['source_row_id'] in ids]
        for reviewed_finding in [checkpoint['tasks'][0]['finding_id'], *related]:
            resolutions.append(learn_source_review(event_id=event, finding_id=reviewed_finding,
                source=source, checks=checks, evidence_refs=[native_evidence, resultref, result['frame_profile_evidence']],
                resolution_type='operator_error', lesson=dict(summary='Compare current board material flanks; keep wrong-role means outside fit until the original operator is requalified',
                    reader_check='ordered_board_material_flanks', original_49Q_operator_requalification_still_required=True)))
    summary = dict(schema='rinkspace.production_feedback_frame_review.v1', source=source,
        native_decode_receipt=decode, native_image=native, existing_frame_task=task,
        original_journal=_ref(journal), old_row_bytes_unchanged=True, review=resultref,
        source_feedback=result['source_feedback'], counts=result['counts'], discoveries=discoveries,
        resolutions=resolutions, predecessor_finding_ids=list(predecessor_findings),
        new_observation_installed=False, raw_mu_Query_W_source_flags_unchanged=True,
        original_operator_requalification_required=True, training_started=False,
        whole_shot_or_new_frame_accuracy_improvement_verified=False)
    atomic_json(output / 'receipt.json', summary)
    return summary
