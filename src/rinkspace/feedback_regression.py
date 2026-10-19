"""Recheck real feedback evidence and execution receipts without optimizing."""
from __future__ import annotations
import json
from pathlib import Path

from rink_feedback.core import ValidationError, source_key
from .feedback_runtime import ROOT, atomic_json, authenticate_refs, feedback_store, native_source
from .feedback_outbox import verify_manual_regressions, replay_source_installation_outbox


def verify_feedback_regressions(event_id, *, project=ROOT, output):
    manual = verify_manual_regressions(project, event_id)
    installations = replay_source_installation_outbox(project, event_id)
    rows = []
    with feedback_store(project, event_id) as store:
        versions = store.db.execute('SELECT case_id,version FROM cases ORDER BY case_id,version').fetchall()
        for cid, version in versions:
            case = store.get_case(cid, version)
            if case.get('kind') != 'recipe':continue
            current = store.get_case(cid)
            if current['version'] != version:continue
            if current['status'] != 'active':
                rows.append(dict(case_id=cid, version=version, status='retracted_excluded'));continue
            authenticate_refs(case['evidence_refs'], project)
            finding_id = case['provenance']['finding_id']
            try:
                finding = store.get_finding(finding_id)
            except ValidationError as error:
                if str(error) != 'unknown finding_id: ' + finding_id:
                    raise
                # Cases can rebuild the searchable index without reconstructing
                # immutable discovery history. Preserve that missing history.
                finding = None
            subjects = set(finding['observation_ids']) if finding is not None else set()
            # Parse only the registered actual_geometry wrapper. Opaque row IDs
            # from other producers retain their full identity.
            native_subjects = {name.partition('::')[2] if name.startswith('actual_geometry:') and '::' in name else name
                for name in subjects}
            original_checks = []
            for ref in case['evidence_refs']:
                if not ref['path'].endswith('review.json'):continue
                review = json.loads(Path(ref['path']).read_text('utf-8-sig'))
                if source_key(native_source(review['source_binding'])) != source_key(case['assertions']['episode_source']):
                    raise ValueError('Original regression review belongs to a different native source/frame')
                membership = review.get('physical_membership_rows', [])
                original_checks += [r for r in membership if r['source_row_id'] in native_subjects]
            uses = []
            for dbrow in store.db.execute('SELECT body FROM uses'):
                use = json.loads(dbrow[0])
                if use['case_id'] != cid or use['version'] != version:continue
                authenticate_refs(use['tool_evidence_refs'], project)
                for ref in use['tool_evidence_refs']:
                    receipt = json.loads(Path(ref['path']).read_text('utf-8-sig'))
                    if source_key(receipt['source']) != source_key(use['source']):
                        raise ValueError('A use receipt points at another exact source')
                    effects = [e for e in receipt['effects'] if e['case_id'] == cid]
                    if not effects:raise ValueError('Claimed case use has no executed reader effect')
                    uses.append(dict(use_id=use['use_id'], source=use['source'],
                        receipt=ref, effects=effects,
                        different_native_frame=source_key(use['source']) != source_key(case['assertions']['episode_source'])))
            expected = (case.get('assertions', {}).get('lesson', {}).get('reader_check') == 'ordered_board_material_flanks')
            matched_subjects = {r['source_row_id'] for r in original_checks}
            complete_subject_coverage = bool(native_subjects) and matched_subjects == native_subjects
            if finding is None:
                regression = 'unknown_missing_finding'
            elif expected and complete_subject_coverage and all(r['status'] == 'wrong_role' for r in original_checks):
                regression = 'passed_original_wrong_role_preserved'
            else:
                regression = 'unknown_original_subject_or_expected_check'
            rows.append(dict(case_id=cid, version=version, status=regression,
                finding_id=finding_id, finding_history_available=finding is not None,
                finding_history_reconstructed=False,
                missing_history_reason=('The derived index lacks the original finding; restoring a case does not restore immutable discovery history.'
                    if finding is None else None),
                original_subject_coverage=dict(expected_native_ids=sorted(native_subjects),
                    matched_native_ids=sorted(matched_subjects), complete=complete_subject_coverage),
                original_subject_ids=sorted(subjects), original_checks=[dict(source_row_id=r['source_row_id'],
                    status=r['status'], current_uv=r['current_uv']) for r in original_checks],
                verified_dimensions=case['assertions']['verified_dimensions'],
                unresolved_dimensions=case['assertions']['unresolved_dimensions'], actual_reader_uses=uses,
                later_frame_consumption_verified=any(u['different_native_frame'] for u in uses),
                independent_new_frame_visual_accuracy_improvement_verified=False))
    report = dict(schema='rinkspace.production-feedback-regressions.v1', event_id=event_id,
        manual_snapshot_regression=manual, installation_outbox_replay=installations,
        learned_procedure_regression=rows, optimization_started=False,
        new_frame_visual_accuracy_improvement_verified=False,
        meaning='Checks original source decisions and actual case-use receipts; these do not establish reduced visual error or whole-shot accuracy.')
    return dict(report, receipt=atomic_json(output, report))
