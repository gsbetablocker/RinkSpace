"""Preserve source-qualified badge exterior readings at one uniform spacing.

Pattern identity and closed-outline RGB qualification come from the independent
capture. This adapter cannot turn an unorganized candidate into an observation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

from capture_independent_materials import spaced_indices
from rinkspace.path_policy import guard_project_write


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qualify(capture, poses, output, evidence):
    capture, poses, output, evidence = map(Path, (capture, poses, output, evidence))
    output = guard_project_write(output.resolve())
    if output.exists():
        raise FileExistsError(output)
    source = capture/'source_rows.jsonl'
    registration = {r['pts']: r for r in map(json.loads, (capture/'registration.jsonl').read_text().splitlines())}
    folder = output/'s07'
    folder.mkdir(parents=True)
    totals = dict(frames=0, qualified_frames=0, rows=0)
    with (folder/'source_rows.jsonl').open('w', encoding='utf8') as stream:
        for record in map(json.loads, source.read_text().splitlines()):
            note = registration[record['pts']]
            public = [r for r in record['pending_rows'] if r['existing_closed_outline_organized']]
            if public and not note['source_pattern_check']['supported']:
                raise ValueError('An organized contour has no supported source pattern')
            selected = set(map(int, spaced_indices([r['uv'] for r in public])))
            rows = []
            for i, original in enumerate(public):
                if i not in selected:
                    continue
                if original['uv'] != original['actual_source_quantiles'][0][1]:
                    raise ValueError('Measured UV no longer equals the actual RGB interface quantile')
                rows.append(dict(original, fit_eligible=True,
                    actual_outer_boundary_or_internal_letter_review_pending=False,
                    physical_instance_identity_review_pending=False,
                    review_state='actual_source_exterior_and_closed_native_shape_supported',
                    source_identity_basis='independent Ad9 CAN red badge reference; known main rig and far-board physical order',
                    source_component_topology='red source flood exterior; internal lettering is not an exterior',
                    visual_review_evidence=str(evidence.resolve()), visual_review_evidence_sha256=sha(evidence),
                    original_pending_capture=str(source.resolve()), original_pending_capture_sha256=sha(source)))
            result = dict(record, rows=rows, fit_eligible=bool(rows),
                review_state='qualified_source_outline' if rows else 'pending_no_qualified_source_outline',
                public_reader_count_before_spacing=len(public), final_source_spacing_px=48.)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            totals['frames'] += 1
            totals['qualified_frames'] += bool(rows)
            totals['rows'] += len(rows)
    shutil.copyfile(poses, folder/'poses.npz')
    (output/'qualification.json').write_text(json.dumps(dict(
        source_capture=str(capture.resolve()), source_sha256=sha(source),
        evidence=str(evidence.resolve()), evidence_sha256=sha(evidence),
        original_candidates_preserved=True, source_UV_unchanged=True,
        navigation_poses_copied_only=True, counts=totals), indent=2)+'\n', encoding='utf8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    qualify(a.capture, a.poses, a.output, a.evidence)
