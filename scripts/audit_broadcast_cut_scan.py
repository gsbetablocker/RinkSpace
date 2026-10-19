"""Check continuity and provenance of external cut scans and sequence boards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_scan(root: Path):
    index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    chunks = [json.loads(Path(row['scan_path']).read_text(encoding='utf-8')) for row in index['chunks']]
    issues = []
    for left, right in zip(chunks, chunks[1:]):
        if left['core_end_pts'] != right['core_start_pts']:
            issues.append(f"gap/overlap {left['core_end_pts']} vs {right['core_start_pts']}")
    if not chunks or chunks[0]['core_start_pts'] != index['range_pts'][0] or chunks[-1]['core_end_pts'] != index['range_pts'][1]:
        issues.append('chunk union does not cover requested range')
    if not index.get('complete'):
        issues.append('scan index not marked complete')
    if len({chunk['source_sha256'] for chunk in chunks}) != 1 or any(chunk['source_sha256'] != index['source_sha256'] for chunk in chunks):
        issues.append('source hash mismatch')
    candidate_count = 0
    samples = []
    for chunk in chunks:
        if chunk['time_base'] != '1/15360' or chunk['authority'] != 'numeric_cut_proposals_only_no_camera_identity':
            issues.append(f"unexpected metadata {chunk['core_start_pts']}")
        if not (chunk['scan_start_pts'] <= chunk['core_start_pts'] < chunk['core_end_pts'] <= chunk['scan_end_pts']):
            issues.append(f"invalid scan margin {chunk['core_start_pts']}")
        expected_proxy_frames = (chunk['scan_end_pts'] - chunk['scan_start_pts']) // 512
        if chunk['decoded_frames'] != expected_proxy_frames or chunk['pts_step_counts'] != {'512': expected_proxy_frames - 1}:
            issues.append(f"proxy frame coverage mismatch {chunk['core_start_pts']}: expected {expected_proxy_frames}, observed {chunk['decoded_frames']} / {chunk['pts_step_counts']}")
        for candidate in chunk['candidates']:
            candidate_count += 1
            if not chunk['core_start_pts'] <= candidate['proposal_pts'] < chunk['core_end_pts']:
                issues.append(f"candidate outside core {candidate['proposal_pts']}")
            if not candidate['before_pts'] < candidate['after_pts'] == candidate['proposal_pts']:
                issues.append(f"invalid before/after PTS {candidate['proposal_pts']}")
            if not Path(candidate['before_image']).is_file() or not Path(candidate['after_image']).is_file():
                issues.append(f"missing candidate image {candidate['proposal_pts']}")
        for sample in chunk['samples']:
            samples.append(sample)
            if not chunk['core_start_pts'] <= sample['source_pts'] < chunk['core_end_pts']:
                issues.append(f"4s sample outside core {sample['source_pts']}")
            if not Path(sample['image_path']).is_file():
                issues.append(f"missing 4s image {sample['source_pts']}")
    expected_4s = {second * 15360 for second in range(0, 19560, 4)
                   if index['range_pts'][0] <= second * 15360 < index['range_pts'][1]}
    actual_4s = {row['source_pts'] for row in samples}
    if actual_4s != expected_4s or len(actual_4s) != len(samples):
        issues.append(f'4s sample coverage mismatch: expected {len(expected_4s)}, observed {len(actual_4s)} unique')
    if candidate_count != index.get('candidate_count'):
        issues.append('candidate count mismatch')
    sequence_dirs = sorted(root.glob('sequence_*'))
    sequences = []
    all_one_second_targets = []
    for directory in sequence_dirs:
        manifest_path = directory / 'sequence.json'
        if not manifest_path.is_file():
            issues.append(f'missing sequence manifest {directory}')
            continue
        sequence = json.loads(manifest_path.read_text(encoding='utf-8'))
        count = sum(len(page['samples']) for page in sequence['pages'])
        if count != sequence['sample_count'] or count != (sequence['range_pts'][1] - sequence['range_pts'][0]) // 15360:
            issues.append(f'incomplete 1s sequence {directory}')
        times = [row['target_second'] for page in sequence['pages'] for row in page['samples']]
        if times != list(range(sequence['range_pts'][0] // 15360, sequence['range_pts'][1] // 15360)):
            issues.append(f'1s targets not contiguous {directory}')
        all_one_second_targets.extend(times)
        if any(not Path(page['page_path']).is_file() for page in sequence['pages']):
            issues.append(f'missing 1s page {directory}')
        sequences.append({'directory': str(directory), 'range_pts': sequence['range_pts'],
                          'sample_count': count, 'page_count': len(sequence['pages'])})
    expected_one_second_targets = list(range(index['range_pts'][0] // 15360, index['range_pts'][1] // 15360))
    if sorted(all_one_second_targets) != expected_one_second_targets:
        issues.append(f'1s sequence coverage mismatch: expected {len(expected_one_second_targets)}, observed {len(all_one_second_targets)}')
    return {'schema_version': 'rinkspace.broadcast_cut_scan_audit.v1',
            'scan_index_path': str(root / 'index.json'), 'range_pts': index['range_pts'],
            'chunk_count': len(chunks), 'candidate_count': candidate_count,
            'existing_4s_sample_refs': len(samples), 'expected_existing_4s_samples': len(expected_4s),
            'one_second_sequences': sequences, 'one_second_sample_count': len(all_one_second_targets),
            'issues': issues, 'pass': not issues,
            'scope': 'structural and file-provenance check only; does not validate cut recall or camera identity'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.scan_root.resolve()
    report = check_scan(root)
    (root / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
