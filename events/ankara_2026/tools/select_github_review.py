"""Select the frozen 2026-10-19 technical review; preserve no chat transcripts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory-zip', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    with zipfile.ZipFile(a.inventory_zip) as z:
        index = json.loads(z.read('FILE_INDEX.json'))['files']
        status = json.loads(z.read('CURRENT_STATUS.json'))
    candidate = json.loads(a.candidate.read_text(encoding='utf-8'))
    rows = {r['archive_path']: dict(r) for r in candidate['records']}
    for r in index:
        logical = r['archive_path']
        include = (
            logical.startswith(('Data/runs/okada_rec_1019/scope_g4/',
                'Data/runs/okada_rec_1019/hist_short_v1/ring_history_qualified/',
                'Data/runs/okada_rec_1019/hist_s09_v1/qualified_ring/',
                'Data/runs/okada_rec_1019/hist_s09_v1/qualified_C0_chain_v2/',
                'Data/runs/okada_rec_1019/hist_s07_v1/chain_qualified_v3/',
                'Initial_v98/run/', 'Initial_v98/evidence/run/',
                'feedback/', 'Evidence/user/')) or
            logical in ('Data/runs/okada_joint9_1018a/roles/frame_roles.json',
                'Data/runs/okada_rec_1019/perf/final_summary.json',
                'direct_grid/original_sources/materialize_direct_screen_grid.py'))
        if include and r['original_path'].lower().startswith(('d:', 'c:')):
            rows.setdefault(logical, dict(r) | {'selection_role': 'source_qualification_or_baseline_evidence'})
    rec = a.inventory_zip.parents[1] / 'runs' / 'okada_rec_1019'
    for source in sorted(rec.glob('*.py')):
        value = source.read_bytes()
        logical = 'Data/runs/okada_rec_1019/' + source.name
        rows.setdefault(logical, {'archive_path': logical, 'original_path': str(source),
            'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest(),
            'role': 'preserved_event_recovery_driver', 'selection_role': 'current_recovery_source'})
    assets = []
    for logical, row in sorted(rows.items()):
        if any(x in logical for x in ('VISIBLE_CONVERSATION', 'RELATED_FEEDBACK_CHAT', 'thread_history')):
            continue
        source = Path(row['original_path'])
        if not source.is_file():
            raise FileNotFoundError(source)
        role = row['selection_role']
        if role == 'current_typed_array_or_W':
            group = 'source_arrays' if '/g9_ice/' in logical else 'weights'
        elif role == 'current_source_packet_and_certificate':
            group = 'observations'
        elif role in ('latest_full_failed_state_and_comparison', 'actual_initial_state_origin'):
            group = 'state'
        elif role in ('actual_backend_source_recipe', 'current_recovery_source') or source.suffix == '.py':
            group = 'code'
        elif logical.startswith('Initial_v98/'):
            group = 'baseline'
        else:
            group = 'evidence'
        assets.append({'logical_path': logical, 'original_path': str(source),
                       'bytes': row['bytes'], 'sha256': row['sha256'], 'role': role, 'group': group})
    result = {'snapshot': 'okada-full9-stopped-20261019', 'status': status,
        'scope': 'Current complete failed state, original observations/W and eligibility evidence, S01/Demo baselines, v98 input failure, source code and selected real pixels. Does not certify a successful fit or complete-video accuracy.',
        'source_array_and_W_unique_references': 1622,
        'assets': assets,
        'reused_code': ['calibration/rinkfit/', 'calibration/program_tracking/', 'calibration/solver_runtime/',
                        'src/rinkspace/', 'src/rink_feedback/', 'environments/', 'events/ankara_2026/feedback/'],
        'excluded': ['Chat transcripts', 'original full videos', 'rendered review videos',
                     'installed environments', 'compiler scratch/obj/pdb', 'bulk decoded frames',
                     '4.7 GB reproducible screen-grid arrays', 'unselected intermediate full checkpoints']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    groups = {}
    for row in assets:
        g = groups.setdefault(row['group'], {'files': 0, 'raw_bytes': 0})
        g['files'] += 1; g['raw_bytes'] += row['bytes']
    print(json.dumps({'assets': len(assets), 'groups': groups}, ensure_ascii=False))


if __name__ == '__main__':
    main()
