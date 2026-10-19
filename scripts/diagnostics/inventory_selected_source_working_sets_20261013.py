"""Inventory selected strict source deltas; never mutate or attach solver inputs."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.source_working_set_inventory import source_working_set_inventory
import calibration.program_tracking.source_working_set_inventory as inventory_implementation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous inventory')
    manifest_sha = file_sha(args.selected_references)
    selected = json.loads(args.selected_references.read_text(encoding='utf8'))
    records = []
    for record in selected['records']:
        ref = record['absolute_inputs']
        pack = StrictAbsoluteSourcePack(ref['path'], expected_sha256=ref['sha256'])
        result = source_working_set_inventory(pack)
        if result['raw_points']!=record['pixels'] or result['raw_rows']!=record['rows']:
            raise ValueError('Selected source counts differ from authenticated pack')
        records.append(dict(cut=record['cut'], source_reference=ref, inventory=result))
        print(json.dumps(dict(cut=record['cut'], pack=Path(ref['path']).name,
            points=result['raw_points'], exact_duplicates=result['exact_duplicate_source_pixel_and_physical_target_occurrences'],
            material_exposure_groups=result['physical_material_exposure_groups'])), flush=True)
        del pack
        gc.collect()
    if file_sha(args.selected_references)!=manifest_sha:
        raise ValueError('Selected input references changed during read-only inventory')
    report = dict(schema='rinkspace.selected-strict-source-deltas-working-set-inventory.v1',
        selected_source_references=dict(path=str(args.selected_references.resolve()), sha256=manifest_sha),
        records=records,
        raw_points=sum(r['inventory']['raw_points'] for r in records),
        raw_rows=sum(r['inventory']['raw_rows'] for r in records),
        per_pack_exact_duplicates=sum(r['inventory']['exact_duplicate_source_pixel_and_physical_target_occurrences'] for r in records),
        original_source_packs_changed=False, solver_working_set_created=False,
        weights_installed=False, fitting_started=False,
        per_pack_duplicate_counts_do_not_replace_cross_pack_public_merge=True,
        inventory_implementation=dict(path=str(Path(inventory_implementation.__file__).resolve()),
            sha256=file_sha(inventory_implementation.__file__)),
        diagnostic_implementation=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(output=str(args.output),points=report['raw_points'],rows=report['raw_rows'],
        per_pack_duplicates=report['per_pack_exact_duplicates'],solver_started=False)), flush=True)


if __name__=='__main__':
    main()
