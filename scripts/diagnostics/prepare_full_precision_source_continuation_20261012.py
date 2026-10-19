"""Freeze only the five unfinished cuts, retaining the four completed packs.

Run after the source implementation repairs are complete. Historical source,
authority and successful outputs must still match their recorded bytes; only
the implementation files inside this checkout receive successor hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha

WORK = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/'
    'current56_shared_active_set_filter_20261009/systematic_continuation_20261010/'
    'full_programme_strict_source_review_20261012')
INVENTORY = WORK/'source_inventory'
COMPLETE = {
    '223403520:R0402_035': ('source_full_precision_exact_query_retry_v1',
        'a0cee34e18b129218c5277b8dcc0ca1984c294f7cf3d7b261204ad4af04807ba'),
    '223403520:R0402_036': ('source_full_precision_batch_v3',
        '81834ec7f9584e7332c67aef974ffcf5cfd8b32f70457b78244a2f8492730dc1'),
    '223403520:R0402_037': ('source_full_precision_exact_query_retry_v1',
        '632c1544ca9b99b5787a42753f127578246789d31dfaa9166e8a69dec109f267'),
    '223403520:R0402_039': ('source_full_precision_typed039_retry_v1',
        '89c1d56c96ff57a4b16b89307eb502f6b820be3d6a7166feed819b3fdc0513d0'),
}


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def prepare_continuation(output, *, template_path=None, inventory_path=None):
    template_path = Path(template_path or INVENTORY/'source_full_precision_final_q50_038_retry_FROZEN_v1.json').resolve()
    inventory_path = Path(inventory_path or INVENTORY/'source_full_precision_batch_FROZEN_v3.json').resolve()
    output = None if output is None else Path(output).resolve()
    if output is not None and output.exists():
        raise FileExistsError('Preserve every earlier frozen source continuation')
    template = json.loads(template_path.read_text(encoding='utf8'))
    inventory = json.loads(inventory_path.read_text(encoding='utf8'))
    if (template.get('schema') != 'rinkspace.exact-source-cut-batch.v1'
            or inventory.get('schema') != template['schema']):
        raise ValueError('Frozen source-cut predecessors required')
    if template.get('ice_graph_class') != 'rinkspace.vision_ice_final_precision_graph:PreciseSourceIceMaterialIdentityGraph':
        raise ValueError('Use the explicit public final-source precision graph successor')
    common = template['common_arguments']
    if (common[common.index('--native-alias-review')+1] != 'rinkspace.native_alias_exact_query_review:review_native_alias'
            or common[common.index('--board-reader')+1] != 'rinkspace.vision_board_typed_material_clearance:TypedMaterialBoardSourceReader'):
        raise ValueError('Retain the exact-query alias and typed physical-board reader interfaces')
    identity_path = Path(common[common.index('--identity')+1])
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    authenticate_source_reference(dict(path=identity['camera_path'], sha256=identity['camera_sha256']))
    with np.load(identity['camera_path'], allow_pickle=False) as camera:
        native = camera['pts'].copy()
        actual = camera['exposure_representative_pts'].copy()
    references = {}
    changed_code = []
    for original in template['references']:
        reference = dict(original)
        path = Path(reference['path']).resolve()
        if path.is_relative_to(ROOT) and path.suffix == '.py':
            reference['sha256'] = file_sha(path)
            if reference['sha256'] != original['sha256']:
                changed_code.append(dict(path=str(path), predecessor_sha256=original['sha256'],
                    successor_sha256=reference['sha256']))
        else:
            authenticate_source_reference(reference)
        references[str(path)] = reference
    for path in (template_path, inventory_path, Path(__file__),
            ROOT/'scripts/admit_programme_source_cuts.py', ROOT/'scripts/admit_programme_source_cut_variant.py'):
        reference = _reference(path)
        references[reference['path']] = reference
    preserved = []
    selected = []
    seen = set()
    for cut in inventory['cuts']:
        mask = (native >= cut['start_pts']) & (native < cut['stop_pts'])
        cut_native = native[mask]
        cut_actual = np.unique(actual[mask])
        if (len(cut_native) != cut['native_count'] or len(cut_actual) != cut['canonical_count']
                or not len(cut_native) or cut_native[0] != cut['start_pts']
                or cut_native[-1]+1 != cut['stop_pts'] or seen.intersection(map(int,cut_native))):
            raise ValueError('Predecessor cut clock does not match the exact frozen programme')
        seen.update(map(int, cut_native))
        if cut['shot_id'] not in COMPLETE:
            selected.append(dict(cut))
            continue
        directory, expected = COMPLETE[cut['shot_id']]
        stem = cut['shot_id'].replace(':', '_')
        source = WORK/directory/(stem+'.npz')
        reference = dict(path=str(source.resolve()), sha256=expected)
        authenticate_source_reference(reference)
        with np.load(source, allow_pickle=False) as pack:
            metadata = json.loads(str(pack['metadata_json'].item()))
            if not np.array_equal(metadata['processed_actual_pts'], cut_actual):
                raise ValueError('A retained successful pack omitted a canonical exposure')
            if metadata.get('physically_compacted_approved_pixels_only') is not True:
                raise ValueError('Retained source pixels must pass the strict public source gate')
        aliases = _reference(source.with_name(stem+'_native_alias_review.npz'))
        with np.load(aliases['path'], allow_pickle=False) as pack:
            if not np.array_equal(pack['native_pts'], cut_native) or not np.array_equal(pack['actual_pts'], actual[mask]):
                raise ValueError('A retained successful pack omitted an exact native alias exposure')
            representative = cut_native == actual[mask]
            if not np.array_equal(pack['alias_native_pts'], cut_native[~representative]):
                raise ValueError('A retained successful pack omitted native alias source checks')
        retirement = _reference(source.with_name(stem+'_retired_old_source_indices.npz'))
        for item in (reference, aliases, retirement):
            references[item['path']] = item
        preserved.append(dict(cut, absolute_inputs=reference, native_alias_review=aliases,
            old_retirement_indices=retirement, original_path_reused=True,
            processing_complete_is_not_manual_visual_completion=True))
    if seen != set(map(int, native)) or len(selected) != 5 or len(preserved) != 4:
        raise ValueError('The five pending and four retained cuts must cover the full original source clock')
    result = dict(template)
    result.update(cuts=sorted(selected,key=lambda c:c['start_pts']), references=list(references.values()),
        preserved_completed_source_cuts=preserved,
        predecessor_manifest_references=[_reference(template_path), _reference(inventory_path)],
        implementation_successors=changed_code,
        continuation_policy=dict(complete_independent_source_replay_for_unfinished_cuts=True,
            completed_source_cuts_reprocessed=False, interrupted_source_journals_preserved=True,
            native_alias_source_checks_skipped=False, solver_started=False,
            source_processing_is_not_manual_audit_completion=True))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf8') as saved:
            json.dump(result, saved, ensure_ascii=False, separators=(',',':'))
    return dict(manifest_reference=None if output is None else _reference(output),
        frozen=output is not None, remaining_cuts=[c['shot_id'] for c in selected],
        retained_cuts=[c['shot_id'] for c in preserved],
        native_source_frames_to_replay=sum(c['native_count'] for c in selected),
        canonical_source_frames_to_replay=sum(c['canonical_count'] for c in selected),
        solver_started=False)


def main():
    parser = argparse.ArgumentParser()
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path)
    destination.add_argument('--check-only', action='store_true')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_continuation(args.output,template_path=args.template,
        inventory_path=args.inventory), ensure_ascii=False, separators=(',',':')), flush=True)


if __name__ == '__main__':
    main()
