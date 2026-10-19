"""Run one frozen source cut with an explicit graph implementation successor.

The shared original-video driver, normalizers, scope gate, clocks and writer
stay in use. A successor is installed before constructing temporal state.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def run_source_cut_variant(manifest_path, *, shot_id, output_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'rinkspace.exact-source-cut-batch.v1':
        raise ValueError('An immutable source-cut variant manifest is required')
    references = manifest['references']
    for reference in references:
        authenticate_source_reference(reference)
    own_path = str(Path(__file__).resolve())
    reference_paths = {str(Path(r['path']).resolve()) for r in references}
    if own_path not in reference_paths:
        raise ValueError('The variant entrypoint must be frozen with its source inputs')
    cuts = [cut for cut in manifest['cuts'] if cut['shot_id'] == shot_id]
    if len(cuts) != 1:
        raise ValueError('One independent original source cut is required')
    common = manifest['common_arguments']
    if any(option in common for option in ('--output', '--start-pts', '--stop-pts')):
        raise ValueError('Only the cut runner owns output and source ranges')
    module_name, class_name = manifest['ice_graph_class'].rsplit(':', 1)
    module = importlib.import_module(module_name)
    if str(Path(module.__file__).resolve()) not in reference_paths:
        raise ValueError('The graph implementation successor must be hash-bound')
    from scripts import admit_programme_source_inputs as driver
    successor = getattr(module, class_name)
    if not issubclass(successor, driver.IceSourceMaterialIdentityGraph):
        raise ValueError('Reuse the public material identity graph interface')
    driver.IceSourceMaterialIdentityGraph = successor
    base_writer = driver.StrictAbsolutePackWriter
    manifest_ref = dict(path=str(manifest_path), sha256=file_sha(manifest_path))

    class VariantReferencedSourceWriter(base_writer):
        def __init__(self, *args, **kwargs):
            kwargs['original_references'] = [*kwargs.get('original_references', ()), manifest_ref]
            super().__init__(*args, **kwargs)

    driver.StrictAbsolutePackWriter = VariantReferencedSourceWriter
    cut = cuts[0]
    old_arguments = sys.argv
    sys.argv = [str(ROOT/'scripts/admit_programme_source_inputs.py'), *common,
        '--start-pts', str(cut['start_pts']), '--stop-pts', str(cut['stop_pts']),
        '--output', str(Path(output_path).resolve())]
    try:
        driver.main()
    finally:
        sys.argv = old_arguments
    for reference in references:
        if file_sha(reference['path']) != reference['sha256']:
            raise ValueError('Frozen source inputs or variant implementation changed')
    print(json.dumps(dict(stage='source_cut_variant_terminal', shot_id=shot_id,
        exit_code=0, output_path=str(Path(output_path).resolve()),
        sha256=file_sha(output_path), manifest_reference=manifest_ref,
        solver_started=False), separators=(',', ':')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--shot-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_source_cut_variant(args.manifest, shot_id=args.shot_id, output_path=args.output)


if __name__ == '__main__':
    main()
