"""Pack authenticated current board readouts without another video replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack
from .strict_model_scope import authenticate_source_reference, file_sha


def write_exact_board_source_delta(delta_reference, *, state_identity_path,
        legacy_inventory_paths, shot_plan_path, output_path):
    """Consume source-approved rows, preserving both calibration and ML roles.

    The authority's previous images register continuity evidence only; they
    do not become processed exposures or new calibration observations.
    """
    from rinkspace.vision_board_source_delta import normalize_exact_board_source_delta
    from scripts.admit_programme_source_inputs import LegacyHeldSourceRoles, whole_shot_role

    path, _ = authenticate_source_reference(delta_reference)
    identity_path = Path(state_identity_path).resolve()
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    scene_path = identity_path.with_name('scene_UNACCEPTED.json')
    scene_ref = dict(path=str(scene_path), sha256=identity['scene_sha256'])
    camera_ref = dict(path=identity['camera_path'], sha256=identity['camera_sha256'])
    for reference in (scene_ref, camera_ref):
        authenticate_source_reference(reference)
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    with np.load(camera_ref['path'], allow_pickle=False) as camera:
        native = camera['pts'].copy()
        actual = camera['exposure_representative_pts'].copy()
    source_clock = {int(n): int(a) for n, a in zip(native, actual)}
    content = json.loads(path.read_text(encoding='utf8'))
    authority_ref = content['source_identity_authority']
    authority_path, _ = authenticate_source_reference(authority_ref)
    authority = json.loads(authority_path.read_text(encoding='utf8'))
    items = normalize_exact_board_source_delta(path,
        expected_sha256=delta_reference['sha256'], scene=scene)
    if not items or any(item['rejected_rows'] for item in items):
        raise ValueError('Actual board delta contains rejected or missing source rows')
    held = LegacyHeldSourceRoles(legacy_inventory_paths)
    plan = json.loads(Path(shot_plan_path).read_text(encoding='utf8'))['shot_plan']
    references = [delta_reference, authority_ref, scene_ref, camera_ref,
        dict(path=str(identity_path), sha256=file_sha(identity_path)),
        dict(path=str(Path(shot_plan_path).resolve()), sha256=file_sha(shot_plan_path)),
        authority['model_navigation_reference'], *held.references]
    for record in authority['physical_instance_records']:
        references.extend(record['source_image_references'])
    module = Path(__file__).resolve()
    from rinkspace import vision_board_source_delta
    references.extend(dict(path=str(p), sha256=file_sha(p)) for p in
        (module, Path(vision_board_source_delta.__file__).resolve()))
    writer = StrictAbsolutePackWriter(output_path, source=items[0]['review'].source,
        expected_actual_pts=np.unique(actual), original_references=references)
    source = writer.source
    for record in authority['physical_instance_records']:
        for binding in record['source_bindings']:
            if (source_clock.get(int(binding['native_pts'])) != int(binding['actual_pts'])
                    or binding['source_sha256'] != source['video_sha256']
                    or binding['time_base'] != source['time_base']):
                raise ValueError('Board continuity evidence has a differing exact source clock')
            writer._register_frame(binding['actual_pts'], binding['native_pts'],
                binding['decoded_native_BGR_sha256'], binding['shot_id'])
    count = 0
    for item in items:
        for row in item['rows']:
            row['split'] = held.split(row)
            row['vision_training_role'] = whole_shot_role(plan, item['source'])
            count += writer.add_row(row, item['review'])
        writer.mark_exposure_processed(item['source']['actual_pts'],
            source_binding=item['source'])
    receipt = writer.finish()
    pack = StrictAbsoluteSourcePack(receipt['path'], expected_sha256=receipt['sha256'])
    return dict(receipt, actual_source_admitted_pixels=count, row_count=len(pack.rows),
        processed_actual_pts=pack.metadata['processed_actual_pts'],
        explicit_held_rows=sum(row['split'] == 'held' for row in pack.rows),
        whole_programme_source_review_complete=False, solver_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', nargs=2, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--legacy-inventory', action='append', type=Path, required=True)
    parser.add_argument('--shot-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_exact_board_source_delta(
        dict(path=args.delta[0], sha256=args.delta[1]), state_identity_path=args.identity,
        legacy_inventory_paths=args.legacy_inventory, shot_plan_path=args.shot_plan,
        output_path=args.output), ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
