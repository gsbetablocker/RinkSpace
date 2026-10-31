"""Prepare an exclusion-only source revision in a new directory; never fit."""
from pathlib import Path
import argparse
import copy
import json

import numpy as np

import prepare_native_multicamera as prepared
from apply_native_multicamera_review import revise_observations
from prepare_spatial_refinement import _write_prepared, _same_bytes
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def revise_exclusions(original, numeric_stations, manifest, path_ids, report, review_path):
    """Authenticate explicit row decisions, then reuse the existing revision helper."""
    if not isinstance(report.get('decisions'), list):
        raise ValueError('Review needs an explicit decisions list')
    if report.get('added_observations'):
        raise ValueError('New observations use prepare_spatial_refinement --source-packets')
    normalized = copy.deepcopy(report)
    images = {}
    lookup = {str(value): i for i, value in enumerate(original['observation_ids'])}
    for decision in normalized['decisions']:
        if decision.get('decision') != 'exclude_preserving_source':
            raise ValueError('Only exclude_preserving_source is supported')
        if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
            raise ValueError('Every exclusion needs its source-based reason')
        paths = []
        for name in ('source_image', 'evidence_image'):
            if decision.get(name):
                path = Path(decision[name])
                paths.append(path.resolve() if path.is_absolute() else (Path(review_path).parent/path).resolve())
        if not paths or len(set(paths)) != 1 or not paths[0].is_file():
            raise ValueError('Every exclusion needs one existing source_image/evidence_image')
        image = paths[0]
        if str(image) not in images:
            images[str(image)] = prepared.digest(image)
        digest = images[str(image)]
        if decision.get('source_image_file_sha256', digest) != digest:
            raise ValueError('Source image file digest differs')
        row = lookup[decision['observation_id']]
        if 'source_row' in decision and int(decision['source_row']) != int(original['source_rows'][row]):
            raise ValueError('Reviewed source_row differs')
        if 'row' in decision and decision['row'] != row:
            raise ValueError('Reviewed mother-set row differs')
        decision.update(source_image=str(image), evidence_image=str(image), source_image_file_sha256=digest)
    result = revise_observations(original, numeric_stations, manifest, path_ids, normalized)
    new, stations, shots, touched, records, additions, origins = result
    if additions or np.any(touched) or np.any(origins < 0):
        raise RuntimeError('Exclusion-only preparation changed or added observations')
    for name in prepared.FIELDS:
        _same_bytes(new[name], original[name][origins], 'retained '+name)
    _same_bytes(stations, numeric_stations[origins], 'retained numeric stations')
    return result


def prepare(run_root, state_path, review_path, output):
    root, state_path, review_path = [Path(p).resolve() for p in (run_root, state_path, review_path)]
    output = guard_project_write(output)
    if output.exists():
        raise FileExistsError('Preserve existing prepared output: '+str(output))
    if root == output or root.is_relative_to(output):
        raise ValueError('Output cannot replace or contain the parent input')
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError('Prepared observation outputs belong outside the OneDrive source tree')
    manifest = prepared.read(root/'joint_manifest.json')
    mapping = prepared.read(root/'mapping.json')
    originals = [root/'point_observations.npz', root/'joint_manifest.json', root/'mapping.json', state_path, review_path]
    parent_hashes = {str(path): prepared.digest(path) for path in originals}
    input_hash = parent_hashes[str(root/'point_observations.npz')]
    state_hash, review_hash = parent_hashes[str(state_path)], parent_hashes[str(review_path)]
    if input_hash != manifest['point_input_sha256']:
        raise ValueError('Current parent input digest differs')
    for key, expected in [('scene_path', 'scene_file_sha256'), ('native_library', 'native_library_sha256')]:
        if prepared.digest(Path(manifest[key])) != manifest[expected]:
            raise ValueError('Current model/runtime differs: '+key)
    original, state = prepared.npz(root/'point_observations.npz'), prepared.npz(state_path)
    model = SceneParameterModel(prepared.read(manifest['scene_path']), Path(manifest['scene_path']), optimize_scene=True)
    prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, original)
    report = prepared.read(review_path)
    for name in ('input_sha256', 'point_input_sha256'):
        if name in report and report[name] != input_hash:
            raise ValueError('Review refers to a different parent input')
    if report.get('state_sha256', state_hash) != state_hash:
        raise ValueError('Review refers to a different numeric state')
    new, numeric, shots, _, decisions, _, origins = revise_exclusions(
        original, state['stations'], manifest, mapping['path_ids'], report, review_path)
    protected = {name: state[name].copy() for name in ('rig', 'controls', 'scene_values')}
    excluded = np.flatnonzero(~np.isin(np.arange(len(original['pts'])), origins))
    receipt_path = output/'source_review_receipt.json'
    provenance = dict(schema='rinkspace.source_review_preparation.v1', parent_run_root=str(root),
        parent_input_sha256=input_hash, parent_numeric_state=str(state_path), parent_numeric_state_sha256=state_hash,
        review=str(review_path), review_sha256=review_hash, receipt=str(receipt_path),
        excluded_rows=len(excluded), retained_rows=len(new['pts']), allowed_action='exclude_preserving_source',
        sampling_executed=False, source_reread_executed=False, fit_executed=False,
        source_UV_changed=False, model_and_motion_chart_changed=False)
    updated, revised_mapping = _write_prepared(output, new, numeric, shots, state, manifest,
                                               mapping, model, provenance)
    # The shared writer's legacy provenance key describes sampling. Retain the
    # parent's value and register this operation under its own preparation key.
    if 'spatial_refinement_preparation' in manifest:
        updated['spatial_refinement_preparation'] = manifest['spatial_refinement_preparation']
    else:
        updated.pop('spatial_refinement_preparation', None)
    updated['source_review_preparation'] = provenance
    prepared.write(output/'joint_manifest.json', updated)
    saved = prepared.npz(output/'initial_pixels.npz')
    prepared.validate_snapshot(saved, model.initial_state.layout, revised_mapping, updated, new)
    for name, value in protected.items():
        _same_bytes(saved[name], value, name)
    archive = output/'excluded_source_observations.npz'
    np.savez(archive, **{name: value[excluded] for name, value in original.items()},
             mother_row_indices=excluded, numeric_stations=state['stations'][excluded])
    for path, digest in parent_hashes.items():
        if prepared.digest(Path(path)) != digest:
            raise RuntimeError('Parent input/state/review changed: '+path)
    receipt = dict(**provenance, new_input_sha256=updated['point_input_sha256'],
        prepared_state=str(output/'initial_pixels.npz'), prepared_state_sha256=prepared.digest(output/'initial_pixels.npz'),
        parent_files_sha256=parent_hashes, parent_files_unchanged=True, seven_unweighted_fields=True,
        retained_source_fields_byte_preserved=True, retained_numeric_stations_byte_preserved=True,
        rig_controls_scene_values_byte_preserved=True, excluded_source_archive=str(archive),
        excluded_source_archive_sha256=prepared.digest(archive), decisions=decisions)
    prepared.write(receipt_path, receipt)
    print(json.dumps(dict(output=str(output), excluded_rows=len(excluded), retained_rows=len(new['pts']),
                          initial_state=str(output/'initial_pixels.npz'), fit_executed=False)), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'state', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run_root, args.state, args.review, args.output)
