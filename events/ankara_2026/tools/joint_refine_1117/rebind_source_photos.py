"""Bind archived Nayeon horizontal photo registrations to a fixed scene."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from shared import (
    DATA, LIBRARY, OUT, NativeCore, NativeSceneGPU, ParameterState,
    SceneParameterModel, arrays, numeric_sha, original_native, read,
    registration, safe_path, save, sha, write,
)


REVIEW = DATA / 'events/ankara_2026/programmes/nayeon_ko_sp/fit_1116/review_all_1116'
PHOTO = REVIEW / 'whole_quality_1117/reviewed_observations/photos.npz'
CURRENT = OUT / 'nayeon_material_polish'
OUTPUT = OUT / 'repair/prepared'
EVIDENCE = (
    ('named_first', 'print_trajectory_1117/first_read/{shot}_reads.json', 109),
    ('named_focus', 'programme_retake_1117/print_focus/{shot}_reads.json', 109),
    ('named_repeat', 'programme_retake_1117/print_v2/{shot}_reads.json', 109),
    ('colour', 'print_trajectory_1117/color_reads/{shot}.json', 105),
    ('crest', 'print_trajectory_1117/crest_outline/{shot}.json', 105),
    ('group_statistic', 'programme_retake_1117/printed/{shot}_photo.json', None),
)
IDENTITY = ('shot', 'pts', 'region', 'world', 'uv')
WORLD_TOLERANCE = 1.e-13


def key(shot, pts, region, world, uv):
    return str(shot), int(pts), str(region), tuple(world), tuple(uv)


def frame_regions(frame, kind):
    if kind == 'group_statistic':
        for ad, record in frame.items():
            yield f'Ad{ad}.letter_group', record, record['source_sha256']
        return
    for ad, regions in frame['ads'].items():
        if kind.startswith('named_'):
            regions = regions['words']
        for label, record in regions.items():
            if kind == 'crest':
                label = 'verified_upper_pigment'
            yield f'{ad}.{label}', record, frame['original_bgr_sha256']


def match_photo_rows(photo, source_sha):
    wanted = {key(*row): i for i, row in enumerate(zip(
        *(photo[field] for field in IDENTITY), strict=True,
    ))}
    if len(wanted) != len(photo['pts']):
        raise ValueError('Duplicate archived photo row')
    matches = {}
    files = {}
    for shot in np.unique(photo['shot']):
        with np.load(PHOTO.with_name(f'{shot}.npz'), allow_pickle=False) as data:
            if data['source_sha256'].item() != source_sha:
                raise ValueError(f'Source video mismatch for {shot}')
            hashes = dict(zip(data['source_frame_pts'], data['source_frame_BGR_sha256'], strict=True))
        for kind, pattern, count in EVIDENCE:
            path = REVIEW / pattern.format(shot=shot)
            if not path.is_file():
                continue
            n = 0
            for pts_text, frame in read(path).items():
                if not pts_text.isdecimal():
                    continue
                pts = int(pts_text)
                for region, evidence, frame_sha in frame_regions(frame, kind):
                    row = wanted.get(key(shot, pts, region, evidence['world'], evidence['uv']))
                    if row is None:
                        continue
                    if row in matches or hashes[pts] != frame_sha:
                        raise ValueError(f'Ambiguous evidence or source-frame mismatch for row {row}')
                    matches[row] = kind, count, evidence.get('paths'), frame_sha
                    n += 1
            if n:
                files[f'{shot}/{kind}'] = {'path': str(path), 'sha256': sha(path), 'rows': n}
    if len(wanted) != len(matches):
        raise ValueError(f'{len(wanted) - len(matches)} photo rows lack exact source evidence')
    return matches, files


def points_at_station(native, path, station):
    spec = native.paths[path]
    if spec['kind'] != 8 or spec['layer'] != 'advertisement':
        raise ValueError(f'Expected native print material curve for {path}')
    course, height = np.asarray(spec['vertices']).T
    knots = spec['knots']
    return np.asarray(native.core.board_points(
        native.scene['board'], np.interp(station, knots, course), np.interp(station, knots, height),
    ))


def archived_samples(native, ad, count):
    samples = defaultdict(list)
    for path, spec in native.paths.items():
        if spec.get('instance_id') != f'print_instance/{ad}':
            continue
        # Reproduce the kind-8 grid used by the archived named and colour readers.
        n = max(count, min(257, 2 * len(spec['vertices'])))
        station = np.linspace(*(spec.get('support') or (0., 1.)), n)
        for xyz, t in zip(points_at_station(native, path, station), station, strict=True):
            samples[tuple(xyz)].append((path, t))
    return samples


def identify_material(photo, matches, old_native):
    n = len(photo['pts'])
    width = max(map(len, old_native.paths))
    paths = np.full((n, 2), '', dtype=f'U{width}')
    stations = np.full((n, 2), np.nan)
    grids = {}
    for i, (region, world) in enumerate(zip(photo['region'], photo['world'], strict=True)):
        _, count, allowed_paths, _ = matches[i]
        if count is None:
            continue
        ad = region.split('.', 1)[0]
        if (ad, count) not in grids:
            grids[ad, count] = archived_samples(old_native, ad, count)
        candidates = grids[ad, count].get(tuple(world), ())
        candidates = [(p, t) for p, t in candidates if allowed_paths is None or p in allowed_paths]
        if not 1 <= len(candidates) <= 2:
            raise ValueError(f'Expected one or two exact material bindings for photo row {i}')
        for j, (path, station) in enumerate(sorted(candidates)):
            paths[i, j] = path
            stations[i, j] = station
    return paths, stations


def rebind_world(native, paths, stations):
    """Evaluate stored (path, station) candidates on a selected fixed scene."""
    world = np.full((*paths.shape, 3), np.nan)
    for path in np.unique(paths[paths != '']):
        mask = paths == path
        world[mask] = points_at_station(native, path, stations[mask])
    double = paths[:, 1] != ''
    if not np.allclose(world[double, 0], world[double, 1], atol=WORLD_TOLERANCE, rtol=0):
        raise ValueError('Two archived bindings no longer give the same point on the selected scene')
    return world[:, 0]


def prepare(scene_path, state_path, output):
    photo = arrays(PHOTO)
    binding = registration('nayeon')
    source_sha = binding['source']['sha256']
    old_geometry_sha = binding['geometry_binding']['value_sha256']
    if photo['source_sha256'].item() != source_sha or photo['geometry_sha256'].item() != old_geometry_sha:
        raise ValueError('Photo archive disagrees with its original programme registration')
    matches, evidence_files = match_photo_rows(photo, source_sha)
    paths, stations = identify_material(photo, matches, original_native('nayeon'))
    values = np.ascontiguousarray(arrays(state_path)['scene_values'])
    model = SceneParameterModel(read(scene_path), scene_path, optimize_scene=True)
    state = ParameterState(model.initial_state.layout, values)
    native = NativeSceneGPU(scene_path, NativeCore(LIBRARY), parameter_state=state)
    world = rebind_world(native, paths, stations)
    used = paths[:, 0] != ''
    count = np.count_nonzero(paths != '', axis=1)
    original = {field: photo[field] for field in ('shot', 'pts', 'region', 'uv', 'weight')}
    packet = output / 'nayeon_photo_bindings.npz'
    identity = dict(source_photo_sha256=sha(PHOTO), source_sha256=source_sha,
                    original_geometry_sha256=old_geometry_sha, scene_sha256=sha(scene_path),
                    state_sha256=sha(state_path), geometry_sha256=numeric_sha(values),
                    native_library_sha256=sha(LIBRARY))
    save(packet, **original, **identity, source_row=np.arange(len(used), dtype=np.int64),
         original_world=photo['world'], world=world, path=paths[:, 0], station=stations[:, 0],
         alternate_path=paths[:, 1], alternate_station=stations[:, 1], usable=used, candidate_count=count,
         binding_status=np.where(used, 'fixed_material_point', 'group_statistic_unbound'),
         evidence_kind=np.asarray([matches[i][0] for i in range(len(used))]),
         source_frame_BGR_sha256=np.asarray([matches[i][3] for i in range(len(used))]))
    receipt = dict(
        schema='rinkspace.nayeon_photo_bindings.v1', photo_archive=str(PHOTO), packet=str(packet),
        packet_sha256=sha(packet), scene=str(scene_path), state=str(state_path), native_library=str(LIBRARY),
        **identity, evidence_files=evidence_files,
        counts=dict(rows=len(used), usable=int(used.sum()), double_candidates=int((count == 2).sum()),
                    group_statistic_unbound=int((~used).sum()), usable_by_shot=dict(Counter(original['shot'][used]))),
        candidate_equivalence_atol_rink_units=WORLD_TOLERANCE,
        observation=dict(rank=1, axis='horizontal', coordinate='native distorted source pixel x',
                         meaning='photometric print registration', xyz_role='fixed scene prediction',
                         xyz_is_source_measurement=False),
        world_policy='Evaluate the stored path/station pairs on any replacement scene before fitting. '
                     'Keep both candidates and verify that they still coincide.',
    )
    write(output / 'nayeon_photo_bindings.json', receipt)
    print(receipt['counts'], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--state', type=Path, default=CURRENT / 'state.npz')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    scene = args.scene or Path(read(CURRENT / 'input_binding.json')['scene'])
    prepare(scene.resolve(), args.state.resolve(), safe_path(args.output))


if __name__ == '__main__':
    main()
