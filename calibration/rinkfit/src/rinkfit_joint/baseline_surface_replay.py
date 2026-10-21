"""Small, immutable replay of historical 3D surfaces on the native Demo clock.

This is a numerical baseline, not a source observation or the current physical
model.  The old camera/scene stays in this adapter.  It must not initialize a
new model by assuming that equal station, angle, or advertisement number means
the same currently verified material point.  No screen cache or solver runs.

``BaselineSurfaceReplay(initial_state, use_patch=False).project(pts, xyz,
scene_sha256=...)`` uses the unpatched bridge and original Brown by default.
``optics='demo_pinhole'`` reproduces the ice renderer's no-Brown convention for
arbitrary historical 3D points; its board/ad result is an explicit extension of
that convention, since the actual Demo texture contains only the ice plane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _bound(ref):
    path = Path(ref.get('file', ref.get('path')))
    if _sha(path) != ref['sha256']:
        raise ValueError('Changed historical dependency: ' + str(path))
    return path


def _npz(path, keys=None):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in (data.files if keys is None else keys)}


def _reference(path):
    return dict(path=str(path), sha256=_sha(path))


class BaselineSurfaceReplay:
    """PTS-exact, SHA-bound historical baseline; no nearest-frame fallback."""

    def __init__(self, initial_state, *, use_patch=False):
        self.use_patch = bool(use_patch)
        self.path = Path(initial_state)
        self.manifest = _read(self.path)
        self.source_path = _bound(self.manifest['source_packet'])
        self.receipt_path = _bound(self.manifest['source_packet_receipt'])
        self.patch_path = _bound(self.manifest['V3_camera_patch'])
        self.scene_path = _bound(self.manifest['old_scene_reference'])
        receipt = _read(self.receipt_path)
        self.scene_sha256 = _sha(self.scene_path)
        self.scene = _read(self.scene_path)
        inputs = receipt['inputs']
        self.camera_path = _bound(inputs['camera'])
        self.ledger_path = _bound(inputs['historical_alignment'])
        if _bound(inputs['scene']) != self.scene_path:
            raise ValueError('Demo and historical scene identity differ')
        identity_path = self.scene_path.with_name('state_identity.json')
        self.identity = _read(identity_path)
        complete_path = self.scene_path.with_name('complete_state_UNACCEPTED.npz')
        if (self.identity['camera_sha256'] != _sha(self.camera_path)
                or self.identity['scene_sha256'] != self.scene_sha256
                or self.identity['state_npz_sha256'] != _sha(complete_path)):
            raise ValueError('The old complete state/camera/scene binding differs')
        q = _npz(complete_path, ('q',))['q']
        if hashlib.sha256(q.tobytes()).hexdigest() != self.identity['q_sha256']:
            raise ValueError('Historical complete q differs')
        self.packet = _npz(self.source_path)
        self.legacy = _npz(self.camera_path, ('pts', 'camera25', 'exposure_representative_pts'))
        self.patch = _npz(self.patch_path)
        self.lookup = {int(p): i for i, p in enumerate(self.packet['pts'])}
        self.legacy_lookup = {int(p): i for i, p in enumerate(self.legacy['pts'])}
        self.patch_lookup = {int(p): i for i, p in enumerate(self.patch['pts'])}
        if (str(self.packet['time_base']) != '1/12800'
                or str(self.packet['native_fps']) != '50/1'
                or not np.all(np.diff(self.packet['pts']) == 256)
                or str(self.patch['source_packet_sha256']) != _sha(self.source_path)):
            raise ValueError('Native clock or patch binding differs')
        indices = self.patch['source_packet_index']
        if (not np.array_equal(self.packet['pts'][indices], self.patch['pts'])
                or not np.array_equal(self.packet['camera25'][indices], self.patch['original_camera25'])):
            raise ValueError('Patch does not apply to the exact saved source packet')
        self.ledger = {}
        for line in self.ledger_path.read_text(encoding='utf8').splitlines():
            entry = json.loads(line)
            self.ledger[int(entry['full_pts'])] = entry
        self.references = {name: _reference(path) for name, path in (
            ('initial_state', self.path), ('source_packet', self.source_path),
            ('source_packet_receipt', self.receipt_path), ('patch', self.patch_path),
            ('old_scene', self.scene_path), ('old_camera', self.camera_path),
            ('old_identity', identity_path), ('old_complete_state', complete_path),
            ('historical_alignment', self.ledger_path))}

    def frame_index(self, native_pts, shot_id=None):
        if isinstance(native_pts, bool) or not isinstance(native_pts, (int, np.integer)):
            raise ValueError('Integer official native PTS required')
        if int(native_pts) not in self.lookup:
            raise ValueError('Exact native PTS missing; no time-offset or nearest-frame fallback')
        index = self.lookup[int(native_pts)]
        if shot_id is not None and shot_id != f's{int(self.packet["shot_index"][index]):02d}':
            raise ValueError('Shot/PTS mismatch')
        return index

    def camera(self, native_pts, *, stage=None, shot_id=None):
        stage = stage or ('demo_patch' if self.use_patch else 'demo_source')
        index = self.frame_index(native_pts, shot_id)
        if stage == 'legacy_ledger':
            left = int(self.packet['anchor_left_legacy_pts'][index])
            right = int(self.packet['anchor_right_legacy_pts'][index])
            lo = int(self.packet['anchor_left_official_pts'][index])
            hi = int(self.packet['anchor_right_official_pts'][index])
            alpha = 0. if lo == hi else (int(native_pts)-lo)/(hi-lo)
            return (self.legacy['camera25'][self.legacy_lookup[left]]*(1-alpha)
                    + self.legacy['camera25'][self.legacy_lookup[right]]*alpha)
        if stage not in ('demo_source', 'demo_patch'):
            raise ValueError('Unknown camera replay stage')
        if stage == 'demo_patch' and int(native_pts) in self.patch_lookup:
            return self.patch['camera25'][self.patch_lookup[int(native_pts)]].copy()
        return self.packet['camera25'][index].copy()

    def provenance(self, native_pts):
        index = self.frame_index(native_pts)
        fields = ('bridge_mode', 'direct_mapping', 'inferred', 'valid',
                  'anchor_left_official_pts', 'anchor_right_official_pts',
                  'anchor_left_legacy_pts', 'anchor_right_legacy_pts')
        result = dict(native_pts=int(native_pts), source_seconds=int(native_pts)/12800,
                      shot_id=f's{int(self.packet["shot_index"][index]):02d}',
                      **{key: self.packet[key][index].item() for key in fields})
        result['bridge_definition'] = _read(self.receipt_path)['bridge_mode'][str(result['bridge_mode'])]
        result['historical_ledger_rows'] = [self.ledger[p] for p in sorted({
            result['anchor_left_legacy_pts'], result['anchor_right_legacy_pts']})]
        result['patch'] = None
        if int(native_pts) in self.patch_lookup:
            pi = self.patch_lookup[int(native_pts)]
            legacy = int(self.patch['legacy_candidate_pts'][pi])
            result['patch'] = dict(changed=bool(self.patch['changed'][pi]),
                legacy_candidate_pts=legacy, candidate_accepted=bool(self.patch['candidate_accepted'][pi]),
                candidate_roi_gray_mae=float(self.patch['candidate_roi_gray_mae'][pi]),
                historical_ledger_row=self.ledger[legacy])
        result.update(numerical_baseline_only=True, source_observation=False,
                      physical_accuracy_accepted=False, current_world_correspondence_certified=False)
        result['default_camera_stage'] = 'demo_patch' if self.use_patch else 'demo_source'
        result['V3_patch_is_default'] = self.use_patch
        result['valid_flag_semantics'] = 'Source packet body flag; not surface visibility or camera accuracy'
        return result

    def project(self, native_pts, world_xyz, *, scene_sha256, stage=None,
                optics='original_brown', shot_id=None):
        if scene_sha256 != self.scene_sha256:
            raise ValueError('Different world chart; explicit material crosswalk required')
        xyz = np.asarray(world_xyz, np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Finite historical scene XYZ required; unknown height is not zero')
        camera = self.camera(native_pts, stage=stage, shot_id=shot_id)
        if optics == 'demo_pinhole':
            camera[11:] = 0.
        elif optics != 'original_brown':
            raise ValueError('Unknown historical optics contract')
        from calibration.program_tracking.projection import safe_project
        return safe_project(camera, xyz)

    def project_many(self, native_pts, world_xyz, *, scene_sha256,
                     stage=None, optics='original_brown'):
        """Small batched replay, with the same old XYZ at each explicit PTS.

        The caller chooses its batch/chunk size. No persistent screen cache is
        created, and frame provenance remains available through ``provenance``.
        """
        projected = [self.project(p, world_xyz, scene_sha256=scene_sha256,
            stage=stage, optics=optics) for p in native_pts]
        if not projected:
            n=len(world_xyz)
            return np.empty((0,n,2)),np.empty((0,n)),np.empty((0,n),bool)
        return tuple(np.stack([row[i] for row in projected]) for i in range(3))

    def inventory(self, *, include_ice_grid=True):
        """Reuse the original finite 3D path producer, including Ad1--26 prints."""
        # These retained renderer dependencies also support standalone scripts
        # and resolve ``runtime`` relative to program_tracking.
        programme_code = Path(__file__).resolve().parents[3]/'program_tracking'
        if str(programme_code) not in sys.path:
            sys.path.insert(0, str(programme_code))
        from calibration.program_tracking.render_result import fixed_scene_paths, advertisement_review_paths
        feature_paths = {}
        for feature in self.scene['ice_features']:
            feature_scene = dict(self.scene, ice_features=[feature])
            for j, (_, points) in enumerate(fixed_scene_paths(feature_scene, draw_grid=False)[2:]):
                digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
                feature_paths.setdefault(digest, []).append(dict(
                    historical_feature_id=feature['id'], local_feature_path_index=j,
                    historical_feature_kind=feature['kind']))
        paths = []
        for i, (kind, xyz) in enumerate(fixed_scene_paths(self.scene, draw_grid=True)):
            if kind == 'grid' and not include_ice_grid:
                continue
            paths.append(dict(id=f'scene_path:{i}', kind=kind, xyz=xyz,
                original_renderer_path_index=i,
                historical_feature_paths=feature_paths.get(
                    hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest(), [])))
        for i, item in enumerate(advertisement_review_paths(self.scene)):
            paths.append(dict(item, id=f'Ad{item["number"]}:{item["role"]}:{i}', kind='advertisement',
                              original_advertisement_path_index=i))
        return paths

    def entity_paths(self, entity_id):
        """Select historical named material paths; never map a current station."""
        paths = self.inventory()
        if entity_id in ('board.contact', 'board.top'):
            kind = 'contact' if entity_id == 'board.contact' else 'board_top'
            selected = [p for p in paths if p['kind'] == kind]
        elif entity_id.startswith('ice:'):
            selected = [p for p in paths if any(f['historical_feature_id'] == entity_id[4:]
                for f in p.get('historical_feature_paths', []))]
        elif entity_id.startswith('print:Ad') and entity_id[8:].isdigit():
            selected = [p for p in paths if p.get('number') == int(entity_id[8:])]
        else:
            raise ValueError('Unknown historical entity notation')
        if not selected:
            raise ValueError('Historical entity is absent; no geometric substitution')
        return selected

    def replay_inventory(self, native_pts, *, stage=None, optics='original_brown'):
        paths = self.inventory()
        offsets = np.r_[0, np.cumsum([len(p['xyz']) for p in paths])]
        xyz = np.concatenate([p['xyz'] for p in paths])
        uv, depth, valid = self.project(native_pts, xyz, scene_sha256=self.scene_sha256,
                                        stage=stage, optics=optics)
        return dict(xyz=xyz, uv=uv, depth=depth, valid=valid, offsets=offsets,
                    paths=[{key: value for key, value in item.items() if key != 'xyz'} for item in paths],
                    provenance=self.provenance(native_pts),
                    stage=stage or ('demo_patch' if self.use_patch else 'demo_source'), optics=optics)


def _stats(first, second, valid):
    error = np.linalg.norm(first[valid]-second[valid], axis=1)
    result = dict(samples=len(error), maximum_px=float(error.max()) if len(error) else None,
                rms_px=float(np.sqrt(np.mean(error*error))) if len(error) else None,
                median_px=float(np.median(error)) if len(error) else None)
    onscreen = valid & np.all((first >= [0,0]) & (first <= [1919,1079]), axis=1)
    error = np.linalg.norm(first[onscreen]-second[onscreen], axis=1)
    result['reference_onscreen'] = dict(samples=len(error), maximum_px=float(error.max()) if len(error) else None,
        rms_px=float(np.sqrt(np.mean(error*error))) if len(error) else None,
        median_px=float(np.median(error)) if len(error) else None)
    return result


def audit(initial_state, compact_seed, output, pts):
    """Two-frame numerical replay, never a full programme cache or video."""
    from .compact_screen_seed import ScreenSeed
    replay = BaselineSurfaceReplay(initial_state)
    seed = ScreenSeed(compact_seed)
    if seed.meta['source_video_sha256'] != replay.manifest['source_identity']['sha256']:
        raise ValueError('Compact and historical bridge belong to different sources')
    output = Path(output)
    if not output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Diagnostic replay arrays must remain in the Data root')
    if output.exists():
        raise FileExistsError(output)
    paths = replay.inventory()
    output.mkdir(parents=True)
    offsets = np.r_[0, np.cumsum([len(p['xyz']) for p in paths])]
    xyz = np.concatenate([p['xyz'] for p in paths])
    path_kinds = np.concatenate([np.full(len(p['xyz']), p['kind']) for p in paths])
    ice = np.isin(path_kinds, ('marking', 'grid'))
    surface = ~ice
    current_links = output.parent/'advertisement_source_crosswalk.json'
    crosswalk = dict(schema='rinkspace.historical_surface_primitive_crosswalk.v1',
        old_scene=replay.references['old_scene'], definitions_are_historical_model_state=True,
        original_advertisement_ids=sorted({p['number'] for p in paths if p['kind']=='advertisement'}),
        same_station_or_angle_current_correspondence=False,
        current_source_crosswalk=_reference(current_links) if current_links.exists() else None,
        current_source_crosswalk_scope='Only independently identified links in that file; equal Ad number does not add a correspondence',
        all_legacy_surfaces_are_source_observations=False,
        compact_scope='ice grid and modeled markings, z=0 only; no board/ad height or projection equivalence',
        advertisements_are_unwrapped_continuous_board_surfaces=True,
        paths=[dict({k:v for k,v in p.items() if k not in ('xyz','inward')},
                    vertex_begin=int(a), vertex_end=int(b),
                    world_vertex_sha256=hashlib.sha256(np.ascontiguousarray(p['xyz']).tobytes()).hexdigest(),
                    z_range=[float(np.min(p['xyz'][:,2])),float(np.max(p['xyz'][:,2]))])
               for p,a,b in zip(paths, offsets[:-1], offsets[1:])])
    report = dict(schema='rinkspace.small_native_historical_surface_replay.v1',
        references=replay.references, frames=[], source_observations_created=False,
        solver_run=False, physical_parameters_imported=False, no_accuracy_acceptance=True,
        validity_means='finite forward ray on original optical branch; no source material ownership or occlusion certification',
        no_brown_board_ad_projection_is_explicit_extension_of_demo_camera=True)
    for native_pts in pts:
        provenance = replay.provenance(native_pts)
        projections = {}
        for stage in ('legacy_ledger','demo_source','demo_patch'):
            for optics in ('original_brown','demo_pinhole'):
                projections[stage+'__'+optics] = replay.project(native_pts, xyz,
                    scene_sha256=replay.scene_sha256, stage=stage, optics=optics)
        original = projections['legacy_ledger__original_brown']
        source = projections['demo_source__original_brown']
        patched = projections['demo_patch__original_brown']
        pinhole = projections['demo_patch__demo_pinhole']
        groups = dict(ice=ice, board_and_ads=surface,
            board_contact=path_kinds=='contact', board_top=path_kinds=='board_top',
            advertisements=path_kinds=='advertisement')
        delta = {}
        for name, select in groups.items():
            delta[name] = dict(
                legacy_ledger_to_source_packet=_stats(original[0],source[0],select&original[2]&source[2]),
                source_packet_to_V3_patch=_stats(source[0],patched[0],select&source[2]&patched[2]),
                patched_original_Brown_to_Demo_pinhole=_stats(patched[0],pinhole[0],select&patched[2]&pinhole[2]))
        origin = np.asarray(seed.meta['world_chart']['normalization']['origin'])
        length = seed.meta['world_chart']['normalization']['L']
        compact_uv, compact_valid = seed.project_reference_xy(native_pts, provenance['shot_id'],
            (xyz[ice,:2]-origin[:2])/length, world_chart_sha256=seed.world_chart_sha256)
        delta['compact_ice_vs_patched_Demo_pinhole'] = _stats(compact_uv,pinhole[0][ice],compact_valid&pinhole[2][ice])
        target = output/f'pts{native_pts}.npz'
        arrays = dict(world_xyz=xyz, path_offsets=offsets, compact_ice_uv=compact_uv, compact_ice_valid=compact_valid)
        for name, (uv, depth, valid) in projections.items():
            arrays.update({name+'__uv':uv, name+'__depth':depth, name+'__valid':valid})
        np.savez_compressed(target, **arrays)
        report['frames'].append(dict(provenance=provenance, numerical_deltas_px=delta,
            arrays=_reference(target), board_ad_compact_delta=None,
            board_ad_compact_delta_reason='No compact board/ad state exists; ice H is not applied to nonzero Z'))
    for name, value in (('crosswalk.json',crosswalk),('receipt.json',report)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-state', required=True)
    parser.add_argument('--compact-seed', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pts', type=int, nargs='+', default=[2405120,1464832])
    args = parser.parse_args()
    report = audit(args.initial_state,args.compact_seed,args.output,args.pts)
    print(json.dumps(dict(output=args.output,frames=[dict(pts=f['provenance']['native_pts'],
        deltas=f['numerical_deltas_px']) for f in report['frames']]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
