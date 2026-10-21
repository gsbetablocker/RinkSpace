"""Native-clock numerical 3D navigation from the exact 8a28 forward.

This is a transfer reference, never an observation generation.  Historical
camera values remain inside BaselineSurfaceReplay; none are copied to a model.
Every ice/board point retains its height and the original bridge role.  Saved
world coordinates are deduplicated; no frame/mesh cache is constructed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .baseline_surface_replay import BaselineSurfaceReplay
from .io_utils import read_json, write_json, file_sha256
from .screen_recovery_state import ScreenRecoveryState


def build_reference(screen_state_file, baseline_state_file, output_dir, *,
                    template_model_file=None):
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable surface navigation generation')
    screen = ScreenRecoveryState(screen_state_file)
    baseline = BaselineSurfaceReplay(baseline_state_file)
    seed = screen.seed
    chart = seed.meta['world_chart']
    # The historical mesh normalization and the compact chart are a declared
    # change of units, not a fitted similarity or a current material crosswalk.
    L = float(chart['normalization']['L'])
    if not L:
        raise ValueError('The audited historical XYZ normalization is missing')
    if baseline.scene_sha256 != seed.meta['world_chart']['template_sha256']:
        raise ValueError('Historical scene and screen chart differ')
    template = None
    if template_model_file:
        model = read_json(template_model_file)
        template = dict(path=str(Path(template_model_file).resolve()), sha256=file_sha256(template_model_file))
        if set(model['rigs']) != {'main', 'secondary', 'end'} or len(model['shots']) != 9:
            raise ValueError('One scene, three rigs, nine shots required')
    pts = np.asarray(baseline.packet['pts'], np.int64)
    shots = np.asarray(baseline.packet['shot_index'], np.int16)
    if len(pts) != 8434 or pts[0] != 256256 or pts[-1] != 2415104 or not np.all(np.diff(pts) == 256):
        raise ValueError('Official native clock differs')
    # Reuse the original initializer's sensor witnesses, with exactly the same
    # mask contract.  They are dense numerical navigation, not source corners.
    pixels = np.asarray([(u, v) for v in (180., 360., 540., 720., 900.)
                         for u in (180., 440., 700., 960., 1220., 1480., 1740.)])
    surfaces = baseline.entity_paths('board.contact') + baseline.entity_paths('board.top')
    static_world = []
    static_entity = []
    entity_names = ['ice:numerical_sensor_plane', 'board.contact', 'board.top']
    for name, path in zip(('board.contact', 'board.top'), surfaces):
        xyz = np.asarray(path['xyz'], float) / L
        # Original paths repeat the closing vertex.  Keep its explicit index;
        # no geometry or source row is modified by this reference producer.
        static_world.append(xyz)
        static_entity.extend([entity_names.index(name)] * len(xyz))
    static_world = np.concatenate(static_world)
    static_entity = np.asarray(static_entity, np.int16)
    worlds = [static_world]
    world_entity = [static_entity]
    frame_rows = []
    point_rows = []
    target_rows = []
    entity_rows = []
    role_rows = []
    frame_summary = []
    cursor = len(static_world)
    # The old 3D boundary is a numerical reference.  It has not acquired source
    # occlusion/visibility certification from merely projecting into the sensor.
    allowed = np.asarray(seed.a['navigation_role_allowed'], bool)
    seed_index = {int(p): i for i, p in enumerate(seed.pts)}
    for fi, (p, sn) in enumerate(zip(pts, shots)):
        p, sn = int(p), int(sn)
        sid = f's{sn:02d}'
        provenance = baseline.provenance(p)
        if sn == 1:
            # S01 remains owned by the full d78 forward.  The old 8a28 state is
            # not silently substituted for that stronger existing state.
            frame_summary.append(dict(native_pts=p, shot_id=sid, navigation_rows=0,
                reason='S01_complete_d78_owner; independent source comparison required',
                bridge_mode=provenance['bridge_mode']))
            continue
        si = seed_index[p]
        xy, inverse_valid = seed.inverse_native_uv(p, sid, pixels)
        ice = np.c_[xy, np.zeros(len(xy))]
        points = np.concatenate((ice, static_world))
        uv, depth, valid = baseline.project(p, points * L,
            scene_sha256=baseline.scene_sha256, stage='demo_source',
            optics='original_brown', shot_id=sid)
        eligible = (np.asarray(valid, bool) & np.isfinite(uv).all(-1)
            & (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920)
            & (uv[:, 1] >= 0) & (uv[:, 1] < 1080))
        eligible[:len(ice)] &= inverse_valid
        index = np.flatnonzero(eligible)
        ids = np.r_[np.arange(cursor, cursor+len(ice)), np.arange(len(static_world))]
        entities = np.r_[np.zeros(len(ice), np.int16), static_entity]
        worlds.append(ice)
        world_entity.append(np.zeros(len(ice), np.int16))
        cursor += len(ice)
        frame_rows.append(np.full(len(index), fi, np.int32))
        point_rows.append(ids[index].astype(np.int32))
        target_rows.append(uv[index])
        entity_rows.append(entities[index])
        role_rows.append(np.full(len(index), bool(allowed[si]), bool))
        frame_summary.append(dict(native_pts=p, shot_id=sid,
            numerical_surface_denominator=len(points), sensor_visible_rows=len(index),
            outside_sensor_or_nonpositive_depth=int((~eligible).sum()),
            navigation_role_allowed=bool(allowed[si]), bridge_mode=provenance['bridge_mode'],
            inferred=provenance['inferred'], source_visibility_qualified=False))
    out.mkdir(parents=True)
    arrays = dict(native_pts=pts, shot_number=shots,
        world_xyz=np.concatenate(worlds), world_entity=np.concatenate(world_entity),
        row_frame_index=np.concatenate(frame_rows), row_world_index=np.concatenate(point_rows),
        target_uv=np.concatenate(target_rows), row_entity=np.concatenate(entity_rows),
        row_navigation_allowed=np.concatenate(role_rows),
        bridge_mode=np.asarray(baseline.packet['bridge_mode']),
        direct_mapping=np.asarray(baseline.packet['direct_mapping']),
        inferred=np.asarray(baseline.packet['inferred']))
    if len(arrays['target_uv']) != len(arrays['row_world_index']):
        raise ValueError('Incomplete numerical navigation rows')
    np.savez_compressed(out/'reference.npz', **arrays)
    import gzip
    with gzip.open(out/'native_frames.json.gz', 'wt', encoding='utf8') as stream:
        json.dump(frame_summary, stream, ensure_ascii=False, separators=(',', ':'))
    report = dict(schema='rinkfit.native_3d_surface_navigation_reference.v1',
        arrays=dict(path=str(out/'reference.npz'), sha256=file_sha256(out/'reference.npz')),
        frame_ledger=dict(path=str(out/'native_frames.json.gz'), sha256=file_sha256(out/'native_frames.json.gz')),
        screen_state=dict(path=str(Path(screen_state_file).resolve()), sha256=file_sha256(screen_state_file)),
        baseline=baseline.references, template_model=template,
        baseline_camera_stage='demo_source; V3 patch excluded',
        baseline_optics='complete_original_Brown',
        native_clock=dict(time_base=[1,12800], fps=[50,1], first_pts=int(pts[0]), last_pts=int(pts[-1]), step_pts=256),
        world_chart=dict(normalization_L=L, axes='historical XYZ unchanged', origin=[0.,0.,0.],
                         scene_sha256=baseline.scene_sha256),
        entity_names=entity_names, frames=len(pts), numerical_rows=len(arrays['target_uv']),
        unique_world_points=len(arrays['world_xyz']),
        board_z_range=(float(static_world[:,2].min()), float(static_world[:,2].max())),
        source_observations_created=0, source_UV_or_generation_modified=False,
        historical_K_R_C_imported_to_physical_model=False, ice_H_used_for_board=False,
        navigation_reference_only=True, physical_seed_generated=False, accepted=False,
        unresolved=['S01 remains full d78, not the old 8a28 plane reference',
                    'Historical-to-native bridge uncertainty retained',
                    'Projected board points have no source occlusion certification',
                    'Real material/advertisement crosswalk is required before likelihood use'])
    write_json(out/'reference.json', report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen-state',required=True)
    p.add_argument('--baseline-state',required=True)
    p.add_argument('--template-model')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    print(json.dumps(build_reference(a.screen_state,a.baseline_state,a.out,
        template_model_file=a.template_model), ensure_ascii=False))


if __name__=='__main__': main()
