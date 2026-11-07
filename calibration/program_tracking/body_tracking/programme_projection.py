"""Save native five-node shoe/blade predictions on the full programme PTS.

Only saved final states, the existing frozen camera and explicit old source
rows are consumed. This module never solves, infers poses, opens a GUI or
prepares full display meshes. Every invocation requires a new output folder.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / 'src')]
import numpy as np
from .rebuild_prepare import load, digest
from .absolute_motion_state import curve_from_packet
from .sequence_skate_state import shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection
from .skate_template import profile_height
from .blade_contact_modes import blade_axes_diagnostics
from .programme_contract import required_clock

PROGRAMME = Path('events/ankara_2026/programmes/okada_mei_sp')
VARIANT = 'programme_trajectory'
SOURCE_FIELDS = ('source_structures', 'source_regions', 'source_visibility',
                 'source_steel_segments', 'source_plane_locations', 'source_ice_grid', 'source_boot_axes')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def difference(actual, expected, label, maximum, tolerance=2e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid saved-state replay comparison: ' + label)
    error = float(np.max(abs(a - b)))
    maximum[label] = max(maximum.get(label, 0.), error)
    if error > tolerance:
        raise ValueError('Final parameter/result state differs: ' + label + ' ' + str(error))


def source_rows(project, assets, pts, *, review_evidence=None):
    """Copy explicitly saved source fields verbatim; retain per-row IDs/status."""
    config_path = project / PROGRAMME / 'viewer3d_source_fusion_support_1106.json'
    config = read(config_path)
    path = assets.resolve(config['blade_review_asset'], base=config_path.parent, must_exist=True)
    original = read(path)
    if tuple(original.get('time_base', ())) != (1, 12800):
        raise ValueError('Existing source review asset uses another source clock')
    result = {key: copy.deepcopy(original.get(key, [])) for key in SOURCE_FIELDS}
    lookup = {int(p): i for i, p in enumerate(pts)}
    additions=[]
    if review_evidence is not None:
        evidence=read(review_evidence)
        for episode in evidence.get('additional_unassigned_source_episodes',[]):
            source_path=Path(episode['source_file'])
            if digest(source_path)!=episode['source_sha256']:
                raise ValueError('Additional source review file changed: '+str(source_path))
            raw=read(source_path)
            if tuple(raw.get('time_base',()))!=(1,12800):
                raise ValueError('Additional source review uses another clock')
            admitted={int(g['pts'])for g in episode['groups']}
            rows=raw.get('frames',[])+raw.get('unassigned_identity_observations',[])
            for row in rows:
                if int(row['pts'])not in admitted:continue
                value=copy.deepcopy(row)
                # Association to a model foot is a latent fit choice, never a
                # new anatomical identity in the source image overlay.
                value.update(foot=row.get('foot'),source_episode_id=episode['id'],
                    overlay_role='source_review; candidate participation separately registered')
                result['source_structures'].append(value)
                additions.append(dict(episode_id=episode['id'],pts=int(row['pts']),
                    source_track_id=row.get('source_track_id'),source_file=str(source_path),
                    source_file_sha256=episode['source_sha256']))
    status = np.full((len(pts), 2), 'unknown', dtype='U48')
    for key, rows in result.items():
        for row in rows:
            stamp = row.get('pts')
            if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp not in lookup:
                raise ValueError('Saved source row lies outside the exact programme PTS: ' + key)
            foot = row.get('foot')
            if foot in (0, 1) and not isinstance(foot, bool):
                status[lookup[stamp], foot] = 'existing_saved_source_rows_present'
    binding = dict(path=str(path), sha256=digest(path),
        configuration=dict(path=str(config_path), sha256=digest(config_path)),
        copied_fields=list(SOURCE_FIELDS), copied_counts={k: len(v) for k, v in result.items()},
        original_source_ids_pts_roles_visibility_uncertainty_preserved=True,
        no_interpolation=True, new_source_observations=len(additions),additional_source_rows=additions,
        review_evidence=None if review_evidence is None else dict(path=str(review_evidence),sha256=digest(review_evidence)))
    return result, status, binding


def fit_participation(prepared,parameter,*,saved=None,document=None):
    """Register admitted factors only when their actual saved objective binds them."""
    prepared,parameter=Path(prepared),Path(parameter)
    saved=load(parameter)if saved is None else saved
    document=read(prepared/'source_evidence.json')if document is None else document
    evidence_sha=digest(prepared/'source_evidence.json')
    if str(saved.get('source_evidence_sha256',''))!=evidence_sha:
        raise ValueError('Saved candidate does not bind this prepared source evidence')
    weights=load(parameter.parent.parent/'objective_block_weights.npz')
    feature_rows=document.get('full_source_skate_structures',{}).get('rows',[])if float(weights.get('full_source_skate_structure',0.))>0 else []
    feature_rows=[{k:copy.deepcopy(row[k])for k in('id','pts','foot','structure_feature','source_status')if k in row}
        for row in feature_rows if row.get('optimization_enabled')and row.get('role')=='fit']
    support_rows=[{k:copy.deepcopy(row[k])for k in('pts','foot','phase')if k in row}
        for row in document.get('source_support_evidence',[]) if row.get('optimization_enabled')and row.get('role')=='fit']
    if float(weights.get('native_source_support_contact',0.))<=0:support_rows=[]
    extra=[]
    for episode in document.get('additional_unassigned_source_episodes',[]):
        # This block has the same finite default weight as SequenceActions.
        if float(weights.get('additional_unassigned_source_'+episode['id'],1.))<=0:continue
        for group in episode['groups']:
            counts=[sum(bool(r.get('optimization_enabled'))and r.get('role')=='fit'for r in b.get('rows',[]))
                for b in group.get('branches',[])]
            extra.append(dict(pts=group['pts'],episode_id=episode['id'],feature_rows_count=max(counts,default=0),
                source_anatomical_identity='unknown',support_enabled=bool(episode.get('support_enabled',False))))
    return dict(description_zh='入拟合范围按本候选实际保存的目标登记；局部源描线不代表完整冰刀或已确认接冰。',
        source_evidence_sha256=evidence_sha,objective_weights_sha256=digest(parameter.parent.parent/'objective_block_weights.npz'),
        feature_rows=feature_rows,support_rows=support_rows,
        unassigned_source_sets=[dict(pts=row['pts'],feature_rows_count=max((len(b.get('rows',[]))for b in row.get('branches',[])),default=0))
            for row in document.get('unassigned_source_skate_sets',[]) if 'pts'in row]+extra)


def generate(project, prepared, parameter, result, output, *, variant=VARIANT, review_evidence=None, batch_frames=512):
    project, prepared, parameter, result, output = map(lambda p: Path(p).resolve(),
                                                     (project, prepared, parameter, result, output))
    from rinkspace.storage import AssetStore
    from .trajectory_backend import load_reference
    from rinkspace.viewer3d.blade_review import BladeReview
    assets = AssetStore(project)
    if type(batch_frames)is not int or batch_frames<1:
        raise ValueError('A positive projection batch size is required')
    if output.exists():
        raise FileExistsError('Every projection registration requires a new output directory: ' + str(output))
    if not output.is_relative_to(Path(assets.data).resolve()):
        raise ValueError('Projection artifacts must stay under the configured Data root')
    hashes = {str(p): digest(p) for p in (prepared / 'input.npz', prepared / 'observations.npz',
              prepared / 'initial.npz', prepared / 'fixed_binding.npz', parameter, result)}
    saved, exported = load(parameter), load(result)
    for field, name in (('prepared_input_sha256', 'input.npz'),
                        ('prepared_observations_sha256', 'observations.npz')):
        if str(saved[field]) != hashes[str(prepared / name)]:
            raise ValueError('Saved final parameter differs from prepared identity: ' + field)
    ref, camera, info, source_document, _ = load_reference(project, prepared)
    validate_checkpoint_identity(saved, ref.fixed_checkpoint_identity)
    # Reuse the camera returned by the explicit prepared loader. No camera is
    # created, resampled or updated by the final-state projection path below.
    ref.absolute_curve = curve_from_packet(saved, seed=False)
    ref.relative_skate_curve = shoe_curve_from_packet(saved, seed=False)
    if ref.relative_skate_curve is None:
        raise ValueError('Final candidate has no saved relative-shoe curve')
    ref.binding.x = np.asarray(saved['x'][:17]).copy()
    ref.binding._parameter_key = ref.binding._packet = None
    ref.binding.reference = ref
    pts = np.asarray(ref.p['pts'])
    expected, shots, tb = required_clock(read(project / PROGRAMME / 'rebuild_scope.json'))
    np.testing.assert_array_equal(pts, expected)
    np.testing.assert_array_equal(exported['pts'], pts)
    if 'source_pts' in exported:
        np.testing.assert_array_equal(exported['source_pts'], pts)
    np.testing.assert_array_equal(ref.p['time_base'], tb)
    np.testing.assert_array_equal(ref.p['shot_ids'], shots)
    np.testing.assert_array_equal(exported['time_base'], tb)
    if len(pts) != 8329 or len(set(shots.tolist())) != 9:
        raise ValueError('Require complete nine-shot 8329-exposure saved inputs')
    times = (pts - pts[0]).astype(float) * tb[0] / tb[1]
    np.testing.assert_allclose(ref.times, times, rtol=0, atol=1e-12)
    ratio = float(ref.binding.ratio)
    np.testing.assert_allclose(exported['body_length_ratio'], ratio, rtol=0, atol=1e-12)
    np.testing.assert_allclose(exported['metres_per_scene_unit'], info['metres_per_scene_unit'], rtol=1e-12)
    geometry = ref.binding.packet()
    p, adapter, edges = geometry['parameters'], np.asarray(geometry['adapter']), np.asarray(geometry['edges'])
    if edges.ndim != 3 or edges.shape[0] != 2 or edges.shape[-1] != 3:
        raise ValueError('Expected two saved template runner edges')
    rear, front = -p.runner_rear_extension_m, p.sole_length_m - p.runner_front_inset_m
    xs = np.asarray([rear, (rear + front) / 2., front])
    stations = np.column_stack((xs, np.zeros(3), profile_height(xs, p)))
    boot_axis = np.stack((adapter[[0, 7]].mean(0), adapter[[1, 8]].mean(0)))
    upper = edges + np.asarray([0., 0., p.blade_web_height_m])
    local = np.concatenate((stations, boot_axis, edges.reshape(-1, 3), upper.reshape(-1, 3)))
    count_edges = edges.shape[0] * edges.shape[1]
    n = len(pts)
    blade_uv = np.empty((n, 2, 3, 2))
    boot_uv = np.empty((n, 2, 2, 2))
    lower_uv = np.empty((n, 2, *edges.shape[:2], 2))
    upper_uv = np.empty_like(lower_uv)
    rotations = np.empty((n, 2, 3, 3))
    origins = np.empty((n, 2, 3))
    materials = np.empty((n, 2, 3, 3))
    readout = np.empty(n)
    replay_errors = {}
    for first in range(0, n, batch_frames):
        frames = np.arange(first, min(first + batch_frames, n))
        size = len(frames)
        tau = np.asarray(camera.readout_times(frames))
        if tau.shape != (size,) or not np.isfinite(tau).all():
            raise ValueError('Expected exact per-exposure frozen camera readout')
        node_times = times[frames, None] + tau[:, None] * np.linspace(-.5, .5, 5)[None]
        state = fk_state(ref, ref.at(node_times.ravel()), np.zeros((size * 5, 69)))
        R = state['R'].reshape(size, 5, 2, 3, 3)
        O = state['O'].reshape(size, 5, 2, 3)
        # Each foot owns the same physical local template points at all five
        # node times. They are simultaneous body states, not mid-point repeats.
        world = O[:, :, :, None] + np.einsum('nlfij,kj->nlfki', R, local)
        uv = moving_projection(camera, world.reshape(size, 5, 2 * len(local), 3), frames)['uv']
        uv = uv.reshape(size, 2, len(local), 2)
        blade_uv[frames], boot_uv[frames] = uv[:, :, :3], uv[:, :, 3:5]
        lower_uv[frames] = uv[:, :, 5:5 + count_edges].reshape(size, 2, *edges.shape[:2], 2)
        upper_uv[frames] = uv[:, :, 5 + count_edges:].reshape(size, 2, *edges.shape[:2], 2)
        midR, midO = R[:, 2], O[:, 2]
        rotations[frames], origins[frames], readout[frames] = midR, midO * ratio, tau
        materials[frames] = (midO[:, :, None] + np.einsum('nfij,kj->nfki', midR, stations)) * ratio
        difference(state['P'].reshape(size, 5, 3)[:, 2] * ratio,
                   exported['P_world_mid'][frames], 'P_world_mid_m', replay_errors)
        difference(state['Q'].reshape(size, 5, 3, 3)[:, 2],
                   exported['Q_world_mid'][frames], 'Q_world_mid', replay_errors)
        difference(state['local_R'].reshape(size, 5, 21, 3, 3)[:, 2],
                   exported['local_R_mid'][frames], 'local_R_mid', replay_errors)
        if 'skate_relative_R_mid' in exported:
            difference(state['skate_relative_R'].reshape(size, 5, 2, 3, 3)[:, 2],
                       exported['skate_relative_R_mid'][frames], 'skate_relative_R_mid', replay_errors)
        if 'hand_local_R_mid' in exported:
            difference(state['hand_local_R'].reshape(size, 5, 2, 15, 3, 3)[:, 2],
                       exported['hand_local_R_mid'][frames], 'hand_local_R_mid', replay_errors)
        difference(midR, exported['skate_R_world_mid'][frames], 'skate_R_world_mid', replay_errors)
        difference(midO * ratio, exported['skate_origin_world_mid'][frames], 'skate_origin_world_mid_m', replay_errors)
        print(json.dumps(dict(event='programme_projection', completed=int(frames[-1] + 1), total=n)), flush=True)
    source, source_status, source_binding = source_rows(project, assets, pts,
        review_evidence=prepared/'source_evidence.json'if review_evidence is None else review_evidence)
    # Participation comes from this saved objective/input, never from the mere
    # presence of an overlay copied for review.
    fit_binding=fit_participation(prepared,parameter,saved=saved,document=source_document)
    axes = blade_axes_diagnostics(rotations)
    rates = np.full((n, 2), np.nan)
    F = rotations[..., :, 0]
    rates[1:] = np.arccos(np.clip(np.sum(F[1:] * F[:-1], axis=-1), -1., 1.)) / (
        np.diff(pts).astype(float) * tb[0] / tb[1])[:, None]
    coco = np.asarray(exported['COCO_uv'])
    if coco.shape != (n, 17, 2) or not np.isfinite(coco).all():
        raise ValueError('Saved same-chain result COCO_uv is invalid')
    values = dict(blade_uv=blade_uv, boot_uv=boot_uv, runner_edges_uv=lower_uv,
        runner_upper_edges_uv=upper_uv, axis_rate_rad_s=rates,
        pitch_degrees=axes['fore_aft_pitch_degrees'], roll_degrees=axes['edge_bank_degrees'],
        station_height_m=materials[..., 2], material_station_world_conditional_m=materials,
        R_world=rotations, origin_world_conditional_m=origins, COCO_uv=coco.copy(),
        source_shoe_evidence_status=source_status,
        source_contact_truth_status=np.full((n, 2), 'unmeasured', dtype='U16'),
        contact_mode=np.full((n, 2), 'unknown', dtype='U16'),
        edge_mode=np.full((n, 2), 'unknown', dtype='U16'), mode_conflict=np.zeros((n, 2), bool), **axes)
    arrays = dict(pts=pts.copy(), time_base=np.asarray(tb), shot_ids=shots.copy(), readout_seconds=readout)
    if not variant or not variant.replace('_', '').isalnum():
        raise ValueError('A stable projection variant identifier is required')
    arrays.update({variant + '_' + key: value for key, value in values.items()})
    for key, value in arrays.items():
        if value.dtype.kind in 'fiu' and key != variant + '_axis_rate_rad_s' and not np.isfinite(value).all():
            raise ValueError('Nonfinite saved projection field: ' + key)
    if not np.isnan(rates[0]).all() or not np.isfinite(rates[1:]).all():
        raise ValueError('Only the first direction-rate sample must remain unknown')
    for path, expected_sha in hashes.items():
        if digest(path) != expected_sha:
            raise ValueError('A saved input changed during projection: ' + path)
    if digest(source_binding['path']) != source_binding['sha256']:
        raise ValueError('Existing source review asset changed during projection')
    review_binding=source_binding['review_evidence']
    if digest(review_binding['path'])!=review_binding['sha256']:
        raise ValueError('Additional source evidence changed during projection')
    output.mkdir(parents=True)
    npz_path, json_path = output / 'programme_blade_review.npz', output / 'programme_blade_review.json'
    np.savez_compressed(npz_path, **arrays)
    document = dict(schema='rinkspace.blade-player-review.v1', time_base=list(tb),
        projection_npz=npz_path.name, variants={variant: variant}, **source,
        source_evidence_binding=source_binding,
        source_structure_fit_variants={variant:fit_binding},
        interpretation='Model UV, axes and station heights are saved candidate predictions; copied source rows retain their original identity/status. No source measurements or contact truth are generated.',
        source_foot_identity='Existing saved foot 0/1 convention only; anatomical identity is not newly inferred',
        source_missing_policy='All other exposure/foot source-shoe evidence remains unknown; source rows are never propagated',
        projection_method='Same frozen ProgrammeCamera, exact per-exposure signed readout, five articulated FK nodes and native moving_projection implicit predicted row',
        fixed_material_stations=['rear', 'middle', 'front'],
        COCO_uv_origin='Copied from the same-chain saved result; midpoint state replay verified, COCO pixels not independently re-estimated',
        prediction_binding=dict(parameter=str(parameter), parameter_sha256=hashes[str(parameter)],
            result=str(result), result_sha256=hashes[str(result)]),
        model_source_separation='Model diagnostics are predictions. Source evidence/unknown statuses remain independent.')
    write(json_path, document)
    BladeReview(json_path, assets=assets, time_base=tb)
    report = dict(schema='rinkspace.programme-five-node-projection.v1', frames=n, shots=9,
        time_base=list(tb), first_last_pts=pts[[0, -1]].tolist(), batch_frames=batch_frames, exposure_nodes=5,
        prepared=str(prepared), parameter=str(parameter), result=str(result), source_file_sha256=hashes,
        source_evidence_binding=source_binding, saved_midpoint_replay_maximum_errors=replay_errors,
        output_asset=dict(path=str(json_path), sha256=digest(json_path)),
        output_projection=dict(path=str(npz_path), sha256=digest(npz_path)),
        original_camera_identity_preserved=True, fixed_identity_verified=True,
        saved_model_predictions_not_source_observations=True,
        source_unknown_foot_exposures=int(np.count_nonzero(source_status == 'unknown')),
        no_fit=True, no_neural_inference=True, no_GUI=True, no_mesh_preparation=True,
        code=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__).resolve())))
    write(output / 'projection_registration.json', report)
    return report


def merge_review_assets(paths, output, *, primary_variant=VARIANT):
    """Combine separately replayed candidates, preserving source rows verbatim."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    arrays={};variants={};bindings={};fit_bindings={};primary=None;source=None;inputs=[]
    for entry in paths:
        entry=Path(entry).resolve();doc=read(entry)
        content={k:doc.get(k,[]) for k in SOURCE_FIELDS}
        if source is None:source=content
        elif source!=content:raise ValueError('Candidates must use the same unchanged source evidence')
        values=load(entry.parent/doc['projection_npz'])
        for key,value in values.items():
            if key in arrays:np.testing.assert_array_equal(value,arrays[key],err_msg='Different projection clock/readout: '+key)
            else:arrays[key]=value
        for variant,prefix in doc['variants'].items():
            if variant in variants:raise ValueError('Duplicate independently projected variant: '+variant)
            variants[variant]=prefix;bindings[variant]=doc['prediction_binding']
            if variant==primary_variant:primary=copy.deepcopy(doc)
        fit_bindings.update(doc.get('source_structure_fit_variants',{}))
        inputs.append(dict(path=str(entry),sha256=digest(entry)))
    if primary is None:raise ValueError('Primary candidate was not independently projected')
    output.mkdir(parents=True)
    npz=output/'programme_blade_review.npz';target=output/'programme_blade_review.json'
    np.savez_compressed(npz,**arrays)
    primary.update(projection_npz=npz.name,variants=variants,prediction_bindings=bindings,
        source_structure_fit_variants=fit_bindings,
        independently_replayed_assets=inputs)
    write(target,primary)
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.blade_review import BladeReview
    project=Path(__file__).resolve().parents[3]
    BladeReview(target,assets=AssetStore(project),time_base=tuple(primary['time_base']))
    return target


def render_comparison(project,asset,source_video,variants,source_pts,output):
    """Static native-source comparisons using the player's exact overlay code."""
    from PIL import Image,ImageDraw
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.source import SourceVideo
    from rinkspace.viewer3d.blade_review import BladeReview,overlay_primitives
    import hashlib
    project,asset,source_video,output=map(lambda p:Path(p).resolve(),(project,asset,source_video,output))
    store=AssetStore(project)
    if output.exists():raise FileExistsError(output)
    if not output.is_relative_to(store.data.resolve()):raise ValueError('Review images must stay in Data')
    doc=read(asset);review=BladeReview(asset,assets=store,time_base=tuple(doc['time_base']))
    if any(v not in review.models for v in variants):raise ValueError('Every reviewed model needs its own saved projections')
    if any(int(p)not in review.index for p in source_pts):raise ValueError('Review PTS outside the saved source clock')
    output.mkdir();source=SourceVideo(source_video,tuple(doc['time_base']));panels=[];rows=[]
    try:
        for stamp in source_pts:
            stamp=int(stamp);rgb=source.read(stamp);row=[]
            for variant in variants:
                im=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(im)
                current=review.at(variant,stamp)
                for item in overlay_primitives(current,include_model_reference=False,include_model_contours=True,
                        include_model_body=True,include_source_structures=True,include_source_legacy=False):
                    points=[tuple(v)for v in item['points']];color=tuple(int(round(255*v))for v in item['color'][:3])
                    if item['kind']=='line':draw.line(points,fill=color,width=max(1,int(round(item['width']))))
                    elif item['kind']=='text':draw.text(points[0],item['text'],fill=color)
                path=output/f'{variant}_{stamp}.png';im.save(path)
                tile=Image.new('RGB',(960,580),(25,25,25));im.thumbnail((960,540));tile.paste(im,(0,40))
                ImageDraw.Draw(tile).text((8,8),f'{stamp/12800.:.2f}s | {variant} | purple=body; pink=steel; other colors=source',(240,240,240))
                row.append(tile);rows.append(dict(variant=variant,pts=stamp,image=str(path),
                    native_source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            panels.append(row)
    finally:source.close()
    sheet=Image.new('RGB',(960*len(variants),580*len(panels)))
    for i,row in enumerate(panels):
        for j,tile in enumerate(row):sheet.paste(tile,(j*960,i*580))
    sheet.save(output/'same_frame_comparison.jpg',quality=92)
    write(output/'review_images.json',dict(asset=str(asset),asset_sha256=digest(asset),source_video=str(source_video),
        same_overlay_function_as_player=True,variants=list(variants),rows=rows,
        interpretation='Saved model body and actual dual steel-edge projections; source layers preserve their own partial/ambiguous status. No observation adapters are drawn. No 3D/contact accuracy acceptance.'))
    return output/'same_frame_comparison.jpg'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'prepared', 'parameter', 'result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant',default=VARIANT)
    parser.add_argument('--review-evidence',type=Path,help='Source-only review overlay; never changes a candidate\'s saved fit participation')
    parser.add_argument('--batch-frames',type=int,default=512)
    args = parser.parse_args(argv)
    report = generate(args.project, args.prepared, args.parameter, args.result, args.output,variant=args.variant,review_evidence=args.review_evidence,batch_frames=args.batch_frames)
    print(json.dumps(dict(asset=report['output_asset']['path'], frames=report['frames'])), flush=True)


if __name__ == '__main__':
    main()
