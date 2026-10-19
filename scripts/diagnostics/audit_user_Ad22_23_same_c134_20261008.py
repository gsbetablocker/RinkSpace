"""Small same-state print/placement/source-clock audit of user issue 06.

No optimizer, full motion graph or Jacobian is constructed. RGB ink bounding
boxes are diagnostic descriptions, never new FIT measurements or weights.
"""
from pathlib import Path
import hashlib, json, os, re, sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]
sys.dont_write_bytecode = True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2, numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry

RUNS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
DIRECTORY = RUNS / 'okada_unlocked_geometry_af8cb_board739_canonical_gate_20261008/accepted_states/c1346ee37dc963340b5c8c1c1cc4c8fe281f8e20b6bcba86bdafaecd53f7af99'
EVIDENCE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261008/c134_full_programme')
PTS = 230036992


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p): return dict(path=str(p), sha256=sha(p))
def bbox(uv):
    uv = np.asarray(uv, float)
    return [*uv.min(axis=0).tolist(), *uv.max(axis=0).tolist()]


def source_ink_bounds(image, number):
    # Explicit source RGB scope, independently seen whole printed components.
    bounds = {22:(400,365,595,462), 23:(75,370,355,450), 20:(1150,368,1650,515)}[number]
    x0,y0,x1,y1 = bounds; hsv = cv2.cvtColor(image[y0:y1,x0:x1], cv2.COLOR_BGR2HSV)
    variants = []
    for value in (140,160,180):
        if number == 22:
            mask = ((hsv[:,:,0] < 13) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] > 35) & (hsv[:,:,2] < 245)
        elif number == 23:
            red = ((hsv[:,:,0] < 13) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] > 45) & (hsv[:,:,2] < 245)
            mask = red | (hsv[:,:,2] < value)
        else:
            mask = (hsv[:,:,0] > 90) & (hsv[:,:,0] < 130) & (hsv[:,:,1] > 40) & (hsv[:,:,2] < 245)
            # Separate the broad blue ISU printed panel from the thin cap.
            count, labels, components, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            assert count > 1
            component = 1 + int(np.argmax(components[1:,cv2.CC_STAT_AREA]))
            mask = labels == component
        y,x = np.where(mask)
        assert len(x)
        variants.append(dict(dark_value_threshold=value, native_ink_bbox_xyxy=[int(x.min()+x0),int(y.min()+y0),int(x.max()+x0),int(y.max()+y0)]))
    return dict(explicit_actual_RGB_ROI_xyxy=bounds, variants=variants,
        selected_people_or_UI_in_ROI=False, approximate_blurred_ink_extent_only=True,
        new_FIT_UV_sigma_or_exact_boundary_measurement=False)


def main():
    identity = read(DIRECTORY / 'state_identity.json')
    scene_path = DIRECTORY / 'scene_UNACCEPTED.json'
    assert sha(scene_path) == identity['scene_sha256']
    assert sha(DIRECTORY / 'complete_state_UNACCEPTED.npz') == identity['state_npz_sha256']
    assert sha(identity['camera_path']) == identity['camera_sha256']
    provider, world, proof = current_complete_source_geometry(DIRECTORY / 'complete_state_UNACCEPTED.npz',
        RUNS / 'user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json')
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    from z190_physical_camera import PhysicalCameraModel
    model, camera_pts, aliases, views = PhysicalCameraModel.load_npz(identity['camera_path'])
    scene = read(scene_path); saved_review = read(DIRECTORY / 'source_review_UNACCEPTED.json')
    tracks = saved_review['additional_native_static_board_tracks']
    packet_path = Path(tracks['contract']['packet_path'])
    assert sha(packet_path) == tracks['contract']['packet_sha256']
    packet = read(packet_path)
    cameras = {}; camera_archive = {}
    with np.load(identity['camera_path'], allow_pickle=False) as a:
        for p, c in zip(a['pts'], a['camera25']): camera_archive[int(p)] = c
    def camera(p):
        p = int(p); actual = int(aliases[p])
        if actual not in cameras: cameras[actual] = model.camera25(views[actual])
        np.testing.assert_array_equal(cameras[actual], camera_archive[p])
        return cameras[actual]
    native_camera = camera(PTS)
    print_scene = {p['advertisement_id']:p for p in scene['source_conditioned_board_prints']['patterns']}
    ads = {int(p['user_number']):p for p in scene['advertisements']}
    report = dict(schema='rinkspace.user-Ad22-Ad23-c134-same-producer-small-audit.v1',
        issue_id='06_Ad22_Ad23_large_shift', source_pts=PTS, actual_exposure_pts=int(aliases[PTS]),
        time_base='1/15360', exact_original_RGB=ref(EVIDENCE / f'06_Ad22_Ad23_large_shift_source_native_{PTS}.png'),
        exact_current_video_frame=ref(EVIDENCE / '06_Ad22_Ad23_large_shift_current_c134_video_frame.png'),
        state_identity=identity, world_proof=proof, source_packet=ref(packet_path),
        camera25_archive_and_actual_model_reproduced_bitexact=True,
        original_UV_sigma_SPLIT_camera_world_parameters_modified=False,
        no_solver_full_motion_graph_or_J_built=True, no_source_promotion=True,
        model_or_whole_video_accuracy_accepted=False, advertisements=[], source_anchors=[],
        native_track_time_coverage=[])
    source = cv2.imread(str(Path(report['exact_original_RGB']['path']))); assert source.shape == (1080,1920,3)
    for number in (20,21,22,23):
        ad = ads[number]
        if number in print_scene:
            literal = print_scene[number]
            template = provider.current.source_print_templates[ad['source_template_id']]
            producer_world, saved_world = [], []
            for c, exported in zip(template['external_contours'], literal['external_contours']):
                uv = np.asarray(c['initialization']['material_coordinates_normalized'],float)
                if c.get('closed') and not np.array_equal(uv[0],uv[-1]): uv = np.vstack((uv,uv[0]))
                xyz = provider.current.source_template_material(number,*uv.T,world)[0]
                np.testing.assert_array_equal(xyz, exported['xyz_scene_units'])
                producer_world.append(xyz); saved_world.append(np.asarray(exported['xyz_scene_units']))
            indices = template['instance_location_offset_indices'][number]
            dimensions = template['width_height_offset_indices']
            datum = np.asarray(template['initialization']['instance_datum_station_zmin_m'][str(number)])
            info = dict(source_template_id=ad['source_template_id'],
                shared_width_height_working_m=provider.current._size(number,world)[:2],
                dimension_offset_logical_indices=dimensions.tolist(), dimension_offsets=world[dimensions].tolist(),
                instance_station_zmin_logical_indices=indices.tolist(), independent_station_zmin_offsets=world[indices].tolist(),
                initialized_station_zmin_working_m=datum.tolist(), current_station_zmin_working_m=(datum+world[indices]).tolist(),
                per_instance_location_is_estimable=True, counterpart_location_not_forced_halfturn=True,
                live_producer_and_exact_render_descriptor_world_bitexact=True)
            xyz = np.concatenate(producer_world)
        else:
            xyz = np.asarray(list(ad['corners_xyz_scene_units'].values()),float)
            info = dict(rectangular_print_extent=True)
        projected, _, valid = project_central_valid_full_brown(native_camera,xyz)
        assert valid.all()
        item = dict(number=number, actual_production_projected_bbox_native_xyxy=bbox(projected),
            current_projection_native_UV=projected.tolist(), geometry=info)
        if number in (20,22,23): item['source_only_ink_extent'] = source_ink_bounds(source,number)
        if number == 21: item['source_context'] = 'WYNDHAM blue square icon visible; much of text occluded by skater, no whole-print bbox qualification.'
        report['advertisements'].append(item)
    saved_anchor = {r['id']:r for r in tracks['rows'] if r['actual_source_anchor']}
    for row in packet['source_anchor_observations']:
        number = int(row['advertisement_number'])
        if number not in (22,23): continue
        material = np.asarray(row['initialization']['material_coordinates_normalized'],float)
        xyz = provider.current.source_template_material(number,*material.T,world)[0]
        projected, _, valid = project_central_valid_full_brown(camera(row['actual_pts']),xyz); assert valid.all()
        errors = projected - np.asarray(row['uv'],float)
        expected = saved_anchor[row['id']]
        np.testing.assert_allclose(projected, expected['projected_UV'], rtol=0, atol=1e-10)
        d = np.linalg.norm(errors,axis=1)
        report['source_anchors'].append(dict(id=row['id'],number=number,source_native_pts=row['native_pts'],
            actual_pts=row['actual_pts'],split=row['split'],points=len(errors),
            seconds_after_user_issue=(row['actual_pts']-PTS)/15360.,median_native_px=float(np.median(d)),
            maximum_native_px=float(d.max()),signed_median_xy_px=np.median(errors,axis=0).tolist(),
            source_raw_frame=row['source_raw_frame'], actual_saved_source_operator_projection_max_difference=float(abs(projected-expected['projected_UV']).max())))
    for number in (22,23):
        rows = [r for r in packet['rows'] if int(r.get('advertisement_id', -1)) == number]
        stamps = sorted({int(r['source_pts']) for r in rows} | {
            int(r['target_pts']) for r in rows if int(r.get('counterpart_advertisement_id',number)) == number})
        report['native_track_time_coverage'].append(dict(number=number,source_native_pts=stamps,
            minimum_seconds_from_issue=min(abs(p-PTS)/15360. for p in stamps),
            user_issue_native_pts_is_scored_endpoint=PTS in stamps,
            FIT_groups=sum(r['split']=='fit' for r in rows),HELD_groups=sum(r['split']=='holdout' for r in rows)))
    from verify_original225_derived_center_20261008 import numeric_data_loads
    from structured_programme_source_candidate import original_review_source
    assert sha(original_review_source.PACKET) == original_review_source.PACKET_SHA256
    compact = numeric_data_loads(original_review_source.PACKET.read_bytes())
    absolute_inventory = []
    for item in compact['programme_absolute_rows']:
        row = item['source_row']
        number = re.match(r'ad(20|21|22|23)(?:[:_]|$)', row['id'], re.I)
        if number is None: continue
        absolute_inventory.append(dict(number=int(number[1]), id=row['id'], block=item['block'],
            collection_role=item['collection_role'], split=row.get('split'),
            group_actual_pts=int(item['group_actual_pts']), points=len(row['uv']),
            seconds_from_user_issue=(int(item['group_actual_pts'])-PTS)/15360.))
    report['original_absolute_Ad20_21_22_23_inventory'] = absolute_inventory
    report['original_absolute_source_inventory_numeric_packet'] = dict(path=str(original_review_source.PACKET),
        sha256=original_review_source.PACKET_SHA256, restricted_numeric_decoder=True, source_operators_not_evaluated=True)
    report['conclusion'] = dict(source_identity_wrong_or_rendering_stale_world_not_demonstrated=True,
        serious_real_issue_confirmed_at_unscored_time=True,
        local_anchor_small_error_does_not_prove_continuous_alignment=True,
        independent_instance_locations_already_free=True,
        shared_size_vs_camera_course_vs_template_remaining_cause_not_identified_by_this_small_audit=True)
    output = EVIDENCE / '06_Ad22_Ad23_same_c134_actual_producer_and_source_coverage_readonly.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=ref(output),projections=[dict(number=r['number'],bbox=r['actual_production_projected_bbox_native_xyxy'],
        source=r.get('source_only_ink_extent')) for r in report['advertisements']], anchors=report['source_anchors'],
        track_time_coverage=report['native_track_time_coverage']),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
