"""Summarize an already computed exact saved-step replay; no model assembly."""
from pathlib import Path
from collections import defaultdict,Counter
import os,json,hashlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import cv2

H=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
SOURCE=H/'current_absolute_F2_B2_related_ads_saved_step_q276_q8a15_causal_response_20261011.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path.resolve()),sha256=sha(path))
def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.ptp(a)<1e-10 or np.ptp(b)<1e-10:return None
    return float(np.corrcoef(a,b)[0,1])
def entity(row):
    if row['physical_group']=='F2':return 'F2:'+row['physical_binding'][2]
    return row['physical_group']

def main():
    report=json.loads(SOURCE.read_text(encoding='utf8'));child=report['child_q']
    identity=json.loads((RUN/'accepted_states'/child/'state_identity.json').read_text(encoding='utf8'))
    camera_path=Path(identity['camera_path']);assert sha(camera_path)==identity['camera_sha256']
    with np.load(camera_path,allow_pickle=False)as z:
        cameras={k:z[k].copy()for k in ('camera25','exposure_representative_pts','installation_id','metadata_json')}
    camera_by_actual={}
    for p,installation,camera in zip(cameras['exposure_representative_pts'],cameras['installation_id'],cameras['camera25']):
        p=int(p)
        if p in camera_by_actual:assert np.array_equal(camera_by_actual[p][1],camera)
        camera_by_actual[p]=(str(installation),camera)
    groups=defaultdict(lambda:defaultdict(list))
    for row in report['rows']:
        installation,camera=camera_by_actual[row['actual_pts']]
        groups[entity(row),installation][row['actual_pts']].append(row)
    out=dict(schema='rinkspace.saved-step-absolute-compensation-by-installation.v1',script=ref(Path(__file__)),
        existing_exact_operator_report=ref(SOURCE),child_camera_archive=ref(camera_path),
        parent_q=report['parent_q'],child_q=child,no_new_model_assembly_fit_source_J_source_decode_or_cache=True,
        camera_orientation_is_saved_world_ray_direction=True,
        focal_is_final_effective_pixel_proxy_not_measured_optical_zoom=True,
        numerical_small_step_cancellation_not_explanation_for_absolute_residual_magnitude=True,
        correlation_is_descriptive_not_identifiability_or_hardware_certificate=True,
        groups=[])
    for (physical,installation),bytime in sorted(groups.items()):
        times=[];reference_R=None
        for at,rows in sorted(bytime.items()):
            camera=camera_by_actual[at][1];R=cv2.Rodrigues(camera[3:6])[0]
            if reference_R is None:reference_R=R
            axis=R.T@np.array([0.,0.,1.])
            rotation=np.degrees(np.arccos(np.clip((np.trace(R@reference_R.T)-1.)/2.,-1.,1.)))
            # Prefer the existing full physical native producer, then the
            # largest original material subset. Duplicate secondary centre
            # rows never create extra exposures or independent witnesses.
            rank=lambda r:(r['family']in('additional_real_native_ice_curves','additional_real_native_board_circles'),
                r['fixed_physical_material_projection_response']['original_physical_station_or_named_role_count'],
                r['collection_role']=='FIT',r['family']!='direct_manual_sparse_correlated_centre_secondary')
            selected=max(rows,key=rank);p=selected['fixed_physical_material_projection_response']
            fits=[r for r in rows if r['cost_in_actual_target']]
            times.append(dict(actual_pts=at,source_seconds=at/15360.,
                yaw_of_saved_viewing_ray_deg=float(np.degrees(np.arctan2(axis[1],axis[0]))),
                elevation_of_saved_viewing_ray_deg=float(np.degrees(np.arctan2(axis[2],np.linalg.norm(axis[:2])))),
                relative_orientation_angle_from_first_available_exposure_deg=float(rotation),
                effective_geometric_focal_px=float(np.exp(.5*(camera[6]+camera[7]))),
                physical_projection_reference=dict(family=selected['family'],id=selected['id'],original_packet_row_index=selected.get('original_packet_row_index')),
                producer_station_count=p['original_physical_station_or_named_role_count'],
                fixed_material_camera_mean_xy_px=p['camera_mean_displacement_xy_px'],
                fixed_material_geometry_mean_xy_px=p['geometry_mean_displacement_xy_px'],
                fixed_material_total_mean_xy_px=p['total_mean_displacement_xy_px'],
                fixed_material_camera_median_magnitude_px=p['camera_displacement_magnitude_px']['median'],
                fixed_material_geometry_median_magnitude_px=p['geometry_displacement_magnitude_px']['median'],
                fixed_material_total_median_magnitude_px=p['total_displacement_magnitude_px']['median'],
                fixed_material_cancellation_ratio_median=p['cancellation_ratio']['median'],
                producer_station_opposed_fraction=p['opposite_camera_geometry_stations']/p['original_physical_station_or_named_role_count'],
                producer_station_opposed_and_80percent_cancelled_fraction=p['opposite_and_at_least_80percent_cancelled_stations']/p['original_physical_station_or_named_role_count'],
                actual_FIT_original_operator_rows=[dict(family=r['family'],id=r['id'],original_packet_row_index=r.get('original_packet_row_index'),
                    points=r['points'],raw_native_p95_px=r['endpoint_same_original_operator']['11']['raw_native_px']['p95'],
                    outside_support_p95_px=r['endpoint_same_original_operator']['11']['outside_support_native_px']['p95'],
                    actual_cost_before=r['endpoint_same_original_operator']['00']['actual_weighted_source_cost'],
                    actual_cost_after=r['endpoint_same_original_operator']['11']['actual_weighted_source_cost'],
                    actual_geometry_cost_component=r['exact_cost_geometry_component'],actual_camera_cost_component=r['exact_cost_camera_component'])for r in fits],
                diagnostic_HELD_operator_rows=len(rows)-len(fits),
                duplicate_centre_or_material_queries_not_counted_as_new_exposures=True))
        cm=np.asarray([r['fixed_material_camera_mean_xy_px']for r in times]);gm=np.asarray([r['fixed_material_geometry_mean_xy_px']for r in times]);tm=np.asarray([r['fixed_material_total_mean_xy_px']for r in times])
        sign=lambda a:np.sign(a).astype(int).tolist()
        rho=[r['fixed_material_cancellation_ratio_median']for r in times]
        focal=[r['effective_geometric_focal_px']for r in times];angles=[r['relative_orientation_angle_from_first_available_exposure_deg']for r in times]
        group=dict(physical_group=physical,installation_id=installation,existing_actual_exposures=len(times),
            actual_FIT_exposures=sum(bool(r['actual_FIT_original_operator_rows'])for r in times),
            camera_mean_xy_sign_sequence=sign(cm),geometry_mean_xy_sign_sequence=sign(gm),total_mean_xy_sign_sequence=sign(tm),
            effective_focal_min_max_px=[min(focal),max(focal)],orientation_relative_angle_min_max_deg=[min(angles),max(angles)],
            exposures_whose_full_physical_producer_is_at_least_80percent_opposed=sum(r['producer_station_opposed_fraction']>=.8 for r in times),
            exposures_whose_full_physical_producer_is_at_least_80percent_strongly_cancelled=sum(r['producer_station_opposed_and_80percent_cancelled_fraction']>=.8 for r in times),
            cancellation_ratio_median_min_max=[min(rho),max(rho)],
            descriptive_Pearson_correlation=dict(cancellation_ratio_vs_effective_focal=corr(rho,focal),
                cancellation_ratio_vs_relative_orientation_angle=corr(rho,angles),
                camera_mean_x_vs_effective_focal=corr(cm[:,0],focal),geometry_mean_x_vs_effective_focal=corr(gm[:,0],focal)),
            per_existing_exposure=times,
            caveats=['Producer stations can include unobserved portions; their displacements are not additional source observations.',
                'Exposure coverage and focal/orientation are observationally confounded; correlations are not causal tests.',
                'All compensation values concern this one saved step; they are not the cause of the existing absolute 20–45px residuals.'])
        out['groups'].append(group)
    # The current six uniform LOW readings have the same declared exterior
    # Ad4 edge. Keep this source subset distinct from older top/bottom/right
    # Ad4 observations rather than averaging unlike material roles.
    config=json.loads(Path('events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json').read_text(encoding='utf8'))
    board_path=Path(config['source_options']['--additional-board-source'])
    board_by_id={r['id']:r for r in json.loads(board_path.read_text(encoding='utf8'))['rows']}
    low=[]
    for r in report['rows']:
        if (r['family']!='additional_real_native_board_edges'or r['physical_group']!='Ad4'
            or not r['id'].startswith('assistant_actual_RGB_uniform_')):continue
        original=board_by_id[r['id']];installation,camera=camera_by_actual[r['actual_pts']]
        assert installation=='okada_sp_installation_2_conditional_z190'
        ep=r['endpoint_same_original_operator'];p=r['fixed_physical_material_projection_response']
        raw=np.asarray(ep['11']['raw_pixel_vectors']);oldraw=np.asarray(ep['00']['raw_pixel_vectors'])
        axis=cv2.Rodrigues(camera[3:6])[0].T@np.array([0.,0.,1.])
        low.append(dict(id=r['id'],actual_pts=r['actual_pts'],source_curve=original['curve'],active_points=r['points'],
            effective_geometric_focal_px=float(np.exp(.5*(camera[6]+camera[7]))),
            yaw_of_saved_viewing_ray_deg=float(np.degrees(np.arctan2(axis[1],axis[0]))),
            elevation_of_saved_viewing_ray_deg=float(np.degrees(np.arctan2(axis[2],np.linalg.norm(axis[:2])))),
            raw_error_x_per_active_source_query_px=raw[:,0].tolist(),raw_error_y_per_active_source_query_px=raw[:,1].tolist(),
            raw_error_x_mean_px=float(raw[:,0].mean()),raw_error_x_median_px=float(np.median(raw[:,0])),
            before_raw_error_x_mean_px=float(oldraw[:,0].mean()),
            outside_support_p95_native_px=ep['11']['outside_support_native_px']['p95'],
            before_cost=ep['00']['actual_weighted_source_cost'],after_cost=ep['11']['actual_weighted_source_cost'],
            cost_camera_component=r['exact_cost_camera_component'],cost_geometry_component=r['exact_cost_geometry_component'],
            saved_step_camera_mean_xy_px=p['camera_mean_displacement_xy_px'],
            saved_step_geometry_mean_xy_px=p['geometry_mean_displacement_xy_px'],
            saved_step_total_mean_xy_px=p['total_mean_displacement_xy_px']))
    low.sort(key=lambda r:r['actual_pts']);assert len(low)==6
    assert all(r['source_curve']==low[0]['source_curve']for r in low)
    out['current_LOW_Ad4_uniform_six_exposure_same_material_source']=dict(native_packet=ref(board_path),
        actual_exposures=6,same_immutable_declared_curve=low[0]['source_curve'],
        all_existing_current_active_points_used=True,model_minus_observed_X_sign=[int(np.sign(r['raw_error_x_mean_px']))for r in low],
        actual_source_rows=low,
        descriptive_Pearson_correlation=dict(mean_absolute_X_error_vs_effective_focal=corr([abs(r['raw_error_x_mean_px'])for r in low],[r['effective_geometric_focal_px']for r in low]),
            signed_mean_X_error_vs_effective_focal=corr([r['raw_error_x_mean_px']for r in low],[r['effective_geometric_focal_px']for r in low])),
        same_sign_or_monotone_focal_only_error_law_not_established=True,
        isolated_45px_exposure_cannot_be_extrapolated_to_whole_LOW_cut=True)
    target=H/'current_absolute_saved_step_compensation_by_installation_q276_q8a15_20261011.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf8'))
        if previous.get('schema')!=out['schema']or previous.get('child_q')!=child:raise FileExistsError(target)
        out['replaces_prior_same_q_summary_sha256']=sha(target)
    target.write_text(json.dumps(out,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=ref(target),groups=len(out['groups']))))
    for group in out['groups']:
        if group['physical_group'].startswith('F2')or group['physical_group']=='Ad24':
            print(json.dumps({k:v for k,v in group.items()if k not in ('per_existing_exposure','caveats')}))

if __name__=='__main__':main()
