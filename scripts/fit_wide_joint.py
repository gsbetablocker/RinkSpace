"""Wide-image scene and physical camera joint fit, with exact structural relations."""
from rinkspace.storage import resolve_asset as _resolve_asset
from pathlib import Path
import argparse
import copy
import inspect
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rinkspace.calibration_companion import Scene
from rinkspace.calibration_companion.joint_scene import JointSceneProblem
from rinkspace.calibration_companion.scene_camera import nearest_polyline_vectors
from fit_joint_scene_video import apply_scene_parameterization
from prepare_joint_scene_evidence import load, save, digest

OUT = _resolve_asset(ROOT, ROOT / 'outputs/calibration/ankara_2026_wide_joint_20260923')
SOURCE = _resolve_asset(ROOT, ROOT / 'outputs/calibration/ankara_2026_v015_joint/hard_structure_20260923/problem.json')
_ACTIVE_PROBLEM = None


def _build_joint_problem(problem):
    """Rebuild the exact solver type from a saved, self-contained problem."""
    scene = Scene(problem['base_scene_path'])
    if problem.get('wide_angle_optics') or problem.get('wide_angle_camera'):
        from rinkspace.calibration_companion.wide_angle_camera import WideAngleJointProblem
        return WideAngleJointProblem(scene, problem)
    return JointSceneProblem(scene, problem)


def _initialize_worker(problem_path):
    global _ACTIVE_PROBLEM
    _ACTIVE_PROBLEM = _build_joint_problem(load(Path(problem_path)))


def _worker_objective(values):
    """Picklable numerical-Jacobian target; each spawn worker owns its solver."""
    problem = _ACTIVE_PROBLEM
    if problem is None:
        raise RuntimeError('wide joint objective used before worker initialization')
    residual = problem.residual(values)
    outside = problem.problem.get('outside_right_margin_experiment')
    if outside is None:
        return residual
    geometry = problem.geometry(values)
    frame_id = next(fid for fid in problem.frames if 'women_sp_high_' in fid)
    actual = problem.camera(values, frame_id).center_xyz[0] - geometry.parameters['bounds'][2]
    required = float(outside['minimum_margin_scene_units'])
    weight = float(outside['hinge_weight'])
    return np.r_[residual, weight * min(actual - required, 0.)]


def _default_worker_count():
    # Each spawned process keeps a full solver and image observations in memory.
    return min(8, max(1, (os.cpu_count() or 1) // 2))


def _resolve_worker_count(requested, parallel_supported):
    workers = _default_worker_count() if requested is None else requested
    if workers < 1:
        raise ValueError('--workers must be at least 1')
    if workers > 1 and not parallel_supported:
        if requested is not None:
            raise ValueError(f'--workers > 1 requires SciPy with least_squares(workers=...), '
                             f'available since SciPy 1.16; installed version is {scipy.__version__}')
        return 1, 'auto_serial_fallback_older_scipy'
    return workers, ('auto' if requested is None else 'explicit')


def prepare(extra=None, multiple=False, camera_min_x=None, support_main_right=False):
    source = load(SOURCE)
    scene = Scene(source['base_scene_path'])
    p = copy.deepcopy(source)
    main_support_ids = {'main279_80302080', 'main279_86138880'} if support_main_right else set()
    p['frames'] = [f for f in p['frames'] if
                   ('women_sp_high_' in f['id'] and (multiple or f['source_pts'] == 56218880))
                   or f['id'] in main_support_ids]
    ids = {f['id'] for f in p['frames']}
    p['observations'] = [o for o in p['observations'] if o['frame_id'] in ids
                         and o['target']['kind'] != 'board_landmark']
    if extra:
        additions = load(extra)
        replaced = set(additions.get('replaces_observation_ids', []))
        p['observations'] = [o for o in p['observations'] if o['id'] not in replaced]
        p['observations'].extend(o for o in additions['observations'] if o['frame_id'] in ids)
        p['additional_observation_file'] = str(extra.resolve())
        p['additional_observation_sha256'] = digest(extra)
    # The lower-right rail samples remain the original measured pixels. With a
    # common corner radius, their unknown tangent point may move through this
    # visible run; fit the connected straight side and adjacent quarter arc.
    continuous_boundary_targets = {
        'women_sp_high_56218880_near_lower_right_rail_ice_facing_top':
            {'kind': 'board_side', 'side': 'right_end', 'edge': 'top', 'continue_corner': 'lower_right'},
        'women_sp_high_56218880_boundary_top_right_curve':
            {'kind': 'board_side', 'side': 'upper_long_side', 'edge': 'contact', 'continue_corner': 'upper_right'},
        'women_sp_high_56218880_boundary_left_visible':
            {'kind': 'board_side', 'side': 'left_end', 'edge': 'contact', 'continue_corner': 'upper_left'},
        'women_sp_high_56218880_left_end_rail_top':
            {'kind': 'board_side', 'side': 'left_end', 'edge': 'top', 'continue_corner': 'upper_left'},
    }
    for observation in p['observations']:
        if observation['id'] in continuous_boundary_targets:
            previous = observation['target']
            revised = continuous_boundary_targets[observation['id']]
            if previous.get('edge') != revised['edge'] or previous['kind'] not in ('board_side', 'board_corner'):
                raise ValueError(f"{observation['id']}: boundary identity changed; review before fitting")
            observation['target'] = revised
            observation['target_revision'] = {
                'reason': 'Unknown straight-side to adjacent-corner tangent point under the common corner radius',
                'pixels_unchanged': True,
                'previous_target': previous}
    p['board_landmarks'] = []
    p['installations'] = [i for i in p['installations'] if i['id'] == 'women_sp_high'
                          or (support_main_right and i['id'] == 'women_sp_main')]
    high_installation = next(i for i in p['installations'] if i['id'] == 'women_sp_high')
    if support_main_right:
        main_installation = next(i for i in p['installations'] if i['id'] == 'women_sp_main')
        # The support camera is not the wide lens. Its existing conditional
        # pinhole initialization and frame poses are jointly fitted separately.
        main_installation['wide_angle_camera'] = {'enabled': False}
        p['cross_view_support'] = {
            'frame_ids': sorted(main_support_ids),
            'purpose': 'same-event right-end top/contact and R1 geometry support',
            'status': 'fit evidence, not independent validation'}
    # Search bounds are numerical domains, not a claimed measured installation.
    high_installation['center_bounds_xyz'] = [[1., -600., 1.], [600., -1., 400.]]
    if camera_min_x is not None:
        high_installation['center_bounds_xyz'][0][0] = float(camera_min_x)
        high_installation['initial_center_xyz'][0] = max(
            float(high_installation['initial_center_xyz'][0]), float(camera_min_x) + 1e-3)
        p['camera_side_experiment'] = {
            'minimum_world_x': float(camera_min_x),
            'status': 'candidate based on the right-end board occlusion complaint; not a surveyed camera position'}
    high_installation['radial_distortion']['bounds'] = [-3.,3.]
    p['curve_samples'] = 256
    p['source_problem_path'] = str(SOURCE)
    p['source_problem_sha256'] = digest(SOURCE)
    p['fit_run'] = True
    p = apply_scene_parameterization(scene, p)
    p['hard_geometry_constraints'].update(
        lane_centerline_through_FR_rows=True, lane_left_open=True,
        equal_corner_radii=True, lane_equal_width=True, lane_right_collinear=True,
        A1_A2_mirror_equal_radius=True,
        lane_rule_source='User follow-ups in wide joint tasks, 2026-09-23')
    # A1 and A2 are paired across the long-axis centerline y=0.  Keep their
    # observed image outlines independent, but solve one X, one |Y| and one
    # outer radius so the geometric relation is exact at every iterate.
    paired = [s for s in p['scene_parameters'] if s.get('feature_id') in ('A1', 'A2')
              and s['kind'] in ('circle_center_x', 'circle_center_y', 'circle_radius')]
    if {(s['feature_id'], s['kind']) for s in paired} != {
            (fid, kind) for fid in ('A1', 'A2')
            for kind in ('circle_center_x', 'circle_center_y', 'circle_radius')}:
        raise ValueError('A1/A2 paired hard constraint requires exactly six independent source parameters')
    p['scene_parameters'] = [s for s in p['scene_parameters'] if s not in paired]
    by_id = {(s['feature_id'], s['kind']): s for s in paired}
    def _intersection(bounds_a, bounds_b):
        result = [max(bounds_a[0], bounds_b[0]), min(bounds_a[1], bounds_b[1])]
        if result[0] >= result[1]:
            raise ValueError('A1/A2 paired parameter bounds do not overlap')
        return result
    paired_evidence = list(dict.fromkeys(oid for s in paired
                                          for oid in s['evidence_observation_ids']))
    x_bounds = _intersection(by_id['A1', 'circle_center_x']['bounds'],
                             by_id['A2', 'circle_center_x']['bounds'])
    a1_y_bounds = by_id['A1', 'circle_center_y']['bounds']
    a2_y_bounds = by_id['A2', 'circle_center_y']['bounds']
    y_distance_bounds = _intersection([-a1_y_bounds[1], -a1_y_bounds[0]],
                                      a2_y_bounds)
    radius_bounds = _intersection(by_id['A1', 'circle_radius']['bounds'],
                                  by_id['A2', 'circle_radius']['bounds'])
    for id_, kind, bounds in (
            ('A_pair_shared_center_x', 'A_pair_shared_center_x', x_bounds),
            ('A_pair_mirror_y_distance', 'A_pair_mirror_y_distance', y_distance_bounds),
            ('A_pair_shared_radius', 'A_pair_shared_radius', radius_bounds)):
        p['scene_parameters'].append(dict(id=id_, kind=kind, bounds=bounds,
                                          feature_ids=['A1', 'A2'],
                                          evidence_observation_ids=paired_evidence))
    p['A_pair_geometry_provenance'] = {
        'hard_relation': 'A1/A2 equal outer radii and centers reflected across long-axis y=0',
        'source': 'User correction, 2026-09-23',
        'direct_image_evidence_ids': paired_evidence}
    # A single physical corner radius replaces four independent fit variables.
    corner_specs = [s for s in p['scene_parameters'] if s['kind'] == 'boundary_corner_radius']
    p['scene_parameters'] = [s for s in p['scene_parameters'] if s['kind'] != 'boundary_corner_radius']
    p['scene_parameters'].append(dict(
        id='shared_corner_radius', kind='shared_corner_radius', bounds=[3., 45.],
        evidence_observation_ids=list(dict.fromkeys(
            oid for spec in corner_specs for oid in spec['evidence_observation_ids']))))
    for role in ('middle','inner','center'):
        evidence=[o['id'] for o in p['observations'] if o['use']=='fit' and
                  o['target'].get('ring_role')==role and o['target'].get('feature_id') in
                  [f'{col}{i}' for col in 'FR' for i in range(1,5)]]
        if evidence:
            p['scene_parameters'].append(dict(id='FR_'+role+'_radius',kind='ring_shared_radius',
                feature_ids=[f'{col}{i}' for col in 'FR' for i in range(1,5)],ring_role=role,
                bounds={'middle':[3.,5.9],'inner':[.7,2.9],'center':[.05,.69]}[role],
                evidence_observation_ids=evidence))
    for kind in ('shared_crease_depth', 'shared_crease_half_width'):
        evidence=[o['id'] for o in p['observations'] if o['use']=='fit' and
                  o['target'].get('feature_id') in ('goal_crease_left','goal_crease_right')]
        if evidence:
            p['scene_parameters'].append(dict(id=kind,kind=kind,bounds=[.1,20.],
                evidence_observation_ids=evidence))
    right_edge_evidence = []
    for fid in ('lane1','lane2','lane3','lane4'):
        evidence=[o['id'] for o in p['observations'] if o['use']=='fit' and o['target'].get('feature_id')==fid]
        if evidence:
            poly=np.asarray(scene.features[fid]['polygon_xyz'])
            for axis in ('x',):
                a=0 if axis=='x' else 1
                mid=float((poly[:,a].min()+poly[:,a].max())/2)
                for edge in ('min','max'):
                    kind=f'lane_bound_{axis}_{edge}'
                    bounds=([-140.,mid-0.01] if edge=='min' else [mid+.01,140.])
                    edge_index={('x','min'):3,('x','max'):1,('y','min'):2,('y','max'):0}[(axis,edge)]
                    direct=[o['id'] for o in p['observations'] if o['id'] in evidence and
                            (o['target']['kind']=='polygon_outline' or o['target'].get('index')==edge_index)]
                    if not direct:
                        continue
                    if edge == 'max':
                        right_edge_evidence.extend(direct)
                    else:
                        p['scene_parameters'].append(dict(id=f'{fid}_{kind}',kind=kind,feature_id=fid,
                            bounds=bounds,evidence_observation_ids=direct))
    # A global similarity scale cannot be observed from photographs. This fixes
    # units only, leaving every relative dimension jointly estimable.
    p['scene_parameters'] = [s for s in p['scene_parameters'] if s['id'] != 'F_R_shared_outer_radius']
    lane_ids = ['lane1','lane2','lane3','lane4']
    width_evidence = [o['id'] for o in p['observations'] if o['use'] == 'fit' and
                      o['target'].get('feature_id') in lane_ids and
                      o['target'].get('index') in (0,2)]
    p['scene_parameters'].append(dict(id='lane_shared_half_width',kind='shared_lane_half_width',
        feature_ids=lane_ids,bounds=[3.,12.],evidence_observation_ids=width_evidence))
    p['scene_parameters'].append(dict(id='lane_shared_right_x',kind='shared_lane_right_x',
        feature_ids=lane_ids,bounds=[1.,140.],
        evidence_observation_ids=list(dict.fromkeys(right_edge_evidence))))
    p['lane_geometry_provenance'] = {
        'equal_width_and_right_collinear': 'User hard constraints, 2026-09-23',
        'individual_left_open_positions': 'Retained or fitted separately; equal length was withdrawn',
        'lane4_direct_support': 'No clear right edge observation in the selected frame'}
    p['scale_gauge'] = {'parameter': 'F_R_shared_outer_radius',
        'value': scene.features['F1']['radius'], 'unit': 'conditional_scene_units',
        'reason': 'Fix global similarity gauge only; no measured metric scale supplied.'}
    p['heldout_shots'] = []
    p['evidence_scope'] = 'Wide image joint development fit; no independent full-shot acceptance claim.'
    return scene, p


def save_result(p, x, dest, optimization):
    geometry = p.geometry(x)
    summary = p.summarize(x)
    fit = dict(schema_version='rinkspace.wide-joint-fit.v1',
        scene_sha256=p.scene.sha256, parameter_names=p.parameter_names,
        parameter_values=x.tolist(), scene_count=p.scene_count,
        parameter_changes=p.parameter_changes(x), retained_parameters=p.retained_parameters,
        before=p.summarize(p.x0), after=summary, optimizer=optimization,
        frames=[{**f, 'camera':p.camera(x, fid).to_dict()} for fid,f in p.frames.items()],
        hard_geometry_checks=geometry.hard_geometry_checks(),
        coordinate_gauge=p.problem.get('scale_gauge'),
        scene_geometry_held_fixed=False, physical_accuracy_verified=False)
    fit['installations'] = []
    for iid,installation in p.installations.items():
        fid=next(fid for fid,f in p.frames.items() if f['installation_id']==iid)
        camera=p.camera(x,fid)
        radial=installation.get('radial_distortion',{})
        fit['installations'].append(dict(id=iid,physical_camera_id=installation.get('physical_camera_id'),
            center_xyz=camera.center_xyz.tolist(), position_status='conditional_joint_fit_not_surveyed',
            radial_k1_image=float(x[p.radial_indices[iid]]) if iid in p.radial_indices else None,
            radial_scale_image=radial.get('scale_image')))
    if hasattr(p, 'optics_report'):
        fit['optics_report'] = p.optics_report(x)
    if p.problem.get('outside_right_margin_experiment'):
        frame_id = next(fid for fid in p.frames if 'women_sp_high_' in fid)
        required = float(p.problem['outside_right_margin_experiment']['minimum_margin_scene_units'])
        actual = float(p.camera(x, frame_id).center_xyz[0] - geometry.parameters['bounds'][2])
        fit['outside_right_margin_experiment'] = {
            **p.problem['outside_right_margin_experiment'],
            'actual_margin_scene_units': actual,
            'condition_satisfied': bool(actual >= required)}
    save(dest/'fit.json', fit)
    save(dest/'geometry.json', dict(features=geometry.features, parameters=geometry.parameters,
        corner_radii=geometry.corner_radii, hard_geometry_checks=geometry.hard_geometry_checks()))
    residuals = []
    cameras={fid:p.camera(x,fid) for fid in p.frames}
    previous_samples=p.samples;p.samples=768
    for o,uv in p.observations:
        delta,depth,fold=p._error(x,o,uv,geometry,cameras)
        if o['type']=='point':delta=-delta
        err = np.linalg.norm(delta,axis=1)
        residuals.append(dict(observation_id=o['id'],frame_id=o['frame_id'],target=o['target'],
            observed_uv=uv.tolist(),predicted_nearest_uv=(uv-delta).tolist(),error_px=err.tolist(),
            use=o['use'],min_depth=float(depth.min()),median_px=float(np.median(err)),
            p95_px=float(np.percentile(err,95)),max_px=float(err.max())))
    p.samples=previous_samples
    save(dest/'residuals.json', residuals)
    try:
        semantic=p.export_scene_data(x,'v0.18-wide-joint')
        if 'hard_structure_update' in semantic:
            semantic['previous_hard_structure_preparation']=semantic.pop('hard_structure_update')
        semantic.setdefault('build_scope',{})['video_fitting_run']=True
        semantic['joint_refinement']['video_refitted_after_hard_structure']=True
        semantic['joint_refinement']['wide_angle_optics']=fit.get('optics_report')
        semantic['joint_refinement']['scale_gauge']=p.problem.get('scale_gauge')
        semantic['coordinate_system']['unit']='conditional_scene_units_with_fixed_FR_outer_radius'
        semantic['coordinate_system']['metres_per_unit']=None
        save(dest/'scene_semantic.json',semantic)
    except ValueError as exc:
        save(dest/'export_diagnostic.json', {'error':str(exc)})
    return fit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',default='baseline')
    ap.add_argument('--extra',type=Path)
    ap.add_argument('--multiple',action='store_true')
    ap.add_argument('--optics',choices=['baseline','radial','brown','full','prism','prism_aspect','rational'],default='baseline')
    ap.add_argument('--start',type=Path)
    ap.add_argument('--sensor-coverage',action='store_true')
    ap.add_argument('--balance-objects',action='store_true')
    ap.add_argument('--shared-lane-width',action='store_true',
                    help='Accepted for old commands; equal width is now always a user hard constraint')
    ap.add_argument('--camera-min-x', type=float,
                    help='Experimental lower bound on camera world X for right-end visibility comparison')
    ap.add_argument('--outside-right-margin', type=float,
                    help='Experimental image-visibility prior: high camera X must exceed fitted right end by this scene-unit margin')
    ap.add_argument('--outside-weight', type=float, default=100.,
                    help='Hinge weight for --outside-right-margin; applies only to the experimental candidate')
    ap.add_argument('--support-main-right', action='store_true',
                    help='Jointly fit two same-event main-camera frames with observed right-end contact')
    ap.add_argument('--near-right-weight', type=float, default=1.,
                    help='Experimental emphasis on the measured near-right board top; 1 retains source weighting')
    ap.add_argument('--max-nfev',type=int,default=250)
    ap.add_argument('--workers',type=int,
                    help='Spawned processes for the numerical Jacobian; 1 uses the serial solver '
                         '(default: at most 8, based on available CPUs)')
    args=ap.parse_args()
    scipy_solver_parameters = inspect.signature(least_squares).parameters
    try:
        args.workers, worker_selection = _resolve_worker_count(
            args.workers, 'workers' in scipy_solver_parameters)
    except ValueError as exc:
        ap.error(str(exc))
    scene,problem=prepare(args.extra,args.multiple,args.camera_min_x,args.support_main_right)
    if args.near_right_weight <= 0:
        ap.error('--near-right-weight must be positive')
    if args.near_right_weight != 1.:
        target_id = 'women_sp_high_56218880_near_lower_right_rail_ice_facing_top'
        target = next((o for o in problem['observations'] if o['id'] == target_id), None)
        if target is None or target['target'].get('edge') != 'top':
            ap.error('--near-right-weight requires the real near-right top observation group')
        target['weight'] = float(target.get('weight', 1.)) * args.near_right_weight
        problem['near_right_weight_experiment'] = {
            'observation_id': target_id, 'weight_multiplier': args.near_right_weight,
            'purpose': 'sensitivity to measured board-top pixels; no inferred contact observations'}
    if args.balance_objects:
        for o in problem['observations']:
            if o['target']['kind']=='circle_outline':
                o['correlation_group_id']=o['frame_id']+':physical_house:'+o['target']['feature_id']
        problem['circle_weighting']='Each physical circle/house shares a correlation group across its concentric rings and visible arcs; dense nearby houses do not each count as four independent placements.'
    if args.outside_right_margin is not None:
        if args.outside_right_margin < 0 or args.outside_weight <= 0:
            ap.error('outside margin must be nonnegative and weight positive')
        problem['outside_right_margin_experiment'] = {
            'minimum_margin_scene_units': args.outside_right_margin,
            'hinge_weight': args.outside_weight,
            'status': 'experimental camera side prior from the user right-lower occlusion observation; not surveyed'}
    dest=OUT/args.tag;dest.mkdir(parents=True,exist_ok=True)
    if args.optics != 'baseline':
        active={'radial':['k2_image','k3_image'],
                'brown':['k2_image','k3_image','p1_image','p2_image'],
                'full':['k2_image','k3_image','p1_image','p2_image','cx','cy'],
                'prism':['k2_image','k3_image','p1_image','p2_image','cx','cy','s1_image','s2_image','s3_image','s4_image'],
                'prism_aspect':['k2_image','k3_image','p1_image','p2_image','cx','cy','s1_image','s2_image','s3_image','s4_image','log_fy_over_fx'],
                'rational':['k2_image','k3_image','p1_image','p2_image','cx','cy','s1_image','s2_image','s3_image','s4_image','log_fy_over_fx','k4_image','k5_image','k6_image']}[args.optics]
        problem['wide_angle_optics']={'active':active,'bounds':{'k2_image':[-10.,10.], 'k3_image':[-10.,10.]}}
        if args.sensor_coverage:
            problem['raw_sensor_coverage']={'enabled':True}
    p=_build_joint_problem(problem)
    save(dest/'problem.json',problem)
    global _ACTIVE_PROBLEM
    _ACTIVE_PROBLEM=p
    x=p.x0.copy()
    if args.start:
        old=load(args.start)
        values=dict(zip(old['parameter_names'],old['parameter_values']))
        if 'scene:A_pair_shared_center_x' not in values and all(
                f'scene:{fid}_{suffix}' in values for fid in ('A1', 'A2')
                for suffix in ('x', 'y', 'radius')):
            values['scene:A_pair_shared_center_x'] = (
                values['scene:A1_x'] + values['scene:A2_x']) / 2.
            values['scene:A_pair_mirror_y_distance'] = (
                -values['scene:A1_y'] + values['scene:A2_y']) / 2.
            values['scene:A_pair_shared_radius'] = (
                values['scene:A1_radius'] + values['scene:A2_radius']) / 2.
        x=np.array([values.get(n,v) for n,v in zip(p.parameter_names,x)])
        x=np.clip(x,p.bounds[0]+1e-9,p.bounds[1]-1e-9)
    if args.sensor_coverage:
        x=p.initialize_sensor_coverage(x)
    started=time.monotonic(); count=[0]
    def callback(intermediate_result):
        count[0]+=1
        if count[0]%10==0:
            save(dest/'checkpoint.json',dict(parameter_names=p.parameter_names,parameter_values=intermediate_result.x.tolist()))
            print(json.dumps(dict(iteration=count[0],cost=float(intermediate_result.cost),seconds=round(time.monotonic()-started,1))),flush=True)
    print(json.dumps(dict(tag=args.tag,parameters=len(x),frames=len(p.frames),
        observations=len(p.observations),workers=args.workers,
        worker_selection=worker_selection,scipy_version=scipy.__version__)),flush=True)
    solver_options=dict(bounds=p.bounds,method='trf',x_scale='jac',
        loss='soft_l1',f_scale=1.,max_nfev=args.max_nfev,ftol=1e-6,xtol=1e-8,gtol=1e-8)
    if 'callback' in scipy_solver_parameters:
        solver_options['callback'] = callback
    if args.workers == 1:
        result=least_squares(_worker_objective,x,**solver_options)
    else:
        context=multiprocessing.get_context('spawn')
        with context.Pool(args.workers,initializer=_initialize_worker,
                          initargs=(str((dest/'problem.json').resolve()),)) as pool:
            result=least_squares(_worker_objective,x,workers=pool.map,**solver_options)
    fit=save_result(p,result.x,dest,dict(success=bool(result.success),message=result.message,
        nfev=result.nfev,cost=float(result.cost),seconds=time.monotonic()-started,
        workers=args.workers,worker_selection=worker_selection,
        checkpoint_callback_available=('callback' in scipy_solver_parameters),
        jacobian_worker_mode=('serial' if args.workers == 1 else 'spawn_pool_map'),
        active_bounds=[p.parameter_names[i] for i in np.flatnonzero(result.active_mask)]))
    print(json.dumps({'optimizer':fit['optimizer'],'after':fit['after']['overall'],'hard':fit['hard_geometry_checks']}),flush=True)


if __name__=='__main__':main()
