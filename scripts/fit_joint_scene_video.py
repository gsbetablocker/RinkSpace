"""Joint semantic rink/camera fit over measured, traceable video observations."""
from rinkspace.storage import resolve_asset as _resolve_asset
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rinkspace.calibration_companion import Scene
from rinkspace.calibration_companion.joint_scene import (
    JointSceneProblem, landmark_boundary_axes, landmark_corner_dependencies,
)
from rinkspace.calibration_companion.camera_identity import resolve_frame_camera, installation_records
from prepare_joint_scene_evidence import OUT, OLD, load, save, digest
from fit_scene3d_real_keyframes import initialize_frame

BASE=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json')
EIGHT_CIRCLES = [f'{side}{row}' for side in ('F', 'R') for row in range(1, 5)]


def scene_parameterization(scene, observations, board_landmarks=()):
    """User's hard layout relations; pixel observations are not altered here."""
    fit = [o for o in observations if o['use'] == 'fit']
    points = {p['id']: p for p in board_landmarks}
    specs = []

    def add(name, kind, bounds, *, feature=None, features=None, corner=None, **extra):
        evidence = []
        for o in fit:
            t = o['target']
            affects = feature is not None and t.get('feature_id') == feature
            if features:
                affects = t.get('feature_id') in features
                if kind == 'circle_shared_radius':
                    affects = (affects and t['kind'] == 'circle_outline'
                               and t.get('ring_role', 'outer') == 'outer')
                if kind == 'symmetric_line_distance' and features == ['goal_line_left', 'goal_line_right']:
                    affects = affects or t.get('feature_id') in ('goal_crease_left', 'goal_crease_right')
            if kind == 'circle_radius':
                affects = affects and t['kind'] == 'circle_outline' and t.get('ring_role', 'outer') == 'outer'
            if kind in ('boundary_length', 'boundary_width'):
                affects = t['kind'] in ('board_side', 'board_corner')
                if t['kind'] == 'board_landmark':
                    point = points[t['landmark_id']]
                    affects = (0 if kind == 'boundary_length' else 1) in landmark_boundary_axes(point)
            elif kind == 'board_height':
                affects = t.get('edge') == 'top' or t['kind'] == 'board_landmark'
            elif corner:
                # Directly observed arcs and details continuing onto those arcs
                # constrain their radius. Merely naming an adjacent side does not
                # establish observation of its otherwise invisible endpoint.
                affects = (t.get('corner') == corner or t.get('side') == corner
                           or t.get('continue_corner') == corner)
                if t['kind'] == 'board_landmark':
                    affects = corner in landmark_corner_dependencies(points[t['landmark_id']])
            if affects:
                evidence.append(o['id'])
        spec = dict(id=name, kind=kind, bounds=bounds, evidence_observation_ids=evidence, **extra)
        if feature is not None:
            spec['feature_id'] = feature
        if features:
            spec['feature_ids'] = list(features)
        if corner:
            spec['corner'] = corner
        specs.append(spec)

    # Wide numerical search intervals, not measured physical rink dimensions.
    # Both dimensions vary; neither is reused as the coordinate scale anchor.
    add('rink_length', 'boundary_length', [1., 2000.])
    add('rink_width', 'boundary_width', [1., 2000.])
    # Six independent positions describe the eight centers: two columns,
    # one first-row ordinate, and three positive row spacings. Individual
    # circle X/Y coordinates are deliberately not solver variables.
    source_bounds = getattr(scene, 'data', {}).get('parameters', {}).get('bounds', [-100., -50., 100., 50.])
    middle_x = (source_bounds[0] + source_bounds[2]) / 2.
    for side in ('F', 'R'):
        members = [f'{side}{i}' for i in range(1, 5)]
        xs = [scene.features[f]['center_xyz'][0] for f in members if f in scene.features]
        initial = float(np.mean(xs)) if xs else (-56. if side == 'F' else 46.)
        interval = [max(-96., initial-30.), min(96., initial+30.)]
        if side == 'F':
            interval[1] = min(interval[1], middle_x-1e-6)
        else:
            interval[0] = max(interval[0], middle_x+1e-6)
        add(side+'_column_x', 'circle_shared_center_x', interval, features=members)
    row1 = [scene.features[f]['center_xyz'][1] for f in ('F1', 'R1') if f in scene.features]
    initial_y = float(np.mean(row1)) if row1 else -36.
    add('FR_row1_y', 'circle_row_origin_y', [max(-46., initial_y-22.), min(46., initial_y+22.)],
        features=EIGHT_CIRCLES)
    for after_row in range(1, 4):
        members = [f'{side}{row}' for side in ('F', 'R') for row in range(after_row+1, 5)]
        add(f'FR_row_gap_{after_row}_{after_row+1}', 'circle_row_gap', [1e-6, 92.],
            features=members, after_row=after_row)
    for fid, feature in scene.features.items():
        if feature['kind'] != 'circle':
            continue
        cx, cy, _ = feature['center_xyz']
        if fid not in (*EIGHT_CIRCLES, 'C0'):
            add(fid+'_x', 'circle_center_x', [max(-96., cx-30.), min(96., cx+30.)], feature=fid)
            add(fid+'_y', 'circle_center_y', [max(-46., cy-22.), min(46., cy+22.)], feature=fid)
        if fid not in EIGHT_CIRCLES:
            add(fid+'_radius', 'circle_radius', [2., 30.], feature=fid)
    add('F_R_shared_outer_radius', 'circle_shared_radius', [2., 15.], features=EIGHT_CIRCLES)
    for name, members, fallback in [('B_distance', ['B1', 'B2'], 24.),
                                     ('goal_distance', ['goal_line_left', 'goal_line_right'], 88.)]:
        distances = [abs(float(np.asarray(scene.features[f]['segments_xyz'])[:, :, 0].mean())-middle_x)
                     for f in members if f in scene.features]
        initial = float(np.mean(distances)) if distances else fallback
        add(name, 'symmetric_line_distance', [max(1e-6, initial-20.), initial+20.], features=members)
    add('board_height', 'board_height', [1., 12.])
    for corner in ['upper_left', 'upper_right', 'lower_left', 'lower_right']:
        add('corner_'+corner, 'boundary_corner_radius', [3., 45.], corner=corner)
    order = [dict(feature_ids=[f'{side}{i}' for i in range(1, 5)], axis=1, min_gap=0.)
             for side in ('F', 'R')]
    order.extend(dict(feature_ids=[f'F{i}', f'R{j}'], axis=0, min_gap=0.)
                 for i in range(1, 5) for j in range(1, 5))
    return specs, {'circle_order': order}


def apply_scene_parameterization(scene, problem, fixed=False):
    specs, constraints = scene_parameterization(scene, problem['observations'], problem.get('board_landmarks', []))
    problem['scene_parameters'] = [] if fixed else specs
    problem['geometric_constraints'] = {} if fixed else constraints
    problem['hard_geometry_constraints'] = {'profile': 'user_20260923_alignment_symmetry'}
    problem['geometry_revision'] = 'user_20260923_hard_grid_center_and_mirror_relations'
    problem['gauge'] = ('Origin and axes retained; length and width vary independently. '
                        'Values remain conditional model units, not measured metres. '
                        'No new absolute scale measurement is introduced.')
    problem['unobserved_feature_policy'] = ('Unobserved independent parameters retain their projected initial values; '
        'hard shared coordinates and radii also determine unseen members. '
        'F/R centers form two columns and four aligned rows with independently variable positive spacings; '
        'C/C0 are fixed at the rink center, B lines and goal line/crease pairs are exact mirrors.')
    return problem


def normalize_observations(data):
    data=copy.deepcopy(data)
    data['frames']=[resolve_frame_camera(f, data.get('installations', [])) for f in data['frames']]
    for f in data['frames']:
        f.setdefault('pixel_frame_id',f"{f['media_track_id']}:pts:{f['source_pts']}:1920x1080")
        f.setdefault('principal_point_px',[960.,540.]);f.setdefault('image_size_wh',[1920,1080])
        f.setdefault('focal_bounds_px',[450.,30000.]);f.setdefault('distortion_opencv',[0.,0.,0.,0.,0.])
        f.setdefault('fy_over_fx',1.)
    byid={f['id']:f for f in data['frames']}
    for o in data['observations']:
        if o.get('use') not in ('fit', 'check'):
            raise ValueError(f"{o['id']}: explicit fit/check use is required")
        target=o['target']
        if target['kind']=='board_curve':
            name=o['physical_group_id']
            sides=[side for side in ('upper_long_side','lower_long_side','left_end','right_end',
                                     'upper_left','upper_right','lower_left','lower_right') if side in name]
            if len(sides)!=1:
                raise ValueError(f"{o['id']}: old board curve needs an identified side/corner")
            side=sides[0]
            edge_claims=[]
            if target.get('edge') is not None:
                if target['edge'] not in ('top','contact'):
                    raise ValueError(f"{o['id']}: unsupported board edge")
                edge_claims.append(target['edge'])
            if name.endswith('_top'):edge_claims.append('top')
            if name.endswith(('_bottom','_contact')):edge_claims.append('contact')
            if target.get('height') is not None:
                height=float(target['height'])
                if not np.isfinite(height) or height<0:
                    raise ValueError(f"{o['id']}: invalid board height")
                edge_claims.append('contact' if height==0 else 'top')
            if not edge_claims or len(set(edge_claims))!=1 or target.get('side',side)!=side:
                raise ValueError(f"{o['id']}: conflicting board side/edge identity")
            edge=edge_claims[0]
            o['target']={'kind':'board_side','side':side,'edge':edge}
        o.setdefault('correlation_group_id',f"{byid[o['frame_id']].get('shot_id','unknown')}_{o['physical_group_id']}")
        # A curve's dense pixels share blur and extraction error. Spatially
        # subsample for speed; track-level weights are normalized in the solver.
        if len(o['pixels_uv'])>64:
            idx=np.linspace(0,len(o['pixels_uv'])-1,64).round().astype(int)
            o['pixels_uv']=np.asarray(o['pixels_uv'])[idx].tolist()
    return data


def make_problem(paths, fixed=False, start=0, scene_path=BASE):
    scene=Scene(scene_path);frames={};observations={};landmarks={};declared_installations={}
    for path in paths:
        data=normalize_observations(load(path))
        for f in data['frames']:
            if f['id'] in frames:
                old=frames[f['id']]
                for key in ('source_id','source_sha256','source_pts','time_base','image_size_wh',
                            'physical_camera_id','installation_id'):
                    if old.get(key)!=f.get(key):
                        raise ValueError(f"{f['id']}: duplicate frame has conflicting {key}")
            frames[f['id']]=f
        for o in data['observations']:
            if o['id'] in observations:
                old=observations[o['id']]
                for key in ('frame_id','target','use','pixels_uv'):
                    if old.get(key)!=o.get(key):
                        raise ValueError(f"{o['id']}: duplicate observation has conflicting {key}")
            observations[o['id']]=o
        for point in data.get('board_landmarks',[]):
            if point['id'] in landmarks:
                old=landmarks[point['id']]
                for key in ('advertisement_instance_id','side','allow_corner','along_parameterization'):
                    if old.get(key)!=point.get(key):
                        raise ValueError(f"{point['id']}: duplicate board detail has conflicting {key}")
            landmarks[point['id']]=point
        for installation in data.get('installations',[]):
            iid=installation['id']
            if iid in declared_installations and declared_installations[iid]!=installation:
                raise ValueError(f'{iid}: conflicting installation declarations')
            declared_installations[iid]=installation
    # A source exposure has only one pose, even when a reused PNG and a tracking
    # JPEG both refer to it. Keep the original direct measurement at seed PTS.
    canonical={};aliases={};unique_frames={}
    for fid,f in frames.items():
        key=(f['media_track_id'],f['source_pts'],tuple(f['image_size_wh']))
        cid=canonical.setdefault(key,fid);aliases[fid]=cid
        if cid==fid:unique_frames[fid]=f
        elif any(unique_frames[cid][key]!=f[key] for key in ('installation_id','physical_camera_id')):
            raise ValueError(f'{cid}/{fid}: same exposure assigned to different cameras/installations')
    frames=unique_frames;unique_obs={};seen=set()
    for oid,o in observations.items():
        o['frame_id']=aliases[o['frame_id']]
        key=(o['frame_id'],json.dumps(o['target'],sort_keys=True),o['use'],o.get('physical_group_id') if o['target'].get('feature_id')=='A1' else '')
        if key not in seen:unique_obs[oid]=o;seen.add(key)
    observations=unique_obs
    obs=list(observations.values())
    for point in landmarks.values():
        point['evidence_observation_ids']=[o['id'] for o in obs if o['target'].get('landmark_id')==point['id'] and o['use']=='fit']
    main_old=load(OLD/'conditional_cameras.json')
    installations=installation_records(frames.values(), declared_installations.values(),
        legacy_main_center=main_old['installations'][0]['center_xyz'],start=start)
    centers={i['id']:i['initial_center_xyz'] for i in installations}
    init_path=OUT/'tracking/camera_initializers.json'
    initializers=load(init_path) if init_path.exists() else {}
    initializers=initializers.get('by_frame_id',{})
    for f in frames.values():
        oo=[o for o in obs if o['frame_id']==f['id'] and o['use']=='fit']
        ring=[o for o in oo if o['target']['kind']=='circle_outline' and len(o['pixels_uv'])>=12]
        if f['physical_camera_id']=='camera_high_oblique':
            if 'initial_rvec_world_to_camera' not in f:
                cam=load(_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v014_high_preview/projection.json'))['camera']
                f.update(initial_rvec_world_to_camera=cam['rvec_world_to_camera'],initial_focal_px=cam['K_px'][0][0])
            if start:
                # Keep the first initializer's orientation; joint solve can
                # change it. Enlarge focal approximately with range.
                f['initial_focal_px']*=1.3
        elif f['id'] in initializers and f['physical_camera_id']=='camera_main_long_side':
            initializer=initializers[f['id']]
            f['initial_rvec_world_to_camera']=initializer['initial_rvec_world_to_camera']
            f['initial_focal_px']=initializer['initial_focal_px']
            f['initialization_evidence']=str(init_path)
        elif 'initial_rvec_world_to_camera' in f and 'initial_focal_px' in f:
            # A new advertisement-only frame may have an image-motion pose
            # initializer supplied by its measured-evidence batch. This is not
            # a fixed pose or an additional geometric observation.
            pass
        elif ring:
            # Use currently visible measured pixels, never a stale seed ellipse.
            r=max(ring,key=lambda o:len(o['pixels_uv']))
            r=copy.deepcopy(r)
            r['ellipse_cv2_image_only']=cv2.fitEllipse(np.asarray(r['pixels_uv'],np.float32))
            rv,fl=initialize_frame(scene,f,[r],np.asarray(centers[f['installation_id']]),1.)
            f.update(initial_rvec_world_to_camera=rv,initial_focal_px=fl)
        else:
            raise ValueError(f"{f['id']}: {f['physical_camera_id']} needs its own pose initializer; no observations were silently dropped")
    obs=list(observations.values())
    used={o['frame_id'] for o in obs};frames={k:v for k,v in frames.items() if k in used}
    installations=[i for i in installations if any(f['installation_id']==i['id'] for f in frames.values())]
    problem={'schema_version':'rinkspace.real-joint-video-problem.v1','geometry_mode':'shared_scene_joint',
                 'scene_sha256':scene.sha256,'synthetic':False,'curve_samples':192,'split_mode':'frame_object',
                 'board_landmarks':list(landmarks.values()),'installations':installations,'frames':list(frames.values()),'observations':obs,
                 'source_observation_files':[str(p) for p in paths],
                 'heldout_shots':['S085','S087','S089']}
    return scene, apply_scene_parameterization(scene, problem, fixed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',nargs='+');ap.add_argument('--tag',default='initial');ap.add_argument('--fixed',action='store_true');ap.add_argument('--start',type=int,default=0);ap.add_argument('--max-nfev',type=int,default=250);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--from-problem', type=Path, help='Reuse an existing problem without altering its observations/cameras')
    ap.add_argument('--prepare-only', action='store_true', help='Write and check the revised parameterization without solving')
    ap.add_argument('--scene', type=Path, help='Base model; defaults to the prepared problem base_scene_path when present')
    args=ap.parse_args()
    paths=[Path(p).resolve() for p in args.data] if args.data else [OLD/'real_observations.json',OUT/'high_seed_observations.json',OUT/'additional_seeds.json']
    dest=OUT/'fits'/args.tag;dest.mkdir(parents=True,exist_ok=True)
    if args.from_problem:
        if args.data or args.resume or args.start:
            ap.error('--from-problem cannot be combined with --data, --resume or --start')
        problem=load(args.from_problem)
        scene=Scene(args.scene or problem.get('base_scene_path') or BASE)
        problem['frames']=[resolve_frame_camera(f, problem.get('installations', [])) for f in problem['frames']]
        problem['installations']=installation_records(problem['frames'], problem.get('installations', []))
        problem=apply_scene_parameterization(scene, problem, args.fixed)
        problem['parameterization_source_problem']=str(args.from_problem.resolve())
        problem['parameterization_source_problem_sha256']=digest(args.from_problem)
    else:
        scene,problem=make_problem(paths,args.fixed,args.start,args.scene or BASE)
    save(dest/'problem.json',problem)
    p=JointSceneProblem(scene,problem)
    print(json.dumps({'frames':len(problem['frames']),'observations':len(problem['observations']),'variables':len(p.x0),'scene_variables':p.scene_count,'tag':args.tag}),flush=True)
    if args.prepare_only:
        residual=p.residual(p.x0)
        geometry=p.geometry(p.x0)
        circle_checks=geometry.circle_validity_checks()
        preserved_fields={}
        if args.from_problem:
            source_problem=load(args.from_problem)
            preserved_fields={key:source_problem[key]==problem[key] for key in
                ('frames','observations','board_landmarks','installations','source_observation_files')}
        save(dest/'preparation_check.json', {
            'fit_run':False, 'model_changed':False,
            'problem_sha256':digest(dest/'problem.json'),
            'active_scene_parameters':[s['id'] for s in p.scene_specs],
            'unobserved_parameters_retained':[s['id'] for s in p.retained_parameters],
            'all_residuals_finite':bool(np.isfinite(residual).all()),
            'jacobian_shape':list(p.jac_sparsity.shape),
            'initial_geometry_check':geometry.parameters,
            'circle_geometry_valid':all(row['valid'] for row in circle_checks),
            'original_observations_and_cameras_preserved':preserved_fields,
            'script_sha256':digest(__file__),
            'core_sha256':digest(ROOT/'src/rinkspace/calibration_companion/joint_scene.py'),
            'note':'Parameter wiring check only; no optimizer, no new video fit or accuracy claim.'})
        return
    t=time.monotonic();checkpoint=dest/'checkpoint.json';resume=None
    if args.resume:
        previous=load(checkpoint)
        if previous['parameter_names']!=p.parameter_names or previous['problem_sha256']!=digest(dest/'problem.json'):
            raise ValueError('checkpoint belongs to a different problem or parameter layout')
        resume=np.asarray(previous['parameter_values'])
    iteration=[0]
    def on_iteration(intermediate_result):
        iteration[0]+=1
        if iteration[0]%5==0:
            cost=float(intermediate_result.cost)
            save(checkpoint,{'parameter_names':p.parameter_names,'parameter_values':intermediate_result.x.tolist(),
                             'cost':cost,'iteration':iteration[0],'problem_sha256':digest(dest/'problem.json')})
            print(json.dumps({'iteration':iteration[0],'cost':cost,'elapsed':round(time.monotonic()-t,1)}),flush=True)
    result=p.solve(args.max_nfev,callback=on_iteration,x0=resume)
    result['elapsed_seconds']=time.monotonic()-t;result['script_sha256']=digest(__file__)
    save(dest/'fit.json',result);save(dest/'scene_semantic.json',p.export_scene_data(p.result.x,'v0.15-joint-video-candidate'))
    print(json.dumps({'optimizer':result['optimizer'],'after':result['after']['overall'],'installations':result['installations'],'elapsed':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
