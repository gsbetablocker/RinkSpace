"""Initialize the new common family from the last complete solved scene.

Old XYZ is used solely to initialize parameters, never as a new observation.
The programme camera, UV, weights and clock are not loaded or changed here.
"""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np

BASE=_runtime_data_root()
sys.path.insert(0, str(_runtime_source_root))
from isolated_solver_bootstrap import install
install()
sys.path.insert(0, 'D:/OneDrive/codex_projs/RinkSpace/src')
from rinkspace.user_contours import apply_user_visible_markings

def apply_latest_geometry_auxiliary(geometry, payload):
    """Consume traceable initial geometry; no source materials are rewritten."""
    aux = payload['geometry_initializer_auxiliary']
    template = geometry.parent
    if not hasattr(template, '_lane_proxy_local') or not hasattr(template, '_crease_unit'):
        raise TypeError('Explicit current structured template parent required')
    template._lane_proxy_local = {int(k): np.asarray(v, float) for k, v in aux['lane_proxy_local_m'].items()}
    template._crease_unit = {k: np.asarray(v, float) for k, v in aux['crease_unit'].items()}
    template.features['curling_centerline_row3'] = aux['latest_row3_feature']
    template.initial[:28] = np.asarray(payload['world66'], float)[:28]
    template._cache.clear()
    geometry._provisional_locations = {int(k): np.asarray(v, float)
                                       for k, v in aux['latest_paired_provisional_locations_m'].items()}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    here = _runtime_data_module_dir(__file__)
    first = json.loads((here/'smooth_source_geometry_initialization.json').read_text(encoding='utf8'))
    marking = json.loads((here/'latest9a8c_named28_initialization_ONLY.json').read_text(encoding='utf8'))
    oldfile = Path(marking['old_scene'])
    if sha(oldfile) != marking['old_scene_sha256']:
        raise ValueError('Latest scene has changed')
    old = json.loads(oldfile.read_text(encoding='utf8'))
    ownerfile = Path(first['owner'])
    if sha(ownerfile) != first['owner_sha256']:
        raise ValueError('Original material owner changed')
    with ownerfile.open('rb') as f:
        g = pickle.load(f)
    q = np.asarray(first['world66'], float).copy()
    q[:28] = marking['values']
    unit = float(g.scene_unit_m)
    bounds = np.asarray(old['parameters']['bounds'], float)
    if np.max(abs(bounds[:2]+bounds[2:])) > 1e-10:
        raise ValueError('Latest symmetric boundary needs an explicit origin migration')
    curve = old['parameters']['corner_curve']
    rx, ry = np.asarray(curve['tangent_extent_xy_scene_units'], float)*unit
    target = np.array([bounds[2]*unit,bounds[3]*unit,rx,ry,curve['exponent'],
                       old['parameters']['board_height']*unit],float)
    course, current, chain = g.print_course_binding(q)
    q[44:50] += np.linalg.solve(chain[:,44:50], target-current)
    course, cq, _ = g.print_course_binding(q)
    if np.max(abs(cq-target)) > 1e-12:
        raise ValueError('Latest boundary initialization failed')
    bottom = np.asarray(course.sample_perimeter(cq)[0],float)
    xy = bottom[:,:2]; edge = np.roll(xy,-1,axis=0)-xy
    lens = np.linalg.norm(edge,axis=1); lens2 = lens*lens
    cumulative = np.r_[0.,np.cumsum(lens)]; perimeter = cumulative[-1]
    ads = {int(a['user_number']):a for a in old['advertisements']}
    station_reports = []
    def station(point, identity, side):
        point = np.asarray(point,float)
        t = np.clip(np.einsum('ij,ij->i',point[:2]-xy,edge)/lens2,0.,1.)
        hits = xy+t[:,None]*edge
        distance2 = np.sum((hits-point[:2])**2,axis=1)
        k = int(np.argmin(distance2))
        station_reports.append({'ad':identity,'side':side,
                                'old_to_new_course_nearest_distance_m':float(np.sqrt(distance2[k])*unit)})
        return float((cumulative[k]+t[k]*lens[k])%perimeter)
    widths = {}
    heights = {}
    def location(n):
        c = ads[n]['corners_xyz_scene_units']
        left = station(c['left_top'],n,'left')
        right = station(c['right_top'],n,'right')
        signed = (right-left+perimeter/2)%perimeter-perimeter/2
        centre = ((left+signed/2)%perimeter)*unit
        widths[n] = float(ads[n].get('width_m',abs(signed)*unit))
        heights[n] = float(ads[n].get('height_m',
                           (c['left_top'][2]-c['left_bottom'][2])*unit))
        return np.array([centre,float(c['left_bottom'][2])*unit])
    for n in (1,2,6,7):
        idx = g._primary_indices[n]
        ad = ads[n]; loc = location(n)
        q[idx] = [loc[0],widths[n],loc[1]+heights[n]/2,heights[n]]
    for n in (3,4,5):
        k = g.parent_parameter_count+4*g.slot[n]
        loc = location(n);ad = ads[n]
        q[k:k+4] = [loc[0],widths[n],loc[1],loc[1]+heights[n]]
    provisional = {int(n):location(int(n)) for n in g._provisional_locations}
    for n,idx in g.location_indices.items():
        q[idx] = provisional[n]
    aux = dict(marking['auxiliary_template_initializers_from_latest_geometry'])
    aux['latest_row3_feature'] = next(f for f in old['ice_features'] if f['id']=='curling_centerline_row3')
    aux['latest_paired_provisional_locations_m'] = {str(n):v.tolist() for n,v in provisional.items()}
    payload = dict(status='LATEST_9A8C_GEOMETRY_INITIALIZATION_NOT_NEW_FIT',world66=q.tolist(),
                   parameter_names=list(g.parameter_names),geometry_initializer_auxiliary=aux,
                   old_scene=str(oldfile),old_scene_sha256=sha(oldfile),
                   old_complete_camera_seed=str(BASE/'current_common_hockey_soft_reference_source_freezes/9a8c1ac9d049/complete_frozen_solver_checkpoint_UNACCEPTED.npz'),
                   current_course6=cq.tolist(),advertisement_station_initialization=station_reports,
                   initialization_uses_old_solution_not_new_source_observations=True,
                   no_old_XYZ_prior_added=True,no_60x30_hard_target=True,
                   original_UV_weights_sigma_and_clock_unchanged=True)
    apply_latest_geometry_auxiliary(g,payload)
    g.validate_parameters(q)
    authority='D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json'
    scene=apply_user_visible_markings(g.scene(q),pairs=[(f'F{i}',f'R{i}')for i in range(1,5)],authority=authority)
    path=here/'latest9a8c_world66_initialization_ONLY.json'
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    (here/'latest9a8c_world66_scene_INITIAL_ONLY.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='READY',path=str(path),sha256=sha(path),course6=cq.tolist(),
                         source_observed_XYZ_or_UV_added=False),ensure_ascii=False))

if __name__=='__main__':
    main()
