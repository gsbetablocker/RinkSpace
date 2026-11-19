"""Read-only comparison of Ankara S14 board corrections to the released checkpoint."""
from argparse import ArgumentParser
from pathlib import Path
import sys

TOOLS = Path.cwd() / 'events/ankara_2026/tools/joint_refine_1117'
sys.path.insert(0, str(TOOLS))

import numpy as np

from flow_depths import rays
from motion_audit_1118 import (
    FLOW_METRICS, active_counts, flow_frames, gaps, ranks, score, summaries, trajectory,
)
from motion_delivery import camera_audit
from repair_motion import frame_median, stat
from review import ji
from shared import (
    EVENT, FIELD, OUT, NativeCore, NativeSceneGPU, ParameterState, SceneParameterModel,
    arrays, numeric_sha, programme_state_sha, read, save, sha, write,
)


BASE = OUT / 'lane4_side_1118/verified'
OLD_RELEASE = EVENT / 'scene/lane4_side_1118.json'
WORK = OUT / 's14_boards_1118/motion_check'
MOTION = BASE.parent / 'motion_evidence'
OLD_MOTION = MOTION / 'comparison_audit/lane4_side_1118/verified/audit.json'
TO_AUDIT = ('jin_s14', 'jin_s03', 'jin_s08', 'jin_s15', 'jin_s22', 'nayeon_s03', 'nayeon_s06')
CAMERA_KEYS = ('pts', 'times', 'rigs', 'controls', 'active_folds')
CONTROLS = {
    'jin_s14': (115.40, 115.42, 115.54, 121.64, 121.66, 122.50, 122.82, 122.98, 123.00, 123.02, 123.04, 123.06, 123.70, 123.82, 123.94, 123.96, 123.98),
    'jin_s03': (39.12,),
    'jin_s08': (69.36, 70.58, 70.84, 71.58, 71.82, 71.84, 71.86),
    'jin_s15': (124.66, 124.72, 124.76, 124.84, 126.30, 126.34, 127.32, 127.34, 127.36),
    'jin_s22': (154.52, 154.56, 154.66, 154.68, 154.88, 154.94, 155.06),
    'nayeon_s03': (42.34, 49.98, 50.80, 51.86, 51.94, 52.38),
    'nayeon_s06': (122.08, 122.10, 122.12, 122.14, 125.12, 125.2, 125.44, 125.54, 125.68),
}
ICE_FAMILIES = (
    'C0', 'B1', 'B2', 'F4', 'R4', 'A4', 'goal_crease_left', 'goal_line_left',
    'goal_crease_right', 'goal_line_right', 'lane1', 'lane2', 'lane3',
)
PROTECTED_S14 = {
    'F4': ('surface_verified_boundary',),
    'goal_line_left': ('inherited_qualified_source', 'reviewed_prior_source'),
    'goal_crease_left': ('surface_verified_boundary',),
    'board.top': ('blue_cap_far', 'blue_rail_pair', 'guarded_source_observation'),
    'board.bottom': ('review_rail_edge', 'reviewed_prior_source'),
    'Ad1': ('whole_panel_side', 'panel_edge'),
    'Ad2': ('whole_panel_side', 'panel_edge'),
    'Ad3': ('whole_panel_side', 'panel_edge'),
    'Ad4': ('whole_panel_side', 'panel_two_materials', 'ink_same_visible_letter', 'panel_edge'),
}


def file_ref(path):
    return dict(path=str(path), sha256=sha(path))


def checked(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError(('Evidence hash mismatch', path))
    return path


def native_for(state, dll):
    field = Path(str(state.get('scene_path', FIELD)))
    mdl = SceneParameterModel(read(field), field, optimize_scene=True)
    return NativeSceneGPU(field, NativeCore(dll),
        parameter_state=ParameterState(mdl.initial_state.layout, state['scene_values']))


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def camera_fields(a, b, name):
    return {k: name+'_'+k in b and exact(a[name+'_'+k], b[name+'_'+k]) for k in CAMERA_KEYS}


def scene_changes(natives):
    old, new = natives
    if list(old.paths) != list(new.paths):
        raise ValueError('Scene path topology/order changed')
    return [k for k in old.paths if not all(
        np.array_equal(old.paths[k].get(field, ()), new.paths[k].get(field, ()))
        for field in ('kind', 'parameters', 'vertices', 'knots', 'support'))]


def sensor(native, state, name, shot):
    """Evaluate the release's nine native-source sensor positions at every exposure."""
    t = state[name+'_times']
    rigs, inv = np.unique(state[name+'_rigs'], axis=0, return_inverse=True)
    samples = np.array([[x, y] for y in (0, 539.5, 1079) for x in (0, 959.5, 1919)])
    jac, warp, inverse = [np.full((len(t), len(samples)), np.nan) for _ in range(3)]
    for rid, rig in enumerate(rigs):
        ids = np.flatnonzero(inv == rid)
        uv = np.tile(samples, (len(ids), 1))
        time = np.repeat(t[ids], len(samples)) + rig[37]*(uv[:,1]/1079.-.5)
        c = native.core.evaluate_motion_camera(rig, state[name+'_controls'], shot['first'], shot['dt'], time)
        ray = np.einsum('nij,nj->ni', c['rotation'], rays(uv, c))
        x, y = (ray[:,:2]/ray[:,2:]).T
        r2 = x*x + y*y
        radial = 1 + r2*(c['k1'] + r2*c['k2'])
        d = c['k1'] + 2*c['k2']*r2
        p, q = c['p1'], c['p2']
        a = radial + 2*x*x*d + 2*p*y + 6*q*x
        b = 2*x*y*d + 2*p*x + 2*q*y
        cc = radial + 2*y*y*d + 6*p*y + 2*q*x
        eig = (a+cc-np.hypot(a-cc, 2*b))/2
        reproj = np.c_[c['cx']+c['fx']*(x*radial+2*p*x*y+q*(r2+2*x*x)),
                       c['cy']+c['fy']*(y*radial+p*(r2+2*y*y)+2*q*x*y)]
        undist = np.c_[c['cx']+c['fx']*x, c['cy']+c['fy']*y]
        valid = (c['status']==0) & (eig > 0) & np.isfinite(reproj).all(1)
        for slot, v in zip((jac, warp, inverse), (eig, np.linalg.norm(undist-uv, axis=1),
                                                      np.linalg.norm(reproj-uv, axis=1))):
            slot[ids] = np.where(valid, v, np.nan).reshape(len(ids), len(samples))
    valid = np.isfinite(jac)
    return dict(checked=int(jac.size), invalid=int((~valid).sum()),
        minimum_sampled_jacobian=float(jac[valid].min()) if valid.any() else None,
        maximum_image_distortion_px=float(warp[valid].max()) if valid.any() else None,
        maximum_sensor_roundtrip_px=float(inverse[valid].max()) if valid.any() else None,
        invalid_ranges=gaps(state[name+'_pts'], (~valid).any(1))), jac, warp, inverse


def compare_flow(index, lag, family, uv_delta, scores, pts):
    matched = []
    common = [np.isfinite(a).all(1) & np.isfinite(b).all(1)
              for a,b in zip(scores[0][:2],scores[1][:2])]
    report = {}
    for i, (disp, curve, _) in enumerate(scores):
        frame = {}
        flow_frames(frame,pts,index,family,lag,uv_delta,disp,curve)
        paired = {}
        flow_frames(paired,pts,index,family,lag,uv_delta,
            np.where(common[0][:, None], disp, np.nan), np.where(common[1][:, None], curve, np.nan))
        matched.append(paired)
        report[i] = frame
    return report, matched


def pair_report(name, pts, before, after):
    same = np.isfinite(before) & np.isfinite(after)
    inc = ranks(name, pts, np.where(same, after-before, np.nan), count=5)
    for item in inc:
        i = np.searchsorted(pts, item['pts'])
        item['old'] = float(before[i])
        item['new'] = float(after[i])
    return dict(old=stat(before[same]), new=stat(after[same]), largest_increases=inc,
        old_only=int((np.isfinite(before)&~np.isfinite(after)).sum()),
        new_only=int((~np.isfinite(before)&np.isfinite(after)).sum()))


def verified_main(name, shot, pts, original):
    source = arrays(checked(dict(path=original['path'],sha256=original['sha256'])))
    retained = arrays(checked(dict(path=original['baseline_motion_record'],
                                  sha256=original['baseline_motion_sha256'])))
    np.testing.assert_array_equal(source['pts'],pts)
    np.testing.assert_array_equal(retained['pts'],pts)
    if 'source_sha256' in source and source['source_sha256'].item() != shot['source_sha256']:
        raise ValueError((name, 'motion source identity'))
    rows = retained['source_row']
    for k in ('index','family','lag'):
        np.testing.assert_array_equal(source[k][rows],retained[k])
    return source, rows


def cached_score(folder, stamp, key, native, state, name, shot, pts, source, rows):
    cache = folder / (key+'.npz')
    if cache.exists():
        item = arrays(cache)
        if str(item['stamp']) == stamp:
            return tuple(item[k] for k in ('displacement','curvature','depth'))
    out = score(native,shot,state,name,pts,source,rows)
    save(cache,stamp=stamp,displacement=out[0],curvature=out[1],depth=out[2])
    return out


def probes(name, shot, pts, references, reviews, natives, states, identities, same_flow, out, cache):
    report = {}
    coverage = [np.zeros(len(pts),bool),np.zeros(len(pts),bool)]
    requested = {round(t*12800) for t in CONTROLS[name]}
    for label, ref in references.items():
        path = checked(dict(path=ref['packet'], sha256=ref['packet_sha256']))
        source = arrays(path)
        np.testing.assert_array_equal(source['pts'], pts)
        if source['source_sha256'].item() != shot['source_sha256']:
            raise ValueError((name,label,'probe source identity'))
        ready = source['keep'].copy()
        manual = reviews.get(path.relative_to(MOTION).as_posix()) if path.is_relative_to(MOTION) else None
        if manual:
            if manual['packet_sha256'] != ref['packet_sha256']:
                raise ValueError((path,'probe exclusions refer to another packet'))
            ready[manual['excluded_rows']] = False
        rows = np.flatnonzero(ready)
        i, lag = source['index'][rows], source['lag'][rows]
        pairs = np.c_[pts[i-lag],pts[i]]
        np.testing.assert_array_equal(source['source_pair_pts'][rows], pairs)
        actual = source['uv'][rows]-source['left_uv'][rows]
        old = cached_score(cache, identities[0]+ref['packet_sha256'], name+'_'+label,
                           natives[0],states[0],name,shot,pts,source,rows)
        new = old if same_flow else score(natives[1],shot,states[1],name,pts,source,rows)
        e = [np.linalg.norm(a[0], axis=1) for a in (old,new)]
        minimum = 3 if label.startswith('ice') else 4
        for c,v in enumerate(e):
            coverage[c] |= np.bincount(i[(lag==1)&np.isfinite(v)],minlength=len(pts)) >= minimum
        common = np.isfinite(e[0]) & np.isfinite(e[1])
        focus, overall = {}, {}
        values = dict(pts=pts,source_rows=rows,pair_pts=pairs,actual=actual,lag=lag,i=i,
            old_displacement=old[0],new_displacement=new[0],old_curvature=old[1],new_curvature=new[1],
            old_depth=old[2],new_depth=new[2],held=source['held'][rows])
        for ell in np.unique(lag):
            mask = common & (lag == ell)
            each = [frame_median(i,x,mask,len(pts),minimum) for x in e]
            overall[str(ell)] = pair_report(name,pts,*each)
            for side,v in zip(('old','new'),each):
                values[side+'_lag'+str(ell)] = v
            for pt in sorted(requested & set(pairs[lag==ell,1])):
                candidate = (lag==ell)&(pairs[:,1]==pt)
                zones = {'all': candidate, 'held': candidate & source['held'][rows]}
                if label.startswith('ice'):
                    zones['foreground'] = candidate & (source['left_uv'][rows,1] >= 730)
                zone_stats = {}
                for zone,sel in zones.items():
                    use = sel & common
                    if sel.any():
                        zone_stats[zone] = dict(rows=int(sel.sum()),matched=int(use.sum()),
                            old=stat(e[0][use]),new=stat(e[1][use]),
                            measured_median_uv=np.median(actual[use],axis=0).tolist() if use.any() else None)
                focus.setdefault(str(pt),dict(seconds=pt/12800))[str(ell)] = zone_stats
        path_out = out/'probes'/(name+'_'+label+'.npz')
        save(path_out,**values)
        report[label] = dict(source=dict(path=str(path),sha256=ref['packet_sha256']),
            source_rows=len(rows), checks=focus, shared_frames=overall, archive=file_ref(path_out))
    return report, coverage


def scalar_summary(before,after):
    paired = np.isfinite(before)&np.isfinite(after)
    return dict(old=stat(before[paired]),new=stat(after[paired]),rows=len(before),
                old_only=int((np.isfinite(before)&~np.isfinite(after)).sum()),
                new_only=int((~np.isfinite(before)&np.isfinite(after)).sum()))


def control_report(name, shot, sources, natives, states, camera_same, changed, out, cache):
    from lane_review import pixel_errors
    from rinkspace.calibration_companion.wide_angle_camera import nearest_valid_polyline_vectors

    binding = sources[name]
    path = checked(binding['packet'])
    parent_error_file = checked(binding['matched_errors'])
    stamp = read(parent_error_file.with_suffix('.json'))
    if stamp['state'] != programme_state_sha(states[0],[name]) or stamp['packet_sha256'] != binding['packet']['sha256']:
        raise ValueError((name,'baseline source score binding'))
    a = arrays(path)
    e0 = arrays(parent_error_file)['error']
    pts = states[0][name+'_pts']
    np.testing.assert_array_equal(pts[a['idx']],a['pts'])
    assert binding['source_sha256'] == shot['source_sha256'] and len(e0) == len(a['pts'])
    families = np.unique(a['family'])
    keep = np.isin(a['family'],ICE_FAMILIES) | np.char.startswith(a['family'],'Ad')
    keep &= (a['fold'] < 9) & np.isfinite(a['uv']).all(1)
    measure = keep & (np.isin(a['path'],changed) if camera_same else True)
    e1 = e0.copy()
    if measure.any():
        e, _ = pixel_errors(natives[1],states[1],shot,name,a,measure)
        e1[measure] = e[measure]
    watch = {}
    archive = {'pts': pts}
    for role, role_mask in [('fit',a['fold'] < 5), ('check',(a['fold']>=5)&(a['fold']<9))]:
        for f in families:
            sel = keep & role_mask & (a['family'] == f)
            if not sel.any():
                continue
            f0,f1 = [
                frame_median(a['idx'],e,sel, len(pts),1) for e in (e0,e1)
            ]
            counts = np.bincount(a['idx'][sel], minlength=len(pts))
            k = f+'_'+role
            archive.update({k+'_old':f0,k+'_new':f1,k+'_rows':counts})
            watch.setdefault(f,{})[role] = dict(
                by_row=scalar_summary(e0[sel],e1[sel]),by_exposure=pair_report(name,pts,f0,f1),
                key_exposures={str(pt):scalar_summary(e0[sel&(a['pts']==pt)],e1[sel&(a['pts']==pt)])
                    for pt in [round(t*12800) for t in CONTROLS[name]] if (sel&(a['pts']==pt)).any()})
    if name != 'jin_s14':
        exact_group = {}
    else:
        # Match the prior control protocol at 1025 samples per finite curve.
        pockets = {}
        for t in (122.98,123.02,123.06):
            for family,mats in PROTECTED_S14.items():
                if family.startswith('board.') and changed:
                    continue  # Superseded rail material is evaluated separately against the new source trace.
                sel = (a['pts']==round(t*12800)) & (a['family']==family) & np.isin(a['material'],mats) & (a['fold']<9)
                if sel.any():
                    pockets[f'{t:.2f} {family}'] = np.flatnonzero(sel)
        ids = np.unique(np.concatenate(list(pockets.values())))
        jobs = sorted(set(zip(a['idx'][ids],a['path'][ids])))
        paths = sorted(set(k for _,k in jobs))
        exact_losses = []
        for native,state in zip(natives,states):
            table = ji.native_paths(native,paths,count=1025)
            values = np.full(len(a['pts']),np.nan)
            jobs_time = [(i,k,state[name+'_times'][i]) for i,k in jobs]
            projected = ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],
                shot['first'],shot['dt'],jobs_time,{k:v[0] for k,v in table.items()})
            for (i,k),(uv,status) in zip(jobs,projected,strict=True):
                group=ids[(a['idx'][ids]==i)&(a['path'][ids]==k)]
                ok = (status==0)&np.isfinite(uv).all(1)
                if ok.sum() > 1:
                    delta,segments = nearest_valid_polyline_vectors(a['uv'][group],uv,ok)
                    if len(segments)==len(group):
                        values[group] = np.linalg.norm(delta,axis=1)
            exact_losses.append(values)
        exact_group = {k:scalar_summary(exact_losses[0][ids],exact_losses[1][ids]) for k,ids in pockets.items()}
    target = out/'controls'/(name+'.npz')
    save(target,**archive)
    return dict(source_packet=binding['packet'],baseline_error_ref=binding['matched_errors'],
        matched_development_evidence=watch, exact_prior_s14_control=exact_group, file=file_ref(target))



APPEARANCE = BASE.parent/'appearance_audit'
VERIFIED_READINGS = APPEARANCE/'verified/source_readings_change.json'


def source_reading_controls(name, shot, natives, states):
    """Same source pixels, fixed measurement and roles from prior protected reads."""
    from repair_motion import project
    from rinkspace.calibration_companion.wide_angle_camera import nearest_valid_polyline_vectors

    pts = states[0][name+'_pts']
    refs = {}
    reading_roles = {}
    old = read(VERIFIED_READINGS)
    if old['state_sha256']['lane4_side_1118/verified'] != sha(BASE/'state.npz'):
        raise ValueError('Protected readings were checked on a different baseline')
    for label, ref in old['shots'].get(name,{}).get('readings',{}).items():
        refs[label]=(checked(dict(path=ref['source_packet'],sha256=ref['source_packet_sha256'])), False)
    if name == 'jin_s14':
        refs.update(
            F4_and_crease_horizontal_tips=(APPEARANCE/'repair/jin_s14_tips.npz',True),
            LANE1_dual_flanks=(APPEARANCE/'final_paint/jin_s14.npz',False),
            A4_raw=(APPEARANCE/'jin_repair_candidates/jin_s14.npz',False),
        )
    for label,(path,tip) in refs.items():
        data = arrays(path)
        source_idx = data['idx']
        a = data['source_mid'] if 'source_mid' in data else data['uv']
        np.testing.assert_array_equal(pts[source_idx],data['pts'])
        if 'source_sha256' in data and data['source_sha256'].item() != shot['source_sha256']:
            raise ValueError((name,label,'direct reading source identity mismatch'))
        keys = np.unique(data['path'])
        jobs = sorted(set(zip(source_idx,data['path'])))
        scores = []
        for native,state in zip(natives,states):
            table = ji.native_paths(native,keys,count=1025)
            values = np.full(len(a),np.nan)
            for i,k in jobs:
                use = (source_idx==i)&(data['path']==k)
                world = table[k][0]
                uv = project(native,shot,state,name,np.full(len(world),i),world)
                valid = np.isfinite(uv).all(1)
                if valid.sum() < 2:
                    continue
                if tip:
                    values[use] = uv[valid,0].max()-a[use,0]
                else:
                    vectors, segments = nearest_valid_polyline_vectors(a[use],uv,valid)
                    if len(segments) == use.sum():
                        values[use] = (np.linalg.norm(vectors,axis=1) if 'normal' not in data
                                       else np.sum(vectors*data['normal'][use],axis=1))
            scores.append(values)
        role = data.get('role',data.get('fold',np.full(len(a),'source_control')))
        family = data.get('family',np.full(len(a),label))
        def metrics(mask):
            result=scalar_summary(abs(scores[0][mask]),abs(scores[1][mask]))
            if 'normal' in data or tip:
                result['signed_p50']={side:stat(x[mask]).get('p50')
                                      for side,x in zip(('old','new'),scores)}
            return result
        by_family = {str(f):metrics(family==f) for f in np.unique(family)}
        by_role = {str(r):metrics(role==r) for r in np.unique(role)}
        by_pts = {str(pt):dict(seconds=float(pt/12800), families={
            str(f):metrics((data['pts']==pt)&(family==f))
            for f in np.unique(family[data['pts']==pt])}) for pt in np.unique(data['pts'])}
        reading_roles[label] = dict(packet=file_ref(path),metric=(
            'horizontal outline extremum' if tip else 'source normal' if 'normal' in data
            else 'finite-curve normal distance'),overall=by_family,roles=by_role,exposures=by_pts)
    return reading_roles


def camera_deltas(name, pts, states, frames):
    before, after = [s[name+'_rigs'] for s in states]
    columns = {
        'base_rotation_matrix': slice(0,9), 'pan_axis': slice(9,12), 'tilt_axis': slice(12,15),
        'pivot_L': slice(15,18), 'lever_L': slice(18,21), 'axis_spacing_L': slice(21,22),
        'entrance_pupil_linear_L': slice(22,23), 'entrance_pupil_quadratic_L': slice(23,24),
        'focal_reference_px': slice(24,25), 'sensor_inverse_radius': slice(25,26),
        'principal_x_reference_px': slice(26,27), 'principal_y_reference_px': slice(27,28),
        'log_pixel_aspect': slice(28,29), 'principal_x_zoom_slope_px': slice(29,30),
        'principal_y_zoom_slope_px': slice(30,31), 'k1_reference': slice(31,32),
        'k1_zoom_slope': slice(32,33), 'k2_reference': slice(33,34), 'k2_zoom_slope': slice(34,35),
        'p1': slice(35,36), 'p2': slice(36,37), 'readout_s': slice(37,38),
    }
    fields = {}
    for field, col in columns.items():
        old, new = before[:,col], after[:,col]
        d = new-old
        fields[field] = dict(old_mid=old[len(old)//2].tolist(),new_mid=new[len(new)//2].tolist(),
            max_abs_component_delta=np.max(abs(d),axis=0).tolist(),vector_delta=stat(np.linalg.norm(d,axis=1)))
    for field in ('pan_axis','tilt_axis'):
        sl=columns[field]
        angle=np.degrees(np.arccos(np.clip(
            np.sum(before[:,sl]*after[:,sl],axis=1) /
            (np.linalg.norm(before[:,sl],axis=1)*np.linalg.norm(after[:,sl],axis=1)), -1.,1.)))
        fields[field]['degrees_between']=stat(angle)
    sl=columns['base_rotation_matrix']
    angle=np.degrees(np.arccos(np.clip((np.sum(before[:,sl]*after[:,sl],axis=1)-1.)/2.,-1.,1.)))
    fields['base_rotation_matrix']['degrees_between']=stat(angle)
    for key,rig,side in [('old',before,'old_'),('new',after,'new_')]:
        z = frames[side+'log_f']-np.log(rig[:,24])
        pupil = rig[:,22]*z+rig[:,23]*z*z
        axial = rig[:,20]+pupil
        fields.setdefault('evaluated_pupil_shift_L',{})[key] = stat(pupil)
        fields.setdefault('evaluated_axial_lever_L',{})[key] = stat(axial)
        fields.setdefault('evaluated_pupil_shift_range_L',{})[key] = [float(np.nanmin(pupil)),float(np.nanmax(pupil))]
        fields.setdefault('evaluated_axial_lever_range_L',{})[key] = [float(np.nanmin(axial)),float(np.nanmax(axial))]
        frames[side+'evaluated_pupil_shift_L'] = pupil
        frames[side+'evaluated_axial_lever_L'] = axial
    for f in ('fx','fy','cx','cy','k1','k2','p1','p2','evaluated_pupil_shift_L','evaluated_axial_lever_L'):
        old, new = [frames[side+'_'+f] for side in ('old','new')]
        abs_delta = abs(new-old)
        fields[f+'_abs_exposure_delta'] = dict(**stat(abs_delta),worst=ranks(name,pts,abs_delta,count=3),
            focus={str(t):[float(old[i]),float(new[i])] for t in CONTROLS[name]
                   if (i:=np.searchsorted(pts,round(t*12800))) < len(pts)
                   and pts[i]==round(t*12800) and np.isfinite(old[i]) and np.isfinite(new[i])})
    displacement=np.linalg.norm(frames['new_centre_L']-frames['old_centre_L'],axis=1)
    fields['conditional_optical_centre_dist_L'] = dict(**stat(displacement),worst=ranks(name,pts,displacement,count=4))
    return fields

def resolve_candidate(value):
    path = Path(value)
    if not path.is_absolute():
        path = OUT / path
    return (path / 'state.npz' if path.is_dir() else path).resolve()


def audit(opt):
    baseline_reg = read(OLD_RELEASE)
    old_dll = checked(baseline_reg['geometry_binding']['native_library'])
    paths = [checked(baseline_reg['checkpoint']), resolve_candidate(opt.candidate)]
    identities = [sha(p) for p in paths]
    libs = [old_dll, opt.library or old_dll]
    ready = [checked(baseline_reg['fit_setup']), opt.prepared]
    older = read(OLD_MOTION)
    published_motion = older['candidates']['lane4_side_1118/verified']
    if identities[0] != published_motion['sha256']:
        raise ValueError(('Baseline audit belongs to another checkpoint',published_motion))
    setup = read(ready[0])['shots']
    new_setup = read(ready[1] / 'setup.json' if ready[1].is_dir() else ready[1])['shots'] if ready[1] else None
    states = [arrays(p) for p in paths]
    natives = [native_for(s,dll) for s,dll in zip(states,libs)]
    changes = scene_changes(natives)
    same_board = np.array_equal(natives[0].scene['board'],natives[1].scene['board'])
    same_library = sha(libs[0]) == sha(libs[1])
    frozen = {n:camera_fields(*states,n) for n in setup}
    names = opt.shots or TO_AUDIT
    out = WORK / opt.output
    if not out.is_relative_to(WORK) or out.resolve() == WORK.resolve():
        raise ValueError('--output must be a child of the motion-check directory')
    out = out.resolve()
    if not out.is_relative_to(WORK.resolve()):
        raise ValueError('--output resolves outside motion-check directory')
    cache = WORK/'cache'/('baseline_'+identities[0][:12]+'_'+sha(old_dll)[:12])
    old_source = read(checked(baseline_reg['source_bindings']))
    new_binding_file = paths[1].parent/'source_bindings.json'
    new_source = read(new_binding_file) if new_binding_file.is_file() else {}
    reviews = read(MOTION/'manual_exclusions.json')
    references = older['jin_source_windows'] | older['other_source_windows']
    old_sensors = read(checked(baseline_reg['optics']))
    result = {}
    for name in names:
        shot = setup[name]
        if new_setup:
            for k in ('source_sha256','first','dt','camera_index'):
                if shot[k] != new_setup[name][k]:
                    raise ValueError((name,'source-clock or installation change',k))
        pts = states[0][name+'_pts']
        np.testing.assert_array_equal(pts,states[1][name+'_pts'])
        np.testing.assert_array_equal(states[0][name+'_times'],states[1][name+'_times'])
        same_cam = same_library and all(frozen[name][k] for k in CAMERA_KEYS[:-1])
        same_flow = same_cam and same_board
        ledger = older['shots'][name]['legacy_motion_input']
        source, rows = verified_main(name,shot,pts,ledger)
        i, lag, family = (source[k][rows] for k in ('index','lag','family'))
        uv_delta=source['uv'][rows]-source['left_uv'][rows]
        sig = identities[0]+ledger['sha256']+ledger['baseline_motion_sha256']
        baseline = cached_score(cache,sig,name,natives[0],states[0],name,shot,pts,source,rows)
        new = baseline if same_flow else score(natives[1],shot,states[1],name,pts,source,rows)
        frames, paired = compare_flow(i,lag,family,uv_delta,(baseline,new),pts)
        comparisons = {m:pair_report(name,pts,paired[0][m],paired[1][m]) for m in FLOW_METRICS}
        probe_data, probe_coverage = probes(name,shot,pts,references.get(name,{}),reviews,natives,states,
                                           identities,same_flow,out,cache)
        data, projection = {'pts':pts}, {}
        for side,native,state,bindings,prepared in zip((0,1),natives,states,(old_source,new_source),(None,new_setup)):
            values=trajectory(native,shot,name,state,pts)
            values.update(frames[side])
            values['fit_rows'], fit_ref = active_counts(name,shot,state,pts,bindings,prepared)
            lens,jac,warp,inverse = sensor(native,state,name,shot)
            stats = summaries(name,pts,values)
            count = np.isfinite(values['board_adjacent'])|np.isfinite(values['ice_adjacent'])|probe_coverage[side]
            stats['coverage']['adjacent_intervals_without_finite_comparison_with_probes']=int((~count[1:]).sum())
            stats['coverage']['missing_motion_ranges_with_probes']=gaps(pts,~count & (np.arange(len(pts))!=0))
            stats['coverage']['fit_binding']=fit_ref
            try:
                physical = camera_audit(native,state,name,shot)
            except ValueError as err:
                physical = {'failure':str(err)}
            projection[('old','new')[side]]=dict(trajectory=stats,physical=physical,sensor=lens)
            for k,v in (values | {'sensor_jacobian':jac,'sensor_warp':warp,'sensor_roundtrip':inverse}).items():
                data[('old','new')[side]+'_'+k]=v
            for k in FLOW_METRICS:
                data[('old','new')[side]+'_matched_'+k]=paired[side][k]
        rig_delta = camera_deltas(name,pts,states,data)
        data_file = out/'frames'/(name+'.npz')
        save(data_file,**data)
        result[name]=dict(source=shot['source'], source_sha256=shot['source_sha256'],
            matched_motion=comparisons, cameras=projection, rig_and_optics_delta=rig_delta, frame_arrays=file_ref(data_file),
            base_track_ledger=ledger, baseline_registered_sensor=old_sensors[name]['after'],
            source_probes=probe_data)
        if not opt.motion_only:
            result[name]['direct_source_controls'] = source_reading_controls(name,shot,natives,states)
            result[name]['controls'] = control_report(name,shot,old_source,natives,states,same_cam,
                list(natives[0].paths) if not same_board else changes,out,cache)
        print(name,len(pts),[(m, tuple(round(comparisons[m][k].get('p95',-1),3)
              for k in ('old','new'))) for m in FLOW_METRICS],
              result[name]['cameras']['new']['trajectory']['coverage']['adjacent_intervals_without_finite_comparison_with_probes'],
              flush=True)
    for p,d in zip(paths,identities):
        if sha(p) != d:
            raise ValueError(('Checkpoint changed during audit',p))
    same_shots = [n for n,v in frozen.items() if all(v.values())]
    cache_report = {'old_motion':file_ref(OLD_MOTION),'old_optics':baseline_reg['optics'],
        'old_source_controls':baseline_reg['source_bindings']}
    report = dict(baseline=file_ref(paths[0]), candidate=file_ref(paths[1]),
        library_before=file_ref(libs[0]),library_after=file_ref(libs[1]),
        scene_before=file_ref(natives[0].scene_path),scene_after=file_ref(natives[1].scene_path),
        source_evidence=cache_report,scene_board_equal=bool(same_board),changed_path_records=list(changes),
        numeric_before=numeric_sha(states[0]['scene_values']),numeric_after=numeric_sha(states[1]['scene_values']),
        time_base=[1,12800],camera_element_byte_identity=frozen,
        unchanged_cameras=dict(shots=same_shots,exposures=sum(len(states[0][n+'_pts']) for n in same_shots)),
        notes='Motion binds the same stored original-image track to each candidate with conditional surface depth. Per-frame medians need 4 board or 3 ice rows. Extended probes are separate correlated development evidence. Source control roles have their original packet assignments. Exact stored camera equality alone leaves changed scene/board appearance unverified.',
        shots=result)
    write(out/'comparison.json',report)
    print(out/'comparison.json',flush=True)


def registry(reg_path, root):
    """Prove exact exported geometry, programme cameras, source clock and packet identities."""
    reg = read(reg_path)
    state_file = checked(reg['checkpoint'])
    state = arrays(state_file)
    geo = reg['geometry_binding']
    snapshot = arrays(checked(geo['snapshot']))
    checked(geo['structure'])
    checked(geo['native_library'])
    np.testing.assert_array_equal(snapshot['scene_values'],state['scene_values'])
    assert snapshot['scene_values_sha256'].item() == numeric_sha(state['scene_values']) == geo['value_sha256']
    assert snapshot['layout_revision'].item() == state['layout_revision'].item() == geo['layout_revision']
    ss = read(checked(reg['fit_setup']))['shots']
    bind = read(checked(reg['source_bindings']))
    q = {}
    for prog,ref in reg['programmes'].items():
        programme = read(checked(ref))
        assert programme['geometry_binding'] == geo
        cam_ref = programme['camera']
        camera = arrays(checked(cam_ref))
        assert camera['scene_values_sha256'].item() == geo['value_sha256']
        assert camera['layout_revision'].item() == geo['layout_revision']
        for name in programme['shots']:
            s = ss[name]
            assert programme['source']['sha256'] == s['source_sha256'] == bind[name]['source_sha256']
            assert programme['fit_observations'][name] == bind[name]['packet']
            obs_file = checked(bind[name]['packet'])
            with np.load(obs_file, allow_pickle=False) as obs:
                np.testing.assert_array_equal(state[name+'_pts'][obs['idx']],obs['pts'])
            sid = s['shot']
            if 'shot_ids' in camera:
                i = camera['shot_ids'].tolist().index(sid)
                lo,hi = camera['control_offsets'][i:i+2]
                u,v = camera['source_offsets'][i:i+2]
                np.testing.assert_array_equal(camera['source_pts'][u:v],state[name+'_pts'])
                np.testing.assert_array_equal(camera['local_exposure_seconds'][u:v],state[name+'_times'])
                np.testing.assert_array_equal(camera['controls'][lo:hi],state[name+'_controls'])
                np.testing.assert_array_equal(camera['rig_by_source_pts'][u:v],state[name+'_rigs'])
                np.testing.assert_array_equal(camera['first_control_time'][i],s['first'])
                np.testing.assert_array_equal(camera['control_dt'][i],s['dt'])
                np.testing.assert_array_equal(camera['camera_indices'][i],s['camera_index'])
            else:
                np.testing.assert_array_equal(camera[sid+'_pts'],state[name+'_pts'])
                np.testing.assert_array_equal(camera[sid+'_controls'],state[name+'_controls'])
                np.testing.assert_array_equal(np.broadcast_to(camera[sid+'_rig'],state[name+'_rigs'].shape),
                    state[name+'_rigs'])
                times = camera.get(sid+'_exposure_times',(state[name+'_pts']-state[name+'_pts'][0])/12800.)
                np.testing.assert_array_equal(times,state[name+'_times'])
            np.testing.assert_array_equal(camera['active_observation_folds'],state[name+'_active_folds'])
            q[name] = dict(exposures=len(state[name+'_pts']),programme=prog,packet=bind[name]['packet'],
                           camera=cam_ref,pts_range=state[name+'_pts'][[0,-1]].tolist(),
                           source_sha256=s['source_sha256'])
        full = programme.get('full_review')
        if full:
            assert full['render_state'] == reg['checkpoint']
            assert full['programme_state_sha256'] == programme_state_sha(state,programme['shots'])
    assert set(ss) == set(q) == set(bind)
    write(root/'registry_proof.json',dict(registration=file_ref(reg_path),checkpoint=file_ref(state_file),
        geometry=geo,shots=q,programmes=len(reg['programmes']),exposures=sum(v['exposures'] for v in q.values())))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--candidate')
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--shots', nargs='+', choices=TO_AUDIT)
    parser.add_argument('--motion-only',action='store_true')
    parser.add_argument('--registration',type=Path)
    options = parser.parse_args()
    if options.candidate:
        audit(options)
    if options.registration:
        dest = (WORK/options.output).resolve()
        if not dest.is_relative_to(WORK.resolve()) or dest == WORK.resolve():
            raise ValueError('Registry proof location must be inside the motion-check directory')
        registry(options.registration,dest)
