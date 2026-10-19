"""Small source-corner compatibility check; never export a programme camera."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from z190_main_source_objective_audit import load_current_graph,DEFAULT_CURRENT,CURRENT,ref,jsonable
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA
from main7_source_bundle_candidate import Bundle
from z190_low_opening_rf_refine import solve_frozen_pivot
from z190_source_camera_diagnostic import absolute_value_jac15,future_groups
from continuous_central_brown4_optin import enable_projection_backend
from ad_landmarks import AdGeometry


def candidate_rows(path,g):
    raw=json.loads(path.read_text(encoding='utf8'));rows=[]
    for r in raw:
        if (r['split']!='fit' or r.get('held_pixels_promoted') is not False
                or r.get('untouched_future_pixels_consumed') is not False
                or r['source_video_sha256']!=SOURCE_SHA or r['source_scene_sha256']!=HARD_SCENE_SHA
                or not r.get('point_correspondence')):
            raise ValueError('New independent source packet is not explicit development corner evidence')
        native=int(r['source_native_pts']);p=g.aliases[native]
        if p!=int(r['pts']) or g.cuts[p]!=1:raise ValueError('Source crosses current exact exposure/cut binding')
        if not any(row['split']=='fit' for stamp,rr in g.bundle.abs_groups if stamp==p for row in rr):
            raise ValueError('New source is not an existing development-fit exposure')
        row=dict(r)
        for k in ('world','uv','sigma'):row[k]=np.asarray(row[k],float)
        if len(row['world'])!=len(row['uv']) or len(row['sigma'])!=len(row['uv']):raise ValueError('Source corner shapes differ')
        expected=[]
        for name in row['physical_ids']:
            ad,role=name.split(':');geometry=AdGeometry(g.scene,int(ad[2:]));expected.append(geometry.corner(geometry.initial,role))
        if not np.array_equal(row['world'],expected):raise ValueError('New source world differs from actual hard-scene ad corner')
        rows.append(row)
    return rows


def residuals(bundle,q,p,rows):
    camera=bundle.camera(q,p)
    result=[]
    for row in rows:
        value=absolute_value_jac15(camera,[row],False).reshape(-1,2)*row['sigma'][:,None]
        for i,physical in enumerate(row['physical_ids']):
            result.append(dict(id=row['id'],physical_id=physical,source_native_pts=row.get('source_native_pts',row['pts']),
                uv=row['uv'][i],sigma=row['sigma'][i],residual_native_xy=value[i],error_native_px=np.linalg.norm(value[i])))
    return result


def run(a):
    if (a.output/'report.json').exists():raise ValueError('Use a fresh single-source check directory')
    a.output.mkdir(parents=True,exist_ok=True);g=load_current_graph(a.current);added=candidate_rows(a.new_source,g)
    stamps={int(r['pts']) for r in added}
    if len(stamps)!=1:raise ValueError('This is explicitly one exposure diagnostic')
    p=stamps.pop();original=[r for stamp,rr in g.bundle.abs_groups if stamp==p for r in rr if r.get('point_correspondence')]
    if not original or any(r['split']!='fit' for r in original):raise ValueError('Original fit corners unavailable')
    future,_=future_groups(g.base,g.scene,g.previous/'all_main_candidate_cameras.npz')
    future_hits=[dict(pts=h['pts'],cut=h['cut']) for h in future if g.aliases.get(int(h['pts']))==p]
    if future_hits:raise ValueError('New exposure is an untouched future source check')
    rows=original+added;base=Bundle([p],{p:rows},[])
    bundle=StandingSourceBundle(base,g.calibration,prior_wide_focal_px=g.manifest['frozen_wide_prior_focal_px'],
        fixed_height=True,fixed_height_scene_units=g.calibration.pivot_xyz[2])
    index=g.bundle.index[p];q0=np.r_[g.q[:6],g.q[6+4*index:10+4*index]]
    if not np.array_equal(bundle.camera(q0,p),g.model.camera25(g.views[p])):raise ValueError('Source check initial camera differs from sole completed warm')
    enable_projection_backend();r,j=bundle.evaluate(q0,include_priors=False);checks=[]
    for c in range(6,10):
        h=1e-7;x,y=q0.copy(),q0.copy();x[c]+=h;y[c]-=h
        fd=(bundle.evaluate(x,False,include_priors=False)-bundle.evaluate(y,False,include_priors=False))/(2*h)
        col=j[:,c].toarray().ravel();checks.append(dict(column=c,relative_l2=float(np.linalg.norm(fd-col)/max(1.,np.linalg.norm(fd))),maximum_absolute=float(np.max(np.abs(fd-col)))))
    if max(c['relative_l2'] for c in checks)>2e-4:raise ValueError('New actual source derivative replay failed')
    before=residuals(bundle,q0,p,rows)
    manifest=dict(schema='rinkspace.z190-main-ad2-single-source-check.v1',not_programme_camera=True,
        source_native_pts=[r['source_native_pts'] for r in added],canonical_pts=p,
        sole_completed_warm=ref(a.current/'physical_main_soft_height_candidate_unaccepted.npz'),
        sole_completed_warm_state=ref(a.current/'optimized_state6rf.npz'),new_direct_source_packet=ref(a.new_source),
        hard_scene=ref(g.scene_path),shared_calibration_frozen=True,saved_height_scene_units=g.calibration.pivot_xyz[2],
        saved_height_m=g.calibration.pivot_xyz[2]*0.31353729994266055,
        original_corner_points=sum(len(r['uv']) for r in original),new_corner_points=sum(len(r['uv']) for r in added),
        original_fit_rows_unchanged=True,original_sigmas_unchanged=True,held_pixels_used=False,future_exposure_hits=future_hits,
        existing_roll_prior_retained=True,prior_wide_focal_px=g.manifest['frozen_wide_prior_focal_px'],
        diagnostic_limits=['Only one exposure six observed advertisement corners; not entire-cut validation',
            'New sigma3 is original policy, not measured precision; horizontal +/-4px/vertical +/-2px and9px source blur retained',
            'No source library, geometry, position, height, full programme state or production pointer changed'])
    (a.output/'input_manifest.json').write_text(json.dumps(jsonable(manifest),ensure_ascii=False,indent=2),encoding='utf8')
    q,history,seconds=solve_frozen_pivot(bundle,q0,a.output)
    after=residuals(bundle,q,p,rows)
    report=dict(manifest,source_only_derivative_replay=checks,shared_state_bit_exact=np.array_equal(q[:6],q0[:6]),
        old_corner_residuals=before,new_corner_residuals=after,
        effective_focal_px_before=np.exp(q0[9]),effective_focal_px_after=np.exp(q[9]),
        rvec_delta=q[6:9]-q0[6:9],seconds=seconds,history=history,
        new_source_uncertainty=[r['source_uncertainty'] for r in added],
        unchanged_ice_or_motion_rows_used=False,source_matching_model_accuracy_accepted=False)
    np.savez_compressed(a.output/'single_exposure_diagnostic_state6rf.npz',q=q,q_warm=q0)
    (a.output/'report.json').write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage='single_source_check_complete',seconds=seconds,shared_state_bit_exact=report['shared_state_bit_exact'],before=before,after=after))))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=DEFAULT_CURRENT)
    p.add_argument('--new-source',type=Path,default=CURRENT/'z190_main_ad2_right_source_audit_v1/ad2_right_external_229194240_source_candidate.json')
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_ad2_single_source_check_v1')
    run(p.parse_args())
