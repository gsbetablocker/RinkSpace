"""Small readonly MAIN1 camera/scene identifiability comparison.

Uses the original FIT perimeter rows and retains every original HELD role.
Local optical variations are hypotheses, never a new checkpoint or target.
No programme graph or motion source is reconstructed.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, io, json, os, pickle, sys, time
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/diagnostics')]
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from structured_programme_source_candidate.original_review_source import PACKET, PACKET_SHA256, observation_signature
from verify_current_native_primitives_20261007 import state_camera_adapter
from z190_source_camera_diagnostic import absolute_value_jac15, source_errors, project_central_valid


class NumericDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy', 'dtype'): return np.dtype
        if (module, name) in (('numpy._core.numeric', '_frombuffer'), ('numpy.core.numeric', '_frombuffer')):
            return np._core.numeric._frombuffer
        raise pickle.UnpicklingError('Numeric source dictionaries only: '+module+'.'+name)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_summary(camera, rows):
    return source_errors(camera, [dict(row, split=row.get('split')) for row in rows])


def normalized_summary(camera, rows):
    r = absolute_value_jac15(camera, rows, False)
    sigma = np.concatenate([np.repeat(row['sigma'], 2) for row in rows])
    d = np.linalg.norm((r*sigma).reshape(-1, 2), axis=1)
    return dict(normalized_rms=float(np.sqrt(np.mean(r*r))), normalized_sum_squares=float(r@r),
                supported_native_pixel_mean=float(d.mean()), supported_native_pixel_p95=float(np.quantile(d, .95)),
                native_UV_points=len(d))


def central_J(fun, x, h=1e-3):
    columns=[]
    for k in range(len(x)):
        d=np.zeros(len(x));d[k]=h
        columns.append((fun(x+d)-fun(x-d))/(2*h))
    return np.array(columns).T


def main():
    ap=argparse.ArgumentParser()
    for name in ('state', 'camera', 'auxiliary', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args=ap.parse_args();start=time.perf_counter()
    arrays=dict(np.load(args.state, allow_pickle=False));cameras=dict(np.load(args.camera, allow_pickle=False))
    provider, world, proof=current_complete_source_geometry(args.state, args.auxiliary)
    install_source_ice_material_view(provider)
    state, replay=state_camera_adapter(arrays, cameras, provider);q=state.evaluation_q(arrays['q'])
    assert sha(PACKET)==PACKET_SHA256
    allrows=NumericDataUnpickler(io.BytesIO(PACKET.read_bytes())).load()['programme_absolute_rows']
    rows=[item for item in allrows if item['block']=='main1']
    native_users=(229256192, 229290240, 229476864)
    user_actual=[state.original.aliases[p] for p in native_users]
    # Exact source exposures near the user images, plus every existing MAIN1
    # ice HELD frame. The HELD measurements remain development/pending checks.
    actuals=sorted(set(user_actual+[int(item['group_actual_pts']) for item in rows if item['collection_role']=='HELD']))
    selected=[item for item in rows if int(item['group_actual_pts']) in actuals]
    for item in selected:
        provider.register_row(item['source_row'], actual_pts=int(item['group_actual_pts']),
                              allow_previously_qualified_held_ad2_representation=True)
    signatures=[observation_signature(item['source_row']) for item in selected]
    features={f['id']:f for f in provider.scene(world)['ice_features']}
    markers={}
    ad,_j,_contract=provider.current.material(4, [0., 0., 1., 1.], [0., 1., 0., 1.], world)
    markers['Ad4_original_print_four_corners']=np.asarray(ad)
    for name in ('B1', 'C0', 'F1', 'F2', 'F3', 'F4'):
        f=features[name]
        if 'segments_xyz' in f:markers[name]=np.asarray(f['segments_xyz'])[0]
        else:
            t=np.linspace(0., 2*np.pi, 33)
            markers[name]=np.asarray(f['center_xyz'])+float(f['radius'])*np.c_[np.cos(t), np.sin(t), np.zeros(len(t))]
    # Identical 0.001 rad / logf increments make the SVD units explicit.
    unit=np.full(4, .001)
    cases=[]
    optic_tests=[('nominal', [0.,0.,0.]), ('PPx_plus100px', [100.,0.,0.]),
                 ('PPx_minus100px', [-100.,0.,0.]), ('PPy_plus100px', [0.,100.,0.]),
                 ('PPy_minus100px', [0.,-100.,0.]), ('K1_plus0.1', [0.,0.,.1]), ('K1_minus0.1', [0.,0.,-.1])]
    for actual in actuals:
        entries=[item for item in selected if int(item['group_actual_pts'])==actual]
        fits=[provider.rebind(item['source_row'], world) for item in entries if item['collection_role']=='FIT']
        held=[provider.rebind(item['source_row'], world) for item in entries if item['collection_role']=='HELD']
        owner=state.bypts[actual];view_indices=owner.start+4*owner.bundle.index[actual]+np.arange(4)
        nominal_camera=owner.bundle.camera(q, actual)
        def camera(x, optical=(0.,0.,0.)):
            trial=q.copy();trial[view_indices]+=unit*x
            result=owner.bundle.camera(trial, actual).copy()
            # PP/K1 hypothesis: same installation pivot and rotating lever,
            # camera R/f remain the existing four per-exposure quantities.
            result[8:10]+=np.asarray(optical[:2]);result[11]+=optical[2]
            return result
        fun=lambda x:absolute_value_jac15(camera(x), fits, False)
        zero=np.zeros(4);r0=fun(zero);J=central_J(fun, zero)
        _u,s,vh=np.linalg.svd(J, full_matrices=False)
        weak=vh[-1]
        projection=lambda c:{name:project_central_valid(c, pts)[0] for name,pts in markers.items()}
        nominal=projection(nominal_camera)
        def displacement(c):
            now=projection(c)
            return {name:dict(dx_mean_px=float((now[name]-nominal[name])[:,0].mean()),
                              dy_mean_px=float((now[name]-nominal[name])[:,1].mean()),
                              maximum_native_displacement_px=float(np.linalg.norm(now[name]-nominal[name],axis=1).max()))
                    for name in markers}
        sensitivity=displacement(camera(weak))
        scale=40./max(sensitivity['Ad4_original_print_four_corners']['maximum_native_displacement_px'],1e-12)
        weak_variants=[]
        for sign in (-1.,1.):
            x=sign*scale*weak;c=camera(x)
            weak_variants.append(dict(sign=sign, delta_R3_logf=(x*unit).tolist(),
                FIT=normalized_summary(c,fits), HELD=pixel_summary(c,held), marker_displacement=displacement(c)))
        optical_results=[]
        # Use ONLY existing FIT observations to restore R/f after each shared
        # optical hypothesis. No HELD source enters this least-squares call.
        for label,optical in optic_tests:
            solve=least_squares(lambda x:absolute_value_jac15(camera(x,optical),fits,False),zero,
                                xtol=1e-11,ftol=1e-11,gtol=1e-11,x_scale='jac')
            c=camera(solve.x,optical)
            optical_results.append(dict(hypothesis=label, PPxy_K1_delta=optical,
                only_existing_FIT_rows_consumed=True, success=bool(solve.success), status=int(solve.status),
                delta_R3_logf=(unit*solve.x).tolist(), objective=float(solve.cost), evaluations=int(solve.nfev),
                FIT=normalized_summary(c,fits), HELD=pixel_summary(c,held), marker_displacement=displacement(c)))
        # Geometric directions are physical logical columns, not camera slots.
        geometry_sensitivity=[]
        for logical in (2,6,12,13,14,15,16,17,44,45,46,47,48,49,54,55,56,57):
            h=1e-6;minus=world.copy();plus=world.copy();minus[logical]-=h;plus[logical]+=h
            legal=[]
            for sign,w in ((-1,minus),(1,plus)):
                try:provider.validate_parameters(w);legal.append(sign)
                except (ValueError,RuntimeError):pass
            # Real legal side on print/ice physical boundaries; do not move q.
            if len(legal)==2:
                fp=[provider.rebind(item['source_row'],plus)for item in entries if item['collection_role']=='FIT']
                fm=[provider.rebind(item['source_row'],minus)for item in entries if item['collection_role']=='FIT']
                jd=(absolute_value_jac15(nominal_camera,fp,False)-absolute_value_jac15(nominal_camera,fm,False))/(2*h)
            elif legal:
                w=plus if legal[0]==1 else minus
                fw=[provider.rebind(item['source_row'],w)for item in entries if item['collection_role']=='FIT']
                jd=(absolute_value_jac15(nominal_camera,fw,False)-r0)/(legal[0]*h)
            else:continue
            he=[]
            for item in [item for item in entries if item['collection_role']=='HELD']:
                row=provider.rebind(item['source_row'],world);r=absolute_value_jac15(nominal_camera,[row],False)
                w=plus if 1 in legal else minus;sgn=1 if 1 in legal else -1
                rr=absolute_value_jac15(nominal_camera,[provider.rebind(item['source_row'],w)],False)
                d=(rr-r)/(sgn*h);dpx=d*np.repeat(row['sigma'],2)
                he.append(dict(id=row['id'],split=row.get('split'),normalized_r_derivative_norm=float(np.linalg.norm(d)),
                    native_residual_derivative_component_max=float(abs(dpx).max()),
                    least_squares_linear_single_column_delta=float(-d@r/(d@d)) if d@d>1e-16 else None))
            geometry_sensitivity.append(dict(logical_world_index=logical,parameter_name=provider.parameter_names[logical],
                real_legal_sides=legal,FIT_normalized_derivative_norm=float(np.linalg.norm(jd)),HELD=he))
        cases.append(dict(actual_pts=actual,user_native_pts=[p for p,a in zip(native_users,user_actual)if a==actual],
            focal_px=float(np.exp(nominal_camera[6])), FIT_row_ids=[r['id']for r in fits],
            FIT=normalized_summary(nominal_camera,fits), HELD=pixel_summary(nominal_camera,held),
            singular_values_per0_001_R3_logf=s.tolist(),camera4_condition=float(s[0]/s[-1])if s[-1]>0 else None,
            weakest_direction_R3_logf_per0_001=weak.tolist(),
            weak_camera40px_Ad4_markers=weak_variants,optical_hypotheses_FIT_only_refit=optical_results,
            geometry_sensitivity=geometry_sensitivity))
        print(json.dumps(dict(actual_pts=actual,FIT_points=sum(len(r['uv'])for r in fits),HELD_ids=[r['id']for r in held],
                              camera4_singular_values=s.tolist(),condition=float(s[0]/s[-1]))),flush=True)
    assert all(observation_signature(item['source_row'])==signature for item,signature in zip(selected,signatures))
    report=dict(schema='rinkspace.readonly-main1-camera-versus-scene-AB.v1',state=str(args.state),state_sha256=sha(args.state),
        complete_q_sha256=hashlib.sha256(arrays['q'].tobytes()).hexdigest(),camera=str(args.camera),camera_sha256=sha(args.camera),
        physical_camera_replay_max=replay,source_packet=str(PACKET),source_packet_sha256=PACKET_SHA256,
        full_source_MAIN1_role_counts=dict(Counter(item['collection_role']for item in rows)),
        FIT_normal_objective_and_original_sigma_preserved=True,HELD_not_consumed_by_refits=True,
        pending_HELD_is_not_confirmed_physical_validation=True,optical_hypotheses_not_adopted_as_model=True,
        user_screenshot_arrows_not_pixel_measurements=True,fullgraph_loaded=False,solver_started=False,
        no_new_checkpoint_no_source_edit=True,geometry_proof=proof,cases=cases,elapsed_seconds=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
