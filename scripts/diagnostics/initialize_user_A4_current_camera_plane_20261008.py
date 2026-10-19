"""A source-bound A4 circle initializer on the existing current physical camera.

Inferred plane coordinates are an initializer, never observed world truth or
verified metres. The complete q and all previous source facts are preserved.
"""
from pathlib import Path
import argparse,hashlib,json,pickle,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.optimize import least_squares
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_camera_local import native_camera_local
from structured_programme_source_candidate.persistent_ice_long_baseline_source import current_native_rays
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor,_axes
from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit


def main():
    ap=argparse.ArgumentParser()
    for key in ('state','camera-identity','owner-runtime-report','ice-packet','output','report'):
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    arrays=dict(np.load(args.state,allow_pickle=False));identity=json.loads(args.camera_identity.read_text(encoding='utf8'))
    parent_arrays=dict(np.load(args.camera_identity.parent/'complete_state_UNACCEPTED.npz',allow_pickle=False))
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False));owner=json.loads(args.owner_runtime_report.read_text(encoding='utf8'))['material_owner']
    install_mappingproxy_pickle();install_geometry_unpickle();provider=pickle.loads(Path(owner['path']).read_bytes())
    state,_=state_camera_adapter(parent_arrays,cameras,provider);layout=state.canonical_parameter_layout
    q=state.evaluation_q(arrays['q']);world_indices=state.geometry_parameter_indices
    ice=NativeIceCurveFactor(state,args.ice_packet,len(q))
    row=next(row for row in ice.rows if row['id']=='user_native230024192_A4_visible_arc_20261008')
    block=state.bypts[row['actual_pts']]
    camera=block.bundle.camera(native_camera_local(block,q[:state.structured_programme_coordinate_chart.core_count]),row['actual_pts'])
    rays=current_native_rays(camera,row['uv']);distance=-camera[2]/rays[:,2]
    if not np.isfinite(distance).all()or np.any(distance<=0.):raise ValueError('Actual A4 source rays do not intersect forward ice plane')
    xyz=(camera[:3]+distance[:,None]*rays)*provider.scene_unit_m;xy=xyz[:,:2]
    datum=xy.mean(axis=0);local=xy-datum
    solution=np.linalg.lstsq(np.c_[2*local,np.ones(len(local))],np.einsum('ij,ij->i',local,local),rcond=None)[0]
    centre0=solution[:2]+datum;radius0=np.sqrt(solution[2]+solution[:2]@solution[:2])
    fit=least_squares(lambda value:np.linalg.norm(xy-value[:2],axis=1)-value[2],np.r_[centre0,radius0],
        bounds=([-np.inf,-np.inf,np.finfo(float).eps],[np.inf,np.inf,np.inf]))
    centre=np.r_[fit.x[:2],0.];radius=float(fit.x[2]);physical=provider.current.effective_world(q[world_indices])
    axis,across=_axes(physical[2]);course,cq,_=provider.current.print_course_binding(physical)
    bottom=course.sample_perimeter(cq,with_jacobian=False)[0]
    wall=_boundary_hit(np.r_[physical[:2],0.]/provider.scene_unit_m,-axis,np.asarray(bottom))*provider.scene_unit_m
    goal=wall+axis*physical[7]
    trial=q.copy();candidate=trial[world_indices].copy()
    candidate[3]=radius;candidate[4]=(centre-goal)@axis;candidate[5]=-(centre-goal)@across
    trial[world_indices]=candidate
    domain_error=None
    try:provider.validate_parameters(candidate)
    except(ValueError,RuntimeError)as exc:domain_error=str(exc)
    report=dict(schema='rinkspace.actual-A4-current-camera-plane-circle-initializer.v1',
        source_pts=row['actual_pts'],source_UV_count=len(row['uv']),source_UV_and_identity_not_changed=True,
        camera_from_complete_q_sha256=hashlib.sha256(arrays['q'].tobytes()).hexdigest(),
        inferred_plane_circle_center_working_m=centre.tolist(),inferred_plane_circle_radius_working_m=radius,
        inferred_circle_maximum_plane_residual_working_m=float(abs(fit.fun).max()),
        original_p3_p4_p5=q[world_indices[[3,4,5]]].tolist(),proposed_p3_p4_p5=candidate[[3,4,5]].tolist(),
        actual_physical_domain_error=domain_error,world_coordinates_not_measurements=True,
        working_metres_not_verified_metric_scale=True,all_parameters_jointly_free_after_initializer=True,
        old_manual_and_real_source_guard_revalidation_required=True,accuracy_accepted=False)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if domain_error is None:
        arrays['q']=layout.canonical_q(trial)
        ident=json.loads(str(arrays['identity_json'].item()));ident['direct_A4_plane_circle_initializer']=str(args.report.resolve());arrays['identity_json']=np.asarray(json.dumps(ident))
        np.savez_compressed(args.output,**arrays)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
