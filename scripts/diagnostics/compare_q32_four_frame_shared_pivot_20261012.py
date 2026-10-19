"""One existing shared-pivot versus fixed-pivot camera condition comparison."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse, json, time, hashlib
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
import q32_frozen_camera_source_context_20261012 as context
from installation_mount_candidate.mount_graph import _pivot_course, _pivot_outward_margin

s=context.s;np=s.np
DEFAULT_PTS=[229702144,230571776,230717696,231678464]
OUTPUT=s.DIAG/'q32_four_existing_frames_fixed_pivot_shared_pivot_condition_comparison_20261012.json'

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pts',nargs='+',type=int,default=DEFAULT_PTS)
    ap.add_argument('--output',type=Path,default=OUTPUT);args=ap.parse_args()
    started=time.perf_counter();c=context.prepare(args.pts);loss=context.loss(c)
    contract=c.manifest['camera_pivot_exterior_constraints'];facts=contract['facts']
    assert s.sha(facts['path'])==facts['sha256']
    # The frozen outline comes from the original exact exterior-domain owner.
    polygon,area=_pivot_course(SimpleNamespace(geometry=c.provider),c.q,contract)
    unit=float(contract['scene_unit_working_m']);p0=np.asarray(c.calibration.pivot_xyz)
    slot=np.asarray(contract['by_installation'][c.installation_id]['pivot_parameter_indices'],int)
    assert np.allclose(c.q[slot]/unit,p0,rtol=1e-13,atol=1e-13)
    def pivot_domain(pivot):
        d=_pivot_outward_margin(polygon,area,np.asarray(pivot)[:2]*unit)
        d.update(pivot_scene_units=np.asarray(pivot).tolist(),pivot_Z_positive=bool(pivot[2]>0.),
            working_metres_not_verified_metric=True)
        d['valid']=bool(d['valid']and d['pivot_Z_positive']);return d
    initial_domain=pivot_domain(p0);assert initial_domain['valid']
    frames=list(c.selected_pts);position={at:i for i,at in enumerate(frames)}
    base_values=context.residual(c,c.baseline_cameras);base_cost=float(.5*loss(base_values**2)[0].sum())
    base_rows=context.metrics(c,c.baseline_cameras);assert np.isclose(sum(r['weighted_current_cost']for r in base_rows),base_cost,rtol=1e-12,atol=1e-12)
    result=dict(schema='rinkspace.fixed-world-existing-shared-pivot-camera-condition-comparison.v1',q_sha256=s.QID,actual_pts=frames,
        installation_id=c.installation_id,config=s.ref(s.CONFIG),input_manifest=s.ref(s.RUN/'input_manifest.json'),
        state_identity=s.ref(c.directory/'state_identity.json'),camera=s.ref(c.identity['camera_path']),script=s.ref(__file__),
        context_script=s.ref(context.__file__),single_frame_source_extraction=s.ref(s.__file__),
        source_RGB_receipt=s.ref(s.DIAG/'q32_shared_pivot_four_existing_source_cache_20261012.json'),
        world_provenance={k:v for k,v in c.world_proof.items()if k!='explicit_saved_physical_geometry_initializer'},
        original_operator_authorities=c.call_proof,native_packets=[s.ref(c.options['--additional-ice-source']),s.ref(c.options['--additional-board-source'])],
        original_source_packet=dict(path=str(s.PACKET),sha256=s.PACKET_SHA256),source_row_count=len(c.records),
        UV_count=sum(len(r['row']['uv'])for r in c.records),source_scalar_count=len(base_values),
        exact_production_scalar_loss_masks=[dict(id=r['source_id'],actual_pts=r['actual_pts'],saved_global_scalar_range=r['saved_global_scalar_range'],
            quadratic_mask=r['quadratic'],active_original_point_indices=r['original_active_point_indices'])for r in c.records],
        fixed_world_PP_Brown_lever_axial_aspect=True,existing_shared_pivot_XYZ_only_not_free_C_per_frame=True,
        original_pivot_exterior_authority=facts,domain_implementation=s.ref(Path(__import__('installation_mount_candidate.mount_graph',fromlist=['_pivot_course']).__file__)),
        omitted_full_programme_motion_priors_and_human_guards=True,soft_optical_height_prior_not_in_this_absolute_source_condition_target=True,
        baseline=dict(current_cost=base_cost,rows=base_rows,pivot_scene_units=p0,pivot_domain=initial_domain,
            cameras=[dict(actual_pts=at,camera25=c.baseline_cameras[at])for at in frames]),modes={},prepare_seconds=c.prepare_seconds,
        no_original_RGB_decode_or_new_raw_image=True,no_complete_motion_source_graph_J_or_profiles=True,no_production_candidate_written=True,
        other_cameras_reconstructed=False,scope='Conditional compatibility of these four current FIT exposures on a frozen world. Local convergence is not a proof of uniqueness, global feasibility/infeasibility, full-programme accuracy, or correct shared installation.',accuracy_accepted=False)
    print(json.dumps(dict(stage='four_frame_actual_source_prepared',rows=len(c.records),points=result['UV_count'],cost=base_cost,
        seconds=c.prepare_seconds,installation=c.installation_id)),flush=True)
    for mode in ['A_fixed_shared_pivot_each_RF4','B_existing_shared_pivot3_each_RF4']:
        t=time.perf_counter();shared=mode.startswith('B_');prefix=3 if shared else 0
        initial=np.r_[p0/100.,*[c.views[at]for at in frames]]if shared else np.concatenate([c.views[at]for at in frames])
        lower=np.full(len(initial),-np.inf);upper=np.full(len(initial),np.inf)
        if shared:lower[2]=0.
        for i,at in enumerate(frames):
            # Original production focal domain 100..50000px, translated to
            # the retained domain-logf ViewState coordinate without new limits.
            bias=c.views[at][3]-float(np.mean(c.baseline_cameras[at][6:8]))
            lower[prefix+4*i+3]=np.log(100.)+bias;upper[prefix+4*i+3]=np.log(50000.)+bias
        sparsity=lil_matrix((len(base_values),len(initial)),dtype=int);cursor=0
        for r in c.records:
            size=2*len(r['row']['uv']);lo=prefix+4*position[r['actual_pts']]
            sparsity[cursor:cursor+size,lo:lo+4]=1
            if shared:sparsity[cursor:cursor+size,:3]=1
            cursor+=size
        assert cursor==len(base_values)
        value_calls=0;invalid_trials=[];last_legal=None
        def cameras(p):
            nonlocal value_calls,last_legal
            value_calls+=1;pivot=np.asarray(p[:3])*100. if shared else p0
            domain=pivot_domain(pivot)
            if not domain['valid']:
                invalid_trials.append(dict(parameters=np.asarray(p).tolist(),pivot_domain=domain))
                raise ValueError('Original shared-pivot positive-Z/outside-or-boundary domain failed')
            calibration=replace(c.calibration,pivot_xyz=tuple(pivot))if shared else c.calibration
            out={at:calibration.camera25(s.ViewState.from_state4(p[prefix+4*i:prefix+4*i+4],effective_pp_offset_px=c.pp_offsets[at]))for i,at in enumerate(frames)}
            last_legal=np.asarray(p).copy();return out
        def residual(p):return context.residual(c,cameras(p))
        record=dict(parameter_count=len(initial),shared_pivot_parameter_count=prefix,per_exposure_RF_parameter_count=4,
            solver='scipy least_squares TRF exact dense linear subproblem; default natural termination/evaluation allowance',
            differentiation='3-point actual source values grouped by exact RF4/shared-pivot sparsity, then J.toarray for direct small solve',
            domain='Original positive pivot Z and exact current-course outside/on boundary; illegal trial raises, no finite proxy residual',
            x_scale='jac',source_and_initial_states_identical=True)
        try:
            def jacobian(p):
                value=residual(p)
                return approx_derivative(residual,p,method='3-point',f0=value,bounds=(lower,upper),sparsity=sparsity.tocsr()).toarray()
            solved=least_squares(residual,initial,jac=jacobian,loss=loss,x_scale='jac',bounds=(lower,upper),tr_solver='exact')
            final_cameras=cameras(solved.x);rows=context.metrics(c,final_cameras);cost=float(sum(r['weighted_current_cost']for r in rows))
            assert np.isclose(cost,float(solved.cost),rtol=1e-12,atol=1e-12)
            pivot=np.asarray(solved.x[:3])*100. if shared else p0
            singular=np.linalg.svd(solved.jac.toarray()if hasattr(solved.jac,'toarray')else solved.jac,compute_uv=False)
            record.update(success=bool(solved.success),status=int(solved.status),termination=solved.message,nfev=int(solved.nfev),njev=int(solved.njev),
                current_cost=cost,optimality=float(solved.optimality),pivot_scene_units=pivot,pivot_delta_scene_units=pivot-p0,
                pivot_domain=pivot_domain(pivot),diagnostic_chart_values=solved.x,rows=rows,final_robust_weighted_singular_values=singular,
                cameras=[dict(actual_pts=at,camera25=final_cameras[at],optical_C_delta_scene_units=final_cameras[at][:3]-c.baseline_cameras[at][:3],
                    logf_delta=float(np.mean(final_cameras[at][6:8]-c.baseline_cameras[at][6:8])),
                    fixed_PP_Brown_aspect=bool(np.array_equal(final_cameras[at][8:],c.baseline_cameras[at][8:])),
                    optical_C_above_ice=bool(final_cameras[at][2]>0.))for at in frames],
                same_frame_actual_source_queries_finite=True,complete_programme_physical_guards_evaluated=False)
        except Exception as error:
            record.update(success=False,error_type=type(error).__name__,error=str(error),no_generic_finite_residual_fallback=True,
                retry_performed=False,last_legal_chart_values=last_legal,comparison_not_completed=True)
        record.update(value_calls=value_calls,invalid_trials=invalid_trials,seconds=time.perf_counter()-t)
        result['modes'][mode]=record;args.output.write_text(json.dumps(result,default=s.plain,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(mode=mode,success=record['success'],cost=record.get('current_cost'),termination=record.get('termination',record.get('error')),
            value_calls=value_calls,seconds=record['seconds'])),flush=True)
    result['total_seconds']=time.perf_counter()-started;args.output.write_text(json.dumps(result,default=s.plain,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(args.output),sha256=s.sha(args.output),bytes=args.output.stat().st_size,total_seconds=result['total_seconds'])),flush=True)

if __name__=='__main__':main()
