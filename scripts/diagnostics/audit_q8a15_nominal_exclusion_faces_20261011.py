"""Classify all saved nominal-exclusion rows; no LP, fit or source graph."""
from pathlib import Path
import copy,time,json
import audit_q8a15_complete_truth_phase_I_20261011 as common
import numpy as np
from scipy.sparse import csr_matrix
from structured_programme_source_candidate.human_constraint_dependency_fd import group_dependencies
from calibration.solver_runtime.finite_rectangle_station_step_constraints import (
    finite_rectangle_station_constraints,_current_boundary_clipped_primitive)
from calibration.solver_runtime.finite_rectangle_projection_fd import projected_station_contexts
from structured_programme_source_candidate.native_camera_local import native_camera_local
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown

def main():
    began=time.perf_counter();a=common.a;dest=common.DEST
    assembly=a.read(dest/'complete_domain_assembly.json');options=a.read(common.CONFIG)['source_options']
    with np.load(dest/'original_complete_truth_domain.npz',allow_pickle=False)as z:
        owned=z['row_group_index'];b=z['original_constraint_lower'];v=z['current_group_violations'];q=z['q'].copy()
        expansion=z['original_full_expansion'];rhs=z['original_full_rhs'];selected=z['selected_evaluation_columns'];selected_c=z['selected_canonical_columns'];fixed_c=z['declared_fixed_canonical_columns']
        A=csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=z['A_shape']);norm=z['reversible_row_norm']
    excluded=np.flatnonzero(rhs-expansion>0.)
    native_bad=np.unique(owned[excluded[excluded<len(b)]])
    assert native_bad.tolist()==[9,10,11,12,42,43]
    common.install_startup_source_auth_reuse_runtime(report_directory=None)
    with common.SourceAuthenticationScope(dest):
        state,loaded,identity,error=a.load_state(common.DIRECTORY,Path(options['--world-init']),a.read(common.RUN/'input_manifest.json'))
        assert error==0.;np.testing.assert_array_equal(q,loaded)
        board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
        human=a.bind_human(state,board,ice,options,len(q))
    local=copy.copy(human);local.records=[human.records[g]for g in native_bad];local.full_stroke_records=[human.full_stroke_records[g]for g in native_bad]
    local.blocks={r['row']['actual_pts']:human.blocks[r['row']['actual_pts']]for r in local.records}
    local.active=np.unique(np.concatenate(group_dependencies(local,local.full_stroke_records)))
    fixed=state.canonical_parameter_layout.canonical_to_evaluation[fixed_c]
    caps=np.array([assembly['all86_groups'][g]['original_cap_px']for g in native_bad])
    result=finite_rectangle_station_constraints(local,q,selected,fixed,full_stroke=True,group_limits=caps,return_step_carrier_response=True)
    response=result['step_carrier_response'];local_group=response['row_group_index'];projection=response['row_carrier_projection']
    query=local.queries(q,full_stroke=True);primary=local.queries(q)
    provenance=result['report']['finite_rectangle_station_local_model']['face_provenance']
    first=len(result['lower'])-len(provenance)
    local_original=result['lower']+response['row_limit_expansion']
    checked=[];all_face_rows=[];group_summary=[];witness=None
    for i,g in enumerate(native_bad):
        saved_rows=np.flatnonzero(owned==g);local_rows=np.flatnonzero(local_group==i)
        assert len(saved_rows)==len(local_rows)
        np.testing.assert_array_equal(b[saved_rows],local_original[local_rows])
        max_J=float(abs(A[saved_rows].toarray()-result['matrix'][local_rows]).max())
        assert max_J==0.,max_J
        saved_to_local=dict(zip(saved_rows.tolist(),local_rows.tolist()))
        bad=np.intersect1d(saved_rows,excluded);record=local.full_stroke_records[i];world=_current_boundary_clipped_primitive(record,q)
        assert world is not None
        block=local.blocks[record['row']['actual_pts']]
        camera=block.bundle.camera(native_camera_local(block,q[:state.structured_programme_coordinate_chart.core_count]),record['row']['actual_pts'])
        pixels,depth,central=project_central_valid_full_brown(camera,world)
        actual=max(float(abs(query[i]['query']['residual']).max()),float(abs(primary[i]['query']['residual']).max()))
        assert actual==v[g]
        types={}
        for row in bad:
            localrow=saved_to_local[int(row)];assert localrow>=first
            prov=provenance[localrow-first];types[prov['kind']]=types.get(prov['kind'],0)+1
            vertex=prov['source_vertex'];qq=query[i]['query']
            item=dict(global_original_row=int(row),native_group=int(g),group_id=assembly['all86_groups'][g]['id'],
                source_actual_pts=int(record['row']['actual_pts']),source_vertex=int(vertex),source_UV=record['uv'][vertex].tolist(),
                face=prov,original_lower_target=float(b[row]),original_group_violation=float(v[g]),
                original_cap=float(caps[i]),original_t_coefficient=float(expansion[row]),
                nominal_t1_zero_step_violation=float(rhs[row]-expansion[row]),
                normalized_nominal_t1_zero_step_violation=float((rhs[row]-expansion[row])/norm[row]),
                original_actual_query_vertex_violation=qq['residual'].reshape(-1,2)[vertex].tolist(),
                original_normal=record['normals'][vertex].tolist(),normal_bounds=record['bounds'][vertex].tolist(),
                tangent_bounds=record['tangent_bounds'][vertex].tolist(),
                original_actual_branch=qq['source_support_branch_signature'][vertex].tolist(),
                original_physical_station=float(qq['finite_primitive_station'][vertex]))
            all_face_rows.append(item)
        worst=int(bad[np.argmax(rhs[bad]-expansion[bad])]);worstface=next(x for x in all_face_rows if x['global_original_row']==worst)
        group_summary.append(dict(native_group=int(g),id=record['row']['id'],curve=record['row']['curve'],guard_vertices=len(record['uv']),
            original_actual_group_violation=actual,original_cap=float(caps[i]),exclusion_rows=len(bad),face_types=types,
            maximum_nominal_exclusion=float((rhs[bad]-expansion[bad]).max()),worst=worstface,
            full_saved_row_matrix_and_targets_reproduced_bitexact=True,world_samples=len(world),
            true_world_endpoints=world[[0,-1]].tolist(),true_projected_endpoints=pixels[[0,-1]].tolist(),
            true_endpoint_depth=depth[[0,-1]].tolist(),true_endpoint_central=central[[0,-1]].tolist()))
        if g==12:
            vertex=worstface['source_vertex'];context=next(c for c in response['finite_station_contexts']if c['group']==i and c['source_vertex']==vertex)
            projected=projected_station_contexts(local,q,[context],selected)
            foot=projected['projected_foot_xy'][0];slope=projected['station_screen_tangent'][0]
            station=float(projected['nominal_physical_station'][0]);ends=projected['physical_endpoints_xy'][0]
            local_affine_endpoints=foot[None,:]+(np.array([0.,1.])-station)[:,None]*slope[None,:]
            n=record['normals'][vertex];t=np.array([-n[1],n[0]]);axes=np.stack((n,t));uv=record['uv'][vertex]
            witness=dict(worst=worstface,world=world.tolist(),projected_world=pixels.tolist(),projected_central=central.tolist(),depth=depth.tolist(),
                nominal_selected_cell=int(qq['nearest_finite_segment_index'][vertex]),
                nominal_actual_query_carrier=[float(qq['signed_normal_px'][vertex]),float(qq['coverage_tangent_px'][vertex])],
                fixed_physical_station=station,projected_foot=foot.tolist(),nominal_station_screen_tangent=slope.tolist(),
                true_endpoint_projected=ends.tolist(),locally_affine_station_endpoint_projected=local_affine_endpoints.tolist(),
                true_minus_local_affine_endpoints=(ends-local_affine_endpoints).tolist(),
                true_endpoints_in_source_normal_tangent=((ends-uv)@axes.T).tolist(),
                local_affine_endpoints_in_source_normal_tangent=((local_affine_endpoints-uv)@axes.T).tolist(),
                actual_original_support_query_residual_px=query[i]['query']['residual'].tolist(),
                nominal_local_station_face_projection_indices=projection.getrow(saved_to_local[worst]).indices.tolist(),
                nominal_local_station_face_projection_coefficients=projection.getrow(saved_to_local[worst]).data.tolist())
    assert len(all_face_rows)==len(excluded)==5282
    list_path=dest/'all_5282_nominal_exclusion_faces.jsonl'
    with list_path.open('w',encoding='utf8')as stream:
        for row in all_face_rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    witness_path=dest/'native12_B1_true_caps_vs_nominal_station_witness.json';common.save(witness_path,witness)
    physical_prefix=assembly['physical_prefix_rows']
    report=dict(schema='rinkspace.all-saved-q8a15-nominal-local-model-exclusion-causal-audit.v1',script=common.ref(__file__),
        original_domain_assembly=common.ref(dest/'complete_domain_assembly.json'),original_domain_arrays=common.ref(dest/'original_complete_truth_domain.npz'),
        failed_LP_evidence=common.ref(dest/'complete_truth_phase_I_causal_witness.json'),complete_q_sha256=common.Q,
        t1_delta0_is_an_original_nonlinear_group_caps_and_physical_domain_witness=True,
        new_local_station_model_incorrectly_excludes_that_same_nominal_q=True,
        all_exclusion_rows=len(excluded),all_exclusions_resolvable_above_existing_1e_minus8=True,exclusion_faces=common.ref(list_path),
        physical_prefix_rows=physical_prefix,physical_prefix_nominal_maximum_deficit=float(rhs[:physical_prefix].max()),
        original_mount_box_rows=len(rhs)-len(b),original_mount_box_nominal_maximum_deficit=float(rhs[len(b):].max()),
        all_physical_prefix_and_box_zero_step_rows_feasible=bool(np.all(rhs[:physical_prefix]<=0)and np.all(rhs[len(b):]<=0)),
        six_group_original_guard_and_saved_local_model_replay=group_summary,
        native12_B1_world_Brown_projection_and_true_cap_witness=common.ref(witness_path),
        implementation_sources=common.station_model_implementation_sources(),
        production_source_limits_weights_support_or_parameters_changed=False,new_fit_J_full_normal_or_LP=False,source_images_written=0,
        physical_model_incompatibility_claimed=False,
        causal_finding='Physical-world straightness does not certify a globally affine Brown projection. Replacing finite bounds of a local screen-tangent FM station chart with actual distant endpoint projections can remove the nominal original support witness.',
        repair_not_applied='A consistent local station/rectangle witness is required; never cure this by dropping arbitrary active supports, changing original endpoints, allowing global bias, or loosening zero caps.',
        seconds=time.perf_counter()-began)
    report_path=dest/'all86_nominal_exclusion_causal_evidence.json';common.save(report_path,report)
    common.emit(stage='all5282_nominal_exclusions_classified',report=str(report_path),sha256=a.sha(report_path),
        types=[dict(native_group=g['native_group'],face_types=g['face_types'])for g in group_summary],native12_B1_witness=str(witness_path))

if __name__=='__main__':main()
