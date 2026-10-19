"""Joint local absolute-source projection diagnostic, never a model fit."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import joint_absolute_source_context_20261011 as ctx
a=ctx.a
np=ctx.np


def baseline_report(c,meta,values):
    rows=[]
    for group in c.groups:
        rows.append({k:group[k]for k in ('family','actual_pts','id','physical_group','source_kind','scalar_rows')})
    return dict(schema='rinkspace.joint-absolute-local-sensitivity-baseline.v1',
        script=a.ref(__file__),context_script=a.ref(ctx.__file__),metadata=a.ref(ctx.METADATA),
        state=a.ref(c.path),camera=a.ref(c.identity['camera_path']),config=a.ref(a.CONFIG),
        original_packet=a.ref(a.PACKET),selected_actual_pts=meta['selected_actual_pts'],
        existing_source_image_exact_byte_read_aliases=c.source_auth_aliases,
        complete_q_sha256=a.CHILD,original_camera_archive_reproduction_max_error=c.camera_error,
        canonical_fixed_parameter_indices=c.fixed_canonical.tolist(),evaluation_fixed_parameter_indices=c.fixed_evaluation.tolist(),
        geometry_evaluation_parameter_indices=c.world_indices.tolist(),
        canonical_evaluation_layout=c.state.canonical_parameter_layout.manifest(),
        baseline=values,source_rows=rows,
        selected_programme_family_operator_row_counts=dict(Counter(r['family']for r in rows)),
        global_wide_original_private25_preserved=True,
        global_other_absolute_families_pending_reconstruction=['external_main_to_low','shared_board_patterns'],
        no_fit_no_candidate_q_no_motion_graph_no_full_source_J_no_decode_or_images=True,
        rawtyped_diagnostic_metric='same actual nominal IRLS row metric; raw supportless typed operator, not a changed production objective',
        scope='Selected programme absolute sources plus current global wide; local column spaces are not complete-source or finite-change certificates',
        load_seconds=c.load_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--derivatives-only',action='store_true')
    args=parser.parse_args();started=time.perf_counter()
    meta=a.read(ctx.METADATA);c=ctx.load_context()
    print(json.dumps(dict(stage='small_absolute_context_loaded',seconds=c.load_seconds,
        actual_programme_exposures=len(meta['selected_actual_pts']),fixed_canonical=c.fixed_canonical.tolist())),flush=True)
    values=ctx.baseline(c,set(meta['selected_actual_pts']))
    report=baseline_report(c,meta,values)
    report['seconds']=time.perf_counter()-started
    destination=a.H/'joint_absolute_q8a15_baseline_20261011.json'
    destination.write_text(json.dumps(report,default=a.plain,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage='absolute_baseline_complete',path=str(destination),sha256=a.sha(destination),
        rows=values['groups'],programme_scalar_rows=values['selected_programme_scalar_rows'],seconds=report['seconds'],
        costs=values['checks'])),flush=True)
    if not args.baseline_only:
        import joint_absolute_source_derivatives_20261011 as derivative
        import joint_absolute_analysis_inputs_20261011 as inputs
        import joint_absolute_global_tail_context_20261011 as tails
        import joint_absolute_subspace_diagnostics_20261011 as subspace
        tail_records=tails.prepare_tails(c)
        assert {record['family']for record in tail_records}=={'external_main_to_low','shared_board_patterns','installation_correlated_roll_hardware_bias'}
        report['global_tail_baselines']=[record['baseline_receipt']for record in tail_records]
        report['global_other_absolute_families_pending_reconstruction']=[]
        groups=list(c.groups)+inputs.wide_groups(c)
        for group in groups[len(c.groups):]:
            group.update(numeric_key='wide',numeric_row_slice=[group['wide_start'],group['wide_stop']])
        for record in tail_records:
            family=record['family']
            groups.append(dict(family=family,id=family,physical_group=family,source_kind='global_tail',actual_pts=None,
                scalar_rows=len(record['actual']),actual=record['actual'],raw=record['raw'],scales=record['scales'],
                quadratic=record['quadratic'],loss=record['loss'],numeric_key=family,report_pixel_components=False,
                row_labels=record['row_labels']))
        files=[__file__,ctx.__file__,derivative.__file__,inputs.__file__,tails.__file__,subspace.__file__]
        before_sources=[a.ref(path)for path in files]
        # A complete tail restores small camera hooks/owners. Verify that it
        # preserves all already-authenticated selected programme source values.
        for group in c.groups:
            if group['source_kind']=='original':value=ctx.original_value(c,group['packet_row_index'],c.q)
            elif group['source_kind']=='manual_centre':value=ctx.native_value(c,group['factor'],c.q,[group['row']])/group['mean_scale']
            else:value=ctx.native_value(c,group['factor'],c.q,[group['row']])
            assert np.max(abs(value-group['actual']))<1e-9,(group['family'],group['id'])
        print(json.dumps(dict(stage='complete_selected_absolute_scope_admitted',programme_scalar_rows=values['selected_programme_scalar_rows'],
            total_unique_scalar_rows=sum(group['scalar_rows']for group in groups),operator_groups=len(groups),
            tail_baselines=report['global_tail_baselines'])),flush=True)
        numeric={};receipts=[]
        for multiplier in (1.,.5):
            numeric[multiplier],receipt=derivative.programme_derivatives(c,multiplier)
            receipts.append(receipt)
            print(json.dumps(dict(stage='programme_local_J_complete',step_multiplier=multiplier,seconds=receipt['seconds'])),flush=True)
            columns,J,wide_receipt=derivative.wide_derivatives(c,multiplier)
            numeric[multiplier]['wide']=(columns,J,J.copy());receipts.append(dict(family='strong_high_wide_source',**wide_receipt))
            print(json.dumps(dict(stage='wide_local_J_complete',**wide_receipt)),flush=True)
            tail_numeric=tails.evaluate_tail_derivatives(c,multiplier)
            for family,entry in tail_numeric.items():
                columns,J,rawJ,receipt=(entry['columns'],entry['actualJ'],entry['rawJ'],entry['receipt'])
                numeric[multiplier][family]=(columns,J,rawJ);receipts.append(dict(receipt,family=family))
            print(json.dumps(dict(stage='global_tail_local_J_complete',step_multiplier=multiplier,
                families=list(tail_numeric))),flush=True)
        report['derivative_receipts']=receipts
        report['derivative_script']=a.ref(derivative.__file__)
        report['programme_and_wide_two_scale_numeric_hashes']={str(h):{str(i):dict(columns=entry[0].tolist(),
            actual_J_sha256=a.hashlib.sha256(entry[1].tobytes()).hexdigest(),raw_J_sha256=a.hashlib.sha256(entry[2].tobytes()).hexdigest())
            for i,entry in arrays.items()}for h,arrays in numeric.items()}
        report['seconds']=time.perf_counter()-started
        mapping,free,partitions=inputs.column_mapping(c,numeric[1.])
        report['actual_parameter_mapping']=mapping
        report['free_parameter_partition_counts']={kind:len(cols)for kind,cols in partitions.items()}
        report['parameter_partitions_evaluation_indices']={kind:cols.tolist()for kind,cols in partitions.items()}
        views={}
        for raw,name in ((False,'actual'),(True,'rawtyped')):
            r,w,first,ownership,offsets=inputs.assemble(groups,numeric[1.],partitions,raw=raw)
            r2,w2,half,ownership2,offsets2=inputs.assemble(groups,numeric[.5],partitions,raw=raw)
            assert np.array_equal(r,r2)and np.array_equal(w,w2)and np.array_equal(ownership,ownership2)
            views[name]=(r,w,first,half,ownership)
        common_scale={}
        for kind in partitions:
            maximum=np.maximum.reduce([np.linalg.norm(view[2][kind]*np.sqrt(view[1])[:,None],axis=0)for view in views.values()]+
                [np.linalg.norm(view[3][kind]*np.sqrt(view[1])[:,None],axis=0)for view in views.values()])
            common_scale[kind]=np.divide(1.,maximum,out=np.ones_like(maximum),where=maximum!=0.)
        analyses={}
        roles=inputs.scalar_roles(groups);observation_mask=roles!='engineering_prior'
        report['scalar_role_counts']=dict(Counter(roles.tolist()))
        for name,(r,w,first,half,ownership)in views.items():
            analyses[name]=subspace.analyze_subspaces(r,first['camera'],first['geometry'],first['nuisance'],
                half['camera'],half['geometry'],half['nuisance'],w,ownership,column_scales=common_scale)
            print(json.dumps(dict(stage='absolute_subspaces_complete',view=name,
                outcomes=analyses[name]['summary']['outcomes'],stability=analyses[name]['summary']['residual_attribution_two_stencil_stability'])),flush=True)
        for name,(r,w,first,half,ownership)in views.items():
            key=name+'_observation_only'
            analyses[key]=subspace.analyze_subspaces(r[observation_mask],first['camera'][observation_mask],first['geometry'][observation_mask],
                first['nuisance'][observation_mask],half['camera'][observation_mask],half['geometry'][observation_mask],
                half['nuisance'][observation_mask],w[observation_mask],ownership[observation_mask],column_scales=common_scale)
            print(json.dumps(dict(stage='observation_only_subspaces_complete',view=name,
                prior_rows_excluded=int(np.count_nonzero(~observation_mask)),outcomes=analyses[key]['summary']['outcomes'])),flush=True)
        report['actual_local_IRLS_diagnostic']=analyses['actual']['summary']
        report['supportless_typed_reference_diagnostic']=analyses['rawtyped']['summary']
        report['actual_observation_only_relaxation']=analyses['actual_observation_only']['summary']
        report['supportless_typed_reference_observation_only_relaxation']=analyses['rawtyped_observation_only']['summary']
        report['source_group_residual_attribution']=inputs.group_report(groups,offsets,analyses['actual'],analyses['rawtyped'])
        report['global_tail_scalar_identity']={record['family']:record['row_labels']for record in tail_records}
        report['scope']='All current absolute FIT at the selected 55 programme exposures, original wide5808, external209, shared patterns104 and their original associated installation bias3 prior; no other programme exposures/motion/guards/prior families. Local relaxed absolute-source column spaces, not finite-change or full-source certificates.'
        report['rawtyped_diagnostic_metric']='Same nominal actual IRLS weights and same column scales across both stencils and both views. Supportless typed references only where source UV carries such a reference; external search-only supports are retained and individually labelled.'
        report['nonlinear_original_selected_absolute_source_cost']=sum(ctx.weighted_source_cost(group['actual'],group['quadratic'],group['loss'])for group in groups)
        report['all_rows_have_original_source_identity']=True
        arrays={name+'__'+key:value for name,analysis in analyses.items()for key,value in analysis['arrays'].items()}
        arrays.update(original_scalar_scales=np.concatenate([group['scales']for group in groups]),
            original_scalar_quadratic=np.concatenate([group['quadratic']for group in groups]),
            source_group_scalar_offsets=offsets,original_scalar_source_role=roles,
            observation_only_original_row_indices=np.flatnonzero(observation_mask))
        report['observation_only_scope']='Same calculated original Jacobians restricted to all pixel references, qualified finite source supports and unknown-material pixel correspondences; excludes the 38 external engineering roll rows and 3 installation bias rows. Neither supported pixels nor unknown material correspondences are promoted to surveyed world anchors.'
        report['schema']='rinkspace.joint-absolute-local-sensitivity.v1'
        array_path=a.H/'joint_absolute_q8a15_local_projection_arrays_20261011.npz'
        np.savez_compressed(array_path,**arrays)
        report['projection_arrays']=dict(a.ref(array_path),bytes=array_path.stat().st_size,
            arrays=list(arrays),no_parameter_solution_no_candidate_q_no_source_J_cache=True)
        assert before_sources==[a.ref(path)for path in files],'Diagnostic sources changed during numerical evaluation'
        report['frozen_diagnostic_sources']=before_sources
        report['seconds']=time.perf_counter()-started
        destination=a.H/'joint_absolute_q8a15_local_sensitivity_20261011.json'
        destination.write_text(json.dumps(report,default=a.plain,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps(dict(stage='joint_absolute_diagnostic_complete',path=str(destination),sha256=a.sha(destination),
            bytes=destination.stat().st_size,array_bytes=array_path.stat().st_size,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
