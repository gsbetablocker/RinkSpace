"""Continue authenticated current 7674 head+PP+shared5 state without old baseline replay.

By default no iteration/accepted-step/time cap is introduced. An explicit
caller-requested accepted-state handoff uses the public control interface.
The public solver owns
physical rejection, true source costs, line searches and numerical convergence.
Every accepted qaug and its whole-shot raw values remain recoverable in Data.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(r'D:\CodexData\RinkSpace_Data')
FIT_ROOT = DATA_ROOT / 'events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014'
RUN_ROOT = FIT_ROOT / 'zoom_poly_72ac'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN_ROOT))
import argparse,json,time,traceback
import numpy as np
from scipy.sparse import save_npz
from native_pp_continuation import Z,sha,write,reference,read_reference,configure_native_runtime
from native_transverse_bundle import MODEL
from native_transverse_metric_bundle import CompleteSourceMetricTransverseBundle
from calibration.program_tracking.first_shot_resume_cache import CompleteSourceReplay
DEFAULT_RUN = Z/'transverse_geometry_q721_nonlinear'
DEFAULT_SEED=DEFAULT_RUN/'accepted_states/0002_8c9e84194031_canonical.npz'
DEFAULT_SOURCE_QA=Z.parent/'head_axis_audit/transverse_q721_QA_v9/report.json'
from calibration.program_tracking.first_shot_alignment_monitor import _target_label
from z190_joint_source_solver import solve_joint_source,source_cost
import z190_joint_source_solver as solver_runtime
import z190_streaming_source_normal as normal_runtime
from calibration.program_tracking import finite_source_line_coverage as coverage_runtime

def stats(a):
    a=np.asarray(a,float)
    return dict(points=len(a),rms_px=float(np.sqrt(np.mean(a*a))),mean_px=float(a.mean()),
        median_px=float(np.median(a)),p95_px=float(np.quantile(a,.95)),max_px=float(a.max()))

def dump_current(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def write_J_block(bank,index,block,cursor):
    path=bank/f'block_{index:02d}.npz';temporary=bank/f'block_{index:02d}.tmp.npz'
    save_npz(temporary,block.J,compressed=False);temporary.replace(path)
    return dict(block=index,row_start=cursor,row_count=len(block.r),J=reference(path),
        J_shape=list(block.J.shape),J_nonzeros=block.J.nnz,row_ledger=block.row_ledger)

def inventories(bundle):
    points=[];frames=[];offset=0
    for f in bundle.frames:
        ids=[];labels=[];uv=[]
        for row in f['rows']:
            ids.extend(row['source_point_ids']);labels.extend([_target_label(row['target'])]*len(row['uv']));uv.extend(row['uv'])
        assert len(ids)==f['source_point_count']
        points.extend((f['index'],f['native_pts'],str(i),str(l),v)for i,l,v in zip(ids,labels,uv,strict=True))
        frames.append(dict(frame=f['index'],native_pts=f['native_pts'],raw_start=offset,raw_count=len(ids)))
        offset+=len(ids)
    return points,frames

def main():
    main_backend=configure_native_runtime(stage='complete-current-transverse-source-resume',native_threads=1)
    p=argparse.ArgumentParser()
    p.add_argument('--binding',default=str(FIT_ROOT/'revised_inputs_rs_v20_native_tail/binding.json'))
    p.add_argument('--association',default=str(FIT_ROOT/'revised_inputs_rs_v20_native_tail/association.json'))
    p.add_argument('--seed',default=str(DEFAULT_SEED))
    p.add_argument('--qa-report',default=str(FIT_ROOT/'head_axis_audit/transverse_state_q721_QA_v3/report.json'))
    p.add_argument('--source-qa-report',default=str(DEFAULT_SOURCE_QA))
    p.add_argument('--model',default=str(MODEL))
    p.add_argument('--cache-manifest',default=str(DEFAULT_RUN/'source_J_cache/latest.json'))
    p.add_argument('--world-cache-manifest',default=str(DEFAULT_RUN/'world_J_cache/latest.json'))
    p.add_argument('--prior-report',default=str(DEFAULT_RUN/'current.json'))
    p.add_argument('--pool-transition',default=str(Z/'geometry_pool_reuse'))
    p.add_argument('--recovery-review',default=str(Z/'print_domain_shared_constraints_QA.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--geometry-workers',type=int,default=20)
    p.add_argument('--source-workers',type=int,default=6)
    p.add_argument('--prepare-only',action='store_true',help='Authenticate/rebuild saved normal and physical guards; never call optimizer')
    p.add_argument('--geometry-context',action='store_true',help='Use separately verified persistent transverse worker context')
    args=p.parse_args()
    if args.geometry_workers<1 or args.source_workers<1:p.error('Worker counts must be positive')
    out=Path(args.output)
    if args.prepare_only and (out/'run_identity.json').exists():raise ValueError('Preparation cannot overwrite an existing optimizer run')
    out.mkdir(exist_ok=args.prepare_only,parents=True);(out/'accepted_states').mkdir(exist_ok=args.prepare_only);started=time.time()
    qa_ref=reference(args.qa_report);source_qa_ref=reference(args.source_qa_report)
    recovery_review=reference(args.recovery_review)
    adapter=CompleteSourceMetricTransverseBundle(Path(args.binding),Path(args.association),Path(args.seed),
        model=Path(args.model),geometry_workers=args.geometry_workers,source_workers=args.source_workers)
    geometry_context=None
    if args.geometry_context:
        from calibration.program_tracking.first_shot_transverse_execution import install_geometry_context
        geometry_context=install_geometry_context(adapter)
    replay=CompleteSourceReplay(adapter,manifest=args.cache_manifest,prior_report=args.prior_report,
        checkpoint=args.seed,pool_transition=args.pool_transition,world_manifest=args.world_cache_manifest).install()
    write(out/'authenticated_initial_cache.json',replay.receipt)
    adapter.public_checkpoint_path=out/'public_solver/latest_accepted_state.npz'
    runtime_refs=[reference(solver_runtime.__file__),reference(normal_runtime.__file__),reference(coverage_runtime.__file__)]
    runtime_refs.append(reference(sys.modules[CompleteSourceReplay.__module__].__file__))
    if args.geometry_context:
        runtime_refs.append(reference(sys.modules[install_geometry_context.__module__].__file__))
    if args.prepare_only:
        try:
            adapter.guard(adapter.start)
            blocks=list(adapter.source_blocks(adapter.start,True))
            r=np.concatenate([block.r for block in blocks])
            cost=source_cost(r,adapter.quadratic)
            if abs(cost-replay.receipt['source_cost'])>1e-8:
                raise ValueError('Authenticated saved source objective cost changed')
            raw=adapter.raw(adapter.start)
            if not np.array_equal(raw,replay.raw) or not np.array_equal(r,replay.r):
                raise ValueError('Authenticated initial source/raw replay changed')
            changed=adapter.start.copy();changed[0]=np.nextafter(changed[0],np.inf)
            try:replay.require_same_state(changed)
            except ValueError:changed_q_rejected=True
            else:raise ValueError('Changed complete state was incorrectly admitted')
            receipt=dict(replay.receipt,preparation_passed=True,optimizer_called=False,
                fit_started=False,source_or_world_FD_recomputed=False,
                raw_projection_recomputed=False,normal_rebuilt_from_authenticated_current_r_J=True,
                conditional_mount_direction_rebuilt=True,current_original_physical_guard_passed=True,
                source_cost_replay_difference=cost-replay.receipt['source_cost'],
                complete_state_mutation_rejected=changed_q_rejected,
                replay_calls=replay.replay_calls,metric_receipt=adapter.metric_receipts[-1],
                actual_worker_counts=dict(geometry=adapter.bundle.geometry_workers,source=adapter.workers),
                geometry_context=geometry_context,producer=reference(__file__),
                elapsed_seconds=time.time()-started)
            write(out/'preparation_QA.json',receipt)
            print(json.dumps(dict(stage='complete_current_source_resume_prepared',
                qaug_sha256=sha(adapter.start),source_cost=cost,source_rows=len(r),
                original_blocks=len(replay.manifest['blocks']),current_blocks=len(blocks),
                geometry_workers=adapter.bundle.geometry_workers,source_workers=adapter.workers,
                fit_started=False,seconds=time.time()-started)),flush=True)
        finally:adapter.close()
        return

    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    optional_export_failures=[]
    def optional_export_failure(stage,error,q):
        receipt=dict(stage=stage,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            qaug_sha256=sha(q)if len(q)==adapter.state.count else None,qbase_sha256=sha(q[:5172]),
            diagnostic_export_only=True,original_source_blocks_and_optimization_continue=True,
            elapsed_seconds=time.time()-started)
        optional_export_failures.append(receipt)
        try:dump_current(out/'optional_diagnostic_export_failures.json',dict(failures=optional_export_failures))
        except Exception:pass
        print(json.dumps(dict(stage='optional_diagnostic_export_failed',failure=receipt)),flush=True)
    original_world=adapter.world;world_cache=out/'world_J_cache';world_tokens=set()
    def persist_public_world(q,jacobian=False):
        worlds,derivatives=original_world(q,jacobian)
        if jacobian:
            geometry=np.asarray(q[adapter.state.world_parameter_indices],float);token=sha(geometry)
            if token not in world_tokens:
                try:
                    world_cache.mkdir(exist_ok=True)
                    keys=list(worlds);payload=dict(world_keys=np.asarray(keys),geometry_values=geometry,
                        world_parameter_indices=adapter.state.world_parameter_indices,qbase=np.asarray(q[:5172],float),q_augmented=np.asarray(q,float),
                        target_descriptors_json=np.asarray([json.dumps(adapter.bundle.descriptors[k],sort_keys=True)for k in keys]))
                    for i,key in enumerate(keys):payload[f'world_{i:04d}']=worlds[key];payload[f'D_{i:04d}']=derivatives[key]
                    name=f'geometry_{len(world_tokens):06d}_{token[:12]}';temporary=world_cache/f'{name}.tmp.npz';path=world_cache/f'{name}.npz'
                    np.savez(temporary,**payload);temporary.replace(path)
                    receipt=dict(schema='rinkspace.exact-current-shared-transverse-world-J-cache.v1',geometry_sha256=token,
                        qbase_sha256=sha(q[:5172]),qaug_sha256=sha(q),cache=reference(path),binding=adapter.bundle.state.binding_reference,
                        source_producers=adapter.source_producers,producer='adapter.world(full7674, True): original166 world FD then shared5 analytic transformation',
                        original_world166_and_print48_retained=True,world_parameter_count=171,shared_geometry_deviations=adapter.model.describe(q[:5172],adapter.state.deviations(q)),world_keys=len(keys),world_FD_for_export=False)
                    dump_current(world_cache/f'{name}.json',receipt)
                    if not world_tokens:dump_current(world_cache/'first.json',receipt)
                    dump_current(world_cache/'latest.json',receipt);world_tokens.add(token)
                except Exception as error:optional_export_failure('exact_current_public_world_J_cache',error,q)
        return worlds,derivatives
    adapter.world=persist_public_world
    points,frames=inventories(adapter.bundle)
    ids=np.asarray([r[2]for r in points]);labels=np.asarray([r[3]for r in points]);nativepts=np.asarray([r[1]for r in points],np.int64)
    frameids=np.asarray([r[0]for r in points],np.int32);uv=np.asarray([r[4]for r in points],float)
    np.savez_compressed(out/'immutable_current_source_ID_inventory.npz',native_pts=nativepts,frame_indices=frameids,
        source_point_ids=ids,source_target_labels=labels,source_uv=uv)
    write(out/'source_frame_map.json',dict(frames=frames,binding=reference(args.binding),source_inventory=reference(out/'immutable_current_source_ID_inventory.npz')))
    records=[];initialraw=None;initial_cost=None
    cache_root=out/'source_J_cache';linearization_count=0
    def exporting_source_blocks(q,jacobian):
        nonlocal linearization_count
        for ref in runtime_refs:assert reference(ref['path'])==ref,'Actual public solver runtime changed'
        if not jacobian:
            yield from adapter.source_blocks(q,False)
            return
        generation=linearization_count
        if np.array_equal(q,replay.q):
            yield from adapter.source_blocks(q,True)
            cache_root.mkdir(exist_ok=True)
            dump_current(cache_root/'reused_initial.json',replay.receipt)
            linearization_count+=1
            return
        bank=cache_root/f'generation_{generation:06d}'
        blocks=[];values=[];cursor=0;export_failed=False
        try:bank.mkdir(exist_ok=True,parents=True)
        except Exception as error:optional_export_failure('full_source_J_cache_directory',error,q);export_failed=True
        for index,block in enumerate(adapter.source_blocks(q,True)):
            count=len(block.r)
            if not export_failed:
                try:
                    values.append(np.asarray(block.r,float).copy())
                    blocks.append(write_J_block(bank,index,block,cursor))
                except Exception as error:optional_export_failure('full_source_J_block_export',error,q);export_failed=True
            cursor+=count
            yield block
        assert cursor==len(adapter.scales)
        linearization_count+=1
        if export_failed:return
        try:
            finish_source_J_export(q,generation,bank,blocks,values,cursor)
        except Exception as error:optional_export_failure('complete_source_J_manifest_export',error,q)
    def finish_source_J_export(q,generation,bank,blocks,values,cursor):
        state_path=bank/'linearization_state.npz';temporary=bank/'linearization_state.tmp.npz'
        np.savez(temporary,**adapter.complete_state_payload(q),r=np.concatenate(values),
            scales=adapter.scales,quadratic=adapter.quadratic)
        temporary.replace(state_path)
        manifest=dict(schema='rinkspace.actual-complete-native-PP-two-axis-transverse-source-J-cache.v1',generation=generation,
            qaug_sha256=sha(q),qbase_sha256=sha(q[:5172]),PP_sha256=sha(adapter.split(q)[1]),
            axis_a_working_m=float(adapter.state.axis(q)),shared_geometry_deviations=adapter.model.describe(q[:5172],adapter.state.deviations(q)),
            head_prefix_sha256=sha(q[:7669]),world_parameter_count=171,
            J_shape=[cursor,adapter.state.count],J_nonzeros=sum(a['J_nonzeros']for a in blocks),
            complete_current_state_r_scales_quadratic=reference(state_path),blocks=blocks,
            source_inventory=reference(out/'immutable_current_source_ID_inventory.npz'),
            source_frame_map=reference(out/'source_frame_map.json'),binding=reference(args.binding),
            source_producers=adapter.source_producers,adapter=reference(__import__('native_transverse_bundle').__file__),transverse_model=adapter.model_reference,
            numerical_recovery_producers=adapter.recovery_references,
            source_J_QA=qa_ref,geometry_column_source_QA=source_qa_ref,all_generations_retained=True,extra_world_FD_or_source_evaluation_for_export=False,
            source_J_before_Huber_weights=True,
            matrix_reconstruction='scipy.sparse.vstack([load_npz(block.J.path) for block in blocks],format="csr")')
        dump_current(bank/'manifest.json',manifest)
        if generation==0:dump_current(cache_root/'first.json',manifest)
        dump_current(cache_root/'latest.json',manifest)
        print(json.dumps(dict(stage='native_PP_two_axis_transverse_complete_source_J_exported',generation=generation,
            qaug_sha256=sha(q),J_shape=manifest['J_shape'],J_nonzeros=manifest['J_nonzeros'],
            latest_manifest=str(cache_root/'latest.json'))),flush=True)
    def metrics(raw):
        result=dict(fullshot=stats(raw),families=[],tail=[])
        for family in ['ice','board']+sorted({str(s).split('.')[0]for s in labels if str(s).startswith('Ad')}):
            mask=np.char.startswith(labels,family+'.')
            if mask.any():result['families'].append(dict(family=family,**stats(raw[mask])))
        for frame in [1234,1239,1247,1248]:
            scope=frameids==frame
            for target in sorted(set(labels[scope])):
                if not str(target).startswith(('ice.','Ad4.','Ad5.','board.')):continue
                mask=scope&(labels==target)
                entry=dict(frame=frame,target=str(target),**stats(raw[mask]))
                if initialraw is not None:entry['same_ID_initial']=stats(initialraw[mask])
                result['tail'].append(entry)
        return result
    def checkpoint(q,event):
        nonlocal initialraw,initial_cost
        if event['checkpoint_kind']not in ['initial','accepted','resumed']:return None
        adapter.guard(q);raw=adapter.raw(q)
        assert raw.shape==(len(points),)and np.isfinite(raw).all()
        old,pp=adapter.split(q)
        if initialraw is None:
            initialraw=raw.copy();initial_r=adapter.evaluate(q,False)
            initial_cost=source_cost(initial_r,adapter.quadratic);cost=initial_cost
            index=0
        else:
            initial_r=None;cost=float(event['next_cost']);index=sum(r['kind']=='accepted'for r in records)+1
        snapshot=out/'accepted_states'/f'{index:04d}_{sha(q)[:12]}.npz'
        payload=adapter.complete_state_payload(q)
        payload.update(point_errors_px=raw,source_point_ids=ids,native_pts_per_point=nativepts)
        if initial_r is not None:payload['formal_r']=initial_r
        np.savez_compressed(snapshot,**payload)
        canonical=out/'accepted_states'/f'{index:04d}_{sha(q)[:12]}_canonical.npz'
        public_bytes=Path(event['checkpoint']).read_bytes()
        from io import BytesIO
        with np.load(BytesIO(public_bytes),allow_pickle=False)as public_state:
            assert np.array_equal(public_state['q'],q),'Public canonical checkpoint differs from reviewed completeq'
            assert 'identity_json'in public_state.files
            for key,value in adapter.checkpoint_arrays().items():assert np.array_equal(public_state['aux__'+key],value),'Canonical aux differs: '+key
        canonical.write_bytes(public_bytes)
        ppstats={axis:dict(min_px=float(pp[:,i].min()),max_px=float(pp[:,i].max()),rms_px=float(np.sqrt(np.mean(pp[:,i]**2))))for i,axis in enumerate('xy')}
        report=dict(schema='rinkspace.complete-native-PP-two-axis-transverse-nonlinear-current-state.v1',kind=event['checkpoint_kind'],accepted_index=index,
            qaug_sha256=sha(q),qbase_sha256=sha(old),PP_sha256=sha(pp),source_objective_cost=cost,
            initial_current_native_objective_cost=initial_cost,source_points=len(points),formal_rows=len(adapter.scales),
            binding=reference(args.binding),source_basis='unchanged authenticated v20 native-group objective; same-source costs compare across camera models',
            snapshot=reference(snapshot),canonical_checkpoint=reference(canonical),public_checkpoint=event['checkpoint'],event=event,
            raw=metrics(raw),PP_statistics=ppstats,mount=adapter.model.mount_envelope(adapter.state,q),axis_a_working_m=float(adapter.state.axis(q)),
            shared_geometry_deviations=adapter.model.describe(q[:5172],adapter.state.deviations(q)),head_prefix_sha256=sha(q[:7669]),
            internal_old_d_box_enforced=False,total_offset_box_does_not_prove_compact_head_physicality=True,
            same_original_source_IDs_UV_sigma_and_noise_basis_throughout_this_run=True,
            all_5172_old_parameters_native_PP_axis_a_and_shared_geometry5_joint=True,world_shared_not_per_frame_geometry=True,fresh_current_source_J_every_new_state=True,
            initial_exact_complete_current_state_full_J_reuse=True,
            only_four_extra_PP_gauge_coordinates_fixed=adapter.fixed.tolist(),accuracy_accepted=False,
            full_grid_render_not_evaluated_by_this_callback=True,video_generated=False,elapsed_seconds=time.time()-started)
        record_path=out/'accepted_states'/f'{index:04d}_{sha(q)[:12]}.json';write(record_path,report)
        compact=dict(kind=event['checkpoint_kind'],accepted_index=index,qaug_sha256=sha(q),qbase_sha256=sha(old),
            objective=cost,full_raw_RMS_px=report['raw']['fullshot']['rms_px'],report=reference(record_path),snapshot=reference(snapshot),canonical_checkpoint=reference(canonical),
            elapsed_seconds=report['elapsed_seconds'])
        records.append(compact);dump_current(out/'current.json',report);dump_current(out/'accepted_state_index.json',dict(records=records))
        print(json.dumps(dict(stage='native_PP_two_axis_transverse_nonlinear_actual_state',**compact,
            tail=[a for a in report['raw']['tail']if a['frame']in[1239,1248]and a['target'].startswith(('ice.C','ice.lane1'))])),flush=True)
        return dict(current_report=reference(record_path),complete_state=reference(snapshot),canonical_checkpoint=reference(canonical),accuracy_accepted=False)
    try:
        kwargs=adapter.solver_kwargs();identity=kwargs['checkpoint_identity']
        kwargs['source_block_factory']=exporting_source_blocks
        identity.update(full_source_J_QA=qa_ref,geometry_column_source_QA=source_qa_ref,all_generations_retained=True,recovery_independent_review=recovery_review,run_wrapper=reference(Path(__file__)),main_backend=main_backend,
            actual_public_solver_and_normal_runtime=runtime_refs,
            source_inventory=reference(out/'immutable_current_source_ID_inventory.npz'),
            no_imposed_iteration_step_or_time_cap=True,
            shared_geometry_is_nested_candidate_not_measured_engineering_fact=True,
            no_added_construction_error_prior_or_numeric_cap=True,
            complete_native_objective_not_old_compressed_objective=True)
        identity['optional_diagnostic_export_cannot_stop_optimizer']=True
        identity['geometry_context']=geometry_context
        write(out/'run_identity.json',identity)
        print(json.dumps(dict(stage='native_PP_two_axis_transverse_complete_nonlinear_start',seed=adapter.seed_receipt,
            native_source_binding=reference(args.binding),source_points=len(points),formal_rows=len(adapter.scales),
            old_columns=5172,PP_columns=2496,axis_a_columns=1,new_shared_geometry_columns=5,world_columns=171,shared_dense=186,source_workers=adapter.workers,geometry_workers=adapter.bundle.geometry_workers,native_threads=1,
            initial_damping=kwargs['initial_damping'],initial_fraction=kwargs['initial_line_search_fraction'])),flush=True)
        result=solve_joint_source(adapter.evaluate,adapter.start,adapter.quadratic,adapter.scales,out/'public_solver',adapter.guard,
            checkpoint_callback=checkpoint,source_stage_timing=True,verbose=True,**kwargs)
        final=adapter.complete_state_payload(result.q);np.savez_compressed(out/'final_complete_state.npz',**final)
        summary=dict(schema='rinkspace.complete-native-PP-two-axis-transverse-nonlinear-run-result.v1',status=result.status,stop_reason=result.stop_reason,
            qaug_sha256=sha(result.q),qbase_sha256=sha(result.q[:5172]),PP_sha256=sha(adapter.split(result.q)[1]),axis_a_working_m=float(adapter.state.axis(result.q)),shared_geometry_deviations=adapter.model.describe(result.q[:5172],adapter.state.deviations(result.q)),
            accepted_iterations=sum(r['accepted']for r in result.history),public_checkpoint=str(result.checkpoint),
            final_complete_state=reference(out/'final_complete_state.npz'),current_report=reference(out/'current.json'),
            evaluations=adapter.evaluation_receipts,worker_receipts=adapter.worker_receipts,
            optional_diagnostic_export_failures=optional_export_failures,
            elapsed_seconds=time.time()-started,fit_has_not_been_accepted_as_visually_accurate=True,video_generated=False)
        write(out/'result.json',summary);print(json.dumps(dict(stage='native_PP_two_axis_transverse_complete_nonlinear_stopped',status=result.status,
            accepted_iterations=summary['accepted_iterations'],qaug_sha256=summary['qaug_sha256'],seconds=summary['elapsed_seconds'])),flush=True)
    except BaseException as error:
        failure=dict(schema='rinkspace.native-PP-nonlinear-recoverable-runtime-failure.v1',type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),last_complete_state=None if not records else records[-1]['snapshot'],
            public_checkpoint=str(out/'public_solver/latest_accepted_state.npz'),elapsed_seconds=time.time()-started,
            failure_does_not_prove_real_camera_unexplainable=True)
        write(out/'run_failure.json',failure);raise
    finally:adapter.close()

if __name__=='__main__':main()
