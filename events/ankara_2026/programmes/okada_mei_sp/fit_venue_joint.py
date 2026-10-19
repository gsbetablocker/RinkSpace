"""One persistent authenticated owner for programme plus static venue rows.

Configuration and a complete fresh append/J proof precede the control-file
handoff. The same owner then runs the public solver; old objective caches are
never installed and the independent venue clock stays outside programme PTS.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType, SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(r'D:\CodexData\RinkSpace_Data')
Z = DATA/'events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac'
sys.path[:0] = [str(ROOT), str(Z), str(Path(__file__).parent)]
import numpy as np
from scipy.sparse import save_npz
from scipy.optimize import least_squares
from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_source_transition import reference, read, checkpoint_reference_verifier
from calibration.program_tracking.first_shot_resume_cache import source_inventory
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage, apply_complete_group_quarantine
from calibration.program_tracking.first_shot_safe_frame_export import export_source_frames, _encode
from calibration.program_tracking.static_venue_source import AdmittedStaticPacket, StaticCameraPolicy, StaticVenueFactor
from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
import z190_joint_source_solver as solver


def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def stats(values):
    values = np.asarray(values, float)
    return dict(points=len(values), rms_px=float(np.sqrt(np.mean(values**2))),
        median_px=float(np.median(values)), p95_px=float(np.quantile(values,.95)), max_px=float(values.max()))


def condition_camera(factor, programme_q, output, seed_reference):
    """Condition only this admitted camera, using literal outer Huber weights."""
    scene = factor.scene_at(programme_q)
    weights = factor.packet.loss_weights.integration_weights
    def loss(z):
        near = z <= 4.; root = np.sqrt(np.maximum(z, np.finfo(float).tiny))
        return np.vstack((np.where(near,z,4.*root-4.),
            np.where(near,1.,2./root),np.where(near,0.,-1./(root**3))))*weights
    def value(x):
        return factor._evaluate_scene(scene,factor.camera_policy.decode(x),False)
    def derivative(x):
        return factor._evaluate_scene(scene,factor.camera_policy.decode(x),True)[1]
    initial = factor.camera_policy.encode()
    result = least_squares(value,initial,jac=derivative,loss=loss,xtol=1e-10,ftol=1e-10,gtol=1e-8)
    residual = value(result.x)
    if not np.isfinite(residual).all():
        raise ValueError('Conditioned static camera produced nonfinite source')
    report = dict(schema='rinkspace.admitted-static-camera-only-conditioning.v1',
        seed_reference=seed_reference, admitted_packet=factor.packet.reference,
        source_identity_sha256=factor.packet.identity_sha256,
        programme_q_sha256=sha(programme_q), geometry_and_programme_parameters_changed=False,
        camera25=factor.camera_policy.decode(result.x).tolist(), coordinates=result.x.tolist(),
        initial_source_cost=solver.source_cost(value(initial),factor.quadratic,factor.packet.loss_weights),
        source_cost=solver.source_cost(residual,factor.quadratic,factor.packet.loss_weights),
        original_sigma_and_literal_outer_Huber_weights=True,
        groups=len(factor.packet.rows), UV=len(residual)//2, rows=len(residual),
        status=int(result.status), message=str(result.message), nfev=int(result.nfev),
        optimality=float(result.optimality), accuracy_accepted=False)
    write(output,report)
    return factor.camera_policy.decode(result.x)


def authenticate_parent(owner, manifest):
    import fit_post_native_identity as identity_entry
    import calibration.program_tracking.first_shot_resume_cache as cache
    strict = checkpoint_reference_verifier(Z/'geometry_pool_reuse')
    original = cache.CompleteSourceReplay
    class RichReferenceVerifier(original):
        def verify_reference(self, value):
            strict.verified, strict.transitions = self.verified, self.transitions
            return strict.verify_reference(value)
    cache.CompleteSourceReplay = RichReferenceVerifier
    try:
        replay = identity_entry.native.BaseFreshCurrentReplay(owner,
            prior_report=manifest['accepted_parent_report']['path'],
            checkpoint=manifest['accepted_parent_checkpoint']['path'], pool_transition=Z/'geometry_pool_reuse')
    finally:
        cache.CompleteSourceReplay = original
    inventory, _ = source_inventory(owner.bundle)
    with np.load(manifest['accepted_parent_inventory']['path'], allow_pickle=False) as saved:
        if set(saved.files) != set(inventory): raise ValueError('Parent source inventory fields changed')
        for key, value in inventory.items():
            if not np.array_equal(saved[key],value): raise ValueError('Parent source inventory changed: '+key)
    return replay.receipt


def build_owner(manifest, out, workers):
    from calibration.program_tracking import first_shot_startup_auth as auth
    from calibration.program_tracking import first_shot_post_native_identity as delta
    from calibration.program_tracking import first_shot_value_execution as value
    from calibration.program_tracking import first_shot_raw_execution as raw
    from calibration.program_tracking import first_shot_fused_execution as fused
    from calibration.program_tracking import row_scene_projection_fast
    import fit_post_native_identity as entry
    spec=importlib.util.spec_from_file_location('venue_startup_recipe_helpers',Z/'launch_family_auth.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    allowlist,recipes=loader.collect_startup_allowlist(manifest['post_native_identity_transition']['path'],auth)
    scope=auth.startup_authentication_scope(evidence_references=allowlist,
        expected_evidence_producer=reference(ROOT/'calibration/program_tracking/first_shot_revision.py'))
    parent_installation=value.install();row_scene_projection_fast.install()
    entry.reuse.mount.recover_finite_extent=True
    entry.reuse.mount.failure_directory=out/'guard_failures'
    owner=None
    try:
        with scope:
            delta.install_post_native_identity(manifest['post_native_identity_transition']['path'])
            owner=entry.IdentityBundle(Path(manifest['binding']['path']),Path(manifest['association']['path']),
                Path(manifest['accepted_parent_checkpoint']['path']),
                geometry_workers=workers[0],source_workers=workers[1])
            parent=authenticate_parent(owner,manifest)
        scope.close()
        write(out/'parent_authentication.json',parent)
        write(out/'startup_authentication.json',dict(scope.receipt(),recipe_references=recipes,
            closed_before_workers=True,one_owner_only=True))
        quarantine=read(manifest['uncertain_group_manifest'])
        for field, expected in (('parent_binding',manifest['binding']),('parent_association',manifest['association']),
                                ('parent_inventory',manifest['accepted_parent_inventory'])):
            if quarantine[field] != expected: raise ValueError('Complete-group source parent mismatch: '+field)
        read(quarantine['actual_source_audit']);read(quarantine['original_RGB_review'])
        receipt=apply_complete_group_quarantine(owner,quarantine['specifications'],
            manifest_reference=manifest['uncertain_group_manifest'])
        write(out/'complete_group_source_transition.json',receipt)
        # The released-domain recover does not use ConditionalMountDirection.
        # Bypass only its unused programme-only diagnostic normal; the public
        # solver builds the real normal from all programme and static rows.
        import native_transverse_bundle as native_runtime
        if owner._unmeasured_mount_domain['conditional_source_metric_may_still_be_built_but_never_projects_to_photo_box'] is not True:
            raise ValueError('Cannot bypass an active conditional metric recovery')
        owner.source_blocks=MethodType(native_runtime.CompleteTransverseBundle.source_blocks,owner)
        owner.metric=None
        from calibration.program_tracking.first_shot_transverse_execution import install_geometry_context
        context=install_geometry_context(owner)
        from calibration.program_tracking.first_shot_pivot_recovery import install_pivot_boundary_recovery
        install_pivot_boundary_recovery(owner,entry_reference=reference(__file__),
            qa_references=(reference(Z/'pivot_nonlinear_QA/recovery_report.json'),))
        raw.install(owner)
        execution=fused.install_worker_execution(owner,parent_installation=parent_installation,
            entry_reference=reference(__file__),validation_references=(
                reference(Z/'value_real_state_QA/report.json'),reference(Z/'pivot_step_QA/report.json')))
        execution.update(full_same_seed_source_r_J_proof_required=False,
            cross_source_basis_old_new_r_J_equality_not_claimed=True,new_source_initial_full_r_J_required=True,
            unused_programme_only_conditional_metric_not_constructed=True,
            complete_source_public_normal_includes_static_cross_terms=True)
        for name in ('frames','state','scales','mask'):
            if hasattr(owner.bundle,'_post_identity_previous_'+name): delattr(owner.bundle,'_post_identity_previous_'+name)
        return owner,context
    except BaseException:
        if owner is not None: owner.close()
        raise
    finally: scope.close()


def build_stage(programme, manifest, out):
    admission=read(manifest['static_material_admission'])
    packet=AdmittedStaticPacket(admission['source_packet']['path'],
        expected_sha256=admission['source_packet']['sha256'],admitted_control_ids=admission['admitted_control_ids'])
    if (len(packet.rows),len(packet.scales)//2,len(packet.scales)) != (
            admission['control_group_count'],admission['control_UV_count'],admission['control_scalar_rows']):
        raise ValueError('Explicit material admission counts differ from source rows')
    seed=read(manifest['static_camera_seed'])
    policy=StaticCameraPolicy('brown13',seed['camera25'])
    def geometry_guard(q):
        programme.state._activate(q)
        return programme.state.original.guard_geometry(q[:programme.state.original.count])
    options=dict(programme_count=len(programme.start),geometry_columns=programme.state.world_parameter_indices,
        scene_at=programme.state.scene,geometry_guard=geometry_guard,
        geometry_fd_step=programme.state.original.provider.fd_step)
    factor=StaticVenueFactor(packet,policy,**options)
    conditioned=condition_camera(factor,programme.start,out/'static_camera_conditioning.json',manifest['static_camera_seed'])
    factor=StaticVenueFactor(packet,StaticCameraPolicy('brown13',conditioned),**options)
    stage=StaticVenueStage(programme,factor,parent_reference=manifest['accepted_parent_report'],
        entry_reference=reference(__file__),source_references=(manifest['static_material_admission'],
            packet.reference,manifest['uncertain_group_manifest'],manifest['static_camera_seed'],
            reference(admission['material_RGB_review'])))
    removed=read(manifest['uncertain_group_manifest'])['expected_removed_formal_rows']
    with np.load(manifest['accepted_parent_checkpoint']['path'],allow_pickle=False) as saved:
        parent_rows=len(saved['scales'])
    if (len(programme.scales),len(factor.scales),len(stage.scales),stage.parameter_count,len(stage.shared)) != (
            parent_rows-removed,admission['control_scalar_rows'],parent_rows-removed+admission['control_scalar_rows'],
            len(programme.start)+policy.count,len(programme.shared)+policy.count):
        raise ValueError('Authenticated complete new-stage row/state/shared counts changed')
    stage.stage_receipt.update(unused_programme_only_conditional_metric_not_constructed=True,
        public_full_source_normal_and_wide_cross_terms_required=True)
    return stage


def export_review(stage, q, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    prefix,_=stage.split_state(q)
    selected=export_source_frames(stage.programme,output_directory=folder/'programme',
        frame_indices=(201,385,589,842,1239,1248),q=prefix)
    block=stage.factor.block(q,False);pixels=block.r*block.scales;groups=[];cursor=0
    for row in stage.factor.packet.rows:
        count=2*len(row['pixels_uv']);sl=slice(cursor,cursor+count)
        distance=np.linalg.norm(pixels[sl].reshape(-1,2),axis=1)
        groups.append(dict(id=row['id'],target=row['target'],source_cost=solver.source_cost(
            block.r[sl],block.quadratic[sl],stage.factor.packet.loss_weights.integration_weights[sl]),**stats(distance)))
        cursor+=count
    if cursor!=len(block.r) or len(groups)!=len(stage.factor.packet.rows): raise ValueError('Review does not cover every admitted wide row')
    arrays={};payload=_encode(dict(q=np.asarray(q),programme_q=prefix,
        scene=stage.programme.state.scene(prefix),camera25=stage.factor.camera_policy.decode(q[stage.factor.camera_columns]),
        wide_source_rows=stage.factor.packet.rows,wide_raw_pixel_vectors=pixels.reshape(-1,2),
        wide_normalized_residual=block.r,wide_formal_scales=block.scales,
        wide_loss_weights=stage.factor.packet.loss_weights.integration_weights,
        independent_source_frame=stage.factor.packet.frame),arrays)
    np.savez_compressed(folder/'static_arrays.npz',**arrays)
    static=dict(schema='rinkspace.safe-current-static-venue-review.v1',q_sha256=sha(q),
        arrays=reference(folder/'static_arrays.npz'),payload=payload,
        group_metrics=groups,source_cost=solver.source_cost(block.r,block.quadratic,block.loss_weights),
        source_identity_sha256=stage.factor.packet.identity_sha256,
        stage_identity=stage.checkpoint_identity(),same_q_scene_camera_and_source=True,
        accuracy_accepted=False,full_scene_render_required=True,pickle_deserialization=False)
    write(folder/'static.json',static)
    return dict(programme=selected,static=reference(folder/'static.json'),
        wide_cost=static['source_cost'],wide_groups=groups)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage-manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--geometry-workers',type=int,default=20);parser.add_argument('--source-workers',type=int,default=6)
    parser.add_argument('--serve',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest_ref=reference(args.stage_manifest);manifest=read(manifest_ref)
    if manifest.get('schema')!='rinkspace.explicit-programme-static-venue-stage.v1': raise ValueError('Explicit new-stage manifest required')
    if manifest.get('entry')!=reference(__file__):raise ValueError('Explicit new-stage entry producer changed')
    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for key in ('accepted_parent_report','accepted_parent_checkpoint','accepted_parent_inventory','binding','association',
                'post_native_identity_transition','uncertain_group_manifest','static_material_admission','static_camera_seed'):
        if reference(manifest[key]['path'])!=manifest[key]: raise ValueError('New-stage input bytes changed: '+key)
    report=read(manifest['accepted_parent_report'])
    if report['canonical_checkpoint']!=manifest['accepted_parent_checkpoint'] or report['qaug_sha256']!=manifest['accepted_parent_q_sha256']:
        raise ValueError('New stage must inherit the explicit accepted parent')
    from native_pp_continuation import configure_native_runtime
    configure_native_runtime(stage='programme-plus-continuous-static-venue',native_threads=1)
    started=time.time();stage=None;programme=None;records=[]
    write(out/'status.json',dict(stage='authenticating_one_owner',pid=os.getpid(),manifest=manifest_ref,fit_started=False))
    try:
        programme,context=build_owner(manifest,out,(args.geometry_workers,args.source_workers))
        # Admission can finish while the one expensive programme owner is
        # authenticating. Select it explicitly before any static/combined J,
        # without rebuilding programme source or silently mutating a manifest.
        if args.serve:
            write(out/'status.json',dict(stage='programme_owner_ready_waiting_static_configuration',
                pid=os.getpid(),fit_started=False,programme_source_rows=len(programme.scales)))
            print(json.dumps(dict(stage='programme_owner_ready_waiting_static_configuration',pid=os.getpid())),flush=True)
            while True:
                config_path=out/'static_configuration_control.json'
                if config_path.exists():
                    config_ref=reference(config_path);config=read(config_ref)
                    if config.get('action')=='close_owner':return
                    if config.get('action')=='configure_static':
                        for key in ('static_material_admission','static_camera_seed'):
                            if key in config:
                                if reference(config[key]['path'])!=config[key]:raise ValueError('Static configuration reference changed')
                                manifest[key]=deepcopy(config[key])
                        write(out/'static_configuration.json',dict(original_stage_manifest=manifest_ref,
                            explicit_configuration_control=config_ref,resolved_manifest=manifest,
                            programme_owner_rebuilt=False,combined_source_J_constructed=False))
                        break
                time.sleep(2.)
        stage=build_stage(programme,manifest,out)
        stage.guard(stage.start)
        inventory,_=source_inventory(programme.bundle)
        np.savez_compressed(out/'immutable_programme_source_inventory.npz',**inventory)
        arrays={};inactive=_encode([dict(frame=f['index'],native_pts=f['native_pts'],groups=f['inactive_source_groups'])
            for f in programme.bundle.frames if f.get('inactive_source_groups')],arrays)
        np.savez_compressed(out/'inactive_evidence.npz',**arrays)
        write(out/'inactive_evidence.json',dict(arrays=reference(out/'inactive_evidence.npz'),payload=inactive,
            original_groups_UV_sigma_targets_and_W_preserved=True))
        cache=out/'source_J_cache';generation=0
        def source_blocks(q,jacobian):
            nonlocal generation
            if not jacobian:
                yield from stage.source_blocks(q,False);return
            bank=cache/('generation_%06d'%generation);bank.mkdir(parents=True,exist_ok=False)
            values=[];blocks=[];cursor=0;losses=[]
            for index,block in enumerate(stage.source_blocks(q,True)):
                path=bank/('block_%02d.npz'%index);save_npz(path,block.J,compressed=False)
                values.append(block.r.copy());losses.append(source_loss_checkpoint_arrays(block.loss_weights,len(block.r)))
                blocks.append(dict(block=index,row_start=cursor,row_count=len(block.r),J=reference(path),
                    J_shape=list(block.J.shape),J_nonzeros=block.J.nnz,row_ledger=block.row_ledger))
                cursor+=len(block.r);yield block
            state_path=bank/'linearization_state.npz'
            np.savez(state_path,**stage.complete_state_payload(q),r=np.concatenate(values),
                scales=stage.scales,quadratic=stage.quadratic)
            # The ordinary programme blocks have implicit unit integration;
            # the final static block's explicit weights remain in its ledger.
            loss_arrays={key+'__block_'+str(i):value for i,item in enumerate(losses)for key,value in item.items()}
            np.savez(bank/'block_loss_weights.npz',**loss_arrays)
            info=dict(schema='rinkspace.actual-programme-static-source-J-cache.v1',generation=generation,
                qaug_sha256=sha(q),J_shape=[cursor,len(q)],J_nonzeros=sum(b['J_nonzeros']for b in blocks),
                complete_current_state_r_scales_quadratic=reference(state_path),blocks=blocks,
                block_loss_weights=reference(bank/'block_loss_weights.npz'),
                source_J_before_Huber_weights=True,source_producers=stage.source_producers,
                source_identity=stage.checkpoint_identity(),full_wide_rows_always_present=True)
            write(bank/'manifest.json',info);write(cache/'latest.json',info);generation+=1
            print(json.dumps(dict(stage='complete_new_source_J_exported',generation=generation-1,
                q_sha256=sha(q),rows=cursor,columns=len(q))),flush=True)
        # The admitted factor's complete J/padding has a separate real proof.
        # Configuration evaluates every current source value and guard once;
        # the formal solver computes its new complete J on its first step.
        costs=[];rows=0
        for block in stage.source_blocks(stage.start,False):
            costs.append(solver.source_cost(block.r,block.quadratic,block.loss_weights));rows+=len(block.r)
        if rows!=len(stage.scales): raise ValueError('Full new-source append QA missed formal rows')
        review=export_review(stage,stage.start,out/'initial_review')
        qa=dict(schema='rinkspace.real-programme-static-stage-append-QA.v1',passed=True,
            owner_pid=os.getpid(),manifest=manifest_ref,source_identity=stage.checkpoint_identity(),
            q_sha256=sha(stage.start),parameter_count=len(stage.start),global_shared_count=len(stage.shared),
            source_rows=rows,programme_source_rows=len(programme.scales),wide_source_rows=len(stage.factor.scales),
            programme_cost=math.fsum(costs[:-1]),wide_cost=costs[-1],combined_cost=math.fsum(costs),
            exact_accepted_programme_prefix_preserved=True,old_r_J_reused=False,
            initial_review=review,formal_optimizer_called=False,
            static_camera_only_conditioning_performed=True,
            complete_new_source_J_required_on_first_formal_step=True,
            context=context,seconds=time.time()-started)
        write(out/'append_QA.json',qa)
        from calibration.program_tracking.first_shot_jacobian_reuse import install_source_jacobian_reuse
        from calibration.program_tracking.first_shot_prediction_refresh import install_prediction_refresh
        install_source_jacobian_reuse(stage,cache_directory=cache,entry_reference=reference(__file__))
        driver=SimpleNamespace(solve_joint_source=solver.solve_joint_source,solver_runtime=solver)
        install_prediction_refresh(stage,driver=driver,entry_reference=reference(__file__),
            minimum_prediction_ratio=.25,reset_initial_fresh_source_fraction=True)
        kwargs=stage.solver_kwargs();kwargs['source_block_factory']=source_blocks
        kwargs['checkpoint_identity'].update(explicit_stage_manifest=manifest_ref,
            source_inventory=reference(out/'immutable_programme_source_inventory.npz'),
            independent_static_packet_identity=stage.factor.packet.identity_sha256,
            completed_append_QA=reference(out/'append_QA.json'),geometry_context=context,
            initial_exact_complete_current_state_full_J_reuse=False)
        write(out/'run_identity.json',kwargs['checkpoint_identity'])
        write(out/'status.json',dict(stage='append_QA_passed_waiting_control',pid=os.getpid(),
            append_QA=reference(out/'append_QA.json'),fit_started=False,source_owner_retained=args.serve))
        print(json.dumps(dict(stage='append_QA_passed_waiting_control',path=str(out/'append_QA.json'),pid=os.getpid())),flush=True)
        if not args.serve: return
        while True:
            control_path=out/'stage_control.json'
            if control_path.exists():
                control=read(reference(control_path))
                if control.get('action')=='close_owner': return
                if control.get('action')=='start_fit':
                    if control.get('append_QA')!=reference(out/'append_QA.json') or control.get('q_sha256')!=sha(stage.start):
                        raise ValueError('Start control must authenticate actual complete append QA/q')
                    break
            time.sleep(2.)
        labels=inventory['source_target_labels'];frameids=inventory['frame_indices']
        def checkpoint(q,event):
            if event['checkpoint_kind']not in ('initial','accepted','resumed'):return None
            stage.guard(q);raw=stage.raw_programme(q);blocks=list(stage.source_blocks(q,False))
            costs=[solver.source_cost(b.r,b.quadratic,b.loss_weights)for b in blocks]
            index=sum(r['kind']=='accepted'for r in records)+(event['checkpoint_kind']=='accepted')
            folder=out/'accepted_states'/('%04d_'%index+sha(q)[:12]);folder.mkdir(parents=True,exist_ok=False)
            canonical=folder/'canonical.npz';canonical.write_bytes(Path(event['checkpoint']).read_bytes())
            with np.load(canonical,allow_pickle=False)as saved:
                if not np.array_equal(saved['q'].view(np.uint64),q.view(np.uint64)):raise ValueError('Full canonical q differs')
            np.savez_compressed(folder/'complete_state.npz',**stage.complete_state_payload(q),
                programme_point_errors_px=raw,source_point_ids=inventory['source_point_ids'],native_pts_per_point=inventory['native_pts'])
            review=export_review(stage,q,folder/'review')
            metrics=[]
            for frame in (201,385,589,842,1239,1248):
                for label in sorted(set(labels[frameids==frame])):
                    mask=(frameids==frame)&(labels==label)
                    metrics.append(dict(frame=int(frame),target=str(label),**stats(raw[mask])))
            record=dict(schema='rinkspace.accepted-programme-static-source-state.v1',kind=event['checkpoint_kind'],
                accepted_index=int(index),qaug_sha256=sha(q),programme_q_sha256=sha(q[:stage.programme_count]),
                source_objective_cost=math.fsum(costs),programme_source_cost=math.fsum(costs[:-1]),wide_source_cost=costs[-1],
                full_programme_raw=stats(raw),selected_programme_target_metrics=metrics,
                all_wide_group_raw_metrics=review['wide_groups'],review=review,
                canonical_checkpoint=reference(canonical),snapshot=reference(folder/'complete_state.npz'),
                source_identity=stage.checkpoint_identity(),event=event,
                accuracy_accepted=False,all_scene_same_q_visual_review_required=True,
                overall_cost_reduction_is_not_geometry_or_alignment_acceptance=True,seconds=time.time()-started)
            write(folder/'report.json',record);write(out/'current.json',record)
            records.append(dict(kind=record['kind'],q_sha256=sha(q),report=reference(folder/'report.json')))
            write(out/'accepted_state_index.json',dict(records=records))
            print(json.dumps(dict(stage='accepted_stage_state',kind=record['kind'],accepted_index=int(index),
                q_sha256=sha(q),programme_cost=record['programme_source_cost'],wide_cost=record['wide_source_cost'])),flush=True)
            return dict(current_report=reference(folder/'report.json'),canonical_checkpoint=reference(canonical),accuracy_accepted=False)
        write(out/'status.json',dict(stage='formal_fit_running',pid=os.getpid(),fit_started=True))
        result=driver.solve_joint_source(stage.evaluate,stage.start,stage.quadratic,stage.scales,out/'public_solver',
            stage.guard,checkpoint_callback=checkpoint,source_stage_timing=True,verbose=True,**kwargs)
        np.savez_compressed(out/'final_complete_state.npz',**stage.complete_state_payload(result.q))
        write(out/'result.json',dict(status=result.status,stop_reason=result.stop_reason,q_sha256=sha(result.q),
            final_complete_state=reference(out/'final_complete_state.npz'),accepted_iterations=sum(r['kind']=='accepted'for r in records),
            accuracy_accepted=False,video_generated=False,seconds=time.time()-started))
    except BaseException as error:
        write(out/'run_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            seconds=time.time()-started,production_parent_not_overwritten=True))
        raise
    finally:
        if stage is not None: stage.close()
        elif programme is not None: programme.close()


if __name__=='__main__':
    main()
