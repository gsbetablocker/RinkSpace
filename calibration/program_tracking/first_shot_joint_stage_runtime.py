"""Owning public-solver/cache/review driver for explicit complete new stages."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import os
import time
from fractions import Fraction

import numpy as np
from scipy.sparse import save_npz

from .first_shot_source_transition import reference, read
from .first_shot_resume_cache import source_inventory
from .first_shot_safe_frame_export import _encode, _decode


def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def stats(values):
    values=np.asarray(values,float)
    return dict(points=len(values),rms_px=float(np.sqrt(np.mean(values**2))),
        median_px=float(np.median(values)),p95_px=float(np.quantile(values,.95)),max_px=float(values.max()))


def review_media_record(frame):
    """Normalize original metadata without inventing a missing file digest."""
    packet=read(frame['source_packet']);source=packet['source']
    if (int(source['native_pts'])!=frame['native_pts']
            or int(source['programme_frame_index_1based'])!=frame['index']):
        raise ValueError('Actual programme media/PTS binding changed')
    keys=('schema','event_id','programme_id','programme_frame_index_1based',
        'native_frame_index_1based','native_frame_index_0based','native_pts','actual_pts',
        'time_base','source_time','source_time_seconds','source','source_video_sha256',
        'image_path','png_sha256','decoded_BGR_sha256','decoded_RGB_sha256','native_hw',
        'original_PTS_verified','mask_path','mask_sha256')
    record={key:source[key]for key in keys if key in source};derived=[]
    if 'png_sha256'not in source:
        import cv2
        raw=Path(source['image_path']).read_bytes()
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None or sha(image)!=source['decoded_BGR_sha256']:
            raise ValueError('Legacy native PNG does not match authenticated decoded_BGR_sha256')
        record['png_sha256']=hashlib.sha256(raw).hexdigest()
        record['native_hw']=list(image.shape[:2])
        derived.extend(('png_sha256','native_hw'))
        record['png_digest_provenance']='actual file bytes; decoded_BGR matches original source packet'
    else:record['png_digest_provenance']='authenticated original source packet'
    if 'native_frame_index_0based'not in record:
        record['native_frame_index_0based']=int(source['native_frame_index_1based'])-1
        derived.append('native_frame_index_0based')
    if 'source_time_seconds'not in record:
        record['source_time_seconds']=float(int(source['native_pts'])*Fraction(source['time_base']))
        derived.append('source_time_seconds')
    record.update(index=int(frame['index']),native_pts=int(frame['native_pts']),
        source_packet=frame['source_packet'],derived_metadata_fields=derived,
        image=dict(path=source['image_path'],sha256=record['png_sha256']),
        mask=dict(path=source['mask_path'],sha256=source['mask_sha256']))
    return record


def preflight_review_media(binding_reference,output):
    """Read only existing packet indices; construct no source graph or owner."""
    binding=read(binding_reference);data=read(binding['complete_fit_input']);base=read(data['base_dataset'])
    frames=base['frames']+data['appended_frames']
    records=[review_media_record(frame)for frame in frames]
    for record in records:
        if (record['source']!=binding['source']['path']
                or record['source_video_sha256']!=binding['source']['sha256']
                or record['time_base']!=binding['source_clock']['time_base']
                or record['actual_pts']!=record['native_pts']
                or record['native_frame_index_0based']!=record['native_frame_index_1based']-1):
            raise ValueError('Preflight media belongs to another native video or clock')
    pts=np.asarray([r['native_pts']for r in records],np.int64)
    if (len(records)!=binding['frame_count']or records[0]['index']!=binding['frame_range'][0]
            or records[-1]['index']!=binding['frame_range'][1]
            or pts[0]!=binding['first_native_pts']or pts[-1]!=binding['last_native_pts']
            or np.any(np.diff(pts)!=binding['source']['native_pts_step'])):
        raise ValueError('Preflight media does not cover the complete official native clock')
    write(output,dict(schema='rinkspace.complete-native-media-preflight.v1',passed=True,
        binding=binding_reference,native_frames=len(records),frames=records,
        legacy_PNG_bytes_and_decoded_BGR_verified=sum('png_sha256'in r['derived_metadata_fields']for r in records),
        native_pts_sha256=sha(pts),source_graph_or_owner_constructed=False,producer=reference(__file__)))
    return reference(output)


def export_complete_review_inputs(programme,output):
    """Reuse the one loaded inventory, clock and original packet metadata."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory,frame_map=source_inventory(programme.bundle)
    np.savez_compressed(output/'immutable_programme_source_inventory.npz',**inventory)
    media=[];formal_rows=[];sigma=[]
    for frame,w,ledger in zip(programme.bundle.frames,programme.bundle.whitening,programme.bundle.ledgers,strict=True):
        record=review_media_record(frame);record['source_ledger']=ledger
        media.append(record)
        sigma.extend(np.asarray(row['sigma'],float)for row in frame['rows'])
        formal_rows.append(int(w.count))
    pts=np.asarray([r['native_pts']for r in media],np.int64)
    if not np.array_equal(pts,programme.checkpoint_arrays()['native_pts']):
        raise ValueError('Full review media no longer uses programme native clock')
    np.savez_compressed(output/'source_sigma_and_formal_offsets.npz',
        sigma_px=np.concatenate(sigma),formal_rows=np.asarray(formal_rows,np.int64),
        formal_offsets=programme.formal_offsets,scales=programme.scales,quadratic=programme.quadratic)
    write(output/'media_manifest.json',dict(schema='rinkspace.complete-native-programme-review-media.v1',
        native_frames=len(media),frames=media,source_frame_map=frame_map,
        source_inventory=reference(output/'immutable_programme_source_inventory.npz'),
        formal_inventory=reference(output/'source_sigma_and_formal_offsets.npz'),
        native_pts_sha256=sha(pts),source_clock_reused_without_frame_conversion=True,
        image_and_mask_hashes_bound_to_authenticated_original_packet_metadata=True,
        legacy_PNG_bytes_rehashed_and_decoded_BGR_authenticated=True,
        all_other_image_and_mask_bytes_rehashed_by_this_metadata_export=False,
        render_worker_must_verify_image_and_mask_bytes_before_use=True,
        another_source_bundle_constructed=False,video_generated=False,producer=reference(__file__)))
    return inventory,reference(output/'media_manifest.json')


def export_effective_review(owner,q,folder,*,physical_export):
    """Keep physical native-review q explicit beside the saved canonical q."""
    qe,chain=owner.effective_q_and_chain(q)
    result=physical_export(owner.physical_stage,qe,folder)
    folder=Path(folder)
    for path,kind in ((folder/'programme/frames.json','programme'),(folder/'static.json','static')):
        receipt=read(reference(path));arrays_ref=receipt['arrays']
        if reference(arrays_ref['path'])!=arrays_ref:raise ValueError('New review arrays changed')
        with np.load(arrays_ref['path'],allow_pickle=False)as saved:
            payload=_decode(receipt['payload'],{key:saved[key].copy()for key in saved.files})
        payload.update(canonical_q_augmented=np.asarray(q).copy(),effective_q_augmented=qe.copy(),
            physical_q_augmented=qe.copy(),
            canonical_auxiliary_arrays=owner.checkpoint_arrays(),complete_model_identity=owner.checkpoint_identity(),
            q_is_physical_effective_state_not_canonical=True)
        arrays={};encoded=_encode(payload,arrays)
        temporary=Path(arrays_ref['path']).with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(arrays_ref['path'])
        receipt.update(arrays=reference(arrays_ref['path']),payload=encoded,
            canonical_q_sha256=sha(q),effective_q_sha256=sha(qe),
            physical_q_sha256=sha(qe),programme_physical_q_sha256=sha(qe[:owner.programme_count]),
            canonical_programme_q_sha256=sha(np.asarray(q)[:owner.programme_count]),
            parameter_chart_role='physical_effective_state_for_native_projection',
            complete_model_identity=owner.checkpoint_identity(),effective_review_producer=reference(__file__))
        if kind=='static':
            receipt['stage_identity']=owner.checkpoint_identity()
            receipt['schema']='rinkspace.safe-current-static-venue-effective-review.v2'
        write(path,receipt)
    save_npz(folder/'effective_chain.npz',chain,compressed=False)
    write(folder/'canonical_and_effective.json',dict(canonical_q_sha256=sha(q),effective_q_sha256=sha(qe),
        physical_q_sha256=sha(qe),programme_physical_q_sha256=sha(qe[:owner.programme_count]),
        chain=reference(folder/'effective_chain.npz'),model_identity=owner.checkpoint_identity(),
        source_guard_scene_raw_and_constraints_use_same_effective_q=True))
    result['programme'].update(manifest=reference(folder/'programme/frames.json'),
        arrays=reference(folder/'programme/arrays.npz'),canonical_q_sha256=sha(q),effective_q_sha256=sha(qe))
    result.update(static=reference(folder/'static.json'),mapping=reference(folder/'canonical_and_effective.json'))
    return result


class JointStageRunner:
    def __init__(self,owner,*,output,manifest_reference,review_export):
        self.owner=owner;self.output=Path(output);self.manifest_reference=manifest_reference
        self.review_export=review_export;self.generation=0;self.records=[];self.started=time.time()
        self.cache=self.output/'source_J_cache'
        self.inventory,self.media=export_complete_review_inputs(owner.programme,self.output)

    def source_blocks(self,q,jacobian):
        from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
        if not jacobian:
            yield from self.owner.source_blocks(q,False);return
        bank=self.cache/('generation_%06d'%self.generation);bank.mkdir(parents=True,exist_ok=False)
        values=[];blocks=[];losses={};cursor=0
        for index,block in enumerate(self.owner.source_blocks(q,True)):
            path=bank/('block_%02d.npz'%index);save_npz(path,block.J,compressed=False)
            values.append(block.r.copy())
            losses.update({key+'__block_'+str(index):value for key,value in
                source_loss_checkpoint_arrays(block.loss_weights,len(block.r)).items()})
            blocks.append(dict(block=index,row_start=cursor,row_count=len(block.r),J=reference(path),
                J_shape=list(block.J.shape),J_nonzeros=block.J.nnz,row_ledger=block.row_ledger))
            cursor+=len(block.r);yield block
        if cursor!=len(self.owner.scales):raise ValueError('Current complete new-source J missed rows')
        np.savez(bank/'linearization_state.npz',**self.owner.complete_state_payload(q),
            r=np.concatenate(values),scales=self.owner.scales,quadratic=self.owner.quadratic)
        np.savez(bank/'block_loss_weights.npz',**losses)
        receipt=dict(schema='rinkspace.complete-canonical-effective-joint-source-J.v1',generation=self.generation,
            qaug_sha256=sha(q),J_shape=[cursor,len(q)],J_nonzeros=sum(b['J_nonzeros']for b in blocks),
            complete_current_state_r_scales_quadratic=reference(bank/'linearization_state.npz'),blocks=blocks,
            block_loss_weights=reference(bank/'block_loss_weights.npz'),
            source_J_before_Huber_weights=True,source_producers=self.owner.source_producers,
            source_identity=self.owner.checkpoint_identity(),full_static_rows_always_present=True)
        write(bank/'manifest.json',receipt);write(self.cache/'latest.json',receipt);self.generation+=1
        print(json.dumps(dict(stage='complete_canonical_effective_source_J_exported',
            generation=self.generation-1,q_sha256=sha(q),rows=cursor,columns=len(q))),flush=True)

    def prepare(self,**extra):
        from z190_joint_source_solver import source_cost
        owner=self.owner;owner.guard(owner.start);costs=[];rows=0
        for block in owner.source_blocks(owner.start,False):
            costs.append(source_cost(block.r,block.quadratic,block.loss_weights));rows+=len(block.r)
        if rows!=len(owner.scales):raise ValueError('Configured source did not include all formal rows')
        review=self.review_export(owner.start,self.output/'initial_review')
        receipt=dict(schema='rinkspace.complete-new-source-model-stage-QA.v1',passed=True,
            manifest=self.manifest_reference,q_sha256=sha(owner.start),parameter_count=len(owner.start),
            source_rows=rows,programme_rows=len(owner.programme.scales),wide_rows=len(owner.factor.scales),
            global_shared_count=len(owner.shared),fixed_parameter_indices=owner.fixed.tolist(),
            programme_cost=math.fsum(costs[:-1]),wide_cost=costs[-1],combined_cost=math.fsum(costs),
            source_identity=owner.checkpoint_identity(),review=review,media_manifest=self.media,
            initial_J_or_old_source_cache_borrowed=False,formal_optimizer_called=False,**extra)
        write(self.output/'append_QA.json',receipt)
        write(self.output/'status.json',dict(stage='new_source_model_QA_waiting_control',pid=os.getpid(),fit_started=False))
        return receipt

    def run(self,*,wait_for_control=True):
        import z190_joint_source_solver as solver
        from .first_shot_jacobian_reuse import install_source_jacobian_reuse
        from .first_shot_prediction_refresh import install_prediction_refresh
        owner=self.owner
        install_source_jacobian_reuse(owner,cache_directory=self.cache,entry_reference=reference(__file__))
        driver=SimpleNamespace(solve_joint_source=solver.solve_joint_source,solver_runtime=solver)
        install_prediction_refresh(owner,driver=driver,entry_reference=reference(__file__),
            minimum_prediction_ratio=.25,reset_initial_fresh_source_fraction=True)
        kwargs=owner.solver_kwargs();kwargs['source_block_factory']=self.source_blocks
        kwargs['checkpoint_identity'].update(explicit_stage_manifest=self.manifest_reference,
            source_inventory=reference(self.output/'immutable_programme_source_inventory.npz'),
            media_manifest=self.media,completed_append_QA=reference(self.output/'append_QA.json'))
        write(self.output/'run_identity.json',kwargs['checkpoint_identity'])
        if wait_for_control:
            while True:
                path=self.output/'stage_control.json'
                if path.exists():
                    control=read(reference(path))
                    if control.get('action')=='close_owner':return None
                    if control.get('action')=='start_fit':
                        if control.get('append_QA')!=reference(self.output/'append_QA.json')or control.get('q_sha256')!=sha(owner.start):
                            raise ValueError('Formal start must authenticate actual new-source/model QA')
                        break
                time.sleep(2.)
        write(self.output/'status.json',dict(stage='formal_fit_running',pid=os.getpid(),fit_started=True))
        result=driver.solve_joint_source(owner.evaluate,owner.start,owner.quadratic,owner.scales,
            self.output/'public_solver',owner.guard,checkpoint_callback=self.checkpoint,
            source_stage_timing=True,verbose=True,**kwargs)
        np.savez_compressed(self.output/'final_complete_state.npz',**owner.complete_state_payload(result.q))
        write(self.output/'result.json',dict(status=result.status,stop_reason=result.stop_reason,q_sha256=sha(result.q),
            final_complete_state=reference(self.output/'final_complete_state.npz'),
            accepted_iterations=sum(r['kind']=='accepted'for r in self.records),accuracy_accepted=False,video_generated=False))
        return result

    def checkpoint(self,q,event):
        from z190_joint_source_solver import source_cost
        if event['checkpoint_kind']not in ('initial','accepted','resumed'):return None
        owner=self.owner;owner.guard(q);raw=owner.raw_programme(q)
        costs=[source_cost(b.r,b.quadratic,b.loss_weights)for b in owner.source_blocks(q,False)]
        index=sum(r['kind']=='accepted'for r in self.records)+(event['checkpoint_kind']=='accepted')
        folder=self.output/'accepted_states'/('%04d_'%index+sha(q)[:12]);folder.mkdir(parents=True,exist_ok=False)
        canonical=folder/'canonical.npz';canonical.write_bytes(Path(event['checkpoint']).read_bytes())
        with np.load(canonical,allow_pickle=False)as saved:
            if not np.array_equal(saved['q'].view(np.uint64),q.view(np.uint64)):raise ValueError('Public full canonical q changed')
            for key,value in owner.checkpoint_arrays().items():
                if not np.array_equal(saved['aux__'+key],value):raise ValueError('Public full auxiliary changed: '+key)
        np.savez_compressed(folder/'complete_state.npz',**owner.complete_state_payload(q),
            programme_point_errors_px=raw,source_point_ids=self.inventory['source_point_ids'],
            native_pts_per_point=self.inventory['native_pts'])
        review=self.review_export(q,folder/'review');labels=self.inventory['source_target_labels']
        frames=self.inventory['frame_indices'];metrics=[]
        for frame in (201,385,589,842,1239,1248):
            for label in sorted(set(labels[frames==frame])):
                mask=(frames==frame)&(labels==label)
                metrics.append(dict(frame=int(frame),target=str(label),**stats(raw[mask])))
        receipt=dict(schema='rinkspace.accepted-canonical-effective-programme-static-state.v1',
            kind=event['checkpoint_kind'],accepted_index=int(index),qaug_sha256=sha(q),
            source_objective_cost=math.fsum(costs),programme_source_cost=math.fsum(costs[:-1]),wide_source_cost=costs[-1],
            full_programme_raw=stats(raw),selected_programme_target_metrics=metrics,
            all_wide_group_raw_metrics=review['wide_groups'],review=review,
            snapshot=reference(folder/'complete_state.npz'),canonical_checkpoint=reference(canonical),
            source_identity=owner.checkpoint_identity(),event=event,media_manifest=self.media,
            accuracy_accepted=False,all_same_effective_q_scene_visual_review_required=True,
            combined_cost_reduction_is_not_geometric_accuracy_acceptance=True,seconds=time.time()-self.started)
        write(folder/'report.json',receipt);write(self.output/'current.json',receipt)
        self.records.append(dict(kind=receipt['kind'],q_sha256=sha(q),report=reference(folder/'report.json')))
        write(self.output/'accepted_state_index.json',dict(records=self.records))
        print(json.dumps(dict(stage='accepted_canonical_effective_stage_state',kind=receipt['kind'],
            accepted_index=int(index),q_sha256=sha(q),programme_cost=costs[:-1],wide_cost=costs[-1])),flush=True)
        return dict(current_report=reference(folder/'report.json'),canonical_checkpoint=reference(canonical),accuracy_accepted=False)
