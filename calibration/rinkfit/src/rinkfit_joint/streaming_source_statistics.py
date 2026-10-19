"""Memory-bounded task scheduling for the unchanged native source estimator.

Only the frozen channel/polarity/seed/normal/mean projection is retained here;
full source receipts remain in the immutable journal. Workers receive a single
exposure and completed receipts stream directly into the final registry.
"""
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,json,time
from .source_worker_memory import snapshot as memory_snapshot
from .native_source import iter_training_native_rgb
from .io_utils import read_jsonl,write_json,file_sha256
from .reference_statistics_schedule import iter_scheduled_source_rows

def compile_streaming_statistics(run,source_rows,frozen_operator,output,*,frame_callable,trials=256,seed=20261018,workers=2,stop_file=None):
    source=Path(source_rows);out=Path(output);frozen=Path(frozen_operator).resolve()
    if out.exists():raise ValueError('new immutable statistics generation required')
    parent_initial_memory=memory_snapshot();frames=defaultdict(list);frame_sources={};keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','navigation_seed_uv','normal','measurement_operator_resolved','decoded_native_BGR_sha256','joint_material_id','joint_material_index')
    for original in iter_scheduled_source_rows(source):
        if original.get('source_geometry_eligible')is not True:continue
        profile=original['source_measurement_audit']['profile'];row={k:original[k]for k in keep}
        row['source_measurement_audit']={'profile':{k:profile[k]for k in ('channel','polarity')}}
        pts=int(row['native_pts']);frames[pts].append(row)
        if '_statistics_source_journal_ref'in original:frame_sources[pts]=original['_statistics_source_journal_ref']
    if not frames:raise ValueError('no actual source-qualified native measurements')
    parent_projected_memory=memory_snapshot();out.mkdir(parents=True);exposures=len(frames);wanted_all=set(frames);seen=set();completed=0;source_count=0;rank=0;shots=set();pending=[];first=True;priority={min(p for p,rs in frames.items()if rs[0]['shot_id']==sid)for sid in {rs[0]['shot_id']for rs in frames.values()}}
    source_ref=dict(file=str(source.resolve()),sha256=file_sha256(source));projection=dict(schema='rinkfit.exact_source_operator_light_projection.v1',original_source_journal=source_ref,stored_profile_fields=['channel','polarity'],full_profile_receipts_not_duplicated=True,source_normal_seed_mean_nativehash_and_identity_fields_preserved=True)
    registry_path=out/'statistics_registry.json';stop_path=Path(stop_file)if stop_file is not None else out/'STOP_AFTER_CLOSED_EXPOSURE';stopped=False
    def stop_requested():return stop_path.exists()
    with registry_path.open('x',encoding='utf8')as registry:
      registry.write('{')
      def accept(future):
        nonlocal first,completed,source_count,rank
        rec=future.result()
        if rec.get('W'):
            if not first:registry.write(',')
            json.dump(rec['source_group_id'],registry,ensure_ascii=False);registry.write(':');json.dump(rec,registry,ensure_ascii=False,separators=(',',':'),allow_nan=False);registry.flush();first=False
        completed+=1;source_count+=len(rec['row_ids']);rank+=rec['rank'];shots.add(rec['shot_id'])
        print(json.dumps(dict(completed_exposures=completed,source_rows=source_count,shots=sorted(shots),native_pts=rec['native_pts'],rank=rec['rank'],actual_base_replay_max_px=rec['actual_mean_replay_max_px'],source_MC_elapsed_seconds=rec.get('source_MC_elapsed_seconds'))),flush=True)
      with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for wanted in (priority,wanted_all-priority):
          source_iterator=iter(iter_training_native_rgb(run,wanted))
          while True:
            if stop_requested():stopped=True;break
            decode_start=time.perf_counter()
            try:pts,rgb,receipt=next(source_iterator)
            except StopIteration:break
            decode_seconds=time.perf_counter()-decode_start
            rows=frames.pop(pts);seen.add(pts)
            job=dict(rows=rows,rgb=rgb,native_receipt=receipt,out=str(out/(rows[0]['shot_id']+'_'+str(pts))),frozen=str(frozen),trials=trials,seed=seed,source_projection=({**projection,'original_source_journal':frame_sources.pop(pts),'reference_schedule':source_ref}if pts in frame_sources else projection),source_decode_seconds=decode_seconds,submitted_wall_time=time.time())
            pending.append(pool.submit(frame_callable,job));del rows,job
            if len(pending)>=2*workers:accept(pending.pop(0))
          while pending:accept(pending.pop(0))
          if stopped:break
      registry.write('}')
    if completed!=len(seen):raise ValueError('submitted and closed source exposure accounting differs')
    if not stopped and(frames or seen!=wanted_all or completed!=exposures):raise ValueError('source exposure accounting incomplete')
    if stopped:
        write_json(out/'resource_boundary_resume.json',dict(status='intentional_resource_boundary_yield',source_identity_recalled=False,closed_unique_native_pts=sorted(seen),unsubmitted_native_pts=sorted(set(frames)),all_submitted_futures_drained=True,no_incomplete_exposure_counted=True,stop_request=dict(file=str(stop_path.resolve()),sha256=file_sha256(stop_path))))
    report=dict(schema='rinkfit.actual_continuous_source_streaming_conditional_statistics.v1',source_rows=source_count,exposures=completed,shots=sorted(shots),rank=rank,trials=trials,original_UV_and_current_source_mean_preserved=True,source_generation=source_ref,frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)),statistics_registry=dict(file=str(registry_path.resolve()),sha256=file_sha256(registry_path)),source_projection=projection,source_rows_removed_from_parent_memory_on_submission=True,completed_receipts_streamed_to_registry=True,parent_memory_counters=dict(before_projection=parent_initial_memory,after_light_projection=parent_projected_memory,after_all_workers=memory_snapshot()),only_one_exposure_in_each_worker=True,full_RGB_noise_draw_estimator_and_covariance_unchanged=True,physical_total_noise_accuracy_claimed=False)
    report.update(status='intentional_resource_boundary_yield'if stopped else'complete',requested_exposures=exposures,unsubmitted_exposures=len(frames),all_submitted_futures_drained=True,source_identity_recalled=False,stop_after_closed_exposure_supported=True)
    write_json(out/'manifest.json',report);return report
