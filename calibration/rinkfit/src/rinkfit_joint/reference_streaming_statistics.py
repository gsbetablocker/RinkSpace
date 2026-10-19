"""Merge immutable source journals by exposure without an all-row parent cache."""
from pathlib import Path
from collections import Counter
import heapq,json,time,concurrent.futures
from .io_utils import read_jsonl,write_json,file_sha256
from .source_worker_memory import snapshot as memory_snapshot
from .source_readout_identity import actual_readout_key


def _source_frames(entry,aliases):
    source=Path(entry['file']);wanted=set(map(int,entry['native_pts']))
    if file_sha256(source)!=entry['sha256']:raise ValueError('Immutable source journal changed')
    keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','navigation_seed_uv','normal','measurement_operator_resolved','decoded_native_BGR_sha256','joint_material_id','joint_material_index')
    current=None;rows=[];seen=set();row_ids=set()
    ref=dict(file=str(source.resolve()),sha256=entry['sha256'])
    for original in read_jsonl(source):
        pts=int(original['native_pts'])
        if pts not in wanted or original.get('source_geometry_eligible')is not True:continue
        rid=original['source_row_id']
        if rid in aliases:
            a=aliases[rid]
            if int(a['native_pts'])!=pts or actual_readout_key(original)!=a['exact_same_actual_measurement_input_sha256']:raise ValueError('Alias is not this exact source readout')
            continue
        if current is not None and pts!=current:
            if pts<current or current in seen:raise ValueError('Original source exposures are not ordered and contiguous')
            seen.add(current);yield current,rows,ref;rows=[];row_ids=set()
        if rid in row_ids:raise ValueError('Readout repeated within one source exposure')
        row_ids.add(rid);current=pts;row={k:original[k]for k in keep};profile=original['source_measurement_audit']['profile'];row['source_measurement_audit']=dict(profile={k:profile[k]for k in ('channel','polarity')});rows.append(row)
    if current is not None:seen.add(current);yield current,rows,ref
    if seen!=wanted:raise ValueError('A scheduled exposure has no canonical actual source rows')


def _schedule(path):
    source=Path(path);schedule=json.loads(source.read_text(encoding='utf8'))
    if schedule['schema']!='rinkfit.reference_only_native_statistics_schedule.v1':raise ValueError('Wrong reference source schedule')
    ref=schedule['actual_source_alias_index'];indexfile=Path(ref['file'])
    if file_sha256(indexfile)!=ref['sha256']:raise ValueError('Alias index changed')
    index=json.loads(indexfile.read_text(encoding='utf8'))
    for key in ('audit_manifest','aliases','producer_script'):
        r=index[key]
        if file_sha256(r['path'])!=r['sha256']:raise ValueError('Actual alias evidence changed')
    aliases={r['source_row_id']:r for r in read_jsonl(index['aliases']['path'])};wanted=set()
    for e in schedule['source_journals']:
        pts=set(map(int,e['native_pts']))
        if len(pts)!=len(e['native_pts'])or wanted&pts:raise ValueError('An exposure is repeated in the statistics schedule')
        wanted.update(pts)
    if len(wanted)!=schedule['requested_exposures']:raise ValueError('Reference exposure accounting differs')
    return schedule,aliases,wanted


def iter_light_exposures(schedule,aliases):
    """Hold at most one light exposure from each immutable journal."""
    iterators=[];heap=[];previous=None
    try:
        for j,entry in enumerate(schedule['source_journals']):
            iterator=iter(_source_frames(entry,aliases));iterators.append(iterator)
            try:pts,rows,ref=next(iterator)
            except StopIteration:raise ValueError('Empty requested source journal')
            heapq.heappush(heap,(pts,j,rows,ref))
        while heap:
            pts,j,rows,ref=heapq.heappop(heap)
            if previous is not None and pts<=previous:raise ValueError('Reference merge repeats or reorders an exposure')
            previous=pts;yield pts,rows,ref
            try:nxt,rs,sr=next(iterators[j]);heapq.heappush(heap,(nxt,j,rs,sr))
            except StopIteration:pass
    finally:
        for iterator in iterators:iterator.close()


def compile_reference_statistics(run,source_rows,frozen_operator,output,*,frame_callable,trials=256,seed=20261018,workers=2,stop_file=None,in_flight_exposures=None):
    from .native_source import iter_training_native_rgb
    source=Path(source_rows);out=Path(output);frozen=Path(frozen_operator).resolve()
    if out.exists():raise ValueError('New immutable statistics generation required')
    before=memory_snapshot();pending_limit=int(in_flight_exposures)if in_flight_exposures is not None else 2*workers
    if pending_limit<workers:raise ValueError('In-flight exposure count cannot starve requested workers')
    schedule,aliases,wanted_all=_schedule(source);out.mkdir(parents=True);source_ref=dict(file=str(source.resolve()),sha256=file_sha256(source));stop=Path(stop_file)if stop_file else out/'STOP_AFTER_CLOSED_EXPOSURE';seen=set();completed=0;source_count=0;rank=0;shots=set();pending=[];first=True;stopped=False;after_heap=None
    frames=iter(iter_light_exposures(schedule,aliases));registry_path=out/'statistics_registry.json'
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
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
                rgb_iterator=iter(iter_training_native_rgb(run,wanted_all))
                while True:
                    if stop.exists():stopped=True;break
                    begin=time.perf_counter()
                    try:pts,rgb,native=next(rgb_iterator)
                    except StopIteration:break
                    decode_seconds=time.perf_counter()-begin
                    expected,rows,ref=next(frames)
                    if expected!=pts:raise ValueError('Actual native decoder exposure and source merge differ')
                    if after_heap is None:after_heap=memory_snapshot()
                    seen.add(pts);projection=dict(schema='rinkfit.exact_source_operator_light_projection.v1',original_source_journal=ref,reference_schedule=source_ref,stored_profile_fields=['channel','polarity'],full_profile_receipts_not_duplicated=True,source_normal_seed_mean_nativehash_and_identity_fields_preserved=True,parent_keeps_one_light_exposure_per_original_journal=True)
                    job=dict(rows=rows,rgb=rgb,native_receipt=native,out=str(out/(rows[0]['shot_id']+'_'+str(pts))),frozen=str(frozen),trials=trials,seed=seed,source_projection=projection,source_decode_seconds=decode_seconds,submitted_wall_time=time.time());pending.append(pool.submit(frame_callable,job));del rows,job,rgb
                    if len(pending)>=pending_limit:accept(pending.pop(0))
                while pending:accept(pending.pop(0))
        finally:frames.close()
        registry.write('}')
    if completed!=len(seen)or(not stopped and seen!=wanted_all):raise ValueError('Submitted/closed exposure partition differs')
    if stopped:write_json(out/'resource_boundary_resume.json',dict(status='intentional_resource_boundary_yield',source_identity_recalled=False,closed_unique_native_pts=sorted(seen),unsubmitted_native_pts=sorted(wanted_all-seen),all_submitted_futures_drained=True,no_incomplete_exposure_counted=True,stop_request=dict(file=str(stop.resolve()),sha256=file_sha256(stop))))
    report=dict(schema='rinkfit.reference_merged_native_source_conditional_statistics.v1',status='intentional_resource_boundary_yield'if stopped else'complete',source_rows=source_count,exposures=completed,requested_exposures=len(wanted_all),unsubmitted_exposures=len(wanted_all-seen),shots=sorted(shots),rank=rank,trials=trials,source_generation=source_ref,frozen_operator=dict(file=str(frozen),sha256=file_sha256(frozen)),statistics_registry=dict(file=str(registry_path.resolve()),sha256=file_sha256(registry_path)),parent_memory_counters=dict(before_schedule=before,heap_loaded=after_heap,after_all_workers=memory_snapshot()),all_submitted_futures_drained=True,stop_after_closed_exposure_supported=True,source_identity_recalled=False,only_one_exposure_in_each_worker=True,all_row_parent_cache_created=False,one_light_exposure_per_source_journal=True,source_profiles_or_arrays_copied=False,full_RGB_noise_draw_estimator_and_covariance_unchanged=True,physical_total_noise_accuracy_claimed=False)
    write_json(out/'manifest.json',report);return report
