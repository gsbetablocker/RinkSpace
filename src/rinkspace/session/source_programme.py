"""Resumable programme source reading and camera-only native GPU navigation."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from fractions import Fraction
import gzip, hashlib, json, os, sqlite3, time
import numpy as np
from .dataset import write_json, digest
from .source_inputs import SourceEvidenceIndex, SourceSupportResolver, source_job_contract, job_reference_path
from .source_observations import (SourceCandidate, normalize_source, source_queries,
    qualify_measurement, feature_registry, export_input)
from .source_navigation import project_navigation, initial_stations
from .source_qualification import build_qualification_index, apply_qualification


def _json_safe(value):
    from ..vision import _source_json
    return _source_json(value)


def intern_candidate_authorities(candidates):
    """Store each complete frame-bound proof once; preserve every candidate."""
    records={};identities={}
    for candidate in candidates:
        authority=candidate.evidence.get('identity_authority')
        if not authority:continue
        pointer=id(authority)
        if pointer not in identities:
            encoded=_json_safe(authority)
            key=hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            identities[pointer]=key;records.setdefault(key,encoded)
        candidate.evidence['identity_authority_ref']='frame_identity_authorities/'+identities[pointer]
        candidate.evidence.pop('identity_authority')
    return records


def _process_memory_bytes():
    import ctypes as c
    from ctypes import wintypes
    class Counters(c.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,c.c_size_t) for k in
            ('peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    info=Counters();info.cb=c.sizeof(info)
    handle=c.windll.kernel32.GetCurrentProcess
    handle.restype=wintypes.HANDLE
    query=c.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[wintypes.HANDLE,c.POINTER(Counters),wintypes.DWORD]
    if not query(handle(),c.byref(info),c.sizeof(info)):raise OSError('GetProcessMemoryInfo failed')
    return int(info.rss)


class ProgrammeLedger:
    def __init__(self, root, job_path, *, previous_job=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'programme.sqlite')
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS revisions(shot TEXT,pts INTEGER,revision INTEGER,path TEXT,sha256 TEXT,reason TEXT,PRIMARY KEY(shot,pts,revision));
            CREATE TABLE IF NOT EXISTS frames(shot TEXT,pts INTEGER,path TEXT,sha256 TEXT,rgb_sha256 TEXT,
                candidate_count INTEGER,measured_count INTEGER,identity_count INTEGER,
                PRIMARY KEY(shot,pts));
        """)
        pin=digest(job_path)
        original=self.db.execute("select value from meta where key='job_sha256'").fetchone()
        old=self.db.execute("select value from meta where key='current_job_sha256'").fetchone() or original
        if old and old[0]!=pin:
            try:
                prior=Path(previous_job) if previous_job is not None else None
                prior_pin=digest(prior) if prior is not None else None
                if prior_pin not in {old[0],original[0]}:
                    raise ValueError('Resume job revision differs; provide its authenticated previous job')
                current=json.loads(Path(job_path).read_text(encoding='utf-8-sig'))
                original=json.loads(prior.read_text(encoding='utf-8-sig'))
                if source_job_contract(current)!=source_job_contract(original):
                    raise ValueError('Resume revision changes the original source or exposure scope')
                record=dict(job_path=str(Path(job_path).resolve()),job_sha256=pin,
                    previous_job_path=str(prior.resolve()),previous_job_sha256=prior_pin,
                    original_frames_preserved=True,previous_task_instructions_used=False,
                    old_navigation_is_not_current_geometry_validation=True)
                self.db.execute('insert or ignore into meta values(?,?)',
                    ('continuation_job/'+pin,json.dumps(record,sort_keys=True,separators=(',',':'))))
            except BaseException:
                self.db.close();raise
        # Keep the initial job pin as provenance, and remember the authorized
        # current revision so its next restart needs no obsolete job file.
        self.db.execute("insert or ignore into meta values('job_sha256',?)",(pin,))
        self.db.execute("insert or replace into meta values('current_job_sha256',?)",(pin,))
        self.db.commit()

    def commit(self, shot, pts, payload):
        data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        name=f'frames/{shot}/{pts}.json.gz';path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if self.db.execute('select 1 from frames where shot=? and pts=?',(shot,pts)).fetchone():
            raise ValueError('Native exposure already committed')
        raw=gzip.compress(data,compresslevel=3,mtime=0)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        candidates=payload['candidates']
        self.db.execute('insert into frames values(?,?,?,?,?,?,?,?)',(shot,pts,name,hashlib.sha256(raw).hexdigest(),
            payload['source_binding']['decoded_native_rgb_sha256'],len(candidates),
            sum(c['localization']=='measured' for c in candidates),sum(c['identity']=='confirmed' for c in candidates)))
        self.db.commit()

    def read(self, shot, pts):
        row=self.db.execute('select path,sha256 from frames where shot=? and pts=?',(shot,pts)).fetchone()
        if row is None:return None
        revised=self.db.execute('select path,sha256 from revisions where shot=? and pts=? order by revision desc limit 1',(shot,pts)).fetchone()
        if revised:row=revised
        raw=(self.root/row[0]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row[1]:raise ValueError('Committed source frame changed')
        return json.loads(gzip.decompress(raw))

    def revise(self,shot,pts,payload,reason):
        prior=self.read(shot,pts)
        if prior is None or prior['source_binding']!=payload['source_binding']:
            raise ValueError('A derived measurement revision must retain original exposure identity')
        number=self.db.execute('select coalesce(max(revision),0)+1 from revisions where shot=? and pts=?',(shot,pts)).fetchone()[0]
        name=f'revisions/{shot}/{pts}_{number}.json.gz';path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        payload['measurement_revision']=dict(number=number,reason=reason,original_frame_preserved=True)
        raw=gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),compresslevel=3,mtime=0)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        self.db.execute('insert into revisions values(?,?,?,?,?,?)',(shot,pts,number,name,hashlib.sha256(raw).hexdigest(),reason))
        self.db.commit()

    def qualify_existing(self):
        changed=0
        for shot,pts in self.db.execute('select shot,pts from frames order by pts').fetchall():
            value=self.read(shot,pts)
            if value.get('identity_policy_revision')==2:continue
            candidates=value['candidates'];known={c['candidate_id']:c for c in candidates}
            authorities=value.get('frame_identity_authorities',{})
            for c in candidates:
                a=c['evidence'].get('identity_authority')
                ref=c['evidence'].get('identity_authority_ref')
                if ref:a=authorities.get(ref.split('/')[-1])
                if not a or a.get('kind')!='authenticated_existing_source_material_identity':continue
                real_human=any(known.get(r['id'],{}).get('review',{}).get('human_source_ref') for r in a.get('source_records',[]))
                if not real_human and not a.get('independent_identity_basis'):
                    c['evidence']['prior_machine_identity']=c['identity']
                    c['identity']='unresolved'
                    c['evidence']['identity_gap']='old_material_name_without_independent_source_identity'
            value['identity_policy_revision']=2
            self.revise(shot,pts,value,'strict_independent_source_identity_requalification');changed+=1
        return changed

    def candidates(self, shot=None):
        sql='select shot,pts from frames';args=()
        if shot:sql+=' where shot=?';args=(shot,)
        for sid,pts in self.db.execute(sql+' order by pts',args).fetchall():
            for row in self.read(sid,pts)['candidates']:yield SourceCandidate(**row)


def navigation_dependencies(ledger, shot, geometry, camera_path, scope_path):
    """Bind navigation to the latest saved observations and current native files."""
    from ..vision_native import file_ref
    geometry.assert_unchanged()
    frames=[]
    for pts,path,sha in ledger.db.execute(
            'select pts,path,sha256 from frames where shot=? order by pts',(shot,)).fetchall():
        revised=ledger.db.execute('select path,sha256 from revisions where shot=? and pts=? '
            'order by revision desc limit 1',(shot,pts)).fetchone()
        if revised:path,sha=revised
        if digest(ledger.root/path)!=sha:raise ValueError('Committed source frame changed')
        frames.append(dict(pts=pts,path=path,sha256=sha))
    camera_path=Path(camera_path)
    camera=json.loads(camera_path.read_text(encoding='utf-8-sig'))
    encoded=json.dumps(frames,sort_keys=True,separators=(',',':')).encode()
    return dict(shot_id=shot,native_geometry=geometry.refs,
        source_frame_count=len(frames),measurement_revision_sha256=hashlib.sha256(encoded).hexdigest(),
        camera_seed=file_ref(camera_path),camera_controls=file_ref(camera_path.parent/camera['controls_ref']),
        scope=file_ref(scope_path))


def navigation_result_target(root, shot, dependencies):
    """Reuse only authenticated results; keep every stale or interrupted run."""
    folder=Path(root)/'navigation'/shot
    token=hashlib.sha256(json.dumps(dependencies,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    original=folder/'result.json'
    matches=[original,*sorted((folder/'revisions').glob(token+'*/result.json'))]
    reason='no_saved_navigation'
    for path in matches:
        if not path.is_file():continue
        try:value=json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError,UnicodeError):
            reason='saved_navigation_result_unreadable';continue
        if value.get('source_dependencies')!=dependencies:
            reason='saved_navigation_dependencies_missing_or_changed';continue
        outputs=value.get('output_sha256',{})
        if (set(outputs)=={'controls.npy','stations.npy','camera.json'} and
                all((path.parent/name).is_file() and digest(path.parent/name)==sha for name,sha in outputs.items())):
            return path,True,'exact_saved_navigation_dependencies_and_outputs'
        reason='saved_navigation_outputs_missing_or_changed'
    if not folder.exists():return original,False,reason
    target=folder/'revisions'/token
    if target.exists():target=target.with_name(token+'_'+str(time.time_ns()))
    return target/'result.json',False,reason


def _verified_job(job_path):
    path=Path(job_path).resolve();job=json.loads(path.read_text(encoding='utf-8-sig'))
    for name in ('scope_ref','scene_ref','source_routes_ref','navigation_ref','vision_config_ref','source_reidentification_ref'):
        ref=job[name]
        if digest(job_reference_path(path,job,name))!=ref['sha256']:raise ValueError('Job input changed: '+name)
    if digest(job['source']['registered_path'])!=job['source']['sha256']:raise ValueError('Original video SHA differs')
    scope=json.loads(job_reference_path(path,job,'scope_ref').read_text(encoding='utf-8-sig'))
    from .source_read import _validated_sequences
    sequences=_validated_sequences(job,scope)
    selected=[batch['shot_id'] for batch,_ in sequences]
    if (len(set(selected))!=len(selected) or selected!=job['selected_shot_ids']):
        raise ValueError('Programme sequences must contain each selected shot exactly once')
    for key in ('sha256','width','height'):
        if job['source'].get(key)!=scope['source'].get(key):
            raise ValueError('Programme source differs from its selected scope: '+key)
    clock=job['source_clock'];expected_clock=scope['source_clock']
    for key in ('time_base_num','time_base_den','native_pts_step'):
        if type(clock.get(key)) is not int or clock[key]<=0:
            raise ValueError('Programme native clock must contain positive integers')
    if (Fraction(clock['time_base_num'],clock['time_base_den'])!=
            Fraction(expected_clock['time_base_num'],expected_clock['time_base_den']) or
            clock['native_pts_step']!=expected_clock['native_pts_step']):
        raise ValueError('Programme native clock differs from its selected scope')
    for batch,_ in sequences:
        count=(batch['stop_pts_exclusive']-batch['first_pts'])//clock['native_pts_step']
        if type(batch.get('native_frame_count')) is not int or batch['native_frame_count']!=count:
            raise ValueError('Requested exposure registry differs from its native interval')
    return path,job


def qualify_source_identities(source, fresh, queries):
    """Keep saved identity and withdrawals within their actual point scope."""
    authorities={}
    for guide in queries:
        if guide.get('identity_confirmed') is not True:continue
        authority=guide.get('identity_authority') or {}
        for record in authority.get('source_records',[]):
            authorities[(guide['id'],record['id'])]=authority
    for candidate in source:
        authority=authorities.get((candidate.native_path_id,candidate.candidate_id))
        if candidate.identity=='named_source_candidate' and not candidate.withdrawal and authority:
            candidate.identity='confirmed';candidate.evidence['identity_authority']=authority
    withdrawn={c.candidate_id for c in source if c.withdrawal}
    def support_ids(value):
        if isinstance(value,dict):
            for row in value.get('source_records',[]):
                if isinstance(row,dict) and row.get('id') is not None:yield row['id']
            for key,child in value.items():
                if key not in ('source_records','withdrawals','withdrawal'):
                    yield from support_ids(child)
        elif isinstance(value,(list,tuple)):
            for child in value:yield from support_ids(child)
    for candidate in fresh:
        dependencies=withdrawn.intersection(support_ids(candidate.evidence.get('identity_authority')))
        if dependencies:
            candidate.identity='unresolved'
            candidate.evidence['prior_identity_withdrawal_requires_independent_reassociation']=True
            candidate.evidence['withdrawn_identity_support_ids']=sorted(dependencies)


def process_exposure(reader, index, resolver, rgb, binding, navigation_state, *, reset=False):
    """Shared exact-RGB frame processing. Discovery and queries use one reader."""
    from ..vision_association import associate,extend_feedback_scope
    geometry=reader.native_geometry
    started=time.perf_counter();source=[]
    envelopes=list(index.frame(binding['shot_id'],binding['source_pts']))
    envelopes.sort(key=lambda e:json.dumps(e.source_payload.get('source_pixel_or_annotation_ref',{}),sort_keys=True))
    for envelope in envelopes:source.extend(normalize_source(envelope,resolver,geometry))
    withdrawals=apply_qualification(index,source,binding["shot_id"],binding["source_pts"])
    source_time=time.perf_counter()-started
    proposals=project_navigation(geometry,navigation_state,binding['source_pts'],dict(search_radius_px=18.))
    navigation_time=time.perf_counter()-started-source_time
    saved=source_queries(source,binding,geometry)
    context=reader.begin_frame(rgb,binding,reset=reset,extract_source_components=True)
    frontend_time=time.perf_counter()-started-source_time-navigation_time
    continued=[] if reset else [dict(g,candidate_kind='verified_temporal')
        for g in reader.continuation_queries(context)]
    dispatched=associate(context,[*continued,*saved,*proposals],geometry)
    extend_feedback_scope(context,[g['id'] for g in dispatched['queries']])
    result=reader.read_queries(context,dispatched['queries'])
    result=reader.commit_frame(context)
    fresh=[]
    for reading in result['guided']['readings']:
        identity=bool(reading.get('physical_id')) and bool(reading.get('identity_authority'))
        fresh.extend(qualify_measurement(context,reading,dict(identity_confirmed=identity,
            identity_authority=reading.get('identity_authority'))))
    # Existing real support remains in full. A new RGB reread is a new candidate,
    # not an overwrite of its human/source ancestor or a second independent W.
    qualify_source_identities(source,fresh,dispatched['queries'])
    candidates=source+fresh
    for c in candidates:
        if c.identity!='confirmed' or c.actual_uv is None or c.native_path_id is None:continue
        try:c.evidence['station_initial']=float(initial_stations(proposals,c.native_path_id,[c.actual_uv])[0])
        except ValueError as exc:c.evidence['navigation_gap']=str(exc)
    authority_records=intern_candidate_authorities(candidates)
    timings=dict(source_adapter_seconds=source_time,navigation_seconds=navigation_time,frontend_seconds=frontend_time,total_seconds=time.perf_counter()-started,
        frontend_calls=context['frontend_calls'],query_rounds=context.get('rounds'),temporal_commits=context['commits'])
    named_counts={}
    for reading in result['guided']['readings']:
        name=reading['expected_id']
        named_counts[name]=named_counts.get(name,0)+len(reading['actual_source_uv'])
    return dict(source_binding=binding,identity_policy_revision=2,candidates=[vars(c) for c in candidates],frame_identity_authorities=authority_records,
        association=dispatched['evidence'],unresolved=dispatched['unresolved'],
        actual_reader_summary=dict(memory_update=result['memory_update'],
            source_feedback=_json_safe(result.get('source_feedback')),
            named_reading_counts=named_counts,
            source_model_sha256=result['model_sha256'],native_geometry=result['native_geometry']),
        timing=timings,source_point_status='machine_result_pending_scoped_visual_review')


def run_programme(job_path, out, *, resume=False, interrupt_after_commit=None, previous_job=None):
    import av, torch
    from ..vision import SourceRinkReader
    from ..vision_native import NativeSourceGeometry
    from ..source_binding import canonical_source_binding
    from .prepare import prepare
    from .gpu_problem import compile_navigation_problem
    path,job=_verified_job(job_path);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=ProgrammeLedger(out,path,previous_job=previous_job)
    if not resume and ledger.db.execute('select count(*) from frames').fetchone()[0]:raise ValueError('Use --resume for committed exposures')
    ledger.qualify_existing()
    log=(out/'programme.log').open('a',encoding='utf-8',buffering=1)
    def report(event):
        record=dict(unix_time=time.time(),**_json_safe(event));line=json.dumps(record,ensure_ascii=True,allow_nan=False)
        log.write(line+'\n');print(line,flush=True)
    print('MAIN LOG: '+str((out/'programme.log').resolve()),flush=True)
    library=Path(job['roots']['data'])/'build/native_s01_audit_20261027/rinkspace_core.dll'
    geometry=NativeSourceGeometry(job_reference_path(path,job,'scene_ref'),library)
    if not all(hasattr(geometry.core._lib,s) for s in ('rs_g1_create','rs_g3_attach','rs_g4_solve','rs_g2_sparse_info')):
        raise ValueError('Current native GPU capabilities unavailable')
    scope=json.loads(job_reference_path(path,job,'scope_ref').read_text(encoding='utf-8-sig'))
    config=json.loads(job_reference_path(path,job,'vision_config_ref').read_text(encoding='utf-8-sig'))
    # Preserve the selected reader's explicit candidate-admission setting.
    clock=Fraction(job['source_clock']['time_base_num'],job['source_clock']['time_base_den'])
    step=job['source_clock']['native_pts_step']
    expected_total=sum(s['native_frame_count'] for s in job['sequences'])
    runtime_config=out/'vision_runtime.json';write_json(runtime_config,config)
    reader=SourceRinkReader(runtime_config,native_geometry=geometry,programme_scope=scope,
        source_reidentification=job_reference_path(path,job,'source_reidentification_ref'))
    source_index=SourceEvidenceIndex(path,out/'source_index.sqlite',reuse_from_job=previous_job)
    build_qualification_index(source_index,report)
    resolver=SourceSupportResolver(source_index.inputs,out/'source_support.sqlite')
    navpath=job_reference_path(path,job,'navigation_ref');navroot=navpath.parent
    nav=json.loads(navpath.read_text(encoding='utf-8-sig'))
    camera_refs={v['shot_id']:v for v in nav['cameras']}
    started=time.perf_counter();commits=0;navigation_results={}
    for shot in job['sequences']:
        sid=shot['shot_id'];camera_path=navroot/camera_refs[sid]['camera_ref']
        camera=json.loads(camera_path.read_text(encoding='utf-8-sig'));controls=np.load(camera_path.parent/camera['controls_ref'],allow_pickle=False)
        state=dict(camera=camera,controls=controls)
        done={p for p, in ledger.db.execute('select pts from frames where shot=?',(sid,))}
        expected=list(range(shot['first_pts'],shot['stop_pts_exclusive'],job['source_clock']['native_pts_step']))
        missing=[p for p in expected if p not in done]
        if missing:
            start=missing[0];warm=start-job['source_clock']['native_pts_step'] if start>shot['first_pts'] else start
            if warm not in done:warm=start
            decoded=[];first=True
            report(dict(stage='shot_source_start',shot=sid,start_pts=start,warm_restart_anchor_pts=warm,exact_reader_memory_resume=False))
            with av.open(job['source']['registered_path']) as container:
                stream=container.streams.video[0]
                if Fraction(stream.time_base)!=clock:raise ValueError('Actual source time base differs')
                container.seek(warm,stream=stream,backward=True)
                for frame in container.decode(stream):
                    if frame.pts<warm:continue
                    if frame.pts>=shot['stop_pts_exclusive']:break
                    pts=int(frame.pts)
                    if pts!=warm+len(decoded)*step:raise ValueError('Missing actual native source exposure')
                    decoded.append(pts);rgb=frame.to_ndarray(format='rgb24')
                    sha=hashlib.sha256(rgb.tobytes()).hexdigest()
                    binding=canonical_source_binding(dict(source_sha256=job['source']['sha256'],source_pts=pts,time_base=str(clock),
                        shot_id=sid,native_hw=list(rgb.shape[:2]),decoded_native_rgb_sha256=sha,evaluation_role='development'),scope)
                    record=process_exposure(reader,source_index,resolver,rgb,binding,state,reset=first);first=False
                    if pts in done:
                        if ledger.read(sid,pts)['source_binding']['decoded_native_rgb_sha256']!=sha:raise ValueError('Warm replay original RGB differs')
                        report(dict(stage='warm_restart_anchor_reread',shot=sid,pts=pts,duplicate_measurement_committed=False));continue
                    ledger.commit(sid,pts,record);commits+=1
                    if True:
                        report(dict(stage='source_frame_committed',shot=sid,pts=pts,source_seconds=float(pts*clock),
                            committed_total=ledger.db.execute('select count(*) from frames').fetchone()[0],
                            measured=sum(c['localization']=='measured' for c in record['candidates']),
                            identified=sum(c['identity']=='confirmed' for c in record['candidates']),
                            frame_seconds=record['timing']['total_seconds'],stage_timing=record['timing'],rss_bytes=_process_memory_bytes(),
                            cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_bytes=torch.cuda.max_memory_allocated()))
                    if interrupt_after_commit is not None and commits==interrupt_after_commit:
                        report(dict(stage='real_interruption_after_durable_frame',shot=sid,pts=pts));raise KeyboardInterrupt()
        verified=[p for p, in ledger.db.execute('select pts from frames where shot=? order by pts',(sid,))]
        if verified!=expected:raise ValueError('Shot has missing committed native exposures')
        scope_path=job_reference_path(path,job,'scope_ref')
        dependencies=navigation_dependencies(ledger,sid,geometry,camera_path,scope_path)
        accepted,reused,reuse_reason=navigation_result_target(out,sid,dependencies)
        report(dict(stage='native_navigation_resume_check',shot=sid,reused=reused,
            reason=reuse_reason,result=str(accepted),original_navigation_preserved=True))
        if reused:
            navigation_results[sid]=str(accepted)
            continue
        revision=out/'inputs'/sid
        if revision.exists():revision=out/'inputs'/(sid+'_'+str(time.time_ns()))
        data=dict(geometry=geometry,job=job,decoded_pts=[p for p, in ledger.db.execute('select pts from frames order by pts')],
            navigation_dir=navroot,index_ref=str(source_index.path),candidate_ledger_ref=str(out/'programme.sqlite'))
        inp=export_input(ledger.candidates(sid),feature_registry(geometry),data,None,'camera_navigation',revision)
        prepared=revision/'prepared';prepare(inp,scope_path,[sid],prepared,core=geometry.core)
        report(dict(stage='native_gpu_navigation_start',shot=sid,prepared=str(prepared),main_log=str(out/'programme.log')))
        geometry.assert_unchanged()
        compiled=compile_navigation_problem(prepared,revision/'navigation_local_state.json',shot_id=sid,core=geometry.core)
        resident=compiled['resident']
        try:
            before=resident.evaluate('cuda')
            solved=resident.solve(progress=lambda r:report(dict(stage='native_gpu_iteration',shot=sid,**r)))
            updated=resident.export_state();after=resident.evaluate('cuda')
            target=accepted.parent;target.mkdir(parents=True,exist_ok=True)
            np.save(target/'controls.npy',updated['controls']);np.save(target/'stations.npy',updated['stations'])
            write_json(target/'camera.json',dict(camera,controls_ref='controls.npy'))
            if navigation_dependencies(ledger,sid,geometry,camera_path,scope_path)!=dependencies:
                raise ValueError('Source or native dependencies changed during navigation; saved state is not a current result')
            write_json(accepted,dict(shot_id=sid,solve=solved,setup=compiled['setup'],plan=compiled['plan'],
                source_rows=len(compiled['source_rows']),actual_source_UV_entered_GPU=True,frozen=compiled['frozen'],
                before_valid=int((before['status']==0).sum()),after_valid=int((after['status']==0).sum()),
                prepared=str(prepared),camera_seed=str(camera_path),user_approval='not_reviewed',
                source_dependencies=dependencies,output_sha256={name:digest(target/name) for name in ('controls.npy','stations.npy','camera.json')}))
            navigation_results[sid]=str(accepted)
            report(dict(stage='native_gpu_navigation_done',shot=sid,**solved))
        finally:resident.close()
    geometry.assert_unchanged()
    total=ledger.db.execute('select count(*) from frames').fetchone()[0]
    if total!=expected_total:raise ValueError('Complete target exposure count not reached')
    result=dict(status='machine_result_pending_scoped_visual_review',processed_native_exposures=total,
        navigation_results=navigation_results,
        shot_counts=ledger.db.execute('select shot,count(*),sum(measured_count),sum(identity_count) from frames group by shot').fetchall(),
        wall_seconds=time.perf_counter()-started,main_log=str(out/'programme.log'),geometry_unchanged=True)
    from ..review.source_programme import render_programme
    result['review']=render_programme(job,ledger,geometry,out/'navigation',out/'review',report,
        navigation_results=navigation_results,job_path=path)
    write_json(out/'result.json',result);return result
