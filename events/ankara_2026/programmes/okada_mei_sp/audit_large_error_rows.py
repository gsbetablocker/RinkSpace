"""Build a pending per-observation source review ledger; never alter solve inputs."""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,io,json,time
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def npz_cache(path):
    # Never index an NpzFile inside a per-observation loop: cache each array once.
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def pending():
    return dict(status='pending',action=None,entity_id=None,evidence_paths=[],comment=None)


def band(error):
    return 'gt80' if error>80 else ('gt40_le80' if error>40 else 'gt20_le40')


def select_jsonl(path,requests,consume,expected_hashes=None):
    """Read one source file once, retaining only requested lines/records."""
    found=set();sha=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line_number,raw in enumerate(stream,1):
            sha.update(raw)
            if line_number in requests:
                consume(line_number,json.loads(raw),hashlib.sha256(raw).hexdigest(),requests[line_number])
                found.add(line_number)
    if found!=set(requests):
        raise ValueError('Missing requested JSONL lines: '+str(path))
    actual=sha.hexdigest()
    if expected_hashes and any(value!=actual for value in expected_hashes):
        raise ValueError('Source JSONL SHA256 differs: '+str(path))
    return dict(path=str(path),sha256=actual,selected_lines=len(found))


def build(args):
    started=time.monotonic();root=args.prepared.resolve(strict=True);out=args.output.resolve()
    targets=[out/name for name in ('observations.jsonl','groups.json','summary.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve existing review/decisions: '+str(out))
    if any(len(str(path).encode('utf-16-le'))//2>=260 or len(path.name)>255 for path in targets):
        raise ValueError('Windows output path limit')
    manifests={};origins={};packet_maps={}
    def manifest(folder):
        key=str(folder)
        if key not in manifests:manifests[key]=read(folder/'joint_manifest.json')
        return manifests[key]
    def origin_arrays(folder):
        key=str(folder)
        if key not in origins:origins[key]=npz_cache(folder/'source_origins.npz')
        return origins[key]
    m=manifest(root);mapping=read(root/'mapping.json')
    source_file=root/'point_observations.npz';points=npz_cache(source_file)
    if digest(source_file)!=m['point_input_sha256']:
        raise ValueError('Prepared source input SHA256 differs')
    residual_raw=args.residuals.resolve(strict=True).read_bytes()
    residual_sha=hashlib.sha256(residual_raw).hexdigest()
    with np.load(io.BytesIO(residual_raw),allow_pickle=False) as archive:
        residuals={key:archive[key] for key in archive.files}
    del residual_raw
    rr=residuals['rows'];ee=residuals['errors']
    if rr.ndim!=1 or ee.shape!=rr.shape or rr.dtype.kind not in 'iu':
        raise ValueError('Invalid residual row/error arrays')
    if np.any(rr<0) or np.any(rr>=len(points['pts'])) or len(np.unique(rr))!=len(rr):
        raise ValueError('Residual rows are invalid or repeated')
    hot=np.flatnonzero(np.isfinite(ee)&(ee>20.));hot=hot[np.argsort(ee[hot],kind='stable')[::-1]]
    rows=rr[hot];errors=ee[hot];parent=args.parent.resolve(strict=True)
    declared=Path(m['source_group_update']['parent_root']).resolve()
    if parent!=declared:raise ValueError('Requested parent differs from prepared parent')
    # Only IDs are needed from the parent archive; they are read and cached once.
    with np.load(parent/'point_observations.npz',allow_pickle=False) as archive:
        parent_ids=archive['observation_ids']
    old_id_membership=np.isin(points['observation_ids'][rows],parent_ids)
    shots=m['shots'];shot_stops=np.array([s['row_stop'] for s in shots]);records=[]
    packet_requests=defaultdict(lambda:defaultdict(list));packet_meta={}
    for j,(ri,error) in enumerate(zip(rows,errors)):
        ri=int(ri);shot=shots[int(np.searchsorted(shot_stops,ri,side='right'))]
        if not shot['row_begin']<=ri<shot['row_stop']:raise ValueError('Unassigned source row')
        pts=int(points['pts'][ri]);pid=mapping['path_ids'][int(points['path_indices'][ri])]
        direct=origin_arrays(root);pr=int(direct['parent_row_indices'][ri]);in_parent=bool(old_id_membership[j])
        if pr>=0 and str(parent_ids[pr])!=str(points['observation_ids'][ri]):
            raise ValueError('Parent row/observation ID differs')
        record=dict(ledger_line=j+1,row=ri,residual_array_index=int(hot[j]),shot=shot['id'],camera_index=int(shot['camera_index']),
            pts=pts,source_seconds=pts*shot['time_base_num']/shot['time_base_den'],time_base=[shot['time_base_num'],shot['time_base_den']],
            path_id=pid,observation_id=str(points['observation_ids'][ri]),uv=points['uv'][ri].tolist(),kind=int(points['kind'][ri]),
            error_px=float(error),error_band=band(float(error)),source_row=int(points['source_rows'][ri]),
            parent_old_vs_new='parent_old' if in_parent else 'new_to_parent',present_in_current_parent_ids=in_parent,
            immediate_parent_row=pr,group_id=f"{shot['id']}:{pts}:{pid}",frame_id=f"{shot['id']}:{pts}",decision=pending(),source_origin_chain=[])
        folder=root;index=ri;seen=set()
        while True:
            key=(str(folder),index)
            if key in seen:raise ValueError('Source origin cycle')
            seen.add(key);mm=manifest(folder);origin_file=folder/'source_origins.npz'
            if not origin_file.is_file():
                record['source_origin_locator']=dict(kind='prepared_input_row_without_further_origin_array',path=str(folder/'point_observations.npz'),row=index,
                    point_input_sha256=mm.get('point_input_sha256'),original_source_input=mm.get('source_inputs',{}).get(shot['id']))
                break
            oo=origin_arrays(folder);pindex=int(oo.get('packet_indices',np.full(0,-1))[index]) if 'packet_indices'in oo else -1
            prow=int(oo['parent_row_indices'][index]);packet_row=int(oo['packet_rows'][index]) if 'packet_rows'in oo else -1
            record['source_origin_chain'].append(dict(prepared_root=str(folder),row=index,parent_row=prow,packet_index=pindex,packet_row=packet_row))
            update=mm.get('source_group_update',{})
            if pindex>=0:
                info=update['packet_inputs'][pindex];packet=Path(info['path']).resolve(strict=True);record['source_origin_locator']=dict(kind='source_packet_row',packet_path=str(packet),packet_index=pindex,packet_row=packet_row,prepared_root=str(folder),source_capture=info.get('source_capture'),source_capture_sha256=info.get('source_capture_sha256'))
                packet_requests[str(packet)][packet_row+1].append(j);packet_meta[str(packet)]=info
                break
            if prow<0 or not update.get('parent_root'):
                record['source_origin_locator']=dict(kind='unresolved_saved_origin',prepared_root=str(folder),row=index,reason='No packet and no parent row/root')
                break
            folder=Path(update['parent_root']).resolve(strict=True);index=prow
        records.append(record)
    print(json.dumps(dict(stage='indexed',rows=len(records),frames=len({r['frame_id']for r in records}),packets=len(packet_requests)),ensure_ascii=False),flush=True)
    qualified_requests=defaultdict(lambda:defaultdict(list));qualified_hashes=defaultdict(set);verified=[]
    for packet_text,requests in packet_requests.items():
        packet=Path(packet_text);pm=read(packet/'mapping.json');packet_maps[packet_text]=pm
        def consume(number,loc,line_sha,indices):
            for j in indices:
                record=records[j];expected=record['source_origin_locator']['packet_row']
                if int(loc.get('packet_row',expected))!=expected or loc['observation_id']!=record['observation_id'] or int(loc['pts'])!=record['pts'] or loc['path_id']!=record['path_id'] or not np.array_equal(loc['uv'],record['uv']):
                    raise ValueError('Packet provenance differs from current observation')
                record['packet_manifest_locator']=dict(path=str(packet/pm.get('source_row_manifest','source_row_manifest.jsonl')),line=number,source_manifest_record=loc)
                source=loc['source_jsonl'];qualified_requests[source][int(loc['source_line'])].append((j,loc));qualified_hashes[source].add(loc['source_jsonl_sha256'])
        f=packet/pm.get('source_row_manifest','source_row_manifest.jsonl')
        verified.append(select_jsonl(f,requests,consume,{pm['source_row_manifest_sha256']} if pm.get('source_row_manifest_sha256')else None))
    raw_requests=defaultdict(lambda:defaultdict(list));raw_hashes=defaultdict(set)
    for source,requests in qualified_requests.items():
        def consume(number,frame,line_sha,items):
            for j,loc in items:
                record=records[j]
                if loc.get('source_line_sha256') and loc['source_line_sha256']!=line_sha:raise ValueError('Qualified source line hash differs')
                row=frame['rows'][int(loc['row_index'])]
                if frame['pts']!=record['pts'] or row['path_id']!=record['path_id'] or not np.array_equal(row['uv'],record['uv']):raise ValueError('Qualified source measurement differs')
                record['source_measurement_record']=row;record['source_BGR_sha256']=frame.get('source_BGR_sha256');record['source_video_sha256']=frame.get('source_video_sha256')
                record['source_registration_label']=row.get('source_registration_label')
                record['source_capture_root']=row.get('source_family_review',{}).get('capture_root')
                locator=row.get('source_readout_locator')
                if isinstance(locator,dict) and locator.get('path') and locator.get('line'):
                    record['raw_readout_locator']=locator;raw_requests[locator['path']][int(locator['line'])].append((j,locator));
                    if locator.get('sha256'):raw_hashes[locator['path']].add(locator['sha256'])
                else:
                    record['registration_context_status']='no_separate_readout_locator_in_saved_source_row'
        verified.append(select_jsonl(source,requests,consume,qualified_hashes[source]))
    print(json.dumps(dict(stage='qualified_sources_resolved',rows_with_source_measurement=sum('source_measurement_record'in r for r in records),raw_files=len(raw_requests)),ensure_ascii=False),flush=True)
    for source,requests in raw_requests.items():
        def consume(number,frame,line_sha,items):
            notes={str(n.get('atlas_index'))+'/'+str(n.get('instance')):n for n in frame.get('patterns',[])}
            for j,locator in items:
                record=records[j];array=locator.get('array','all_actual_readouts');index=int(locator['index']);raw_row=frame[array][index]
                if raw_row.get('path_id')!=record['path_id'] or not np.array_equal(raw_row.get('uv'),record['uv']):raise ValueError('Original raw source row differs')
                record['raw_source_readout_record']=raw_row;label=raw_row.get('source_registration_label',record.get('source_registration_label'));record['source_registration_label']=label
                note=notes.get(label)
                if note is not None:
                    record['source_registration_context']={k:value for k,value in note.items() if k not in ('actual_readouts',)}
                    record['source_registration']=note.get('registration');reg=note.get('registration',{})
                    record['registration_cc']=reg.get('correlation',reg.get('cc'));record['registration_H']=reg.get('H');record['registration_context_status']='resolved_from_original_registration_frame'
                else:
                    record['registration_context_status']='no_matching_pattern_note_in_original_frame'
                    if isinstance(frame.get('registration'),dict):record['source_registration']=frame['registration']
        verified.append(select_jsonl(source,requests,consume,raw_hashes[source]))
    groups=defaultdict(list);frames=defaultdict(list);paths=defaultdict(list)
    for record in records:
        groups[record['group_id']].append(record);frames[record['frame_id']].append(record);paths[record['path_id']].append(record)
    def refs(items):return dict(rows=[r['row']for r in items],observation_ids=[r['observation_id']for r in items],ledger_lines=[r['ledger_line']for r in items])
    group_list=[]
    for gid,items in groups.items():
        a=items[0];uv=np.array([r['uv']for r in items]);group_list.append(dict(group_id=gid,shot=a['shot'],pts=a['pts'],source_seconds=a['source_seconds'],path_id=a['path_id'],row_count=len(items),max_error_px=max(r['error_px']for r in items),uv_bounds=[uv.min(axis=0).tolist(),uv.max(axis=0).tolist()],parent_old_vs_new=dict(Counter(r['parent_old_vs_new']for r in items)),**refs(items)))
    group_list.sort(key=lambda g:(-g['max_error_px'],g['shot'],g['pts'],g['path_id']))
    frame_list=[]
    for fid,items in frames.items():
        a=items[0];maximum=max(r['error_px']for r in items);frame_list.append(dict(frame_id=fid,shot=a['shot'],pts=a['pts'],source_seconds=a['source_seconds'],row_count=len(items),max_error_px=maximum,error_band=band(maximum),path_ids=sorted({r['path_id']for r in items}),group_ids=sorted({r['group_id']for r in items}),**refs(items)))
    frame_list.sort(key=lambda f:(-f['max_error_px'],f['shot'],f['pts']))
    path_list=[dict(path_id=pid,row_count=len(items),frame_count=len({r['frame_id']for r in items}),frame_ids=sorted({r['frame_id']for r in items}),group_ids=sorted({r['group_id']for r in items}),max_error_px=max(r['error_px']for r in items),**refs(items))for pid,items in paths.items()]
    by_shot={}
    for shot in shots:
        sid=shot['id'];rs=[r for r in records if r['shot']==sid];fs=[f for f in frame_list if f['shot']==sid]
        by_shot[sid]=dict(rows=len(rs),frames=len(fs),row_bands=dict(Counter(r['error_band']for r in rs)),frame_max_error_bands=dict(Counter(f['error_band']for f in fs)),parent_old_vs_new=dict(Counter(r['parent_old_vs_new']for r in rs)))
    summary=dict(schema='rinkspace.large_error_source_review.v1',created_unix_seconds=time.time(),threshold='finite error_px > 20; no input row removed or classified',
        residuals=dict(path=str(args.residuals.resolve()),sha256=residual_sha,rows=len(rr),finite_rows=int(np.count_nonzero(np.isfinite(ee))),invalid_rows=int(np.count_nonzero(~np.isfinite(ee)))),
        prepared_root=str(root),point_input_sha256=m['point_input_sha256'],current_parent_root=str(parent),review_rows=len(records),review_frames=len(frame_list),review_groups=len(group_list),review_paths=len(path_list),
        row_bands=dict(Counter(r['error_band']for r in records)),frame_max_error_bands=dict(Counter(f['error_band']for f in frame_list)),parent_old_vs_new=dict(Counter(r['parent_old_vs_new']for r in records)),by_shot=by_shot,
        source_origin_kinds=dict(Counter(r['source_origin_locator']['kind']for r in records)),registration_context_status=dict(Counter(r.get('registration_context_status','not_applicable_no_packet_source')for r in records)),
        source_capture_roots=dict(Counter(r.get('source_capture_root')or'unknown_or_inherited_without_packet'for r in records)),source_registration_labels=dict(Counter(r.get('source_registration_label')or'not_available'for r in records)),
        verified_source_files=verified,decisions=dict(pending=len(records),automatic_decisions=0),input_observations_modified=False,solver_started=False,elapsed_seconds=time.monotonic()-started)
    assert sum(len(items)for items in groups.values())==len(hot)==len(records)
    assert {r['row']for r in records}==set(map(int,rows)) and all(r['decision']['status']=='pending'for r in records)
    if digest(args.residuals)!=residual_sha:raise ValueError('Current residuals changed during indexing; rerun against a frozen residual snapshot')
    out.mkdir(parents=True,exist_ok=True)
    with targets[0].open('x',encoding='utf8') as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    targets[1].write_text(json.dumps(dict(schema=summary['schema'],decision_ledger='observations.jsonl',decision_slots='Every observation has a pending decision slot in the ledger; group/frame/path lists reference exact ledger lines.',groups=group_list,frames=frame_list,paths=path_list),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    targets[2].write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:summary[k]for k in ['review_rows','review_frames','review_groups','review_paths','frame_max_error_bands','parent_old_vs_new','registration_context_status','elapsed_seconds']},ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','parent','residuals','output']:parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())
