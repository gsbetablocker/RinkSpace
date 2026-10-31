"""Assemble already qualified independent captures without re-reading source RGB.

The first qualified frame in the explicitly ordered capture list is retained.
Other captures fill missing frames only. No point or eligibility decision is
created; immutable source record locations and all original evidence remain.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
from build_independent_shot_packets import build
from capture_independent_ad4_glyph_gap import write
from capture_independent_badge import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source-scope',type=Path,required=True);p.add_argument('--shot',required=True);p.add_argument('--captures',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--latest-source-groups',action='store_true',help='Newest explicitly qualified PTS/path group replaces only that group; preserve all other qualified paths');a=p.parse_args();root=a.root.resolve();out=a.output.resolve();sid=a.shot
    if out.exists():raise FileExistsError(out)
    scope=json.loads(a.source_scope.read_text('utf8'));expected=sorted(set(scope[sid]['pts']));sources=[];candidates={pt:[]for pt in expected}
    for number,folder in enumerate(a.captures):
        folder=folder.resolve();path=folder/'qualified'/sid/'source_rows.jsonl'
        if not path.is_file():path=folder/sid/'source_rows.jsonl'
        identity=sha(path);sources.append(dict(capture=str(folder),source_rows=str(path),source_rows_sha256=identity))
        seen=set()
        for line_number,line in enumerate(path.open('rb'),1):
            record=json.loads(line);pt=record['pts']
            if pt not in candidates:continue
            if pt in seen:raise ValueError('Duplicate native source exposure in a capture')
            seen.add(pt)
            # Keep only fit inputs/header here. Full raw candidate and original
            # frame records remain at their immutable original paths.
            keep={k:v for k,v in record.items()if k not in ['pending_rows','all_actual_glyph_rows']}
            candidates[pt].append((keep,dict(capture_index=number,source_rows=str(path),source_rows_sha256=identity,source_line=line_number,source_line_sha256=hashlib.sha256(line).hexdigest())))
    out.mkdir(parents=True);qualified=out/'qualified';folder=qualified/sid;folder.mkdir(parents=True);selected=[];receipt=[]
    for pt in expected:
        items=candidates[pt]
        if not items:raise ValueError('No actual source record for requested PTS '+str(pt))
        hashes={r['source_BGR_sha256']for r,l in items}
        if len(hashes)!=1:raise ValueError('Source captures disagree on original RGB identity')
        record,location=next(((r,l)for r,l in items if r.get('fit_eligible')is True and r.get('rows')),items[0]);group_origins={}
        if a.latest_source_groups:
            grouped={}
            for source_record,source_location in items:
                if source_record.get('fit_eligible')is not True or not source_record.get('rows'):continue
                for path in dict.fromkeys(q['path_id']for q in source_record['rows']):
                    grouped[path]=[q for q in source_record['rows']if q['path_id']==path];group_origins[path]=source_location
                record,location=source_record,source_location
            if grouped:record=dict(record,rows=[q for group in grouped.values()for q in group],fit_eligible=True)
        if record.get('rows')and record.get('fit_eligible')is not True:raise ValueError('Unqualified rows cannot be promoted by merge')
        record=dict(record,original_source_capture_record=location,original_source_group_records=group_origins,pending_candidates_preserved_at_original_source_records=True);selected.append(record);receipt.append(dict(pts=pt,rows=len(record.get('rows',[])),source=location,source_groups=group_origins,other_capture_row_counts=[dict(capture_index=l['capture_index'],rows=len(r.get('rows',[])),fit_eligible=r.get('fit_eligible')is True)for r,l in items]))
    (folder/'source_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in selected),encoding='utf8');shutil.copyfile(root/sid/'navigation/poses_reanchored.npz',folder/'poses.npz');binding=dict(source_video_sha256=selected[0]['source_video_sha256'],source_sha256=selected[0]['source_video_sha256'],independent_source_capture=True,source_captures=sources,source_scope=str(a.source_scope.resolve()),source_scope_sha256=sha(a.source_scope),selected_shot=sid,requested_native_PTS=expected,selection='Newest explicitly qualified native PTS/path group wins; other existing qualified groups preserved'if a.latest_source_groups else'First explicitly qualified frame wins; later captures only fill missing native frames, no point duplication',source_RGB_eligibility_and_point_values_changed=False,original_source_records_preserved=True,fit_eligible=True,fit_or_upsert_run=False,script_sha256=sha(__file__))
    write(qualified/'binding.json',binding);write(out/'source_frame_selection.json',receipt);build(root,qualified,out/'source_packets',shot_ids=[sid]);packet=out/'source_packets'/sid;z=dict(np.load(packet/'point_observations.npz',allow_pickle=False));points=[(r['pts'],q)for r in selected if r['fit_eligible']for q in r['rows']];assert np.array_equal(z['pts'],[pt for pt,q in points]);assert np.array_equal(z['uv'],[q['uv']for pt,q in points]);assert len(set(z['observation_ids']))==len(points);missing=[r['pts']for r in selected if not r['rows']];counts={}
    for r,l in [(r,r['original_source_capture_record'])for r in selected if r['rows']]:counts[str(l['capture_index'])]=counts.get(str(l['capture_index']),0)+1
    report=dict(complete=True,shot=sid,requested_frames=len(expected),qualified_frames=len(expected)-len(missing),rows=len(points),missing_native_PTS=missing,selected_frames_by_capture_index=counts,source_captures=sources,packet=str(packet),packet_sha256=sha(packet/'point_observations.npz'),point_UV_PTS_and_eligibility_exactly_preserved=True,new_source_measurements_created=False,fit_or_upsert_run=False)
    write(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
