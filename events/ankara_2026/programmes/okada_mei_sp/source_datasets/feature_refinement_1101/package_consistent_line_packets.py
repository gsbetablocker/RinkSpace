"""Validate exact source exclusions, then pack only nonempty qualified shots."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
BASE=Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/feature_refinement_1101')
RUN=BASE.parent
PROJECT=Path('D:/OneDrive/codex_projs/RinkSpace')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rowkey(sid,pt,row):return (sid,pt,row['path_id'],*row['uv'])
def main(args):
    qualified=BASE/f'qualified_{args.variant}_lines'
    review_name='short_guard_line_reviews.json' if args.variant=='short_guard' else f'line_{args.variant}_reviews.json'
    summary=read(qualified/'qualification_summary.json');reviews=read(BASE/review_name)
    proof=read(BASE/f'material_{args.variant}/readout_preservation_check.json')
    known={}
    for frag in proof['known_error_fragments']:
        for uv in frag['UV']:known[(frag['shot'],frag['pts'],frag['path_id'],*uv)]=frag
    withheld={(v['shot'],v['pts'],v['path_id'],*v['uv']):v for v in reviews['withheld_source_uv']}
    assert len(known)==sum(f['rows'] for f in proof['known_error_fragments']) and known and len(withheld)==17
    bad_keys=set(known)|set(withheld);pending_seen=set();qualified_hits=[];nonempty=[];empty=[]
    for sid,counts in summary['shots'].items():
        (nonempty if counts['qualified_rows']>0 else empty).append(sid)
        with (qualified/sid/'pending.jsonl').open(encoding='utf8') as f:
            for line in f:
                item=json.loads(line);row=item['row'];k=rowkey(sid,row['pts'],row)
                if k in bad_keys:pending_seen.add(k)
        with (qualified/sid/'source_rows.jsonl').open(encoding='utf8') as f:
            for line in f:
                frame=json.loads(line)
                for row in frame['rows']:
                    k=rowkey(sid,frame['pts'],row)
                    if k in bad_keys:qualified_hits.append(k)
    assert pending_seen==bad_keys and not qualified_hits and nonempty
    assert all(summary['shots'][sid]['qualified_rows']>0 for sid in nonempty)
    check=dict(schema='rinkspace.consistent_line_qualification_checks.v1',known_error_fragment_rows=len(known),known_error_UV_rows_in_pending=sum(k in pending_seen for k in known),
        known_error_UV_rows_in_qualified=0,withheld_source_UV_rows=17,withheld_UV_rows_in_pending=sum(k in pending_seen for k in withheld),withheld_UV_rows_in_qualified=0,
        all_original_readout_rows_preserved=proof['all_source_readout_rows'],all_original_readout_fields_exactly_preserved=proof['all_original_row_fields_exactly_preserved'],
        packet_shots=nonempty,no_packet_required_shots=empty,qualified_rows=summary['qualified_rows'],remaining_pending_rows=summary['remaining_pending_rows'],
        empty_packet_deletion_allowed=False,prepared_modified=False,prepare_executed=False,fit_executed=False)
    output=BASE/f'{args.variant}_line_qualification_checks.json';assert not output.exists();output.write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('QUALIFICATION_CHECKS '+json.dumps(check,ensure_ascii=False),flush=True)
    command=[sys.executable,'-B',str(PROJECT/'events/ankara_2026/programmes/okada_mei_sp/build_registered_source_packets.py'),
        '--prepared',str(RUN/'prepared_source_recovered'),'--capture',str(qualified),
        '--output',str(BASE/f'{args.variant}_line_packets'),'--shots',','.join(nonempty)]
    subprocess.run(command,cwd=str(PROJECT),check=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('consistent','identity','short_guard'),default='consistent')
    main(parser.parse_args())
