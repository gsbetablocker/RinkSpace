"""Reference-only exposure schedules for unchanged native readout statistics."""
from pathlib import Path
import json
from .io_utils import read_jsonl,file_sha256
from .source_readout_identity import actual_readout_key


def iter_scheduled_source_rows(path):
    source=Path(path)
    if source.suffix.lower()!='.json':
        yield from read_jsonl(source);return
    schedule=json.loads(source.read_text(encoding='utf8'))
    if schedule.get('schema')!='rinkfit.reference_only_native_statistics_schedule.v1':
        raise ValueError('Unknown reference source statistics schedule')
    alias_ref=schedule['actual_source_alias_index'];index_path=Path(alias_ref['file'])
    if file_sha256(index_path)!=alias_ref['sha256']:raise ValueError('Actual readout alias index changed')
    index=json.loads(index_path.read_text(encoding='utf8'))
    for key in ('audit_manifest','aliases','producer_script'):
        ref=index[key]
        if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Alias evidence changed')
    aliases={r['source_row_id']:r for r in read_jsonl(index['aliases']['path'])}
    all_wanted=set();seen=set()
    for entry in schedule['source_journals']:
        wanted=set(map(int,entry['native_pts']))
        if len(wanted)!=len(entry['native_pts'])or all_wanted.intersection(wanted):raise ValueError('Exposure repeated in source schedule')
        all_wanted.update(wanted);journal=Path(entry['file'])
        if file_sha256(journal)!=entry['sha256']:raise ValueError('Immutable native source journal changed')
        encountered=set();ids=set()
        for row in read_jsonl(journal):
            pts=int(row['native_pts'])
            if pts not in wanted or row.get('source_geometry_eligible')is not True:continue
            rid=row['source_row_id']
            if rid in aliases:
                a=aliases[rid]
                if int(a['native_pts'])!=pts or actual_readout_key(row)!=a['exact_same_actual_measurement_input_sha256']:
                    raise ValueError('Alias is not this exact source readout')
                continue
            if rid in ids:raise ValueError('Same native readout repeated in scheduled source journal')
            ids.add(rid);encountered.add(pts)
            row['_statistics_source_journal_ref']=dict(file=str(journal.resolve()),sha256=entry['sha256'])
            yield row
        if encountered!=wanted:raise ValueError('Scheduled exposure has no canonical qualified source rows')
        seen.update(encountered)
    if len(all_wanted)!=schedule['requested_exposures']or seen!=all_wanted:
        raise ValueError('Reference source exposure coverage incomplete')
