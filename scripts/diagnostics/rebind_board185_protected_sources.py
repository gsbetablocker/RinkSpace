"""Authenticate unchanged protected observations in the appended board container."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'source_pixel_review_additional/main1_Ad4_temporal_existing_motion_readonly/whole_span_RGB_boundary_candidate'
MODE_FIELDS = ('boundary_location_observation', 'boundary_location_measurement')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')

def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def main():
    config = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_same757_original_support_rectangle_20261011.json'
    options = deepcopy(read(config)['source_options'])
    board = OUT / 'native_board185_original109_q50_soft_locations_plus76_continuous_Ad4_RGB_UNADMITTED_CANDIDATE_20261011.json'
    new_board = read(board)
    old_board = read(options['--additional-board-source'])
    assert len(old_board['rows']) == 109 and len(new_board['rows']) == 185
    strip = lambda row: {k:v for k,v in row.items() if k not in MODE_FIELDS}
    assert all(canonical(strip(a)) == canonical(strip(b)) for a,b in zip(old_board['rows'], new_board['rows'][:109]))
    newref = ref(board)
    proofs = []
    def rebound(key, filename):
        oldpath = Path(options[key]); previous = read(oldpath); current = deepcopy(previous)
        changes = []
        for record in current['records']:
            if record.get('family') != 'native_board_edge':
                continue
            original_ref = record['packet_reference']
            assert ref(original_ref['path'])['sha256'] == original_ref['sha256']
            original = read(original_ref['path'])['rows'][record['original_row_index']]
            new = new_board['rows'][record['original_row_index']]
            assert hashlib.sha256(canonical(original)).hexdigest() == record['source_row_canonical_json_sha256']
            assert canonical(strip(original)) == canonical(strip(new))
            original_record = deepcopy(record)
            record['packet_reference'] = newref
            record['source_row_canonical_json_sha256'] = hashlib.sha256(canonical(new)).hexdigest()
            observed = lambda r: {k:v for k,v in r.items() if k not in ('packet_reference', 'source_row_canonical_json_sha256')}
            assert canonical(observed(record)) == canonical(observed(original_record))
            changes.append(dict(id=record['id'], original_row_index=record['original_row_index'],
                before=original_ref, after=newref, original_source_row_sha256=original_record['source_row_canonical_json_sha256'],
                current_source_row_sha256=record['source_row_canonical_json_sha256'],
                all_observation_support_role_fields_exact=True))
        current['board185_source_container_rebinding'] = dict(parent_authority_reference=ref(oldpath),
            new_board_reference=newref, changed_source_metadata_fields=list(MODE_FIELDS),
            protected_observations_UV_sigma_normal_support_world_roles_unchanged=True, records=changes)
        path=OUT/filename;save(path,current);options[key]=str(path)
        proofs.append(dict(source_option=key,parent=ref(oldpath),derived=ref(path),records=changes))
        return current,path
    human,hpath=rebound('--direct-human-truth-authority','human_same54_rebound_BOARD185.json')
    guard_old=Path(options['--direct-human-full-stroke-guard']);guard=read(guard_old);before=deepcopy(guard)
    guard['primary_authority_reference']=ref(hpath)
    for record,primary in zip(guard['records'],human['records']):
        assert record['id']==primary['id']
        record['authority_source_row_canonical_json_sha256']=primary['source_row_canonical_json_sha256']
    path=OUT/'human_same54_BOARD185_full_guard.json';save(path,guard);options['--direct-human-full-stroke-guard']=str(path)
    assert guard['guard_arrays_reference']==before['guard_arrays_reference']
    assert ref(guard['guard_arrays_reference']['path'])['sha256']==guard['guard_arrays_reference']['sha256']
    for key,name in [('--direct-human-stroke-cap-support','human_same54_BOARD185_caps.json'),
                     ('--practical-human-location-envelope','human_same54_BOARD185_location_NO_EXTRA_EXPANSION.json')]:
        oldpath=Path(options[key]);value=read(oldpath);original=deepcopy(value)
        if key=='--direct-human-stroke-cap-support':
            value['primary_authority_reference']=ref(hpath)
            value['full_trace_reference']=ref(path)
        else:
            value['authority_references']['native']=ref(hpath)
        assert value['records']==original['records']
        p=OUT/name;save(p,value);options[key]=str(p)
    confirmed,cpath=rebound('--user-confirmed-source-truth-authority','confirmed_same30_rebound_BOARD185_ICE87.json')
    options['--additional-board-source']=str(board)
    # This handoff is solely for the actual public-factor invariance check. A
    # production startup must independently bind the revised target/caps.
    options.pop('--truth-restoration-reviewed-manual-append-audit',None)
    handoff=SYS/'native_board185_protected_source_options_Handoff_20261011.json'
    save(handoff,dict(schema='rinkspace.protected-source-container-handoff.v1',source_options=options,
        production_startup_authorized_by_this_handoff=False,source_config_reference=ref(config),
        container_row_rebinding=proofs,full_guard_arrays_unchanged=True,
        source_guard_UV_support_caps_location_roles_unchanged=True,human_groups=len(human['records']),confirmed_groups=len(confirmed['records'])))
    print(json.dumps(dict(handoff=ref(handoff),human=ref(hpath),confirmed=ref(cpath),board=newref)))

if __name__=='__main__':main()
