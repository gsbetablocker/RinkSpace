"""Resume source adapters without replacing original rows or trusting stale pins."""
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rinkspace.session.source_inputs import SourceEvidenceIndex, SourceInputError, SourceSupportResolver


def _job(tmp_path, name, **changes):
    directory=tmp_path/name;directory.mkdir()
    routes=json.dumps(dict(schema='rinkspace.handoff.source_routes.v1',source_groups={})).encode()
    (directory/'routes.json').write_bytes(routes)
    spec=dict(roots=dict(repo=str(tmp_path),data=str(tmp_path)),source={'sha256':'a'*64},
        source_clock={'native_pts_step':256},selected_shot_ids=['s02'],
        sequences=[dict(shot_id='s02',first_pts=256,stop_pts_exclusive=768)],
        source_routes_ref=dict(path='routes.json',sha256=hashlib.sha256(routes).hexdigest()),**changes)
    path=directory/'job.json';path.write_text(json.dumps(spec),encoding='utf8')
    return path


def test_revised_instructions_reuse_exact_source_index_read_only(tmp_path):
    old=_job(tmp_path,'old',intent='old task',scene_ref={'sha256':'b'*64})
    new=_job(tmp_path,'new',intent='replacement task',scene_ref={'sha256':'c'*64})
    database=tmp_path/'source.sqlite'
    initial=SourceEvidenceIndex(old,database)
    initial.db.execute("insert into gaps values('original-gap','s02','original preserved failure')")
    initial.db.commit();initial.close();before=database.read_bytes()
    reader=SourceEvidenceIndex(new,database,read_only=True,reuse_from_job=old)
    assert reader.db.execute('select * from gaps').fetchall()==[('original-gap','s02','original preserved failure')]
    assert reader.job['intent']=='replacement task'
    assert reader.reuse_provenance['previous_task_instructions_used'] is False
    with pytest.raises(SourceInputError,match='read-only'):reader.build()
    with pytest.raises(sqlite3.OperationalError):reader.db.execute('delete from gaps')
    reader.close();assert database.read_bytes()==before


@pytest.mark.parametrize('change',['source','source_clock','sequences','routes','prior_job'])
def test_reuse_rejects_changed_source_or_unauthenticated_predecessor(tmp_path,change):
    old=_job(tmp_path,'old',intent='old');new=_job(tmp_path,'new',intent='new')
    database=tmp_path/'source.sqlite';SourceEvidenceIndex(old,database).close()
    path=old if change=='prior_job' else new;spec=json.loads(path.read_bytes())
    if change=='routes':
        route=path.parent/'routes.json';route.write_bytes(route.read_bytes()+b'\n')
        spec['source_routes_ref']['sha256']=hashlib.sha256(route.read_bytes()).hexdigest()
    elif change=='prior_job':spec['intent']='modified after indexing'
    else:spec[change]={'changed':True}
    path.write_text(json.dumps(spec),encoding='utf8');before=database.read_bytes()
    with pytest.raises(SourceInputError):SourceEvidenceIndex(new,database,read_only=True,reuse_from_job=old)
    assert database.read_bytes()==before


def test_cached_support_cannot_accept_a_conflicting_declared_sha(tmp_path):
    path=tmp_path/'support.sqlite';db=sqlite3.connect(path)
    db.executescript('CREATE TABLE support_rows(file TEXT,line INTEGER,payload TEXT);'
                    'CREATE TABLE support_files(file TEXT,sha256 TEXT,complete INTEGER);')
    db.execute('insert into support_files values(?,?,1)',('source.jsonl','a'*64))
    db.execute('insert into support_rows values(?,?,?)',('source.jsonl',7,json.dumps({'uv':[12.5,34.25]})))
    db.commit();db.close();before=path.read_bytes()
    resolver=SourceSupportResolver(SimpleNamespace(repo_root=tmp_path),path)
    reference=dict(file='source.jsonl',jsonl_line_0based=7,sha256='a'*64)
    assert resolver.resolve(reference)['payload']['uv']==[12.5,34.25]
    with pytest.raises(SourceInputError,match='SHA'):
        resolver.resolve(dict(reference,sha256='b'*64))
    with pytest.raises(SourceInputError,match='line index'):
        resolver.resolve(dict(reference,jsonl_line_0based=True))
    resolver.support_db.close();assert path.read_bytes()==before


def test_support_file_cache_keeps_the_requested_pointer_in_provenance(tmp_path):
    path=tmp_path/'source.json';path.write_text(json.dumps({'rows':[{'uv':[1,2]},{'uv':[3,4]}]}))
    resolver=SourceSupportResolver(SimpleNamespace(repo_root=tmp_path,resolve_registered=lambda _:path))
    base=dict(file='source.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    first=resolver.resolve(dict(base,json_pointer='/rows/0'))
    second=resolver.resolve(dict(base,json_pointer='/rows/1'))
    assert first['provenance']['reference']['json_pointer']=='/rows/0'
    assert second['provenance']['reference']['json_pointer']=='/rows/1'
    assert second['payload']['uv']==[3,4]


@pytest.mark.parametrize('schema',['rinkfit.dense_native_board_true_material_boundaries_source_frame.v1',
    'rinkfit.dense_native_board_source_frame.v1'])
def test_board_material_frame_context_survives_row_pointer_without_changing_source(tmp_path,schema):
    document=dict(schema=schema,
        native_pts=256,shot_id='s02',time_base='1/12800',decoded_native_BGR_sha256='d'*64,
        paired_blue_cap_and_white_board_body_semantics=True,
        yellow_contact_from_current_global_band_not_accumulated_extrapolation=True,
        local_material_requalification=dict(actual_current_RGB_reread=True,original_anchored_source_UV_preserved=True,
            changes=[{'original_uv':[20.,40.],'current_uv':[20.,41.]}]),
        roles={'board.top':{},'board.contact':{}},rows=[dict(uv=[20.,40.]),dict(uv=[60.,44.])])
    path=tmp_path/'source.json';path.write_text(json.dumps(document))
    before=path.read_bytes();digest=hashlib.sha256(before).hexdigest()
    resolver=SourceSupportResolver(SimpleNamespace(repo_root=tmp_path,resolve_registered=lambda _:path))
    for i in range(2):
        result=resolver.resolve(dict(file='source.json',sha256=digest,json_pointer=f'/rows/{i}'))
        assert result['payload']==document['rows'][i]
        context=result['source_frame_context']
        assert context['material_roles']==['board.top','board.contact'] and context['native_pts']==256
        assert context['local_material_requalification']==dict(actual_current_RGB_reread=True,original_anchored_source_UV_preserved=True)
        assert 'rows' not in context and result['provenance']['actual_file_sha256']==digest
    assert path.read_bytes()==before


def test_replacement_job_keeps_existing_exposure_records_and_original_job_pin(tmp_path):
    from rinkspace.session.source_programme import ProgrammeLedger
    old=_job(tmp_path,'old',intent='old');new=_job(tmp_path,'new',intent='new',scene_ref={'sha256':'c'*64})
    root=tmp_path/'programme';ledger=ProgrammeLedger(root,old)
    payload=dict(source_binding={'source_pts':256,'decoded_native_rgb_sha256':'d'*64},
        candidates=[dict(candidate_id='original',localization='measured',identity='unresolved',
            original_uv=[12.5,23.25],source_group_id='original-correlated-group')])
    ledger.commit('s02',256,payload);ledger.db.close()
    original_frame=(root/'frames/s02/256.json.gz').read_bytes()
    resumed=ProgrammeLedger(root,new,previous_job=old)
    assert resumed.read('s02',256)==payload
    assert resumed.db.execute("select value from meta where key='job_sha256'").fetchone()[0]==hashlib.sha256(old.read_bytes()).hexdigest()
    resumed.db.close();assert (root/'frames/s02/256.json.gz').read_bytes()==original_frame
    spec=json.loads(new.read_bytes());spec['sequences'][0]['stop_pts_exclusive']=1024
    new.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='exposure scope'):ProgrammeLedger(root,new,previous_job=old)
    assert (root/'frames/s02/256.json.gz').read_bytes()==original_frame
