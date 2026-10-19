import hashlib
import json
import sqlite3

from rinkspace.vision_data import join_user_decisions


def test_human_semantics_join_uses_video_clock_and_native_domain(tmp_path):
    db=tmp_path/'audit.sqlite';image=tmp_path/'annotation.png';image.write_bytes(b'fixture')
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE observations(id TEXT,observation_id TEXT,feature_id TEXT,source_pts INTEGER,source_video_sha256 TEXT,time_base TEXT)')
        connection.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?)',[
            ('good','F3_outer','F3',256,'a'*64,'1/12800'),
            ('other_video','F3_outer','F3',256,'b'*64,'1/12800'),
            ('other_clock','F3_outer','F3',256,'a'*64,'1/15360')])
    decision=dict(id='human',source_pts=256,time_base='1/12800',physical_identity='F3:outer',
        identity_authority='direct_user_confirmation_fixture',image_path=str(image),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),source_video_sha256='a'*64)
    path=tmp_path/'decisions.json';path.write_text(json.dumps(dict(decisions=[decision])))
    assert join_user_decisions(db,path)['linked_observations']==1
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT obs_id FROM human_semantic_links').fetchall()==[('good',)]
    # A subsequent incomplete legacy declaration must withdraw derived links;
    # it cannot reuse the previous PTS-only authority.
    decision.pop('source_video_sha256');path.write_text(json.dumps(dict(decisions=[decision])))
    assert join_user_decisions(db,path)['linked_observations']==0
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT * FROM human_semantic_links').fetchall()==[]
