"""Preserve authenticated later withdrawals without importing old geometry."""
from __future__ import annotations
import hashlib,json
import numpy as np


def build_qualification_index(index, progress=None):
    db=index.db
    db.executescript("""
      CREATE TABLE IF NOT EXISTS qualification(id TEXT PRIMARY KEY,shot TEXT,pts INTEGER,payload TEXT);
      CREATE INDEX IF NOT EXISTS qualification_frame ON qualification(shot,pts);
    """)
    refs={q['reference']['sha256']:q for g in index.inputs.routes['source_groups'].values()
          for q in g.get('latest_qualification_refs',[]) if q.get('name')=='qualification'}
    reader,_=index.inputs._core()
    for sha,ref in refs.items():
        marker='verified_qualification/'+sha
        if db.execute('select 1 from meta where key=?',(marker,)).fetchone():continue
        q=index.inputs.read_json_ref(ref)
        binding_ref=q['source_qualification_binding'];binding=index.inputs.read_json_ref(binding_ref)
        withdrawal=binding['withdrawn_source_store_ordinals']
        raw=index.inputs.resolve_registered(withdrawal['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=withdrawal['sha256']:raise ValueError('Withdrawal array changed')
        ordinals=set(map(int,np.frombuffer(raw,dtype=withdrawal['dtype'])))
        if len(ordinals)!=withdrawal['shape'][0]:raise ValueError('Withdrawal identity count differs')
        seen=set()
        for part in binding['qualified_source_parts']:
            n=0
            for n,row in enumerate(reader(index.inputs.resolve_reference(part['reference'])),1):
                ordinal=part['ordinal_start']+n-1
                if ordinal not in ordinals:continue
                if row.get('physical_instance_qualification')!='unknown' or row.get('source_fit_eligible') is not False:
                    raise ValueError('Later withdrawal and sealed source qualification disagree')
                oid=row['source_point_id'];seen.add(ordinal)
                evidence=dict(reason=row['qualification_reason_code'],review_id=row.get('qualification_review_id'),
                    source_store_ordinal=ordinal,qualification_ref=ref,binding_ref=binding_ref,
                    source_point_id=oid,original_uv=row['uv'],source_feature_id=row['feature_id'],
                    disposition='identity_unknown',human_approval_granted=False)
                db.execute('insert or replace into qualification values(?,?,?,?)',
                    (oid,row['shot_id'],row['native_pts'],json.dumps(evidence,separators=(',',':'))))
            if n!=part['row_count']:raise ValueError('Latest qualification stream row count differs')
            if progress:progress(dict(stage='latest_qualification_stream_verified',group=part['name'],rows=n,withdrawn_found=len(seen)))
        if seen!=ordinals:db.rollback();raise ValueError('Some withdrawn source IDs are missing')
        db.execute('insert into meta values(?,?)',(marker,str(len(seen))));db.commit()
    return dict(withdrawn_records=db.execute('select count(*) from qualification').fetchone()[0])


def apply_qualification(index, candidates, shot, pts):
    rows={oid:json.loads(raw) for oid,raw in index.db.execute(
        'select id,payload from qualification where shot=? and pts=?',(shot,pts))}
    for c in candidates:
        q=rows.get(c.candidate_id)
        if q is None:continue
        if c.original_uv!=q['original_uv']:raise ValueError('Withdrawn source identity reused with different UV')
        c.withdrawal=q;c.identity='unresolved'
    return rows
