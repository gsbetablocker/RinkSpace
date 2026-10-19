"""Source-authenticated finite path intervals; numerical geometry is unchanged."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path

def apply_source_path_domains(model,request_file):
    path=Path(request_file).resolve();source=json.loads(path.read_text(encoding='utf8'));out=copy.deepcopy(model)
    q=source['Ad8_legacy_WY'];mi=int(q['material_index']);pi=int(q['path_index']);old=out['materials'][mi]['paths'][pi]
    if old['id']!=q['path_id'] or old['kind']!=2 or len(old['points'])!=int(q['vertex_count']):raise ValueError('source finite path identity mismatch')
    intervals=q['legal_closed_station_intervals'];previous=-1.
    for lo,hi in intervals:
        if not 0<=lo<=hi<=1 or lo<=previous:raise ValueError('invalid source finite station intervals')
        previous=hi
    ref=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    old['allowed_station_intervals']=copy.deepcopy(intervals);old['station_domain_source_refs']=[ref]
    out['path_domain_semantics']=dict(schema='rinkfit.source_finite_station_domain.v1',interpretation='union of closed original station intervals; each connected component is separate and never reclosed',source_refs=[ref],correction='source-confirmed false inter-letter ink connectors excluded; historical complete path is not geometrically equivalent',original_numeric_blocks_unchanged=out['blocks']==model['blocks'])
    return out,dict(schema='rinkfit.source_finite_path_domain_application.v1',material_index=mi,path_index=pi,path_id=old['id'],intervals=intervals,original_numeric_blocks_unchanged=out['blocks']==model['blocks'],original_path_points_unchanged=old['points']==model['materials'][mi]['paths'][pi]['points'],source=ref)

def station_allowed(path,station,tolerance=0.):
    import numpy as np
    s=np.asarray(station);allowed=np.zeros(s.shape,dtype=bool)
    for lo,hi in path.get('allowed_station_intervals',[[0.,1.]]):allowed|=(s>=lo-tolerance)&(s<=hi+tolerance)
    return allowed
