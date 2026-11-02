"""Retain a source-initialized endpoint zoom shape with one free level offset."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import csr_matrix,issparse
from .motion_chart import cubic_samples
from .multi_shot_motion import exposure_offset_seconds


def retain_endpoint_zoom_shape(directions,labels,shots,dt,max_readout,configuration,controls):
    path=Path(configuration).resolve();doc=json.loads(path.read_text(encoding='utf-8-sig'));source=Path(doc['source_shape_state']).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=doc['source_shape_sha256']:raise ValueError('Source zoom shape state changed')
    with np.load(source,allow_pickle=False) as loaded:reference=loaded['controls']
    if reference.shape!=controls.shape:raise ValueError('Source zoom shape uses a different motion layout')
    sparse=issparse(directions);B=directions.tocsr() if sparse else csr_matrix(directions.reshape(-1,directions.shape[2]));labels=list(labels);proof=[];seen=set()
    for item in doc['anchors']:
        sid=item['shot']
        if sid in seen:raise ValueError('Declare one endpoint zoom interval per shot')
        seen.add(sid);shot=next(s for s in shots if s['id']==sid);aa,bb=item['first_pts'],item['last_pts']
        if aa!=shot['first_pts'] or not aa<=bb<=shot['last_pts']:raise ValueError('Source zoom shape must start at the registered shot endpoint')
        times=np.array([(p-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den'] for p in (aa,bb)])+exposure_offset_seconds(shot,np.array([aa,bb],np.int64))+np.array([-max_readout/2,max_readout/2]);indices,_=cubic_samples(shot['control_stop']-shot['control_begin'],shot['first_control_time'],dt,times);fine=np.arange(indices.min(),indices.max()+1)+shot['control_begin'];rows=3*fine+2;touch=np.asarray(B[rows].getnnz(axis=0)).ravel()>0;selected=[j for j,label in enumerate(labels) if label[0]==sid and label[2]==2 and touch[j]]
        if not selected:raise ValueError('Selected scope has no endpoint zoom directions')
        keep=[j for j in range(B.shape[1]) if j not in set(selected)];new_q=len(keep)+1;tr=np.r_[keep,selected];tc=np.r_[np.arange(len(keep)),np.full(len(selected),len(keep))];T=csr_matrix((np.ones(len(tr)),(tr,tc)),shape=(B.shape[1],new_q));B=(B@T).tocsr();labels=[labels[j] for j in keep]+[(sid,'source_zoom_shape_offset',2)];prefix=B[rows].toarray()
        if np.max(abs(prefix-prefix[:1]))>2e-12 or abs(prefix[0,-1]-1.)>2e-12:raise ValueError('Zoom offset basis does not preserve the continuous endpoint shape')
        delta=controls[fine,2]-reference[fine,2]
        if np.ptp(delta)>1e-10:raise ValueError('Resume does not preserve the declared source zoom shape')
        proof.append(dict(shot=sid,first_pts=aa,last_pts=bb,source_shape_fine_controls=fine.tolist(),original_zoom_columns=len(selected),retained_zoom_columns=1,free_log_f_level=True,max_basis_shape_change=float(np.max(abs(prefix-prefix[:1]))),source_shape_resume_deviation=float(np.ptp(delta))))
    result=B if sparse else B.toarray().reshape(len(controls),3,B.shape[1]);meta=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_shape_state=str(source),source_shape_sha256=doc['source_shape_sha256'],basis_only_no_objective_or_source_uv_changes=True,anchors=proof)
    return result,labels,meta


def validate_endpoint_zoom_shape(controls,configuration):
    if configuration is None:return
    with np.load(configuration['source_shape_state'],allow_pickle=False) as loaded:reference=loaded['controls']
    for item in configuration['anchors']:
        fine=item['source_shape_fine_controls'];delta=controls[fine,2]-reference[fine,2]
        if np.ptp(delta)>1e-10:raise RuntimeError('Accepted state changed the retained source endpoint zoom shape')
