"""Typed source-instance evidence, separate from material-position profiles.

This is a partial audited-scope gate. Unreviewed rows receive no confirmation;
empirical edge intervals are neither world identity nor calibrated covariance.
"""
from pathlib import Path
import numpy as np
from .io_utils import read_json
from .dense_joint_packet import _array
from .dense_joint_proofs import _ref,_checked,_write

SUPPORT_KINDS={0:'no_positive_instance_support',1:'actual_source_connected_instance',2:'verified_static_track_to_confirmed_instance',3:'human_source_identity'}
UNKNOWN_REASON='inference_only_full_curve_instance_unverified'


def build_actual_instance_audit(*,packet,actual_review_report,output):
    path=Path(packet).resolve();doc=read_json(path);report_path=Path(actual_review_report).resolve();report=read_json(report_path)
    if report.get('schema')!='rinkfit.actual_dense_neighborhood_source_review.v1' or _checked(report['source_packet'],report_path.parent).resolve()!=path:raise ValueError('Actual source review must bind this packet')
    scope=report['review_scope']
    if scope.get('selection_by_model_residual')is not False or report.get('raw_source_UV_ID_PTS_changed')is not False:raise ValueError('Model residuals cannot grant or withdraw instance qualification')
    material_evidence=read_json(_checked(report['A2_material_evidence'],report_path.parent))
    if material_evidence.get('RGB_matches_actual_current_accepted_exposure')is not True or material_evidence.get('qualification_changed')is not False:raise ValueError('Actual unchanged native source inputs required')
    members=report['A2']['rows'];lookup={r['selected_row_id']:r for r in members}
    packet_rows={ident:i for i,ident in enumerate(doc['source_row_ids'])if ident in lookup}
    if set(packet_rows)!=set(lookup):raise ValueError('Reviewed source IDs are absent from packet')
    n0=int(doc['dense_source_binding']['row_start']);ordinals=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent)
    si=_array(doc['arrays']['shot_index'],path.parent);pts=_array(doc['arrays']['native_pts'],path.parent)
    rows=[]
    for ident,index in sorted(packet_rows.items(),key=lambda z:z[1]):
        item=lookup[ident]
        if index<n0 or item.get('world_instance_state')!='unknown' or item.get('positive_world_instance_evidence')!=[]:raise ValueError('This actual report currently supports unknown instance state only')
        if doc['shot_ids'][int(si[index])]!=scope['shot_id'] or int(pts[index])!=int(scope['native_pts']):raise ValueError('Audited native exposure differs')
        interval=np.asarray(item['source_normal_midpoint_support_interval_px'],float)
        if interval.shape!=(2,)or not np.isfinite(interval).all()or interval[0]>interval[1]or item.get('statistical_sigma_issued')is not False:raise ValueError('Finite non-statistical empirical source interval required')
        rows.append((int(ordinals[index-n0]),index,interval))
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True)
    arrays={}
    for name,a in [('source_store_ordinal',np.asarray([r[0]for r in rows],dtype='<u8')),('packet_row',np.asarray([r[1]for r in rows],dtype='<u8')),('instance_state',np.zeros(len(rows),dtype='u1')),('instance_support_kind',np.zeros(len(rows),dtype='u1')),('normal_midpoint_empirical_interval_px',np.asarray([r[2]for r in rows],dtype='<f8'))]:
        target=out/(name+'.bin');a.tofile(target);arrays[name]=dict(_ref(target),dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
    refs=[_ref(report_path),report['actual_source_review'],report['operator_excerpt'],report['A2_material_evidence'],report['source_RGB'],report['source_UV_overlay'],material_evidence['guide'],material_evidence['reader'],material_evidence['original_producer_policy']]
    manifest=dict(schema='rinkfit.typed_dense_source_instance_support_audit.v1',source_packet=_ref(path),actual_review_consumer=_ref(__file__),actual_source_review_report=_ref(report_path),audited_scope=dict(shot_id=scope['shot_id'],native_pts=scope['native_pts'],feature_id='ice.A2:outer',raw_rows=len(rows)),arrays=arrays,state_codes={'0':'unknown','1':'source_confirmed'},support_kind_codes={str(k):v for k,v in SUPPORT_KINDS.items()},evidence_refs=refs,source_confirmed_rows=0,world_instance_unknown_rows=len(rows),unknown_reason=UNKNOWN_REASON,paired_material_profile_can_grant_world_identity=False,unreviewed_source_rows_default_to_confirmed=False,entire_dense_source_instance_certified=False,source_UV_ID_PTS_qualification_unchanged=True,empirical_interval_kind='mean_of_paired_empirical_edge_IQR_q25_q75',statistical_sigma_or_probability_issued=False,material_width_used_as_position_sigma=False)
    result=_write(out/'manifest.json',manifest)
    receipt=assert_actual_instance_audit(result['file'],path)
    _write(out/'actual_consumption.json',receipt)
    return dict(instance_audit=result,actual_consumption=_ref(out/'actual_consumption.json'))


def assert_actual_instance_audit(manifest_path,packet):
    manifest_path=Path(manifest_path).resolve();m=read_json(manifest_path);path=Path(packet).resolve();doc=read_json(path)
    if m.get('schema')!='rinkfit.typed_dense_source_instance_support_audit.v1'or m['source_packet']!=_ref(path):raise ValueError('Typed instance audit binds another packet')
    for ref in m['evidence_refs']+ [m['actual_review_consumer']]:_checked(ref,manifest_path.parent)
    a={k:_array(v,manifest_path.parent)for k,v in m['arrays'].items()};n=int(m['audited_scope']['raw_rows'])
    if any(len(v)!=n for v in a.values())or len(set(map(int,a['source_store_ordinal'])))!=n:raise ValueError('Typed audit does not contain its complete unique scope')
    n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent)
    for o,i,state,kind in zip(a['source_store_ordinal'],a['packet_row'],a['instance_state'],a['instance_support_kind']):
        if int(i)<n0 or int(ordinal[int(i)-n0])!=int(o):raise ValueError('Typed audit source ordinal differs from actual packet')
        if int(state)==0:
            if int(kind)!=0:raise ValueError('Unknown instance cannot claim positive evidence kind')
        else:
            # Positive support needs its own source evidence issuer. This audit
            # consumer intentionally cannot mint such evidence from profiles.
            raise ValueError('No positive world-instance issuer is installed in this narrow audited source case')
    if m.get('paired_material_profile_can_grant_world_identity')is not False or m.get('unreviewed_source_rows_default_to_confirmed')is not False:raise ValueError('Profile or default instance promotion is forbidden')
    return dict(schema='rinkfit.typed_dense_source_instance_support_consumption.v1',passed=True,instance_audit=_ref(manifest_path),source_packet=_ref(path),actual_consumer=_ref(__file__),audited_raw_rows=n,unknown_rows=int(np.count_nonzero(a['instance_state']==0)),source_confirmed_rows=0,unreviewed_rows_not_certified=True,source_pixels_decoded=0,source_UV_or_qualification_changed=False)
