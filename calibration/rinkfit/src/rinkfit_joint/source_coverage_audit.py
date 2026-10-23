"""Audit the entire current source store against immutable packet identities.

Missing positive instance evidence is reported separately from an explicitly
unknown instance. This audit does not grant eligibility or modify observations.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,json
import numpy as np
from .io_utils import read_json,read_jsonl
from .dense_joint_packet import _array
from .dense_typed_source_fit import SourceInstanceGate
from .observed_primal_restart import checked,reference,write_new


def audit(packet_path,registry_path,output):
    p=Path(packet_path).resolve();packet=read_json(p);binding=read_json(checked(packet['dense_qualification_binding']))
    gate=SourceInstanceGate(registry_path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    a={k:_array(packet['arrays'][k],p.parent)for k in ('uv','native_pts','shot_index','material_index','admission','fit_role')}
    original=_array(packet['dense_source_binding']['source_store_ordinal'],p.parent)
    n0=int(packet['dense_source_binding']['row_start']);lookup={int(v):n0+i for i,v in enumerate(original)}
    seen=set();count=Counter();byshot=Counter();bykind=Counter();positive=Counter();mismatch=[]
    path=out/'source_ledger.csv.gz'
    with gzip.open(path,'wt',encoding='utf8',newline='')as stream:
        w=csv.writer(stream);w.writerow(('store_ordinal','part_index','selected_row_id','shot_id','native_pts','packet_row','active','direct_pixel','surface','feature','source_kind','instance_evidence_state'))
        for part_index,part in enumerate(binding['qualified_source_parts']):
            for local,row in enumerate(read_jsonl(part['reference'])):
                ordinal=int(part['ordinal_start'])+local;ri=lookup.get(ordinal);active=False;count['display_rows']+=1
                if ri is not None:
                    if ri in seen:raise ValueError('Duplicate current source ordinal')
                    seen.add(ri);active=bool(a['admission'][ri]and a['fit_role'][ri]);count['packet_dense_rows']+=1
                    same=(row['selected_row_id']==packet['source_row_ids'][ri]and row['shot_id']==packet['shot_ids'][int(a['shot_index'][ri])]and int(row['native_pts'])==int(a['native_pts'][ri])and np.array_equal(np.asarray(row['uv'],dtype='<f8').view('u8'),np.asarray(a['uv'][ri],dtype='<f8').view('u8')))
                    if not same:mismatch.append(ri)
                    if active and (row.get('source_UV_is_direct_measurement')is not True or row.get('physical_surface')not in ('ice','board')):raise ValueError('Active row lacks direct static surface pixels')
                explicit_unknown=row.get('physical_instance_qualification')in ('unknown','pending','unverified')or row.get('source_fit_eligible')is False
                support=gate.rows.get(row['selected_row_id'])
                if explicit_unknown or (support is not None and support['instance_state']!=1):state='explicit_unknown'
                elif support is not None:
                    if ri is None or gate.check(row,int(a['material_index'][ri]))!=0:raise ValueError('Positive evidence identity mismatch')
                    state='positive_finite_instance'
                else:state='positive_instance_not_compiled'
                if active and state=='explicit_unknown':raise ValueError('Explicit unknown enters current fit')
                count['active'if active else 'not_active']+=1
                if ri is not None:positive[state+('_active'if active else '_inactive')]+=1
                if active:byshot[row['shot_id']]+=1;bykind[(row['shot_id'],row['source_kind'],row['physical_surface'])]+=1
                w.writerow((ordinal,part_index,row['selected_row_id'],row['shot_id'],row['native_pts'],''if ri is None else ri,int(active),int(row.get('source_UV_is_direct_measurement')is True),row.get('physical_surface'),row['feature_id'],row['source_kind'],state))
    if len(seen)!=len(original)or mismatch:raise ValueError('Incomplete/mismatched packet source identity: '+str(mismatch[:10]))
    result=dict(schema='rinkfit.complete_current_source_coverage_audit.v1',packet=reference(p),registry=reference(registry_path),current_source_parts=binding['qualified_source_parts'],
        counts=dict(count),dense_instance_evidence_counts=dict(positive),active_dense_by_shot=dict(byshot),active_by_kind=[dict(shot=k[0],source_kind=k[1],surface=k[2],rows=v)for k,v in bykind.items()],
        every_packet_dense_row_UV_bitwise_PTS_ID_checked=True,explicit_unknown_active_rows=0,
        missing_positive_evidence_is_not_proof_of_wrong_pixels=True,source_qualification_changed=False,
        full_positive_finite_instance_coverage=not positive['positive_instance_not_compiled_active'],
        new_source_pixels_decoded=False,source_ledger=reference(path),reader=reference(__file__))
    write_new(out/'report.json',result);print(json.dumps({k:result[k]for k in ('counts','dense_instance_evidence_counts','active_dense_by_shot','full_positive_finite_instance_coverage')}),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--registry',required=True);p.add_argument('--out',required=True);p.add_argument('--code-aliases');a=p.parse_args()
    from .exact_dependency_aliases import dependency_alias_context
    with dependency_alias_context(reference(a.code_aliases)if a.code_aliases else None):audit(a.packet,a.registry,a.out)

if __name__=='__main__':main()
