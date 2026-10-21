"""Versioned source statistics consuming actual current visibility components.

The source worker owns source qualification and connected support. This compiler
never reconnects a rejected/masked interval, never counts repeated raw pixels as
new samples, and never calls item ordinal a correlation segment. Local empirical
operator covariances do not establish an unmeasured joint covariance between
operators sharing a declared correlation group.
"""
from __future__ import annotations
import argparse, gzip, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from .io_utils import read_jsonl, write_json, file_sha256, object_sha256
from .source_statistics import StatisticsUnavailable, fit_representative_statistics

IMPLEMENTATION_SHA256=file_sha256(__file__)


def connected_support(items):
    """Merge duplicate evidence along explicitly admitted, ordered graph edges.

    Each vertex is one packet plus original raw UV. Multiple display segments
    can reference it, while only worker-declared within-component adjacency can
    create an edge. A branch has no single HAC order and remains unavailable.
    """
    vertices={}; edges=defaultdict(set); input_components=[]; all_audits=0; failed=0
    duplicate_instances=0; breaks=[]; blocked_edges=set(); failed_keys=set()
    for item in items:
        support=item.get('current_support_indices')
        parts=item.get('current_support_segments')
        if support is None or parts is None:
            raise StatisticsUnavailable('source worker did not declare current connected support segments')
        flat=[i for part in parts for i in part]
        if len(flat)!=len(set(flat)) or set(flat)!=set(support):
            raise ValueError('current support components do not partition admitted raw indices')
        if any(not part for part in parts):raise ValueError('empty current source support component')
        audits={a['raw_index']:a for a in item.get('actual_source_audits',[])}
        if len(audits)!=len(item.get('actual_source_audits',[])):raise ValueError('duplicate raw audit indices')
        all_audits+=len(audits);failed+=sum(a.get('current_uv') is None for a in audits.values())
        indices={}; packet=item['source_packet_sha256']
        failed_keys.update((packet,tuple(a['raw_uv'])) for a in audits.values() if a.get('current_uv') is None)
        for index in support:
            audit=audits.get(index)
            if audit is None or audit.get('current_uv') is None:
                raise ValueError('admitted support index has no current native readout')
            key=(packet,tuple(audit['raw_uv'])); current=np.asarray(audit['current_uv'],float)
            if current.shape!=(2,) or not np.isfinite(current).all():raise ValueError('nonfinite current raw source readout')
            if key in vertices:
                duplicate_instances+=1
                if not np.array_equal(vertices[key]['uv'],current):
                    raise StatisticsUnavailable('one original raw readout has conflicting current measurements')
            else:vertices[key]={'uv':current,'origins':[]}
            vertices[key]['origins'].append(dict(statistical_input_id=item['statistical_input_id'],raw_index=index))
            indices[index]=key
        for ordinal,part in enumerate(parts):
            keys=[indices[i] for i in part]
            if len(keys)!=len(set(keys)):raise StatisticsUnavailable('a declared source segment repeats one raw pixel')
            for a,b in zip(keys[:-1],keys[1:]):
                edges[a].add(b);edges[b].add(a)
            input_components.append(dict(statistical_input_id=item['statistical_input_id'],
                component_ordinal=ordinal,ordered_raw_indices=part))
        for rejected_link in item.get('source_visibility_link_rejections',[]):
            a,b=rejected_link['between_raw_indices']
            if a in audits and b in audits:
                blocked_edges.add(frozenset(((packet,tuple(audits[a]['raw_uv'])),(packet,tuple(audits[b]['raw_uv'])))))
        breaks.extend(dict(statistical_input_id=item['statistical_input_id'],**x)
                      for x in item.get('source_visibility_link_rejections',[]))
    if failed_keys & set(vertices):
        raise StatisticsUnavailable('same original raw readout has inconsistent source admission across overlapping groups; common reader required')
    for pair in blocked_edges:
        if len(pair)!=2:continue
        a,b=tuple(pair);edges[a].discard(b);edges[b].discard(a)
    remaining=set(vertices); components=[]
    while remaining:
        first=min(remaining,key=str);stack=[first]; component=set()
        while stack:
            key=stack.pop()
            if key in component:continue
            component.add(key);stack.extend(edges[key]-component)
        remaining-=component
        if any(len(edges[x]&component)>2 for x in component):
            raise StatisticsUnavailable('overlapping ordered source supports form a branching HAC graph')
        endpoints=[x for x in component if len(edges[x]&component)<=1]
        # A genuine closed source contour has no endpoint; choose a stable seam,
        # retain the closing edge in provenance, and do not duplicate its pixel.
        start=min(endpoints or component,key=str);order=[];previous=None;current=start
        while current not in order:
            order.append(current);following=sorted((edges[current]&component)-({previous} if previous else set()),key=str)
            if not following:break
            previous,current=current,following[0]
        if set(order)!=component:raise StatisticsUnavailable('source component has no unambiguous path ordering')
        components.append((order,not bool(endpoints) and len(component)>2))
    raw=[];labels=[];provenance=[];keys=[]
    for label,(order,closed) in enumerate(components):
        for key in order:
            raw.append(vertices[key]['uv']);labels.append(label);keys.append(key)
        provenance.append(dict(component_id=label,closed=closed,raw_count=len(order),
            ordered_original_raw_uv=[list(key[1]) for key in order],
            ordered_source_packet_sha256=[key[0] for key in order],
            original_memberships=[vertices[key]['origins'] for key in order]))
    proof=dict(worker_connected_components_consumed=True,item_ordinals_used_as_segments=False,
        duplicate_admitted_instances=duplicate_instances,unique_current_raw_readouts=len(raw),
        actual_audit_instances=all_audits,failed_actual_audit_instances_retained=failed,
        input_components=input_components,merged_connected_components=provenance,
        source_visibility_link_rejections=breaks,globally_respected_rejected_visibility_links=len(blocked_edges),
        hidden_span_or_index_gap_reconnected=False)
    return raw,labels,keys,proof


def _admitted(row):
    # solver_eligible remains False until numeric covariance exists. It cannot
    # gate the estimator whose job is to produce that covariance. Source
    # identity and actual pixel support are separate, authoritative inputs.
    if row.get('source_admitted') is False:return False
    return row.get('physical_identity_status') in {'confirmed','confirmed_static_material',
        'typed_identity_confirmed','typed_native_material_supported','typed_identity_current_pixel_supported'}


def _records(value):
    return read_jsonl(value) if isinstance(value,(str,Path)) else iter(value)


def _reference(value,binding):
    if isinstance(value,(str,Path)):
        return dict(path=str(value),sha256=file_sha256(value))
    if binding is None:raise ValueError('streaming source statistics require a pinned input binding')
    return dict(kind='qualified_source_generation_stream',input_binding=dict(path=str(binding),sha256=file_sha256(binding)))


def compile_sources(audited_rows,retained_support,output_dir,*,input_binding=None):
    input_refs=dict(audited_rows=_reference(audited_rows,input_binding),retained_support=_reference(retained_support,input_binding))
    out=Path(output_dir)
    if out.exists():raise ValueError('statistics generation must be new')
    out.mkdir(parents=True)
    rows={r['source_row_id']:r for r in _records(audited_rows)}
    support=_records(retained_support);families=defaultdict(list);denominators=Counter()
    for item in support:
        denominators['support_item_instances']+=1
        denominators['actual_source_audit_instances']+=len(item.get('actual_source_audits',[]))
        denominators['failed_actual_source_audit_instances']+=sum(x.get('current_uv') is None for x in item.get('actual_source_audits',[]))
        denominators['admitted_support_instances']+=len(item.get('current_support_indices',[]))
        denominators['declared_current_support_components']+=len(item.get('current_support_segments',[]))
        denominators['source_visibility_link_rejections']+=len(item.get('source_visibility_link_rejections',[]))
        byidentity=defaultdict(list)
        for identifier in item['representative_row_ids']:
            row=rows.get(identifier)
            if row is None or row.get('point_correspondence'):continue
            identity=tuple(row.get(k) for k in ['target_id','physical_surface','measurement_operator_resolved','boundary_role'])
            byidentity[identity].append(identifier)
        for identity,ids in byidentity.items():
            selected=dict(item);selected['representative_row_ids']=ids
            families[(item['source_family_id'],identity)].append(selected)
    decisions=[];counts=Counter();raw_owners={};family_groups=defaultdict(list)
    for ordinal,((family,identity),items) in enumerate(sorted(families.items(),key=lambda x:str(x[0]))):
        identifier='statistics-v2:'+object_sha256([family,list(identity)])[:24]
        ids=list(dict.fromkeys(i for item in items for i in item['representative_row_ids']))
        accepted=[i for i in ids if rows[i].get('current_uv') is not None and _admitted(rows[i])]
        record=dict(source_group_id=family,source_statistics_group_id=identifier,physical_identity=list(identity),
            representative_source_row_ids=accepted,source_inputs=[x['statistical_input_id'] for x in items],
            identity_admission_required=True,source_operator_eligibility_distinct_from_final_solver_eligibility=True,fit_role='development',
            historical_source_selection_not_independent_validation=True,
            raw_and_derived_representatives_both_counted=False,
            source_correlation_family_id=family,declared_correlation_metadata=[
                dict(statistical_input_id=x['statistical_input_id'],shot_id=x.get('shot_id'),native_pts=x.get('native_pts'),
                     declared_correlation_ids=x.get('source_curve_family_metadata',{}).get('declared_correlation_ids'),
                     circle_family_coupling_hint=x.get('source_curve_family_metadata',{}).get('circle_family_coupling_hint'),
                     joint_group_navigation=x.get('source_curve_family_metadata',{}).get('joint_group_navigation')) for x in items])
        family_groups[family].append(identifier)
        try:
            raw,segments,keys,component_proof=connected_support(items)
            record.update(raw_readouts=len(raw),support_component_proof=component_proof)
            reused=sorted({raw_owners[k] for k in keys if k in raw_owners})
            if reused:
                record['cross_operator_raw_reuse']=reused
                raise StatisticsUnavailable('actual readout also belongs to another source operator; joint operator covariance unknown')
            if not accepted:raise StatisticsUnavailable('no source-supported typed representative rows')
            kind='free_ellipse' if any(x.get('source_curve_family_metadata',{}).get('source_circle_fit') for x in items) else 'cubic_graph'
            result=fit_representative_statistics(raw,[rows[i]['current_uv'] for i in accepted],curve_kind=kind,segments=segments)
            W=result['W'];name=f'g{ordinal:05d}_W.f64';W.astype('<f8').tofile(out/name)
            uvname=f'g{ordinal:05d}_uv.f64';result['current_uv'].astype('<f8').tofile(out/uvname)
            record.update(status='numeric_source_covariance_ready',statistic_kind='native_readout_operator_propagation',
                rank=len(W),W=dict(file=name,sha256=file_sha256(out/name),dtype='<f8',shape=list(W.shape)),
                remeasured_uv=dict(file=uvname,sha256=file_sha256(out/uvname),dtype='<f8',shape=list(result['current_uv'].shape)),
                proof=result['proof'])
            for key in keys:raw_owners[key]=identifier
            counts['groups_ready']+=1;counts['statistical_rank']+=len(W);counts['represented_rows']+=len(accepted)
        except (StatisticsUnavailable,ValueError,np.linalg.LinAlgError) as error:
            record.update(status='statistics_not_available',reason=str(error));counts['groups_missing']+=1
        decisions.append(record)
    # Retain connected correlation dependencies across source-family containers,
    # including explicit whole-curve IDs that span several source packets/groups.
    parent={x['source_statistics_group_id']:x['source_statistics_group_id'] for x in decisions}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    tokens=defaultdict(list)
    for record in decisions:
        gid=record['source_statistics_group_id']
        tokens['family:'+record['source_correlation_family_id']].append(gid)
        for metadata in record['declared_correlation_metadata']:
            declared=metadata.get('declared_correlation_ids') or []
            if isinstance(declared,str):declared=[declared]
            for value in declared:tokens['declared:'+str(value)].append(gid)
            hint=metadata.get('circle_family_coupling_hint')
            if hint:tokens['circle_hint:'+object_sha256([metadata.get('shot_id'),metadata.get('native_pts'),hint])].append(gid)
    for members in tokens.values():
        for gid in members[1:]:parent[root(gid)]=root(members[0])
    correlated=defaultdict(list)
    for gid in parent:correlated[root(gid)].append(gid)
    shared=[dict(source_correlation_component_id='correlation-v2:'+object_sha256(sorted(g))[:24],
                 source_statistics_group_ids=sorted(g),joint_covariance_status='unknown',
                 independent_block_diagonal_assumption_authorized=False)
            for g in correlated.values() if len(g)>1]
    component_by_group={gid:x['source_correlation_component_id'] for x in shared for gid in x['source_statistics_group_ids']}
    for record in decisions:
        if record['source_statistics_group_id'] in component_by_group:
            record['source_correlation_component_id']=component_by_group[record['source_statistics_group_id']]
            record['joint_family_covariance_status']='unknown'
            record['local_operator_covariance_does_not_establish_family_independence']=True
        else:record['joint_family_covariance_status']='one_operator'
    with gzip.open(out/'groups.jsonl.gz','wt',encoding='utf8') as stream:
        for record in decisions:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    write_json(out/'decisions.json',decisions);write_json(out/'correlation_groups.json',shared)
    counts['shared_correlation_families_with_unknown_joint_covariance']=len(shared)
    result=dict(schema='rinkfit.joint9.source_statistics.v2',counts=dict(counts),input_denominators=dict(denominators),
        input_denominators_are_instances_not_independent_statistical_samples=True,
        inputs=input_refs,
        implementation_sha256=IMPLEMENTATION_SHA256,worker_connected_components_consumed=True,
        local_operator_covariances_ready=counts['groups_missing']==0,
        full_W_ready=counts['groups_missing']==0 and not shared,
        raw_and_representatives_both_counted=False,independent_representative_sigma_used=False,
        PSD_floor_or_ridge_used=False,formal_statistical_optimum_claimed=False,
        unknown_joint_group_covariance_not_replaced_by_independent_blocks=True,
        correlation_groups=dict(path=str(out/'correlation_groups.json'),sha256=file_sha256(out/'correlation_groups.json')))
    write_json(out/'manifest.json',result);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rows',required=True)
    ap.add_argument('--support',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    print(json.dumps(compile_sources(args.rows,args.support,args.out),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
