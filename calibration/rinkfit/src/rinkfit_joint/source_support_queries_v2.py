"""Frozen local acquisition Queries with actual ordered support membership.

No representative-to-representative edge is ever built. Missing operators,
ambiguous branches and missing component membership retain engineering pixels
and an explicit unavailable Query. A deterministic acquisition window is not W,
a statistical sigma, or a tangential precision estimate.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import gzip,json
import numpy as np
from PIL import Image
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics_v2 import _records,_reference,_admitted
from .source_worker import _channels,_sample

IMPLEMENTATION_SHA256=file_sha256(__file__)


def _point_evidence(row,packet):
    parts=row['source_pointer'].strip('/').split('/')
    value=packet
    for key in parts[:-2]:value=value[int(key)] if isinstance(value,list) else value[key]
    evidence=value.get('source_evidence')
    if isinstance(evidence,list):return evidence[int(parts[-1])]
    return evidence if isinstance(evidence,dict) else {}


def _component_hits(seed,normal,low,high,components,actual=None):
    """Attach only to real raw vertices/edges inside worker components."""
    hits=[]
    tangent=np.array([-normal[1],normal[0]])
    for component in components:
        raw=np.asarray(component['raw_uv'],float);current=np.asarray(component['current_uv'],float)
        exact=actual is not None and any(np.array_equal(x,np.asarray(actual,float)) for x in raw)
        offsets=[]
        for a,b in zip(current[:-1],current[1:]):
            side=np.array([(a-seed)@tangent,(b-seed)@tangent])
            if side[0]*side[1]>0:continue
            delta=side[1]-side[0]
            if abs(delta)<=1e-12:
                if max(abs(side))<=1e-10:
                    lo,hi=sorted(((a-seed)@normal,(b-seed)@normal))
                    if hi>=low and lo<=high:offsets.append(max(low,lo))
            else:
                station=-side[0]/delta
                if 0<=station<=1:
                    alpha=float((a+station*(b-a)-seed)@normal)
                    if low<=alpha<=high:offsets.append(alpha)
        # A singleton requires an exact actual-source binding; no artificial
        # radius or q25/q75 box is used to assign it to a representative.
        if exact or (actual is None and offsets):hits.append(component['component_id'])
    return sorted(set(hits))


def _legacy_query(row,channels):
    audit=row.get('source_measurement_audit') or {};profile=audit.get('profile') or {}
    if audit.get('status')!='current_native_material_interface' or profile.get('valid') is not True:
        raise ValueError('no frozen single-interface local operator')
    normal=np.asarray(audit.get('normal'),float);seed=np.asarray(row['source_uv'],float)
    if normal.shape!=(2,) or abs(np.linalg.norm(normal)-1)>1e-9:raise ValueError('invalid frozen local normal')
    channel=profile.get('channel');polarity=profile.get('polarity')
    if channel not in (0,1,2) or polarity not in (-1,1):raise ValueError('missing frozen channel or polarity')
    offsets=np.arange(-3.,3.0001,.25)
    # Replay only the already frozen channel/polarity; no identity/channel or
    # normal reselection. This is the actual old worker's finite acquisition.
    sampled=_sample(channels,seed+offsets[:,None]*normal)[:,channel]
    gradient=np.gradient(sampled,.25);signed=gradient*polarity;absolute=abs(gradient)
    peaks=[i for i in range(1,len(signed)-1) if signed[i]>0 and signed[i]>signed[i-1] and signed[i]>=signed[i+1]]
    if len(peaks)!=1:raise ValueError('frozen local profile has multiple or missing material branches')
    k=peaks[0]
    if k!=int(absolute.argmax()):raise ValueError('frozen polarity is not the recorded material branch')
    offset=float(offsets[k]);v=absolute[k-1:k+2];den=v[0]-2*v[1]+v[2]
    if abs(den)>1e-9:offset+=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
    if offset!=float(profile['offset_px']):raise ValueError('frozen legacy operator replay changed its measured alpha')
    if not np.array_equal(seed+offset*normal,np.asarray(row['current_uv'],float)):
        raise ValueError('frozen legacy Query does not replay exact current UV')
    return seed,normal,offset,offsets,dict(kind='frozen_legacy_gaussian_r3_linear25_peak',
        channel=channel,polarity=polarity,normal_channel_or_material_reselected=False,
        raw_operator='source_worker._remeasure single interface',unique_current_local_material_branch=True)


def _cubic_query(row,evidence):
    audit=row.get('source_measurement_audit') or {};operator=evidence.get('frozen_point_operator')
    if not isinstance(operator,dict):raise ValueError('missing actual frozen cubic point operator')
    normal=np.asarray(operator['normal'],float);seed=np.asarray(operator['navigation_seed_uv'],float)
    config=operator.get('source_measurement_audit',{}).get('profile',{})
    if config.get('channel') not in (0,1,2) or config.get('polarity') not in (-1,1):
        raise ValueError('missing frozen cubic channel/polarity')
    offsets=np.asarray(audit.get('actual_profile_normal_offsets'),float)
    if offsets.shape!=(49,) or not np.array_equal(offsets,np.linspace(-3.,3.,49)):
        raise ValueError('actual frozen cubic acquisition is not original +/-3/49')
    if (audit.get('numeric_valid') is not True or audit.get('material_reselected') is not False
        or audit.get('normal_channel_polarity_reselected') is not False
        or audit.get('operator_version')!='fixed_outer_material_continuous_cubic_peak_v1'
        or audit.get('synthetic_window_boundary_count')!=0
        or len(audit.get('maximal_stationary_roots',[]))!=1 or audit.get('flat_peak_intervals')):
        raise ValueError('frozen cubic local material branch is not unique and qualified')
    offset=float(audit['normal_offset_px'])
    if not np.array_equal(normal,np.asarray(audit['normal'])) or not np.array_equal(seed+offset*normal,np.asarray(row['current_uv'])):
        raise ValueError('frozen cubic Query changed original current UV/normal/alpha')
    return seed,normal,offset,offsets,dict(kind='fixed_outer_material_continuous_cubic_peak_v1',
        channel=config['channel'],polarity=config['polarity'],normal_channel_or_material_reselected=False,
        original_49_offsets_preserved=True,unique_current_local_material_branch=True,
        frozen_operator_sha256=object_sha256(operator),frozen_operator_code_ref=row.get('frozen_operator_code_ref'))


def compile_queries(audited_rows,retained_support,output_dir,*,input_binding,statistics_dir):
    out=Path(output_dir)
    if out.exists():raise ValueError('query table generation must be new')
    binding=read_json(input_binding);generation=read_json(binding['generation']['path'])
    if file_sha256(binding['generation']['path'])!=binding['generation']['sha256']:raise ValueError('changed generation binding')
    refs=generation['references'];rows=list(_records(audited_rows));row_by={r['source_row_id']:r for r in rows}
    if len(rows)!=len(row_by):raise ValueError('duplicate source rows')
    statistics_dir=Path(statistics_dir);stat_manifest=read_json(statistics_dir/'manifest.json')
    stat_by_row=defaultdict(list)
    for group in read_jsonl(statistics_dir/'groups.jsonl.gz'):
        for rid in group.get('representative_source_row_ids',[]):
            stat_by_row[rid].append(dict(source_statistics_group_id=group['source_statistics_group_id'],
                source_correlation_family_id=group['source_correlation_family_id'],
                source_correlation_component_id=group.get('source_correlation_component_id'),
                joint_family_covariance_status=group.get('joint_family_covariance_status'),
                local_numeric_covariance_status=group['status']))
    by_row=defaultdict(list);components=[];denominators=Counter();rejections=[]
    for item in _records(retained_support):
        denominators['support_items']+=1;audits={x['raw_index']:x for x in item.get('actual_source_audits',[])}
        denominators['actual_source_audit_instances']+=len(audits)
        denominators['failed_actual_source_audit_instances']+=sum(x.get('current_uv') is None for x in audits.values())
        parts=item.get('current_support_segments')
        if parts is None:
            rejections.append(dict(statistical_input_id=item['statistical_input_id'],reason='current ordered support components not supplied'))
            continue
        flat=[i for part in parts for i in part]
        if len(flat)!=len(set(flat)) or set(flat)!=set(item['current_support_indices']):raise ValueError('worker support segments do not partition admitted raw indices')
        blocked={frozenset(x['between_raw_indices']) for x in item.get('source_visibility_link_rejections',[])}
        for ordinal,part in enumerate(parts):
            if any(frozenset((a,b)) in blocked for a,b in zip(part[:-1],part[1:])):
                raise ValueError('declared component crosses explicit rejected visibility link')
            actual=[audits[i] for i in part]
            if any(x.get('current_uv') is None for x in actual):raise ValueError('component contains rejected actual readout')
            component=dict(component_id='source-component-v2:'+object_sha256([item['statistical_input_id'],item['source_packet_sha256'],part])[:24],
                statistical_input_id=item['statistical_input_id'],source_family_id=item['source_family_id'],
                shot_id=item['shot_id'],native_pts=item['native_pts'],ordered_raw_indices=part,
                source_packet_sha256=item['source_packet_sha256'],raw_uv=[x['raw_uv'] for x in actual],
                current_uv=[x['current_uv'] for x in actual],
                explicit_visibility_link_rejections=item.get('source_visibility_link_rejections',[]),
                edges_come_only_from_ordered_current_support_segments=True)
            components.append(component)
            for rid in item['representative_row_ids']:by_row[rid].append(component)
    out.mkdir(parents=True);counts=Counter();unavailable_reasons=Counter();records=[];packet_cache={};channels_cache={}
    for row in sorted(rows,key=lambda r:(r['native_pts'],r['source_row_id'])):
        rid=row['source_row_id'];kind=1 if row.get('point_correspondence') else 0
        pixel=row.get('current_uv') if row.get('current_uv') is not None else row['source_uv']
        record=dict(source_row_id=rid,native_pts=row['native_pts'],shot_id=row['shot_id'],kind=kind,
            engineering_pixel_uv=pixel,engineering_pixel_basis='audited_current_operator_mean',
            original_source_uv=row['source_uv'],audited_current_uv=row.get('current_uv'),source_UV_or_current_UV_modified=False,
            physical_identity_status=row.get('physical_identity_status'),query_available=False,
            engineering_family_dependency_ids=stat_by_row.get(rid,[]),
            engineering_exposure_id=f"{row['shot_id']}:{row['native_pts']}",
            engineering_normalization_family_id=row.get('source_family_id'),
            fixed_entity_station=bool(kind),source_component_ids=[])
        try:
            if kind:raise ValueError('fixed entity corner uses kind1 and has no sliding Query')
            if row.get('manual_reference_preserved'):raise ValueError('exact manual click has no invented profile Query')
            if row.get('current_uv') is None or not _admitted(row):raise ValueError('source identity or current material support not qualified')
            logical=row['source_packet'];ref=refs[logical]
            if logical not in packet_cache:
                if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('changed exact source packet')
                packet_cache[logical]=read_json(ref['path'])
            evidence=_point_evidence(row,packet_cache[logical])
            if row.get('schema')=='rinkfit.native_frozen_cubic_remeasurement_row.v1':
                seed,normal,alpha,offsets,operator=_cubic_query(row,evidence)
            else:
                key=row['native_png']
                if key not in channels_cache:
                    channels_cache.clear()
                    if file_sha256(key)!=row['native_png_sha256']:raise ValueError('changed exact native PNG')
                    channels_cache[key]=_channels(np.asarray(Image.open(key).convert('RGB')))
                seed,normal,alpha,offsets,operator=_legacy_query(row,channels_cache[key])
            actual=evidence.get('actual_native_readout_uv')
            if actual is None and row.get('evidence_class')=='native_pixel_claim':actual=row['source_uv']
            memberships=_component_hits(seed,normal,float(offsets[0]),float(offsets[-1]),by_row[rid],actual=actual)
            if len(memberships)!=1:raise ValueError('local frozen Query has no unique ordered actual support component')
            query=dict(source_row_id=rid,source_factor_row_id=rid,alpha=offsets.tolist(),
                native_uv=(seed+offsets[:,None]*normal).tolist(),normal=normal.tolist(),
                actual_acquisition_window_preserved=True,deterministic_query_is_not_tangential_noise_precision=True,
                source_component_id=memberships[0],normal_channel_or_material_reselected=False)
            record.update(query_available=True,query=query,query_uv=seed.tolist(),profile_normal=normal.tolist(),
                initial_alpha=alpha,alpha_bounds=[float(offsets[0]),float(offsets[-1])],
                source_component_ids=memberships,actual_local_operator=operator)
            counts['queries_available']+=1
            if operator.get('original_49_offsets_preserved'):counts['frozen_Ad24_queries']+=1
        except (ValueError,KeyError,TypeError,IndexError) as error:
            record['query_unavailable_reason']=str(error);counts['queries_unavailable']+=1;unavailable_reasons[str(error)]+=1
            record['engineering_pixel_uv']=row['source_uv']
            record['engineering_pixel_basis']='original_source_UV_without_unique_frozen_Query'
            counts['fixed_entity_rows']+=kind
        records.append(record)
    def write_lines(name,values):
        with gzip.open(out/name,'wt',encoding='utf8') as stream:
            for value in values:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    write_lines('query_table.jsonl.gz',records);write_lines('support_components.jsonl.gz',components)
    write_json(out/'support_component_rejections.json',rejections)
    result=dict(schema='rinkfit.actual_ordered_support_queries.v2',rows=len(records),counts=dict(counts),query_unavailable_reasons=dict(unavailable_reasons),
        input_denominators=dict(denominators),components=len(components),
        input_binding=_reference(iter(()),input_binding),statistics_manifest=dict(path=str(statistics_dir/'manifest.json'),sha256=file_sha256(statistics_dir/'manifest.json')),
        implementation_sha256=IMPLEMENTATION_SHA256,source_row_id_keys=True,row_index_assigned_only_by_packet_builder=True,
        queries=dict(path=str(out/'query_table.jsonl.gz'),sha256=file_sha256(out/'query_table.jsonl.gz')),
        support_components=dict(path=str(out/'support_components.jsonl.gz'),sha256=file_sha256(out/'support_components.jsonl.gz')),
        no_representative_adjacency_or_gap_interpolation=True,source_UV_or_current_UV_modified=False,
        noise_W_or_sigma_or_engineering_E_values_invented=False,
        missing_queries_keep_existing_engineering_source_pixels=True,
        formal_W_ready=stat_manifest.get('full_W_ready',False))
    write_json(out/'manifest.json',result);return result
