"""Install source-reviewed replacements at an immutable batch boundary.

Only previously inactive, unweighted source rows are accepted by this migration.
Active statistical W groups require their separate full-covariance migration;
this transaction cannot silently change those distributions or source support.
"""
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,write_jsonl,file_sha256,object_sha256
from .packet_storage import write_packet
from .source_recovery_review import reference,checked


def install_reviewed_source_revision(parent_packet,proposals_path,output):
    parent=Path(parent_packet).resolve();document=read_json(parent);proposal_file=Path(proposals_path).resolve()
    proposal=read_json(proposal_file);out=Path(output).resolve()
    if out.exists():raise FileExistsError('Use a new immutable source/packet generation')
    if checked(proposal['parent_packet'])!=parent:raise ValueError('Source proposals belong to another packet')
    arrays={}
    for key,desc in document['arrays'].items():
        path=Path(desc['file']);path=path if path.is_absolute() else parent.parent/path
        if file_sha256(path)!=desc['sha256']:raise ValueError('Changed parent source array')
        arrays[key]=np.memmap(path,dtype=desc['dtype'],mode='r',shape=tuple(desc['shape']))
    active=(arrays['admission']==1)&(arrays['fit_role']==1);indices=[];revisions=[]
    known={i for g in document['groups']if g.get('rank',0)>0 for i in g['rows']}
    known.update(document.get('independent_rows',[]))
    for item in proposal['rows']:
        i=int(item['row_index'])
        if i in indices or active[i] or i in known:raise ValueError('This migration accepts distinct inactive nonstatistical rows only')
        if document['source_row_ids'][i]!=item['source_row_id']:raise ValueError('Exact source-row identity differs')
        if int(arrays['native_pts'][i])!=item['native_pts'] or int(arrays['material_index'][i])!=item['material_index']:raise ValueError('Source clock or physical material differs')
        if not np.array_equal(arrays['uv'][i],item['old_packet_uv']):raise ValueError('Old source mean differs')
        if not all(item['checks'].get(k) is True for k in ('semantic_class','physical_surface','physical_instance','measurement_operator','visibility','position_support')):
            raise ValueError('Every source dimension needs explicit current evidence')
        for ref in item['evidence_refs']:checked(ref)
        uv=np.asarray(item['current_uv'],float);interval=np.asarray(item['source_query_native_uv'],float)
        if uv.shape!=(2,)or interval.shape!=(2,2)or not np.isfinite(np.r_[uv,interval.ravel()]).all():raise ValueError('Invalid current native source measurement')
        if arrays['kind'][i]!=0:raise ValueError('This migration only handles sliding material profiles')
        indices.append(i);revisions.append(item)
    out.mkdir(parents=True);changes={k:np.array(arrays[k])for k in ('uv','query_uv','profile_normal','admission','row','alpha','alpha_bounds','E')}
    queries={q['row_index']:q for q in document.get('queries',[])}
    for item in revisions:
        i=item['row_index'];uv=np.asarray(item['current_uv']);edge=np.asarray(item['source_query_native_uv']);direction=edge[1]-edge[0];direction/=np.linalg.norm(direction)
        ab=(edge-uv)@direction
        if not ab[0]<0<ab[1]:raise ValueError('Measured source mean outside actual acquired transition interval')
        changes['uv'][i]=uv;changes['query_uv'][i]=uv;changes['profile_normal'][i]=item['source_measurement_direction'];changes['row'][i]=uv[1]
        changes['admission'][i]=1;changes['alpha'][i]=0;changes['alpha_bounds'][i]=ab;changes['E'][i]=item['engineering_E_px']
        queries[i]=dict(row_index=i,alpha=ab.tolist(),native_uv=edge.tolist(),normal=direction.tolist(),source_row_id=item['source_row_id'],
            actual_acquisition_window_preserved=True,new_source_readout_generation=True,deterministic_query_is_not_tangential_noise_precision=True)
    for key,values in changes.items():
        file=out/(key+'.bin');values.tofile(file);document['arrays'][key]={**document['arrays'][key],**dict(file=str(file),sha256=file_sha256(file))}
    # The new measurement has no calibrated covariance. Keep its engineering
    # Query/RS/finite-curve constraints without assigning statistical precision.
    refs=[dict(file=str(proposal_file),sha256=file_sha256(proposal_file))]
    for item in revisions:
        document['groups'].append(dict(id='source_review:'+object_sha256(item)[:24],rows=[item['row_index']],rank=0,W=[],
            statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=refs,
            unknown_noise_source_has_statistical_precision=False,E_RS_Query_and_finite_material_constraints_retained=True))
    newactive=(changes['admission']==1)&(arrays['fit_role']==1)
    partition=np.zeros(document['row_count'],np.int16)
    for g in document['groups']:partition[g['rows']]+=1
    partition[document['independent_rows']]+=1
    if not np.array_equal(partition,newactive.astype(np.int16)):raise ValueError('New source packet active partition differs')
    document.update(queries=list(queries.values()),active_rows=int(newactive.sum()),
        recapture_rows=sorted(set(document.get('recapture_rows',[]))|set(indices)),
        parent_source_review_packet=dict(file=str(parent),sha256=file_sha256(parent)),
        source_review_generation=dict(file=str(out/'observation_generation.json'),sha256='pending'),
        new_source_nuisance_state_recomputed=False,source_placeholder_station_half_is_not_solver_ready=True)
    for i in indices:
        document['allowed_paths'][str(i)]=list(range(8))
        document['source_dataset_ids'][i]='native_review:'+out.name+':'+document['source_dataset_ids'][i]
    generation=dict(schema='rinkfit.reviewed_source_generation.v1',parent_packet=reference(parent),
        proposals=reference(proposal_file),rows=revisions,changed_row_indices=indices,
        source_video_SHA_PTS_material_and_original_raw_journal_preserved=True,old_packet_immutable=True,
        accepted_statistical_precision_supplied=False,requires_new_station_row_alpha_path_capture=True,
        independent_original_UV_preserved_in_history=True,source_origin='actual_native_RGB_material_relocation')
    write_json(out/'observation_generation.json',generation)
    document['source_review_generation']['sha256']=file_sha256(out/'observation_generation.json')
    document['source_identity']=object_sha256(dict(parent_source_identity=document.get('source_identity'),review_generation=document['source_review_generation']))
    document['source_manifests']+=refs+[dict(file=str(out/'observation_generation.json'),sha256=file_sha256(out/'observation_generation.json'))]
    document['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in document['groups']if g.get('statistic_kind')=='constraint_only_unknown_noise')
    document['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in document['groups'])
    from .board_source_qualification_guard import BOUND_ARRAYS,FAMILY_PREFIXES
    import hashlib,json
    affected=np.asarray([i for i,sid in enumerate(document['source_row_ids'])if str(sid).startswith(FAMILY_PREFIXES)and newactive[i]],dtype='<i8')
    certificate=dict(schema='rinkfit.board_material_fit_qualification.v3_reviewed_revision',
        parent_packet=reference(parent),observation_generation=reference(out/'observation_generation.json'),
        producer_refs=[reference(Path(__file__)),reference(Path(__file__).with_name('board_source_qualification_guard.py'))],
        membership_array_sha256={k:document['arrays'][k]['sha256']for k in BOUND_ARRAYS},
        count=len(affected),active_board_rows_sha256=hashlib.sha256(affected.tobytes()).hexdigest(),
        source_row_ids_sha256=hashlib.sha256(json.dumps(document['source_row_ids'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        source_dataset_ids_sha256=hashlib.sha256(json.dumps(document['source_dataset_ids'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        inherited_source_evidence_is_not_new_review=True,new_exact_native_reading_scope=indices)
    from .measurement_operator import measurement_operator_binding
    certificate['measurement_operator_binding'] = measurement_operator_binding(document)
    write_json(out/'qualified_source_revision.json',certificate)
    document['board_material_source_qualification']=dict(file=str(out/'qualified_source_revision.json'),sha256=file_sha256(out/'qualified_source_revision.json'))
    write_packet(out/'packet.json',document)
    report=dict(schema='rinkfit.reviewed_source_packet_delta.v1',parent_packet=reference(parent),new_packet=reference(out/'packet.json'),
      generation=reference(out/'observation_generation.json'),changed_rows=indices,old_active_rows=int(active.sum()),new_active_rows=int(newactive.sum()),
      old_good_source_UV_and_all_S01_rows_unchanged=True,new_noise_W_precision=False,
      next_fit_requires_auxiliary_rebind=True,next_fit_consumed=False,
      source_changes=[dict(source_row_id=r['source_row_id'],native_pts=r['native_pts'],old_UV=r['old_packet_uv'],new_UV=r['current_uv'])for r in revisions])
    write_json(out/'packet_delta.json',report);return report
