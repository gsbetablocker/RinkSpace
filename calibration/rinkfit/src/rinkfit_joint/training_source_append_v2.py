"""Explicit source append for an existing thin selection; old rows never backfill."""
from pathlib import Path
import numpy as np
from .training_source_selection_v2 import checked, require, reference
from .io_utils import read_json


def same_ref(a, b):
    return Path(a['file']).resolve() == Path(b['file']).resolve() and a['sha256'] == b['sha256']


def verify_append(proof_file, current_ref, previous, reader):
    proof_ref=reference(proof_file); proof=read_json(checked(proof_ref))
    require(proof['schema']=='rinkfit.authenticated_native_source_append_proof.v1' and proof['passed'] is True,
            'Authenticated native source append proof required')
    require(same_ref(proof['source_packet'],current_ref) and same_ref(proof['original_state_packet'],previous['source_packet']),
            'Append proof binds another current/original source packet')
    n0=int(proof['parent_raw_rows']);n=int(proof['current_raw_rows'])
    require(previous['source_row_count']==n0 and reader.packet['row_count']==n and n>n0 and proof['new_source_rows']==n-n0,
            'Append prefix/full raw denominator differs')
    checked(proof['parent_packet']); checked(proof['original_state_packet']); checked(proof['source_evidence'])
    require(reader.packet['native_ad_source_binding']==proof['source_registry'] and
            proof['source_registry']['row_start']==n0 and proof['source_registry']['count']==n-n0,
            'Actual appended registry differs from independently checked source')
    require(len(proof['prefix_arrays'])==len(reader.packet['arrays']) and
            {x['array'] for x in proof['prefix_arrays']}==set(reader.packet['arrays']) and
            all(x['prefix_bitwise_equal'] for x in proof['prefix_arrays']), 'Incomplete original array prefix proof')
    require(not proof['source_model_projection_used'] and not proof['failed_or_missing_measurements_filled'] and
            not proof['old_failed_queries_restored'] and not proof['new_known_W_created'] and
            proof['all_new_groups_rank_zero'] and proof['source_midline_definition']=='arithmetic_width_midpoint_of_two_actual_material_edges',
            'Append fabricated physical/source/statistical authority')
    rows=proof['rows']; require(len(rows)==n-n0 and {x['raw_row'] for x in rows}==set(range(n0,n)), 'Append proof lost raw source rows')
    cols={k:reader.load(k) for k in ('native_pts','shot_index','material_index','path','uv','kind')}
    from .compact_identity_withhold_v2 import _source_identities
    identities,_=_source_identities(reader.path,range(n0,n))
    for row in rows:
        i=int(row['raw_row'])
        require(row['source_row_id']==identities[i] and row['native_pts']==int(cols['native_pts'][i]) and row['shot_id']==reader.packet['shot_ids'][int(cols['shot_index'][i])] and
                row['material_index']==int(cols['material_index'][i]) and row['path']==int(cols['path'][i]) and
                np.array_equal(row['source_uv'],cols['uv'][i]) and cols['kind'][i]==0 and
                row['source_identity_confirmed'] is True and row['reading_status']=='measured_paired_stroke_midline' and
                row['max_absolute_replay_difference_px']<=1e-6, 'Appended RGB/source tuple differs from actual packet')
    require(all(int(g['rank'])==0 for g in reader.packet['groups'] if any(i>=n0 for i in g['rows'])),
            'Append selection may not replace positive-rank source covariance')
    # Re-check current prefix bytes against its immediate authenticated parent;
    # a saved pass flag alone cannot authorize another full-array generation.
    import json
    parent=json.loads(Path(checked(proof['parent_packet'])).read_text(encoding='utf-8-sig'))
    from .effective_input_audit_v1 import CheckedBytes
    store=CheckedBytes()
    for key,desc in reader.packet['arrays'].items():
        current=reader.load(desc); old=store.array(parent['arrays'][key],Path(proof['parent_packet']['file']).parent)[0]
        require(np.array_equal(np.ascontiguousarray(current[:n0]).view('u1'),np.ascontiguousarray(old).view('u1')),
                'New packet changed original raw prefix '+key)
    return proof_ref,proof,n0,n


def appended_representatives(reader, model, eligible, canonical):
    """Producer v4's exact 48px/2-direction and component/source extrema rule."""
    ids=np.asarray(eligible,dtype=np.int64)
    if not len(ids):return ids
    uv=reader.load('uv')[ids];normal=reader.load('profile_normal')[ids]
    shot=reader.load('shot_index')[ids];pts=reader.load('native_pts')[ids]
    mat=reader.load('material_index')[ids];path=reader.load('path')[ids];kind=reader.load('kind')[ids]
    protect=np.array([model['materials'][int(mi)]['label'].startswith(('board.texture.','ice.static.')) for mi in mat])|(kind==1)
    spacing=48.;cells=np.floor(uv/spacing).astype(np.int64)
    direction=np.minimum((np.mod(np.arctan2(normal[:,1],normal[:,0]),np.pi)/(np.pi/2)).astype(np.int16),1)
    keys=np.rec.fromarrays([shot,pts,canonical[mat],kind,cells[:,0],cells[:,1],direction],names='shot,pts,entity,kind,x,y,direction')
    distance=np.sum((uv-(cells+.5)*spacing)**2,axis=1);order=np.lexsort((ids,distance,keys));sorted_keys=keys[order]
    select=protect.copy();select[order[np.r_[True,sorted_keys[1:]!=sorted_keys[:-1]]]]=True
    for arrays,names in (([shot,pts,canonical[mat],kind],'shot,pts,entity,kind'),([shot,mat,path],'shot,material,path')):
        key=np.rec.fromarrays(arrays,names=names);order=np.argsort(key,kind='stable');sk=key[order]
        starts=np.r_[0,np.flatnonzero(sk[1:]!=sk[:-1])+1];ends=np.r_[starts[1:],len(order)]
        for a,b in zip(starts,ends):
            local=order[a:b];xy=uv[local]
            select[local[[np.argmin(xy[:,0]),np.argmax(xy[:,0]),np.argmin(xy[:,1]),np.argmax(xy[:,1])]]]=True
    return ids[select]


def declaration(proof_ref, proof, original_selection, appended_fit_rows):
    return dict(schema='rinkfit.source_append_selection.v1', parent_packet=proof['parent_packet'],
                original_state_packet=proof['original_state_packet'], parent_raw_rows=proof['parent_raw_rows'],
                current_raw_rows=proof['current_raw_rows'], append_source_proof=proof_ref,
                appended_fit_rows=appended_fit_rows, original_selection=original_selection,
                old_domain_no_backfill=True, maximum_added_fit_rows=appended_fit_rows['shape'][0],
                representative_policy=dict(pixel_grid_spacing=48,orientation_bins=2,component_extrema_retained=True,
                                           static_contour_support=True,selection_uses_fit_residual=False))


def validate_append_primal_policy(saved, cfg, current_packet_ref):
    """Launch preflight for the separately consumed native append-primal ABI."""
    import json
    from .observed_primal_restart import checkpoint_header
    from .dense_joint_run import representative_identity_ancestry
    p=cfg['representative_source_append_primal_v1']
    require(p['schema']=='rinkfit.authenticated_source_append_primal.v1' and
            p['from_source_packet']['sha256']==saved['source_packet_sha256'] and
            same_ref(p['source_packet'],current_packet_ref), 'Append primal old/current source identity')
    for name,saved_name in (('from_model','model'),('from_checkpoint','checkpoint'),('from_nuisance_manifest','explicit_nuisance_manifest')):
        require(same_ref(p[name],saved[saved_name]),'Append complete primal binding '+name);checked(p[name])
    require(p['from_state_token']==saved['state_token'] and p['old_domain_reseeded'] is False and
            p['tail_initialization']=='exact_authenticated_packet_station_row_alpha_path' and p['same_objective_resume'] is False,
            'Append may only initialize new raw source rows')
    proof=read_json(checked(p['append_source_proof']));current=json.loads(checked(current_packet_ref).read_text(encoding='utf-8-sig'))
    parent=json.loads(checked(p['parent_packet']).read_text(encoding='utf-8-sig'));old=json.loads(checked(p['from_source_packet']).read_text(encoding='utf-8-sig'))
    require(same_ref(proof['source_packet'],current_packet_ref) and same_ref(proof['original_state_packet'],p['from_source_packet']) and
            same_ref(current['dense_parent_packet'],p['parent_packet']) and same_ref(parent['dense_parent_packet'],p['from_source_packet']) and
            current['native_ad_source_binding']==proof['source_registry'],'Append actual metadata ancestry')
    n0,n=p['from_raw_rows'],p['to_raw_rows']
    require(old['row_count']==parent['row_count']==n0 and current['row_count']==n and n>n0 and
            proof['parent_raw_rows']==n0 and proof['current_raw_rows']==n,'Append raw denominator')
    for key in ('station','row','alpha','path'):
        require(current['arrays'][key]==p['tail_source_columns'][key],'Append actual tail initializer column');checked(p['tail_source_columns'][key])
    nuisance=read_json(checked(saved['explicit_nuisance_manifest']));base=Path(saved['explicit_nuisance_manifest']['file']).parent
    require(nuisance['packet_sha256']==saved['source_packet_sha256'] and nuisance['raw_rows']==n0,'Original complete raw auxiliary identity')
    for name,width in (('station.f64',8),('row.f64',8),('alpha.f64',8),('path.u32',4)):
        item=nuisance['files'][name];path=checked(dict(file=str(base/name),sha256=item['sha256']))
        require(path.stat().st_size==item['size_bytes']==n0*width,'Original complete raw auxiliary bytes')
    header=checkpoint_header(checked(saved['checkpoint']))
    require(header['state_token']==saved['state_token']==p['from_state_token'] and
            header['fingerprint']==p['from_runtime_fingerprint'] and header['source_packet_sha256']==saved['source_packet_sha256'],
            'Original complete checkpoint header/token/fingerprint')
    from .effective_input_prepare_v1 import actual_source_qualification
    qualification=read_json(checked(actual_source_qualification(cfg)))
    require(same_ref(qualification['source_packet'],current_packet_ref) and same_ref(qualification['parent_packet'],p['parent_packet']),
            'Actual source append qualification ancestry')
    chain=representative_identity_ancestry(saved['source_packet_sha256'],p['parent_packet'],parent['native_material_identity_revision'],qualification['parent_qualification'])
    require(chain['raw_rows']==n0 and not chain['rows'],'Old source domain may only withdraw, not rebind')
    selected=read_json(checked(cfg['representative_source_selection_v1']));extension=selected['source_append_selection_v1']
    require(same_ref(selected['original_selection'],saved['representative_source_selection_v1']) and
            same_ref(extension['append_source_proof'],p['append_source_proof']) and extension['old_domain_no_backfill'] is True,
            'Append selector differs from complete original representatives')
    return dict(schema='rinkfit.authenticated_source_append_primal_preflight.v1',passed=True,old_raw_rows=n0,current_raw_rows=n,
                complete_old_state_token=saved['state_token'],same_objective_resume=False,native_complete_roundtrip_still_required=True)
