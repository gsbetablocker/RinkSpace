"""Read-only statistics/source-operator self audit with immutable row/group scope.

A matching hash or operator token is never promoted to physical pixel identity.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,re
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract
from .packet_compile import SnapshotResolver

def audit_statistics(project,run,output):
    project=Path(project).resolve();run=Path(run).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable audit generation required')
    out.mkdir(parents=True);contract=EventMeasurementContract(project,'ankara_2026')
    packet_names=['joint_packet_v1','joint_packet_v2','joint_packet_v3','joint_packet_dense_v1','corrected_membership_v1','corrected_membership_v2']
    packets=[];hits=defaultdict(int);active_maps=[];refs=[];source_codes=Counter();current_codes=Counter()
    def load_array(p,src,name):
        ref=p['arrays'][name];path=Path(ref['file']);path=path if path.is_absolute() else src.parent/path
        return np.fromfile(path,ref['dtype']).reshape(ref['shape'])
    for name in packet_names:
        path=run/name/'packet.json'
        if not path.exists():continue
        p=json.loads(path.read_text(encoding='utf8'));mask=(load_array(p,path,'admission')==1)&(load_array(p,path,'fit_role')==1);native=load_array(p,path,'native_pts')
        bit=1<<len(packets);packets.append(dict(name=name,file=str(path),sha256=file_sha256(path),active=int(mask.sum()),bit=bit))
        for i in np.flatnonzero(mask):hits[(int(native[i]),p['source_row_ids'][i])]|=bit
        if name=='corrected_membership_v2':current=p;current_path=path;current_mask=mask;current_native=native
        if name=='joint_packet_dense_v1':baseline=p;baseline_path=path;baseline_mask=mask
    current_uv=load_array(current,current_path,'uv');current_E=load_array(current,current_path,'E')
    ix={rid:i for i,rid in enumerate(current['source_row_ids'][current['new_source_start']:],start=current['new_source_start'])}
    group_index={};group_map={}
    for g in current['groups']:
        for i in g['rows']:
            if i in group_index:raise ValueError('current active statistical row repeated')
            group_index[i]=g['id']
        group_map[g['id']]=g
    selected=list(read_jsonl(run/'admission_v4/solver_rows.jsonl.gz'));selected_by={r['source_row_id']:r for r in selected}
    selected_exposures={(r['shot_id'],int(r['native_pts'])) for r in read_jsonl(run/'source_audit_v5/relocated_rows.jsonl.gz')}
    stat_rows={};stat_reg=json.loads((run/'statistics_combined_v2/statistics_registry.json').read_text(encoding='utf8'))
    for gid,g in stat_reg.items():
        for rid in g.get('row_ids',[]):stat_rows[rid]=(gid,g)
    dense_reg=json.loads((run/'dense_statistics_ready_v1/statistics_registry.json').read_text(encoding='utf8'))
    for gid,g in dense_reg.items():
        for rid in g.get('row_ids',[]):stat_rows[rid]=(gid,g)
    counts=Counter();families=defaultdict(lambda:dict(rows=[],current_groups=set(),source_groups=set(),entities=[],curves=[]));manual=[];same_exposure=[]
    auditfile=out/'new_source_rows.jsonl.gz'
    inputs=[('selected',run/'source_audit_v5/relocated_rows.jsonl.gz'),('dense',run/'dense_admission_v1/normalized_source_journal.jsonl.gz')]
    with gzip.open(auditfile,'wt',encoding='utf8') as f:
      for dataset,path in inputs:
        refs.append(dict(file=str(path),sha256=file_sha256(path)))
        for raw in read_jsonl(path):
            rid=raw['source_row_id'];r=selected_by.get(rid,raw) if dataset=='selected' else raw;i=ix.get(rid)
            active=bool(i is not None and current_mask[i]);prior=hits.get((int(r['native_pts']),rid),0);codes=[];m=r.get('source_measurement_audit') or {};is_entity=bool(r.get('point_correspondence'))
            inspected=contract.inspect(r);codes.extend(inspected['definition_problems']);operator=inspected['actual_operator'];expected=inspected['expected']
            if active and not np.array_equal(current_uv[i],np.asarray(r['current_uv'],float)):codes.append('packet_active_mean_differs_from_source_generation')
            manual_click=m.get('status')=='manual_native_click_preserved' or r.get('evidence_class')=='manual_native_click'
            if manual_click:
                if r.get('current_uv') is not None and not np.array_equal(np.asarray(r['current_uv']),np.asarray(r['source_uv'])):codes.append('manual_click_current_UV_silently_changed')
                if rid in stat_rows and not is_entity:codes.append('manual_click_noise_wrongly_uses_automatic_profile_Jacobian')
                manual.append(dict(source_row_id=rid,native_pts=r['native_pts'],current_active=active,packet_hit_mask=prior,codes=codes.copy()))
            if dataset=='selected' and rid in stat_rows and not is_entity and not manual_click:
                codes.extend(['old_pixel_W_linearizes_current_peak_not_frozen_original_seed_mean','old_pixel_W_float64_colour_differs_actual_float32_operator','old_pixel_W_is_not_verified_actual_measurement_noise'])
            if dataset=='dense' and (r['shot_id'],int(r['native_pts'])) in selected_exposures:
                codes.append('same_selected_exposure_dense_must_not_be_counted_again');same_exposure.append(dict(source_row_id=rid,native_pts=r['native_pts'],active=active))
            if dataset=='dense' and rid in stat_rows:
                codes.extend(['dense_five_tangent_profile_geometry_surrogate_not_source_noise_likelihood','dense_anchor_originally_lacked_existing_fact_contract_guard'])
            flank_summary=None
            if operator=='paint_center' and not manual_click:
                fl=m.get('flank_uv',[]);pr=m.get('flank_profiles',[])
                if len(fl)==2 and r.get('current_uv') is not None:
                    a=np.asarray(fl,float);sep=float(np.linalg.norm(a[1]-a[0]));delta=float(np.linalg.norm(a.mean(0)-np.asarray(r['current_uv'])))
                    flank_summary=dict(separation_px=sep,midpoint_to_current_mean_px=delta,channels=[z.get('channel') for z in pr],polarities=[z.get('polarity') for z in pr])
                    if sep==0:codes.append('two_width_flanks_are_the_same_actual_coordinate')
                    if delta>1e-9:codes.append('paint_current_mean_is_not_its_recorded_flank_midpoint')
                    if len(pr)==2 and pr[0].get('channel')==pr[1].get('channel') and pr[0].get('polarity')==pr[1].get('polarity'):codes.append('same_channel_flanks_have_same_polarity_requires_material_reaudit')
                codes.append('old_two_flank_absgradient_does_not_prove_same_stroke_plateau')
            if is_entity and rid in stat_rows:
                gid,sg=stat_rows[rid]
                if sg.get('rank')!=2 or len(sg.get('row_ids',[]))!=1:codes.append('entity_2D_rank_or_fixed_source_group_invalid')
                codes.append('entity_per_patch_conditional_covariance_not_complete_shared_family_noise')
            if expected and expected.get('physical_role')=='declared_nested_target_material_interface':codes.append('target_ring_layer_requires_actual_declared_material_pair_proof_not_token')
            if expected and expected.get('operator') in ['poster_boundary','ink_boundary']:codes.append('printed_or_panel_contour_actual_visible_material_segment_certificate_required')
            if expected and expected.get('operator') in ['board_upper_material_edge','ice_board_interface']:codes.append('generic_absgradient_has_not_proved_user_outer_board_role')
            if active:
                key=(r['shot_id'],int(r['native_pts']),r['source_family_id']);fam=families[key];fam['rows'].append(i);fam['current_groups'].add(group_index.get(i));fam['source_groups'].add(r.get('statistical_group_id') or stat_rows.get(rid,(None,None))[0]);fam['entities' if is_entity else 'curves'].append(rid)
            for code in set(codes):source_codes[code]+=1;current_codes[code]+=int(active)
            counts[dataset+'_raw_rows']+=1;counts[dataset+'_current_active']+=int(active);counts[dataset+'_prior_active_any']+=int(prior!=0)
            compact=dict(source_row_id=rid,source_family_id=r['source_family_id'],native_pts=r['native_pts'],shot_id=r['shot_id'],dataset=dataset,joint_row=i,current_active=active,packet_hit_mask=prior,actual_operator=operator,expected_operator=expected.get('operator') if expected else None,expected_physical_role=expected.get('physical_role') if expected else None,actual_mean_operator_status=m.get('status'),statistical_group_id=r.get('statistical_group_id') or stat_rows.get(rid,(None,None))[0],point_correspondence=is_entity,source_and_current_UV_bound_hashes_are_physical_identity_proof=False,codes=sorted(set(codes)),flanks=flank_summary)
            f.write(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
    famout=[]
    for (shot,native,family),v in families.items():
        if len(v['current_groups'])>1:
            famout.append(dict(source_family_id=family,shot_id=shot,native_pts=native,rows=v['rows'],current_statistical_groups=sorted(x for x in v['current_groups'] if x),source_group_count=len(v['source_groups']),entity_rows=v['entities'],curve_rows=v['curves'],shared_family_covariance_complete=False,old_geometric_group_weight_count=len(v['current_groups']),replacement='family_geometry_initialization_v1 combines all same-family geometry rows without claiming noise covariance'))
    write_json(out/'source_family_fragmentation.json',famout);write_json(out/'manual_click_rows.json',manual);write_json(out/'same_selected_exposure_dense_rows.json',same_exposure)
    with gzip.open(out/'statistical_groups.jsonl.gz','wt',encoding='utf8') as f:
      for g in baseline['groups']:
        old=g['rows'];after=[i for i in old if current_mask[i]];is_s01=all(i<current['new_source_start'] for i in old)
        typ='inherited_original_full_W_numerically_unchanged_source_definition_audit_pending' if is_s01 else ('per_entity_conditional_two_direction_surrogate' if str(g.get('source_group_id','')).startswith('entity:') else ('dense_empirical_tangent_geometry_surrogate' if any(i>=baseline['newly_appended_source_start'] for i in old) else 'current_peak_linearization_wrong_frozen_mean_operator'))
        rec=dict(id=g['id'],source_group_id=g.get('source_group_id'),rank=g['rank'],rows=old,baseline_active_rows=len(old),corrected_active_rows=len(after),removed_rows=sorted(set(old)-set(after)),source_statistics_interpretation=typ,source_W=g['W'],covariance_subselection_method='original whole group retained' if len(old)==len(after) else ('whole physical group withdrawn' if not after else 'requires exact covariance marginal, never W column slicing'),formal_new_noise_likelihood_validated=False if not is_s01 else None)
        f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n')
    contract.verify_unchanged();write_json(out/'confirmed_fact_refs.json',dict(sha256=contract.digest,refs=contract.references));write_json(out/'packet_hits_legend.json',packets)
    report=dict(schema='rinkfit.statistics_actual_measurement_self_audit.v1',counts=dict(counts),all_source_code_counts=dict(source_codes),current_active_code_counts=dict(current_codes),fragmented_current_source_families=len(famout),fragmented_current_source_rows=sum(len(x['rows']) for x in famout),manual_click_rows=len(manual),manual_click_current_active=sum(x['current_active'] for x in manual),same_selected_exposure_dense_rows=len(same_exposure),same_selected_exposure_dense_active=sum(x['active'] for x in same_exposure),current_packet=dict(file=str(current_path),sha256=file_sha256(current_path)),input_refs=refs,contract_sha256=contract.digest,source_UV_or_flags_modified=False,noise_W_bytes_modified=False,physical_RGB_identity_audit_completed=False,formal_new_source_W_approved=False,old_source_noise_claim='withdrawn for current-seed/precision/manual-operator mismatches; immutable historical bytes retained',current_solver_geometry_use='explicit source-family/shot balanced native geometry surrogate only; S01 original W retained as inherited authority',next_source_statistics='new physical_source_operator frozen original-seed/current-mean replay, actual ordered material proof, whole shared RGB exposure/family covariance; never relabel previous wrong operator W')
    write_json(out/'manifest.json',report);return report
