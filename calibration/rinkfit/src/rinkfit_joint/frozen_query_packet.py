"""Retain real normal-profile acquisition conditions independently of noise W."""
from pathlib import Path
from collections import defaultdict,Counter
import gzip,json
import numpy as np
from .io_utils import file_sha256,read_jsonl,write_json,object_sha256


def canonicalize_appended_aliases(packet,out,*,old_row_count,actual_source_aliases):
    out=Path(out);p=packet;checked={}
    def ref(f):
        f=Path(f).resolve();k=str(f)
        if k not in checked:checked[k]=file_sha256(f)
        return {'file':k,'sha256':checked[k]}
    def load(k):
        d=p['arrays'][k];return np.memmap(d['file'],dtype=d['dtype'],mode='r',shape=tuple(d['shape']))
    indexfile=Path(actual_source_aliases);idx=json.loads(indexfile.read_text(encoding='utf8'))
    for d in (idx['audit_manifest'],idx['aliases'],idx['producer_script']):
        if ref(d.get('file',d.get('path')))['sha256']!=d['sha256']:raise ValueError('changed actual source alias evidence')
    ad=np.array(load('admission'));role=load('fit_role');pts=load('native_pts');uv=load('uv');mi=load('material_index');kind=load('kind');E=load('E');active=(ad==1)&(role==1)
    byid={(rid,int(pts[i])):i for i,rid in enumerate(p['source_row_ids'])};decisions=[]
    for a in read_jsonl(idx['aliases'].get('file',idx['aliases'].get('path'))):
        i=byid.get((a['source_row_id'],int(a['native_pts'])))
        if i is None or not active[i]:continue
        if i<old_row_count:raise ValueError('new audit changes inherited active source membership; full covariance marginal required')
        j=byid.get((a['actual_source_readout_alias_of'],int(a['native_pts'])))
        if j is None or not active[j]or i==j:raise ValueError('canonical source readout not currently active')
        if not np.array_equal(uv[i],uv[j])or mi[i]!=mi[j]or kind[i]!=kind[j]or E[i]!=E[j]:raise ValueError('source readout alias conflicts with current physical measurement')
        decisions.append(dict(joint_row=i,canonical_joint_row=j,source_row_id=a['source_row_id'],actual_source_readout_alias_of=a['actual_source_readout_alias_of'],native_pts=int(a['native_pts']),exact_same_actual_measurement_input_sha256=a['exact_same_actual_measurement_input_sha256'],physical_identity_withdrawn=False))
    removed={x['joint_row']for x in decisions}
    if any(x['canonical_joint_row']in removed for x in decisions):raise ValueError('unresolved canonical readout alias chain')
    ad[list(removed)]=0;active=(ad==1)&(role==1);groups=[]
    for g in p['groups']:
        kept=[i for i in g['rows']if active[i]]
        if kept==g['rows']:groups.append(g);continue
        if g.get('statistic_kind')!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]:raise ValueError('never slice a statistical W when canonicalizing source aliases')
        if kept:groups.append({**g,'rows':kept})
    f=out/'admission.bin';ad.tofile(f);p['arrays']['admission']={**p['arrays']['admission'],**ref(f)};p['groups']=groups;p['active_rows']=int(active.sum());p['recapture_rows']=np.flatnonzero(active&(np.arange(len(ad))>=p['new_source_start'])).tolist()
    p['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise');p['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups)
    proof=out/'new_source_alias_canonicalization.json';write_json(proof,dict(actual_source_aliases=ref(indexfile),only_new_raw_rows=decisions,old_statistical_W_or_inherited_admission_changed=False,source_mean_material_and_raw_history_unchanged=True));p['newly_appended_actual_source_alias_membership']=ref(proof)
    return p


def add_unknown_genuine_profile_queries(packet,out,*,project_run):
    p=packet;out=Path(out);run=Path(project_run);n=p['row_count'];cache={}
    def ref(f):
        f=Path(f).resolve();k=str(f)
        if k not in cache:cache[k]=file_sha256(f)
        return {'file':k,'sha256':cache[k]}
    def load(k):
        d=p['arrays'][k];return np.memmap(d['file'],dtype=d['dtype'],mode='r',shape=tuple(d['shape']))
    ad=load('admission');role=load('fit_role');uv=load('uv');pts=load('native_pts');material=load('material_index');kind=load('kind');shot_index=load('shot_index');alpha=np.array(load('alpha'));bounds=np.array(load('alpha_bounds'));queries=list(p.get('queries',[]));existing={q['row_index']for q in queries}
    unknown={i for g in p['groups']if g.get('statistic_kind')=='constraint_only_unknown_noise'for i in g['rows']}
    journal_refs={str(Path(d['file']).resolve()):d for d in p.get('source_family_journal_refs',[])};needed=defaultdict(dict);unsupported=[]
    for i in sorted(unknown):
        if not(ad[i]==1 and role[i]==1):raise ValueError('unknown source not active under engineering constraints')
        if i in existing:continue
        dataset=p['source_dataset_ids'][i]
        if kind[i]!=0 or not dataset.startswith('actual_geometry:'):
            unsupported.append(dict(joint_row=i,source_row_id=p['source_row_ids'][i],reason='actual manual/entity or non-profile readout; no deterministic normal Query asserted'));continue
        tag=dataset.split(':')[1];parts=tag.split('.')
        file=run/parts[0]/parts[1]/'source_rows.jsonl.gz'if len(parts)==2 else run/tag/'source_rows.jsonl.gz';file=file.resolve()
        if str(file)not in journal_refs:raise ValueError('actual source journal absent from immutable source closure')
        if ref(file)['sha256']!=journal_refs[str(file)]['sha256']:raise ValueError('changed actual source journal')
        needed[file][(p['source_row_ids'][i],int(pts[i]))]=i
    provenance=[];added=[];seen=set();max_error=0.;counts=Counter();records=out/'actual_unknown_profile_query_proofs.jsonl.gz'
    with gzip.open(records,'wt',encoding='utf8')as stream:
      for file,lookup in needed.items():
        journal_index=len(provenance);provenance.append(ref(file))
        for r in read_jsonl(file):
            i=lookup.get((r['source_row_id'],int(r['native_pts'])))
            if i is None:continue
            if i in seen:raise ValueError('duplicate actual source profile record')
            seen.add(i);a=r.get('source_measurement_audit',{});profile=a.get('profile',{});continuous=a.get('continuous_cubic_material_profile',{});cfg=profile.get('configuration',{})
            genuine=(r.get('source_geometry_eligible')is True and a.get('actual_outer_material_order_verified')is True and a.get('reflection_excluded')is True and a.get('internal_interface_used')is False and a.get('global_color_rule_used_as_object_identity')is False and profile.get('interpolation')=='continuous_cubic'and profile.get('valid')is True and cfg.get('material_channel_polarity_and_search_seed_frozen')is True and continuous.get('operator_version')=='fixed_outer_material_continuous_cubic_peak_v1'and continuous.get('material_reselected')is False and continuous.get('numeric_valid')is True)
            finite_receipt=a.get('source_material_receipt',{});finite=r.get('source_generation')=='Ad4_R_current_source_v2'
            if finite:
                certificate=finite_receipt.get('actual_finite_material_certificate',{});continuous=finite_receipt.get('actual_point_reading',{});cfg=finite_receipt.get('configuration',{});frozen=finite_receipt.get('frozen_point_operator',{})
                genuine=(r.get('source_geometry_eligible')is True and r.get('physical_identity_passed')is True and int(r.get('joint_material_index',-1))==73 and r.get('shot_id')=='s09' and r.get('measurement_operator_resolved')=='ink_boundary' and r.get('pixels_from_model')is False and r.get('source_ink_geometry_inferred_from_template')is False and certificate.get('source_current_local_R_identity_actual_RGB_review')is True and certificate.get('no_model_geometry_for_source_UV')is True and finite_receipt.get('source_mean_from_model_projection')is False and finite_receipt.get('measurement_operator_material_identity_reselected_in_replay')is False and continuous.get('numeric_valid')is True and continuous.get('material_reselected')is False and continuous.get('operator_version')=='fixed_outer_material_continuous_cubic_peak_v1')
                profile={**frozen.get('source_measurement_audit',{}).get('profile',{}),'normal_offsets':continuous.get('actual_profile_normal_offsets')}
            if not genuine:
                unsupported.append(dict(joint_row=i,source_row_id=r['source_row_id'],reason='source record does not certify this actual frozen continuous profile operator'));continue
            off=np.asarray(profile['normal_offsets'],float);normal=np.asarray(r['normal'],float);seed=np.asarray(r['navigation_seed_uv'],float);mu=np.asarray(r['current_uv'],float)
            if off.ndim!=1 or len(off)<4 or not np.isfinite(off).all()or not np.all(np.diff(off)>0)or normal.shape!=(2,)or seed.shape!=(2,)or abs(np.linalg.norm(normal)-1)>1e-9:raise ValueError('invalid actual finite source profile geometry')
            if profile.get('channel')not in (0,1,2)or profile.get('polarity')not in(-1,1):raise ValueError('actual source profile channel/polarity missing')
            if (not finite and not np.array_equal(off,np.asarray(cfg['normal_offsets'])))or not np.array_equal(off,np.asarray(continuous['actual_profile_normal_offsets'])):raise ValueError('source frozen window differs from actual acquisition')
            if int(r['joint_material_index'])!=int(material[i])or not np.array_equal(mu,uv[i]):raise ValueError('genuine Query must not change source mean or physical material')
            if finite and(not np.array_equal(seed,np.asarray(frozen['navigation_seed_uv']))or not np.array_equal(normal,np.asarray(frozen['normal']))or cfg.get('profile_halfwidth_native_px')!=3.):raise ValueError('finite actual source frozen ray/window mismatch')
            t=float(continuous['normal_offset_px']);replay=seed+t*normal;err=float(np.max(abs(replay-mu)));max_error=max(max_error,err)
            lo=float(off[0]);hi=float(off[-1])
            if err>1e-9 or not(lo<=t<=hi)or not np.array_equal(np.asarray(continuous['normal']),normal)or not np.array_equal(np.asarray(continuous['current_uv']),mu):raise ValueError('actual profile Query is not the same source mean')
            q=dict(row_index=i,alpha=[lo,hi],native_uv=[(seed+lo*normal).tolist(),(seed+hi*normal).tolist()],normal=normal.tolist(),source_row_id=p['source_row_ids'][i],actual_acquisition_window_preserved=True,deterministic_query_is_not_tangential_noise_precision=True)
            queries.append(q);alpha[i]=t;bounds[i]=[lo,hi];added.append(i);counts[p['shot_ids'][int(shot_index[i])]]+=1
            condition=dict(native_pts=int(r['native_pts']),source_row_id=r['source_row_id'],current_uv=mu.tolist(),navigation_seed_uv=seed.tolist(),normal=normal.tolist(),normal_offsets=off.tolist(),channel=profile['channel'],polarity=profile['polarity'],configuration=cfg,operator_version=continuous['operator_version'])
            proof=dict(joint_row=i,source_journal_index=journal_index,frozen_actual_acquisition_sha256=object_sha256(condition),base_mean_replay_max_px=err,source_noise_W_created=False)
            stream.write(json.dumps(proof,ensure_ascii=False,separators=(',',':'))+'\n')
    expected=sum(len(x)for x in needed.values())
    if len(seen)!=expected:raise ValueError('unknown actual profile missing from source journal')
    for k,value in [('alpha',alpha),('alpha_bounds',bounds)]:
        f=out/(k+'.bin');value.tofile(f);p['arrays'][k]={**p['arrays'][k],**ref(f)}
    write_json(out/'non_profile_or_unsupported_query_rows.json',unsupported)
    report=dict(schema='rinkfit.actual_frozen_profile_conditions_without_noise_precision.v1',actual_source_journals=provenance,new_genuine_profile_queries=len(added),by_shot=dict(counts),new_query_row_indices=added,non_profile_or_unsupported_rows=len(unsupported),unsupported=ref(out/'non_profile_or_unsupported_query_rows.json'),proofs=ref(records),actual_base_mean_replay_max_px=max_error,source_UV_normal_window_operator_or_material_changed=False,new_noise_W_ridge_floor_or_sigma_created=False,old_S01_queries_or_four_aux_changed=False,world_projection_used_to_create_or_enlarge_query=False)
    write_json(out/'genuine_profile_query_report.json',report);p['queries']=queries;p['unknown_noise_actual_frozen_query_conditions']=ref(out/'genuine_profile_query_report.json');p['new_actual_unknown_noise_profile_queries']=len(added);p['only_new_profile_Query_alpha_and_bounds_initialized']=True;p['source_UV_PTS_material_and_measurement_operator_changed']=False;p['unknown_noise_source_statistical_precision_supplied']=False
    return p
