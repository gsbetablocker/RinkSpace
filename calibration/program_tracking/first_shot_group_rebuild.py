"""Exact reviewed native-support correction after the frozen v21 pipeline.

Only one explicitly addressed complete derived group can change. Historical
pixels, readouts, sigma, operator and localization evidence remain archived.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MethodType
import numpy as np
from .first_shot_source_transition import reference,read,_digest,compare_source_frames,validate_manifest
from .first_shot_source_evidence import rebuild_final_native_line_estimator
from .first_shot_noise_revision import make_noise_replacement
from .source_evidence_whitening import SourceWhiteningOperator,latent_whitening_operator

SCHEMA='rinkspace.reviewed-post-native-group-support-rebuild.v1'
TRANSITION_SCHEMA='rinkspace.verified-whole-source-group-continuation.v1'
FIELD='post_native_source_group_revision'
_CONFIG=None


def position_scatter(line):
    """Same archived LOO degree-one/two nuisance and RSS/df variance rule."""
    points=np.asarray([r['uv'] for r in line['direct_readouts']],float)
    t=(points-line['point_origin'])@line['unit_tangent']
    d=(points-line['point_origin'])@line['unit_normal']
    x=(t-t.mean())/np.ptp(t);fits=[]
    for degree in (1,2):
        X=np.vander(x,degree+1,increasing=True);rank=int(np.linalg.matrix_rank(X));df=len(t)-rank
        if rank!=degree+1 or df<=2:continue
        c=np.linalg.lstsq(X,d,rcond=None)[0];residual=d-X@c
        Q=np.linalg.qr(X,mode='reduced')[0];leverage=np.sum(Q*Q,axis=1)
        if not np.all(1-leverage>1e-10):continue
        rss=float(residual@residual);loo=residual/(1-leverage)
        fits.append(dict(degree=degree,rank=rank,residual_df=df,RSS_px2=rss,
            LOO_RMS_px=float(np.sqrt(np.mean(loo*loo))),dof_sigma_px=float(np.sqrt(rss/df))))
    if not fits:raise ValueError('Original source-only variance rule is not estimable')
    chosen=min(fits,key=lambda v:(v['LOO_RMS_px'],v['degree']))
    if not np.isfinite(chosen['dof_sigma_px']) or chosen['dof_sigma_px']<=0:
        raise ValueError('Native source scatter is not positive and finite')
    return chosen,fits


def derive(item):
    if item.get('schema')!=SCHEMA:raise ValueError('Explicit reviewed group revision required')
    audit=read(item['support_audit']);native=read(item['native_readout_packet'])
    packet=read(item['source_packet']);binding=read(item['base_binding'])
    if (audit.get('review_complete')is not True or audit.get('authority')!='agent_rgb_review'
            or audit.get('projection_or_residual_used_for_classification')is not False
            or audit['source_packet']!=item['native_readout_packet']
            or packet['ice_source_readout_packet']!=item['native_readout_packet']):
        raise ValueError('Independent complete original native RGB scope required')
    for source in (packet['source'],native['source']):
        if (source['native_pts']!=item['native_pts'] or source['actual_pts']!=item['native_pts']
                or source['time_base']!=binding['source_clock']['time_base']
                or source['source_video_sha256']!=binding['source']['sha256']):
            raise ValueError('Reviewed support has another native source clock/video')
    if packet['source']['decoded_BGR_sha256']!=native['source']['decoded_BGR_sha256']:
        raise ValueError('Reviewed support RGB differs from original packet')
    line=next(f['source_line_fit'] for f in native['features'] if f['id']==item['feature_id'])
    model=read(binding['model']);registry=read(model['feature_registry'])
    expected_target=registry['targets'][item['feature_id']]
    from .first_shot_fit import _continuous_line_metadata
    expected_metadata=_continuous_line_metadata(expected_target,read(model['scene']))
    if (item['source_registry_reference']!=model['feature_registry'] or item['initial_scene_reference']!=model['scene']
            or item['material_target']!=expected_target or item['expected_line_metadata']!=expected_metadata):
        raise ValueError('Recorded target/continuous metadata must equal the exact source registry and initial scene')
    rows=line['direct_readouts'];reviews=audit['all_readouts'];kept=[];excluded=[]
    if len(rows)!=len(reviews) or len(rows)!=audit['original_native_query_count']:
        raise ValueError('Every original native support entry must be independently reviewed')
    checked=set()
    for i,(raw,review) in enumerate(zip(rows,reviews,strict=True)):
        if (review['native_readout_array_index']!=i
                or review['native_query_index']!=raw['source_evidence']['query_index']
                or review['original_UV']!=raw['uv'] or review['original_sigma_px']!=raw['sigma_px']
                or review['source_evidence']!=raw['source_evidence']
                or review.get('authority')!='agent_rgb_review' or review.get('user_confirmed')is not False
                or review.get('actually_viewed_RGB')is not True):
            raise ValueError('Review does not preserve the exact native readout/evidence')
        for ref in review['evidence_references']:
            key=ref['path'],ref['sha256']
            if key not in checked:
                if reference(ref['path'])!={k:ref[k] for k in ('path','sha256')}:
                    raise ValueError('Original RGB evidence changed')
                checked.add(key)
        if review.get('admit_to_rebuilt_estimator')is True:kept.append(i)
        elif review.get('admit_to_rebuilt_estimator')is False:excluded.append(i)
        else:raise ValueError('Each support decision must be explicit')
    if ([rows[i]['source_evidence']['query_index'] for i in kept]!=audit['retained_query_indices']
            or [rows[i]['source_evidence']['query_index'] for i in excluded]!=audit['quarantined_query_indices']
            or kept!=item['retained_original_array_indices'] or excluded!=item['excluded_original_array_indices']):
        raise ValueError('Support disposition differs from the exact reviewed scope')
    members=item['original_members'];ids=tuple(m['source_point_id'] for m in members)
    if ids!=tuple(item['original_noise_group']['new_operator']['source_point_ids']):
        raise ValueError('Original complete group membership differs')
    if [(m['source_point_id'],m['original_UV']) for m in members]!=[(m['source_point_id'],m['UV']) for m in audit['current_v21_immutable_representatives']]:
        raise ValueError('Original representative IDs/stations differ from reviewed inventory')
    old_choice,_=position_scatter(line);old_noise=item['original_noise_group']
    old_method=old_noise['noise_method'];L=np.asarray(old_noise['localization_noise_factor'])
    if (old_method.get('schema')!='rinkspace.source-only-native-position-scatter-candidate.v1'
            or old_method['selected_nuisance_degree']!=old_choice['degree']
            or not np.array_equal(L,old_choice['dof_sigma_px']*np.eye(len(rows)))):
        raise ValueError('Original group is not the authenticated existing scatter-noise rule')
    filtered=deepcopy(line);filtered['direct_readouts']=[deepcopy(rows[i]) for i in kept]
    rebuilt=rebuild_final_native_line_estimator(filtered,item['feature_id'],ids,
        np.asarray([m['original_UV'] for m in members]),group_id=item['group_id'])
    choice,fits=position_scatter(filtered)
    method=deepcopy(old_method);method.update(selected_nuisance_degree=choice['degree'],
        native_variance_px2=choice['RSS_px2']/choice['residual_df'],native_residual_df=choice['residual_df'])
    factor=choice['dof_sigma_px']*np.eye(len(kept))
    final=latent_whitening_operator(ids,rebuilt['physical_sensitivity'],
        explicit_noise_factor=rebuilt['influence']@factor,physical_basis=rebuilt['physical_basis'],
        group_id=item['group_id'],covariance_origin='native_image_localization_noise_through_unchanged_estimator')
    supplied=item.get('reconstructed')
    if supplied is not None and (supplied['mean_contract']!=rebuilt['metadata']
            or not np.array_equal(supplied['derived_UV'],rebuilt['uv'])
            or not np.array_equal(supplied['native_localization_factor'],factor)
            or not np.array_equal(SourceWhiteningOperator.from_record(supplied['complete_operator']).matrix,final.matrix)
            or supplied['noise_method']!=method):
        raise ValueError('Saved whole-group mean/D/covariance/W does not reproduce the reviewed support')
    return rebuilt,filtered,factor,method,fits


def read_transition(path):
    ref=reference(path);manifest=read(ref)
    if manifest.get('schema')!=TRANSITION_SCHEMA:raise ValueError('Explicit whole-group transition required')
    item=read(manifest['group_revision'])
    if item['base_binding']!=manifest['base_binding']:raise ValueError('Foreign group revision base')
    old,new=read(manifest['base_binding']),read(manifest['revised_binding'])
    before,after=read(manifest['base_association']),read(manifest['revised_association'])
    compare=deepcopy(new)
    if compare.pop(FIELD,None)!=manifest['group_revision'] or compare!=old:
        raise ValueError('Only the explicit post-native group revision binding field may change')
    a,b=deepcopy(before),deepcopy(after)
    if a.pop('source_binding')!=manifest['base_binding'] or b.pop('source_binding')!=manifest['revised_binding'] or a!=b:
        raise ValueError('Association may only bind the new exact source recipe')
    read(manifest['authenticated_v21_transition'])
    v21=validate_manifest(manifest['authenticated_v21_transition']['path'])[1]
    if v21['revised_binding']!=manifest['base_binding'] or v21['revised_association']!=manifest['base_association']:
        raise ValueError('Whole-group correction must follow the exact authenticated v21 source')
    for source in item['existing_scatter_rule_producers']:read_source=reference(source['path']);assert read_source==source
    derive(item)
    return ref,manifest,item,before,after


def _save_failure_frame(frame,item,diagnostic):
    """Keep one actual target frame for narrow replay; never mask gate errors."""
    try:
        import pickle
        directory=Path(item['support_audit']['path']).parent/'actual_frame_failures'
        directory.mkdir(exist_ok=True)
        producer=reference(__file__);token=producer['sha256'][:12]
        payload=pickle.dumps(frame,protocol=pickle.HIGHEST_PROTOCOL)
        name=token+'_'+hashlib.sha256(payload).hexdigest()[:12]
        target=directory/(name+'.pkl')
        if not target.exists():target.write_bytes(payload)
        receipt=dict(producer=producer,actual_frame=reference(target),diagnostic=diagnostic,
            only_target_frame_saved=True,full_source_bundle_not_saved=True)
        (directory/(name+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        return receipt['actual_frame']
    except Exception:return None


def apply_group(frame,item,item_ref):
    if frame['native_pts']!=item['native_pts'] or frame['source_packet']!=item['source_packet']:
        raise ValueError('Whole-group correction received another source frame')
    groups=frame['whitening_groups'];selected=[g for g in groups if g['operator'].group_id==item['group_id']]
    if len(selected)!=1:raise ValueError('One exact complete derived source group required')
    group=selected[0];points=[(r,j) for r in frame['rows'] for j in range(len(r['uv']))]
    members=[points[i] for i in group['point_indices']]
    if group.get('native_line_replacement')is not None:raise ValueError('This correction is explicitly for a derived group')
    if (_digest(group['operator'].to_record())!=_digest(item['original_noise_group']['new_operator'])
            or _digest(group.get('localization_noise_revision'))!=_digest(item['original_noise_group'])):
        raise ValueError('Current complete W/noise proof differs from the archived original group')
    for (row,j),member in zip(members,item['original_members'],strict=True):
        actual=dict(source_point_id=row['source_point_ids'][j],UV=row['uv'][j].tolist(),
            sigma_px=float(row['sigma'][j]),source_feature_id=row['source_feature_id'],target=row['target'],
            line_metadata={k:row[k] for k in ('_continuous_world_line','_line_start_extent_geometry_columns','_line_end_extent_geometry_columns') if k in row})
        expected=dict(source_point_id=member['source_point_id'],UV=member['original_UV'],
            sigma_px=member['original_sigma_px'],source_feature_id=item['feature_id'],target=item['material_target'],
            line_metadata=item['expected_line_metadata'])
        differences={k:dict(expected=expected[k],actual=actual[k]) for k in expected if actual[k]!=expected[k]}
        if differences:
            diagnostic=_save_failure_frame(frame,item,dict(stage='exact_original_member',differences=differences))
            raise ValueError('Original complete group membership differs: '+json.dumps(dict(
                source_point_id=member['source_point_id'],differences=differences,diagnostic=diagnostic),ensure_ascii=False))
    rebuilt,line,factor,method,fits=derive(item)
    positions={i:k for k,i in enumerate(group['point_indices'])}
    estimator=dict(schema='rinkspace.native-correlated-source-estimator-replacement.v1',
        source_packet=frame['source_packet'],native_readout_packet=item['native_readout_packet'],
        source_feature_id=item['feature_id'],original_members=deepcopy(item['original_members']),
        estimator_contract=rebuilt['metadata'],reviewed_support_group_revision=item_ref,
        raw_whitening_row_or_column_slice_used=False)
    rows=[];offset=0
    for old in frame['rows']:
        take=[offset+j in positions for j in range(len(old['uv']))]
        if any(take) and not all(take):raise ValueError('Whole-group edit cannot partially replace a source row')
        if all(take):
            indices=[positions[offset+j] for j in range(len(old['uv']))]
            row=dict(old,uv=rebuilt['uv'][indices].copy(),original_source_uv=old['uv'].copy(),
                source_estimator_revision=estimator,derived_correlated_source_representative=True,
                new_independent_native_readout=False)
        else:row=old
        rows.append(row);offset+=len(old['uv'])
    result=dict(frame,rows=rows)
    baseline=dict(group,operator=rebuilt['operator'])
    noise=make_noise_replacement(result,baseline,physical_basis=rebuilt['physical_basis'],
        estimator_output_sensitivity=rebuilt['physical_sensitivity'],coefficient_influence=rebuilt['influence'],
        original_support_factor=np.diag([r['sigma_px'] for r in line['direct_readouts']]),
        localization_noise_factor=factor,evidence_references=[item_ref,item['support_audit']],noise_method=method)
    updated=dict(group,operator=SourceWhiteningOperator.from_record(noise['new_operator']),
        original_support_operator=rebuilt['operator'].to_record(),localization_noise_revision=noise,
        source_estimator_revision=estimator,post_native_support_rebuild=item_ref,
        original_pre_rebuild_operator=group['operator'].to_record(),
        original_pre_rebuild_localization_noise_revision=deepcopy(group['localization_noise_revision']))
    result['whitening_groups']=[updated if g is group else g for g in groups]
    return result


def compare_group_transition(old_frames,new_frames,item,item_ref):
    normalized=[];used=0;changed_rows=[];formal_offset=0
    from .first_shot_fit import FrozenWhitening
    for old,new in zip(old_frames,new_frames,strict=True):
        if old['native_pts']==item['native_pts']:
            expected=apply_group(old,item,item_ref)
            def record(frame):
                from .first_shot_source_transition import _target
                rows=[dict({k:v for k,v in row.items() if k!='_target_key'},target=_target(row['target'])) for row in frame['rows']]
                return dict(frame,rows=rows,whitening_groups=[dict(g,operator=g['operator'].to_record()) for g in frame['whitening_groups']])
            if _digest(record(expected))!=_digest(record(new)):
                raise ValueError('New group/frame differs beyond its exact deterministic reviewed reconstruction')
            cursor=0
            for group in old['whitening_groups']:
                if group['operator'].group_id==item['group_id']:
                    changed_rows.extend(range(formal_offset+cursor,formal_offset+cursor+group['operator'].rank))
                cursor+=group['operator'].rank
            normalized.append(old);used+=1
        else:normalized.append(new)
        formal_offset+=FrozenWhitening(old).count
    if used!=1:raise ValueError('Exactly one reviewed complete source group must change')
    proof=compare_source_frames(old_frames,normalized,{},{});proof.pop('_retained_formal_indices');proof.pop('frames')
    proof.update(whole_group_reconstructions=1,verified_operator_changes=1,
        retained_UV_ID_sigma_clock_and_nonidentity_source_inputs_byte_exact=False,
        all_correlated_whitening_operators_unchanged_no_rows_or_columns_sliced=False,
        source_ID_sigma_clock_order_unchanged_and_original_UV_archived=True,
        only_explicit_group_derived_UV_and_full_operator_reconstructed=True,
        derived_representatives_preserved=len(item['original_members']),
        verified_original_native_support_count=len(read(item['support_audit'])['all_readouts']),
        excluded_native_query_indices=read(item['support_audit'])['quarantined_query_indices'],
        only_changed_formal_indices=changed_rows,raw_whitening_matrix_not_sliced=True,
        all_other_source_inputs_UV_ID_sigma_W_and_clock_exact=True)
    return proof


def install_group_rebuilt_bundle(path):
    global _CONFIG
    if _CONFIG is not None:raise ValueError('Group rebuild constructor already installed')
    ref,manifest,item,before,after=read_transition(path)
    from .first_shot_revised_bundle import install_revised_source_bundle
    install_revised_source_bundle(manifest['authenticated_v21_transition']['path'])
    import native_transverse_bundle as runtime
    _CONFIG=dict(reference=ref,manifest=manifest,item=item,before=before,after=after,original=runtime.FirstShotSourceBundle)
    runtime.FirstShotSourceBundle=GroupRebuiltSourceBundle
    return dict(producer=reference(__file__),transition_manifest=ref,group_revision=manifest['group_revision'])


class GroupRebuiltSourceBundle:
    def __new__(cls,binding_path,policy,**kwargs):
        cfg=_CONFIG;manifest=cfg['manifest'];item=cfg['item'];item_ref=manifest['group_revision']
        if reference(binding_path)!=manifest['revised_binding'] or policy!=cfg['after']:
            raise ValueError('Whole-group constructor received a foreign revised source')
        bundle=cfg['original'](manifest['base_binding']['path'],cfg['before'],**kwargs)
        try:
            if bundle._geometry_pool is not None or bundle._source_pool is not None:
                raise ValueError('Source reconstruction must precede all workers/evaluation')
            from .first_shot_revised_bundle import _CONFIG as v21
            comparison=compare_source_frames(bundle._authenticated_previous_frames,bundle.frames,
                v21['changes'],{},v21['quarantines'])
            del bundle._authenticated_previous_frames
            old_frames=bundle.frames;old_scales=bundle.scales.copy();old_mask=bundle.quadratic.copy()
            frames=[apply_group(f,item,item_ref) if f['native_pts']==item['native_pts'] else f for f in old_frames]
            from .first_shot_fit import FrozenWhitening,_target_key
            from .first_shot_physical_state import FirstShotPhysicalState
            state=FirstShotPhysicalState.from_binding(binding_path)
            for name in ('initial_q','parent_q','parent_shared_slots','pts','world_parameter_indices','global_shared_indices'):
                if not np.array_equal(getattr(bundle.state,name),getattr(state,name)):
                    raise ValueError('Group source correction changed physical chart '+name)
            descriptors=OrderedDict();updated=[]
            for frame in frames:
                rows=[]
                for old in frame['rows']:
                    row=dict(old);target=deepcopy(row['target'])
                    if target.get('kind')=='board_edge' and 'association_policy'in target:target['association_policy']=deepcopy(policy)
                    row['target']=target;row['_target_key']=_target_key(target);rows.append(row)
                    descriptors.setdefault(row['_target_key'],target)
                updated.append(dict(frame,rows=rows))
            # Board policy rebinding is already normalized by the shared exact
            # comparator. Compare again after the final current descriptors.
            scope=compare_group_transition(old_frames,updated,item,item_ref)
            bundle.state=state;bundle.policy=deepcopy(policy);bundle.frames=updated
            bundle.descriptors=descriptors;bundle.keys=tuple(descriptors);bundle.target_descriptors=tuple(descriptors.values())
            bundle.initial_target_proof=state.validate_initial_targets(bundle.target_descriptors)
            bundle.whitening=[FrozenWhitening(f) for f in updated]
            bundle.scales=np.concatenate([w.scales for w in bundle.whitening]);bundle.quadratic=np.zeros(len(bundle.scales),bool)
            changed=np.asarray(scope['only_changed_formal_indices'],int);keep=np.ones(len(old_scales),bool);keep[changed]=False
            if not np.array_equal(old_scales[keep],bundle.scales[keep]) or not np.array_equal(old_mask,bundle.quadratic):
                raise ValueError('Group correction changed unrelated source scales/masks')
            bundle.ledgers=[]
            for frame,w in zip(updated,bundle.whitening,strict=True):
                h=hashlib.sha256()
                for row in frame['rows']:
                    h.update(json.dumps(row['source_point_ids'],separators=(',',':')).encode());h.update(row['uv'].tobytes());h.update(row['sigma'].tobytes())
                for group in frame['whitening_groups']:h.update(group['operator'].matrix.tobytes())
                bundle.ledgers.append(dict(family='accepted_official_static_source',programme_index=frame['index'],
                    native_pts=frame['native_pts'],time_base='1/12800',source_points=frame['source_point_count'],
                    formal_rows=w.count,source_inventory_sha256=h.hexdigest(),source_packet=frame['source_packet'],association_policy=deepcopy(policy)))
            from . import finite_source_line_coverage as coverage
            original_compile=coverage._compile_one
            def compile_one(frame,row,feature,native_ref):
                if frame['native_pts']!=item['native_pts'] or 'ice.'+row['target']['curve_id']!=item['feature_id']:
                    return original_compile(frame,row,feature,native_ref)
                if native_ref!=item['native_readout_packet']:raise ValueError('Rebuilt coverage has another native packet')
                rebuilt,line,*_=derive(item);feature=deepcopy(feature);feature['source_line_fit']=line
                value=original_compile(frame,row,feature,native_ref)
                evidence=dict(value.evidence,reviewed_support_group_revision=item_ref,
                    historical_native_readout_count=len(read(item['support_audit'])['all_readouts']),
                    current_verified_native_support_count=len(line['direct_readouts']),
                    excluded_native_query_indices=read(item['support_audit'])['quarantined_query_indices'])
                return replace(value,evidence=evidence)
            coverage._compile_one=compile_one
            try:
                bundle.finite_coverage_inventory,bundle.finite_coverage_revision_proof=coverage.read_coverage_revision(
                    state.binding['finite_source_line_coverage_revision'],state,updated)
            finally:coverage._compile_one=original_compile
            bundle._world_cache.clear();bundle._finite_coverage_cache.clear()
            bundle._authenticated_group_previous_frames=old_frames
            bundle._authenticated_group_previous_scales=old_scales;bundle._authenticated_group_previous_mask=old_mask
            bundle.group_rebuild_receipt=dict(producer=reference(__file__),transition_manifest=cfg['reference'],
                group_revision=item_ref,source_scope=scope,old_noise_and_native_authentication_unmodified=True,
                support_audit=item['support_audit'],original_UV_ID_sigma_and_full_group_archived=True,
                original_source_files_modified=False,new_basis_full_r_J_required=True)
            return bundle
        except BaseException as error:
            try:
                actual=next(f for f in bundle.frames if f['native_pts']==item['native_pts'])
                _save_failure_frame(actual,item,dict(stage='source_group_constructor',type=type(error).__name__,message=str(error)))
            except Exception:pass
            bundle.close();raise


def attach_group_pipeline(owner):
    receipt=owner.bundle.group_rebuild_receipt
    refs=(reference(__file__),receipt['transition_manifest'],receipt['group_revision'],receipt['support_audit'])
    for ref in refs:
        if ref not in owner.source_producers:owner.source_producers.append(deepcopy(ref))
    previous=owner.checkpoint_identity
    def identity(adapter):
        result=previous();result['reviewed_post_native_whole_group_reconstruction']=deepcopy(receipt);return result
    owner.checkpoint_identity=MethodType(identity,owner)
    return receipt
