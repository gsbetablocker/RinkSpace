"""Same-generation, explicitly scoped comparison of native replay arrays.

This consumes production CP / independent-RS / finite-material reprofile
outputs and their SHA ledgers. It does not silently recompute a different
camera, source UV, correspondence, or denominator on the CPU.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .joint_metrics import PacketArrays,genuine_query_operators,summarize
from .io_utils import read_json,file_sha256


def _checked_array(root,name,dtype,shape,ledger):
    if name not in ledger:raise ValueError('native output missing SHA binding: '+name)
    path=Path(root)/name
    if file_sha256(path)!=ledger[name]['sha256']:raise ValueError('native output SHA changed: '+str(path))
    value=np.fromfile(path,dtype=dtype)
    if value.size!=int(np.prod(shape)):raise ValueError('native output shape changed: '+str(path))
    return value.reshape(shape)


def _load_modes(spec,n,packet_sha,packet_file,packet,*,legacy_unbound_diagnostic=False,verified_identity=None):
    state=Path(spec['state']);report=read_json(state/'value_report.json');ledger=report['files'];result={}
    from .report_identity import require_replay_binding
    report['source_objective_identity']=verified_identity if verified_identity is not None else require_replay_binding(packet_file,packet,report,state,allow_legacy_diagnostic=legacy_unbound_diagnostic)
    for name,prefix in (('fixed_station_independent_RS','raw_station_RS_'),('legal_material_reprofile','')):
        result[name]=dict(predicted=_checked_array(state,prefix+'predicted_uv.f64','<f8',(n,2),ledger),
            error=_checked_array(state,prefix+'error_native_uv.f64','<f8',(n,2),ledger),
            valid=_checked_array(state,prefix+'valid.u8','u1',(n,),ledger).astype(bool),
            semantics=report.get('raw_projection_semantics') if prefix else report.get('profiled_native_export_semantics'))
    explicit=spec.get('explicit')
    if explicit:
        eroot=Path(explicit);meta=read_json(eroot/'manifest.json')
        if meta['packet_sha256']!=packet_sha or meta['raw_rows']!=n:raise ValueError('explicit CP output belongs to another exact source generation')
        files=meta['files'];mode=dict(predicted=_checked_array(eroot,'explicit_predicted_uv.f64','<f8',(n,2),files),
            error=_checked_array(eroot,'explicit_error_native_uv.f64','<f8',(n,2),files),
            valid=_checked_array(eroot,'explicit_valid.u8','u1',(n,),files).astype(bool),
            RS=_checked_array(eroot,'explicit_RS_equality.f64','<f8',(n,),files),
            Query=_checked_array(eroot,'explicit_query_equality.f64','<f8',(n,2),files),
            semantics=meta['valid_semantics'],state_token=meta['state_token'],failure_masks={})
        for kind in ('arithmetic','ray_domain','station_domain','visibility','width_domain'):
            mode['failure_masks'][kind]=_checked_array(eroot,'explicit_'+kind+'_valid.u8','u1',(n,),files).astype(bool)
        result['CP_explicit']=mode
    else:result['CP_explicit']=dict(unavailable='complete explicit-auxiliary prediction export not supplied; independent RS is not substituted')
    return result,report



INFINITY_JSON = '+Infinity'


def _same_denominator_statistics(distance,ids,valid,E,mode_name,failure_masks=None):
    """Every active ID stays in N; undefined native errors sort as +infinity.

    CP arithmetic-finite errors remain defined even when physical domains fail.
    Independent/profiling modes require their native-valid flag; a failed root
    cannot acquire a finite score from a leftover arithmetic buffer.
    """
    ids=np.asarray(ids,dtype=int);failure_masks=failure_masks or {};n=len(ids)
    finite=np.isfinite(distance[ids]);mode_valid=np.asarray(valid[ids],bool)
    if mode_name=='CP_explicit':
        arithmetic_mask=failure_masks.get('arithmetic')
        arithmetic=finite if arithmetic_mask is None else np.asarray(arithmetic_mask,bool)[ids]
        defined=finite&arithmetic
        policy='CP arithmetic-finite original operator errors retained, including physically invalid domains; undefined arithmetic errors sort as +Infinity'
    else:
        arithmetic=finite;defined=finite&mode_valid
        policy='native RS/Query/projection failure or undefined error is extended-real +Infinity; population N never shrinks'
    ordered=np.sort(np.where(defined,np.asarray(distance[ids],float),np.inf))
    def number(x):return float(x) if np.isfinite(x) else INFINITY_JSON
    def quantile(p):return None if not n else number(ordered[max(0,int(np.ceil(p*n))-1)])
    undefined=int((~defined).sum())
    if not n:rms=None;rms_status='empty_population'
    elif undefined:rms=INFINITY_JSON;rms_status='not_finite_undefined_observations'
    else:
        scale=float(ordered[-1]);rms=0. if scale==0 else scale*float(np.sqrt(np.mean((ordered/scale)**2)));rms_status='finite'
    return dict(N=n,active_rows=n,defined_error_rows=int(defined.sum()),undefined_error_rows=undefined,
        mode_invalid_rows=int((~mode_valid).sum()),arithmetic_finite_but_mode_invalid_rows=int((finite&arithmetic&~mode_valid).sum()),
        rms_px=rms,rms_status=rms_status,median_px=quantile(.5),p95_px=quantile(.95),max_px=quantile(1.),
        quantile_definition='nearest rank: sorted ceil(p*N)-1; undefined observations occupy +Infinity, never zero or a finite cap',
        extended_real_undefined_value=INFINITY_JSON,error_semantics=policy,
        domain_failure_counts={key:int(np.sum(~np.asarray(mask,bool)[ids])) for key,mask in failure_masks.items()},
        rows_above_original_E=int(np.sum(np.where(defined,distance[ids],np.inf)>E[ids])),
        all_operator_errors_within_original_E=bool(n and defined.all() and np.all(distance[ids]<=E[ids])),
        all_mode_valid=bool(n and mode_valid.all()),formal_or_physical_acceptance_claimed=False)


def _population_summary(distance,ids,valid,E,mode_name,failure_masks=None):
    # Preserve the old field names for readers while explicitly identifying
    # their conditional denominator. New comparisons use the full-N object.
    conditional=summarize(distance,ids,valid,E)
    return dict(conditional,legacy_fields_are_conditional_valid_only=True,
        conditional_valid_statistics=conditional,
        conditional_valid_statistics_semantics='Legacy RMS/quantiles/max condition on this state/mode valid finite subset; valid_rows is their denominator',
        same_denominator_statistics=_same_denominator_statistics(distance,ids,valid,E,mode_name,failure_masks))


def _finite_json(value):
    a=np.asarray(value)
    if a.ndim:return [_finite_json(v) for v in a]
    return float(a) if np.isfinite(a) else None


def _source_evidence_record(i,mode_name,data,distance,valid,E,sourceids,datasets,sid,targets,pts,uv,pred):
    masks=data.get('failure_masks',{});finite=bool(np.isfinite(distance[i]))
    arithmetic_mask=masks.get('arithmetic')
    arithmetic=finite and (mode_name!='CP_explicit' or arithmetic_mask is None or bool(arithmetic_mask[i]))
    defined=arithmetic and (mode_name=='CP_explicit' or bool(valid[i]))
    reasons=[key for key,mask in masks.items() if not bool(mask[i])]
    if not bool(valid[i]):reasons.append('mode_invalid')
    if not finite:reasons.append('undefined_operator_error')
    record=dict(source_dataset_id=str(datasets[i]),source_row_id=str(sourceids[i]),row_index=int(i),shot_id=str(sid[i]),
        material=str(targets[i]),native_pts=int(pts[i]),source_seconds=float(pts[i]/12800),source_uv=_finite_json(uv[i]),
        predicted_uv=_finite_json(pred[i]),available_arithmetic_operator_error_px=float(distance[i]) if arithmetic else None,
        same_denominator_error_px=float(distance[i]) if defined else INFINITY_JSON,mode_valid=bool(valid[i]),
        arithmetic_finite=arithmetic,failure_flags=reasons,E_px=_finite_json(E[i]))
    if 'RS' in data:record['CP_RS_equality_px']=_finite_json(data['RS'][i]);record['CP_Query_equality_px']=_finite_json(data['Query'][i])
    return record


def _source_failure_evidence(out,filename,mode_name,data,ids,distance,valid,E,sourceids,datasets,sid,targets,pts,uv,pred,shots):
    masks=data.get('failure_masks',{});finite=np.isfinite(distance)
    arithmetic=finite if mode_name!='CP_explicit' else finite&np.asarray(masks.get('arithmetic',finite),bool)
    failed=ids[~valid[ids]|~arithmetic[ids]]
    def record(i):return _source_evidence_record(int(i),mode_name,data,distance,valid,E,sourceids,datasets,sid,targets,pts,uv,pred)
    arithmetic_tails=[];invalid_arithmetic_tails=[];representatives=[]
    for shot in shots:
        rows=ids[(sid[ids]==shot)&arithmetic[ids]]
        arithmetic_tails.extend(record(i) for i in rows[np.argsort(distance[rows])[-3:][::-1]])
        bad=rows[~valid[rows]];invalid_arithmetic_tails.extend(record(i) for i in bad[np.argsort(distance[bad])[-3:][::-1]])
        rr=failed[sid[failed]==shot];ordered=rr[np.lexsort((rr,pts[rr]))];seen=set()
        for i in ordered:
            if int(pts[i]) in seen:continue
            representatives.append(record(i));seen.add(int(pts[i]))
            if len(seen)==3:break
    path=Path(out)/filename
    with gzip.open(path,'wt',encoding='utf8',compresslevel=1) as stream:
        for i in failed:stream.write(json.dumps(record(i),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    return dict(worst_arithmetic_finite_source_rows=arithmetic_tails,
        worst_invalid_arithmetic_finite_source_rows=invalid_arithmetic_tails,
        source_failure_evidence=dict(rows=len(failed),representative_rows=representatives,
            representative_selection='first three failed native exposures per shot; no finite error ranking is invented for failed roots',
            complete_ledger=dict(file=str(path.resolve()),sha256=file_sha256(path),size_bytes=path.stat().st_size,
                format='jsonl.gz',all_failed_rows_retained=True,undefined_error_value=INFINITY_JSON)))


def _numbers(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return dict(n=len(x),rms=float(np.sqrt(np.mean(x*x))) if len(x) else None,median=float(np.median(x)) if len(x) else None,
        P95=float(np.quantile(x,.95)) if len(x) else None,max=float(np.max(x)) if len(x) else None)


def _coverage(exposures,pts,rows,valid):
    have=np.unique(pts[rows]);good=np.unique(pts[rows[valid[rows]]]);mask=np.isin(exposures,have);gaps=[];begin=None
    for i,value in enumerate(np.r_[mask,True]):
        if not value and begin is None:begin=i
        elif value and begin is not None:gaps.append((begin,i));begin=None
    worst=max(gaps,key=lambda ab:ab[1]-ab[0]) if gaps else None
    return dict(native_exposures=len(exposures),source_visible_exposures=len(have),valid_source_exposures=len(good),
        unsupported_exposures=int((~mask).sum()),coverage_fraction=float(mask.mean()),
        longest_source_gap_exposures=0 if worst is None else worst[1]-worst[0],
        longest_source_gap_seconds=0. if worst is None else (worst[1]-worst[0])*.02,
        longest_source_gap_PTS=None if worst is None else [int(exposures[worst[0]]),int(exposures[worst[1]-1])])


def _temporal(pred,uv,valid,active,kind,sid,mat,model,pts,queries,sourceids):
    # Only named physical-point material paths establish a 2D stationary
    # identity. Free station and Query correspondences cannot become tracks.
    physical=np.asarray([bool(m['paths']) and all(p.get('physical_point',False) for p in m['paths']) for m in model['materials']])
    groups=defaultdict(list)
    for i in np.flatnonzero(active&(kind==1)&physical[mat]&~queries):groups[(str(sid[i]),int(mat[i]))].append(int(i))
    delta=defaultdict(list);second=defaultdict(list);worst=[];ambiguous=0
    for (shot,material),rows in groups.items():
        bypts=defaultdict(list)
        for i in rows:bypts[int(pts[i])].append(i)
        # Do not choose among competing readings by model error.
        track={pt:items[0] for pt,items in bypts.items() if len(items)==1}
        ambiguous+=sum(len(items) for items in bypts.values() if len(items)>1)
        for pt,i in sorted(track.items()):
            if pt-256 not in track:continue
            j=track[pt-256]
            if valid[i] and valid[j] and np.isfinite(pred[[j,i]]).all():
                de=(pred[i]-pred[j])-(uv[i]-uv[j]);delta[shot].append(float(np.linalg.norm(de)))
            if pt-512 not in track:continue
            k=track[pt-512]
            if valid[i] and valid[j] and valid[k] and np.isfinite(pred[[k,j,i]]).all():
                d2=(pred[i]-2*pred[j]+pred[k])-(uv[i]-2*uv[j]+uv[k]);value=float(np.linalg.norm(d2));second[shot].append(value)
                record=dict(shot_id=shot,material=model['materials'][material]['label'],native_pts=[pt-512,pt-256,pt],
                    source_row_ids=[sourceids[x] for x in (k,j,i)],source_uv=uv[[k,j,i]].tolist(),predicted_uv=pred[[k,j,i]].tolist(),
                    source_second_difference_px=(uv[i]-2*uv[j]+uv[k]).tolist(),model_second_difference_px=(pred[i]-2*pred[j]+pred[k]).tolist(),
                    error_second_difference_norm_px=value)
                worst.append(record)
    worst.sort(key=lambda r:r['error_second_difference_norm_px'],reverse=True)
    keep=[]
    for shot in model['shots']:
        keep.extend([r for r in worst if r['shot_id']==shot][:3])
    return dict(semantics='Delta error and Delta2 error of the same named stationary physical point, consecutive native PTS, never across cuts',
        by_shot={s:dict(first_difference_error_px=_numbers(delta[s]),second_difference_error_px=_numbers(second[s])) for s in model['shots']},
        ambiguous_same_entity_same_exposure_readings_not_chosen=ambiguous,worst_native_triplets=keep,
        free_station_or_Query_tracks_fabricated=False)


def _comparison_scope(packet,model,pts,shot_index):
    """Use the packet's actual shot registry; a partial run is never a full programme."""
    shot_ids=packet['shot_ids']
    if not shot_ids or len(set(shot_ids))!=len(shot_ids) or set(model['shots'])!=set(shot_ids):
        raise ValueError('Comparison model shot set differs from the exact source packet scope')
    intervals=[];exposures=[];last=None
    for index,shot_id in enumerate(shot_ids):
        scope=model['shots'][shot_id]['scope'];first=int(scope['first_pts']);end=int(scope['last_pts'])
        if first!=scope['first_pts'] or end!=scope['last_pts'] or first>end or first%256 or end%256:
            raise ValueError('Comparison shot scope is not on the original native PTS lattice')
        if first<256256 or end>2415104 or (last is not None and first<=last):
            raise ValueError('Comparison shot scopes overlap or leave the official programme range')
        source=np.asarray(pts)[np.asarray(shot_index)==index]
        if np.any((source<first)|(source>end)|((source-first)%256!=0)):
            raise ValueError('Source exposure lies outside its declared comparison shot scope')
        interval=dict(shot_id=shot_id,first_pts=first,last_pts=end,native_pts_step=256,
            native_exposures=(end-first)//256+1)
        intervals.append(interval);exposures.extend(range(first,end+1,256));last=end
    full=exposures==list(range(256256,2415105,256))
    return dict(schema='rinkfit.same_native_source_shot_scope.v1',shot_ids=shot_ids,
        shots=intervals,native_exposures=len(exposures),full_programme_scope=full,
        coverage_scope='complete_official_programme' if full else 'declared_shots_only',
        native_time_base='1/12800',native_pts_step=256,
        official_programme_first_pts=256256,official_programme_last_pts=2415104)


def compare(packet_file,state_specs,output,*,legacy_unbound_diagnostic=False):
    out=Path(output)
    if out.exists():raise ValueError('immutable recovery comparison output exists')
    arrays=PacketArrays(packet_file);packet=arrays.packet;n=int(packet['row_count']);packet_sha=file_sha256(packet_file)
    uv=arrays.load('uv');pts=arrays.load('native_pts');mat=arrays.load('material_index');kind=arrays.load('kind');si=arrays.load('shot_index');ad=arrays.load('admission')==1;role=arrays.load('fit_role');active=ad&(role==1);E=arrays.load('E').reshape(n);ids=np.flatnonzero(active)
    queries,normals=genuine_query_operators(packet,n);sourceids=packet['source_row_ids'];sid=np.asarray(packet['shot_ids'])[si]
    datasets=packet.get('source_dataset_ids',['joint']*n);keys=list(zip(datasets,sid.tolist(),pts.tolist(),sourceids))
    if len(set(keys))!=len(keys):raise ValueError('compound dataset/shot/native-PTS/source row keys are not unique')
    specs=read_json(state_specs) if isinstance(state_specs,(str,Path)) else state_specs
    if isinstance(specs,dict):specs=specs['states']
    if not specs or len({s['label'] for s in specs})!=len(specs):raise ValueError('unique comparison state labels required')
    identity=hashlib.sha256();identity.update(json.dumps(sourceids,separators=(',',':')).encode());identity.update(np.asarray(uv,dtype='<f8').tobytes());identity.update(np.asarray(kind,dtype='u1').tobytes());identity.update(normals.astype('<f8').tobytes());identity.update(np.asarray(mat,dtype='<u4').tobytes());identity.update(np.asarray(pts,dtype='<i8').tobytes());identity.update(np.asarray(si,dtype='<u2').tobytes());identity.update(json.dumps(datasets,separators=(',',':')).encode())
    report=dict(schema='rinkfit.recovery_same_source_comparison.v2',packet=dict(path=str(Path(packet_file).resolve()),sha256=packet_sha),
        exact_source_ID_UV_operator_identity_sha256=identity.hexdigest(),raw_rows=n,active_rows=len(ids),inactive_rows=n-len(ids),
        common_denominator_rows=len(ids),original_UV_changed=False,invalid_rows_removed=False,states={},
        formal_statistics_or_metric_accuracy_claimed=False,source_unknown_and_statistics_unknown_separate=True,
        statistics_contract=dict(schema='rinkfit.full_active_denominator_extended_real.v1',legacy_statistics='conditional valid finite subset; retained for compatibility',
            authoritative_population_statistics='same_denominator_statistics retains every active source ID in N',
            CP_policy='all arithmetic-finite errors included even for invalid domains; arithmetic-undefined observations prevent a finite RMS',
            native_RS_policy='failed or undefined projections are +Infinity, never zero, capped or removed',
            quantiles='full-N nearest rank including +Infinity',failure_evidence='complete failed-source JSONL ledgers plus invalid-domain finite-pixel tails',
            temporal_statistics='existing temporal difference statistics remain conditional on valid same-entity consecutive samples; failure sources are separately retained'))
    reference_material_labels=None
    from .report_identity import require_replay_binding,mark_unbound_diagnostic
    # Reject every mismatched input before writing even the first state's ledger.
    verified_identities={};shared_scope=None
    for spec in specs:
        state=Path(spec['state']);value=read_json(state/'value_report.json')
        verified_identities[spec['label']]=require_replay_binding(packet_file,packet,value,state,load=arrays.load,allow_legacy_diagnostic=legacy_unbound_diagnostic)
        model_path=Path(spec.get('model',state/'model.json'))
        if file_sha256(model_path)!=value['model_sha256']:raise ValueError('model and replay SHA differ: '+spec['label'])
        scope=_comparison_scope(packet,read_json(model_path),pts,si)
        if shared_scope is None:shared_scope=scope
        elif shared_scope!=scope:raise ValueError('Candidate comparison native shot scope differs from the baseline')
    report['scope']=shared_scope;report['native_exposures']=shared_scope['native_exposures']
    report['full_programme_scope']=shared_scope['full_programme_scope']
    out.mkdir(parents=True)
    for state_index,spec in enumerate(specs):
        state=Path(spec['state']);model_path=Path(spec.get('model',state/'model.json'));model=read_json(model_path);modes,value=_load_modes(spec,n,packet_sha,packet_file,packet,legacy_unbound_diagnostic=legacy_unbound_diagnostic,verified_identity=verified_identities[spec['label']])
        if file_sha256(model_path)!=value['model_sha256']:raise ValueError('model and replay SHA differ: '+spec['label'])
        labels=np.asarray([m['label'] for m in model['materials']]);targets=labels[mat]
        if reference_material_labels is None:reference_material_labels=targets
        elif not np.array_equal(targets,reference_material_labels):raise ValueError('same numeric material indices changed physical identity')
        categories=np.asarray(['ice' if t.startswith('ice.') else 'board' if t.startswith('board.') else 'advertisement' if t.startswith('Ad') or '.print.' in t or t.startswith('print_component.') else 'other' for t in targets])
        entry=dict(model=dict(path=str(model_path.resolve()),sha256=file_sha256(model_path)),state_directory=str(state.resolve()),native_exposures=shared_scope['native_exposures'],scope=shared_scope,modes={},source_objective_identity=value['source_objective_identity'])
        for mode_name,data in modes.items():
            if 'unavailable' in data:entry['modes'][mode_name]=data;continue
            pred=data['predicted'];error=data['error'];valid=data['valid'];finite=np.isfinite(error).all(1)
            if np.max(np.abs(pred[active&finite]-uv[active&finite]-error[active&finite]),initial=0)>1e-8:raise ValueError('native source residual definition changed')
            distance=np.hypot(error[:,0],error[:,1]);distance[queries]=np.abs(np.sum(error[queries]*normals[queries],axis=1))
            population=lambda rows:_population_summary(distance,rows,valid,E,mode_name,data.get('failure_masks'))
            result=dict(semantics=data.get('semantics'),overall=population(ids),by_shot={},by_material={},by_source_second={},
                invalid_reason_counts=value.get('projection_classification_counts'),coverage={})
            for shot in model['shots']:
                rr=ids[sid[ids]==shot];scope=model['shots'][shot]['scope'];ee=np.arange(scope['first_pts'],scope['last_pts']+1,256,dtype=np.int64)
                result['by_shot'][shot]=population(rr);result['coverage'][shot]={}
                for family in ('ice','board','advertisement','other'):
                    rows=rr[categories[rr]==family];summary=population(rows);summary.update(_coverage(ee,pts,rows,valid));summary['raw_unqualified_or_inactive_rows']=int(np.sum((sid==shot)&(categories==family)&~active));result['coverage'][shot][family]=summary
                for sec in np.unique(pts[rr]//12800):
                    rows=rr[pts[rr]//12800==sec];result['by_source_second'][shot+'/'+str(int(sec))]=summarize(distance,rows,valid,E)
            for label in sorted(set(targets[ids])):result['by_material'][str(label)]=population(ids[targets[ids]==label])
            for special in ('s01/Ad2.entity','s01/ice'):
                rr=ids[(sid[ids]=='s01')&((kind[ids]==1)&np.char.startswith(targets[ids].astype(str),'Ad2.') if special.endswith('entity') else categories[ids]=='ice')]
                result[special]=summarize(distance,rr,valid,E)
            available=ids[valid[ids]&np.isfinite(distance[ids])];tails=[]
            for shot in model['shots']:
                rr=available[sid[available]==shot];worst=rr[np.argsort(distance[rr])[-3:][::-1]]
                tails.extend(dict(source_row_id=sourceids[int(i)],row_index=int(i),shot_id=str(sid[i]),material=str(targets[i]),native_pts=int(pts[i]),source_seconds=float(pts[i]/12800),source_uv=uv[i].tolist(),predicted_uv=pred[i].tolist(),error_px=float(distance[i]),E_px=float(E[i])) for i in worst)
            result['worst_native_source_rows']=tails
            result['worst_native_source_rows_semantics']='legacy valid finite subset only; arithmetic-finite invalid-domain tails and all failed sources are retained separately'
            result.update(_source_failure_evidence(out,f'state_{state_index:02d}_{mode_name}_failed_sources.jsonl.gz',mode_name,data,ids,distance,valid,E,sourceids,datasets,sid,targets,pts,uv,pred,model['shots']))
            result['source_motion_error']=_temporal(pred,uv,valid,active,kind,sid,mat,model,pts,queries,sourceids)
            if 'RS' in data:
                result['CP_equalities']=dict(RS_absolute_px=_numbers(np.abs(data['RS'][ids])),Query_norm_px=_numbers(np.linalg.norm(data['Query'][ids[queries[ids]]],axis=1)),
                    query_unavailable_rows=int(np.sum(~np.isfinite(data['Query'][ids[queries[ids]]]).all(1))),
                    domain_failure_counts={key:int(np.sum(active&~mask)) for key,mask in data['failure_masks'].items()})
            entry['modes'][mode_name]=result
        report['states'][spec['label']]=entry
    if any(not x['source_objective_identity']['original_target_verified'] for x in report['states'].values()):
        report=mark_unbound_diagnostic(report)
        report['original_target_verified']=False
    else:report['original_target_verified']=True
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(out/'comparison.json'),states=list(report['states']),same_denominator=len(ids),native_exposures=shared_scope['native_exposures'])))
    return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--states',required=True);p.add_argument('--out',required=True);p.add_argument('--legacy-unbound-diagnostic',action='store_true');a=p.parse_args(argv);compare(a.packet,a.states,a.out,legacy_unbound_diagnostic=a.legacy_unbound_diagnostic)

if __name__=='__main__':main()
