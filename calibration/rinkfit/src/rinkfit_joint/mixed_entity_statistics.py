"""Whole-native RGB conditional covariance for scalar profiles and 2-D entities.

Each encoded RGB draw is shared by every readout in an exposure. Entity source
operators keep their two-dimensional output and original validity domain. A
failed synthetic source is recorded as unknown noise; trials are never censored.
"""
from pathlib import Path
from collections import Counter, defaultdict
import copy, hashlib, json, time
import numpy as np
from .io_utils import read_jsonl, write_json, file_sha256, object_sha256
from .paint_statistics import _prepare, _load_frozen
from .paint_pair_statistics import _measure_pair
from .source_statistics import supported_factor_whitening, StatisticsUnavailable
from .source_statistics_storage import encode_rng_state
from .ptx_source_filter import PTXSourceFilterCUDA
from .source_worker_memory import snapshot as memory_snapshot


def component_layout(rows):
    row_index=[];directions=[];slices=[]
    for i,row in enumerate(rows):
        lo=len(row_index)
        if row['operator_kind']=='physical_corner_2d':
            row_index.extend((i,i));directions.extend(((1.,0.),(0.,1.)))
        else:
            n=np.asarray(row['source_normal'],float)
            if n.shape!=(2,) or abs(np.linalg.norm(n)-1)>1e-9:
                raise ValueError('Scalar source direction is not its actual unit normal')
            row_index.append(i);directions.append(n)
        slices.append(slice(lo,len(row_index)))
    return np.asarray(row_index,np.int64),np.asarray(directions,float),slices


def _raw_channels(rgb):
    a=rgb.astype(np.float32)
    return np.stack((.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2],
                     a[:,:,0]-a[:,:,1],a[:,:,2]-.5*(a[:,:,0]+a[:,:,1])))


def _entity_branch(result):
    raw=result['raw_source_operator_result']
    return json.dumps([[(p.get('inward_station_px'),p.get('profile',{}).get('channel'),
                        p.get('profile',{}).get('polarity'))
                       for p in edge.get('profile_audits',[])]
                      for edge in raw.get('edge_fits',[])],separators=(',',':'))


def _row_hash(row,current):
    if current:
        return object_sha256(dict(native_pts=row['native_pts'],current_uv=row['current_uv'],
            source_row_id=row['source_row_id'],measurement_operator=row['measurement_operator_resolved']))
    return object_sha256(dict(native_pts=row['native_pts'],source_uv=row['source_uv'],
        source_row_id=row['source_row_id']))


def _load_scalar_cache(path,rows,native,operator,trials,seed):
    if path is None:return {},None,{}
    path=Path(path);record=json.loads(path.read_text(encoding='utf8'))
    if record['seed']!=seed or record['trials']!=trials or record['actual_mean_replay_max_px']!=0:
        raise ValueError('Prior joint sample stream differs in trial count/seed/base mean')
    nr=record['native_receipt']
    if any(nr[k]!=native[k]for k in ('native_pts','decoded_native_BGR_sha256','decoded_native_RGB_sha256')):
        raise ValueError('Prior MC samples belong to another RGB exposure')
    if record['frozen_operator']['sha256']!=file_sha256(operator.__file__):
        raise ValueError('Prior scalar estimator bytes differ')
    artifact=record['replay_arrays']
    if file_sha256(artifact['file'])!=artifact['sha256']:
        raise ValueError('Prior full trial artifact changed')
    cached={};cached_diagnostics={}
    with np.load(artifact['file'])as z:
        if 'normal_shift_delta_px'in z:
            ids={str(rid):j for j,rid in enumerate(z['source_row_ids'])}
            for i,row in enumerate(rows):
                j=ids.get(row['source_row_id'])
                if j is None or row['operator_kind']!='continuous_cubic_point':continue
                rid=row['source_row_id']
                if record['source_measurements_sha256'][rid]!=_row_hash(row,False) or record['current_measurements_sha256'][rid]!=_row_hash(row,True):
                    raise ValueError('Cached scalar source hashes differ')
                if not np.array_equal(z['actual_mean_uv'][j],row['current_uv'])or not np.array_equal(z['profile_normals'][j],row['source_normal']):
                    raise ValueError('Cached scalar output mean or direction differs')
                cached[i]=z['normal_shift_delta_px'][:,j].copy()
                cached_diagnostics[i]=dict(boundary=int(z['synthetic_window_boundary_counts'][j])if 'synthetic_window_boundary_counts'in z else None,flat=int(z['synthetic_flat_peak_counts'][j])if 'synthetic_flat_peak_counts'in z else None)
    return cached,dict(parent_receipt=dict(file=str(path.resolve()),sha256=file_sha256(path)),
        reused_rows=len(cached),same_whole_RGB_draw_stream=True,full_cross_covariance_recomputed=True,
        compact_parent_without_trials_is_not_reused_as_joint_samples=True),cached_diagnostics


def _compact_components(D,C,row_index,directions,means,boundary,flat,branches):
    diagnostic=np.zeros(D.shape[1],bool)
    for j,i in enumerate(row_index):
        diagnostic[j]=bool(boundary[i] or flat[i] or branches[i] or C[j,j]<=0)
    saved=dict(schema=np.asarray('rinkfit.mixed_native_readout_component_covariance.v1'),
        component_row_index=row_index,component_directions=directions,
        actual_mean_uv=means,component_covariance_px2=C,MC_trial_count=np.asarray(len(D),np.int64),
        MC_component_shift_mean_px=D.mean(0),
        MC_component_shift_minmax_px=np.stack((D.min(0),D.max(0))),
        MC_component_shift_deltas_sha256=np.asarray(hashlib.sha256(np.ascontiguousarray(D,dtype='<f8').tobytes()).hexdigest()),
        synthetic_window_boundary_counts=boundary,synthetic_flat_peak_counts=flat,
        source_discrete_branch_switch_counts=branches,diagnostic_nonregular_component_indices=np.flatnonzero(diagnostic))
    if diagnostic.any():saved['diagnostic_nonregular_component_shift_delta_px']=D[:,diagnostic]
    return saved


def frame(rgb,native,rows,operator,backend,out,*,trials=256,seed=20261018,scalar_cache=None):
    """Consume one RGB exposure. Does not create material/source eligibility."""
    from PIL import Image
    started=time.perf_counter();owned_before=memory_snapshot();out=Path(out);out.mkdir()
    if len({(r['shot_id'],int(r['native_pts']))for r in rows})!=1:
        raise ValueError('Mixed shot or native PTS in shared RGB task')
    if int(rows[0]['native_pts'])!=int(native['native_pts']) or len({r['source_row_id']for r in rows})!=len(rows):
        raise ValueError('Duplicate readout or wrong native exposure')
    n=len(rows);means=np.asarray([r['current_uv']for r in rows],float)
    ri,directions,slices=component_layout(rows);d=len(ri);points=[];pairs={};wide_pairs={};entities=[]
    entity_modules={};masks={};operator_refs={};mask_refs={}
    for i,row in enumerate(rows):
        kind=row['operator_kind']
        if kind=='continuous_cubic_point':points.append(i)
        elif kind=='paint_flank':pairs.setdefault(row['source_pair_id'],[]).append(i)
        elif kind=='wide_paint_flank':
            wide_pairs.setdefault(row['source_pair_id'],[]).append(i)
            source=row['wide_pair_source'];op=source['current_pair_operator'];ref=op['pure_remeasure_entry'];key=ref['file']
            if file_sha256(key)!=ref['sha256']:raise ValueError('Full wide-pair source entry changed')
            if key not in entity_modules:entity_modules[key]=_load_frozen(key)
            operator_refs[key]=dict(file=key,sha256=ref['sha256'])
            fr=source['current_native_frame_source'];mf=fr['mask_file'];mh=fr['mask_sha256']
            if mf not in masks:
                if file_sha256(mf)!=mh:raise ValueError('Current wide-pair visibility mask changed')
                masks[mf]=np.asarray(Image.open(mf)).astype(bool)
            mask_refs[mf]=dict(file=mf,sha256=mh)
            if fr['RGB_sha256']!=native['decoded_native_RGB_sha256']:raise ValueError('Full wide pair uses another native RGB exposure')
        elif kind=='physical_corner_2d':
            entities.append(i);corner=row['current_corner_operator'];ref=corner['pure_remeasure_entry'];key=ref['file']
            if file_sha256(key)!=ref['sha256']:raise ValueError('Current entity source entry changed')
            if key not in entity_modules:entity_modules[key]=_load_frozen(key)
            operator_refs[key]=dict(file=key,sha256=ref['sha256'])
            mf=corner['native_mask_file'];mh=corner['native_mask_sha256']
            if mf not in masks:
                if file_sha256(mf)!=mh:raise ValueError('Current source visibility mask changed')
                masks[mf]=np.asarray(Image.open(mf)).astype(bool)
            mask_refs[mf]=dict(file=mf,sha256=mh)
            if not corner['all_support_pixels_unmasked']:raise ValueError('Entity source support crosses mask')
            fr=row['current_native_frame_source']
            if fr['RGB_sha256']!=native['decoded_native_RGB_sha256']:
                raise ValueError('Entity source mean comes from another actual RGB exposure')
        else:raise ValueError('Unknown actual source estimator '+str(kind))
    for key,ix in wide_pairs.items():
        if [rows[i]['flank_side_index']for i in ix]!=[0,1]:raise ValueError('Wide-pair 4-D source order changed')
    prepared=[];pairkeys=list(pairs)
    for key in pairkeys:
        ix=pairs[key]
        if [rows[i]['flank_side_index']for i in ix]!=[0,1]:raise ValueError('Paint 4-D source column order changed')
        prepared+=_prepare([dict(source_measurement_audit=dict(source_material_receipt=rows[ix[0]]['source_material_receipt']))])
    shadow=[];filter_seconds=0.;profile_seconds=0.
    channels=sorted({int(rows[i]['frozen_point_operator']['source_measurement_audit']['profile']['channel'])for i in points}|{p['ch']for p in prepared})
    def prepare(image,check=False):
        nonlocal filter_seconds
        start=time.perf_counter();raw=_raw_channels(image);g,c=backend(raw,include_gaussian=True)
        coeff={ch:c[ch]for ch in channels}
        if check:
            errors={}
            if channels:
                reference=operator.prepare_cubic_channels(image)
                errors['cubic']=max(float(abs(coeff[ch]-reference[ch]).max())for ch in channels)
            if entities:
                m=entity_modules[rows[entities[0]]['current_corner_operator']['pure_remeasure_entry']['file']]
                reference=m.prepare_corner_channels(image);errors['entity_gaussian']=float(abs(g-reference).max())
            if wide_pairs:
                first=rows[next(iter(wide_pairs.values()))[0]]['wide_pair_source'];m=entity_modules[first['current_pair_operator']['pure_remeasure_entry']['file']]
                reference=m.prepare_pair_channels(image);errors['wide_pair_gaussian']=float(abs(g-reference).max())
            if any(v!=0 for v in errors.values()):raise ValueError('GPU source channels differ from exact source arithmetic '+str(errors))
            shadow.append(errors)
        filter_seconds+=time.perf_counter()-start;return g,coeff
    discrete_rows=entities+[i for ix in wide_pairs.values()for i in ix]
    branch_base={};branch_counts={i:Counter()for i in discrete_rows};invalid={i:[]for i in discrete_rows}
    def measure(image,g,coeff,selected,trial=None):
        nonlocal profile_seconds
        begin=time.perf_counter();values=np.full(d,np.nan);boundary=np.zeros(n,np.int64);flat=np.zeros(n,np.int64)
        for i in points:
            if i not in selected:continue
            result=operator.replay_outer_cubic(image,rows[i]['frozen_point_operator'],coefficients=coeff)
            if not result['numeric_valid']:raise ValueError('Scalar source estimator undefined; no censored trials permitted')
            values[slices[i]]=result['normal_offset_px'];boundary[i]=result['synthetic_window_boundary_count'];flat[i]=len(result['flat_peak_intervals'])
        if prepared:
            positions,b,_=_measure_pair(image,prepared,coefficients=coeff)
            for k,key in enumerate(pairkeys):
                for side,i in enumerate(pairs[key]):values[slices[i]]=positions[k,side];boundary[i]=b[k,side]
        for i in entities:
            row=rows[i];op=row['current_corner_operator'];module=entity_modules[op['pure_remeasure_entry']['file']]
            result=module.remeasure_type1_shared_rgb(image,row,masks[op['native_mask_file']],channels=g)
            signature=_entity_branch(result);branch_counts[i][signature]+=int(trial is not None)
            if trial is None:branch_base[i]=signature
            if result['valid']:
                values[slices[i]]=result['current_uv']
            elif trial is None:raise ValueError('Current entity source mean has no valid exact base replay')
            else:invalid[i].append(dict(trial=trial,reason=result['raw_source_operator_result'].get('reason','undefined_actual_estimator')))
        for key,ix in wide_pairs.items():
            source=rows[ix[0]]['wide_pair_source'];op=source['current_pair_operator'];module=entity_modules[op['pure_remeasure_entry']['file']]
            result=module.remeasure_pair_shared_rgb(image,source,masks[source['current_native_frame_source']['mask_file']],channels=g)
            signature=json.dumps(result.get('branch_signature'),sort_keys=True,separators=(',',':'))
            for side,i in enumerate(ix):
                branch_counts[i][signature]+=int(trial is not None)
                if trial is None:branch_base[i]=signature
                if result['valid']:
                    q=np.asarray(result['actual_pair_4D_UV'][side]);origin=np.asarray(op['original_current_inner_edge_navigation_uv']);normal=np.asarray(rows[i]['source_normal'])
                    shift=float((q-origin)@normal)
                    if np.linalg.norm(q-origin-shift*normal)>1e-9:raise ValueError('Actual wide-pair readout left its frozen normal acquisition')
                    values[slices[i]]=shift
                elif trial is None:raise ValueError('Actual wide pair has no valid base source replay')
                else:invalid[i].append(dict(trial=trial,reason=result['raw_source_operator_result'].get('reason','undefined_full_wide_pair_estimator')))
        profile_seconds+=time.perf_counter()-begin;return values,boundary,flat
    g,c=prepare(rgb,True);base,bound,baseflat=measure(rgb,g,c,set(range(n)))
    replay=[]
    for i,row in enumerate(rows):
        if i in entities:replay.append(base[slices[i]])
        else:
            origin=(row['frozen_point_operator']['navigation_seed_uv']if row['operator_kind']=='continuous_cubic_point'else(row['wide_pair_source']['current_pair_operator']['original_current_inner_edge_navigation_uv']if row['operator_kind']=='wide_paint_flank'else row['source_material_receipt']['original_navigation_seed_uv']))
            replay.append(np.asarray(origin)+base[slices[i]][0]*np.asarray(row['source_normal']))
    error=float(abs(np.asarray(replay)-means).max())
    if error>1e-9:raise ValueError('Whole-exposure source mean differs from exact estimator '+str(error))
    cached,reuse,cached_diagnostics=_load_scalar_cache(scalar_cache,rows,native,operator,trials,seed)
    selected=set(range(n))-set(cached);D=np.full((trials,d),np.nan)
    boundary=np.zeros(n,np.int64);flat=np.zeros(n,np.int64)
    for i,values in cached.items():
        D[:,slices[i]]=values[:,None];metadata=cached_diagnostics[i]
        if metadata['boundary']is not None:boundary[i]=metadata['boundary']
        if metadata['flat']is not None:flat[i]=metadata['flat']
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(native['native_pts'])]));rng_initial=copy.deepcopy(rng.bit_generator.state)
    rgb64=rgb.astype(np.float64);draw_seconds=0.
    for k in range(trials):
        begin=time.perf_counter();draw=rng.uniform(-.5,.5,rgb.shape);np.add(rgb64,draw,out=draw);draw_seconds+=time.perf_counter()-begin
        g,c=prepare(draw,k==0);values,b,f=measure(draw,g,c,selected,trial=k)
        for i in selected:D[k,slices[i]]=values[slices[i]]-base[slices[i]];boundary[i]+=b[i];flat[i]+=f[i]
    del draw,rgb64,g,c
    # Whole source rows with any undefined outcome remain engineering constraints;
    # no noisy trial is removed and no finite surrogate variance is substituted.
    eligible=[i for i in range(n)if np.isfinite(D[:,slices[i]]).all()]
    rejected=[i for i in range(n)if i not in eligible]
    kept_columns=np.concatenate([np.arange(slices[i].start,slices[i].stop)for i in eligible])if eligible else np.empty(0,int)
    kept=D[:,kept_columns];renumber={old:new for new,old in enumerate(eligible)}
    kept_ri=np.asarray([renumber[int(ri[j])]for j in kept_columns],np.int64);kept_directions=directions[kept_columns]
    branch_switch=np.zeros(n,np.int64)
    for i in discrete_rows:branch_switch[i]=sum(count for key,count in branch_counts[i].items()if key!=branch_base[i])
    begin=time.perf_counter();F=(kept-kept.mean(0)).T/np.sqrt(trials-1);C=F@F.T
    try:Wc,proof=supported_factor_whitening(F)
    except StatisticsUnavailable as exc:Wc=np.empty((0,len(kept_columns)));proof=dict(reason=str(exc),PSD_floor_or_ridge_added=False)
    B=np.zeros((len(kept_columns),2*len(eligible)))
    for j,i in enumerate(kept_ri):B[j,2*i:2*i+2]=kept_directions[j]
    W=Wc@B;identity_error=float(abs(Wc@C@Wc.T-np.eye(len(Wc))).max(initial=0.))
    if identity_error>1e-7:raise ValueError('Mixed component covariance whitening identity failed '+str(identity_error))
    proof['whitening_identity_max_error']=identity_error;cov_seconds=time.perf_counter()-begin
    W.astype('<f8').tofile(out/'W.f64')
    artifacts=_compact_components(kept,C,kept_ri,kept_directions,means[eligible],boundary[eligible],flat[eligible],branch_switch[eligible])
    artifacts['source_row_ids']=np.asarray([rows[i]['source_row_id']for i in eligible])
    if rejected:
        failed_columns=np.concatenate([np.arange(slices[i].start,slices[i].stop)for i in rejected])
        artifacts['unsupported_source_row_ids']=np.asarray([rows[i]['source_row_id']for i in rejected])
        artifacts['unsupported_trial_component_deltas_px']=D[:,failed_columns]
    np.savez_compressed(out/'joint_component_replay.npz',**artifacts)
    pairhash={key:object_sha256(dict(native_pts=native['native_pts'],source_row_id=key,actual_flank_uv=[rows[i]['current_uv']for i in pairs[key]],measurement_operator='finite_width_paint_material_flank_pair'))for key in pairkeys}
    for key,ix in wide_pairs.items():pairhash[key]=object_sha256(dict(native_pts=native['native_pts'],source_row_id=key,actual_flank_uv=[rows[i]['current_uv']for i in ix],measurement_operator='finite_width_paint_material_flank_pair'))
    supported_rows=[rows[i]for i in eligible]
    record=dict(source_group_id='actual_mixed_exposure:'+rows[0]['shot_id']+':'+str(native['native_pts']),
        row_ids=[r['source_row_id']for r in supported_rows],source_candidate_row_ids=[r['source_row_id']for r in rows],
        source_family_ids=sorted({r['source_family_id']for r in rows}),native_pts=native['native_pts'],shot_id=rows[0]['shot_id'],
        status='usable_conditional_joint_native_mixed_source_W'if len(W)else'no_supported_conditional_noise_information',
        rank=len(W),W=dict(file=str(out/'W.f64'),sha256=file_sha256(out/'W.f64'),dtype='<f8',shape=list(W.shape)),
        statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',numeric_whitening_verified=bool(len(W)),
        actual_mean_replay_max_px=error,float32_channel_pipeline_error=0.,
        source_measurements_sha256={r['source_row_id']:_row_hash(r,False)for r in supported_rows},
        current_measurements_sha256={r['source_row_id']:_row_hash(r,True)for r in supported_rows},
        current_pair_measurements_sha256=pairhash,native_receipt=native,
        frozen_operator=dict(file=str(Path(operator.__file__).resolve()),sha256=file_sha256(operator.__file__)),
        entity_frozen_source_operator_refs=list(operator_refs.values()),source_mask_refs=list(mask_refs.values()),
        trials=trials,seed=seed,compute_backend_GPU=True,GPU_source_channel_shadow=shadow,
        conditional_model=dict(encoded_RGB_noise='independent uniform[-.5,+.5] per encoded native RGB pixel/channel, no clipping; same draw shared by all source rows',
            scalar_selection='actual source seed/normal/channel/polarity/windows frozen; same exact nonlinear replay',
            entity_selection='original source seed/tangents/mask fixed; original two-edge operator including channel/polarity selection and source validity domain replayed unchanged',
            wide_paint_selection='entire source wide-window peak-pair selection and both flank refinements replayed; source ellipse navigation/mask/config fixed, not replaced by local flank windows',
            covariance='MC256 complete same-exposure component covariance including 2-D corner/profile cross terms; no PSD ridge/floor or independent representative noise',
            entity_domain='Any source row with an undefined synthetic outcome stays noise-unknown; no trial censoring or quality truncation'),
        physical_total_error_calibrated=False,shared_full_native_RGB_domain=True,
        independent_rows_or_PSD_floor_added=False,reused_same_noise_domain_scalar_samples=reuse,
        source_receipts_sha256=object_sha256(rows),proof=proof,
        replay_arrays=dict(file=str(out/'joint_component_replay.npz'),sha256=file_sha256(out/'joint_component_replay.npz')),
        component_covariance_layout='component_row_index + component_directions; scalar normals or physical-corner (u,v) identity basis',
        MC_estimator_bias_component_px_quantiles=np.quantile(kept.mean(0),[0,.1,.5,.9,1]).tolist()if len(kept_columns)else[],
        component_std_px_quantiles=np.quantile(np.sqrt(np.diag(C)),[0,.1,.5,.9,1]).tolist()if len(kept_columns)else[],
        entity_full_2D_component_rows=[rows[i]['source_row_id']for i in entities if i in eligible],
        complete_wide_pair_estimators_replayed=list(wide_pairs),
        source_statistics_producer=dict(file=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),
        unsupported_source_estimator_domain=[dict(source_row_id=rows[i]['source_row_id'],failures=invalid.get(i,[]),physical_source_withdrawn=False,engineering_constraints_retained=True)for i in rejected],
        unsupported_entity_estimator_domain=[dict(source_row_id=rows[i]['source_row_id'],failures=invalid.get(i,[]),physical_source_withdrawn=False,engineering_constraints_retained=True)for i in rejected if i in entities],
        entity_branch_switch_counts={rows[i]['source_row_id']:int(branch_switch[i])for i in entities},
        wide_pair_branch_switch_counts={key:int(branch_switch[ix[0]])for key,ix in wide_pairs.items()},
        synthetic_window_boundary_count=int(boundary.sum()),synthetic_flat_peak_interval_count=int(flat.sum()),
        cached_source_diagnostics_complete=all(all(v is not None for v in q.values())for q in cached_diagnostics.values()),
        cached_prior_source_diagnostics_retained_in_parent_receipt=reuse,
        full_trials_stored=False,full_MC_draw_stream_exactly_replayable=True,
        storage_policy='component covariance/W, mean/basis, RNG state/operator/native refs, bias/range/digest; actual nonregular diagnostics only',
        source_MC_elapsed_seconds=time.perf_counter()-started,
        timings_seconds=dict(RGB_draw_and_add=draw_seconds,GPU_filters_and_shadow=filter_seconds,source_remeasure=profile_seconds,covariance_and_whitening=cov_seconds))
    record['RNG_replay_contract']=dict(numpy_version=np.__version__,bit_generator=rng.bit_generator.__class__.__name__,
        seed_sequence_entropy=[seed,int(native['native_pts'])],state_integer_encoding='hexadecimal unsigned128 strings',
        initial_state=encode_rng_state(rng_initial),final_state=encode_rng_state(rng.bit_generator.state),
        draw_method='Generator.uniform(-.5,+.5,shape=RGB.shape)',draw_dtype='float64',image_shape=list(rgb.shape),
        image_order='C row-major RGB',trials=trials,same_exposure_stream_shared_by_all_source_rows=True)
    # The exact dependencies are retained separately from the scalar operator.
    from . import entity_corners, source_worker,circular_line_source
    dependencies=([entity_corners,source_worker]if entities else[])+([circular_line_source,source_worker]if wide_pairs else[])
    record['entity_source_dependencies']=list({str(Path(m.__file__).resolve()):dict(file=str(Path(m.__file__).resolve()),sha256=file_sha256(m.__file__))for m in dependencies}.values())
    record['memory_counters']=dict(before_frame=owned_before,after_frame=memory_snapshot())
    write_json(out/'receipt.json',record);return record


def compile_joint_native_statistics(run,factor_rows,frozen_operator,output,*,gpu_ptx,trials=256,seed=20261018,scalar_cache_by_pts=None,stop_file=None):
    from .native_source import iter_training_native_rgb
    out=Path(output)
    if out.exists():raise ValueError('New immutable mixed statistics generation required')
    out.mkdir();write_json(out/'statistics_registry.json',{});frames=defaultdict(list)
    for row in read_jsonl(factor_rows):frames[int(row['native_pts'])].append(row)
    requested=set(frames);operator=_load_frozen(frozen_operator);cache=scalar_cache_by_pts or{};records={};stopped=False;before_context=memory_snapshot();loaded_context=None
    stop=Path(stop_file)if stop_file else out/'STOP_AFTER_CLOSED_EXPOSURE'
    with PTXSourceFilterCUDA(1080,1920,batch=3,ptx_file=gpu_ptx)as backend:
        loaded_context=memory_snapshot()
        iterator=iter(iter_training_native_rgb(run,requested))
        while True:
            if stop.exists():stopped=True;break
            try:pts,rgb,native=next(iterator)
            except StopIteration:break
            rows=frames.pop(pts);record=frame(rgb,native,rows,operator,backend,out/(rows[0]['shot_id']+'_'+str(pts)),trials=trials,seed=seed,scalar_cache=cache.get(str(pts)))
            records[record['source_group_id']]=record
            write_json(out/'statistics_registry.json',records)
            print(json.dumps(dict(completed_exposures=len(records),native_pts=pts,rows=len(record['row_ids']),rank=record['rank'],mean_replay_max=record['actual_mean_replay_max_px'],unsupported_entity_rows=len(record['unsupported_entity_estimator_domain']),seconds=record['source_MC_elapsed_seconds'])),flush=True)
    report=dict(schema='rinkfit.mixed_native_profile_and_entity_conditional_statistics.v1',exposures=len(records),requested_exposures=len(requested),unsubmitted_native_pts=sorted(frames),status='intentional_resource_boundary_yield'if stopped else'complete',
        factor_rows=sum(len(r['row_ids'])for r in records.values()),rank=sum(r['rank']for r in records.values()),trials=trials,
        source_factor_rows=dict(file=str(Path(factor_rows).resolve()),sha256=file_sha256(factor_rows)),
        owned_process_memory=dict(before_context=before_context,context_loaded=loaded_context,after_context_close=memory_snapshot()),
        all_sources_share_full_exposure_RGB_noise=True,statistical_admission_or_material_identity_added=False,
        statistics_registry=dict(file=str(out/'statistics_registry.json'),sha256=file_sha256(out/'statistics_registry.json')))
    write_json(out/'manifest.json',report);return report
