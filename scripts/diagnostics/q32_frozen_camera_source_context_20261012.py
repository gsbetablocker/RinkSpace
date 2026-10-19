"""Extracted small source preparation for q32 camera-condition comparisons.

This uses the existing public material/value/sampling implementations. It does
not reconstruct the complete source graph, motion profiles or other cameras.
"""
from types import SimpleNamespace, MethodType
from copy import deepcopy
import json, hashlib, pickle, time
import compare_q32_single_frame_camera_20261012 as s

np=s.np

def prepare(actual_pts):
    started=time.perf_counter(); selected=tuple(map(int,actual_pts))
    if len(set(selected))!=len(selected): raise ValueError('Distinct actual exposure PTS required')
    opts=s.read(s.CONFIG)['source_options']; manifest=s.read(s.RUN/'input_manifest.json')
    directory=s.RUN/'accepted_states'/s.QID; identity=s.read(directory/'state_identity.json')
    path=directory/'complete_state_UNACCEPTED.npz'; assert s.sha(path)==identity['state_npz_sha256']
    fields=['q','quadratic','scales','aux__canonical_pts','aux__block_start','aux__block_lengths','aux__block_names',
        'aux__installation_xyz_slots','aux__all_installation_optics_installation_order',
        'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices',
        'aux__current_complete_parameter_layout_json']
    with np.load(path,allow_pickle=False)as z: saved={k:z[k].copy()for k in fields}
    assert hashlib.sha256(saved['q'].tobytes()).hexdigest()==s.QID
    assert s.sha(identity['camera_path'])==identity['camera_sha256']
    with np.load(identity['camera_path'],allow_pickle=False)as z:
        cams={k:z[k].copy()for k in ['pts','camera25','exposure_representative_pts','installation_id','view_state4','effective_pp_offset_px']}
        meta=json.loads(str(z['metadata_json'].item()))
    indices={}
    for at in selected:
        ix=np.flatnonzero(cams['pts']==at)
        if len(ix)!=1 or int(cams['exposure_representative_pts'][ix[0]])!=at:
            raise ValueError('Selected PTS must be an exact real exposure, without an alias substitution')
        indices[at]=int(ix[0])
    installations={str(cams['installation_id'][i])for i in indices.values()}
    if len(installations)!=1: raise ValueError('Only one shared physical installation is supported')
    installation=installations.pop(); cal=s.SharedCalibration.from_manifest(meta['calibrations'][installation])
    baseline={at:cams['camera25'][i].copy()for at,i in indices.items()}
    views={at:cams['view_state4'][i].copy()for at,i in indices.items()}
    offsets={at:cams['effective_pp_offset_px'][i].copy()for at,i in indices.items()}
    for at in selected:
        assert np.array_equal(cal.camera25(s.ViewState.from_state4(views[at],effective_pp_offset_px=offsets[at])),baseline[at])
    provider,world,proof=s.current_complete_source_geometry(path,opts['--world-init'])
    decl=json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
    layout=s.CompleteProgrammeParameterLayout(decl['original_complete_parameter_count'],decl['additional_world_parameter_count'])
    assert layout.manifest()==decl
    chart=s.PairedProgrammeCoordinateChart(len(world)-66);q=layout.evaluation_q(saved['q'])
    assert np.array_equal(q[np.asarray(chart.world_indices)],world)
    aliases=dict(zip(map(int,cams['pts']),map(int,cams['exposure_representative_pts'])));actual_index={}
    for i,p in enumerate(cams['exposure_representative_pts']):actual_index.setdefault(int(p),i)
    order=saved['aux__all_installation_optics_installation_order'].tolist();n=len(order)
    optics=saved['aux__current_all_installation_optics_parameter_indices'].reshape(n,3)
    mounts=saved['aux__current_shared_mount_parameter_indices'].reshape(n,3)
    bypts={};cursor=0
    for name,start,length in zip(saved['aux__block_names'],saved['aux__block_start'],saved['aux__block_lengths']):
        stamps=saved['aux__canonical_pts'][cursor:cursor+int(length)//4]
        ids={str(cams['installation_id'][actual_index[int(p)]])for p in stamps};assert len(ids)==1
        install_index=order.index(ids.pop())
        block=SimpleNamespace(name=str(name),start=int(start),xyz_slots=saved['aux__installation_xyz_slots'][install_index],
            optics_chart=SimpleNamespace(indices=optics[install_index],mount_indices=mounts[install_index]),
            bundle=SimpleNamespace(index={int(p):i for i,p in enumerate(stamps)}))
        bypts.update({int(p):block for p in stamps});cursor+=len(stamps)
    state=SimpleNamespace(geometry=provider,geometry_parameter_indices=np.asarray(chart.world_indices),
        structured_programme_coordinate_chart=chart,original=SimpleNamespace(aliases=aliases),bypts=bypts,
        current_user_physical_course_scope_source=opts.get('--physical-course-scope-source'))
    ice=s.NativeIceCurveFactor(state,opts['--additional-ice-source'],len(q));board=s.NativeBoardEdgeFactor(state,opts['--additional-board-source'],len(q))
    integration=s.SourceWithNativeBaseCurveIntegration(SimpleNamespace(native_ice_factor=ice,native_board_factor=board,
        contract={},count=len(q),scales=np.empty(0),quadratic=np.empty(0,bool)),opts['--native-base-curve-integration-policy'])
    trial=s.read(s.RUN/'actual_trial_family_cost_pass_0042.json');assert trial['complete_q_sha256']==s.QID
    ledgers={r['original_row_ledger']['family']:r['original_row_ledger']for r in trial['families']}
    records=[];frozen={}
    for family,factor in [('additional_real_native_ice_curves',ice),('additional_real_native_board_edges',board)]:
        active=factor._qualified_active_fit_view()if hasattr(factor,'_qualified_active_fit_view')else None
        active=factor if active is None else active
        begin,end=ledgers[family]['original_start'],ledgers[family]['original_stop']
        assert np.array_equal(saved['scales'][begin:end],active.scales)
        assert np.array_equal(saved['quadratic'][begin:end],active.quadratic)
        weights,norm=s.loss_weight_arrays(integration._payloads[family],active.row_count);cursor=0
        for row in active.rows:
            size=2*len(row['uv']);lo,hi=cursor,cursor+size;cursor=hi
            if row['actual_pts']not in selected:continue
            key=json.dumps(row['curve'],sort_keys=True);xyz=active._world(q,row['curve']).copy();frozen[id(active),key]=xyz
            mapping=(next(m['retained_original_point_indices']for m in active._qualified_original_index_map if m['id']==row['id'])
                     if hasattr(active,'_qualified_original_index_map')else list(range(len(row['uv']))))
            records.append(dict(family=family,row=row,factor=active,weights=weights[lo:hi].copy(),
                normalization=np.ones(size)if norm is None else norm[lo:hi].copy(),quadratic=saved['quadratic'][begin+lo:begin+hi].copy(),
                saved_global_scalar_range=[begin+lo,begin+hi],source_id=row['id'],actual_pts=int(row['actual_pts']),
                physical_group=row.get('physical_curve_id',row['curve']['kind']),original_active_point_indices=mapping,world=xyz))
        assert cursor==active.row_count
        def cached(self,value,curve=None):return frozen[id(self),json.dumps(curve,sort_keys=True)]
        active._world=MethodType(cached,active)
    assert s.sha(s.PACKET)==s.PACKET_SHA256
    originals=pickle.loads(s.PACKET.read_bytes())['programme_absolute_rows'];calls,call_proof=s.load_original_absolute_evaluators()
    selected_indices={i for i,r in enumerate(originals)if r['collection_role']=='FIT'and int(r['group_actual_pts'])in selected}
    corrections={r['original_packet_row_index']:r for r in s.read(opts['--original-ice-pixel-correction-source'])['records']}
    keeps={}
    for b in manifest['original_absolute_input_compaction']['blocks']:
        for r in b['row_maps']:keeps[b['block'],r['group_order'],r['row_order']]=r['kept_original_point_indices']
    def current_original(index):
        item=originals[index];old=item['source_row'];row=deepcopy(old);change=corrections.get(index)
        if change:
            row['uv']=np.asarray(row['uv'],float).copy();row['uv'][change['point_indices']]=change['corrected_uv']
            if change.get('remove_original_normal_interval'):row.pop('normal_interval',None)
            elif change.get('corrected_normal_interval_px')is not None:
                row['normal_interval']=np.asarray(row['normal_interval'],float).copy();row['normal_interval'][change['point_indices']]=change['corrected_normal_interval_px']
        keep=np.asarray(keeps.get((item['block'],item['group_order'],item['row_order']),list(range(len(old['uv'])))),int)
        for key in ('uv','sigma','normal','normal_interval','source_uv'):
            if key in row and len(np.asarray(row[key]))==len(old['uv']):row[key]=np.asarray(row[key])[keep].copy()
        return row,keep
    # All four selected frames lack this separate installed mode. A future
    # caller must use the public authenticated installer rather than dropping
    # a support field; reject such rows explicitly until that adapter is used.
    mode_packet=s.read(opts['--original-measured-transition-location-source'])
    overlap=selected_indices&{r['original_packet_row_index']for r in mode_packet['records']}
    if overlap:raise ValueError('Selected source rows need public measured-location certificate installation: '+str(sorted(overlap)))
    compact=s.read(manifest['original_absolute_input_compaction']['packet_path'])
    overlap=selected_indices&{r['original_entry_index']for r in compact['records']}
    if overlap:raise ValueError('Selected source rows need the existing nearboard49 loss payload: '+str(sorted(overlap)))
    masks={};ranges={};counts=dict(zip(map(str,saved['aux__block_names']),map(int,saved['aux__block_lengths']//4)))
    selected_blocks={originals[i]['block']for i in selected_indices}
    for b in manifest['original_absolute_input_compaction']['blocks']:
        name=b['block']
        if name not in selected_blocks:continue
        ledger=ledgers['programme:'+name];cursor=int(ledger['original_start'])+2*int(b['motion_point_count'])
        end=int(ledger['original_stop'])-counts[name];assert int(b['compact_start'])==int(ledger['original_start'])
        for index,item in enumerate(originals):
            if item['block']!=name or item['collection_role']!='FIT':continue
            old=item['source_row'];keep=keeps.get((name,item['group_order'],item['row_order']),list(range(len(old['uv']))))
            size=2*len(keep);stop=cursor+size
            assert stop<=end and np.array_equal(saved['scales'][cursor:stop],np.repeat(np.asarray(old['sigma'])[keep],2))
            if index in selected_indices:masks[index]=saved['quadratic'][cursor:stop].copy();ranges[index]=[cursor,stop]
            cursor=stop
        assert cursor==end
    ice_sources={r['original_packet_row_index']:r for r in manifest['original_ice_curve_integration']['source_rows']}
    board_sources={(r['actual_pts'],r['group_order'],r['row_order']):r for r in manifest['original_board_curve_integration']['source_rows']}
    support=s.OriginalDetectorSupport(manifest['original_board_curve_integration']['source_video_sha256'],
                                    manifest['original_board_curve_integration']['source_station_gap_policy'])
    plans={'board':[],'ice':[]};pending=[]
    for index in sorted(selected_indices):
        item=originals[index];old=item['source_row'];row,keep=current_original(index);at=int(item['group_actual_pts'])
        if not len(keep):continue
        entry=ice_sources.get(index);category='ice'if entry is not None else 'board'
        if entry is None:entry=board_sources.get((at,item['group_order'],item['row_order']))
        name=f'programme:{item["block"]}:{item["group_order"]}:{item["row_order"]}'
        if entry is not None:
            stored=np.asarray(old['sigma'],float);assert keep.tolist()==entry['original_point_indices']
            assert hashlib.sha256(np.asarray(row['uv'],float).tobytes()).hexdigest()==entry['current_active_UV_sha256']
            if category=='ice':
                working,prior,_=s.original_working_sigma(old,stored,entry.get('original_declared_scan_group_count')or 0)
                mass=s.isolated_anchor_masses(row['uv']);physical=json.dumps(entry['physical_identity'],separators=(',',':'))
                operator=json.dumps({k:row.get(k)for k in ('curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side')},sort_keys=True)
            else:
                working,prior,_=s.declared_working_sigma(old,stored);physical=entry['physical_curve_id']
                operator=str(row.get('curve_observation_kind','finite_curve_vector_distance'))
                mass=(np.ones(len(keep))if entry['support_basis']=='isolated_measured_corrected_RGB_anchors'
                    else support.masses(old,row['uv'],at,original_point_indices=keep)[0])
                assert np.isclose(mass.sum(),entry['source_support_mass'],rtol=1e-12,atol=1e-12)
            contract=manifest['original_'+category+'_curve_integration']
            plans[category].append(dict(id=name,split='fit',actual_pts=at,source_video_sha256=contract['source_video_sha256'],
                physical_curve_id=physical,operator_id=operator,uv=np.asarray(row['uv']),stored_sigma_px=stored[keep],
                working_sigma_px=working[keep],prior_sampling_multiplier=prior[keep],point_integration_mass=mass,
                sampling_kind=['measured']*len(keep),source_anchor_identity_authority=contract['duplicate_anchor_policy']['source_anchor_identity_authority']))
        else:physical=str(old['id']);category=None
        provider.register_row(old,actual_pts=at);bound=provider.rebind(row,world)
        pending.append(dict(family='original_absolute',row=bound,call=calls[item['block']],source_id=f'{index}:{old["id"]}',
            actual_pts=at,physical_group=physical,original_packet_row_index=index,original_active_point_indices=keep.tolist(),
            quadratic=masks[index],saved_global_scalar_range=ranges[index],category=category,plan_name=name,
            weights=np.ones(2*len(keep)),normalization=np.ones(2*len(keep)),world=np.asarray(bound['world']).copy()))
    for category,rows in plans.items():
        if not rows:continue
        contract=manifest['original_'+category+'_curve_integration'];plan=s.compile_curve_sampling(rows,duplicate_anchor_policy=contract['duplicate_anchor_policy'])
        for group in plan['groups']:
            match=next(g for g in contract['groups']if g['actual_pts']==group['actual_pts']and g['physical_curve_id']==group['physical_curve_id'])
            assert json.dumps(group,sort_keys=True,default=s.plain)==json.dumps(match,sort_keys=True,default=s.plain)
        for r in pending:
            if r['category']==category:
                info=plan['rows'][r['plan_name']];r.update(weights=np.repeat(info['point_integration_weights'],2),
                    normalization=np.repeat(info['working_normalization_multiplier'],2))
    records.extend(pending)
    # Suppression is a current source-role operation, not a robust-loss change.
    for r in records:
        retired=manifest['original_source_role_retirement'].get('global_scalar_slots',[])
        if any(r['saved_global_scalar_range'][0]<=x<r['saved_global_scalar_range'][1]for x in retired):
            raise ValueError('Selected source contains an explicitly retired scalar; public retirement adapter required')
    return SimpleNamespace(selected_pts=selected,options=opts,manifest=manifest,identity=identity,directory=directory,path=path,
        saved=saved,q=q,provider=provider,world=world,world_proof=proof,installation_id=installation,calibration=cal,
        baseline_cameras=baseline,views=views,pp_offsets=offsets,records=records,call_proof=call_proof,
        prepare_seconds=time.perf_counter()-started)

def record_value(c,r,camera,raw=False):
    row=s.raw_row(r['row'])if raw else r['row']
    if r['family']=='original_absolute':return r['call'](camera,[row],False)
    return r['factor']._residual(c.q,[row],cameras={r['actual_pts']:camera},
        **({'apply_optical_support':not raw}if r['family'].endswith('board_edges')else {}))

def residual(c,cameras):return np.concatenate([record_value(c,r,cameras[r['actual_pts']])*r['normalization']for r in c.records])

def loss(c):
    weights=np.concatenate([r['weights']for r in c.records]);quadratic=np.concatenate([r['quadratic']for r in c.records])
    def evaluate(z):
        sq=np.sqrt(z);rho=np.empty((3,len(z)));robust=(~quadratic)&(z>4.)
        rho[0]=z;rho[1]=1.;rho[2]=0.;rho[0,robust]=4.*sq[robust]-4.
        rho[1,robust]=2./sq[robust];rho[2,robust]=-1./(z[robust]*sq[robust])
        return rho*weights[None,:]
    return evaluate

def metrics(c,cameras):
    out=[]
    for r in c.records:
        camera=cameras[r['actual_pts']];actual=record_value(c,r,camera);raw=record_value(c,r,camera,True);sigma=np.asarray(r['row']['sigma'])
        rawpx=raw.reshape(-1,2)*sigma[:,None];outside=actual.reshape(-1,2)*sigma[:,None]
        out.append(dict(family=r['family'],id=r['source_id'],actual_pts=r['actual_pts'],physical_group=r['physical_group'],points=len(sigma),
            raw_native_px=s.stats(np.linalg.norm(rawpx,axis=1)),outside_support_native_px=s.stats(np.linalg.norm(outside,axis=1)),
            raw_mean_vector_xy_px=rawpx.mean(axis=0),outside_mean_vector_xy_px=outside.mean(axis=0),
            weighted_current_cost=s.weighted_source_cost(actual,r['quadratic'],s.SourceLossWeights(r['weights'],r['normalization']))))
    return out
