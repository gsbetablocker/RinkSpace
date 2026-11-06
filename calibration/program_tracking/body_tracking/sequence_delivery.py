"""Prepare one real whole-S05 player candidate; no fit, network, GUI or video.

The commands action is safe before the candidate exists. Every other action
requires genuine final result, parameter and summary files with all951 PTS.
"""
from __future__ import annotations
import argparse,copy,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')
CASE=Path('common/exmr_hand_nofoot')
CONFIG='viewer3d_sequence_global_1105.json'
ORIGINAL='continuous_source'
PARENT='sequence_parent'
CURRENT='sequence_global'
KEY_SECONDS=(76.,81.60,81.62,87.60,89.98,92.40,93.50,95.)
REQUIRED=['com_world_mid','blade_material_world_mid','skate_witness_world_mid']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial)+'\n',encoding='utf8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def locations(a):
    global PARENT,CURRENT,KEY_SECONDS
    project=Path(a.project).resolve();sys.path[:0]=[str(project),str(project/'src')]
    from rinkspace.storage import AssetStore
    store=AssetStore(project,data_root=a.data_root);base=store.data/'runs/body_s01_20261029';sequence=base/'body_sequence_1105'
    family=getattr(a,'family','global');blade=family=='blade_image';structure=family=='source_structure'
    if structure:
        PARENT='sequence_blade_image';CURRENT='sequence_source_structure'
        KEY_SECONDS=(76.,78.4,80.4,81.6,81.62,87.64,92.44,93.5,95.)
        config_name='viewer3d_sequence_source_structure_1106.json';out=sequence/'delivery_source_structure_1106'
        stage=sequence/'full_source_fit_1106';parameter=project/PROGRAMME/'spin_sequence_source_structure_1106.npz'
        parent_stage=sequence/'global_blade_image_final';parent_parameter=project/PROGRAMME/'spin_sequence_blade_image_1105.npz'
        source_config=project/PROGRAMME/'viewer3d_sequence_blade_1105.json'
        session='source_structure_1106';launcher='START_3D_SEQUENCE_SOURCE_STRUCTURE.cmd'
    else:
        PARENT='sequence_global' if blade else 'sequence_parent'
        CURRENT='sequence_blade_image' if blade else 'sequence_global'
        KEY_SECONDS=(76.,81.60,81.62,87.64,89.98,92.44,93.50,95.) if blade else (76.,81.60,81.62,87.60,89.98,92.40,93.50,95.)
        config_name='viewer3d_sequence_blade_1105.json' if blade else CONFIG
        out=sequence/('delivery_blade_image' if blade else 'delivery')
        stage=sequence/('global_blade_image_final' if blade else 'global_fit_v3')
        parameter=project/PROGRAMME/('spin_sequence_blade_image_1105.npz' if blade else 'spin_sequence_global_1105.npz')
        parent_stage=sequence/'global_fit_v3' if blade else base/'blade_modes_1105/spin_safe'
        parent_parameter=project/PROGRAMME/('spin_sequence_global_1105.npz' if blade else 'spin_modes_spin_safe_1105.npz')
        source_config=project/PROGRAMME/(CONFIG if blade else 'viewer3d_runner_support_1105.json')
        session='sequence_blade_image_1105' if blade else 'sequence_global_1105'
        launcher='START_3D_SEQUENCE_BLADE.cmd' if blade else 'START_3D_SEQUENCE_GLOBAL.cmd'
    loc=dict(project=project,store=store,base=base,sequence=sequence,out=out,family=family,
        stage=store.resolve(a.stage) if a.stage else stage,
        parameter=store.resolve(a.parameter) if a.parameter else parameter,
        parent_stage=store.resolve(a.parent_stage) if getattr(a,'parent_stage',None) else parent_stage,
        parent_parameter=store.resolve(a.parent_parameter) if getattr(a,'parent_parameter',None) else parent_parameter,
        formal=project/PROGRAMME/config_name,staged=out/config_name,source_config=source_config,
        source_asset=store.resolve(a.source_asset) if getattr(a,'source_asset',None) else None,
        session=session,launcher=launcher)
    if structure:
        for path in (loc['formal'],loc['staged'],loc['parameter'],out/'sequence_blade_review.json',out/'sequence_blade_review.npz'):
            guard(path,project)
    return loc


def retained_variants(loc,before):
    # Keep the existing 39-step result path and content-bound cache unchanged.
    return [copy.deepcopy(v) for v in before['variants'] if v['id']=='sequence_global'] if loc['family']=='source_structure' else []


def guard(p,project):
    from rinkspace.path_policy import guard_project_write
    p=Path(p).resolve()
    if len(str(p).encode('utf-16-le'))//2>=260 or any(len(v.encode('utf-16-le'))//2>255 for v in p.parts):raise ValueError('Windows path too long: '+str(p))
    return guard_project_write(p,project_root=project)


def checked(loc,include_legacy=False):
    pts=np.arange(972800,1216001,256,dtype=np.int64)
    before=read(loc['source_config'])
    original=next(v for v in before['variants'] if v['id']==ORIGINAL)
    original_stage=Path(original['result']).parents[3]
    original_summary=read(original_stage/'run_summary.json')
    defs={ORIGINAL:dict(stage=original_stage,parameter=Path(original_summary['parameter']),label=original['label']),PARENT:dict(stage=loc['parent_stage'],parameter=loc['parent_parameter'],label='原14步钢片源轮廓整段候选（未验收）' if loc['family']=='source_structure' else '39步全序列诊断基线（未视觉通过）' if loc['family']=='blade_image' else '父候选 spin_safe（完整76–95秒）'),
        CURRENT:dict(stage=loc['stage'],parameter=loc['parameter'],label='完整源轮廓＋条件支撑XY整段候选（76–95秒，未验收）' if loc['family']=='source_structure' else '新钢片源轮廓全序列候选（完整76–95秒，未验收）' if loc['family']=='blade_image' else '新全序列联合候选（完整76–95秒，未验收）')}
    if include_legacy:
        defs.update(legacy_entry=dict(stage=loc['base']/'blade_modes_1105/entry',parameter=loc['project']/PROGRAMME/'spin_modes_entry_1105.npz',label='Retained old entry window'),
            legacy_release=dict(stage=loc['base']/'blade_modes_1105/release',parameter=loc['project']/PROGRAMME/'spin_modes_release_1105.npz',label='Retained old release window'))
    for vid,v in defs.items():
        v['result']=v['stage']/CASE/'fitted/result.npz';v['raw_result']=v['result'];v['summary_path']=v['stage']/'run_summary.json'
        for p in (v['parameter'],v['result'],v['summary_path'],v['stage']/CASE/'input.npz',v['stage']/CASE/'compact_smplx.npz'):
            if not p.is_file():raise FileNotFoundError('Actual completed '+vid+' artifact is absent: '+str(p))
        summary=read(v['summary_path'])
        if Path(summary['result']).resolve()!=v['result'].resolve() or Path(summary['parameter']).resolve()!=v['parameter'].resolve():raise ValueError(vid+' summary identity mismatch')
        z=load(v['result']);v.update(saved=z,summary=summary)
        np.testing.assert_array_equal(z['pts'],pts);np.testing.assert_array_equal(z['source_pts'],pts);np.testing.assert_array_equal(z['time_base'],[1,12800])
        np.testing.assert_array_equal(load(v['stage']/CASE/'input.npz')['pts'],pts)
        for k,shape in [('skate_R_world_mid',(951,2,3,3)),('skate_origin_world_mid',(951,2,3)),('COCO_uv',(951,17,2))]:
            if z[k].shape!=shape or not np.isfinite(z[k]).all():raise ValueError(vid+' invalid '+k)
        R=z['skate_R_world_mid'];assert np.allclose(R.swapaxes(-1,-2)@R,np.eye(3),rtol=0,atol=1e-6)
        for k in REQUIRED:
            if k not in z and k!='blade_material_world_mid':raise ValueError(vid+' missing '+k)
    if tuple(before['source_time_base'])!=(1,12800):raise ValueError('Player source clock mismatch')
    return pts,defs,before


def source_runner_rows(asset):
    """Read existing source runner labels for display; never create source lines."""
    rows=[]
    for frame in asset.get('source_structures',[]):
        feature=frame.get('features',{}).get('runner_lower_edge',{})
        points=feature.get('points_uv',[])
        if not points:continue
        rows.append(dict(pts=frame['pts'],foot=frame['foot'],role=frame.get('role','unknown'),
            id=f"source_runner_{frame['pts']}_{frame.get('foot')}_{frame.get('source_track_id','')}",
            evidence_status=feature.get('status','unknown'),source_status=feature.get('status','unknown'),
            polyline_uv=points,pixel_uncertainty_radius_px=feature['pixel_uncertainty_radius_px']))
    return rows


def modes(z,n):
    if 'sequence_contact_modes' not in z:return np.full((n,2),'unknown',dtype='U32'),np.full((n,2),'unknown',dtype='U32')
    value=np.asarray(z['sequence_contact_modes'])
    if value.shape!=(n,2):raise ValueError('sequence_contact_modes shape')
    if value.dtype.kind in 'iu':value=np.asarray(z['sequence_contact_mode_names'])[value]
    status=np.asarray(z.get('sequence_contact_status',np.full((n,2),'inferred')))
    if status.shape!=(n,2):raise ValueError('sequence_contact_status shape')
    return value.astype('U32'),status.astype('U32')


def blade_assets(loc,pts,defs,before):
    from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
    from calibration.program_tracking.body_tracking.skate_physics import batch_features,fk_state
    from calibration.program_tracking.body_tracking.skate_template import profile_height
    from calibration.program_tracking.body_tracking.blade_contact_modes import blade_axes_diagnostics
    from rinkspace.viewer3d.blade_review import BladeReview
    os.environ.setdefault('RINKSPACE_BODY_NATIVE_LIBRARY',str(loc['store'].data/'runs/body_perf_1103/native_cpu/rinkspace_core.dll'))
    out=loc['out'];out.mkdir(parents=True,exist_ok=True)
    source_asset=Path(loc.get('source_asset') or before['blade_review_asset']);asset=read(source_asset);original_asset=copy.deepcopy(asset)
    old_arrays=load(source_asset.parent/asset['projection_npz']);arrays={k:v.copy() for k,v in old_arrays.items()};np.testing.assert_array_equal(arrays['pts'],pts)
    source_status=np.full((len(pts),2,2),'unknown',dtype='U16')
    temporal=read(loc['sequence']/'temporal_regions.json')
    for frame in temporal['clock']:
        i=int(np.searchsorted(pts,frame['pts']))
        for row in frame['regions']:
            if row['role']=='fit':source_status[i,row['foot'],0 if row['feature']=='heel' else 1]=row['status']
    for vid in (PARENT,CURRENT):
        v=defs[vid];z=v['saved'];query=SavedBladeProjection(loc['project'],v['stage'],v['parameter']);v['query']=query
        np.testing.assert_array_equal(query.pts,pts)
        packet=query.binding.packet();p=packet['parameters'];R=z['skate_R_world_mid'];O=z['skate_origin_world_mid']
        evaluated=fk_state(query.reference,query.reference.at(query.times),np.zeros((len(pts),69)))
        np.testing.assert_allclose(evaluated['R'],R,rtol=0,atol=2e-9);np.testing.assert_allclose(evaluated['O']*query.ratio,O,rtol=0,atol=2e-9)
        representative=np.asarray([round(t*12800) for t in KEY_SECONDS],dtype=np.int64)
        source_indices=np.searchsorted(pts,representative)
        projected_body=query.project_points(representative,0,np.zeros((1,3)),include_body=True)['COCO_uv']
        np.testing.assert_allclose(projected_body,z['COCO_uv'][source_indices],rtol=0,atol=2e-6)
        v['projection_verification']=dict(source_pts=representative.tolist(),R_O_all951_match_saved=True,
            maximum_COCO_native_pixel_difference=float(np.max(np.abs(projected_body-z['COCO_uv'][source_indices]))),
            coordinates='actual native Brown five-node rolling-shutter projection')
        f=batch_features(query.binding,R,O/query.ratio);h=f['height'];q=f['local_point'];ii=np.arange(len(pts))[:,None];ff=np.arange(2)[None]
        eligible=np.asarray([i for i,k in enumerate(f['ids']) if k in packet['template']['support_feature_ids']['glide']],int)
        picks=np.asarray([i for i,k in enumerate(f['semantics']) if k=='toe_pick'],int)
        best=eligible[np.argmin(h[...,eligible],axis=-1)];qr=q[ii,ff,best]
        rear=-p.runner_rear_extension_m;front=p.sole_length_m-p.runner_front_inset_m
        xs=np.array([rear,(rear+front)/2,front]);stations=np.c_[xs,np.zeros(3),profile_height(xs,p)]
        points=np.r_[stations,packet['adapter'][[0,7]].mean(0)[None],packet['adapter'][[1,8]].mean(0)[None]]
        uv=np.stack([query.project_points(pts,foot,points)['uv'] for foot in (0,1)],axis=1)
        rates=np.full((len(pts),2),np.nan);F=R[...,0]
        rates[1:]=np.arccos(np.clip(np.sum(F[1:]*F[:-1],axis=-1),-1,1))/(np.diff(pts)/12800.)[:,None]
        world=O[:,:,None]+np.einsum('nfij,kj->nfki',R,stations)*query.ratio
        if 'blade_material_world_mid' in z:
            np.testing.assert_allclose(world,z['blade_material_world_mid'],rtol=0,atol=2e-9)
        else:
            # Export omitted one optional display diagnostic. Derive only the
            # fixed model stations from savedR/O and the same frozen template;
            # keep every original result array and its source SHA untouched.
            display=dict(z)
            display['blade_material_world_mid']=world
            display['display_derivation_json']=np.array(json.dumps(dict(
                source_result=str(v['raw_result']),source_result_sha256=sha(v['raw_result']),
                derived_field='blade_material_world_mid',role='display-only model diagnostic',
                formula='saved_skate_origin + saved_skate_R @ fixed_template_rear_mid_front',
                source_observation=False,fit=False),sort_keys=True))
            display_path=out/(vid+'_display_result.npz')
            # Reuse an existing source-bound display packet, notably the39-step
            # parent. Its path/bytes preserve the original content-bound cache.
            reusable=[Path(row['result']) for row in before['variants'] if row['id']==vid]
            reusable.append(display_path)
            reused=False
            for previous in reusable:
                if not previous.is_file():continue
                previous_packet=load(previous)
                provenance=json.loads(str(previous_packet.get('display_derivation_json','{}')))
                if provenance.get('source_result_sha256')!=sha(v['raw_result']):continue
                np.testing.assert_allclose(previous_packet['blade_material_world_mid'],world,rtol=0,atol=2e-9)
                for key in z:np.testing.assert_array_equal(previous_packet[key],z[key])
                display_path=previous;display=previous_packet;reused=True;break
            if not reused:np.savez_compressed(display_path,**display)
            v['result']=display_path;v['saved']=display
            with np.load(display_path,allow_pickle=False) as packed:
                for key in z:np.testing.assert_array_equal(packed[key],z[key])
            print(json.dumps(dict(display_derived_packet=str(display_path),original_result_preserved=str(v['raw_result']))),flush=True)
        u=(qr[...,0]-rear)/(front-rear);mode,status=modes(z,len(pts));axes=blade_axes_diagnostics(R)
        values=dict(blade_uv=uv[:,:,:3],boot_uv=uv[:,:,3:],axis_rate_rad_s=rates,R_world=R,origin_world_conditional_m=O,
            pitch_degrees=axes['fore_aft_pitch_degrees'],roll_degrees=axes['edge_bank_degrees'],runner_u=u,runner_min_x_fraction=u,
            material_station_world_conditional_m=world,station_height_m=world[...,2],runner_minus_pick_m=h[ii,ff,best]-h[...,picks].min(-1),contact_mode=mode,inferred_contact_mode=mode,inferred_contact_status=status,
            edge_mode=np.full((len(pts),2),'unknown',dtype='U32'),mode_conflict=np.zeros((len(pts),2),bool),source_boot_region_status=source_status,
            source_contact_state=np.asarray(z.get('source_support_mode_mid',np.full((len(pts),2),'unknown'))).astype('U32'),
            source_observation_uv=z.get('source_uv_native',np.full((len(pts),17,2),np.nan)),source_observation_use=z.get('source_observation_use',np.zeros((len(pts),17),bool)),**axes)
        if loc['family'] in ('blade_image','source_structure'):
            edges=np.asarray(packet['edges'])
            edge_uv=np.stack([query.project_points(pts,foot,edges.reshape(-1,3))['uv'].reshape(len(pts),*edges.shape[:2],2) for foot in (0,1)],axis=1)
            values['runner_edges_uv']=edge_uv
            if loc['family']=='source_structure':
                upper=edges+[0,0,p.blade_web_height_m]
                values['runner_upper_edges_uv']=np.stack([query.project_points(pts,foot,upper.reshape(-1,3))['uv'].reshape(len(pts),*edges.shape[:2],2)for foot in (0,1)],axis=1)
            contours=(dict(rows=source_runner_rows(asset)) if loc['family']=='source_structure' else
                read(loc['stage']/'support_and_free_shoe_regions.json').get('visible_runner_contours',{}))
            contour_status=np.full((len(pts),2),'not_directly_observed',dtype='U32')
            contour_role=np.full((len(pts),2),'none',dtype='U16')
            for source_row in contours.get('rows',[]):
                if source_row.get('evidence_status')!='observed':continue
                if source_row.get('foot')not in (0,1):continue
                row_index=int(np.searchsorted(pts,int(source_row['pts'])))
                assert pts[row_index]==int(source_row['pts'])
                contour_status[row_index,int(source_row['foot'])]='observed'
                contour_role[row_index,int(source_row['foot'])]=source_row['role']
            values.update(pose_estimate_status=np.full(len(pts),'inferred_joint_sequence',dtype='U32'),
                source_runner_contour_status=contour_status,source_runner_contour_role=contour_role,
                source_contact_truth_status=np.full((len(pts),2),'unmeasured',dtype='U32'))
        for key,value in values.items():
            name=vid+'_'+key
            if loc['family']=='source_structure' and vid==PARENT and name in old_arrays:
                # The parent is the same frozen candidate. Preserve all source
                # statuses and projection bytes already bound to its old asset.
                if key in ('blade_uv','boot_uv','R_world','origin_world_conditional_m','runner_edges_uv'):
                    np.testing.assert_allclose(value,old_arrays[name],rtol=0,atol=2e-6)
                value=old_arrays[name]
            arrays[name]=value
        asset['variants'][vid]=vid;v['profile']=values
    for key in ('source_boot_axes','source_regions','source_visibility','source_steel_segments','source_structures','source_plane_locations','source_ice_grid'):assert asset.get(key)==original_asset.get(key)
    if loc['family']=='blade_image':
        contours=read(loc['stage']/'support_and_free_shoe_regions.json').get('visible_runner_contours',{})
        observed=[r for r in contours.get('rows',[]) if r.get('evidence_status')=='observed']
        for row in observed:
            asset['source_steel_segments'].append(dict(pts=int(row['pts']),foot=int(row['foot']),points=row['polyline_uv'],role=row['role'],
                source_semantic='observed_visible_runner_subcontour',evidence_status='observed',pixel_uncertainty_radius_px=row['pixel_uncertainty_radius_px'],
                material_correspondence='latent_geometry_match; no fixed material ID',contact_truth='unmeasured'))
        asset['observed_runner_contour_document']=contours
        asset['sequence_pose_estimate_status']=dict(pts=pts.tolist(),status=['inferred_joint_sequence']*len(pts),
            scope='All951 native exposures have an estimated trajectory; no direct steel contour at a frame does not remove its pose',
            direct_contour_rows_by_role={role:sum(r['role']==role for r in observed) for role in ('fit','review')},
            source_contact_truth='unmeasured',source_mode_truth='unmeasured; model modes remain inferred')
        asset['source_steel_segment_scope']='Existing source steel plus newly source-observed polylines; no model-generated source geometry'
    if loc['family']=='source_structure':
        asset.update(upstream_source_review_asset=str(source_asset),
            source_structure_passthrough='Source structures, plane locations, grids and statuses are preserved from the supplied source asset; no source observation is generated by delivery',
            source_support_xy_semantics='Camera/Z0/scale-conditioned same-source footprint; not independent measured 3-D or confirmed contact')
    asset.update(projection_npz='sequence_blade_review.npz',temporal_inferred_regions=str(loc['sequence']/'temporal_regions.json'),
        model_source_separation='Candidate axes/u/modes are saved model diagnostics. Source unknown and inferred region status remain separate.',
        inferred_mode_semantics='Duration-aware model/source-conditioned inference, never measured edge/contact truth')
    projection=out/'sequence_blade_review.npz';np.savez_compressed(projection,**arrays);target=out/'sequence_blade_review.json';write(target,asset)
    for key in old_arrays:np.testing.assert_array_equal(arrays[key],old_arrays[key])
    review=BladeReview(target,time_base=[1,12800]);assert len(review.index)==951
    for vid in (PARENT,CURRENT):assert all(review.at(vid,int(pt))['model'] is not None for pt in pts)
    report=dict(path=str(projection),sha256=sha(projection),actual_R_O_match_saved=True,source_rows_preserved=True,original_projection_arrays_preserved=True)
    if loc['family']=='source_structure':report.update(source_asset=str(source_asset),source_asset_sha256=sha(source_asset),source_structure_fields_passed_through=True)
    return target,report


def register(loc,publish=False,include_legacy=False):
    pts,defs,before=checked(loc,include_legacy);asset,projection=blade_assets(loc,pts,defs,before);config=copy.deepcopy(before)
    config['variants']=[]
    for vid,v in defs.items():
        config['variants'].append(dict(id=vid,label=v['label'],kind='fullbody',result=str(v['result']),shell_topology='saved',body_reference_label='exmr_hand_nofoot',required_saved_diagnostics=REQUIRED,body_refinement_intervals_seconds=[[76.,95.]]))
    retained=retained_variants(loc,before)
    config['variants'].extend(retained)
    config.update(session_id=loc['session'],default_variant=CURRENT,comparison_variants=[ORIGINAL,PARENT,CURRENT],initial_source_pts=int(pts[0]),blade_review_asset=str(asset),
        body_window_scope='One unified76-95s state, all951 native exposures, no independent window fitting',review_status='Saved whole-sequence joint candidate; source/contact/3D accuracy unaccepted',
        source_sequence_ledger=str(loc['sequence']/'source_sequence.json'),temporal_inferred_regions=str(loc['sequence']/'temporal_regions.json'),
        retained_legacy_configuration=str(loc['source_config']))
    if loc['family']=='source_structure':
        config.update(source_structure_review_asset=str(asset),
            source_structure_review_frames=sorted({int(r['pts']) for r in read(asset).get('source_structures',[])}),
            upstream_source_structure_review_asset=str(loc.get('source_asset') or before.get('source_structure_review_asset',before['blade_review_asset'])),
            review_status='完整源轮廓＋条件支撑XY的76–95秒连续候选；人体／鞋刀／接触准确性未验收')
    if include_legacy:
        config['legacy_window_comparisons']=dict(entry=dict(variant='legacy_entry',seconds=[81.2,82.0]),spin=dict(variant=PARENT,seconds=[87.6,90.0]),release=dict(variant='legacy_release',seconds=[92.4,93.6]))
    expected_ids=[ORIGINAL,PARENT,CURRENT]+(['legacy_entry','legacy_release'] if include_legacy else [])+[v['id'] for v in retained]
    assert [v['id'] for v in config['variants']]==expected_ids and config['default_variant']==CURRENT and config['initial_source_pts']==972800
    if loc['family']=='source_structure':config.update(initial_source_pts=1121792,source_review_model_runner_default=True)
    target=loc['formal'] if publish else loc['staged']
    if publish:guard(target,loc['project'])
    if target.exists():
        previous=read(target)
        if (previous.get('session_id')!=loc['session'] and not previous.get('source_structure_review_asset')) or next((v['result'] for v in previous['variants'] if v['id']==CURRENT),None)!=str(defs[CURRENT]['result']):
            raise FileExistsError('Preserve unrelated existing configuration: '+str(target))
    if target.exists():
        # Keep a subsequently reviewed source-only layer for this same frozen
        # candidate; cache preparation must not silently erase source readings.
        previous=read(target)
        source_asset=previous.get('source_structure_review_asset')
        if source_asset and Path(source_asset).is_file() and not (loc['family']=='source_structure' and loc.get('source_asset')):
            config['session_id']=previous['session_id']
            config['source_structure_review_asset']=source_asset
            config['blade_review_asset']=source_asset
            config['initial_source_pts']=previous.get('initial_source_pts',config['initial_source_pts'])
            config['source_structure_review_frames']=previous.get('source_structure_review_frames',[])
    write(target,config)
    report=dict(schema='rinkspace.sequence-delivery.v1',config=str(target),source_frames=951,time_base=[1,12800],first_last_pts=pts[[0,-1]],
        variants={vid:dict(result=str(v['result']),result_sha256=sha(v['result']),source_result=str(v['raw_result']),source_result_sha256=sha(v['raw_result']),parameter=str(v['parameter']),parameter_sha256=sha(v['parameter']),projection_verification=v.get('projection_verification')) for vid,v in defs.items()},
        blade_review=projection,existing_config_preserved=dict(path=str(loc['source_config']),sha256=sha(loc['source_config'])),
        no_gui=True,no_fit=True,no_network=True,no_video=True,accuracy_accepted=False,code=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).resolve())))
    write(loc['out']/'delivery.json',report);print(json.dumps(dict(registered=str(target),frames=951)),flush=True);return target


def configuration(loc):
    for key in ('formal','staged'):
        if loc[key].is_file():return loc[key]
    raise FileNotFoundError('Register actual completed candidate first')


def prepare_caches(loc,include_legacy=False):
    checked(loc,include_legacy)
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import FullSurface,prepare_variant
    catalog=ResultCatalog(project=loc['project'],data_root=loc['store'].data,config=configuration(loc))
    ids=[ORIGINAL,PARENT,CURRENT]+(['legacy_entry','legacy_release'] if include_legacy else [])+[v['id'] for v in retained_variants(loc,read(loc['source_config']))];surface=None;products={}
    for vid in ids:
        variant=catalog.variant(vid);path,_=catalog.cache_path(variant)
        if not (path/'metadata.json').is_file():
            if loc['family']=='source_structure' and vid!=CURRENT:
                raise FileNotFoundError('Existing baseline display cache is absent; preserve its original cache instead of rebuilding: '+str(path))
            if surface is None:surface=FullSurface(catalog)
            path=prepare_variant(catalog,variant,surface)
        with np.load(path/'state.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['pts'],catalog.pts)
            for key in ('skate_R','skate_origin','com_world_mid','blade_material_world_mid'):
                if key not in z:raise ValueError('Prepared cache missing '+key)
        products[vid]=dict(path=str(path),metadata=str(path/'metadata.json'),frames=951,pose_estimate_status='inferred_joint_sequence',source_contact_truth='unmeasured')
    write(loc['out']/'display_caches.json',products);return products


def fit_image(im,size):
    from PIL import Image
    scale=min(size[0]/im.width,size[1]/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
    target=Image.new('RGB',size,'white');target.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return target


def side_panels(defs,index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    panels=[];metadata=[]
    for vid in (PARENT,CURRENT):
        saved=defs[vid]['saved'];fig=plt.figure(figsize=(6.4,3.2),dpi=100)
        for foot in (0,1):
            ax=fig.add_subplot(1,2,foot+1,projection='3d',proj_type='ortho')
            points=np.concatenate([defs[v]['saved']['skate_lower_world_mid'][index,foot] for v in (PARENT,CURRENT)])
            centre=points.mean(0);span=max(.42,float(np.ptp(points,axis=0).max())*1.3)
            lower=saved['skate_lower_world_mid'][index,foot];R=saved['skate_R_world_mid'][index,foot];O=saved['skate_origin_world_mid'][index,foot]
            ax.scatter(lower[:,0],lower[:,1],lower[:,2],s=1,c='gray',alpha=.45)
            ax.plot(*saved['blade_material_world_mid'][index,foot].T,color='cyan',linewidth=2)
            for j,(label,color) in enumerate([('F','red'),('L','green'),('U','blue')]):
                tip=O+.08*R[:,j];ax.plot(*np.stack([O,tip]).T,color=color);ax.text(*tip,label,fontsize=8)
            tip=O+np.array([0,0,.08]);ax.plot(*np.stack([O,tip]).T,color='gold');ax.text(*tip,'n',fontsize=8)
            ax.set(xlim=(centre[0]-span/2,centre[0]+span/2),ylim=(centre[1]-span/2,centre[1]+span/2),zlim=(centre[2]-span/2,centre[2]+span/2),title=f'{vid} foot{foot}',zlabel='raw world Z m')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=-90);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks(np.linspace(centre[2]-span/2,centre[2]+span/2,4));ax.tick_params(labelsize=7)
            metadata.append(dict(variant=vid,foot=foot,world_centre=centre,shared_cube_span_m=span,absolute_world_Z=True))
        fig.subplots_adjust(left=.01,right=.99,bottom=.08,top=.9,wspace=.05)
        buffer=io.BytesIO();fig.savefig(buffer,format='png');plt.close(fig);buffer.seek(0);panels.append(Image.open(buffer).convert('RGB').copy())
    return panels,metadata


def contour_geometry_match(curves,source):
    """Display-only closest-segment matching; no fixed material/contact claim."""
    curves=np.asarray(curves,float);points=np.asarray(source,float)
    starts=curves[:,:-1].reshape(-1,2);ends=curves[:,1:].reshape(-1,2);vectors=ends-starts
    alpha=np.clip(np.einsum('nki,ki->nk',points[:,None]-starts[None],vectors)/np.maximum(np.sum(vectors*vectors,axis=-1),1e-20)[None],0,1)
    candidates=starts[None]+alpha[...,None]*vectors[None]
    distance=np.linalg.norm(points[:,None]-candidates,axis=-1);choice=np.argmin(distance,axis=1);i=np.arange(len(points));segments=curves.shape[1]-1
    return dict(model_closest_uv=candidates[i,choice].tolist(),distance_px=distance[i,choice].tolist(),
        latent_runner_side=(choice//segments).tolist(),latent_runner_u=((choice%segments+alpha[i,choice])/segments).tolist(),
        material_correspondence='latent nearest projected segment; not observed fixed material',contact_truth='unmeasured')


def static_frames(loc):
    pts,defs,before=checked(loc);config=read(configuration(loc));asset_path=Path(config['blade_review_asset']);asset=read(asset_path);projection=load(asset_path.parent/asset['projection_npz'])
    from PIL import Image,ImageDraw,ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
    os.environ.setdefault('RINKSPACE_BODY_NATIVE_LIBRARY',str(loc['store'].data/'runs/body_perf_1103/native_cpu/rinkspace_core.dll'))
    cache=load(loc['sequence']/'source_roi_cache.npz');clock=read(loc['sequence']/'source_sequence.json')['clock'];reader=SourceVideo(before['source_video'],(1,12800))
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17);outputs=[]
    contour_doc=(dict(rows=source_runner_rows(asset)) if loc['family']=='source_structure' else
        read(loc['stage']/'support_and_free_shoe_regions.json').get('visible_runner_contours',{}) if loc['family']=='blade_image' else {})
    contour_rows=[row for row in contour_doc.get('rows',[]) if row.get('evidence_status')=='observed']
    for vid in (PARENT,CURRENT):
        defs[vid]['query']=SavedBladeProjection(loc['project'],defs[vid]['stage'],defs[vid]['parameter'])
        if 'blade_material_world_mid' not in defs[vid]['saved']:
            defs[vid]['saved']['blade_material_world_mid']=projection[vid+'_material_station_world_conditional_m']
    for seconds in KEY_SECONDS:
        pt=round(seconds*12800);index=int(np.searchsorted(pts,pt));assert pts[index]==pt
        rgb=reader.read(int(pt));rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest();assert rgb_sha==clock[index]['decoded_source_rgb_sha256']
        im=Image.fromarray(rgb);sheet=Image.new('RGB',(1920,1110),'white');draw=ImageDraw.Draw(sheet);bounds=cache['feet_bounds_native_xyxy'][index]
        observed=[row for row in contour_rows if int(row['pts'])==pt]
        titles=([('source structures / status retained',None),('14-step blade baseline / unaccepted',PARENT),('full source + conditional XY / unaccepted',CURRENT)] if loc['family']=='source_structure' else
            [('source steel observed',None),('39-step baseline / pose inferred',PARENT),('steel-source candidate / pose inferred',CURRENT)] if loc['family']=='blade_image' else [('clean source',None),('parent spin_safe',PARENT),('new unified whole sequence',CURRENT)])
        matches=[]
        for column,(title,vid) in enumerate(titles):
            panel=im.copy();d=ImageDraw.Draw(panel,'RGBA')
            if loc['family']=='source_structure':
                from rinkspace.viewer3d.blade_review import overlay_primitives
                current=dict(model=None,source_structures=[r for r in asset.get('source_structures',[])if int(r['pts'])==pt])
                for primitive in overlay_primitives(current,include_model_reference=False,include_model_contours=False,include_source_structures=True,include_source_legacy=False):
                    d.line([tuple(v)for v in primitive['points']],fill=tuple(round(v*255)for v in primitive['color']),width=max(1,round(primitive['width'])))
            for row in ([] if loc['family']=='source_structure' else observed):
                poly=[tuple(v) for v in row['polyline_uv']];radius=float(row['pixel_uncertainty_radius_px'])
                d.line(poly,fill=(25,220,65,70),width=max(1,round(2*radius)),joint='curve')
                d.line(poly,fill=(10,240,60,255),width=2)
                d.text((poly[0][0],poly[0][1]-24),f"SOURCE observed {row['role']} r={radius:g}px",font=font,fill=(10,240,60,255),stroke_width=1,stroke_fill='black')
            if vid:
                coco=defs[vid]['saved']['COCO_uv'][index]
                for a,b in [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:d.line([tuple(coco[a]),tuple(coco[b])],fill='orange',width=2)
                query=defs[vid]['query'];local=query.binding.packet()['adapter']
                for foot,color in ((0,'orange'),(1,'cyan')):
                    if loc['family']!='source_structure':
                        boot=query.project_points(np.array([pt]),foot,local)['uv'][0]
                        for ids in (list(range(7))+[0],list(range(7,14))+[7]):d.line([tuple(v) for v in boot[ids]],fill=color,width=2)
                        blade=projection[vid+'_blade_uv'][index,foot];d.line([tuple(v) for v in blade],fill=color,width=3)
                        for uv,label in zip(blade,['R','M','F']):d.text(tuple(uv),f'{foot}{label}',font=font,fill=color,stroke_width=1,stroke_fill='black')
                    if loc['family'] in ('blade_image','source_structure'):
                        edges=projection[vid+'_runner_edges_uv'][index,foot]
                        for edge in edges:d.line([tuple(v) for v in edge],fill=(225,20,235,255),width=3)
                        upper_key=vid+'_runner_upper_edges_uv'
                        if upper_key in projection:
                            for edge in projection[upper_key][index,foot]:d.line([tuple(v)for v in edge],fill=(255,115,190,255),width=2)
                for row in observed:
                    if row.get('foot')not in (0,1):continue
                    match=contour_geometry_match(projection[vid+'_runner_edges_uv'][index,int(row['foot'])],row['polyline_uv'])
                    match.update(source_row_id=row.get('id'),source_role=row['role'],source_evidence_status='observed',model_pose_status='inferred',
                        variant=vid,foot=int(row['foot']),pixel_uncertainty_radius_px=row['pixel_uncertainty_radius_px'],
                        outside_visible_band_px=np.maximum(np.asarray(match['distance_px'])-float(row['pixel_uncertainty_radius_px']),0).tolist())
                    matches.append(match)
            sheet.paste(panel.resize((640,360)),(column*640,34));draw.text((column*640+8,7),f'{title} {seconds:.2f}s PTS{pt}',font=font,fill='black')
            for foot in (0,1):
                crop=panel.crop(tuple(map(int,bounds[foot])));sheet.paste(fit_image(crop,(310,260)),(column*640+5+foot*320,421))
                draw.text((column*640+7+foot*320,397),f'source navigation ROI{foot}',font=font,fill='black')
            draw.text((column*640+8,691),'GREEN source observed; MAGENTA actual runner' if loc['family'] in ('blade_image','source_structure') else 'Candidate axes/modes; source contact unknown',font=font,fill='black')
        sides,coordinates=side_panels(defs,index)
        for column,panel in enumerate(sides,1):sheet.paste(panel,(column*640,750))
        notes=['Same native exposure; same source crops.','GREEN: observed steel subcontour + radius.','MAGENTA: actual native5-node curved edges.','Pose and closest side/u: inferred geometry.','Source role/radius retained per exactPTS.','No fixed material ID or measured contact.'] if loc['family'] in ('blade_image','source_structure') else ['Same native exposure; same source crops.','Orange/cyan: saved model projections.','F/L/U and n: model and ice normal.','u/minimum/mode: MODEL diagnostics.','Source missing/unknown stay distinct.','No measured contact/edge/steel endpoint.']
        for j,note in enumerate(notes):draw.text((10,768+j*32),note,font=font,fill='black')
        target=loc['out']/f'same_frame_{pt}.png';sheet.save(target)
        outputs.append(dict(path=str(target),sha256=sha(target),pts=pt,source_seconds=seconds,source_rgb_sha256=rgb_sha,source_navigation_bounds_native_xyxy=bounds,side_coordinates=coordinates,source_observed_runner_contours=observed,latent_geometry_matches=matches,new_source_measurements=0))
    reader.close();write(loc['out']/'static_frames.json',dict(family=loc['family'],frames=outputs,exact_pts=True,source_contour_status='observed where explicitly provided; no fixed material identity',model_pose_and_geometry_correspondence='inferred',fullsize_source_pngs_written=0,video_generated=False));return outputs


def commands(loc):
    python=loc['store'].data/'runtime_tools/vision_cuda/Scripts/python.exe'
    prefix=f'& "{python}" -B -m calibration.program_tracking.body_tracking.sequence_delivery'
    args=f'--project "{loc["project"]}" --data-root "{loc["store"].data}" --family {loc["family"]}'
    if loc['family']=='source_structure':
        args+=f' --stage "{loc["stage"]}" --parameter "{loc["parameter"]}" --parent-stage "{loc["parent_stage"]}" --parent-parameter "{loc["parent_parameter"]}"'
        if loc.get('source_asset'):args+=f' --source-asset "{loc["source_asset"]}"'
    print(prefix+' check '+args);print(prefix+' all '+args+' --publish-config')
    if loc['family']=='source_structure':print(prefix+' prepare '+args)
    print('Explicit GUI entry after preparation: '+str(loc['project']/'launchers'/loc['launcher']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['commands','check','register','prepare','static','all'])
    p.add_argument('--project',required=True);p.add_argument('--data-root',required=True);p.add_argument('--stage');p.add_argument('--parameter');p.add_argument('--family',choices=['global','blade_image','source_structure'],default='global');p.add_argument('--parent-stage');p.add_argument('--parent-parameter');p.add_argument('--source-asset',help='Read-only source review asset to pass through into the new candidate projection asset')
    p.add_argument('--publish-config',action='store_true');p.add_argument('--include-legacy',action='store_true');a=p.parse_args();loc=locations(a)
    if a.action=='commands':commands(loc);return
    if a.action=='check':
        pts,defs,before=checked(loc,a.include_legacy);report=dict(real_saved_candidates=list(defs),frames=len(pts),no_writes=True)
        if loc['family']=='source_structure':report['retained_existing_variants']=[v['id'] for v in retained_variants(loc,before)]
        print(json.dumps(report));return
    if a.action in ('register','all'):register(loc,a.publish_config,a.include_legacy)
    if a.action in ('prepare','all'):print(json.dumps(dict(display_caches=prepare_caches(loc,a.include_legacy))),flush=True)
    if a.action in ('static','all'):print(json.dumps(dict(static_frames=len(static_frames(loc)))),flush=True)


if __name__=='__main__':main()
