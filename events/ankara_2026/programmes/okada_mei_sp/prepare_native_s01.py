"""Data-only d78 S01 migration onto the explicitly selected current native state.

Source pixels, clock, original whitening, approvals and historical state are immutable.
This programme adapter records every association and loss of native correspondence.
"""
from pathlib import Path
import json,hashlib,re,collections
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.domain.print_assets import resolve_instance

ROOT=Path(__file__).resolve().parents[4]
DATA=Path(r"D:\CodexData\RinkSpace_Data")
RUN=DATA/"runs/native_joint_s01_20261026"
PREPARED=DATA/"runs/rebuild_a_20261024/prepared_s01_v2"
OLD=DATA/"runs/rebuild_a_20261024/imported_d78"
SCENE=ROOT/"events/ankara_2026/scene/field/model.json"
CAMERA=Path(__file__).parent/"ad_cameras/s01.json"
LIBRARY=DATA/"build/native_unclipped_20261027/rinkspace_core.dll"
# Identity comes from the reviewed packets' measurement_operator and the
# two measured paint/ice shoulders, not the historical display suffix outer.
C0_SOURCE_ID='d78_material_144/path0'
C0_MEASUREMENT='same_reviewed_paint_width_centreline_original_packet_paint_center'
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

POINT_FIELDS={'source_rows','path_indices','stations','pts','uv','kind','observation_ids'}

def camera_for_snapshot(camera,snapshot):
    """Use an explicitly saved numerical grid, preserving the source PTS origin."""
    camera=dict(camera)
    for saved,key in [('motion_grid_first_control_time','first_control_time'),('motion_grid_control_dt','control_dt')]:
        if saved in snapshot:
            value=float(snapshot[saved])
            if not np.isfinite(value) or (key=='control_dt' and value<=0):raise ValueError('Invalid saved native motion grid')
            camera[key]=value
    for saved,key in [('fit_first_pts','first_pts'),('fit_stop_pts_exclusive','stop_pts_exclusive')]:
        if saved in snapshot:camera[key]=int(snapshot[saved])
    return camera

def selected_s01_exposures():
    scope=read(Path(__file__).with_name('rebuild_scope.json'))
    shot=next(s for s in scope['shots'] if s['id']=='s01')['processing']
    return np.arange(shot['first_pts'],shot['stop_pts_exclusive'],scope['source_clock']['native_pts_step'],dtype=np.int64)

def load_pixel_input(directory):
    """Only the current point schema is numerical input; no historical statistics."""
    directory=Path(directory)
    with np.load(directory/'point_observations.npz',allow_pickle=False) as source:
        if set(source.files)!=POINT_FIELDS:raise ValueError('Point input must contain exactly the seven raw-point fields')
        z={key:source[key] for key in source.files}
    n=len(z['pts'])
    if n==0:raise ValueError('Empty point input')
    for key,value in z.items():
        if value.shape!=((n,2) if key=='uv' else (n,)):raise ValueError('Point field shape: '+key)
    for key in ('source_rows','path_indices','pts','kind'):
        if z[key].dtype.kind not in 'iu':raise ValueError('Integer point field required: '+key)
    if not np.isin(z['kind'],[0,1]).all():raise ValueError('Unknown fixed/curve point kind')
    if z['observation_ids'].dtype.kind not in 'SU':raise ValueError('Stable observation IDs must be strings')
    ids=z['observation_ids'].astype(str)
    if np.any(ids=='') or len(np.unique(ids))!=n:raise ValueError('Missing or duplicate stable observation ID')
    if not np.isfinite(z['uv']).all() or not np.isfinite(z['stations']).all():raise ValueError('Nonfinite original point')
    mapping=read(directory/'mapping.json')
    if np.any(z['path_indices']<0) or np.any(z['path_indices']>=len(mapping['path_ids'])):raise ValueError('Native path index outside mapping')
    with np.load(directory/'initialized_points.npz',allow_pickle=False) as seed:
        if set(seed.files)!={'stations'}:raise ValueError('Point initializer may contain only numerical stations')
        stations=seed['stations']
    if stations.shape!=(n,) or not np.isfinite(stations).all():raise ValueError('Invalid station initializer')
    fixed=z['kind']==0
    if not np.array_equal(stations[fixed],z['stations'][fixed]):raise ValueError('Fixed material station changed by initializer')
    return z,mapping,stations
def array(root,ref):
    p=Path(root)/ref["path"]
    if digest(p)!=ref["sha256"]:raise ValueError("Source identity changed: "+str(p))
    a=np.load(p,allow_pickle=False)
    if list(a.shape)!=ref["shape"] or a.dtype.str!=ref["dtype"]:raise ValueError("Source layout changed")
    return a

def closest_polyline(points,vertices,knots):
    """Preparation-only material association on exact native compiled SZ segments."""
    vertices=np.asarray(vertices,float).reshape(-1,2);knots=np.asarray(knots,float)
    if len(vertices)==1:return np.zeros(len(points)),np.linalg.norm(points-vertices[0],axis=1)
    left=vertices[:-1];delta=np.diff(vertices,axis=0);norm=np.sum(delta*delta,axis=1)
    stations=np.empty(len(points));distance=np.empty(len(points))
    for begin in range(0,len(points),256):
        p=points[begin:begin+256];offset=p[:,None,:]-left[None,:,:]
        t=np.clip(np.einsum("ijk,jk->ij",offset,delta)/np.maximum(norm,1e-300),0.,1.)
        d2=np.sum((offset-t[:,:,None]*delta)**2,axis=2);j=np.argmin(d2,axis=1);i=np.arange(len(p))
        stations[begin:begin+len(p)]=knots[j]+t[i,j]*(knots[j+1]-knots[j]);distance[begin:begin+len(p)]=np.sqrt(d2[i,j])
    return stations,distance

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    core=NativeCore(LIBRARY);scene=read(SCENE);model=SceneParameterModel(scene,SCENE,optimize_scene=True)
    state=model.initial_state;native=NativeSceneGPU(SCENE,core,parameter_state=state)
    camera=read(CAMERA);controls=np.load(CAMERA.parent/camera["controls_ref"],allow_pickle=False)
    source=read(PREPARED/"manifest.json");a={k:array(PREPARED,v) for k,v in source["arrays"].items()}
    features=source["features"];old={p["id"]:p for p in read(OLD/"scene.json")["paths"]}
    old_station=np.load(OLD/"station_f64.npy");original_ids=a["observation_ids"].astype(str);lookup={key:i for i,key in enumerate(original_ids)}
    count=len(original_ids);mapped=np.full(count,"",dtype="U150");station=np.zeros(count);reason=np.full(count,"",dtype="U120")
    candidates={};banks={};evidence=[];assets={};material_paths={};aliases=collections.defaultdict(list);old_feature_candidates=collections.defaultdict(set)
    for ref,asset in read_scene_assets(scene,SCENE):
        n=ref["advertisement_number"];assets[n]=asset;parent=(SCENE.parent/ref["path"]).resolve().parent
        template=next(t for t in asset["templates"] if t["id"]==ref["template_id"])
        for path in template["paths"]:
            if "material_uv" in path:material_paths[f"print_instance/Ad{n}::"+path["id"]]=np.asarray(path["material_uv"],float)
            for label in [path["id"],*path.get("logical_labels",[]),path.get("historical_material_id"),path.get("historical_curve_id")]:
                if label:aliases[label].append(f"print_instance/Ad{n}::"+path["id"])
        store=asset.get("observation_store")
        if n not in (3,7,8,9,10) or store is None:continue
        bank_path=parent/store["arrays"]["path"];correspondence_path=parent/store["source_correspondence"]["path"]
        for file,ref2 in ((bank_path,store["arrays"]),(correspondence_path,store["source_correspondence"])):
            if digest(file)!=ref2["sha256"]:raise ValueError("Association evidence changed: "+str(file))
        bank=np.load(bank_path,allow_pickle=False);correspondence=read(correspondence_path);banks[n]=(bank,correspondence)
        evidence.append(dict(advertisement=n,path=str(correspondence_path),sha256=digest(correspondence_path)))
        def assign(key,pts,uv,paths):
            if key not in lookup:return
            row=lookup[key]
            if int(a["pts"][row])!=int(pts) or not np.array_equal(a["uv"][row],uv):raise ValueError("Association modifies immutable source UV/PTS")
            ids=[f"print_instance/Ad{n}::"+x for x in paths if x and f"print_instance/Ad{n}::"+x in native.paths]
            candidates[row]=ids
            if ids:old_feature_candidates[features[a["feature_index"][row]]["id"]].update(ids)
        if n==7:
            for r in correspondence["rows"]:assign(f"raw:12:s01_original:{r['native_pts']}:{r['source_point_id']}",r['native_pts'],r['source_uv'],r['candidate_feature_ids'])
        elif n==3:
            registry=store['curve_registry']
            for i,key in enumerate(bank['stable_id'].astype(str)):
                record=registry[int(bank['curve_index'][i])]
                assign(key,bank['pts'][i],bank['uv'][i],[record.get('template_path_id')])
        else:
            for i,key in enumerate(bank['stable_id'].astype(str)):
                assign(key,bank['pts'][i],bank['uv'][i],[str(bank['current_feature_id'][i])])
            if n==8:
                for record in correspondence.get('remain_excluded',[]):
                    key=record['stable_id']
                    if key in lookup:candidates[lookup[key]]=[]
    # These narrow source contours have unambiguous glyph identities. Their
    # material-chart translation is used ONLY for crosswalk association; it
    # never changes a native template, camera, UV, covariance or observation.
    ad4_glyph_anchors={'current_component006':'JUNIOR_J','current_component009':'JUNIOR_I',
        'current_component011_inner':'JUNIOR_O_inner','current_component013_inner':'JUNIOR_R_inner'}
    anchor_delta=[]
    for f in features:
        suffix=f['label'].removeprefix('Ad4.print.')
        if suffix in ad4_glyph_anchors:
            raw=1.-np.asarray(old[f['id']]['original_definition']['points'],float)
            current=material_paths['print_instance/Ad4::Ad4.print.shot4_'+ad4_glyph_anchors[suffix]+'/path0']
            anchor_delta.append((current.min(0)+current.max(0)-raw.min(0)-raw.max(0))/2)
    ad4_text_translation=np.median(anchor_delta,axis=0)
    source_sz=np.full((count,2),np.nan);source_material_uv=np.full((count,2),np.nan);feature_records=[]
    pathorder=list(native.paths);path_index={p:i for i,p in enumerate(pathorder)}
    association_margin=np.full(count,np.nan);association_distance=np.full(count,np.nan)
    for fi,f in enumerate(features):
        rows=np.flatnonzero(a['feature_index']==fi)
        if not len(rows):continue
        identity=f['id'];p=old[identity];label=f['label'];pool=[];method=''
        if label.startswith('ice.'):
            if identity in native.paths:
                mapped[rows]=identity;station[rows]=old_station[rows];reason[rows]='same_confirmed_physical_ice_curve'
                if identity==C0_SOURCE_ID:reason[rows]=C0_MEASUREMENT
            else:reason[rows]='no_current_native_ice_entity'
            feature_records.append(dict(id=identity,label=label,rows=len(rows),method=str(reason[rows[0]])));continue
        if label.startswith('board.'):
            target='board/continuous/'+('top' if '.top.' in label else 'bottom')
            if target not in native.paths:raise ValueError('Missing confirmed continuous board '+target)
            segment=int(p['parameters'][0]);values=np.asarray(core.board_stations_from_legacy(scene['board'],segment,old_station[rows]),float)
            mapped[rows]=target;station[rows]=values;reason[rows]='same_continuous_board_with_legacy_chart_converted'
            feature_records.append(dict(id=identity,label=label,rows=len(rows),method=str(reason[rows[0]])));continue
        n=int(re.match(r'Ad(\d+)',label).group(1));vertices=np.asarray(p['vertices'],float)
        old_knots=np.linspace(0.,1.,len(vertices));s=old_station[rows]
        if len(vertices)==1:xy=np.repeat(vertices,len(rows),axis=0)
        else:xy=np.c_[np.interp(s,old_knots,vertices[:,0]),np.interp(s,old_knots,vertices[:,1])]
        source_sz[rows]=xy
        raw=np.asarray(p['original_definition']['points'],float)
        raw_knots=np.linspace(0.,1.,len(raw))
        source_material_uv[rows]=1.-np.c_[np.interp(s,raw_knots,raw[:,0]),np.interp(s,raw_knots,raw[:,1])]
        if f['kind']=='material_point':
            # Preserve the actual material corner: source chart identifies which
            # endpoint in the current, orientation-corrected panel it refers to.
            edge=label.split('.')[0]+('.top_edge' if label.endswith('_top') else '.bottom_edge')
            pool=aliases[edge];method='same_material_corner_current_panel_endpoint'
            if not pool:raise ValueError('Missing panel edge for confirmed corner '+label)
            pts=[];targets=[]
            for path_id in pool:
                v=np.asarray(native.paths[path_id]['vertices']).reshape(-1,2)
                for endpoint in (0,1):pts.append(v[0 if endpoint==0 else -1]);targets.append((path_id,float(endpoint)))
            j=int(np.argmin(np.linalg.norm(np.asarray(pts)-xy[0],axis=1)));target,value=targets[j]
            mapped[rows]=target;station[rows]=value;reason[rows]=method
            feature_records.append(dict(id=identity,label=label,rows=len(rows),method=method,target=target,station=value));continue
        pool=list(dict.fromkeys(aliases[label]+aliases[identity]));method='same_named_material_in_current_native'
        if not pool and n==10:pool=sorted(old_feature_candidates[identity]);method='source_bank_confirmed_named_component_crosswalk'
        if n==11:pool=['print_instance/Ad11::Ad11.outer_ring.outer'];method='same_outermost_print_ring'
        if n==4 and '.print.' in label:
            suffix=label.split('.print.')[1]
            if suffix=='current_component005':pool=[];method='central_skater_graphic_explicitly_absent_from_current_model'
            elif suffix.startswith('word_ANKARA') or suffix in ('current_component000','current_component004_inner'):
                pool=[k for k in pathorder if k.startswith('print_instance/Ad4::Ad4.ANKARA.')];method='same_ANKARA_material_union_native_split_contours'
                if suffix.endswith('_inner'):pool=[k for k in pool if k.endswith(('boundary_1','boundary_2','boundary_3','boundary_5'))]
            elif suffix in ad4_glyph_anchors:
                pool=['print_instance/Ad4::Ad4.print.shot4_'+ad4_glyph_anchors[suffix]+'/path0'];method='same_identified_JUNIOR_glyph_material'
            else:
                word='GRAND_' if suffix=='current_tail000' else 'PRIX_' if suffix=='current_tail002' else ''
                pool=[k for k in pathorder if k.startswith('print_instance/Ad4::Ad4.print.shot4_'+word)];method='same_JUNIOR_GRAND_PRIX_material_native_split_contours'
                if suffix.endswith('_inner'):pool=[k for k in pool if '_inner' in k]
                else:pool=[k for k in pool if '_inner' not in k]
        if n==3 and '.print.' in label:
            pool=sorted(old_feature_candidates[identity]);method='existing_source_material_crosswalk_for_current_official_artwork'
            if label.endswith(('component006','component007')):pool=[];method='old_DAIKICHI_characters_explicitly_removed_from_current_model'
        if n in (7,8,9) and '.print.' in label:
            pool=sorted(old_feature_candidates[identity]);method='existing_source_material_crosswalk'
        # A stable source row correspondence takes precedence over a feature-wide fallback.
        sets=collections.defaultdict(list)
        for row in rows:
            chosen=candidates.get(int(row),pool)
            if n==3 and not chosen and pool:chosen=pool
            if n==8 and not chosen:reason[row]='source_false_W_Y_bridge_without_native_material';continue
            if not chosen:
                reason[row]='source_decoration_or_script_not_in_current_native_model' if n==9 else method if not pool else 'source_correspondence_has_no_current_material'
                continue
            sets[tuple(sorted(chosen))].append(int(row))
        for chosen,rr in sets.items():
            rr=np.asarray(rr,int);ds=[];ss=[]
            for target in chosen:
                q=native.paths[target]
                if n in (4,6) and '.print.' in label:
                    material=source_material_uv[rr].copy()
                    if n==4 and 'ANKARA' not in target:material+=ad4_text_translation
                    v=material_paths[target]
                    top=native.paths[f'print_instance/Ad{n}::Ad{n}.top_edge/path0']['vertices']
                    bottom=native.paths[f'print_instance/Ad{n}::Ad{n}.bottom_edge/path0']['vertices']
                    metric=np.array([np.ptp(np.asarray(top)[:,0]),abs(np.mean(np.asarray(top)[:,1])-np.mean(np.asarray(bottom)[:,1]))])
                    st,dist=closest_polyline(material*metric,v*metric,np.linspace(0.,1.,len(v)))
                else:
                    st,dist=closest_polyline(source_sz[rr],q['vertices'],q.get('knots') or np.linspace(0.,1.,len(q['vertices'])))
                ss.append(st);ds.append(dist)
            ds=np.asarray(ds).T;ss=np.asarray(ss).T;j=np.argmin(ds,axis=1);ii=np.arange(len(rr))
            mapped[rr]=np.asarray(chosen)[j];station[rr]=ss[ii,j];reason[rr]=method
            association_distance[rr]=ds[ii,j]
            if len(chosen)>1:association_margin[rr]=np.partition(ds,1,axis=1)[:,1]-ds[ii,j]
        feature_records.append(dict(id=identity,label=label,rows=len(rows),method=method,
            targets=dict(collections.Counter(mapped[rows].tolist())),unmapped=int(np.count_nonzero(mapped[rows]==''))))
    reason[a['active']==0]='original_source_exclusion_preserved'
    selected=np.flatnonzero((mapped!='')&(a['active']!=0));rowmap=np.full(count,-1,np.int32);rowmap[selected]=np.arange(len(selected))
    # Marginalize only genuinely absent source coordinates. Keep whole W when no rows leave.
    group_members=a['noise_group_members'];group_weights=a['noise_group_weights'];offsets=[0];woffsets=[0];members=[];weights=[];ranks=[];group_records=[]
    for g in source['noise_groups']:
        rr=group_members[g['member_start']:g['member_start']+g['member_count']].astype(int);keep=rowmap[rr]>=0
        if not np.any(keep):continue
        w=group_weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)].reshape(g['rank'],2*len(rr))
        if hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()!=g['matrix_sha256']:raise ValueError('Original W identity changed')
        new=w;discard_rank=0
        if not np.all(keep):
            columns=np.repeat(keep,2);wm=w[:,~columns]
            u,sv,_=np.linalg.svd(wm,full_matrices=True);tolerance=(sv[0] if len(sv) else 0.)*max(wm.shape)*np.finfo(float).eps
            discard_rank=int(np.count_nonzero(sv>tolerance));new=u[:,discard_rank:].T@w[:,columns]
        members.extend(rowmap[rr[keep]].tolist());weights.extend(new.ravel().tolist());ranks.append(new.shape[0]);offsets.append(len(members));woffsets.append(len(weights))
        group_records.append(dict(id=g['id'],source_matrix_sha256=g['matrix_sha256'],rank=g['rank'],retained_rank=new.shape[0],
            original_member_rows=rr.tolist(),retained_member_rows=rr[keep].tolist(),marginalized_source_rank=discard_rank,
            transformation='unchanged_W' if np.all(keep) else 'left_nullspace_of_missing_columns_exact_gaussian_marginal'))
    for row in selected[a['noise_group_index'][selected]<0]:
        xx,xy,yy=a['covariance'][row];w=np.linalg.solve(np.linalg.cholesky([[xx,xy],[xy,yy]]),np.eye(2))
        members.append(int(rowmap[row]));weights.extend(w.ravel());ranks.append(2);offsets.append(len(members));woffsets.append(len(weights))
    if sorted(members)!=list(range(len(selected))):raise ValueError('Incomplete original noise partition')
    q=read(PREPARED/'queries.json');queries={k:array(PREPARED,v) for k,v in q['arrays'].items()};query_source=queries['source_rows'].astype(int)
    if not np.all(rowmap[query_source]>=0):raise ValueError('A source profile lacks current physical identity')
    query_ids=np.full(len(selected),-1,np.int32);query_ids[rowmap[query_source]]=np.arange(len(query_source))
    initial_alpha=np.load(OLD/'initial_alpha_f64.npy')[query_source]
    np.savez(RUN/'source_binding.npz',source_rows=selected,path_indices=np.array([path_index[k] for k in mapped[selected]],np.int32),
        stations=station[selected],pts=a['pts'][selected],uv=a['uv'][selected],kind=a['kind'][selected],legacy_kind=a['legacy_kind'][selected],
        original_local_ids=a['local_index'][selected],observation_ids=a['observation_ids'][selected],covariance=a['covariance'][selected],
        group_offsets=np.asarray(offsets,np.int32),weight_offsets=np.asarray(woffsets,np.int32),members=np.asarray(members,np.int32),
        ranks=np.asarray(ranks,np.int32),weights=np.asarray(weights),query_ids=query_ids,query_numerator_u=queries['numerator_u'],
        query_numerator_v=queries['numerator_v'],query_denominator=queries['denominator'],query_bounds=queries['bounds'],query_initial_alpha=initial_alpha,
        engineering_pixel_budget=a['engineering_pixel_budget'][selected])
    np.savez(RUN/'mapping_audit.npz',path_ids=mapped,stations=station,reason=reason,source_sz=source_sz,source_material_uv=source_material_uv,
        association_distance=association_distance,association_margin=association_margin)
    write(RUN/'mapping.json',dict(schema='rinkspace.d78_current_native_mapping.v1',source=str(PREPARED),source_manifest_sha256=digest(PREPARED/'manifest.json'),
        scene_path=str(SCENE),scene_revision=native.scene_revision,camera_path=str(CAMERA),camera_sha256=digest(CAMERA),
        control_sha256=digest(CAMERA.parent/camera['controls_ref']),native_parameters_replaced_by_dataset=False,
        path_ids=pathorder,features=feature_records,evidence=evidence,source_rows=count,selected_rows=len(selected),
        frames=len(np.unique(a['pts'][selected])),excluded_reasons=dict(collections.Counter(reason[~np.isin(np.arange(count),selected)].tolist())),
        correlation_groups=group_records,formal_rank=int(sum(ranks)),profile_rows=len(query_source),
        ad4_text_association_translation=ad4_text_translation.tolist(),material_identity_uses_unfitted_template_for_ads=[4,6],fit_started=False,visual_mapping_review='pending',source_association_numbers_are_not_new_human_approvals=True))
    print(json.dumps(dict(selected=len(selected),frames=len(np.unique(a['pts'][selected])),formal_rank=sum(ranks),
        excluded=dict(collections.Counter(reason[~np.isin(np.arange(count),selected)].tolist())))),flush=True)
    return native,state,camera,controls

def prepare_pixel_observations():
    """Current point-only input: restore C0 without replaying other associations.

    Historical W/sigma/query/budget data stays in its original archive. None
    is copied into this input or allowed to replace current native parameters.
    """
    output=RUN/'point_observations.npz'
    if output.exists():
        raise ValueError('Point input already prepared; do not replay unchanged preparation')
    before={str(p):digest(p) for p in (SCENE,CAMERA,CAMERA.parent/read(CAMERA)['controls_ref'])}
    source=read(PREPARED/'manifest.json')
    a={k:array(PREPARED,source['arrays'][k]) for k in ('feature_index','active','pts','uv','kind','observation_ids')}
    old=dict(np.load(RUN/'source_binding.npz',allow_pickle=False))
    mapping=read(RUN/'mapping.json');audit=dict(np.load(RUN/'mapping_audit.npz',allow_pickle=False))
    fi=next(i for i,f in enumerate(source['features']) if f['id']==C0_SOURCE_ID)
    c0=np.flatnonzero((a['feature_index']==fi)&(a['active']!=0))
    evidence=read(RUN/'c0_packet_semantics_audit.json')
    if len(c0)!=3018 or evidence['operators']!={'paint_center':198} or evidence['uv_mismatch']:
        raise ValueError('C0 source measurement evidence differs from inspected packets')
    if np.intersect1d(old['source_rows'],c0).size:raise ValueError('C0 already present')
    if C0_SOURCE_ID not in mapping['path_ids']:raise ValueError('Current native C0 missing')
    selected=np.sort(np.r_[old['source_rows'],c0]);kept=np.searchsorted(selected,old['source_rows']);added=np.searchsorted(selected,c0)
    paths=np.empty(len(selected),np.int32);paths[kept]=old['path_indices'];paths[added]=mapping['path_ids'].index(C0_SOURCE_ID)
    stations=np.empty(len(selected));stations[kept]=old['stations'];stations[added]=np.load(OLD/'station_f64.npy',allow_pickle=False)[c0]
    points=dict(source_rows=selected,path_indices=paths,stations=stations,pts=a['pts'][selected],uv=a['uv'][selected],kind=a['kind'][selected],observation_ids=a['observation_ids'][selected])
    for key in ('uv','pts','kind','observation_ids','path_indices','stations'):
        if not np.array_equal(points[key][kept],old[key]):raise ValueError('Unchanged input modified: '+key)
    initial=dict(np.load(RUN/'initialized_stations.npz',allow_pickle=False))
    seeded=stations.copy();seeded[kept]=initial['stations']
    distance=np.full(len(selected),np.nan);distance[kept]=initial['seed_screen_distance']
    audit['path_ids'][c0]=C0_SOURCE_ID;audit['reason'][c0]=C0_MEASUREMENT;audit['stations'][c0]=stations[added]
    mapping['selected_rows']=len(selected)
    for f in mapping['features']:
        if f['id']==C0_SOURCE_ID:f['method']=C0_MEASUREMENT;f['unmapped']=0
    mapping['excluded_reasons'].pop('source_outer_paint_boundary_has_no_current_centreline_equivalent',None)
    mapping['historical_statistics_archive']=str(RUN/'source_binding.npz')
    for key in ('correlation_groups','formal_rank','profile_rows'):mapping.pop(key,None)
    mapping['current_point_input']=str(output)
    mapping['objective']='sum of squared original source pixel distances, unweighted, no robust loss, no error allowance'
    mapping['C0_measurement_evidence']=dict(path=str(RUN/'c0_packet_semantics_audit.json'),operator='paint_center',restored_rows=len(c0),frames=198,
        current_user_clarification='Reviewed as width centreline; outer was a historical display label',
        unresolved_packet_point_id_joins=len(evidence['missing']),unresolved_ids_do_not_remove_reviewed_source_points=True)
    mapping['native_parameters_replaced_by_dataset']=False
    np.savez(output,**points)
    np.savez(RUN/'initialized_points.npz',stations=seeded,seed_screen_distance=distance)
    for name in ('mapping.json','mapping_audit.npz'):
        p=RUN/name;backup=p.with_name(p.stem+'_before_c0'+p.suffix)
        if backup.exists():raise ValueError('Historical backup already exists: '+str(backup))
        p.replace(backup)
    write(RUN/'mapping.json',mapping);np.savez(RUN/'mapping_audit.npz',**audit)
    if any(digest(Path(p))!=sha for p,sha in before.items()):raise ValueError('Native parameter authority changed')
    receipt=dict(rows=len(selected),frames=len(np.unique(points['pts'])),restored_C0_rows=len(c0),restored_C0_frames=198,
        input=str(output),sha256=digest(output),fields=list(points),original_uv_and_pts_unchanged=True,other_associations_unchanged=True,
        native_parameter_hashes_unchanged=before,weights_used=False,covariance_used=False,robust_loss=False,error_allowance=False,legacy_query_constraints=False)
    write(RUN/'point_input.json',receipt);print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pixels',action='store_true');args=parser.parse_args()
    prepare_pixel_observations()
