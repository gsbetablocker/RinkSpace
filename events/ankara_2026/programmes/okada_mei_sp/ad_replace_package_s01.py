"""Publish and merge immutable S01 ad observations into the current native input."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import collections, hashlib, importlib.util, json, re
import numpy as np

OUT=b.OUT
ADS=OUT/'ads_only'
BUNDLE=OUT/'native_bundle'


def authority_check():
    d=b.read(OUT/'authority_inventory.json')
    for r in [d['scene'],*d['assets']]:
        if b.sha(r['path'])!=r['sha256']:raise ValueError('Native model changed since source review: '+r['path'])
    return d


def current_native():
    sys.path.insert(0,str(b.ROOT/'src'))
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.parameters import SceneParameterModel
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    core=NativeCore(b.DATA/'build/native_joint_20261026/rinkspace_core.dll')
    model=SceneParameterModel(b.read(b.SCENE),b.SCENE,optimize_scene=True)
    native=NativeSceneGPU(b.SCENE,core,parameter_state=model.initial_state)
    return core,model,native


def corner_binding(row,match,data,aa,core,model):
    """Use the named material corner's existing native parameter position."""
    from rinkspace.domain.print_assets import _print_context,_print_seed
    n=int(data['ad'][row]);label=str(data['labels'][row]);candidate=match['path_id'];corner=label.split('.')[-1]
    if n==7 and '.row' in label:
        entity=label.split('.')[1];target='Ad7.'+('top_' if entity.startswith('row1_') else 'bottom_')+entity.split('_',1)[1]+'_part00'
        if candidate!=target:return None
        xside='left';yside='top' if corner=='top_left' else 'bottom'
    else:
        parts=corner.split('_')
        if len(parts)!=2 or parts[0] not in ('left','right') or parts[1] not in ('top','bottom'):return None
        xside,yside=parts;targets=aa[n]['aliases'].get(f'Ad{n}.{yside}_edge',[])
        if len(targets)!=1:return None
        target=targets[0]
        boundary=f'Ad{n}.'+('left_vertical' if xside=='left' else 'right_vertical')
        permitted={target,*aa[n]['aliases'].get(boundary,[])}
        if n==5:permitted.update(aa[n]['aliases'].get('Ad5.print.native3976_'+('blue_bar' if yside=='top' else 'green_bar'),[]))
        if candidate not in permitted:return None
    context=_print_context(aa[n]['asset'],f'print_instance/Ad{n}',model.initial_state)
    seed=np.asarray(_print_seed(aa[n]['paths'][target],context,core),float)
    x=seed[:,0];y=seed[:,1];xt=x.min() if xside=='left' else x.max()
    # A specific letter corner or rectangle endpoint, not a nearest projected
    # point on an unfitted native curve.
    ix=np.flatnonzero(np.isclose(x,xt,atol=1e-9,rtol=0))
    j=int(ix[np.argmin(y[ix]) if yside=='top' else np.argmax(y[ix])])
    return target,j/(len(seed)-1),dict(corner=corner,native_seed_vertex_index=j,native_seed_vertex_count=len(seed),semantic_selection='named_left_right_top_bottom_corner_in_current_native_material_chart')


def publish():
    auth=authority_check();data=b.load();aa=b.assets();core,model,native=current_native()
    records={};exclusions={};evidence_files=[];counts=collections.defaultdict(collections.Counter)
    for p in sorted((OUT/'verified_color_final').glob('*.json')):
        d=b.read(p)
        if d['version']!='source_entity_color_word_v4':raise ValueError('Incomplete physical color and whole-word review')
        evidence_files.append(dict(path=str(p),sha256=b.sha(p)))
        for r in d['confirmed']:
            row=r['row']
            if not r.get('physical_color_verified') or not r.get('directional_word_clarity_verified'):raise ValueError('Local grayscale evidence alone cannot admit a point')
            if row in records or row in exclusions:raise ValueError('Duplicate review row')
            records[row]=dict(r,evidence_path=str(p),source_pts=d['pts'])
        for r in d['excluded']:
            row=r['row']
            if row in records or row in exclusions:raise ValueError('Duplicate review row')
            exclusions[row]=dict(r,evidence_path=str(p))
    all_ad=set(map(int,np.flatnonzero(data['ad']>0)))
    if all_ad!=set(records)|set(exclusions):raise ValueError('Every latest ad source row needs a final decision')
    stations={};corner_details={}
    for row,r in list(records.items()):
        if str(data['ids'][row])!=r['stable_id'] or int(data['pts'][row])!=r['source_pts']:raise ValueError('Review source identity changed')
        path=r['path_id'];n=int(re.match(r'Ad(\d+)\.',path).group(1))
        if path not in aa[n]['paths'] or f'print_instance/Ad{n}::{path}' not in native.paths:raise ValueError('Unmodeled entity admitted')
        if data['kind'][row]==1:
            bound=corner_binding(row,r,data,aa,core,model)
            if bound is None:
                exclusions[row]=dict(row=row,stable_id=r['stable_id'],reason='fixed_material_corner_has_no_verified_same_named_native_corner',evidence_path=r['evidence_path']);records.pop(row);continue
            path,station,detail=bound;r['path_id']=path;stations[row]=station;corner_details[row]=detail
        elif data['kind'][row]==0:stations[row]=.5
        else:raise ValueError('Unexpected advertisement profile semantics')
    order=np.array(sorted(records),np.int64)
    paths=np.array(["print_instance/"+records[int(r)]["path_id"].split(".")[0]+"::"+records[int(r)]["path_id"] for r in order])
    path_ids=list(native.paths);lookup={p:i for i,p in enumerate(path_ids)}
    # Legacy semantic kind=0 curve,1 fixed point; current native kind=1 curve,0 fixed point.
    kind=np.where(data['kind'][order]==1,0,1).astype(np.uint8)
    points=dict(source_rows=order,path_indices=np.array([lookup[p] for p in paths],np.int32),stations=np.array([stations[int(r)] for r in order]),pts=data['pts'][order].copy(),uv=data['uv'][order].copy(),kind=kind,observation_ids=data['ids'][order].copy())
    ADS.mkdir(exist_ok=True)
    np.savez(ADS/'point_observations.npz',**points)
    np.savez(ADS/'initialized_points.npz',stations=points['stations'],seed_screen_distance=np.full(len(order),np.nan))
    camera=b.ROOT/'events/ankara_2026/programmes/okada_mei_sp/ad_cameras/s01.json';cam=b.read(camera)
    mapping=dict(schema='rinkspace.native_ad_replacement.v1',scene_path=str(b.SCENE),scene_revision=native.scene_revision,camera_path=str(camera),camera_sha256=b.sha(camera),control_sha256=b.sha(camera.parent/cam['controls_ref']),path_ids=path_ids,selected_rows=len(order),source_rows=len(data['uv']),frames=len(np.unique(points['pts'])),native_parameters_replaced_by_dataset=False,source_identity_uses_unfitted_native_geometry=False,current_point_input=str(ADS/'point_observations.npz'),objective='current point-only native consumer; no weights, noise allowances or geometry supplied by replacement',fit_started=False)
    b.write(ADS/'mapping.json',mapping)
    se=b.read(b.FULL/'semantic/manifest.json')
    np.savez_compressed(ADS/'source_provenance.npz',source_rows=order,stable_ids=data['ids'][order],original_dataset=data['datasets'][order],legacy_source_kind=data['kind'][order],original_active=data['old_active'][order],source_sigma=b.array(se['arrays']['source_sigma'])[order])
    # Every current model entity retains its actual parameter dependencies.
    from rinkspace.domain.print_assets import resolve_instance
    registry=[]
    for n,a in aa.items():
        if not 2<=n<=12:continue
        instance,template,size,placement=resolve_instance(a['asset'],f'print_instance/Ad{n}')
        internal=template.get('internal_geometry',{})
        groups={g['id']:g for g in internal.get('groups',[])} if isinstance(internal.get('groups'),list) else internal.get('groups',{})
        for path in a['paths'].values():
            binding=internal.get('path_bindings',{}).get(path['id'])
            dep=[block['id'] for block in model.initial_state.blocks.values() if block['kind']=='board_shape']+list(model.dependencies(f"print_instance/Ad{n}::{path['id']}"))
            registry.append(dict(path_id=f"print_instance/Ad{n}::{path['id']}",advertisement=n,template_id=template['id'],role=path.get('role'),parent_contour_key=path.get('parent_contour_key'),native_internal_binding=binding,parameter_block_ids=dep,native_asset=str(a['path']),native_asset_sha256=b.sha(a['path']),accepted_rows=int(np.count_nonzero(paths==f"print_instance/Ad{n}::{path['id']}"))))
    b.write(ADS/'native_entities.json',dict(scene=auth['scene'],entities=registry,native_parameter_values_copied_from_dataset=False))
    import gzip
    with gzip.open(ADS/'decisions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in sorted(all_ad):
            n=int(data['ad'][row]);accepted=row in records
            r=records[row] if accepted else exclusions[row]
            entry=dict(source_row=row,stable_id=str(data['ids'][row]),advertisement=n,native_pts=int(data['pts'][row]),time_base='1/12800',uv=data['uv'][row].tolist(),source_dataset=str(data['datasets'][row]),previous_label=str(data['labels'][row]),previous_active=bool(data['old_active'][row]),decision='include' if accepted else 'exclude',evidence=r)
            if row in corner_details:entry['native_corner_binding']=corner_details[row]
            f.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')
            counts[n]['source_rows']+=1;counts[n]['included' if accepted else 'excluded']+=1
            if accepted:counts[n][str(data['datasets'][row])]+=1
            else:counts[n]['reason/'+r['reason']]+=1
    # Preserve original statistical/correlation definitions as provenance only.
    provenance=dict(packet_manifest=dict(path=str(b.FULL/'packet/manifest.json'),sha256=b.sha(b.FULL/'packet/manifest.json')),semantic_manifest=dict(path=str(b.FULL/'semantic/manifest.json'),sha256=b.sha(b.FULL/'semantic/manifest.json')),source_evidence_index=dict(path=str(OUT/'source_evidence_index.json'),sha256=b.sha(OUT/'source_evidence_index.json')),native_authority_inventory=dict(path=str(OUT/'authority_inventory.json'),sha256=b.sha(OUT/'authority_inventory.json')),all_image_review_files=evidence_files,source_mask_index=dict(path=str(OUT/'visibility.json'),sha256=b.sha(OUT/'visibility.json')),original_sigma_groups_and_history_preserved=True,new_independence_or_noise_values_invented=False,consumer_does_not_load_statistics_as_constraints=True)
    provenance['review_recipes']=[dict(path=str(p),sha256=b.sha(p)) for p in [OUT/'final_verification_recipe.json',OUT/'cross_owner_review.json',OUT/'source_lineage.json',OUT/'direct_color_source_review.json',OUT/'color_review_summary.json',OUT/'final_color_decisions.json',OUT/'cross_color_review.json',OUT/'boundary_identity_review.json',OUT/'current_binding_color_audit.json',OUT/'current_explicit_conflicts.json',*Path(__file__).parent.glob('ad_*s01.py')] if p.is_file()]
    b.write(ADS/'provenance.json',provenance)
    b.write(ADS/'manifest.json',dict(schema='rinkspace.s01_ad_replacement.v1',scope='existing latest S01 Ad2-Ad12 source observations; current native entities only',source_rows=len(all_ad),included_rows=len(order),excluded_rows=len(exclusions),per_ad={str(k):dict(v) for k,v in counts.items()},final_native_advertisement_counts=dict(collections.Counter(path.split('::')[0].split('/')[-1] for path in paths)),original_source_UV_PTS_stable_IDs_unchanged=True,native_parameters_unchanged=True,source_identity_uses_current_native_geometry_distances=False,legacy_kind_to_native_kind={'0_finite_curve':1,'1_fixed_material_corner':0},files={p.name:dict(path=str(p),sha256=b.sha(p)) for p in ADS.iterdir() if p.is_file() and p.name!='manifest.json'},model_authority=auth['scene']))
    authority_check();print(json.dumps(dict(published_ad_rows=len(order),excluded=len(exclusions),per_ad={k:dict(v) for k,v in counts.items()})),flush=True)


def seed_ads():
    authority_check()
    sys.path.insert(0,str(b.ROOT/'src'))
    # Reuse the public initializer on this package only. The current native
    # scene/camera values are read; curve membership is already fixed by images.
    module_path=Path(__file__).with_name('initialize_native_s01.py')
    spec=importlib.util.spec_from_file_location('ad_package_initializer',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.prep.RUN=ADS
    before=dict(np.load(ADS/'point_observations.npz'))
    module.initialize()
    after=dict(np.load(ADS/'point_observations.npz'))
    if any(not np.array_equal(before[k],after[k]) for k in before):raise ValueError('Native initialization modified source input')
    initialized=np.load(ADS/'initialized_points.npz')
    if not np.array_equal(initialized['stations'][before['kind']==0],before['stations'][before['kind']==0]):raise ValueError('Fixed material corner moved on native curve')
    m=b.read(ADS/'manifest.json')
    for p in ADS.iterdir():
        if p.is_file() and p.name!='manifest.json':m['files'][p.name]=dict(path=str(p),sha256=b.sha(p))
    m['native_curve_station_initialization_only']=True;m['camera_and_native_geometry_optimization_executed']=False
    b.write(ADS/'manifest.json',m);authority_check()


def merge(target=b.CURRENT,output=BUNDLE):
    """Make a fresh directly consumable bundle; never overwrite a running input."""
    target=Path(target).resolve();output=Path(output).resolve()
    if output==target or output.is_relative_to(target):raise ValueError('Use a separate output directory')
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be new or empty')
    authority_check()
    add=dict(np.load(ADS/'point_observations.npz',allow_pickle=False));addseed=dict(np.load(ADS/'initialized_points.npz',allow_pickle=False));am=b.read(ADS/'mapping.json')
    old=dict(np.load(target/'point_observations.npz',allow_pickle=False));oldseed=dict(np.load(target/'initialized_points.npz',allow_pickle=False));mapping=b.read(target/'mapping.json')
    fingerprints={name:b.sha(target/name) for name in ['point_observations.npz','initialized_points.npz','mapping.json']}
    for key in ('scene_revision','camera_sha256','control_sha256'):
        if mapping.get(key)!=am.get(key):
            raise ValueError('Current input and replacement authority differ: '+key)
    manifest=b.read(ADS/'manifest.json')
    for item in manifest['files'].values():
        if b.sha(item['path'])!=item['sha256']:
            raise ValueError('Replacement file changed before merge: '+item['path'])
    oldpaths=np.asarray(mapping['path_ids'])[old['path_indices']]
    isad=np.array([bool(re.match(r'^print_instance/Ad(?:[2-9]|1[012])::',p)) for p in oldpaths])
    keep=np.flatnonzero(~isad)
    allpaths=list(mapping['path_ids']);pindex={p:i for i,p in enumerate(allpaths)}
    apaths=np.asarray(am['path_ids'])[add['path_indices']]
    for path in apaths:
        if path not in pindex:pindex[path]=len(allpaths);allpaths.append(str(path))
    merged={}
    for key in old:
        if key not in add:raise ValueError('Current consumer field requires explicit compatibility: '+key)
        new=np.array([pindex[p] for p in apaths],np.int32) if key=='path_indices' else add[key]
        if key=='observation_ids':merged[key]=np.concatenate([old[key][keep].astype(str),new.astype(str)])
        else:merged[key]=np.concatenate([old[key][keep],new])
    n=len(merged['pts']);ids=merged['observation_ids'].astype(str)
    if len(set(ids))!=n:raise ValueError('Duplicate stable observation identity in merged package')
    if 'stations' not in oldseed or 'stations' not in addseed:
        raise ValueError('Both inputs require station initializers')
    # The live pixel consumer currently saves only stations. Preserve its
    # initializer schema; optional diagnostics remain in the ad-only package.
    if set(oldseed)-{'stations','seed_screen_distance'}:
        raise ValueError('Unknown current initializer fields require explicit review')
    init={k:np.concatenate([oldseed[k][keep],addseed[k]]) for k in oldseed}
    for key in old:
        if not np.array_equal(merged[key][:len(keep)].astype(old[key].dtype),old[key][keep]):raise ValueError('Non-ad source changed: '+key)
    for key in init:
        if not np.array_equal(init[key][:len(keep)],oldseed[key][keep],equal_nan=True):raise ValueError('Non-ad initialization changed')
    output.mkdir(parents=True,exist_ok=True);np.savez(output/'point_observations.npz',**merged);np.savez(output/'initialized_points.npz',**init)
    old_by_id={str(s):p for s,p in zip(old['observation_ids'].astype(str),oldpaths)}
    changed=sum(str(s) in old_by_id and old_by_id[str(s)]!=p for s,p in zip(add['observation_ids'],apaths))
    added=sum(str(s) not in old_by_id for s in add['observation_ids'])
    adids=set(add['observation_ids'].astype(str))
    removed=sum(str(s) not in adids for s in old['observation_ids'][isad].astype(str))
    for key in ['ad4_text_association_translation','material_identity_uses_unfitted_template_for_ads']:mapping.pop(key,None)
    mapping.update(schema='rinkspace.current_native_with_replaced_s01_ads.v1',path_ids=allpaths,selected_rows=n,frames=len(np.unique(merged['pts'])),current_point_input=str(output/'point_observations.npz'),native_parameters_replaced_by_dataset=False,source_identity_uses_unfitted_native_geometry=False,advertisement_replacement=dict(path=str(ADS/'manifest.json'),sha256=b.sha(ADS/'manifest.json')),nonadvertisement_source=dict(path=str(target),sha256=fingerprints['point_observations.npz'],rows=len(keep)),fit_started=False,visual_mapping_review='independent_source_entity_RGB_and_existing_SAM2_mask_checks_complete; no claim of fitted scene accuracy')
    mapping['features']=[f for f in mapping.get('features',[]) if not f.get('label','').startswith('Ad')]
    mapping['features']+=b.read(ADS/'native_entities.json')['entities']
    b.write(output/'mapping.json',mapping)
    receipt=dict(schema='rinkspace.current_point_input_ad_replacement.v1',target_snapshot=str(target),target_file_sha256=fingerprints,input=str(output/'point_observations.npz'),sha256=b.sha(output/'point_observations.npz'),rows=n,unchanged_nonad_rows=len(keep),old_ad_rows=int(isad.sum()),new_ad_rows=len(add['pts']),added_stable_ids=added,removed_old_stable_ids=removed,changed_native_entity_for_existing_ids=changed,all_nonad_fields_and_initializers_bitwise_preserved=True,C0_preserved_rows=int(np.count_nonzero(oldpaths[keep]=='d78_material_144/path0')),source_ids_uv_pts_unchanged=True,native_parameters_changed=False,solver_run=False)
    b.write(output/'point_input.json',receipt)
    if any(b.sha(target/name)!=digest for name,digest in fingerprints.items()):raise ValueError('Live input changed during merge; output is not a current snapshot')
    authority_check();print(json.dumps(receipt),flush=True)


def rebind_state(state_file,target=b.CURRENT,bundle=BUNDLE,output=None):
    """Transfer an explicitly selected native numeric state without changing it."""
    authority_check();state_file=Path(state_file);target=Path(target);bundle=Path(bundle)
    output=Path(output) if output else bundle/'rebound_numeric_state.npz'
    if output.exists():raise ValueError('Do not overwrite a numeric snapshot')
    old=dict(np.load(target/'point_observations.npz',allow_pickle=False))
    new=dict(np.load(bundle/'point_observations.npz',allow_pickle=False))
    source_hash=b.sha(state_file);state=dict(np.load(state_file,allow_pickle=False))
    if b.sha(state_file)!=source_hash:raise ValueError('Chosen live snapshot changed during read')
    required={'controls','stations','board','layout','rig','scene_values','source_rows','point_input_sha256'}
    if set(state)!=required:raise ValueError('Numeric-state schema needs explicit review')
    if str(state['point_input_sha256'])!=b.sha(target/'point_observations.npz') or not np.array_equal(state['source_rows'],old['source_rows']):raise ValueError('Chosen state does not belong to the current input snapshot')
    mapping=b.read(bundle/'mapping.json');paths=np.asarray(mapping['path_ids'])[new['path_indices']]
    previous={str(s):i for i,s in enumerate(old['observation_ids'].astype(str))}
    stations=np.load(bundle/'initialized_points.npz')['stations'].copy()
    preserved=0
    for i,(sid,path) in enumerate(zip(new['observation_ids'].astype(str),paths)):
        if not re.match(r'^print_instance/Ad(?:[2-9]|1[012])::',path):
            stations[i]=state['stations'][previous[sid]];preserved+=1
    result={k:value.copy() for k,value in state.items() if k not in ('stations','source_rows','point_input_sha256')}
    result.update(stations=stations,source_rows=new['source_rows'].copy(),point_input_sha256=np.array(b.sha(bundle/'point_observations.npz')))
    for k in ('controls','board','layout','rig','scene_values'):
        if not np.array_equal(state[k],result[k]):raise ValueError('Native numeric values changed')
    output.parent.mkdir(parents=True,exist_ok=True);np.savez(output,**result)
    receipt=dict(source_state=str(state_file),source_state_sha256=source_hash,output=str(output),output_sha256=b.sha(output),
        native_global_parameter_arrays_bitwise_unchanged=['controls','board','layout','rig','scene_values'],
        global_array_sha256={k:hashlib.sha256(result[k].tobytes()).hexdigest() for k in ('controls','board','layout','rig','scene_values')},
        unchanged_nonad_numeric_stations=preserved,advertisement_stations='new package initialization; current consumer refines finite curves before the first solver step',
        fixed_material_corner_stations_preserved=True,source_identity_not_derived_from_chosen_state=True,optimizer_run=False)
    b.write(output.with_suffix('.json'),receipt);authority_check();print(json.dumps(receipt),flush=True)


def validate(bundle=BUNDLE):
    authority_check();data=b.load();manifest=b.read(ADS/'manifest.json')
    z=dict(np.load(ADS/'point_observations.npz',allow_pickle=False));rows=z['source_rows']
    for k,source in [('uv',data['uv']),('pts',data['pts']),('observation_ids',data['ids'])]:
        if not np.array_equal(z[k],source[rows]):raise ValueError('Source array changed: '+k)
    expected=np.where(data['kind'][rows]==1,0,1).astype('uint8')
    if not np.array_equal(z['kind'],expected):raise ValueError('Fixed/curve semantics changed')
    registry=b.read(ADS/'native_entities.json')['entities'];allowed={r['path_id'] for r in registry}
    mapping=b.read(ADS/'mapping.json');paths=np.asarray(mapping['path_ids'])[z['path_indices']]
    if not set(paths)<=allowed:raise ValueError('Retired native entity found')
    index=b.read(OUT/'source_evidence_index.json');banned={int(x) for x in index['withdrawals']}|{int(x) for x in index['retired']}
    if set(map(int,rows))&banned:raise ValueError('Explicit source withdrawal or retired entity reintroduced')
    byrow={int(r):i for i,r in enumerate(rows)}
    known_corrections={}
    for file in (OUT/'verified_color_final').glob('*.json'):
        decision=b.read(file)
        for record in decision['confirmed']+decision['excluded']:
            if record['row'] in (29614,29615,337718,337614):known_corrections[record['row']]=record
    for r in (29614,29615):
        if r in byrow:
            if paths[byrow[r]]!='print_instance/Ad2::Ad2.print.K':raise ValueError('Known A/K misbinding not repaired')
        elif known_corrections[r].get('identified_native_entity')!='Ad2.print.K':raise ValueError('Blurred known K identity lost')
    for row,path in [(337718,'Ad4.ANKARA.native440.boundary_5'),(337614,'Ad4.ANKARA.native440.boundary_4')]:
        if row in byrow:
            if paths[byrow[row]]!='print_instance/Ad4::'+path:raise ValueError('ANKARA falsely bound as GRAND')
        elif known_corrections[row].get('identified_native_entity')!=path:raise ValueError('Actual ANKARA first A identity not recorded')
    if 28179 in byrow:raise ValueError('Known occlusion point readmitted')
    for r in range(355554,355578):
        if r not in byrow or z['kind'][byrow[r]]!=0:raise ValueError('Verified new fixed letter corner missing or sliding')
    if len(set(z['observation_ids']))!=len(rows):raise ValueError('Duplicate ad observation identity')
    if not np.isfinite(z['uv']).all() or not np.isfinite(z['stations']).all():raise ValueError('Invalid point array')
    if not np.isin(z['pts'],np.arange(256256,575489,256)).all():raise ValueError('Non-S01 or resampled PTS in input')
    for atlas in b.read(OUT/'atlases.json')['atlases']:
        if b.sha(atlas['image'])!=atlas['image_sha256']:raise ValueError('Source reference changed')
    visibility=b.read(OUT/'visibility.json')
    if visibility['mask_frames']!=1248:raise ValueError('Missing native SAM2 exposure')
    for mask in visibility['frames']:
        if b.sha(mask['mask_path'])!=mask['mask_sha256']:raise ValueError('SAM2 source mask changed')
    bundle=Path(bundle);receipt=b.read(bundle/'point_input.json');merged=dict(np.load(bundle/'point_observations.npz'))
    if b.sha(bundle/'point_observations.npz')!=receipt['sha256']:raise ValueError('Merged input changed')
    # Verify the delivered arrays independently, not merely the merge receipt.
    target=Path(receipt['target_snapshot'])
    for name,digest in receipt['target_file_sha256'].items():
        if b.sha(target/name)!=digest:raise ValueError('Target snapshot changed: '+name)
    old=dict(np.load(target/'point_observations.npz',allow_pickle=False))
    oldmap=b.read(target/'mapping.json');oldpaths=np.asarray(oldmap['path_ids'])[old['path_indices']]
    keep=np.flatnonzero([not re.match(r'^print_instance/Ad(?:[2-9]|1[012])::',p) for p in oldpaths])
    if set(merged)!=set(old) or len(merged['pts'])!=len(keep)+len(rows):
        raise ValueError('Merged consumer schema or row count changed')
    mergedmap=b.read(bundle/'mapping.json');mergedpaths=np.asarray(mergedmap['path_ids'])[merged['path_indices']]
    for key in old:
        if not np.array_equal(merged[key][:len(keep)].astype(old[key].dtype),old[key][keep]):
            raise ValueError('Delivered non-ad source changed: '+key)
        if key!='path_indices' and not np.array_equal(merged[key][len(keep):],z[key]):
            raise ValueError('Delivered ad source differs from replacement: '+key)
    if not np.array_equal(mergedpaths[len(keep):],paths):raise ValueError('Delivered native entities differ')
    oldseed=dict(np.load(target/'initialized_points.npz',allow_pickle=False))
    mergedseed=dict(np.load(bundle/'initialized_points.npz',allow_pickle=False))
    addseed=dict(np.load(ADS/'initialized_points.npz',allow_pickle=False))
    if set(mergedseed)!=set(oldseed):raise ValueError('Initializer schema changed')
    for key in oldseed:
        if not np.array_equal(mergedseed[key][:len(keep)],oldseed[key][keep],equal_nan=True):
            raise ValueError('Delivered non-ad initialization changed: '+key)
        if not np.array_equal(mergedseed[key][len(keep):],addseed[key],equal_nan=True):
            raise ValueError('Delivered ad initialization changed: '+key)
    evidence_records={}
    for entry in b.read(ADS/'provenance.json')['all_image_review_files']:
        if b.sha(entry['path'])!=entry['sha256']:raise ValueError('Final source evidence changed')
        for record in b.read(entry['path'])['confirmed']:evidence_records[record['row']]=record
    for i,row in enumerate(rows):
        record=evidence_records[int(row)]
        if z['kind'][i]==1 and paths[i].split('::',1)[1]!=record['path_id']:
            raise ValueError('Final entity correction missing from published point')
    changes=b.read(OUT/'boundary_identity_review.json')['changes']
    for change in changes:
        row=change['row']
        if change['decision']=='exclude' and row in byrow:raise ValueError('Withdrawn boundary readmitted')
        if change['decision']=='include' and row in byrow and z['kind'][byrow[row]]==1:
            if paths[byrow[row]].split('::',1)[1]!=change['current']:
                raise ValueError('Boundary side correction missing')

    seed=np.load(ADS/'initialized_points.npz')
    if not np.isfinite(seed['stations']).all():raise ValueError('Invalid native station initialization')
    if not np.array_equal(seed['stations'][z['kind']==0],z['stations'][z['kind']==0]):raise ValueError('Native fixed corner changed')
    for entry in manifest['files'].values():
        if b.sha(entry['path'])!=entry['sha256']:raise ValueError('Published package file changed: '+entry['path'])
    if set(map(int,rows)) & set(visibility['inside_person_mask_rows']):raise ValueError('SAM2-occluded point in published input')
    evaluation=b.read(ADS/'initialized_evaluation.json')
    if evaluation['invalid']!=0:raise ValueError('Current native projection cannot consume every replacement point')
    output=dict(schema='rinkspace.ad_replacement_validation.v1',passed=True,source_rows_checked=manifest['source_rows'],included_ad_rows=len(rows),
        source_pixels_pts_stable_ids_bitwise_identical=True,explicit_withdrawals_and_retired_entities_absent=True,
        all_24_added_letter_corners_preserved_as_native_fixed_corners=True,known_A_to_K_misbinding_corrected_rows=[29614,29615],known_ANKARA_first_A_identity_restored_rows=[337718,337614],color_word_review_completed=True,known_identity_points_may_remain_withdrawn_for_blur=True,
        known_occluded_point_absent=28179,original_Ad6_white_interior_withdrawals_absent=78,
        existing_SAM2_mask_frames_verified=1248,Ad11_corrected_mask_frames=15,native_entity_registry_rows=len(registry),
        current_native_projection_invalid_rows=evaluation['invalid'],all_native_model_hashes_unchanged=True,
        immutable_nonad_rows=receipt['unchanged_nonad_rows'],C0_rows_preserved=receipt['C0_preserved_rows'],
        consumer_fields=list(merged),consumer_kind_semantics='0 fixed material point; 1 finite curve',
        delivered_nonad_arrays_and_initializers_independently_compared=True,delivered_ad_arrays_and_entities_match_final_decisions=True,final_boundary_revision_checked=True,new_fit_started=False,full_scene_accuracy_claimed=False,replacement_manifest_sha256=b.sha(ADS/'manifest.json'),
        native_bundle_receipt_sha256=b.sha(bundle/'point_input.json'))
    state_path=bundle/'rebound_numeric_state.npz'
    if state_path.exists():
        state=np.load(state_path);r=b.read(state_path.with_suffix('.json'))
        if any(hashlib.sha256(state[k].tobytes()).hexdigest()!=v for k,v in r['global_array_sha256'].items()):raise ValueError('Rebound native numerical parameters changed')
        if str(state['point_input_sha256'])!=receipt['sha256'] or not np.array_equal(state['source_rows'],merged['source_rows']):raise ValueError('Rebound state belongs to another data input')
        output['chosen_native_global_parameters_transferred_bitwise']=True
    b.write(OUT/'validation.json',output);print(json.dumps(output),flush=True)
    return output


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['publish','seed','merge','rebind-state','validate']);p.add_argument('--target',type=Path,default=b.CURRENT);p.add_argument('--output',type=Path,default=BUNDLE);p.add_argument('--state',type=Path);p.add_argument('--state-output',type=Path);a=p.parse_args()
    if a.stage=='publish':publish()
    elif a.stage=='seed':seed_ads()
    elif a.stage=='merge':merge(a.target,a.output)
    elif a.stage=='validate':validate(a.output)
    else:
        if a.state is None:p.error('--state is required for rebind-state')
        rebind_state(a.state,a.target,a.output,a.state_output)
