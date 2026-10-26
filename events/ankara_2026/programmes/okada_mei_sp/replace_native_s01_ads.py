"""Replace all S01 advertisement points, preserving the live non-ad/native state."""
from pathlib import Path
import argparse,collections,gzip,hashlib,importlib.util,io,json,shutil,zipfile
import numpy as np

spec=importlib.util.spec_from_file_location('s01prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU

PACKAGE=prep.DATA/'runs/native_ad_replacement_s01_20261026/S01_ads_incomplete_baseline.zip'
STAGE=prep.RUN/'ad_baseline_f494811e9d72'
CURRENT_FILES=('point_observations.npz','initialized_points.npz','mapping.json','point_input.json','accepted_pixels_latest.npz')
GLOBALS=('controls','rig','scene_values','board','layout')

def require(condition,message):
    if not condition:raise ValueError(message)

def zipped_arrays(archive,name):
    with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as z:return {k:z[k] for k in z.files}

def equal(a,b):
    if a.dtype.kind in 'SU' or b.dtype.kind in 'SU':return np.array_equal(a.astype(str),b.astype(str))
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def native_guards():
    files=[p for p in (prep.ROOT/'native').rglob('*') if p.is_file()]
    files.extend((prep.ROOT/'src/rinkspace/core').glob('*.py'))
    files.extend([prep.ROOT/'src/rinkspace/session/gpu_geometry.py',prep.SCENE,prep.CAMERA,
        prep.CAMERA.parent/prep.read(prep.CAMERA)['controls_ref'],prep.LIBRARY,
        Path(__file__).with_name('fit_native_s01_joint.py'),Path(__file__).with_name('prepare_native_s01.py'),
        prep.ROOT/'events/ankara_2026/model_constraints/camera_hardware/z190_mount_working_ranges_20261026.json'])
    return {str(p):prep.digest(p) for p in files}

def verify_guards(hashes):
    for path,digest in hashes.items():require(prep.digest(path)==digest,'Protected source/native authority changed: '+path)

def prepare():
    require(not (STAGE/'replacement_receipt.json').exists(),'Prepared replacement already exists; inspect it rather than overwriting')
    guarded=native_guards();current_hashes={n:prep.digest(prep.RUN/n) for n in CURRENT_FILES}
    old,mapping,old_seed=prep.load_pixel_input(prep.RUN)
    with np.load(prep.RUN/'accepted_pixels_latest.npz',allow_pickle=False) as z:state={k:z[k] for k in z.files}
    require(str(state['point_input_sha256'])==current_hashes['point_observations.npz'] and equal(state['source_rows'],old['source_rows']),'Current numeric state belongs to another input')
    model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=NativeSceneGPU(prep.SCENE,NativeCore(prep.LIBRARY),parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    old_paths=np.asarray(mapping['path_ids'])[old['path_indices']]
    keep=np.flatnonzero([native.paths[p]['layer']!='advertisement' for p in old_paths])
    remove=np.flatnonzero([native.paths[p]['layer']=='advertisement' for p in old_paths])
    with zipfile.ZipFile(PACKAGE) as archive:
        manifest=json.loads(archive.read('incomplete_baseline_manifest.json'))
        for entry in manifest['files']:
            data=archive.read(entry['path'])
            require(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'Package manifest mismatch: '+entry['path'])
        ad=zipped_arrays(archive,'ads_only/point_observations.npz');am=json.loads(archive.read('ads_only/mapping.json'))
        full=zipped_arrays(archive,'native_bundle/point_observations.npz');fm=json.loads(archive.read('native_bundle/mapping.json'))
        ad_seed=zipped_arrays(archive,'ads_only/initialized_points.npz');full_seed=zipped_arrays(archive,'native_bundle/initialized_points.npz')
        require(set(ad)==prep.POINT_FIELDS and set(full)==prep.POINT_FIELDS,'Unexpected numerical fields in replacement')
        require(set(full_seed)=={'stations'} and set(ad_seed)=={'stations','seed_screen_distance'},'Unexpected initializer fields')
        require(am['path_ids']==mapping['path_ids']==fm['path_ids'],'Path registry changed; explicit remapping required')
        require(am['scene_revision']==native.scene_revision==mapping['scene_revision'],'Scene revision mismatch')
        ap=np.asarray(am['path_ids'])[ad['path_indices']]
        require(all(native.paths[p]['layer']=='advertisement' for p in ap),'Replacement contains a non-advertisement entity')
        # Independently assemble the expected replacement from the LIVE data.
        # Keep the verified package payload bytes to retain its incremental-base hash.
        for key in prep.POINT_FIELDS:
            expected=np.concatenate((old[key][keep].astype(full[key].dtype),ad[key].astype(full[key].dtype)))
            require(equal(full[key],expected),'Bundled replacement differs from live non-ads plus new ads: '+key)
            require(equal(old[key][keep],full[key][:len(keep)]),'Live non-ad field changed: '+key)
        expected_seed=np.r_[old_seed[keep],ad_seed['stations']]
        require(equal(full_seed['stations'],expected_seed),'Live non-ad initializer changed')
        require(np.array_equal(np.unique(full['pts']),np.unique(old['pts'])),'Exposure coverage changed')
        require(np.isin(ad['pts'],np.unique(old['pts'])).all(),'Advertisement PTS outside current S01')
        ids=ad['observation_ids'].astype(str);by_id={sid:i for i,sid in enumerate(ids)}
        by_row={int(row):i for i,row in enumerate(ad['source_rows'])}
        require(len(by_id)==len(ids)==len(by_row),'Duplicate replacement identity')
        require(not(set(ids)&set(old['observation_ids'][keep].astype(str))),'Advertisement duplicates a live non-ad point')
        provenance=json.loads(archive.read('ads_only/provenance.json'));ref=provenance['source_evidence_index']
        require(prep.digest(ref['path'])==ref['sha256'],'Explicit source withdrawal index changed')
        source_index=prep.read(ref['path']);banned=set(map(int,source_index['withdrawals']))|set(map(int,source_index['retired']))
        require(not(set(by_row)&banned),'Explicitly withdrawn or retired source point returned')
        reviewed=prep.read(prep.RUN/'ad4_contact_review.json')['decisions'];old_ids=old['observation_ids'].astype(str)
        live_index={sid:i for i,sid in enumerate(old_ids)}
        for record in reviewed:
            sid=record['observation_id'];require(sid not in by_id,'Known Ad4 contact/error point returned as advertisement')
            fingerprint=(ad['pts']==record['pts'])&np.all(ad['uv']==record['uv'],axis=1)
            require(not fingerprint.any(),'Known bad Ad4 pixel returned under another ID')
            if record['action']=='rebind':
                i=live_index[sid];require(i in keep and old_paths[i]==record['new_path_id'],'Current contact correction missing')
            else:require(sid not in set(full['observation_ids'].astype(str)),'Excluded Ad4 point returned')
        # Check delivered observations against final decisions, not the package's
        # pass/fail summary. Correlations and historical sigma remain provenance.
        seen=set();included=0;excluded=0
        with gzip.GzipFile(fileobj=io.BytesIO(archive.read('ads_only/decisions.jsonl.gz'))) as stream:
            for line in stream:
                d=json.loads(line);sid=d['stable_id']
                if d['decision']=='exclude':
                    excluded+=1;require(sid not in by_id,'Excluded decision returned');continue
                require(d['decision']=='include' and sid in by_id,'Unexpected decision or missing included point')
                i=by_id[sid];require(sid not in seen,'Duplicate final decision');seen.add(sid);included+=1
                require(d['source_row']==int(ad['source_rows'][i]) and d['native_pts']==int(ad['pts'][i]),'Source identity/time changed')
                require(np.array_equal(ad['uv'][i],np.asarray(d['uv'],float)),'Original source UV changed')
                require(ap[i].split('::',1)[1]==d['evidence']['path_id'],'Final reviewed native identity differs')
                if ad['kind'][i]==0:
                    binding=d['native_corner_binding'];station=binding['native_seed_vertex_index']/max(1,binding['native_seed_vertex_count']-1)
                    require(ad['stations'][i]==station and ad_seed['stations'][i]==station,'Fixed corner became a sliding point')
        require(included==len(ad['pts']) and len(seen)==len(ids),'Final decision coverage mismatch')
        boundary=json.loads(archive.read('boundary_identity_review.json'))['changes']
        for d in boundary:
            if d['decision']=='exclude':require(d['row'] not in by_row,'Excluded panel edge returned')
            elif d['row'] in by_row:
                i=by_row[d['row']]
                if ad['kind'][i]==1:require(ap[i].split('::',1)[1]==d['current'],'Panel edge side correction lost')
        for row,target in ((29614,'Ad2.print.K'),(29615,'Ad2.print.K'),(337718,'Ad4.ANKARA.native440.boundary_5'),(337614,'Ad4.ANKARA.native440.boundary_4')):
            if row in by_row:require(ap[by_row[row]].split('::',1)[1]==target,'Known letter identity regressed')
        require(28179 not in by_row,'Known occluded point returned')
        for row in range(355554,355578):require(row in by_row and ad['kind'][by_row[row]]==0,'Added fixed letter corner missing')
        # Nothing from the replacement's rig/geometry snapshot is installed.
        rebound={k:v.copy() for k,v in state.items() if k not in ('stations','source_rows','point_input_sha256')}
        rebound.update(stations=np.r_[state['stations'][keep],ad_seed['stations']],source_rows=full['source_rows'].copy(),
            point_input_sha256=np.array(manifest['point_input_sha256']))
        for key in GLOBALS:require(equal(rebound[key],state[key]),'Live global parameter changed: '+key)
        STAGE.mkdir(exist_ok=True)
        (STAGE/'point_observations.npz').write_bytes(archive.read('native_bundle/point_observations.npz'))
        (STAGE/'initialized_points.npz').write_bytes(archive.read('native_bundle/initialized_points.npz'))
    sha=prep.digest(STAGE/'point_observations.npz');require(sha==manifest['point_input_sha256'],'Baseline input fingerprint mismatch')
    current_mapping={k:mapping[k] for k in ('scene_path','scene_revision','camera_path','camera_sha256','control_sha256','path_ids')}
    current_mapping.update(schema='rinkspace.current_native_advertisement_replacement.v1',selected_rows=len(full['pts']),frames=len(np.unique(full['pts'])),
        native_parameters_replaced_by_dataset=False,current_point_input=str(prep.RUN/'point_observations.npz'),
        objective='unweighted_squared_pixel_distance',replacement_baseline_id=manifest['baseline_id'],
        previous_mapping_archive=str(STAGE/'previous/mapping.json'),original_source_points_not_reestimated=True)
    prep.write(STAGE/'mapping.json',current_mapping);np.savez(STAGE/'rebound_numeric_state.npz',**rebound)
    receipt=dict(schema='rinkspace.current_point_input_ad_replacement.v1',baseline_id=manifest['baseline_id'],
        package=str(PACKAGE),package_sha256=prep.digest(PACKAGE),input=str(prep.RUN/'point_observations.npz'),sha256=sha,
        rows=len(full['pts']),frames=len(np.unique(full['pts'])),old_ad_rows=len(remove),new_ad_rows=len(ad['pts']),unchanged_nonad_rows=len(keep),
        C0_preserved_rows=int(np.sum(old_paths[keep]=='d78_material_144/path0')),ad4_contact_rebindings_preserved=32,ad4_withdrawals_still_absent=15,
        advertisement_frames=len(np.unique(ad['pts'])),frames_without_advertisements=len(np.unique(full['pts']))-len(np.unique(ad['pts'])),
        fields=sorted(prep.POINT_FIELDS),weights_used=False,covariance_used=False,robust_loss=False,error_allowance=False,legacy_query_constraints=False,
        excluded_from_numerical_input=['source_sigma','seed_screen_distance','correlations','transition_widths','old_groups','native_parameters_from_package'],
        nonad_values_initializers_and_numeric_stations_unchanged=True,global_native_parameter_arrays_unchanged=list(GLOBALS),
        all_final_ad_decisions_checked=included,excluded_decisions_checked=excluded,explicit_withdrawals_and_retired_sources_checked=len(banned),
        incomplete_test_baseline=True,visual_accuracy_accepted=False,optimizer_steps=0)
    prep.write(STAGE/'point_input.json',receipt)
    prep.load_pixel_input(STAGE)
    verify_guards(guarded)
    require(all(prep.digest(prep.RUN/n)==h for n,h in current_hashes.items()),'Live input changed while preparing')
    prep.write(STAGE/'replacement_receipt.json',dict(stage='prepared',current_file_hashes=current_hashes,protected_files=guarded,
        input_receipt=receipt,prepared_file_hashes={n:prep.digest(STAGE/n) for n in ('point_observations.npz','initialized_points.npz','mapping.json','point_input.json','rebound_numeric_state.npz')}))
    print(json.dumps(receipt,ensure_ascii=True))

def install():
    r=prep.read(STAGE/'replacement_receipt.json');require(r['stage']=='prepared','Replacement is already installed or has another state')
    v=prep.read(STAGE/'fit_input_validation.json')
    require(v['passed'] and v['optimizer_steps']==0 and v['input_sha256']==r['input_receipt']['sha256'],'Full current GPU input validation required')
    verify_guards(r['protected_files'])
    require(all(prep.digest(prep.RUN/n)==h for n,h in r['current_file_hashes'].items()),'Live input/state changed after preparation')
    require(all(prep.digest(STAGE/n)==h for n,h in r['prepared_file_hashes'].items()),'Prepared replacement changed after validation')
    previous=STAGE/'previous';previous.mkdir(exist_ok=True)
    for name,digest in r['current_file_hashes'].items():
        dst=previous/name
        if not dst.exists():shutil.copy2(prep.RUN/name,dst)
        require(prep.digest(dst)==digest,'Previous-version archive mismatch')
    # Prepare every replacement before installing any; each individual rename
    # is atomic. A crash cannot pass the input/state digest checks half-installed.
    names={n:n for n in ('point_observations.npz','initialized_points.npz','mapping.json','point_input.json')}
    names['accepted_pixels_latest.npz']='rebound_numeric_state.npz'
    for target,source in names.items():shutil.copy2(STAGE/source,prep.RUN/(target+'.replacement_pending'))
    for target in names:(prep.RUN/(target+'.replacement_pending')).replace(prep.RUN/target)
    z,_,_=prep.load_pixel_input(prep.RUN)
    with np.load(prep.RUN/'accepted_pixels_latest.npz',allow_pickle=False) as s:
        require(str(s['point_input_sha256'])==prep.digest(prep.RUN/'point_observations.npz') and equal(s['source_rows'],z['source_rows']),'Installed numeric state/input mismatch')
        with np.load(previous/'accepted_pixels_latest.npz',allow_pickle=False) as old:
            for key in GLOBALS:require(equal(s[key],old[key]),'Installed native global state changed: '+key)
    verify_guards(r['protected_files'])
    r.update(stage='installed',installed_file_hashes={n:prep.digest(prep.RUN/n) for n in CURRENT_FILES},gpu_input_validation=v,
        original_optimizer_stop_request_preserved=(prep.RUN/'stop_requested').exists(),production_fit_started=False)
    prep.write(STAGE/'replacement_receipt.json',r)
    print(json.dumps(dict(installed=True,input=str(prep.RUN/'point_observations.npz'),sha256=r['input_receipt']['sha256'],rows=len(z['pts']),report=str(STAGE/'replacement_receipt.json')),ensure_ascii=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','install']);args=parser.parse_args()
    prepare() if args.action=='prepare' else install()
