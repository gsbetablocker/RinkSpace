"""Rebase a checked advertisement revision onto the live S01 observations/state."""
from pathlib import Path
import argparse, collections, gzip, importlib.util, json
import numpy as np

spec=importlib.util.spec_from_file_location('replacement',Path(__file__).with_name('replace_native_s01_ads.py'))
replacement=importlib.util.module_from_spec(spec);spec.loader.exec_module(replacement)
prep=replacement.prep
require=replacement.require
equal=replacement.equal


def prepare(increment,delta):
    stage=replacement.STAGE
    require(not (stage/'replacement_receipt.json').exists(),'Prepared revision already exists')
    guarded=replacement.native_guards();guarded[str(Path(__file__))]=prep.digest(Path(__file__))
    hashes={n:prep.digest(prep.RUN/n) for n in replacement.CURRENT_FILES}
    z,m,initial=prep.load_pixel_input(prep.RUN)
    state=dict(np.load(prep.RUN/'accepted_pixels_latest.npz',allow_pickle=False))
    require(str(state['point_input_sha256'])==hashes['point_observations.npz'] and equal(state['source_rows'],z['source_rows']),'Live numeric state/input mismatch')
    manifest=prep.read(increment/'increment_manifest.json')
    require(prep.digest(delta)==prep.digest(increment/'delta_operations.jsonl.gz'),'Attached delta is not this revision')
    for e in manifest['files']:
        require(prep.digest(increment/e['path'])==e['sha256'],'Revision hash mismatch: '+e['path'])
    current_receipt=prep.read(prep.RUN/'point_input.json')
    require(current_receipt['advertisement_baseline_id']==manifest['baseline_id'],'Wrong advertisement baseline')
    a=dict(np.load(increment/'ads_only/point_observations.npz',allow_pickle=False))
    am=prep.read(increment/'ads_only/mapping.json')
    ai=dict(np.load(increment/'ads_only/initialized_points.npz',allow_pickle=False))
    require(set(a)==prep.POINT_FIELDS and set(ai)=={'stations','seed_screen_distance'},'Unexpected payload schema')
    for key in ('scene_revision','camera_sha256','control_sha256'):
        require(m[key]==am[key],'Native authority mismatch: '+key)
    model=replacement.SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=replacement.NativeSceneGPU(prep.SCENE,replacement.NativeCore(prep.LIBRARY),
        parameter_state=replacement.ParameterState(model.initial_state.layout,state['scene_values']))
    require(native.scene_revision==m['scene_revision'],'Current scene authority changed')
    paths=np.asarray(m['path_ids']);oldpaths=paths[z['path_indices']]
    apaths=np.asarray(am['path_ids'])[a['path_indices']]
    require(all(p in native.paths and native.paths[p]['layer']=='advertisement' for p in apaths),'Non-ad or unknown entity in ad payload')
    keep=np.flatnonzero([native.paths[p]['layer']!='advertisement' for p in oldpaths])
    removed=np.flatnonzero([native.paths[p]['layer']=='advertisement' for p in oldpaths])
    lookup={p:i for i,p in enumerate(paths)}
    require(all(p in lookup for p in apaths),'Revision adds unverified native geometry')
    mapped={**a,'path_indices':np.array([lookup[p] for p in apaths],np.int32)}
    expected=prep.selected_s01_exposures()
    require(manifest['source_scope']['first_pts']==int(expected[0]) and manifest['source_scope']['last_pts']==int(expected[-1]),'Wrong processing scope')
    require(np.isin(a['pts'],expected).all(),'Ad exposure outside current scope')
    by_id={str(s):i for i,s in enumerate(a['observation_ids'])}
    old_id={str(s):i for i,s in enumerate(z['observation_ids'])}
    require(len(by_id)==len(a['pts']),'Duplicate advertisement identity')
    actions={};counts=collections.Counter()
    with gzip.open(delta,'rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);sid=d['stable_id'];action=d['action']
            require(sid not in actions,'Duplicate delta operation');actions[sid]=d;counts[action]+=1
            if action in ('add','retain','rebind'):
                require(sid in by_id,'Missing included delta observation');i=by_id[sid]
                require(apaths[i]==d['current_native_id'],'Delta/payload entity mismatch')
                if action=='add':
                    require(sid not in old_id,'Added observation already exists')
                    require(int(a['source_rows'][i])==d['source_row'] and int(a['pts'][i])==d['pts'],'Added identity/time mismatch')
                else:
                    require(sid in old_id,'Retained observation absent from live data');j=old_id[sid]
                    require(oldpaths[j]==d['previous_native_id'],'Live ad association differs from delta baseline')
                    for key in ('source_rows','pts','uv','kind','stations'):
                        require(np.array_equal(a[key][i],z[key][j]),'Retained source field changed: '+key)
            else:
                require(action in ('exclude_before_new_start','remove_redundant_ad','remove_unverified_ad'),'Unknown delta operation')
                require(sid not in by_id,'Removed advertisement returned')
    require(dict(counts)==manifest['operations'],'Delta count mismatch')
    require(set(by_id)<=set(actions) and all(str(z['observation_ids'][i]) in actions for i in removed),'Incomplete delta coverage')
    # Keep the current non-ad records, not the stale non-ad copies in combined_input.
    full={k:np.concatenate((z[k][keep],mapped[k])) for k in z}
    require(len(np.unique(full['observation_ids']))==len(full['pts']) and len(np.unique(full['source_rows']))==len(full['pts']),'Duplicate combined identity')
    require(np.array_equal(np.unique(full['pts']),expected),'Source exposure coverage changed')
    for key in z:require(equal(full[key][:len(keep)],z[key][keep]),'Live non-ad field changed: '+key)
    allids=set(full['observation_ids']);keep_ids=set(z['observation_ids'][keep])
    reviewed=prep.read(prep.RUN/'ad4_contact_review.json')['decisions']
    for d in reviewed:
        require(d['observation_id'] not in by_id,'Known Ad4 contact point returned as ad')
        require(not np.any((a['pts']==d['pts'])&np.all(a['uv']==d['uv'],axis=1)),'Known Ad4 pixel returned under another ID')
        if d['action']=='rebind':
            require(d['observation_id'] in keep_ids and oldpaths[old_id[d['observation_id']]]==d['new_path_id'],'Contact rebind lost')
        else:require(d['observation_id'] not in allids,'Excluded Ad4 point returned')
    ice=prep.read(Path(__file__).with_name('ice_identity_review_20261026.json'))
    for d in ice['decisions']:require(d['observation_id'] not in allids,'Excluded ice identity returned')
    # Read only final selected source records. Diagnostic profiles/thresholds are
    # provenance and never become numerical fields or weights.
    seen=set();checked=0
    with gzip.open(increment/'evidence/pointwise_checks.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);sid=d['stable_id']
            if sid not in by_id:continue
            i=by_id[sid];require(sid not in seen,'Duplicate pointwise source record');seen.add(sid)
            require(d['decision'] in ('include_original_pixel','new_actual_source_boundary') and d['fit_sampling']['selected'],'Unselected source record returned')
            require(d['path_id']==apaths[i] and int(d['source_row'])==int(a['source_rows'][i]) and int(d['pts'])==int(a['pts'][i]),'Source evidence identity mismatch')
            require(np.array_equal(d['uv'],a['uv'][i]) and int(d['kind'])==int(a['kind'][i]),'Source evidence pixel/type mismatch')
            require(len(d['source_BGR_sha256'])==64,'Missing source image identity');checked+=1
    require(checked==len(a['pts']),'Missing selected source evidence')
    ad_stations=ai['stations'].copy();reused=0
    for sid,i in by_id.items():
        if actions[sid]['action']=='retain':ad_stations[i]=state['stations'][old_id[sid]];reused+=1
    fixed=a['kind']==0
    require(np.array_equal(ad_stations[fixed],a['stations'][fixed]),'Fixed material corner station changed')
    stage.mkdir(exist_ok=True)
    np.savez(stage/'point_observations.npz',**full)
    np.savez(stage/'initialized_points.npz',stations=np.r_[initial[keep],ad_stations])
    sha=prep.digest(stage/'point_observations.npz')
    rebound={k:v.copy() for k,v in state.items()}
    rebound.update(stations=np.r_[state['stations'][keep],ad_stations],source_rows=full['source_rows'],point_input_sha256=np.array(sha))
    for key in state:
        if key not in ('stations','source_rows','point_input_sha256'):require(equal(rebound[key],state[key]),'Native parameter state changed: '+key)
    np.savez(stage/'rebound_numeric_state.npz',**rebound)
    mapping=dict(m,schema='rinkspace.live_native_ad_increment.v1',selected_rows=len(full['pts']),frames=len(expected),
        current_point_input=str(prep.RUN/'point_observations.npz'),advertisement_increment_id=manifest['increment_id'],native_parameters_replaced_by_dataset=False)
    prep.write(stage/'mapping.json',mapping)
    receipt=dict(schema='rinkspace.live_native_ad_increment.v1',input=str(prep.RUN/'point_observations.npz'),sha256=sha,
        prior_sha256=hashes['point_observations.npz'],advertisement_baseline_id=manifest['baseline_id'],increment_id=manifest['increment_id'],
        package_manifest=str(increment/'increment_manifest.json'),package_manifest_sha256=prep.digest(increment/'increment_manifest.json'),
        attached_delta=str(delta),attached_delta_sha256=prep.digest(delta),rows=len(full['pts']),frames=len(expected),
        old_ad_rows=len(removed),new_ad_rows=len(a['pts']),unchanged_nonad_rows=len(keep),retained_ad_numeric_stations=reused,
        selected_ad_source_records_checked=checked,operations=dict(counts),source_scope=manifest['source_scope'],
        fields=sorted(prep.POINT_FIELDS),weights_used=False,covariance_used=False,error_allowance=False,robust_loss=False,
        nonad_values_initializers_and_numeric_stations_unchanged=True,native_parameters_replaced_by_dataset=False,
        global_native_parameter_arrays_unchanged=list(replacement.GLOBALS),excluded_ice_points_still_absent=len(ice['decisions']),
        ad4_contact_rebindings_preserved=sum(d['action']=='rebind' for d in reviewed),ad4_withdrawals_still_absent=sum(d['action']!='rebind' for d in reviewed),
        fixed_material_corners=int(fixed.sum()),advertisement_frames=len(np.unique(a['pts'])),
        excluded_from_numerical_input=['seed_screen_distance','source_profiles','sampling_tolerances','observability_thresholds','package_native_parameters'],
        coverage_limitations=prep.read(increment/'evidence/coverage_gaps.json'),visual_accuracy_accepted=False,optimizer_steps=0)
    prep.write(stage/'point_input.json',receipt);prep.load_pixel_input(stage)
    replacement.verify_guards(guarded)
    require(all(prep.digest(prep.RUN/n)==v for n,v in hashes.items()),'Live input changed during preparation')
    prep.write(stage/'replacement_receipt.json',dict(stage='prepared',current_file_hashes=hashes,protected_files=guarded,input_receipt=receipt,
        prepared_file_hashes={n:prep.digest(stage/n) for n in ('point_observations.npz','initialized_points.npz','mapping.json','point_input.json','rebound_numeric_state.npz')}))
    print(json.dumps({k:v for k,v in receipt.items() if k!='coverage_limitations'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','install']);p.add_argument('--increment',type=Path,required=True);p.add_argument('--delta',type=Path);a=p.parse_args()
    manifest=prep.read(a.increment/'increment_manifest.json')
    require(manifest['increment_id']=='s01_ads_checked_increment002_20261026','Unreviewed increment revision')
    replacement.STAGE=prep.RUN/'ad_increment002'
    prepare(a.increment,a.delta or a.increment/'delta_operations.jsonl.gz') if a.action=='prepare' else replacement.install()
