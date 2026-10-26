"""Apply an explicit source identity review without replacing native values."""
from pathlib import Path
import argparse, importlib.util, io
import numpy as np

spec=importlib.util.spec_from_file_location('replacement',Path(__file__).with_name('replace_native_s01_ads.py'))
replacement=importlib.util.module_from_spec(spec);spec.loader.exec_module(replacement)
prep=replacement.prep
require=replacement.require
STAGE=prep.RUN/'ad_increment002/ad4_binding_review/input'
REVIEW=Path(__file__).with_name('ad4_boundary_identity_review_20261026.json')


def load_snapshot(path):
    # Release the Windows file handle before any GPU work or evidence inspection.
    with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as z:
        return {key:z[key] for key in z.files}


def prepare():
    require((prep.RUN/'stop_requested').exists(),'Stop the live solver before preparing a replacement')
    require(not (STAGE/'replacement_receipt.json').exists(),'Prepared repair already exists')
    review=prep.read(REVIEW)
    hashes={name:prep.digest(prep.RUN/name) for name in replacement.CURRENT_FILES}
    require(hashes['point_observations.npz']==review['before_point_input_sha256'],'Review belongs to another input')
    guarded=replacement.native_guards()
    z,mapping,initial=prep.load_pixel_input(prep.RUN)
    snapshot=load_snapshot(prep.RUN/'accepted_pixels_latest.npz')
    require(str(snapshot['point_input_sha256'])==hashes['point_observations.npz'],'State/input mismatch')
    require(replacement.equal(snapshot['source_rows'],z['source_rows']),'State row mismatch')
    ids={str(s):i for i,s in enumerate(z['observation_ids'])}
    paths=np.asarray(mapping['path_ids'])[z['path_indices']]
    keep=np.ones(len(paths),bool);changes=[]
    for decision in review['decisions']:
        i=ids[decision['observation_id']]
        require(int(z['source_rows'][i])==decision['source_row'] and int(z['pts'][i])==decision['pts'],'Review row/time changed')
        require(np.array_equal(z['uv'][i],decision['uv']) and paths[i]==decision['old_path_id'],'Review pixel/identity changed')
        require(int(z['kind'][i])==decision['kind']==1,'These reviewed points are curves, not corners')
        if decision['action']=='exclude':keep[i]=False
        else:changes.append((i,decision['new_path_id']))

    core=replacement.NativeCore(prep.LIBRARY)
    model=replacement.SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    state=replacement.ParameterState(model.initial_state.layout,snapshot['scene_values'])
    native=replacement.NativeSceneGPU(prep.SCENE,core,parameter_state=state)
    camera={**prep.camera_for_snapshot(prep.read(prep.CAMERA),snapshot),'rig':snapshot['rig'].tolist()}
    full={key:value.copy() for key,value in z.items()};seeds=initial.copy();numeric=snapshot['stations'].copy();station_records=[]
    for i,target in changes:
        require(target in native.paths,'Reviewed target absent from current native')
        # Identity is already established from source pixels. This only initializes
        # the numerical coordinate on that particular native curve.
        low,high=0.,1.
        if target.startswith('board/'):
            same=np.flatnonzero((paths==target)&(z['pts']==z['pts'][i]))
            require(len(same)>1,'No same-exposure board segment for station initialization')
            low=float(snapshot['stations'][same].min());high=float(snapshot['stations'][same].max())
        ts=np.linspace(low,high,4097)
        with native.resident(camera=camera,controls=snapshot['controls'],path_ids=[target]*len(ts),stations=ts,
                             pts=np.full(len(ts),z['pts'][i],np.int64),observed=np.zeros((len(ts),2))) as problem:
            evaluation=problem.evaluate('cuda')
        # A complete ice curve can leave the image, pass behind the camera,
        # or fail the rolling-shutter contraction proof outside its visible part.
        # Search each valid connected projected piece, never bridge an invalid
        # span. These samples only initialize an already source-confirmed ID;
        # every actual observation still has to pass full native validation.
        status=evaluation['status'];valid=status==0
        require(np.all(np.isin(status,[0,4,5,8])),'Unexpected target projection failure')
        pieces=[]
        for first,last in np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2):
            if last-first>1:
                station,distance=prep.closest_polyline(z['uv'][i:i+1],evaluation['values'][first:last,3:5],ts[first:last])
                pieces.append((float(distance[0]),float(station[0])))
        require(bool(pieces),'No valid connected target curve projection')
        distance,s=min(pieces);full['path_indices'][i]=mapping['path_ids'].index(target)
        full['stations'][i]=s;seeds[i]=s;numeric[i]=s
        station_records.append(dict(observation_id=str(z['observation_ids'][i]),new_path_id=target,
                                    initializer=s,initializer_distance_px=distance,
                                    projected_samples=int(valid.sum()),invalid_projection_samples_by_status={str(int(code)):int(count) for code,count in zip(*np.unique(status[~valid],return_counts=True))}))

    full={key:value[keep] for key,value in full.items()}
    unchanged=keep.copy();unchanged[[i for i,_ in changes]]=False
    for key in prep.POINT_FIELDS-{'path_indices','stations'}:
        require(replacement.equal(full[key],z[key][keep]),'Immutable point field changed: '+key)
    require(np.array_equal(seeds[unchanged],initial[unchanged]) and np.array_equal(numeric[unchanged],snapshot['stations'][unchanged]),'Unreviewed stations changed')
    STAGE.mkdir(parents=True,exist_ok=True)
    np.savez(STAGE/'point_observations.npz',**full)
    np.savez(STAGE/'initialized_points.npz',stations=seeds[keep])
    sha=prep.digest(STAGE/'point_observations.npz')
    rebound={key:value.copy() for key,value in snapshot.items()}
    rebound.update(stations=numeric[keep],source_rows=full['source_rows'],point_input_sha256=np.array(sha))
    for key in snapshot.keys()-{'stations','source_rows','point_input_sha256'}:
        require(replacement.equal(rebound[key],snapshot[key]),'Native numerical state changed: '+key)
    np.savez(STAGE/'rebound_numeric_state.npz',**rebound)
    mapping=dict(mapping,selected_rows=len(full['pts']),source_identity_review=str(REVIEW))
    prep.write(STAGE/'mapping.json',mapping)
    final_paths=np.asarray(mapping['path_ids'])[full['path_indices']]
    ads=np.char.startswith(final_paths,'print_instance/')
    receipt=dict(schema='rinkspace.live_native_source_identity_repair.v1',input=str(prep.RUN/'point_observations.npz'),sha256=sha,
        prior_sha256=hashes['point_observations.npz'],prior_input_receipt=str(STAGE/'previous/point_input.json'),
        increment_id=review.get('review_id','s01_ads_checked_increment002_20261026'),source_identity_review=str(REVIEW),source_identity_review_sha256=prep.digest(REVIEW),
        rows=len(full['pts']),frames=len(np.unique(full['pts'])),new_ad_rows=int(ads.sum()),nonad_rows=int((~ads).sum()),
        rebound_rows=len(changes),excluded_rows=int((~keep).sum()),unchanged_nonad_rows=int(np.sum(unchanged&~np.char.startswith(paths,'print_instance/'))),
        fields=sorted(prep.POINT_FIELDS),original_UV_PTS_ID_preserved=True,weights_used=False,covariance_used=False,error_allowance=False,robust_loss=False,
        native_parameters_replaced_by_dataset=False,global_native_parameter_arrays_unchanged=list(replacement.GLOBALS),
        unreviewed_points_and_initializers_and_numeric_stations_unchanged=True,visual_accuracy_accepted=False,optimizer_steps=0)
    prep.write(STAGE/'point_input.json',receipt)
    prep.write(STAGE/'station_initialization.json',station_records)
    prep.load_pixel_input(STAGE)
    replacement.verify_guards(guarded)
    require(all(prep.digest(prep.RUN/name)==value for name,value in hashes.items()),'Live files changed during preparation')
    prep.write(STAGE/'replacement_receipt.json',dict(stage='prepared',current_file_hashes=hashes,protected_files=guarded,input_receipt=receipt,
        prepared_file_hashes={name:prep.digest(STAGE/name) for name in ('point_observations.npz','initialized_points.npz','mapping.json','point_input.json','rebound_numeric_state.npz')}))
    print(sha,len(full['pts']),len(changes),int((~keep).sum()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','install'])
    parser.add_argument('--run-dir',type=Path);parser.add_argument('--scene-path',type=Path)
    parser.add_argument('--review',type=Path);parser.add_argument('--stage',type=Path)
    args=parser.parse_args()
    if args.run_dir is not None:prep.RUN=args.run_dir.resolve(strict=True)
    if args.scene_path is not None:prep.SCENE=args.scene_path.resolve(strict=True)
    if args.review is not None:REVIEW=args.review.resolve(strict=True)
    if args.stage is not None:STAGE=args.stage.resolve()
    STAGE.relative_to(prep.RUN)
    replacement.STAGE=STAGE
    prepare() if args.action=='prepare' else replacement.install()
