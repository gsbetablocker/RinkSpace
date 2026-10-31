"""Measure actual RGB lines inside independently read original-image searches.

Reference polylines locate searches only. Native identities are explicit source
interpretations, retained with their evidence; no projected or clicked pixel is
ever emitted as an observation. Each measured pair retains both RGB interfaces.
"""
from pathlib import Path
import argparse, hashlib, json
import cv2, numpy as np
from capture_independent_badge import exact_frames
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,plain,write
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import qualify_visible_line_fragments


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(args):
    if args.output.exists():raise FileExistsError(args.output)
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'))
    manifest=json.loads((args.run_root/'joint_manifest.json').read_text('utf8'))
    references=json.loads(args.references.read_text('utf-8-sig'))
    state=dict(np.load(args.state,allow_pickle=False));library=Path(manifest['native_library'])
    if digest(library)!=str(state['native_library_provenance_sha256']):raise ValueError('Native provenance changed')
    scene_path=Path(manifest['scene_path']);scene=json.loads(scene_path.read_text('utf8'))
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    sampled=sample_entities(ParameterState(model.initial_state.layout,state['scene_values']).apply_scene(scene),NativeCore(library))
    inventory={p['curve_id']:p for p in native_ice_inventory(sampled['cuts'],sampled['xyz'])}
    images=exact_frames(Path(index['source']['resolved_path']),[r['pts']for r in references['frames']])
    args.output.mkdir(parents=True);cv2.setNumThreads(2)
    write(args.output/'binding.json',dict(schema='rinkspace.independent_source_line_references.v1',references=str(args.references),
        references_sha256=digest(args.references),source_sha256=index['source']['actual_sha256'],
        native_scene=str(scene_path),native_scene_sha256=digest(scene_path),state=str(args.state),state_sha256=digest(args.state),
        adapter_sha256=digest(__file__),native_paths=[{k:v for k,v in p.items()if k!='xyz'}for p in inventory.values()],
        previous_new_shot_observations_read=False,source_search_is_not_measurement=True,fit_eligible=False))
    streams={}
    try:
        for reference in references['frames']:
            sid=reference['shot'];pts=reference['pts'];folder=args.output/sid;folder.mkdir(exist_ok=True)
            if sid not in streams:streams[sid]=(folder/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)
            image=images[pts];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);mask,mask_path=source_exclusion(root,sid,pts)
            rows=[];reports=[];proof=image.copy()
            for j,feature in enumerate(reference['features']):
                path=dict(inventory[feature['curve_id']]);path['material_signal']=feature.get('signal',path['material_signal'])
                reader=IceProgrammeSourceReader(None,source_sha256=index['source']['actual_sha256'],time_base='1/12800',
                    native_paths=[path],sample_spacing_px=24.,search_radius_px=feature.get('radius',32.),separate_bands=True,
                    prepared_source_context=dict(source_sha256=index['source']['actual_sha256'],time_base='1/12800',authority=[dict(path=str(scene_path),sha256=digest(scene_path))]))
                guide=np.asarray(feature['search_uv'],float)
                if guide.ndim!=2 or guide.shape[1]!=2 or len(guide)<2:raise ValueError('Expected a finite source search polyline')
                z=reader.read(rgb,pts,None,mask,shot_id=sid,native_projections={path['native_path_id']:dict(uv=guide,valid=np.ones(len(guide),bool))})
                fragments,report=qualify_visible_line_fragments(z,[0],mask)
                raw=folder/f'{pts}_{j}_readout.npz';np.savez_compressed(raw,**{k:v for k,v in z.items()if isinstance(v,np.ndarray)})
                ids=sorted(set(i for f in fragments for i in f['candidate_ids']))
                ids=sorted(ids,key=lambda i:int(z['candidate_query'][i]))
                chosen=[ids[k]for k in spaced_indices(z['candidate_uv'][ids])]
                for i in chosen:
                    row=dict(path_id=path['native_path_id'],entity=path['curve_id'],uv=z['candidate_uv'][i].tolist(),
                        actual_source_sides=z['candidate_sides'][i].tolist(),actual_source_quantiles=z['candidate_side_quantiles_uv'][i].tolist(),
                        measurement_operator='paired_material_boundary_midline',capture_family='independent_source_line_reference',
                        model_UV_synthesized=False,source_identity_basis=feature['identity_basis'],source_reference=str(args.references),
                        raw_readout=str(raw),raw_candidate_id=i,material_signal=path['material_signal'])
                    rows.append(row);cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,200,0),-1)
                if chosen:
                    xy=z['candidate_uv'][chosen[len(chosen)//2]];cv2.putText(proof,path['curve_id'],tuple(np.rint(xy+[0,-10]).astype(int)),0,.5,(0,150,0),1,cv2.LINE_AA)
                reports.append(dict(feature=feature,rows=len(chosen),fragments=fragments,qualification=report))
            streams[sid].write(json.dumps(plain(dict(pts=pts,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                source_video_sha256=index['source']['actual_sha256'],mask_path=str(mask_path),mask_sha256=digest(mask_path),
                rows=rows,fit_eligible=False,review_state='pending_source_material_and_identity_review')),ensure_ascii=False)+'\n')
            write(folder/f'{pts}_diagnostic.json',dict(pts=pts,features=reports))
            cv2.imwrite(str(folder/f'{pts}_source_candidates.jpg'),proof)
            print(json.dumps(dict(shot=sid,pts=pts,rows=len(rows),features=[(r['feature']['curve_id'],r['rows'])for r in reports])),flush=True)
    finally:
        for stream in streams.values():stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','run-root','state','references','output'):p.add_argument('--'+key,type=Path,required=True)
    capture(p.parse_args())
