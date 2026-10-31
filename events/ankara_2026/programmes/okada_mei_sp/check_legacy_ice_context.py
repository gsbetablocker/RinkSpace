"""Check original RGB alternatives against a shared source-shape context.

Run via run_native_entry.py. The shared affine organizes known native shapes;
its coordinates are never observations. This diagnostic does not admit data.
"""
from pathlib import Path
import argparse,json,gzip,hashlib
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.session.native_shape_guidance import register_native_shapes,point_to_polyline
from rinkspace.session.source_shape_context import transition_intersects_path
from rinkspace.session.material_tracks import assemble_fragments


def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def main(args):
    binding=read(args.capture/'binding.json');manifest=read(binding['prepared_manifest_path'])
    state=dict(np.load(binding['state_path']));raw=read(manifest['scene_path'])
    model=SceneParameterModel(raw,Path(manifest['scene_path']),optimize_scene=True)
    ps=ParameterState(model.initial_state.layout,state['scene_values']);core=NativeCore(manifest['native_library'])
    sample=sample_native_review_scene(raw,core,Path(manifest['scene_path']),parameter_state=ps,observed_path_ids=set())
    cuts={c['id']:c for c in sample['cuts']};paths=binding['native_inventory']
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    reviews=read(args.anchors);args.output.mkdir(parents=True,exist_ok=True)
    for frame in reviews['frames']:
        sid,pts=frame['shot'],frame['pts'];record=read(args.capture/sid/f'{pts}.json.gz')
        z=dict(np.load(args.capture/sid/f'{pts}.npz'))
        if record['source_BGR_sha256']!=frame['source_BGR_sha256']:raise ValueError('Source review pixels changed')
        shot=next(s for s in manifest['shots'] if s['id']==sid)
        clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
        p=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,sample['xyz'])
        polylines={};curve_sets={}
        for path in paths:
            cut=cuts[path['native_path_id']];lo,hi=cut['begin'],cut['end']
            valid=(p['status'][lo:hi]==0)&np.isfinite(p['uv'][lo:hi]).all(1)
            runs=np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2)
            pieces=[]
            for part,(a,b) in enumerate(runs):
                if b-a<2:continue
                name=path['curve_id']+'#'+str(part)
                polylines[name]=p['uv'][lo+a:lo+b];pieces.append(name)
            curve_sets[path['curve_id']]=pieces
        observed=[];names=[]
        for group in frame['groups']:
            ids=np.asarray(group['candidate_ids'],int);name=group['entity']
            ci=next(i for i,path in enumerate(paths) if path['curve_id']==name)
            if not np.all(z['query_curve'][z['candidate_query'][ids]]==ci):raise ValueError('Source anchor identity differs')
            pieces=curve_sets[name]
            if not pieces:raise ValueError('Anchor has no finite native support: '+name)
            uv=z['candidate_uv'][ids]
            matches=np.stack([point_to_polyline(uv,polylines[piece]).distance_px for piece in pieces])
            chosen=matches.argmin(axis=0)
            observed.extend(uv);names.extend(pieces[i] for i in chosen)
        fit=register_native_shapes(np.asarray(observed),np.asarray(names),polylines)
        eligible=np.zeros(len(z['candidate_uv']),bool);details={}
        for ci,path in enumerate(paths):
            name=path['curve_id'];ids=np.flatnonzero((z['query_curve'][z['candidate_query']]==ci)&((z['candidate_flags']&3)==0))
            if not len(ids) or not curve_sets[name]:continue
            # For a midline the uncertainty comes from both actually measured
            # flanks. For a material interface both stored sides are identical.
            quantiles=z['candidate_side_quantiles_uv'][ids].mean(axis=1)
            supported=np.zeros(len(ids),bool)
            for piece in curve_sets[name]:
                curve=polylines[piece]@fit.affine[:2,:2].T+fit.affine[:2,2]
                supported|=transition_intersects_path(quantiles,curve)
            eligible[ids[supported]]=True
            details[name]=dict(clean_candidates=len(ids),context_supported=int(supported.sum()))
        # Two distinct measured transitions explaining the same query remain
        # ambiguous. A common shape is not permission to choose a nearest UV.
        unique=np.bincount(z['candidate_query'][eligible],minlength=len(z['query_uv']))==1
        eligible&=unique[z['candidate_query']]
        fragments=assemble_fragments(z,paths,eligible_candidates=eligible)
        selected=sorted({i for f in fragments for i in f['candidate_ids']})
        np.savez_compressed(args.output/f'{sid}_{pts}.npz',eligible=eligible,
            selected_candidate_ids=np.asarray(selected,np.int32),shared_affine=fit.affine)
        note=dict(shot=sid,pts=pts,anchor_groups=frame['groups'],source_BGR_sha256=record['source_BGR_sha256'],
            shared_affine=fit.affine.tolist(),anchor_rms_px=fit.rms_distance_px,details=details,fragments=fragments,
            source_UV_unchanged=True,observations_created=False,fit_eligible=False,
            physical_identity_still_requires_source_and_layout_evidence=True)
        (args.output/f'{sid}_{pts}.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(shot=sid,pts=pts,rms=fit.rms_distance_px,selected=len(selected),
            paths=[(paths[f['curve']]['curve_id'],len(f['candidate_ids'])) for f in fragments])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','anchors','output'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
