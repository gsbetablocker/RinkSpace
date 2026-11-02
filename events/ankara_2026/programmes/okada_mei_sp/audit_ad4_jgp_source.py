"""Read Ad4 JGP directly from original frames and compare existing source rows.

This audit does not change a prepared input or run an optimizer. Registration
only supplies searches; the public RGB transition reader supplies the pixels.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json
import cv2
import numpy as np
from programme_source_frames import iter_registration_source_frames
from capture_ad4_source_repair import read, sha, source_translation_seed
from capture_independent_ad4_glyph_gap import read_public, save_profiles, safe, write
from rinkspace.session.print_patterns import PrintPattern
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.path_policy import guard_project_write
from audit_ad4_source_correspondence import source_station, inverse_photo
import prepare_native_multicamera as prepared


def main(a):
    out=guard_project_write(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cv2.setNumThreads(2)
    m=read(a.prepared/'joint_manifest.json');mapping=read(a.prepared/'mapping.json')
    points=prepared.npz(a.prepared/'point_observations.npz');state=prepared.npz(a.state)
    scene_path=Path(m['scene_path']);scene=read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,m,points)
    if sha(a.native_library)!=str(state['native_library_provenance_sha256']):raise ValueError('DLL differs')
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4)
    asset=read((scene_path.parent/ref['path']).resolve());template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    native_paths={p['id']:p for p in template['paths']}
    atlas=read(a.reference);source=cv2.imread(atlas['source_image'])
    if sha(atlas['source_image'])!=atlas['image_sha256']:raise ValueError('Reference image changed')
    usable=deepcopy(atlas);usable['paths']=[p for p in usable['paths'] if p.get('proof',{}).get('reference_image_has_unoccluded_this_contour',True)]
    pattern=PrintPattern(usable,source)
    mask_ref=atlas.get('reference_exclusion_mask')
    if mask_ref:
        path=mask_ref['path'] if isinstance(mask_ref,dict) else mask_ref
        exclusion=cv2.imread(str(path),0)
        pattern.mask[cv2.dilate((exclusion>0).astype('uint8'),np.ones((7,7),'uint8'))>0]=0
    core=NativeCore(a.native_library);parameter=ParameterState(model.initial_state.layout,state['scene_values'])
    board=parameter.apply_scene(scene)['board'];timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    shot=next(s for s in m['shots'] if s['id']==a.shot);reference=[];world=[]
    for p in usable['paths']:
        q=np.asarray(p['pixels']);path=native_paths[p['path_id']];u=np.asarray(path['material_uv'])
        expected=len(u)+(0 if np.array_equal(u[0],u[-1]) or not path['closed'] else 1)
        if len(q)!=expected:raise ValueError('Source/template vertex order differs')
        take=np.arange(0,len(q),max(1,len(q)//30));s=take/(len(q)-1)
        reference.extend(q[take]);world.extend(evaluate_print_path(asset,'print_instance/Ad4',p['path_id'],board,core,s,parameter_state=parameter,jacobian=False)['xyz'])
    reference=np.asarray(reference);world=np.asarray(world);reports=[]
    if a.saved_readouts:
        source_paths={p['path_id']:p for p in usable['paths']}
        for pt in a.pts:
            saved=read(a.saved_readouts/f'{pt}.json');H=np.asarray(saved['registration']['H']);result=[]
            clock=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
            for row in saved['rows']:
                name=row['path_id'].split('::')[-1];vertices=np.asarray(source_paths[name]['pixels'])
                query=inverse_photo(H,row['search_uv'])
                s=source_station(vertices,query,source_paths[name]['source_closed'],0.)
                xyz=evaluate_print_path(asset,'print_instance/Ad4',name,board,core,[s['station']],parameter_state=parameter,jacobian=False)['xyz']
                prediction=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,xyz)
                if prediction['status'][0]!=0:raise ValueError('Invalid source-query projection')
                delta=prediction['uv'][0]-row['uv'];normal=np.asarray(row['normal'])
                result.append(dict(path_id=row['path_id'],measured_uv=row['uv'],predicted_uv=prediction['uv'][0].tolist(),
                    residual_px=delta.tolist(),source_normal_residual_px=float(delta@normal),query_source_edge_distance_px=s['source_edge_distance_px']))
            residuals=np.asarray([r['residual_px'] for r in result]);normal=np.asarray([r['source_normal_residual_px'] for r in result])
            reports.append(dict(pts=pt,rows=result,signed_median=np.median(residuals,0).tolist(),
                same_source_query_error_p50_p95=np.percentile(np.linalg.norm(residuals,axis=1),[50,95]).tolist(),
                source_normal_absolute_error_p50_p95=np.percentile(abs(normal),[50,95]).tolist()))
        write(out/'report.json',dict(frames=reports,state_sha256=sha(a.state),saved_readouts=str(a.saved_readouts),optimizer_started=False))
        print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in reports]),flush=True)
        return
    for frame in iter_registration_source_frames(a.shot,bootstrap_root=a.bootstrap,selected_pts=a.pts):
        pt=frame['pts'];image=frame['image'];mask=frame['source_exclusion_mask']
        clock=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
        projection=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,world)
        good=(projection['status']==0)&np.isfinite(projection['uv']).all(1)
        H,_=cv2.findHomography(reference[good],projection['uv'][good],0)
        seed=source_translation_seed(pattern,image,H)
        registration=pattern.register(image,mask.astype('uint8'),H if seed is None else seed[0])
        rows=[];captured=None
        if registration['status']=='source_pattern_registered':
            binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,shot_id=a.shot,time_base='1/12800',decoded_native_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image[:,:,::-1]).tobytes()).hexdigest())
            rows,readout,captured=read_public(pattern,image,mask,registration,binding)
            save_profiles(out/f'{pt}_profiles.npz',captured)
        report=dict(pts=int(pt),source_BGR_sha256=frame['source_BGR_sha256'],registration=registration,rows=rows,source_UV_modified=False)
        if rows:
            stats=[]
            for name in sorted({r['path_id'] for r in rows}):
                new=np.asarray([r['uv'] for r in rows if r['path_id']==name]);identity=name if name.startswith('print_instance/') else 'print_instance/Ad4::'+name
                oldix=np.flatnonzero((points['pts']==pt)&(points['path_indices']==mapping['path_ids'].index(identity)))
                old=points['uv'][oldix]
                stats.append(dict(path_id=name,old_rows=len(old),new_rows=len(new),
                    nearest_new_reading_distance_px=[] if not len(old) else np.min(np.linalg.norm(old[:,None]-new,axis=2),axis=1).tolist()))
            report['comparison']=stats
        write(out/f'{pt}.json',safe(report));reports.append(dict(pts=int(pt),rows=len(rows),status=registration['status'],correlation=registration.get('correlation')))
        picture=image.copy()
        for r in rows:cv2.circle(picture,tuple(np.rint(r['uv']).astype(int)),2,(0,0,255),-1)
        xy=np.asarray([r['uv'] for r in rows]) if rows else projection['uv'][good]
        lo=np.maximum(np.floor(xy.min(0)-20).astype(int),[0,0]);hi=np.minimum(np.ceil(xy.max(0)+20).astype(int),[1920,1080])
        cv2.imwrite(str(out/f'{pt}_source_crop.png'),image[lo[1]:hi[1],lo[0]:hi[0]])
        cv2.imwrite(str(out/f'{pt}_points_crop.png'),picture[lo[1]:hi[1],lo[0]:hi[0]])
    write(out/'report.json',dict(frames=reports,source_reference=str(a.reference),state_sha256=sha(a.state),source_input_sha256=sha(a.prepared/'point_observations.npz'),optimizer_started=False))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','native-library','reference','bootstrap','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pts',type=int,nargs='+',required=True)
    p.add_argument('--shot',default='s09',choices=['s01','s02','s03','s04','s05','s06','s07','s08','s09'])
    p.add_argument('--saved-readouts',type=Path,help='Reuse actual source readings for CPU-only projection comparison')
    main(p.parse_args())
