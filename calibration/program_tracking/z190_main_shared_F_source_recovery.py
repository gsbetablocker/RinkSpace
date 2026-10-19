"""Recover complete main-cut R/f against a versioned common F-column scene.

The baseline remains immutable. Every retained original source pixel, role
and sigma is preserved. All named F curves bind one concentric common scene;
held/pending/future observations stay unfitted. No body or video is produced.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from ad_landmarks import AdGeometry
from shared_F_scene_candidate import load_candidate,rebind
from z190_main_cut_depth_profile_recovery import prepare,CURRENT,MANIFEST
from z190_main_zoom_k1_source_information import read,write,ref
from z190_low8_direct_source_rf_recovery import FixedCutBundle,objective_direction_check
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_source_camera_diagnostic import source_errors
from z190_low_opening_rf_refine import convert_row
from z190_continuous_source_bundle import HARD_SCENE_SHA,SOURCE_SHA
from continuous_central_brown4_optin import enable_projection_backend


def added_source(paths,scene,aliases,index):
    rows=[]
    for path in paths:
        packet=read(path)
        if packet['source_video_sha256']!=SOURCE_SHA or not packet.get('source_only_interval_reading'):
            raise ValueError('Native source-only event binding required')
        if 'scene_sha256'in packet:
            if packet['scene_sha256']!=HARD_SCENE_SHA:raise ValueError('Source baseline scene changed')
        else:
            parent=Path(packet['original_lower_packet'])
            if ref(parent)['sha256']!=packet['original_lower_packet_sha256'] or read(parent)['scene_sha256']!=HARD_SCENE_SHA:
                raise ValueError('Append-only source parent binding changed')
            if not packet['append_new_regions_only']:raise ValueError('Source may only append reviewed regions')
        for raw in packet['groups']:
            row=dict(raw);row['pts']=aliases[int(raw['source_native_pts'])]
            if row['pts']not in index:raise ValueError('Source crosses the selected complete cut')
            if raw['split']not in ('fit','held')or raw['point_correspondence']:raise ValueError('Explicit finite curve source roles required')
            world=np.asarray(raw['world'],float)
            if raw['id'].endswith('_outer'):
                feature=next(f for f in scene['ice_features']if f['id']==raw['id'].removesuffix('_outer'))
                center=np.asarray(feature['center_xyz'],float)
                if np.max(np.abs(np.linalg.norm(world[:,:2]-center[:2],axis=1)-feature['radius']))>1e-7 or np.any(world[:,2]!=0.):
                    raise ValueError('Source world must equal its baseline external ring')
            elif raw['id']=='ad7_left_vertical':
                geometry=AdGeometry(scene,7);lo=geometry.corner(geometry.initial,'left_bottom');hi=geometry.corner(geometry.initial,'left_top')
                if not np.allclose(world[[0,-1]],np.stack((lo,hi)),rtol=0,atol=1e-9) and not np.allclose(world[[0,-1]],np.stack((hi,lo)),rtol=0,atol=1e-9):
                    raise ValueError('Ad7 source must use the actual modeled finite left boundary')
                if np.any(world[:,2]<min(lo[2],hi[2])-1e-9)or np.any(world[:,2]>max(lo[2],hi[2])+1e-9):
                    raise ValueError('Ad7 finite world source height changed')
            else:raise ValueError('Unsupported source curve identity')
            row.update(world=world,uv=np.asarray(raw['uv'],float),normal=np.asarray(raw['source_normal_uv'],float),
                normal_interval=np.asarray(raw['normal_interval_px'],float),
                sigma=np.full(len(raw['uv']),float(raw.get('sigma_source_choice_px',raw.get('sigma_px')))),origin=str(path))
            rows.append(convert_row(row))
    return rows


def records(model,views,authority,aliases,index,original,candidate,additional):
    output=[]
    sources=[(role,authority[key])for role,key in [('fit','fit_rows'),('held','held_rows'),('pending','pending_rows')]]
    sources += [('future_unfitted_identity_check',[r for g in authority['untouched_future_groups']for r in g['rows']]),
                ('held_ad2_41300',read(Path(read(MANIFEST)['new41300held']['path'])))]
    sources +=[(f'new_{split}',[r for r in additional if r['split']==split])for split in ('fit','held')]
    for role,rows in sources:
        for raw in rows:
            p=aliases[int(raw['pts'])]
            if p not in index:continue
            row=rebind(convert_row(raw),original,candidate)
            output.append(dict(role=role,pts=int(raw['pts']),id=raw['id'],region=raw.get('source_region'),
                source_uv_sha256=hashlib.sha256(np.asarray(raw['uv'],float).tobytes()).hexdigest(),
                error=source_errors(model.camera25(views[p]),[row])[0]))
    return output


def prepare_bundle(args):
    args.output=args.output.resolve()
    if not args.output.is_relative_to(Path('D:/CodexData').resolve()):raise ValueError('Bulk output stays outside OneDrive')
    args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    source_args=SimpleNamespace(current=args.current,cut=args.cut,output=args.output,minimum_unknown_depth_m=args.minimum_unknown_depth_m,
        new_ice_source=None,new_ice_motion_source=args.new_ice_motion_source,
        source_admission_cache=getattr(args,'source_admission_cache',None))
    old,q,model,native,aliases,views,cal,authority,manifest,_=prepare(source_args)
    original,candidate,binding=load_candidate(args.candidate_scene,Path(manifest['scene']['path']))
    additional=added_source(args.source_packet,original,aliases,old.index)
    observed={p:[rebind(r,original,candidate)for r in rr]for p,rr in old.abs_groups}
    for row in additional:
        if row['split']=='fit':observed.setdefault(row['pts'],[]).append(rebind(row,original,candidate))
    groups=[(p,observed[p])for p in old.stamps if p in observed]
    absolute_sigma=np.concatenate([np.repeat(r['sigma'],2)for p,rr in groups for r in rr])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,
        motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,absolute_sigma])
    bundle=FixedCutBundle(base,cal,old.motion_kernel,reference_pts=old.reference_pts,
        initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,
        scene=candidate,finite_planes=old.finite_planes,finite_records=old.finite_records)
    bundle.source_quadratic_mask=np.r_[np.zeros(bundle.motion_rows,bool),np.concatenate([
        np.full(2*len(r['uv']),r.get('curve_observation_kind')=='horizontal_extent',bool)for p,rr in groups for r in rr])]
    write(args.output/'baseline_actual_source_errors.json',records(model,views,authority,aliases,bundle.index,original,original,additional))
    write(args.output/'new_scene_retained_RF_source_errors.json',records(model,views,authority,aliases,bundle.index,original,candidate,additional))
    manifest=dict(manifest,schema='rinkspace.main-complete-cut-shared-F-scene-source-recovery.v1',
        original_source_preparation=manifest,shared_F_scene_binding=binding,
        additional_source=[ref(p)for p in args.source_packet],
        additional_fit_points=sum(len(r['uv'])for r in additional if r['split']=='fit'),
        additional_held_points=sum(len(r['uv'])for r in additional if r['split']=='held'),
        source_extent_quadratic=True,source_extent_loss_role='verified whole-ring support is not discarded by nearest-arc or robust outlier loss',
        engineering_source_sigma_retained=True,scene_version_shared_for_source_and_render=True,
        other_cut_camera_values_unchanged=True,other_cuts_not_validated_against_new_scene=True,entrypoint=ref(__file__))
    write(args.output/'shared_scene_input_manifest.json',manifest)
    return SimpleNamespace(bundle=bundle,q=q,model=model,native=native,aliases=aliases,views=views,
        authority=authority,original=original,candidate=candidate,additional=additional,manifest=manifest,args=args)


def run(args):
    prepared=prepare_bundle(args)
    if args.prepare_only:return
    bundle,q,model,native,aliases,views=prepared.bundle,prepared.q,prepared.model,prepared.native,prepared.aliases,prepared.views
    authority,original,candidate,additional,manifest=prepared.authority,prepared.original,prepared.candidate,prepared.additional,prepared.manifest
    objective_direction_check(bundle,q,args.output)
    fitted,history,seconds=fixed_mount_rf_solve(bundle,q,args.output,require_undamped_precision=True)
    updated=dict(views);updated.update({p:CanonicalView(views[p].installation_id,bundle.view(fitted,p))for p in bundle.stamps})
    path=args.output/'complete_main_cut_shared_F_camera_UNACCEPTED.npz'
    model.save_npz(path,native,aliases,updated,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,nn,aa,vv=PhysicalCameraModel.load_npz(path)
    other=[p for p in views if p not in bundle.index]
    if aa!=aliases or not np.array_equal(nn,native)or any(not np.array_equal(saved.camera25(vv[p]),model.camera25(views[p]))for p in other):
        raise ValueError('Untouched camera exposure or source alias changed')
    write(args.output/'candidate_actual_source_errors.json',records(saved,vv,authority,aliases,bundle.index,original,candidate,additional))
    write(args.output/'report.json',dict(manifest,camera_candidate=ref(path),solve_seconds=seconds,iterations=len(history),
        body_rebound=False,full_programme_exported=False,accuracy_accepted=False))
    print({'candidate':str(path),'seconds':seconds,'actual_exposures':len(bundle.stamps)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,default=CURRENT);parser.add_argument('--cut',type=int,choices=(1,7,9),required=True)
    parser.add_argument('--candidate-scene',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-packet',type=Path,action='append',default=[])
    parser.add_argument('--new-ice-motion-source',type=Path);parser.add_argument('--minimum-unknown-depth-m',type=float,default=1.)
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
