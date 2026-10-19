"""Unaccepted full main9 fixed-installation R/f source recovery.

Starts from immutable complete 7625 camera, never previous rejected main9 v2.
Uses the original complete motion admission, exact aliases and finite material.
Private C adapter measures unique x at fixed source y. Original roles/UV stay.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,copy,json,argparse,hashlib
from types import SimpleNamespace
import cv2,numpy as np
P=Path('D:/OneDrive/codex_projs/RinkSpace/three_frame_refit/program_tracking');sys.path.insert(0,str(P))
from continuous_central_brown4_optin import enable_projection_backend
enable_projection_backend()
from z190_main_cut_depth_profile_recovery import prepare,CURRENT,CURRENT_SHA,MANIFEST
from z190_main_zoom_k1_source_information import read,write,ref
from z190_continuous_source_bundle import SOURCE_SHA,HARD_SCENE_SHA
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve
from z190_low_opening_rf_refine import convert_row
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from shared_F_scene_candidate import load_candidate,rebind
from ice_refine import _world_curves
from main9_typed_source_adapter import Main9TypedFixedCutBundle,typed_absolute_value_jac15
N=Path(__file__).parent.parent
HERE=Path(__file__).parent
SCENE=N/'three_installation_shared_F_column_keys_v2_guarded/shared_Fx_fixed_height/candidate_scene_concentric_DIAGNOSTIC_ONLY.json'
SCENE_SHA='427806726d1cae8e5a276bf94acd9c4551296f96cf6987da295b0c5940db01bb'
PACKETS=[HERE/'main9_ice_source/main9_R3_R4_external_boundary_packet_qualified.json',HERE/'main9_ice_source/main9_C_C0_paint_support_packet_qualified.json',N/'main9_board_ice_key_source_v1/core_key_visible_board_edges_source.json']

def new_rows(aliases,original,scene):
    result=[];bindings=[];worlds=_world_curves(original)
    for path in PACKETS:
        packet=read(path);source_sha=packet.get('source_video_sha256',packet.get('source_sha256'))
        if source_sha!=SOURCE_SHA or packet['scene_sha256']!=HARD_SCENE_SHA or packet.get('source_reading_uses_camera_or_model'):
            raise ValueError('New source must bind unchanged same-event source/native readings and hard93c identity')
        bindings.append(ref(path))
        for g in packet['groups']:
            row=convert_row(copy.deepcopy(g));native=int(row['source_native_pts']);row['pts']=aliases[native]
            if row['split']not in('fit','held'):raise ValueError('Explicit source role required')
            row['world']=np.asarray(row['world'],float);row['uv']=np.asarray(row['uv'],float)
            if 'source_normal_uv'in row:row['normal']=np.asarray(row['source_normal_uv'],float)
            if 'normal_interval_px'in row:row['normal_interval']=np.asarray(row['normal_interval_px'],float)
            if 'sigma'not in row:row['sigma']=np.full(len(row['uv']),float(row.get('sigma_px',row.get('sigma_source_choice_px'))))
            if row.get('point_correspondence'):raise ValueError('New paint/board source is curve/interval only')
            if row['id']in worlds and not np.allclose(row['world'],worlds[row['id']],rtol=0,atol=2e-14):raise ValueError('Source physical complete world curve differs from hard93c')
            row['origin']=str(path);row['original_native_uv_sha256']=hashlib.sha256(row['uv'].tobytes()).hexdigest()
            if row['id']=='C':
                row.update(curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='y',scanline_branch='unique_x',fit_axes=['x'],source_y_role='source horizontal scanline label, no fitted dy')
                result.append(rebind(row,original,scene));continue
            if row.get('curve_observation_kind')=='horizontal_extent':
                result.append(rebind(row,original,scene));continue
            # R3/R4 groups contain source-explicit upper and lower samples.
            # Split observations by that immutable arc annotation, no UV change.
            arcs=[s.get('arc')for s in row.get('samples',[])]
            if row['id'].endswith('_outer')and arcs and all(a in('upper','lower','C0_upper','C0_lower')for a in arcs):
                for arc in sorted(set(arcs)):
                    ix=np.flatnonzero(np.array(arcs)==arc);r=copy.deepcopy(row)
                    for key in('uv','normal','normal_interval','sigma'):r[key]=row[key][ix].copy()
                    r['samples']=[copy.deepcopy(row['samples'][i])for i in ix];upper=arc.endswith('upper')
                    r.update(curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',scanline_branch='min_y'if upper else'max_y',must_preserve_upper_branch=upper,source_region=arc)
                    result.append(rebind(r,original,scene))
            else:result.append(rebind(row,original,scene))
    return result,bindings

def source_errors(bundle,q,groups):
    records=[]
    for p,rr in groups:
        for row in rr:
            residual=typed_absolute_value_jac15(bundle.camera(q,p),[row],False).reshape(-1,2)*row['sigma'][:,None]
            records.append({'actual_pts':p,'native_source_pts':row.get('source_native_pts',row.get('pts',p)),'id':row['id'],'source_region':row.get('source_region'),'split':row.get('split','original_fit'),'origin':row.get('origin'),'kind':row.get('curve_observation_kind','original'),'sample_count':len(row['uv']),'outside_support_xy_px':residual.tolist(),'outside_support_norm_median_px':float(np.median(np.linalg.norm(residual,axis=1))),'outside_support_norm_max_px':float(np.max(np.linalg.norm(residual,axis=1)))})
    return records

def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);originalout=out/'original_source_adapter';originalout.mkdir()
    cv2.setNumThreads(1)
    if ref(CURRENT)['sha256']!=CURRENT_SHA or ref(SCENE)['sha256']!=SCENE_SHA:raise ValueError('Fixed current camera/conditional scene changed')
    fd=read(HERE/'main9_ice_source/typed_y_unique_x_audit/actual_source_C_typed_unique_x_FD_domain.json')
    if fd['private_adapter_sha256']!=ref(HERE/'main9_typed_source_adapter.py')['sha256']:raise ValueError('Actual source typed C adapter must pass current FD before solve')
    prep=SimpleNamespace(current=CURRENT,cut=9,output=originalout,minimum_unknown_depth_m=1.,new_ice_source=None,new_ice_motion_source=None)
    old,q,model,native,aliases,views,cal,authority,oldmanifest,_=prepare(prep)
    original,scene,binding=load_candidate(SCENE,Path(read(MANIFEST)['scene']['path']));write(out/'candidate_scene_world_source_binding.json',binding)
    rows,bindings=new_rows(aliases,original,scene)
    original_groups=[(p,[rebind(r,original,scene)for r in rr])for p,rr in old.abs_groups]
    observed={p:list(rr)for p,rr in original_groups};held={}
    for row in rows:
        p=row['pts']
        if p not in old.index:raise ValueError('New source crosses complete cut9')
        (observed if row['split']=='fit'else held).setdefault(p,[]).append(row)
    groups=[(p,observed[p])for p in old.stamps if p in observed];heldgroups=[(p,held[p])for p in old.stamps if p in held]
    absolute_sigma=np.concatenate([np.concatenate([np.repeat(r['sigma'],2)for r in rr])for _,rr in groups])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,absolute_sigma])
    bundle=Main9TypedFixedCutBundle(base,cal,old.motion_kernel,reference_pts=old.reference_pts,initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,finite_planes=old.finite_planes,finite_records=old.finite_records,scene=scene)
    manifest=dict(schema='rinkspace.main9.full-original-source-typed-C-C0-fixed-mount-Rf-recovery.v1',current=ref(CURRENT),candidate_scene=ref(SCENE),source_world_binding=binding,original_source=ref(originalout/'input_manifest.json'),new_source_packets=bindings,actual_exposures=len(bundle.stamps),original_motion_rows=oldmanifest['original_motion_rows'],active_original_motion_rows=len(bundle.a),previous_finite_material_rows=len(bundle.finite_records),new_fit_samples=sum(len(r['uv'])for r in rows if r['split']=='fit'),new_held_samples=sum(len(r['uv'])for r in rows if r['split']=='held'),held_pending_future_not_fit=True,source_UV_and_original_roles_unchanged=True,current_fixed_main_pivot_lever_PP_k1=True,all_other_cut_views_unchanged=True,scene_conditional_shared_F_translation_only=True,source_sigma_not_statistical=True,original_motion_weight_unchanged=True,private_typed_adapter=ref(HERE/'main9_typed_source_adapter.py'),C_exact_y_role='fixed source scanline; x only',FD_domain_audit=ref(HERE/'main9_ice_source/typed_y_unique_x_audit/actual_source_C_typed_unique_x_FD_domain.json'),accuracy_accepted=False,video_generated=False)
    write(out/'input_manifest.json',manifest)
    serial=[]
    for r in rows:
        serial.append({k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in r.items()})
    write(out/'admitted_new_source_rows.json',serial)
    write(out/'baseline_new_source_errors.json',source_errors(bundle,q,[(p,[r for r in rr if r.get('origin')in [str(b)for b in PACKETS]])for p,rr in groups]+heldgroups))
    np.savez_compressed(out/'initial_current_state.npz',q=q,canonical_pts=bundle.stamps)
    print(json.dumps({'stage':'typed_main9_complete_source_prepared','actual_exposures':len(bundle.stamps),'active_motion_rows':len(bundle.a),'new_fit_samples':manifest['new_fit_samples'],'new_held_samples':manifest['new_held_samples']}),flush=True)
    if not args.solve:return
    fitted,history,seconds=fixed_mount_rf_solve(bundle,q,out)
    if not np.array_equal(fitted[:4],q[:4]):raise ValueError('Fixed main mechanical installation changed')
    updated=dict(views);updated.update({p:CanonicalView(views[p].installation_id,bundle.view(fitted,p))for p in bundle.stamps});target=out/'complete_main9_typed_C_C0_camera_UNACCEPTED.npz';model.save_npz(target,native,aliases,updated,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,nn,aa,vv=PhysicalCameraModel.load_npz(target);untouched=[p for p in views if p not in bundle.index]
    if not np.array_equal(nn,native)or aa!=aliases or not all(np.array_equal(saved.camera25(vv[p]),model.camera25(views[p]))for p in untouched):raise ValueError('Untouched programme cut/source alias changed')
    write(out/'candidate_new_source_errors.json',source_errors(bundle,fitted,[(p,[r for r in rr if r.get('origin')in [str(b)for b in PACKETS]])for p,rr in groups]+heldgroups))
    write(out/'report.json',dict(manifest,candidate=ref(target),iterations=len(history),solve_seconds=seconds,untouched_actual_exposures=len(untouched),disposition='all-model whole-source visual review required UNACCEPTED'))
    print(json.dumps({'stage':'typed_main9_complete_source_candidate_saved','path':str(target),'seconds':seconds}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--solve',action='store_true');run(parser.parse_args())
