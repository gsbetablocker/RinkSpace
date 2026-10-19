"""Opt-in full secondary restoration with one shared physical board height.

All camera pivots in XY, nonzero lever, optics, rink XY and paint stay fixed.
The board height is an image-conditioned estimate, not a new measurement.
Every true top-edge source and rendered board mesh receives the same height;
printed advertisements retain their original independent world Z values.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
from rinkspace.storage import AssetStore
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import csr_matrix,hstack
from z190_secondary_confirmed_nearboard_source import (
    prepare_full_secondary_confirmed_nearboard,typed_nearboard_absolute_value_jac15,NEAR_KIND)
from z190_nearboard_analytic_source import analytic_nearboard_absolute_value_jac15
from z190_secondary_connected_cap_source import connected_lower_left_cap,connected_cap_absolute_value_jac15
from z190_secondary_qualified_core_source import source_records
from z190_secondary_nonzero_mount_full_source import write,ref,IDENTIFIER
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_joint_source_solver import solve_joint_source,make_profile_update
from continuous_central_brown4_optin import enable_projection_backend
from render_result import SceneRenderer,decode_selected

TEMP=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp')
UPPER=TEMP/'source_solver_audit/secondary_upper_cap_source_only_v1/upper_cap_source_boundary_QUALIFIED_NO_WORLD.json'
UPPER_SHA='4c41f416b9dbba4f4a11ffb0c0d28f07215a6064132436bcccec4b7ff22905f9'
WARM=TEMP/'source_solver_audit/secondary_confirmed_near_far_full_Pz_RF_v1/physical_secondary_confirmed_nearboard_height_RF_UNACCEPTED.npz'
TOP_SOURCE_IDS=frozenset(('board_top','board_top_far_visible','board_top_near_visible',
    'lower_long_side_board_top_confirmed_source_interval'))


class SharedBoardHeightGeometry:
    """Explicit top-edge binding, reusable by a later joint three-camera solve."""
    def __init__(self,scene):
        self.base=scene
        self.base_top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
        self.base_height_scene=float(scene['parameters']['board_height'])
        if not np.allclose(self.base_top[:,2],self.base_height_scene,atol=1e-12,rtol=0):
            raise ValueError('One common baseline physical board top required')
        self.base_height_m=self.base_height_scene*SCENE_UNIT_M
        # Printed artwork does not stretch when the conditional cap height changes.
        from render_result import advertisement_boundary_paths
        paths=advertisement_boundary_paths(scene)
        self.minimum_print_height_m=max([float(np.max(np.asarray(p['xyz'])[:,2]))*SCENE_UNIT_M for p in paths],default=0.)

    def validate(self,height_m):
        height_m=float(height_m)
        if not np.isfinite(height_m)or height_m<=self.minimum_print_height_m+1e-9:
            raise ValueError('Board cap must stay above unchanged printed advertisement geometry')
        return height_m

    def is_top(self,row):
        return row['id']in TOP_SOURCE_IDS or row.get('curve_observation_kind')==NEAR_KIND

    def rebind(self,row,height_m):
        height_m=self.validate(height_m);result=dict(row)
        if not self.is_top(row):return result
        world=np.asarray(row['world'],float)
        if not np.allclose(world[:,2],self.base_height_scene,atol=1e-12,rtol=0):
            raise ValueError('Top source is not bound to the original physical board height')
        moved=world.copy();moved[:,2]=height_m/SCENE_UNIT_M;result['world']=moved
        if 'upper_world'in row:
            upper=np.asarray(row['upper_world'],float)
            if not np.allclose(upper[:,2],self.base_height_scene,atol=1e-12,rtol=0):
                raise ValueError('Connected upper cap is not bound to the original physical board height')
            upper=upper.copy();upper[:,2]=height_m/SCENE_UNIT_M;result['upper_world']=upper
        return result

    def scene(self,height_m):
        h=self.validate(height_m)/SCENE_UNIT_M;out=deepcopy(self.base)
        out['parameters']['board_height']=h
        out['parameters']['board_height_status']='conditional_complete_secondary_continuous_image_fit_not_measured'
        top=np.asarray(out['board_surface']['top_edge_xyz'],float);top[:,2]=h
        out['board_surface']['top_edge_xyz']=top.tolist()
        for panel in out['board_surface']['panels']:panel['z_range']=[panel['z_range'][0],h]
        for mesh in out['meshes']:
            if mesh.get('name')!=out['board_surface']['mesh_id']:continue
            positions=np.asarray(mesh['positions'],float)
            oldtop=np.isclose(positions[:,2],self.base_height_scene,rtol=0,atol=1e-12)
            positions[oldtop,2]=h;mesh['positions']=positions.tolist()
            if int(oldtop.sum())!=len(self.base_top):raise ValueError('Rendered board top mesh inventory changed')
        for field in ('advertisements','advertisement_3d_corners','ice_features','ice_grid','ice_boundary'):
            if out[field]!=self.base[field]:raise ValueError('Shared height changed independent geometry:'+field)
        out['shared_board_height_source_diagnostic']=dict(height_m=float(height_m),base_height_m=self.base_height_m,
            applies_to='one entire physical board across installations; component currently restores secondary only',
            source_status='conditional image-derived estimate, not measured',printed_advertisement_Z_unchanged=True,
            board_XY_and_four_congruent_symmetric_corner_shape_unchanged=True,accuracy_accepted=False)
        return out


def merge_upper_cap_source(prepared,path=UPPER):
    """Join qualified upper and original lower rows as one exposure group."""
    path=Path(path)
    if ref(path)['sha256']!=UPPER_SHA:raise ValueError('Frozen qualified source packet changed')
    packet=json.loads(path.read_text(encoding='utf8'))
    if not packet['accepted_source_measurement_qualification']or not packet['no_camera_loaded']or not packet['no_world_binding']:
        raise ValueError('Source-only human-qualified upper cap required')
    upper_world,upper_geometry=connected_lower_left_cap(prepared['scene'])
    oldfit=prepared['confirmed_nearboard_fit_rows'];oldheld=prepared['confirmed_nearboard_held_rows']
    lookup={int(row['pts']):row for row in oldfit+oldheld};changes={};ledger=[]
    held_actual={int(prepared['aliases'][p])for p in packet['visual_held_source_native_pts']}
    for record in packet['records']:
        p=int(record['source_native_pts']);actual=int(prepared['aliases'][p]);original=lookup[actual]
        raw=cv2.imread(record['source_png']['path'])
        if raw is None or hashlib.sha256(raw.tobytes()).hexdigest()!=record['native_bgr_sha256']:
            raise ValueError('Upper source native pixels changed')
        if (actual in held_actual)!=(original.get('actual_admission')=='held/reportonly'):
            raise ValueError('User source canonical exposure must remain held')
        upper=[dict(y=float(m['source_y']),x_interval_px=list(map(float,m['x_interval_px'])),
            source_native_pts=[p],upper_source_packet=ref(path))for m in record['measurements']]
        original_y={float(m['y'])for m in original['source_horizontal_intervals']}
        overlapping=[m for m in upper if float(m['y'])in original_y]
        # Keep the old lower UV/support exactly when an upper read overlaps
        # its scanline. A second read of the same physical row is not weight.
        upper=[m for m in upper if float(m['y'])not in original_y]
        allmeasure=sorted(upper+deepcopy(original['source_horizontal_intervals']),key=lambda m:m['y'])
        if len({m['y']for m in allmeasure})!=len(allmeasure):raise ValueError('Physical source scanline duplicated')
        lohi=np.asarray([m['x_interval_px']for m in allmeasure]);mid=lohi.mean(axis=1);count=len(allmeasure)
        row=dict(original,uv=np.c_[mid,[m['y']for m in allmeasure]],normal=np.tile([1.,0.],(count,1)),
            normal_interval=lohi-mid[:,None],sigma=np.full(count,4.*np.sqrt(count)),
            source_horizontal_intervals=allmeasure,upper_source_qualification=ref(path),
            upper_world=upper_world.copy(),upper_world_domain=upper_geometry,
            upper_source_world_binding='existing connected lowerstraight/lower-left-corner/leftshort cap chain; no source station or tangent assigned',
            original_lower_source_uv=original['uv'].copy(),original_lower_source_sigma=original['sigma'].copy(),
            sigma_modelling_decision='one correlated physical cap edge per actual exposure; original engineering sigma4 times sqrt(combined upper+lower row count); original lower sigma separately retained',
            source_upper_background='white vertical board face/foot or ice according to source; all are same cap front edge',
            physical_identity_from_continuous_native_source=True,source_UV_from_camera=False)
        changes[actual]=row
        ledger.append(dict(source_native_pts=p,actual_pts=actual,role=row['split'],lower_original_scan_count=len(original['uv']),
            new_upper_scan_count=len(upper),duplicate_upper_scans_preserving_original=overlapping,
            combined_scan_count=count,old_lower_sigma=original['sigma'],
            combined_sigma=row['sigma'],original_world_unchanged=True,original_lower_UV_support_roles_unchanged=True,
            upper_exact_user_source_never_fit=actual in held_actual))
    fit=[changes.get(int(r['pts']),r)for r in oldfit];held=[changes.get(int(r['pts']),r)for r in oldheld]
    return fit,held,changes,ledger


class SecondarySharedBoardHeightBlock:
    """q=[common physical board height m, secondary Pz m, all actual R3/logf]."""
    def __init__(self,bundle,local_q,geometry):
        self.bundle,self.local_q,self.geometry=bundle,local_q,geometry
        self.groups=bundle.abs_groups

    def local(self,q):
        local=self.local_q.copy();local[0]=q[1]/(100.*SCENE_UNIT_M);local[4:]=q[2:];return local

    def evaluate(self,q,jacobian=True):
        h=float(q[0]);local=self.local(q)
        self.bundle.abs_groups=[(p,[self.geometry.rebind(r,h)for r in rows])for p,rows in self.groups]
        try:
            if not jacobian:return self.bundle.evaluate(local,False)
            values,oldj=self.bundle.evaluate(local,True);count=sum(2*len(r['uv'])for _,rows in self.groups for r in rows)
            derivative=np.zeros(count);cursor=0;step=1e-5
            for p,rows in self.groups:
                camera=self.bundle.camera(local,p)
                for row in rows:
                    n=2*len(row['uv'])
                    if self.geometry.is_top(row):
                        plus=self.geometry.rebind(row,h+step);minus=self.geometry.rebind(row,h-step)
                        derivative[cursor:cursor+n]=(connected_cap_absolute_value_jac15(camera,[plus],False)-
                            connected_cap_absolute_value_jac15(camera,[minus],False))/(2*step)
                    cursor+=n
            column=np.r_[np.zeros(self.bundle.motion_rows),derivative,np.zeros(len(values)-self.bundle.motion_rows-count)]
            jac=hstack((csr_matrix(column[:,None]),oldj[:,0]/(100.*SCENE_UNIT_M),oldj[:,4:]),format='csr')
            return values,jac
        finally:self.bundle.abs_groups=self.groups

    def records(self,q,records):
        local=self.local(q);result=[]
        for rec in records:
            row=rec['row'];actual=int(row['pts']);cam=self.bundle.camera(local,actual)
            bound=self.geometry.rebind(row,q[0])
            error=connected_cap_absolute_value_jac15(cam,[bound],False).reshape(-1,2)*row['sigma'][:,None]
            result.append(dict(actual_pts=actual,id=row['id'],original_split=row.get('split'),source_role=row.get('source_role'),
                report_role=rec['report_role'],entered_fit_objective=bool(rec['entered_fit_objective']),
                actual_admission='solver_development'if rec['entered_fit_objective']else'held_or_pending_report_only',
                not_blind=True,source_uv=row['uv'],source_sigma=row['sigma'],curve_observation_kind=row.get('curve_observation_kind'),
                outside_support_xy_px=error,median_outside_support_px=float(np.median(np.linalg.norm(error,axis=1))),
                maximum_outside_support_px=float(np.max(np.linalg.norm(error,axis=1))),
                true_board_top_rebound=self.geometry.is_top(row),printed_advertisement_Z_unchanged=str(row['id']).startswith('ad')))
        return result


def prepare(output,*,warm=WARM,upper=UPPER,candidate_scene_path=None,**kwargs):
    out=Path(output).resolve()
    if not out.is_relative_to(TEMP.resolve()):raise ValueError('Bulk output must stay in current temporary root')
    out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    prepared=prepare_full_secondary_confirmed_nearboard(out/'complete_source_factory',candidate_scene_path,**kwargs)
    old=prepared['bundle'];fit,held,changes,correlation=merge_upper_cap_source(prepared,upper)
    groups=[(p,[changes.get(int(r['pts']),r)if r.get('curve_observation_kind')==NEAR_KIND else r for r in rows])for p,rows in old.abs_groups]
    sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rows in groups for r in rows])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,
        motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,sigma])
    bundle=MovingPivotSourceBundle(base,old.calibration,old.motion_kernel,reference_pts=old.reference_pts,
        initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,finite_planes=old.finite_planes,
        finite_records=old.finite_records,scene=prepared['scene'],absolute_evaluator=connected_cap_absolute_value_jac15)
    prepared.update(bundle=bundle,confirmed_nearboard_fit_rows=fit,confirmed_nearboard_held_rows=held)
    local=prepared['q'].copy()
    if warm is not None:
        model,native,aliases,views=PhysicalCameraModel.load_npz(warm)
        if not np.array_equal(native,prepared['native'])or aliases!=prepared['aliases']:raise ValueError('Warm native/actual aliases changed')
        calibration=model.calibrations[IDENTIFIER]
        for field in ('lever_xyz','principal_x_fraction','principal_y_fraction','k1'):
            if getattr(calibration,field)!=getattr(old.calibration,field):raise ValueError('Warm changed fixed optics/mechanics')
        if calibration.pivot_xyz[:2]!=old.calibration.pivot_xyz[:2]:raise ValueError('Warm changed fixed pivot XY')
        local[0]=calibration.pivot_xyz[2]/100.
        local[4:]=np.concatenate([np.r_[views[p].view.rvec_world_to_camera,views[p].view.log_f_ratio+np.log(1920.)]for p in bundle.stamps])
    geometry=SharedBoardHeightGeometry(prepared['scene']);block=SecondarySharedBoardHeightBlock(bundle,local,geometry)
    q=np.r_[geometry.base_height_m,local[0]*100*SCENE_UNIT_M,local[4:]]
    records=source_records(prepared)
    for row in held:records.append(dict(row=row,report_role='user_visual_held_confirmed_complete_cap',entered_fit_objective=False))
    for rec in records:
        rec['row']=dict(rec['row']);rec['row']['pts']=prepared['aliases'][int(rec['row']['pts'])]
        geometry.rebind(rec['row'],q[0])
    if len(bundle.stamps)!=1061 or len(bundle.a)!=39679 or int(bundle.finite_mask.sum())!=6:
        raise ValueError('Complete physical secondary source scope changed')
    held_actual={int(prepared['aliases'][230422016])}
    if any(p in held_actual and r.get('curve_observation_kind')==NEAR_KIND for p,rows in groups for r in rows):
        raise ValueError('Exact visual-held cap source entered objective')
    priors=len(bundle.stamps)+1
    scales=np.r_[bundle.native_residual_sigma,np.ones(priors)]
    quadratic=np.r_[np.zeros(bundle.motion_rows,bool),np.concatenate([np.full(2*len(r['uv']),r.get('curve_observation_kind')in('horizontal_extent','clipped_horizontal_extent'),bool)for p,rows in groups for r in rows]),np.ones(priors,bool)]
    manifest=dict(schema='rinkspace.secondary.complete-source.common-board-height.connected-upper-cap.Pz.Rf.v2',source=ref(__file__),
        original_factory=ref(out/'complete_source_factory/input_manifest.json'),source_video_sha256=SOURCE_SHA,
        upper_source=ref(upper),warm_unaccepted_component=ref(warm)if warm else None,
        free=['one shared physical board height metres','one secondary pivot height metres','all1061 actual R3/logf'],
        fixed=['pivot XY','nonzero lever','PP','k1','metre scale','entire board XY','four congruent symmetric corner shape','all ice geometry','printed advertisement XYZ'],
        board_height_is_conditional_image_fit_not_measured=True,all1061_actual=True,original_motion_rows=len(bundle.a),
        original_six_finite_material_planes_unchanged=True,old_ordinary_source_UV_sigma_roles_unchanged=True,
        same_cap_upper_lower_correlation_redeclared=correlation,user_source_230422016_and_aliases_never_fit=True,
        original_pending_and_all_held_not_promoted=True,height_prior_once=1,roll_prior_once_per_actual=len(bundle.stamps),
        upper_source_domain='connected current lowerstraight/lower-left-corner/leftshort cap; no precise source world station',
        original_lower_straight_world_and_operator_unchanged=True,
        new_board_height_prior_added=False,component_only_other_installations_Rf_not_recovered=True,
        production_changed=False,accuracy_accepted=False,video_generated=False)
    write(out/'input_manifest.json',manifest);write(out/'all_fit_and_report_source_records.json',records)
    return SimpleNamespace(output=out,prepared=prepared,block=block,q=q,records=records,scales=scales,quadratic=quadratic,manifest=manifest)


def check_prepared(state):
    block,q=state.block,state.q;values,jac=block.evaluate(q,True);checks=[]
    for column in (0,1):
        step=1e-6;plus=q.copy();minus=q.copy();plus[column]+=step;minus[column]-=step
        vp=block.evaluate(plus,False);vm=block.evaluate(minus,False);fd=(vp-vm)/(2*step)
        analytic=jac[:,column].toarray().ravel();sl=slice(block.bundle.motion_rows,None)
        absolute_error=float(np.linalg.norm(fd[sl]-analytic[sl])/max(1.,np.linalg.norm(fd[sl])))
        weights=np.ones(len(values));outside=~state.quadratic
        weights[outside]=np.minimum(1.,2./np.maximum(np.abs(values[outside]),1e-12))
        numeric=float((weights*values)@fd);direction=float((weights*values)@analytic)
        gradient_error=abs(numeric-direction)/max(1.,abs(numeric))
        errors=[];cursor=block.bundle.motion_rows
        for p,rows in block.groups:
            for row in rows:
                n=2*len(row['uv']);delta=fd[cursor:cursor+n]-analytic[cursor:cursor+n]
                errors.append(dict(pts=p,id=row['id'],max_sigma_derivative_error=float(np.max(np.abs(delta))),
                    finite_difference_norm=float(np.linalg.norm(fd[cursor:cursor+n])),
                    analytic_norm=float(np.linalg.norm(analytic[cursor:cursor+n]))));cursor+=n
        checks.append(dict(column=column,absolute_source_relative_error=absolute_error,
            full_profiled_objective_gradient_relative_error=gradient_error,numeric_gradient=numeric,analytic_gradient=direction,
            largest_source_derivative_differences=sorted(errors,key=lambda r:r['max_sigma_derivative_error'],reverse=True)[:8]))
    if len(values)!=len(state.scales)or jac.shape!=(len(values),len(q)):raise ValueError('Full source CSR inventory changed')
    passed=max(c['absolute_source_relative_error']for c in checks)<2e-4 and max(c['full_profiled_objective_gradient_relative_error']for c in checks)<5e-4
    block.geometry.scene(q[0]);write(state.output/'actual_board_height_Pz_full_source_FD.json',dict(checks=checks,passed=passed,
        parameters=len(q),residual_rows=len(values),complete_motion_and_native_sigma_preserved=True,accuracy_accepted=False))
    print(dict(stage='full_secondary_shared_board_height_prepared',parameters=len(q),residual_rows=len(values),FD=checks),flush=True)
    if not passed:raise ValueError('True board/Pz full source finite difference failed; diagnostics saved')


def export_state(state,q,folder,*,draw=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);local=state.block.local(q)
    scene=state.block.geometry.scene(q[0]);write(folder/'candidate_scene_UNACCEPTED.json',scene)
    calibration=state.block.bundle.current_calibration(local)
    model=PhysicalCameraModel({IDENTIFIER:calibration})
    views={p:CanonicalView(IDENTIFIER,state.block.bundle.view(local,p))for p in state.block.bundle.stamps}
    camera_path=folder/'physical_secondary_shared_board_height_Pz_RF_UNACCEPTED.npz'
    model.save_npz(camera_path,state.prepared['native'],state.prepared['aliases'],views,source_sha256=SOURCE_SHA,time_base='1/15360')
    write(folder/'actual_source_review.json',dict(board_height_m=float(q[0]),secondary_Pz_m=float(q[1]),
        all_sources=state.block.records(q,state.records),camera=ref(camera_path),scene=ref(folder/'candidate_scene_UNACCEPTED.json'),
        exact_user_upper_and_lower_held=True,accuracy_accepted=False,other_installations_not_recovered=True))
    if draw:
        renderer=SceneRenderer(scene,draw_grid=True,draw_advertisements=True)
        current_renderer=SceneRenderer(state.prepared['scene'],draw_grid=True,draw_advertisements=True)
        pts=[229521664,229526272,230372096,230381312,230390528,230395904,230415104,230418944,230422016,230422784,230426624,230507264]
        for p,raw in decode_selected(AssetStore(ROOT).source('data/input/oxXu1avhrvs.mp4'),pts):
            actual=state.prepared['aliases'][p]
            cur,_=current_renderer.draw(raw,state.prepared['current'].camera25(state.prepared['views'][actual]))
            new,_=renderer.draw(raw,model.camera25(views[actual]))
            for label,im in [('CURRENT7625',cur),('SHARED BOARD HEIGHT Pz FULL1061 UNACCEPTED',new)]:
                cv2.rectangle(im,(0,0),(1920,36),(15,15,15),-1)
                cv2.putText(im,f'{label} PTS{p} | all model all ads',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1)
            cv2.imwrite(str(folder/f'{p}_native_raw_current_candidate_all_model.jpg'),np.vstack([raw,cur,new]),[cv2.IMWRITE_JPEG_QUALITY,98])
            if p==230422016:cv2.imwrite(str(folder/f'{p}_native_candidate_all_model.jpg'),new,[cv2.IMWRITE_JPEG_QUALITY,98])
    return camera_path


def run(args):
    state=prepare(args.output,warm=args.warm,upper=args.upper);check_prepared(state)
    if not args.solve:return state
    counter=[0]
    def review(q,event=None):
        i=counter[0];counter[0]+=1
        folder=state.output/'actual_source_reviews'/f'{i:04d}'
        export_state(state,q,folder,draw=i<3 or i%5==0)
        write(state.output/'latest_actual_source_review.json',dict(source_review=ref(folder/'actual_source_review.json'),
            camera=ref(folder/'physical_secondary_shared_board_height_Pz_RF_UNACCEPTED.npz'),scene=ref(folder/'candidate_scene_UNACCEPTED.json'),
            event=event,accuracy_accepted=False))
    def guard(q):
        state.block.geometry.validate(q[0]);local=state.block.local(q)
        if not np.isfinite(q).all()or np.any(local[7::4]<np.log(100.))or np.any(local[7::4]>np.log(50000.)):
            raise ValueError('Physical finite focal domain')
        state.block.bundle.validate_finite_extents(state.block.bundle.cache(local,False))
    result=solve_joint_source(state.block.evaluate,state.q,state.quadratic,state.scales,state.output/'solver',guard,
        update_profiles=make_profile_update([state.block]),checkpoint_callback=review,checkpoint_identity=state.manifest,
        checkpoint_arrays=dict(secondary_actual_pts=np.asarray(state.block.bundle.stamps,np.int64)))
    export_state(state,result.q,state.output/'final_component',draw=True)
    write(state.output/'result.json',dict(status=result.status,stop_reason=result.stop_reason,seconds=result.seconds,
        source_reviews=counter[0],board_height_m=float(result.q[0]),secondary_Pz_m=float(result.q[1]),
        full1061_solved=True,accuracy_accepted=False,other_installations_not_recovered=True,production_changed=False))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--warm',type=Path,default=WARM)
    parser.add_argument('--upper',type=Path,default=UPPER);parser.add_argument('--solve',action='store_true')
    run(parser.parse_args())
