"""Full-source shared rink dimensions and three physical camera mounts.

Confirmed relationships remain shared. Numeric house radius, common row
pitch and the two column X positions were inherited estimates, not surveyed
dimensions. This candidate keeps the current metric, all pivot XY, levers,
optics, boards and ads. Original source pixels/roles/materials are immutable.
"""
from __future__ import annotations
import argparse,copy,io,json,time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from z190_joint_shared_F_height_source import N,SCENE,module_at,main_input,with_main1_clipped_support
from z190_main_cut_depth_profile_recovery import CURRENT
from z190_main_zoom_k1_source_information import read,write,ref
from z190_low_shared_F_scene_rf_recovery import prepare as prepare_low
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from z190_low_opening_rf_refine import convert_row
from shared_F_scene_candidate import rebind as rebind_F
from continuous_central_brown4_optin import enable_projection_backend
from shared_rink_numeric_geometry import SharedRinkNumericGeometry
from z190_joint_source_solver import solve_joint_source,make_profile_update
import z190_secondary_qualified_core_source as secondary_factory


class RinkBlock:
    def __init__(self,name,prepared,geometry,height_slot,start,absolute=absolute_value_jac15,*,geometry_parameter_count=4):
        self.name,self.prepared,self.geometry=name,prepared,geometry
        self.bundle,self.q0=prepared.bundle,prepared.q.copy()
        self.height_slot,self.start,self.absolute=height_slot,start,absolute
        self.geometry_parameter_count=int(geometry_parameter_count)
        self.global_parameter_count=self.geometry_parameter_count+3
        if self.geometry_parameter_count not in (4,5) or self.height_slot not in range(self.geometry_parameter_count,self.global_parameter_count):
            raise ValueError('Explicit geometry dimensions and installation height slot required')
        if self.start<self.global_parameter_count:raise ValueError('Local source views must follow all global parameters')
        self.groups=self.bundle.abs_groups
        # Installation height is represented once globally; retain every roll.
        self.bundle.priors=lambda q,cache:tuple(x[1:]for x in MovingPivotSourceBundle.priors(self.bundle,q,cache))
        self.quadratic=np.r_[np.zeros(self.bundle.motion_rows,bool),np.concatenate([
            np.full(2*len(r['uv']),r.get('curve_observation_kind')in('horizontal_extent','clipped_horizontal_extent'),bool)
            for p,rows in self.groups for r in rows]),np.ones(len(self.bundle.stamps),bool)]

    def local(self,q):
        local=self.q0.copy();local[0]=q[self.height_slot]/(100.*SCENE_UNIT_M)
        local[4:]=q[self.start:self.start+len(local)-4];return local

    def evaluate(self,q,jacobian):
        local=self.local(q)
        ng=self.geometry_parameter_count
        groups=[(p,[self.geometry.rebind(r,q[:ng])for r in rows])for p,rows in self.groups]
        self.bundle.abs_groups=groups
        try:
            if jacobian:residual,oldj=self.bundle.evaluate(local,True)
            else:return self.bundle.evaluate(local,False)
            # Numeric geometry columns affect only original absolute factors.
            # Central source differences retain the actual branch/extent/
            # interval operator. No repeated motion evaluation is required.
            abs_count=sum(2*len(r['uv'])for p,rows in groups for r in rows)
            geo=np.zeros((abs_count,ng));cursor=0;step=1e-5
            for p,rows in self.groups:
                cam=self.bundle.camera(local,p)
                for r in rows:
                    count=2*len(r['uv'])
                    if self.geometry.feature_id(r) is not None:
                        for c in range(ng):
                            plus,minus=q[:ng].copy(),q[:ng].copy();plus[c]+=step;minus[c]-=step
                            geo[cursor:cursor+count,c]=(self.absolute(cam,[self.geometry.rebind(r,plus)],False)-self.absolute(cam,[self.geometry.rebind(r,minus)],False))/(2*step)
                    cursor+=count
            geometry_column=vstack((csr_matrix((self.bundle.motion_rows,ng)),csr_matrix(geo),csr_matrix((len(residual)-self.bundle.motion_rows-abs_count,ng))),format='csr')
            heights=[]
            for c in range(3):
                heights.append(oldj[:,0]/(100.*SCENE_UNIT_M)if c==self.height_slot-ng else csr_matrix((len(residual),1)))
            prefix=hstack((geometry_column,*heights),format='csr')
            before=csr_matrix((len(residual),self.start-self.global_parameter_count));after=csr_matrix((len(residual),len(q)-self.start-len(local)+4))
            return residual,hstack((prefix,before,oldj[:,4:],after),format='csr')
        finally:self.bundle.abs_groups=self.groups

    def absolute_records(self,q,review_groups=None):
        local=self.local(q);out=[]
        groups=self.groups if review_groups is None else review_groups
        for p,rows in groups:
            cam=self.bundle.camera(local,p)
            for row in rows:
                geometry_values=(self.geometry_values(q)if hasattr(self,'geometry_values')else q[:self.geometry_parameter_count])
                moved=self.geometry.rebind(row,geometry_values);error=self.absolute(cam,[moved],False).reshape(-1,2)*row['sigma'][:,None]
                norm=np.linalg.norm(error,axis=1)
                source_pts=row.get('source_native_pts',row.get('pts',p))
                # Corner groups retain one original native timestamp per
                # source corner. Keep that ledger rather than coercing it
                # into a fabricated single-frame scalar.
                source_pts=[int(x)for x in source_pts]if isinstance(source_pts,(list,tuple,np.ndarray))else int(source_pts)
                out.append(dict(block=self.name,actual_pts=int(p),source_native_pts=source_pts,
                    id=row['id'],region=row.get('source_region'),split=row.get('split','fit'),source_role=row.get('source_role'),
                    solver_admitted=review_groups is None,review_report_role=row.get('review_report_role'),
                    source_split_is_preserved_metadata_not_solver_admission=True,
                    outside_support_xy_px=error.tolist(),median_outside_support_norm_px=float(np.median(norm)),maximum_outside_support_norm_px=float(np.max(norm))))
        return out


def main_held(prepared):
    groups={}
    for role in('held_rows','pending_rows'):
        for raw in prepared.authority[role]:
            p=prepared.aliases[int(raw['pts'])]
            if p not in prepared.bundle.index:continue
            row=rebind_F(convert_row(raw),prepared.original,prepared.candidate)
            groups.setdefault(p,[]).append(row)
    for group in prepared.authority.get('untouched_future_groups',[]):
        for raw in group['rows']:
            p=prepared.aliases[int(raw['pts'])]
            if p not in prepared.bundle.index:continue
            row=rebind_F(convert_row(raw),prepared.original,prepared.candidate)
            row=dict(row,review_report_role='future_unfitted_identity_check',review_only_never_admitted=True)
            groups.setdefault(p,[]).append(row)
    if hasattr(prepared,'additional_held_groups'):
        for p,rows in prepared.additional_held_groups:groups.setdefault(p,[]).extend(rows)
    else:
        for r in prepared.additional:
            if r['split']=='held':groups.setdefault(r['pts'],[]).append(rebind_F(r,prepared.original,prepared.candidate))
    return sorted(groups.items())


def prepare_joint(output,warm):
    output.mkdir(parents=True,exist_ok=False);enable_projection_backend();cv2.setNumThreads(1)
    model,native,aliases,views=PhysicalCameraModel.load_npz(CURRENT)
    cache_root=N/'joint_full_source_shared_rink_dimensions_preparation_v1'
    def cache(path):return path if path.exists() else None
    main1=main_input(output/'main1_source',1,cache(cache_root/'main1_source/original_motion_source_admission.npz'));with_main1_clipped_support(main1)
    main7=main_input(output/'main7_source',7,cache(cache_root/'main7_source/original_motion_source_admission.npz'))
    secmod=secondary_factory
    sec=secmod.prepare_full_secondary_qualified_branches(output/'secondary_source',SCENE)
    secondary=SimpleNamespace(**sec);secondary.q=sec['q']
    low=prepare_low(output/'low_source');low.q=low.q0
    main9mod=module_at(N/'ice_alignment_review/main9_full_source_factory.py')
    main9=main9mod.prepare_full_main9_core_source(output/'main9_source',SCENE,
        source_admission_cache=cache(cache_root/'main9_source/original_source_adapter/original_motion_source_admission.npz'))
    geometry=SharedRinkNumericGeometry(read(SCENE))
    items=[('main1',main1,absolute_value_jac15),('main7',main7,absolute_value_jac15),
           ('secondary',secondary,absolute_value_jac15),('low',low,absolute_value_jac15),
           ('main9',main9,main9mod.typed_absolute_value_jac15)]
    ids=sorted(model.calibrations);slots={name:4+i for i,name in enumerate(ids)}
    parts=[np.r_[np.zeros(4),[model.calibrations[name].pivot_xyz[2]*SCENE_UNIT_M for name in ids]]]
    blocks=[];start=7
    for name,p,absolute in items:
        b=RinkBlock(name,p,geometry,slots[p.bundle.calibration.installation_id],start,absolute)
        blocks.append(b);parts.append(p.q[4:]);start+=len(p.q)-4
    q=np.concatenate(parts);production_q=q.copy()
    production_q[0]=-main1.manifest['shared_F_scene_binding']['F_column_dx_m']
    warm_binding=None
    if warm:
        data=warm.read_bytes()
        with np.load(io.BytesIO(data),allow_pickle=False)as z:
            old=z['q'];starts=z['block_start'];lengths=z['block_lengths']
        if len(starts)!=4:raise ValueError('Warm source branch must retain main1/main7/secondary/low blocks')
        q[0]=old[0];q[4:7]=old[1:4]
        for b,oldstart,length in zip(blocks[:4],starts,lengths):
            if length!=len(b.q0)-4:raise ValueError('Warm retained actual exposure block changed')
            q[b.start:b.start+length]=old[oldstart:oldstart+length]
        warm_binding=dict(checkpoint=ref(warm),starting_state_only=True,source_geometry_previous_branch_unaccepted=True,
            main9_uses_current_7625_RF=True,old_main4_rejected_candidate_not_read=True)
    held={b.name:main_held(b.prepared)for b in blocks if b.name.startswith('main')}
    held['low']={}
    for role,old,row,p in low.source_rows:
        if row['split']!='fit':held['low'].setdefault(p,[]).append(row)
    held['low']=sorted(held['low'].items())
    # Secondary preserves all source roles, including independently reserved
    # complete R2 right extent. Its adapter supplies the common bound rows.
    held['secondary']=secmod.review_held_groups(sec)
    return SimpleNamespace(model=model,native=native,aliases=aliases,views=views,geometry=geometry,blocks=blocks,
        q=q,production_q=production_q,ids=ids,slots=slots,held=held,warm_binding=warm_binding,output=output)


def run(args):
    prepared=prepare_joint(args.output.resolve(),args.warm);blocks=prepared.blocks;ids=prepared.ids;slots=prepared.slots
    refs={name:next(b for b in blocks if b.bundle.calibration.installation_id==name)for name in ids}
    quadratic=np.r_[np.concatenate([b.quadratic for b in blocks]),np.ones(3,bool)]
    scales=np.concatenate([np.r_[b.bundle.native_residual_sigma,np.ones(len(b.bundle.stamps))]for b in blocks]+[np.ones(3)])
    def evaluate(q,jacobian=True):
        results=[b.evaluate(q,jacobian)for b in blocks]
        values=[r[0]if jacobian else r for r in results];matrices=[r[1]for r in results]if jacobian else []
        for name in ids:
            b=refs[name];local=b.local(q);bundle=b.bundle;p=bundle.reference_pts
            cal=bundle.current_calibration(local);cam,_,vj=cal.camera25_and_jacobians(bundle.view(local,p))
            values.append(np.array([(cam[2]*SCENE_UNIT_M-1.6)/.18]))
            if jacobian:
                row=np.zeros((1,len(q)));row[0,slots[name]]=1./.18
                pos=b.start+4*bundle.index[p];row[0,pos:pos+4]=vj[2]*SCENE_UNIT_M/.18;matrices.append(csr_matrix(row))
        value=np.concatenate(values)
        return(value,vstack(matrices,format='csr'))if jacobian else value
    def guard(q):
        prepared.geometry.validate_parameters(q[:4])
        if not np.isfinite(q).all()or np.any(q[4:7]<=0):raise ValueError('Physical installation pivot heights')
        for b in blocks:
            focal=q[b.start+3:b.start+len(b.q0)-4:4]
            if np.any((focal<np.log(100.))|(focal>np.log(50000.))):raise ValueError('Physical focal numerical domain')
    def diagnostics(q):
        return dict(F_column_dx_from_hard_scene_m=float(q[0])+refs[ids[0]].prepared.manifest['shared_F_scene_binding']['F_column_dx_m'],
            R_column_dx_m=float(q[1]),row_pitch_delta_m=float(q[2]),outer_radius_delta_m=float(q[3]),
            pivot_heights_m={name:float(q[slots[name]])for name in ids})
    def source_review(q,record=None):
        rows=[]
        for b in blocks:
            rows.extend(b.absolute_records(q));rows.extend(b.absolute_records(q,prepared.held[b.name]))
        write(prepared.output/'latest_actual_source_review.json',dict(state_diagnostics=diagnostics(q),iteration=record,rows=rows,
            raw_source_pixels_roles_sigma_unchanged=True,accuracy_accepted=False,complete_shot_review_still_required=True))
        scene=prepared.geometry.scene(q[:4]);write(prepared.output/'latest_candidate_scene_UNACCEPTED.json',scene)
    manifest=dict(schema='rinkspace.full-source.shared-numeric-rink-dimensions-physical-mounts.v1',current=ref(CURRENT),scene_reference=ref(SCENE),
        warm_start=prepared.warm_binding,blocks=[dict(name=b.name,input_manifest=b.prepared.manifest if hasattr(b.prepared,'manifest')else b.prepared.input_manifest,
            actual=len(b.bundle.stamps),motion=len(b.bundle.a),source_absolute_points=sum(len(r['uv'])for p,rows in b.groups for r in rows))for b in blocks],
        actual_exposures=sum(len(b.bundle.stamps)for b in blocks),source_motion_rows=sum(len(b.bundle.a)for b in blocks),
        geometry_numeric_schema=prepared.geometry.manifest(),geometry_adapter=ref(ROOT/'calibration/program_tracking/shared_rink_numeric_geometry.py'),
        free_geometry_metres=['one complete F column X increment','one complete R column X increment','one common paired equally-spaced row pitch increment','one common outer radius increment with inherited nested ratios'],
        fixed=['current metric','all pivotXY','constant nonzero levers','PP/k1','boards/ads/remaining rink markings','row midpoint'],
        source_uv_roles_sigmas_finite_materials_unchanged=True,main4_RF_not_recovered=True,body_rebound=False,full_programme_exported=False,accuracy_accepted=False)
    write(prepared.output/'input_manifest.json',manifest)
    source_review(prepared.production_q,dict(stage='current_production_source_baseline'))
    (prepared.output/'latest_actual_source_review.json').replace(prepared.output/'current_production_actual_source_review.json')
    source_review(prepared.q,dict(stage='initial_unaccepted_start'))
    if args.prepare_only:return
    # Check seven actual complete-source cost directions, with the same
    # profiled positive-depth objective and source-engineering sigma.
    from main7_source_bundle_candidate import huber_cost
    def cost(r):return huber_cost(r[~quadratic],True)+.5*float(r[quadratic]@r[quadratic])
    residual,jac=evaluate(prepared.q);weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12));weights[quadratic]=1.
    gradient=np.asarray(jac.T@(weights*residual)).ravel();checks=[]
    for c in range(7):
        step=1e-5;plus,minus=prepared.q.copy(),prepared.q.copy();plus[c]+=step;minus[c]-=step
        fd=(cost(evaluate(plus,False))-cost(evaluate(minus,False)))/(2*step)
        rel=abs(fd-gradient[c])/max(1.,abs(fd),abs(gradient[c]))
        checks.append(dict(column=c,actual_source_cost_direction=float(gradient[c]),central_source_cost_direction=float(fd),relative_error=float(rel)))
    write(prepared.output/'actual_global_source_cost_direction_checks.json',checks)
    print(json.dumps(dict(stage='joint_rink_dimensions_prepared',actual=manifest['actual_exposures'],motion=manifest['source_motion_rows'],checks=checks)),flush=True)
    if max(c['relative_error']for c in checks)>.005:raise ValueError('Complete actual source cost derivative failed')
    del jac,residual
    result=solve_joint_source(evaluate,prepared.q,quadratic,scales,prepared.output,guard,
        update_profiles=make_profile_update(blocks),checkpoint_callback=source_review,
        checkpoint_identity=dict(input_manifest=ref(prepared.output/'input_manifest.json')),
        checkpoint_arrays=dict(block_start=np.array([b.start for b in blocks]),block_lengths=np.array([len(b.q0)-4 for b in blocks]),
            block_names=np.array([b.name for b in blocks]),canonical_pts=np.concatenate([b.bundle.stamps for b in blocks])),
        diagnostic_values=diagnostics)
    q,history,seconds=result
    source_review(q,dict(stage='branch_final_state',seconds=seconds))
    cals=dict(prepared.model.calibrations);views=dict(prepared.views)
    for b in blocks:
        local=b.local(q);cal=b.bundle.current_calibration(local);cals[cal.installation_id]=cal
        for p in b.bundle.stamps:views[p]=CanonicalView(cal.installation_id,b.bundle.view(local,p))
    model=PhysicalCameraModel(cals);path=prepared.output/'full_camera_WITH_UNRECOVERED_MAIN4_UNACCEPTED.npz'
    model.save_npz(path,prepared.native,prepared.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    write(prepared.output/'report.json',dict(manifest,seconds=seconds,camera=ref(path),scene=ref(prepared.output/'latest_candidate_scene_UNACCEPTED.json'),
        main4_shared_height_changed_but_RF_current=True,branch_status=result.status,stop_reason=result.stop_reason,history=history))
    print(dict(camera=str(path),accuracy_accepted=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path);parser.add_argument('--prepare-only',action='store_true');run(parser.parse_args())
