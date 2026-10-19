"""Joint full-source common F-column and three independent mounting heights.

All three fixed pivot XY and constant nonzero levers/optics are preserved.
Each exposure keeps its original source-motion graph and material admission.
Only one complete concentric F-column X is shared across all installations.
The full-source camera states and held evidence determine whether this works.
"""
from __future__ import annotations
import argparse,copy,importlib.util,json,time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import csr_matrix,diags,hstack,vstack
from scipy.sparse.linalg import spsolve
from z190_main_shared_F_source_recovery import prepare_bundle,records
from z190_main_cut_depth_profile_recovery import CURRENT
from z190_main_zoom_k1_source_information import read,write,ref
from z190_low_shared_F_scene_rf_recovery import prepare as prepare_low
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from z190_source_camera_diagnostic import absolute_value_jac15,source_errors
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from main7_source_bundle_candidate import huber_cost
from continuous_central_brown4_optin import enable_projection_backend

N=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp')
KEY=N/'three_installation_shared_F_column_keys_v2_guarded/shared_Fx_fixed_height'
SCENE=KEY/'candidate_scene_concentric_DIAGNOSTIC_ONLY.json'


def module_at(path):
    spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main_input(output,cut,source_admission_cache=None):
    source=N/'ice_alignment_review/main1_ice_source'
    args=SimpleNamespace(current=CURRENT,cut=cut,output=output,candidate_scene=SCENE,minimum_unknown_depth_m=1.,
        new_ice_motion_source=source/'main1_three_red_paint_motion_material_qualified.json'if cut==1 else None,
        source_packet=[source/'main1_F3_external_boundary_packet_qualified.json',source/'main1_F3_extent_Ad7_source_packet_qualified.json']if cut==1 else [],
        source_admission_cache=source_admission_cache)
    return prepare_bundle(args)


def with_main1_clipped_support(prepared):
    bundle=prepared.bundle;groups={p:list(rows)for p,rows in bundle.abs_groups}
    source=read(N/'ice_alignment_review/main1_ice_source/main1_F3_extent_Ad7_source_packet_qualified.json')
    context=source['source_left_extent_inequality_only'];p=prepared.aliases[int(context['pts'])]
    parent=next(r for r in prepared.additional if r['pts']==p and r.get('extent_side')=='right')
    row=dict(id='F3_outer',pts=p,source_native_pts=int(context['pts']),split='fit',source_region='clipped_left',
        curve_observation_kind='clipped_horizontal_extent',extent_side='left',world=parent['world'].copy(),
        uv=np.array([[0.,0.]]),sigma=np.array([4.]),point_correspondence=False,
        source_uv_is_image_boundary_context_only=True,clip_boundary_x=0.,clip_blur_allowance_px=45.,
        source_authority=context,source_evidence=source['source_clipped_left_evidence'],
        engineering_allowance_role='conservative roughly half of the 84.6px right-transition source motion-blur envelope; not calibrated confidence',
        no_visible_tangent_UV_created=True)
    # Bind the whole curve to the already guarded common scene, keeping its
    # source boundary context separate from observed tangent coordinates.
    dx=prepared.manifest['shared_F_scene_binding']['F_column_dx_m'];row['world'][:,0]+=dx/SCENE_UNIT_M
    groups.setdefault(p,[]).append(row)
    absgroups=[(p,groups[p])for p in bundle.stamps if p in groups]
    sig=np.concatenate([np.repeat(r['sigma'],2)for p,rows in absgroups for r in rows])
    base=SimpleNamespace(stamps=bundle.stamps,index=bundle.index,a=bundle.a,b=bundle.b,ia=bundle.ia,ib=bundle.ib,
        motion_sigma=bundle.motion_sigma,abs_groups=absgroups,native_residual_sigma=np.r_[bundle.motion_sigma,sig])
    updated=MovingPivotSourceBundle(base,bundle.calibration,bundle.motion_kernel,reference_pts=bundle.reference_pts,
        initial_reference_optical_height_scene=bundle.initial_reference_optical_height_scene,
        finite_planes=bundle.finite_planes,finite_records=bundle.finite_records,scene=bundle.scene)
    prepared.bundle=updated
    return row


class Block:
    def __init__(self,bundle,q0,name,height_slot,start):
        self.bundle,self.q0,self.name,self.height_slot,self.start=bundle,q0.copy(),name,height_slot,start
        self.groups=bundle.abs_groups
        # Exactly one installation reference-height prior is added globally.
        bundle.priors=lambda q,cache:tuple(x[1:]for x in MovingPivotSourceBundle.priors(bundle,q,cache))
        self.quadratic=np.r_[np.zeros(bundle.motion_rows,bool),np.concatenate([
            np.full(2*len(r['uv']),r.get('curve_observation_kind')in ('horizontal_extent','clipped_horizontal_extent'),bool)
            for p,rows in self.groups for r in rows]),np.ones(len(bundle.stamps),bool)]

    def local(self,q):
        local=self.q0.copy();local[0]=q[self.height_slot]/(100.*SCENE_UNIT_M)
        local[4:]=q[self.start:self.start+len(local)-4];return local

    def evaluate(self,q,jacobian):
        bundle=self.bundle;local=self.local(q);dx=q[0]/SCENE_UNIT_M
        groups=[]
        for p,rows in self.groups:
            moved=[]
            for row in rows:
                if str(row['id']).split('_')[0]in ('F1','F2','F3','F4'):
                    row=dict(row);row['world']=row['world'].copy();row['world'][:,0]+=dx
                moved.append(row)
            groups.append((p,moved))
        bundle.abs_groups=groups
        try:
            if jacobian:residual,oldj=bundle.evaluate(local,True)
            else:residual=bundle.evaluate(local,False)
            if not jacobian:return residual
            dxcol=np.zeros(len(residual));cursor=bundle.motion_rows
            for p,rows in groups:
                cam=bundle.camera(local,p)
                for row in rows:
                    count=2*len(row['uv'])
                    if str(row['id']).split('_')[0]in ('F1','F2','F3','F4'):
                        _,raw=absolute_value_jac15(cam,[row],True)
                        dxcol[cursor:cursor+count]=-raw[:,0]*.01/SCENE_UNIT_M
                    cursor+=count
            height=np.zeros((len(residual),3));height[:,self.height_slot-1]=oldj[:,0].toarray().ravel()/(100.*SCENE_UNIT_M)
            prefix=csr_matrix(np.c_[dxcol,height])
            before=csr_matrix((len(residual),self.start-4));after=csr_matrix((len(residual),len(q)-self.start-len(local)+4))
            return residual,hstack((prefix,before,oldj[:,4:],after),format='csr')
        finally:bundle.abs_groups=self.groups


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    original,_,aliases,allviews=PhysicalCameraModel.load_npz(CURRENT)
    main1=main_input(out/'main1_source',1);clip=with_main1_clipped_support(main1)
    main7=main_input(out/'main7_source',7)
    secondary_module=module_at(N/'source_solver_audit/secondary_full_source_factory.py')
    secondary=secondary_module.prepare_full_secondary_core_source(out/'secondary_source',SCENE)
    low=prepare_low(out/'low_source')
    items=[('main1',main1.bundle,main1.q),('main7',main7.bundle,main7.q),
           ('secondary',secondary['bundle'],secondary['q']),('low',low.bundle,low.q0)]
    ids=sorted({b.calibration.installation_id for name,b,q in items})
    slots={name:i+1 for i,name in enumerate(ids)};blocks=[];start=4
    qparts=[np.r_[0.,[original.calibrations[name].pivot_xyz[2]*SCENE_UNIT_M for name in ids]]]
    for name,bundle,q in items:
        blocks.append(Block(bundle,q,name,slots[bundle.calibration.installation_id],start))
        qparts.append(q[4:]);start+=len(q)-4
    q=np.concatenate(qparts);refs={name:next(b for b in blocks if b.bundle.calibration.installation_id==name)for name in ids}
    quadratic=np.r_[np.concatenate([b.quadratic for b in blocks]),np.ones(3,bool)]
    scales=np.concatenate([np.r_[b.bundle.native_residual_sigma,np.ones(len(b.bundle.stamps))]for b in blocks]+[np.ones(3)])
    def evaluate(q,jacobian=True):
        results=[b.evaluate(q,jacobian)for b in blocks]
        values=[r[0]if jacobian else r for r in results];matrices=[r[1]for r in results]if jacobian else []
        for name in ids:
            block=refs[name];local=block.local(q);bundle=block.bundle;p=bundle.reference_pts
            cal=bundle.current_calibration(local);cam,_,vj=cal.camera25_and_jacobians(bundle.view(local,p))
            values.append(np.array([(cam[2]*SCENE_UNIT_M-1.6)/.18]))
            if jacobian:
                row=np.zeros((1,len(q)));row[0,slots[name]]=1./.18
                pos=block.start+4*bundle.index[p];row[0,pos:pos+4]=vj[2]*SCENE_UNIT_M/.18
                matrices.append(csr_matrix(row))
        value=np.concatenate(values)
        return(value,vstack(matrices,format='csr'))if jacobian else value
    def cost(r):return huber_cost(r[~quadratic],True)+.5*float(r[quadratic]@r[quadratic])
    source_bindings=dict(current=ref(CURRENT),scene_base=ref(SCENE),main1=main1.manifest,main7=main7.manifest,
        secondary=secondary['input_manifest'],low=low.manifest,clipped_left_context=clip,
        global_variables=['one common complete F-column X increment in metres','three independent shared installation pivot heights in metres','all retained actual source R3/logf'],
        fixed=['all pivot XY','all nonzero constant levers','PP/k1','radii/rowY/R/Lane/board/ad/metric'],
        source_roles_and_UV_sigma_preserved=True,old_main4_candidate_not_used=True,main4_main9_not_recovered_in_this_diagnostic=True,
        body_rebound=False,full_programme_exported=False,accuracy_accepted=False,entrypoint=ref(__file__))
    write(out/'input_manifest.json',source_bindings)
    # Replay actual shared-source cost directions. Depth profiles can have
    # KKT switches, so this checks the envelope cost, rather than claiming a
    # smooth residual derivative on every untyped near-zero motion row.
    residual,jac=evaluate(q);weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12));weights[quadratic]=1.
    gradient=np.asarray(jac.T@(weights*residual)).ravel();checks=[]
    for column in range(4):
        h=1e-5;plus=q.copy();minus=q.copy();plus[column]+=h;minus[column]-=h
        fd=(cost(evaluate(plus,False))-cost(evaluate(minus,False)))/(2*h)
        relative=abs(fd-gradient[column])/max(1.,abs(fd),abs(gradient[column]))
        checks.append(dict(column=column,actual_source_cost_direction=float(gradient[column]),central_source_cost_direction=float(fd),relative_error=relative))
    write(out/'actual_global_source_cost_direction_checks.json',checks)
    print(json.dumps(dict(stage='joint_full_source_prepared',actual=sum(len(b.bundle.stamps)for b in blocks),motion=sum(len(b.bundle.a)for b in blocks),global_source_cost_checks=checks)),flush=True)
    if max(c['relative_error']for c in checks)>5e-3:raise ValueError('Global actual source envelope cost derivative failed')
    damping=1e-3;history=[];started=time.perf_counter()
    while True:
        for block in blocks:
            block.bundle.robust_phase=True
            preliminary=block.evaluate(q,False)
            block.bundle.profile_weights=np.minimum(1.,2./np.maximum(np.abs(preliminary[:block.bundle.motion_rows]),1e-12)).reshape(-1,2)
        residual,jac=evaluate(q);before=cost(residual);weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12));weights[quadratic]=1.
        jw=jac.multiply(np.sqrt(weights)[:,None]);normal=(jw.T@jw).tocsc();gradient=jac.T@(weights*residual)
        diagonal=np.maximum(normal.diagonal(),1.);gn=spsolve(normal+diags(1e-8*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
        gn_step=float(np.max(np.abs((jac@gn)*scales)));accepted=False;reason=None
        for trial in range(12):
            step=spsolve(normal+diags(damping*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A');nextq=q+step
            try:
                if not np.isfinite(nextq).all()or np.any(nextq[1:4]<=0.):raise ValueError('Physical independent pivot height domain')
                for block in blocks:
                    focal=nextq[block.start+3:block.start+len(block.q0)-4:4]
                    if np.any((focal<np.log(100.))|(focal>np.log(50000.))):raise ValueError('Physical focal numerical domain')
                after=cost(evaluate(nextq,False))
                if after<before-1e-10*max(1.,before):q=nextq;accepted=True;damping=max(damping*.3,1e-12);break
                reason='objective_not_lower_at_profile_precision'
            except(ValueError,RuntimeError)as exc:reason=str(exc)[:180]
            damping*=10.
        record=dict(iteration=len(history),accepted=accepted,cost=before,next_cost=after if accepted else before,
            trial=trial+1,damping=damping,failure_reason=None if accepted else reason,
            maximum_undamped_GN_native_step_px=gn_step,scaled_gradient_maximum=float(np.max(np.abs(gradient)/np.sqrt(diagonal))),
            F_column_dx_from_hard_scene_m=main1.manifest['shared_F_scene_binding']['F_column_dx_m']+float(q[0]),
            pivot_heights_m={name:float(q[slots[name]])for name in ids},elapsed_seconds=time.perf_counter()-started)
        history.append(record);write(out/'iterations.json',history)
        np.savez_compressed(out/'latest_accepted_state.npz',q=q,block_start=np.array([b.start for b in blocks]),block_lengths=np.array([len(b.q0)-4 for b in blocks]))
        print(json.dumps(dict(stage='joint_full_source_iteration',**record)),flush=True)
        if not accepted or gn_step<.05:break
    scene=copy.deepcopy(main1.candidate)
    for f in scene['ice_features']:
        if f['id']in ('F1','F2','F3','F4'):
            f['center_xyz'][0]+=float(q[0])/SCENE_UNIT_M
            for ring in f['rings']:ring['center_xyz'][0]+=float(q[0])/SCENE_UNIT_M
    write(out/'candidate_scene_concentric_UNACCEPTED.json',scene)
    cals=dict(original.calibrations);updated=dict(allviews);recovered=[]
    for block in blocks:
        local=block.local(q);cal=block.bundle.current_calibration(local);cals[cal.installation_id]=cal
        for p in block.bundle.stamps:updated[p]=CanonicalView(cal.installation_id,block.bundle.view(local,p));recovered.append(p)
    model=PhysicalCameraModel(cals);_,native,full_aliases,_=PhysicalCameraModel.load_npz(CURRENT)
    path=out/'full_camera_WITH_UNRECOVERED_MAIN4_MAIN9_UNACCEPTED.npz'
    model.save_npz(path,native,full_aliases,updated,source_sha256=SOURCE_SHA,time_base='1/15360')
    write(out/'report.json',dict(source_bindings,camera_candidate=ref(path),scene_candidate=ref(out/'candidate_scene_concentric_UNACCEPTED.json'),
        recovered_actual_exposures=len(recovered),unrecovered_main_camera_Pz_changed=True,history=history,seconds=time.perf_counter()-started))
    print({'camera':str(path),'scene':str(out/'candidate_scene_concentric_UNACCEPTED.json'),'accuracy_accepted':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
