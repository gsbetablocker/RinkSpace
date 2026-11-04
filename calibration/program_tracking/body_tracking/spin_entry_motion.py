"""Continuous S05 entry-motion repair on saved ExMR body/hand and footwear.

Retains the solved spin as a reference layer. The entire 76--95 source clip
is adjustable, including the previously frozen flying entry. No inference,
render alignment, fixed spin centre or airborne-to-ice projection is used.
"""
from pathlib import Path
import argparse,copy,json,os,shutil,time
import numpy as np
from scipy.interpolate import PchipInterpolator,make_interp_spline
from scipy.sparse import vstack
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .support_root_placement import restore_parent,smoothstep
from .action_integration import ClipReference
from .exmr_integration import PROGRAMME,CASE_FIELD,case_identity,save_packet
from .skate_checkpoint import checkpoint_identity
from .skate_integration import JointContactProblem,restore_checkpoint_layers,source_evidence
from .body_execution import BodyExecution,BodyExecutionConfig
from .spin_grasp_factors import SpinGraspFactors
from .entry_motion_factors import EntryMotionFactors
from .ballistic_com_factors import BallisticCOMFactors
from .support_image import SupportImageFactors
from .skate_run_policy import RunPolicy
from .stage_quality import StageQualityTrace
from .support_refinement import measure


class SpinEntryFactors(SpinGraspFactors):
    def __init__(self,problem,relations):
        super().__init__(problem,relations)
        self.entry=EntryMotionFactors(problem,relations['factor_parameters']['entry_motion'])
        self.ballistic=BallisticCOMFactors(problem,relations['factor_parameters']['flight_ballistic'])

    def evaluate(self,x,jacobian=False):
        base=super().evaluate(x,jacobian);extra=self.entry.evaluate(x,jacobian)
        flight=self.ballistic.evaluate(x,jacobian)
        extra['block_slices'][self.ballistic.name]=slice(len(extra['residual']),len(extra['residual'])+len(flight['residual']))
        extra['residual']=np.r_[extra['residual'],flight['residual']]
        if jacobian:extra['jac']=vstack((extra['jac'],flight['jac']),format='csr')
        extra['parts'].update(flight['parts'])
        start=len(base['residual']);base['residual']=np.r_[base['residual'],extra['residual']]
        if jacobian:base['jac']=vstack((base['jac'],extra['jac']),format='csr')
        base['parts'].update(extra['parts'])
        for name,s in extra['block_slices'].items():base['block_slices'][name]=slice(start+s.start,start+s.stop)
        base['metrics']['entry_motion']=extra['metrics'];base['metrics']['flight_com_ballistic']=flight['metrics'];self.last=base
        return base


def prepare(project,base,previous,parent_parameter,output,gravity_refine=False):
    """Consume the saved candidate once; do not repeat prior inference/pointing."""
    old,camera,info,parent=restore_parent(project,base,previous,output/'parent_geometry',parent_parameter)
    case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True)
    old_case=previous/'common/exmr_hand_nofoot'
    for name in ('input.npz','observations.npz','compact_smplx.npz','export_source.npz'):
        shutil.copy2(old_case/name,case/name)
    for name in ('exmr_body_hand_nofoot.npz','palm_grip_geometry.json','support_and_free_shoe_regions.json','source_support_track.npz'):
        shutil.copy2(previous/name,output/name)
    initial=load(old_case/'initial.npz');times=old.times;source=old.p['pts']/12800.
    current=old.at(times)
    # Safe surrounding states seed a monotone entry depth. They are a model
    # initialization, not newly claimed 3D measurements. Z and articulation
    # remain exactly as saved; native image/physics factors determine the fit.
    anchors=np.array([80.,80.4,80.8,82.4,82.8,83.2])
    index=np.searchsorted(source,anchors)
    if not np.allclose(source[index],anchors,rtol=0,atol=1e-10):raise ValueError('Entry source anchors absent')
    path=PchipInterpolator(anchors,current['P'][index,:2],axis=0,extrapolate=True)
    weight=smoothstep((source-80.4)/.4)*smoothstep((82.8-source)/.4)
    shift=np.zeros_like(initial['P']);shift[:,:2]=weight[:,None]*(path(source)-current['P'][:,:2])
    if gravity_refine:
        from types import SimpleNamespace
        from scipy.interpolate import CubicSpline
        from .skate_integration import window_factory
        relations_before=json.loads((previous/'grasp_relations.json').read_text())
        state=window_factory('spin_support_v2')(old,(75.5,95.5),'full')
        shell=SimpleNamespace(reference=old,camera=camera,binding=old.binding,body=SimpleNamespace(state=state),size=state.size+17)
        flight=BallisticCOMFactors(shell,relations_before['factor_parameters']['flight_ballistic'])
        C=flight.geometry_at(np.zeros((len(flight.times),69)))[0].position_world_m
        tt=flight.clock.elapsed_s;A=np.c_[np.ones(len(tt)),tt]
        linear=A@np.linalg.lstsq(A,C+flight.gravity_offset,rcond=None)[0]
        correction=(linear-flight.gravity_offset-C)/old.binding.ratio
        knots=np.r_[81.20,81.28,old.p['pts'][flight.frames]/12800.,81.52,81.60]
        values=np.vstack((np.zeros((2,3)),correction,np.zeros((2,3))))
        dense=CubicSpline(knots,values,bc_type='clamped')(np.clip(source,knots[0],knots[-1]))
        dense[(source<knots[0])|(source>knots[-1])]=0.
        shift=dense
    initial['P']=initial['P']+shift
    curve=make_interp_spline(times,initial['P'],k=3)
    initial.update(root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),root_curve_knots=curve.t,
        root_curve_controls=curve.c,root_curve_degree=np.array(3),root_curve_origin=np.array('entry depth branch removed from saved reference; same native PTS, current body layers retained'))
    np.savez_compressed(case/'initial.npz',**initial)
    config=dict(source_interval_seconds=[80.8,82.4],taper_seconds=.3,acceleration_scale_m_s2=4.,weight=1.,exclude_source_intervals_seconds=[[81.32,81.52]],
        status='finite phase-specific engineering prior; no calibrated acceleration acceptance threshold',vertical_acceleration_untouched=True)
    relations=json.loads((previous/'grasp_relations.json').read_text())
    relations['factor_parameters']['entry_motion']=config
    relations['factor_parameters']['flight_ballistic']=dict(residual_form='acceleration' if gravity_refine else 'position',acceleration_scale_m_s2=4.,source_interval_seconds=[81.36,81.44],source_id=relations['source_id'],gravity_world_mps2=[0.,0.,-9.81],position_scale_m=.03,acceleration_diagnostic_scale_m_s2=4.,weight=20.,confidence=.8,relative_mass_per_skate=.02,parameter_status='finite engineering uncertainties for a declared mass proxy, not calibrated athlete mass or hard physical acceptance')
    # 81.70 source exposure visibly has landed. Leave the ambiguous takeoff/
    # airborne interval without a support factor; start after a small buffer.
    relations['support_intervals'][0]['start_pts']=round(81.62*12800)
    source_review=base/'spin_support_foot_1104/entry_repair/native_1044480.png'
    relations['entry_contact_review']=dict(source_pts=1044480,source_seconds=81.60,source_frame=str(source_review),
        source_frame_sha256=digest(source_review),support_start_seconds=81.62,support_foot=1,
        anatomical_identity='same physical support episode traced back from the existing right-foot association; no foot switch visible',
        onset_uncertainty='81.32 takeoff boundary; 81.36/81.40/81.44 both blades airborne; 81.48--81.56 landing uncertainty; 81.60 clearly grounded',
        provenance='native source-image sequence review',interval_is_inference=True,per_frame_contact_truth=False)
    write_json(output/'grasp_relations.json',relations)
    document=json.loads((output/'support_and_free_shoe_regions.json').read_text())
    ref=ClipReference(old,case,camera.p,initial)
    restore_checkpoint_layers(ref,parent,include_primary=True)
    ref.grasp_relations=relations;ref.grasp_geometry=json.loads((output/'palm_grip_geometry.json').read_text())
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance={'status':'existing admitted source regions; native observations unchanged'})
    identity=case_identity(project,base,output,'exmr_hand_nofoot',info)
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    ref.hashes={k:identity[k] for k in ('native_input_sha256','camera_prior_sha256','shared_s01_parameters_sha256')}
    write_json(case/'comparison_case_identity.json',identity)
    seed=dict(method=('current mass-proxy CoM projected to known-gravity trajectory as initial reference, then full joint acceleration-constrained solve' if gravity_refine else 'monotone XY reference bridge across the diagnosed 81.3--81.8 depth branch; native cubic root curve'),
        bridge_source_seconds=([81.20,81.60] if gravity_refine else [80.4,82.8]),full_weight_source_seconds=([81.36,81.44] if gravity_refine else [80.8,82.4]),surrounding_model_anchor_seconds=([] if gravity_refine else anchors.tolist()),
        native_samples=len(source),root_z_unchanged=not gravity_refine,Q_body_hands_unchanged=True,old_refinement_layers_retained=True,
        initialization_not_3d_truth=True,parent_parameter=str(parent_parameter),parent_parameter_sha256=digest(parent_parameter),
        max_reference_shift_m=float(np.linalg.norm(shift,axis=1).max()*ref.binding.ratio))
    write_json(output/'entry_initialization.json',seed)
    return ref,camera,info,parent,document,relations,seed


def run(project,base,previous,parent_parameter,output,parameter,budget=600.,steps=8,gravity_refine=False):
    project,base,previous,parent_parameter,output,parameter=map(lambda x:Path(x).resolve(),(project,base,previous,parent_parameter,output,parameter))
    from rinkspace.path_policy import guard_project_write
    guard_project_write(parameter,project_root=project)
    if output.exists() or parameter.exists():raise FileExistsError('Existing entry result is preserved')
    native=base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll'
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(native)
    output.mkdir(parents=True)
    print(json.dumps(dict(main_log=str(output/'progress.jsonl'),output=str(output),budget_seconds=budget,max_accepted=steps,source_context_seconds=[76,95])),flush=True)
    started=time.perf_counter();execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
    try:
        with (output/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log,execution.limits():
            ref,camera,info,parent,doc,relations,seed=prepare(project,base,previous,parent_parameter,output,gravity_refine)
            execution.attach(ref)
            # Extend both curve boundaries beyond the source range, so all
            # source exposures and their five shutter nodes have free controls.
            problem=JointContactProblem(ref,camera,output/'common/exmr_hand_nofoot/observations.npz',(75.5,95.5),[],output/'solve',log,
                'spin_support_v2',optimize_shared=False,record_acceptance_before_jacobian=True)
            runtime=copy.deepcopy(doc)
            for row in runtime['observations']:row['objective_weight']*=relations['factor_parameters']['shoe_region_weight_multiplier']
            problem.support_image=SupportImageFactors(problem,runtime,arc_ids={})
            problem.action_factors=SpinEntryFactors(problem,relations)
            mass=problem.action_factors.ballistic.model
            write_json(output/'mass_model.json',{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in mass.items()})
            problem.al.penalty=float(parent['penalty'])
            if np.any(parent['multiplier']!=0):raise ValueError('Nonzero parent AL requires row-label mapping before expanding the time domain')
            problem.checkpoint_extra=dict(hand_pose_schema_version=parent['hand_pose_schema_version'].copy(),
                grasp_relations_sha256=np.array(digest(output/'grasp_relations.json')),grasp_geometry_sha256=np.array(digest(output/'palm_grip_geometry.json')),
                source_support_track_sha256=np.array(digest(output/'source_support_track.npz')),
                initialization_provenance_json=np.array(json.dumps(seed,sort_keys=True)),entry_motion_config_json=np.array(json.dumps(relations['factor_parameters']['entry_motion'],sort_keys=True)))
            # Keep the existing immutable original-G comparison and current
            # source mask. The entry statistics are reported separately.
            parent_report=base/'spin_support_foot_1104/solve/candidate.json'
            baseline=json.loads(parent_report.read_text())['cross_stage_quality']['original_A']
            review=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
            problem.stage_quality=StageQualityTrace(output/'cross_stage_quality.json',dict(parent_parameter_sha256=digest(parent_parameter),
                fixed_original_G_baseline=str(parent_report),source_mask='same current 88--92 source mask; separate full-entry diagnostics'),baseline,
                lambda x,v:measure(problem,review,x,v))
            x=np.r_[ref.binding.x,np.zeros(problem.body.size)]
            np.savez_compressed(output/'initial_checkpoint.npz',**problem.checkpoint_fields(x))
            tick=time.perf_counter();v=problem.evaluate(x,True);elapsed=time.perf_counter()-tick
            problem.jacobian_seconds_hint=elapsed
            write_json(output/'execution.json',dict(jacobian_seconds=elapsed,source_frames=problem.body.n,active_columns=problem.body.state.active.tolist(),
                free_controls_at_all_source_frames=bool(np.all(np.max(np.abs(problem.body.state.basis(ref.times)),axis=1)>0)),
                body_support_seconds=list(problem.body.state.seconds),execution=execution.record(),new_reference=seed))
            remaining=budget-(time.perf_counter()-started)
            if remaining<2*elapsed:raise RuntimeError('Measured budget cannot fund actual progress')
            result,report=problem.solve(x,RunPolicy(wall_seconds=remaining,max_accepted=steps,max_outer=1))
            saved=load(problem.output/'candidate.npz')
            initial_archive=project/PROGRAMME/('spin_entry_gravity_initial_1104.npz' if gravity_refine else 'spin_entry_motion_initial_1104.npz')
            guard_project_write(initial_archive,project_root=project)
            if initial_archive.exists():raise FileExistsError('Preserve existing entry initial reference')
            shutil.copy2(output/'common/exmr_hand_nofoot/initial.npz',initial_archive)
            saved.update(initial_reference=np.array(initial_archive.relative_to(project).as_posix()),initial_reference_sha256=np.array(digest(initial_archive)),
                original_gvhmr_pose_anchor=np.array(False),whole_source_state_adjustable=np.array(True))
            np.savez_compressed(parameter,**saved)
            ref.new_refinements.append((problem.body.state,saved['x'][17:].copy()))
            supports=source_evidence(output/'common/exmr_hand_nofoot',shot_id='s05',source_pts=ref.p['pts'],records=doc['supports'])
            packet=save_packet(problem,info,supports,doc,output/'common/exmr_hand_nofoot',status='joint_fit',report=report)
            summary=dict(parameter=str(parameter),result=str(output/'common/exmr_hand_nofoot/fitted/result.npz'),accepted=problem.accepted,
                retained_step=report['selection']['retained_step'],wall_seconds=time.perf_counter()-started,budget_seconds=budget,
                source_context_seconds=[76,95],source_frames=len(packet['pts']),whole_shot_adjustable=True,accuracy_accepted=False,
                parent_parameter=str(parent_parameter),initialization=seed,stop=report['stop'])
            write_json(output/'run_summary.json',summary)
            print(json.dumps({k:summary[k] for k in ('parameter','result','accepted','retained_step','wall_seconds')}),flush=True)
            return summary
    finally:execution.close()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--previous-output',type=Path,required=True);p.add_argument('--parent-parameter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parameter',type=Path,required=True);p.add_argument('--gravity-refine',action='store_true');p.add_argument('--budget',type=float,default=600.);p.add_argument('--max-steps',type=int,default=8);a=p.parse_args(argv)
    return run(a.project,a.base_root,a.previous_output,a.parent_parameter,a.output,a.parameter,a.budget,a.max_steps,a.gravity_refine)

if __name__=='__main__':main()
