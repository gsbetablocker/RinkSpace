"""Native-aware complete-graph assembly hooks for one shared F2 delta.

Unlike ``CompositeFactorTape``, this module never rekeys a constructed native
factor. Each group's model and prior are rekeyed first, then the production
builder/native manager create factors with correct internal keys. Use
``append_builders_to_one_graph`` and ``NativeManagerMux`` for a single global
factor namespace suitable for ``CompleteFactorSession``; concatenating two
already-built graph copies remains diagnostic-only because native fallback
records would retain their separate local factor indices.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gtsam
import numpy as np

import fit_continuous_rig_camera as base
from backend.isam2 import IncrementalCameraGraph
import fit_continuous_source_identity_repaired as main_entry
import fit_low_end_temporal_principal as low_entry
from continuous_full_timeline_regularization import add_complete_motion_priors
from continuous_quadratic_rig_optin import add_quadratic_priors
from continuous_reference_calibration import (
    ReferenceCalibration, attach_reference_factors)
from continuous_rig_rekey import SCENE_RING_KEYS, rekey_rig_pair
from continuous_shared_f2_composite import (
    F2_KEY, captured_f2_spec, shift_absolute_f2_callback,
    scene_prior_in_candidate_coordinates)
from continuous_time_principal_optin import add_time_principal_priors
from shared_f2_nested_factor import NestedF2Factor


@contextmanager
def capture_low_builder_context(context: dict):
    """Observe one original preflight build; restore every runtime wrapper.

    This is for a dedicated independent capture process. It does not change
    source-selection code on disk or modify a running solver process.
    """
    import historic_development_ads as ads
    import continuous_low_end_f1_correlated_optin as f1
    import continuous_time_principal_optin as time_model
    saved=[]
    def patch(module,name,record):
        old=getattr(module,name)
        def wrapper(*args,**kwargs):
            result=old(*args,**kwargs)
            record(args,kwargs,result)
            return result
        setattr(module,name,wrapper)
        saved.append((module,name,old))
    patch(base,'build_graph',lambda a,k,r: context.update(
        data=a[0],pairs=a[1],model=a[2],prior=a[3],clock=a[4],
        valid_pairs=a[5],graph=r[0],counts=r[1]))
    patch(ads,'attach_development_ad_factors',lambda a,k,r: context.update(
        ad_rows=a[4],construction_initial=a[3],ad_metadata=r))
    patch(low_entry,'attach_reference_factors',lambda a,k,r: context.update(
        references=a[2],reference_metadata=r))
    patch(f1,'attach_correlated_f1_factors',lambda a,k,r: context.update(
        f1_inventory=a[3],f1_metadata=r))
    patch(time_model,'add_time_principal_priors',lambda a,k,r: context.update(
        time_principal_metadata=a[1],time_principal_priors=r))
    try:
        yield
    finally:
        for module,name,old in reversed(saved):
            setattr(module,name,old)


@contextmanager
def capture_main_builder_context(context: dict):
    """Use the source-clock driver's existing full-context capture, plus
    the exact original attach initializer required for callback replay."""
    import historic_development_ads as ads
    from fit_source_pose_clock_repaired import capture_context
    with capture_context(context):
        previous=ads.attach_development_ad_factors
        def wrapped(*args,**kwargs):
            result=previous(*args,**kwargs)
            context['construction_initial']={int(k):v.copy()
                                             for k,v in args[3].items()}
            return result
        ads.attach_development_ad_factors=wrapped
        try:
            yield
        finally:
            ads.attach_development_ad_factors=previous


@dataclass
class RebuiltRig:
    camera_id: str
    graph: object
    manager: object
    resources: dict
    states: dict[int, np.ndarray]
    prior: dict[int, np.ndarray]
    programme: object
    references: object
    factor_counts: dict
    original_factors: int
    f2_released_factor_count: int
    key_mapping: dict[int, int] | None = None
    new_source_reference_model: object | None = None
    new_source_reference_rows: list | None = None


@contextmanager
def append_builders_to_one_graph(graph: IncrementalCameraGraph):
    """Make production ``base.build_graph`` append to a caller-owned graph.

    Each native manager sees the graph's true global pending-factor length
    while constructing its factors. This gives source-edge fallback records
    globally unique IDs and correct internal rekeyed node IDs. Use only in
    a dedicated joint construction process; no source file is altered.
    """
    original=base.IncrementalCameraGraph
    def existing_graph(state_dim,*args,**kwargs):
        if state_dim!=graph.state_dim:
            raise ValueError('Joint graph state dimension differs')
        if kwargs.get('window_size',0)>graph.window_size:
            raise ValueError('Joint graph window is too small for one rig')
        return graph
    base.IncrementalCameraGraph=existing_graph
    try:
        yield
    finally:
        base.IncrementalCameraGraph=original


class NativeManagerMux:
    """Route complete-graph native preflight/fallback to the owning manager.

    The low temporal-principal manager owns affine priors only and has zero
    source-edge records. Main owns the source-edge records with *global* IDs
    because both graphs were appended to the same IncrementalCameraGraph.
    Existing CompleteFactorSession can use this object unchanged.
    """
    def __init__(self, main_manager, low_manager):
        if set(main_manager._records)&set(low_manager._records):
            raise ValueError('Native manager factor IDs collide')
        self.main_manager=main_manager
        self.low_manager=low_manager
        self._records={**main_manager._records,**low_manager._records}

    def __getattr__(self,name):
        return getattr(self.main_manager,name)

    def verify_accepted(self, graph, states, **kwargs):
        failures=[]
        reports={}
        for name,manager in (('main_long_side',self.main_manager),
                             ('low_end',self.low_manager)):
            found,report=manager.verify_accepted(graph,states,**kwargs)
            failures.extend(found)
            reports[name]=report
        return failures,dict(schema='rinkspace.multi-rig-native-parity.v1',
                             by_rig=reports)

    def replace_unsupported(self,graph,error):
        factor_id=int(error.factor_id)
        owners=[manager for manager in (self.main_manager,self.low_manager)
                if factor_id in manager._records]
        if len(owners)!=1:
            raise ValueError('Native fallback factor has no unique rig owner')
        return owners[0].replace_unsupported(graph,error)

    def metadata(self):
        main=self.main_manager.metadata()
        low=self.low_manager.metadata()
        return dict(schema='rinkspace.multi-rig-native-manager.v1',
                    by_rig={'main_long_side':main,'low_end':low},
                    fallback_events=[*main.get('fallback_events',[]),
                                     *low.get('fallback_events',[])])

    def close_after_release(self):
        # _records is an independent dict of the managers' native records;
        # each record owns its GTSAM factor. Release this extra owner before
        # the managers unload their native callbacks/DLLs.
        self._records.clear()
        self.low_manager.close_after_release()
        self.main_manager.close_after_release()


def new_joint_graph(total_state_count: int):
    """Use the same backend settings as the complete source rig builders."""
    if total_state_count<2:
        raise ValueError('Joint graph needs at least two states')
    return IncrementalCameraGraph(11,window_size=total_state_count+2,
        relinearize_threshold=1e-5,refinement_updates=1,
        optimization='dogleg',dogleg_initial_delta=.001,diagnose_cost=True)


def rekey_reference_calibration(prepared: ReferenceCalibration, rekeyed,
                                initial: dict[int, np.ndarray]):
    """Move only reference graph-node IDs; source PTS/UV/XYZ are copied."""
    rows = []
    for original in prepared.references:
        row = copy.deepcopy(original)
        for field in ('frame_key', 'control_key'):
            row[field] = rekeyed.key_mapping.get(int(row[field]), int(row[field]))
        rows.append(row)
    metadata = copy.deepcopy(prepared.metadata)
    metadata['new_graph_nodes'] = {
        str(pts): {field: rekeyed.key_mapping.get(int(value), int(value))
                   for field, value in node.items()}
        for pts, node in metadata['new_graph_nodes'].items()}
    if set(rekeyed.references.frames) != {int(r['source_pts']) for r in rows}:
        raise ValueError('Reference rekey lost source exposure identity')
    return ReferenceCalibration(rekeyed.references, initial, rows, metadata)


def candidate_delta_data(data: dict, q_saved, *, expected_f2: int):
    """Copy captured data and change only the 2001 callback's scene coordinate.

    Captured residual/Jacobian closures and their pointwise precision/robust
    transforms remain the original ones inside the additive wrapper. Other
    source factors and all source metadata keep their original objects.
    """
    selected = 0
    retained = []
    for original in data['retained']:
        if captured_f2_spec(original) and int(original['keys'][0]) in data['pts']:
            keys, residual, derivative, options = shift_absolute_f2_callback(
                original, q_saved)
            spec = dict(original)
            args = list(original['args'])
            args[1] = residual
            spec['args'] = tuple(args)
            spec['kwargs'] = dict(original['kwargs'], jac_fn=derivative)
            if (tuple(map(int, spec['keys'])) != keys
                    or spec['kwargs']['residual_dim'] != options['residual_dim']
                    or spec['kwargs']['sigma'] != options['sigma']):
                raise ValueError('Captured F2 factor options changed')
            retained.append(spec)
            selected += 1
        else:
            retained.append(original)
    if selected != expected_f2:
        raise ValueError(f'Expected {expected_f2} captured F2 factors, got {selected}')
    return {**data, 'retained': retained}


def _base_native_graph(camera_id, context, rekeyed, source_data,
                       args, resources, *, q_saved, release_count):
    if set(rekeyed.states) != set(rekeyed.prior):
        raise ValueError('Rekeyed original state/prior keys differ')
    original_camera_prior = {k: v.copy() for k, v in rekeyed.prior.items()}
    initial = {**original_camera_prior, F2_KEY: np.zeros(11)}
    if F2_KEY in original_camera_prior:
        raise ValueError('Camera prior unexpectedly already has scene key')
    if camera_id == 'low_end':
        manager, affine = low_entry.make_linear_prior_backend(
            args, rekeyed.programme, original_camera_prior, resources)
        source_data = candidate_delta_data(source_data, q_saved,
            expected_f2=release_count)
        native_edges = None  # time-principal old native source role is unsupported
    elif camera_id == 'main_long_side':
        manager, affine = main_entry.continuity.make_native_backend(
            args, rekeyed.programme, original_camera_prior, resources)
        source_data = candidate_delta_data(source_data, q_saved,
            expected_f2=release_count)
        native_edges = manager
    else:
        raise ValueError('Only reviewed main/low rigs supported')
    graph, counts = base.build_graph(source_data, context['pairs'],
        rekeyed.programme, initial, context['clock'], context['valid_pairs'],
        .25, native_edge_manager=native_edges, graph_owner=resources,
        affine_priors=affine)
    graph._adaptive_curve_worker_count = int(args.reference_curve_workers)
    return graph, manager, affine, initial, counts, source_data


def rebuild_main_complete(context, phase, programme, references, states, prior,
                          q_saved, args, *, resources=None) -> RebuiltRig:
    """Rebuild all 27,481 main factors from preselected original context.

    Context keys match ``fit_source_pose_clock_repaired.capture_context`` plus
    ``construction_initial`` (the old attach_* initial, not a prior target).
    The 14 new F2 source references are attached by a separate explicit step.
    """
    resources = {} if resources is None else resources
    required = {'data','pairs','clock','valid_pairs','references','ad_rows',
                'near_rows','motion_rows','motion_metadata','construction_initial'}
    if required-set(context):
        raise ValueError(f'Incomplete original main builder context: {required-set(context)}')
    rekeyed = rekey_rig_pair(programme, references, states, prior, phase,
        key_start=3_000_000_000, reference_shot_start=11_000,
        reserved_keys=SCENE_RING_KEYS)
    factor_start=len(resources['graph']._pending) if 'graph' in resources else 0
    graph, manager, affine, initial, counts, data = _base_native_graph(
        'main_long_side', context, rekeyed, context['data'], args, resources,
        q_saved=q_saved, release_count=0)
    old_build_initial = {rekeyed.key_mapping.get(int(k),int(k)): v.copy()
                         for k,v in context['construction_initial'].items()}
    if set(old_build_initial) != set(rekeyed.states):
        raise ValueError('Main source attachment initializer keyset changed')
    old_build_initial[F2_KEY] = np.zeros(11)
    refs = rekey_reference_calibration(context['references'], rekeyed, initial)
    attach_reference_factors(graph, data, refs, adaptive_curves=True)
    from historic_development_ads import attach_development_ad_factors
    attach_development_ad_factors(graph, data, rekeyed.programme,
                                  old_build_initial, context['ad_rows'])
    add_quadratic_priors(graph, rekeyed.programme, old_build_initial,
                         affine_priors=affine)
    add_complete_motion_priors(graph, rekeyed.programme,
        context['clock'], phase['exposure_clock'], context['valid_pairs'],
        affine_priors=affine)
    from main_nearboard_normal_factors import attach_main_nearboard_normal_factors
    attach_main_nearboard_normal_factors(graph,data,rekeyed.programme,
        old_build_initial,context['near_rows'])
    from continuous_reviewed_surface_motion import attach_reviewed_main_motion_factors
    attach_reviewed_main_motion_factors(graph,data,rekeyed.programme,
        old_build_initial,context['motion_rows'],
        reviewed_metadata=context['motion_metadata'])
    expected = int(phase['graph_factors_with_source_calibration'])
    if len(graph._pending)-factor_start != expected:
        raise ValueError(f'Main native-aware factor count '
                         f'{len(graph._pending)-factor_start} != {expected}')
    return RebuiltRig('main_long_side',graph,manager,resources,
        {**rekeyed.states,F2_KEY:np.zeros(11)},
        {**rekeyed.prior,F2_KEY:np.zeros(11)},
        rekeyed.programme,rekeyed.references,counts,expected,0,
        rekeyed.key_mapping)


def rebuild_main_source_pose_complete(context, phase, programme, references,
                                     states, prior, preflight_initial, q_saved,
                                     args, *, resources=None) -> RebuiltRig:
    """Rebuild the corrected source-pose/clock graph, preserving its priors.

    ``context`` is the original source-only capture. ``phase`` and all four
    model/state inputs come from the distinct v14 source-pose reconstruction.
    Camera factors retain their original UV while pose binding and temporal
    priors follow v14. No old exposure alias replaces a v14 pose key.
    """
    import fit_source_pose_clock_repaired as pose_entry
    from historic_development_ads import attach_development_ad_factors
    from continuous_reviewed_surface_motion import attach_reviewed_main_motion_factors
    resources={} if resources is None else resources
    required={'data','pairs','references','ad_rows','near_rows','motion_rows',
              'motion_metadata'}
    if required-set(context):
        raise ValueError(f'Incomplete source-pose capture: {required-set(context)}')
    if not phase.get('source_pose_groups') or not phase.get('source_camera_rebinding'):
        raise ValueError('New source-pose graph phase is required')
    if set(states)!=set(prior) or set(states)!=set(preflight_initial):
        raise ValueError('v14 accepted, prior, and preflight node keys differ')
    rekeyed=rekey_rig_pair(programme,references,states,prior,phase,
        key_start=3_000_000_000,reference_shot_start=11_000,
        reserved_keys=SCENE_RING_KEYS)
    attachment_initial={rekeyed.key_mapping.get(int(k),int(k)):
                        np.asarray(value,np.float64).copy()
                        for k,value in preflight_initial.items()}
    if set(attachment_initial)!=set(rekeyed.states):
        raise ValueError('Rekeyed v14 preflight initializer does not cover model')
    attachment_initial[F2_KEY]=np.zeros(11)
    clock={int(k):float(v) for k,v in phase['frame_clock_virtual_pts'].items()}
    valid_pairs={tuple(map(int,row)) for row in
                 phase['valid_temporal_acceleration_pairs']}
    source_context={**context,'clock':clock,'valid_pairs':valid_pairs}
    factor_start=len(resources['graph']._pending) if 'graph' in resources else 0
    graph,manager,affine,initial,counts,data=_base_native_graph(
        'main_long_side',source_context,rekeyed,context['data'],args,resources,
        q_saved=q_saved,release_count=0)
    prepared=rekey_reference_calibration(context['references'],rekeyed,initial)
    attach_reference_factors(graph,data,prepared,adaptive_curves=True)
    pose_entry.attach_same_ad_observations(graph,context,rekeyed.programme,
                                           attachment_initial)
    add_quadratic_priors(graph,rekeyed.programme,attachment_initial,
                         affine_priors=affine)
    add_complete_motion_priors(graph,rekeyed.programme,clock,
        phase['exposure_clock'],valid_pairs,affine_priors=affine)
    pose_entry.attach_same_near_observations(graph,context,rekeyed.programme,
                                             attachment_initial)
    attach_reviewed_main_motion_factors(graph,data,rekeyed.programme,
        attachment_initial,context['motion_rows'],
        reviewed_metadata=context['motion_metadata'])
    expected=int(phase['graph_factors_with_source_calibration'])
    if len(graph._pending)-factor_start!=expected:
        raise ValueError(f'v14 source-pose factor count '
                         f'{len(graph._pending)-factor_start} != {expected}')
    return RebuiltRig('main_long_side',graph,manager,resources,
        {**rekeyed.states,F2_KEY:np.zeros(11)},
        {**rekeyed.prior,F2_KEY:np.zeros(11)},
        rekeyed.programme,rekeyed.references,counts,expected,0,
        rekeyed.key_mapping)


def rebuild_low_complete(context, phase, programme, references, states, prior,
                         q_saved, args, *, resources=None) -> RebuiltRig:
    """Rebuild original low factors with all four F2 callbacks scene-active."""
    resources = {} if resources is None else resources
    required = {'data','pairs','clock','valid_pairs','references','ad_rows',
                'f1_inventory','construction_initial'}
    if required-set(context):
        raise ValueError(f'Incomplete original low builder context: {required-set(context)}')
    rekeyed = rekey_rig_pair(programme, references, states, prior, phase,
        key_start=4_000_000_000, reference_shot_start=12_000,
        reserved_keys=SCENE_RING_KEYS)
    factor_start=len(resources['graph']._pending) if 'graph' in resources else 0
    graph, manager, affine, initial, counts, data = _base_native_graph(
        'low_end',context,rekeyed,context['data'],args,resources,
        q_saved=q_saved,release_count=4)
    old_build_initial = {rekeyed.key_mapping.get(int(k),int(k)):v.copy()
                         for k,v in context['construction_initial'].items()}
    if set(old_build_initial) != set(rekeyed.states):
        raise ValueError('Low source attachment initializer keyset changed')
    old_build_initial[F2_KEY] = np.zeros(11)
    from historic_development_ads import attach_development_ad_factors
    attach_development_ad_factors(graph,data,rekeyed.programme,
        old_build_initial,context['ad_rows'])
    refs = rekey_reference_calibration(context['references'],rekeyed,initial)
    attach_reference_factors(graph,data,refs,adaptive_curves=True)
    from continuous_low_end_f1_correlated_optin import attach_correlated_f1_factors
    attach_correlated_f1_factors(graph,data,refs,context['f1_inventory'])
    add_time_principal_priors(graph,rekeyed.metadata['time_principal'],
        amplitude_sigma_px=20.,acceleration_sigma_px_s15=20.,
        affine_priors=affine)
    expected=int(phase['graph_factors_with_source_calibration'])
    if len(graph._pending)-factor_start!=expected:
        raise ValueError(f'Low native-aware factor count '
                         f'{len(graph._pending)-factor_start} != {expected}')
    return RebuiltRig('low_end',graph,manager,resources,
        {**rekeyed.states,F2_KEY:np.zeros(11)},
        {**rekeyed.prior,F2_KEY:np.zeros(11)},
        rekeyed.programme,rekeyed.references,counts,expected,4,
        rekeyed.key_mapping)


def concatenate_native_built_rigs(*rigs: RebuiltRig):
    """One GTSAM graph with already-correct internal keys, no factor rekeying.

    Both native managers and both graph callback owners must stay alive while
    this graph is evaluated. Native fallback/LM trial routing is not yet wired;
    do not optimize this graph.
    """
    if not rigs or len({rig.camera_id for rig in rigs})!=len(rigs):
        raise ValueError('Composite needs distinct physical camera rigs')
    states={}
    graph=gtsam.NonlinearFactorGraph()
    for rig in rigs:
        for key,value in rig.states.items():
            if key in states and (key!=F2_KEY or
                                  not np.array_equal(states[key],value)):
                raise ValueError(f'Composite state key collision: {key}')
            states[key]=value
        for factor in rig.graph._pending:
            graph.add(factor)
    if F2_KEY not in states:
        raise ValueError('Shared F2 scene node is missing')
    return graph,states


def append_main_f2_source_references(rebuilt: RebuiltRig, candidate_scene: dict,
                                     q_saved, released_f2: dict):
    """Attach 14 actual-image F2 references and their same-image curves.

    Original 11 reference factors remain unchanged. These 14 are separate
    references (28 new frame/C-control nodes), with source-only F2 observations
    and source-backed companion curves at each actual video PTS.
    """
    if rebuilt.camera_id!='main_long_side' or F2_KEY not in rebuilt.states:
        raise ValueError('Main F2 source refs require a rebuilt shared-scene graph')
    from collections import Counter
    import json
    from continuous_brown_adapter import adapt_factor
    from continuous_reference_calibration import seed_reference_initial
    from adaptive_curve_parallel import AdaptiveCurveJob
    from probe_shared_f2_nested import main_sources,add_main_reference_rig
    from robust_observations import pointwise_huber
    from secondary_nearboard_adaptive_curve import (
        AdaptiveProjectedPolyline,adaptive_curve_callback)
    feature=next(f for f in candidate_scene['ice_features'] if f['id']=='F2')
    ring=next(r for r in feature['rings'] if r['role']=='outer')
    center=np.asarray(ring['center_xyz'][:2],np.float64)
    radius=float(ring['radius'])
    sources,companions,frames=main_sources(candidate_scene,center,radius)
    camera_states={k:v.copy() for k,v in rebuilt.states.items() if k!=F2_KEY}
    model,seed_rows=add_main_reference_rig(rebuilt.programme,camera_states,
                                            sources,frames)
    added=set(camera_states)-set(rebuilt.states)
    if len(seed_rows)!=14 or len(added)!=28:
        raise ValueError('Main 14 F2 sources did not allocate 28 fresh graph nodes')
    rebuilt.states.update({key:camera_states[key] for key in added})
    # The 28 fresh reference nodes have no historical camera prior. Their
    # checkpoint initializer payload is the exact seed used for construction;
    # the factor graph below separately defines each explicit regularizer.
    rebuilt.prior.update({key:camera_states[key].copy() for key in added})
    rebuilt.new_source_reference_model=model
    rebuilt.new_source_reference_rows=seed_rows
    before=len(rebuilt.graph._pending)
    source_factors=0
    source_factor_indices=[]
    companion_factors=0
    uv_total=0
    for row in seed_rows:
        pts=int(row['source_pts'])
        frame_key=int(row['frame_key'])
        control_key=int(row['control_key'])
        for key,live in ((frame_key,(3,4,5,6)),
                         (control_key,(0,1,2))):
            select=np.eye(11)[[i for i in range(11) if i not in live]]
            rebuilt.graph.add_observation((key,),lambda q,s=select:s@q[0],
                jac_fn=lambda q,s=select:[s],residual_dim=len(select),sigma=1.)
        select=np.eye(11)[:3]
        rebuilt.graph.add_observation((control_key,),
            lambda q,s=select:s@q[0],jac_fn=lambda q,s=select:[s],
            residual_dim=3,sigma=np.full(3,.25/100.))
        ring_factor=NestedF2Factor(model,sources[pts],
            set(rebuilt.states)-{F2_KEY},{},curve_mode='adaptive')
        rebuilt.graph.add_observation(ring_factor.keys,ring_factor.residual,
            jac_fn=ring_factor.jacobian,**ring_factor.options)
        source_factor_indices.append(len(rebuilt.graph._pending)-1)
        if not hasattr(rebuilt.graph, '_adaptive_curve_jobs'):
            rebuilt.graph._adaptive_curve_jobs=[]
        if not hasattr(rebuilt.graph, '_shared_f2_dynamic_jobs'):
            rebuilt.graph._shared_f2_dynamic_jobs=[]
        job_index=len(rebuilt.graph._adaptive_curve_jobs)
        rebuilt.graph._adaptive_curve_jobs.append(
            ring_factor.adaptive_job(np.zeros(11)))
        ring_factor._prepared_scene_token=np.zeros(11).tobytes()
        rebuilt.graph._shared_f2_dynamic_jobs.append((job_index,ring_factor))
        source_factors+=1
        uv_total+=len(sources[pts].uv)
        occurrences=Counter(json.dumps(item['target'],sort_keys=True)
                            for item,_ in companions[pts])
        for item,world in companions[pts]:
            uv=np.asarray(item['pixels_uv'],np.float64)
            sigma=float(item['sigma_px'])
            target=item['target']
            measurements=[dict(x_interval_px=(float(x),float(x)),
                               y=float(y)) for x,y in uv]
            curve=AdaptiveProjectedPolyline(
                world,measurements,tolerance_px=.01,
                closed=target['kind']=='circle_outline',
                factor_family='original_reference_curve',
                no_central_reason='reference_curve_no_central_segment')
            raw,jac,prime=adaptive_curve_callback(curve,
                model.frames[pts].template25,(pts,),with_prime=True)
            raw,jac=pointwise_huber(raw,jac,group_size=2,delta=2*sigma)
            factor_sigma=sigma*np.sqrt(len(uv)*occurrences[
                json.dumps(target,sort_keys=True)])
            adapted=adapt_factor((pts,),raw,jac,
                dict(residual_dim=2*len(uv),sigma=factor_sigma),
                set(rebuilt.states),{},model)
            if adapted is None:
                raise ValueError('A same-image main source curve became constant')
            keys,residual,derivative,options=adapted
            rebuilt.graph.add_observation(keys,residual,jac_fn=derivative,
                                           **options)
            rebuilt.graph._adaptive_curve_jobs.append(AdaptiveCurveJob(
                curve,np.asarray(model.frames[pts].template25,np.float64).copy(),
                model,pts,(pts,),prime))
            companion_factors+=1
            uv_total+=len(uv)
    # Original scene gauge is zero in released coordinates; at candidate
    # delta zero it retains the saved q offset, never silently recentered.
    released_outer=float(released_f2['outer_radius'])
    released_middle=float(released_f2['middle_radius'])
    target,sigma=scene_prior_in_candidate_coordinates(q_saved,
        np.r_[[.01,.01,.003],np.full(8,1e-9)])
    rebuilt.graph.add_prior(F2_KEY,target,sigma)
    def guard(q):
        outer=released_outer+100.*(np.asarray(q[0])[2]+q_saved[2])
        return np.array([max(0.,released_middle+.05-outer)])
    def derivative(q):
        jac=np.zeros((1,11))
        if released_outer+100.*(np.asarray(q[0])[2]+q_saved[2])<released_middle+.05:
            jac[0,2]=-100.
        return [jac]
    rebuilt.graph.add_observation((F2_KEY,),guard,jac_fn=derivative,
                                   residual_dim=1,sigma=.001)
    expected=14*3+14+companion_factors+2
    if len(rebuilt.graph._pending)-before!=expected:
        raise ValueError('New main F2 factors do not match declared inventory')
    return dict(existing_reference_frames=11,new_reference_frames=14,
        new_reference_graph_nodes=28,source_f2_factors=source_factors,
        source_f2_factor_indices=source_factor_indices,
        companion_curve_factors=companion_factors,
        actual_source_uv=uv_total,scene_prior_and_guard_factors=2,
        adaptive_f2_worker_jobs=source_factors,
        adaptive_companion_worker_jobs=companion_factors,
        c_control_sigma_scene_units=.25,
        c_control_sigma_graph_state_units=.25/100.,
        c_control_actual_factor_target_graph_state=[0.,0.,0.],
        new_reference_prior_packet_role='seed initializers, not historical camera prior targets',
        new_reference_frame_rotation_logzoom_prior_added=False,
        new_reference_inactive_coordinate_factors_target_zero=True,
        added_factor_count=expected,
        total_factor_count=len(rebuilt.graph._pending),
        released_prior_target=target.tolist())

