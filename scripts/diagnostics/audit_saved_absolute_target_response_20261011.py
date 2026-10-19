"""Replay current absolute targets on one saved accepted step; no fitting/J/RGB.

Source values use the original operators and exact existing physical producer.
Crossed camera/world states are sensitivity diagnostics, never candidate models.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
from copy import deepcopy
from collections import Counter, defaultdict
import os, sys, json, pickle, hashlib, time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(Path(__file__).parent)]
import numpy as np
from calibration.solver_runtime import install, configure_native_runtime
install(); configure_native_runtime(stage='readonly_absolute_target_response', native_threads=1)
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from rinkspace.source_sampling import compile_curve_sampling
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.original_review_source import (
    PACKET, PACKET_SHA256, observation_signature, load_original_absolute_evaluators)
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor,SourceWithAdditionalNativeBoardMaterialPoints
from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements
from structured_programme_source_candidate.current_full_source import CurrentPatterns
from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor
from z190_shared_board_pattern_source import _project
from structured_programme_source_candidate.human_stroke_truth import (
    HumanStrokeTruth, UserConfirmedSourceTruth, direct_manual_centre_factors,
    finite_stroke_queries)
from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
from calibration.solver_runtime.original_ice_curve_integration import original_working_sigma
from calibration.solver_runtime.original_board_curve_integration import isolated_anchor_masses
from calibration.solver_runtime.source_loss import (
    weighted_source_cost, SourceLossWeights, loss_weight_arrays)

R = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
RUN = R/'okada_review86_q276_vertical_print_station_one_cycle_20261011'
H = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
CHILD = '8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'
PARENT = '2766454eeb8282632e6bf1226d0305ffcda70b74fbe7d916b2614b8672f8b308'
CONFIG = ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
ADS = {1, 2, 4, 14, 17, 24, 25, 26}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
def stats(value):
    value=np.asarray(value, float)
    return dict(count=len(value), mean=float(value.mean()), median=float(np.median(value)),
                p95=float(np.quantile(value,.95)), maximum=float(value.max())) if len(value) else dict(count=0)
def selected_curve(row):
    c=row.get('curve', {})
    if c.get('kind')=='house_ring' and c.get('side')=='F' and c.get('row')==2: return 'F2:'+c['ring']
    if c.get('kind')=='current_ice_segment' and c.get('feature_id')=='B2': return 'B2'
    n=c.get('advertisement_number', c.get('advertisement_id'))
    if n in ADS: return 'Ad'+str(n)
    return None

def raw_row(row):
    """Diagnostic zero support, while retaining the original typed operator."""
    x=dict(row)
    if 'normal_interval' in x:
        if x.get('original_outer_source_qualification_operator'):
            # The typed qualification insists on a strictly positive width;
            # the smallest IEEE-754 interval implements its zero-width limit
            # without changing its endpoint/scanline operator.
            x['normal_interval']=np.tile([np.nextafter(0.,-1.),np.nextafter(0.,1.)],(len(x['uv']),1))
        else:x.pop('normal_interval')
    if '_source_optical_support' in x: x['_source_optical_support']=None
    x.pop('point_support_bounds_xy_px',None)
    x.pop('_source_point_bounds',None)
    x['source_point_bounds_authorized']=False
    return x

def summarize(values, sigma, quadratic, weights, normalization):
    raw, outside=values
    pix_raw=raw.reshape(-1,2)*np.asarray(sigma)[:,None]
    pix_out=outside.reshape(-1,2)*np.asarray(sigma)[:,None]
    loss=None if weights is None else SourceLossWeights(weights, normalization)
    work=outside if normalization is None else outside*normalization
    return dict(raw_native_px=stats(np.linalg.norm(pix_raw,axis=1)),
                outside_support_native_px=stats(np.linalg.norm(pix_out,axis=1)),
                zero_outside_points=int(np.count_nonzero(np.all(pix_out==0.,axis=1))),
                nonzero_raw_inside_support_points=int(np.count_nonzero((np.linalg.norm(pix_raw,axis=1)>1e-8)&np.all(pix_out==0.,axis=1))),
                raw_pixel_vectors=pix_raw, outside_support_pixel_vectors=pix_out,
                actual_weighted_source_cost=weighted_source_cost(outside,quadratic,loss),
                active_cost_derivative_scalar_count=int(np.count_nonzero(work)),
                Huber_saturated_scalar_count=int(np.count_nonzero((~quadratic)&(abs(work)>2.))),
                quadratic_scalar_count=int(np.count_nonzero(quadratic)))

def response(evaluate, sigma, quadratic, weights, normalization, dq):
    values={(g,c):evaluate(float(g),float(c)) for g in (0,1) for c in (0,1)}
    summaries={str(g)+str(c):summarize(v,sigma,quadratic,weights,normalization) for (g,c),v in values.items()}
    raw={k:v[0].reshape(-1,2)*sigma[:,None] for k,v in values.items()}
    geo=.5*((raw[1,0]-raw[0,0])+(raw[1,1]-raw[0,1]))
    cam=.5*((raw[0,1]-raw[0,0])+(raw[1,1]-raw[1,0]))
    total=raw[1,1]-raw[0,0]
    denom=np.linalg.norm(geo,axis=1)+np.linalg.norm(cam,axis=1)
    dot=np.einsum('ni,ni->n',geo,cam)
    cancellation=np.divide(np.linalg.norm(total,axis=1),denom,out=np.ones(len(denom)),where=denom>1e-10)
    costs={k:summaries[str(k[0])+str(k[1])]['actual_weighted_source_cost'] for k in values}
    h=1e-3
    # Only two one-dimensional existing-step value differences, no source J.
    derivative={}
    for label,(g,c) in {'geometry':(h,0.),'camera':(0.,h),'coupled':(h,h)}.items():
        plus=summarize(evaluate(g,c),sigma,quadratic,weights,normalization)
        minus=summarize(evaluate(-g,-c),sigma,quadratic,weights,normalization)
        derivative[label]=(plus['actual_weighted_source_cost']-minus['actual_weighted_source_cost'])/(2*h)
    return dict(endpoint_same_original_operator=summaries,
                exact_raw_response_camera_px=cam, exact_raw_response_geometry_px=geo,
                exact_raw_response_total_px=total,
                response_sum_max_error_px=float(abs(cam+geo-total).max()),
                opposite_camera_geometry_points=int(np.count_nonzero(dot<0.)),
                opposite_and_at_least_80percent_cancelled_points=int(np.count_nonzero((dot<0)&(cancellation<.2)&(denom>1e-8))),
                response_cancellation_ratio=stats(cancellation),
                weighted_cost_delta=costs[1,1]-costs[0,0],
                exact_cost_geometry_component=.5*((costs[1,0]-costs[0,0])+(costs[1,1]-costs[0,1])),
                exact_cost_camera_component=.5*((costs[0,1]-costs[0,0])+(costs[1,1]-costs[1,0])),
                one_dimensional_cost_directional_derivative_at_parent=derivative,
                directional_stencil_fraction_of_accepted_update=h,
                crossed_states_not_deliverable=True, projection_response_is_not_global_KKT_gradient_proof=True)

def fixed_material_projection_response(world, camera, at):
    """Actual producer stations/roles stay fixed; no UV reassociation here."""
    worlds={g:np.asarray(world(float(g)),float)for g in (0,1)}
    assert worlds[0].shape==worlds[1].shape
    values={(g,c):np.asarray(_project(camera(at,float(c)),worlds[g],False)[0],float)
            for g in (0,1)for c in (0,1)}
    geo=.5*((values[1,0]-values[0,0])+(values[1,1]-values[0,1]))
    cam=.5*((values[0,1]-values[0,0])+(values[1,1]-values[1,0]))
    total=values[1,1]-values[0,0]
    dot=np.einsum('ni,ni->n',geo,cam);denom=np.linalg.norm(geo,axis=1)+np.linalg.norm(cam,axis=1)
    ratio=np.divide(np.linalg.norm(total,axis=1),denom,out=np.ones(len(denom)),where=denom>1e-10)
    return dict(original_physical_station_or_named_role_count=len(worlds[0]),
        no_observed_UV_correspondence_invented=True,no_nearest_curve_reassociation=True,
        exact_additive_camera_geometry_error_px=float(abs(cam+geo-total).max()),
        camera_displacement_magnitude_px=stats(np.linalg.norm(cam,axis=1)),
        geometry_displacement_magnitude_px=stats(np.linalg.norm(geo,axis=1)),
        total_displacement_magnitude_px=stats(np.linalg.norm(total,axis=1)),
        camera_mean_displacement_xy_px=cam.mean(axis=0),geometry_mean_displacement_xy_px=geo.mean(axis=0),
        total_mean_displacement_xy_px=total.mean(axis=0),
        opposite_camera_geometry_stations=int(np.count_nonzero(dot<0.)),
        opposite_and_at_least_80percent_cancelled_stations=int(np.count_nonzero((dot<0)&(ratio<.2)&(denom>1e-8))),
        cancellation_ratio=stats(ratio))

def main():
    started=time.perf_counter(); options=read(CONFIG)['source_options']; manifest=read(RUN/'input_manifest.json')
    directory=RUN/'accepted_states'/CHILD; identity=read(directory/'state_identity.json')
    path=directory/'complete_state_UNACCEPTED.npz'; assert sha(path)==identity['state_npz_sha256']
    keys=('q','scales','quadratic','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
          'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
          'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
          'aux__current_complete_parameter_layout_json','aux__shared_pattern_start')
    with np.load(path,allow_pickle=False) as z: small={k:z[k].copy() for k in keys}
    direction=next((RUN/'actual_complete_source_directions').glob(PARENT+'_*.npz'))
    with np.load(direction,allow_pickle=False) as z: q0=z['q'].copy(); step=z['step'].copy()
    q1=small['q']; alpha=read(RUN/'solver/iterations.json')[0]['global_schur_trial_records'][-1]['whole_source_step_fraction']
    difference=np.flatnonzero(q1!=q0+alpha*step); assert difference.tolist()==[11,12],difference
    assert hashlib.sha256(q0.tobytes()).hexdigest()==PARENT
    assert hashlib.sha256(q1.tobytes()).hexdigest()==CHILD
    provider,world1,proof=current_complete_source_geometry(path,options['--world-init'])
    with np.load(identity['camera_path'],allow_pickle=False) as z: cameras={k:z[k].copy() for k in z.files}
    state,camera_error=state_camera_adapter(small,cameras,provider)
    eq0=state.evaluation_q(q0); eq1=state.evaluation_q(q1); de=eq1-eq0
    world_indices=state.geometry_parameter_indices; world0=eq0[world_indices]; world1=eq1[world_indices]
    def camera(at,fraction):
        block=state.bypts[int(at)]
        return block.bundle.camera(eq0+fraction*de,int(at))
    scope=manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope:state.current_user_physical_course_scope_source=scope['path']
    board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(eq1))
    ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(eq1))
    scene=read(directory/'scene_UNACCEPTED.json')
    pattern_base=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,
        facet_authority=scene['shared_board_patterns']['facet_authority'])
    pattern_original=SharedBoardPatternSourceFactor(pattern_base)
    owner=SimpleNamespace(start=int(small['aux__shared_pattern_start']),state=state,geometry=pattern_base,
        factor=pattern_original,external_factor=pattern_original,_original_course_pattern_geometry=pattern_base,
        _original_course_pattern_factor=pattern_original,_original_course_external_pattern_factor=pattern_original,
        course_world_geometry=provider,fit=[],held=[])
    owner.bind_current_geometry=MethodType(CurrentPatterns.bind_current_geometry,owner);state.patterns=owner
    prefix_count=len(eq1)-4
    material=NativeBoardMaterialPointFactor(state,options['--additional-board-material-point-source'],prefix_count)
    empty=SimpleNamespace(count=prefix_count,scales=np.empty(0),quadratic=np.empty(0,bool),contract={},updater=None,
        blocks=lambda q,jacobian:iter(()),guard=lambda q:provider.validate_parameters(q[world_indices]))
    placement=SourceWithIndependentPatternPlacements(SourceWithAdditionalNativeBoardMaterialPoints(empty,material),state,
        'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/model_constraints/user_advertisement_pairs_20261005.json')
    assert np.array_equal(placement.extend_initial(eq1[:prefix_count],suffix=eq1[prefix_count:]),eq1)
    circle=NativeBoardCircleFactor(state,options['--additional-board-circle-source'],prefix_count-6)
    factors={'native_board':board,'native_ice':ice}
    base=SimpleNamespace(count=len(eq1),scales=np.empty(0),quadratic=np.empty(0,bool),contract={},
                         native_board_factor=board,native_ice_factor=ice)
    integration=SourceWithNativeBaseCurveIntegration(base,options['--native-base-curve-integration-policy'])
    before_families=read(RUN/'latest_initial_J_family_timings.json')
    after_families=read(RUN/'actual_trial_family_cost_pass_0001.json')['families']
    ledger_by_family={x['original_row_ledger']['family']:x for x in after_families}
    output=dict(schema='rinkspace.saved-current-absolute-target-response.v1',script=ref(__file__),
        config=ref(CONFIG),input_manifest=ref(RUN/'input_manifest.json'),direction=ref(direction),
        parent_q=PARENT,child_q=CHILD,accepted_fraction=alpha,pivot_retracted_canonical_columns=difference,
        exact_camera_archive_reproduction_max=camera_error,
        no_full_source_graph_or_J_or_fit=True,no_source_images_decoded_or_written=True,
        source_UV_sigma_supports_roles_unmodified=True,crossed_states_are_sensitivity_only=True,
        native_complete_family_cost_reproduction=[],rows=[],protection_rows=[])
    native_layouts={}
    for name,factor in factors.items():
        family='additional_real_native_ice_curves' if name=='native_ice' else 'additional_real_native_board_edges'
        active=factor._qualified_active_fit_view() if hasattr(factor,'_qualified_active_fit_view') else None
        active=factor if active is None else active
        rows=active.rows; payload=integration._payloads[family]; w,norm=loss_weight_arrays(payload,active.row_count)
        ledger=ledger_by_family[family]['original_row_ledger']; begin=ledger['original_start']
        assert np.array_equal(small['scales'][begin:ledger['original_stop']],active.scales)
        assert np.array_equal(small['quadratic'][begin:ledger['original_stop']],active.quadratic)
        rr0=active._residual(eq0,rows,cameras={at:camera(at,0.) for at in active.blocks})
        rr1=active._residual(eq1,rows,cameras={at:camera(at,1.) for at in active.blocks})
        c0=weighted_source_cost(rr0,active.quadratic,payload); c1=weighted_source_cost(rr1,active.quadratic,payload)
        expected=ledger_by_family[family]['normalized_cost']; assert abs(c1-expected)<2e-8,(family,c1,expected)
        before_saved=next(x['actual_normalized_cost']for x in before_families['families']if x['family']==family)
        assert abs(c0-before_saved)<2e-8,(family,c0,before_saved)
        output['native_complete_family_cost_reproduction'].append(dict(family=family,before=c0,after=c1,saved_before=before_saved,before_error=c0-before_saved,saved_after=expected,error=c1-expected))
        cursor=0; layouts={}
        for row in rows:
            count=2*len(row['uv']); layouts[row['id']]=(cursor,cursor+count);cursor+=count
        native_layouts[name]=layouts
        for role,rr in [('FIT',rows),('HELD',factor.held)]:
            for row in rr:
                physical=selected_curve(row)
                if physical is None:continue
                sigma=np.asarray(row['sigma'],float); count=2*len(sigma)
                if role=='FIT': lo,hi=layouts[row['id']];weights=w[lo:hi];normalization=None if norm is None else norm[lo:hi]
                else: weights=normalization=None
                at=int(row['actual_pts']); original_signature=observation_signature(row)
                def evaluate(g,c,row=row,at=at,factor=factor):
                    qg=eq0+g*de; cam={at:camera(at,c)}
                    raw=(factor._residual(qg,[row],cameras=cam,apply_optical_support=False) if name=='native_board'
                         else factor._residual(qg,[raw_row(row)],cameras=cam))
                    return raw,factor._residual(qg,[row],cameras=cam)
                record=dict(family=family,collection_role=role,id=row['id'],physical_group=physical,
                            actual_pts=at,native_pts=row['pts'],points=len(sigma),sigma=sigma,
                            source_optical_normal_support_authorized=row.get('source_optical_normal_support_authorized',False),
                            support_bounds=row.get('normal_interval',row.get('normal_interval_px')),
                            source_integration_weights=weights,working_normalization=normalization,
                            cost_in_actual_target=role=='FIT')
                record.update(response(evaluate,sigma,np.ones(count,bool),weights,normalization,de))
                record['fixed_physical_material_projection_response']=fixed_material_projection_response(
                    lambda g:factor._world(eq0+g*de,row['curve']),camera,at)
                assert original_signature==observation_signature(row)
                output['rows'].append(record)
    print(json.dumps(dict(stage='native_target_response_complete',rows=len(output['rows']),costs=output['native_complete_family_cost_reproduction'])),flush=True)
    for factor,family in [(material,'additional_real_native_board_material_points'),(circle,'additional_real_native_board_circles')]:
        def projected_residual(q,rows,cams,factor=factor):
            if factor is material:
                with placement.pattern_placement_context(q):return factor._residual(q[:factor.parameter_count],rows,cameras=cams)
            return factor._residual(q[:factor.parameter_count],rows,cameras=cams)
        c0=weighted_source_cost(projected_residual(eq0,factor.rows,{at:camera(at,0.)for at in factor.blocks}),factor.quadratic,None)
        c1=weighted_source_cost(projected_residual(eq1,factor.rows,{at:camera(at,1.)for at in factor.blocks}),factor.quadratic,None)
        expected=ledger_by_family[family]['normalized_cost'];assert abs(c1-expected)<2e-8,(family,c1,expected)
        output['native_complete_family_cost_reproduction'].append(dict(family=family,before=c0,after=c1,saved_after=expected,error=c1-expected))
        for role,rr in [('FIT',factor.rows),('HELD',factor.held)]:
            for row in rr:
                physical=selected_curve(row)
                if physical is None:continue
                at=int(row['actual_pts']);sigma=np.asarray(row['sigma'],float)
                def evaluate(g,c,row=row,at=at):
                    qg=eq0+g*de;cams={at:camera(at,c)}
                    return projected_residual(qg,[raw_row(row)],cams),projected_residual(qg,[row],cams)
                record=dict(family=family,collection_role=role,id=row['id'],physical_group=physical,actual_pts=at,
                    native_pts=row['pts'],points=len(sigma),sigma=sigma,support_bounds=row.get('normal_interval_px',row.get('point_support_bounds_xy_px')),
                    cost_in_actual_target=role=='FIT')
                record.update(response(evaluate,sigma,np.ones(2*len(sigma),bool),None,None,de));output['rows'].append(record)
                def world(g):
                    qg=eq0+g*de
                    if factor is material:
                        with placement.pattern_placement_context(qg):return factor._world(qg[:factor.parameter_count],row)
                    return factor.world(qg[:factor.parameter_count],row['advertisement_number'])
                record['fixed_physical_material_projection_response']=fixed_material_projection_response(world,camera,at)
    assert sha(PACKET)==PACKET_SHA256
    originals=pickle.loads(PACKET.read_bytes())['programme_absolute_rows']; calls,call_proof=load_original_absolute_evaluators()
    corrections={r['original_packet_row_index']:r for r in read(options['--original-ice-pixel-correction-source'])['records']}
    retained={}
    for block in manifest['original_absolute_input_compaction']['blocks']:
        for row in block['row_maps']: retained[block['block'],row['group_order'],row['row_order']]=row['kept_original_point_indices']
    def current_original(index):
        item=originals[index]; original=item['source_row']; row=deepcopy(original)
        change=corrections.get(index)
        if change:
            points=change['point_indices']; row['uv']=np.asarray(row['uv'],float).copy(); row['uv'][points]=change['corrected_uv']
            if change.get('remove_original_normal_interval'):row.pop('normal_interval',None)
            elif change.get('corrected_normal_interval_px') is not None:
                row['normal_interval']=np.asarray(row['normal_interval'],float).copy();row['normal_interval'][points]=change['corrected_normal_interval_px']
        keep=retained.get((item['block'],item['group_order'],item['row_order']),list(range(len(row['uv']))))
        for key in ('uv','sigma','normal','normal_interval','source_uv'):
            if key in row and len(np.asarray(row[key]))==len(original['uv']):row[key]=np.asarray(row[key])[keep].copy()
        return row,np.asarray(keep,int)
    # Reconstruct the exact installed ICE integration using the same public
    # group compiler, all installed ICE rows, original working sigma and UV.
    integration_records=[];original_ice_slots={}
    packet_ref=manifest['original_ice_curve_integration']['original_packet']
    for source in manifest['original_ice_curve_integration']['source_rows']:
        index=source['original_packet_row_index'];item=originals[index];row,keep=current_original(index)
        assert keep.tolist()==source['original_point_indices']
        assert hashlib.sha256(np.asarray(row['uv'],float).tobytes()).hexdigest()==source['current_active_UV_sha256']
        old=item['source_row'];stored=np.asarray(old['sigma'],float)
        working,prior,_=original_working_sigma(old,stored,source.get('original_declared_scan_group_count') or 0)
        name=f"programme:{item['block']}:{item['group_order']}:{item['row_order']}"
        integration_records.append(dict(id=name,split='fit',actual_pts=int(item['group_actual_pts']),source_video_sha256=manifest['original_ice_curve_integration']['source_video_sha256'],
            physical_curve_id=json.dumps(source['physical_identity'],separators=(',',':')),
            operator_id=json.dumps({k:row.get(k) for k in ('curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side')},sort_keys=True),
            uv=np.asarray(row['uv'],float),stored_sigma_px=stored[keep],working_sigma_px=working[keep],prior_sampling_multiplier=prior[keep],
            point_integration_mass=isolated_anchor_masses(row['uv']),sampling_kind=['measured']*len(keep),source_anchor_identity_authority=packet_ref))
        original_ice_slots[index]=name
    plan=compile_curve_sampling(integration_records,duplicate_anchor_policy=manifest['original_ice_curve_integration']['duplicate_anchor_policy'])
    assert json.dumps(plan['groups'],sort_keys=True,default=plain)==json.dumps(manifest['original_ice_curve_integration']['groups'],sort_keys=True,default=plain)
    original_selected=[]
    for index,item in enumerate(originals):
        old=item['source_row'];ids=provider.source_course.original.pair_panels.identities(old) or []
        want='F2' in str(old['id']) or old['id']=='B2' or any(n in ADS for n,role in ids)
        if not want:continue
        try:
            provider.register_row(old,actual_pts=int(item['group_actual_pts']),allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
            binding=provider._binding(old)[1]
        except Exception as error:
            output['rows'].append(dict(family='original_absolute',original_packet_row_index=index,status='binding_failed',id=old['id'],error=str(error)))
            continue
        row,keep=current_original(index);at=int(item['group_actual_pts']);physical=binding.get('name') or '+'.join('Ad'+str(n) for n in sorted({n for n,role in binding.get('ids',ids)}))
        sigma=np.asarray(row['sigma'],float); count=2*len(sigma)
        if index in original_ice_slots:
            info=plan['rows'][original_ice_slots[index]];weights=np.repeat(info['point_integration_weights'],2);normalization=np.repeat(info['working_normalization_multiplier'],2)
        else:weights=normalization=None
        def evaluate(g,c,row=row,at=at,item=item):
            bound=provider.rebind(row,world0+g*(world1-world0));call=calls[item['block']];cam=camera(at,c)
            return np.asarray(call(cam,[raw_row(bound)],False)),np.asarray(call(cam,[bound],False))
        record=dict(family='original_absolute',original_packet_row_index=index,collection_role=item['collection_role'],
            block=item['block'],id=row['id'],physical_group=physical,physical_binding=binding['contract'].get('physical_identity',binding.get('ids')),
            actual_pts=at,native_pts=int(row['pts']),points=len(sigma),sigma=sigma,
            current_pixel_correction=index in corrections,retained_original_point_indices=keep,
            support_bounds=row.get('normal_interval',row.get('point_support_bounds_xy_px')),
            source_integration_weights=weights,working_normalization=normalization,cost_in_actual_target=item['collection_role']=='FIT')
        try:
            record.update(response(evaluate,sigma,np.zeros(count,bool),weights,normalization,de));record['status']='evaluated'
            record['fixed_physical_material_projection_response']=fixed_material_projection_response(
                lambda g:provider.rebind(row,world0+g*(world1-world0))['world'],camera,at)
        except Exception as error:record.update(status='operator_failed',error=str(error),generic_fallback=False)
        output['rows'].append(record);original_selected.append(index)
    print(json.dumps(dict(stage='original_absolute_response_complete',rows=len(original_selected))),flush=True)
    human=HumanStrokeTruth(state,options['--direct-human-truth-authority'],board,ice,parameter_count=len(eq1))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard']);human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support']);human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    confirmed=UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,parameter_count=len(eq1),
                                     circle_factor=circle,material_factor=material,placement_source=placement)
    for family,truth,records in [('native',human,human.full_stroke_records)]:
        for index,record in enumerate(records):
            row=record['row'];physical=selected_curve(row)
            if physical is None:continue
            factor=record['factor'];at=int(row['actual_pts']);values={}
            for g,c in [(0,0),(1,0),(0,1),(1,1)]:
                query=finite_stroke_queries(camera(at,c),factor._world(eq0+g*de,row['curve']),record['uv'],record['normals'],record['bounds'],tangent_bounds=record.get('tangent_bounds'),
                    association='nearest_finite_segment'if row['curve']['kind']in('board_top','ice_board_contact')else'normal_scanline')
                values[str(g)+str(c)]=dict(maximum_violation_px=float(abs(query['residual']).max()),
                    signed_normal_px=query['signed_normal_px'],coverage_tangent_px=query['coverage_tangent_px'],
                    outside_support_residual_px=query['residual'],support_branch_signature=query['source_support_branch_signature'])
            output['protection_rows'].append(dict(family=family,group=index,id=row['id'],physical_group=physical,actual_pts=at,
                original_normal_bounds=record['bounds'],original_tangent_bounds=record.get('tangent_bounds'),points=len(record['uv']),endpoints=values,
                hard_support_operator_is_not_centre_objective=True,source_band_not_changed=True))
    for index,item in enumerate(confirmed.records):
        row=item['row'];physical=selected_curve(row)
        if physical is None:continue
        factor=item['factor'];at=int(row['actual_pts']);values={}
        for g,c in [(0,0),(1,0),(0,1),(1,1)]:
            qg=eq0+g*de;cams={a:camera(a,c)for a in factor.blocks}
            value,lo,hi,detail=confirmed._one_record_carrier(qg,item,cams,factor_cameras={id(factor):cams})
            violation=value-np.clip(value,lo,hi)
            values[str(g)+str(c)]=dict(maximum_violation_px=float(abs(violation).max()),carrier_native_px=value,
                lower_native_px=lo,upper_native_px=hi,outside_support_residual_px=violation,query_detail=detail)
        output['protection_rows'].append(dict(family='confirmed',group=index,id=row['id'],physical_group=physical,actual_pts=at,
            points=len(row['uv']),endpoints=values,hard_support_operator_is_not_centre_objective=True,source_band_not_changed=True))
    centres=direct_manual_centre_factors(state,human,options['--additional-board-source'],options['--additional-ice-source'],len(eq1))
    centre_ledgers=[x for x in after_families if x['original_row_ledger']['family']=='direct_manual_sparse_correlated_centre_secondary']
    for factor,stored in zip(centres,centre_ledgers):
        family='direct_manual_sparse_correlated_centre_secondary';mean_scale=factor._manual_centre_mean_scale
        c0=weighted_source_cost(factor._residual(eq0,factor.rows,cameras={at:camera(at,0.)for at in factor.blocks})/mean_scale,factor.quadratic,None)
        c1=weighted_source_cost(factor._residual(eq1,factor.rows,cameras={at:camera(at,1.)for at in factor.blocks})/mean_scale,factor.quadratic,None)
        assert abs(c1-stored['normalized_cost'])<2e-8,(family,c1,stored['normalized_cost'])
        output['native_complete_family_cost_reproduction'].append(dict(family=family,before=c0,after=c1,saved_after=stored['normalized_cost'],error=c1-stored['normalized_cost']))
        cursor=0
        for row in factor.rows:
            n=2*len(row['sigma']);scale=mean_scale[cursor:cursor+n];cursor+=n
            physical=selected_curve(row)
            if physical is None:continue
            at=int(row['actual_pts']);sigma=np.asarray(row['sigma'],float)
            def evaluate(g,c,row=row,at=at,scale=scale):
                result=factor._residual(eq0+g*de,[row],cameras={at:camera(at,c)})/scale
                return result,result
            record=dict(family=family,collection_role='FIT',id=row['id'],physical_group=physical,actual_pts=at,native_pts=row['pts'],
                points=len(sigma),sigma=sigma,correlated_trace_mean_scale=scale,cost_in_actual_target=True)
            record.update(response(evaluate,sigma*scale[::2],np.ones(n,bool),None,None,de));output['rows'].append(record)
            record['fixed_physical_material_projection_response']=fixed_material_projection_response(
                lambda g:factor._world(eq0+g*de,row['curve']),camera,at)
    output['original_absolute_evaluators']=call_proof
    output['original_ice_integration_exact_group_reproduction']=True
    output['status_counts']=dict(Counter(r.get('status','evaluated') for r in output['rows']))
    output['coverage_by_physical_group_family_role']=[]
    grouping=defaultdict(list)
    for row in output['rows']:grouping[row['physical_group'],row['family'],row['collection_role']].append(row)
    for (physical,family,role),rows in sorted(grouping.items()):
        result=dict(physical_group=physical,family=family,collection_role=role,operator_rows=len(rows),
            actual_exposures=sorted({r['actual_pts']for r in rows}),cost_is_actual_FIT_objective=role=='FIT')
        success=[r for r in rows if r.get('status','evaluated')=='evaluated']
        result['summed_costs']={ep:sum(r['endpoint_same_original_operator'][ep]['actual_weighted_source_cost']for r in success)for ep in ('00','10','01','11')}
        result['nonzero_raw_and_exact_zero_outside_points_at_child']=sum(r['endpoint_same_original_operator']['11']['nonzero_raw_inside_support_points']for r in success)
        result['exact_zero_cost_operator_rows_at_child']=[r.get('original_packet_row_index',r['id'])for r in success if r['endpoint_same_original_operator']['11']['actual_weighted_source_cost']==0.]
        result['one_dimensional_directional_derivative_at_parent']={label:sum(r['one_dimensional_cost_directional_derivative_at_parent'][label]for r in success)for label in ('geometry','camera','coupled')}
        output['coverage_by_physical_group_family_role'].append(result)
    output['coverage_limits']=dict(original_compact_packet_selected_registered_F2_B2_and_Ads=sorted(ADS),
        current_native_board_ice_material_circle_and_manual_centre_rows=True,
        whole_motion_objective_or_86_constraint_dual_not_recomputed=True,
        fixed_physical_projection_uses_existing_producer_samples_not_added_source_queries=True,
        other_global_tail_families_not_decomposed_here=['external_main_to_low','shared_board_patterns','wide_anchor_source'],
        hard_support_excess_is_not_current_allowable_group_cap_excess=True,
        HELD_costs_are_readonly_diagnostics_not_in_objective=True,
        directional_derivatives_of_saved_update_not_all_parameter_gradient_or_cause_of_update=True)
    output['seconds']=time.perf_counter()-started
    destination=H/'current_absolute_F2_B2_related_ads_saved_step_q276_q8a15_causal_response_20261011.json'
    if destination.exists():
        old=read(destination)
        if old.get('schema')!=output['schema']or old.get('parent_q')!=PARENT or old.get('child_q')!=CHILD:raise FileExistsError(destination)
        output['replaces_prior_same_q_diagnostic_sha256']=sha(destination)
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False,default=plain)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=ref(destination),rows=len(output['rows']),protection_rows=len(output['protection_rows']),seconds=output['seconds'],status_counts=output['status_counts'])),flush=True)

if __name__=='__main__':main()
