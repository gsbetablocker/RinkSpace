"""One fixed-world, same-source RF4/free-C camera comparison; no production state.

Actual native/original value operators, current source qualification and public
sampling compilers are reused. Only 231678464 is evaluated. All world curves are
cached before either solve; the original complete motion graph is never loaded.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
from copy import deepcopy
import hashlib, json, os, sys, time, pickle

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from calibration.solver_runtime import install, configure_native_runtime
install(); configure_native_runtime(stage='single_frame_condition_comparison', native_threads=1)
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from rinkspace.source_sampling import compile_curve_sampling
from calibration.solver_runtime.source_loss import SourceLossWeights, loss_weight_arrays, weighted_source_cost
from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
from calibration.solver_runtime.original_board_curve_integration import (
    OriginalDetectorSupport, declared_working_sigma, isolated_anchor_masses)
from calibration.solver_runtime.original_ice_curve_integration import original_working_sigma
from structured_programme_source_candidate.paired_programme_chart import (
    CompleteProgrammeParameterLayout, PairedProgrammeCoordinateChart)
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.original_review_source import (
    PACKET, PACKET_SHA256, load_original_absolute_evaluators)
from z190_physical_camera import SharedCalibration, ViewState
from projection import project_central_valid

QID = '917c0fa45f80af8e16c74d0f2d56486fdb500b591357add3dc2b240b3de64cdf'
PTS = 231678464
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
DIAG = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
CONFIG = ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json'
OUTPUT = DIAG/'q32_231678464_fixed_world_RF4_freeC7_condition_comparison_20261012.json'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)
def stats(v):
    v=np.asarray(v,float)
    return dict(count=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.quantile(v,.95)), maximum=float(v.max()))
def raw_row(row):
    out=dict(row)
    out.pop('normal_interval',None)
    out['_source_optical_support']=None
    out.pop('point_support_bounds_xy_px',None)
    return out

def main():
    started=time.perf_counter(); opts=read(CONFIG)['source_options']; manifest=read(RUN/'input_manifest.json')
    identity=read(RUN/'accepted_states'/QID/'state_identity.json')
    state_path=RUN/'accepted_states'/QID/'complete_state_UNACCEPTED.npz'
    assert sha(state_path)==identity['state_npz_sha256']
    fields=['q','quadratic','scales','aux__canonical_pts','aux__block_start','aux__block_lengths','aux__block_names',
            'aux__installation_xyz_slots','aux__all_installation_optics_installation_order',
            'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices',
            'aux__current_complete_parameter_layout_json']
    with np.load(state_path,allow_pickle=False) as z: saved={k:z[k].copy() for k in fields}
    assert hashlib.sha256(saved['q'].tobytes()).hexdigest()==QID
    assert sha(identity['camera_path'])==identity['camera_sha256']
    with np.load(identity['camera_path'],allow_pickle=False) as z:
        cameras={k:z[k].copy() for k in ['pts','camera25','exposure_representative_pts','installation_id','view_state4','effective_pp_offset_px']}
        meta=json.loads(str(z['metadata_json'].item()))
    ix=int(np.flatnonzero(cameras['pts']==PTS)[0]); assert int(cameras['exposure_representative_pts'][ix])==PTS
    c0=cameras['camera25'][ix]; v0=cameras['view_state4'][ix]; pp=cameras['effective_pp_offset_px'][ix]
    calibration=SharedCalibration.from_manifest(meta['calibrations'][str(cameras['installation_id'][ix])])
    assert np.array_equal(calibration.camera25(ViewState.from_state4(v0,effective_pp_offset_px=pp)),c0)
    provider,world,world_proof=current_complete_source_geometry(state_path,opts['--world-init'])
    decl=json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
    layout=CompleteProgrammeParameterLayout(decl['original_complete_parameter_count'],decl['additional_world_parameter_count'])
    assert layout.manifest()==decl
    chart=PairedProgrammeCoordinateChart(len(world)-66); q=layout.evaluation_q(saved['q'])
    assert np.array_equal(q[np.asarray(chart.world_indices)],world)
    aliases=dict(zip(map(int,cameras['pts']),map(int,cameras['exposure_representative_pts'])))
    actual_index={}
    for i,p in enumerate(cameras['exposure_representative_pts']): actual_index.setdefault(int(p),i)
    order=saved['aux__all_installation_optics_installation_order'].tolist(); n=len(order)
    optics=saved['aux__current_all_installation_optics_parameter_indices'].reshape(n,3)
    mounts=saved['aux__current_shared_mount_parameter_indices'].reshape(n,3)
    # These are slot/alias dictionaries only: no other camera is reconstructed.
    bypts={}; cursor=0
    for name,start,length in zip(saved['aux__block_names'],saved['aux__block_start'],saved['aux__block_lengths']):
        pts=saved['aux__canonical_pts'][cursor:cursor+int(length)//4]
        ids={str(cameras['installation_id'][actual_index[int(p)]]) for p in pts}; assert len(ids)==1
        installation=order.index(ids.pop())
        block=SimpleNamespace(name=str(name),start=int(start),xyz_slots=saved['aux__installation_xyz_slots'][installation],
            optics_chart=SimpleNamespace(indices=optics[installation],mount_indices=mounts[installation]),
            bundle=SimpleNamespace(index={int(p):i for i,p in enumerate(pts)}))
        bypts.update({int(p):block for p in pts}); cursor+=len(pts)
    state=SimpleNamespace(geometry=provider,geometry_parameter_indices=np.asarray(chart.world_indices),
        structured_programme_coordinate_chart=chart,original=SimpleNamespace(aliases=aliases),bypts=bypts,
        current_user_physical_course_scope_source=opts.get('--physical-course-scope-source'))
    ice=NativeIceCurveFactor(state,opts['--additional-ice-source'],len(q))
    board=NativeBoardEdgeFactor(state,opts['--additional-board-source'],len(q))
    integration=SourceWithNativeBaseCurveIntegration(SimpleNamespace(native_ice_factor=ice,native_board_factor=board,
        contract={},count=len(q),scales=np.empty(0),quadratic=np.empty(0,bool)),opts['--native-base-curve-integration-policy'])
    families=read(RUN/'actual_trial_family_cost_pass_0042.json'); assert families['complete_q_sha256']==QID
    ledger={r['original_row_ledger']['family']:r['original_row_ledger'] for r in families['families']}
    records=[]; frozen_world={}
    for family,factor in [('additional_real_native_ice_curves',ice),('additional_real_native_board_edges',board)]:
        active=factor._qualified_active_fit_view() if hasattr(factor,'_qualified_active_fit_view') else None
        active=factor if active is None else active
        start=ledger[family]['original_start']; stop=ledger[family]['original_stop']
        assert np.array_equal(saved['scales'][start:stop],active.scales)
        assert np.array_equal(saved['quadratic'][start:stop],active.quadratic)
        weights,norm=loss_weight_arrays(integration._payloads[family],active.row_count)
        cursor=0
        for row in active.rows:
            size=2*len(row['uv']); lo,hi=cursor,cursor+size; cursor=hi
            if row['actual_pts']!=PTS: continue
            key=json.dumps(row['curve'],sort_keys=True)
            frozen_world[(id(active),key)]=active._world(q,row['curve']).copy()
            records.append(dict(family=family,row=row,factor=active,weights=weights[lo:hi].copy(),
                normalization=np.ones(size) if norm is None else norm[lo:hi].copy(),quadratic=saved['quadratic'][start+lo:start+hi].copy(),
                saved_global_scalar_range=[start+lo,start+hi],
                source_id=row['id'],physical_group=row.get('physical_curve_id',row['curve']['kind']),
                original_active_point_indices=(next(m['retained_original_point_indices'] for m in active._qualified_original_index_map if m['id']==row['id'])
                    if hasattr(active,'_qualified_original_index_map') else list(range(len(row['uv']))))))
        assert cursor==active.row_count
        def cached(self,value,curve=None):
            if curve is None: raise ValueError('Only explicit selected source curves are cached')
            return frozen_world[(id(self),json.dumps(curve,sort_keys=True))]
        active._world=MethodType(cached,active)
    assert sha(PACKET)==PACKET_SHA256
    originals=pickle.loads(PACKET.read_bytes())['programme_absolute_rows']; calls,call_proof=load_original_absolute_evaluators()
    corrections={r['original_packet_row_index']:r for r in read(opts['--original-ice-pixel-correction-source'])['records']}
    keep={}
    for block in manifest['original_absolute_input_compaction']['blocks']:
        for r in block['row_maps']: keep[block['block'],r['group_order'],r['row_order']]=r['kept_original_point_indices']
    # Saved production masks are per scalar: C0 right extent is quadratic,
    # although the other original rows in this exposure use Huber2.
    compact=next(b for b in manifest['original_absolute_input_compaction']['blocks'] if b['block']=='main9')
    main_ledger=ledger['programme:main9']
    cursor=int(main_ledger['original_start'])+2*int(compact['motion_point_count'])
    counts=dict(zip(map(str,saved['aux__block_names']),map(int,saved['aux__block_lengths']//4)))
    end=int(main_ledger['original_stop'])-counts['main9']; original_masks={};original_ranges={}
    assert int(compact['compact_start'])==int(main_ledger['original_start'])
    for index,item in enumerate(originals):
        if item['block']!='main9' or item['collection_role']!='FIT': continue
        old=item['source_row']; retained=keep.get((item['block'],item['group_order'],item['row_order']),list(range(len(old['uv']))))
        size=2*len(retained);stop=cursor+size
        assert stop<=end and np.array_equal(saved['scales'][cursor:stop],np.repeat(np.asarray(old['sigma'])[retained],2))
        if index in range(10115,10120):
            original_masks[index]=saved['quadratic'][cursor:stop].copy();original_ranges[index]=[cursor,stop]
        cursor=stop
    assert cursor==end
    support=OriginalDetectorSupport(manifest['original_board_curve_integration']['source_video_sha256'],
                                    manifest['original_board_curve_integration']['source_station_gap_policy'])
    plans={'board':[],'ice':[]}; pending=[]
    for index in range(10115,10120):
        item=originals[index]; old=item['source_row']; assert item['collection_role']=='FIT' and item['group_actual_pts']==PTS
        row=deepcopy(old); change=corrections.get(index)
        if change:
            row['uv']=np.asarray(row['uv'],float).copy(); row['uv'][change['point_indices']]=change['corrected_uv']
            if change.get('remove_original_normal_interval'): row.pop('normal_interval',None)
            elif change.get('corrected_normal_interval_px') is not None:
                row['normal_interval']=np.asarray(row['normal_interval'],float).copy()
                row['normal_interval'][change['point_indices']]=change['corrected_normal_interval_px']
        retained=np.asarray(keep.get((item['block'],item['group_order'],item['row_order']),list(range(len(old['uv'])))),int)
        for k in ('uv','sigma','normal','normal_interval','source_uv'):
            if k in row and len(np.asarray(row[k]))==len(old['uv']):row[k]=np.asarray(row[k])[retained].copy()
        category='board' if index<10117 else 'ice'; integration_key='original_'+category+'_curve_integration'
        entries=manifest[integration_key]['source_rows']
        entry=next(r for r in entries if (r.get('original_packet_row_index')==index if category=='ice'
                   else r['group_order']==item['group_order'] and r['row_order']==item['row_order'] and r['actual_pts']==PTS))
        assert retained.tolist()==entry['original_point_indices']
        assert hashlib.sha256(np.asarray(row['uv'],float).tobytes()).hexdigest()==entry['current_active_UV_sha256']
        stored=np.asarray(old['sigma'],float)
        if category=='ice':
            working,prior,_=original_working_sigma(old,stored,entry.get('original_declared_scan_group_count') or 0)
            mass=isolated_anchor_masses(row['uv']); physical=json.dumps(entry['physical_identity'],separators=(',',':'))
            operator=json.dumps({k:row.get(k) for k in ('curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side')},sort_keys=True)
        else:
            working,prior,_=declared_working_sigma(old,stored); physical=entry['physical_curve_id']
            operator=str(row.get('curve_observation_kind','finite_curve_vector_distance'))
            mass=(np.ones(len(retained)) if entry['support_basis']=='isolated_measured_corrected_RGB_anchors'
                else support.masses(old,row['uv'],PTS,original_point_indices=retained)[0])
            assert np.isclose(mass.sum(),entry['source_support_mass'],rtol=1e-12,atol=1e-12)
        name=f'programme:{item["block"]}:{item["group_order"]}:{item["row_order"]}'
        plans[category].append(dict(id=name,split='fit',actual_pts=PTS,source_video_sha256=manifest[integration_key]['source_video_sha256'],
            physical_curve_id=physical,operator_id=operator,uv=np.asarray(row['uv']),stored_sigma_px=stored[retained],
            working_sigma_px=working[retained],prior_sampling_multiplier=prior[retained],point_integration_mass=mass,
            sampling_kind=['measured']*len(retained),source_anchor_identity_authority=manifest[integration_key]['duplicate_anchor_policy']['source_anchor_identity_authority']))
        provider.register_row(old,actual_pts=PTS); bound=provider.rebind(row,world)
        pending.append(dict(family='original_absolute',row=bound,call=calls[item['block']],source_id=f'{index}:{old["id"]}',
            physical_group=physical,original_packet_row_index=index,original_active_point_indices=retained.tolist(),category=category,plan_name=name,
            quadratic=original_masks[index],saved_global_scalar_range=original_ranges[index]))
    for category,rows in plans.items():
        contract=manifest['original_'+category+'_curve_integration']
        plan=compile_curve_sampling(rows,duplicate_anchor_policy=contract['duplicate_anchor_policy'])
        for group in plan['groups']:
            match=next(g for g in contract['groups'] if g['actual_pts']==PTS and g['physical_curve_id']==group['physical_curve_id'])
            assert json.dumps(group,sort_keys=True,default=plain)==json.dumps(match,sort_keys=True,default=plain)
        for record in pending:
            if record['category']!=category: continue
            info=plan['rows'][record['plan_name']];size=2*len(record['row']['uv'])
            record.update(weights=np.repeat(info['point_integration_weights'],2),
                normalization=np.repeat(info['working_normalization_multiplier'],2))
            records.append(record)
    assert len(records)==15 and sum(len(r['row']['uv']) for r in records)==355
    weights=np.concatenate([r['weights'] for r in records]); normalization=np.concatenate([r['normalization'] for r in records])
    quadratic=np.concatenate([r['quadratic'] for r in records])
    def evaluate_record(r,camera,raw=False):
        row=raw_row(r['row']) if raw else r['row']
        if r['family']=='original_absolute':return r['call'](camera,[row],False)
        return r['factor']._residual(q,[row],cameras={PTS:camera},**({'apply_optical_support':not raw} if r['family'].endswith('board_edges') else {}))
    def evaluate(camera): return np.concatenate([evaluate_record(r,camera) for r in records])
    def loss(z):
        sq=np.sqrt(z); rho=np.empty((3,len(z))); robust=(~quadratic)&(z>4.)
        rho[0]=z;rho[1]=1.;rho[2]=0.
        rho[0,robust]=4.*sq[robust]-4.;rho[1,robust]=2./sq[robust];rho[2,robust]=-1./(z[robust]*sq[robust])
        return rho*weights[None,:]
    def metrics(camera):
        result=[]
        for r in records:
            actual=evaluate_record(r,camera); raw=evaluate_record(r,camera,True);sigma=np.asarray(r['row']['sigma'])
            rawpx=raw.reshape(-1,2)*sigma[:,None];outside=actual.reshape(-1,2)*sigma[:,None]
            result.append(dict(family=r['family'],id=r['source_id'],physical_group=r['physical_group'],points=len(sigma),
                raw_native_px=stats(np.linalg.norm(rawpx,axis=1)),outside_support_native_px=stats(np.linalg.norm(outside,axis=1)),
                raw_mean_vector_xy_px=rawpx.mean(axis=0),outside_mean_vector_xy_px=outside.mean(axis=0),
                weighted_current_cost=weighted_source_cost(actual,r['quadratic'],SourceLossWeights(r['weights'],r['normalization']))))
        return result
    def physical(camera):
        worlds=[r['row']['world'] if r['family']=='original_absolute' else frozen_world[(id(r['factor']),json.dumps(r['row']['curve'],sort_keys=True))] for r in records]
        projections=[project_central_valid(camera,np.asarray(w),with_jacobian=False) for w in worlds]
        return dict(camera25_finite=bool(np.isfinite(camera).all()),optical_C_Z_scene_units=float(camera[2]),optical_C_above_ice=bool(camera[2]>0.),
            all_PP_Brown_aspect_fields_byte_exact=bool(np.array_equal(camera[8:],c0[8:]) and camera[6]-camera[7]==c0[6]-c0[7]),
            all_cached_complete_curve_vertices_forward=bool(all(np.all(x[1]>0.) for x in projections)),
            cached_complete_curve_vertex_projection_valid_fraction=float(np.mean(np.concatenate([x[2] for x in projections]))),
            minimum_cached_complete_curve_vertex_depth=float(min(np.min(x[1]) for x in projections)),
            same_frame_actual_observation_operators_return_finite=bool(np.isfinite(evaluate(camera)).all()),
            complete_production_physical_guards_evaluated=False,
            cached_complete_curve_statistics_are_not_query_validity='Complete perimeter includes invisible/behind-camera vertices. Actual value operators retain their original valid finite-segment and typed-query selection.')
    baseline=evaluate(c0); current_cost=weighted_source_cost(baseline,quadratic,SourceLossWeights(weights,normalization))
    assert np.isclose(.5*loss((baseline*normalization)**2)[0].sum(),current_cost,rtol=1e-13,atol=1e-13)
    result=dict(schema='rinkspace.single-frame-fixed-world-current-source-camera-condition-comparison.v1',q_sha256=QID,actual_pts=PTS,
        source_time_base='1/15360',config=ref(CONFIG),input_manifest=ref(RUN/'input_manifest.json'),state_identity=ref(RUN/'accepted_states'/QID/'state_identity.json'),
        camera=ref(identity['camera_path']),script=ref(__file__),world_provenance={k:v for k,v in world_proof.items() if k!='explicit_saved_physical_geometry_initializer'},original_source_packet=dict(path=str(PACKET),sha256=PACKET_SHA256),
        native_packets=[ref(opts['--additional-ice-source']),ref(opts['--additional-board-source'])],original_operator_authorities=call_proof,
        fixed_world_and_PP_Brown_aspect_mount_pivot=True,physical_source_instance_rows=15,actual_active_UV_count=355,
        native_scalar_count=sum(len(r['quadratic']) for r in records if r['family']!='original_absolute'),
        original_quadratic_scalar_count=sum(int(r['quadratic'].sum()) for r in records if r['family']=='original_absolute'),
        total_quadratic_scalar_count=int(quadratic.sum()),original_Huber_scalar_count=int((~quadratic).sum()),
        exact_production_scalar_loss_masks=[dict(id=r['source_id'],saved_global_scalar_range=r['saved_global_scalar_range'],
            quadratic_mask=r['quadratic'],active_original_point_indices=r['original_active_point_indices']) for r in records],
        original_absolute_saved_scalar_order_exhaustively_matches_scales_and_slice=True,
        projection_value_return_contract='project_central_valid(camera, world, with_jacobian=False) -> UV, depth, valid',
        same_current_original_correction_qualification_compaction_and_curve_budget=True,production_state_written=False,
        original_full_programme_motion_J_or_profiles_loaded=False,other_8502_cameras_reconstructed=False,
        baseline=dict(camera25=c0,current_cost=current_cost,rows=metrics(c0),physical_domain=physical(c0)),modes={},
        scope='Conditional fixed-world source compatibility only; local convergence does not prove global compatibility, infeasibility or uniqueness; all sources are FIT/development, no independent accuracy acceptance.',
        accuracy_accepted=False)
    print(json.dumps(dict(stage='single_frame_original_interface_preparation_complete',rows=15,points=355,baseline_cost=current_cost,
        seconds=time.perf_counter()-started)),flush=True)
    for mode in ['fixed_rig_RF4','free_camera_C_R_logf7']:
        t=time.perf_counter()
        if mode=='fixed_rig_RF4':
            initial=v0.copy()
            def camera(p):return calibration.camera25(ViewState.from_state4(p,effective_pp_offset_px=pp))
        else:
            initial=np.r_[c0[:3]/100.,c0[3:6],float(np.mean(c0[6:8]))]
            def camera(p):
                out=c0.copy();out[:3]=p[:3]*100.;out[3:6]=p[3:6];out[6:8]+=p[6]-float(np.mean(c0[6:8]));return out
        def residual(p):return evaluate(camera(p))*normalization
        record=dict(parameter_count=len(initial),parameter_chart='ViewState R3/domain logf' if len(initial)==4 else 'C3/100,R3,final mean logf',
            solver='scipy.optimize.least_squares default TRF termination and default evaluation allowance; 3-point actual value FD; x_scale=jac',
            original_component_Huber2_quadratic_and_outside_integration_weight=True)
        try:
            j0=approx_derivative(residual,initial,method='3-point');record['initial_unrobust_working_value_singular_values']=np.linalg.svd(j0,compute_uv=False)
            solved=least_squares(residual,initial,jac='3-point',loss=loss,x_scale='jac')
            candidate=camera(solved.x);j1=approx_derivative(residual,solved.x,method='3-point')
            record.update(success=bool(solved.success),status=int(solved.status),termination=solved.message,nfev=int(solved.nfev),njev=int(solved.njev),
                current_cost=float(solved.cost),actual_recomputed_cost=weighted_source_cost(evaluate(candidate),quadratic,SourceLossWeights(weights,normalization)),
                optimality=float(solved.optimality),diagnostic_camera25=candidate,diagnostic_chart_values=solved.x,
                optical_C_delta_scene_units=candidate[:3]-c0[:3],rotation_vector_delta=candidate[3:6]-c0[3:6],logf_delta=float(np.mean(candidate[6:8]-c0[6:8])),
                focal_px=float(np.exp(np.mean(candidate[6:8]))),final_unrobust_working_value_singular_values=np.linalg.svd(j1,compute_uv=False),
                final_solver_robust_weighted_singular_values=np.linalg.svd(solved.jac,compute_uv=False),rows=metrics(candidate),physical_domain=physical(candidate))
            assert np.isclose(record['current_cost'],record['actual_recomputed_cost'],rtol=1e-12,atol=1e-12)
            if len(initial)==7:
                rig=calibration.camera25(calibration.view_from_camera25(candidate))
                record['free_C_minus_fixed_rig_C_same_R_f_scene_units']=candidate[:3]-rig[:3]
                record['free_C_gap_from_fixed_rig_scene_units']=float(np.linalg.norm(candidate[:3]-rig[:3]))
        except Exception as error:
            record.update(success=False,error_type=type(error).__name__,error=str(error),generic_residual_fallback=False,retry_performed=False)
        record['seconds']=time.perf_counter()-t;result['modes'][mode]=record
        OUTPUT.write_text(json.dumps(result,default=plain,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(mode=mode,success=record['success'],cost=record.get('current_cost'),termination=record.get('termination',record.get('error')),seconds=record['seconds'])),flush=True)
    result['total_seconds']=time.perf_counter()-started
    OUTPUT.write_text(json.dumps(result,default=plain,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(OUTPUT),sha256=sha(OUTPUT),bytes=OUTPUT.stat().st_size,total_seconds=result['total_seconds'])),flush=True)

if __name__=='__main__':main()
