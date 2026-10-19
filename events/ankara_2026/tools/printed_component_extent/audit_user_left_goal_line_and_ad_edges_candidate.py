"""Data-only same-q public 49-group audit; no full source graph or J/H pool."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics'),str(HERE)]
sys.dont_write_bytecode=True
import numpy as np
from verify_actual_printed_extent_candidate import load_state,read,sha
from calibration.solver_runtime import configure_native_runtime
configure_native_runtime(stage='same_q_user_source_append_49groups_read_only',native_threads=1)
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth,UserConfirmedSourceTruth,direct_manual_centre_factors
from structured_programme_source_candidate.current_wide_source import CurrentWideSource
from structured_programme_source_candidate.source_block_target_extension_replay import _native_append_plan

def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def exact(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return set(a)==set(b)and all(exact(v,b[k])for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
    return type(a)==type(b)and a==b
def bind_human(state,board,ice,paths,count):
    human=HumanStrokeTruth(state,paths['--direct-human-truth-authority'],board,ice,parameter_count=count)
    human.bind_full_stroke_guard(paths['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(paths['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(paths['--practical-human-location-envelope'])
    return human
def wide_factor(state,identity,options,count,location):
    authority=read(options['--original-wide-human-truth-authority']);trace=read(authority['trace_packet_reference']['path'])
    provider=state.geometry
    factor=CurrentWideSource.__new__(CurrentWideSource);factor.geometry=provider.current;factor.qualified_scene=provider.qualified_scene
    factor.geometry_parameter_indices=state.geometry_parameter_indices;factor.observations=[];factor.ad_packet={'rows':[]}
    factor.template_camera25=np.zeros(25);factor.trace_packet=trace
    factor.trace_uv=[np.asarray(r['observed_centres_source_px'],float)for r in trace['contours']]
    with np.load(Path(identity['_accepted_directory'])/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        idx=np.asarray(z['aux__current_wide_camera_parameter_indices'],int).copy()
    scene=read(Path(identity['_accepted_directory'])/'scene_UNACCEPTED.json')
    cameras=[r['camera25']for r in scene['frame_cameras']if r.get('source_pts')==56218880];assert len(cameras)==1
    factor.template_camera25=np.asarray(cameras[0],float)
    truth=WideStrokeTruth(factor,options['--original-wide-human-truth-authority'],count,state.canonical_parameter_layout.canonical_to_evaluation[idx])
    truth.bind_practical_location_envelope(location);return truth
def values(human,wide,confirmed,q):
    primary=human.queries(q);full=human.queries(q,full_stroke=True);w=wide.queries(q);cv,cl,ch,cd=confirmed._carriers(q)
    buffers=dict(native=np.maximum([abs(r['query']['residual']).max()for r in primary],[abs(r['query']['residual']).max()for r in full]),
        wide=np.asarray([abs(r['query']['residual']).max()for r in w]),confirmed=np.asarray([r['maximum_violation_px']for r in cd]))
    return dict(primary=primary,full=full,wide=w,confirmed_carriers=(cv,cl,ch),confirmed_details=cd,buffers=buffers)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state-directory',type=Path,required=True);args=parser.parse_args()
    started=time.perf_counter();directory=args.state_directory.resolve();run=directory.parent.parent
    manifest=read(run/'input_manifest.json')
    options=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_conditional_views_20261009.json')['source_options']
    tracebase=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261009/a13_complete_programme/left_end_line_04_10_38_800')
    out=tracebase/'full_truth_append_candidate';receipt=read(out/'prepared_user_left_goal_line_full_truth_append_candidate.json')
    adbase=tracebase.parent/'registered_ad_edges_and_left_goal_source_append';adreceipt=read(adbase/'board_edge_append_receipt_v1.json')
    for item in list(receipt['paths'].values())+[adreceipt['board_packet'],adreceipt['new_confirmed_record_additions']]:assert sha(item['path'])==item['sha256']
    newpaths=dict(options)
    for option,key in [('--additional-ice-source','ice'),('--direct-human-truth-authority','human_authority'),('--direct-human-full-stroke-guard','full_guard'),('--direct-human-stroke-cap-support','stroke_caps'),('--practical-human-location-envelope','location')]:newpaths[option]=receipt['paths'][key]['path']
    newpaths['--additional-board-source']=adreceipt['board_packet']['path']
    # Rebind packet-container identities only; every original confirmed source
    # record, query, support and uncertainty stays exact apart from this ref.
    confirmed_meta=copy.deepcopy(read(options['--user-confirmed-source-truth-authority']));oldconfirmed=copy.deepcopy(confirmed_meta['records']);changes=[]
    for i,record in enumerate(confirmed_meta['records']):
        path=newpaths['--additional-board-source'if record['family']=='native_board_edge'else'--additional-ice-source']
        before=copy.deepcopy(record);record['packet_reference']=ref(path)
        retained=copy.deepcopy(record);retained['packet_reference']=before['packet_reference'];assert retained==before
        original=read(before['packet_reference']['path'])['rows'][record['original_row_index']]
        current=read(path)['rows'][record['original_row_index']];assert original==current
        changes.append(dict(id=record['id'],before=before['packet_reference'],after=record['packet_reference'],all_other_fields_exact=True,original_source_row_all_fields_exact=True))
    additions=read(adreceipt['new_confirmed_record_additions']['path'])['records'];assert len(additions)==5
    confirmed_meta['records'].extend(copy.deepcopy(additions));confirmed_meta['record_count']=len(confirmed_meta['records'])
    confirmed_meta['source_container_append_binding']=dict(parent_authority=ref(options['--user-confirmed-source-truth-authority']),new_source_authority=adreceipt['new_confirmed_record_additions'],original6_records_exact_except_packet_reference=True,packet_reference_changes=changes,old_rows_and_original_query_supports_preserved=True)
    confirmedpath=out/'confirmed_parent6_plus_user_five_ad_edges_with_exact_appended_packet_bindings_v1_CANDIDATE.json';dump(confirmedpath,confirmed_meta)
    newpaths['--user-confirmed-source-truth-authority']=str(confirmedpath)
    auxiliary=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json')
    state,q,identity,camera_error=load_state(directory,auxiliary,manifest);assert camera_error==0.;identity['_accepted_directory']=str(directory)
    qbefore=q.tobytes();caps=read(directory/'truth_restoration_state.json');assert caps['complete_q_sha256']==identity['q_sha256']
    board_old=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice_old=NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    board_new=NativeBoardEdgeFactor(state,newpaths['--additional-board-source'],len(q));ice_new=NativeIceCurveFactor(state,newpaths['--additional-ice-source'],len(q))
    plan=_native_append_plan(dict(additional_real_native_board_edges=board_old.contract,additional_real_native_ice_curves=ice_old.contract),dict(additional_real_native_board_edges=board_new.contract,additional_real_native_ice_curves=ice_new.contract),('additional_real_native_board_edges','additional_real_native_ice_curves'))
    public_old_rows={}
    for name,old,new in [('board',board_old,board_new),('ice',ice_old,ice_new)]:
        oldpacket=read(old.packet_path);newpacket=read(new.packet_path);assert newpacket['rows'][:len(oldpacket['rows'])]==oldpacket['rows']
        assert np.array_equal(old.scales,new.scales[:len(old.scales)])and np.array_equal(old.quadratic,new.quadratic[:len(old.quadratic)])
        oldvalues=old._residual(q,old.rows+old.held);newvalues=new._residual(q,new.rows[:len(old.rows)]+new.held[:len(old.held)])
        assert exact(oldvalues,newvalues)
        public_old_rows[name]=dict(FIT_groups=len(old.rows),HELD_groups=len(old.held),old_all_fit_and_held_value_bytes_exact=True,old_scales_and_mask_bytes_exact=True,added_FIT_scalars=new.row_count-old.row_count)
    oldhuman=bind_human(state,board_old,ice_old,options,len(q));newhuman=bind_human(state,board_new,ice_new,newpaths,len(q))
    oldwide=wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope']);newwide=wide_factor(state,identity,newpaths,len(q),newpaths['--practical-human-location-envelope'])
    oldconfirmedtruth=UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board_old,ice_old,parameter_count=len(q))
    newconfirmedtruth=UserConfirmedSourceTruth(state,confirmedpath,board_new,ice_new,parameter_count=len(q))
    oldresult=values(oldhuman,oldwide,oldconfirmedtruth,q);newresult=values(newhuman,newwide,newconfirmedtruth,q)
    support_fields=('uv','normals','bounds','tangent_bounds','original_raster_normal_bounds','original_source_tangent_bounds')
    for olditems,newitems in [(oldhuman.records,newhuman.records[:35]),(oldhuman.full_stroke_records,newhuman.full_stroke_records[:35]),(oldwide.records,newwide.records)]:
        assert len(olditems)==len(newitems)
        for a,b in zip(olditems,newitems):
            for key in support_fields:
                if key in a:assert exact(a[key],b[key]),key
    for a,b in zip(oldconfirmedtruth.records,newconfirmedtruth.records[:6]):
        for key in ('radius','normals','bounds'):assert exact(a[key],b[key]),key
    for key,n in [('primary',35),('full',35),('wide',2)]:assert exact(oldresult[key],newresult[key][:n]),key
    for oldbuffer,newbuffer in zip(oldresult['confirmed_carriers'],newresult['confirmed_carriers']):assert exact(oldbuffer,newbuffer[:len(oldbuffer)])
    assert exact(oldresult['confirmed_details'],newresult['confirmed_details'][:6])
    location=read(newpaths['--practical-human-location-envelope']);uncertainties=dict(native=np.asarray([r.get('engineering_uncorrelated_location_scale_px',r['normal_expansion_px'])for r in location['records']if r['family']=='native'],float),wide=np.asarray([r.get('engineering_uncorrelated_location_scale_px',r['normal_expansion_px'])for r in location['records']if r['family']=='wide'],float),confirmed=np.asarray([item['radius'].min()for item in newconfirmedtruth.records],float))
    ids=dict(native=[r['id']for r in newresult['primary']],wide=[r['id']for r in newresult['wide']],confirmed=[r['id']for r in newresult['confirmed_details']])
    audits={}
    for family,oldn in [('native',35),('wide',2),('confirmed',6)]:
        value=newresult['buffers'][family];oldcap=caps['families'][family];uncertainty=uncertainties[family]
        assert len(oldcap['group_ids'])==oldn and ids[family][:oldn]==oldcap['group_ids']
        assert exact(value[:oldn],oldresult['buffers'][family])
        assert np.array_equal(np.asarray(oldcap['uncertainties'],float),uncertainty[:oldn])
        assert np.array_equal(np.asarray(oldcap['limits'],float),np.zeros(oldn))and np.all(value[:oldn]==0.)
        initial=np.where(value<=1e-8,0.,value+uncertainty)
        assert np.array_equal(initial[:oldn],np.asarray(oldcap['limits'],float))
        audits[family]=dict(old_groups=oldn,new_total_groups=len(value),old_queries_support_uncertainties_and_zero_limits_exact=True,original_maximum_violation_px=float(value[:oldn].max()),rows=[dict(id=i,violation_px=float(v),engineering_uncertainty_px=float(u),existing_fresh_initial_limit_px=float(c))for i,v,u,c in zip(ids[family],value,uncertainty,initial)])
    assert sum(a['new_total_groups']for a in audits.values())==49
    yellow_detail={}
    for key in ('primary','full'):
        entry=newresult[key][-1];query=entry['query'];imax=int(np.argmax(np.abs(query['residual'])))
        yellow_detail[key]=dict(query_count=len(query['signed_normal_px']),maximum_normal_violation_px=float(abs(query['normal_violation_px']).max()),maximum_tangent_coverage_violation_px=float(abs(query['coverage_violation_px']).max()),
            normal_signed_px=dict(minimum=float(query['signed_normal_px'].min()),median=float(np.median(query['signed_normal_px'])),maximum=float(query['signed_normal_px'].max())),
            finite_scanline_intersection_count=int(np.count_nonzero(query['finite_scanline_intersection'])),nonintersecting_source_vertex_indices=np.flatnonzero(~query['finite_scanline_intersection']).tolist(),
            largest_violation_flat_scalar_index=imax,largest_violation_source_vertex_index=imax//2,largest_violation_component='normal'if imax%2==0 else'tangent_coverage',
            signed_normal_px=query['signed_normal_px'].tolist(),normal_violation_px=query['normal_violation_px'].tolist(),coverage_tangent_px=query['coverage_tangent_px'].tolist(),coverage_violation_px=query['coverage_violation_px'].tolist())
    # Same source frozen query / dense guard / pixel cell cap identities.
    sr=read(tracebase/'registered_user_yellow_left_goal_line_source_trace_v2.json')
    with np.load(sr['source_trace_arrays']['path'],allow_pickle=False)as z:
        for key in ('uv','normal','normal_interval_px'):assert np.array_equal(z[key],newhuman.full_stroke_records[-1]['uv'if key=='uv'else'source_normals'if key=='normal'else'original_raster_normal_bounds'])
    assert newhuman.records[-1]['row']['pts']==230995968 and newhuman.records[-1]['row']['actual_pts']==230995712
    assert q.tobytes()==qbefore
    oldcentres=direct_manual_centre_factors(state,oldhuman,options['--additional-board-source'],options['--additional-ice-source'],len(q))
    newcentres=direct_manual_centre_factors(state,newhuman,newpaths['--additional-board-source'],newpaths['--additional-ice-source'],len(q))
    oldcentrecount=sum(f.row_count for f in oldcentres);newcentrecount=sum(f.row_count for f in newcentres)
    report=dict(schema='rinkspace.same-complete-q-public-user-source-append-49groups-audit.v1',passed=True,complete_state_reference=ref(directory/'complete_state_UNACCEPTED.npz'),complete_q_sha256=identity['q_sha256'],camera_archive_reproduce_max=camera_error,
        candidate_source_paths={k:ref(v)for k,v in newpaths.items()if k in('--additional-ice-source','--additional-board-source','--direct-human-truth-authority','--direct-human-full-stroke-guard','--direct-human-stroke-cap-support','--practical-human-location-envelope','--user-confirmed-source-truth-authority')},
        old43_restoration_state=ref(directory/'truth_restoration_state.json'),families=audits,old_public_source_rows=public_old_rows,native_exact_append_plan=plan,
        confirmed_old6_container_reference_rebindings=changes,old43_actual_public_query_arrays_and_all_supports_bitexact=True,old43_caps_fresh_init_still_zero=True,
        existing_fresh_initializer_without_old_truth_restoration_state_is_equivalent_for_all_original43=True,new6_groups_use_only_actual_new_violation_and_original_engineering_uncertainty=True,
        source_presentation_pts=230995968,source_actual_exposure_pts=230995712,source_pixels_and_registered_UV_not_modified=True,source_RGB_byte_identity_between_aliases_not_claimed=True,
        new_yellow_line_public_normal_and_finite_tangent_coverage=yellow_detail,
        new_sparse_yellow_queries=33,new_complete_yellow_guard_vertices=647,new_drawing_footprint_caps=2,total_truth_groups=49,added_board_FIT_scalars=board_new.row_count-board_old.row_count,added_ice_FIT_scalars=ice_new.row_count-ice_old.row_count,
        old_parent_source_rows=caps['source_target']['source_rows'],new_source_scalar_count_before_direct_centre_append=caps['source_target']['source_rows']+board_new.row_count-board_old.row_count+ice_new.row_count-ice_old.row_count,
        original_direct_human_centre_scalars=oldcentrecount,new_direct_human_centre_scalars=newcentrecount,added_direct_human_centre_scalars=newcentrecount-oldcentrecount,
        new_complete_source_scalar_count=caps['source_target']['source_rows']+board_new.row_count-board_old.row_count+ice_new.row_count-ice_old.row_count+newcentrecount-oldcentrecount,
        full_graph_created=False,full_source_J_H_or_pool_created=False,production_runtime_or_config_modified=False,accuracy_accepted=False,seconds=time.perf_counter()-started)
    reportpath=out/'same_q4bb_public_user_left_goal_line_and_five_ad_edges_49groups_audit.json';dump(reportpath,report)
    print(json.dumps(dict(passed=True,path=str(reportpath),seconds=report['seconds'],families={k:dict(groups=v['new_total_groups'],new_rows=v['rows'][v['old_groups']:])for k,v in audits.items()}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
