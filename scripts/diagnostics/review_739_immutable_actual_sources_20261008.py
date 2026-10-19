"""Independent all225 and native edge/crease source replay on immutable q.

No motion graph, J, actor, fit, pixel remeasurement or source admission runs.
Raw source-centre distances and the existing optical-support objective differ.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from verify_original225_derived_center_20261008 import numeric_data_loads
from structured_programme_source_candidate import original_review_source as original
from structured_programme_source_candidate import ad14_source_role_certificate
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_numeric(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Non-numeric report object: '+type(value).__name__)
def distances(raw):
    d=np.linalg.norm(np.asarray(raw,float).reshape(-1,2),axis=1)
    return dict(mean_px=float(d.mean()),median_px=float(np.median(d)),p95_px=float(np.quantile(d,.95)),maximum_px=float(d.max()))

def evaluate(directory,auxiliary,board_packet,ice_packet,rows):
    identity=read(directory/'state_identity.json')
    assert sha(identity['camera_path'])==identity['camera_sha256']
    assert sha(directory/'scene_UNACCEPTED.json')==identity['scene_sha256']
    manifest=read(directory.parents[1]/'input_manifest.json')
    assert sha(directory.parents[1]/'input_manifest.json')==identity['input_manifest_sha256']
    for key,path in (('additional_real_native_board_edges',board_packet),('additional_real_native_ice_curves',ice_packet)):
        assert manifest[key]['packet_sha256']==sha(path)
    provider,world,proof=current_complete_source_geometry(directory/'complete_state_UNACCEPTED.npz',auxiliary)
    install_source_ice_material_view(provider)
    source=original.evaluate_original_review_rows(provider,world,identity['camera_path'],rows=rows,
        complete_q_sha256=identity['q_sha256'])
    assert source['complete_numeric_source_evaluation'] and len(source['rows'])==225
    with np.load(directory/'complete_state_UNACCEPTED.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files if k=='q' or k.startswith('aux__')}
    with np.load(identity['camera_path'],allow_pickle=False) as z:cameras={k:z[k].copy() for k in z.files}
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    q=state.evaluation_q(arrays['q'])
    board=NativeBoardEdgeFactor(state,board_packet,len(q));board_review=board.review(q)
    board_source={r['id']:r for r in board.rows+board.held}
    for r in board_review['rows']:
        s=board_source[r['id']]
        r.update(source_uv=s['uv'].tolist(),source_sigma_px=s['sigma'].tolist(),
            source_normal=s['normal'].tolist(),source_normal_interval_px=s['normal_interval'].tolist(),
            curve=s['curve'],source_raw_frame=s['source_raw_frame'],source_frame_sha256=sha(s['source_raw_frame']),
            finite_world_sample_count=len(board._world(q,s['curve'])))
    ice=NativeIceCurveFactor(state,ice_packet,len(q));icecam=ice._cameras(q);crease=[]
    scene=read(directory/'scene_UNACCEPTED.json');features={f['id']:f for f in scene['ice_features']}
    for s in ice.rows+ice.held:
        if s['curve']['kind']!='current_goal_crease':continue
        w=ice._world(q,s['curve']);exported=np.asarray(features[s['feature_id']]['polygon_xyz'])
        if not np.array_equal(exported[0],exported[-1]):exported=np.vstack((exported,exported[0]))
        assert np.array_equal(w,exported)
        raw=curve_vector_error_full_brown(icecam[s['actual_pts']],w,s['uv'],with_jacobian=False)
        if isinstance(raw,tuple):raw=raw[0]
        raw=np.asarray(raw,float).reshape(-1,2)
        supported=ice._residual(q,[s],cameras=icecam).reshape(-1,2)*s['sigma'][:,None]
        crease.append(dict(id=s['id'],native_pts=s['pts'],actual_pts=s['actual_pts'],split=s['split'],
            source_uv=s['uv'].tolist(),source_sigma_px=s['sigma'].tolist(),source_raw_frame=s['source_raw_frame'],
            source_frame_sha256=sha(s['source_raw_frame']),raw_source_center_error=distances(raw),
            actual_supported_factor_error=distances(supported),raw_model_minus_source_vector_px=raw.tolist(),
            actual_supported_factor_vector_px=supported.tolist(),source_optical_support_present=s['_source_optical_support'] is not None,
            world_equals_complete_current_exported_polygon=True))
    print(json.dumps({'q':identity['q_sha256'],'original_rows':len(source['rows']),
                      'native_board_rows':len(board_review['rows']),'goal_crease_rows':len(crease)}),flush=True)
    return dict(state=identity,world_proof=proof,actual_camera25_archive_reproduction_max=camera_error,
                original225=source,native_board=board_review,goal_crease=crease)

def main():
    ap=argparse.ArgumentParser()
    for k in ('baseline','state-dir','auxiliary','board-packet','ice-packet','output'):
        ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    # Both immutable numerical packets permit only NumPy numeric buffer/dtype
    # constructors. The Ad14 certificate reads the same authenticated packet.
    safe=SimpleNamespace(loads=numeric_data_loads)
    original.pickle=safe;ad14_source_role_certificate.pickle=safe
    rows=original.load_original_review_rows()
    before=evaluate(a.baseline,a.auxiliary,a.board_packet,a.ice_packet,rows)
    after=evaluate(a.state_dir,a.auxiliary,a.board_packet,a.ice_packet,rows)
    changes=original.compare_original_review_reports(before['original225'],after['original225'])
    old={(r['block'],r['group_order'],r['row_order']):r for r in before['original225']['rows']}
    raw_changes=[]
    for r in after['original225']['rows']:
        b=old[(r['block'],r['group_order'],r['row_order'])]
        raw_changes.append(dict(id=r['id'],block=r['block'],actual_pts=r['actual_pts'],
            original_source_split=r['original_source_split'],before_raw=b['original_pixel_error_report'],
            after_raw=r['original_pixel_error_report'],before_outside_support_p95_px=b['original_operator_outside_support_p95_px'],
            after_outside_support_p95_px=r['original_operator_outside_support_p95_px']))
    report=dict(schema='rinkspace.immutable-739-target-independent-actual-source-review.v1',before=before,after=after,
        original225_actual_same_operator_changes=changes,original225_raw_and_support_changes=raw_changes,
        source_UV_sigma_roles_support_clock_unchanged=True,HELD_promoted=False,
        numeric_only_restricted_source_decoder=True,whole_graph_J_actor_solver_loaded=False,
        complete_shot_visual_accuracy_accepted=False,source_and_screenshot_checks_are_not_whole_video_acceptance=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=json_numeric)+'\n',encoding='utf8')
    print(json.dumps({'report':str(a.output),'sha256':sha(a.output)}),flush=True)
if __name__=='__main__':main()
