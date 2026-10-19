"""Replay original static-board FIT and held sources using one complete state.

Read-only public factor evaluation. No fitting, reweighting or source admission.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,float)
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()))

def main():
    p=argparse.ArgumentParser()
    for name in ('state-directory','source-packet','auxiliary','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();identity=json.loads((a.state_directory/'state_identity.json').read_text(encoding='utf8'))
    source=a.state_directory/'complete_state_UNACCEPTED.npz'
    if sha(source)!=identity['state_npz_sha256']or sha(identity['camera_path'])!=identity['camera_sha256']:
        raise ValueError('Complete immutable state or camera archive differs')
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
        'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
        'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
        'aux__current_complete_parameter_layout_json')
    with np.load(source,allow_pickle=False)as archive:small={k:archive[k].copy()for k in keys}
    if hashlib.sha256(small['q'].tobytes()).hexdigest()!=identity['q_sha256']:
        raise ValueError('Complete q identity differs')
    provider,_,_=current_complete_source_geometry(source,a.auxiliary)
    state,camera_error=state_camera_adapter(small,dict(np.load(identity['camera_path'],allow_pickle=False)),provider)
    q=state.evaluation_q(small['q']);factor=NativeBoardTrackFactor(state,a.source_packet,len(q));rows=[]
    for row in factor.rows+factor.held+factor.anchor_rows+factor.anchor_held:
        _,predicted,world=factor._evaluate(q,row)
        observed=row['uv']if row.get('_anchor')else row['uv1'];delta=predicted-observed
        rows.append(dict(id=row['id'],split=row['split'],source_pts=int(row.get('source_pts',row.get('pts',0))),
            target_pts=int(row.get('actual_pts',row.get('target_actual_pts',row.get('target_pts',0)))),
            advertisement_number=row.get('advertisement_number',row.get('counterpart_advertisement_id',row.get('advertisement_id'))),
            actual_source_anchor=bool(row.get('_anchor')),native_error_px=stats(np.linalg.norm(delta,axis=1)),
            signed_median_xy_px=np.median(delta,axis=0).tolist(),observed_UV=observed.tolist(),projected_UV=predicted.tolist()))
    result=dict(schema='rinkspace.readonly-complete-native-board-source-replay.v1',q_sha256=identity['q_sha256'],
        state_path=str(source),source_packet=dict(path=str(a.source_packet),sha256=sha(a.source_packet)),
        actual_camera_archive_reproduce_max=camera_error,source_factor_contract=factor.contract,rows=rows,
        source_or_parameter_modifications=False,whole_programme_accuracy_accepted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(q_sha256=identity['q_sha256'],row_groups=len(rows),camera_error=camera_error,
        latest_secondary_anchors=[{k:r[k]for k in ('id','split','native_error_px','signed_median_xy_px')}for r in rows if 'secondary' in r['id']])))

if __name__=='__main__':main()
