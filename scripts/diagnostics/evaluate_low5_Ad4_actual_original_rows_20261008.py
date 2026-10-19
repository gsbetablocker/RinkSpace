"""Small original low5 Ad4 FIT/HELD replay; no programme graph or new source."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from verify_original225_derived_center_20261008 import numeric_data_loads
from structured_programme_source_candidate import original_review_source as review
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from z190_physical_camera import PhysicalCameraModel

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    base=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    old=base/'okada_global_geometry_boardtracks_midtime_absolute_20261007/accepted_states/696fd93d920fe5ebd3590d707ed03196587e779a48f661fd9124df6c31b50351'
    final=base/'okada_unlocked_geometry_696f_derived_C0_step_domains_20261008/accepted_states/af8cb83286f4125560452831b8113aa4230acb1037f64f49bff354d35723b4ba'
    aux=base/'user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json'
    assert sha(review.PACKET)==review.PACKET_SHA256
    packet=numeric_data_loads(review.PACKET.read_bytes())
    rows=[(i,item)for i,item in enumerate(packet['programme_absolute_rows'])if item['block']=='low'
          and int(item['group_actual_pts'])in(229945856,229953536,229961216)and 'ad4:' in item['source_row']['id']]
    assert len(rows)==8
    review.pickle=SimpleNamespace(loads=numeric_data_loads);calls,authorities=review.load_original_absolute_evaluators()
    results=[]
    for directory in (old,final):
        ident=read(directory/'state_identity.json');provider,world,proof=current_complete_source_geometry(directory/'complete_state_UNACCEPTED.npz',aux)
        assert sha(ident['camera_path'])==ident['camera_sha256']
        model,pts,aliases,views=PhysicalCameraModel.load_npz(ident['camera_path']);cameras={};stats=[]
        for index,item in rows:
            source=item['source_row'];actual=int(item['group_actual_pts']);representative=aliases[actual]
            if representative not in cameras:cameras[representative]=model.camera25(views[representative])
            provider.register_row(source,actual_pts=actual,allow_previously_qualified_held_ad2_representation=True)
            current=provider.rebind(source,world);assert review.observation_signature(source)==review.observation_signature(current)
            r=np.asarray(calls['low'](cameras[representative],[current],False),float)
            px=(r*np.repeat(np.asarray(source['sigma']),2)).reshape(-1,2);d=np.linalg.norm(px,axis=1)
            stats.append(dict(packet_index=index,id=source['id'],actual_pts=actual,native_pts=source.get('source_native_pts'),
                              split=source.get('split'),collection_role=item['collection_role'],points=len(source['uv']),
                              source_UV=np.asarray(source['uv']).tolist(),source_sigma=np.asarray(source['sigma']).tolist(),
                              original_operator_native_vectors_px=px.tolist(),median_px=float(np.median(d)),p95_px=float(np.quantile(d,.95)),
                              maximum_px=float(d.max()),normalized_rms=float(np.sqrt(np.mean(r*r))),
                              source_role_and_split_not_changed=True))
        results.append(dict(state=ident,actual_original_low_operator=authorities['low'],rows=stats))
        print(json.dumps(dict(state=ident['q_sha256'],rows=[{k:r[k]for k in ('id','actual_pts','collection_role','p95_px')}for r in stats])),flush=True)
    out=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/external_expert_20261007/all_original_absolute_source_audit/native_rgb_review/low5_Ad4_existing_FIT_HELD_696f_to_af8_actual_operator_20261008.json')
    out.write_text(json.dumps(dict(schema='rinkspace.low5-Ad4-existing-FIT-HELD-actual-small-replay.v1',source_packet_sha256=review.PACKET_SHA256,
                                  states=results,source_UV_sigma_roles_splits_unchanged=True,source_HELD_not_promoted=True,
                                  full_source_graph_loaded=False,new_observations_added=0,accuracy_accepted=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(out),sha256=sha(out))),flush=True)
if __name__=='__main__':main()
