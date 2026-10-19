"""Replay a bounded original print time scope on immutable actual states."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from verify_original225_derived_center_20261008 import numeric_data_loads
from structured_programme_source_candidate import original_review_source as review
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from z190_physical_camera import PhysicalCameraModel
from z190_source_camera_diagnostic import source_errors
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state-dir',type=Path,action='append',required=True)
    ap.add_argument('--auxiliary',type=Path,required=True);ap.add_argument('--block',required=True)
    ap.add_argument('--ad-number',type=int,required=True);ap.add_argument('--lower-pts',type=int,required=True)
    ap.add_argument('--upper-pts',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert sha(review.PACKET)==review.PACKET_SHA256
    data=numeric_data_loads(review.PACKET.read_bytes());rows=[]
    for index,item in enumerate(data['programme_absolute_rows']):
        row=item['source_row'];number=a.ad_number;name=str(row['id']).lower()
        if item['block']!=a.block or not a.lower_pts<=int(item['group_actual_pts'])<=a.upper_pts:continue
        if not (f'ad{number}:' in name or f'ad{number}_'in name or f'advertisement{number}.'in name
                or any(str(i).startswith(f'ad{number}:')for i in(row.get('physical_ids')or []))):continue
        rows.append((index,item))
    assert rows
    review.pickle=SimpleNamespace(loads=numeric_data_loads);calls,auth=review.load_original_absolute_evaluators();results=[]
    for directory in a.state_dir:
        identity=read(directory/'state_identity.json');owner,world,proof=current_complete_source_geometry(directory/'complete_state_UNACCEPTED.npz',a.auxiliary)
        assert sha(identity['camera_path'])==identity['camera_sha256'];model,pts,aliases,views=PhysicalCameraModel.load_npz(identity['camera_path']);cams={};stats=[]
        for index,item in rows:
            source=item['source_row'];actual=int(item['group_actual_pts']);representative=aliases[actual]
            if representative not in cams:cams[representative]=model.camera25(views[representative])
            owner.register_row(source,actual_pts=actual,allow_previously_qualified_held_ad2_representation=True)
            current=owner.rebind(source,world);assert review.observation_signature(source)==review.observation_signature(current)
            r=np.asarray(calls[a.block](cams[representative],[current],False),float)
            pixel=(r*np.repeat(np.asarray(source['sigma']),2)).reshape(-1,2);d=np.linalg.norm(pixel,axis=1)
            display=review.original_pixel_error_report(cams[representative],current,a.block,source_errors)
            stats.append(dict(packet_index=index,id=source['id'],actual_pts=actual,canonical_representative_pts=int(representative),
                              source_row_pts=int(source['pts']),native_pts=source.get('source_native_pts'),split=source.get('split'),
                              collection_role=item['collection_role'],UV_count=len(source['uv']),source_sigma=np.asarray(source['sigma']).tolist(),
                              source_origin=source.get('origin'),normalized_rms=float(np.sqrt(np.mean(r*r))),
                              original_operator_outside_support_median_px=float(np.median(d)),original_operator_outside_support_p95_px=float(np.quantile(d,.95)),
                              original_operator_outside_support_maximum_px=float(d.max()),original_pixel_error_report=display,
                              source_observation_signature=review.observation_signature(source),source_roles_or_splits_changed=False))
        results.append(dict(state=identity,actual_original_evaluator=auth[a.block],rows=stats))
        print(json.dumps(dict(q_sha256=identity['q_sha256'],rows=[{k:r[k]for k in ('id','actual_pts','collection_role','original_operator_outside_support_p95_px','normalized_rms')}for r in stats])),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='rinkspace.actual-original-print-bounded-time-source-replay.v1',source_packet_sha256=review.PACKET_SHA256,
                                       block=a.block,ad_number=a.ad_number,lower_pts=a.lower_pts,upper_pts=a.upper_pts,states=results,
                                       source_UV_sigma_roles_PTS_splits_unchanged=True,HELD_not_promoted=True,whole_graph_loaded=False,
                                       new_source_rows=0,accuracy_accepted=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(a.output),sha256=sha(a.output))),flush=True)
if __name__=='__main__':main()
