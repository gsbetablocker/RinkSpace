"""Replay the authenticated original HELD225 on the full current world layout."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.original_review_source import (
    load_original_review_rows,evaluate_original_review_rows,observation_signature)
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    for name in ('state','camera-identity','auxiliary','output'):parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();begin=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary);install_source_ice_material_view(provider)
    identity=read(a.camera_identity)
    if sha(identity['camera_path'])!=identity['camera_sha256']:raise ValueError('Frozen canonical camera source changed')
    with np.load(a.state,allow_pickle=False)as z:q=z['q'].copy()
    rows=load_original_review_rows();before=[observation_signature(r['source_row'])for r in rows]
    report=evaluate_original_review_rows(provider,world,identity['camera_path'],rows=rows,
        complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest())
    assert len(report['rows'])==225 and report['source_UV_context_count']==3495
    assert all(observation_signature(r['source_row'])==signature for r,signature in zip(rows,before))
    assert report['world_parameter_count']==len(world)==98
    assert not any(r.get('error_phase')=='current_material_binding'for r in report['rows'])
    report.update(actual_state_path=str(a.state),actual_state_sha256=sha(a.state),world_proof=proof,
        all_original_UV_sigma_split_and_operator_signatures_preserved=True,
        source_HELD_registration_does_not_add_FIT_observations=True,whole_source_graph_or_J_not_constructed=True,
        elapsed_seconds=time.perf_counter()-begin)
    target=a.output/'original225_current_world98_review_check_v1.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,
        default=lambda v:v.item()if isinstance(v,np.generic)else v.tolist())+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(target),world_count=len(world),statuses=report['evaluation_status_counts'],
        source_rows=report['source_rows'],source_contexts=report['source_UV_context_count'],
        source_UV_sigma_roles_preserved=True,seconds=report['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
