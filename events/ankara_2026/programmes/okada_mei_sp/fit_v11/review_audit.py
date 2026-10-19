"""Same complete physical model with current native source points and plan view."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from rinkfit_start.semantic_resolution import load_semantics
from rinkfit_start.regression import NativeRegression


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','checkpoint-sha256','source-selection','compiled-adapter-dir','output-dir','run-root','semantic-manifest']:
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--first-frame-only',action='store_true');a=ap.parse_args()
    base=Path(__file__).with_name('review_native.py')
    spec=importlib.util.spec_from_file_location('native_review_shared',base);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cp=json.loads(Path(a.checkpoint).read_text(encoding='utf-8'))
    if cp.get('schema')=='rinkfit.complete_recomputed_native_candidate.v1':
        assert cp['semantic']['sha256']==sha(a.semantic_manifest),'Current native points and candidate metrics must use the same semantic scope'
    mdoc=json.loads(Path(cp['model']['path']).read_text(encoding='utf-8'))
    reg=NativeRegression(a.run_root,a.semantic_manifest)
    load_semantics(a.semantic_manifest,reg.n,expected_sha256=sha(a.semantic_manifest))
    scene=m.NativeScene(mdoc);X,records,_,_=m.inventory(scene,mdoc)
    plan=np.full((1080,640,3),(245,247,248),np.uint8)
    def planuv(xyz):return np.c_[320+xyz[:,1]*1000,540-xyz[:,0]*1000]
    q=planuv(X)
    for r in records:
        pts=np.rint(q[r['begin']:r['end']]).astype(np.int32)
        if r['material_id']<=-2:
            if len(pts):cv2.putText(plan,r['label'],tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.35,(25,70,100),1,cv2.LINE_AA)
        elif len(pts)>1:cv2.polylines(plan,[pts],False,r['color'],1,cv2.LINE_AA)
    cv2.putText(plan,'SAME COMPLETE MODEL',(90,20),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,50,70),1,cv2.LINE_AA)
    cv2.putText(plan,'Normalized L=1; metric scale unverified',(25,1065),cv2.FONT_HERSHEY_SIMPLEX,.46,(35,60,75),1,cv2.LINE_AA)
    original_draw=m.draw
    def draw(native,projection,geometry,pts,metrics):
        left,record=original_draw(native,projection,geometry,pts,metrics)
        ids=np.flatnonzero(reg.active&(reg.pts==pts))
        for uv in reg.uv[ids]:
            u,v=np.rint(uv).astype(int)
            if 0<=u<1920 and 0<=v<1080:cv2.circle(left,(u,v),2,(255,100,245),-1,cv2.LINE_AA)
        right=plan.copy()
        cv2.putText(right,'Native source PTS %d'%pts,(35,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,50,70),1,cv2.LINE_AA)
        record.update(current_active_source_points=int(len(ids)),source_points_from_exact_native_PTS=True,
            historical_or_excluded_source_points_drawn=0,plan_view_same_complete_physical_model=True)
        return np.hstack((left,right)),record
    m.draw=draw;m.WH=(2560,1080)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    proof=dict(schema='rinkfit.same_model_current_native_points_plan_review.v1',
        wrapper=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        base=dict(path=str(base),sha256=sha(base)),semantic=dict(path=a.semantic_manifest,sha256=sha(a.semantic_manifest)),
        complete_checkpoint=dict(path=a.checkpoint,sha256=sha(a.checkpoint)),
        raw_rows=reg.n,active_rows=int(reg.active.sum()),excluded_source_rows=np.flatnonzero(~reg.active).tolist(),
        source_clock='exact native PTS, no inter-frame source-point propagation',
        current_points_are_raw_native_UV=True,all_model_paths_and_ad_numbers=True,
        model_materials=len(mdoc['materials']),model_paths=sum(len(z['paths'])for z in mdoc['materials']),
        conditional_scale='L=1; not verified metric',video_generated=False)
    proof_path=out/'review_extension.json'
    if proof_path.exists():
        previous=json.loads(proof_path.read_text(encoding='utf-8'))
        for key in proof:
            if key!='video_generated':assert previous.get(key)==proof[key],'Changed review extension context '+key
    proof_path.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    m.run(a)
    proof['video_generated']=not a.first_frame_only
    (out/'review_extension.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')

if __name__=='__main__':main()
