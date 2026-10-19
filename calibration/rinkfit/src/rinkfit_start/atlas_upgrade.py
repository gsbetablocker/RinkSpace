"""Estimate previously unknown component Q against explicit common P candidates.

This consumes all available frame mappings in each connected component.  P is
an initialization candidate with declared provenance; the result is conditional
and never turns old camera/geometry into measured metric truth. Derived Q/H/P
factors are absent from the final physical likelihood.
"""
from pathlib import Path
import json
import numpy as np
from .data import SeedCatalog,sha256_file
from .pipeline import read_run
from .runtime import atomic_json


def estimate_chart(mappings,metric_images,*,rank_tolerance=1e-8):
    if len(mappings)!=len(metric_images) or not mappings:raise ValueError('aligned nonempty frame maps required')
    T=np.array([[.001,0,-.9595],[0,.001,-.5395],[0,0,1.]])
    equations=[]
    for H,A in zip(mappings,metric_images):
        H=T@np.asarray(H)@np.linalg.inv(T);A=T@np.asarray(A)
        a=A.reshape(-1);a/=np.linalg.norm(a)
        B=(np.eye(9)-np.outer(a,a))@np.kron(H,np.eye(3))
        B/=max(np.linalg.norm(B),1e-300);equations.append(B)
    design=np.concatenate(equations);col=np.linalg.norm(design,axis=0);col[col==0]=1
    _,singular,V=np.linalg.svd(design/col,full_matrices=False)
    Qprime=(V[-1]/col).reshape(3,3);Q=np.linalg.inv(T)@Qprime
    Q/=np.linalg.norm(Q);rank=int(np.count_nonzero(singular>singular[0]*rank_tolerance))
    invertible=bool(abs(np.linalg.det(Q))/np.linalg.norm(Q)**3>1e-14)
    gap=float(singular[-2]/max(singular[-1],1e-16));r=[]
    for H,A in zip(mappings,metric_images):
        B=np.asarray(H)@Q;A=np.asarray(A);scale=float(np.sum(A*B)/np.sum(A*A))
        r.append(float(np.linalg.norm(B-scale*A)/np.linalg.norm(B)))
    # Noisy approximate GS maps can have numerical rank9; the nullspace gap
    # and replay residual are recorded rather than fabricating an exact null.
    ready=invertible and len(mappings)>1 and gap>=10 and max(r)<.05
    return {'Q_metric_to_component_atlas':Q.tolist() if invertible else None,
      'status':'conditional_common_P_upgrade' if ready else 'weak_conditional_upgrade' if invertible else 'singular_unknown_Q',
      'input_frames':len(mappings),'normalized_design_rank':rank,'singular_values':singular.tolist(),
      'nullspace_gap':gap,'relative_matrix_residual_max':max(r),'relative_matrix_residual_rms':float(np.sqrt(np.mean(np.square(r)))),
      'metric_truth_established':False,'final_likelihood_factor':False}


def upgrade_component_atlases(run_dir):
    root,run=read_run(run_dir);data=Path(run['data_dir']);seeds=SeedCatalog(data)
    with np.load(data/'camera_seeds.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    if not np.array_equal(arrays['frame_indices'],np.arange(1,1249)):raise ValueError('whole native camera clock required')
    P=[]
    for R,C,f,c in zip(arrays['R_world_to_camera'],arrays['center_world'],arrays['focal_px'],arrays['principal_xy']):
        K=np.array([[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1.]])
        P.append(K@np.c_[R,-R@C])
    out=root/'atlas_upgrade';out.mkdir(exist_ok=True);summary={}
    for plane in seeds.planes:
        path=root/'curve_graph'/f'{plane}.json'
        if not path.exists():path=root/'graph'/f'{plane}.json'
        graph=json.loads(path.read_text(encoding='utf-8'));records=[];M=seeds.embedding(plane)
        frames={s['frame_id']:s for s in graph['frame_ledger']}
        for component in graph['components']:
            ids=[i for i in component['frame_ids'] if frames[i]['H_atlas_to_native'] is not None]
            if not ids:record={'status':'unknown_Q_no_original_component_mapping','Q_metric_to_component_atlas':None,'input_frames':0}
            else:record=estimate_chart([frames[i]['H_atlas_to_native'] for i in ids],[P[i-1]@M for i in ids])
            record.update(component_id=component['component_id'],frame_indices=ids,physical_plane_M=M.tolist(),
                provenance='estimated unknown Q using explicit all-time conditional legacy common P; GS/no-distortion approximation only',
                camera_seed_sha256=sha256_file(data/'camera_seeds.npz'),graph_sha256=sha256_file(path),
                Q_direction='q_component=Q xi_metric; H_component Q ~ P M')
            records.append(record)
        atomic_json(out/f'{plane}.json',records);summary[plane]={'components':len(records),
          'conditional_upgrades':sum(r['status']=='conditional_common_P_upgrade' for r in records),
          'weak_upgrades':sum(r['status']=='weak_conditional_upgrade' for r in records),
          'still_unknown':sum(r['Q_metric_to_component_atlas'] is None for r in records),'metric_truth_established':False}
    atomic_json(out/'summary.json',summary);return summary


def common_camera_from_atlases(run_dir):
    from .metric_upgrade import PlaneHomographyInput,common_projection_from_planes
    from .runtime import atomic_npz,json_value
    from scipy.linalg import rq
    root,run=read_run(run_dir);out=root/'atlas_upgrade';seeds=SeedCatalog(run['data_dir'])
    available={i:[] for i in range(1,1249)}
    for plane in seeds.planes:
        graph_path=root/'curve_graph'/f'{plane}.json'
        if not graph_path.exists():graph_path=root/'graph'/f'{plane}.json'
        graph=json.loads(graph_path.read_text(encoding='utf-8'));frames={s['frame_id']:s for s in graph['frame_ledger']}
        for chart in json.loads((out/f'{plane}.json').read_text(encoding='utf-8')):
            Q=chart['Q_metric_to_component_atlas']
            if Q is None:continue
            for i in chart['frame_indices']:
                available[i].append(PlaneHomographyInput(plane,f'frame:{i}',seeds.embedding(plane),
                    np.array(frames[i]['H_atlas_to_native']),np.array(Q),chart['provenance'],True,'conditional_world_seed'))
    records=[];P=[];frames=[];cameras=[]
    for i in range(1,1249):
        result=common_projection_from_planes(available[i]);record=json_value(result)
        for key in ['nullspace_gap','identifiable_subspace_condition']:
            if record[key] is not None and not np.isfinite(record[key]):record[key]='infinity'
        record.update(frame=i,rank_8_plane_count=len(available[i]),old_camera_is_only_conditional_initializer=True)
        if result.P is not None:
            mat=np.array(result.P);A=mat[:,:3]
            K,R=rq(A);sign=np.sign(np.diag(K));sign[sign==0]=1;D=np.diag(sign);K=K@D;R=D@R
            if np.linalg.det(R)<0:mat=-mat;K,R=rq(mat[:,:3]);sign=np.sign(np.diag(K));sign[sign==0]=1;D=np.diag(sign);K=K@D;R=D@R
            K/=K[2,2];C=-np.linalg.solve(mat[:,:3],mat[:,3])
            good=np.isfinite(K).all() and np.isfinite(C).all() and np.linalg.det(R)>0 and K[0,0]>0 and K[1,1]>0
            record['RQ']=None if not good else {'K':K.tolist(),'R':R.tolist(),'C':C.tolist(),'skew_px':float(K[0,1]),'positive_depth_not_yet_certified':True}
            P.append(mat);frames.append(i)
            if good:cameras.append({'frame':i,'K':K.tolist(),'R':R.tolist(),'C':C.tolist(),'status':record['status']})
        records.append(record)
    atomic_json(out/'common_camera_ledger.json',records);atomic_json(out/'camera_candidates.json',cameras)
    atomic_npz(out/'conditional_common_P.npz',{'frame_indices':np.array(frames,dtype='<i4'),'P':np.array(P).reshape(-1,3,4)})
    summary={'full_frames_in_ledger':1248,'common_P_candidates':len(P),'finite_proper_RQ_candidates':len(cameras),
      'initializer_ready':sum(r['status']=='initializer_ready' for r in records),'physical_fit_completed':False,'all_derived_factors_deleted_in_final_problem':True}
    atomic_json(out/'common_camera_summary.json',summary);return summary
