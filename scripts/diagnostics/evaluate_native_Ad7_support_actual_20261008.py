"""Actual native Ad7 edge/support replay; no full graph or source mutation."""
from pathlib import Path
import argparse, hashlib, json, os, sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from z190_source_camera_diagnostic import absolute_value_jac15

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
    v=np.asarray(v,float)
    return dict(median_px=float(np.median(v)),p95_px=float(np.quantile(v,.95)),maximum_px=float(v.max()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state-dir',type=Path,action='append',required=True)
    for k in ('packet','auxiliary','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();results=[]
    for directory in a.state_dir:
        identity=read(directory/'state_identity.json')
        assert sha(identity['camera_path'])==identity['camera_sha256']
        with np.load(directory/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
            arrays={k:z[k].copy()for k in z.files if k=='q' or k.startswith('aux__')}
        with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
        provider,_,proof=current_complete_source_geometry(directory/'complete_state_UNACCEPTED.npz',a.auxiliary)
        state,camera_error=state_camera_adapter(arrays,cameras,provider)
        q=state.evaluation_q(arrays['q']).copy()
        factor=NativeBoardEdgeFactor(state,a.packet,len(q))
        archived_input=read(directory.parents[1]/'input_manifest.json')
        contract=archived_input['additional_real_native_board_edges']
        assert contract['packet_sha256']==sha(a.packet)==factor.contract['packet_sha256']
        actual_cameras=factor._cameras(q);rows=[]
        for row in factor.rows+factor.held:
            if row['curve'].get('advertisement_number')!=7:continue
            camera=actual_cameras[row['actual_pts']]
            derived=factor._source_rows(q,row,camera)
            raw_rows=[{k:v for k,v in d.items()if k!='normal_interval'}for d in derived]
            raw=np.asarray(absolute_value_jac15(camera,raw_rows,False),float)
            supported=np.asarray(absolute_value_jac15(camera,derived,False),float)
            sig=np.repeat(row['sigma'],2)
            raw_px=(raw*sig).reshape(-1,2);outside_px=(supported*sig).reshape(-1,2)
            normals=np.asarray(row['normal'],float)
            normals=normals/np.linalg.norm(normals,axis=1)[:,None]
            signed=np.einsum('ni,ni->n',raw_px,normals)
            bounds=np.asarray(row['normal_interval'],float)
            branches=np.where(signed<=bounds[:,0],-1,np.where(signed>=bounds[:,1],1,0))
            world=factor._world(q,row['curve'])
            rows.append(dict(id=row['id'],split=row['split'],native_pts=int(row['pts']),actual_pts=int(row['actual_pts']),
                camera_block=state.bypts[row['actual_pts']].name,UV=np.asarray(row['uv']).tolist(),sigma_px=row['sigma'].tolist(),
                source_normal=normals.tolist(),source_normal_interval_px=bounds.tolist(),raw_residual_vector_px=raw_px.tolist(),
                raw_signed_normal_px=signed.tolist(),nominal_support_branch=branches.tolist(),raw_center_error=stats(np.linalg.norm(raw_px,axis=1)),
                outside_source_support_error=stats(np.linalg.norm(outside_px,axis=1)),outside_source_support_vector_px=outside_px.tolist(),
                normalized_supported_rms=float(np.sqrt(np.mean(supported*supported))),curve=row['curve'],
                finite_world_edge_xyz=np.asarray(world).tolist(),finite_world_point_count=len(world),
                source_raw_frame=row.get('source_raw_frame'),source_frame_sha256=row.get('source_file_sha256',row.get('source_native_RGB_sha256')),
                actual_operator='NativeBoardEdgeFactor._source_rows -> absolute_value_jac15 with unchanged SourceNormalIntervals',
                branch_semantics='Same complete physical sampled Ad7 material edge; native finite-curve closest support. No alternative print edge, missing edge or source inverse projection.',
                source_UV_sigma_PTS_support_split_changed=False))
        results.append(dict(state=identity,camera_archive_reproduction_max=camera_error,actual_factor_contract=factor.contract,rows=rows))
        print(json.dumps({'state':identity['q_sha256'],'rows':[dict(id=r['id'],pts=r['actual_pts'],split=r['split'],raw_p95=r['raw_center_error']['p95_px'],outside_p95=r['outside_source_support_error']['p95_px'],bounds=r['source_normal_interval_px'])for r in rows]}),flush=True)
    report=dict(schema='rinkspace.actual-native-ad7-edge-support-audit.v1',packet_path=str(a.packet),packet_sha256=sha(a.packet),
        states=results,whole_graph_or_pickle_loaded=False,source_rows_modified=False,HELD_promoted=False,new_source_rows=0,accuracy_accepted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'report':str(a.output),'sha256':sha(a.output)}),flush=True)
if __name__=='__main__':main()
