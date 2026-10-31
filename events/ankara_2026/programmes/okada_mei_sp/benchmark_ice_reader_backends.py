"""Compare CPU/CUDA RGB reads on identical frozen real-source query arrays.

Run through run_native_entry.py. Results are benchmark-only, never fit inputs.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,time
import av,cv2,numpy as np
from rinkspace import vision_programme_ice_review as review
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from capture_independent_materials import source_exclusion
from capture_independent_ice import ice_source_mask


def read(path):
    b=Path(path).read_bytes();return json.loads(gzip.decompress(b) if str(path).endswith('.gz') else b)


def run(args):
    cv2.setNumThreads(2)
    started=time.perf_counter();import torch
    torch.set_num_threads(1)
    from rinkspace.vision_source_cuda_profiles import batched_actual_profiles_cuda
    if not torch.cuda.is_available():raise RuntimeError('No CUDA device')
    torch.ones(1,device='cuda').sum().item();torch.cuda.synchronize()
    startup=time.perf_counter()-started
    binding=read(args.capture/'binding.json');index=read(args.root/'index.json')
    paths=[dict(p,xyz=np.zeros((2,3))) for p in binding['native_inventory']]
    ci_by_name={p['native_path_id']:i for i,p in enumerate(paths)}
    projection={p['native_path_id']:dict(uv=np.zeros((2,2)),valid=np.ones(2,bool)) for p in paths}
    readers={backend:review.IceProgrammeSourceReader(None,source_sha256=binding['source_video_sha256'],
        time_base='1/12800',native_paths=paths,separate_bands=True,profile_backend=backend,
        prepared_source_context=dict(source_sha256=binding['source_video_sha256'],time_base='1/12800',
            authority=[dict(path=str(args.capture/'binding.json'),sha256=hashlib.sha256((args.capture/'binding.json').read_bytes()).hexdigest())])) for backend in ('cpu','cuda')}
    result=[];original_queries=review._source_queries
    try:
        for sid,pts in [('s07',1501184),('s08',1779712),('s09',1961216)]:
            record=read(args.capture/sid/f'{pts}.json.gz');z=dict(np.load(args.capture/sid/f'{pts}.npz'))
            with av.open(index['source']['resolved_path']) as con:
                stream=con.streams.video[0];con.seek(pts,stream=stream,backward=True)
                image=next(f for f in con.decode(stream) if f.pts==pts).to_ndarray(format='bgr24')
            assert hashlib.sha256(image.tobytes()).hexdigest()==record['source_BGR_sha256']
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);mask,_=source_exclusion(args.root,sid,pts);mask,_=ice_source_mask(rgb,mask)
            queries={i:tuple(z[k][z['query_curve']==i].astype(float) for k in ('query_uv','query_normal','query_interval')) for i in range(len(paths))}
            def frozen_queries(path,*unused):return queries[ci_by_name[path['native_path_id']]]
            review._source_queries=frozen_queries
            profile_frame=_MaterialProfileFrame(rgb,dict(source_sha256=binding['source_video_sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid))
            profile_args=(profile_frame,z['query_uv'].astype(float),z['query_normal'].astype(float),z['query_interval'].astype(float),np.asarray([paths[i]['material_signal'] for i in z['query_curve']]))
            times={b:[] for b in readers};sample_times={b:[] for b in readers};outputs={}
            for repeat in range(4):
                for backend in (('cpu','cuda') if repeat%2==0 else ('cuda','cpu')):
                    torch.cuda.synchronize();tick=time.perf_counter()
                    (review._batched_actual_profiles if backend=='cpu' else batched_actual_profiles_cuda)(*profile_args)
                    torch.cuda.synchronize();duration=time.perf_counter()-tick
                    if repeat:sample_times[backend].append(duration)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    outputs[backend]=readers[backend].read(rgb,pts,None,mask,shot_id=sid,native_projections=projection)
                    torch.cuda.synchronize();duration=time.perf_counter()-tick
                    if repeat:times[backend].append(duration)
            comparisons={}
            for key,a in outputs['cpu'].items():
                if not isinstance(a,np.ndarray):continue
                b=outputs['cuda'][key];same_shape=a.shape==b.shape
                exact=same_shape and np.array_equal(a,b,equal_nan=True)
                comparisons[key]=dict(same_shape=same_shape,exact=exact,max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if same_shape and a.size else None)
            item=dict(shot=sid,pts=pts,queries=len(z['query_uv']),source_BGR_sha256=record['source_BGR_sha256'],
                rgb_read_s={b:dict(samples=v,median=float(np.median(v))) for b,v in times.items()},
                sampling_only_s={b:dict(samples=v,median=float(np.median(v))) for b,v in sample_times.items()},
                comparisons=comparisons,candidate_counts={b:len(v['candidate_uv']) for b,v in outputs.items()})
            result.append(item);print(json.dumps(item),flush=True)
    finally:review._source_queries=original_queries
    document=dict(device=torch.cuda.get_device_name(),torch_cuda_startup_s=startup,frames=result,
        query_generation_common_and_excluded=True,identical_frozen_float32_query_arrays_used_for_both=True,
        original_RGB_and_mask_verified=True,includes_image_upload_and_candidate_processing=True,
        concurrent_CPU_capture_running=True,fit_inputs_written=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
