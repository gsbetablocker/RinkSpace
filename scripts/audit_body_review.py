"""Offline integrity and saved-observation metric audit; no fitter or network.

Run after extraction: python AUDIT.py --metrics
Run against a ZIP: python AUDIT.py --zip review.zip --metrics
The default hash check uses only the Python standard library. --metrics needs
NumPy; --pixels additionally needs PyAV and checks original decoded pixels.
"""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile


class Store:
    def __init__(self,root=None,archive=None):
        self.root=Path(root or Path(__file__).resolve().parent);self.archive=zipfile.ZipFile(archive) if archive else None
    def read(self,name):
        if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe member path')
        return self.archive.read(name) if self.archive else (self.root/name).read_bytes()
    def json(self,name):return json.loads(self.read(name).decode('utf8'))
    def arrays(self,name):
        import numpy as np
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stats(values):
    import numpy as np
    a=np.asarray(values,float).ravel()
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if len(a) else dict(n=0)


def resample(points,n=24):
    import numpy as np
    p=np.asarray(points,float);d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t=np.linspace(0,d[-1],n)
    return np.stack([np.interp(t,d,p[:,i]) for i in range(2)],axis=1)


def distance(points,line):
    import numpy as np
    a=np.asarray(line,float)[:-1];v=np.diff(np.asarray(line,float),axis=0);p=np.asarray(points,float)
    q=p[:,None]-a[None];t=np.clip(np.sum(q*v[None],axis=2)/np.maximum(np.sum(v*v,axis=1),1e-16)[None],0,1)
    return np.sqrt(np.min(np.sum((p[:,None]-(a[None]+t[:,:,None]*v[None]))**2,axis=2),axis=1))


def metrics(store,index):
    import numpy as np
    p=store.arrays(index['body_input']);comparison=store.json(index['body_comparison']);visible=~p['lower_body_inferred']
    np.testing.assert_array_equal(p['pts'],np.arange(index['source']['first_pts'],index['source']['last_pts']+index['source']['pts_step'],index['source']['pts_step']))
    selected=np.zeros_like(p['source_qualified']);ids=index['body_source_indices'];selected[:,ids]=p['source_qualified'][:,ids];selected&=visible[:,None]
    body={};zbody={}
    for mode,member in index['body_results'].items():
        z=store.arrays(member);zbody[mode]=z;np.testing.assert_array_equal(p['pts'],z['pts'])
        body[mode]=stats(np.linalg.norm(z['COCO_uv']-p['source_uv'],axis=2)[selected]);expected=comparison['models'][mode]['source_body_reprojection_px']
        for key in ['median','p95','maximum']:np.testing.assert_allclose(body[mode][key],expected[key],rtol=1e-10,atol=1e-9)
    binding=store.json(index['footwear_binding']);fitrows=store.json(index['footwear_fit_review'])['blade_curves'];checkrows=store.json(index['footwear_check_review'])['blade_curves']
    final=store.arrays(index['footwear_final_result']);blind=store.arrays(index['footwear_shared_fit_result']);parameters=store.arrays(index['footwear_parameters'])
    footwear={}
    for name,curves in [('previous_B',zbody['pose']['blade_uv']),('calibrated_blind_pose',blind['equipment_uv'][:,:,14:31]),('calibrated_observed_pose',final['equipment_uv'][:,:,14:31])]:
        footwear[name]={}
        for key,rows in [('fitted_blade_curves',fitrows),('development_blade_curves',checkrows)]:
            values=[]
            for row in rows:
                i=int(np.searchsorted(p['pts'],row['pts']));values.extend(distance(resample(row['uv']),curves[i,row['foot']]))
            result=stats(values);footwear[name][key]=result;expected=binding['models'][name][key]['raw_pixel_distance']
            for metric in ['median','p95','maximum']:np.testing.assert_allclose(result[metric],expected[metric],rtol=1e-10,atol=1e-9)
    np.testing.assert_array_equal(parameters['P_world_m'],zbody['pose']['P_world_m']);np.testing.assert_array_equal(parameters['Q_world'],zbody['pose']['Q_world'])
    np.testing.assert_array_equal(final['x'][:14],blind['x'][:14]);np.testing.assert_array_equal(parameters['G_world_m'][:,:2],parameters['P_world_m'][:,:2])
    assert not parameters['G_world_m'][:,2].any();assert not parameters['contact_allowed_for_root'].any()
    assert len(p['pts'])==index['source']['target_exposures'] and int(p['lower_body_inferred'].sum())==index['source']['lower_body_inferred_exposures']
    return dict(body_visible_source_pixel_errors=body,footwear_curve_pixel_errors=footwear,source_exposures=len(p['pts']),lower_body_inferred=int(p['lower_body_inferred'].sum()),
        final_pelvis_equals_B=True,footwear_geometry_frozen_during_pose_inference=True,contact_feedback_disabled=True,external_3d_accuracy_verified=False)


def pixels(store,index):
    import av
    receipt=store.json(index['source_clip_receipt']);raw=store.read(index['source_clip']);packet_rows=[];frames={};crop_rows=store.json(index['source_crop_catalog']);crop_by_pts={r['pts']:r for r in crop_rows};crop_checked=0
    with av.open(io.BytesIO(raw)) as c:
        s=c.streams.video[0]
        assert str(s.time_base)==receipt['time_base']
        for p in c.demux(s):
            if p.pts is not None:packet_rows.append(dict(pts=p.pts,dts=p.dts,bytes=p.size,sha256=hashlib.sha256(bytes(p)).hexdigest(),keyframe=p.is_keyframe))
    assert packet_rows==receipt['packet_identity']
    with av.open(io.BytesIO(raw)) as c:
        s=c.streams.video[0];s.codec_context.thread_count=2
        for frame in c.decode(s):
            if receipt['target_first_pts']<=frame.pts<=receipt['target_last_pts_inclusive']:
                image=frame.to_ndarray(format='bgr24');frames[str(frame.pts)]=hashlib.sha256(image.tobytes()).hexdigest()
                if frame.pts in crop_by_pts:
                    r=crop_by_pts[frame.pts];x0,y0,x1,y1=r['crop_xyxy'];cropped=image[y0:y1,x0:x1];assert hashlib.sha256(cropped[:,:,::-1].tobytes()).hexdigest()==r['decoded_native_crop_rgb_sha256'];crop_checked+=1
    assert frames==receipt['pixel_hashes_by_pts'];assert crop_checked==len(crop_rows)
    return dict(packet_identity_matches=True,target_decoded_exposures=len(frames),all_source_crop_RGB_hashes_match=True,source_crops_checked=crop_checked,all_target_decoded_bgr_pixels_match_recorded_original=True)


def audit(store,with_metrics=False,with_pixels=False):
    manifest=store.json('MANIFEST.json');bad=[]
    names=[r['path'] for r in manifest['files']]
    if len(names)!=len(set(names)):raise ValueError('Duplicate manifest member')
    for row in manifest['files']:
        try:
            data=store.read(row['path'])
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:bad.append(row['path'])
        except Exception as error:bad.append(dict(path=row['path'],error=str(error)))
    if bad:raise ValueError(json.dumps(bad,ensure_ascii=False))
    index=store.json('REVIEW_INDEX.json');required=[]
    for key in ['body_input','body_comparison','footwear_binding','footwear_fit_review','footwear_check_review','footwear_parameters','footwear_final_result','footwear_shared_fit_result','source_clip','source_clip_receipt']:
        store.read(index[key]);required.append(index[key])
    for stage in index['stages']:
        for key in ['config','report','result','events']:
            if stage.get(key):store.read(stage[key])
    result=dict(schema='rinkspace.offline-body-review-audit.v1',verified_files=len(names),all_file_hashes_and_sizes_match=True,all_stage_index_links_resolve=True,
        manifest_self_hash_is_external=True,no_network_or_author_data_paths_used=True)
    if with_metrics:result['metrics']=metrics(store,index)
    if with_pixels:result['pixels']=pixels(store,index)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--zip',type=Path);ap.add_argument('--metrics',action='store_true');ap.add_argument('--pixels',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    try:result=audit(Store(args.root,args.zip),args.metrics,args.pixels)
    except Exception as error:
        print(json.dumps(dict(ok=False,error=str(error)),ensure_ascii=False),file=sys.stderr);raise SystemExit(1)
    text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if args.output:args.output.write_text(text,encoding='utf8')
