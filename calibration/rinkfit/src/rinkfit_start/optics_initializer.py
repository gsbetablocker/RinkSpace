"""Conditional four-parameter optics initialization from immutable old rays.

Derived old camera projections are used only to initialize the existing shared
physical model. They never enter its final objective or replace source UV/W/E.
"""
from pathlib import Path
import copy,json,numpy as np
from .data import sha256_file
from .native_scene import camera_from_event
from .runtime import atomic_json


def _distort(xy,k):
    x,y=xy.T;k1,k2,p1,p2=k.T;q=x*x+y*y;L=1+k1*q+k2*q*q
    return np.c_[x*L+2*p1*x*y+p2*(q+2*x*x),y*L+p1*(q+2*y*y)+2*p2*x*y]


def inverse_old_source_rays(uv,focal,principal,brown):
    target=(uv-principal)/focal;xy=target.copy()
    for iteration in range(40):
        error=_distort(xy,brown)-target
        if np.max(np.linalg.norm(error*focal,axis=1))<1e-7:break
        x,y=xy.T;k1,k2,p1,p2=brown.T;q=x*x+y*y;L=1+k1*q+k2*q*q;A=k1+2*k2*q
        a=L+2*x*x*A+2*p1*y+6*p2*x;b=2*x*y*A+2*p1*x+2*p2*y;d=L+2*y*y*A+6*p1*y+2*p2*x;det=a*d-b*b
        if np.any(det<=0)or not np.isfinite(det).all():raise ValueError('Old source ray inverse has uncertified/singular local Jacobian')
        delta=np.c_[(d*error[:,0]-b*error[:,1])/det,(-b*error[:,0]+a*error[:,1])/det]
        cost=np.sum(error*error,axis=1);step=np.ones(len(xy));candidate=xy-delta
        for backtrack in range(20):
            bad=np.sum((_distort(candidate,brown)-target)**2,axis=1)>cost+1e-30
            if not bad.any():break
            step[bad]*=.5;candidate[bad]=xy[bad]-step[bad,None]*delta[bad]
        xy=candidate
    native_error=np.linalg.norm((_distort(xy,brown)-target)*focal,axis=1)
    if not np.isfinite(xy).all()or native_error.max()>1e-6:raise ValueError('Full original source-ray inverse did not converge')
    return xy,native_error,iteration+1


def initialize_pixel_optics(packet_dir,base_model_path,output_dir):
    root=Path(packet_dir);out=Path(output_dir)
    if out.exists()and any(out.iterdir()):raise ValueError('Never overwrite an optics candidate')
    out.mkdir(parents=True,exist_ok=True);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    model=json.loads(Path(base_model_path).read_text(encoding='utf-8'));n=int(manifest['observations'])
    uv=np.fromfile(root/'uv.f64','<f8').reshape(n,2);frame=np.fromfile(root/'frame.i32','<i4');old=np.load(root/'camera_seeds.npz')
    if n!=182017 or len(np.unique(frame))!=1248 or not np.array_equal(old['frame_indices'],np.arange(1,1249)):raise ValueError('Full exact shot required')
    if sha256_file(root/'camera_seeds.npz')!=model['source_hashes']['camera_seeds.npz']:raise ValueError('Old camera source seed SHA mismatch')
    focal=old['focal_px'][frame-1];principal=old['principal_xy'][frame-1];brown=old['brown_k1_k2_p1_p2'][frame-1]
    xy,inverse_error,iterations=inverse_old_source_rays(uv,focal,principal,brown)
    q=np.sum(xy*xy,axis=1);z=np.log(np.sqrt(focal[:,0]*focal[:,1])/model['camera']['f_ref'])
    # The actual native sensitivity is f*x*r² (k1) and f*x*r⁴ (k2),
    # and likewise in v. Narrow views contribute much less r^5 sensitivity.
    base1=focal*xy*q[:,None];base2=base1*q[:,None]
    design=np.stack([base1,base1*z[:,None],base2,base2*z[:,None]],axis=-1).reshape(2*n,4)
    target=(base1*brown[:,0,None]+base2*brown[:,1,None]).reshape(-1)
    coefficients,residuals,rank,singular=np.linalg.lstsq(design,target,rcond=None)
    if rank!=4:raise ValueError('Four-parameter native pixel sensitivity design is rank deficient')
    original=np.array([model['blocks'][0]['values'][j]for j in [25,26,27,28]])
    before=(design@original-target).reshape(n,2);after=(design@coefficients-target).reshape(n,2)
    candidate=copy.deepcopy(model);candidate['model_id']='shared_superellipse_fixed_head_legacy_pixel_rays032'
    for index,value in zip([25,26,27,28],coefficients):
        if not(candidate['blocks'][0]['lower'][index]<=value<=candidate['blocks'][0]['upper'][index]):raise ValueError('Pixel-derived initializer outside existing parameter bounds')
        candidate['blocks'][0]['values'][index]=float(value)
    camera=camera_from_event(candidate);certificate=camera.optical_domain_certificate()
    if not certificate['passed']:raise ValueError('Pixel-derived initializer does not pass full frozen disk/zoom Brown certificate')
    def metrics(e):
        r=np.linalg.norm(e,axis=1)
        return dict(rows=len(r),median=float(np.median(r)),p95=float(np.quantile(r,.95)),max=float(r.max()),rms=float(np.sqrt(np.mean(r*r))))
    report=dict(schema='rinkfit.conditional_native_pixel_optics_seed.v1',source_rows=n,frames=1248,source_identity=model['source_identity'],base_model=dict(path=str(base_model_path),sha256=sha256_file(base_model_path)),
      camera_seed_sha256=sha256_file(root/'camera_seeds.npz'),UV_sha256=sha256_file(root/'uv.f64'),inverse_original_ray_max_native_error_px=float(inverse_error.max()),inverse_iterations=iterations,source_ray_radius_range=[float(np.sqrt(q).min()),float(np.sqrt(q).max())],
      native_pixel_design_rank=int(rank),native_pixel_design_singular_values=singular.tolist(),native_pixel_design_condition=float(singular[0]/singular[-1]),coefficient_order=['k1','k1_z','k2','k2_z'],before_coefficients=original.tolist(),after_coefficients=coefficients.tolist(),
      conversion_reprojection_error_before=metrics(before),conversion_reprojection_error_after=metrics(after),all_time_zoom_full_disk_certificate=certificate,
      per_frame=[dict(frame=f,rows=int(np.count_nonzero(frame==f)),before=metrics(before[frame==f]),after=metrics(after[frame==f]))for f in range(1,1249)],
      purpose='Conditional low-dimensional initialization only; derived old projection is absent from final physical objective; actual source UV/W/E and all four radial degrees of freedom unchanged.',physical_fit_success=False)
    candidate['provenance']['optical_candidate']='legacy_source_ray_native_pixel_weighted_four_radial'
    candidate['provenance']['pixel_optics_initialization']=dict(source_camera_sha256=report['camera_seed_sha256'],source_UV_sha256=report['UV_sha256'],source_rows=n,frames=1248,design='full native f*x*r^2 and f*x*r^4 sensitivities, unweighted pixel conversion LS',derived_old_projection_used_only_for_initialization=True,additional_final_observation_or_factor=False,report=str(out/'optics_initialization.json'))
    atomic_json(out/'model.json',candidate);report['model_sha256']=sha256_file(out/'model.json');atomic_json(out/'optics_initialization.json',report)
    np.savez_compressed(out/'original_source_ray_conversion.npz',source_rows=np.arange(n),source_ray_xy=xy,inverse_error_px=inverse_error,conversion_pixel_error_before=before,conversion_pixel_error_after=after)
    return report
