"""Real-image edge measurements for the explicitly selected v0.14 keyframes.

Reuses the local survey/extract_ring_edges.py approach: semantic search bands
only initialize a normal gradient search; saved pixels are actual image edge
maxima, never projected model samples or ellipse centres. The refined ellipse
is an image-only diagnostic/initializer, not a set of point correspondences.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import json, hashlib, sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v014_main')
DEST=OUT/'ice_evidence'
SCENE=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sample(a,xy):
    shape=xy.shape[:-1]
    return cv2.remap(a.astype(np.float32),xy[...,0].astype(np.float32).reshape(-1,1),xy[...,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).reshape(shape)
def parabolic_peaks(score,step):
    j=score.argmax(axis=1);rows=np.arange(len(j));lo=np.maximum(j-1,0);hi=np.minimum(j+1,score.shape[1]-1)
    aa,bb,cc=score[rows,lo],score[rows,j],score[rows,hi]
    den=aa-2*bb+cc
    off=np.divide(.5*(aa-cc),den,out=np.zeros_like(aa),where=abs(den)>1e-6)
    return j,np.clip(off,-.5,.5)*step
def outside_exclusions(points,boxes):
    good=np.ones(points.shape[:-1],bool)
    for x0,y0,x1,y1 in boxes:
        good&=~((points[...,0]>=x0)&(points[...,0]<=x1)&(points[...,1]>=y0)&(points[...,1]<=y1))
    return good

def ellipse_points(ellipse,n=160):
    (cx,cy),(da,db),angle=ellipse
    t=np.linspace(0,2*np.pi,n,endpoint=False);v=np.c_[da/2*np.cos(t),db/2*np.sin(t)]
    a=np.deg2rad(angle);r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    pts=v@r.T+[cx,cy];normal=np.c_[np.cos(t)/(da/2),np.sin(t)/(db/2)]@r.T
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    return pts,normal

def extract_ring(im,spec):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),1.4)
    cx,cy,a,b,angle=spec['seed_ellipse_cx_cy_a_b_angle']
    ell=((float(cx),float(cy)),(float(2*a),float(2*b)),float(angle))
    exclusions=spec.get('exclude_boxes',[])
    iterations=[]
    for radius in ([16.] if spec.get('partial_no_ellipse_refit') else [16.,10.,7.]):
        pts,normal=ellipse_points(ell)
        offsets=np.arange(-radius,radius+.01,.5)
        candidates=pts[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        # Outward brightness rise across a dark/coloured paint outer edge.
        score=sample(gray,candidates+normal[:,None,:]*2)-sample(gray,candidates-normal[:,None,:]*2)
        j,off=parabolic_peaks(score,.5)
        measured=pts+normal*(offsets[j]+off)[:,None]
        strength=score[np.arange(len(j)),j]
        good=(j>1)&(j<len(offsets)-2)&(strength>spec.get('min_gradient_strength',3.))&outside_exclusions(measured,exclusions)
        good&=(measured[:,0]>8)&(measured[:,0]<im.shape[1]-9)&(measured[:,1]>8)&(measured[:,1]<im.shape[0]-9)
        if np.count_nonzero(good)<(6 if spec.get('partial_no_ellipse_refit') else 18): raise ValueError(f'Not enough observed ring edges: {spec["feature_id"]}')
        if spec.get('partial_no_ellipse_refit'):
            iterations.append({'search_radius_px':radius,'retained':int(good.sum()),'no_full_ellipse_fitted_to_partial_arc':True})
            break
        # Fit only to current image edge maxima and reject image-only outliers.
        fitted=cv2.fitEllipse(measured[good].astype(np.float32))
        (fx,fy),(fd1,fd2),fa=fitted
        ar=np.deg2rad(fa);r=np.array([[np.cos(ar),-np.sin(ar)],[np.sin(ar),np.cos(ar)]])
        local=(measured-[fx,fy])@r
        val=(local[:,0]/(fd1/2))**2+(local[:,1]/(fd2/2))**2-1
        grad=2*local/np.array([fd1/2,fd2/2])**2
        distance=val/np.maximum(np.linalg.norm(grad,axis=1),1e-6)
        good&=abs(distance)<max(3.,np.percentile(abs(distance[good]),85)*1.7)
        ell=cv2.fitEllipse(measured[good].astype(np.float32))
        iterations.append({'search_radius_px':radius,'retained':int(good.sum()),'ellipse_cv2':ell})
    # Conservatively quantify the blur from the width of the measured gradient
    # lobe (FWHM/2). This is a noise scale, not a calibrated confidence interval.
    widths_all=[]
    for row,k in zip(score,j):
        half=row[k]/2;lo=k;hi=k
        while lo>0 and row[lo-1]>half:lo-=1
        while hi<len(row)-1 and row[hi+1]>half:hi+=1
        widths_all.append((hi-lo+1)*.5)
    # Independent image QA found weak competing reflections at a few points.
    # A common, projection-free scale-stability rule rejects them without
    # selecting points on camera residuals or adding hand-written coordinates.
    scale_peaks=[]
    offsets_qa=np.arange(-8.,8.01,.5)
    scan=measured[:,None,:]+normal[:,None,:]*offsets_qa[None,:,None]
    for smooth_sigma in [.8,2.4]:
        gg=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),smooth_sigma)
        ss=sample(gg,scan+normal[:,None,:]*2)-sample(gg,scan-normal[:,None,:]*2)
        jj,oo=parabolic_peaks(ss,.5);scale_peaks.append(offsets_qa[jj]+oo)
    scale_disagreement=abs(scale_peaks[0]-scale_peaks[1])
    good&=scale_disagreement<=4.
    widths=np.asarray(widths_all)[good].tolist()
    sigma=max(2.,float(np.median(widths)/2))
    return {'pixels_uv':measured[good].round(4).tolist(),'sigma_px':sigma,'ellipse_cv2_image_only':ell,
       'method':'image_gradient_normal_maxima_with_subpixel_parabolic_refinement; no camera/model used',
       'edge_definition':'outer paint-to-ice brightness transition, unobserved/occluded arcs omitted',
       'seed_is_search_region_only':True,'iterations':iterations,
       'gradient_strength_4px_gray_levels':strength[good].round(3).tolist(),
       'gradient_fwhm_px':widths,'uncertainty_basis':'max(2px, median gradient FWHM/2); shared blur/paint-edge ambiguity remains systematic',
       'accepted_mask_in_last_search':good.tolist(),'search_spec':spec,
       'image_only_scale_stability_max_disagreement_px':4.,'retained_scale_disagreement_px':scale_disagreement[good].round(3).tolist(),
       'pixel_review_revision':'image-only QA correction; no camera residuals used to accept pixels'}

def extract_line(im,spec):
    gray=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.4)
    a,b=np.asarray(spec['search_axis_endpoints'],float);t=(b-a)/np.linalg.norm(b-a);normal=np.array([-t[1],t[0]])
    bases=np.linspace(a,b,max(30,int(np.linalg.norm(b-a)/13)))
    offsets=np.arange(-spec.get('half_band_px',50),spec.get('half_band_px',50)+.1,.5)
    xy=bases[:,None,:]+normal[None,None,:]*offsets[None,:,None]
    profiles=sample(gray,xy)
    # Remove linear illumination variation in the cross-line search profile.
    endpoint=(profiles[:,:5].mean(axis=1)[:,None]*(offsets[-1]-offsets)[None,:]+profiles[:,-5:].mean(axis=1)[:,None]*(offsets-offsets[0])[None,:])/(offsets[-1]-offsets[0])
    dark=endpoint-profiles
    rows=np.arange(len(bases));j=dark.argmax(axis=1);peaks=dark[rows,j]
    measured=[];widths=[];strength=[]
    for i,k in enumerate(j):
        row=dark[i];half=row[k]*.5;lo=k;hi=k
        while lo>0 and row[lo-1]>half:lo-=1
        while hi<len(row)-1 and row[hi+1]>half:hi+=1
        if lo<=1 or hi>=len(row)-2 or peaks[i]<6 or hi-lo<5:continue
        # Two half-height crossings determine paint-band midline; no endpoint
        # correspondence is created at either end of this image search span.
        lower=offsets[lo-1]+.5*(half-row[lo-1])/max(row[lo]-row[lo-1],1e-5)
        upper=offsets[hi]+.5*(row[hi]-half)/max(row[hi]-row[hi+1],1e-5)
        point=bases[i]+normal*((lower+upper)/2)
        if not outside_exclusions(point[None],spec.get('exclude_boxes',[]))[0]:continue
        if point[0]<10 or point[0]>im.shape[1]-11 or point[1]<10 or point[1]>im.shape[0]-11:continue
        measured.append(point);widths.append(upper-lower);strength.append(peaks[i])
    measured=np.asarray(measured)
    if len(measured)<10:raise ValueError('line has too few real samples')
    vx,vy,x,y=cv2.fitLine(measured.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
    dist=(measured-[x,y])@np.array([-vy,vx]);med=np.median(dist);mad=np.median(abs(dist-med))*1.4826
    good=abs(dist-med)<max(4.,3*mad)
    sigma=max(2.,float(mad),float(np.median(widths)*.12))
    return {'pixels_uv':measured[good].round(4).tolist(),'sigma_px':sigma,
       'line_cv2_image_only':[float(vx),float(vy),float(x),float(y)],
       'method':'image cross-profile half-height paired edges, midpoint, Huber image-only outlier rejection',
       'edge_definition':'paint-band centreline; crop/search ends are not physical line endpoints',
       'paint_width_px':np.array(widths)[good].round(3).tolist(),
       'image_line_normal_MAD_px':float(mad),'contrast_gray_levels':np.array(strength)[good].round(3).tolist(),
       'uncertainty_basis':'max(2px, image-line MAD, 12% measured paint band width); blur and true paint-band asymmetry systematic',
       'search_spec':spec}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    specs=json.loads((OUT/'measurement_search_regions.json').read_text(encoding='utf-8'))
    frames=json.loads((OUT/'source_selection/recommended_frames.json').read_text(encoding='utf-8'))
    if isinstance(frames,dict):frames=frames.get('frames',frames.get('recommended_frames'))
    bypts={f['source_pts']:f for f in frames};records=[]
    lock={'scene_sha256':sha(SCENE),'fit_physical_objects':['F3','R3','B1','B2'],
        'check_physical_objects':['F2','R2','F4','R4','ice_board_contact','boards_top'],
        'split':'whole physical object across every selected frame, including every ring and centre mark',
        'selection':'fixed before any camera solver was run; choose using visibility/field of view only',
        'acceptance_basis':'per-object normal-distance error relative to the measured edge blur; no total localization accuracy claim from one curve',
        'assumptions':['fixed v0.14 scene','shared optical centre within this installation','fx=fy native square pixels','principal point at half native dimensions','zero OpenCV distortion initially, unverified'],
        'not_wholly_unseen_clips':True,'history':'original images and identities previously examined; new measured pixels and these physical check groups excluded from this fit'}
    lockpath=OUT/'observation_split_pre_fit.json'
    if not lockpath.exists():save(lockpath,lock)
    for frame_spec in specs:
        f=bypts[frame_spec['source_pts']];im=cv2.imread(f['image_path']);preview=im.copy();fid=f"main_{f['source_pts']}"
        for j,s in enumerate(frame_spec['features']):
            obs=extract_ring(im,s) if s['kind']=='ring' else extract_line(im,s)
            feature=s['feature_id'];use='fit' if feature in lock['fit_physical_objects'] else 'check'
            obs.update({'id':f'{fid}_{feature}_{j}','frame_id':fid,'feature_id':feature,'physical_group_id':feature,
                'type':'curve','target':{'kind':'circle_outline','feature_id':feature} if s['kind']=='ring' else {'kind':'line_segment','feature_id':feature},
                'use':use,'surface_class':'ice_marking','identity_status':'visually_associated_to_user_corrected_scene_feature; not separately_user_confirmed',
                'xyz_status':'fixed_user_PPT_schematic_not_independently_measured','source_frame':f,
                'image_sha256_verified_now':sha(f['image_path'])})
            records.append(obs)
            color=(60,240,70) if use=='fit' else (0,180,255)
            for p in obs['pixels_uv']:cv2.circle(preview,tuple(np.rint(p).astype(int)),2,color,-1,cv2.LINE_AA)
            p=np.median(obs['pixels_uv'],axis=0).astype(int)
            cv2.putText(preview,feature+' '+use,(max(10,p[0]-60),p[1]-25),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
        cv2.putText(preview,f"oxXu1avhrvs | PTS {f['source_pts']} @ 1/15360 | measured pixels",(20,1050),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3)
        cv2.imwrite(str(DEST/f'{fid}_observations.png'),preview)
    save(DEST/'observations.json',records)
    save(DEST/'extraction_provenance.json',{'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':sha(__file__),
        'search_config_sha256':sha(OUT/'measurement_search_regions.json'),'scene_sha256':sha(SCENE),
        'adapted_image_method_from':'outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/survey/extract_ring_edges.py',
        'pixel_positions_not_from_language_model_or_projection':True,'coarse_search_regions_and_semantic_labels_selected_by_visual_review':True})
    print(json.dumps([{'id':o['id'],'n':len(o['pixels_uv']),'sigma':o['sigma_px']} for o in records]))

if __name__=='__main__':main()
