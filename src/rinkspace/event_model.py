"""Source-backed projective rink atlas and explicit curve correspondences.

Conic contours are constraints, never paired ellipse centres or equal-angle
material points. All maps built here remain candidates in an unchecked lens
domain. A reference photograph sets the gauge, not the rink boundary.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import project
from .io import dump_json, load_json, sha256


def ellipse_points(ellipse, n=96):
    cx, cy, rx, ry, angle = ellipse
    a = np.deg2rad(angle)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[rx*np.cos(theta), ry*np.sin(theta)] @ r.T + [cx, cy]


def ellipse_conic(ellipse):
    cx, cy, rx, ry, angle = ellipse
    if min(rx, ry) <= 0:
        raise ValueError('Positive ellipse radii required')
    a = np.deg2rad(angle)
    t = np.array([[rx*np.cos(a), -ry*np.sin(a), cx],
                  [rx*np.sin(a), ry*np.cos(a), cy], [0, 0, 1.]])
    q = np.linalg.inv(t)
    return q.T @ np.diag([1., 1., -1.]) @ q


def conic_distance(xy, conic):
    p = np.c_[xy, np.ones(len(xy))]
    q = p @ conic
    return np.sum(q*p, axis=1) / np.maximum(np.linalg.norm(2*q[:, :2], axis=1), 1e-9)


def fit_curve_map(model_ellipses, image_ellipses):
    """Fit image_from_model by symmetric conic-normal residual, not extrema ties.

    Ellipse centres/extrema only seed the numerical search. The objective uses
    entire conics, with no claim that equal ellipse parameters identify points.
    """
    from scipy.optimize import least_squares
    if len(model_ellipses) != len(image_ellipses) or len(model_ellipses) < 3:
        raise ValueError('At least three identified physical curves are required')
    for ellipses in (model_ellipses,image_ellipses):
        centres=np.asarray([e[:2] for e in ellipses])
        if any(np.linalg.norm(centres[i]-centres[j])<5 for i in range(len(centres)) for j in range(i)):
            raise ValueError('Distinct spatial physical curves required; repeated/concentric groups are degenerate')
    scale = 1000.
    def scaled(e):
        return [*(np.asarray(e[:4])/scale), e[4]]
    ms = [scaled(e) for e in model_ellipses]
    ts = [scaled(e) for e in image_ellipses]
    mp, tp = [ellipse_points(e, 48) for e in ms], [ellipse_points(e, 48) for e in ts]
    mc, tc = [ellipse_conic(e) for e in ms], [ellipse_conic(e) for e in ts]
    h0, _ = cv2.findHomography(np.concatenate(mp), np.concatenate(tp), 0)
    if h0 is None:
        raise ValueError('Curve initialisation is degenerate')
    def unpack(p):
        return np.r_[p, 1.].reshape(3, 3)
    def residual(p):
        h = unpack(p)
        try:
            hi = np.linalg.inv(h)
            return np.concatenate([conic_distance(project(h, a), c) for a,c in zip(mp,tc)] +
                                  [conic_distance(project(hi, b), c) for b,c in zip(tp,mc)]) * scale
        except (ValueError, np.linalg.LinAlgError):
            return np.full(len(ms)*96, 1e5)
    fits=[]
    for reverse in (False,True):
        for shift in (0,12,24,36):
            seed=[np.roll(b[::-1] if reverse else b,shift,axis=0) for b in tp]
            initial,_=cv2.findHomography(np.concatenate(mp),np.concatenate(seed),0)
            if initial is None:continue
            f=least_squares(residual,(initial/initial[2,2]).ravel()[:8],loss='soft_l1',f_scale=2.,
                            max_nfev=2500,x_scale='jac',ftol=1e-6,xtol=1e-6,gtol=1e-6)
            try:
                # Reject a fold or a projective horizon through observed curves.
                hp=unpack(f.x)
                poly=cv2.convexHull(np.concatenate(mp).astype(np.float32)).reshape(-1,2)
                den=np.c_[poly,np.ones(len(poly))]@hp[2,:]
                warped=project(hp,poly)
                if den.min()*den.max()<=0 or cv2.contourArea(warped.astype(np.float32),oriented=True)*cv2.contourArea(poly.astype(np.float32),oriented=True)<=0:continue
                fits.append(f)
            except (ValueError,np.linalg.LinAlgError):continue
    if not fits:raise ValueError('All curve-map hypotheses fold the support or cross a horizon')
    fit=min(fits,key=lambda f:np.mean(np.minimum(np.abs(residual(f.x)),30)))
    h = np.diag([scale,scale,1]) @ unpack(fit.x) @ np.diag([1/scale,1/scale,1])
    h /= h[2,2]
    e = np.abs(residual(fit.x))
    if np.linalg.cond(fit.jac)>1e8:
        raise ValueError('Curve geometry is numerically underconstrained')
    return h, {'curve_normal_fit_median_sample_px': float(np.median(e)),
               'curve_normal_fit_p95_sample_px': float(np.percentile(e,95)),
               'physical_group_count':len(ms), 'independent_validation':False,
               'ellipse_extrema_used_as_correspondences':False,
               'solver_rank':int(np.linalg.matrix_rank(fit.jac)),
               'solver_condition':float(np.linalg.cond(fit.jac)),
               'solver_converged':bool(fit.success),'solver_termination':str(fit.message),
               'usable_as_validated_geometry':False,
               'annotation_uncertainty_sample_px':4.,
               'status':'candidate'}


def polygon_mask(shape, polygon, boxes=()):
    m = np.zeros(shape[:2], np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(m, [np.rint(polygon).astype(np.int32)], 255)
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(m, (round(x1),round(y1)), (round(x2),round(y2)), 0, -1)
    return m


def build_event_model(root, config_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    spec = load_json(config_path)
    output.mkdir(parents=True, exist_ok=True)
    (output/'views').mkdir(exist_ok=True)
    model = {k:v for k,v in spec.items() if k != 'view_specs'}
    model.update(schema_version='rinkspace.event-model.v1', status='candidate',
                 geometry_level='projective',unit='reference_px',
                 length_m=None,width_m=None,corner_radius_m=None,
                 lens_model_id='uncalibrated', image_domain='raw_px_distortion_unchecked',
                 reference_views=[],registration_graph=[],landmarks=[],
                 coverage_regions=[],geometry_evidence=[],
                 overview_path='model_overview.jpg')
    gauge = spec['view_specs'][0]
    by_id = {c['physical_id']:c for c in gauge['curves']}
    for v in spec['view_specs']:
        path=root/v['image']
        im=cv2.imread(str(path))
        if im is None: raise FileNotFoundError(path)
        im=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
        dest=output/'views'/f"{v['id']}.jpg"
        cv2.imwrite(str(dest),im,[cv2.IMWRITE_JPEG_QUALITY,93])
        view={k:val for k,val in v.items() if k not in ('image','curves')}
        view.update(image_path=f"views/{v['id']}.jpg",image_width=960,image_height=540,
                    source_image_sha256=sha256(path), sample_scale_from_image=960/v['source_width'],
                    physical_ids=[c['physical_id'] for c in v.get('curves',[])],
                    observations=v.get('curves',[]),
                    annotation_method='assistant_visual_sparse_curve_selection_not_user_confirmed',
                    image_domain='raw_px_distortion_unchecked', lens_model_id='uncalibrated')
        if v['id']==gauge['id']:
            h=np.eye(3);quality={'status':'gauge_only_not_validation'}
        elif v.get('disconnected'):
            h=None;quality={'status':'unconnected_evidence_no_global_grid'}
        else:
            common=[c for c in v['curves'] if c['physical_id'] in by_id]
            q,quality=fit_curve_map([by_id[c['physical_id']]['ellipse'] for c in common],
                                    [c['ellipse'] for c in common])
            h=np.linalg.inv(q);h/=h[2,2]
            model['registration_graph'].append({'from_view':v['id'],'to_view':gauge['id'],
                'physical_ids':[c['physical_id'] for c in common], 'method':'three_identified_conics',
                'H_model_from_sample':h.tolist(),'quality':quality})
        view['H_model_from_sample']=None if h is None else h.tolist()
        view['H_model_from_image']=None if h is None else (h@np.diag([960/v['source_width'],540/v['source_height'],1])).tolist()
        view['quality']=quality
        view['graph_component_id']='main' if h is not None else v['id']
        model['reference_views'].append(view)
        for c in v.get('curves',[]):
            found=next((p for p in model['landmarks'] if p['physical_id']==c['physical_id']),None)
            if found is None:
                found={'physical_id':c['physical_id'],'label':c.get('label',c['physical_id']),
                       'type':'painted_ring_outer_curve','status':'candidate_identity',
                       'curve_model':ellipse_points(by_id[c['physical_id']]['ellipse']).tolist() if c['physical_id'] in by_id else None,
                       'observations':[]}
                model['landmarks'].append(found)
            found['observations'].append({'view_id':v['id'],'source_id':v['source_id'],
                'source_sha256':v['source_sha256'],'pts':v['pts'],'time_base':v['time_base'],
                'source_time_s':v['source_time_s'],'image_path':view['image_path'],
                'ellipse_sample_px':c['ellipse'],'physical_id':c['physical_id'],
                'method':'assistant_sparse_curve_selection','uncertainty_sample_px':4})
        if h is not None:
            polygon=project(h,v['ice_polygon_sample'])
            curves=[ellipse_points(c['ellipse']) for c in v.get('curves',[])]
            support=project(h,v['geometric_support_polygon_sample']) if v.get('geometric_support_polygon_sample') else (project(h,cv2.convexHull(np.concatenate(curves).astype(np.float32)).reshape(-1,2)) if curves else np.empty((0,2)))
            model['coverage_regions'].append({'id':v['id'],'status':'direct_image_candidate_geometry',
                'polygon_model':support.tolist(),'texture_polygon_model':polygon.tolist(),'source_view_ids':[v['id']],
                'texture_observed':True,'independent_localization_validated':False})
            model['geometry_evidence'].append(v['id'])
    # The model includes nonconnected views as evidence, without fabricating H.
    dump_json(output/'rink_model.json',model)
    render_model_overview(output,model)
    return model


def render_model_overview(output,model):
    output=Path(output)
    gauge=model['reference_views'][0]
    im=Image.open(output/gauge['image_path']).convert('RGB')
    d=ImageDraw.Draw(im)
    for p in model['landmarks']:
        if not p.get('curve_model'):continue
        xy=p['curve_model'];d.line([tuple(x) for x in xy+[xy[0]]],fill='#ffbe55',width=2)
        x,y=np.mean(xy,axis=0);d.text((x,y),p['physical_id'],fill='#101923',stroke_width=2,stroke_fill='#ffd06c')
    d.rectangle((0,0,960,32),fill='#101923')
    d.text((12,10),'PROJECTIVE / CANDIDATE / photographed evidence; no metre scale',fill='white')
    im.save(output/'model_overview.jpg',quality=92)


def load_model(path):
    path=Path(path)
    m=load_json(path)
    if m.get('geometry_level')!='projective' or m.get('unit')!='reference_px':
        raise ValueError('This atlas backend only supports explicit projective geometry')
    if not m.get('reference_views'):raise ValueError('No model anchors')
    return m
