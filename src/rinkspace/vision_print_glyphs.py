"""Source-only white printed glyph contours, holes and one content chart.

Names/order are confirmed inputs. This measures ink interfaces inside a broad
source crop; it is not autonomous advertisement identity or fit admission.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.interpolate import splprep,splev


def transform(points, H):
    p=np.c_[np.asarray(points,float),np.ones(len(points))]@np.asarray(H,float).T
    if np.any(np.abs(p[:,2])<1e-12):raise ValueError('source chart crosses infinity')
    return p[:,:2]/p[:,2:]


def clean_outline(points, *, native_tolerance=.55):
    """Keep sharp corners and supported straight strokes; smooth pixel steps.

    This changes the template representation, not the measured source contour.
    Curves are sampled back to the native core's finite-polyline representation.
    """
    p=np.asarray(points,float)
    if np.linalg.norm(p[0]-p[-1])<1e-9:p=p[:-1]
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]
    coarse=cv2.approxPolyDP(p.astype(np.float32),.75,True)[:,0].astype(float)
    before=coarse-np.roll(coarse,1,axis=0);after=np.roll(coarse,-1,axis=0)-coarse
    lengths=np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)
    angle=np.arccos(np.clip(np.sum(before*after,axis=1)/np.maximum(lengths,1e-9),-1,1))
    corners=coarse[(angle>np.deg2rad(55))&(np.minimum(np.linalg.norm(before,axis=1),np.linalg.norm(after,axis=1))>2.)]
    indices=sorted({int(np.argmin(np.sum((p-c)**2,axis=1))) for c in corners})
    if len(indices)<2:
        q=np.vstack((p,p[0]));tck,_=splprep(q.T,s=len(q)*.045,per=True)
        result=np.column_stack(splev(np.linspace(0,1,512),tck))
        result[-1]=result[0]
        return result
    segments=[]
    for a,b in zip(indices,indices[1:]+indices[:1]):
        q=p[a:b+1] if b>a else np.vstack((p[a:],p[:b+1]))
        centre=q.mean(0);_,_,V=np.linalg.svd(q-centre,full_matrices=False)
        normal=V[-1];residual=(q-centre)@normal
        straight=float(np.quantile(np.abs(residual),.98))<=native_tolerance
        segments.append({'q':q,'normal':normal,'offset':-normal@centre,'straight':straight})
    joints=[]
    for i,segment in enumerate(segments):
        previous=segments[i-1];joint=segment['q'][0]
        if previous['straight'] and segment['straight']:
            lines=np.vstack((previous['normal'],segment['normal']))
            if abs(np.linalg.det(lines))>.08:
                candidate=np.linalg.solve(lines,-np.array([previous['offset'],segment['offset']]))
                if np.linalg.norm(candidate-joint)<1.5:joint=candidate
        joints.append(joint)
    result=[]
    for i,segment in enumerate(segments):
        start,end=joints[i],joints[(i+1)%len(joints)];q=segment['q'].copy();q[0]=start;q[-1]=end
        if segment['straight']:curve=np.linspace(start,end,max(2,int(np.linalg.norm(end-start)*2)))
        else:
            w=np.ones(len(q));w[[0,-1]]=10000
            tck,_=splprep(q.T,w=w,s=len(q)*.045,k=min(3,len(q)-1))
            curve=np.column_stack(splev(np.linspace(0,1,max(24,len(q)//2)),tck));curve[0]=start;curve[-1]=end
        result.extend(curve[:-1])
    result=np.asarray(result);return np.vstack((result,result[0]))


def extract_glyphs(image_bgr, roi, component_order, *, threshold=140.,
                   blur_sigma=.55, supersample=4, simplify_native_px=.3):
    """Return original-space subpixel interface candidates and their topology.

    Interpolation improves contour localization, not source resolution. One
    threshold is used for the entire line; no font/model UV supplies its shape.
    Saturated colored background is excluded by the minimum RGB channel.
    """
    x,y,X,Y=map(int,roi)
    if not (0<=x<X<=image_bgr.shape[1] and 0<=y<Y<=image_bgr.shape[0]):
        raise ValueError('glyph search region outside native image')
    signal=np.min(image_bgr[y:Y,x:X].astype(np.float32),axis=2)
    signal=cv2.GaussianBlur(signal,(0,0),blur_sigma)
    ss=int(supersample)
    enlarged=cv2.resize(signal,None,fx=ss,fy=ss,interpolation=cv2.INTER_CUBIC)
    mask=(enlarged>=threshold).astype(np.uint8)*255
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:raise ValueError('no photographed white glyphs')
    def bbox(c):return cv2.boundingRect(c)
    outer=[i for i,c in enumerate(contours) if hierarchy[0,i,3]<0 and
           cv2.contourArea(c)>=18*ss*ss and bbox(c)[3]>=4*ss and
           bbox(c)[0]>0 and bbox(c)[1]>0 and
           bbox(c)[0]+bbox(c)[2]<mask.shape[1] and bbox(c)[1]+bbox(c)[3]<mask.shape[0]]
    outer.sort(key=lambda i:bbox(contours[i])[0])
    if len(outer)!=len(component_order):
        raise ValueError(f'expected {len(component_order)} visible ink components, got {len(outer)}')
    records={}
    for i,key in zip(outer,component_order):
        children=[j for j,c in enumerate(contours) if hierarchy[0,j,3]==i and cv2.contourArea(c)>3*ss*ss]
        if len(children)>1:raise ValueError('multiple unclassified printed holes: '+key)
        for index,name,parent in [(i,key,None)]+[(j,key+'_inner',key) for j in children]:
            dense=(contours[index][:,0].astype(float)+.5)/ss-.5+[x,y]
            if not np.array_equal(dense[0],dense[-1]):dense=np.vstack((dense,dense[0]))
            simplified=clean_outline(dense)
            records[name]={'id':name,'closed':True,'source_native_uv':dense.tolist(),
                'template_source_uv':simplified.tolist(),'parent_contour_key':parent,
                'role':'letter_hole_boundary' if parent else 'letter_material_boundary'}
    return {'paths':records,'extraction':{'signal':'minimum_native_RGB_channel',
        'threshold':float(threshold),'gaussian_sigma_native_px':float(blur_sigma),
        'supersample_for_localization':ss,'upsampling_adds_source_resolution':False,
        'simplification_tolerance_native_px':float(simplify_native_px),
        'template_representation':'corner_preserving_straight_strokes_and_smoothed_native_edges',
        'template_source_pixels_are_rewritten':False,
        'search_roi_native_xyxy':list(roi),'component_identity_is_given':True,
        'background_pattern_used':False,'model_projection_used':False}}


def panel_chart(features):
    """A rectangular source chart from four actual finite edge readouts.

    The line intersections define this chart only. They are not new observed
    corner points or a declaration that arbitrary boards are planar.
    """
    lines={};evidence={}
    for feature in features:
        key=feature['id'].rsplit('.',1)[-1]
        if key not in ('top_edge','bottom_edge','left_vertical','right_vertical'):continue
        values=[point for segment in feature['segments'] for point in segment.get('source_native_readout_uv',[])]
        p=np.asarray(values,float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<3:raise ValueError('actual source edge support missing: '+key)
        centre=p.mean(0);_,_,V=np.linalg.svd(p-centre,full_matrices=False)
        normal=V[-1];line=np.r_[normal,-normal@centre]
        lines[key]=line;evidence[key]={'native_readout_uv':p.tolist(),
            'line_rms_native_px':float(np.sqrt(np.mean((p@normal+line[2])**2))),
            'raw_feature':feature}
    if len(lines)!=4:raise ValueError('four photographed panel edges required')
    corners=[]
    for a,b in [('top_edge','left_vertical'),('top_edge','right_vertical'),
                ('bottom_edge','right_vertical'),('bottom_edge','left_vertical')]:
        p=np.cross(lines[a],lines[b]);corners.append(p[:2]/p[2])
    H=cv2.getPerspectiveTransform(np.asarray(corners,np.float32),np.asarray([[0,0],[1,0],[1,1],[0,1]],np.float32))
    return {'H':H.tolist(),'edge_evidence':evidence,
        'derived_chart_intersections_native_uv':np.asarray(corners).tolist(),
        'intersections_are_new_source_observations':False,'physical_aspect_verified':False}


def validate_holes(paths, required_parents):
    holes={p['parent_contour_key'] for p in paths.values() if p.get('parent_contour_key')}
    if holes!=set(required_parents):raise ValueError('printed-hole topology differs from visible A/O')
    for key,record in paths.items():
        parent=record.get('parent_contour_key')
        if parent:
            outer=np.asarray(paths[parent]['template_source_uv'],np.float32)
            if any(cv2.pointPolygonTest(outer,tuple(map(float,p)),False)<=0
                   for p in record['template_source_uv']):
                raise ValueError('hole leaves its photographed parent glyph: '+key)
