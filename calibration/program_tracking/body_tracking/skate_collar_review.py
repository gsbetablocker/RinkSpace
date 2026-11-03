"""Independent skin/inner-wall sections below the actual open skate collar.

This checks the supplied posed triangle surfaces, not the cuff driver rule.
No hull, cylinder, pose correction or gap-based geometry adjustment is used.
"""
from collections import Counter
from pathlib import Path
import hashlib,json

import numpy as np


def _cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _self_intersects(polygon,tolerance):
    n=len(polygon);i,j=np.triu_indices(n,1)
    use=(j!=i+1)&~((i==0)&(j==n-1));i,j=i[use],j[use]
    if not len(i):return False
    a,b=polygon[i],polygon[(i+1)%n];c,d=polygon[j],polygon[(j+1)%n]
    area_tolerance=tolerance*np.maximum(np.linalg.norm(b-a,axis=1),np.linalg.norm(d-c,axis=1))
    o1,o2=_cross(b-a,c-a),_cross(b-a,d-a)
    o3,o4=_cross(d-c,a-c),_cross(d-c,b-c)
    straddle=((o1<=area_tolerance)&(o2>=-area_tolerance)|(o2<=area_tolerance)&(o1>=-area_tolerance))
    straddle&=((o3<=area_tolerance)&(o4>=-area_tolerance)|(o4<=area_tolerance)&(o3>=-area_tolerance))
    overlap=(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+tolerance).all(axis=1)
    return bool(np.any(straddle&overlap))


def _polygon_is_convex(polygon,tolerance):
    edge=np.roll(polygon,-1,axis=0)-polygon
    turn=_cross(edge,np.roll(edge,-1,axis=0))
    threshold=tolerance*(np.linalg.norm(edge,axis=1)+np.linalg.norm(np.roll(edge,-1,axis=0),axis=1))
    return bool(np.all(turn>=-threshold) and np.any(turn>threshold))


def _section_polygon(vertices,faces,height,tolerance):
    """Slice triangles and require one simple closed graph, without a hull.

    Intersections share their original mesh edge/vertex identity. This avoids
    welding an actually open seam merely because its endpoints look close.
    """
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    triangle=vertices[faces];distance=triangle[:,:,2]-height
    candidates=np.flatnonzero((distance.min(axis=1)<=tolerance)&(distance.max(axis=1)>=-tolerance))
    nodes={};edges=set()
    for index in candidates:
        ids=faces[index];d=distance[index]
        if np.all(abs(d)<=tolerance):return None,'coplanar_triangle_at_section'
        hits={}
        for a in range(3):
            b=(a+1)%3
            if abs(d[a])<=tolerance:hits[('vertex',int(ids[a]))]=triangle[index,a,:2]
            if (d[a]<-tolerance and d[b]>tolerance) or (d[a]>tolerance and d[b]<-tolerance):
                lo,hi=sorted((int(ids[a]),int(ids[b])))
                point=triangle[index,a]+d[a]/(d[a]-d[b])*(triangle[index,b]-triangle[index,a])
                hits[('edge',lo,hi)]=point[:2]
        if len(hits)==1:continue  # tangency contributes no section segment
        if len(hits)!=2:
            if hits:return None,'ambiguous_triangle_section'
            continue
        keys=list(hits)
        if np.linalg.norm(hits[keys[0]]-hits[keys[1]])<=tolerance:continue
        for key,point in hits.items():
            if key not in nodes:nodes[key]=(len(nodes),point)
        a,b=(nodes[key][0] for key in keys);edges.add(tuple(sorted((a,b))))
    if not edges:return None,'no_triangle_intersection'
    adjacency=[set() for _ in nodes]
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    if any(len(row)!=2 for row in adjacency):
        return None,'open_section' if any(len(row)<2 for row in adjacency) else 'branched_section'
    order=[0];previous=-1;current=0
    while True:
        following=next(v for v in adjacency[current] if v!=previous)
        if following==order[0]:break
        if following in order:return None,'branched_section'
        order.append(following);previous,current=current,following
    if len(order)!=len(nodes):return None,'multiple_section_loops'
    if len(order)<3:return None,'degenerate_section'
    points=np.asarray([row[1] for row in nodes.values()])[order]
    if _self_intersects(points,tolerance):return None,'self_intersecting_section'
    area=float(np.sum(_cross(points,np.roll(points,-1,axis=0)))/2)
    if abs(area)<=tolerance*tolerance:return None,'zero_area_section'
    return points if area>0 else points[::-1].copy(),None


def _point_segment_distance(points,polygon):
    a=polygon;b=np.roll(polygon,-1,axis=0);vector=b-a
    fraction=np.clip(np.einsum('pev,ev->pe',points[:,None]-a,vector)/np.sum(vector*vector,axis=1),0,1)
    closest=a[None]+fraction[:,:,None]*vector[None]
    distance=np.linalg.norm(points[:,None]-closest,axis=-1)
    index=np.argmin(distance,axis=1)
    return distance[np.arange(len(points)),index],closest[np.arange(len(points)),index]


def _inside_polygon(points,polygon):
    a=polygon;b=np.roll(polygon,-1,axis=0)
    # Half-open ray crossings retain concavities; the boundary is handled by
    # the caller's actual distance rather than an inflated polygon.
    selected=(a[None,:,1]>points[:,None,1])!=(b[None,:,1]>points[:,None,1])
    denominator=b[:,1]-a[:,1]
    safe=np.where(denominator==0,1.,denominator)
    crossing=(b[:,0]-a[:,0])[None]*(points[:,None,1]-a[None,:,1])/safe[None]+a[None,:,0]
    return (np.count_nonzero(selected&(points[:,None,0]<crossing),axis=1)%2)==1


def _signed_gap(body,inner,tolerance):
    """Exact polygon-boundary gap for a simple body and convex inner loop.

    Outside distance to a convex set is convex along each straight body
    segment, so the worst outside point occurs at a section vertex. For an
    entirely contained body, both directions of edge/vertex distances give
    the exact minimum separation between polygon segments.
    """
    distance,nearest=_point_segment_distance(body,inner)
    inside=_inside_polygon(body,inner)|(distance==0.)
    if not np.all(inside):
        outside=np.where(inside,-np.inf,distance);index=int(np.argmax(outside))
        return -float(outside[index]),body[index],nearest[index]
    reverse,on_body=_point_segment_distance(inner,body)
    first,second=int(np.argmin(distance)),int(np.argmin(reverse))
    if distance[first]<=reverse[second]:return float(distance[first]),body[first],nearest[first]
    return float(reverse[second]),on_body[second],inner[second]


def _signed_gap_general(body,inner,tolerance,error_bound):
    """Bound the minimum signed gap over every actual body section segment.

    Signed distance to a simple polygon is 1-Lipschitz. Two endpoint values
    f0/f1 on an interval of length L therefore give the certified lower bound
    (f0+f1-L)/2. Only intervals that can improve the current minimum beyond
    error_bound are bisected. This also detects a body edge crossing a cavity
    concavity when both body vertices are inside; no hull or point-only claim
    replaces that continuous edge check.
    """
    def values(points):
        distance,nearest=_point_segment_distance(points,inner)
        signed=np.where(_inside_polygon(points,inner),distance,-distance)
        return signed,nearest
    endpoint,nearest=values(body);index=int(np.argmin(endpoint))
    best=float(endpoint[index]);body_point=body[index].copy();wall_point=nearest[index].copy()
    a=body.copy();b=np.roll(body,-1,axis=0).copy()
    fa=endpoint;fb=np.roll(endpoint,-1);certified_lower=np.inf
    while len(a):
        lower=(fa+fb-np.linalg.norm(b-a,axis=1))/2-tolerance
        active=lower<best-error_bound
        if np.any(~active):certified_lower=min(certified_lower,float(lower[~active].min()))
        if not np.any(active):break
        a,b,fa,fb=a[active],b[active],fa[active],fb[active]
        middle=(a+b)/2;fm,nearest=values(middle)
        index=int(np.argmin(fm))
        if fm[index]<best:best=float(fm[index]);body_point=middle[index].copy();wall_point=nearest[index].copy()
        a,b=np.concatenate((a,middle)),np.concatenate((middle,b))
        fa,fb=np.concatenate((fa,fm)),np.concatenate((fm,fb))
    return best,min(best,certified_lower),body_point,wall_point


def review_collar_gap(root,out,ref,info,packet):
    """Review every source exposure and write only ``out/collar_gap.json``.

    Body triangles are skinned from the packet's effective local rotations.
    They are mapped through the same static T_FB into B before slicing. The
    current binding regenerates the inner mesh, so historical render vertex
    counts in the supplied packet do not constrain this check.
    """
    root,out=Path(root),Path(out);binding=ref.binding;surface=binding.surface
    pts=np.asarray(packet.get('source_pts',packet['pts']));n=len(pts)
    np.testing.assert_array_equal(pts,ref.p['pts'])
    ratio=float(info['length_export_ratio'])
    np.testing.assert_allclose(ratio,binding.ratio,rtol=0,atol=1e-12)
    if not np.isfinite(ratio) or ratio<=0:raise ValueError('Invalid conditional metric ratio')
    geometry=binding.packet();count=len(geometry['xs']);collar=np.unique(geometry['collar'])
    if np.any(collar>=count*6):raise ValueError('Collar boundary includes tongue or invalid section vertices')
    inner_faces=np.asarray(binding.faces)[np.asarray(binding.region_ids)!=3]
    tolerance_m=1e-8;tolerance=tolerance_m/ratio;offset_m=.001;gap_error_m=1e-5
    weights=np.asarray(surface.weights);face_array=np.asarray(surface.faces,int)
    selected_faces=[]
    for knee,ankle,toe in ((4,7,10),(5,8,11)):
        # Include every triangle incident to the anatomical skin subset. The
        # single-loop condition detects a slice intersecting this selection's
        # boundary; no artificial cap closes the body section.
        relevant=weights[:,[knee,ankle,toe]].sum(axis=1)>.1
        selected_faces.append(face_array[relevant[face_array].any(axis=1)])
    sample_ids=np.unique(np.concatenate(selected_faces).ravel())
    local_faces=[np.searchsorted(sample_ids,faces) for faces in selected_faces]
    R_FB,t_FB,_=binding.parameters();records=[];warned=False
    for first in range(0,n,32):
        last=min(first+32,n);ids=np.arange(first,last)
        local_R=np.asarray(packet['local_R_mid'])[ids]
        _,joints,_,frames=ref.body.forward_rotations(local_R,with_frames=True)
        samples=surface.sample_vertices(joints,frames,sample_ids,double_precision=True)
        P=np.asarray(packet['P_world_mid'])[ids]/ratio
        Q=np.asarray(packet.get('Q_world_mid',packet.get('Q_world')))[ids]
        evaluated=binding.evaluate(P,Q,joints,frames)
        for foot,(ankle,faces) in enumerate(zip((7,8),local_faces)):
            R_B=frames[:,ankle]@R_FB[foot]
            O_B=joints[:,ankle]+np.einsum('nij,j->ni',frames[:,ankle],t_FB[foot])
            body=np.einsum('nji,nvj->nvi',R_B,samples-O_B[:,None])
            np.testing.assert_allclose(evaluated['boot_R'][:,foot],Q@R_B,rtol=0,atol=1e-10)
            np.testing.assert_allclose(evaluated['boot_O'][:,foot],P+np.einsum('nij,nj->ni',Q,O_B),rtol=0,atol=1e-10)
            for index,frame in enumerate(ids):
                inner=evaluated['inner_local'][index,foot]
                height=float(np.min(inner[collar,2])-offset_m/ratio)
                row=dict(pts=int(pts[frame]),source_seconds=float(pts[frame]/12800.),foot=foot,
                         status='unknown',slice_B_z_m=height*ratio)
                inner_loop,reason=_section_polygon(inner,inner_faces,height,tolerance)
                if reason:row['unknown_reason']='inner:'+reason;records.append(row);continue
                body_loop,reason=_section_polygon(body[index],faces,height,tolerance)
                if reason:row['unknown_reason']='body:'+reason;records.append(row);continue
                convex=_polygon_is_convex(inner_loop,tolerance)
                if convex:
                    gap,body_point,wall_point=_signed_gap(body_loop,inner_loop,tolerance);lower=gap
                else:gap,lower,body_point,wall_point=_signed_gap_general(body_loop,inner_loop,tolerance,gap_error_m/ratio)
                gap*=ratio;lower*=ratio
                row.update(status='measured_geometry',minimum_signed_gap_m=gap,maximum_outside_m=max(0.,-gap),
                           minimum_signed_gap_lower_bound_m=lower,minimum_signed_gap_upper_bound_m=gap,
                           maximum_outside_upper_bound_m=max(0.,-lower),
                           body_section_vertices=len(body_loop),inner_section_vertices=len(inner_loop),
                           inner_section_convex=convex,
                           body_section_convex=_polygon_is_convex(body_loop,tolerance),
                           body_witness_B_m=(np.r_[body_point,height]*ratio).tolist(),
                           inner_witness_B_m=(np.r_[wall_point,height]*ratio).tolist())
                records.append(row)
                if gap<-.003 and not warned:
                    print(json.dumps(dict(event='collar_gap_violation',pts=row['pts'],foot=foot,minimum_signed_gap_m=gap,
                                          note='independent triangle section, supplied pose unchanged')),flush=True)
                    warned=True
    records.sort(key=lambda row:(row['pts'],row['foot']))
    known=[row for row in records if row['status']=='measured_geometry']
    unknown=Counter(row['unknown_reason'] for row in records if row['status']=='unknown')
    worst=min(known,key=lambda row:row['minimum_signed_gap_m']) if known else None
    body_identity=hashlib.sha256()
    for key in ('P_world_mid','local_R_mid'):
        values=np.ascontiguousarray(packet[key]);body_identity.update(key.encode());body_identity.update(values.tobytes())
    body_identity.update(np.ascontiguousarray(packet.get('Q_world_mid',packet.get('Q_world'))).tobytes())
    report=dict(schema='rinkspace.skate-collar-section-review.v1',source_run=str(root),
        source_frames=n,source_first_last_pts=[int(pts[0]),int(pts[-1])],source_time_base=[1,12800],
        evaluated_foot_exposures=len(records),known_foot_exposures=len(known),unknown_foot_exposures=len(records)-len(known),
        unknown_reasons=dict(unknown),worst=worst,
        maximum_outside_m=max((row['maximum_outside_m'] for row in known),default=None),
        maximum_outside_upper_bound_m=max((row['maximum_outside_upper_bound_m'] for row in known),default=None),
        body_state_sha256=body_identity.hexdigest(),shared_parameters_solver_unit=binding.x.tolist(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binding_implementation_sha256=hashlib.sha256(Path(__file__).with_name('skate_binding.py').read_bytes()).hexdigest(),
        surface='actual full SMPL-X posed triangles; generic shared shape, not measured athlete skin',
        selection=dict(joints_by_foot=[[4,7,10],[5,8,11]],vertex_joint_weight_sum_above=.1,
                       incident_triangles_by_foot=[len(faces) for faces in selected_faces],sample_vertices=len(sample_ids)),
        section=dict(axis='B z',height='1mm below minimum co-deformed inner collar-rim vertex; tongue excluded',
                     below_rim_m=offset_m,inner_sections=count,inner_vertices=len(geometry['inner']),
                     graph_tolerance_m=tolerance_m,require_single_closed_simple_loop=True,require_convex_inner=False,
                     body_and_inner_concavities_preserved=True,convex_hull_used=False,
                     nonconvex_gap_method='actual point-in-polygon signed distance; adaptive segment minimum with 1-Lipschitz lower bounds',
                     nonconvex_gap_numerical_bound_m=gap_error_m),
        units='conditional scene metres; body and cavity lengths converted once',
        body_or_upper_modified=False,independent_of_cuff_driver_rule=True,
        pressure_or_material_measurement=False,complete_3d_collision_certification=False,
        limitations='one horizontal section below each posed collar; unknown sections are not passes; no inferred pressure or material tolerance',
        records=records)
    out.mkdir(parents=True,exist_ok=True)
    (out/'collar_gap.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report
