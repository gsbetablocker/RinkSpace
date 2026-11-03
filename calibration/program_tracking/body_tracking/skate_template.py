"""Metric, brand-independent lower-skate reference geometry.

Coordinate B: x heel-to-toe, y local left, z up; metres. x=0 is the
outsole rear, and the lowest unrolled runner point has z=0. This is NOT an
anatomical ankle frame. Existing body/footwear registration supplies T_FB.

The exact contact arcs are independent of the polygonal display mesh.
Factory primary-rocker examples support 7/8 ft candidates; every other
default here is an explicit engineering starting point, not a product scan
or an athlete measurement. No upper/collar/inner-foot model is fabricated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

FT = .3048
SCHEMA = "rinkspace.metric-lower-skate.v1"


@dataclass(frozen=True)
class SkateParameters:
    sole_length_m: float = .250
    sole_width_m: float = .095
    sole_thickness_m: float = .007
    heel_block_height_m: float = .025
    front_mount_height_m: float = .055
    rear_mount_height_m: float = .055
    plate_thickness_m: float = .002
    blade_width_m: float = .0038
    blade_web_height_m: float = .012
    hollow_radius_m: float = .0127
    main_rocker_m: float = 7 * FT
    secondary_rocker_m: float = .330
    rocker_center_fraction: float = .40
    rocker_join_fraction: float = .70
    runner_rear_extension_m: float = .015
    runner_front_inset_m: float = .005
    toe_head_length_m: float = .024
    render_step_m: float = .004

    def validate(self):
        a = asdict(self)
        if not all(np.isfinite(v) for v in a.values()):
            raise ValueError("All skate parameters must be finite")
        if not .12 <= self.sole_length_m <= .36:
            raise ValueError("sole_length_m outside broad 0.12..0.36 m input guard; check units")
        if not .15*self.sole_length_m < self.sole_width_m < .65*self.sole_length_m:
            raise ValueError("Sole width/length is inconsistent")
        if not 0 < self.blade_width_m < .15*self.sole_width_m:
            raise ValueError("Blade must be positive and much narrower than outsole")
        if self.hollow_radius_m <= self.blade_width_m/2:
            raise ValueError("Hollow radius must exceed blade half width")
        for name in ("sole_thickness_m", "heel_block_height_m", "plate_thickness_m", "blade_web_height_m",
                     "front_mount_height_m", "rear_mount_height_m", "runner_rear_extension_m",
                     "runner_front_inset_m", "toe_head_length_m", "render_step_m"):
            if a[name] <= 0:
                raise ValueError(name + " must be positive")
        if self.sole_thickness_m >= .08*self.sole_length_m or self.plate_thickness_m >= self.sole_thickness_m:
            raise ValueError("Invalid sole/plate thickness")
        if max(self.front_mount_height_m,self.rear_mount_height_m) >= .6*self.sole_length_m:
            raise ValueError("Mount height exceeds broad reference-family input guard")
        if self.heel_block_height_m >= .25*self.sole_length_m or self.runner_rear_extension_m >= .20*self.sole_length_m:
            raise ValueError("Heel block or rear extension exceeds broad input guard")
        if not .20 <= self.rocker_center_fraction < self.rocker_join_fraction <= .80:
            raise ValueError("Invalid longitudinal rocker stations")
        if self.main_rocker_m <= self.sole_length_m:
            raise ValueError("Primary radius is too small for this reference family; check metres vs feet")
        if not 0 < self.secondary_rocker_m <= self.main_rocker_m:
            raise ValueError("Secondary radius must be positive and no larger than main radius")
        if self.runner_front_inset_m >= .08*self.sole_length_m or self.toe_head_length_m >= .20*self.sole_length_m:
            raise ValueError("Invalid front runner/head extent")
        if self.render_step_m > .05*self.sole_length_m:
            raise ValueError("Rendering step is too coarse for this template")
        return self


def _arc_point(arc, theta):
    theta = np.asarray(theta, float)
    return (np.asarray(arc["center"]) + arc["radius_m"] *
            (np.cos(theta)[..., None]*np.asarray(arc["axis_cos"]) +
             np.sin(theta)[..., None]*np.asarray(arc["axis_sin"])))


def _profile(parameters):
    p = parameters
    xc = p.rocker_center_fraction*p.sole_length_m
    xj = p.rocker_join_fraction*p.sole_length_m
    x0 = -p.runner_rear_extension_m
    x1 = p.sole_length_m-p.runner_front_inset_m
    t0 = np.arcsin((x0-xc)/p.main_rocker_m)
    tj = np.arcsin((xj-xc)/p.main_rocker_m)
    main = dict(id="main", center=[xc, 0., p.main_rocker_m], axis_cos=[0., 0., -1.],
                axis_sin=[1., 0., 0.], radius_m=p.main_rocker_m,
                theta_start_rad=float(t0), theta_end_rad=float(tj))
    qj = _arc_point(main, tj)
    c2 = qj-p.secondary_rocker_m*np.array([np.sin(tj), 0., -np.cos(tj)])
    argument = (x1-c2[0])/p.secondary_rocker_m
    if not -1 < argument < 1:
        raise ValueError("Secondary radius cannot reach runner end on an increasing-x arc")
    t1 = float(np.arcsin(argument))
    if not tj < t1 < np.pi/2:
        raise ValueError("Secondary arc has invalid orientation")
    secondary = dict(id="secondary", center=c2.tolist(), axis_cos=[0., 0., -1.],
                     axis_sin=[1., 0., 0.], radius_m=p.secondary_rocker_m,
                     theta_start_rad=float(tj), theta_end_rad=t1)
    return main, secondary, (x0, xc, xj, x1)


def profile_height(x, parameters=SkateParameters()):
    """Exact continuous longitudinal lower-edge height, in local metres."""
    parameters.validate()
    main, secondary, stations = _profile(parameters)
    q = np.asarray(x, float)
    if np.any(q < stations[0]-1e-12) or np.any(q > stations[-1]+1e-12):
        raise ValueError("Requested x outside runner arc domain")
    out = np.empty_like(q)
    for select, arc in ((q <= stations[2], main), (q > stations[2], secondary)):
        dx = q[select]-arc["center"][0]
        out[select] = arc["center"][2]-np.sqrt(arc["radius_m"]**2-dx**2)
    return out


def _sole_half_width(x, p):
    return .5*p.sole_width_m*np.interp(np.asarray(x)/p.sole_length_m,
        [0., .12, .32, .68, .85, 1.], [.70, .76, .78, 1., .94, .65])


def _sole_bottom(x, p):
    return np.interp(np.asarray(x)/p.sole_length_m, [0., .30, .62, 1.],
        [p.rear_mount_height_m+p.heel_block_height_m,
         p.rear_mount_height_m+p.heel_block_height_m,
         p.front_mount_height_m, p.front_mount_height_m])


def _tube(rings, name):
    """Close ordered x sections whose yz contours are CCW viewed from +x."""
    r = np.asarray(rings, float)
    n, m, _ = r.shape
    vertices = np.concatenate([r.reshape(-1, 3), r[0].mean(0, keepdims=True), r[-1].mean(0, keepdims=True)])
    faces = []
    for j in range(m):
        k = (j+1) % m
        faces += [[n*m, k, j], [n*m+1, (n-1)*m+j, (n-1)*m+k]]
    for i in range(n-1):
        for j in range(m):
            k = (j+1) % m
            a, b, c, d = i*m+j, i*m+k, (i+1)*m+k, (i+1)*m+j
            faces += [[a,b,c],[a,c,d]]
    return dict(name=name, vertices=vertices, faces=np.asarray(faces, np.int32))


def _rect_tube(xs, widths, lows, highs, name):
    rings = []
    for x,w,lo,hi in zip(xs, widths, lows, highs):
        if hi <= lo or w <= 0:
            raise ValueError("Invalid nonpositive section in " + name)
        rings.append([[x,-w,lo],[x,w,lo],[x,w,hi],[x,-w,hi]])
    return _tube(rings, name)


def _triangulate_polygon(points):
    """Ear clipping for a simple CCW x/z polygon, retaining pick concavities."""
    q = np.asarray(points, float)
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    if sum(cross(a,b) for a,b in zip(q,np.roll(q,-1,axis=0))) <= 0:
        raise ValueError("Polygon must have positive x/z area")
    pending = list(range(len(q))); faces=[]
    while len(pending)>3:
        for k,b in enumerate(pending):
            a,c = pending[k-1],pending[(k+1)%len(pending)]
            if cross(q[b]-q[a],q[c]-q[b]) <= 1e-14:
                continue
            def inside(i):
                return all(cross(q[v]-q[u],q[i]-q[u]) >= -1e-14 for u,v in ((a,b),(b,c),(c,a)))
            if any(inside(i) for i in pending if i not in (a,b,c)):
                continue
            faces.append([a,b,c]);pending.pop(k);break
        else:
            raise ValueError("Degenerate or self-intersecting polygon")
    faces.append(pending)
    return np.asarray(faces,np.int32)


def _prism_xz(points, half_width, name):
    q = np.asarray(points,float); n=len(q)
    if half_width <= 0:
        raise ValueError("Positive extrusion half width required")
    cap = _triangulate_polygon(q)
    vertices = np.vstack([np.c_[q[:,0],np.full(n,-half_width),q[:,1]],
                          np.c_[q[:,0],np.full(n,half_width),q[:,1]]])
    faces = cap.tolist() + (cap[:,::-1]+n).tolist()
    for a in range(n):
        b=(a+1)%n
        faces += [[a,a+n,b+n],[a,b+n,b]]
    return dict(name=name,vertices=vertices,faces=np.asarray(faces,np.int32))


def _points_group(name, semantic, points):
    return dict(id=name, semantic=semantic, points=np.asarray(points,float))


def _jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def build_skate(parameters=SkateParameters()):
    """Return grouped surface mesh, exact contact primitives, and assembly info.

    The holder plates/posts are derived from the declared sole attachment
    planes. There is no independent per-frame blade pitch or free mounting
    transform. Calibrating the shared assembly must regenerate its geometry.
    """
    p = parameters.validate(); L=p.sole_length_m
    main,secondary,stations=_profile(p)
    arcs=[]
    for side,y in (("y_negative",-.5*p.blade_width_m),("y_positive",.5*p.blade_width_m)):
        for source in (main,secondary):
            arc={**source,"center":list(source["center"]),"id":side+"_"+source["id"],"semantic":"runner","edge_side":side}
            arc["center"][1]=y;arcs.append(arc)
    xs=np.unique(np.r_[np.arange(stations[0],stations[-1],p.render_step_m),stations])
    zs=profile_height(xs,p)
    yy=np.linspace(-.5*p.blade_width_m,.5*p.blade_width_m,5)
    hollow=np.sqrt(p.hollow_radius_m**2-yy**2)-np.sqrt(p.hollow_radius_m**2-(.5*p.blade_width_m)**2)
    if float(hollow.max()) >= p.blade_web_height_m:
        raise ValueError("Hollow cuts through blade web")
    rings=[]
    for x,z in zip(xs,zs):
        bottom=np.c_[np.full(len(yy),x), yy, z+hollow]
        rings.append(np.vstack([bottom,[x,yy[-1],z+p.blade_web_height_m],[x,yy[0],z+p.blade_web_height_m]]))
    meshes=[_tube(rings,"steel_runner")]

    # A single connected coarse pick head. The three tooth tips are explicit
    # candidates; roots and other solid vertices still participate in collision.
    xend,zend=stations[-1],float(profile_height(stations[-1],p))
    scale=p.toe_head_length_m/.024
    lower_offsets=np.array([[0,0],[.004,.008],[.007,.003],[.010,.014],
                             [.014,.009],[.017,.020],[.020,.014],[.024,.025]])*scale
    lower=lower_offsets+np.array([xend,zend])
    top_end=[xend+p.toe_head_length_m,zend+.025*scale+p.blade_web_height_m]
    polygon=np.vstack([lower,top_end,[xend,zend+p.blade_web_height_m]])
    head=_prism_xz(polygon,.5*p.blade_width_m,"toe_pick_head");meshes.append(head)
    tip_ids=np.array([2,4,6],int)
    tip_points=head["vertices"][np.r_[tip_ids,tip_ids+len(polygon)]]

    sole_x=L*np.array([0.,.12,.28,.30,.32,.62,.68,.85,.97,1.])
    bottom=_sole_bottom(sole_x,p)
    meshes.append(_rect_tube(sole_x,_sole_half_width(sole_x,p),bottom,bottom+p.sole_thickness_m,"outsole"))
    heel_x=L*np.array([0.,.12,.28]);heel_w=.92*_sole_half_width(heel_x,p)
    meshes.append(_rect_tube(heel_x,heel_w,np.full(3,p.rear_mount_height_m),_sole_bottom(heel_x,p),"heel_block"))
    mount_records=[]
    plate_definitions=[("rear",L*np.array([.04,.12,.24]),p.rear_mount_height_m),
                       ("front",L*np.array([.63,.68,.85,.97]),p.front_mount_height_m)]
    for name,xx,height in plate_definitions:
        supporting_width=.92*_sole_half_width(xx,p) if name=="rear" else _sole_half_width(xx,p)
        ww=supporting_width-.003
        if np.any(ww<=p.blade_width_m):raise ValueError("Mount plate no longer fits supporting sole")
        meshes.append(_rect_tube(xx,ww,np.full(len(xx),height-p.plate_thickness_m),np.full(len(xx),height),name+"_mount_plate"))
        mount_records.append(dict(id=name,plate_x_interval_m=xx[[0,-1]].tolist(),plane_z_m=height,
                                  attachment_part="heel_block" if name=="rear" else "outsole",
                                  top_gap_m=0.,minimum_lateral_inset_m=.003))
    post_records=[]
    for name,xf,height in (("rear_post",.14,p.rear_mount_height_m),
                           ("front_post",.70,p.front_mount_height_m),
                           ("front_nose_post",.92,p.front_mount_height_m)):
        xb=L*np.array([xf-.035,xf+.035]);xt=L*np.array([xf-.018,xf+.018])
        samples=np.r_[xb,np.linspace(*xb,9)]
        tops=profile_height(samples,p)+p.blade_web_height_m
        base=float(tops.min()-.001)
        ceiling=height-p.plate_thickness_m
        if ceiling-base < .005:
            raise ValueError("Nonpositive/insufficient holder clearance in "+name)
        if base <= float(profile_height(samples,p).max())+.001:
            raise ValueError("Post embedding would approach the running edge")
        # Convex trapezoid; its lower face embeds >=1 mm into the steel web.
        polygon=np.array([[xb[0],base],[xb[1],base],[xt[1],ceiling],[xt[0],ceiling]])
        meshes.append(_prism_xz(polygon,p.blade_width_m*.9,name))
        post_records.append(dict(id=name,base_z_m=base,plate_bottom_z_m=ceiling,
                                 minimum_web_overlap_m=float(tops.min()-base),plate_gap_m=0.))

    vertices=[];faces=[];face_parts=[];vertex_parts=[];part_ranges={}
    for mesh in meshes:
        offset=sum(len(a) for a in vertices)
        vertices.append(mesh["vertices"]);faces.append(mesh["faces"]+offset)
        face_parts.extend([mesh["name"]]*len(mesh["faces"]))
        vertex_parts.extend([mesh["name"]]*len(mesh["vertices"]))
        part_ranges[mesh["name"]]=[offset,offset+len(mesh["vertices"])]
    vertices=np.vstack(vertices);faces=np.vstack(faces)
    if vertices[:,2].min() < -1e-12:
        raise ValueError("Generated solids penetrate the horizontal reference plane")
    # Separate collision materials preserve different permitted small ice
    # penetration tolerances. A strict structural guard must not accidentally
    # recheck the blade with zero tolerance. Exact arcs additionally guard
    # continuous curvature that the display tessellation approximates.
    runner_guard=_points_group("runner_mesh_guard","runner",meshes[0]["vertices"])
    head_guard=_points_group("toe_head_guard","toe_pick",head["vertices"])
    guards=_points_group("structural_guard","solid_guard",np.vstack([m["vertices"] for m in meshes[2:]]))
    picks=_points_group("tooth_tips","toe_pick",tip_points)
    picks["support_eligible"]=True
    for group in (runner_guard,head_guard,guards):group["support_eligible"]=False
    out=dict(schema=SCHEMA,parameters=asdict(p),coordinate_convention=dict(units="m",x="heel_to_toe",y="local_left_not_medial_lateral",z="up",
             origin="outsole_rear_x; running_edge_lowest_z; y_midplane",transform="column vectors; X_world=T_WB @ X_B"),
             vertices=vertices,faces=faces,face_parts=np.asarray(face_parts),vertex_parts=np.asarray(vertex_parts),part_ranges=part_ranges,
             contact_arcs=arcs,contact_point_groups=[picks,runner_guard,head_guard,guards],
             support_feature_ids=dict(glide=[a['id'] for a in arcs],pick=['tooth_tips']),
             mounting_patches=mount_records,post_connections=post_records,
             derived_dimensions=dict(outsole_length_m=L,steel_overall_length_m=stations[-1]+p.toe_head_length_m-stations[0],
                 mounting_plate_span_m=.93*L,
                 note="Mounting span is front edge of toe plate minus rear edge of heel plate; it is not total steel length or foot length"),
             upper_interface=dict(sole_top_stations=np.c_[sole_x,np.zeros(len(sole_x)),bottom+p.sole_thickness_m],
                 sole_half_widths=_sole_half_width(sole_x,p),heel_seat=[.14*L,0.,p.rear_mount_height_m+p.heel_block_height_m+p.sole_thickness_m],
                 forefoot_seat=[.75*L,0.,p.front_mount_height_m+p.sole_thickness_m],
                 note="External construction anchors, not measured plantar/insole/ankle points; keep current open collar and fit inner cavity separately"),
             parameter_provenance=dict(primary_rocker=dict(status="manufacturer_candidate_prior_not_athlete_measurement",
                 candidates_m=[7*FT,8*FT],sources=["https://www.mkblades.com/gold-star","https://www.johnwilsonskates.com/gold-seal"]),
                 blade_width=dict(status="generic_prior_not_athlete_measurement",default_m=.0038,
                     source_interval_m=[.14*.0254,.155*.0254],
                     source="https://pnmresources.s3.amazonaws.com/pID-677/topic-430268/PSA%20EL%20Mem%20Package%20Course%203%20-%20Understanding%20The%20Figure%20Skating%20Blade.pdf"),
                 other_defaults="Engineering initialization only; dimensions, secondary profile, hollow, tooth locations and mounting heights require shared calibration",
                 bounds="Broad implementation/unit guards; not physiological or equipment standards"),
             invariants=dict(runner_main_secondary="G1 tangent continuous, generally not G2",mounting="derived from sole/heel planes; no free blade pitch",
                 surface_vs_contact="display tessellation independent from exact arc minima",upper="not generated",anatomical_binding="not supplied"))
    return out


def transform_contact_primitives(template, rotation, translation):
    """Apply one proper rigid B-to-world transform to exact contacts."""
    R,t=np.asarray(rotation,float),np.asarray(translation,float)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Finite 3x3 R and 3-vector t required")
    if not np.allclose(R.T@R,np.eye(3),atol=1e-9) or not np.isclose(np.linalg.det(R),1.,atol=1e-9):
        raise ValueError("A proper rigid rotation is required; no reflection or second scale")
    arcs=[]
    for arc in template["contact_arcs"]:
        a=dict(arc);a["center"]=(R@np.asarray(arc["center"])+t).tolist()
        for key in ("axis_cos","axis_sin"):a[key]=(R@np.asarray(arc[key])).tolist()
        arcs.append(a)
    groups=[dict(id=g["id"],semantic=g["semantic"],support_eligible=g.get("support_eligible",False),
                 points_world=np.asarray(g["points"])@R.T+t) for g in template["contact_point_groups"]]
    return dict(arcs=arcs,point_groups=groups)


def template_contact_features(template, rotation, translation):
    """Flat world-space list accepted directly by skate_contact.frame_residuals.

    Keep support membership explicit via template['support_feature_ids'];
    full solid guards must not become selected glide/pick support features.
    """
    world=transform_contact_primitives(template,rotation,translation)
    features=[dict(a,kind="arc") for a in world['arcs']]
    for group in world['point_groups']:
        f=dict(group,kind="points");f['points']=f.pop('points_world');features.append(f)
    return features


def export_template(destination,parameters=SkateParameters()):
    """Save an independently generated reference, never an original donor asset."""
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    template=build_skate(parameters)
    np.savez_compressed(dest/"lower_skate_geometry.npz",vertices=template["vertices"],faces=template["faces"],face_parts=template["face_parts"])
    meta={k:v for k,v in template.items() if k not in ("vertices","faces","face_parts","vertex_parts")}
    (dest/"lower_skate_geometry.json").write_text(json.dumps(_jsonable(meta),ensure_ascii=False,indent=2),encoding="utf-8")
    return template


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--main-rocker-ft",type=float,choices=[7.,8.],default=7.)
    args=parser.parse_args()
    result=export_template(args.output,SkateParameters(main_rocker_m=args.main_rocker_ft*FT))
    print(json.dumps(dict(vertices=len(result["vertices"]),faces=len(result["faces"]),parts=list(result["part_ranges"]),exact_arcs=len(result["contact_arcs"]))))
