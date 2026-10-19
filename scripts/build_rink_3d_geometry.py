"""Build an editable polygonal rink plane and closed vertical boards.

Input is the authoritative user-PPT schematic. No standard rink dimensions,
camera positions, advertising geometry or calibrated metric scale are imposed.
Default outputs are JSON mesh data and topology checks. Exchange OBJ/MTL and
glTF 2.0 GLB files are produced only with --export-exchange.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.13-women-sp/baseline/user_ppt_layout.json")
DEFAULT_OUT = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.14-scene3d")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rounded_boundary(bounds, radius, grid_spacing, arc_steps):
    """Counterclockwise polygon, including straight-side vertices <= grid size."""
    x0, y0, x1, y1 = bounds
    if not (x1 > x0 and y1 > y0 and 0 < radius < min(x1-x0, y1-y0)/2):
        raise ValueError("Expected ordered bounds and a positive, nondegenerate corner radius")
    corners = [(x1-radius,y1-radius,0), (x0+radius,y1-radius,90),
               (x0+radius,y0+radius,180), (x1-radius,y0+radius,270)]
    steps = max(arc_steps, int(math.ceil(math.pi*radius/(2*grid_spacing))))
    vertices = []
    for i, (cx, cy, first) in enumerate(corners):
        for a in np.linspace(first, first+90, steps+1):
            p = np.array([cx+radius*math.cos(math.radians(a)),cy+radius*math.sin(math.radians(a))])
            if not vertices or np.linalg.norm(p-vertices[-1]) > 1e-8:
                vertices.append(p)
        nx, ny, angle = corners[(i+1) % 4]
        end = np.array([nx+radius*math.cos(math.radians(angle)),ny+radius*math.sin(math.radians(angle))])
        start = vertices[-1].copy()
        count = max(1, int(math.ceil(np.linalg.norm(end-start)/grid_spacing)))
        for t in np.linspace(0,1,count+1)[1:]:
            vertices.append(start+t*(end-start))
    if np.linalg.norm(vertices[-1]-vertices[0]) < 1e-8:
        vertices.pop()
    return np.asarray(vertices)


def inside_convex(points, polygon, strict=False):
    p = np.asarray(points, float)
    a, edge = polygon, np.roll(polygon,-1,axis=0)-polygon
    delta = p[:,None,:]-a[None,:,:]
    cross = edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
    return np.all(cross > 1e-8, axis=1) if strict else np.all(cross >= -1e-7, axis=1)


def make_ice(boundary, bounds, spacing):
    x0,y0,x1,y1 = bounds
    xs = np.arange(math.ceil(x0/spacing),math.floor(x1/spacing)+1)*spacing
    ys = np.arange(math.ceil(y0/spacing),math.floor(y1/spacing)+1)*spacing
    xx, yy = np.meshgrid(xs,ys)
    grid = np.c_[xx.ravel(),yy.ravel()]
    grid = grid[inside_convex(grid,boundary,strict=True)]
    xy = np.vstack([boundary,grid])
    triangles = Delaunay(xy).simplices.copy()
    a,b,c = xy[triangles[:,0]],xy[triangles[:,1]],xy[triangles[:,2]]
    signed = (b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    triangles[signed < 0] = triangles[signed < 0][:,[0,2,1]]
    return {"name":"ice_surface", "positions":np.c_[xy,np.zeros(len(xy))].tolist(),
            "triangles":triangles.tolist(), "color":[0.86,0.93,0.96],
            "role":"ice_plane_z0", "boundary_vertex_indices":list(range(len(boundary))),
            "grid_interior_vertex_indices":list(range(len(boundary),len(xy))),
            "closed_volume":False,
            "topology":"triangulated_disk_with_the_shared_rink_perimeter"}


def make_boards(boundary,bounds,radius,height,thickness):
    n = len(boundary)
    ib = np.c_[boundary,np.zeros(n)]
    it = np.c_[boundary,np.full(n,height)]
    positions = [ib,it]
    faces = []
    for i in range(n):
        j=(i+1)%n
        # Inner face normal points toward the ice.
        faces.extend([[i,n+j,j],[i,n+i,n+j]])
    if thickness > 0:
        x0,y0,x1,y1 = bounds
        closest = np.clip(boundary,[x0+radius,y0+radius],[x1-radius,y1-radius])
        normal = boundary-closest
        normal /= np.linalg.norm(normal,axis=1)[:,None]
        outer = boundary+thickness*normal
        positions += [np.c_[outer,np.zeros(n)],np.c_[outer,np.full(n,height)]]
        for i in range(n):
            j=(i+1)%n
            ob,ob_next,ot,ot_next=2*n+i,2*n+j,3*n+i,3*n+j
            faces.extend([[ob,ob_next,ot_next],[ob,ot_next,ot]])
            faces.extend([[n+i,ot,ot_next],[n+i,ot_next,n+j]])
            faces.extend([[i,ob_next,ob],[i,j,ob_next]])
    return {"name":"rink_boards", "positions":np.vstack(positions).tolist(),
            "triangles":faces,"color":[0.89,0.9,0.92],
            "role":"vertical_boards_at_the_same_ice_perimeter",
            "inner_bottom_vertex_indices":list(range(n)),
            "inner_top_vertex_indices":list(range(n,2*n)),
            "closed_volume":thickness>0,
            "topology":"closed_annular_solid" if thickness>0 else "closed_perimeter_zero_thickness_wall"}


def edge_counts(faces):
    edges=Counter()
    directed=Counter()
    for a,b,c in faces:
        for i,j in [(a,b),(b,c),(c,a)]:
            edges[tuple(sorted((int(i),int(j))))]+=1
            directed[(int(i),int(j))]+=1
    return edges,directed


def validate(meshes,boundary,height,thickness):
    ice,boards=meshes
    p=np.asarray(ice['positions']);t=np.asarray(ice['triangles'])
    normals=np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]])
    assert np.all(normals[:,2]>1e-10), "Ice triangles must be nondegenerate and face up"
    assert np.all(inside_convex(p[:,:2],boundary)), "Ice/grid vertex is outside rink polygon"
    assert np.all(inside_convex(p[t].mean(axis=1)[:,:2],boundary)), "Ice triangle is outside rink"
    edges,_=edge_counts(t)
    n=len(boundary)
    actual={e for e,count in edges.items() if count==1}
    expected={tuple(sorted((i,(i+1)%n))) for i in range(n)}
    assert actual==expected, "Triangulated boundary must exactly match board base edges"
    assert all(v<=2 for v in edges.values()), "Nonmanifold ice edge"
    polygon_area=float(np.sum(boundary[:,0]*np.roll(boundary[:,1],-1)-boundary[:,1]*np.roll(boundary[:,0],-1))/2)
    triangles_area=float(normals[:,2].sum()/2)
    assert np.isclose(polygon_area,triangles_area,rtol=1e-10,atol=1e-8), "Ice has a gap or overlaps"
    bp=np.asarray(boards['positions']);bt=np.asarray(boards['triangles'])
    assert np.array_equal(bp[:n,:2],boundary) and np.all(bp[:n,2]==0)
    assert np.array_equal(bp[n:2*n,:2],boundary) and np.all(bp[n:2*n,2]==height)
    bn=np.cross(bp[bt[:,1]]-bp[bt[:,0]],bp[bt[:,2]]-bp[bt[:,0]])
    assert np.all(np.linalg.norm(bn,axis=1)>1e-10), "Degenerate board face"
    be,bd=edge_counts(bt)
    if thickness>0:
        assert all(v==2 for v in be.values()), "Thick boards must be a watertight solid"
        assert all(bd[(a,b)]==1 and bd[(b,a)]==1 for a,b in be), "Inconsistent board winding"
        volume=float(np.einsum('ij,ij->i',bp[bt[:,0]],np.cross(bp[bt[:,1]],bp[bt[:,2]])).sum()/6)
        assert volume>0, "Board solid normals point inward"
    else:
        volume=None
        assert sum(v==1 for v in be.values())==2*n
    return {"all_checks_passed":True,"ice_vertex_count":len(p),"ice_triangle_count":len(t),
            "boundary_vertex_count":n,"ice_area_model_units_squared":triangles_area,
            "board_vertex_count":len(bp),"board_triangle_count":len(bt),
            "ice_grid_inside_boundary":True,"ice_surface_has_no_gaps_or_overlaps":True,
            "board_inner_base_exactly_matches_ice_boundary":True,
            "board_perimeter_closed":True,"thick_board_mesh_watertight":thickness>0,
            "board_solid_signed_volume_model_units_cubed":volume,
            "note":"These are mesh/topology checks, not real-rink geometry validation."}


def write_obj(meshes,path):
    lines=["# RinkSpace normalized rink width 100; schematic units, not metres.",
           "# Coordinates: X right, Y up in plan, Z above ice.",f"mtllib {path.with_suffix('.mtl').name}"]
    materials=[];offset=1
    for mesh in meshes:
        name=mesh['name'];r,g,b=mesh['color']
        materials += [f"newmtl {name}",f"Ka {r*.2:.6f} {g*.2:.6f} {b*.2:.6f}",
                      f"Kd {r:.6f} {g:.6f} {b:.6f}","Ks 0.08 0.08 0.08","Ns 12","d 1","illum 2",""]
        lines += [f"o {name}",f"g {name}",f"usemtl {name}","s off"]
        lines += [f"v {x:.10g} {y:.10g} {z:.10g}" for x,y,z in mesh['positions']]
        lines += ["f "+" ".join(str(i+offset) for i in face) for face in mesh['triangles']]
        offset+=len(mesh['positions'])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    path.with_suffix('.mtl').write_text("\n".join(materials),encoding="utf-8")


def write_glb(meshes,path):
    """Minimal glTF 2.0, with independent mesh nodes and embedded binary data."""
    binary=bytearray()
    doc={"asset":{"version":"2.0","generator":"RinkSpace build_rink_3d_geometry.py",
                  "extras":{"units":"normalized_rink_width_100_not_metres", "coordinate_conversion":"JSON/OBJ Z-up to glTF Y-up: [x,z,-y]"}},
         "scene":0,"scenes":[{"nodes":list(range(len(meshes)))}],"nodes":[],"meshes":[],
         "materials":[],"buffers":[],"bufferViews":[],"accessors":[]}
    def accessor(array,component,type_,target,with_bounds=False):
        while len(binary)%4:binary.append(0)
        array=np.ascontiguousarray(array)
        view=len(doc['bufferViews']);offset=len(binary);binary.extend(array.tobytes())
        doc['bufferViews'].append({"buffer":0,"byteOffset":offset,"byteLength":array.nbytes,"target":target})
        item={"bufferView":view,"componentType":component,"count":len(array),"type":type_}
        if with_bounds:item.update(min=array.min(axis=0).tolist(),max=array.max(axis=0).tolist())
        idx=len(doc['accessors']);doc['accessors'].append(item);return idx
    for index,mesh in enumerate(meshes):
        source=np.asarray(mesh['positions'],float);faces=np.asarray(mesh['triangles'],dtype='<u4')
        xyz=np.c_[source[:,0],source[:,2],-source[:,1]]
        # Use a flat normal per exported triangle, preserving sharp top/bottom
        # board edges. Mesh topology remains indexed/checked in JSON and OBJ.
        expanded=xyz[faces].reshape(-1,3)
        tri=expanded.reshape(-1,3,3)
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        normals=np.repeat(normals,3,axis=0)
        pi=accessor(expanded.astype('<f4'),5126,'VEC3',34962,True)
        ni=accessor(normals.astype('<f4'),5126,'VEC3',34962)
        ii=accessor(np.arange(len(expanded),dtype='<u4'),5125,'SCALAR',34963)
        doc['materials'].append({"name":mesh['name'],"doubleSided":True,
            "pbrMetallicRoughness":{"baseColorFactor":[*mesh['color'],1.],"metallicFactor":0.,"roughnessFactor":.72}})
        doc['meshes'].append({"name":mesh['name'],"primitives":[{"attributes":{"POSITION":pi,"NORMAL":ni},"indices":ii,"material":index,"mode":4}]})
        doc['nodes'].append({"mesh":index,"name":mesh['name']})
    while len(binary)%4:binary.append(0)
    doc['buffers']=[{"byteLength":len(binary)}]
    encoded=json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    encoded+=b' '*((-len(encoded))%4)
    total=12+8+len(encoded)+8+len(binary)
    path.write_bytes(struct.pack('<III',0x46546C67,2,total)+struct.pack('<II',len(encoded),0x4E4F534A)+encoded+
                     struct.pack('<II',len(binary),0x004E4942)+binary)
    # Re-read binary container and accessor ranges, rather than trust the writer.
    content=path.read_bytes();magic,version,size=struct.unpack_from('<III',content)
    assert magic==0x46546C67 and version==2 and size==len(content)
    jl,jt=struct.unpack_from('<II',content,12);parsed=json.loads(content[20:20+jl])
    bl,bt=struct.unpack_from('<II',content,20+jl)
    assert jt==0x4E4F534A and bt==0x004E4942 and bl==parsed['buffers'][0]['byteLength']
    for v in parsed['bufferViews']:assert v['byteOffset']+v['byteLength']<=bl and v['byteOffset']%4==0
    return {"container_header_valid":True,"buffer_ranges_valid":True,"mesh_nodes":len(parsed['nodes'])}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--bounds',type=float,nargs=4,metavar=('XMIN','YMIN','XMAX','YMAX'))
    ap.add_argument('--corner-radius',type=float)
    ap.add_argument('--grid-spacing',type=float,default=5.)
    ap.add_argument('--board-height',type=float,default=4.)
    ap.add_argument('--board-thickness',type=float,default=.6)
    ap.add_argument('--arc-steps',type=int,default=24,help='Minimum subdivisions per quarter circle')
    ap.add_argument('--export-exchange',action='store_true',help='Also export optional OBJ/MTL and GLB exchange files')
    args=ap.parse_args()
    if not (args.grid_spacing>0 and args.board_height>0 and args.board_thickness>=0 and args.arc_steps>=2):
        raise ValueError('Invalid grid spacing, board dimensions or arc subdivision count')
    baseline_bytes=args.baseline.read_bytes();baseline=json.loads(baseline_bytes)
    feature=next(f for f in baseline['ice_features'] if f['id']=='rink_boundary')
    bounds=list(args.bounds if args.bounds is not None else feature['bounds'])
    radius=args.corner_radius if args.corner_radius is not None else feature['radius']
    boundary=rounded_boundary(bounds,radius,args.grid_spacing,args.arc_steps)
    meshes=[make_ice(boundary,bounds,args.grid_spacing),
            make_boards(boundary,bounds,radius,args.board_height,args.board_thickness)]
    checks=validate(meshes,boundary,args.board_height,args.board_thickness)
    args.out.mkdir(parents=True,exist_ok=True)
    exports={}
    if args.export_exchange:
        obj=args.out/'rink_scene.obj';glb=args.out/'rink_scene.glb'
        write_obj(meshes,obj)
        checks['glb_container']=write_glb(meshes,glb)
        exports={"obj":{"path":obj.name,"sha256":sha256(obj.read_bytes()),"axes":"Z-up"},
                 "mtl":{"path":obj.with_suffix('.mtl').name,"sha256":sha256(obj.with_suffix('.mtl').read_bytes())},
                 "glb":{"path":glb.name,"sha256":sha256(glb.read_bytes()),"axes":"Y-up"}}
    record={"schema_version":"rinkspace.scene-geometry.v1","model_id":"ankara_2026_shared",
            "revision":"v0.14-scene3d","status":"user_layout_3d_schematic_not_calibrated",
            "units":{"name":"normalized_rink_width_100","metres_per_unit":None,
                     "axes":"right-handed: X right in plan, Y up in plan, Z above ice", "ice_z":0.,
                     "glb_axes":"glTF Y-up; positions converted [x,y,z] -> [x,z,-y]"},
            "parameters":{"bounds":bounds,"corner_radius":radius,"grid_spacing":args.grid_spacing,
                          "board_height":args.board_height,"board_thickness":args.board_thickness,
                          "arc_steps_per_quarter":max(args.arc_steps,int(math.ceil(math.pi*radius/(2*args.grid_spacing)))),
                          "geometry_source":"user_PPT_rounded_rectangle_initial_hypothesis",
                          "board_height_status":"adjustable_demonstration_value_not_measured",
                          "board_thickness_status":"adjustable_demonstration_value_not_measured",
                          "corner_radius_status":"adjustable_PPT_display_value_not_measured"},
            "boundary_xy":boundary.tolist(),"boundary_closed":True,
            "boundary_closure_convention":"last vertex connects to first; first vertex is not repeated",
            "meshes":meshes,"checks":checks,
            "source":{"baseline_path":str(args.baseline.resolve()),"baseline_sha256":sha256(baseline_bytes),
                      "user_ppt_sha256":baseline['source']['sha256'],
                      "user_ppt_provenance":"Hash inherited from previously extracted user-PPT baseline; PPT not re-read by this build",
                      "script_path":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).read_bytes())},
            "exports":exports,
            "scope":{"ice_mesh":True,"closed_perimeter_vertical_boards":True,
                     "camera_positions_added":False,"advertisements_added":False,"ice_markings_added":False,
                     "physical_accuracy_verified":False}}
    geometry_path=args.out/'scene_geometry.json'
    geometry_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checks_path=args.out/'scene_geometry_checks.json'
    checks_record={"schema_version":"rinkspace.scene-geometry-checks.v1",
                   "geometry_path":geometry_path.name,"geometry_sha256":sha256(geometry_path.read_bytes()),
                   "source_baseline_sha256":record['source']['baseline_sha256'],
                   "checks":checks,"exchange_exports_requested":args.export_exchange}
    checks_path.write_text(json.dumps(checks_record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"output":str(args.out),"checks_path":str(checks_path),
                      "meshes":[m['name'] for m in meshes],"checks":checks},ensure_ascii=False))


if __name__=='__main__':
    main()
