"""Render the saved LSp programme, original broadcast, fitted ice and COM."""
from __future__ import annotations

import argparse
from copy import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.footwear_binding import transform
from calibration.program_tracking.body_tracking.programme_projection import required_clock
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from rinkspace.review.render import source_frames, validate_video
from rinkspace.viewer3d.source import SourceVideo

W, H = 1920, 1080
ROWS = 193
FPS = 50
TB = Fraction(1, 12800)
TRAIL = 31
COM_COLOR = (32, 198, 255)
BODY_COLOR = (240, 229, 51)
SHOE_COLOR = (232, 236, 127)
GRID_COLOR = (160, 125, 82)
RUNNER_COLORS = ((255, 237, 33), (247, 61, 236))

PROJECTION = """
#version 430
layout(local_size_x=128) in;
layout(std430, binding=0) readonly buffer CameraRows { vec4 camera[]; };
layout(std430, binding=1) readonly buffer MovingXYZ { vec4 moving[]; };
layout(std430, binding=2) readonly buffer StaticXYZ { vec4 frozen[]; };
layout(std430, binding=3) writeonly buffer Result { vec4 projected[]; };
uniform uint moving_count;
uniform uint total_count;
uniform float radius2;
vec3 position_at(uint p, float s) {
    if (p >= moving_count) return frozen[p-moving_count].xyz;
    bool first = s < 0.5;
    float u = first ? 4.0*s-1.0 : 4.0*s-3.0;
    uint k = p + (first ? 0u : 2u)*moving_count;
    return moving[k].xyz*(u*(u-1.0)*0.5)
         + moving[k+moving_count].xyz*(1.0-u*u)
         + moving[k+2u*moving_count].xyz*(u*(u+1.0)*0.5);
}
vec4 at_xyz(vec3 world, float row) {
    float s = clamp(row/1079.0,-0.25,1.25);
    float station = (s+0.25)*128.0;
    uint k = min(uint(station),191u);
    float u = station-float(k);
    uint a = 5u*k, b = a+5u;
    vec4 X = vec4(world,1.0);
    vec3 q = vec3(dot(X,mix(camera[a],camera[b],u)),
                  dot(X,mix(camera[a+1u],camera[b+1u],u)),
                  dot(X,mix(camera[a+2u],camera[b+2u],u)));
    if (q.z <= 0.000001) return vec4(0.0,0.0,-1.0,1e12);
    vec2 d = q.xy/q.z;
    float r = dot(d,d);
    if (r > 64.0) return vec4(0.0,0.0,-1.0,1e12);
    vec4 K = mix(camera[a+3u],camera[b+3u],u);
    vec4 B = mix(camera[a+4u],camera[b+4u],u);
    float l = 1.0+r*(B.x+B.y*r), xy = 2.0*d.x*d.y;
    vec2 distorted = d*l + vec2(B.z*xy + B.w*(r+2.0*d.x*d.x),
                                 B.z*(r+2.0*d.y*d.y) + B.w*xy);
    float scale_a = min(1.0,1.0+B.x*r+B.y*r*r);
    float scale_b = min(1.0,1.0+3.0*B.x*r+5.0*B.y*r*r);
    if (B.y > 0.0) {
        float a = clamp(-B.x/(2.0*B.y),0.0,r);
        float b = clamp(-3.0*B.x/(10.0*B.y),0.0,r);
        scale_a = min(scale_a,1.0+B.x*a+B.y*a*a);
        scale_b = min(scale_b,1.0+3.0*B.x*b+5.0*B.y*b*b);
    }
    float lower = min(scale_a,scale_b)-6.0*length(B.zw)*sqrt(r);
    float depth = r <= radius2 && lower > 1e-7 ? q.z : -1.0;
    return vec4(K.xy*distorted+K.zw,depth,r);
}
vec4 at_row(uint p, float row) {
    return at_xyz(position_at(p,clamp(row/1079.0,0.0,1.0)),row);
}
void main() {
    uint p = gl_GlobalInvocationID.x;
    if (p >= total_count) return;
    float row = 539.5;
    vec4 value = vec4(0.0);
    float error = 1e12;
    for (int i=0; i<20; ++i) {
        value = at_row(p,row);
        if (value.w >= 1e11) break;
        error = abs(value.y-row);
        if (error < 0.00025) break;
        row = value.y;
    }
    float depth = value.z;
    vec3 X = position_at(p,clamp(value.y/1079.0,0.0,1.0));
    if (isnan(error) || isinf(error) || error > 0.025 ||
        at_xyz(X,0.0).z <= 0.0 || at_xyz(X,1079.0).z <= 0.0)
        depth = -1.0;
    projected[p] = vec4(value.xy,depth,error);
}
"""
VERTEX = """
#version 430
in vec4 point;
out float valid_point;
void main() {
    vec2 s = (point.xy+vec2(.5))/vec2(1920,1080);
    valid_point = point.z;
    gl_Position = vec4(2.0*s.x-1.0,1.0-2.0*s.y,0.0,1.0);
}
"""
GEOMETRY = """
#version 430
layout(triangles) in;
layout(triangle_strip,max_vertices=3) out;
in float valid_point[];
void main() {
    if (valid_point[0] <= 0.0 || valid_point[1] <= 0.0 || valid_point[2] <= 0.0) return;
    for (int i=0; i<3; ++i) {
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""
FRAGMENT = """
#version 430
uniform float mask_value;
out float mask;
void main() { mask=mask_value; }
"""


def write_json(path, value):
    with path.open('x', encoding='utf8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def log(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def read_json(path):
    return Path(path).read_text(encoding='utf-8')


def bound_file(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f'Saved asset changed: {path}')
    return path


def with_w(points):
    result = np.zeros((*points.shape[:-1], 4), np.float32)
    result[..., :3] = points
    return result


def lines_in_ice(outline, metres_per_scene_unit):
    """Same one-metre origin and spacing as the review grid on the fitted outline."""
    low, high = outline[:, :2].min(0), outline[:, :2].max(0)
    first, second = outline[:-1], outline[1:]
    spacing = 1. / metres_per_scene_unit
    grid = []
    for axis in (0, 1):
        along = 1 - axis
        for value in np.arange(low[axis]+spacing, high[axis], spacing):
            keep = (first[:, axis] <= value) != (second[:, axis] <= value)
            a, b = first[keep], second[keep]
            fraction = (value-a[:, axis])/(b[:, axis]-a[:, axis])
            cross = a[:, along]+(b[:, along]-a[:, along])*fraction
            if len(cross) < 2:
                raise ValueError('Grid line missed the fitted ice outline')
            points = np.zeros((49, 3))
            points[:, axis] = value
            points[:, along] = np.linspace(cross.min(), cross.max(), len(points))
            grid.append(points)
    return grid


def scene_paths(core, state, scene_root, parent_sha, scale):
    record = json.loads(read_json(scene_root/'review_scene.json'))
    if record['camera_state_sha256'] != parent_sha:
        raise ValueError('Ice model is not bound to the same numeric camera scene')
    with np.load(scene_root/'review_scene.npz', allow_pickle=False) as archive:
        points = archive['xyz']
    ice = []
    for cut in record['cuts']:
        if cut['layer'] != 'ice':
            continue
        xyz = points[cut['begin']:cut['end']]
        if len(xyz) < 2:
            raise ValueError(f'Empty fitted ice marking: {cut["id"]}')
        np.testing.assert_allclose(xyz[:,2], 0., atol=1e-12, rtol=0)
        idx = np.unique(np.rint(np.linspace(0, len(xyz)-1, min(len(xyz),129))).astype(int))
        xyz = xyz[idx]
        if cut.get('closed') and np.linalg.norm(xyz[-1]-xyz[0]) > 1e-10:
            xyz = np.vstack([xyz, xyz[0]])
        ice.append(xyz)
    board = state['board']
    perimeter = core.board_perimeter(board)
    samples = np.linspace(0.,perimeter,2049)
    outline = np.array(core.board_points(board,samples,np.zeros(len(samples))))
    np.testing.assert_allclose(outline[-1],outline[0],rtol=0,atol=1e-10)
    np.testing.assert_allclose(outline[:,2],0.,rtol=0,atol=1e-12)
    grid = lines_in_ice(outline, scale)
    return [('grid',line) for line in grid]+[('ice',line) for line in ice]+[('boundary',outline[::4])]


class Raster:
    def __init__(self, moving_count, paths, body_faces, shoe_faces):
        self.ctx = moderngl.create_standalone_context(require=430)
        self.moving_count = moving_count
        xyz = np.concatenate([path for _, path in paths])
        self.total_count = moving_count + len(xyz)
        self.camera = self.ctx.buffer(reserve=ROWS*5*16)
        self.nodes = self.ctx.buffer(reserve=5*moving_count*16)
        self.static = self.ctx.buffer(with_w(xyz).tobytes())
        self.uv = self.ctx.buffer(reserve=self.total_count*16)
        for i, buffer in enumerate((self.camera,self.nodes,self.static,self.uv)):
            buffer.bind_to_storage_buffer(i)
        self.projector = self.ctx.compute_shader(PROJECTION)
        self.projector['moving_count'].value = moving_count
        self.projector['total_count'].value = self.total_count
        self.program = self.ctx.program(vertex_shader=VERTEX, geometry_shader=GEOMETRY,
                                        fragment_shader=FRAGMENT)
        self.meshes = []
        for indices in (body_faces,shoe_faces):
            ibo = self.ctx.buffer(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
            self.meshes.append(self.ctx.vertex_array(self.program, [(self.uv,'4f','point')],
                                                     index_buffer=ibo,index_element_size=4))
        self.tex = self.ctx.texture((W,H),1,dtype='f1')
        self.fbo = self.ctx.framebuffer([self.tex])
        self.fbo.use()

    def project(self, packed_camera, nodes, radius):
        self.camera.write(packed_camera.tobytes())
        self.nodes.write(with_w(nodes).tobytes())
        self.projector['radius2'].value = radius*radius
        self.projector.run(group_x=(self.total_count+127)//128)
        self.ctx.memory_barrier()
        return np.frombuffer(self.uv.read(),np.float32).reshape(-1,4)

    def mask(self, uv, mesh_vertex_count):
        coords = uv[:mesh_vertex_count]
        valid = coords[:,2] > 0
        if not valid.any():
            return None
        coords = coords[valid,:2]
        x0,y0 = np.maximum(np.floor(coords.min(axis=0)).astype(int)-4, (0,0))
        x1,y1 = np.minimum(np.ceil(coords.max(axis=0)).astype(int)+5, (W,H))
        if x1 <= x0 or y1 <= y0:
            return None
        self.uv.write(uv.tobytes())
        self.fbo.use()
        self.fbo.clear(0.,0.,0.,0.)
        for i, value in enumerate((1.,.5)):
            self.program['mask_value'].value = value
            self.meshes[i].render(moderngl.TRIANGLES)
        width,height = int(x1-x0),int(y1-y0)
        mask = np.frombuffer(self.fbo.read((int(x0),int(H-y1),width,height),
                                          components=1,alignment=1),np.uint8)
        return (slice(y0,y1),slice(x0,x1)),mask.reshape(height,width)[::-1]


class Programme:
    def __init__(self, args):
        project = args.project
        config = json.loads(read_json(args.config))
        variant = next(v for v in config['variants'] if v['id']==config['latest_review_variant'])
        self.variant = variant['id']
        self.scope_path = getattr(args,'scope',None) or args.config.parent/'rebuild_scope.json'
        self.scope = json.loads(read_json(self.scope_path))
        self.pts, self.shots, time_base = required_clock(self.scope)
        np.testing.assert_array_equal(time_base,[1,12800])
        self.source_spec = self.scope['source'] | self.scope['source_clock']
        self.source = bound_file(self.source_spec['registered_path'],self.source_spec['sha256'])
        if self.source != Path(config['source_video']):
            raise ValueError('Programme config and source scope name different inputs')
        self.result_path = bound_file(variant['result'],variant['result_sha256'])
        self.parameter = bound_file(variant['parameter'],variant['parameter_sha256'])
        camera_id = variant['camera_binding']
        self.camera_record_path = bound_file(camera_id['record']['path'],camera_id['record']['sha256'])
        camera_record = json.loads(read_json(self.camera_record_path))
        camera_state = bound_file(camera_id['state']['path'],camera_id['state']['sha256'])
        old_camera = json.loads(read_json(self.camera_record_path.with_name('camera_solver_refined_1102.json')))
        previous = bound_file(old_camera['state']['compact_path'],old_camera['state']['compact_sha256'])
        native_scene = load(camera_state)
        original = load(previous)
        for key in ('rig','controls','board','scene_values','layout'):
            np.testing.assert_array_equal(native_scene[key], original[key], err_msg=key)
        new_deps = json.loads(str(native_scene['joint_numeric_dependencies_json']))
        old_deps = json.loads(str(original['joint_numeric_dependencies_json']))
        for key in ('shots','control_dt','first_control_time'):
            if new_deps[key] != old_deps[key]:
                raise ValueError(f'Original camera clock dependency changed: {key}')
        manifest = json.loads(read_json(Path(camera_record['prepared'])/'joint_manifest.json'))
        self.scene_authority = bound_file(manifest['scene_path'],camera_record['scene_sha256'])
        self.state_hash = camera_id['state']['sha256']

        shared,self.camera,self.camera_info,_,_ = load_reference(project,args.prepared,args.source_evidence)
        saved = load(self.parameter)
        validate_checkpoint_identity(saved,shared.fixed_checkpoint_identity)
        if self.camera_info['state_sha256'] != self.state_hash:
            raise ValueError('Projection camera differs from saved programme')
        np.testing.assert_array_equal(shared.p['pts'],self.pts)
        np.testing.assert_array_equal(shared.p['shot_ids'],self.shots)
        self.clock = shared.times
        self.reference = absolute_reference(shared,curve_from_packet(saved),root=args.prepared)
        self.reference.relative_skate_curve = shoe_curve_from_packet(saved)
        self.reference.binding = copy(shared.binding)
        self.binding = self.reference.binding
        self.binding.reference = self.reference
        self.binding.x = saved['x'][:17].copy()
        self.binding._packet = self.binding._parameter_key = None
        self.surface = self.binding.surface
        self.mass = ContinuousMassGeometry(self.reference)
        self.scale = float(self.camera_info['metres_per_scene_unit'])
        self.solver_scale = float(self.camera_info['solver_metres_per_scene_unit'])
        np.testing.assert_allclose(self.scale/self.solver_scale,self.binding.ratio,rtol=0,atol=1e-12)
        with np.load(self.result_path,allow_pickle=False) as result:
            np.testing.assert_array_equal(result['pts'],self.pts)
            np.testing.assert_array_equal(result['source_shot_ids'],self.shots)
            self.com = result['com_world_mid'].copy()
            self.coco = result['COCO_uv'].copy()
        self.com_ground = self.com/self.scale
        self.com_ground[:,2] = 0.
        self.readout = self.camera.readout_times(np.arange(len(self.pts)))
        template = self.binding.packet()
        self.lower = np.asarray(template['template']['vertices'])
        self.upper = np.asarray(template['upper'])
        self.body_vertices = len(self.surface.shaped)
        self.shoe_vertices = len(self.lower)+len(self.upper)
        self.mesh_vertices = self.body_vertices+2*self.shoe_vertices
        self.body_faces = np.asarray(self.surface.faces)
        shell = np.vstack([template['template']['faces'],self.binding.faces+len(self.lower)])
        self.shoe_faces = np.vstack([shell+self.body_vertices,shell+self.body_vertices+self.shoe_vertices])
        self.scene_root = args.data_root/config['scene_root']
        self.paths = scene_paths(self.camera.core,native_scene,self.scene_root,
                                 old_camera['state']['compact_sha256'],self.scale)
        self.show_runner = getattr(args,'show_runner',False)
        self.runner_local = np.asarray(template['edges'])[:,::4].reshape(-1,3)
        self.runner_start = self.mesh_vertices+1+TRAIL
        self.runner_count = len(self.runner_local)*2 if self.show_runner else 0
        self.raster = Raster(self.runner_start+self.runner_count,self.paths,self.body_faces,self.shoe_faces)
        self.scene_points = np.concatenate([path for _,path in self.paths])
        self.path_slices = []
        at = self.raster.moving_count
        for kind, xyz in self.paths:
            self.path_slices.append((kind,slice(at,at+len(xyz))))
            at += len(xyz)

    def camera_rows(self, frames):
        cams = np.empty((len(frames),ROWS,5,4),np.float32)
        radius = np.empty(len(frames))
        for ix,camera,local in self.camera._groups(frames):
            p = camera.p
            offsets = np.linspace(-.75,.75,ROWS)*p['rig'][37]
            t = (p['motion_times'][local,None]+offsets).ravel()
            pose = camera.core.evaluate_motion_camera(p['rig'],p['controls'],
                    float(p['first_control_time']),float(p['control_dt']),t)
            if pose['status'].any():
                raise ValueError('Frozen camera failed at a selected sensor row')
            R = pose['rotation']
            T = -np.einsum('nij,nj->ni',R,pose['centre'])
            rows = np.concatenate([np.concatenate([R,T[...,None]],axis=2),
                                   pose['values'][:,16:].reshape(-1,2,4)],axis=1)
            cams[ix] = rows.reshape(len(ix),ROWS,5,4)
            radius[ix] = p['rig'][25]
        return cams,radius

    def geometry(self, frames):
        tau = self.readout[frames]
        nodes = self.clock[frames,None]+tau[:,None]*np.linspace(-.5,.5,5)
        base = self.reference.at(nodes.ravel())
        pose = fk_state(self.reference,base,np.zeros((nodes.size,69)))
        local = self.surface.sample_vertices(pose['native'],pose['frames'],
                                            hand_local_R=pose.get('hand_local_R'))
        skin = np.einsum('nij,nvj->nvi',pose['Q'],local)+pose['P'][:,None]
        deformation = self.binding.deformation(pose['native'],pose['frames'],
                                               skate_relative_R=pose['skate_relative_R'])
        upper = self.binding.deform(self.upper,deformation)
        boots = np.concatenate([transform(pose['R'],pose['O'],self.lower),
                                transform(pose['R'],pose['O'],upper)],axis=2)
        shoes = boots.reshape(len(skin),-1,3)
        C = self.mass.com(self.mass.from_pose(pose)[0]).reshape(len(frames),5,3)
        np.testing.assert_allclose(C[:,2],self.com[frames],rtol=0,atol=3e-10)
        C[...,2] = 0.
        mesh = np.concatenate([skin,shoes],axis=1).reshape(len(frames),5,self.mesh_vertices,3)/self.solver_scale
        earlier = np.maximum(frames[:,None]-4*np.arange(TRAIL,0,-1),0)
        trail = np.broadcast_to(self.com_ground[earlier][:,None],(len(frames),5,TRAIL,3))
        parts = [mesh,(C/self.scale)[:,:,None],trail]
        if self.show_runner:
            runner=transform(pose['R'],pose['O'],self.runner_local)
            parts.append(runner.reshape(len(frames),5,self.runner_count,3)/self.solver_scale)
        return np.concatenate(parts,axis=2)

    def compare_native(self, frames, nodes, projected):
        """Compare an actual spread of moving mesh, COM and static ice points."""
        rng = np.random.default_rng(int(self.pts[frames[0]]))
        vertices = np.r_[rng.choice(self.body_vertices,128,replace=False),
                         np.linspace(self.body_vertices,self.mesh_vertices-1,34,dtype=int),
                         self.mesh_vertices,self.mesh_vertices+1,self.mesh_vertices+TRAIL]
        if self.show_runner:
            vertices=np.r_[vertices,np.linspace(self.runner_start,self.runner_start+self.runner_count-1,16,dtype=int)]
        checks = []
        for j, frame in enumerate(frames):
            f = np.array([frame])
            uv = projected[j]
            sampled = uv[vertices]
            both = (sampled[:,2]>0) & np.all(
                (sampled[:,:2]>=[-3,-3]) & (sampled[:,:2]<[W+3,H+3]),axis=-1)
            move_max = 0.
            if both.any():
                ref = moving_projection(self.camera,nodes[j:j+1,:,vertices[both]]*self.solver_scale,f)['uv'][0]
                move_max = float(np.max(np.linalg.norm(ref-sampled[both,:2],axis=-1)))
            good = (uv[self.raster.moving_count:,2]>0) & np.all(
                (uv[self.raster.moving_count:,:2]>=[0,0]) &
                (uv[self.raster.moving_count:,:2]<[W,H]),axis=-1)
            ids = np.flatnonzero(good)
            ids = ids[np.linspace(0,len(ids)-1,min(len(ids),72)).astype(int)]
            if not len(ids):
                raise ValueError('No visible fitted ice samples for native comparison')
            cs = self.camera.cameras[self.shots[frame]]
            p = cs.p
            local = frame-self.camera.first[self.shots[frame]]
            exact = cs.core.project_motion(p['rig'],p['controls'],float(p['first_control_time']),
                                           float(p['control_dt']),H,p['motion_times'][local],self.scene_points[ids])
            valid = exact['status']==0
            if not valid.any():
                raise ValueError('No comparable ice points for this camera exposure')
            a = uv[self.raster.moving_count+ids[valid],:2]
            static_max = float(np.max(np.linalg.norm(exact['uv'][valid]-a,axis=-1)))
            if max(move_max,static_max) > .15:
                raise ValueError(f'Native projection replay differs by {max(move_max,static_max):.4f} px at {self.pts[frame]}')
            checks.append(dict(pts=int(self.pts[frame]),shot=str(self.shots[frame]),
                               moving_points=int(both.sum()),ice_points=int(valid.sum()),
                               ice_points_native_rejected=int((~valid).sum()),
                               moving_max_error_px=move_max,ice_max_error_px=static_max))
        return checks


def visible_polylines(projected):
    valid = (projected[:,2]>0)&(np.abs(projected[:,:2])<20000).all(1)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    pixel = np.rint(projected[ids,:2]*16.).astype(np.int32)
    split = np.flatnonzero(np.diff(ids)!=1)+1
    return [p for p in np.split(pixel,split) if len(p)>1]


def source_time(pts):
    ms = int(pts)*1000//12800
    return f'{ms//60000:02d}:{ms//1000%60:02d}.{ms%1000:03d}'


def legend(show_runner=False):
    image = Image.new('RGB',(860 if show_runner else 610,44),(23,29,37))
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    draw = ImageDraw.Draw(image)
    for x, text, color in ((14,'人体与冰鞋',BODY_COLOR),(151,'COM 冰面投影',COM_COLOR),
                           (324,'拟合冰面网格', (192,188,172))):
        draw.line((x,22,x+16,22),fill=color[::-1],width=3)
        draw.text((x+24,9),text,font=font,fill=(238,242,248))
    if show_runner:
        for x,text,color in zip((475,595),('左刀下缘','右刀下缘'),RUNNER_COLORS):
            draw.line((x,22,x+16,22),fill=color[::-1],width=3)
            draw.text((x+24,9),text,font=font,fill=(238,242,248))
    return np.array(image)[:,:,::-1].copy()


def compose(raw, programme, frame, projected, panel):
    canvas = raw
    grid = np.zeros((H,W),np.uint8)
    paths = {'grid': [],'ice': [],'boundary': []}
    for kind, section in programme.path_slices:
        paths[kind].extend(visible_polylines(projected[section]))
    for kind, weight, thickness in (('grid',72,1),('ice',119,1),('boundary',168,2)):
        if paths[kind]:
            cv2.polylines(grid,paths[kind],False,weight,thickness,cv2.LINE_AA,shift=4)
    grid_pixels = grid>0
    if grid_pixels.any():
        a = grid[grid_pixels,None].astype(np.float32)/255.
        canvas[grid_pixels] = (canvas[grid_pixels]*(1.-a)+np.asarray(GRID_COLOR)*a).astype(np.uint8)

    coverage = programme.raster.mask(projected,programme.mesh_vertices)
    body_pixels = 0
    if coverage is not None:
        box, mask = coverage
        roi = canvas[box]
        body_pixels = int(np.count_nonzero(mask))
        for use, color, opacity in ((mask>=200,BODY_COLOR,.23),((mask>=64)&(mask<200),SHOE_COLOR,.24)):
            if not np.any(use):
                continue
            roi[use] = ((1.-opacity)*roi[use]+opacity*np.asarray(color)).astype(np.uint8)
            contour,_ = cv2.findContours(use.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi,contour,-1,color,1,cv2.LINE_AA)

    if programme.show_runner:
        runner = projected[programme.runner_start:programme.runner_start+programme.runner_count]
        for foot,color in zip(runner.reshape(2,2,-1,4),RUNNER_COLORS):
            for edge in foot:
                lines=visible_polylines(edge)
                if lines:
                    cv2.polylines(canvas,lines,False,(20,25,32),4,cv2.LINE_AA,shift=4)
                    cv2.polylines(canvas,lines,False,color,2,cv2.LINE_AA,shift=4)
    head = projected[programme.mesh_vertices]
    track = np.vstack([projected[programme.mesh_vertices+1:programme.runner_start],head])
    trail = visible_polylines(track)
    if trail:
        cv2.polylines(canvas,trail,False,(41,50,57),4,cv2.LINE_AA,shift=4)
        cv2.polylines(canvas,trail,False,COM_COLOR,2,cv2.LINE_AA,shift=4)
    com_visible = bool(head[2]>0 and np.all((head[:2]>=[0,0])&(head[:2]<[W,H])))
    if com_visible:
        xy = tuple(np.rint(head[:2]).astype(int))
        cv2.circle(canvas,xy,9,(34,41,49),-1,cv2.LINE_AA)
        cv2.circle(canvas,xy,7,COM_COLOR,2,cv2.LINE_AA)
        cv2.circle(canvas,xy,2,COM_COLOR,-1,cv2.LINE_AA)

    y0,x0 = H-58,14
    overlay = canvas[y0:y0+44,x0:x0+panel.shape[1]]
    cv2.addWeighted(overlay,.20,panel,.80,0,overlay)
    cv2.putText(canvas,f'{programme.shots[frame].upper()}  {source_time(programme.pts[frame])}',
                (x0+(721 if programme.show_runner else 474),y0+27),cv2.FONT_HERSHEY_SIMPLEX,.58,(247,250,253),1,cv2.LINE_AA)
    return canvas,body_pixels,com_visible,int(grid_pixels.sum())


def sample_frames(p):
    starts = np.r_[0,np.flatnonzero(p.shots[1:]!=p.shots[:-1])+1]
    stops = np.r_[starts[1:],len(p.pts)]
    key = np.unique(np.r_[starts,(starts+stops-1)//2,stops-1,
                          np.searchsorted(p.pts,[round(t*12800) for t in (157.82,162.74,163.10,164.16)])])
    return starts,stops,key


def project_batch(p, frames):
    xyz = p.geometry(frames)
    cams,radii = p.camera_rows(frames)
    results = [p.raster.project(cam,nodes,radius) for cam,nodes,radius in zip(cams,xyz,radii)]
    return xyz,cams,results


def preview(args,p):
    target = args.output/'preview_final'
    target.mkdir(parents=True,exist_ok=False)
    _,_,sample = sample_frames(p)
    panel = legend(p.show_runner)
    audits = []
    started = time.perf_counter()
    reader = SourceVideo(str(p.source),(1,12800))
    try:
        for i in range(0,len(sample),8):
            frames = sample[i:i+8]
            xyz,cams,results = project_batch(p,frames)
            audits.extend(p.compare_native(frames,xyz,results))
            for f,uv in zip(frames,results):
                pts = int(p.pts[f])
                raw = cv2.cvtColor(reader.read(pts),cv2.COLOR_RGB2BGR)
                frame,b,g,s = compose(raw,p,f,uv,panel)
                if not cv2.imwrite(str(target/f'{p.shots[f]}_{pts}.png'),frame):
                    raise OSError('Unable to save selected source exposure')
                log('preview_exposure',pts=pts,shot=str(p.shots[f]),body_pixels=b,
                    com_on_screen=g,grid_pixels=s)
    finally:
        reader.close()
    write_json(target/'projection_checks.json',audits)
    log('preview_verified',frames=len(sample),native_projection_comparison=audits,
        seconds=time.perf_counter()-started)


def encoder(args, p, dest, stderr):
    src_start = float(p.pts[0]*TB)
    duration = len(p.pts)/FPS
    flags = [
        str(args.ffmpeg),'-hide_banner','-loglevel','warning','-nostdin','-n',
        '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{W}x{H}',
        '-framerate',str(FPS),'-i','pipe:0',
        '-ss',f'{src_start:.3f}','-i',str(p.source),
        '-map','0:v:0','-map','1:a:0','-map_metadata','-1','-map_chapters','-1',
        '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-c:v','h264_nvenc','-preset','p4','-cq','17','-b:v','0',
        '-fps_mode','passthrough','-video_track_timescale','12800',
        '-af',f'atrim=duration={duration:.3f},asetpts=PTS-STARTPTS',
        '-c:a','aac','-b:a','256k','-movflags','+faststart',str(dest)]
    return subprocess.Popen(flags,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                            stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)


def render(args,p):
    args.output.mkdir(parents=True,exist_ok=True)
    out = args.output/'okada_mei_sp_9shots_body_com_overlay.mp4'
    pending = args.output/'okada_mei_sp_9shots_body_com_overlay.pending.mp4'
    if out.exists() or pending.exists():
        raise FileExistsError('Preserve earlier export and use a new path')
    cv2.setNumThreads(1)
    starts,stops,audit_frames = sample_frames(p)
    audio = av.open(str(p.source))
    try:
        if len(audio.streams.audio)!=1 or audio.streams.audio[0].codec_context.sample_rate!=48000:
            raise ValueError('Expected one original 48 kHz programme audio stream')
    finally:
        audio.close()
    panel = legend(p.show_runner)
    reader = iter(source_frames(p.source,p.source_spec,p.pts))
    counts = np.zeros((len(p.pts),3),np.int32)
    body_data = np.zeros((len(p.pts),4),np.float32)
    audits = []
    started = time.perf_counter()
    progress = started
    log('render_started',renderer=p.raster.ctx.info['GL_RENDERER'],native_frames=len(p.pts),
        shots=len(starts),body_vertices=p.body_vertices,body_faces=len(p.body_faces),
        scene_paths=len(p.paths),camera_sha256=p.state_hash)
    with (args.output/'ffmpeg.log').open('xb') as error:
        enc = encoder(args,p,pending,error)
        try:
            for i in range(0,len(p.pts),args.batch):
                frames = np.arange(i,min(i+args.batch,len(p.pts)))
                xyz,cams,results = project_batch(p,frames)
                keep = np.flatnonzero(np.isin(frames,audit_frames))
                if len(keep):
                    audits.extend(p.compare_native(frames[keep],xyz[keep],[results[j] for j in keep]))
                for f,uv in zip(frames,results):
                    pts,raw = next(reader)
                    if pts!=p.pts[f]:
                        raise ValueError(f'Expected source PTS {p.pts[f]}, read {pts}')
                    pixels,b,c,g = compose(raw,p,f,uv,panel)
                    counts[f] = b,int(c),g
                    body_data[f] = uv[p.mesh_vertices]
                    enc.stdin.write(memoryview(np.ascontiguousarray(pixels)).cast('B'))
                now = time.perf_counter()
                if i==0 or now-progress>=12. or frames[-1]==len(p.pts)-1:
                    log('render_progress',frames=int(frames[-1])+1,total=len(p.pts),
                        source_seconds=float(p.pts[frames[-1]]*TB),shot=str(p.shots[frames[-1]]),
                        elapsed_s=round(now-started,2),render_fps=round((frames[-1]+1)/(now-started),1))
                    progress = now
            if next(reader,None) is not None:
                raise ValueError('Decoded more original frames than the saved programme')
            enc.stdin.close()
            if enc.wait()!=0:
                raise RuntimeError('FFmpeg encoder or source audio failed; see ffmpeg.log')
        finally:
            reader.close()
            if enc.poll() is None:
                enc.terminate()
                enc.wait()
    log('validating_encoded_stream',frames=len(p.pts),duration_s=len(p.pts)/FPS)
    check = validate_video(pending,p.pts,TB)
    with av.open(str(pending)) as container:
        v,a = container.streams.video[0],container.streams.audio[0]
        if (v.width,v.height,v.average_rate)!=(W,H,50) or len(container.streams.audio)!=1:
            raise ValueError('Encoded video dimensions, fps or audio stream mismatch')
        wanted = Fraction(len(p.pts),FPS)
        audio_start = Fraction(a.start_time)*a.time_base
        audio_duration = Fraction(a.duration)*a.time_base
        if abs(audio_start)>Fraction(1,1000) or abs(audio_duration-wanted)>Fraction(1,30):
            raise ValueError('Encoded original-audio timing mismatch')
    segments=[]
    for a,b in zip(starts,stops):
        segments.append(dict(shot=str(p.shots[a]),source_first_pts=int(p.pts[a]),
                             source_last_pts=int(p.pts[b-1]),source_first_seconds=float(p.pts[a]*TB),
                             source_last_seconds=float(p.pts[b-1]*TB),frames=int(b-a),
                             visible_body_frames=int((counts[a:b,0]>0).sum()),
                             visible_com_frames=int(counts[a:b,1].sum()),grid_frames=int((counts[a:b,2]>0).sum())))
    if len(audits)!=len(audit_frames) or not np.all(counts[:,2]>0):
        raise ValueError('Incomplete native projection audit or fitted rink coverage')
    np.savez_compressed(args.output/'source_frame_map.npz',source_pts=p.pts,shot_ids=p.shots,
                        source_time_base=[1,12800],encoded_pts=p.pts-p.pts[0],
                        body_mask_pixels=counts[:,0],com_on_screen=counts[:,1].astype(bool),
                        rink_grid_pixels=counts[:,2],com_ice_projection_uv=body_data[:,:2],
                        com_projection_valid=body_data[:,2]>0)
    pending.rename(out)
    report = dict(video=str(out),video_sha256=digest(out),candidate=p.variant,
                  parameter_sha256=digest(p.parameter),result_sha256=digest(p.result_path),
                  source=str(p.source),source_sha256=p.source_spec['sha256'],
                  source_time_base=[1,12800],camera_state_sha256=p.state_hash,
                  scene_authority_sha256=digest(p.scene_authority),
                  render_native_camera_library=p.camera_info['native_library_sha256'],
                  render_script_sha256=digest(Path(__file__)),
                  ice_sample_asset_sha256=digest(p.scene_root/'review_scene.npz'),
                  source_first_seconds=float(p.pts[0]*TB),source_last_seconds=float(p.pts[-1]*TB),
                  end_seconds_exclusive=float((p.pts[-1]+256)*TB),
                  source_pts_count=len(p.pts),image_size=[W,H],playback_fps=FPS,
                  original_audio_trim=True,audio_start_seconds=float(audio_start),
                  audio_duration_seconds=float(audio_duration),scene_grid_paths=len(p.paths),
                  segments=segments,clock_check=check,native_projection_checks=audits,
                  native_projection_max_error_px=max(max(r['moving_max_error_px'],r['ice_max_error_px']) for r in audits),
                  projection='Original Brown camera and its rolling readout, sampled at 193 rows spanning the sensor and a margin for GPU replay. Five exact articulated body samples per exposure.',
                  com='Same saved segment mass model as the body. World Z set to fitted ice plane, then projected using the current exposure and camera. Last 2.48 s in rink coordinates.',
                  runner_overlay=dict(enabled=p.show_runner,component='both fixed lower runner edges per skate',
                                      projection='same attached mesh and rolling native camera'),
                  notes=['Review rendering only. No source segmentation or correction for foreground occluders.',
                         'The grid uses the fitted rink boundary and the programme conditional scale.',
                         'The COM mark represents the fitted model, including during flight.'],
                  elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'verification.json',report)
    log('complete',frames=len(p.pts),video=str(out),seconds=report['elapsed_seconds'],
        native_projection_max_error_px=report['native_projection_max_error_px'],segments=segments)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','data-root','config','prepared','source-evidence','output','ffmpeg'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--show-runner',action='store_true',help='Show the left and right material runner edges in distinct colours')
    parser.add_argument('--scope',type=Path,help='Explicit programme clock and source registration')
    parser.add_argument('--batch',type=int,default=12)
    args=parser.parse_args()
    programme=Programme(args)
    if args.preview:
        preview(args,programme)
    else:
        render(args,programme)


if __name__=='__main__':
    main()
