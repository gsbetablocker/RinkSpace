"""Check the actual player in a standalone OpenGL context, without desktop input.

This verifies display plumbing and source-clock controls, not body accuracy.
It never enters Viewer.run() and never falls back to a visible window backend.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import imgui
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aitviewer.viewer import Viewer

from .app import ResultsPlayer
from .data import ResultCatalog, digest
from .framing import fit_comparison_camera, comparison_frustum_bounds


class HeadlessResultsPlayer(ResultsPlayer):
    # Viewer.__init__ consults C.window_type only when this attribute is None.
    window_type='headless'


def camera_state(player):
    camera=player.scene.camera
    return dict(position=camera.position.copy(),target=camera.target.copy(),up=camera.up.copy(),
                is_ortho=bool(camera.is_ortho),ortho_size=float(camera.ortho_size))


def set_camera(player,state):
    camera=player.scene.camera
    for key in ('position','target','up'):setattr(camera,key,state[key].copy())
    camera.is_ortho=state['is_ortho'];camera.ortho_size=state['ortho_size']


def same_camera(player,state):
    current=camera_state(player)
    return all(np.array_equal(current[key],value) for key,value in state.items())


def geometry_digest(sequence):
    identity=hashlib.sha256()
    for name,array in [('vertices',sequence.vertices),*sorted(sequence.state.items())]:
        identity.update(name.encode());identity.update(str(array.shape).encode());identity.update(str(array.dtype).encode())
        identity.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return identity.hexdigest()


def continuous_playback(player,catalog,seconds):
    """Render each requested source exposure using the actual playback clock."""
    start,end=map(float,seconds)
    if not np.isfinite([start,end]).all() or end<=start:raise ValueError('Playback interval must be finite and increasing')
    first,last=catalog.frame_for_seconds(start),catalog.frame_for_seconds(end)
    if last<=first:raise ValueError('Playback interval contains fewer than two source exposures')
    fps=float(player.playback_fps)
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid playback FPS')
    player.seek(first);records=[];started=time.perf_counter()
    def record():
        records.append(dict(local_frame=player.frame,source_pts=int(catalog.pts[player.frame]),
                            displayed_source_pts=int(player.sequence.state['pts'][player.frame]),
                            source_seconds=float(catalog.seconds[player.frame])))
    record();player.toggle_animation(True)
    try:
        for expected in range(first+1,last+1):
            # Advance one exposure despite binary rounding around int(dt*fps).
            render_time=player._last_frame_rendered_at+(1.+1e-8)/fps
            player.on_render(render_time,1./fps,export=True)
            record()
            if player.frame!=expected:raise AssertionError('Continuous playback skipped or repeated a source exposure')
    finally:player.toggle_animation(False)
    expected_pts=catalog.pts[first:last+1]
    actual_pts=np.array([row['source_pts'] for row in records])
    displayed_pts=np.array([row['displayed_source_pts'] for row in records])
    continuous=bool(np.array_equal(actual_pts,expected_pts) and np.array_equal(displayed_pts,expected_pts))
    elapsed=time.perf_counter()-started
    return dict(requested_seconds=[start,end],playback_fps=fps,source_time_base=list(catalog.time_base),
                source_first_last_pts=[int(expected_pts[0]),int(expected_pts[-1])],
                exposures=len(records),rendered_steps=last-first,no_skipped_or_repeated_exposures=continuous,
                source_pts_deltas=np.diff(actual_pts).tolist(),records=records,
                rendering_wall_seconds=elapsed,rendered_steps_per_wall_second=(last-first)/max(elapsed,1e-12),
                wall_render_rate_is_not_source_playback_fps=True)


def verify(config,output,project=None,data_root=None,variants=('ref1102','obs1102','full1102'),seconds=(36.02,26.26),wear_inspection=False,view='free',boot_section=0,inspection_layers=(),fit_comparison_bounds=False,playback_seconds=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='rinkspace.headless-player-check.v1',passed=False,
                method='Complete ResultsPlayer subclass with moderngl standalone headless context and its own OpenGL renderer; no desktop input or visible window',
                checks=[],comparisons=[],accuracy_accepted=False,contact_status='unknown')
    started=time.perf_counter();run_stamp=str(time.time_ns());player=None;global_session=None;global_identity=None

    def check(name,condition,**details):
        report['checks'].append(dict(check=name,passed=bool(condition),**details))
        if not condition:raise AssertionError(name)

    try:
        catalog=ResultCatalog(project=project,data_root=data_root,config=config)
        global_session=catalog.session_root/'player_state.json'
        global_identity=digest(global_session) if global_session.exists() else None
        # Override before construction: __init__, seek, camera changes and
        # periodic saves all write only into this isolated verification session.
        catalog.session_root=output/'session';catalog.session_root.mkdir(parents=True,exist_ok=True)
        report.update(config=str(catalog.config_path.resolve()),session_root=str(catalog.session_root),
                      scene_label=catalog.config.get('scene_label'),display_metres_per_scene_unit=catalog.scale,
                      frames=len(catalog.pts),variants=list(variants))
        configured_comparison=catalog.config.get('comparison_variants',[])
        source_variants=list(dict.fromkeys([*variants,*configured_comparison]))
        for name in source_variants:catalog.variant(name)
        sources=[catalog.root/'input.npz',*[catalog.result_path(catalog.variant(name)) for name in source_variants]]
        source_identity={str(path):digest(path) for path in sources}
        player=HeadlessResultsPlayer(catalog,variant=None if configured_comparison else variants[0],restore=False,size=(960,720))
        player.set_wear_inspection(bool(wear_inspection),inspection_layers)
        player.boot_section=int(boot_section)
        report['inspection']=dict(wear_inspection=bool(wear_inspection),view=view,boot_section=int(boot_section),diagnostic_layers=list(inspection_layers),fit_comparison_bounds=bool(fit_comparison_bounds))
        report['opengl']={key:player.ctx.info.get(key) for key in ('GL_VENDOR','GL_RENDERER','GL_VERSION','GL_SHADING_LANGUAGE_VERSION')}
        report['window_class']=type(player.window).__module__+'.'+type(player.window).__name__
        check('standalone headless backend',player.window_type=='headless' and player.window.name=='headless')
        player._init_scene()
        # export=True omits the GUI and cannot catch actual ImGui upload
        # failures. Exercise the same overlay renderer as the native window.
        player.on_render(0.,0.,export=False)
        player.on_render(.02,.02,export=False)
        usage=player.imgui.rinkspace_buffer_usage
        check('actual GUI buffers fit rendered command lists',bool(usage) and all(
            row['required_bytes']>0 and row['required_bytes']<=row['capacity_bytes'] for row in usage.values()),**usage)
        report['gui_buffer_usage']=usage
        check('native source sequence and full surface',player.scene.n_frames==len(catalog.pts) and player.sequence.vertices.shape==(len(catalog.pts),10475,3))
        check('ordered source exposures',np.all(np.diff(catalog.pts)>0))
        if configured_comparison:
            startup_variant=player.variant['id'];startup_frame=player.frame;startup_camera=camera_state(player)
            check('fresh startup uses configured default variant',startup_variant==catalog.config['default_variant'],variant=startup_variant)
            check('fresh startup uses configured source PTS',startup_frame==int(np.argmin(abs(catalog.pts-catalog.config['initial_source_pts']))),source_pts=int(catalog.pts[startup_frame]))
            check('fresh startup uses application comparison framing',player.comparison_framing is not None)
            startup=dict(variant=startup_variant,source_pts=int(catalog.pts[startup_frame]),view=catalog.config.get('initial_view','free'),
                         configured_comparison=list(configured_comparison),test_camera_override=False,framing=player.comparison_framing,panels=[])
            for name in configured_comparison:
                player.load_variant(name)
                check('fresh startup variant switch preserves source PTS and camera',player.frame==startup_frame and same_camera(player,startup_camera),variant=name)
                player.on_render(0.,0.,export=True)
                bounds=comparison_frustum_bounds(player,player.sequence,startup_frame,left_inset=player.comparison_framing.get('left_inset_pixels',0.))
                check('actual application startup contains compared body and gear',bounds['all_body_and_gear_inside'],variant=name,**bounds)
                check('actual application startup leaves compared body and gear clear of controls',bounds['all_body_and_gear_in_view_region'],variant=name)
                startup['panels'].append(dict(variant=name,**bounds))
            player.load_variant(startup_variant)
            report['fresh_startup']=startup
        player.follow=False
        comparison_sequences={}
        if fit_comparison_bounds:
            for name in variants:
                player.load_variant(name);comparison_sequences[name]=player.sequence
        try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
        except OSError:font=ImageFont.load_default()

        for source_seconds in seconds:
            frame=catalog.frame_for_seconds(source_seconds);player.seek(frame);player.set_view(view)
            framing=fit_comparison_camera(player,comparison_sequences,frame) if fit_comparison_bounds else getattr(player,'comparison_framing',None)
            reference_camera=camera_state(player);panels=[];entries=[]
            for name in variants:
                player.load_variant(name)
                if wear_inspection:
                    check('independent diagnostic layers match request',all(player.visible[key]==(key in inspection_layers) for key in ('footbed','cavity','axes')),
                          variant=name,visible_layers=[key for key,value in player.visible.items() if value])
                check('switch preserves source PTS and observing camera',player.frame==frame and same_camera(player,reference_camera),
                      variant=name,source_pts=int(catalog.pts[frame]),source_seconds=float(catalog.seconds[frame]))
                check('candidate remains unaccepted',player.sequence.meta['accuracy_accepted'] is False,variant=name)
                player.on_render(0.,0.,export=False)
                if catalog.source_video:
                    check('actual GUI source image matches selected exposure',player.source_texture_pts==int(catalog.pts[frame]) and not player.source_error,variant=name,source_pts=int(catalog.pts[frame]),source_texture_pts=player.source_texture_pts)
                before=geometry_digest(player.sequence)
                player.on_render(0.,0.,export=True)
                image=player.renderer.get_current_frame_as_image().convert('RGB')
                pixels=np.asarray(image);spread=float(np.std(pixels.astype(float)))
                check('OpenGL framebuffer is nonempty',image.size==(960,720) and spread>1.,variant=name,standard_deviation=spread)
                check('render leaves saved geometry unchanged',geometry_digest(player.sequence)==before,variant=name)
                panels.append(image)
                entries.append(dict(variant=name,source_pts=int(catalog.pts[frame]),source_seconds=float(catalog.seconds[frame]),
                                    window_optimized=bool(player.sequence.state['window_mask'][frame]),geometry_sha256=before,
                                    framebuffer_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),cache=str(player.sequence.path)))
                state=player.sequence.state
                entries[-1]['wear_diagnostics']={key:state[key][frame].tolist() for key in ('cavity_penetration','heel_drift','forefoot_error','blade_tilt_rad','upper_deformation','effective_ankle_relative_rotation') if key in state}
                if fit_comparison_bounds or (framing is not None and name in configured_comparison):
                    bounds=comparison_frustum_bounds(player,player.sequence,frame,left_inset=framing.get('left_inset_pixels',0.),right_inset=framing.get('right_inset_pixels',0.),support_foot=framing.get('support_foot'))
                    check('common camera contains full body and gear',bounds['all_body_and_gear_inside'],variant=name,source_pts=int(catalog.pts[frame]),**bounds)
                    check('common camera keeps full body and gear in unobscured view region',bounds['all_body_and_gear_in_view_region'],variant=name)
                    entries[-1]['framing']=bounds
                entries[-1]['shell_topology']=player.variant.get('shell_topology','saved' if 'boot_surface' in state else 'open_collar')
                entries[-1]['enabled_layers']=[key for key,nodes in player.layers.items() if any(node.enabled for node in nodes)]
                entries[-1]['boot_face_audit']=[]
                for node in player.layers['shoes']:
                    if not node.name.startswith('Boot candidate'):continue
                    points=node.current_vertices;faces=node.faces;used=np.unique(faces)
                    triangles=points[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                    vertex_normals=np.zeros_like(points)
                    for face,normal in zip(faces,normals):
                        for vertex in face:vertex_normals[vertex]+=normal
                    lengths=np.linalg.norm(vertex_normals[used],axis=-1)
                    check('referenced shoe normals finite and nonzero',np.isfinite(lengths).all() and np.all(lengths>0),variant=name,shoe=node.name)
                    details=dict(shoe=node.name,faces=len(faces),unused_vertices=np.setdiff1d(np.arange(len(points)),used).tolist(),minimum_used_normal_length=float(lengths.min()))
                    if boot_section==3:
                        if 'template_version' in state:
                            foot=int(node.name.rsplit(' ',1)[-1])-1
                            check('section uses saved upper vertices',np.allclose(points,state['skate_upper'][frame,foot],atol=1e-7),variant=name,shoe=node.name)
                            axis=state['skate_R'][frame,foot,:,1]
                            side=int(float(axis@(player.scene.camera.position-points.mean(axis=0)))>=0)
                            removed=state['skate_upper_faces'][state['skate_upper_region_ids']==side]
                            same_side=np.array([any(np.array_equal(face,row) for row in removed) for face in faces])
                        else:
                            axis=points[7:].mean(axis=0)-points[:7].mean(axis=0)
                            side=1 if float(axis@(player.scene.camera.position-points.mean(axis=0)))>=0 else 0
                            same_side=(faces>=7).all(axis=1) if side else (faces<7).all(axis=1)
                        check('section removes actual camera-facing side',not np.any(same_side),variant=name,shoe=node.name,removed_side=side)
                        details['removed_camera_side']=side
                    entries[-1]['boot_face_audit'].append(details)
            montage=Image.new('RGB',(960*len(panels),760),(25,30,39));draw=ImageDraw.Draw(montage)
            for index,(panel,entry) in enumerate(zip(panels,entries)):
                x=index*960;montage.paste(panel,(x,40))
                draw.text((x+12,8),f"{entry['variant']} | source {entry['source_seconds']:.3f}s | PTS {entry['source_pts']}",font=font,fill='white')
            path=output/f"compare_{int(catalog.pts[frame])}.png"
            # Retain the preceding run's actual pixels when final exports change.
            # Historical checks keep valid references to their original images.
            if path.exists():path=output/f"compare_{int(catalog.pts[frame])}_{run_stamp}.png"
            montage.save(path)
            report['comparisons'].append(dict(image=str(path),same_pts_same_camera=True,framing=framing,camera={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in reference_camera.items()},panels=entries))

        if playback_seconds is not None:
            report['continuous_playback']=continuous_playback(player,catalog,playback_seconds)
            check('continuous playback preserves every source exposure',report['continuous_playback']['no_skipped_or_repeated_exposures'],
                  exposures=report['continuous_playback']['exposures'],source_first_last_pts=report['continuous_playback']['source_first_last_pts'],playback_fps=report['continuous_playback']['playback_fps'])

        # Exercise the player's actual source clock, independent of wall-clock
        # rendering speed. AITViewer on_render advances int((t-last)*fps).
        player.seek(catalog.frame_for_seconds(seconds[0]));frame=player.frame
        source_pts=int(catalog.pts[frame]);player.seek(frame+1)
        check('next exposure uses source PTS',player.frame==frame+1 and int(catalog.pts[player.frame])==int(catalog.pts[frame+1]),previous_pts=source_pts,next_pts=int(catalog.pts[player.frame]))
        player.seek(frame)
        check('previous exposure returns exact PTS',player.frame==frame and int(catalog.pts[player.frame])==source_pts)
        player.toggle_animation(True);render_time=player._last_frame_rendered_at+.1
        expected_frames=int((render_time-player._last_frame_rendered_at)*player.playback_fps)
        player.on_render(render_time,.1,export=True)
        check('playback render clock advances source frames',expected_frames==5 and player.frame==frame+expected_frames,
              delta_render_seconds=.1,playback_fps=player.playback_fps,advanced_frames=player.frame-frame,source_pts=int(catalog.pts[player.frame]))
        player.toggle_animation(False);paused_frame=player.frame;paused_camera=camera_state(player)
        player.on_render(render_time+.5,.5,export=True)
        check('pause leaves source frame and camera fixed',player.frame==paused_frame and same_camera(player,paused_camera))
        before=geometry_digest(player.sequence);camera=player.scene.camera
        camera.position=camera.target+np.array([2.4,-1.7,1.8])
        player.on_render(render_time+.6,.1,export=True)
        check('programmatic observation angle leaves source geometry fixed',player.frame==paused_frame and geometry_digest(player.sequence)==before)
        camera.is_ortho=True;camera.ortho_size=.9
        player.on_render(render_time+.7,.1,export=True)
        check('programmatic zoom leaves source geometry fixed',player.frame==paused_frame and geometry_digest(player.sequence)==before)
        check('input and candidate files unchanged',{str(path):digest(path) for path in sources}==source_identity)
        report['source_sha256']=source_identity
        report['passed']=True
    except Exception as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if player is not None:
            try:
                player.save_session();player.scene.release();Viewer.close(player)
                imgui.destroy_context(player.imgui_ctx);player.window.destroy()
            except Exception as error:
                report.update(passed=False,cleanup_error=repr(error))
        if global_session is not None:
            unchanged=(digest(global_session) if global_session.exists() else None)==global_identity
            report['checks'].append(dict(check='user global player session unchanged',passed=unchanged,path=str(global_session)))
            if not unchanged:report['passed']=False
        report['elapsed_seconds']=time.perf_counter()-started
        report_path=output/'checks.json'
        if report_path.exists():
            history=output/'history';history.mkdir(exist_ok=True)
            report_path.replace(history/f'checks_{run_stamp}.json')
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--project');parser.add_argument('--data-root')
    parser.add_argument('--variants',nargs='+',default=['ref1102','obs1102','full1102'])
    parser.add_argument('--seconds',nargs='+',type=float,default=[36.02,26.26])
    parser.add_argument('--wear-inspection',action='store_true')
    parser.add_argument('--view',choices=['free','feet','feet_oblique','feet_side','feet_end','support_top','support_side','side','end','top','rink'],default='free')
    parser.add_argument('--boot-section',type=int,choices=[0,1,2,3],default=0,help='0: full shell, 1/2: remove fixed side, 3: remove side facing observing camera')
    parser.add_argument('--inspection-layers',nargs='*',choices=['footbed','cavity','axes'],default=[],help='Independent optional diagnostic overlays; default is a clean physical comparison')
    parser.add_argument('--fit-comparison-bounds',action='store_true',help='Diagnostic orthographic camera fitting all selected candidates at the same source PTS; keeps the requested view direction')
    parser.add_argument('--playback-seconds',nargs=2,type=float,metavar=('START','END'),help='Render every exposure in this source interval through the actual playback clock and record its PTS')
    args=parser.parse_args();report=verify(args.config,args.output,args.project,args.data_root,args.variants,args.seconds,args.wear_inspection,args.view,args.boot_section,args.inspection_layers,args.fit_comparison_bounds,args.playback_seconds)
    print(json.dumps(dict(passed=report['passed'],output=str(Path(args.output).resolve()),opengl=report.get('opengl'),error=report.get('error')),ensure_ascii=False),flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
