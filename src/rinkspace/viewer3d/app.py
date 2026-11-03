"""Native AITViewer player for saved RinkSpace results; never runs a fit."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import imgui
from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.scene.node import Node
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.spheres import Spheres
from .data import ResultCatalog, CachedSequence, shell_triangles, ribbon_triangles, simplify_curve
from .framing import fit_comparison_camera


class FrameObserver(Node):
    def __init__(self, player):
        super().__init__(name='Source clock', n_frames=len(player.catalog.pts), is_selectable=False)
        self.player = player
    def on_frame_update(self):
        self.player.frame_changed(self.current_frame_id)


class ResultsPlayer(Viewer):
    def __init__(self, catalog, variant=None, restore=True, size=(1600, 900)):
        self.catalog = catalog
        C.update_conf(dict(window_type='pyglet', device='cpu', z_up=True, scene_fps=50, playback_fps=50,
            run_animations=False, auto_set_floor=False, auto_set_camera_target=False,
            server_enabled=False, backface_culling=False, shadows_enabled=False,
            background_color=[.12, .15, .19, 1.], font_size=16,
            export_dir=str(catalog.session_root)))
        super().__init__(title='RinkSpace 三维结果播放器 · S01', size=size)
        # pyimgui retains filename bytes on this IO wrapper.
        self._imgui_io = imgui.get_io()
        self._imgui_io.ini_file_name = str(catalog.session_root / 'imgui.ini')
        self.scene.floor.enabled = False
        self.scene.origin.enabled = False
        self.scene.camera_target.enabled = False
        self.scene.trackball.enabled = False
        self.scene.ambient_strength = 1.1
        self._zh_ranges = imgui.get_io().fonts.get_glyph_ranges_chinese_full()
        self.zh_font = imgui.get_io().fonts.add_font_from_file_ttf(
            'C:/Windows/Fonts/msyh.ttc', 18 * self.window.pixel_ratio, glyph_ranges=self._zh_ranges)
        self.imgui.refresh_font_texture()
        self.gui_controls = {'rinkspace': self.gui_rinkspace, 'exit': self.gui_exit,
                             'go_to_frame': self.gui_go_to_frame}
        self.control_panel_width=425
        self.follow = True
        self.speed = 1.
        self.surface_alpha = .72
        self.boot_alpha = .62
        self.wear_inspection = False
        self.boot_section = 0
        self.ice_alpha = .48
        self.visible = dict(surface=True, skeleton=True, shoes=True, trajectory=True, pelvis=True,
                            rink_lines=True, boards=True, ice=True, wear=False,
                            footbed=False, cavity=False, axes=False, collar=False)
        self.layers = {}
        self.variant_nodes = []
        self.last_p = None
        self.observer = None
        self._last_saved = 0.
        self._last_status = None
        self.error = ''
        self.view_label = '自由视角'
        self._build_rink()
        old = self._read_session() if restore else {}
        self.wear_inspection=bool(old.get('wear_inspection',False)) if restore else bool(catalog.config.get('initial_wear_inspection',False))
        self.boot_section=int(old.get('boot_section',0)) if restore else 0
        self.visible['wear']=self.wear_inspection
        self.set_wear_inspection(self.wear_inspection,old.get('inspection_layers',[]) if restore else [])
        selected = variant or old.get('variant', catalog.config['default_variant'])
        if not variant and selected not in {v['id'] for v in catalog.variants}:
            selected=catalog.config['default_variant']
        self.load_variant(selected)
        self.observer = FrameObserver(self)
        self.scene.add(self.observer)
        initial_pts = old.get('source_pts', catalog.config['initial_source_pts'])
        self.scene.current_frame_id = int(np.argmin(abs(catalog.pts - initial_pts)))
        self.last_p = self.sequence.state['P'][self.frame].copy()
        if restore and old.get('camera'):self.set_view('free',fit_comparison=False)
        else:self.set_view(catalog.config.get('initial_view','free') if not restore else 'free')
        if old.get('camera'):
            try:
                c = old['camera'];cam = self.scene.camera
                for attr in ('position', 'target', 'up'):setattr(cam, attr, np.asarray(c[attr]))
                cam.is_ortho = bool(c['is_ortho']);cam.ortho_size = float(c['ortho_size'])
                self.follow = bool(old.get('follow', True));self.view_label = '上次视角'
            except (ValueError, KeyError, TypeError):pass
        self.save_session()

    @property
    def frame(self):return int(self.scene.current_frame_id)

    def _read_session(self):
        try:return json.loads((self.catalog.session_root/'player_state.json').read_text(encoding='utf8'))
        except (FileNotFoundError, ValueError):return {}

    def _build_rink(self):
        geo = self.catalog.scene_geometry();bottom, top = geo['bottom'], geo['top']
        boundary = bottom[:-1] if np.allclose(bottom[0], bottom[-1]) else bottom
        center = np.array([boundary[:,0].mean(), boundary[:,1].mean(), 0.])
        vertices = np.vstack([center, boundary]);n = len(boundary)
        faces = np.asarray([(0, i+1, (i+1)%n+1) for i in range(n)], np.int32)
        if np.cross(vertices[faces[0,1]]-center, vertices[faces[0,2]]-center)[2]<0:faces=faces[:,::-1].copy()
        self.ice = Meshes(vertices, faces, name='Frozen rink ice Z=0', color=(.76,.85,.91,self.ice_alpha), pickable=False, cast_shadow=False)
        self.layers['ice']=[self.ice]
        assert len(bottom)==len(top)
        boards = Meshes(np.vstack([bottom,top]), ribbon_triangles(len(bottom)),
            name='Frozen board geometry', color=(.8,.83,.87,.36), pickable=False, cast_shadow=False)
        self.layers['boards']=[boards]
        groups = {'ice': [], 'board': [], 'artwork': []}
        for cut in geo['cuts']:
            p = simplify_curve(geo['xyz'][cut['begin']:cut['end']])
            if len(p)<2:continue
            group = 'artwork' if cut['kind']=='installed_artwork_polyline' else ('ice' if cut['kind'].startswith('ice_') else 'board')
            groups[group].append(np.stack([p[:-1],p[1:]],axis=1).reshape(-1,3))
        line_nodes=[]
        for group,color,radius in [('ice',(.22,.57,.88,1),.009),('board',(.4,.7,.87,1),.008),('artwork',(.6,.77,.56,1),.006)]:
            if groups[group]:line_nodes.append(Lines(np.concatenate(groups[group]),mode='lines',r_base=radius,color=color,name='Frozen '+group+' lines',is_selectable=False))
        self.layers['rink_lines']=line_nodes
        self.scene.add(self.ice,boards,*line_nodes)

    def load_variant(self, name):
        definition = self.catalog.variant(name)
        path,_ = self.catalog.cache_path(definition)
        if not (path/'metadata.json').is_file():
            from .prepare import prepare_variant
            path = prepare_variant(self.catalog,definition)
        sequence = CachedSequence(path)
        np.testing.assert_array_equal(sequence.state['pts'], self.catalog.pts)
        frame = self.frame
        s = sequence.state
        last_p = s['P'][frame].copy()
        connections = np.asarray([(int(p),j) for j,p in enumerate(s['parents']) if j and p>=0],np.int32)
        dynamic='boot_surface' in s
        topology=definition.get('shell_topology','saved' if dynamic else 'open_collar')
        if dynamic and topology!='saved':raise ValueError('A saved dynamic boot surface requires its fixed saved topology')
        if not dynamic and topology not in ('open_collar','closed_hull'):raise ValueError('Historical boot requires open_collar or closed_hull topology')
        boot_surface=s.get('boot_surface',s['gear'][:,:,:14])
        shells={}
        for foot in range(2):
            if s['gear'].shape[2]!=48:continue
            if dynamic:faces=s['boot_faces'][foot] if s['boot_faces'].ndim==3 else s['boot_faces']
            elif topology=='closed_hull':faces=shell_triangles(boot_surface[0,foot],topology='closed_hull')
            else:faces=shell_triangles(boot_surface[0,foot])
            section=getattr(self,'boot_section',0)
            if section:
                if section==3:
                    # Remove the actual side facing this observing camera. Use
                    # side centroids in the current frame, never alter vertices.
                    points=boot_surface[frame,foot];center=points.mean(axis=0)
                    axis=points[7:].mean(axis=0)-points[:7].mean(axis=0)
                    camera_vector=np.asarray(self.scene.camera.position)-center
                    remove_first=float(axis@camera_vector)<0
                else:remove_first=section==1
                side=(faces<7).all(axis=1) if remove_first else (faces>=7).all(axis=1)
                faces=faces[~side]
            shells[foot]=faces
        layers = self.layers.copy()
        nodes = []

        def keep(node):
            nodes.append(node)
            return node

        try:
            body=keep(Meshes(sequence.vertices,sequence.faces,name='Saved full SMPL-X surface',
                color=(.58,.69,.84,self.surface_alpha),pickable=False,cast_shadow=False))
            skeleton=keep(Skeletons(s['joints'],connections,radius=.009,color=(.55,1.,.28,1),name='Native skeleton'))
            layers['surface']=[body];layers['skeleton']=[skeleton]
            equipment=[]
            for foot,color in enumerate([(.98,.43,.14,1),(.12,.85,1.,1)]):
                gear=s['gear'][:,foot]
                if gear.shape[1]==48:
                    alpha=.40 if getattr(self,'wear_inspection',False) else getattr(self,'boot_alpha',.62)
                    shell=keep(Meshes(boot_surface[:,foot],shells[foot],color=(*color[:3],alpha),
                        name='Boot candidate '+str(foot+1),pickable=False,cast_shadow=False))
                    strip=keep(Meshes(gear[:,14:],ribbon_triangles(17),color=color,
                        name='Blade lower edge strip '+str(foot+1),pickable=False,cast_shadow=False))
                    equipment.extend([shell,strip])
                    for offset in (14,31):equipment.append(keep(Lines(gear[:,offset:offset+17],r_base=.003,color=color,name='Blade edge '+str(foot+1),is_selectable=False)))
                else:equipment.append(keep(Lines(gear,r_base=.004,color=color,name='Old blade proxy '+str(foot+1),is_selectable=False)))
            layers['shoes']=equipment
            footbed_nodes=[];cavity_nodes=[];axis_nodes=[];collar_nodes=[]
            if topology!='closed_hull' and s['gear'].shape[2]==48:
                loop=np.array([4,5,12,11,4])
                for foot in range(2):
                    collar_nodes.append(keep(Lines(boot_surface[:,foot,loop],r_base=.004,
                        color=(1.,.95,.2,1) if foot==0 else (1.,.3,.85,1),
                        name='Open collar boundary '+str(foot+1),is_selectable=False)))
            if 'footbed' in s:
                for foot in range(2):
                    points=s['footbed'][:,foot];loop=np.concatenate([points,points[:,:1]],axis=1)
                    footbed_nodes.append(keep(Lines(loop,r_base=.0025,color=(1.,.9,.25,1),name='Footbed proxy '+str(foot+1),is_selectable=False)))
            if 'cavity' in s:
                for foot in range(2):
                    faces=s['cavity_faces'][foot] if s['cavity_faces'].ndim==3 else s['cavity_faces']
                    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
                    points=s['cavity'][:,foot][:,edges].reshape(len(self.catalog.pts),-1,3)
                    cavity_nodes.append(keep(Lines(points,mode='lines',r_base=.0015,color=(.75,.55,1.,.7),name='Cavity prior '+str(foot+1),is_selectable=False)))
            if 'anchors' in s:
                for index,label in enumerate(s['anchor_labels']):
                    color=[(1.,.3,.4,1),(.35,1.,.4,1),(1.,.8,.2,1),(.2,.85,1.,1)][index%4]
                    footbed_nodes.append(keep(Spheres(s['anchors'][:,:,index],radius=.007,color=color,name='Anchor '+str(label),is_selectable=False)))
            if 'axes' in s:
                for index,label in enumerate(s['axis_labels']):
                    points=s['axes'][:,:,index].reshape(len(self.catalog.pts),-1,3)
                    color=[(.95,.25,.25,1),(.3,.95,.3,1),(.3,.6,1.,1)][index%3]
                    axis_nodes.append(keep(Lines(points,mode='lines',r_base=.0025,color=color,name='Axis '+str(label),is_selectable=False)))
            layers.update(wear=[],footbed=footbed_nodes,cavity=cavity_nodes,axes=axis_nodes,collar=collar_nodes)
            trajectory=keep(Lines(s['G'],r_base=.018,color=(1.,.75,.12,1),name='Whole S01 pelvis vertical projection',is_selectable=False))
            layers['trajectory']=[trajectory]
            vertical=keep(Lines(np.stack([s['P'],s['G']],axis=1),r_base=.007,color=(1.,.27,.67,1),name='P to G world vertical',is_selectable=False))
            pelvis=keep(Spheres(s['P'][:,None],radius=.032,color=(1.,.27,.67,1),name='P pelvis',is_selectable=False))
            ground=keep(Spheres(s['G'][:,None],radius=.04,color=(1.,.78,.12,1),name='G pelvis projection',is_selectable=False))
            layers['pelvis']=[vertical,pelvis,ground]
            for key in ('surface','skeleton','shoes','trajectory','pelvis','wear','footbed','cavity','axes','collar'):
                for node in layers[key]:
                    node.current_frame_id=frame
                    node.enabled=self.visible.get(key,False)
            # Node.add resets enabled by default; pass each prepared layer value.
            # Keep every old node alive until all additions have succeeded.
            for node in nodes:self.scene.add(node,enabled=node.enabled)
        except Exception as error:
            # AITViewer 1.14.2 Node.remove calls release(), including children.
            # Constructors/frame preparation allocate no OpenGL resources.
            for node in reversed(nodes):
                try:
                    if node in self.scene.nodes:self.scene.remove(node)
                    else:node.release()
                except Exception as cleanup_error:
                    error.add_note('Uncommitted viewer node cleanup failed: '+str(cleanup_error))
            raise

        old_nodes = self.variant_nodes
        self.sequence=sequence;self.variant=definition
        self.layers=layers;self.variant_nodes=nodes;self.last_p=last_p
        if old_nodes:
            self.scene.remove(*old_nodes)
            # Clear the global normals LRU only after a successful switch.
            Meshes.compute_vertex_and_face_normals.cache_clear()
        self.error=''

    def apply_visibility(self):
        for key,nodes in self.layers.items():
            for node in nodes:node.enabled=self.visible[key]

    def set_wear_inspection(self,enabled,diagnostic_layers=()):
        """A clean physical comparison; optional proxies are separate layers."""
        self.wear_inspection=bool(enabled)
        if enabled:
            if not hasattr(self,'_pre_wear_visibility'):self._pre_wear_visibility=self.visible.copy()
            self.visible.update(surface=True,shoes=True,skeleton=False,trajectory=False,pelvis=False,
                                rink_lines=False,boards=False,ice=False,wear=True,collar=True,
                                footbed='footbed' in diagnostic_layers,cavity='cavity' in diagnostic_layers,axes='axes' in diagnostic_layers)
        else:
            if hasattr(self,'_pre_wear_visibility'):
                self.visible.update(self._pre_wear_visibility);del self._pre_wear_visibility
            self.visible.update(wear=False,collar=False,footbed=False,cavity=False,axes=False)
        self.apply_visibility()
        for node in self.layers.get('shoes',[]):
            if node.name.startswith('Boot candidate'):node.color=(*node.color[:3],.40 if enabled else self.boot_alpha)

    def frame_changed(self, frame):
        p=self.sequence.state['P'][frame]
        if self.follow and self.last_p is not None:
            displacement=p-self.last_p
            cam=self.scene.camera
            cam.position=cam.position+displacement;cam.target=cam.target+displacement
        self.last_p=p.copy()

    def seek(self, frame):
        self.toggle_animation(False)
        self.scene.current_frame_id=int(np.clip(frame,0,len(self.catalog.pts)-1))
        self.save_session()

    def comparison_sequences(self):
        """Reuse saved display sequences without creating render nodes or fitting."""
        names=self.catalog.config.get('comparison_variants',[])
        cache=getattr(self,'_comparison_sequences',{})
        self._comparison_sequences=cache
        sequences={}
        for name in names:
            if name==self.variant['id']:
                sequence=self.sequence
            else:
                definition=self.catalog.variant(name);path,_=self.catalog.cache_path(definition)
                sequence=cache.get(name)
                if sequence is None or sequence.path!=path:
                    if not (path/'metadata.json').is_file():
                        from .prepare import prepare_variant
                        path=prepare_variant(self.catalog,definition)
                    sequence=CachedSequence(path)
            np.testing.assert_array_equal(sequence.state['pts'],self.catalog.pts)
            cache[name]=sequence;sequences[name]=sequence
        return sequences

    def set_view(self, view, *, fit_comparison=True):
        cam=self.scene.camera;p=self.sequence.state['P'][self.frame].copy();p[2]=.8
        cam.up=np.array([0.,0.,1.]);cam.is_ortho=False
        if view=='top':
            cam.position=p+[0,0,10];cam.target=p;cam.up=np.array([0.,1.,0.]);cam.is_ortho=True;cam.ortho_size=2.2
        elif view=='side':
            cam.position=p+[0,-8,0];cam.target=p;cam.is_ortho=True;cam.ortho_size=1.5
        elif view=='end':
            cam.position=p+[8,0,0];cam.target=p;cam.is_ortho=True;cam.ortho_size=1.5
        elif view=='rink':
            cam.position=np.array([42.,-48.,40.]);cam.target=np.zeros(3);self.follow=False
        elif view in ('feet','feet_oblique','feet_side'):
            p=self.sequence.state['gear'][self.frame].reshape(-1,3).mean(axis=0)
            if view=='feet_oblique':offset=[.40,-.65,.95]
            elif view=='feet_side':
                points=self.sequence.state.get('boot_surface',self.sequence.state['gear'][:,:,:14])[self.frame]
                axis=np.mean(points[:,7:]-points[:,:7],axis=(0,1))
                if np.linalg.norm(axis)<1e-8:axis=np.mean(points[0,7:]-points[0,:7],axis=0)
                axis/=np.linalg.norm(axis)
                offset=axis*.95+np.array([0.,0.,.16])
            else:offset=[.55,-.75,.4]
            cam.position=p+offset;cam.target=p;cam.is_ortho=True;cam.ortho_size=.34
        else:cam.position=p+[2.0,-2.7,1.3];cam.target=p
        self.comparison_framing=None
        if fit_comparison and view in ('free','top','side','end') and self.catalog.config.get('comparison_variants'):
            self.comparison_framing=fit_comparison_camera(self,self.comparison_sequences(),self.frame,
                left_inset=min(getattr(self,'control_panel_width',425)+28,self.window_size[0]-1))
        self.view_label={'top':'俯视','side':'侧视 XZ','end':'侧视 YZ','rink':'全冰场','free':'自由视角','feet':'足鞋近看','feet_oblique':'鞋口斜俯视','feet_side':'足鞋同侧剖切视角'}[view]
        self.save_session()

    def save_session(self):
        if not hasattr(self,'variant'):return
        cam=self.scene.camera;frame=self.frame
        state=dict(schema='rinkspace.viewer-session.v1',pid=os.getpid(),variant=self.variant['id'],
            local_frame=frame,source_pts=int(self.catalog.pts[frame]),source_seconds=float(self.catalog.seconds[frame]),
            playing=bool(self.run_animations),follow=self.follow,visible=self.visible,
            wear_inspection=self.wear_inspection,boot_section=self.boot_section,
            inspection_layers=[key for key in ('footbed','cavity','axes') if self.visible.get(key,False)],
            camera=dict(position=cam.position.tolist(),target=cam.target.tolist(),up=cam.up.tolist(),
                        is_ortho=bool(cam.is_ortho),ortho_size=float(cam.ortho_size)),
            geometry_cache=str(self.sequence.path),accuracy_accepted=False)
        path=self.catalog.session_root/'player_state.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');temporary.replace(path)
        self._last_saved=time.monotonic()

    def on_render(self,render_time,frame_time,**kwargs):
        super().on_render(render_time,frame_time,**kwargs)
        if time.monotonic()-self._last_saved>1:self.save_session()
        if getattr(self, "verification", None) and not kwargs.get("export", False):self.verification.tick(self,render_time)

    def close(self):
        if hasattr(self,'variant'):self.save_session()
        # Keep the context alive until Viewer.run finishes its last buffer swap.
        self.window.is_closing = True

    def center_view_on_selection(self):
        if hasattr(self,'sequence'):self.set_view('free')

    def gui_rinkspace(self):
        imgui.set_next_window_position(14,14,condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.control_panel_width,self.window.size[1]-28,condition=imgui.FIRST_USE_EVER)
        with imgui.font(self.zh_font):
            imgui.begin('RinkSpace 三维结果播放器')
            imgui.text('S01  |  原片 22.120–44.960 秒  |  50 帧/秒')
            imgui.text_wrapped('冰场：'+self.catalog.config.get('scene_label','本轮人体实验配套的冻结场景'))
            imgui.text_colored('当前人体结果：候选，尚未验收',1.,.78,.32)
            imgui.separator()
            labels=[q['label'] for q in self.catalog.variants]
            index=next(i for i,v in enumerate(self.catalog.variants) if v['id']==self.variant['id'])
            imgui.set_next_item_width(-1)
            changed,index=imgui.combo('##result',index,labels)
            if changed:
                try:self.load_variant(self.catalog.variants[index]['id']);self.save_session()
                except Exception as error:self.error=str(error)
            if self.error:imgui.text_wrapped('加载失败：'+self.error)
            f=self.frame;seconds=self.catalog.seconds[f]
            imgui.text('当前 candidate id：'+self.variant['id'])
            window=self.sequence.state.get('window_mask')
            binding=self.sequence.state.get('footwear_binding_applied_mask')
            refined=self.sequence.state.get('body_refinement_window_mask')
            if binding is not None:
                imgui.text_wrapped(self.catalog.config.get('footwear_scope_label','装备配准作用范围按保存的 mask 显示。'))
                imgui.text_colored('本帧装备：共享穿合绑定已作用' if bool(binding[f]) else '本帧装备：沿用保存的参考几何',1.,.78,.32)
                if refined is not None:
                    reference=self.variant.get('body_reference_label','fullbody1102')
                    imgui.text('本帧人体：本轮窗口优化候选' if bool(refined[f]) else '本帧人体：沿用 '+reference)
                    intervals=self.variant.get('body_refinement_intervals_seconds',self.catalog.config.get('body_refinement_intervals_seconds',[]))
                    if intervals:imgui.text_wrapped('人体新增优化仅原片 '+ '、'.join('%g–%g 秒'%tuple(interval) for interval in intervals)+'；其他人体沿用 '+reference+'。')
                imgui.text_wrapped('整节目人体尚未完成；鞋内配合与接触仍未验收。')
            elif window is not None:
                if bool(window[f]):imgui.text_colored('本帧：本轮窗口内优化候选，未核验',1.,.78,.32)
                else:imgui.text_colored('本帧：段外旧 J1 参考，未参与本轮优化',1.,.7,.35)
                imgui.text_wrapped('全 S01 时间轴用于对照；本轮只优化标记窗口，整节目人体尚未完成。')
            else:imgui.text('本帧：历史保存候选，未参与本轮全身优化')
            imgui.text('原片时间：%06.3f 秒  |  PTS %d'%(seconds,self.catalog.pts[f]))
            imgui.text('S01 帧：%d / %d'%(f+1,len(self.catalog.pts)))
            if imgui.button('上一帧  ,'):self.seek(f-1)
            imgui.same_line()
            if imgui.button('暂停  空格' if self.run_animations else '播放  空格'):self.toggle_animation(not self.run_animations)
            imgui.same_line()
            if imgui.button('下一帧  .'):self.seek(f+1)
            imgui.set_next_item_width(-1)
            changed,value=imgui.slider_int('##frame',f+1,1,len(self.catalog.pts),format='%d')
            if changed:self.seek(value-1)
            imgui.set_next_item_width(150)
            changed,value=imgui.input_int('跳到 S01 帧',f+1)
            if changed:self.seek(value-1)
            imgui.set_next_item_width(150)
            changed,index=imgui.combo('播放速度',[.25,.5,1.,2.].index(self.speed),['0.25 倍','0.5 倍','1 倍','2 倍'])
            if changed:self.speed=[.25,.5,1.,2.][index];self.playback_fps=50*self.speed
            _,self.follow=imgui.checkbox('播放时跟随运动员（保留观察角度）',self.follow)
            imgui.separator()
            for i,(label,key) in enumerate([('自由 / 回到人体  X','free'),('侧视 XZ','side'),('侧视 YZ','end'),('俯视','top'),('全冰场','rink')]):
                if i in (1,3,4):imgui.same_line()
                if imgui.button(label):self.set_view(key)
            imgui.text('左键拖动：旋转  |  右键拖动：平移')
            imgui.text('滚轮：缩放；暂停后也可自由观察')
            imgui.separator()
            for label,key in [('人体表面','surface'),('人体骨架','skeleton'),('鞋和冰刀','shoes'),('整段骨盆投影轨迹','trajectory'),('骨盆 P、地面投影 G 与竖线','pelvis'),('冰场线和广告','rink_lines'),('围挡','boards'),('冰面','ice')]:
                changed,self.visible[key]=imgui.checkbox(label,self.visible[key])
                if changed:self.apply_visibility()
            imgui.set_next_item_width(175)
            changed,self.surface_alpha=imgui.slider_float('人体透明度',self.surface_alpha,.08,1.,format='%.2f')
            if changed:self.layers['surface'][0].color=(.58,.69,.84,self.surface_alpha)
            imgui.set_next_item_width(175)
            changed,self.ice_alpha=imgui.slider_float('冰面透明度',self.ice_alpha,.02,1.,format='%.2f')
            if changed:self.ice.color=(.76,.85,.91,self.ice_alpha)
            imgui.separator()
            s=self.sequence.state;gear=s['gear'][f]
            changed,inspection=imgui.checkbox('穿戴检查：人体、半透明鞋与鞋口',self.wear_inspection)
            if changed:self.set_wear_inspection(inspection)
            if imgui.button('足鞋近看'):self.set_view('feet')
            imgui.same_line()
            if imgui.button('鞋口斜俯视'):self.set_view('feet_oblique')
            imgui.same_line()
            if imgui.button('同侧视角'):self.set_view('feet_side')
            for label,key in [('足床与锚点','footbed'),('内腔线框','cavity'),('小腿/下鞋/刀具坐标轴','axes'),('标色鞋口边界','collar')]:
                changed,self.visible[key]=imgui.checkbox(label,self.visible[key])
                if changed:self.apply_visibility()
            imgui.set_next_item_width(175)
            changed,section=imgui.combo('鞋壳剖面',self.boot_section,['不隐藏','隐藏侧面 0','隐藏侧面 1','隐藏朝向相机侧面'])
            if changed:
                previous=self.boot_section;self.boot_section=section
                try:self.load_variant(self.variant['id'])
                except Exception as error:self.boot_section=previous;self.error=str(error)
            topology=self.variant.get('shell_topology','saved' if 'boot_surface' in s else 'open_collar')
            imgui.text('鞋面拓扑：'+{'closed_hull':'历史封闭凸包','open_collar':'22 面开放鞋口','saved':'保存的固定拓扑 / 动态上帮'}.get(topology,topology))
            if self.wear_inspection:
                if 'cavity' not in s:imgui.text_wrapped('本结果没有保存内腔、足床与锚点，开放鞋口本身不代表穿合完成。')
                if 'anchor_labels' in s:imgui.text_wrapped('锚点（红/绿/黄/青）：'+' / '.join(str(x) for x in s['anchor_labels']))
                if 'axis_labels' in s:imgui.text_wrapped('小腿 / 下鞋 / 刀具轴：XYZ 依次为红 / 绿 / 蓝')
                for key,label in [('cavity_penetration','内腔越界'),('heel_drift','足跟偏移'),('forefoot_error','前掌错位')]:
                    if key in s:imgui.text('%s 0/1：%.2f / %.2f 条件毫米'%(label,*(s[key][f]*1000)))
                if 'effective_ankle_relative_rotation' in s:
                    trace=np.trace(s['effective_ankle_relative_rotation'][f],axis1=-2,axis2=-1)
                    angles=np.degrees(np.arccos(np.clip((trace-1)/2,-1,1)))
                    imgui.text('有效小腿—足段转角：%.1f / %.1f 度'%tuple(angles))
                if 'blade_tilt_rad' in s:imgui.text('世界刀具倾角：%.1f / %.1f 度'%tuple(np.degrees(s['blade_tilt_rad'][f])))
                if 'upper_deformation' in s:
                    for foot in range(2):imgui.text('上帮 %d 系数：%s'%(foot,' / '.join('%.4g'%x for x in s['upper_deformation'][f,foot])))
                imgui.text_wrapped('内腔/间隙为工程先验；有效转角不等同解剖背屈/内翻，形变系数不代表材料刚度。')
            imgui.text('粉色 P：骨盆；黄色 G：骨盆竖直投影')
            imgui.text('橙 / 青：两只鞋刀，左右身份尚未确认')
            imgui.text('接冰状态：未知（显示程序不强制贴冰）')
            if gear.shape[1]==5:imgui.text_colored('此旧版只有刀刃代理，没有拟合鞋壳',1.,.7,.35)
            else:imgui.text('鞋刀：保存的刚性鞋壳与刀刃下缘')
            if bool(s['source_lower_body_inferred'][f]):imgui.text_colored('本帧下肢观测不足，含推断',1.,.45,.35)
            if bool(s['source_identity_suspect'][f]):imgui.text_colored('本帧存在左右身份疑点',1.,.45,.35)
            imgui.text('统一世界坐标：Z 向上；单位：米')
            imgui.text('显示尺度：%.9g 米 / 场景单位'%self.catalog.scale)
            imgui.text_wrapped('体型沿用原实验的近似身高假设。三维显示用于查错，不代表鞋刀或接冰已准确。')
            imgui.end()




def main():
    parser=argparse.ArgumentParser(description='RinkSpace saved S01 result player')
    parser.add_argument('--project');parser.add_argument('--data-root');parser.add_argument('--config')
    parser.add_argument('--variant');parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--self-check',action='store_true',help='Verify this application against its own rendered frames; exits after checks')
    args=parser.parse_args()
    catalog=ResultCatalog(args.project,args.data_root,args.config)
    player=ResultsPlayer(catalog,args.variant,restore=not args.fresh)
    if args.self_check:
        from .verify import PlayerVerification
        player.verification=PlayerVerification(catalog)
    player.run()


if __name__=='__main__':main()
