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
from .framing import fit_comparison_camera, comparison_panel_sizes
from .imgui_buffers import install_dynamic_buffers
from .source import SourceVideo, SupportObservations, region_outline, source_value, observation_used, load_world_regions


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
        super().__init__(title='RinkSpace 三维结果播放器 · '+catalog.config.get('shot',''), size=size)
        install_dynamic_buffers(self.imgui)
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
        # Viewer renders 3D before GUI callbacks. Read its source frame before
        # controls can seek/switch the state intended for the next render.
        self.gui_controls = {'source': self.gui_source, 'rinkspace': self.gui_rinkspace, 'exit': self.gui_exit,
                             'go_to_frame': self.gui_go_to_frame}
        panel_sizes=comparison_panel_sizes(self.window_size[0],bool(catalog.source_video))
        self.control_panel_width=panel_sizes['control_width']
        self.source_panel_width=panel_sizes['source_width']
        self.follow = True
        self.speed = 1.
        self.surface_alpha = .72
        self.boot_alpha = .62
        self.wear_inspection = False
        self.boot_section = 0
        self.ice_alpha = .48
        self.visible = dict(surface=True, skeleton=True, shoes=True, trajectory=True, pelvis=True,
                            rink_lines=True, boards=True, ice=True, wear=False,
                            footbed=False, cavity=False, axes=False, collar=False,
                            contact=True, skate_reference=False, com=True, blade_material=True)
        for key,value in catalog.config.get('initial_visible',{}).items():
            if key in self.visible:self.visible[key]=bool(value)
        self.layers = {}
        self.variant_nodes = []
        self.last_p = None
        self.observer = None
        self._last_saved = 0.
        self._last_status = None
        self.error = ''
        self.view_label = '自由视角'
        self.view_mode = 'free'
        self.source_reader = None
        self.source_texture = None
        self.source_texture_pts = None
        self.source_error = ''
        self.source_regions = (SupportObservations(catalog.support_observations,catalog.time_base,shot=catalog.config.get('source_shot'))
                               if catalog.support_observations else None)
        self.world_region_nodes = []
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
        if self.catalog.support_world_regions:
            for row in load_world_regions(self.catalog.support_world_regions):
                points=np.asarray(row['world_polyline_m'],float)
                if not np.array_equal(points[0],points[-1]):points=np.r_[points,points[:1]]
                color=(1.,.64,.12,1.) if row['role']=='fit' else (.75,.4,1.,1.)
                node=Lines(points,r_base=.0015,color=color,
                    name='Source nominal-plane conditional range '+str(row.get('id','')),is_selectable=False)
                self.world_region_nodes.append((row,node));self.scene.add(node,enabled=False)

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
        versioned_skate='template_version' in s
        dynamic=versioned_skate or 'boot_surface' in s
        topology=definition.get('shell_topology','saved' if dynamic else 'open_collar')
        if dynamic and topology!='saved':raise ValueError('A saved dynamic boot surface requires its fixed saved topology')
        if not dynamic and topology not in ('open_collar','closed_hull'):raise ValueError('Historical boot requires open_collar or closed_hull topology')
        boot_surface=s['skate_upper'] if versioned_skate else s.get('boot_surface',s['gear'][:,:,:14])
        shells={}
        for foot in range(2):
            if s['gear'].shape[2]!=48:continue
            if versioned_skate:faces=s['skate_upper_faces']
            elif dynamic:faces=s['boot_faces'][foot] if s['boot_faces'].ndim==3 else s['boot_faces']
            elif topology=='closed_hull':faces=shell_triangles(boot_surface[0,foot],topology='closed_hull')
            else:faces=shell_triangles(boot_surface[0,foot])
            section=getattr(self,'boot_section',0)
            if section:
                if versioned_skate:
                    region=0 if section==1 else 1
                    if section==3:
                        camera_vector=np.asarray(self.scene.camera.position)-boot_surface[frame,foot].mean(axis=0)
                        region=int(float(s['skate_R'][frame,foot,:,1]@camera_vector)>=0)
                    faces=faces[s['skate_upper_region_ids']!=region]
                elif section==3:
                    # Remove the actual side facing this observing camera. Use
                    # side centroids in the current frame, never alter vertices.
                    points=boot_surface[frame,foot];center=points.mean(axis=0)
                    axis=points[7:].mean(axis=0)-points[:7].mean(axis=0)
                    camera_vector=np.asarray(self.scene.camera.position)-center
                    remove_first=float(axis@camera_vector)<0
                else:remove_first=section==1
                if not versioned_skate:
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
                if versioned_skate:
                    alpha=getattr(self,'boot_alpha',.40 if getattr(self,'wear_inspection',False) else .62)
                    equipment.append(keep(Meshes(boot_surface[:,foot],shells[foot],color=(*color[:3],alpha),
                        name='Boot candidate '+str(foot+1),pickable=False,cast_shadow=False)))
                    parts=s['skate_lower_face_parts']
                    face_colors=np.tile((*color[:3],1.),(len(parts),1))
                    for part in np.unique(parts):
                        if 'pick' in str(part):face_colors[parts==part]=(.95,.83,.38,1.)
                        elif 'runner' in str(part) or 'blade' in str(part):face_colors[parts==part]=(.78,.84,.91,1.)
                        elif 'plate' in str(part) or 'pillar' in str(part) or 'pedestal' in str(part):face_colors[parts==part]=(.56,.63,.71,1.)
                    equipment.append(keep(Meshes(s['skate_lower'][:,foot],s['skate_lower_faces'],face_colors=face_colors,
                        flat_shading=True,name='Saved connected skate solids '+str(foot+1),pickable=False,cast_shadow=False)))
                    for edge in range(2):
                        equipment.append(keep(Lines(s['skate_edge'][:,foot,edge],r_base=.0012,
                            color=(1.,.95,.25,1.) if edge==0 else (.85,.65,1.,1.),
                            name='Exact blade edge local '+str(edge)+' foot '+str(foot+1),is_selectable=False)))
                elif gear.shape[1]==48:
                    alpha=getattr(self,'boot_alpha',.40 if getattr(self,'wear_inspection',False) else .62)
                    shell=keep(Meshes(boot_surface[:,foot],shells[foot],color=(*color[:3],alpha),
                        name='Boot candidate '+str(foot+1),pickable=False,cast_shadow=False))
                    strip=keep(Meshes(gear[:,14:],ribbon_triangles(17),color=color,
                        name='Blade lower edge strip '+str(foot+1),pickable=False,cast_shadow=False))
                    equipment.extend([shell,strip])
                    for offset in (14,31):equipment.append(keep(Lines(gear[:,offset:offset+17],r_base=.003,color=color,name='Blade edge '+str(foot+1),is_selectable=False)))
                else:equipment.append(keep(Lines(gear,r_base=.004,color=color,name='Old blade proxy '+str(foot+1),is_selectable=False)))
            layers['shoes']=equipment
            footbed_nodes=[];cavity_nodes=[];axis_nodes=[];collar_nodes=[]
            contact_nodes=[];reference_nodes=[];com_nodes=[];material_nodes=[]
            if 'com_world_mid' in s:
                cross=np.eye(3)[:,None,:]*np.array([-1.,1.])[None,:,None]
                points=s['com_world_mid'][:,None,:]+cross.reshape(1,6,3)*.035
                com_nodes.append(keep(Lines(points,mode='lines',r_base=.0035,
                    color=(.45,1.,.22,1.),name='Saved approximate mass-model COM cross; not measured COM',is_selectable=False)))
            if 'blade_material_world_mid' in s:
                for index,(label,color) in enumerate((('rear',(.3,.45,1.,1.)),('middle',(1.,.86,.15,1.)),('front',(.94,.98,1.,1.)))):
                    material_nodes.append(keep(Spheres(s['blade_material_world_mid'][:,:,index],radius=.005,
                        color=color,name='Saved fixed blade material station '+label,is_selectable=False)))
            if versioned_skate:
                for foot in range(2):
                    diamond=np.array([[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,-1]],float)
                    edges=np.array([(pole,side) for pole in (0,5) for side in range(1,5)]+[(1,2),(2,3),(3,4),(4,1)])
                    marker=s['skate_witness'][:,foot,None,:]+diamond[edges].reshape(1,-1,3)*.008
                    contact_nodes.append(keep(Lines(marker,mode='lines',r_base=.0015,
                        color=(1.,.2,.6,1.),name='Nonpenetration witness '+str(foot+1),is_selectable=False)))
                    reference_nodes.append(keep(Lines(s['skate_reference'][:,foot],r_base=.002,
                        color=(.35,1.,.55,1.) if foot==0 else (.4,.65,1.,1.),
                        name='Fixed blade material reference trajectory '+str(foot+1),is_selectable=False)))
                    reference_nodes.append(keep(Spheres(s['skate_reference'][:,foot,None],radius=.003,
                        color=(.35,1.,.55,1.),name='Fixed blade material reference '+str(foot+1),is_selectable=False)))
                    if 'skate_collar_boundary' in s:
                        ids=s['skate_collar_boundary'];loop=np.r_[ids,ids[:1]]
                        collar_nodes.append(keep(Lines(boot_surface[:,foot,loop],r_base=.002,
                            color=(1.,.95,.2,1.),name='Saved open collar boundary '+str(foot+1),is_selectable=False)))
            elif topology!='closed_hull' and s['gear'].shape[2]==48:
                loop=np.array([4,5,12,11,4])
                for foot in range(2):
                    collar_nodes.append(keep(Lines(boot_surface[:,foot,loop],r_base=.004,
                        color=(1.,.95,.2,1) if foot==0 else (1.,.3,.85,1),
                        name='Open collar boundary '+str(foot+1),is_selectable=False)))
            footbed_key='skate_footbed' if versioned_skate else 'footbed'
            if footbed_key in s:
                for foot in range(2):
                    points=s[footbed_key][:,foot];loop=np.concatenate([points,points[:,:1]],axis=1)
                    footbed_nodes.append(keep(Lines(loop,r_base=.0025,color=(1.,.9,.25,1),name='Footbed proxy '+str(foot+1),is_selectable=False)))
            cavity_key='skate_cavity' if versioned_skate else 'cavity'
            if cavity_key in s:
                for foot in range(2):
                    faces=s['skate_cavity_faces'] if versioned_skate else (s['cavity_faces'][foot] if s['cavity_faces'].ndim==3 else s['cavity_faces'])
                    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
                    points=s[cavity_key][:,foot][:,edges].reshape(len(self.catalog.pts),-1,3)
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
            layers.update(wear=[],footbed=footbed_nodes,cavity=cavity_nodes,axes=axis_nodes,collar=collar_nodes,
                          contact=contact_nodes,skate_reference=reference_nodes,com=com_nodes,blade_material=material_nodes)
            trajectory=keep(Lines(s['G'],r_base=.018,color=(1.,.75,.12,1),name='Whole saved clip pelvis vertical projection',is_selectable=False))
            layers['trajectory']=[trajectory]
            vertical=keep(Lines(np.stack([s['P'],s['G']],axis=1),r_base=.007,color=(1.,.27,.67,1),name='P to G world vertical',is_selectable=False))
            pelvis=keep(Spheres(s['P'][:,None],radius=.032,color=(1.,.27,.67,1),name='P pelvis',is_selectable=False))
            ground=keep(Spheres(s['G'][:,None],radius=.04,color=(1.,.78,.12,1),name='G pelvis projection',is_selectable=False))
            layers['pelvis']=[vertical,pelvis,ground]
            for key in ('surface','skeleton','shoes','trajectory','pelvis','wear','footbed','cavity','axes','collar','contact','skate_reference','com','blade_material'):
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
        previous_variant=getattr(self,'variant',{}).get('id')
        self.sequence=sequence;self.variant=definition
        self.layers=layers;self.variant_nodes=nodes;self.last_p=last_p
        if old_nodes:
            self.scene.remove(*old_nodes)
            # Clear the global normals LRU only after a successful switch.
            Meshes.compute_vertex_and_face_normals.cache_clear()
        self.error=''
        if getattr(self,'view_mode','')=='grasp':
            compared=self.catalog.config.get('comparison_variants',[])
            # Shared comparison bounds already include these candidates.
            # Refitting from float32 camera coordinates would drift on switches.
            if previous_variant not in compared or definition['id'] not in compared:
                self.fit_grasp_view()
        if getattr(self,'view_mode','') in ('support_top','support_side'):
            self.last_view_center=self.support_view_center()

    def apply_visibility(self):
        for key,nodes in self.layers.items():
            for node in nodes:node.enabled=self.visible.get(key,False)

    def set_wear_inspection(self,enabled,diagnostic_layers=()):
        """A clean physical comparison; optional proxies are separate layers."""
        self.wear_inspection=bool(enabled)
        if enabled:
            if not hasattr(self,'_pre_wear_visibility'):
                self._pre_wear_visibility=self.visible.copy()
                self._pre_wear_boot_alpha=getattr(self,'boot_alpha',.62);self.boot_alpha=.40
            self.visible.update(surface=True,shoes=True,skeleton=False,trajectory=False,pelvis=False,
                                rink_lines=False,boards=False,ice=False,wear=True,collar=True,
                                footbed='footbed' in diagnostic_layers,cavity='cavity' in diagnostic_layers,axes='axes' in diagnostic_layers)
        else:
            if hasattr(self,'_pre_wear_visibility'):
                self.visible.update(self._pre_wear_visibility);del self._pre_wear_visibility
                self.boot_alpha=self._pre_wear_boot_alpha;del self._pre_wear_boot_alpha
            self.visible.update(wear=False,collar=False,footbed=False,cavity=False,axes=False)
        self.apply_visibility()
        for node in self.layers.get('shoes',[]):
            if node.name.startswith('Boot candidate'):node.color=(*node.color[:3],self.boot_alpha)

    def frame_changed(self, frame):
        p=self.sequence.state['P'][frame]
        if getattr(self,'view_mode','')=='grasp':
            if self.follow:self.fit_grasp_view(frame)
        elif getattr(self,'view_mode','') in ('support_top','support_side'):
            center=self.support_view_center()
            if self.follow and getattr(self,'last_view_center',None) is not None:
                displacement=center-self.last_view_center
                cam=self.scene.camera
                cam.position=cam.position+displacement;cam.target=cam.target+displacement
            self.last_view_center=center
        elif self.follow and self.last_p is not None:
            displacement=p-self.last_p
            cam=self.scene.camera
            cam.position=cam.position+displacement;cam.target=cam.target+displacement
        self.last_p=p.copy()
        for row,node in getattr(self,'world_region_nodes',[]):
            node.enabled=row['pts']==int(self.catalog.pts[frame])

    def support_view_center(self):
        """Only select the saved physical foot; never choose by minimum height."""
        foot=int(self.catalog.config.get('support_foot',1))
        if foot not in (0,1):raise ValueError('Invalid fixed support_foot')
        if self.catalog.config.get('comparison_variants'):
            points=np.concatenate([q.state['gear'][self.frame,foot].reshape(-1,3) for q in self.comparison_sequences().values()])
            return (points.min(axis=0)+points.max(axis=0))*.5
        s=self.sequence.state
        return s.get('skate_lower',s['gear'])[self.frame,foot].mean(axis=0).copy()

    def fit_grasp_view(self,frame=None):
        """Use one world-space view for all saved candidates at this source PTS."""
        frame=self.frame if frame is None else int(frame)
        sequences=dict(self.comparison_sequences()) if self.catalog.config.get('comparison_variants') else {}
        sequences[self.variant['id']]=self.sequence
        self.comparison_framing=fit_comparison_camera(self,sequences,frame,
            grasp_hand=self.catalog.config.get('grasp_hand','left'),free_foot=self.catalog.config.get('free_foot',0),
            left_inset=min(getattr(self,'control_panel_width',425)+28,self.window_size[0]-1),
            right_inset=min(getattr(self,'source_panel_width',630)+20,max(0,self.window_size[0]-getattr(self,'control_panel_width',425)-29)) if getattr(self.catalog,'source_video',None) else 0.)
        return self.comparison_framing

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
        if view in ('support_top','support_side'):
            p=self.support_view_center();self.last_view_center=p.copy()
            cam.position=p+([0.,0.,1.4] if view=='support_top' else [0.,-1.4,.08]);cam.target=p
            if view=='support_top':cam.up=np.array([0.,1.,0.])
            cam.is_ortho=True;cam.ortho_size=.32
            if getattr(self.catalog,'source_video',None):
                width,height=self.window_size
                left=min(getattr(self,'control_panel_width',425)+28,width-1);right=min(getattr(self,'source_panel_width',630)+20,width-left-1)
                back=cam.position-cam.target;back/=np.linalg.norm(back)
                horizontal=np.cross(cam.up,back);horizontal/=np.linalg.norm(horizontal)
                shift=horizontal*((left-right)/height*cam.ortho_size)
                cam.position-=shift;cam.target-=shift
        elif view=='grasp':
            cam.position=p+[.65,-1.1,.45];cam.target=p;cam.is_ortho=True;cam.ortho_size=.35
        elif view=='top':
            cam.position=p+[0,0,10];cam.target=p;cam.up=np.array([0.,1.,0.]);cam.is_ortho=True;cam.ortho_size=2.2
        elif view=='side':
            cam.position=p+[0,-8,0];cam.target=p;cam.is_ortho=True;cam.ortho_size=1.5
        elif view=='end':
            cam.position=p+[8,0,0];cam.target=p;cam.is_ortho=True;cam.ortho_size=1.5
        elif view=='rink':
            cam.position=np.array([42.,-48.,40.]);cam.target=np.zeros(3);self.follow=False
        elif view in ('feet','feet_oblique','feet_side','feet_end'):
            state=self.sequence.state
            p=state.get('skate_lower',state['gear'])[self.frame].reshape(-1,3).mean(axis=0)
            if view=='feet_oblique':offset=[.40,-.65,.95]
            elif view in ('feet_side','feet_end'):
                if 'skate_R' in state:axis=state['skate_R'][self.frame,0,:,1 if view=='feet_side' else 0].copy()
                else:
                    points=state.get('boot_surface',state['gear'][:,:,:14])[self.frame]
                    axis=np.mean(points[:,7:]-points[:,:7],axis=(0,1))
                    if np.linalg.norm(axis)<1e-8:axis=np.mean(points[0,7:]-points[0,:7],axis=0)
                    if np.linalg.norm(axis)<1e-8:axis=np.array([0.,1.,0.])
                    axis/=np.linalg.norm(axis)
                    if view=='feet_end':axis=np.cross(axis,[0.,0.,1.])
                offset=axis*.95+np.array([0.,0.,.16])
            else:offset=[.55,-.75,.4]
            cam.position=p+offset;cam.target=p;cam.is_ortho=True;cam.ortho_size=.34
        else:cam.position=p+[2.0,-2.7,1.3];cam.target=p
        self.comparison_framing=None
        if fit_comparison and view in ('free','top','side','end') and self.catalog.config.get('comparison_variants'):
            self.comparison_framing=fit_comparison_camera(self,self.comparison_sequences(),self.frame,
                left_inset=min(getattr(self,'control_panel_width',425)+28,self.window_size[0]-1),
                right_inset=min(getattr(self,'source_panel_width',630)+20,max(0,self.window_size[0]-getattr(self,'control_panel_width',425)-29)) if getattr(self.catalog,'source_video',None) else 0.)
        if fit_comparison and view in ('support_top','support_side') and self.catalog.config.get('comparison_variants'):
            self.comparison_framing=fit_comparison_camera(self,self.comparison_sequences(),self.frame,
                support_foot=int(self.catalog.config.get('support_foot',1)),
                left_inset=min(self.control_panel_width+28,self.window_size[0]-1),
                right_inset=min(self.source_panel_width+20,max(0,self.window_size[0]-self.control_panel_width-29)) if getattr(self.catalog,'source_video',None) else 0.)
        if fit_comparison and view=='grasp':self.fit_grasp_view()
        self.view_mode=view
        self.view_label={'top':'俯视','side':'侧视 XZ','end':'侧视 YZ','rink':'全冰场','free':'自由视角','feet':'足鞋近看','feet_oblique':'鞋口斜俯视','feet_side':'足鞋同侧剖切视角','feet_end':'足鞋正视','support_top':'支撑足俯视','support_side':'支撑足侧视','grasp':'抓握／自由鞋特写'}[view]
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
        if self.source_reader is not None:self.source_reader.close()
        self.release_source_texture()
        # Keep the context alive until Viewer.run finishes its last buffer swap.
        self.window.is_closing = True

    def center_view_on_selection(self):
        if hasattr(self,'sequence'):self.set_view('free')

    def release_source_texture(self):
        if self.source_texture is not None:
            self.imgui.remove_texture(self.source_texture)
            self.source_texture.release()
            self.source_texture=None
            self.source_texture_pts=None

    def gui_source(self):
        """The same selected integer PTS drives source pixels and saved 3D."""
        if not self.catalog.source_video:return
        imgui.set_next_window_position(self.window.size[0]-self.source_panel_width-20,14,condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.source_panel_width,min(730,self.window.size[1]-28),condition=imgui.FIRST_USE_EVER)
        with imgui.font(self.zh_font):
            expanded,_=imgui.begin('同帧原片 / 原图支撑区域')
            if expanded:
                pts=int(self.catalog.pts[self.frame]);state=self.sequence.state
                imgui.text('原片 PTS %d  |  time_base %d/%d'%(pts,*self.catalog.time_base))
                imgui.text('当前保存版本：'+self.variant['id'])
                mask=state.get('support_image_in_optimization')
                used='unknown（未保存）' if mask is None else ('是' if np.any(mask) else '否')
                imgui.text('此版本是否使用新增原图观测：'+used)
                try:
                    if self.source_reader is None:self.source_reader=SourceVideo(self.catalog.source_video,self.catalog.time_base)
                    if self.source_texture_pts!=pts:
                        image=self.source_reader.read(pts)
                        height,width=image.shape[:2]
                        if self.source_texture is None or self.source_texture.size!=(width,height):
                            self.release_source_texture()
                            self.source_texture=self.ctx.texture((width,height),3)
                            self.imgui.register_texture(self.source_texture)
                        self.source_texture.write(image.tobytes())
                        self.source_texture_pts=pts;self.source_image_size=(width,height)
                    self.source_error=''
                except Exception as error:
                    self.source_error=str(error)
                if self.source_error:
                    imgui.text_wrapped('当前原片帧不可用：'+self.source_error)
                elif self.source_texture_pts==pts:
                    rows=self.source_regions.at(pts) if self.source_regions else []
                    width,height=self.source_image_size
                    display_width=max(100.,imgui.get_content_region_available()[0])
                    self.gui_source_image(rows,(0.,0.,float(width),float(height)),display_width)
                    if rows:
                        points=np.concatenate([region_outline(row) for row in rows])
                        lower=np.maximum(points.min(0)-35,0);upper=np.minimum(points.max(0)+35,[width,height])
                        imgui.text('当前源区域放大（fit 橙 / review 紫）')
                        crop_width=min(display_width,240.*(upper[0]-lower[0])/max(upper[1]-lower[1],1.))
                        self.gui_source_image(rows,(*lower,*upper),crop_width)
                    else:imgui.text('本曝光无原图区域；不沿用相邻帧区域。')
                    for row in rows:
                        used=observation_used(state,row,self.frame)
                        used_label='unknown' if used=='unknown' else ('是' if used else '否')
                        imgui.text_wrapped('%s | foot=%s | %s | %s | 本PTS/foot新增观测入优化=%s'%
                            (row.get('id',''),row.get('foot','unknown'),row['role'],row['kind'],used_label))
                imgui.text_wrapped('区域来自原图；3D范围为同源名义平面条件诊断，不是第二份观测。')
            imgui.end()

    def gui_source_image(self,rows,crop,display_width):
        """Draw native pixel ellipses over a source texture, without projection."""
        left,top,right,bottom=crop;width,height=self.source_image_size
        scale=display_width/max(right-left,1.);display_height=(bottom-top)*scale
        x,y=imgui.get_cursor_screen_pos()
        imgui.image(self.source_texture.glo,display_width,display_height,
            uv0=(left/width,top/height),uv1=(right/width,bottom/height))
        draw=imgui.get_window_draw_list()
        for row in rows:
            outline=region_outline(row);points=(outline-[left,top])*scale+[x,y]
            color=imgui.get_color_u32_rgba(*((1.,.64,.12,1.) if row['role']=='fit' else (.75,.4,1.,1.)))
            for a,b in zip(points[:-1],points[1:]):draw.add_line(*map(float,a),*map(float,b),color,2.)

    def gui_height_plot(self,label,series,band=None):
        """Plot saved millimetres against source time, leaving NaN gaps open."""
        imgui.text(label)
        finite=[np.asarray(values)[np.isfinite(values)] for values,_ in series]
        samples=np.concatenate([values for values in finite if len(values)]+[np.array([-6.,6.])])
        low,high=float(samples.min()),float(samples.max());pad=max(1.,(high-low)*.08)
        low-=pad;high+=pad
        imgui.text('%.1f 至 %.1f 条件毫米  |  原片 %.2f–%.2f 秒'%(low,high,self.catalog.seconds[0],self.catalog.seconds[-1]))
        x,y=imgui.get_cursor_screen_pos();width=max(100.,imgui.get_content_region_available()[0]);height=72.
        draw=imgui.get_window_draw_list()
        color=imgui.get_color_u32_rgba
        draw.add_rect_filled(x,y,x+width,y+height,color(.07,.09,.12,1.))
        def py(value):return y+height-(value-low)/(high-low)*height
        if band is not None:
            draw.add_rect_filled(x,py(band[1]),x+width,py(band[0]),color(.3,.65,.35,.20))
        draw.add_line(x,py(0.),x+width,py(0.),color(.65,.7,.75,.65))
        seconds=self.catalog.seconds
        px=x+(seconds-seconds[0])/(seconds[-1]-seconds[0])*width
        for values,rgba in series:
            values=np.asarray(values);valid=np.isfinite(values)
            for index in np.flatnonzero(valid[:-1]&valid[1:]):
                draw.add_line(float(px[index]),py(float(values[index])),float(px[index+1]),py(float(values[index+1])),color(*rgba),1.5)
            # Isolated supported exposures remain visible without inventing a
            # continuous support interval through neighbouring unknowns.
            isolated=valid & ~np.r_[False,valid[:-1]] & ~np.r_[valid[1:],False]
            for index in np.flatnonzero(isolated):
                draw.add_circle_filled(float(px[index]),py(float(values[index])),1.5,color(*rgba))
        cursor=float(px[self.frame]);draw.add_line(cursor,y,cursor,y+height,color(1.,1.,1.,.8))
        imgui.dummy(width,height)

    def gui_rinkspace(self):
        imgui.set_next_window_position(14,14,condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.control_panel_width,self.window.size[1]-28,condition=imgui.FIRST_USE_EVER)
        with imgui.font(self.zh_font):
            imgui.begin('RinkSpace 三维结果播放器')
            imgui.text('%s | 原片 %.3f–%.3f 秒'%(self.catalog.config['shot'],self.catalog.seconds[0],self.catalog.seconds[-1]))
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
                else:imgui.text_colored('本帧：段外保存参考，未参与本轮优化',1.,.7,.35)
                imgui.text_wrapped('保存时间轴用于对照；本轮只优化标记窗口，整节目人体尚未完成。')
            else:imgui.text('本帧：历史保存候选，未参与本轮全身优化')
            imgui.text('原片时间：%06.3f 秒  |  PTS %d'%(seconds,self.catalog.pts[f]))
            imgui.text('当前片段帧：%d / %d'%(f+1,len(self.catalog.pts)))
            s=self.sequence.state
            for foot in range(2):
                phase=source_value(s,'source_action_phase',f,foot)
                mode=source_value(s,'source_support_mode',f,foot)
                confidence=source_value(s,'source_support_confidence',f,foot)
                confidence_label='unknown' if confidence=='unknown' else '%.2f'%confidence
                used=source_value(s,'support_image_in_optimization',f,foot)
                used_label='unknown' if used=='unknown' else str(bool(used))
                imgui.text_wrapped('足 %d | source_action_phase=%s | source_support_mode=%s | source_support_confidence=%s | support_image_in_optimization=%s'%
                    (foot,phase,mode,confidence_label,used_label))
            if imgui.button('上一帧  ,'):self.seek(f-1)
            imgui.same_line()
            if imgui.button('暂停  空格' if self.run_animations else '播放  空格'):self.toggle_animation(not self.run_animations)
            imgui.same_line()
            if imgui.button('下一帧  .'):self.seek(f+1)
            imgui.set_next_item_width(-1)
            changed,value=imgui.slider_int('##frame',f+1,1,len(self.catalog.pts),format='%d')
            if changed:self.seek(value-1)
            imgui.set_next_item_width(150)
            changed,value=imgui.input_int('跳到片段帧',f+1)
            if changed:self.seek(value-1)
            imgui.set_next_item_width(150)
            changed,index=imgui.combo('播放速度',[.25,.5,1.,2.].index(self.speed),['0.25 倍','0.5 倍','1 倍','2 倍'])
            if changed:self.speed=[.25,.5,1.,2.][index];self.playback_fps=50*self.speed
            _,self.follow=imgui.checkbox('播放时跟随运动员（保留观察角度）',self.follow)
            imgui.text('取消跟随后，当前观察位置固定；拖动仍可自由旋转／平移。')
            imgui.separator()
            for i,(label,key) in enumerate([('自由 / 回到人体  X','free'),('侧视 XZ','side'),('侧视 YZ','end'),('俯视','top'),('全冰场','rink')]):
                if i in (1,3,4):imgui.same_line()
                if imgui.button(label):self.set_view(key)
            if imgui.button('支撑足 %d 俯视'%self.catalog.config.get('support_foot',1)):self.set_view('support_top')
            imgui.same_line()
            if imgui.button('支撑足侧视'):self.set_view('support_side')
            if imgui.button('抓握／自由鞋特写'):self.set_view('grasp')
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
            imgui.same_line()
            if imgui.button('足鞋正视'):self.set_view('feet_end')
            imgui.set_next_item_width(175)
            changed,self.boot_alpha=imgui.slider_float('鞋帮透明度',self.boot_alpha,.08,1.,format='%.2f')
            if changed:
                for node in self.layers['shoes']:
                    if node.name.startswith('Boot candidate'):node.color=(*node.color[:3],self.boot_alpha)
            for label,key in [('足床与锚点','footbed'),('内腔线框','cavity'),('小腿/下鞋/刀具坐标轴','axes'),('标色鞋口边界','collar')]:
                changed,self.visible[key]=imgui.checkbox(label,self.visible[key])
                if changed:self.apply_visibility()
            imgui.set_next_item_width(175)
            changed,section=imgui.combo('鞋壳剖面',self.boot_section,['不隐藏','隐藏侧面 0','隐藏侧面 1','隐藏朝向相机侧面'])
            if changed:
                previous=self.boot_section;self.boot_section=section
                try:self.load_variant(self.variant['id'])
                except Exception as error:self.boot_section=previous;self.error=str(error)
            topology=self.variant.get('shell_topology','saved' if 'template_version' in s or 'boot_surface' in s else 'open_collar')
            imgui.text('鞋面拓扑：'+{'closed_hull':'历史封闭凸包','open_collar':'22 面开放鞋口','saved':'保存的固定拓扑 / 动态上帮'}.get(topology,topology))
            if self.wear_inspection:
                if 'cavity' not in s and 'skate_cavity' not in s:imgui.text_wrapped('本结果没有保存内腔、足床与锚点，开放鞋口本身不代表穿合完成。')
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
            if 'com_world_mid' in s:
                changed,self.visible['com']=imgui.checkbox('质量近似 COM ×（绿，非真实质心）',self.visible['com'])
                if changed:self.apply_visibility()
                imgui.text_wrapped('COM 只读取保存的条件米制质量近似；显示程序不从骨盆或表面重新推算。')
            else:imgui.text('本候选未保存质量近似 COM，无法显示。')
            if 'blade_material_world_mid' in s:
                changed,self.visible['blade_material']=imgui.checkbox('刀材料点 ●（后蓝／中黄／前白）',self.visible['blade_material'])
                if changed:self.apply_visibility()
                imgui.text_wrapped('每只刀三个固定材料点，按后／中／前区分；它们不是接冰点。')
            else:imgui.text('本候选未保存三刀材料点，无法显示。')
            if 'template_version' in s:
                imgui.text_wrapped('鞋刀模板：'+str(s['template_version'])+'；绑定：'+str(s['binding_version']))
                imgui.text('鞋刀：保存的完整下鞋实体、上鞋与精确双刃')
                for label,key in [('witness 接触候选 ◇（洋红，非真接冰）','contact'),('刀上固定参考点与整段轨迹','skate_reference')]:
                    changed,self.visible[key]=imgui.checkbox(label,self.visible[key])
                    if changed:self.apply_visibility()
                for foot in range(2):
                    mode=str(s['skate_contact_mode'][f,foot]);confidence=s['skate_contact_confidence'][f,foot]
                    label={'unknown':'未知','airborne':'腾空','glide':'滑行候选','pick':'刀齿候选'}[mode]
                    imgui.text('足 %d 接触：%s；保存置信度 %.2f'%(foot,label,confidence))
                imgui.text('允许入冰后的最小余量 0/1：%.2f / %.2f 条件毫米'%tuple(s['skate_clearance_m'][f]*1000))
                gap=s['skate_support_gap_m'][f]
                imgui.text('已知支持组候选最低高：'+('未知' if not np.isfinite(gap) else '%.2f 条件毫米'%(gap*1000)))
                self.gui_height_plot('最小非穿透余量（橙 / 青）；零线为允许边界',
                    [(s['skate_clearance_m'][:,0]*1000,(.98,.43,.14,1.)),(s['skate_clearance_m'][:,1]*1000,(.12,.85,1.,1.))])
                band=np.array([-2.,3.])*self.sequence.meta['uniform_unit_reexpression_ratio']
                self.gui_height_plot('已知支持组候选最低高；空白为未知',
                    [(s['skate_support_gap_m']*1000,(.55,1.,.65,1.))],band=band)
                imgui.text_wrapped('绿带 %.2f 至 %.2f 条件毫米为首轮工程支持范围（候选原单位 −2 至 +3 毫米）。候选最低高不判定具体哪只足接冰。witness 会沿刀弧移动，其导数不是刀具速度。'%tuple(band))
                imgui.text_wrapped('刃线按局部 y 两侧显示，未推定内/外刃；真实入冰量和压力未知。')
            else:
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
