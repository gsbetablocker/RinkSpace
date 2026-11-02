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
        self.follow = True
        self.speed = 1.
        self.surface_alpha = .72
        self.ice_alpha = .48
        self.visible = dict(surface=True, skeleton=True, shoes=True, trajectory=True, pelvis=True,
                            rink_lines=True, boards=True, ice=True)
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
        selected = variant or old.get('variant', catalog.config['default_variant'])
        self.load_variant(selected)
        self.observer = FrameObserver(self)
        self.scene.add(self.observer)
        initial_pts = old.get('source_pts', catalog.config['initial_source_pts'])
        self.scene.current_frame_id = int(np.argmin(abs(catalog.pts - initial_pts)))
        self.last_p = self.sequence.state['P'][self.frame].copy()
        self.set_view('free')
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
        old_nodes = self.variant_nodes
        self.variant_nodes=[]
        self.sequence=sequence;self.variant=definition
        s=sequence.state
        body=Meshes(sequence.vertices,sequence.faces,name='Saved full SMPL-X surface',
            color=(.58,.69,.84,self.surface_alpha),pickable=False,cast_shadow=False)
        connections=np.asarray([(int(p),j) for j,p in enumerate(s['parents']) if j and p>=0],np.int32)
        skeleton=Skeletons(s['joints'],connections,radius=.009,color=(.55,1.,.28,1),name='Native skeleton')
        self.layers['surface']=[body];self.layers['skeleton']=[skeleton]
        equipment=[]
        for foot,color in enumerate([(.98,.43,.14,1),(.12,.85,1.,1)]):
            gear=s['gear'][:,foot]
            if gear.shape[1]==48:
                shell=Meshes(gear[:,:14],shell_triangles(gear[0,:14]),color=(*color[:3],.62),
                    name='Boot candidate '+str(foot+1),pickable=False,cast_shadow=False)
                strip=Meshes(gear[:,14:],ribbon_triangles(17),color=color,
                    name='Blade lower edge strip '+str(foot+1),pickable=False,cast_shadow=False)
                equipment.extend([shell,strip])
                for offset in (14,31):equipment.append(Lines(gear[:,offset:offset+17],r_base=.003,color=color,name='Blade edge '+str(foot+1),is_selectable=False))
            else:equipment.append(Lines(gear,r_base=.004,color=color,name='Old blade proxy '+str(foot+1),is_selectable=False))
        self.layers['shoes']=equipment
        trajectory=Lines(s['G'],r_base=.018,color=(1.,.75,.12,1),name='Whole S01 pelvis vertical projection',is_selectable=False)
        self.layers['trajectory']=[trajectory]
        vertical=Lines(np.stack([s['P'],s['G']],axis=1),r_base=.007,color=(1.,.27,.67,1),name='P to G world vertical',is_selectable=False)
        pelvis=Spheres(s['P'][:,None],radius=.032,color=(1.,.27,.67,1),name='P pelvis',is_selectable=False)
        ground=Spheres(s['G'][:,None],radius=.04,color=(1.,.78,.12,1),name='G pelvis projection',is_selectable=False)
        self.layers['pelvis']=[vertical,pelvis,ground]
        for key in ('surface','skeleton','shoes','trajectory','pelvis'):
            self.variant_nodes.extend(self.layers[key])
        if old_nodes:
            self.scene.remove(*old_nodes)
            # AITViewer's global LRU otherwise retains removed mesh objects and their frame normals.
            Meshes.compute_vertex_and_face_normals.cache_clear()
        self.scene.add(*self.variant_nodes)
        for node in self.variant_nodes:node.current_frame_id=frame
        self.last_p=s['P'][frame].copy()
        self.apply_visibility()
        self.error=''

    def apply_visibility(self):
        for key,nodes in self.layers.items():
            for node in nodes:node.enabled=self.visible[key]

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

    def set_view(self, view):
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
        else:cam.position=p+[2.0,-2.7,1.3];cam.target=p
        self.view_label={'top':'俯视','side':'侧视 XZ','end':'侧视 YZ','rink':'全冰场','free':'自由视角'}[view]
        self.save_session()

    def save_session(self):
        if not hasattr(self,'variant'):return
        cam=self.scene.camera;frame=self.frame
        state=dict(schema='rinkspace.viewer-session.v1',pid=os.getpid(),variant=self.variant['id'],
            local_frame=frame,source_pts=int(self.catalog.pts[frame]),source_seconds=float(self.catalog.seconds[frame]),
            playing=bool(self.run_animations),follow=self.follow,visible=self.visible,
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
        imgui.set_next_window_size(425,self.window.size[1]-28,condition=imgui.FIRST_USE_EVER)
        with imgui.font(self.zh_font):
            imgui.begin('RinkSpace 三维结果播放器')
            imgui.text('S01  |  原片 22.120–44.960 秒  |  50 帧/秒')
            imgui.text('冰场：本轮人体实验配套的冻结场景')
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
            imgui.text('粉色 P：骨盆；黄色 G：骨盆竖直投影')
            imgui.text('橙 / 青：两只鞋刀，左右身份尚未确认')
            imgui.text('接冰状态：未知（显示程序不强制贴冰）')
            if gear.shape[1]==5:imgui.text_colored('此旧版只有刀刃代理，没有拟合鞋壳',1.,.7,.35)
            else:imgui.text('鞋刀：保存的刚性鞋壳与刀刃下缘')
            if bool(s['source_lower_body_inferred'][f]):imgui.text_colored('本帧下肢观测不足，含推断',1.,.45,.35)
            if bool(s['source_identity_suspect'][f]):imgui.text_colored('本帧存在左右身份疑点',1.,.45,.35)
            imgui.text('统一世界坐标：Z 向上；单位：米')
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
