"""Application-level checks using the player's own renderer, without desktop input."""
from __future__ import annotations
import json
import time
import numpy as np


class PlayerVerification:
    def __init__(self,catalog):
        self.catalog=catalog;self.phase=0;self.results=[];self.start=time.monotonic()
        self.output=catalog.session_root/'verification';self.output.mkdir(exist_ok=True)
        self.failed=False

    def check(self,name,condition,**details):
        self.results.append(dict(check=name,passed=bool(condition),**details))
        if not condition:raise AssertionError(name)

    def snapshot(self,player,name):
        # AITViewer renders only its own 3D framebuffer; no desktop screenshots are read.
        player.export_frame(str(self.output/(name+'.png')))

    def tick(self,p,t):
        try:
            phase=self.phase;self.phase+=1
            if phase==0:
                p.renderer.get_current_frame_as_image().save(self.output/'controls_and_current.png')
                self.check('1143 original source exposures',len(self.catalog.pts)==1143 and p.scene.n_frames==1143)
                self.check('source PTS advance equals 20ms',np.all(np.diff(self.catalog.pts)==256))
                self.check('current full surface and boots',p.sequence.vertices.shape==(1143,10475,3) and p.sequence.state['gear'].shape==(1143,2,48,3))
                self.check('unaccepted accuracy retained',p.sequence.meta['accuracy_accepted'] is False)
                p.seek(695);p.set_view('free');self.snapshot(p,'current_free')
                self.camera={a:getattr(p.scene.camera,a).copy() for a in ('position','target','up')}
                self.g=p.sequence.state['G'][695].copy();p.seek(696)
                delta=p.sequence.state['P'][696]-p.sequence.state['P'][695]
                self.check('single source frame and follow translation',p.frame==696 and np.allclose(p.scene.camera.position-self.camera['position'],delta,atol=2e-6))
                p.seek(695)
                self.camera={a:getattr(p.scene.camera,a).copy() for a in ('position','target','up')}
                self.switched=[]
            elif 1<=phase<=7:
                variant=self.catalog.variants[phase-1]['id'];p.load_variant(variant)
                self.check('switch '+variant+' preserves frame and camera',p.frame==695 and all(np.array_equal(getattr(p.scene.camera,a),v) for a,v in self.camera.items()))
                self.check('reconstructed saved geometry '+variant,p.sequence.meta['saved_coco_mid_maximum_reconstruction_error_m']<1e-7 and p.sequence.meta['full_vs_compact_coco_maximum_error_m']<2e-5)
                self.switched.append(variant)
            elif phase==8:
                self.snapshot(p,'old_fixed_pose');p.load_variant('j1');p.set_view('side')
                self.check('side projection orthographic and Z up',p.scene.camera.is_ortho and np.allclose(p.scene.camera.up,[0,0,1]))
                self.snapshot(p,'current_side');p.set_view('top')
                self.check('top camera has non-collinear up',abs(np.dot(p.scene.camera.up,p.scene.camera.target-p.scene.camera.position))<1e-5)
                self.snapshot(p,'current_top');p.set_view('free')
                self.paused=p.frame;self.paused_camera=p.scene.camera.position.copy()
            elif phase==9:
                p.renderer.get_current_frame_as_image().save(self.output/'controls_and_current.png')
                self.check('pause keeps source frame',p.frame==self.paused)
                p.scene.camera.position=p.scene.camera.position+np.array([1.,-.6,.3])
                self.rotated=p.scene.camera.position.copy()
                self.check('paused free inspection leaves pose fixed',p.frame==self.paused)
            elif phase==10:
                self.check('pause does not recenter free camera',np.array_equal(p.scene.camera.position,self.rotated))
                p.seek(100);self.check('seek to native frame',p.frame==100 and self.catalog.pts[p.frame]==308736)
                p.seek(99);self.check('previous native frame',p.frame==99)
                self.play_start=p.frame;self.play_time=t;p.toggle_animation(True)
            elif phase>=11:
                if p.frame==self.play_start and t-self.play_time<1.5:return
                self.check('playback advances source exposures',p.frame!=self.play_start)
                p.toggle_animation(False);p.seek(695);p.follow=True;p.set_view('free')
                p.save_session()
                report=dict(passed=all(x['passed'] for x in self.results),checks=self.results,
                    method='Application API + own OpenGL renderer; no desktop UI input',
                    screenshots=[str(x) for x in self.output.glob('*.png')],elapsed_seconds=time.monotonic()-self.start)
                (self.output/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
                print(json.dumps(report,ensure_ascii=False),flush=True);p.close()
        except Exception as error:
            self.failed=True
            (self.output/'checks.json').write_text(json.dumps(dict(passed=False,checks=self.results,error=repr(error)),ensure_ascii=False,indent=2),encoding='utf8')
            p.close();raise
