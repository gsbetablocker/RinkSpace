"""Third demo: recover frozen exposure bindings and use transparent gold heat.

All source artifacts remain immutable. Only caller-owned in-memory camera and
display trajectory rows from the recorded local patch are replaced.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from calibration.program_tracking.demo_revision_sample import RevisionSample
from calibration.program_tracking.demo_camera_patch import apply_patch
from calibration.program_tracking.demo_layers import IceLayers
from calibration.program_tracking.demo_particle_pass import ParticleOverlay
from calibration.program_tracking.render_demo_full import FIRST,LAST,DATA,FPS

OUT=DATA/'results/demo_full_20261016/v3'
CAMERA_PATCH=OUT/'camera_patch.npz'
TRAJECTORY_PATCH=OUT/'trajectory_patch.npz'
HEAT=dict(palette='champagne_gold',sigma_m=1.5,warm_alpha=.55,
          cold_alpha=.055,cold_rgb=[67,176,220],
          rgb_light=[255,220,150],rgb_deep=[250,175,65])


class V3Sample(RevisionSample):
    def __init__(self,version='v2',contact_path=None,mask_path=None,
                 camera_path=None,trajectory_path=None):
        super().__init__(version=version,contact_path=contact_path,mask_path=mask_path)
        camera_path=Path(camera_path) if camera_path else CAMERA_PATCH
        trajectory_path=Path(trajectory_path) if trajectory_path else TRAJECTORY_PATCH
        camera=apply_patch(self.packet,camera_path)
        with np.load(trajectory_path,allow_pickle=False) as z:
            trajectory={k:z[k].copy() for k in z.files}
        idx=trajectory['source_packet_index']
        if not np.array_equal(self.pts[idx],trajectory['pts']):
            raise ValueError('Trajectory patch native PTS mismatch')
        if not np.array_equal(self.xy[idx],trajectory['original_xy_m']):
            raise ValueError('Trajectory patch requires exact frozen v2 presentation XY')
        if not np.array_equal(trajectory['pts'],camera['pts']):
            raise ValueError('Camera and trajectory patches must bind the same exposures')
        self.xy=self.xy.copy()
        self.xy[idx]=trajectory['xy_m']
        self.contact['xy_m']=self.xy
        display_xy=gaussian_filter1d(self.xy,sigma=.16*FPS,axis=0,mode='nearest')
        raw=np.linalg.norm(np.gradient(display_xy,self.times,axis=0),axis=1)*3.6
        self.speed=gaussian_filter1d(raw,sigma=.26*FPS,mode='nearest')
        self.layers=IceLayers(self.scene,float(self.packet['unit']),self.xy,self.times,
            fade_seconds=2.5,old_alpha=.22,line_width_m=.09,sigma_m=HEAT['sigma_m'],
            heat_palette=HEAT['palette'],particle_style='particles',heat_update_anchor=0)
        self._ending_prepared=None
        self.particles=ParticleOverlay(self,first_pts=FIRST,last_pts=LAST)
        self.camera_override_path=str(camera_path)
        self.trajectory_override_path=str(trajectory_path)
