"""Local visual revisions for the second full-programme demo.

The source/camera/body models and historical masks remain immutable. This
adapter changes presentation trajectory, selected native-frame masks and heat.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from calibration.program_tracking.render_demo_full import FullSample,FIRST,LAST,OUT as FULL_OUT
from calibration.program_tracking.demo_layers import IceLayers
from calibration.program_tracking.demo_particle_pass import ParticleOverlay
from calibration.program_tracking.render_demo_last_shot import WH
from calibration.program_tracking.render_demo_intro import IntroShotSample

OUT=FULL_OUT/'v2'
CONTACT=OUT/'trajectory_packet.npz'
MASKS=OUT/'foreground_masks.npz'
HEAT=dict(palette='hot_red',sigma_m=1.5,warm_alpha=.58,
          rgb_light=[225,28,20],rgb_deep=[135,6,16])
MASK_WINDOWS=((1300224,1306880),(1456384,1469952))


class RevisionSample(FullSample):
    def __init__(self,version='v2',contact_path=None,mask_path=None):
        super().__init__(version=version,contact_path=contact_path or CONTACT)
        self.layers=IceLayers(self.scene,float(self.packet['unit']),self.xy,self.times,
            fade_seconds=2.5,old_alpha=.22,line_width_m=.09,sigma_m=HEAT['sigma_m'],
            heat_palette=HEAT['palette'],particle_style='particles',heat_update_anchor=0)
        self._ending_prepared=None
        self.particles=ParticleOverlay(self,first_pts=FIRST,last_pts=LAST)
        mask_path=Path(mask_path) if mask_path is not None else MASKS
        with np.load(mask_path,allow_pickle=False) as p:
            self._native_mask_pts=p['pts'].astype(np.int64)
            self._native_masks=p['masks'].copy()
            self._native_mask_status=(p['status'].astype(str) if 'status' in p
                                      else np.full(len(p['pts']),'native50_local_resegmented_mask'))
        if (self._native_mask_pts.ndim!=1 or self._native_masks.dtype!=np.uint8 or
            self._native_masks.shape!=(len(self._native_mask_pts),540,120) or
            np.any(np.diff(self._native_mask_pts)<=0)):
            raise ValueError('Local foreground masks must be ordered nativePTS packed960x540')
        self._native_mask_lookup={int(p):i for i,p in enumerate(self._native_mask_pts)}
        required=np.concatenate([np.arange(a,b+256,256,dtype=np.int64) for a,b in MASK_WINDOWS])
        if not set(required.tolist()).issubset(self._native_mask_lookup):
            raise ValueError('Reviewed foreground repair window has missing masks')
        self.mask_override_path=str(mask_path)

    def foreground(self,index):
        pts=int(self.pts[index])
        if FIRST<=pts<=575488:
            return IntroShotSample.foreground(self,index)
        row=self._native_mask_lookup.get(pts)
        if row is None:
            return super().foreground(index)
        raw=np.unpackbits(self._native_masks[row],axis=1)[:,:960].astype(np.float32)
        mask=cv2.resize(raw,WH,interpolation=cv2.INTER_LINEAR)
        mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
        mask=cv2.GaussianBlur(mask,(0,0),.9)
        mask[0:255,0:555]=1.
        mask[934:1010,1670:1880]=1.
        self.last_mask_label=str(self._native_mask_status[row])
        self.mask_counts[self.last_mask_label]=self.mask_counts.get(self.last_mask_label,0)+1
        return mask

    def draw(self,native,pts):
        rendered,base,texture,foreground=super().draw(native,int(pts))
        u=float(np.clip((int(pts)-FIRST)/12800./.5,0.,1.))
        u=u*u*(3.-2.*u)
        if u<1.:
            rendered=np.clip(native.astype(np.float32)*(1.-u)+rendered.astype(np.float32)*u+.5,0,255).astype(np.uint8)
        return rendered,base,texture,foreground
