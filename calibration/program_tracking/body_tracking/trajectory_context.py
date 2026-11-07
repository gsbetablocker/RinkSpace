"""One source/identity/world contract across physical photography cuts."""
from dataclasses import dataclass
from pathlib import Path
import json,hashlib
import numpy as np
from .programme_contract import required_clock,boundary_receipt,protect_primary_arrays

@dataclass(frozen=True)
class ProgrammeContext:
    source_sha256:str
    source_pts:np.ndarray
    shot_ids:np.ndarray
    time_base:tuple
    shared_body_id:str
    shared_shape_sha256:str
    metres_per_scene_unit:float
    equipment_sha256:str
    frozen_camera_sha256:str

    @classmethod
    def from_scope(cls,scope_path,*,body_id,shape,metres_per_scene_unit,equipment_sha256,camera_sha256):
        scope=json.loads(Path(scope_path).read_text(encoding='utf-8'));pts,shots,tb=required_clock(scope)
        shape=np.asarray(shape,dtype='<f8')
        if shape.ndim!=1 or not np.isfinite(shape).all()or not float(metres_per_scene_unit)>0:raise ValueError('One finite shared body identity and metric binding required')
        return cls(scope['source']['sha256'],pts,shots,tb,body_id,hashlib.sha256(shape.tobytes()).hexdigest(),float(metres_per_scene_unit),equipment_sha256,camera_sha256)

    def packet(self):
        return dict(schema='rinkspace.programme-context.v1',source_sha256=self.source_sha256,source_frames=len(self.source_pts),first_pts=int(self.source_pts[0]),last_pts=int(self.source_pts[-1]),time_base=list(self.time_base),shared_body_id=self.shared_body_id,shared_shape_sha256=self.shared_shape_sha256,metres_per_scene_unit=self.metres_per_scene_unit,equipment_sha256=self.equipment_sha256,frozen_camera_sha256=self.frozen_camera_sha256,world='same frozen scene; pelvis P, G=(Px,Py,0); COM separate',no_per_shot_world_alignment=True)
