"""Named observation layout, independent of dense render/contact topology.

The v1 adapter deliberately retains the historic 14+2*17 image landmarks.
Those points are generated from the selected template and binding; they are
not the lower-skate mesh and never determine its spatial contact accuracy.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FootwearLayout:
    shell_count: int = 14
    edge_count: int = 17
    body_count: int = 17
    sides: tuple = ('left', 'right')

    @property
    def per_foot(self): return self.shell_count+2*self.edge_count
    @property
    def equipment_count(self): return len(self.sides)*self.per_foot
    @property
    def point_count(self): return self.equipment_count+self.body_count
    @property
    def body_landmarks(self): return np.arange(self.equipment_count,self.point_count)
    def footwear_landmarks(self,side,kind='all'):
        f=self.sides.index(side) if isinstance(side,str) else int(side)
        base=f*self.per_foot
        groups=dict(all=np.arange(self.per_foot),shell=np.arange(self.shell_count),
            blade=np.arange(self.shell_count,self.per_foot),
            edge_negative_y=np.arange(self.shell_count,self.shell_count+self.edge_count),
            edge_positive_y=np.arange(self.shell_count+self.edge_count,self.per_foot),
            heel=np.array([0,7]),toe=np.array([1,8]))
        return base+groups[kind]
    def record(self):
        return dict(schema='rinkspace.named-footwear-layout.v1',
            body_landmarks=self.body_landmarks.tolist(),
            footwear_landmarks={s:{k:self.footwear_landmarks(s,k).tolist() for k in
                ('all','shell','blade','edge_negative_y','edge_positive_y','heel','toe')} for s in self.sides},
            render_parts='separate template part_ranges; no positional splice',
            contact_features='exact template arcs and semantic solid guards')


OBSERVATION_LAYOUT=FootwearLayout()
