"""Immutable, source-indexed input and explicitly conditional geometry seeds."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

FIELDS = ('source_point_ids','source_target_labels','native_pts','frame_indices','source_uv')


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def readonly(value):
    a=np.array(value,copy=True); a.setflags(write=False); return a


@dataclass(frozen=True)
class Observations:
    manifest: dict
    source_point_ids: np.ndarray
    source_target_labels: np.ndarray
    native_pts: np.ndarray
    frame_indices: np.ndarray
    source_uv: np.ndarray

    @classmethod
    def load(cls, directory: Path, *, verify_hashes: bool=True):
        directory=Path(directory)
        manifest=json.loads((directory/'input_manifest.json').read_text(encoding='utf-8'))
        with np.load(directory/'observations.npz',allow_pickle=False) as z:
            values={k:readonly(z[k]) for k in FIELDS}
        result=cls(manifest=manifest,**values)
        result.validate(verify_hashes=verify_hashes)
        return result

    def validate(self, *, verify_hashes=True):
        m=self.manifest; n=m['observations']
        for name in FIELDS:
            arr=getattr(self,name); expected=m['arrays'][name]
            if arr.shape[0] != n or list(arr.shape)!=expected['shape'] or arr.dtype.str!=expected['dtype_str']:
                raise ValueError(f'Source shape/dtype mismatch: {name}')
            if verify_hashes:
                actual=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
                if actual!=expected['sha256_c_contiguous_array_bytes']:
                    raise ValueError(f'Immutable source hash mismatch: {name}')
        if self.source_uv.shape!=(n,2) or not np.isfinite(self.source_uv).all():
            raise ValueError('Invalid native UV')
        frames=np.unique(self.frame_indices)
        expected=np.arange(m['frame_min'],m['frame_max']+1)
        if not np.array_equal(frames,expected) or len(frames)!=m['frames']:
            raise ValueError('Source frame coverage changed')
        pts=m['pts_origin']+m['pts_step']*(self.frame_indices-m['frame_min'])
        if not np.array_equal(pts,self.native_pts): raise ValueError('PTS/frame binding changed')
        if np.any(np.diff(self.frame_indices)<0): raise ValueError('Source order changed')
        # Point identifiers alone are not unique. Check the actual source key.
        pairs=np.rec.fromarrays([self.native_pts,self.source_point_ids],names='pts,id')
        if len(np.unique(pairs))!=n: raise ValueError('Duplicate (PTS, source_point_id)')
        if len(np.unique(self.source_target_labels))!=m['material_labels']:
            raise ValueError('Source material identity count changed')
        w,h=m['image_wh']
        if (np.any(self.source_uv<0) or np.any(self.source_uv[:,0]>w-1)
                or np.any(self.source_uv[:,1]>h-1)):
            raise ValueError('Native source pixel outside declared image')

    def indices_for_frame(self, frame: int) -> np.ndarray:
        left=np.searchsorted(self.frame_indices,frame,'left')
        right=np.searchsorted(self.frame_indices,frame,'right')
        return np.arange(left,right,dtype=np.int64)

    def record_keys(self, indices):
        return tuple(f"{self.manifest['shot_id']}|{int(self.native_pts[i])}|{self.source_point_ids[i]}" for i in indices)

    def relative_seconds(self, frame: int) -> float:
        num,den=self.manifest['time_base']
        return (frame-self.manifest['frame_min'])*self.manifest['pts_step']*num/den

    def summary(self):
        return {'observations':len(self.source_uv),'frames':int(len(np.unique(self.frame_indices))),
                'source_materials':int(len(np.unique(self.source_target_labels))),
                'unique_point_ids_without_pts':int(len(np.unique(self.source_point_ids))),
                'source_keys_are_composite':True,'all_original_rows_retained':True,
                'pixel_fit_evaluated':False}


class SeedCatalog:
    def __init__(self,directory):
        directory=Path(directory)
        self.catalog=json.loads((directory/'seeds.json').read_text(encoding='utf-8'))
        self.materials={x['label']:x for x in self.catalog['materials']}
        self.variants=self.catalog['variants']; self.planes=self.catalog['planes']
        with np.load(directory/'world_seeds.npz',allow_pickle=False) as z:
            self.world={k:readonly(z[k]) for k in z.files}
        with np.load(directory/'camera_seeds.npz',allow_pickle=False) as z:
            self.cameras={k:readonly(z[k]) for k in z.files}

    def validate_source(self,observations):
        labels=set(observations.source_target_labels.tolist())
        if labels!=set(self.materials): raise ValueError('Catalog must account for every source label')
        for label,material in self.materials.items():
            for name in material['variants']:
                if name not in self.variants: raise ValueError(f'Missing variant for {label}')

    def plane_for_material(self,label):
        assignments={self.variants[k]['plane_candidate'] for k in self.materials[label]['variants']}
        return assignments.pop() if len(assignments)==1 else None

    def embedding(self,plane):
        p=self.planes[plane]
        M=np.zeros((4,3)); M[:3,0]=p['axis_u']; M[:3,1]=p['axis_v']; M[:3,2]=p['origin']; M[3,2]=1
        return M

    def plane_coordinates(self,plane,points):
        p=self.planes[plane]; basis=np.array([p['axis_u'],p['axis_v']],float).T
        xyz=np.asarray(points,float)-p['origin']; uv=xyz@basis
        if not np.allclose(uv@basis.T,xyz,atol=1e-9,rtol=0):
            raise ValueError('Nonplanar points cannot enter a constant-H comparator')
        return uv

    def historical_H(self,plane,frame):
        i=frame-1; c=self.cameras
        R=c['R_world_to_camera'][i]; C=c['center_world'][i]
        fx,fy=c['focal_px'][i]; cx,cy=c['principal_xy'][i]
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        P=K@np.column_stack((R,-R@C))
        H=P@self.embedding(plane)
        # This is an explicitly GS/no-distortion historical initializer.
        # It is never a physical truth constraint or a full-source result.
        return H/np.linalg.norm(H)

    def coverage(self,observations):
        values={p:{'observations':0,'frames':set(),'materials':[]} for p in self.planes}
        unsupported=[]
        for label in self.materials:
            indices=np.flatnonzero(observations.source_target_labels==label)
            plane=self.plane_for_material(label)
            if plane is None: unsupported.append(label); continue
            values[plane]['observations']+=len(indices)
            values[plane]['frames'].update(observations.frame_indices[indices].tolist())
            values[plane]['materials'].append(label)
        for v in values.values(): v['frames']=sorted(v['frames'])
        return {'scope':'conditional planar initialization only','planes':values,
                'requires_3d_or_unresolved_branch':unsupported,
                'all_observation_obligations_retained':True,'whole_source_accuracy_accepted':False}

