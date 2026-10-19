"""Isolated variable-count common hockey coordinate map; no graph installation.

New geometry coordinates follow the unchanged original physical core34225.
The original bias3 and high-camera25 are remapped after them. All historical
q/CSR data/order and profile buffers remain recoverable without truncation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from installation_mount_candidate.mount_graph import MountOpticsChart
from shared_advertisement_pairs_candidate.paired_ad_graph import MountedPairedAdOpticsChart
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES


OLD_CORE_COUNT=34225
OLD_FULL_COUNT=34253
OLD_WORLD_INDICES=tuple([0,1,2,3,4,*range(34189,34195),*range(34213,34225)])
PAIRED_INDICES=tuple(range(34213,34225))


@dataclass(frozen=True)
class HockeyCoordinateChart:
    hockey_parameter_count:int=20

    def __post_init__(self):
        if self.hockey_parameter_count not in (20,23):
            raise ValueError('Declared hockey20, or hockey20 plus optional three FR ratio coordinates required')

    @property
    def core_count(self):return OLD_CORE_COUNT+self.hockey_parameter_count
    @property
    def base_count(self):return self.core_count+3
    @property
    def full_count(self):return self.base_count+25
    @property
    def hockey_indices(self):return tuple(range(OLD_CORE_COUNT,self.core_count))
    @property
    def bias_indices(self):return tuple(range(self.core_count,self.base_count))
    @property
    def high_indices(self):return tuple(range(self.base_count,self.full_count))
    @property
    def world_indices(self):return OLD_WORLD_INDICES+self.hockey_indices
    @property
    def old_to_new(self):return np.r_[np.arange(OLD_CORE_COUNT),np.arange(self.core_count,self.full_count)]

    def full_q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.full_count,)or not np.isfinite(q).all():
            raise ValueError('Actual whole declared hockey/shared/bias/high25 state required')
        return q

    def core_q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.core_count,)or not np.isfinite(q).all():
            raise ValueError('Actual whole physical hockey state required; no unowned tail truncation')
        return q

    def extend_full_q(self,original_q,initial_hockey_deltas=None):
        q=np.asarray(original_q,dtype=float)
        d=np.zeros(self.hockey_parameter_count)if initial_hockey_deltas is None else np.asarray(initial_hockey_deltas,dtype=float)
        if q.shape!=(OLD_FULL_COUNT,)or d.shape!=(self.hockey_parameter_count,)or not np.isfinite(q).all()or not np.isfinite(d).all():
            raise ValueError('Exact original full34253 plus declared common geometry initial coordinates required')
        out=np.r_[q[:OLD_CORE_COUNT],d,q[OLD_CORE_COUNT:]]
        if not np.array_equal(out[self.old_to_new],q):
            raise ValueError('Historical source state changed during coordinate extension')
        return out

    def old_full_q(self,q):return self.full_q(q)[self.old_to_new]

    def map_old_full_csr(self,matrix):
        if getattr(matrix,'format',None)!='csr'or matrix.shape[1]!=OLD_FULL_COUNT:
            raise ValueError('Original full34253 CSR storage required; no implicit canonicalization')
        if len(matrix.indices)and (matrix.indices.min()<0 or matrix.indices.max()>=OLD_FULL_COUNT):
            raise ValueError('Original source CSR contains undeclared coordinates')
        indices=self.old_to_new[matrix.indices].astype(matrix.indices.dtype,copy=False)
        out=csr_matrix((matrix.data,indices,matrix.indptr),shape=(matrix.shape[0],self.full_count),copy=False)
        out.has_sorted_indices=matrix.has_sorted_indices
        out.has_canonical_format=matrix.has_canonical_format
        return out

    def map_old_shared_indices(self,shared):
        shared=np.asarray(shared,dtype=int)
        if (shared.shape!=(62,)or len(set(shared.tolist()))!=62 or
            shared.min()<0 or shared.max()>=34228 or
            not set(range(34225,34228)).issubset(shared)):
            raise ValueError('Original shared62 including statistical bias3 and excluding high25 required')
        return np.r_[self.old_to_new[shared],self.hockey_indices]

    def manifest(self):
        return dict(schema='rinkspace.common-hockey-coordinate-chart.v1',
                    original_physical_count=OLD_CORE_COUNT,original_full_count=OLD_FULL_COUNT,
                    physical_count=self.core_count,base_with_statistics_count=self.base_count,
                    full_parameter_count=self.full_count,new_geometry_parameter_count=self.hockey_parameter_count,
                    original_world_indices=list(OLD_WORLD_INDICES),world_indices=list(self.world_indices),
                    hockey_parameter_indices=list(self.hockey_indices),statistical_bias_indices=list(self.bias_indices),
                    high_camera25_indices=list(self.high_indices),
                    original_complete_to_current_parameter_indices=self.old_to_new.tolist(),
                    old_core_coordinate_indices_unchanged=True,
                    original_shared_count=62,current_shared_count=62+self.hockey_parameter_count,
                    required_fresh_complete_source_direction_count=87+self.hockey_parameter_count,
                    profile_values_roles_and_keys_changed=False,whole_source_graph_installed=False,
                    no_source_or_standard_reference_prior_installed=True)


@dataclass(frozen=True)
class MountedHockeyOpticsChart(MountOpticsChart):
    """Same physical mount/lens function on a declared expanded whole core."""
    hockey_parameter_count:int=20
    paired_ad_parameter_indices:tuple[int,...]=PAIRED_INDICES

    @property
    def coordinate_chart(self):return HockeyCoordinateChart(self.hockey_parameter_count)
    @property
    def full_parameter_count(self):return self.coordinate_chart.core_count

    def delta(self,q):
        q=self.coordinate_chart.core_q(q)
        if (self.original_parameter_count!=34204 or self.paired_ad_parameter_indices!=PAIRED_INDICES or
            len(self.indices)!=3 or tuple(self.indices)!=tuple(range(self.indices[0],self.indices[0]+3))or
            self.indices[0]not in(34204,34207,34210)or
            tuple(self.mount_indices)!=tuple(range(34195+self.indices[0]-34204,34198+self.indices[0]-34204))):
            raise ValueError('Original authenticated mount9/lens9/pair12 chart and current common hockey world required')
        return q[np.asarray(self.indices)]*np.asarray(PHYSICAL_COLUMN_SCALES)

    @classmethod
    def from_original(cls,original,hockey_parameter_count=20):
        if type(original)is not MountedPairedAdOpticsChart or original.full_parameter_count!=OLD_CORE_COUNT:
            raise ValueError('Actual original paired/core34225 optics chart required')
        return cls(original.installation_id,original.indices,original.original_parameter_count,
                   original.physical_steps,original.mount_indices,hockey_parameter_count)

    def manifest(self,base):
        result=MountOpticsChart.manifest(self,base)
        result.update(whole_physical_parameter_count=self.full_parameter_count,
                      current_camera_chart='original mount/lens/pair function plus declared shared hockey geometry',
                      common_hockey_coordinate_contract=self.coordinate_chart.manifest(),
                      new_hockey_coordinates_do_not_change_camera_function=True)
        return result
