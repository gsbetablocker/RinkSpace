"""Explicit replacement of historical world43 by current named world66."""
from dataclasses import dataclass
import numpy as np
from shared_ice_hockey_geometry_candidate.hockey_source_chart import HockeyCoordinateChart,MountedHockeyOpticsChart
from installation_mount_candidate.mount_graph import MountOpticsChart
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES

OLD=HockeyCoordinateChart(20)
OLD_WORLD=np.asarray(OLD.world_indices,int)
WORLD_INDICES=np.r_[OLD_WORLD,np.arange(OLD.core_count,OLD.core_count+23)]
CORE_COUNT=OLD.core_count+23
BASE_COUNT=CORE_COUNT+3
FULL_COUNT=BASE_COUNT+25

@dataclass(frozen=True)
class PairedProgrammeCoordinateChart:
    additional_world_parameter_count: int = 0
    def __post_init__(self):
        if (isinstance(self.additional_world_parameter_count,bool)
                or int(self.additional_world_parameter_count)!=self.additional_world_parameter_count
                or self.additional_world_parameter_count<0):
            raise ValueError('A nonnegative explicit additional world dimension is required')
    @property
    def core_count(self):return CORE_COUNT+int(self.additional_world_parameter_count)
    @property
    def base_count(self):return self.core_count+3
    @property
    def full_count(self):return self.base_count+25
    @property
    def world_indices(self):
        return tuple(int(k) for k in np.r_[WORLD_INDICES,np.arange(CORE_COUNT,self.core_count)])
    @property
    def high_indices(self):return tuple(range(self.base_count,self.full_count))
    @property
    def bias_indices(self):return tuple(range(self.core_count,self.base_count))
    @property
    def world_parameter_count(self):return 66+int(self.additional_world_parameter_count)
    def core_q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.core_count,) or not np.isfinite(q).all():raise ValueError('Whole physical programme state and declared additional world coordinates required')
        return q
    def initialize_full_q(self,old_full_q,new_world):
        old=OLD.full_q(old_full_q);w=np.asarray(new_world,float)
        if w.shape!=(self.world_parameter_count,) or not np.isfinite(w).all():raise ValueError('Complete declared physical world required')
        core=old[:OLD.core_count].copy();core[OLD_WORLD]=w[:43]
        return np.r_[core,w[43:],old[OLD.core_count:]]
    def manifest(self):
        return dict(schema='rinkspace.original-world43-to-current-named66.v1',physical_count=self.core_count,
            base_count=self.base_count,full_count=self.full_count,world_parameter_indices=list(self.world_indices),
            world_parameter_count=self.world_parameter_count,original_world66_logical_prefix_preserved=True,
            additional_world_parameter_count=int(self.additional_world_parameter_count),
            retained_camera_view_mount_lens_and_pattern_indices=True,
            historical_independent_world43_replaced=True,accuracy_accepted=False)

@dataclass(frozen=True)
class PairedProgrammeMountOpticsChart(MountOpticsChart):
    additional_world_parameter_count: int = 0
    @property
    def full_parameter_count(self):return self.coordinate_chart.core_count
    @property
    def coordinate_chart(self):return PairedProgrammeCoordinateChart(self.additional_world_parameter_count)
    def delta(self,q):
        q=self.coordinate_chart.core_q(q)
        return q[np.asarray(self.indices)]*np.asarray(PHYSICAL_COLUMN_SCALES)
    @classmethod
    def from_original(cls,old,*,additional_world_parameter_count=0):
        if type(old)is not MountedHockeyOpticsChart or old.hockey_parameter_count!=20:raise TypeError('Historical physical hockey20 chart required')
        return cls(old.installation_id,old.indices,old.original_parameter_count,old.physical_steps,old.mount_indices,
                   additional_world_parameter_count=additional_world_parameter_count)


@dataclass(frozen=True)
class CompleteProgrammeParameterLayout:
    """Keep the saved complete prefix while existing components use a core.

    The evaluation chart places extra physical coordinates before component
    suffixes. The canonical checkpoint appends them after every old suffix.
    This is a permutation, not a new model, missing state or parameter tie.
    """
    original_complete_count: int
    additional_world_parameter_count: int
    @property
    def count(self):return int(self.original_complete_count+self.additional_world_parameter_count)
    @property
    def evaluation_to_canonical(self):
        n=int(self.original_complete_count);extra=int(self.additional_world_parameter_count)
        result=np.r_[np.arange(CORE_COUNT),np.arange(n,n+extra),np.arange(CORE_COUNT,n)].astype(int)
        if n<CORE_COUNT or extra<0 or len(np.unique(result))!=self.count:raise ValueError('Complete one-to-one programme layout required')
        return result
    @property
    def canonical_to_evaluation(self):return np.argsort(self.evaluation_to_canonical)
    def evaluation_q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.count,)or not np.isfinite(q).all():raise ValueError('Finite canonical complete state required')
        return q[self.evaluation_to_canonical]
    def canonical_q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.count,)or not np.isfinite(q).all():raise ValueError('Finite evaluation complete state required')
        return q[self.canonical_to_evaluation]
    def canonical_indices(self,indices):return self.evaluation_to_canonical[np.asarray(indices,int)]
    def manifest(self):
        return dict(schema='rinkspace.complete-state-preserved-prefix-evaluation-permutation.v1',
            original_complete_parameter_count=int(self.original_complete_count),parameter_count=self.count,
            original_physical_core_parameter_count=CORE_COUNT,
            evaluation_physical_core_parameter_count=CORE_COUNT+int(self.additional_world_parameter_count),
            additional_world_parameter_count=int(self.additional_world_parameter_count),
            evaluation_to_canonical_segments=[dict(evaluation_start=0,canonical_start=0,count=CORE_COUNT),
                dict(evaluation_start=CORE_COUNT,canonical_start=int(self.original_complete_count),count=int(self.additional_world_parameter_count)),
                dict(evaluation_start=CORE_COUNT+int(self.additional_world_parameter_count),canonical_start=CORE_COUNT,count=int(self.original_complete_count)-CORE_COUNT)],
            canonical_checkpoint_old_complete_prefix_preserved=True)


class CanonicalProgrammeProfileUpdate:
    def __init__(self,base,layout):self.base,self.layout=base,layout
    def __call__(self,q):return self.base(self.layout.evaluation_q(q))
    def __getattr__(self,name):return getattr(self.base,name)


class CanonicalCompleteProgrammeSource:
    """Expose the original source interfaces in the canonical complete chart."""
    _mapped_indices=frozenset(('camera_indices','circle_indices','placement_indices','shared_schur_indices'))
    def __init__(self,base,layout):
        if base.count!=layout.count:raise ValueError('Evaluation source and complete permutation differ')
        self.base,self.parameter_layout=base,layout;self.count=base.count
        self.quadratic,self.scales=base.quadratic,base.scales
        self.updater=CanonicalProgrammeProfileUpdate(base.updater,layout)
        self.contract=dict(schema='rinkspace.original-source-canonical-complete-parameter-layout.v1',
            evaluation_source_contract=base.contract,coordinate_layout=layout.manifest(),
            source_rows=len(self.scales),parameter_count=self.count,
            source_UV_sigma_roles_and_numerical_residuals_unchanged_by_permutation=True)
        for name in ('native_ice_factor','native_board_factor','native_board_material_point_factor',
                     'circle_factor','native_board_track_factor','persistent_ice_factor','reference_factor'):
            factor=getattr(base,name,None)
            if factor is not None:factor.canonical_parameter_layout=layout
    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        result=getattr(base,name)
        return self.parameter_layout.canonical_indices(result)if name in self._mapped_indices else result
    def evaluation_q(self,q):return self.parameter_layout.evaluation_q(q)
    def evaluation_core(self,q):return self.evaluation_q(q)[:CORE_COUNT+self.parameter_layout.additional_world_parameter_count]
    def pattern_placement_context(self,q):
        """Bind physical placement offsets from the canonical complete q."""
        return self.base.pattern_placement_context(self.evaluation_q(q))
    def guard(self,q):return self.base.guard(self.evaluation_q(q))
    def unpack(self,q):return self.base.unpack(self.evaluation_q(q))
    def blocks(self,q,jacobian=True):
        from z190_streaming_source_normal import SourceRowBlock
        for block in self.base.blocks(self.evaluation_q(q),jacobian):
            J=None
            if jacobian:
                J=block.J if self.parameter_layout.additional_world_parameter_count==0 else block.J[:,self.parameter_layout.canonical_to_evaluation].tocsr()
                if not J.has_sorted_indices:J.sort_indices()
            yield SourceRowBlock(block.r,J,block.quadratic,block.scales,block.row_ledger,getattr(block,'loss_weights',None))
    def export_pattern_scene(self,scene,q):return self.base.export_pattern_scene(scene,self.evaluation_q(q))
    def export_additional_circle_scene(self,scene,q):return self.base.export_additional_circle_scene(scene,self.evaluation_q(q))
    def export_native_board_print_scene(self,scene,q):return self.base.export_native_board_print_scene(scene,self.evaluation_q(q))
    def pattern_source_review(self,q):return self.base.pattern_source_review(self.evaluation_q(q))
    def material_point_source_review(self,q):return self.base.material_point_source_review(self.evaluation_q(q))
