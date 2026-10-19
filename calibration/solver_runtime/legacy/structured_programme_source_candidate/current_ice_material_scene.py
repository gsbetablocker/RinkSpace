"""Bounded ice-material views from the exact shared scene producer.

This opt-in path avoids constructing print scenes and meshes for an absolute
ice-marking row. It calls the same producer's feature transform and footprint
checks on the same current perimeter, with its original auxiliary state.
Public scene export remains the complete shared scene.
"""
from collections import OrderedDict
import sys
import numpy as np


def allocated_bytes(value,seen=None):
    seen=set()if seen is None else seen
    if id(value)in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict):size+=sum(allocated_bytes(k,seen)+allocated_bytes(v,seen)for k,v in value.items())
    elif isinstance(value,(list,tuple)):size+=sum(allocated_bytes(v,seen)for v in value)
    return size


class CurrentIceMaterialView:
    def __init__(self,current_geometry66,*,max_cache_bytes=32*1024**2):
        self.current=current_geometry66;self.producer=current_geometry66.parent
        if self.current.parameter_count<66 or self.producer.parameter_count!=50:
            raise TypeError('Explicit shared current named50+paired16 producer required')
        if not all(hasattr(self.producer,k)for k in ('_feature_scene','parent_scene','features')):
            raise TypeError('The current producer must own the original feature transform')
        self.maximum=int(max_cache_bytes)
        if self.maximum<0:raise ValueError('Nonnegative explicit cache allocation budget required')
        self.cache=OrderedDict();self.cache_bytes=0;self.hits=0;self.misses=0
        self.first_entry_accounted_bytes=None;self.minimum_entry_accounted_bytes=None
        self.maximum_entry_accounted_bytes=None;self.peak_cache_accounted_bytes=0;self.evictions=0

    def _ensure_statistics(self):
        # Old parent-owned instances/pickles predate these diagnostic fields.
        # Migrate only diagnostics; never touch the geometry, key or cache.
        self.__dict__.setdefault('first_entry_accounted_bytes',None)
        self.__dict__.setdefault('minimum_entry_accounted_bytes',None)
        self.__dict__.setdefault('maximum_entry_accounted_bytes',None)
        self.__dict__.setdefault('peak_cache_accounted_bytes',getattr(self,'cache_bytes',0))
        self.__dict__.setdefault('evictions',0)

    def __setstate__(self,state):
        self.__dict__.update(state);self._ensure_statistics()

    def features(self,q):
        self._ensure_statistics()
        q=np.asarray(q,float)
        if q.shape!=(self.current.parameter_count,)or not np.isfinite(q).all():raise ValueError('Finite complete current physical world required')
        q=self.current.effective_world(q)
        # Only these columns enter _feature_scene or its perimeter. Every
        # print-placement parameter remains on its original material route.
        key=np.r_[q[:28],q[44:50]].tobytes()
        key+=b'1'if getattr(self.current,'strict_half_circle_crease',False)else b'0'
        key+=b'1'if getattr(self.current,'derive_hockey_center_from_course',False)else b'0'
        key+=b'1'if getattr(self.current,'shared_hockey_curling_axis',False)else b'0'
        extent_indices=list(getattr(self.current,'lane_right_extension_indices',{}).values())
        if extent_indices:key+=q[extent_indices].tobytes()
        crease_indices=list(getattr(self.current,'goal_crease_transverse_indices',{}).values())
        if crease_indices:key+=q[crease_indices].tobytes()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key][0]
        self.misses+=1
        course,cq,_=self.current.print_course_binding(q)
        bottom,*_=course.sample_perimeter(cq,with_jacobian=False)
        features=[]
        for original in self.producer.parent_scene['ice_features']:
            feature=dict(original)
            if 'rings'in original:feature['rings']=[dict(ring)for ring in original['rings']]
            features.append(feature)
        scene=dict(ice_features=features,
            ice_boundary=dict(points_xyz=bottom.tolist()))
        produced=self.producer._feature_scene(scene,q[:50])
        placement=getattr(self.current,'apply_goal_crease_transverse_placements',None)
        if callable(placement):produced=placement(produced,q)
        if extent_indices:
            from rinkspace.user_contours import apply_user_visible_markings
            produced.update(self.current.ice_extent_state(q))
            produced=apply_user_visible_markings(produced,pairs=[('F1','R1'),('F2','R2'),('F3','R3'),('F4','R4')],
                authority='D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json')
        features={f['id']:f for f in produced['ice_features']}
        size=allocated_bytes(features)+sys.getsizeof(key)+128
        if self.first_entry_accounted_bytes is None:self.first_entry_accounted_bytes=size
        self.minimum_entry_accounted_bytes=size if self.minimum_entry_accounted_bytes is None else min(size,self.minimum_entry_accounted_bytes)
        self.maximum_entry_accounted_bytes=size if self.maximum_entry_accounted_bytes is None else max(size,self.maximum_entry_accounted_bytes)
        while self.cache and self.cache_bytes+size>self.maximum:
            _,(_,previous)=self.cache.popitem(last=False);self.cache_bytes-=previous;self.evictions+=1
        if size<=self.maximum:
            self.cache[key]=(features,size);self.cache_bytes+=size
            self.peak_cache_accounted_bytes=max(self.peak_cache_accounted_bytes,self.cache_bytes)
        return features

    def statistics(self):
        self._ensure_statistics()
        # The first-entry projection is diagnostic only. Every actual entry
        # still uses the original recursive allocation measurement above.
        columns=list(range(28))+list(range(44,50))+list(getattr(self.current,'lane_right_extension_indices',{}).values())+list(getattr(self.current,'goal_crease_transverse_indices',{}).values())
        states=1+2*len(columns)
        return dict(schema='rinkspace.exact-current-ice-material-cache-measured-allocation.v1',
            entries=len(self.cache),cache_accounted_bytes=self.cache_bytes,max_cache_bytes=self.maximum,
            peak_cache_accounted_bytes=self.peak_cache_accounted_bytes,hits=self.hits,misses=self.misses,
            evictions=self.evictions,first_entry_accounted_bytes=self.first_entry_accounted_bytes,
            minimum_entry_accounted_bytes=self.minimum_entry_accounted_bytes,
            maximum_entry_accounted_bytes=self.maximum_entry_accounted_bytes,
            exact_key_world_columns=columns,
            symmetric_full_key_FD_working_set_states=states,
            working_set_bytes_from_first_entry_estimate=None if self.first_entry_accounted_bytes is None else states*self.first_entry_accounted_bytes,
            working_set_first_entry_projection_is_estimate_not_measured=True,
            every_cache_entry_accounted_by_original_recursive_allocated_bytes=True,
            engineering_budget_not_user_rule=True,source_or_solver_changed=False)

    def __getstate__(self):
        state=dict(self.__dict__);state.update(cache=OrderedDict(),cache_bytes=0,hits=0,misses=0,
            first_entry_accounted_bytes=None,minimum_entry_accounted_bytes=None,
            maximum_entry_accounted_bytes=None,peak_cache_accounted_bytes=0,evictions=0);return state


def install_source_ice_material_view(binder,*,max_cache_bytes=32*1024**2):
    """Opt in at a solver safe point; preserve complete public scene exports."""
    from .paired_location_programme_geometry import PairedLocationProgrammeGeometry
    if type(binder)is not PairedLocationProgrammeGeometry:
        raise TypeError('Qualified named66 material owner required')
    view=CurrentIceMaterialView(binder.current,max_cache_bytes=max_cache_bytes)
    binder._source_ice_material_view=view
    return view
