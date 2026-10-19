"""Bounded shared centre offsets for individual painted concentric-ring layers.

Tests the remaining positional association with all original evidence kept.
Circles retain their physical radii and circular shape in the common ice plane.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
from fit_joint import BASE,PTS,read,save,fit_stage,closest_delta,safe_project
from fit_marker_refinement import MarkerProblem
from relaxed_rings import RING_SPECS,apply_ring_offsets

class RingProblem(MarkerProblem):
    def __init__(self):
        super().__init__()
        self.ring_start=len(self.x0);self.ring_count=len(RING_SPECS)
        self.ring_by_feature={}
        for i,s in enumerate(RING_SPECS):self.ring_by_feature.setdefault(s['feature_id'],[]).append(i)
        self.x0=np.r_[self.x0,np.zeros(self.ring_count)]
        self.names.extend([s['name'] for s in RING_SPECS])
        self.bounds=(np.r_[self.bounds[0],[s['lower'] for s in RING_SPECS]],np.r_[self.bounds[1],[s['upper'] for s in RING_SPECS]])
        self.ring_reference_fit=None

    def geometry(self,x):
        base=super().geometry(x)
        if not hasattr(self,'ring_start') or len(x)<=self.ring_start:return base
        return apply_ring_offsets(base,np.asarray(x[self.ring_start:]))

    def geometry_key(self,x):return super().geometry_key(x)+tuple(x[self.ring_start:])

    @lru_cache(maxsize=160)
    def cached_geometry(self,key):
        split=self.geometry_count+2+self.marker_count
        return apply_ring_offsets(super().cached_geometry(key[:split]),np.asarray(key[split:]))

    def cached_delta(self,gkey,ckey,oid,samples):
        feature=self.by_id[oid]['target'].get('feature_id')
        start=self.geometry_count+2
        local_marker=tuple(gkey[start+i] for i in self.marker_by_feature.get(feature,[]))
        local_ring=tuple(gkey[start+self.marker_count+i] for i in self.ring_by_feature.get(feature,[]))
        return self._ring_delta(gkey[:start],feature,local_marker,local_ring,ckey,oid,samples)

    @lru_cache(maxsize=384)
    def _ring_geometry(self,basekey,feature,local_marker,local_ring):
        geo=super()._local_geometry(basekey,feature,local_marker)
        if feature not in self.ring_by_feature:return geo
        offsets=np.zeros(self.ring_count);offsets[self.ring_by_feature[feature]]=local_ring
        return apply_ring_offsets(geo,offsets)

    @lru_cache(maxsize=4096)
    def _ring_delta(self,basekey,feature,local_marker,local_ring,ckey,oid,samples):
        geo=self._ring_geometry(basekey,feature,local_marker,local_ring)
        world=self.curve_world(geo,self.by_id[oid]);uv,_,valid=safe_project(np.asarray(ckey),world)
        return closest_delta(self.by_id[oid]['uv'],uv,valid)

    def residual(self,x,reg=.01):
        return np.r_[super().residual(x,reg),.5*np.asarray(x[self.ring_start:])/.08]

    def save_result(self,x,path,stage_record):
        super().save_result(x,path,stage_record);data=read(path)
        data['model_class']='shared_scene_with_small_marker_and_ring_center_offsets'
        data['ring_position_refinement']={'reference_fit':self.ring_reference_fit,'same_centres_in_every_frame':True,'circular_shapes_and_radii_preserved':True,'prior_scale_scene_units':.08,'prior_multiplier':.5,'parameters':[dict(s,value=float(v)) for s,v in zip(RING_SPECS,x[self.ring_start:])]}
        save(path,data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',required=True);ap.add_argument('--stage',choices=['rings','joint','full'],default='rings');ap.add_argument('--out',default=str(BASE/'work/ring_refinement'));ap.add_argument('--nfev',type=int,default=120);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--tail-scale',type=float,default=3.)
    a=ap.parse_args();p=RingProblem();d=read(a.resume);x=p.x0.copy();names=d['parameter_names']
    if names!=p.names[:len(names)]:raise ValueError('reference parameter names mismatch')
    x[:len(names)]=d['parameter_values'];p.tail_scale=a.tail_scale
    p.reference_fit=d.get('marker_refinement',{}).get('reference_fit') or str(Path(a.resume).as_posix())
    p.ring_reference_fit=d.get('ring_position_refinement',{}).get('reference_fit') or str(Path(a.resume).as_posix())
    active=list(range(p.ring_start,len(x)))
    if a.stage in ('joint','full'):
        active+=list(range(p.marker_start,p.ring_start))
        active += [j for pts in PTS for j in range(p.slices[('camera',pts)].start,p.slices[('camera',pts)].start+23)]
        active += [j for sl in p.slices_ad().values() for j in range(sl.start,sl.stop)]
    if a.stage=='full':
        active += list(range(p.geometry_count))+[p.corner_index,p.corner_aspect_index]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fit_stage(p,x,active,a.stage,out,max_nfev=a.nfev,workers=a.workers)

if __name__=='__main__':main()
