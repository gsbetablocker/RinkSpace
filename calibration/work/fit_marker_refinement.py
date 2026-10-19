"""Small shared physical marking adjustments, authorized on 2026-09-24.

All three cameras see the same modified world features. Original image
observations are never moved. The previously fitted boundary and basic marking
shapes remain the reference while positional relationships may relax.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
from fit_joint import Problem,BASE,PTS,read,save,fit_stage,closest_delta,safe_project
from relaxed_markers import MARKER_SPECS,apply_marker_offsets

class MarkerProblem(Problem):
    def __init__(self):
        super().__init__()
        self.marker_start=len(self.x0)
        self.marker_specs=MARKER_SPECS
        self.marker_count=len(MARKER_SPECS)
        self.marker_by_feature={}
        for i,s in enumerate(MARKER_SPECS):self.marker_by_feature.setdefault(s['feature_id'],[]).append(i)
        self.x0=np.r_[self.x0,np.zeros(self.marker_count)]
        self.bounds=(np.r_[self.bounds[0],[s['lower'] for s in MARKER_SPECS]],np.r_[self.bounds[1],[s['upper'] for s in MARKER_SPECS]])
        self.names.extend([s['name'] if s['name'].startswith('marker:') else 'marker:'+s['name'] for s in MARKER_SPECS])
        self.marker_prior_scales=np.array([.01 if 'angle' in s['component'] or 'rotation' in s['component'] else .25 for s in MARKER_SPECS])
        self.reference_fit=None

    def geometry(self,x):
        base=super().geometry(x)
        if not hasattr(self,'marker_start') or len(x)<=self.marker_start:return base
        return apply_marker_offsets(base,np.asarray(x[self.marker_start:self.marker_start+self.marker_count]))

    def geometry_key(self,x):
        return super().geometry_key(x)+tuple(x[self.marker_start:self.marker_start+self.marker_count])

    @lru_cache(maxsize=160)
    def cached_geometry(self,key):
        base=super().cached_geometry(key[:self.geometry_count+2])
        return apply_marker_offsets(base,np.asarray(key[self.geometry_count+2:]))

    def cached_delta(self,gkey,ckey,oid,samples):
        feature=self.by_id[oid]['target'].get('feature_id')
        indices=self.marker_by_feature.get(feature,[])
        basekey=gkey[:self.geometry_count+2]
        localkey=tuple(gkey[self.geometry_count+2+i] for i in indices)
        return self._local_delta(basekey,feature,localkey,ckey,oid,samples)

    @lru_cache(maxsize=384)
    def _local_geometry(self,basekey,feature,localkey):
        base=super().cached_geometry(basekey)
        if feature not in self.marker_by_feature:return base
        values=np.zeros(self.marker_count)
        values[self.marker_by_feature[feature]]=localkey
        return apply_marker_offsets(base,values)

    @lru_cache(maxsize=4096)
    def _local_delta(self,basekey,feature,localkey,ckey,oid,samples):
        geo=self._local_geometry(basekey,feature,localkey)
        world=self.curve_world(geo,self.by_id[oid])
        projected,_,valid=safe_project(np.asarray(ckey),world)
        return closest_delta(self.by_id[oid]['uv'],projected,valid)

    def residual(self,x,reg=.01):
        ordinary=super().residual(x,reg)
        # Explicit weak preference for the existing physical positions; the
        # image evidence may move them within the declared small search range.
        prior=.5*np.asarray(x[self.marker_start:self.marker_start+self.marker_count])/self.marker_prior_scales
        return np.r_[ordinary,prior]

    def save_result(self,x,path,stage_record):
        super().save_result(x,path,stage_record)
        data=read(path)
        data['model_class']='shared_scene_with_small_marker_position_offsets'
        data['marker_refinement']={'authorization':'User 2026-09-24: small departures from uniform placement, parallelism and symmetry may be fitted','reference_fit':self.reference_fit,'same_offsets_in_all_frames':True,'all_original_observations_retained':True,'reference_boundary_and_basic_shapes_fixed_during_marker_refinement':True,'prior_residual':'0.5 * physical_parameter_offset / recorded_scale','parameters':[dict(s,value=float(v),prior_scale=float(scale)) for s,v,scale in zip(MARKER_SPECS,x[self.marker_start:],self.marker_prior_scales)]}
        save(path,data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',required=True);ap.add_argument('--stage',choices=['markers','joint'],default='markers');ap.add_argument('--out',default=str(BASE/'work/marker_refinement'));ap.add_argument('--nfev',type=int,default=100);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--tail-scale',type=float,default=3.)
    a=ap.parse_args();p=MarkerProblem();d=read(a.resume);x=p.x0.copy();names=d['parameter_names']
    if names!=p.names[:len(names)]:raise ValueError('reference parameter names mismatch')
    x[:len(names)]=d['parameter_values'];p.tail_scale=a.tail_scale
    p.reference_fit=d.get('marker_refinement',{}).get('reference_fit') or str(Path(a.resume).as_posix())
    active=list(range(p.marker_start,len(x)))
    if a.stage=='joint':
        active += [j for pts in PTS for j in range(p.slices[('camera',pts)].start,p.slices[('camera',pts)].start+23)]
        active += [j for sl in p.slices_ad().values() for j in range(sl.start,sl.stop)]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fit_stage(p,x,active,a.stage,out,max_nfev=a.nfev,workers=a.workers)

if __name__=='__main__':main()
