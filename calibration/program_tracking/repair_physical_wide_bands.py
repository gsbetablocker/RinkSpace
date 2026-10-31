"""Remeasure straight paint across its physical transverse direction."""
from __future__ import annotations
from copy import deepcopy
import argparse
import cv2
import numpy as np
from .extract_first_shot_ice import (Extractor,NativeProfiles,DATA,ROOT,read,write,sha,
    _source_queries,current_broadcast_exclusion,current_ice_visible_domain)
from .physical_ice_width import projected_width_profile,restore_physical_profile_readout
from .native_ice_material_repair import whole_colour_bands
from .projection import ground_xy

RD=DATA/'repair_20261014'
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'


class PhysicalBandProfiles(NativeProfiles):
 def __init__(self,rgb,source,mask,camera):
  super().__init__(rgb,source,mask);self.camera=camera
  color=rgb.astype(np.float32)
  self.fields={'dark':-(.2126*color[...,0]+.7152*color[...,1]+.0722*color[...,2]),
      'blue':color[...,2]-(color[...,0]+color[...,1])*.5,
      'red':color[...,0]-(color[...,1]+color[...,2])*.5}

 def values(self,profile,signal,tangent):
  h,w=self.frame.rgb.shape[:2]
  uv=np.asarray(profile['source_uv'],np.float32)
  samples=uv[None]+np.arange(-3,4)[:,None,None]*tangent
  values=cv2.remap(self.fields[signal],samples[...,0].astype(np.float32),samples[...,1].astype(np.float32),cv2.INTER_LINEAR)
  # Every bilinear contributor belongs to the visible current ice surface.
  floor=np.floor(samples).astype(int);valid=np.ones(samples.shape[:2],bool)
  for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
   x=floor[...,0]+dx;y=floor[...,1]+dy
   inside=(x>=0)&(x<w)&(y>=0)&(y<h)
   good=np.zeros_like(inside);good[inside]=~self.mask[y[inside],x[inside]]
   valid&=good
  counts=valid.sum(0);self.last_tangent_valid_counts=counts
  ordered=np.sort(np.where(valid,values,np.inf),axis=0)
  columns=np.arange(len(counts));lo=np.maximum(0,(counts-1)//2);hi=np.maximum(0,counts//2)
  result=(ordered[lo,columns]+ordered[hi,columns])*.5
  result[counts==0]=np.nan
  return result

 def sample_complete_colour_band(self,path,guide,spacing):
  h,w=self.frame.rgb.shape[:2]
  q,n,_=_source_queries(path,guide,np.ones(len(guide)),(h,w),35,60)
  rows=[];rejected=[]
  for k,query in enumerate(q):
   initial=projected_width_profile(self.camera,path,query,image_hw=(h,w),radius_px=1)
   if initial is None:continue
   cross=abs(float(initial['query_width_direction_uv']@n[k]))
   if cross<.01:continue
   radius=185 if path['material_signal']=='red' else 70
   profile=projected_width_profile(self.camera,path,query,image_hw=(h,w),radius_px=radius/cross,step_px=1/cross)
   # Scalar coordinates are normal-equivalent search units, while each
   # sample remains on the exact Brown-projected physical width section.
   profile['offsets']=profile['offsets']*cross
   tangent=np.array([-n[k,1],n[k,0]])
   xy=np.rint(profile['source_uv']).astype(int);valid=profile['valid'].copy()
   ids=np.flatnonzero(valid);valid[ids]&=~self.mask[xy[ids,1],xy[ids,0]]
   if valid.sum()<40:continue
   signal=path['material_signal'];colour=self.values(profile,signal,tangent)
   material=self.values(profile,'dark',tangent) if signal=='blue' else colour
   counts=self.last_tangent_valid_counts.copy()
   valid&=np.isfinite(material)&np.isfinite(colour)
   # Keep independent visible runs: an occlusion cannot become a shoulder.
   edges=np.flatnonzero(np.r_[True,valid[1:]!=valid[:-1],True])
   for start,end in zip(edges[:-1],edges[1:]):
    if not valid[start] or end-start<40:continue
    offsets=profile['offsets'][start:end];values=material[start:end]
    bands=whole_colour_bands(np.zeros(2),np.array([1.,0.]),offsets,values,np.ones(len(offsets),bool),maximum_width=175 if signal=='red' else 110)
    for band in bands:
     lo,hi=band['complete_band_native_interval'];left,right=band['sides'][:,0]
     inside=(profile['offsets']>left)&(profile['offsets']<right)&valid
     outside=(((profile['offsets']>=lo)&(profile['offsets']<left-3))|((profile['offsets']>right+3)&(profile['offsets']<=hi)))&valid
     tint=float(np.median(colour[inside])-np.median(colour[outside])) if inside.any() and outside.any() else 0.
     if signal=='blue' and tint<1.2:
      rejected.append(dict(query=k,reason='dark_band_without_blue_material_support',tint=tint));continue
     quantiles=band['quantiles'];sigma=float(np.max(np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1))/1.349)
     scalar=dict(uv=band['uv'].tolist(),sigma_px=max(1.,sigma),source_evidence=dict(
      query_index=k,actual_source_sides_uv=band['sides'].tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),
      complete_band_native_interval_px=[lo,hi],signed_interface_iqr_contrast=[band['amplitude']*.5]*2,
      transition_sigma_px=max(1.,sigma),material_signal=signal,joint_band_width_px=band['width_px'],
      measurement_origin='current_native_physical_width_complete_paint_ice_transitions',
      boundary_signal='luminance_with_blue_material_identity' if signal=='blue' else 'complete_red_material_profile',
      blue_identity_contrast=tint,actual_source_support_outside_person_and_broadcast_masks=True,
      tangent_median_sampling='only_visible_current_ice_bilinear_contributors',
      tangent_sampling_unit_direction_uv=tangent.tolist(),
      physical_width_profile_normal_equivalent_sample_step=1.,
      tangent_source_valid_sample_count_range=[int(counts[(profile['offsets']>=lo)&(profile['offsets']<=hi)].min()),int(counts[(profile['offsets']>=lo)&(profile['offsets']<=hi)].max())]))
     restored=restore_physical_profile_readout(self.camera,profile,scalar)
     if restored is None:continue
     ev=restored['source_evidence'];sides=np.asarray(ev['actual_source_sides_uv'])
     ev['joint_band_width_px']=float(abs((sides[1]-sides[0])@n[k]))
     ev['physical_width_profile_native_sampling_step_px']=1/cross
     context=np.vstack([restored['uv'],sides,np.asarray(ev['actual_source_side_quantiles_uv']).reshape(-1,2)])
     if (context<[0,0]).any() or (context>[w-1,h-1]).any():continue
     ip=np.rint(context).astype(int)
     if self.mask[ip[:,1],ip[:,0]].any():continue
     rows.append(restored)
  return rows,dict(query_count=len(q),measured_count=len(rows),operator='physical_transverse_current_material_two_interfaces_ground_midpoint',rejections=rejected)


def install_user_guide(extractor):
 path=RD/'user_annotations/user_B2_1248_yellow_midline.json'
 annotation=read(path);line=next(p for p in extractor.paths if p['feature_id']=='B2')
 xy,valid=ground_xy(extractor.cameras[1248],np.asarray(annotation['native_uv']))
 assert valid.all()
 center=xy.mean(0);direction=xy[-1]-xy[0];direction/=np.linalg.norm(direction)
 extent=(np.asarray(line['xyz'])[:,:2]-center)@direction
 world=np.c_[center+np.linspace(extent.min(),extent.max(),257)[:,None]*direction,np.zeros(257)]
 guide=dict(frame=1248,curve_id='B2:segment0',kind='line',uv=annotation['native_uv'],
   authority=annotation['authority'],_world_xyz=world)
 extractor.guides.append(guide);extractor.line_identity.guides.append(guide)


def produce(indices):
 extractor=Extractor(RD,MODEL);install_user_guide(extractor)
 for i in indices:
  source=extractor.rows[i]['source'];raw=cv2.imread(source['image_path'])
  broadcast,_=current_broadcast_exclusion(raw);outside,_=current_ice_visible_domain(i)
  native=PhysicalBandProfiles(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside,extractor.cameras[i])
  features=[];coverage=[]
  for path in extractor.paths:
   if path['feature_id'] not in ('B1','B2','C'):continue
   guide,origin=extractor.guide(path,i)
   if len(guide)<2:continue
   guide,origin,policy=extractor.line_identity.select_guide(path,i,guide,origin)
   feature,_,_=extractor.line(dict(path,_line_identity_policy=policy),guide,native);feature['navigation_origin']=origin
   if feature['point_count']:features.append(feature)
   coverage.append(dict(target_id=feature['target_id'],points=feature['point_count'],navigation_origin=origin,native_material_diagnostic=feature.get('extraction_diagnostic')))
  extractor.line_identity.record(i,features)
  write(RD/f'physical_wide_proposals/{i:06d}.json',dict(source=source,programme_frame_index_1based=i,features=features,
    point_count=sum(f['point_count'] for f in features),per_feature_coverage=coverage,not_applied=True))
  if len(indices)<30 or i%100==0:print(i,[(f['target_id'],f['point_count']) for f in features],flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--indices',default='300,400,500,1245,1246,1247,1248');parser.add_argument('--all',action='store_true');args=parser.parse_args()
 cv2.setNumThreads(1);produce(list(range(201,1249)) if args.all else [int(x) for x in args.indices.split(',')])
