"""Independent source RGB-vector paint readout; never changes frozen batch inputs."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.vision_programme_ice_review import (IceProgrammeSourceReader,
    _batched_actual_profiles,_actual_profile_candidates,_candidate_context_excluded)
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.far_rail_source_components import source_exclusion
from scripts.review_programme_source_inputs import _observed_ice_region,_ice_exclusion
from rinkspace.programme_source_measurement_cache import restore_exact_navigation_queries
from calibration.program_tracking.strict_absolute_source import normalize_selected_ice_rows

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')


def rgb_profiles(rgb,points,normals,intervals):
    size=int(np.ceil(np.max(intervals[:,1]-intervals[:,0])))+1
    offsets=intervals[:,:1]+np.arange(size,dtype=float)[None]
    valid=offsets<=intervals[:,1:]+.25
    tangent=np.c_[-normals[:,1],normals[:,0]]
    samples=(points[:,None,None]+normals[:,None,None]*offsets[:,None,:,None]
        +tangent[:,None,None]*np.arange(-2,3)[None,:,None,None])
    xy=np.rint(samples.reshape(-1,size,2).astype(np.float32)*32.)/32.
    source=cv2.remap(rgb.astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)
    profiles=np.median(source.reshape(len(points),5,size,3),axis=1)
    for i in range(len(points)):
        end=int(valid[i].sum())
        if end<size:profiles[i,end:]=profiles[i,end-1]
    smooth=np.stack([cv2.GaussianBlur(profiles[:,:,k].astype(float),(5,1),1.,
        borderType=cv2.BORDER_REPLICATE) for k in range(3)],axis=-1)
    return offsets,profiles,smooth,valid


def read_rgb_vector_components(point,normal,offsets,rgb,smooth,luma,luma_smooth):
    """Both RGB-vector interfaces, with all source components retained."""
    second=np.diff(luma_smooth,n=2)
    noise=float(1.4826*np.median(np.abs(second-np.median(second))))
    flank=min(10,max(3,len(offsets)//10))
    background=np.linspace(float(np.median(luma[:flank])),float(np.median(luma[-flank:])),len(luma))
    material=np.maximum(background-luma_smooth,0.)
    active=material>max(2.,3*noise,.1*float(material.max()))
    starts=np.flatnonzero(active&~np.r_[False,active[:-1]])
    ends=np.flatnonzero(active&~np.r_[active[1:],False]);components=[]
    for a,b in zip(starts,ends):
        if b-a<3:continue
        if components and a-components[-1][1]-1<=3:components[-1][1]=int(b)
        else:components.append([int(a),int(b)])
    gradient=np.diff(smooth,axis=0);output=[]
    for a,b in components:
        lo,hi=max(0,a-10),min(len(offsets)-1,b+10)
        mass=material[lo:hi+1];total=float(mass.sum())
        if total<=0:continue
        split_offset=float(np.interp(.5,(np.cumsum(mass)-mass*.5)/total,offsets[lo:hi+1]))
        split=int(np.clip(np.searchsorted(offsets,split_offset),lo+1,hi-1))
        inside=int(a+np.argmin(luma_smooth[a:b+1]))
        paint=np.median(rgb[max(a,inside-1):min(b+1,inside+2)],axis=0)
        outside=np.asarray([np.median(rgb[lo:min(lo+4,split)],axis=0),
            np.median(rgb[max(split+1,hi-3):hi+1],axis=0)])
        directions=np.asarray([paint-outside[0],outside[1]-paint],float)
        lengths=np.linalg.norm(directions,axis=1)
        if (lengths<=1.).any():continue
        directions/=lengths[:,None]
        quantiles=[];amounts=[]
        for begin,end,direction in ((lo,split,directions[0]),(split,hi,directions[1])):
            signed=gradient[begin:end]@direction
            weight=np.maximum(signed,0.);amount=float(weight.sum())
            if amount<=max(1.,3*noise):break
            quantile=np.interp([.25,.5,.75],np.r_[0.,np.cumsum(weight)]/amount,offsets[begin:end+1])
            quantiles.append(point+normal*quantile[:,None]);amounts.append(amount)
        if len(quantiles)!=2:continue
        quantiles=np.asarray(quantiles,float);sides=quantiles[:,1];center=sides.mean(0)
        output.append(dict(uv=center,sides=sides,quantiles=quantiles,
            cropped=lo==0 or hi==len(offsets)-1,component_profile_indices=[a,b],
            full_RGB_interface_profile_intervals=[[lo,split],[split,hi]],
            actual_paint_prototype_RGB=paint.tolist(),actual_exterior_RGB=outside.tolist(),
            actual_signed_RGB_gradient_directions=directions.tolist(),actual_signed_RGB_gradient_amount=amounts,
            source_noise_luminance_second_difference=noise))
    return output


def main():
    cv2.setNumThreads(4)
    manifest=json.loads((WORK/'source_pixels/source_measurements.json').read_text())
    identity_path=Path(manifest['camera_identity']);identity=json.loads(identity_path.read_text())
    scene=json.loads((identity_path.parent/'scene_UNACCEPTED.json').read_text())
    roles=json.loads((WORK.parents[3]/'observations/vision_training/paint_role_contract_current.json').read_text())
    reader=IceProgrammeSourceReader(scene,source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],role_contract=roles)
    auth=ROOT/'events/ankara_2026/model_constraints/user_authorized_same_state_model_navigation_20261012_d7d4998380d40a72.json'
    context=SameStateIceModelContextIdentity(manifest['inventory'],state_identity_path=identity_path,
        navigation_authority=dict(path=str(auth),sha256=hashlib.sha256(auth.read_bytes()).hexdigest()),
        source_sha256=manifest['source_sha256'],time_base=manifest['time_base'])
    selector=TemporalIceIdentitySelector(manifest['inventory'],source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],model_context_identity=context)
    with np.load(identity['camera_path']) as camera:
        cameras={int(p):q for p,q in zip(camera['pts'],camera['camera25'])}
        representatives={int(p):int(q) for p,q in zip(camera['pts'],camera['exposure_representative_pts'])}
    ci=next(i for i,v in enumerate(manifest['inventory']) if v['curve_id']=='B1:segment0')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    wanted={229256448,229256704,229256960,229257216,229257472}
    arrays={};frames=[]
    with av.open(manifest['source_path']) as container:
        stream=container.streams.video[0];stream.thread_count=4;container.seek(min(wanted),stream=stream,backward=True)
        for native in container.decode(stream):
            if native.pts is None or native.pts<min(wanted):continue
            if native.pts>max(wanted):break
            if native.pts not in wanted:continue
            pts=int(native.pts);bgr=native.to_ndarray(format='bgr24');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            binding=dict(source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],source_pts=pts,
                actual_pts=representatives[pts],native_pts=pts,shot_id='223403520:R0402_034',native_hw=[1080,1920],
                decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest())
            excluded=source_exclusion(bgr,masks.static_scene_exclusion(pts)[0]);board=measure_board_source_frame(bgr,excluded,binding)
            boundaries=board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
            mask=_ice_exclusion(excluded,boundaries);scope=_observed_ice_region(boundaries)
            old=reader.read(rgb,pts,cameras[pts],mask,shot_id=binding['shot_id']);old['source_binding']=binding
            restore_exact_navigation_queries(old,reader,cameras[pts])
            query_ids=np.flatnonzero(old['query_curve']==ci)
            points=old['query_uv'][query_ids];normals=old['query_normal'][query_ids];intervals=old['query_interval'][query_ids]
            frame=_MaterialProfileFrame(rgb,binding,old['decoded_native_RGB_sha256'])
            offsets,full_rgb,smooth_rgb,valid=rgb_profiles(rgb,points,normals,intervals)
            luma_offsets,luma,luma_smooth,luma_valid=_batched_actual_profiles(frame,points,normals,intervals,np.array(['dark']*len(points)))
            if not np.array_equal(offsets,luma_offsets) or not np.array_equal(valid,luma_valid):raise ValueError('Source profile clocks differ')
            measured={k:v for k,v in old.items()};old_curve=old['query_curve'][old['candidate_query']]
            retained=np.flatnonzero(old_curve!=ci)
            for key in ('uv','sides','query','flags','side_quantiles_uv'):
                measured['candidate_'+key]=np.asarray(old['candidate_'+key])[retained]
            new={key:[] for key in ('uv','sides','query','flags','side_quantiles_uv')};evidence=[]
            for i,query_id in enumerate(query_ids):
                use=valid[i]
                components=read_rgb_vector_components(points[i],normals[i],offsets[i,use],full_rgb[i,use],
                    smooth_rgb[i,use],luma[i,use],luma_smooth[i,use])
                for candidate in components:
                    blocked=any(_candidate_context_excluded(mask,p,normals[i]) for p in [candidate['uv'],*candidate['sides']])
                    new['uv'].append(candidate['uv']);new['sides'].append(candidate['sides']);new['query'].append(query_id)
                    new['flags'].append(int(candidate['cropped'])|(int(blocked)<<1)|(int(len(components)>1)<<2)|8)
                    new['side_quantiles_uv'].append(candidate['quantiles'])
                    evidence.append({key:value for key,value in candidate.items() if key not in ('uv','sides','quantiles')})
            for key in new:
                value=np.asarray(new[key],dtype=old['candidate_'+key].dtype)
                if key in ('uv','sides','side_quantiles_uv'):value=value.reshape((-1,)+old['candidate_'+key].shape[1:])
                measured['candidate_'+key]=np.concatenate([measured['candidate_'+key],value])
            # Candidate RGB-vector pairs are independent source readouts. All
            # old chroma readouts are kept losslessly in this separate delta.
            selected=dict(rows=[],decisions=[])
            normalized=dict(rows=[],rejected_rows=[])
            if representatives[pts]==pts:
                selected=selector.consume(rgb,measured,mask,source_ice_scope_mask=scope)
                normalized=normalize_selected_ice_rows(selected,measured,
                    source_binding=dict(binding,source_path=manifest['source_path']),curve_inventory=manifest['inventory'])
                if normalized['rejected_rows']:raise ValueError('RGB vector row failed source proof gate')
            rows=[r for r in selected['rows'] if r['curve_id']=='B1:segment0']
            for label,measurement in (('original_chroma',old),('RGB_vector',measured)):
                for key in ('query_curve','query_uv','query_normal','query_interval','query_status',
                        'candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv'):
                    arrays[f'{pts}_{label}_{key}']=measurement[key]
            for label,value in (('offsets',offsets),('RGB',full_rgb),('smoothed_RGB',smooth_rgb),('valid',valid),
                    ('luminance',luma),('smoothed_luminance',luma_smooth)):
                arrays[f'{pts}_actual_source_profiles_{label}']=value
            frames.append(dict(source_binding=binding,decoded_native_RGB_sha256=old['decoded_native_RGB_sha256'],
                curve_id='B1:segment0',actual_RGB_profile_query_indices=query_ids.tolist(),
                retained_original_candidate_indices=retained.tolist(),appended_RGB_candidate_prefix=len(retained),
                actual_RGB_candidate_evidence=evidence,selected_B1_rows=rows,
                model_context_decisions=[r for r in selected.get('model_context_decisions',[]) if r['curve_id']=='B1:segment0'],
                B1_decisions=[r for r in selected['decisions'] if r['curve_id']=='B1:segment0'],
                normalized_B1_rows=[r for r in normalized['rows'] if r.get('curve_id')=='B1:segment0'],
                native_alias_actual_readout_only=representatives[pts]!=pts))
            print(json.dumps(dict(native_pts=pts,actual_pts=representatives[pts],RGB_readout_candidates=len(new['uv']),
                B1_centers=sum(len(r['uv']) for r in rows),model_context=frames[-1]['model_context_decisions'],
                decisions=frames[-1]['B1_decisions']),ensure_ascii=False))
    arrays['metadata_json']=json.dumps(dict(schema='rinkspace.actual-RGB-vector-blue-paint-delta.v1',frames=frames,
        source_measurement_method='observed_signed_RGB_vector_interface_q50_midpoint',
        actual_quantization_step_px=1.,actual_interpolation_grid_px=1/32.,
        old_chroma_failure_preserved=True,width_used_to_fill_or_smooth_UV=False,
        source_observations_inserted_into_FIT=False,frozen_batch_modified=False,
        full_programme_accuracy_review_complete=False),ensure_ascii=False,separators=(',',':'),
        default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item())
    output=WORK/'B1_actual_RGB_vector_two_side_delta_229256448_229257472.npz'
    if output.exists():raise ValueError('No evidence overwrite')
    np.savez_compressed(output,**arrays)


if __name__=='__main__':main()
