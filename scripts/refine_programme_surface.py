"""Refine fast relative-surface motion; explicitly sampled, never a CCD claim."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,copy,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface,witnesses_packet,validate_response_policy
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('project','prepared','parameter','base-scan','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-evidence',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def emit(event,**kw):print(json.dumps(dict(event=event,seconds=time.monotonic()-started,**kw)),flush=True)
    saved=load(a.parameter);root,_,info,doc,_=load_reference(a.project,a.prepared,a.source_evidence)
    ref=absolute_reference(root,curve_from_packet(saved),root=a.prepared);ref.relative_skate_curve=shoe_curve_from_packet(saved)
    ref.binding=copy.copy(root.binding);ref.binding.reference=ref;ref.binding.x=saved['x'][:17].copy()
    ref.binding._packet=None;ref.binding._parameter_key=None
    geometry=CollisionSurface(ref,doc);base=load(a.base_scan)
    validate_response_policy(base,geometry.response_policy)
    base_record=json.loads(a.base_scan.with_suffix('.json').read_text(encoding='utf8'))
    if base_record['parameter_sha256']!=digest(a.parameter):raise ValueError('Base surface scan is not from this exact saved candidate')
    np.testing.assert_array_equal(base['source_pts'],ref.p['pts'])
    cfg=json.loads(Path(doc['programme_structure']['policy']).read_text(encoding='utf8'))
    travel=float(cfg['geometry']['surface_scale_m']);defect=float(cfg['geometry']['contact_allowance_m'])
    if not travel>0 or not defect>0:raise ValueError('Adaptive geometry scales must be positive')
    def vertices(ts):
        s=ref.at(np.asarray(ts));return geometry.vertices(fk_state(ref,s,np.zeros((len(ts),69))))
    def measure(intervals):
        lo,hi=intervals.T
        mid=(lo+hi)/2
        if np.any((mid==lo)|(mid==hi)):
            raise ArithmeticError('Adaptive time interval reached machine resolution')
        # Adjacent intervals share endpoints. Evaluate each clock in this
        # bounded batch once, then gather the original left/mid/right triples.
        clock,inverse=np.unique(np.r_[lo,mid,hi],return_inverse=True)
        L,M,R=np.split(vertices(clock)[inverse],3)
        motion=np.maximum(np.linalg.norm(M-L,axis=-1).max(1),np.linalg.norm(R-M,axis=-1).max(1))
        curvature=np.linalg.norm(M-(L+R)/2,axis=-1).max(1)
        return mid,motion,curvature

    intervals=np.c_[base['times'][:-1],base['times'][1:]];extra=[];levels=[]
    # Prime the immutable FP64 skin tensors before sharing the model.
    if len(intervals):
        vertices(intervals[0,:1])
    with ThreadPoolExecutor(max_workers=8) as pool:
        while len(intervals):
            following=[];largest_travel=0.;largest_defect=0.;bad_count=0
            batches=(intervals[k:k+32] for k in range(0,len(intervals),32))
            for k,(mid,motion,curvature) in enumerate(pool.map(measure,batches)):
                lo,hi=intervals[k*32:k*32+len(mid)].T
                bad=(motion>travel)|(curvature>defect)
                largest_travel=max(largest_travel,float(motion.max(initial=0)))
                largest_defect=max(largest_defect,float(curvature.max(initial=0)))
                if bad.any():
                    extra.extend(mid[bad].tolist());bad_count+=int(bad.sum())
                    following.extend(np.c_[lo[bad],mid[bad]].tolist());following.extend(np.c_[mid[bad],hi[bad]].tolist())
                if (k+1)%64==0:
                    emit('refinement_progress',level=len(levels)+1,intervals_checked=k*32+len(mid),total_intervals=len(intervals))
            levels.append(dict(level=len(levels)+1,intervals=len(intervals),refined_intervals=bad_count,
                largest_observed_half_interval_vertex_motion_m=largest_travel,largest_midpoint_chord_defect_m=largest_defect))
            emit('refinement_level',**levels[-1]);intervals=np.asarray(following).reshape(-1,2)
    extra=np.unique(extra);records=[];groups=[];counts=[]
    def scan(item):
        i,V=item;w,d=geometry.scan_sample(V,float(extra[i]));selected={}
        for row in w:
            pair=tuple(sorted((int(geometry.parts[row[5]]),int(geometry.parts[row[6]]))))
            if pair not in selected or row[4]<selected[pair][4]:selected[pair]=row
        pairs=d.pop('pairs');d.pop('grip_pairs');pairs_by_part={}
        for f,g in pairs:
            pair=tuple(sorted((int(geometry.parts[f]),int(geometry.parts[g]))));pairs_by_part[pair]=pairs_by_part.get(pair,0)+1
        return i,list(selected.values()),pairs_by_part,d
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k in range(0,len(extra),32):
            V=vertices(extra[k:k+32])
            for i,w,g,d in pool.map(scan,[(k+j,v) for j,v in enumerate(V)]):
                records.extend((i,row) for row in w);groups.extend((i,f,g,n) for (f,g),n in g.items());counts.append(d)
            if k%256==0 or k+32>=len(extra):emit('adaptive_surface_scan',samples=min(k+32,len(extra)),total=len(extra),witnesses=len(records))
    added=witnesses_packet(records);times=np.union1d(base['times'],extra);oldidx=np.searchsorted(times,base['times']);newidx=np.searchsorted(times,extra)
    rowkeys=('sample','vertex','face_vertices','bary','normal','initial_gap','face_a','face_b')
    merged={key:np.concatenate((oldidx[base[key]] if key=='sample' else base[key],newidx[added[key]] if key=='sample' else added[key])) for key in rowkeys}
    order=np.argsort(merged['sample'],kind='stable');merged={key:value[order] for key,value in merged.items()}
    oldgroups=base['pair_counts'].copy();oldgroups[:,0]=oldidx[oldgroups[:,0]]
    newgroups=np.asarray(groups,dtype=int).reshape(-1,4)
    if len(newgroups):newgroups[:,0]=newidx[newgroups[:,0]]
    for key,diagkey in [('crossing_counts','crossing_pairs'),('grip_crossing_counts','source_grip_crossing_pairs'),('topology_uncertain_candidates','topology_uncertain_candidates')]:
        values=np.zeros(len(times),dtype=int);values[oldidx]=base[key];values[newidx]=[c[diagkey] for c in counts];merged[key]=values
    merged.update(times=times,source_seconds=times+geometry.origin,source_pts=base['source_pts'],parts=base['parts'],
        pair_counts=np.concatenate((oldgroups,newgroups)),adaptive_sample=np.isin(times,extra),
        surface_response_policy=np.array(geometry.response_policy))
    np.savez_compressed(a.output/'surface_scan.npz',**merged)
    report=dict(schema='rinkspace.adaptive-surface-scan.v1',parameter_sha256=digest(a.parameter),camera_sha256=info.get('state_sha256'),
        surface_response_policy=geometry.response_policy,
        base_scan=str(a.base_scan),base_scan_sha256=digest(a.base_scan),native_exposures=len(ref.times),base_samples=len(base['times']),
        added_samples=len(extra),total_samples=len(times),selected_witnesses=len(merged['sample']),
        added_crossing_sample_count=sum(c['crossing_pairs']>0 for c in counts),total_crossing_pairs=int(merged['crossing_counts'].sum()),
        observed_motion_scale_m=travel,midpoint_chord_defect_scale_m=defect,levels=levels,
        temporal_scope='Adaptive sampled refinement from actual root-local vertices; observed endpoint/midpoint motion is not a certified speed bound or nonlinear CCD.',
        threshold_meaning='Existing surface residual normalization and conditional contact allowance drive sampling density; neither is an accuracy acceptance threshold.',
        limitations=['Unexamined interior events remain possible without certified swept bounds.','Closed-solid containment and coplanar overlap are not certified.'],
        accuracy_accepted=False,elapsed_seconds=time.monotonic()-started)
    (a.output/'surface_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8');emit('complete',**report)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
