"""Audit existing source support and re-read visible static ice and print features."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import monotonic
from collections import Counter, defaultdict

import json
import joint_inputs as ji
ji.enable_historical_source_readers()
from joint_inputs import cv2, np, OUT, HIMG, W, TOP, BOTTOM, TB, STEP, sparse_pts
import read_current_material as material
from read_ring_secondary import annulus
from scipy.spatial import cKDTree


def valid_runs(p):
    u,s=p['uv'],p['status']
    v=(s==0)&np.isfinite(u).all(1)&(u[:,0]>3)&(u[:,0]<1917)&(u[:,1]>135)&(u[:,1]<1067)
    return u[v]


def feature_catalog(native):
    ice=ji.source_reader().ice_paths(native,native.scene)
    print_paths=material.print_configs(native,tuple(range(1,27)))
    # Front/back print and inset lettering have different identity. Only
    # model-declared outlines, outside panel edges and badge rims are scanned.
    return ice,print_paths


def packed_projection(shot, native, frames, world, names):
    # Batch several exact exposures through the same saved rolling-row camera.
    # Only these search seeds are projected. Every retained UV is read from RGB.
    cat=list(names)
    z=[world[k] for k in cat]
    offsets=np.r_[0,np.cumsum(list(map(len,z)))]
    stack=np.vstack(z)
    paths={'search':stack}
    times=shot.times
    idx=shot.at(np.asarray(frames))
    pred=ji.project_groups(native,shot.rigs,shot.controls,shot.first,shot.dt,
                           [(int(i),'search',times[i]) for i in idx],paths)
    slices={k:slice(a,b) for k,a,b in zip(cat,offsets[:-1],offsets[1:],strict=True)}
    return {int(p):{'uv':q[0],'status':q[1]} for p,q in zip(frames,pred,strict=True)},slices,stack


def acquire(shot, native, p, raw, mask, z, slices, world, ice, prints, centre, camera_position):
    raw_mask=mask.copy()

    zsl=lambda k:material.subproj(z,slices[k])
    feet=ji.source_reader().visible_foot(zsl(BOTTOM)['uv'][zsl(BOTTOM)['status']==0])
    fit=[]
    for key, rows in ji.source_reader().ice_frame(np.ascontiguousarray(raw[:,:,::-1]),raw_mask,z,ice,slices,feet).items():
        mode,lab,curved=ice[key]
        mark=ji.family(key,native)
        if mode=='house':
            # The dedicated concentric annulus reader below pairs the two
            # transitions. Inner or centre samples retain the published data.
            continue
        fit.append(material.used_rows(key,lab,mark,rows,'measured_paired_paint_edges'))
    for num in ('F1','F2','R1','R2'):
        keys={p['label'].split(':')[-1]:key for key,p in native.paths.items() if p['label'].startswith('ice.'+num+':')}
        if not {'outer','middle'}<=keys.keys():continue
        v=zsl(keys['outer'])
        if len(valid_runs(v))<10:continue
        rows=annulus(raw,raw_mask,v,zsl(keys['middle']))
        if rows and (num[0]=='F' or rows[0]['tone']=='red'):
            for label in ('outer','middle'):
                data=[dict(uv=r[label],source_normal=r['normal'],offset=r['offset'],tone=r['tone'],
                           opposite_uv=r['middle' if label=='outer' else 'outer']) for r in rows]
                fit.append(material.used_rows(keys[label],f'ice.{num}:{label}',num,data,'measured_paired_annulus'))
    fit+=material.rail_reads(raw,raw_mask,shot.id,z,slices,world,centre,camera_position)
    # Do not replace an existing qualified upper blue lip merely because an
    # alternate branch is bright. Edge polarity and rail-strip continuity are
    # part of the existing material readers.
    visprint={k:v for k,v in prints.items() if len(valid_runs(zsl(k)))>=5}
    if visprint:fit+=material.read_prints(raw,raw_mask,z,slices,visprint)
    return fit


def basic_group(g,native):
    if g['path'] not in native.paths:return None
    uv=np.asarray(g.get('uv',[]),float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not len(uv) or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>=[W,HIMG]):
        return None
    return g


def audit_frames(shot,native):
    if shot.evidence is None:return None
    rows=list(ji.lines(shot.evidence['path']))
    np.testing.assert_array_equal([v['pts'] for v in rows],shot.pts)
    counts=Counter(); role_counts=Counter()
    frame_gaps=defaultdict(list)
    for row in rows:
        labels=set()
        for group in row['groups']:
            g=basic_group(group,native)
            if not g:
                counts['unmatched_or_invalid_groups']+=1
                continue
            counts['all_source_rows']+=len(g['uv'])
            role_counts[g.get('material','unspecified')]+=len(g['uv'])
            labels.add(ji.family(g['path'],native))
        for key in labels:frame_gaps[key].append(row['pts'])
    return rows,dict(counts),dict(role_counts),{k:len(v) for k,v in frame_gaps.items()}


def sampled_coverage(native,shot,ev,okada,mapping,selected):
    if ev is not None:
        return {int(p):ev[i] for i,p in zip(shot.at(selected),selected,strict=True)}
    s=shot.shot
    uv,paths,pts=[okada[k][s['row_begin']:s['row_stop']] for k in ('uv','path_indices','pts')]
    use=np.flatnonzero(np.isin(pts,selected))
    rows={int(p):dict(pts=int(p),groups=[],fold='published') for p in selected}
    a=defaultdict(list)
    for i in use:
        key=mapping[paths[i]]
        if key in native.paths:
            a[int(pts[i]),key].append(uv[i].tolist())
    for (p,k),points in a.items():
        rows[p]['groups'].append(dict(path=k,uv=points,material='okada_1102_original_pixels'))
    return rows


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('part',choices=('jin','okada'))
    ap.add_argument('--chunk',type=int,default=0)
    ap.add_argument('--chunks',type=int,default=1)
    ap.add_argument('--stride',type=int,default=8)
    args=ap.parse_args()
    cv2.setNumThreads(1)
    t0=monotonic()
    a,native,shots,ov=ji.inputs()
    ice,prints=feature_catalog(native)
    # The source survey goes through all 24 shots; more samples near the
    # reported windows are actual native PTS, not a resampled timeline.
    ids=[k for k,s in shots.items() if s.program==args.part][args.chunk::args.chunks]
    root=OUT/f'{args.part}_survey_qualified'
    root.mkdir(exist_ok=True)
    if args.part=='jin':
        world={k:v[0] for k,v in ji.native_paths(native,(TOP,BOTTOM,*ice,*prints),count=115).items()}
        ref=world[TOP]
        centre=(ref.min(0)+ref.max(0))/2
        probe=ref.copy()
        probe[:,:2]+=.014*(centre[:2]-probe[:,:2])/np.maximum(.001,np.linalg.norm(centre[:2]-probe[:,:2],axis=1)[:,None])
        world['inward_probe']=probe
        masks=dict(ji.source_reader().mask_index())
        # Raw source scan is conducted on the complete Q874 exposure clock.
        # Static source gates and material colours are inherited from the
        # previously validated Ankara reader.
    else:
        okada=ji.load_arrays(ji.check(ov['source']['observations'])['path'],('uv','pts','path_indices'))
        mapping=ji.read(Path(a['ok_mref']['path']).with_name('mapping.json'))['path_ids']
    for ident in ids:
        shot=shots[ident]
        print('start',ident,len(shot.pts),'exposures',flush=True)
        ev=audit_frames(shot,native)
        selected=sparse_pts(shot,args.stride,focus=True)
        prior=sampled_coverage(native,shot,ev[0] if ev else None,okada if ev is None else None,mapping if ev is None else None,selected)
        fresh=Counter(); existing=Counter(); skipped=Counter()
        latest=root/(ident+'.jsonl')
        with latest.open('x',encoding='utf-8',buffering=1) as fp:
            if args.part=='jin':
                project_cache={}
                for i,(p,raw) in enumerate(ji.source_reader().exact_frames(shot.source,selected.tolist(),time_base=TB)):
                    if p not in project_cache:
                        project_cache,slices,xyz=packed_projection(shot,native,selected[i:i+6],world,world)
                    entry=prior[p]
                    h=sha256(raw.tobytes()).hexdigest()
                    if h!=entry['source_BGR_sha256']:
                        raise ValueError((ident,'original raw frame differs',p))
                    r=shot.at(p)
                    cam=native.core.evaluate_motion_camera(shot.rigs[r],shot.controls,shot.first,shot.dt,np.atleast_1d(shot.times[r]))
                    try:
                        mask=material.mask_frame(masks,p)
                    except (KeyError,ValueError) as e:
                        skipped[type(e).__name__]+=1
                        groups=[]
                    else:
                        groups=acquire(shot,native,p,raw,mask,project_cache.pop(p),slices,xyz,ice,prints,centre,cam['centre'][0])
                    row=dict(pts=p,source_BGR_sha256=h,fold=entry.get('fold',entry.get('prior_fold')),
                             previous=entry['groups'],fresh=groups)
                    for g in groups:fresh[ji.family(g['path'],native)]+=len(g['uv'])
                    for g in entry['groups']:
                        if g['path'] in native.paths:existing[ji.family(g['path'],native)]+=len(g['uv'])
                    fp.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            else:
                for p,entry in prior.items():
                    fp.write(json.dumps(dict(pts=p,fold='published',previous=entry['groups'],fresh=[]),separators=(',',':'))+'\n')
                    for g in entry['groups']:existing[ji.family(g['path'],native)]+=len(g['uv'])
        summary=dict(programme=shot.program,shot=shot.id,first_pts=int(shot.pts[0]),last_pts=int(shot.pts[-1]),
                     exposure_count=len(shot.pts),source_read_count=len(selected) if args.part=='jin' else 0,
                     exposures_selected=len(selected),existing_sampled=existing,
                     fresh_candidates=fresh,missing_source_masks=skipped)
        if ev:summary.update(original_ledger_counts=ev[1],original_ledger_roles=ev[2],original_ledger_coverage=ev[3])
        ji.write(root/(ident+'.json'),summary)
        print('done',ident,len(selected),sum(fresh.values()),len(fresh),'elapsed',round(monotonic()-t0,1),flush=True)


if __name__=='__main__':
    main()
