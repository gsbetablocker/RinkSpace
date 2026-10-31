"""Read original-image materials for independent new-shot registration.

Native paths and independent image motion navigate searches. Actual material
interfaces supply UV; captured candidates remain separate from fit approval.
"""
from pathlib import Path
import argparse,json,hashlib,time
import cv2,numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import qualify_visible_line_fragments
from rinkspace.session.panel_boundaries import read_panel_boundary
from capture_native_panels import native_panel_inventory
from prepare_independent_navigation import frames


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x)for x in v]
    return v


def write(path,value):Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2)+'\n',encoding='utf8')


_SOURCE_MASK_DOCUMENTS={}


def _source_mask_document(path,expected_sha256=None):
    """Cache parsed provenance, invalidating on actual file replacement/change."""
    path=Path(path).resolve();stat=path.stat()
    signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    cached=_SOURCE_MASK_DOCUMENTS.get(path)
    if cached is None or cached[0]!=signature:
        data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
        if path.suffix=='.jsonl':
            value={}
            for line in data.decode('utf8').splitlines():
                if not line.strip():continue
                record=json.loads(line);binding=record['source_binding']
                key=(binding['shot_id'],int(binding['source_pts']))
                if key in value:raise ValueError(f'Duplicate source-mask exposure: {key} in {path}')
                value[key]=record
        else:value=json.loads(data)
        cached=(signature,value,actual);_SOURCE_MASK_DOCUMENTS[path]=cached
    if expected_sha256 is not None and cached[2]!=expected_sha256:
        raise ValueError(f'Source-mask provenance SHA256 mismatch: {path}')
    return cached[1],cached[2]


def _source_mask_path(value,parent):
    path=Path(value)
    return (path if path.is_absolute() else parent/path).resolve()


def _source_mask_record(records,sid,pts):
    record=records.get((sid,int(pts)))
    if record is None:raise ValueError(f'Missing same-exposure source-mask provenance: {sid}/{pts}')
    binding=record['source_binding'];rgb=record.get('decoded_native_RGB_sha256')
    if binding.get('shot_id')!=sid or binding.get('source_pts')!=int(pts) or binding.get('native_hw')!=[1080,1920]:
        raise ValueError(f'Invalid source-mask exposure binding: {sid}/{pts}')
    if not binding.get('source_sha256') or not binding.get('time_base') or not isinstance(rgb,str) or len(rgb)!=64:
        raise ValueError(f'Missing source-mask RGB binding: {sid}/{pts}')
    return record


def _read_source_mask(path,expected_sha256=None):
    data=Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError(f'Source-mask PNG SHA256 mismatch: {path}')
    im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape!=(1080,1920):raise ValueError('Invalid current-source person mask')
    return im


def source_exclusion(root,sid,pts):
    root=Path(root);path=root/'person_masks_refined/full'/sid/f'{pts}.png'
    if path.exists():
        refined_ledger=root/'person_masks_refined/full/frames.jsonl'
        refined_records,refined_sha=_source_mask_document(refined_ledger)
        refined=_source_mask_record(refined_records,sid,pts)
        if _source_mask_path(refined['mask_path'],refined_ledger.parent)!=path.resolve():
            raise ValueError(f'Refined source-mask path does not match exposure: {sid}/{pts}')
        unresolved=refined.get('diagnostic',{}).get('unresolved_support')
        if not isinstance(unresolved,bool):raise ValueError(f'Source-mask unresolved support unknown: {sid}/{pts}')
        im=_read_source_mask(path,refined['mask_sha256'])
        if unresolved:
            # A nonempty partial refined foreground cannot establish support.
            # The root explicitly binds a reusable same-RGB guard record; do
            # not manufacture a source path or silently fall back on failure.
            binding_path=root/'person_mask_guard_binding.json'
            if not binding_path.exists():
                raise FileNotFoundError(f'Unresolved source-mask same-exposure guard binding missing: {binding_path}')
            guard_binding,_=_source_mask_document(binding_path)
            base_ledger=root/'person_masks/frames.jsonl'
            for name,expected in (('base_ledger',base_ledger),('refined_ledger',refined_ledger)):
                reference=guard_binding[name]
                if _source_mask_path(reference['path'],root)!=expected.resolve():
                    raise ValueError(f'Unresolved source-mask ledger binding mismatch: {name}')
            if guard_binding['refined_ledger']['sha256']!=refined_sha:
                raise ValueError(f'Unresolved source-mask refined ledger SHA256 mismatch: {refined_ledger}')
            if not base_ledger.exists():
                raise FileNotFoundError(f'Unresolved source-mask same-exposure base ledger missing: {base_ledger}')
            base_records,_=_source_mask_document(base_ledger,guard_binding['base_ledger']['sha256'])
            base=_source_mask_record(base_records,sid,pts)
            if base['source_binding']!=refined['source_binding'] or base['decoded_native_RGB_sha256']!=refined['decoded_native_RGB_sha256']:
                raise ValueError(f'Unresolved source-mask base is not the same source RGB: {sid}/{pts}')
            base_path=root/'person_masks'/sid/f'{pts}.png'
            if _source_mask_path(base['mask_path'],base_ledger.parent)!=base_path.resolve():
                raise ValueError(f'Unresolved source-mask base path does not match exposure: {sid}/{pts}')
            if not base_path.exists():
                raise FileNotFoundError(f'Unresolved source-mask same-exposure base PNG missing: {base_path}')
            base_im=_read_source_mask(base_path,base['mask_sha256'])
            guard_reference=guard_binding['guard_ledger']
            guard_ledger=_source_mask_path(guard_reference['path'],root)
            if not guard_ledger.exists():
                raise FileNotFoundError(f'Unresolved source-mask same-exposure guard ledger missing: {guard_ledger}')
            guards,_=_source_mask_document(guard_ledger,guard_reference['sha256'])
            guard=guards.get((sid,int(pts)))
            if guard is None:raise ValueError(f'Unresolved source-mask same-exposure guard missing: {sid}/{pts}')
            if (guard['source_binding']!=refined['source_binding'] or
                guard.get('source_RGB_sha256')!=refined['decoded_native_RGB_sha256'] or
                guard.get('unresolved_support') is not True or guard.get('policy')!='base_or_refined_same_RGB'):
                raise ValueError(f'Unresolved source-mask guard RGB/policy mismatch: {sid}/{pts}')
            for name,record,original in (('base_mask',base,base_path),('refined_mask',refined,path)):
                if (guard[name]['sha256']!=record['mask_sha256'] or
                    _source_mask_path(guard[name]['path'],guard_ledger.parent)!=original.resolve()):
                    raise ValueError(f'Unresolved source-mask guard provenance mismatch: {name}/{sid}/{pts}')
            path=_source_mask_path(guard['guard_candidate']['path'],guard_ledger.parent)
            if not path.exists():
                raise FileNotFoundError(f'Unresolved source-mask same-exposure guard PNG missing: {path}')
            guard_im=_read_source_mask(path,guard['guard_candidate']['sha256'])
            if not np.isin(guard_im,[0,255]).all() or not np.array_equal(guard_im,cv2.bitwise_or(base_im,im)):
                raise ValueError(f'Unresolved source-mask guard is not the actual raw binary union: {sid}/{pts}')
            im=guard_im
    else:
        path=root/'person_masks'/sid/f'{pts}.png'
        if not path.exists():raise FileNotFoundError(f'Current original-frame person mask pending: {path}')
        im=_read_source_mask(path)
    # A missing detector/SAM foreground is missing support, not visible background.
    # Keep the whole frame pending until a current-image mask is available.
    if not np.any(im):return np.ones((1080,1920),bool),path
    mask=cv2.dilate((im>0).astype('uint8'),np.ones((7,7),'uint8')).astype(bool)
    mask[:290,:570]=True;mask[940:,1580:]=True
    return mask,path


def native_ice_inventory(cuts,xyz):
    paths=[]
    for cut in cuts:
        if cut['layer']!='ice':continue
        p=xyz[cut['begin']:cut['end']]
        if len(p)<2 or np.linalg.matrix_rank(p-p[0])!=1:continue
        name=cut['label'].removeprefix('ice.');feature,_,role=name.partition(':')
        signal='blue' if feature in ('B1','B2') else 'red' if feature=='C' or feature.startswith(('goal_line','goal_trapezoid')) else 'dark'
        paths.append(dict(curve_id=name,feature_id=feature,world_curve_role=role,native_path_id=cut['id'],xyz=p,
            operator='paint_center',material_signal=signal,material_transition=None,query_sampling='continuous_arc_length',closed=False))
    return paths


def spaced_indices(uv,spacing=48.):
    """Thin redundancy along one supported contour; preserve short features."""
    uv=np.asarray(uv);n=len(uv)
    if n<=4:return np.arange(n)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
    count=max(4,int(np.ceil(arc[-1]/spacing))+1)
    return np.unique(np.argmin(abs(arc[:,None]-np.linspace(0,arc[-1],count)[None]),axis=0))


def capture(args):
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'));base=json.loads((root.parent/'baseline.json').read_text('utf8'))
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts'];lookup={c['id']:c for c in cuts}
    core=NativeCore(base['inputs']['library']['path']);paths=native_ice_inventory(cuts,world['xyz']);panels=[p for p in native_panel_inventory(cuts) if p['panel_instance'] in ('Ad1','Ad2','Ad3','Ad4','Ad5','Ad6','Ad7')]
    for p in panels:
        if p['panel_instance']=='Ad6':p['material_signal']='red'
    reader=IceProgrammeSourceReader(None,source_sha256=index['source']['actual_sha256'],time_base='1/12800',native_paths=paths,
        sample_spacing_px=24.,search_radius_px=40.,separate_bands=True,prepared_source_context=dict(source_sha256=index['source']['actual_sha256'],time_base='1/12800',authority=[dict(path=base['inputs']['scene']['path'],sha256=base['inputs']['scene']['sha256'])]))
    args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);start=time.monotonic()
    write(args.output/'binding.json',dict(schema='rinkspace.independent_material_capture.v1',source_sha256=index['source']['actual_sha256'],baseline=base['inputs'],native_ice_inventory=[{k:v for k,v in p.items()if k!='xyz'}for p in paths],panels=panels,independent_source_capture=True,legacy_new_shot_inputs_read=False,query_spacing_px=24,fit_spacing_px=48,short_feature_minimum_samples=4,nav_is_not_observation=True,person_mask_root=str(root/'person_masks'),pixel_domain='native_distorted',source_time_base='1/12800'))
    for sid in args.shots:
        shot=next(s for s in index['shots']if s['shot_id']==sid);cam=2 if shot['known_rig_id']=='end'else 0
        navfile=root/sid/'navigation/poses_reanchored.npz'
        if not navfile.exists():navfile=root/sid/'navigation/poses.npz'
        nav=dict(np.load(navfile));poses=nav['poses'].copy();pts_index={int(p):i for i,p in enumerate(nav['pts'])};folder=args.output/sid;folder.mkdir(exist_ok=True)
        # Current source navigation is kept verbatim; any revised navigation is an explicit later input.
        np.savez_compressed(folder/'poses.npz',pts=nav['pts'],poses=poses)
        rowfile=folder/'source_rows.jsonl'
        if rowfile.exists():raise FileExistsError(rowfile)
        counts={};total=0;frame_count=0
        representatives={f['actual_pts']for f in shot['frames']}
        with rowfile.open('w',encoding='utf8',buffering=1) as output,(folder/'readouts.jsonl').open('w',encoding='utf8',buffering=1) as notes:
            for fi,(pts,image) in enumerate(frames(index,shot)):
                if args.pts and pts not in args.pts:continue
                if not args.pts and fi%args.stride and pts!=shot['processing']['last_pts'] and pts not in representatives:continue
                mask,mask_path=source_exclusion(root,sid,pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);theta=poses[pts_index[pts]]
                def project(xyz):return core.project_motion(world['rig'][cam],np.tile(theta,(6,1)),-2.,1.,1080,0.,xyz)
                all_projection=project(world['xyz']);all_uv=all_projection['uv'];valid=all_projection['status']==0
                def piece(path):
                    c=lookup[path];p=all_uv[c['begin']:c['end']].copy();p[~valid[c['begin']:c['end']]]=np.nan;return p
                binding=dict(source_sha256=index['source']['actual_sha256'],source_pts=pts,time_base='1/12800',shot_id=sid,pixel_domain='native_distorted',width=1920,height=1080)
                rows=[];reports={};panel_count=0
                for panel in panels:
                    projections=[piece(p)for p in panel['path_ids']]
                    pp,report=read_panel_boundary(rgb,projections,mask,binding=binding,search_radius=112.,spacing=24.,material_signal=panel['material_signal'])
                    reports[panel['panel_instance']]=report
                    for ci,path in enumerate(panel['path_ids']):
                        group=[p for p in pp if p['curve']==ci]
                        for k in spaced_indices([p['uv']for p in group]):
                            q=group[k];rows.append(dict(q,path_id=path,entity=panel['panel_instance'],source_identity_basis='independent known-rig native ordering and complete source colored-panel exterior',capture_family='panel'));panel_count+=1
                iceprojections={p['native_path_id']:dict(uv=piece(p['native_path_id']),valid=np.isfinite(piece(p['native_path_id'])).all(1))for p in paths}
                if args.panels_only:
                    z={};fragments=[];ice_report={'status':'deferred_until_independent_panel_initialization'}
                else:
                    z=reader.read(rgb,pts,None,mask,shot_id=sid,native_projections=iceprojections)
                    fragments,ice_report=qualify_visible_line_fragments(z,list(range(len(paths))),mask)
                for fragment in fragments:
                    ci=fragment['curve'];ids=np.asarray(fragment['candidate_ids'],int)
                    for k in spaced_indices(z['candidate_uv'][ids]):
                        i=int(ids[k]);p=paths[ci];rows.append(dict(path_id=p['native_path_id'],entity=p['curve_id'],uv=z['candidate_uv'][i].tolist(),actual_source_sides=z['candidate_sides'][i].tolist(),actual_source_quantiles=z['candidate_side_quantiles_uv'][i].tolist(),measurement_operator='paired_material_boundary_midline',capture_family='ice_straight',model_UV_synthesized=False,source_identity_basis='pending independent source geometry/topology review',fragment=fragment))
                # Preserve cross-identity conflicts for review; they are never silently treated as independent evidence.
                conflicts=[]
                if rows:
                    xy=np.array([r['uv']for r in rows]);delta=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
                    a,b=np.where(np.triu(delta<3.,1))
                    conflicts=[[int(i),int(j)]for i,j in zip(a,b)if rows[i]['path_id']!=rows[j]['path_id']]
                for row in rows:counts[row['path_id']]=counts.get(row['path_id'],0)+1
                record=dict(pts=pts,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_video_sha256=index['source']['actual_sha256'],mask_path=str(mask_path),mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),fit_eligible=False,review_state='pending_independent_identity_and_material_review',rows=rows,cross_identity_nearby_pairs=conflicts)
                output.write(json.dumps(plain(record),ensure_ascii=False)+'\n');notes.write(json.dumps(plain(dict(pts=pts,panels=reports,ice=ice_report)),ensure_ascii=False)+'\n')
                if args.pts or pts in representatives or fi%250==0:
                    proof=image.copy()
                    for row in rows:
                        xy=tuple(np.rint(row['uv']).astype(int));cv2.circle(proof,xy,3,(0,0,255)if row['capture_family']=='panel'else(0,220,30),-1)
                    for fragment in fragments:
                        ci=fragment['curve'];ids=fragment['candidate_ids'];xy=tuple(np.rint(z['candidate_uv'][ids[len(ids)//2]]).astype(int));cv2.putText(proof,paths[ci]['curve_id'],xy,cv2.FONT_HERSHEY_SIMPLEX,.5,(20,160,0),1,cv2.LINE_AA)
                    cv2.imwrite(str(folder/f'{pts}_source_candidates.jpg'),proof)
                total+=len(rows);frame_count+=1
                if frame_count%10==0 or args.pts:print(json.dumps(dict(shot=sid,frames=frame_count,rows=total,pts=pts,panels=panel_count,ice=len(rows)-panel_count,seconds=time.monotonic()-start)),flush=True)
        write(folder/'coverage.json',dict(frames=frame_count,rows=total,by_path=counts,source_uv_synthesized=False,fit_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shots',nargs='+',default=['s07','s08','s09']);p.add_argument('--pts',nargs='+',type=int);p.add_argument('--stride',type=int,default=1);p.add_argument('--panels-only',action='store_true');capture(p.parse_args())
if __name__=='__main__':main()
