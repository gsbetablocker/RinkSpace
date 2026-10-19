"""Collect unchanged actual source keys from the three camera installations.

This only copies source evidence and numerical camera seeds. It performs no
camera/scene solve and keeps the secondary independent checks outside fitting.
"""
from __future__ import annotations
import argparse, hashlib, json
from functools import lru_cache
from pathlib import Path
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from main7_source_camera_candidate import B, TMP, PROGRAM, arrays, sha
from low_shared_scene_source_check import source_keys


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    return value


def pixel_hash(uv):
    return hashlib.sha256(np.asarray(uv,np.float64).tobytes()).hexdigest()


@lru_cache(maxsize=32)
def origin_ad_points(origin):
    """Read role labels from the original measured packet, not its world XYZ."""
    file=Path(origin);result=[]
    if not file.exists() or file.suffix!='.json':return result
    data=json.loads(file.read_text(encoding='utf8'))
    if isinstance(data,dict) and 'observations'in data:
        result.extend(x for x in data['observations']if x.get('status')=='measured')
    elif isinstance(data,list):
        for frame in data:
            for ad in frame.get('ads',[frame['ad']] if 'ad'in frame else []):
                for role,uv in zip(ad.get('roles',[]),ad.get('uv',ad.get('original_uv',[]))):
                    result.append(dict(pts=frame.get('source_pts',frame.get('pts')),uv=uv,
                                       user_number=ad['id'],role=role))
    elif isinstance(data,dict) and 'fit_row'in data and 'profiles'in data:
        # This original native ad7 packet explicitly identifies each measured
        # half-height intersection as left_top or left_bottom.
        for item in data['profiles']:
            result.append(dict(pts=data['source_pts'],uv=item['half_height_intersection'],user_number=7,role=item['role']))
    return result


def ad_roles_from_original_pixels(row, ad_observations):
    """Restore measured ad roles by source PTS/UV, never nearest world pixels."""
    if row.get('physical_ids'): return list(row['physical_ids'])
    if not row.get('point_correspondence'): return None
    native=int(row.get('source_native_pts',row['pts']))
    source=[x for x in ad_observations if int(x['pts'])==native and x.get('status')=='measured']
    source+=origin_ad_points(str(row.get('origin','')))
    result=[]
    for uv in np.asarray(row['uv'],float):
        hits=[x for x in source if np.array_equal(np.asarray(x['uv'],float),uv)]
        identities={f"ad{x['user_number']}:{x['role']}" for x in hits}
        if len(identities)!=1: return None
        result.append(identities.pop())
    return result


def source_row(raw, ad_observations):
    row=serial(dict(raw))
    row['point_correspondence']=bool(row.get('point_correspondence',False))
    row['uv']=np.asarray(row['uv'],float).reshape(-1,2).tolist()
    row['world']=np.asarray(row['world'],float).reshape(-1,3).tolist()
    count=len(row['uv'])
    sigma=row.get('sigma',row.get('sigma_px'))
    row['sigma']=(np.full(count,sigma).tolist() if np.isscalar(sigma) else serial(sigma))
    if len(row['sigma'])!=count: raise ValueError('Source sigma shape changed')
    row['source_uv_float64_sha256']=pixel_hash(row['uv'])
    row['source_uv_modified']=False
    physical=ad_roles_from_original_pixels(row,ad_observations)
    if physical:
        row['physical_ids']=physical
        row['point_identity_binding']='original measured source PTS and bit-exact native UV; no world-nearest inference'
    return row


def collect(a):
    a.output.mkdir(parents=True,exist_ok=True)
    scene=json.loads(a.scene.read_text(encoding='utf8'))
    shotdata=json.loads((PROGRAM/'source_shots.json').read_text(encoding='utf8'))
    shots=shotdata['shots']
    def cut_of(p):
        return next(i for i,s in enumerate(shots) if int(s['start_pts'])<=p<int(s['end_pts']))
    adfile=TMP/'program_ad_corners_native/corners_joint_v9.json'
    ad_observations=json.loads(adfile.read_text(encoding='utf8'))['observations']
    packet=json.loads(a.main.read_text(encoding='utf8'))
    cameras=arrays(a.main_seed);main_seed=dict(zip(map(int,cameras['pts']),cameras['camera25']))
    groups=[];held=[]
    for stamp,source in packet.items():
        p=int(stamp);rows=[source_row(r,ad_observations) for r in source]
        groups.append(dict(pts=p,cut=cut_of(p),installation=0,camera25_seed=main_seed[p].tolist(),rows=rows,
            camera_seed_status='full actual-source bundle candidate, not accuracy accepted',
            optics_policy='conditional centered square-pixel zero-Brown, not measured hardware'))
    secondary=json.loads(a.secondary.read_text(encoding='utf8'))
    def secondary_group(frame,independent):
        p=int(frame['pts']);observed=[]
        for raw in frame.get('curves',[])+frame.get('points',[]):
            row=dict(raw,pts=p,split='holdout' if independent else 'fit',
                     original_source_split=raw.get('split','unknown'))
            row['uv']=raw.get('uv',raw.get('source_uv'))
            row['world']=raw.get('world',raw.get('world_xyz'))
            row['id']=raw.get('id',raw.get('curve_id'))
            row['sigma']=raw.get('sigma',raw.get('sigma_px'))
            row['origin']=raw.get('origin',str(a.secondary))
            if raw.get('point_correspondence'):
                number=raw.get('ad_id',raw.get('user_number'));role=raw.get('role')
                if number is not None and role is not None: row['physical_ids']=[f'ad{number}:{role}']
                elif ':' in row['id']: row['physical_ids']=[row['id']]
            observed.append(source_row(row,ad_observations))
        return dict(pts=p,cut=int(frame['shot_index']),installation=1,camera25_seed=frame['camera25'],rows=observed,
            camera_seed_status=frame.get('camera_seed_kind','continuous actual source graph branch seed; unaccepted'),
            optics_policy='conditional centered square-pixel zero-Brown, not measured hardware',
            independent_of_geometry_key_fit=independent)
    groups.extend(secondary_group(f,False) for f in secondary['rows'])
    held.extend(secondary_group(f,True) for f in secondary.get('held_rows',[]))
    lowrows,lowpts,lowbindings=source_keys(a.base/'low_geometry_checks',scene)
    original=arrays(B/'candidate_poses.npz');templates=dict(zip(map(int,original['pts']),original['camera25']))
    lowwarm=arrays(a.low_seed);low_seed=dict(zip(map(int,lowwarm['pts']),lowwarm['camera25']))
    for p in lowpts:
        groups.append(dict(pts=p,cut=cut_of(p),installation=2,camera25_seed=low_seed[p].tolist(),
            camera25_optics_template=templates[p].tolist(),rows=[source_row(r,ad_observations)for r in lowrows if r['pts']==p],
            camera_seed_status='fixed historical optics source-key compatibility candidate; unaccepted',
            optics_policy='keep original per-key PP/skew/Brown and logfx-logfy fixed; common installation C and R/f only'))
    groups.sort(key=lambda x:(x['installation'],x['pts']));held.sort(key=lambda x:x['pts'])
    missing=[dict(pts=g['pts'],id=r['id'],origin=r['origin'])for g in groups for r in g['rows']
             if r['point_correspondence'] and not r.get('physical_ids')]
    result=dict(schema='rinkspace.three-installation-unchanged-source-keys.v1',groups=groups,held_groups=held,
        scene_path=str(a.scene),scene_sha256=sha(a.scene),source_path=shotdata['source_path'],
        source_timebase_denominator=15360,source_native_dimensions=[1920,1080],source_native_rate=60,
        no_new_source_uv=True,no_new_scene=True,no_new_solve=True,accuracy_accepted=False,
        point_identity_unresolved=missing,
        input_bindings=[dict(path=str(f),sha256=sha(f))for f in (a.main,a.main_seed,a.secondary,a.low_seed,adfile)]+lowbindings,
        secondary_provenance_sidecar=str(a.secondary.parent/'source_provenance_and_integrity.json'),
        secondary_point_policy='points copied exactly once; ads field is the same table and is not copied again',
        ignored_nonmodel_image_reference='B was never assigned a world line, never fitted and is excluded',
        geometry_rebinding_policy='source UV/normal/support remain fixed; replace world only by retained physical identity in each candidate scene')
    file=a.output/'all_installation_source_keys.json';file.write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
    summary=dict(path=str(file),sha256=sha(file),counts={str(n):sum(g['installation']==n for g in groups)for n in range(3)},
        source_curve_groups=sum(len(g['rows'])for g in groups),source_samples=sum(len(r['uv'])for g in groups for r in g['rows']),
        held_key_count=len(held),held_samples=sum(len(r['uv'])for g in held for r in g['rows']),
        unresolved_point_identity=missing,source_pixels_unchanged=True,no_solver_run=True)
    (a.output/'package_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','scene','main','main-seed','secondary','low-seed','output'):
        p.add_argument('--'+name,type=Path,required=True)
    collect(p.parse_args())
