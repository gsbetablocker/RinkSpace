"""Materialize and consume the actual Demo image-plane initialization.

The 7,186 H matrices are loaded directly, never estimated by an optimizer and
never converted to old R/C/K. This is a working image-plane projection state,
not a three-camera physical checkpoint. Its world chart is explicitly the old
Demo template; passing current material coordinates as this chart is an error.

CLI:
  python compact_screen_seed.py prepare --source AUDIT_ROOT --out NEW_DIR
  python compact_screen_seed.py check --seed NEW_DIR/seed.json

Runtime:
  seed = ScreenSeed('NEW_DIR/seed.json')
  uv, valid = seed.project_reference_xy(1464832, 's07', xy,
      world_chart_sha256=seed.world_chart_sha256)
  uv, valid, offsets, kinds = seed.project_inventory(1464832, 's07')

An external solver may pass an image-plane delta evaluated by its documented
source-supported time basis. This module does not allocate independent free
per-frame parameters and does not silently optimize anything.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

SCHEMA='rinkspace.compact_image_plane_seed.v1'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _safe_child(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes explicit input root')
    return p


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path,obj):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def _project(H,xy):
    points=np.asarray(xy,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('Expected finite N x 2 points in the declared reference chart')
    q=np.c_[points,np.ones(len(points))]@H.T
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=q[:,:2]/q[:,2:3]
    return uv,q[:,2]


def _inventory(scene):
    """Reproduce the audited renderer's deterministic path sampling and order."""
    paths=[]
    for kind,points in (('contact',scene['ice_boundary']['points_xyz']),
                        ('board_top',scene['board_surface']['top_edge_xyz'])):
        ring=np.asarray(points,float)
        if not np.array_equal(ring[0],ring[-1]):ring=np.vstack((ring,ring[0]))
        paths.append((kind,ring))
    for feature in scene['ice_features']:
        if feature['kind']=='polygon':continue
        if feature['kind']=='circle':
            rings=feature.get('rings',[dict(center_xyz=feature.get('center_xyz'),radius=feature.get('radius'),role='outer')])
            for ring in rings:
                if ring['role']!='outer':continue
                t=np.linspace(0,2*np.pi,181);centre=np.asarray(ring['center_xyz'],float)
                paths.append(('marking',centre+np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]*float(ring['radius'])))
        elif feature['kind']=='polyline':
            for points in feature['segments_xyz']:
                points=np.asarray(points,float)
                paths.append(('marking',np.vstack([np.linspace(x,y,32) for x,y in zip(points[:-1],points[1:])]) if len(points)>1 else points))
    edges={}
    for cell in scene['ice_grid']['cells']:
        polygon=np.asarray(cell['polygon_xyz'],float)
        for x,y in zip(polygon,np.roll(polygon,-1,axis=0)):
            if np.linalg.norm(x-y)<=1e-10:continue
            key=tuple(sorted((tuple(np.round(x,9)),tuple(np.round(y,9)))))
            edges[key]=('grid',np.linspace(x,y,9))
    paths.extend(edges.values())
    for feature in scene['ice_features']:
        if feature['kind']=='polygon':
            polygon=np.asarray(feature['polygon_xyz'],float);omitted=set(feature.get('open_edges',[]))
            if not omitted:paths.append(('marking',np.vstack((polygon,polygon[:1]))))
            else:
                for i,(x,y) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
                    if i not in omitted:paths.append(('marking',np.linspace(x,y,32)))
            continue
        if feature['kind']!='circle':continue
        for ring in feature.get('rings',[]):
            if ring['role']=='outer':continue
            t=np.linspace(0,2*np.pi,181);center=np.asarray(ring['center_xyz'],float)
            paths.append(('marking',center+np.column_stack((np.cos(t),np.sin(t),np.zeros(len(t))))*float(ring['radius'])))
    return [(i,kind,np.asarray(xyz,float)) for i,(kind,xyz) in enumerate(paths) if kind in ('grid','marking')]


class ScreenSeed:
    """Direct initialization used by drawing, navigation and registration code."""
    def __init__(self,manifest):
        self.path=Path(manifest).resolve();self.meta=_read(self.path)
        if self.meta.get('schema')!=SCHEMA:raise ValueError('Unsupported seed schema')
        ref=self.meta['arrays'];p=_safe_child(self.path.parent,ref['file'])
        if file_sha(p)!=ref['sha256']:raise ValueError('Seed arrays changed')
        with np.load(p,allow_pickle=False) as z:self.a={key:z[key].copy() for key in z.files}
        self.world_chart_sha256=self.meta['world_chart_sha256']
        if object_sha(self.meta['world_chart'])!=self.world_chart_sha256:raise ValueError('World chart changed')
        self.pts=self.a['native_pts'];self.shots=self.a['shot_number'];H=self.a['H_reference_xy_to_native']
        if self.pts.dtype.kind not in 'iu' or H.shape!=(len(self.pts),3,3):raise ValueError('Bad clock or H shape')
        if np.any(np.diff(self.pts)<=0) or len(np.unique(self.pts))!=len(self.pts) or not np.isfinite(H).all():raise ValueError('Duplicate/unordered/nonfinite seed')
        if set(self.shots.tolist())!=set(range(2,10)) or np.any(np.diff(self.pts)!=256):raise ValueError('The expected complete native S02-S09 scope is missing')
        for key,value in self.a.items():value.setflags(write=False)

    def frame_index(self,native_pts,shot_id):
        if isinstance(native_pts,bool) or not isinstance(native_pts,(int,np.integer)):raise ValueError('Native PTS must be an integer')
        if shot_id=='s01':raise ValueError('S01 uses its preserved d78 physical forward model, not Demo H')
        i=int(np.searchsorted(self.pts,native_pts))
        if i==len(self.pts) or int(self.pts[i])!=native_pts:raise ValueError('No exact native frame; interpolation or nearest-frame fallback is not performed')
        if shot_id!=f's{int(self.shots[i]):02d}':raise ValueError('Shot/PTS mismatch or cross-cut request')
        return i

    def matrix(self,native_pts,shot_id,delta=None):
        i=self.frame_index(native_pts,shot_id);H=self.a['H_reference_xy_to_native'][i]
        if delta is None:return H.copy()
        d=np.asarray(delta,float)
        if d.shape!=(8,) or not np.isfinite(d).all():raise ValueError('Eight finite SL(3) image-plane coordinates required')
        if not np.any(d):return H.copy()  # Preserve baseline matrix bytes at zero correction.
        from scipy.linalg import expm
        A=np.array([[d[0],d[1],d[2]],[d[3],d[4],d[5]],[d[6],d[7],-d[0]-d[4]]])
        w,h=self.meta['image_size'];N=np.array([[2./w,0.,-1.],[0.,2./h,-1.],[0.,0.,1.]])
        return np.linalg.solve(N,expm(A)@N@H)

    def project_reference_xy(self,native_pts,shot_id,xy,*,world_chart_sha256,delta=None):
        if world_chart_sha256!=self.world_chart_sha256:raise ValueError('Reference chart mismatch; current template coordinates need an explicit geometric bridge')
        i=self.frame_index(native_pts,shot_id);uv,z=_project(self.matrix(native_pts,shot_id,delta),xy)
        valid=np.isfinite(uv).all(axis=1)&(self.a['front_sign'][i]*z>0)
        return uv,valid

    def inverse_native_uv(self,native_pts,shot_id,uv,*,delta=None):
        q=np.asarray(uv,float)
        xy,z=_project(np.linalg.inv(self.matrix(native_pts,shot_id,delta)),q)
        return xy,np.isfinite(xy).all(axis=1)&(np.abs(z)>0)

    def project_inventory(self,native_pts,shot_id,*,delta=None):
        uv,valid=self.project_reference_xy(native_pts,shot_id,self.a['reference_xy'],world_chart_sha256=self.world_chart_sha256,delta=delta)
        return uv,valid,self.a['path_offsets'],self.a['path_kind']

    def project_s01_world(self,native_pts,world,*,source_root):
        """Use the original d78 forward function; no S01 model synthesis.

        The explicit source_root supplies the reviewed local NumPy evaluator.
        Importing this evaluation module does not start a solver.
        """
        import sys
        if isinstance(native_pts,bool) or not isinstance(native_pts,(int,np.integer)):raise ValueError('Integer native PTS required')
        s=self.meta['s01'];scope=s['scope']
        if not scope['first_pts']<=native_pts<=scope['last_pts'] or (native_pts-scope['first_pts'])%256:raise ValueError('Not a native S01 frame')
        path=_safe_child(self.path.parent,s['projection_model']['file'])
        if file_sha(path)!=s['projection_model']['sha256']:raise ValueError('Preserved S01 projection model changed')
        sys.path.insert(0,str(Path(source_root).resolve()/'Project/calibration/rinkfit/src'))
        from rinkfit_start.native_scene import camera_from_event
        camera=camera_from_event(_read(path));time=(native_pts-s['time_origin_pts'])/12800.
        return camera.project_rs(np.asarray(world,float),time)


def prepare(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Choose a new output directory')
    index={x['archive_path']:x for x in _read(source/'FILE_INDEX.json')['files']};checked={}
    def file(relative):
        p=_safe_child(source,relative);ref=index.get(relative)
        if ref is None or p.stat().st_size!=ref['bytes'] or file_sha(p)!=ref['sha256']:raise ValueError('Input missing or changed: '+relative)
        checked[relative]=ref['sha256'];return p
    original=_read(file('direct_grid/state/initial_state.json'));proof=_read(file('direct_grid/probe/layout_evidence.json'))
    scene=_read(file('direct_grid/inputs/scene_8a28_UNACCEPTED.json'));top=_read(file('direct_grid/state/topology.json'))
    current=_read(file('Data/runs/okada_rec_1019/grid_input_v1/model.json'));d78=file('s01/d78/model.json')
    def arr(name):return np.load(file('direct_grid/probe/'+name+'.npy'),allow_pickle=False)
    pts=arr('native_pts');shots=arr('shot_number');H=arr('H_normalized_world_xy_to_native');xy=arr('world_xy');target=arr('target_uv');roles=arr('frame_role_allows_numerical_navigation');valid=arr('probe_valid');vertex_ids=arr('source_vertex_index')
    if not np.all(valid):raise ValueError('No interpolation fallback for absent H')
    if original['source_identity']['sha256']!=current['source_clock']['source_video_sha256']:raise ValueError('Different source video')
    if original['source_identity']['time_base']!='1/12800' or current['source_clock']['frame_step_pts']!=256:raise ValueError('Different native source clock')
    if current['s01_model_sha256']!=file_sha(d78):raise ValueError('S01 identity differs from the intended preserved baseline')
    expected=[];expected_shots=[]
    for sid,shot in current['shots'].items():
        if sid=='s01':continue
        p=np.arange(shot['scope']['first_pts'],shot['scope']['last_pts']+1,256,dtype=np.int64);expected.extend(p.tolist());expected_shots.extend([int(sid[1:])]*len(p))
    order=np.argsort(expected)
    if not np.array_equal(pts,np.asarray(expected)[order]) or not np.array_equal(shots,np.asarray(expected_shots)[order]):raise ValueError('Full native S02-S09 coverage mismatch')
    q=np.einsum('fij,fkj->fki',H,np.concatenate([xy,np.ones((*xy.shape[:-1],1))],axis=-1));replay=q[...,:2]/q[...,2:3]
    error=np.linalg.norm(replay-target,axis=-1);maximum=float(np.max(error));front=np.sign(q[:,0,2])
    if maximum>1e-8 or np.any(front[:,None]*q[...,2]<=0):raise ValueError('Original four-point numerical transfer or front chart failed')
    geometry=_inventory(scene);allxyz=[];offsets=[0];kinds=[];sample_hash_checks=[]
    if len(geometry)!=len(top['paths']):raise ValueError('Original path count mismatch')
    for (i,k,x),record in zip(geometry,top['paths']):
        raw=np.ascontiguousarray(x,dtype='<f8')
        if i!=record['original_renderer_path_index'] or k!=record['kind'] or len(raw)!=record['vertex_end']-record['vertex_begin']:raise ValueError('Original path identity/order/count differs')
        digest=hashlib.sha256(raw.tobytes()).hexdigest()
        sample_hash_checks.append({'original_renderer_path_index':i,'original_sha256':record['original_world_vertex_bytes_sha256'],'local_parameterized_sampling_sha256':digest,'bytes_equal':digest==record['original_world_vertex_bytes_sha256']})
        allxyz.append(raw);offsets.append(offsets[-1]+len(raw));kinds.append(0 if k=='grid' else 1)
    xyz=np.concatenate(allxyz);norm=proof['normalization'];reference_xy=(xyz[:,:2]-np.asarray(norm['origin'])[:2])/norm['L']
    if np.any(xyz[:,2]!=0):raise ValueError('Only the modeled ice plane belongs in this H state')
    # Exact original H/world/UV arrays are the initialized numerical state.
    # Trigonometric sampling may differ by runtime ULPs; keep the original
    # hashes intact and do not call these derived arrays a bytewise recreation
    # of the original 4.714 GB cache. All actual stored grid-vertex witnesses
    # must still agree in the explicit source template chart.
    vertex_error=float(np.max(np.abs(reference_xy[vertex_ids]-xy)))
    if vertex_error>2e-13:raise ValueError('Original grid witness coordinates differ from the reference template')
    world_chart={'schema':'rinkspace.explicit_reference_template_chart.v1','template_sha256':checked['direct_grid/inputs/scene_8a28_UNACCEPTED.json'],'normalization':norm,'axes':'original Demo x/y, right-handed z=0','role':'historical numerical layout, not current geometric truth'}
    out.mkdir(parents=True);arrays=out/'seed_arrays.npz'
    np.savez_compressed(arrays,native_pts=pts,shot_number=shots,H_reference_xy_to_native=H,front_sign=front,
        navigation_role_allowed=roles,reference_xy=reference_xy,path_offsets=np.asarray(offsets,np.int64),path_kind=np.asarray(kinds,np.uint8),
        verification_world_xy=xy,verification_target_uv=target)
    shutil.copyfile(d78,out/'s01_projection_model.json')
    s01=current['shots']['s01']
    meta={'schema':SCHEMA,'initialized_projection_state':True,'physical_joint_checkpoint_initialized':False,
          'source_video_sha256':original['source_identity']['sha256'],'source_video_id':original['source_identity']['id'],'time_base':[1,12800],'native_step_pts':256,
          'image_size':[1920,1080],'world_chart':world_chart,'world_chart_sha256':object_sha(world_chart),
          'frames':len(pts),'first_pts':int(pts[0]),'last_pts':int(pts[-1]),'per_shot':{f's{int(s):02d}':int((shots==s).sum()) for s in np.unique(shots)},
          'arrays':{'file':arrays.name,'sha256':file_sha(arrays),'bytes':arrays.stat().st_size},
          's01':{'scope':s01['scope'],'time_origin_pts':s01['time_origin_pts'],'projection_model':{'file':'s01_projection_model.json','sha256':file_sha(d78)},'complete_auxiliary_state_ref':{'archive_path':'s01/d78/complete_state_manifest.json','sha256':file_sha(file('s01/d78/complete_state_manifest.json'))},'screen_demo_rows':0},
          'inventory':{'paths':len(geometry),'vertices_per_frame':len(xyz),'screen_uv_cache_materialized':False,
                       'coordinate_source':'same SHA-bound parameterized original template, locally sampled for display',
                       'numpy_version':np.__version__,'all_original_world_vertex_hashes_equal':all(x['bytes_equal'] for x in sample_hash_checks),
                       'original_path_hash_mismatches':sum(not x['bytes_equal'] for x in sample_hash_checks),
                       'original_grid_witness_coordinate_max_abs_difference_L':vertex_error,
                       'original_full_cache_bytewise_reconstruction_claimed':False},
          'numeric_transfer':{'four_point_targets':int(error.size),'maximum_error_px':maximum,'rms_error_px':float(np.sqrt(np.mean(error*error))),'H_reestimated':False,'old_R_C_K_imported_into_current_physical_model':False},
          'original_input_sha256':checked,
          'use':{'drawing_and_navigation':'All native frames immediately available; exact existing template projection at zero delta.','registration':'Use source-supported correction coordinates and real-pixel factors in a separate candidate; keep the initial state immutable.','physical_lift':'Needs explicit old/current template reconciliation and a separately initialized physical model.','accuracy':'No independent source-pixel accuracy, metric scale or three-camera calibration is certified.','roles':'Existing protected-frame roles are preserved; reused Demo states do not establish independent validation.','visibility':'Returned validity is finite positive projective depth; it is not board/athlete/subtitle occlusion or the original metric near-plane mask.'}}
    _write(out/'seed.json',meta)
    _write(out/'GEOMETRY_SAMPLING_PROVENANCE.json',{'original_topology_sha256':checked['direct_grid/state/topology.json'],'numpy_version':np.__version__,'checks':sample_hash_checks,'original_hashes_retained':True,'original_complete_cache_replaced':False})
    result=check(out/'seed.json');_write(out/'CHECK.json',result)
    return result


def check(manifest):
    seed=ScreenSeed(manifest);H=seed.a['H_reference_xy_to_native'];xy=seed.a['verification_world_xy'];uv=seed.a['verification_target_uv'];maximum=0.;roundtrip=0.;zero_equal=True
    for i,(pts,sn) in enumerate(zip(seed.pts,seed.shots)):
        sid=f's{int(sn):02d}';zero=seed.matrix(int(pts),sid,np.zeros(8));zero_equal &= np.array_equal(H[i],zero)
        pred,valid=seed.project_reference_xy(int(pts),sid,xy[i],world_chart_sha256=seed.world_chart_sha256,delta=np.zeros(8))
        if not valid.all():raise ValueError('Originally visible verification point became invalid')
        maximum=max(maximum,float(np.max(np.linalg.norm(pred-uv[i],axis=1))));back,good=seed.inverse_native_uv(int(pts),sid,pred)
        if not good.all():raise ValueError('Reference inverse failed')
        roundtrip=max(roundtrip,float(np.max(np.abs(back-xy[i]))))
    if not zero_equal or maximum>1e-8:raise ValueError('Zero correction does not preserve existing Demo')
    cases=[('wrong_shot',lambda:seed.frame_index(int(seed.pts[0]),'s09')),
           ('off_lattice',lambda:seed.frame_index(int(seed.pts[0])+1,'s02')),
           ('s01_wrong_owner',lambda:seed.frame_index(256256,'s01')),
           ('wrong_world_chart',lambda:seed.project_reference_xy(int(seed.pts[0]),'s02',xy[0],world_chart_sha256='current-template-is-not-the-reference-chart'))]
    rejected=[]
    for name,fn in cases:
        try:fn()
        except ValueError:rejected.append(name)
        else:raise ValueError('Expected identity failure did not occur: '+name)
    return {'schema':'rinkspace.compact_screen_seed_check.v1','frames_checked':len(seed.pts),'points_checked':int(len(seed.pts)*4),'all_H_bytes_equal_at_zero_delta':zero_equal,
            'maximum_original_probe_error_px':maximum,'maximum_reference_inverse_error_L':roundtrip,'identity_failures_checked':rejected,
            'inventory_vertices_per_frame':len(seed.a['reference_xy']),'arrays_bytes':seed.meta['arrays']['bytes'],'actual_video_fit_run':False,'physical_joint_fit_accepted':False}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--source',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('check');q.add_argument('--seed',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.out) if a.command=='prepare' else check(a.seed),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
