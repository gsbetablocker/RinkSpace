
"""Compare actual Q874 scene bindings with the Okada provisional final."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
PROG=ROOT/'events/ankara_2026/programmes'
OKADA=PROG/'okada_mei_sp'
JIN=PROG/'jin_shuxian_sp'
sys.path.insert(0,str(ROOT/'src'))

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.storage import AssetStore

ASSETS=AssetStore(ROOT)
OUT=ASSETS.data/'runs/native_programme_20261031/review_okada_q874_1113'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path, expected=None):
    path=ASSETS.resolve(path, must_exist=True)
    with path.open('rb') as f:
        sha=hashlib.file_digest(f,'sha256').hexdigest()
    if expected and sha!=expected:
        raise ValueError('Pinned input was modified: '+str(path))
    return dict(path=str(path),sha256=sha)


def check(ref):
    return pin(ref['path'],ref['sha256'])


def write(path,row):
    with path.open('x',encoding='utf8') as f:
        json.dump(row,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def numeric_hash(v):
    return hashlib.sha256(v.tobytes()).hexdigest()


def load_arrays(path,names):
    with np.load(path,allow_pickle=False) as x:
        return {k:x[k] for k in names}


def make_scene(path,core,values,value_sha=None):
    if value_sha and numeric_hash(values)!=value_sha:
        raise ValueError('Scene values do not match their saved revision')
    scene=Path(path)
    raw=read(scene)
    model=SceneParameterModel(raw,scene,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,values)
    native=NativeSceneGPU(scene,core,parameter_state=state)
    return state,native,raw


def sample(native,key,n=129):
    p=native.paths[key]
    lo,hi=(0.,native.core.board_perimeter(native.scene['board'])) if p['kind']==7 else (p.get('support') or (0.,1.))
    t=np.linspace(lo,hi,n)
    if p['kind']==8:
        s,z=np.asarray(p['vertices']).T
        return np.asarray(native.core.board_points(native.scene['board'],np.interp(t,p['knots'],s),np.interp(t,p['knots'],z)))
    return np.asarray(native.core.sample_path(native.scene['board'],p['kind'],p['parameters'],p.get('vertices',[]),t))


def changes(a,b):
    result=[]
    for k in sorted(set(a)|set(b)):
        x,y=a[k] if k in a else None,b[k] if k in b else None
        if x is None and y is None:continue
        if x is None or y is None or not np.array_equal(x,y):
            result.append(dict(block=k,before=None if x is None else x.tolist(),
                               after=None if y is None else y.tolist()))
    return result


def compare(a,b):
    common=set(a.paths)&set(b.paths)
    unchanged=[]
    moved={}
    for k in sorted(common):
        try:
            x,y=sample(a,k),sample(b,k)
        except Exception as e:
            raise ValueError((k,a.paths[k]['kind'],b.paths[k]['kind'])) from e
        z=np.linalg.norm(y-x,axis=1)
        if z.max()>2.e-12:
            moved[k]=dict(layer=b.paths[k]['layer'],label=b.paths[k]['label'],
                          max_world_L=float(z.max()),median_world_L=float(np.median(z)))
        else:
            unchanged.append(k)
    return dict(before=len(a.paths),after=len(b.paths),unchanged_paths=unchanged,moved_paths=moved,
                added_paths=sorted(set(b.paths)-set(a.paths)),removed_paths=sorted(set(a.paths)-set(b.paths)))


def context():
    original_ref=pin(OKADA/'fit_v11/camera_solver_refined_1102.json')
    original=read(original_ref['path'])
    old_receipt=read(original['review']['render_snapshot'])
    frozen=Path(old_receipt['frozen_run_root'])
    ok_mref=pin(frozen/'joint_manifest.json',old_receipt['manifest_sha256'])
    om=read(ok_mref['path'])
    old_scene=pin(om['scene_path'],om['scene_file_sha256'])
    old_core_ref=check(original['native_library'])
    old_core=NativeCore(old_core_ref['path'])
    ok_sref=check(original['state'])
    orig=load_arrays(ok_sref['path'],('scene_values','rig','controls','stations'))
    os,on,oread=make_scene(old_scene['path'],old_core,orig['scene_values'])

    registered=read(JIN/'joint_review_1113.json')
    ji_ref=check(registered['previous_binding'])
    ji=read(ji_ref['path'])
    if ji['revision']!='q874':
        raise ValueError('The registered Jin predecessor is not Q874')
    previous=read(check(ji['previous_numeric_binding'])['path'])
    before_q874=[previous[k]==ji[k] for k in ('scene_values_sha256','s14_scene_values_sha256','s14_scene','base_full_state')]
    jm_ref=check(ji['prepared_manifest'])
    jm=read(jm_ref['path'])
    jc_ref=pin(jm['native_library'],jm['native_library_sha256'])
    jc=NativeCore(jc_ref['path'])
    base_ref=check(ji['base_full_state'])
    state=load_arrays(base_ref['path'],('scene_values',))
    base_scene=pin(jm['scene_path'],jm['scene_file_sha256'])
    js,jn,jread=make_scene(base_scene['path'],jc,state['scene_values'],ji['scene_values_sha256'])
    s14_ref=check(ji['s14_calibration'])
    s14_values=load_arrays(s14_ref['path'],('scene_values','scene_path','scene_values_sha256','scene_file_sha256','placement_ids','placements'))
    s14_scene=check(ji['s14_scene'])
    if (str(s14_values['scene_file_sha256'])!=s14_scene['sha256'] or
        str(s14_values['scene_values_sha256'])!=ji['s14_scene_values_sha256']):
        raise ValueError('Q874 S14 has an unexpected scene reference')
    ss,sn,sread=make_scene(s14_scene['path'],jc,s14_values['scene_values'],ji['s14_scene_values_sha256'])
    source=read(OKADA/'rebuild_scope.json')
    return locals()


def run():
    a=context()
    os,on,js,jn,ss,sn=[a[k] for k in ('os','on','js','jn','ss','sn')]
    change0=changes(os,js)
    change1=changes(js,ss)
    g0=compare(on,jn)
    g1=compare(jn,sn)
    g2=compare(on,sn)
    b={
        'okada_final':a['original_ref'],'okada_state':a['ok_sref'],'okada_scene':a['old_scene'],
        'okada_camera_core':a['old_core_ref'],'okada_manifest':a['ok_mref'],
        'q874_index':a['ji_ref'],'q874_manifest':a['jm_ref'],'q874_base_state':a['base_ref'],
        'q874_common_scene':a['base_scene'],'q874_geometry_core':a['jc_ref'],'q874_s14_state':a['s14_ref'],
        'q874_s14_scene':a['s14_scene'],'q874_reused_prior_geometry':all(a['before_q874']),
        'q874_only_changed_source_shots':a['ji']['changed_source_shots'],
        'okada_final_scene_values_sha256':numeric_hash(os.values),
        'q874_common_scene_values_sha256':numeric_hash(js.values),'q874_s14_scene_values_sha256':numeric_hash(ss.values),
        'board_equal':np.array_equal(on.scene['board'],sn.scene['board']),
        'original_to_q874_common_blocks':change0,'q874_common_to_s14_blocks':change1,
        'original_to_q874_common_paths':g0,'q874_common_to_s14_paths':g1,'original_to_s14_paths':g2,
        's14_saved_placement_ids':a['s14_values']['placement_ids'].tolist(),
    }
    write(OUT/'geometry_comparison.json',b)
    print('Q874 uses Q873 geometry',b['q874_reused_prior_geometry'],'with changes to',b['q874_only_changed_source_shots'])
    print('State sizes',len(os.values),len(js.values),len(ss.values),'Board numeric equal',b['board_equal'])
    for key,blocks in [('old -> base',change0),('base -> S14',change1)]:
        print(key,'changed blocks',len(blocks))
        for r in blocks:
            x,y=r['before'],r['after']
            if x is None or y is None:print(r)
            else:print(r['block'],[(j,float(u),float(v)) for j,(u,v) in enumerate(zip(x,y,strict=True)) if u!=v])
    for g in (g0,g1,g2):
        print('paths',g['before'],g['after'],'moved',len(g['moved_paths']),'added',len(g['added_paths']),'removed',g['removed_paths'])
        print('largest geometry moves',sorted(g['moved_paths'].items(),key=lambda kv:-kv[1]['max_world_L'])[:14])
    print('S14 saved installation IDs',b['s14_saved_placement_ids'])

if __name__=='__main__':run()
