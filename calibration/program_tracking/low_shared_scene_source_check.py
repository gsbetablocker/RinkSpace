"""Check a revised common rink against real low-installation source keys.

No original native UV, boundary or ads move. One C is shared by all low cuts;
R/f are resected per source key. This is a geometry compatibility diagnostic,
not a full-cut camera or accuracy acceptance.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from main7_source_camera_candidate import (B,TMP,arrays,np,sha,common_centre,
    errors,AdGeometry,SceneRenderer,decode_selected,PersonMasks,DEFAULT_MASKS,
    DEFAULT_PEOPLE,PROGRAM)
from ice_refine import _world_curves
from main7_source_bundle_candidate import Bundle,solve
from main_principal_point_source_diagnostic import PrincipalBundle
import cv2


class FixedTemplateOptics(Bundle):
    """Keep historic empirical optics fixed; only common C and R/f can move."""
    def __init__(self,bundle,templates):
        self.__dict__.update(bundle.__dict__);self.templates=templates

    def camera(self,q,p):
        c=self.templates[p].copy();i=self.index[p]
        c[:3]=q[:3]*100.;c[3:6]=q[3+4*i:6+4*i]
        c[6:8]+=q[6+4*i]-np.mean(self.templates[p][6:8])
        return c


def source_keys(base,scene):
    rows=[];bindings=[];curves=_world_curves(scene)
    for shot in (3,5,8):
        file=base/f'shot{shot}_actual_source/source_rows.json'
        bindings.append(dict(path=str(file),sha256=sha(file)))
        for raw in json.loads(file.read_text(encoding='utf8')):
            row=dict(raw)
            for key in ('uv','sigma'):row[key]=np.asarray(row[key],float)
            row['world']=curves[row['id']];row['original_source_split']=row['split']
            row['split']='fit';row['authority']='source identity proposal; geometry compatibility diagnostic'
            rows.append(row)
    adfile=TMP/'program_ad_corners_native/corners_joint_v9.json'
    bindings.append(dict(path=str(adfile),sha256=sha(adfile)))
    addata=json.loads(adfile.read_text(encoding='utf8'))['observations']
    shots=json.loads((PROGRAM/'source_shots.json').read_text(encoding='utf8'))['shots']
    def in_low(p):return any(int(shots[s]['start_pts'])<=p<int(shots[s]['end_pts']) for s in (3,5,8))
    stamps=sorted({r['pts'] for r in rows}|{int(r['pts']) for r in addata if r.get('status')=='measured' and in_low(int(r['pts']))})
    for p in stamps:
        observations=[x for x in addata if int(x['pts'])==p and x.get('status')=='measured']
        if observations:
            ads=[AdGeometry(scene,int(x['user_number'])) for x in observations]
            rows.append(dict(pts=p,id='known_ad_corners',split='fit',point_correspondence=True,
                uv=np.array([x['uv'] for x in observations],float),
                world=np.array([ad.corner(ad.initial,x['role']) for ad,x in zip(ads,observations)]),
                sigma=np.array([x['sigma_px'] for x in observations],float),origin=str(adfile)))
    for shot in (3,5,8):
        file=TMP/f'programme_native_evidence_v13/driver_inputs/source_edges/shot{shot:02d}.npz'
        edge=arrays(file);index={int(p):i for i,p in enumerate(edge['pts'])}
        bindings.append(dict(path=str(file),sha256=sha(file)))
        for p in stamps:
            if p not in index:continue
            i=index[p];lo,hi=map(int,edge['offsets'][i:i+2]);uv=edge['observed_uv'][lo:hi];kind=edge['edge_id'][lo:hi]
            for k,name,owner,key in ((0,'far_top','board_surface','top_edge_xyz'),(1,'far_contact','ice_boundary','points_xyz')):
                observed=uv[(kind==k)&(uv[:,0]>40)&(uv[:,0]<1880)][::6].copy()
                if len(observed)<3:continue
                world=np.asarray(scene[owner][key],float)
                if not np.array_equal(world[0],world[-1]):world=np.vstack([world,world[:1]])
                rows.append(dict(pts=p,id=name,split='fit',uv=observed,world=world,
                    sigma=np.full(len(observed),4. if k==0 else 7.),origin=str(file)))
    return rows,stamps,bindings


def run(a):
    a.output.mkdir(parents=True,exist_ok=True);packet=arrays(B/'candidate_poses.npz')
    original={int(p):c for p,c in zip(packet['pts'],packet['camera25'])}
    oldscene=json.loads((B/'candidate_scene.json').read_text(encoding='utf8'))
    newscene=json.loads(a.scene.read_text(encoding='utf8'));started=time.perf_counter()
    results={};candidates={};binding=None
    for name,scene in [('original_scene',oldscene),('revised_scene',newscene)]:
        rows,stamps,binding=source_keys(a.base,scene)
        bundle=Bundle(stamps,{p:[r for r in rows if r['pts']==p] for p in stamps},[])
        if a.fixed_template_optics:bundle=FixedTemplateOptics(bundle,original)
        seeds=original
        if a.warm:
            warm=arrays(a.warm/f'{name}_key_cameras.npz')
            seeds={int(p):c for p,c in zip(warm['pts'],warm['camera25'])}
        q0=np.r_[np.median(np.array([seeds[p][:3] for p in stamps]),axis=0)/100.,
            np.concatenate([np.r_[seeds[p][3:6],np.mean(seeds[p][6:8])] for p in stamps])]
        if a.shared_principal:
            bundle=PrincipalBundle(bundle);q0=np.r_[q0[:3],np.array([960.,540.])/1000.,q0[3:]]
        solver_output=a.output/name;solver_output.mkdir(exist_ok=True)
        q,history,seconds=solve(bundle,q0,solver_output)
        cameras={p:bundle.camera(q,p) for p in stamps}
        metrics=dict(C=(q[:3]*100).tolist(),shared_principal=None if a.fixed_template_optics else
            (q[3:5]*1000).tolist() if a.shared_principal else [960.,540.],
            optics_status='fixed original per-key empirical optics, not hardware calibration' if a.fixed_template_optics else 'conditional pinhole',
            seconds=seconds,history=history)
        candidates[name]=cameras
        results[name]=dict(metrics=metrics,source_errors=[dict(pts=p,
            original=errors(original[p],[r for r in rows if r['pts']==p]),
            refit=errors(cameras[p],[r for r in rows if r['pts']==p])) for p in stamps])
        np.savez_compressed(a.output/f'{name}_key_cameras.npz',pts=np.array(stamps),camera25=np.array([cameras[p] for p in stamps]))
        serial=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in rows]
        (a.output/f'{name}_source_rows.json').write_text(json.dumps(serial,ensure_ascii=False),encoding='utf8')
    report=dict(accuracy_accepted=False,complete_low_cuts=False,
        revised_scene_path=str(a.scene),revised_scene_sha256=sha(a.scene),
        original_scene_sha256=sha(B/'candidate_scene.json'),bindings=binding,
        assumptions=['one common low-installation C',
            'historic empirical PP/skew/Brown and focal ratio kept exactly fixed, unverified optics' if a.fixed_template_optics else
            'one shared unknown principal point; square-pixel pinhole' if a.shared_principal else 'conditional centred square-pixel pinhole',
            'source identity proposals require review'],
        results=results,seconds=time.perf_counter()-started)
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    source=json.loads((PROGRAM/'source_shots.json').read_text(encoding='utf8'))
    renderer=SceneRenderer(newscene);masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    for p,img in decode_selected(Path(source['source_path']),stamps):
        mask=masks.get(p,require_sam2=True)[0]
        before,_=SceneRenderer(oldscene).draw(img,original[p],mask,title='Original B / low installation')
        after,_=renderer.draw(img,candidates['revised_scene'][p],mask,title='Common revised rink / low shared C / key diagnostic')
        cv2.imwrite(str(a.output/f'{p}_source_comparison.jpg'),np.vstack([cv2.resize(before,(1280,720)),cv2.resize(after,(1280,720))]))
    print(json.dumps(dict(seconds=report['seconds'],results={k:dict(C=v['metrics']['C'],principal=v['metrics']['shared_principal'],seconds=v['metrics']['seconds']) for k,v in results.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','scene','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shared-principal',action='store_true')
    p.add_argument('--warm',type=Path)
    p.add_argument('--fixed-template-optics',action='store_true')
    a=p.parse_args()
    if a.fixed_template_optics and (a.shared_principal or a.warm):p.error('Template optics check starts from B and cannot use pinhole warm/PP mode.')
    run(a)
