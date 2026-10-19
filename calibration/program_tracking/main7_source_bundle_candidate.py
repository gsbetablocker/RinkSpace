"""Joint source-ray bundle for shot 7, with one main-installation centre.

The source correspondences are real adjacent exposures, not a smoothing prior.
Both endpoints remain variables, so a later absolute rink/ad observation can
correct earlier accumulated drift. Other main-camera cuts contribute only
their separately sourced absolute keyframes; no cross-cut motion is invented.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
if str(P) not in sys.path:sys.path.insert(0,str(P))
from main7_source_camera_candidate import (B,TMP,arrays,sha,pinhole,q7,source_rows,
    value_jac,value_only,errors,AdGeometry,SceneRenderer,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE,decode_selected)
from ice_refine import _world_curves
import cv2,numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve


def external_rows(scene,base):
    """Import immutable source pixels from independently classified main cuts."""
    rows=json.loads((base/'tail_absolute_v1/source_rows.json').read_text(encoding='utf8'))
    for row in rows:
        for k in ('world','uv','sigma'):row[k]=np.asarray(row[k],float)
    source=base/'pelvis/main4_geometry/main4_source_anchors.json'
    curves=_world_curves(scene);far=np.asarray(scene['ice_boundary']['points_xyz'],float)
    far=far[far[:,1]>20];curves['far_contact']=far.copy();curves['far_top']=far.copy()
    curves['far_top'][:,2]=scene['parameters']['board_height']
    for frame in json.loads(source.read_text(encoding='utf8')):
        pts=int(frame['source_pts'])
        for row in frame['source_observations']:
            name=row['id'];nameworld='far_top' if name=='board_top' else 'far_contact' if name=='ice_board_contact' else name
            uv=np.asarray(row['uv'],float)
            rows.append(dict(pts=pts,id=name,split='fit',uv=uv,world=curves[nameworld],
                sigma=np.full(len(uv),row['sigma_px']),origin=str(source)))
        ad=frame['ad'];geom=AdGeometry(scene,int(ad['id']));uv=np.asarray(ad['original_uv'],float)
        rows.append(dict(pts=pts,id='known_ad_corners',split='fit',uv=uv,
            world=np.asarray([geom.corner(geom.initial,role) for role in ad['roles']]),
            sigma=np.full(len(uv),ad['sigma_px']),point_correspondence=True,origin=str(source)))
    bindings=[dict(path=str(source),sha256=sha(source)),
        dict(path=str(base/'tail_absolute_v1/source_rows.json'),sha256=sha(base/'tail_absolute_v1/source_rows.json'))]
    lane_file=base/'tail_lane_diagnostic/source_rows.json'
    if lane_file.exists():
        lane=json.loads(lane_file.read_text(encoding='utf8'))
        if any(int(r['pts'])!=231705344 or not r['id'].startswith('lane') for r in lane):
            raise ValueError('Separately reviewed tail Lane source packet changed scope')
        for row in lane:
            for k in ('world','uv','sigma'):row[k]=np.asarray(row[k],float)
            row['original_source_split']=row['split'];row['split']='fit'
            row['fit_authority']='parent explicitly requested reviewed actual Lane source fitting; not a blind holdout'
        rows.extend(lane);bindings.append(dict(path=str(lane_file),sha256=sha(lane_file)))
    return rows,bindings


def relative_value_jac(first,second,a,b,*,jacobian=True):
    """Analytic native-pixel pair residual, derivative for both R/logf endpoints.

    First and second are batches of 4-vectors; matches index their endpoints
    before this call. The centre cancels for a stationary optical installation.
    """
    unique_a,inverse_a=np.unique(first,axis=0,return_inverse=True)
    unique_b,inverse_b=np.unique(second,axis=0,return_inverse=True)
    def rotation(q):
        values=[cv2.Rodrigues(x[:3]) for x in q]
        return np.array([x[0] for x in values]),np.array([x[1].reshape(3,3,3) for x in values])
    ra,da=rotation(unique_a);rb,db=rotation(unique_b)
    ra,da=ra[inverse_a],da[inverse_a];rb,db=rb[inverse_b],db[inverse_b]
    return _relative_kernel(np.exp(first[:,3]),np.exp(second[:,3]),ra,rb,da,db,a,b,jacobian=jacobian)


def _relative_kernel(fa,fb,ra,rb,da,db,a,b,*,jacobian=True,principal=(960.,540.),principal_jacobian=False):
    """Identical pair geometry with rotations cached once per exposure."""
    local_a=np.c_[(a-principal)/fa[:,None],np.ones(len(a))]
    ray=np.einsum('ni,nij->nj',local_a,ra)
    local=np.einsum('ni,nji->nj',ray,rb)
    if not np.isfinite(local).all() or np.any(local[:,2]<=.05):
        raise ValueError('Source pair trial left the forward optical domain')
    normalized=local[:,:2]/local[:,2:]
    projected=fb[:,None]*normalized+principal
    if not jacobian:return (projected-b).ravel()
    d_uv=np.zeros((len(a),2,3));d_uv[:,0,0]=fb/local[:,2];d_uv[:,1,1]=fb/local[:,2]
    d_uv[:,:,2]=-fb[:,None]*local[:,:2]/local[:,2:]**2
    prev=np.zeros((len(a),2,4));curr=np.zeros_like(prev)
    for k in range(3):
        d_ray=np.einsum('ni,nij->nj',local_a,da[:,k])
        d_local=np.einsum('ni,nji->nj',d_ray,rb)
        prev[:,:,k]=np.einsum('nij,nj->ni',d_uv,d_local)
        d_local=np.einsum('ni,nji->nj',ray,db[:,k])
        curr[:,:,k]=np.einsum('nij,nj->ni',d_uv,d_local)
    d_a=-local_a.copy();d_a[:,2]=0.
    d_local=np.einsum('ni,nij->nj',d_a,ra)
    d_local=np.einsum('ni,nji->nj',d_local,rb)
    prev[:,:,3]=np.einsum('nij,nj->ni',d_uv,d_local)
    curr[:,:,3]=fb[:,None]*normalized
    if principal_jacobian:
        pp=np.zeros((len(a),2,2))
        for k in range(2):
            d_a=np.zeros_like(local_a);d_a[:,k]=-1./fa
            d_ray=np.einsum('ni,nij->nj',d_a,ra);d_local=np.einsum('ni,nji->nj',d_ray,rb)
            pp[:,:,k]=np.einsum('nij,nj->ni',d_uv,d_local);pp[:,k,k]+=1.
        return (projected-b).ravel(),prev.reshape(-1,4),curr.reshape(-1,4),pp.reshape(-1,2)*1000.
    return (projected-b).ravel(),prev.reshape(-1,4),curr.reshape(-1,4)


class Bundle:
    def __init__(self,stamps,observations,pairs):
        self.stamps=stamps;self.index={p:i for i,p in enumerate(stamps)};self.observations=observations;self.view_offset=3
        a,b,ia,ib,sigma=[],[],[],[],[]
        for pa,pb,x,y in pairs:
            a.append(x);b.append(y);ia.extend([self.index[pa]]*len(x));ib.extend([self.index[pb]]*len(x))
            sigma.extend([max(2.,np.sqrt(len(x))/3.)]*len(x))
        self.a=np.vstack(a) if a else np.empty((0,2));self.b=np.vstack(b) if b else np.empty((0,2))
        self.ia=np.asarray(ia,dtype=np.int32);self.ib=np.asarray(ib,dtype=np.int32)
        self.motion_sigma=np.repeat(np.asarray(sigma),2)
        self.motion_rows=2*len(self.a)
        ia2=np.repeat(self.ia,2);ib2=np.repeat(self.ib,2)
        self.swap_rows=ia2>ib2
        lower=np.minimum(ia2,ib2);upper=np.maximum(ia2,ib2)
        columns=[np.c_[3+4*lower[:,None]+np.arange(4),3+4*upper[:,None]+np.arange(4)].astype(np.int32).ravel()]
        counts=[np.full(self.motion_rows,8,dtype=np.int32)]
        offset=self.motion_rows
        self.abs_groups=[]
        for p in stamps:
            rows=observations.get(p,[])
            if not rows:continue
            count=2*sum(len(r['uv']) for r in rows);self.abs_groups.append((p,rows))
            cols=np.r_[np.arange(3),3+4*self.index[p]+np.arange(4)]
            columns.append(np.broadcast_to(cols,(count,7)).astype(np.int32).ravel());counts.append(np.full(count,7,dtype=np.int32));offset+=count
        self.indices=np.concatenate(columns);self.indptr=np.r_[0,np.cumsum(np.concatenate(counts),dtype=np.int64)].astype(np.int32)
        if len(self.indices)>=np.iinfo(np.int32).max:raise ValueError('Source graph exceeds 32-bit sparse index capacity')
        self.residual_rows=offset
        absolute_sigma=[np.concatenate([np.repeat(r['sigma'],2) for r in rows]) for p,rows in self.abs_groups]
        self.native_residual_sigma=np.r_[self.motion_sigma,np.concatenate(absolute_sigma) if absolute_sigma else np.empty(0)]

    def camera(self,q,p):return pinhole(np.r_[q[:3],q[3+4*self.index[p]:7+4*self.index[p]]])

    def evaluate(self,q,jacobian=True):
        views=q[3:].reshape(-1,4)
        rotations=[cv2.Rodrigues(x[:3]) for x in views]
        rot=np.asarray([v[0] for v in rotations]);drot=np.asarray([v[1].reshape(3,3,3) for v in rotations]) if jacobian else None
        focal=np.exp(views[:,3]);values=[];data=[]
        for lo in range(0,len(self.a),65536):
            hi=min(lo+65536,len(self.a));ia=self.ia[lo:hi];ib=self.ib[lo:hi];scale=self.motion_sigma[2*lo:2*hi]
            relative=_relative_kernel(focal[ia],focal[ib],rot[ia],rot[ib],None if drot is None else drot[ia],
                None if drot is None else drot[ib],self.a[lo:hi],self.b[lo:hi],jacobian=jacobian)
            if jacobian:
                r,ja,jb=relative;block=np.c_[ja,jb];swap=self.swap_rows[2*lo:2*hi]
                block[swap]=np.c_[jb[swap],ja[swap]];data.append((block/scale[:,None]).ravel())
            else:r=relative
            values.append(r/scale)
        for p,rows in self.abs_groups:
            if jacobian:
                v,j=value_jac(self.camera(q,p),rows);data.append(j.ravel())
            else:v=value_only(self.camera(q,p),rows)
            values.append(v)
        r=np.concatenate(values)
        if not jacobian:return r
        j=csr_matrix((np.concatenate(data),self.indices,self.indptr),shape=(self.residual_rows,len(q)))
        return r,j


def huber_cost(r,robust):
    if not robust:return .5*float(r@r)
    a=np.abs(r);return float(np.where(a<=2.,.5*a*a,2.*a-2.).sum())


def solve(bundle,q0,output,*,verbose=True):
    """Sparse analytic LM; rejected optical trials never become observations."""
    q=q0.copy();history=[];started=time.perf_counter();damping=1e-3
    for robust in (False,True):
        for iteration in range(30):
            r,j=bundle.evaluate(q);cost=huber_cost(r,robust)
            weight=np.minimum(1.,2./np.maximum(np.abs(r),1e-12)) if robust else np.ones(len(r))
            jw=j.multiply(np.sqrt(weight)[:,None]);normal=(jw.T@jw).tocsc();gradient=j.T@(weight*r)
            diagonal=np.maximum(normal.diagonal(),1.)
            accepted=False;reason='';step=np.zeros_like(q)
            for trial in range(12):
                step=spsolve(normal+diags(damping*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
                nextq=q+step
                try:
                    if not np.isfinite(nextq).all() or not .1<100*nextq[2]<100 or np.any((nextq[bundle.view_offset+3::4]<np.log(100))|(nextq[bundle.view_offset+3::4]>np.log(50000))):
                        raise ValueError('Candidate numerical parameter domain')
                    nextcost=huber_cost(bundle.evaluate(nextq,False),robust)
                    if nextcost<cost:
                        q=nextq;accepted=True;damping=max(damping*.3,1e-12);break
                    reason='objective_not_lower'
                except (ValueError,RuntimeError) as exc:reason=str(exc)[:120]
                except Exception as exc:
                    # Typed physical domain errors are failed LM trials. All
                    # other programming errors must remain fatal and visible.
                    from reference_board_pair import BoardIntersectionError
                    if not isinstance(exc,BoardIntersectionError):raise
                    reason=exc.reason if hasattr(exc,'reason') else str(exc)[:120]
                damping*=10.
            item=dict(phase='huber' if robust else 'linear',iteration=iteration,cost=cost,
                next_cost=nextcost if accepted else cost,accepted=accepted,trials=trial+1,damping=damping,
                normalized_gradient=float(np.max(np.abs(gradient)/np.sqrt(diagonal))),
                max_step=float(np.max(np.abs(step))),failure_reason=None if accepted else reason)
            # Parameter tolerances depend on C/R/f conditioning. Measure the
            # proposed step in the actual native pixel observations instead;
            # Once every linearized native-source change is below 0.05 px,
            # extra cost digits cannot resolve visible multi-pixel errors.
            # A dense graph must not keep iterating just because millions of
            # residuals make tiny subpixel changes add up in the objective.
            item['maximum_linearized_source_step_px']=float(np.max(np.abs((j@step)*bundle.native_residual_sigma)))
            history.append(item)
            (output/'bundle_iterations.json').write_text(json.dumps(history,indent=2),encoding='utf8')
            np.savez_compressed(output/'bundle_partial.npz',pts=np.array(bundle.stamps),
                camera25=np.array([bundle.camera(q,p) for p in bundle.stamps]))
            if verbose and iteration%5==0:print(json.dumps(dict(stage=item['phase'],iteration=iteration,cost=item['next_cost'],seconds=time.perf_counter()-started)),flush=True)
            if not accepted or item['maximum_linearized_source_step_px']<.05:break
    return q,history,time.perf_counter()-started


def run(directory,output,seed_file=None,main4_directory=None):
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scene=json.loads((B/'candidate_scene.json').read_text(encoding='utf8'));rows,bindings=source_rows(scene)
    extra,external_bindings=external_rows(scene,directory.parents[1]);rows.extend(extra);bindings.extend(external_bindings)
    seed=arrays(directory/'continuous_candidate_cameras.npz');pooled=arrays(directory/'pooled_common_C_keys.npz')
    canonical={int(p):int(r) for p,r in zip(seed['pts'],seed['exposure_representative_pts'])}
    main4=None
    if main4_directory is not None:
        main4_directory=Path(main4_directory)
        main4=arrays(main4_directory/'main4_B_reparameterized_numerical_seed.npz')
        if np.any(main4['camera25'][:,11:]!=0) or np.any(main4['camera25'][:,6]!=main4['camera25'][:,7]):
            raise ValueError('Main4 numerical initializer is not the same declared pinhole model')
        canonical.update({int(p):int(r) for p,r in zip(main4['pts'],main4['exposure_representative_pts'])})
        file=main4_directory/'main4_bundle_absolute_source_rows.json'
        added=json.loads(file.read_text(encoding='utf8'))
        for row in added:
            for k in ('world','uv','sigma'):row[k]=np.asarray(row[k],float)
            if row['pts'] not in canonical or not 229702144<=row['pts']<229945856:
                raise ValueError('Main4 absolute row outside its own classified cut')
        rows.extend(added);bindings.append(dict(path=str(file),sha256=sha(file)))
    stamps=sorted(set(canonical.values())|{canonical.get(r['pts'],r['pts']) for r in extra});observations={p:[] for p in stamps}
    seen=set()
    for row in sorted(rows,key=lambda r:r['pts']):
        if row['split']!='fit':continue
        p=canonical.get(row['pts'],row['pts'])
        key=(p,row['id'])
        if key in seen:continue
        seen.add(key);observations[p].append(row)
    # Raw native far-board pixels stay source observations. Prefer an already
    # reviewed row for the same physical curve/exposure; no residual selection.
    edge=arrays(TMP/'programme_native_evidence_v13/driver_inputs/source_edges/shot07.npz')
    far=np.asarray(scene['ice_boundary']['points_xyz'],float)
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    if not np.array_equal(far[0],far[-1]):far=np.vstack([far,far[0]])
    if not np.array_equal(top[0],top[-1]):top=np.vstack([top,top[0]])
    for i,p in enumerate(edge['pts']):
        p=int(p)
        if canonical[p]!=p:continue
        lo,hi=map(int,edge['offsets'][i:i+2]);uv=edge['observed_uv'][lo:hi];kind=edge['edge_id'][lo:hi]
        for n,name,world in ((0,'far_top',top),(1,'far_contact',far)):
            if (p,name) in seen or (p,'board_top' if n==0 else 'ice_board_contact') in seen:continue
            selected=uv[(kind==n)&(uv[:,1]<650)&(uv[:,0]>40)&(uv[:,0]<1880)][::6].copy()
            if len(selected)>=3:observations[p].append(dict(pts=p,id=name,split='fit',uv=selected,world=world,
                sigma=np.full(len(selected),4.),origin=str(TMP/'programme_native_evidence_v13/driver_inputs/source_edges/shot07.npz')))
    initial={canonical[int(p)]:c for p,c in zip(seed['pts'],seed['camera25'])}
    if main4 is not None:initial.update({canonical[int(p)]:c for p,c in zip(main4['pts'],main4['camera25'])})
    initial.update({canonical.get(int(p),int(p)):c for p,c in zip(pooled['pts'],pooled['camera25'])})
    # An independently resected near/short-end source key initializes the
    # correct perimeter branch. It is only an initializer; all residuals below
    # still use the unchanged actual source pixels and shared physical scene.
    near_seed=directory/'near_key_consistency/candidate.npz'
    if near_seed.exists():
        near=arrays(near_seed)
        initial.update({canonical[int(p)]:c for p,c in zip(near['pts'],near['camera25'])})
    centre=pooled['camera25'][0,:3]
    if seed_file is not None:
        latest=arrays(seed_file)
        if not np.array_equal(latest['pts'],seed['pts']):raise ValueError('Warm source bundle seed lacks the exact native PTS set')
        initial.update({canonical[int(p)]:c for p,c in zip(latest['pts'],latest['camera25'])})
        key_file=Path(seed_file).parent/'shared_main_key_cameras.npz'
        if key_file.exists():
            key=arrays(key_file);initial.update({canonical.get(int(p),int(p)):c for p,c in zip(key['pts'],key['camera25'])})
        centre=latest['camera25'][0,:3]
    q0=np.r_[centre/100.,np.concatenate([q7(initial[p])[3:] for p in stamps])]
    m=arrays(directory/'source_motion_matches.npz');pairs=[]
    for i,(pa,pb) in enumerate(m['pair_pts']):
        lo,hi=map(int,m['offsets'][i:i+2]);pairs.append((int(pa),int(pb),m['source_uv'][lo:hi],m['target_uv'][lo:hi]))
    if main4 is not None:
        file=main4_directory/'main4_source_motion_matches.npz';m4=arrays(file)
        for i,(pa,pb) in enumerate(m4['pair_pts']):
            if not 229702144<=pa<229945856 or not 229702144<=pb<229945856:
                raise ValueError('Main4 motion pair crosses its classified cut')
            lo,hi=map(int,m4['offsets'][i:i+2]);pairs.append((int(pa),int(pb),m4['source_uv'][lo:hi],m4['target_uv'][lo:hi]))
        bindings.append(dict(path=str(file),sha256=sha(file)))
    bundle=Bundle(stamps,observations,pairs)
    # Check both-endpoint analytic source-ray derivatives on real rows before
    # trusting a large sparse graph. Preserve the numerical evidence report.
    pa,pb,a,b=pairs[0];ia,ib=bundle.index[pa],bundle.index[pb];x=q0[3:].reshape(-1,4)[ia];y=q0[3:].reshape(-1,4)[ib]
    r,ja,jb=relative_value_jac(np.tile(x,(len(a),1)),np.tile(y,(len(a),1)),a,b)
    analytic=np.c_[ja,jb];fd=[];h=1e-7
    for k in range(8):
        xp,yp=x.copy(),y.copy();xm,ym=x.copy(),y.copy()
        (xp if k<4 else yp)[k%4]+=h;(xm if k<4 else ym)[k%4]-=h
        rp=relative_value_jac(np.tile(xp,(len(a),1)),np.tile(yp,(len(a),1)),a,b)[0]
        rm=relative_value_jac(np.tile(xm,(len(a),1)),np.tile(ym,(len(a),1)),a,b)[0]
        fd.append((rp-rm)/(2*h))
    fd=np.array(fd).T;relative=np.linalg.norm(analytic-fd)/np.linalg.norm(fd)
    check=dict(source_pair_pts=[pa,pb],real_rows=len(a),maximum_difference=float(np.max(np.abs(analytic-fd))),relative_difference=float(relative))
    (output/'real_source_pair_jacobian_check.json').write_text(json.dumps(check,indent=2),encoding='utf8')
    if relative>1e-6:raise ValueError('Both-endpoint source-pair derivative failed finite differences')
    q,history,seconds=solve(bundle,q0,output)
    cameras={p:bundle.camera(q,p) for p in stamps};pts=seed['pts']
    if main4 is not None:
        np.savez_compressed(output/'main4_candidate_cameras.npz',pts=main4['pts'],
            camera25=np.array([cameras[canonical[int(p)]] for p in main4['pts']]),
            exposure_representative_pts=main4['exposure_representative_pts'],status=np.full(len(main4['pts']),'global_source_candidate_unvalidated'))
    np.savez_compressed(output/'candidate_cameras.npz',pts=pts,camera25=np.array([cameras[canonical[int(p)]] for p in pts]),
        exposure_representative_pts=seed['exposure_representative_pts'],status=np.full(len(pts),'global_source_candidate_unvalidated'))
    np.savez_compressed(output/'shared_main_key_cameras.npz',pts=pooled['pts'],
        camera25=np.array([cameras[canonical.get(int(p),int(p))] for p in pooled['pts']]))
    report=dict(schema='rinkspace.main7-source-ray-bundle.v1',accuracy_accepted=False,
        method='both real source-ray endpoints variable; absolute source marks; shared main centre; no smoothing prior',
        native_frames=len(pts),canonical_frames=len(set(canonical.values())),main4_native_frames=0 if main4 is None else len(main4['pts']),
        separate_other_cut_keyframes=len(extra),
        residual_rows=bundle.residual_rows,unknowns=len(q),seconds=seconds,shared_C=(q[:3]*100.).tolist(),
        source_bindings=bindings,source_ray_jacobian_check=check,history=history,
        source_groups=[dict(pts=r['pts'],id=r['id'],split=r['split'],origin=r['origin'],points=len(r['uv'])) for r in rows],
        source_absolute_errors=[dict(native_pts=r['pts'],errors=errors(cameras[canonical.get(r['pts'],r['pts'])],[r])) for r in rows],
        centre_is_conditional=True,lens_assumptions=['square_pixels','principal960_540','zero_skew','zero_Brown'])
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    old=arrays(B/'candidate_poses.npz');lookup={int(p):i for i,p in enumerate(old['pts'])}
    renderer=SceneRenderer(scene);masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    source=json.loads((P/'source_shots.json').read_text(encoding='utf8'))
    inspect=[230518016,230547968,230565376,230593280,230571776,230676480,230699520,230708224,230722560,230779136,230898432]
    for p,image in decode_selected(Path(source['source_path']),inspect):
        mask=masks.get(p,require_sam2=True)[0]
        before,_=renderer.draw(image,old['camera25'][lookup[p]],mask,title='Existing saved B / source PTS '+str(p))
        after,_=renderer.draw(image,cameras[canonical[p]],mask,title='Joint source-ray + absolute marks / unvalidated')
        cv2.imwrite(str(output/f'{p}_before_after.jpg'),np.vstack([cv2.resize(before,(1280,720)),cv2.resize(after,(1280,720))]))
    print(json.dumps(dict(stage='complete',shared_C=report['shared_C'],seconds=seconds,accuracy_accepted=False)),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=Path)
    ap.add_argument('--main4-directory',type=Path)
    args=ap.parse_args();run(args.directory,args.output,args.seed,args.main4_directory)
