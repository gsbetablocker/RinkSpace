"""Replay the current full main source Jacobian and eliminate R/f nuisance.

No camera optimization, candidate state export, source decoding, or rendering.
The one optical hypothesis is installation-shared k1=k0+s*log(f/f_reference).
Native source ray matches are kept unchanged and assume the existing fixed-C
infinite-ray objective; this information test does not assign material depth.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,time
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import diags,save_npz,vstack
from scipy.sparse.linalg import splu,lsmr
from z190_main_source_objective_audit import CURRENT,IDENTIFIER,ref,jsonable
from z190_main_source_authority_rf_restore import AUTHORITY,AUTHORITY_SHA,ADMISSION_SHA,numeric_identity
from main7_source_camera_candidate import sha
from main7_source_bundle_candidate import Bundle
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA,SCENE_UNIT_M
from z190_zoom_radial_source_audit import ZoomRadialBundle,zoom_radial_kernel
from z190_physical_camera import PhysicalCameraModel,ViewState
from z190_low_opening_rf_refine import convert_row
from z190_source_camera_diagnostic import absolute_value_jac15
from continuous_central_brown4_optin import enable_projection_backend

CURRENT_PHYSICAL=CURRENT/'z190_full_programme_pelvis_source_v6/z190_full_programme_camera_candidate.npz'
HELD=CURRENT/'z190_main_ad2_two_prespecified_held_sources_v1/held_source_candidate.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,record):Path(path).write_text(json.dumps(jsonable(record),ensure_ascii=False,indent=2),encoding='utf8')
def uv_sha(row):return hashlib.sha256(np.asarray(row['uv'],np.float64).tobytes()).hexdigest()

def prepare(args):
    """Current physical-only R/f, complete authoritative fit, original full rays."""
    if sha(args.authority)!=AUTHORITY_SHA:raise ValueError('Complete authority inventory changed')
    packet=read(args.authority);ar=packet['source_authority_admission']
    if ar['sha256']!=ADMISSION_SHA or sha(ar['path'])!=ADMISSION_SHA:raise ValueError('Admission changed')
    admission=read(ar['path'])
    if admission['scene_sha256']!=HARD_SCENE_SHA or admission['source_sha256']!=SOURCE_SHA:raise ValueError('Scene/source changed')
    original=admission['original_packet']
    if sha(original['path'])!=original['sha256']:raise ValueError('Original absolute inventory changed')
    old=read(original['path'])
    if Counter(map(numeric_identity,old['fit_rows']))!=Counter(map(numeric_identity,packet['fit_rows']+packet['pending_rows'])):
        raise ValueError('Admission changed numeric measurements/geometry/multiplicity')
    if old['held_rows']!=packet['held_rows'] or old['untouched_future_groups']!=packet['untouched_future_groups']:
        raise ValueError('Held/future changed')
    if len(packet['pending_rows'])!=68 or any(r['split']!='unfitted_identity_pending' for r in packet['pending_rows']):
        raise ValueError('Expected exactly 68 unfitted identity-pending rows')
    if any(r['split']!='fit' for r in packet['fit_rows']):raise ValueError('Nonfit row admitted')
    for role in ('fit_rows','held_rows','pending_rows'):
        for r in packet[role]:
            if uv_sha(r)!=r['source_uv_float64_sha256']:raise ValueError('UV trace changed')
    binding_path=args.current.with_suffix('.binding.json');binding=read(binding_path)
    if binding['exported_camera_sha256']!=sha(args.current) or binding['scene_sha256']!=HARD_SCENE_SHA:
        raise ValueError('Current whole physical binding changed')
    if args.current.resolve()!=CURRENT_PHYSICAL.resolve():raise ValueError('Use current v6 physical only')
    whole,native_all,aliases_all,views_all=PhysicalCameraModel.load_npz(args.current)
    calibration=whole.calibrations[IDENTIFIER]
    views={p:v for p,v in views_all.items()if v.installation_id==IDENTIFIER}
    aliases={n:p for n,p in aliases_all.items()if p in views}
    native=np.asarray([int(n)for n in native_all if int(n)in aliases])
    if len(views)!=5591 or len(native)!=6642:raise ValueError('Incomplete current main exposures')
    if not np.isclose(calibration.pivot_xyz[2]*SCENE_UNIT_M,1.6229072902078119,atol=1e-12,rtol=0):
        raise ValueError('Current independent saved main height changed')
    manifest_ref=admission['current_source_manifest']
    if sha(manifest_ref['path'])!=manifest_ref['sha256']:raise ValueError('Current source binding changed')
    source_manifest=read(manifest_ref['path'])
    component=next(b for b in binding['fitted_installation_bindings']if b['exposures']==5591)
    if component['source_input_manifest']!=manifest_ref:raise ValueError('v6 is not numeric v5 main source component')
    baseline=read(admission['baseline_full_source_manifest']['path']);argv=baseline['run_command_argv']
    def arg(name):return Path(argv[argv.index('--'+name)+1])
    base,previous,scene=map(arg,['base','previous-source-data','scene'])
    if sha(scene)!=HARD_SCENE_SHA:raise ValueError('Hard geometry changed')
    seven=previous/'candidate_cameras.npz'
    if not seven.exists():seven=previous/'main7_candidate_cameras.npz'
    paths={1:base/'main1_geometry/main1_B_reparameterized_numerical_seed.npz',4:previous/'main4_candidate_cameras.npz',
           7:seven,9:base/'main9_geometry/main9_B_reparameterized_numerical_seed.npz'}
    cuts={};classification=[]
    for cut,path in paths.items():
        z=np.load(path,allow_pickle=False)
        for n,p in zip(z['pts'],z['exposure_representative_pts']):
            n,p=int(n),int(p)
            if aliases.get(n)!=p or p in cuts and cuts[p]!=cut:raise ValueError('Exposure alias/cut changed')
            cuts[p]=cut
        classification.append(dict(cut=cut,source_inventory=ref(path),numeric_camera_arrays_read=False))
    if set(cuts)!=set(views):raise ValueError('Incomplete exposure classification')
    observed={p:[]for p in views}
    for r in packet['fit_rows']:observed[aliases[r['pts']]].append(convert_row(r))
    expected={229181696,229189376,229192448,229193984}
    actual={aliases[r['pts']]for r in packet['fit_rows']if r['id'].startswith('ad2_right_external')}
    if actual!=expected:raise ValueError('Four direct Ad2 fit exposures changed')
    held=read(args.held)
    if len(held)!=1 or held[0]['source_native_pts']!=229191168 or held[0]['pts']!=229190912:
        raise ValueError('Prespecified 41.300 corner-held scope changed')
    if held[0]['split']!='holdout' or held[0]['source_scene_sha256']!=HARD_SCENE_SHA or uv_sha(held[0])!=held[0]['source_uv_float64_sha256']:
        raise ValueError('New held source trace/split changed')
    if any(uv_sha(r)==uv_sha(held[0])for r in packet['fit_rows']):raise ValueError('Held corners entered fit')
    rays={};raybindings=[]
    for b in source_manifest['source_bindings']:
        if b.get('role')!='within_cut_observed_source_ray_pairs':continue
        if sha(b['path'])!=b['sha256']:raise ValueError('Ray source changed')
        arc=np.load(b['path'],allow_pickle=False);z={k:arc[k]for k in arc.files};pairs=[]
        for i,(pa,pb)in enumerate(z['pair_pts']):
            pa,pb=int(pa),int(pb)
            if aliases.get(pa)!=pa or aliases.get(pb)!=pb or cuts[pa]!=b['cut'] or cuts[pb]!=b['cut'] or pa==pb:
                raise ValueError('Source pair direction/exposure changed')
            lo,hi=map(int,z['offsets'][i:i+2]);pairs.append((pa,pb,z['source_uv'][lo:hi],z['target_uv'][lo:hi]))
        rays[b['cut']]=pairs
        raybindings.append(dict(**b,pairs=len(pairs),points=len(z['source_uv']),bad_bridges_excluded=z.get('excluded_original_pair_indices',[])))
    if set(rays)!={1,4,7,9} or len(rays[1])!=1384 or sum(len(x[2])for x in rays[1])!=246171:
        raise ValueError('Full original source rays changed')
    return SimpleNamespace(packet=packet,admission=admission,calibration=calibration,views=views,aliases=aliases,native=native,
        cuts=cuts,observed=observed,rays=rays,raybindings=raybindings,classification=classification,
        baseline=baseline,scene=scene,source_manifest=source_manifest,held=held,binding_path=binding_path,
        model=PhysicalCameraModel({IDENTIFIER:calibration}))

def make_bundle(g,cut):
    stamps=sorted(p for p in g.views if g.cuts[p]==cut)
    original=Bundle(stamps,g.observed,g.rays[cut])
    base=StandingSourceBundle(original,g.calibration,prior_wide_focal_px=g.baseline['frozen_wide_prior_focal_px'],
        fixed_height=True,fixed_height_scene_units=g.calibration.pivot_xyz[2])
    zoom=ZoomRadialBundle(base,'k1_slope')
    rf=np.concatenate([np.r_[g.views[p].view.rvec_world_to_camera,g.views[p].view.log_f_ratio+np.log(1920.)]for p in stamps])
    q0=np.r_[base.initial_shared(),rf];q=np.r_[np.asarray(g.calibration.pivot_xyz)/100.,0.,0.,0.,rf]
    for p in stamps:
        if not np.array_equal(zoom.camera(q,p),g.model.camera25(g.views[p])):raise ValueError('Slope-zero camera25 not bitexact')
    return base,zoom,q0,q

def derivative_checks(g,cut):
    """Both real source endpoints, including dk/dlogf at a nonzero math probe."""
    target={1:229193984,7:230676480}.get(cut,min(p for p in g.views if g.cuts[p]==cut and g.observed[p]))
    target=g.aliases.get(target,target)
    ip,pair=next(((i,x)for i,x in enumerate(g.rays[cut])if target in x[:2]),(0,g.rays[cut][0]))
    pa,pb,a,b=pair;a=np.asarray(a[:8],float);b=np.asarray(b[:8],float)
    ca,cb=g.views[pa].view,g.views[pb].view
    va=np.r_[ca.rvec_world_to_camera,ca.log_f_ratio+np.log(1920.)]
    vb=np.r_[cb.rvec_world_to_camera,cb.log_f_ratio+np.log(1920.)]
    pp=np.asarray([1920*g.calibration.principal_x_fraction[0],1080*g.calibration.principal_y_fraction[0]])
    def kernel(va,vb,s,jac):
        ra,da=cv2.Rodrigues(va[:3]);rb,db=cv2.Rodrigues(vb[:3]);n=len(a)
        xa=va[3]-np.log(1920.)-g.calibration.q_reference;xb=vb[3]-np.log(1920.)-g.calibration.q_reference
        return zoom_radial_kernel(np.full(n,np.exp(va[3])),np.full(n,np.exp(vb[3])),np.repeat(ra[None],n,axis=0),
            np.repeat(rb[None],n,axis=0),np.repeat(da.reshape(3,3,3)[None],n,axis=0),np.repeat(db.reshape(3,3,3)[None],n,axis=0),
            a,b,np.repeat(pp[None],n,axis=0),np.repeat(pp[None],n,axis=0),np.full(n,g.calibration.k1[0]+s*xa),
            np.full(n,g.calibration.k1[0]+s*xb),np.full(n,xa),np.full(n,xb),s,np.zeros(2),jac)
    result=[];step=1e-6
    for slope in (0.,1e-3):
        _,ja,jb,js=kernel(va,vb,slope,True)
        for endpoint in ('source_logf','target_logf','shared_k1_slope'):
            aa,bb=va.copy(),vb.copy();am,bm=va.copy(),vb.copy();sp,sm=slope,slope
            if endpoint=='source_logf':aa[3]+=step;am[3]-=step;analytic=ja[:,3]
            elif endpoint=='target_logf':bb[3]+=step;bm[3]-=step;analytic=jb[:,3]
            else:sp+=step;sm-=step;analytic=js[:,2]
            fd=(kernel(aa,bb,sp,False)-kernel(am,bm,sm,False))/(2*step)
            relative=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-30))
            if relative>2e-5:raise ValueError('Actual-source two-endpoint derivative failed')
            result.append(dict(math_probe_slope=slope,column=endpoint,max_native_error=float(np.max(np.abs(fd-analytic))),relative_error=relative))
    # Actual absolute source row checks the PhysicalCameraModel chain separately.
    row=next((r for r in g.observed[target]if r['id'].startswith('ad2_right_external')),g.observed[target][0])
    from dataclasses import replace
    c=replace(g.calibration,k1=(g.calibration.k1[0],1e-3))
    v=g.views[target].view;cam,sj,vj=c.camera25_and_jacobians(v);_,raw=absolute_value_jac15(cam,[row],True)
    scaling=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    analytic=raw@(scaling*vj[:15,3])
    vp=ViewState(v.rvec_world_to_camera,v.log_f_ratio+step);vm=ViewState(v.rvec_world_to_camera,v.log_f_ratio-step)
    fd=(absolute_value_jac15(c.camera25(vp),[row],False)-absolute_value_jac15(c.camera25(vm),[row],False))/(2*step)
    err=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-30))
    if err>2e-5:raise ValueError('Actual absolute zoom-f derivative failed')
    return dict(pair_index=ip,pair_pts=[pa,pb],original_local_rows=list(range(len(a))),source_uv=a,target_uv=b,
        source_uv_sha256=hashlib.sha256(a.tobytes()).hexdigest(),target_uv_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        ray_checks=result,absolute=dict(pts=target,id=row['id'],source_uv_float64_sha256=uv_sha(row),
        relative_error=err,max_sigma_error=float(np.max(np.abs(fd-analytic)))),math_probe_not_optimized_or_exported=True)

def eliminate(matrix,weights):
    """Project slope column outside all R/f columns, without ridge/damping."""
    weighted=matrix.multiply(weights[:,None]).tocsr();s=weighted[:,0].toarray().ravel();a=weighted[:,1:].tocsr()
    norms=np.sqrt(np.asarray(a.power(2).sum(axis=0)).ravel())
    if np.any(norms==0):raise ValueError('Unobserved nuisance column must be diagnosed')
    a=(a@diags(1./norms)).tocsr();h=(a.T@a).tocsc();rhs=np.asarray(a.T@s).ravel()
    method='undamped_column_scaled_sparse_normal_Schur'
    try:
        lu=splu(h);z=lu.solve(rhs)
        for _ in range(3):
            e=s-a@z;normal=np.asarray(a.T@e).ravel()
            if np.linalg.norm(normal,np.inf)<=1e-9*max(np.linalg.norm(rhs,np.inf),1e-30):break
            z+=lu.solve(normal)
        del lu
    except RuntimeError:
        method='column_scaled_LSMR_projection_no_damping';r=lsmr(a,s,atol=1e-11,btol=1e-11,conlim=1e15);z=r[0]
    e=s-a@z;normal=np.asarray(a.T@e).ravel()
    normal_relative=float(np.linalg.norm(normal,np.inf)/max(np.linalg.norm(rhs,np.inf),1e-30))
    info=float(e@e);raw=float(s@s)
    verified=normal_relative<1e-7 and np.isfinite(info)
    result=dict(method=method,nuisance_columns=a.shape[1],rows=a.shape[0],projection_stationarity_relative=normal_relative,
        numerical_projection_verified=verified,conditional_fixed_RF_information=raw,
        nuisance_eliminated_slope_information=info if verified else None,
        information_fraction_after_RF=info/raw if verified and raw>0 else None,
        local_sigma_k1_slope=1./np.sqrt(info) if verified and info>0 else None,
        sigma_semantics='local weighted linear source objective with declared sigmas, not hardware calibration confidence',
        damping_or_added_gauge_prior=0.)
    return result,np.asarray(z/norms),np.asarray(e)

def run(args):
    args.output.mkdir(parents=True,exist_ok=args.resume);cv2.setNumThreads(1);enable_projection_backend()
    g=prepare(args)
    manifest=dict(schema='rinkspace.main-z190-full-source-k1-slope-information-input.v1',current_physical=ref(args.current),
        current_binding=ref(g.binding_path),absolute_authority=ref(args.authority),admission=g.packet['source_authority_admission'],
        new_corner_held=ref(args.held),scene=ref(g.scene),source_sha256=SOURCE_SHA,source_time_base='1/15360',
        installation=IDENTIFIER,frozen_Cxyz=g.calibration.pivot_xyz,frozen_height_m=g.calibration.pivot_xyz[2]*SCENE_UNIT_M,
        frozen_PP_coefficients=[g.calibration.principal_x_fraction,g.calibration.principal_y_fraction],frozen_k1_intercept=g.calibration.k1[0],
        q_reference=g.calibration.q_reference,sole_initial='current_v6_physical_main_states_numeric_exact_v5',
        hypothesis='one shared k1_degree_1; k1=k0+s*log(f_effective/f_reference); all R/f nuisance',
        current_full_source_role='authority fit only; pending68 and held including41.300 absent from J',
        motion_assumption='unchanged original infinite source rays at fixed C; no finite material assignment',
        motion_sigma='original Bundle max(2,sqrt(pair_point_count)/3); native UV sigma once',
        scalar_Huber_threshold_sigma=2.,roll_prior='original per-exposure physical roll sigma0.5degree',
        constant_optics_prior='five original constant-intercept rows retained as constants; zero active derivatives, no information',
        fit_rows=len(g.packet['fit_rows']),pending_rows=68,held_rows=len(g.packet['held_rows']),new_held_points=2,
        native_main_frames=len(g.native),actual_main_exposures=len(g.views),classification=g.classification,rays=g.raybindings,
        rows_selected_by_residual=False,source_rows_removed_by_residual=False,nonlinear_solver_called=False,camera_exported=False)
    if args.resume:
        if read(args.output/'input_manifest.json')!=jsonable(manifest):raise ValueError('Resume inputs differ')
    else:write(args.output/'input_manifest.json',manifest)
    results=[];begin=time.perf_counter()
    for cut in (1,4,7,9):
        folder=args.output/f'cut{cut}'
        if args.resume and (folder/'information.json').exists():
            record=read(folder/'information.json');saved=np.load(folder/'residual_state_rows.npz',allow_pickle=False)
            saved_rf=saved['current_unchanged_rf'];stamps=saved['canonical_pts']
            expected=np.array([np.r_[g.views[int(p)].view.rvec_world_to_camera,g.views[int(p)].view.log_f_ratio+np.log(1920.)]for p in stamps])
            if not np.array_equal(saved_rf,expected) or set(stamps)!=set(p for p in g.views if g.cuts[p]==cut):
                raise ValueError('Resume saved R/f differs from current physical')
            if sha(record['derivative_checks']['path'])!=record['derivative_checks']['sha256']:raise ValueError('Saved derivative trace changed')
            results.append(record);print(json.dumps(dict(stage='reuse_verified_completed_information',cut=cut)),flush=True)
            continue
        print(json.dumps(dict(stage='build_source_J',cut=cut)),flush=True)
        folder.mkdir(exist_ok=args.resume);base,bundle,q0,q=make_bundle(g,cut)
        r0=base.evaluate(q0,False,include_priors=False);r,j=bundle.evaluate(q,True,include_priors=False)
        maxdiff=float(np.max(np.abs(r-r0)))
        if maxdiff>1e-6:raise ValueError('Zero slope source objective not restored')
        checks=derivative_checks(g,cut)
        retained=j[:,np.r_[5,np.arange(6,len(q))]].tocsr();retained.eliminate_zeros();del j
        prior,pj=bundle.priors(q);roll=prior[5:];roll_j=pj[5:,np.r_[5,np.arange(6,len(q))]].tocsr()
        if len(roll)!=len(bundle.stamps)or roll_j[:,0].nnz:raise ValueError('Roll prior layout or optical derivative changed')
        if pj[:5].nnz and np.any(pj[:5].data):raise ValueError('Frozen optics priors must have zero derivative')
        weights=np.sqrt(np.minimum(1.,2./np.maximum(np.abs(r),1e-12)))
        save_npz(folder/'source_jacobian_slope_rf.npz',retained,compressed=True);save_npz(folder/'roll_jacobian_slope_rf.npz',roll_j,compressed=True)
        np.savez_compressed(folder/'residual_state_rows.npz',normalized_source_residual=r,native_residual_sigma=base.native_residual_sigma,
            scalar_Huber_sqrt_weights=weights,normalized_roll_residual=roll,canonical_pts=np.asarray(bundle.stamps,np.int64),
            current_unchanged_rf=q[6:].reshape(-1,4),zero_slope_state6rf=q)
        write(folder/'derivative_checks.json',checks)
        row_trace=[];off=bundle.motion_rows
        for p,rows in bundle.abs_groups:
            for row in rows:
                n=2*len(row['uv']);row_trace.append(dict(start=off,end=off+n,pts=p,id=row['id'],source_native_pts=row.get('source_native_pts',row['pts']),
                    source_uv_float64_sha256=uv_sha(row),split='fit'));off+=n
        write(folder/'row_layout.json',dict(motion_scalar_rows=bundle.motion_rows,absolute=row_trace,
            columns=['shared_k1_slope']+[f'{p}:{axis}'for p in bundle.stamps for axis in ('R0','R1','R2','log_effective_f_px')]))
        print(json.dumps(dict(stage='linear_RF_nuisance_elimination',cut=cut,rows=len(r),RF_columns=4*len(bundle.stamps))),flush=True)
        data,dy,de=eliminate(retained,weights)
        total,ty,te=eliminate(vstack((retained,roll_j),format='csr'),np.r_[weights,np.ones(len(roll))])
        np.savez_compressed(folder/'nuisance_projection_only.npz',data_unit_slope_RF_response=dy,data_projected_slope_column=de,
            data_plus_roll_unit_slope_RF_response=ty,data_plus_roll_projected_slope_column=te)
        record=dict(cut=cut,actual_exposures=len(bundle.stamps),motion_points=len(bundle.a),fit_absolute_rows=len(row_trace),
            source_scalar_rows=len(r),slope_zero_camera25_bitexact=True,slope_zero_original_objective_max_sigma=maxdiff,
            derivative_checks=ref(folder/'derivative_checks.json'),data_only=data,data_plus_original_roll=total,
            no_source_selection_or_camera_state_update=True)
        write(folder/'information.json',record);results.append(record);print(json.dumps(jsonable(record)),flush=True)
        del base,bundle,q0,q,r0,r,retained,roll_j,weights,dy,de,ty,te,pj;gc.collect()
    summary=dict(schema='rinkspace.main-z190-full-source-k1-slope-nuisance-information.v1',input_manifest=ref(args.output/'input_manifest.json'),
        cuts=results,seconds=time.perf_counter()-begin,nonlinear_solver_called=False,camera_exported=False,
        shared_slope_across_cuts=True,independent_RF_nuisance_blocks=True,held_and_pending_absent_from_information=True,
        accuracy_accepted=False,source_or_geometry_independence_not_proven_by_information=True)
    for key in ('data_only','data_plus_original_roll'):
        verified=all(r[key]['numerical_projection_verified']for r in results)
        info=sum(r[key]['nuisance_eliminated_slope_information']for r in results)if verified else None
        summary[key]=dict(numerical_projection_verified=verified,slope_information=info,local_sigma_k1_slope=1/np.sqrt(info)if info else None,
            slope_information_by_cut={r['cut']:r[key]['nuisance_eliminated_slope_information']for r in results})
    write(args.output/'report.json',summary)
    print(json.dumps(jsonable({k:v for k,v in summary.items()if k!='cuts'})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=CURRENT_PHYSICAL)
    p.add_argument('--authority',type=Path,default=AUTHORITY);p.add_argument('--held',type=Path,default=HELD)
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_zoom_k1_information_v1')
    p.add_argument('--resume',action='store_true',help='Reuse completed information packs only after unchanged-input/current-Rf checks')
    run(p.parse_args())
