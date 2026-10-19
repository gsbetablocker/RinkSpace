"""Attribute current Z190 source R/f corrections to actual factors and priors.

Load the latest physical source package directly. No older camera seed, source
filter, held fitting, geometry mutation or shared-centre update is involved.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from continuous_central_brown4_optin import enable_projection_backend
from z190_physical_camera import PhysicalCameraModel
from z190_physical_camera import CanonicalView, ViewState
from z190_continuous_source_bundle import StandingSourceBundle
from main7_source_bundle_candidate import huber_cost
from z190_source_camera_diagnostic import jsonable, source_errors, solve_with_crew_priors
from render_result import SceneRenderer, decode_selected
from ice_refine import _curve_vector_error

SOURCE = Path('D:/OneDrive/codex_projs/RinkSpace/data/input/oxXu1avhrvs.mp4')
IDENTIFIER = 'okada_sp_installation_1_conditional_z190'
FOCUS = (229497600, 230415104)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_graph(physical, rows_path, rays_path, *, fixed_height=True):
    model, native, aliases, views = PhysicalCameraModel.load_npz(physical)
    calibration = model.calibrations[IDENTIFIER]
    stamps = sorted(views)
    index = {p:i for i,p in enumerate(stamps)}
    packet = json.loads(rows_path.read_text(encoding='utf8'))
    groups = {}
    for raw in packet['fit_rows']:
        if raw['split'] != 'fit':
            raise ValueError('Original development role changed')
        row = dict(raw)
        for k in ('world','uv','sigma','normal','normal_interval'):
            if k in row: row[k] = np.asarray(row[k], float)
        if hashlib.sha256(row['uv'].tobytes()).hexdigest() != row['source_uv_float64_sha256']:
            raise ValueError('Original source UV changed')
        groups.setdefault(aliases[int(row['pts'])], []).append(row)
    abs_groups = [(p, groups[p]) for p in stamps if p in groups]
    rays = dict(np.load(rays_path, allow_pickle=False))
    pairs, offset = rays['pair_pts'], rays['offset']
    counts = np.diff(offset)
    if offset[0] != 0 or offset[-1] != len(rays['source_uv']):
        raise ValueError('Original motion offsets changed')
    ia = np.repeat([index[int(p)] for p in pairs[:,0]], counts)
    ib = np.repeat([index[int(p)] for p in pairs[:,1]], counts)
    if any(aliases[int(p)] != int(p) for p in pairs.ravel()):
        raise ValueError('Source motion endpoints must be actual canonical exposures')
    sigma = np.full(2*len(ia), 2.)
    absolute_sigma = np.concatenate([np.repeat(r['sigma'],2) for _, group in abs_groups for r in group])
    base = SimpleNamespace(stamps=stamps, index=index, a=rays['source_uv'], b=rays['target_uv'],
        ia=ia, ib=ib, motion_sigma=sigma, abs_groups=abs_groups,
        native_residual_sigma=np.r_[sigma, absolute_sigma])
    # This graph contains exactly the original source packets, including all
    # measured ad corners and the original held rows kept outside fitting.
    bundle = StandingSourceBundle(base, calibration, prior_wide_focal_px=min(
        np.exp(calibration.camera25(views[p].view)[6]) for p in stamps), fixed_height=fixed_height)
    if fixed_height and not np.isclose(calibration.pivot_xyz[2],bundle.crew_hypothesis.height_target_scene,
                                     rtol=0.,atol=1e-12):
        raise ValueError('Completed soft-height package requires fixed_height=False; saved Cz must not be reset')
    # A completed soft-height physical package is the exact warm state. The
    # working-height target remains a prior; never replace its saved Cz here.
    shared=np.r_[np.asarray(calibration.pivot_xyz)/100.,calibration.principal_x_fraction[0],
                 calibration.principal_y_fraction[0],calibration.k1[0]]
    q = np.r_[shared, np.concatenate([
        np.r_[views[p].view.rvec_world_to_camera, views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
    return bundle, q, packet, rays, model, native, aliases, views


def roll(q4):
    r = cv2.Rodrigues(np.asarray(q4[:3]))[0]
    return float(np.rad2deg(np.arctan2(r[0,2], -r[1,2])))


def component_rows(bundle):
    n = bundle.source_residual_rows
    return dict(motion=slice(0,bundle.motion_rows), absolute=slice(bundle.motion_rows,n),
                optics_prior=slice(n,n+5), roll_prior=slice(n+5,None))


def summarize(bundle, q, residual=None):
    r = bundle.evaluate(q, False) if residual is None else residual
    components = {}
    for name, sl in component_rows(bundle).items():
        values = r[sl]
        components[name] = dict(rows=len(values), cost=huber_cost(values,True) if name in ('motion','absolute')
            else .5*float(values@values), maximum_abs_sigma=float(np.max(np.abs(values))))
    return components


def render_focus(args, packet, model, aliases, views):
    scene = json.loads(args.scene.read_text(encoding='utf8'))
    renderer = SceneRenderer(scene, draw_grid=True, draw_advertisements=True)
    records = []
    for p, frame in decode_selected(SOURCE, list(FOCUS)):
        camera = model.camera25(views[aliases[p]])
        image, metrics = renderer.draw(frame, camera)
        candidates=[(role,r) for role in ('fit_rows','pending_rows') for r in packet.get(role,[])
                    if int(r['pts']) == p and r['id'] == 'C']
        if len(candidates)!=1:
            raise ValueError('C source review requires one original fit or pending record')
        role,row=candidates[0]
        identity_pending=role=='pending_rows'
        uv, world = np.asarray(row['uv']), np.asarray(row['world'])
        predicted = uv if identity_pending else uv+_curve_vector_error(camera,world,uv).reshape(-1,2)
        for i,(a,b) in enumerate(zip(uv,predicted)):
            a,b = tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int))
            cv2.circle(image,a,6,(0,230,255),2,cv2.LINE_AA)
            if not identity_pending:
                cv2.circle(image,b,4,(20,20,255),2,cv2.LINE_AA)
                cv2.arrowedLine(image,b,a,(0,230,255),1,cv2.LINE_AA,tipLength=.15)
        if identity_pending:
            cv2.putText(image,'Source red-band identity pending; these UVs were not fitted',
                        (24,52),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,230,255),2,cv2.LINE_AA)
        path = args.output/f'{p}_local_rf_all_model.jpg'
        cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,97])
        crop = args.output/f'{p}_local_rf_C_crop.jpg'
        cv2.imwrite(str(crop),image[820:1060],[cv2.IMWRITE_JPEG_QUALITY,97])
        records.append(dict(pts=p, native_frame_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            overlay=str(path), crop=str(crop), measured_UV_unchanged=True,
            source_identity_pending=identity_pending,source_record_role=role,
            native_source_errors=None if identity_pending else source_errors(camera,[{**row,'uv':uv,'world':world,'sigma':np.asarray(row['sigma'])}]),
            all_inventory_metrics=metrics))
    return records


class FixedSharedRF:
    """Keep every current calibration slot fixed; optionally audit roll prior.

    The sparse pattern and original relative/absolute source rows are retained.
    Zero shared derivative columns are intentional frozen variables. Removing
    working roll priors is a labeled diagnostic, never a source-point filter.
    """
    def __init__(self, bundle, shared, roll_priors):
        self.base,self.shared,self.roll_priors=bundle,np.asarray(shared).copy(),roll_priors
        for name in ('source_residual_rows','native_residual_sigma','view_offset',
                     'crew_fixed_height','crew_hypothesis'):
            setattr(self,name,getattr(bundle,name))

    def evaluate(self,q,jacobian=True):
        if not np.array_equal(q[:6],self.shared):
            raise ValueError('Current shared calibration is frozen')
        value=self.base.evaluate(q,jacobian)
        if jacobian:
            r,j=value
            j.data[j.indices<6]=0.
            return (r,j) if self.roll_priors else (r[:self.source_residual_rows+5],j[:self.source_residual_rows+5])
        return value if self.roll_priors else value[:self.source_residual_rows+5]


def connect_rf(args,bundle,q,packet,native,aliases,roll_priors):
    folder=args.output/('connected_original_priors' if roll_priors else 'connected_roll_prior_audit')
    folder.mkdir(parents=True,exist_ok=True)
    frozen=FixedSharedRF(bundle,q[:6],roll_priors)
    solved,history,seconds=solve_with_crew_priors(frozen,q,folder)
    if not np.array_equal(solved[:6],q[:6]):
        raise ValueError('Shared calibration moved in R/f-only recovery')
    calibration=bundle.current_calibration(solved)
    model=PhysicalCameraModel({IDENTIFIER:calibration})
    states={p:CanonicalView(IDENTIFIER,ViewState(tuple(solved[6+4*i:9+4*i]),
        float(solved[9+4*i]-np.log(1920.))))for i,p in enumerate(bundle.stamps)}
    model.save_npz(folder/'physical_dense_rf_diagnostic.npz',native,aliases,states,
        source_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',time_base='1/15360')
    np.savez_compressed(folder/'diagnostic_state6rf.npz',q=solved)
    groups=[]
    for split in ('fit_rows','held_rows'):
        for raw in packet[split]:
            row=dict(raw)
            for k in ('world','uv','sigma','normal','normal_interval'):
                if k in row:row[k]=np.asarray(row[k],float)
            p=int(row['pts']);camera=model.camera25(states[aliases[p]])
            groups.append(dict(pts=p,id=row['id'],split=row['split'],fit_role=split,
                errors=source_errors(camera,[row])[0]))
    details=dict(schema='rinkspace.current-z190-connected-fixed-calibration-rf.v1',
        accuracy_accepted=False,shared_Cxyz_PP_k1_unchanged=True,source_UV_and_split_and_sigmas_unchanged=True,
        original_roll_prior_retained=roll_priors,roll_prior_change_status='original' if roll_priors else 'diagnostic factor audit only',
        source_exposures=len(bundle.stamps),native_frames=len(native),
        original_motion_points=len(bundle.a),source_cost_components=summarize(bundle,solved),
        seconds=seconds,source_errors=groups,history=history,
        source_C_reviews=render_focus(SimpleNamespace(scene=args.scene,output=folder),packet,model,aliases,states))
    (folder/'report.json').write_text(json.dumps(jsonable(details),indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage=folder.name,seconds=seconds,
        cost_components=details['source_cost_components'],source_C_errors=[r for r in groups if r['id']=='C']))),flush=True)
    return details


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    enable_projection_backend()
    bundle,q,packet,rays,_,native,aliases,views = load_source_graph(args.physical,args.rows,args.rays)
    local_model, _, local_aliases, local_views = PhysicalCameraModel.load_npz(args.local)
    if not np.array_equal(bundle.calibration.shared_state(),local_model.calibrations[IDENTIFIER].shared_state()):
        raise ValueError('This attribution requires unchanged current shared calibration')
    r,j = bundle.evaluate(q)
    slices = component_rows(bundle)
    weight = np.ones(len(r)); weight[:bundle.source_residual_rows] = np.minimum(
        1.,2./np.maximum(np.abs(r[:bundle.source_residual_rows]),1e-12))
    qlocal = q.copy()
    for p, item in local_views.items():
        start = 6+4*bundle.index[aliases[p]]
        qlocal[start:start+4] = np.r_[item.view.rvec_world_to_camera,item.view.log_f_ratio+np.log(1920.)]
    keys = []
    for p in local_views:
        key = aliases[p]; start = 6+4*bundle.index[key]
        d = np.zeros_like(q); d[start:start+4] = qlocal[start:start+4]-q[start:start+4]
        actual = q+d
        key_record = dict(pts=p, canonical_pts=key, delta_rvec_logf=d[start:start+4],
            current_roll_deg=roll(q[start:start+4]), local_roll_deg=roll(actual[start:start+4]),
            current_effective_f_px=float(np.exp(q[start+3])),local_effective_f_px=float(np.exp(actual[start+3])),
            direction_components={})
        projected = j@d
        for name,sl in slices.items():
            key_record['direction_components'][name] = dict(
                gradient_dot_local_step=float(np.dot((weight*r)[sl],projected[sl])),
                quadratic_step_norm_squared=float(np.dot((weight*projected)[sl],projected[sl])))
        adjacent = np.flatnonzero(np.any(rays['pair_pts']==key,axis=1))
        edge_records=[]
        newr=bundle.evaluate(actual,False)
        for e in adjacent:
            lo,hi=map(int,rays['offset'][e:e+2]); a,b=r[2*lo:2*hi],newr[2*lo:2*hi]
            edge_records.append(dict(edge_index=int(e),pts=rays['pair_pts'][e].tolist(),points=hi-lo,
                cost_before=huber_cost(a,True),cost_local_jump=huber_cost(b,True),
                native_median_before_px=float(np.median(np.linalg.norm(a.reshape(-1,2)*2.,axis=1))),
                native_median_local_jump_px=float(np.median(np.linalg.norm(b.reshape(-1,2)*2.,axis=1)))))
        key_record['incident_original_motion_edges']=edge_records
        key_record['actual_component_costs_current']=summarize(bundle,q,r)
        key_record['actual_component_costs_local_key']=summarize(bundle,actual,newr)
        keys.append(key_record)
    report=dict(schema='rinkspace.current-z190-original-source-objective-conflict.v1',
        shared_Cxyz_PP_k1_fixed=True,source_UV_roles_sigmas_unchanged=True,held_outside_fitting=True,
        original_motion_pairs=len(rays['pair_pts']),original_motion_points=len(bundle.a),
        canonical_exposures=len(bundle.stamps),source_absolute_development_keys=len(local_views),
        current_cost_components=summarize(bundle,q,r),
        simultaneous_local_keys_cost_components=summarize(bundle,qlocal),
        note='Local key jumps measure conflict, not a connected solution; neighbor R/f can move in a full graph.',
        bindings=[dict(path=str(p),sha256=sha(p))for p in (args.physical,args.local,args.rows,args.rays,args.scene)],
        key_direction_components=keys,source_C_reviews=render_focus(args,packet,local_model,local_aliases,local_views))
    (args.output/'report.json').write_text(json.dumps(jsonable(report),indent=2),encoding='utf8')
    print(json.dumps(jsonable({k:report[k]for k in ('current_cost_components','simultaneous_local_keys_cost_components')})),flush=True)
    print(json.dumps(jsonable([r for r in keys if r['pts'] in FOCUS]),indent=2),flush=True)
    if args.connect_rf:
        # First recover all exposures under the exact original graph/prior.
        # Starting at the current full source solution avoids injecting sparse
        # underconstrained key jumps into unrelated continuous source states.
        connect_rf(args,bundle,q,packet,native,aliases,True)
        connect_rf(args,bundle,q,packet,native,aliases,False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('physical','local','rows','rays','scene','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--connect-rf',action='store_true')
    run(p.parse_args())
