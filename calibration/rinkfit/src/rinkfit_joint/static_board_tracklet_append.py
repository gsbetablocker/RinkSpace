"""Source-qualified board textures as shared unknown bounded ad-surface points.

One track owns two latent material coordinates, never one world point per frame.
The existing native kind-2 single-vertex affine operator supplies board geometry.
"""
from collections import defaultdict, Counter
from pathlib import Path
import copy
import json
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256

OPERATOR = 'native_static_board_RGB_texture_2D_track_v1'
BUNDLE_SCHEMA = 'rinkfit.native_static_board_tracklet_qualified_bundle.v1'
GATE_SCHEMA = 'rinkfit.static_board_tracklet_point_qualification.v1'
RECEIPT_SCHEMA = 'rinkfit.static_board_tracklet_point_receipt.v1'


def _ref(path):
    p = Path(path).resolve()
    return dict(file=str(p), sha256=file_sha256(p))


def _verified(ref):
    p = Path(ref.get('file', ref.get('path'))).resolve()
    if file_sha256(p) != ref['sha256']:
        raise ValueError('Changed exact board texture source dependency: '+str(p))
    return p


def _source_contract(generation_path, qualification_path, roles_path):
    generation = read_json(generation_path); q = read_json(qualification_path)
    if (generation.get('schema') != 'rinkfit.native_board_print_unknown_surface_texture_candidate_generation.v1'
        or q.get('schema') != 'rinkfit.native_board_print_static_texture_source_qualification.v1'
        or _verified(q['candidate_generation']) != Path(generation_path).resolve()
        or q.get('observations_from_model_or_ice_H') is not False
        or q.get('UV_moved_or_interpolated') is not False):
        raise ValueError('Exact native printed board source qualification required')
    rows_path = _verified(generation['candidate_rows'])
    if _verified(q['candidate_rows']) != rows_path:
        raise ValueError('Qualification refers to different source rows')
    for ref in generation['reader_modules']+q['reviewed_images']:
        _verified(ref)
    replay = read_json(_verified(q['numeric_replay']))
    rgb = read_json(_verified(q['current_RGB_receipt']))
    if (not replay['all_current_source_hashes_and_exclusion_masks_valid']
        or not replay['all_emitted_rows_current_local_two_direction_supported']
        or replay['source_UV_modified'] or not rgb['all_candidate_224_UV_and_RGB_hashes_matched']):
        raise ValueError('Current RGB/visibility/local 2D replay is incomplete')
    for key in ('LK_current_UV_max_abs_diff_px','FB_max_abs_difference_px',
                'NCC_max_abs_difference','tensor_second_direction_ratio_max_abs_diff'):
        if replay[key] != 0.:
            raise ValueError('Actual current source numerical evidence changed')
    source = generation['source_video']; scope = read_json(_verified(generation['scope']))
    if (source['id'] != 'O6j8L8TmfKo' or source['native_fps'] != '50/1'
        or source['time_base'] != '1/12800' or source['native_pts_step'] != 256
        or source['sha256'] != scope['source']['sha256']):
        raise ValueError('Official native source clock differs')
    if _verified(generation['roles']) != Path(roles_path).resolve():
        raise ValueError('Board source role generation differs')
    roles = read_json(roles_path)['frame_roles']; shots = {s['id']:s for s in scope['shots']}
    original = list(read_jsonl(rows_path)); diagnostics = {(r['track_id'], r['native_pts']):r for r in replay['rows_evidence']}
    rgbrows = {(r['track_id'], r['native_pts']):r for r in rgb['rows']}
    accepted = {t['track_id']:t for t in q['per_track'] if t['qualified_for_unknown_shared_board_surface_point_initialization']}
    exact = {r['candidate_line_1based']:r for r in q['qualified_source_line_index']}
    if len(exact) != q['qualified_source_rows']:
        raise ValueError('Qualified board source row index repeated')
    groups = defaultdict(list); used = set()
    for line, index in sorted(exact.items()):
        r = original[line-1]; k = (r['track_id'], r['native_pts']); s = shots[r['shot_id']]
        import hashlib
        token = hashlib.sha256(json.dumps(r['source_uv'], separators=(',',':')).encode('ascii')).hexdigest()
        d = diagnostics.get(k); e = rgbrows.get(k)
        if (r['track_id'] not in accepted or k in used or index['track_id'] != r['track_id']
            or index['native_pts'] != r['native_pts'] or token != index['candidate_source_uv_sha256']
            or r['source_uv'] != r['current_uv'] or r['pixels_from_model'] is not False
            or r['ice_H_used'] is not False or r['known_world_corner_correspondence'] is not False
            or r['surface_class'] != 'board' or r['surface_instance_id'] != accepted[r['track_id']]['board_instance_id']
            or r['source_video_sha256'] != source['sha256'] or r['time_base'] != [1,12800]
            or r['shot_id'] == 's01' or not s['first_pts'] <= r['native_pts'] <= s['last_pts']
            or (r['native_pts']-s['first_pts']) % 256 or r['fit_role'] != 'development'
            or roles.get(str(r['native_pts']),{}).get('role') != 'development_candidate'
            or d is None or e is None or d['source_uv'] != r['source_uv'] or e['source_uv'] != r['source_uv']
            or not d['same_RGB'] or not d['current_exclusion_mask_clear_for_LK_support']
            or d['two_direction_recomputed']['status'] != 'entity_corner_two_direction_support'):
            raise ValueError('Printed board source identity, original UV, material, or current exposure differs')
        used.add(k); groups[r['track_id']].append(r)
    if len(used) != q['qualified_source_rows'] or len(groups) != q['qualified_tracks']:
        raise ValueError('Exact reviewed board texture denominator differs')
    for k, rows in groups.items():
        rows.sort(key=lambda r:r['native_pts'])
        if (len(rows) < 2 or len({r['shot_id'] for r in rows}) != 1
            or np.any(np.diff([r['native_pts'] for r in rows]) != 256)
            or len(rows) != accepted[k]['source_rows']):
            raise ValueError('Printed static board track crosses a cut, source gap, or unmeasured extension')
    return groups


def _ad_block(model, instance):
    ids = {p['advertisement_block'] for m in model['materials'] if m['label'].startswith(instance+'.')
           for p in m['paths'] if p.get('advertisement_block',-1) >= 0}
    if len(ids) != 1:
        raise ValueError('Source-confirmed ad must have one existing shared installation')
    return ids.pop()


def _ray_surface_seed(model, scene, camera, local, uv, ad_block):
    """Intersect the true observed-row ray and finite continuous board; no ice H."""
    from scipy.optimize import brentq, minimize_scalar
    from .wide_optics_candidate import _camera_geometry, _inverse_brown
    uv = np.asarray(uv,float); t = local+camera.tau*(uv[1]/1079.-.5)
    R,C,lf,z = _camera_geometry(camera,np.array([t])); R,C,lf,z = R[0],C[0],lf[0],z[0]
    pp = np.asarray(camera.optics['principal_point'])+z*np.asarray(camera.optics['principal_linear'])
    fxy = np.exp(lf+np.array([camera.optics['eta']/2,-camera.optics['eta']/2]))
    coefficients = np.array([[np.polynomial.polynomial.polyval(z,camera.optics['k1']),
        np.polynomial.polynomial.polyval(z,camera.optics['k2']),camera.optics['p1'],camera.optics['p2']]])
    coefficients += max(camera.wide_boundary-lf,0.)**2*camera.wide_coefficients
    ray = R.T@np.r_[_inverse_brown(((uv-pp)/fxy)[None],coefficients)[0],1.]
    low,high = 0.,np.inf
    for axis,bound in ((0,.5),(1,scene.width/2)):
        if abs(ray[axis]) < np.finfo(float).eps:
            if abs(C[axis]) > bound:raise ValueError('Board ray misses finite bounding box')
            continue
        a,b = sorted(((-bound-C[axis])/ray[axis],(bound-C[axis])/ray[axis]))
        low,high = max(low,a),min(high,b)
    if not 0. <= low < high < np.inf:
        raise ValueError('Board ray has no positive finite surface segment')
    membership = lambda a:float(scene.containment((C+a*ray)[None,:2])[0])
    mid = minimize_scalar(membership,bounds=(low,high),method='bounded',options={'xatol':1e-14}).x
    if membership(mid) > 0:raise ValueError('Source ray misses convex finite board')
    lengths=np.array([scene.long,scene.arc,scene.short,scene.arc,scene.long,scene.arc,scene.short,scene.arc]); edges=np.r_[0,np.cumsum(lengths)]
    ad = np.asarray(model['blocks'][ad_block]['values'],float); candidates=[]
    for a,b in ((low,mid),(mid,high)):
        fa,fb=membership(a),membership(b)
        if fa < -1e-13 and fb < -1e-13:continue # camera can be inside the ice XY outline
        lam = a if abs(fa)<1e-13 else b if abs(fb)<1e-13 else brentq(membership,a,b,xtol=1e-14)
        X=C+lam*ray
        fits=[minimize_scalar(lambda s:float(np.sum((scene.board(np.array([s]),np.array([X[2]]))[0,:2]-X[:2])**2)),
            bounds=(edges[i],edges[i+1]),method='bounded',options={'xatol':1e-14})for i in range(8)]
        fit=min(fits,key=lambda f:f.fun); s=fit.x
        u=max((abs(X[0])-(.5-scene.rx))/scene.rx,0.);v=max((abs(X[1])-(scene.width/2-scene.ry))/scene.ry,0.)
        normal=-np.array([np.sign(X[0])*u**(scene.p-1)/scene.rx,np.sign(X[1])*v**(scene.p-1)/scene.ry])
        front=bool(normal@(C[:2]-X[:2])>=0.); coordinate=np.array([(s-ad[0])/ad[1],(X[2]-ad[2])/(ad[3]-ad[2])])
        allowed=bool(front and 0<=X[2]<=scene.height and np.all(coordinate>=0) and np.all(coordinate<=1))
        candidates.append(dict(ray_distance=float(lam),world_L1=X.tolist(),surface_s_h=[float(s),float(X[2])],
            unknown_ad_uv_initialization=coordinate.tolist(),inward_face=front,within_finite_board_height=bool(0<=X[2]<=scene.height),
            within_source_named_numeric_installation=bool(np.all(coordinate>=0) and np.all(coordinate<=1)),
            admitted_as_initialization=allowed,inverse_surface_world_numerical_error_L=float(np.sqrt(fit.fun))))
    legal=[r for r in candidates if r['admitted_as_initialization']]
    if len(legal)!=1:raise ValueError('Ad instance/finite board branch has no unique legal ray initializer')
    return np.asarray(legal[0]['unknown_ad_uv_initialization']),candidates


def validate_point_receipt(row,receipt,material,material_index):
    if (row.get('measurement_operator_resolved') != OPERATOR or not row.get('point_correspondence')
        or receipt.get('schema') != RECEIPT_SCHEMA or receipt['existing_material_index'] != material_index
        or receipt['target_id'] != material['label'] or receipt['native_pts'] != row['native_pts']
        or row.get('surface_class') != 'board' or row.get('surface_instance_id') != receipt['source_board_instance_id']
        or row.get('physical_track_id') != receipt['track_id'] or row.get('world_position_is_shared_track_latent') is not True
        or any(row.get(k) is not False for k in ('known_world_corner_correspondence','ice_H_used','existing_world_corner_or_XY_used','pixels_from_model'))
        or receipt['original_UV_preserved'] != row['source_uv'] or row['source_uv'] != row['current_uv']):
        raise ValueError('Shared board texture source redirects its mean or physical surface')
    expected=object_sha256(dict(source_row_id=row['source_row_id'],native_pts=row['native_pts'],current_2D_UV=row['current_uv'],estimator=OPERATOR))
    if receipt['current_2D_source_mean_sha256'] != expected:
        raise ValueError('Board texture source mean changed')
    found=[i for i,p in enumerate(material['paths'])if p['id']==receipt['existing_point_path']]
    if len(found)!=1:raise ValueError('Shared board track must have one unique physical point')
    path=material['paths'][found[0]]
    if (path['kind']!=2 or path.get('points')!=[[0.,0.]] or path.get('physical_point')is not True
        or path.get('closed',False) or path.get('advertisement_block')!=receipt['advertisement_block']
        or path.get('geometry_block')!=receipt['geometry_block'] or path.get('geometry_indices')
        or path.get('geometry_offset',0)!=0):
        raise ValueError('Board texture requires the native bounded unknown ad-surface point operator')
    return found[0]


def load_point_qualification(manifest_path,manifest,reference,verify):
    for name in ('source_generation','static_surface_qualification','numeric_source_replay','current_RGB_receipt',
                 'roles','input_cumulative_journal','qualification_receipts','current_model','track_material_bindings','gate_code'):
        verify(manifest[name])
    groups=_source_contract(verify(manifest['source_generation']),verify(manifest['static_surface_qualification']),verify(manifest['roles']))
    actual={(k,r['native_pts']):r for k,rr in groups.items()for r in rr};model=read_json(verify(manifest['current_model']))
    rows=list(read_jsonl(verify(manifest['input_cumulative_journal'])));receipts={r['source_row_id']:r for r in read_jsonl(verify(manifest['qualification_receipts']))}
    bindings=read_json(verify(manifest['track_material_bindings'])); bind={r['track_id']:r for r in bindings}
    if len(rows)!=len(actual) or len(receipts)!=len(actual) or set(bind)!=set(groups):
        raise ValueError('Board texture qualification lost complete track/source denominator')
    used=set()
    for row in rows:
        key=(row['physical_track_id'],row['native_pts']);old=actual[key];receipt=receipts[row['source_row_id']];b=bind[key[0]]
        if key in used or row['source_uv']!=old['source_uv'] or row['current_rgb_sha256']!=old['current_rgb_sha256']:
            raise ValueError('Board source RGB, UV or track identity changed')
        used.add(key);mi=b['material_index'];mat=model['materials'][mi];validate_point_receipt(row,receipt,mat,mi)
        block=model['blocks'][b['geometry_block']];p=mat['paths'][0];values=block['values']
        if (p['advertisement_block']!=b['advertisement_block'] or len(values)!=6 or block['constant']
            or block.get('fixed_indices')!=[0,1,3,4] or [values[k]for k in [0,1,3,4]]!=[1.,0.,0.,1.]
            or any(not 0<=values[k]<=1 or block['lower'][k]!=0 or block['upper'][k]!=1 for k in [2,5])
            or receipt['entity_kind']!=1 or receipt['entity_station_fixed']is not True
            or receipt.get('new_material_suffix')is not True or receipt.get('new_actual_source_geometry_qualified')is not True
            or any(receipt.get(k)is not False for k in ('solver_eligible','statistical_objective_eligible','numeric_statistics_ready'))):
            raise ValueError('Board texture requires exactly two unknown bounded material coordinates')
    return receipts,[dict(file=str(manifest_path),sha256=reference['sha256']),manifest['qualification_receipts'],
        dict(manifest['input_cumulative_journal'],role='qualified_current_entity_source_journal')]


def prepare_static_board_tracklets(parent_model,generation,qualification,roles,output):
    """Build a typed source bundle, never issue or edit a numerical packet."""
    from scipy.optimize import least_squares
    from .joint_model import graph_from_joint,as_event_view
    from .camera_graph import CameraGraph
    from .joint_replay import render_visibility
    from rinkfit_start.native_scene import NativeScene
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('New immutable board texture physical bundle required')
    groups=_source_contract(generation,qualification,roles);q=read_json(qualification);gen=read_json(generation)
    model=read_json(parent_model);result=copy.deepcopy(model);cg=CameraGraph(graph_from_joint(model));scene=NativeScene(as_event_view(model,'s01'))
    rows=[];receipts=[];latents=[];mapping={};fit_reports=[];projected=[];source_uv=[]
    for k, originals in sorted(groups.items()):
        first=originals[0];sid=first['shot_id'];instance=first['surface_instance_id'];adid=_ad_block(model,instance);ad=np.asarray(model['blocks'][adid]['values'])
        camera=cg.native_camera(sid);pts=np.array([r['native_pts']for r in originals],np.int64);times=cg.local_seconds(sid,pts);uv=np.asarray([r['source_uv']for r in originals])
        seed,hits=_ray_surface_seed(model,scene,camera,float(times[0]),uv[0],adid)
        def evaluate(x):
            X=scene.board(np.full(len(times),ad[0]+ad[1]*x[0]),np.full(len(times),ad[2]+(ad[3]-ad[2])*x[1]));pr=camera.project_rs(X,times)
            records=[dict(begin=0,end=len(X),label='board.texture',board_material_surface=True)]
            visible=render_visibility(camera,scene,X,records,times,pr['uv'])
            if not np.all(pr['valid']&visible):raise ValueError('Finite shared board point has invalid native RS or physical visibility')
            return pr['uv']
        before=evaluate(seed);fit=least_squares(lambda x:(evaluate(x)-uv).ravel(),seed,bounds=([0.,0.],[1.,1.]),method='trf',xtol=1e-12,ftol=1e-12,gtol=1e-9)
        coord=fit.x;after=evaluate(coord);mi=len(result['materials']);bi=len(result['blocks']);label='board.texture.'+k.replace('/','.');pathid='static_board_track/'+k
        result['blocks'].append(dict(name='board_texture_shared_uv/'+k,values=[1.,0.,float(coord[0]),0.,1.,float(coord[1])],
            lower=[1.,0.,0.,0.,1.,0.],upper=[1.,0.,1.,0.,1.,1.],scales=[1.]*6,constant=False,fixed_indices=[0,1,3,4]))
        result['materials'].append(dict(label=label,paths=[dict(id=pathid,variant_id='source_static_board_unknown_print_uv',
            kind=2,points=[[0.,0.]],geometry_block=bi,advertisement_block=adid,physical_point=True,closed=False,
            provenance=json.dumps(dict(track_id=k,source_board_instance=instance,static_surface_qualification=_ref(qualification),
                geometry_role='unknown_shared_material_uv_on_existing_continuous_board_ad',entity_station_fixed=0.,
                parent_camera_ray_and_numeric_installation_are_initialization_only=True,known_world_corner_correspondence=False),ensure_ascii=False))]))
        mapping[label]=dict(material_index=mi,geometry_ready=True)
        latents.append(dict(track_id=k,board_instance_id=instance,material_index=mi,path_index=0,geometry_block=bi,advertisement_block=adid,
            native_path_kind=2,points=[[0.,0.]],initial_unknown_material_uv=coord.tolist(),ray_seed_unknown_material_uv=seed.tolist(),
            shared_native_rows=len(originals),free_coordinate_indices=[2,5],fixed_indices=[0,1,3,4],
            world_coordinate_truth_or_covariance_claimed=False,one_shared_latent_per_track=True))
        fit_reports.append(dict(track_id=k,source_rows=len(originals),source_ray_board_intersections=hits,
            source_UV_changed=False,ray_seed_RS_RMS_px=float(np.sqrt(np.mean(np.sum((before-uv)**2,axis=1)))),
            shared_point_only_RS_RMS_px=float(np.sqrt(np.mean(np.sum((after-uv)**2,axis=1)))),
            camera_scene_existing_ad_blocks_optimized=False,optimizer_success=bool(fit.success),
            station_fixed=0.,Query_or_station_sliding=False,source_covariance_claimed=False))
        projected.extend(after);source_uv.extend(uv)
        for original in originals:
            pts0=original['native_pts'];rid='static_board_texture/'+k+'/'+str(pts0)
            row=dict(original,source_row_id=rid,source_family_id=k+'/'+str(pts0),physical_track_id=k,
                target_id=label,joint_material_id=label,joint_material_index=mi,current_uv=original['source_uv'],
                point_correspondence=True,measurement_operator_resolved=OPERATOR,source_geometry_eligible=True,
                statistical_objective_eligible=False,numeric_statistics_ready=False,solver_eligible=False,
                frame_role='development_candidate',normal=[0.,0.],entity_kind=1,entity_station=0.,
                world_position_is_shared_track_latent=True,
                physical_position='one shared unknown bounded Ad material [u,v] per track; not a known world corner')
            receipt=dict(schema=RECEIPT_SCHEMA,source_row_id=rid,track_id=k,native_pts=pts0,shot_id=sid,
                source_uv=original['source_uv'],current_rgb_sha256=original['current_rgb_sha256'],status='accepted_static_board_print',
                target_id=label,existing_material_index=mi,existing_point_path=pathid,point_path_kind=2,
                geometry_block=bi,advertisement_block=adid,source_board_instance_id=instance,
                physical_point=True,entity_kind=1,entity_station_fixed=True,entity_station=0.,new_material_suffix=True,
                new_actual_source_geometry_qualified=True,original_UV_preserved=original['source_uv'],
                current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=pts0,current_2D_UV=original['source_uv'],estimator=OPERATOR)),
                solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,
                original_flags_preserved=dict(manual_reference_preserved=False),static_surface_qualification=_ref(qualification),
                numeric_source_replay=q['numeric_replay'],known_world_corner_correspondence=False,unknown_shared_surface_coordinates=True)
            rows.append(row);receipts.append(receipt)
    out.mkdir(parents=True);result['static_board_shared_uv_append']=dict(parent_model=_ref(parent_model),
        source_generation=_ref(generation),static_surface_qualification=_ref(qualification),tracks=len(groups),source_rows=len(rows),
        unchanged_parent_blocks_materials_rigs_scene_shots=True,unknown_coordinates_per_track=2,
        source_qualified_material_identity_is_not_an_exact_world_or_template_corner_correspondence=True)
    write_json(out/'model.json',result);write_json(out/'latents.json',latents);write_jsonl(out/'source_rows.jsonl',rows)
    write_jsonl(out/'qualification_receipts.jsonl',receipts);write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    write_json(out/'initialization_report.json',dict(schema='rinkfit.source_board_shared_point_initialization.v1',per_track=fit_reports,
        initializer='current actual native camera observed-row ray / finite continuous board intersection; source-confirmed Ad instance selects branch',
        ice_H_or_template_corresponding_station_used=False,unknown_shared_material_coordinates_per_track=2,
        actual_shared_latent_count=len(groups),actual_free_coordinates=2*len(groups),camera_scene_or_parent_values_changed=False,
        optical_camera_ad_geometry_calibrated=False,accepted_full_physical_fit=False))
    np.savez_compressed(out/'source_projection.npz',source_uv=np.asarray(source_uv),prediction_uv=np.asarray(projected))
    gate=dict(schema=GATE_SCHEMA,source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        numeric_source_replay=q['numeric_replay'],current_RGB_receipt=q['current_RGB_receipt'],roles=_ref(roles),
        input_cumulative_journal=_ref(out/'source_rows.jsonl'),qualification_receipts=_ref(out/'qualification_receipts.jsonl'),
        current_model=_ref(out/'model.json'),track_material_bindings=_ref(out/'latents.json'),gate_code=_ref(__file__),
        qualified_new_measurement_rows=len(rows),old_raw_source_flags_false_are_not_overwritten=True,
        new_measurement_version_geometry_qualification_only=True)
    write_json(out/'qualification_gate.json',gate);counts=dict(Counter(r['shot_id']for r in rows))
    chunk=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,
        source_dataset_tag='static_board_tracklets_physical_v1',source_rows_ref=_ref(out/'source_rows.jsonl'),
        geometry_rows_ref=_ref(out/'source_rows.jsonl'),source_gate_manifest=_ref(out/'qualification_gate.json'),
        actual_source_rows=len(rows),by_shot=counts)
    write_json(out/'chunk_manifest.json',chunk)
    scripts=[Path(qualification).parent/n for n in ('review_decode.py','numeric_source_replay.py','terminal_diagnose.py')]
    bundle=dict(schema=BUNDLE_SCHEMA,parent_model=_ref(parent_model),source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        numeric_source_replay=q['numeric_replay'],current_RGB_receipt=q['current_RGB_receipt'],source_rows_ref=_ref(out/'source_rows.jsonl'),
        qualification_receipts=_ref(out/'qualification_receipts.jsonl'),qualification_gate=_ref(out/'qualification_gate.json'),
        track_material_bindings=_ref(out/'latents.json'),world_latent_bindings=_ref(out/'latents.json'),
        reader_module_refs=[_ref(__file__),*gen['reader_modules'],*[_ref(p)for p in scripts]],
        source_ledger_preserved=True,closed_chunks=[_ref(out/'chunk_manifest.json')],typed_entity_qualification=[_ref(out/'qualification_gate.json')],
        material_map=_ref(out/'material_map.json'),model=_ref(out/'model.json'),roles=_ref(roles),source_rows=len(rows),tracks=len(groups),
        free_shared_coordinates=2*len(groups),by_shot=counts,source_noise_statistics_supplied=False,
        absolute_source_UV_only=True,independent_displacement_factors=False,packet_issued=False)
    write_json(out/'manifest.json',bundle)
    load_point_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
    return bundle
