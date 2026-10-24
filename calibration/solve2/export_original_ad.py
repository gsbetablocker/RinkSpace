"""Export current delivered Ad2--12 evidence into the original advertisement API.

No optimizer is started. New whole-ad similarity values are explicitly reset;
only current static source shapes and their current source bindings are reused.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration' / 'rinkfit' / 'src'))
sys.path.insert(0, str(ROOT / 'calibration' / 'solve2'))
from rinksolve.model import load_model_bundle
from rinksolve.preparation import load_prepared
from rinksolve.observations import split_observation_id, observation_id
from rinkfit_start.ad_chart import PrintOnlyAdChart, effective_affine
from rinkfit_start.native_scene import NativeScene, camera_from_event


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path):
    p = Path(path).resolve()
    if len(str(p).encode('utf-16-le')) // 2 >= 260 or any(len(v.encode('utf-16-le')) // 2 > 255 for v in p.parts):
        raise ValueError('Windows path too long: ' + str(p))
    return p


def save_json(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def binary(directory, name, values, dtype):
    path = checked(Path(directory) / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(values, dtype=dtype)
    a.tofile(path)
    return dict(file=str(path), dtype=a.dtype.str, shape=list(a.shape), layout='C', byte_order='little', sha256=sha(path))


def export(prepared_dir, current_model_dir, original_model, output, *, data_root, replace_export=False, ad_id=12, scope_rows=None):
    ad_id=int(ad_id)
    if ad_id not in range(2,13):
        raise ValueError("Original advertisement export supports Ad2--12")
    ad_name=f"Ad{ad_id}"
    print_only=ad_id>=8
    chart_ad_ids=(ad_id,) if print_only else ()
    physical_dof=6 if print_only else 4
    out = checked(output)
    out.relative_to(Path(data_root).resolve())
    if out.exists() and any(out.iterdir()) and not replace_export:
        raise FileExistsError('Refuse to overwrite current export: ' + str(out))
    prepared = load_prepared(prepared_dir, data_root=data_root)
    current = load_model_bundle(current_model_dir)
    reference = load_model_bundle(Path(data_root) / 'runs/solve2_s01/ceres_regression/d78_exact')
    base = json.loads(Path(original_model).read_text(encoding='utf8'))
    key = f'artwork_group/{ad_name}/whole_material_object/similarity'
    reset = current.state.updated({key: np.zeros(4)})
    curves = {c.id: c for c in current.scene.curves}
    active = prepared.selection_current_active_rows()
    requested_scope=None
    if scope_rows is not None:
        requested_scope=np.asarray(np.load(scope_rows),dtype=np.int64)
        requested_base=requested_scope[requested_scope<prepared.row_count]
        if requested_scope.ndim!=1 or len(np.unique(requested_scope))!=len(requested_scope) or not np.all(np.isin(requested_base,active)) or (ad_id!=7 and len(requested_base)!=len(requested_scope)):
            raise ValueError('Current clear-frame scope must name unique currently admitted prepared source rows')
        active=active[np.isin(active,requested_scope)]
    bound = prepared.bound_curve_ids(active)
    registry={v['id']:v for v in prepared.manifest['curve_registry']}
    missing_target=[str(cid) for cid in set(bound) if str(cid) not in curves and
                    (registry[str(cid)].get('entity_id')==ad_name or registry[str(cid)]['label'].startswith(ad_name+'.'))]
    if missing_target:raise ValueError('Affected current target bindings need their delivered incremental update: '+str(missing_target))
    selected = np.asarray([str(cid) in curves and curves[str(cid)].definition.get('shared_layout_block_id') == key for cid in bound],dtype=bool)
    rows = active[selected]
    if requested_scope is not None and not np.array_equal(np.sort(rows),np.sort(requested_base)):
        raise ValueError('A current clear-frame scope row is not bound to this advertisement')
    if not len(rows) and not(ad_id==7 and requested_scope is not None and np.any(requested_scope>=prepared.row_count)):
        raise ValueError(f'Current delivered {ad_name} source inventory is empty')
    kinds = prepared.arrays['semantic_kind'][rows]
    if np.any((kinds!=0)&(kinds!=1)):
        raise ValueError(f'{ad_name} export requires finite curves or actual fixed entity corners; genuine Query needs its own delivered packet')
    current_ids = [str(v.decode('utf8') if isinstance(v, (bytes, np.bytes_)) else v) for v in prepared.observation_ids(rows)]
    if len(set(current_ids)) != len(rows):
        raise ValueError('Duplicate original stable observation ID')
    split = [split_observation_id(v) for v in current_ids]
    pts = np.asarray(prepared.arrays['native_pts'][rows], dtype='<i8')
    if any(v[1] != int(t) for v, t in zip(split, pts)):
        raise ValueError('Stable source ID/native PTS mismatch')
    uv = np.asarray(prepared.arrays['immutable_uv'][rows], dtype='<f8')
    E = np.asarray(prepared.arrays['engineering_budget_px'][rows], dtype='<f8')
    station = np.asarray(prepared.arrays['station_seed'][rows], dtype='<f8')
    row = np.asarray(prepared.arrays['row_auxiliary_seed'][rows], dtype='<f8')
    if not np.isfinite(np.r_[uv.ravel(), E, station, row]).all() or np.any(E <= 0) or np.any((station < 0) | (station > 1)) or np.any((row < 0) | (row > 1079)):
        raise ValueError('Current source or legal initial auxiliary is invalid')
    used_ids = list(dict.fromkeys(str(v) for v in bound[selected]))
    used_ids += [c.id for c in current.scene.curves if c.definition.get('shared_layout_block_id')==key and c.id not in used_ids]
    material = np.asarray([used_ids.index(str(v)) for v in bound[selected]], dtype='<u4')
    inverse = np.full(prepared.row_count, -1, dtype=np.int64)
    inverse[rows] = np.arange(len(rows))
    source_identity = hashlib.sha256(b''.join([np.ascontiguousarray(a).tobytes() for a in (rows, pts, uv)]) + '\n'.join(current_ids).encode('utf8')).hexdigest()


    # Delivered fixed artwork corners are observations, not per-frame stations.
    # They live outside the legacy raw SoA and must be appended explicitly.
    extra_definitions={}
    extra_origins={}
    source_rows_full=rows.copy()
    sigma_values=np.asarray(prepared.arrays['source_sigma'][rows],dtype='<f8').copy()
    supplemental=[]
    if ad_id==7:
        features={f['id']:f for f in prepared.manifest.get('ad7_fixed_source_features',[])}
        from types import SimpleNamespace
        for original_index,item in enumerate(prepared.manifest.get('ad7_fixed_observations',[])):
            synthetic_row=prepared.row_count+original_index
            if requested_scope is not None and synthetic_row not in requested_scope:continue
            feature=features[item['fixed_feature_id']]
            if feature.get('station_is_optimizer_variable') or item.get('per_observation_world_station'):
                raise ValueError('Delivered fixed Ad7 corners must not acquire local station DOF')
            parent_id=feature['curve_id'];fixed=float(feature['fixed_station']);cid='fixed_corner/'+feature['id']
            if cid not in extra_definitions:
                definition=dict(current.scene.resolved_definition(curves[parent_id],reset))
                pp=np.asarray(definition['unwrapped_station_height'],float)
                x=fixed*(len(pp)-1);j=min(max(0,int(np.floor(x))),max(0,len(pp)-2))
                point=pp[0] if len(pp)==1 else pp[j]+(x-j)*(pp[j+1]-pp[j])
                definition.update(unwrapped_station_height=[point.tolist()],closed=False,source_fixed_feature_id=feature['id'])
                extra_definitions[cid]=definition;extra_origins[cid]=(parent_id,fixed)
                curves[cid]=SimpleNamespace(id=cid,label=feature['id'],definition=definition)
                used_ids.append(cid)
            index=len(supplemental)
            stable=observation_id('ad7_fixed_material_corners',int(item['native_pts']),item['id'])
            if stable in current_ids:raise ValueError('Duplicate delivered supplemental source')
            current_ids.append(stable);supplemental.append(dict(item))
            material=np.r_[material,np.uint32(used_ids.index(cid))]
            pts=np.r_[pts,np.int64(item['native_pts'])]
            uv=np.vstack([uv,np.asarray(item['uv_native_pixels'],float)])
            E=np.r_[E,float(item['engineering_budget_px'])]
            station=np.r_[station,0.]
            row=np.r_[row,float(item['uv_native_pixels'][1])]
            kinds=np.r_[kinds,np.uint8(1)]
            sigma_values=np.r_[sigma_values,np.nan]
            source_rows_full=np.r_[source_rows_full,np.int64(synthetic_row)]
    N=len(pts)
    if requested_scope is not None and not np.array_equal(np.sort(source_rows_full),np.sort(requested_scope)):
        raise ValueError('The current scope contains an unknown supplemental source row')
    split=[split_observation_id(v)for v in current_ids]
    source_identity=hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes()for a in (source_rows_full,pts,uv))+'\n'.join(current_ids).encode('utf8')).hexdigest()

    # Reuse the exact d78 physical state. Only this whole artwork affine is free.
    model = copy.deepcopy(base)
    for block in model['blocks']:
        block['constant'] = True
    ai = next(i for i, b in enumerate(model['blocks']) if b['name'] == f'ad/{ad_id}')
    witness = np.asarray(model['blocks'][ai]['values'], dtype=float)
    old_qs = sorted({p['geometry_block'] for m in base['materials'] for p in m['paths'] if p.get('advertisement_block') == ai and p.get('geometry_block', -1) >= 0})
    if not old_qs:
        raise ValueError(f'Original {ad_name} affine domain is missing')
    qdefs = [base['blocks'][i] for i in old_qs]
    if any((d['lower'], d['upper'], d['scales']) != (qdefs[0]['lower'], qdefs[0]['upper'], qdefs[0]['scales']) for d in qdefs):
        raise ValueError(f'Original {ad_name} affine boxes differ; do not invent a shared domain')
    qi = len(model['blocks'])
    model['blocks'].append(dict(name=f'print/{ad_name}_current_whole', values=[1., 0., 0., 0., 1., 0.], lower=qdefs[0]['lower'], upper=qdefs[0]['upper'], scales=qdefs[0]['scales'], constant=not print_only))
    model['blocks'][ai]['constant']=print_only
    model['source_identity'] = source_identity
    model['model_id'] = base['model_id'] + f'/current_{ad_name}_original_'+('affine6' if print_only else 'installation4')
    model['print_only_ad_ids'] = list(chart_ad_ids)
    origin = witness[[0, 2]]
    span = np.array([witness[1], witness[3] - witness[2]])
    if np.any(span <= 0):
        raise ValueError('Original common witness is noninvertible')
    materials = []
    for cid in used_ids:
        c = curves[cid]
        d = extra_definitions[cid] if cid in extra_definitions else current.scene.resolved_definition(c, reset)
        if d['kind'] != 'board_material_polyline' or d.get('shared_layout_block_id') != key:
            raise ValueError(f'{ad_name} shape is outside the current static shared source geometry')
        points = np.asarray(d['unwrapped_station_height'], dtype=float)
        local = (points - origin) / span
        path = dict(id=cid, variant_id=cid, kind=2, geometry_block=qi, advertisement_block=ai,
                    points=local.tolist(), closed=bool(d.get('closed', False)), physical_point=len(local)==1,
                    provenance=json.dumps(dict(current_curve_id=cid, current_definition=d, new_stage_similarity_reset_to_zero=True), ensure_ascii=False))
        is_outer=not print_only and '/installation' in str(d.get('original_path_id',''))
        if is_outer:path.pop('geometry_block')
        materials.append(dict(label=c.label, paths=[path], observed_installation_boundary=is_outer))
    # Retain the original fixed print blocks' domain witnesses unchanged.
    # The observed current graph above alone uses the new shared affine.
    legacy = copy.deepcopy(base['materials'])
    for mat in legacy:
        mat['unused_legacy_witness'] = True
    model['materials'] = materials + legacy
    model['current_input_scope'] = dict(schema='rinkfit.current_advertisement_input.v1',
                                       scope_id=ad_name, print_only_ad_ids=list(chart_ad_ids))
    model.setdefault('provenance', {})['current_prepared_ad_export'] = dict(
        source_model=str(Path(current_model_dir).resolve()), prepared=str(Path(prepared_dir).resolve()),
        original_physical_model=str(Path(original_model).resolve()), source_geometry_only=True,
        current_clear_frame_scope=None if scope_rows is None else dict(path=str(Path(scope_rows).resolve()),sha256=sha(scope_rows),rows=len(requested_scope),pending_other_current_rows_are_not_permanent_source_rejections=True),
        rejected_new_similarity_initial_values=current.state.get(key).tolist(), used_similarity=[0., 0., 0., 0.],
        physical_DOF=(f'one shared original affine6 for {ad_name}; original installation is dependent witness' if print_only else f'original ad/{ad_id} observed installation4 free; shared print affine6 fixed identity'),
        original_ad_and_print_boxes_unchanged=True, camera_ice_board_fixed_to_original_d78=True,
        old_optimizer_history_and_duals_imported=False)
    chart = PrintOnlyAdChart(model, ad_ids=chart_ad_ids)
    physical0 = chart.pack()
    decoded, domain = chart.unpack(physical0, validate=True, return_domain=True)
    if len(chart.free) != physical_dof or len(chart.print_ad) != (1 if print_only else 0):
        raise AssertionError('Export leaked physical degrees of freedom')
    scene = NativeScene(decoded)
    world = np.empty((N, 3))
    world_current = np.empty_like(world)
    for mi, cid in enumerate(used_ids):
        use = np.flatnonzero(material == mi)
        world[use] = scene.world(mi, 0, station[use])
        world_current[use] = current.scene.world_path(extra_origins[cid][0], np.full(len(use),extra_origins[cid][1]), reset) if cid in extra_origins else current.scene.world_path(cid, station[use], reset)
    world_delta = float(np.max(np.abs(world - world_current)))
    if world_delta > 1e-10:
        raise AssertionError('Current static source geometry changed in the original API')
    clock = (pts - 256256) / 12800.
    times = clock + base['blocks'][0]['values'][31] * (row / 1079. - .5)
    camera = camera_from_event(decoded)
    forward = camera.at(world, times)
    current_uv = current.operator('S01').project_at(world_current, times)['uv']
    uv_delta = float(np.max(np.abs(forward['uv'] - current_uv)))
    if uv_delta > 1e-6:
        raise AssertionError('Original camera/source forward mapping changed')
    # Compare old source curves without overriding incoming verified revisions.
    old_checks = []
    reference_ids={c.id for c in reference.scene.curves}
    for cid in used_ids:
        if cid not in reference_ids:
            continue
        s = np.linspace(0, 1, 17)
        a = current.scene.world_path(cid, s, reset)
        b = reference.scene.world_path(cid, s, reference.state)
        delta = float(np.max(np.abs(a - b)))
        old_checks.append(dict(curve_id=cid, stations=17,
            reset_world_vs_d78_exact_max=delta,
            current_geometry_revision_preserved=bool(delta > 1e-10)))

    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'model.json', model)
    for name, values, dtype in [('station.f64', station, '<f8'), ('path.u32', np.zeros(N), '<u4'), ('initial_row.f64', row, '<f8'), ('initial_alpha.f64', np.zeros(N), '<f8')]:
        binary(out, name, values, dtype)
    pd, sd, qd = out / 'packet', out / 'semantic', out / 'query'
    pd.mkdir(exist_ok=True); sd.mkdir(exist_ok=True); qd.mkdir(exist_ok=True)
    arrays = {}
    for key0, name, value, dtype in [('native_uv', 'uv.f64', uv, '<f8'), ('source_rows', 'row.u64', np.arange(N), '<u8'), ('native_pts', 'pts.i64', pts, '<i8'), ('frame_index', 'frame.i32', 1 + (pts - 256256) // 256, '<i4'), ('material_index', 'material.u32', material, '<u4'), ('point_id_index', 'point.u32', np.arange(N), '<u4'), ('engineering_budget_px', 'budget.f64', E, '<f8'), ('source_rows_full', 'source_rows_full.u64', source_rows_full, '<u8')]:
        arrays[key0] = binary(pd, name, value, dtype)
    strings = dict(material_labels=[curves[k].label for k in used_ids], source_point_ids=current_ids,
                   stable_observation_ids=current_ids, original_source_ids=[v[2] for v in split], original_source_dataset_ids=[v[0] for v in split],
                   source_point_ids_are_complete_current_keys=True, original_ids_preserved_verbatim=True,
                   supplemental_fixed_corner_observations=supplemental,
                   source_rows_full_include_appended_typed_supplement=bool(supplemental))
    save_json(pd / 'string_tables.json', strings)
    sa = {}
    for key0, name, value, dtype in [('kind', 'kind.u8', kinds, '|u1'), ('variant_index', 'variant.u32', material, '<u4'), ('query_uv', 'query_uv.f64', np.zeros_like(uv), '<f8'), ('profile_normal', 'normal.f64', np.zeros_like(uv), '<f8'), ('alpha_bounds', 'alpha_bounds.f64', np.zeros_like(uv), '<f8'), ('source_sigma', 'sigma.f64', sigma_values, '<f8'), ('ordinary_rows', 'ordinary_rows.u64', np.empty(0), '<u8')]:
        sa[key0] = binary(sd, name, value, dtype)
    statmask = np.zeros(N, dtype='u1')
    groups, memberships, weights, independent = [], [], [], []
    formal_rank = 0
    for factor in prepared.manifest['statistical_factors']:
        if factor['residual_count'] == 0:
            continue
        original_rows = np.asarray(factor['rows'], dtype=np.int64)
        if not np.any(inverse[original_rows] >= 0):
            continue
        if np.any(inverse[original_rows] < 0):
            raise ValueError(f'A complete current W factor crosses {ad_name} scope; constant support must be explicit')
        local_rows = inverse[original_rows]
        if np.any(statmask[local_rows]):
            raise ValueError('Statistical source counted twice')
        statmask[local_rows] = 1
        start, rank = factor['residual_start'], factor['residual_count']
        cols = (2 * original_rows[:, None] + np.arange(2)).ravel()
        W = prepared.whitening[start:start + rank, cols].toarray()
        formal_rank += rank
        if factor['kind'] == 'original_independent_scalar_sigma':
            sigma = prepared.arrays['source_sigma'][original_rows[0]]
            if len(local_rows) != 1 or rank != 2 or not np.array_equal(W, np.eye(2) / sigma):
                raise ValueError('Independent source operator differs from the current original sigma')
            independent.append(int(local_rows[0]))
            continue
        groups.append(dict(group_id=factor['id'], rank=rank, member_start=len(memberships), member_count=len(local_rows),
            weight_start=len(weights), matrix_shape=list(W.shape), matrix_sha256=hashlib.sha256(W.tobytes()).hexdigest(),
            covariance_origin='unchanged complete current prepared factor ' + factor['id'], one_statistical_measurement_per_original_group=True,
            native_pts=int(pts[local_rows[0]]), frame=int(1 + (pts[local_rows[0]] - 256256) // 256), source_prepared_kind=factor['kind']))
        memberships.extend(local_rows.tolist()); weights.extend(W.ravel().tolist())
    sa['group_rows'] = binary(sd, 'group_rows.u64', memberships, '<u8')
    sa['weights'] = binary(sd, 'weights.f64', weights, '<f8')
    sa['ordinary_rows'] = binary(sd, 'ordinary_rows.u64', independent, '<u8')
    (sd / 'groups.jsonl').write_text(''.join(json.dumps(g, ensure_ascii=False) + '\n' for g in groups), encoding='utf8')
    (sd / 'runs.jsonl').write_text('', encoding='utf8')
    am = binary(sd, 'active.u8', np.ones(N), '|u1')
    sm = binary(sd, 'statistical_active.u8', statmask, '|u1')
    stats = dict(schema='rinkfit.active_statistics.v1', active_observations=N, statistical_observations=int(statmask.sum()),
                 formal_residual_rows=formal_rank, original_group_count=len(groups), independent_rows=len(independent),
                 active_source_mask_sha256=am['sha256'], statistical_active_mask_sha256=sm['sha256'],
                 statistical_partition_exactly_once=True, budget_only_observations=int(np.sum(statmask == 0)),
                 budget_only_noise_missing_is_explicit=True, new_W_sigma_or_weight_invented=False)
    save_json(sd / 'active_statistics.json', stats)
    semantic = dict(schema='rinkfit.semantic_packet.v1', observations=N, shot_id='s01', frames=1248,
        material_labels=len(used_ids), semantic_counts=dict(finite_curve=int(np.sum(kinds==0)), entity_corner=int(np.sum(kinds==1)), true_profile=0),
        arrays=sa, assets={'groups.jsonl': dict(file=str(sd / 'groups.jsonl'), sha256=sha(sd / 'groups.jsonl')),
                           'runs.jsonl': dict(file=str(sd / 'runs.jsonl'), sha256=sha(sd / 'runs.jsonl'))},
        variant_ids=used_ids, active_source_mask=am, statistical_active_mask=sm, active_observations=N,
        active_statistics=dict(file=str(sd / 'active_statistics.json'), sha256=sha(sd / 'active_statistics.json')),
        source_runs=dict(proven_runs=0), profiles=0, statistics=stats,
        current_prepared_export=dict(schema='rinkfit.current_prepared_source_adapter.v1', prepared_manifest=str(Path(prepared_dir).resolve() / 'manifest.json'),
             source_observation_manifest=prepared.manifest['source_observation_manifest'], source_rows_full=arrays['source_rows_full'],
             original_raw_rows=prepared.row_count, source_identity=source_identity, prior_withdrawals_not_reactivated=True,
             source_preparation_or_qualification_replayed=False, active_and_statistical_masks_are_distinct=True))
    scope = dict(schema='rinkfit.current_advertisement_input.v1', scope_id=ad_name, source_frame_count=1248,
        group_count=len(groups), independent_count=len(independent), formal_W_rank=formal_rank, run_count=0,
        statistical_active_mask=sm, print_only_ad_ids=list(chart_ad_ids))
    model['current_input_scope'] = scope
    save_json(out / 'model.json', model)
    semantic['current_input_scope'] = scope
    semantic['frames'] = len(np.unique(pts))
    semantic['resolved_material_index'] = arrays['material_index']
    semantic['resolved_variant_index'] = sa['variant_index']
    semantic['resolved_variant_ids'] = used_ids
    save_json(sd / 'manifest.json', semantic)
    qa = {}
    for k, val, dt in [('source_rows', np.empty(0), '<u8'), ('offsets', [0], '<u8'), ('alpha', np.empty(0), '<f8'), ('uv', np.empty((0, 2)), '<f8'), ('bounds', np.empty((0, 2)), '<f8'), ('numerator_u', np.empty((0, 4)), '<f8'), ('numerator_v', np.empty((0, 4)), '<f8'), ('denominator', np.empty((0, 4)), '<f8')]:
        qa[k] = binary(qd, k + '.bin', val, dt)
    save_json(qd / 'certificates.json', dict(profiles=0, no_query_fabricated=True))
    query = dict(schema='rinkfit.immutable_native_query_curve_packet.v1', profiles=0,
        semantic_manifest=dict(path=str(sd / 'manifest.json'), sha256=sha(sd / 'manifest.json')), arrays=qa,
        certificates=dict(path=str(qd / 'certificates.json'), sha256=sha(qd / 'certificates.json')))
    save_json(qd / 'manifest.json', query)
    qa0 = binary(out, 'query_initial_alpha.f64', [], '<f8')
    save_json(out / 'query_initialization.json', dict(query_packet_sha256=sha(qd / 'manifest.json'), alpha=qa0))
    packet = dict(schema='rinkfit.event_packet.v1', observations=N, frame_count=len(np.unique(pts)), source_frame_count=1248, observed_frame_count=len(np.unique(pts)),
        shot_id='s01', native_image_wh=[1920, 1080], native_time_base=[1, 12800], pts_origin=256256,
        arrays=arrays, string_tables=dict(file=str(pd / 'string_tables.json'), sha256=sha(pd / 'string_tables.json')), assets={},
        immutable_source_hashes={'observations.npz': source_identity}, source_identity_kind='current compact source_rows_full, native_pts, raw UV and complete stable IDs; legacy key is an ABI tag',
        semantic_packet=dict(manifest=str(sd / 'manifest.json'), sha256=sha(sd / 'manifest.json')),
        query_packet=dict(manifest=str(qd / 'manifest.json'), sha256=sha(qd / 'manifest.json')),
        event_model=dict(manifest=str(out / 'model.json'), sha256=sha(out / 'model.json')), initial_state_directory=str(out),
        current_prepared_export=semantic['current_prepared_export'], current_input_scope=scope, primary_key=['source_dataset_id', 'native_pts', 'original_source_id'])
    save_json(pd / 'manifest.json', packet)
    save_json(out / 'initializer_manifest.json',dict(
        schema='rinkfit.current_source_numerical_seed.v1',
        semantic_manifest=dict(path=str(sd / 'manifest.json'),sha256=sha(sd / 'manifest.json')),
        files={name:dict(path=str(out/name),sha256=sha(out/name)) for name in
            ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')},
        seed_only_not_complete_finite_foot_optimization=True,
        use_original_full2_for_new_source_or_geometry=True,
        accepted_unchanged_full2_state_may_be_consumed_directly=True))
    report = dict(schema='rinkfit.current_ad_original_affine_export.v1', ad_id=ad_id, Ceres_or_Clarabel_started=False,
        packet=str(pd / 'manifest.json'), semantic=str(sd / 'manifest.json'), query=str(qd / 'manifest.json'),
        model=str(out / 'model.json'), initial_state_directory=str(out), print_only_ad_ids=list(chart_ad_ids), raw_rows=N,
        original_rows=int(np.sum(rows < 182017)), new_budget_only_rows=int(np.sum(source_rows_full >= 182017)), supplemental_fixed_corner_rows=len(supplemental),
        statistical_source_rows=int(statmask.sum()), W_rank=stats['formal_residual_rows'], statistical_groups=len(groups), independent_rows=len(independent),
        all_original_UV_PTS_ID_E_W_unchanged=True, compact_row_mapping=arrays['source_rows_full'],
        current_prepared_withdrawal_reference=dict(manifest=str(Path(prepared_dir).resolve() / 'manifest.json'), sha256=sha(Path(prepared_dir) / 'manifest.json')),
        reset_similarity=old_checks, ignored_new_similarity=current.state.get(key).tolist(),
        original_d78_model_sha256=sha(original_model), original_boxes_preserved=True, original_domain_decoder=chart.report(),
        decoded_original_domain=domain, current_static_source_world_delta_max=world_delta, original_camera_forward_delta_px_max=uv_delta,
        physical_parameters_free=physical_dof, temporary_ad_installation_parameters_free=0 if print_only else 4, camera_ice_board_parameters_free=0,
        initializer_is_not_true_RS_or_KKT_certificate=True,
        new_source_or_geometry_requires_current_original_full2_before_qbar=True,
        accepted_full2_entry='current_ad_prepare.py --model ACCEPTED/model.json --initial-dir ACCEPTED_FULL2', original_model_state_is_native_recomputed_candidate_not_formal_PHR_terminal=True)
    save_json(out / 'mapping.json', report)
    return report



def apply_material_delta(input_dir, delta_path):
    """Apply only explicitly reviewed material bindings and their station seeds.

    The compact raw material/variant columns retain their previous binding.
    All evidence/statistics bytes stay unchanged; resolved columns carry edits.
    """
    root = checked(input_dir)
    pp, sp, qp = root / 'packet/manifest.json', root / 'semantic/manifest.json', root / 'query/manifest.json'
    packet = json.loads(pp.read_text(encoding='utf8'))
    semantic = json.loads(sp.read_text(encoding='utf8'))
    model = json.loads((root / 'model.json').read_text(encoding='utf8'))
    delta = json.loads(Path(delta_path).read_text(encoding='utf8'))
    records = delta.get('records')
    if records is None and delta.get('schema')=='rinkspace.Ad7.specific_material_binding_correction.v1':
        records=[dict(source_id=r['stable_observation_id'],original_row=r['source_full_row'],native_pts=r['native_pts'],immutable_uv=r['raw_uv'],old_curve_id=r['previous_curve_id'],new_curve_id=r['correct_curve_id'],evidence_image=delta['reviewed_original_native_files'])for r in delta['deltas']]
    if records is None:raise ValueError('An explicit reviewed per-source material delta is required')
    receipt = root / 'binding_delta_01.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf8'))
        if previous['reviewed_delta']['sha256'] == sha(delta_path):
            return previous
        raise FileExistsError('A different source delta is already recorded')
    def array(desc):
        p = Path(desc.get('file', desc.get('path')))
        if sha(p) != desc['sha256']:
            raise ValueError('Current input array changed: ' + str(p))
        return np.fromfile(p, desc['dtype']).reshape(desc['shape'])
    tables = json.loads(Path(packet['string_tables']['file']).read_text(encoding='utf8'))
    ids = tables['stable_observation_ids']
    rows_full = array(packet['arrays']['source_rows_full'])
    pts = array(packet['arrays']['native_pts'])
    uv = array(packet['arrays']['native_uv'])
    material = array(semantic['resolved_material_index']).copy()
    variant = array(semantic['resolved_variant_index']).copy()
    statmask = array(semantic['statistical_active_mask'])
    station = np.fromfile(root / 'station.f64', '<f8')
    row = np.fromfile(root / 'initial_row.f64', '<f8')
    scene, camera = NativeScene(model), camera_from_event(model)
    untouched = {str(Path(v['file'])): sha(v['file']) for v in packet['arrays'].values()}
    untouched.update({str(Path(v['file'])): sha(v['file']) for v in semantic['arrays'].values()})
    for name in ('path.u32', 'initial_row.f64', 'initial_alpha.f64'):
        untouched[str(root / name)] = sha(root / name)
    changes = []
    for record in records:
        if ids.count(record['source_id']) != 1:
            raise ValueError('Reviewed source ID is absent or ambiguous')
        i = ids.index(record['source_id'])
        if int(rows_full[i]) != record['original_row'] or int(pts[i]) != record['native_pts'] or uv[i].tolist() != record['immutable_uv']:
            raise ValueError('Reviewed native source bytes disagree')
        old_path = model['materials'][int(material[i])]['paths'][0]
        if old_path['id'] != record['old_curve_id']:
            raise ValueError('Reviewed previous binding differs')
        matches = [(j, p) for j, m in enumerate(model['materials']) for p in m['paths'] if p['id'] == record['new_curve_id']]
        if len(matches) != 1 or matches[0][1]['kind'] != 2 or matches[0][1].get('physical_point', False):
            raise ValueError('Reviewed new finite contour is absent or ambiguous')
        j, path = matches[0]
        if statmask[i]:
            raise ValueError('This delta needs explicit complete-W group handling before changing a statistical source')
        # This is a legal initializer on the already image-confirmed contour;
        # native reprojection is never used to decide the material identity.
        grid = np.linspace(0., 1., 8193)
        t = (int(pts[i]) - 256256) / 12800. + model['blocks'][0]['values'][31] * (row[i] / 1079. - .5)
        forward = camera.at(scene.world(j, 0, grid), t)
        grid_uv = forward['uv']
        a, b = grid_uv[:-1], grid_uv[1:]
        edge = b - a
        den = np.sum(edge * edge, axis=1)
        w = np.clip(np.divide(np.sum((uv[i] - a) * edge, axis=1), den, out=np.zeros_like(den), where=den > 0), 0., 1.)
        distance = np.sum((a + w[:, None] * edge - uv[i]) ** 2, axis=1)
        distance[~np.isfinite(distance)] = np.inf
        k = int(np.argmin(distance))
        seed = float((k + w[k]) / (len(grid) - 1))
        if not np.isfinite(distance[k]) or not 0 <= seed <= 1:
            raise ValueError('No finite legal initialization on the reviewed contour')
        material[i] = j
        variant[i] = semantic['resolved_variant_ids'].index(path['variant_id'])
        changes.append(dict(source_id=record['source_id'], compact_row=i, source_row_full=int(rows_full[i]), native_pts=int(pts[i]),
            immutable_uv=uv[i].tolist(), old_curve_id=old_path['id'], new_curve_id=path['id'],
            old_station_seed=float(station[i]), new_station_seed=seed, formal_row_unchanged=float(row[i]),
            station_seed_method='legal dense native projection grid on image-confirmed finite contour; numerical seed only',
            image_review_evidence=record['evidence_image'], material_identity_selected_by_native_image=True,
            per_frame_landmark_created=False))
        station[i] = seed
    save_json(root / 'binding_delta_previous_packet.json', packet)
    save_json(root / 'binding_delta_previous_semantic.json', semantic)
    before = dict(packet_sha256=sha(pp), semantic_sha256=sha(sp), station_sha256=sha(root / 'station.f64'))
    semantic['original_material_index'] = packet['arrays']['material_index']
    semantic['resolved_material_index'] = binary(root / 'semantic', 'resolved_material.u32', material, '<u4')
    semantic['resolved_variant_index'] = binary(root / 'semantic', 'resolved_variant.u32', variant, '<u4')
    semantic['current_specific_binding_delta'] = dict(path=str(Path(delta_path).resolve()), sha256=sha(delta_path),
        records=changes, previous_semantic=dict(path=str(root / 'binding_delta_previous_semantic.json'), sha256=sha(root / 'binding_delta_previous_semantic.json')))
    save_json(sp, semantic)
    station.astype('<f8').tofile(root / 'station.f64')
    query = json.loads(qp.read_text(encoding='utf8'))
    query['semantic_manifest'] = dict(path=str(sp), sha256=sha(sp))
    save_json(qp, query)
    init_path = root / 'query_initialization.json'
    init = json.loads(init_path.read_text(encoding='utf8'))
    init['query_packet_sha256'] = sha(qp)
    save_json(init_path, init)
    packet['semantic_packet']['sha256'] = sha(sp)
    packet['query_packet']['sha256'] = sha(qp)
    save_json(pp, packet)
    if any(sha(path) != digest for path, digest in untouched.items()):
        raise AssertionError('A raw source/statistical/unchanged auxiliary array changed')
    result = dict(schema='rinkfit.current_specific_binding_delta_applied.v1',
        reviewed_delta=dict(path=str(Path(delta_path).resolve()), sha256=sha(delta_path)), records=changes,
        previous_input=before, current_packet_sha256=sha(pp), current_semantic_sha256=sha(sp), current_station_sha256=sha(root / 'station.f64'),
        source_arrays_sigma_W_E_ID_PTS_unchanged=True, raw_binding_columns_retained_as_history=True,
        no_source_retired=True, all_other_source_bindings_unchanged=True, solver_started=False,
        unchanged_array_sha256=untouched)
    save_json(receipt, result)
    initializer_path=root/'initializer_manifest.json'
    if initializer_path.exists():
        initializer=json.loads(initializer_path.read_text(encoding='utf8'))
        initializer['semantic_manifest']=dict(path=str(sp),sha256=sha(sp))
        initializer['files']={name:dict(path=str(root/name),sha256=sha(root/name))for name in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')}
        save_json(initializer_path,initializer)
    mapping_path = root / 'mapping.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf8'))
    mapping['specific_binding_delta'] = dict(path=str(receipt), sha256=sha(receipt), affected_sources=len(changes))
    save_json(mapping_path, mapping)
    return result



def apply_I_geometry_delta(input_dir, geometry_path):
    """Replace the one verified stale I contour, including its same-entity alias."""
    root = checked(input_dir)
    mp, pp = root / 'model.json', root / 'packet/manifest.json'
    model = json.loads(mp.read_text(encoding='utf8'))
    geometry = json.loads(Path(geometry_path).read_text(encoding='utf8'))
    points = np.asarray(geometry['current_w139_transformed_points'], dtype=float)
    if points.shape != (10, 2) or not geometry['current_w139_is_same_10_vertex_I_entity'] or not geometry['current_w139_closed']:
        raise ValueError('This correction requires the reviewed same ten-vertex closed I entity')
    receipt = root / 'geometry_delta_012.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf8'))
        if previous['current_geometry_evidence']['sha256'] == sha(geometry_path):
            return previous
        raise FileExistsError('A different I geometry repair is already present')
    before = copy.deepcopy(model)
    ai = next(i for i, b in enumerate(model['blocks']) if b['name'] == 'ad/12')
    ad = np.asarray(model['blocks'][ai]['values'], dtype=float)
    span = np.array([ad[1], ad[3] - ad[2]])
    local = (np.vstack([points, points[0]]) - ad[[0, 2]]) / span
    repaired = []
    for mat in model['materials']:
        if mat.get('unused_legacy_witness', False):
            continue
        for path in mat['paths']:
            if path['id'] not in ('d78_material_012/path0', 'w139/path0'):
                continue
            if len(path['points']) != 11 or not path.get('closed', False):
                raise ValueError('Old I finite path topology differs from the reviewed vertex correspondence')
            q = model['blocks'][path['geometry_block']]['values']
            if q != [1., 0., 0., 0., 1., 0.]:
                raise ValueError('Do not install source geometry into an already moved affine state')
            repaired.append(dict(path_id=path['id'], old_points=path['points'], new_points=local.tolist(),
                                 vertex_count=11, unique_vertex_count=10, station_index_order_unchanged=True))
            path['points'] = local.tolist()
    if {v['path_id'] for v in repaired} != {'d78_material_012/path0'}:
        raise ValueError('Current observed I path was not located')
    model.setdefault('provenance', {})['current_Ad12_I_geometry_repair'] = dict(
        current_geometry_evidence=dict(path=str(Path(geometry_path).resolve()), sha256=sha(geometry_path)),
        repaired_paths=[v['path_id'] for v in repaired], reason='source-delivered same I contour lies between the current H and R; stale original I was outside R',
        original_d78_012_preservation_is_explicitly_superseded=True, shield_011_camera_ice_board_preserved=True,
        no_new_shape_or_parameter_freedom=True, no_source_pixels_or_admission_changed=True)
    chart = PrintOnlyAdChart(model, ad_ids=(12,))
    _, domain = chart.unpack(chart.pack(), validate=True, return_domain=True)
    if model['blocks'] != before['blocks'] or model['camera'] != before['camera']:
        raise AssertionError('A physical parameter or camera definition changed')
    changed_paths = [(a['id'], b['id']) for ma, mb in zip(before['materials'], model['materials'])
                     for a, b in zip(ma['paths'], mb['paths']) if a != b]
    if changed_paths != [(v['path_id'], v['path_id']) for v in repaired]:
        raise AssertionError('Unrelated geometry changed')
    save_json(root / 'geometry_delta_previous_model.json', before)
    packet = json.loads(pp.read_text(encoding='utf8'))
    arrays_before = copy.deepcopy(packet['arrays'])
    save_json(mp, model)
    packet['event_model']['sha256'] = sha(mp)
    save_json(pp, packet)
    if arrays_before != packet['arrays']:
        raise AssertionError('A source array descriptor changed')
    result = dict(schema='rinkfit.current_Ad12_I_static_geometry_delta.v1',
        current_geometry_evidence=dict(path=str(Path(geometry_path).resolve()), sha256=sha(geometry_path)),
        old_model=dict(path=str(root / 'geometry_delta_previous_model.json'), sha256=sha(root / 'geometry_delta_previous_model.json')),
        current_model=dict(path=str(mp), sha256=sha(mp)), records=repaired,
        original_d78_012_preservation_is_explicitly_superseded=True, camera_all_parameter_values_unchanged=True,
        shield_011_and_every_other_path_unchanged=True, UV_PTS_ID_E_W_sigma_binding_and_auxiliary_files_unchanged=True,
        original_affine_witness_domain_passed=domain, solver_started=False)
    save_json(receipt, result)
    initializer_path=root/'initializer_manifest.json'
    if initializer_path.exists():
        initializer=json.loads(initializer_path.read_text(encoding='utf8'))
        initializer['semantic_manifest']=dict(path=str(sp),sha256=sha(sp))
        initializer['files']={name:dict(path=str(root/name),sha256=sha(root/name))for name in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')}
        save_json(initializer_path,initializer)
    mapping_path = root / 'mapping.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf8'))
    for v in mapping['reset_similarity']:
        if v['curve_id'] == 'd78_material_012/path0' and 'reset_world_vs_d78_exact_max' in v:
            v['pre_geometry_repair_reset_world_vs_d78_exact_max'] = v.pop('reset_world_vs_d78_exact_max')
            v['final_geometry_is_explicit_verified_source_repair'] = True
    for key in ('current_static_source_world_delta_max', 'original_camera_forward_delta_px_max'):
        if key in mapping:
            mapping['pre_geometry_repair_' + key] = mapping.pop(key)
    mapping['specific_I_geometry_delta'] = dict(path=str(receipt), sha256=sha(receipt), affected_physical_entities=1)
    save_json(mapping_path, mapping)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', required=True); p.add_argument('--model', required=True)
    p.add_argument('--original-model', required=True); p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='D:/CodexData/RinkSpace_Data')
    p.add_argument('--ad-id',type=int,choices=tuple(range(2,13)),default=12)
    p.add_argument('--scope-rows',help='Delivered current clear-frame prepared-row subset; complete W factors must remain intact')
    p.add_argument('--replace-export', action='store_true', help='Refresh this detached input export; never a solver checkpoint')
    a = p.parse_args()
    print(json.dumps(export(a.prepared, a.model, a.original_model, a.output, data_root=a.data_root, replace_export=a.replace_export, ad_id=a.ad_id, scope_rows=a.scope_rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
