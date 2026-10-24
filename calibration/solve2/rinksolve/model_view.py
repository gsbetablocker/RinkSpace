"""Read-only inspection of the exact selected model package; no solver or geometry edits."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone
from .store import StoreError


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise StoreError(f"模型当前文件不可读: {path}: {exc}", 503, "model_unavailable") from exc


def _points(value, label):
    if not isinstance(value, list):
        raise StoreError(f"{label}: 应为点数组", 503, "invalid_model_geometry")
    if any(not isinstance(p, list) or len(p) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in value):
        raise StoreError(f"{label}: 必须为有限世界XYZ", 503, "invalid_model_geometry")
    return value


def inspection_snapshot(programme, data, native_pts=None):
    binding = programme.get("inspection_model")
    if isinstance(binding, dict) and binding.get("kind") == "runtime_model":
        return runtime_snapshot(programme, data, native_pts)
    if not isinstance(binding, dict) or not binding.get("package"):
        raise StoreError("尚未选择可显示的当前模型包", 503, "model_not_selected")
    package = Path(binding["package"])
    metadata = _load(package / "initialization.json")
    scene = _load(package / "scene.json")
    raw_paths = _load(package / "paths.json")
    rig_pack = _load(package / "rig_reference.json")
    if not isinstance(raw_paths, list):
        raise StoreError("paths.json必须为数组", 503, "invalid_model_geometry")
    if rig_pack.get("coordinate_system") != scene.get("coordinate_system", {}).get("id"):
        raise StoreError("三机位与冰场坐标系不一致", 503, "coordinate_system_mismatch")
    expected_identity = metadata.get("historical_demo", {})
    actual_identity = rig_pack.get("model_source_identity", {})
    if actual_identity.get("scene_sha256") != expected_identity.get("scene", {}).get("sha256") or actual_identity.get("original_nine_shot_q_sha256") != expected_identity.get("original_q_identity"):
        raise StoreError("三机位与冰场不属于同一初值模型", 503, "model_identity_mismatch")
    paths = []
    entities = []
    ad_paths = {}
    for row in raw_paths:
        points = _points(row.get("xyz"), row.get("id"))
        identity = row["id"]
        record = dict(id=identity, label=row.get("label", identity), kind=row.get("group", "unknown"),
                      points=points, closed=bool(row.get("closed")), owner="scene", evidence=row.get("provenance"),
                      initial_geometry_only=True, constraint_semantics_unassigned=True,
                      samples_are_physical_landmarks=False, geometry_status=row.get("geometry_status"))
        paths.append(record)
        if row.get("group") not in {"grid", "advertisement", "advertisement_print"}:
            entities.append(dict(id=identity, label=record["label"], kind=record["kind"], status="initial_geometry_only", evidence=record["evidence"], observations_status="not_imported"))
        if row.get("group") in {"advertisement", "advertisement_print"}:
            ad_id = identity.split(":", 1)[0]
            ad_paths.setdefault(ad_id, []).append(record)
    catalogue = {}
    inventory_path = programme.get("ad_inventory")
    if inventory_path and Path(inventory_path).is_file():
        inventory = _load(inventory_path)
        catalogue = {row["id"]: row for row in inventory.get("advertisements", [])}
    artwork_rows = []
    artwork_path = programme.get("artwork_collection")
    if artwork_path and Path(artwork_path).is_file():
        artwork_rows = _load(artwork_path).get("templates", [])
    ads = []
    source_ads = {"Ad"+str(a["user_number"]): a for a in scene.get("advertisements", [])}
    for ad_id in sorted(set(ad_paths) | set(source_ads) | set(programme.get("advertisement_ids", [])), key=lambda s: int(s[2:])):
        a = source_ads.get(ad_id, {})
        identity = catalogue.get(ad_id, {})
        matching = ad_paths.get(ad_id, [])
        footprint = next((p["points"] for p in matching if "bottom_edge" in p["id"]), [])
        if not footprint:
            footprint = next((p["points"] for p in matching if len(p["points"]) > 2), [])
        ads.append(dict(id=ad_id, label=ad_id + (" · " + identity["observed_identity"] if identity.get("observed_identity") else ""), footprint=footprint, path_ids=[p["id"] for p in matching],
                        template_id=None, installation_id=ad_id, status="initial_geometry_only" if matching else "geometry_missing", source="scene.json / paths.json",
                        identity_status=identity.get("identity_status", "not_imported"), observed_identity=identity.get("observed_identity"), evidence=identity.get("evidence", []), artwork_components=identity.get("artwork_components", []), 
                        parameters=dict(surface="continuous_board_unfolded", corners_xyz=a.get("corners_xyz_scene_units"),
                                        bottom_z_scene_units=a.get("bottom_z_scene_units"), top_z_scene_units=a.get("top_z_scene_units")),
                        observations_status="not_imported", template_status="component_geometry_available_mapping_pending" if any(ad_id in t.get("ads", []) for t in artwork_rows) else "HD_component_missing_or_in_progress",
                        template_candidates=[dict(id=t["id"], path=t["path"], kind=t.get("kind"), match=t.get("match"), named_components=t.get("named_components")) for t in artwork_rows if ad_id in t.get("ads", [])]))
    rigs = rig_pack.get("rigs", [])
    if len(rigs) != 3 or len({r.get("id") for r in rigs}) != 3:
        raise StoreError("需三个独立机位的显示记录", 503, "rig_registry_incomplete")
    xy = [p[:2] for row in paths for p in row["points"]]
    for rig in rigs:
        position = rig.get("position_xyz")
        if position is not None:
            _points([position], rig["id"])
            xy.append(position[:2])
        if rig.get("pivot_xyz") is not None:
            _points([rig["pivot_xyz"]], rig["id"]+" pivot")
            xy.append(rig["pivot_xyz"][:2])
    if not xy:
        raise StoreError("当前模型没有可显示的几何", 503, "empty_model")
    bounds = [min(p[0] for p in xy), min(p[1] for p in xy), max(p[0] for p in xy), max(p[1] for p in xy)]
    files = [package / n for n in ("initialization.json", "scene.json", "paths.json", "rig_reference.json")]
    for extra_path in (inventory_path, artwork_path):
        if extra_path and Path(extra_path).is_file():
            files.append(Path(extra_path))
    revision = binding.get("revision", 1)
    version = ":".join(str(p.stat().st_mtime_ns) for p in files)
    return dict(schema="rinksolve.inspection_model.v1", revision=revision, version=version,
                model_id=binding.get("id", "demo_s01_initialization"), name=binding.get("label"),
                status="初始化参照；新求解尚未运行", unit="conditional_scene_units", verified_metre_scale=False,
                coordinate_system=scene.get("coordinate_system"), top_view_axes=dict(horizontal="x",vertical="y",up="z",xy_indices=[0,1],y_positive="up"),
                bounds=bounds, rink_bounds=scene.get("parameters",{}).get("bounds"), paths=paths, ads=ads, entities=entities, rigs=rigs,
                source=str(package), source_model_id=scene.get("model_id"), state_id=None, engine_running=False,
                updated_at=datetime.fromtimestamp(max(p.stat().st_mtime for p in files),timezone.utc).isoformat(),
                warnings=["当前显示成功demo的独立初始化包，不代表新物理模型或S01验收通过。", "圈线与广告路径完整显示；采样顶点尚未被定义为物理定位点。", "三机位是同一demo坐标系内指定源PTS的历史光心；固定转轴尚未导入，不能把光心当云台中心。", "广告图案模板与观测归属正在重整；缺失几何不会摆放示意。"],
                missing_ad_ids=["Ad"+str(n) for n in range(1,27) if "Ad"+str(n) not in ad_paths],
                initialization=dict(package=str(package),camera_contract=metadata.get("camera_contract"),s01_scope=metadata.get("clock")))


def artwork_info(programme):
    rows=[]; proposals={}
    if programme.get('artwork_collection'):
        rows.extend(_load(programme['artwork_collection']).get('templates',()))
    if programme.get('artwork_layouts'):
        layouts=_load(programme['artwork_layouts'])
        proposals={p['installation_id']:p for p in layouts.get('composite_proposals',())}
        links={}
        for ad,p in proposals.items():
            for component in p.get('components',()):
                for ref in ([component['template_id']] if component.get('template_id') else [])+list(component.get('candidate_template_ids',())):
                    links.setdefault(ref,set()).add(ad)
        for t in layouts.get('new_templates',()):
            rows.append(dict(t,ads=sorted(links.get(t['id'],())),kind=t.get('extent'),named_components=t.get('counts')))
    return rows,proposals


def runtime_snapshot(programme,data,native_pts=None):
    from .runtime_state import model_store
    binding=programme['inspection_model'];store=model_store(binding['package'],data)
    current,model,current_location=store.inspection_record(native_pts)
    paths=[];entities=[];ad_paths={}
    for curve in current['curves']:
        points=_points(curve['xyz'],curve['id'])
        group=curve.get('group','ice')
        item=dict(id=curve['id'],label=curve['label'],kind=group,points=points,
            closed=curve['kind']=='ice_circle' or len(points)>2 and points[0]==points[-1],
            owner=curve['entity_id'],evidence=curve.get('source_path_id'),
            initial_geometry_only=curve.get('initial_geometry_only',True),
            samples_are_physical_landmarks=False,geometry_status=curve.get('semantic_status'),
            initialization_source=curve.get('initialization_source'),
            parameter_state_id=current['parameter_state_id'])
        paths.append(item)
        if group in ('advertisement','advertisement_print'):
            match=re.search(r'Ad([0-9]+)(?![0-9])',curve['label'])
            if match:ad_paths.setdefault('Ad'+match[1],[]).append(item)
        else:entities.append(dict(id=item['id'],label=item['label'],kind=group,status='initial_geometry_only',
                                 evidence=item['evidence'],parameter_state_id=current['parameter_state_id']))
    for row in current['grid']:
        paths.append(dict(id=row['id'],label=row['id'],kind='grid',points=_points(row['xyz'],row['id']),closed=False,
            owner='scene',is_observation=False,geometry_status=row['role'],parameter_state_id=current['parameter_state_id']))
    for suffix in ('bottom','top'):
        points=_points(current['board']['xyz_'+suffix],'board '+suffix)
        paths.append(dict(id='current_board_'+suffix,label='围挡'+suffix,kind='board',points=points,closed=True,
                          owner='scene',parameter_state_id=current['parameter_state_id']))
    inventory=_load(programme['ad_inventory']) if programme.get('ad_inventory') else {}
    identities={r['id']:r for r in inventory.get('advertisements',())}
    artworks,proposals=artwork_info(programme)
    actual_templates={}
    for curve in model.scene.curves:
        match=re.search(r'Ad([0-9]+)(?![0-9])',curve.label)
        if not match:continue
        record=actual_templates.setdefault('Ad'+match[1],{'geometry_curve_count':0,'source_bound_curve_count':0,'source_template_ids':set()})
        record['geometry_curve_count']+=1
        definition=model.scene.resolved_definition(curve,model.state)
        template=definition.get('source_template_id')
        if template:
            record['source_bound_curve_count']+=1
            record['source_template_ids'].add(str(template))
    ads=[]
    for installation in current['installations']:
        ad_id=installation['id'].split('/')[-1];identity=identities.get(ad_id,{})
        matching=ad_paths.get(ad_id,[])
        boundary=installation.get('boundary_xyz')
        if boundary:
            record=dict(id=installation['id']+'/boundary',label=ad_id+' '+installation['extent_role'],kind='advertisement',
                points=_points(boundary,ad_id),closed=True,owner=installation['id'],initial_geometry_only=True,
                extent_role=installation['extent_role'],parameter_state_id=current['parameter_state_id'])
            paths.append(record);matching=matching+[record]
        footprint=boundary or next((p['points'] for p in matching if p['points']),[])
        candidates=[t for t in artworks if ad_id in t.get('ads',())]
        binding_info=actual_templates.get(ad_id,{'geometry_curve_count':0,'source_bound_curve_count':0,'source_template_ids':set()}).copy()
        binding_info['source_template_ids']=sorted(binding_info['source_template_ids'])
        objects=[a for a in model.artworks if re.search(r'Ad([0-9]+)(?![0-9])',a.id) and re.search(r'Ad([0-9]+)(?![0-9])',a.id).group(0)==ad_id]
        binding_info['object_source_and_layout_contract_registered']=any(bool((a.template or a.source_templates) and a.layout) for a in objects)
        binding_info['pixel_entity_associations_validated_by_this_record']=False
        template_status=('source_template_and_object_contract_bound' if binding_info['object_source_and_layout_contract_registered'] else 'curve_template_bound_object_contract_pending') if binding_info['source_template_ids'] else 'component_geometry_available_mapping_pending' if candidates else 'HD_component_missing_or_in_progress'
        ads.append(dict(id=ad_id,label=ad_id+((' · '+identity['observed_identity']) if identity.get('observed_identity') else ''),
            footprint=footprint,path_ids=[p['id'] for p in matching],installation_id=installation['id'],
            extent_role=installation['extent_role'],template_id=installation.get('template_id'),
            status='initial_geometry_only' if footprint else 'geometry_missing',identity_status=identity.get('identity_status','not_imported'),
            observed_identity=identity.get('observed_identity'),evidence=identity.get('evidence',[]),
            parameters=installation,parameter_state_id=current['parameter_state_id'],
            template_status=template_status,actual_template_binding=binding_info,
            template_candidates=[dict(id=t['id'],path=t['path'],kind=t.get('kind'),match=t.get('match'),named_components=t.get('named_components')) for t in candidates],
            static_layout_proposal=proposals.get(ad_id)))
    rigs=[]
    for original in current['rigs']:
        rig=dict(original)
        rig.update(position_xyz=original.get('inspection_position_world'),pivot_xyz=original.get('pivot_world'),
            direction_xyz=original.get('inspection_direction_world'),position_kind=original.get('inspection_position_kind'),
            status='current_initialization_pose' if original['available'] else 'historical_demo_initial_reference_not_pivot',
            source_pts=original.get('inspection_source_pts'),source_seconds=original.get('inspection_source_pts',0)/12800)
        rigs.append(rig)
    xy=[p[:2] for row in paths for p in row['points']]+[r['position_xyz'][:2] for r in rigs if r.get('position_xyz')]
    bounds=[min(p[0] for p in xy),min(p[1] for p in xy),max(p[0] for p in xy),max(p[1] for p in xy)]
    board=current['board']['xyz_bottom'];rink_bounds=[min(p[0] for p in board),min(p[1] for p in board),max(p[0] for p in board),max(p[1] for p in board)]
    from .schema import SharedCurvePointPosition,to_json_value
    point_definitions={point.id:point for point in model.landmarks}
    landmark_registry=[]
    for point in current.get('landmarks',()):
        definition=point_definitions[point['id']]
        shared=isinstance(definition.position,SharedCurvePointPosition)
        entry=dict(point,kind='fixed_material_landmark',status='当前固定实体的候选世界位置',
            parameter_state_id=current['parameter_state_id'],
            position_definition=to_json_value(definition.position),
            position_shared_across_all_exposures=True,per_exposure_station=False,
            coordinates_are_verified_measurements=False,
            position_semantics='固定材料身份，全曝光共用一个待估世界位置；当前数值可纠正。' if shared else '同一源图实体角点，经共同对象安装和场地派生；不是采样顶点或逐帧自由点。')
        if shared:
            entry['parameter_block_id']=definition.position.station_variable_block_id
            entry['shared_station_value']=float(model.state.get(definition.position.station_variable_block_id)[0])
            entry['short_label']='C-'+point['id'].rsplit('_',1)[-1]
            entry['source_query']='cross_'+point['id'].rsplit('_',1)[-1]
        else:
            entry['short_label']=point['id'].split('/')[-1]
            entry['source_query']='Ad7'
        landmark_registry.append(entry)
    metadata_paths=[current_location/'model.json',current_location/'parameters.npz']
    version=current['parameter_sha256']+':'+str(current['model_revision'])+':'+str(current['selected_exposure']['native_pts'])
    return dict(schema='rinksolve.inspection_model.v1',revision=current['model_revision'],version=version,
        model_id=current['model_id'],name=binding.get('label','当前共同物理模型'),
        status='本来的第一镜头：当时参数原样保留' if binding.get('restored_original') else '当前共同物理候选；S01整段待验收',read_only=bool(binding.get('read_only')),restored_original=bool(binding.get('restored_original')),unit='scene_unit_L1',verified_metre_scale=False,
        coordinate_system=dict(id=current['coordinate_system_id'],unit='scene_unit_L1',x_axis='用户俯视图向右',y_axis='用户俯视图向上',z_axis='冰面上方'),
        top_view_axes=dict(horizontal='x',vertical='y',up='z',xy_indices=[0,1],y_positive='up'),bounds=bounds,rink_bounds=rink_bounds,
        paths=paths,ads=ads,entities=entities,rigs=rigs,landmarks=landmark_registry,source=str(current_location),
        state_id=current['parameter_state_id'],parameter_revision=current['parameter_revision'],parameter_sha256=current['parameter_sha256'],
        engine_running=None,numerical_run_status_source='/api/state',selected_exposure=current['selected_exposure'],geometry_family=current['geometry_family'],
        shot_clock=dict(first_pts=model.shots[0].first_native_pts,last_pts=model.shots[0].last_native_pts,step_pts=model.shots[0].step_native_pts,time_base_num=model.shots[0].time_base_num,time_base_den=model.shots[0].time_base_den),
        updated_at=datetime.fromtimestamp(max(p.stat().st_mtime for p in metadata_paths),timezone.utc).isoformat(),
        warnings=['冰场和主相机均为当时第一镜头原参数，未做共同场地转换。','原第一镜头不含另外两机位的位置，不从后来候选补入；后续镜头锁定。'] if binding.get('restored_original') else ['所有线、图案、围挡与主机位来自同一当前参数状态。','侧/端机位为明确初值光心参考，固定安装待估；S02–S09锁定。'],
        missing_ad_ids=[a['id'] for a in ads if not a['footprint']],missing_items=current['missing_items'],
        provenance=current['provenance'],initialization=dict(inspection_contract=current['inspection_contract'],acceptance=False))
