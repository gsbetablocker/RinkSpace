"""Data-only import of mixed-source advertisement material records.

Historical source pixels remain template evidence in their original clock.
An inherited contour is an alias, never another observation or another weight.
The adapter does not measure pixels, authorize fitting, or estimate placement.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


def import_selected_parts(selection):
    """Import explicitly chosen parts with separate image and video evidence.

    This consumes a current selection, not a historical production chain. Still
    artwork never acquires programme PTS. Original open source fragments may
    reference a complete template contour without inventing their missing UV.
    """
    if selection.get('schema')!='rinkspace.print_selection.v1':
        raise ValueError('unsupported explicit print selection')
    template=deepcopy(selection['template']);paths=template['paths']
    identities=[p['id'] for p in paths]
    if len(identities)!=len(set(identities)):raise ValueError('duplicate print part identity')
    for p in paths:
        if type(p.get('closed')) is not bool:raise ValueError('explicit print closure required')
        geometry=p.get('geometry')
        if geometry is None:
            uv=p['material_uv']
            if len(uv)<2 or any(len(q)!=2 or not all(math.isfinite(float(v)) for v in q) for q in uv):
                raise ValueError('invalid selected material coordinates')
        else:
            if 'material_uv' in p:raise ValueError('analytic print geometry cannot have a polyline fallback')
            if geometry.get('kind')!='unfolded_square':raise ValueError('unsupported analytic print geometry')
            centre=geometry.get('center_uv',[]);side=geometry.get('side_over_panel_height')
            if len(centre)!=2 or not all(math.isfinite(float(v)) for v in centre):
                raise ValueError('invalid square print centre')
            if type(side) not in (float,int) or not math.isfinite(side) or side<=0 or p['closed'] is not True:
                raise ValueError('invalid square print side or closure')
        parent=p.get('parent_contour_key')
        if parent is not None and parent not in identities:raise ValueError('selected print hole parent missing')
        if p['role'].endswith('hole_boundary') and parent is None:raise ValueError('selected print hole requires its parent')
        if p.get('model_inclusion')=='reference_only':raise ValueError('reference-only detail must not enter the print model')
    observations=deepcopy(selection['observations']);sources=deepcopy(selection['sources'])
    for o in observations:
        if o['feature_id'] not in identities:raise ValueError('source observation refers to missing print part')
        if o['source_id'] not in sources:raise ValueError('missing original source identity')
        source=sources[o['source_id']]
        if source['kind']!='video_frame' or type(o['pts']) is not int or Fraction(o['time_base'])<=0:
            raise ValueError('programme observation requires its own video clock')
        if o['pts']!=source['pts'] or o['time_base']!=source['time_base']:
            raise ValueError('source and observation clocks differ')
        uv=o['source_native_uv']
        if len(uv)<2 or any(len(q)!=2 or not all(math.isfinite(float(v)) for v in q) for q in uv):
            raise ValueError('invalid original source pixels')
        if o.get('active_for_fit') is not False:raise ValueError('source import does not authorize fitting')
    result={'schema':'rinkspace.ad_asset_import.v1','advertisement_number':selection['advertisement_number'],
        'sources':sources,'templates':[template],'instances':[deepcopy(selection['primary_instance'])],
        'observations':observations,'aliases':deepcopy(selection.get('aliases',[])),
        'source_correlated_groups':deepcopy(selection.get('source_correlated_groups',[])),
        'shape_evidence':deepcopy(selection.get('shape_evidence',[])),
        'unassigned_evidence':deepcopy(selection.get('unassigned_evidence',[])),
        'provenance':deepcopy(selection['provenance']),'limitations':deepcopy(selection['limitations']),
        'fit_executed':False,'counts':{'unique_template_paths':len(paths),
            'unique_contour_observations':len(observations),'texture_template_paths':0}}
    from ..domain.print_assets import attach_paired_instance,initialize_instance
    pair=selection['pair']
    result=attach_paired_instance(result,pair['primary_instance_id'],pair['paired_instance_id'],
        pair['size_block_id'],authority=pair['authority'])
    for seed in selection['installation_initializations']:
        result=initialize_instance(result,seed['instance_id'],seed['parameters'],provenance=seed['provenance'])
        instance=next(i for i in result['instances'] if i['id']==seed['instance_id'])
        instance['source_observation_ids']=[o['id'] for o in observations if o['instance_id']==seed['instance_id']]
        if instance['source_observation_ids']:
            instance['source_status']='actual_native_source_fragments_available'
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def _pointer(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _source(record, supplement=None):
    """Supplement an explicitly selected source, never an enclosing template."""
    raw = deepcopy(record)
    supplement = supplement or {}
    aliases = {
        "sha256": ("source_sha256", "video_sha256", "source_video_sha256"),
        "pts": ("source_pts", "native_pts", "pts"),
        "time_base": ("time_base",),
        "video_path": ("source_video_path", "video_path"),
        "frame_path": ("source_frame_path", "path", "frame_path"),
        "frame_sha256": ("source_frame_sha256", "frame_sha256"),
        "native_hw": ("native_hw",),
    }
    result = {}
    for target, names in aliases.items():
        values = [x[k] for x in (record, supplement) for k in names if k in x and x[k] is not None]
        # The nested source's sha256 is the PNG hash, not its video hash.
        if target == "frame_sha256" and "source_frame_path" not in record and "path" in record:
            values += [record["sha256"]] if record.get("sha256") else []
        if values:
            if target == "time_base":
                values = [str(Fraction(v)) for v in values]
            if any(v != values[0] for v in values[1:]):
                raise ValueError(f"conflicting explicit advertisement source {target}")
            result[target] = deepcopy(values[0])
    if not re.fullmatch(r"[0-9a-f]{64}", result.get("sha256", "")):
        raise ValueError("explicit video SHA256 is required for each advertisement source")
    if type(result.get("pts")) is not int:
        raise ValueError("native integer PTS is required for each advertisement source")
    if "time_base" not in result or Fraction(result["time_base"]) <= 0:
        raise ValueError("explicit positive native time_base is required for each advertisement source")
    result["id"] = "source/" + _digest([result["sha256"], result["time_base"]])[:24]
    result["raw_record"] = raw
    result["explicit_supplement"] = deepcopy(supplement)
    return result


def _validate_contour(record, key):
    if record.get("id") != key:
        raise ValueError(f"contour key/id mismatch: {key}")
    if type(record.get("closed")) is not bool:
        raise ValueError(f"explicit contour closure is required: {key}")
    for field in ("source_native_uv", "material_uv"):
        points = record.get(field)
        if (not isinstance(points, list) or len(points) < 2 or
                any(not isinstance(p, list) or len(p) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                    for p in points)):
            raise ValueError(f"finite original {field} is required: {key}")
    if len(record["source_native_uv"]) != len(record["material_uv"]):
        raise ValueError(f"source/material vertex correspondence changed: {key}")


def import_mixed_ad_template(path: str | Path, current_source: dict) -> dict:
    """Import root historical contours plus an explicitly sourced nested set.

    ``current_source`` supplies the nested source's video SHA/time base when
    absent from its old PNG-only manifest. Its values must not conflict with
    the nested record. It never supplements or rewrites the historical source.
    Returned observations are finite contour records, inactive for fitting;
    raw uncertainty/correlation declarations are retained without fabricating
    a covariance, point independence, or an expanded approval scope.
    """
    path = Path(path).resolve()
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    number = data.get("advertisement_number")
    if type(number) is not int or number <= 0:
        raise ValueError("explicit physical advertisement_number is required")
    nested = data.get("advertisements", {}).get(str(number))
    if not isinstance(nested, dict) or nested.get("advertisement_number") != number:
        raise ValueError("explicit nested advertisement material is required")
    historical = _source(data["source"])
    current = _source(nested["source"], current_source)
    root = data.get("contours")
    children = nested.get("contours")
    if not isinstance(root, dict) or not isinstance(children, dict):
        raise ValueError("named source contour dictionaries are required")
    container = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                 "schema": data.get("schema"), "revision": data.get("revision"),
                 "model_id": data.get("model_id")}
    observations, provenance, paths, aliases = [], [], [], []
    canonical = {}
    template_id = f"print_template/Ad{number}"

    def check_physical_number(record):
        if "advertisement_id" in record and record["advertisement_id"] != number:
            raise ValueError("source contour belongs to a different physical advertisement")

    def append(key, record, source, pointer, evidence_role):
        check_physical_number(record)
        _validate_contour(record, key)
        observation_id = "ad_source/" + _digest([
            number, key, source["sha256"], source["pts"], source["time_base"],
            record["source_native_uv"], record["material_uv"], record["closed"]])[:32]
        provenance_id = "ad_import/" + _digest([container["sha256"], pointer])[:32]
        feature_id = f"Ad{number}.print.{key}"
        observations.append({
            "id": observation_id, "original_id": record["id"], "feature_id": feature_id,
            "source_id": source["id"], "source_sha256": source["sha256"],
            "pts": source["pts"], "time_base": source["time_base"],
            "source_native_uv": deepcopy(record["source_native_uv"]),
            "source_frame_path": source.get("frame_path"),
            "source_frame_sha256": source.get("frame_sha256"),
            "material_uv": deepcopy(record["material_uv"]), "closed": record["closed"],
            "role": record.get("role"), "evidence_role": evidence_role,
            "provenance_id": provenance_id, "active_for_fit": False,
            "approval_inferred": False, "new_measurement_created": False,
            "statistical_model": "not_inferred_from_polyline_vertices",
        })
        provenance.append({"id": provenance_id, "container": deepcopy(container),
                           "json_pointer": pointer, "raw_record": deepcopy(record),
                           "raw_source_record": deepcopy(source["raw_record"]),
                           "explicit_source_supplement": deepcopy(source["explicit_supplement"]),
                           "review_scope_extended": False})
        paths.append({"id": feature_id, "original_id": record["id"],
                      "template_id": template_id, "source_observation_id": observation_id,
                      "material_uv": deepcopy(record["material_uv"]),
                      "closed": record["closed"], "role": record.get("role"),
                      "parent_contour_key": record.get("parent_contour_key"),
                      "separate_inner_contour_keys": deepcopy(record.get("separate_inner_contour_keys", [])),
                      "raw_record": deepcopy(record)})
        canonical[key] = observation_id

    for key, record in root.items():
        append(key, record, historical, f"/contours/{_pointer(key)}", "historical_template_evidence")
    for key, record in children.items():
        check_physical_number(record)
        pointer = f"/advertisements/{number}/contours/{_pointer(key)}"
        if key in root:
            # The historical repair added only the physical number. No source,
            # uncertainty, closure, or revision difference may be hidden here.
            comparable = {k: v for k, v in record.items() if k != "advertisement_id"}
            original = {k: v for k, v in root[key].items() if k != "advertisement_id"}
            if comparable != original:
                raise ValueError(f"same contour ID has different content; explicit revision required: {key}")
            aliases.append({"container": deepcopy(container), "json_pointer": pointer,
                            "canonical_observation_id": canonical[key],
                            "canonical_json_pointer": f"/contours/{_pointer(key)}",
                            "source_id": historical["id"], "new_observation": False,
                            "additional_statistical_weight": False, "raw_record": deepcopy(record)})
        else:
            append(key, record, current, pointer, "current_programme_source_contour")
    keys = set(canonical)
    for item in paths:
        if item["parent_contour_key"] is not None and item["parent_contour_key"] not in keys:
            raise ValueError("internal-hole parent contour is missing")
        if any(key not in keys for key in item["separate_inner_contour_keys"]):
            raise ValueError("declared internal-hole contour is missing")
    return {"schema": "rinkspace.ad_asset_import.v1", "advertisement_number": number,
            "sources": {s["id"]: s for s in (historical, current)},
            "templates": [{"id": template_id, "advertisement_number": number, "paths": paths,
                           "display_reference_size": deepcopy(data.get('extraction', {}).get('rectified_preview_width_height')),
                           "shape_status": "legacy_coarse_source_initializer_not_accepted_artwork",
                           "source_shape_accuracy_promoted": False}],
            "instances": [{"id": f"print_instance/Ad{number}", "template_id": template_id,
                           "placement_status": "not_estimated_by_source_import"}],
            "observations": observations, "aliases": aliases, "provenance": provenance,
            "container": container, "fit_executed": False,
            "counts": {"historical_unique_contours": len(root),
                       "current_unique_contours": len(children)-len(aliases),
                       "copied_aliases": len(aliases), "unique_contour_observations": len(observations)}}


def append_source_atlas(assets: dict, path: str | Path, source: dict, *, frame_sha256: str) -> dict:
    """Bind existing native-frame glyph readouts without inventing a chart.

    An atlas may contain better resolved pixels than a historical template yet
    have no complete panel rectangle. Those pixels attach to the same named
    paths as evidence; they never replace normalized geometry by inverse model
    projection. The original material-chart initializer stays explicit.
    """
    result = deepcopy(assets)
    path = Path(path).resolve()
    payload = path.read_bytes()
    atlas = json.loads(payload.decode('utf-8-sig'))
    if atlas.get('source_model_projection_used') is not False:
        raise ValueError('source-only atlas evidence is required')
    if not re.fullmatch(r'[0-9a-f]{64}', frame_sha256):
        raise ValueError('explicit atlas native-frame SHA256 is required')
    frame = atlas['source_image']['file']
    identity = _source({'native_pts': atlas['native_pts'], 'source_frame_path': frame,
                        'source_frame_sha256': frame_sha256}, source)
    paths = {p['id']: p for t in result['templates'] for p in t['paths']}
    known = {row['id'] for row in result['observations']}
    def pixels(points):
        # JSON 1 and 1.0 describe the same native pixel. Normalize only the
        # comparison key, never the stored UV, dtype or original record.
        return tuple(tuple(float(v) for v in point) for point in points)
    def measurement_identity(row):
        return (row['feature_id'],row['source_sha256'],row['pts'],str(Fraction(row['time_base'])),
                pixels(row['source_native_uv']),row['closed'])
    known_measurements = {measurement_identity(row) for row in result['observations']}
    container = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                 'schema': atlas['schema'], 'revision': atlas.get('revision')}
    added = 0
    for index, record in enumerate(atlas['paths']):
        label = record['label']
        if label not in paths:
            raise ValueError('atlas glyph has no confirmed template identity: '+label)
        points = record['source_native_uv']
        if (type(record.get('closed')) is not bool or not isinstance(points, list) or len(points)<2
                or any(not isinstance(p,list) or len(p)!=2 or
                       any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points)):
            raise ValueError('atlas needs original finite UV and explicit topology')
        oid = 'ad_source/' + _digest([label,identity['sha256'],identity['pts'],
                                     identity['time_base'],points,record['closed']])[:32]
        signature = (label,identity['sha256'],identity['pts'],str(Fraction(identity['time_base'])),
                     pixels(points),record['closed'])
        if oid in known or signature in known_measurements:
            raise ValueError('atlas observation already imported; do not double count')
        known.add(oid)
        known_measurements.add(signature)
        pid = 'ad_import/' + _digest([container['sha256'],index])[:32]
        result['observations'].append({
            'id':oid,'original_id':label,'feature_id':label,'source_id':identity['id'],
            'source_sha256':identity['sha256'],'pts':identity['pts'],'time_base':identity['time_base'],
            'source_native_uv':deepcopy(points),'source_frame_path':frame,
            'source_frame_sha256':frame_sha256,'material_uv':None,'closed':record['closed'],
            'role':paths[label]['role'],'evidence_role':'current_programme_source_glyph',
            'provenance_id':pid,'active_for_fit':False,'approval_inferred':False,
            'new_measurement_created':False,'statistical_model':'not_inferred_from_polyline_vertices',
            'material_chart_status':'absent_no_four_corner_or_model_warp_invented'})
        result['provenance'].append({'id':pid,'container':container,'json_pointer':f'/paths/{index}',
            'raw_record':deepcopy(record),'raw_source_record':deepcopy(atlas['source_image']),
            'explicit_source_supplement':deepcopy(source),'source_extraction':{
                'threshold_green':atlas.get('threshold_green'),
                'source_model_projection_used':False},'review_scope_extended':False})
        target = paths[label]
        target.setdefault('source_observation_ids',[target['source_observation_id']]).append(oid)
        target['preferred_pixel_evidence_id'] = oid
        target['template_shape_refinement_pending'] = True
        added += 1
    exposure = {k:deepcopy(identity[k]) for k in ('pts','frame_path','frame_sha256') if k in identity}
    registry = result['sources'].setdefault(identity['id'],identity)
    registry.setdefault('exposures',[{k:deepcopy(registry[k]) for k in ('pts','frame_path','frame_sha256') if k in registry}])
    if exposure not in registry['exposures']:
        registry['exposures'].append(exposure)
    result['counts']['higher_resolution_source_contours'] = result['counts'].get('higher_resolution_source_contours',0)+added
    result['counts']['unique_contour_observations'] = len(result['observations'])
    result['counts']['unique_template_paths'] = sum(len(t['paths']) for t in result['templates'])
    result['counts']['source_native_vertices'] = sum(len(r['source_native_uv']) for r in result['observations'])
    return result


def apply_glyph_rebuild(assets: dict, rebuild_path: str | Path) -> dict:
    """Replace the template representation; keep every older source record.

    The current Ad2 selection contains letters and the panel boundary only.
    Texture records leave this current selection and remain in an explicit
    historical/excluded collection. They cannot become fit rows by accident.
    """
    result=deepcopy(assets);path=Path(rebuild_path).resolve();payload=path.read_bytes()
    rebuilt=json.loads(payload.decode('utf8'))
    number=result['advertisement_number']
    if rebuilt['advertisement_number']!=number or not rebuilt['text_content_confirmed']:
        raise ValueError('rebuild must bind the same advertisement and confirmed text')
    old_paths={p['id']:p for p in result['templates'][0]['paths']}
    texture_ids={p['id'] for p in old_paths.values() if p.get('role')=='visible_blue_ribbon_material_boundary'}
    removed=[r for r in result['observations'] if r['feature_id'] in texture_ids]
    result['excluded_source_records']=removed
    result['observations']=[r for r in result['observations'] if r['feature_id'] not in texture_ids]
    result['excluded_source_reason']='user selects printed text and advertisement outer boundary only; no texture'
    new_paths=[]
    for index,record in enumerate(rebuilt['paths']):
        if record['role'] not in ('letter_material_boundary','letter_hole_boundary','panel_outer_boundary'):
            raise ValueError('unsupported background content in current glyph template')
        source=_source(record['source']);label=record['id']
        oid='ad_source/'+_digest([label,source['sha256'],source['pts'],source['time_base'],record['source_native_uv']])[:32]
        pid='ad_import/'+_digest([hashlib.sha256(payload).hexdigest(),index])[:32]
        old=old_paths.get(label,{})
        result['observations'].append({'id':oid,'original_id':record['original_id'],
            'feature_id':label,'source_id':source['id'],'source_sha256':source['sha256'],
            'pts':source['pts'],'time_base':source['time_base'],
            'source_frame_path':source['frame_path'],'source_frame_sha256':source['frame_sha256'],
            'source_native_uv':deepcopy(record['source_native_uv']),
            'closed':record['closed'],'role':record['role'],'material_uv':None,
            'provenance_id':pid,'evidence_role':'template_shape_source',
            'active_for_fit':False,'approval_inferred':False,'new_measurement_created':record['role']!='panel_outer_boundary',
            'statistical_model':'same_contour_source_pixels_are_correlated_not_independent_vertices'})
        result['provenance'].append({'id':pid,'container':{'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()},
            'json_pointer':f'/paths/{index}','raw_record':deepcopy(record),
            'text_content_authority':'direct_user_confirmation','pixel_accuracy_user_approved':False})
        registry=result['sources'].setdefault(source['id'],source)
        exposure={k:deepcopy(source[k]) for k in ('pts','frame_path','frame_sha256')}
        registry.setdefault('exposures',[{k:deepcopy(registry[k]) for k in ('pts','frame_path','frame_sha256') if k in registry}])
        if exposure not in registry['exposures']:registry['exposures'].append(exposure)
        new_paths.append({'id':label,'original_id':record['original_id'],
            'template_id':result['templates'][0]['id'],'source_observation_id':oid,
            'source_observation_ids':list(dict.fromkeys(old.get('source_observation_ids',
                [old['source_observation_id']] if old.get('source_observation_id') else [])+[oid])),
            'preferred_pixel_evidence_id':oid,'material_uv':deepcopy(record['material_uv']),
            'closed':record['closed'],'role':record['role'],'parent_contour_key':record['parent_contour_key'],
            'template_shape_refinement_pending':False})
    template=result['templates'][0];template['paths']=new_paths;template['text']=rebuilt['text']
    template['text_content_authority']='direct_user_confirmation';template['display_reference_size']=[1536,256]
    template['shape_status']='source_rebuilt_pending_user_visual_review'
    template['shape_review']={'status':'pending_user_visual_review','text_content_confirmed':True}
    template['material_chart_status']='source_four_edge_panel_chart_initialization'
    template['rebuild']={'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()}
    result['counts'].update(unique_template_paths=len(new_paths),texture_template_paths=0,
        excluded_historical_texture_contours=len(removed),unique_contour_observations=len(result['observations']),
        source_native_vertices=sum(len(r['source_native_uv']) for r in result['observations']))
    result['current_content_selection']=['printed_text','panel_outer_boundary']
    result['limitations']=['文字内容已由用户确定；字形与字距来自清晰原片及同一全局校正。',
        '模板等待本次图审；安装位置、物理尺寸与相机尚未重新拟合。']
    return result
