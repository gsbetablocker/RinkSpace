"""Adapt existing source guides to current native geometry, without changing either."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def file_ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


class NativeSourceGeometry:
    """Read-only native authority for identity, primitive and measurement semantics.

    Source UV, neural shapes and old world coordinates never modify this scene.
    Shape fitting in the reader is an image-evidence diagnostic, not a world model.
    """
    def __init__(self, scene_path, library_path):
        from .core.native import NativeCore
        from .review.scene import sample_scene
        from .domain.scene_assets import read_scene_assets
        self.scene_path = Path(scene_path).resolve()
        self.scene = json.loads(self.scene_path.read_text(encoding='utf-8-sig'))
        self.refs = [file_ref(self.scene_path), file_ref(library_path)]
        self.core = NativeCore(library_path)
        self.samples = sample_scene(self.scene, self.core, self.scene_path)
        self.paths = {}
        self.ambiguous_aliases = set()
        declared = {p['id']:p for p in self.scene['paths']}
        self.scene_assets = read_scene_assets(self.scene, self.scene_path)
        from .domain.print_assets import resolve_instance, resolve_print_curve
        retained = {p['id']: dict(p) for p in self.samples['cuts']}
        for reference, assets in self.scene_assets:
            for instance_id in reference.get('instance_ids', [reference['instance_id']]):
                instance, template, size, placement = resolve_instance(assets, instance_id)
                for path in template['paths']:
                    bound = resolve_print_curve(assets, instance_id, path['id'])
                    retained.setdefault(bound.path_id, dict(id=bound.path_id, label=bound.path_id,
                        entity_id=instance.get('entity_id', 'entity/'+instance_id.rsplit('/',1)[-1]),
                        kind='print_curve', layer='advertisement', measurement='contour',
                        **bound.as_record(), geometry_localized=size['values'] is not None and placement['values'] is not None))
        for cut in retained.values():
            row = {**cut, **declared.get(cut['id'], {})}
            keys = [row['id'], row['label']]
            if row['layer'] == 'advertisement':
                instance = row['instance_id'].rsplit('/', 1)[-1]
                template = row['template_path_id']
                # The installed instance supplies identity, including equal-content pairs.
                if template.startswith(row['template_id'].rsplit('/', 1)[-1]+'.'):
                    keys.append(instance+'.'+template.split('.', 1)[-1])
            for key in keys:
                if key in self.ambiguous_aliases:
                    continue
                if key in self.paths and self.paths[key]['id'] != row['id']:
                    self.ambiguous_aliases.add(key)
                    self.paths.pop(key)
                else:
                    self.paths[key] = row
        # read_scene_assets authenticates the exact current asset SHA references.
        self.current_paths = {row['id']: row for row in self.paths.values()}
        for reference, _ in self.scene_assets:
            asset_path = Path(reference['path'])
            if not asset_path.is_absolute():
                asset_path = self.scene_path.parent / asset_path
            self.refs.append(file_ref(asset_path))

    def assert_unchanged(self):
        for ref in self.refs:
            if file_ref(ref['path'])['sha256'] != ref['sha256']:
                raise ValueError('Current native geometry changed during source reading: '+ref['path'])

    def lookup(self, identity):
        if identity in self.paths:
            return self.paths[identity]
        if 'ice.'+str(identity) in self.paths:
            return self.paths['ice.'+str(identity)]
        # Historical S02 packets explicitly named the same blue half-ellipse
        # perimeter with :boundary. This is a spelling adapter to an existing
        # contour, never a new path or a conversion from an edge to a midline.
        key=str(identity).removeprefix('ice.')
        if key in ('goal_crease_left:boundary','goal_crease_right:boundary'):
            row=self.paths.get('ice.'+key.removesuffix(':boundary'))
            if row and row.get('measurement')=='contour' and row.get('role')=='physical_half_ellipse_boundary':
                return row
        # Legacy source packets named an edge; the installed template may
        # store it as edge/path0. Resolve only a single current native path.
        candidates = {row['id']:row for key,row in self.paths.items()
                      if key.startswith(str(identity)+'/path')}
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        # Old board segment names locate portions of the current continuous board.
        if str(identity).startswith(('board_top.', 'board.top.')):
            return self.paths.get('board.top')
        if str(identity).startswith(('board_contact.', 'board.contact.')):
            return self.paths.get('board.bottom')
        return None

    def bind(self, guides):
        self.assert_unchanged()
        result, unresolved, conflicts = [], [], []
        for incoming in guides:
            guide = dict(incoming)
            row = self.lookup(guide['id'])
            if row is None:
                unresolved.append(dict(id=guide['id'], reason='no_unambiguous_current_native_path',
                                       original_source_preserved=True))
                continue
            projection = guide.get('model_search_projection')
            if projection and projection.get('model_revision') != self.refs[0]['sha256']:
                raise ValueError('Historical geometry projection cannot replace current native search geometry')
            if row['layer'] == 'ice':
                primitive = {'ice_segment':'line', 'ice_circle':'circle', 'ice_ellipse':'conic'}.get(row['kind'], 'free_curve')
                role = ('painted_stroke_center_axis' if row['measurement'] == 'centreline'
                        else 'filled_disc_boundary')
                guide['kind'] = 'ice_paint'
            elif row['layer'] == 'board':
                primitive = 'free_curve'
                bottom = row['label'] == 'board.bottom'
                guide['kind'] = 'ice_contact' if bottom else 'board_top'
                role = 'ice_board_contact' if bottom else 'board_top_edge'
            else:
                primitive = 'free_curve'
                guide['kind'] = 'ad_boundary'
                role = ('printed_pattern_internal_hole_contour' if 'hole' in str(row.get('role'))
                        else 'advertisement_visible_boundary')
            old_role = guide.get('physical_role')
            old_center = old_role == 'painted_stroke_center_axis'
            if old_role and old_center != (role == 'painted_stroke_center_axis'):
                # Retain original evidence, but do not relabel old boundary UV as a centreline.
                guide['historical_source_definition'] = dict(physical_role=old_role,
                    primitive=guide.get('primitive'), origin=guide.get('origin'))
                guide['origin'] = 'source_annotation_search_hint'
                conflicts.append(dict(id=guide['id'], historical_role=old_role, current_role=role,
                    old_pixels_relabelled=False, current_native_geometry_wins=True))
            guide.update(primitive=primitive, physical_role=role,
                native_geometry=dict(scene=self.refs[0], path_id=row['id'], entity_id=row['entity_id'],
                    measurement=row.get('measurement', 'contour'), native_kind=row['kind']),
                world_geometry_optimized=False)
            if role == 'painted_stroke_center_axis':
                guide.setdefault('signal', 'dark_band')
            result.append(guide)
        return result, dict(scene=self.refs[0], library=self.refs[1],
            model_path_count=len(self.samples['cuts']), unresolved=unresolved, conflicts=conflicts,
            native_geometry_authoritative=True, geometry_modified=False,
            source_UV_rewritten=False, fitted_image_shapes_modify_world_geometry=False)

    def ice_inventory(self):
        """Measurement definitions come only from the loaded native paths."""
        result = []
        for row in self.current_paths.values():
            if row['layer'] != 'ice':
                continue
            name = row['label'].removeprefix('ice.')
            feature, _, suffix = name.partition(':')
            # Source-material colour is a search channel, not world geometry.
            signal = 'blue' if feature in ('B1', 'B2') else 'red' if feature in ('C', 'C0') or feature.startswith(('goal_line', 'goal_trapezoid', 'faceoff')) else 'dark'
            result.append(dict(curve_id=name, feature_id=feature, world_curve_role=suffix,
                operator='paint_center' if row['measurement']=='centreline' else 'material_interface',
                material_signal=signal, material_transition=None, native_path_id=row['id'],
                native_kind=row['kind'], measurement=row['measurement'], source_role_authority=self.refs[0],
                physical_identity_confirmed_by_navigation=False))
        return result

    def check_output(self, guides, guided):
        """Only the scene's retained paths may reach the named point output."""
        if len(guides) != len(guided['readings']):
            raise ValueError('Native output lost its per-guide correspondence')
        paths = []
        for guide, reading in zip(guides, guided['readings']):
            definition = self.lookup(guide['id'])
            bound = guide.get('native_geometry', {})
            if definition is None or bound.get('path_id') != definition['id']:
                raise ValueError('Output is not bound to a retained current native path: '+guide['id'])
            if bound.get('scene') != self.refs[0]:
                raise ValueError('Output refers to another native scene revision')
            reading['native_geometry'] = dict(bound)
            reading['native_path_id'] = definition['id']
            reading['native_measurement'] = definition.get('measurement', 'contour')
            if definition['layer'] == 'advertisement':
                reading['print_binding'] = {key:definition.get(key) for key in
                    ('instance_id','template_id','template_path_id','role','parent_contour_key')}
            paths.append(definition['id'])
        return dict(retained_current_native_paths_only=True, scene=self.refs[0],
            emitted_path_ids=sorted(set(paths)), old_detail_geometry_imported=False)


def accepted_packet_guides(packet, binding_record, acceptance, *, packet_path, binding_path, acceptance_path,
                           source_binding, search_radius_px=18., programme_scope=None):
    """Reuse exact accepted source UV and groups; historical world geometry is ignored."""
    from .vision_geometry import source_annotations_to_guides
    pref, bref, aref = map(file_ref, (packet_path, binding_path, acceptance_path))
    source = packet['source']
    stamp = source_binding['source_pts']
    if (type(stamp) is not int or packet['source_pts'] != stamp
            or source['source_video_sha256'] != source_binding['source_sha256']
            or source['time_base'] != source_binding['time_base']):
        raise ValueError('Accepted source packet belongs to another native exposure')
    if programme_scope is not None:
        from .source_binding import canonical_source_binding
        from .vision_geometry import _same_frame
        original = dict(source_sha256=source['source_video_sha256'], source_pts=packet['source_pts'],
            time_base=source['time_base'], native_hw=source['native_hw'],
            shot_id=source.get('shot_id', packet.get('shot_id', source_binding['shot_id'])))
        for key in ('event_id', 'rig_id', 'installation_id'):
            if key in source or key in packet:
                original[key] = source.get(key, packet.get(key))
        adapted = canonical_source_binding(original, programme_scope, allow_legacy_event_installation=True)
        _same_frame(source_binding, adapted)
    elif source_binding.get('identity_schema') == 'rinkspace.source_installation.v2':
        raise ValueError('Legacy source acceptance requires its registered installation mapping')
    if acceptance.get('accepted') is not True:
        raise ValueError('Explicit saved source acceptance is required for this adapter')
    rows = [r for r in acceptance['frames'] if r['native_pts'] == stamp]
    if (len(rows) != 1 or rows[0]['packet_sha256'] != pref['sha256']
            or rows[0]['binding_sha256'] != bref['sha256']
            or binding_record['source_packet']['sha256'] != pref['sha256']):
        raise ValueError('Current packet/binding is not covered by this saved acceptance')
    groups = {}
    for row in binding_record['observations']:
        groups.setdefault((row['source_feature_id'], row['source_segment_index']), set()).add(row['dependence_group'])
    annotations = []
    feature_ids = {f["id"] for f in packet["features"]}
    for feature in packet['features']:
        identity = feature['id']
        ice = identity.startswith('ice.')
        kind = ('ice_paint' if ice else 'ice_contact' if identity.startswith('board_contact') else
                'board_top' if identity.startswith('board_top') else 'ad_boundary')
        operator = feature.get('measurement_operator')
        role = ('painted_stroke_center_axis' if operator == 'paint_center' else
                'filled_disc_boundary' if ice else 'ice_board_contact' if kind == 'ice_contact' else
                'board_top_edge' if kind == 'board_top' else 'advertisement_visible_boundary')
        points, segments, point_groups = [], [], []
        for si, segment in enumerate(feature['segments']):
            points.extend(deepcopy(segment['uv']))
            segments.extend([si]*len(segment['uv']))
            point_groups.extend([sorted(groups.get((identity,si), set()))]*len(segment['uv']))
        if not points:
            continue
        annotation = dict(feature_id=identity, group_id=feature.get('source_evidence_group_id', identity),
            kind=kind, primitive='free_curve', physical_role=role, uv=points, segment_ids=segments,
            source_binding=source_binding, state='user_confirmed_visual_supervision',
            authority=dict(kind='user_confirmed_visual_supervision',reference=aref['path'],
                human_authority=True, acceptance=aref, packet=pref, binding=bref,
                           scope='exact_saved_source_packet_and_binding'),
            search_radius_px=search_radius_px)
        if role == 'painted_stroke_center_axis':
            annotation['signal'] = {'dark':'dark_band','red':'red_excess','blue':'blue_excess'}.get(
                feature.get('material_signal'), 'dark_band')
        if kind in ('board_top', 'ice_contact'):
            suffix = identity.split('.', 1)[-1]
            counterpart = ('board_contact.' if kind == 'board_top' else 'board_top.')+suffix
            annotation['source_board_view'] = ('front_face' if counterpart in feature_ids else
                feature.get('source_board_view', 'unresolved'))
        parts = [annotation]
        shape = feature.get('geometric_interpretation', {})
        if ice and role == 'filled_disc_boundary' and shape.get('kind') == 'projected_physical_circle_ellipse':
            # This is the accepted IMAGE-curve interpretation, never its old world
            # centre/radii. It supplies branch navigation only, not new model geometry.
            center = np.asarray(shape['center_uv'], float)
            uv = np.asarray(points, float); seg = np.asarray(segments)
            normal = np.zeros_like(uv)
            for si in np.unique(seg):
                ids = np.flatnonzero(seg==si)
                if len(ids)>1:
                    tangent = np.gradient(uv[ids], axis=0)
                    normal[ids] = np.c_[-tangent[:,1],tangent[:,0]]
                else:
                    normal[ids] = uv[ids]-center
            # Orient the search normals consistently downward. The existing
            # circle organiser uses opposite signed material transitions on the two arcs.
            normal[normal[:,1]<0] *= -1
            parts=[]
            for branch,keep in (('upper',uv[:,1]<center[1]),('lower',uv[:,1]>=center[1])):
                ids=np.flatnonzero(keep)
                if not len(ids): continue
                original_transition = feature.get('material_transition')
                outward_lightens = original_transition=='dark_to_light'
                polarity = ('lightening_along_normal' if (branch=='lower')==outward_lightens
                            else 'darkening_along_normal')
                part={**annotation,'uv':uv[ids].tolist(),'segment_ids':seg[ids].tolist(),
                    'normals':normal[ids].tolist(),'primitive':'circle','visible_arc_branch':branch,
                    'rgb_transition_polarity':polarity,'material_outward_transition':original_transition,
                    'transition_authority':annotation['authority'],'original_source_point_indices':ids.tolist()}
                if original_transition not in ('dark_to_light','light_to_dark'):
                    part.pop('rgb_transition_polarity')
                parts.append(part)
        converted = source_annotations_to_guides(parts,source_binding=source_binding,
                                                annotation_source_binding=source_binding)
        for guide in converted['guides']:
            original = next(p for p in parts if p.get('visible_arc_branch') == guide.get('visible_arc_branch'))
            ids = original.get('original_source_point_indices', list(range(len(points))))
            guide['original_source_point_indices'] = ids
            if original.get('material_outward_transition'):
                guide['material_outward_transition'] = original['material_outward_transition']
            guide['original_dependence_groups_per_point'] = [point_groups[i] for i in ids]
            if 'source_board_view' in original:
                guide['source_board_view'] = original['source_board_view']
            guide['accepted_source_packet'] = pref
            guide['historical_world_geometry_imported'] = False
            annotations.append(guide)
    return annotations, dict(acceptance=aref, packet=pref, binding=bref,
        accepted_original_point_count=packet['point_count'], acceptance_scope_unchanged=True,
        original_UV_and_groups_preserved=True, current_model_accuracy_accepted=False)
