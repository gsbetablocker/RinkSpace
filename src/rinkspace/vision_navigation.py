"""Current native motion projection supplies search locations, never observations.

The renderer and the source reader share native scene sampling and the implicit
rolling-shutter camera evaluator.  A camera prior says where to look; only the
reader may emit current RGB samples, and projection supplies no identity proof.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from .compat.d78_entities import sample_entities
from .domain.runtime_rigs import _camera_data
from .domain.scene_assets import scene_content_identity
from .session.camera_exposure import exposure_controls
from .vision_native import file_ref


PROJECTION_TAG = 'current_native_motion_projection'


def _relative(owner, reference):
    path = Path(reference)
    return (path if path.is_absolute() else owner.parent/path).resolve(strict=True)


def _selected_indices(uv, valid, spacing, maximum_gap):
    """Keep exact native projections, separating clipped/invalid path intervals."""
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero((np.diff(indices) != 1) |
        (np.linalg.norm(np.diff(uv[indices], axis=0), axis=1) > maximum_gap))+1
    kept, segments, segment_id = [], [], 0
    for run in np.split(indices, breaks):
        if len(run) < 2:
            continue
        chosen = [int(run[0])]
        for index in run[1:]:
            if np.linalg.norm(uv[index]-uv[chosen[-1]]) >= spacing:
                chosen.append(int(index))
        if chosen[-1] != run[-1]:
            chosen.append(int(run[-1]))
        # A collapsed projected path has no source normal to search along.
        if np.linalg.norm(uv[chosen]-uv[chosen[0]], axis=1).max() < 1e-6:
            continue
        segments.extend([segment_id]*len(chosen))
        kept.extend(chosen)
        segment_id += 1
    return np.asarray(kept, dtype=np.int64), np.asarray(segments, dtype=np.int64)


class NativeNavigation:
    """Read-only seed-set adapter; project one explicitly bound native exposure.

    ``seed_set_path`` is the portable navigation.json exported by the existing
    joint-camera importer.  Camera/controls hashes, source SHA, clocks, dimensions,
    shot domain and every current scene asset are checked, not inferred from names.
    """
    def __init__(self, geometry, seed_set_path, *, scope=None,
                 search_radius_px=24., sample_spacing_px=8.):
        self.geometry = geometry
        self.seed_set_path = Path(seed_set_path).resolve(strict=True)
        self.seed_set = json.loads(self.seed_set_path.read_text(encoding='utf-8-sig'))
        if self.seed_set.get('schema') != 'rinkspace.navigation_seed_set.v1':
            raise ValueError('Navigation requires an exported navigation_seed_set.v1')
        self.radius, self.spacing = float(search_radius_px), float(sample_spacing_px)
        if not np.isfinite([self.radius, self.spacing]).all() or min(self.radius, self.spacing) <= 0:
            raise ValueError('Navigation search radius and sample spacing must be positive and finite')
        self.source = self.seed_set['source']
        clock = self.seed_set['source_clock']
        self.clock = Fraction(clock['time_base_num'], clock['time_base_den'])
        self.step = clock['native_pts_step']
        if type(self.step) is not int or self.step <= 0:
            raise ValueError('Navigation needs a positive integer native PTS step')
        self.hw = [self.source['height'], self.source['width']]
        if any(type(v) is not int or v <= 1 for v in self.hw):
            raise ValueError('Navigation requires original native image dimensions')
        if scope is not None:
            if (any(scope['source'][k] != self.source[k] for k in ('id', 'sha256', 'width', 'height'))
                    or Fraction(scope['source_clock']['time_base_num'], scope['source_clock']['time_base_den']) != self.clock
                    or scope['source_clock']['native_pts_step'] != self.step
                    or any(scope[k] != self.seed_set[k] for k in ('event_id', 'programme_id'))):
                raise ValueError('Navigation source/event/clock/image metadata differs from programme scope')
        binding = self.seed_set.get('source_binding', {})
        self._check_saved_binding(binding)
        self.refs = [file_ref(self.seed_set_path)]
        self.cameras = {}
        for row in self.seed_set['cameras']:
            shot = row['shot_id']
            if shot in self.cameras:
                raise ValueError('Duplicate navigation shot: '+shot)
            path = _relative(self.seed_set_path, row['camera_ref'])
            camera, _, control_path, _, controls = _camera_data(path)
            cref, kref = file_ref(path), file_ref(control_path)
            if (cref['sha256'] != row['camera_sha256'] or kref['sha256'] != row['controls_sha256']
                    or control_path != _relative(self.seed_set_path, row['controls_ref'])):
                raise ValueError('Navigation camera/controls identity differs from the exported seed set')
            if (camera['shot_id'] != shot or camera['source_id'] != self.source['id']
                    or Fraction(camera['time_base_num'], camera['time_base_den']) != self.clock
                    or Fraction(camera['time_base']) != self.clock
                    or [camera['image_height'], camera['image_width']] != self.hw):
                raise ValueError('Navigation camera shot/source/clock/image metadata mismatch')
            origin = camera.get('origin', {})
            claims = [v for v in (camera.get('source_sha256'), origin.get('source_sha256'),
                origin.get('source_binding', {}).get('source_sha256')) if v is not None]
            if not claims or any(v != self.source['sha256'] for v in claims):
                raise ValueError('Navigation camera source SHA mismatch or absent source binding')
            self._check_saved_binding(origin.get('source_binding', {}))
            for key in ('first_pts', 'stop_pts_exclusive', 'reference_pts'):
                if type(camera[key]) is not int:
                    raise ValueError('Camera exposure domain requires native integer PTS')
            processing = row['processing']
            if any(camera[k] != processing[k] for k in ('first_pts', 'stop_pts_exclusive')):
                raise ValueError('Navigation shot domain differs from exported processing interval')
            if scope is not None:
                matches = [s for s in scope['shots'] if s['id'] == shot]
                if len(matches) != 1 or any(camera[k] != matches[0]['processing'][k]
                    for k in ('first_pts', 'stop_pts_exclusive')):
                    raise ValueError('Navigation shot domain differs from current programme scope')
            self.cameras[shot] = (camera, controls, cref, kref)
            self.refs.extend((cref, kref))
        geometry.assert_unchanged()
        self.scene_content_revision = scene_content_identity(geometry.scene, geometry.scene_path)
        # Existing review sampling is 129 stations per base path. Densify using
        # the SAME native evaluator; long visible board/ice runs otherwise have
        # insufficient search queries. Installed print samples are already dense.
        dense = sample_entities(geometry.scene, geometry.core, samples=2049)
        self.cuts = [dict(cut) for cut in dense['cuts']]
        pieces, offset = [dense['xyz']], len(dense['xyz'])
        for cut in geometry.samples['cuts']:
            if cut['layer'] != 'advertisement':
                continue
            xyz = geometry.samples['xyz'][cut['begin']:cut['end']]
            self.cuts.append(dict(cut, begin=offset, end=offset+len(xyz)))
            pieces.append(xyz)
            offset += len(xyz)
        self.xyz = np.vstack(pieces)
        self.signals = {row['native_path_id']: {'blue':'blue_excess', 'red':'red_excess',
            'dark':'dark_band'}[row['material_signal']] for row in geometry.ice_inventory()}

    def _check_saved_binding(self, binding):
        if (binding.get('source_sha256') != self.source['sha256']
                or binding.get('source_id') != self.source['id']
                or binding.get('image_size') != self.hw[::-1]
                or binding.get('native_pts_step') != self.step
                or any(Fraction(*binding.get(k, (0, 1))) != self.clock
                       for k in ('model_time_base', 'scope_time_base'))):
            raise ValueError('Saved navigation source SHA/clock/image binding mismatch')

    def assert_unchanged(self):
        self.geometry.assert_unchanged()
        for ref in self.refs:
            if file_ref(ref['path'])['sha256'] != ref['sha256']:
                raise ValueError('Navigation dependency changed during source reading: '+ref['path'])

    def project(self, source_binding):
        """Return current-frame guides plus audit receipt, without reading RGB."""
        self.assert_unchanged()
        binding = deepcopy(source_binding)
        if (binding.get('source_sha256') != self.source['sha256']
                or Fraction(binding['time_base']) != self.clock
                or binding.get('native_hw') != self.hw
                or binding.get('source_id', self.source['id']) != self.source['id']):
            raise ValueError('Navigation requires matching source SHA, clock and native image size')
        shot, pts = binding['shot_id'], binding['source_pts']
        if shot not in self.cameras:
            raise ValueError('No explicitly selected navigation camera for shot '+str(shot))
        camera, controls, cref, kref = self.cameras[shot]
        if type(pts) is not int or (pts-camera['first_pts']) % self.step:
            raise ValueError('Navigation requires an actual native exposure PTS on the shot clock')
        _, _, t = exposure_controls(camera, controls, pts)
        projection = self.geometry.core.project_motion(camera['rig'], controls,
            camera['first_control_time'], camera['control_dt'], self.hw[0], t, self.xyz)
        uv = projection['uv']
        valid = (projection['status'] == 0) & np.isfinite(uv).all(axis=1) & (projection['depth'] > 0)
        in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= self.hw[1]-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= self.hw[0]-1)
        metadata = dict(model_revision=self.geometry.refs[0]['sha256'],
            scene_content_revision=self.scene_content_revision,
            scene_dependencies=deepcopy(self.geometry.refs), camera=cref, controls=kref,
            source_binding=binding, evaluator='NativeCore.project_motion',
            readout_mode=camera['readout_mode'], signed_readout_seconds=camera['rig'][37],
            native_representation_samples=len(self.xyz), source_pixels_measured=False,
            image_alignment_checked=False, visibility_and_occlusion_confirmed=False)
        # The per-frame receipt carries the complete dependency list once.
        # Every guide retains the immutable scene/camera/controls identity.
        guide_projection = {key:value for key,value in metadata.items()
                            if key not in ('scene_dependencies', 'native_representation_samples')}
        guides = []
        for cut in self.cuts:
            begin, end = cut['begin'], cut['end']
            indices, segments = _selected_indices(uv[begin:end], in_frame[begin:end],
                self.spacing, max(96., self.radius*4))
            if not len(indices):
                continue
            guide = dict(id=cut['id'], display_label=cut['label'],
                origin='model_projection_search_hint', navigation_provenance=PROJECTION_TAG,
                identity_confirmed=False, identity_authority=None,
                hint_is_source_measurement=False, source_binding=binding,
                points_uv=uv[begin+indices].tolist(), segment_ids=segments.tolist(),
                closed=bool(cut['closed'] and in_frame[begin:end].all() and len(set(segments)) == 1),
                search_radius_px=self.radius, model_search_projection=guide_projection,
                native_sample_indices=(begin+indices).tolist())
            if cut['id'] in self.signals:
                guide['signal'] = self.signals[cut['id']]
            guides.append(guide)
        guides, native_binding = self.geometry.bind(guides)
        report = dict(schema='rinkspace.native_navigation_projection.v1', source_binding=binding,
            seed_set=self.refs[0], model=metadata, current_native_path_count=len(self.cuts),
            projected_in_frame_path_count=len(guides), query_point_count=sum(len(g['points_uv']) for g in guides),
            queried_paths_by_kind=dict(Counter(g['kind'] for g in guides)),
            native_projection_status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(projection['status'],return_counts=True))},
            unlocalized_print_instance_ids=self.geometry.samples['inventory']['unlocalized_print_instance_ids'],
            missing_entities=deepcopy(self.geometry.samples['missing_entities']),
            native_binding=native_binding, new_observations=0, identity_confirmed_by_projection=False,
            fit_executed=False, user_approval='not_reviewed')
        return guides, report


def _query_array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _uncovered_query(incoming, retained, *, geometry=None):
    """Keep unqueried pieces; a path name alone does not cover its whole image."""
    def key(g):
        d = geometry.lookup(g['id']) if geometry is not None else None
        return d['id'] if d is not None else g['id']
    uv = _query_array(incoming['points_uv']).astype(float)
    segments = _query_array(incoming.get('segment_ids', np.zeros(len(uv), int)))
    covered = np.zeros(len(uv), bool)
    # Pixel query support only. This neither confirms identity nor creates UV.
    spacing = float(incoming.get('sample_spacing_px', 8.))
    for old in retained:
        if key(old) != key(incoming):
            continue
        points = _query_array(old['points_uv']).astype(float)
        seg = _query_array(old.get('segment_ids', np.zeros(len(points), int)))
        good = np.isfinite(points).all(1)
        if old.get('temporal_flow_supported') is not None:
            good &= _query_array(old['temporal_flow_supported']).astype(bool)
        if old.get('last_measured_query_indices') is not None:
            measured = np.zeros(len(points), bool)
            measured[np.asarray(old['last_measured_query_indices'], int)] = True
            good &= measured
        ids = np.flatnonzero(good)
        for group in np.split(ids, np.flatnonzero((np.diff(ids)>1)|(np.diff(seg[ids])!=0))+1):
            if not len(group):
                continue
            part = points[group]
            distance = np.linalg.norm(uv[:,None]-part[None], axis=2).min(axis=1)
            if len(part)>1:
                a,b = part[:-1],part[1:]
                v=b-a; vv=np.einsum('ij,ij->i',v,v)
                t=np.clip(np.einsum('nmi,mi->nm',uv[:,None]-a[None],v)/np.maximum(vv,1e-12),0,1)
                distance=np.minimum(distance,np.linalg.norm(uv[:,None]-(a[None]+t[:,:,None]*v),axis=2).min(axis=1))
            covered |= distance <= spacing*.5
    keep = ~covered
    if not keep.any():
        return None
    # An isolated missing endpoint still needs its neighbour for a tangent.
    for i in np.flatnonzero(keep):
        if i>0 and segments[i-1]==segments[i]:keep[i-1]=True
        if i+1<len(uv) and segments[i+1]==segments[i]:keep[i+1]=True
    ids=np.flatnonzero(keep)
    result=dict(incoming,points_uv=uv[ids],segment_ids=np.cumsum(np.r_[0,
        (np.diff(ids)>1)|(segments[ids[1:]]!=segments[ids[:-1]])]).astype(int))
    for field in ('normals','temporal_flow_supported','original_source_point_indices',
                  'original_dependence_groups_per_point','native_sample_indices',
                  'query_reference_endpoint_indices','original_source_profile_indices',
                  'material_profile_half_extent_px_per_point','material_profile_intervals_px',
                  'source_material_interface_query_uv'):
        if field in incoming:
            value=incoming[field]
            if len(value)!=len(uv):raise ValueError('Query metadata lost its point correspondence: '+field)
            result[field]=[value[int(i)] for i in ids] if isinstance(value,list) else value[ids]
    result.pop('last_measured_query_indices',None)
    if incoming.get('material_identity_continuation'):
        support=incoming['material_identity_continuation']
        result['material_identity_continuation']=dict(support,members={str(j):support['members'][str(int(i))]
            for j,i in enumerate(ids) if str(int(i)) in support.get('members',{})})
    result['query_coverage']=dict(original_query_count=len(uv),new_query_count=len(ids),
        preserved_existing_source_runs=True,interpolated_pixels_emitted=False)
    return result


def sample_source_search_runs(guide, geometry, spacing_px=8.):
    """Densify a retained paint-axis search, never its observed point array."""
    definition=geometry.lookup(guide['id']) if geometry is not None else None
    if not definition or definition.get('measurement')!='centreline' or guide.get('kind')!='ice_paint':
        return guide
    uv=_query_array(guide['points_uv']).astype(float)
    segment=_query_array(guide.get('segment_ids',np.zeros(len(uv),int)))
    samples=[];segments=[];normals=[];support=[]
    for sid in dict.fromkeys(segment.tolist()):
        indices=np.flatnonzero(segment==sid)
        if len(indices)<2:continue
        path=uv[indices];delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
        for j,extent in enumerate(length):
            if extent<=0:continue
            count=max(1,int(np.ceil(extent/spacing_px)))
            for fraction in np.linspace(0,1,count+1)[:None if j==len(length)-1 else -1]:
                samples.append(path[j]+delta[j]*fraction);segments.append(int(sid))
                normals.append(np.array([-delta[j,1],delta[j,0]])/extent)
                support.append([int(indices[j]),int(indices[j+1])])
    if len(samples)<=len(uv):return guide
    result=dict(guide,points_uv=np.asarray(samples),segment_ids=np.asarray(segments),normals=np.asarray(normals),
        origin='source_annotation_search_hint',source_reference_navigation_only=True,
        original_reference_query_uv=uv,query_reference_endpoint_indices=support,
        query_UV_are_interpolated_navigation=True,interpolated_UV_used_as_measurements=False,
        reference_UV_used_as_measurement=False)
    for field in ('original_source_point_indices','original_dependence_groups_per_point','confirmed_source_uv'):
        if field in result:result['reference_'+field]=result.pop(field)
    if 'material_identity_continuation' in result:
        result['reference_material_identity_continuation']=result.pop('material_identity_continuation')
    return result


def merge_verified_source_guides(existing, additional, *, binding, geometry=None):
    """Use authenticated current-source references only to query missing runs."""
    from .vision_geometry import _same_frame
    retained=list(existing);added=[]
    for incoming in additional:
        _same_frame(binding,incoming['source_binding'])
        authority=incoming.get('identity_authority') or {}
        if incoming.get('identity_confirmed') is not True or not authority:
            raise ValueError('Source gap queries require independently authenticated identity')
        # The accepted-packet adapter verifies the exact receipt and file hashes.
        # Other callers use associate_candidates for machine support evidence.
        if authority.get('kind') != 'user_confirmed_visual_supervision' or not authority.get('packet'):
            raise ValueError('Current reference query lacks its exact accepted source packet')
        candidate=sample_source_search_runs(incoming,geometry)
        guide=_uncovered_query(candidate,retained,geometry=geometry)
        if guide is None:continue
        guide=dict(guide,reference_query_only=True,reference_UV_used_as_measurement=False,
            query_id=f"{guide['id']}::reference:{authority['packet']['sha256']}:{guide.get('visible_arc_branch','')}")
        retained.append(guide);added.append(guide['query_id'])
    return retained,dict(added_query_ids=added,source_reference_query_count=len(added),
        actual_RGB_reread_required=True,reference_UV_used_as_measurement=False)


def attach_native_navigation_context(queries, projections, *, binding, geometry):
    """Retain native field-order context even where source queries cover it.

    These complete paths are navigation, not additional queries or observations.
    An unconfirmed old source name cannot move this independent native context.
    """
    from .vision_geometry import _same_frame
    contexts={}
    for guide in projections:
        metadata=guide.get('model_search_projection')
        if not metadata:continue
        definition=geometry.lookup(guide['id'])
        if definition is None or definition['layer']!='ice':continue
        if metadata.get('model_revision')!=geometry.refs[0]['sha256']:
            raise ValueError('Identity navigation belongs to another native revision')
        source=metadata.get('source_binding',guide.get('source_binding'))
        if source is not None:_same_frame(binding,source)
        elif metadata.get('native_pts')!=binding['source_pts']:
            raise ValueError('Identity navigation belongs to another exposure')
        contexts[definition['id']]=dict(native_path_id=definition['id'],
            source_binding=deepcopy(binding),native_scene=geometry.refs[0],
            model_search_projection=metadata,points_uv=guide['points_uv'],
            segment_ids=guide.get('segment_ids',np.zeros(len(guide['points_uv']),int)),
            projected_UV_used_as_source_pixel_truth=False)
    result=[]
    for incoming in queries:
        guide=incoming;definition=geometry.lookup(guide['id'])
        if definition and definition['id'] in contexts:
            guide=dict(guide,native_identity_navigation=contexts[definition['id']])
        result.append(guide)
    return result


def merge_navigation_guides(existing, additional, *, binding, geometry=None):
    """Refresh pending model hints; source-anchored/verified tracks take priority.

    This runs AFTER temporal transport and source reidentification. Querying a
    named path, including consulting feedback for it, cannot grant its identity.
    """
    from .vision_geometry import _same_frame
    def key(guide):
        definition = geometry.lookup(guide['id']) if geometry is not None else None
        return definition['id'] if definition is not None else guide['id']
    fresh, seen = [], set()
    for incoming in additional:
        guide = dict(incoming)
        if (guide.get('origin') != 'model_projection_search_hint'
                or guide.get('navigation_provenance') != PROJECTION_TAG
                or guide.get('identity_confirmed') is not False
                or guide.get('identity_authority')
                or guide.get('hint_is_source_measurement') is not False):
            raise ValueError('Additional navigation must be unconfirmed model-only search hints')
        _same_frame(binding, guide['source_binding'])
        identity = key(guide)
        if identity in seen:
            raise ValueError('Duplicate additional navigation path: '+identity)
        seen.add(identity)
        fresh.append(guide)
    retained, replaced = [], []
    for guide in existing:
        pending_projection = (guide.get('navigation_provenance') == PROJECTION_TAG
            and guide.get('identity_confirmed') is not True and not guide.get('identity_authority')
            and guide.get('origin') != 'user_confirmed_visual_supervision')
        if pending_projection:
            replaced.append(guide['id'])
        else:
            retained.append(guide)
    appended, skipped = [], []
    for guide in fresh:
        remainder = _uncovered_query(guide, retained, geometry=geometry)
        if remainder is None:
            skipped.append(guide['id'])
        else:
            remainder['query_id'] = guide.get('query_id',guide['id']+'::native_search')
            appended.append(remainder)
    queries=retained+appended
    if geometry is not None:
        queries=attach_native_navigation_context(queries,fresh,binding=binding,geometry=geometry)
    return queries, dict(supplied_path_count=len(fresh),
        dispatched_path_count=len(appended), refreshed_pending_path_ids=replaced,
        preserved_source_or_verified_path_ids=skipped,
        identity_confirmed_by_navigation=False, main_guides_or_reidentification_reset=False)
