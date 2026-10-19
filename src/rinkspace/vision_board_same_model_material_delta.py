"""Same-scene navigation and actual RGB rereads of known Ad25/26 instances.

An independent future delta reader; it neither mutates nor imports a running
programme reader's tracking state. Actual instance texture/field context and
current printed interfaces are required. A model projection only bounds search.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.program_tracking.projection import safe_project
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
from calibration.program_tracking.strict_model_scope import authenticate_source_reference
from .vision_board_material_point_profiles import MATERIAL_ROLES, read_printed_material_corners
from .vision_board_model_source_profiles import read_circle_material_interface
from .vision_programme_board_review import _source_orientation_preserved


class KnownSameStateBoardMaterialDelta:
    def __init__(self, delta_path, *, expected_sha256):
        path, _ = authenticate_source_reference(dict(path=str(delta_path), sha256=expected_sha256))
        self.delta = json.loads(Path(path).read_text(encoding='utf8'))
        authority_path, _ = authenticate_source_reference(self.delta['source_identity_authority'])
        self.authority = json.loads(Path(authority_path).read_text(encoding='utf8'))
        identity_path, _ = authenticate_source_reference(self.authority['model_navigation_reference'])
        identity = json.loads(Path(identity_path).read_text(encoding='utf8'))
        camera_path, _ = authenticate_source_reference(dict(path=identity['camera_path'], sha256=identity['camera_sha256']))
        scene_path, _ = authenticate_source_reference(self.authority['scene_reference'])
        self.scene = json.loads(Path(scene_path).read_text(encoding='utf8'))
        if hashlib.sha256(Path(scene_path).read_bytes()).hexdigest() != identity['scene_sha256']:
            raise ValueError('The source delta and camera must use the same scene')
        with np.load(camera_path, allow_pickle=False) as saved:
            self.camera = dict(zip(map(int, saved['pts']), saved['camera25']))
            self.clock = dict(zip(map(int, saved['pts']), map(int, saved['exposure_representative_pts'])))
        self.sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.014)
        self.templates = []
        masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
        for measurement in self.delta['measurements']:
            binding = measurement['source']
            number = int(measurement['advertisement_rows'][0]['advertisement_number'])
            fact = next(f for f in self.authority['physical_instance_records'] if f['advertisement_id'] == number)
            ref = fact['source_image_references'][1]
            source_path, _ = authenticate_source_reference(ref)
            rgb = cv2.imread(str(source_path))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != binding['decoded_native_BGR_sha256']:
                raise ValueError('Known board template differs from its exact original source')
            xyz = self._xyz(number)
            navigation, _, visible = safe_project(self.camera[binding['native_pts']], xyz)
            if not visible.all():
                raise ValueError('Known original board model navigation is not optically valid')
            if number == 25:
                # Original source visually reread outer shield guides. These
                # are NAV, including the rounded tip that has no observation.
                navigation = np.asarray([[247., 303.], [294., 303.], [271., 379.]])
            mask = masks.static_scene_exclusion(binding['native_pts'])[0]
            low = np.maximum(0, np.floor(navigation.min(0)-70)).astype(int)
            high = np.minimum([1920, 1080], np.ceil(navigation.max(0)+71)).astype(int)
            region = rgb[low[1]:high[1], low[0]:high[0]]
            roi = (~mask[low[1]:high[1], low[0]:high[0]]).astype(np.uint8)*255
            keys, descriptors = self.sift.detectAndCompute(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), roi)
            xy = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)+low
            self.templates.append(dict(number=number, binding=binding, measurement=measurement,
                navigation=navigation, xyz=xyz, xy=xy, descriptors=descriptors,
                approved_shots=fact['approved_shot_ids']))

    def _xyz(self, number):
        shared = self.scene['shared_board_patterns']
        if number == 25:
            return np.asarray([shared['shield_fixed_material_points_xyz_scene_units']['25'][r] for r in MATERIAL_ROLES])
        return np.asarray(shared['circle_outer_stroke_paths_xyz_scene_units']['26'])

    def _register(self, image, mask, template, projected):
        low = np.maximum(0, np.floor(projected.min(0)-96)).astype(int)
        high = np.minimum([1920, 1080], np.ceil(projected.max(0)+97)).astype(int)
        if np.any(high <= low):
            return None
        region = image[low[1]:high[1], low[0]:high[0]]
        roi = (~mask[low[1]:high[1], low[0]:high[0]]).astype(np.uint8)*255
        keys, descriptors = self.sift.detectAndCompute(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), roi)
        source = template['descriptors']
        if source is None or descriptors is None or min(len(source), len(descriptors)) < 8:
            return None
        matches = [m for m, n in cv2.BFMatcher().knnMatch(source, descriptors, k=2) if m.distance < .76*n.distance]
        if len(matches) < 8:
            return None
        first = template['xy'][[m.queryIdx for m in matches]]
        second = np.asarray([keys[m.trainIdx].pt for m in matches], np.float32)+low
        matrix, inliers = cv2.findHomography(first, second, cv2.RANSAC, 1.)
        if matrix is None or inliers is None:
            return None
        keep = inliers.ravel().astype(bool)
        if (keep.sum() < 8 or keep.mean() < .55 or len(np.unique(first[keep], axis=0)) < 8
                or np.ptp(first[keep], axis=0).min() < 8.
                or not _source_orientation_preserved(matrix, first[keep])):
            return None
        error = np.linalg.norm(cv2.perspectiveTransform(first[None], matrix)[0]-second, axis=1)
        if error[keep].max() > 1.5:
            return None
        transported = cv2.perspectiveTransform(template['navigation'].astype(np.float32)[None], matrix)[0]
        if np.max(np.linalg.norm(transported-projected, axis=1)) > 128.:
            return None
        return transported, dict(kind='known_physical_board_actual_RGB_context_registration_with_same_model_navigation',
            original_source_feature_uv=first[keep], current_source_feature_uv=second[keep],
            actual_source_registration_error_px=error[keep], source_guides_and_matrix_are_navigation_only=True,
            original_person_union_exclusion_retained=True)

    def read(self, image, exclusion_mask, source_binding, *, precomputed_measurement=None):
        """Return current actual rows plus explicit pending instances, never fit."""
        binding = dict(source_binding)
        native = int(binding.get('native_pts', binding.get('source_pts')))
        original_source = self.templates[0]['binding']
        if (image.shape != (1080,1920,3) or image.dtype != np.uint8
                or binding.get('source_sha256', binding.get('video_sha256')) != original_source['source_sha256']
                or binding.get('time_base') != original_source['time_base']):
            raise ValueError('Actual board delta source video/timebase/native RGB changed')
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if (native not in self.camera or self.clock[native] != int(binding['actual_pts'])
                or digest != binding.get('decoded_native_BGR_sha256', binding.get('decoded_BGR_sha256'))):
            raise ValueError('Board model navigation requires the same exact frozen original clock/BGR')
        binding.update(native_pts=native, source_pts=native, decoded_native_BGR_sha256=digest,
            decoded_BGR_sha256=digest, source_path=self.templates[0]['binding']['source_path'],
            source_sha256=self.templates[0]['binding']['source_sha256'],
            video_sha256=self.templates[0]['binding']['source_sha256'], time_base='1/15360')
        if native != int(binding['actual_pts']):
            return dict(source=binding, advertisement_rows=[],
                unconfirmed_physical_instances=[dict(reason='native_alias_requires_its_separate_actual_readout')],
                new_FIT_observations_created=False, whole_programme_review_complete=False)
        rows, pending = [], []
        for template in self.templates:
            number = template['number']
            if binding['shot_id'] not in template['approved_shots']:
                pending.append(dict(advertisement_id=number,reason='physical_field_order_not_yet_verified_in_this_cut'))
                continue
            original = template['binding']
            if native == original['native_pts']:
                if digest != original['decoded_native_BGR_sha256']:
                    raise ValueError('Exact board identity anchor source changed')
                rows.extend(deepcopy(template['measurement']['advertisement_rows']))
                continue
            if binding['actual_pts'] == original['actual_pts']:
                pending.append(dict(advertisement_id=number,reason='native_alias_requires_its_separate_actual_readout'))
                continue
            projected, _, good = safe_project(self.camera[native], template['xyz'])
            if not good.all() or not np.any((projected >= 0).all(1) & (projected < [1920,1080]).all(1)):
                continue
            registered = self._register(image, exclusion_mask, template, projected)
            if registered is None:
                pending.append(dict(advertisement_id=number,reason='unknown_or_competing_current_physical_instance_context'))
                continue
            navigation, registration = registered
            profile_image=image.astype(np.float32,copy=False)
            if number == 25:
                found = read_printed_material_corners(profile_image, exclusion_mask, navigation)
                new_rows = []
                for index, readout in found.items():
                    row = deepcopy(template['measurement']['advertisement_rows'][index])
                    row.update(uv=readout['uv'][None],normal=np.empty((0,2)),
                        anchor_point_indices=np.asarray([index],np.int32),source_material_point_readout=readout,
                        actual_2d_readout_bounds_xy_px=readout['actual_2d_readout_bounds_xy_px'][None])
                    row['source_readout_uncertainty']['actual_2d_readout_bounds_xy_px']=readout['actual_2d_readout_bounds_xy_px'][None]
                    new_rows.append(row)
            else:
                measured = read_circle_material_interface(profile_image, exclusion_mask, navigation)
                readings = measured['readings']
                indices = np.asarray([r['navigation_index'] for r in readings],int)
                uv = np.asarray([r['uv'] for r in readings],float).reshape(-1,2)
                new_rows=[]
                if len(uv):
                    cuts=np.flatnonzero((np.diff(indices)!=1)|(np.linalg.norm(np.diff(uv,axis=0),axis=1)>6.))+1
                    for segment,keep in enumerate(np.split(np.arange(len(uv)),cuts)):
                        row=deepcopy(template['measurement']['advertisement_rows'][0])
                        n=np.asarray([readings[i]['profile_navigation_normal'] for i in keep]);adj=len(keep)>=2
                        if adj:
                            tangent=np.gradient(uv[keep],axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
                            derived=np.c_[-tangent[:,1],tangent[:,0]];derived[np.sum(derived*n,axis=1)<0]*=-1
                            factor=np.sum(derived*n,axis=1);n=derived
                        else:factor=np.ones(len(keep))
                        blur=np.asarray([readings[i]['actual_readout_blur_interval_px'] for i in keep])*factor[:,None]
                        row.update(uv=uv[keep],normal=n,anchor_point_indices=indices[keep],source_boundary_segment_index=segment,
                            source_neighbour_adjacency_confirmed=adj,normal_from_actual_source_neighbour_pixels=adj,
                            normal_measurement_origin='same_current_source_visible_contour_adjacent_measured_UV' if adj else 'model_navigation_normal_not_source_truth')
                        row['source_readout_uncertainty'].update(actual_readout_blur_interval_px=blur,normal_interval_px=blur,
                            source_native_pixel_quantization_step_px=1.)
                        new_rows.append(row)
            for row in new_rows:
                row.update(native_pts=native,actual_pts=int(binding['actual_pts']),shot_id=binding['shot_id'],
                    decoded_native_BGR_sha256=digest,decoded_BGR_sha256=digest)
                context=deepcopy(row['source_identity_navigation_evidence'])
                context.update(source_binding=binding,actual_source_context_registration=registration)
                row['source_identity_navigation_evidence']=context
                row['temporal_evidence'].update(current_actual_pts=int(binding['actual_pts']),
                    previous_actual_pts=int(original['actual_pts']),previous_shot_id=original['shot_id'],
                    previous_decoded_BGR_sha256=original['decoded_native_BGR_sha256'],
                    direction='forward' if original['actual_pts']<binding['actual_pts'] else 'backward',
                    source_identity_navigation_evidence=deepcopy(context))
            rows.extend(new_rows)
            if not new_rows:
                pending.append(dict(advertisement_id=number,reason='no_current_unique_visible_material_interface'))
        for row in rows:
            if row['curve']['kind'] == 'shared_board_pattern_material_points':
                row.setdefault('anchor_point_indices',np.asarray([MATERIAL_ROLES.index(row['material_roles'][0])],np.int32))
            elif row.get('normal_measurement_origin') == 'same_current_source_visible_contour_adjacent_measured_UV':
                row['normal_from_actual_source_neighbour_pixels']=True
                row['source_readout_uncertainty']['source_native_pixel_quantization_step_px']=1.
        return dict(source=binding,advertisement_rows=rows,unconfirmed_physical_instances=pending,
            original_person_union_exclusion_retained=True,projected_UV_used_as_source_pixel_truth=False,
            whole_programme_review_complete=False)
