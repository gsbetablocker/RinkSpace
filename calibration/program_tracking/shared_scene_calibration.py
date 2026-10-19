"""Shared painted-circle calibration, preserving immutable source pixels.

One displacement and outer-radius change is shared by every observation of
each selected physical circle. The released scene remains untouched. A solve
exports a separate candidate scene with its own content hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from candidate_scene_geometry_audit import audit_exported_circle_geometry
from shared_markers import RingPixels, add_shared_ring_prior, ring_world


class SharedRingScene:
    def __init__(self, scene: dict, options: dict):
        self.scene = scene
        self.options = options
        names = list(options.get('features', []))
        if len(names) != len(set(names)) or not names:
            raise ValueError('Shared rings need unique physical feature IDs')
        features = {row['id']: row for row in scene['ice_features']}
        self.features = {}
        for name in names:
            feature = features[name]
            if feature['kind'] != 'circle':
                raise ValueError(f'{name} is not a physical circle')
            rings = {r.get('role', 'outer'): r
                     for r in (feature.get('rings') or [feature])}
            outer = rings['outer']
            self.features[name] = {
                'feature': feature,
                'center': np.asarray(outer['center_xyz'][:2], float),
                'outer_radius': float(outer['radius']),
                'middle_radius': float(rings.get('middle', {}).get('radius', 0.)),
            }
        self.keys = {name: 2000+i for i, name in enumerate(sorted(names))}
        self.initial = {key: np.zeros(11) for key in self.keys.values()}

    def add_priors(self, graph):
        for name, key in self.keys.items():
            feature = self.features[name]
            add_shared_ring_prior(
                graph, key, feature['outer_radius'], feature['middle_radius'],
                center_sigma_scene=float(self.options.get('center_sigma_scene', 1.)),
                radius_sigma_scene=float(self.options.get('radius_sigma_scene', .3)))

    def group(self, pts, name, camera, world, uv, sigma, source_file=''):
        feature_id, role = name.rsplit('_', 1)
        if feature_id not in self.features:
            return None
        meta = self.features[feature_id]
        return RingPixels(feature_id, role, int(pts), 'programme', name,
                          source_file, np.asarray(camera).copy(),
                          np.asarray(world).copy(), meta['center'].copy(),
                          meta['outer_radius'], meta['middle_radius'],
                          np.asarray(uv).copy(), np.asarray(sigma).copy())

    def world(self, group, states):
        return ring_world(group, states[self.keys[group.feature_id]])

    def warm_start(self, directory: Path, guesses: dict):
        report = json.loads((directory/'diagnostics.json').read_text(encoding='utf8'))
        previous = report.get('shared_ring_candidates', {})
        for name, row in previous.items():
            if name not in self.keys:
                raise ValueError('Warm candidate has a ring not enabled in this solve')
            q = np.asarray(row['state11'], float)
            if q.shape != (11,) or not np.isfinite(q).all():
                raise ValueError('Invalid saved shared-ring state')
            guesses[self.keys[name]] = q.copy()

    def describe(self, states):
        result = {}
        for name, key in self.keys.items():
            q = states[key]
            meta = self.features[name]
            result[name] = {
                'state11': q.tolist(),
                'center_delta_scene': (q[:2]*100.).tolist(),
                'outer_radius_delta_scene': float(q[2]*100.),
                'outer_radius_scene': float(meta['outer_radius']+100.*q[2]),
                'all_source_frames_share_this_geometry': True,
            }
        return result

    def export(self, output: Path, states, base_hash: str,
               ad_geometries: dict | None = None) -> tuple[dict, str, Path]:
        scene = copy.deepcopy(self.scene)
        for feature in scene['ice_features']:
            name = feature['id']
            if name not in self.keys:
                continue
            q = states[self.keys[name]]
            dx, dy, dr = q[:3]*100.
            if 'center_xyz' in feature:
                feature['center_xyz'][0] += float(dx)
                feature['center_xyz'][1] += float(dy)
            if 'radius' in feature:
                feature['radius'] += float(dr)
            for ring in feature.get('rings', []):
                ring['center_xyz'][0] += float(dx)
                ring['center_xyz'][1] += float(dy)
                if ring['role'] == 'outer':
                    ring['radius'] += float(dr)
            feature['geometry_status'] = 'joint_original3_programme_candidate_unvalidated'
        if ad_geometries:
            for ad in scene['advertisements']:
                candidate = ad_geometries.get(int(ad['user_number']))
                if candidate:
                    ad['corners_xyz_scene_units'] = candidate['corners_xyz_scene_units']
                    ad['geometry_status'] = 'joint_source_ad_candidate_unvalidated'
        scene['programme_calibration_candidate'] = {
            'status': 'unvalidated', 'base_model_sha256': base_hash,
            'shared_ring_candidates': self.describe(states),
            'source_pixels_modified': False,
        }
        scene['shared_geometry_checks'] = audit_exported_circle_geometry(scene, self.scene, base_hash)
        file = output/'candidate_scene.json'
        payload = (json.dumps(scene, ensure_ascii=False, indent=2)+'\n').encode('utf8')
        file.write_bytes(payload)
        return scene, hashlib.sha256(payload).hexdigest(), file
