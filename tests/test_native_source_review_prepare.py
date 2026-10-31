"""Exclusion preparation preserves real rig/scene array shapes and source rows."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROGRAMME = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(PROGRAMME))
spec = importlib.util.spec_from_file_location('source_review_prepare', PROGRAMME/'prepare_native_source_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class SourceReviewPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rink_source_review_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'source.png'
        self.source.write_bytes(b'original source evidence fixture')
        self.layout = dict(size=280, layout_revision='test-layout', scene_content_revision='test-scene')
        self.model = SimpleNamespace(initial_state=SimpleNamespace(layout=self.layout))
        self.z = dict(pts=np.array([1000, 1256, 2000, 2256, 3000, 3256], np.int64),
            observation_ids=np.array(['point'+str(i) for i in range(6)]),
            uv=np.arange(12, dtype=float).reshape(6, 2)+10,
            kind=np.array([1, 0, 1, 1, 0, 1], np.uint8), stations=np.full(6, .5),
            source_rows=np.array([100, 105, 120, 121, 150, 151], np.int64),
            path_indices=np.array([0, 1, 0, 1, 0, 1], np.int32))
        self.mapping = dict(path_ids=['native/path0', 'native/path1'])
        self.shots = [dict(id='s0'+str(i+1), row_begin=2*i, row_stop=2*i+2,
            control_begin=5*i, control_stop=5*i+5, reference_pts=int(self.z['pts'][2*i]),
            first_pts=int(self.z['pts'][2*i]), last_pts=int(self.z['pts'][2*i+1]),
            stop_pts_exclusive=int(self.z['pts'][2*i+1]+256), first_control_time=-.04,
            time_base_num=1, time_base_den=12800, knot_stride=1, motion_reference_time=0., camera_index=i)
            for i in range(3)]
        self.manifest = dict(shots=self.shots, cameras=[dict(id='rig'+str(i)) for i in range(3)],
                             control_dt=.02, first_control_time=0., max_readout=.02)
        rig = np.arange(129, dtype=float).reshape(3, 43)/100
        rig[:, 26:28] = [959.5, 539.5]
        rig[:, [28, 29, 30]] = 0
        self.state = dict(rig=rig, controls=np.arange(45, dtype=float).reshape(15, 3),
            scene_values=np.arange(280, dtype=float), stations=np.array([.31, .5, .42, .61, .5, .71]),
            source_rows=self.z['source_rows'].copy())

    def decisions(self):
        return dict(decisions=[dict(observation_id=str(self.z['observation_ids'][i]),
            pts=int(self.z['pts'][i]), expected_path_id=self.mapping['path_ids'][self.z['path_indices'][i]],
            uv=self.z['uv'][i].tolist(), source_row=int(self.z['source_rows'][i]),
            decision='exclude_preserving_source', reason='source material identity explicitly unresolved',
            source_image='source.png') for i in (0, 3)])

    def test_filter_preserves_seven_fields_numeric_stations_and_shot_order(self):
        before = copy.deepcopy(self.z)
        result = review.revise_exclusions(self.z, self.state['stations'], self.manifest,
                                          self.mapping['path_ids'], self.decisions(), self.root/'review.json')
        new, stations, shots, touched, records, additions, origins = result
        np.testing.assert_array_equal(origins, [1, 2, 4, 5])
        self.assertEqual(set(new), review.prepared.FIELDS)
        for key in new:
            np.testing.assert_array_equal(new[key], before[key][origins])
            np.testing.assert_array_equal(self.z[key], before[key])
        np.testing.assert_array_equal(stations, self.state['stations'][origins])
        self.assertEqual([(s['row_begin'], s['row_stop']) for s in shots], [(0, 1), (1, 2), (2, 4)])
        self.assertFalse(touched.any())
        self.assertFalse(additions)
        self.assertEqual([r['source_row'] for r in records], [100, 121])

    def test_other_actions_and_additions_are_rejected(self):
        for action in ('correct_uv', 'rebind_preserving_uv', 'unknown'):
            data = self.decisions()
            data['decisions'][0]['decision'] = action
            with self.assertRaisesRegex(ValueError, 'Only exclude'):
                review.revise_exclusions(self.z, self.state['stations'], self.manifest,
                    self.mapping['path_ids'], data, self.root/'review.json')
        data = self.decisions()
        data['added_observations'] = [dict(observation_id='new')]
        with self.assertRaisesRegex(ValueError, 'source-packets'):
            review.revise_exclusions(self.z, self.state['stations'], self.manifest,
                self.mapping['path_ids'], data, self.root/'review.json')

    def test_prepared_directory_validates_real_shapes_without_mutating_parent(self):
        parent = self.root/'mother'
        parent.mkdir()
        scene, library = self.root/'scene.json', self.root/'native.dll'
        scene.write_text('{}', encoding='utf-8')
        library.write_bytes(b'current runtime identity fixture; no library is loaded')
        np.savez(parent/'point_observations.npz', **self.z)
        m = dict(self.manifest, point_input_sha256=review.prepared.digest(parent/'point_observations.npz'),
            scene_path=str(scene), scene_file_sha256=review.prepared.digest(scene),
            native_library=str(library), native_library_sha256=review.prepared.digest(library))
        review.prepared.write(parent/'joint_manifest.json', m)
        review.prepared.write(parent/'mapping.json', self.mapping)
        review.prepared.bind_snapshot(self.state, self.layout, self.mapping, m)
        np.savez(parent/'initial_pixels.npz', **self.state)
        report = self.decisions()
        report['input_sha256'] = m['point_input_sha256']
        review_path = self.root/'review.json'
        review.prepared.write(review_path, report)
        protected = {str(p): review.prepared.digest(p) for p in parent.iterdir()}
        output = self.root/'reviewed'
        with patch.object(review, 'SceneParameterModel', return_value=self.model):
            receipt = review.prepare(parent, parent/'initial_pixels.npz', review_path, output)
        z = review.prepared.npz(output/'point_observations.npz')
        v = review.prepared.npz(output/'initial_pixels.npz')
        updated, mapping = review.prepared.read(output/'joint_manifest.json'), review.prepared.read(output/'mapping.json')
        review.prepared.validate_snapshot(v, self.layout, mapping, updated, z)
        self.assertEqual(v['rig'].shape, (3, 43))
        self.assertEqual(v['controls'].shape, (15, 3))
        self.assertEqual(v['scene_values'].shape, (280,))
        self.assertEqual(v['stations'].shape, (4,))
        np.testing.assert_array_equal(v['source_rows'], [105, 120, 150, 151])
        for name in ('rig', 'controls', 'scene_values'):
            self.assertEqual(v[name].tobytes(), self.state[name].tobytes())
        for path, digest in protected.items():
            self.assertEqual(review.prepared.digest(Path(path)), digest)
        self.assertTrue(receipt['parent_files_unchanged'])
        self.assertEqual(receipt['excluded_rows'], 2)
        self.assertFalse(receipt['fit_executed'])
        self.assertEqual(set(z), review.prepared.FIELDS)


if __name__ == '__main__':
    unittest.main()
