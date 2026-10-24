"""Observed anchor diagnostics bind the source row clock across all shots."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rinkfit_joint import dense_fixed_anchor_guard as guard


class AnchorClockTests(unittest.TestCase):
    def run_guard(self, root, name, clock=None, baseline=None):
        uv = np.array([[10., 100.], [20., 200.]])
        arrays = dict(kind=np.ones(2, dtype=int), material_index=np.zeros(2, dtype=int),
                      admission=np.ones(2, dtype=int), fit_role=np.ones(2, dtype=int),
                      uv=uv, native_pts=np.array([256, 512]), shot_index=np.array([0, 1]), E=np.ones(2))
        packet = dict(row_count=2, shot_ids=['s01', 's02'], source_row_ids=['source1', 'source2'],
                      arrays={key: {'column': key} for key in arrays})
        class Loader:
            def __init__(self, path): self.packet = packet
            def load(self, key): return arrays[key]
        class Scene:
            def __init__(self, model): pass
            def world_rows(self, materials, branches, stations):
                return np.c_[uv, np.zeros(2)], np.ones(2, bool), None
            def path(self, material, branch): return dict(physical_point=True)
        class Camera:
            tau = 1.
            def at(self, world, time):
                return dict(uv=np.c_[world[:, 0], (time + .5) * 1079.], valid=np.ones(len(world), bool))
            def project_rs(self, world, time):
                valid = np.ones(len(world), bool)
                return dict(uv=world[:, :2], valid=valid, bracketed=valid,
                            certificate=dict(certified=valid), root_equality_px=np.zeros(len(world)))
        class Graph:
            def __init__(self, model): pass
            def native_camera(self, shot): return Camera()
            def local_seconds(self, shot, pts): return np.zeros(len(pts))
        model_path, packet_path = root / 'model.json', root / 'packet.json'
        model_path.write_text(json.dumps(dict(materials=[dict(label='Ad1.point')],source_clock=dict(time_base=[1,12800],frame_step_pts=256),shots={'s01':dict(time_origin_pts=256,scope=dict(first_pts=256,last_pts=256)),'s02':dict(time_origin_pts=512,scope=dict(first_pts=512,last_pts=512))},source_graph_ref=str(root/'missing_initializer.json'))),'utf8')
        packet_path.write_text('{}', 'utf8')
        prefix = dict(count=2, source_offset=0, arrays=dict(path='paths', row='historical_rows'))
        def source_column(column, parent):
            return np.array([800., 900.]) if column == 'historical_rows' else np.zeros(2, dtype=int)
        with patch.multiple(guard, PacketArrays=Loader, NativeScene=Scene, native_camera_from_joint=lambda *args:Camera(),
                            as_event_view=lambda model, shot: model,
                            find_overlay=lambda *args: (prefix, dict(overlay=dict(file=str(root / 'overlay.json')))),
                            material_visibility_records=lambda *args: None,
                            render_visibility=lambda cam, scene, world, *args: np.ones(len(world), bool)),                 patch.object(guard, 'source_col', side_effect=source_column) as columns,                 contextlib.redirect_stdout(io.StringIO()):
            kwargs = {} if clock is None else dict(observed_row_clock=clock)
            report = guard.evaluate_fixed_anchors(model_path, packet_path, root / name, baseline, **kwargs)
        return report, columns.call_count

    def test_default_source_clock_matches_s01_and_later_shots(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, loaded = self.run_guard(Path(tmp), 'source')
            self.assertEqual(result['observed_row_clock'], 'source_uv')
            self.assertEqual(loaded, 1)  # Historical row column is not consumed.
            self.assertFalse(result['legacy_graph_initializer_required'])
            self.assertFalse(result['legacy_graph_initializer_available'])
            for shot in ('s01', 's02'):
                self.assertAlmostEqual(result['by_shot'][shot]['observed_row_rawUV']['rms_px'], 0.)

    def test_historical_clock_is_explicit_and_only_changes_s01(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, loaded = self.run_guard(Path(tmp), 'historical', 'historical_s01')
            self.assertEqual(result['observed_row_clock'], 'historical_s01')
            self.assertEqual(loaded, 2)
            self.assertAlmostEqual(result['by_shot']['s01']['observed_row_rawUV']['rms_px'], 700.)
            self.assertAlmostEqual(result['by_shot']['s02']['observed_row_rawUV']['rms_px'], 0.)

    def test_baseline_comparison_uses_same_bound_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_guard(root, 'baseline')
            result, _ = self.run_guard(root, 'candidate', baseline=root / 'baseline/report.json')
            self.assertEqual(result['same_source_comparison']['overall']['newly_invalid'], 0)
            self.assertEqual(result['same_source_comparison']['overall']['sum_squared_error_delta_on_common_valid'], 0.)
            for clock in ('historical_s01', 'source_uv'):
                baseline = json.loads((root / 'baseline/report.json').read_text('utf8'))
                if clock == 'source_uv': baseline.pop('observed_row_clock')
                file = root / ('mismatch_' + clock + '.json')
                file.write_text(json.dumps(baseline), 'utf8')
                with self.assertRaisesRegex(ValueError, 'different or unbound clock'):
                    guard.evaluate_fixed_anchors(root / 'absent.json', root / 'absent_packet.json',
                                                 root / ('bad_' + clock), file, clock)

    def test_unknown_clock_fails_before_loading_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Unknown observed-row clock'):
                guard.evaluate_fixed_anchors('absent', 'absent', Path(tmp) / 'bad', observed_row_clock='implicit')


if __name__ == '__main__':
    unittest.main()
