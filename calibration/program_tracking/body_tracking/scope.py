"""One active source range for future grid fitting and body trajectory work."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from rinkspace.storage import AssetStore, resolve_asset

PROJECT = Path(__file__).resolve().parents[3]
SOURCE_SHOTS = PROJECT / 'calibration/program_tracking/source_shots.json'


@dataclass(frozen=True)
class ProgrammeScope:
    source: dict
    active: dict

    @property
    def pts(self):
        return np.arange(self.active['start_pts'], self.active['end_pts_exclusive'],
                         self.active['frame_step_pts'], dtype=np.int64)

    @property
    def shots(self):
        return [(i, self.source['shots'][i]) for i in self.active['shot_indices']]

    @property
    def time_base(self):
        return Fraction(self.source['source_time_base'])

    def programme_seconds(self, pts):
        # Keep the established cache clock: do not reset after excluding shot0.
        return (np.asarray(pts) - self.source['window_pts'][0]) * float(self.time_base)

    def assert_pts(self, pts):
        if not np.array_equal(np.asarray(pts, np.int64), self.pts):
            raise ValueError('Expected every active original PTS exactly once in order')


def load_scope(path=SOURCE_SHOTS):
    source = json.loads(resolve_asset(PROJECT, path).read_text(encoding='utf-8-sig'))
    source['source_path_original'] = source.get('source_path_original', source['source_path'])
    source['source_path'] = str(AssetStore(PROJECT).source(source['source_path']))
    active = source.get('analysis_scope')
    if active is None:
        raise ValueError('Explicit user-authorized analysis_scope is required')
    if set(active['applies_to']) != {'rink_grid_fitting', 'body_trajectory'}:
        raise ValueError('Grid fitting and body tracking must share this active scope')
    scope = ProgrammeScope(source, active)
    selected = np.concatenate([np.arange(s['start_pts'], s['end_pts'],
                                         active['frame_step_pts'], dtype=np.int64)
                               for _, s in scope.shots])
    scope.assert_pts(selected)
    if len(scope.pts) != active['frame_count']:
        raise ValueError('Active frame count differs from its PTS range')
    archive = np.arange(*source['window_pts'], active['frame_step_pts'])
    excluded = np.concatenate([np.arange(e['start_pts'], e['end_pts_exclusive'],
                                          active['frame_step_pts'])
                               for e in active['excluded']])
    if len(archive) != source['native_frame_count'] or not np.array_equal(
            np.sort(np.concatenate((excluded, scope.pts))), archive):
        raise ValueError('Active/excluded ranges must partition the original cache')
    if len(excluded) != 17 or active['shot_indices'] != list(range(1, 10)):
        raise ValueError('Current user exclusion is exactly the pre-programme shot0')
    return scope
