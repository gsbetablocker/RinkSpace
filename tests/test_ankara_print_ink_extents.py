from pathlib import Path

import numpy as np
import pytest


def test_horizontal_and_vertical_ink_extrema_stay_on_the_print(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'events/ankara_2026/tools/joint_refine_1117'
    monkeypatch.syspath_prepend(str(scripts))
    from feedback_fit import bind_extrema, ji

    # An open horizontal arc has distinct side endpoints and interior top.
    stations = np.linspace(0, 1, 513)
    arc = np.column_stack((100 + 40*stations, 60 + 120*(stations-.25)**2))
    monkeypatch.setattr(ji, 'native_paths', lambda *args, **kwargs: {'ink': (arc, stations)})
    monkeypatch.setattr(ji, 'project_groups', lambda *args: [(arc, np.zeros(len(arc), dtype=int))])
    shot = {'shots': {'s': {'first': 0., 'dt': .02}}}
    state = {'s_times': np.array([0.]), 's_controls': None, 's_rigs': None}
    rows = {'path': np.array(['ink']*4), 'shot': np.array(['s']*4), 'idx': np.zeros(4, int),
            'material': np.array(['ink_extent_xmin', 'ink_extent_xmax', 'ink_extent_ymin', 'ink_extent_ymax']),
            'station': np.full(4, np.nan)}
    bind_extrema(None, state, shot, rows, np.ones(4, bool))
    np.testing.assert_allclose(rows['station'], [0., 1., .25, 1.], atol=1e-9)

    # A rink outline tip requires visible curvature around the extremum.
    rows['material'][0] = 'outline_xmin'
    with pytest.raises(ValueError, match='Missing interior outline extremum'):
        bind_extrema(None, state, shot, rows, np.array([True, False, False, False]))
