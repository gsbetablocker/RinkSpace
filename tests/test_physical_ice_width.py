"""Decoded-image bounds and real Python callers, using synthetic RGB/projection."""
import ast
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.physical_ice_width import (
    projected_width_profile, sample_physical_width_field, valid_profile_contributors,
)


class PlaneCamera:
    def ground_xy(self, uv):
        return np.asarray(uv)/100., np.ones(len(uv), bool)

    def project(self, xyz):
        return np.asarray(xyz)[:, :2]*100., np.ones(len(xyz))


PATH = dict(xyz=[[0., 0., 0.], [0., 1., 0.]], material_signal='blue')
SOURCE = dict(source_video_sha256='a'*64, native_pts=283136, time_base='1/12800')


def profile(hw, uv, radius=90.):
    return projected_width_profile(PlaneCamera(), PATH, uv, image_hw=hw, radius_px=radius)


@pytest.mark.parametrize('hw,uv', [((1080, 1920), (960, 540)),
                                  ((2160, 3840), (2000, 1500)),
                                  ((720, 1280), (640, 360))])
def test_in_frame_sections_keep_all_181_samples(hw, uv):
    result = profile(hw, uv)
    assert len(result['valid']) == 181
    assert result['valid'].all()
    assert valid_profile_contributors(result, np.zeros(hw, bool)).all()
    np.testing.assert_array_equal(result['query_uv'], uv)
    np.testing.assert_array_equal(result['source_uv'][:, 1], uv[1])
    assert result['source_uv'][0, 0] > uv[0] > result['source_uv'][-1, 0]
    # For 1080p the old image-bound expression has exactly the same behavior.
    if hw == (1080, 1920):
        pixels = result['source_uv']
        np.testing.assert_array_equal(result['valid'],
            (pixels >= [0., 0.]).all(1) & (pixels <= [1919., 1079.]).all(1))


@pytest.mark.parametrize('hw,uv', [((720, 1280), (1500, 900)),
                                  ((1080, 1920), (2000, 1500)),
                                  ((2160, 3840), (-200, 500))])
def test_out_of_frame_sections_are_still_rejected(hw, uv):
    result = profile(hw, uv)
    assert not result['valid'].any()
    assert not valid_profile_contributors(result, np.zeros(hw, bool)).any()


@pytest.mark.parametrize('dy,dx', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_each_bilinear_contributor_must_be_unmasked(dy, dx):
    hw = (720, 1280)
    result = profile(hw, (640.25, 360.25), radius=0.)
    mask = np.zeros(hw, bool)
    assert valid_profile_contributors(result, mask)[0]
    mask[360+dy, 640+dx] = True
    assert not valid_profile_contributors(result, mask)[0]


@pytest.mark.parametrize('uv', [(1279., 360.), (640., 719.)])
def test_image_edge_does_not_supply_four_bilinear_contributors(uv):
    result = profile((720, 1280), uv, radius=0.)
    assert result['valid'][0]
    assert not valid_profile_contributors(result, np.zeros((720, 1280), bool))[0]


@pytest.mark.parametrize('hw', [(0, 1920), (-1, 720), (1080., 1920.),
                              (1080,), (1080, 1920, 3)])
def test_image_dimensions_must_be_positive_integers(hw):
    with pytest.raises(ValueError, match='decoded image'):
        profile(hw, (10., 10.))


def test_mask_rgb_and_cached_field_must_match_decoded_profile_size():
    result = profile((720, 1280), (640, 360))
    with pytest.raises(ValueError, match='mask.*dimensions differ'):
        valid_profile_contributors(result, np.zeros((1080, 1920), bool))
    with pytest.raises(ValueError, match='RGB.*dimensions differ'):
        sample_physical_width_field(np.zeros((1080, 1920, 3), np.uint8), result)
    rgb = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(ValueError, match='field.*dimensions differ'):
        sample_physical_width_field(rgb, result, field=np.zeros((1080, 1920)))
    np.testing.assert_array_equal(sample_physical_width_field(rgb, result), 0.)


def test_physical_reader_rejects_image_mask_mismatch_before_readout():
    from calibration.program_tracking.extract_first_shot_ice import NativeProfiles
    with pytest.raises(ValueError, match='mask.*dimensions differ'):
        NativeProfiles(np.zeros((720, 1280, 3), np.uint8), SOURCE,
                       np.zeros((1080, 1920), bool))


@pytest.mark.parametrize('module_name,class_name,method', [
    ('repair_first_shot_lane1_width', 'PhysicalLaneProfiles', 'sample'),
    ('repair_physical_B1_visibility', 'LocalPhysicalBlue', 'sample_complete_colour_band'),
    ('repair_physical_wide_bands', 'PhysicalBandProfiles', 'sample_complete_colour_band'),
])
def test_existing_readers_pass_actual_4k_rgb_size(monkeypatch, module_name, class_name, method):
    import importlib
    module = importlib.import_module('calibration.program_tracking.'+module_name)
    hw = (2160, 3840)
    query = np.array([[2000., 1500.]])
    query_sizes = []
    sections = []

    def queries(path, guide, depths, image_hw, *args):
        query_sizes.append(tuple(image_hw))
        return query, np.array([[-1., 0.]]), np.array([90.])

    def project(*args, **kwargs):
        section = projected_width_profile(*args, **kwargs)
        assert section['valid'].all()
        sections.append(section)
        return section

    monkeypatch.setattr(module, '_source_queries', queries)
    monkeypatch.setattr(module, 'projected_width_profile', project)
    reader_name = 'whole_colour_bands' if module_name == 'repair_physical_wide_bands' else 'narrow_dark_paint'
    monkeypatch.setattr(module, reader_name, lambda *args, **kwargs: [])
    reader = getattr(module, class_name)(np.zeros((*hw, 3), np.uint8), SOURCE,
                                      np.zeros(hw, bool), PlaneCamera())
    guide = np.array([[2000., 1400.], [2000., 1600.]])
    kwargs = {} if method == 'sample' else dict(spacing=35)
    getattr(reader, method)(PATH, guide, **kwargs)
    assert query_sizes == [hw]
    assert len(sections) == 2
    assert all(section['image_hw'] == hw for section in sections)
    assert reader.frame.binding['native_hw'] == list(hw)
    if module_name == 'repair_physical_wide_bands':
        np.testing.assert_array_equal(reader.last_tangent_valid_counts, 7)


def test_readonly_diagnostic_passes_its_decoded_rgb_size_without_running_script():
    # Evaluate only the existing call expression: importing this historical
    # script would load real files and write diagnostic output at module scope.
    filename = Path(__file__).resolve().parents[1]/'calibration/program_tracking/diagnose_native_B2_1248_material.py'
    tree = ast.parse(filename.read_text(encoding='utf-8'))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'projected_width_profile')
    rgb = np.zeros((720, 1280, 3), np.uint8)
    result = eval(compile(ast.Expression(call), str(filename), 'eval'), dict(
        projected_width_profile=projected_width_profile, camera=PlaneCamera(), path=PATH,
        row=dict(source_evidence=dict(query_uv=[640., 360.])), rgb=rgb))
    assert result['image_hw'] == rgb.shape[:2]
    assert result['valid'].all()
