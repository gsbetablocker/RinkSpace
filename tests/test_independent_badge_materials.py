"""Badge color adapters preserve default red decisions and actual RGB telemetry."""
from pathlib import Path
import json
import sys
import numpy as np

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
if str(PROGRAMME) not in sys.path:sys.path.insert(0,str(PROGRAMME))
import capture_independent_badge as badge
import capture_independent_badge_full as full


def scene(colour):
    rgb=np.full((360,640,3),235,np.uint8);rgb[80:280,140:500]=colour
    # Closed white glyphs cannot become the exterior measurement.
    rgb[130:200,230:248]=235;rgb[130:200,280:298]=235
    contour=np.array([[139.5,79.5],[499.5,79.5],[499.5,279.5],[139.5,279.5],[139.5,79.5]])
    mask=np.zeros(rgb.shape[:2],bool)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='synthetic_badge')
    return rgb,contour,mask,binding


def test_existing_ad9_default_is_identical_to_explicit_red():
    args=scene([200,25,25])
    implicit=badge.read_with_pending_exterior(*args)
    explicit=badge.read_with_pending_exterior(*args,material_signal='red')
    assert implicit[0] and len(implicit[2])>=7
    assert json.dumps(implicit,sort_keys=True)==json.dumps(explicit,sort_keys=True)


def test_blue_badge_reuses_unchanged_public_outer_reader_and_full_proof():
    rgb,search,mask,binding=scene([20,45,170]);original=rgb.copy();search_before=search.copy()
    rows,record,raw,proof=full.read_profile_proof(rgb,search,mask,binding,material_signal='blue')
    assert rows and len(raw)>=7
    assert record['source_component_method']=='blue_source_flood_exterior'
    assert 'reader_color_seed_roi' in proof and 'actual_luminance_profiles' in proof
    assert len(proof['all_RGB_candidate_uv'])>=len(raw)
    for row in rows:
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])
        assert not row['model_UV_synthesized']
    np.testing.assert_array_equal(rgb,original);np.testing.assert_array_equal(search,search_before)