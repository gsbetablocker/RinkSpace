import cv2
import numpy as np

from rinkspace.session.material_rings import source_ring_hypotheses, nested_source_rings, source_ring_color_evidence
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader


def test_nested_source_rings_reject_streak_and_keep_original_readouts():
    image = np.full((500, 800, 3), 230, np.uint8)
    definitions, projections = [], {}
    theta = np.linspace(0., 2.*np.pi, 361)
    for role, axes, level, transition in [
        ('outer', (240, 95), 120, 'dark_to_light'),
        ('middle', (165, 65), 230, 'light_to_dark'),
        ('inner', (85, 33), 130, 'dark_to_light')]:
        cv2.ellipse(image, (400, 250), axes, 0., 0., 360., (level,)*3, -1)
        curve = 'house:'+role
        guide = np.c_[414+axes[0]*np.cos(theta), 244+axes[1]*np.sin(theta)]
        definitions.append(dict(curve_id=curve, feature_id='house', world_curve_role=role,
            operator='material_interface', material_signal='dark', material_transition=transition,
            native_path_id=curve, xyz=np.zeros((len(theta), 3)), center_xyz=np.zeros(3)))
        projections[curve] = dict(uv=guide, valid=np.ones(len(theta), bool), centre_uv=np.array([414.,244.]))
    cv2.line(image, (80, 365), (720, 365), (90, 90, 90), 3)
    image = cv2.GaussianBlur(image, (7,7), 1.5)
    reader = IceProgrammeSourceReader(None, source_sha256='0'*64, time_base='1/12800',
        native_paths=definitions, sample_spacing_px=12., search_radius_px=40.)
    measurement = reader.read(image, 100, None, np.zeros(image.shape[:2], bool),
        shot_id='test', native_projections=projections)
    hypotheses = {p['world_curve_role']:source_ring_hypotheses(image, measurement, i)
                  for i,p in enumerate(definitions)}
    result = nested_source_rings(hypotheses)
    assert result['status'] == 'source_nested_stack_hypothesis'
    for role, ids in result['selected'].items():
        assert len(ids) > 15
        xy = measurement['candidate_side_quantiles_uv'][ids, 0, 1]
        axes = {'outer':(240.,95.), 'middle':(165.,65.), 'inner':(85.,33.)}[role]
        level = ((xy-[400.,250.]) / axes)**2
        assert np.max(np.abs(level.sum(1)-1.)) < .10
        assert np.all(np.abs(xy[:,1]-365) > 8.)
    assert not result['physical_instance_confirmed']
    # Ellipse-shaped neutral shading is not evidence of this event's coloured
    # targets, even when it perfectly satisfies geometric nesting.
    assert not source_ring_color_evidence(image, measurement, result['selected'])['selected']
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (400,250), (240,95), 0., 0., 360., 255, -1)
    cv2.ellipse(mask, (400,250), (165,65), 0., 0., 360., 0, -1)
    mask = cv2.GaussianBlur(mask, (7,7), 1.5).astype(float)/255.
    coloured = image.astype(float)
    coloured[:,:,0] += 10.*mask
    evidence = source_ring_color_evidence(coloured, measurement, result['selected'])
    assert evidence['status'] == 'source_colored_ring_stack'
    assert evidence['colors']['outer'] == 'red'


def test_nested_source_rings_never_fill_missing_material_layer():
    assert nested_source_rings({'inner':[], 'middle':[], 'outer':[]})['selected'] == {}
