from pathlib import Path
import copy,json,sys
import cv2,numpy as np,pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'events/ankara_2026/programmes/okada_mei_sp')]
from calibration.program_tracking.audit_first_shot_circle_material_identity import centreline_house_conflicts,observed_house_fill
from rinkspace.vision_board_reader import source_front_face_top
from ad_entity_measure_s01 import current_glyph_material_pair


def test_photographed_word_palette_does_not_require_saturated_artwork_blue():
    from ad_entity_measure_s01 import photographed_material_colors,physical_color
    source=np.full((60,80,3),190,np.uint8);source[10:50,20:60]=[105,100,97]
    q=np.array([[20.,25.],[20.,35.],[60.,25.],[60.,35.]])
    assert not physical_color('Ad8.print.glyph_W.boundary.0000',source,q).any()
    assert photographed_material_colors(dict(color=source),np.eye(3),source,q).all()
    wrong=source.copy();wrong[10:50,20:60]=[30,40,190]
    assert not photographed_material_colors(dict(color=source),np.eye(3),wrong,q).any()


def test_real_material_ownership_checks_straight_lines_without_changing_ring_reads():
    features={};entities={}
    for role,radius in [('outer',40),('middle',30),('inner',20),('center',10)]:
        name=f'ice.R3:{role}';rows=[]
        for i,t in enumerate(np.linspace(0,2*np.pi,64,endpoint=False)):
            rows.append(dict(uv=[70+radius*np.cos(t),70+radius*np.sin(t)],theta=float(t),source_row_id=f'{name}/{i}'))
        features[name]=dict(native_path_id=name,rows=rows);entities[name]=dict(measurement='contour')
    name='ice.lane3:edge0'
    features[name]=dict(native_path_id=name,rows=[dict(uv=[70,35],source_row_id='false_fill'),dict(uv=[70,15],source_row_id='independent_line')])
    entities[name]=dict(measurement='centreline');before=copy.deepcopy(features)
    bad,_,_=centreline_house_conflicts(features,entities)
    assert {v['source_row_id']for v in bad}=={'false_fill'}
    assert features==before


def test_missing_house_arc_does_not_create_a_material_patch_on_opposite_side():
    features={}
    for role,radius in [('outer',40),('middle',30)]:
        rows=[dict(uv=[70+radius*np.cos(t),70+radius*np.sin(t)],theta=float(t))for t in np.linspace(0,np.pi/2,10)]
        features[f'ice.R3:{role}']=dict(rows=rows)
    mask,_=observed_house_fill(features,shape=(140,140))
    assert mask[94,94] and not mask[45,45]


def test_ad_search_face_starts_below_blue_cap_and_respects_occlusion():
    rgb=np.full((140,200,3),210,np.uint8);rgb[20:29]=[30,55,150];rgb[45:100,30:170]=[130,40,125]
    hint=np.c_[np.arange(10,190,8),np.full(23,20)]
    binding=dict(source_sha256='synthetic-test',source_pts=256,time_base='1/12800',shot_id='test')
    lower,receipt=source_front_face_top(rgb,binding,hint,np.zeros((140,200),bool))
    assert len(lower)>10 and np.all((lower[:,1]>27)&(lower[:,1]<30))
    assert receipt['creates_observations']is False
    blocked,_=source_front_face_top(rgb,binding,hint,np.ones((140,200),bool))
    assert len(blocked)==0


def test_background_texture_is_not_a_letter_and_actual_letter_edge_is_kept():
    gray=np.full((120,120),90,np.uint8);gray[:,30:35]=94;gray[20:100,50:75]=200
    mask=np.zeros_like(gray);context=np.array([[45,15],[80,15],[80,105],[45,105]])
    wrong=current_glyph_material_pair(gray,mask,context,[30,60],[1,0],[-1,0,1])
    right=current_glyph_material_pair(gray,mask,context,[50,60],[1,0],[-1,0,1])
    assert not wrong['resolved'] and right['resolved']
    mask[55:65,45:55]=255
    assert not current_glyph_material_pair(gray,mask,context,[50,60],[1,0],[-1,0,1])['resolved']


def test_actual_s02_I_background_false_reads_are_reproduced():
    base=Path('D:/CodexData/RinkSpace_Data/runs/native_points_s02_20261027')
    source=base/'source_frames/710144.png';records=base/'source_repair/peer_review/I_710144_detail.json'
    if not source.exists()or not records.exists():pytest.skip('Local official original source is not public')
    image=cv2.imread(str(source));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows=json.loads(records.read_text(encoding='utf-8'));context=np.array([v['uv']for v in rows])
    vis=next(v for v in json.loads((base/'evidence/visibility.json').read_text())['frames']if v['pts']==710144)
    mask=cv2.imread(vis['mask_path'],0);results={}
    for row in rows:
        sp=row['source_record']['source_profile'];query=sp.get('search_uv',row.get('original_uv',row['uv']))
        results[row['source_row']]=current_glyph_material_pair(gray,mask,context,query,sp['normal'],sp['offsets'])['resolved']
    assert all(not results[i]for i in [94887,94888,94890,94891])
    assert all(results[i]for i in [94866,94867,94870,94873])


def test_actual_ad5_native_graphic_border_rereads_below_blue_rail(tmp_path):
    import repair_points_s02 as repair
    if not(repair.old.OUT/'source_frames/585216.png').exists():pytest.skip('Local official original source is not public')
    # This regression rereads authenticated RGB and source packets only; it
    # does not use a navigation camera or the current native scene. Keep it
    # independent of the historical camera manifest's obsolete scene hash.
    repair.nonice([585216],output_dir=tmp_path,
                  prepare_context=lambda: None,verify_context=lambda: None)
    packet=json.loads((tmp_path/'nonice/585216.json').read_text(encoding='utf-8'))
    group=packet['borders']['print_instance/Ad5::Ad5.top_edge/path0']
    assert group['native_material_role']=='logo_outer_boundary'
    assert group['whole_panel_role_required']is False
    assert len(group['rows'])>0
    cap=np.asarray(packet['board_cap_domain']['lower_blue_cap_uv']);cap=cap[np.argsort(cap[:,0])]
    for row in group['rows']:
        x,y=row['uv']
        assert y>np.interp(x,cap[:,0],cap[:,1])+2
        assert not any(row['source_evidence']['source_flags'].values())


def test_actual_ad5_wrong_lower_border_and_correct_green_border_are_distinguished():
    import repair_s02_source_identities as repair
    if not(repair.OUT/'nonice/584704.json').exists():pytest.skip('Local current source repair evidence is not public')
    _,by,_,vis=repair.inputs();pt=584704;packet=repair.read(repair.OUT/f'nonice/{pt}.json')
    supports=repair.read(repair.BEFORE/f'support/{pt}.json');pid='print_instance/Ad5::Ad5.bottom_edge/path0'
    group=[r for r in by[pt]if r['native_path_id']==pid]
    cap=np.array(packet['board_cap_domain']['lower_blue_cap_uv']);cap=cap[np.argsort(cap[:,0])]
    contact=np.array([r['uv']for r in supports['board/continuous/bottom']['rows']]);contact=contact[np.argsort(contact[:,0])]
    bad=repair.border_failures(group,packet['borders'][pid],cap,cv2.imread(vis[pt]['image']),contact)
    assert len(bad)==1 and len(group)>1
    wrong=next(iter(bad.values()))
    assert wrong['reason']=='ad_pixel_on_actual_yellow_foot_ice_contact_not_ad_material'
    assert np.linalg.norm(np.array(wrong['uv'])-[1731.4623195958186,466.83726486390066])<.001


def test_blue_cap_exit_cannot_be_the_stronger_white_to_red_advertisement_edge():
    rgb=np.full((140,200,3),210,np.uint8);rgb[20:29]=[70,80,130];rgb[40:100]=[220,30,30]
    hints=np.c_[np.arange(10,190,8),np.full(23,20)]
    lower,_=source_front_face_top(rgb,dict(source_sha256='test',source_pts=256,time_base='1/12800',shot_id='test'),hints,np.zeros((140,200),bool))
    assert len(lower)>10 and np.all((lower[:,1]>27)&(lower[:,1]<30))
