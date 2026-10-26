"""Complete source evidence for retained entities absent from earlier navigation."""
from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
import ad_entity_measure_s01 as measure
b=r.b;OUT=r.OUT

def run():
    ats=b.read(b.OUT/'atlases.json')['atlases'];doc=b.read(OUT/'source_identity_charts.json');evidence=[]
    # This historic plate is a median of real, rectified RGB frames. It is only
    # a source-pattern guide; no median pixel is admitted as an observation.
    folder=b.DATA/'runs/okada_joint6_1021b/ad_shape_repair';plate=b.read(folder/'ad6_native_plate.json');entry=next(x for x in plate if x['frame']==650);source650=b.read(b.OUT/'visibility.json')['frames'][650]
    source_record=next(x for x in r.VIS.values() if Path(x['image']).name=='000650.png')
    expected=cv2.imread(entry['source'])
    if hashlib.sha256(expected.tobytes()).hexdigest()!=source_record['source_BGR_sha256']:raise ValueError('Ad6 composite source is not authenticated exposure 650')
    H=np.linalg.inv(np.asarray(entry['H']));doc['records']['74']={}
    lineage=dict(kind='actual_RGB_median_reference_for_navigation_only',manifest=str(folder/'ad6_native_plate.json'),manifest_sha256=b.sha(folder/'ad6_native_plate.json'),creator=str(folder/'ad6_native_plate.py'),creator_sha256=b.sha(folder/'ad6_native_plate.py'),plate_sha256=ats[74]['image_sha256'],original_sources=[dict(path=x['source'],sha256=b.sha(x['source']),rectification=x['H']) for x in plate],authentic_reference_pts=source_record['pts'],median_reference_is_never_observed_UV=True,native_geometry_used=False)
    for pt,c in doc['records']['9'].items():
        HH=np.asarray(c['H'])@H
        if r.geometry(HH,r.reference(ats[74])) is None:continue
        doc['records']['74'][pt]=dict(H=HH.tolist(),atlas_id=74,pts=int(pt),method='actual_RGB_composite_guide_via_authenticated_source_exposure',source_lineage=lineage,via_atlas=9,source_BGR_sha256=r.VIS[int(pt)]['source_BGR_sha256'],native_geometry_used=False)
    evidence.append(dict(atlas=74,frames=len(doc['records']['74']),lineage=lineage))
    # The unobscured D is defined by a different real photograph of the same
    # print. PRIX and neighboring N support this source-to-source bridge.
    cs=b.read(OUT/'photo_bridge_69_4.json');cs=[x for x in cs if x['pattern_score'] and x['geometry']]
    bridge=max(cs,key=lambda x:x['pattern_score']['pattern_correlation_median']);doc['records']['69']={}
    for pt,c in doc['records']['4'].items():
        HH=np.asarray(c['H'])@np.asarray(bridge['H'])
        doc['records']['69'][pt]=dict(H=HH.tolist(),atlas_id=69,pts=int(pt),method='actual_unoccluded_letter_reference_join',reference_join=bridge,via_atlas=4,source_BGR_sha256=r.VIS[int(pt)]['source_BGR_sha256'],native_geometry_used=False)
    evidence.append(dict(atlas=69,frames=len(doc['records']['69']),bridge=bridge))
    # Ad4 left purple/white interface is plainly identifiable in the authentic
    # ANKARA source frame. Read that interface; preserve its partial-arc status.
    pt=368640;im,gray,mask=r.current(pt);query=np.column_stack([np.full(64,332.),np.arange(405.,533.,2.)]);n=np.tile([-1.,0.],(len(query),1));f=r.VIS[pt]
    profiles=measure.read_source_material_transition_profiles(np.ascontiguousarray(im[:,:,::-1]),binding=dict(source_sha256=f['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id='s01',event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080),search_points_uv=query,normals=n,search_intervals_px=np.tile([-22.,22.],(len(query),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
    good=[p for p in profiles if not any(p['source_flags'].values())];q=np.array([p['actual_source_uv'] for p in good])
    if len(q)<20:raise ValueError('Actual Ad4 left interface remains unresolved')
    path=dict(path_id='Ad4.left_vertical/path0',pixels=q.tolist(),contour_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),proof=dict(kind='actual_current_source_partial_material_interface',source_pts=pt,source_BGR_sha256=f['source_BGR_sha256'],source_path_role='left outer purple print boundary; ANKARA lies to its right',not_a_fixed_corner_reference=True,original_source_profile_count=len(good)))
    extra=dict(id=81,ad=4,image=f['image'],image_sha256=b.sha(f['image']),paths=[path],partial_source_reference=True)
    b.write(OUT/'additional_source_atlases.json',dict(atlases=[extra],native_geometry_modified=False))
    doc['records']['81']={pt:dict(c,atlas_id=81,method='authenticated_same_exposure_partial_interface_reference',source_profile_authority=path['proof']) for pt,c in doc['records']['6'].items()}
    evidence.append(dict(atlas=81,frames=len(doc['records']['81']),reference=extra))
    b.write(OUT/'missing_retained_entity_sources.json',dict(evidence=evidence));b.write(OUT/'source_identity_charts.json',doc)
    print(json.dumps(dict(added_chart_frames={str(i):len(doc['records'][str(i)]) for i in [69,74,81]},source_material_guide_samples=len(q))),flush=True)
if __name__=='__main__':run()
