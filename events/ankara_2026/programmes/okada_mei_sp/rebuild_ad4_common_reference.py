"""Reconstruct one rigid print chart from a complete original-source panel.

Photographed high-resolution contours retain their vertices and identities.
Both words are registered into the SAME exposure, whose four material sides
define one chart. No per-letter fit variables, projected UV measurements, or
changes to ice tolerance are introduced.
"""
from pathlib import Path
import argparse, copy, hashlib, json
import cv2
import numpy as np
from programme_source_frames import iter_registration_source_frames
from capture_ad4_source_repair import read, sha, word_reference, source_translation_seed
from capture_independent_ad4_glyph_gap import safe, write
from rinkspace.core.native import NativeCore
from rinkspace.path_policy import guard_project_write
from rinkspace.session.print_patterns import PrintPattern, project
from rinkspace.vision_source_shapes import read_source_material_transition_profiles


def source_panel(image, mask, binding):
    # These search windows were read from the complete original PTS 2388480.
    # Queries never become observations: only actual RGB q50 readings do.
    xx=np.r_[np.arange(248.,720.,24.),np.arange(1128.,1600.,24.)]
    searches={
        'top':(np.c_[xx,np.full(len(xx),335.)],[0.,-1.]),
        'bottom':(np.c_[xx,np.full(len(xx),404.)],[0.,1.]),
        'left':(np.c_[np.full(9,226.),np.linspace(345.,394.,9)],[-1.,0.]),
        'right':(np.c_[np.full(9,1627.),np.linspace(344.,393.,9)],[1.,0.]),
    }
    lines={};proof={}
    for name,(queries,normal) in searches.items():
        normals=np.tile(normal,(len(queries),1))
        result=read_source_material_transition_profiles(np.ascontiguousarray(image[:,:,::-1]),
            binding=binding,search_points_uv=queries,normals=normals,
            search_intervals_px=np.tile([-9.,9.],(len(queries),1)),transition='dark_to_light',
            tangent_half_width_px=1,smoothing_sigma_px=.6,
            engineering_support_margin_px=.25,keep_profiles=True)
        uv=[]
        for r in result['readings']:
            q=np.asarray(r['actual_source_uv'])
            if q.shape!=(2,) or any(r['source_flags'].values()):continue
            x,y=np.rint(q).astype(int)
            if mask[y-3:y+4,x-3:x+4].any():continue
            uv.append(q)
        uv=np.asarray(uv)
        if len(uv)<4:raise ValueError(f'{name}: fewer than four real material readings')
        fit=cv2.fitLine(uv.astype('float32'),cv2.DIST_L1,0,.001,.001).ravel().astype(float)
        line=np.array([-fit[1],fit[0],fit[1]*fit[2]-fit[0]*fit[3]])
        distances=np.abs(np.c_[uv,np.ones(len(uv))]@line)
        if np.quantile(distances,.95)>2.:raise ValueError(f'{name}: source panel side is unresolved')
        lines[name]=line
        proof[name]=dict(actual_source_uv=uv.tolist(),line=line.tolist(),
            source_line_p95_px=float(np.quantile(distances,.95)),public_readings=result['readings'])
    def intersection(a,b):
        p=np.cross(lines[a],lines[b]);return p[:2]/p[2]
    corners=np.array([intersection('top','left'),intersection('top','right'),
        intersection('bottom','right'),intersection('bottom','left')])
    H=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],'float32'),corners.astype('float32'))
    return H,dict(sides=proof,source_corners_TL_TR_BR_BL=corners.tolist(),
        corners_are_intersections_of_measured_finite_material_sides=True,
        direct_source_chart_approximation='planar panel; no claim of surveyed printing master')


def main(a):
    out=guard_project_write(a.output.resolve())
    if (out/'common_source_chart.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    manifest=read(a.prepared/'joint_manifest.json');state=dict(np.load(a.state,allow_pickle=False))
    if sha(a.native_library)!=str(state['native_library_provenance_sha256']):raise ValueError('Frozen DLL differs')
    core=NativeCore(a.native_library)
    frame=next(iter_registration_source_frames('s09',bootstrap_root=a.bootstrap,selected_pts=[2388480]))
    image=frame['image'];mask=frame['source_exclusion_mask']
    binding=dict(source_sha256=frame['source_video_sha256'],source_pts=2388480,
        shot_id='s09',time_base='1/12800',decoded_native_rgb_sha256=frame['source_RGB_sha256'])
    panel_H,panel_proof=source_panel(image,mask,binding)
    write(out/'panel_source_proof.json',panel_proof)
    scene_path=Path(manifest['scene_path']);scene=read(scene_path)
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4)
    old_asset_path=(scene_path.parent/ref['path']).resolve();asset=read(old_asset_path)
    template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    source_assets={p['id']:p for p in template['paths']}
    ankara,_,_,ankara_atlas=word_reference(manifest,state,core)
    jgp_atlas=read(a.jgp_reference)
    jgp_image=cv2.imread(jgp_atlas['source_image'])
    if sha(jgp_atlas['source_image'])!=jgp_atlas['image_sha256']:raise ValueError('JGP source changed')
    register_atlas=copy.deepcopy(jgp_atlas)
    register_atlas['paths']=[p for p in register_atlas['paths']
        if p['proof']['reference_image_has_unoccluded_this_contour']]
    jgp=PrintPattern(register_atlas,jgp_image)
    reference_mask=jgp_atlas.get('reference_exclusion_mask')
    if reference_mask:
        path=reference_mask['path'] if isinstance(reference_mask,dict) else reference_mask
        exclusion=cv2.imread(str(path),0)
        jgp.mask[cv2.dilate((exclusion>0).astype('uint8'),np.ones((7,7),'uint8'))>0]=0
    registrations={};new_uv={};mapped_source={}
    for label,pattern,atlas in [('ANKARA',ankara,ankara_atlas),('JGP',jgp,jgp_atlas)]:
        old=[];pixels=[]
        for p in atlas['paths']:
            q=np.asarray(p['pixels']);u=np.asarray(source_assets[p['path_id']]['material_uv'])
            q=q[:len(u)]
            if len(q)!=len(u):raise ValueError('Source vertex index binding differs')
            old.extend(u);pixels.extend(q)
        to_old,_=cv2.findHomography(np.asarray(pixels),np.asarray(old),0)
        seed=panel_H@to_old
        seed_proof=None
        if label=='ANKARA' and a.word_navigation is not None:
            for line in a.word_navigation.open(encoding='utf8'):
                record=json.loads(line)
                if record['pts']==2388480:
                    saved=record['word_registration']
                    if saved['status']!='source_pattern_registered':raise ValueError('Navigation reference is unresolved')
                    seed=np.asarray(saved['H'])
                    seed_proof=dict(path=str(a.word_navigation),sha256=sha(a.word_navigation),
                        same_exposure_photographic_registration_navigation_only=True)
                    break
            if seed_proof is None:raise ValueError('Same source exposure navigation missing')
        translation=source_translation_seed(pattern,image,seed)
        if translation is not None:seed=translation[0]
        registration=pattern.register(image,mask.astype('uint8'),seed)
        registration['navigation_source']=seed_proof
        registration['photographic_translation_navigation']=None if translation is None else translation[1]
        if registration['status']!='source_pattern_registered':
            write(out/f'{label}_failed_registration.json',registration)
            raise ValueError(label+': complete photographic word did not register')
        registrations[label]=registration
        H=np.asarray(registration['H'])
        for p in atlas['paths']:
            count=len(source_assets[p['path_id']]['material_uv'])
            q=project(H,np.asarray(p['pixels'])[:count])
            mapped_source[p['path_id']]=q
            new_uv[p['path_id']]=project(np.linalg.inv(panel_H),q)
    proof=dict(schema='rinkspace.ad4_common_source_chart.v1',source_binding=binding,
        source_BGR_sha256=frame['source_BGR_sha256'],original_asset=dict(path=str(old_asset_path),sha256=sha(old_asset_path)),
        original_state=dict(path=str(a.state),sha256=sha(a.state)),panel_source_proof=str(out/'panel_source_proof.json'),
        panel_H_from_unit_chart=panel_H.tolist(),registrations=registrations,
        JGP_source_reference=dict(path=str(a.jgp_reference),sha256=sha(a.jgp_reference)),
        no_individual_glyph_variables=True,source_observations_modified=False,
        measurement_scope='source template reconstruction; not independent validation',user_acceptance='pending')
    for p in template['paths']:
        if p['id'] not in new_uv:continue
        p['material_uv']=new_uv[p['id']].tolist()
        p['common_source_chart_revision']=dict(reference_pts=2388480,
            original_vertex_count=len(p['material_uv']),source_vertex_order_preserved=True,
            reference_source_vertices=mapped_source[p['id']].tolist(),evidence=str(out/'common_source_chart.json'))
    group=template['internal_geometry']['groups'][0]
    vertices=np.concatenate(list(new_uv.values()))
    group['reference_bounds']=np.r_[vertices.min(0),vertices.max(0)].tolist()
    template['common_source_chart_revision']=proof
    for block in asset['parameter_blocks']:
        if block['id']==group['parameter_block_id']:block['values']=[0.]*10
    guard_project_write(a.asset_output.resolve());guard_project_write(a.scene_output.resolve())
    if a.asset_output.exists() or a.scene_output.exists():raise FileExistsError('Preserve previous model candidates')
    write(a.asset_output,asset)
    ref['path']=str(a.asset_output.resolve());ref['sha256']=sha(a.asset_output)
    ref['template_acceptance']=dict(status='source_chart_rebuilt_pending_joint_and_user_review',
        authority='current_user_requested_Ad4_repair',source_reference_pts=2388480)
    write(a.scene_output,scene);write(out/'common_source_chart.json',proof)
    picture=image.copy()
    for q in mapped_source.values():cv2.polylines(picture,[np.rint(q).astype('int32')],True,(0,255,0),1,cv2.LINE_AA)
    for s in panel_proof['sides'].values():
        for q in s['actual_source_uv']:cv2.circle(picture,tuple(np.rint(q).astype(int)),2,(0,0,255),-1)
    cv2.imwrite(str(out/'common_reference_overlay.png'),picture)
    print(json.dumps(dict(asset=str(a.asset_output),scene=str(a.scene_output),paths=len(new_uv),
        correlations={k:v['correlation'] for k,v in registrations.items()},
        panel_corners=panel_proof['source_corners_TL_TR_BR_BL']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','state','native-library','bootstrap','jgp-reference','output','asset-output','scene-output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--word-navigation',type=Path)
    main(p.parse_args())
