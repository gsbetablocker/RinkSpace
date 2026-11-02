"""Reconstruct the fixed Ad4 chart after the complete native camera inverse.

Source pixels stay measurements. The inverse is conditional on one frozen
camera state; it removes its lens and rolling-shutter map before panel-side
rectification. No vertex or new shape parameter is added to joint fitting.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import cv2
import numpy as np

import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _write_prepared
from build_ad3_completion import layout_without_content_identity
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.print_assets import native_instance_parameters
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.session.print_patterns import project
from rinkspace.path_policy import guard_project_write


def inverse_plane(projector, pixels, initial, *, tolerance_px=1e-7):
    """Square forward-map inversion, never a least-squares observation fit."""
    pixels=np.asarray(pixels,float);q=np.asarray(initial,float).copy()
    if q.shape!=pixels.shape or q.ndim!=2 or q.shape[1]!=2:
        raise ValueError('Plane inverse needs matching finite two-coordinate arrays')
    h=1e-5;iterations=0
    while True:
        pred=projector(q);r=pred-pixels;error=np.linalg.norm(r,axis=1)
        if not np.isfinite(error).all():raise ValueError('Nonfinite plane inverse')
        if float(error.max())<=tolerance_px:return q,dict(iterations=iterations,max_roundtrip_px=float(error.max()))
        # This guard detects a broken/noninvertible forward map; it is not a
        # fitting budget. A well-conditioned local inverse converges in 2-4 steps.
        if iterations>=25:raise ValueError('Native plane inverse did not converge')
        columns=[]
        for axis in range(2):
            shift=np.zeros(2);shift[axis]=h
            columns.append((projector(q+shift)-projector(q-shift))/(2*h))
        jac=np.stack(columns,axis=2)
        if np.min(np.linalg.svd(jac,compute_uv=False)[:,-1])<1e-7:
            raise ValueError('Singular material-to-source inverse')
        delta=np.linalg.solve(jac,r[...,None])[...,0]
        scale=np.ones(len(q));next_q=q-delta
        next_error=np.linalg.norm(projector(next_q)-pixels,axis=1)
        while np.any(next_error>error):
            bad=next_error>error;scale[bad]*=.5
            if np.min(scale)<2**-20:raise ValueError('Inverse Newton cannot reduce roundtrip error')
            next_q=q-scale[:,None]*delta
            next_error=np.linalg.norm(projector(next_q)-pixels,axis=1)
        q=next_q;iterations+=1


def main(a):
    for path in (a.output,a.asset_output,a.scene_output):
        guard_project_write(path.resolve())
        if path.exists():raise FileExistsError(path)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    points=prepared.npz(a.prepared/'point_observations.npz');state=prepared.npz(a.state)
    scene_path=Path(m['scene_path']);scene=prepared.read(scene_path)
    old_model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    prepared.validate_snapshot(state,old_model.initial_state.layout,mapping,m,points)
    if prepared.digest(a.native_library)!=str(state['native_library_provenance_sha256']):raise ValueError('Frozen DLL differs')
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4)
    asset_path=(scene_path.parent/ref['path']).resolve();asset=prepared.read(asset_path)
    template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    chart=template['common_source_chart_revision'];pt=chart['source_binding']['source_pts']
    proof=prepared.read(chart['panel_source_proof']);old_H=np.asarray(chart['panel_H_from_unit_chart'])
    params=ParameterState(old_model.initial_state.layout,state['scene_values'])
    board=params.apply_scene(scene)['board'];core=NativeCore(a.native_library)
    ip=native_instance_parameters(asset,'print_instance/Ad4',parameter_state=params)
    basis=np.asarray(core.sample_path(board,5,ip,[[0.,0.],[1.,0.],[0.,1.]],[0.,.5,1.]))
    origin=basis[0];axes=np.array([basis[1]-origin,basis[2]-origin])
    grid=np.array([[0,0],[1,0],[1,1],[0,1],[.2,.4],[.8,.7]],float)
    actual=np.asarray(core.sample_path(board,5,ip,grid,np.linspace(0,1,len(grid))))
    plane_error=float(np.max(abs(actual-(origin+grid@axes))))
    if plane_error>1e-12:raise ValueError('Ad4 is not one affine board plane; use a curved-surface inverse')
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    shot=next(s for s in m['shots'] if s['id']=='s09')
    clock=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
    def forward(q):
        result=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,origin+q@axes)
        if np.any(result['status']):raise ValueError('Native source reconstruction projection invalid')
        return result['uv']
    def invert(pixels):return inverse_plane(forward,pixels,project(np.linalg.inv(old_H),np.asarray(pixels)))
    lines={};sides={};scale=np.asarray(ip[3:]);roundtrips=[]
    for name,side in proof['sides'].items():
        pixels=np.asarray(side['actual_source_uv']);q,roundtrip=invert(pixels);roundtrips.append(roundtrip)
        metric=q*scale
        fit=cv2.fitLine(metric.astype('float32'),cv2.DIST_L1,0,1e-10,1e-10).ravel().astype(float)
        line=np.array([-fit[1],fit[0],fit[1]*fit[2]-fit[0]*fit[3]])
        lines[name]=line;sides[name]=dict(source_uv=pixels.tolist(),inverse_material_uv=q.tolist(),line_metric=line.tolist())
    corners=[]
    for names in [('top','left'),('top','right'),('bottom','right'),('bottom','left')]:
        cross=np.cross(lines[names[0]],lines[names[1]])
        corners.append(cross[:2]/cross[2]/scale)
    corners=np.asarray(corners)
    canonical=cv2.getPerspectiveTransform(corners.astype('float32'),np.array([[0,0],[1,0],[1,1],[0,1]],'float32'))
    revised={};deltas={}
    for path in template['paths']:
        old=path.get('common_source_chart_revision')
        if old is None or path.get('source_support_parent'):continue
        pixels=np.asarray(old['reference_source_vertices']);q,roundtrip=invert(pixels);roundtrips.append(roundtrip)
        uv=project(canonical,q);previous=np.asarray(path['material_uv'])
        if uv.shape!=previous.shape:raise ValueError('Original contour vertices changed')
        revised[path['id']]=uv;path['material_uv']=uv.tolist()
        deltas[path['id']]=dict(vertices=len(uv),max_material_change=np.max(abs(uv-previous),axis=0).tolist())
        path['camera_rectification']=dict(source_pts=pt,original_common_chart_vertices=previous.tolist(),native_inverse=True)
    if not revised:raise ValueError('No photographed Ad4 contours to rectify')
    for path in template['paths']:
        parent=path.get('source_support_parent')
        if parent:
            path['material_uv']=revised[parent][path['source_support']['parent_vertex_indices']].tolist()
    group=template['internal_geometry']['groups'][0]
    vertices=np.concatenate(list(revised.values()));group['reference_bounds']=np.r_[vertices.min(0),vertices.max(0)].tolist()
    shape=group['parameter_block_id'];block=params.blocks[shape]
    numeric=dict(state);numeric['scene_values']=state['scene_values'].copy()
    numeric['scene_values'][block['offset']:block['offset']+block['size']]=0.
    for b in asset['parameter_blocks']:
        if b['id']==shape:b['values']=[0.]*b['size']
    receipt=dict(schema='rinkspace.ad4_camera_rectified_source_chart.v1',
        source_reference_pts=pt,parent_prepared=str(a.prepared),parent_state=dict(path=str(a.state),sha256=prepared.digest(a.state)),
        parent_asset=dict(path=str(asset_path),sha256=prepared.digest(asset_path)),
        native_library=dict(path=str(a.native_library),sha256=prepared.digest(a.native_library)),
        source_panel_proof=chart['panel_source_proof'],source_corners_material=corners.tolist(),
        material_to_canonical_H=canonical.tolist(),native_plane_error=plane_error,
        max_native_inverse_roundtrip_px=max(x['max_roundtrip_px'] for x in roundtrips),
        contour_changes=deltas,source_sides=sides,shape_block_reset=shape,
        source_UV_PTS_and_identity_unchanged=True,no_new_geometry_parameters=True,
        conditional_camera_inverse=True,final_validation_performed=False)
    template['camera_rectification_revision']={k:v for k,v in receipt.items() if k not in ('source_sides','contour_changes')}
    prepared.write(a.asset_output,asset)
    ref.update(path=str(a.asset_output.resolve()),sha256=prepared.digest(a.asset_output))
    prepared.write(a.scene_output,scene)
    model=SceneParameterModel(scene,a.scene_output,optimize_scene=True)
    if layout_without_content_identity(model.initial_state.layout)!=layout_without_content_identity(old_model.initial_state.layout):
        raise ValueError('Source rectification changed parameter layout')
    native=NativeSceneGPU(a.scene_output,core,parameter_state=ParameterState(model.initial_state.layout,numeric['scene_values']))
    mapped=deepcopy(mapping);mapped['scene_revision']=native.scene_revision
    target=deepcopy(m);target.update(scene_path=str(a.scene_output.resolve()),scene_file_sha256=prepared.digest(a.scene_output),scene_revision=native.scene_revision)
    updated,mapped=_write_prepared(a.output,points,state['stations'],m['shots'],numeric,target,mapped,model,receipt,link_input=a.prepared/'point_observations.npz')
    if 'spatial_refinement_preparation' in m:updated['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:updated.pop('spatial_refinement_preparation',None)
    updated['camera_rectified_ad4_chart']=str(a.output/'rectification.json')
    prepared.write(a.output/'joint_manifest.json',updated);prepared.write(a.output/'rectification.json',receipt)
    saved=prepared.npz(a.output/'initial_pixels.npz')
    prepared.validate_snapshot(saved,model.initial_state.layout,mapped,updated,points)
    for key in ('rig','controls','stations','board','layout'):
        if not np.array_equal(saved[key],state[key]):raise ValueError('Protected state changed: '+key)
    print(json.dumps(dict(output=str(a.output),changed_contours=len(revised),plane_error=plane_error,max_roundtrip_px=receipt['max_native_inverse_roundtrip_px'],max_material_delta=np.max([x['max_material_change'] for x in deltas.values()],axis=0).tolist())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','native-library','asset-output','scene-output','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
