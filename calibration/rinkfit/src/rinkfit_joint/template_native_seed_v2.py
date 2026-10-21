"""One provisional native scene from identified historical material primitives.

Only the geometric scene is changed.  No old camera, K/R/C or sampled
same-station/same-angle source correspondence is used.  Carrier/conic and
finite-support coefficients are historical numerical initializers, never new
source observations.  All shots, including S01, consume the same changed scene.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from .io_utils import read_json, write_json, file_sha256
from .joint_model import as_event_view, validate_physical_domain
from rinkfit_start.native_scene import NativeScene, camera_from_event


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha256(path))


def bound(reference):
    path = Path(reference.get('file', reference.get('path')))
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Changed geometry/identity evidence: '+str(path))
    return path


def stats(values, valid=None):
    values = np.asarray(values, float).reshape(-1)
    good = np.isfinite(values)
    if valid is not None:
        good &= np.asarray(valid, bool).reshape(-1)
    v = values[good]
    return dict(rows=len(values), valid=int(good.sum()), invalid=int((~good).sum()),
        rms=float(np.sqrt(np.mean(v*v))) if len(v) else None,
        median=float(np.median(v)) if len(v) else None,
        p95=float(np.quantile(v,.95)) if len(v) else None,
        maximum=float(v.max()) if len(v) else None)


def _line(points):
    q = np.asarray(points, float).reshape(2,2)
    line = np.cross(np.r_[q[0],1.], np.r_[q[1],1.])
    line /= np.linalg.norm(line[:2])
    if line[np.flatnonzero(np.abs(line)>1e-14)[0]] < 0:
        line = -line
    return line


def _conic(center, radius):
    x,y = center
    return np.array([[1.,0.,-x],[0.,1.,-y],[-x,-y,x*x+y*y-radius*radius]])


def _old_curve(scene, item, length, origin):
    feature = next(f for f in scene['ice_features'] if f['id']==item['feature_id'])
    role = item['world_curve_role']
    if item['current_path_kind']==1:
        rings = {r['role']:r for r in feature.get('rings',[])}
        ring = rings.get(role,feature)
        return dict(center=(np.asarray(ring.get('center_xyz',feature['center_xyz']))-origin)[:2]/length,
                    radius=float(ring['radius'])/length)
    if role.startswith('edge'):
        edge=int(role[4:]);polygon=np.asarray(feature['polygon_xyz'])
        endpoints=polygon[[edge,(edge+1)%len(polygon)]]
    else:
        endpoints=np.asarray(feature['segments_xyz'][int(role.replace('segment',''))])
    if endpoints.shape!=(2,3) or np.any(endpoints[:,2]!=0):
        raise ValueError('Only one explicitly named finite ice segment is supported')
    return dict(endpoints=(endpoints-origin)[:,:2]/length)


class NativeBoardInverse:
    """Invert actual XYZ on the chosen common board, without borrowing stations."""
    def __init__(self,scene):
        self.scene=scene
        self.samples=np.linspace(0.,scene.perimeter,32768,endpoint=False)
        self.tree=cKDTree(scene.board(self.samples,0.)[:,:2])

    def __call__(self,xy):
        xy=np.asarray(xy,float)
        _,indices=self.tree.query(xy)
        step=self.scene.perimeter/len(self.samples)
        stations=[];errors=[]
        for point,index in zip(xy,np.atleast_1d(indices)):
            seed=self.samples[index]
            def distance(s):
                delta=self.scene.board(np.array([s]),0.)[0,:2]-point
                return float(delta@delta)
            result=minimize_scalar(distance,bounds=(seed-2*step,seed+2*step),
                method='bounded',options={'xatol':1e-14})
            stations.append(result.x%self.scene.perimeter)
            errors.append(np.sqrt(result.fun))
        return np.asarray(stations),np.asarray(errors)


def build_candidate(parent_file, audit_file, crosswalk_file, ad_crosswalk_file):
    parent=read_json(parent_file);result=copy.deepcopy(parent)
    audit=read_json(audit_file);cross=read_json(crosswalk_file)
    scene_file=bound(cross['old_scene']);old=read_json(scene_file)
    role_file=bound(cross['role_contract'])
    ad_cross=read_json(ad_crosswalk_file)
    if len(cross['paired_paths'])!=len(audit['curves']):
        raise ValueError('Incomplete historical named primitive crosswalk')
    length=float(audit['normalization']['L']);origin=np.asarray(audit['normalization']['origin'],float)
    if np.any(origin!=0.) or not np.allclose(old['parameters']['bounds'],[-length/2,-old['parameters']['bounds'][3],length/2,old['parameters']['bounds'][3]],atol=1e-12,rtol=0):
        raise ValueError('Native centered right-handed L=1 board needs an explicit supported gauge')
    if result['coordinate_gauge']['handedness']!='right' or result['coordinate_gauge']['ice_z']!=0:
        raise ValueError('Different native world axes')
    curve=old['parameters']['corner_curve'];params=old['parameters']
    scene5=[(params['bounds'][3]-params['bounds'][1])/length,
        curve['tangent_extent_xy_scene_units'][0]/length,
        curve['tangent_extent_xy_scene_units'][1]/length,
        curve['exponent'],params['board_height']/length]
    if not np.allclose(scene5,audit['old_normalized_scene5'],atol=1e-14,rtol=0):
        raise ValueError('Audit and original scene geometry differ')
    result['blocks'][result['scene_block']]['values']=scene5
    audit_lookup={r['curve_id']:r for r in audit['curves']}
    lookup={m['label']:i for i,m in enumerate(result['materials'])}
    assigned={};records=[]
    def assign(block,index,value):
        key=(block,index);value=float(value)
        if key in assigned and abs(assigned[key]-value)>1e-13:
            raise ValueError('Two identified primitives demand incompatible shared coefficients')
        assigned[key]=value;result['blocks'][block]['values'][index]=value
    for item in cross['paired_paths']:
        evidence=audit_lookup[item['curve_id']]
        mi=lookup[evidence['current_material_label']];pi=item['current_path_index']
        path=result['materials'][mi]['paths'][pi]
        if path['kind']!=item['current_path_kind'] or path['id']!=item['current_path_id']:
            raise ValueError('Named source primitive topology differs: '+item['curve_id'])
        oldcurve=_old_curve(old,item,length,origin);bi=path['geometry_block']
        before=np.asarray(parent['blocks'][bi]['values'],float)
        if path['kind']==0:
            if not np.allclose(before[:4].reshape(2,2),evidence['current_endpoints_L'],atol=1e-13,rtol=0):
                raise ValueError('Parent line differs from the audited template')
            for index,value in enumerate(oldcurve['endpoints'].reshape(-1)):
                assign(bi,index,value)
            rec=dict(carrier=_line(oldcurve['endpoints']).tolist(),
                historical_finite_endpoints_L=oldcurve['endpoints'].tolist(),
                terminal_extent_source_measured=False)
        elif path['kind']==1:
            ri=path.get('radius_index',2)
            if not np.allclose(before[:2],evidence['current_center_L'],atol=1e-13,rtol=0) or abs(before[ri]-evidence['current_radius_L'])>1e-13:
                raise ValueError('Parent conic differs from the audited template')
            assign(bi,0,oldcurve['center'][0]);assign(bi,1,oldcurve['center'][1]);assign(bi,ri,oldcurve['radius'])
            rec=dict(conic=_conic(oldcurve['center'],oldcurve['radius']).tolist(),
                historical_center_L=oldcurve['center'].tolist(),historical_radius_L=oldcurve['radius'])
        else:
            raise ValueError('Unsupported primitive must remain unresolved')
        records.append(dict(curve_id=item['curve_id'],material_index=mi,path_index=pi,block=bi,
            identity_basis=item['same_entity_basis'],numerical_seed_only=True,
            source_qualification_granted=False,pointwise_station_or_angle_correspondence=False,**rec))
    # Evaluate actual stored board vertices against the same continuous native
    # shape. Polygon facets are historical rendering support, not a second scene.
    native=NativeScene(as_event_view(result,'s01'));inverse=NativeBoardInverse(native)
    oldbottom=(np.asarray(old['ice_boundary']['points_xyz'])-origin)/length
    oldtop=(np.asarray(old['board_surface']['top_edge_xyz'])-origin)/length
    vertex_s,vertex_error=inverse(oldbottom[:,:2])
    samples=np.concatenate([a[None,:]+np.linspace(0.,1.,9)[:,None]*(b-a)[None,:]
        for a,b in zip(oldbottom,np.roll(oldbottom,-1,axis=0))])
    _,facet_error=inverse(samples[:,:2])
    ads=[]
    for link in ad_cross['links']:
        prefix=link['material_prefix'];number=int(prefix[2:])
        if link.get('model_geometry_used_as_source_identity') is not False:
            raise ValueError('Advertisement identity needs independent source evidence')
        if number not in (2,3,5):
            raise ValueError('Unexpected independent crosswalk requires explicit semantic review')
        original=next(a for a in old['advertisements'] if a['user_number']==number)
        if original['extent_scope']!=('internal_graphic_only' if number==5 else 'external_border'):
            raise ValueError('Old advertisement boundary has a different physical role')
        roles=('left_bottom','left_top','right_bottom','right_top')
        xyz=np.asarray([original['corners_xyz_scene_units'][role] for role in roles])/length
        stations,distance=inverse(xyz[:,:2]);stations=np.unwrap(stations/native.perimeter*2*np.pi)/(2*np.pi)*native.perimeter
        uv=[];ad_blocks=set()
        for role in roles:
            mi=lookup[prefix+'.'+role]
            if mi!=link['material_indices'][prefix+'.'+role]:
                raise ValueError('Current physical corner crosswalk changed')
            path=result['materials'][mi]['paths'][0]
            if path['kind']!=2 or not path.get('physical_point') or path.get('geometry_block',-1)>=0:
                raise ValueError('Confirmed corner must use its direct actual installation chart')
            ad_blocks.add(path['advertisement_block']);uv.append(path['points'][0])
        if len(ad_blocks)!=1:
            raise ValueError('Confirmed four corners do not share one physical installation')
        uv=np.asarray(uv,float);along=np.linalg.lstsq(np.c_[np.ones(4),uv[:,0]],stations,rcond=None)[0]
        vertical=np.linalg.lstsq(np.c_[1-uv[:,1],uv[:,1]],xyz[:,2],rcond=None)[0]
        if along[1]<=0:
            raise ValueError('Native positive-width installation cannot express this corner orientation')
        values=[float(along[0]%native.perimeter),float(along[1]),*vertical.tolist()]
        bi=next(iter(ad_blocks));result['blocks'][bi]['values']=values
        reconstructed=native.board(along[0]+along[1]*uv[:,0],vertical[0]+(vertical[1]-vertical[0])*uv[:,1])
        ads.append(dict(advertisement_id=number,block=bi,physical_definition=link['physical_definition'],
            historical_extent_scope=original['extent_scope'],native_s_width_z0_z1=values,
            old_actual_corner_XYZ_L=xyz.tolist(),corner_chart_projection_error_L=stats(distance),
            four_role_reconstruction_error_L=stats(np.linalg.norm(reconstructed-xyz,axis=1)),
            old_station_copied=False,unknown_print_contours_remapped=False,
            inherited_relative_print_charts_unchanged=True,source_corner_UV_created=False))
    changed=[i for i,(a,b) in enumerate(zip(parent['blocks'],result['blocks'])) if a['values']!=b['values']]
    if any(parent['blocks'][i]['name'].startswith(('rig/','shot/')) for i in changed):
        raise AssertionError('Geometry seed changed a camera')
    proposal=dict(schema='rinkfit.shared_template_native_seed.v2',status='temporary_unaccepted_geometry_candidate',
        parent=ref(parent_file),historical_scene=ref(scene_file),template_audit=ref(audit_file),
        ice_identity_crosswalk=ref(crosswalk_file),ad_identity_crosswalk=ref(ad_crosswalk_file),role_contract=ref(role_file),
        gauge=dict(old_to_native_scale=1/length,L=length,origin=origin.tolist(),rotation=np.eye(3).tolist(),
            axes='old x/y retained; right handed z above ice; same uniform scale in x/y/z',
            native_length=1.,verified_metric_scale=False,current_to_reference_similarity_removed=True),
        shared_scene5=scene5,geometry_blocks_changed=changed,ice_entities=records,advertisements=ads,
        old_board_vertex_surface_error_L=stats(vertex_error),old_board_facet_chord_to_continuous_error_L=stats(facet_error),
        old_board_top_height_max_error_L=float(np.max(np.abs(oldtop[:,2]-native.height))),
        known_limitations=[
            'Historical scene is a numerical initializer, not source truth or metric evidence.',
            'Continuous native superellipse cannot reproduce every interior point of the old faceted wall exactly.',
            'Unpaired goal-crease boundaries and absent ring roles are retained, not invented.',
            'Only Ad2/3/5 independently identified four-role installation boundaries are transferred.',
            'Unidentified print charts and other ad installations keep their numeric definitions; the shared board change still moves their XYZ.',
            'S01 shares this changed geometry and must be reinitialized/refitted; its old complete-state source auxiliaries are only impact witnesses.'],
        historical_K_R_C_loaded=False,source_UV_or_roles_changed=False,source_point_correspondences_created=False,
        one_shared_scene_for_all_shots=True,S01_geometry_preserved=False,source_accuracy_accepted=False)
    transform=dict(schema='rinkfit.explicit_old_xyz_to_native.v1',operation='XYZ/L',L=length,
        normalization_L=length,origin=origin.tolist(),axis_rotation=np.eye(3).tolist(),
        homogeneous_matrix=np.diag([1/length,1/length,1/length,1.]).tolist(),
        old_scene=ref(scene_file),old_axes='X right in original plan; Y up; Z above ice',
        native_axes='same right handed axes; ice Z=0; length L=1',
        no_template_similarity_applied=True,reference_target='original_8a28_unpatched',
        source_truth_or_metric_scale_certified=False)
    proposal['native_old_xyz_transform']=transform
    result['native_old_xyz_transform']=copy.deepcopy(transform)
    result['template_native_seed_v2']=proposal
    result['certificates'].update(joint_fit_complete=False,all_source_replay_complete=False,
        source_accuracy_accepted=False,shared_geometry_reinitialized=True)
    return result,proposal


def geometry_check(parent,model,proposal):
    before=NativeScene(as_event_view(parent,'s01'));after=NativeScene(as_event_view(model,'s01'))
    records=[]
    for item in proposal['ice_entities']:
        mi=item['material_index'];pi=item['path_index'];path=after.path(mi,pi)
        q=after.values[path['geometry_block']]
        if path['kind']==0:
            error=float(np.max(np.abs(_line(q[:4])-np.asarray(item['carrier']))))
            support=float(np.max(np.abs(q[:4].reshape(2,2)-item['historical_finite_endpoints_L'])))
        else:
            error=float(np.max(np.abs(_conic(q[:2],q[path.get('radius_index',2)])-item['conic'])))
            support=0.
        records.append(dict(curve_id=item['curve_id'],coefficient_max_error=error,finite_support_max_error_L=support))
    from .joint_replay import inventory
    w0,p0=inventory(parent);w1,p1=inventory(model)
    if [(r['label'],r['path'],r.get('station_domain_interval')) for r in p0]!=[(r['label'],r['path'],r.get('station_domain_interval')) for r in p1] or w0.shape!=w1.shape:
        raise ValueError('Shared inventory topology unexpectedly changed')
    camera=camera_from_event(as_event_view(model,'s01'))
    frames=[]
    for pts in (305408,406016):
        local=(pts-model['shots']['s01']['time_origin_pts'])/12800.
        a=camera.project_rs(w0,np.full(len(w0),local));b=camera.project_rs(w1,np.full(len(w1),local))
        inside=a['valid']&np.all((a['uv']>=[0,0])&(a['uv']<=[1919,1079]),axis=1)
        frames.append(dict(native_pts=pts,source_seconds=pts/12800.,
            original_grid_onscreen_points=int(inside.sum()),
            projection_change_px=stats(np.linalg.norm(a['uv']-b['uv'],axis=1),inside&b['valid']),
            new_RS_invalid_on_original_onscreen=int((inside&~b['valid']).sum()),
            model_station_samples_are_display_samples_not_source_correspondences=True))
    return dict(named_carrier_conic_coefficients=records,physical_domain=validate_physical_domain(model),
        shared_inventory_paths=len(p1),shared_inventory_vertices=len(w1),
        whole_inventory_world_displacement_L=stats(np.linalg.norm(w1-w0,axis=1)),
        S01_grid_impact=frames,cameras_unchanged=True,one_scene=True)


def s01_uv_impact(parent,model,manifest_file,output):
    """Actual original S01 UV at preserved source identities/nuisance witnesses.

    Stations are reused only to measure displacement of the previous model.
    They are not declared corresponding source points or reprofiled optima.
    """
    from .baseline_auxiliary import checked_array,inherited_semantic_array
    manifest=read_json(manifest_file);oldmodel=read_json(bound(manifest['model']))
    semantic_file=bound(manifest['semantic']);semantic=read_json(semantic_file)
    prediction_file=bound(manifest['predictions'])
    with np.load(prediction_file,allow_pickle=False) as z:
        station=z['stations'];branch=z['branches'];old_uv=z['uv'];oldvalid=z['RS_valid'];active=z['active_source_mask'];pts=z['native_pts']
    N=int(manifest['source_rows'])
    raw=manifest['raw_source_arrays_verified_exact']
    uv=np.fromfile(bound(raw['native_uv']),'<f8').reshape(N,2)
    oldmat=inherited_semantic_array(semantic_file,'resolved_material_index')
    labels0=np.asarray([m['label'] for m in oldmodel['materials']]);lookup={m['label']:i for i,m in enumerate(model['materials'])}
    mat=np.asarray([lookup[labels0[int(i)]] for i in oldmat],np.int64)
    rows=np.flatnonzero(active);mi=mat[rows];pi=branch[rows];ss=station[rows]
    for m,b in np.unique(np.c_[oldmat[rows],pi],axis=0):
        new_m=lookup[labels0[m]]
        if oldmodel['materials'][m]['paths'][b]['id']!=model['materials'][new_m]['paths'][b]['id']:
            raise ValueError('S01 original finite material/path identity changed')
    before=NativeScene(as_event_view(parent,'s01'));after=NativeScene(as_event_view(model,'s01'))
    w0,ok0,_=before.world_rows(mi,pi,ss);w1,ok1,_=after.world_rows(mi,pi,ss)
    camera=camera_from_event(as_event_view(model,'s01'));local=(pts[rows]-model['shots']['s01']['time_origin_pts'])/12800.
    check=camera.project_rs(w0,local)
    parity=stats(np.linalg.norm(check['uv']-old_uv[rows],axis=1),ok0&check['valid']&oldvalid[rows])
    if parity['maximum']>1e-6:
        raise ValueError('Parent does not reproduce the original S01 numerical witness')
    new=camera.project_rs(w1,local)
    kind=checked_array(semantic['arrays']['kind'],semantic_file.parent)[rows]
    normal=checked_array(semantic['arrays']['profile_normal'],semantic_file.parent)[rows]
    prior_error=old_uv[rows]-uv[rows];new_error=new['uv']-uv[rows]
    prior_op=np.where(kind==2,np.abs(np.sum(prior_error*normal,axis=1)),np.linalg.norm(prior_error,axis=1))
    new_op=np.where(kind==2,np.abs(np.sum(new_error*normal,axis=1)),np.linalg.norm(new_error,axis=1))
    valid=ok1&new['valid'];categories={
        'all':np.ones(len(rows),bool),'fixed_material_corners':kind==1,
        'identified_ice':np.isin(mi,[r['material_index'] for r in model['template_native_seed_v2']['ice_entities']]),
        'identified_Ad2_3_5':np.array([model['materials'][int(i)]['label'].split('.')[0] in ('Ad2','Ad3','Ad5') for i in mi])}
    summaries={name:dict(rows=int(select.sum()),before_original_source_operator_px=stats(prior_op[select],oldvalid[rows][select]),
        after_original_source_operator_px=stats(new_op[select],valid[select]),
        same_witness_UV_displacement_px=stats(np.linalg.norm(new['uv'][select]-old_uv[rows][select],axis=1),valid[select]&oldvalid[rows][select]),
        after_RS_invalid=int((select&~valid).sum())) for name,select in categories.items()}
    np.savez_compressed(Path(output)/'s01_impact.npz',source_row=rows,native_pts=pts[rows],material_index=mi,path=pi,
        prior_station=ss,source_uv=uv[rows],parent_uv=old_uv[rows],candidate_uv=new['uv'],candidate_valid=valid)
    return dict(manifest=ref(manifest_file),rows=N,active_rows=len(rows),parent_original_prediction_parity_px=parity,
        original_source_UV_unchanged=True,source_fit_or_independent_validation=False,
        source_nuisance_not_reprofiled=True,station_witness_does_not_establish_new_point_correspondence=True,
        metrics=summaries,arrays=ref(Path(output)/'s01_impact.npz'))


def run(parent_file,audit_file,crosswalk_file,ad_crosswalk_file,s01_manifest,output):
    start=time.perf_counter();output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Candidate diagnostics belong in Data')
    model,proposal=build_candidate(parent_file,audit_file,crosswalk_file,ad_crosswalk_file)
    output.mkdir(parents=True);write_json(output/'model.json',model);write_json(output/'proposal.json',proposal)
    parent=read_json(parent_file)
    print(json.dumps(dict(stage='candidate_written',model=str(output/'model.json'),changed_blocks=len(proposal['geometry_blocks_changed']))),flush=True)
    checks=geometry_check(parent,model,proposal)
    write_json(output/'geometry_check.json',checks)
    print(json.dumps(dict(stage='geometry_checked',passed=checks['physical_domain']['passed'])),flush=True)
    impact=s01_uv_impact(parent,model,s01_manifest,output)
    write_json(output/'s01_impact.json',impact)
    receipt=dict(schema='rinkfit.template_native_seed_execution.v2',candidate=ref(output/'model.json'),
        proposal=ref(output/'proposal.json'),geometry_check=ref(output/'geometry_check.json'),
        S01_impact=ref(output/'s01_impact.json'),module=ref(__file__),seconds=time.perf_counter()-start,
        candidate_generated=True,source_accuracy_accepted=False,one_shared_scene=True,historical_camera_loaded=False)
    write_json(output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','audit','crosswalk','ad-crosswalk','s01-manifest','output'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();run(a.parent,a.audit,a.crosswalk,a.ad_crosswalk,a.s01_manifest,a.output)


if __name__=='__main__':main()
