"""Read actual Ad7 held roles/render materials and Ad4 FIT time coverage."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
sys.dont_write_bytecode=True
import numpy as np
from verify_original225_derived_center_20261008 import numeric_data_loads
from structured_programme_source_candidate import original_review_source as review
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from calibration.program_tracking.render_result import advertisement_boundary_paths

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def convert(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v).__name__)
def main():
    ap=argparse.ArgumentParser()
    for name in ('state-dir','auxiliary','board-packet','retirement-packet','output'):ap.add_argument('--'+name,required=True,type=Path)
    a=ap.parse_args();identity=read(a.state_dir/'state_identity.json')
    assert sha(review.PACKET)==review.PACKET_SHA256
    packet=numeric_data_loads(review.PACKET.read_bytes());items=packet['programme_absolute_rows']
    provider,world,proof=current_complete_source_geometry(a.state_dir/'complete_state_UNACCEPTED.npz',a.auxiliary)
    scene=read(a.state_dir/'scene_UNACCEPTED.json');assert sha(a.state_dir/'scene_UNACCEPTED.json')==identity['scene_sha256']
    # Numeric inspection of Ad7 alone; this is not a partial-model review image.
    # Current source-template ads deliberately have no generic rectangle corners.
    ad7_scene=dict(scene,advertisements=[r for r in scene['advertisements']if int(r['user_number'])==7])
    adpaths=advertisement_boundary_paths(ad7_scene)
    assert len(adpaths)==4
    checks=[];fit4=[];fit7=[]
    for index,item in enumerate(items):
        row=item['source_row'];name=str(row['id']);ids=row.get('physical_ids') or []
        is_ad7=('ad7' in name.lower() or 'advertisement7' in name.lower() or any(str(i).startswith('ad7:')for i in ids))
        is_ad4=('ad4:' in name.lower() or 'advertisement4' in name.lower() or any(str(i).startswith('ad4:')for i in ids))
        if (is_ad4 or is_ad7) and item['collection_role']=='FIT':
            rec=dict(packet_index=index,id=name,actual_pts=int(item['group_actual_pts']),native_pts=row.get('source_native_pts'),
                     source_row_pts=int(row['pts']),block=item['block'],split=row.get('split'),physical_ids=ids,
                     UV_count=len(row['uv']),observation_signature=review.observation_signature(row))
            (fit4 if is_ad4 else fit7).append(rec)
        if not (is_ad7 and item['collection_role']=='HELD'):continue
        provider.register_row(row,actual_pts=int(item['group_actual_pts']),allow_previously_qualified_held_ad2_representation=True)
        bound=provider._binding(row)[1];current=provider.rebind(row,world);values=np.asarray(current['world'],float)
        roles=bound.get('ids',[]);match=[]
        for path in adpaths:
            if not any(path['role'].startswith(str(role).split('_')[0]) for n,role in roles):continue
            xyz=np.asarray(path['xyz'],float);first,last=xyz[0],xyz[-1];delta=last-first
            t=np.sum((values-first)*delta,axis=1)/np.dot(delta,delta)
            nearest=first+np.clip(t,0,1)[:,None]*delta
            match.append(dict(render_role=path['role'],maximum_world_to_render_finite_line_distance_scene_units=float(np.linalg.norm(values-nearest,axis=1).max()),
                              projected_support_fraction_range=[float(t.min()),float(t.max())]))
        checks.append(dict(packet_index=index,id=name,actual_pts=int(item['group_actual_pts']),source_row_pts=int(row['pts']),
                           split=row.get('split'),block=item['block'],source_binding_ids=roles,
                           current_material_coordinates=bound.get('material_coordinates'),actual_source_columns=bound['columns'],
                           source_world_xyz_scene_units=values,render_geometry_comparison=match,
                           source_UV_sigma_roles_unchanged=review.observation_signature(current)==review.observation_signature(row)))
    board=read(a.board_packet);board4=[{k:r.get(k)for k in ('id','pts','actual_pts','split','source_raw_frame','curve')}for r in board['rows']
                                    if (r.get('curve',{}).get('advertisement_number')==4 or 'ad4' in r.get('id','').lower())]
    retire=read(a.retirement_packet);locator=retire['source_row_locator']
    for r in fit7:
        r['retired_original_false_corner_in_current_config']=(r['block']==locator['block']and r['id']==locator['id']and r['actual_pts']==locator['pts'])
    report=dict(schema='rinkspace.actual-Ad7-held-material-role-render-and-Ad4-coverage-audit.v1',state_identity=identity,
                source_packet_sha256=review.PACKET_SHA256,whole_graph_loaded=False,source_rows_modified=False,
                Ad7_retirement_scope={k:retire.get(k)for k in ('source_row_locator','observed_identity','confirmed_print_corner_retained','proposed_revision','reasoning','scope')},
                Ad7_actual_original_HELD_bindings=checks,Ad7_literal_original_FIT_before_certified_retirement=fit7,Ad4_actual_original_FIT=fit4,
                current_native_board_Ad4_rows=board4,board_packet_path=str(a.board_packet),board_packet_sha256=sha(a.board_packet),
                literal_ad_ids_only_not_an_inventory_of_untagged_inferred_instances=True,accuracy_accepted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=convert)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(a.output),sha256=sha(a.output),Ad7_HELD=len(checks),Ad7_FIT=len(fit7),Ad4_FIT=len(fit4),native_Ad4_rows=len(board4),
                          render_mismatch_max=max((r['maximum_world_to_render_finite_line_distance_scene_units']for c in checks for r in c['render_geometry_comparison']),default=0.))),flush=True)
if __name__=='__main__':main()
