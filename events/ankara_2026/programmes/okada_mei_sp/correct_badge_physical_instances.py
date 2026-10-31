"""Apply explicitly reviewed physical badge instances without changing source UV.

The failed input is immutable. Appearance reference and the original incorrect
label remain in each corrected source artifact. No camera/station fit is run.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
from collections import Counter
import numpy as np
from capture_native_panels import single_panel,panel_material_metadata
from build_independent_shot_packets import build

RANGES=[(1445376,1445376,'Ad12'),(1446400,1449216,'Ad12'),(1449984,1449984,'Ad12'),(1451264,1451776,'Ad12'),
    (1568000,1569024,'Ad25'),(1569536,1582080,'Ad25'),(1583104,1583104,'Ad25')]


def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('root','input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();old=a.input.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    manifest_path=root.parent/'prepared_joint_ice/joint_manifest.json';manifest=json.loads(manifest_path.read_text('utf8'));scene_path=Path(manifest['scene_path']);scene=json.loads(scene_path.read_text('utf8'))
    metadata=panel_material_metadata(scene,scene_path);catalogue={ad:single_panel(ad,'outer_ring')for ad in('Ad12','Ad25')};paths={ad:p['path_ids'][0]for ad,p in catalogue.items()};cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts']
    for ad,path in paths.items():
        cut=next(c for c in cuts if c['id']==path)
        if not cut['closed']or metadata[path]['role']!='logo_outer_boundary'or catalogue[ad]['material_signal']!='blue':raise ValueError('Current native catalogue badge exterior differs')
    source=old/'qualified/s07/source_rows.jsonl';source_sha=sha(source);old_packet=old/'source_packets/s07';old_hashes={p.name:sha(p)for p in old_packet.iterdir()if p.is_file()}
    audit=root/'ad12_identity_audit';review=audit/'identity_review.json';out.mkdir(parents=True);folder=out/'qualified/s07';folder.mkdir(parents=True)
    approval=dict(schema='rinkspace.explicit_physical_print_instance_correction.v1',authority='Explicit main-session original-image review and current user task instruction',
        main_camera_known_direction_and_ad_order_used=True,appearance_similarity_alone_is_not_identity=True,
        intervals=[dict(first_pts=lo,last_pts=hi,physical_instance=ad,native_path_id=paths[ad],camera_view='known main looking toward negative X end'if ad=='Ad12'else'known main looking toward positive X end',
            surrounding_ad_order='Ad13 / Ad12 / Ad11 / Ad10 / Ad9'if ad=='Ad12'else'Ad1 ALL SKATING and Ad26 red ministry badge / Ad25 / Ad24 Turkish round badge')for lo,hi,ad in RANGES],
        root_reviewed_PTS=[1451776,1568000,1583104],root_reviewed_source_proofs=[str(audit/'source_review'/f's07_pts{pt}_identity_HOLD_candidates.jpg')for pt in[1451776,1568000,1583104]],
        original_failed_source=str(source),original_failed_source_sha256=source_sha,original_identity_audit=str(review),original_identity_audit_sha256=sha(review),
        appearance_reference=str(root/'ad12_shield_pilot_context/reference.json'),appearance_reference_instance='Ad12',
        material_and_shape_measurement_qualification_unchanged=True,source_UV_and_quantiles_unchanged=True)
    approval_path=out/'physical_instance_approval.json';approval_path.write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n',encoding='utf8');approval_sha=sha(approval_path);counts=Counter();frames=Counter();seen=[];selected=[]
    def correct(row,ad,original_label):
        original_qualification=row.get('qualification')
        result=dict(row,entity=ad,path_id=paths[ad],appearance_reference_instance='Ad12',appearance_reference_native_path_id=paths['Ad12'],
            original_stored_entity=row.get('entity'),original_stored_path_id=row.get('path_id'),physical_instance_label_corrected=row.get('path_id')!=paths[ad],
            actual_native_path_material_metadata=metadata[paths[ad]],source_shape_qualification_reused_unchanged=True,
            physical_instance_identity_review_pending=False,physical_instance_approval=str(approval_path),physical_instance_approval_sha256=approval_sha,
            physical_instance_identity_basis='Explicit main-session source-frame full-view review: known main-rig end direction and neighboring physical advertisements; not appearance-template similarity')
        if original_qualification is not None:result['original_appearance_qualification']=original_qualification;result['qualification']=dict(original_qualification,known_source_instance=ad,physical_instance_approval=str(approval_path),physical_instance_approval_sha256=approval_sha)
        return result
    with(folder/'source_rows.jsonl').open('x',encoding='utf8')as stream:
        for number,line in enumerate(source.open(encoding='utf8'),1):
            r=json.loads(line);matches=[ad for lo,hi,ad in RANGES if lo<=r['pts']<=hi]
            if r['fit_eligible']:
                if len(matches)!=1:raise ValueError('Qualified frame outside explicit physical-identity approval')
                ad=matches[0];r=dict(r,rows=[correct(q,ad,True)for q in r['rows']],pending_rows=[correct(q,ad,True)for q in r.get('pending_rows',[])],
                    original_failed_source_record=dict(path=str(source),sha256=source_sha,line=number),physical_instance_approval=str(approval_path),physical_instance_approval_sha256=approval_sha)
                counts[ad]+=len(r['rows']);frames[ad]+=1;seen.append(r['pts']);selected.append(r)
            stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    assert dict(frames)=={'Ad12':17,'Ad25':56}and dict(counts)=={'Ad12':71,'Ad25':325}
    shutil.copyfile(old/'qualified/s07/poses.npz',folder/'poses.npz');binding=json.loads((old/'qualified/binding.json').read_text('utf8'));binding.update(physical_instance_approval=str(approval_path),physical_instance_approval_sha256=approval_sha,corrected_native_path_ids=paths,qualified_rows_by_physical_instance=dict(counts),qualified_frames_by_physical_instance=dict(frames),original_failed_source=str(source),original_failed_source_sha256=source_sha,appearance_reference_instance='Ad12',appearance_match_does_not_assign_physical_instance=True,source_UV_quantiles_and_shape_qualification_unchanged=True);(out/'qualified/binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    build(root,out/'qualified',out/'source_packets',shot_ids=['s07']);new_packet=out/'source_packets/s07';old_arrays=dict(np.load(old_packet/'point_observations.npz',allow_pickle=False));new_arrays=dict(np.load(new_packet/'point_observations.npz',allow_pickle=False))
    for key in('pts','uv','kind','stations','source_rows'):
        assert old_arrays[key].dtype==new_arrays[key].dtype and old_arrays[key].tobytes()==new_arrays[key].tobytes(),key
    for name in('controls.npy','shot_seed.json'):
        shutil.copyfile(old_packet/name,new_packet/name);assert sha(old_packet/name)==sha(new_packet/name)
    mapping=json.loads((new_packet/'mapping.json').read_text('utf8'));assert mapping['path_ids']==[paths['Ad12'],paths['Ad25']]
    expected=[paths['Ad12']if int(pt)<1568000 else paths['Ad25']for pt in new_arrays['pts']];assert [mapping['path_ids'][i]for i in new_arrays['path_indices']]==expected
    for record in selected:
        old_record=next(json.loads(l)for l in source.open(encoding='utf8')if json.loads(l)['pts']==record['pts'])
        for before,after in zip(old_record['rows'],record['rows'],strict=True):
            for key in('uv','normal','actual_source_quantiles','actual_source_sides','candidate_flags','measurement_operator'):
                if key in before:assert json.dumps(before[key],sort_keys=True)==json.dumps(after[key],sort_keys=True),(record['pts'],key)
    for name,digest in old_hashes.items():assert sha(old_packet/name)==digest
    report=dict(complete=True,frames=73,rows=396,frames_by_instance=dict(frames),rows_by_instance=dict(counts),path_ids=mapping['path_ids'],catalogue=catalogue,current_path_metadata={ad:metadata[path]for ad,path in paths.items()},physical_instance_approval=str(approval_path),physical_instance_approval_sha256=approval_sha,original_failed_packet=str(old_packet),original_failed_packet_hashes_unchanged=True,source_UV_PTS_kind_station_source_rows_bytes_preserved=True,source_quantiles_and_original_shape_qualification_preserved=True,old_controls_and_shot_seed_bytes_preserved=True,camera_or_fit_state_modified=False,new_measurements_generated=False,fit_executed=False,packet=str(new_packet),packet_sha256=sha(new_packet/'point_observations.npz'),packet_manifest=str(out/'source_packets/packets_manifest.json'),qualified_source_rows=str(folder/'source_rows.jsonl'),implementation=str(Path(__file__).resolve()),implementation_sha256=sha(__file__))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({k:report[k]for k in['frames','rows','frames_by_instance','rows_by_instance','path_ids','packet']}),flush=True)


if __name__=='__main__':main()