import copy,json
from pathlib import Path
import cv2
import numpy as np
import pytest
from rinkfit_joint import dense_visible_review as review
from rinkfit_joint.joint_review_packet import CurrentReviewPacket


def fixture_packet(tmp_path, name, *, changed=False):
    root=tmp_path/name;root.mkdir()
    def save_json(name,value):
        p=root/name;p.write_text(json.dumps(value),encoding='utf8');return p
    def arr(name,a):
        a=np.asarray(a);p=root/name;a.tofile(p)
        return dict(review.ref(p),dtype=a.dtype.str,shape=list(a.shape))
    physical=np.array([1,1,0]if changed else[1,1,1],'u1')
    material=np.array([0,0,np.iinfo(np.uint32).max]if changed else[0,2,3],'<u4')
    arrays={k:arr(k+'.bin',a) for k,a in dict(uv=np.array([[100,100],[200,200],[300,300]],'<f8'),
        native_pts=np.array([256,256,512],'<i8'),shot_index=np.zeros(3,'<u2'),material_index=material,
        path=np.zeros(3,'<u4'),admission=physical,fit_role=physical).items()}
    ids=[b'source/B',b'source/C'];blob=b''.join(ids)
    reg=dict(row_start=1,count=2,source_id_offsets=arr('offsets.u64',np.array([0,len(ids[0]),len(blob)],'<u8')),
        source_id_utf8=arr('ids.u8',np.frombuffer(blob,'u1')))
    labels=[dict(index=0,label='kept')]if changed else[dict(index=i,label='old'+str(i))for i in range(4)]
    packet=save_json('packet.json',dict(row_count=3,shot_ids=['s01'],arrays=arrays,source_row_ids=['source/A'],
        dense_source_binding=reg,material_binding=dict(materials=labels)))
    q=dict(physical_mask=arrays['admission'],training_mask=arrays['admission'],
        fit_rows=arr('fit.u64',np.array([1]if changed else[1,2],'<u8')),
        inspection_rows=arr('inspection.u64',np.array([0],'<u8')))
    manifest=save_json('manifest.json',dict(schema='rinkfit.current_joint_input.v1',logical_source_id='logical',raw_rows=3,
        packet=review.ref(packet),qualification=q))
    return packet,review.ref(manifest)


def test_current_compact_ID_and_changed_material_are_bound_to_actual_side(tmp_path):
    packet,manifest=fixture_packet(tmp_path,'candidate',changed=True)
    reader=CurrentReviewPacket(packet,current_input=manifest,required_rows=[1,2])
    assert reader.packet['source_row_ids'][1]=='source/B'
    active=reader.source_binding(1);retired=reader.source_binding(2)
    assert active['material_index']==0 and active['qualification']['representative_fit']
    assert retired['retired_material'] and not retired['qualification']['physical']
    from rinkfit_joint.joint_review_packet import source_point_reader
    from rinkfit_joint.joint_display import CurrentSourcePoints
    with source_point_reader(reader):
        points=CurrentSourcePoints(packet,dict(materials=[dict(label='Ad1.print.kept')]),
            dict(shots=[dict(id='s01',first_pts=256,last_pts=512)]),native_pts=[256,512])
    assert points.ledger[256]['representative_fit_points']==1
    assert not points.by_pts.get(512)  # Retired source remains metadata/annotation, never a fitted dot.
    frame=dict(raw_row=2,source_row_id='source/C',source_uv=[300.,300.],shot_id='s01',native_pts=512)
    review._bind_frame_sources([frame],packet,reader=reader)
    assert frame['material_index']==np.iinfo(np.uint32).max
    with pytest.raises(ValueError,match='source ID changed'):
        review._bind_frame_sources([dict(frame,source_row_id='another')],packet,reader=reader)


def test_render_routes_each_model_to_its_own_packet_without_reselecting_source(tmp_path,monkeypatch):
    old,old_manifest=fixture_packet(tmp_path,'old');new,new_manifest=fixture_packet(tmp_path,'new',changed=True)
    baseline=tmp_path/'baseline';candidate=tmp_path/'candidate';baseline.mkdir();candidate.mkdir()
    for folder in (baseline,candidate):(folder/'model.json').write_text('{}',encoding='utf8')
    scope=tmp_path/'scope.json';scope.write_text('{}',encoding='utf8')
    fixed=[dict(raw_row=1,source_row_id='source/B',source_uv=[200.,200.],shot_id='s01',native_pts=256,
        expected_native_crop_xyxy=[40,80,360,320]),dict(raw_row=2,source_row_id='source/C',source_uv=[300.,300.],shot_id='s01',native_pts=512,
        expected_native_crop_xyxy=[140,180,460,420])]
    plan=dict(schema='rinkfit.current_sixshot_visible_review_plan.v1',packet=review.ref(old),
        baseline_packet=review.ref(old),candidate_packet=review.ref(new),baseline_current_input_v1=old_manifest,
        candidate_current_input_v1=new_manifest,objective_changed=True,operator={},baseline_model=review.ref(baseline/'model.json'),
        scope=review.ref(scope),fixed_frames=fixed,fixed_focus_frames=[])
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan),encoding='utf8')
    calls=[]
    def snapshots(model,scope,output,*,packet_file,**kwargs):
        from rinkfit_joint import joint_display
        bound=joint_display.PacketArrays(packet_file)
        calls.append((Path(model),Path(packet_file),bound.source_binding(1)['material_index']))
        output.mkdir();frames=[]
        for pts in [256,512]:
            image=output/(str(pts)+'.png');cv2.imwrite(str(image),np.zeros((1080,1920,3),'u1'))
            frames.append(dict(shot_id='s01',native_pts=pts,path=str(image),decoded_native_BGR_sha256='same_raster'))
        return dict(frames=frames)
    monkeypatch.setattr(review,'render_snapshots',snapshots)
    result=review.render(planfile,candidate,tmp_path/'review',fixed_preview=True)
    assert calls==[(baseline/'model.json',old,2),(candidate/'model.json',new,0)]
    assert result['objective_changed'] and not result['full_raw_domain_compared']
    assert result['cards'][0]['baseline_source']['material_index']==2
    assert result['cards'][0]['candidate_source']['material_index']==0
    assert result['cards'][1]['candidate_source']['retired_material']
    assert [c['source_uv']for c in result['cards']]==[x['source_uv']for x in fixed]
    assert [c['crop_xyxy']for c in result['cards']]==[x['expected_native_crop_xyxy']for x in fixed]
    with pytest.raises(ValueError,match='existing same-source'):
        review.render(planfile,candidate,tmp_path/'not_a_full_comparison')
    assert not (tmp_path/'not_a_full_comparison').exists()
