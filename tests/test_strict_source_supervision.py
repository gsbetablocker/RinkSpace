from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking.test_strict_absolute_source import make_pack,fixture_inputs
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview,REVIEW_SCHEMA,file_sha
from calibration.program_tracking.strict_absolute_source import StrictAbsolutePackWriter,normalize_selected_ice_rows
from rinkspace.strict_source_supervision import (
    strict_positive_annotations,write_strict_negative_source_pack,
    StrictSourceTrainingInputs,strict_negative_point_loss,
    StrictSourceStreamingTrainer,
)


CLASSES=['background','ice_paint','ad_boundary','shadow_or_reflection']


def test_center_normal_comes_from_actual_sides_and_support_is_not_paint_width(tmp_path):
    pack,_=make_pack(tmp_path)
    row=deepcopy(pack.fit_rows[0])
    authority=dict(path=str(tmp_path/'strict_source.npz'),sha256=file_sha(tmp_path/'strict_source.npz'))
    assert strict_positive_annotations([row],authority=authority,classes=CLASSES)==[]
    row['normal']=np.array([[1.,0.]])  # An unused navigation normal is not truth.
    row['source_readout_uncertainty']=dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True,paint_width_used_as_center_tolerance=False,
        measurement_method='explicit source reader fixture',quantization_step_px=.125,
        actual_readout_blur_interval_px=[-.125,.125],normal_interval_px=[-.25,.25])
    annotations=strict_positive_annotations([row],authority=authority,classes=CLASSES)
    assert len(annotations)==1
    expected=row['source_sides_uv'][:,1]-row['source_sides_uv'][:,0]
    expected/=np.linalg.norm(expected,axis=1)[:,None]
    assert np.array_equal(np.asarray(annotations[0]['normal']),expected)
    assert annotations[0]['interval_px']==[[-.25,.25]]
    assert annotations[0]['state']=='source_reviewed_interval'
    assert annotations[0]['pixel_ground_truth'] is False
    row['source_readout_uncertainty']['paint_width_used_as_center_tolerance']=True
    assert strict_positive_annotations([row],authority=authority,classes=CLASSES)==[]


def test_pending_and_wrong_human_identity_cannot_be_positive_truth(tmp_path):
    pack,_=make_pack(tmp_path);row=deepcopy(pack.fit_rows[0])
    authority=dict(path=str(tmp_path/'strict_source.npz'),sha256=file_sha(tmp_path/'strict_source.npz'))
    row['admission_status']='pending_exact_source_review'
    assert strict_positive_annotations([row],authority=authority,classes=CLASSES)==[]


def test_only_actual_reviewed_role_absence_exports_negative(tmp_path):
    pack,_=make_pack(tmp_path)
    proof=deepcopy(pack.proof_templates[0]);proof.update(uv=[311.,412.],decision='exclude',scope='reflection',
        static_material_confirmed=False,physical_material_id='ad7.reflection',
        reject_only_authority='actual_visual_role_absence',actual_source_RGB_role_absence_confirmed=True,
        confidence_used_as_rejection_evidence=False,model_probability_used_as_rejection_evidence=False,
        excluded_kinds=['ice_paint','ad_boundary'],shot_id='main1:0',split='fit')
    pending=deepcopy(proof);pending['uv']=[312.,413.]
    pending.pop('actual_source_RGB_role_absence_confirmed')
    review=ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=pack.metadata['source'],reviewed_endpoints=[proof,pending]))
    reference=write_strict_negative_source_pack(tmp_path/'role_absence.npz',review)
    assert reference['points']==1
    inputs=StrictSourceTrainingInputs(dict(path=str(tmp_path/'strict_source.npz'),sha256=file_sha(tmp_path/'strict_source.npz')),
        negative_reference=reference,classes=CLASSES)
    assert len(inputs.negative_by_native[110])==1
    assert inputs.negative_by_native[110][0]['uv']==[311.,412.]


def test_negative_loss_preserves_unknown_semantics_and_protects_human_pixels():
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    logits=torch.zeros((1,len(CLASSES),8,8),requires_grad=True)
    negatives=[dict(uv=[3.1,3.5],excluded_kinds=['ice_paint','ad_boundary'])]
    loss,count=strict_negative_point_loss(logits,negatives,CLASSES)
    assert count==1 and float(loss)>0
    loss.backward()
    assert logits.grad[0,1,3,3]>0
    assert logits.grad[0,0,3,3]<0 and logits.grad[0,3,3,3]<0
    human=[dict(state='user_confirmed_visual_supervision',kind='ice_paint',uv=[[4.9,3.5]])]
    with pytest.raises(ValueError,match='human pixels'):
        strict_negative_point_loss(logits,negatives,CLASSES,human_annotations=human)


def test_vision_train_hook_is_explicit_and_does_not_train_on_import(monkeypatch,tmp_path):
    from rinkspace import vision,strict_source_supervision
    path=tmp_path/'stream_train.json';path.write_text(json.dumps(dict(strict_source_stream_training=dict(absolute_inputs={}))),encoding='utf8')
    seen=[]
    monkeypatch.setattr(strict_source_supervision,'train_strict_source_stream',lambda *a,**kw:seen.append((a,kw)) or dict(mock=True))
    assert vision.train(path,resume=True,candidate=True,steps=7)==dict(mock=True)
    assert seen[0][1]['steps']==7 and seen[0][1]['resume'] is True


def test_shared_frame_crop_consumer_uses_actual_targets_and_skips_held(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    bgr=np.zeros((1080,1920,3),np.uint8)
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    binding['decoded_native_BGR_sha256']=hashlib.sha256(bgr.tobytes()).hexdigest()
    measurement['source_binding']=deepcopy(binding)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    row=normalized['rows'][0]
    row['source_readout_uncertainty']=dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True,paint_width_used_as_center_tolerance=False,
        measurement_method='explicit source reader fixture',quantization_step_px=.125,
        actual_readout_blur_interval_px=[-.125,.125],normal_interval_px=[-.25,.25])
    writer=StrictAbsolutePackWriter(tmp_path/'train_fixture.npz',source=normalized['review'].source,expected_actual_pts=[100])
    writer.add_row(row,normalized['review'])
    # Same decoded frame can hold separate calibration FIT/HELD source rows.
    # Frame provenance must remain independent of both calibration role and
    # the original complete-shot ML role.
    held=deepcopy(row);held['split']='held';held['family']='held_calibration_source'
    writer.add_row(held,normalized['review']);receipt=writer.finish()
    inputs=StrictSourceTrainingInputs(receipt,classes=CLASSES)
    assert len(inputs.rows_by_native[binding['native_pts']])==2
    assert 'split' not in inputs.frame_bindings[binding['native_pts']]
    assert len(inputs.frame_supervision(bgr,binding)['annotations'])==1
    class SmallCpuTestHead(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros((1,len(CLASSES),1,1)))
        def encode_context(self,images):return []
        def forward(self,images,**kwargs):
            h,w=images.shape[-2:]
            return dict(logits=self.weight.expand(1,len(CLASSES),h,w)[:,None],
                primitive_logits=torch.zeros((1,1,3,h,w))+self.weight.sum()*0)
    network=SmallCpuTestHead();optimizer=torch.optim.SGD(network.parameters(),lr=.1)
    trainer=StrictSourceStreamingTrainer(network,optimizer,inputs,classes=CLASSES,device='cpu',crop_size=32)
    result=trainer.consume_frame(bgr,binding,shot_role='development')
    assert result['optimizer_steps']==0 and network.weight.count_nonzero()==0
    result=trainer.consume_frame(bgr,binding,shot_role='train')
    assert result['optimizer_steps']==1 and result['crop_batches'][0]['positive_points']==1
    assert network.weight[0,1,0,0]>0
    assert result['temporal_prediction_used_as_position'] is False
