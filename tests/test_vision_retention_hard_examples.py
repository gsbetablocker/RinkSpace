from copy import deepcopy
import pytest

torch=pytest.importorskip('torch')
torch.set_num_threads(1)
from rinkspace.vision_retention_hard_examples import NativeRetentionHardExamples,protect_feedback_from_conflicting_original_regions

CLASSES=['background','ice_paint','shadow_or_reflection']
BINDING=dict(source_sha256='a'*64,source_pts=110,time_base='1/100',shot_id='main1',native_hw=[8,8])

def comparison(collection,identity,field,indices):
    return dict(source_binding=BINDING,newly_lost_confirmed_support=[dict(collection=collection,
        identity=identity,field=field,newly_wrong_original_indices=indices)])

def test_lost_human_index_replays_its_exact_original_uv_only():
    annotation=dict(state='user_confirmed_visual_supervision',group_id='C',kind='ice_paint',
        source_point_indices=[10,20],uv=[[1.,1.],[3.,4.]])
    hard=NativeRetentionHardExamples(binding=BINDING,annotations=[annotation],classes=CLASSES,whole_shot_role='train')
    assert hard.observe(comparison('human_groups','C','semantic_wrong_source_point_indices',[20]))==1
    logits=torch.zeros((1,3,8,8),requires_grad=True);primitive=torch.zeros((1,3,8,8),requires_grad=True)
    hard.loss(logits,primitive,binding=BINDING).backward()
    assert logits.grad[0,1,4,3]<0 and logits.grad[0,0,4,3]>0
    assert logits.grad[0,:,1,1].count_nonzero()==0
    active=torch.nonzero(logits.grad)[:,2:]
    assert ((active>=torch.tensor([4,3]))&(active<=torch.tensor([5,4]))).all()
    with pytest.raises(ValueError,match='originally visible'):
        hard.observe(comparison('human_groups','C','semantic_wrong_source_point_indices',[19]))

def test_reject_only_region_keeps_allowed_roles_unknown_and_excludes_unreviewed_pixels():
    mask=torch.zeros((1,1,8,8),dtype=torch.bool);mask[0,0,4,4]=True
    region=dict(region_id='source_reflection',kind=None,excluded_kinds=['ice_paint'],authority={'reference':'original review'})
    hard=NativeRetentionHardExamples(binding=BINDING,annotations=[],classes=CLASSES,
        reviewed_regions=[(region,mask)],whole_shot_role='train')
    hard.observe(comparison('reviewed_regions','source_reflection','semantic_wrong_native_flat_indices',[36]))
    logits=torch.zeros((1,3,8,8),requires_grad=True);primitive=torch.zeros_like(logits)
    hard.loss(logits,primitive,binding=BINDING).backward()
    assert logits.grad[0,1,4,4]>0
    assert logits.grad[0,0,4,4]<0 and logits.grad[0,2,4,4]<0
    with pytest.raises(ValueError,match='Unknown region'):
        hard.observe(comparison('reviewed_regions','source_reflection','semantic_wrong_native_flat_indices',[37]))

def test_held_shot_and_changed_source_frame_cannot_train_hard_examples():
    hard=NativeRetentionHardExamples(binding=BINDING,annotations=[],classes=CLASSES,whole_shot_role='development')
    with pytest.raises(ValueError,match='Held whole shots'):
        hard.loss(torch.zeros((1,3,8,8)),torch.zeros((1,3,8,8)),binding=BINDING)
    changed=deepcopy(BINDING);changed['source_pts']=111
    with pytest.raises(ValueError,match='source_pts'):
        hard.observe(dict(source_binding=changed,newly_lost_confirmed_support=[]))

def test_fractional_feedback_conflict_checks_all_actual_bilinear_source_pixels():
    mask=torch.zeros((1,1,8,8),dtype=torch.bool);mask[0,0,4,4]=True
    feedback=[dict(kind='shadow_or_reflection',uv=[[3.25,3.5]])]
    with pytest.raises(ValueError,match='original reviewed'):
        protect_feedback_from_conflicting_original_regions(feedback,[(dict(kind='ice_paint'),mask)],native_hw=(8,8))
    protect_feedback_from_conflicting_original_regions(feedback,[(dict(kind=None,excluded_kinds=['ice_paint']),mask)],native_hw=(8,8))
