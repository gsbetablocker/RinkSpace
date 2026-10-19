import pytest
torch=pytest.importorskip('torch')
from rinkspace.vision_network import RinkVisionNet
from types import SimpleNamespace,MethodType


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA autocast regression')
def test_semantic_only_current_frame_has_gradient_after_previous_no_grad():
    torch.manual_seed(7)
    network=RinkVisionNet(num_classes=6,width=8).cuda()
    for name,parameter in network.named_parameters():
        parameter.requires_grad_(name.startswith('semantic_head.'))
    images=torch.rand((1,2,3,32,32),device='cuda')
    frozen={name:parameter.detach().clone() for name,parameter in network.named_parameters()
            if not parameter.requires_grad}
    optimizer=torch.optim.AdamW(network.parameters(),lr=1e-4)
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        with torch.no_grad():previous=network(images[:,:1])
        current=network(images[:,1:2],hidden=previous['hidden'])
        loss=current['logits'][:,0].float().square().mean()
    assert loss.requires_grad
    loss.backward()
    assert bool(torch.isfinite(network.semantic_head.weight.grad).all())
    assert bool(network.semantic_head.weight.grad.abs().sum()>0)
    optimizer.step()
    assert all(torch.equal(frozen[name],parameter) for name,parameter in network.named_parameters()
               if name in frozen)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA autocast regression')
def test_full_native_cached_context_matches_uncached_inference():
    torch.manual_seed(19)
    net=RinkVisionNet(num_classes=6,width=8).cuda().eval()
    net.backbone=torch.nn.Identity()
    net.backbone.config=SimpleNamespace(fpn_hidden_size=3)
    net.backbone_projections=torch.nn.ModuleList([
        torch.nn.Conv2d(3,32,1).cuda() for _ in range(2)])
    net.backbone_detail=torch.nn.Conv2d(3,8,1).cuda()
    def features(self,frame):
        return tuple(torch.nn.functional.interpolate(frame,size=size,mode='bilinear',
            align_corners=False).contiguous(memory_format=torch.channels_last)
            for size in ((19,29),(11,17)))
    net._pretrained_features=MethodType(features,net)
    images=torch.rand((1,2,3,72,112),device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        context=net.encode_context(images)
        native=net(images)
        cached=net(images,context_features=context)
        explicit=net(images,context_features=context,context_crop=(0,0,112,72),context_native_hw=(72,112))
        crop=net(images[:,:,:,:48,:80],context_features=context,
            context_crop=(0,0,80,48),context_native_hw=(72,112))
    for key in ('logits','primitive_logits','hidden'):
        assert torch.equal(native[key],cached[key])
        assert torch.equal(native[key],explicit[key])
    assert crop['logits'].shape[-2:]==(48,80)
    assert torch.isfinite(crop['logits']).all()
