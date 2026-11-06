import torch
from torch import nn
from torch.nn import functional as F
from .geometry import pixel_box,canvas_box,canvas_content
from .alignment import unpad,align_local_logits,align_global_logits
def checkpoint_vss_forward(block,value):
    if block.training and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(block._forward,value,use_reentrant=False,preserve_rng_state=True)
    return block._forward(value)

def enable_block_checkpointing(module):
    # Change execution on these model instances only. Preserve all parameter
    # names and math; do not edit the pinned author source or use reentrant DDP.
    from types import MethodType
    count=0
    for block in module.modules():
        if type(block).__name__=='VSSBlock':
            assert hasattr(block,'_forward')
            assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in block.modules())
            block.forward=MethodType(checkpoint_vss_forward,block)
            count+=1
    return count

def fp32_frequency(module,high,low,training):
    # MMCV's native CARAFE does not dispatch BF16. Disable autocast for every
    # operation inside FreqFusion, including the mask-producing convolutions.
    from .fusion import fuse_frequency
    with torch.autocast(high.device.type,enabled=False):
        return fuse_frequency(module,high.float(),low.float(),training)

def local_size(size,region):
    a,b,c,d=pixel_box(size,region)
    return c-a,d-b

def local_content(sizes,regions):
    return [canvas_content(local_size(s,r),512) for s,r in zip(sizes,regions)]

def boxes(sizes,regions,canvas=512):
    return [canvas_box(s,r,canvas) for s,r in zip(sizes,regions)]

def align_global(logits,sizes):
    return torch.stack([align_global_logits(x,s) for x,s in zip(logits,sizes)])

def crop_global(feature,size,region):
    x0,y0,x1,y1=canvas_box(size,region,feature.shape[-1])
    if x1<=x0 or y1<=y0:raise ValueError('Empty global ROI')
    return feature[...,y0:y1,x0:x1]

def crop_local(feature,size,region):
    return unpad(feature,local_size(size,region))

def selected_regions(logits,sizes):
    probability=logits.detach().float().softmax(1)
    entropy=-(probability*probability.clamp_min(1e-8).log()).sum(1)
    regions=[]
    for i,size in enumerate(sizes):
        scores=[]
        for r in range(4):
            x0,y0,x1,y1=canvas_box(size,r)
            scores.append(entropy[i,y0:y1,x0:x1].mean())
        regions.append(int(torch.stack(scores).argmax()))
    return regions

def aligned_local(local,sizes,regions):
    values=[];supports=[]
    for x,size,r in zip(local,sizes,regions):
        a,b,c,d=canvas_box(size,r)
        z=align_local_logits(x,size,r)
        values.append(F.pad(z,(a,512-c,b,512-d)))
        support=torch.ones((d-b,c-a),device=z.device,dtype=torch.bool)
        supports.append(F.pad(support,(a,512-c,b,512-d)))
    return torch.stack(values),torch.stack(supports)

def baseline_logits(global_logits,local_logits,sizes,regions):
    result=[];masks=[]
    for g,l,size,r in zip(global_logits,local_logits,sizes,regions):
        x0,y0,x1,y1=canvas_box(size,r)
        local=align_local_logits(l,size,r)
        local_conf=local.float().softmax(0).amax(0)
        global_conf=g[:,y0:y1,x0:x1].float().softmax(0).amax(0)
        # The acceptance decision is intentionally the original discrete policy.
        accept=(local_conf>global_conf+.05).detach()
        accept[:4]=False;accept[-4:]=False;accept[:,:4]=False;accept[:,-4:]=False
        mask=F.pad(accept,(x0,512-x1,y0,512-y1))
        placed=F.pad(local,(x0,512-x1,y0,512-y1))
        result.append(torch.where(mask[None],placed,g))
        masks.append(mask)
    return torch.stack(result),torch.stack(masks)
