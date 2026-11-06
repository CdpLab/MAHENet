"""Three explicitly bounded training objectives, independent of test metrics."""
import torch
import torch.distributed as dist
from torch.nn import functional as F


def global_dice_loss(logits,labels,ignore_index=255,eps=1e-6):
    valid=labels!=ignore_index;safe=labels.masked_fill(~valid,0)
    probability=logits.float().softmax(1)*valid.unsqueeze(1)
    target=F.one_hot(safe,logits.shape[1]).permute(0,3,1,2).float()*valid.unsqueeze(1)
    intersection=(probability*target).sum((0,2,3))
    denominator=probability.square().sum((0,2,3))+target.square().sum((0,2,3))
    stats=torch.stack((intersection,denominator,target.sum((0,2,3))))
    # Every rank enters even when its local batch is entirely ignored.
    if dist.is_initialized() and dist.get_world_size()>1:
        from torch.distributed.nn.functional import all_reduce
        stats=all_reduce(stats,op=dist.ReduceOp.SUM)
    present=stats[2]>0
    if not present.any():return logits.sum()*0
    return (1-(2*stats[0,present]+eps)/(stats[1,present]+eps)).mean()


def reliable_interior(labels,ignore_index=255):
    x=F.pad(labels.float().unsqueeze(1),(1,1,1,1),value=float(ignore_index))
    high=F.max_pool2d(x,3,stride=1).squeeze(1)
    low=-F.max_pool2d(-x,3,stride=1).squeeze(1)
    return (labels!=ignore_index)&(high==low)


def false_positive_penalty(logits,labels,threshold=.5,ignore_index=255):
    valid=reliable_interior(labels,ignore_index)
    safe=labels.masked_fill(labels==ignore_index,0)
    values,indices=logits.float().topk(2,dim=1)
    top_is_gt=indices[:,0]==safe
    wrong_logit=torch.where(top_is_gt,values[:,1],values[:,0])
    wrong_class=torch.where(top_is_gt,indices[:,1],indices[:,0])
    probability=(wrong_logit-torch.logsumexp(logits.float(),dim=1)).exp()
    selected=valid&(probability.detach()>threshold)
    if not selected.any():return logits.sum()*0
    p=probability[selected].clamp(max=1-1e-6);category=wrong_class[selected]
    penalty=-torch.log1p(-p)
    sums=logits.new_zeros(logits.shape[1],dtype=torch.float32).scatter_add(0,category,penalty)
    counts=logits.new_zeros(logits.shape[1],dtype=torch.float32).scatter_add(0,category,torch.ones_like(penalty))
    active=counts>0
    return (sums[active]/counts[active]).mean()


def false_positive_weight(global_step,maximum=.02,warmup_steps=1574):
    return maximum*min(1.,max(0.,global_step/warmup_steps))
