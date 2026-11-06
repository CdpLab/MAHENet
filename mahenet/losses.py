import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .loss_base import DiceLoss,edge_targets
from .regularizers import global_dice_loss,false_positive_penalty,false_positive_weight
def auxiliary_ce_memory_efficient(auxiliary,labels,weights):
    """Exact dense weighted CE, recomputed one image at a time during backward."""
    valid=labels!=255
    if not valid.any():return auxiliary.sum()*0
    denominator=weights[labels[valid]].sum()
    def image_sum(logits,target):
        full=F.interpolate(logits.unsqueeze(0),size=target.shape[-2:],mode='bilinear',align_corners=False)
        return F.cross_entropy(full,target.unsqueeze(0),weight=weights,ignore_index=255,reduction='sum')
    total=auxiliary.sum()*0
    for logits,target in zip(auxiliary,labels):
        if torch.is_grad_enabled() and logits.requires_grad:
            value=checkpoint(image_sum,logits,target,use_reentrant=True,preserve_rng_state=False)
        else:value=image_sum(logits,target)
        total=total+value
    return total/denominator

def semantic_ce(logits,labels,weights,smoothing):
    if not 0<=smoothing<1:raise ValueError('Invalid smoothing coefficient')
    if not (labels!=255).any():return logits.sum()*0
    return F.cross_entropy(logits,labels,weight=weights,ignore_index=255,label_smoothing=smoothing)


def combined_loss(outputs,labels,config,global_step=0):
    kind=config['loss_kind']
    expected=3 if kind=='mahenet_aux' else 2
    if kind not in ('mahenet','mahenet_aux','upernet','segman') or len(outputs)!=expected:
        raise ValueError('Output/loss contract mismatch')
    values=tuple(x.float() for x in outputs);logits=values[0]
    zero=sum(x.sum()*0 for x in values)
    if not (labels!=255).any() and not config.get('global_dice',False):
        return zero,{k:zero.detach() for k in ('ce','dice','edge','auxiliary_ce')}
    weights=logits.new_tensor(config['class_weights'])
    if weights.shape!=(37,) or not torch.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('Expected 37 positive finite class weights')
    ce=semantic_ce(logits,labels,weights,config.get('label_smoothing',0.))
    dice=global_dice_loss(logits,labels) if config.get('global_dice',False) else DiceLoss(37,255)(logits,labels)
    edge=zero;extra=zero
    if kind!='upernet' and config['edge_weight']:
        auxiliary=values[1]
        targets,valid=edge_targets(labels,auxiliary.shape[-2:],255)
        if valid.any():
            positives=targets[valid].sum();negatives=valid.sum()-positives
            edge=F.binary_cross_entropy_with_logits(auxiliary[valid],targets[valid],
                pos_weight=(negatives/positives.clamp_min(1)).clamp(max=100))
    if kind in ('upernet','mahenet_aux'):
        auxiliary=values[1] if kind=='upernet' else values[2]
        if kind=='mahenet_aux':extra=auxiliary_ce_memory_efficient(auxiliary,labels,weights)
        else:
            resized=F.interpolate(auxiliary,size=labels.shape[-2:],mode='bilinear',align_corners=False)
            extra=F.cross_entropy(resized,labels,weight=weights,ignore_index=255)
    loss=config['alpha']*ce+config.get('dice_weight',1-config['alpha'])*dice+config['edge_weight']*edge+config.get('auxiliary_weight',0.)*extra
    false_positive=zero
    fp_weight=false_positive_weight(global_step) if config.get('false_positive',False) else 0.
    if config.get('false_positive',False):
        false_positive=checkpoint(false_positive_penalty,logits,labels,use_reentrant=False) if torch.is_grad_enabled() else false_positive_penalty(logits,labels)
        loss=loss+fp_weight*false_positive
    return loss,{'ce':ce.detach(),'dice':dice.detach(),'edge':edge.detach(),'auxiliary_ce':extra.detach(),'false_positive':false_positive.detach(),'false_positive_weight':logits.new_tensor(fp_weight)}


