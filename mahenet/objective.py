import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .losses import combined_loss
def conflict_ce(logits,target,global_pred,local_pred,support,weights):
    valid=support&(target!=255)
    cases=[valid&(global_pred==target)&(local_pred!=target),
           valid&(global_pred!=target)&(local_pred==target)]
    if not any(bool(mask.any()) for mask in cases):return logits.sum()*0
    pixel=F.cross_entropy(logits.float(),target,weight=weights,ignore_index=255,reduction='none')
    sums=[]
    for mask in cases:
        if mask.any():sums.append(pixel[mask].sum()/weights[target[mask]].sum().clamp_min(1))
    return torch.stack(sums).mean()

def loss_for(outputs,global_target,local_target,c,step):
    original=c['loss']
    final_config=copy.deepcopy(original)
    final_config.update(loss_kind='mahenet',edge_weight=0.,auxiliary_weight=0.)
    final,parts=combined_loss((outputs['final'],outputs['final'].new_zeros(1)),global_target,final_config,step)
    global_loss,_=combined_loss(outputs['global'],global_target,original,step)
    local_loss,_=combined_loss(outputs['local'],local_target,original,step)
    args=(outputs['final'],global_target,outputs['global_pred'],outputs['local_pred'],
          outputs['local_support'],outputs['final'].new_tensor(original['class_weights']))
    conflict=checkpoint(conflict_ce,*args,use_reentrant=False) if torch.is_grad_enabled() else conflict_ce(*args)
    total=final+.25*global_loss+.25*local_loss+.1*conflict
    return total,{'final':final.detach(),'global':global_loss.detach(),'local':local_loss.detach(),'conflict':conflict.detach()}
