"""Strict continuation contract; no model/optimizer/RNG reset."""
import random
import numpy as np
import torch

def validate_resume(saved,history,config):
    assert saved['config']==config,'Resume config differs'
    epoch=saved['epoch'];step=saved['global_step']
    assert 1<=epoch<=20 and step==707*epoch
    assert len(history)==epoch and [r['epoch'] for r in history]==list(range(1,epoch+1))
    assert history[-1]['global_step']==step==saved['scheduler']['last_epoch']
    assert len(saved['rng_by_rank'])==2 and saved['optimizer']['state']
    assert saved['head_only']==(epoch<=2)
    if 'miou' in history[0]:assert abs(saved['best_miou']-max(r['miou'] for r in history))<1e-12
    return epoch+1,step

def restore_rng(saved,rank):
    rng=saved['rng_by_rank'][rank]
    random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
