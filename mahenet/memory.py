"""Activation recomputation only; same module, parameters, BN updates and RNG."""
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_batchnorm(module):
    saved = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
             for m in module.modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, mean, variance, count in saved:
                m.running_mean.copy_(mean)
                m.running_var.copy_(variance)
                m.num_batches_tracked.copy_(count)


def recompute_module(module, image):
    if module.training and torch.is_grad_enabled():
        return checkpoint(module, image, use_reentrant=False, preserve_rng_state=True,
                          context_fn=lambda: (nullcontext(), preserve_batchnorm(module)))
    return module(image)
