import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from third_party.freqfusion import FreqFusion
def make_freqfusion(hr_channels,lr_channels,compressed=64):
 from mmcv.ops.carafe import carafe
 from third_party import freqfusion
 if freqfusion.carafe is not carafe:raise RuntimeError("Native MMCV CARAFE is required")
 return FreqFusion(hr_channels,lr_channels,scale_factor=1,lowpass_kernel=5,highpass_kernel=3,
  up_group=1,compressed_channels=compressed,align_corners=False,feature_resample=True,
  feature_resample_group=4,comp_feat_upsample=True,use_high_pass=True,use_low_pass=True,
  hr_residual=True,semi_conv=True,hamming_window=True,feature_resample_norm=True)
def fuse_frequency(module,high,low,training):
 if high.shape[-2]!=2*low.shape[-2] or high.shape[-1]!=2*low.shape[-1]:raise ValueError("Expected adjacent pyramid levels")
 if training and torch.is_grad_enabled():_,high,low=checkpoint(module,high,low,use_checkpoint=False,use_reentrant=False)
 else:_,high,low=module(high,low,use_checkpoint=False)
 return high+low
