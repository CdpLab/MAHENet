import torch
from torch import nn
from torch.nn import functional as F
from third_party.vmamba.vmamba import Backbone_VSSM
from third_party.vmamba.csms6s import WITH_SELECTIVESCAN_OFLEX
class SemanticEncoder(nn.Module):
 def __init__(self,checkpoint=None):
  super().__init__()
  if not WITH_SELECTIVESCAN_OFLEX:raise RuntimeError("Build the native VMamba selective_scan_oflex extension first")
  self.backbone=Backbone_VSSM(out_indices=(0,1,2,3),pretrained=None,dims=96,
   depths=(2,2,8,2),ssm_d_state=1,ssm_dt_rank="auto",ssm_ratio=1.0,
   ssm_conv=3,ssm_conv_bias=False,forward_type="v05_noz",mlp_ratio=4.0,
   downsample_version="v3",patchembed_version="v2",drop_path_rate=.2,norm_layer="ln2d")
  if checkpoint:
   saved=torch.load(checkpoint,map_location="cpu")["model"]
   report=self.backbone.load_state_dict(saved,strict=False)
   missing={f"outnorm{i}.{k}" for i in range(4) for k in ("weight","bias")}
   if set(report.missing_keys)!=missing or any(not k.startswith("classifier.") for k in report.unexpected_keys):
    raise ValueError("Unexpected encoder initialization mismatch")
 def forward(self,image):return self.backbone(image)
