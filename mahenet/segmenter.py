import torch
from torch import nn
from torch.nn import functional as F
from .backbone import SemanticEncoder
from .layers import TextureEncoder,MaterialAwareAttention,ASPPFPNFusion,EdgeEnhancedDecoder,resize_bilinear
from .fusion import make_freqfusion,fuse_frequency
from .context import SpatialMaterialContext
class Segmenter(nn.Module):
 def __init__(self,checkpoint=None):
  super().__init__()
  self.semantic_encoder=SemanticEncoder(checkpoint)
  # Preserve the original seeded module-construction sequence used for branch training.
  self.texture_encoder=TextureEncoder();self.material_attention=MaterialAwareAttention(96)
  self.fusion=ASPPFPNFusion();self.decoder=EdgeEnhancedDecoder(96,37)
  self.fusion,self.decoder,self.texture_adapter=ASPPFPNFusion(out_channels=192),EdgeEnhancedDecoder(192,37),nn.Conv2d(96,192,1,bias=False)
  self.frequency_fusion=make_freqfusion(192,192);self.frequency_fusion_p3=make_freqfusion(192,192)
  self.material_context=SpatialMaterialContext()
  del self.texture_encoder,self.material_attention,self.texture_adapter
 def forward(self,image):
  s2,s3,s4,s5=self.semantic_encoder(image);lateral=self.fusion.lateral[0](s2)
  p5=self.fusion.deep_aspp(s5)
  p4=self.fusion.refine[2](resize_bilinear(p5,size=s4.shape[-2:])+self.fusion.lateral[2](s4))
  high=self.fusion.lateral[1](s3)
  p3=fuse_frequency(self.frequency_fusion_p3,high,p4,self.training)
  p3=self.fusion.refine[1](p3);p3,coarse=self.material_context(p3)
  p2=fuse_frequency(self.frequency_fusion,lateral,p3,self.training)
  fused=self.fusion.refine[0](p2)
  logits,edge=self.decoder(fused,image.shape[-2:])
  return logits,edge,coarse
