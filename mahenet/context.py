"""Image-local material-region aggregation, inspired by OCR, not official OCR."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpatialMaterialContext(nn.Module):
    def __init__(self,in_channels=192,classes=37,channels=64):
        super().__init__();self.in_channels=in_channels;self.channels=channels
        self.coarse=nn.Conv2d(in_channels,classes,1)
        self.project=nn.Conv2d(in_channels,channels,1,bias=False)
        self.query=nn.Conv2d(in_channels,channels,1,bias=False)
        self.key=nn.Linear(channels,channels,bias=False)
        self.value=nn.Linear(channels,channels,bias=False)
        self.output=nn.Conv2d(channels,in_channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.in_channels:raise ValueError('Context requires BCHW with declared channels')
        batch,_,height,width=x.shape
        coarse=self.coarse(x)
        weights=coarse.flatten(2).softmax(dim=2)
        embedded=F.gelu(self.project(x))
        # PyTorch2.1 CPU exact-GELU backward mishandles this transposed BMM's
        # strided upstream gradient. Layout normalization preserves the same
        # activation/derivative and also works for CUDA; no formula is changed.
        if embedded.requires_grad:embedded.register_hook(lambda gradient:gradient.contiguous())
        embedding=embedded.flatten(2).transpose(1,2)
        prototypes=torch.bmm(weights,embedding)
        query=self.query(x).flatten(2).transpose(1,2)
        attention=(torch.bmm(query,self.key(prototypes).transpose(1,2))/math.sqrt(self.channels)).softmax(dim=2)
        context=torch.bmm(attention,self.value(prototypes)).transpose(1,2).reshape(batch,self.channels,height,width)
        return x+self.output(context),coarse
