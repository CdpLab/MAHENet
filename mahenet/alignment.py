import torch
from torch import nn
from torch.nn import functional as F
from .geometry import canvas_content,canvas_box,pixel_box
def unpad(x,source_size):
    left,top,w,h=canvas_content(source_size,x.shape[-1])
    return x[...,top:top+h,left:left+w]

def align_local_logits(local_logits,source_size,index):
    a,b,c,d=pixel_box(source_size,index)
    local=unpad(local_logits,(c-a,d-b))
    x0,y0,x1,y1=canvas_box(source_size,index)
    return F.interpolate(local[None].float(),size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0]

def align_global_logits(logits, source_size):
    """Unpad at native canvas size, resize content, then pad to the fixed512 protocol."""
    left, top, w, h = canvas_content(source_size, logits.shape[-1])
    l, t, ow, oh = canvas_content(source_size, 512)
    content = logits[:, top:top+h, left:left+w].float()
    if (h, w) != (oh, ow):
        content = F.interpolate(content[None], (oh, ow), mode='bilinear', align_corners=False)[0]
    return F.pad(content, (l, 512-l-ow, t, 512-t-oh))
