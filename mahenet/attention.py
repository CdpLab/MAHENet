import torch
from torch import nn
from torch.nn import functional as F
from .ops import crop_global,crop_local,local_size
from .geometry import pixel_box,canvas_box,canvas_content
def xy_grid(h,w,device,dtype,rect=(0.,0.,1.,1.)):
    a,b,c,d=rect
    yy,xx=torch.meshgrid((torch.arange(h,device=device,dtype=torch.float32)+.5)/h,
                        (torch.arange(w,device=device,dtype=torch.float32)+.5)/w,indexing='ij')
    return torch.stack((a+xx*(c-a),b+yy*(d-b)))[None].to(dtype)

def region_rect(size,r):
    a,b,c,d=pixel_box(size,r);w,h=size
    return a/w,b/h,c/w,d/h

def attention(query,key,value,heads=2):
    b,c,h,w=query.shape;m=key.shape[-1];dh=c//heads
    q=query.flatten(2).transpose(1,2).reshape(b,h*w,heads,dh).transpose(1,2)
    k=key.transpose(1,2).reshape(b,m,heads,dh).transpose(1,2)
    v=value.transpose(1,2).reshape(b,m,heads,dh).transpose(1,2)
    z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    return z.transpose(1,2).reshape(b,h*w,c).transpose(1,2).reshape(b,c,h,w)

class GlobalGuide(nn.Module):
    def __init__(self,channels=192,hidden=32):
        super().__init__()
        self.query=nn.Conv2d(channels,hidden,1,bias=False)
        self.key=nn.Conv2d(channels,hidden,1,bias=False)
        self.value=nn.Conv2d(channels,hidden,1,bias=False)
        self.position=nn.Conv2d(2,hidden,1,bias=False)
        self.output=nn.Conv2d(hidden,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,local,global_,sizes,regions):
        values=[]
        for i,(size,r) in enumerate(zip(sizes,regions)):
            query=self.query(local[i:i+1]);h,w=query.shape[-2:]
            rect=region_rect(size,r)
            l,t,cw,ch=canvas_content(local_size(size,r),w)
            # Map local canvas centers to source-image coordinates; mask its padding.
            uv=xy_grid(h,w,query.device,torch.float32)
            px,py=uv[:,0:1]*w,uv[:,1:2]*h
            valid=(px>=l)&(px<l+cw)&(py>=t)&(py<t+ch)
            pos=torch.cat((rect[0]+(px-l)/cw*(rect[2]-rect[0]),
                           rect[1]+(py-t)/ch*(rect[3]-rect[1])),1).to(query.dtype)
            query=query+self.position(pos)
            # Pool only unpadded valid features; no padded token can be attended.
            from .alignment import unpad
            scene=unpadded=unpad(global_[i:i+1],size)
            roi=crop_global(global_[i:i+1],size,r)
            keys=[];vals=[]
            for source,extent in ((scene,(0.,0.,1.,1.)),(roi,rect)):
                tokens=F.adaptive_avg_pool2d(source,(8,8))
                keys.append((self.key(tokens)+self.position(xy_grid(8,8,tokens.device,tokens.dtype,extent))).flatten(2))
                vals.append(self.value(tokens).flatten(2))
            context=attention(query,torch.cat(keys,2),torch.cat(vals,2))
            values.append(local[i:i+1]+self.output(context)*valid.to(context.dtype))
        return torch.cat(values)
