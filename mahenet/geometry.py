"""Shared normalized crop definitions for images, labels and writeback."""
import hashlib


def selector_holdout(name):return int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%10==0


def selector_dev(name):return int(hashlib.sha256(('dev:'+name).encode()).hexdigest()[:8],16)%5==0


def pixel_box(size,index):
    if index not in range(4):raise ValueError('Four candidate regions only')
    w,h=size;x=index%2;y=index//2
    return (round(w*x/2),round(h*y/2),round(w*(x+1)/2),round(h*(y+1)/2))


def canvas_content(size,canvas=512):
    w,h=size;scale=canvas/max(w,h);nw,nh=round(w*scale),round(h*scale)
    left,top=(canvas-nw)//2,(canvas-nh)//2
    return left,top,nw,nh


def canvas_box(size,index,canvas=512):
    left,top,w,h=canvas_content(size,canvas);sw,sh=size
    x0,y0,x1,y1=pixel_box(size,index)
    return left+round(x0*w/sw),top+round(y0*h/sh),left+round(x1*w/sw),top+round(y1*h/sh)


def label_box(image_size,label_size,index):
    sw,sh=image_size;lw,lh=label_size;x0,y0,x1,y1=pixel_box(image_size,index)
    return round(x0*lw/sw),round(y0*lh/sh),round(x1*lw/sw),round(y1*lh/sh)
