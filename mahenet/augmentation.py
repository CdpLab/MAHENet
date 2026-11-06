import random
import torch
import numpy as np
from PIL import Image,ImageEnhance
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from .data_base import DMS37Dataset
def crop_probability(strategy,epoch):
    if not 1<=epoch<=20:raise ValueError('Expected actual epoch1..20')
    if strategy=='crop25_finish' and epoch>=16:return 0.
    if strategy in ('crop25','crop25_uniform','crop25_finish'):return .25
    if strategy=='none':return 0.
    raise ValueError('Unknown data strategy')

def detail_crop_pair(image,label,rng,probability=.25,rare_priority=.5):
    if not 0<=probability<=1 or not 0<=rare_priority<=1:raise ValueError('Invalid probability')
    if rng.random()>=probability:return image,label,{'cropped':False}
    values=np.asarray(label,dtype=np.uint8);classes=[int(c) for c in np.unique(values) if c<37]
    if not classes:return image,label,{'cropped':False}
    rare=[c for c in (1,32) if c in classes]
    # Consume the priority draw even at zero so the ablation has paired RNG.
    options=rare if rare and rng.random()<rare_priority else classes
    category=rng.choice(options);candidates=np.flatnonzero(values==category)
    y,x=divmod(int(candidates[rng.randrange(len(candidates))]),label.width)
    scale=rng.choice((.5,.65,.8));width=max(1,round(label.width*scale));height=max(1,round(label.height*scale))
    left=min(max(0,x-width//2),label.width-width);top=min(max(0,y-height//2),label.height-height)
    box=(left,top,left+width,top+height)
    norm=(left/label.width,top/label.height,(left+width)/label.width,(top+height)/label.height)
    image_box=tuple(round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(norm))
    return image.crop(image_box),label.crop(box),{'cropped':True,'target_class':category,'label_box':box,'image_box':image_box,'scale':scale}

class CropDMS37Dataset(DMS37Dataset):
    def __init__(self,*args,strategy='crop25',**kwargs):
        super().__init__(*args,**kwargs);self.strategy=strategy;self.epoch=1

    def set_epoch(self,epoch):self.epoch=epoch;crop_probability(self.strategy,epoch)

    def __getitem__(self,index):
        self.detail_rng=random.Random((torch.initial_seed()+index*1000003)%(2**63))
        return super().__getitem__(index)

    def _resize_pair(self,image,label):
        if self.augment:
            image,label,_=detail_crop_pair(image,label,self.detail_rng,crop_probability(self.strategy,self.epoch),0 if self.strategy=='crop25_uniform' else .5)
        return super()._resize_pair(image,label)

def letterbox(image,size,nearest=False,fill=(124,116,104),reference_size=None):
    w,h=reference_size or image.size;scale=size/max(w,h);width,height=round(w*scale),round(h*scale)
    image=TF.resize(image,[height,width],InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR)
    left,top=(size-width)//2,(size-height)//2
    return TF.pad(image,[left,top,size-width-left,size-height-top],fill=fill)

def normalized(image):return TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
