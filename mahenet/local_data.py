import json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageEnhance
from torchvision.transforms import functional as TF
from .data_base import DMS37Dataset
from .augmentation import CropDMS37Dataset,letterbox,normalized
from .geometry import selector_holdout,pixel_box,label_box
from .paths import CACHE
def cache_path(name):
    if Path(name).name!=name:raise ValueError('Expected source filename')
    return CACHE/name


def cached(name):
    with Path(str(cache_path(name))+'.npz').open('rb') as stream:
        with np.load(stream) as z:return {k:z[k] for k in z.files}


def crop_input(image,label,index,degraded=False,augment=False):
    if degraded:
        # Match the whole-image512 information bottleneck, then recover exact source coordinates.
        w,h=image.size;scale=512/max(w,h)
        image=image.resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
    image_size=image.size
    image=image.crop(pixel_box(image_size,index));label=label.crop(label_box(image_size,label.size,index))
    reference=image.size
    image=letterbox(image,512);label=letterbox(label,512,True,255,reference)
    if augment:
        if random.random()<.5:image,label=TF.hflip(image),TF.hflip(label)
        image=ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))).enhance(random.uniform(.8,1.2))
    return normalized(image),torch.from_numpy(np.asarray(label,dtype=np.int64).copy())


class LocalDataset(DMS37Dataset):
    def __init__(self,*args,mode='expert_train',selected=None,degraded=False,**kwargs):
        super().__init__(*args,**kwargs);self.mode=mode;self.selected=selected;self.degraded=degraded
        if mode=='expert_train':self.samples=[s for s in self.samples if not selector_holdout(s[0].name)]
        if mode=='selector_targets':self.samples=[s for s in self.samples if selector_holdout(s[0].name)]
    def __getitem__(self,index):
        image_path,label_path=self.samples[index];name=image_path.name
        z=cached(name);region=int(z['entropy_index']) if self.selected is None else self.selected[name]
        with Image.open(image_path) as im:image=im.convert('RGB')
        with Image.open(label_path) as im:label=self._encode_label(im)
        tensor,target=crop_input(image,label,region,degraded=self.degraded,augment=self.mode=='expert_train')
        if self.mode!='expert_train':
            _,target_image=self._resize_pair(image,label)
            target=torch.from_numpy(np.asarray(target_image,dtype=np.int64).copy())
        return tensor,target,name


def make_dataset(c,split):
    kw=dict(root=c['data']['root'],split=split,image_size=512,augment=False,manifest=c['data']['manifest'])
    if c['variant']=='expert':return LocalDataset(**kw,mode='expert_train' if split=='train' else 'validation')
    if split=='train':
        kw['augment']=True;return CropDMS37Dataset(**kw,strategy='crop25')
    return DMS37Dataset(**kw)


def move_input(value,device):return value.to(device,non_blocking=True)
def batch_size(value):return len(value)
