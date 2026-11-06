from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image,ImageEnhance
from torchvision.transforms import functional as TF
from .data_base import DMS37Dataset
from .geometry import selector_holdout,pixel_box,label_box
from .augmentation import letterbox,normalized
class PairedDataset(DMS37Dataset):
    def __init__(self,c,split):
        super().__init__(root=c['data']['root'],split=split,image_size=512,augment=False,manifest=c['data']['manifest'])
        self.global_size=c['experiment']['global_size'];self.training=split=='train'
        if self.training:self.samples=[s for s in self.samples if not selector_holdout(s[0].name)]

    def __getitem__(self,index):
        ip,lp=self.samples[index]
        with Image.open(ip) as f:image=f.convert('RGB')
        with Image.open(lp) as f:label=self._encode_label(f)
        if self.training:
            if random.random()<.5:image,label=TF.hflip(image),TF.hflip(label)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.2))
        _,global_label=self._resize_pair(image,label)
        locals=[];targets=[]
        for region in range(4):
            crop=image.crop(pixel_box(image.size,region))
            target=label.crop(label_box(image.size,label.size,region))
            locals.append(normalized(letterbox(crop,512)))
            target=letterbox(target,512,True,255,crop.size)
            targets.append(torch.from_numpy(np.asarray(target,dtype=np.int64).copy()))
        inputs={'global':normalized(letterbox(image,self.global_size)),'local':torch.stack(locals),'size':torch.tensor(image.size)}
        return inputs,torch.from_numpy(np.asarray(global_label,dtype=np.int64).copy()),torch.stack(targets),ip.name

def move(inputs,device):
    return {k:v.to(device,non_blocking=True) if k=='global' else v for k,v in inputs.items()}
