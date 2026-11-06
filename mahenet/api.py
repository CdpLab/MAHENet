"""Load a self-contained inference checkpoint without branch initialization files."""
from pathlib import Path
import json,torch
from .model import MAHENet
from .augmentation import normalized,letterbox
from .geometry import pixel_box
def load_model(path,device='cuda'):
 saved=torch.load(path,map_location='cpu')
 config=saved['config']
 config['model']['pretrained_checkpoint']=None
 config['architecture']['global_checkpoint']=None;config['architecture']['expert_checkpoint']=None
 model=MAHENet(config)
 model.load_state_dict(saved['model'],strict=True)
 model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).float().eval()
 return model,config
def pack_image(image,global_size):
 return {'global':normalized(letterbox(image,global_size)).unsqueeze(0),
  'local':torch.stack([normalized(letterbox(image.crop(pixel_box(image.size,r)),512)) for r in range(4)]).unsqueeze(0),
  'size':torch.tensor([image.size])}
def to_device(inputs,device):
 return {k:v.to(device,non_blocking=True) if k=='global' else v for k,v in inputs.items()}
def setup():
 torch.set_num_threads(4);torch.manual_seed(3407)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
