from .paths import REPO
from .train_runtime import digest,verify_frozen,validate_config,validate_resume
from .segmenter import Segmenter
from .losses import combined_loss
from .branch import evaluate37
from .augmentation import CropDMS37Dataset
from .data_base import DMS37Dataset
from .local_data import LocalDataset
ROOT=TASK=REPO;PROTOCOL_TASK=REPO/'protocol'
def build(c):return Segmenter(c['model']['pretrained_checkpoint'])
def gradient_groups(name):
 return ('semantic_encoder.','fusion.','decoder.','material_context.','frequency_fusion.')
def make_dataset(c,split):
 kw=dict(root=c['data']['root'],split=split,image_size=512,augment=False,manifest=c['data']['manifest'])
 if c['variant']=='expert':return LocalDataset(**kw,mode='expert_train' if split=='train' else 'validation')
 if split=='train':
  kw['augment']=True
  return CropDMS37Dataset(**kw,strategy='crop25')
 return DMS37Dataset(**kw)
def move_input(value,device):return value.to(device,non_blocking=True)
def batch_size(value):return len(value)
