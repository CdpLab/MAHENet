"""Generate the frozen global-branch cache used to train the local expert."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,os,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
 os.environ['MAHENET_CACHE']=str(Path(a.cache).resolve())
 import numpy as np,torch
 from PIL import Image
 from torch.utils.data import DataLoader
 from mahenet.segmenter import Segmenter
 from mahenet.data_base import DMS37Dataset
 from mahenet.selection_features import region_features
 from mahenet.api import setup
 root=Path(__file__).resolve().parents[1];setup()
 saved=torch.load(a.checkpoint,map_location='cpu');model=Segmenter()
 model.load_state_dict(saved['model'],strict=True);model=model.cuda().float().eval()
 folder=Path(a.cache);folder.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for split in ['train','validation']:
   ds=DMS37Dataset(a.data_root,split,512,False,root/'protocol/splits.json')
   paths={ip.name:ip for ip,_ in ds.samples}
   for images,labels,names in DataLoader(ds,batch_size=4,num_workers=4,shuffle=False,pin_memory=True):
    logits=model(images.cuda())[0].float();confidence,pred=logits.softmax(1).max(1)
    for i,name in enumerate(names):
     output=folder/(name+'.npz')
     if output.exists():raise FileExistsError(output)
     with Image.open(paths[name]) as im:size=im.size
     features=region_features(logits[i],images[i].cuda(),size).cpu().numpy()
     temp=Path(str(output)+'.tmp.npz')
     np.savez_compressed(temp,pred=pred[i].cpu().numpy().astype('uint8'),confidence=confidence[i].cpu().numpy().astype('float16'),features=features,entropy_index=np.int64(features[:,37].argmax()),source_size=np.asarray(size,dtype='int32'));temp.replace(output)
if __name__=='__main__':main()
