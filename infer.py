"""Predict a 512x512 DMS37 label map from one RGB image."""
import argparse
from pathlib import Path
import torch,numpy as np
from PIL import Image
from mahenet.api import load_model,pack_image,to_device,setup
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();setup();model,c=load_model(a.checkpoint)
 with Image.open(a.image) as im:image=im.convert('RGB')
 with torch.inference_mode():pred=model(to_device(pack_image(image,c['experiment']['global_size']),'cuda'))['final'].argmax(1)[0].cpu().byte().numpy()
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pred,mode='L').save(out)
 print(str(out))
if __name__=='__main__':main()
