"""Evaluate all37 classes and boundary F1 using the fixed data protocol."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mahenet.api import load_model,to_device,setup
from mahenet.data import PairedDataset
from mahenet.metrics import SegmentationMetrics
from mahenet.branch import finish_metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data-root',required=True);p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 setup();model,c=load_model(a.checkpoint)
 root=Path(__file__).resolve().parent;c['data']['root']=a.data_root;c['data']['manifest']=str(root/'protocol/splits.json')
 ds=PairedDataset(c,a.split);assert len(ds)==(4047 if a.split=='test' else 4044)
 metrics=SegmentationMetrics(37,255,2);names_hash=hashlib.sha256();target_hash=hashlib.sha256();n=0
 with torch.inference_mode():
  for inputs,labels,_,names in DataLoader(ds,batch_size=1,shuffle=False,num_workers=4,pin_memory=True):
   prediction=model(to_device(inputs,'cuda'))['final'].argmax(1).cpu()
   metrics.update(prediction,labels);n+=len(names)
   target_hash.update(labels.contiguous().numpy().tobytes())
   for name in names:names_hash.update((name+'\n').encode())
   if n%500==0:print(json.dumps({'images':n,'total':len(ds)}),flush=True)
 result=finish_metrics(metrics,True)
 expected=json.loads((root/'protocol/pixel_audit.json').read_text())[a.split]
 assert result['gt_pixels']==[row['pixels_at512'] for row in expected['class_support']]
 result.update(images=n,evaluated_images=n,split=a.split,parameters=sum(p.numel() for p in model.parameters()),
   boundary_raw_counts={k:int(getattr(metrics,k)) for k in ['boundary_pred_matched','boundary_gt_matched','boundary_pred','boundary_gt']},
   ground_truth_sha256=target_hash.hexdigest(),image_order_sha256=names_hash.hexdigest(),
   precision='FP32+TF32',checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest())
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps({k:result[k] for k in ['miou','pacc','macc','bf1']}))
if __name__=='__main__':main()
