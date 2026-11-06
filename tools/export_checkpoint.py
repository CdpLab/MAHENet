"""Export a completed validation-selected checkpoint for inference or stage initialization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 path=Path(a.checkpoint);out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 saved=torch.load(path,map_location='cpu');history=json.loads((path.parent/'history.json').read_text())
 assert [x['epoch'] for x in history]==list(range(1,21))
 best=max(history,key=lambda x:x['miou']);assert saved['epoch']==best['epoch']
 out.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'model':saved['model'],'config':saved['config'],'selected_epoch':saved['epoch'],'validation_metrics':{k:best[k] for k in ['miou','pacc','macc','bf1']}},out)
 print(str(out))
if __name__=='__main__':main()
