"""Audit fixed image lists,37-class mapping,and the adaptation subset."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args()
 data=Path(a.data_root);manifest=json.loads((ROOT/'protocol/splits.json').read_text());counts={}
 seen=set()
 for split,n in [('train',18873),('validation',4044),('test',4047)]:
  rows=manifest['splits'][split];assert len(rows)==n
  for row in rows:
   assert row['image_path'] not in seen;seen.add(row['image_path'])
   for key in ['image_path','label_path']:
    if not (data/row[key]).is_file():raise FileNotFoundError(data/row[key])
  counts[split]=len(rows)
 actual=[r['image_path'] for r in manifest['splits']['train'] if int(hashlib.sha256(Path(r['image_path']).name.encode()).hexdigest()[:8],16)%10!=0]
 assert actual==(ROOT/'protocol/train_adaptation.txt').read_text().splitlines() and len(actual)==16968
 print(json.dumps({'complete':True,'split_counts':counts,'adaptation_training_images':len(actual),'classes':37,'ignore_index':255}))
if __name__=='__main__':main()
