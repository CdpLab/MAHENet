"""Portable paths and validation for the published training entry points."""
import os,json,hashlib
from pathlib import Path
TASK=ROOT=REPO=Path(__file__).resolve().parents[1]
PROTOCOL_TASK=REPO/'protocol'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):
 value=json.loads(Path(path).read_text())
 if isinstance(value,dict) and 'data' in value and 'training' in value:
  value['data']['root']=str(Path(os.environ.get('MAHENET_DATA',value['data']['root'])).resolve())
  value['data']['manifest']=str((REPO/value['data']['manifest']).resolve())
  for block,key in [('model','pretrained_checkpoint'),('architecture','global_checkpoint'),('architecture','expert_checkpoint')]:
   item=value.get(block,{}).get(key)
   if item:value[block][key]=str((REPO/item).resolve())
 return value
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def validate(c):
 assert c['classes']==37 and c['seed']==3407 and c['training']['epochs']==20
 assert c['training']['world_size']==2 and c['training']['per_gpu_batch_size']==12 and c['training']['accumulation']==1
 assert digest(c['data']['manifest'])==c['manifest_sha256']
 assert c['data']['image_size']==512 and c['data']['label_protocol']=='dms37'
def verify():
 snap=json.loads((REPO/'source_snapshot.json').read_text())
 for p,h in snap.items():
  if digest(REPO/p)!=h:raise RuntimeError('Source changed since snapshot: '+p)
 return snap
verify_frozen=verify
validate_config=validate
def validate_resume(saved,c):
 if saved['config']!=c:raise ValueError('Resume configuration mismatch')
