"""Download released weights with SHA256 verification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='all');a=p.parse_args()
 manifest=json.loads((ROOT/'weights/manifest.json').read_text());items=manifest['weights']+[manifest['encoder']]
 items=[x for x in items if a.name in ['all',x['name'],x.get('stage')]]
 if not items:raise ValueError('Unknown weight name')
 for item in items:
  out=ROOT/'weights'/item['name']
  if out.exists():
   if sha(out)==item['sha256']:print('Verified '+out.name);continue
   raise ValueError('Existing weight checksum mismatch: '+out.name)
  url=item.get('url')
  if not url:raise RuntimeError('This weight is not released yet: '+item['name'])
  temp=out.with_suffix(out.suffix+'.download')
  urllib.request.urlretrieve(url,temp)
  if sha(temp)!=item['sha256']:raise ValueError('Downloaded checksum mismatch: '+out.name)
  temp.replace(out);print('Downloaded '+out.name)
if __name__=='__main__':main()
