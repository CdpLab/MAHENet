"""Launch global-branch, local-expert, or joint MAHENet training."""
import argparse,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',required=True,choices=['global','expert','joint'])
 p.add_argument('--data-root',required=True);p.add_argument('--cache',default=str(ROOT/'cache'))
 p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--smoke',action='store_true')
 a=p.parse_args();os.environ['MAHENET_DATA']=str(Path(a.data_root).resolve());os.environ['MAHENET_CACHE']=str(Path(a.cache).resolve())
 stage='mahenet' if a.stage=='joint' else a.stage
 sys.argv=[str(ROOT/'train.py'),'--config',str(ROOT/'configs'/(stage+'.json')),'--output',a.output]
 if a.resume:sys.argv+=['--resume',a.resume]
 if a.smoke:sys.argv+=['--smoke']
 runpy.run_module('mahenet.train_joint' if a.stage=='joint' else 'mahenet.train_branch',run_name='__main__')
if __name__=='__main__':main()
