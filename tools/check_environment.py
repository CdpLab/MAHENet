"""Inspect required native operators without silently substituting implementations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,torch,importlib.metadata as md
def main():
 from third_party.vmamba.csms6s import WITH_SELECTIVESCAN_OFLEX
 from mmcv.ops.carafe import carafe
 result={'python_packages':{},'cuda_available':torch.cuda.is_available(),'native_selective_scan_oflex':bool(WITH_SELECTIVESCAN_OFLEX),'native_carafe_module':carafe.__module__}
 for name in ['torch','torchvision','timm','triton','mmcv-full','einops','numpy','Pillow']:
  result['python_packages'][name]=md.version(name)
 assert WITH_SELECTIVESCAN_OFLEX
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
