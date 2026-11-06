import json,unittest,hashlib
from pathlib import Path
import torch
from mahenet.geometry import pixel_box,canvas_box,canvas_content,selector_holdout
from mahenet.alignment import align_global_logits
from mahenet.attention import GlobalGuide
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 def test_class_mapping_contiguous(self):
  p=json.loads((ROOT/'protocol/dms37.json').read_text())
  self.assertEqual([r['target_id'] for r in p['mapping']],list(range(37)))
  self.assertEqual(p['ignore_index'],255)
 def test_all_splits_disjoint(self):
  m=json.loads((ROOT/'protocol/splits.json').read_text())['splits']
  names=[{r['image_path'] for r in m[s]} for s in ['train','validation','test']]
  self.assertEqual([len(s) for s in names],[18873,4044,4047])
  for i in range(3):
   for j in range(i):self.assertFalse(names[i]&names[j])
 def test_adaptation_subset_exact(self):
  m=json.loads((ROOT/'protocol/splits.json').read_text())['splits']['train']
  expected=[r['image_path'] for r in m if not selector_holdout(Path(r['image_path']).name)]
  actual=(ROOT/'protocol/train_adaptation.txt').read_text().splitlines()
  self.assertEqual(actual,expected);self.assertEqual(len(actual),16968)
 def test_quadrants_partition_original(self):
  self.assertEqual(pixel_box((400,200),0),(0,0,200,100))
  self.assertEqual(pixel_box((400,200),3),(200,100,400,200))
 def test_alignment_uses_actual_canvas(self):
  value=torch.ones(37,768,768)
  out=align_global_logits(value,(400,200))
  self.assertEqual(tuple(out.shape),(37,512,512))
  self.assertTrue(torch.all(out[:,128:384]==1));self.assertTrue(torch.all(out[:,:128]==0))
 def test_zero_initial_guidance_is_identity(self):
  torch.manual_seed(3407);module=GlobalGuide()
  x=torch.randn(1,192,16,16);scene=torch.randn(1,192,24,24)
  self.assertTrue(torch.equal(module(x,scene,[(400,200)],[0]),x))
if __name__=='__main__':unittest.main()
