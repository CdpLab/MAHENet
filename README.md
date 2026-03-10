<h1 align="center">Object Material Perception via Multi-Scale Feature Fusion and Multi-Supervision Optimization</h1>

 <p align="center">Dapeng Chen, Juncheng Lou, Hao Wu, Lina Wei, Chengcheng Hua, and Jia Liu</p>
  <p align="center">Nanjing University of Information Science and Technology</p>
 
</p>

---
<h2 align="center">ABSTRACT</h2>
Image material perception aims to achieve pixel-level accurate recognition of the surface materials of objects in an image and is a key step for constructing fine-grained visual understanding capability. To improve material perception accuracy and boundary quality in complex real scenes, this paper proposes a material-aware hybrid encoding network for material perception. First, a semantic–texture dual-branch hybrid encoder based on Swin-T and an atrous convolution pyramid is used to extract, in parallel, complementary global semantic context and multi-scale local texture features. To further enhance texture discriminability, we design a material-aware attention module that integrates Gabor filter priors to adaptively calibrate key feature channels. Then, a multi-scale feature fusion module integrated with a feature pyramid is adopted to effectively fuse deep semantic features and shallow detailed features. Finally, an edge-enhanced decoder, under the guidance of auxiliary supervision signals, reconstructs segmentation maps with clear boundaries and rich details. Extensive experiments on the public Dense Material Segmentation dataset show that our method achieves an average pixel accuracy of 83.2% and a mean intersection-over-union of 73.8%, outperforming existing methods and exhibiting strong generalization ability. In particular, the proposed method shows clear advantages when handling materials with directional textures such as wood and fabric.

</p>
<img src="https://github.com/wuhao0109/MAHENet/blob/main/images/Framework%20Diagram.jpg" width="75%" height="75%">
## Code
### Configure environment
```
conda env create -f environment.yaml
conda activate MAHENet
```
### Download pre-trained model
[download here](https://drive.google.com/file/d/1KbW3mG2Pz9ieXKotFQJ52YNNG8eBCNko/view?usp=sharing)
### Inference
```
python inference.py --jit_path MAHENet_model.pt --image_folder dataset/images --output_folder dataset/results
```
