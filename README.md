<h1 align="center">MAHENet: Object Material Perception via Multi-Scale Feature Fusion and Multi-Supervision Optimization</h1>

<p align="center">
Dapeng Chen, Juncheng Lou, Hao Wu, Lina Wei, Chengcheng Hua, and Jia Liu∗<br>
Nanjing University of Information Science and Technology
</p>

---
<h2 align="center">ABSTRACT</h2>
Image material perception aims to achieve pixel-level precise recognition of surface materials of various objects in an image, it is a crucial task for fine-grained visual understanding. To improve material perception accuracy and boundary quality in complex real-world scenes, this study proposes a novel material perception network, termed MAHENet. Specifically, a semantic-texture hybrid encoder based on Swin-T and a dilated convolution pyramid is first employed to extract complementary global semantic context and multi-scale local texture features in parallel. To further enhance texture discriminability, a material-aware attention module incorporating Gabor filter priors is designed to adaptively calibrate key feature channels. Subsequently, a multi-scale fusion module integrated with a feature pyramid is introduced to effectively aggregate deep semantic representations and shallow detailed features. Finally, an edge-enhanced decoder, guided by auxiliary supervision signals, reconstructs segmentation maps with sharp boundaries and fine details. Extensive experiments on the public DMS dataset demonstrate that MAHENet achieves an average pixel accuracy of 83.2\% and a mean intersection-over-union ratio of 73.8\%, outperforming existing methods and demonstrating strong generalization capability. Notably, the proposed method shows clear advantages in handling materials with directional textures, such as wood and fabric.

<p align="center">
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
