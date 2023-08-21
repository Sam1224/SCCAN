# Self-Calibrated Cross Attention Network for Few-Shot Segmentation

This repository contains the code for our ICCV 2023 [paper](https://arxiv.org/abs/2308.09294) "*Self-Calibrated Cross Attention Network for Few-Shot Segmentation*".

> **Abstract**: *The key to the success of few-shot segmentation (FSS) lies in how to effectively utilize support samples. Most solutions compress support foreground (FG) features into prototypes, but lose some spatial details. Instead, others use cross attention to fuse query features with uncompressed support FG. Query FG could be fused with support FG, however, query background (BG) cannot find matched BG features in support FG, yet inevitably integrates dissimilar features. Besides, as both query FG and BG are combined with support FG, they get entangled, thereby leading to ineffective segmentation. To cope with these issues, we design a self-calibrated cross attention (SCCA) block. For efficient patch-based attention, query and support features are firstly split into patches. Then, we design a patch alignment module to align each query patch with its most similar support patch for better cross attention. Specifically, SCCA takes a query patch as Q, and groups the patches from the same query image and the aligned patches from the support image as K&V. In this way, the query BG features are fused with matched BG features (from query patches), and thus the aforementioned issues will be mitigated. Moreover, when calculating SCCA, we design a scaled-cosine mechanism to better utilize the support features for similarity calculation. Extensive experiments conducted on PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> demonstrate the superiority of our model, e.g., the mIoU score under 5-shot setting on COCO-20<sup>i</sup> is 5.6%+ better than previous state-of-the-arts.* 

<p align="middle">
  <img src="figure/overview.png">
</p>

### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.0
```
> conda env create -f env_{ubuntu,windows}.yaml
```

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

The directory structure is:

    ../
    ├── SCCAN/
    └── data/
        ├── VOCdevkit2012/
        │   └── VOC2012/
        │       ├── JPEGImages/
        │       ├── ...
        │       └── SegmentationClassAug/
        └── MSCOCO2014/           
            ├── annotations/
            │   ├── train2014/ 
            │   └── val2014/
            ├── train2014/
            └── val2014/

### Models

- Download the pre-trained backbones from [here](https://drive.google.com/file/d/1M0pUB1ghGI4GgwmMbLaGRFyHl4WOB0iE/view?usp=sharing) and put them into the `initmodel` directory.
- Download [exp.zip](https://drive.google.com/file/d/1qMn7s0GL6ljVVRlHnRxHWd4pou2hVvJ9/view?usp=sharing) and compress it to obtain all pretrained models for PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>.

### Usage

- Change configuration via the `.yaml` files in `config`, then run the following commands for training and testing.

- **Meta-training**
  - *1-shot*
  ```
  CUDA_VISIBLE_DEVICES=0 python train_sccan.py --config=config/{pascal,coco}/{pascal,coco}_split{0,1,2,3}_resnet{50,101}.yaml
  ```
  - *5-shot*
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train_sccan.py --config=config/{pascal,coco}/{pascal,coco}_split{0,1,2,3}_resnet{50,101}_5s.yaml
  ```

- **Meta-testing**
  - *1-shot*
  ```
  CUDA_VISIBLE_DEVICES=0 python test_sccan.py --config=config/{pascal,coco}/{pascal,coco}_split{0,1,2,3}_resnet{50,101}.yaml
  ```
  - *5-shot*
  ```
  CUDA_VISIBLE_DEVICES=0 python test_sccan.py --config=config/{pascal,coco}/{pascal,coco}_split{0,1,2,3}_resnet{50,101}_5s.yaml
  ```

### Performance

Performance comparison with the state-of-the-arts. 

1. ##### PASCAL-5<sup>i</sup>

<p align="middle">
  <img src="figure/pascal_sota.png">
</p>

2. ##### COCO-20<sup>i</sup>

<p align="middle">
  <img src="figure/coco_sota.png">
</p>

### Visualization

<p align="middle">
    <img src="figure/visualization.png">
</p>

### References

This repo is mainly built based on [BAM](https://github.com/chunbolang/BAM). Thanks for their great work!

## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@article{xu2023selfcalibrated,
  title={Self-Calibrated Cross Attention Network for Few-Shot Segmentation},
  author={Xu, Qianxiong and Zhao, Wenting and Lin, Guosheng and Long, Cheng},
  journal={arXiv preprint arXiv:2308.09294},
  year={2023}
}
```