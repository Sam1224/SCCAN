# Self-Calibrated Cross Attention Network for Few-Shot Segmentation

This repository contains the code for our ICCV 2023 [paper](https://arxiv.org/abs/2308.09294) "*Self-Calibrated Cross Attention Network for Few-Shot Segmentation*".

> **Abstract**: *The key to the success of few-shot segmentation (FSS) lies in how to effectively utilize support samples. Most solutions compress support foreground (FG) features into prototypes, but lose some spatial details. Instead, others use cross attention to fuse query features with uncompressed support FG. Query FG could be fused with support FG, however, query background (BG) cannot find matched BG features in support FG, yet inevitably integrates dissimilar features. Besides, as both query FG and BG are combined with support FG, they get entangled, thereby leading to ineffective segmentation. To cope with these issues, we design a self-calibrated cross attention (SCCA) block. For efficient patch-based attention, query and support features are firstly split into patches. Then, we design a patch alignment module to align each query patch with its most similar support patch for better cross attention. Specifically, SCCA takes a query patch as Q, and groups the patches from the same query image and the aligned patches from the support image as K&V. In this way, the query BG features are fused with matched BG features (from query patches), and thus the aforementioned issues will be mitigated. Moreover, when calculating SCCA, we design a scaled-cosine mechanism to better utilize the support features for similarity calculation. Extensive experiments conducted on PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> demonstrate the superiority of our model, e.g., the mIoU score under 5-shot setting on COCO-20<sup>i</sup> is 5.6%+ better than previous state-of-the-arts.* 

<p align="middle">
  <img src="figure/overview.png">
</p>

## Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.0
```
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets

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

## Models

- Download the pre-trained backbones from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EbntykE8vXFMotK31vYk8iABRhFwvgVDt93koaIA63YgJQ?e=rE0swx) and put them into the `initmodel` directory.
- Download [exp.zip](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EatmRPwDkCFFtpB3S4ejV4cBvuAMmhqvoCDvg2r446WAFw) and compress it to obtain all pretrained models for PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>.

## Usage

- Change configuration via the `.yaml` files in `config`, then run the following commands for training and testing.

- **Meta-training**
  - *1/5-shot* for PASCAL-5<sup>i</sup>
  ```
  CUDA_VISIBLE_DEVICES=0 python train_sccan.py --config=config/pascal/pascal_split{0,1,2,3}_resnet{50,101}{_5s}.yaml
  ```
  - *1/5-shot* for COCO-20<sup>i</sup>
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train_sccan.py --config=config/coco/coco_split{0,1,2,3}_resnet{50,101}{_5s}.yaml
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

## Performance

Performance comparison with the state-of-the-arts. 

1. ### PASCAL-5<sup>i</sup>

<p align="middle">
  <img src="figure/pascal_sota.png">
</p>

2. ### COCO-20<sup>i</sup>

<p align="middle">
  <img src="figure/coco_sota.png">
</p>

## Visualization

<p align="middle">
    <img src="figure/visualization.png">
</p>

## References

This repo is mainly built based on [BAM](https://github.com/chunbolang/BAM). Thanks for their great work!

## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@InProceedings{Xu_2023_ICCV,
    author    = {Xu, Qianxiong and Zhao, Wenting and Lin, Guosheng and Long, Cheng},
    title     = {Self-Calibrated Cross Attention Network for Few-Shot Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {655-665}
}
```
