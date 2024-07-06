# Ef-RAFT
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-raft-for-efficient-optical-flow/optical-flow-estimation-on-sintel-clean)](https://paperswithcode.com/sota/optical-flow-estimation-on-sintel-clean?p=rethinking-raft-for-efficient-optical-flow)


 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-raft-for-efficient-optical-flow/optical-flow-estimation-on-kitti-2015-train)](https://paperswithcode.com/sota/optical-flow-estimation-on-kitti-2015-train?p=rethinking-raft-for-efficient-optical-flow)

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-raft-for-efficient-optical-flow/optical-flow-estimation-on-sintel-final)](https://paperswithcode.com/sota/optical-flow-estimation-on-sintel-final?p=rethinking-raft-for-efficient-optical-flow)

This repository contains the source code for [Ef-RAFT: Rethinking RAFT for Efficient Optical Flow](https://arxiv.org/abs/2401.00833)<br/>

<img src="Diagram.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name efraft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```


## Required Data
To evaluate/train RAFT, you will need to download the required datasets and put them in ```datasets/``` directory. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

## How to run?
For training on 2 GPUs, run the following command. Training logs will be written to the `runs` directory, which can be visualized using tensorboard.
```Shell
./train_standard.sh
```

For running on a single RTX GPU, training can be accelerated using mixed precision, and can be done with the following command. You can expect similiar results in this setting (1 GPU).
```Shell
./train_mixed.sh
```

You can evaluate a trained model using `evaluate.py`.
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

### Quantitative Results
Comparison of the proposed method with existing
techniques on the Sintel and KITTI datasets. Metrics in green, blue, and
red denote the first, second, and third-best results, respectively.
<p align="center">
<img src="Results.png" width="700" height="500">
<p/>


### Qualitative Results
Qualitative comparison between the proposed method and RAFT. Frames with orange and blue labels are from Sintel
and KITTI datasets, respectively.
<img src="Visualization.png">


 ## Citation
If you use this repository for your research or wish to refer to our method, please use the following BibTeX entry:
```bibtex
@inproceedings{eslami2024rethinking,
  title={Rethinking RAFT for efficient optical flow},
  author={Eslami, Navid and Arefi, Farnoosh and Mansourian, Amir M and Kasaei, Shohreh},
  booktitle={2024 13th Iranian/3rd International Machine Vision and Image Processing Conference (MVIP)},
  pages={1--7},
  year={2024},
  organization={IEEE}
}
```

### Acknowledgement
This codebase is heavily borrowed from [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT). Thanks for their excellent work.

