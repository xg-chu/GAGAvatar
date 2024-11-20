<h1 align="center"><b><img src="./demos/gagavatar_logo.png" width="520"/></b></h1>
<h3 align="center">
    <a href='https://arxiv.org/abs/2410.07971'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
    <a href='https://xg-chu.site/project_gagavatar/'><img src='https://img.shields.io/badge/Project-Page-blue'></a> &nbsp; 
    <a href='https://www.youtube.com/watch?v=9244ZgOl4Xk'><img src='https://img.shields.io/badge/Youtube-Video-red'></a> &nbsp; 
    <a href='https://github.com/xg-chu/GAGAvatar_track/'><img src='https://img.shields.io/badge/Data-Tracker-red'></a> &nbsp; 
</h3>

<h5 align="center">
    <a href="https://xg-chu.site">Xuangeng Chu</a><sup>1</sup>&emsp;
    <a href="https://www.mi.t.u-tokyo.ac.jp/harada/">Tatsuya Harada</a><sup>1,2</sup>
    <br>
    <sup>1</sup>The University of Tokyo,
    <sup>2</sup>RIKEN AIP
</h5>

<h3 align="center">
🤩 NeurIPS 2024 🤩
</h3>

<div align="center"> 
    <div align="center"> 
        <b><img src="./demos/teaser.gif" alt="drawing" width="960"/></b>
    </div>
    <b>
        GAGAvatar reconstructs controllable 3D head avatars from single images.
    </b>
    <br>
        GAGAvatar achieves one-shot 3DGS-based head reconstruction and <b>⚡️real-time⚡️</b> reenactment.
    <br>
        🔥 More results can be found in our <a href="https://xg-chu.github.io/project_gagavatar/">Project Page</a>. 🔥
</div>

<!-- ## TO DO
We are now preparing the <b>pre-trained model and quick start materials</b> and will release it within a week. -->

## Installation
### Clone the project
```
git clone --recurse-submodules git@github.com:xg-chu/GAGAvatar.git
cd GAGAvatar
```

### Build environment
```
conda env create -f environment.yml
conda activate GAGAvatar
```
### Install the 3DGS renderer

<details>
<summary><span>What’s the difference between this version and the original 3DGS?</span></summary>

- We changed the number of channels so that 3D Gaussians carry 32-dim features.
- We changed the package name to avoid conflict with the original Gaussian splatting.

</details>

```
git clone --recurse-submodules git@github.com:xg-chu/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
rm -rf ./diff-gaussian-rasterization
```

### Prepare resources
Prepare resources with:
```
bash ./build_resources.sh
```

Also prepare resources for GAGAvatar_track using: 
```
cd core/libs/GAGAvatar_track
bash ./build_resources.sh
```

## Quick Start Guide
Driven by another **image**:
```
# This will track the images online, which is slow.
python inference.py -d ./demos/examples/2.jpg -i ./demos/examples/1.jpg
```

Driven by a tracked **video**:
```
python inference.py -d ./demos/drivers/obama -i ./demos/examples/1.jpg
``` 
Driven by a tracked **image_lmdb**:
```
python inference.py -d ./demos/drivers/vfhq_demo -i ./demos/examples/1.jpg
```

To test the inference speed, refer to the ```speed_test()``` function in ```inference.py```.

To test your own images online, refer to ```lines 52-55``` in ```inference.py```.

To test your own driving sequences (videos/images), refer to [GAGAvatar_track](https://github.com/xg-chu/GAGAvatar_track/) and demo sequences to build your own driving sequence.

## Training Guide
<details>
<summary><span>You can use the pre-trained model directly, but if you need to retrain on your data:</span></summary>

<!-- Building the dataset used for training requires img_lmdb, optim.pkl and dataset.json. -->
### Step 1: Building the image LMDB
Build ```img_lmdb``` yourself.


All the images should be cropped as inference. (Refer to line 218 in ```core/libs/GAGAvatar_track/engines/engine_core.py```) 

Dump images using ```core/libs/utils_lmdb.py```, there is also an API for building lmdb: ```dump(key_name, payload)```, payload should be tensor with (3, 512, 512), in [0, 255]. 

015252 is video id (used when sampling), 99 is frame id (0 is the first frame, other frames id can be discontinuous).
```
img_lmdb:
    '015252_99' : image payload
```

### Step 2: Track the image LMDB
Using ```track_lmdb.py``` in ```GAGAvatar_track```, you should get a ```optim.pkl```.
```
optim.pkl:
- dict_keys(['000000_0', …])
    - "000000_0": dict_keys(['bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix'])
```

### Step 3: Split the dataset
Build ```dataset.json``` yourself, it should contain the keys in ```img_lmdb``` and ```optim.pkl```.
```
dataset.json: {
    "train": ["000000_0", "000000_5", ..., '001384_654'],
    "val":   ["015209_0", ..., "015218_7"], 
    "test":  ["015203_0", ..., "015252_139"]
}
```

### Step 4: Modify the config and train
```
python train.py --config gaga --dataset vfhq
```

</details>

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{
    chu2024gagavatar,
    title={Generalizable and Animatable Gaussian Head Avatar},
    author={Xuangeng Chu and Tatsuya Harada},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=gVM2AZ5xA6}
}
```

## Acknowledgements
Some part of our work is built based on FLAME, StyleMatte, EMICA and VGGHead. 
The GAGAvatar Logo is designed by Caihong Ning.
We also thank the following projects for sharing their great work.
- **GPAvatar**: https://github.com/xg-chu/GPAvatar
- **FLAME**: https://flame.is.tue.mpg.de
- **StyleMatte**: https://github.com/chroneus/stylematte
- **EMICA**: https://github.com/radekd91/inferno
- **VGGHead**: https://github.com/KupynOrest/head_detector
