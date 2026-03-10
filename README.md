# CompEvent

Official repository for the paper:

**CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring**

Mingchen Zhong, Xin Lu, Dong Li, Senyan Xu, Ruixuan Jiang, Xueyang Fu, Baocai Yin  

AAAI 2026

---

## Overview

Low-light video capture often requires long exposure times, which leads to severe motion blur and loss of scene details. Event cameras provide high temporal resolution and high dynamic range signals that can complement RGB frames in challenging environments.

We propose **CompEvent**, a complex-valued neural network framework that enables **full-process fusion of event streams and RGB frames** for joint low-light enhancement and motion deblurring.

Our framework introduces:

- **Complex Temporal Alignment GRU (CTA-GRU)**  
  for iterative temporal alignment and fusion of video and event streams.

- **Complex Space-Frequency Learning Module**  
  for unified feature learning in both spatial and frequency domains.

By representing RGB features and event features as the **real and imaginary components of complex tensors**, CompEvent enables deep cross-modal interaction throughout the entire restoration pipeline.

---

## Repository Status

Current release:

- [x] Paper and project description
- [x] Training code
- [x] Inference scripts
- [ ] Pretrained models
- [ ] Dataset preparation instructions

The repository will be updated with more components and detailed instructions.

---

## Paper

arXiv: https://arxiv.org/abs/2511.14469  

If you find this work helpful, please consider citing our paper.

---

## Citation

```bibtex
@article{zhong2025compevent,
  title={CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring},
  author={Zhong, Mingchen and Lu, Xin and Li, Dong and Xu, Senyan and Jiang, Ruixuan and Fu, Xueyang and Yin, Baocai},
  journal={AAAI},
  year={2026}
}
```

## Directory Structure

```text
code/
└── CompEvent/
    ├── base_code/                      # BasicSR Library Basic Training and Testing Code
    └── CSFL/                           # CompEvent Main Code (Modified based on the BasicSR framework)
        ├── basicsr/
        │   ├── train.py                # training entry
        │   ├── test.py                 # basic testing entry
        │   ├── data/
        │   │   ├── dataloader_dataset.py      # RELED dataset
        │   │   └── lol_patch_video_dataset.py # LOL-Blur dataset
        │   ├── models/
        │   │   ├── model_manager_model.py     # training pipeline manager
        │   │   ├── lol_blur_model.py
        │   │   ├── archs/
        │   │   │   ├── CompEvent_arch.py      # main CompEvent network
        │   │   │   └── ComplexBiGRU.py        # CTA-GRU related module
        │   │   └── losses/
        │   └── utils/
        ├── options/
        │   ├── train/
        │   │   ├── RELED/CompEvent-RELED.yml
        │   │   └── LOL_Blur/CompEvent-LOL_Blur.yml
        │   └── test/RELED/CompEvent-test.yml
        ├── requirements.txt
        └── setup.py
```

## Environment Preparation

It is recommended to use Conda to create a standalone environment (example):

```bash
conda create -n compevent python=3.9 -y
conda activate compevent
```

Install PyTorch (please select the official installation command according to your CUDA version):

```bash
pip install torch torchvision torchaudio
```

Install repository dependencies:

```bash
cd CompEvent/CSFL
pip install -r requirements.txt
pip install einops timm pandas tensorboard thop
python setup.py develop --no_cuda_ext

```

Note:
- If you want to compile CUDA This can be extended by removing `--no_cuda_ext`.

- Some scripts contain hard-coded absolute paths (such as `/code/...`, `/output/...`). Please modify them according to your local environment.

## Training

Enter directory:

```bash
cd CompEvent/CSFL

```
Single-card training (Example RELED dataset):

```bash
python basicsr/train.py -opt options/train/RELED/CompEvent-RELED.yml

```
Multi-card training (Example RELED dataset, 4 cards):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 \
basicsr/train.py -opt options/train/RELED/CompEvent-RELED.yml --launcher pytorch

```
## Inference

```bash
cd CompEvent/CSFL
python tools/test_compevent.py --opt options/test/RELED/CompEvent-test.yml

```
The following can be configured in `options/test/RELED/CompEvent-test.yml`:

- `inference.mode`: `full` or `tile`

- `inference.tile_size`, `inference.overlap`

- `inference.input_dir`, `inference.output_dir`

- `inference.weight`

## Data Path and Configuration Modification

Please prioritize modifying the following fields in each `.yml` configuration:

- `datasets.*.dataroot`

- `path.root`

- `path.pretrain_network_g`

- `inference.weight`, `inference.output_dir`

Example configuration file:

- `CompEvent/CSFL/options/train/RELED/CompEvent-RELED.yml`

- `CompEvent/CSFL/options/train/LOL_Blur/CompEvent-LOL_Blur.yml`

- `CompEvent/CSFL/options/test/RELED/CompEvent-test.yml`

## Data Organization (Reading Conventions in the Current Code)

Since the complete data preparation documentation has not yet been released, the following are the key reading conventions in the current code for preliminary integration:

### RELED

The training set (`Train_Video_Dataset`) is expected to contain the following in each scene directory:

- `event_voxel_parsed/00000~00003/*.npz`

- `blur_processed_parsed/00000~00003/*.png`

- `gt_processed_parsed/00000~00003/*.png`

The test set (`Test_Video_Dataset`) is expected to contain the following in each scene directory:

- `event_voxel/*.npz`

- `blur_processed/*.png`

- `gt_processed/*.png`

### LOL-Blur

Expected directory contents:

- `blur/*.png`

- `sharp/*.png`

- `event_voxel_16/*.npz`

## Results and Logs

- CSFL training will by default create `experiments/`, logs, and model storage directories under `path.root`.

- Inference results are saved by default to `inference.output_dir` in the test configuration.

## Notes

This open-source version is released initially with the goal of being "trainable and inference-capable". Pre-trained models and complete data preparation processes will be added later.
