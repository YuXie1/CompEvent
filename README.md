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
