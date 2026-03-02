# PRDUN: PSF-Aware Ring Convolution Deep Unfolding Network for Spatially Varying Aberration Correction

Official code repository for the paper:
**PRDUN: PSF-Aware Ring Convolution Deep Unfolding Network for Spatially Varying Aberration Correction**.

## Highlights

- We propose a lens imaging model based on spatially varying ring convolution, and build upon it a unified operator-level framework for lens aberration estimation and correction tasks.
- Building on this framework, we propose a physics-driven aberration calibration method that requires only a single calibration image and no pretraining, whose inferred representation can be seamlessly integrated into a non-blind aberration correction pipeline.
- We design a PSF-aware ring convolution deep unfolding network (PRCDUN), equipped with PSF-modulated Window Attention (PMWA) and PSF-guided Deformable Attention (PGDA), to enable local and global interactions between PSF statistical priors and image features.
- We further incorporate a curriculum learning strategy in the Seidel parameter space to improve PDUN’s robustness under forward-model mismatch and enhance its generalization across diverse aberration distributions.

## Repository Structure

```text
PRDUN/
├── model/
│   ├── Net.py
│   ├── Unroll.py
│   ├── PMWA_block.py
│   └── PGDA_block.py
├── functions/
│   ├── PSF_compute.py
│   ├── polar_trans.py
│   ├── dataset.py
│   └── mosaic_func.py
├── Seldel_MLP/
│   └── MLP.py
├── Demsc/
├── PSF_estimate.py
├── test.py
└── test_real.py
```

## Environment

Suggested environment: Python 3.10+ and CUDA-enabled PyTorch.

Install core dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pillow scipy matplotlib lpips pytorch-msssim
```

## Inference

### 1) Synthetic / benchmark evaluation

Use `test.py` for test-set evaluation and metric export.

```bash
python test.py
```

Outputs include:

- reconstructed images
- `metrics.txt`
- `per_sample_metrics.csv`

### 2) Real-world image restoration

Use `test_real.py` for real-image inference.

```bash
python test_real.py
```



## License

Please add your preferred open-source license (e.g., MIT, Apache-2.0) to this repository.
