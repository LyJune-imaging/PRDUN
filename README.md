# PRDUN: PSF-Aware Ring Convolution Deep Unfolding Network for Spatially Varying Aberration Correction

Official code repository for the paper:
**PRDUN: PSF-Aware Ring Convolution Deep Unfolding Network for Spatially Varying Aberration Correction**.

## Highlights

- We propose a lens imaging model based on spatially varying ring convolution, and build upon it a unified operator-level framework for lens aberration estimation and correction tasks.
- Building on this framework, we propose a physics-driven aberration calibration method that requires only a single calibration image and no pretraining, whose inferred representation can be seamlessly integrated into a non-blind aberration correction pipeline.
- We design a PSF-aware ring convolution deep unfolding network (PRCDUN), equipped with PSF-modulated Window Attention (PMWA) and PSF-guided Deformable Attention (PGDA), to enable local and global interactions between PSF statistical priors and image features.
- We further incorporate a curriculum learning strategy in the Seidel parameter space to improve PDUN's robustness under forward-model mismatch and enhance its generalization across diverse aberration distributions.

## Repository Structure

```text
PRDUN/
├── model/                # Proximal network construction for RPCDUN
│   ├── Net.py
│   ├── Unroll.py
│   ├── PMWA_block.py
│   └── PGDA_block.py
├── functions/            # Utility functions (PSF computation, polar transform, dataset, mosaic, etc.)
│   ├── PSF_compute.py
│   ├── polar_trans.py
│   ├── dataset.py
│   └── mosaic_func.py
├── Seidel_MLP/           # MLP-based PSF estimation module
│   └── MLP.py
├── Demsc/                # Preliminary demosaicing module
├── PSF_estimate.py       # Entry point for PSF estimation task
├── test.py               # Entry point for aberration correction evaluation
└── test_real.py          # Entry point for real-world experiments
```

| Directory / File | Description |
|:---|:---|
| `model/` | Proximal network construction for RPCDUN, including the unrolling architecture, PMWA block, and PGDA block. |
| `functions/` | Utility functions for PSF computation, polar coordinate transformation, dataset loading, and mosaic processing. |
| `Seidel_MLP/` | MLP module used for PSF estimation in the Seidel parameter space. |
| `Demsc/` | Preliminary demosaicing module for raw image pre-processing. |
| `PSF_estimate.py` | Entry script for the PSF estimation task. |
| `test.py` | Entry script for aberration correction evaluation on synthetic/benchmark data. |
| `test_real.py` | Entry script for real-world image restoration experiments. |

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

- Reconstructed images
- `metrics.txt`
- `per_sample_metrics.csv`

### 2) Real-world image restoration

Use `test_real.py` for real-image inference.

```bash
python test_real.py
```

## Results

### Aberration Correction on Synthetic Data

![Synthetic results](figs/synthetic_results.png)

### Aberration Correction on Real-World Data

![Real-world results](figs/real_world_results.png)

### PSF Estimation

![PSF estimation results](figs/psf_estimation.png)

> 📌 Please place your visualization images in the `figs/` directory:
> ```text
> figs/
> ├── synthetic_results.png
> ├── real_world_results.png
> └── psf_estimation.png
> ```

## Training

> ⚠️ **Note:** The training code will be released upon acceptance of the paper. Stay tuned!

## License

Please add your preferred open-source license (e.g., MIT, Apache-2.0) to this repository.
