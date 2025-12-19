# AAE5303 - 3D Gaussian Splatting Demo with OpenSplat

<div align="center">

![Course](https://img.shields.io/badge/AAE5303-Robust_Control_Technology-darkblue?style=for-the-badge)
![3DGS](https://img.shields.io/badge/3D_Gaussian-Splatting-blue?style=for-the-badge)
![OpenSplat](https://img.shields.io/badge/Framework-OpenSplat-green?style=for-the-badge)

**Robust Control Technology in Low-Altitude Aerial Vehicle**

*Hong Kong Polytechnic University - Master Program*

</div>

---

## 📖 Overview

This repository provides a **demonstration** of 3D Gaussian Splatting (3DGS) for novel view synthesis using UAV aerial imagery. It serves as a reference implementation for the AAE5303 course assignment, showcasing the complete pipeline from data preparation to result visualization.

### What You Will Learn

- ✅ Setting up OpenSplat build environment
- ✅ Preparing COLMAP-formatted input data
- ✅ Training 3D Gaussian Splatting models
- ✅ Analyzing training dynamics
- ✅ Generating quality visualizations

---

## 📁 Repository Structure

```
AAE5303_opensplat_demo/
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── figures/                  # Training visualization results
│   ├── training_loss_curve.png
│   ├── loss_distribution.png
│   ├── convergence_analysis.png
│   └── summary_dashboard.png
├── scripts/                  # Analysis scripts
│   └── analyze_training.py
├── output/                   # Generated outputs
│   └── training_report.json
├── docs/                     # Documentation
│   └── training_log.txt
└── leaderboard/              # Leaderboard submission guide
    ├── README.md
    ├── LEADERBOARD_SUBMISSION_GUIDE.md
    └── submission_template.json
```

---

## 🚀 Quick Start

### Prerequisites

- Linux environment (Ubuntu 20.04+ recommended)
- CMake 3.16+
- Python 3.8+
- libtorch 2.1.2+
- OpenCV

### 1. Build OpenSplat

```bash
# Clone OpenSplat repository
git clone https://github.com/pierotofy/OpenSplat
cd OpenSplat

# Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ ..
make -j$(nproc)
```

### 2. Prepare Dataset

Your dataset should be in COLMAP format:

```
your_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### 3. Run Training

```bash
./opensplat /path/to/your_dataset \
    -n 30000 \
    -o output.ply \
    --sh-degree 3 \
    --ssim-weight 0.2
```

### 4. Analyze Results

```bash
pip install -r requirements.txt
python scripts/analyze_training.py --log your_training.log
```

---

## 📊 Demo Results

This demo was trained on the **HKisland** UAV aerial dataset with the following configuration:

| Configuration | Value |
|--------------|-------|
| Training Iterations | 300 |
| Number of Images | 534 |
| SSIM Weight | 0.2 |
| SH Degree | 3 |

### Training Metrics

| Metric | Value |
|--------|-------|
| Initial Loss | 0.2164 |
| Final Loss | 0.2079 |
| Minimum Loss | 0.1511 |
| Output Gaussians | 1,441,245 |
| PLY File Size | 340.8 MB |

### Visualizations

<table>
<tr>
<td align="center"><b>Training Loss Curve</b></td>
<td align="center"><b>Loss Distribution</b></td>
</tr>
<tr>
<td><img src="figures/training_loss_curve.png" width="400"/></td>
<td><img src="figures/loss_distribution.png" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Convergence Analysis</b></td>
<td align="center"><b>Summary Dashboard</b></td>
</tr>
<tr>
<td><img src="figures/convergence_analysis.png" width="400"/></td>
<td><img src="figures/summary_dashboard.png" width="400"/></td>
</tr>
</table>

---

## 📚 Background: 3D Gaussian Splatting

### Core Concept

3D Gaussian Splatting represents scenes using millions of 3D Gaussian primitives, each characterized by:

| Property | Description |
|----------|-------------|
| **Position (μ)** | 3D mean position in world coordinates |
| **Covariance (Σ)** | 3×3 matrix defining shape and orientation |
| **Opacity (α)** | Transparency value for blending |
| **Spherical Harmonics** | View-dependent color representation |

### Rendering Equation

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

### Loss Function

$$\mathcal{L} = (1 - \lambda) \cdot \mathcal{L}_1 + \lambda \cdot (1 - SSIM)$$

where $\lambda = 0.2$ is the SSIM weight.

---

## 🎯 Assignment Guidelines

### For Students

1. **Setup**: Follow the Quick Start guide to set up your environment
2. **Dataset**: Use the provided dataset or prepare your own UAV imagery
3. **Training**: Train your model with appropriate hyperparameters
4. **Analysis**: Generate visualizations using the provided scripts
5. **Submission**: Follow the [Leaderboard Submission Guide](leaderboard/LEADERBOARD_SUBMISSION_GUIDE.md)

### Recommended Hyperparameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `num-iters` | 30,000 | Training iterations |
| `sh-degree` | 3 | Spherical harmonics degree |
| `ssim-weight` | 0.2 | SSIM loss weight |
| `refine-every` | 100 | Densification interval |

### Tips for Better Results

- 🔧 Use GPU acceleration (CUDA) for faster training
- 📈 Train for at least 30,000 iterations for quality results
- 🎯 Ensure good camera coverage in your input images
- 📊 Monitor loss curves for convergence

---

## 🏆 Leaderboard

The course includes a leaderboard to evaluate student implementations. See the [`leaderboard/`](leaderboard/) folder for:

- **Evaluation Metrics**: PSNR, SSIM, LPIPS
- **Submission Format**: JSON template
- **Submission Guide**: Step-by-step instructions

---

## 📖 References

1. Kerbl, B., et al. (2023). **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. *ACM SIGGRAPH*.

2. Schönberger, J. L., & Frahm, J. M. (2016). **Structure-from-Motion Revisited**. *CVPR*.

3. [OpenSplat GitHub Repository](https://github.com/pierotofy/OpenSplat)

4. Wang, Z., et al. (2004). **Image Quality Assessment: From Error Visibility to Structural Similarity**. *IEEE TIP*.

---

## 📧 Contact

For questions about this assignment, please contact the course instructor.

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

</div>
