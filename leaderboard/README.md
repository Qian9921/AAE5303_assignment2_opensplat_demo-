# 🏆 AAE5303 3D Gaussian Splatting Assignment Leaderboard

<div align="center">

![Leaderboard](https://img.shields.io/badge/Leaderboard-System-gold?style=for-the-badge)
![Evaluation](https://img.shields.io/badge/Evaluation-Framework-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0-green?style=for-the-badge)

**Comprehensive Evaluation Framework for 3D Gaussian Splatting Assignments**

*A Fair, Transparent, and Reproducible Assessment System*

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Evaluation Philosophy](#-evaluation-philosophy)
3. [Metrics Specification](#-metrics-specification)
4. [Scoring System](#-scoring-system)
5. [Submission Guidelines](#-submission-guidelines)
6. [Evaluation Pipeline](#-evaluation-pipeline)
7. [Leaderboard Categories](#-leaderboard-categories)
8. [Technical Implementation](#-technical-implementation)
9. [FAQ](#-frequently-asked-questions)
10. [Changelog](#-changelog)

---

## 🎯 Overview

### Purpose

This leaderboard system is designed to provide a **fair, comprehensive, and educational** assessment framework for the AAE5303 3D Gaussian Splatting assignment. The system evaluates student submissions across multiple dimensions including rendering quality, computational efficiency, and innovation.

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Fairness** | All submissions evaluated using identical metrics and procedures |
| **Transparency** | Clear scoring criteria with detailed feedback |
| **Reproducibility** | Automated evaluation ensures consistent results |
| **Educational Value** | Metrics chosen to reinforce learning objectives |
| **Incentivization** | Bonus points for innovation and improvement |

### Key Features

- 🔬 **Multi-dimensional evaluation** across quality, efficiency, and innovation
- 📊 **Automated scoring pipeline** with reproducible results
- 📈 **Real-time leaderboard updates** after each submission
- 🎓 **Detailed feedback reports** for learning improvement
- 🏅 **Multiple award categories** recognizing different achievements

---

## 🧠 Evaluation Philosophy

### Why Multiple Metrics?

3D Gaussian Splatting involves inherent trade-offs between different objectives:

```
                    ┌─────────────────┐
                    │  QUALITY        │
                    │  (PSNR, SSIM)   │
                    └────────┬────────┘
                             │
              Trade-off      │      Trade-off
                   ┌─────────┴─────────┐
                   │                   │
          ┌────────▼────────┐ ┌────────▼────────┐
          │  EFFICIENCY     │ │  COMPACTNESS    │
          │  (Speed, Memory)│ │  (Model Size)   │
          └─────────────────┘ └─────────────────┘
```

A single metric cannot capture all aspects of a good 3DGS implementation. Therefore, we employ a **balanced multi-metric evaluation** that rewards:

1. **High rendering quality** (visual fidelity)
2. **Computational efficiency** (practical usability)
3. **Model compactness** (deployment feasibility)
4. **Innovation** (creative problem-solving)

### Learning Objectives Alignment

| Learning Objective | Evaluated By |
|-------------------|--------------|
| Understanding 3DGS fundamentals | Baseline completion |
| Optimization techniques | Quality metrics improvement |
| Computational efficiency | Training time, memory usage |
| Research methodology | Documentation quality |
| Innovation capability | Bonus metrics |

---

## 📏 Metrics Specification

### Category 1: Rendering Quality Metrics (50 points)

#### 1.1 Peak Signal-to-Noise Ratio (PSNR) - 20 points

**Definition**: PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise.

**Mathematical Formula**:

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right) = 20 \cdot \log_{10}\left(\frac{MAX_I}{\sqrt{MSE}}\right)$$

where:
- $MAX_I$ is the maximum possible pixel value (255 for 8-bit images)
- $MSE = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2$

**Interpretation**:
| PSNR Range | Quality Level | Typical Score |
|------------|---------------|---------------|
| > 35 dB | Excellent | 18-20 points |
| 30-35 dB | Good | 14-18 points |
| 25-30 dB | Acceptable | 10-14 points |
| 20-25 dB | Poor | 5-10 points |
| < 20 dB | Unacceptable | 0-5 points |

**Scoring Formula**:
```python
def psnr_score(psnr_value, max_points=20):
    """
    Calculate PSNR score using a sigmoid-based mapping.
    
    Args:
        psnr_value: Measured PSNR in dB
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Baseline: 20 dB = 0 points, 35 dB = max points
    normalized = (psnr_value - 20) / 15  # Normalize to [0, 1] range
    normalized = max(0, min(1, normalized))  # Clamp
    
    # Apply sigmoid for smooth scoring
    score = max_points * (1 / (1 + math.exp(-6 * (normalized - 0.5))))
    return round(score, 2)
```

**Implementation Notes**:
- PSNR is computed per-image and averaged across all test views
- Images are compared at the same resolution
- Color space: RGB (linear or sRGB as specified)
- Border pixels may be excluded to avoid edge artifacts

#### 1.2 Structural Similarity Index (SSIM) - 20 points

**Definition**: SSIM measures the perceived quality of images based on structural information, luminance, and contrast.

**Mathematical Formula**:

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- $\mu_x$, $\mu_y$ are the local means
- $\sigma_x$, $\sigma_y$ are the local standard deviations
- $\sigma_{xy}$ is the cross-covariance
- $C_1 = (K_1 L)^2$, $C_2 = (K_2 L)^2$ are stability constants
- $L$ is the dynamic range, $K_1 = 0.01$, $K_2 = 0.03$

**Interpretation**:
| SSIM Range | Quality Level | Typical Score |
|------------|---------------|---------------|
| > 0.95 | Excellent | 18-20 points |
| 0.90-0.95 | Good | 14-18 points |
| 0.85-0.90 | Acceptable | 10-14 points |
| 0.80-0.85 | Poor | 5-10 points |
| < 0.80 | Unacceptable | 0-5 points |

**Scoring Formula**:
```python
def ssim_score(ssim_value, max_points=20):
    """
    Calculate SSIM score using linear mapping with thresholds.
    
    Args:
        ssim_value: Measured SSIM (0 to 1)
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Baseline: 0.75 = 0 points, 0.95 = max points
    normalized = (ssim_value - 0.75) / 0.20
    normalized = max(0, min(1, normalized))
    
    score = max_points * normalized
    return round(score, 2)
```

**Implementation Notes**:
- Window size: 11×11 pixels (default)
- Gaussian weighting with σ = 1.5
- Multi-scale SSIM (MS-SSIM) may be used for bonus evaluation

#### 1.3 Learned Perceptual Image Patch Similarity (LPIPS) - 10 points

**Definition**: LPIPS uses deep neural network features to measure perceptual similarity, correlating better with human perception than traditional metrics.

**Mathematical Formula**:

$$LPIPS(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} ||w_l \odot (\hat{y}_{hw}^l - \hat{x}_{hw}^l)||_2^2$$

where:
- $\hat{y}^l$, $\hat{x}^l$ are normalized feature maps from layer $l$
- $w_l$ are learned weights
- Backbone: VGG-16 or AlexNet

**Interpretation**:
| LPIPS Range | Quality Level | Typical Score |
|-------------|---------------|---------------|
| < 0.05 | Excellent | 9-10 points |
| 0.05-0.10 | Good | 7-9 points |
| 0.10-0.15 | Acceptable | 5-7 points |
| 0.15-0.20 | Poor | 2-5 points |
| > 0.20 | Unacceptable | 0-2 points |

**Scoring Formula**:
```python
def lpips_score(lpips_value, max_points=10):
    """
    Calculate LPIPS score (lower is better).
    
    Args:
        lpips_value: Measured LPIPS (0 to 1, lower is better)
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Baseline: 0.20 = 0 points, 0.05 = max points
    normalized = (0.20 - lpips_value) / 0.15
    normalized = max(0, min(1, normalized))
    
    score = max_points * normalized
    return round(score, 2)
```

---

### Category 2: Efficiency Metrics (25 points)

#### 2.1 Model Compactness - Gaussian Count (10 points)

**Definition**: The total number of 3D Gaussian primitives in the final model. Fewer Gaussians with equivalent quality indicates better optimization.

**Rationale**: 
- Fewer Gaussians = faster rendering
- Fewer Gaussians = smaller model files
- Efficient representations demonstrate understanding of optimization

**Scoring Formula**:
```python
def gaussian_count_score(num_gaussians, baseline=1500000, max_points=10):
    """
    Score based on Gaussian count efficiency.
    
    Lower count relative to baseline = higher score.
    Baseline is the initial SfM point count.
    
    Args:
        num_gaussians: Final Gaussian count
        baseline: Initial point count from SfM
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Ratio of final to initial
    ratio = num_gaussians / baseline
    
    if ratio <= 0.5:
        # Significant reduction: bonus territory
        score = max_points
    elif ratio <= 1.0:
        # Maintained or reduced: good
        score = max_points * (1.0 - (ratio - 0.5))
    elif ratio <= 2.0:
        # Some growth: acceptable
        score = max_points * 0.5 * (2.0 - ratio)
    else:
        # Excessive growth: penalty
        score = 0
    
    return round(max(0, score), 2)
```

**Benchmarks**:
| Gaussian Count (relative to SfM) | Assessment | Score Range |
|----------------------------------|------------|-------------|
| < 50% of initial | Excellent | 9-10 |
| 50-100% of initial | Good | 6-9 |
| 100-150% of initial | Acceptable | 3-6 |
| 150-200% of initial | Poor | 1-3 |
| > 200% of initial | Excessive | 0-1 |

#### 2.2 Model File Size (8 points)

**Definition**: The size of the output PLY/SPLAT file in megabytes.

**Scoring Formula**:
```python
def file_size_score(size_mb, max_points=8):
    """
    Score based on output file size.
    
    Args:
        size_mb: File size in megabytes
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Baseline: 500MB = 0 points, 50MB = max points
    if size_mb <= 50:
        score = max_points
    elif size_mb <= 500:
        score = max_points * (500 - size_mb) / 450
    else:
        score = 0
    
    return round(max(0, score), 2)
```

#### 2.3 Training Efficiency (7 points)

**Definition**: Training time normalized by the number of iterations and hardware capability.

**Scoring Formula**:
```python
def training_efficiency_score(time_per_iter_ms, device_type, max_points=7):
    """
    Score based on training efficiency.
    
    Args:
        time_per_iter_ms: Average milliseconds per iteration
        device_type: 'cpu', 'cuda', or 'mps'
        max_points: Maximum achievable points
    
    Returns:
        Score between 0 and max_points
    """
    # Device-specific baselines (ms per iteration)
    baselines = {
        'cuda': {'excellent': 50, 'poor': 500},
        'mps': {'excellent': 100, 'poor': 1000},
        'cpu': {'excellent': 5000, 'poor': 50000}
    }
    
    baseline = baselines.get(device_type, baselines['cpu'])
    
    if time_per_iter_ms <= baseline['excellent']:
        score = max_points
    elif time_per_iter_ms >= baseline['poor']:
        score = 0
    else:
        range_size = baseline['poor'] - baseline['excellent']
        normalized = (baseline['poor'] - time_per_iter_ms) / range_size
        score = max_points * normalized
    
    return round(max(0, score), 2)
```

---

### Category 3: Documentation & Methodology (15 points)

#### 3.1 Report Quality (8 points)

**Evaluation Rubric**:

| Criterion | Excellent (2 pts) | Good (1.5 pts) | Acceptable (1 pt) | Poor (0.5 pts) |
|-----------|-------------------|----------------|-------------------|----------------|
| **Structure** | Clear, logical organization with all sections | Minor organizational issues | Some sections missing | Disorganized |
| **Technical Depth** | Deep understanding demonstrated | Good understanding | Basic understanding | Superficial |
| **Visualization** | Professional, informative figures | Good figures | Basic figures | Missing/poor figures |
| **Writing Quality** | Clear, concise, professional | Minor issues | Some clarity issues | Difficult to understand |

#### 3.2 Reproducibility (4 points)

**Evaluation Criteria**:
- Clear instructions for environment setup (1 pt)
- Complete command documentation (1 pt)
- Configuration files provided (1 pt)
- Random seed documentation (1 pt)

#### 3.3 Code Quality (3 points)

**Evaluation Criteria**:
- Code comments and documentation (1 pt)
- Clean, readable code structure (1 pt)
- Error handling and edge cases (1 pt)

---

### Category 4: Bonus Points (Up to 15 points)

#### 4.1 Innovation Bonus (0-10 points)

**Eligible Innovations**:

| Innovation Type | Description | Points |
|-----------------|-------------|--------|
| **Novel Loss Function** | Implementing additional loss terms (depth, normal, etc.) | 2-4 |
| **Improved Densification** | Better adaptive density control strategies | 2-3 |
| **Compression Techniques** | Post-training model compression | 2-3 |
| **Anti-aliasing** | Mip-splatting or similar techniques | 2-3 |
| **Speed Optimization** | Significant training/rendering speedup | 2-3 |
| **Novel Applications** | Creative applications of 3DGS | 2-4 |

#### 4.2 Quality Improvement Bonus (0-3 points)

Bonus points for exceeding baseline metrics:
- PSNR > 30 dB: +1 point
- SSIM > 0.92: +1 point
- LPIPS < 0.08: +1 point

#### 4.3 Efficiency Improvement Bonus (0-2 points)

- Gaussian count < 50% of initial: +1 point
- File size < 100 MB with good quality: +1 point

---

## 📊 Scoring System

### Total Score Calculation

```
Total Score = Quality (50) + Efficiency (25) + Documentation (15) + Bonus (up to 15)

Maximum Possible: 105 points (90 base + 15 bonus)
```

### Grade Mapping

| Score Range | Grade | Description |
|-------------|-------|-------------|
| 95-105 | A+ | Exceptional - publication quality |
| 85-94 | A | Excellent - exceeds all expectations |
| 75-84 | B+ | Very Good - above average |
| 65-74 | B | Good - meets expectations |
| 55-64 | C+ | Satisfactory - basic requirements met |
| 45-54 | C | Passing - minimum requirements |
| < 45 | F | Failing - requirements not met |

### Weighted Score Formula

```python
def calculate_total_score(metrics):
    """
    Calculate the total score from individual metrics.
    
    Args:
        metrics: Dictionary containing all metric values
    
    Returns:
        Dictionary with component scores and total
    """
    scores = {}
    
    # Quality metrics (50 points)
    scores['psnr'] = psnr_score(metrics['psnr'])
    scores['ssim'] = ssim_score(metrics['ssim'])
    scores['lpips'] = lpips_score(metrics['lpips'])
    scores['quality_total'] = scores['psnr'] + scores['ssim'] + scores['lpips']
    
    # Efficiency metrics (25 points)
    scores['gaussian_count'] = gaussian_count_score(
        metrics['num_gaussians'], 
        metrics['initial_points']
    )
    scores['file_size'] = file_size_score(metrics['file_size_mb'])
    scores['training_efficiency'] = training_efficiency_score(
        metrics['time_per_iter_ms'],
        metrics['device_type']
    )
    scores['efficiency_total'] = (
        scores['gaussian_count'] + 
        scores['file_size'] + 
        scores['training_efficiency']
    )
    
    # Documentation (15 points) - manual evaluation
    scores['documentation'] = metrics.get('documentation_score', 0)
    
    # Bonus points
    scores['bonus'] = calculate_bonus(metrics)
    
    # Total
    scores['total'] = (
        scores['quality_total'] + 
        scores['efficiency_total'] + 
        scores['documentation'] + 
        scores['bonus']
    )
    
    return scores
```

---

## 📝 Submission Guidelines

### Required Files

```
submission/
├── output/
│   ├── {student_id}_hkisland.ply          # Required: Trained model
│   └── cameras.json                        # Required: Camera poses
├── logs/
│   └── training.log                        # Required: Full training log
├── report/
│   └── report.pdf                          # Required: Assignment report
├── code/
│   └── modifications/                      # Optional: Code modifications
│       ├── README.md                       # Description of changes
│       └── *.cpp / *.hpp                   # Modified source files
├── rendered/
│   └── test_views/                         # Optional: Rendered test views
│       ├── view_001.png
│       ├── view_002.png
│       └── ...
└── metadata.json                           # Required: Submission metadata
```

### Metadata File Format

```json
{
    "student_id": "12345678",
    "student_name": "John Doe",
    "submission_date": "2024-12-17T10:00:00Z",
    "training_config": {
        "num_iterations": 30000,
        "device": "cuda",
        "sh_degree": 3,
        "ssim_weight": 0.2,
        "learning_rates": {
            "means": 0.00016,
            "scales": 0.005,
            "quats": 0.001,
            "features_dc": 0.0025,
            "features_rest": 0.000125,
            "opacities": 0.05
        }
    },
    "results_summary": {
        "final_loss": 0.15,
        "num_gaussians": 1200000,
        "training_time_hours": 2.5
    },
    "innovations": [
        "Implemented depth regularization loss",
        "Added progressive training schedule"
    ],
    "notes": "Any additional notes about the submission"
}
```

### Submission Checklist

- [ ] PLY file is valid and loadable
- [ ] Training log contains all iterations
- [ ] Report follows the required structure
- [ ] Metadata file is complete and accurate
- [ ] All file names follow naming convention
- [ ] Code modifications (if any) are documented
- [ ] Submission size is under 500MB (excluding large PLY files)

### Naming Convention

```
{student_id}_{dataset}_{description}.{extension}

Examples:
- 12345678_hkisland_final.ply
- 12345678_hkisland_training.log
- 12345678_report.pdf
```

---

## 🔄 Evaluation Pipeline

### Automated Evaluation Process

```
┌─────────────────┐
│  Submission     │
│  Upload         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │
│  Check          │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Valid?  │
    └────┬────┘
    No   │   Yes
    │    │
    ▼    ▼
┌───────┐ ┌─────────────────┐
│Reject │ │  Load PLY       │
│       │ │  Model          │
└───────┘ └────────┬────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│  Render Test    │ │  Extract Model  │
│  Views          │ │  Statistics     │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│  Compute        │ │  Compute        │
│  Quality Metrics│ │  Efficiency     │
└────────┬────────┘ └────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Calculate      │
         │  Final Score    │
         └────────┬────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Update         │
         │  Leaderboard    │
         └────────┬────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Generate       │
         │  Feedback Report│
         └─────────────────┘
```

### Test View Rendering

For quality evaluation, models are rendered from a set of held-out test viewpoints:

1. **Test View Selection**: 10% of images are withheld for testing
2. **Rendering Settings**: Full resolution, no downscaling
3. **Background**: White background (RGB: 255, 255, 255)
4. **Output Format**: PNG, 8-bit per channel

### Quality Computation

```python
def evaluate_quality(rendered_dir, ground_truth_dir):
    """
    Compute all quality metrics between rendered and ground truth images.
    
    Args:
        rendered_dir: Directory containing rendered test views
        ground_truth_dir: Directory containing ground truth images
    
    Returns:
        Dictionary of quality metrics
    """
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    lpips_model = lpips.LPIPS(net='vgg')
    
    rendered_files = sorted(Path(rendered_dir).glob('*.png'))
    
    for rendered_path in rendered_files:
        gt_path = Path(ground_truth_dir) / rendered_path.name
        
        if not gt_path.exists():
            continue
        
        # Load images
        rendered = cv2.imread(str(rendered_path))
        gt = cv2.imread(str(gt_path))
        
        # Ensure same size
        if rendered.shape != gt.shape:
            rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]))
        
        # PSNR
        mse = np.mean((rendered.astype(float) - gt.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
        else:
            psnr = float('inf')
        psnr_values.append(psnr)
        
        # SSIM
        ssim_val = ssim(gt, rendered, channel_axis=2, data_range=255)
        ssim_values.append(ssim_val)
        
        # LPIPS
        rendered_tensor = lpips.im2tensor(rendered)
        gt_tensor = lpips.im2tensor(gt)
        lpips_val = lpips_model(rendered_tensor, gt_tensor).item()
        lpips_values.append(lpips_val)
    
    return {
        'psnr': {
            'mean': np.mean(psnr_values),
            'std': np.std(psnr_values),
            'min': np.min(psnr_values),
            'max': np.max(psnr_values)
        },
        'ssim': {
            'mean': np.mean(ssim_values),
            'std': np.std(ssim_values),
            'min': np.min(ssim_values),
            'max': np.max(ssim_values)
        },
        'lpips': {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'min': np.min(lpips_values),
            'max': np.max(lpips_values)
        }
    }
```

---

## 🏅 Leaderboard Categories

### Main Leaderboard

The primary ranking based on total score:

| Rank | Student ID | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussians | Size | Quality | Efficiency | Doc | Bonus | **Total** |
|------|------------|-------|-------|--------|-----------|------|---------|------------|-----|-------|-----------|
| 🥇 | - | - | - | - | - | - | - | - | - | - | - |
| 🥈 | - | - | - | - | - | - | - | - | - | - | - |
| 🥉 | - | - | - | - | - | - | - | - | - | - | - |

### Special Category Awards

#### 🎨 Best Visual Quality Award
Highest combined PSNR + SSIM score, recognizing exceptional rendering quality.

#### ⚡ Most Efficient Award
Best quality-to-size ratio, recognizing efficient model representations.

#### 🔬 Innovation Award
Most creative and impactful technical innovations.

#### 📚 Best Documentation Award
Highest documentation score, recognizing excellent technical writing.

#### 🚀 Most Improved Award
Greatest improvement from baseline to final submission.

---

## 💻 Technical Implementation

### Evaluation Server Setup

```bash
# System requirements
- Python 3.8+
- PyTorch 1.12+ with CUDA support
- OpenCV 4.5+
- lpips package
- scikit-image

# Installation
pip install torch torchvision
pip install opencv-python
pip install lpips
pip install scikit-image
pip install numpy matplotlib
```

### Running Evaluation Locally

Students can run the evaluation script locally to estimate their scores:

```bash
# Clone evaluation tools
git clone https://github.com/course/aae5303-eval-tools

# Run evaluation
python evaluate.py \
    --ply_path ./output/model.ply \
    --gt_images ./data/test_views/ \
    --output ./evaluation_results.json
```

### PLY File Validation

```python
def validate_ply(ply_path):
    """
    Validate a 3DGS PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    try:
        with open(ply_path, 'rb') as f:
            # Read header
            header = b''
            while b'end_header' not in header:
                line = f.readline()
                if not line:
                    return False, "Invalid PLY: no end_header found", None
                header += line
            
            header_str = header.decode('utf-8')
            
            # Check format
            if 'binary_little_endian' not in header_str:
                return False, "Invalid PLY: must be binary little endian", None
            
            # Extract vertex count
            match = re.search(r'element vertex (\d+)', header_str)
            if not match:
                return False, "Invalid PLY: no vertex count found", None
            
            num_vertices = int(match.group(1))
            
            # Check required properties
            required = ['x', 'y', 'z', 'f_dc_0', 'opacity', 'scale_0', 'rot_0']
            for prop in required:
                if f'property float {prop}' not in header_str:
                    return False, f"Invalid PLY: missing property {prop}", None
            
            # Validate file size
            expected_size = estimate_ply_size(header_str, num_vertices)
            actual_size = os.path.getsize(ply_path)
            
            if abs(actual_size - expected_size) > 1024:  # Allow 1KB tolerance
                return False, f"Invalid PLY: size mismatch", None
            
            metadata = {
                'num_vertices': num_vertices,
                'file_size_mb': actual_size / (1024 * 1024),
                'sh_degree': count_sh_coefficients(header_str)
            }
            
            return True, None, metadata
            
    except Exception as e:
        return False, f"Error reading PLY: {str(e)}", None
```

---

## ❓ Frequently Asked Questions

### General Questions

**Q: How many submissions are allowed?**

A: Students may submit up to 3 times. Only the highest-scoring submission will be considered for the final grade.

**Q: Can I use a different dataset?**

A: No, all submissions must use the provided HKisland dataset to ensure fair comparison.

**Q: What if my training doesn't converge?**

A: Partial credit is given based on achieved metrics. Document any issues encountered in your report.

### Technical Questions

**Q: What training iterations are recommended?**

A: A minimum of 7,000 iterations is recommended. The baseline uses 30,000 iterations.

**Q: Can I modify the OpenSplat source code?**

A: Yes! Code modifications are encouraged and can earn bonus points. Document all changes clearly.

**Q: What hardware will be used for evaluation?**

A: Evaluation will be performed on a standardized system with NVIDIA RTX 3090. Training time comparisons account for different hardware.

### Scoring Questions

**Q: Is documentation score subjective?**

A: Documentation is evaluated using a standardized rubric by multiple graders, with scores averaged.

**Q: How are ties broken?**

A: Ties are broken by: 1) Quality score, 2) Efficiency score, 3) Submission time.

**Q: Can bonus points exceed the maximum?**

A: Bonus points are capped at 15, but exceptional work may receive special recognition.

---

## 📜 Changelog

### Version 1.0 (December 2024)
- Initial release of evaluation framework
- Established baseline metrics and scoring system
- Created submission guidelines and templates

### Planned Updates
- [ ] Add video rendering evaluation
- [ ] Include temporal consistency metrics
- [ ] Support for compressed model formats

---

## 📞 Contact

For questions about the evaluation system:
- **Course Instructor**: [Email]
- **Teaching Assistants**: [Email]
- **Technical Issues**: [GitHub Issues]

---

<div align="center">

**AAE5303 - Advanced Topics in Aerospace Engineering**

*Hong Kong Polytechnic University*

*Fair • Transparent • Educational*

</div>

