# AAE5303 - Leaderboard Submission Guide

## 📁 Evaluation Dataset

**AMtown02** sequence from MARS-LVIG / UAVScenes Dataset

| Resource | Link |
|----------|------|
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |

---

## 🎯 Baseline Results

Before you start, review the baseline implementation to understand the workflow and benchmark:

📂 **Baseline Directory**: [../baseline/README.md](../baseline/README.md)

### Baseline Performance (300 iterations, CPU-only)

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Loss** | 0.0888 | Final loss after 300 iterations |
| **Loss Reduction** | 58.9% | From initial 0.2164 |
| **Estimated PSNR** | 20-22 dB | Approximate (test set evaluation required) |
| **Estimated SSIM** | 0.75-0.80 | Approximate (test set evaluation required) |

**Goal**: Your submission should aim to surpass these baseline metrics!

---

## 📊 Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **PSNR** | ↑ Higher is better | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | ↑ Higher is better | Structural Similarity Index (0-1) |
| **LPIPS** | ↓ Lower is better | Learned Perceptual Image Patch Similarity (0-1) |

For detailed metric definitions and reference implementations, see [README.md](./README.md).

---

## 📄 JSON Submission Format

Submit your results using the following JSON format:

```json
{
    "group_id": "YOUR_GROUP_ID",
    "group_name": "Your Group Name",
    "metrics": {
        "psnr": 25.67,
        "ssim": 0.8834,
        "lpips": 0.1052
    },
    "submission_date": "YYYY-MM-DD"
}
```

### Field Descriptions

| Field | Type | Description | Example | Required |
|-------|------|-------------|---------|----------|
| `group_id` | string | Your unique group identifier | `"Group_01"` | ✅ Yes |
| `group_name` | string | Your group/team name | `"Team Alpha"` | ✅ Yes |
| `metrics.psnr` | number | PSNR value in dB (2 decimals) | `25.67` | ✅ Yes |
| `metrics.ssim` | number | SSIM value 0-1 (4 decimals) | `0.8834` | ✅ Yes |
| `metrics.lpips` | number | LPIPS value 0-1 (4 decimals) | `0.1052` | ✅ Yes |
| `submission_date` | string | Date in YYYY-MM-DD format | `"2024-12-25"` | ✅ Yes |

### File Naming Convention

Your submission file **must** follow this naming pattern:

```
{GroupID}_leaderboard.json
```

**Examples:**
- `Group_01_leaderboard.json`
- `Group_02_leaderboard.json`
- `TeamAlpha_leaderboard.json`

---

## 🚀 Step-by-Step Submission Process

### Step 1: Train Your Model

Train your 3D Gaussian Splatting model on the **AMtown02 dataset**. See the baseline for reference:

```bash
# Example: Train with 3000 iterations
cd /root/OpenSplat/build
./opensplat /root/OpenSplat/data/AMtown02_colmap \
    --cpu -n 3000 -d 4 -o my_model.ply
```

**Tips for better results:**
- Train for more iterations (3,000-30,000)
- Use GPU if available
- Tune hyperparameters
- Reduce downscale factor if memory permits

### Step 2: Render Test Images

Use your trained model to render the test set images:

```python
# Pseudo-code (implement using your chosen viewer/renderer)
for test_image_pose in test_set:
    rendered_image = render_gaussian_splat(model, test_image_pose)
    save_image(rendered_image, output_dir)
```

**Important:** 
- Test set will be provided separately
- Render at the original resolution
- Save as PNG format

### Step 3: Calculate Metrics

Use the provided evaluation script:

```python
#!/usr/bin/env python3
"""Calculate metrics for leaderboard submission"""

import numpy as np
import json
from pathlib import Path
from datetime import date
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips
import cv2

def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_metrics(rendered_dir: str, gt_dir: str) -> dict:
    """
    Calculate PSNR, SSIM, LPIPS for all test images.
    
    Args:
        rendered_dir: Directory with your rendered images
        gt_dir: Directory with ground truth test images
    
    Returns:
        Dictionary with averaged metrics
    """
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='vgg')
    
    rendered_files = sorted(Path(rendered_dir).glob('*.png'))
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    print(f"Evaluating {len(rendered_files)} images...")
    
    for i, rendered_path in enumerate(rendered_files):
        gt_path = Path(gt_dir) / rendered_path.name
        
        if not gt_path.exists():
            print(f"Warning: Ground truth not found for {rendered_path.name}")
            continue
        
        # Load images
        rendered = load_image(str(rendered_path))
        gt = load_image(str(gt_path))
        
        # Check dimensions match
        if rendered.shape != gt.shape:
            print(f"Warning: Shape mismatch for {rendered_path.name}")
            continue
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(gt, rendered, data_range=255)
        psnr_list.append(psnr)
        
        # Calculate SSIM
        ssim = structural_similarity(gt, rendered, channel_axis=2, data_range=255)
        ssim_list.append(ssim)
        
        # Calculate LPIPS
        rendered_t = torch.from_numpy(rendered).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1
        gt_t = torch.from_numpy(gt).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1
        with torch.no_grad():
            lpips_val = lpips_model(rendered_t, gt_t).item()
        lpips_list.append(lpips_val)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(rendered_files)} images")
    
    metrics = {
        'psnr': round(np.mean(psnr_list), 2),
        'ssim': round(np.mean(ssim_list), 4),
        'lpips': round(np.mean(lpips_list), 4)
    }
    
    print(f"\n✅ Metrics calculated:")
    print(f"   PSNR:  {metrics['psnr']} dB")
    print(f"   SSIM:  {metrics['ssim']}")
    print(f"   LPIPS: {metrics['lpips']}")
    
    return metrics

def generate_submission_json(group_id: str, group_name: str, metrics: dict, output_path: str):
    """Generate submission JSON file"""
    submission = {
        "group_id": group_id,
        "group_name": group_name,
        "metrics": metrics,
        "submission_date": str(date.today())
    }
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    print(f"\n✅ Submission file created: {output_path}")
    print(json.dumps(submission, indent=4))

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate metrics for leaderboard submission')
    parser.add_argument('--rendered', required=True, help='Directory with rendered images')
    parser.add_argument('--gt', required=True, help='Directory with ground truth images')
    parser.add_argument('--group-id', required=True, help='Your group ID')
    parser.add_argument('--group-name', required=True, help='Your group name')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Calculate metrics
    metrics = calculate_metrics(args.rendered, args.gt)
    
    # Generate submission
    generate_submission_json(args.group_id, args.group_name, metrics, args.output)
    
    print("\n🎉 Ready to submit! Upload your JSON file to the leaderboard website.")
```

**Usage:**
```bash
python3 calculate_metrics.py \
    --rendered ./my_rendered_images/ \
    --gt ./test_ground_truth/ \
    --group-id "Group_01" \
    --group-name "Team Alpha" \
    --output "Group_01_leaderboard.json"
```

### Step 4: Validate Your Submission

Before submitting, validate your JSON file:

```python
#!/usr/bin/env python3
"""Validate submission JSON format"""

import json
from datetime import datetime

def validate_submission(json_path: str) -> bool:
    """Validate submission JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['group_id', 'group_name', 'metrics', 'submission_date']
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check metrics
        required_metrics = ['psnr', 'ssim', 'lpips']
        for metric in required_metrics:
            if metric not in data['metrics']:
                print(f"❌ Missing metric: {metric}")
                return False
        
        # Validate metric ranges
        psnr = data['metrics']['psnr']
        ssim = data['metrics']['ssim']
        lpips = data['metrics']['lpips']
        
        if not (0 < psnr < 100):
            print(f"❌ PSNR out of reasonable range: {psnr}")
            return False
        
        if not (0 <= ssim <= 1):
            print(f"❌ SSIM out of range [0,1]: {ssim}")
            return False
        
        if not (0 <= lpips <= 1):
            print(f"❌ LPIPS out of range [0,1]: {lpips}")
            return False
        
        # Validate date format
        try:
            datetime.strptime(data['submission_date'], '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {data['submission_date']}")
            return False
        
        print("✅ Submission file is valid!")
        print(f"\n📊 Your metrics:")
        print(f"   Group: {data['group_name']} ({data['group_id']})")
        print(f"   PSNR:  {psnr} dB")
        print(f"   SSIM:  {ssim}")
        print(f"   LPIPS: {lpips}")
        print(f"   Date:  {data['submission_date']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 validate_submission.py <submission.json>")
        sys.exit(1)
    
    is_valid = validate_submission(sys.argv[1])
    sys.exit(0 if is_valid else 1)
```

**Usage:**
```bash
python3 validate_submission.py Group_01_leaderboard.json
```

### Step 5: Submit to Leaderboard

> **📢 Submission website will be announced later**

Once the leaderboard website is live:
1. Go to the submission portal
2. Upload your JSON file
3. Verify your information
4. Submit!

Results will appear on the leaderboard within 24 hours.

---

## ✅ Submission Checklist

Before submitting, make sure you have:

- [ ] Trained your model on AMtown02 dataset
- [ ] Rendered all test set images
- [ ] Calculated all three metrics (PSNR, SSIM, LPIPS)
- [ ] Created JSON file with correct format
- [ ] Used correct file naming: `{GroupID}_leaderboard.json`
- [ ] Validated JSON file format
- [ ] Verified metrics are reasonable (compare with baseline)
- [ ] Included correct group ID and name
- [ ] Used today's date in YYYY-MM-DD format

---

## 🏆 Leaderboard Rankings

The leaderboard will display rankings based on:

### Primary Rankings
1. **Best PSNR** 🥇
2. **Best SSIM** 🥈  
3. **Best LPIPS** 🥉

### Overall Ranking 🏆

The overall ranking combines all three metrics using a weighted score:

```python
# Normalized scores (0-100 scale)
psnr_normalized = (psnr - psnr_min) / (psnr_max - psnr_min) * 100
ssim_normalized = ssim * 100
lpips_normalized = (1 - lpips) * 100  # Inverted since lower is better

# Combined score (equal weights)
overall_score = (psnr_normalized + ssim_normalized + lpips_normalized) / 3
```

---

## ❓ Frequently Asked Questions

### Q1: How many times can I submit?

**A:** You can submit multiple times. Only your best submission (highest overall score) will be displayed on the leaderboard.

### Q2: Can I see the test set images?

**A:** No, the test set is held-out to ensure fair evaluation. You will only receive camera poses for rendering.

### Q3: What if my metrics are worse than the baseline?

**A:** That's okay! The baseline is just a reference. Focus on understanding the workflow and gradually improving your results.

### Q4: Can I use methods other than OpenSplat?

**A:** Yes! You can use any 3D Gaussian Splatting implementation (original 3DGS, Mip-Splatting, etc.) as long as you submit results for the AMtown02 dataset.

### Q5: How do I get GPU access?

**A:** Contact the course instructor. GPU access may be provided for registered students.

### Q6: My model file is too large. What should I do?

**A:** You only need to submit the metrics JSON file, not the model itself. The PLY file stays on your system.

---

## 📚 Additional Resources

- **Baseline Implementation**: [../baseline/README.md](../baseline/README.md)
- **Metric Definitions**: [README.md](./README.md)
- **OpenSplat GitHub**: https://github.com/pierotofy/OpenSplat
- **3DGS Paper**: Kerbl et al., SIGGRAPH 2023
- **UAVScenes Dataset**: https://github.com/sijieaaa/UAVScenes

---

## 📧 Support

If you encounter issues:

1. Check the baseline implementation
2. Review the FAQ above
3. Contact course instructor
4. Post on course forum

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

December 2024

</div>
