# 🏆 AAE5303 3D Gaussian Splatting - Leaderboard

## 📊 Evaluation Metrics

The leaderboard evaluates submissions using three standard rendering quality metrics:

### PSNR (Peak Signal-to-Noise Ratio) ↑

**Higher is better**

Measures pixel-level reconstruction accuracy between rendered images and ground truth.

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

---

### SSIM (Structural Similarity Index) ↑

**Higher is better** (Range: 0 to 1)

Measures perceptual quality based on structural information, luminance, and contrast.

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

---

### LPIPS (Learned Perceptual Image Patch Similarity) ↓

**Lower is better** (Range: 0 to 1)

Uses deep neural network features to measure perceptual similarity.

---

## 📁 Evaluation Dataset

**AMtown02** from MARS-LVIG / UAVScenes Dataset

- **MARS-LVIG Dataset**: https://mars.hku.hk/dataset.html
- **UAVScenes**: https://github.com/sijieaaa/UAVScenes

---

## 📄 Submission Format

Submit a JSON file with the following format:

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

See [submission_template.json](./submission_template.json) for the template file.

---

## 🌐 Leaderboard Website & Baseline

> **📢 The leaderboard website and baseline results will be announced later.**
