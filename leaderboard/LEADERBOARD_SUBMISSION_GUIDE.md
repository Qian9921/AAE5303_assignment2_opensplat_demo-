# AAE5303 - Leaderboard Submission Guide

## 📁 Evaluation Dataset

**AMtown02** sequence from MARS-LVIG / UAVScenes Dataset

| Resource | Link |
|----------|------|
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |

---

## 📊 Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **PSNR** | ↑ Higher is better | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | ↑ Higher is better | Structural Similarity Index (0-1) |
| **LPIPS** | ↓ Lower is better | Learned Perceptual Image Patch Similarity (0-1) |

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

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `group_id` | string | Your group ID | `"Group_01"` |
| `group_name` | string | Your group name | `"Team Alpha"` |
| `metrics.psnr` | number | PSNR value in dB | `25.67` |
| `metrics.ssim` | number | SSIM value (0-1) | `0.8834` |
| `metrics.lpips` | number | LPIPS value (0-1) | `0.1052` |
| `submission_date` | string | Date (YYYY-MM-DD) | `"2024-12-17"` |

### File Naming

`{GroupID}_leaderboard.json`

Example: `Group_01_leaderboard.json`

---

## 🌐 Leaderboard Website & Baseline

> **📢 The leaderboard submission website and baseline results will be announced later.**
