# AAE5303 3D Gaussian Splatting Assignment - Detailed Grading Rubric

<div align="center">

**Comprehensive Scoring Guidelines for Instructors and Students**

Version 1.0 | December 2024

</div>

---

## 📊 Score Distribution Overview

| Category | Points | Percentage | Evaluation Type |
|----------|--------|------------|-----------------|
| **Quality Metrics** | 50 | 47.6% | Automated |
| **Efficiency Metrics** | 25 | 23.8% | Automated |
| **Documentation** | 15 | 14.3% | Manual |
| **Bonus Points** | 15 | 14.3% | Hybrid |
| **Total** | **105** | **100%** | - |

---

## 1. Quality Metrics Rubric (50 Points)

### 1.1 PSNR Score (20 Points)

Peak Signal-to-Noise Ratio measures pixel-level accuracy of rendered images.

| Score Range | PSNR (dB) | Qualitative Assessment | Description |
|-------------|-----------|------------------------|-------------|
| 18-20 | ≥ 33 | **Exceptional** | Near-perfect reconstruction, publication quality |
| 15-17 | 30-33 | **Excellent** | High-quality reconstruction with minimal artifacts |
| 12-14 | 27-30 | **Good** | Solid reconstruction, some minor issues |
| 9-11 | 24-27 | **Satisfactory** | Acceptable quality, noticeable artifacts |
| 6-8 | 21-24 | **Below Average** | Visible quality issues |
| 3-5 | 18-21 | **Poor** | Significant reconstruction errors |
| 0-2 | < 18 | **Unacceptable** | Major failures in reconstruction |

**Scoring Formula:**
```
score = max(0, min(20, (PSNR - 20) / 15 * 20))
```

**What affects PSNR:**
- Training iterations (more = better)
- Learning rate tuning
- Densification parameters
- Image resolution during training

---

### 1.2 SSIM Score (20 Points)

Structural Similarity Index measures perceptual quality based on structure, luminance, and contrast.

| Score Range | SSIM | Qualitative Assessment | Description |
|-------------|------|------------------------|-------------|
| 18-20 | ≥ 0.93 | **Exceptional** | Structural details perfectly preserved |
| 15-17 | 0.90-0.93 | **Excellent** | Very high structural fidelity |
| 12-14 | 0.87-0.90 | **Good** | Good structural preservation |
| 9-11 | 0.84-0.87 | **Satisfactory** | Acceptable structural quality |
| 6-8 | 0.81-0.84 | **Below Average** | Some structural degradation |
| 3-5 | 0.78-0.81 | **Poor** | Significant structural issues |
| 0-2 | < 0.78 | **Unacceptable** | Major structural failures |

**Scoring Formula:**
```
score = max(0, min(20, (SSIM - 0.75) / 0.20 * 20))
```

**What affects SSIM:**
- Edge preservation in reconstruction
- Texture detail rendering
- Color accuracy
- Spherical harmonics degree

---

### 1.3 LPIPS Score (10 Points)

Learned Perceptual Image Patch Similarity uses deep features to assess perceptual quality.

| Score Range | LPIPS | Qualitative Assessment | Description |
|-------------|-------|------------------------|-------------|
| 9-10 | < 0.07 | **Exceptional** | Perceptually indistinguishable |
| 7-8 | 0.07-0.10 | **Excellent** | Very high perceptual quality |
| 5-6 | 0.10-0.13 | **Good** | Good perceptual similarity |
| 3-4 | 0.13-0.16 | **Satisfactory** | Acceptable perceptual quality |
| 1-2 | 0.16-0.19 | **Poor** | Noticeable perceptual differences |
| 0 | ≥ 0.19 | **Unacceptable** | Major perceptual discrepancies |

**Scoring Formula:**
```
score = max(0, min(10, (0.20 - LPIPS) / 0.15 * 10))
```

**What affects LPIPS:**
- High-frequency detail preservation
- Semantic content accuracy
- View-dependent effects handling
- Overall visual coherence

---

## 2. Efficiency Metrics Rubric (25 Points)

### 2.1 Gaussian Count Efficiency (10 Points)

Evaluates how efficiently the model represents the scene with fewer Gaussians.

| Score Range | Ratio to Initial | Assessment | Description |
|-------------|------------------|------------|-------------|
| 9-10 | < 0.5× | **Exceptional** | Significant pruning while maintaining quality |
| 7-8 | 0.5-0.75× | **Excellent** | Effective density control |
| 5-6 | 0.75-1.0× | **Good** | Maintained or slightly reduced |
| 3-4 | 1.0-1.5× | **Satisfactory** | Moderate growth acceptable |
| 1-2 | 1.5-2.0× | **Poor** | Excessive Gaussian proliferation |
| 0 | > 2.0× | **Unacceptable** | Uncontrolled growth |

**Scoring Formula:**
```python
ratio = final_gaussians / initial_points
if ratio <= 0.5:
    score = 10
elif ratio <= 1.0:
    score = 10 * (1.0 - (ratio - 0.5))
elif ratio <= 2.0:
    score = 5 * (2.0 - ratio)
else:
    score = 0
```

**What affects Gaussian count:**
- Densification threshold settings
- Pruning aggressiveness
- Training iterations (affects convergence)
- Scene complexity

---

### 2.2 File Size Efficiency (8 Points)

Evaluates the output model file size for deployment feasibility.

| Score Range | File Size | Assessment | Description |
|-------------|-----------|------------|-------------|
| 7-8 | < 100 MB | **Excellent** | Highly deployable |
| 5-6 | 100-200 MB | **Good** | Practical for most applications |
| 3-4 | 200-350 MB | **Satisfactory** | Acceptable for desktop |
| 1-2 | 350-500 MB | **Poor** | Large but usable |
| 0 | > 500 MB | **Excessive** | Impractical for deployment |

**Scoring Formula:**
```
score = max(0, min(8, (500 - size_mb) / 450 * 8))
```

---

### 2.3 Training Efficiency (7 Points)

Evaluates training speed normalized by hardware capability.

| Device | Excellent (ms/iter) | Poor (ms/iter) |
|--------|---------------------|----------------|
| CUDA GPU | < 50 | > 500 |
| Apple MPS | < 100 | > 1000 |
| CPU | < 5000 | > 50000 |

**Note:** CPU submissions are not penalized for slower training as this is expected.

---

## 3. Documentation Rubric (15 Points)

### 3.1 Report Structure and Organization (4 Points)

| Score | Criteria |
|-------|----------|
| 4 | All required sections present, logical flow, clear organization, appropriate length |
| 3 | Minor organizational issues, one section weak or missing minor elements |
| 2 | Some sections missing or poorly organized, flow is difficult to follow |
| 1 | Major sections missing, disorganized, difficult to navigate |
| 0 | Report does not follow required structure |

**Checklist:**
- [ ] Introduction with clear objectives
- [ ] Methodology with sufficient detail
- [ ] Results with appropriate visualizations
- [ ] Discussion of findings
- [ ] Conclusions and future work
- [ ] References properly formatted

---

### 3.2 Technical Depth and Understanding (4 Points)

| Score | Criteria |
|-------|----------|
| 4 | Demonstrates deep understanding of 3DGS, insightful analysis, connects theory to practice |
| 3 | Good understanding shown, mostly accurate technical content, reasonable analysis |
| 2 | Basic understanding, some technical inaccuracies, surface-level analysis |
| 1 | Limited understanding demonstrated, significant inaccuracies |
| 0 | Fundamental misunderstanding of core concepts |

**Assessment Areas:**
- Explanation of Gaussian splatting algorithm
- Understanding of loss functions
- Comprehension of optimization process
- Analysis of results in context

---

### 3.3 Visualizations and Figures (4 Points)

| Score | Criteria |
|-------|----------|
| 4 | Professional quality figures, informative, well-labeled, enhance understanding |
| 3 | Good figures, mostly clear and informative, minor issues with labels/legends |
| 2 | Basic figures, some clarity issues, missing some important visualizations |
| 1 | Poor quality figures, missing labels, difficult to interpret |
| 0 | Missing required visualizations or figures are unusable |

**Required Visualizations:**
- [ ] Training loss curve
- [ ] Before/after comparisons
- [ ] Quality metrics summary
- [ ] Model statistics

---

### 3.4 Writing Quality (3 Points)

| Score | Criteria |
|-------|----------|
| 3 | Clear, concise, professional academic writing, proper grammar and spelling |
| 2 | Minor writing issues, generally clear but some awkward phrasing |
| 1 | Frequent errors, unclear passages, unprofessional tone |
| 0 | Incomprehensible or machine-generated without review |

---

## 4. Bonus Points Rubric (Up to 15 Points)

### 4.1 Quality Excellence Bonus (Up to 3 Points)

| Bonus | Criteria |
|-------|----------|
| +1 | PSNR > 30 dB |
| +1 | SSIM > 0.92 |
| +1 | LPIPS < 0.08 |

### 4.2 Efficiency Excellence Bonus (Up to 2 Points)

| Bonus | Criteria |
|-------|----------|
| +1 | Gaussian count < 50% of initial |
| +1 | File size < 100 MB with PSNR > 25 dB |

### 4.3 Innovation Bonus (Up to 10 Points)

| Innovation Type | Points | Requirements |
|-----------------|--------|--------------|
| Novel loss function | 2-4 | Implemented and documented, measurable impact |
| Improved densification | 2-3 | Clear improvement over baseline |
| Compression technique | 2-3 | Significant size reduction with quality |
| Anti-aliasing method | 2-3 | Mip-splatting or equivalent |
| Speed optimization | 2-3 | Measurable training/rendering speedup |
| Novel application | 2-4 | Creative use beyond basic reconstruction |

**Innovation Evaluation Criteria:**
1. **Novelty**: Is this significantly different from the baseline?
2. **Implementation**: Is the implementation correct and complete?
3. **Documentation**: Is the innovation well-documented?
4. **Impact**: Does it measurably improve results?

---

## 5. Penalty Conditions

### 5.1 Automatic Penalties

| Condition | Penalty |
|-----------|---------|
| Invalid PLY file | -50% of quality score |
| Missing training log | -5 points |
| Missing metadata | -3 points |
| Late submission (per day) | -10% of total |
| Plagiarism detected | -100% (automatic fail) |

### 5.2 Quality Penalties

| Condition | Penalty |
|-----------|---------|
| Report exceeds page limit | -1 per extra page |
| Missing required section | -2 per section |
| No references | -2 points |
| Incorrect file naming | -1 point |

---

## 6. Grade Boundaries

| Grade | Score Range | Percentage | Description |
|-------|-------------|------------|-------------|
| A+ | 95-105 | 90%+ | Exceptional, publication quality |
| A | 85-94 | 81-90% | Excellent, exceeds expectations |
| B+ | 75-84 | 71-80% | Very good, above average |
| B | 65-74 | 62-71% | Good, meets expectations |
| C+ | 55-64 | 52-62% | Satisfactory, basic requirements |
| C | 45-54 | 43-52% | Passing, minimum requirements |
| D | 35-44 | 33-43% | Below expectations |
| F | < 35 | < 33% | Failing |

---

## 7. Appeal Process

Students may appeal their grade within 7 days of release:

1. **Informal Review**: Contact TA to discuss concerns
2. **Formal Appeal**: Submit written appeal to course instructor
3. **Re-evaluation**: Independent re-grading by different TA
4. **Final Decision**: Course instructor makes final determination

**Valid Appeal Grounds:**
- Calculation error in automated scoring
- Misunderstanding of submission contents
- Technical issues during submission
- Documented extenuating circumstances

**Invalid Appeal Grounds:**
- Disagreement with rubric criteria
- Comparison with other students' grades
- Requests for extra credit opportunities

---

## 8. Sample Score Calculation

### Example: Strong Submission

```
Quality Metrics:
  - PSNR: 28.5 dB → 11.33 points
  - SSIM: 0.89 → 14.00 points
  - LPIPS: 0.11 → 6.00 points
  - Subtotal: 31.33/50 points

Efficiency Metrics:
  - Gaussians: 0.8× initial → 7.00 points
  - File Size: 280 MB → 3.91 points
  - Training: GPU, 80ms/iter → 5.88 points
  - Subtotal: 16.79/25 points

Documentation:
  - Structure: 4/4
  - Technical: 3/4
  - Figures: 3/4
  - Writing: 3/3
  - Subtotal: 13/15 points

Bonus:
  - PSNR > 30: No → 0
  - SSIM > 0.92: No → 0
  - Innovation: Depth regularization → 3 points
  - Subtotal: 3/15 points

TOTAL: 31.33 + 16.79 + 13 + 3 = 64.12 points
GRADE: B
```

---

## 9. Inter-Rater Reliability

For manual evaluation components (documentation, innovation):

1. Each submission reviewed by 2 graders
2. Scores averaged if difference < 2 points
3. Third grader adjudicates if difference ≥ 2 points
4. All graders use this rubric consistently

---

## 10. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial release |

---

<div align="center">

**Questions about grading?**

Contact the course TAs or post on the discussion forum.

</div>

