#!/usr/bin/env python3
"""
AAE5303 3D Gaussian Splatting Assignment - Submission Evaluation Script

This script provides comprehensive evaluation of 3DGS submissions including:
- Quality metrics (PSNR, SSIM, LPIPS)
- Efficiency metrics (Gaussian count, file size, training time)
- Automated scoring based on the leaderboard rubric

Usage:
    python evaluate_submission.py --ply_path <path> --gt_dir <path> [options]

Author: AAE5303 Course Team
Version: 1.0
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Optional imports for full functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Image-based metrics will be skipped.")

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using fallback metrics.")

try:
    import torch
    import lpips as lpips_module
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Perceptual metrics will be skipped.")


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class QualityMetrics:
    """Container for quality metrics."""
    psnr_mean: float = 0.0
    psnr_std: float = 0.0
    psnr_min: float = 0.0
    psnr_max: float = 0.0
    ssim_mean: float = 0.0
    ssim_std: float = 0.0
    ssim_min: float = 0.0
    ssim_max: float = 0.0
    lpips_mean: float = 1.0
    lpips_std: float = 0.0
    lpips_min: float = 1.0
    lpips_max: float = 1.0
    num_images_evaluated: int = 0


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""
    num_gaussians: int = 0
    initial_points: int = 0
    gaussian_ratio: float = 1.0
    file_size_mb: float = 0.0
    training_time_seconds: float = 0.0
    time_per_iteration_ms: float = 0.0
    num_iterations: int = 0
    device_type: str = "cpu"


@dataclass
class Scores:
    """Container for computed scores."""
    # Quality scores (50 points total)
    psnr_score: float = 0.0
    ssim_score: float = 0.0
    lpips_score: float = 0.0
    quality_total: float = 0.0
    
    # Efficiency scores (25 points total)
    gaussian_count_score: float = 0.0
    file_size_score: float = 0.0
    training_efficiency_score: float = 0.0
    efficiency_total: float = 0.0
    
    # Documentation score (15 points) - manual
    documentation_score: float = 0.0
    
    # Bonus points (up to 15)
    innovation_bonus: float = 0.0
    quality_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    bonus_total: float = 0.0
    
    # Final score
    total_score: float = 0.0
    grade: str = "F"


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    student_id: str
    submission_date: str
    ply_path: str
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    efficiency_metrics: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    scores: Scores = field(default_factory=Scores)
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ==============================================================================
# PLY File Handling
# ==============================================================================

def parse_ply_header(ply_path: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Parse the header of a PLY file and extract metadata.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Tuple of (success, metadata_dict, errors_list)
    """
    errors = []
    metadata = {
        'format': None,
        'num_vertices': 0,
        'properties': [],
        'sh_degree': 0,
        'iteration': 0
    }
    
    try:
        with open(ply_path, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
                if len(header_lines) > 1000:  # Safety limit
                    errors.append("Header too long - invalid PLY file")
                    return False, metadata, errors
            
            header = '\n'.join(header_lines)
            
            # Check format
            if 'binary_little_endian' in header:
                metadata['format'] = 'binary_little_endian'
            elif 'binary_big_endian' in header:
                metadata['format'] = 'binary_big_endian'
            elif 'ascii' in header:
                metadata['format'] = 'ascii'
            else:
                errors.append("Unknown PLY format")
                return False, metadata, errors
            
            # Extract vertex count
            match = re.search(r'element vertex (\d+)', header)
            if match:
                metadata['num_vertices'] = int(match.group(1))
            else:
                errors.append("No vertex count found in header")
                return False, metadata, errors
            
            # Extract properties
            properties = re.findall(r'property (\w+) (\w+)', header)
            metadata['properties'] = properties
            
            # Count SH coefficients to determine degree
            f_rest_count = len([p for p in properties if p[1].startswith('f_rest_')])
            if f_rest_count > 0:
                # SH degree: (degree+1)^2 - 1 coefficients per channel, 3 channels
                # f_rest has (degree+1)^2 - 1 coefficients per channel
                coeffs_per_channel = f_rest_count // 3
                # Solve: coeffs = (d+1)^2 - 1 => d = sqrt(coeffs+1) - 1
                metadata['sh_degree'] = int(math.sqrt(coeffs_per_channel + 1)) - 1 + 1
            else:
                metadata['sh_degree'] = 0
            
            # Extract iteration from comment
            match = re.search(r'comment Generated by opensplat at iteration (\d+)', header)
            if match:
                metadata['iteration'] = int(match.group(1))
            
            # Validate required properties
            required_props = ['x', 'y', 'z', 'opacity', 'scale_0', 'rot_0']
            prop_names = [p[1] for p in properties]
            for req in required_props:
                if req not in prop_names:
                    errors.append(f"Missing required property: {req}")
            
            if errors:
                return False, metadata, errors
                
            return True, metadata, errors
            
    except FileNotFoundError:
        errors.append(f"File not found: {ply_path}")
        return False, metadata, errors
    except Exception as e:
        errors.append(f"Error reading PLY file: {str(e)}")
        return False, metadata, errors


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0.0


# ==============================================================================
# Training Log Parsing
# ==============================================================================

def parse_training_log(log_path: str) -> Dict[str, Any]:
    """
    Parse an OpenSplat training log file.
    
    Args:
        log_path: Path to training log file
        
    Returns:
        Dictionary with training statistics
    """
    result = {
        'num_iterations': 0,
        'final_loss': 0.0,
        'min_loss': float('inf'),
        'losses': [],
        'initial_points': 0,
        'num_images': 0,
        'device': 'cpu',
        'training_time_estimate': 0.0
    }
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract device
        if 'Using CUDA' in content:
            result['device'] = 'cuda'
        elif 'Using MPS' in content:
            result['device'] = 'mps'
        else:
            result['device'] = 'cpu'
        
        # Extract initial points
        match = re.search(r'Reading (\d+) points', content)
        if match:
            result['initial_points'] = int(match.group(1))
        
        # Count loaded images
        result['num_images'] = content.count('Loading ')
        
        # Extract step losses
        pattern = r'Step (\d+): ([\d.]+)'
        matches = re.findall(pattern, content)
        
        for step, loss in matches:
            loss_val = float(loss)
            result['losses'].append(loss_val)
            result['min_loss'] = min(result['min_loss'], loss_val)
        
        if result['losses']:
            result['num_iterations'] = len(result['losses'])
            result['final_loss'] = result['losses'][-1]
        
        if result['min_loss'] == float('inf'):
            result['min_loss'] = 0.0
            
    except Exception as e:
        print(f"Warning: Could not parse training log: {e}")
    
    return result


# ==============================================================================
# Quality Metrics Computation
# ==============================================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (HxWxC, uint8)
        img2: Second image (HxWxC, uint8)
        
    Returns:
        PSNR value in dB
    """
    if SKIMAGE_AVAILABLE:
        return psnr_func(img1, img2, data_range=255)
    else:
        # Fallback implementation
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255**2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        img1: First image (HxWxC, uint8)
        img2: Second image (HxWxC, uint8)
        
    Returns:
        SSIM value (0 to 1)
    """
    if SKIMAGE_AVAILABLE:
        return ssim_func(img1, img2, channel_axis=2, data_range=255)
    else:
        # Simplified fallback - not recommended for actual use
        return 0.0


def compute_lpips(img1: np.ndarray, img2: np.ndarray, model=None) -> float:
    """
    Compute LPIPS perceptual similarity between two images.
    
    Args:
        img1: First image (HxWxC, uint8)
        img2: Second image (HxWxC, uint8)
        model: Pre-loaded LPIPS model (optional)
        
    Returns:
        LPIPS value (0 to 1, lower is better)
    """
    if not LPIPS_AVAILABLE:
        return 0.15  # Return middle-ground estimate
    
    try:
        if model is None:
            model = lpips_module.LPIPS(net='vgg')
        
        # Convert to tensor format
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        
        with torch.no_grad():
            lpips_val = model(img1_tensor, img2_tensor).item()
        
        return lpips_val
    except Exception as e:
        print(f"Warning: LPIPS computation failed: {e}")
        return 0.15


def evaluate_quality(rendered_dir: str, gt_dir: str) -> QualityMetrics:
    """
    Evaluate quality metrics between rendered and ground truth images.
    
    Args:
        rendered_dir: Directory containing rendered images
        gt_dir: Directory containing ground truth images
        
    Returns:
        QualityMetrics object with computed values
    """
    metrics = QualityMetrics()
    
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available, skipping quality evaluation")
        return metrics
    
    rendered_path = Path(rendered_dir)
    gt_path = Path(gt_dir)
    
    if not rendered_path.exists() or not gt_path.exists():
        print(f"Warning: Image directories not found")
        return metrics
    
    # Find matching images
    rendered_files = list(rendered_path.glob('*.png')) + list(rendered_path.glob('*.jpg'))
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    # Load LPIPS model once if available
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips_module.LPIPS(net='vgg')
        except:
            pass
    
    for rendered_file in rendered_files:
        # Try to find matching ground truth
        gt_file = gt_path / rendered_file.name
        if not gt_file.exists():
            # Try alternative extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_gt = gt_path / (rendered_file.stem + ext)
                if alt_gt.exists():
                    gt_file = alt_gt
                    break
        
        if not gt_file.exists():
            continue
        
        # Load images
        rendered_img = cv2.imread(str(rendered_file))
        gt_img = cv2.imread(str(gt_file))
        
        if rendered_img is None or gt_img is None:
            continue
        
        # Resize if necessary
        if rendered_img.shape != gt_img.shape:
            rendered_img = cv2.resize(rendered_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # Compute metrics
        psnr_val = compute_psnr(gt_img, rendered_img)
        ssim_val = compute_ssim(gt_img, rendered_img)
        lpips_val = compute_lpips(gt_img, rendered_img, lpips_model)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)
    
    # Aggregate results
    if psnr_values:
        metrics.psnr_mean = float(np.mean(psnr_values))
        metrics.psnr_std = float(np.std(psnr_values))
        metrics.psnr_min = float(np.min(psnr_values))
        metrics.psnr_max = float(np.max(psnr_values))
    
    if ssim_values:
        metrics.ssim_mean = float(np.mean(ssim_values))
        metrics.ssim_std = float(np.std(ssim_values))
        metrics.ssim_min = float(np.min(ssim_values))
        metrics.ssim_max = float(np.max(ssim_values))
    
    if lpips_values:
        metrics.lpips_mean = float(np.mean(lpips_values))
        metrics.lpips_std = float(np.std(lpips_values))
        metrics.lpips_min = float(np.min(lpips_values))
        metrics.lpips_max = float(np.max(lpips_values))
    
    metrics.num_images_evaluated = len(psnr_values)
    
    return metrics


# ==============================================================================
# Scoring Functions
# ==============================================================================

def calculate_psnr_score(psnr_value: float, max_points: float = 20.0) -> float:
    """
    Calculate score from PSNR value.
    
    Scoring curve:
    - 20 dB = 0 points
    - 27.5 dB = 10 points (50%)
    - 35 dB = 20 points (100%)
    """
    if psnr_value <= 20:
        return 0.0
    if psnr_value >= 35:
        return max_points
    
    # Linear interpolation
    normalized = (psnr_value - 20) / 15
    score = max_points * normalized
    
    return round(min(max_points, max(0, score)), 2)


def calculate_ssim_score(ssim_value: float, max_points: float = 20.0) -> float:
    """
    Calculate score from SSIM value.
    
    Scoring curve:
    - 0.75 = 0 points
    - 0.85 = 10 points (50%)
    - 0.95 = 20 points (100%)
    """
    if ssim_value <= 0.75:
        return 0.0
    if ssim_value >= 0.95:
        return max_points
    
    normalized = (ssim_value - 0.75) / 0.20
    score = max_points * normalized
    
    return round(min(max_points, max(0, score)), 2)


def calculate_lpips_score(lpips_value: float, max_points: float = 10.0) -> float:
    """
    Calculate score from LPIPS value (lower is better).
    
    Scoring curve:
    - 0.20 = 0 points
    - 0.125 = 5 points (50%)
    - 0.05 = 10 points (100%)
    """
    if lpips_value >= 0.20:
        return 0.0
    if lpips_value <= 0.05:
        return max_points
    
    normalized = (0.20 - lpips_value) / 0.15
    score = max_points * normalized
    
    return round(min(max_points, max(0, score)), 2)


def calculate_gaussian_count_score(
    num_gaussians: int, 
    initial_points: int,
    max_points: float = 10.0
) -> float:
    """
    Calculate score based on Gaussian count efficiency.
    
    Rewards models that maintain or reduce Gaussian count relative to initial SfM points.
    """
    if initial_points <= 0:
        return max_points * 0.5  # Default if baseline unknown
    
    ratio = num_gaussians / initial_points
    
    if ratio <= 0.5:
        # Significant reduction - excellent
        score = max_points
    elif ratio <= 1.0:
        # Maintained or reduced - good
        score = max_points * (1.0 - (ratio - 0.5))
    elif ratio <= 2.0:
        # Some growth - acceptable
        score = max_points * 0.5 * (2.0 - ratio)
    else:
        # Excessive growth - poor
        score = 0.0
    
    return round(min(max_points, max(0, score)), 2)


def calculate_file_size_score(size_mb: float, max_points: float = 8.0) -> float:
    """
    Calculate score based on file size.
    
    Scoring curve:
    - <= 50 MB = 8 points (100%)
    - 275 MB = 4 points (50%)
    - >= 500 MB = 0 points
    """
    if size_mb <= 50:
        return max_points
    if size_mb >= 500:
        return 0.0
    
    score = max_points * (500 - size_mb) / 450
    
    return round(min(max_points, max(0, score)), 2)


def calculate_training_efficiency_score(
    time_per_iter_ms: float,
    device_type: str,
    max_points: float = 7.0
) -> float:
    """
    Calculate score based on training efficiency.
    
    Device-specific baselines (ms per iteration):
    - CUDA: 50ms (excellent) to 500ms (poor)
    - MPS: 100ms (excellent) to 1000ms (poor)
    - CPU: 5000ms (excellent) to 50000ms (poor)
    """
    baselines = {
        'cuda': {'excellent': 50, 'poor': 500},
        'mps': {'excellent': 100, 'poor': 1000},
        'cpu': {'excellent': 5000, 'poor': 50000}
    }
    
    baseline = baselines.get(device_type, baselines['cpu'])
    
    if time_per_iter_ms <= baseline['excellent']:
        return max_points
    if time_per_iter_ms >= baseline['poor']:
        return 0.0
    
    range_size = baseline['poor'] - baseline['excellent']
    normalized = (baseline['poor'] - time_per_iter_ms) / range_size
    score = max_points * normalized
    
    return round(min(max_points, max(0, score)), 2)


def calculate_bonus_points(
    quality_metrics: QualityMetrics,
    efficiency_metrics: EfficiencyMetrics,
    innovations: List[str] = None
) -> Tuple[float, float, float]:
    """
    Calculate bonus points for exceptional performance.
    
    Returns:
        Tuple of (quality_bonus, efficiency_bonus, innovation_bonus)
    """
    quality_bonus = 0.0
    efficiency_bonus = 0.0
    innovation_bonus = 0.0
    
    # Quality bonuses
    if quality_metrics.psnr_mean > 30:
        quality_bonus += 1.0
    if quality_metrics.ssim_mean > 0.92:
        quality_bonus += 1.0
    if quality_metrics.lpips_mean < 0.08:
        quality_bonus += 1.0
    
    # Efficiency bonuses
    if efficiency_metrics.gaussian_ratio < 0.5:
        efficiency_bonus += 1.0
    if efficiency_metrics.file_size_mb < 100 and quality_metrics.psnr_mean > 25:
        efficiency_bonus += 1.0
    
    # Innovation bonus (requires manual evaluation)
    if innovations:
        # Each documented innovation can earn up to 2 points
        innovation_bonus = min(10.0, len(innovations) * 2.0)
    
    return quality_bonus, efficiency_bonus, innovation_bonus


def determine_grade(total_score: float) -> str:
    """Determine letter grade from total score."""
    if total_score >= 95:
        return "A+"
    elif total_score >= 85:
        return "A"
    elif total_score >= 75:
        return "B+"
    elif total_score >= 65:
        return "B"
    elif total_score >= 55:
        return "C+"
    elif total_score >= 45:
        return "C"
    else:
        return "F"


# ==============================================================================
# Main Evaluation Function
# ==============================================================================

def evaluate_submission(
    ply_path: str,
    student_id: str = "unknown",
    gt_dir: Optional[str] = None,
    rendered_dir: Optional[str] = None,
    training_log: Optional[str] = None,
    documentation_score: float = 0.0,
    innovations: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Perform complete evaluation of a submission.
    
    Args:
        ply_path: Path to the submitted PLY file
        student_id: Student identifier
        gt_dir: Directory containing ground truth images
        rendered_dir: Directory containing rendered images
        training_log: Path to training log file
        documentation_score: Manual documentation score (0-15)
        innovations: List of documented innovations
        
    Returns:
        EvaluationResult object with all metrics and scores
    """
    result = EvaluationResult(
        student_id=student_id,
        submission_date=datetime.now().isoformat(),
        ply_path=ply_path
    )
    
    # Step 1: Validate PLY file
    print(f"\n{'='*60}")
    print(f"Evaluating submission: {student_id}")
    print(f"{'='*60}")
    print("\n[1/5] Validating PLY file...")
    
    valid, ply_metadata, errors = parse_ply_header(ply_path)
    
    if not valid:
        result.validation_passed = False
        result.validation_errors = errors
        print(f"  ✗ Validation failed: {errors}")
        return result
    
    result.validation_passed = True
    print(f"  ✓ PLY file is valid")
    print(f"    - Vertices: {ply_metadata['num_vertices']:,}")
    print(f"    - SH Degree: {ply_metadata['sh_degree']}")
    print(f"    - Iteration: {ply_metadata['iteration']}")
    
    # Step 2: Extract efficiency metrics
    print("\n[2/5] Computing efficiency metrics...")
    
    result.efficiency_metrics.num_gaussians = ply_metadata['num_vertices']
    result.efficiency_metrics.file_size_mb = get_file_size_mb(ply_path)
    
    print(f"  - Gaussian count: {result.efficiency_metrics.num_gaussians:,}")
    print(f"  - File size: {result.efficiency_metrics.file_size_mb:.2f} MB")
    
    # Parse training log if available
    if training_log and os.path.exists(training_log):
        log_data = parse_training_log(training_log)
        result.efficiency_metrics.initial_points = log_data['initial_points']
        result.efficiency_metrics.num_iterations = log_data['num_iterations']
        result.efficiency_metrics.device_type = log_data['device']
        
        if log_data['initial_points'] > 0:
            result.efficiency_metrics.gaussian_ratio = (
                result.efficiency_metrics.num_gaussians / log_data['initial_points']
            )
        
        print(f"  - Initial points: {log_data['initial_points']:,}")
        print(f"  - Iterations: {log_data['num_iterations']}")
        print(f"  - Device: {log_data['device']}")
    else:
        # Use PLY vertex count as baseline estimate
        result.efficiency_metrics.initial_points = ply_metadata['num_vertices']
        result.efficiency_metrics.gaussian_ratio = 1.0
        result.warnings.append("Training log not provided - using default values")
    
    # Step 3: Compute quality metrics
    print("\n[3/5] Computing quality metrics...")
    
    if rendered_dir and gt_dir:
        result.quality_metrics = evaluate_quality(rendered_dir, gt_dir)
        print(f"  - Images evaluated: {result.quality_metrics.num_images_evaluated}")
        print(f"  - PSNR: {result.quality_metrics.psnr_mean:.2f} ± {result.quality_metrics.psnr_std:.2f} dB")
        print(f"  - SSIM: {result.quality_metrics.ssim_mean:.4f} ± {result.quality_metrics.ssim_std:.4f}")
        print(f"  - LPIPS: {result.quality_metrics.lpips_mean:.4f} ± {result.quality_metrics.lpips_std:.4f}")
    else:
        # Estimate from training loss if available
        if training_log and os.path.exists(training_log):
            log_data = parse_training_log(training_log)
            # Rough estimation based on final loss
            estimated_psnr = 30 - log_data['final_loss'] * 30
            result.quality_metrics.psnr_mean = max(15, min(35, estimated_psnr))
            result.quality_metrics.ssim_mean = max(0.6, min(0.95, 0.95 - log_data['final_loss']))
            result.quality_metrics.lpips_mean = max(0.05, min(0.3, log_data['final_loss'] * 0.5))
            result.warnings.append("Quality metrics estimated from training loss")
            print(f"  - Estimated PSNR: {result.quality_metrics.psnr_mean:.2f} dB")
            print(f"  - Estimated SSIM: {result.quality_metrics.ssim_mean:.4f}")
            print(f"  - Estimated LPIPS: {result.quality_metrics.lpips_mean:.4f}")
        else:
            result.warnings.append("No quality evaluation possible without rendered images")
            print("  ⚠ Skipped - no rendered images provided")
    
    # Step 4: Calculate scores
    print("\n[4/5] Calculating scores...")
    
    scores = result.scores
    
    # Quality scores
    scores.psnr_score = calculate_psnr_score(result.quality_metrics.psnr_mean)
    scores.ssim_score = calculate_ssim_score(result.quality_metrics.ssim_mean)
    scores.lpips_score = calculate_lpips_score(result.quality_metrics.lpips_mean)
    scores.quality_total = scores.psnr_score + scores.ssim_score + scores.lpips_score
    
    print(f"  Quality Scores:")
    print(f"    - PSNR: {scores.psnr_score}/20")
    print(f"    - SSIM: {scores.ssim_score}/20")
    print(f"    - LPIPS: {scores.lpips_score}/10")
    print(f"    - Total: {scores.quality_total}/50")
    
    # Efficiency scores
    scores.gaussian_count_score = calculate_gaussian_count_score(
        result.efficiency_metrics.num_gaussians,
        result.efficiency_metrics.initial_points
    )
    scores.file_size_score = calculate_file_size_score(result.efficiency_metrics.file_size_mb)
    scores.training_efficiency_score = calculate_training_efficiency_score(
        result.efficiency_metrics.time_per_iteration_ms,
        result.efficiency_metrics.device_type
    )
    scores.efficiency_total = (
        scores.gaussian_count_score + 
        scores.file_size_score + 
        scores.training_efficiency_score
    )
    
    print(f"  Efficiency Scores:")
    print(f"    - Gaussian count: {scores.gaussian_count_score}/10")
    print(f"    - File size: {scores.file_size_score}/8")
    print(f"    - Training efficiency: {scores.training_efficiency_score}/7")
    print(f"    - Total: {scores.efficiency_total}/25")
    
    # Documentation score (manual input)
    scores.documentation_score = documentation_score
    print(f"  Documentation Score: {scores.documentation_score}/15")
    
    # Bonus points
    quality_bonus, efficiency_bonus, innovation_bonus = calculate_bonus_points(
        result.quality_metrics,
        result.efficiency_metrics,
        innovations
    )
    scores.quality_bonus = quality_bonus
    scores.efficiency_bonus = efficiency_bonus
    scores.innovation_bonus = innovation_bonus
    scores.bonus_total = min(15.0, quality_bonus + efficiency_bonus + innovation_bonus)
    
    print(f"  Bonus Points:")
    print(f"    - Quality bonus: {scores.quality_bonus}")
    print(f"    - Efficiency bonus: {scores.efficiency_bonus}")
    print(f"    - Innovation bonus: {scores.innovation_bonus}")
    print(f"    - Total: {scores.bonus_total}/15")
    
    # Total score
    scores.total_score = (
        scores.quality_total +
        scores.efficiency_total +
        scores.documentation_score +
        scores.bonus_total
    )
    scores.grade = determine_grade(scores.total_score)
    
    # Step 5: Generate summary
    print("\n[5/5] Generating summary...")
    print(f"\n{'='*60}")
    print(f"  FINAL SCORE: {scores.total_score:.1f}/105")
    print(f"  GRADE: {scores.grade}")
    print(f"{'='*60}")
    
    if result.warnings:
        print("\n  Warnings:")
        for warning in result.warnings:
            print(f"    ⚠ {warning}")
    
    return result


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(result: EvaluationResult, output_path: str):
    """
    Generate a detailed evaluation report.
    
    Args:
        result: EvaluationResult object
        output_path: Path to save the report
    """
    report = {
        'evaluation_summary': {
            'student_id': result.student_id,
            'submission_date': result.submission_date,
            'validation_passed': result.validation_passed,
            'total_score': result.scores.total_score,
            'grade': result.scores.grade
        },
        'quality_metrics': asdict(result.quality_metrics),
        'efficiency_metrics': asdict(result.efficiency_metrics),
        'scores': asdict(result.scores),
        'warnings': result.warnings,
        'validation_errors': result.validation_errors
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def generate_leaderboard_entry(result: EvaluationResult) -> Dict[str, Any]:
    """
    Generate a leaderboard entry from evaluation result.
    
    Args:
        result: EvaluationResult object
        
    Returns:
        Dictionary suitable for leaderboard display
    """
    return {
        'student_id': result.student_id,
        'psnr': round(result.quality_metrics.psnr_mean, 2),
        'ssim': round(result.quality_metrics.ssim_mean, 4),
        'lpips': round(result.quality_metrics.lpips_mean, 4),
        'gaussians': result.efficiency_metrics.num_gaussians,
        'size_mb': round(result.efficiency_metrics.file_size_mb, 1),
        'quality_score': round(result.scores.quality_total, 1),
        'efficiency_score': round(result.scores.efficiency_total, 1),
        'doc_score': round(result.scores.documentation_score, 1),
        'bonus': round(result.scores.bonus_total, 1),
        'total': round(result.scores.total_score, 1),
        'grade': result.scores.grade,
        'submission_date': result.submission_date
    }


# ==============================================================================
# Command Line Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AAE5303 3D Gaussian Splatting Submission Evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with PLY file only
  python evaluate_submission.py --ply_path ./output/model.ply --student_id 12345678

  # Full evaluation with rendered images
  python evaluate_submission.py \\
      --ply_path ./output/model.ply \\
      --gt_dir ./data/test_images/ \\
      --rendered_dir ./output/rendered/ \\
      --training_log ./logs/training.log \\
      --student_id 12345678 \\
      --output ./evaluation_report.json

  # Evaluation with documentation score
  python evaluate_submission.py \\
      --ply_path ./output/model.ply \\
      --student_id 12345678 \\
      --doc_score 12.5
        """
    )
    
    parser.add_argument('--ply_path', required=True,
                        help='Path to the PLY model file')
    parser.add_argument('--student_id', default='unknown',
                        help='Student ID for identification')
    parser.add_argument('--gt_dir',
                        help='Directory containing ground truth images')
    parser.add_argument('--rendered_dir',
                        help='Directory containing rendered images')
    parser.add_argument('--training_log',
                        help='Path to training log file')
    parser.add_argument('--doc_score', type=float, default=0.0,
                        help='Manual documentation score (0-15)')
    parser.add_argument('--innovations', nargs='+',
                        help='List of documented innovations')
    parser.add_argument('--output', default='evaluation_report.json',
                        help='Output path for evaluation report')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Run evaluation
    result = evaluate_submission(
        ply_path=args.ply_path,
        student_id=args.student_id,
        gt_dir=args.gt_dir,
        rendered_dir=args.rendered_dir,
        training_log=args.training_log,
        documentation_score=args.doc_score,
        innovations=args.innovations
    )
    
    # Generate report
    generate_report(result, args.output)
    
    # Print leaderboard entry
    print("\nLeaderboard Entry:")
    entry = generate_leaderboard_entry(result)
    print(json.dumps(entry, indent=2))
    
    return 0 if result.validation_passed else 1


if __name__ == '__main__':
    sys.exit(main())

