#!/usr/bin/env python3
"""
AAE5303 3D Gaussian Splatting Assignment - Leaderboard Generator

This script aggregates evaluation results and generates leaderboard visualizations.

Usage:
    python generate_leaderboard.py --results_dir <dir> --output <output_dir>

Author: AAE5303 Course Team
Version: 1.0
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


# ==============================================================================
# Configuration
# ==============================================================================

GRADE_BOUNDARIES = {
    'A+': 95,
    'A': 85,
    'B+': 75,
    'B': 65,
    'C+': 55,
    'C': 45,
    'D': 35,
    'F': 0
}

COLORS = {
    'gold': '#FFD700',
    'silver': '#C0C0C0',
    'bronze': '#CD7F32',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'background': '#F8F9FA'
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all evaluation result JSON files from a directory.
    
    Args:
        results_dir: Directory containing evaluation JSON files
        
    Returns:
        List of evaluation result dictionaries
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return results
    
    for json_file in results_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded: {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"Error loading {json_file.name}: {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    return results


def extract_leaderboard_data(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract and format data for leaderboard display.
    
    Args:
        results: List of evaluation results
        
    Returns:
        List of formatted leaderboard entries
    """
    entries = []
    
    for result in results:
        try:
            # Handle both nested and flat structures
            if 'evaluation_summary' in result:
                summary = result['evaluation_summary']
                quality = result.get('quality_metrics', {})
                efficiency = result.get('efficiency_metrics', {})
                scores = result.get('scores', {})
            else:
                summary = result
                quality = result
                efficiency = result
                scores = result
            
            entry = {
                'student_id': summary.get('student_id', 'Unknown'),
                'submission_date': summary.get('submission_date', ''),
                'psnr': quality.get('psnr_mean', 0),
                'ssim': quality.get('ssim_mean', 0),
                'lpips': quality.get('lpips_mean', 1),
                'num_gaussians': efficiency.get('num_gaussians', 0),
                'file_size_mb': efficiency.get('file_size_mb', 0),
                'quality_score': scores.get('quality_total', 0),
                'efficiency_score': scores.get('efficiency_total', 0),
                'documentation_score': scores.get('documentation_score', 0),
                'bonus_score': scores.get('bonus_total', 0),
                'total_score': scores.get('total_score', summary.get('total_score', 0)),
                'grade': scores.get('grade', summary.get('grade', 'F'))
            }
            
            entries.append(entry)
            
        except Exception as e:
            print(f"Error extracting data: {e}")
    
    # Sort by total score descending
    entries.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Add rank
    for i, entry in enumerate(entries, 1):
        entry['rank'] = i
    
    return entries


# ==============================================================================
# Statistics Calculation
# ==============================================================================

def calculate_statistics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate class statistics from leaderboard entries.
    
    Args:
        entries: List of leaderboard entries
        
    Returns:
        Dictionary of statistics
    """
    if not entries:
        return {}
    
    scores = [e['total_score'] for e in entries]
    psnr_values = [e['psnr'] for e in entries if e['psnr'] > 0]
    ssim_values = [e['ssim'] for e in entries if e['ssim'] > 0]
    lpips_values = [e['lpips'] for e in entries if e['lpips'] < 1]
    gaussian_counts = [e['num_gaussians'] for e in entries if e['num_gaussians'] > 0]
    file_sizes = [e['file_size_mb'] for e in entries if e['file_size_mb'] > 0]
    
    # Grade distribution
    grade_dist = {}
    for grade in GRADE_BOUNDARIES.keys():
        grade_dist[grade] = sum(1 for e in entries if e['grade'] == grade)
    
    # Pass rate (C or above)
    passing_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C']
    pass_count = sum(1 for e in entries if e['grade'] in passing_grades)
    
    return {
        'total_submissions': len(entries),
        'score_stats': {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        },
        'quality_stats': {
            'psnr': {
                'mean': float(np.mean(psnr_values)) if psnr_values else 0,
                'std': float(np.std(psnr_values)) if psnr_values else 0,
                'min': float(np.min(psnr_values)) if psnr_values else 0,
                'max': float(np.max(psnr_values)) if psnr_values else 0
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)) if ssim_values else 0,
                'std': float(np.std(ssim_values)) if ssim_values else 0,
                'min': float(np.min(ssim_values)) if ssim_values else 0,
                'max': float(np.max(ssim_values)) if ssim_values else 0
            },
            'lpips': {
                'mean': float(np.mean(lpips_values)) if lpips_values else 0,
                'std': float(np.std(lpips_values)) if lpips_values else 0,
                'min': float(np.min(lpips_values)) if lpips_values else 0,
                'max': float(np.max(lpips_values)) if lpips_values else 0
            }
        },
        'efficiency_stats': {
            'gaussians': {
                'mean': float(np.mean(gaussian_counts)) if gaussian_counts else 0,
                'std': float(np.std(gaussian_counts)) if gaussian_counts else 0,
                'min': float(np.min(gaussian_counts)) if gaussian_counts else 0,
                'max': float(np.max(gaussian_counts)) if gaussian_counts else 0
            },
            'file_size': {
                'mean': float(np.mean(file_sizes)) if file_sizes else 0,
                'std': float(np.std(file_sizes)) if file_sizes else 0,
                'min': float(np.min(file_sizes)) if file_sizes else 0,
                'max': float(np.max(file_sizes)) if file_sizes else 0
            }
        },
        'grade_distribution': grade_dist,
        'pass_rate': pass_count / len(entries) if entries else 0
    }


def identify_award_winners(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Identify winners for special awards.
    
    Args:
        entries: List of leaderboard entries
        
    Returns:
        Dictionary mapping award names to winner information
    """
    if not entries:
        return {}
    
    awards = {}
    
    # Best Visual Quality (highest quality score)
    quality_sorted = sorted(entries, key=lambda x: x['quality_score'], reverse=True)
    if quality_sorted:
        winner = quality_sorted[0]
        awards['best_quality'] = {
            'student_id': winner['student_id'],
            'psnr': winner['psnr'],
            'ssim': winner['ssim'],
            'quality_score': winner['quality_score']
        }
    
    # Most Efficient (best quality-to-size ratio)
    efficiency_entries = [e for e in entries if e['file_size_mb'] > 0]
    if efficiency_entries:
        for e in efficiency_entries:
            e['quality_size_ratio'] = e['quality_score'] / e['file_size_mb']
        
        efficiency_sorted = sorted(efficiency_entries, 
                                   key=lambda x: x['quality_size_ratio'], 
                                   reverse=True)
        winner = efficiency_sorted[0]
        awards['most_efficient'] = {
            'student_id': winner['student_id'],
            'num_gaussians': winner['num_gaussians'],
            'file_size_mb': winner['file_size_mb'],
            'ratio': winner['quality_size_ratio']
        }
    
    # Innovation Award (highest bonus score)
    bonus_sorted = sorted(entries, key=lambda x: x['bonus_score'], reverse=True)
    if bonus_sorted and bonus_sorted[0]['bonus_score'] > 0:
        winner = bonus_sorted[0]
        awards['innovation'] = {
            'student_id': winner['student_id'],
            'bonus_score': winner['bonus_score']
        }
    
    # Best Documentation (highest documentation score)
    doc_sorted = sorted(entries, key=lambda x: x['documentation_score'], reverse=True)
    if doc_sorted:
        winner = doc_sorted[0]
        awards['best_documentation'] = {
            'student_id': winner['student_id'],
            'documentation_score': winner['documentation_score']
        }
    
    return awards


# ==============================================================================
# Markdown Generation
# ==============================================================================

def generate_markdown_leaderboard(
    entries: List[Dict[str, Any]], 
    stats: Dict[str, Any],
    awards: Dict[str, Dict[str, Any]]
) -> str:
    """
    Generate markdown formatted leaderboard.
    
    Args:
        entries: List of leaderboard entries
        stats: Statistics dictionary
        awards: Awards dictionary
        
    Returns:
        Markdown string
    """
    now = datetime.now().strftime("%B %d, %Y %H:%M UTC")
    
    md = f"""# 🏆 AAE5303 3DGS Assignment Leaderboard

**Last Updated**: {now}

**Total Submissions**: {stats.get('total_submissions', 0)} | **Pass Rate**: {stats.get('pass_rate', 0)*100:.1f}%

---

## 📊 Main Leaderboard

| Rank | Student ID | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Gaussians | Size (MB) | Quality | Efficiency | Doc | Bonus | **Total** | Grade |
|:----:|:----------:|:------:|:------:|:-------:|:---------:|:---------:|:-------:|:----------:|:---:|:-----:|:---------:|:-----:|
"""
    
    # Add entries
    for entry in entries:
        rank = entry['rank']
        if rank == 1:
            rank_str = "🥇"
        elif rank == 2:
            rank_str = "🥈"
        elif rank == 3:
            rank_str = "🥉"
        else:
            rank_str = str(rank)
        
        md += f"| {rank_str} | {entry['student_id']} | {entry['psnr']:.2f} | {entry['ssim']:.4f} | {entry['lpips']:.3f} | {entry['num_gaussians']:,} | {entry['file_size_mb']:.1f} | {entry['quality_score']:.1f} | {entry['efficiency_score']:.1f} | {entry['documentation_score']:.0f} | {entry['bonus_score']:.0f} | **{entry['total_score']:.1f}** | {entry['grade']} |\n"
    
    # Add awards section
    md += """
---

## 🏅 Special Awards

"""
    
    if 'best_quality' in awards:
        a = awards['best_quality']
        md += f"""### 🎨 Best Visual Quality Award
**Winner**: Student {a['student_id']}
- PSNR: {a['psnr']:.2f} dB
- SSIM: {a['ssim']:.4f}
- Quality Score: {a['quality_score']:.1f}/50

"""
    
    if 'most_efficient' in awards:
        a = awards['most_efficient']
        md += f"""### ⚡ Most Efficient Award
**Winner**: Student {a['student_id']}
- Gaussians: {a['num_gaussians']:,}
- File Size: {a['file_size_mb']:.1f} MB
- Quality-to-Size Ratio: {a['ratio']:.4f}

"""
    
    if 'innovation' in awards:
        a = awards['innovation']
        md += f"""### 🔬 Innovation Award
**Winner**: Student {a['student_id']}
- Bonus Points: {a['bonus_score']:.0f}

"""
    
    if 'best_documentation' in awards:
        a = awards['best_documentation']
        md += f"""### 📚 Best Documentation Award
**Winner**: Student {a['student_id']}
- Documentation Score: {a['documentation_score']:.0f}/15

"""
    
    # Add statistics section
    score_stats = stats.get('score_stats', {})
    quality_stats = stats.get('quality_stats', {})
    
    md += f"""---

## 📈 Class Statistics

### Score Distribution

| Metric | Value |
|--------|-------|
| **Mean Score** | {score_stats.get('mean', 0):.1f} |
| **Median Score** | {score_stats.get('median', 0):.1f} |
| **Std Deviation** | {score_stats.get('std', 0):.1f} |
| **Highest Score** | {score_stats.get('max', 0):.1f} |
| **Lowest Score** | {score_stats.get('min', 0):.1f} |

### Quality Metrics Summary

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| PSNR (dB) | {quality_stats.get('psnr', {}).get('mean', 0):.2f} | {quality_stats.get('psnr', {}).get('std', 0):.2f} | {quality_stats.get('psnr', {}).get('min', 0):.2f} | {quality_stats.get('psnr', {}).get('max', 0):.2f} |
| SSIM | {quality_stats.get('ssim', {}).get('mean', 0):.4f} | {quality_stats.get('ssim', {}).get('std', 0):.4f} | {quality_stats.get('ssim', {}).get('min', 0):.4f} | {quality_stats.get('ssim', {}).get('max', 0):.4f} |
| LPIPS | {quality_stats.get('lpips', {}).get('mean', 0):.4f} | {quality_stats.get('lpips', {}).get('std', 0):.4f} | {quality_stats.get('lpips', {}).get('min', 0):.4f} | {quality_stats.get('lpips', {}).get('max', 0):.4f} |

---

<div align="center">

*Generated by AAE5303 Evaluation System v1.0*

</div>
"""
    
    return md


# ==============================================================================
# Visualization Generation
# ==============================================================================

def generate_visualizations(
    entries: List[Dict[str, Any]], 
    stats: Dict[str, Any],
    output_dir: str
):
    """
    Generate visualization figures for the leaderboard.
    
    Args:
        entries: List of leaderboard entries
        stats: Statistics dictionary
        output_dir: Directory to save figures
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualizations - matplotlib not available")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    
    # 1. Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    scores = [e['total_score'] for e in entries]
    
    bins = [35, 45, 55, 65, 75, 85, 95, 105]
    n, bins_out, patches = ax.hist(scores, bins=bins, edgecolor='white', linewidth=1.2)
    
    # Color by grade
    grade_colors = {
        'F': COLORS['danger'],
        'D': COLORS['warning'],
        'C': COLORS['accent'],
        'C+': COLORS['accent'],
        'B': COLORS['primary'],
        'B+': COLORS['primary'],
        'A': COLORS['success'],
        'A+': COLORS['gold']
    }
    
    for i, patch in enumerate(patches):
        if i == 0:
            patch.set_facecolor(COLORS['danger'])
        elif i == 1:
            patch.set_facecolor(COLORS['warning'])
        elif i == 2:
            patch.set_facecolor(COLORS['accent'])
        elif i == 3:
            patch.set_facecolor(COLORS['primary'])
        elif i == 4:
            patch.set_facecolor(COLORS['secondary'])
        else:
            patch.set_facecolor(COLORS['success'])
    
    ax.set_xlabel('Total Score', fontweight='bold')
    ax.set_ylabel('Number of Students', fontweight='bold')
    ax.set_title('Score Distribution', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'score_distribution.png'}")
    
    # 2. Quality vs Efficiency Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [e['file_size_mb'] for e in entries]
    quality = [e['quality_score'] for e in entries]
    psnr = [e['psnr'] for e in entries]
    
    scatter = ax.scatter(sizes, psnr, c=quality, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
    
    # Annotate top 3
    for entry in entries[:3]:
        ax.annotate(f"#{entry['rank']}", 
                   (entry['file_size_mb'], entry['psnr']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Quality Score', fontweight='bold')
    
    ax.set_xlabel('File Size (MB)', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('Quality vs Efficiency Trade-off', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'quality_vs_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'quality_vs_efficiency.png'}")
    
    # 3. Metrics Radar Chart (for top 3)
    if len(entries) >= 3:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['PSNR', 'SSIM', 'LPIPS\n(inv)', 'Gaussians\n(inv)', 'File Size\n(inv)']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Normalize values for radar
        max_psnr = max(e['psnr'] for e in entries)
        max_ssim = max(e['ssim'] for e in entries)
        max_gaussians = max(e['num_gaussians'] for e in entries)
        max_size = max(e['file_size_mb'] for e in entries)
        
        colors_top3 = [COLORS['gold'], COLORS['silver'], COLORS['bronze']]
        
        for i, entry in enumerate(entries[:3]):
            values = [
                entry['psnr'] / max_psnr,
                entry['ssim'],
                1 - entry['lpips'],  # Inverted
                1 - (entry['num_gaussians'] / max_gaussians),  # Inverted
                1 - (entry['file_size_mb'] / max_size)  # Inverted
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f"#{i+1}: {entry['student_id']}", color=colors_top3[i])
            ax.fill(angles, values, alpha=0.25, color=colors_top3[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Top 3 Comparison', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.savefig(output_path / 'top3_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'top3_radar.png'}")
    
    # 4. Grade Distribution Pie Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    
    grade_dist = stats.get('grade_distribution', {})
    grades = [g for g, c in grade_dist.items() if c > 0]
    counts = [grade_dist[g] for g in grades]
    
    if counts:
        colors_pie = [COLORS['success'] if g in ['A+', 'A'] else
                     COLORS['primary'] if g in ['B+', 'B'] else
                     COLORS['accent'] if g in ['C+', 'C'] else
                     COLORS['warning'] if g == 'D' else
                     COLORS['danger'] for g in grades]
        
        wedges, texts, autotexts = ax.pie(counts, labels=grades, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
        ax.set_title('Grade Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path / 'grade_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'grade_distribution.png'}")


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate AAE5303 3DGS Assignment Leaderboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing evaluation JSON files')
    parser.add_argument('--output', default='./leaderboard_output',
                        help='Output directory for leaderboard files')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AAE5303 Leaderboard Generator")
    print("=" * 60)
    
    # Load results
    print("\n[1/4] Loading evaluation results...")
    results = load_evaluation_results(args.results_dir)
    
    if not results:
        print("No results found. Exiting.")
        return 1
    
    print(f"Loaded {len(results)} submissions")
    
    # Extract leaderboard data
    print("\n[2/4] Processing leaderboard data...")
    entries = extract_leaderboard_data(results)
    stats = calculate_statistics(entries)
    awards = identify_award_winners(entries)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown
    print("\n[3/4] Generating leaderboard markdown...")
    markdown = generate_markdown_leaderboard(entries, stats, awards)
    
    md_path = output_path / 'LEADERBOARD.md'
    with open(md_path, 'w') as f:
        f.write(markdown)
    print(f"Saved: {md_path}")
    
    # Save JSON data
    json_path = output_path / 'leaderboard_data.json'
    with open(json_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'entries': entries,
            'statistics': stats,
            'awards': awards
        }, f, indent=2)
    print(f"Saved: {json_path}")
    
    # Generate visualizations
    if not args.no_viz:
        print("\n[4/4] Generating visualizations...")
        figures_dir = output_path / 'figures'
        generate_visualizations(entries, stats, str(figures_dir))
    else:
        print("\n[4/4] Skipping visualizations")
    
    print("\n" + "=" * 60)
    print("Leaderboard generation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

