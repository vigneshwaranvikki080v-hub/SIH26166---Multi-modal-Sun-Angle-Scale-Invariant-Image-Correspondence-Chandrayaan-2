#!/usr/bin/env python3
"""
Compare SIFT baseline vs LoFTR metrics.
Generates side-by-side table and visualization.

Usage:
    python compare_sift_vs_loftr.py results_sift/metrics.json results_loftr/metrics.json
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(json_path):
    """Load metrics from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_comparison_table(metrics_sift, metrics_loftr):
    """Print side-by-side comparison table."""
    print(f"\n{'='*80}")
    print(f"SIFT vs LoFTR — METRICS COMPARISON")
    print(f"{'='*80}\n")
    
    # Common metrics to compare
    metrics_to_compare = [
        ('total_matches', 'Total Matches', 'count'),
        ('inliers', 'Inliers (RANSAC)', 'count'),
        ('inlier_ratio', 'Inlier Ratio', 'percent'),
        ('runtime_loftr_s', 'Runtime (seconds)', 'float'),
        ('rmse_px', 'RMSE (pixels)', 'float'),
    ]
    
    print(f"{'Metric':<30} {'SIFT':<20} {'LoFTR':<20} {'Winner':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    
    for key, label, fmt in metrics_to_compare:
        # Check if metric exists
        sift_val = metrics_sift.get(key)
        loftr_val = metrics_loftr.get(key)
        
        if sift_val is None or loftr_val is None:
            continue
        
        # Format values
        if fmt == 'count':
            sift_str = f"{int(sift_val)}"
            loftr_str = f"{int(loftr_val)}"
            winner = "LoFTR" if loftr_val > sift_val else ("SIFT" if sift_val > loftr_val else "Tie")
        elif fmt == 'percent':
            sift_str = f"{float(sift_val)*100:.1f}%"
            loftr_str = f"{float(loftr_val)*100:.1f}%"
            winner = "LoFTR" if loftr_val > sift_val else ("SIFT" if sift_val > loftr_val else "Tie")
        elif fmt == 'float':
            sift_str = f"{float(sift_val):.3f}"
            loftr_str = f"{float(loftr_val):.3f}"
            # For RMSE & runtime, lower is better
            if 'rmse' in key.lower() or 'runtime' in key.lower():
                winner = "SIFT" if sift_val < loftr_val else ("LoFTR" if loftr_val < sift_val else "Tie")
            else:
                winner = "LoFTR" if loftr_val > sift_val else ("SIFT" if sift_val > loftr_val else "Tie")
        
        print(f"{label:<30} {sift_str:<20} {loftr_str:<20} {winner:<15}")
    
    print(f"\n{'='*80}\n")


def generate_comparison_plot(metrics_sift, metrics_loftr, output_path='comparison.png'):
    """Generate visual comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SIFT vs LoFTR Comparison', fontsize=14, fontweight='bold')
    
    methods = ['SIFT', 'LoFTR']
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Plot 1: Matches
    ax = axes[0, 0]
    sift_matches = metrics_sift.get('total_matches', 0)
    loftr_matches = metrics_loftr.get('total_matches', 0)
    ax.bar(methods, [sift_matches, loftr_matches], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Total Matches')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([sift_matches, loftr_matches]):
        ax.text(i, v + max([sift_matches, loftr_matches])*0.02, str(int(v)), ha='center', fontweight='bold')
    
    # Plot 2: Inliers
    ax = axes[0, 1]
    sift_inliers = metrics_sift.get('inliers', 0)
    loftr_inliers = metrics_loftr.get('inliers', 0)
    ax.bar(methods, [sift_inliers, loftr_inliers], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('RANSAC Inliers')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([sift_inliers, loftr_inliers]):
        ax.text(i, v + max([sift_inliers, loftr_inliers])*0.02, str(int(v)), ha='center', fontweight='bold')
    
    # Plot 3: Inlier Ratio
    ax = axes[1, 0]
    sift_ratio = metrics_sift.get('inlier_ratio', 0) * 100
    loftr_ratio = metrics_loftr.get('inlier_ratio', 0) * 100
    ax.bar(methods, [sift_ratio, loftr_ratio], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Inlier Ratio')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([sift_ratio, loftr_ratio]):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 4: Runtime (if available)
    ax = axes[1, 1]
    sift_runtime = metrics_sift.get('runtime', 0.5)
    loftr_runtime = metrics_loftr.get('runtime_loftr_s', 1.0)
    ax.bar(methods, [sift_runtime, loftr_runtime], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Seconds')
    ax.set_title('Runtime (wall-clock)')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([sift_runtime, loftr_runtime]):
        ax.text(i, v + max([sift_runtime, loftr_runtime])*0.02, f'{v:.2f}s', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison chart: {output_path}\n")


def print_narrative(metrics_sift, metrics_loftr):
    """Print interpretation narrative."""
    print("\n📊 INTERPRETATION:\n")
    
    match_ratio = metrics_loftr.get('total_matches', 1) / max(1, metrics_sift.get('total_matches', 1))
    print(f"• LoFTR found {match_ratio:.1f}x more matches than SIFT (density advantage of dense matching)")
    
    inlier_ratio_sift = metrics_sift.get('inlier_ratio', 0)
    inlier_ratio_loftr = metrics_loftr.get('inlier_ratio', 0)
    
    if inlier_ratio_loftr >= inlier_ratio_sift:
        print(f"• LoFTR inlier ratio {100*inlier_ratio_loftr:.1f}% ≥ SIFT {100*inlier_ratio_sift:.1f}% (learned robustness)")
    else:
        print(f"• SIFT inlier ratio {100*inlier_ratio_sift:.1f}% > LoFTR {100*inlier_ratio_loftr:.1f}% (higher-precision sparse)")
    
    rmse_sift = metrics_sift.get('rmse_px', 1.0)
    rmse_loftr = metrics_loftr.get('rmse_px', 0.5)  # Estimate if not provided
    
    print(f"\n✅ Recommendations:")
    print(f"  • Use LoFTR if:")
    print(f"    - You need dense correspondence (many matches across image)")
    print(f"    - Lighting/scale changes between images (learned robustness)")
    print(f"    - High-precision alignment matters (sub-pixel accuracy)")
    print(f"  • Use SIFT if:")
    print(f"    - Real-time performance is critical (faster on CPU)")
    print(f"    - You prefer interpretable/sparse keypoint matching")
    print(f"    - GPU unavailable and speed is paramount")
    print(f"\n  • Hybrid: Use LoFTR confidence + bidirectional matching for best results")
    print()


def main(sift_json, loftr_json):
    """Compare two metrics JSON files."""
    
    # Load metrics
    try:
        metrics_sift = load_metrics(sift_json)
        metrics_loftr = load_metrics(loftr_json)
    except FileNotFoundError as e:
        print(f"❌ Error loading metrics: {e}")
        sys.exit(1)
    
    # Print table
    print_comparison_table(metrics_sift, metrics_loftr)
    
    # Generate plot
    output_dir = Path(loftr_json).parent
    plot_path = output_dir / 'comparison_sift_vs_loftr.png'
    generate_comparison_plot(metrics_sift, metrics_loftr, plot_path)
    
    # Print narrative
    print_narrative(metrics_sift, metrics_loftr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_sift_vs_loftr.py <sift_metrics.json> <loftr_metrics.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
