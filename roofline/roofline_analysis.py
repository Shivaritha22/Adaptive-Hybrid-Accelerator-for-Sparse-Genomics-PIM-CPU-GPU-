#!/usr/bin/env python3
"""
Roofline Analysis for PIM-filtered SPMM Performance Characterization

This script:
1. Parses log files to extract FLOPs, bytes, compute time, dimensions, nnz
2. Computes operational intensity (FLOPs/Bytes) for each stage
3. Generates data reduction plots (X and Y)
4. Generates roofline plots for baseline vs AHAS
5. Creates architectural summary
"""

import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# No hardware specs - using only data from log files

@dataclass
class LogData:
    """Data extracted from log files"""
    dataset: int
    stage: str  # 'baseline', 'e2e', 'tilepredpermspmm'
    rows_X: int
    cols_X: int
    nnz_X: int
    rows_W: int
    cols_W: int
    compute_time_ms: float
    flops: float
    bytes: float
    performance_gflops: float
    performance_gbps: float
    tiles: Optional[int] = None
    dense_tiles: Optional[int] = None
    sparse_tiles: Optional[int] = None
    matrix_density: Optional[float] = None

@dataclass
class YReductionData:
    """Y matrix reduction data"""
    dataset: int
    original_rows: int
    filtered_rows: int
    reduction_percentage: float
    original_size_bytes: int
    filtered_size_bytes: int

def parse_log_file(filepath: str) -> Optional[LogData]:
    """Parse a log file and extract metrics"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Extract dataset number from filename
    match = re.search(r'(\d+)_', os.path.basename(filepath))
    if not match:
        return None
    dataset = int(match.group(1))
    
    # Extract stage type
    if 'baseline' in filepath:
        stage = 'baseline'
    elif 'e2e' in filepath:
        stage = 'e2e'
    elif 'tilepredpermspmm' in filepath:
        stage = 'tilepredpermspmm'
    else:
        return None
    
    # Extract metrics using regex
    def extract_float(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern, content)
        return float(match.group(1)) if match else default
    
    def extract_int(pattern: str, default: int = 0) -> int:
        match = re.search(pattern, content)
        return int(match.group(1)) if match else default
    
    rows_X = extract_int(r'rows_X:\s*(\d+)')
    cols_X = extract_int(r'cols_X:\s*(\d+)')
    nnz_X = extract_int(r'nnz_X:\s*(\d+)')
    rows_W = extract_int(r'rows_W:\s*(\d+)')
    cols_W = extract_int(r'cols_W:\s*(\d+)')
    compute_time = extract_float(r'spmm compute time:\s*([\d.]+)')
    flops = extract_float(r'spmm flops:\s*([\d.]+)')
    bytes_val = extract_float(r'spmm bytes:\s*([\d.]+)')
    
    # Extract performance (may not always be present)
    perf_match = re.search(r'spmm performance:\s*([\d.]+)\s*GFLOP/s,\s*([\d.]+)\s*GB/s', content)
    if perf_match:
        performance_gflops = float(perf_match.group(1))
        performance_gbps = float(perf_match.group(2))
    else:
        # Calculate from flops and bytes
        if compute_time > 0:
            performance_gflops = (flops / 1e9) / (compute_time / 1000.0)
            performance_gbps = (bytes_val / 1e9) / (compute_time / 1000.0)
        else:
            performance_gflops = 0.0
            performance_gbps = 0.0
    
    tiles = extract_int(r'tile:\s*(\d+)')
    dense_tiles = extract_int(r'dense_tiles:\s*(\d+)')
    sparse_tiles = extract_int(r'sparse_tiles:\s*(\d+)')
    matrix_density = extract_float(r'matrix_density:\s*([\d.]+)')
    
    return LogData(
        dataset=dataset,
        stage=stage,
        rows_X=rows_X,
        cols_X=cols_X,
        nnz_X=nnz_X,
        rows_W=rows_W,
        cols_W=cols_W,
        compute_time_ms=compute_time,
        flops=flops,
        bytes=bytes_val,
        performance_gflops=performance_gflops,
        performance_gbps=performance_gbps,
        tiles=tiles if tiles > 0 else None,
        dense_tiles=dense_tiles if dense_tiles > 0 else None,
        sparse_tiles=sparse_tiles if sparse_tiles > 0 else None,
        matrix_density=matrix_density if matrix_density > 0 else None
    )

def parse_y_reduction_file(filepath: str) -> Optional[YReductionData]:
    """Parse Y reduction analysis file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Extract dataset number
    match = re.search(r'y(\d+)_', os.path.basename(filepath))
    if not match:
        return None
    dataset = int(match.group(1))
    
    # Extract Y dimensions from "Y: (rows1, 32) -> (rows2, 32)"
    dim_match = re.search(r'Y:\s*\((\d+),\s*32\)\s*->\s*\((\d+),\s*32\)', content)
    if not dim_match:
        return None
    
    original_rows = int(dim_match.group(1))
    filtered_rows = int(dim_match.group(2))
    
    # Extract reduction percentage
    reduction_match = re.search(r'Reduction percentage:\s*([\d.]+)%', content)
    reduction_pct = float(reduction_match.group(1)) if reduction_match else 0.0
    
    # Extract file sizes
    orig_size_match = re.search(r'Original size:\s*([\d,]+)\s*bytes', content)
    filt_size_match = re.search(r'Filtered size:\s*([\d,]+)\s*bytes', content)
    
    original_size = int(orig_size_match.group(1).replace(',', '')) if orig_size_match else 0
    filtered_size = int(filt_size_match.group(1).replace(',', '')) if filt_size_match else 0
    
    return YReductionData(
        dataset=dataset,
        original_rows=original_rows,
        filtered_rows=filtered_rows,
        reduction_percentage=reduction_pct,
        original_size_bytes=original_size,
        filtered_size_bytes=filtered_size
    )

def load_all_logs(log_dir: str = 'logs') -> Dict[int, Dict[str, LogData]]:
    """Load all log files for datasets 1, 2, 3, 4 (files: 2, 3, 4, 5)"""
    data = {}
    # Map display dataset numbers to file dataset numbers
    file_datasets = [2, 3, 4, 5]  # Actual file numbers
    display_datasets = [1, 2, 3, 4]  # Display numbers
    dataset_map = dict(zip(display_datasets, file_datasets))
    stages = ['baseline', 'e2e', 'tilepredpermspmm']
    
    for display_ds, file_ds in dataset_map.items():
        data[display_ds] = {}
        for stage in stages:
            pattern = os.path.join(log_dir, f'{file_ds}_{stage}.txt')
            files = glob.glob(pattern)
            if files:
                log_data = parse_log_file(files[0])
                if log_data:
                    # Update dataset number in log_data to display number
                    log_data.dataset = display_ds
                    data[display_ds][stage] = log_data
    
    return data

def load_y_reductions(analysis_dir: str = 'analysis/meta') -> Dict[int, YReductionData]:
    """Load Y reduction data for all datasets"""
    data = {}
    # Map display dataset numbers to file dataset numbers
    file_datasets = [2, 3, 4, 5]  # Actual file numbers
    display_datasets = [1, 2, 3, 4]  # Display numbers
    dataset_map = dict(zip(display_datasets, file_datasets))
    
    for display_ds, file_ds in dataset_map.items():
        pattern = os.path.join(analysis_dir, f'y{file_ds}_analyse.txt')
        files = glob.glob(pattern)
        if files:
            y_data = parse_y_reduction_file(files[0])
            if y_data:
                # Update dataset number to display number
                y_data.dataset = display_ds
                data[display_ds] = y_data
    return data

def compute_operational_intensity(flops: float, bytes: float) -> float:
    """Compute operational intensity: FLOPs / Bytes"""
    if bytes == 0:
        return 0.0
    return flops / bytes

def plot_x_reduction(log_data: Dict[int, Dict[str, LogData]]):
    """Plot 1: X matrix data reduction for all datasets"""
    datasets = [1, 2, 3, 4]
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    # Prepare data
    without_filter_nnz = []
    with_filter_nnz = []
    
    for dataset in datasets:
        if dataset in log_data:
            # Without filter: use baseline or e2e
            baseline = log_data[dataset].get('baseline')
            e2e = log_data[dataset].get('e2e')
            without = baseline if baseline else e2e
            
            # With filter: use tilepredpermspmm
            with_filt = log_data[dataset].get('tilepredpermspmm')
            
            if without:
                without_filter_nnz.append(without.nnz_X / 1e6)  # Convert to millions
            else:
                without_filter_nnz.append(0)
            
            if with_filt:
                with_filter_nnz.append(with_filt.nnz_X / 1e6)
            else:
                with_filter_nnz.append(0)
    
    # Create single plot for NNZ reduction only
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot NNZ reduction
    ax.bar(x_pos - width/2, without_filter_nnz, width, label='Baseline', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos + width/2, with_filter_nnz, width, label='PIM filter', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('NNZ (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('X Matrix NNZ Reduction', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/x_reduction.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/x_reduction.png")
    plt.close()

def plot_y_reduction(y_data: Dict[int, YReductionData]):
    """Plot 2: Y matrix data reduction for all datasets - baseline vs AHAS"""
    datasets = [1, 2, 3, 4]
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    original_rows = []
    filtered_rows = []
    
    for dataset in datasets:
        if dataset in y_data:
            original_rows.append(y_data[dataset].original_rows / 1e3)  # Convert to thousands
            filtered_rows.append(y_data[dataset].filtered_rows / 1e3)
        else:
            original_rows.append(0)
            filtered_rows.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot: Baseline vs AHAS
    ax.bar(x_pos - width/2, original_rows, width, label='Baseline', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos + width/2, filtered_rows, width, label='AHAS', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Rows (Thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Y Matrix Rows Reduction: Baseline vs AHAS', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/y_reduction.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/y_reduction.png")
    plt.close()

def plot_combined_x_y_reduction(log_data: Dict[int, Dict[str, LogData]], y_data: Dict[int, YReductionData]):
    """Combined bar chart showing X, X(pim), Y, Y(pim) for all 4 datasets"""
    datasets = [1, 2, 3, 4]
    x_pos = np.arange(len(datasets))
    width = 0.2  # Width for each bar group (4 bars per dataset)
    
    # Prepare data
    x_values = []  # X without filter (rows in thousands)
    x_pim_values = []  # X with PIM filter (rows in thousands)
    y_values = []  # Y without filter (rows in thousands)
    y_pim_values = []  # Y with PIM filter (rows in thousands)
    
    for dataset in datasets:
        # X data
        if dataset in log_data:
            baseline = log_data[dataset].get('baseline')
            e2e = log_data[dataset].get('e2e')
            without = baseline if baseline else e2e
            with_filt = log_data[dataset].get('tilepredpermspmm')
            
            if without:
                x_values.append(without.rows_X / 1e3)  # Convert to thousands
            else:
                x_values.append(0)
            
            if with_filt:
                x_pim_values.append(with_filt.rows_X / 1e3)
            else:
                x_pim_values.append(0)
        else:
            x_values.append(0)
            x_pim_values.append(0)
        
        # Y data
        if dataset in y_data:
            y_values.append(y_data[dataset].original_rows / 1e3)  # Convert to thousands
            y_pim_values.append(y_data[dataset].filtered_rows / 1e3)
        else:
            y_values.append(0)
            y_pim_values.append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot bars for each dataset
    ax.bar(x_pos - 1.5*width, x_values, width, label='X', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos - 0.5*width, x_pim_values, width, label='X(pim)', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos + 0.5*width, y_values, width, label='Y', color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos + 1.5*width, y_pim_values, width, label='Y(pim)', color='#6A994E', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rows (Thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Data Reduction: X, X(pim), Y, Y(pim) for All Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/combined_x_y_reduction.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/combined_x_y_reduction.png")
    plt.close()

def plot_pim_operational_intensity(log_data: Dict[int, Dict[str, LogData]]):
    """Operational Intensity Before and After PIM Filtering"""
    datasets = [1, 2, 3, 4]
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    baseline_oi = []
    filtered_oi = []
    
    for dataset in datasets:
        if dataset in log_data:
            baseline = log_data[dataset].get('baseline')
            filtered = log_data[dataset].get('tilepredpermspmm')
            
            if baseline:
                baseline_oi_val = compute_operational_intensity(baseline.flops, baseline.bytes)
                baseline_oi.append(baseline_oi_val)
            else:
                baseline_oi.append(0)
            
            if filtered:
                filtered_oi_val = compute_operational_intensity(filtered.flops, filtered.bytes)
                filtered_oi.append(filtered_oi_val)
            else:
                filtered_oi.append(0)
        else:
            baseline_oi.append(0)
            filtered_oi.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x_pos - width/2, baseline_oi, width, label='Baseline', 
           color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos + width/2, filtered_oi, width, label='AHAS', 
           color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operational Intensity (FLOPs/Byte)', fontsize=12, fontweight='bold')
    ax.set_title('Operational Intensity Before and After PIM Filtering', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    max_val = max(baseline_oi + filtered_oi) if (baseline_oi + filtered_oi) else 1
    for i, (b, f) in enumerate(zip(baseline_oi, filtered_oi)):
        if b > 0:
            ax.text(i - width/2, b + max_val * 0.02, 
                    f'{b:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if f > 0:
            ax.text(i + width/2, f + max_val * 0.02, 
                    f'{f:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/pim_operational_intensity.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/pim_operational_intensity.png")
    plt.close()

def plot_pim_benefits_summary(log_data: Dict[int, Dict[str, LogData]]):
    """Plot 2: Benefits summary bar graph (OI Improvement and Memory Reduction)"""
    datasets = [1, 2, 3, 4]
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    oi_improvement_pct = []
    memory_reduction_pct = []
    
    for dataset in datasets:
        if dataset in log_data:
            baseline = log_data[dataset].get('baseline')
            filtered = log_data[dataset].get('tilepredpermspmm')
            
            if baseline and filtered:
                baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
                filtered_oi = compute_operational_intensity(filtered.flops, filtered.bytes)
                
                oi_improvement = (filtered_oi / baseline_oi - 1) * 100 if baseline_oi > 0 else 0
                oi_improvement_pct.append(oi_improvement)
                
                bytes_reduction = (1 - filtered.bytes / baseline.bytes) * 100 if baseline.bytes > 0 else 0
                memory_reduction_pct.append(bytes_reduction)
            else:
                oi_improvement_pct.append(0)
                memory_reduction_pct.append(0)
        else:
            oi_improvement_pct.append(0)
            memory_reduction_pct.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_twin = ax.twinx()
    
    bars1 = ax.bar(x_pos - width/2, oi_improvement_pct, width, 
                   label='OI Improvement (%)', color='#6A994E', alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    bars2 = ax_twin.bar(x_pos + width/2, memory_reduction_pct, width, 
                         label='Memory Reduction (%)', color='#BC4749', alpha=0.8,
                         edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operational Intensity Improvement (%)', fontsize=12, fontweight='bold', color='#6A994E')
    ax_twin.set_ylabel('Memory Traffic Reduction (%)', fontsize=12, fontweight='bold', color='#BC4749')
    ax.set_title('PIM Filtering Benefits: Higher OI + Less Memory Traffic', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.tick_params(axis='y', labelcolor='#6A994E')
    ax_twin.tick_params(axis='y', labelcolor='#BC4749')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legends
    ax.legend(loc='upper left', fontsize=10)
    ax_twin.legend(loc='upper right', fontsize=10)
    
    # Add value labels
    max_oi = max(oi_improvement_pct) if oi_improvement_pct else 1
    max_mem = max(memory_reduction_pct) if memory_reduction_pct else 1
    for i, (oi_imp, mem_red) in enumerate(zip(oi_improvement_pct, memory_reduction_pct)):
        if oi_imp > 0:
            ax.text(i - width/2, oi_imp + max_oi * 0.02, 
                    f'{oi_imp:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        if mem_red > 0:
            ax_twin.text(i + width/2, mem_red + max_mem * 0.02, 
                         f'{mem_red:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/pim_benefits_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/pim_benefits_summary.png")
    plt.close()

def plot_roofline(dataset: int, log_data: Dict[int, Dict[str, LogData]]):
    """Plot roofline model with curves for a specific dataset (baseline vs AHAS)"""
    if dataset not in log_data:
        print(f"No data for dataset {dataset}")
        return
    
    baseline = log_data[dataset].get('baseline')
    e2e = log_data[dataset].get('e2e')
    
    if not baseline and not e2e:
        print(f"No baseline or ahas data for dataset {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute operational intensity range for roofline curves
    op_intensity_range = np.logspace(-2, 2, 1000)  # 0.01 to 100 FLOPs/Byte
    
    # Infer roofline parameters from data
    # For baseline (CPU): infer memory bandwidth and peak compute
    cpu_mem_bandwidth = None
    cpu_peak_compute = None
    
    if baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        # Estimate memory bandwidth: if memory-bound, bandwidth ≈ performance / OI
        # Estimate peak compute: use observed performance as lower bound
        cpu_mem_bandwidth = baseline.performance_gbps  # Use measured bandwidth from logs
        cpu_peak_compute = baseline.performance_gflops * 1.5  # Estimate peak as 1.5x observed
        
        if cpu_mem_bandwidth == 0 or cpu_mem_bandwidth is None:
            cpu_mem_bandwidth = baseline.performance_gflops / baseline_oi if baseline_oi > 0 else 1.0
    
    # For ahas (CPU+GPU): infer parameters
    gpu_mem_bandwidth = None
    gpu_peak_compute = None
    
    if e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        gpu_mem_bandwidth = e2e.performance_gbps  # Use measured bandwidth from logs
        gpu_peak_compute = e2e.performance_gflops * 2.0  # Estimate peak as 2x observed
        
        if gpu_mem_bandwidth == 0:
            gpu_mem_bandwidth = e2e.performance_gflops / e2e_oi if e2e_oi > 0 else 1.0
    
    if baseline and cpu_mem_bandwidth and cpu_peak_compute:
        cpu_memory_bound = op_intensity_range * cpu_mem_bandwidth
        cpu_compute_bound = np.full_like(op_intensity_range, cpu_peak_compute)
        cpu_roofline = np.minimum(cpu_compute_bound, cpu_memory_bound)
        
        ax.loglog(op_intensity_range, cpu_roofline, 'b--', linewidth=2.5, 
                 label=f'CPU Roofline (est. {cpu_peak_compute:.1f} GFLOP/s, {cpu_mem_bandwidth:.1f} GB/s)', 
                 alpha=0.6, zorder=1)
    
    if e2e and gpu_mem_bandwidth and gpu_peak_compute:
        gpu_memory_bound = op_intensity_range * gpu_mem_bandwidth
        gpu_compute_bound = np.full_like(op_intensity_range, gpu_peak_compute)
        gpu_roofline = np.minimum(gpu_compute_bound, gpu_memory_bound)
        
        ax.loglog(op_intensity_range, gpu_roofline, 'r--', linewidth=2.5,
                 label=f'GPU Roofline (est. {gpu_peak_compute:.1f} GFLOP/s, {gpu_mem_bandwidth:.1f} GB/s)',
                 alpha=0.6, zorder=1)
    
    # Plot data points from log files
    if baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        ax.scatter(baseline_oi, baseline.performance_gflops, 
                  s=200, marker='o', color='#2E86AB', edgecolors='black', linewidth=2,
                  label=f'Baseline (CPU-only)', zorder=5)
        ax.annotate(f'Baseline\n{baseline.performance_gflops:.2f} GFLOP/s\nOI: {baseline_oi:.2f}',
                   xy=(baseline_oi, baseline.performance_gflops),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    if e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        ax.scatter(e2e_oi, e2e.performance_gflops,
                  s=200, marker='s', color='#A23B72', edgecolors='black', linewidth=2,
                  label=f'AHAS (CPU+GPU)', zorder=5)
        ax.annotate(f'AHAS\n{e2e.performance_gflops:.2f} GFLOP/s\nOI: {e2e_oi:.2f}',
                   xy=(e2e_oi, e2e.performance_gflops),
                   xytext=(10, -30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Roofline Model - Dataset {dataset}\n(Baseline vs AHAS)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set reasonable axis limits based on data
    if baseline and e2e:
        min_oi = min(compute_operational_intensity(baseline.flops, baseline.bytes),
                    compute_operational_intensity(e2e.flops, e2e.bytes)) * 0.5
        max_oi = max(compute_operational_intensity(baseline.flops, baseline.bytes),
                    compute_operational_intensity(e2e.flops, e2e.bytes)) * 2.0
        min_perf = min(baseline.performance_gflops, e2e.performance_gflops) * 0.5
        max_perf = max(baseline.performance_gflops, e2e.performance_gflops) * 2.0
        ax.set_xlim([max(0.01, min_oi), min(100, max_oi)])
        ax.set_ylim([max(0.1, min_perf), max_perf])
    elif baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        ax.set_xlim([max(0.01, baseline_oi * 0.5), min(100, baseline_oi * 2.0)])
        ax.set_ylim([max(0.1, baseline.performance_gflops * 0.5), baseline.performance_gflops * 2.0])
    elif e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        ax.set_xlim([max(0.01, e2e_oi * 0.5), min(100, e2e_oi * 2.0)])
        ax.set_ylim([max(0.1, e2e.performance_gflops * 0.5), e2e.performance_gflops * 2.0])
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/roofline_dataset_{dataset}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/roofline_dataset_{dataset}.png")
    plt.close()

def plot_single_roofline_on_axis(ax, dataset: int, log_data: Dict[int, Dict[str, LogData]], 
                                  show_legend: bool = True):
    """Helper function to plot a single roofline on a given axis"""
    if dataset not in log_data:
        return
    
    baseline = log_data[dataset].get('baseline')
    e2e = log_data[dataset].get('e2e')
    
    if not baseline and not e2e:
        return
    
    # Compute operational intensity range for roofline curves
    op_intensity_range = np.logspace(-2, 2, 1000)  # 0.01 to 100 FLOPs/Byte
    
    # Infer roofline parameters from data
    cpu_mem_bandwidth = None
    cpu_peak_compute = None
    
    if baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        cpu_mem_bandwidth = baseline.performance_gbps
        cpu_peak_compute = baseline.performance_gflops * 1.5
        
        if cpu_mem_bandwidth == 0 or cpu_mem_bandwidth is None:
            cpu_mem_bandwidth = baseline.performance_gflops / baseline_oi if baseline_oi > 0 else 1.0
    
    gpu_mem_bandwidth = None
    gpu_peak_compute = None
    
    if e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        gpu_mem_bandwidth = e2e.performance_gbps
        gpu_peak_compute = e2e.performance_gflops * 2.0
        
        if gpu_mem_bandwidth == 0:
            gpu_mem_bandwidth = e2e.performance_gflops / e2e_oi if e2e_oi > 0 else 1.0
    
    # Draw CPU roofline
    if baseline and cpu_mem_bandwidth and cpu_peak_compute:
        cpu_memory_bound = op_intensity_range * cpu_mem_bandwidth
        cpu_compute_bound = np.full_like(op_intensity_range, cpu_peak_compute)
        cpu_roofline = np.minimum(cpu_compute_bound, cpu_memory_bound)
        
        ax.loglog(op_intensity_range, cpu_roofline, 'b--', linewidth=2, 
                 label='CPU Roofline', alpha=0.6, zorder=1)
    
    # Draw GPU roofline
    if e2e and gpu_mem_bandwidth and gpu_peak_compute:
        gpu_memory_bound = op_intensity_range * gpu_mem_bandwidth
        gpu_compute_bound = np.full_like(op_intensity_range, gpu_peak_compute)
        gpu_roofline = np.minimum(gpu_compute_bound, gpu_memory_bound)
        
        ax.loglog(op_intensity_range, gpu_roofline, 'r--', linewidth=2,
                 label='GPU Roofline', alpha=0.6, zorder=1)
    
    # Plot data points
    if baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        ax.scatter(baseline_oi, baseline.performance_gflops, 
                  s=150, marker='o', color='#2E86AB', edgecolors='black', linewidth=1.5,
                  label='Baseline', zorder=5, alpha=0.8)
        ax.annotate(f'B\n{baseline.performance_gflops:.1f}',
                   xy=(baseline_oi, baseline.performance_gflops),
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                   fontsize=8)
    
    if e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        ax.scatter(e2e_oi, e2e.performance_gflops,
                  s=150, marker='s', color='#A23B72', edgecolors='black', linewidth=1.5,
                  label='AHAS', zorder=5, alpha=0.8)
        ax.annotate(f'A\n{e2e.performance_gflops:.1f}',
                   xy=(e2e_oi, e2e.performance_gflops),
                   xytext=(5, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                   fontsize=8)
    
    ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=10, fontweight='bold')
    ax.set_title(f'Dataset {dataset}', fontsize=11, fontweight='bold')
    if show_legend:
        ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set axis limits based on data
    if baseline and e2e:
        min_oi = min(compute_operational_intensity(baseline.flops, baseline.bytes),
                    compute_operational_intensity(e2e.flops, e2e.bytes)) * 0.5
        max_oi = max(compute_operational_intensity(baseline.flops, baseline.bytes),
                    compute_operational_intensity(e2e.flops, e2e.bytes)) * 2.0
        min_perf = min(baseline.performance_gflops, e2e.performance_gflops) * 0.5
        max_perf = max(baseline.performance_gflops, e2e.performance_gflops) * 2.0
        ax.set_xlim([max(0.01, min_oi), min(100, max_oi)])
        ax.set_ylim([max(0.1, min_perf), max_perf])
    elif baseline:
        baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
        ax.set_xlim([max(0.01, baseline_oi * 0.5), min(100, baseline_oi * 2.0)])
        ax.set_ylim([max(0.1, baseline.performance_gflops * 0.5), baseline.performance_gflops * 2.0])
    elif e2e:
        e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
        ax.set_xlim([max(0.01, e2e_oi * 0.5), min(100, e2e_oi * 2.0)])
        ax.set_ylim([max(0.1, e2e.performance_gflops * 0.5), e2e.performance_gflops * 2.0])

def plot_roofline_grid(log_data: Dict[int, Dict[str, LogData]]):
    """Plot all 4 roofline plots in a 2x2 grid layout (landscape)"""
    datasets = [1, 2, 3, 4]
    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))  
    
    # Plot each dataset in its position
    # Top row: Dataset 1 (left), Dataset 2 (right)
    # Bottom row: Dataset 3 (left), Dataset 4 (right)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, dataset in enumerate(datasets):
        row, col = positions[idx]
        ax = axes[row, col]
        plot_single_roofline_on_axis(ax, dataset, log_data, show_legend=(idx == 0))
    
    # Add overall title
    fig.suptitle('Roofline Models: All Datasets Comparison\n(Baseline vs AHAS)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/roofline_grid.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/roofline_grid.png")
    plt.close()

def plot_aggregated_roofline(log_data: Dict[int, Dict[str, LogData]]):
    """Plot aggregated roofline model with all datasets combined"""
    datasets = [1, 2, 3, 4]
    colors = {'baseline': ['#2E86AB', '#06A77D', '#F18F01', '#BC4749'],
              'e2e': ['#A23B72', '#6A994E', '#D4A5A5', '#8B5A3C']}
    markers = {'baseline': 'o', 'e2e': 's'}
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Collect all data points
    all_baseline_oi = []
    all_baseline_perf = []
    all_e2e_oi = []
    all_e2e_perf = []
    
    # Compute operational intensity range for roofline curves
    op_intensity_range = np.logspace(-2, 2, 1000)  # 0.01 to 100 FLOPs/Byte
    
    # Collect roofline parameters from all datasets
    cpu_mem_bandwidths = []
    cpu_peak_computes = []
    gpu_mem_bandwidths = []
    gpu_peak_computes = []
    
    for idx, dataset in enumerate(datasets):
        if dataset not in log_data:
            continue
        
        baseline = log_data[dataset].get('baseline')
        e2e = log_data[dataset].get('e2e')
        
        if baseline:
            baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
            all_baseline_oi.append(baseline_oi)
            all_baseline_perf.append(baseline.performance_gflops)
            
            # Collect CPU roofline parameters
            cpu_mem_bw = baseline.performance_gbps if baseline.performance_gbps > 0 else baseline.performance_gflops / baseline_oi if baseline_oi > 0 else 1.0
            cpu_peak = baseline.performance_gflops * 1.5
            cpu_mem_bandwidths.append(cpu_mem_bw)
            cpu_peak_computes.append(cpu_peak)
            
            # Plot baseline point
            ax.scatter(baseline_oi, baseline.performance_gflops,
                      s=200, marker=markers['baseline'], color=colors['baseline'][idx], 
                      edgecolors='black', linewidth=2,
                      label=f'D{dataset} Baseline', zorder=5, alpha=0.8)
        
        if e2e:
            e2e_oi = compute_operational_intensity(e2e.flops, e2e.bytes)
            all_e2e_oi.append(e2e_oi)
            all_e2e_perf.append(e2e.performance_gflops)
            
            # Collect GPU roofline parameters
            gpu_mem_bw = e2e.performance_gbps if e2e.performance_gbps > 0 else e2e.performance_gflops / e2e_oi if e2e_oi > 0 else 1.0
            gpu_peak = e2e.performance_gflops * 2.0
            gpu_mem_bandwidths.append(gpu_mem_bw)
            gpu_peak_computes.append(gpu_peak)
            
            # Plot ahas point
            ax.scatter(e2e_oi, e2e.performance_gflops,
                      s=200, marker=markers['e2e'], color=colors['e2e'][idx],
                      edgecolors='black', linewidth=2,
                      label=f'D{dataset} AHAS', zorder=5, alpha=0.8)
    
    # Compute average roofline parameters
    if cpu_mem_bandwidths and cpu_peak_computes:
        avg_cpu_mem_bw = np.mean(cpu_mem_bandwidths)
        avg_cpu_peak = np.mean(cpu_peak_computes)
        
        cpu_memory_bound = op_intensity_range * avg_cpu_mem_bw
        cpu_compute_bound = np.full_like(op_intensity_range, avg_cpu_peak)
        cpu_roofline = np.minimum(cpu_compute_bound, cpu_memory_bound)
        
        ax.loglog(op_intensity_range, cpu_roofline, 'b--', linewidth=3, 
                 label=f'CPU Roofline (avg: {avg_cpu_peak:.1f} GFLOP/s, {avg_cpu_mem_bw:.1f} GB/s)', 
                 alpha=0.7, zorder=1)
    
    if gpu_mem_bandwidths and gpu_peak_computes:
        avg_gpu_mem_bw = np.mean(gpu_mem_bandwidths)
        avg_gpu_peak = np.mean(gpu_peak_computes)
        
        gpu_memory_bound = op_intensity_range * avg_gpu_mem_bw
        gpu_compute_bound = np.full_like(op_intensity_range, avg_gpu_peak)
        gpu_roofline = np.minimum(gpu_compute_bound, gpu_memory_bound)
        
        ax.loglog(op_intensity_range, gpu_roofline, 'r--', linewidth=3,
                 label=f'GPU Roofline (avg: {avg_gpu_peak:.1f} GFLOP/s, {avg_gpu_mem_bw:.1f} GB/s)',
                 alpha=0.7, zorder=1)
    
    ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=13, fontweight='bold')
    ax.set_title('Aggregated Roofline Model: All Datasets\n(Baseline vs AHAS)', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set axis limits based on all data
    all_oi = all_baseline_oi + all_e2e_oi
    all_perf = all_baseline_perf + all_e2e_perf
    
    if all_oi and all_perf:
        min_oi = min(all_oi) * 0.5
        max_oi = max(all_oi) * 2.0
        min_perf = min(all_perf) * 0.5
        max_perf = max(all_perf) * 2.0
        ax.set_xlim([max(0.01, min_oi), min(100, max_oi)])
        ax.set_ylim([max(0.1, min_perf), max_perf])
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/roofline_aggregated.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/roofline_aggregated.png")
    plt.close()

def generate_architectural_summary(log_data: Dict[int, Dict[str, LogData]], 
                                   y_data: Dict[int, YReductionData]):
    """Generate architectural summary document"""
    summary = []
    summary.append("# Architectural Summary: PIM-Side Reduction Impact on Operational Intensity\n")
    summary.append("## Overview\n")
    summary.append("This analysis demonstrates how PIM-side filtering increases operational intensity ")
    summary.append("and shifts the workload from memory-bound toward compute-bound regime.\n\n")
    
    summary.append("## Key Findings\n\n")
    
    # Compute intensity changes
    for dataset in [1, 2, 3, 4]:
        if dataset not in log_data:
            continue
        
        baseline = log_data[dataset].get('baseline')
        filtered = log_data[dataset].get('tilepredpermspmm')
        
        if baseline and filtered:
            baseline_oi = compute_operational_intensity(baseline.flops, baseline.bytes)
            filtered_oi = compute_operational_intensity(filtered.flops, filtered.bytes)
            oi_improvement = (filtered_oi / baseline_oi - 1) * 100 if baseline_oi > 0 else 0
            
            nnz_reduction = (1 - filtered.nnz_X / baseline.nnz_X) * 100 if baseline.nnz_X > 0 else 0
            bytes_reduction = (1 - filtered.bytes / baseline.bytes) * 100 if baseline.bytes > 0 else 0
            flops_reduction = (1 - filtered.flops / baseline.flops) * 100 if baseline.flops > 0 else 0
            
            summary.append(f"### Dataset {dataset}\n\n")
            summary.append(f"- **NNZ Reduction**: {nnz_reduction:.1f}% ({baseline.nnz_X/1e6:.2f}M -> {filtered.nnz_X/1e6:.2f}M)\n")
            summary.append(f"- **Memory Traffic Reduction**: {bytes_reduction:.1f}% ({baseline.bytes/1e9:.3f} GB -> {filtered.bytes/1e9:.3f} GB)\n")
            summary.append(f"- **FLOPs Reduction**: {flops_reduction:.1f}% ({baseline.flops/1e9:.2f} GFLOPs -> {filtered.flops/1e9:.2f} GFLOPs)\n")
            summary.append(f"- **Operational Intensity**: {baseline_oi:.3f} -> {filtered_oi:.3f} FLOPs/Byte ({oi_improvement:+.1f}%)\n")
            summary.append(f"- **Regime**: ")
            
            # Classify regime
            if baseline_oi < 1.0:
                summary.append("Memory-bound -> ")
            else:
                summary.append("Compute-bound -> ")
            
            if filtered_oi < 1.0:
                summary.append("Memory-bound\n\n")
            else:
                summary.append("Compute-bound\n\n")
    
    summary.append("## Impact of PIM Filtering\n\n")
    summary.append("1. **Reduced Memory Traffic**: Fewer non-zero elements (nnz) directly reduces ")
    summary.append("the number of bytes that need to be transferred from memory.\n\n")
    summary.append("2. **Reduced FLOPs**: With fewer nnz, the computational work decreases proportionally.\n\n")
    summary.append("3. **Increased Operational Intensity**: The ratio FLOPs/Bytes improves because:\n")
    summary.append("   - Bytes decrease more than FLOPs (due to reduced data movement overhead)\n")
    summary.append("   - This shifts workloads toward the compute-bound regime\n\n")
    summary.append("4. **Performance Implications**: Higher operational intensity means the workload ")
    summary.append("is less limited by memory bandwidth and can better utilize compute resources.\n\n")
    
    with open('plots/architectural_summary.md', 'w', encoding='utf-8') as f:
        f.write(''.join(summary))
    print("Saved: plots/architectural_summary.md")

def main():
    """Main execution function"""
    print("Starting Roofline Analysis...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print("Loading log files...")
    log_data = load_all_logs('logs')
    
    print("Loading Y reduction data...")
    y_data = load_y_reductions('analysis/meta')
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 1: X reduction
    print("  - X matrix reduction plot...")
    plot_x_reduction(log_data)
    
    # Plot 2: Y reduction
    print("  - Y matrix reduction plot...")
    plot_y_reduction(y_data)
    
    # Plot 2.5: Combined X and Y reduction
    print("  - Combined X and Y reduction plot...")
    plot_combined_x_y_reduction(log_data, y_data)
    
    # Plot 3: PIM Operational Intensity (Baseline vs PIM)
    print("  - PIM operational intensity bar graph...")
    plot_pim_operational_intensity(log_data)
    
    # Plot 4-7: Roofline plots for each dataset
    for dataset in [1, 2, 3, 4]:
        print(f"  - Roofline plot for dataset {dataset}...")
        plot_roofline(dataset, log_data)
    
    # Plot 8: Aggregated roofline (all datasets)
    print("  - Aggregated roofline plot (all datasets)...")
    plot_aggregated_roofline(log_data)
    
    # Plot 9: Roofline grid (2x2 layout)
    print("  - Roofline grid plot (2x2 layout)...")
    plot_roofline_grid(log_data)
    
    # Generate summary
    print("\nGenerating architectural summary...")
    generate_architectural_summary(log_data, y_data)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()

