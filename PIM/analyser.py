import os
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime

def load_file(filepath):
    """Load file based on its extension"""
    if not os.path.exists(filepath):
        return None
    
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.h5' or ext == '.hdf5':
            with h5py.File(filepath, 'r') as f:
                # Try to get all datasets
                data = {}
                def get_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = np.array(obj)
                f.visititems(get_datasets)
                return data if data else np.array(f[list(f.keys())[0]])
        elif ext == '.npy':
            return np.load(filepath)
        elif ext == '.npz':
            return np.load(filepath)
        elif ext == '.csv':
            return np.loadtxt(filepath, delimiter=',')
        elif ext == '.txt':
            # Try to load as numpy array first
            try:
                return np.loadtxt(filepath)
            except:
                # If that fails, read as text
                with open(filepath, 'r') as f:
                    return f.read()
        elif ext == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            # Try numpy load as default
            try:
                return np.load(filepath)
            except:
                with open(filepath, 'r') as f:
                    return f.read()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_file_size(filepath):
    """Get file size in bytes"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def calculate_data_reduction(original_data, filtered_data):
    """Calculate data reduction metrics"""
    if original_data is None or filtered_data is None:
        return None, None, None
    
    # Handle different data types
    if isinstance(original_data, dict):
        # For h5 files with multiple datasets
        orig_size = sum(np.size(v) for v in original_data.values())
        filt_size = sum(np.size(v) for v in filtered_data.values())
        orig_bytes = sum(v.nbytes if isinstance(v, np.ndarray) else 0 for v in original_data.values())
        filt_bytes = sum(v.nbytes if isinstance(v, np.ndarray) else 0 for v in filtered_data.values())
    elif isinstance(original_data, np.ndarray):
        orig_size = original_data.size
        filt_size = filtered_data.size
        orig_bytes = original_data.nbytes
        filt_bytes = filtered_data.nbytes
    elif isinstance(original_data, (list, tuple)):
        orig_size = len(original_data)
        filt_size = len(filtered_data)
        orig_bytes = orig_size * 8  # Approximate
        filt_bytes = filt_size * 8
    else:
        # For text or other types, use string length
        orig_size = len(str(original_data))
        filt_size = len(str(filtered_data))
        orig_bytes = orig_size
        filt_bytes = filt_size
    
    reduction = orig_size - filt_size
    reduction_bytes = orig_bytes - filt_bytes
    ratio = filt_size / orig_size if orig_size > 0 else 0
    
    return reduction, reduction_bytes, ratio

def calculate_data_movement_reduction(original_data, filtered_data):
    """Calculate data movement reduction (difference in values)"""
    if original_data is None or filtered_data is None:
        return None
    
    try:
        if isinstance(original_data, dict) and isinstance(filtered_data, dict):
            # For h5 files, calculate movement for each dataset
            total_movement = 0
            for key in original_data.keys():
                if key in filtered_data:
                    orig = original_data[key]
                    filt = filtered_data[key]
                    if isinstance(orig, np.ndarray) and isinstance(filt, np.ndarray):
                        if orig.shape == filt.shape:
                            movement = np.sum(np.abs(orig - filt))
                            total_movement += movement
            return total_movement
        elif isinstance(original_data, np.ndarray) and isinstance(filtered_data, np.ndarray):
            if original_data.shape == filtered_data.shape:
                return np.sum(np.abs(original_data - filtered_data))
            else:
                # Different shapes - calculate based on overlapping indices
                min_size = min(original_data.size, filtered_data.size)
                orig_flat = original_data.flatten()[:min_size]
                filt_flat = filtered_data.flatten()[:min_size]
                return np.sum(np.abs(orig_flat - filt_flat))
    except Exception as e:
        print(f"Error calculating data movement: {e}")
        return None
    
    return None

def find_annotation_files(base_path, annotation_type, annotation_num):
    """Find annotation files in X or Y subdirectories with same filename"""
    # Both original and filtered use the same filename (no _pim suffix)
    possible_names = [
        f"{annotation_type}{annotation_num}.h5",
        f"{annotation_type}{annotation_num}.hdf5",
        f"{annotation_type}_{annotation_num}.h5",
        f"{annotation_type}_{annotation_num}.hdf5",
    ]
    
    # Check in X and Y subdirectories specifically
    for subdir in ['X', 'Y']:
        subdir_path = os.path.join(base_path, subdir)
        if os.path.exists(subdir_path):
            for name in possible_names:
                filepath = os.path.join(subdir_path, name)
                if os.path.exists(filepath):
                    return filepath
    
    # Also check in base path and other subdirectories as fallback
    for name in possible_names:
        filepath = os.path.join(base_path, name)
        if os.path.exists(filepath):
            return filepath
    
    # Check in all subdirectories as fallback
    for root, dirs, files in os.walk(base_path):
        for name in possible_names:
            filepath = os.path.join(root, name)
            if os.path.exists(filepath):
                return filepath
    
    return None

def compare_annotation(annotation_type, annotation_num, original_path, filtered_path, log_file):
    """Compare annotation between original and filtered folders"""
    print(f"\n{'='*60}")
    print(f"Comparing {annotation_type}{annotation_num}")
    print(f"{'='*60}")
    
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Comparing {annotation_type}{annotation_num}\n")
    log_file.write(f"{'='*60}\n")
    
    # Find files - same filename in original and filtered folders (in X or Y subdirectories)
    original_file = find_annotation_files(original_path, annotation_type, annotation_num)
    filtered_file = find_annotation_files(filtered_path, annotation_type, annotation_num)
    
    if not original_file:
        msg = f"Original file for {annotation_type}{annotation_num} not found in {original_path}"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")
        return
    
    if not filtered_file:
        msg = f"Filtered file for {annotation_type}{annotation_num} not found in {filtered_path}"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")
        return
    
    print(f"Original file: {original_file}")
    print(f"Filtered file: {filtered_file}")
    log_file.write(f"Original file: {original_file}\n")
    log_file.write(f"Filtered file: {filtered_file}\n")
    
    # Load files
    original_data = load_file(original_file)
    filtered_data = load_file(filtered_file)
    
    if original_data is None:
        msg = f"Could not load original file: {original_file}"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")
        return
    
    if filtered_data is None:
        msg = f"Could not load filtered file: {filtered_file}"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")
        return
    
    # Get file sizes
    original_size = get_file_size(original_file)
    filtered_size = get_file_size(filtered_file)
    size_reduction = original_size - filtered_size
    size_reduction_percent = (size_reduction / original_size * 100) if original_size > 0 else 0
    
    print(f"\nFile Size Comparison:")
    print(f"  Original file size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
    print(f"  Filtered file size: {filtered_size:,} bytes ({filtered_size/1024:.2f} KB)")
    print(f"  Size reduction: {size_reduction:,} bytes ({size_reduction/1024:.2f} KB, {size_reduction_percent:.2f}%)")
    
    log_file.write(f"\nFile Size Comparison:\n")
    log_file.write(f"  Original file size: {original_size:,} bytes ({original_size/1024:.2f} KB)\n")
    log_file.write(f"  Filtered file size: {filtered_size:,} bytes ({filtered_size/1024:.2f} KB)\n")
    log_file.write(f"  Size reduction: {size_reduction:,} bytes ({size_reduction/1024:.2f} KB, {size_reduction_percent:.2f}%)\n")
    
    # Calculate data reduction
    reduction, reduction_bytes, ratio = calculate_data_reduction(original_data, filtered_data)
    
    if reduction is not None:
        print(f"\nData Reduction:")
        print(f"  Data points reduced: {reduction:,}")
        print(f"  Data size reduced: {reduction_bytes:,} bytes ({reduction_bytes/1024:.2f} KB)")
        print(f"  Reduction ratio: {ratio:.4f} ({ratio*100:.2f}% of original)")
        print(f"  Reduction percentage: {(1-ratio)*100:.2f}%")
        
        log_file.write(f"\nData Reduction:\n")
        log_file.write(f"  Data points reduced: {reduction:,}\n")
        log_file.write(f"  Data size reduced: {reduction_bytes:,} bytes ({reduction_bytes/1024:.2f} KB)\n")
        log_file.write(f"  Reduction ratio: {ratio:.4f} ({ratio*100:.2f}% of original)\n")
        log_file.write(f"  Reduction percentage: {(1-ratio)*100:.2f}%\n")
    
    # Calculate data movement reduction
    movement_reduction = calculate_data_movement_reduction(original_data, filtered_data)
    
    if movement_reduction is not None:
        print(f"\nData Movement Reduction:")
        print(f"  Total absolute difference: {movement_reduction:.4f}")
        print(f"  Average absolute difference per element: {movement_reduction / (filtered_data.size if isinstance(filtered_data, np.ndarray) else len(str(filtered_data))):.6f}")
        
        log_file.write(f"\nData Movement Reduction:\n")
        log_file.write(f"  Total absolute difference: {movement_reduction:.4f}\n")
        if isinstance(filtered_data, np.ndarray):
            log_file.write(f"  Average absolute difference per element: {movement_reduction / filtered_data.size:.6f}\n")
        elif isinstance(filtered_data, dict):
            total_elements = sum(v.size for v in filtered_data.values() if isinstance(v, np.ndarray))
            if total_elements > 0:
                log_file.write(f"  Average absolute difference per element: {movement_reduction / total_elements:.6f}\n")
    
    log_file.flush()

def read_meta_file(annotation_num, base_path, log_file):
    """Read meta file and extract rare gene information"""
    print(f"\n{'='*60}")
    print(f"Reading Meta File for annotation {annotation_num}")
    print(f"{'='*60}")
    
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Reading Meta File for annotation {annotation_num}\n")
    log_file.write(f"{'='*60}\n")
    
    # Try to find meta file with different naming patterns
    meta_patterns = [
        f"{annotation_num}_meta.txt",
        f"d{annotation_num}_meta.txt",
        f"*{annotation_num}*meta*.txt",
    ]
    
    meta_file = None
    for pattern in meta_patterns:
        # Search in base directory
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == f"{annotation_num}_meta.txt" or (str(annotation_num) in file and "meta" in file.lower() and file.endswith('.txt')):
                    meta_file = os.path.join(root, file)
                    break
            if meta_file:
                break
        if meta_file:
            break
    
    if not meta_file or not os.path.exists(meta_file):
        msg = f"Meta file for annotation {annotation_num} not found"
        print(f"INFO: {msg}")
        log_file.write(f"INFO: {msg}\n")
        return
    
    print(f"Meta file found: {meta_file}")
    log_file.write(f"Meta file: {meta_file}\n")
    
    try:
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        
        rare_genes = None
        noise_genes = None
        housekeep_genes = None
        variance_stats = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("rare gene:"):
                try:
                    rare_genes = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("noise gene:"):
                try:
                    noise_genes = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("common housekeep:"):
                try:
                    housekeep_genes = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("variance min:"):
                try:
                    variance_stats['min'] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("variance max:"):
                try:
                    variance_stats['max'] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("variance mean:"):
                try:
                    variance_stats['mean'] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("genes with zero variance:"):
                try:
                    variance_stats['zero_variance'] = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("unique variance values:"):
                try:
                    variance_stats['unique_values'] = int(line.split(":")[1].strip())
                except:
                    pass
        
        print(f"\nRare Gene Information:")
        log_file.write(f"\nRare Gene Information:\n")
        
        if rare_genes is not None:
            print(f"  Rare genes retained: {rare_genes:,}")
            log_file.write(f"  Rare genes retained: {rare_genes:,}\n")
        
        if noise_genes is not None:
            print(f"  Noise genes: {noise_genes:,}")
            log_file.write(f"  Noise genes: {noise_genes:,}\n")
        
        if housekeep_genes is not None:
            print(f"  Housekeeping genes: {housekeep_genes:,}")
            log_file.write(f"  Housekeeping genes: {housekeep_genes:,}\n")
        
        if variance_stats:
            print(f"\nVariance Statistics:")
            log_file.write(f"\nVariance Statistics:\n")
            if 'min' in variance_stats:
                print(f"  Min variance: {variance_stats['min']:.6f}")
                log_file.write(f"  Min variance: {variance_stats['min']:.6f}\n")
            if 'max' in variance_stats:
                print(f"  Max variance: {variance_stats['max']:.6f}")
                log_file.write(f"  Max variance: {variance_stats['max']:.6f}\n")
            if 'mean' in variance_stats:
                print(f"  Mean variance: {variance_stats['mean']:.6f}")
                log_file.write(f"  Mean variance: {variance_stats['mean']:.6f}\n")
            if 'zero_variance' in variance_stats:
                print(f"  Genes with zero variance: {variance_stats['zero_variance']:,}")
                log_file.write(f"  Genes with zero variance: {variance_stats['zero_variance']:,}\n")
            if 'unique_values' in variance_stats:
                print(f"  Unique variance values: {variance_stats['unique_values']:,}")
                log_file.write(f"  Unique variance values: {variance_stats['unique_values']:,}\n")
        
        log_file.flush()
        
    except Exception as e:
        msg = f"Error reading meta file: {e}"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")

def compare_h5_files(annotation_type, annotation_num, original_path, filtered_path, log_file):
    """Compare h5 files between original and filtered folders"""
    print(f"\n{'='*60}")
    print(f"Comparing H5 files for {annotation_type}{annotation_num}")
    print(f"{'='*60}")
    
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Comparing H5 files for {annotation_type}{annotation_num}\n")
    log_file.write(f"{'='*60}\n")
    
    # Find specific h5 files - same filename in original and filtered folders
    original_file = find_annotation_files(original_path, annotation_type, annotation_num)
    filtered_file = find_annotation_files(filtered_path, annotation_type, annotation_num)
    
    if not original_file:
        msg = f"Original H5 file {annotation_type}{annotation_num}.h5 not found in original/X or original/Y"
        print(f"INFO: {msg}")
        log_file.write(f"INFO: {msg}\n")
        return
    
    if not filtered_file:
        msg = f"Filtered H5 file {annotation_type}{annotation_num}.h5 not found in filtered/X or filtered/Y"
        print(f"INFO: {msg}")
        log_file.write(f"INFO: {msg}\n")
        return
    
    print(f"\nAnalyzing: {original_file}")
    log_file.write(f"\nAnalyzing: {original_file}\n")
    print(f"Comparing with: {filtered_file}")
    log_file.write(f"Comparing with: {filtered_file}\n")
    
    # Load h5 files
    original_data = load_file(original_file)
    filtered_data = load_file(filtered_file)
    
    if original_data is None or filtered_data is None:
        msg = f"Could not load one or both files"
        print(f"ERROR: {msg}")
        log_file.write(f"ERROR: {msg}\n")
        return
    
    # Get file sizes
    orig_size = get_file_size(original_file)
    filt_size = get_file_size(filtered_file)
    size_reduction = orig_size - filt_size
    
    print(f"\nFile Size Comparison:")
    print(f"  Original size: {orig_size:,} bytes ({orig_size/1024:.2f} KB)")
    print(f"  Filtered size: {filt_size:,} bytes ({filt_size/1024:.2f} KB)")
    print(f"  Size reduction: {size_reduction:,} bytes ({size_reduction/1024:.2f} KB, {size_reduction/orig_size*100:.2f}%)")
    
    log_file.write(f"\nFile Size Comparison:\n")
    log_file.write(f"  Original size: {orig_size:,} bytes ({orig_size/1024:.2f} KB)\n")
    log_file.write(f"  Filtered size: {filt_size:,} bytes ({filt_size/1024:.2f} KB)\n")
    log_file.write(f"  Size reduction: {size_reduction:,} bytes ({size_reduction/1024:.2f} KB, {size_reduction/orig_size*100:.2f}%)\n")
    
    # Calculate data reduction
    reduction, reduction_bytes, ratio = calculate_data_reduction(original_data, filtered_data)
    
    if reduction is not None:
        print(f"\nData Reduction:")
        print(f"  Data points reduced: {reduction:,}")
        print(f"  Data size reduced: {reduction_bytes:,} bytes ({reduction_bytes/1024:.2f} KB)")
        print(f"  Reduction ratio: {ratio:.4f} ({ratio*100:.2f}% of original)")
        print(f"  Reduction percentage: {(1-ratio)*100:.2f}%")
        
        log_file.write(f"\nData Reduction:\n")
        log_file.write(f"  Data points reduced: {reduction:,}\n")
        log_file.write(f"  Data size reduced: {reduction_bytes:,} bytes ({reduction_bytes/1024:.2f} KB)\n")
        log_file.write(f"  Reduction ratio: {ratio:.4f} ({ratio*100:.2f}% of original)\n")
        log_file.write(f"  Reduction percentage: {(1-ratio)*100:.2f}%\n")
    
    # Calculate data movement reduction
    movement_reduction = calculate_data_movement_reduction(original_data, filtered_data)
    
    if movement_reduction is not None:
        print(f"\nData Movement Reduction:")
        print(f"  Total absolute difference: {movement_reduction:.4f}")
        if isinstance(filtered_data, np.ndarray):
            avg_diff = movement_reduction / filtered_data.size
            print(f"  Average absolute difference per element: {avg_diff:.6f}")
            log_file.write(f"\nData Movement Reduction:\n")
            log_file.write(f"  Total absolute difference: {movement_reduction:.4f}\n")
            log_file.write(f"  Average absolute difference per element: {avg_diff:.6f}\n")
        elif isinstance(filtered_data, dict):
            total_elements = sum(v.size for v in filtered_data.values() if isinstance(v, np.ndarray))
            if total_elements > 0:
                avg_diff = movement_reduction / total_elements
                print(f"  Average absolute difference per element: {avg_diff:.6f}")
                log_file.write(f"\nData Movement Reduction:\n")
                log_file.write(f"  Total absolute difference: {movement_reduction:.4f}\n")
                log_file.write(f"  Average absolute difference per element: {avg_diff:.6f}\n")
    
    # Compare datasets
    if isinstance(original_data, dict) and isinstance(filtered_data, dict):
        print(f"\nDataset Comparison:")
        log_file.write(f"\nDataset Comparison:\n")
        for key in set(list(original_data.keys()) + list(filtered_data.keys())):
            if key in original_data and key in filtered_data:
                orig_arr = original_data[key]
                filt_arr = filtered_data[key]
                if isinstance(orig_arr, np.ndarray) and isinstance(filt_arr, np.ndarray):
                    reduction = orig_arr.size - filt_arr.size
                    ratio = filt_arr.size / orig_arr.size if orig_arr.size > 0 else 0
                    print(f"  {key}: {orig_arr.shape} -> {filt_arr.shape}, reduction: {reduction:,} ({ratio*100:.2f}%)")
                    log_file.write(f"  {key}: {orig_arr.shape} -> {filt_arr.shape}, reduction: {reduction:,} ({ratio*100:.2f}%)\n")
    
    log_file.flush()

def parse_annotation_input(user_input):
    """Parse user input to extract annotation type and number
    Examples: 'd0' -> ('d', '0'), 'y5' -> ('y', '5'), 'd2.h5' -> ('d', '2')
    """
    user_input = user_input.strip().lower()
    
    # Remove .h5 extension if present
    if user_input.endswith('.h5') or user_input.endswith('.hdf5'):
        user_input = user_input[:-4] if user_input.endswith('.h5') else user_input[:-5]
    
    # Extract annotation type (d or y) and number
    if user_input.startswith('d'):
        annotation_type = 'd'
        annotation_num = user_input[1:]
    elif user_input.startswith('y'):
        annotation_type = 'y'
        annotation_num = user_input[1:]
    else:
        # Try to extract from format like "d0.h5 d0_pim.h5" or just number
        parts = user_input.split()
        if len(parts) > 0:
            first_part = parts[0]
            if first_part.startswith('d'):
                annotation_type = 'd'
                annotation_num = first_part[1:].replace('_pim', '').replace('.h5', '')
            elif first_part.startswith('y'):
                annotation_type = 'y'
                annotation_num = first_part[1:].replace('_pim', '').replace('.h5', '')
            else:
                # Assume it's just a number, default to 'd'
                annotation_type = 'd'
                annotation_num = first_part
        else:
            annotation_type = 'd'
            annotation_num = user_input
    
    return annotation_type, annotation_num

def main():
    """Main function"""
    # Get annotation from user (e.g., "d0", "y0", "d2", etc.)
    user_input = input("Enter annotation (e.g., d0, y0, d2, y5): ").strip()
    
    if not user_input:
        print("Error: Annotation cannot be empty")
        print("Examples: d0, y0, d2, y5")
        return
    
    # Parse input to get annotation type and number
    annotation_type, annotation_num = parse_annotation_input(user_input)
    
    print(f"\nParsed annotation: Type={annotation_type}, Number={annotation_num}")
    
    # Define paths
    base_path = Path(__file__).parent
    original_path = base_path / "original"
    filtered_path = base_path / "filtered"
    analysis_path = base_path / "analysis"
    
    # Create analysis directory if it doesn't exist
    analysis_path.mkdir(exist_ok=True)
    
    # Open log file - name based on input annotation
    log_file_path = analysis_path / f"{annotation_type}{annotation_num}_analyse.txt"
    
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'='*60}\n"
        header += f"Annotation Analysis - {annotation_type}{annotation_num}\n"
        header += f"Timestamp: {timestamp}\n"
        header += f"{'='*60}\n"
        
        print(header)
        log_file.write(header)
        
        # Compare the specific annotation type
        print(f"\n[1/3] Comparing {annotation_type}{annotation_num} annotations...")
        compare_annotation(annotation_type, annotation_num, str(original_path), str(filtered_path), log_file)
        
        # Read meta file for rare gene information (only for 'd' annotations)
        if annotation_type == 'd':
            print(f"\n[2/3] Reading meta file for rare gene information...")
            read_meta_file(annotation_num, str(base_path), log_file)
        else:
            print(f"\n[2/3] Skipping meta file (only available for 'd' annotations)")
            log_file.write(f"\nSkipping meta file (only available for 'd' annotations)\n")
        
        # Compare h5 files
        print(f"\n[3/3] Comparing H5 files ({annotation_type}{annotation_num}.h5 vs {annotation_type}{annotation_num}_pim.h5)...")
        compare_h5_files(annotation_type, annotation_num, str(original_path), str(filtered_path), log_file)
        
        # Write footer
        footer = f"\n{'='*60}\n"
        footer += f"Analysis completed at {timestamp}\n"
        footer += f"{'='*60}\n"
        
        print(footer)
        log_file.write(footer)
    
    print(f"\nAnalysis complete! Results saved to: {log_file_path}")

if __name__ == "__main__":
    main()

