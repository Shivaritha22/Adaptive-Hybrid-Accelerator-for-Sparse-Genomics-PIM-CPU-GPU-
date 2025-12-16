import h5py
import numpy as np
import argparse
import sys

def compare_h5_files(file1_path, file2_path, tolerance_min=1e-3, tolerance_max=1e-5):
    """
    Compare two H5 files and print mismatches in matrix values.
    
    Args:
        file1_path: Path to first H5 file (cuda.h5)
        file2_path: Path to second H5 file (check.h5)
        tolerance_min: Minimum tolerance (0.001)
        tolerance_max: Maximum tolerance (0.00001)
    """
    mismatch_count = 0
    
    try:
        with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
            # Get all dataset names from both files
            datasets1 = []
            datasets2 = []
            
            def collect_datasets1(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets1.append(name)
            
            def collect_datasets2(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets2.append(name)
            
            f1.visititems(collect_datasets1)
            f2.visititems(collect_datasets2)
            
            # Use datasets from file1 as reference
            datasets = datasets1
            
            if not datasets:
                print("No datasets found in the files.")
                return
            
            # Compare each dataset
            for dataset_name in datasets:
                if dataset_name not in datasets2:
                    print(f"Warning: Dataset '{dataset_name}' not found in {file2_path}")
                    continue
                
                data1 = f1[dataset_name][:]
                data2 = f2[dataset_name][:]
                
                # Check if shapes match
                if data1.shape != data2.shape:
                    print(f"Shape mismatch for '{dataset_name}': {data1.shape} vs {data2.shape}")
                    continue
                
                # Flatten arrays for comparison
                flat1 = data1.flatten()
                flat2 = data2.flatten()
                
                # Compare element by element
                for idx in range(len(flat1)):
                    val1 = flat1[idx]
                    val2 = flat2[idx]
                    
                    # Check if values are close within tolerance range
                    # Use relative tolerance: atol = tolerance_max, rtol = tolerance_min
                    if not np.isclose(val1, val2, atol=tolerance_max, rtol=tolerance_min):
                        print(f"{val1}     {val2}")
                        mismatch_count += 1
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"\nTotal number of mismatches: {mismatch_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two H5 files and print mismatches in matrix values.')
    parser.add_argument('file1', help='First H5 file (e.g., cuda.h5)')
    parser.add_argument('file2', help='Second H5 file (e.g., check.h5)')
    parser.add_argument('--tolerance-min', type=float, default=0.001, 
                        help='Minimum tolerance (default: 0.001)')
    parser.add_argument('--tolerance-max', type=float, default=0.00001, 
                        help='Maximum tolerance (default: 0.00001)')
    
    args = parser.parse_args()
    
    compare_h5_files(args.file1, args.file2, tolerance_min=args.tolerance_min, tolerance_max=args.tolerance_max)

