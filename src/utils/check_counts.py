### Libraries ###
import os
import sys
from collections import defaultdict

### Counting Function ###
def check_file_counts(root_dir=".", method_filter=None):

    ## 1. Walk and Count ##
    stats = defaultdict(lambda: defaultdict(int))
    all_methods_found = set()
    max_found = 0    

    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        
        # Look for folders like "M1", "M2", "M10"
        if folder_name.startswith("M") and folder_name[1:].isdigit():
            # Parent is the dataset (e.g., "monk1" or "monk1_05")
            parent_dir = os.path.basename(os.path.dirname(root))
            
            # Count valid result files
            pkl_count = sum(1 for f in files if f.endswith(".pkl") and "S" in f)
            
            if pkl_count > 0:
                stats[parent_dir][folder_name] = pkl_count
                all_methods_found.add(folder_name)
                if pkl_count > max_found:
                    max_found = pkl_count

    ## 2. Handle Empty Case ##
    if not stats:
        print(f"No results found in {os.path.abspath(root_dir)}")
        return

    ## 3. Sort Methods Numerically (M1, M2, ..., M10) ##
    methods = sorted(list(all_methods_found), key=lambda x: int(x[1:]))
    datasets = sorted(stats.keys())
    
    # Apply filter
    if method_filter:
        methods = [m for m in methods if m in method_filter]

    ## 4. Print Dynamic Table ##
    print(f"\nTarget Count (Detected): {max_found}")
    
    # Dynamic Header
    header = f"{'DATASET':<30} | "
    header += " | ".join([f"{m:<6}" for m in methods])
    
    print(header)
    print("-" * len(header))

    # Dynamic Rows
    missing_info = []
    for ds in datasets:
        row_str = f"{ds:<30} | "
        all_complete = True
        for m in methods:
            c = stats[ds].get(m, 0)
            
            if c == max_found:
                val = f"{c}"       
            elif c == 0:
                val = "-"
                all_complete = False
                missing_info.append((ds, m, list(range(max_found))))
            else:
                val = f"{c}/{max_found}"
                all_complete = False
                # Find which seeds are missing
                method_dir = os.path.join(root_dir, ds, m) if len(datasets) > 1 else os.path.join(root_dir, m)
                found_seeds = set()
                if os.path.isdir(method_dir):
                    for f in os.listdir(method_dir):
                        if f.endswith(".pkl") and "S" in f:
                            try:
                                seed = int(f.split("S")[1].split(".")[0])
                                found_seeds.add(seed)
                            except ValueError:
                                pass
                missing_seeds = [s for s in range(max_found) if s not in found_seeds]
                missing_info.append((ds, m, missing_seeds))
            
            row_str += f"{val:<6} | "
        row_str += "✅" if all_complete else "❌"
        print(row_str)

    # Missing Seeds Report
    if missing_info:
        print(f"\n📋 Missing Seeds:")
        for ds, m, seeds in missing_info:
            print(f"  {ds} / {m}: seeds {seeds}")

### Main ###
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default=".")
    parser.add_argument("--methods", "-m", nargs="+", help="Only show these methods (e.g., M1 M3 M5)")
    args = parser.parse_args()
    check_file_counts(args.dir, method_filter=args.methods)