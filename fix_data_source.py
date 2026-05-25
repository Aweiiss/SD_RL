#!/usr/bin/env python3
"""Fix data_source in parquet files for veRL compatibility."""

import pandas as pd
import sys

def fix_parquet(path, new_source='math'):
    """Change data_source column to a veRL-supported type."""
    print(f"Processing: {path}")
    df = pd.read_parquet(path)
    
    if 'data_source' not in df.columns:
        print(f"  Warning: no 'data_source' column, skipping")
        return
    
    old_sources = df['data_source'].unique()
    print(f"  Before: data_source = {old_sources}")
    
    df['data_source'] = new_source
    df.to_parquet(path, index=False)
    
    new_sources = df['data_source'].unique()
    print(f"  After:  data_source = {new_sources}")
    print(f"  Saved:  {path}")

if __name__ == "__main__":
    # Fix both train and test files
    files = [
        "/home/lingquh1xx/L2598/Temp/verl/data/deepmath/train_sample_6144.parquet",
        "/home/lingquh1xx/L2598/Temp/verl/data/deepmath/test.parquet",
    ]
    
    for f in files:
        try:
            fix_parquet(f, new_source='math')
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
    
    print("\nAll files fixed successfully!")
