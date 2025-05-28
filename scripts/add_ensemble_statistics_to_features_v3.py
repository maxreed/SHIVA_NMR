#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd

def process_file(inpath, outpath):
    df = pd.read_csv(inpath)
    # identify the grouping and target columns
    group_col = "resnum_h"
    # # 1. distance_nn1
    # cols = ["distance_nn1"]
    # # 2. any column ending in _nn2, _nn3, _nn4, or _nn5
    # cols += df.filter(regex=r"_nn[2-5]$").columns.tolist()
    # 3. any column starting with sin_ or cos_
    cols = df.filter(regex=r"^(ss_)").columns.tolist()
    # # 4. any column starting with ring
    # cols += df.filter(regex=r"^ring").columns.tolist()
    # make unique
    cols = sorted(set(cols))

    # for each target col, compute and append mean and std per group
    for c in cols:
        mean_col = f"{c}_mean"
        std_col  = f"{c}_std"
        df[mean_col] = df.groupby(group_col)[c].transform("mean")

    # ensure output dir exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)

def main(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    for infile in glob.glob(os.path.join(input_dir, "*.csv")):
        fname = os.path.basename(infile)
        outfile = os.path.join(output_dir, fname)
        print(f"Processing {fname} → {outfile}")
        process_file(infile, outfile)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_ensemble_statistics_to_features.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
