#!/usr/bin/env python3
import argparse
import os
import sys
import re

# allow import from ../scripts if needed
HERE = os.path.dirname(__file__)
SCRIPTS = os.path.abspath(os.path.join(HERE, 'scripts'))
sys.path.insert(0, SCRIPTS)

import pandas as pd
from make_multimodel_pdb_from_dir_addH import generate_frames_from_pdb_dir
from find_backbone_neighbors import extract_backbone_neighbors_over_trajectory
from transform_neighbour_features import transform_neighbor_features
from flatten_neighbors import flatten_neighbor_rows
from aromatic_features import extract_aromatic_ring_features
from dssp_features_individual_withSS_fromPDBdir import compute_dssp_stats_framewise
from hbond_top3_features import compute_top3_hbond_features, build_hbond_defs


def process_ensemble(ensemble_dir, output_path, k_nn=20, k_arom=3):
    """
    Run full feature pipeline on a single ensemble directory, saving to output_path.
    """

    # 1. Generate hydrogenated frames (list so we can reuse)
    print("Loading PDB files into a trajectory...")
    frames = list(generate_frames_from_pdb_dir(ensemble_dir))

    # 2. Backbone neighbor extraction
    print("Extracting backbone neighbors over trajectory...")
    df_nn = extract_backbone_neighbors_over_trajectory(frames, k=k_nn)

    # 3. Transform & one-hot encode neighbors
    print("Transforming neighbor features...")
    df_t = transform_neighbor_features(df_nn)
    # 4. Flatten neighbor rows
    df_flat = flatten_neighbor_rows(df_t, k=k_nn)

    # 5) compute dssp stats
    print("Computing DSSP stats...")
    dssp_stats = compute_dssp_stats_framewise(pdb_dir=ensemble_dir)
    dssp_stats = dssp_stats.rename(columns={'residue':'resnum_h'}) # renames the residue column to allow easier integration of the dataframe.
    df_with_dssp = df_flat.merge(dssp_stats, on=['frame','resnum_h'], how='left')

    # 6) aromatic features – reusing same list of frames
    print("Computing aromatic features stats...")
    df_arom = extract_aromatic_ring_features(frames_generator=frames, k=k_arom)
    df_arom_drop = df_arom.drop(columns=[f'ring{i}_resnum' for i in range(1, k_arom+1)], errors='ignore')

    # 7) merge aromatic features into existing feature data frame
    df_with_arom = df_with_dssp.merge(df_arom_drop, on=['frame','resnum_h'], how='left')

    # 8) adding h-bond specific features
    print("Adding h-bonding features...")
    hbond_defs = build_hbond_defs(frames)
    hbonds = compute_top3_hbond_features(frames, hbond_defs)
    df_final = df_with_arom.merge(
        hbonds,
        left_on=['frame','resnum_h'],
        right_on=['frame','residue'],
        how='left'
    ).drop(columns='residue')

    # we don't need the direction vector to the first nearest neighbor, nor its atom type
    to_drop = [c for c in df_final.columns if re.fullmatch(r'atype_[A-Za-z0-9]+_nn1', c)]
    to_drop.extend(["rx_nn1","ry_nn1","rz_nn1"])
    df_final.drop(columns=to_drop, inplace=True)

    # Save
    df_final.to_csv(output_path, index=False)
    return len(df_final)


def main():
    parser = argparse.ArgumentParser(description="Batch feature extraction over BioEmu ensembles.")
    parser.add_argument('--input-dir', required=True,
                        help="Path to bioemu_outputs directory containing ensemble subdirs")
    parser.add_argument('--output-dir', required=True,
                        help="Directory in which to save per-ensemble CSVs")
    parser.add_argument('-k', type=int, default=20, help="Neighbors per H (backbone)")
    parser.add_argument('--karom', type=int, default=3, help="Nearest aromatic rings per H")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over subdirectories
    for name in sorted(os.listdir(args.input_dir)):
        ens_dir = os.path.join(args.input_dir, name)
        if not os.path.isdir(ens_dir):
            continue
        out_file = os.path.join(args.output_dir, f"{name}.csv")
        if os.path.exists(out_file):
            print(out_file + " already exists, skipping.")
            continue
        try:
            n = process_ensemble(ens_dir, out_file, k_nn=args.k, k_arom=args.karom)
            print(f"{name}: wrote {n} rows to {out_file}")
        except Exception as e:
            print(f"Error processing {name}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
