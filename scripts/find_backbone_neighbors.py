#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial import cKDTree

# allow import from ../scripts if needed
HERE = os.path.dirname(__file__)
SCRIPTS = os.path.abspath(os.path.join(HERE, 'scripts'))
sys.path.insert(0, SCRIPTS)

# -----------------------------------------------------------------------------
# Hard‑coded Amber99SB atom‐type map: (residue_name → {atom_name: amber_type})
# Expand this dictionary for all amino acids you need.
# Hard‑coded Amber99SB atom‐type map for the 20 standard amino acids
AMBER99SB_TYPE_MAP = {
    'ALA': {
        'N':'N',   'H':'H',    'CA':'CT', 'HA':'H1',
        'CB':'CT', 'HB1':'H1', 'HB2':'H1','HB3':'H1',
        'C':'C',   'O':'O'
    },
    'ARG': {
        'N':'N', 'H':'H', 'CA':'CT', 'HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'CD':'CT','HD2':'H1','HD3':'H1',
        'NE':'NA','HE':'H', 'CZ':'C',
        'NH1':'NH1','HH11':'H','HH12':'H',
        'NH2':'NH2','HH21':'H','HH22':'H',
        'C':'C','O':'O'
    },
    'ASN': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','OD1':'O',
        'ND2':'NA','HD21':'H','HD22':'H',
        'C':'C','O':'O'
    },
    'ASP': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','OD1':'O','OD2':'O',
        'C':'C','O':'O'
    },
    'CYS': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'SG':'S','HG':'H1',
        'C':'C','O':'O'
    },
    'GLN': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'OE1':'O','NE2':'NA','HE21':'H','HE22':'H',
        'C':'C','O':'O'
    },
    'GLU': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'CD':'C','OE1':'O','OE2':'O',
        'C':'C','O':'O'
    },
    'GLY': {
        'N':'N','H':'H','CA':'CT','HA2':'H1',
        'HA3':'H1','C':'C','O':'O'
    },
    'HIS': {
        # using HID (proton on ND1) convention
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','ND1':'N','HD1':'H',
        'CD2':'C','CE1':'C','NE2':'NE','HE2':'H',
        'C':'C','O':'O'
    },
    'ILE': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB':'H1','CG1':'CT','HG12':'H1',
        'HG13':'H1','CG2':'CT','HG21':'H1','HG22':'H1',
        'HG23':'H1','CD1':'CT','HD11':'H1','HD12':'H1',
        'HD13':'H1','C':'C','O':'O'
    },
    'LEU': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'CD1':'CT','HD11':'H1','HD12':'H1','HD13':'H1',
        'CD2':'CT','HD21':'H1','HD22':'H1','HD23':'H1',
        'C':'C','O':'O'
    },
    'LYS': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'CD':'CT','HD2':'H1','HD3':'H1',
        'CE':'CT','HE2':'H1','HE3':'H1',
        'NZ':'N3','HZ1':'H1','HZ2':'H1','HZ3':'H1',
        'C':'C','O':'O'
    },
    'MET': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'SD':'S','CE':'CT','HE1':'H1','HE2':'H1','HE3':'H1',
        'C':'C','O':'O'
    },
    'PHE': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','CD1':'C','HD1':'H',
        'CD2':'C','HD2':'H','CE1':'C','HE1':'H',
        'CE2':'C','HE2':'H','CZ':'C','HZ':'H',
        'C':'C','O':'O'
    },
    'PRO': {
        'N':'N2','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'CT','HG2':'H1','HG3':'H1',
        'CD':'CT','HD2':'H1','HD3':'H1',
        'C':'C','O':'O'
    },
    'SER': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'OG':'OH','HG':'H','C':'C','O':'O'
    },
    'THR': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB':'H1','CG2':'CT','HG21':'H1',
        'HG22':'H1','HG23':'H1','OG1':'OH','HG1':'H',
        'C':'C','O':'O'
    },
    'TRP': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','CD1':'C','HD1':'H','NE1':'NE','HE1':'H',
        'CE2':'C','HE2':'H','CD2':'C','CE3':'C','HE3':'H',
        'CZ2':'C','HZ2':'H','CH2':'CR','HH2':'H','CZ3':'C',
        'HZ3':'H','C':'C','O':'O'
    },
    'TYR': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB2':'H1','HB3':'H1',
        'CG':'C','CD1':'C','HD1':'H','CD2':'C','HD2':'H',
        'CE1':'C','HE1':'H','CE2':'C','HE2':'H','CZ':'C',
        'OH':'OH','HH':'H','C':'C','O':'O'
    },
    'VAL': {
        'N':'N','H':'H','CA':'CT','HA':'H1',
        'CB':'CT','HB':'H1','CG1':'CT','HG11':'H1',
        'HG12':'H1','HG13':'H1','CG2':'CT','HG21':'H1',
        'HG22':'H1','HG23':'H1','C':'C','O':'O'
    }
}

# -----------------------------------------------------------------------------
def get_amber_type(resname, atom_name):
    """Lookup Amber type or fallback to atom_name if missing."""
    return AMBER99SB_TYPE_MAP.get(resname, {}).get(atom_name, atom_name)

def extract_backbone_neighbors_frame(traj, positions_A, k=20):
    """
    Pure NumPy + SciPy neighbor search for a single frame.
    Returns a DataFrame of nearest neighbours for each backbone amide H.
    """
    atoms     = list(traj.topology.atoms)
    res_count = traj.topology.n_residues

    # Build bonded map
    bonded = {atom.index: set() for atom in atoms}
    for a1, a2 in traj.topology.bonds:
        bonded[a1.index].add(a2.index)
        bonded[a2.index].add(a1.index)

    # Identify backbone amide H’s
    H_indices = []
    for res in traj.topology.residues:
        if res.is_protein and 0 < res.index < res_count - 1:
            Ns = [a.index for a in res.atoms if a.name == 'N']
            if not Ns:
                continue
            N_idx = Ns[0]
            for neigh in bonded[N_idx]:
                atom = traj.topology.atom(neigh)
                if atom.element and atom.element.symbol == 'H':
                    H_indices.append(neigh)
    if not H_indices:
        return pd.DataFrame()

    # Build heavy-atom list by name
    atom_names    = [a.name for a in atoms]
    heavy_indices = [i for i,name in enumerate(atom_names) if not name.startswith('H')]

    # Coordinates in Å
    coords_A     = np.array([[p.x, p.y, p.z] for p in positions_A])
    H_coords     = coords_A[H_indices]
    heavy_coords = coords_A[heavy_indices]

    # KD-Tree neighbor search
    tree = cKDTree(heavy_coords)
    dists, idxs = tree.query(H_coords, k=k)

    # Metadata lookups
    resnums   = [a.residue.index for a in atoms]
    resnames  = [a.residue.name  for a in atoms]

    # Assemble DataFrame rows
    rows = []
    for i_H, h_idx in enumerate(H_indices):
        h_pos = H_coords[i_H]
        for j in range(k):
            heavy_local_idx = idxs[i_H, j]
            n_idx  = heavy_indices[heavy_local_idx]
            n_pos  = heavy_coords[heavy_local_idx]
            dx, dy, dz = h_pos - n_pos
            amber_t = get_amber_type(resnames[n_idx], atom_names[n_idx])
            rows.append({
                'h_index':      h_idx,
                'resnum_h':     resnums[h_idx],
                'resname_h':    resnames[h_idx],
                'neigh_index':  n_idx,
                'resnum_n':     resnums[n_idx],
                'resname_n':    resnames[n_idx],
                'neigh_label':  f"{resnums[n_idx]}{atom_names[n_idx]}",  # new column
                'amber_type':   amber_t,
                'dx':           float(dx),
                'dy':           float(dy),
                'dz':           float(dz),
                'distance':     float(dists[i_H, j]),
            })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
def extract_backbone_neighbors_over_trajectory(loaded_pdb, k=20):
    """
    Loop over each frame, call extract_backbone_neighbors_frame, and tag with frame index.
    """

    all_dfs = []
    for frame_idx, (topology, positions) in enumerate(
        loaded_pdb,
        start=0
    ):
        traj = md.Trajectory(
            xyz=[[(p.x, p.y, p.z) for p in positions]],
            topology=md.Topology.from_openmm(topology)
        )
        df = extract_backbone_neighbors_frame(traj, positions, k=k)
        if not df.empty:
            df.insert(0, 'frame', frame_idx)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# -----------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Extract backbone amide H neighbors with Amber atom types."
    )
    p.add_argument('--pdb', required=True, help="Input PDB with hydrogens (Å)")
    p.add_argument('--xtc', required=True, help="Input XTC trajectory")
    p.add_argument('--out', required=True, help="Output CSV")
    p.add_argument('-k',   type=int, default=20, help="Neighbors per H")
    args = p.parse_args()

    df = extract_backbone_neighbors_over_trajectory(
        pdb_path=args.pdb,
        xtc_path=args.xtc,
        k=args.k
    )
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == '__main__':
    main()
