import numpy as np
import pandas as pd

def transform_neighbor_features(df):
    """
    Given the neighbor‐list DataFrame (with columns
    frame, h_index, resnum_h, resname_h, neigh_label, dx, dy, dz, distance, amber_type),
    returns a new DataFrame with:
      - rotated & unit‐normalized displacement vectors (rx, ry, rz)
      - distance replaced by 1/distance
      - one‐hot columns for resname_n and amber_type
    """
    out_frames = []
    # process each amide‐H neighborhood
    for (frame, h_idx), group in df.groupby(['frame', 'h_index'], sort=False):
        # 1) find the N neighbor and C neighbor
        resnum_h = group['resnum_h'].iat[0]
        label_N  = f"{resnum_h}N"
        label_C  = f"{resnum_h-1}C"
        try:
            v_NH = group.loc[group['neigh_label'] == label_N, ['dx','dy','dz']].values[0]
            v_C  = group.loc[group['neigh_label'] == label_C,  ['dx','dy','dz']].values[0]
        except IndexError:
            # skip if either is missing
            continue

        # 2) build orthonormal basis
        e1 = v_NH / np.linalg.norm(v_NH)
        # project v_C out of e1
        v_perp = v_C - np.dot(v_C, e1)*e1
        e2 = v_perp / np.linalg.norm(v_perp)
        e3 = np.cross(e1, e2)
        R = np.vstack([e1, e2, e3])  # rows are new basis axes

        # 3) rotate & normalize all displacements in this group
        disp = group[['dx','dy','dz']].values  # shape (k,3)
        rotated = disp.dot(R.T)               # project into new basis
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        unit_rot = rotated / norms            # unit‐normalize

        # 4) build output DataFrame for this group
        df_grp = group.copy().reset_index(drop=True)
        df_grp['rx'] = unit_rot[:,0]
        df_grp['ry'] = unit_rot[:,1]
        df_grp['rz'] = unit_rot[:,2]
        df_grp['distance'] = 1.0 / df_grp['distance']

        out_frames.append(df_grp)

    # 5) concatenate all frames back together
    df2 = pd.concat(out_frames, ignore_index=True)

    # 6) one‐hot encode resname_n and amber_type
    df2 = pd.get_dummies(
        df2,
        columns=['resname_n','amber_type'],
        prefix=['res','atype'],
        dtype=int
    )

    # 1) Define your full category lists once:
    ALL_RESIDUES = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
                    'HIS','ILE','LEU','LYS','MET','PHE','PRO',
                    'SER','THR','TRP','TYR','VAL']
    # these must match exactly your prefixing; pandas will name them 'res_<AA>'
    expected_res_cols = [f"res_{aa}" for aa in ALL_RESIDUES]

    # For amber types, extract the full set from your map:
    from find_backbone_neighbors import AMBER99SB_TYPE_MAP  # or import where defined
    # we don't use the hydrogen types so we can delete them.
    REMOVE = {'H','H1'}
    for resname, atom_map in AMBER99SB_TYPE_MAP.items():
        # remove any atom_name → amber_type entries where amber_type is in REMOVE
        for atom_name in [a for a,t in atom_map.items() if t in REMOVE]:
            del atom_map[atom_name]

    all_amber = set()
    for mp in AMBER99SB_TYPE_MAP.values():
        all_amber |= set(mp.values())
    expected_amber_cols = [f"atype_{t}" for t in sorted(all_amber)]

    # 2) Insert any missing one-hot columns:
    for col in expected_res_cols + expected_amber_cols:
        if col not in df2.columns:
            df2[col] = 0

    # 3) (Optional) Reorder so dummies always in the same place:
    #    pull out all non-dummy columns first, then append expected_dummies in order
    non_dummy = [c for c in df2.columns if not (c.startswith('res_') or c.startswith('atype_'))]
    df2 = df2[ non_dummy + expected_res_cols + expected_amber_cols ]

    return df2
