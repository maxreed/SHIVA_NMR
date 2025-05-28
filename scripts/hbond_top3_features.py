import numpy as np
import pandas as pd
import mdtraj as md

def build_hbond_defs(frames_generator, skip_neighbor_seq=True):
    """
    Scan the first frame to collect all backbone NH → O pairs.

    Parameters
    ----------
    frames_generator : iterable yielding (openmm.Topology, positions)
        Your pipeline’s frame generator.
    skip_neighbor_seq : bool, default=True
        If True, do not pair residue i’s NH with its own O or its i±1 neighbors’ O.

    Returns
    -------
    hbond_defs : dict
        Mapping hbond_id → (donor_atom_index, hydrogen_atom_index, acceptor_atom_index).
        hbond_id is a string "{donor_residue}_to_{acceptor_residue}_{o_idx}".
    """
    # grab first frame
    openmm_top0, _ = next(iter(frames_generator))
    md_top = md.Topology.from_openmm(openmm_top0)

    # collect atom indices by (residue.index)
    donors = {}     # residue → N atom idx
    hydrogens = {}  # residue → H atom idx
    acceptors = {}  # residue → list of O atom idxs

    for atom in md_top.atoms:
        ridx = atom.residue.index
        if atom.name == 'N':
            donors[ridx] = atom.index
        elif atom.name in ('H','H1','H2','H3'):  # adapt to your naming
            # pick the backbone H (you may need to filter by bonded partner)
            hydrogens[ridx] = atom.index
        elif atom.name == 'O':
            acceptors.setdefault(ridx, []).append(atom.index)

    hbond_defs = {}
    for i_res, d_idx in donors.items():
        h_idx = hydrogens.get(i_res)
        if h_idx is None:
            continue  # no protonated N found
        for j_res, o_list in acceptors.items():
            if skip_neighbor_seq and abs(j_res - i_res) <= 1:
                continue
            for o_idx in o_list:
                key = f"{i_res}_to_{j_res}_{o_idx}"
                hbond_defs[key] = (d_idx, h_idx, o_idx)

    return hbond_defs

def compute_top3_hbond_features(frames_generator, hbond_atom_indices,
                                 dist_thresh=3.5, angle_thresh=120.0):
    """
    For each frame and each backbone amide (donor), compute up to 3 strongest H-bonds.

    Parameters:
    - frames_generator: iterable yielding (openmm.Topology, positions) per frame
    - hbond_atom_indices: dict mapping hbond_id -> (donor_idx, hydrogen_idx, acceptor_idx)
    - dist_thresh: max D-A distance (nm)
    - angle_thresh: min D-H-A angle (degrees)

    Returns:
      DataFrame with columns:
        frame, residue,
        dist_DA_1, angle_DHA_1, is_hbond_1,
        dist_DA_2, angle_DHA_2, is_hbond_2,
        dist_DA_3, angle_DHA_3, is_hbond_3
      where 'residue' is the donor residue index.
    """
    # collect frames
    frames = list(frames_generator)
    if not frames:
        # build empty DataFrame with correct cols
        cols = ['frame','residue'] + [f for i in range(1,4) for f in [f'dist_DA_{i}', f'angle_DHA_{i}', f'is_hbond_{i}']]
        return pd.DataFrame(columns=cols)

    # convert topology once
    openmm_top0, _ = frames[0]
    md_top = md.Topology.from_openmm(openmm_top0)

    records = []
    # loop frames and raw stats
    for f, (_, positions) in enumerate(frames):
        coords = np.array([[p.x, p.y, p.z] for p in positions])
        for h_id, (d_idx, h_idx, a_idx) in hbond_atom_indices.items():
            D, H, A = coords[d_idx], coords[h_idx], coords[a_idx]
            dist_DA = np.linalg.norm(D - A)
            v1, v2 = D - H, A - H
            if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
                angle_DHA = np.nan
            else:
                cosang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                angle_DHA = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            is_hb = (dist_DA <= dist_thresh) and (angle_DHA >= angle_thresh)
            # donor residue index
            donor_res = md_top.atom(d_idx).residue.index
            records.append({
                'frame': f,
                'residue': donor_res,
                'dist_DA': float(dist_DA),
                'angle_DHA': float(angle_DHA),
                'is_hbond': bool(is_hb)
            })
    df = pd.DataFrame.from_records(records)

    # filter to true hbonds, then pick top 3 by distance
    df_true = df[df['is_hbond']]
    df_sorted = df_true.sort_values(['frame','residue','dist_DA'])
    df_top3 = (df_sorted
               .groupby(['frame','residue'])
               .head(3)
               .copy()
               )
    # assign rank
    df_top3['rank'] = df_top3.groupby(['frame','residue']).cumcount() + 1
    # pivot to wide
    df_wide = df_top3.pivot(index=['frame','residue'],
                             columns='rank',
                             values=['dist_DA','angle_DHA','is_hbond'])
    # flatten columns
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    # ensure all rank columns present
    for i in range(1,4):
        for prefix in ['dist_DA', 'angle_DHA', 'is_hbond']:
            col = f"{prefix}_{i}"
            if col not in df_wide.columns:
                df_wide[col] = np.nan

    # reorder columns
    col_order = ['frame','residue'] + [f for i in range(1,4) for f in [f'dist_DA_{i}', f'angle_DHA_{i}']]
    return df_wide[col_order]
