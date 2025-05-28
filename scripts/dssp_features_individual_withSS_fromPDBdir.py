import numpy as np
import pandas as pd
import mdtraj as md
import glob, os

def compute_dssp_stats_framewise(pdb_dir: str) -> pd.DataFrame:
    """
    Load a directory of PDB files (each representing a frame) and compute per-frame φ/ψ and full DSSP one-hot.

    - pdb_dir: directory containing PDB files of individual states

    Returns a DataFrame with columns:
      frame, residue,
      sin_phi_prev, cos_phi_prev, sin_phi, cos_phi, sin_phi_next, cos_phi_next,
      sin_psi_prev, cos_psi_prev, sin_psi, cos_psi, sin_psi_next, cos_psi_next,
      and one-hot columns ss_prev_<code>, ss_<code>, ss_next_<code> for all DSSP codes: H, B, E, G, I, T, S, C.
    """
    # Collect and sort PDB files
    pdb_paths = sorted(glob.glob(os.path.join(pdb_dir, '*.pdb')))
    if len(pdb_paths) < 1:
        raise ValueError(f"No PDB files found in directory: {pdb_dir}")

    # Use the first PDB as topology
    topology_path = pdb_paths[0]

    # Load all PDBs as trajectory
    traj = md.load(pdb_paths, top=topology_path)

    n_frames = traj.n_frames
    n_res = traj.n_residues

    # Compute φ/ψ
    phi_idx, phi_ang = md.compute_phi(traj)
    psi_idx, psi_ang = md.compute_psi(traj)
    sin_phi = np.sin(phi_ang); cos_phi = np.cos(phi_ang)
    sin_psi = np.sin(psi_ang); cos_psi = np.cos(psi_ang)
    resid_phi = [traj.topology.atom(p[1]).residue.index for p in phi_idx]
    resid_psi = [traj.topology.atom(p[1]).residue.index for p in psi_idx]
    phi_map = {res: i for i, res in enumerate(resid_phi)}
    psi_map = {res: i for i, res in enumerate(resid_psi)}

    # Compute full DSSP codes
    dssp_arr = md.compute_dssp(traj, simplified=False)
    # Map blanks to coil 'C'
    dssp_arr = np.where(dssp_arr == ' ', 'C', dssp_arr)

    # Define fixed SS code list
    ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']

    # Prepare records
    records = []
    for f in range(n_frames):
        for res in range(n_res):
            rec = {'frame': f, 'residue': res}
            # Dihedral features
            for backbone, amap, sarr, carr in [
                ('phi', phi_map, sin_phi, cos_phi),
                ('psi', psi_map, sin_psi, cos_psi)
            ]:
                for offset, lbl in [(-1, '_prev'), (0, ''), (1, '_next')]:
                    sin_k = f'sin_{backbone}{lbl}'
                    cos_k = f'cos_{backbone}{lbl}'
                    tgt = res + offset
                    if tgt in amap:
                        idx = amap[tgt]
                        rec[sin_k] = float(sarr[f, idx])
                        rec[cos_k] = float(carr[f, idx])
                    else:
                        rec[sin_k] = np.nan
                        rec[cos_k] = np.nan

            # SS one-hot for fixed codes
            for offset, lbl in [(-1, '_prev'), (0, ''), (1, '_next')]:
                tgt = res + offset
                ss_val = dssp_arr[f, tgt] if 0 <= tgt < n_res else None
                for ss in ss_types:
                    col = f'ss{lbl}_{ss}'
                    rec[col] = 1 if ss_val == ss else 0

            records.append(rec)

    # Build DataFrame with guaranteed columns order
    cols = ['frame', 'residue'] + \
           [f for backbone in ['phi', 'psi'] for lbl in ['_prev', '', '_next'] for f in [f'sin_{backbone}{lbl}', f'cos_{backbone}{lbl}']] + \
           [f for lbl in ['_prev', '', '_next'] for ss in ss_types for f in [f'ss{lbl}_{ss}']]
    df = pd.DataFrame.from_records(records, columns=cols)
    return df
