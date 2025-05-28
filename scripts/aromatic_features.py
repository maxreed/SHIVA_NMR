import numpy as np
import pandas as pd
import mdtraj as md

# Aromatic ring definitions
AROMATIC_RINGS = {
    'PHE': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TYR': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TRP': ['CD2','CE2','CE3','CZ2','CZ3','CH2'],
    'HIS': ['CG','ND1','CD2','CE1','NE2'],
}
INV_SCALE = 5.0
# Standard amino acids
STANDARD_AA = set([
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
])

def extract_aromatic_ring_features(frames_generator, k=3):
    """
    For each frame from frames_generator yielding (topology, positions):
    compute for each backbone amide H the k nearest aromatic ring centroids in
    the local NH-CC frame and the orientation (normal) of each ring plane.

    Returns DataFrame with rows per (frame, resnum_h)
    and columns:
      ring{i}_resname, ring{i}_invdist, ring{i}_dx, ring{i}_dy, ring{i}_dz,
      ring{i}_nx, ring{i}_ny, ring{i}_nz  (components of ring plane normal)
    Missing rings are NaN; one-hot encodes resnames.
    """
    records = []
    for frame_idx, (topology, positions) in enumerate(frames_generator):
        traj_top = md.Topology.from_openmm(topology)
        coords = np.array([[p.x, p.y, p.z] for p in positions])
        atoms_list = list(topology.atoms())
        bonded = {a.index: set() for a in atoms_list}
        for b in topology.bonds():
            bonded[b[0].index].add(b[1].index)
            bonded[b[1].index].add(b[0].index)
        # precompute ring centroids and normals
        ring_list = []
        for res in topology.residues():
            if res.name not in AROMATIC_RINGS:
                continue
            pts = np.array([coords[a.index] for a in res.atoms() if a.name in AROMATIC_RINGS[res.name]])
            if pts.shape[0] < 3:
                continue
            centroid = pts.mean(axis=0)
            # plane normal via SVD: last singular vector
            u, s, vh = np.linalg.svd(pts - centroid)
            normal = vh[-1]
            # ensure consistent orientation
            normal /= np.linalg.norm(normal)
            ring_list.append((res.name, res.index, centroid, normal))
        # process each backbone H
        for res in topology.residues():
            rn = res.name; rnum = res.index
            if rn not in STANDARD_AA or rnum==0 or rnum==traj_top.n_residues-1:
                continue
            Ns = [a.index for a in res.atoms() if a.name=='N']
            if not Ns: continue
            N_idx = Ns[0]
            Hs = [nbr for nbr in bonded[N_idx] if atoms_list[nbr].element and atoms_list[nbr].element.symbol=='H']
            if not Hs: continue
            H_idx = Hs[0]; h_pos = coords[H_idx]
            # local frame
            n_pos = coords[N_idx]; prev_res = list(topology.residues())[rnum-1]
            C_idx = [a.index for a in prev_res.atoms() if a.name=='C'][0]; c_pos = coords[C_idx]
            e1 = (h_pos - n_pos); e1 /= np.linalg.norm(e1)
            vC = (h_pos - c_pos); v_perp = vC - np.dot(vC,e1)*e1; e2 = v_perp/np.linalg.norm(v_perp)
            e3 = np.cross(e1,e2); Rmat = np.vstack([e1,e2,e3])
            # distances and rotated vectors
            feats = []
            for (name, idx, cent, normal) in ring_list:
                vec = cent - h_pos; d = np.linalg.norm(vec)
                if d==0: continue
                invd = INV_SCALE/d
                routed = Rmat.dot(vec); norm = np.linalg.norm(routed)
                if norm==0: continue
                direc = routed/norm
                # rotate normal into local frame
                n_rot = Rmat.dot(normal)
                feats.append((name, invd, direc, n_rot))
            feats.sort(key=lambda x: -x[1])
            rec = {'frame': frame_idx, 'resnum_h': rnum}
            for i in range(k):
                suffix = f'{i+1}'
                if i < len(feats):
                    name, invd, direc, n_rot = feats[i]
                    rec.update({
                        f'ring{suffix}_resname': name,
                        f'ring{suffix}_invdist': invd,
                        f'ring{suffix}_dx': -direc[0], # the direction vectors are fliped in sign versus what we have for the nearest neighbors. this doesn't actually matter, but i correct it here to be consistent.
                        f'ring{suffix}_dy': -direc[1],
                        f'ring{suffix}_dz': -direc[2],
                        f'ring{suffix}_nx': -n_rot[0],
                        f'ring{suffix}_ny': -n_rot[1],
                        f'ring{suffix}_nz': -n_rot[2],
                    })
                else:
                    for fld in ['resname','invdist','dx','dy','dz','nx','ny','nz']:
                        rec[f'ring{i+1}_{fld}'] = pd.NA
            records.append(rec)
    df = pd.DataFrame(records)
    # one-hot encode ring*_resname
    rescols = [c for c in df.columns if '_resname' in c]
    df = pd.get_dummies(df, columns=rescols, prefix=rescols, dtype=int)
    
    # next bit added because if there was no residues of a given type, it wouldn't make a boolean variable for that type.

    # build the list of all expected dummy names
    expected = []
    for i in range(1, k+1):
        for res in AROMATIC_RINGS:
            expected.append(f"ring{i}_resname_{res}")

    # add any missing dummy columns with 0
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    expected = []
    for i in range(1, k+1):
        for res in AROMATIC_RINGS:
            expected.append(f"ring{i}_resname_{res}")

    # split out other columns
    other_cols = [c for c in df.columns if not c.startswith('ring') or '_resname_' not in c]

    # now reassemble with dummy columns in the exact order we want
    df = df[ other_cols + expected ]

    return df
