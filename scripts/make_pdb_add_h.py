#!/usr/bin/env python3
"""
make_multimodel_pdb_module.py

Provides:
  - generate_hydrogenated_frames(): a generator yielding (topology, positions_A) per frame
  - main(): CLI to write a multi‑model PDB just like before

Usage as a module:

    from make_multimodel_pdb_module import generate_hydrogenated_frames

    for idx, (topology, positions_A) in enumerate(
            generate_hydrogenated_frames(
                pdb_path="input.pdb",
                xtc_path="traj.xtc",
                ffxml="amber99sb.xml",
                pH=7.0
            ), start=1
        ):
        # `topology` is an OpenMM Topology
        # `positions_A` is a list of Vec3 in Å units for this frame
        # → here you could compute distances, angles, whatever!
        dist = (positions_A[atom1_index] - positions_A[atom2_index]).norm()
        ...

Usage from the command line:

    python make_multimodel_pdb_module.py \
        --pdb input.pdb \
        --xtc traj.xtc \
        --out multi.pdb \
        [--ffxml amber99sb.xml] \
        [--pH 7.0]
"""

import argparse
import MDAnalysis as mda
from openmm.app import PDBFile, Modeller, ForceField
from openmm.vec3 import Vec3

def generate_hydrogenated_frames(pdb_path, xtc_path, ffxml=None, pH=7.0):
    """
    Generator that yields (topology, positions_A) per frame.

    - pdb_path:   path to input PDB (Å units)
    - xtc_path:   path to XTC trajectory
    - ffxml:      optional force field XML name/path
    - pH:         protonation pH

    Yields:
      topology      : an openmm.app.Topology object
      positions_A   : list of Vec3 positions in Å for this frame (heavy + H)
    """
    # 1) Load & convert PDB → nm
    pdb = PDBFile(pdb_path)
    nm_init_positions = [Vec3(p.x*0.1, p.y*0.1, p.z*0.1) for p in pdb.getPositions()]
    base_modeller = Modeller(pdb.topology, nm_init_positions)

    # 2) Prepare force field
    ff = ForceField(ffxml) if ffxml else ForceField('amber99sb.xml')

    # 3) Load trajectory
    u = mda.Universe(pdb_path, xtc_path)

    # 4) Loop over frames
    for ts in u.trajectory:
        # A) scale this frame's heavy-atom coords → nm
        frame_nm = [Vec3(*pos)*0.1 for pos in u.atoms.positions]
        base_modeller.positions = frame_nm

        # B) add hydrogens
        modeller = Modeller(base_modeller.topology, base_modeller.positions)
        modeller.addHydrogens(ff, pH=pH)

        # C) scale back → Å
        positions_A = [Vec3(p.x*10, p.y*10, p.z*10) for p in modeller.getPositions()]

        # yield topology + Å positions
        yield modeller.topology, positions_A


def main():
    p = argparse.ArgumentParser(
        description="Make a multimodel PDB with per-frame protonation (Å↔nm handling)."
    )
    p.add_argument('--pdb',   required=True, help="Input PDB (Å)")
    p.add_argument('--xtc',   required=True, help="Input trajectory XTC")
    p.add_argument('--out',   required=True, help="Output multi‑model PDB (Å)")
    p.add_argument('--ffxml', help="Force field XML (e.g. amber99sb.xml)")
    p.add_argument('--pH',    type=float, default=7.0, help="pH for protonation")
    args = p.parse_args()

    with open(args.out, 'w') as out:
        for idx, (topology, positions_A) in enumerate(
            generate_hydrogenated_frames(
                pdb_path=args.pdb,
                xtc_path=args.xtc,
                ffxml=args.ffxml,
                pH=args.pH
            ), start=1
        ):
            out.write(f"MODEL     {idx}\n")
            PDBFile.writeFile(topology, positions_A, out, keepIds=True)
            out.write("ENDMDL\n")
    print(f"→ Wrote {idx} frames (with hydrogens) to {args.out}")

if __name__ == "__main__":
    main()
