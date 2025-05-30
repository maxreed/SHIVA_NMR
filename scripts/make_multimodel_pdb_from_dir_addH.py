#!/usr/bin/env python3
"""
make_multimodel_pdb_from_dir.py  (Amber 99SB default)

Adds any missing heavy atoms and then adds hydrogens exactly as in the original
make_pdb_add_h.py, but operates on a directory of PDB frames.

CLI:
    python make_multimodel_pdb_from_dir.py --dir frames --out combined.pdb
    # optional:
    #   --ffxml amber14-all.xml
    #   --pH    7.4
"""

import argparse
import glob
import os
import sys
from typing import Generator, List, Tuple

from openmm.app import PDBFile, Modeller, ForceField, Topology
from openmm.vec3 import Vec3
from openmm.app import element as elem


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _load_forcefield(ffxml: str | None) -> ForceField:
    """Load chosen FF (default Amber 99SB, no water model)."""
    return ForceField(ffxml) if ffxml else ForceField("amber99sb.xml")

# excludes hydrogens because the protocol for adding hydrogens can result in different protonation states.
def _atom_signature(top: Topology) -> tuple:
    # skip all H atoms
    return tuple(
        (a.name, int(a.residue.id), a.residue.name)
        for a in top.atoms()
        if a.element != elem.hydrogen
    )


def _augment_model(
    pdb: PDBFile, ff: ForceField, ph: float
) -> Tuple[Topology, List[Vec3]]:
    """Fill in missing heavy atoms + hydrogens, return positions in Å."""
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=ph)       # hydrogens (only if absent)

    # Convert nm → Å
    pos_A = [Vec3(p.x * 10, p.y * 10, p.z * 10) for p in modeller.positions]
    return modeller.topology, pos_A


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #
def generate_frames_from_pdb_dir(
    pdb_dir: str, ffxml: str | None = None, ph: float = 7.0
) -> Generator[Tuple[Topology, List[Vec3]], None, None]:
    """
    Yield (Topology, positions_Å) for each alphabetically sorted PDB frame,
    after completing heavy atoms and hydrogens with the selected force field.
    """
    ff = _load_forcefield(ffxml)

    pdb_paths = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_paths:
        raise FileNotFoundError(f"No PDB files in '{pdb_dir}'")

    ref_sig = ref_top = None
    for idx, pdb_path in enumerate(pdb_paths, 1):
        try:
            pdb = PDBFile(pdb_path)
            top, pos_A = _augment_model(pdb, ff, ph)
        except Exception as exc:
            print(f"[WARN] Frame {idx:>3} {os.path.basename(pdb_path)} – {exc}. Skipped.",
                  file=sys.stderr)
            continue

        sig = _atom_signature(top)
        if ref_sig is None:
            ref_sig, ref_top = sig, top
            yield ref_top, pos_A
        elif sig == ref_sig:
            yield ref_top, pos_A
        else:
            print(f"[WARN] Frame {idx:>3} {os.path.basename(pdb_path)} – "
                  "atom order mismatch. Skipped.", file=sys.stderr)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _write_multimodel_pdb(pdb_dir: str, out_path: str,
                          ffxml: str | None, ph: float) -> None:
    with open(out_path, "w") as fh:
        for m, (top, pos_A) in enumerate(
            generate_frames_from_pdb_dir(pdb_dir, ffxml, ph), 1
        ):
            fh.write(f"MODEL     {m}\n")
            PDBFile.writeFile(top, pos_A, fh, keepIds=True)
            fh.write("ENDMDL\n")
    print(f"✓ Wrote {m} models to '{out_path}'")


def main():
    ap = argparse.ArgumentParser(
        description="Concatenate PDB directory into multi-model PDB; "
                    "adds missing heavy atoms & hydrogens "
                    "(Amber 99SB default, no water model)."
    )
    ap.add_argument("--dir", required=True, help="Directory with PDB frames")
    ap.add_argument("--out", required=True, help="Output multi-model PDB file")
    ap.add_argument("--ffxml", default=None,
                    help="Force-field XML (default amber99sb.xml)")
    ap.add_argument("--pH", type=float, default=7.0,
                    help="pH for protonation (default 7.0)")
    args = ap.parse_args()

    _write_multimodel_pdb(args.dir, args.out, args.ffxml, args.pH)


if __name__ == "__main__":
    main()
