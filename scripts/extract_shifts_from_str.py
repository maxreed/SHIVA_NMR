# march 28, 2025
# this takes selected atom shifts out of a star file.
# these can be used as y data during training.
# example usage:
# python extract_H_shifts_from_str.py input.str output.csv --atom-types H HN
# the first passed atom type is used to name the columns in the output file

import argparse
import pandas as pd
import pynmrstar


def extract_atom_shifts(star_file, atom_types=None):
    if atom_types is None:
        atom_types = ["H", "HN"]

    atom_types = [atom.strip() for atom in atom_types if atom.strip()]
    if not atom_types:
        raise ValueError("atom_types must contain at least one atom name.")

    primary_atom = atom_types[0].lower()
    target_atom_names = set(atom_types)

    entry = pynmrstar.Entry.from_file(star_file)

    # Find the assigned_chemical_shifts saveframe
    shift_frame = None
    for sf in entry:
        if sf.category == "assigned_chemical_shifts":
            shift_frame = sf
            break

    if not shift_frame:
        raise ValueError("No 'assigned_chemical_shifts' saveframe found.")

    # Look for the loop with Atom_chem_shift.Atom_ID and Atom_chem_shift.Val
    correct_loop = None
    for loop in shift_frame.loops:
        tag_names = [tag.strip().lstrip("_") for tag in loop.get_tag_names()]
        if (
            "Atom_chem_shift.Atom_ID" in tag_names
            and "Atom_chem_shift.Val" in tag_names
        ):
            correct_loop = loop
            break

    if not correct_loop:
        raise ValueError(
            "No chemical shift loop with expected Atom_chem_shift fields found."
        )

    tags = [tag.strip().lstrip("_") for tag in correct_loop.get_tag_names()]

    def get_index(field_name):
        if field_name not in tags:
            raise ValueError(f"Missing expected tag: {field_name}")
        return tags.index(field_name)

    seq_id_idx = get_index("Atom_chem_shift.Seq_ID")
    atom_id_idx = get_index("Atom_chem_shift.Atom_ID")
    val_idx = get_index("Atom_chem_shift.Val")

    shifts_by_residue = {}
    for row in correct_loop:
        atom_name = row[atom_id_idx].strip()
        if atom_name not in target_atom_names:
            continue

        try:
            seq_id = int(row[seq_id_idx])
            shift_val = float(row[val_idx])
            shifts_by_residue[seq_id] = shift_val
        except (ValueError, TypeError):
            continue

    # Extract sequence
    sequence = None
    for sf in entry:
        if sf.category == "entity" and "Polymer_seq_one_letter_code" in sf:
            raw_seq = sf["Polymer_seq_one_letter_code"][0]
            sequence = raw_seq.replace("\n", "").replace(" ", "")
            break

    if sequence is None:
        raise ValueError("Sequence not found in entity saveframe.")

    shift_data = []
    for i, aa in enumerate(sequence, start=1):
        shift = shifts_by_residue.get(i, float("nan"))
        shift_data.append([i - 1, aa, shift])

    shift_df = pd.DataFrame(
        shift_data,
        columns=[f"resnum_{primary_atom}", "res", f"amide_{primary_atom}_shift"]
    )

    return shift_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract selected atom chemical shifts from an NMR-STAR file."
    )
    parser.add_argument(
        "input_star",
        help="Path to input NMR-STAR file"
    )
    parser.add_argument(
        "output_csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--atom-types",
        nargs="+",
        default=["H", "HN"],
        help="One or more atom names to accept (default: H HN)"
    )

    args = parser.parse_args()

    these_shifts = extract_atom_shifts(
        args.input_star,
        atom_types=args.atom_types,
    )
    these_shifts.to_csv(args.output_csv, index=False)
    print(f"✅ Wrote output to {args.output_csv}")
