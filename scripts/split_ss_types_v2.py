# this one doesn't drop ss type for the line in consideration.
import pandas as pd
from pathlib import Path
import argparse

def process_csv_file(input_file, output_dir_H, output_dir_E, output_dir_C):
    df = pd.read_csv(input_file)

    # Define mapping of SS columns to simplified SS type
    ss_type_map = {
        'H': ['ss_H', 'ss_G', 'ss_I'],
        'E': ['ss_E', 'ss_B'],
        'C': ['ss_T', 'ss_S', 'ss_C']
    }

    # Determine SS type for each row
    def determine_ss_type(row):
        for ss_type, cols in ss_type_map.items():
            if any(row.get(col, 0) == 1 for col in cols):
                return ss_type
        return None

    df['ss_type'] = df.apply(determine_ss_type, axis=1)

    base_name = Path(input_file).stem
    counts = {'H': 0, 'E': 0, 'C': 0}

    for ss_type, output_dir in zip(['H', 'E', 'C'], [output_dir_H, output_dir_E, output_dir_C]):
        subset = df[df['ss_type'] == ss_type].drop(columns=['ss_type'])
        if not subset.empty:
            output_file = Path(output_dir) / f"{base_name}_ss_{ss_type}.csv"
            subset.to_csv(output_file, index=False)
            counts[ss_type] = len(subset)

    print(f"{input_file.name}: H={counts['H']} rows, E={counts['E']} rows, C={counts['C']} rows")

def main():
    parser = argparse.ArgumentParser(description="Split CSV rows by secondary structure type, preserving contextual SS columns.")
    parser.add_argument("input_dir", help="Directory with input CSV files.")
    parser.add_argument("output_dir_H", help="Directory for H-type output CSVs.")
    parser.add_argument("output_dir_E", help="Directory for E-type output CSVs.")
    parser.add_argument("output_dir_C", help="Directory for C-type output CSVs.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    for file in input_dir.glob("*.csv"):
        process_csv_file(file, args.output_dir_H, args.output_dir_E, args.output_dir_C)

if __name__ == "__main__":
    main()
