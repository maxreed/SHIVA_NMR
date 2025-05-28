import csv
import os

def append_csvs(output_filename='test_predictions_all.csv'):
    # Names of the CSV files to append
    input_files = ['test_predictions_C.csv',
                   'test_predictions_E.csv',
                   'test_predictions_H.csv']

    # Check that each input file exists
    for fname in input_files:
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Required file not found: {fname}")

    # Open the output file for writing
    with open(output_filename, 'w', newline='') as fout:
        writer = None

        # Iterate through each input file
        for idx, fname in enumerate(input_files):
            with open(fname, 'r', newline='') as fin:
                reader = csv.reader(fin)
                header = next(reader)

                # Write header once at the start
                if idx == 0:
                    writer = csv.writer(fout)
                    writer.writerow(header)

                # Append all rows from this file
                for row in reader:
                    writer.writerow(row)

    print(f"All files appended into '{output_filename}'.")

if __name__ == '__main__':
    append_csvs()
