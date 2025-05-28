#!/usr/bin/env python3
import argparse
import pandas as pd

def main(csv_path):
    # 1. Load data
    df = pd.read_csv(csv_path)
    # Expect columns: source_id, actual, predicted

    # 2. Parse source_id → num1, num2, num3
    #    e.g. "12_3_45" → num1="12", num2="3", num3="45"
    parts = df['source_id'].str.split('_', expand=True)
    parts.columns = ['num1', 'num2', 'num3']
    df = pd.concat([df, parts], axis=1)

    # 3. Group by (num1, num3), average predicted, keep actual (identical within group)
    grouped = df.groupby(['num1', 'num3']).agg(
        predicted_mean = ('predicted', 'mean'),
        predicted_std =('predicted', 'std')
    ).reset_index()

    # 4. Compute metrics
    y_pred = grouped['predicted_mean']

    grouped.to_csv("test_after_avg.csv")
    print("Averaged over sub-states.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate test predictions by num1+num3 and evaluate."
    )
    parser.add_argument(
        "csv_path",
        help="Path to test_predictions.csv"
    )
    args = parser.parse_args()
    main(args.csv_path)
