import os
import glob
import pandas as pd
import numpy as np
import re
import joblib

def extract_base_id(filename):
    match = re.search(r"(bmr\d+_\d+)", filename)
    return match.group(1) if match else None

DATA_DIR_test = "feats_SS_ensemble_H"
csv_files_test = sorted(glob.glob(os.path.join(DATA_DIR_test, "*.csv")))
csv_files_test = [f for f in csv_files_test if os.path.isfile(f)]
test_files = csv_files_test

print("\nTest files:")
for f in test_files:
    print("  ", os.path.basename(f))

def load_data(file_list):
    X, ids = [], []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            frame = df.iloc[1:, 0].astype(float).values
            resnum_h = df.iloc[1:, 1].astype(float).values
            features = df.iloc[1:, 2:].astype(float).values
            X.append(features)
            # Extract numeric ID from file name, repeat for all rows
            match = re.search(r"(\d+)", os.path.basename(file_path))
            file_id = match.group(1) if match else "unknown"
            composite_ids = [
                f"{file_id}_{int(f)}_{int(r)}"
                for f, r in zip(frame, resnum_h)
            ]
            ids.extend(composite_ids)
        except Exception as e:
            print(f" - Error reading {file_path}: {e}")
    if not X:
        raise ValueError("No valid data found in provided files.")
    return np.vstack(X), ids

# Normalize / de-normalize helpers
def normalize_target(y):
    return (y - 6) / 4

def denormalize_target(y_norm):
    return y_norm * 4 + 6

# Load data
X_test, test_ids = load_data(test_files)
print(X_test.shape)

model_name = "weights/xgb_h_SSensemble_it1_2x_est600_dep7_purge_isolated_H.pkl"
model = joblib.load(model_name)
print("Loaded model " + model_name)

# Predict (normalized)
test_preds_norm = model.predict(X_test)

# De-normalize predictions and targets
test_preds = denormalize_target(test_preds_norm)

pd.DataFrame({
    "source_id": test_ids,
    "predicted": test_preds
}).to_csv("test_predictions_H.csv", index=False)

print("Prediction complete!")

