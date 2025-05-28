import pandas as pd

def flatten_neighbor_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Flatten neighbor rows so that each (frame, resnum_h) group of k rows
    becomes a single row with suffixes _nn1 ... _nnk for all other columns.

    Parameters:
      df: DataFrame after transform_neighbor_features
      k: number of neighbors per H

    Returns:
      flattened DataFrame sorted by resnum_h and frame
    """
    # 1) Drop unwanted columns
    df = df.drop(columns=[
        'resname_h', 'h_index', 'neigh_index',
        'resnum_n', 'neigh_label',
        'dx', 'dy', 'dz'
    ], errors='ignore')

    records = []
    # 2) Group by frame and h_index (resnum_h)
    for (frame, resnum_h), grp in df.groupby(['frame', 'resnum_h'], sort=False):
        grp = grp.reset_index(drop=True)
        record = {'frame': frame, 'resnum_h': resnum_h}
        # 3) Flatten each neighbor row
        for i in range(len(grp)):
            suffix = f'_nn{i+1}'
            for col in grp.columns:
                if col in ['frame', 'resnum_h']:
                    continue
                record[col + suffix] = grp.at[i, col]
        records.append(record)

    # 4) Build DataFrame and sort
    flat_df = pd.DataFrame(records)
    flat_df = flat_df.sort_values(['resnum_h', 'frame']).reset_index(drop=True)
    return flat_df
