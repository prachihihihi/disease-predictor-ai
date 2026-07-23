import pandas as pd
import numpy as np
import argparse


def load_training(path="data/Training.csv"):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def augment_row(row, symptom_cols, drop_rate, add_rate, rng):
    row = row.copy()
    present = [c for c in symptom_cols if row[c] == 1]
    absent = [c for c in symptom_cols if row[c] == 0]

    n_drop = int(len(present) * drop_rate)
    if n_drop > 0 and len(present) > 1:
        drop_cols = rng.choice(present, size=min(n_drop, len(present) - 1), replace=False)
        row[drop_cols] = 0

    n_add = int(len(present) * add_rate)
    if n_add > 0 and len(absent) > 0:
        add_cols = rng.choice(absent, size=min(n_add, len(absent)), replace=False)
        row[add_cols] = 1

    return row


def augment_dataset(df, drop_rate=0.2, add_rate=0.1, copies=3, seed=42):
    rng = np.random.default_rng(seed)
    symptom_cols = [c for c in df.columns if c != "prognosis"]

    augmented_rows = [df.copy()]

    for _ in range(copies):
        noisy = df.apply(lambda r: augment_row(r, symptom_cols, drop_rate, add_rate, rng), axis=1)
        augmented_rows.append(noisy)

    augmented_df = pd.concat(augmented_rows, ignore_index=True)
    augmented_df = augmented_df.drop_duplicates().reset_index(drop=True)
    return augmented_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--add_rate", type=float, default=0.1)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/Training_augmented.csv")
    args = parser.parse_args()

    df = load_training()
    aug_df = augment_dataset(df, args.drop_rate, args.add_rate, args.copies)
    aug_df.to_csv(args.out, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Augmented rows: {len(aug_df)}")
    print(f"Saved to {args.out}")