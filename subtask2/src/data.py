import os
import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LANGUAGE_MAP, LABEL_COLUMNS


def clean_text(text: str) -> str:
    """
    Basic text normalization for social media posts.
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # Normalize user mentions and URLs
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def load_single_language_data(
    lang_code: str,
    data_dir: str | os.PathLike,
    split: str = "train",
) -> pd.DataFrame:
    """
    Load data for a single language.
    For train split, ensure multi-label columns exist and are int.
    """
    data_dir = os.fspath(data_dir)
    file_path = os.path.join(data_dir, f"{lang_code}.csv")

    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df["lang"] = lang_code

        if "id" not in df.columns or "text" not in df.columns:
            print(f"[WARN] Missing 'id' or 'text' in {file_path}")
            return pd.DataFrame()

        # Train split: ensure label columns are present and clean
        if split == "train":
            missing_labels = [c for c in LABEL_COLUMNS if c not in df.columns]
            if missing_labels:
                print(f"[WARN] Missing label columns {missing_labels} in {file_path}")
            else:
                df[LABEL_COLUMNS] = (
                    df[LABEL_COLUMNS]
                    .fillna(0)
                    .astype(int)
                )

        return df

    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return pd.DataFrame()


def load_all_languages_data(
    data_dir: str | os.PathLike,
    split: str = "train",
) -> pd.DataFrame:
    """
    Load and concatenate all languages.
    Prints basic statistics for sanity checking.
    """
    data_dir = os.fspath(data_dir)
    all_data = []

    print(f"\n{'='*60}")
    print(f"Loading {split.upper()} data from: {data_dir}")
    print(f"{'='*60}\n")

    for lang_code in LANGUAGE_MAP.keys():
        df = load_single_language_data(lang_code, data_dir, split=split)

        if df.empty:
            continue

        all_data.append(df)
        lang_name = LANGUAGE_MAP[lang_code]

        if split == "train" and all(c in df.columns for c in LABEL_COLUMNS):
            total = len(df)
            positives = df[LABEL_COLUMNS].sum()
            pos_info = " | ".join(
                [f"{c}:{int(positives[c]):4d}" for c in LABEL_COLUMNS]
            )
            print(f"{lang_name:12s} ({lang_code}): {total:5d} samples | {pos_info}")
        else:
            print(f"{lang_name:12s} ({lang_code}): {len(df):5d} samples")

    if not all_data:
        raise ValueError(f"No data loaded from {data_dir}")

    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"Total {split} samples: {len(combined_df):,}")

    if split == "train" and all(c in combined_df.columns for c in LABEL_COLUMNS):
        total = len(combined_df)
        positives = combined_df[LABEL_COLUMNS].sum()
        print("Label distribution (total positives / percentage):")
        for col in LABEL_COLUMNS:
            count = int(positives[col])
            pct = (count / total) * 100 if total > 0 else 0.0
            print(f"  {col:15s}: {count:6d} ({pct:5.2f}%)")

    print(f"{'='*60}\n")

    return combined_df

def preprocess_dataframe(
    df: pd.DataFrame,
    split: str = "train",
) -> pd.DataFrame:
    """
    Clean and normalize text column.
    """
    # Drop NaN or empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    df = df.copy()
    df["text_cleaned"] = df["text"].apply(clean_text)

    # Drop empty after cleaning
    df = df[df["text_cleaned"].astype(str).str.strip() != ""]

    return df.reset_index(drop=True)

def create_language_stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a train/validation split stratified by language only.
    This is the recommended strategy for multi-label subtask2.
    """
    if "lang" not in df.columns:
        raise ValueError("DataFrame must contain 'lang' column for stratification.")

    print(f"\n[INFO] Creating language-stratified split "
          f"(val_size={val_size:.2f}, seed={seed})")

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["lang"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"[INFO] Train size: {len(train_df):,}")
    print(f"[INFO] Val size:   {len(val_df):,}")

    # Optional sanity check: per-language counts
    print("\n[INFO] Per-language sample counts (train / val):")
    for lang_code in sorted(df["lang"].unique()):
        n_train = (train_df["lang"] == lang_code).sum()
        n_val = (val_df["lang"] == lang_code).sum()
        print(f"  {lang_code:>3s}: train={n_train:5d}, val={n_val:5d}")

    return train_df, val_df
