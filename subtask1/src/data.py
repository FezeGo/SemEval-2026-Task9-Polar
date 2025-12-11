import os
import re
from math import ceil
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



from .config import LANGUAGE_MAP


def clean_text(text: str) -> str:

    if pd.isna(text) or text is None:
        return ""

    text = str(text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def load_single_language_data(
    lang_code: str,
    data_dir: str | os.PathLike,
    split: str = "train",
) -> pd.DataFrame:
    
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

        return df

    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return pd.DataFrame()


def load_all_languages_data(
    data_dir: str | os.PathLike,
    split: str = "train",
) -> pd.DataFrame:

    data_dir = os.fspath(data_dir)
    all_data = []

    for lang_code in LANGUAGE_MAP.keys():
        df = load_single_language_data(lang_code, data_dir, split=split)

        if df.empty:
            continue

        all_data.append(df)

        lang_name = LANGUAGE_MAP[lang_code]
        if split == "train" and "polarization" in df.columns:
            pol_count = df["polarization"].sum()
            total = len(df)
            pol_pct = (pol_count / total * 100) if total > 0 else 0
            print(
                f"{lang_name:12s} ({lang_code}): "
                f"{total:5d} samples, {pol_count:5d} polarized ({pol_pct:5.2f}%)"
            )
        else:
            print(f"{lang_name:12s} ({lang_code}): {len(df):5d} samples")

    if not all_data:
        raise ValueError(f"No data loaded from {data_dir}")

    combined_df = pd.concat(all_data, ignore_index=True)

    if split == "train" and "polarization" in combined_df.columns:
        total_pol = combined_df["polarization"].sum()
        total = len(combined_df)
        pol_pct = (total_pol / total * 100) if total > 0 else 0
        print(
            f"\n[INFO] Total train samples: {total}, "
            f"polarized: {total_pol} ({pol_pct:.2f}%)"
        )

    return combined_df


def preprocess_dataframe(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:

    df = df.dropna(subset=["text"])

    df = df[df["text"].astype(str).str.strip() != ""]

    df = df.copy()
    df["text_cleaned"] = df["text"].apply(clean_text)

    df = df[df["text_cleaned"].astype(str).str.strip() != ""]

    df = df.reset_index(drop=True)
    return df

# def create_stratified_split(
#     df: pd.DataFrame,
#     val_size: float = 0.15,
#     seed: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:

#     if "lang" not in df.columns or "polarization" not in df.columns:
#         raise ValueError("DataFrame must contain 'lang' and 'polarization' columns.")

#     df = df.copy()
#     df["polarization"] = df["polarization"].astype(int)

#     df["strat_col"] = df["lang"] + "_" + df["polarization"].astype(str)

#     train_df, val_df = train_test_split(
#         df,
#         test_size=val_size,
#         random_state=seed,
#         stratify=df["strat_col"],
#     )

#     train_df = train_df.drop(columns=["strat_col"]).reset_index(drop=True)
#     val_df = val_df.drop(columns=["strat_col"]).reset_index(drop=True)

#     print("\n[INFO] Per-language polarization rate (train vs. val):")
#     for lang_code in sorted(df["lang"].unique()):
#         train_lang = train_df[train_df["lang"] == lang_code]
#         val_lang = val_df[val_df["lang"] == lang_code]

#         train_pol_pct = train_lang["polarization"].mean() * 100 if len(train_lang) > 0 else 0
#         val_pol_pct = val_lang["polarization"].mean() * 100 if len(val_lang) > 0 else 0

#         print(
#             f"{lang_code:>3s} | "
#             f"train: {len(train_lang):5d} samples, {train_pol_pct:5.2f}% polarized | "
#             f"val: {len(val_lang):5d} samples, {val_pol_pct:5.2f}% polarized"
#         )

#     return train_df, val_df


def create_stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if "lang" not in df.columns or "polarization" not in df.columns:
        raise ValueError("DataFrame must contain 'lang' and 'polarization' columns.")

    df = df.copy()
    df["polarization"] = df["polarization"].astype(int)

    df["strat_col"] = df["lang"] + "_" + df["polarization"].astype(str)

    n_samples = len(df)
    n_strata = df["strat_col"].nunique()
    test_size_abs = ceil(n_samples * val_size)

    use_stratify = test_size_abs >= n_strata

    if not use_stratify:
        print(
            f"[WARN] Data too small for stratified split "
            f"(test_size={test_size_abs}, n_strata={n_strata}). "
            f"Falling back to random split without stratify."
        )
        stratify_col = None
    else:
        stratify_col = df["strat_col"]

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_col,
    )

    train_df = train_df.drop(columns=["strat_col"], errors="ignore").reset_index(drop=True)
    val_df = val_df.drop(columns=["strat_col"], errors="ignore").reset_index(drop=True)

    print("\n[INFO] Per-language polarization rate (train vs. val):")
    for lang_code in sorted(df["lang"].unique()):
        train_lang = train_df[train_df["lang"] == lang_code]
        val_lang = val_df[val_df["lang"] == lang_code]

        train_pol_pct = train_lang["polarization"].mean() * 100 if len(train_lang) > 0 else 0
        val_pol_pct = val_lang["polarization"].mean() * 100 if len(val_lang) > 0 else 0

        print(
            f"{lang_code:>3s} | "
            f"train: {len(train_lang):5d} samples, {train_pol_pct:5.2f}% polarized | "
            f"val: {len(val_lang):5d} samples, {val_pol_pct:5.2f}% polarized"
        )

    return train_df, val_df

