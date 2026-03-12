import re
import pandas as pd


def clean_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text_cleaned"] = df["text"].apply(clean_text)
    df = df[df["text_cleaned"] != ""].reset_index(drop=True)
    print(f"Preprocessing: {initial_len} -> {len(df)} (Removed {initial_len - len(df)})")
    return df