import torch
import pandas as pd
from torch.utils.data import Dataset

from src.utils.io import build_language_file_path, ensure_data_dir, read_csv_if_exists
from src.utils.text import preprocess_dataframe
from .constants import LANGUAGE_MAP


def load_single_language_data(lang_code: str, data_dir: str) -> pd.DataFrame:
    file_path = build_language_file_path(data_dir, lang_code)
    df = read_csv_if_exists(file_path)

    if df.empty:
        return pd.DataFrame()

    try:
        df["lang"] = lang_code

        if "polarization" in df.columns:
            return df[["id", "text", "polarization", "lang"]]

        return df[["id", "text", "lang"]]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_all_languages_data(data_dir: str, split: str, labelled: bool = True) -> pd.DataFrame:
    ensure_data_dir(data_dir)

    all_data = []
    print(f"Loading [{split}] data from: {data_dir}")

    for lang_code in LANGUAGE_MAP.keys():
        df = load_single_language_data(lang_code, data_dir)
        if df is None or df.empty:
            continue

        if "lang" not in df.columns:
            df["lang"] = lang_code

        if labelled and "polarization" not in df.columns:
            raise ValueError(
                f"Label column 'polarization' not found in {lang_code} {split} data."
            )

        all_data.append(df)

    if not all_data:
        raise ValueError(
            f"No data loaded from {data_dir} with split={split}. Check file paths / naming."
        )

    return pd.concat(all_data, ignore_index=True)


class PolarizationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.texts = dataframe["text_cleaned"].tolist()
        self.labels = (
            dataframe["polarization"].tolist()
            if "polarization" in dataframe.columns
            else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def prepare_dataframe(data_dir: str, split: str, labelled: bool = True) -> pd.DataFrame:
    df = load_all_languages_data(data_dir=data_dir, split=split, labelled=labelled)
    return preprocess_dataframe(df)