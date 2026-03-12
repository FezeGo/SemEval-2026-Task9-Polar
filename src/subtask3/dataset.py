import torch
import pandas as pd
from torch.utils.data import Dataset

from src.utils.io import build_language_file_path, ensure_data_dir, read_csv_if_exists
from src.utils.text import preprocess_dataframe
from .constants import LANGUAGE_MAP, LABEL_COLUMNS


def load_single_language_data(lang_code: str, data_dir: str, labelled: bool) -> pd.DataFrame:
    file_path = build_language_file_path(data_dir, lang_code)
    df = read_csv_if_exists(file_path)

    if df.empty:
        return pd.DataFrame()

    df["lang"] = lang_code

    if labelled:
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns {missing} in {file_path}")
        return df[["id", "text", "lang"] + LABEL_COLUMNS]

    return df[["id", "text", "lang"]]


def load_all_languages_data(data_dir: str, split: str, labelled: bool = True) -> pd.DataFrame:
    ensure_data_dir(data_dir)

    all_data = []
    print(f"Loading [{split}] data from: {data_dir}")

    for lang_code in LANGUAGE_MAP.keys():
        df = load_single_language_data(lang_code, data_dir, labelled=labelled)
        if df is None or df.empty:
            continue
        all_data.append(df)

    if not all_data:
        raise ValueError(f"No data loaded from {data_dir} with split={split}")

    return pd.concat(all_data, ignore_index=True)


class PolarizationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.texts = dataframe["text_cleaned"].tolist()

        if all(col in dataframe.columns for col in LABEL_COLUMNS):
            self.labels = dataframe[LABEL_COLUMNS].values.astype("float32")
        else:
            self.labels = None

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


class PredictionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.texts = dataframe["text_cleaned"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )


def prepare_dataframe(data_dir: str, split: str, labelled: bool = True) -> pd.DataFrame:
    df = load_all_languages_data(data_dir=data_dir, split=split, labelled=labelled)
    return preprocess_dataframe(df)