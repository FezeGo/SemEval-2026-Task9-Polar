from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class PolarizationDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        if "text_cleaned" in dataframe.columns:
            self.texts = dataframe["text_cleaned"].astype(str).tolist()
        else:
            self.texts = dataframe["text"].astype(str).tolist()

        if "polarization" in dataframe.columns:
            self.labels: Optional[list[int]] = (
                dataframe["polarization"].astype(int).tolist()
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class PredictionDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        if "text_cleaned" in dataframe.columns:
            self.texts = dataframe["text_cleaned"].astype(str).tolist()
        else:
            self.texts = dataframe["text"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
