from typing import Optional

import torch
import pandas as pd
from torch.utils.data import Dataset

from .config import LABEL_COLUMNS


class PolarizationDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):
        self.texts = dataframe["text_cleaned"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Check whether labels are available (train / val) or not (dev / test)
        has_all_labels = all(col in dataframe.columns for col in LABEL_COLUMNS)
        if has_all_labels:
            # multi-label targets: shape (N, num_labels)
            self.labels: Optional[torch.Tensor] = torch.tensor(
                dataframe[LABEL_COLUMNS].values,
                dtype=torch.float,
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,   # padding is handled by DataCollator
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item
