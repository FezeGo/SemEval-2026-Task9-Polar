import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from src.utils.seed import set_seed
from src.utils.text import preprocess_dataframe
from src.subtask2.dataset import load_all_languages_data
from src.subtask2.constants import LABEL_COLUMNS, NUM_LABELS


# =========================
# Paths
# =========================
TEST_DIR = "data/test_phase/subtask2/test"
SUBMISSION_DIR = "outputs/subtask2/ensemble_submission"

XLM_R_MODEL_PATH = "checkpoints/subtask2/xlmr_final_model"
MDEBERTA_MODEL_PATH = "checkpoints/subtask2/mdeberta_final_model"


# =========================
# Config
# =========================
SEED = 42
MAX_LENGTH = 256

BS_XLMR = 64
BS_MDEBERTA = 128

ENSEMBLE_METHOD = "weighted"   # "simple" or "weighted"
XLM_R_WEIGHT = 0.7
MDEBERTA_WEIGHT = 0.3

THRESHOLD = 0.5


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


def infer_probs_multilabel(
    model_path: str,
    dataframe: pd.DataFrame,
    max_length: int = 256,
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = PredictionDataset(dataframe, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    probs_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Infer: {os.path.basename(model_path)}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits)
            probs_all.append(probs.cpu().numpy())

    return np.vstack(probs_all).astype(np.float32)


def main():
    set_seed(SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    test_raw = load_all_languages_data(TEST_DIR, split="test", labelled=False)
    test_df = preprocess_dataframe(test_raw)

    probs_xlmr = infer_probs_multilabel(
        XLM_R_MODEL_PATH,
        test_df,
        max_length=MAX_LENGTH,
        batch_size=BS_XLMR,
    )
    probs_mdeberta = infer_probs_multilabel(
        MDEBERTA_MODEL_PATH,
        test_df,
        max_length=MAX_LENGTH,
        batch_size=BS_MDEBERTA,
    )

    assert probs_xlmr.shape == (len(test_df), NUM_LABELS)
    assert probs_mdeberta.shape == (len(test_df), NUM_LABELS)

    if ENSEMBLE_METHOD == "simple":
        probs_ens = (probs_xlmr + probs_mdeberta) / 2.0
    else:
        probs_ens = XLM_R_WEIGHT * probs_xlmr + MDEBERTA_WEIGHT * probs_mdeberta

    preds = (probs_ens >= THRESHOLD).astype(np.int64)

    for j, col in enumerate(LABEL_COLUMNS):
        test_df.loc[:, col] = preds[:, j]

    test_df[LABEL_COLUMNS] = test_df[LABEL_COLUMNS].astype(int)

    for lang_code in sorted(test_df["lang"].unique()):
        out_df = test_df[test_df["lang"] == lang_code][["id"] + LABEL_COLUMNS].copy()
        out_path = os.path.join(SUBMISSION_DIR, f"pred_{lang_code}.csv")
        out_df.to_csv(out_path, index=False)

        pos_rate = float(out_df[LABEL_COLUMNS].values.mean() * 100.0)
        print(f"{lang_code}: saved {out_path} | N={len(out_df):>6} | Pos%={pos_rate:>6.2f}%")

    print(f"\nEnsemble submission saved to: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()