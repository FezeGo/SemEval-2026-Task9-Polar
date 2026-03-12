import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from src.utils.seed import set_seed
from src.utils.text import preprocess_dataframe
from src.subtask1.dataset import load_all_languages_data


# =========================
# Paths
# =========================
TEST_DIR = "data/test_phase/subtask1/test"
SUBMISSION_DIR = "outputs/subtask1/ensemble_submission"

XLM_R_MODEL_PATH = "checkpoints/subtask1/xlmr_final_model"
MDEBERTA_MODEL_PATH = "checkpoints/subtask1/mdeberta_final_model"


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
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def infer_probs_binary(
    model_path: str,
    dataframe: pd.DataFrame,
    max_length: int = 256,
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = PredictionDataset(dataframe, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Infer: {os.path.basename(model_path)}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits.squeeze(-1)).view(-1)
            probs_all.extend(probs.cpu().numpy().tolist())

    return np.asarray(probs_all, dtype=np.float32)


def main():
    set_seed(SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    test_raw = load_all_languages_data(TEST_DIR, split="test", labelled=False)
    test_df = preprocess_dataframe(test_raw)

    probs_xlmr = infer_probs_binary(
        XLM_R_MODEL_PATH,
        test_df,
        max_length=MAX_LENGTH,
        batch_size=BS_XLMR,
    )
    probs_mdeberta = infer_probs_binary(
        MDEBERTA_MODEL_PATH,
        test_df,
        max_length=MAX_LENGTH,
        batch_size=BS_MDEBERTA,
    )

    assert len(probs_xlmr) == len(test_df)
    assert len(probs_mdeberta) == len(test_df)

    if ENSEMBLE_METHOD == "simple":
        probs_ens = (probs_xlmr + probs_mdeberta) / 2.0
    else:
        probs_ens = XLM_R_WEIGHT * probs_xlmr + MDEBERTA_WEIGHT * probs_mdeberta

    preds = (probs_ens >= THRESHOLD).astype(np.int64)

    test_df["prediction"] = preds
    test_df["probability"] = probs_ens

    for lang_code in sorted(test_df["lang"].unique()):
        out_df = test_df[test_df["lang"] == lang_code][["id", "prediction"]].copy()
        out_df.columns = ["id", "polarization"]

        out_path = os.path.join(SUBMISSION_DIR, f"pred_{lang_code}.csv")
        out_df.to_csv(out_path, index=False)

        n_pos = int(out_df["polarization"].sum())
        total = len(out_df)
        ratio = (n_pos / total * 100.0) if total > 0 else 0.0
        print(f"{lang_code}: saved {out_path} | N={total:>6} | Pol={n_pos:>5} ({ratio:.2f}%)")

    print(f"\nEnsemble submission saved to: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()