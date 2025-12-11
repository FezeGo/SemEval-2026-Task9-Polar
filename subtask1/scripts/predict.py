import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from subtask1.src.config import (
    DEV_DIR,
    SUBMISSIONS_DIR,
    DEFAULT_TRAINING_CONFIG,
    LANGUAGE_MAP,
    MODELS_DIR,
)
from subtask1.src.data import (
    load_all_languages_data,
    preprocess_dataframe,
)
from subtask1.src.datasets import PredictionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prediction on dev set for SemEval 2026 Task 9 - Subtask 1"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the trained run (used to locate model directory under MODELS_DIR).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.batch_size * 2,
        help="Batch size for prediction (default: 2x train batch size).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.max_length,
        help=f"Max sequence length (default: {DEFAULT_TRAINING_CONFIG.max_length}).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = MODELS_DIR / args.run_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_dir = SUBMISSIONS_DIR / args.run_name
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    print("[INFO] Loading dev data from:", DEV_DIR)
    dev_data_raw = load_all_languages_data(DEV_DIR, split="dev")
    dev_data_processed = preprocess_dataframe(dev_data_raw, split="dev")

    if "polarization" in dev_data_processed.columns:
        print("[WARN] 'polarization' column found in dev, dropping it for prediction.")
        dev_data_processed = dev_data_processed.drop(columns=["polarization"])

    print(
        f"[INFO] Dev samples: {len(dev_data_processed)}, "
        f"languages: {dev_data_processed['lang'].nunique()}"
    )

    dev_dataset = PredictionDataset(
        dataframe=dev_data_processed,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("[INFO] Starting prediction on dev set...")
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())

    if len(all_preds) != len(dev_data_processed):
        raise RuntimeError(
            f"Number of predictions ({len(all_preds)}) does not match number of samples ({len(dev_data_processed)})."
        )

    dev_data_processed = dev_data_processed.copy()
    dev_data_processed["prediction"] = all_preds

    total_polarized = int(sum(all_preds))
    total_samples = len(all_preds)
    pol_pct = total_polarized / total_samples * 100 if total_samples > 0 else 0.0

    print(
        f"[INFO] Predictions generated: {total_samples} samples, "
        f"polarized: {total_polarized} ({pol_pct:.2f}%)"
    )

    print("\n" + "=" * 70)
    print("Per-Language Prediction Statistics")
    print("=" * 70)
    print(f"{'Language':<12} {'Samples':>8} {'Pred Polarized':>15} {'Pred Pol%':>12}")
    print("-" * 70)

    for lang_code in sorted(dev_data_processed["lang"].unique()):
        lang_df = dev_data_processed[dev_data_processed["lang"] == lang_code]
        n_samples = len(lang_df)
        n_polarized = int(lang_df["prediction"].sum())
        pol_pct_lang = n_polarized / n_samples * 100 if n_samples > 0 else 0.0

        lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
        print(
            f"{lang_name:<12} "
            f"{n_samples:8d} "
            f"{n_polarized:15d} "
            f"{pol_pct_lang:11.2f}%"
        )

    print("=" * 70 + "\n")

    print("[INFO] Generating per-language submission files...")
    files_generated = []

    for lang_code in sorted(dev_data_processed["lang"].unique()):
        lang_df = dev_data_processed[dev_data_processed["lang"] == lang_code][
            ["id", "prediction"]
        ].copy()

        lang_df.columns = ["id", "polarization"]

        out_file = os.path.join(output_dir, f"pred_{lang_code}.csv")
        lang_df.to_csv(out_file, index=False)

        files_generated.append(out_file)
        print(f"  Generated: pred_{lang_code}.csv ({len(lang_df)} samples)")

    print("\n" + "=" * 70)
    print("[INFO] ALL PREDICTION FILES GENERATED")
    print("=" * 70)
    print(f"[INFO] Submission files location:\n  {output_dir}\n")


if __name__ == "__main__":
    main()
