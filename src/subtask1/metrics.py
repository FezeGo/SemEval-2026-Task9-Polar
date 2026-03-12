import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.evaluation import (
    sigmoid_numpy,
    ensure_model_list,
    set_models_eval,
    build_eval_dataloader,
    average_ensemble_probabilities,
)
from .constants import LANGUAGE_MAP
from .dataset import PolarizationDataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits).squeeze()
    labels = np.asarray(labels)

    probs = sigmoid_numpy(logits)
    preds = (probs >= 0.5).astype(int)

    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def evaluate_by_language(
    models,
    tokenizer,
    dataframe,
    thresholds_by_lang=None,
    batch_size=32,
    max_length=256,
    average_mode="macro",
    print_details=True,
):
    models = ensure_model_list(models)
    set_models_eval(models)

    results = {}

    if print_details:
        print("\n" + "=" * 120)
        print(
            f"{'Language':<15} {'Samples':>8} {'Macro-P':>10} "
            f"{'Macro-R':>10} {'Macro-F1':>10} {'Pos-F1':>10} "
            f"{'Thr':>7} {'Pol%':>8}"
        )
        print("-" * 120)

    for lang_code in sorted(dataframe["lang"].unique()):
        lang_df = dataframe[dataframe["lang"] == lang_code].reset_index(drop=True)
        lang_dataset = PolarizationDataset(lang_df, tokenizer, max_length)
        loader = build_eval_dataloader(lang_dataset, tokenizer, batch_size)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                all_labels.extend(labels.cpu().numpy().tolist())

                probs_avg = average_ensemble_probabilities(models, batch)
                all_probs.extend(probs_avg.cpu().numpy().tolist())

        all_probs = np.asarray(all_probs, dtype=np.float32)
        all_labels = np.asarray(all_labels, dtype=np.int64)

        threshold = 0.5 if thresholds_by_lang is None else float(thresholds_by_lang.get(lang_code, 0.5))
        all_preds = (all_probs >= threshold).astype(np.int64)

        macro_p = precision_score(all_labels, all_preds, average=average_mode, zero_division=0)
        macro_r = recall_score(all_labels, all_preds, average=average_mode, zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average=average_mode, zero_division=0)
        pos_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        pol_rate = (float(all_labels.sum()) / len(all_labels) * 100.0) if len(all_labels) > 0 else 0.0

        results[lang_code] = {
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "pos_f1": float(pos_f1),
            "threshold": float(threshold),
            "samples": int(len(lang_df)),
            "pol_rate": float(pol_rate),
        }

        if print_details:
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
            print(
                f"{lang_name:<15} {len(lang_df):>8,} {macro_p:>10.4f} "
                f"{macro_r:>10.4f} {macro_f1:>10.4f} {pos_f1:>10.4f} "
                f"{threshold:>7.2f} {pol_rate:>7.2f}%"
            )

    overall_macro_f1 = float(np.mean([v["macro_f1"] for v in results.values()])) if results else 0.0

    if print_details:
        print("-" * 120)
        print(f"{'AVERAGE':<15} {'':>8} {'':>10} {'':>10} {overall_macro_f1:>10.4f}")
        print("=" * 120 + "\n")

    return results, overall_macro_f1