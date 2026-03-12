import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.evaluation import (
    sigmoid_numpy,
    ensure_model_list,
    set_models_eval,
    build_eval_dataloader,
    average_ensemble_probabilities,
    format_threshold_for_print,
)
from .constants import LANGUAGE_MAP
from .dataset import PolarizationDataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
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
    label_columns,
    thresholds_by_lang=None,
    batch_size=32,
    max_length=256,
    average_mode="macro",
    print_details=True,
):
    models = ensure_model_list(models)
    set_models_eval(models)

    num_labels = len(label_columns)
    results = {}

    if print_details:
        print("\n" + "=" * 140)
        print(
            f"{'Language':<15} {'Samples':>8} "
            f"{'Macro-P':>10} {'Macro-R':>10} {'Macro-F1':>10} "
            f"{'Micro-F1':>10} {'Thr':>10} {'Pos%':>8}"
        )
        print("-" * 140)

    for lang_code in sorted(dataframe["lang"].unique()):
        lang_df = dataframe[dataframe["lang"] == lang_code].reset_index(drop=True)
        if len(lang_df) == 0:
            continue

        lang_dataset = PolarizationDataset(lang_df, tokenizer, max_length)
        loader = build_eval_dataloader(lang_dataset, tokenizer, batch_size)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                all_labels.append(labels.cpu().numpy())

                probs_avg = average_ensemble_probabilities(models, batch)
                all_probs.append(probs_avg.cpu().numpy())

        all_probs = np.vstack(all_probs).astype(np.float32)
        all_labels = np.vstack(all_labels).astype(np.int64)

        if all_probs.shape[1] != num_labels:
            raise ValueError(f"[{lang_code}] probs dim={all_probs.shape[1]} != num_labels={num_labels}")
        if all_labels.shape[1] != num_labels:
            raise ValueError(f"[{lang_code}] labels dim={all_labels.shape[1]} != num_labels={num_labels}")

        if thresholds_by_lang is None:
            threshold = 0.5
        else:
            threshold = thresholds_by_lang.get(lang_code, 0.5)

        if isinstance(threshold, (list, tuple, np.ndarray)):
            thr_vec = np.asarray(threshold, dtype=np.float32).reshape(1, -1)
            if thr_vec.shape[1] != num_labels:
                raise ValueError(
                    f"Threshold vector length mismatch for {lang_code}: "
                    f"got {thr_vec.shape[1]}, need {num_labels}"
                )
            all_preds = (all_probs >= thr_vec).astype(np.int64)
        else:
            threshold = float(threshold)
            all_preds = (all_probs >= threshold).astype(np.int64)

        macro_p = precision_score(all_labels, all_preds, average=average_mode, zero_division=0)
        macro_r = recall_score(all_labels, all_preds, average=average_mode, zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average=average_mode, zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        pos_rate = float(all_labels.mean() * 100.0) if all_labels.size > 0 else 0.0

        results[lang_code] = {
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "threshold": threshold,
            "samples": int(len(lang_df)),
            "pos_rate": float(pos_rate),
        }

        if print_details:
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
            thr_print = format_threshold_for_print(threshold)
            print(
                f"{lang_name:<15} {len(lang_df):>8,} "
                f"{macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} "
                f"{micro_f1:>10.4f} {thr_print:>10} {pos_rate:>7.2f}%"
            )

    overall_macro_f1 = float(np.mean([v["macro_f1"] for v in results.values()])) if results else 0.0
    overall_micro_f1 = float(np.mean([v["micro_f1"] for v in results.values()])) if results else 0.0

    if print_details:
        print("-" * 140)
        print(
            f"{'AVERAGE':<15} {'':>8} {'':>10} {'':>10} "
            f"{overall_macro_f1:>10.4f} {overall_micro_f1:>10.4f}"
        )
        print("=" * 140 + "\n")

    return results, overall_macro_f1