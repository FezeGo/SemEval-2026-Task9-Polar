from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from .config import LANGUAGE_MAP
from .datasets import PolarizationDataset


def compute_metrics(eval_pred) -> Dict[str, float]:

    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=1)

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if len(f1_per_class) == 2:
        metrics["f1_class_0"] = f1_per_class[0]
        metrics["f1_class_1"] = f1_per_class[1]

    return metrics

def evaluate_by_language(
    model,
    tokenizer,
    dataframe,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float]:

    if "lang" not in dataframe.columns or "polarization" not in dataframe.columns:
        raise ValueError("DataFrame must contain 'lang' and 'polarization' columns for evaluation.")

    if device is None:
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model.to(device)
    model.eval()

    results: Dict[str, Dict[str, float]] = {}

    if verbose:
        print("\n" + "=" * 80)
        print("Per-Language Evaluation")
        print("=" * 80)
        print(f"{'Language':<12} {'Samples':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Pol%':>8}")
        print("-" * 80)

    with torch.no_grad():
        for lang_code in sorted(dataframe["lang"].unique()):
            lang_df = dataframe[dataframe["lang"] == lang_code].reset_index(drop=True)

            if len(lang_df) == 0:
                continue

            dataset = PolarizationDataset(
                dataframe=lang_df,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            all_preds = []
            all_labels = []

            for batch in loader:
                if "labels" not in batch:
                    raise ValueError("PolarizationDataset used for evaluation must provide 'labels'.")

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            pol_rate = float(np.mean(all_labels) * 100.0) if len(all_labels) > 0 else 0.0

            results[lang_code] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "samples": len(lang_df),
                "polarization_rate": pol_rate,
            }

            if verbose:
                lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
                print(
                    f"{lang_name:<12} "
                    f"{len(lang_df):8d} "
                    f"{precision:10.4f} "
                    f"{recall:10.4f} "
                    f"{f1:10.4f} "
                    f"{pol_rate:8.2f}"
                )

    if results:
        overall_macro_f1 = float(np.mean([r["f1"] for r in results.values()]))
    else:
        overall_macro_f1 = 0.0

    if verbose:
        print("-" * 80)
        print(f"{'Overall macro F1 (avg over langs)':<40} {overall_macro_f1:10.4f}")
        print("=" * 80 + "\n")

    return results, overall_macro_f1
