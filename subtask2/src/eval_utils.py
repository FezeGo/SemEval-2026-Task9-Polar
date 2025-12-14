from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from .config import LABEL_COLUMNS, LANGUAGE_MAP


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    labels = labels.astype(int)

    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {"macro_precision": precision, "macro_recall": recall, "macro_f1": f1}



def tune_thresholds_per_label(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = LABEL_COLUMNS,
    grid_min: float = 0.1,
    grid_max: float = 0.9,
    grid_steps: int = 17,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    
    probs = sigmoid(logits)
    thresholds = np.linspace(grid_min, grid_max, grid_steps)

    best_thresholds: Dict[str, float] = {}
    best_f1s: Dict[str, float] = {}

    for i, label in enumerate(label_names):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        best_f1 = -1.0
        best_t = 0.5

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[label] = float(best_t)
        best_f1s[label] = float(best_f1)

    return best_thresholds, best_f1s


def apply_thresholds(
    logits: np.ndarray,
    thresholds: Dict[str, float],
    label_names: List[str] = LABEL_COLUMNS,
) -> np.ndarray:

    probs = sigmoid(logits)
    preds = np.zeros_like(probs, dtype=int)

    for i, label in enumerate(label_names):
        t = thresholds.get(label, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    return preds


def predict_with_thresholds(
    trainer,
    dataset,
    thresholds: Dict[str, float],
) -> np.ndarray:

    output = trainer.predict(dataset)
    logits = output.predictions

    preds = apply_thresholds(logits, thresholds)
    return preds


def evaluate_by_language(
    dataframe,
    preds: np.ndarray,
    label_names: List[str] = LABEL_COLUMNS,
) -> Dict[str, Dict[str, float]]:

    results = {}

    for lang_code in sorted(dataframe["lang"].unique()):
        idx = dataframe["lang"] == lang_code
        y_true = dataframe.loc[idx, label_names].values
        y_pred = preds[idx]

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        per_label_f1 = {
            label: f1_score(
                y_true[:, i],
                y_pred[:, i],
                zero_division=0
            )
            for i, label in enumerate(label_names)
        }

        results[lang_code] = {
            "language": LANGUAGE_MAP.get(lang_code, lang_code),
            "macro_f1": macro_f1,
            **per_label_f1,
        }

    return results
