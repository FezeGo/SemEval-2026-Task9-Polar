import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def ensure_model_list(models):
    if isinstance(models, (list, tuple)):
        return list(models)
    return [models]


def set_models_eval(models) -> None:
    for model in models:
        model.eval()


def build_eval_dataloader(dataset, tokenizer, batch_size: int) -> DataLoader:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )


def average_ensemble_probabilities(models, batch: dict) -> torch.Tensor:
    probs_sum = None

    for model in models:
        batch_on_device = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch_on_device)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

        if logits.ndim > 1 and logits.shape[-1] == 1:
            probs = probs.squeeze(-1)

        probs_sum = probs if probs_sum is None else (probs_sum + probs)

    return (probs_sum / len(models)).detach()


def format_threshold_for_print(threshold) -> str:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        arr = np.asarray(threshold, dtype=np.float32).flatten().tolist()
        return "[" + ",".join(f"{x:.2f}" for x in arr) + "]"
    return f"{float(threshold):.2f}"