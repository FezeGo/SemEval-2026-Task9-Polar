import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from subtask2.src.config import (
    TRAIN_DIR,
    MODELS_DIR,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_THRESHOLD_CONFIG,
    LABEL_COLUMNS,
    NUM_LABELS,
)
from subtask2.src.data import (
    load_all_languages_data,
    preprocess_dataframe,
    create_language_stratified_split,
)
from subtask2.src.datasets import PolarizationDataset
from subtask2.src.eval_utils import compute_metrics, tune_thresholds_per_label


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(train_df, label_cols):
    eps = 1e-6
    total = len(train_df)
    pos = train_df[label_cols].sum(axis=0).values.astype(np.float32)
    neg = total - pos
    pw = neg / (pos + eps)
    return torch.tensor(pw, dtype=torch.float32)

class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight.to(logits.device),
        )

        return (loss, outputs) if return_outputs else loss


def main():
    cfg = DEFAULT_TRAINING_CONFIG
    loss_cfg = DEFAULT_LOSS_CONFIG
    th_cfg = DEFAULT_THRESHOLD_CONFIG

    set_seed(cfg.seed)

    run_name = f"{Path(cfg.model_name).name}_{loss_cfg.loss_type}_bs{cfg.batch_size}_lr{cfg.learning_rate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = MODELS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Run name: {run_name}")
    print(f"[INFO] Output dir: {output_dir}")

    train_raw = load_all_languages_data(TRAIN_DIR, split="train")
    train_df = preprocess_dataframe(train_raw, split="train")
    train_df, val_df = create_language_stratified_split(
        train_df, val_size=cfg.val_split, seed=cfg.seed
    )


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


    train_dataset = PolarizationDataset(train_df, tokenizer, cfg.max_length)
    val_dataset = PolarizationDataset(val_df, tokenizer, cfg.max_length)


    hf_config = AutoConfig.from_pretrained(cfg.model_name, num_labels=NUM_LABELS)
    hf_config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=hf_config)

    pos_weight = compute_pos_weight(train_df, LABEL_COLUMNS)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    trainer.save_model(str(output_dir / "checkpoint-best"))
    tokenizer.save_pretrained(str(output_dir / "checkpoint-best"))

    print("[INFO] Tuning per-label thresholds on validation set ...")
    val_out = trainer.predict(val_dataset)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids

    best_thresholds, best_f1s = tune_thresholds_per_label(
        logits=val_logits,
        labels=val_labels,
        label_names=LABEL_COLUMNS,
        grid_min=th_cfg.grid_min,
        grid_max=th_cfg.grid_max,
        grid_steps=th_cfg.grid_steps,
    )

    # Save thresholds & config snapshot
    with open(output_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, ensure_ascii=False, indent=2)

    with open(output_dir / "threshold_f1s.json", "w", encoding="utf-8") as f:
        json.dump(best_f1s, f, ensure_ascii=False, indent=2)

    meta = {
        "run_name": run_name,
        "model_name": cfg.model_name,
        "loss_type": loss_cfg.loss_type,
        "focal_gamma": loss_cfg.focal_gamma,
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "max_length": cfg.max_length,
        "val_split": cfg.val_split,
        "seed": cfg.seed,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Done. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
