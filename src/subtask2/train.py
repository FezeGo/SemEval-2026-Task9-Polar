import math
import os
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from src.config.training_config import TrainingConfig
from src.utils.seed import set_seed
from src.utils.trainer import WeightedBCETrainer

from .constants import NUM_LABELS, LABEL_COLUMNS, ID2LABEL, LABEL2ID
from .dataset import prepare_dataframe, PolarizationDataset
from .metrics import compute_metrics


TRAIN_DIR = "data/test_phase/train"
DEV_DIR = "data/test_phase/dev"
OUTPUT_DIR = "outputs/subtask2"
MODEL_DIR = "checkpoints/subtask2"


def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    return model, tokenizer


def compute_pos_weight(df: pd.DataFrame) -> np.ndarray:
    pos_counts = df[LABEL_COLUMNS].sum(axis=0).values.astype("float64")
    neg_counts = len(df) - pos_counts
    pos_weight_vec = neg_counts / np.clip(pos_counts, 1.0, None)

    print("pos_counts:", dict(zip(LABEL_COLUMNS, pos_counts.tolist())))
    print("pos_weight_vec:", dict(zip(LABEL_COLUMNS, pos_weight_vec.tolist())))
    return pos_weight_vec.astype("float32")


def build_stage1_args(config: TrainingConfig, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        bf16=True,
        tf32=True,
    )


def build_stage2_args(
    config: TrainingConfig,
    output_dir: str,
    num_train_epochs: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=True,
        tf32=True,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
    )


def extract_best_epoch(trainer) -> int:
    state = trainer.state

    if state.best_metric is not None:
        print(f"Best dev metric (macro_f1): {state.best_metric:.4f}")

    if state.best_model_checkpoint is not None:
        print(f"Best checkpoint: {state.best_model_checkpoint}")

    best_epoch = None
    best_metric = state.best_metric

    if best_metric is not None:
        for log in state.log_history:
            if "eval_macro_f1" in log and log["eval_macro_f1"] == best_metric:
                best_epoch = int(math.ceil(log["epoch"]))
                break

    if best_epoch is None:
        best_epoch = int(math.ceil(state.epoch or 1))

    best_epoch = max(best_epoch, 1)
    print(f"Selected best_epoch = {best_epoch}")
    return best_epoch


def train_and_select_best_epoch(config: TrainingConfig) -> int:
    print("\n===== Stage 1: train on TRAIN, select best epoch on DEV =====")

    train_df = prepare_dataframe(TRAIN_DIR, split="train", labelled=True)
    dev_df = prepare_dataframe(DEV_DIR, split="dev", labelled=True)

    model, tokenizer = build_model_and_tokenizer(config.model_name)
    pos_weight_vec = compute_pos_weight(train_df)

    training_args = build_stage1_args(
        config=config,
        output_dir=os.path.join(OUTPUT_DIR, "stage1_train_dev"),
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight_vec,
        model=model,
        args=training_args,
        train_dataset=PolarizationDataset(train_df, tokenizer, config.max_length),
        eval_dataset=PolarizationDataset(dev_df, tokenizer, config.max_length),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("\nStage 1 dev evaluation:")
    print(f"eval_macro_f1: {eval_results.get('eval_macro_f1', 0):.4f}")
    print(f"eval_precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"eval_recall:    {eval_results.get('eval_recall', 0):.4f}")
    print(f"eval_loss:      {eval_results.get('eval_loss', 0):.4f}")

    best_epoch = extract_best_epoch(trainer)

    stage1_best_dir = os.path.join(MODEL_DIR, "stage1_best_model")
    trainer.save_model(stage1_best_dir)
    tokenizer.save_pretrained(stage1_best_dir)
    print(f"Stage 1 best model saved to: {stage1_best_dir}")

    return best_epoch


def retrain_on_full_data(config: TrainingConfig, best_epoch: int) -> None:
    print("\n===== Stage 2: retrain on TRAIN + DEV =====")

    train_df = prepare_dataframe(TRAIN_DIR, split="train", labelled=True)
    dev_df = prepare_dataframe(DEV_DIR, split="dev", labelled=True)
    full_train_df = pd.concat([train_df, dev_df], ignore_index=True)

    final_model, tokenizer = build_model_and_tokenizer(config.model_name)
    pos_weight_final = compute_pos_weight(full_train_df)

    final_args = build_stage2_args(
        config=config,
        output_dir=os.path.join(OUTPUT_DIR, "stage2_train_full"),
        num_train_epochs=best_epoch,
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight_final,
        model=final_model,
        args=final_args,
        train_dataset=PolarizationDataset(full_train_df, tokenizer, config.max_length),
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=None,
        callbacks=[],
    )

    trainer.train()

    final_model_dir = os.path.join(MODEL_DIR, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"Final model saved to: {final_model_dir}")


def main():
    config = TrainingConfig()
    set_seed(config.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_epoch = train_and_select_best_epoch(config)
    retrain_on_full_data(config, best_epoch=best_epoch)


if __name__ == "__main__":
    main()