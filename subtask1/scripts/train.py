import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from subtask1.src.config import (
    TRAIN_DIR,
    DEV_DIR,
    MODELS_DIR,
    LOGS_DIR,
    DEFAULT_TRAINING_CONFIG,
)
from subtask1.src.data import (
    load_all_languages_data,
    preprocess_dataframe,
    create_stratified_split,
)
from subtask1.src.datasets import PolarizationDataset
from subtask1.src.eval_utils import compute_metrics, evaluate_by_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train mDeBERTa-v3-base for SemEval 2026 Task 9 - Subtask 1"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="A name for this training run (used for output_dir, logging, etc.).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.seed,
        help=f"Random seed (default: {DEFAULT_TRAINING_CONFIG.seed})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_TRAINING_CONFIG.model_name,
        help=f"HF model name or path (default: {DEFAULT_TRAINING_CONFIG.model_name})",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.max_length,
        help=f"Max sequence length (default: {DEFAULT_TRAINING_CONFIG.max_length})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.batch_size,
        help=f"Per-device train batch size (default: {DEFAULT_TRAINING_CONFIG.batch_size})",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_epochs,
        help=f"Number of training epochs (default: {DEFAULT_TRAINING_CONFIG.num_epochs})",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.learning_rate,
        help=f"Learning rate (default: {DEFAULT_TRAINING_CONFIG.learning_rate})",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------
    # Prepare output dirs
    # -----------------------------
    run_output_dir = MODELS_DIR / args.run_name
    run_log_dir = LOGS_DIR / args.run_name

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    # -----------------------------
    # Fix random seed
    # -----------------------------
    set_seed(args.seed)

    # -----------------------------
    # Load data
    # -----------------------------
    print("[INFO] Loading training data from:", TRAIN_DIR)
    train_data_raw = load_all_languages_data(TRAIN_DIR, split="train")
    train_data_processed = preprocess_dataframe(train_data_raw, split="train")

    # Stratified split by lang + polarization
    print("[INFO] Creating stratified train/val split...")
    train_df, val_df = create_stratified_split(
        train_data_processed,
        val_size=DEFAULT_TRAINING_CONFIG.val_split,
        seed=args.seed,
    )

    print(f"[INFO] Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # -----------------------------
    # Load tokenizer & model
    # -----------------------------
    print(f"[INFO] Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # -----------------------------
    # Build datasets
    # -----------------------------
    train_dataset = PolarizationDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = PolarizationDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------------
    # TrainingArguments
    # -----------------------------
    cfg = DEFAULT_TRAINING_CONFIG

    ls_factor = getattr(cfg, "label_smoothing_factor", 0.0)
    if isinstance(ls_factor, (tuple, list)):
        ls_factor = float(ls_factor[0])

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        run_name=args.run_name,

        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_strategy="steps",
        logging_steps=200,
        logging_dir=str(run_log_dir),

        fp16=cfg.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=args.seed,
        report_to="none",
        label_smoothing_factor=ls_factor,
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished.")

    # Save final model (best checkpoint already saved due to load_best_model_at_end=True)
    print("[INFO] Saving final model to:", run_output_dir)
    trainer.save_model(run_output_dir)

    # -----------------------------
    # Final evaluation by language
    # -----------------------------
    print("[INFO] Evaluating on validation set by language...")
    _lang_results, overall_f1 = evaluate_by_language(
        model=trainer.model,
        tokenizer=tokenizer,
        dataframe=val_df,
        batch_size=args.batch_size * 2,
        max_length=args.max_length,
        verbose=True,
    )
    print(f"[INFO] Final val overall macro F1 (avg over langs): {overall_f1:.4f}")


if __name__ == "__main__":
    main()
