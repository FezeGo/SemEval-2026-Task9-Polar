import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from subtask2.src.config import (
    DEV_DIR,
    SUBMISSIONS_DIR,
    MODELS_DIR,
    LABEL_COLUMNS,
    NUM_LABELS,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_THRESHOLD_CONFIG,
)
from subtask2.src.data import load_all_languages_data, preprocess_dataframe
from subtask2.src.datasets import PolarizationDataset
from subtask2.src.eval_utils import apply_thresholds


def resolve_checkpoint_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "checkpoint-best").is_dir():
        return path / "checkpoint-best"
    return path


def find_latest_run(models_dir: Path) -> Path:
    runs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No runs found in {models_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def load_thresholds(run_dir: Path) -> Dict[str, float]:
    th_cfg = DEFAULT_THRESHOLD_CONFIG

    candidates = [
        run_dir / "thresholds.json",
        run_dir.parent / "thresholds.json",
    ]

    for c in candidates:
        if c.is_file():
            with open(c, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            for lab in LABEL_COLUMNS:
                thresholds.setdefault(lab, 0.5)
            return {k: float(v) for k, v in thresholds.items()}

    if th_cfg.strategy == "fixed":
        return {lab: float(th_cfg.fixed_threshold) for lab in LABEL_COLUMNS}

    return {lab: 0.5 for lab in LABEL_COLUMNS}


def build_trainer_for_prediction(
    checkpoint_dir: Path,
    per_device_eval_batch_size: int,
) -> tuple[Trainer, AutoTokenizer]:

    cfg = DEFAULT_TRAINING_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    hf_config = AutoConfig.from_pretrained(
        str(checkpoint_dir),
        num_labels=NUM_LABELS,
    )
    hf_config.problem_type = "multi_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir),
        config=hf_config,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    pred_args = TrainingArguments(
        output_dir=str(checkpoint_dir / "_pred_tmp"),
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_drop_last=False,
        fp16=cfg.fp16,
        report_to="none",
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=pred_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer, tokenizer


def save_language_submissions(
    dev_df: pd.DataFrame,
    preds: np.ndarray,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in sorted(dev_df["lang"].unique()):
        idx = dev_df["lang"] == lang
        lang_df = dev_df.loc[idx, ["id"]].copy()
        lang_preds = preds[idx.values]

        out = pd.concat(
            [
                lang_df.reset_index(drop=True),
                pd.DataFrame(lang_preds, columns=LABEL_COLUMNS).astype(int),
            ],
            axis=1,
        )

        out_path = output_dir / f"pred_{lang}.csv"
        out.to_csv(out_path, index=False)

    print(f"[INFO] Saved per-language submissions to: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_or_ckpt_dir",
        type=str,
        default=None,
        help="Run dir or checkpoint-best/. If omitted, use latest run.",
    )
    parser.add_argument(
        "--dev_dir",
        type=str,
        default=str(DEV_DIR),
        help="Dev data directory containing <lang>.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output dir. Default: SUBMISSIONS_DIR/<run_name>/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Eval batch size. Default uses config.eval_batch_size.",
    )
    args = parser.parse_args()

    # Resolve run / checkpoint
    if args.run_or_ckpt_dir is None:
        print("[INFO] --run_or_ckpt_dir not provided, using latest run")
        run_dir = find_latest_run(Path(MODELS_DIR))
        ckpt_dir = resolve_checkpoint_dir(run_dir)
    else:
        ckpt_dir = resolve_checkpoint_dir(Path(args.run_or_ckpt_dir))
        run_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoint-best" else ckpt_dir

    print(f"[INFO] Using checkpoint: {ckpt_dir}")

    cfg = DEFAULT_TRAINING_CONFIG
    eval_bs = args.batch_size if args.batch_size is not None else cfg.eval_batch_size

    # Output dir
    if args.output_dir is None:
        out_dir = Path(SUBMISSIONS_DIR) / run_dir.name
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()

    # Load thresholds
    thresholds = load_thresholds(run_dir)

    # Load dev data
    dev_raw = load_all_languages_data(args.dev_dir, split="dev")
    dev_df = preprocess_dataframe(dev_raw, split="dev")

    # Build trainer
    trainer, tokenizer = build_trainer_for_prediction(
        ckpt_dir,
        per_device_eval_batch_size=eval_bs,
    )
    dev_dataset = PolarizationDataset(dev_df, tokenizer, cfg.max_length)

    # Predict
    pred_out = trainer.predict(dev_dataset)
    dev_logits = pred_out.predictions

    # Apply thresholds
    dev_preds = apply_thresholds(dev_logits, thresholds)

    # Save
    save_language_submissions(dev_df, dev_preds, out_dir)


if __name__ == "__main__":
    main()
