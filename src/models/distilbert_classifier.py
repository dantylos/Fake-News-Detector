"""
DistilBERT fine-tuning for binary fake/real classification.
Uses HuggingFace Trainer API with early stopping.
Tuned for GTX 1650 with subsampling and fp16.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Path constants
ROOT             = Path(__file__).resolve().parents[2]
SPLITS_DIR       = ROOT / "data" / "splits"
TRANSFORMER_PATH = ROOT / "models" / "distilbert"

# Configuration
TRANSFORMER_MODEL        = "distilbert-base-uncased"
MAX_TOKEN_LENGTH         = 256    # ~first 3-4 paragraphs (news "inverted pyramid")
TRAIN_BATCH_SIZE         = 16     # fits in 4 GB VRAM with 256 tokens + fp16
EVAL_BATCH_SIZE          = 32
NUM_EPOCHS               = 3
LEARNING_RATE            = 2e-5
WEIGHT_DECAY             = 0.01
WARMUP_RATIO             = 0.06
LABEL_SMOOTHING          = 0.1
GRADIENT_ACCUM_STEPS     = 1
RANDOM_SEED              = 42
EARLY_STOP_PATIENCE      = 1
MAX_TRAIN_SAMPLES        = 10_000
MAX_VAL_SAMPLES          = 2_000

LOGS_DIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


class ArticleDataset(Dataset):
    """PyTorch Dataset wrapping HuggingFace tokeniser output dicts."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(eval_pred) -> dict:
    """Callback for HuggingFace Trainer -- returns accuracy and weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def _load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        log.critical(f"Split not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path, engine="pyarrow")
    if "text_truncated" not in df.columns:
        log.critical(f"Column 'text_truncated' missing in {name}.parquet")
        sys.exit(1)
    return df


def _encode_labels(series: pd.Series) -> list[int]:
    return [1 if l == "fake" else 0 for l in series]


def _compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Computes inverse-frequency class weights for imbalanced datasets."""
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        return torch.ones(2)
    weights = len(labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to CrossEntropyLoss."""

    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    log.info("=" * 60)
    log.info("  DistilBERT Fine-Tuning Pipeline")
    log.info("=" * 60)

    train_df = _load_split("train")
    val_df   = _load_split("val")

    # Subsample for GPU memory / training time constraints
    if len(train_df) > MAX_TRAIN_SAMPLES:
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df, train_size=MAX_TRAIN_SAMPLES,
            stratify=train_df["label"], random_state=RANDOM_SEED,
        )
        train_df = train_df.reset_index(drop=True)
        log.info(f"  Subsampled train set to {len(train_df)} articles")
    if len(val_df) > MAX_VAL_SAMPLES:
        from sklearn.model_selection import train_test_split
        val_df, _ = train_test_split(
            val_df, train_size=MAX_VAL_SAMPLES,
            stratify=val_df["label"], random_state=RANDOM_SEED,
        )
        val_df = val_df.reset_index(drop=True)
        log.info(f"  Subsampled val set to {len(val_df)} articles")

    y_train = _encode_labels(train_df["label"])
    y_val   = _encode_labels(val_df["label"])

    cuda_available = torch.cuda.is_available()
    device_msg = "CUDA (GPU)" if cuda_available else "CPU"
    log.info(f"  Device         : {device_msg}")
    log.info(f"  Model          : {TRANSFORMER_MODEL}")
    log.info(f"  Epochs         : {NUM_EPOCHS} (early stopping patience={EARLY_STOP_PATIENCE})")
    log.info(f"  Batch size     : {TRAIN_BATCH_SIZE}")
    log.info(f"  Max tokens     : {MAX_TOKEN_LENGTH}")
    log.info(f"  Train size     : {len(train_df)}")
    log.info(f"  Val size       : {len(val_df)}")

    if not cuda_available:
        log.info("  CPU-only training -- estimated 30-60 min depending on hardware.")

    # Check class imbalance
    counts = np.bincount(y_train, minlength=2)
    imbalance_ratio = counts.max() / counts.sum() if counts.sum() > 0 else 0.5
    use_weighted_loss = imbalance_ratio > 0.60
    class_weights = None

    if use_weighted_loss:
        class_weights = _compute_class_weights(y_train)
        log.info(
            f"  Class imbalance detected ({imbalance_ratio:.1%} majority).  "
            f"Using weighted loss: {class_weights.tolist()}"
        )

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    log.info("Tokenising train set...")
    train_enc = tokenizer(
        train_df["text_truncated"].fillna("").tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    log.info("Tokenising validation set...")
    val_enc = tokenizer(
        val_df["text_truncated"].fillna("").tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    train_dataset = ArticleDataset(train_enc, y_train)
    val_dataset   = ArticleDataset(val_enc, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL, num_labels=2
    )
    model.config.id2label = {0: "real", 1: "fake"}
    model.config.label2id = {"real": 0, "fake": 1}

    TRANSFORMER_PATH.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(TRANSFORMER_PATH),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUM_STEPS,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        learning_rate               = LEARNING_RATE,
        label_smoothing_factor      = LABEL_SMOOTHING,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = cuda_available,
        use_cpu                     = not cuda_available,
        seed                        = RANDOM_SEED,
        logging_steps               = 50,
        report_to                   = "none",
        save_total_limit            = 2,
    )

    TrainerClass = WeightedTrainer if use_weighted_loss else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    if use_weighted_loss:
        trainer_kwargs["class_weights"] = class_weights

    trainer = TrainerClass(**trainer_kwargs)
    trainer.train()

    model.save_pretrained(str(TRANSFORMER_PATH))
    tokenizer.save_pretrained(str(TRANSFORMER_PATH))
    log.info(f"Transformer model saved -> {TRANSFORMER_PATH}")

    # Quick evaluation on test and OOD sets
    eval_results: dict = {}
    model.eval()
    for split_name in ["test", "ood_test"]:
        split_path = SPLITS_DIR / f"{split_name}.parquet"
        if not split_path.exists():
            continue
        split_df = pd.read_parquet(split_path, engine="pyarrow")
        if len(split_df) == 0 or "text_truncated" not in split_df.columns:
            log.info(f"  {split_name}: SKIPPED")
            continue

        texts = split_df["text_truncated"].fillna("").tolist()
        y_true = _encode_labels(split_df["label"])

        all_preds = []
        for i in range(0, len(texts), EVAL_BATCH_SIZE):
            batch = texts[i:i + EVAL_BATCH_SIZE]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_TOKEN_LENGTH,
            )
            if cuda_available:
                enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)

        acc = accuracy_score(y_true, all_preds)
        f1  = f1_score(y_true, all_preds, average="weighted", zero_division=0)
        log.info(f"  {split_name}: acc={acc:.4f}  f1={f1:.4f}")
        report_str = classification_report(
            y_true, all_preds, target_names=["real", "fake"], zero_division=0
        )
        log.info("\n" + report_str)
        eval_results[split_name] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}

    import json
    from datetime import datetime, timezone
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metric_log = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "model":         "distilbert-base-uncased (fine-tuned v2)",
        "config": {
            "epochs":               NUM_EPOCHS,
            "learning_rate":        LEARNING_RATE,
            "batch_size":           TRAIN_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUM_STEPS,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS,
            "label_smoothing":      LABEL_SMOOTHING,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "max_token_length":     MAX_TOKEN_LENGTH,
            "fp16":                 cuda_available,
        },
        "results": eval_results,
    }
    log_path = LOGS_DIR / "distilbert_metrics.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metric_log, f, indent=2)
    log.info(f"  Metric log saved -> {log_path}")

    log.info("DistilBERT training complete.  Next: python -m src.evaluation.evaluate")


if __name__ == "__main__":
    main()
