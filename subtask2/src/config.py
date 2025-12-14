from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "dataset" / "dev_phase" / "subtask2"
TRAIN_DIR = DATA_ROOT / "train"
DEV_DIR = DATA_ROOT / "dev"

OUTPUT_ROOT = PROJECT_ROOT / "subtask2" / "outputs"
MODELS_DIR = OUTPUT_ROOT / "models"
LOGS_DIR = OUTPUT_ROOT / "logs"
SUBMISSIONS_DIR = OUTPUT_ROOT / "submissions"

LANGUAGE_MAP = {
    "amh": "Amharic",
    "arb": "Arabic",
    "ben": "Bengali",
    "deu": "German",
    "eng": "English",
    "fas": "Persian",
    "hau": "Hausa",
    "hin": "Hindi",
    "ita": "Italian",
    "khm": "Khmer",
    "mya": "Burmese",
    "nep": "Nepali",
    "ori": "Oriya",
    "pan": "Punjabi",
    "pol": "Polish",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "tel": "Telugu",
    "tur": "Turkish",
    "urd": "Urdu",
    "zho": "Chinese",
}

LABEL_COLUMNS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]

NUM_LABELS = len(LABEL_COLUMNS)

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "microsoft/mdeberta-v3-base"
    max_length: int = 256

    # Optimizaion
    batch_size: int = 24
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"

    # General
    seed: int = 42
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    logging_steps = 200

    # Validation
    val_split: float = 0.15
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True

@dataclass
class ThresholdConfig:

    strategy: str = "per_label"

    # Used when strategy == "fixed"
    fixed_threshold: float = 0.5

    # Used when strategy == "per_label"
    grid_min: float = 0.1
    grid_max: float = 0.9
    grid_steps: int = 17



DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_THRESHOLD_CONFIG = ThresholdConfig()