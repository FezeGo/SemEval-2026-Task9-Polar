from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "dataset" / "dev_phase" / "subtask1"
TRAIN_DIR = DATA_ROOT / "train"
DEV_DIR = DATA_ROOT / "dev"

OUTPUT_ROOT = PROJECT_ROOT / "subtask1" / "outputs"
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

@dataclass
class TrainingConfig:
    model_name: str = "microsoft/mdeberta-v3-base"
    max_length: int = 256
    batch_size: int = 24
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 5
    val_split: float = 0.15
    seed: int = 42
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2
    label_smoothing_factor: float = 0.1
    fp16: bool = True
    lr_scheduler_type: str = "linear"

DEFAULT_TRAINING_CONFIG = TrainingConfig()