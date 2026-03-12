from dataclasses import dataclass

@dataclass
class TrainingConfig:

    model_name: str = "xlm-roberta-large"   # "microsoft/mdeberta-v3-base"

    max_length: int = 256

    batch_size: int = 32                    # 64

    gradient_accumulation_steps: int = 1

    learning_rate: float = 1e-5             # 2e-5

    num_epochs: int = 4                     # 5

    seed: int = 42

    weight_decay: float = 0.01

    warmup_ratio: float = 0.1

    seed: int = 42