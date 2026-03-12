import torch
import torch.nn as nn
from transformers import Trainer


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss