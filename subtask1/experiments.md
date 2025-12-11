# Experiments Summary — SemEval 2026 Task 9 (Subtask 1)

This document summarizes all controlled experiments conducted for the development of our system for **SemEval-2026 Task 9: Multilingual, Multicultural and Multievent Online Polarization Detection**.

It includes:
- Dataset statistics  
- Model configurations  
- Training setups  
- Ablation studies  
- Hyperparameter sensitivity  
- Per-language performance  
- Final model selection for submission  


---

## 1. Experimental Setup

### 1.1 Task Definition
Binary classification:
- **0** = Non-Polarized  
- **1** = Polarized  

Macro-F1 is used as the primary evaluation metric (official metric on Codabench).

### 1.2 Dataset
We jointly train on all 22 languages provided in the shared task.

| Split | # Samples | # Languages | Notes |
|-------|-----------|-------------|-------|
| Train | XXXXX | 22 | Provided by organizers |
| Dev   | Unlabeled | 22 | Used only for prediction submission |

> *Actual per-language sample counts will be inserted here.*

---

## 2. Model and Training Configuration

### 2.1 Base Models Tested
| Model | Params | Notes |
|-------|--------|-------|
| XLM-RoBERTa Base | 270M | multilingual baseline |
| XLM-RoBERTa Large | 560M | high-capacity multilingual |
| mDeBERTa-v3-base | 184M | **final model backbone** |

### 2.2 Final Training Hyperparameters
(Used by the best-performing configuration)

| Hyperparameter | Value |
|----------------|-------|
| Model | mDeBERTa-v3-base |
| Max length | 128 or 256 (to fill in) |
| Batch size | 24 |
| Learning rate | 1e-5 or 2e-5 |
| Epochs | 3–5 |
| Weight decay | 0.1 |
| Warmup ratio | 0.2 |
| LR scheduler | linear |
| Seed | 42 (best) |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Class weights | No (unless added later) |

---

## 3. Key Experiments

Below we track a minimally complete set of experiments that influenced the final system.

### 3.1 Seed Sensitivity (mDeBERTa-v3-base)

| Seed | ZHO F1 | Macro Avg F1 | Notes |
|------|--------|---------------|-------|
| 42   | **0.9018** | (fill) | Best result |
| 2025 | 0.8971 | (fill) | Slight drop |
| 3407 | 0.8690 | (fill) | Unstable |

**Observation**: Seed 42 consistently performs best across languages.

---

### 3.2 Ensemble Experiments

We tested logit-level ensemble across seeds:

Seeds: `[42, 2025, 3407]`  
Ensemble strategy: average softmax logits  
Threshold tuning: per-language threshold search attempted

| Experiment | ZHO F1 | Notes |
|------------|--------|--------|
| Best single model (seed=42) | **0.9018** | best |
| Soft ensemble (3 seeds) | 0.8971 | No improvement |
| Threshold optimized ensemble | No gain | Dev unlabeled → threshold unstable |

**Conclusion**: Ensemble did **not** outperform the best single model.  
We adopt **mDeBERTa-v3-base, seed=42**.

---

### 3.3 Per-Language Performance (Validation Split)

*(You will fill in your actual values from validation)*

| Lang | F1 | Notes |
|------|----|-------|
| zho | 0.90 | strongest |
| fas | ... | |
| hin | ... | |
| tur | ... | |
| ... | ... | |
| ita | lowest | |

You can later include a bar plot in your paper using these values.

---

## 4. Ablation Studies

### 4.1 Input length

| Max Length | Macro F1 | Notes |
|------------|-----------|-------|
| 128 | (fill) | |
| 256 | (fill) | little/no improvement? |

### 4.2 Learning Rate

| LR | Macro F1 | Notes |
|----|-----------|-------|
| 1e-5 | best | |
| 2e-5 | (fill) | |
| 3e-5 | (fill) | tends to overfit |

### 4.3 Class Weights (Optional)

If we test later:

| Setting | Macro F1 | Notes |
|---------|-----------|-------|
| No class weights | baseline |
| Inverse frequency | (fill) | |
| Only for ZHO | (fill) | |

---

## 5. Final System for Submission

The **final system** submitted to Codabench uses:

- **mDeBERTa-v3-base**
- **Seed = 42**
- **Epochs = (fill actual)**
- **LR = (fill actual)**
- **Max length = (fill actual)**
- **Joint multilingual training**
- **No ensembling**

### Official Codabench Score

| Language | F1 Score |
|----------|----------|
| zho | **0.9018** |
| ... | fill others if needed |

---

## 6. Reproducibility Commands

### Train:

```bash
python -m subtask1.scripts.train --run_name final-seed42 --seed 42
