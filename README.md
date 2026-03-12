# YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization

This repository contains the official implementation of the **YEZE** system for **SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization**.

🔗 Official task website:  
https://polar-semeval.github.io

📦 Official dataset repository:  
https://github.com/Polar-SemEval/data-public/

Our system follows a **unified multilingual framework** based on **XLM-RoBERTa-large** and **mDeBERTa-v3-base**, supporting all official languages and all subtasks under the polarization detection setting.


## 📌 Task Overview

SemEval-2026 Task 9 focuses on detecting **polarization in online social media posts** under multilingual, multicultural, and multievent scenarios.

The task consists of three subtasks:

- **Subtask 1**: Polarization Detection (**binary classification**)
- **Subtask 2**: Polarization Target Type (**multi-label classification**)
- **Subtask 3**: Polarization Manifestation (**multi-label classification**)

Key challenges include:
- strong **cross-lingual distribution shift**
- **low-resource languages**
- cultural and event diversity
- severe **label imbalance**
- implicit and nuanced polarization expressions


## 🌍 Supported Languages

The official dataset includes the following 22 languages:

- Amharic (`amh`)
- Arabic (`arb`)
- Bengali (`ben`)
- Burmese (`mya`)
- Chinese (`zho`)
- English (`eng`)
- German (`deu`)
- Hausa (`hau`)
- Hindi (`hin`)
- Italian (`ita`)
- Khmer (`khm`)
- Nepali (`nep`)
- Odia (`ori`)
- Persian (`fas`)
- Polish (`pol`)
- Punjabi (`pan`)
- Russian (`rus`)
- Spanish (`spa`)
- Swahili (`swa`)
- Telugu (`tel`)
- Turkish (`tur`)
- Urdu (`urd`)

> Note: In the official setting, **Subtasks 1–2** cover all 22 languages. **Subtask 3** is evaluated on **18 languages** (excluding `ita`/`pol`/`rus`/`mya` in the official release).


## 🧠 System Overview

### Backbones
- **XLM-RoBERTa-large** (`xlm-roberta-large`)
- **mDeBERTa-v3-base** (`microsoft/mdeberta-v3-base`)

### Modeling choices (high level)
- **Independent per-subtask models** (no shared multi-task pipeline in the final submission)
- **Binary relevance** decomposition for multi-label subtasks (S2/S3)
- **Imbalance-aware training** (class weighting / weighted BCE)
- **Heterogeneous ensemble** of XLM-R and mDeBERTa via weighted probability averaging
- **Global threshold** for final decisions (per-label threshold tuning was avoided due to overfitting under extreme sparsity)

### Official evaluation metric
- **Macro-F1** (label-wise Macro-F1 for multi-label subtasks)


## 📍 Subtask 1: Polarization Detection (Binary)

### Labels
- `0`: non-polarized  
- `1`: polarized  

### Input/Output format (per language)
- Input: `id,text,...`
- Output: `id,polarization`


## 📍 Subtask 2: Polarization Target Type (Multi-label)

### Labels (5)
| Label | Description |
|------|------------|
| `political` | Political / ideological polarization |
| `racial/ethnic` | Racial or ethnic polarization |
| `religious` | Religious polarization |
| `gender/sexual` | Gender identity / sexual orientation polarization |
| `other` | Other target types |

### Output format (per language)
- Output: `id,political,racial/ethnic,religious,gender/sexual,other` (0/1 per label)


## 📍 Subtask 3: Manifestation Identification (Multi-label)

### Labels (6)
| Label | Description |
|------|------------|
| `stereotype` | Stereotyping / generalizations |
| `vilification` | Vilifying / demonizing |
| `dehumanization` | Dehumanizing language |
| `extreme_language` | Extreme / absolutist / incendiary expressions |
| `lack_of_empathy` | Lack of empathy |
| `invalidation` | Dismissing / invalidating others |

### Output format (per language)
- Output: `id,stereotype,vilification,dehumanization,extreme_language,lack_of_empathy,invalidation` (0/1 per label)


## 📂 Dataset Layout (data-public)

Expected paths (per language CSV):
- `data-public/train/{lang}.csv`
- `data-public/dev/{lang}.csv`
- `data-public/test/{lang}.csv`

Example:
- `data-public/dev/amh.csv`
- `data-public/train/amh.csv`
- `data-public/test/amh.csv`

We align all predictions and gold labels by the `id` column.


## 📤 Submission Format

Predictions are generated **per language**, matching the official format:
- `subtask_1/pred_{lang}.csv`
- `subtask_2/pred_{lang}.csv`
- `subtask_3/pred_{lang}.csv`

Each file must contain the `id` column and the required label columns for the corresponding subtask.


## 🧪 Reproducibility

Environment (as used in our experiments):

- **Python**: `3.11+`
- **Core Packages**:
  - `torch==2.8.0+cu128`
  - `transformers==4.57.3`
  - `tokenizers==0.22.2`
  - `pandas==2.3.2`, `numpy==2.3.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.6`
  - `wandb==0.23.1`

## 📦 Official Data (with Gold Labels)

The organizers released the official dataset (including gold labels for `dev` and `test`) at:

- https://github.com/Polar-SemEval/data-public/

We use the official CSV files directly and perform evaluation by aligning with gold labels via the `id` field.