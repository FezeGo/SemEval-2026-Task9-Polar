# SemEval2026-Task9-Polar

This repository contains our implementation for **SemEval 2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization**.

🔗 Official task website:  
https://polar-semeval.github.io

Our system adopts a **unified multilingual modeling framework** based on **mDeBERTa-v3**, supporting more than 20 languages and addressing multiple subtasks under the polarization detection setting.

---

## 📌 Task Overview

SemEval-2026 Task 9 focuses on detecting **polarization in online social media posts** under multilingual, multicultural, and multievent scenarios.

The task consists of multiple subtasks:

- **Subtask 1**: Polarization Detection (Binary Classification)
- **Subtask 2**: Polarization Type Classification (Multi-label Classification)

Key challenges include:

- Strong **cross-lingual distribution shift**
- **Low-resource languages**
- Cultural and event diversity
- Severe **label imbalance**
- Implicit and nuanced polarization expressions

---

## 🌍 Supported Languages

The official dataset includes the following 22 languages:

- Amharic (`amh`)
- Arabic (`arb`)
- Bengali (`ben`)
- German (`deu`)
- English (`eng`)
- Persian (`fas`)
- Hausa (`hau`)
- Hindi (`hin`)
- Italian (`ita`)
- Khmer (`khm`)
- Burmese (`mya`)
- Nepali (`nep`)
- Oriya (`ori`)
- Punjabi (`pan`)
- Polish (`pol`)
- Russian (`rus`)
- Spanish (`spa`)
- Swahili (`swa`)
- Telugu (`tel`)
- Turkish (`tur`)
- Urdu (`urd`)
- Chinese (`zho`)

---

## 🧠 Model Overview

We adopt a **unified multilingual training strategy** shared across all subtasks:

- **Backbone**: `microsoft/mdeberta-v3-base`
- **Training**: Joint multilingual training
- **Encoder sharing** across languages and tasks
- **Evaluation metric**: Macro-F1 (official Codabench metric)

Common design principles:

- Language-agnostic representation learning
- Robust text normalization
- Careful handling of class imbalance

---

## 📍 Subtask 1: Polarization Detection

### 🎯 Task Definition

Subtask 1 aims to determine whether a given post expresses **polarization**.

- `0`: non-polarized  
- `1`: polarized  

This is a **binary classification task**.

---

### 🔧 Modeling Strategy

- **Architecture**: mDeBERTa-v3-base + linear classifier
- **Loss**: Binary Cross-Entropy
- **Training**: Unified multilingual training across all languages
- **Split**: Language- and label-stratified train/validation split

---

### 📊 Evaluation

- **Metric**: Macro-F1
- **Analysis**:
  - Per-language performance
  - Polarization rate per language

---

## 📍 Subtask 2: Polarization Type Classification

### 🎯 Task Definition

Subtask 2 focuses on identifying the **target/type of polarization** in a post.

This is a **multi-label classification task**, where each post may belong to **multiple categories simultaneously**.

---

### 🏷️ Labels

| Label | Description |
|------|------------|
| Political | Political or ideological polarization |
| Racial/Ethnic | Racial or ethnic polarization |
| Religious | Religious polarization |
| Gender/Sexual | Gender identity or sexual orientation polarization |
| Other | Other types (e.g., media, economy, institutions) |

Each instance is represented as a **5-dimensional binary vector**.

---

### 🔧 Modeling Strategy

- **Architecture**: mDeBERTa-v3-base + multi-label classification head
- **Activation**: Sigmoid
- **Problem type**: `multi_label_classification`

---

### ⚖️ Loss Functions

To handle severe label imbalance, we experiment with:

- **Weighted Binary Cross-Entropy (BCE)**  
  - Per-label positive class weights
- **Focal Loss (BCE-based)**  
  - Focusing parameter γ = 2.0
  - Combined with per-label positive weights

---

### 📊 Threshold Tuning

Instead of using a fixed threshold (0.5), we apply:

- **Per-label threshold tuning**
- Thresholds optimized on validation Macro-F1
- Applied during inference on the dev/test set

---

### 📤 Submission Format

Predictions are generated **per language**, following the official submission format:

