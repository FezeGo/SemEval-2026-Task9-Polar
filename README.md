# SemEval2026-Task9-Polar

This repository contains our implementation for **SemEval 2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization**.

🔗 Official task website:  
https://polar-semeval.github.io

Our system focuses on building a **unified multilingual polarization detection model** based on **mDeBERTa-v3**, supporting more than 20 languages.

---

## 📌 Task Description

SemEval-2026 Task 9 aims to detect whether an online post expresses **polarization** under multilingual, multicultural, and multievent settings.  
This is a **binary classification task**:

- `0`: non-polarized  
- `1`: polarized  

The key challenges include:

- Strong **cross-lingual distribution shift**
- **Low-resource languages**
- **Cultural and event diversity**
- **Label imbalance across languages**

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

We adopt a **multilingual unified training strategy** based on:

- **Backbone**: `microsoft/mdeberta-v3-base`
- **Task**: Binary classification
- **Training**: Joint training on all languages
- **Evaluation Metric**: **Macro-F1** (official Codabench metric)

Key features of our system:

- Unified multilingual encoder
- Language-stratified train/validation split
- Robust text normalization and cleaning
- Multi-seed stability experiments
- Logits-based ensemble (explored)
- Language-wise evaluation and error analysis
