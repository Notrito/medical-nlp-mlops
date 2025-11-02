# Medical Specialty Classification - MLOps Pipeline

> NLP classification system for medical transcriptions using DVC and MLflow

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [MLOps Stack](#mlops-stack)
- [Key Insights](#key-insights)
- [Future Work](#future-work)

---

## Overview

Multi-class text classification pipeline to predict medical specialties from clinical transcriptions. Built with **full MLOps practices** including experiment tracking, pipeline orchestration, and reproducibility.

**Key Features:**
- 🔄 Reproducible pipeline with **DVC**
- 📊 Experiment tracking with **MLflow**
- 🤖 Comparison of general vs domain-specific language models
- 🏥 Medical NLP on real-world clinical data (20 specialties)

**Tech Stack:** Python 3.12, PyTorch, Transformers, DVC, MLflow, scikit-learn

---

## Dataset

**Source:** [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) - Medical transcriptions from Kaggle

**Statistics:**
- Original samples: 4,966 medical transcriptions
- After chunking: 10,606 samples (2000 chars, 10% overlap)
- Classes: 20 medical specialties (top 19 + "Others")
- Split: 80/10/10 (train/val/test) stratified with `random_state=786`

**Class Distribution (Top 5):**
1. Surgery - 1,103 samples
2. Consult - History and Phy. - 516 samples  
3. Cardiovascular / Pulmonary - 372 samples
4. Orthopedic - 355 samples
5. Radiology - 273 samples

---

## Project Structure
```
medical-nlp-ops/
├── data/
│   ├── raw/
│   │   ├── mtsamples.csv           # Original dataset
│   │   └── mtsamples.csv.dvc       # DVC tracking file
│   ├── processed/
│   │   ├── train.csv               # 8,485 samples
│   │   ├── val.csv                 # 1,061 samples
│   │   ├── test.csv                # 1,060 samples
│   │   └── label_mapping.json      # Class encoding (0-19)
│   ├── tokenized/                  # DistilBERT tokenized data
│   ├── tokenized_biobert/          # BioBERT tokenized data
│   └── mtsamples_chunked.csv       # Intermediate chunking output
├── src/
│   ├── config.py                   # Configuration constants
│   ├── data_preparation.py         # Preprocessing & splitting
│   ├── tokenization.py             # Tokenization pipeline
│   └── train.py                    # Training with MLflow tracking
├── scripts/
│   └── download_data.py            # Kaggle dataset downloader
├── models/
│   ├── distilbert/                 # DistilBERT trained model + checkpoints
│   └── biobert/                    # BioBERT trained model + checkpoints
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory data analysis
│   └── 02_tokenize.ipynb          # Tokenization experiments
├── mlruns/                         # MLflow experiment tracking
├── dvc.yaml                        # DVC pipeline stages
├── params.yaml                     # Training hyperparameters
└── requirements.txt                # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.12
- CUDA-compatible GPU (optional, recommended)
- Git & DVC

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Notrito/medical-nlp-mlops.git
cd medical-nlp-mlops
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Pull data with DVC:**
```bash
dvc pull
```

---

## Usage

### Reproduce Full Pipeline

Run the complete DVC pipeline (data preparation → tokenization → training):
```bash
dvc repro
```

### Run Individual Stages

**Data Preparation:**
```bash
python src/data_preparation.py
```

**Tokenization:**
```bash
# DistilBERT
python src/tokenization.py --model distilbert-base-uncased --input data/processed --output data/tokenized

# BioBERT
python src/tokenization.py --model dmis-lab/biobert-base-cased-v1.2 --input data/processed --output data/tokenized_biobert
```

**Training:**
```bash
# DistilBERT
python src/train.py --model distilbert-base-uncased --data_dir data/tokenized --output_dir models/distilbert --epochs 6 --batch_size 8 --fp16

# BioBERT
python src/train.py --model dmis-lab/biobert-base-cased-v1.2 --data_dir data/tokenized_biobert --output_dir models/biobert --epochs 6 --batch_size 4 --fp16
```

### View Experiments

Launch MLflow UI to explore training metrics:
```bash
mlflow ui
```

Open browser at `http://localhost:5000`

---

## Results

### Model Comparison (6 epochs each)

| Model | Val F1 Macro | Val Accuracy | Test F1 Macro | Test Accuracy | Train Loss | Val Loss |
|-------|--------------|--------------|---------------|---------------|------------|----------|
| **DistilBERT** | 0.147 | 0.336 | **0.163** | **0.377** | 0.596 | 1.844 |
| **BioBERT** | 0.140 | 0.333 | 0.160 | 0.371 | 0.575 | 1.861 |

### Training Configuration

**DistilBERT:**
- Batch size: 8
- Gradient accumulation: 4 steps
- Learning rate: 2e-5
- FP16: Enabled
- Max sequence length: 128 tokens

**BioBERT:**
- Batch size: 4
- Gradient accumulation: 8 steps
- Learning rate: 2e-5
- FP16: Enabled
- Max sequence length: 128 tokens

### Key Observations

✅ **DistilBERT slightly outperforms BioBERT** despite being a general-purpose model
- Likely due to small dataset size (~10K samples)
- Domain-specific models benefit more from larger datasets

✅ **Both models show overfitting** (low val F1, high train-val loss gap)
- Suggests need for regularization or data augmentation

✅ **GPU optimization successful** (fp16 + gradient accumulation)
- Models trained on 4GB GPU without OOM errors

---

## MLOps Stack

### Pipeline Orchestration - DVC

**3-stage reproducible pipeline:**
1. `data_preparation` - Chunking, splitting, label encoding
2. `tokenization` - Model-specific tokenization
3. `training` - Model training with checkpointing

**Configuration:** `dvc.yaml` + `params.yaml` for hyperparameter versioning

### Experiment Tracking - MLflow

- Training loss logged every 50 steps
- Validation metrics at epoch end
- Automatic run organization by model type
- Resume training from checkpoints supported

### Version Control

- **Git:** Source code versioning
- **DVC:** Data & model versioning (`.dvc` files in Git)
- **MLflow:** Experiment metadata & metrics

---

## Key Insights

### 1. Domain-Specific Models Don't Always Win
**Finding:** DistilBERT (general) matched/exceeded BioBERT (medical-specific) performance.

**Hypothesis:** 
- Small dataset (~10K samples) insufficient to leverage BioBERT's domain knowledge
- Medical vocabulary overlap between models reduces BioBERT's advantage
- DistilBERT's efficiency allows better optimization with limited data

### 2. Data Quality > Model Complexity
**Observation:** Low F1 scores (0.14-0.16) suggest:
- Class imbalance issues (Surgery: 1,103 vs minority classes: <100)
- Chunking strategy may split coherent medical context
- 20-class problem is inherently challenging

**Potential improvements:**
- Better chunking (semantic vs character-based)
- Class rebalancing techniques
- Reduce to top-10 specialties

### 3. MLOps Best Practices Applied
✅ Reproducible pipeline (DVC)  
✅ Experiment tracking (MLflow)  
✅ Modular, documented code  
✅ Version-controlled hyperparameters  
✅ GPU resource optimization (fp16, gradient accumulation)

---

## Future Work

### Model Improvements
- [ ] Increase training epochs (10-15) with early stopping
- [ ] Experiment with learning rate scheduling

### Data Engineering
- [ ] Semantic chunking (sentence-aware splitting)
- [ ] Data augmentation (back-translation, synonym replacement)
- [ ] Address class imbalance (oversampling, focal loss)
- [ ] Feature engineering with medical NER

### MLOps Enhancements
- [ ] Add model registry (MLflow Model Registry)

### Analysis
- [ ] Per-class performance analysis (confusion matrix)
- [ ] Error analysis on misclassifications

---
