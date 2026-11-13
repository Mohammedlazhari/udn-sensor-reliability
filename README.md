# UNDN Sensor Reliability — Master's Thesis

**Deep Learning Dual-Model for Predicting and Validating Sensor Data in Underwater Named Data Networks (UNDN)**

**Author:** Mohammed Lazhari Krobba  
**Institution:** Amar Telidji University, Laghouat, Algeria  
**Degree:** M.Sc. Data Science & Artificial Intelligence (13.82/20)  
**Defended:** June 22, 2025  
**Publication:** AAIET 2025 Poster

---

## 🎯 Highlights

- **5 architectures benchmarked:** LSTM, GRU, CNN-1D, Transformer, TCN
- **Key finding:** TCN best generalization (MSE 1.13°C) despite weaker test MSE
- **Dual-model architecture:** Time-series forecaster + MLP validator (93% accuracy, 0.9291 F1)
- **Dataset:** Underwater Surface Temperature (Santa Catarina coast, 7 sites, 2012-2014)
- **Implementation:** TensorFlow/Keras, Python, Jupyter (no external DL frameworks beyond TF)
- **MLOps practices:** Model versioning, experiment tracking, statistical validation, Docker-ready

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/udn-sensor-reliability.git
cd udn-sensor-reliability

# Option 1: Docker (recommended)
docker compose up --build

# Option 2: Local environment
pip install -r requirements-lock.txt
python scripts/train_all.py --config experiments/configs/tcn.yaml
```

---

## 🏗 Architecture

### Dual-Model Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    UNDERWATER SENSOR DATA                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                      │
│  • 64-timestep sliding windows                              │
│  • 85/15 chronological split (no leakage)                   │
│  • Min-Max normalization (temperature 0-1)                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   TIME-SERIES FORECASTER │     │      VALIDATOR (MLP)     │
│   (5 Architectures)      │     │   (Binary Classification)│
│                          │     │                          │
│  • LSTM                  │     │  • 7 features            │
│  • GRU                   │     │  • Z-score normalization │
│  • CNN-1D                │     │  • 93% accuracy          │
│  • Transformer           │     │  • 0.9291 F1             │
│  • TCN ✓ BEST            │     │                          │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECISION SYSTEM                           │
│  • Forecast next sensor reading                             │
│  • Validate current reading (normal/anomaly)                │
│  • Trigger alert if anomaly detected                        │
└─────────────────────────────────────────────────────────────┘
```

### Model Architectures

| Model | Key Components | Parameters |
|-------|---------------|------------|
| **LSTM** | 2×LSTM(16) → Dense(8) → Dense(1) | ~1.2K |
| **GRU** | 2×GRU(16) → Dense(8) → Dense(1) | ~1.1K |
| **CNN-1D** | Conv1D(128,3) → MaxPool → Conv1D(64,3) → Conv1D(32,3) → MaxPool → Flatten → Dense(32) → Dropout(0.4) → Dense(1) | ~15K |
| **Transformer** | Embed(64) → PosEnc → MultiHeadAtt(6 heads) → FFN(64) → LN → GlobalAvgPool → Dense(1) | ~25K |
| **TCN** | 4×DilatedConv1D(dilations=[1,2,4,8]) → Dense(1) | ~8K |
| **MLP Classifier** | Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(1, sigmoid) | ~5K |

---

## 📊 Results

### Time-Series Forecasting (MSE on Future Unseen Data)

| Model | Test MSE | Forecast MSE | Generalization Gap |
|-------|----------|--------------|-------------------|
| **TCN** | 2.87 | **1.13** | **-1.74** ✓ |
| Transformer | 1.95 | 12.47 | +10.52 |
| CNN-1D | 1.82 | 15.93 | +14.11 |
| LSTM | 1.78 | 18.34 | +16.56 |
| GRU | 1.81 | 17.92 | +16.11 |

> **Key Insight:** TCN is the **only model with negative generalization gap** — it learns temporal dynamics, not memorization.

### MLP Classifier (Fault Detection)

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% |
| Precision | 91.89% |
| Recall | 93.95% |
| Specificity | 91.67% |
| F1-Score | **92.91%** |

**Confusion Matrix:** TP=9993, TN=9709, FP=882, FN=643

---

## 🔬 Reproducibility

- **Environment:** `requirements-lock.txt` / `Dockerfile` / `docker-compose.yml`
- **Seeds:** All seeds fixed in `experiments/configs/*.yaml`
- **Data:** `data/processed/` (train/test splits) or download script in `scripts/download_data.py`
- **Compute:** Runs on CPU (8GB RAM sufficient) / GPU optional
- **CI/CD:** GitHub Actions run tests on every push (`.github/workflows/`)

```bash
# Full reproducibility check
docker compose run --rm reproducibility-check
# Verifies: seed consistency, data integrity, metric reproducibility
```

---

## 📖 Citation

```bibtex
@article{krobba2025undn,
  title={Improving Data Reliability in Underwater Named Data Networks via Deep Learning-Based Prediction and Validation},
  author={Krobba, Mohammed Lazhari},
  journal={Proceedings of the International Conference on Applied Artificial Intelligence and Emerging Technologies (AAIET)},
  year={2025},
  note={Poster Presentation}
}
```

---

## 📁 Project Structure

```
udn-sensor-reliability/
├── data/
│   ├── raw/                    # Original CSV files (git-lfs or Zenodo)
│   ├── processed/              # Preprocessed train/test splits
│   └── README.md               # Data sources, preprocessing steps
├── models/
│   ├── lstm_gru/
│   ├── cnn1d/
│   ├── transformer/
│   ├── tcn/
│   ├── seq2seq_tcn/
│   ├── autoformer/
│   └── mlp_classifier/
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── saved_model/        # TensorFlow SavedModel format
│       ├── metrics.json        # MSE, MAE, accuracy, F1, etc.
│       ├── training_history.csv
│       └── config.yaml         # Hyperparameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training_lstm_gru.ipynb
│   ├── 04_training_cnn_transformer.ipynb
│   ├── 05_training_tcn.ipynb
│   ├── 06_training_mlp_classifier.ipynb
│   ├── 07_results_analysis.ipynb
│   └── 08_statistical_validation.ipynb
├── experiments/
│   ├── configs/
│   ├── logs/                   # TensorBoard/MLflow logs (or link)
│   └── results/                # CSV predictions, confusion matrices
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── scripts/
│   ├── preprocess.py
│   ├── train_all.py
│   ├── evaluate_all.py
│   └── statistical_tests.py
├── .github/workflows/
│   ├── test.yml
│   └── evaluate.yml
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

---

## 📄 License

MIT License — see LICENSE file.