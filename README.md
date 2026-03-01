# AI for Health – SRIP 2026

This repository contains a **task completed under the AI for Health project of SRIP 2026**.  
The objective of this task is to analyze overnight physiological signals and build a machine learning pipeline to detect abnormal breathing patterns during sleep.

---

## 📋 Project Overview

This project analyzes overnight sleep data to identify breathing irregularities such as apnea and hypopnea using physiological signals.  
A complete pipeline was implemented to classify **30-second windows** of sleep data as **normal** or **abnormal** breathing patterns.

### 🎯 Objectives
1. Visualize physiological signals with annotated breathing events  
2. Create a labeled dataset from raw time-series data  
3. Train a machine learning model for abnormal breathing detection  
4. Evaluate performance using participant-wise validation  

---

## 📊 Dataset

### Participants
- **5 subjects**: AP01, AP02, AP03, AP04, AP05  
- **~8 hours** of sleep data per participant  
- **Total windows**: ~8,800  
- **Window size**: 30 seconds  
- **Overlap**: 50%  

### Signals Collected

| Signal | Sampling Rate | Description |
|------|---------------|-------------|
| SpO₂ | 4 Hz | Oxygen saturation |
| Sleep Profile | 30-second epochs | Sleep stages |
| Flow Events | Event-based | Hypopnea, apnea, body events |

### Class Distribution

| Label | Count | Percentage |
|------|-------|------------|
| Normal | 8,631 | 98.1% |
| Hypopnea | 166 | 1.9% |
| Obstructive Apnea | 1 | <0.1% |
| Body Event | 2 | <0.1% |

⚠️ **Key Challenge:** Extreme class imbalance (≈98% Normal vs 2% Abnormal)

---

## 🔧 Methodology

### 1️⃣ Data Preprocessing (`create_dataset.py`)
- Parsed timestamp-based physiological signals  
- Aligned multi-rate signals using timestamps  
- Generated 30-second sliding windows with 50% overlap  
- Labeled windows based on event overlap  
  - **>50% overlap → Abnormal**
- Extracted features:
  - `avg_spo2`, `min_spo2`, `max_spo2`, `std_spo2`

---

### 2️⃣ Visualization (`vis.py`)
- Plotted overnight SpO₂ signals (~8 hours)
- Overlaid annotated breathing events
- Saved participant-wise PDF visualizations

---

### 3️⃣ Model Training (`train_model.py`)
- **Model**: Random Forest Classifier  
- **Class Weighting**: Balanced  
- **Threshold Tuning**: Optimized for F1-score  
  - Final threshold = **0.35**
- **Evaluation Focus**:
  - AP04 (only participant with breathing events)

---

## 📈 Results

### Performance Metrics (AP04 Test Set)

| Metric | Value |
|------|-------|
| Accuracy | 77.00% |
| Precision | 18.39% |
| Recall | 47.06% |
| F1-Score | 26.45% |
| ROC-AUC | 70.37% |

---

### Confusion Matrix

```
                    Predicted Normal   Predicted Abnormal
True Normal               282                  71
True Abnormal              18                  16
```

---

### Interpretation
- Model shows meaningful discriminative ability  
- Nearly half of abnormal breathing events detected  
- Low precision is expected due to class imbalance  
- Limited annotations restrict generalization  

---

## 📁 Project Structure

```
SRIP-2026-AI-Health/
│
├── Data/
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
│
├── Dataset/
│   └── breathing_dataset.csv
│
├── Visualizations/
│   └── AP04_visualization.pdf
│
├── models/
│   ├── results.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── scripts/
│   ├── vis.py
│   ├── create_dataset.py
│   └── train_model.py
│
├── README.md
└── requirements.txt
```

---

## 🚀 Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Visualizations
```bash
python scripts/vis.py -name "Data/AP04" -out_dir "Visualizations"
```

### Create Dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

### Train Model
```bash
python scripts/train_model.py -dataset "Dataset/breathing_dataset.csv" -output "models"
```

---

## 🔍 Key Challenges & Solutions

| Challenge | Solution |
|--------|----------|
| Extreme class imbalance | Class weighting |
| Different sampling rates | Timestamp alignment |
| Limited abnormal events | Participant-focused evaluation |
| Strict default threshold | Tuned decision threshold |

---

## 💡 Future Improvements
- Include more participants with breathing events  
- Add temporal and desaturation-based features  
- Apply deep learning (LSTM / GRU)  
- Explore multi-signal fusion  

---

<div align="center">

**AI for Health – SRIP 2026**  
**Shivam Naik**

</div>
