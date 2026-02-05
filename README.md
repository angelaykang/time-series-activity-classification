# Time Series Classification for Activity Recognition

A machine learning pipeline for classifying human activities from sensor time series data using the AReM (Activity Recognition with Ambient Sensing) dataset. This project covers the full workflow from feature extraction to model training and evaluation.

## Overview

The AReM dataset contains sensor readings from 7 activity classes: bending1, bending2, cycling, lying, sitting, standing, and walking. Each activity is recorded as multivariate time series from 6 sensors. This project:

1. **Extracts statistical features** from raw time series (mean, median, std, quartiles, min, max)
2. **Trains binary classifiers** to distinguish bending vs. other activities
3. **Trains multi-class classifiers** to predict all 7 activity classes
4. **Compares methods**: RFE + Logistic Regression, L1-penalized Logistic Regression, Gaussian NB, Multinomial NB

## Key Features

- **Feature Engineering**: Statistical aggregation (central tendency, dispersion, position) per time series segment
- **Segmented Feature Extraction**: Splits each time series into \(l\) segments for richer representations
- **Feature Selection**: RFE (Recursive Feature Elimination) and RFECV for optimal feature count
- **Bootstrap Confidence Intervals**: 90% CI for standard deviation of extracted features
- **Binary Classification**: Bending vs. other activities with ROC curves and case-control adjustment for class imbalance
- **Multi-class Classification**: All 7 activity classes with micro/macro ROC analysis
- **Model Comparison**: Logistic Regression (RFE, L1), Gaussian NB, Multinomial NB

## Dataset

The AReM dataset is organized by activity:
- `bending1/`, `bending2/` — bending movements
- `cycling/`, `lying/`, `sitting/`, `standing/`, `walking/`

Each folder contains `dataset1.csv`, `dataset2.csv`, etc. Train/test split: bending uses datasets 1–2 for test; others use 1–3 for test.

**Dataset Source**: [Activity Recognition system based on Multisensor data fusion (AReM)](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem) — UCI Machine Learning Repository

## Project Structure

```
time-series-activity-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── AReM/
│       ├── bending1/
│       ├── bending2/
│       ├── cycling/
│       ├── lying/
│       ├── sitting/
│       ├── standing/
│       └── walking/
└── notebooks/
    └── activity_classification.ipynb
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/angelaykang/time-series-activity-classification.git
cd time-series-activity-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Ensure the AReM data is in `data/AReM/` (included in repo)
2. **Run Jupyter from the project root** so paths resolve correctly:
```bash
cd time-series-activity-classification
jupyter notebook notebooks/activity_classification.ipynb
```

## Methodology

### 1. Feature Extraction
- Load CSV time series, parse 6 sensor columns
- Per time series (or per segment): min, max, mean, median, std, Q1, Q3
- Bootstrap confidence intervals for feature variability

### 2. Binary Classification (Bending vs. Other)
- Logistic Regression with RFE for feature selection
- Optimal segment count \(l\) and feature count \(p\) via cross-validation
- Case-control adjustment for class imbalance
- L1-penalized Logistic Regression for comparison

### 3. Multi-class Classification (All 7 Activity Classes)
- L1-penalized multinomial Logistic Regression
- Gaussian Naive Bayes
- Multinomial Naive Bayes (with MinMax scaling)
- ROC curves: per-class, micro-average, macro-average

## Results

- **Binary**: High accuracy with strong class separation (AUC ~1.0 on train)
- **Multi-class**: L1 multinomial and Gaussian NB achieve ~10% test error on 7 classes
- **Best features**: Mean, max, and standard deviation show strong discriminative power across activities

## Technologies Used

- **Python 3.8+**
- **pandas** — Data manipulation
- **numpy** — Numerical computing
- **scikit-learn** — Logistic Regression, RFE, Naive Bayes, metrics
- **scipy** — Bootstrap confidence intervals
- **statsmodels** — Logit p-values
- **matplotlib & seaborn** — Visualization
