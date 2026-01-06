# Fake Job Posting Detection — End-to-End Machine Learning & NLP Project

This project builds a complete fraud-detection pipeline to classify job postings as *legitimate* or *fraudulent* using exploratory data analysis, interpretable feature engineering, and multiple machine learning models.  
The objective is to understand behavioural patterns of fraudulent job ads and develop an imbalance-aware detection system suitable for real-world screening and deployment.

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
A comprehensive EDA was conducted to understand dataset structure, class imbalance, missingness patterns, and key text differences between legitimate and fraudulent postings.  
Key insights:
- The dataset is **highly imbalanced** — fraudulent postings represent a small minority (~4.8%).
- Missingness is **class-informative** (fraudulent postings show systematically higher missing fields).
- Text fields such as **description**, **company_profile**, **requirements**, and **benefits** contain strong discrimination signals through length, richness, and completeness patterns.

Visualisations used include class distribution plots, missingness by class heatmaps, text length/richness plots, feature correlation heatmaps, and precision–recall curves.

---

## Feature Engineering

A structured and multi-stage feature engineering pipeline was developed to capture both structural and linguistic fraud signals:

- **Text Cleaning:**  
  Lowercasing, punctuation/whitespace normalisation, and safe token cleaning (without introducing leakage).

- **Vectorisation:**  
  TF-IDF representation for unstructured text (sparse high-dimensional features).

- **Interpretable Engineered Features:**  
  - Text length and richness metrics (word counts, richness score)
  - Structural missingness counts (missing key fields as signals)
  - Scam keyword indicators (lexicon-based)
  - Skill emphasis features (soft vs technical signal ratios)
  - Writing-style indicators (e.g., punctuation patterns where applicable)

- **Final Feature Space:**  
  Combined TF-IDF features + engineered numerical features for hybrid models, enabling both lexical modelling and interpretable behavioural pattern capture.

---

## Machine Learning Models

Multiple models were trained and compared under an imbalanced classification setting.

### Final Thesis Models (Locked)
- Gradient Boosting (GBM) — weak baseline
- SVM (TF-IDF + engineered features) — recall-oriented behaviour
- LightGBM — balanced performance model
- XGBoost — high-precision benchmark

### Evaluation Design (Imbalance-Aware)
- Stratified train/test split
- Fraud-focused metrics:
  - Precision (fraud class)
  - Recall (fraud class)
  - F1-score (fraud class)
  - PR-AUC (Precision–Recall AUC)
- Precision–Recall curves for threshold-independent comparison
- Decision-threshold tuning to support deployment trade-offs

---

## Results & Evaluation

Below is the comparative performance of the final models (fraud-class metrics reported):

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | PR-AUC |
|------|----------:|------------------:|---------------:|-----------------:|------:|
| LightGBM — Balanced Performance | 0.973154 | 0.751634 | 0.664740 | 0.705521 | 0.793270 |
| XGBoost — High Precision Benchmark | 0.965884 | 0.629442 | 0.716763 | 0.670270 | 0.770293 |
| SVM (TF-IDF + Engineered) — Recall-Oriented | 0.974553 | 0.988095 | 0.479769 | 0.645914 | 0.779714 |
| Gradient Boosting (GBM) — Weak Baseline | 0.963087 | 0.872727 | 0.277457 | 0.421053 | 0.616447 |

---

### Interpretation

- LightGBM achieved the strongest overall balance between precision and recall, yielding the best fraud F1-score and PR-AUC.
- XGBoost produced stronger precision-oriented behaviour (useful when false positives are costly), while maintaining competitive recall.
- SVM showed recall-oriented behaviour depending on threshold choice and was useful for high-risk screening scenarios where missing fraud is costly.
- GBM served as a weak baseline, illustrating how accuracy can be misleading under severe imbalance.

---

## Technology Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM  

---

## Repository Files

- `fake job postings project.ipynb` — main end-to-end notebook (EDA → features → modelling → evaluation)
- `Data Science Project Report.pdf` — final MSc project report

---

## How to Run the Project

```bash
pip install -r requirements.txt
jupyter notebook
# Open: 'fake job postings project.ipynb'