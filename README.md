# Fake Job Posting Detection — End-to-End Machine Learning & Deep Learning Project

This project builds a complete fraud-detection pipeline to classify job postings as *real* or *fake* using a combination of exploratory data analysis, advanced feature engineering, machine learning models, and deep learning architectures.  
The objective is to understand the behavioural patterns of fraudulent job ads and create a reliable detection system suitable for real-world deployment.

---

##  Project Pipeline

### 1. Exploratory Data Analysis (EDA)
A comprehensive EDA was conducted to understand dataset structure, missing values, correlations, and major categorical distributions.  
Key insights:
- Dataset is **highly imbalanced** — fake job posts represent a small fraction.
- Columns such as **employment_type**, **required_experience**, **industry**, and **function** hold a major share of the dataset’s information.
- Text fields like **description**, **requirements**, and **benefits** carry strong fraud-detection signals.

Multiple visualisations (distribution plots, wordclouds, imbalance plots, correlation heatmaps) were used to uncover actionable patterns.

---

##  Feature Engineering

A structured and multi-stage feature engineering pipeline was developed:

- **Text Preprocessing:**  
  Lowercasing, tokenisation, stopword removal, lemmatisation.
  
- **Vectorisation:**  
  TF-IDF, word/character count features, and n-gram based signals.

- **Categorical Encodings:**  
  Label Encoding + One-Hot Encoding for structured data.

- **Balancing Technique:**  
  SMOTE was used to counter class imbalance and improve recall on the fake class.

- **Final Feature Matrix:**  
  Combined structured variables + engineered text features for ML models,  
  and separate clean textual sequences for LSTM/BiLSTM models.

---

##  Machine Learning & Deep Learning Models

Multiple models were trained and compared to understand which algorithms best capture fraud-related patterns.

### Machine Learning Models
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  
- SVM (TF-IDF Based)

### Deep Learning Models
- Artificial Neural Network (ANN)  
- **BiLSTM (Description-Only)** — strongest text-based model

### Hyperparameter Tuning
- GridSearchCV and RandomizedSearchCV for ML models  
- Learning rate, dropout, batch size, and epoch tuning for neural networks  
- Model selection based on F1-Score (Fake class)

---

## Results & Evaluation

Below is the performance comparison of all models (values taken from your project):

### **Model Performance Heatmap**

> *(This table corresponds to the heatmap image in your project)*

| Model                       | Accuracy   | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
|----------------------------|------------|-------------------|----------------|------------------|
| **BiLSTM (Description Only)** | **0.974832** | 0.798561          | 0.641618       | 0.711538         |
| **XGBoost**                 | 0.972875   | 0.756757          | 0.647399       | 0.697819         |
| **LightGBM**                | 0.972595   | 0.721893          | 0.705202       | 0.713450         |
| SVM + TF-IDF                | 0.969239   | 0.652174          | **0.780347**   | 0.710526         |
| Random Forest               | 0.966723   | **0.982143**       | 0.317919       | 0.480349         |
| Gradient Boosting           | 0.965045   | 0.875000          | 0.323699       | 0.472574         |
| CatBoost                    | 0.958613   | 0.557604          | 0.699422       | 0.620513         |

---

### Interpretation

- **BiLSTM (Description-Only)** achieved the **highest overall accuracy** and one of the strongest F1-scores, showing its ability to extract rich semantic information from job descriptions.
  
- **XGBoost** showed the **best balance of precision + recall** among machine learning models, making it highly reliable for fraud detection in structured + text-engineered features.

- **LightGBM** also performed exceptionally well with stable recall and F1.

- **SVM + TF-IDF** achieved the **highest recall**, meaning it caught the most fake job posts — important in fraud detection scenarios.

- **Random Forest and Gradient Boosting** showed inflated precision but extremely low recall (i.e., they missed many fraudulent cases), making them unsuitable for this task.

**Final Recommended Models:**  
✔ **BiLSTM (text-based deep learning)**  
✔ **XGBoost (structured + text features)**

---

## Technology Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM, CatBoost  
- TensorFlow / Keras  
- NLTK / spaCy  

---

## How to Run the Project

```bash
pip install -r requirements.txt
jupyter notebook
# Open: '1. EDA, Feature Engineering And Model Building.ipynb'
