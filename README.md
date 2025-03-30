# credit-card-fraud-detection-machine-learning-project

A complete machine learning project for detecting fraudulent transactions in credit card data using XGBoost, Random Forest, and Logistic Regression, with a strong focus on imbalanced classification, evaluation metrics, and model interpretability.

---

## Problem Statement

Credit card fraud is a major concern for financial institutions and consumers. The goal of this project is to build and compare machine learning models to accurately detect fraudulent transactions, which represent only a very small fraction of the total.

---

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Total transactions: **284,807**
- Fraudulent transactions: **492** (≈ 0.17%)
- Features: 
  - `V1`–`V28`: Principal Components (PCA)
  - `Time`: Time in seconds since the first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = legitimate, 1 = fraud)

---

## Technologies

- Python (pandas, numpy, seaborn, matplotlib)
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- SHAP (model explainability)

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis
- Outlier detection
- Class imbalance visualization

### 2. Data Preparation
- SMOTE for oversampling minority class
- Train/test split with stratification

### 3. Modeling & Evaluation
Models trained and evaluated:
- **XGBoost** (with hyperparameter tuning via GridSearchCV)
- **Random Forest**
- **Logistic Regression**

Metrics computed:
- Precision, Recall, F1-score (especially on fraud class)
- ROC AUC
- Precision-Recall AUC
- Confusion Matrix
- ROC and PR curves

### 4. Interpretability
- SHAP values computed for XGBoost
- Global and local feature impact visualizations

---

## Results Summary (Fraud Class - Class 1)

| Model               | Precision | Recall | F1-score | ROC AUC | PR AUC |
|--------------------|-----------|--------|----------|---------|--------|
| **XGBoost**         | 0.8723    | 0.8311 | 0.8512   | 0.9817  | 0.8476 |
| Random Forest       | 0.3272    | 0.8378 | 0.4706   | 0.9779  | 0.6935 |
| Logistic Regression | 0.0816    | 0.8581 | 0.1491   | 0.9523  | 0.6829 |

 **XGBoost** achieved the best overall balance between precision and recall, especially considering the high class imbalance.

---

## Key Takeaways

- Handling class imbalance is crucial: SMOTE helped all models improve recall significantly.
- XGBoost was the most effective model, both in raw performance and interpretability.
- Evaluation based on **PR AUC** is more appropriate than accuracy in this context.

---

## Future Work

- Try LightGBM and CatBoost for comparison
- Deploy a real-time fraud scoring API with Streamlit or FastAPI
- Add threshold tuning for precision/recall trade-off
- Use time-aware validation (e.g., TimeSeriesSplit)

---

## References

- [Imbalanced-learn documentation](https://imbalanced-learn.org/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [Original Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## Author

Francesco Astarita 
*Data Scientist | Machine Learning Engineer*
