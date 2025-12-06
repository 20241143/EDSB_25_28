---
title: Bank Telemarketing Predictor
emoji: 📞
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.0.1"
app_file: app.py
pinned: false
---

# 📞 Bank Telemarketing Campaign Success Predictor

This interactive **Gradio application** predicts whether a customer will subscribe to a **term deposit** when contacted during a bank telemarketing campaign.

It uses a fully engineered **XGBoost classification model**, trained as part of the **Enterprise Data Science Bootcamp Final Project**, using the the Bank Marketing Dataset.

---

# 🎯 What This App Does

Given customer features (age, job, contact history) and economic indicators (Euribor, CPI, employment variation), the model predicts:

- **Probability of subscription**  
- **Binary classification (0 = no, 1 = yes)** using a tuned and optimized decision threshold  

This allows campaign teams to prioritize high-potential customers and simulate different outreach scenarios.

---

# 🧠 Behind the Model

## 🌟 Final Algorithm: **XGBoost Classifier**
Selected after comparing:
- Baseline Logistic Regression  
- Tuned Logistic Regression (feature selection, WOE encoding, PCA)  
- Tuned XGBoost (final winner)

XGBoost was chosen because it captured **non-linear patterns**, **interactions**, and **imbalanced data behavior** significantly better than linear models.

---

# 🏗️ How the Model Was Built

## 🔧 1. Data Preparation
- Loaded and cleaned the Bank Marketing Dataset  
- Standardized column names  
- Handled missing & “unknown” data  
- Removed duplicates  
- Encoded cyclical time signals:  
  - `month_sin`, `month_cos`  
  - `day_of_week_sin`, `day_of_week_cos`  

## 🎨 2. Feature Engineering
Engineered transformations included:

### Customer-level features:
- Job grouping → `job` normalized to broader job families  
- Education simplification → reduced cardinality  
- Age binning and interactions  
- `job × age_bin` interaction  

### Numeric interactions:
- `age × emp_var_rate`  
- `cons_price_idx × cons_conf_idx`  
- Normalized Euribor: `euribor_nrm`

### Campaign intensity:
- `campaign_log`  
- `contacts_ratio`  

### Macroeconomic volatility:
- Row-wise volatility index from three indicators  

### PCA:
- Applied to macroeconomic block  
- Extracted 2 principal components explaining >85% variance  

---

# ⚖️ 3. Handling Class Imbalance

The dataset’s positive class (“subscription = yes”) is only ~11%.

To fix this:

- Used **SMOTE** in training to oversample minority class  
- Optimized decision **threshold** for **best F1-score**

This avoids models biased toward predicting “no”.

---

# 🔍 4. Hyperparameter Optimization

Used **GridSearchCV** with cross-validation to tune:

- Learning rate  
- Max depth  
- Number of estimators  
- Column and row sampling  
- Min child weight  
- SMOTE parameters  
- WOE encoder regularization  
- Feature selection cutoffs  

---

# 🧪 5. Final Model Performance

_On test set (after threshold tuning):_

| Metric | Score |
|-------|-------|
| **AUC** | ~0.79 |
| **Accuracy** | ~0.87 |
| **Precision** | ~0.44 |
| **Recall** | ~0.54 |
| **F1-score** | ~0.48 |

### Interpretation:
- Excellent performance for class 0 (non-subscribers)  
- Balanced precision/recall for class 1 (subscribers)  
- Optimal trade-off for maximizing campaign conversion targeting  

---