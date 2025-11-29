# 📊 Baseline Logistic Regression — Analysis

## 1. Overview

The baseline model is a **scikit-learn Logistic Regression** trained on the cleaned dataset using:

- no feature engineering  
- no scaling  
- no PCA  
- raw numeric features  
- one-hot encoded categorical variables  
- default L2 regularization  
- SHAP for interpretability  

This baseline serves as the **reference point** to measure improvements from feature engineering and advanced models such as Random Forest, XGBoost, and an engineered Logistic Regression.

---

## 2. Key Metrics

All metrics were saved in:

📄 `../97-model_metrics/baseline/logistic_regression_baseline_metrics.csv`

**Included metrics:**

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  
- Log-loss  
- Brier score  
- Majority-class baseline accuracy  

### Interpretation

- The baseline model provides **reasonable classification performance**, but far from optimal.  
- This is expected since the model uses raw features and no transformations.  
- These metrics serve as the baseline for evaluating improvements from feature-engineered and non-linear models.

---

## 3. Important Observations

### 3.1 Multicollinearity Is Still High

The macroeconomic variables:

- `emp_var_rate`  
- `euribor3m`  
- `nr_employed`  
- `cons_price_idx`  

are **highly correlated**. SHAP confirms they dominate the model’s importance.

This causes:

- unstable coefficients  
- overlapping contributions  
- reduced interpretability  

### ✔ What this means

- Logistic Regression remains usable, but its interpretability is weakened.  
- **PCA on the macroeconomic features is recommended** for the feature-engineered models.

---

### 3.2 SHAP Highlights Key Predictors

SHAP Global Importance reveals strong predictors:

#### 🔹 Macroeconomic indicators  
- `emp_var_rate`  
- `euribor3m`  
- `nr_employed`  
- `cons_price_idx`  

These explain most predictive power.

#### 🔹 Contact method  
- `contact_telephone` → strong **negative** effect.

#### 🔹 Previous campaign outcome  
- `poutcome_success` → strong positive effect  
- `poutcome_nonexistent` → moderate positive effect  

These patterns align with prior analyses of this dataset.

---

### 3.3 Low-Importance Features

Many one-hot encoded features contribute very little:

- education levels  
- marital statuses  
- housing and loan flags  
- rare job categories  
- `default_yes` (rare → unstable estimate)

These can be removed or grouped in feature engineering.

---

## 4. SHAP Visualizations

### 📌 4.1 SHAP Summary Plot  
Global importance and directionality.

![SHAP Summary Plot](../97-model_metrics/baseline/summary_plot.png)

---

### 📌 4.2 SHAP Bar Plot  
Mean absolute impact.

![SHAP Bar Plot](../97-model_metrics/baseline/bar_plot.png)

---

### 📌 4.3 SHAP Heatmap (Sampled 1000 rows)  
Instance-level behavior and feature clustering.

![SHAP Heatmap](../97-model_metrics/baseline/heatmap_plot.png)

---

## 5. Interpretation of the Baseline

The scikit-learn Logistic Regression baseline shows:

- **Macroeconomic variables dominate predictions**  
- **contact_telephone** strongly reduces probability of subscription  
- **previous successful outcomes** boost the probability  
- **many categorical variables provide little predictive signal**  
- **non-linear and interaction patterns** cannot be captured by the baseline  

This indicates:

- Strong room for improvement  
- Expected performance gains from PCA + feature engineering + tree-based models  
- SHAP confirms consistent and explainable patterns

The baseline is therefore **valid, interpretable, and useful**, while clearly leaving space for improvement.

---

## 6. Why PCA and Feature Engineering Are Needed Next

### Required Improvements:
- PCA on macroeconomic variables  
- Scaling for engineered logistic regression  
- Bucketing (age groups, campaign bins)  
- Interaction terms (e.g., contact × month)  
- Simplification of categorical variables  

### Additional Modeling Improvements:
- Feature-engineered Logistic Regression  
- Random Forest  
- XGBoost  
- Better calibration  
- SHAP comparison across all models  

These steps will significantly outperform the baseline.

---

## 7. Next Steps

The next notebook (**Feature Engineering**) will:

1. Apply PCA to macroeconomic features  
2. Collapse or remove low-frequency categorical levels  
3. Add engineered variables  
4. Prepare datasets for:  
   - Feature-Engineered Logistic Regression  
   - Random Forest  
   - XGBoost  
5. Compare models using:  
   - ROC–AUC  
   - Precision and Recall  
   - F1-score  
   - Calibration  
   - SHAP interpretability  

This baseline establishes a clear and measurable starting point for assessing improvements.

