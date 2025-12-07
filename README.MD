# 📞 Predicting the Success of Bank Telemarketing Campaigns

This repository contains the complete workflow for building, evaluating, and deploying a predictive model to estimate whether a customer will **subscribe to a term deposit** following a bank’s telemarketing campaign.  
The project was developed as the **Final Project for the Enterprise Data Science Bootcamp 2024/25**.

It includes:

- Full **EDA**, preprocessing and feature engineering  
- Multiple **modeling approaches** and comparison  
- **SHAP interpretability**  
- A fully interactive **Hugging Face Space (Gradio app)**  
- CI/CD workflow for automatic deployment  
- Detailed analysis documentation and metrics

# 👥 Team Members
| Name | Student ID |
|------|------------|
| Diogo Ferreira | 20241143 |
| Francisco Moutinho | 20240818 |
| Pedro Boto | 20241144 |
| João Carvalho | 20241340 |

**Navigation**
- [Home](README.MD)
- [How to Setup GIT and Hugging Face Spaces](How_to_setup_git_hugging.MD)
- [Baseline Analysis](Analysis_baseline.MD)

### Requirements
- See [requirements.txt](requirements.txt) for dependencies.

# 🗂 Project Structure
```
📦 EDSB_25_28
│
├── .github/
│   └── workflows/
│       └── deploy_to_hf.yml
│
├── 01-data/
│   ├── 01-bronze/
│   ├── 02-silver/
│   └── 03-gold/
│
├── 02-notebooks/
│   ├── 01-EDA.ipynb
│   ├── 02-Data Cleaning and Preprocessing.ipynb
│   ├── 03-Baseline.ipynb
│   ├── 04-Logistic Regression.ipynb
│   ├── 05-XGBoost.ipynb
│   ├── 06-best_model.ipynb
│   └── utils.py
│
├── 96-huggingface_space/
│   ├── app.py
│   ├── model.pkl
│   ├── README.md
│   └── requirements.txt
│
├── 97-model_metrics/
│   ├── Baseline/
│   ├── Logistic_Regression_Model/
│   ├── Logistic_Regression_Model_yyyyMMdd_hhmmss
│   ├── XGBoost_Model/
│   └── XGBoost_Model_yyyyMMdd_hhmmss
│
├── 98-reports/
│
├── 99-old/
│
├── .env
├── .gitignore
├── Analysis_baseline.md
├── How_to_setup_git_hugging.md
├── README.md
└── requirements.txt
```

# 📘 Project Overview

The **Bank Marketing Dataset** contains customer demographics, contact history, and macroeconomic indicators.  
The target variable is whether the customer subscribed to a term deposit (`y = yes/no`).

**Goal:** Build and deploy a model that predicts subscription likelihood during a telemarketing call.


# 🔍 1. Exploratory Data Analysis (EDA)

The EDA notebook covers:

### **Univariate Analysis**
- Age distributions  
- Economic indicator trends  
- Campaign activity patterns (`campaign`, `pdays`, `previous`)  
- Target imbalance (~11% positive class)

### **Bivariate & Multivariate Analysis**
- Subscription rate by job, education, month, contact type  
- Numerical features vs. target (boxplots)  
- Macro indicators vs. subscription patterns  
- Pearson correlation & Cramér’s V heatmaps  
- Several pairplots (customer behavior, campaign dynamics, economics)

### **Key Insights**
- Subscribers are more common during **economic downturns** (low Euribor, negative employment variation).  
- **Contact type**, **month**, and **poutcome** are highly predictive.  
- Repeated contacts reduce success likelihood (diminishing returns).  
- Age groups show different subscription probabilities.


# 🧹 2. Data Cleaning & Preprocessing

Performed in *Notebook 02*:

✔ Normalized column names  
✔ Removed duplicates  
✔ Encoded `pdays` as binary (contacted before / not contacted)  
✔ Mapped `month` and `day_of_week` to numeric  
✔ Removed leakage column `duration`  
✔ Saved clean base dataset to `02-silver/`  

Full details available in `02-Data Cleaning and Preprocessing.ipynb`.


# 🤖 3. Modeling Workflow

We developed and compared several models:

### 3.1 Baseline Logistic Regression  
*Notebook 03*  
- One-hot encoding only  
- No scaling, no PCA, no feature engineering  
- SHAP for interpretability  

See full analysis:  
📄 [Analysis Baseline](Analysis_baseline.MD)


### 3.2 Engineered Logistic Regression  
*Notebook 04*  
- PCA on macro indicators  
- WOE encoding for categorical variables  
- Numeric & categorical feature selection  
- SMOTEENN oversampling  
- Threshold optimization for F1-score  
- Full SHAP interpretation


### 3.3 XGBoost (Final Model)
*Notebook 05*  
- Most performant model  
- Captures nonlinear interactions  
- Feature engineering + PCA + WOE + SMOTE  
- Hyperparameter tuning (GridSearchCV)  
- Threshold tuning (maximize F1-score)

# 🏆 4. Final Model Selection & Packaging

*Notebook 06*

From all models in the **gold** directory:

- Automatically evaluated stored F1-scores  
- Selected best-performing XGBoost model  
- Rebuilt training pipeline with selected hyperparameters  
- Exported `model.pkl` for Hugging Face deployment  

This ensures the final model is:
- Reproducible  
- Version-controlled  
- Ready for API or UI deployment  

# 🚀 5. Hugging Face Deployment

The interactive app is located in: 96-huggingface_space/

It contains:

- **app.py** → Gradio UI  
- **model.pkl** → final model in ModelWrapper format  
- **README.md** → Space documentation  

Setup & deployment instructions:  
📄 [How to Setup GIT and Hugging Face Spaces](How_to_setup_git_hugging.MD)

# 📈 6. Final Model Evaluation

### XGBoost Performance (optimized threshold)

| Metric       | Value |
|--------------|--------|
| Accuracy     | ~0.87 |
| Precision    | ~0.45 |
| Recall       | ~0.57 |
| F1-score     | ~0.51 |
| ROC-AUC      | ~0.80 |

These metrics indicate a **balanced recall-focused model**, suitable when false negatives (missed potential subscribers) are costlier than false positives.

# 🏫 7. Explainability (SHAP)

SHAP analysis shows:

### Top Predictive Features
- Outcome of previous contact (`poutcome`)  
- Contact communication type  
- Economic factors (Euribor, employment variation)  
- Job × Age interactions  
- Customer's previous campaign engagement  

Visualization assets stored in:  
`97-model_metrics/`

# 🪞 8. Reproducibility

### Run locally:

```bash
pip install -r requirements.txt
python 96-huggingface_space/app.py
```